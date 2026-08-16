"""Immutable REST2 run directories, and a launch path that does not know which machine it is on.

The launcher this replaces hardcoded `CUDA_VISIBLE_DEVICES=0/1/2`, assumed three GPUs, backgrounded
with `nohup`, and wrote logs to a repository-relative `logs/`. None of that survives being run from
another repository, and two of them are silently wrong rather than loudly wrong: on a machine with
one GPU the second and third processes land back on device 0, where two REST2 processes sharing a
card without CUDA MPS run about 3.7x slower each.

So: one process controls its own configured replicas on one selected device, scheduling is the
consuming repository's problem, and every artifact goes inside the run directory.

Run directories are immutable and named

    RUN_ROOT/YYYYMMDDTHHMMSSZ__SYSTEM_ID__rest2__CONFIG_HASH/

The config hash derives from the canonical serialisation of the two manifests, so the same
calculation gets the same hash on any machine, and any change to a seed or a scientific setting
changes it. The timestamp keeps repeated launches of the same configuration distinct, and the
directory is refused if it already exists.

`status.json` distinguishes running / completed / failed / interrupted. `completed` means the
requested budget was reached — a smoke run or an interrupted run never writes it, because
"finished" and "stopped" are different facts and only one of them licenses using the numbers.
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from . import provenance
from .bundle import BUNDLE_MANIFEST, validate_bundle
from .config import exchange_rounds, resolve_config
from .fingerprint import check_compatible, fingerprint
from .schemas import (
    ExperimentManifest,
    PLATFORMS,
    SystemManifest,
    config_hash,
    load_experiment,
    load_system,
)

#: Deterministic exit codes. A consuming scheduler needs to tell "this configuration is wrong"
#: from "this machine cannot run it" from "the simulation itself failed" without parsing text.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MANIFEST = 3
EXIT_ENVIRONMENT = 4
EXIT_BUNDLE = 5
EXIT_RUN_EXISTS = 6
EXIT_RUNTIME = 7
EXIT_INCOMPATIBLE = 8
EXIT_INTERRUPTED = 130

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


class RunExists(RuntimeError):
    """The target run directory already exists; runs are immutable."""


class IncompatibleExperiment(RuntimeError):
    """An overriding experiment describes a different System than the bundle contains."""


def run_dir_name(system_id: str, chash: str, *, stamp: Optional[str] = None) -> str:
    return f"{stamp or provenance.run_stamp()}__{system_id}__rest2__{chash}"


def create_run_dir(out_root: Path, system_id: str, chash: str, *,
                   stamp: Optional[str] = None) -> Path:
    """Create the immutable run directory, refusing to reuse one."""
    root = Path(out_root).resolve()
    path = root / run_dir_name(system_id, chash, stamp=stamp)
    if path.exists():
        raise RunExists(
            f"run directory already exists: {path}\n"
            "Runs are immutable. Launch again (the timestamp will differ) or choose another "
            "--out-root; do not write a second run into an existing directory."
        )
    path.mkdir(parents=True)
    return path


def write_status(run_dir: Path, status: str, **extra: Any) -> Path:
    payload = {"status": status, "updated_utc": provenance.utc_timestamp(), **extra}
    return provenance.write_json(Path(run_dir) / "status.json", payload)


def read_status(run_dir: Path) -> dict:
    path = Path(run_dir) / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def configure_device(platform: str, device: Optional[str]) -> dict[str, Optional[str]]:
    """Validate the platform/device pair and set the environment this process will use.

    `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set explicitly. Without it the CUDA runtime orders devices by
    a heuristic that can differ from `nvidia-smi`'s numbering, so "--device 1" and the device the
    user watched in nvidia-smi are not necessarily the same card.
    """
    if platform not in PLATFORMS:
        raise ValueError(f"--platform must be one of {PLATFORMS}, got {platform!r}")
    if device is not None and platform == "CPU":
        raise ValueError("--device is meaningless for --platform CPU; omit it")
    env: dict[str, Optional[str]] = {"CUDA_DEVICE_ORDER": None, "CUDA_VISIBLE_DEVICES": None}
    if platform == "CUDA":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if device is not None:
            # DeviceIndex is passed to OpenMM through the config; CUDA_VISIBLE_DEVICES is left
            # alone so the index in the manifest means what nvidia-smi shows.
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


class _Tee:
    """Write to a file and to the original stream, so a run is watchable and also self-contained."""

    def __init__(self, path: Path, mirror) -> None:
        self._fh = open(path, "w", encoding="utf-8", buffering=1)
        self._mirror = mirror

    def write(self, text: str) -> int:
        self._fh.write(text)
        with contextlib.suppress(Exception):
            self._mirror.write(text)
        return len(text)

    def flush(self) -> None:
        self._fh.flush()
        with contextlib.suppress(Exception):
            self._mirror.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


def launch_rest2(
    bundle_dir: Path,
    experiment_path: Optional[Path],
    out_root: Path,
    *,
    platform: str,
    device: Optional[str] = None,
) -> tuple[int, Optional[Path]]:
    """Run REST2 from a prepared bundle into a fresh immutable run directory.

    Returns `(exit_code, run_dir)`. The run directory exists whenever one could be created, so a
    failure is still inspectable — `status.json` and `stderr.log` are inside it.
    """
    from escort_ais.systems.explicit_baseline import run_rest2_remd

    bundle_dir = Path(bundle_dir).resolve()
    manifest = validate_bundle(bundle_dir)

    # The experiment may be overridden at launch (e.g. a longer budget against the same prepared
    # box); by default the run uses the one the bundle was prepared with.
    exp_path = Path(experiment_path).resolve() if experiment_path else (
        bundle_dir / "experiment.prepare.yaml"
    )
    system = load_system(bundle_dir / "system.yaml")
    experiment = load_experiment(exp_path)

    configure_device(platform, device)
    cfg = resolve_config(system, experiment, platform=platform, device=device)

    # BEFORE the run directory exists and before any OpenMM context: an overriding experiment may
    # change how long and where the run goes, never what System it runs. Checked whenever an
    # override was supplied; the bundle's own experiment is compatible by construction.
    if experiment_path is not None:
        reason = check_compatible(manifest, cfg)
        if reason:
            raise IncompatibleExperiment(reason)

    chash = config_hash(system.doc, experiment.doc)
    run_dir = create_run_dir(out_root, system.system_id, chash)

    # copy the inputs in before doing anything that can fail, so even an immediate crash leaves a
    # directory that says what was attempted
    (run_dir / "system.yaml").write_text(
        (bundle_dir / "system.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "experiment.yaml").write_text(
        exp_path.read_text(encoding="utf-8"), encoding="utf-8")
    provenance.write_json(run_dir / "resolved_config.json", cfg)
    (run_dir / BUNDLE_MANIFEST).write_text(
        (bundle_dir / BUNDLE_MANIFEST).read_text(encoding="utf-8"), encoding="utf-8")

    # `_load_bundle` addresses a System by the `<stem>_system.xml` convention and reads
    # `<stem>_topology.pdb` and `<stem>_simbox.json` beside it. The portable bundle uses bare
    # names, so stage a view of it under that convention inside the run directory. Copies, not
    # symlinks: the run directory has to stay meaningful after the bundle is deleted or moved.
    staged = run_dir / "inputs"
    staged.mkdir()
    for src, dst in (("system.xml", "bundle_system.xml"),
                     ("topology.pdb", "bundle_topology.pdb"),
                     ("simbox.json", "bundle_simbox.json")):
        shutil.copy2(bundle_dir / src, staged / dst)
    shutil.copy2(bundle_dir / "equilibrated_state.xml", staged / "equilibrated_state.xml")

    planned_rounds = exchange_rounds(cfg)
    run_manifest = {
        "schema_version": 1,
        "kind": "explicit-solvent-rest2-run",
        "run_id": run_dir.name,
        "config_hash": chash,
        "prepared_system_fingerprint": fingerprint(cfg),
        "started_utc": provenance.utc_timestamp(),
        "invocation": provenance.invocation(),
        "bundle": {
            "path_name": bundle_dir.name,
            "config_hash": manifest.get("config_hash"),
            "prepared_system_fingerprint": (manifest.get("prepared_system") or {}).get(
                "fingerprint"),
            "experiment_overridden": experiment_path is not None,
            "files": manifest.get("files", {}),
        },
        "system": manifest["system"],
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "ladder_status": experiment.ladder_status,
            "n_rungs": experiment.n_rungs,
            "scale_factors": experiment.scale_factors,
            "master_seed": experiment.master_seed,
            "total_ns_per_replica": cfg["production"]["remd"]["total_ns_per_replica"],
            "planned_exchange_rounds": planned_rounds,
        },
        "platform": {
            "platform": platform,
            "device": device,
            "precision": cfg["production"]["precision"],
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        },
        "environment": provenance.environment_block(),
    }
    provenance.write_json(run_dir / "run_manifest.json", run_manifest)
    write_status(run_dir, STATUS_RUNNING, run_id=run_dir.name,
                 planned_exchange_rounds=planned_rounds)

    out_tee = _Tee(run_dir / "stdout.log", sys.stdout)
    err_tee = _Tee(run_dir / "stderr.log", sys.stderr)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_tee, err_tee
    code = EXIT_OK
    try:
        result = run_rest2_remd(
            cfg, staged / "bundle_system.xml", staged / "equilibrated_state.xml", run_dir, "rest2"
        )
        _flatten_outputs(run_dir)
        observed = count_exchange_rounds(run_dir)
        complete = observed >= planned_rounds and str(
            result.get("status", "")) != "already-complete"
        write_status(
            run_dir,
            STATUS_COMPLETED if complete else STATUS_INTERRUPTED,
            run_id=run_dir.name,
            planned_exchange_rounds=planned_rounds,
            observed_exchange_rounds=observed,
            driver_status=result.get("status"),
            note=None if complete else (
                "the requested budget was not reached; this is not a completed production run"
            ),
        )
        if not complete:
            code = EXIT_RUNTIME
    except KeyboardInterrupt:
        write_status(run_dir, STATUS_INTERRUPTED, run_id=run_dir.name,
                     planned_exchange_rounds=planned_rounds,
                     observed_exchange_rounds=count_exchange_rounds(run_dir),
                     note="interrupted by signal; the budget was NOT reached")
        code = EXIT_INTERRUPTED
    except BaseException as exc:                   # noqa: BLE001 - status must always be written
        import traceback

        traceback.print_exc()
        write_status(run_dir, STATUS_FAILED, run_id=run_dir.name,
                     planned_exchange_rounds=planned_rounds,
                     observed_exchange_rounds=count_exchange_rounds(run_dir),
                     error=f"{type(exc).__name__}: {exc}")
        code = EXIT_RUNTIME
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        out_tee.close()
        err_tee.close()
    return code, run_dir


def _flatten_outputs(run_dir: Path) -> None:
    """Lift the driver's `rest2/` subtree to the documented run-directory layout.

    `run_rest2_remd` writes `<out_dir>/<suffix>/replica_XX/` and
    `<out_dir>/<suffix>_exchange_attempts.csv`. The portable layout puts `replica_*/` and
    `exchange_attempts.csv` at the top of the run directory, so a consumer does not need to know
    the suffix convention.
    """
    inner = run_dir / "rest2"
    if inner.is_dir():
        for child in sorted(inner.iterdir()):
            target = run_dir / child.name
            if not target.exists():
                child.rename(target)
        with contextlib.suppress(OSError):
            inner.rmdir()
    src = run_dir / "rest2_exchange_attempts.csv"
    dst = run_dir / "exchange_attempts.csv"
    if src.is_file() and not dst.exists():
        src.rename(dst)


def count_exchange_rounds(run_dir: Path) -> int:
    """Distinct exchange rounds actually recorded, read back from the log.

    Counted from the file rather than from the driver's return value: the log is what a consumer
    will read, so if the two ever disagree the status should reflect the artifact.
    """
    path = Path(run_dir) / "exchange_attempts.csv"
    if not path.is_file():
        path = Path(run_dir) / "rest2_exchange_attempts.csv"
    if not path.is_file():
        return 0
    steps: set[str] = set()
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = row.get("step") or row.get("time_ps")
                if key:
                    steps.add(str(key))
    except Exception:                              # noqa: BLE001
        return 0
    return len(steps)


def install_signal_handlers() -> None:
    """Turn SIGTERM into KeyboardInterrupt so a scheduler's kill writes `interrupted`.

    Without this a scheduled job that is preempted leaves `status: running` for ever, which later
    reads as "still going" rather than "stopped".
    """

    def _raise(signum, frame):                     # noqa: ANN001, ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, _raise)
