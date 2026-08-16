"""Immutable REST2 run directories, and a launch path that does not know which machine it is on.

The launcher this replaces hardcoded `CUDA_VISIBLE_DEVICES=0/1/2`, assumed three GPUs, backgrounded
with `nohup`, and wrote logs to a repository-relative `logs/`. None of that survives being run from
another repository, and two of them are silently wrong rather than loudly wrong: on a machine with
one GPU the second and third processes land back on device 0, where two REST2 processes sharing a
card without CUDA MPS run about 3.7x slower each.

So: one process controls its own configured replicas on one selected device, scheduling is the
consuming repository's problem, and every artifact goes inside the run directory.

A run directory is either created fresh or continued in place -- never overwritten, and never
silently reused.

    --run-name NAME    exactly RUN_ROOT/NAME, with nothing appended
    (omitted)          RUN_ROOT/YYYYMMDDTHHMMSSZ__SYSTEM_ID__METHOD__CONFIG_HASH/
    --resume-run DIR   that same directory, extended by this invocation

The config hash derives from the canonical serialisation of the two manifests, so the same
calculation gets the same hash on any machine, and any change to a seed or a scientific setting
changes it. The timestamp keeps repeated launches distinct; a user-supplied name is used verbatim,
because a name that arrives decorated is not the name that was asked for.

A fresh run refuses an existing directory and says to resume it explicitly. A resume validates the
continuity contract in `run_state.json` BEFORE loading any state, and refuses with the differing
fields named if the configuration describes a different calculation. `n_chunks` is what THIS
invocation adds, so resuming extends the run rather than finding it already finished.

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

from . import provenance, runstate
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
    """The target run directory already exists, or a resume target does not."""


class IncompatibleExperiment(RuntimeError):
    """An overriding experiment describes a different System than the bundle contains."""


def run_dir_name(system_id: str, chash: str, *, method: str = "rest2",
                 stamp: Optional[str] = None) -> str:
    """The DEFAULT name for a fresh unnamed run: timestamped, and identifying enough to sort.

    A user-supplied `--run-name` is used verbatim instead -- no timestamp, system or hash is
    appended to it. A name the user chose that then arrives decorated is not the name they chose,
    and it makes the directory unpredictable to anything scripted around it.
    """
    return f"{stamp or provenance.run_stamp()}__{system_id}__{method}__{chash}"


def resolve_run_dir(out_root: Path, system_id: str, chash: str, *, method: str = "rest2",
                    run_name: Optional[str] = None, resume_run: Optional[Path] = None,
                    stamp: Optional[str] = None) -> tuple[Path, bool]:
    """Return `(run_dir, is_resume)` for the fresh-run / resume contract.

    Exactly one of `run_name` and `resume_run` may be given. A fresh run refuses to write into an
    existing directory; a resume writes into the directory it was given and never creates a
    sibling, because a resume that silently starts a new run produces two partial trajectories and
    no error.
    """
    if run_name is not None and resume_run is not None:
        raise ValueError(
            "--run-name and --resume-run are mutually exclusive: the first names a NEW run, the "
            "second continues an existing one. Pick which of the two this is."
        )

    if resume_run is not None:
        path = Path(resume_run).resolve()
        if not path.is_dir():
            raise RunExists(
                f"--resume-run {path} does not exist. A resume continues an existing run in place; "
                "it does not create one. Start a fresh run instead."
            )
        return path, True

    root = Path(out_root).resolve()
    if run_name is not None:
        name = str(run_name).strip()
        if not name or "/" in name or name in (".", ".."):
            raise ValueError(f"--run-name {run_name!r} is not a usable directory name")
        path = root / name
    else:
        path = root / run_dir_name(system_id, chash, method=method, stamp=stamp)

    if path.exists():
        raise RunExists(
            f"run directory already exists: {path}\n"
            "A fresh run will not overwrite one. Choose another --run-name or --out-root, or pass "
            "--resume-run to continue this run in place."
        )
    path.mkdir(parents=True)
    return path, False


def create_run_dir(out_root: Path, system_id: str, chash: str, *,
                   stamp: Optional[str] = None) -> Path:
    """Create a fresh, default-named run directory, refusing to reuse one."""
    path, _ = resolve_run_dir(out_root, system_id, chash, stamp=stamp)
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


def _stage_run_directory(bundle_dir: Path, exp_path: Path, cfg: dict, manifest: dict,
                         system, experiment, out_root: Path, *, method: str, chash: str,
                         run_name: Optional[str], resume_run: Optional[Path],
                         platform: str, device: Optional[str]) -> tuple[Path, bool]:
    """Create or reopen the run directory, enforcing the continuity contract on a resume.

    Returns `(run_dir, is_resume)`. On a resume, the contract is checked BEFORE any file is opened
    or copied, so an incompatible continuation cannot append a single byte to an existing run.
    """
    contract = runstate.continuity_contract(
        cfg, method=method, fingerprint_value=fingerprint(cfg),
        n_particles=(manifest.get("composition") or {}).get("n_atoms"),
        bundle_config_hash=manifest.get("config_hash"),
    )
    run_dir, is_resume = resolve_run_dir(
        out_root, system.system_id, chash, method=method,
        run_name=run_name, resume_run=resume_run,
    )
    if is_resume:
        runstate.assert_continuable(run_dir, contract)     # refuses before anything is written
        runstate.record_invocation(run_dir, {
            "started_utc": provenance.utc_timestamp(),
            "invocation": provenance.invocation(),
            "adds_chunks": cfg["production"]["remd" if method == "rest2" else "md"]["n_chunks"],
            "platform": platform, "device": device,
        })
        return run_dir, True

    # fresh run: copy the inputs in before anything that can fail, so even an immediate crash
    # leaves a directory that says what was attempted
    (run_dir / "system.yaml").write_text(
        (bundle_dir / "system.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "experiment.yaml").write_text(exp_path.read_text(encoding="utf-8"), encoding="utf-8")
    provenance.write_json(run_dir / "resolved_config.json", cfg)
    (run_dir / BUNDLE_MANIFEST).write_text(
        (bundle_dir / BUNDLE_MANIFEST).read_text(encoding="utf-8"), encoding="utf-8")

    staged = run_dir / "inputs"
    staged.mkdir(exist_ok=True)
    for src, dst in (("system.xml", "bundle_system.xml"),
                     ("topology.pdb", "bundle_topology.pdb"),
                     ("simbox.json", "bundle_simbox.json")):
        shutil.copy2(bundle_dir / src, staged / dst)
    shutil.copy2(bundle_dir / "equilibrated_state.xml", staged / "equilibrated_state.xml")

    runstate.write_run_state(
        run_dir, method=method, continuity=contract,
        run_id=run_dir.name, config_hash=chash,
        created_utc=provenance.utc_timestamp(),
        invocations=[{
            "started_utc": provenance.utc_timestamp(),
            "invocation": provenance.invocation(),
            "adds_chunks": cfg["production"]["remd" if method == "rest2" else "md"]["n_chunks"],
            "platform": platform, "device": device,
        }],
    )
    return run_dir, False


def launch_rest2(
    bundle_dir: Path,
    experiment_path: Optional[Path],
    out_root: Path,
    *,
    platform: str,
    device: Optional[str] = None,
    omega_exclusion: Optional[bool] = None,
    run_name: Optional[str] = None,
    resume_run: Optional[Path] = None,
) -> tuple[int, Optional[Path]]:
    """Run REST2 from a prepared bundle, into a new run directory or an existing one.

    Returns `(exit_code, run_dir)`. The run directory exists whenever one could be created, so a
    failure is still inspectable — `status.json` and `stderr.log` are inside it.
    """
    from .rest2 import run_rest2_remd

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
    cfg = resolve_config(system, experiment, platform=platform, device=device,
                         omega_exclusion=omega_exclusion)

    # BEFORE the run directory exists and before any OpenMM context: an overriding experiment may
    # change how long and where the run goes, never what System it runs. Checked whenever an
    # override was supplied; the bundle's own experiment is compatible by construction.
    if experiment_path is not None:
        reason = check_compatible(manifest, cfg)
        if reason:
            raise IncompatibleExperiment(reason)

    chash = config_hash(system.doc, experiment.doc)
    run_dir, is_resume = _stage_run_directory(
        bundle_dir, exp_path, cfg, manifest, system, experiment, out_root,
        method="rest2", chash=chash, run_name=run_name, resume_run=resume_run,
        platform=platform, device=device,
    )
    # `_load_bundle` addresses a System by the `<stem>_system.xml` convention and reads
    # `<stem>_topology.pdb` and `<stem>_simbox.json` beside it, so the bundle is staged under that
    # convention inside the run directory. Copies, not symlinks: the run directory has to stay
    # meaningful after the bundle is deleted or moved.
    staged = run_dir / "inputs"

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
            "n_chunks": cfg["production"]["remd"]["n_chunks"],
            "chunk_ns": cfg["production"]["remd"]["chunk_ns"],
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
            cfg, staged / "bundle_system.xml", staged / "equilibrated_state.xml", run_dir, ""
        )
        _flatten_outputs(run_dir)          # no-op for runs written directly; kept for older trees
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


def launch_md(
    bundle_dir: Path,
    experiment_path: Optional[Path],
    out_root: Path,
    *,
    platform: str,
    device: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_run: Optional[Path] = None,
) -> tuple[int, Optional[Path]]:
    """Run conventional explicit-water MD from a prepared bundle.

    One walker at `production.md.scale_factor`; no replicas, no ladder, no exchange machinery is
    constructed. It shares the run directory, provenance, continuity, restart and reporting
    implementation with REST2 -- only the propagation differs, which is the point of keeping the
    two algorithms in separate modules but one launcher.
    """
    from .md import run_md

    bundle_dir = Path(bundle_dir).resolve()
    manifest = validate_bundle(bundle_dir)
    exp_path = Path(experiment_path).resolve() if experiment_path else (
        bundle_dir / "experiment.prepare.yaml"
    )
    system = load_system(bundle_dir / "system.yaml")
    experiment = load_experiment(exp_path)

    configure_device(platform, device)
    cfg = resolve_config(system, experiment, platform=platform, device=device)
    if experiment_path is not None:
        reason = check_compatible(manifest, cfg)
        if reason:
            raise IncompatibleExperiment(reason)

    if (cfg.get("_declared") or {}).get("production.md") != "experiment":
        raise IncompatibleExperiment(
            f"{exp_path} declares no `md:` block, so it does not say how long a conventional-MD "
            "run should be. A REST2 plan describes a ladder, not a single walker, and falling back "
            "to the package default would silently launch "
            f"{cfg['production']['md']['n_chunks']} x {cfg['production']['md']['chunk_ns']} ns. "
            "Add an `md:` block with n_chunks and chunk_ns."
        )

    chash = config_hash(system.doc, experiment.doc)
    run_dir, is_resume = _stage_run_directory(
        bundle_dir, exp_path, cfg, manifest, system, experiment, out_root,
        method="md", chash=chash, run_name=run_name, resume_run=resume_run,
        platform=platform, device=device,
    )
    staged = run_dir / "inputs"

    mcfg = cfg["production"]["md"]
    planned_chunks = int(mcfg["n_chunks"])
    provenance.write_json(run_dir / "run_manifest.json", {
        "schema_version": 1,
        "kind": "explicit-solvent-md-run",
        "run_id": run_dir.name,
        "config_hash": chash,
        "prepared_system_fingerprint": fingerprint(cfg),
        "started_utc": provenance.utc_timestamp(),
        "invocation": provenance.invocation(),
        "method": "md",
        "system": manifest["system"],
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "master_seed": experiment.master_seed,
            "n_chunks": planned_chunks,
            "chunk_ns": mcfg["chunk_ns"],
            "scale_factor": mcfg["scale_factor"],
        },
        "platform": {"platform": platform, "device": device,
                     "precision": cfg["production"]["precision"]},
        "environment": provenance.environment_block(),
    })
    write_status(run_dir, STATUS_RUNNING, run_id=run_dir.name, method="md",
                 planned_chunks=planned_chunks)

    out_tee = _Tee(run_dir / "stdout.log", sys.stdout)
    err_tee = _Tee(run_dir / "stderr.log", sys.stderr)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_tee, err_tee
    code = EXIT_OK
    try:
        result = run_md(cfg, staged / "bundle_system.xml", staged / "equilibrated_state.xml",
                        run_dir, "")
        complete = str(result.get("status", "")) != "interrupted"
        write_status(run_dir, STATUS_COMPLETED if complete else STATUS_INTERRUPTED,
                     run_id=run_dir.name, method="md", planned_chunks=planned_chunks,
                     driver_status=result.get("status"),
                     lifetime_chunks=result.get("lifetime_chunks_completed"))
        if not complete:
            code = EXIT_RUNTIME
    except KeyboardInterrupt:
        write_status(run_dir, STATUS_INTERRUPTED, run_id=run_dir.name, method="md",
                     note="interrupted by signal; the budget was NOT reached")
        code = EXIT_INTERRUPTED
    except BaseException as exc:                   # noqa: BLE001 - status must always be written
        import traceback

        traceback.print_exc()
        write_status(run_dir, STATUS_FAILED, run_id=run_dir.name, method="md",
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
