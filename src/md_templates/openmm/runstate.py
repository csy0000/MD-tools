"""Durable run state: continuity contracts, crash-safe restarts, and lifetime accounting.

Three things live here, shared by conventional MD and REST2 so neither grows its own copy.

**The continuity contract.** A resume must not silently continue a run under a different
Hamiltonian. Before any state is loaded, the settings that define the calculation are compared with
the ones the run recorded, and a mismatch is refused with the differing fields named. Runtime
extension fields -- how many chunks THIS invocation adds -- are deliberately excluded: asking for
more work is not a change to the calculation.

**Crash-safe restarts.** A restart is written as a *generation*: a directory of files that is
complete before anything points at it. Two independently replaced files are not an atomic pair, so
the commit is a single atomic replacement of one small record that names the generation. A crash
part-way through writing generation N leaves generation N-1 committed and intact, and recovery
ignores the uncommitted tail rather than half-adopting it.

**Lifetime accounting.** Statistics describe the whole run directory, not the invocation that
happened to finish it. Counters are reconstructed from durable history when a run is opened, and an
attempt is only counted once the generation containing its resulting state is committed.

Format versioning: every file written here carries `schema_version`, and readers validate it. A
directory written by an older layout is reported with what to do about it, never silently
reinterpreted.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

#: Bumped when the on-disk meaning of these files changes. Readers refuse a version they do not know.
RUN_STATE_SCHEMA = 1

RUN_STATE_FILE = "run_state.json"
RESTART_DIR = "restart"
COMMITTED_FILE = "committed.json"


class RunStateError(RuntimeError):
    """The run directory's durable state is missing, unreadable, or from another format."""


class IncompatibleContinuation(RuntimeError):
    """The requested continuation describes a different calculation than the run recorded."""


# ---------------------------------------------------------------------------------------------
# atomic filesystem primitives
# ---------------------------------------------------------------------------------------------

def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write via a temporary file in the SAME directory, then `os.replace`.

    Same directory because `os.replace` is only atomic within a filesystem. The temporary is
    flushed and fsynced before the rename, so the replacement cannot expose a partially written
    file even if the machine loses power between the two operations.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with open(os.devnull, "w"):
            pass
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: dict) -> Path:
    return atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=False) + "\n")
                              .encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# the continuity contract
# ---------------------------------------------------------------------------------------------

#: Settings whose change makes a continuation a DIFFERENT calculation. Compared before any state is
#: loaded, so an incompatible resume cannot append a single frame.
CONTINUITY_PATHS: tuple[str, ...] = (
    # identity of the thing being simulated
    "system.slug",
    "forcefield.protein",
    "forcefield.water",
    "forcefield.ligand",
    "forcefield.ligand_charge_method",
    "solvation.water_model",
    "solvation.box_shape",
    "solvation.padding_nm",
    "solvation.positive_ion",
    "solvation.negative_ion",
    "solvation.neutralize",
    # the Hamiltonian
    "system_build.nonbonded_method",
    "system_build.nonbonded_cutoff_nm",
    "system_build.switch_distance_nm",
    "system_build.use_dispersion_correction",
    "system_build.ewald_error_tolerance",
    "system_build.constraints",
    "system_build.rigid_water",
    "system_build.hydrogen_mass_amu",
    "system_build.hmr_scope",
    "system_build.remove_cm_motion",
    "system_build.minimum_image_margin_nm",
    "rest2.omega_exclusion",
    "rest2.proline_like_residues",
    "rest2.max_proline_ring_size",
    # the dynamics
    "integrator.kind",
    "integrator.timestep_fs",
    "integrator.temperature_k",
    "integrator.friction_per_ps",
    "production.ensemble",
    "production.precision",
    # the plan's granularity -- chunk LENGTH is continuity-defining, chunk COUNT is not
    "production.md.chunk_ns",
    "production.md.scale_factor",
    "production.remd.chunk_ns",
    "production.remd.exchange_interval_ps",
    "production.remd.scale_factors",
)

#: Explicitly NOT compared. Asking for more chunks is the supported way to extend a run; treating
#: it as an incompatibility would make extension impossible, which is the whole point of resuming.
EXTENSION_PATHS: tuple[str, ...] = (
    "production.md.n_chunks",
    "production.remd.n_chunks",
)


def _get(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def continuity_contract(cfg: dict, *, method: str, fingerprint_value: Optional[str] = None,
                        n_particles: Optional[int] = None,
                        bundle_config_hash: Optional[str] = None) -> dict[str, Any]:
    """The projection a continuation must reproduce, as a flat mapping.

    `method` is included because conventional MD and REST2 are different runs even when every
    other setting agrees -- resuming one as the other would append incompatible history.
    """
    contract: dict[str, Any] = {"method": str(method)}
    if fingerprint_value is not None:
        contract["prepared_system_fingerprint"] = fingerprint_value
    if n_particles is not None:
        contract["n_particles"] = int(n_particles)
    if bundle_config_hash is not None:
        contract["bundle_config_hash"] = bundle_config_hash
    for path in CONTINUITY_PATHS:
        value = _get(cfg, path)
        if isinstance(value, (list, tuple)):
            value = list(value)
        contract[path] = value
    return contract


def compare_continuity(stored: dict, current: dict) -> list[tuple[str, Any, Any]]:
    """Differing fields as `(field, recorded, requested)`, ignoring extension fields."""
    diffs: list[tuple[str, Any, Any]] = []
    for key in sorted(set(stored) | set(current)):
        if key in EXTENSION_PATHS:
            continue
        was, now = stored.get(key), current.get(key)
        if isinstance(was, float) and isinstance(now, float):
            if abs(was - now) > 1e-12:
                diffs.append((key, was, now))
        elif was != now:
            diffs.append((key, was, now))
    return diffs


def assert_continuable(run_dir: Path, current: dict) -> dict:
    """Refuse an incompatible continuation BEFORE any output file is opened.

    Returns the run state on success. The error names every differing field, because "incompatible"
    without the list leaves the user guessing which of thirty settings moved.
    """
    state = read_run_state(run_dir)
    stored = state.get("continuity") or {}
    if not stored:
        raise IncompatibleContinuation(
            f"{run_dir} has no recorded continuity contract, so it cannot be verified as the same "
            "calculation. It was produced by an earlier layout; start a fresh run."
        )
    diffs = compare_continuity(stored, current)
    if diffs:
        lines = "\n".join(f"    {field}: recorded {was!r}, requested {now!r}"
                          for field, was, now in diffs)
        raise IncompatibleContinuation(
            f"cannot resume {run_dir}: the requested configuration describes a different "
            f"calculation.\n{lines}\n"
            "  Continuation extends a run; it does not re-specify it. Start a fresh run instead."
        )
    return state


# ---------------------------------------------------------------------------------------------
# the run-state record
# ---------------------------------------------------------------------------------------------

def read_run_state(run_dir: Path) -> dict:
    path = Path(run_dir) / RUN_STATE_FILE
    if not path.is_file():
        raise RunStateError(
            f"{path} does not exist. A resume needs the run state written by a previous "
            "invocation; a directory without one cannot be verified or continued."
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunStateError(f"{path} is not readable JSON: {exc}") from None
    version = state.get("schema_version")
    if version != RUN_STATE_SCHEMA:
        raise RunStateError(
            f"{path} has schema_version {version!r}; this build writes {RUN_STATE_SCHEMA}. "
            "The run-state format changed and this directory cannot be continued safely. "
            "Start a fresh run, or keep using the build that wrote it."
        )
    return state


def write_run_state(run_dir: Path, *, method: str, continuity: dict, **extra: Any) -> Path:
    payload = {
        "schema_version": RUN_STATE_SCHEMA,
        "kind": "md-templates-run-state",
        "method": str(method),
        "continuity": continuity,
        **extra,
    }
    return atomic_write_json(Path(run_dir) / RUN_STATE_FILE, payload)


def update_run_state(run_dir: Path, **fields: Any) -> dict:
    """Read-modify-write the run state atomically, preserving unknown keys."""
    state = read_run_state(run_dir)
    state.update(fields)
    atomic_write_json(Path(run_dir) / RUN_STATE_FILE, state)
    return state


def record_invocation(run_dir: Path, entry: dict) -> dict:
    """Append one invocation to the run's append-only history."""
    state = read_run_state(run_dir)
    history = list(state.get("invocations") or [])
    history.append(entry)
    state["invocations"] = history
    atomic_write_json(Path(run_dir) / RUN_STATE_FILE, state)
    return state


# ---------------------------------------------------------------------------------------------
# restart generations
# ---------------------------------------------------------------------------------------------

def generation_dir(run_dir: Path, generation: int) -> Path:
    return Path(run_dir) / RESTART_DIR / f"gen_{int(generation):04d}"


def committed_generation(run_dir: Path) -> Optional[int]:
    """The generation the commit record points at, or None if nothing is committed.

    Only this record decides what is committed. The presence of files under a generation directory
    means nothing on its own: a crash mid-write leaves exactly that.
    """
    path = Path(run_dir) / RESTART_DIR / COMMITTED_FILE
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if rec.get("schema_version") != RUN_STATE_SCHEMA:
        raise RunStateError(
            f"{path} has schema_version {rec.get('schema_version')!r}; this build writes "
            f"{RUN_STATE_SCHEMA}."
        )
    gen = rec.get("generation")
    return int(gen) if isinstance(gen, int) else None


def commit_generation(run_dir: Path, generation: int, *, members: list[str],
                      attempts_committed: Optional[int] = None, **extra: Any) -> Path:
    """Point the run at a generation that is ALREADY fully written.

    Call only after every member file exists and is closed. The commit is one atomic replacement,
    which is what makes "committed" a single fact rather than an inference over several files.
    """
    gdir = generation_dir(run_dir, generation)
    missing = [m for m in members if not (gdir / m).is_file()]
    if missing:
        raise RunStateError(
            f"refusing to commit generation {generation}: {missing} missing from {gdir}. "
            "A commit record must point only at a complete restart."
        )
    record = {
        "schema_version": RUN_STATE_SCHEMA,
        "generation": int(generation),
        "members": list(members),
        "attempts_committed": attempts_committed,
        **extra,
    }
    path = Path(run_dir) / RESTART_DIR / COMMITTED_FILE
    atomic_write_json(path, record)
    _prune_generations(run_dir, keep_from=int(generation) - 1)
    return path


def committed_record(run_dir: Path) -> dict:
    path = Path(run_dir) / RESTART_DIR / COMMITTED_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _prune_generations(run_dir: Path, *, keep_from: int) -> None:
    """Drop generations older than the previous committed one.

    The previous generation is retained deliberately: if the newest committed restart turns out to
    be unloadable, there is still one good state to fall back to rather than nothing.
    """
    root = Path(run_dir) / RESTART_DIR
    if not root.is_dir():
        return
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("gen_"):
            continue
        try:
            gen = int(child.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if gen < keep_from:
            for f in child.iterdir():
                f.unlink(missing_ok=True)
            child.rmdir()


def restart_members(replica: Optional[int] = None) -> tuple[str, str]:
    """`(checkpoint_name, state_name)` for a replica, or for a single walker when None."""
    stem = "walker" if replica is None else f"replica_{int(replica):02d}"
    return f"{stem}.chk", f"{stem}.state.xml"


def save_restart(sim, gdir: Path, *, replica: Optional[int] = None) -> tuple[Path, Path]:
    """Write BOTH restart forms for one simulation into a generation directory.

    The checkpoint is the exact same-platform continuation; the serialized State is the portable
    fallback and carries positions, velocities, box vectors, time and parameters. Both are written
    through a temporary file and renamed, so a crash cannot leave a truncated file under a name the
    commit record will later point at.
    """
    from openmm import XmlSerializer

    gdir = Path(gdir)
    gdir.mkdir(parents=True, exist_ok=True)
    chk_name, state_name = restart_members(replica)

    fd, tmp_chk = tempfile.mkstemp(dir=str(gdir), prefix=f".{chk_name}.", suffix=".tmp")
    os.close(fd)
    sim.saveCheckpoint(tmp_chk)
    os.replace(tmp_chk, gdir / chk_name)

    state = sim.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                 enforcePeriodicBox=False)
    atomic_write_bytes(gdir / state_name, XmlSerializer.serialize(state).encode("utf-8"))
    return gdir / chk_name, gdir / state_name


def load_restart(sim, gdir: Path, *, replica: Optional[int] = None,
                 allow_state_fallback: bool = True) -> str:
    """Restore one simulation from a committed generation. Returns "checkpoint" or "state".

    The binary checkpoint is preferred because it continues bit-for-bit on the same platform. If it
    is missing, corrupt, or was written by another platform, the portable State is used instead --
    that preserves positions, velocities, box, time and parameters, so the continuation is
    physically valid, but the stochastic integrator's internal stream is NOT restored and the
    trajectory diverges from the one an uninterrupted run would have produced. Reported, never
    silent.
    """
    from openmm import XmlSerializer

    gdir = Path(gdir)
    chk_name, state_name = restart_members(replica)
    chk, state_path = gdir / chk_name, gdir / state_name

    if chk.is_file():
        try:
            sim.loadCheckpoint(str(chk))
            return "checkpoint"
        except Exception as exc:                                     # noqa: BLE001
            if not allow_state_fallback:
                raise
            print(f"[restart] checkpoint {chk} unusable ({type(exc).__name__}: {exc}); "
                  "falling back to the serialized State", flush=True)

    if not allow_state_fallback or not state_path.is_file():
        raise RunStateError(
            f"no usable restart in {gdir}: checkpoint "
            f"{'unusable' if chk.is_file() else 'missing'} and "
            f"{'State missing' if not state_path.is_file() else 'State fallback disabled'}. "
            "Refusing to append output to a run that cannot be continued."
        )
    sim.context.setState(XmlSerializer.deserialize(state_path.read_text(encoding="utf-8")))
    return "state"
