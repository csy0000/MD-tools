#!/usr/bin/env python
"""The authoritative replica-exchange output validator. ONE implementation, copied into projects.

`MD-project` calls this through `openmm-md --verify-only`. It does not reimplement the schema and
it does not check file existence: a file that exists is exactly what a crashed run leaves behind.
Every check below OPENS the storage and reads it.
"""
import json
import math
from pathlib import Path

import numpy as np

import replica_statistics as statistics
import replica_storage as storage


class ValidationResult:
    def __init__(self):
        self.problems = []
        self.facts = {}

    @property
    def ok(self):
        return not self.problems

    def fail(self, message):
        self.problems.append(message)
        return self

    def note(self, key, value):
        self.facts[key] = value
        return self

    def as_dict(self):
        return {"ok": self.ok, "problems": list(self.problems), "facts": dict(self.facts)}


def _load_manifest(path, result):
    manifest = Path(path)
    if not manifest.is_file():
        result.fail(f"manifest {manifest} does not exist")
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as failure:
        result.fail(f"manifest {manifest} is not readable JSON: {failure}")
        return None


def _check_manifest_paths(manifest_path, record, result):
    """A manifest may only name files beside itself.

    One that pointed elsewhere could certify storage it does not live with, which is how a run
    comes to be validated against another run's output.
    """
    directory = Path(manifest_path).resolve().parent
    for key, name in (record.get("storage") or {}).items():
        if key in ("schema", "authoritative") or name is None:
            continue
        if Path(name).is_absolute() or Path(name).name != name:
            result.fail(
                f"manifest storage.{key} is {name!r}; it must be a bare filename beside the "
                f"manifest, never a path that can point outside the run directory")
        elif not (directory / name).is_file():
            result.fail(f"manifest storage.{key}={name} does not exist beside the manifest")


def validate_replica_output(*, analysis, checkpoint=None, manifest=None, expect_completed=True):
    """Open the storage and decide whether it is a complete, coherent replica-exchange run."""
    result = ValidationResult()
    analysis_path = Path(analysis)
    result.note("analysis_netcdf", str(analysis_path))
    if not analysis_path.is_file():
        return result.fail(f"analysis NetCDF {analysis_path} does not exist")

    record = None
    if manifest is not None:
        record = _load_manifest(manifest, result)
        if record is not None:
            _check_manifest_paths(manifest, record, result)

    reporter = None
    try:
        try:
            reporter = storage.ReplicaReporter(analysis_path, mode="r")
        except storage.StorageError as failure:
            return result.fail(str(failure))
        _validate(reporter, analysis_path, checkpoint, record, result,
                  expect_completed=expect_completed)
    finally:
        if reporter is not None:
            reporter.close()
    return result


def _validate(reporter, analysis_path, checkpoint, record, result, *, expect_completed):
    schema = reporter.schema
    result.note("schema", schema)
    if schema != storage.SCHEMA_VERSION:
        result.fail(f"storage schema is {schema!r}, not {storage.SCHEMA_VERSION!r}")

    identity = reporter.identity
    if not isinstance(identity, dict):
        result.fail("the storage carries no scientific identity, so what it was created with "
                    "cannot be established")
        identity = {}
    n_states = int(identity.get("n_states") or reporter.n_states())
    result.note("n_states", n_states)

    try:
        last = reporter.last_iteration()
    except Exception as failure:
        return result.fail(
            f"the last committed iteration could not be read ({type(failure).__name__}: "
            f"{failure}); the analysis file is missing variables a run must have written")
    result.note("last_committed_iteration", last)
    if last < 0:
        result.fail("no iteration was ever committed to this storage")
        return result

    # -- mapping ---------------------------------------------------------------------------------
    try:
        mapping = reporter.mapping()
    except Exception as failure:
        result.fail(f"the walker mapping could not be read ({type(failure).__name__}: {failure})")
        mapping = None
    if mapping is not None:
        result.note("mapping_shape", list(mapping.shape))
        if mapping.shape[0] != last + 1:
            result.fail(
                f"the mapping holds {mapping.shape[0]} row(s) but the last committed iteration is "
                f"{last}; the storage is truncated relative to its own counter")
        ok, offending = statistics.mapping_is_permutation_every_iteration(
            mapping, n_states=n_states)
        if not ok:
            result.fail(
                f"{len(offending)} mapping row(s) are not a permutation of the {n_states} states "
                f"(first: {offending[:8]}); a state was unoccupied or doubly occupied")
        # The walker view must invert the stored state view exactly.
        inverse = statistics.walker_view(mapping)
        rebuilt = statistics.walker_view(inverse)
        if not np.array_equal(rebuilt, mapping):
            result.fail("the walker view and the thermodynamic-state view do not invert each "
                        "other; the two cannot both be describing this run")

    # -- decision energies ---------------------------------------------------------------------------
    try:
        u, evaluated = reporter.reduced_potentials(last)
        result.note("reduced_potential_shape", list(u.shape))
        if u.shape != (n_states, n_states):
            result.fail(f"reduced potentials at iteration {last} have shape {u.shape}; expected "
                        f"({n_states}, {n_states})")
        chosen = np.asarray(evaluated, dtype=bool)
        if chosen.any() and not np.all(np.isfinite(u[chosen])):
            result.fail(f"reduced potentials at iteration {last} contain non-finite values where "
                        f"they were evaluated")
        result.note("reduced_potentials_evaluated", int(chosen.sum()))
    except Exception as failure:
        result.fail(f"reduced potentials could not be read ({type(failure).__name__}: {failure})")

    # -- exchange accounting ---------------------------------------------------------------------------
    stride = identity.get("exchange_stride_segments")
    try:
        accepted, proposed = reporter.statistics()
        events = reporter.reservoir_events()
        stats = statistics.lifetime_statistics(
            accepted, proposed, tau=identity.get("tau") or list(range(n_states)),
            exchange_stride=stride, reservoir_events=events)
        result.note("exchange_iterations", stats["exchange_iterations"])
        if stats["schedule"] is not None:
            result.note("expected_exchange_iterations",
                        stats["schedule"]["expected_exchange_iterations"])
            if not stats["schedule"]["agrees_with_schedule"]:
                result.fail(
                    f"the stored exchange history does not match the configured stride of "
                    f"{stride}: {len(stats['schedule']['scheduled_iterations_without_proposals'])}"
                    f" scheduled iteration(s) proposed nothing and "
                    f"{len(stats['schedule']['unscheduled_iterations_with_proposals'])} "
                    f"unscheduled one(s) did")
        if stats["reservoir"]:
            result.note("reservoir_attempts", stats["reservoir"]["attempts"])
    except Exception as failure:
        result.fail(f"exchange statistics could not be read ({type(failure).__name__}: {failure})")

    # -- stored frames -------------------------------------------------------------------------------
    try:
        result.note("stored_frames", reporter.frames())
        if reporter.frames() == 0:
            result.fail("no coordinate frame was ever stored")
    except Exception as failure:
        result.fail(f"stored frames could not be counted ({type(failure).__name__}: {failure})")

    # -- the checkpoint --------------------------------------------------------------------------------
    if checkpoint is not None:
        path = Path(checkpoint)
        result.note("checkpoint_netcdf", str(path))
        if not path.is_file():
            result.fail(f"checkpoint NetCDF {path} does not exist")
        else:
            try:
                state = storage.ReplicaCheckpoint(path).read()
                result.note("checkpoint_iteration", state["iteration"])
                result.note("checkpoint_walkers", len(state["configurations"]))
                if len(state["configurations"]) != n_states:
                    result.fail(
                        f"the checkpoint holds {len(state['configurations'])} walker(s) but the "
                        f"ladder has {n_states} states")
                for index, configuration in enumerate(state["configurations"]):
                    if not np.all(np.isfinite(configuration.positions)):
                        result.fail(f"walker {index} in the checkpoint has non-finite positions")
                    if not np.all(np.isfinite(configuration.velocities)):
                        result.fail(f"walker {index} in the checkpoint has non-finite velocities")
                if sorted(state["state_to_walker"]) != list(range(n_states)):
                    result.fail("the checkpoint's mapping is not a permutation")
                if state["iteration"] > last:
                    result.fail(
                        f"the checkpoint is at iteration {state['iteration']} but the analysis "
                        f"file committed only up to {last}; the two disagree about what ran")
            except storage.StorageError as failure:
                result.fail(str(failure))
            except Exception as failure:
                result.fail(f"the checkpoint could not be read ({type(failure).__name__}: "
                            f"{failure})")

    # -- the manifest, cross-checked ---------------------------------------------------------------------
    if isinstance(record, dict):
        _cross_check(record, identity, last, analysis_path, checkpoint, result,
                     expect_completed=expect_completed)
    elif expect_completed:
        # Without a manifest the budget still has to be met, and the storage knows it.
        budget = identity.get("total_segments")
        sidecar = storage.read_run_state(analysis_path)
        if isinstance(sidecar, dict) and sidecar.get("budget_segments") is not None:
            budget = sidecar["budget_segments"]
            result.note("budget_source", f"run-state sidecar ({sidecar.get('status')})")
        else:
            result.note("budget_source", "scientific identity (the original request)")
        if budget is not None and last + 1 < int(budget):
            result.fail(
                f"the storage committed {last + 1} segment(s), short of the {int(budget)} "
                f"recorded. An incomplete budget is not a completed run.")
    return result


def _cross_check(record, identity, last, analysis_path, checkpoint, result, *, expect_completed):
    if expect_completed and record.get("run_status") != "completed":
        result.fail(f"manifest records run_status={record.get('run_status')!r}; only 'completed' "
                    f"is a finished run")

    names = record.get("storage") or {}
    if names.get("analysis_netcdf") not in (None, analysis_path.name):
        result.fail(f"manifest names analysis_netcdf={names.get('analysis_netcdf')!r} but the "
                    f"storage validated is {analysis_path.name!r}")
    if checkpoint is not None and names.get("checkpoint_netcdf") not in (
            None, Path(checkpoint).name):
        result.fail(f"manifest names checkpoint_netcdf={names.get('checkpoint_netcdf')!r} but the "
                    f"checkpoint validated is {Path(checkpoint).name!r}")

    stated = record.get("scientific_identity")
    if isinstance(stated, dict) and isinstance(identity, dict) and identity:
        keys = (set(stated) | set(identity)) - {"format"}
        differences = [k for k in sorted(keys) if stated.get(k) != identity.get(k)]
        if differences:
            result.fail(
                f"the manifest's scientific identity disagrees with the storage's own in "
                f"{differences[:8]}. A manifest copied from a different run, or a changed "
                f"configuration, fails here.")

    committed = record.get("iterations_committed")
    if committed is not None and int(committed) != last + 1:
        result.fail(f"manifest records {int(committed)} committed iteration(s) but the storage "
                    f"holds {last + 1}")
    expected = record.get("segments_expected")
    completed = record.get("segments_completed")
    if expect_completed and expected is not None and completed is not None:
        if int(completed) < int(expected):
            result.fail(f"manifest promised {int(expected)} segment(s) and recorded "
                        f"{int(completed)}")
    value = record.get("production_ps_per_replica")
    if isinstance(value, float) and not math.isfinite(value):
        result.fail("manifest production_ps_per_replica is not finite")


def format_report(result, *, title="replica-exchange output validation"):
    lines = [f"# {title}"]
    for key, value in result.facts.items():
        lines.append(f"#   {key}: {value}")
    if result.ok:
        lines.append("# VALID: the storage is readable, coherent and complete.")
    else:
        lines.append(f"# INVALID: {len(result.problems)} problem(s)")
        lines.extend(f"#   - {problem}" for problem in result.problems)
    return "\n".join(lines)
