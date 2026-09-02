#!/usr/bin/env python
"""The authoritative replica-exchange output validator. ONE implementation, copied into projects.

A generated replica script calls this through `--verify-only`. It does not reimplement the schema and
it does not check file existence: a file that exists is exactly what a crashed run leaves behind.
Every check below OPENS the storage and reads it.
"""
import json
import math
from pathlib import Path

import numpy as np

from . import statistics
from . import storage


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
        # The block mixes filenames with description. `schema` names a format, `authoritative`
        # names another KEY of this block, and `coordinate_indexing` says how the stored
        # coordinates are indexed -- none of them is a file, and checking them as files failed
        # every extended run with "walker does not exist beside the manifest".
        if key in ("schema", "authoritative", "coordinate_indexing") or name is None:
            continue
        if Path(name).is_absolute() or Path(name).name != name:
            result.fail(
                f"manifest storage.{key} is {name!r}; it must be a bare filename beside the "
                f"manifest, never a path that can point outside the run directory")
        elif not (directory / name).is_file():
            result.fail(f"manifest storage.{key}={name} does not exist beside the manifest")


def validate_replica_output(*, analysis, checkpoint=None, manifest=None, expect_completed=True,
                            reconcilable=False):
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
                  expect_completed=expect_completed, reconcilable=reconcilable)
    finally:
        if reporter is not None:
            reporter.close()
    return result


def _validate(reporter, analysis_path, checkpoint, record, result, *, expect_completed,
              reconcilable=False):
    schema = reporter.schema
    result.note("schema", schema)
    if schema in storage.SUPERSEDED_SCHEMAS:
        # Identified, explained, and refused -- not silently reinterpreted as if it were current.
        result.fail(f"{schema} is superseded: {storage.SUPERSEDED_SCHEMAS[schema]}")
        return result
    if schema != storage.SCHEMA_VERSION:
        result.fail(f"storage schema is {schema!r}, not {storage.SCHEMA_VERSION!r}")

    identity = reporter.identity
    if not isinstance(identity, dict):
        result.fail("the storage carries no scientific identity, so what it was created with "
                    "cannot be established")
        identity = {}
    n_states = int(identity.get("n_states") or reporter.n_states())
    result.note("n_states", n_states)

    # An unfinished migration transaction is reported, never finished. Validation is read-only:
    # it must not create the field, backfill it, commit the event or clear the marker, because a
    # user who asked to inspect a file did not ask to change it.
    # The SAME strict validator the driver and the reconciliation use. An invalid record is a
    # reported failure here, never something quietly downgraded to "reconcilable".
    try:
        pending = reporter.validated_pending_migration()
    except Exception as failure:                            # noqa: BLE001 - reported, not raised
        pending = None
        result.fail(f"the pending-migration record is not usable "
                    f"({type(failure).__name__}: {failure})")
    if pending:
        result.note("pending_migration_fields", list(pending.get("fields") or []))
        result.note("pending_migration_transaction", pending.get("transaction_id"))
        result.note("pending_migration_created_utc", pending.get("created_utc"))
        if reconcilable:
            # A CONTINUATION is about to reconcile this, so it is a state to report, not a reason
            # to refuse. Failing here instead would be a deadlock: the only thing that can finish
            # the transaction is the resume that this check would block.
            result.note("pending_migration", "will be reconciled before propagation")
        else:
            result.fail(
                f"an incomplete migration transaction is recorded in this file: field(s) "
                f"{list(pending.get('fields') or [])} were being added when a previous process "
                f"stopped. Nothing here will finish it -- verification is read-only. Run "
                f"`--resume` or `--extend N`, which reconcile the transaction before any step is "
                f"propagated.")
    result.note("migration_events", len(reporter.migration_history()))

    try:
        last = reporter.last_exchange()
    except Exception as failure:
        return result.fail(
            f"the last committed exchange could not be read ({type(failure).__name__}: "
            f"{failure}); the analysis file is missing variables a run must have written")
    result.note("last_committed_exchange", last)
    result.note("whole_frames", reporter.last_frame() + 1)
    result.note("solute_frames", reporter.last_solute_frame() + 1)
    if last < 0:
        result.fail("no exchange was ever committed to this storage")
        return result

    # Each stream's completion marker must name a row the file actually holds. The marker is
    # written LAST precisely so that a crash mid-row leaves it BEHIND the data; a marker AHEAD of
    # the data cannot arise that way, so it means the file disagrees with itself and a resume
    # would read rows that were never written.
    for label, marker, available in (
            ("last_exchange", last, reporter.n_exchange_rows()),
            ("last_frame", reporter.last_frame(), reporter.n_whole_frames()),
            ("last_solute_frame", reporter.last_solute_frame(), reporter.n_solute_frames())):
        if marker >= available:
            result.fail(
                f"{label} is {marker} but the file holds only {available} row(s) of that "
                f"stream. The marker is committed after the data it describes, so it can lag "
                f"behind but never lead: this file is inconsistent with itself and must not be "
                f"resumed or extended.")

    # The three streams are independent, and each carries its own absolute step. Steps must be
    # strictly increasing within a stream: a repeated step means a restart duplicated a record.
    for name, steps in (("exchange", reporter.exchange_steps()),
                        ("whole frame", reporter.frame_steps()),
                        ("solute frame", reporter.solute_steps())):
        array = np.asarray(steps)
        if array.size and not np.all(np.diff(array) > 0):
            result.fail(
                f"{name} steps are not strictly increasing; a restart duplicated a record or "
                f"wrote them out of order")
    result.note("exchange_steps_last", int(reporter.exchange_steps()[-1])
                if reporter.exchange_steps().size else None)

    # Strictly increasing is not enough: the stored steps must be the SCHEDULED ones. A run that
    # dropped an attempt in the middle still has increasing steps, and its budget can still be
    # reached by the rows that remain, so nothing else here would notice the gap.
    scheduled = ((identity.get("schedule") or {}).get("exchange_steps")
                 if isinstance(identity.get("schedule"), dict) else None)
    steps = reporter.exchange_steps()
    if scheduled and steps.size:
        expected_steps = np.arange(1, steps.size + 1, dtype=np.int64) * int(scheduled)
        if not np.array_equal(np.asarray(steps, dtype=np.int64), expected_steps):
            missing = sorted(set(expected_steps.tolist()) - set(int(s) for s in steps))
            result.fail(
                f"the stored exchange steps are not the scheduled ones (every "
                f"{int(scheduled)} step(s)); first mismatch at row "
                f"{int(np.argmax(np.asarray(steps, dtype=np.int64) != expected_steps))}"
                + (f", missing scheduled step(s) {missing[:4]}" if missing else "")
                + ". An attempt that was scheduled and never recorded leaves a gap no budget "
                  "check can see.")

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
                f"the mapping holds {mapping.shape[0]} row(s) but the last committed exchange is "
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
    try:
        accepted, proposed = reporter.statistics()
        events = reporter.reservoir_events()
        stats = statistics.lifetime_statistics(
            accepted, proposed, tau=identity.get("tau") or list(range(n_states)),
            reservoir_events=events)
        result.note("exchanges_committed", stats["exchanges_committed"])
        # The identity records what was ORIGINALLY requested, and `--extend` legitimately runs
        # past it: a run extended twice holds more rows than the protocol ever asked for. So the
        # identity's count is a FLOOR, and the authoritative budget is the schedule the run
        # actually finished under -- the manifest's, or the sidecar's, whichever exists.
        expected = identity.get("number_of_exchanges")
        actual_budget = None
        if isinstance(record, dict):
            schedule = record.get("schedule")
            if isinstance(schedule, dict):
                actual_budget = schedule.get("number_of_exchanges")
        if actual_budget is not None:
            result.note("budget_exchanges", int(actual_budget))
            if expect_completed and stats["exchanges_committed"] != int(actual_budget):
                result.fail(
                    f"the storage holds {stats['exchanges_committed']} exchange row(s) but the "
                    f"run finished under a budget of {int(actual_budget)} attempts")
        elif expect_completed and expected is not None and stats["exchanges_committed"] < int(
                expected):
            result.fail(
                f"the storage holds {stats['exchanges_committed']} exchange row(s), fewer than "
                f"the {int(expected)} attempts the run was created to make")
        if stats["reservoir"]:
            result.note("reservoir_attempts", stats["reservoir"]["attempts"])
    except Exception as failure:
        result.fail(f"exchange statistics could not be read ({type(failure).__name__}: {failure})")

    # -- the coordinate streams ------------------------------------------------------------------
    if reporter.last_frame() < 0:
        result.fail("no whole-system frame was ever stored")
    solute_interval = identity.get("solute_output_interval_steps")
    if solute_interval and reporter.last_solute_frame() < 0:
        result.fail(
            "a solute output interval was configured but no solute frame was stored; the "
            "frequent solute stream is a promise the storage must keep")

    # -- the checkpoint --------------------------------------------------------------------------------
    if checkpoint is not None:
        path = Path(checkpoint)
        result.note("checkpoint_netcdf", str(path))
        if not path.is_file():
            result.fail(f"checkpoint NetCDF {path} does not exist")
        else:
            try:
                state = storage.ReplicaCheckpoint(path).read()
                result.note("checkpoint_step", state["step"])
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
                if state["exchange_index"] > last:
                    result.fail(
                        f"the checkpoint is at exchange {state['exchange_index']} but the analysis "
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
        budget = identity.get("total_steps")
        sidecar = storage.read_run_state(analysis_path)
        if isinstance(sidecar, dict) and sidecar.get("total_steps") is not None:
            budget = sidecar["total_steps"]
            result.note("budget_source", f"run-state sidecar ({sidecar.get('status')})")
        else:
            result.note("budget_source", "scientific identity (the original request)")
        steps = reporter.exchange_steps()
        reached = int(steps[-1]) if steps.size else 0
        if budget is not None and reached < int(budget):
            result.fail(
                f"the storage reached step {reached}, short of the {int(budget)} recorded. An "
                f"incomplete budget is not a completed run.")
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

    committed = record.get("exchanges_committed")
    if committed is not None and int(committed) != last + 1:
        result.fail(f"manifest records {int(committed)} committed exchange(s) but the storage "
                    f"holds {last + 1}")
    expected = record.get("steps_expected")
    completed = record.get("steps_completed")
    if expect_completed and expected is not None and completed is not None:
        if int(completed) < int(expected):
            result.fail(f"manifest promised {int(expected)} step(s) and recorded "
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
