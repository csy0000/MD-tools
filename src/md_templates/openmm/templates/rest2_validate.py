#!/usr/bin/env python
"""The authoritative REST2 output validator. ONE implementation, copied into every project.

`MD-project` calls this through `openmm-rest2 --verify-only`. It does not reimplement the NetCDF
schema, does not import private OpenMMTools objects, and does not parse raw NetCDF variables --
because a second reader of a format is a second thing that can disagree with it.

WHAT "COMPLETE" MEANS HERE
    Not `Path.is_file()`. A file that exists proves that a path was created, which is exactly what
    a crashed run leaves behind. Every check below OPENS the storage and READS from it through the
    public `MultiStateReporter` API, and the run is complete only if the storage itself says so.

    Concretely this rejects: a truncated analysis or checkpoint file, a structurally valid NetCDF
    missing the variables a REST2 run must have written, a manifest copied from a different run, a
    missing final checkpoint, a stored iteration count short of the promised budget, a mapping row
    that is not a permutation, non-finite reduced potentials, and a scientific identity that
    disagrees between the manifest and the storage.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

#: OpenMMTools versions whose storage layout this validator understands.
SUPPORTED_OPENMMTOOLS = ("0.26.0",)


class ValidationResult:
    """Problems found, and the facts that were established while looking."""

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

    def __repr__(self):                                  # pragma: no cover - debugging aid
        return f"<ValidationResult ok={self.ok} problems={len(self.problems)}>"


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


def _read_sidecar(storage_path):
    """The run-state sidecar beside the analysis NetCDF, or None. Never raises."""
    candidate = Path(storage_path).with_name(Path(storage_path).stem + ".runstate.json")
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _check_manifest_paths(manifest_path, record, result):
    """A manifest may only name files beside itself.

    A manifest that pointed elsewhere could certify storage it does not live with -- which is how a
    run comes to be validated against another run's NetCDF.
    """
    directory = Path(manifest_path).resolve().parent
    storage = record.get("storage") or {}
    for key in ("analysis_netcdf", "checkpoint_netcdf", "run_state"):
        name = storage.get(key)
        if name is None:
            continue
        if os.path.isabs(name) or os.path.basename(name) != name:
            result.fail(
                f"manifest storage.{key} is {name!r}; it must be a bare filename beside the "
                f"manifest, never a path that can point outside the run directory")
            continue
        if not (directory / name).is_file():
            result.fail(f"manifest storage.{key}={name} does not exist beside the manifest")
    return storage


def validate_rest2_output(*, storage, checkpoint=None, manifest=None, expect_completed=True):
    """Open the REST2 storage and decide whether it is a complete, coherent run.

    `storage` is the analysis NetCDF (`-x`). `checkpoint` and `manifest` are optional; when a
    manifest is given it is cross-checked against the storage rather than trusted.
    """
    from openmmtools.multistate import MultiStateReporter

    result = ValidationResult()
    storage_path = Path(storage)
    result.note("analysis_netcdf", str(storage_path))

    if not storage_path.is_file():
        return result.fail(f"analysis NetCDF {storage_path} does not exist")

    record = None
    if manifest is not None:
        record = _load_manifest(manifest, result)
        if record is not None:
            _check_manifest_paths(manifest, record, result)

    reporter = None
    try:
        try:
            reporter = MultiStateReporter(str(storage_path), open_mode="r")
        except Exception as failure:
            # MultiStateReporter opens the analysis file AND its checkpoint companion, so a
            # failure here does not identify which of the two is at fault. Say so rather than
            # blaming the analysis file, which sent a reader looking at the wrong file.
            companion = storage_path.with_name(storage_path.stem + "_checkpoint.nc")
            return result.fail(
                f"the REST2 storage could not be opened through MultiStateReporter "
                f"({type(failure).__name__}: {failure}). MultiStateReporter opens both the "
                f"analysis file ({storage_path.name}) and its checkpoint companion "
                f"({companion.name}), so either may be truncated or corrupt.")

        _validate_open_storage(reporter, storage_path, checkpoint, record, result,
                               expect_completed=expect_completed)
    finally:
        if reporter is not None:
            try:
                reporter.close()
            except Exception:
                pass
    return result


def _validate_open_storage(reporter, storage_path, checkpoint, record, result, *,
                           expect_completed):
    # -- metadata and the identity it carries ---------------------------------------------------
    metadata = None
    try:
        metadata = reporter.read_dict("metadata")
    except Exception as failure:
        result.fail(f"reporter metadata could not be read ({type(failure).__name__}: {failure})")

    identity = None
    if isinstance(metadata, dict):
        identity = metadata.get("rest2_identity")
        versions = metadata.get("rest2_versions") or {}
        stored_version = versions.get("openmmtools")
        result.note("openmmtools_recorded", stored_version)
        if stored_version is not None and stored_version not in SUPPORTED_OPENMMTOOLS:
            result.fail(
                f"storage records openmmtools {stored_version}, which this validator does not "
                f"support (supported: {list(SUPPORTED_OPENMMTOOLS)})")
    if not isinstance(identity, dict):
        result.fail(
            "storage carries no rest2_identity in reporter metadata, so what it was created with "
            "cannot be established")

    # -- iteration accounting ---------------------------------------------------------------------
    try:
        last = int(reporter.read_last_iteration(last_checkpoint=False))
    except Exception as failure:
        return result.fail(
            f"the last committed iteration could not be read ({type(failure).__name__}: "
            f"{failure}); the analysis NetCDF is missing variables a REST2 run must have written")
    result.note("last_committed_iteration", last)

    try:
        last_checkpoint = int(reporter.read_last_iteration(last_checkpoint=True))
    except Exception as failure:
        result.fail(f"no readable checkpoint iteration ({type(failure).__name__}: {failure})")
        last_checkpoint = None
    result.note("last_checkpoint_iteration", last_checkpoint)

    # Which budget is authoritative depends on what is available, and the difference matters.
    #
    #   manifest  -- the completed run's own record, updated by every invocation. Exact.
    #   sidecar   -- the run-state record, also updated on extension. Exact.
    #   metadata  -- written ONCE at creation and never rewritten (OpenMMTools stores it with a
    #                fixed dimension), so it records the ORIGINAL request. An extended run
    #                legitimately has MORE iterations than this, and demanding equality reported a
    #                correctly extended run as incomplete.
    expected_iterations, budget_source, exact = None, None, True
    if isinstance(record, dict) and record.get("iterations_expected") is not None:
        expected_iterations, budget_source = int(record["iterations_expected"]), "manifest"
    else:
        sidecar = _read_sidecar(storage_path)
        if isinstance(sidecar, dict) and sidecar.get("iterations_requested") is not None:
            expected_iterations = int(sidecar["iterations_requested"])
            budget_source = f"run-state sidecar ({sidecar.get('status')})"
        elif isinstance(metadata, dict):
            plan = (metadata.get("rest2_plan") or {}).get("iterations_requested")
            if plan is not None:
                expected_iterations, budget_source, exact = int(plan), "reporter metadata", False
    result.note("iterations_expected", expected_iterations)
    result.note("budget_source", budget_source)

    if expect_completed and expected_iterations is not None:
        if exact and last != expected_iterations:
            result.fail(
                f"the storage's last committed iteration is {last} but the run promised "
                f"{expected_iterations} (from the {budget_source}). An incomplete budget is not "
                f"a completed run.")
        elif not exact and last < expected_iterations:
            result.fail(
                f"the storage's last committed iteration is {last}, short of the "
                f"{expected_iterations} iterations recorded at creation. An incomplete budget is "
                f"not a completed run.")
        elif not exact and last > expected_iterations:
            result.note("extended_beyond_original_request", last - expected_iterations)

    # -- the walker-to-state mapping -----------------------------------------------------------------
    n_states = None
    if isinstance(identity, dict):
        n_states = identity.get("number_of_replicas")
    try:
        mapping = np.asarray(reporter.read_replica_thermodynamic_states())
    except Exception as failure:
        result.fail(f"the walker-to-state mapping could not be read "
                    f"({type(failure).__name__}: {failure})")
        mapping = None
    if mapping is not None:
        result.note("mapping_shape", list(mapping.shape))
        if mapping.ndim != 2:
            result.fail(f"mapping has shape {mapping.shape}; expected (iterations, walkers)")
        else:
            if n_states is not None and mapping.shape[1] != int(n_states):
                result.fail(
                    f"mapping has {mapping.shape[1]} walkers but the identity records "
                    f"{int(n_states)} replicas")
            if mapping.shape[0] < last + 1:
                result.fail(
                    f"mapping holds {mapping.shape[0]} rows but the last committed iteration is "
                    f"{last}; the storage is truncated relative to its own iteration counter")
            states = int(n_states) if n_states else int(mapping.max()) + 1
            expected_row = list(range(states))
            bad = [int(i) for i, row in enumerate(mapping[:last + 1])
                   if sorted(int(v) for v in row) != expected_row]
            if bad:
                result.fail(
                    f"{len(bad)} mapping row(s) are not a permutation of the {states} "
                    f"thermodynamic states (first offenders: {bad[:8]}); a state was unoccupied "
                    f"or doubly occupied")

    # -- the final checkpoint must be readable ----------------------------------------------------
    if last_checkpoint is not None:
        try:
            sampler_states = reporter.read_sampler_states(iteration=last_checkpoint)
        except Exception as failure:
            result.fail(
                f"sampler states at the last checkpoint (iteration {last_checkpoint}) could not "
                f"be read ({type(failure).__name__}: {failure}); the checkpoint NetCDF is "
                f"missing or truncated")
            sampler_states = None
        if sampler_states is None:
            result.fail(
                f"the last checkpoint (iteration {last_checkpoint}) holds no sampler states")
        else:
            result.note("checkpoint_replicas", len(sampler_states))
            try:
                positions = np.asarray(sampler_states[0].positions._value)
                if not np.all(np.isfinite(positions)):
                    result.fail("the final checkpoint contains non-finite coordinates")
                result.note("checkpoint_positions_shape", list(positions.shape))
            except Exception as failure:
                result.fail(f"the final checkpoint's coordinates are unreadable "
                            f"({type(failure).__name__}: {failure})")

    # -- reduced potentials -------------------------------------------------------------------------
    try:
        energies, _neighborhoods, _unsampled = reporter.read_energies(iteration=last)
        energies = np.asarray(energies)
        result.note("reduced_potential_shape", list(energies.shape))
        if energies.ndim != 2 or energies.shape[0] != energies.shape[1]:
            result.fail(
                f"reduced potentials at iteration {last} have shape {energies.shape}; expected a "
                f"square (replicas, states) matrix")
        elif n_states is not None and energies.shape[0] != int(n_states):
            result.fail(
                f"reduced potentials at iteration {last} are {energies.shape[0]}x"
                f"{energies.shape[1]} but the identity records {int(n_states)} replicas")
        if not np.all(np.isfinite(energies)):
            result.fail(
                f"reduced potentials at iteration {last} contain non-finite values (NaN or inf)")
    except Exception as failure:
        result.fail(f"reduced potentials at iteration {last} could not be read "
                    f"({type(failure).__name__}: {failure})")

    # -- mixing statistics for every expected mixing event -------------------------------------------
    stride = None
    if isinstance(identity, dict):
        stride = identity.get("exchange_stride_iterations")
    try:
        accepted, proposed = reporter.read_mixing_statistics(slice(None))
        accepted, proposed = np.asarray(accepted), np.asarray(proposed)
        per_iteration = proposed.sum(axis=(1, 2))
        observed = [int(i) for i in np.nonzero(per_iteration)[0] if i <= last]
        result.note("mixing_events_observed", len(observed))
        if stride:
            expected = [i for i in range(1, last + 1) if i % int(stride) == 0]
            result.note("mixing_events_expected", len(expected))
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            if missing:
                result.fail(
                    f"{len(missing)} scheduled mixing event(s) have no stored statistics "
                    f"(first: {missing[:8]}); the configured stride is {stride}")
            if extra:
                result.fail(
                    f"{len(extra)} iteration(s) recorded mixing statistics outside the configured "
                    f"stride of {stride} (first: {extra[:8]})")
    except Exception as failure:
        result.fail(f"mixing statistics could not be read "
                    f"({type(failure).__name__}: {failure})")

    # -- the checkpoint file itself -------------------------------------------------------------------
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        result.note("checkpoint_netcdf", str(checkpoint_path))
        if not checkpoint_path.is_file():
            result.fail(f"checkpoint NetCDF {checkpoint_path} does not exist")
        else:
            try:
                import netCDF4
                with netCDF4.Dataset(str(checkpoint_path), "r") as dataset:
                    result.note("checkpoint_variables", sorted(dataset.variables)[:12])
            except Exception as failure:
                result.fail(
                    f"checkpoint NetCDF {checkpoint_path} could not be opened "
                    f"({type(failure).__name__}: {failure}); a truncated file fails here")

    # -- the manifest, cross-checked against the storage ------------------------------------------------
    if isinstance(record, dict):
        _cross_check_manifest(record, identity, metadata, last, storage_path, checkpoint,
                              result, expect_completed=expect_completed)
    return result


def _cross_check_manifest(record, identity, metadata, last, storage_path, checkpoint, result, *,
                          expect_completed):
    if expect_completed and record.get("run_status") != "completed":
        result.fail(
            f"manifest records run_status={record.get('run_status')!r}; only 'completed' is a "
            f"finished run")

    stored_names = (record.get("storage") or {})
    if stored_names.get("analysis_netcdf") not in (None, storage_path.name):
        result.fail(
            f"manifest names analysis_netcdf={stored_names.get('analysis_netcdf')!r} but the "
            f"storage validated is {storage_path.name!r}")
    if checkpoint is not None and stored_names.get("checkpoint_netcdf") not in (
            None, Path(checkpoint).name):
        result.fail(
            f"manifest names checkpoint_netcdf={stored_names.get('checkpoint_netcdf')!r} but the "
            f"checkpoint validated is {Path(checkpoint).name!r}")

    manifest_identity = record.get("scientific_identity")
    if isinstance(identity, dict) and isinstance(manifest_identity, dict):
        keys = (set(identity) | set(manifest_identity)) - {"format"}
        differences = [k for k in sorted(keys)
                       if identity.get(k) != manifest_identity.get(k)]
        if differences:
            result.fail(
                "the manifest's scientific identity disagrees with the storage's reporter "
                f"metadata in {differences[:8]}. A manifest copied from a different run, or a "
                f"changed configuration, fails here.")
    elif isinstance(identity, dict) and manifest_identity is None:
        result.fail("the manifest carries no scientific_identity to compare with the storage")

    completed = record.get("iterations_completed")
    if completed is not None and int(completed) != last:
        result.fail(
            f"manifest says {int(completed)} iterations completed but the storage's last "
            f"committed iteration is {last}")

    expected = record.get("iterations_expected")
    if expect_completed and expected is not None and completed is not None:
        if int(completed) < int(expected):
            result.fail(
                f"manifest promised {int(expected)} iterations and recorded {int(completed)}")

    recorded_events = record.get("exchange_attempts_recorded")
    observed = result.facts.get("mixing_events_observed")
    if recorded_events is not None and observed is not None and int(recorded_events) != observed:
        result.fail(
            f"manifest records {int(recorded_events)} mixing events but the storage holds "
            f"{observed}")

    for key in ("production_per_replica_ps",):
        value = record.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            result.fail(f"manifest {key} is not finite")


def format_report(result, *, title="REST2 output validation"):
    """A short human-readable report. The exit status is what a caller should act on."""
    lines = [f"# {title}"]
    for key, value in result.facts.items():
        lines.append(f"#   {key}: {value}")
    if result.ok:
        lines.append("# VALID: the storage is readable, coherent and complete.")
    else:
        lines.append(f"# INVALID: {len(result.problems)} problem(s)")
        lines.extend(f"#   - {problem}" for problem in result.problems)
    return "\n".join(lines)
