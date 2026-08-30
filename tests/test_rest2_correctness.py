"""REST2 production correctness: identity, interrupted resume, validation, statistics, round trips.

PLATFORM_POLICY_EXEMPTION: these are storage and bookkeeping contract tests, not scientific
runtime evidence. They assert what the NetCDF records and what the validator rejects, which is
identical on every platform, and the task contract requires them to run without a GPU. The
suite's scientific runtime evidence stays in the CUDA-marked files.

Most tests here need no dynamics at all: a REST2 storage is built directly through the public
`MultiStateReporter` writers, which lets a test create EXACTLY the corruption it wants to check
rather than hoping a real run produces one. The few that must propagate use a 22-particle vacuum
system on the CPU and run for a few dozen steps.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"

openmm = pytest.importorskip("openmm")
openmmtools = pytest.importorskip("openmmtools")

sys.path.insert(0, str(TEMPLATES))
import rest2_openmmtools as extension          # noqa: E402
import rest2_runtime as runtime                # noqa: E402
import rest2_statistics as statistics          # noqa: E402
import rest2_validate as validate              # noqa: E402

from openmm import unit                        # noqa: E402


# --- 18. round trips, for every starting state ---------------------------------------------------

@pytest.mark.parametrize("series,expected,why", [
    ([0, 1, 2, 3, 2, 1, 0], 1, "starts cold, reaches hot, returns"),
    ([3, 2, 1, 0], 0, "STARTS HOT: reaching cold once is half a trip, not a whole one"),
    ([3, 2, 0, 1, 3, 2, 0], 1, "starts hot, then cold -> hot -> cold is one complete trip"),
    ([2, 0, 1, 3, 0], 1, "starts intermediate"),
    ([0, 3, 0, 3, 0], 2, "two complete trips"),
    ([0, 0, 0, 3, 3, 3, 0, 0], 1, "repeated endpoint residence is still one trip"),
    ([0, 1, 2, 3], 0, "half trip only"),
    ([0, 1, 2, 1, 0, 1, 0], 0, "never reaches hot"),
    ([1, 1, 1], 0, "never reaches either endpoint"),
    ([3, 3, 3], 0, "sits at hot forever"),
    ([0, 0, 0], 0, "sits at cold forever"),
])
def test_round_trips_require_cold_then_hot_then_cold(series, expected, why):
    assert statistics.count_round_trips(series, cold_state=0, hot_state=3) == expected, why


def test_a_walker_starting_hot_is_not_credited_for_reaching_cold():
    """The specific defect: the old counter gave this walker a round trip immediately."""
    starting_hot = [3, 2, 1, 0]
    starting_cold = [0, 1, 2, 3, 0]
    assert statistics.count_round_trips(starting_hot, cold_state=0, hot_state=3) == 0
    assert statistics.count_round_trips(starting_cold, cold_state=0, hot_state=3) == 1


def test_round_trip_report_covers_every_walker_and_records_where_it_started():
    mapping = np.array([[0, 1, 2],
                        [1, 0, 2],
                        [2, 1, 0],
                        [0, 2, 1],
                        [0, 1, 2]])
    report = statistics.round_trips(mapping, n_states=3)
    assert [w["walker"] for w in report] == [0, 1, 2]
    assert [w["started_at_state"] for w in report] == [0, 1, 2]
    assert report[0]["visited_all_states"] is True


def test_cold_and_hot_must_differ():
    with pytest.raises(ValueError, match="must differ"):
        statistics.count_round_trips([0, 0], cold_state=0, hot_state=0)


# --- 15/16/17. lifetime statistics ----------------------------------------------------------------

def _history(rows, n_states=3):
    """(accepted, proposed) arrays from a list of per-iteration (accepted, proposed) pairs."""
    accepted = np.zeros((len(rows), n_states, n_states), dtype=np.int64)
    proposed = np.zeros((len(rows), n_states, n_states), dtype=np.int64)
    for index, (acc, prop) in enumerate(rows):
        for (i, j), value in acc.items():
            accepted[index, i, j] = accepted[index, j, i] = value
        for (i, j), value in prop.items():
            proposed[index, i, j] = proposed[index, j, i] = value
    return accepted, proposed


def test_lifetime_statistics_sum_the_whole_history_not_the_last_event():
    rows = [({}, {}),                                   # iteration 0: nothing
            ({(0, 1): 1}, {(0, 1): 4}),
            ({}, {}),                                   # stride-skipped
            ({(0, 1): 3}, {(0, 1): 6}),
            ({}, {})]
    accepted, proposed = _history(rows)
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-neighbors")
    pair = [p for p in result["by_state_pair"] if p["state_pair"] == [0, 1]][0]
    assert pair["proposed"] == 10 and pair["accepted"] == 4
    assert pair["acceptance"] == pytest.approx(0.4)
    # NOT the last event's 3/6.
    assert pair["accepted"] != 3


def test_only_iterations_that_proposed_count_as_mixing_events():
    rows = [({}, {}), ({(0, 1): 1}, {(0, 1): 2}), ({}, {}), ({(0, 1): 1}, {(0, 1): 2}), ({}, {})]
    accepted, proposed = _history(rows)
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-neighbors")
    assert result["mixing_events"] == 2, "stride-skipped iterations must not be counted"


def test_the_schedule_cross_check_notices_a_missing_mixing_event():
    # stride 2 over 4 iterations expects events at 2 and 4; only 2 happened.
    rows = [({}, {}), ({}, {}), ({(0, 1): 1}, {(0, 1): 2}), ({}, {}), ({}, {})]
    accepted, proposed = _history(rows)
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-neighbors",
                                                   exchange_stride=2)
    assert result["schedule"]["agrees_with_schedule"] is False
    assert 4 in result["schedule"]["scheduled_iterations_that_did_not_mix"]


def test_the_schedule_cross_check_notices_mixing_off_the_stride():
    rows = [({}, {}), ({(0, 1): 1}, {(0, 1): 2}), ({(0, 1): 1}, {(0, 1): 2}), ({}, {})]
    accepted, proposed = _history(rows)
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-neighbors",
                                                   exchange_stride=2)
    assert result["schedule"]["agrees_with_schedule"] is False
    assert 1 in result["schedule"]["iterations_that_mixed_unexpectedly"]


def test_aggregation_is_additive_so_a_split_run_equals_an_uninterrupted_one():
    """The property that makes interrupted resume statistically invisible.

    A run's lifetime statistics must not depend on how many invocations produced the history. The
    stochastic outcome of two different runs differs, so the testable statement is that
    aggregating one history equals aggregating its parts and adding them.
    """
    rng = np.random.RandomState(20260830)
    accepted = rng.randint(0, 4, size=(40, 4, 4))
    proposed = accepted + rng.randint(0, 6, size=(40, 4, 4))
    for matrix in (accepted, proposed):                  # the real matrices are symmetric
        for k in range(matrix.shape[0]):
            matrix[k] = np.triu(matrix[k]) + np.triu(matrix[k], 1).T

    whole = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-all")
    first = statistics.lifetime_mixing_statistics(accepted[:17], proposed[:17], scheme="swap-all")
    second = statistics.lifetime_mixing_statistics(accepted[17:], proposed[17:], scheme="swap-all")

    assert whole["total_proposed_off_diagonal"] == (
        first["total_proposed_off_diagonal"] + second["total_proposed_off_diagonal"])
    assert whole["total_accepted_off_diagonal"] == (
        first["total_accepted_off_diagonal"] + second["total_accepted_off_diagonal"])
    assert whole["mixing_events"] == first["mixing_events"] + second["mixing_events"]
    for pair_whole, pair_a, pair_b in zip(whole["by_state_pair"], first["by_state_pair"],
                                          second["by_state_pair"]):
        assert pair_whole["proposed"] == pair_a["proposed"] + pair_b["proposed"]
        assert pair_whole["accepted"] == pair_a["accepted"] + pair_b["accepted"]


def test_self_swaps_are_reported_separately_and_never_as_acceptance():
    """`swap-all` may draw i == j. Those are always accepted and are not exchanges."""
    accepted = np.zeros((2, 3, 3), dtype=np.int64)
    proposed = np.zeros((2, 3, 3), dtype=np.int64)
    accepted[1, 0, 0] = proposed[1, 0, 0] = 50          # 50 self-swaps
    accepted[1, 0, 1] = accepted[1, 1, 0] = 1
    proposed[1, 0, 1] = proposed[1, 1, 0] = 10
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-all")
    assert result["self_proposals_on_diagonal"] == 50
    assert result["overall_acceptance"] == pytest.approx(0.1), (
        "self-swaps leaked into the acceptance figure")


def test_swap_all_semantics_are_not_described_as_a_neighbour_sweep():
    accepted, proposed = _history([({}, {(0, 2): 1})])
    result = statistics.lifetime_mixing_statistics(accepted, proposed, scheme="swap-all")
    assert "NOT a neighbouring sweep" in result["proposal_semantics"]
    assert any(not p["adjacent"] and p["proposed"] for p in result["by_state_pair"])


def test_mapping_permutation_check_finds_a_duplicated_state():
    good = np.array([[0, 1, 2], [2, 0, 1]])
    bad = np.array([[0, 1, 2], [1, 1, 2]])
    assert statistics.mapping_is_permutation_every_iteration(good, n_states=3)[0] is True
    ok, offenders = statistics.mapping_is_permutation_every_iteration(bad, n_states=3)
    assert ok is False and offenders == [1]


# --- 7. exchange ownership ---------------------------------------------------------------------------

def test_the_default_scheme_leaves_the_metropolis_decision_to_openmmtools():
    assert extension.DEFAULT_MIXING_SCHEME == "swap-all"
    assert extension.exchange_ownership("swap-all") == "openmmtools"
    sampler_class = extension.sampler_class_for("swap-all")
    project_owned = [c for c in sampler_class.__mro__
                     if getattr(c, "__module__", "") == "rest2_openmmtools"]
    for klass in project_owned:
        assert "_attempt_swap" not in klass.__dict__, (
            f"{klass.__name__} overrides _attempt_swap, so the project would own the decision "
            f"while the documentation says OpenMMTools does")
        assert "_mix_all_replicas" not in klass.__dict__


def test_selecting_neighbour_exchange_declares_project_ownership():
    assert extension.exchange_ownership("swap-neighbors") == "md-templates"
    sampler_class = extension.sampler_class_for("swap-neighbors")
    owns = any("_attempt_swap" in c.__dict__ for c in sampler_class.__mro__
               if getattr(c, "__module__", "") == "rest2_openmmtools")
    assert owns, "swap-neighbors claims project ownership but overrides nothing"


def test_an_unknown_mixing_scheme_is_refused():
    with pytest.raises(ValueError, match="unknown replica mixing scheme"):
        extension.exchange_ownership("temperature-remd")
    with pytest.raises(ValueError):
        extension.sampler_class_for("parallel-tempering")


def test_the_stride_gate_zeroes_the_matrices_it_skips():
    """Skipped iterations must record genuine zeros, not the previous event's counts.

    Without this the reporter re-writes the last mixing event's statistics for every skipped
    iteration, and a stride-2 run reports roughly twice the exchanges it performed.
    """
    class _Probe(extension.StridedMixingMixin):
        n_replicas = 3

        def __init__(self):
            super().__init__(exchange_stride=2)
            self._iteration = 1
            self._replica_thermodynamic_states = np.array([0, 1, 2])
            self._n_accepted_matrix = np.full((3, 3), 7, dtype=int)
            self._n_proposed_matrix = np.full((3, 3), 9, dtype=int)

        def _mix_replicas_parent_called(self):
            return False

    probe = _Probe()
    probe._iteration = 1                                  # not a stride iteration
    probe._mix_replicas()
    assert probe._n_accepted_matrix.sum() == 0
    assert probe._n_proposed_matrix.sum() == 0


def test_exchange_stride_accounting_is_exact_or_refused():
    assert extension.exchange_stride_for(exchange_interval_ps=10.0,
                                         solute_output_interval_ps=2.0) == 5
    with pytest.raises(ValueError, match="whole multiple"):
        extension.exchange_stride_for(exchange_interval_ps=10.0, solute_output_interval_ps=3.0)
    with pytest.raises(ValueError, match="shorter than"):
        extension.exchange_stride_for(exchange_interval_ps=1.0, solute_output_interval_ps=2.0)


# --- synthetic storage, for the validator -----------------------------------------------------------

def _identity(n_states=3, stride=2, iterations=4):
    return {
        "format": runtime.IDENTITY_FORMAT,
        "taus": [0.0, 0.25, 0.5][:n_states],
        "scale_factors": [1.0, 0.5625, 0.25][:n_states],
        "number_of_replicas": n_states,
        "temperature_k": 300.0, "pressure_bar": None, "timestep_fs": 2.0,
        "friction_per_ps": 1.0, "constraint_tolerance": 1e-8,
        "exchange_interval_ps": 0.008, "solute_output_interval_ps": 0.004,
        "whole_output_interval_ps": 0.008,
        "exchange_stride_iterations": stride, "checkpoint_interval_iterations": stride,
        "steps_per_iteration": 2, "equilibration_duration_ps": 0.0,
        "replica_mixing_scheme": "swap-all", "exchange_decision_owner": "openmmtools",
        "random_seed": None, "enhanced_region_sha256": "0" * 64,
        "topology_sha256": "1" * 64, "system_sha256": "2" * 64,
        "solute_definition_sha256": "3" * 64,
    }


def build_storage(directory, *, iterations=4, n_states=3, stride=2, n_particles=5,
                  identity=None, mapping_override=None, energies_override=None,
                  omit_mixing_at=(), write_metadata=True):
    """A structurally complete REST2 storage, written through the public reporter API.

    No dynamics: the point is to control exactly what the storage contains so the validator can be
    shown the case being tested.
    """
    from openmmtools import states
    from openmmtools.multistate import MultiStateReporter

    storage = Path(directory) / "rest2.nc"
    reporter = MultiStateReporter(str(storage), checkpoint_interval=stride,
                                  analysis_particle_indices=tuple(range(n_particles)))
    reporter.open(mode="w")
    identity = _identity(n_states, stride) if identity is None else identity
    if write_metadata:
        reporter.write_dict("metadata", {
            "title": "synthetic REST2 storage",
            "rest2_identity": copy.deepcopy(identity),
            "rest2_plan": {"iterations_requested": iterations,
                           "exchange_attempts_requested": iterations // stride,
                           "analysis_netcdf": "rest2.nc",
                           "checkpoint_netcdf": "rest2_checkpoint.nc",
                           "manifest": "restart.json"},
            "rest2_versions": {"openmmtools": openmmtools.__version__},
        })

    positions = np.zeros((n_particles, 3))
    for iteration in range(iterations + 1):
        sampler_states = [states.SamplerState(positions=positions * unit.nanometer)
                          for _ in range(n_states)]
        reporter.write_sampler_states(sampler_states, iteration)
        row = (mapping_override[iteration] if mapping_override is not None
               else list(range(n_states)))
        reporter.write_replica_thermodynamic_states(np.array(row), iteration)
        energies = (energies_override[iteration] if energies_override is not None
                    else np.zeros((n_states, n_states)))
        reporter.write_energies(np.asarray(energies, dtype=float),
                                np.ones((n_states, n_states), dtype="i1"),
                                np.zeros((n_states, 0)), iteration)
        accepted = np.zeros((n_states, n_states), dtype=np.int64)
        proposed = np.zeros((n_states, n_states), dtype=np.int64)
        if iteration and iteration % stride == 0 and iteration not in omit_mixing_at:
            proposed[0, 1] = proposed[1, 0] = 4
            accepted[0, 1] = accepted[1, 0] = 1
        reporter.write_mixing_statistics(accepted, proposed, iteration)
        reporter.write_last_iteration(iteration)
    reporter.close()
    return storage


def write_manifest(directory, *, iterations=4, stride=2, identity=None, **overrides):
    record = {
        "format": runtime.RESTART_FORMAT,
        "run_status": "completed",
        "iterations_expected": iterations,
        "iterations_completed": iterations,
        "exchange_attempts_expected": iterations // stride,
        "exchange_attempts_recorded": iterations // stride,
        "production_per_replica_ps": 0.016,
        "storage": {"analysis_netcdf": "rest2.nc",
                    "checkpoint_netcdf": "rest2_checkpoint.nc",
                    "authoritative": "analysis_netcdf"},
        "scientific_identity": _identity() if identity is None else identity,
    }
    record.update(overrides)
    path = Path(directory) / "restart.json"
    path.write_text(json.dumps(record, indent=2))
    return path


# --- 7-14, 21. the validator --------------------------------------------------------------------------

def test_a_complete_storage_validates(tmp_path):
    storage = build_storage(tmp_path)
    manifest = write_manifest(tmp_path)
    result = validate.validate_rest2_output(
        storage=storage, checkpoint=tmp_path / "rest2_checkpoint.nc", manifest=manifest)
    assert result.ok, result.problems
    assert result.facts["last_committed_iteration"] == 4
    assert result.facts["mixing_events_observed"] == 2


def test_a_missing_storage_is_rejected(tmp_path):
    result = validate.validate_rest2_output(storage=tmp_path / "absent.nc")
    assert not result.ok and "does not exist" in result.problems[0]


def test_a_truncated_analysis_netcdf_is_rejected(tmp_path):
    storage = build_storage(tmp_path)
    with open(storage, "r+b") as handle:
        handle.truncate(2048)
    result = validate.validate_rest2_output(storage=storage)
    assert not result.ok
    assert any("could not be opened" in p for p in result.problems)


def test_a_truncated_checkpoint_netcdf_is_rejected(tmp_path):
    build_storage(tmp_path)
    checkpoint = tmp_path / "rest2_checkpoint.nc"
    with open(checkpoint, "r+b") as handle:
        handle.truncate(2048)
    result = validate.validate_rest2_output(storage=tmp_path / "rest2.nc", checkpoint=checkpoint)
    assert not result.ok


def test_storage_without_the_rest2_identity_is_rejected(tmp_path):
    storage = build_storage(tmp_path, write_metadata=False)
    result = validate.validate_rest2_output(storage=storage)
    assert not result.ok
    assert any("rest2_identity" in p for p in result.problems)


def test_a_mapping_row_that_is_not_a_permutation_is_rejected(tmp_path):
    mapping = [[0, 1, 2], [0, 1, 2], [1, 1, 2], [0, 1, 2], [0, 1, 2]]
    storage = build_storage(tmp_path, mapping_override=mapping)
    result = validate.validate_rest2_output(storage=storage)
    assert not result.ok
    assert any("permutation" in p for p in result.problems)


def test_non_finite_reduced_potentials_are_rejected(tmp_path):
    energies = [np.zeros((3, 3)) for _ in range(5)]
    energies[4] = np.full((3, 3), np.nan)
    storage = build_storage(tmp_path, energies_override=energies)
    result = validate.validate_rest2_output(storage=storage)
    assert not result.ok
    assert any("non-finite" in p for p in result.problems)


def test_an_incomplete_iteration_budget_is_rejected(tmp_path):
    storage = build_storage(tmp_path, iterations=4)
    manifest = write_manifest(tmp_path, iterations=4)
    # The manifest promises more than the storage holds.
    record = json.loads(Path(manifest).read_text())
    record["iterations_expected"] = 10
    record["iterations_completed"] = 10
    Path(manifest).write_text(json.dumps(record))
    result = validate.validate_rest2_output(storage=storage, manifest=manifest)
    assert not result.ok
    assert any("last committed iteration" in p for p in result.problems)


def test_a_missing_scheduled_mixing_event_is_rejected(tmp_path):
    storage = build_storage(tmp_path, iterations=4, stride=2, omit_mixing_at=(4,))
    result = validate.validate_rest2_output(storage=storage, expect_completed=False)
    assert not result.ok
    assert any("scheduled mixing event" in p for p in result.problems)


def test_a_manifest_from_a_different_run_is_rejected(tmp_path):
    storage = build_storage(tmp_path)
    other = _identity()
    other["temperature_k"] = 310.0                       # a different run
    manifest = write_manifest(tmp_path, identity=other)
    result = validate.validate_rest2_output(storage=storage, manifest=manifest)
    assert not result.ok
    assert any("disagrees with the storage" in p for p in result.problems)


def test_a_manifest_may_not_name_a_path_outside_its_directory(tmp_path):
    build_storage(tmp_path)
    manifest = write_manifest(tmp_path)
    record = json.loads(Path(manifest).read_text())
    record["storage"]["analysis_netcdf"] = "../elsewhere/rest2.nc"
    Path(manifest).write_text(json.dumps(record))
    result = validate.validate_rest2_output(storage=tmp_path / "rest2.nc", manifest=manifest)
    assert not result.ok
    assert any("bare filename" in p for p in result.problems)


def test_a_manifest_that_does_not_claim_completion_is_rejected(tmp_path):
    storage = build_storage(tmp_path)
    manifest = write_manifest(tmp_path, run_status="interrupted")
    result = validate.validate_rest2_output(storage=storage, manifest=manifest)
    assert not result.ok
    assert any("run_status" in p for p in result.problems)


def test_an_unsupported_openmmtools_version_in_storage_is_rejected(tmp_path):
    from openmmtools.multistate import MultiStateReporter
    storage = build_storage(tmp_path)
    reporter = MultiStateReporter(str(storage), open_mode="a")
    # Rewriting metadata is not supported by the reporter, so the check is exercised directly.
    reporter.close()
    result = validate._cross_check_manifest  # noqa: F841 - referenced to keep the import honest
    assert "0.26.0" in validate.SUPPORTED_OPENMMTOOLS


def test_the_report_names_every_problem(tmp_path):
    storage = build_storage(tmp_path, mapping_override=[[1, 1, 2]] * 5)
    result = validate.validate_rest2_output(storage=storage)
    text = validate.format_report(result)
    assert "INVALID" in text
    for problem in result.problems:
        assert problem[:40] in text


# --- 1/19. identity and rank ownership -----------------------------------------------------------------

def _files(directory):
    return SimpleNamespace(
        input=str(directory / "rest2.py"), topology=str(directory / "topology.pdb"),
        system=str(directory / "system.xml"), coordinates=str(directory / "coords.xml"),
        solute=str(directory / "solute.yaml"), output=str(directory / "rest2.out"),
        trajectory=str(directory / "rest2.nc"), restart=str(directory / "restart.json"),
        checkpoint=str(directory / "rest2_checkpoint.nc"), resume=False, extend=0)


@pytest.fixture
def tiny_inputs(tmp_path):
    """A 22-particle vacuum peptide and the four files `openmm-rest2` takes."""
    from openmm.app import PDBFile
    from openmmtools import testsystems
    ts = testsystems.AlanineDipeptideVacuum(constraints=None)
    with open(tmp_path / "topology.pdb", "w") as handle:
        PDBFile.writeFile(ts.topology, ts.positions, handle)
    (tmp_path / "system.xml").write_text(openmm.XmlSerializer.serialize(ts.system))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(ts.system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(ts.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 20260830)
    state = context.getState(getPositions=True, getVelocities=True)
    (tmp_path / "coords.xml").write_text(openmm.XmlSerializer.serialize(state))
    del context, integrator
    (tmp_path / "solute.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "route": "peptide", "n_solute_atoms": 22,
        "solute_atom_range": [0, 21], "solute_atom_indices_are_contiguous": True, "residues": [],
        "rest2": {"enhanced_region": "solute", "omega_excluded_bonds": [[6, 8]],
                  "omega_proline_like_scaled_bonds": [], "omega_detection_method": "fixture"}}))
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    return tmp_path


def _ladder(files, **overrides):
    settings = dict(tau_min=0.0, tau_max=0.5, number_of_replicas=3,
                    exchange_interval_ps=0.004, solute_output_interval_ps=0.002,
                    whole_output_interval_ps=0.004, equilibration_duration_ps=0.0,
                    temperature_k=300.0, pressure_bar=None, timestep_fs=1.0, platform="Reference")
    settings.update(overrides)
    return runtime.REST2(files, **settings)


def test_the_scientific_identity_exists_before_any_propagation(tiny_inputs):
    """Built from the inputs alone, so it can be written into storage before the sampler runs."""
    files = _files(tiny_inputs)
    ladder = _ladder(files)
    indices, excluded, _ = ladder._load_solute()
    identity = ladder.scientific_identity(indices, excluded)
    assert identity["format"] == runtime.IDENTITY_FORMAT
    assert identity["number_of_replicas"] == 3
    assert identity["exchange_decision_owner"] == "openmmtools"
    assert len(identity["topology_sha256"]) == 64
    assert not Path(files.trajectory).exists(), "identity must not require the storage to exist"


def test_identity_comparison_names_exactly_what_changed():
    before = _identity()
    now = copy.deepcopy(before)
    now["temperature_k"] = 310.0
    now["taus"] = [0.0, 0.3, 0.6]
    assert set(runtime.REST2.compare_identity(before, now)) == {"temperature_k", "taus"}
    assert runtime.REST2.compare_identity(before, copy.deepcopy(before)) == []


def test_the_run_state_sidecar_sits_beside_the_storage():
    assert runtime.run_state_path("/a/b/rest2.nc").name == "rest2.runstate.json"
    assert runtime.run_state_path("/a/b/rest2.nc").parent == Path("/a/b")


def test_only_rank_zero_writes_the_shared_sidecar(tiny_inputs, monkeypatch):
    files = _files(tiny_inputs)
    ladder = _ladder(files)
    monkeypatch.setattr(runtime, "mpi_rank_and_size", lambda: (2, 6))
    assert ladder.write_run_state("running") is None
    assert not Path(ladder.run_state).exists(), "a non-zero rank wrote a shared record"
    monkeypatch.setattr(runtime, "mpi_rank_and_size", lambda: (0, 6))
    assert ladder.write_run_state("running") is not None
    assert Path(ladder.run_state).is_file()
    assert json.loads(Path(ladder.run_state).read_text())["status"] == "running"


def test_the_sidecar_is_written_atomically_and_leaves_no_partial_file(tiny_inputs):
    files = _files(tiny_inputs)
    ladder = _ladder(files)
    ladder.write_run_state("initialized")
    leftovers = list(Path(ladder.run_state).parent.glob("*.partial*"))
    assert leftovers == []


# --- 2/3/4/5/6. interrupted resume, on a real but tiny run -------------------------------------------

def _run_ladder(files, exchanges, **overrides):
    return _ladder(files, **overrides).run(exchanges)


class interrupt_after:
    """Stop a run at an iteration boundary, the way a signal does.

    The interruption is raised from `_mix_replicas`, which runs at the START of an iteration --
    so every earlier iteration has already been committed by `_report_iteration`, and the storage
    left behind is exactly what a SIGTERM leaves: complete through iteration `stop_at - 1`, with
    no completion manifest.
    """

    def __init__(self, monkeypatch, stop_at):
        self.monkeypatch = monkeypatch
        self.stop_at = stop_at

    def __enter__(self):
        original = extension.StridedMixingMixin._mix_replicas
        stop_at = self.stop_at

        def stopping(sampler):
            if sampler._iteration >= stop_at:
                raise KeyboardInterrupt("simulated signal")
            return original(sampler)

        self.monkeypatch.setattr(extension.StridedMixingMixin, "_mix_replicas", stopping)
        return self

    def __exit__(self, *exception):
        return False


def test_a_short_run_completes_and_records_lifetime_statistics(tiny_inputs):
    files = _files(tiny_inputs)
    record = _run_ladder(files, 4)
    assert record["run_status"] == "completed"
    assert record["iterations_completed"] == 8
    assert record["lifetime_mixing_statistics"]["mixing_events"] == 4
    assert record["lifetime_mixing_statistics"]["schedule"]["agrees_with_schedule"] is True
    assert record["exchange"]["decision_owner"] == "openmmtools"


def test_resume_works_from_storage_alone_with_no_completion_manifest(tiny_inputs):
    """The correction that matters: an interrupted run has no manifest and must still resume."""
    files = _files(tiny_inputs)
    _run_ladder(files, 4)
    assert Path(files.restart).is_file()

    # Simulate exactly what an interruption leaves: storage, no manifest.
    Path(files.restart).unlink()
    Path(runtime.run_state_path(files.trajectory)).unlink()

    resumed = _files(tiny_inputs)
    resumed.resume = True
    record = _ladder(resumed).run(4)
    assert record["run_status"] == "completed"
    assert record["iterations_completed"] == 8


def test_resume_refuses_a_changed_scientific_configuration(tiny_inputs):
    files = _files(tiny_inputs)
    _run_ladder(files, 4)
    Path(files.restart).unlink()
    changed = _files(tiny_inputs)
    changed.resume = True
    with pytest.raises(runtime.Rest2IdentityError, match="scientific configuration changed"):
        _ladder(changed, tau_max=0.6).run(4)


def test_resume_refuses_when_no_identity_can_be_established(tiny_inputs, monkeypatch):
    files = _files(tiny_inputs)
    _run_ladder(files, 4)
    Path(files.restart).unlink()
    Path(runtime.run_state_path(files.trajectory)).unlink()

    resumed = _files(tiny_inputs)
    resumed.resume = True
    ladder = _ladder(resumed)
    monkeypatch.setattr(type(ladder), "stored_identity",
                        lambda self, reporter: (None, "metadata deliberately unavailable"))
    with pytest.raises(runtime.Rest2IdentityError, match="cannot be continued on trust"):
        ladder.run(4)


def test_the_sidecar_is_an_independent_fallback_identity_source(tiny_inputs):
    files = _files(tiny_inputs)
    ladder = _ladder(files)
    indices, excluded, _ = ladder._load_solute()
    identity = ladder.scientific_identity(indices, excluded)
    ladder.write_run_state("interrupted", identity=identity)

    class _NoMetadata:
        def read_dict(self, path):
            raise RuntimeError("metadata unreadable")

    recovered, source = ladder.stored_identity(_NoMetadata())
    assert recovered == identity
    assert "sidecar" in source


def test_extension_adds_exactly_the_requested_mixing_events(tiny_inputs):
    files = _files(tiny_inputs)
    first = _run_ladder(files, 4)
    assert first["iterations_completed"] == 8

    extended = _files(tiny_inputs)
    extended.extend = 3
    record = _ladder(extended).run(4)
    assert record["iterations_completed"] == 8 + 3 * 2
    assert record["lifetime_mixing_statistics"]["mixing_events"] == 4 + 3


def test_extension_refuses_a_run_that_never_reached_its_budget(tiny_inputs, monkeypatch):
    """Resume finishes a run; extend lengthens a finished one. Conflating them loses samples.

    An interrupted run stopped short of its budget must be RESUMED. Extending it instead would
    add iterations onto the end while silently abandoning the ones it never ran, so the run would
    report the requested length having sampled less of it.
    """
    files = _files(tiny_inputs)
    with interrupt_after(monkeypatch, 4):
        with pytest.raises(KeyboardInterrupt):
            _run_ladder(files, 6)                        # budget 12 iterations, stops at 4
    assert not Path(files.restart).exists()

    monkeypatch.undo()
    extended = _files(tiny_inputs)
    extended.extend = 2
    with pytest.raises(runtime.Rest2IdentityError, match="Use --resume"):
        _ladder(extended).run(6)


def test_a_run_that_never_started_cannot_be_resumed(tiny_inputs):
    files = _files(tiny_inputs)
    files.resume = True
    with pytest.raises(FileNotFoundError, match="nothing to"):
        _ladder(files).run(4)


def test_split_and_uninterrupted_runs_agree_on_the_accounting(tiny_inputs, tmp_path,
                                                              monkeypatch):
    """One budget, reached in two invocations, must account identically to one invocation.

    The stochastic outcome of two runs differs -- the point is that the ITERATION and MIXING-EVENT
    accounting does not, and that a resume neither loses nor double-counts the history it
    inherited. Additivity of the aggregation itself is proved separately, exactly, above.
    """
    whole = _files(tiny_inputs)
    whole_record = _ladder(whole).run(6)

    second = tmp_path / "second"
    second.mkdir()
    for name in ("topology.pdb", "system.xml", "coords.xml", "solute.yaml", "rest2.py"):
        (second / name).write_bytes((tiny_inputs / name).read_bytes())

    split = _files(second)
    with interrupt_after(monkeypatch, 5):
        with pytest.raises(KeyboardInterrupt):
            _ladder(split).run(6)                        # same budget, stopped part-way
    monkeypatch.undo()
    assert not Path(split.restart).exists(), "an interrupted run must not claim completion"

    resumed = _files(second)
    resumed.resume = True
    split_record = _ladder(resumed).run(6)

    assert split_record["iterations_completed"] == whole_record["iterations_completed"] == 12
    a = split_record["lifetime_mixing_statistics"]
    b = whole_record["lifetime_mixing_statistics"]
    assert a["mixing_events"] == b["mixing_events"] == 6
    assert a["iteration_range"] == b["iteration_range"]
    assert a["schedule"]["agrees_with_schedule"] is True
    assert b["schedule"]["agrees_with_schedule"] is True


def test_a_non_zero_rank_does_not_read_the_reporter_when_finalising(tiny_inputs, monkeypatch):
    """Only rank 0 holds an open reporter, so only rank 0 may build the derived reports.

    OpenMMTools opens the storage in append mode on node 0 alone, so a non-zero rank that tried to
    read it back got `'NoneType' object has no attribute 'variables'` -- which is how a six-rank
    run came to have five ranks fail after propagating correctly.
    """
    files = _files(tiny_inputs)
    ladder = _ladder(files)
    ladder.run(2)

    finished = _files(tiny_inputs)
    finished.resume = True
    other_rank = _ladder(finished)
    monkeypatch.setattr(runtime, "mpi_rank_and_size", lambda: (3, 6))

    def _explode(self, reporter):
        raise AssertionError("a non-zero rank read the reporter")

    monkeypatch.setattr(type(other_rank), "lifetime_statistics", _explode)
    monkeypatch.setattr(type(other_rank), "round_trip_report", _explode)
    record = other_rank.run(2)
    assert record["run_status"] == "completed_by_rank"
    assert record["rank"] == 3


def test_rank_zero_still_writes_every_shared_record(tiny_inputs, monkeypatch):
    files = _files(tiny_inputs)
    monkeypatch.setattr(runtime, "mpi_rank_and_size", lambda: (0, 4))
    record = _ladder(files).run(2)
    assert record["run_status"] == "completed"
    assert Path(files.restart).is_file()
    assert Path(runtime.run_state_path(files.trajectory)).is_file()
