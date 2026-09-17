"""The corrections this branch makes to the owned REST2 runtime.

rREST2 is archived (0.5.4); its velocity-policy cases moved to
archive/rREST2/tests/test_rest2_phase_space_corrections.py.

Each test names the defect it pins. They are written so that they FAIL against the previous
implementation: a passing run here is evidence the specific behaviour changed, not that the
module merely imports.

PLATFORM_POLICY_EXEMPTION: scheduling arithmetic, storage layout, identity digests and reservoir
contracts. The two that propagate use a 22-particle vacuum peptide for a handful of steps to
exercise bookkeeping rather than to produce scientific evidence; the CUDA-marked files hold the
runtime evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
SRC = Path(__file__).resolve().parents[1] / "src"

openmm = pytest.importorskip("openmm")
sys.path.insert(0, str(SRC))

from md_tools.rest2 import identity as hamiltonian_identity                                        # noqa: E402
from md_tools.md import phase_space                                                 # noqa: E402
from md_tools.remd import storage as storage
from md_tools.remd.schedule import EVENT_ORDER, EventSchedule            # noqa: E402
from md_tools.remd.protocol import ProtocolError, REST2Protocol          # noqa: E402


def _peptide_system(n=22, periodic=False):
    system = openmm.System()
    for _ in range(n):
        system.addParticle(12.0)
    force = openmm.NonbondedForce()
    for _ in range(n):
        force.addParticle(0.0, 0.1, 0.1)
    if periodic:
        force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0),
                                              openmm.Vec3(0, 0, 2)))
    system.addForce(force)
    return system


# --- 1. exact integer-step scheduling, and no user-facing segment_ps --------------------------

def test_segment_ps_is_refused_with_a_migration_message():
    """`segment_ps` bundled four independent schedules into one number. It is gone, and asking
    for it is an error that names what replaced it rather than being silently ignored."""
    with pytest.raises(ProtocolError) as caught:
        REST2Protocol(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=2.0,
                      exchange_interval_ps=2.0, number_of_exchanges=4, segment_ps=2.0)
    message = str(caught.value)
    assert "segment_ps" in message
    assert "whole_output_interval_ps" in message and "solute_output_interval_ps" in message


def test_schedules_are_independent_and_land_on_exact_integer_steps():
    """The defect: one interval drove exchange, output and checkpoint together, so a solute
    stream could not be denser than the exchange period. Here it is 10x denser, and every event
    step is an exact integer multiple -- no float drift, no rounding into a neighbouring step."""
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=10.0, number_of_exchanges=20,
                             whole_output_interval_ps=100.0, solute_output_interval_ps=2.0,
                             checkpoint_interval_ps=10.0)
    counts = schedule.counts()
    assert counts["exchange"] == 20
    assert counts["solute"] == 100          # denser than the exchange period
    assert counts["whole"] == 2             # sparser than it
    assert schedule.total_steps == 100_000

    for kind, interval in (("exchange", 5_000), ("solute", 1_000), ("whole", 50_000)):
        steps = schedule.event_steps(kind)
        assert all(step % interval == 0 for step in steps), kind
        assert steps == sorted(set(steps)), kind


def test_simultaneous_events_resolve_in_one_fixed_order():
    """When several schedules coincide, the order has to be defined once.

    Collective variables FIRST, so a row on an exchange boundary describes the configuration as
    propagated rather than one that arrived from another rung; exchange next, so a frame written
    at that step is the post-exchange state; and the checkpoint last, so it can describe
    everything already written.
    """
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=2.0, number_of_exchanges=2,
                             whole_output_interval_ps=2.0, solute_output_interval_ps=2.0,
                             checkpoint_interval_ps=2.0, cv_interval_steps=1_000)
    at = schedule.events_at(1_000)
    assert at == [kind for kind in EVENT_ORDER if kind in at]
    assert set(at) == set(EVENT_ORDER), "a scheduled event kind is missing from this coincidence"
    # The two orderings the conventions actually depend on, asserted rather than implied by the
    # tuple's spelling.
    assert at.index("cv") < at.index("exchange"), "CV rows would become post-exchange"
    assert at.index("exchange") < at.index("whole") < at.index("checkpoint")


def test_a_schedule_without_collective_variables_simply_has_no_cv_events():
    """Reporting is off by default, and an absent interval must schedule nothing at all."""
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=2.0, number_of_exchanges=2,
                             whole_output_interval_ps=2.0, checkpoint_interval_ps=2.0)
    assert schedule.cv_steps is None
    assert "cv" not in schedule.events_at(1_000)


def test_propagation_never_steps_past_an_event():
    """The scheduler hands out the quantum to the NEXT event, so no event can be straddled."""
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=10.0, number_of_exchanges=3,
                             whole_output_interval_ps=4.0, solute_output_interval_ps=3.0)
    step, seen = 0, []
    while step < schedule.total_steps:
        target = schedule.next_event_step(step)
        assert target > step
        step = target
        seen.append(step)
    for kind in ("exchange", "whole", "solute"):
        assert set(schedule.event_steps(kind)) <= set(seen), kind


# --- 2/3. a phase-space reservoir stores velocities; a DCD is not one --------------------------

def test_phase_space_round_trip_preserves_velocities_and_absolute_steps(tmp_path):
    path = tmp_path / "source.nc"
    writer = phase_space.PhaseSpaceWriter(path, n_atoms=4, periodic=False, identity={"x": 1})
    velocities = []
    for index in range(3):
        v = np.full((4, 3), 0.5 + index, dtype=float)
        velocities.append(v)
        writer.append(positions=np.zeros((4, 3)), velocities=v, box=None,
                      step=(index + 1) * 500, time_ps=(index + 1) * 1.0)
    writer.close()

    with phase_space.PhaseSpaceReader(path) as reader:
        assert reader.validate(expect_atoms=4, expect_periodic=False) == []
        assert list(reader.steps()) == [500, 1000, 1500]
        for index, expected in enumerate(velocities):
            _, stored, _, step, _ = reader.frame(index)
            assert np.allclose(stored, expected)
            assert step == (index + 1) * 500


def test_a_configuration_only_file_is_refused_as_a_phase_space_source(tmp_path):
    """A reservoir with no recorded momentum cannot honour `velocity_policy: stored`, and the
    refusal has to happen when the file is opened rather than at the first refresh."""
    path = tmp_path / "no_velocities.nc"
    writer = phase_space.PhaseSpaceWriter(path, n_atoms=4, periodic=False, identity={})
    writer.append(positions=np.zeros((4, 3)), velocities=np.zeros((4, 3)), box=None,
                  step=500, time_ps=1.0)
    writer.close()
    with phase_space.PhaseSpaceReader(path) as reader:
        problems = reader.validate(expect_atoms=4, expect_periodic=False,
                                   require_velocities=True)
    assert problems, "an all-zero velocity frame is not a phase-space sample"


# --- 9. same complete Hamiltonian identity, recomputed on both sides ---------------------------

def test_identity_compares_the_serialized_system_not_a_stored_claim():
    left = _peptide_system()
    right = _peptide_system()
    record = hamiltonian_identity.identity_record(
        left, tau=0.5, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    same = hamiltonian_identity.identity_record(
        right, tau=0.5, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    hamiltonian_identity.require_same_hamiltonian(record, same)   # identical build: no raise

    right.getForce(0).setParticleParameters(0, 0.1, 0.1, 0.1)
    changed = hamiltonian_identity.identity_record(
        right, tau=0.5, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    assert changed["tau"] == record["tau"]
    assert changed["temperature_k"] == record["temperature_k"]
    with pytest.raises(hamiltonian_identity.HamiltonianMismatch):
        hamiltonian_identity.require_same_hamiltonian(record, changed)


def test_the_unscaled_reference_is_not_recorded_as_the_rung_at_tau_zero():
    """The defect this pins: the driver fingerprinted the UNSCALED reference system while
    labelling the record with tau_max, so the digest and the tau described different objects.
    A reference is now recorded as tau=None, which is a different claim from tau=0.0."""
    system = _peptide_system()
    reference = hamiltonian_identity.identity_record(
        system, tau=None, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    rung = hamiltonian_identity.identity_record(
        system, tau=0.0, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    assert reference["tau"] is None and rung["tau"] == 0.0
    with pytest.raises(hamiltonian_identity.HamiltonianMismatch):
        hamiltonian_identity.require_same_hamiltonian(reference, rung)


def test_the_enhanced_region_is_part_of_the_identity():
    """Two runs from one base System with different solute selections are different Hamiltonians
    at every tau above zero, and the System digest alone cannot see it."""
    system = _peptide_system()
    wide = hamiltonian_identity.identity_record(
        system, tau=0.5, temperature_k=300.0, ensemble="NVT", solute_indices=range(22))
    narrow = hamiltonian_identity.identity_record(
        system, tau=0.5, temperature_k=300.0, ensemble="NVT", solute_indices=range(10))
    assert wide["system_sha256"] == narrow["system_sha256"]
    with pytest.raises(hamiltonian_identity.HamiltonianMismatch):
        hamiltonian_identity.require_same_hamiltonian(wide, narrow)


# --- 3. separate streams reach separate files, each with its own marker ------------------------

def test_the_solute_stream_is_its_own_file_with_its_own_completion_marker(tmp_path):
    """A dense solute stream and a sparse whole-system stream cannot share one frame dimension:
    the previous single-dimension layout forced them to the same interval."""
    path = tmp_path / "rest2.nc"
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=2.0, number_of_exchanges=2,
                             whole_output_interval_ps=4.0, solute_output_interval_ps=1.0)
    assert storage.solute_path(path).name == "rest2.solute.nc"
    assert storage.solute_path(path) != path

    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=2, has_box=False,
        identity={"schedule": schedule.describe()}, metadata={})
    try:
        counts = schedule.counts()
        # The two streams are budgeted separately: the solute stream is four times as dense as
        # the whole-system one, which a single shared frame dimension could not express.
        assert counts["solute"] == 4 * counts["whole"]
    finally:
        reporter.close()
    assert storage.solute_path(path).exists(), "the solute stream is a file of its own"


# --- 10. extension builds on the budget the run reached ----------------------------------------

def test_extending_twice_adds_to_the_budget_each_time():
    """The defect: `--extend` lengthened the PROTOCOL's budget rather than the run's, so a second
    extension produced a total below the steps already run. The loop then propagated nothing and
    the summary reported "220000 of 210000" while exiting successfully."""
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=2.0, number_of_exchanges=200,
                             whole_output_interval_ps=10.0, solute_output_interval_ps=1.0)
    once = schedule.extended(20)
    assert once.total_steps == 220_000

    # What the driver reconstructs on a second extension: how far the STORED budget already runs
    # beyond the protocol's, plus the newly requested amount.
    already = (once.total_steps - schedule.total_steps) // schedule.exchange_steps
    twice = schedule.extended(already + 10)
    assert already == 20
    assert twice.total_steps == 230_000 > once.total_steps


# --- storage corruption: a completion marker cannot lead its own data --------------------------

def test_a_marker_ahead_of_its_data_is_refused(tmp_path):
    """The marker is committed after the rows it describes, so a crash leaves it BEHIND them. A
    marker ahead of them cannot arise that way: the file disagrees with itself."""
    from md_tools.remd import validate as replica_validate

    path = tmp_path / "rest2.nc"
    schedule = EventSchedule(timestep_fs=2.0, exchange_interval_ps=2.0, number_of_exchanges=2,
                             whole_output_interval_ps=2.0, solute_output_interval_ps=2.0)
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=2, has_box=False,
        identity={"n_states": 2, "schedule": schedule.describe()}, metadata={})
    reporter.close()

    import netCDF4
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables["last_exchange"][0] = 99

    result = replica_validate.validate_replica_output(analysis=path, expect_completed=False)
    assert not result.ok
    assert any("holds only" in problem for problem in result.problems)


# --- a stated platform must reach the runtime -------------------------------------------------


def test_the_platform_is_not_part_of_the_scientific_identity():
    """Physics is what a continuation must agree with. Resuming on another machine's platform is
    legitimate, so the platform is carried but never compared."""
    protocol = REST2Protocol(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=2.0,
                             exchange_interval_ps=2.0, number_of_exchanges=2, platform="CPU")
    assert protocol.platform == "CPU"
    assert "platform" not in protocol.describe()


def test_the_generated_replica_protocol_states_its_platform():
    """The requested platform must actually reach the runtime.

    Ported from the retired `emit.replica_protocol_file`. It is carried but never compared:
    resuming on another machine's platform is legitimate, so it is not part of the scientific
    identity -- which the test below this one asserts.
    """
    from md_tools.remd.generated import protocol_file_text

    ladder = {"protocol": "REST2", "solvent": "implicit", "n_states": 2, "tau_max": 0.5,
              "exchange_interval_steps": 1000, "number_of_exchanges": 10,
              "state_trajectory": True, "rem_log": True, "neighbour_acceptance_report": True,
              "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "pressure_bar": 1.0,
                           "friction_per_ps": 1.0, "barostat_interval_steps": 25,
                           "restraint_kcal_per_mol_A2": 1.0, "seed": 1, "platform": "CPU",
                           "tau": 0.0, "phase_space_printout": 0}}
    text = protocol_file_text(ladder)
    assert "platform='CPU'" in text or 'platform="CPU"' in text
