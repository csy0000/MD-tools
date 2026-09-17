"""rREST2 cases moved out of tests/test_rest2_edge_contracts.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

openmm = pytest.importorskip('openmm')

from md_tools.md import phase_space
from md_tools.rest2 import identity as hamiltonian_identity
from md_tools.remd import reservoir as rrest2_reservoir
from md_tools.remd.protocol import REST2Protocol

TEMPLATES = Path(__file__).resolve().parents[1] / 'src' / 'md_tools' / 'remd'

def _protocol():
    return REST2Protocol(tau=[0.0, 0.25, 0.5], temperature_k=300.0, timestep_fs=2.0,
                         exchange_interval_ps=2.0, number_of_exchanges=4, random_seed=7)



def _declaration(tmp_path, **overrides):
    source = {"phase_space": "src/cmd.phase_space.nc", "start_time_ps": 0.0,
              "end_time_ps": 10.0, "frames": 3}
    source.update(overrides.pop("source", {}))
    document = {
        "format": rrest2_reservoir.DECLARATION_FORMAT,
        "weighting": "boltzmann", "ensemble": "NVT", "prepared_directory": "reservoir",
        "velocity_policy": "stored", "refresh_interval_exchanges": 2, "random_seed": 11,
        "source": source,
    }
    document.update(overrides)
    # The declaration lives in the STAGE directory (`<system>/rREST2/reservoir.yaml`), and
    # `source.phase_space` is relative to the system root above it. The fixture mirrors that.
    stage = tmp_path / "rREST2"
    stage.mkdir(exist_ok=True)
    path = stage / "reservoir.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path



def _system(n_atoms=4):
    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0)
    force = openmm.NonbondedForce()
    for _ in range(n_atoms):
        force.addParticle(0.0, 0.1, 0.1)
    system.addForce(force)
    return system



def _phase_space_file(path, *, n_frames=4, n_atoms=4, velocity=1.0, periodic=False,
                      system=None, protocol=None):
    """A source in the versioned phase-space format -- the ONE source format, both policies."""
    identity = {"hamiltonian": {"system_sha256": "x"}}
    if system is not None:
        protocol = protocol or _protocol()
        identity = {"hamiltonian": hamiltonian_identity.identity_record(
            system, tau=protocol.tau[-1], temperature_k=protocol.temperature_k,
            ensemble="NVT", solute_indices=(), excluded_bonds=())}
    writer = phase_space.PhaseSpaceWriter(path, n_atoms=n_atoms, periodic=periodic,
                                          identity=identity)
    for index in range(n_frames):
        writer.append(positions=np.full((n_atoms, 3), 0.1 * (index + 1)),
                      velocities=np.full((n_atoms, 3), velocity),
                      box=(np.eye(3) * 2.0 if periodic else None),
                      step=(index + 1) * 500, time_ps=(index + 1) * 1.0)
    writer.close()
    return path



def test_open_passes_the_requested_policy_into_preparation(tmp_path, monkeypatch):
    """The gap: `open()` parsed `velocity_policy` and then called `_prepare()` without it, so the
    helper's own default `"stored"` decided how the source was validated. An explicit `maxwell`
    declaration was prepared under `stored` rules and refused for velocities it never intended to
    use."""
    seen = {}

    @classmethod
    def spy(cls, *args, **kwargs):
        seen.update(kwargs)
        raise rrest2_reservoir.ReservoirError("stop here")

    monkeypatch.setattr(rrest2_reservoir.PreparedReservoir, "_prepare", spy)
    path = _declaration(tmp_path, velocity_policy="maxwell")
    with pytest.raises(rrest2_reservoir.ReservoirError):
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    assert seen.get("policy") == "maxwell", (
        "preparation was not told which velocity policy the declaration asked for")



def test_omitting_the_field_selects_stored(tmp_path, monkeypatch):
    seen = {}

    @classmethod
    def spy(cls, *args, **kwargs):
        seen.update(kwargs)
        raise rrest2_reservoir.ReservoirError("stop here")

    monkeypatch.setattr(rrest2_reservoir.PreparedReservoir, "_prepare", spy)
    path = _declaration(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.pop("velocity_policy")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(rrest2_reservoir.ReservoirError):
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    assert seen.get("policy") == "stored"



def test_an_unknown_policy_is_refused_and_never_defaulted(tmp_path):
    path = _declaration(tmp_path, velocity_policy="thermostat")
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    message = str(caught.value)
    assert "thermostat" in message and "stored" in message and "maxwell" in message



def test_zero_velocities_are_refused_under_stored_and_accepted_under_maxwell(tmp_path):
    """`maxwell` uses the source POSITIONS and BOX and deliberately ignores the recorded velocity
    values, so a source whose velocities are identically zero is usable under it and is not usable
    under `stored`. This is the whole reason the policy has to reach the validator."""
    path = _phase_space_file(tmp_path / "zero.nc", velocity=0.0)
    with phase_space.PhaseSpaceReader(path) as reader:
        under_stored = reader.validate(expect_atoms=4, expect_periodic=False,
                                       require_velocities=True)
        under_maxwell = reader.validate(expect_atoms=4, expect_periodic=False,
                                        require_velocities=False)
    assert under_stored, "a configuration-only frame is not a phase-space sample under `stored`"
    assert under_maxwell == [], "under `maxwell` the recorded velocity values are not used"



def test_no_code_path_turns_stored_into_maxwell(tmp_path):
    """There is no automatic fallback. The only way to redraw is to ask for it."""
    source = (TEMPLATES / "reservoir.py").read_text(encoding="utf-8")
    driver = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    for text, name in ((source, "rrest2_reservoir.py"), (driver, "replica_driver.py")):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert 'policy = "maxwell"' not in stripped, name
            assert "velocity_policy = 'maxwell'" not in stripped, name
            assert 'velocity_policy = "maxwell"' not in stripped, name



def test_the_source_format_is_the_same_for_both_policies():
    """`maxwell` is not DCD support. A DCD carries no Hamiltonian identity, no absolute source
    step, and no completion marker, so it cannot satisfy this reservoir contract under either
    policy."""
    source = (TEMPLATES / "reservoir.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for claim in ("maxwell allows a dcd", "dcd is allowed under maxwell",
                  "coordinate-only source is supported"):
        assert claim not in lowered



def test_replacement_is_refused_before_anything_is_written(tmp_path):
    """The contradiction this closes: selection could draw one frame twice, materialisation sorted
    the indices, and the reader's own strictly-increasing-step invariant then rejected the file it
    had just written. Duplicates also give one configuration extra statistical weight while the
    declared frame count overstates the effective reservoir size."""
    directory = tmp_path / "src"
    directory.mkdir()
    _phase_space_file(directory / "cmd.phase_space.nc", n_frames=4)
    path = _declaration(tmp_path, source={"phase_space": "src/cmd.phase_space.nc",
                                          "start_time_ps": 0.0, "end_time_ps": 10.0,
                                          "frames": 3, "allow_sampling_with_replacement": True})
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    message = str(caught.value)
    assert "allow_sampling_with_replacement" in message
    assert not (tmp_path / "rREST2" / "reservoir" / "reservoir.nc").exists(), (
        "a prepared file was written for a request that must be refused")
    assert not (tmp_path / "rREST2" / "reservoir" / "reservoir.yaml").exists(), (
        "a manifest was left behind describing a reservoir that was never materialised")



def test_asking_for_more_frames_than_exist_says_what_is_available(tmp_path):
    directory = tmp_path / "src"
    directory.mkdir()
    system = _system(4)
    _phase_space_file(directory / "cmd.phase_space.nc", n_frames=3, system=system)
    path = _declaration(tmp_path, source={"phase_space": "src/cmd.phase_space.nc",
                                          "start_time_ps": 0.0, "end_time_ps": 10.0,
                                          "frames": 99})
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=system)
    message = str(caught.value)
    assert "99" in message, "the message does not say what was requested"
    assert "3" in message, "the message does not say how many distinct frames are eligible"



def test_a_seeded_maxwell_draw_is_reproducible():
    """What the recorded seed is worth: the same seed gives the same momenta, so a refresh can be
    reproduced from the storage alone rather than by replaying the whole rule RNG."""
    from openmm import unit

    system = _system(8)
    drawn = []
    for _ in range(2):
        integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        context = openmm.Context(system, integrator,
                                 openmm.Platform.getPlatformByName("Reference"))
        # Spread out: eight particles stacked at the origin give an infinite
        # nonbonded energy, and the drawn state comes back as NaN.
        context.setPositions(np.arange(24, dtype=float).reshape(8, 3) * unit.nanometer)
        context.setVelocitiesToTemperature(300.0 * unit.kelvin, 4242)
        drawn.append(context.getState(getVelocities=True).getVelocities(asNumpy=True)
                     .value_in_unit(unit.nanometer / unit.picosecond))
    assert np.array_equal(drawn[0], drawn[1])
    assert np.any(drawn[0]), "a Maxwell draw at 300 K is not identically zero"



def test_only_the_owning_rank_draws_and_the_array_is_shared():
    """Serial and MPI must install the SAME momenta. Every rank drawing for itself would give
    each a different array from the same seed once the contexts differ, so the owning rank draws
    and the result is shared."""
    driver = (TEMPLATES / "driver.py").read_text(encoding="utf-8")
    block = driver[driver.index("if installed_velocities is None:"):
                   driver.index("replacement = Configuration(")]
    assert "state_index in self.owned" in block, "any rank could draw"
    assert "allgather" in block, "the drawn array is not shared with the other ranks"
    assert block.index("state_index in self.owned") < block.index("allgather")



def test_the_storage_records_the_seed_each_maxwell_refresh_drew_from(tmp_path):
    from md_tools.remd import storage as storage
    from md_tools.remd.engine import Configuration

    path = tmp_path / "rest2.nc"
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2}, metadata={})
    try:
        zeros = np.zeros((2, 2), dtype=np.int64)
        reporter.write_exchange(0, step=500, time_ps=1.0, state_to_walker=[0, 1],
                                proposed=zeros, accepted=zeros, u=np.zeros((2, 2)),
                                u_evaluated=zeros.astype(np.int8),
                                reservoir=(1, 3, 2000, 1, 987654))
        reporter.write_exchange(1, step=1000, time_ps=2.0, state_to_walker=[0, 1],
                                proposed=zeros, accepted=zeros, u=np.zeros((2, 2)),
                                u_evaluated=zeros.astype(np.int8), reservoir=None)
    finally:
        reporter.close()

    reader = storage.ReplicaReporter(path, mode="r")
    try:
        seeds = reader.reservoir_velocity_seeds()
        assert list(seeds) == [987654, -1], (
            "the seed of a Maxwell refresh is not recoverable from the storage")
    finally:
        reader.close()
    assert Configuration is not None



def test_the_statistics_report_the_seeds_that_were_actually_used():
    from md_tools.remd import statistics as statistics

    accepted = np.zeros((3, 2, 2), dtype=np.int64)
    proposed = np.zeros((3, 2, 2), dtype=np.int64)
    events = np.array([[-1, -1, -1, -1], [1, 4, 2000, 1], [1, 2, 3000, 1]])
    seeds = np.array([-1, 555, 556])
    stats = statistics.lifetime_statistics(accepted, proposed, tau=[0.0, 0.5],
                                           reservoir_events=events,
                                           reservoir_velocity_seeds=seeds)
    assert stats["reservoir"]["velocity_seeds_used"] == [555, 556]

    stored_mode = statistics.lifetime_statistics(
        accepted, proposed, tau=[0.0, 0.5], reservoir_events=events,
        reservoir_velocity_seeds=np.array([-1, -1, -1]))
    assert stored_mode["reservoir"]["velocity_seeds_used"] == [], (
        "under `stored` nothing is drawn, so no seed may be reported as used")



def test_the_prepared_manifest_names_the_policy_it_was_materialised_under():
    source = (TEMPLATES / "reservoir.py").read_text(encoding="utf-8")
    manifest_block = source[source.index('"format": "md-tools-prepared-reservoir/v1"'):
                            source.index('"citations"')]
    assert '"velocity_policy": policy' in manifest_block, (
        "the manifest records only which policies the FORMAT supports, not the one selected")
