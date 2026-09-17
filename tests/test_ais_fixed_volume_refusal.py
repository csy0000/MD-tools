"""AIS refuses a barostatted System, and does it before anything is created.

WHY THIS FILE EXISTS. AIS switches V(lambda) = (1 - lambda) V0 + lambda V1 at FIXED VOLUME: a
pressure-volume term in the work would make the path measure something the Jarzynski and Crooks
relations are not written for. The refusal lives in `md_tools.ais.two_state.pair_plan`, which the
preflight reaches through `TwoStateHamiltonian`, and it applies to EITHER end state.

That refusal existed and was UNTESTED. Two tests touched barostats and AIS:

  * `test_ais.py::test_the_switching_system_carries_no_barostat` asserts the System the switcher
    prepares carries none -- a property of what the scaler builds;
  * `test_ais_build_md.py` asserts the run record's ensemble string says "no barostat".

Neither hands a barostatted System to the preflight and asserts it is rejected, so the guard that
actually protects a user could have been deleted and both would still pass.

PLATFORM_POLICY_EXEMPTION: no Context, no platform, no propagation. `_prepare_ais` resolves the
timestep, loads -p2/-s2 and pairs the two serialised Systems, so the refusal is reachable without a
device -- which is the point: it happens before any output exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools.run.preflight import LoadedInputs, PreflightError, _prepare_ais


def _periodic_system(*, barostat: bool, epsilon: float = 0.5):
    """Two particles in a box, with HBonds-free masses so the timestep resolves cleanly.

    `epsilon` is the parameter the second end state changes: V0 and V1 must differ in a parameter
    and in nothing else, or the pair is refused for a reason this file is not about."""
    import openmm
    from openmm import unit

    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
        nonbonded.addParticle(0.0, 0.3, epsilon)
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(3, 0, 0) * unit.nanometer,
                                          openmm.Vec3(0, 3, 0) * unit.nanometer,
                                          openmm.Vec3(0, 0, 3) * unit.nanometer))
    if barostat:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    return system


PDB = ("CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
       "ATOM      1  C1  UNK A   1       0.000   0.000   0.000  1.00  0.00           C\n"
       "ATOM      2  C2  UNK A   1       1.500   0.000   0.000  1.00  0.00           C\n"
       "END\n")


def _end_state_1(tmp_path, *, barostat: bool = False):
    """-p2/-s2 on disk: the same two particles with a different epsilon."""
    import openmm

    path = tmp_path / "V1.xml"
    path.write_text(openmm.XmlSerializer.serialize(
        _periodic_system(barostat=barostat, epsilon=0.8)), encoding="utf-8")
    return {"topology2": tmp_path / "two.pdb", "system2": path}


def _loaded(tmp_path, *, barostat: bool):
    """A `LoadedInputs` built the way `load_inputs` builds one, without reading files.

    `count_barostats` is what populates the field in the real loader, so it is used here too
    rather than hard-coding a number: a test that asserts on its own arithmetic would keep
    passing if the counter changed.
    """
    import openmm
    from openmm.app import PDBFile

    from md_tools.md._stages import count_barostats

    system = _periodic_system(barostat=barostat)
    pdb_path = tmp_path / "two.pdb"
    pdb_path.write_text(PDB, encoding="utf-8")
    # -s on disk as well: the preflight digests it for the run identity once the pair is accepted,
    # and a missing file would fail THERE instead of on the source this file leaves out.
    (tmp_path / "two.xml").write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    pdb = PDBFile(str(pdb_path))
    return LoadedInputs(
        topology_path=pdb_path, system_path=tmp_path / "two.xml", pdb=pdb, system=system,
        particles=system.getNumParticles(),
        implicit=not system.usesPeriodicBoundaryConditions(),
        barostats=count_barostats(system))


ARGUMENTS = dict(
    source=Path("unused-because-the-refusal-precedes-it"),
    dynamics={"timestep_fs": 2.0},
    ais={"switching_steps": 100, "number_of_paths": 1,
         "parameter_update_interval_steps": 1, "observation_interval_steps": 10},
    reporting={"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 0},
    source_config=None)


def test_a_barostatted_system_is_refused_by_name(tmp_path):
    """The guard the fixed-volume invariant actually rests on."""
    loaded = _loaded(tmp_path, barostat=True)
    assert loaded.barostats == 1, "the fixture must actually carry a barostat"

    with pytest.raises(PreflightError) as refusal:
        _prepare_ais(loaded, **_end_state_1(tmp_path), **ARGUMENTS)

    message = str(refusal.value)
    assert "V0 carries a barostat" in message
    assert "FIXED VOLUME" in message, (
        "the refusal must say WHY, not merely that something is wrong: a user who asked for NPT "
        f"needs to know the work integral is what forbids it. Got: {message}")
    assert "without one" in message, "it must also say what to do instead"


def test_a_barostat_in_the_second_end_state_is_refused_by_name(tmp_path):
    """V1 is a System the user supplies separately; a barostat there is the same defect.

    The mixed System is assembled from BOTH end states' forces, so a barostat in -s2 alone would
    put pressure coupling into the switch exactly as one in -s would."""
    loaded = _loaded(tmp_path, barostat=False)
    with pytest.raises(PreflightError) as refusal:
        _prepare_ais(loaded, **_end_state_1(tmp_path, barostat=True), **ARGUMENTS)
    message = str(refusal.value)
    assert "V1 carries a barostat" in message and "FIXED VOLUME" in message, message


def test_the_refusal_precedes_the_source_and_the_schedule(tmp_path):
    """Nothing is read and nothing is created before the box is judged.

    `source` here is a string that is not a path to anything, and `mdtraj` never opens it. If the
    barostat check ever moves below `choose_frames`, this test fails with a file error instead of
    a PreflightError -- which is the regression worth catching, because by then the refusal would
    be arriving after the source ensemble had been read and, in the real flow, after `-odir`.
    """
    loaded = _loaded(tmp_path, barostat=True)
    end_state_1 = _end_state_1(tmp_path)
    with pytest.raises(PreflightError) as refusal:
        _prepare_ais(loaded, **end_state_1, **ARGUMENTS)
    assert "FIXED VOLUME" in str(refusal.value)
    # The refusal left nothing behind: the only files are the three the fixtures wrote.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["V1.xml", "two.pdb", "two.xml"]


def test_a_system_without_a_barostat_gets_past_this_check(tmp_path):
    """The guard must not refuse everything.

    A barostat-free System passes the fixed-volume check and fails LATER, on the source file this
    test deliberately does not provide. Asserting "not the barostat message" is the whole point:
    it distinguishes a guard that discriminates from one that always fires.
    """
    loaded = _loaded(tmp_path, barostat=False)
    assert loaded.barostats == 0

    with pytest.raises((Exception, PreflightError)) as failure:
        _prepare_ais(loaded, **_end_state_1(tmp_path), **ARGUMENTS)
    assert ARGUMENTS["source"].name in str(failure.value), (
        f"the pair must be accepted and the failure arrive at the source; got {failure.value!r}")
    assert "FIXED VOLUME" not in str(failure.value), (
        "a System with no barostat must not be refused for having one")
