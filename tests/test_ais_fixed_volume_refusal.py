"""AIS refuses a barostatted System, and does it before anything is created.

WHY THIS FILE EXISTS. `docs/openmm_methods/AIS/AIS_implementation.md` rests the analytic
dispersion-tail restoration on fixed volume: the correction is a function of the epsilons AND the
volume, so it is a function of tau alone only while the box is constant. The physics requires it
independently of that optimisation -- a pressure-volume term in the work would make the path
measure something the Jarzynski and Crooks relations are not written for.

That refusal existed and was UNTESTED. Two tests touched barostats and AIS:

  * `test_ais.py::test_the_switching_system_carries_no_barostat` asserts the System the switcher
    prepares carries none -- a property of what the scaler builds;
  * `test_ais_build_md.py` asserts the run record's ensemble string says "no barostat".

Neither hands a barostatted System to the preflight and asserts it is rejected, so the guard that
actually protects a user could have been deleted and both would still pass.

PLATFORM_POLICY_EXEMPTION: no Context, no platform, no propagation. `_prepare_ais` resolves the
timestep and then reads `loaded.barostats`, both off the serialised System, so the refusal is
reachable without a device -- which is the point: it happens before any output exists.
"""
from __future__ import annotations

import pytest

from md_tools.run.preflight import LoadedInputs, PreflightError, _prepare_ais


def _periodic_system(*, barostat: bool):
    """Two particles in a box, with HBonds-free masses so the timestep resolves cleanly."""
    import openmm
    from openmm import unit

    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
        nonbonded.addParticle(0.0, 0.3, 0.5)
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(3, 0, 0) * unit.nanometer,
                                          openmm.Vec3(0, 3, 0) * unit.nanometer,
                                          openmm.Vec3(0, 0, 3) * unit.nanometer))
    if barostat:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    return system


def _loaded(tmp_path, *, barostat: bool):
    """A `LoadedInputs` built the way `load_inputs` builds one, without reading files.

    `count_barostats` is what populates the field in the real loader, so it is used here too
    rather than hard-coding a number: a test that asserts on its own arithmetic would keep
    passing if the counter changed.
    """
    from openmm.app import PDBFile

    from md_tools.md._stages import count_barostats

    system = _periodic_system(barostat=barostat)
    pdb_path = tmp_path / "two.pdb"
    pdb_path.write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  C1  UNK A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  UNK A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "END\n", encoding="utf-8")
    pdb = PDBFile(str(pdb_path))
    return LoadedInputs(
        topology_path=pdb_path, system_path=tmp_path / "two.xml", pdb=pdb, system=system,
        particles=system.getNumParticles(),
        implicit=not system.usesPeriodicBoundaryConditions(),
        barostats=count_barostats(system))


ARGUMENTS = dict(
    source="unused-because-the-refusal-precedes-it",
    dynamics={"timestep_fs": 2.0},
    ais={"tau_start": 0.5, "tau_end": 0.0, "switching_steps": 100,
         "parameter_update_interval_steps": 1, "observation_interval_steps": 10},
    reporting={"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 0},
    source_config=None)


def test_a_barostatted_system_is_refused_by_name(tmp_path):
    """The guard the fixed-volume invariant actually rests on."""
    loaded = _loaded(tmp_path, barostat=True)
    assert loaded.barostats == 1, "the fixture must actually carry a barostat"

    with pytest.raises(PreflightError) as refusal:
        _prepare_ais(loaded, **ARGUMENTS)

    message = str(refusal.value)
    assert "barostat" in message
    assert "FIXED VOLUME" in message, (
        "the refusal must say WHY, not merely that something is wrong: a user who asked for NPT "
        f"needs to know the work integral is what forbids it. Got: {message}")
    assert "without a barostat" in message, "it must also say what to do instead"


def test_the_refusal_precedes_the_source_and_the_schedule(tmp_path):
    """Nothing is read and nothing is created before the box is judged.

    `source` here is a string that is not a path to anything, and `mdtraj` never opens it. If the
    barostat check ever moves below `choose_frames`, this test fails with a file error instead of
    a PreflightError -- which is the regression worth catching, because by then the refusal would
    be arriving after the source ensemble had been read and, in the real flow, after `-odir`.
    """
    loaded = _loaded(tmp_path, barostat=True)
    with pytest.raises(PreflightError) as refusal:
        _prepare_ais(loaded, **ARGUMENTS)
    assert "FIXED VOLUME" in str(refusal.value)
    # The refusal left nothing behind: the only file in tmp_path is the PDB the fixture wrote.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["two.pdb"]


def test_a_system_without_a_barostat_gets_past_this_check(tmp_path):
    """The guard must not refuse everything.

    A barostat-free System passes the fixed-volume check and fails LATER, on the source file this
    test deliberately does not provide. Asserting "not the barostat message" is the whole point:
    it distinguishes a guard that discriminates from one that always fires.
    """
    loaded = _loaded(tmp_path, barostat=False)
    assert loaded.barostats == 0

    with pytest.raises(Exception) as failure:
        _prepare_ais(loaded, **ARGUMENTS)
    assert "FIXED VOLUME" not in str(failure.value), (
        "a System with no barostat must not be refused for having one")
