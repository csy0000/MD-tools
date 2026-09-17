"""Annealed importance sampling: the path, the source contract, and the work bookkeeping.

Split by cost, as elsewhere in this suite:

    (unmarked)  the schedule arithmetic and configuration validation. Nothing is built.
    slow        one system built, one project generated, generated source inspected.
    gpu         energies, forces or dynamics. CUDA, and nowhere else.

The scientific claim this file has to support is narrow and specific: a Context switched along the
lambda path is, at every lambda, exactly `(1 - lambda) V0 + lambda V1` of two separately
constructed end-state Systems, and the work columns beside the coordinates are the work of the
switching that produced them. Both are checked against something independent -- separately
evaluated end states, and an analytic telescoping sum -- rather than against themselves.
"""
from __future__ import annotations

from pathlib import Path

import csv
import importlib.util
import shutil
import subprocess
import sys

import pytest
import yaml

from .conftest import ALA_PDB, run_cli

# A switching duration chosen so the arithmetic is exact and the run is seconds long, not because
# it is a scientifically meaningful path: 40 steps at 2 fs is 40 updates, which divides by the 20
# observation intervals. This is a correctness test.
SMOKE_SWITCHING_PS = 0.08
SMOKE_TRAJECTORIES = 2


# --- the schedule, with nothing built -------------------------------------------------------



def test_the_schedule_is_evenly_spaced_endpoint_inclusive_and_exact():
    """21 observations are 20 equal intervals in lambda, 0 -> 1, and both endpoints are exact."""
    from md_tools import ais

    # 40 steps, observed every 2 -> 21 observations counting both endpoints.
    schedule = ais.switching_schedule(
        switching_steps=40,
        parameter_update_interval_steps=1, observation_interval_steps=2, timestep_fs=2.0)

    assert schedule["switching_steps"] == 40
    assert schedule["number_of_updates"] == 40
    assert schedule["updates_per_observation"] == 2
    observations = schedule["observations"]
    assert len(observations) == 21
    assert observations[0]["lambda"] == 0.0 and observations[0]["protocol_step"] == 0
    # Exactly, not nearly: the last row is reported as the work of reaching V1.
    assert observations[-1]["lambda"] == 1.0
    assert observations[-1]["protocol_step"] == 40

    spacing = [observations[i + 1]["lambda"] - observations[i]["lambda"] for i in range(20)]
    assert all(abs(step - spacing[0]) < 1e-12 for step in spacing), "lambda steps are not equal"
    assert spacing[0] > 0, "lambda runs 0 -> 1, never downhill"
    assert schedule["lambdas"][0] == 0.0 and schedule["lambdas"][-1] == 1.0

    # lambda is the ONE persisted protocol coordinate. A second one -- the tau of the retired
    # single-topology switch, or its scale factors -- would offer a reader another coordinate to
    # take as authoritative, and the two disagreeing would then be a real possibility.
    for entry in observations:
        assert not {"tau", "s", "sqrt_s"} & set(entry)
    assert "taus" not in schedule

    # Observation 0 is the source configuration before any parameter change and before any
    # propagation, which is what makes its work exactly zero rather than nearly zero.
    assert schedule["observation_zero_precedes_all_work"] is True

    # The step counts are what run; ps is derived for the reader.
    assert schedule["switching_ps"] == pytest.approx(0.08)


@pytest.mark.parametrize("steps, update_interval, observe_interval, expected", [
    # Each case isolates ONE rule: the inputs satisfy every earlier check so that the message
    # under test is the one that fires.
    #
    # 15 steps, update every 1 (15 % 1 == 0), observe every 2 -> 15 % 2 != 0, so the last
    # observation would not land at lambda = 1.
    (15, 1, 2, "observation_interval_steps"),
    # 20 steps, update every 3 -> 20 % 3 != 0, so the final parameter change lands mid-interval.
    (20, 3, 5, "parameter_update_interval_steps"),
    # 12 steps, update every 2 (12 % 2 == 0), observe every 3 (12 % 3 == 0), but 3 % 2 != 0, so
    # observations would not sit on the parameter-update grid.
    (12, 2, 3, "would not land on the"),
])
def test_a_schedule_that_would_have_to_be_rounded_is_refused(steps, update_interval,
                                                             observe_interval, expected):
    """Every inexact division is refused with the arithmetic that would fix it, never rounded."""
    from md_tools import ais

    with pytest.raises(ValueError) as error:
        ais.switching_schedule(switching_steps=steps,
                               parameter_update_interval_steps=update_interval,
                               observation_interval_steps=observe_interval, timestep_fs=2.0)
    assert expected in str(error.value)








# --- generation ------------------------------------------------------------------------------











def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module










@pytest.fixture(scope="module")
def ais_project(tmp_path_factory):
    """A built system, small enough to run in seconds.

    These tests need a serialised System and its solute selection, not a whole generated project.
    The project layout they used to be built from is gone; the science they check is not.
    """
    import subprocess
    import sys as _sys

    work = tmp_path_factory.mktemp("ais")
    (work / "small.config").write_text(
        "solvent:\n  padding_nm: 0.5\n  cutoff_nm: 0.6\n", encoding="utf-8")
    built = subprocess.run(
        [_sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ALA_PDB),
         "-os", "built.xml", "-op", "built.pdb", "-log", "built.log",
         "--config", "small.config"],
        cwd=work, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    return work


def _generated_scaling(project=None):
    """The scaling module. Installed now, not copied into a generated project."""
    from md_tools import rest2 as rest2_scaling

    return rest2_scaling


def _solute_indices(project):
    """The solute selection, derived the way the runtime derives it."""
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices

    return solute_atom_indices(PDBFile(str(Path(project) / "built.pdb")).topology)


def _excluded_bonds(project):
    from openmm.app import PDBFile

    from md_tools.openmm.system import classify_unscaled_torsions

    topology = PDBFile(str(Path(project) / "built.pdb")).topology
    omega = classify_unscaled_torsions(topology, _solute_indices(project))
    return [tuple(int(a) for a in bond) for bond in omega.get("unscaled_central_bonds", [])]


def _end_states(project):
    """The pair the previous AIS switched between, expressed as two files' worth of Systems.

    V0 is the REST2 System at tau = 0.5, the ensemble the old tau switch started from, built by the
    same `build_scaled_system` a REST2 rung is built with; V1 is the physical System. AIS itself
    scales nothing: it mixes whatever two end states it is given.
    """
    from openmm import XmlSerializer

    scaling = _generated_scaling(project)
    base = XmlSerializer.deserialize((Path(project) / "built.xml").read_text())
    v0 = scaling.build_scaled_system(base, _solute_indices(project), 0.5,
                                     _excluded_bonds(project))
    return v0, base


@pytest.mark.slow
def test_the_switching_system_carries_no_barostat(ais_project):
    """Fixed volume is a property of the System, not a runtime flag.

    `built.xml` carries the NPT chain's barostat, so the explicit fixture's physical System is
    exactly the case the rule exists for: a pair containing one is refused, and the mixed System
    built from a pair without one carries none.
    """
    from md_tools.ais.two_state import EndStateError, TwoStateHamiltonian

    v0, v1 = _end_states(ais_project)
    barostats = [i for i in range(v1.getNumForces())
                 if "Barostat" in v1.getForce(i).__class__.__name__]
    if barostats:
        with pytest.raises(EndStateError, match="barostat"):
            TwoStateHamiltonian(v0, v1)
        v0, v1 = _without_barostat(v0), _without_barostat(v1)
    system = TwoStateHamiltonian(v0, v1).system
    names = [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]
    assert not any("Barostat" in name for name in names), names


def _without_barostat(system):
    for index in reversed(range(system.getNumForces())):
        if "Barostat" in system.getForce(index).__class__.__name__:
            system.removeForce(index)
    return system


# --- the two-state switch against its end states, on CUDA -----------------------------------

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 1.0])
def test_dynamic_switching_reproduces_the_static_end_state_mixture(ais_project, lam):
    """The Hamiltonian must actually change through a Context parameter, and match the end states.

    This is what separates exact switching from interpolating between two endpoint energies
    after the fact: the separately constructed end-state Systems -- V0 from `build_scaled_system`,
    the function REST2 builds a rung with, and V1 the physical System -- are evaluated in their own
    Contexts, and the live switched Context is compared with `(1 - lambda) V0 + lambda V1` on the
    total energy AND on every atom's force. Forces are the stronger test: an energy can agree by
    cancellation, a full force array cannot.
    """
    import numpy as np
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm.app import PDBFile

    from md_tools.ais.two_state import TwoStateHamiltonian

    v0, v1 = (_without_barostat(system) for system in _end_states(ais_project))
    pdb = PDBFile(str(ais_project / "built.pdb"))
    assert _excluded_bonds(ais_project), "the ALA fixture should have omega bonds to exclude"

    platform = Platform.getPlatformByName("CUDA")
    # Double precision so the comparison is limited by the Hamiltonians, not by the platform.
    properties = {"Precision": "double"}

    def context_for(system):
        context = Context(system, VerletIntegrator(0.001 * unit.femtosecond), platform,
                          properties)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(pdb.positions)
        return context

    def measure(context):
        result = context.getState(getEnergy=True, getForces=True)
        return (result.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.array(result.getForces().value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer)))

    reference = context_for(v0)
    e0, f0 = measure(reference)
    del reference
    reference = context_for(v1)
    e1, f1 = measure(reference)
    del reference
    static_energy = (1.0 - lam) * e0 + lam * e1
    static_forces = (1.0 - lam) * f0 + lam * f1

    hamiltonian = TwoStateHamiltonian(v0, v1)
    context = context_for(hamiltonian.system)
    hamiltonian.set_lambda(context, lam)
    dynamic_energy, dynamic_forces = measure(context)

    tolerance = 1e-9 * max(abs(e0), abs(e1))
    assert dynamic_energy == pytest.approx(static_energy, abs=max(1e-6, tolerance), rel=0)
    assert np.abs(dynamic_forces - static_forces).max() < 1e-5

    # ...and switching away and back must land on the same Hamiltonian, not on a compounded one.
    hamiltonian.set_lambda(context, 0.4)
    hamiltonian.set_lambda(context, lam)
    again_energy, again_forces = measure(context)
    assert again_energy == pytest.approx(static_energy, abs=max(1e-6, tolerance), rel=0)
    assert np.abs(again_forces - static_forces).max() < 1e-5


@pytest.mark.slow
def test_unscaled_torsions_are_left_alone_in_the_scaled_end_state(ais_project):
    """The torsions REST2 leaves unscaled, checked on the V0 an AIS switch starts from.

    AIS no longer scales anything, so the unscaled-torsion rule cannot be enforced by the switch;
    it has to already be true of the end-state file. Each torsion of the REST2-built V0 is checked
    against the physical System through `torsion_is_scaled`, the ONE rule every scaler asks -- so
    under convention v3 an improper (e.g. C-N-CA-H) is expected unscaled, exactly as an amide
    omega is, rather than being reported as a torsion the scaler forgot.
    """
    from openmm import PeriodicTorsionForce, unit

    from md_tools.rest2.hamiltonian import system_bond_graph, torsion_is_scaled

    v0, base = _end_states(ais_project)
    solute = set(_solute_indices(ais_project))
    unscaled_bonds = {frozenset((int(a), int(b))) for a, b in _excluded_bonds(ais_project)}
    bonds = system_bond_graph(base)

    def torsions(system):
        force = next(system.getForce(i) for i in range(system.getNumForces())
                     if isinstance(system.getForce(i), PeriodicTorsionForce))
        return [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]

    unscaled_seen = scaled_seen = 0
    for original, now in zip(torsions(base), torsions(v0)):
        atoms = original[:4]
        # Stripped of units so the comparison is between numbers, not Quantities.
        k_before = original[6].value_in_unit(unit.kilojoule_per_mole)
        k_after = now[6].value_in_unit(unit.kilojoule_per_mole)
        label = "-".join(str(a) for a in atoms)
        if torsion_is_scaled(atoms, solute, unscaled_bonds, bonds):
            assert k_after == pytest.approx(k_before * 0.25), \
                f"solute torsion {label} was not scaled by s"
            scaled_seen += 1
        else:
            assert k_after == k_before, f"torsion {label} was scaled but the rule leaves it alone"
            unscaled_seen += int(all(a in solute for a in atoms))
    assert unscaled_seen and scaled_seen, (unscaled_seen, scaled_seen)


@pytest.mark.gpu
@pytest.mark.slow
def test_frozen_coordinate_work_telescopes_to_the_endpoint_energy_difference(ais_project):
    """With coordinates held fixed, the summed increments must be exactly the endpoint difference.

    delta_W_j = V(lambda_{j+1}, x) - V(lambda_j, x) with x unchanged, so every intermediate term
    cancels and the total is V1(x) - V0(x). Anything else means the increments are not the quantity
    the record calls work.
    """
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm.app import PDBFile

    from md_tools.ais.two_state import TwoStateHamiltonian

    v0, v1 = (_without_barostat(system) for system in _end_states(ais_project))
    pdb = PDBFile(str(ais_project / "built.pdb"))

    # The schedule is computed, not read from a generated file: `switching_schedule` is the one
    # implementation of the arithmetic.
    from md_tools.ais import switching_schedule

    schedule = switching_schedule(switching_steps=40, parameter_update_interval_steps=1,
                                  observation_interval_steps=2, timestep_fs=2.0)
    lambdas = schedule["lambdas"]

    hamiltonian = TwoStateHamiltonian(v0, v1)
    live = hamiltonian.system
    context = Context(live, VerletIntegrator(0.001 * unit.femtosecond),
                      Platform.getPlatformByName("CUDA"), {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(pdb.positions)
    hamiltonian.set_lambda(context, lambdas[0])

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    start_energy = energy()
    cumulative = 0.0
    for index in range(len(lambdas) - 1):
        before = energy()                        # no propagation: x is frozen
        hamiltonian.set_lambda(context, lambdas[index + 1])
        cumulative += energy() - before
    end_energy = energy()

    assert cumulative == pytest.approx(end_energy - start_energy, abs=1e-6, rel=0)
    # ...and the endpoint difference is the one the derivative reads in a single evaluation.
    assert end_energy - start_energy == pytest.approx(hamiltonian.difference(context),
                                                      abs=1e-5, rel=1e-9)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_constant_lambda_diagnostic_path_does_zero_work(ais_project):
    """A path that never changes the Hamiltonian does no work, whatever the coordinates do.

    The public schedule always runs 0 -> 1, and this is why a constant-lambda path is a diagnostic
    rather than a configuration: it is what pins the work convention to the parameter change
    rather than to the propagation.
    """
    from openmm import Context, LangevinMiddleIntegrator, Platform, unit
    from openmm.app import PDBFile

    from md_tools.ais.two_state import TwoStateHamiltonian

    v0, v1 = (_without_barostat(system) for system in _end_states(ais_project))
    pdb = PDBFile(str(ais_project / "built.pdb"))

    hamiltonian = TwoStateHamiltonian(v0, v1)
    live = hamiltonian.system
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                          2.0 * unit.femtoseconds)
    integrator.setRandomNumberSeed(20260827)
    context = Context(live, integrator, Platform.getPlatformByName("CUDA"),
                      {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 4242)
    hamiltonian.set_lambda(context, 0.5)

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    cumulative = 0.0
    for _ in range(20):
        before = energy()
        hamiltonian.set_lambda(context, 0.5)     # the same lambda, every time
        cumulative += energy() - before
        integrator.step(2)                       # and the coordinates DO move
    assert cumulative == pytest.approx(0.0, abs=1e-6)


# --- a tiny CUDA run ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ais_run(ais_project):
    """The common chain, a cMD source sampled under V0, and two AIS paths. On CUDA."""
    pytest.importorskip("mdtraj", reason="the AIS runtime reads its source trajectory with MDTraj")
    from .conftest import run_stage

    project = ais_project / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        result = run_stage(project / stage)
        assert result.returncode == 0, result.stdout + result.stderr
    produced = run_stage(project / "cMD")
    assert produced.returncode == 0, produced.stdout + produced.stderr

    switched = run_stage(project / "AIS")
    assert switched.returncode == 0, switched.stdout + switched.stderr
    return project / "AIS"














# --- the shared production stage contract -------------------------------------
#
# AIS was the last production method without one. It froze its schedule in
# `path_definition.yaml` and refused a definition that disagreed with the ensemble its inputs were
# prepared from, which covered the source. What it could not do was recognise its own completed run
# or notice a coordinated edit -- one that moves `md.config.yaml` and `stage.yaml` together, so the
# runtime-request check finds them consistent and passes.






