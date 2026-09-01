"""Annealed importance sampling: the path, the source contract, and the work bookkeeping.

Split by cost, as elsewhere in this suite:

    (unmarked)  the schedule arithmetic and configuration validation. Nothing is built.
    slow        one system built, one project generated, generated source inspected.
    gpu         energies, forces or dynamics. CUDA, and nowhere else.

The scientific claim this file has to support is narrow and specific: a Context switched along the
tau path is, at every tau, the same Hamiltonian as a separately constructed static REST2 rung, and
the work columns beside the coordinates are the work of the switching that produced them. Both are
checked against something independent -- a separately built System, and an analytic telescoping
sum -- rather than against themselves.
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
    """21 observations are 20 equal intervals in tau, and both endpoints are exact."""
    from md_tools.openmm import ais

    # 40 steps, observed every 2 -> 21 observations counting both endpoints.
    schedule = ais.switching_schedule(
        tau_start=0.5, tau_end=0.0, switching_steps=40,
        parameter_update_interval_steps=1, observation_interval_steps=2, timestep_fs=2.0)

    assert schedule["switching_steps"] == 40
    assert schedule["number_of_updates"] == 40
    assert schedule["updates_per_observation"] == 2
    observations = schedule["observations"]
    assert len(observations) == 21
    assert observations[0]["tau"] == 0.5 and observations[0]["protocol_step"] == 0
    # Exactly, not nearly: the last row is reported as the work of reaching tau_end.
    assert observations[-1]["tau"] == 0.0
    assert observations[-1]["protocol_step"] == 40

    spacing = [observations[i + 1]["tau"] - observations[i]["tau"] for i in range(20)]
    assert all(abs(step - spacing[0]) < 1e-12 for step in spacing), "tau steps are not equal"

    # tau is the ONE persisted protocol coordinate. The scale factors are derived inside the
    # scaler; persisting them too would offer a reader a second coordinate to take as
    # authoritative, and s and tau disagreeing would then be a real possibility.
    for entry in observations:
        assert "s" not in entry and "sqrt_s" not in entry

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
    # observation would not land at tau_end.
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
    from md_tools.openmm import ais

    with pytest.raises(ValueError) as error:
        ais.switching_schedule(tau_start=0.5, tau_end=0.0, switching_steps=steps,
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
    from md_tools.openmm.templates import rest2_scaling

    return rest2_scaling


def _solute_indices(project):
    """The solute selection, derived the way the runtime derives it."""
    from openmm.app import PDBFile

    from md_tools.runtime.stage import solute_atom_indices

    return solute_atom_indices(PDBFile(str(Path(project) / "built.pdb")).topology)


def _excluded_bonds(project):
    from openmm.app import PDBFile

    from md_tools.openmm.system import classify_omega_bonds

    topology = PDBFile(str(Path(project) / "built.pdb")).topology
    omega = classify_omega_bonds(topology, _solute_indices(project), route="peptide",
                                 ligand_sdf=None)
    return [tuple(int(a) for a in bond) for bond in omega.get("omega_unscaled_bonds", [])]


@pytest.mark.slow
def test_the_switching_system_carries_no_barostat(ais_project):
    """Fixed volume is a property of the System, not a runtime flag."""
    from openmm import XmlSerializer

    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize((ais_project / "built.xml").read_text())
    switcher = scaling.TauSwitcher(base, _solute_indices(ais_project), [])
    system = switcher.prepared_system(0.5)
    names = [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]
    assert not any("Barostat" in name for name in names), names


# --- dynamic switching against static REST2 ---------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5])
def test_dynamic_switching_reproduces_a_static_rest2_system(ais_project, tau):
    """The Hamiltonian must actually change through force parameters, and match a built rung.

    This is what separates exact switching from interpolating between two endpoint energies: a
    separately constructed `build_scaled_system(..., tau)` -- the same function REST2 uses to build
    a rung -- and the live switched Context are compared on the total energy AND on every atom's
    force. Forces are the stronger test: an energy can agree by cancellation, a full force array
    cannot.
    """
    import numpy as np
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize((ais_project / "built.xml").read_text())
    pdb = PDBFile(str(ais_project / "built.pdb"))
    solute = _solute_indices(ais_project)
    omega = _excluded_bonds(ais_project)
    assert omega, "the ALA fixture should have omega bonds to exclude"

    platform = Platform.getPlatformByName("CUDA")
    # Double precision so the comparison is limited by the Hamiltonians, not by the platform.
    properties = {"Precision": "double"}

    def measure(context):
        result = context.getState(getEnergy=True, getForces=True)
        return (result.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.array(result.getForces().value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer)))

    static = scaling.build_scaled_system(base, solute, tau, omega)
    reference = Context(static, VerletIntegrator(0.001 * unit.femtosecond), platform, properties)
    reference.setPeriodicBoxVectors(*static.getDefaultPeriodicBoxVectors())
    reference.setPositions(pdb.positions)
    static_energy, static_forces = measure(reference)
    del reference

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(0.5)
    context = Context(live, VerletIntegrator(0.001 * unit.femtosecond), platform, properties)
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(pdb.positions)
    switcher.set_tau(context, live, tau)
    dynamic_energy, dynamic_forces = measure(context)

    assert dynamic_energy == pytest.approx(static_energy, abs=1e-6, rel=0)
    assert np.abs(dynamic_forces - static_forces).max() < 1e-6

    # ...and switching away and back must land on the same Hamiltonian, not on a compounded one:
    # every set_tau restores the unscaled parameters before scaling.
    switcher.set_tau(context, live, 0.4)
    switcher.set_tau(context, live, tau)
    again_energy, again_forces = measure(context)
    assert again_energy == pytest.approx(static_energy, abs=1e-6, rel=0)
    assert np.abs(again_forces - static_forces).max() < 1e-6


@pytest.mark.gpu
@pytest.mark.slow
def test_omega_excluded_torsions_are_left_alone_by_the_dynamic_switcher(ais_project):
    """The same omega bonds REST2 leaves unscaled, checked on the switched force parameters."""
    from openmm import PeriodicTorsionForce, XmlSerializer
    from openmm.app import PDBFile

    from openmm import unit

    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize((ais_project / "built.xml").read_text())
    solute = set(_solute_indices(ais_project))
    omega = {frozenset((int(a), int(b))) for a, b in _excluded_bonds(ais_project)}

    def torsions(system):
        force = next(system.getForce(i) for i in range(system.getNumForces())
                     if isinstance(system.getForce(i), PeriodicTorsionForce))
        return [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]

    unscaled = torsions(base)
    switched = torsions(scaling.TauSwitcher(base, solute, omega).prepared_system(0.5))

    excluded_seen = scaled_seen = 0
    for original, now in zip(unscaled, switched):
        i, j, k, l = original[:4]
        # Stripped of units so the comparison is between numbers, not Quantities.
        k_before = original[6].value_in_unit(unit.kilojoule_per_mole)
        k_after = now[6].value_in_unit(unit.kilojoule_per_mole)
        in_solute = all(a in solute for a in (i, j, k, l))
        if in_solute and frozenset((int(j), int(k))) in omega:
            assert k_after == k_before, f"omega torsion {i}-{j}-{k}-{l} was scaled"
            excluded_seen += 1
        elif in_solute:
            assert k_after == pytest.approx(k_before * 0.25), \
                f"solute torsion {i}-{j}-{k}-{l} was not scaled by s"
            scaled_seen += 1
        else:
            assert k_after == k_before, "an environment torsion was scaled"
    assert excluded_seen and scaled_seen, (excluded_seen, scaled_seen)


@pytest.mark.gpu
@pytest.mark.slow
def test_frozen_coordinate_work_telescopes_to_the_endpoint_energy_difference(ais_project):
    """With coordinates held fixed, the summed increments must be exactly the endpoint difference.

    delta_W_j = U(tau_{j+1}, x) - U(tau_j, x) with x unchanged, so every intermediate term cancels
    and the total is U(tau_end, x) - U(tau_start, x). Anything else means the increments are not
    the quantity the record calls work.
    """
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize((ais_project / "built.xml").read_text())
    pdb = PDBFile(str(ais_project / "built.pdb"))
    solute = _solute_indices(ais_project)
    omega = _excluded_bonds(ais_project)

    # The schedule is computed, not read from a generated file: `path_definition.yaml` belonged
    # to the retired route, and `switching_schedule` is the one implementation of the arithmetic.
    from md_tools.openmm.ais import switching_schedule

    schedule = switching_schedule(tau_start=0.5, tau_end=0.0, switching_steps=40,
                                  parameter_update_interval_steps=1,
                                  observation_interval_steps=2, timestep_fs=2.0)
    taus = schedule["taus"]

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(taus[0])
    context = Context(live, VerletIntegrator(0.001 * unit.femtosecond),
                      Platform.getPlatformByName("CUDA"), {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(pdb.positions)

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    start_energy = energy()
    cumulative = 0.0
    for index in range(len(taus) - 1):
        before = energy()                        # no propagation: x is frozen
        switcher.set_tau(context, live, taus[index + 1])
        cumulative += energy() - before
    end_energy = energy()

    assert cumulative == pytest.approx(end_energy - start_energy, abs=1e-6, rel=0)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_constant_tau_diagnostic_path_does_zero_work(ais_project):
    """A path that never changes the Hamiltonian does no work, whatever the coordinates do.

    The public forward configuration refuses tau_start == tau_end, and this is why that refusal is
    a configuration rule rather than a physical impossibility: the diagnostic is meaningful and it
    is what pins the work convention to the parameter change rather than to the propagation.
    """
    from openmm import (Context, LangevinMiddleIntegrator, Platform, XmlSerializer, unit)
    from openmm.app import PDBFile

    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize((ais_project / "built.xml").read_text())
    pdb = PDBFile(str(ais_project / "built.pdb"))
    solute = _solute_indices(ais_project)
    omega = _excluded_bonds(ais_project)

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(0.5)
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                          2.0 * unit.femtoseconds)
    integrator.setRandomNumberSeed(20260827)
    context = Context(live, integrator, Platform.getPlatformByName("CUDA"),
                      {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 4242)

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    cumulative = 0.0
    for _ in range(20):
        before = energy()
        switcher.set_tau(context, live, 0.5)     # the same tau, every time
        cumulative += energy() - before
        integrator.step(2)                       # and the coordinates DO move
    assert cumulative == pytest.approx(0.0, abs=1e-6)


# --- a tiny CUDA run ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ais_run(ais_project):
    """The common chain, a fixed-tau cMD source at tau = 0.5, and two AIS paths. On CUDA."""
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






