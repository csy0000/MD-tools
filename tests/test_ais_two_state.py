"""The two-state AIS Hamiltonian: V(lambda) = (1 - lambda) V0 + lambda V1, and its refusals.

Reference platform, double precision, so every identity here is checked at the arithmetic's own
resolution rather than at a GPU's. CUDA evidence for the same identity lives in the gpu lane.

V1 is built two ways. Once as the REST2 System at tau = 0.5 -- the switch the previous AIS made,
now expressed as a pair of files -- and once by hand-editing parameters, so nothing here depends on
AIS knowing what REST2 is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openmm import (Context, MonteCarloBarostat, NonbondedForce, Platform,
                    PeriodicTorsionForce, VerletIntegrator, XmlSerializer, unit)
from openmm.app import PME, ForceField, HBonds, Modeller, NoCutoff, PDBFile

from md_tools.ais.two_state import (LAMBDA_PARAMETER, TWO_STATE_SCHEMA, EndStateError,
                                    SchemaError, TwoStateHamiltonian, end_state_potentials,
                                    pair_plan, require_compatible_schema)

ALA = Path(__file__).parent / "data" / "ALA.pdb"
KJ = unit.kilojoule_per_mole


def _clone(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


@pytest.fixture(scope="module")
def vacuum():
    pdb = PDBFile(str(ALA))
    system = ForceField("amber14-all.xml").createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                                        constraints=HBonds)
    return pdb.topology, pdb.positions, system


@pytest.fixture(scope="module")
def solvated():
    pdb = PDBFile(str(ALA))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    model = Modeller(pdb.topology, pdb.positions)
    model.addSolvent(forcefield, padding=0.8 * unit.nanometer)
    system = forcefield.createSystem(model.topology, nonbondedMethod=PME,
                                     nonbondedCutoff=0.8 * unit.nanometer, constraints=HBonds)
    return model.topology, model.positions, system


def _hand_edited(system, solute=range(22)):
    """V1 by hand: solute charges x0.7, epsilons x0.5, torsion barriers x0.6."""
    edited = _clone(system)
    for force in edited.getForces():
        if isinstance(force, NonbondedForce):
            for i in solute:
                q, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, q * 0.7, sigma, epsilon * 0.5)
        elif isinstance(force, PeriodicTorsionForce):
            for t in range(force.getNumTorsions()):
                *atoms, periodicity, phase, k = force.getTorsionParameters(t)
                force.setTorsionParameters(t, *atoms, periodicity, phase, k * 0.6)
    return edited


def _rest2(topology, system, tau=0.5):
    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    return build_scaled_system(system, solute_atom_indices(topology), tau)


def _context(system, topology, positions):
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    if topology.getPeriodicBoxVectors() is not None:
        context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    context.setPositions(positions)
    return context


def _energy_and_forces(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(KJ),
            state.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer))


@pytest.mark.parametrize("which", ["vacuum", "solvated"])
@pytest.mark.parametrize("edit", ["rest2", "hand"])
def test_the_mixed_potential_is_exactly_linear_between_the_end_states(which, edit, request):
    topology, positions, v0 = request.getfixturevalue(which)
    v1 = _rest2(topology, v0) if edit == "rest2" else _hand_edited(v0)
    hamiltonian = TwoStateHamiltonian(v0, v1)

    e0, f0 = _energy_and_forces(_context(v0, topology, positions))
    e1, f1 = _energy_and_forces(_context(v1, topology, positions))
    mixed = _context(hamiltonian.system, topology, positions)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        hamiltonian.set_lambda(mixed, lam)
        energy, forces = _energy_and_forces(mixed)
        scale = max(abs(e0), abs(e1))
        assert energy == pytest.approx((1 - lam) * e0 + lam * e1, rel=1e-10, abs=1e-8 * scale)
        assert np.abs(forces - ((1 - lam) * f0 + lam * f1)).max() < 1e-6
        assert hamiltonian.difference(mixed) == pytest.approx(e1 - e0, rel=1e-10, abs=1e-6)


def test_the_work_from_the_derivative_is_the_finite_difference_at_frozen_coordinates(solvated):
    topology, positions, v0 = solvated
    hamiltonian = TwoStateHamiltonian(v0, _rest2(topology, v0))
    context = _context(hamiltonian.system, topology, positions)
    for before, after in ((0.0, 0.01), (0.3, 0.37), (0.99, 1.0)):
        hamiltonian.set_lambda(context, before)
        u_before = _energy_and_forces(context)[0]
        derivative_work = (after - before) * hamiltonian.difference(context)
        hamiltonian.set_lambda(context, after)
        u_after = _energy_and_forces(context)[0]
        assert derivative_work == pytest.approx(u_after - u_before, rel=1e-9, abs=1e-6)


def test_work_telescopes_over_a_schedule_at_one_coordinate(vacuum):
    topology, positions, v0 = vacuum
    hamiltonian = TwoStateHamiltonian(v0, _hand_edited(v0))
    context = _context(hamiltonian.system, topology, positions)
    lambdas = np.linspace(0.0, 1.0, 11)
    total = sum((b - a) * hamiltonian.difference(context) for a, b in zip(lambdas, lambdas[1:]))
    e0 = _energy_and_forces(_context(v0, topology, positions))[0]
    hamiltonian.set_lambda(context, 1.0)
    assert total == pytest.approx(_energy_and_forces(context)[0] - e0, rel=1e-10, abs=1e-8)


def test_the_observation_probe_returns_both_end_state_potentials(solvated):
    topology, positions, v0 = solvated
    v1 = _rest2(topology, v0)
    hamiltonian = TwoStateHamiltonian(v0, v1)
    context = _context(hamiltonian.system, topology, positions)
    hamiltonian.set_lambda(context, 0.4)
    observed = hamiltonian.observe(context, 0.4, precision="double")
    assert observed["potential_v0_kj_mol"] == pytest.approx(
        _energy_and_forces(_context(v0, topology, positions))[0], rel=1e-10, abs=1e-6)
    assert observed["potential_v1_kj_mol"] == pytest.approx(
        _energy_and_forces(_context(v1, topology, positions))[0], rel=1e-10, abs=1e-6)
    assert observed["potential_direct_kj_mol"] == pytest.approx(
        0.6 * observed["potential_v0_kj_mol"] + 0.4 * observed["potential_v1_kj_mol"],
        rel=1e-12, abs=1e-8)


def test_only_the_differing_forces_are_mixed(vacuum):
    topology, _, v0 = vacuum
    plan = pair_plan(v0, _rest2(topology, v0))
    assert set(plan["mixed_force_classes"]) <= {"NonbondedForce", "PeriodicTorsionForce"}
    shared = [type(v0.getForce(i)).__name__ for i in plan["shared_forces"]]
    assert "HarmonicBondForce" in shared and "HarmonicAngleForce" in shared


def test_end_state_potentials_invert_the_mixture():
    v0, v1 = end_state_potentials(direct=-70.0, difference=40.0, lam=0.25)
    assert (v0, v1) == (-80.0, -40.0)
    assert 0.75 * v0 + 0.25 * v1 == -70.0


# -- refusals -----------------------------------------------------------------------------------

def _refusal(v0, v1) -> str:
    with pytest.raises(EndStateError) as caught:
        pair_plan(v0, v1)
    return str(caught.value)


def test_identical_end_states_are_refused(vacuum):
    assert "same Hamiltonian" in _refusal(vacuum[2], _clone(vacuum[2]))


def test_a_different_particle_count_is_refused(vacuum, solvated):
    assert "particle(s)" in _refusal(vacuum[2], solvated[2])


def test_a_different_mass_is_refused(vacuum):
    v1 = _hand_edited(vacuum[2])
    v1.setParticleMass(3, v1.getParticleMass(3) * 2)
    assert "mass" in _refusal(vacuum[2], v1)


def test_a_different_constraint_is_refused(vacuum):
    v1 = _hand_edited(vacuum[2])
    a, b, distance = v1.getConstraintParameters(0)
    v1.setConstraintParameters(0, a, b, distance * 1.01)
    assert "constraint" in _refusal(vacuum[2], v1)


def test_a_barostat_in_either_end_state_is_refused(solvated):
    v0 = solvated[2]
    v1 = _hand_edited(v0)
    v1.addForce(MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin))
    v0_with = _clone(v0)
    v0_with.addForce(MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin))
    assert "barostat" in _refusal(v0, v1)
    assert "barostat" in _refusal(v0_with, _hand_edited(v0_with))


def test_a_different_force_layout_is_refused(vacuum):
    v1 = _hand_edited(vacuum[2])
    v1.removeForce(0)
    assert "force lists differ" in _refusal(vacuum[2], v1)


def test_a_different_long_range_treatment_is_refused(solvated):
    v1 = _hand_edited(solvated[2])
    for force in v1.getForces():
        if isinstance(force, NonbondedForce):
            force.setCutoffDistance(1.0 * unit.nanometer)
    assert "cutoff" in _refusal(solvated[2], v1)


def test_a_different_dispersion_correction_is_refused(solvated):
    v1 = _hand_edited(solvated[2])
    for force in v1.getForces():
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(not force.getUseDispersionCorrection())
    assert "dispersion_correction" in _refusal(solvated[2], v1)


def test_every_problem_is_named_at_once(vacuum):
    v1 = _hand_edited(vacuum[2])
    v1.setParticleMass(3, v1.getParticleMass(3) * 2)
    v1.removeForce(0)
    message = _refusal(vacuum[2], v1)
    assert "mass" in message and "force lists differ" in message


# -- schema -------------------------------------------------------------------------------------

def test_this_builds_schema_is_accepted():
    require_compatible_schema({"ais_schema": {"name": TWO_STATE_SCHEMA["name"],
                                              "version": TWO_STATE_SCHEMA["version"]}},
                              what="a checkpoint")


@pytest.mark.parametrize("recorded", [
    {"decomposition_schema": {"name": "rest2-lambda-basis", "version": 2}},
    {"decomposition_schema": {"name": "rest2-tau-quadratic-basis", "version": 1}},
    {"work_measurement": "work"},
])
def test_a_single_topology_record_is_refused_by_name(recorded):
    with pytest.raises(SchemaError) as caught:
        require_compatible_schema(recorded, what="a checkpoint")
    assert "two-state-linear" in str(caught.value)


def test_the_lambda_parameter_is_the_only_global_the_switch_adds(vacuum):
    topology, _, v0 = vacuum
    hamiltonian = TwoStateHamiltonian(v0, _hand_edited(v0))
    switch = hamiltonian.system.getForce(hamiltonian.switch_index)
    assert [switch.getGlobalParameterName(i)
            for i in range(switch.getNumGlobalParameters())] == [LAMBDA_PARAMETER]
    assert switch.getNumEnergyParameterDerivatives() == 1
