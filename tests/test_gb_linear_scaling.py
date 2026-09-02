"""The generalised-Born contribution scales linearly in tau, and that changes the identity.

GB is the solute's coupling to a continuum standing in for solvent, so the whole contribution is a
solute-environment interaction and follows `1 - tau`, not the solute-solute `(1 - tau)^2`.

These are energy tests, not parameter tests. The ordinary nonbonded force and the GB force are put
in separate OpenMM force groups and evaluated independently, so each ratio is measured on the
energy that force actually produces -- including a GB term with no charge dependence, which charge
scaling alone would leave untouched.

PLATFORM_POLICY_EXEMPTION: single-point energies on a small synthetic implicit system, evaluated on
the Reference platform. No dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
openmm = pytest.importorskip("openmm")

# The internals of the scaling are what these tests exercise, so they name the
# module rather than the package facade -- `md_tools.rest2` exports the public
# API, and a test of `_scale_cmap` is not using the public API.
from md_tools.rest2 import scaler as scaling                                    # noqa: E402
from openmm import unit                                            # noqa: E402

TAUS = [0.0, 0.25, 0.5]
NONBONDED_GROUP, GB_GROUP = 0, 1


def _implicit_system(n_atoms=4):
    """A whole-system implicit model: ordinary nonbonded plus a GB force whose terms include one
    with NO charge dependence -- the term that exposes charge-only scaling as insufficient."""
    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setForceGroup(NONBONDED_GROUP)
    for index in range(n_atoms):
        nonbonded.addParticle(0.4 if index % 2 == 0 else -0.4, 0.3, 0.5)
    system.addForce(nonbonded)

    gb = openmm.CustomGBForce()
    gb.setForceGroup(GB_GROUP)
    gb.addPerParticleParameter("q")
    # a charge-driven pair term, and a per-particle term that does not depend on charge at all
    gb.addComputedValue("I", "0.0", openmm.CustomGBForce.ParticlePairNoExclusions)
    gb.addEnergyTerm("-0.5*q^2", openmm.CustomGBForce.SingleParticle)
    gb.addEnergyTerm("2.5", openmm.CustomGBForce.SingleParticle)          # no charge dependence
    gb.addEnergyTerm("-q1*q2/0.5", openmm.CustomGBForce.ParticlePairNoExclusions)
    for index in range(n_atoms):
        gb.addParticle([0.4 if index % 2 == 0 else -0.4])
    system.addForce(gb)
    return system


def _positions(n_atoms=4):
    return np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0],
                     [0.0, 0.7, 0.0], [0.0, 0.0, 0.8]])[:n_atoms] * unit.nanometer


def _group_energies(system, positions):
    """Ordinary-nonbonded and GB energies, measured separately."""
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    out = {}
    for name, group in (("nonbonded", NONBONDED_GROUP), ("gb", GB_GROUP)):
        state = context.getState(getEnergy=True, groups={group})
        out[name] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return out


# --- the identity ------------------------------------------------------------------------------

def test_the_identity_is_version_two_and_names_the_gb_rule():
    assert scaling.REST2_IMPLEMENTATION["version"] == 2
    assert scaling.REST2_IMPLEMENTATION["generalized_born_scale"] == "1-tau"
    assert scaling.REST2_IMPLEMENTATION["solute_solute_nonbonded_scale"] == "(1-tau)^2"


def test_version_one_is_refused_for_continuation():
    """A version bump here says the energy function changed, so continuing across one would join
    samples from two different ensembles."""
    with pytest.raises(ValueError, match="different Hamiltonian"):
        scaling.require_compatible_implementation(
            {"name": "rest2-no-bond-angle-omega", "version": 1}, what="the parent run")


def test_version_two_is_accepted_and_absent_identity_is_not_invented():
    scaling.require_compatible_implementation(dict(scaling.REST2_IMPLEMENTATION))
    scaling.require_compatible_implementation({})        # nothing recorded -> nothing to refuse
    scaling.require_compatible_implementation(None)


def test_an_unknown_identity_is_refused_without_a_guess():
    with pytest.raises(ValueError, match="not one this build implements"):
        scaling.require_compatible_implementation(
            {"name": "something-else", "version": 7})


# --- cloned states -------------------------------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_a_cloned_state_scales_gb_linearly_and_nonbonded_quadratically(tau):
    system = _implicit_system()
    positions = _positions()
    reference = _group_energies(system, positions)

    scaled = scaling.build_scaled_system(system, range(system.getNumParticles()), tau)
    measured = _group_energies(scaled, positions)

    solute_solute, solute_environment = scaling.scaling_for_tau(tau)
    assert measured["gb"] == pytest.approx(reference["gb"] * solute_environment, rel=1e-9), (
        f"tau={tau}: the whole GB contribution must follow (1-tau)")
    assert measured["nonbonded"] == pytest.approx(
        reference["nonbonded"] * solute_solute, rel=1e-9), (
        f"tau={tau}: a wholly-solute nonbonded energy follows (1-tau)^2")


def test_the_two_groups_scale_by_different_factors_at_the_same_tau():
    """The point of the correction: at tau = 0.5 GB halves while nonbonded quarters."""
    system = _implicit_system()
    positions = _positions()
    reference = _group_energies(system, positions)
    measured = _group_energies(
        scaling.build_scaled_system(system, range(4), 0.5), positions)

    assert measured["gb"] / reference["gb"] == pytest.approx(0.5, rel=1e-9)
    assert measured["nonbonded"] / reference["nonbonded"] == pytest.approx(0.25, rel=1e-9)


def test_the_charge_independent_gb_term_is_scaled_too():
    """A term with no charge dependence is exactly what charge scaling would miss."""
    system = openmm.System()
    system.addParticle(12.0)
    gb = openmm.CustomGBForce()
    gb.setForceGroup(GB_GROUP)
    gb.addPerParticleParameter("q")
    gb.addComputedValue("I", "0.0", openmm.CustomGBForce.ParticlePairNoExclusions)
    gb.addEnergyTerm("7.0", openmm.CustomGBForce.SingleParticle)     # constant: charge-free
    gb.addParticle([0.0])
    system.addForce(gb)

    positions = np.zeros((1, 3)) * unit.nanometer
    before = _group_energies(system, positions)["gb"]
    after = _group_energies(scaling.build_scaled_system(system, [0], 0.25), positions)["gb"]
    assert before == pytest.approx(7.0)
    assert after == pytest.approx(7.0 * 0.75, rel=1e-9), (
        "a charge-free GB term must still follow (1-tau)")


# --- live switching ---------------------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_live_switching_reaches_the_same_energies_as_a_cloned_state(tau):
    system = _implicit_system()
    positions = _positions()
    cloned = _group_energies(scaling.build_scaled_system(system, range(4), tau), positions)

    switcher = scaling.TauSwitcher(system, range(4))
    prepared = switcher.prepared_system(0.0)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(prepared, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    switcher.set_tau(context, prepared, tau)

    live = {}
    for name, group in (("nonbonded", NONBONDED_GROUP), ("gb", GB_GROUP)):
        live[name] = context.getState(getEnergy=True, groups={group}).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
    del context, integrator

    assert live["gb"] == pytest.approx(cloned["gb"], rel=1e-9)
    assert live["nonbonded"] == pytest.approx(cloned["nonbonded"], rel=1e-9)


def test_repeated_switching_does_not_compound_the_gb_factor():
    """Every switch restores from the unscaled reference first, so 0.5 -> 0.25 -> 0.5 lands back
    on the same energy rather than on 0.5 applied twice."""
    system = _implicit_system()
    positions = _positions()
    reference = _group_energies(system, positions)

    switcher = scaling.TauSwitcher(system, range(4))
    prepared = switcher.prepared_system(0.0)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(prepared, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)

    seen = []
    for tau in (0.5, 0.25, 0.5):
        switcher.set_tau(context, prepared, tau)
        seen.append(context.getState(getEnergy=True, groups={GB_GROUP}).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole))
    del context, integrator

    assert seen[0] == pytest.approx(reference["gb"] * 0.5, rel=1e-9)
    assert seen[1] == pytest.approx(reference["gb"] * 0.75, rel=1e-9)
    assert seen[2] == pytest.approx(seen[0], rel=1e-9), "the factor was compounded"


def test_tau_zero_leaves_the_energies_untouched():
    system = _implicit_system()
    positions = _positions()
    reference = _group_energies(system, positions)
    measured = _group_energies(scaling.build_scaled_system(system, range(4), 0.0), positions)
    assert measured["gb"] == pytest.approx(reference["gb"], rel=1e-12)
    assert measured["nonbonded"] == pytest.approx(reference["nonbonded"], rel=1e-12)
