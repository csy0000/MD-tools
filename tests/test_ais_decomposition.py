"""The three-group tau basis is exact, and each group holds the physics it claims.

`U(tau, x) = U_unscaled(x) + a U_linear(x) + a^2 U_quadratic(x)`, with `a = 1 - tau`.

The claim is an IDENTITY, not a fit, so these tests are not "the residual is small": they build
systems in which the answer is known term by term, and check that each term lands in the group the
convention names. A decomposition that reproduced the total while putting the generalised-Born
energy in the quadratic group would pass a residual test and be useless for reweighting, because
the tau dependence it predicts away from the sampled point would be wrong.

Every energy here is measured from a real OpenMM Context through the real `TauSwitcher`, never
from parameter inspection: what a reweighting consumes is energies, and a parameter that is scaled
correctly but never reaches the compiled kernel is exactly the failure this catches.

PLATFORM_POLICY_EXEMPTION: single-point energies on small synthetic systems, evaluated on the
Reference platform. No dynamics, no trajectory, no scientific result -- the object under test is
the algebra of the decomposition, which is platform-independent by construction. The same identity
is verified on real CUDA in the GPU lane.
"""
from __future__ import annotations

import math

import pytest

openmm = pytest.importorskip("openmm")

from md_tools.ais.decomposition import (BASIS_PROBE_AMPLITUDES,  # noqa: E402
                                        DECOMPOSITION_SCHEMA, ComponentProbe, Components,
                                        DecompositionError, quadratic_through,
                                        require_compatible_schema)
from md_tools.rest2.scaler import TauSwitcher  # noqa: E402
from openmm import unit  # noqa: E402

#: Reference-platform double precision. The identity is exact in exact arithmetic, so the only
#: slack needed is rounding on sums of order 1e2 kJ/mol.
TOLERANCE = 1e-7

SOLUTE = (0, 1, 2, 3)
ENVIRONMENT = (4, 5)
ALL_ATOMS = SOLUTE + ENVIRONMENT


def _positions(n):
    return [openmm.Vec3(0.3 * i, 0.1 * (i % 3), 0.05 * (i % 2)) for i in range(n)]


def _mixed_system(*, with_cmap=True, with_torsions=True, with_exceptions=True):
    """Solute and environment, every scaled force class, plus deliberately unscaled ones.

    Charges and epsilons differ between solute and environment so that a term landing in the wrong
    group changes the number rather than merely the label.
    """
    system = openmm.System()
    for _ in ALL_ATOMS:
        system.addParticle(12.0)

    nonbonded = openmm.NonbondedForce()
    for index in ALL_ATOMS:
        charge = 0.5 if index in SOLUTE else -0.31
        nonbonded.addParticle(charge * (1 if index % 2 == 0 else -1), 0.30, 0.65)
    if with_exceptions:
        # A solute-solute 1-4 (quadratic) and a solute-environment one (linear).
        nonbonded.addException(0, 3, 0.21, 0.28, 0.44)
        nonbonded.addException(2, 4, -0.17, 0.31, 0.39)
    system.addForce(nonbonded)

    # Deliberately unscaled by the convention: they must land wholly in the unscaled group.
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 250000.0)
    bonds.addBond(4, 5, 0.15, 250000.0)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.9, 400.0)
    system.addForce(angles)

    if with_torsions:
        torsions = openmm.PeriodicTorsionForce()
        torsions.addTorsion(0, 1, 2, 3, 2, 0.0, 12.0)        # wholly solute, eligible
        torsions.addTorsion(1, 2, 3, 4, 2, 0.0, 9.0)         # reaches the environment: untouched
        system.addForce(torsions)

    if with_cmap:
        cmap = openmm.CMAPTorsionForce()
        size = 4
        cmap.addMap(size, [0.7 * (i + 1) for i in range(size * size)])
        cmap.addTorsion(0, 0, 1, 2, 3, 0, 1, 2, 3)           # wholly solute
        system.addForce(cmap)
    return system


def _implicit_system():
    """A whole-system implicit model: every atom is solute, so the ONLY linear term is GB.

    `_scale_customgb` refuses a partial enhanced region -- a generalised-Born energy is not
    separable per atom -- so isolating GB means making the solute the whole system, which is
    exactly the configuration supported implicit REST2 runs in.
    """
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    nonbonded = openmm.NonbondedForce()
    for index in range(4):
        nonbonded.addParticle(0.4 if index % 2 == 0 else -0.4, 0.3, 0.5)
    system.addForce(nonbonded)

    gb = openmm.CustomGBForce()
    gb.addPerParticleParameter("q")
    gb.addComputedValue("I", "0.0", openmm.CustomGBForce.ParticlePairNoExclusions)
    gb.addEnergyTerm("-0.5*q^2", openmm.CustomGBForce.SingleParticle)
    gb.addEnergyTerm("2.5", openmm.CustomGBForce.SingleParticle)      # no charge dependence
    gb.addEnergyTerm("-q1*q2/0.5", openmm.CustomGBForce.ParticlePairNoExclusions)
    for index in range(4):
        gb.addParticle([0.4 if index % 2 == 0 else -0.4])
    system.addForce(gb)
    return system


class _Probe:
    """A switcher, a prepared System and a live Context, ready to be measured."""

    def __init__(self, base, solute, excluded_bonds=(), tau0=0.5):
        self.switcher = TauSwitcher(base, solute, excluded_bonds)
        self.system = self.switcher.prepared_system(tau0)
        integrator = openmm.VerletIntegrator(0.001)
        platform = openmm.Platform.getPlatformByName("Reference")
        self.context = openmm.Context(self.system, integrator, platform)
        self.context.setPositions(_positions(self.system.getNumParticles()))
        self.probe = ComponentProbe(self.switcher, self.system)

    def energy(self, tau):
        self.switcher.set_tau(self.context, self.system, tau)
        return self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    def components(self, tau=0.5):
        return self.probe.measure(self.context, restore_tau=tau)


# --- the identity ------------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
def test_the_reconstruction_reproduces_the_measured_potential(tau):
    """At every tau, from ONE measurement of the components. Endpoint and interior alike."""
    probe = _Probe(_mixed_system(), SOLUTE)
    components = probe.components()
    assert abs(components.total_at(tau) - probe.energy(tau)) < TOLERANCE, (
        f"tau={tau}: {components}")


def test_the_potential_is_quadratic_in_the_amplitude_and_not_merely_close():
    """Five amplitudes, a quadratic through three of them, exact at the other two.

    A cubic or higher dependence would fit the three probe nodes and miss the others. This is what
    distinguishes "the model is right" from "the fit interpolates its own nodes".
    """
    probe = _Probe(_mixed_system(), SOLUTE)
    components = probe.components()
    for tau in (0.37, 0.62):                              # neither is a probe node
        assert abs(components.total_at(tau) - probe.energy(tau)) < TOLERANCE, tau


def test_the_unscaled_group_is_what_survives_at_zero_amplitude():
    probe = _Probe(_mixed_system(), SOLUTE)
    components = probe.components()
    # tau = 1 is a = 0. It is not a legal protocol coordinate, so it is reached through the
    # amplitude probe -- which is the reason that API exists.
    probe.switcher.set_amplitude(probe.context, probe.system, 0.0)
    at_zero = probe.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert abs(components.unscaled - at_zero) < TOLERANCE


# --- each group holds the physics it claims ------------------------------------------------------

def test_bonds_and_angles_alone_are_wholly_unscaled():
    """No nonbonded, no torsions: everything must land in the unscaled group."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 250000.0)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.9, 400.0)
    system.addForce(angles)

    components = _Probe(system, (0, 1, 2, 3)).components()
    assert abs(components.linear) < TOLERANCE
    assert abs(components.quadratic) < TOLERANCE
    assert abs(components.unscaled) > 1.0, "the test system carries no energy to classify"


def test_an_all_environment_system_has_no_scaled_components():
    """Nothing is solute, so nothing scales, whatever forces are present."""
    components = _Probe(_mixed_system(), solute=()).components()
    assert abs(components.linear) < TOLERANCE
    assert abs(components.quadratic) < TOLERANCE


def test_generalized_born_appears_in_the_linear_basis():
    """The whole GB energy is solute-environment and carries `a`, not `a^2`.

    Every atom is solute here, so the nonbonded force contributes only to the quadratic group and
    the linear group can hold nothing but GB. Measuring the linear component against the GB energy
    itself is therefore a direct check rather than a difference of totals.
    """
    system = _implicit_system()
    solute = tuple(range(system.getNumParticles()))
    with_gb = _Probe(system, solute).components()

    # The same system with the GB force removed: its linear component must vanish.
    without = _implicit_system()
    for index in reversed(range(without.getNumForces())):
        if isinstance(without.getForce(index), openmm.CustomGBForce):
            without.removeForce(index)
    without_gb = _Probe(without, solute).components()

    assert abs(without_gb.linear) < TOLERANCE, "a pure NonbondedForce solute has no linear term"
    assert abs(with_gb.linear) > 1.0, "the GB energy did not reach the linear group"
    assert abs(with_gb.quadratic - without_gb.quadratic) < TOLERANCE, (
        "GB leaked into the quadratic group, which is the v1 Hamiltonian, not this one")


def test_solute_solute_nonbonded_and_one_four_land_in_the_quadratic_group():
    """Remove the 1-4 exceptions and only the quadratic group may move."""
    with_exceptions = _Probe(_mixed_system(with_exceptions=True), SOLUTE).components()
    without = _Probe(_mixed_system(with_exceptions=False), SOLUTE).components()
    assert abs(with_exceptions.quadratic - without.quadratic) > 1e-3, (
        "the solute-solute 1-4 exception made no difference to the quadratic group")
    # The solute-environment exception moves the linear group; the unscaled group must not move.
    assert abs(with_exceptions.unscaled - without.unscaled) < TOLERANCE


def test_eligible_solute_torsions_are_quadratic_and_reach_only_that_group():
    with_torsions = _Probe(_mixed_system(with_torsions=True), SOLUTE).components()
    without = _Probe(_mixed_system(with_torsions=False), SOLUTE).components()
    assert abs(with_torsions.quadratic - without.quadratic) > 1e-3
    assert abs(with_torsions.linear - without.linear) < TOLERANCE


def test_solute_cmap_is_quadratic():
    with_cmap = _Probe(_mixed_system(with_cmap=True), SOLUTE).components()
    without = _Probe(_mixed_system(with_cmap=False), SOLUTE).components()
    assert abs(with_cmap.quadratic - without.quadratic) > 1e-3
    assert abs(with_cmap.linear - without.linear) < TOLERANCE


def test_an_excluded_omega_torsion_moves_from_the_quadratic_group_to_the_unscaled_one():
    """The omega exclusion is not cosmetic: it changes which group the torsion's energy is in.

    Excluding the central bond 1-2 leaves the wholly-solute torsion unscaled, so its energy must
    leave the quadratic group and appear in the unscaled one -- with the TOTAL at tau = 0
    unchanged, because at a = 1 every group is weighted 1.
    """
    scaled = _Probe(_mixed_system(), SOLUTE).components()
    excluded = _Probe(_mixed_system(), SOLUTE, excluded_bonds=((1, 2),)).components()

    moved = scaled.quadratic - excluded.quadratic
    assert moved > 1e-3, "excluding the omega bond did not remove the torsion from the quadratic"
    assert abs((excluded.unscaled - scaled.unscaled) - moved) < TOLERANCE, (
        "the excluded torsion's energy did not reappear in the unscaled group")
    assert abs(excluded.total_at(0.0) - scaled.total_at(0.0)) < TOLERANCE, (
        "the physical (tau = 0) Hamiltonian must be the same either way")


# --- work ---------------------------------------------------------------------------------------

def test_the_component_works_sum_to_the_directly_measured_work():
    """Forward switching. The total is measured from the Hamiltonian, the parts from the fit."""
    probe = _Probe(_mixed_system(), SOLUTE)
    tau_before, tau_after = 0.5, 0.4
    components = probe.components(tau=tau_before)
    measured = probe.energy(tau_after) - probe.energy(tau_before)
    work = components.work_between(tau_before, tau_after)
    assert abs(work.total - measured) < TOLERANCE, (work, measured)


def test_the_unscaled_component_work_is_zero_and_is_written_as_such():
    probe = _Probe(_mixed_system(), SOLUTE)
    work = probe.components().work_between(0.5, 0.1)
    assert work.unscaled == 0.0
    assert abs(work.linear) > 1e-6 and abs(work.quadratic) > 1e-6, (
        "a switch that moved nothing cannot demonstrate that the unscaled part stayed put")


def test_reverse_switching_negates_every_component():
    """The sign convention is `U(after) - U(before)`, and it holds component by component."""
    probe = _Probe(_mixed_system(), SOLUTE)
    components = probe.components()
    forward = components.work_between(0.5, 0.0)
    reverse = components.work_between(0.0, 0.5)
    for name in ("unscaled", "linear", "quadratic"):
        assert abs(getattr(forward, name) + getattr(reverse, name)) < TOLERANCE, name


def test_frozen_coordinate_work_telescopes_to_the_endpoint_difference():
    """Many small steps at frozen coordinates sum to one big one, in every component.

    This is the property the incremental columns exist for: a reader summing `delta_work_*` over a
    path must land where `U(tau_end) - U(tau_start)` is, and must do so in each group separately.
    """
    probe = _Probe(_mixed_system(), SOLUTE)
    components = probe.components()
    taus = [0.5 - 0.05 * i for i in range(11)]            # 0.5 -> 0.0 in ten steps
    accumulated = {"unscaled": 0.0, "linear": 0.0, "quadratic": 0.0}
    for before, after in zip(taus, taus[1:]):
        step = components.work_between(before, after)
        for name in accumulated:
            accumulated[name] += getattr(step, name)

    whole = components.work_between(taus[0], taus[-1])
    for name, value in accumulated.items():
        assert abs(value - getattr(whole, name)) < TOLERANCE, name
    measured = probe.energy(taus[-1]) - probe.energy(taus[0])
    assert abs(sum(accumulated.values()) - measured) < TOLERANCE


# --- the fit and the schema ----------------------------------------------------------------------

def test_the_probe_restores_the_tau_it_was_given():
    """A leaked probe amplitude would integrate the path under the wrong Hamiltonian, silently."""
    probe = _Probe(_mixed_system(), SOLUTE)
    at_start = probe.energy(0.5)
    probe.probe.measure(probe.context, restore_tau=0.5)
    after = probe.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert abs(after - at_start) < TOLERANCE


def test_the_probe_restores_tau_even_when_an_evaluation_raises():
    probe = _Probe(_mixed_system(), SOLUTE)
    at_start = probe.energy(0.5)

    calls = {"n": 0}
    real = probe.switcher.set_amplitude

    def exploding(context, system, amplitude):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(context, system, amplitude)

    probe.switcher.set_amplitude = exploding
    with pytest.raises(RuntimeError, match="boom"):
        probe.probe.measure(probe.context, restore_tau=0.5)
    probe.switcher.set_amplitude = real
    after = probe.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    assert abs(after - at_start) < TOLERANCE, "the failed probe left the Context at a probe amplitude"


def test_the_probe_counts_its_evaluations_and_its_cost():
    probe = _Probe(_mixed_system(), SOLUTE)
    probe.components()
    probe.components()
    assert probe.probe.evaluations == 2 * len(BASIS_PROBE_AMPLITUDES) == 6
    assert probe.probe.seconds > 0.0


def test_three_repeated_nodes_are_refused_rather_than_producing_a_number():
    with pytest.raises(DecompositionError, match="distinct"):
        quadratic_through((0.5, 0.5, 1.0), (1.0, 1.0, 2.0))


def test_the_fit_recovers_a_known_quadratic_exactly():
    coefficients = (-1.25e5, 3.5e4, -7.75e3)
    values = [coefficients[0] + coefficients[1] * a + coefficients[2] * a * a
              for a in BASIS_PROBE_AMPLITUDES]
    got = quadratic_through(BASIS_PROBE_AMPLITUDES, values)
    for expected, actual in zip(coefficients, got):
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-6)


def test_a_record_written_before_the_basis_existed_is_refused_by_name():
    with pytest.raises(DecompositionError, match="no component-decomposition schema"):
        require_compatible_schema({"cumulative_work_kj_mol": 1.0}, what="an old checkpoint")


def test_a_record_from_another_basis_version_is_refused():
    with pytest.raises(DecompositionError, match="not summable"):
        require_compatible_schema(
            {"decomposition_schema": {"name": DECOMPOSITION_SCHEMA["name"], "version": 99}})


def test_the_current_schema_is_accepted():
    require_compatible_schema({"decomposition_schema": {
        "name": DECOMPOSITION_SCHEMA["name"], "version": DECOMPOSITION_SCHEMA["version"]}})


def test_components_carry_the_contributions_and_the_basis_separately():
    components = Components(unscaled=1.0, linear=10.0, quadratic=100.0)
    linear, quadratic = components.contributions_at(0.25)
    assert linear == pytest.approx(0.75 * 10.0)
    assert quadratic == pytest.approx(0.75 * 0.75 * 100.0)
    assert components.total_at(0.25) == pytest.approx(1.0 + 7.5 + 56.25)
