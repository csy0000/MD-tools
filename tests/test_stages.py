"""The arithmetic and scheduling the generated scripts depend on, without running dynamics.

These are the parts of the scientific contract that can be checked exactly: seeds, the exchange
criterion, the scaling convention and the GPU grouping. Everything here imports the same module
files that are copied into a generated project.
"""
from __future__ import annotations

import time

import pytest

from .conftest import template_module

stages = template_module("md_stages")
from md_tools import rest2 as scaling

#: The exchange mathematics moved out of the scaler to the ladder that uses it -- one
#: implementation, verified bit-identical to the copy that was removed. These tests
#: followed it rather than being deleted with the duplicate.
exchange = template_module("replica_engine")
rules = template_module("exchange_rules")


# --- seeds -----------------------------------------------------------------

def test_every_replica_gets_a_distinct_integrator_velocity_and_barostat_seed():
    """One seed shared across a ladder correlates the rungs, so it samples less than it looks."""
    seeds = {(replica, purpose): stages.derive_seed(20260101, "REST2", replica, purpose)
             for replica in range(8)
             for purpose in ("integrator", "velocities", "barostat")}
    assert len(set(seeds.values())) == len(seeds), "seeds collide"
    assert all(1 <= value < 2 ** 31 - 1 for value in seeds.values()), "outside OpenMM's range"


def test_a_seed_is_never_zero_because_openmm_reads_zero_as_choose_randomly():
    values = [stages.derive_seed(base, "cMD", "integrator") for base in range(500)]
    assert 0 not in values


def test_the_same_base_and_purpose_always_give_the_same_seed():
    assert (stages.derive_seed(7, "REST2", 3, "integrator")
            == stages.derive_seed(7, "REST2", 3, "integrator"))


# --- positional restraint ---------------------------------------------------

def _system(*, periodic):
    """A two-particle System that is periodic or not, decided the way a real one is: by its forces."""
    from openmm import NonbondedForce, System, unit

    system = System()
    for _ in range(2):
        system.addParticle(12.0 * unit.amu)
    nonbonded = NonbondedForce()
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 0.5)
    if periodic:
        nonbonded.setNonbondedMethod(NonbondedForce.PME)       # as an explicit solvent build does
        system.setDefaultPeriodicBoxVectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
    else:
        nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)   # as implicit GBn2 does
    system.addForce(nonbonded)
    assert system.usesPeriodicBoundaryConditions() is periodic
    return system


def _reference_positions():
    from openmm import unit

    return unit.Quantity([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], unit.nanometer)


class _ContextHolder:
    """`set_restraint` takes a Simulation; a bare Context is all these tests build."""

    def __init__(self, context):
        self.context = context


def test_an_explicit_periodic_restraint_uses_the_minimum_image():
    """With a box, an atom that crosses a face must be pulled to the nearest image of r0."""
    system = _system(periodic=True)
    force = stages.add_positional_restraint(system, _reference_positions(), [0])

    assert "periodicdistance" in force.getEnergyFunction()
    assert force.usesPeriodicBoundaryConditions() is True
    assert system.usesPeriodicBoundaryConditions() is True


def test_an_implicit_nonperiodic_restraint_is_plain_cartesian():
    """There is no box, so there is no minimum image to take."""
    system = _system(periodic=False)
    force = stages.add_positional_restraint(system, _reference_positions(), [0])

    expression = force.getEnergyFunction().replace(" ", "")
    assert "periodicdistance" not in expression, expression
    assert "(x-x0)^2" in expression and "(y-y0)^2" in expression and "(z-z0)^2" in expression


def test_restraining_an_implicit_system_does_not_make_it_periodic():
    """The failure this guards against is a lie the System tells about itself.

    OpenMM answers `System.usesPeriodicBoundaryConditions()` by asking its Forces, and a
    `periodicdistance` restraint answers yes. Adding one to an implicit GBn2 system therefore
    flipped the System to "periodic" while it still had no meaningful box -- and anything that
    later reads that property is told something untrue about the physics being sampled.
    """
    system = _system(periodic=False)
    stages.add_positional_restraint(system, _reference_positions(), [0, 1])
    assert system.usesPeriodicBoundaryConditions() is False


def test_both_forms_give_the_same_energy_away_from_a_box_face():
    """Why this survived: the two expressions agree everywhere the solute normally sits.

    1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2 and the displacement is 0.2 nm, so
    U = 0.5 * 418.4 * 0.2^2 = 8.368 kJ/mol under either expression.
    """
    from openmm import Context, Platform, VerletIntegrator, unit

    energies = {}
    for periodic in (True, False):
        system = _system(periodic=periodic)
        stages.add_positional_restraint(system, _reference_positions(), [0])
        context = Context(system, VerletIntegrator(0.001 * unit.picoseconds),
                          Platform.getPlatformByName("Reference"))
        if periodic:
            context.setPeriodicBoxVectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
        context.setPositions([(0.2, 0.0, 0.0), (1.0, 0.0, 0.0)])
        holder = _ContextHolder(context)

        # Difference against k = 0 so this measures the RESTRAINT, not the nonbonded force that
        # also lives in the System. Taking the raw total here would silently fold that in.
        stages.set_restraint(holder, 0.0)
        baseline = context.getState(
            getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        stages.set_restraint(holder, 1.0)
        total = context.getState(
            getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies[periodic] = total - baseline
        del context

    assert energies[True] == pytest.approx(8.368, abs=1e-6)
    assert energies[False] == pytest.approx(energies[True], abs=1e-9)


def test_the_restraint_strength_and_selected_atoms_are_unchanged_by_the_fix():
    """Only the distance expression moved. Strength, parameter name and atom selection did not."""
    for periodic in (True, False):
        system = _system(periodic=periodic)
        force = stages.add_positional_restraint(system, _reference_positions(), [1])
        assert force.getNumParticles() == 1
        index, parameters = force.getParticleParameters(0)
        assert index == 1, "the restraint must be on the atom it was given"
        assert [round(v, 6) for v in parameters] == [0.1, 0.0, 0.0]
        assert force.getGlobalParameterName(0) == stages.RESTRAINT_PARAMETER
        assert force.getGlobalParameterDefaultValue(0) == 0.0, "restraints start off"


# --- durations -------------------------------------------------------------

def test_a_duration_that_is_not_a_whole_number_of_steps_is_refused():
    """A segment rounded shorter by one step drifts the exchange schedule while looking healthy."""
    assert stages.steps_for(10.0, 2.0) == 5000
    with pytest.raises(SystemExit) as error:
        stages.steps_for(0.003, 2.0)
    assert "whole number" in str(error.value)


# --- REST2 exchange criterion ----------------------------------------------

def test_the_pv_terms_cancel_at_a_common_beta_and_pressure():
    """Replicas differ by Hamiltonian, not by thermostat, so beta is shared and pV drops out.

    The swap moves each configuration WITH its box, so the same two volumes appear on both sides
    of the criterion. Checked against the plain energy-only form.
    """
    beta = 1.0 / (0.008314462618 * 300.0)
    pressure, volume_i, volume_j = 1.0, 27.0, 31.5          # deliberately different volumes
    energy_i, energy_j, energy_ij, energy_ji = -2500.0, -2480.0, -2495.0, -2470.0

    with_pv = exchange.exchange_log_acceptance(
        exchange.reduced_potential(energy_i, beta, pressure_bar=pressure, volume_nm3=volume_i),
        exchange.reduced_potential(energy_j, beta, pressure_bar=pressure, volume_nm3=volume_j),
        exchange.reduced_potential(energy_ij, beta, pressure_bar=pressure, volume_nm3=volume_j),
        exchange.reduced_potential(energy_ji, beta, pressure_bar=pressure, volume_nm3=volume_i))
    energy_only = beta * (energy_i + energy_j - energy_ij - energy_ji)

    assert with_pv == pytest.approx(energy_only, rel=1e-12, abs=1e-12)


def test_the_criterion_is_the_documented_expression():
    beta = 1.0 / (0.008314462618 * 300.0)
    values = (-10.0, -20.0, -5.0, -8.0)
    reduced = [exchange.reduced_potential(v, beta) for v in values]
    assert exchange.exchange_log_acceptance(*reduced) == pytest.approx(
        beta * (values[0] + values[1] - values[2] - values[3]))


def test_a_two_replica_ladder_offers_its_pair_in_the_phase_the_runner_falls_back_to():
    """Alternating regardless would exchange on every other round only, pressure_bar=and a run resumed on an
    odd index would keep landing on the empty phase."""
    assert rules.alternating_pairs(2, 0) == [(0, 1)]
    assert rules.alternating_pairs(2, 1) == []
    for attempt_index in range(6):
        phase = attempt_index % 2
        pairs = rules.alternating_pairs(2, phase) or rules.alternating_pairs(2, (phase + 1) % 2)
        assert pairs == [(0, 1)], attempt_index


# --- the scaling convention ------------------------------------------------

def test_the_ladder_scales_solute_solute_by_the_square_and_solute_environment_linearly():
    """(1-tau)^2 on solute-solute and (1-tau) on solute-environment, applied to charges,
    epsilons and exceptions."""
    from openmm import NonbondedForce, System

    tau = 0.2
    solute_solute, solute_environment = scaling.scaling_for_tau(tau)
    assert solute_solute == pytest.approx(0.64)
    assert solute_environment == pytest.approx(0.8)

    system = System()
    force = NonbondedForce()
    for _ in range(3):
        system.addParticle(12.0)
    force.addParticle(0.5, 0.3, 0.8)          # 0: solute
    force.addParticle(-0.4, 0.3, 0.6)         # 1: solute
    force.addParticle(0.2, 0.3, 0.4)          # 2: environment
    force.addException(0, 1, 0.1, 0.3, 0.5)   # solute-solute      -> s
    force.addException(0, 2, 0.2, 0.3, 0.7)   # solute-environment -> sqrt(s)
    system.addForce(force)

    scaled = scaling.build_scaled_system(system, [0, 1], tau)
    out = scaled.getForce(0)

    charge, _, epsilon = out.getParticleParameters(0)
    assert charge.value_in_unit(charge.unit) == pytest.approx(0.5 * solute_environment)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.8 * solute_solute)

    charge, _, epsilon = out.getParticleParameters(2)
    assert charge.value_in_unit(charge.unit) == pytest.approx(0.2), "environment must not scale"
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.4)

    _, _, product, _, epsilon = out.getExceptionParameters(0)
    assert product.value_in_unit(product.unit) == pytest.approx(0.1 * solute_solute)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.5 * solute_solute)

    _, _, product, _, epsilon = out.getExceptionParameters(1)
    assert product.value_in_unit(product.unit) == pytest.approx(0.2 * solute_environment)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.7 * solute_environment)


def test_the_cold_rung_is_the_unmodified_system():
    assert scaling.scaling_for_tau(0.0) == (1.0, 1.0)


# --- GPU placement -----------------------------------------------------------
#
# `device_groups` and `propagate_segment` are retired with the thread-based launchers that used
# them: one process opened N Contexts and hand-scheduled them across devices. The ladder now runs
# one MPI rank per state, so concurrency is the process model rather than a schedule to prove, and
# the only remaining decision is which device each rank takes. That is
# `replica_engine.select_device_for_rank`, asserted below.


# --- REST2 equilibration scheduling ------------------------------------------

def test_the_same_device_policy_serves_equilibration_and_production():
    """One policy, not two that can drift apart.

    Equilibration and production are the same ladder under `replica_executor`, so they cannot disagree
    about placement by construction. This asserts the single decision point still exists and that
    nothing has grown a second one.
    """
    from .conftest import REPO_ROOT

    root = REPO_ROOT / "src" / "md_tools"
    choosers = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")
                      if "def select_device_for_rank(" in path.read_text(encoding="utf-8")
                      or "def device_groups(" in path.read_text(encoding="utf-8"))
    assert choosers == ["openmm/templates/replica_engine.py"], choosers


def test_one_device_is_chosen_per_rank_and_ranks_never_share():
    """`device_groups` placed replicas on devices for the retired thread-based launcher. The
    ladder now runs one MPI rank per state, so placement is per rank -- and two ranks landing on
    one device is the failure that used to halve throughput silently.
    """
    from .conftest import template_module

    engine = template_module("replica_engine")
    devices = [0, 1, 2, 3]
    # The engine returns the device index as OpenMM wants it -- a string for `DeviceIndex`.
    chosen = [engine.select_device_for_rank(rank, 4, devices)[0] for rank in range(4)]
    assert sorted(int(device) for device in chosen) == devices, \
        f"ranks did not get distinct devices: {chosen}"

    # Fewer devices than ranks is a real configuration, and it must wrap deterministically
    # rather than leaving a rank with nothing.
    two = [engine.select_device_for_rank(rank, 4, [0, 1])[0] for rank in range(4)]
    assert all(device is not None for device in two), two
    assert {int(device) for device in two} == {0, 1}, two

    # No visible device is not a placement decision to guess at.
    assert engine.select_device_for_rank(0, 4, [])[0] is None


def test_a_failing_replica_is_never_reported_as_a_finished_run():
    """A ladder with a dead rung must not be declared complete and then produced from.

    `propagate_segment` awaited every thread and let exceptions out. The ladder is MPI now, so the
    same guarantee lives in the driver: any failure records the run as failed or interrupted and
    is then re-raised, and only the path that did NOT raise reaches `_finish`.
    """
    from .conftest import REPO_ROOT

    driver = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
              / "replica_driver.py").read_text(encoding="utf-8")
    handler = driver[driver.index("except BaseException as failure:"):]
    handler = handler[:handler.index("finally:")]
    assert '"interrupted" if interrupted else "failed"' in handler, \
        "a failed run no longer records that it failed"
    assert handler.rstrip().endswith("raise"), \
        "the driver swallows the failure instead of re-raising it"
    # `_finish` -- which writes the completion manifest -- is reachable only after the try block.
    assert "return self._finish(" in driver[driver.index("finally:"):]


# --- the nonpolar term is worth knowing the size of --------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_the_ace_nonpolar_term_is_a_real_energy_difference(tmp_path):
    """Pins the measurement the docstring quotes, so a silent default flip would fail here.

    The ~16 kJ/mol between construction routes is NOT the radii and not an opaque branch: it is
    the ACE surface-area term. With it enabled, ParmEd and pure OpenMM agree to ~0.002 kJ/mol.
    """
    from openmm import Context, Platform, VerletIntegrator, app, unit
    import parmed as pmd

    from .conftest import ALA_PDB

    # The implicit builder is called DIRECTLY, because this test needs the AmberTools
    # intermediates (prmtop/rst7) that `build-top` does not retain -- it keeps built.pdb and
    # built.xml, and the preparation directory the retired route archived is gone. The builder
    # itself is unchanged and maintained.
    from md_tools.build.top import _sys_document, resolve_build_config
    from md_tools.openmm.builders import Log, _build_implicit, _legacy_cfg
    from md_tools.openmm.system_config import resolve_sys_config

    work = tmp_path
    staging = work / "_work"
    staging.mkdir()
    config = work / "gb.config"
    config.write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    resolved = resolve_build_config(config)
    resolved.pop("_stated", None)
    cfg = _legacy_cfg(resolve_sys_config(_sys_document(resolved)))
    record = _build_implicit(ALA_PDB, cfg, staging, route="peptide",
                             log=Log(staging / "build.log", echo=False))

    structure = pmd.load_file(str(staging / "system.prmtop"),
                              xyz=str(staging / "system.rst7"))
    pmd.tools.changeRadii(structure, "mbondi3").execute()
    platform = Platform.getPlatformByName("CUDA")

    def total(system):
        context = Context(system, VerletIntegrator(0.001), platform)
        context.setPositions(structure.positions)
        value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        del context
        return value

    common = dict(nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                  implicitSolvent=app.GBn2, removeCMMotion=True)
    without = total(structure.createSystem(**common, useSASA=False))
    with_sasa = total(structure.createSystem(**common, useSASA=True))
    native = total(app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml").createSystem(
        app.PDBFile(str(record["topology_pdb"])).topology,
        nonbondedMethod=app.NoCutoff, constraints=app.HBonds, removeCMMotion=True))

    assert with_sasa - without == pytest.approx(16.05, abs=0.5), \
        "the nonpolar term should be ~16 kJ/mol on ACE-ALA-NME"
    # Once the nonpolar choice matches, ParmEd and pure OpenMM are the same Hamiltonian.
    assert native == pytest.approx(with_sasa, abs=0.05), (native, with_sasa)
