"""Setting tau by global parameter must be the SAME Hamiltonian as re-uploading it.

`TauSwitcher.set_amplitude` rewrote every solute particle, exception and torsion and called
`updateParametersInContext` on each affected force. Measured on this machine, 22-atom alanine,
CUDA at mixed precision, ff14SB + GBn2:

    set_tau, re-upload            2.93 ms
    one energy evaluation         0.076 ms

so a tau change cost 38x an evaluation and was 95% of an AIS `work`-mode update. The physics was
the rest. `CustomGBForce` already avoided this through `rest2_scale_gb`; the nonbonded terms and
the eligible torsions now carry `rest2_a` and `rest2_a2` the same way, and a switch is three
`setParameter` calls at 0.0025 ms -- 1172x cheaper.

THE SPEED IS NOT THE CLAIM UNDER TEST HERE. The claim is that it is the same Hamiltonian, and
these tests are the proof: the two routes are compared across tau, across the three probe
amplitudes, and across perturbed configurations, on every force field and solvent combination
MD-tools supports.

WHERE IT IS NOT USED, AND WHY. Two cases decline the fast path and both were found by measuring,
not by reading:

  * A periodic NonbondedForce using the long-range dispersion correction. OpenMM computes that
    tail term from the particles' stored epsilon and does not apply parameter offsets to it, so
    the offset route would keep its tau = 0 correction while the potential moved -- 2.6 kJ/mol at
    tau = 0.5 on a small solvated peptide, growing with tau, and surviving double precision. Such
    a System keeps the re-upload.
  * CMAPTorsionForce, which has no global to attach: OpenMM has no CustomCMAPTorsionForce. A
    System with CMAP uses globals for everything else and re-uploads that one force.

PLATFORM_POLICY_EXEMPTION: these compare two routes to one potential on whatever platform is
available, and skip when there is no GPU. No sampling is performed and no result is produced.
"""
from __future__ import annotations

import numpy as np
import pytest

from openmm import (Context, LangevinMiddleIntegrator, NonbondedForce, Platform,
                    XmlSerializer, unit)
from openmm.app import ForceField, HBonds, Modeller, PDBFile, PME

from md_tools.rest2.scaler import (REST2_A_PARAMETER, REST2_A2_PARAMETER, TauSwitcher,
                                   global_switching_refusal, switches_by_global_parameter)

from .conftest import ALA_PDB

#: Ten taus across the usable range plus the three amplitudes a basis probe visits. `a = 1` is
#: the unscaled Hamiltonian and `a = 0` a fully decoupled solute -- the two ends where an error in
#: the offset arrangement would be most visible, and both are included deliberately.
TAUS = tuple(float(t) for t in np.linspace(0.0, 0.9, 10))
PROBE_AMPLITUDES = (0.0, 0.5, 1.0)

#: Same order as the re-upload path's own reconstruction tolerance. The measured worst case on
#: every combination below is ~5e-05 kJ/mol, so this is not a threshold tuned to let it pass.
TOLERANCE_KJ_MOL = 1e-3


def _platform():
    try:
        return Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    except Exception:                                        # noqa: BLE001 - absence, not failure
        return Platform.getPlatformByName("Reference"), {}


def _implicit(force_field):
    pdb = PDBFile(str(ALA_PDB))
    ff = ForceField(force_field, "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, constraints=HBonds)
    return system, pdb.topology, pdb.positions, tuple(range(system.getNumParticles()))


def _explicit(force_field):
    pdb = PDBFile(str(ALA_PDB))
    ff = ForceField(force_field, "amber14/tip3p.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ff, model="tip3p", padding=0.75 * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=PME,
                             nonbondedCutoff=0.75 * unit.nanometer, constraints=HBonds)
    # The peptide is the solute; the water is environment. This is what brings the `a` branch --
    # solute-environment pairs -- into existence at all.
    return system, modeller.topology, modeller.positions, tuple(range(pdb.topology.getNumAtoms()))


CASES = {
    "ff14SB+GBn2": lambda: _implicit("amber14/protein.ff14SB.xml"),
    "ff19SB+GBn2": lambda: _implicit("amber19/protein.ff19SB.xml"),   # CMAPTorsionForce
    "ff14SB+TIP3P": lambda: _explicit("amber14/protein.ff14SB.xml"),  # PME, solute-environment
    "ff19SB+TIP3P": lambda: _explicit("amber19/protein.ff19SB.xml"),  # both at once
}


def _energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_two_routes_give_the_same_potential(case):
    """Across tau, across probe amplitudes, and across perturbed configurations.

    Perturbed configurations matter: at one geometry a scaling error in a term that happens to be
    near zero there is invisible. Displacing every atom makes each scaled group contribute.
    """
    system, topology, positions, solute = CASES[case]()
    platform, properties = _platform()
    switcher = TauSwitcher(system, solute)

    def context_for(global_parameters):
        prepared = switcher.prepared_system(0.0, global_parameters=global_parameters)
        context = Context(prepared, LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
            platform, properties)
        context.setPositions(positions)
        return prepared, context

    uploaded_system, uploaded = context_for(False)
    global_system, by_global = context_for(True)

    rng = np.random.default_rng(20260909)
    reference = np.array(positions.value_in_unit(unit.nanometer))
    configurations = 8 if "TIP3P" in case else 20

    worst = 0.0
    for _ in range(configurations):
        displaced = reference + rng.normal(0.0, 0.01, reference.shape)
        uploaded.setPositions(displaced)
        by_global.setPositions(displaced)
        for tau in TAUS:
            switcher.set_tau(uploaded, uploaded_system, tau)
            switcher.set_tau(by_global, global_system, tau)
            worst = max(worst, abs(_energy(uploaded) - _energy(by_global)))
        for amplitude in PROBE_AMPLITUDES:
            switcher.set_amplitude(uploaded, uploaded_system, amplitude)
            switcher.set_amplitude(by_global, global_system, amplitude)
            worst = max(worst, abs(_energy(uploaded) - _energy(by_global)))

    assert worst < TOLERANCE_KJ_MOL, (
        f"{case}: the two routes disagree by up to {worst:.3e} kJ/mol over "
        f"{configurations} configurations x ({len(TAUS)} tau + {len(PROBE_AMPLITUDES)} "
        f"amplitudes)")


def test_an_implicit_system_actually_takes_the_fast_path():
    """Otherwise the equivalence test above would pass by comparing the re-upload with itself."""
    system, _, _, solute = CASES["ff14SB+GBn2"]()
    prepared = TauSwitcher(system, solute).prepared_system(0.0)
    assert switches_by_global_parameter(prepared)
    assert global_switching_refusal(system) is None


def test_an_explicit_system_with_the_dispersion_correction_declines_and_says_why():
    """The refusal is the finding, so it is asserted rather than merely allowed.

    Turning the correction off would make the two routes agree, and would be changing the physics
    to suit the implementation. So the System keeps the re-upload and the reason is reportable.
    """
    system, _, _, solute = CASES["ff14SB+TIP3P"]()
    assert any(isinstance(system.getForce(i), NonbondedForce)
               and system.getForce(i).getUseDispersionCorrection()
               for i in range(system.getNumForces())), "this case must have the correction on"

    refusal = global_switching_refusal(system)
    assert refusal is not None
    assert "dispersion" in refusal

    switcher = TauSwitcher(system, solute)
    prepared = switcher.prepared_system(0.0)
    assert not switches_by_global_parameter(prepared), "it must fall back, not reparameterise"
    assert "dispersion" in switcher.switching_note(prepared)


def test_turning_the_dispersion_correction_off_lets_the_same_system_take_the_fast_path():
    """Pins the reason: it is that one setting, not 'explicit solvent' or 'PME' in general.

    PME itself is fine -- `addParticleParameterOffset` is supported on it and the reciprocal-space
    sum follows the offset charges. Only the analytic tail correction does not.
    """
    system, _, _, solute = CASES["ff14SB+TIP3P"]()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(False)
    assert global_switching_refusal(system) is None
    assert switches_by_global_parameter(TauSwitcher(system, solute).prepared_system(0.0))


def test_a_prepared_system_starts_at_the_tau_it_was_asked_for():
    """A Context built and never switched must be at its own tau, not at the amplitude 1 the
    offsets were attached at. Otherwise a path that never calls set_tau integrates the wrong
    Hamiltonian and nothing says so."""
    system, topology, positions, solute = CASES["ff14SB+GBn2"]()
    platform, properties = _platform()
    switcher = TauSwitcher(system, solute)

    for tau in (0.0, 0.25, 0.5):
        by_global = switcher.prepared_system(tau, global_parameters=True)
        uploaded = switcher.prepared_system(tau, global_parameters=False)
        contexts = []
        for prepared in (by_global, uploaded):
            context = Context(prepared, LangevinMiddleIntegrator(
                300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
                platform, properties)
            context.setPositions(positions)
            contexts.append(context)
        assert abs(_energy(contexts[0]) - _energy(contexts[1])) < TOLERANCE_KJ_MOL, (
            f"an unswitched Context is not at tau = {tau}")


def test_the_globals_survive_serialisation():
    """The System is what a resumed run deserialises, so the offsets must be in the XML.

    A System that lost its offsets on a round trip would deserialise with every solute charge and
    epsilon at ZERO -- a silently decoupled solute that still runs.
    """
    system, _, _, solute = CASES["ff14SB+GBn2"]()
    prepared = TauSwitcher(system, solute).prepared_system(0.3)
    revived = XmlSerializer.deserialize(XmlSerializer.serialize(prepared))
    assert switches_by_global_parameter(revived)
    for index in range(revived.getNumForces()):
        force = revived.getForce(index)
        if isinstance(force, NonbondedForce):
            assert force.getNumParticleParameterOffsets() > 0
            names = {force.getGlobalParameterName(i)
                     for i in range(force.getNumGlobalParameters())}
            assert {REST2_A_PARAMETER, REST2_A2_PARAMETER} <= names
