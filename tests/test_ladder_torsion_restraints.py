"""A ladder may carry torsion restraints, and an identical bias cancels from the exchange.

hpREST2 runs a REST2 ladder with restraints on every rung. The property that makes that sound is
exact rather than statistical, so it is tested by arithmetic rather than by sampling:

    u_i(x) and u_j(x) both contain W(x), so log alpha = (u_i(x_j) + u_j(x_i)) - (u_i(x_i) +
    u_j(x_j)) is the same number with the restraint as without it.

That holds only because the restraint is added AFTER the REST2 scaling and is identical on every
rung. A restraint scaled by tau, or one that differed between rungs, would enter the acceptance
probability and bias the ladder towards whichever rung restrains least -- silently, with every
output still looking well formed. The test below computes both differences and requires equality.

PLATFORM_POLICY_EXEMPTION: single-point energies of alanine dipeptide on the CPU. What is under
test is an identity between energies, which is platform-independent; the GPU evidence that a
restrained ladder RUNS is `test_hprest2_gpu_evidence.py`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("ladder-restraints")]

#: Two torsions of alanine dipeptide, by index, as `configs/md/cv.yaml` names them.
PHI = (4, 6, 8, 14)
PSI = (6, 8, 14, 16)

#: What `UmbrellaRestraint.record()` produces, which is what the preflight hands the builder.
RESTRAINTS = [
    {"cv": "phi_ALA", "form": "harmonic", "centre_deg": -60.0,
     "force_constant_kj_mol_rad2": 500.0, "atom_indices": list(PHI)},
    {"cv": "psi_ALA", "form": "flat_bottom", "centre_deg": 140.0, "half_width_deg": 30.0,
     "force_constant_kj_mol_rad2": 250.0, "atom_indices": list(PSI)},
]

TAUS = (0.0, 0.25, 0.5)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """ALA in implicit solvent: the System every rung below is scaled from."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    work = tmp_path_factory.mktemp("ladder-restraints")
    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", "sys.config"],
        cwd=work, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return work


def _rungs(built, restraints):
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.remd.protocol import build_rung_systems

    system = XmlSerializer.deserialize((built / "built.xml").read_text(encoding="utf-8"))
    pdb = PDBFile(str(built / "built.pdb"))
    solute = list(range(system.getNumParticles()))          # implicit: the solute IS the system
    systems, audit = build_rung_systems(system, solute, TAUS, restraints=restraints)
    return systems, audit, pdb


def _restraint_forces(system):
    from openmm import CustomTorsionForce

    return [system.getForce(i) for i in range(system.getNumForces())
            if isinstance(system.getForce(i), CustomTorsionForce)
            and "torsion_restraint_k" in system.getForce(i).getEnergyFunction()]


def _energy(system, positions):
    from openmm import Context, VerletIntegrator, Platform, unit

    context = Context(system, VerletIntegrator(0.001 * unit.picoseconds),
                      Platform.getPlatformByName("Reference"))
    try:
        context.setPositions(positions)
        return float(context.getState(getEnergy=True).getPotentialEnergy()
                     .value_in_unit(unit.kilojoule_per_mole))
    finally:
        del context


def test_every_rung_carries_the_same_restraint_force(built):
    """Identical on every rung is the whole basis of the cancellation; asserted, not assumed."""
    from openmm import XmlSerializer

    systems, audit, _pdb = _rungs(built, RESTRAINTS)
    serialised = set()
    for system in systems:
        forces = _restraint_forces(system)
        assert len(forces) == 2, "one force per form: harmonic and flat-bottom"
        serialised.add("|".join(XmlSerializer.serialize(force) for force in forces))
    assert len(serialised) == 1, "the rungs carry different restraint forces"
    assert audit["ladder_restraints"]["scaled_by_rest2"] is False
    assert audit["ladder_restraints"]["restraints"] == RESTRAINTS


def test_the_restraint_is_biased_from_the_first_step(built):
    """The global parameter's DEFAULT is 1.0, so a Context is restrained without a runtime call."""
    systems, _audit, _pdb = _rungs(built, RESTRAINTS)
    for system in systems:
        for force in _restraint_forces(system):
            names = [force.getGlobalParameterName(i)
                     for i in range(force.getNumGlobalParameters())]
            index = names.index("torsion_restraint_k")
            assert force.getGlobalParameterDefaultValue(index) == 1.0


def test_a_ladder_without_restraints_is_built_exactly_as_before(built):
    """The default path must be untouched: no force added, no audit key."""
    from openmm import XmlSerializer

    plain, audit, _pdb = _rungs(built, ())
    assert "ladder_restraints" not in audit
    for system in plain:
        assert _restraint_forces(system) == []
    restrained, _audit, _pdb = _rungs(built, RESTRAINTS)
    for one, other in zip(plain, restrained):
        assert len(_restraint_forces(other)) == 2
        assert one.getNumForces() + 2 == other.getNumForces(), \
            "the restrained rung differs from the plain one by the two restraint forces alone"
        assert XmlSerializer.serialize(one) != XmlSerializer.serialize(other)


def test_the_bias_cancels_from_the_exchange_criterion(built):
    """THE test. log alpha is identical with the restraint and without it, to machine precision.

    Computed the way `reduced_potential_of` does: each configuration evaluated in each rung's own
    System. Two configurations are needed, because a swap's acceptance depends on the cross terms;
    the second is the first displaced, which is enough to make every term differ.
    """
    import numpy as np
    from openmm import unit

    plain, _audit, pdb = _rungs(built, ())
    restrained, _audit2, _pdb2 = _rungs(built, RESTRAINTS)

    x_i = np.array(pdb.positions.value_in_unit(unit.nanometer))
    generator = np.random.default_rng(20260913)
    x_j = x_i + 0.02 * generator.standard_normal(x_i.shape)
    positions = [x_i * unit.nanometer, x_j * unit.nanometer]

    def log_alpha(systems):
        u = [[_energy(systems[rung], positions[which]) for which in (0, 1)] for rung in (0, 1)]
        # (u_i(x_j) + u_j(x_i)) - (u_i(x_i) + u_j(x_j)), in energy units; beta is common.
        return (u[0][1] + u[1][0]) - (u[0][0] + u[1][1])

    without = log_alpha(plain)
    with_bias = log_alpha(restrained)
    assert abs(with_bias - without) < 1e-6, (
        f"the restraint changed the exchange criterion by {with_bias - without:.6e} kJ/mol; it "
        f"must cancel exactly")
    # And the restraint is doing something: the energies themselves must differ.
    assert abs(_energy(restrained[0], positions[0]) - _energy(plain[0], positions[0])) > 1.0


@pytest.mark.parametrize("protocol, reservoir, refused",
                         [("rREST2", True, True), ("REST2", False, False)])
def test_restraints_with_a_reservoir_are_refused(tmp_path, protocol, reservoir, refused):
    """An rREST2 reservoir sample carries no bias, so it cannot enter a restrained ladder.

    The refresh replaces the top rung's configuration with one drawn from a distribution generated
    WITHOUT the restraint. The rung then samples neither the biased nor the unbiased ensemble, and
    every output still looks well formed -- so the combination is refused rather than documented.
    """
    from md_tools.build.md import ConfigError, resolve_md_config

    # The unrestrained case must be REST2, not rREST2 with the reservoir off: rREST2 IS the
    # reservoir variant and refuses `enabled: false` for its own reason, which would make this
    # parameter pass for nothing to do with restraints.
    config = tmp_path / f"{protocol}.config"
    config.write_text(
        f"protocol: {protocol}\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 1}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 100, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 0}\n"
        "reporting: {crd_printout_solute: 50, info_printout: 50, checkpoint_printout: 50}\n"
        "collective_variables: {file: cv.yaml, interval_steps: 50}\n"
        "umbrella: {file: umbrella.yaml}\n"
        f"reservoir: {{enabled: {'true' if reservoir else 'false'}, path: reservoir}}\n"
        "rest2: {number_of_replicas: 2, tau_max: 0.5, exchange_interval_steps: 50, "
        "number_of_exchanges: 4}\n", encoding="utf-8")
    if refused:
        with pytest.raises(ConfigError, match="drawn from a distribution generated WITHOUT"):
            resolve_md_config(config)
    else:
        resolve_md_config(config)


def test_a_restraint_that_differs_between_rungs_does_not_cancel(built):
    """The counter-example, so the test above cannot pass for a trivial reason."""
    import numpy as np
    from openmm import unit

    weaker = [{**RESTRAINTS[0], "force_constant_kj_mol_rad2": 50.0}]
    systems, _audit, pdb = _rungs(built, RESTRAINTS)
    rung_one, _audit2, _pdb = _rungs(built, weaker)

    x_i = np.array(pdb.positions.value_in_unit(unit.nanometer))
    generator = np.random.default_rng(4)
    x_j = x_i + 0.02 * generator.standard_normal(x_i.shape)
    positions = [x_i * unit.nanometer, x_j * unit.nanometer]
    mixed = [systems[0], rung_one[1]]                    # rung 0 and rung 1 restrained differently
    plain, _a, _p = _rungs(built, ())

    def log_alpha(pair):
        u = [[_energy(pair[rung], positions[which]) for which in (0, 1)] for rung in (0, 1)]
        return (u[0][1] + u[1][0]) - (u[0][0] + u[1][1])

    assert abs(log_alpha(mixed) - log_alpha([plain[0], plain[1]])) > 1e-3, \
        "a bias that differs between rungs must NOT cancel; if it does, the test above proves " \
        "nothing"
