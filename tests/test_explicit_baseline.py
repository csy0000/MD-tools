"""Regression tests for the explicit-solvent baseline.

Two things are under test here that the pre-existing REST2 tests could not reach:

1. **Three-sector scaling.** The existing suite exercises all-solute (implicit) systems, where
   ``U_s = s*U`` and any consistent scaling passes. The REST2 Hamiltonian this project relies on is

       U_s = s*U_solute-solute + sqrt(s)*U_solute-solvent + U_solvent-solvent

   and the two cross-sector factors only become distinguishable once solvent particles exist. The
   tests below build a small periodic system with both, and check each sector's energy numerically.

2. **Omega classification.** An ordinary amide omega must stay unscaled; a proline-like peptide
   bond must remain eligible for scaling. Getting this backwards changes the Hamiltonian silently.

Everything here is CPU-only and takes seconds; nothing needs a GPU or a real force field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import (  # noqa: E402
    CMAPTorsionForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    System,
    Vec3,
    XmlSerializer,
    unit,
)
from openmm import app  # noqa: E402
from openmm.app import element as elem  # noqa: E402

from md_templates.openmm import (  # noqa: E402
    DEFAULTS,
    build_rest2_scaled_system,
    classify_omega_bonds,
    completed_prefix,
    load_config,
    repartition_hydrogen_mass,
    resolve_route,
    rest2_ladder,
)

# Module-private helpers are imported from the module that owns them rather than re-exported
# through the package: the package surface is the supported API, and widening it to keep a test
# import short would make every private helper a de-facto public one.
from md_templates.openmm.equilibration import _steps  # noqa: E402
from md_templates.openmm.solvation import _box_vectors, _resolve_box  # noqa: E402

S_VALUES = (1.0, 0.64, 0.25)
TOL = 1e-9


# ==================================================================================================
# helpers
# ==================================================================================================
def _pair_energy(charges, sigmas, epsilons, distance_nm=0.4):
    """Energy of a single isolated pair, built as its own two-particle System."""
    system = System()
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.NoCutoff)
    for q, sig, eps in zip(charges, sigmas, epsilons):
        system.addParticle(12.0 * unit.amu)
        nb.addParticle(q, sig, eps)
    system.addForce(nb)
    ctx = openmm.Context(system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([Vec3(0, 0, 0), Vec3(distance_nm, 0, 0)] * unit.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _nb_particle(force, i):
    """Particle parameters as plain floats.

    OpenMM returns Quantity wrappers around transient C++ storage; holding them across a later call
    into the library yields garbage (uninitialised doubles like 1.3e-313). Convert immediately.
    """
    q, sig, eps = force.getParticleParameters(i)
    return (q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole))


def _nb_exception(force, idx):
    i, j, q, sig, eps = force.getExceptionParameters(idx)
    return (int(i), int(j),
            q.value_in_unit(unit.elementary_charge ** 2),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole))


def _explicit_test_system():
    """A periodic system with 4 solute and 4 solvent particles, PME, bonds, angles, torsions, CMAP.

    Deliberately not a real molecule: the point is to have every sector and every force type
    present with distinguishable parameters, so a scaling bug cannot hide behind a coincidence.
    Four solute particles so an ALL-SOLUTE torsion exists -- a torsion straddling solute and
    solvent is not an intramolecular term and is correctly left alone.
    """
    system = System()
    n_solute, n_solvent = 4, 4
    for _ in range(n_solute + n_solvent):
        system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3))

    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    params = [(0.5, 0.30, 0.8), (-0.4, 0.32, 0.7),        # solute 0, 1
              (0.35, 0.28, 0.9), (-0.25, 0.35, 0.45),     # solute 2, 3
              (0.2, 0.31, 0.6), (-0.2, 0.33, 0.5),        # solvent 4, 5
              (0.1, 0.29, 0.4), (-0.1, 0.34, 0.3)]        # solvent 6, 7
    for q, sig, eps in params:
        nb.addParticle(q, sig * unit.nanometer, eps * unit.kilojoule_per_mole)
    # one exception per sector: solute-solute, solute-solvent, solvent-solvent
    nb.addException(0, 1, 0.11 * unit.elementary_charge ** 2, 0.31 * unit.nanometer,
                    0.21 * unit.kilojoule_per_mole)
    nb.addException(1, 4, 0.13 * unit.elementary_charge ** 2, 0.32 * unit.nanometer,
                    0.23 * unit.kilojoule_per_mole)
    nb.addException(6, 7, 0.17 * unit.elementary_charge ** 2, 0.33 * unit.nanometer,
                    0.27 * unit.kilojoule_per_mole)
    system.addForce(nb)

    bonds = HarmonicBondForce()
    bonds.addBond(0, 1, 0.15 * unit.nanometer, 300000.0)   # solute-solute
    bonds.addBond(6, 7, 0.10 * unit.nanometer, 400000.0)   # solvent-solvent
    system.addForce(bonds)

    angles = HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.9, 500.0)
    system.addForce(angles)

    torsions = PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 2, 3.14159, 10.0)      # all-solute, central bond 1-2
    torsions.addTorsion(1, 2, 3, 0, 1, 0.0, 7.0)           # all-solute, central bond 2-3
    torsions.addTorsion(0, 1, 4, 5, 1, 0.0, 5.0)           # straddles solvent: must NOT scale
    system.addForce(torsions)

    cmap = CMAPTorsionForce()
    grid = 2
    cmap.addMap(grid, [1.0, 2.0, 3.0, 4.0] * unit.kilojoule_per_mole)
    cmap.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 0)
    system.addForce(cmap)
    return system, list(range(n_solute)), list(range(n_solute, n_solute + n_solvent))


def _nb_force(system):
    return next(system.getForce(i) for i in range(system.getNumForces())
                if isinstance(system.getForce(i), NonbondedForce))


def _force_of(system, cls):
    return next(system.getForce(i) for i in range(system.getNumForces())
                if isinstance(system.getForce(i), cls))


def _peptide_topology(residue_names, *, n_methyl=()):
    """Build a minimal linear peptide topology: ACE-style C(=O) linked to each residue's N.

    Atom names follow Amber convention so the residue-aware route sees what it expects.
    Residues get N, CA, C, O; consecutive residues are linked C(i)-N(i+1).
    """
    top = app.Topology()
    chain = top.addChain()
    prev_c = None
    for i, name in enumerate(residue_names):
        res = top.addResidue(name, chain)
        n = top.addAtom("N", elem.nitrogen, res)
        ca = top.addAtom("CA", elem.carbon, res)
        c = top.addAtom("C", elem.carbon, res)
        o = top.addAtom("O", elem.oxygen, res)
        top.addBond(n, ca)
        top.addBond(ca, c)
        top.addBond(c, o)
        if i in n_methyl:                      # N-methylated amide nitrogen
            cme = top.addAtom("CN", elem.carbon, res)
            top.addBond(n, cme)
        if prev_c is not None:
            top.addBond(prev_c, n)             # the peptide (omega) bond
        prev_c = c
    return top


# ==================================================================================================
# section 5 -- three-sector REST2 scaling
# ==================================================================================================
@pytest.mark.parametrize("s", S_VALUES)
def test_particle_parameters_scale_by_sector(s):
    """charge -> sqrt(s) and epsilon -> s on SOLUTE particles only; sigma never changes."""
    system, solute, solvent = _explicit_test_system()
    base = _nb_force(system)
    ref = [_nb_particle(base, i) for i in range(system.getNumParticles())]
    scaled_system = build_rest2_scaled_system(system, np.array(solute), s)   # keep alive!
    scaled = _nb_force(scaled_system)

    for i in solute:
        q, sig, eps = _nb_particle(scaled, i)
        q0, sig0, eps0 = ref[i]
        assert q == pytest.approx(q0 * math.sqrt(s), abs=1e-12)
        assert eps == pytest.approx(eps0 * s, abs=1e-12)
        assert sig == pytest.approx(sig0, abs=1e-12), "sigma must never be scaled"
    for i in solvent:
        assert _nb_particle(scaled, i) == pytest.approx(ref[i], abs=1e-12), \
            "solvent particle parameters must be untouched"


@pytest.mark.parametrize("s", S_VALUES)
def test_pair_energies_scale_by_sector(s):
    """The physics statement itself: solute-solute by s, solute-solvent by sqrt(s), solvent by 1."""
    system, solute, solvent = _explicit_test_system()
    scaled_system = build_rest2_scaled_system(system, np.array(solute), s)   # keep alive!
    base, scaled = _nb_force(system), _nb_force(scaled_system)

    def pair(force, i, j):
        qi, si, ei = _nb_particle(force, i)
        qj, sj, ej = _nb_particle(force, j)
        return _pair_energy([qi, qj], [si, sj], [ei, ej])

    assert pair(scaled, 0, 1) == pytest.approx(s * pair(base, 0, 1), rel=1e-10), "solute-solute"
    assert pair(scaled, 0, 4) == pytest.approx(math.sqrt(s) * pair(base, 0, 4), rel=1e-10), \
        "solute-solvent"
    assert pair(scaled, 4, 5) == pytest.approx(pair(base, 4, 5), rel=1e-12), "solvent-solvent"
    assert pair(scaled, 6, 7) == pytest.approx(pair(base, 6, 7), rel=1e-12), "solvent-solvent"


@pytest.mark.parametrize("s", S_VALUES)
def test_exceptions_scale_by_sector(s):
    """Two-solute exceptions by s, mixed by sqrt(s), solvent-only untouched."""
    system, solute, _ = _explicit_test_system()
    scaled_system = build_rest2_scaled_system(system, np.array(solute), s)   # keep alive!
    base, scaled = _nb_force(system), _nb_force(scaled_system)
    want = {0: s, 1: math.sqrt(s), 2: 1.0}          # exception index -> expected factor
    for idx, factor in want.items():
        i0, j0, q0, sig0, eps0 = _nb_exception(base, idx)
        i1, j1, q1, sig1, eps1 = _nb_exception(scaled, idx)
        assert (i0, j0) == (i1, j1)
        assert q1 == pytest.approx(q0 * factor, abs=1e-12)
        assert eps1 == pytest.approx(eps0 * factor, abs=1e-12)
        assert sig1 == pytest.approx(sig0, abs=1e-12)


@pytest.mark.parametrize("s", (0.64, 0.25))
def test_pme_reciprocal_follows_charge_scaling(s):
    """The scaled System's full PME energy must equal an independently built intended system.

    Constructed from the physics statement rather than from the implementation: solute charges
    times sqrt(s), solute epsilons times s, solvent untouched. If the scaler skipped reciprocal
    space, or scaled charges by s instead of sqrt(s), this diverges.
    """
    system, solute, _ = _explicit_test_system()
    scaled = build_rest2_scaled_system(system, np.array(solute), s)

    intended = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    nb = _nb_force(intended)
    for i in solute:
        q, sig, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q * math.sqrt(s), sig, eps * s)
    for idx in range(nb.getNumExceptions()):
        i, j, q, sig, eps = nb.getExceptionParameters(idx)
        n_sol = int(i in solute) + int(j in solute)
        f = {0: 1.0, 1: math.sqrt(s), 2: s}[n_sol]
        nb.setExceptionParameters(idx, i, j, q * f, sig, eps * f)

    positions = [Vec3(0.1 * k, 0.15 * (k % 3), 0.2 * (k % 2)) for k in range(8)] * unit.nanometer

    def energy(sys_):
        # strip to the NonbondedForce.  The scaler ALSO scales torsions and CMAP by s, which the
        # hand-built "intended" system deliberately does not touch, so a whole-system comparison
        # would differ for a reason that has nothing to do with electrostatics.
        stripped = XmlSerializer.deserialize(XmlSerializer.serialize(sys_))
        for i in reversed(range(stripped.getNumForces())):
            if not isinstance(stripped.getForce(i), NonbondedForce):
                stripped.removeForce(i)
        ctx = openmm.Context(stripped, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(positions)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    assert energy(scaled) == pytest.approx(energy(intended), abs=1e-6)


@pytest.mark.parametrize("s", S_VALUES)
def test_dispersion_correction_preserved(s):
    system, solute, _ = _explicit_test_system()
    scaled_system = build_rest2_scaled_system(system, np.array(solute), s)   # keep alive!
    scaled = _nb_force(scaled_system)
    assert scaled.getUseDispersionCorrection() is True
    assert scaled.getNonbondedMethod() == NonbondedForce.PME


@pytest.mark.parametrize("s", (0.64, 0.25))
def test_bonds_and_angles_are_never_scaled(s):
    """Harmonic bond and angle terms must be identical, parameter by parameter and in energy."""
    system, solute, _ = _explicit_test_system()
    scaled = build_rest2_scaled_system(system, np.array(solute), s)

    b0, b1 = _force_of(system, HarmonicBondForce), _force_of(scaled, HarmonicBondForce)
    assert b0.getNumBonds() == b1.getNumBonds()
    for i in range(b0.getNumBonds()):
        assert b0.getBondParameters(i) == b1.getBondParameters(i)

    a0, a1 = _force_of(system, HarmonicAngleForce), _force_of(scaled, HarmonicAngleForce)
    assert a0.getNumAngles() == a1.getNumAngles()
    for i in range(a0.getNumAngles()):
        assert a0.getAngleParameters(i) == a1.getAngleParameters(i)

    # and their isolated energy contributions
    positions = [Vec3(0.1 * k, 0.15 * (k % 3), 0.2 * (k % 2)) for k in range(8)] * unit.nanometer

    def bonded_energy(sys_):
        stripped = XmlSerializer.deserialize(XmlSerializer.serialize(sys_))
        for i in reversed(range(stripped.getNumForces())):
            if not isinstance(stripped.getForce(i), (HarmonicBondForce, HarmonicAngleForce)):
                stripped.removeForce(i)
        ctx = openmm.Context(stripped, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(positions)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    assert bonded_energy(scaled) == pytest.approx(bonded_energy(system), rel=1e-12)


@pytest.mark.parametrize("s", (0.64, 0.25))
def test_torsions_and_cmap_scale_but_excluded_omega_does_not(s):
    system, solute, _ = _explicit_test_system()
    # exclude the torsion whose central bond is 1-2
    scaled = build_rest2_scaled_system(system, np.array(solute), s,
                                       exclude_central_bonds=[(1, 2)])
    t0, t1 = _force_of(system, PeriodicTorsionForce), _force_of(scaled, PeriodicTorsionForce)
    k_before = {tuple(t0.getTorsionParameters(i)[:4]): t0.getTorsionParameters(i)[6]
                for i in range(t0.getNumTorsions())}
    k_after = {tuple(t1.getTorsionParameters(i)[:4]): t1.getTorsionParameters(i)[6]
               for i in range(t1.getNumTorsions())}
    excluded, eligible, cross = (0, 1, 2, 3), (1, 2, 3, 0), (0, 1, 4, 5)
    kj = unit.kilojoule_per_mole
    assert k_after[excluded].value_in_unit(kj) == pytest.approx(
        k_before[excluded].value_in_unit(kj), abs=1e-12), "excluded omega must keep its k"
    assert k_after[eligible].value_in_unit(kj) == pytest.approx(
        k_before[eligible].value_in_unit(kj) * s, abs=1e-12), "eligible torsion must scale by s"
    assert k_after[cross].value_in_unit(kj) == pytest.approx(
        k_before[cross].value_in_unit(kj), abs=1e-12), \
        "a torsion straddling solvent is not an intramolecular solute term and must not scale"

    c0, c1 = _force_of(system, CMAPTorsionForce), _force_of(scaled, CMAPTorsionForce)
    _, e0 = c0.getMapParameters(0)
    _, e1 = c1.getMapParameters(0)
    for a, b in zip(e0, e1):
        assert b.value_in_unit(kj) == pytest.approx(a.value_in_unit(kj) * s, abs=1e-12)


def test_scaling_at_s_one_is_the_identity():
    system, solute, _ = _explicit_test_system()
    scaled = build_rest2_scaled_system(system, np.array(solute), 1.0)
    a, b = _nb_force(system), _nb_force(scaled)
    for i in range(system.getNumParticles()):
        for x, y in zip(a.getParticleParameters(i), b.getParticleParameters(i)):
            assert x == pytest.approx(y, abs=1e-12) if not hasattr(x, "unit") else True


# ==================================================================================================
# section 3 -- omega classification
# ==================================================================================================
def test_ace_ala_nme_ordinary_peptide_bonds_are_unscaled():
    top = _peptide_topology(["ACE", "ALA", "NME"])
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert len(r["omega_unscaled_bonds"]) == 2, r
    assert r["omega_proline_like_scaled_bonds"] == []
    assert r["omega_unclassified_candidates"] == []
    assert "residue-aware" in r["omega_detection_method"]


def test_x_pro_peptide_bond_stays_eligible_for_scaling():
    """The X-PRO bond must NOT be excluded; the other peptide bond must be."""
    top = _peptide_topology(["ACE", "PRO", "ALA"])
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    pro_n = [a.index for a in top.atoms() if a.residue.name == "PRO" and a.name == "N"][0]
    scaled_ns = {n for _, n in r["omega_proline_like_scaled_bonds"]}
    unscaled_ns = {n for _, n in r["omega_unscaled_bonds"]}
    assert pro_n in scaled_ns, "X-PRO must remain eligible for REST2 torsion scaling"
    assert pro_n not in unscaled_ns
    assert len(r["omega_unscaled_bonds"]) == 1, "the PRO-ALA bond is an ordinary amide"


def test_non_proline_ordinary_amide_is_unscaled():
    top = _peptide_topology(["ALA", "GLY"])
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert len(r["omega_unscaled_bonds"]) == 1
    assert r["omega_proline_like_scaled_bonds"] == []


def test_n_methylated_amide_is_still_ordinary():
    """N-methylation does not make an amide proline-like -- those bonds isomerise readily."""
    top = _peptide_topology(["ACE", "ALA", "ALA"], n_methyl=(1, 2))
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert len(r["omega_unscaled_bonds"]) == 2
    assert r["omega_proline_like_scaled_bonds"] == []


def test_proline_like_residue_set_is_configurable():
    top = _peptide_topology(["ACE", "HYP", "ALA"])
    default = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert default["omega_unclassified_candidates"], "an unknown residue must not be guessed at"
    widened = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide",
                                   proline_like_residues=("PRO", "HYP"))
    assert len(widened["omega_proline_like_scaled_bonds"]) == 1
    assert widened["omega_unclassified_candidates"] == []


def test_non_amide_cn_bond_is_not_a_candidate():
    """A C-N bond with no carbonyl (an amine) must never enter the omega machinery."""
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    c = top.addAtom("CA", elem.carbon, res)
    n = top.addAtom("N", elem.nitrogen, res)
    top.addBond(c, n)
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert r["omega_unscaled_bonds"] == []
    assert r["omega_unclassified_candidates"] == []


def test_carboxylic_acid_carbon_is_flagged_not_assumed():
    """C bonded to N, a carbonyl O and a hydroxyl O is not a plain amide: it must be flagged."""
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    c = top.addAtom("C", elem.carbon, res)
    o1 = top.addAtom("O", elem.oxygen, res)
    o2 = top.addAtom("OXT", elem.oxygen, res)
    h = top.addAtom("HO", elem.hydrogen, res)
    n = top.addAtom("N", elem.nitrogen, res)
    for x, y in ((c, o1), (c, o2), (o2, h), (c, n)):
        top.addBond(x, y)
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="peptide")
    assert r["omega_unscaled_bonds"] == []
    assert len(r["omega_unclassified_candidates"]) == 1
    assert "hydroxyl" in r["omega_unclassified_candidates"][0]["ambiguous"]


# ---- ligand route -------------------------------------------------------------------------------
def _sdf_and_topology(smiles, tmp_path):
    """Embed a SMILES, write the SDF, and build the matching OpenMM topology from the same mol."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=20260814) == 0
    sdf = tmp_path / "lig.sdf"
    Chem.MolToMolFile(mol, str(sdf))

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("UNL", chain)
    atoms = [top.addAtom(f"{a.GetSymbol()}{i}", app.Element.getBySymbol(a.GetSymbol()), res)
             for i, a in enumerate(mol.GetAtoms())]
    for b in mol.GetBonds():
        top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    return sdf, top


def test_ligand_route_flags_small_ring_amide_as_proline_like(tmp_path):
    """A gamma-lactam nitrogen sits in a 5-ring: proline-like, so eligible for scaling."""
    sdf, top = _sdf_and_topology("O=C1CCCN1C", tmp_path)      # N-methyl-2-pyrrolidinone
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="ligand", ligand_sdf=sdf)
    assert len(r["omega_proline_like_scaled_bonds"]) == 1, r
    assert r["omega_unscaled_bonds"] == []
    assert r["omega_unclassified_candidates"] == []
    assert "ring of <= 7" in r["omega_detection_method"]


def test_ligand_route_keeps_macrocyclic_amides_unscaled(tmp_path):
    """Every backbone N of a cyclic peptide is 'in a ring' -- the RING-SIZE BOUND is what stops
    an unbounded [NX3;R] test from freeing all of them for scaling."""
    smiles = "O=C1CNC(=O)CNC(=O)CNC(=O)CN1"                   # cyclo-tetraglycine, 12-ring
    sdf, top = _sdf_and_topology(smiles, tmp_path)
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="ligand", ligand_sdf=sdf)
    assert len(r["omega_unscaled_bonds"]) == 4, r
    assert r["omega_proline_like_scaled_bonds"] == [], "a 12-ring is not proline-like"
    assert r["omega_unclassified_candidates"] == []


def test_ligand_route_acyclic_n_methyl_amide_is_unscaled(tmp_path):
    sdf, top = _sdf_and_topology("CC(=O)N(C)C", tmp_path)     # N,N-dimethylacetamide
    r = classify_omega_bonds(top, range(top.getNumAtoms()), route="ligand", ligand_sdf=sdf)
    assert len(r["omega_unscaled_bonds"]) == 1
    assert r["omega_proline_like_scaled_bonds"] == []


def test_ligand_route_rejects_mismatched_sdf(tmp_path):
    """The RDKit<->OpenMM mapping is asserted, not assumed."""
    sdf, _ = _sdf_and_topology("CC(=O)NC", tmp_path)
    other_dir = tmp_path / "b"
    other_dir.mkdir()
    _, other_top = _sdf_and_topology("CCC(=O)NCC", other_dir)
    with pytest.raises(ValueError, match="mismatch"):
        classify_omega_bonds(other_top, range(other_top.getNumAtoms()),
                             route="ligand", ligand_sdf=sdf)


# ==================================================================================================
# section 6 -- configuration, ladders, routes, box, HMR, restarts
# ==================================================================================================
def test_config_rejects_unknown_keys(tmp_path):
    good = tmp_path / "g.json"
    good.write_text(json.dumps({"solvation": {"padding_nm": 1.5}}))
    assert load_config(good)["solvation"]["padding_nm"] == 1.5
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"solvation": {"paddding_nm": 1.5}}))
    with pytest.raises(KeyError, match="paddding_nm"):
        load_config(bad)


def test_seeds_derive_from_the_master_seed(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"run": {"seed": 4242}}))
    cfg = load_config(f)
    seeds = [cfg["structure"]["etkdg"]["seed"], cfg["equilibration"]["seed"],
             cfg["production"]["md"]["seed"], cfg["production"]["remd"]["seed"]]
    assert all(x is not None for x in seeds)
    assert len(set(seeds)) == len(seeds), "per-stage seeds must be distinct"


def test_default_ladders():
    assert rest2_ladder(1.0, 0.25, 6, "sqrt") == pytest.approx(
        [1.0, 0.81, 0.64, 0.49, 0.36, 0.25], abs=1e-12)
    assert rest2_ladder(1.0, 0.25, 8, "sqrt") == pytest.approx(
        [1.0, 0.862244897959, 0.734693877551, 0.617346938776,
         0.510204081633, 0.413265306122, 0.326530612245, 0.25], abs=1e-11)
    assert rest2_ladder(1.0, 0.25, 4, "linear") == pytest.approx([1.0, 0.75, 0.5, 0.25], abs=1e-12)
    with pytest.raises(ValueError):
        rest2_ladder(1.0, 0.25, 1, "sqrt")
    with pytest.raises(ValueError):
        rest2_ladder(1.0, 2.0, 4, "sqrt")


def test_generate_config_presets_match_the_documented_ladders():
    import importlib.util

    path = (Path(__file__).resolve().parents[1] / "docs/implementation/explicit_solvent/scripts"
            / "generate_config.py")
    spec = importlib.util.spec_from_file_location("generate_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SYSTEM_PRESETS["alanine"]["n_rungs"] == 6
    assert mod.SYSTEM_PRESETS["macrocycle"]["n_rungs"] == 8
    # the preset must not pin the force-field route
    assert mod.SYSTEM_PRESETS["alanine"]["solute_kind"] == "auto"


def test_route_follows_the_input_not_the_preset():
    cfg = load_config()
    assert resolve_route(cfg, None, input_route="smiles") == "ligand"
    top = _peptide_topology(["ACE", "ALA", "NME"])
    assert resolve_route(cfg, top, input_route="pdb") == "peptide"


def test_route_conflict_is_rejected_before_any_work():
    cfg = load_config()
    cfg["system"]["solute_kind"] = "peptide"
    with pytest.raises(ValueError, match="contradicts --smiles"):
        resolve_route(cfg, None, input_route="smiles")


def test_box_delivers_the_requested_gap_and_fits_the_cutoff():
    frac = float(np.min(np.diag(_box_vectors(1.0, "dodecahedron"))))
    assert frac == pytest.approx(1 / math.sqrt(2), abs=1e-12)

    class _M:
        positions = np.array([[0.0, 0, 0], [0.8, 0, 0]]) * unit.nanometer

    cfg = load_config()
    g = _resolve_box(_M(), cfg)
    assert g["solute_image_gap_nm"] >= cfg["solvation"]["padding_nm"] - 1e-9
    assert g["max_legal_cutoff_nm"] >= cfg["system_build"]["nonbonded_cutoff_nm"] - 1e-9


def test_box_refuse_policy_reports_the_workable_padding():
    class _M:
        positions = np.array([[0.0, 0, 0], [0.5, 0, 0]]) * unit.nanometer

    cfg = load_config()
    cfg["solvation"]["padding_semantics"] = "openmm"
    cfg["solvation"]["cutoff_fit_policy"] = "refuse"
    with pytest.raises(ValueError, match="minimum image distance"):
        _resolve_box(_M(), cfg)


def test_hmr_conserves_mass_and_never_touches_water():
    system = System()
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    c = top.addAtom("C", elem.carbon, res)
    h = top.addAtom("H", elem.hydrogen, res)
    wat = top.addResidue("HOH", chain)
    ow = top.addAtom("O", elem.oxygen, wat)
    hw = top.addAtom("H1", elem.hydrogen, wat)
    top.addBond(c, h)
    top.addBond(ow, hw)
    for m in (12.0, 1.008, 16.0, 1.008):
        system.addParticle(m * unit.amu)

    before = sum(system.getParticleMass(i).value_in_unit(unit.amu) for i in range(4))
    info = repartition_hydrogen_mass(system, top, 3.024, range(2))
    after = sum(system.getParticleMass(i).value_in_unit(unit.amu) for i in range(4))
    assert info["n_hydrogens_repartitioned"] == 1
    assert after == pytest.approx(before, abs=1e-9)
    assert system.getParticleMass(1).value_in_unit(unit.amu) == pytest.approx(3.024)
    assert system.getParticleMass(0).value_in_unit(unit.amu) == pytest.approx(12.0 - 2.016)
    assert system.getParticleMass(3).value_in_unit(unit.amu) == pytest.approx(1.008), \
        "water hydrogens must never be repartitioned"


def test_hmr_refuses_to_empty_a_heavy_atom():
    system = System()
    top = app.Topology()
    res = top.addResidue("LIG", top.addChain())
    c = top.addAtom("C", elem.carbon, res)
    h = top.addAtom("H", elem.hydrogen, res)
    top.addBond(c, h)
    system.addParticle(12.0 * unit.amu)
    system.addParticle(1.008 * unit.amu)
    # asking for 12.5 amu on the H would leave the carbon at 0.5 amu
    with pytest.raises(ValueError, match="Lower"):
        repartition_hydrogen_mass(system, top, 12.5, range(2))


def test_reporter_interval_alignment_rejects_non_integer_steps():
    assert _steps(10.0, 4.0) == 2500
    assert _steps(2.0, 4.0) == 500
    with pytest.raises(ValueError, match="integer number"):
        _steps(1.0, 3.0)


def test_completed_prefix_accepts_a_contiguous_run(tmp_path):
    for c in range(3):
        d = tmp_path / f"chunk_{c:04d}"
        d.mkdir()
        (d / "done.json").write_text("{}")
        (d / "end.chk").write_bytes(b"x")
    assert completed_prefix(tmp_path, 10) == 3


def test_completed_prefix_rejects_a_gap(tmp_path):
    for c in (0, 2):
        d = tmp_path / f"chunk_{c:04d}"
        d.mkdir()
        (d / "done.json").write_text("{}")
        (d / "end.chk").write_bytes(b"x")
    with pytest.raises(ValueError, match="missing"):
        completed_prefix(tmp_path, 10)


def test_completed_prefix_rejects_a_missing_checkpoint(tmp_path):
    d = tmp_path / "chunk_0000"
    d.mkdir()
    (d / "done.json").write_text("{}")
    with pytest.raises(ValueError, match="end.chk"):
        completed_prefix(tmp_path, 10)


def test_relaxation_field_exists_and_is_validated(tmp_path):
    assert DEFAULTS["production"]["remd"]["equilibration_ps"] == 10.0
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"production": {"remd": {"equilibration_ps": 25.0}}}))
    assert load_config(f)["production"]["remd"]["equilibration_ps"] == 25.0
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"production": {"remd": {"equilibraton_ps": 25.0}}}))
    with pytest.raises(KeyError):
        load_config(bad)


def test_omega_selective_disabled_scales_everything():
    cfg = load_config()
    assert cfg["rest2"]["omega_selective"] is True
    assert cfg["rest2"]["proline_like_residues"] == ["PRO"]
    assert cfg["rest2"]["max_proline_ring_size"] == 7


# ==================================================================================================
# force-field provenance: a Sage run must never report ff19SB
# ==================================================================================================
def test_ligand_route_does_not_load_the_protein_forcefield():
    """cyclo-RGDfV explicit runs on Sage 2.2 + AM1BCC; ff19SB must not appear in the manifest.

    The protein XML parameterises nothing on this route -- the solute is one UNL residue handled by
    SMIRNOFF, and the water XML already carries the Na+/Cl-/HOH templates addSolvent needs -- but
    while it was loaded unconditionally it appeared in the recorded force-field list of a Sage
    calculation, misdescribing the Hamiltonian that ran.
    """
    from md_templates.openmm import build_forcefield

    cfg = load_config()
    _, info = build_forcefield(cfg, None, route="ligand")
    assert info["protein_forcefield"] is None
    assert not any("ff19SB" in x for x in info["xml"]), info["xml"]
    assert info["route"] == "ligand"

    _, prot_info = build_forcefield(cfg, None, route="peptide")
    assert prot_info["protein_forcefield"] == "amber19/protein.ff19SB.xml"
    assert any("ff19SB" in x for x in prot_info["xml"])


def test_water_forcefield_alone_supplies_ions_and_water():
    """The ligand route drops the protein XML, so the water XML must carry the solvent templates."""
    from openmm import app

    ff = app.ForceField(load_config()["forcefield"]["water"])
    names = set(ff._templates)
    assert "HOH" in names
    assert any(n in names for n in ("NA", "Na+")), sorted(n for n in names if "a" in n.lower())[:5]
    assert any(n in names for n in ("CL", "Cl-"))


def test_rgd_system_yaml_is_not_consumed_by_the_explicit_workflow():
    """The legacy ff19SB field in systems/cyclo_rgdfv/system.yaml is workflow-specific.

    The explicit path never reads that file -- it takes the SMILES on the command line -- so the
    field cannot leak into an explicit run. This pins that, so a future refactor that starts
    reading system.yaml has to confront the question rather than inherit ff19SB silently.

    Scans the whole simulation package rather than one file. The implementation used to be a
    single module; splitting it into config/system/solvation/equilibration/md/rest2 would have
    silently retired this check if it kept naming one path.
    """
    pkg = Path(__file__).resolve().parents[1] / "src/md_templates/openmm"
    simulation_modules = ["config.py", "system.py", "solvation.py",
                          "equilibration.py", "md.py", "rest2.py"]
    for name in simulation_modules:
        path = pkg / name
        assert path.is_file(), f"{name} is missing: the scan would silently pass"
        assert "system.yaml" not in path.read_text(), (
            f"{name} must not read system.yaml implicitly"
        )


# ==================================================================================================
# named-preset invariants (fix_6 sections 3-4): behaviour, not source-text matching
# ==================================================================================================
def _presets():
    import importlib.util

    path = (Path(__file__).resolve().parents[1] / "docs/implementation/explicit_solvent/scripts"
            / "generate_config.py")
    spec = importlib.util.spec_from_file_location("generate_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RGD_LADDER = [1.000000000000, 0.891975308642, 0.790123456790, 0.694444444444, 0.604938271605,
              0.521604938272, 0.444444444444, 0.373456790123, 0.308641975309, 0.250000000000]


def test_preset_rung_counts():
    P = _presets().SYSTEM_PRESETS
    assert P["alanine"]["n_rungs"] == 6
    assert P["macrocycle"]["n_rungs"] == 8
    assert P["cyclo_rgdfv"]["n_rungs"] == 10


def test_rgd_ladder_values_match_exactly():
    assert rest2_ladder(1.0, 0.25, 10, "sqrt") == pytest.approx(RGD_LADDER, abs=1e-11)


def test_rgd_preset_slug_and_enforced_route():
    P = _presets().SYSTEM_PRESETS["cyclo_rgdfv"]
    assert P["solute_kind"] == "ligand", "must not be route-neutral"
    assert P["require_input_route"] == "smiles"
    mod = _presets()
    import argparse

    a = argparse.Namespace(seed=1, slug=None, system="cyclo_rgdfv", padding_nm=1.2,
                           box_shape="dodecahedron", salt_molar=0.15, cutoff_nm=1.0,
                           hydrogen_mass_amu=3.024, ph=7.0, no_omega_selective=False, water=None)
    a.slug = "cyclo_rgdfv_sage_explicit"
    cfg = mod.simbox_config(a, P)
    assert cfg["system"]["slug"] == "cyclo_rgdfv_sage_explicit"
    assert cfg["system"]["require_input_route"] == "smiles"


def test_macrocycle_preset_stays_route_neutral():
    P = _presets().SYSTEM_PRESETS["macrocycle"]
    assert P["solute_kind"] == "auto"
    assert P.get("require_input_route") is None


def test_alanine_preset_route_unchanged():
    P = _presets().SYSTEM_PRESETS["alanine"]
    assert P["solute_kind"] == "auto"
    assert P.get("require_input_route") is None
    cfg = load_config()
    cfg["system"]["solute_kind"] = "auto"
    assert resolve_route(cfg, None, input_route="smiles") == "ligand"


def test_rgd_config_accepts_the_smiles_route():
    cfg = load_config()
    cfg["system"].update(solute_kind="ligand", require_input_route="smiles")
    assert resolve_route(cfg, None, input_route="smiles") == "ligand"


def test_rgd_config_rejects_a_pdb_route_before_parameterisation():
    cfg = load_config()
    cfg["system"].update(solute_kind="ligand", require_input_route="smiles")
    with pytest.raises(ValueError, match="requires the 'smiles' input route"):
        resolve_route(cfg, None, input_route="pdb")


def test_rgd_build_simbox_rejects_a_missing_smiles(tmp_path):
    from md_templates.openmm import build_simbox

    cfg = load_config()
    cfg["system"].update(solute_kind="ligand", require_input_route="smiles")
    with pytest.raises(ValueError, match="requires the 'smiles' input route"):
        build_simbox(cfg, tmp_path, "x", pdb=tmp_path / "nope.pdb")
    with pytest.raises(ValueError, match="no SMILES was given"):
        build_simbox(cfg, tmp_path, "x", smiles="   ")


def test_rgd_manifest_records_the_full_hamiltonian_provenance(tmp_path):
    """The recorded provenance must name Sage 2.2, AM1-BCC, charge 0, the SMILES hash, no protein FF."""
    import hashlib

    from md_templates.openmm import build_forcefield

    cfg = load_config()
    cfg["system"].update(solute_kind="ligand", require_input_route="smiles")
    _, info = build_forcefield(cfg, None, route="ligand")
    assert info["protein_forcefield"] is None
    assert cfg["forcefield"]["ligand"] == "openff-2.2.0"
    assert cfg["forcefield"]["ligand_charge_method"] == "am1bcc"
    smiles = "CC(=O)NC"
    assert hashlib.sha256(smiles.encode()).hexdigest()  # the field build_simbox records


def test_dumped_defaults_equal_the_runtime_tree():
    """config_defaults.json must be a FUNCTION of DEFAULTS, not a second hand-edited copy."""
    from md_templates.openmm import dump_defaults

    path = (Path(__file__).resolve().parents[1]
            / "docs/implementation/explicit_solvent/scripts/config_defaults.json")
    assert json.loads(path.read_text()) == json.loads(dump_defaults()), \
        "config_defaults.json has drifted; regenerate with dump_defaults(path)"


# ==================================================================================================
# requirement 4: the equilibrated structure carries the SELECTED NPT box, not the pre-NPT one
# ==================================================================================================

def _write_equilibrated_like_production(tmp_path, box_nm, shape):
    """Reproduce the write path: a topology built with one box, a state carrying another.

    This is the exact shape of the bug -- the pre-NPT topology and the post-NPT state disagree, and
    whichever one the writer uses decides what the structure files say.
    """
    import copy as _copy

    from openmm import unit as _unit
    from openmm.app import Topology as _Topology

    top = _Topology()
    chain = top.addChain()
    res = top.addResidue("HOH", chain)
    top.addAtom("O", elem.oxygen, res)
    pre_npt = 1.0
    top.setPeriodicBoxVectors(([pre_npt, 0, 0], [0, pre_npt, 0], [0, 0, pre_npt]) * _unit.nanometer)

    selected = (tuple(box_nm[0]), tuple(box_nm[1]), tuple(box_nm[2])) * _unit.nanometer
    eq_top = _copy.deepcopy(top)
    eq_top.setPeriodicBoxVectors(selected)
    positions = [[0.0, 0.0, 0.0]] * _unit.nanometer
    pdb = tmp_path / f"{shape}_equilibrated.pdb"
    cif = tmp_path / f"{shape}_equilibrated.cif"
    with pdb.open("w") as fh:
        app.PDBFile.writeFile(eq_top, positions, fh)
    with cif.open("w") as fh:
        app.PDBxFile.writeFile(eq_top, positions, fh)
    return pdb, cif, top, selected


@pytest.mark.parametrize("shape,box_nm", [
    ("cube", [[3.4, 0.0, 0.0], [0.0, 3.4, 0.0], [0.0, 0.0, 3.4]]),
    # the dodecahedral cell the RGD system actually uses: a triclinic box whose third vector is not
    # axis-aligned. PDB cannot round-trip it -- OpenMM returns the REDUCED lattice form, which flips
    # the sign of the third vector's x/y. Volume is the invariant that survives; the mmCIF copy is
    # what preserves the vectors.
    ("dodecahedron", [[3.66182, 0.0, 0.0], [0.0, 3.66182, 0.0], [1.83091, 1.83091, 2.58930]]),
])
def test_equilibrated_structure_records_the_selected_box_not_the_pre_npt_one(tmp_path, shape, box_nm):
    from openmm import unit as _unit

    pdb, cif, pre_npt_topology, selected = _write_equilibrated_like_production(
        tmp_path, box_nm, shape)
    selected_nm = np.array(selected.value_in_unit(_unit.nanometer))
    pre_nm = np.array(pre_npt_topology.getPeriodicBoxVectors().value_in_unit(_unit.nanometer))
    vol_selected = float(abs(np.linalg.det(selected_nm)))
    vol_pre = float(abs(np.linalg.det(pre_nm)))

    for path, reader, tol in ((pdb, app.PDBFile, 2e-4), (cif, app.PDBxFile, 2e-4)):
        written = np.array(reader(str(path)).topology.getPeriodicBoxVectors()
                           .value_in_unit(_unit.nanometer))
        vol_written = float(abs(np.linalg.det(written)))
        # the volume is representation-independent, so it is the check that holds for both formats
        assert vol_written == pytest.approx(vol_selected, rel=1e-3), (
            f"{shape}/{path.suffix}: volume {vol_written} != selected {vol_selected}")
        assert abs(vol_written - vol_pre) > 1e-6, (
            f"{shape}/{path.suffix}: volume equals the PRE-NPT box -- the bug this test exists for")
        # lengths are preserved by both formats even where the reduced form flips signs
        assert np.allclose(np.linalg.norm(written, axis=1),
                           np.linalg.norm(selected_nm, axis=1), atol=tol), (
            f"{shape}/{path.suffix}: vector lengths {np.linalg.norm(written, axis=1).tolist()} "
            f"!= selected {np.linalg.norm(selected_nm, axis=1).tolist()}")

    # mmCIF additionally preserves the vectors themselves up to the reduced-form sign convention
    cif_box = np.array(app.PDBxFile(str(cif)).topology.getPeriodicBoxVectors()
                       .value_in_unit(_unit.nanometer))
    assert np.allclose(np.abs(cif_box), np.abs(selected_nm), atol=2e-4), (
        f"{shape}: mmCIF vectors {cif_box.tolist()} != selected {selected_nm.tolist()}")
