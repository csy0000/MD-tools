"""Which torsions REST2 leaves unscaled: amide omega, aromatic ring bonds, other double bonds, impropers.

The user's decision of 2026-09-16 (docs/amber-like-fix/REST2-scaler.md §10), replacing "ordinary
amide omega only". Scaling any of these lets a hot state leave a planar geometry the physical state
never leaves -- an amide isomerising, a ring puckering, a double bond twisting, a trivalent centre
pyramidalising -- so the hot states sample conformations that state 0 does not, and only the
unscaled terms keep them honest.

Two halves, tested separately because they are separate contracts:

  * WHICH BONDS -- `md_tools.openmm.system.unscaled_torsions`: chemistry from residue names (a table
    for proteins, checked against OpenMM's own templates) or from an SDF's bond orders;
  * WHAT HAPPENS TO THE TERMS -- `md_tools.rest2.hamiltonian`: every proper torsion across an
    unscaled central bond, and every improper, keeps its force constant at every tau. An improper
    is found from the System's bond graph, not from atom order: Amber writes the centre third,
    SMIRNOFF second.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping, RDKit perception and System parameter reads. No
Context is created.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from .conftest import make_dataset_root

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")

_SYMBOLS = {"C": elem.carbon, "N": elem.nitrogen, "O": elem.oxygen, "H": elem.hydrogen,
            "S": elem.sulfur}


# --- small molecules, from an SDF ---------------------------------------------------------------

def _ligand(directory: Path, smiles: str, name: str = "LIG"):
    rdkit = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=20260916)
    topology = app.Topology()
    residue = topology.addResidue(name, topology.addChain())
    atoms = [topology.addAtom(a.GetSymbol(), _SYMBOLS[a.GetSymbol()], residue)
             for a in mol.GetAtoms()]
    for bond in mol.GetBonds():
        topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    sdf = directory / f"{name}.sdf"
    rdkit.MolToMolFile(mol, str(sdf))
    return topology, [a.index for a in atoms], {name: sdf}, mol


def _by_class(result):
    out: dict[str, set] = {}
    for entry in result["central_bonds"]:
        out.setdefault(entry["class"], set()).add(tuple(sorted(entry["bond"])))
    return out


def test_paracetamol_keeps_its_amide_and_its_six_ring_bonds(tmp_path):
    from md_tools.openmm.system import unscaled_torsions

    topology, solute, sdfs, mol = _ligand(tmp_path, "CC(=O)Nc1ccc(O)cc1")
    result = unscaled_torsions(topology, solute, residue_sdfs=sdfs)
    classes = _by_class(result)

    assert len(classes.get("amide_omega", ())) == 1
    assert len(classes.get("aromatic_ring", ())) == 6
    assert classes.get("double_bond", set()) == set(), (
        "C=O has no torsion across it: a terminal oxygen is never a central bond")
    ring = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in mol.GetBonds() if b.GetIsAromatic()}
    assert classes["aromatic_ring"] == ring
    assert sorted(result["unscaled_central_bonds"]) == sorted(
        classes["amide_omega"] | classes["aromatic_ring"])
    assert result["unscaled_impropers"] is True


def test_a_chain_double_bond_is_kept(tmp_path):
    from md_tools.openmm.system import unscaled_torsions

    topology, solute, sdfs, mol = _ligand(tmp_path, "C/C=C/C(=O)O")      # crotonic acid
    classes = _by_class(unscaled_torsions(topology, solute, residue_sdfs=sdfs))
    double = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
              if b.GetBondTypeAsDouble() == 2.0 and b.GetBeginAtom().GetSymbol() == "C"
              and b.GetEndAtom().GetSymbol() == "C"}
    assert classes.get("double_bond") == double
    assert "amide_omega" not in classes, "an acid is not an amide"


def test_a_proline_like_amide_is_scaled_but_its_neighbours_rules_still_apply(tmp_path):
    from md_tools.openmm.system import unscaled_torsions

    topology, solute, sdfs, _ = _ligand(tmp_path, "CC(=O)N1CCCC1")       # N-acetylpyrrolidine
    result = unscaled_torsions(topology, solute, residue_sdfs=sdfs)
    assert "amide_omega" not in _by_class(result)
    assert len(result["proline_like_scaled_bonds"]) == 1


def test_a_small_molecule_without_bond_orders_is_unclassified_even_without_an_amide(tmp_path):
    """Chinolin has no amide, so the omega rule needed no SDF for it. Its ring bonds do."""
    from md_tools.openmm.system import UnclassifiedTorsionError, unscaled_torsions

    topology, solute, _sdfs, _ = _ligand(tmp_path, "c1ccc2ncccc2c1", name="CHI")
    result = unscaled_torsions(topology, solute, enforce=False)
    assert result["unclassified"], "no bond orders: its ring bonds cannot be named"
    assert "CHI" in result["unclassified"][0]["ambiguous"]
    with pytest.raises(UnclassifiedTorsionError, match="CHI"):
        unscaled_torsions(topology, solute)


def test_chinolin_with_its_sdf_keeps_its_eleven_ring_bonds(tmp_path):
    from md_tools.openmm.system import unscaled_torsions

    topology, solute, sdfs, _ = _ligand(tmp_path, "c1ccc2ncccc2c1", name="CHI")
    assert len(_by_class(unscaled_torsions(topology, solute,
                                           residue_sdfs=sdfs))["aromatic_ring"]) == 11


# --- proteins, from a residue table -------------------------------------------------------------

def _residue_templates():
    path = os.path.join(os.path.dirname(app.__file__), "data", "residues.xml")
    return {res.get("name"): res for res in ET.parse(path).getroot().findall("Residue")}


def test_every_bond_in_the_protein_table_exists_in_openmms_templates():
    """An atom name typed wrong would exempt nothing, silently. So the table is checked."""
    from md_tools.openmm.system import PROTEIN_UNSCALED_BONDS

    templates = _residue_templates()
    for residue, classes in PROTEIN_UNSCALED_BONDS.items():
        template = templates["HIS" if residue in ("HID", "HIE", "HIP") else residue]
        bonded = {frozenset((b.get("from"), b.get("to"))) for b in template.findall("Bond")}
        for kind, pairs in classes.items():
            for pair in pairs:
                assert frozenset(pair) in bonded, f"{residue} {kind} {pair} is not a bond"


def test_the_protein_table_is_the_decided_one():
    from md_tools.openmm.system import PROTEIN_UNSCALED_BONDS

    counts = {name: {kind: len(pairs) for kind, pairs in classes.items()}
              for name, classes in PROTEIN_UNSCALED_BONDS.items()}
    assert counts["PHE"] == {"aromatic_ring": 6}
    assert counts["TYR"] == {"aromatic_ring": 6}
    assert counts["TRP"] == {"aromatic_ring": 10}
    for his in ("HIS", "HID", "HIE", "HIP"):
        assert counts[his] == {"aromatic_ring": 5}
    assert counts["ARG"] == {"double_bond": 3}


def _peptide(names):
    """Each residue's atoms from OpenMM's templates, hydrogens included, chained C(i)-N(i+1).

    Hydrogens matter: a torsion runs across ARG's CZ-NH1 only because NH1 carries hydrogens.
    """
    templates = _residue_templates()
    topology = app.Topology()
    chain = topology.addChain()
    previous_c = None
    for name in names:
        template = templates[name]
        residue = topology.addResidue(name, chain)
        atoms = {}
        for bond in template.findall("Bond"):
            for label in (bond.get("from"), bond.get("to")):
                if label.startswith("-") or label in ("OXT", "HXT", "H2", "H3") or label in atoms:
                    continue
                atoms[label] = topology.addAtom(label, _SYMBOLS[label[0]], residue)
        for bond in template.findall("Bond"):
            a, b = bond.get("from"), bond.get("to")
            if a in atoms and b in atoms:
                topology.addBond(atoms[a], atoms[b])
        if previous_c is not None:
            topology.addBond(previous_c, atoms["N"])
        previous_c = atoms["C"]
    return topology


def test_a_peptide_keeps_its_rings_its_guanidinium_and_its_backbone_amides():
    from md_tools.openmm.system import unscaled_torsions

    topology = _peptide(["PHE", "TYR", "TRP", "HIS", "ARG", "PRO", "ALA"])
    result = unscaled_torsions(topology, [a.index for a in topology.atoms()])
    classes = _by_class(result)
    assert len(classes["aromatic_ring"]) == 6 + 6 + 10 + 5
    assert len(classes["double_bond"]) == 3
    # PHE-TYR, TYR-TRP, TRP-HIS, HIS-ARG and PRO-ALA are ordinary; ARG-PRO is proline-like.
    assert len(classes["amide_omega"]) == 5
    assert len(result["proline_like_scaled_bonds"]) == 1
    assert result["unclassified"] == []


# --- the Hamiltonian ------------------------------------------------------------------------------

def _ala(tmp_path):
    from openmm import XmlSerializer

    build = make_dataset_root(tmp_path) / "build"
    return (XmlSerializer.deserialize((build / "built.xml").read_text(encoding="utf-8")),
            app.PDBFile(str(build / "built.pdb")).topology)


def _terms(system):
    from openmm import PeriodicTorsionForce, unit

    for force in system.getForces():
        if isinstance(force, PeriodicTorsionForce):
            return [(tuple(force.getTorsionParameters(i)[:4]),
                     force.getTorsionParameters(i)[6].value_in_unit(unit.kilojoule_per_mole))
                    for i in range(force.getNumTorsions())]
    return []


def test_every_improper_keeps_its_force_constant_at_every_tau(tmp_path):
    from md_tools.rest2.hamiltonian import build_scaled_system, is_improper, system_bond_graph

    base, _topology = _ala(tmp_path)
    solute = range(base.getNumParticles())
    bonds = system_bond_graph(base)
    before = _terms(base)
    impropers = [n for n, (atoms, _k) in enumerate(before) if is_improper(atoms, bonds)]
    assert impropers, "ff14SB ACE-ALA-NME carries improper terms; the fixture must exercise them"

    after = _terms(build_scaled_system(base, solute, 0.5, excluded_bonds=()))
    for n in impropers:
        assert after[n][1] == pytest.approx(before[n][1], rel=0, abs=0), f"improper {n} scaled"
    proper = [n for n in range(len(before)) if n not in impropers and before[n][1] != 0.0]
    assert all(after[n][1] == pytest.approx(0.25 * before[n][1], rel=1e-12) for n in proper)


def test_an_improper_is_found_by_the_bond_graph_not_by_atom_order(tmp_path):
    from md_tools.rest2.hamiltonian import is_improper

    bonds = {frozenset(p) for p in [(0, 1), (1, 2), (2, 3), (1, 4), (1, 5)]}
    assert not is_improper((0, 1, 2, 3), bonds), "a bonded chain is a proper torsion"
    assert is_improper((4, 1, 0, 5), bonds), "centre second (SMIRNOFF)"
    assert is_improper((0, 4, 1, 5), bonds), "centre third (Amber)"


def test_a_torsion_the_bonds_do_not_explain_is_refused_not_left_unscaled():
    """A System with an incomplete bond force would make every proper look improper."""
    import openmm

    from md_tools.rest2.hamiltonian import build_scaled_system, torsion_kind

    assert torsion_kind((0, 1, 2, 3), {frozenset((0, 1))}) is None
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)               # 1-2 and 2-3 missing
    system.addForce(bonds)
    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 2, 0.0, 10.0)
    system.addForce(torsions)
    with pytest.raises(ValueError, match="neither a bonded chain"):
        build_scaled_system(system, range(4), 0.5)


def test_the_switch_off_scales_impropers_too(tmp_path):
    from md_tools.rest2.hamiltonian import build_scaled_system, is_improper, system_bond_graph

    base, _topology = _ala(tmp_path)
    bonds = system_bond_graph(base)
    before = _terms(base)
    after = _terms(build_scaled_system(base, range(base.getNumParticles()), 0.5,
                                       excluded_bonds=(), unscaled_impropers=False))
    for n, (atoms, k) in enumerate(before):
        if is_improper(atoms, bonds):
            assert after[n][1] == pytest.approx(0.25 * k, rel=1e-12)


def test_the_report_counts_the_impropers_it_left_alone(tmp_path):
    from md_tools.rest2.hamiltonian import is_improper, system_bond_graph, torsion_exclusion_report

    base, _topology = _ala(tmp_path)
    bonds = system_bond_graph(base)
    report = torsion_exclusion_report(base, range(base.getNumParticles()), [])
    expected = sum(1 for atoms, _k in _terms(base) if is_improper(atoms, bonds))
    assert report["n_unscaled_impropers"] == expected
    assert len(report["unscaled_improper_indices"]) == expected


def test_the_convention_changed_version_and_the_old_one_cannot_be_continued():
    from md_tools.rest2 import REST2_IMPLEMENTATION, require_compatible_implementation

    assert REST2_IMPLEMENTATION["name"] == "rest2-unscaled-torsions"
    assert REST2_IMPLEMENTATION["version"] == 3
    assert REST2_IMPLEMENTATION["unscaled_torsions"] == [
        "ordinary amide omega", "aromatic ring bonds", "other double bonds", "impropers"]
    with pytest.raises(ValueError, match="rest2-no-bond-angle-omega/v2"):
        require_compatible_implementation({"name": "rest2-no-bond-angle-omega", "version": 2})
