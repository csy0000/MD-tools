"""The molecular map: what a residue IS, read from the graph rather than from a name.

WHY INDEX AND NAME INDEPENDENCE IS TESTED SO HARD

    The whole reason this module exists is that residue and atom NAMES are absent from a
    SMILES-built solute, so anything keyed on them silently does nothing. A mapper that quietly
    depended on atom ORDER instead would have replaced one hidden assumption with another --
    and the failure would look identical: plausible residues, wrong atoms, corrections applied
    to whatever happened to sit at those indices.

    So the same molecule is mapped again with its atoms renumbered, and the answer has to be the
    same chemistry pointing at the correspondingly moved atoms.

WHAT MUST BE REFUSED

    Refusals are as much the contract as the successes. A partial map is worse than none: it
    would correct some residues' radii and silently not others, producing a Hamiltonian that is
    neither the corrected one nor the baseline.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.peptide_map import PeptideMapError, map_cyclic_peptide

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem                                                    # noqa: E402

#: Head-to-tail cyclo(Gly-L-Asp-L-Arg): both mbondi3-correctable groups, 43 atoms, and small
#: enough that everything here runs in milliseconds.
CYCLO_GDR = "O=C1NCC(=O)N[C@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]"
#: The RGDfV integration case.
CYCLO_RGDFV = ("CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)"
               "[C@H](CCCNC(N)=[NH2+])NC1=O")


def _mapped(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, map_cyclic_peptide(mol)


def _sequence(mapped):
    return {(r.residue, r.handedness) for r in mapped.residues}


# --- the integration case ---------------------------------------------------------------------

def test_rgdfv_maps_to_five_residues_with_d_phenylalanine():
    _mol, mapped = _mapped(CYCLO_RGDFV)
    assert len(mapped.residues) == 5
    assert _sequence(mapped) == {("ARG", "L"), ("GLY", None), ("ASP", "L"),
                                 ("PHE", "D"), ("VAL", "L")}
    assert mapped.formal_charge == 0


def test_rgdfv_has_five_cyclic_links_including_the_closure():
    mol, mapped = _mapped(CYCLO_RGDFV)
    assert len(mapped.links) == 5
    for carbon, nitrogen in mapped.links:
        assert mol.GetBondBetweenAtoms(carbon, nitrogen) is not None
    # Every residue is entered once and left once: that is what makes it a single cycle rather
    # than a branch that happens to close somewhere.
    assert len({c for c, _n in mapped.links}) == 5
    assert len({n for _c, n in mapped.links}) == 5


def test_rgdfv_presents_exactly_two_carboxylate_oxygens_and_five_guanidinium_hydrogens():
    """The counts the correction depends on, from chemistry alone."""
    mol, mapped = _mapped(CYCLO_RGDFV)
    assert len(mapped.carboxylate_oxygens) == 2
    assert len(mapped.guanidinium_hydrogens) == 5
    for index in mapped.carboxylate_oxygens:
        assert mol.GetAtomWithIdx(index).GetSymbol() == "O"
    for index in mapped.guanidinium_hydrogens:
        atom = mol.GetAtomWithIdx(index)
        assert atom.GetSymbol() == "H"
        assert atom.GetNeighbors()[0].GetSymbol() == "N"


def test_rgdfv_yields_fifteen_cyclic_torsions():
    _mol, mapped = _mapped(CYCLO_RGDFV)
    torsions = mapped.torsions()
    assert len(torsions) == 15
    assert sum(1 for t in torsions if t["kind"] == "omega") == 5
    for torsion in torsions:
        assert len(set(torsion["atoms"])) == 4


def test_the_mapped_omega_bonds_agree_with_the_existing_classifier_by_construction():
    """Cross-check: the map's cyclic links must be the same bonds omega torsions turn about."""
    _mol, mapped = _mapped(CYCLO_RGDFV)
    from_links = {frozenset(link) for link in mapped.links}
    from_torsions = {frozenset(t["atoms"][1:3]) for t in mapped.torsions()
                     if t["kind"] == "omega"}
    assert from_links == from_torsions


# --- index and name independence ----------------------------------------------------------------

def test_the_map_does_not_depend_on_atom_order():
    """Renumber every atom and the chemistry must be unchanged, pointing at the moved atoms."""
    mol = Chem.AddHs(Chem.MolFromSmiles(CYCLO_RGDFV))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    original = map_cyclic_peptide(mol)

    order = list(range(mol.GetNumAtoms()))
    # A deterministic, thorough permutation: reversal moves every atom.
    order.reverse()
    permuted = Chem.RenumberAtoms(mol, order)
    Chem.AssignStereochemistry(permuted, cleanIt=True, force=True)
    moved = map_cyclic_peptide(permuted)

    assert _sequence(moved) == _sequence(original)
    assert moved.formal_charge == original.formal_charge
    # `order[new] = old`, so the atom that was `old` now sits at `order.index(old)`.
    where = {old: new for new, old in enumerate(order)}
    assert set(moved.carboxylate_oxygens) == {where[i] for i in original.carboxylate_oxygens}
    assert set(moved.guanidinium_hydrogens) == {where[i] for i in original.guanidinium_hydrogens}


def test_the_map_ignores_residue_and_atom_names_entirely():
    """Names are set to something misleading; the map must not change.

    A SMILES-built solute has one made-up residue name, and the failure this whole change fixes
    was caused by trusting exactly such a name.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(CYCLO_GDR))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    before = map_cyclic_peptide(mol)

    misleading = Chem.Mol(mol)
    for atom in misleading.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("ARG")            # every atom claims to be arginine
        info.SetName(" HH11")                 # and to be a guanidinium hydrogen
        info.SetResidueNumber(1)
        atom.SetMonomerInfo(info)
    Chem.AssignStereochemistry(misleading, cleanIt=True, force=True)
    after = map_cyclic_peptide(misleading)

    assert _sequence(after) == _sequence(before)
    assert after.carboxylate_oxygens == before.carboxylate_oxygens
    assert after.guanidinium_hydrogens == before.guanidinium_hydrogens


# --- residue coverage ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # cyclo(Gly-L-Glu-L-Ala): glutamate, one methylene longer than aspartate.
    ("O=C1NCC(=O)N[C@H](CCC(=O)[O-])C(=O)N[C@H]1C", ("GLY", "GLU", "ALA")),
    # cyclo(Gly-L-Pro-L-Ala): proline's ring closes onto its own backbone nitrogen.
    ("O=C1NCC(=O)N2CCC[C@H]2C(=O)N[C@H]1C", ("GLY", "PRO", "ALA")),
    # cyclo(Gly-L-Cys-L-Ala): the L = R exception.
    ("O=C1NCC(=O)N[C@H](CS)C(=O)N[C@H]1C", ("GLY", "CYS", "ALA")),
])
def test_compact_fixtures_map_their_residues(smiles, expected):
    _mol, mapped = _mapped(smiles)
    assert tuple(sorted(mapped.sequence)) == tuple(sorted(expected))


@pytest.mark.parametrize("smiles", [
    "O=C1NCC(=O)N[C@H](CS)C(=O)N[C@H]1C",
    "O=C1NCC(=O)N[C@@H](CS)C(=O)N[C@H]1C",
])
def test_cysteine_inverts_the_cip_to_handedness_mapping_and_alanine_does_not(smiles):
    """L-Cys is R while L-Ala is S, and the mapper must not use one table for both.

    Asserted as the RELATION between the CIP code and the reported label, for both enantiomers,
    rather than by hand-deriving which SMILES is the L form -- that derivation is exactly the
    error the test exists to catch, so making the test depend on it would prove nothing.
    """
    mol, mapped = _mapped(smiles)
    cysteine = next(r for r in mapped.residues if r.residue == "CYS")
    alanine = next(r for r in mapped.residues if r.residue == "ALA")

    cys_cip = mol.GetAtomWithIdx(cysteine.ca).GetPropsAsDict().get("_CIPCode")
    ala_cip = mol.GetAtomWithIdx(alanine.ca).GetPropsAsDict().get("_CIPCode")

    assert cysteine.handedness == {"R": "L", "S": "D"}[cys_cip]
    assert alanine.handedness == {"S": "L", "R": "D"}[ala_cip]
    # And the two tables really are different: same letter, opposite label.
    if cys_cip == ala_cip:
        assert cysteine.handedness != alanine.handedness


def test_glycine_is_achiral_and_reported_as_such():
    _mol, mapped = _mapped(CYCLO_RGDFV)
    glycine = next(r for r in mapped.residues if r.residue == "GLY")
    assert glycine.handedness is None


def test_d_and_l_of_the_same_residue_are_distinguished():
    _mol, left = _mapped("O=C1NCC(=O)N[C@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]")
    _mol2, right = _mapped("O=C1NCC(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]")
    assert next(r.handedness for r in left.residues if r.residue == "ASP") != \
        next(r.handedness for r in right.residues if r.residue == "ASP")


# --- neutral controls ---------------------------------------------------------------------------

def test_a_neutral_carboxylic_acid_is_not_a_carboxylate_and_is_not_corrected():
    """`ASH` is its own identity. ParmEd's NAME rule would correct `AS4`; chemistry does not."""
    _mol, mapped = _mapped("O=C1NCC(=O)N[C@H](CC(=O)O)C(=O)N[C@H]1C")
    assert "ASH" in mapped.sequence and "ASP" not in mapped.sequence
    assert mapped.carboxylate_oxygens == ()


def test_a_neutral_guanidine_is_not_a_guanidinium_and_is_not_corrected():
    _mol, mapped = _mapped("O=C1NCC(=O)N[C@H](C)C(=O)N[C@H]1CCCNC(N)=N")
    assert "ARN" in mapped.sequence and "ARG" not in mapped.sequence
    assert mapped.guanidinium_hydrogens == ()


def test_a_neutral_peptide_has_nothing_to_correct():
    _mol, mapped = _mapped("O=C1NCC(=O)N[C@H](C)C(=O)N[C@H]1C")
    assert mapped.carboxylate_oxygens == ()
    assert mapped.guanidinium_hydrogens == ()
    assert mapped.formal_charge == 0


# --- refusals ------------------------------------------------------------------------------------

def test_a_linear_peptide_is_refused_rather_than_partially_mapped():
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    with pytest.raises(PeptideMapError) as refusal:
        map_cyclic_peptide(mol)
    message = str(refusal.value)
    assert "LINEAR" in message or "head-to-tail" in message
    assert "solute.kind: ligand" in message


def test_an_unsupported_side_chain_is_refused_by_name_of_the_residue():
    """A modified side chain must say which residue and offer the ligand route."""
    # cyclo(Gly-Ala-X) where X carries a nitro group: not a canonical side chain.
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C1NCC(=O)N[C@H](C[N+](=O)[O-])C(=O)N[C@H]1C"))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    with pytest.raises(PeptideMapError) as refusal:
        map_cyclic_peptide(mol)
    message = str(refusal.value)
    assert "residue" in message
    assert "solute.kind: ligand" in message


def test_unspecified_stereochemistry_is_refused_rather_than_guessed():
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C1NCC(=O)NC(C)C(=O)N[C@H]1C"))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    with pytest.raises(PeptideMapError) as refusal:
        map_cyclic_peptide(mol)
    assert "stereochemistry" in str(refusal.value)


def test_a_molecule_that_is_not_a_peptide_is_refused():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    with pytest.raises(PeptideMapError) as refusal:
        map_cyclic_peptide(mol)
    assert "amide" in str(refusal.value) or "peptide" in str(refusal.value)


def test_two_disconnected_molecules_are_refused():
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C1NCC(=O)N[C@H](C)C(=O)N[C@H]1C.O"))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    with pytest.raises(PeptideMapError) as refusal:
        map_cyclic_peptide(mol)
    assert "fragment" in str(refusal.value)


# --- provenance -----------------------------------------------------------------------------------

def test_the_digest_is_stable_for_the_same_molecule_and_changes_with_the_chemistry():
    _mol, one = _mapped(CYCLO_RGDFV)
    _mol2, two = _mapped(CYCLO_RGDFV)
    assert one.digest() == two.digest()

    _mol3, neutral = _mapped("O=C1NCC(=O)N[C@H](CC(=O)O)C(=O)N[C@H]1C")
    assert neutral.digest() != one.digest()
