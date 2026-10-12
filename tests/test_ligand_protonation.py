"""Open Babel's pH model, applied to a small molecule BEFORE it is parameterised.

WHAT IS ASSERTED, AND WHY EACH ONE CAN FAIL

  * the species at a pH: an acid is an anion at 7 and neutral at 2, an amine the reverse, and a
    molecule the model has no rule for comes back unchanged -- each is a different input with a
    different expected answer, so a function that returned its input (or always ionised) fails;
  * a result that is a different MOLECULE is refused: the guard is exercised with an Open Babel
    stand-in that returns one, because a real run never produces it;
  * a 3D input keeps its heavy atoms where they were and returns the SAME OBJECT when nothing
    changed;
  * `none` is a recorded choice and changes nothing; Open Babel missing is an error, not `none`.

The pH model is a table, not a pKa predictor. The test of that sentence is that a phosphate and an
imidazole are left alone, which is the documented limit rather than an endorsement of it.

PLATFORM_POLICY_EXEMPTION: RDKit and Open Babel only. No force field, no dynamics.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openbabel")
Chem = pytest.importorskip("rdkit.Chem")

from md_tools.openmm import ligand_protonation as lp  # noqa: E402


@pytest.mark.parametrize("smiles, ph, expected", [
    ("CC(=O)O", 7.0, "CC(=O)[O-]"),
    ("CC(=O)O", 2.0, "CC(=O)O"),
    ("CCN", 7.0, "CC[NH3+]"),
    ("CCN", 12.0, "CCN"),
    ("NC(=N)N", 7.0, "NC(N)=[NH2+]"),
    ("CC(=O)Nc1ccc(O)cc1", 7.0, "CC(=O)Nc1ccc(O)cc1"),       # no rule applies: unchanged
    ("OP(=O)(O)O", 7.0, "OP(=O)(O)O"),                       # the documented limit of a table
])
def test_the_species_at_the_requested_ph(smiles, ph, expected):
    out, record = lp.protonate_smiles(smiles, ph)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    assert record["method"] == "openbabel" and record["ph"] == ph
    assert record["changed"] is (Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                                 != Chem.MolToSmiles(Chem.MolFromSmiles(expected)))
    assert record["openbabel_version"]


def test_a_molecule_that_needed_no_change_keeps_the_exact_string_it_was_written_with():
    out, record = lp.protonate_smiles("c1ccccc1O", 7.0)
    assert out == "c1ccccc1O" and record["changed"] is False and record["smiles_out"] == out


def test_stereochemistry_and_a_double_bond_survive_the_pH_model():
    out, _ = lp.protonate_smiles("C[C@H](N)C(=O)O", 7.0)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == "C[C@H]([NH3+])C(=O)[O-]"
    out, _ = lp.protonate_smiles("C/C=C/C(=O)O", 7.0)
    assert "/C=C/" in out or "\\C=C\\" in out, out


def test_none_is_a_recorded_choice_and_changes_nothing():
    out, record = lp.protonate_smiles("CC(=O)O", 7.0, method="none")
    assert out == "CC(=O)O"
    assert record["method"] == "none" and record["changed"] is False


def _embedded(smiles):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=20261012)
    return mol


def test_a_3d_molecule_changes_only_hydrogens_and_keeps_its_heavy_atoms_in_place():
    mol = _embedded("CC(=O)O")
    out, record = lp.protonate_molecule(mol, 7.0, route="sdf")
    assert record["changed"] is True and out is not mol
    assert Chem.GetFormalCharge(out) == -1
    assert record["net_formal_charge"] == [0, -1]
    assert record["changes"] == [{"heavy_atom": 3, "element": "O", "formal_charge": [0, -1],
                                  "n_hydrogens": [1, 0]}]
    heavy_in = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    heavy_out = [a.GetIdx() for a in out.GetAtoms() if a.GetAtomicNum() > 1]
    moved = np.abs(mol.GetConformer().GetPositions()[heavy_in]
                   - out.GetConformer().GetPositions()[heavy_out]).max()
    assert moved < 1e-3


def test_a_3d_molecule_that_needed_no_change_comes_back_as_the_same_object():
    mol = _embedded("CC(=O)Nc1ccc(O)cc1")
    out, record = lp.protonate_molecule(mol, 7.0, route="mol2")
    assert out is mol and record["changed"] is False and record["input_route"] == "mol2"


def test_an_amine_gains_a_hydrogen_and_a_zwitterion_comes_out_with_both_charges():
    out, record = lp.protonate_molecule(_embedded("C[C@H](N)C(=O)O"), 7.0, route="sdf")
    assert sorted(a.GetFormalCharge() for a in out.GetAtoms() if a.GetFormalCharge()) == [-1, 1]
    assert record["net_formal_charge"] == [0, 0] and record["changed"] is True


class _Different:
    """Stand-in for a pH model that returns another MOLECULE. A real Open Babel never does, so the
    only way to know the guard can fail is to hand it one that does."""


def test_a_result_with_another_heavy_atom_graph_is_refused():
    before = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    after = Chem.AddHs(Chem.MolFromSmiles("CCC(=O)O"))
    with pytest.raises(lp.LigandProtonationError, match="heavy-atom graph"):
        lp._require_same_molecule(before, after, what="a test molecule")


def test_a_result_that_loses_a_stereocentre_is_refused():
    before = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    after = Chem.AddHs(Chem.MolFromSmiles("CC(N)C(=O)O"))
    with pytest.raises(lp.LigandProtonationError, match="stereo"):
        lp._require_same_molecule(before, after, what="a test molecule")


def test_the_same_molecule_in_two_protomers_passes_the_guard():
    lp._require_same_molecule(Chem.AddHs(Chem.MolFromSmiles("CC(=O)O")),
                              Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]")), what="acetic acid")


def test_missing_openbabel_is_an_error_naming_the_way_out_and_never_a_silent_none(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "openbabel" or name.startswith("openbabel."):
            raise ImportError("no openbabel here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(lp.LigandProtonationError, match="ligand_method: none"):
        lp.protonate_smiles("CC(=O)O", 7.0)
    with pytest.raises(lp.LigandProtonationError, match="Nothing was built"):
        lp.protonate_molecule(_embedded("CC(=O)O"), 7.0, route="sdf")
    # and `none` needs no Open Babel at all
    assert lp.protonate_smiles("CC(=O)O", 7.0, method="none")[0] == "CC(=O)O"


def test_the_summary_says_what_happened_in_one_line():
    _, record = lp.protonate_smiles("CC(=O)O", 7.0)
    line = lp.protonation_summary(record)
    assert "pH 7.0" in line and "+0 -> -1" in line and "CC(=O)[O-]" in line
    assert "exactly as its input encodes it" in lp.protonation_summary(
        lp.protonate_smiles("CC(=O)O", 7.0, method="none")[1])
