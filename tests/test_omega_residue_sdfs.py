"""One SDF PER RESIDUE NAME: `classify_omega_bonds(..., residue_sdfs={name: path})`.

`ligand_sdf` maps ONE SDF onto every non-standard solute residue at once. That is right for what
`build-top` builds from a `.smi`/`.sdf` -- one molecule, one residue -- and wrong as soon as a solute
carries two different non-standard residues: no single SDF describes both, so the mapping fails on
atom count and every candidate is refused for a reason that says nothing about which file is
missing. `build-top --rest2-scaler` (docs/amber-like-fix/REST2-scaler.md §5) supplies a map
instead, and each SDF is mapped onto the atoms of each INSTANCE of its residue, separately.

The rule per candidate is unchanged (`test_omega_evidence_by_residue.py`): a declared proline-like
name wins, a known protein residue is read from the residue, anything else from bond orders, and
with no evidence it is unclassified. This file covers only where the bond orders come from.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping and RDKit perception. No force field, no dynamics.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.system import classify_omega_bonds

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")

#: An ordinary amide, no ring at the nitrogen.
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
#: N-acetylpyrrolidine: the amide nitrogen sits in a 5-membered ring, so it is PROLINE-LIKE.
ACETYLPYRROLIDINE = "CC(=O)N1CCCC1"

_SYMBOLS = {"C": elem.carbon, "N": elem.nitrogen, "O": elem.oxygen, "H": elem.hydrogen}


def _molecule(smiles):
    rdkit = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=20260916)
    return mol


def _system(directory, residues):
    """A topology holding one residue per (name, smiles) entry, each in its own chain, plus one SDF
    per distinct NAME written from the same RDKit molecule the residue was built from."""
    rdkit = pytest.importorskip("rdkit.Chem")
    topology = app.Topology()
    sdfs, atoms_of = {}, []
    molecules = {}
    for name, smiles in residues:
        mol = molecules.setdefault(name, _molecule(smiles))
        residue = topology.addResidue(name, topology.addChain())
        atoms = [topology.addAtom(a.GetSymbol(), _SYMBOLS[a.GetSymbol()], residue)
                 for a in mol.GetAtoms()]
        for bond in mol.GetBonds():
            topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
        atoms_of.append([a.index for a in atoms])
        if name not in sdfs:
            sdfs[name] = directory / f"{name}.sdf"
            rdkit.MolToMolFile(mol, str(sdfs[name]))
    solute = [a.index for a in topology.atoms()]
    return topology, solute, sdfs, atoms_of


def _within(bonds, atoms):
    return [b for b in bonds if all(i in atoms for i in b)]


def test_each_residue_is_read_from_its_own_sdf(tmp_path):
    topology, solute, sdfs, (mo1, mo2) = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    result = classify_omega_bonds(topology, solute, residue_sdfs=sdfs)

    assert result["omega_unclassified_candidates"] == []
    assert len(_within(result["omega_unscaled_bonds"], mo1)) == 1, "paracetamol: ordinary"
    assert len(_within(result["omega_proline_like_scaled_bonds"], mo2)) == 1, (
        "acetylpyrrolidine: ring-locked nitrogen, proline-like")


def test_the_whole_set_mapping_cannot_do_this(tmp_path):
    """Why the map exists: one SDF over two different residues fails, for both of them."""
    topology, solute, sdfs, _ = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    result = classify_omega_bonds(topology, solute, ligand_sdf=sdfs["MO1"])
    assert len(result["omega_unclassified_candidates"]) == 2


def test_a_residue_missing_from_the_map_is_unclassified_by_name(tmp_path):
    topology, solute, sdfs, (mo1, mo2) = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    result = classify_omega_bonds(topology, solute, residue_sdfs={"MO1": sdfs["MO1"]})

    unknown = result["omega_unclassified_candidates"]
    assert len(unknown) == 1 and unknown[0]["nitrogen_residue"] == "MO2"
    assert "MO2" in unknown[0]["ambiguous"] and "SDF" in unknown[0]["ambiguous"]
    assert len(_within(result["omega_unscaled_bonds"], mo1)) == 1, (
        "the residue that HAS an SDF is still decided")


def test_swapped_sdfs_are_refused_not_believed(tmp_path):
    topology, solute, sdfs, _ = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    swapped = {"MO1": sdfs["MO2"], "MO2": sdfs["MO1"]}
    result = classify_omega_bonds(topology, solute, residue_sdfs=swapped)

    assert len(result["omega_unclassified_candidates"]) == 2
    assert all("mismatch" in c["ambiguous"] for c in result["omega_unclassified_candidates"])


def test_every_instance_of_a_residue_is_mapped_separately(tmp_path):
    """Two copies of MO1: one SDF, mapped onto each copy's own atoms. Mapping it onto both at once
    would fail on atom count."""
    topology, solute, sdfs, (first, second) = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO1", PARACETAMOL)])
    result = classify_omega_bonds(topology, solute, residue_sdfs=sdfs)

    assert result["omega_unclassified_candidates"] == []
    assert len(_within(result["omega_unscaled_bonds"], first)) == 1
    assert len(_within(result["omega_unscaled_bonds"], second)) == 1


def test_a_declared_proline_like_name_needs_no_sdf(tmp_path):
    topology, solute, sdfs, (mo1, mo2) = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    result = classify_omega_bonds(topology, solute, residue_sdfs={"MO1": sdfs["MO1"]},
                                  proline_like_residues=("PRO", "MO2"))
    assert result["omega_unclassified_candidates"] == []
    assert len(_within(result["omega_proline_like_scaled_bonds"], mo2)) == 1


def test_the_method_names_which_sdf_described_which_residue(tmp_path):
    topology, solute, sdfs, _ = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    method = classify_omega_bonds(topology, solute, residue_sdfs=sdfs)["omega_detection_method"]
    assert "MO1.sdf" in method and "MO2.sdf" in method, method


def test_both_kinds_of_evidence_at_once_is_a_caller_error(tmp_path):
    topology, solute, sdfs, _ = _system(tmp_path, [("MO1", PARACETAMOL)])
    with pytest.raises(ValueError, match="residue_sdfs"):
        classify_omega_bonds(topology, solute, ligand_sdf=sdfs["MO1"], residue_sdfs=sdfs)


def test_the_enforcing_entry_point_takes_the_same_evidence(tmp_path):
    from md_tools.openmm.system import UnclassifiedOmegaError, omega_exclusions

    topology, solute, sdfs, _ = _system(
        tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    decided = omega_exclusions(topology, solute, residue_sdfs=sdfs)
    assert decided["omega_unclassified_candidates"] == []
    with pytest.raises(UnclassifiedOmegaError, match="MO2"):
        omega_exclusions(topology, solute, residue_sdfs={"MO1": sdfs["MO1"]})
    # and the proline-like declaration reaches the classifier through it
    omega_exclusions(topology, solute, residue_sdfs={"MO1": sdfs["MO1"]},
                     proline_like_residues=("PRO", "MO2"))
