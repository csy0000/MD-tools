"""WHICH EVIDENCE the omega classifier uses, decided per candidate from the residue name.

`test_omega_classification_cases.py` covers what the classifier DECIDES. This file covers how it
decides which evidence to decide from, which used to be a `route` argument every caller had to
supply and which four of them supplied wrongly.

THE DEFECT THAT PRODUCED THIS FILE. A REST2 ladder over paracetamol -- built from a `.sdf`, one
`UNL` residue, `built.sdf` sitting beside the System -- was refused on CUDA with

    1 amide candidate(s) could not be classified as ordinary or proline-like
    nitrogen residue 'UNL' is neither a known protein residue nor listed in
    rest2.proline_like_residues, so THE PEPTIDE ROUTE cannot say ...

The peptide route, for a ligand, with the SDF one directory away. `run/preflight.py` (twice),
`reference/rest2_export.py` and `rest2/selection.py` passed `route="peptide"` or nothing at all,
while `build/rungs.py` and `remd/generated.py` threaded the real route through -- so the rung
WRITER and the rung VALIDATOR disagreed about the same ladder, and the refusing half ran.
`build/rungs.py`'s own comment had predicted it: "a second derivation is two answers waiting to
disagree, and the disagreement would be invisible".

THE RULE, and it is per CANDIDATE rather than per run, which is what makes a mixed protein+ligand
system classifiable at all:

  1. the amide NITROGEN's residue is a known protein residue (or a declared proline-like name)
     -> decide from the residue, as before;
  2. otherwise -> decide from the SDF's bond orders, with the ring-size bound;
  3. otherwise (no SDF) -> REFUSE, naming what is missing.

Step 3 is deliberate. Falling back to "it looks like an amide" for an unknown residue would
silently change the Hamiltonian of any protein carrying a modified residue, which is the exact
class of error the unclassified list exists to stop.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping and RDKit perception. No force field, no dynamics.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.system import classify_omega_bonds

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")

#: Paracetamol. One amide, an aromatic ring, a hydroxyl -- and not a peptide, so the residue route
#: has nothing to say about it. This is the molecule the CUDA ladder refused.
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"

_SYMBOLS = {"C": elem.carbon, "N": elem.nitrogen, "O": elem.oxygen, "H": elem.hydrogen}


def _molecule(smiles):
    rdkit = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
    assert mol is not None, smiles
    AllChem.EmbedMolecule(mol, randomSeed=20260916)
    return mol


def _write_sdf(mol, path):
    rdkit = pytest.importorskip("rdkit.Chem")
    rdkit.MolToMolFile(mol, str(path))
    return path


def _ligand_only(smiles, residue_name, directory):
    """A one-residue topology and the SDF describing it, from ONE RDKit molecule.

    Element order and bond graph agree by construction, which is what the classifier asserts
    before it will trust an SDF against a topology.
    """
    mol = _molecule(smiles)
    topology = app.Topology()
    residue = topology.addResidue(residue_name, topology.addChain())
    atoms = [topology.addAtom(a.GetSymbol(), _SYMBOLS[a.GetSymbol()], residue)
             for a in mol.GetAtoms()]
    for bond in mol.GetBonds():
        topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    sdf = _write_sdf(mol, directory / f"{residue_name.lower()}.sdf")
    return topology, [a.index for a in atoms], sdf


def _protein_plus_ligand(smiles, residue_name, directory):
    """ACE-ALA-NME in one chain and a ligand residue in another, with an SDF for the LIGAND ONLY.

    The mixed case is the reason the decision is per candidate: the protein's two omegas have
    residue evidence and the ligand's has none, and no single route is right for both. It is also
    the case a whole-solute SDF mapping cannot serve -- the SDF describes a fraction of the solute.
    """
    topology = app.Topology()
    chain = topology.addChain()
    named = {}

    def residue(name, atoms):
        res = topology.addResidue(name, chain)
        for label, symbol in atoms.items():
            named[label] = topology.addAtom(label, _SYMBOLS[symbol], res)

    residue("ACE", {"ACE_CH3": "C", "ACE_C": "C", "ACE_O": "O"})
    residue("ALA", {"ALA_N": "N", "ALA_H": "H", "ALA_CA": "C", "ALA_CB": "C",
                    "ALA_C": "C", "ALA_O": "O"})
    residue("NME", {"NME_N": "N", "NME_H": "H", "NME_CH3": "C"})
    for a, b in (("ACE_CH3", "ACE_C"), ("ACE_C", "ACE_O"), ("ACE_C", "ALA_N"),
                 ("ALA_N", "ALA_H"), ("ALA_N", "ALA_CA"), ("ALA_CA", "ALA_CB"),
                 ("ALA_CA", "ALA_C"), ("ALA_C", "ALA_O"), ("ALA_C", "NME_N"),
                 ("NME_N", "NME_H"), ("NME_N", "NME_CH3")):
        topology.addBond(named[a], named[b])

    mol = _molecule(smiles)
    ligand_chain = topology.addChain()
    ligand_residue = topology.addResidue(residue_name, ligand_chain)
    ligand_atoms = [topology.addAtom(a.GetSymbol(), _SYMBOLS[a.GetSymbol()], ligand_residue)
                    for a in mol.GetAtoms()]
    for bond in mol.GetBonds():
        topology.addBond(ligand_atoms[bond.GetBeginAtomIdx()],
                         ligand_atoms[bond.GetEndAtomIdx()])

    sdf = _write_sdf(mol, directory / f"{residue_name.lower()}.sdf")
    solute = [a.index for a in topology.atoms()]
    return topology, solute, sdf, named, [a.index for a in ligand_atoms]


# --- 1. a ligand residue, with an SDF and no route told to anyone --------------------------------

def test_a_ligand_residue_is_classified_from_the_sdf_without_being_told_a_route(tmp_path):
    """THE REPRODUCTION, reduced to a unit test. Nobody passes a route; the residue name decides."""
    topology, solute, sdf = _ligand_only(PARACETAMOL, "UNL", tmp_path)
    result = classify_omega_bonds(topology, solute, ligand_sdf=sdf)

    assert result["omega_unclassified_candidates"] == [], (
        "paracetamol's amide has an SDF to be read from; refusing it is the defect")
    assert len(result["omega_unscaled_bonds"]) == 1, result["omega_unscaled_bonds"]
    assert result["omega_proline_like_scaled_bonds"] == []


def test_the_method_records_which_evidence_each_decision_used(tmp_path):
    """The record must say SDF, not 'peptide/residue-aware', or the audit trail is wrong."""
    topology, solute, sdf = _ligand_only(PARACETAMOL, "UNL", tmp_path)
    method = classify_omega_bonds(topology, solute, ligand_sdf=sdf)["omega_detection_method"]
    assert "SMARTS" in method or "SDF" in method, method


# --- 2. no SDF: refuse, and say what is missing --------------------------------------------------

def test_a_non_standard_residue_without_an_sdf_refuses_rather_than_guessing(tmp_path):
    """A protein carrying a modified residue has NO SDF, so there is no second opinion to take.

    Guessing "it looks like an amide" would silently change that run's Hamiltonian, which is the
    error the unclassified list exists to prevent.
    """
    topology, solute, _sdf = _ligand_only(PARACETAMOL, "UNL", tmp_path)
    result = classify_omega_bonds(topology, solute)          # deliberately no SDF

    assert result["omega_unclassified_candidates"], "no evidence must mean no decision"


def test_the_refusal_names_the_missing_sdf_so_a_reader_can_act(tmp_path):
    topology, solute, _sdf = _ligand_only(PARACETAMOL, "UNL", tmp_path)
    candidate = classify_omega_bonds(topology, solute)["omega_unclassified_candidates"][0]
    evidence = candidate["ambiguous"].lower()
    assert "sdf" in evidence, candidate["ambiguous"]
    assert "unl" in evidence or "residue" in evidence, candidate["ambiguous"]


# --- 3. a declared proline-like name still short-circuits everything -----------------------------

def test_a_declared_proline_like_name_is_honoured_before_any_sdf_is_read(tmp_path):
    """`rest2.proline_like_residues` is the human answer to the block, and it must still win --
    including when an SDF exists that would have said something else."""
    topology, solute, sdf = _ligand_only(PARACETAMOL, "UNL", tmp_path)
    result = classify_omega_bonds(topology, solute, ligand_sdf=sdf,
                                  proline_like_residues=("PRO", "UNL"))
    assert result["omega_unclassified_candidates"] == []
    assert len(result["omega_proline_like_scaled_bonds"]) == 1
    assert result["omega_unscaled_bonds"] == []


# --- 4. the mixed system, which is why the decision is per candidate -----------------------------

def test_a_mixed_system_reads_residues_for_the_protein_and_the_sdf_for_the_ligand(tmp_path):
    """Three amides: two with residue evidence, one with none, and an SDF covering only the third.

    A whole-solute SDF mapping cannot serve this -- the SDF's atom count is a fraction of the
    solute's -- so the mapping has to be restricted to the residues the SDF actually describes.
    """
    topology, solute, sdf, named, ligand_indices = _protein_plus_ligand(
        PARACETAMOL, "UNL", tmp_path)
    result = classify_omega_bonds(topology, solute, ligand_sdf=sdf)

    assert result["omega_unclassified_candidates"] == [], (
        "every amide here has evidence available: two residue-named, one in the SDF")
    bonds = {tuple(sorted(pair)) for pair in result["omega_unscaled_bonds"]}
    protein = {tuple(sorted((named["ACE_C"].index, named["ALA_N"].index))),
               tuple(sorted((named["ALA_C"].index, named["NME_N"].index)))}
    assert protein <= bonds, "the protein's two backbone omegas must still be protected"
    ligand_bonds = {b for b in bonds if all(i in ligand_indices for i in b)}
    assert len(ligand_bonds) == 1, "the ligand's amide must be protected too"


def test_a_mixed_system_still_refuses_the_ligand_when_no_sdf_describes_it(tmp_path):
    """The protein half is decidable and the ligand half is not, so the run must stop."""
    topology, solute, _sdf, named, _ligand = _protein_plus_ligand(
        PARACETAMOL, "UNL", tmp_path)
    result = classify_omega_bonds(topology, solute)

    assert result["omega_unclassified_candidates"], "the ligand amide has no evidence"
    bonds = {tuple(sorted(pair)) for pair in result["omega_unscaled_bonds"]}
    assert tuple(sorted((named["ACE_C"].index, named["ALA_N"].index))) in bonds, (
        "the protein's omegas are decidable from residue names and must still be decided")


# --- 5. the retired argument ---------------------------------------------------------------------

def test_route_is_no_longer_an_argument(tmp_path):
    """Retired deliberately: it was a second answer to a question the residue name already answers,
    and four of its six call sites answered it wrongly."""
    import inspect

    parameters = inspect.signature(classify_omega_bonds).parameters
    assert "route" not in parameters, (
        "`route` is retired; the evidence is chosen per candidate from the residue name")
    assert "ligand_sdf" in parameters
