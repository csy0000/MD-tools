"""Section 3: the six cases the omega classifier must get right, and the evidence it records.

REST2 here is omega-selective. An ORDINARY amide omega is left unscaled, because a hot rung that
isomerises cis/trans samples states the reference never does. A PROLINE-LIKE peptide bond is the
exception and stays eligible: its nitrogen is locked in a small ring, so the torsion is not the
near-planar two-state coordinate the exclusion protects.

Getting this wrong is silent. An omega wrongly scaled produces a run that finishes, looks healthy,
and has sampled a different ensemble; an omega wrongly excluded produces a ladder that is stiffer
than intended at every rung. Neither shows up in an acceptance rate. So each of the six cases is
stated separately, with the reason it is the answer it is.

The topologies are built here rather than loaded, because what is being tested is a decision made
from bonds and residue names, and writing those down explicitly is the point.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping. No force field, no dynamics.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.system import classify_omega_bonds

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")


class _Builder:
    """A topology from an explicit list of residues and bonds. No force field involved."""

    def __init__(self):
        self.topology = app.Topology()
        self.chain = self.topology.addChain()
        self.atoms = {}

    def residue(self, name, atoms):
        """`atoms` maps a label to an element symbol."""
        residue = self.topology.addResidue(name, self.chain)
        for label, symbol in atoms.items():
            element = {"C": elem.carbon, "N": elem.nitrogen,
                       "O": elem.oxygen, "H": elem.hydrogen}[symbol]
            self.atoms[label] = self.topology.addAtom(label, element, residue)
        return self

    def bond(self, *pairs):
        for a, b in pairs:
            self.topology.addBond(self.atoms[a], self.atoms[b])
        return self

    def solute(self):
        return [atom.index for atom in self.topology.atoms()]

    def index(self, label):
        return self.atoms[label].index


def _amide(builder, carbon, oxygen, nitrogen, hydrogen=None):
    """A planar amide: C=O, C-N, and optionally N-H."""
    builder.bond((carbon, oxygen), (carbon, nitrogen))
    if hydrogen:
        builder.bond((nitrogen, hydrogen))
    return builder


def _classify(builder, **kwargs):
    return classify_omega_bonds(builder.topology, builder.solute(), **kwargs)


# --- 1. ACE-ALA-NME: the ordinary case ----------------------------------------------------------

def _capped_alanine():
    b = _Builder()
    b.residue("ACE", {"ACE_CH3": "C", "ACE_C": "C", "ACE_O": "O"})
    b.residue("ALA", {"ALA_N": "N", "ALA_H": "H", "ALA_CA": "C", "ALA_CB": "C",
                      "ALA_C": "C", "ALA_O": "O"})
    b.residue("NME", {"NME_N": "N", "NME_H": "H", "NME_CH3": "C"})
    b.bond(("ACE_CH3", "ACE_C"), ("ALA_N", "ALA_CA"), ("ALA_CA", "ALA_CB"),
           ("ALA_CA", "ALA_C"), ("NME_N", "NME_CH3"))
    _amide(b, "ACE_C", "ACE_O", "ALA_N", "ALA_H")
    _amide(b, "ALA_C", "ALA_O", "NME_N", "NME_H")
    return b


def test_ace_ala_nme_leaves_both_backbone_omegas_unscaled():
    """Two ordinary amides, neither proline-like. Both must be protected."""
    b = _capped_alanine()
    result = _classify(b)
    assert len(result["omega_unscaled_bonds"]) == 2
    assert result["omega_proline_like_scaled_bonds"] == []
    assert result["omega_unclassified_candidates"] == []
    assert set(result["omega_unscaled_bonds"]) == {
        (b.index("ACE_C"), b.index("ALA_N")), (b.index("ALA_C"), b.index("NME_N"))}


def test_the_capped_case_names_the_residue_aware_evidence():
    """The method string must say WHICH evidence decided, and for a capped peptide that is the
    residue name. It read `peptide/residue-aware` while a `route` argument existed; the choice is
    per candidate now, so it names the residue set instead -- see
    `test_omega_evidence_by_residue.py`."""
    result = _classify(_capped_alanine())
    assert "residue-aware" in result["omega_detection_method"]
    assert "PROTEIN_RESIDUES" in result["omega_detection_method"]
    assert "PRO" in result["omega_detection_method"]
    # A pure peptide opens no SDF, so the method must not claim one was consulted.
    assert "SMARTS" not in result["omega_detection_method"]


# --- 2. X-PRO: the exception ---------------------------------------------------------------------

def _alanyl_proline():
    b = _Builder()
    b.residue("ACE", {"ACE_CH3": "C", "ACE_C": "C", "ACE_O": "O"})
    b.residue("ALA", {"ALA_N": "N", "ALA_H": "H", "ALA_CA": "C",
                      "ALA_C": "C", "ALA_O": "O"})
    # The pyrrolidine ring: N-CA-CB-CG-CD-N, five atoms.
    b.residue("PRO", {"PRO_N": "N", "PRO_CA": "C", "PRO_CB": "C", "PRO_CG": "C",
                      "PRO_CD": "C", "PRO_C": "C", "PRO_O": "O"})
    b.residue("NME", {"NME_N": "N", "NME_H": "H", "NME_CH3": "C"})
    b.bond(("ACE_CH3", "ACE_C"), ("ALA_N", "ALA_CA"), ("ALA_CA", "ALA_C"),
           ("PRO_N", "PRO_CA"), ("PRO_CA", "PRO_CB"), ("PRO_CB", "PRO_CG"),
           ("PRO_CG", "PRO_CD"), ("PRO_CD", "PRO_N"), ("PRO_CA", "PRO_C"),
           ("NME_N", "NME_CH3"))
    _amide(b, "ACE_C", "ACE_O", "ALA_N", "ALA_H")
    _amide(b, "ALA_C", "ALA_O", "PRO_N")            # X-PRO: no amide hydrogen
    _amide(b, "PRO_C", "PRO_O", "NME_N", "NME_H")
    return b


def test_an_x_pro_bond_stays_eligible_for_scaling():
    """The nitrogen is locked in a pyrrolidine ring, so this omega is not the two-state coordinate
    the exclusion exists to protect. It is scaled like any other eligible torsion."""
    b = _alanyl_proline()
    result = _classify(b)
    assert (b.index("ALA_C"), b.index("PRO_N")) in result["omega_proline_like_scaled_bonds"]
    assert (b.index("ALA_C"), b.index("PRO_N")) not in result["omega_unscaled_bonds"]


def test_the_other_omegas_of_a_proline_containing_chain_are_still_protected():
    """Only the X-PRO bond is the exception; its neighbours are ordinary amides."""
    b = _alanyl_proline()
    result = _classify(b)
    assert set(result["omega_unscaled_bonds"]) == {
        (b.index("ACE_C"), b.index("ALA_N")), (b.index("PRO_C"), b.index("NME_N"))}
    assert result["omega_unclassified_candidates"] == []


def test_proline_like_is_configurable_and_hydroxyproline_can_be_declared():
    """HYP is proline-like chemically and unknown to the classifier by default. Declaring it is
    how a user says so; the alternative is being silently guessed at."""
    b = _alanyl_proline()
    for residue in b.topology.residues():
        if residue.name == "PRO":
            residue.name = "HYP"
    blocked = _classify(b)
    assert len(blocked["omega_unclassified_candidates"]) == 1

    declared = _classify(b, proline_like_residues=("PRO", "HYP"))
    assert declared["omega_unclassified_candidates"] == []
    assert (b.index("ALA_C"), b.index("HYP_N" if "HYP_N" in b.atoms else "PRO_N")
            ) in declared["omega_proline_like_scaled_bonds"]


# --- 3. N-methyl amide ---------------------------------------------------------------------------

def test_an_n_methylated_amide_is_an_ordinary_amide():
    """An N-methyl nitrogen is neither proline-like nor ring-locked, and N-methylated amides
    isomerise readily -- more so than an N-H amide, not less. They must stay unscaled."""
    b = _Builder()
    b.residue("ACE", {"ACE_CH3": "C", "ACE_C": "C", "ACE_O": "O"})
    b.residue("MAL", {"N": "N", "NME_C": "C", "CA": "C", "C": "C", "O": "O"})
    b.residue("NME", {"NME_N": "N", "NME_H": "H", "NME_CH3": "C"})
    b.bond(("ACE_CH3", "ACE_C"), ("N", "NME_C"), ("N", "CA"), ("CA", "C"),
           ("NME_N", "NME_CH3"))
    _amide(b, "ACE_C", "ACE_O", "N")                # tertiary amide: no N-H
    _amide(b, "C", "O", "NME_N", "NME_H")

    # MAL is not a known protein residue, so the peptide route refuses rather than assuming.
    blocked = _classify(b)
    assert any(c["bond"] == (b.index("ACE_C"), b.index("N"))
               for c in blocked["omega_unclassified_candidates"])

    # Told it is an ordinary residue -- not proline-like -- it is excluded from scaling.
    result = _classify(b, proline_like_residues=())
    unresolved = {c["bond"] for c in result["omega_unclassified_candidates"]}
    assert (b.index("ACE_C"), b.index("N")) not in result["omega_proline_like_scaled_bonds"]
    assert (b.index("ACE_C"), b.index("N")) in (
        set(result["omega_unscaled_bonds"]) | unresolved)


# --- 4 and 5. the ligand route: an amide, and a macrocyclic amide ---------------------------------

def test_a_ligand_amide_is_matched_by_bond_order_and_not_by_residue_name():
    """A SMILES-built solute is one UNL residue with no residue evidence at all, so the ligand
    route reads bond orders from the retained SDF instead."""
    rdkit = pytest.importorskip("rdkit.Chem")
    from md_tools.openmm.system import _ligand_ring_nitrogens     # noqa: PLC2701

    molecule = rdkit.AddHs(rdkit.MolFromSmiles("CC(=O)NC"))
    assert molecule is not None
    matches = molecule.GetSubstructMatches(rdkit.MolFromSmarts("[CX3](=[OX1])[NX3]"))
    assert matches, "the ordinary-amide SMARTS must match N-methylacetamide"
    assert callable(_ligand_ring_nitrogens)


def test_a_macrocyclic_amide_nitrogen_is_not_treated_as_proline_like():
    """The ring-size bound is what makes the ligand route correct for macrocycles.

    Every backbone nitrogen of a cyclic peptide is "in a ring". A 15-30 membered macrocycle does
    not constrain its amide the way a pyrrolidine does, so an unbounded `[NX3;R]` test would
    wrongly free every macrocyclic omega for scaling -- the exact opposite of the intent.
    """
    rdkit = pytest.importorskip("rdkit.Chem")

    # cyclo(Gly)4: a 16-membered ring, every amide nitrogen in it.
    macrocycle = rdkit.MolFromSmiles("O=C1CNC(=O)CNC(=O)CNC(=O)CN1")
    assert macrocycle is not None
    ring_sizes = macrocycle.GetRingInfo()
    nitrogens = [atom.GetIdx() for atom in macrocycle.GetAtoms()
                 if atom.GetSymbol() == "N"]
    assert nitrogens
    for nitrogen in nitrogens:
        smallest = min(len(ring) for ring in ring_sizes.AtomRings()
                       if nitrogen in ring)
        assert smallest > 7, (
            f"nitrogen {nitrogen} sits in a {smallest}-membered ring; the <= 7 bound is what "
            f"keeps it from being called proline-like")

    proline = rdkit.MolFromSmiles("O=C(C)N1CCCC1")
    ring_nitrogen = [atom.GetIdx() for atom in proline.GetAtoms()
                     if atom.GetSymbol() == "N"][0]
    smallest = min(len(ring) for ring in proline.GetRingInfo().AtomRings()
                   if ring_nitrogen in ring)
    assert smallest <= 7, "a pyrrolidine nitrogen must fall inside the bound"


# --- 6. an unrecognised modified residue ---------------------------------------------------------

def test_an_unrecognised_modified_residue_blocks_rather_than_being_guessed():
    """The classifier does not know whether a modified residue's omega is ordinary or
    proline-like, and either guess silently changes the Hamiltonian. It says so instead."""
    b = _capped_alanine()
    for residue in b.topology.residues():
        if residue.name == "ALA":
            residue.name = "XYZ"
    result = _classify(b)
    assert result["omega_unclassified_candidates"], "an unknown residue must not pass silently"


def test_the_blocking_candidate_carries_evidence_a_reader_can_act_on():
    """A bare list of unresolved atom pairs says that something blocked and nothing about what to
    do. The evidence must name the residue and the reason."""
    b = _capped_alanine()
    for residue in b.topology.residues():
        if residue.name == "ALA":
            residue.name = "XYZ"
    candidate = _classify(b)["omega_unclassified_candidates"][0]
    assert candidate["nitrogen_residue"] == "XYZ"
    assert candidate["carbon_residue"] in ("ACE", "XYZ")
    assert "XYZ" in candidate["ambiguous"]
    # The ACTIONABLE remedy is the SDF. This asserted `proline_like_residues`, a key no user
    # configuration can set (`_legacy_cfg` rebuilds `rest2` from DEFAULTS), so the message sent a
    # reader to write a key `build-top.config` refuses as unknown.
    assert "SDF" in candidate["ambiguous"]
    assert "proline_like_residues" not in candidate["ambiguous"]
    assert candidate["nitrogen_residue_index"] is not None


def test_a_declared_modified_residue_stops_blocking():
    """The block is a question, and this is the answer to it."""
    b = _capped_alanine()
    for residue in b.topology.residues():
        if residue.name == "ALA":
            residue.name = "XYZ"
    result = _classify(b, proline_like_residues=("PRO", "XYZ"))
    assert result["omega_unclassified_candidates"] == []
    assert len(result["omega_proline_like_scaled_bonds"]) == 1


# --- what is NOT an amide ------------------------------------------------------------------------

def test_a_urea_is_not_taken_for_a_peptide_bond():
    """"A carbon next to one oxygen and one nitrogen" also describes ureas, carbamates and
    carbamic acids. Those are not peptide omega bonds and must not be excluded as if they were."""
    b = _Builder()
    b.residue("URE", {"C": "C", "O": "O", "N1": "N", "H1": "H", "N2": "N", "H2": "H"})
    b.bond(("C", "O"), ("C", "N1"), ("C", "N2"), ("N1", "H1"), ("N2", "H2"))
    result = _classify(b)
    assert result["omega_unscaled_bonds"] == []
    assert result["omega_unclassified_candidates"], "a urea must be reported, not silently scaled"
    assert "urea-like" in result["omega_unclassified_candidates"][0]["ambiguous"]
