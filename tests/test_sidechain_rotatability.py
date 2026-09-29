"""The sidechain rotatability table, and its agreement with the torsion classifier.

The amino acids are a small fixed set, so which sidechain bonds are rotatable is a lookup
(`md_tools.rest2.sidechains`, the user's point of 2026-09-20). A table is only worth having if it
cannot drift from the code that scales torsions, so this file checks it three ways:

  * COMPLETE: every sidechain central bond of the force field's own residue template appears in
    the table, and nothing else does;
  * AGREED: for every residue, the bonds the table calls `fixed` are exactly the ones
    `classify_unscaled_torsions` leaves unscaled, with the same class;
  * LOAD-BEARING: `regions` checks a selected sidechain against the table while it resolves, so a
    table that disagreed with the classifier refuses the build.

The topologies are built from the force field's residue templates, so every supported residue is
covered without 25 structures and without tleap.

PLATFORM_POLICY_EXEMPTION: topologies and tables; no System is created and no Context exists.
"""
from __future__ import annotations

from pathlib import Path
import pytest

openmm = pytest.importorskip("openmm")
from openmm.app import ForceField, Topology, element                            # noqa: E402

from md_tools.openmm.system import PROTEIN_RESIDUES, classify_unscaled_torsions  # noqa: E402
from md_tools.rest2.sidechains import (SIDECHAIN_BONDS, SIDECHAIN_TABLE_VERSION,  # noqa: E402
                                       SidechainTableError, describe_residue, fixed_bonds,
                                       scalable_bonds, sidechain_entry)

BACKBONE = {"N", "H", "H1", "H2", "H3", "CA", "HA", "HA2", "HA3", "C", "O", "OXT"}
#: Names `PROTEIN_RESIDUES` carries that amber14 has no template for, and why.
NO_TEMPLATE = {"HIS": "the ambiguous name; a built system carries HID/HIE/HIP",
               "NMA": "tleap's spelling of the NME cap"}


@pytest.fixture(scope="module")
def templates():
    return ForceField("amber14-all.xml")._templates


def _template_topology(template, *, copies=1, join=None):
    """A Topology of `copies` of one residue template, optionally bonded (a disulfide)."""
    topology = Topology()
    chain = topology.addChain()
    added = []
    for copy_index in range(copies):
        residue = topology.addResidue(template.name, chain)
        atoms = {}
        for atom in template.atoms:
            symbol = atom.element.symbol if atom.element is not None else None
            atoms[atom.name] = topology.addAtom(
                atom.name, element.get_by_symbol(symbol) if symbol else None, residue)
        for first, second in template.bonds:
            topology.addBond(atoms[template.atoms[first].name],
                             atoms[template.atoms[second].name])
        added.append(atoms)
    if join is not None and copies == 2:
        topology.addBond(added[0][join], added[1][join])
    return topology


def _central_bonds(topology):
    """The sidechain central bonds of a topology, by atom name: both ends have another neighbour,
    and the pair is not wholly backbone."""
    neighbours: dict[int, set[int]] = {}
    for bond in topology.bonds():
        neighbours.setdefault(bond.atom1.index, set()).add(bond.atom2.index)
        neighbours.setdefault(bond.atom2.index, set()).add(bond.atom1.index)
    out = set()
    for bond in topology.bonds():
        first, second = bond.atom1, bond.atom2
        if len(neighbours[first.index]) < 2 or len(neighbours[second.index]) < 2:
            continue
        names = (first.name.upper(), second.name.upper())
        if all(name in BACKBONE for name in names):
            continue
        out.add(tuple(sorted(names)))
    return out


# --- 1. the table covers every supported residue ---------------------------------------------------

def test_every_supported_residue_has_an_entry_and_nothing_else_does():
    assert set(SIDECHAIN_BONDS) == set(PROTEIN_RESIDUES), (
        "the table and PROTEIN_RESIDUES must name the same residues")


def test_an_unsupported_residue_is_refused_by_name():
    with pytest.raises(SidechainTableError, match="no sidechain rotatability is defined"):
        sidechain_entry("LIG")


@pytest.mark.parametrize("name", sorted(n for n in PROTEIN_RESIDUES if n not in NO_TEMPLATE))
def test_the_table_lists_exactly_the_templates_sidechain_central_bonds(templates, name):
    entry = sidechain_entry(name)
    topology = _template_topology(templates[name])
    central = _central_bonds(topology)
    if entry.get("cap"):
        # A cap is backbone throughout (`regions.atom_category`), so it HAS no sidechain bond.
        assert entry["bonds"] == ()
        return
    listed = scalable_bonds(name) | set(fixed_bonds(name))
    partnered = {bond["bond"] for bond in entry["bonds"] if bond.get("needs_partner")}
    assert listed - partnered == central, f"{name}: table {sorted(listed)} vs {sorted(central)}"


def test_the_names_without_a_template_are_documented():
    assert set(NO_TEMPLATE) <= set(SIDECHAIN_BONDS)
    for name in NO_TEMPLATE:
        assert sidechain_entry(name)["note"], f"{name} must say why it has no template"


# --- 2. the table and the classifier agree, residue by residue -------------------------------------

@pytest.mark.parametrize("name", sorted(n for n in PROTEIN_RESIDUES if n not in NO_TEMPLATE))
def test_the_classifier_leaves_unscaled_exactly_what_the_table_calls_fixed(templates, name):
    topology = _template_topology(templates[name])
    atoms = list(topology.atoms())
    classified = classify_unscaled_torsions(topology, [a.index for a in atoms])
    assert not classified["unclassified"], classified["unclassified"]
    by_name = {tuple(sorted((atoms[a].name.upper(), atoms[b].name.upper()))): entry["class"]
               for entry in classified["central_bonds"]
               for a, b in [entry["bond"]]}
    expected = fixed_bonds(name)
    assert by_name == expected, f"{name}: classifier {by_name} vs table {expected}"


def test_a_disulfide_makes_chi2_a_central_bond_and_leaves_it_scalable(templates):
    """CYX's CB-SG is the table's `needs_partner` case, and SG-SG is nobody's fixed bond."""
    topology = _template_topology(templates["CYX"], copies=2, join="SG")
    assert ("CB", "SG") in _central_bonds(topology)
    atoms = list(topology.atoms())
    classified = classify_unscaled_torsions(topology, [a.index for a in atoms])
    assert classified["central_bonds"] == [], "a disulfide is not an unscaled class"
    assert fixed_bonds("CYX") == {}


@pytest.mark.parametrize("name, bond, kind", [
    ("ASN", ("CG", "ND2"), "amide_omega"), ("GLN", ("CD", "NE2"), "amide_omega"),
    # One CATEGORY for both rules now; the finer reason lives in each entry's own label.
    ("ARG", ("CZ", "NE"), "non_rotatable"), ("ARG", ("CZ", "NH1"), "non_rotatable"),
    ("PHE", ("CD1", "CG"), "non_rotatable"), ("TYR", ("CE1", "CZ"), "non_rotatable"),
    ("TRP", ("CD1", "NE1"), "non_rotatable"), ("HID", ("CG", "ND1"), "non_rotatable"),
    ("HIE", ("CE1", "NE2"), "non_rotatable"), ("HIP", ("CD2", "NE2"), "non_rotatable"),
])
def test_the_bonds_the_user_named_are_fixed_with_the_class_they_have(name, bond, kind):
    assert fixed_bonds(name)[tuple(sorted(bond))] == kind


@pytest.mark.parametrize("name, bond", [
    ("ASP", ("CG", "OD1")), ("ASP", ("CG", "OD2")),
    ("GLU", ("CD", "OE1")), ("GLU", ("CD", "OE2")),
])
def test_a_carboxylate_is_not_a_central_bond_at_all(templates, name, bond):
    """Not "fixed": no proper torsion runs across it, so there is nothing to scale or protect."""
    assert tuple(sorted(bond)) in {tuple(sorted(b)) for b in sidechain_entry(name)["not_central"]}
    assert tuple(sorted(bond)) not in _central_bonds(_template_topology(templates[name]))
    assert tuple(sorted(bond)) not in scalable_bonds(name)
    assert tuple(sorted(bond)) not in fixed_bonds(name)


@pytest.mark.parametrize("name, count", [("ARG", 4), ("LYS", 4), ("MET", 3), ("GLU", 3),
                                         ("PRO", 4), ("VAL", 1), ("ALA", 0), ("GLY", 0)])
def test_the_chi_count_per_residue(name, count):
    assert len([b for b in sidechain_entry(name)["bonds"] if b["kind"] == "chi"]) == count


def test_describe_residue_marks_each_bond(capsys):
    text = describe_residue("TYR")
    assert SIDECHAIN_TABLE_VERSION in text
    ring = [line for line in text.splitlines() if "benzene ring" in line]
    assert len(ring) == 6 and all("UNSCALED" in line and "[non_rotatable]" in line
                                  for line in ring)
    assert [line for line in text.splitlines() if "chi1" in line][0].strip().startswith("scaled")
    assert [line for line in text.splitlines() if "hydroxyl rotation" in line][0].strip(
        ).startswith("scaled")


# --- 3. the table is load-bearing in `regions` -----------------------------------------------------

def test_a_table_that_disagreed_with_the_classifier_refuses_the_region(monkeypatch, tmp_path):
    """The check is not decoration: make the table wrong and the build must refuse."""
    from md_tools.rest2 import sidechains
    from md_tools.rest2.selection import SelectionError

    from .selective_rest2_fixture import RESIDUE, build_fixture
    from .test_selective_rest2_selection import _select

    fixture = build_fixture()
    config = {"sidechain_scaling_list": f":{RESIDUE['A.PHE']}"}
    assert _select(fixture, config, tmp_path)[1].torsion_bonds, "the control: it resolves"

    honest = sidechains.SIDECHAIN_BONDS["PHE"]
    lying = dict(honest, bonds=tuple(
        dict(entry, kind="chi", label="chi-not-really") if entry["kind"] == "fixed" else entry
        for entry in honest["bonds"]))
    monkeypatch.setitem(sidechains.SIDECHAIN_BONDS, "PHE", lying)
    with pytest.raises(SelectionError, match="disagree"):
        _select(fixture, config, tmp_path)


# --- an aromatic ring C-N bond is classified, not "undecided" -------------------------------
#
# Reported from a student's folate build, 2026-09-28. `build-top --rest2-scaler` refused with
# "1 item(s) could not be classified ... bond 29-31: FOL0 C -> FOL0 N: RDKit found no
# ordinary-amide match", over a bond it had ALREADY protected.

FOLATE_SMILES = "c1cc(ccc1C(=O)N[C@@H](CCC(=O)O)C(=O)O)NCc2cnc3c(n2)C(=O)N=C(N3)N"


def _folate_topology_and_sdf(tmp_path):
    from openff.toolkit.topology import Molecule

    mol = Molecule.from_smiles(FOLATE_SMILES, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    mol.name = "FOL"
    sdf = tmp_path / "FOL.sdf"
    mol.to_file(str(sdf), file_format="sdf")
    topology = mol.to_topology().to_openmm()
    for residue in topology.residues():
        residue.name = "FOL"
    return topology, sdf


@pytest.mark.slow
def test_an_aromatic_ring_c_n_bond_is_not_an_undecided_amide(tmp_path):
    """Folate's pterin 4-oxo lactam is a C-N bond whose carbon bears an oxygen, so it reaches the
    amide test -- and fails `[CX3](=[OX1])[NX3]` for a DIFFERENT reason in each tautomer, measured
    on the SDF as the classifier reads it:

        1H (this molecule)  the lactam N has no hydrogen and two connections -> [NX3] fails
        3H (standard)       the N is fine; the CARBON is aromatic, and SMARTS `C` is ALIPHATIC

    Neither matches, so switching tautomer is not a workaround.

    The bond is already `aromatic_ring` in `central_bonds`, so its torsions are already unscaled
    and nothing is undecided. Refusing the build over it stopped a correct calculation, and the
    refusal could not be answered: the exclusions file only ADDS protection, which it has.
    """
    topology, sdf = _folate_topology_and_sdf(tmp_path)
    report = classify_unscaled_torsions(topology, list(range(topology.getNumAtoms())),
                                        ligand_sdf=sdf)

    assert report["unclassified"] == [], (
        "an amide candidate another rule already classified must not block the build")

    protected = {tuple(e["bond"]): e["class"] for e in report["central_bonds"]}
    assert protected.get((26, 28)) == "non_rotatable", protected

    # The decision is RECORDED, with the amide test's own reason, rather than dropped silently.
    resolved = report["amide_candidates_resolved_by_another_rule"]
    assert [e["bond"] for e in resolved] == [[26, 28]]
    assert resolved[0]["resolved_by"] == "non_rotatable"
    assert "no ordinary-amide match" in resolved[0]["amide_test"]

    # And the REAL amide is still an amide, not quietly swept into the same bucket.
    assert protected.get((6, 8)) == "amide_omega", protected


@pytest.mark.slow
def test_a_candidate_no_rule_can_place_is_still_refused(tmp_path):
    """The guard on the fix: dropping resolved candidates must not drop unresolvable ones.

    With no SDF at all there are no bond orders, so nothing about this residue is classified --
    and it must still refuse, or the fix would have turned a real ambiguity into silence.
    """
    topology, _ = _folate_topology_and_sdf(tmp_path)
    report = classify_unscaled_torsions(topology, list(range(topology.getNumAtoms())))

    assert report["unclassified"], "a solute with no bond orders must still be refused"
    assert report["amide_candidates_resolved_by_another_rule"] == []


@pytest.mark.slow
def test_relaxing_the_amide_carbon_to_any_element_would_scale_an_aromatic_ring_bond():
    """Why the one-line SMARTS fix was NOT taken, pinned so nobody takes it later.

    `[#6X3](=[OX1])[#7X3]` is the obvious repair for an aromatic carbonyl carbon. On folate's
    STANDARD 3H tautomer it does match the pterin lactam -- and that nitrogen sits in a
    6-membered ring, inside `max_proline_ring_size` (7), so the bond would then be classified
    proline-like and therefore SCALED. An aromatic ring bond that is currently protected would
    start having its torsions scaled, silently.

    This test asserts the two facts that make that true, so a future relaxation has to confront
    them rather than discover the consequence in a production run.
    """
    import tempfile

    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    three_h = "c1cc(ccc1C(=O)N[C@@H](CCC(=O)O)C(=O)O)NCc2cnc3c(n2)C(=O)NC(=N3)N"
    mol = Molecule.from_smiles(three_h, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    mol.name = "FOL"
    path = Path(tempfile.mkdtemp()) / "fol3h.sdf"
    mol.to_file(str(path), file_format="sdf")
    rd = Chem.MolFromMolFile(str(path), removeHs=False)

    def matches(pattern):
        return any(sorted(m[0::2]) == [26, 28]
                   for m in rd.GetSubstructMatches(Chem.MolFromSmarts(pattern)))

    assert not matches("[CX3](=[OX1])[NX3]"), "the shipped pattern must still not match it"
    assert matches("[#6X3](=[OX1])[#7X3]"), (
        "if this stops matching, the hazard below no longer applies and this test is obsolete")

    rings = [len(r) for r in rd.GetRingInfo().AtomRings() if 28 in r]
    assert rings and min(rings) <= 7, (
        f"N28 rings {rings}: the proline-like bound is 7, so a match would make this bond "
        f"proline-like and SCALE an aromatic ring bond")


@pytest.mark.slow
def test_the_refusal_names_the_ATOMS_not_just_their_elements(tmp_path):
    """The message used to print the literal letters "C" and "N" -- the elements, not the atoms.

    Every amide candidate in a residue therefore produced the same text, and the letters read
    exactly like atom NAMES. On folate that is actively misleading: the PDB component's
    benzoylglutamate amide really does have atoms named `C` and `N`, so "FOL0 C -> FOL0 N" points
    a reader at that bond while the bond in question was the pterin lactam. The bond INDEX in the
    same line was correct throughout; only the label lied. (It cost a wrong diagnosis on
    2026-09-28, and the correction had to be retracted in turn.)
    """
    import tempfile

    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    from md_tools.ligands.package import default_atom_names
    from md_tools.openmm.system import UnclassifiedTorsionError, unscaled_torsions

    # The trigger is a UREA -- the carbonyl carbon carries two nitrogens, so `_amide_candidates`
    # THE TRIGGER IS NO SDF, and that is the third one this test has had in a day.
    #
    # It began as a deprotonated benzoyl amide, which stopped refusing when the amidate gap closed
    # (`[CX3](=[OX1])[NX2,NX3]`). It became a urea, which stopped refusing two commits later when
    # a urea was measured and left scalable. Both vehicles were CHEMISTRY, and this release spent
    # its time moving chemistry out of the refusing set -- so any chemical trigger was going to rot
    # again, and the second one did, caught by the gate after passing every fast lane.
    #
    # Missing bond orders is not chemistry. It is the one thing that still refuses BY DESIGN and
    # cannot be argued out of it: with no SDF the classifier sees nothing, so nothing is protected,
    # and scaling a whole aromatic ring because a file was forgotten is the failure the refusal
    # exists to prevent. The subject of this test -- does the message name the ATOMS rather than
    # the bare element letters -- never depended on which trigger fired.
    smi = "c1cc(ccc1C(=O)N[C@@H](CCC(=O)O)C(=O)O)NCc2cnc3c(n2)C(=O)NC(=N3)N"
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    mol.name = "FOL"
    sdf = Path(tempfile.mkdtemp()) / "FOL.sdf"
    mol.to_file(str(sdf), file_format="sdf")
    topology = mol.to_topology().to_openmm()
    for atom, name in zip(topology.atoms(),
                          default_atom_names(Chem.AddHs(Chem.MolFromSmiles(smi)))):
        atom.name = name
    for residue in topology.residues():
        residue.name = "FOL"

    with pytest.raises(UnclassifiedTorsionError) as refusal:
        unscaled_torsions(topology, list(range(topology.getNumAtoms())))   # no ligand_sdf

    line = next(ln.strip() for ln in str(refusal.value).splitlines()
                if ln.strip().startswith("bond "))
    atoms = list(topology.atoms())
    index = [int(x) for x in line.split(":")[0].removeprefix("bond ").split("-")]
    for i in index:
        assert atoms[i].name in line, (
            f"the refusal must name atom {i} ({atoms[i].name}); got: {line}")
    # And it must not be the bare element letters that started this.
    assert " C -> " not in line and line.count(" N:") == 0, line
