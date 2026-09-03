"""Every way `cv.yaml` is refused, and the one way it is accepted.

WHY THE REFUSALS ARE THE INTERESTING TESTS

    A CV series that names the wrong four atoms is indistinguishable from one that names the right
    four: same column heading, same range, same plausible-looking distribution. Nothing downstream
    catches it. So the value of this schema is entirely in what it declines to guess, and each
    test below is one guess that is not made.
"""
from __future__ import annotations

import pytest

from md_tools.cv import CVDefinitionError, parse_cv_definition

GOOD = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


def test_a_valid_definition_parses_and_carries_its_own_digest():
    """The control, and the provenance claim: the digest is of the exact bytes given."""
    import hashlib

    definition = parse_cv_definition(GOOD, particles=20)
    assert definition.names == ("phi",)
    assert definition.variables[0].indices == (4, 6, 8, 14)
    assert definition.digest == hashlib.sha256(GOOD.encode("utf-8")).hexdigest()
    assert definition.resolved()["units"] == "degrees"
    assert definition.resolved()["columns"] == ["phi"]


def test_a_duplicate_yaml_key_is_refused():
    """PyYAML keeps the last value silently; the earlier one appears nowhere at all."""
    text = GOOD + "collective_variables:\n  - {name: psi, type: torsion, atom_indices: [0,1,2,3]}\n"
    with pytest.raises(CVDefinitionError, match="duplicate key"):
        parse_cv_definition(text, particles=20)


def test_an_unsupported_type_is_refused_by_name_with_the_schema_version():
    """A `distance` written against a future schema must fail loudly, not shorten the CSV."""
    text = GOOD.replace("type: torsion", "type: distance")
    with pytest.raises(CVDefinitionError, match="'distance'.*schema version 1"):
        parse_cv_definition(text, particles=20)


def test_an_unknown_field_is_refused():
    """A field silently ignored is a definition whose author believes it took effect."""
    text = GOOD + "    weight: 2.0\n"
    with pytest.raises(CVDefinitionError, match="unknown field"):
        parse_cv_definition(text, particles=20)


def test_a_wrong_schema_version_is_refused():
    with pytest.raises(CVDefinitionError, match="schema_version"):
        parse_cv_definition(GOOD.replace("schema_version: 1", "schema_version: 2"), particles=20)


def test_two_variables_with_the_same_name_are_refused():
    """Names are CSV headings: one column would silently overwrite the other in every reader."""
    text = GOOD + "  - {name: phi, type: torsion, atom_indices: [0, 1, 2, 3]}\n"
    with pytest.raises(CVDefinitionError, match="both named 'phi'"):
        parse_cv_definition(text, particles=20)


def test_a_name_that_is_not_csv_safe_is_refused():
    text = GOOD.replace("name: phi", 'name: "phi,psi"')
    with pytest.raises(CVDefinitionError, match="name must be"):
        parse_cv_definition(text, particles=20)


@pytest.mark.parametrize("indices, fragment", [
    ("[4, 6, 8]", "exactly four"),
    ("[4, 6, 8, 14, 15]", "exactly four"),
    ("[4, 6, 8, -1]", "zero-based"),
    ("[4, 6, 8, 99]", "out of range"),
    ("[4, 6, 8, 6]", "distinct"),
    ("[4, 6, 8, 1.5]", "must be integers"),
])
def test_bad_index_lists_are_refused(indices, fragment):
    """Clamping, truncating or de-duplicating any of these would produce a plausible wrong answer."""
    with pytest.raises(CVDefinitionError, match=fragment):
        parse_cv_definition(GOOD.replace("[4, 6, 8, 14]", indices), particles=20)


def test_both_atom_indices_and_atoms_is_refused():
    """They can disagree, and there is no correct rule for which one wins."""
    text = GOOD + "    atoms: [{chain: A, residue: '1', atom: N}]\n"
    with pytest.raises(CVDefinitionError, match="exactly one of"):
        parse_cv_definition(text, particles=20)


def test_neither_atom_indices_nor_atoms_is_refused():
    text = "schema_version: 1\ncollective_variables:\n  - {name: phi, type: torsion}\n"
    with pytest.raises(CVDefinitionError, match="exactly one of"):
        parse_cv_definition(text, particles=20)


def test_an_empty_variable_list_is_refused():
    """Enabled reporting that produces a CSV with no columns is a silent no-op."""
    with pytest.raises(CVDefinitionError, match="non-empty"):
        parse_cv_definition("schema_version: 1\ncollective_variables: []\n")


# --- selector resolution, against a real topology ---------------------------------------------

@pytest.fixture
def topology():
    """Two chains that deliberately share residue ids and atom names.

    An ambiguous selector is only ambiguous if something in the topology can be matched twice --
    a single-chain fixture would let a first-match bug pass every test.
    """
    openmm_app = pytest.importorskip("openmm.app")
    from openmm.app import Element, Topology

    top = Topology()
    for chain_id in ("A", "B"):
        chain = top.addChain(id=chain_id)
        for residue_id in ("1", "2"):
            residue = top.addResidue("ALA", chain, id=residue_id)
            for name in ("N", "CA", "C"):
                top.addAtom(name, Element.getBySymbol("C"), residue)
    return top


SELECTED = """\
schema_version: 1
collective_variables:
  - name: psi
    type: torsion
    atoms:
      - {chain: A, residue: "1", atom: N}
      - {chain: A, residue: "1", atom: CA}
      - {chain: A, residue: "1", atom: C}
      - {chain: A, residue: "2", atom: N}
"""


def test_selectors_resolve_to_the_atoms_they_name(topology):
    """Chain A's residues are atoms 0-5; residue 2's N is index 3."""
    definition = parse_cv_definition(SELECTED, topology=topology, particles=12)
    assert definition.variables[0].indices == (0, 1, 2, 3)
    # The selectors survive into the sidecar: an index list alone cannot be checked against a
    # topology by a person reading the output months later.
    assert definition.resolved()["collective_variables"][0]["atoms"][0]["chain"] == "A"


def test_a_selector_naming_the_other_chain_resolves_there_and_not_by_position(topology):
    """Chain B's atoms are 6-11. A selector that ignored `chain` would return 0-3 again."""
    definition = parse_cv_definition(SELECTED.replace("chain: A", "chain: B"),
                                     topology=topology, particles=12)
    assert definition.variables[0].indices == (6, 7, 8, 9)


def test_a_selector_matching_nothing_is_refused(topology):
    with pytest.raises(CVDefinitionError, match="no atom in the topology matches"):
        parse_cv_definition(SELECTED.replace("atom: N}", "atom: HZ3}", 1),
                            topology=topology, particles=12)


def test_a_selector_matching_two_atoms_is_refused_rather_than_taking_the_first():
    """THE first-match rule this schema does not have.

    An atom name repeated inside one residue -- which real PDBs do carry, for alternate locations
    and for hydrogens named by a convention that does not disambiguate -- makes the selector
    genuinely ambiguous. Silently resolving it to the first measures a torsion nobody asked for,
    in a file that looks entirely correct.
    """
    pytest.importorskip("openmm.app")
    from openmm.app import Element, Topology

    top = Topology()
    chain = top.addChain(id="A")
    residue = top.addResidue("ALA", chain, id="1")
    for name in ("N", "CA", "C", "N"):          # `N` twice, contiguously
        top.addAtom(name, Element.getBySymbol("C"), residue)
    second = top.addResidue("ALA", chain, id="2")
    top.addAtom("N", Element.getBySymbol("C"), second)

    with pytest.raises(CVDefinitionError, match="matches 2 atoms"):
        parse_cv_definition(SELECTED, topology=top, particles=5)


def test_a_partial_selector_is_refused(topology):
    """Omitting `chain` would match by accident in any multi-chain system."""
    text = SELECTED.replace('- {chain: A, residue: "1", atom: N}', '- {residue: "1", atom: N}', 1)
    with pytest.raises(CVDefinitionError, match="all of chain, residue and atom"):
        parse_cv_definition(text, topology=topology, particles=12)


def test_a_residue_id_is_compared_as_written_not_as_an_integer(topology):
    """`52A` and `52` are different residues; int() would conflate them."""
    with pytest.raises(CVDefinitionError, match="no atom in the topology matches"):
        parse_cv_definition(SELECTED.replace('residue: "1"', 'residue: "1A"'),
                            topology=topology, particles=12)
