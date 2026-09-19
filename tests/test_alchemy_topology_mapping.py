"""The A->B atom map: one-to-one, both directions, and the chemistry it is refused for.

The map-shape and fixture-pair tests use the real registered-form packages. The refusal tests for
ring changes, stereochemistry, charge and force-field mismatches need molecules no fixture
package holds; they use `_Stand`, which carries exactly the attributes `validate_map` reads
(an RDKit molecule with a 3D conformer, atom names, and the metadata fields named below). No
energy is computed from a stand-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rdkit")

from tests.alchemy_fixtures import CHLOROETHANE, CORE, ETHANE, ETHANOL, package  # noqa: E402


@dataclass
class _Stand:
    mol: object
    reference: str
    charge_method: str = "am1bcc"
    forcefield: str = "openff-2.2.1"
    atom_names: tuple = field(init=False)
    metadata: dict = field(init=False)
    conventions: dict = field(init=False)

    def __post_init__(self):
        counts: dict[str, int] = {}
        names = []
        for atom in self.mol.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
            names.append(f"{atom.GetSymbol()}{counts[atom.GetSymbol()]}")
        self.atom_names = tuple(names)
        self.metadata = {
            "forcefield": {"family": "smirnoff", "resource": self.forcefield},
            "charges": {"method": self.charge_method, "backend_id": "ambertools-sqm"},
            "chemical_state": {"net_formal_charge": sum(a.GetFormalCharge()
                                                        for a in self.mol.GetAtoms())},
        }
        self.conventions = {"coulomb14scale": 5 / 6, "lj14scale": 0.5,
                            "combining_rule": "lorentz-berthelot"}


def _stand(smiles: str, label: str, **kwargs) -> _Stand:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260919
    assert AllChem.EmbedMolecule(mol, params) == 0
    return _Stand(mol=mol, reference=f"LOCAL-{label}/param_000000000000", **kwargs)


def _map(a, b, pairs):
    from md_tools.alchemy.topology_mapping import AtomMap

    return AtomMap.from_pairs(a, b, pairs)


def _validate(a, b, pairs, mode="hybrid"):
    from md_tools.alchemy.topology_mapping import validate_map

    return validate_map(a, b, _map(a, b, pairs), mode)


@pytest.fixture(scope="module")
def eta():
    return package(ETHANE)


@pytest.fixture(scope="module")
def cle():
    return package(CHLOROETHANE)


@pytest.fixture(scope="module")
def eoh():
    return package(ETHANOL)


# ------------------------------------------------------------------------------------------------
# the map as a function
# ------------------------------------------------------------------------------------------------
def test_both_directions_are_recorded_and_are_inverses(eta, cle):
    amap = _map(eta, cle, {n: n for n in CORE})
    record = amap.record(eta, cle)
    assert {int(a): b for a, b in record["a_to_b"].items()} == amap.a_to_b
    assert {int(b): a for b, a in record["b_to_a"].items()} == amap.b_to_a
    assert all(amap.b_to_a[b] == a for a, b in amap.a_to_b.items())
    # hand check against the fixture .pdb files: ethane H5 is atom 6, chloroethane H5 is atom 7
    assert amap.a_to_b[eta.atom_names.index("H5")] == cle.atom_names.index("H5") == 7
    assert amap.inverse().inverse() == amap
    assert amap.inverse().record(cle, eta)["a_to_b"] == record["b_to_a"]


def test_a_stored_map_round_trips_and_an_edited_one_is_refused(eta, cle):
    from md_tools.alchemy.topology_mapping import AtomMap, MapError

    amap = _map(eta, cle, {n: n for n in CORE})
    record = amap.record(eta, cle)
    assert AtomMap.from_record(record, eta, cle) == amap
    edited = {**record, "b_to_a": {**record["b_to_a"], "0": 1}}
    with pytest.raises(MapError, match="not inverses"):
        AtomMap.from_record(edited, eta, cle)
    renamed = {**record, "by_name": [["C1", "C2"]] + record["by_name"][1:]}
    with pytest.raises(MapError, match="does not reproduce"):
        AtomMap.from_record(renamed, eta, cle)
    with pytest.raises(MapError, match="not "):
        AtomMap.from_record(record, eta, package(ETHANOL))


@pytest.mark.parametrize("pairs,message", [
    ([("C1", "C1"), ("C1", "C2")], "mapped twice"),
    ([("C1", "C1"), ("C2", "C1")], "mapped twice"),
    ([("C9", "C1")], "names 0 atoms"),
    ([(0, 99)], "outside"),
    ([], "empty"),
])
def test_a_map_that_is_not_one_to_one_is_refused(eta, cle, pairs, message):
    from md_tools.alchemy.topology_mapping import MapError

    with pytest.raises(MapError, match=message):
        _map(eta, cle, pairs)


# ------------------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------------------
def test_hybrid_reports_one_unique_group_per_side_with_its_attachment(eta, cle):
    report = _validate(eta, cle, {n: n for n in CORE})
    a = report["unique_components"]["A"]
    b = report["unique_components"]["B"]
    assert [[eta.atom_names[i] for i in c["atoms"]] for c in a] == [["H6"]]
    assert [[cle.atom_names[i] for i in c["atoms"]] for c in b] == [["Cl1"]]
    assert b[0]["attachments"] == [(cle.atom_names.index("Cl1"), cle.atom_names.index("C2"))]


def test_a_hydrogen_mapped_to_a_heavy_atom_is_single_topology_only(eta, cle):
    from md_tools.alchemy.topology_mapping import MapError

    pairs = {**{n: n for n in CORE}, "H6": "Cl1"}
    with pytest.raises(MapError, match="single-topology construction"):
        _validate(eta, cle, pairs, "hybrid")
    report = _validate(eta, cle, pairs, "single")
    [change] = report["notes"]["element_changes"]
    assert (change["a_name"], change["b_name"], change["hydrogen_to_heavy"]) == ("H6", "Cl1", True)


def test_single_topology_needs_one_side_fully_mapped(eta, cle):
    from md_tools.alchemy.topology_mapping import MapError

    with pytest.raises(MapError, match="that is a hybrid topology"):
        _validate(eta, cle, {n: n for n in CORE}, "single")


def test_separated_and_unknown_modes_are_refused_by_name(eta, cle):
    from md_tools.alchemy.topology_mapping import MapError

    with pytest.raises(MapError, match="separated topology is deferred"):
        _validate(eta, cle, {n: n for n in CORE}, "separated")
    with pytest.raises(MapError, match="not one of"):
        _validate(eta, cle, {n: n for n in CORE}, "hybird")


def test_a_map_between_other_packages_is_refused(eta, cle, eoh):
    from md_tools.alchemy.topology_mapping import MapError, validate_map

    with pytest.raises(MapError, match="the map is between"):
        validate_map(eta, eoh, _map(eta, cle, {n: n for n in CORE}), "hybrid")


# ------------------------------------------------------------------------------------------------
# chemistry refusals
# ------------------------------------------------------------------------------------------------
def test_a_net_charge_change_is_refused():
    from md_tools.alchemy.topology_mapping import MapError

    a, b = _stand("CC(=O)O", "ACID"), _stand("CC(=O)[O-]", "ACETATE")
    pairs = [(i, i) for i in range(4)]
    with pytest.raises(MapError, match="net formal charge changes"):
        _validate(a, b, pairs)


@pytest.mark.parametrize("field_,value,message", [
    ("forcefield", "openff-2.1.0", "ONE force field"),
    ("charge_method", "nagl", "charge method differs"),
])
def test_endpoints_from_different_parameter_models_are_refused(field_, value, message):
    from md_tools.alchemy.topology_mapping import MapError

    a = _stand("CCO", "A")
    b = _stand("CCO", "B", **{field_: value})
    with pytest.raises(MapError, match=message):
        _validate(a, b, [(i, i) for i in range(a.mol.GetNumAtoms())])


def test_a_ring_opening_is_refused():
    """Cyclopropane -> propane with all carbons mapped: the ring bond has no counterpart."""
    from md_tools.alchemy.topology_mapping import MapError

    ring, chain = _stand("C1CC1", "RING"), _stand("CCC", "CHAIN")
    with pytest.raises(MapError, match="bond graph changes"):
        _validate(ring, chain, [(0, 0), (1, 1), (2, 2)])


def test_a_partially_mapped_ring_is_refused_as_a_two_anchor_dummy_group():
    """Methylcyclohexane -> toluene mapping only the methyl and its ring carbon's neighbours."""
    from md_tools.alchemy.topology_mapping import MapError

    a, b = _stand("C1CCCCC1", "CHX"), _stand("C1CCCCC1C", "MCHX")
    # map two adjacent ring carbons with the same ring membership; the rest of B's ring is a
    # unique group bridging them
    with pytest.raises(MapError, match="attached to the mapped core by 2 bonds"):
        _validate(b, a, [(0, 0), (1, 1)])


def test_a_ring_membership_change_is_refused():
    from md_tools.alchemy.topology_mapping import MapError

    a, b = _stand("C1CC1C", "MCP"), _stand("CCCC", "BUT")
    # the methyl carbon (in no ring) mapped to a ring carbon's counterpart
    with pytest.raises(MapError, match="ring"):
        _validate(a, b, [(0, 0), (3, 1)])


def test_an_inverted_stereocentre_is_refused_and_a_retained_one_passes():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from md_tools.alchemy.topology_mapping import MapError

    def enantiomer(smiles, label):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        assert AllChem.EmbedMolecule(mol, params) == 0
        return _Stand(mol=mol, reference=f"LOCAL-{label}/param_000000000000")

    r = enantiomer("F[C@](Cl)(Br)C", "R")
    s = enantiomer("F[C@@](Cl)(Br)C", "S")
    r2 = enantiomer("F[C@](Cl)(Br)C", "R2")
    same_order = [(i, i) for i in range(r.mol.GetNumAtoms())]
    with pytest.raises(MapError, match="inverts stereocentre"):
        _validate(r, s, same_order)
    report = _validate(r, r2, same_order)
    stereo = next(c for c in report["checks"] if c["check"] == "stereochemistry-preserved")
    assert stereo["detail"]["centres"] == [[1, 1]]


def test_a_stereocentre_the_map_cannot_fix_is_refused():
    from md_tools.alchemy.topology_mapping import MapError

    a, b = _stand("F[C@](Cl)(Br)C", "R"), _stand("F[C@](Cl)(Br)C", "R2")
    # only two of the centre's neighbours mapped: its configuration in B is not determined
    with pytest.raises(MapError, match="fewer than three"):
        _validate(a, b, [(0, 0), (1, 1), (4, 4)])


def test_dual_mode_only_places_and_restrains(eta, cle):
    report = _validate(eta, cle, {"C1": "C1", "C2": "C2"}, "dual")
    assert [c["check"] for c in report["checks"]][-1] == "dual-map-role"


# ------------------------------------------------------------------------------------------------
# the validated automatic map
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["chloroethane", "ethanol", "pentane", "propanoate"])
def test_the_automatic_map_is_the_explicit_one_up_to_symmetry(pair):
    from md_tools.alchemy.topology_mapping import _signature, propose_map, validate_map
    from tests.alchemy_fixtures import ACETATE, ACETATE_TO_PROPANOATE, PENTANE, PROPANOATE

    a, b, explicit = {
        "chloroethane": (ETHANE, CHLOROETHANE, None), "ethanol": (ETHANE, ETHANOL, None),
        "pentane": (ETHANE, PENTANE, None),
        "propanoate": (ACETATE, PROPANOATE, ACETATE_TO_PROPANOATE)}[pair]
    a, b = package(a), package(b)
    expected = _map(a, b, explicit or {n: n for n in CORE})
    proposed, report = propose_map(a, b, "hybrid")
    assert _signature(a.mol, b.mol, proposed.pairs) == _signature(a.mol, b.mol, expected.pairs)
    validate_map(a, b, proposed, "hybrid")
    assert report["method"] == "rdkit-fmcs-heavy/1"
    assert report["mapped_atoms"] == len(expected.pairs)
    again, _ = propose_map(a, b, "hybrid")
    assert again == proposed                     # deterministic


def test_an_ambiguous_automatic_map_is_refused_with_its_alternatives():
    """Ethanolamine -> propane: the shared C-C can put A's N-side carbon on propane's end or its
    middle carbon; both keep four hydrogens and validate, and no symmetry relates them."""
    from md_tools.alchemy.topology_mapping import MapError, propose_map

    a, b = _stand("NCCO", "ETANOLAMINE"), _stand("CCC", "PROPANE")
    with pytest.raises(MapError, match="chemically ambiguous: 2 maps"):
        propose_map(a, b, "hybrid")


def test_a_symmetric_tie_is_not_ambiguous():
    """Ethylene glycol -> ethanol: two placements of O-C-C in A, related by A's symmetry."""
    from md_tools.alchemy.topology_mapping import propose_map

    a, b = _stand("OCCO", "GLYCOL"), _stand("CCO", "ETHANOL")
    amap, report = propose_map(a, b, "hybrid")
    assert report["symmetry_equivalent_best"] >= 2 and len(amap.pairs) >= 3


def test_single_topology_needs_an_explicit_map(eta, cle):
    from md_tools.alchemy.topology_mapping import MapError, propose_map

    with pytest.raises(MapError, match="not offered for single topology"):
        propose_map(eta, cle, "single")


def test_an_automatic_map_builds_a_plan_that_recovers():
    from md_tools.alchemy.topology import build_topology_plan
    from md_tools.alchemy.topology_mapping import propose_map
    from tests.alchemy_fixtures import water_environment

    a, b = package(ETHANE), package(CHLOROETHANE)
    amap, _ = propose_map(a, b, "hybrid")
    plan = build_topology_plan(a, b, amap, water_environment(), mode="hybrid")
    assert len(plan.common) == 7 and len(plan.a_only) == 1 and len(plan.b_only) == 1


def test_a_proposed_map_written_for_review_reproduces_the_plan():
    """The review file the CLI would write: the proposal's map record, as YAML, read back through
    AtomMap.from_record, builds the identical plan."""
    import yaml

    from md_tools.alchemy.topology import build_topology_plan
    from md_tools.alchemy.topology_mapping import AtomMap, propose_map
    from tests.alchemy_fixtures import water_environment

    a, b = package(ETHANE), package(CHLOROETHANE)
    proposed, _ = propose_map(a, b, "hybrid")
    reread = AtomMap.from_record(yaml.safe_load(yaml.safe_dump(proposed.record(a, b))), a, b)
    assert reread == proposed
    env = water_environment()
    assert build_topology_plan(a, b, proposed, env, mode="hybrid").sha256 == \
        build_topology_plan(a, b, reread, env, mode="hybrid").sha256
