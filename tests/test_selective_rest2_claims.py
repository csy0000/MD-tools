"""`claimed_region_differences`: a claimed region against the one recorded with the states.

The ONE comparison build-md uses when a REST2 configuration claims selector keys about saved
states (S0's design, 2026-09-19). The claim is resolved by the one resolver, so a different
spelling of the same region agrees. Exclusion files are compared by content, not by path, and an
invalid claim is refused rather than reported.

PLATFORM_POLICY_EXEMPTION: topology resolution and record comparison only; no Context.
"""
from __future__ import annotations

import pytest

from md_tools.rest2.regions import claimed_region_differences
from md_tools.rest2.selection import ScalingSelection, SelectionError

from .selective_rest2_fixture import RESIDUE, build_fixture
from .test_selective_rest2_selection import _exclusions, _select, mapping_for


@pytest.fixture(scope="module")
def fx():
    return build_fixture()


@pytest.fixture(scope="module")
def built(fx, tmp_path_factory):
    """The fixture on disk, with the ligands' SDFs a region's classification reads."""
    return fx.write(tmp_path_factory.mktemp("built"))


#: The directory records and claims resolve against unless a test gives its own.
WHERE = {}


@pytest.fixture(autouse=True)
def _where(built):
    WHERE["dir"] = built


def _record(fx, config, where=None):
    return _select(fx, config, where or WHERE["dir"])[1].to_document()


def _compare(fx, claim, document, where=None, **kwargs):
    where = where or WHERE["dir"]
    kwargs.setdefault("ligand_mapping", mapping_for(fx))
    return claimed_region_differences(fx.topology, claim, config_dir=where,
                                      implicit=False, selection_document=document, **kwargs)


BASE = {"backbone_scaling_list": ":2-3", "sidechain_scaling_list": ":4",
        "ligand_scaling_dict": {"L01": {"mask": ":13"}}}


def test_the_same_claim_agrees(fx):
    assert _compare(fx, BASE, _record(fx, BASE)) == []


def test_a_different_spelling_of_the_same_region_agrees(fx):
    claim = dict(BASE, backbone_scaling_list=":2,3")
    assert _compare(fx, claim, _record(fx, BASE)) == []


def test_a_different_residue_differs(fx):
    claim = dict(BASE, backbone_scaling_list=":2-4")
    differences = _compare(fx, claim, _record(fx, BASE))
    assert any("hot nonbonded atoms differ" in d for d in differences)
    assert any(d.startswith("residue 4:") and "backbone+sidechain" in d for d in differences)


def test_the_same_residue_in_another_category_differs(fx):
    """PHE as backbone recorded, PHE as sidechain claimed: same residues, different region."""
    recorded = _record(fx, {"backbone_scaling_list": ":4"})
    differences = _compare(fx, {"sidechain_scaling_list": ":4"}, recorded)
    assert "residue 4: recorded PHE as backbone, claimed PHE as sidechain" in differences
    assert any("hot nonbonded atoms differ" in d for d in differences)


def test_the_twin_copy_differs(fx):
    claim = dict(BASE, ligand_scaling_dict={"L01": {"mask": f":{RESIDUE['LGA#2']}"}})
    differences = _compare(fx, claim, _record(fx, BASE))
    assert "residue 13: recorded LGA as ligand, claimed not selected" in differences
    assert "residue 15: recorded not selected, claimed LGA as ligand" in differences


def test_a_relabelled_instance_agrees(fx):
    """A label is provenance: the same copy under another name is the same Hamiltonian."""
    claim = dict(BASE, ligand_scaling_dict={"L99": {"mask": ":13"}})
    assert _compare(fx, claim, _record(fx, BASE)) == []


def test_an_exclusion_file_elsewhere_with_the_same_content_agrees(fx, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    _exclusions(first / "L01.yaml", [("C2", "C3")])
    _exclusions(second / "L01.yaml", [("C2", "C3")])
    claim = dict(BASE, ligand_scaling_dict={"L01": {"mask": ":13",
                                                    "torsion_exclusions": "L01.yaml"}})
    recorded = _record(fx, claim, first)
    assert _compare(fx, claim, recorded, second) == []
    _exclusions(second / "L01.yaml", [("N1", "C4")])
    differences = _compare(fx, claim, recorded, second)
    assert any("torsion exclusions (resolved content)" in d for d in differences), differences


def test_an_exclusion_file_differing_only_by_a_comment_agrees(fx, tmp_path):
    """The file's BYTES are provenance; the bonds it names are the Hamiltonian."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    _exclusions(first / "L01.yaml", [("C2", "C3")])
    _exclusions(second / "L01.yaml", [("C2", "C3")])
    with (second / "L01.yaml").open("a", encoding="utf-8") as handle:
        handle.write("# reviewed 2026-09-19\n")
    claim = dict(BASE, ligand_scaling_dict={"L01": {"mask": ":13",
                                                    "torsion_exclusions": "L01.yaml"}})
    assert _compare(fx, claim, _record(fx, claim, first), second) == []


def test_an_auto_claim_against_a_file_record_differs(fx, tmp_path):
    _exclusions(tmp_path / "L01.yaml", [("C2", "C3")])
    with_file = dict(BASE, ligand_scaling_dict={"L01": {"mask": ":13",
                                                        "torsion_exclusions": "L01.yaml"}})
    differences = _compare(fx, BASE, _record(fx, with_file, tmp_path), tmp_path)
    assert any("torsion exclusions" in d for d in differences), differences


def test_an_explicit_claim_against_a_legacy_record_is_a_sentence_not_a_crash(fx):
    legacy = ScalingSelection(solute_atoms=(0, 1)).to_document()
    differences = _compare(fx, BASE, legacy)
    assert differences == [differences[0]] and "explicit claim against a legacy record" in \
        differences[0]
    v1 = {"format": "md-tools-solute-selection/1.0", "solute_atoms": [0, 1],
          "unscaled_torsion_central_bonds": []}
    assert "a 1.0 (pre-0.6.1) record" in _compare(fx, BASE, v1)[0]


def test_no_claim_against_an_explicit_record_differs_and_against_a_legacy_one_agrees(fx):
    assert "names no selector" in _compare(fx, {}, _record(fx, BASE))[0]
    assert _compare(fx, {}, ScalingSelection(solute_atoms=(0,)).to_document()) == []


def test_a_missing_record_is_a_sentence(fx):
    assert "no recorded selection" in _compare(fx, BASE, None)[0]


def test_an_invalid_claim_is_refused_not_reported(fx):
    with pytest.raises(SelectionError, match="atom selector"):
        _compare(fx, dict(BASE, backbone_scaling_list=":2@CA"), _record(fx, BASE))
    with pytest.raises(SelectionError, match="compact form"):
        _compare(fx, {"ligand_scaling_dict": {"L01": "x.yaml"}}, _record(fx, BASE))


def test_a_record_of_another_topology_differs(fx):
    other = build_fixture(ligand_order=("LGB", "LGA", "LGA"))
    recorded = _record(other, {"backbone_scaling_list": ":2-3"})
    differences = _compare(fx, {"backbone_scaling_list": ":2-3"}, recorded)
    assert any("different topology" in d for d in differences), differences
