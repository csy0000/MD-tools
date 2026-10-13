"""A bespoke-torsion `.offxml`: what is accepted as a refinement, and how it enters a package's identity.

Each guard below can fail on its own. A stock file must not be accepted as a refinement (it would
make a stock package under another name); a file that is not the base plus additions must not be
accepted (it is a different force field wearing the base's name); a method with no file must be
refused rather than quietly parameterised with the stock force field; and a refined package must
never match a stock request, or two Hamiltonians would be one package.

PLATFORM_POLICY_EXEMPTION: force-field file verification and criteria comparison. No dynamics.
"""
from __future__ import annotations

import pytest

pytest.importorskip("openff.toolkit")

from md_tools.ligands.refinement import (RefinementError, refinement_identity,  # noqa: E402
                                         refinement_record, resolve_refinement)

BASE = "openff-2.2.1"
BASE_FILE = "openff_unconstrained-2.2.1.offxml"


def _force_field():
    from openff.toolkit import ForceField

    return ForceField(BASE_FILE)


def _write(ff, path):
    ff.to_file(str(path))
    return path


@pytest.fixture(scope="module")
def refined(tmp_path_factory):
    """Sage 2.2.1 plus one appended proper torsion, the way BespokeFit appends them."""
    from openff.toolkit.typing.engines.smirnoff.parameters import ProperTorsionHandler

    ff = _force_field()
    handler = ff.get_parameter_handler("ProperTorsions")
    handler.add_parameter({"smirks": "[#6:1]-[#6X3:2](=[#8])-[#7:3]-[#6:4]", "id": "t9001",
                           "periodicity1": 3, "phase1": "0.0 * degree", "k1": "1.5 * kilocalorie_per_mole",
                           "idivf1": "1.0"})
    assert isinstance(handler, ProperTorsionHandler)
    return _write(ff, tmp_path_factory.mktemp("offxml") / "bespoke.offxml")


def test_a_force_field_that_adds_a_torsion_is_a_refinement(refined):
    record = refinement_record("mace-omol-bespokefit", refined, base_forcefield=BASE)
    assert record["added_proper_torsion_ids"] == ["t9001"]
    assert record["method"] == "mace-omol-bespokefit"
    assert record["base_file"] == BASE_FILE and len(record["offxml_sha256"]) == 64
    assert record["reference"]["program"] == "mace-omol"


def test_the_stock_force_field_is_not_a_refinement(tmp_path):
    stock = _write(_force_field(), tmp_path / "stock.offxml")
    with pytest.raises(RefinementError, match="adds no proper torsion"):
        refinement_record("bespokefit", stock, base_forcefield=BASE)


def test_a_force_field_that_changes_a_base_torsion_is_refused(tmp_path):
    ff = _force_field()
    handler = ff.get_parameter_handler("ProperTorsions")
    handler.parameters[0].k1 = handler.parameters[0].k1 * 2
    handler.add_parameter({"smirks": "[#6:1]-[#6:2]-[#6:3]-[#6:4]", "id": "t9002",
                           "periodicity1": 2, "phase1": "0.0 * degree",
                           "k1": "1.0 * kilocalorie_per_mole", "idivf1": "1.0"})
    path = _write(ff, tmp_path / "changed.offxml")
    with pytest.raises(RefinementError, match="does not contain .* unchanged"):
        refinement_record("bespokefit", path, base_forcefield=BASE)


def test_a_different_base_is_refused(refined):
    with pytest.raises(RefinementError, match="unchanged"):
        refinement_record("bespokefit", refined, base_forcefield="openff-2.1.0")


def test_a_missing_file_and_an_unknown_method_are_refused(tmp_path, refined):
    with pytest.raises(RefinementError, match="does not exist"):
        refinement_record("bespokefit", tmp_path / "nope.offxml", base_forcefield=BASE)
    with pytest.raises(RefinementError, match="not one of"):
        refinement_record("mace-omol-gap", refined, base_forcefield=BASE)


def test_a_file_that_is_not_a_force_field_is_refused(tmp_path):
    junk = tmp_path / "junk.offxml"
    junk.write_text("not xml", encoding="utf-8")
    with pytest.raises(RefinementError, match="cannot read|can read"):
        refinement_record("bespokefit", junk, base_forcefield=BASE)


def test_a_relative_file_is_found_beside_the_configuration(refined):
    record = refinement_record("bespokefit", refined.name, base_forcefield=BASE,
                               base_dir=refined.parent)
    assert record["offxml"] == "bespoke.offxml"


def test_none_resolves_to_no_refinement_and_a_method_without_a_file_is_refused():
    assert resolve_refinement({"torsion_refinement": "none"}, base_forcefield=BASE) is None
    with pytest.raises(RefinementError, match="does not run the fit"):
        resolve_refinement({"torsion_refinement": "mace-off23-bespokefit",
                            "refinement_offxml": None}, base_forcefield=BASE)


def test_the_identity_is_the_method_and_the_hash_and_nothing_else(refined):
    record = refinement_record("bespokefit", refined, base_forcefield=BASE)
    assert refinement_identity(record) == {"method": "bespokefit",
                                           "offxml_sha256": record["offxml_sha256"]}
    assert refinement_identity(None) is None


def _metadata(refinement):
    forcefield = {"family": "smirnoff", "resource": BASE}
    if refinement:
        forcefield["refinement"] = refinement
    return {"forcefield": forcefield}


def _request(refinement):
    return {"family": "smirnoff", "resource": BASE,
            "refinement": refinement_identity(refinement)}


def _compare(want, have):
    """The `forcefield` branch of the one comparison, on otherwise identical criteria."""
    from md_tools.ligands.match import CRITERIA_SCHEMA, matches

    shared = {"schema_version": CRITERIA_SCHEMA,
              "topology": {"skeleton_digest": "t", "skeleton_smiles": "C"},
              "protonation": {"digest": "p", "canonical_smiles": "C"},
              "charges": {"method": "am1bcc", "scheme": "am1bcc", "backend_id": "ambertools-sqm",
                          "backend_recorded": True, "model": None}}
    return matches({**shared, "forcefield": want}, {**shared, "forcefield": have})


def test_a_refined_package_never_matches_a_stock_request_or_another_refinement(refined, tmp_path):
    record = refinement_record("mace-omol-bespokefit", refined, base_forcefield=BASE)
    stock = {"family": "smirnoff", "resource": BASE, "refinement": None}

    assert _compare(stock, stock).matched
    assert _compare(_request(record), _request(record)).matched

    verdict = _compare(stock, _request(record))
    assert not verdict.matched and "different torsion refinement" in verdict.reasons["forcefield"]
    assert not _compare(_request(record), stock).matched

    other = dict(record, method="bespokefit")
    assert not _compare(_request(record), _request(other)).matched
    changed = dict(record, offxml_sha256="0" * 64)
    assert not _compare(_request(record), _request(changed)).matched


def test_a_package_written_before_refinement_existed_reads_as_stock():
    from md_tools.ligands.match import _refinement_identity

    assert _refinement_identity(_metadata(None)["forcefield"].get("refinement")) is None


def test_a_stock_package_derives_criteria_without_a_refinement_key():
    """Every `parameter.config` written before refinement existed is re-derived and COMPARED on
    load; an added `refinement: null` made 138 tests' committed packages stop describing
    themselves. Absent is the stock form."""
    from md_tools.ligands.match import _forcefield_criteria

    assert _forcefield_criteria("smirnoff", BASE, None) == {"family": "smirnoff", "resource": BASE}
    assert "refinement" in _forcefield_criteria(
        "smirnoff", BASE, {"method": "bespokefit", "offxml_sha256": "a" * 64})
