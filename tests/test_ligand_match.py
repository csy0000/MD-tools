"""Catalog search: reuse on a match, parameterise on a difference, and a near match is a difference.

One definition decides it (`md_tools.ligands.match.matches`), and both the search and the "differs,
so parameterise" branch consult it. The three things compared are the molecule's topology, its
protonation state, and the charge method including the implementation that produced the numbers;
the force field is compared too.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")

from tests.test_ligand_mapping import _package  # noqa: E402
from tests.test_ligand_packages import _molecule  # noqa: E402


def _request(mol, *, method="am1bcc", forcefield="openff-2.2.1", backend="ambertools-sqm",
             scheme=None):
    """What a build would ask for, without resolving the local toolkit registry."""
    from md_tools.ligands.identity import chemical_state
    from md_tools.ligands.match import CRITERIA_SCHEMA, charge_identity, topology_identity
    from md_tools.ligands.package import _prepared_molecule

    prepared = _prepared_molecule(mol)
    state = chemical_state(prepared)
    return {"schema_version": CRITERIA_SCHEMA,
            "topology": topology_identity(prepared),
            "protonation": {k: state[k] for k in ("digest", "fixed_h_inchikey",
                                                  "net_formal_charge", "canonical_smiles")},
            "charges": charge_identity(method, scheme=scheme or method, backend_id=backend),
            "forcefield": {"family": "smirnoff", "resource": forcefield}}


def test_a_package_declares_what_a_build_must_match(tmp_path):
    package = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    criteria = package.criteria
    assert criteria["parameter_id"] == package.parameter_id
    assert criteria["charges"]["backend_id"] == "ambertools-sqm"
    assert criteria["forcefield"]["resource"] == "openff-2.2.1"
    assert criteria["topology"]["n_heavy_atoms"] == 11
    # It is derived, so editing it by hand is caught rather than believed.
    from md_tools.ligands import PackageError, load_package

    text = (package.path / "parameter.config").read_text()
    (package.path / "parameter.config").write_text(
        text.replace("backend_id: ambertools-sqm", "backend_id: openeye"))
    with pytest.raises(PackageError, match="does not describe this package"):
        load_package(package.path)


@pytest.mark.parametrize("difference", ["none", "protonation", "topology", "charge-backend",
                                        "charge-method", "charge-model", "forcefield"])
def test_only_an_exact_match_permits_reuse(tmp_path, difference):
    from md_tools.ligands.match import matches

    package = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    candidate = copy.deepcopy(package.criteria)
    request = _request(_molecule("CC(=O)Nc1ccc(O)cc1"))
    expected_reason = None
    if difference == "protonation":
        request = _request(_molecule("CC(=O)Nc1ccc([O-])cc1"))
        expected_reason = "protonation"
    elif difference == "topology":
        request = _request(_molecule("CCO"))
        expected_reason = "topology"
    elif difference == "charge-backend":
        request = _request(_molecule("CC(=O)Nc1ccc(O)cc1"), backend="openeye",
                           scheme="am1bccelf10")
        expected_reason = "charges"
    elif difference == "charge-method":
        request = _request(_molecule("CC(=O)Nc1ccc(O)cc1"), method="am1bcc_nagl",
                           backend="openff-nagl")
        expected_reason = "charges"
    elif difference == "charge-model":
        request = _request(_molecule("CC(=O)Nc1ccc(O)cc1"))
        candidate["charges"]["model"] = {"name": "a-model.pt", "sha256": "0" * 64}
        expected_reason = "charges"
    elif difference == "forcefield":
        request = _request(_molecule("CC(=O)Nc1ccc(O)cc1"), forcefield="openff-2.1.0")
        expected_reason = "forcefield"

    verdict = matches(request, candidate)
    assert verdict.matched is (difference == "none")
    if expected_reason is not None:
        assert verdict.reasons[expected_reason] != "same"
        assert verdict.differences and expected_reason in verdict.differences[0]


def test_the_search_reports_what_it_considered_and_why(tmp_path):
    from md_tools.ligands.catalog import search_for_match

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    catalog = tmp_path / "catalog"

    found, report = search_for_match(_request(_molecule("CC(=O)Nc1ccc(O)cc1")), [catalog])
    assert found is not None and found.reference == tyl.reference
    assert report["decision"] == "reuse" and report["matched"] == tyl.reference
    assert all(c["matched"] for c in report["considered"] if c["reference"] == tyl.reference)

    # A protomer of the same compound: same topology, different state, so no reuse -- and the
    # report says which comparison failed rather than only that nothing matched.
    found, report = search_for_match(_request(_molecule("CC(=O)Nc1ccc([O-])cc1")), [catalog])
    assert found is None and report["decision"] == "parameterise"
    tyl_row = next(c for c in report["considered"] if c["reference"] == tyl.reference)
    assert tyl_row["compared"]["topology"] == "same"
    assert "different chemical state" in tyl_row["compared"]["protonation"]

    # An unreadable entry is reported as skipped, not silently ignored: the ethanol package's
    # declaration is corrupted, and ethanol is then searched for.
    ethanol_entry = next((catalog / "CHEMBL545").glob("param_*"))
    (ethanol_entry / "parameter.config").write_text(": not yaml", encoding="utf-8")
    found, report = search_for_match(_request(_molecule("CCO")), [catalog])
    assert found is None and report["decision"] == "parameterise"
    skipped = [c for c in report["considered"] if c.get("skipped")]
    assert len(skipped) == 1 and "could not be read" in skipped[0]["skipped"]


def test_a_package_written_before_the_criteria_file_still_loads_and_is_searchable(tmp_path):
    """A build on an earlier commit wrote three files. Those packages are older, not incomplete.

    Refusing them would break builds and reference bundles that carry a copy, so the criteria are
    derived from metadata.json instead, which is where they come from in the first place.
    """
    from md_tools.ligands import load_package
    from md_tools.ligands.catalog import search_for_match

    package = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    (package.path / "parameter.config").unlink()

    legacy = load_package(package.path)
    assert legacy.criteria["declared_in_package"] is False
    assert legacy.criteria["charges"]["backend_id"] == "ambertools-sqm"
    assert legacy.parameter_id == package.parameter_id

    found, report = search_for_match(_request(_molecule("CCO")), [tmp_path / "catalog"])
    assert found is not None and found.reference == package.reference
    assert report["decision"] == "reuse"


def test_an_empty_or_missing_catalog_means_parameterise(tmp_path):
    from md_tools.ligands.catalog import search_for_match

    found, report = search_for_match(_request(_molecule("CCO")), [tmp_path / "nothing-here"])
    assert found is None and report["decision"] == "parameterise" and report["considered"] == []


def test_a_package_that_cannot_name_its_charge_implementation_refuses_by_name(tmp_path):
    """It is refused, and the refusal names the package -- it does not arrive as a traceback.

    A package whose record cannot say which implementation produced its charges cannot be matched
    against a build's requirements: AM1-BCC through sqm, through OpenEye and through NAGL are three
    different results. What must not happen is a raw ValueError escaping build-top's ligand
    validation, which is what a caller saw when a stale copy under a build's own ligands/ shadowed
    a good catalog entry.
    """
    import json

    from md_tools.ligands import PackageError, load_package

    package = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    metadata = json.loads((package.path / "metadata.json").read_text())
    metadata["charges"].pop("backend_id")
    metadata["charges"].pop("backend", None)
    (package.path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (package.path / "parameter.config").unlink()

    with pytest.raises(PackageError) as refused:
        load_package(package.path)
    message = str(refused.value)
    assert str(package.path) in message
    assert "charge backend" in message or "implementation" in message
    # PackageError is a ValueError, so a caller catching either still catches this one.
    assert isinstance(refused.value, ValueError)
