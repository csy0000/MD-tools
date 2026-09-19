"""Stated aliases on a REUSED package: never applied, never silent.

Aliases are names, not identity, so a catalog match is reused whatever names the configuration
states, and the reused package keeps the names it was written with. The same configuration writes
its aliases when nothing matches. Before this was recorded, the two outcomes were
indistinguishable in the build record: a stated setting that did its job on one machine and
nothing on another, with no line saying which.
"""
from __future__ import annotations

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")

from tests.test_ligand_mapping import _package  # noqa: E402


def _attach(tmp_path, package_like, *, roots, aliases):
    from md_tools.ligands.build import attach_ligand_package, write_prepared_molecule

    work = tmp_path / f"work{len(list(tmp_path.glob('work*')))}"
    work.mkdir()
    sdf, pdb = work / "prepared.sdf", work / "prepared.pdb"
    write_prepared_molecule(package_like, package_like.mol.GetConformer().GetPositions(),
                            package_like.residue_name, sdf, pdb)
    return attach_ligand_package(
        prepared_sdf=sdf, prepared_pdb=pdb, staging=work,
        settings={"forcefield": package_like.metadata["forcefield"]["resource"],
                  "charge_method": "am1bcc",
                  "compound_id": package_like.compound_id, "aliases": aliases,
                  "parameters": "search", "catalog_roots": roots,
                  "residue_name": package_like.residue_name,
                  "input_name": "in.sdf", "input_sha256": "0" * 64})["record"]


def test_stated_aliases_on_a_catalog_match_are_recorded_as_not_applied(tmp_path):
    from md_tools.ligands.build import aliases_not_applied_notice

    eoh = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    assert eoh.summary()["aliases"] == []

    record = _attach(tmp_path, eoh, roots=[tmp_path / "catalog"], aliases=["ethanol", "EtOH"])
    assert record["how"] == "reused (catalog search)"
    assert record["reference"] == eoh.reference
    assert record["aliases"] == []
    assert record["stated_aliases_not_applied"] == ["EtOH", "ethanol"]
    notice = aliases_not_applied_notice(record)
    assert notice is not None and "NOT applied" in notice and eoh.reference in notice

    # Nothing stated, nothing to say.
    quiet = _attach(tmp_path, eoh, roots=[tmp_path / "catalog"], aliases=[])
    assert quiet["stated_aliases_not_applied"] == []
    assert aliases_not_applied_notice(quiet) is None


def test_the_same_aliases_are_written_when_nothing_matches(tmp_path):
    from md_tools.ligands.build import aliases_not_applied_notice

    eoh = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    record = _attach(tmp_path, eoh, roots=[], aliases=["ethanol", "EtOH"])
    assert record["how"] == "created"
    assert record["aliases"] == ["EtOH", "ethanol"]
    assert record["stated_aliases_not_applied"] == []
    assert aliases_not_applied_notice(record) is None


def test_registering_an_aliased_copy_of_a_registered_package_says_the_names_were_not_added(
        tmp_path):
    """Write-once is correct -- aliases are not identity -- but the dropped names are announced."""
    from md_tools.ligands import import_package_from_system
    from tests.test_ligand_catalog import _run
    from tests.test_ligand_mapping import _charges, _molecule, _system_with_charges

    plain = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    mol = _molecule("CCO")
    aliased = import_package_from_system(
        mol, system=_system_with_charges(mol, _charges(mol)), atom_indices=range(mol.GetNumAtoms()),
        compound_id="CHEMBL545", residue_name="EOH", out_root=tmp_path / "aliased",
        forcefield="sage-2.2.1", charge_method="am1bcc", aliases=["ethanol"],
        charge_provenance={"scheme": "am1bcc", "backend_id": "ambertools-sqm"})
    assert aliased.parameter_id == plain.parameter_id
    md_data = tmp_path / "MD_DATA"
    md_data.mkdir()

    first = _run(["--ligand-package", str(plain.path)], tmp_path, md_data)
    assert first.returncode == 0, first.stderr
    assert "NOTE" not in first.stderr

    again = _run(["--ligand-package", str(aliased.path)], tmp_path, md_data)
    assert again.returncode == 0, again.stderr
    assert "already registered, kept" in again.stdout
    assert "aliases ['ethanol']" in again.stderr and "were NOT added" in again.stderr
