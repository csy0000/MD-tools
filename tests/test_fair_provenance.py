"""FAIR provenance: identity, force-field record, checksums, and the retrofit of 0.3.x data.

Nothing here integrates a molecular system, so nothing here needs a GPU. The runtime records that
DO need one are exercised in test_stage_layout.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .conftest import REPO_ROOT



# --- implementation identity -------------------------------------------------

def test_the_installed_fingerprint_is_deterministic_and_content_addressed():
    """`git_commit` is null from a wheel, so something else must identify the implementation."""
    from md_tools.openmm.provenance_min import installed_fingerprint

    first, second = installed_fingerprint(), installed_fingerprint()
    assert first == second, "the fingerprint must not depend on filesystem ordering"
    assert first["file_count"] > 10
    assert len(first["value"]) == 64


def test_the_fingerprint_changes_when_the_implementation_changes(tmp_path, monkeypatch):
    """A single edited line must produce a different fingerprint, or it identifies nothing."""
    import md_tools.openmm.provenance_min as module

    package = tmp_path / "md_tools" / "openmm"
    package.mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    fake = package / "provenance_min.py"
    fake.write_text("")
    monkeypatch.setattr(module, "__file__", str(fake))
    before = module.installed_fingerprint()["value"]
    (package / "a.py").write_text("x = 2\n")
    assert module.installed_fingerprint()["value"] != before


def test_identity_is_never_entirely_null():
    """Git metadata is often unavailable; the record must still say which code this is."""
    from md_tools.openmm.provenance_min import implementation_identity

    identity = implementation_identity()
    assert identity["installed_fingerprint"]["value"], "fingerprint must always be present"
    assert identity["version"], "the installed version must be recorded"


def test_environment_records_absent_packages_as_null_not_missing():
    """Null distinguishes 'not installed' from 'nobody looked'."""
    from md_tools.openmm.provenance_min import environment_versions

    versions = environment_versions()
    for name in ("python", "openmm", "openff_toolkit", "parmed", "rdkit"):
        assert name in versions, name


# --- checksum manifests ------------------------------------------------------



# --- forcefield.json ---------------------------------------------------------

def _record_for(solvent, route, reported=None, implicit_report=None):
    from md_tools.openmm.config import resolve_sys_config
    from md_tools.openmm.defaults import sys_defaults
    from md_tools.openmm.forcefield_record import build_forcefield_record

    resolved = resolve_sys_config(sys_defaults(solvent=solvent, peptide=(route == "peptide")))
    record = {}
    if reported is not None:
        record["forcefield"] = reported
    if implicit_report is not None:
        record["implicit_report"] = implicit_report
    return build_forcefield_record(resolved=resolved, route=route, record=record,
                                   inputs_dir=Path("."), artifacts={})


def test_explicit_peptide_records_the_qualified_water_resource_that_was_loaded():
    """`ForceField()` is given `amber19/opc.xml`; the short `opc.xml` is only the user's label.

    The qualified file also carries the Na+/Cl- templates this box needed, so recording the short
    name describes a file that would have failed to solvate it.
    """
    record = _record_for("OPC", "peptide", reported={
        "xml": ["amber19-all.xml", "amber19/opc.xml"], "route": "peptide",
        "xml_includes": {"amber19-all.xml": ["amber19/protein.ff19SB.xml"],
                         "amber19/opc.xml": []},
        "protein_forcefield": "amber19-all.xml", "water": "amber19/opc.xml", "ligand": None,
        "nonbonded": {"method": "PME", "cutoff_nm": 1.0, "switching": False,
                      "switch_distance_nm": None, "dispersion_correction": True,
                      "ewald_error_tolerance": 0.0005}})

    assert record["protein"]["openmm_resource"] == "amber19-all.xml"
    # the wrapper is a manifest of includes; the record must name the file that carried ff19SB
    assert record["protein"]["openmm_resource_includes"] == ["amber19/protein.ff19SB.xml"]
    assert record["protein"]["tleap_resource"] is None
    assert record["water"]["openmm_resource"] == "amber19/opc.xml"
    assert record["water"]["requested_label"] == "OPC"
    assert record["builder"]["openmm_xml_loaded"] == ["amber19-all.xml", "amber19/opc.xml"]
    assert record["builder"]["route"] == "openmm.app.ForceField.createSystem"
    # the whole nonbonded treatment, not just the method: each of these changes the energy
    nonbonded = record["nonbonded"]
    assert nonbonded["method"] == "PME"
    assert nonbonded["cutoff_nm"] == 1.0
    assert nonbonded["switching"] is False and nonbonded["switch_distance_nm"] is None
    assert nonbonded["dispersion_correction"] is True
    assert nonbonded["ewald_error_tolerance"] == 0.0005


def test_the_ligand_only_route_records_no_protein_force_field():
    """It deliberately does not load ff19SB; naming it would attribute parameters to nothing."""
    record = _record_for("OPC", "ligand", reported={
        "xml": ["amber19/opc.xml"], "route": "ligand", "protein_forcefield": None,
        "water": "amber19/opc.xml",
        "nonbonded": {"method": "PME", "cutoff_nm": 1.0, "switching": False,
                      "switch_distance_nm": None, "dispersion_correction": True,
                      "ewald_error_tolerance": 0.0005},
        "ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc"}})

    assert record["protein"]["forcefield"] is None
    assert record["protein"]["openmm_resource"] is None
    assert "loads no protein force field" in record["protein"]["note"]
    assert "amber19-all.xml" not in (record["builder"]["openmm_xml_loaded"] or [])
    # the exact resource, kept distinct from the human-facing label the user typed
    assert record["ligand"]["openff_resource"] == "openff-2.2.1"
    assert record["ligand"]["requested_label"] == "sage-2.2.1"
    assert record["ligand"]["charge_method"] == "am1bcc"


def test_implicit_peptide_records_tleap_not_an_openmm_protein_xml():
    """No OpenMM protein XML constructed this System; tleap wrote the topology."""
    record = _record_for("GBn2", "peptide", implicit_report={
        "protein_forcefield": "leaprc.protein.ff19SB", "implicit_model": "GBn2",
        "radii": "mbondi3"})

    assert record["protein"]["openmm_resource"] is None, "no OpenMM protein XML was loaded"
    assert record["protein"]["tleap_resource"] == "leaprc.protein.ff19SB"
    assert record["builder"]["route"] == "parmed.Structure.createSystem"
    assert record["builder"]["openmm_xml_loaded"] is None
    assert record["builder"]["tleap_used"] is True
    assert record["implicit_solvent"]["model"] == "GBn2"
    assert record["implicit_solvent"]["radii"] == "mbondi3"
    assert record["water"]["openmm_resource"] is None
    assert record["nonbonded"]["method"] == "NoCutoff"
    assert record["explicit_solvent"] is None


def test_implicit_ligand_records_the_openff_provenance_used_before_parmed():
    record = _record_for("GBn2", "ligand", implicit_report={
        "implicit_model": "GBn2", "radii": "mbondi3", "protein_forcefield": None,
        "forcefield_info": {"route": "ligand", "protein_forcefield": None, "water": None,
                            "ligand": {"forcefield": "openff-2.2.1",
                                       "charge_method": "am1bcc"}}})

    assert record["protein"]["openmm_resource"] is None
    assert record["ligand"]["openff_resource"] == "openff-2.2.1"
    assert record["ligand"]["charge_method"] == "am1bcc"
    assert record["builder"]["route"] == "parmed.Structure.createSystem"
    assert record["implicit_solvent"]["radii"] == "mbondi3"




# --- the 0.3.x retrofit ------------------------------------------------------

def _legacy_fixture(root: Path, *, with_hash: bool = True) -> Path:
    """A synthetic 0.3.x tree. Small and fixed; no private data, no $MD_DATA, no OpenMM."""
    import hashlib

    (root / "inputs").mkdir(parents=True)
    (root / "MD" / "eq" / "nvt_1kcal").mkdir(parents=True)
    (root / "MD" / "cMD").mkdir(parents=True)

    original = root / "source_ALA.pdb"
    original.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\nEND\n")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()

    for name in ("system.xml", "topology.pdb", "initial_state.xml", "solute.pdb"):
        (root / "inputs" / name).write_text(f"<synthetic-{name}/>\n")
    (root / "inputs" / "solute.yaml").write_text("n_solute_atoms: 22\n")
    (root / "inputs" / "resolved_sys.config.yaml").write_text(yaml.safe_dump({
        "solvation": "explicit",
        "forcefield": {"protein": "amber19-all.xml", "water": "opc.xml"},
        "solvent": {"model": "OPC", "box_shape": "dodecahedron", "padding_nm": 2.0},
        "constraints": {"type": "HBonds", "rigid_water": True},
        "solute": {"peptide": True}}))
    (root / "inputs" / "provenance.yaml").write_text(yaml.safe_dump({
        "md_tools": {"version": "0.3.1", "git_commit": None},
        "generated": {"timestamp": "2026-08-20T10:00:00Z",
                      "input_hashes": ({original.name: digest} if with_hash else {})}}))
    (root / "MD" / "md.config.yaml").write_text(yaml.safe_dump({
        "methods": ["cMD"], "cMD": {"ensemble": "NPT", "duration_ns": 1.0}}))
    (root / "MD" / "eq" / "nvt_1kcal" / "resolved_stage.yaml").write_text("stage: nvt_1kcal\n")
    (root / "MD" / "cMD" / "whole_system.dcd").write_bytes(b"frames")
    return original


def _complete_fixture(root: Path) -> Path:
    """A 0.3.x tree that genuinely earns grade A: exact identity and every required record."""
    original = _legacy_fixture(root)
    provenance = yaml.safe_load((root / "inputs" / "provenance.yaml").read_text())
    provenance["md_tools"].update({
        "git_commit": "0" * 40,
        "installed_fingerprint": "a" * 64,
    })
    (root / "inputs" / "provenance.yaml").write_text(yaml.safe_dump(provenance))
    # md.config.yaml declares cMD only, so a cMD record is what closure requires here.
    (root / "MD" / "cMD" / "resolved_run.yaml").write_text(
        yaml.safe_dump({"record_kind": "cmd_run", "status": "completed"}))
    (root / "MD" / "minimization").mkdir(parents=True, exist_ok=True)
    (root / "MD" / "minimization" / "resolved_stage.yaml").write_text("stage: minimization\n")
    return original




def _snapshot(root: Path):
    import hashlib

    return {p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}




































# --- the inventory that replaced sys-gen's checksum manifest ---------------------------------------
#
# `write_checksum_manifest` / `verify_checksum_manifest` went with sys-gen. Every property they
# guaranteed is now `md_tools.registry.inventory`'s to guarantee, because that is what writes the
# SHA256SUMS a registered dataset carries. The assertions are preserved verbatim in meaning.

def test_the_inventory_is_deterministic_and_detects_mutation(tmp_path):
    from md_tools.registry import inventory
    from md_tools.registry.errors import RegistrationError

    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub" / "c.txt").write_text("gamma")

    entries = inventory.build(tmp_path)
    assert inventory.build(tmp_path) == entries, "the inventory must not depend on directory order"
    assert [e["path"] for e in entries] == ["a.txt", "b.txt", "sub/c.txt"], (
        "paths must be sorted relative POSIX paths")

    written = inventory.write(tmp_path, entries)
    assert written.name == "SHA256SUMS"
    assert "SHA256SUMS" not in written.read_text(), "the manifest cannot hash itself"
    inventory.verify(tmp_path, entries)                      # clean: does not raise

    (tmp_path / "a.txt").write_text("tampered")
    with pytest.raises(RegistrationError, match="a.txt"):
        inventory.verify(tmp_path, entries)

    (tmp_path / "a.txt").unlink()
    with pytest.raises(RegistrationError, match="missing: a.txt"):
        inventory.verify(tmp_path, entries)


def test_the_inventory_refuses_a_symlink_rather_than_following_it(tmp_path):
    """`_keep_preparation_artifacts` refused a colliding retained artifact; the same class of
    problem -- a file that is not what it appears to be -- is refused here, where the bytes that
    will be moved are decided."""
    from md_tools.registry import inventory
    from md_tools.registry.errors import RegistrationError

    (tmp_path / "real.txt").write_text("content")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    with pytest.raises(RegistrationError, match="symlink"):
        inventory.build(tmp_path)
