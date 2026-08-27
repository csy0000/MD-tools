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

RETROFIT = REPO_ROOT / "scripts" / "retrofit_fair_v030.py"


# --- implementation identity -------------------------------------------------

def test_the_installed_fingerprint_is_deterministic_and_content_addressed():
    """`git_commit` is null from a wheel, so something else must identify the implementation."""
    from md_templates.openmm.provenance_min import installed_fingerprint

    first, second = installed_fingerprint(), installed_fingerprint()
    assert first == second, "the fingerprint must not depend on filesystem ordering"
    assert first["file_count"] > 10
    assert len(first["value"]) == 64


def test_the_fingerprint_changes_when_the_implementation_changes(tmp_path, monkeypatch):
    """A single edited line must produce a different fingerprint, or it identifies nothing."""
    import md_templates.openmm.provenance_min as module

    package = tmp_path / "md_templates" / "openmm"
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
    from md_templates.openmm.provenance_min import implementation_identity

    identity = implementation_identity()
    assert identity["installed_fingerprint"]["value"], "fingerprint must always be present"
    assert identity["version"], "the installed version must be recorded"


def test_environment_records_absent_packages_as_null_not_missing():
    """Null distinguishes 'not installed' from 'nobody looked'."""
    from md_templates.openmm.provenance_min import environment_versions

    versions = environment_versions()
    for name in ("python", "openmm", "openff_toolkit", "parmed", "rdkit"):
        assert name in versions, name


# --- checksum manifests ------------------------------------------------------

def test_the_checksum_manifest_is_deterministic_and_detects_mutation(tmp_path):
    from md_templates.openmm.sysgen import verify_checksum_manifest, write_checksum_manifest

    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub" / "c.txt").write_text("gamma")

    first = write_checksum_manifest(tmp_path).read_text()
    second = write_checksum_manifest(tmp_path).read_text()
    assert first == second, "the manifest must not depend on directory ordering"
    assert "SHA256SUMS" not in first, "the manifest cannot hash itself"
    assert [line.split("  ", 1)[1] for line in first.strip().splitlines()] == \
        ["a.txt", "b.txt", "sub/c.txt"], "paths must be sorted relative POSIX paths"
    assert verify_checksum_manifest(tmp_path)["ok"] is True

    (tmp_path / "a.txt").write_text("tampered")
    result = verify_checksum_manifest(tmp_path)
    assert result["ok"] is False and result["mismatched"] == ["a.txt"]

    (tmp_path / "a.txt").unlink()
    assert verify_checksum_manifest(tmp_path)["missing"] == ["a.txt"]


# --- forcefield.json ---------------------------------------------------------

def _record_for(solvent, route, reported=None, implicit_report=None):
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults
    from md_templates.openmm.forcefield_record import build_forcefield_record

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


def test_a_retained_artifact_collision_is_refused(tmp_path):
    """Two routes can both produce `solute.sdf`; flattening them would mis-checksum one."""
    from md_templates.openmm.config import ConfigError
    from md_templates.openmm.sysgen import _keep_preparation_artifacts

    staging = tmp_path / "_work"
    (staging / "structure").mkdir(parents=True)
    (staging / "structure" / "solute.sdf").write_text("first\n")
    out = tmp_path / "inputs"

    kept = _keep_preparation_artifacts(staging, out)
    assert kept == ["preparation/structure/solute.sdf"], "the subpath must be preserved"

    # an identical rerun is not a collision, and must still be listed
    assert _keep_preparation_artifacts(staging, out) == kept

    (staging / "structure" / "solute.sdf").write_text("DIFFERENT\n")
    with pytest.raises(ConfigError, match="different content"):
        _keep_preparation_artifacts(staging, out)


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
        "md_templates": {"version": "0.3.1", "git_commit": None},
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
    provenance["md_templates"].update({
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


def _run_retrofit(*args):
    return subprocess.run([sys.executable, str(RETROFIT), *[str(a) for a in args]],
                          capture_output=True, text=True, timeout=600)


def _snapshot(root: Path):
    import hashlib

    return {p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_0_3_x_tree_keeps_its_own_ff19sb_opc_identity_and_2nm_box(tmp_path):
    """0.4 changed the defaults. It did not change what 0.3.x ran, and must not say it did.

    The retrofit reads; it never fills a gap from the current configuration. A 0.3.x bundle that
    recorded ff19SB + OPC at 2.0 nm has to come back out saying exactly that -- and a bundle that
    recorded nothing has to come back out saying `unknown`, not `ff14SB`.
    """
    _legacy_fixture(tmp_path)
    out = tmp_path / "fair"
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", out, "--original-input", tmp_path / "source_ALA.pdb")
    assert result.returncode == 0, result.stdout + result.stderr

    text = "".join(p.read_text() for p in sorted(out.rglob("*"))
                   if p.suffix in (".yaml", ".json", ".md"))
    assert "amber19-all.xml" in text and "OPC" in text
    # nothing from the 0.4 defaults may appear anywhere in a record about 0.3.x data
    for value in ("amber14-all.xml", "amber14/tip3p.xml", "TIP3P", "sage-2.2.1", "openff-2.2.1"):
        assert value not in text, f"{value} is a 0.4 default and cannot describe 0.3.x data"

    system = yaml.safe_load((out / "system-record.yaml").read_text())
    resolved = system["resolved_system_config"]
    assert resolved["evidence"] == "recorded"
    assert resolved["value"]["forcefield"]["protein"] == "amber19-all.xml"
    assert resolved["value"]["solvent"]["model"] == "OPC"
    assert resolved["value"]["solvent"]["padding_nm"] == 2.0

    forcefield = json.loads((out / "forcefield.json").read_text())
    assert forcefield["protein"]["openmm_resource"] == "amber19-all.xml"
    assert forcefield["water"]["model"] == "OPC"
    assert forcefield["explicit_solvent"]["padding_nm"] == 2.0
    assert forcefield["package_versions"]["evidence"] == "unknown"


def test_a_0_3_x_tree_with_no_recorded_force_field_stays_unknown(tmp_path):
    """The dangerous case: a gap that the current default would fit neatly into."""
    _legacy_fixture(tmp_path)
    (tmp_path / "inputs" / "resolved_sys.config.yaml").unlink()
    out = tmp_path / "fair"
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", out)
    assert result.returncode == 0, result.stdout + result.stderr

    text = "".join(p.read_text() for p in sorted(out.rglob("*"))
                   if p.suffix in (".yaml", ".json", ".md"))
    for value in ("amber14-all.xml", "amber19-all.xml", "TIP3P", "OPC", "sage-2.2"):
        assert value not in text, f"a missing record must stay unknown, not become {value}"

    system = yaml.safe_load((out / "system-record.yaml").read_text())
    assert system["resolved_system_config"]["evidence"] == "unknown"
    assert system["resolved_system_config"]["value"] is None
    forcefield = json.loads((out / "forcefield.json").read_text())
    assert forcefield["evidence"] == "unknown"
    assert "nothing can be stated" in forcefield["note"]


def test_the_retrofit_never_modifies_the_source(tmp_path):
    """The one guarantee that matters: legacy data is read, never touched."""
    _legacy_fixture(tmp_path)
    before = {**_snapshot(tmp_path / "inputs"), **_snapshot(tmp_path / "MD")}

    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr

    after = {**_snapshot(tmp_path / "inputs"), **_snapshot(tmp_path / "MD")}
    assert after == before, "the retrofit modified the source tree"
    assert not (tmp_path / "inputs" / "SHA256SUMS").exists(), "nothing may be added to inputs/"


def test_a_version_string_alone_is_not_exact_identity(tmp_path):
    """The old fixture has `git_commit: null` and no cMD record, so it must NOT reach A.

    `0.3.1` names a release, not the build that ran; two builds of one version can differ.
    """
    original = _legacy_fixture(tmp_path)
    (tmp_path / "env.yaml").write_text(yaml.safe_dump({"openmm": "8.6.0", "python": "3.12.13"}))

    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out", "--original-input", original,
                           "--environment", tmp_path / "env.yaml")
    assert result.returncode == 0, result.stdout + result.stderr
    classification = json.loads(
        (tmp_path / "out" / "validation.json").read_text())["classification"]
    assert classification["grade"] == "B", classification
    codes = {r["code"] for r in classification["reasons"]}
    assert "implementation_identity_not_exact" in codes, codes
    assert "cmd_runtime_record_missing" in codes, codes


def test_a_genuinely_complete_fixture_grades_a(tmp_path):
    """Exact identity plus every record the declared protocol requires."""
    original = _complete_fixture(tmp_path)
    (tmp_path / "env.yaml").write_text(yaml.safe_dump({"openmm": "8.6.0", "python": "3.12.13"}))

    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out", "--original-input", original,
                           "--environment", tmp_path / "env.yaml")
    assert result.returncode == 0, result.stdout + result.stderr
    validation = json.loads((tmp_path / "out" / "validation.json").read_text())
    assert validation["classification"]["grade"] == "A", validation["classification"]["reasons"]
    assert validation["source_verification"]["unmodified"] is True


def test_the_verified_original_input_is_retained_and_checksummed(tmp_path):
    original = _complete_fixture(tmp_path)
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                  "--output", tmp_path / "out", "--original-input", original)
    kept = tmp_path / "out" / "original_inputs" / original.name
    assert kept.is_file(), "a verified original must be retained in the candidate"
    assert kept.read_bytes() == original.read_bytes()

    record = yaml.safe_load((tmp_path / "out" / "system-record.yaml").read_text())
    entry = record["original_input"]["value"]
    assert entry["retained_path"] == f"original_inputs/{original.name}"
    manifest = (tmp_path / "out" / "SHA256SUMS").read_text()
    assert f"original_inputs/{original.name}" in manifest


def test_the_source_verification_is_a_real_before_after_comparison(tmp_path):
    """`source_modified: false` written unconditionally is a claim, not a check."""
    _legacy_fixture(tmp_path)
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                  "--output", tmp_path / "out")
    verification = json.loads(
        (tmp_path / "out" / "validation.json").read_text())["source_verification"]
    assert "SHA-256" in verification["method"] and "mtime" in verification["method"]
    assert verification["files_before"] == verification["files_after"] > 0
    assert verification["changed"] == [] and verification["added"] == []
    assert verification["removed"] == [] and verification["unmodified"] is True


@pytest.mark.parametrize("output_name", ["fair-registration", "out"])
def test_every_manifest_path_resolves_for_any_output_name(tmp_path, output_name):
    """The sidecar prefix was hardcoded, so any other output name broke every path."""
    import hashlib

    original = _complete_fixture(tmp_path)
    out = tmp_path / output_name
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", out, "--original-input", original)
    assert result.returncode == 0, result.stdout + result.stderr

    lines = [l for l in (out / "SHA256SUMS").read_text().splitlines() if l.strip()]
    assert lines
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = tmp_path / relative           # one documented root: the common project root
        assert path.is_file(), f"{relative} does not resolve from the project root"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    assert any(l.endswith(f"{output_name}/system-record.yaml") for l in lines), \
        f"the sidecar prefix must be the real directory name, not a hardcoded one"


def test_no_record_contains_a_required_absolute_path(tmp_path):
    original = _complete_fixture(tmp_path)
    (tmp_path / "env.yaml").write_text(yaml.safe_dump({"openmm": "8.6.0"}))
    out = tmp_path / "out"
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD", "--output", out,
                  "--original-input", original, "--environment", tmp_path / "env.yaml")

    for name in ("system-record.yaml", "run-record.yaml", "file-inventory.yaml", "SHA256SUMS"):
        text = (out / name).read_text()
        assert str(tmp_path) not in text, f"{name} embeds an absolute path"
        assert "common_root" not in text, f"{name} still records an absolute common_root"


def test_without_the_original_input_the_grade_drops_to_b_with_reasons(tmp_path):
    _legacy_fixture(tmp_path)
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out")
    assert result.returncode == 0
    classification = json.loads((tmp_path / "out" / "validation.json").read_text())["classification"]
    assert classification["grade"] == "B"
    codes = {r["code"] for r in classification["reasons"]}
    assert "original_input_absent" in codes, codes
    assert all(r["detail"] for r in classification["reasons"]), "every reason must be readable"


def test_a_missing_md_config_grades_c(tmp_path):
    _legacy_fixture(tmp_path)
    (tmp_path / "MD" / "md.config.yaml").unlink()
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out")
    assert result.returncode == 0
    classification = json.loads((tmp_path / "out" / "validation.json").read_text())["classification"]
    assert classification["grade"] == "C", classification


def test_a_mismatched_original_input_is_a_hard_failure(tmp_path):
    _legacy_fixture(tmp_path)
    wrong = tmp_path / "wrong.pdb"
    wrong.write_text("NOT THE FILE THIS SYSTEM WAS BUILT FROM\n")
    # named as the recorded input so the name lookup succeeds and only the hash disagrees
    renamed = tmp_path / "source_ALA.pdb"
    renamed.write_text(wrong.read_text())

    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out", "--original-input", renamed)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "mismatch" in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_a_non_empty_output_directory_is_refused(tmp_path):
    _legacy_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "already-here.txt").write_text("do not clobber me")
    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD", "--output", out)
    assert result.returncode == 2
    assert (out / "already-here.txt").read_text() == "do not clobber me"


def test_evidence_labels_are_used_and_inferred_is_never_one(tmp_path):
    _legacy_fixture(tmp_path)
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                  "--output", tmp_path / "out")
    text = ((tmp_path / "out" / "system-record.yaml").read_text()
            + (tmp_path / "out" / "run-record.yaml").read_text())
    assert "evidence: recorded" in text
    assert "evidence: unknown" in text
    assert "inferred" not in text, "a guess must never be labelled as a scientific value"

    counts = json.loads((tmp_path / "out" / "validation.json").read_text())["evidence_counts"]
    assert set(counts) <= {"recorded", "derived", "user_supplied", "unknown"}, counts


def test_the_retrofit_assigns_no_dataset_id_and_moves_no_data(tmp_path):
    """Identity, storage and lifecycle belong to MD-data, not here."""
    _legacy_fixture(tmp_path)
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                  "--output", tmp_path / "out")
    validation = json.loads((tmp_path / "out" / "validation.json").read_text())
    joined = " ".join(validation["not_performed"]).lower()
    assert "dataset identifier" in joined and "$md_data" in joined
    assert validation["handoff_required_from_md_data"]

    blob = " ".join(p.read_text() for p in (tmp_path / "out").iterdir() if p.is_file())
    assert "doi:" not in blob.lower(), "no identifier may be minted here"


def test_the_reconstructed_command_is_never_called_the_original(tmp_path):
    _legacy_fixture(tmp_path)
    _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                  "--output", tmp_path / "out")
    run = yaml.safe_load((tmp_path / "out" / "run-record.yaml").read_text())
    entry = run["reconstructed_command"]
    assert entry["evidence"] == "derived"
    assert "was not recorded" in entry["source"]


def test_the_retrofit_needs_only_python_and_pyyaml():
    """It must run where OpenMM, CUDA and the checkout are absent."""
    source = RETROFIT.read_text()
    for forbidden in ("import openmm", "from openmm", "import md_templates", "from md_templates",
                      "requests", "urllib"):
        assert forbidden not in source, forbidden
