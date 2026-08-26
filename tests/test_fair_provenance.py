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

@pytest.mark.parametrize("solvent,implicit", [("OPC", False), ("GBn2", True)])
def test_the_forcefield_record_states_what_applies_and_nulls_what_does_not(solvent, implicit):
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults
    from md_templates.openmm.forcefield_record import build_forcefield_record

    resolved = resolve_sys_config(sys_defaults(solvent=solvent))
    record = build_forcefield_record(resolved=resolved, route="peptide", record={},
                                     inputs_dir=Path("."), artifacts={})
    assert record["format"] == "md-templates-forcefield/v1"
    assert record["protein"]["openmm_resource"] == "amber19-all.xml"

    if implicit:
        assert record["water"]["openmm_resource"] is None, "implicit solvent has no water model"
        assert record["explicit_solvent"] is None
        assert record["implicit_solvent"] == {"model": "GBn2", "radii": "mbondi3"}
        assert record["nonbonded"]["method"] == "NoCutoff", "no box means no PME"
        assert record["nonbonded"]["cutoff_nm"] is None
        assert "ParmEd" in record["builder"]["notes"]
    else:
        assert record["water"]["openmm_resource"] == "opc.xml"
        assert record["implicit_solvent"] is None
        assert record["nonbonded"]["method"] == "PME"
        assert record["explicit_solvent"]["box_shape"] == "dodecahedron"


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


def _run_retrofit(*args):
    return subprocess.run([sys.executable, str(RETROFIT), *[str(a) for a in args]],
                          capture_output=True, text=True, timeout=600)


def _snapshot(root: Path):
    import hashlib

    return {p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


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


def test_a_verified_original_and_environment_grade_a(tmp_path):
    original = _legacy_fixture(tmp_path)
    (tmp_path / "env.yaml").write_text(yaml.safe_dump({"openmm": "8.6.0", "python": "3.12.13"}))

    result = _run_retrofit("--inputs", tmp_path / "inputs", "--md", tmp_path / "MD",
                           "--output", tmp_path / "out", "--original-input", original,
                           "--environment", tmp_path / "env.yaml")
    assert result.returncode == 0, result.stdout + result.stderr
    validation = json.loads((tmp_path / "out" / "validation.json").read_text())
    assert validation["classification"]["grade"] == "A", validation["classification"]
    assert validation["source_modified"] is False


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
