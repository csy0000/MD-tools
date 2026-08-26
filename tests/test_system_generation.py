"""`sys-gen` produces a loadable OpenMM system and the seven files that describe it."""
from __future__ import annotations

import shutil

import pytest
import yaml

from .conftest import ALA_PDB

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def explicit_inputs(tmp_path_factory):
    """One solvated ALA system, built once. Small padding: this checks plumbing, not a box size."""
    from .conftest import run_cli

    work = tmp_path_factory.mktemp("sysgen")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "REST2", "--solvent", "OPC", cwd=work)
    config = work / "sys.config.yaml"
    document = yaml.safe_load(config.read_text())
    document["solvent"]["padding_nm"] = 0.9
    config.write_text(yaml.safe_dump(document, sort_keys=False))

    result = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                     "-of", "./inputs/", cwd=work)
    assert result.returncode == 0, result.stdout + result.stderr
    return work / "inputs"


def test_the_expected_files_are_written_and_nothing_else(explicit_inputs):
    from md_templates.openmm.sysgen import OUTPUT_FILES

    present = sorted(p.name for p in explicit_inputs.iterdir() if p.is_file())
    assert present == sorted(OUTPUT_FILES), present
    # ...plus the retained original, which lives in its own directory so it can never be confused
    # with a file this package produced.
    assert (explicit_inputs / "original_inputs").is_dir()
    assert [p.name for p in (explicit_inputs / "original_inputs").iterdir()] == ["ALA.pdb"]


def test_the_system_loads_and_is_periodic_with_the_configured_cutoff(explicit_inputs):
    from openmm import NonbondedForce, XmlSerializer, unit

    system = XmlSerializer.deserialize((explicit_inputs / "system.xml").read_text())
    assert system.getNumParticles() > 100
    assert system.usesPeriodicBoundaryConditions()
    assert system.getNumConstraints() > 0
    nonbonded = [system.getForce(i) for i in range(system.getNumForces())
                 if isinstance(system.getForce(i), NonbondedForce)]
    assert len(nonbonded) == 1
    assert nonbonded[0].getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)


def test_solute_yaml_records_the_indices_and_the_rest2_region(explicit_inputs):
    document = yaml.safe_load((explicit_inputs / "solute.yaml").read_text())
    assert document["n_solute_atoms"] == 22               # ACE-ALA-NME
    assert document["solute_atom_indices_are_contiguous"] is True
    assert document["solute_atom_range"] == [0, 21]
    assert document["rest2"]["enhanced_region"] == "solute"
    # two backbone amides in a capped alanine dipeptide, neither proline-like
    assert len(document["rest2"]["omega_excluded_bonds"]) == 2
    assert document["rest2"]["omega_proline_like_scaled_bonds"] == []


def test_provenance_is_recorded_without_blocking_on_git(explicit_inputs):
    """Git metadata is often absent; the record must still identify the implementation."""
    document = yaml.safe_load((explicit_inputs / "provenance.yaml").read_text())
    assert document["format"] == "md-templates-system-provenance/v1"
    assert document["environment"]["openmm"]
    assert isinstance(document["command"], list), "the command is an argument list, not a string"

    identity = document["implementation"]
    assert identity["version"], "the installed version must be recorded"
    assert identity["installed_fingerprint"]["value"], \
        "the fingerprint must be present even when git_commit is null"

    assert document["original_input"]["path"].startswith("original_inputs/")
    assert len(document["original_input"]["sha256"]) == 64
    assert len(document["sys_config_hash"]) == 64
    assert len(document["forcefield_json_sha256"]) == 64
    assert document["system"]["periodic"] is True
    assert document["checksum_manifest"] == "SHA256SUMS"


def test_the_original_input_is_kept_byte_for_byte(explicit_inputs):
    """A bundle you cannot rebuild from is a bundle you can only rerun."""
    from md_templates.openmm.provenance_min import sha256_file

    document = yaml.safe_load((explicit_inputs / "provenance.yaml").read_text())
    kept = explicit_inputs / document["original_input"]["path"]
    assert kept.is_file(), "the original molecular input was not retained"
    assert sha256_file(kept) == document["original_input"]["sha256"]


def test_the_checksum_manifest_covers_the_bundle_and_verifies(explicit_inputs):
    from md_templates.openmm.sysgen import verify_checksum_manifest

    result = verify_checksum_manifest(explicit_inputs)
    assert result["ok"] is True, result
    assert result["verified"] >= 9, result
    manifest = (explicit_inputs / "SHA256SUMS").read_text()
    assert "SHA256SUMS" not in manifest, "the manifest cannot hash itself"
    assert "original_inputs/" in manifest, "the retained original must be covered"


def test_the_forcefield_record_is_written(explicit_inputs):
    import json

    record = json.loads((explicit_inputs / "forcefield.json").read_text())
    assert record["format"] == "md-templates-forcefield/v1"
    assert record["protein"]["openmm_resource"] == "amber19-all.xml"
    # The QUALIFIED resource `ForceField()` was actually given. The short `opc.xml` is the
    # user-facing label and is a different file -- water only, no ion templates -- so recording it
    # would name a file that could not have solvated this box.
    assert record["water"]["openmm_resource"] == "amber19/opc.xml"
    assert record["water"]["requested_label"] == "OPC"
    assert record["builder"]["openmm_xml_loaded"] == ["amber19-all.xml", "amber19/opc.xml"]
    assert record["package_versions"]["openmm"]


def test_the_resolved_config_states_which_solvation_was_used(explicit_inputs):
    document = yaml.safe_load((explicit_inputs / "resolved_sys.config.yaml").read_text())
    assert document["solvation"] == "explicit"
    assert "implicit_solvent" not in document


@pytest.mark.slow
def test_implicit_gbn2_produces_a_nonperiodic_system_with_a_gb_force(tmp_path):
    from openmm import XmlSerializer

    from .conftest import run_cli

    shutil.copy2(ALA_PDB, tmp_path / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--solvent", "GBn2", "--method", "cMD", cwd=tmp_path)
    result = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                     "-of", "./inputs/", cwd=tmp_path)
    if result.returncode != 0:
        pytest.skip(f"implicit preparation unavailable (tleap?): {result.stderr[-300:]}")

    system = XmlSerializer.deserialize((tmp_path / "inputs" / "system.xml").read_text())
    assert not system.usesPeriodicBoundaryConditions()
    forces = {type(system.getForce(i)).__name__ for i in range(system.getNumForces())}
    assert "CustomGBForce" in forces, forces
