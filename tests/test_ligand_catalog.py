"""The shared parameter catalog: `data-register --ligand-package` and `--find-ligand`."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")

from tests.test_ligand_mapping import _package  # noqa: E402

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm", "data-register"]


def _run(args, tmp_path, md_data):
    """No user configuration at all: the catalog needs $MD_DATA, not an identity."""
    import os

    env = {**os.environ, "MD_DATA": str(md_data), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    env.pop("MD_TOOLS_CONFIG", None)
    return subprocess.run(CLI + args, capture_output=True, text=True, env=env, timeout=600)


def test_register_is_verified_write_once_and_searchable(tmp_path):
    package = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    md_data = tmp_path / "MD_DATA"
    md_data.mkdir()

    dry = _run(["--ligand-package", str(package.path), "--dry-run"], tmp_path, md_data)
    assert dry.returncode == 0, dry.stderr
    assert "would register" in dry.stdout
    assert not (md_data / "parameters").exists()

    first = _run(["--ligand-package", str(package.path)], tmp_path, md_data)
    assert first.returncode == 0, first.stderr
    destination = md_data / "parameters" / "ligands" / "CHEMBL112" / package.parameter_id
    assert destination.is_dir() and "registered" in first.stdout

    again = _run(["--ligand-package", str(package.path)], tmp_path, md_data)
    assert again.returncode == 0 and "already registered, kept" in again.stdout

    verified = _run(["--ligand-package", str(package.path), "--verify-only"], tmp_path, md_data)
    assert verified.returncode == 0, verified.stderr

    found = _run(["--find-ligand", "tyl"], tmp_path, md_data)
    assert found.returncode == 0 and package.reference in found.stdout


def test_a_modified_package_is_not_registered_and_dataset_flags_are_refused(tmp_path):
    package = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    md_data = tmp_path / "MD_DATA"
    md_data.mkdir()
    metadata = json.loads((package.path / "metadata.json").read_text())
    metadata["atoms"][0]["partial_charge_e"] += 0.2
    (package.path / "metadata.json").write_text(json.dumps(metadata))
    refused = _run(["--ligand-package", str(package.path)], tmp_path, md_data)
    assert refused.returncode == 2 and "partial charge" in refused.stderr
    assert not (md_data / "parameters").exists()

    mixed = _run(["--ligand-package", str(package.path), "-year", "2026"], tmp_path, md_data)
    assert mixed.returncode == 2 and "take no dataset options" in mixed.stderr
