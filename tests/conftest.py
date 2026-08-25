"""Shared fixtures. Deliberately few: these tests check user-visible behaviour, not internals."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALA_PDB = Path(__file__).resolve().parent / "data" / "ALA.pdb"


def run_cli(module: str, *args, cwd: Path | None = None):
    """Invoke an entry point the way a user does."""
    return subprocess.run([sys.executable, "-m", f"md_templates.cli.{module}", *args],
                          capture_output=True, text=True, cwd=str(cwd or REPO_ROOT))


@pytest.fixture
def md_openmm(tmp_path):
    def call(*args, cwd=None):
        return run_cli("md_openmm", *args, cwd=cwd or tmp_path)
    return call


@pytest.fixture
def md_template(tmp_path):
    def call(*args, cwd=None):
        return run_cli("md_template", *args, cwd=cwd or tmp_path)
    return call
