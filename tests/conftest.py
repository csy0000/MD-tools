"""Shared fixtures. Deliberately few: these tests check user-visible behaviour, not internals."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALA_PDB = Path(__file__).resolve().parent / "data" / "ALA.pdb"


def pytest_addoption(parser):
    parser.addoption(
        "--error-on-skip", action="store_true", default=False,
        help="turn every skip into a failure. Release validation uses this: a scientific smoke "
             "test that skipped because a dependency was missing is a test that did not run, and "
             "reporting that as a pass with a note is how a broken release ships.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.config.getoption("--error-on-skip"):
        report.outcome = "failed"
        report.longrepr = (f"{item.nodeid} skipped, and --error-on-skip forbids skips here:\n"
                           f"{report.longrepr}")


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


def template_module(name: str):
    """Import one of the modules that is copied into a generated project.

    The generated scripts import these by filename beside themselves, so there is no package to
    import them from. Loading them by path is how a test exercises the same code the run does.
    """
    import importlib.util

    path = REPO_ROOT / "src" / "md_templates" / "openmm" / "templates" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_template_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dcd_header(path: Path) -> dict:
    """Frame and atom counts straight out of a DCD header.

    No DCD reader is a declared dependency of this repository, and adding one to assert a frame
    count would be a dependency bought for a test. The header carries both numbers, and the atom
    count is the part that matters: a solute-subset trajectory read against the whole-system
    topology is the mistake being guarded against.
    """
    import struct

    raw = Path(path).read_bytes()
    frames = struct.unpack("<i", raw[8:12])[0]
    interval = struct.unpack("<i", raw[16:20])[0]
    offset = 4 + 84 + 4
    title_bytes = struct.unpack("<i", raw[offset:offset + 4])[0]
    offset += 4 + title_bytes + 4
    atoms = struct.unpack("<i", raw[offset + 4:offset + 8])[0]
    return {"frames": frames, "interval": interval, "atoms": atoms}
