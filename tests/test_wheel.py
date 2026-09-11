"""The built wheel carries everything the six public commands need.

A generator that works from the checkout and a wheel that is missing the file it copies are the
same bug seen from two places, and only this test sees it from the second.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

pytestmark = pytest.mark.slow

#: The packages a generated script imports. They are ORDINARY PACKAGES now, so what the wheel must
#: contain is decided by `[tool.setuptools.packages.find]` rather than by a hand-computed import
#: closure over a directory of loose modules. The closure existed because the old layout put those
#: modules on `sys.path` by hand and a missing one would only fail at run time; a package cannot be
#: half-shipped that way.
RUNTIME_PACKAGES = ("md_tools/md", "md_tools/rest2", "md_tools/remd", "md_tools/ais")


def _declared_version() -> str:
    """The one version in pyproject.toml, read without a TOML parser dependency."""
    import re

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version"
    return match.group(1)


def test_the_package_and_project_versions_agree():
    """Two places state the version; a release where they disagree ships a lie in its metadata."""
    import re

    init = (REPO_ROOT / "src" / "md_tools" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    assert match, "md_tools/__init__.py has no __version__"
    assert match.group(1) == _declared_version()


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """Build the wheel and install it into a bare prefix, with no source checkout on the path."""
    from tests.wheel_build import build_wheel_from_copy

    work = tmp_path_factory.mktemp("wheel")
    # From a copy of the working tree, never the checkout: see `tests/wheel_build.py` for the
    # concurrency bug that building in `<repo>/build/` produces under xdist.
    wheel = build_wheel_from_copy(REPO_ROOT, work)

    site = work / "site"
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                                "--target", str(site), str(wheel)],
                               capture_output=True, text=True, timeout=900)
    assert installed.returncode == 0, installed.stdout[-3000:] + installed.stderr[-3000:]
    return site, work


def _outside(site, work, *args):
    """Run a command against the INSTALLED package, from a directory that is not the checkout."""
    import os

    environment = dict(os.environ, PYTHONPATH=str(site))
    environment.pop("PYTHONSTARTUP", None)
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          cwd=str(work), env=environment, timeout=600)


def test_the_wheel_contains_every_runtime_package(installed):
    """A generated script imports `md_tools.md`, `md_tools.remd` or `md_tools.ais`. All of them,
    and the scaler they share, must be in the wheel or the script fails at its first line."""
    site, _ = installed
    for package in RUNTIME_PACKAGES:
        directory = site / package
        assert directory.is_dir(), f"{package} did not survive packaging"
        assert (directory / "__init__.py").is_file(), f"{package} shipped without __init__.py"
        assert any(directory.glob("*.py")), f"{package} shipped empty"

    # The retired layout must not come back: `openmm/templates/` held installed runtime code
    # under a name that said it held templates, and a stale build tree shipping it once caused a
    # deleted module to reappear in a wheel.
    assert not (site / "md_tools" / "openmm" / "templates").exists(), \
        "openmm/templates/ is back in the wheel"


def test_the_public_commands_import_from_the_wheel_rather_than_the_checkout(installed):
    """`build-top`, `build-md` and `data-register` are the three public entry points."""
    site, work = installed
    result = _outside(site, work, "-c", """
import md_tools
from md_tools.build.top import build_topology
from md_tools.build.md import build_scripts
from md_tools.registry.register import register_dataset
from md_tools.ais import run_ais as ais_main
assert callable(build_topology) and callable(build_scripts)
assert callable(register_dataset) and callable(ais_main)
print(md_tools.__file__)
""")
    assert result.returncode == 0, result.stdout + result.stderr
    origin = result.stdout.strip().splitlines()[-1]
    assert origin.startswith(str(site)), origin
    assert str(REPO_ROOT / "src") not in origin, "imported the checkout, not the wheel"


@pytest.mark.parametrize("command", [
    ("md_tools.cli.md_openmm", "--help"),
    ("md_tools.cli.md_openmm", "build-top", "--help"),
    ("md_tools.cli.md_openmm", "build-md", "--help"),
    ("md_tools.cli.md_openmm", "data-register", "--help"),
])
def test_each_public_command_runs_from_outside_the_checkout(installed, command):
    site, work = installed
    result = _outside(site, work, "-m", *command)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_the_built_wheel_carries_the_declared_version(installed):
    """What was built, not what was asked for."""
    _site, work = installed
    wheels = list((work / "dist").glob("md_tools-*.whl"))
    assert len(wheels) == 1, wheels
    built = wheels[0].name.split("-")[1]
    assert built == _declared_version(), f"wheel is {built}, pyproject says {_declared_version()}"


def test_the_installed_package_reports_the_declared_version(installed):
    site, work = installed
    result = _outside(site, work, "-c",
                      "import md_tools; print(md_tools.__version__)")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == _declared_version()


def test_the_console_scripts_are_installed(installed):
    site, _ = installed
    scripts = {path.name for path in (site.parent / "site" / "bin").iterdir()} \
        if (site.parent / "site" / "bin").is_dir() else set()
    # `md-openmm` is the ONLY executable this distribution installs. `md-template` was the
    # environment installer and is retired; a second entry point would be a second way in.
    assert "md-openmm" in scripts, scripts
    assert "md-template" not in scripts, scripts
    assert "openmm-md" not in scripts, scripts
