"""The built wheel carries everything the six public commands need.

A generator that works from the checkout and a wheel that is missing the file it copies are the
same bug seen from two places, and only this test sees it from the second.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

pytestmark = pytest.mark.slow

#: Every file `md-gen` copies into a generated project, plus the scripts themselves.
#: Every file `md-gen` copies into a generated project. Derived from mdgen.py rather than kept by
#: hand, because a stale list here passed for a whole release while the wheel was quietly picking
#: up a deleted `run.sh` from a cached build tree.
def _template_files() -> set[str]:
    import re

    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "mdgen.py").read_text(
        encoding="utf-8")
    found = set(re.findall(r'TEMPLATES / "([A-Za-z0-9_.]+)"', source))
    assert found, "mdgen.py copies no templates -- the pattern stopped matching"
    return found


TEMPLATE_FILES = _template_files()


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
    work = tmp_path_factory.mktemp("wheel")
    built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                            "--wheel-dir", str(work / "dist"), str(REPO_ROOT)],
                           capture_output=True, text=True, timeout=900)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    wheels = list((work / "dist").glob("md_tools-*.whl"))
    assert len(wheels) == 1, wheels

    site = work / "site"
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                                "--target", str(site), str(wheels[0])],
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


def test_the_wheel_contains_every_generated_project_file(installed):
    """Every template md-gen copies must be in the wheel, and nothing dead should be."""
    site, _ = installed
    templates = site / "md_tools" / "openmm" / "templates"
    assert templates.is_dir(), "the templates directory did not survive packaging"
    present = {path.name for path in templates.iterdir() if path.is_file()}
    assert TEMPLATE_FILES <= present, f"missing from the wheel: {TEMPLATE_FILES - present}"

    # A template in the wheel that md-gen never copies is dead weight, and usually the sign of a
    # cached build tree shipping a file that was deleted from the checkout.
    tracked = {path.name for path in
               (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates").iterdir()
               if path.is_file()}
    assert present <= tracked, f"the wheel carries files not in the checkout: {present - tracked}"


def test_both_generators_import_from_the_wheel_rather_than_the_checkout(installed):
    """`sys-gen` and `md-gen` are the two scientifically central entry points."""
    site, work = installed
    result = _outside(site, work, "-c", """
import md_tools
from md_tools.openmm.sysgen import generate_system
from md_tools.openmm.mdgen import generate_md
assert callable(generate_system) and callable(generate_md)
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
