"""`md-template install` must produce an environment the other five commands can actually use.

The failure this file exists to prevent: a command that reports success after installing `openmm`
alone, leaving an environment where `sys-gen` dies part-way through building a system.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_templates.install.openmm import (CONDA_PACKAGES, REQUIRED_EXECUTABLES, REQUIRED_IMPORTS,
                                         _problems, verify_environment, warnings_for)


def test_the_solve_includes_every_dependency_the_workflows_import():
    """openmm alone cannot build a system: OpenFF and AmberTools do the parameterisation."""
    specs = " ".join(CONDA_PACKAGES)
    for package in ("pyyaml", "numpy", "openff-toolkit", "openmmforcefields",
                    "ambertools", "parmed", "rdkit"):
        assert package in specs, f"{package} is not in the solve"


def test_every_required_import_is_actually_imported_by_this_package():
    """A required dependency nobody imports is a dependency bought for nothing."""
    source = "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                       for path in Path(__file__).resolve().parents[1].joinpath("src").rglob("*.py"))
    for module, _package in REQUIRED_IMPORTS:
        root = module.split(".")[0]
        # both spellings: `import parmed` and `from openff.toolkit import Molecule`
        assert f"import {root}" in source or f"from {root}" in source, \
            f"{module} is required but never imported"


def test_a_missing_package_is_named_with_the_package_that_fixes_it():
    report = {"import_errors": {"openff.toolkit": "ModuleNotFoundError: no openff"},
              "executables": {"sqm": "/x/sqm", "antechamber": "/x/antechamber",
                              "tleap": "/x/tleap"},
              "openff_toolkits": ["AmberToolsToolkitWrapper"], "am1bcc_ready": True,
              "cpu_check": "ok", "reference_check": "ok", "cuda_check": "no CUDA platform"}
    problems = _problems(report)
    assert any("openff.toolkit" in p and "openff-toolkit" in p for p in problems), problems


def test_missing_ambertools_executables_are_named():
    report = {"import_errors": {}, "executables": {"sqm": None, "antechamber": None,
                                                   "tleap": "/x/tleap"},
              "openff_toolkits": [], "am1bcc_ready": False,
              "cpu_check": "ok", "reference_check": "ok", "cuda_check": "no CUDA platform"}
    problems = _problems(report)
    assert any("ambertools" in p for p in problems), problems
    for name in ("sqm", "antechamber"):
        assert any(name in p for p in problems), problems
    assert not any("tleap" in p for p in problems), "tleap is present and must not be reported"


def test_an_unusable_cuda_platform_is_a_failure_but_absent_hardware_is_not():
    """A listed CUDA platform that cannot take a step fails at run time, so it fails here."""
    base = {"import_errors": {}, "executables": dict.fromkeys(REQUIRED_EXECUTABLES, "/x"),
            "openff_toolkits": ["AmberToolsToolkitWrapper"], "am1bcc_ready": True,
            "cpu_check": "ok", "reference_check": "ok"}
    broken = dict(base, cuda_available=True, cuda_check="RuntimeError: no CUDA-capable device")
    assert _problems(broken), "an unusable CUDA platform must fail"

    cpu_only = dict(base, cuda_available=False, cuda_check="no CUDA platform")
    assert _problems(cpu_only) == [], "a CPU-only machine is a supported machine"
    assert any("MD_PLATFORM" in note for note in warnings_for(cpu_only)), \
        "a CPU-only machine should be warned that generated runs default to CUDA"


def test_plugins_for_absent_hardware_are_a_note_not_a_failure():
    """conda-forge ships HIP plugins; they cannot load on an NVIDIA box and that is fine."""
    base = {"import_errors": {}, "executables": dict.fromkeys(REQUIRED_EXECUTABLES, "/x"),
            "openff_toolkits": ["AmberToolsToolkitWrapper"], "am1bcc_ready": True,
            "cpu_check": "ok", "reference_check": "ok", "cuda_available": True,
            "cuda_check": "ok",
            "plugin_load_failures": ["Error loading library libOpenMMHIP.so: libhiprtc.so.6"]}
    assert _problems(base) == []
    assert warnings_for(base), "it should still be reported"

    fatal = dict(base, plugin_load_failures=["Error loading library libOpenMMCUDA.so"])
    assert _problems(fatal), "a CUDA plugin failure is fatal"


@pytest.mark.slow
def test_the_environment_running_these_tests_passes_validation():
    """The suite's own environment is a real one, so validating it exercises the whole probe."""
    report = verify_environment(Path(sys.prefix), strict=False)
    assert report["problems"] == [], report["problems"]
    assert report["am1bcc_ready"] is True
    assert report["cpu_check"] == "ok"
    assert set(report["versions"]) >= {"openmm", "openff.toolkit", "parmed", "rdkit"}


@pytest.mark.slow
def test_validation_records_package_versions_in_machine_yaml(tmp_path):
    """Which OpenFF and AmberTools built a system is part of what the system is."""
    stack = tmp_path / "stack"
    run = subprocess.run([sys.executable, "-m", "md_templates.cli.md_template", "init",
                          "--target-dir", str(stack)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

    run = subprocess.run([sys.executable, "-m", "md_templates.cli.md_template", "install",
                          "--target-dir", str(stack), "--validate", sys.prefix],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]

    record = yaml.safe_load((stack / "machine.yaml").read_text())["installed"]["openmm"]
    assert record["problems"] == []
    assert record["package_versions"]["openff.toolkit"]
    assert record["executables"]["sqm"], "AmberTools sqm was not recorded"
    assert record["am1bcc_ready"] is True
    assert record["validated_utc"]


def test_a_prefix_with_no_environment_fails_with_a_clear_message(tmp_path):
    from md_templates.install.openmm import InstallError

    with pytest.raises(InstallError, match="no environment"):
        verify_environment(tmp_path / "nothing")
