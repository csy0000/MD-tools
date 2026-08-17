"""Phase 4 gate: one OpenMM implementation, reachable from every historical import path.

The move is only safe if two things are true at once, and they pull in opposite directions:

* **exactly one implementation exists.** A migration that copies instead of moving leaves two files
  that drift apart, and the second one is always the one someone edits.
* **every path that ever worked still works.** `md_templates.openmm.X`, `python -m
  md_templates.openmm.cli`, and the installed `md-openmm` script are published interfaces.

The compatibility matrix below is the explicit statement of the second, checked rather than
described. `test_only_one_implementation_of_each_module_exists` is the first.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "md_templates"

#: legacy import path -> the module that now holds the code.
COMPATIBILITY_MATRIX = {
    # engine implementation, moved in Phase 4
    "md_templates.openmm.system": "md_templates.engines.openmm.system",
    "md_templates.openmm.solvation": "md_templates.engines.openmm.solvation",
    "md_templates.openmm.equilibration": "md_templates.engines.openmm.equilibration",
    "md_templates.openmm.md": "md_templates.engines.openmm.methods.md",
    "md_templates.openmm.rest2": "md_templates.engines.openmm.methods.rest2",
    "md_templates.openmm.bundle": "md_templates.engines.openmm.bundle",
    "md_templates.openmm.bundlecheck": "md_templates.engines.openmm.bundlecheck",
    "md_templates.openmm.bundleinfo": "md_templates.engines.openmm.bundleinfo",
    "md_templates.openmm.provenance": "md_templates.engines.openmm.provenance",
    "md_templates.openmm.envcheck": "md_templates.engines.openmm.platform",
    "md_templates.openmm.runner": "md_templates.engines.openmm.runner",
    "md_templates.openmm.config": "md_templates.engines.openmm.config",
    "md_templates.openmm.schemas": "md_templates.engines.openmm.schemas",
    "md_templates.openmm.adapter": "md_templates.engines.openmm.adapter",
    "md_templates.openmm.restart": "md_templates.engines.openmm.restart",
    "md_templates.openmm.cli": "md_templates.engines.openmm.cli",
    # engine-neutral contracts, moved in Phase 3
    "md_templates.openmm.runstate": "md_templates.core.persistence",
    "md_templates.openmm.bundlev2": "md_templates.core.bundle",
    "md_templates.openmm.fingerprint": "md_templates.core.fingerprint",
    "md_templates.openmm.spec.resolve": "md_templates.core.config.resolve",
    "md_templates.openmm.spec.models": "md_templates.core.config.models",
    "md_templates.openmm.spec.canonical": "md_templates.core.config.canonical",
    "md_templates.openmm.spec.units": "md_templates.core.config.units",
    "md_templates.openmm.spec.diffs": "md_templates.core.config.diffs",
    "md_templates.openmm.spec.migrate": "md_templates.core.config.migrate",
    "md_templates.openmm.spec.adapter": "md_templates.engines.openmm.adapter",
}


@pytest.mark.parametrize("legacy,current", sorted(COMPATIBILITY_MATRIX.items()))
def test_every_legacy_import_path_resolves_to_the_moved_module(legacy, current):
    assert importlib.import_module(legacy) is importlib.import_module(current)


def test_only_one_implementation_of_each_module_exists():
    """The compatibility namespace must contain aliases and re-exports, never a second copy.

    Measured by what the file DEFINES, not by how long it is. `openmm/__init__.py` is legitimately
    fifty lines of re-exports -- that is the compatibility surface, and counting lines would flag it.
    A module that defines functions or classes, on the other hand, holds behaviour, and two copies
    of behaviour drift apart with the second one being whichever someone edits.
    """
    import ast

    offenders = []
    for path in sorted((PACKAGE / "openmm").rglob("*.py")):
        # `_compat` is the aliasing machinery itself: two small helpers that exist only to make the
        # namespace work, and there is nowhere else for them to live.
        if "__pycache__" in path.parts or path.name == "_compat.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = [node.name for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if defined:
            offenders.append(f"{path.relative_to(PACKAGE)} defines {defined}")
    assert offenders == [], offenders


def test_the_compatibility_namespace_only_re_exports():
    """Every module under the legacy namespace is an alias, a re-export, or the package itself."""
    for path in sorted((PACKAGE / "openmm").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        assert "_compat" in source or path.name in ("__init__.py", "_compat.py"), path.name


def test_the_engine_package_holds_the_implementation():
    """The complement: the code really is somewhere, and that somewhere is the engine tree."""
    engine = PACKAGE / "engines" / "openmm"
    modules = [p for p in engine.rglob("*.py") if "__pycache__" not in p.parts]
    assert len(modules) >= 16, [p.name for p in modules]
    total = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in modules)
    assert total > 4000, total


def test_no_module_outside_the_engine_defines_openmm_simulation_code():
    """A second provider would show up as engine imports outside `engines/`."""
    import ast

    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or "_packaged" in path.parts:
            continue
        if "engines" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] in {"openmm", "openff"}:
                    offenders.append(f"{path.relative_to(PACKAGE)}: {module}")
    assert offenders == [], offenders


# ------------------------------------------------------------------------------------------------
# the command-line entry points, which an alias alone does not cover
# ------------------------------------------------------------------------------------------------

def run_cli(*args: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE.parent)
    return subprocess.run([sys.executable, "-m", "md_templates.openmm.cli", *args],
                          capture_output=True, text=True, timeout=300, env=env,
                          cwd=str(REPO_ROOT.parent))


def test_python_dash_m_on_the_legacy_cli_still_runs():
    """Running a module as `__main__` executes the shim file, not the alias target.

    Without an explicit guard this exits 0 having printed nothing -- a command that silently
    succeeds at doing nothing, which is worse than one that fails.
    """
    result = run_cli("validate-system", "--system", "cyclo_rgdfv", "--no-chemistry")
    assert result.returncode == 0, result.stderr
    assert "cyclo_rgdfv" in result.stdout


def test_the_legacy_cli_help_still_names_every_command():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    for command in ("prepare", "md", "rest2", "bundle", "config", "smoke", "validate-env"):
        assert command in result.stdout, command


def test_the_entry_point_target_is_importable_and_callable():
    """`md-openmm = md_templates.openmm.cli:main` in pyproject must keep resolving."""
    module = importlib.import_module("md_templates.openmm.cli")
    assert callable(module.main)
    assert module.build_parser().prog == "md-openmm"


def test_the_shipped_manifests_travel_with_the_engine():
    from md_templates.engines.openmm.schemas import load_system, shipped_system

    manifests = PACKAGE / "engines" / "openmm" / "manifests"
    assert (manifests / "systems").is_dir() and (manifests / "experiments").is_dir()
    assert not (PACKAGE / "openmm" / "manifests").exists(), "the data must not be duplicated"

    resolved = shipped_system("cyclo_rgdfv")
    assert Path(resolved).is_file() and "engines" in str(resolved), resolved
    assert load_system(resolved) is not None
