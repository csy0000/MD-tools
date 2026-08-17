"""Every import in the package resolves — including the deferred ones inside functions.

This exists because of two real failures during Phase 3. Moving modules rewrote import statements,
and the ones *inside functions* are invisible to the non-slow suite: nothing imports them until the
function runs, and the functions in question only run in an end-to-end prepare. Both breaks
therefore survived a green 577-test run and surfaced four minutes into a slow gate:

    ImportError: cannot import name 'DEFAULTS' from 'md_templates.core.config'
    ModuleNotFoundError: No module named 'md_templates.core.config.adapter'

A deferred import is a promise that a name will be there later. This checks the promise now, in a
second, by walking the AST of every module and resolving every import target it names — without
executing any of the functions.

It complements rather than replaces end-to-end tests: it proves the module and the attribute exist,
not that calling the function does the right thing.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "md_templates"

#: Third-party packages that may legitimately be absent in a minimal environment. Their imports are
#: all deferred by design, and the point of this file is our own wiring, not the user's install.
OPTIONAL_ROOTS = {"openmm", "openff", "rdkit", "mdtraj", "parmed", "openmmtools", "yaml", "numpy",
                  "scipy", "pandas", "pytest"}


def module_name_for(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(relative.parts).removesuffix(".__init__")


def iter_source_modules():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "_packaged" in path.parts:
            continue
        yield path, module_name_for(path)


def resolve_relative(module_name: str, is_package: bool, level: int, target: str | None) -> str:
    """Turn a relative import inside `module_name` into an absolute module path."""
    parts = module_name.split(".")
    if not is_package:
        parts = parts[:-1]
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, target]) if target else ".".join(base)


def import_targets(path: Path, module_name: str):
    """Yield `(module, attribute_or_None)` for every import anywhere in the file."""
    is_package = path.name == "__init__.py"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            target = (resolve_relative(module_name, is_package, node.level, node.module)
                      if node.level else node.module)
            if not target:
                continue
            for alias in node.names:
                yield target, (None if alias.name == "*" else alias.name)


@pytest.mark.parametrize("path,module_name", list(iter_source_modules()),
                         ids=[name for _, name in iter_source_modules()])
def test_every_import_in_the_module_resolves(path: Path, module_name: str):
    failures = []
    for target, attribute in import_targets(path, module_name):
        root = target.split(".")[0]
        if root in OPTIONAL_ROOTS or root in {"__future__"}:
            continue
        if root != "md_templates":
            # standard library and anything else already installed: existence is enough
            if importlib.util.find_spec(target) is None:
                failures.append(f"{target} (module not found)")
            continue
        try:
            module = importlib.import_module(target)
        except Exception as exc:                                       # noqa: BLE001
            failures.append(f"{target}: {type(exc).__name__}: {exc}")
            continue
        if attribute and not hasattr(module, attribute):
            # a submodule that has not been imported yet is still a legitimate target
            if importlib.util.find_spec(f"{target}.{attribute}") is None:
                failures.append(f"{target}.{attribute} (name not found)")
    assert failures == [], f"{module_name} names imports that do not resolve: {failures}"


def test_the_check_covers_deferred_imports_not_just_top_level_ones():
    """A guard that only saw module-level imports would have caught neither Phase 3 failure."""
    path = PACKAGE_ROOT / "engines" / "openmm" / "cli.py"
    targets = list(import_targets(path, "md_templates.engines.openmm.cli"))
    top_level = {alias.name for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.Import) for alias in node.names}
    assert len(targets) > len(top_level) + 5, (len(targets), len(top_level))
    assert any(target.endswith("adapter") for target, _ in targets), \
        "the adapter import that broke Phase 3 must be among the targets this test resolves"


def test_every_module_is_individually_importable():
    """Catches a module that only works because something else imported it first."""
    failures = []
    for _, module_name in iter_source_modules():
        try:
            importlib.import_module(module_name)
        except Exception as exc:                                       # noqa: BLE001
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    assert failures == [], failures


def test_no_module_uses_an_undefined_name():
    """The other half of a move: a function body that still names something no longer imported.

    `forcefield_provenance` moved to the engine and kept using `BUNDLE_SCHEMA_VERSION`, which was no
    longer in scope. Imports all resolved, every module imported cleanly, and the whole non-slow
    suite passed -- the `NameError` only fired when the function ran, four minutes into a slow gate,
    for the third time in this phase.

    pyflakes answers exactly this question statically. Scoped to undefined names and unused imports
    of moved symbols; it is a correctness gate, not a style gate.
    """
    pyflakes = pytest.importorskip("pyflakes.api", reason="pyflakes is not installed")
    from pyflakes.reporter import Reporter

    class _Collect:
        def __init__(self):
            self.problems = []

        def unexpectedError(self, filename, msg):
            self.problems.append(f"{filename}: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):
            self.problems.append(f"{filename}:{lineno}: syntax error: {msg}")

        def flake(self, message):
            text = str(message)
            if "undefined name" in text or "may be undefined" in text:
                self.problems.append(text)

    reporter = _Collect()
    for path, _ in iter_source_modules():
        pyflakes.check(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT)),
                       Reporter(reporter, reporter) if False else reporter)
    assert reporter.problems == [], reporter.problems
