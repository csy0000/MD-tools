"""Every module the replica driver imports must actually be reachable when a ladder launches.

The old failure this guards against: the runtime modules were COPIED into each generated project
from a hand-maintained list, and a helper left off that list was not a missing feature -- it was an
ImportError the moment the ladder launched, invisible to every test that imported the templates
from the source tree where all of them are present. Three modules (`rem_log`, `amber_trajectory`,
`state_trajectories`) reached a released copy list that way.

MD-tools removed the copy list: `md_tools.runtime.replica` puts the INSTALLED templates directory
on `sys.path`, so the executor's bare `from replica_driver import ReplicaRun` resolves from the
distribution. The list cannot go stale because there is no list.

What can still go wrong is packaging: a module present in the source tree but not shipped in the
wheel fails in exactly the same way, and just as invisibly. So the closure is still computed from
the driver's actual imports, and checked against the directory the runtime will really use.

PLATFORM_POLICY_EXEMPTION: static analysis only. Nothing runs.
"""
from __future__ import annotations

import ast
from pathlib import Path

from md_tools.runtime.replica import templates_directory

TEMPLATES = templates_directory()
LOCAL = {path.name for path in TEMPLATES.glob("*.py")}

#: What the executor imports by bare name, and therefore what must sit beside it on sys.path.
ROOTS = ("replica_executor.py", "replica_driver.py", "replica_runtime.py")


def _module_level_imports(name):
    """Only what is imported when the module is IMPORTED.

    A lazy import inside a function is a runtime need, not a launch-time one, and the two fail very
    differently: one is an error the moment `mpiexec` starts, the other only if that path is taken.
    """
    tree = ast.parse((TEMPLATES / name).read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        for module in modules:
            candidate = f"{module.split('.')[0]}.py"
            if candidate in LOCAL:
                found.append(candidate)
    return found


def _closure(*roots):
    """Every local module reachable from `roots` through module-level imports."""
    seen, queue = set(), list(roots)
    while queue:
        name = queue.pop()
        if name in seen or name not in LOCAL:
            continue
        seen.add(name)
        queue.extend(_module_level_imports(name))
    return seen


def test_the_templates_directory_is_the_installed_one():
    """Not a path assembled from the repository root: what the RUNTIME will put on sys.path."""
    assert TEMPLATES.is_dir(), TEMPLATES
    assert (TEMPLATES / "replica_driver.py").is_file()


def test_every_module_the_driver_imports_is_present_beside_it():
    closure = _closure(*ROOTS)
    missing = sorted(name for name in closure if not (TEMPLATES / name).is_file())
    assert not missing, (
        f"the replica driver's import closure needs {missing}, which are not in "
        f"{TEMPLATES}. A ladder launched from an installed wheel would fail with ImportError.")


def test_the_state_trajectory_modules_are_in_the_closure():
    """The three that were once missing. Named explicitly so the regression stays named."""
    closure = _closure(*ROOTS)
    for name in ("rem_log.py", "amber_trajectory.py", "state_trajectories.py"):
        assert name in closure, f"{name} is no longer reachable from the driver"


def test_the_closure_is_actually_computed_and_not_empty():
    """A closure that silently came out empty would make every assertion above vacuous."""
    closure = _closure(*ROOTS)
    assert len(closure) >= 10, sorted(closure)
    assert "replica_driver.py" in closure
