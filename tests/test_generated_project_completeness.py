"""A generated REST2 project must contain every module its driver imports.

The generated directory runs with nothing but its own files on the path. A helper left out of the
copy list is not a missing feature -- it is an ImportError the moment the ladder launches, and it
is invisible to every test that imports the templates from the source tree, where all of them are
present. That is how three new modules (`rem_log`, `amber_trajectory`, `state_trajectories`)
reached a released copy list without being copied.

So the list is checked against the driver's ACTUAL import-time closure, computed from the source,
rather than against a list someone remembered to update.

PLATFORM_POLICY_EXEMPTION: static analysis and file copying. Nothing runs.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "src" / "md_templates" / "openmm" / "templates"
SIMPLE = REPO / "src" / "md_templates" / "openmm" / "simple.py"

LOCAL = {path.name for path in TEMPLATES.glob("*.py")}


def _module_level_imports(name):
    """Only what is imported when the module is IMPORTED. A lazy import inside a function is a
    runtime need, not a launch-time one, and the two fail very differently."""
    tree = ast.parse((TEMPLATES / name).read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        found += [f"{module}.py" for module in modules if f"{module}.py" in LOCAL]
    return found


def import_closure(entry):
    seen, queue = set(), [entry]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue += _module_level_imports(name)
    return seen


def copied_helpers():
    """The list `simple.py` copies into a generated REST2 directory."""
    tree = ast.parse(SIMPLE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "helpers"
                and isinstance(node.value, ast.List)):
            return {element.value for element in node.value.elts
                    if isinstance(element, ast.Constant)}
    raise AssertionError("no `helpers = [...]` list found in simple.py")


def test_every_module_the_driver_imports_is_copied_into_a_generated_project():
    needed = import_closure("replica_driver.py") - {"replica_driver.py"}
    missing = sorted(needed - copied_helpers())
    assert not missing, (
        f"a generated REST2 directory would not contain {missing}, and importing "
        f"replica_driver there raises ImportError at launch")


def test_the_driver_itself_is_copied():
    assert "replica_driver.py" in copied_helpers()


def test_the_new_state_trajectory_modules_are_among_them():
    """Named explicitly: these are the three that were missed, and a regression here is silent."""
    copied = copied_helpers()
    for name in ("rem_log.py", "amber_trajectory.py", "state_trajectories.py"):
        assert name in copied, name


def test_no_helper_is_listed_that_does_not_exist():
    """A name in the list that is not a file makes generation fail with a copy error."""
    missing = sorted(name for name in copied_helpers() if not (TEMPLATES / name).is_file())
    assert not missing, missing
