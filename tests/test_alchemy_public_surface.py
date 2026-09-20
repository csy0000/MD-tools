"""`md_tools.alchemy` still exports what it exported, and its entry points still take what they took.

The list is `tests/alchemy_public_surface.py`, which says why it exists: a rewrite once truncated
`campaign.py` and deleted two public functions while every test still passed.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

from tests.alchemy_public_surface import ALCHEMY_SIGNATURES, ALCHEMY_SURFACE


@pytest.mark.parametrize("module_name", sorted(ALCHEMY_SURFACE))
def test_the_module_still_exports_its_public_names(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in ALCHEMY_SURFACE[module_name] if not hasattr(module, name)]
    assert not missing, (
        f"{module_name} no longer exports {missing}. Either the edit that removed them was "
        f"unintended -- a rewrite that truncated the file, as happened on 2026-09-20 -- or the "
        f"removal is deliberate, in which case take the name out of tests/alchemy_public_surface.py "
        f"in the same commit, where a reviewer sees it.")


@pytest.mark.parametrize("target", sorted(ALCHEMY_SIGNATURES))
def test_entry_points_still_take_the_parameters_callers_pass(target):
    module_name, _, attribute = target.rpartition(".")
    function = getattr(importlib.import_module(module_name), attribute)
    parameters = inspect.signature(function).parameters
    missing = [p for p in ALCHEMY_SIGNATURES[target] if p not in parameters]
    assert not missing, (
        f"{target} no longer accepts {missing}. A caller passing one by name would fail, or -- "
        f"worse, for a keyword with a default -- would silently stop getting the behaviour it "
        f"asked for.")


def test_the_surface_list_names_only_things_that_exist():
    """The other direction: a name left behind after a rename guards nothing."""
    stale = []
    for module_name, names in ALCHEMY_SURFACE.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            stale.append(f"{module_name} (no such module)")
            continue
        stale += [f"{module_name}.{n}" for n in names if not hasattr(module, n)]
    for target in ALCHEMY_SIGNATURES:
        module_name, _, attribute = target.rpartition(".")
        if not hasattr(importlib.import_module(module_name), attribute):
            stale.append(f"{target} (no such attribute)")
    assert not stale, f"the surface list names things that do not exist: {stale}"


def test_the_guard_notices_a_deleted_export(monkeypatch):
    """The guard can FAIL: delete an export and the test that covers it must not pass."""
    import md_tools.alchemy.campaign as campaign

    monkeypatch.delattr(campaign, "matched_leg_report")
    with pytest.raises(AssertionError, match="no longer exports"):
        test_the_module_still_exports_its_public_names("md_tools.alchemy.campaign")
