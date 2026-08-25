"""`tests/DEPENDENCIES.yaml` must describe the suite that exists.

A scheduling map is only safe to fan work out from if it is complete: a module missing from it is
a module a parallel run will either skip or misorder. Every allowlist in this repository that
nothing checked has drifted, so this one is checked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS = REPO_ROOT / "tests"
MAP = TESTS / "DEPENDENCIES.yaml"


def _map() -> dict:
    return yaml.safe_load(MAP.read_text())


def _listed() -> set[str]:
    return {entry["module"] for entry in _map()["modules"]}


def _on_disk() -> set[str]:
    return {p.name for p in TESTS.glob("test_*.py")}


def test_every_test_module_is_listed():
    """A module absent from the map is invisible to any scheduler reading it."""
    missing = _on_disk() - _listed()
    assert not missing, (
        f"these test modules are not in DEPENDENCIES.yaml: {sorted(missing)}. Add them with their "
        f"measured cost and what they need, or a parallel run will misorder them.")


def test_the_map_lists_no_module_that_has_been_deleted():
    stale = _listed() - _on_disk()
    assert not stale, f"DEPENDENCIES.yaml names modules that no longer exist: {sorted(stale)}"


def test_every_module_declares_a_cost_and_a_resource_class():
    known = set(_map()["resource_notes"]) | {"pure-python"}
    problems = {}
    for entry in _map()["modules"]:
        if not isinstance(entry.get("seconds"), (int, float)):
            problems[entry["module"]] = "no measured seconds"
        unknown = set(entry.get("needs") or []) - known
        if unknown:
            problems[entry["module"]] = f"undocumented resource class {sorted(unknown)}"
    assert not problems, problems


def test_the_phases_are_ordered_and_declare_their_barriers():
    """The barrier after phase 0 is what stops derived artifacts being regenerated twice."""
    phases = _map()["phases"]
    ids = [p["id"] for p in phases]
    assert ids == sorted(ids), f"phases must read in execution order: {ids}"
    source, derived = phases[0], phases[1]
    assert source["barrier_after"] is True, (
        "derived artifacts must not be regenerated while source fixes are still landing")
    assert derived["parallel"] is False


def test_the_declared_critical_path_is_still_the_slowest_module():
    """If the map names the wrong floor, its whole speedup argument is wrong.

    Recorded rather than computed at runtime: measuring it here would mean running the suite from
    inside the suite. It is re-measured with `pytest --durations=0` when the numbers matter.
    """
    slowest = max(_map()["modules"], key=lambda e: e["seconds"])
    assert slowest["module"] in MAP.read_text().split("THE CRITICAL PATH IS ONE MODULE.")[1][:600], (
        f"the map's critical-path note does not mention {slowest['module']}, which is now the "
        f"slowest module at {slowest['seconds']} s")


def test_the_non_parallelisable_decisions_are_documented_with_their_reason():
    """Each entry has to say what the WRONG fix looks like, or it cannot prevent it."""
    entries = _map()["decisions_that_do_not_parallelise"]
    assert entries
    for entry in entries:
        assert entry.get("symptom") and entry.get("wrong_fix") and entry.get("why")
