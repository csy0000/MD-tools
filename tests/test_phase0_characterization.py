"""Phase 0 of the PR3-PR8 migration: what must survive the whole campaign, frozen before it starts.

A structural migration breaks interfaces far more easily than it breaks physics, and it breaks them
quietly: a command loses an option, a symbol stops being importable from where it always was, an
exception's base class changes and a caller's `except` clause silently stops matching. The
compatibility goldens under `tests/goldens/` already freeze the scientific contracts. This file
freezes everything else, and it is deliberately written *before* any module moves.

Three guards, each answering a different way of losing something:

  public surface   every command, option, default, choice, public name and exception MRO
  golden bytes     the seven goldens compared as BYTES, not merely recomputed
  module map       every module accounted for, and every claimed destination actually importable
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "tests" / "baseline" / "public_surface.json"
MODULE_MAP = REPO_ROOT / "tests" / "baseline" / "module_map.json"
GOLDEN_DIR = REPO_ROOT / "tests" / "goldens"
CAPTURE = REPO_ROOT / "scripts" / "capture_public_surface.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import capture_public_surface  # noqa: E402


# ------------------------------------------------------------------------------------------------
# the public surface
# ------------------------------------------------------------------------------------------------

def test_public_surface_matches_the_frozen_baseline():
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    actual = capture_public_surface.build_surface()
    assert actual == expected, (
        "the public surface moved. If that is intended, regenerate with "
        "`python scripts/capture_public_surface.py` and justify the diff in the PR3-PR8 journal -- "
        "the campaign's whole premise is that this surface survives."
    )


def test_the_capture_script_is_deterministic_and_writes_nothing_in_check_mode():
    before = BASELINE.read_bytes()
    result = subprocess.run([sys.executable, str(CAPTURE), "--check"],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert BASELINE.read_bytes() == before


def test_every_md_openmm_command_still_exists():
    """The legacy CLI is a compatibility contract for the whole campaign."""
    surface = json.loads(BASELINE.read_text(encoding="utf-8"))["cli"]
    for command in ["md-openmm", "md-openmm prepare", "md-openmm md", "md-openmm rest2",
                    "md-openmm smoke", "md-openmm validate-env", "md-openmm validate-system",
                    "md-openmm validate-bundle", "md-openmm bundle validate",
                    "md-openmm bundle inspect", "md-openmm bundle relocate-check",
                    "md-openmm config list-profiles", "md-openmm config init",
                    "md-openmm config validate", "md-openmm config resolve",
                    "md-openmm config diff", "md-openmm config explain",
                    "md-openmm config migrate"]:
        assert command in surface, command
    assert surface["md-openmm"]["prog"] == "md-openmm"


def test_platform_and_device_options_are_part_of_the_frozen_surface():
    """The GPU/platform contract is an interface promise, not only a runtime one."""
    surface = json.loads(BASELINE.read_text(encoding="utf-8"))["cli"]
    for command in ("md-openmm prepare", "md-openmm md", "md-openmm rest2"):
        dests = {a["dest"] for a in surface[command]["actions"]}
        assert "platform" in dests, command
        assert "device" in dests, command
    platform = next(a for a in surface["md-openmm prepare"]["actions"] if a["dest"] == "platform")
    assert platform["choices"] is not None
    for accelerator in ("CUDA", "OpenCL", "CPU"):
        assert accelerator in platform["choices"], platform["choices"]


# ------------------------------------------------------------------------------------------------
# the goldens, compared as bytes
# ------------------------------------------------------------------------------------------------

#: Recorded at the campaign base. `capture_goldens.py --check` recomputes and compares content;
#: this compares the committed FILES, so a regenerated-but-equivalent rewrite is still visible.
GOLDEN_SHA256 = {
    "bundle_contract.json": "514b14c308a0e58883243ba972a4855c60ec3233e18514b08afc1b91f3545a49",
    "configuration_hashes.json": "a42d15344e5871a487160d8ff06151bc11ad3eb3ed5369b49c250a890958ee6f",
    "format_equivalence.json": "eca1873de9e7e7d115417d7b33056ce6beb0a71620040b306c4498e82c677b7d",
    "profiles.json": "ba524a6dd49679c594333a3a5b7d53d734e442212b4d8ac93f546119a689abd5",
    "rest2_defaults.json": "a9420bd1b1a7a9226cb5a5b64a2e9d96802ec89c12a60a46ae0ebe9a31c613d2",
    "runstate_contract.json": "2ef132a8ca41664e8ab3bcab1a7181b3a7ad538479ad3ad3ce1c4ff89bcb304c",
    "seed_derivation.json": "95beae1da61dd0ae1268d5ab833d36fc745b87e7e6cb033a80c02ebf4e404cac",
}


@pytest.mark.parametrize("name", sorted(GOLDEN_SHA256))
def test_golden_files_are_byte_identical_to_the_campaign_base(name):
    digest = hashlib.sha256((GOLDEN_DIR / name).read_bytes()).hexdigest()
    assert digest == GOLDEN_SHA256[name], (
        f"{name} changed during the migration. The campaign's first invariant is that these seven "
        f"files do not move; investigate rather than regenerating."
    )


def test_there_are_exactly_seven_goldens():
    assert sorted(p.name for p in GOLDEN_DIR.glob("*.json")) == sorted(GOLDEN_SHA256)


# ------------------------------------------------------------------------------------------------
# the module map
# ------------------------------------------------------------------------------------------------

def load_map() -> dict:
    return json.loads(MODULE_MAP.read_text(encoding="utf-8"))


def test_every_existing_module_is_accounted_for_in_the_map():
    mapped = load_map()["modules"]
    existing = capture_public_surface._packaged_modules()
    missing = sorted(set(existing) - set(mapped))
    assert missing == [], (
        f"modules exist with no destination recorded: {missing}. Add them to "
        f"tests/baseline/module_map.json so a later move cannot hide coverage loss."
    )


def test_map_entries_are_well_formed():
    for section in ("modules", "tests"):
        for name, entry in load_map()[section].items():
            assert entry["status"] in ("planned", "moved", "unchanged"), (name, entry)
            assert isinstance(entry["destination"], str) and entry["destination"], name
            if entry["status"] == "planned":
                assert entry.get("phase") in (3, 4, 5, 6, 7, 8), (name, entry)


def test_destinations_claimed_as_moved_are_importable():
    """A map that claims a move must be checkable, or it is just a wish."""
    import importlib

    for name, entry in load_map()["modules"].items():
        if entry["status"] != "moved":
            continue
        importlib.import_module(entry["destination"])
        if entry.get("compat_shim"):
            importlib.import_module(name), f"{name} must remain importable as a compatibility shim"


def test_test_files_named_in_the_map_exist_at_source_or_destination():
    for name, entry in load_map()["tests"].items():
        source, destination = REPO_ROOT / name, REPO_ROOT / entry["destination"]
        assert source.is_file() or destination.is_file(), (
            f"{name} is neither where it was nor where the map says it went"
        )
