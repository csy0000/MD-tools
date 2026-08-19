"""The captured compatibility goldens must still describe current behaviour.

These are the tests that make the migration safe: they were captured from the baseline before any
structural edit, and they consume the committed files rather than recomputing an expectation. A test
that regenerated its own fixture would pass whatever the code did.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "goldens"
CAPTURE = REPO_ROOT / "scripts" / "capture_goldens.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import capture_goldens  # noqa: E402


def golden(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------
# 32. every committed golden matches current behaviour
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(capture_goldens.GENERATORS))
def test_committed_golden_matches_current_behaviour(name):
    expected = golden(name)
    actual = capture_goldens.GENERATORS[name]()
    assert actual == expected, (
        f"{name} no longer describes current behaviour. If the change is intended and scientifically "
        f"justified, regenerate deliberately with `python scripts/capture_goldens.py` and explain the "
        f"diff in the journal."
    )


def test_golden_check_mode_is_clean_and_writes_nothing():
    """The documented check command exits 0 and leaves the fixtures byte-identical."""
    before = {p.name: p.read_bytes() for p in GOLDEN_DIR.glob("*.json")}
    result = subprocess.run([sys.executable, str(CAPTURE), "--check"],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    after = {p.name: p.read_bytes() for p in GOLDEN_DIR.glob("*.json")}
    assert after == before


def test_ordinary_test_run_does_not_regenerate_fixtures():
    """Importing and calling the generators must not touch the committed files."""
    before = {p.name: p.read_bytes() for p in GOLDEN_DIR.glob("*.json")}
    for generator in capture_goldens.GENERATORS.values():
        generator()
    after = {p.name: p.read_bytes() for p in GOLDEN_DIR.glob("*.json")}
    assert after == before


# ------------------------------------------------------------------------------------------------
# 33. canonical hashes remain exactly unchanged
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(capture_goldens.CONFIGS))
def test_canonical_hashes_unchanged(case):
    from md_templates.openmm.spec import canonical, resolve

    expected = golden("configuration_hashes.json")[case]
    result = resolve.resolve_spec(json.loads(json.dumps(capture_goldens.CONFIGS[case])))
    assert result["hashes"] == expected["hashes"]
    assert canonical.sha256_of(canonical.dump_model(result["spec"])) == \
        expected["canonical_json_sha256"]


def test_yaml_and_json_still_canonicalise_identically():
    expected = golden("format_equivalence.json")
    assert capture_goldens._format_equivalence_golden() == expected
    assert expected["identical_canonical_bytes"] is True
    assert expected["identical_hashes"] is True


def test_seed_derivation_unchanged():
    """`master + index in STAGE_ORDER`, explicit overrides preserved."""
    expected = golden("seed_derivation.json")
    assert expected["stage_order"] == ["structure", "equilibration", "md", "rest2"]
    assert expected["cases"]["default"]["resolved"] == {
        "structure": 20260814, "equilibration": 20260815, "md": 20260816, "rest2": 20260817,
    }
    assert capture_goldens._seed_goldens() == expected


# ------------------------------------------------------------------------------------------------
# 34. profile selection and profile hashes remain unchanged
# ------------------------------------------------------------------------------------------------

def test_profile_hashes_and_default_selection_unchanged():
    from md_templates.openmm.spec import resolve

    expected = golden("profiles.json")
    assert capture_goldens._profile_goldens() == expected
    for route in ("smiles", "pdb"):
        for method in ("md", "rest2"):
            assert resolve.select_profile(route, method)["profile_id"] == \
                expected["default_selection"][f"{route}/{method}"]


def test_rest2_defaults_unchanged():  # noqa: D401
    """Ladder, omega exclusion and proline classification are scientific settings."""
    from md_templates.openmm.tau import build_tau_ladder, scale_factors_for_ladder

    expected = golden("rest2_defaults.json")
    assert capture_goldens._rest2_defaults_golden() == expected
    assert expected["runtime_defaults"]["omega_exclusion"] is True
    for ladder in expected["profile_ladders"].values():
        # The ladder is now DECLARED as tau and derived to s. The physics it resolves to is the
        # same ladder as before: tau 0 -> 0.5 gives s 1.0 -> 0.25.
        assert ladder["minimum"] == 0.0 and ladder["maximum"] == 0.5
        assert ladder["interpolation"] == "linear"
        scale_factors = scale_factors_for_ladder(
            build_tau_ladder(ladder["minimum"], ladder["maximum"], ladder["count"]))
        assert scale_factors[0] == 1.0 and scale_factors[-1] == 0.25
        assert scale_factors == sorted(scale_factors, reverse=True)


# ------------------------------------------------------------------------------------------------
# persistent schemas: PR 1 must not move them
# ------------------------------------------------------------------------------------------------

def test_bundle_contract_unchanged():
    from md_templates.openmm import bundlev2

    expected = golden("bundle_contract.json")
    assert bundlev2.BUNDLE_SCHEMA_VERSION == expected["bundle_schema_version"] == 2
    assert dict(sorted(bundlev2.REQUIRED_ROLES.items())) == expected["required_roles"]


def test_runstate_contract_unchanged():
    from md_templates.openmm import runstate

    expected = golden("runstate_contract.json")
    assert runstate.RUN_STATE_SCHEMA == expected["run_state_schema"]
    assert list(runstate.CONTINUITY_PATHS) == expected["continuity_paths"]
    assert list(runstate.EXTENSION_PATHS) == expected["extension_paths"]


def test_no_template_identity_leaked_into_persistent_contracts():
    """PR 1 adds catalog metadata; none of it may reach a scientific or continuity projection."""
    from md_templates.openmm import bundlev2, runstate
    from md_templates.openmm.spec import canonical

    surfaces = (
        list(runstate.CONTINUITY_PATHS)
        + list(runstate.EXTENSION_PATHS)
        + list(bundlev2.REQUIRED_ROLES)
        + list(canonical.EXECUTION_ONLY)
        + list(canonical.EXTENSION_ONLY)
    )
    for banned in ("template_id", "template_path", "template_identity", "registry"):
        assert not any(banned in entry for entry in surfaces), banned

    for case in capture_goldens.CONFIGS:
        blob = json.dumps(golden("configuration_hashes.json")[case])
        assert "template" not in blob
