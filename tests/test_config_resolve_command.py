"""`md-openmm config resolve` -- the command a user runs to see and override defaults.

It had NO test, and it crashed on every configuration that resolved successfully: it read
`spec.protocol.production.total`, an attribute that exists on neither MDProduction nor
REST2Production. These tests exercise the command through its real entry point rather than the
resolver underneath it, because the resolver was never the broken part.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REST2_CONFIG = REPO_ROOT / "test" / "ala" / "REST2" / "alanine_rest2.json"


def _resolve(*args):
    """Run the command exactly as a user would, and parse what it prints."""
    proc = subprocess.run(
        [sys.executable, "-m", "md_templates.openmm.cli", "config", "resolve", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, f"config resolve failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_config_resolve_succeeds_on_a_rest2_configuration():
    """The regression: this raised AttributeError instead of printing anything."""
    out = _resolve(str(REST2_CONFIG))
    assert out["profile"]["profile_id"] == "explicit-rest2-peptide-v2"


def test_derived_reports_a_segment_and_never_invents_a_total():
    """How many segments run is a launcher decision, so a total here would be fabricated."""
    derived = _resolve(str(REST2_CONFIG))["derived"]
    assert "protocol.production.duration_per_segment_ps" in derived
    assert not any("total" in key for key in derived), (
        f"a total cannot be derived from the configuration alone: {derived}")


def test_derived_step_counts_are_exact_and_consistent():
    derived = _resolve(str(REST2_CONFIG))["derived"]
    steps = derived["steps_per_segment"]
    assert steps == pytest.approx(
        derived["protocol.production.duration_per_segment_ps"]
        / derived["protocol.integrator.timestep_ps"])
    assert steps % derived["number_of_exchanges_per_segment"] == 0, (
        "the model refuses a segment that is not a whole number of exchanges, so the derived "
        "report must not present a rounded one")
    assert derived["steps_per_exchange"] * derived["number_of_exchanges_per_segment"] == steps


def test_an_override_changes_the_value_and_is_attributed_to_the_cli():
    """Overriding is the documented way to alter a default; provenance must say where it came from."""
    out = _resolve(str(REST2_CONFIG), "--set", "protocol.integrator.timestep=2 fs")
    # dump_model emits the canonical form ({value, unit}); `as_written` is provenance and is
    # deliberately not part of the canonical configuration.
    assert out["configuration"]["protocol"]["integrator"]["timestep"] == {"value": 0.002,
                                                                         "unit": "ps"}
    assert out["sources"]["protocol.integrator.timestep"] == "cli"
    assert out["derived"]["protocol.integrator.timestep_ps"] == pytest.approx(0.002)


def test_unoverridden_fields_are_attributed_to_the_profile():
    """The complement: what the user did NOT set must be traceable to the profile that supplied it."""
    out = _resolve(str(REST2_CONFIG))
    # a field this document does NOT restate -- it restates hydrogen_mass, so that one is
    # attributed to the document and would not test the profile layer at all
    assert out["sources"]["build.nonbonded.method"].startswith("profile:")
    assert out["sources"]["build.forcefield.charge_method"].startswith("profile:")


def test_an_override_that_breaks_an_invariant_is_refused_not_rounded():
    """A 3 fs timestep does not divide a 5 ns segment; rounding would run a different length."""
    proc = subprocess.run(
        [sys.executable, "-m", "md_templates.openmm.cli", "config", "resolve",
         str(REST2_CONFIG), "--set", "protocol.integrator.timestep=3 fs"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "whole number" in (proc.stdout + proc.stderr)
