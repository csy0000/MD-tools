"""The README's stated defaults must be the defaults.

Documentation drifts silently, and a user reading it configures the wrong thing without any error to
tell them. That happened here: a defaults change moved every profile to HMR 3.024 amu and 4 fs, and
the README went on saying "implicit profiles do not repartition hydrogen mass and use a 2 fs
timestep" -- and separately, that preparing cyclo-RGDfV "costs ~27 minutes of AM1-BCC charge
derivation", after the default became NAGL and that step became about a second.

Both statements were false the moment the commit landed, and nothing failed. An external project
reading the README would have configured against them.

These tests read the SHIPPED PROFILES and check the README agrees. They are deliberately narrow:
they pin the handful of numbers a consumer acts on, not prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES = REPO_ROOT / "src" / "md_templates" / "openmm" / "spec" / "profiles"
README = REPO_ROOT / "README.md"


def _profiles():
    out = {}
    for path in sorted(PROFILES.glob("*.json")):
        if "smoke" in path.stem:
            continue                       # deliberately tiny, not a documented default
        out[path.stem] = json.loads(path.read_text())["defaults"]
    return out


def test_every_profile_agrees_on_the_timestep_the_readme_states():
    """One number, stated once in the README, true of every shipped profile."""
    timesteps = {name: d["protocol"]["integrator"]["timestep"] for name, d in _profiles().items()}
    assert set(timesteps.values()) == {"4 fs"}, timesteps
    assert "**4 fs**" in README.read_text(), "the README must state the timestep it ships"


def test_every_profile_agrees_on_hydrogen_mass_repartitioning():
    masses = {name: d["build"].get("hydrogen_mass") for name, d in _profiles().items()}
    assert set(masses.values()) == {"3.024 amu"}, masses
    scopes = {name: d["build"].get("hmr_scope") for name, d in _profiles().items()}
    assert set(scopes.values()) == {"solute"}, scopes
    assert "3.024 amu" in README.read_text()


def test_the_readme_does_not_still_claim_implicit_skips_hmr():
    """The exact sentence that was false for one commit."""
    text = README.read_text().lower()
    assert "do not repartition hydrogen mass" not in text
    assert "use a 2 fs timestep" not in text


def test_every_ligand_profile_agrees_on_the_charge_method():
    methods = {name: (d["build"]["forcefield"] or {}).get("charge_method")
               for name, d in _profiles().items()
               if (d["build"]["forcefield"] or {}).get("small_molecule")}
    assert methods, "there must be ligand profiles to check"
    assert set(methods.values()) == {"am1bcc_nagl"}, methods
    assert "am1bcc_nagl" in README.read_text(), "the README must name the ligand charge default"


def test_the_readme_does_not_still_quote_the_am1bcc_preparation_cost():
    """It said ~27 minutes of AM1-BCC; the default is now NAGL and takes about a second."""
    assert "27 minutes of AM1-BCC" not in README.read_text()


def test_the_runtime_default_matches_the_ligand_profiles():
    """config.py DEFAULTS and the profiles are separate places; they must not disagree."""
    from md_templates.openmm.config import DEFAULTS

    profile_methods = {(d["build"]["forcefield"] or {}).get("charge_method")
                       for d in _profiles().values()
                       if (d["build"]["forcefield"] or {}).get("small_molecule")}
    assert DEFAULTS["forcefield"]["ligand_charge_method"] in profile_methods


def test_the_readme_scope_table_names_both_solvents_and_both_methods():
    """The first thing a consumer reads must describe what is actually supported."""
    text = README.read_text()
    for claim in ("explicit water", "implicit solvent", "REST2", "conventional MD"):
        assert claim.lower() in text.lower(), claim
    assert "nonperiodic constant temperature" in text.lower(), (
        "the implicit production ensemble must be named, since it is neither NVT nor NPT")
