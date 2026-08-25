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


# -------------------------------------------------------------------------------------------
# The same fact is declared in five places. They must agree.
#
# The ligand charge method is stated in the profiles, in config.py DEFAULTS, in the packaged system
# manifests, in the worked-example configs and in the README. A defaults change updated the first
# two and missed the rest, and nothing failed until the full suite ran -- because the assertions
# that would have caught it were slow-marked. These tests close that gap by checking the places
# themselves rather than the tests that read them.
# -------------------------------------------------------------------------------------------
import yaml  # noqa: E402

MANIFESTS = REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"

#: Deliberate exceptions, each for a stated reason rather than because it was awkward.
#: - small_macrocycle_smoke: a tiny installability check, not a documented default. It keeps its own
#:   settings exactly as the cpu-smoke PROFILE does.
#: - ace_ala_nme: the peptide route. There is no ligand, so charge_method is null by definition.
_MANIFEST_EXCEPTIONS = {"small_macrocycle_smoke", "ace_ala_nme"}


def _ligand_charge_default() -> str:
    methods = {(d["build"]["forcefield"] or {}).get("charge_method")
               for d in _profiles().values()
               if (d["build"]["forcefield"] or {}).get("small_molecule")}
    assert len(methods) == 1, f"the ligand profiles disagree: {methods}"
    return methods.pop()


def test_packaged_system_manifests_use_the_ligand_charge_default():
    """These were missed when the default moved, and only the slow suite noticed."""
    expected = _ligand_charge_default()
    offenders = {}
    for path in sorted(MANIFESTS.glob("*.yaml")):
        if path.stem in _MANIFEST_EXCEPTIONS:
            continue
        doc = yaml.safe_load(path.read_text()) or {}
        stated = ((doc.get("parameterization") or doc.get("ligand_build") or {})
                  .get("charge_method"))
        if stated is not None and stated != expected:
            offenders[path.name] = stated
    assert not offenders, (
        f"packaged manifests disagree with the ligand default {expected!r}: {offenders}")


def test_worked_example_configs_use_the_ligand_charge_default():
    """A shipped example that names the old default is refused by the consistency check.

    That is not a cosmetic mismatch: `ligand_build.charge_model` must agree with the resolved
    `forcefield.ligand_charge_method`, so a stale example fails at generation with a message about
    chemistry that never ran.
    """
    expected = _ligand_charge_default()
    offenders = {}
    for path in sorted((REPO_ROOT / "test").rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for block, key in (("ligand_build", "charge_model"),
                           ("forcefield", "ligand_charge_method"),
                           ("parameterization", "charge_method")):
            stated = (doc.get(block) or {}).get(key)
            if stated is not None and stated != expected:
                offenders[f"{path.relative_to(REPO_ROOT)}:{block}.{key}"] = stated
    assert not offenders, (
        f"worked-example configs disagree with the ligand default {expected!r}: {offenders}")


def test_the_two_declared_charge_methods_are_the_supported_ones():
    """Whatever the default is, it must be a method build_forcefield actually implements."""
    from md_templates.openmm.system import NAGL_AM1BCC_METHODS

    supported = {"am1bcc", *NAGL_AM1BCC_METHODS}
    assert _ligand_charge_default() in supported


def test_no_place_declares_a_timestep_that_disagrees_with_the_profiles():
    """The timestep is stated in profiles and in worked-example configs."""
    expected = {d["protocol"]["integrator"]["timestep"] for d in _profiles().values()}
    assert expected == {"4 fs"}
    offenders = {}
    for path in sorted((REPO_ROOT / "test").rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        stated = ((doc.get("protocol") or {}).get("integrator") or {}).get("timestep")
        # a worked example may deliberately shrink durations, but not the integration timestep,
        # because that is a Hamiltonian-integration choice rather than a length
        if stated is not None and stated not in expected:
            offenders[str(path.relative_to(REPO_ROOT))] = stated
    assert not offenders, f"worked examples disagree with the profile timestep: {offenders}"
