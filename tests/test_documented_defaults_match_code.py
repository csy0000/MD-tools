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


def _profiles(*, performance: bool = False):
    """Shipped profiles. `performance=True` selects the opt-in `-hmr-` variants instead.

    The two tiers are checked separately because they legitimately state DIFFERENT numbers: the
    defaults are conservative, the opt-ins are fast. Lumping them together would make the guard
    assert that everything agrees, which is exactly the drift-hiding it exists to prevent.
    """
    out = {}
    for path in sorted(PROFILES.glob("*.json")):
        if "smoke" in path.stem:
            continue                       # deliberately tiny, not a documented default
        if path.stem.endswith("-hmr-v1") is not performance:
            continue
        out[path.stem] = json.loads(path.read_text())["defaults"]
    return out


def test_every_default_profile_agrees_on_the_timestep_the_readme_states():
    """One number, stated once in the README, true of every shipped DEFAULT profile."""
    timesteps = {name: d["protocol"]["integrator"]["timestep"] for name, d in _profiles().items()}
    assert set(timesteps.values()) == {"2 fs"}, timesteps
    assert "**2 fs**" in README.read_text(), "the README must state the timestep it ships"


def test_every_performance_profile_agrees_on_the_timestep_the_readme_states():
    timesteps = {name: d["protocol"]["integrator"]["timestep"]
                 for name, d in _profiles(performance=True).items()}
    assert set(timesteps.values()) == {"4 fs"}, timesteps
    assert "**4 fs**" in README.read_text(), "the README must state the opt-in timestep too"


def test_default_profiles_state_that_they_do_not_repartition_hydrogen_mass():
    masses = {name: d["build"].get("hydrogen_mass") for name, d in _profiles().items()}
    assert set(masses.values()) == {None}, masses
    scopes = {name: d["build"].get("hmr_scope") for name, d in _profiles().items()}
    assert set(scopes.values()) == {"none"}, scopes


def test_performance_profiles_agree_on_hydrogen_mass_repartitioning():
    perf = _profiles(performance=True)
    masses = {name: d["build"].get("hydrogen_mass") for name, d in perf.items()}
    assert set(masses.values()) == {"3.024 amu"}, masses
    scopes = {name: d["build"].get("hmr_scope") for name, d in perf.items()}
    assert set(scopes.values()) == {"solute"}, scopes
    assert "3.024 amu" in README.read_text()


def test_the_readme_separates_the_conservative_default_from_the_opt_in():
    """Both tiers must be stated, or a reader cannot tell which one they are getting."""
    text = README.read_text()
    assert "**2 fs**" in text and "**4 fs**" in text
    lowered = text.lower()
    assert "hmr" in lowered or "hydrogen mass repartition" in lowered
    # the opt-in must be named as such, not presented as what everyone gets
    assert "opt-in" in lowered or "opt in" in lowered


def test_every_ligand_profile_agrees_on_the_charge_method():
    methods = {name: (d["build"]["forcefield"] or {}).get("charge_method")
               for name, d in _profiles().items()
               if (d["build"]["forcefield"] or {}).get("small_molecule")}
    assert methods, "there must be ligand profiles to check"
    assert set(methods.values()) == {"am1bcc"}, methods
    readme = README.read_text()
    assert "am1bcc" in readme, "the README must name the ligand charge default"
    # NAGL must be documented as an opt-in, and never described as the same calculation
    assert "am1bcc_nagl" in readme, "the README must document the NAGL opt-in"
    for forbidden in ("identical to AM1-BCC", "same as AM1-BCC", "equivalent to AM1-BCC"):
        assert forbidden.lower() not in readme.lower(), (
            f"the README claims NAGL is {forbidden!r}; it is a trained approximation, not that "
            "calculation")


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
_MANIFEST_EXCEPTIONS = {
    # a tiny installability check, not a documented default; it keeps the cpu-smoke settings
    "small_macrocycle_smoke",
    # the peptide route: no ligand, so charge_method is null by definition
    "ace_ala_nme",
    # DELIBERATELY PINNED to NAGL. This manifest reproduces cyclo-(RGDfV) trajectories that were
    # generated with NAGL charges; repointing it at the am1bcc default would change the
    # Hamiltonian under an unchanged name. The manifest states the pin and the reason, exactly as
    # it already does for its TIP3P-FB water.
    "cyclo_rgdfv",
}


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
    assert expected == {"2 fs"}
    offenders = {}
    for path in sorted((REPO_ROOT / "test").rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        stated = ((doc.get("protocol") or {}).get("integrator") or {}).get("timestep")
        declares_hmr = bool((doc.get("build") or {}).get("hydrogen_mass"))
        # A worked example may deliberately shrink durations, but not the integration timestep --
        # that is a Hamiltonian-integration choice rather than a length. The ONE exception is an
        # example that also declares HMR: those two examples exist to demonstrate the opt-in, and
        # 4 fs is correct there precisely BECAUSE the hydrogen mass is declared with it. An example
        # at 4 fs WITHOUT HMR is the unstable combination and is still refused.
        allowed = {*expected, "4 fs"} if declares_hmr else expected
        if stated is not None and stated not in allowed:
            offenders[str(path.relative_to(REPO_ROOT))] = stated
    assert not offenders, f"worked examples disagree with the profile timestep: {offenders}"
