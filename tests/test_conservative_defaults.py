"""Conservative scientific defaults, and the opt-ins that are deliberately NOT defaults.

The release review found that HMR, a 4 fs timestep and NAGL charges had become the defaults for
every simulation. Each is defensible as a choice and none is defensible as a silent one: they
change the integration and the Hamiltonian for work that was never asked whether it wanted them.

Defaults are therefore the conservative settings, and the performance options are reachable only by
naming them. These tests pin BOTH halves -- a default that quietly becomes fast again, and an
opt-in that stops working, are both regressions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from md_templates.openmm.spec.resolve import PROFILE_DIR, list_profiles

#: The conservative settings the review specified, as they appear in a resolved profile.
CONSERVATIVE_TIMESTEP = "2 fs"
PERFORMANCE_TIMESTEP = "4 fs"
PERFORMANCE_HYDROGEN_MASS = "3.024 amu"


def _profiles() -> dict[str, dict]:
    return {p["profile_id"]: p for p in list_profiles()}


def _shipped_defaults() -> dict[str, dict]:
    """Every profile that is NOT an opt-in performance variant."""
    return {k: v for k, v in _profiles().items() if not k.endswith("-hmr-v1")}


# -------------------------------------------------------------------------------------------
# Conservative defaults
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("profile_id", sorted(_shipped_defaults()))
def test_default_profiles_integrate_at_two_femtoseconds(profile_id):
    profile = _profiles()[profile_id]
    assert profile["defaults"]["protocol"]["integrator"]["timestep"] == CONSERVATIVE_TIMESTEP


@pytest.mark.parametrize("profile_id", sorted(_shipped_defaults()))
def test_default_profiles_do_not_repartition_hydrogen_mass(profile_id):
    """2 fs with repartitioned hydrogens would be conservative in one place and not the other."""
    build = _profiles()[profile_id]["defaults"]["build"]
    assert build.get("hydrogen_mass") is None, build.get("hydrogen_mass")
    assert build.get("hmr_scope") in (None, "none"), build.get("hmr_scope")


def _ligand_profiles() -> list[str]:
    """Only profiles that actually parameterise a small molecule; the rest have nothing to charge."""
    return sorted(k for k, v in _profiles().items()
                  if (v["defaults"]["build"].get("forcefield") or {}).get("small_molecule"))


@pytest.mark.parametrize("profile_id", [p for p in _ligand_profiles()
                                        if not p.endswith("-hmr-v1")])
def test_default_ligand_charges_are_standard_am1bcc(profile_id):
    """NAGL is a trained approximation of AM1-BCC, not AM1-BCC; it cannot be the default."""
    forcefield = _profiles()[profile_id]["defaults"]["build"]["forcefield"]
    assert forcefield["charge_method"] == "am1bcc"


def test_no_shipped_profile_requests_nagl():
    """The blanket check: NAGL must be reachable only by explicit selection."""
    offenders = {k: (v["defaults"]["build"].get("forcefield") or {}).get("charge_method")
                 for k, v in _profiles().items()
                 if (v["defaults"]["build"].get("forcefield") or {}).get("charge_method")
                 not in (None, "am1bcc")}
    assert not offenders, offenders


def test_the_module_level_defaults_agree_with_the_profiles():
    """config.DEFAULTS is a second declaration of the same facts and drifted from them before."""
    from md_templates.openmm.config import DEFAULTS

    assert DEFAULTS["integrator"]["timestep_fs"] == 2.0
    assert DEFAULTS["system_build"]["hydrogen_mass_amu"] is None
    assert DEFAULTS["system_build"]["hmr_scope"] == "none"
    assert DEFAULTS["forcefield"]["ligand_charge_method"] == "am1bcc"


def test_the_other_conservative_settings_are_unchanged():
    """These were already correct; pinned so the revert did not disturb them."""
    from md_templates.openmm.config import DEFAULTS

    assert DEFAULTS["forcefield"]["protein"] == "amber19/protein.ff19SB.xml"
    assert DEFAULTS["forcefield"]["water"] == "amber19/opc.xml"
    assert DEFAULTS["forcefield"]["ligand"] == "openff-2.2.0"          # Sage 2.2
    assert DEFAULTS["solvation"]["water_model"] == "opc"
    assert DEFAULTS["solvation"]["box_shape"] == "dodecahedron"
    assert DEFAULTS["solvation"]["padding_nm"] == 2.0
    assert DEFAULTS["solvation"]["ionic_strength_molar"] == 0.15
    assert DEFAULTS["system_build"]["nonbonded_cutoff_nm"] == 1.0


# -------------------------------------------------------------------------------------------
# The performance opt-in
# -------------------------------------------------------------------------------------------
def test_a_performance_profile_exists_for_every_selectable_default():
    """An opt-in nobody can reach for their route is not an opt-in."""
    shipped = {k for k, v in _profiles().items()
               if v.get("is_default") or k.startswith("implicit-")}
    shipped = {k for k in shipped if not k.endswith("-hmr-v1")}
    for profile_id in shipped:
        base = profile_id.rsplit("-v", 1)[0]
        assert f"{base}-hmr-v1" in _profiles(), f"no performance variant for {profile_id}"


@pytest.mark.parametrize("profile_id",
                         sorted(k for k in _profiles() if k.endswith("-hmr-v1")))
def test_performance_profiles_use_four_femtoseconds_with_repartitioned_hydrogens(profile_id):
    profile = _profiles()[profile_id]
    assert profile["defaults"]["protocol"]["integrator"]["timestep"] == PERFORMANCE_TIMESTEP
    build = profile["defaults"]["build"]
    assert build["hydrogen_mass"] == PERFORMANCE_HYDROGEN_MASS
    assert build["hmr_scope"] == "solute"


@pytest.mark.parametrize("profile_id", [p for p in _ligand_profiles()
                                        if p.endswith("-hmr-v1")])
def test_a_performance_profile_never_also_changes_the_charge_model(profile_id):
    """HMR and NAGL are independent opt-ins. Bundling them would smuggle a Hamiltonian change
    into a request for a faster timestep."""
    assert _profiles()[profile_id]["defaults"]["build"]["forcefield"]["charge_method"] == "am1bcc"


@pytest.mark.parametrize("profile_id",
                         sorted(k for k in _profiles() if k.endswith("-hmr-v1")))
def test_no_performance_profile_is_selected_automatically(profile_id):
    assert _profiles()[profile_id]["is_default"] is False


def test_4_fs_is_never_paired_with_unrepartitioned_hydrogens_anywhere():
    """The combination that is actually unstable, checked across every profile at once."""
    offenders = []
    for profile_id, profile in _profiles().items():
        timestep = profile["defaults"]["protocol"]["integrator"]["timestep"]
        mass = profile["defaults"]["build"].get("hydrogen_mass")
        if timestep == PERFORMANCE_TIMESTEP and not mass:
            offenders.append(profile_id)
    assert not offenders, offenders


# -------------------------------------------------------------------------------------------
# Explicit and implicit remain distinct
# -------------------------------------------------------------------------------------------
def test_explicit_and_implicit_defaults_stay_distinct():
    """An implicit document resolved against explicit defaults drags in PME, a cutoff and a
    barostatted stage, and is then refused one field at a time."""
    explicit = _profiles()["explicit-md-peptide-v2"]["defaults"]["build"]
    implicit = _profiles()["implicit-md-peptide-v1"]["defaults"]["build"]
    assert explicit.get("solvation") is not None
    assert implicit.get("implicit") is not None
    assert implicit.get("solvation") is None, "an implicit profile must not carry a water box"
    assert explicit.get("rigid_water") is True
    assert implicit.get("rigid_water") in (None, False)


def test_implicit_profiles_specify_gbn2_with_mbondi3():
    for profile_id, profile in _profiles().items():
        if not profile_id.startswith("implicit-"):
            continue
        implicit = profile["defaults"]["build"]["implicit"]
        assert implicit["model"] == "GBn2"
        assert implicit["radii"] == "mbondi3"
