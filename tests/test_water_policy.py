"""The default water model is derived from the force fields that will actually be assigned.

A protein force field and a small-molecule force field are each parameterised against water, and not
against the same water, so one default for "explicit solvent" mispairs one of them by construction.

These tests interrogate the FORCE-FIELD FAMILY rather than the input label. `peptide` and `ligand`
say which reader parsed the input; the force-field fields say which parameters get assigned, and it
is the parameters that were fitted. The distinction is not academic: a complex carries both, so a
label-driven rule has no answer for it.

Sources for the pairings are cited in `water_policy.py`: ff19SB/OPC from Tian et al. JCTC 2020
(doi:10.1021/acs.jctc.9b00591), ff14SB/TIP3P from Maier et al. JCTC 2015
(doi:10.1021/acs.jctc.5b00255), Sage/TIP3P from Boothroyd et al. JCTC 2023
(doi:10.1021/acs.jctc.3c00039).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")

FF19SB = "amber19/protein.ff19SB.xml"
SAGE = "openff-2.2.0"
OPC = ("amber19/opc.xml", "opc")
TIP3P = ("amber19/tip3p.xml", "tip3p")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


# ---------------------------------------------------------------------------------------------
# the four v2 defaults
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("profile_id,expected", [
    ("explicit-md-peptide-v2", OPC),
    ("explicit-rest2-peptide-v2", OPC),
    ("explicit-md-ligand-v2", TIP3P),
    ("explicit-rest2-ligand-v2", TIP3P),
])
def test_the_four_v2_profiles_carry_their_force_field_partner(profile_id, expected):
    from md_templates.openmm.spec.resolve import load_profile

    build = load_profile(profile_id)["defaults"]["build"]
    assert (build["forcefield"]["water"], build["solvation"]["water_model"]) == expected


@pytest.mark.parametrize("route,method,expected", [
    ("pdb", "md", OPC), ("pdb", "rest2", OPC),
    ("smiles", "md", TIP3P), ("smiles", "rest2", TIP3P),
])
def test_the_default_alias_resolves_to_that_pairing(route, method, expected):
    from md_templates.openmm.spec.resolve import select_profile

    profile = select_profile(route, method)
    assert profile["profile_id"].endswith("-v2")
    build = profile["defaults"]["build"]
    assert (build["forcefield"]["water"], build["solvation"]["water_model"]) == expected


# ---------------------------------------------------------------------------------------------
# derivation from the family, including the complex rule
# ---------------------------------------------------------------------------------------------

def test_ff19sb_alone_gives_opc():
    from md_templates.openmm.water_policy import resolve_default_water

    got = resolve_default_water({"protein": FF19SB}, solute_kind="peptide")
    assert (got["water"], got["water_model"]) == OPC
    assert got["basis"] == "ff19SB" and got["mixed"] is False


def test_sage_alone_gives_plain_tip3p_not_tip3p_fb():
    """TIP3P-FB is a different ForceBalance refit and is not what Sage was trained with."""
    from md_templates.openmm.water_policy import resolve_default_water

    got = resolve_default_water({"ligand": SAGE}, solute_kind="ligand")
    assert (got["water"], got["water_model"]) == TIP3P
    assert "tip3pfb" not in got["water"]


def test_the_tleap_spelling_of_ff19sb_resolves_the_same_way():
    """`leaprc.protein.ff19SB` and `amber19/protein.ff19SB.xml` are one choice in two dialects.

    The implicit route builds through tleap and records the leaprc spelling. Matching on the family
    rather than on an exact resource string is what keeps them from disagreeing.
    """
    from md_templates.openmm.water_policy import resolve_default_water

    assert resolve_default_water({"protein": "leaprc.protein.ff19SB"},
                                 solute_kind="peptide")["basis"] == "ff19SB"


def test_ff14sb_gives_tip3p_because_that_is_what_it_was_developed_in():
    from md_templates.openmm.water_policy import resolve_default_water

    got = resolve_default_water({"protein": "amber14/protein.ff14SB.xml"}, solute_kind="peptide")
    assert (got["water"], got["water_model"]) == TIP3P
    assert got["basis"] == "ff14SB"


def test_a_complex_defaults_to_opc_and_says_that_it_is_a_compatibility_choice():
    """One box carries one water model, so one force field must run off its fitting partner.

    The rule resolves to the protein's partner. What matters as much as the value is that the
    rationale says so: a bundle that silently recorded `opc` would let a later report imply both
    force fields were used as published.
    """
    from md_templates.openmm.water_policy import resolve_default_water

    got = resolve_default_water({"protein": FF19SB, "ligand": SAGE}, solute_kind="complex")
    assert (got["water"], got["water_model"]) == OPC
    assert got["mixed"] is True
    assert "ff19SB" in got["basis"] and "Sage" in got["basis"]
    lowered = got["rationale"].lower()
    assert "compatibility" in lowered and "not a validated pairing" in lowered


def test_the_solute_kind_filters_fields_but_never_decides_the_answer():
    """A peptide bundle may carry an unused package-default ligand entry; it must not vote."""
    from md_templates.openmm.water_policy import resolve_default_water

    both = {"protein": FF19SB, "ligand": SAGE}
    assert resolve_default_water(both, solute_kind="peptide")["basis"] == "ff19SB"
    assert resolve_default_water(both, solute_kind="ligand")["basis"].startswith("OpenFF Sage")


# ---------------------------------------------------------------------------------------------
# ambiguity is refused, not guessed
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("forcefield,kind", [
    ({"protein": "my_lab_ff.xml"}, "peptide"),
    ({"ligand": "gaff-2.11"}, "ligand"),
    ({"protein": "my_lab_ff.xml", "ligand": "gaff-2.11"}, "complex"),
    ({}, "peptide"),
])
def test_an_unrecognised_force_field_is_refused_with_a_usable_message(forcefield, kind):
    from md_templates.openmm.water_policy import AmbiguousWaterPolicy, resolve_default_water

    with pytest.raises(AmbiguousWaterPolicy) as raised:
        resolve_default_water(forcefield, solute_kind=kind)
    message = str(raised.value)
    assert "forcefield.water" in message and "solvation.water_model" in message, (
        "the refusal must name the fields that resolve it")


def test_a_half_recognised_complex_still_resolves_from_the_recognised_half():
    """ff19SB with an unrecognised small molecule is not ambiguous: the protein rule still applies.

    Refusing here would block a legitimate build over a small-molecule force field this package does
    not enumerate, while the protein force field -- the dominant error term, and the one the rule
    resolves in favour of anyway -- is known.
    """
    from md_templates.openmm.water_policy import resolve_default_water

    got = resolve_default_water({"protein": FF19SB, "ligand": "gaff-2.11"}, solute_kind="complex")
    assert (got["water"], got["water_model"]) == OPC
    assert got["basis"] == "ff19SB"


# ---------------------------------------------------------------------------------------------
# v1 compatibility, implicit, and overrides
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("profile_id", [
    "explicit-md-peptide-v1", "explicit-rest2-peptide-v1",
    "explicit-md-ligand-v1", "explicit-rest2-ligand-v1",
])
def test_v1_profiles_stay_resolvable_nondefault_and_keep_their_historical_water(profile_id):
    from md_templates.openmm.spec.resolve import load_profile

    profile = load_profile(profile_id)
    assert profile["is_default"] is False
    assert profile["superseded_by"] == profile_id.replace("-v1", "-v2")
    assert profile["defaults"]["build"]["forcefield"]["water"] == "amber19/tip3pfb.xml", (
        "a v1 profile must keep recording the water its runs actually used")


def test_implicit_profiles_carry_no_explicit_water_settings():
    from md_templates.openmm.spec.resolve import list_profiles

    for profile in list_profiles():
        if not profile["profile_id"].startswith("implicit-"):
            continue
        build = profile["defaults"]["build"]
        assert build["forcefield"].get("water") is None, profile["profile_id"]
        assert "solvation" not in build, profile["profile_id"]


def test_every_shipped_profile_pairs_water_parameters_with_matching_geometry():
    from md_templates.openmm.solvation import reconcile_water_model
    from md_templates.openmm.spec.resolve import list_profiles

    for profile in list_profiles():
        build = profile["defaults"]["build"]
        water = build["forcefield"].get("water")
        model = (build.get("solvation") or {}).get("water_model")
        if water and model:
            _, note = reconcile_water_model(water, model)
            assert note is None, f"{profile['profile_id']}: {note}"


# ---------------------------------------------------------------------------------------------
# end to end: provenance records the choice, its basis, and the ion source
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_generation_records_the_water_choice_its_basis_and_the_ion_source(tmp_path):
    config = tmp_path / "system.json"
    config.write_text(json.dumps({"system": {"id": "ace_ala_nme", "type": "protein"}}))
    out = tmp_path / "bundle"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    manifest = json.loads((out / "system_manifest.json").read_text())
    resolved = manifest["resolved_system_config"]
    assert resolved["forcefield"]["water"] == "amber19/opc.xml"
    assert resolved["solvation"]["water_model"] == "opc"

    policy = manifest["water_policy"]
    assert policy["basis"] == "ff19SB"
    assert policy["mixed"] is False
    assert policy["ion_parameters"]["source"] == "amber19/opc.xml"

    source = manifest["value_sources"]["forcefield.water"]
    assert source.startswith("package default"), source
    assert "ff19SB" in source, "provenance must say WHICH rule chose it, not merely that one did"


@pytest.mark.slow
def test_an_explicit_user_override_wins_and_is_recorded_as_user_input(tmp_path):
    """The escape hatch the ambiguity error recommends has to actually work."""
    config = tmp_path / "system.json"
    config.write_text(json.dumps({
        "system": {"id": "ace_ala_nme", "type": "protein"},
        "forcefield": {"protein": FF19SB, "water": "amber19/tip3p.xml",
                       "ligand": None, "ligand_charge_method": None},
        "solvation": {"water_model": "tip3p"},
    }))
    out = tmp_path / "bundle"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    manifest = json.loads((out / "system_manifest.json").read_text())
    resolved = manifest["resolved_system_config"]
    assert resolved["forcefield"]["water"] == "amber19/tip3p.xml", (
        "the package default overrode an explicit user choice")
    assert resolved["solvation"]["water_model"] == "tip3p"
    assert manifest["value_sources"]["forcefield.water"] == "user input"


@pytest.mark.slow
def test_the_ion_parameters_really_are_water_model_specific():
    """The provenance claim is only worth recording if the parameters actually differ.

    OpenMM ships Joung-Cheatham ions inside each water force-field file, so they follow the water
    model with no separate choice. Asserted through `createSystem` rather than by reading the XML,
    because what matters is the parameters the System ends up with.
    """
    from openmm import NonbondedForce, app, unit

    def sodium_epsilon(water_xml):
        forcefield = app.ForceField("amber19/protein.ff19SB.xml", water_xml)
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("NA", chain)
        topology.addAtom("NA", app.element.sodium, residue)
        system = forcefield.createSystem(topology)
        nonbonded = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        return nonbonded.getParticleParameters(0)[2].value_in_unit(unit.kilojoule_per_mole)

    opc, tip3p = sodium_epsilon("amber19/opc.xml"), sodium_epsilon("amber19/tip3p.xml")
    assert abs(opc - tip3p) > 0.1, (
        f"Na+ epsilon is {opc} with OPC and {tip3p} with TIP3P; if these were equal, recording the "
        "ion source would be documenting a distinction without a difference")
