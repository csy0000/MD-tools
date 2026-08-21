"""OPC is the active explicit-solvent default, and the change is versioned rather than silent.

Finding 7. ff19SB was parameterised against OPC, so TIP3P-FB was the wrong active default. Changing
a named profile's water model in place would have been worse than the original defect: every
existing configuration naming `explicit-md-peptide-v1` would silently start running different
physics under the same name.

So the change is versioned. `-v2` profiles carry OPC and are the defaults; `-v1` profiles keep
TIP3P-FB, stay resolvable by name, and are marked non-default with an explicit `superseded_by`.
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

EXPLICIT_ROUTES = [("pdb", "md"), ("pdb", "rest2"), ("smiles", "md"), ("smiles", "rest2")]
V2_PROFILES = ["explicit-md-peptide-v2", "explicit-md-ligand-v2",
               "explicit-rest2-peptide-v2", "explicit-rest2-ligand-v2"]

#: The water each route defaults to, and why they differ. ff19SB's CMAPs were trained against QM
#: surfaces in solution and validated with OPC; Sage's LJ refit was conditioned on plain TIP3P.
#: Pairing either with the other's water is a documented mismatch, so the default follows the solute.
ROUTE_WATER = {"pdb": ("amber19/opc.xml", "opc"), "smiles": ("amber19/tip3p.xml", "tip3p")}
PROFILE_WATER = {"peptide": ("amber19/opc.xml", "opc"), "ligand": ("amber19/tip3p.xml", "tip3p")}


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


@pytest.mark.parametrize("route,method", EXPLICIT_ROUTES)
def test_the_default_alias_resolves_to_the_v2_profile_for_that_route(route, method):
    from md_templates.openmm.spec.resolve import select_profile

    profile = select_profile(route, method)
    assert profile["profile_id"].endswith("-v2"), profile["profile_id"]
    build = profile["defaults"]["build"]
    water, model = ROUTE_WATER[route]
    assert build["forcefield"]["water"] == water
    assert build["solvation"]["water_model"] == model


@pytest.mark.parametrize("profile_id", V2_PROFILES)
def test_every_v2_profile_names_its_route_water_in_both_places(profile_id):
    """The force field and the packing model are separate fields and both must agree.

    They mean different things -- one supplies parameters, the other picks the geometry template
    Modeller packs with -- so a profile can name one model's parameters while packing another's
    geometry, which is exactly the mismatch reconcile_water_model exists to catch.
    """
    from md_templates.openmm.spec.resolve import load_profile

    kind = "peptide" if "peptide" in profile_id else "ligand"
    water, model = PROFILE_WATER[kind]
    build = load_profile(profile_id)["defaults"]["build"]
    assert build["forcefield"]["water"] == water
    assert build["solvation"]["water_model"] == model


@pytest.mark.parametrize("profile_id", [p.replace("-v2", "-v1") for p in V2_PROFILES])
def test_v1_profiles_are_compatibility_only_but_still_resolvable(profile_id):
    """Existing configurations must keep resolving to the water model they were run with."""
    from md_templates.openmm.spec.resolve import load_profile

    profile = load_profile(profile_id)
    assert profile["is_default"] is False, f"{profile_id} must not be selected by `default`"
    assert profile["superseded_by"] == profile_id.replace("-v1", "-v2")
    assert profile["defaults"]["build"]["forcefield"]["water"] == "amber19/tip3pfb.xml"
    assert "COMPATIBILITY ONLY" in profile["description"]


def test_exactly_one_default_per_explicit_route_and_method():
    """Two defaults is an ambiguity the resolver refuses; zero is a broken alias."""
    from md_templates.openmm.spec.resolve import list_profiles

    for route, method in EXPLICIT_ROUTES:
        defaults = [p["profile_id"] for p in list_profiles()
                    if p.get("route") == route and p.get("method") == method
                    and p.get("is_default")]
        assert len(defaults) == 1, f"{route}/{method}: {defaults}"


def test_the_package_defaults_use_opc_without_any_profile():
    """System generation has its own defaults, and they were TIP3P-FB independently of profiles."""
    from md_templates.openmm.config import DEFAULTS

    assert DEFAULTS["forcefield"]["water"] == "amber19/opc.xml"
    assert DEFAULTS["solvation"]["water_model"] == "opc"


def test_implicit_profiles_declare_no_water_model_at_all():
    """Implicit solvent has no water; a water model there would name parameters for nothing."""
    from md_templates.openmm.spec.resolve import list_profiles

    for profile in list_profiles():
        if not profile["profile_id"].startswith("implicit-"):
            continue
        build = profile["defaults"]["build"]
        assert build["forcefield"].get("water") is None, profile["profile_id"]
        assert "solvation" not in build, profile["profile_id"]
        assert build["implicit"]["model"] == "GBn2"


@pytest.mark.slow
def test_system_generation_without_a_water_override_produces_opc(tmp_path):
    """The end-to-end check: provenance must record OPC, not a profile label that implies it."""
    config = tmp_path / "system.json"
    config.write_text(json.dumps({"system": {"id": "ace_ala_nme", "type": "protein"}}))
    out = tmp_path / "bundle"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = json.loads((out / "system_manifest.json").read_text())
    resolved = manifest["resolved_system_config"]
    assert resolved["forcefield"]["water"] == "amber19/opc.xml"
    assert resolved["solvation"]["water_model"] == "opc"

    forcefield = json.loads((out / "forcefield.json").read_text())
    assert forcefield["water"] == "amber19/opc.xml"
    assert "opc" in json.dumps(forcefield["xml"]).lower()

    # and the water actually built is 4-site: OPC carries a virtual site per molecule
    from openmm import XmlSerializer

    system = XmlSerializer.deserialize((out / "system.xml").read_text())
    massless = sum(1 for i in range(system.getNumParticles())
                   if system.getParticleMass(i).value_in_unit_system(
                       __import__("openmm").unit.md_unit_system) == 0.0)
    assert massless > 0, "OPC is a 4-site model; a 3-site box would have no virtual sites"


@pytest.mark.slow
def test_the_active_alanine_examples_use_opc():
    """A worked example that still named TIP3P-FB would document the superseded default."""
    for path in ("test/ala/REST2/system_config.json", "test/ala/cMD/explicit/system_config.json"):
        document = json.loads((REPO_ROOT / path).read_text())
        forcefield = document.get("forcefield") or {}
        solvation = document.get("solvation") or {}
        if "water" in forcefield:
            assert forcefield["water"] == "amber19/opc.xml", path
        if "water_model" in solvation:
            assert solvation["water_model"] == "opc", path

    for path in ("test/ala/REST2/md_config.json", "test/ala/cMD/explicit/md_config.json"):
        document = json.loads((REPO_ROOT / path).read_text())
        profile = document.get("profile")
        if isinstance(profile, str) and profile.startswith("explicit-"):
            assert profile.endswith("-v2"), f"{path} names {profile}"


# ---------------------------------------------------------------------------------------------
# the two water fields must agree about how many sites water has
# ---------------------------------------------------------------------------------------------

def test_a_three_site_forcefield_is_not_packed_with_four_site_geometry():
    """The regression that moving the default to OPC actually caused.

    `forcefield.water` and `solvation.water_model` are independent settings. A manifest that names
    a water force field but no packing model leaves the packing model at its default -- and once
    that default became OPC, a config naming `tip3pfb.xml` was packed with 4-site geometry. OpenMM
    then fails deep inside `addSolvent` with "No template found for residue 3 (HOH). The residue
    contains extra sites", which names neither setting involved.
    """
    from md_templates.openmm.solvation import reconcile_water_model

    model, note = reconcile_water_model("amber19/tip3pfb.xml", "opc")
    assert model == "tip3pfb", "the packing model did not follow the force field"
    assert note is not None and "3-site" in note["reason"] and "4-site" in note["reason"]
    assert note["requested_water_model"] == "opc"
    assert note["resolved_water_model"] == "tip3pfb"


@pytest.mark.parametrize("water_xml,water_model", [
    ("amber19/opc.xml", "opc"),            # the new default pair
    ("amber19/tip3pfb.xml", "tip3p"),      # the v1 pair: same site count, deliberately different
    ("amber19/tip4pfb.xml", "tip4pew"),    # 4-site parameters packed from the 4-site stand-in box
])
def test_a_coherent_pair_is_left_exactly_as_declared(water_xml, water_model):
    """Same site count is not a conflict. tip3pfb parameters from a tip3p box is the documented
    arrangement, and reconciliation must not rewrite it -- doing so would change the resolved
    configuration, and with it the hash, of every existing v1 run."""
    from md_templates.openmm.solvation import reconcile_water_model

    model, note = reconcile_water_model(water_xml, water_model)
    assert model == water_model
    assert note is None


def test_an_unrecognised_water_resource_is_left_alone_rather_than_guessed():
    from md_templates.openmm.solvation import reconcile_water_model

    model, note = reconcile_water_model("custom/my_water.xml", "tip3p")
    assert (model, note) == ("tip3p", None)


def test_the_shipped_v1_and_v2_profiles_are_each_internally_coherent():
    """Whatever a profile declares, its two water fields must describe the same model."""
    from md_templates.openmm.solvation import reconcile_water_model
    from md_templates.openmm.spec.resolve import list_profiles

    for profile in list_profiles():
        build = profile["defaults"]["build"]
        water = build["forcefield"].get("water")
        model = (build.get("solvation") or {}).get("water_model")
        if not water or not model:
            continue
        _, note = reconcile_water_model(water, model)
        assert note is None, f"{profile['profile_id']} is incoherent: {note['reason']}"


# ---------------------------------------------------------------------------------------------
# the default water follows the solute, because the two force fields disagree about water
# ---------------------------------------------------------------------------------------------

def test_the_default_water_follows_the_solute_kind():
    """ff19SB wants OPC, Sage wants TIP3P, and a mixed system cannot have both.

    ff19SB's amino-acid-specific CMAPs were trained against QM energy surfaces computed in solution
    and validated with OPC; with TIP3P it over-stabilises helices, which is the property the CMAPs
    exist to get right. Sage's Lennard-Jones refit was conditioned on plain TIP3P. There is one box
    and one water model, so a complex resolves in ff19SB's favour: the protein backbone is the
    dominant error term.
    """
    from md_templates.openmm.config import default_water_for

    assert default_water_for("peptide") == ("amber19/opc.xml", "opc")
    assert default_water_for("complex") == ("amber19/opc.xml", "opc")
    assert default_water_for("ligand") == ("amber19/tip3p.xml", "tip3p")


def test_the_ligand_default_is_plain_tip3p_and_not_tip3p_fb():
    """TIP3P-FB is a separate ForceBalance refit with different charges and LJ terms.

    It is a better water model on its own merits, but it is not what Sage's parameters were
    conditioned against, and it is what the superseded v1 profiles used. Naming `tip3pfb.xml` here
    would reintroduce the pairing this change exists to correct while looking correct.
    """
    from md_templates.openmm.config import default_water_for
    from md_templates.openmm.spec.resolve import load_profile

    assert default_water_for("ligand")[0] == "amber19/tip3p.xml"
    for profile_id in ("explicit-md-ligand-v2", "explicit-rest2-ligand-v2"):
        assert load_profile(profile_id)["defaults"]["build"]["forcefield"]["water"] == \
            "amber19/tip3p.xml"
        assert load_profile(profile_id.replace("-v2", "-v1"))["defaults"]["build"]["forcefield"][
            "water"] == "amber19/tip3pfb.xml", "v1 must keep recording what it actually ran"


def test_both_route_defaults_are_internally_coherent():
    """Whatever each route defaults to, its parameters and its packing geometry must agree."""
    from md_templates.openmm.config import default_water_for
    from md_templates.openmm.solvation import reconcile_water_model

    for kind in ("peptide", "ligand", "complex"):
        water, model = default_water_for(kind)
        _, note = reconcile_water_model(water, model)
        assert note is None, f"{kind} default is incoherent: {note}"


@pytest.mark.slow
def test_generation_gives_each_route_its_own_water_without_any_override(tmp_path):
    """The end-to-end check, on both routes, with no water named anywhere in the input."""
    peptide = tmp_path / "peptide"
    config = tmp_path / "peptide.json"
    config.write_text(json.dumps({"system": {"id": "ace_ala_nme", "type": "protein"}}))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(peptide), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = json.loads((peptide / "system_manifest.json").read_text())["resolved_system_config"]
    assert resolved["forcefield"]["water"] == "amber19/opc.xml"
    assert resolved["solvation"]["water_model"] == "opc"

    ligand = tmp_path / "ligand"
    smiles = tmp_path / "lig.smi"
    smiles.write_text("O=C1CNC(=O)CNC(=O)CN1 ggg\n")
    lig_config = tmp_path / "ligand.json"
    lig_config.write_text(json.dumps({
        "system": {"id": "ggg", "type": "ligand"},
        # the SMILES route requires the chemistry to be declared, not inferred
        "ligand_build": {"formal_charge": 0, "stereochemistry_policy": "from_smiles",
                         "protonation_policy": "as_given", "conformer_generation": "etkdgv3",
                         "charge_model": "am1bcc", "parameterization_route": "openff-2.2.0"},
    }))
    result = _run(SYSTEM_GEN, "-i", str(smiles), "-o", str(ligand), "--config", str(lig_config))
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = json.loads((ligand / "system_manifest.json").read_text())["resolved_system_config"]
    assert resolved["forcefield"]["water"] == "amber19/tip3p.xml", \
        "the ligand route inherited the peptide route's water"
    assert resolved["solvation"]["water_model"] == "tip3p"

    # and the provenance must say the package chose it, and on what basis
    manifest = json.loads((ligand / "system_manifest.json").read_text())
    sources = manifest.get("sources") or manifest.get("value_sources") or {}
    if "forcefield.water" in sources:
        # the provenance names the force-field family that decided it, which is more specific than
        # the solute label it used to name
        assert "Sage" in sources["forcefield.water"], sources["forcefield.water"]
