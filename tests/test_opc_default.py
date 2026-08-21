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


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


@pytest.mark.parametrize("route,method", EXPLICIT_ROUTES)
def test_the_default_alias_resolves_to_the_v2_opc_profile(route, method):
    from md_templates.openmm.spec.resolve import select_profile

    profile = select_profile(route, method)
    assert profile["profile_id"].endswith("-v2"), profile["profile_id"]
    build = profile["defaults"]["build"]
    assert build["forcefield"]["water"] == "amber19/opc.xml"
    assert build["solvation"]["water_model"] == "opc"


@pytest.mark.parametrize("profile_id", V2_PROFILES)
def test_every_v2_profile_names_opc_in_both_places(profile_id):
    """The force field and the packing model are separate fields and both must say OPC.

    They mean different things -- one supplies parameters, the other picks the geometry template
    Modeller packs with -- so a profile can name OPC parameters while packing TIP3P geometry.
    """
    from md_templates.openmm.spec.resolve import load_profile

    build = load_profile(profile_id)["defaults"]["build"]
    assert build["forcefield"]["water"] == "amber19/opc.xml"
    assert build["solvation"]["water_model"] == "opc"


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
