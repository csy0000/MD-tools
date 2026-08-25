"""Compare a BUILT System against the configuration that asked for it.

This is the guard whose absence let the defect ship. `test_conservative_defaults` compares a
profile's timestep against that same profile's hydrogen mass -- a profile against itself -- so it
passes for any profile that states both, however the builder actually behaves. Nothing compared a
profile against the System it produces, and that is precisely where the two disagreed: every
implicit `-hmr-v1` profile declared 3.024 amu and 4 fs while the builder returned 1.008 amu
hydrogens.

So these tests deserialize `system.xml` and check it against the resolved build:

* hydrogen masses against `build.hydrogen_mass`;
* constraint count against `build.constraints`;
* periodicity against the solvation mode;
* `CustomGBForce` present for implicit and absent for explicit.

COVERAGE. The guard inspects BUILD-defining fields only, and a profile's production method (md vs
rest2) does not change the System, so profiles are grouped by the build they produce and one bundle
is built per distinct group. `test_every_shipped_profile_is_covered_by_a_build` asserts that the
grouping actually accounts for every profile, so a new profile that falls outside it fails loudly
rather than being silently untested.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
               / "systems" / "ace_ala_nme.pdb")

#: Phenol, the standard small molecule for charge-derivation tests (see tests/molecules.py).
#: The guard checks mass, constraints and periodicity, none of which depend on which molecule it
#: is, so it must not pay for a macrocycle: 0.6 s here against ~40 min for cyclo-(RGDfV). Phenol
#: over ethanol because an aromatic ring and a hydroxyl exercise bond perception for 0.3 s more.
LIGAND_SMILES = "c1ccc(cc1)O"

#: Small enough to keep the explicit builds quick. Padding does not enter anything this guard
#: compares -- box SIZE is checked by the geometry tests, box EXISTENCE is what matters here.
TEST_PADDING_NM = 0.6


def _profiles() -> list[dict]:
    from md_templates.openmm.spec.resolve import list_profiles

    return list_profiles()


def _group_of(profile: dict) -> tuple:
    """The build a profile produces: route, solvent treatment, and repartitioning."""
    build = profile["defaults"]["build"]
    solvation = "implicit" if build.get("implicit") else "explicit"
    mass = build.get("hydrogen_mass")
    return (profile["route"], solvation, str(mass) if mass else None,
            build.get("hmr_scope") or "none")


def _groups() -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    for profile in _profiles():
        out.setdefault(_group_of(profile), []).append(profile["profile_id"])
    return {k: sorted(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def _build_bundle_or_raise(tmp_path: Path, group: tuple) -> Path:
    route, solvation, mass, scope = group
    config: dict = {"randomness": {"master_seed": 20260825}}
    if solvation == "implicit":
        config["solvation"] = {"mode": "implicit", "implicit_model": "GBn2", "radii": "mbondi3"}
    else:
        config["solvation"] = {"water_model": "opc", "box_shape": "dodecahedron",
                               "padding_nm": TEST_PADDING_NM, "ionic_strength_molar": 0.15}
    if mass is not None:
        config["system_build"] = {"hydrogen_mass_amu": float(str(mass).split()[0]),
                                  "hmr_scope": scope}

    out = tmp_path / "bundle"
    if route == "pdb":
        config["system"] = {"id": "ace_ala_nme", "type": "protein"}
        source = ALANINE_PDB
    else:
        config["system"] = {"id": "test_ligand", "type": "ligand"}
        # every field the SMILES route requires: none is guessable, and a missing one is a
        # refusal rather than a default
        config["ligand_build"] = {"formal_charge": 0,
                                  "stereochemistry_policy": "from_smiles",
                                  "protonation_policy": "as_given",
                                  "conformer_generation": "etkdgv3",
                                  "charge_model": "am1bcc",
                                  "parameterization_route": "openff-2.2.0"}
        source = tmp_path / "ligand.smi"
        source.write_text(LIGAND_SMILES + "\n")

    config_path = tmp_path / "system_config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(SYSTEM_GEN), "-i", str(source), "-o", str(out),
         "--config", str(config_path)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{(result.stdout + result.stderr)[-1500:]}")
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """One bundle per distinct build, built once for the module.

    Each build is attempted independently and its outcome recorded, rather than letting the first
    failure skip the whole module -- which is what happened when the SMILES route was missing a
    required field: eight builds that would have passed reported as skips, and a skip that hides a
    build failure is indistinguishable from one that reports a missing dependency.
    """
    out = {}
    for index, group in enumerate(_groups()):
        directory = tmp_path_factory.mktemp(f"g{index}")
        try:
            out[group] = {"bundle": _build_bundle_or_raise(directory, group), "error": None}
        except RuntimeError as error:
            out[group] = {"bundle": None, "error": str(error)}
    return out


def _resolve(built: dict, group: tuple) -> Path:
    """The bundle for `group`, failing loudly with the build output if it could not be built."""
    entry = built[group]
    if entry["bundle"] is None:
        pytest.fail(f"the bundle for {group} could not be built:\n{entry['error']}")
    return entry["bundle"]


def test_every_shipped_profile_is_covered_by_a_build():
    """No profile may fall outside the grouping and go silently unchecked."""
    covered = {pid for ids in _groups().values() for pid in ids}
    everything = {p["profile_id"] for p in _profiles()}
    assert covered == everything, everything - covered


@pytest.mark.slow
@pytest.mark.parametrize("group", list(_groups()), ids=lambda g: f"{g[0]}-{g[1]}-{'hmr' if g[2] else 'plain'}")
def test_the_built_system_matches_the_configuration_that_asked_for_it(built, group):
    from openmm import CustomGBForce, XmlSerializer, unit

    route, solvation, mass, scope = group
    bundle = _resolve(built, group)
    system = XmlSerializer.deserialize((bundle / "system.xml").read_text())
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(system.getNumParticles())]
    record = json.loads((bundle / "system_simbox.json").read_text())

    # ---- hydrogen mass against what the configuration declared -------------------------------
    hydrogens = [m for m in masses if 0 < m < 5.0]
    assert hydrogens, "no hydrogen-mass particles found"
    if mass is None:
        assert max(hydrogens) < 1.5, (
            f"the configuration declared no repartitioning but the System carries "
            f"{max(hydrogens)} amu hydrogens")
    else:
        target = float(str(mass).split()[0])
        solute_hydrogens = [m for m in hydrogens if abs(m - target) < 1e-6]
        assert solute_hydrogens, (
            f"the configuration declared {target} amu hydrogens but the System carries "
            f"{sorted(set(round(h, 3) for h in hydrogens))}")

    # ---- constraints against build.constraints ------------------------------------------------
    # The two routes record this differently -- the implicit build states `constraints`, the
    # explicit one states `n_constraints` -- so both shapes are read rather than one being assumed.
    if "constraints" in record:
        assert record["constraints"] == "HBonds"
    assert record["n_constraints"] == system.getNumConstraints(), (
        "the recorded constraint count disagrees with the System it describes")
    assert system.getNumConstraints() > 0, (
        "HBonds was declared but the System constrains nothing, so the timestep rationale that "
        "depends on it does not hold")
    # every constrained pair must involve a hydrogen: that is what HBonds means
    hydrogen_indices = {i for i, m in enumerate(masses) if 0 < m < 5.0}
    for c in range(min(system.getNumConstraints(), 200)):
        i, j, _ = system.getConstraintParameters(c)
        assert i in hydrogen_indices or j in hydrogen_indices, (
            f"constraint {c} joins {i} and {j}, neither of which is a hydrogen")

    # ---- periodicity against the solvation mode -----------------------------------------------
    periodic = system.usesPeriodicBoundaryConditions()
    forces = {type(system.getForce(i)).__name__ for i in range(system.getNumForces())}
    if solvation == "implicit":
        assert not periodic, "an implicit System must not be periodic"
        assert any(f == "CustomGBForce" for f in forces), forces
        assert record["geometry"] is None
    else:
        assert periodic, "an explicit solvent System must be periodic"
        assert not any(f == "CustomGBForce" for f in forces), (
            f"an explicit System carries a GB force: {forces}")
        assert record["geometry"] is not None

    # ---- the provenance must state the same thing as the System -------------------------------
    stated = record["hmr"]
    if mass is None:
        assert stated["scope"] == "none"
        assert stated["target_hydrogen_mass_amu"] is None
        assert stated["n_hydrogens_repartitioned"] == 0
    else:
        assert stated["scope"] in ("solute", "all", "all-non-water")
        assert stated["target_hydrogen_mass_amu"] == pytest.approx(float(str(mass).split()[0]))
        assert stated["n_hydrogens_repartitioned"] > 0


@pytest.mark.slow
def test_a_repartitioned_build_conserves_mass_against_its_plain_counterpart(built):
    """Pairwise, per route and solvation: repartitioning moves mass, it does not create it."""
    from openmm import XmlSerializer, unit

    def _total(bundle: Path) -> float:
        system = XmlSerializer.deserialize((bundle / "system.xml").read_text())
        return sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                   for i in range(system.getNumParticles()))

    compared = 0
    for group, entry in built.items():
        route, solvation, mass, _scope = group
        if mass is None or entry["bundle"] is None:
            continue
        bundle = entry["bundle"]
        plain = next((e["bundle"] for (r, s, m, _), e in built.items()
                      if r == route and s == solvation and m is None and e["bundle"]), None)
        if plain is None:
            continue
        assert _total(bundle) == pytest.approx(_total(plain), abs=1e-5), (
            f"{route}/{solvation}: repartitioning changed the total mass")
        compared += 1
    assert compared, "no repartitioned/plain pair was compared, so this asserted nothing"
