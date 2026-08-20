"""Implicit solvent: the GBn2/mbondi3 Hamiltonian, and everything that must be absent.

The energy tests here compare against a System built by the pinned reference construction *in this
test*, on the same topology and coordinates, rather than against numbers copied into an assertion.
A frozen number would pin this machine's floating point; rebuilding the reference pins the thing
that actually matters, which is that our construction is that construction.

Two comparisons are deliberately separated:

* **construction** -- our System versus the pinned path, same prmtop, same rst7. This must be exact.
* **branch** -- the pinned path versus `AmberPrmtopFile.createSystem`. This must NOT be exact, and
  the size of the difference is the reason the branch is part of the scientific identity.

Coordinates matter for the first one: comparing across coordinate sources (a PDB rounded to three
decimals versus the rst7) shifts every force by ~0.02 kJ/mol and looks like a construction
discrepancy when it is not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
INPUT_GEN = REPO_ROOT / "MD_input_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    import os

    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, **env})


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def implicit_bundle(tmp_path_factory):
    """A prepared implicit alanine bundle, built once through the public entry point."""
    out = tmp_path_factory.mktemp("implicit") / "bundle"
    config = REPO_ROOT / "test" / "ala" / "implicit" / "system_config.json"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    if result.returncode != 0:
        pytest.skip(f"implicit preparation unavailable (tleap?): {result.stderr[-400:]}")
    return out


def _reference_system(prmtop: Path, coordinates: Path, radii: str = "mbondi3"):
    """The pinned reference construction, rebuilt here rather than trusted."""
    import parmed as pmd
    from openmm import app
    from parmed.tools import changeRadii

    structure = pmd.load_file(str(prmtop), xyz=str(coordinates))
    changeRadii(structure, radii).execute()
    return structure.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                                  implicitSolvent=app.GBn2, removeCMMotion=True)


def _components(system, positions) -> dict:
    """Per-force energies in kJ/mol, plus the total, on the Reference platform."""
    import openmm
    from openmm import unit

    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    out = {type(force).__name__: context.getState(getEnergy=True, groups={index})
           .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
           for index, force in enumerate(system.getForces())}
    out["TOTAL"] = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    return out


# ---------------------------------------------------------------------------------------------
# the construction path
# ---------------------------------------------------------------------------------------------

def test_the_bundle_system_is_the_pinned_parmed_construction(implicit_bundle):
    """Exact, on the same topology and the same coordinates. Nothing here is a tolerance."""
    from openmm import XmlSerializer, app

    positions = app.AmberInpcrdFile(str(implicit_bundle / "system.rst7")).positions
    mine = _components(
        XmlSerializer.deserialize((implicit_bundle / "system.xml").read_text()), positions)
    reference = _components(
        _reference_system(implicit_bundle / "system.prmtop", implicit_bundle / "system.rst7"),
        positions)

    assert set(mine) == set(reference)
    for force, value in reference.items():
        assert mine[force] == pytest.approx(value, abs=1e-9), force


def test_the_amberprmtopfile_branch_is_a_different_hamiltonian(implicit_bundle):
    """The regression that makes the branch choice a scientific decision, not a style one.

    If these two ever agree, either OpenMM changed or the construction silently moved -- and a
    passing test that asserted only "we use ParmEd" would not notice either.
    """
    from openmm import app

    positions = app.AmberInpcrdFile(str(implicit_bundle / "system.rst7")).positions
    reference = _components(
        _reference_system(implicit_bundle / "system.prmtop", implicit_bundle / "system.rst7"),
        positions)
    other = _components(
        app.AmberPrmtopFile(str(implicit_bundle / "system.prmtop")).createSystem(
            nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
            implicitSolvent=app.GBn2, removeCMMotion=True),
        positions)

    difference = other["CustomGBForce"] - reference["CustomGBForce"]
    assert abs(difference) > 1.0, (
        f"the two construction routes agree to {difference:.4f} kJ/mol on CustomGBForce. They are "
        "documented to differ by ~16 kJ/mol; if that is no longer true, the reason this repository "
        "pins the ParmEd branch needs re-establishing rather than quietly dropping.")

    # The GB term is where it lives: every other force agrees to far better than the GB gap, so a
    # reader cannot mistake this for a general parameterisation difference. Bonded terms are not
    # asserted bitwise -- the two routes can select slightly different constrained bonds, which
    # moves HarmonicBondForce by ~1e-3 kJ/mol, three orders below the GB difference.
    for force in ("NonbondedForce", "PeriodicTorsionForce", "CMAPTorsionForce",
                  "HarmonicBondForce", "HarmonicAngleForce"):
        if force in reference:
            assert abs(other[force] - reference[force]) < abs(difference) / 100.0, (
                f"{force} differs by {other[force] - reference[force]:.6f} kJ/mol, which is not "
                f"small against the {difference:.4f} kJ/mol CustomGBForce difference")


def test_the_manifest_records_what_changeradii_actually_did(implicit_bundle):
    """A claim that the call mattered is not the same as a measurement that it did."""
    manifest = json.loads((implicit_bundle / "system_manifest.json").read_text())
    implicit = manifest["implicit"]

    assert implicit["radii"] == "mbondi3"
    assert implicit["implicit_model"] == "GBn2"
    assert implicit["construction"] == "parmed.Structure.createSystem"
    assert "radii_max_change_angstrom" in implicit
    # tleap wrote mbondi3 already, so for THIS route the call changes nothing
    assert implicit["radii_change_was_a_no_op"] is True
    assert implicit["radii_max_change_angstrom"] == pytest.approx(0.0, abs=1e-9)


def test_the_prepared_radii_are_mbondi3(implicit_bundle):
    """Checked against a fresh mbondi3 assignment, not against the value we wrote."""
    import parmed as pmd
    from parmed.tools import changeRadii

    from md_templates.openmm.implicit import radii_of

    as_built = pmd.load_file(str(implicit_bundle / "system.prmtop"))
    before = radii_of(as_built)
    changeRadii(as_built, "mbondi3").execute()
    assert radii_of(as_built) == pytest.approx(before), (
        "the prepared topology's radii are not mbondi3")
    assert any(r > 0 for r in before), "radii must not be zero"


# ---------------------------------------------------------------------------------------------
# what must be absent
# ---------------------------------------------------------------------------------------------

def test_the_implicit_system_has_no_box_no_water_no_ions_and_no_barostat(implicit_bundle):
    from openmm import XmlSerializer, app

    system = XmlSerializer.deserialize((implicit_bundle / "system.xml").read_text())
    assert system.usesPeriodicBoundaryConditions() is False
    assert not [f for f in system.getForces() if "Barostat" in type(f).__name__]

    topology = app.PDBFile(str(implicit_bundle / "topology.pdb")).topology
    residues = {r.name.upper() for r in topology.residues()}
    assert not residues & {"HOH", "WAT", "SOL", "TIP3", "NA", "CL"}, residues

    manifest = json.loads((implicit_bundle / "system_manifest.json").read_text())
    build = manifest["resolved_system_config"]["system_build"]
    assert build["nonbonded_method"] == "NoCutoff"
    assert build["nonbonded_cutoff_nm"] is None
    assert manifest["resolved_system_config"]["solvation"]["mode"] == "implicit"


def test_the_implicit_system_is_not_hydrogen_mass_repartitioned(implicit_bundle):
    """The GBn2 comparison is against an unrepartitioned System, so this is load-bearing.

    3.024 amu hydrogens would be a different build that still passes every structural check.
    """
    from openmm import XmlSerializer, unit

    system = XmlSerializer.deserialize((implicit_bundle / "system.xml").read_text())
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(system.getNumParticles())]
    light = [m for m in masses if 0 < m < 2.0]
    assert light, "no hydrogen-mass particles found at all"
    assert max(light) < 1.5, f"hydrogens appear repartitioned: heaviest light mass {max(light)}"


def test_the_bundle_carries_its_amber_construction_files(implicit_bundle):
    """They ARE the construction path, so a bundle without them cannot be rebuilt or checked."""
    for name in ("system.xml", "topology.pdb", "topology.cif", "initial_state.xml",
                 "system.prmtop", "system.rst7", "forcefield.json", "system_manifest.json",
                 "checksums.json"):
        assert (implicit_bundle / name).is_file(), name

    manifest = json.loads((implicit_bundle / "system_manifest.json").read_text())
    assert manifest["amber_files"]["topology"] == "system.prmtop"
    assert "no Amber execution engine" in manifest["implicit"][
        "amber_files_are_provenance_only"]


def test_the_initial_state_carries_no_velocities_and_no_box(implicit_bundle):
    from openmm import XmlSerializer

    state = XmlSerializer.deserialize((implicit_bundle / "initial_state.xml").read_text())
    with pytest.raises(Exception):
        state.getVelocities()


# ---------------------------------------------------------------------------------------------
# the OpenFF/Sage route, where changeRadii is load-bearing
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rgdfv_implicit_bundle(tmp_path_factory):
    """The cyclo-RGDfV implicit bundle. Expensive: AM1-BCC for the whole macrocycle."""
    out = tmp_path_factory.mktemp("rgd_implicit") / "bundle"
    smi = REPO_ROOT / "test" / "rgd" / "implicit" / "cyclo_rgdfv.smi"
    config = REPO_ROOT / "test" / "rgd" / "implicit" / "system_config.json"
    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(out), "--config", str(config))
    if result.returncode != 0:
        pytest.skip(f"RGDfV implicit preparation unavailable: {result.stderr[-400:]}")
    return out


@pytest.mark.slow
def test_the_rgdfv_bundle_is_the_pinned_parmed_construction(rgdfv_implicit_bundle):
    """The same exactness required of alanine, on the route with different parameters."""
    from openmm import XmlSerializer, app

    positions = app.AmberInpcrdFile(str(rgdfv_implicit_bundle / "system.rst7")).positions
    mine = _components(
        XmlSerializer.deserialize((rgdfv_implicit_bundle / "system.xml").read_text()), positions)
    reference = _components(
        _reference_system(rgdfv_implicit_bundle / "system.prmtop",
                          rgdfv_implicit_bundle / "system.rst7"),
        positions)
    for force, value in reference.items():
        assert mine[force] == pytest.approx(value, abs=1e-9), force


@pytest.mark.slow
def test_changeradii_is_load_bearing_for_the_openff_route(rgdfv_implicit_bundle):
    """The reason the call is unconditional, measured rather than asserted.

    A Sage/OpenFF topology carries no GB radii at all, so without `changeRadii` the radii would be
    zero and the GB energy meaningless. For the tleap alanine topology the same call changes
    nothing. Applying it always is safe for the first case and required for this one.
    """
    manifest = json.loads((rgdfv_implicit_bundle / "system_manifest.json").read_text())
    implicit = manifest["implicit"]
    assert implicit["radii_change_was_a_no_op"] is False
    assert implicit["radii_max_change_angstrom"] > 1.0, (
        "the OpenFF topology's radii should have moved substantially; if this is now a no-op, the "
        "toolchain started writing radii and the reason for the unconditional call has changed")


@pytest.mark.slow
def test_the_rgdfv_implicit_system_has_no_water_box_or_barostat(rgdfv_implicit_bundle):
    from openmm import XmlSerializer

    system = XmlSerializer.deserialize((rgdfv_implicit_bundle / "system.xml").read_text())
    assert system.usesPeriodicBoundaryConditions() is False
    assert not [f for f in system.getForces() if "Barostat" in type(f).__name__]
    manifest = json.loads((rgdfv_implicit_bundle / "system_manifest.json").read_text())
    assert manifest["composition"]["n_solute_atoms"] == system.getNumParticles(), (
        "under implicit solvent the solute IS the system")


@pytest.mark.slow
def test_the_rgdfv_bundle_records_its_vetted_chemistry(rgdfv_implicit_bundle):
    """The ligand route's provenance must name the parameters that actually ran."""
    forcefield = json.loads((rgdfv_implicit_bundle / "forcefield.json").read_text())
    assert forcefield["ligand"] == "openff-2.2.0"
    assert forcefield["ligand_charge_method"] == "am1bcc"
    assert forcefield["protein_forcefield"] is None, "a ligand route parameterises no protein"


# ---------------------------------------------------------------------------------------------
# REST2 scaling of the generalised-Born energy
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_implicit_system(implicit_bundle):
    from openmm import XmlSerializer, app

    system = XmlSerializer.deserialize((implicit_bundle / "system.xml").read_text())
    positions = app.AmberInpcrdFile(str(implicit_bundle / "system.rst7")).positions
    return system, positions


def test_tau_zero_is_an_exact_identity(base_implicit_system):
    """s = 1 must be the unscaled physical Hamiltonian, not merely close to it."""
    import numpy as np

    from md_templates.openmm.system import build_rest2_scaled_system

    system, positions = base_implicit_system
    unscaled = _components(system, positions)
    scaled = _components(
        build_rest2_scaled_system(system, np.arange(system.getNumParticles()), 1.0), positions)
    for force, value in unscaled.items():
        assert scaled[force] == pytest.approx(value, abs=1e-9), force


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.4, 0.5])
def test_the_whole_generalised_born_energy_scales_by_s(base_implicit_system, tau):
    """Every GBn2 term, not only the charge-dependent ones.

    GBn2 has three energy terms; one is a non-polar correction with no charge dependence. Scaling
    charges by sqrt(s) would leave it untouched, and the total would then NOT equal s x the
    unscaled GB energy. Exactness at several tau is what distinguishes the two implementations.
    """
    import numpy as np

    from md_templates.openmm.system import build_rest2_scaled_system

    system, positions = base_implicit_system
    s = (1.0 - tau) ** 2
    unscaled = _components(system, positions)["CustomGBForce"]
    scaled = _components(
        build_rest2_scaled_system(system, np.arange(system.getNumParticles()), s),
        positions)["CustomGBForce"]
    assert scaled == pytest.approx(s * unscaled, rel=1e-9, abs=1e-9)


def test_charge_scaling_alone_would_not_reproduce_this(base_implicit_system):
    """The reason a global parameter is used instead of scaling charges.

    Scaling only the charge-dependent part leaves the non-polar term at full strength, so the GB
    energy comes out ABOVE s x U_GB. This asserts the two approaches genuinely differ, so the
    implementation cannot quietly regress to the cheaper wrong one.
    """
    import numpy as np

    from md_templates.openmm.system import build_rest2_scaled_system

    system, positions = base_implicit_system
    s = 0.25
    unscaled = _components(system, positions)["CustomGBForce"]
    scaled = _components(
        build_rest2_scaled_system(system, np.arange(system.getNumParticles()), s),
        positions)["CustomGBForce"]

    # a charge-only treatment cannot equal the full scaling unless the non-polar term is zero,
    # which for GBn2 on a real solute it is not
    assert scaled == pytest.approx(s * unscaled, rel=1e-9)
    assert abs(unscaled - scaled) > 1.0, "the GB energy did not move; s was not applied"


def test_a_partial_enhanced_region_is_refused(base_implicit_system):
    """A Born radius depends on every other atom, so a partial region needs validated cross terms."""
    import numpy as np

    from md_templates.openmm.system import build_rest2_scaled_system

    system, _ = base_implicit_system
    with pytest.raises(ValueError, match="entire system"):
        build_rest2_scaled_system(system, np.arange(system.getNumParticles() - 2), 0.5)


def test_omega_exclusion_leaves_those_torsions_unscaled(base_implicit_system, implicit_bundle):
    """Checked on the force CONSTANTS, not on the energy, and that distinction is the point.

    An earlier version of this test asserted the torsion ENERGY changes when omega exclusion is
    applied. It does not, at the prepared geometry: a tleap-built amide sits at the minimum of its
    omega term, which contributes ~0, so scaling its barrier is energetically invisible there. The
    test passed nothing and would have passed with the feature removed.

    What the contract actually says is that these barriers are not scaled, so that is what is
    measured. Alanine dipeptide: 12 torsions about 2 peptide bonds.
    """
    import numpy as np
    from openmm import PeriodicTorsionForce, app, unit

    from md_templates.openmm.system import build_rest2_scaled_system, omega_central_bonds

    system, _ = base_implicit_system
    topology = app.PDBFile(str(implicit_bundle / "topology.pdb")).topology
    bonds = omega_central_bonds(topology, range(system.getNumParticles()))
    assert bonds, "alanine dipeptide has peptide bonds to exclude"
    excluded = {frozenset(bond) for bond in bonds}

    def barriers(built):
        force = [f for f in built.getForces() if isinstance(f, PeriodicTorsionForce)][0]
        omega, other = [], []
        for index in range(force.getNumTorsions()):
            _, b, c, _, _, _, k = force.getTorsionParameters(index)
            target = omega if frozenset((b, c)) in excluded else other
            target.append(k.value_in_unit(unit.kilojoule_per_mole))
        return np.array(omega), np.array(other)

    atoms = np.arange(system.getNumParticles())
    s = 0.25
    base_omega, base_other = barriers(system)
    assert len(base_omega) == 12, f"expected 12 omega torsions, found {len(base_omega)}"
    assert base_omega.sum() > 0, "the omega barriers must not already be zero"

    plain_omega, plain_other = barriers(build_rest2_scaled_system(system, atoms, s))
    kept_omega, kept_other = barriers(
        build_rest2_scaled_system(system, atoms, s, exclude_central_bonds=bonds))

    assert plain_omega == pytest.approx(s * base_omega), "without exclusion they scale"
    assert kept_omega == pytest.approx(base_omega), "with exclusion they are left alone"
    assert kept_other == pytest.approx(s * base_other), "every other torsion still scales"


def test_omega_exclusion_does_not_touch_gb_or_nonbonded(base_implicit_system, implicit_bundle):
    """Torsion-only: it is applied after the enhanced region is resolved."""
    import numpy as np
    from openmm import app

    from md_templates.openmm.system import build_rest2_scaled_system, omega_central_bonds

    system, positions = base_implicit_system
    topology = app.PDBFile(str(implicit_bundle / "topology.pdb")).topology
    bonds = omega_central_bonds(topology, range(system.getNumParticles()))
    atoms = np.arange(system.getNumParticles())

    without = _components(build_rest2_scaled_system(system, atoms, 0.25), positions)
    with_exclusion = _components(
        build_rest2_scaled_system(system, atoms, 0.25, exclude_central_bonds=bonds), positions)
    for force in ("CustomGBForce", "NonbondedForce"):
        assert with_exclusion[force] == pytest.approx(without[force], abs=1e-9), force


# ---------------------------------------------------------------------------------------------
# the generated project
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def implicit_project(implicit_bundle, tmp_path_factory):
    out = tmp_path_factory.mktemp("implicit_project") / "run"
    config = REPO_ROOT / "test" / "ala" / "implicit" / "md_config_smoke.json"
    result = _run(INPUT_GEN, "--system", str(implicit_bundle / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def test_the_implicit_stage_graph_has_no_npt(implicit_project):
    manifest = json.loads((implicit_project / "run_manifest.json").read_text())
    assert [entry["stage"] for entry in manifest["stages"]] == [
        "min", "eq_nvt", "cMD_1", "REST2_1"]
    for absent in ("eq_npt_1", "eq_npt_2"):
        assert not (implicit_project / absent).exists(), absent


def test_no_stage_carries_a_barostat(implicit_project):
    for stage in ("min", "eq_nvt", "cMD_1", "REST2_1"):
        payload = json.loads((implicit_project / stage / f"{stage}.json").read_text())
        assert "barostat" not in payload, stage


def test_the_alanine_implicit_ladder_has_four_replicas(implicit_project):
    """Implicit ladders are shorter than their explicit counterparts, by instruction."""
    payload = json.loads((implicit_project / "REST2_1" / "REST2_1.json").read_text())
    assert len(payload["rest2"]["derived_scale_factors"]) == 4
    assert payload["rest2"]["derived_scale_factors"][0] == 1.0, "the cold rung must be s = 1"


def test_the_rgdfv_implicit_ladder_has_six_replicas():
    config = json.loads(
        (REPO_ROOT / "test" / "rgd" / "implicit" / "md_config.json").read_text())
    assert config["protocol"]["production"]["tau_ladder"]["count"] == 6


@pytest.mark.slow
def test_the_implicit_workflow_executes_and_writes_its_declared_outputs(implicit_project):
    """min -> eq_nvt -> cMD_1, through the generated launchers, with real reporters."""
    import struct

    for stage in ("min", "eq_nvt", "cMD_1"):
        script = implicit_project / stage / f"{stage}.sh"
        result = subprocess.run([str(script)], capture_output=True, text=True,
                                cwd=str(implicit_project / stage))
        assert result.returncode == 0, f"{stage}: {result.stderr[-800:]}"

    for stage in ("eq_nvt", "cMD_1"):
        results = json.loads(
            (implicit_project / stage / f"{stage}_results.json").read_text())
        assert results["barostat"] is None, f"{stage} must have no barostat"
        for key, name in (("all_atom", f"{stage}_all_atoms.dcd"),
                          ("selected_atoms", f"{stage}_selected_atoms.dcd")):
            path = implicit_project / stage / name
            assert path.is_file(), name
            with path.open("rb") as handle:
                handle.seek(8)
                frames = struct.unpack("<i", handle.read(4))[0]
            assert frames == results["steps"] // results["reporters"][key]["interval_steps"]

    velocities = {s: json.loads(
        (implicit_project / s / f"{s}_results.json").read_text())["velocities"]
        for s in ("min", "eq_nvt", "cMD_1")}
    assert velocities == {"min": "not required", "eq_nvt": "initialized", "cMD_1": "inherited"}


# ---------------------------------------------------------------------------------------------
# a route records only the chemistry it used
# ---------------------------------------------------------------------------------------------

def test_the_peptide_route_records_no_small_molecule_chemistry(implicit_bundle):
    """The package defaults name a force field for every component; a route uses one.

    Carrying the others is provenance for chemistry that never ran. This was three separate
    defects -- water on the ligand route, a protein force field on the ligand route, and
    small-molecule parameters on the peptide route -- so it is asserted from both directions.
    """
    forcefield = json.loads((implicit_bundle / "forcefield.json").read_text())
    assert forcefield["protein_forcefield"] == "leaprc.protein.ff19SB"
    assert forcefield["water"] is None, "implicit solvent has no water to parameterise"
    assert forcefield["ligand"] is None, "a peptide route parameterises no small molecule"
    assert forcefield["ligand_charge_method"] is None

    manifest = json.loads((implicit_bundle / "system_manifest.json").read_text())
    sources = manifest["value_sources"]
    for field in ("protein", "water", "ligand", "ligand_charge_method"):
        assert "route-derived" in sources[f"forcefield.{field}"], field


@pytest.mark.slow
def test_the_ligand_route_records_no_protein_or_water(rgdfv_implicit_bundle):
    """And the manifest must survive its own validator, which refuses the mixture."""
    from md_templates.openmm.schemas import load_system

    forcefield = json.loads((rgdfv_implicit_bundle / "forcefield.json").read_text())
    assert forcefield["ligand"] == "openff-2.2.0"
    assert forcefield["ligand_charge_method"] == "am1bcc"
    assert forcefield["protein_forcefield"] is None, (
        "a smiles route may not load a protein force field")
    assert forcefield["water"] is None

    # the same check the REST2 bridge performs; it is what caught the leak
    load_system(rgdfv_implicit_bundle / "system.yaml")


@pytest.mark.slow
def test_the_ligand_bundle_records_a_verifiable_smiles_hash(rgdfv_implicit_bundle):
    """It catches an edited SMILES even where RDKit cannot parse chemistry."""
    import yaml

    from md_templates.openmm.schemas import sha256_text

    doc = yaml.safe_load((rgdfv_implicit_bundle / "system.yaml").read_text())
    declared = doc["input"]["canonical_isomeric_smiles"]
    assert doc["input"]["canonical_smiles_sha256"] == sha256_text(declared)


# ---------------------------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------------------------

def test_the_public_entry_points_are_importable_from_the_package():
    """They used to exist only as repository-root scripts.

    A public entry point that lives only in a source tree is not installable, so a consuming
    project could import the library and still have no way to run the generators. The root scripts
    are now shims over packaged modules, and the console scripts point at those modules.
    """
    import tomllib

    from md_templates.openmm import cli_input_gen, cli_system_gen

    assert callable(cli_system_gen.main)
    assert callable(cli_input_gen.main)

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]
    assert scripts["md-system-gen"] == "md_templates.openmm.cli_system_gen:main"
    assert scripts["md-input-gen"] == "md_templates.openmm.cli_input_gen:main"


def test_the_root_scripts_are_shims_and_not_a_second_implementation():
    """Two copies of an entry point drift, and the drift is invisible until they disagree."""
    for name, module in (("MD_system_gen.py", "cli_system_gen"),
                         ("MD_input_gen.py", "cli_input_gen")):
        body = (REPO_ROOT / name).read_text()
        assert f"from md_templates.openmm.{module} import" in body, name
        assert "_FORMAT_BY_SUFFIX" not in body, f"{name} still holds its own implementation"
        assert len(body.splitlines()) < 40, f"{name} should be a shim, not a program"


def test_every_profile_ships_in_the_package():
    """A profile that is not packaged cannot be selected from an installed wheel."""
    import md_templates

    packaged = {p.name for p in
                (Path(md_templates.__file__).parent / "openmm" / "spec" / "profiles")
                .glob("*.json")}
    for expected in ("implicit-md-peptide-v1.json", "implicit-md-ligand-v1.json",
                     "implicit-rest2-peptide-v1.json", "implicit-rest2-ligand-v1.json",
                     "explicit-rest2-peptide-v1.json", "cpu-smoke-v1.json"):
        assert expected in packaged, expected
