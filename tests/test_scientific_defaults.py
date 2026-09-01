"""The 0.4 scientific defaults, traced from the public setting to the built System.

Documentation and YAML agreeing proves nothing on its own: this repository has already shipped a
configuration requesting `hydrogen_mass_amu: 3.024` beside a System carrying 1.008 amu hydrogens,
and every check that read only the configuration passed. So each test here ends at an artefact --
a serialized `System`, a generated `stage.yaml`, a written record -- rather than at a default.

Split by cost, and marked accordingly:

    (unmarked)  pure geometry, resolution and generated source. No System is built.
    slow        one system built and loaded per combination.
    gpu         minimisation or integration, which runs on CUDA and nowhere else.

The evidence for the choices themselves is `docs/md-defaults-scientific-rationale.md`; this file
only checks that what is implemented is what is documented.
"""
from __future__ import annotations

import shutil

import pytest
import yaml

from .conftest import ALA_PDB, run_cli, run_stage


# --- the box gate, without building anything ---------------------------------

class _Positions:
    """The two attributes `_resolve_box` reads off a Modeller, and nothing else."""

    def __init__(self, coordinates):
        from openmm import unit

        self.positions = unit.Quantity(coordinates, unit.nanometer)


def _geometry(*, padding_nm, radius_nm, cutoff_nm=1.0, shape="dodecahedron"):
    from md_tools.openmm.builder_defaults import DEFAULTS
    from md_tools.openmm.solvation import _resolve_box

    cfg = yaml.safe_load(yaml.safe_dump(DEFAULTS))
    cfg["solvation"].update({"padding_nm": padding_nm, "box_shape": shape})
    cfg["system_build"]["nonbonded_cutoff_nm"] = cutoff_nm
    # A two-atom solute of exactly the requested bounding radius, centred on the origin.
    solute = _Positions([(-radius_nm, 0.0, 0.0), (radius_nm, 0.0, 0.0)])
    return _resolve_box(solute, cfg)


def test_the_default_padding_reaches_the_cutoff_and_minimum_image_gate():
    """1.5 nm is a REQUESTED solute-to-box clearance, and four distances are not interchangeable.

    OpenMM's `addSolvent` padding sets the box WIDTH. What a self-interaction argument cares about
    is the distance to the nearest periodic COPY, which is the shortest lattice translation. What
    OpenMM's cutoff legality check compares against is the minimum reduced-box HEIGHT, which for a
    rhombic dodecahedron is width/sqrt(2) -- 29% smaller. All four are recorded separately.
    """
    geometry = _geometry(padding_nm=1.5, radius_nm=1.2)

    assert geometry["padding_nm_requested"] == 1.5
    # width = max(2R + padding, 2*padding) -- exactly Modeller.addSolvent. The second term is the
    # one people forget: for a small solute the box is set by the padding alone, not by the solute.
    assert geometry["box_width_nm"] == pytest.approx(2 * 1.2 + 1.5)
    assert geometry["shortest_lattice_translation_nm"] == pytest.approx(3.9)
    # ...so the clearance to the nearest periodic COPY is the requested padding, and no more
    assert geometry["solute_image_clearance_nm"] == pytest.approx(1.5)
    assert geometry["min_reduced_box_height_nm"] == pytest.approx(3.9 / 2 ** 0.5, rel=1e-4)
    # the gate itself: 2*cutoff + margin, and the box must clear it
    assert geometry["required_cutoff_height_nm"] == pytest.approx(2.1)
    assert geometry["min_reduced_box_height_nm"] > geometry["required_cutoff_height_nm"]
    assert geometry["grown_for_cutoff"] is False
    assert geometry["max_legal_cutoff_nm"] > 1.0


def test_a_solute_small_enough_that_1p5nm_would_break_the_cutoff_grows_the_box():
    """The gate is enforced on the BUILT box, not argued from the padding number.

    A rhombic dodecahedron's reduced-box height is width/sqrt(2), so a request that looks generous
    as a padding can still fall below 2*cutoff + margin. The box is then grown until it clears,
    which is what makes "the clearance is enough" a statement about the box that was built rather
    than about the number that was typed.
    """
    geometry = _geometry(padding_nm=1.0, radius_nm=0.1)

    assert geometry["grown_for_cutoff"] is True
    assert geometry["min_reduced_box_height_nm"] >= geometry["required_cutoff_height_nm"]
    # growing only ever increases the clearance; it is never shrunk back to the threshold
    assert geometry["solute_image_clearance_nm"] > 1.0
    assert geometry["box_width_nm"] > geometry["box_width_requested_nm"]


def test_the_conservative_2nm_option_gives_more_clearance_and_stays_selectable():
    from md_tools.openmm.defaults import CONSERVATIVE_PADDING_NM, DEFAULT_PADDING_NM

    assert (DEFAULT_PADDING_NM, CONSERVATIVE_PADDING_NM) == (1.5, 2.0)
    default = _geometry(padding_nm=DEFAULT_PADDING_NM, radius_nm=1.2)
    conservative = _geometry(padding_nm=CONSERVATIVE_PADDING_NM, radius_nm=1.2)
    assert conservative["solute_image_clearance_nm"] > default["solute_image_clearance_nm"]
    assert conservative["box_volume_nm3"] > default["box_volume_nm3"]


# --- what the generated project integrates -----------------------------------

@pytest.fixture(scope="module")
def default_project(tmp_path_factory):
    """`inputs/` + `MD/` at the DEFAULT combination, sized to run in seconds.

    Only the durations are shrunk. The force fields, water model, padding, cutoff, thermostat and
    barostat frequency are whatever `sys-config` writes with no arguments, which is the point.
    """
    work = tmp_path_factory.mktemp("defaults")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    assert run_cli("md_openmm", "sys-config", "--method", "cMD", "REST2",
                   cwd=work).returncode == 0

    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    path = work / "md.config.yaml"
    protocol = yaml.safe_load(path.read_text())
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.05
    protocol["REST2"].update({"number_of_replicas": 2, "equilibration_duration_ps": 0.05,
                              "exchange_interval_ps": 0.05, "number_of_exchanges": 2,
                              "tau_max": 0.05, "checkpoint_interval_ps": 0.05,
                              "whole_system_interval_ps": 0.05, "solute_interval_ps": 0.05})
    path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work


@pytest.mark.slow
def test_the_built_default_system_is_ff14sb_tip3p_pme_1nm(default_project):
    """The record and the serialized System must agree, and both must say ff14SB + TIP3P."""
    import json

    from openmm import NonbondedForce, XmlSerializer, unit

    record = json.loads((default_project / "inputs" / "forcefield.json").read_text())
    assert record["protein"]["openmm_resource"] == "amber14-all.xml"
    assert "amber14/protein.ff14SB.xml" in record["protein"]["openmm_resource_includes"]
    assert record["water"]["openmm_resource"] == "amber14/tip3p.xml"
    assert record["water"]["model"] == "TIP3P"
    assert record["explicit_solvent"]["box_shape"] == "dodecahedron"
    assert record["explicit_solvent"]["padding_nm"] == 1.5

    system = XmlSerializer.deserialize((default_project / "inputs" / "system.xml").read_text())
    nonbonded = next(system.getForce(i) for i in range(system.getNumForces())
                     if isinstance(system.getForce(i), NonbondedForce))
    assert record["nonbonded"]["method"] == "PME"
    assert nonbonded.getNonbondedMethod() == NonbondedForce.PME
    assert nonbonded.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(
        record["nonbonded"]["cutoff_nm"])
    assert nonbonded.getUseDispersionCorrection() == record["nonbonded"]["dispersion_correction"]
    assert nonbonded.getEwaldErrorTolerance() == pytest.approx(
        record["nonbonded"]["ewald_error_tolerance"])
    assert nonbonded.getUseSwitchingFunction() is record["nonbonded"]["switching"]

    # the built box must clear the gate the record claims it clears
    geometry = record["explicit_solvent"]["box_geometry"]
    assert geometry["min_reduced_box_height_nm"] >= geometry["required_cutoff_height_nm"]

    # sys-gen never puts a barostat in the System; that belongs to the stage that runs one
    assert not any("Barostat" in type(system.getForce(i)).__name__
                   for i in range(system.getNumForces()))


@pytest.mark.slow
def test_the_default_system_has_unmodified_hydrogen_masses(default_project):
    """2 fs without HMR is the baseline, and the baseline is checked on the masses."""
    from openmm import XmlSerializer, unit

    system = XmlSerializer.deserialize((default_project / "inputs" / "system.xml").read_text())
    solute = yaml.safe_load((default_project / "inputs" / "solute.yaml").read_text())
    masses = [system.getParticleMass(i).value_in_unit(unit.amu)
              for i in range(int(solute["n_solute_atoms"]))]
    assert max(m for m in masses if m < 4.0) < 1.1, "a hydrogen above 1.1 amu means HMR ran"

    protocol = yaml.safe_load((default_project / "MD" / "md.config.yaml").read_text())
    assert protocol["common"]["timestep_fs"] == 2.0
    assert protocol["common"]["friction_per_ps"] == 1.0


@pytest.mark.slow
def test_the_generated_project_records_the_thermostat_and_the_barostat_it_will_use(
        default_project):
    provenance = yaml.safe_load((default_project / "MD" / "provenance.yaml").read_text())
    protocol = provenance["protocol"]
    assert protocol["thermostat"]["integrator"] == "openmm.LangevinMiddleIntegrator"
    assert protocol["thermostat"]["friction_per_ps"] == 1.0
    assert protocol["timestep_fs"] == 2.0
    coupling = protocol["pressure_coupling"]
    assert coupling["barostat"] == "openmm.MonteCarloBarostat"
    assert coupling["frequency_steps"] == 25
    assert coupling["interval_ps"] == pytest.approx(0.05)
    assert coupling["active_in_stages"] == ["eq/npt_1kcal", "eq/npt_free"]
    assert coupling["present_but_inactive_in_stages"] == ["minimization", "eq/nvt_1kcal"]


@pytest.mark.gpu
@pytest.mark.slow
def test_the_configured_barostat_frequency_is_what_the_npt_stages_actually_run(default_project):
    """The public setting, traced to a barostat that attempted moves on a GPU.

    The whole common chain is run, because `barostats_active` is only meaningful once a Context
    exists: a Force property changed after the Context is built is invisible until it is
    reinitialised, and that is exactly the mistake this checks against.
    """
    project = default_project / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        result = run_stage(project / stage)
        assert result.returncode == 0, result.stdout + result.stderr

    records = {stage: yaml.safe_load((project / stage / "resolved_stage.yaml").read_text())
               for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free")}
    for stage, record in records.items():
        assert record["platform"] == "CUDA", stage
        # one barostat is present in every explicit stage so the Force layout, and therefore the
        # checkpoint layout, does not change along the chain
        assert record["barostats_in_system"] == 1, stage

    assert records["minimization"]["barostats_active"] == 0
    assert records["eq/nvt_1kcal"]["barostats_active"] == 0
    for stage in ("eq/npt_1kcal", "eq/npt_free"):
        assert records[stage]["barostats_active"] == 1, stage
        assert records[stage]["barostat_frequency_steps"] == 25, stage
        assert records[stage]["barostat_interval_ps"] == pytest.approx(0.05), stage


@pytest.mark.gpu
@pytest.mark.slow
def test_every_rest2_replica_runs_the_configured_barostat_frequency(default_project):
    """A per-replica barostat with its own seed, at the one configured interval."""
    project = default_project / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        run_stage(project / stage)

    equilibrated = run_stage(project / "REST2", script="equilibrate.py")
    assert equilibrated.returncode == 0, equilibrated.stdout + equilibrated.stderr
    produced = run_stage(project / "REST2")
    assert produced.returncode == 0, produced.stdout + produced.stderr

    record = yaml.safe_load((project / "REST2" / "resolved_run.yaml").read_text())
    assert record["pressure_coupling"]["frequency_steps"] == 25
    assert record["pressure_coupling"]["interval_ps"] == pytest.approx(0.05)
    assert record["integrator"]["friction_per_ps"] == 1.0
    assert record["integrator"]["kind"] == "LangevinMiddleIntegrator"
    # every replica reported exactly one active barostat while it propagated
    for replica in record["replicas"]:
        assert replica["barostats_active"] == 1, replica


# --- the alternative and the implicit route ----------------------------------

@pytest.mark.slow
def test_the_opc_alternative_still_builds_and_records_ff19sb_opc(tmp_path):
    """`--solvent OPC` must remain a working choice, not only a documented one."""
    import json

    shutil.copy2(ALA_PDB, tmp_path / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "--solvent", "OPC", cwd=tmp_path)
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=tmp_path)
    assert built.returncode == 0, built.stdout + built.stderr

    record = json.loads((tmp_path / "inputs" / "forcefield.json").read_text())
    assert record["protein"]["openmm_resource"] == "amber19-all.xml"
    assert "amber19/protein.ff19SB.xml" in record["protein"]["openmm_resource_includes"]
    assert record["water"]["openmm_resource"] == "amber19/opc.xml"
    assert record["water"]["model"] == "OPC"
    # OPC is 4-site and addSolvent cannot build a box for it; the stand-in is recorded, not hidden
    assert record["explicit_solvent"]["water_packing_model"] == "tip4pew"
    assert record["explicit_solvent"]["water_packing_substituted"] is True
    assert record["nonbonded"]["method"] == "PME"


@pytest.mark.slow
def test_the_implicit_peptide_route_is_ff14sb_gbn2_mbondi3_without_sasa_and_has_no_barostat(
        tmp_path):
    import json

    from openmm import XmlSerializer

    shutil.copy2(ALA_PDB, tmp_path / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "--solvent", "GBn2", cwd=tmp_path)
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=tmp_path)
    assert built.returncode == 0, built.stdout + built.stderr

    record = json.loads((tmp_path / "inputs" / "forcefield.json").read_text())
    implicit = record["implicit_solvent"]
    assert record["protein"]["tleap_resource"] == "leaprc.protein.ff14SB"
    assert record["protein"]["openmm_resource"] is None
    assert (implicit["model"], implicit["radii"]) == ("GBn2", "mbondi3")
    assert implicit["nonpolar_sasa"] is False
    assert implicit["support_status"] == "supported"
    assert implicit["amber_igb8_parity_claimed"] is True
    assert implicit["parameter_coverage"]["all_atoms_covered_by_gbn2_fit"] is True
    assert record["water"]["openmm_resource"] is None
    assert record["nonbonded"]["method"] == "NoCutoff"

    system = XmlSerializer.deserialize((tmp_path / "inputs" / "system.xml").read_text())
    assert not system.usesPeriodicBoundaryConditions()
    assert not any("Barostat" in type(system.getForce(i)).__name__
                   for i in range(system.getNumForces()))

    # and no NPT stage, so no barostat can appear downstream either
    protocol = yaml.safe_load((tmp_path / "md.config.yaml").read_text())
    assert protocol["common"]["pressure_bar"] is None
    assert protocol["common"]["barostat_frequency_steps"] is None
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=tmp_path)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    stages = yaml.safe_load(
        (tmp_path / "MD" / "md.config.yaml").read_text())["paths"]["common_stages"]
    assert stages == ["minimization", "eq/nvt_1kcal", "eq/nvt_free"]
    for stage in stages:
        entry = yaml.safe_load((tmp_path / "MD" / stage / "stage.yaml").read_text())
        assert entry["barostat_frequency_steps"] is None, stage
        assert entry["barostat_active"] is False, stage


@pytest.mark.slow
def test_an_implicit_ligand_outside_the_gbn2_element_fit_is_labelled_experimental(tmp_path):
    """GB-Neck2 has fitted parameters for H, C, N, O and S. A chlorine gets ParmEd's fallback.

    Nothing fails and nothing is missing -- every atom has a radius and a parameter set -- which is
    exactly why the shortfall has to be measured on the built force and published, rather than left
    for a reader to infer from a force-field name.
    """
    import json

    (tmp_path / "ligand.smi").write_text("Clc1ccc(cc1)C(=O)NC\n")
    run_cli("md_openmm", "sys-config", "--method", "cMD", "--peptide", "false",
            "--solvent", "GBn2", cwd=tmp_path)
    built = run_cli("md_openmm", "sys-gen", "-i", "./ligand.smi", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=tmp_path)
    assert built.returncode == 0, built.stdout + built.stderr

    record = json.loads((tmp_path / "inputs" / "forcefield.json").read_text())
    implicit = record["implicit_solvent"]
    coverage = implicit["parameter_coverage"]

    assert coverage["measured"] is True
    assert coverage["n_atoms_with_unfitted_gbn2_parameters"] >= 1
    assert 17 in coverage["unfitted_atomic_numbers"], "chlorine is outside the GB-Neck2 fit"
    assert coverage["all_atoms_covered_by_gbn2_fit"] is False
    # mbondi3's adjustments are keyed on GLU/ASP/ARG residue names and OXT; a UNL ligand has none
    assert coverage["mbondi3_reduces_to_mbondi2"] is True
    assert implicit["support_status"] == "experimental"
    assert implicit["amber_igb8_parity_claimed"] is False
    assert "not GB-Neck2" in implicit["support_note"]
    # the Sage parameters and the charge route are still named exactly
    assert record["ligand"]["openff_resource"] == "openff-2.2.1"
    assert record["ligand"]["charge_method"] == "am1bcc"


# --- the optional 4 fs performance setting -----------------------------------

@pytest.fixture(scope="module")
def hmr_project(tmp_path_factory):
    """The documented HMR option applied to the default combination, and generated."""
    from pathlib import Path

    work = tmp_path_factory.mktemp("hmr")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    run_cli("md_openmm", "sys-config", "--method", "cMD", cwd=work)

    example = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs/examples/hmr-4fs.yaml").read_text())
    system_path = work / "sys.config.yaml"
    document = yaml.safe_load(system_path.read_text())
    document["constraints"].update(example["sys.config.yaml"]["constraints"])
    system_path.write_text(yaml.safe_dump(document, sort_keys=False))

    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    protocol_path = work / "md.config.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["common"].update(example["md.config.yaml"]["common"])
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.04
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work


@pytest.mark.slow
def test_hmr_moves_mass_within_each_group_and_leaves_water_alone(hmr_project):
    """Conservation is checked where repartitioning happens, not only on the box total."""
    import json

    from openmm import XmlSerializer, unit

    system = XmlSerializer.deserialize((hmr_project / "inputs" / "system.xml").read_text())
    solute = int(yaml.safe_load(
        (hmr_project / "inputs" / "solute.yaml").read_text())["n_solute_atoms"])
    masses = [system.getParticleMass(i).value_in_unit(unit.amu)
              for i in range(system.getNumParticles())]

    assert pytest.approx(3.024, abs=1e-6) == max(m for m in masses[:solute] if m < 5.0)
    assert min(masses[:solute]) > 0.0, "no heavy atom may be driven to a nonpositive mass"
    # water is never repartitioned: its hydrogens are constrained and its rotation would change
    assert max(m for m in masses[solute:] if m < 5.0) < 1.1

    record = json.loads((hmr_project / "inputs" / "forcefield.json").read_text())
    constraints = record["constraints"]
    assert constraints["hydrogen_mass_amu"] == 3.024
    assert constraints["n_hydrogens_repartitioned"] > 0
    conservation = constraints["hmr_group_conservation"]
    assert conservation["max_group_mass_change_amu"] == pytest.approx(0.0, abs=1e-6)
    assert conservation["total_mass_amu_before"] == pytest.approx(
        conservation["total_mass_amu_after"], abs=1e-3)
    assert conservation["lightest_heavy_atom_amu"] > 0.0


@pytest.mark.slow
def test_the_hmr_project_generates_a_four_femtosecond_integrator(hmr_project):
    protocol = yaml.safe_load((hmr_project / "MD" / "md.config.yaml").read_text())
    assert protocol["common"]["timestep_fs"] == 4.0
    provenance = yaml.safe_load((hmr_project / "MD" / "provenance.yaml").read_text())["protocol"]
    assert provenance["timestep_fs"] == 4.0
    assert provenance["constraints"]["hydrogen_mass_repartitioning"] is True
    # 25 steps is a shorter time at 4 fs than at 2 fs, and the record says so in both units
    assert provenance["pressure_coupling"]["frequency_steps"] == 25
    assert provenance["pressure_coupling"]["interval_ps"] == pytest.approx(0.10)
    for stage in protocol["paths"]["common_stages"]:
        assert yaml.safe_load(
            (hmr_project / "MD" / stage / "stage.yaml").read_text())["timestep_fs"] == 4.0


@pytest.mark.gpu
@pytest.mark.slow
def test_the_four_femtosecond_chain_integrates_on_cuda(hmr_project):
    """A stability smoke run, and nothing more.

    Picoseconds of stable integration says the constraint/mass combination is not immediately
    divergent. It is NOT evidence that kinetics, diffusion or any other time-dependent observable
    from a 4 fs HMR run matches a 2 fs one -- see the rationale document.
    """
    project = hmr_project / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        result = run_stage(project / stage)
        assert result.returncode == 0, result.stdout + result.stderr
        record = yaml.safe_load((project / stage / "resolved_stage.yaml").read_text())
        assert record["platform"] == "CUDA"
        assert record["timestep_fs"] == 4.0
    npt = yaml.safe_load((project / "eq/npt_free" / "resolved_stage.yaml").read_text())
    assert npt["barostats_active"] == 1
    assert npt["barostat_interval_ps"] == pytest.approx(0.10)
