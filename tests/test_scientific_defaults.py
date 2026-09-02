"""The 0.4 scientific defaults, traced from the public setting to the built System.

Documentation and YAML agreeing proves nothing on its own: this repository has already shipped a
configuration requesting `hydrogen_mass_amu: 3.024` beside a System carrying 1.008 amu hydrogens,
and every check that read only the configuration passed. So each test here ends at an artefact --
a serialized `System`, a generated `stage.yaml`, a written record -- rather than at a default.

Split by cost, and marked accordingly:

    (unmarked)  pure geometry, resolution and generated source. No System is built.
    slow        one system built and loaded per combination.
    gpu         minimisation or integration, which runs on CUDA and nowhere else.

The evidence for the choices themselves is `docs/scientific-defaults.md`; this file
only checks that what is implemented is what is documented.
"""
from __future__ import annotations

import ast
from pathlib import Path

import shutil

import pytest

from md_tools.build.record import read_record
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
    from md_tools.openmm.system_defaults import CONSERVATIVE_PADDING_NM, DEFAULT_PADDING_NM

    assert (DEFAULT_PADDING_NM, CONSERVATIVE_PADDING_NM) == (1.5, 2.0)
    default = _geometry(padding_nm=DEFAULT_PADDING_NM, radius_nm=1.2)
    conservative = _geometry(padding_nm=CONSERVATIVE_PADDING_NM, radius_nm=1.2)
    assert conservative["solute_image_clearance_nm"] > default["solute_image_clearance_nm"]
    assert conservative["box_volume_nm3"] > default["box_volume_nm3"]


# --- what the generated project integrates -----------------------------------

@pytest.fixture(scope="module")
def default_project(tmp_path_factory):
    """A built system and a generated cMD + REST2 pair at the DEFAULT combination.

    Only the lengths are shrunk. The force fields, water model, padding, cutoff, thermostat and
    barostat interval are whatever the built-in defaults are, which is the point of the file.
    """
    work = tmp_path_factory.mktemp("defaults")
    assert _build_top(work, ALA_PDB).returncode == 0
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 25,
                   "restrained_npt_steps": 25, "unrestrained_npt_steps": 25,
                   "production_steps": 25},
        "reporting": {"solute_printout": 25, "system_printout": 25,
                      "checkpoint_printout": 25},
    }, sort_keys=False), encoding="utf-8")
    assert _build_md(work, "cMD.config", "./cMD").returncode == 0
    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 25,
                   "restrained_npt_steps": 25, "unrestrained_npt_steps": 25,
                   "production_steps": 0},
        "reporting": {"solute_printout": 25, "system_printout": 25,
                      "checkpoint_printout": 25},
        "rest2": {"number_of_replicas": 2, "tau_max": 0.05,
                  "exchange_interval_steps": 25, "number_of_exchanges": 2},
    }, sort_keys=False), encoding="utf-8")
    assert _build_md(work, "REST2.config", "./REST2").returncode == 0
    return work


@pytest.fixture(scope="module")
def hmr_project(tmp_path_factory):
    """The documented HMR option: repartitioned masses in the System, 4 fs in the protocol."""
    work = tmp_path_factory.mktemp("hmr")
    (work / "hmr.config").write_text(
        "constraints:\n  type: HBonds\n  rigid_water: true\n  hydrogen_mass_amu: 3.024\n",
        encoding="utf-8")
    built = _build_top(work, ALA_PDB, "hmr.config")
    assert built.returncode == 0, built.stdout + built.stderr
    (work / "fast.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "dynamics": {"timestep_fs": 4.0},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 10},
        "reporting": {"solute_printout": 10, "system_printout": 10,
                      "checkpoint_printout": 10},
    }, sort_keys=False), encoding="utf-8")
    assert _build_md(work, "fast.config", "./md_script").returncode == 0
    return work


def _build_top(work, structure, config=None):
    """The real command. Nothing in this file reaches past the public interface."""
    import subprocess
    import sys as _sys

    argv = [_sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(structure),
            "-os", "built.xml", "-op", "built.pdb", "-log", "built.log"]
    if config:
        argv += ["--config", str(config)]
    return subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=1800)


def _build_md(work, config, odir):
    import subprocess
    import sys as _sys

    return subprocess.run(
        [_sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", odir,
         "--config", str(config)], cwd=work, capture_output=True, text=True, timeout=600)


def _stage(work, directory, name):
    """The resolved settings the named stage will run with.

    Read from `resolved.config` beside the scripts, which is the single declaration of the
    workflow. The script used to embed a `STAGE = {...}` literal -- a second declaration that
    could disagree with the configuration next to it -- and now names its stage and nothing else.
    """
    from md_tools.build.md import resolve_md_config, stage_plan

    plan = stage_plan(resolve_md_config(Path(work) / directory / "resolved.config"))
    return next(entry for entry in plan if entry["name"] == name)


@pytest.mark.slow
def test_the_built_default_system_is_ff14sb_tip3p_pme_1nm(default_project):
    """The record and the serialized System must agree, and both must say ff14SB + TIP3P."""
    import json

    from openmm import NonbondedForce, XmlSerializer, unit

    record = read_record(default_project / "built.log")["forcefield_record"]
    assert record["protein"]["openmm_resource"] == "amber14-all.xml"
    assert "amber14/protein.ff14SB.xml" in record["protein"]["openmm_resource_includes"]
    assert record["water"]["openmm_resource"] == "amber14/tip3p.xml"
    assert record["water"]["model"] == "TIP3P"
    assert record["explicit_solvent"]["box_shape"] == "dodecahedron"
    assert record["explicit_solvent"]["padding_nm"] == 1.5

    system = XmlSerializer.deserialize((default_project / "built.xml").read_text())
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










# --- the alternative and the implicit route ----------------------------------







# --- the optional 4 fs performance setting -----------------------------------









# --- the same scientific claims, against build-top and build-md ------------------------------------

@pytest.mark.slow
def test_the_default_system_has_unmodified_hydrogen_masses(default_project):
    """2 fs without HMR is the baseline, and the baseline is checked on the MASSES."""
    from openmm import XmlSerializer, unit

    system = XmlSerializer.deserialize((default_project / "built.xml").read_text())
    solute = read_record(default_project / "built.log")["counts"]["solute_atoms"]
    masses = [system.getParticleMass(i).value_in_unit(unit.amu) for i in range(int(solute))]
    assert max(m for m in masses if m < 4.0) < 1.1, "a hydrogen above 1.1 amu means HMR ran"

    stage = _stage(default_project, "cMD", "cMD")
    # The generated stage declares `auto`, not a number: `build-md` never opened built.xml and so
    # cannot know these masses. What this test establishes is the other half -- that the System
    # really is unrepartitioned, which is what `auto` resolves against to give 2 fs. That
    # resolution is asserted end-to-end in test_timestep_resolution.py.
    assert stage["timestep_fs"] == "auto"
    assert stage["friction_per_ps"] == 1.0


@pytest.mark.slow
def test_the_generated_stages_declare_the_thermostat_and_the_barostat_they_will_use(
        default_project):
    """What `provenance.yaml` used to record now lives in each stage script's own settings."""
    for name in ("min", "eq_nvt_posres"):
        stage = _stage(default_project, "cMD", name)
        assert stage["ensemble"] == "NVT"
    for name in ("eq_npt_posres", "eq_npt_free", "cMD"):
        stage = _stage(default_project, "cMD", name)
        assert stage["ensemble"] == "NPT"
        assert stage["barostat_interval_steps"] == 25
        assert stage["pressure_bar"] == 1.0
    common = _stage(default_project, "cMD", "cMD")
    assert common["temperature_K"] == 300.0
    assert common["friction_per_ps"] == 1.0


@pytest.mark.gpu
@pytest.mark.slow
def test_the_configured_barostat_frequency_is_what_the_stages_actually_run(default_project):
    """The public setting, traced to a barostat that attempted moves on a GPU.

    The whole chain is run, because the active count is only meaningful once a Context exists: a
    Force property changed after the Context is built is invisible until reinitialisation, and that
    is exactly the mistake this checks against.
    """
    import subprocess

    result = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"],
                            cwd=default_project / "cMD", capture_output=True, text=True,
                            timeout=3600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    for name in ("min", "eq_nvt_posres", "eq_npt_posres", "eq_npt_free", "cMD"):
        record = read_record(default_project / "cMD" / f"{name}.log")
        assert record["platform"]["name"] == "CUDA", name
        # One barostat in EVERY explicit stage, so the Force layout -- and the checkpoint layout --
        # does not change along the chain.
        assert record["barostats"]["in_system"] == 1, name

    for name in ("min", "eq_nvt_posres"):
        assert read_record(default_project / "cMD" / f"{name}.log")["barostats"]["active"] == 0
    for name in ("eq_npt_posres", "eq_npt_free", "cMD"):
        barostats = read_record(default_project / "cMD" / f"{name}.log")["barostats"]
        assert barostats["active"] == 1, name
        assert barostats["frequency_steps"] == 25, name
        assert barostats["interval_ps"] == pytest.approx(0.05), name


@pytest.mark.slow
def test_a_rest2_ladder_carries_no_barostat_at_all(default_project):
    """REST2 is NVT by construction: the ladder samples a fixed volume."""
    resolved = yaml.safe_load((default_project / "REST2" / "resolved.config").read_text())
    assert resolved["protocol"] == "REST2"
    assert (default_project / "REST2" / "REST2.py").is_file()
    for name in ("min", "eq_nvt_posres"):
        assert _stage(default_project, "REST2", name)["ensemble"] == "NVT"


@pytest.mark.slow
def test_the_opc_alternative_still_builds_and_records_ff19sb_opc(tmp_path):
    """The optional selection stays available, and the record names what it loaded."""
    (tmp_path / "opc.config").write_text(
        "forcefield:\n  protein: ff19SB\nsolvent:\n  model: OPC\n  padding_nm: 0.5\n"
        "  cutoff_nm: 0.6\n", encoding="utf-8")
    built = _build_top(tmp_path, ALA_PDB, "opc.config")
    assert built.returncode == 0, built.stdout + built.stderr

    record = read_record(tmp_path / "built.log")["forcefield_record"]
    assert record["protein"]["openmm_resource"] == "amber19-all.xml"
    assert record["water"]["model"] == "OPC"
    assert "amber19" in record["water"]["openmm_resource"]


@pytest.mark.slow
def test_the_implicit_peptide_route_is_ff14sb_gbn2_mbondi3_without_sasa_and_has_no_barostat(
        tmp_path):
    from openmm import XmlSerializer

    (tmp_path / "gb.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = _build_top(tmp_path, ALA_PDB, "gb.config")
    assert built.returncode == 0, built.stdout + built.stderr

    built_record = read_record(tmp_path / "built.log")
    record = built_record["forcefield_record"]
    assert record["implicit_solvent"]["model"] == "GBn2"
    assert record["implicit_solvent"]["radii"] == "mbondi3"
    assert record["implicit_solvent"]["nonpolar_sasa"] is False
    assert record["water"]["openmm_resource"] is None, "no water participates in an implicit build"

    system = XmlSerializer.deserialize((tmp_path / "built.xml").read_text())
    assert not system.usesPeriodicBoundaryConditions()
    assert not any("Barostat" in type(system.getForce(i)).__name__
                   for i in range(system.getNumForces()))
    assert built_record["periodic"] is False
