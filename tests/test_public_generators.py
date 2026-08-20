"""The two public generators: routing, refusals, and the staged output contract.

These test the RESPONSIBILITY BOUNDARY as much as the mechanics. The split only holds if
MD_system_gen.py cannot run dynamics and MD_input_gen.py cannot change the chemistry, so those are
asserted rather than assumed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
INPUT_GEN = REPO_ROOT / "MD_input_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
               / "systems" / "ace_ala_nme.pdb")


def _run(script: Path, *args, cwd: Path | None = None):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run([sys.executable, str(script), *args], cwd=str(cwd or REPO_ROOT),
                          env=env, capture_output=True, text=True, timeout=1800)


SYSTEM_CONFIG = {
    "system": {"id": "ace_ala_nme", "type": "protein"},
    "forcefield": {"protein": "amber19/protein.ff19SB.xml", "water": "amber19/opc.xml",
                   "ligand": None, "ligand_charge_method": None},
    "solvation": {"water_model": "opc", "box_shape": "dodecahedron", "padding_nm": 1.2,
                  "ionic_strength_molar": 0.15},
    "system_build": {"nonbonded_cutoff_nm": 1.0, "hydrogen_mass_amu": 3.024,
                     "hmr_scope": "solute"},
}

MD_CONFIG = {
    "profile": "explicit-rest2-peptide-v1",
    "protocol": {
        "integrator": {"kind": "langevin-middle", "timestep": "4 fs",
                       "temperature": "300 K", "friction": "1 /ps"},
        "equilibration": {"protocol": "simple", "minimize_max_iterations": 1000,
                          "nvt": "10 ps", "npt": "10 ps", "npt_free": "10 ps"},
        "production": {"method": "rest2", "enhanced_region": {"type": "solute"},
                       "tau_ladder": {"minimum": 0.0, "maximum": 0.5, "count": 6,
                                      "interpolation": "linear"},
                       "exchange": {"n_exchange_per_segment": 1000,
                                    "exchange_interval": "5 ps"},
                       "omega_exclusion": {"enabled": True, "definition": "peptide_omega"}},
    },
    "execution": {"platform": "CUDA", "precision": "mixed",
                  "reporting": {"all_atom": "100 ps", "solute": "10 ps",
                                "state": "10 ps", "checkpoint": "5000 ps"}},
    "conventional_md": {"duration": "1 ns"},
    "minimization": {"restraint": {"force_constant_kcal_per_mol_angstrom2": 1.0}},
    "randomness": {"master_seed": 20260820},
}


# ---------------------------------------------------------------------------------------------
# both scripts exist, import safely, and help usefully
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", [SYSTEM_GEN, INPUT_GEN])
def test_the_public_scripts_exist_at_the_repository_root(script):
    assert script.is_file(), f"{script.name} must be a root-level public entry point"


@pytest.mark.parametrize("script", [SYSTEM_GEN, INPUT_GEN])
def test_help_works_without_heavy_imports(script):
    """--help must work even where OpenMM is unavailable: it is how a user discovers the tool."""
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "-o" in result.stdout and "--config" in result.stdout


def test_system_gen_help_states_it_runs_no_dynamics():
    result = _run(SYSTEM_GEN, "--help")
    assert "never minimises" in result.stdout or "no dynamics" in result.stdout.lower()


def test_input_gen_help_distinguishes_inherit_from_resume():
    """The distinction is the whole point of --inherit and must be discoverable."""
    result = _run(INPUT_GEN, "--help")
    assert "not a checkpoint" in result.stdout.lower()


# ---------------------------------------------------------------------------------------------
# input routing and refusals
# ---------------------------------------------------------------------------------------------

def test_an_unsupported_extension_is_refused(tmp_path):
    bogus = tmp_path / "thing.xyz"
    bogus.write_text("nope")
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(bogus), "-o", str(tmp_path / "out"),
                  "--config", str(config))
    assert result.returncode != 0
    assert "unsupported input extension" in result.stderr


def test_an_ambiguous_pdb_must_declare_its_system_type(tmp_path):
    """A PDB may be a peptide, a ligand, or a complex. Guessing is how a peptide becomes a ligand."""
    config = tmp_path / "c.json"
    config.write_text(json.dumps({k: v for k, v in SYSTEM_CONFIG.items() if k != "system"}))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(tmp_path / "out"),
                  "--config", str(config))
    assert result.returncode != 0
    assert "does not say what kind of system" in result.stderr


def test_a_declared_pdb_system_type_is_accepted(tmp_path):
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(tmp_path / "out"),
                  "--config", str(config), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "system type  : protein" in result.stdout


def test_smiles_requires_explicit_ligand_build_fields(tmp_path):
    smi = tmp_path / "lig.smi"
    smi.write_text("CCO\n")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"system": {"id": "eth", "type": "ligand"}}))
    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(tmp_path / "out"), "--config", str(config))
    assert result.returncode != 0
    for field in ("formal_charge", "stereochemistry_policy", "protonation_policy",
                  "conformer_generation", "charge_model", "parameterization_route"):
        assert field in result.stderr, field


def test_a_mol_or_smi_input_is_a_ligand_without_being_told(tmp_path):
    """Only the genuinely unambiguous cases are inferred."""
    smi = tmp_path / "lig.smi"
    smi.write_text("CCO\n")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({
        "ligand_build": {"formal_charge": 0, "stereochemistry_policy": "from_smiles",
                         "protonation_policy": "as_given", "conformer_generation": "etkdgv3",
                         "charge_model": "am1bcc", "parameterization_route": "openff-2.2.0"}}))
    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(tmp_path / "out"),
                  "--config", str(config), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "system type  : ligand" in result.stdout


def test_system_config_rejects_protocol_settings(tmp_path):
    """A protocol setting written here would never be applied; silence would be the worst outcome."""
    from md_templates.openmm import system_prep
    with pytest.raises(ValueError, match="contains protocol settings"):
        system_prep._runtime_cfg_from_system_config(
            dict(SYSTEM_CONFIG, production={"method": "rest2"}),
            Path("x.pdb"), "pdb", "protein")


def test_an_existing_nonempty_destination_is_refused(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "already_here").write_text("x")
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(dest), "--config", str(config))
    assert result.returncode != 0
    assert "not empty" in result.stderr


# ---------------------------------------------------------------------------------------------
# the prepared bundle, and the boundary it must not cross
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared_system(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("gen_system")
    config = work / "system_config.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    out = work / "ala_system"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    return out


@pytest.mark.slow
def test_the_bundle_carries_everything_a_protocol_needs(prepared_system):
    from md_templates.openmm.system_prep import REQUIRED_BUNDLE_FILES
    for name in REQUIRED_BUNDLE_FILES:
        assert (prepared_system / name).is_file(), name


@pytest.mark.slow
def test_system_xml_is_never_the_only_topology(prepared_system):
    """A serialized System has parameters but no atom names; it is not a topology."""
    assert (prepared_system / "topology.pdb").is_file()
    assert (prepared_system / "topology.cif").is_file()


@pytest.mark.slow
def test_the_bundle_stops_before_minimisation(prepared_system):
    manifest = json.loads((prepared_system / "system_manifest.json").read_text())
    assert "never been integrated" in manifest["prepared_through"]
    # an initial State has positions and box vectors but no velocities: nothing has moved
    state = (prepared_system / "initial_state.xml").read_text()
    assert "<Positions" in state
    assert "<Velocities" not in state


@pytest.mark.slow
def test_neutralising_ions_are_reported_separately_from_added_salt(prepared_system):
    salt = json.loads((prepared_system / "system_manifest.json").read_text())["salt"]
    assert "n_salt_pairs" in salt and "n_neutralizing_ions" in salt


@pytest.mark.slow
def test_every_checksum_matches(prepared_system):
    import hashlib
    checksums = json.loads((prepared_system / "checksums.json").read_text())["files"]
    for name, expected in checksums.items():
        actual = hashlib.sha256((prepared_system / name).read_bytes()).hexdigest()
        assert actual == expected, name


@pytest.mark.slow
def test_adapter_status_does_not_claim_amber_or_gromacs_work(prepared_system):
    status = json.loads((prepared_system / "system_manifest.json").read_text())["adapter_status"]
    assert status["openmm"] == "implemented"
    assert "not implemented" in status["amber"]
    assert "not implemented" in status["gromacs"]


# ---------------------------------------------------------------------------------------------
# the staged project
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_project(prepared_system, tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("gen_project")
    config = work / "md_config.json"
    config.write_text(json.dumps(MD_CONFIG))
    out = work / "ala_run"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    return out


@pytest.mark.slow
def test_the_stage_directories_are_named_as_documented(generated_project):
    from md_templates.openmm.input_gen import STAGE_ORDER
    assert STAGE_ORDER == ("min", "eq_nvt", "eq_npt", "cMD_1", "REST2_1")
    for stage in STAGE_ORDER:
        assert (generated_project / stage / f"{stage}.json").is_file(), stage
        assert (generated_project / stage / f"{stage}.sh").is_file(), stage


@pytest.mark.slow
def test_each_stage_owns_a_readable_launcher(generated_project):
    for stage in ("min", "REST2_1"):
        script = generated_project / stage / f"{stage}.sh"
        assert os.access(script, os.X_OK), f"{stage}.sh must be executable"
        body = script.read_text()
        assert "md_templates.openmm.stage" in body, "must call a package module"
        assert f"{stage}.json" in body, "must name its own configuration"


@pytest.mark.slow
def test_every_stage_names_its_input_and_its_predecessor(generated_project):
    """An implicit hand-off is how a stage silently starts from the wrong coordinates."""
    expected = {"min": "MD_system_gen.py", "eq_nvt": "min", "eq_npt": "eq_nvt",
                "cMD_1": "eq_npt", "REST2_1": "cMD_1"}
    for stage, producer in expected.items():
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert payload["input"]["produced_by"] == producer, stage
        assert payload["input"]["system_xml"].endswith("system.xml")
        assert payload["input"]["state"].endswith(".xml")


@pytest.mark.slow
def test_a_state_carries_between_stages_not_a_pdb(generated_project):
    """Positions alone would discard velocities and box vectors at every boundary."""
    for stage in ("eq_nvt", "eq_npt", "cMD_1", "REST2_1"):
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert payload["input"]["state"].endswith("_final_state.xml"), stage


@pytest.mark.slow
def test_the_restrained_stages_carry_the_documented_restraint(generated_project):
    for stage in ("min", "eq_nvt", "eq_npt"):
        restraint = json.loads((generated_project / stage / f"{stage}.json").read_text())["restraint"]
        assert restraint["force_constant_kcal_per_mol_angstrom2"] == 1.0, stage
        assert restraint["selection"]["type"] == "solute"
        # the reference is the PREPARED coordinates, so the restraint means the same thing
        # in every restrained stage rather than drifting with the previous one
        assert restraint["reference"] == "inputs/initial_state.xml", stage


@pytest.mark.slow
def test_production_stages_carry_no_restraint(generated_project):
    for stage in ("cMD_1", "REST2_1"):
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert payload["restraint"] is None, stage


@pytest.mark.slow
def test_the_stage_projections_are_distinct(generated_project):
    """min, NVT, NPT, cMD and REST2 must not collapse into one generic block."""
    npt = json.loads((generated_project / "eq_npt" / "eq_npt.json").read_text())
    rest2 = json.loads((generated_project / "REST2_1" / "REST2_1.json").read_text())
    nvt = json.loads((generated_project / "eq_nvt" / "eq_nvt.json").read_text())
    assert "barostat" in npt and npt["barostat"]["type"] == "MonteCarloBarostat"
    assert "barostat" not in nvt, "NVT must not carry a barostat"
    assert "rest2" in rest2 and "rest2" not in npt
    assert nvt["steps"] == 2_500 and npt["steps"] == 2_500      # 10 ps at 4 fs


@pytest.mark.slow
def test_the_rest2_stage_carries_tau_and_its_derived_scaling(generated_project):
    rest2 = json.loads((generated_project / "REST2_1" / "REST2_1.json").read_text())["rest2"]
    taus = rest2["tau_ladder"]["tau_values"]
    scale = rest2["derived_scale_factors"]
    assert len(taus) == 6
    assert taus == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], abs=1e-15)
    assert scale == pytest.approx([(1 - t) ** 2 for t in taus], abs=1e-15)
    assert rest2["omega_exclusion"]["enabled"] is True
    assert rest2["enhanced_region"]["type"] == "solute"


@pytest.mark.slow
def test_no_segment_count_appears_in_any_stage_json(generated_project):
    """Segment count is an execution choice; Bash owns it."""
    def keys_everywhere(node):
        """Every key name in the document, at any depth."""
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys_everywhere(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys_everywhere(item)

    for stage in ("cMD_1", "REST2_1"):
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        names = set(keys_everywhere(payload))
        # a FIELD carrying a segment count is the defect; prose explaining that Bash owns the
        # count is the intended behaviour, so the check is on key names, not raw text
        for retired in ("n_chunks", "number_of_segments", "n_segments", "segment_count"):
            assert retired not in names, f"{stage} has a {retired} field"


@pytest.mark.slow
def test_bash_owns_segment_repetition(generated_project):
    body = (generated_project / "run_all.sh").read_text()
    assert "NUMBER_OF_SEGMENTS" in body
    assert "EXECUTION choice" in body or "execution choice" in body.lower()


@pytest.mark.slow
def test_the_project_root_holds_only_a_driver_a_manifest_and_a_log(generated_project):
    from md_templates.openmm.input_gen import STAGE_ORDER
    entries = {p.name for p in generated_project.iterdir()}
    assert entries == {*STAGE_ORDER, "inputs", "run_all.sh", "run_manifest.json", "run.log"}


@pytest.mark.slow
def test_the_run_manifest_records_generation_not_execution(generated_project):
    manifest = json.loads((generated_project / "run_manifest.json").read_text())
    assert manifest["generated_not_executed"] is True
    assert manifest["completed_segments"] == 0
    assert manifest["system"]["n_solute_atoms"] == 22


@pytest.mark.slow
def test_the_generator_does_not_mutate_the_system_bundle(prepared_system, generated_project):
    """Generating a protocol must not touch the system it was generated from."""
    import hashlib
    checksums = json.loads((prepared_system / "checksums.json").read_text())["files"]
    for name, expected in checksums.items():
        actual = hashlib.sha256((prepared_system / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} changed during protocol generation"


@pytest.mark.slow
def test_md_config_may_not_restate_the_chemistry(prepared_system, tmp_path):
    from md_templates.openmm import input_gen
    manifest = json.loads((prepared_system / "system_manifest.json").read_text())
    manifest["_bundle_dir"] = str(prepared_system)
    with pytest.raises(ValueError, match="system-building settings"):
        input_gen._resolved_spec(dict(MD_CONFIG, solvation={"water_model": "tip3p"}), manifest)


@pytest.mark.slow
def test_a_tampered_bundle_is_refused(prepared_system, tmp_path):
    """A protocol generated against a corrupted bundle would run and produce numbers."""
    import shutil
    from md_templates.openmm import input_gen
    copy = tmp_path / "tampered"
    shutil.copytree(prepared_system, copy)
    (copy / "topology.pdb").write_text("corrupted")
    with pytest.raises(ValueError, match="does not match its own checksums"):
        input_gen._verify_bundle(copy / "system_manifest.json")


@pytest.mark.slow
def test_generation_refuses_an_existing_nonempty_project(prepared_system, generated_project,
                                                         tmp_path):
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(generated_project), "--config", str(config))
    assert result.returncode != 0
    assert "not empty" in result.stderr


@pytest.mark.slow
def test_a_failed_generation_leaves_no_partial_project(prepared_system, tmp_path):
    """Transactional: a half-written project would run without complaint and mean nothing."""
    config = tmp_path / "bad.json"
    bad = json.loads(json.dumps(MD_CONFIG))
    bad["protocol"]["production"]["exchange"]["exchange_interval"] = "6 fs"   # 1.5 steps at 4 fs
    config.write_text(json.dumps(bad))
    dest = tmp_path / "should_not_exist"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(dest), "--config", str(config))
    assert result.returncode != 0
    assert not dest.exists(), "a failed generation must leave nothing behind"


@pytest.mark.slow
def test_dry_run_writes_nothing(prepared_system, tmp_path):
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    dest = tmp_path / "dry"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(dest), "--config", str(config), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not dest.exists()


@pytest.mark.slow
def test_inherit_records_lineage_and_says_it_is_not_a_resume(prepared_system, generated_project,
                                                             tmp_path):
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    dest = tmp_path / "child"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(dest), "--config", str(config),
                  "--inherit", str(generated_project / "run_manifest.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    lineage = json.loads((dest / "run_manifest.json").read_text())["lineage"]
    assert lineage["inherited_from"].endswith("run_manifest.json")
    assert "NOT a checkpoint resume" in lineage["note"]


@pytest.mark.slow
def test_every_generated_stage_validates_without_a_gpu(generated_project):
    from md_templates.openmm.stage import validate_stage
    for stage in ("min", "eq_nvt", "eq_npt", "cMD_1", "REST2_1"):
        result = validate_stage(generated_project / stage / f"{stage}.json")
        assert result["stage"] == stage
        # only 'min' has all its inputs before anything runs; the rest declare a predecessor
        if stage == "min":
            assert result["problems"] == []
        else:
            assert all("does not exist yet" in p for p in result["problems"]), result["problems"]


# ---------------------------------------------------------------------------------------------
# execution
#
# Generating a layout proves nothing about whether it runs. These execute the single-shot stages
# on CPU and check the physics that the stage projection is supposed to carry.
# ---------------------------------------------------------------------------------------------

def _cpu_md_config() -> dict:
    config = json.loads(json.dumps(MD_CONFIG))
    config["execution"]["platform"] = "CPU"
    config["execution"].pop("precision", None)
    config["execution"]["reporting"] = {"all_atom": "0.2 ps", "solute": "0.2 ps",
                                        "state": "0.2 ps", "checkpoint": "0.4 ps"}
    config["protocol"]["equilibration"].update(
        {"minimize_max_iterations": 500, "nvt": "0.4 ps", "npt": "0.4 ps", "npt_free": "0.4 ps"})
    config["conventional_md"]["duration"] = "0.4 ps"
    config["protocol"]["production"]["exchange"] = {"n_exchange_per_segment": 2,
                                                    "exchange_interval": "0.2 ps"}
    return config


@pytest.fixture(scope="module")
def executed_project(prepared_system, tmp_path_factory) -> Path:
    """Generate a tiny CPU project and RUN its single-shot stages."""
    from md_templates.openmm import stage as stage_mod

    work = tmp_path_factory.mktemp("exec_project")
    config = work / "md_config.json"
    config.write_text(json.dumps(_cpu_md_config()))
    out = work / "run"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr

    for name in ("min", "eq_nvt", "eq_npt", "cMD_1"):
        payload = json.loads((out / name / f"{name}.json").read_text())
        rc = stage_mod.execute_stage(out / name / f"{name}.json", payload)
        assert rc == 0, name
    return out


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["min", "eq_nvt", "eq_npt", "cMD_1"])
def test_each_executed_stage_writes_its_endpoint_artifacts(executed_project, stage):
    """A stage that ran owns its outputs; that is what makes it independently rerunnable."""
    d = executed_project / stage
    for name in (f"{stage}_final_state.xml", f"{stage}_final.pdb",
                 f"{stage}.chk", f"{stage}_results.json"):
        assert (d / name).is_file(), f"{stage}/{name}"


@pytest.mark.slow
def test_the_restraint_reaches_the_solute_and_only_the_equilibration_stages(executed_project):
    """22 restrained atoms is the alanine solute; production must be unrestrained."""
    for stage in ("min", "eq_nvt", "eq_npt"):
        results = json.loads((executed_project / stage / f"{stage}_results.json").read_text())
        assert results["n_restrained_atoms"] == 22, stage
    cmd = json.loads((executed_project / "cMD_1" / "cMD_1_results.json").read_text())
    assert cmd["n_restrained_atoms"] == 0


@pytest.mark.slow
def test_minimisation_produces_a_usable_state(executed_project):
    """What minimisation must guarantee, and what it must NOT be assumed to guarantee.

    OpenMM's minimizeEnergy does not minimise the reported potential. It replaces constraints with
    stiff harmonic terms, minimises that surrogate, and restores the constraints -- so with HBonds
    constrained and few iterations the REPORTED energy can rise even though the minimisation is
    working correctly. Measured on this system: 50 iterations -36074 -> -33719 (up), 500 -> -36945
    (down), 5000 -> -37674 (down).

    So the invariant asserted here is that the endpoint is finite and usable, not that the number
    went down. The energy trend is checked at a sane iteration count instead.
    """
    results = json.loads((executed_project / "min" / "min_results.json").read_text())
    energy = results["potential_after_kj_mol"]
    assert energy == energy, "potential energy is NaN"          # NaN != NaN
    assert abs(energy) < 1e9, "potential energy diverged"
    assert results["box_volume_nm3"] > 0
    assert results["minimizer_max_iterations"] == 500


@pytest.mark.slow
def test_minimisation_lowers_the_energy_at_a_sane_iteration_count(executed_project):
    """At 500 iterations the surrogate and the reported potential agree in direction."""
    results = json.loads((executed_project / "min" / "min_results.json").read_text())
    assert results["potential_after_kj_mol"] < results["potential_before_kj_mol"]


@pytest.mark.slow
def test_only_the_npt_stages_change_the_box(executed_project):
    """The barostat must be on where the protocol says and nowhere else."""
    volumes = {s: json.loads((executed_project / s / f"{s}_results.json").read_text())["box_volume_nm3"]
               for s in ("min", "eq_nvt", "eq_npt")}
    assert volumes["min"] == pytest.approx(volumes["eq_nvt"], rel=1e-9), \
        "NVT must not change the box volume"
    assert volumes["eq_npt"] != pytest.approx(volumes["eq_nvt"], rel=1e-9), \
        "NPT must be able to change the box volume"


@pytest.mark.slow
def test_a_stage_consumes_its_predecessors_endpoint(executed_project):
    """Energies must be continuous across the hand-off, or velocities were dropped."""
    min_out = json.loads((executed_project / "min" / "min_results.json").read_text())
    nvt_in = json.loads((executed_project / "eq_nvt" / "eq_nvt_results.json").read_text())
    assert nvt_in["potential_before_kj_mol"] == pytest.approx(
        min_out["potential_after_kj_mol"], rel=1e-4), \
        "eq_nvt did not start from min's endpoint"


@pytest.mark.slow
def test_a_stage_refuses_to_run_before_its_predecessor(prepared_system, tmp_path):
    """Running out of order must fail loudly, not start from the wrong coordinates."""
    from md_templates.openmm import stage as stage_mod

    config = tmp_path / "md.json"
    config.write_text(json.dumps(_cpu_md_config()))
    out = tmp_path / "ooo"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stderr
    with pytest.raises(SystemExit, match="does not exist yet"):
        stage_mod.main(["--config", str(out / "eq_nvt" / "eq_nvt.json")])


@pytest.mark.slow
def test_rest2_execution_is_delegated_and_says_so(executed_project):
    """REST2 owns a restart boundary, so it is handed to the runner rather than re-plumbed."""
    from md_templates.openmm import stage as stage_mod
    payload = json.loads((executed_project / "REST2_1" / "REST2_1.json").read_text())
    with pytest.raises(SystemExit, match="competing restart authority"):
        stage_mod.execute_stage(executed_project / "REST2_1" / "REST2_1.json", payload)


# ---------------------------------------------------------------------------------------------
# relocation, and the RGDfV protocol
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_system_bundle_survives_relocation(prepared_system, tmp_path):
    """A bundle is only portable if it verifies somewhere else, with no source checkout in sight."""
    import shutil
    from md_templates.openmm import input_gen

    moved = tmp_path / "elsewhere" / "renamed_system"
    moved.parent.mkdir(parents=True)
    shutil.copytree(prepared_system, moved)
    # verification must pass from the NEW location: nothing may depend on the original path
    manifest = input_gen._verify_bundle(moved / "system_manifest.json")
    assert manifest["composition"]["n_solute_atoms"] == 22


@pytest.mark.slow
def test_a_relocated_bundle_still_generates_a_project(prepared_system, tmp_path):
    import shutil
    moved = tmp_path / "moved_system"
    shutil.copytree(prepared_system, moved)
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    out = tmp_path / "from_moved"
    result = _run(INPUT_GEN, "--system", str(moved / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "REST2_1" / "REST2_1.json").is_file()


def test_no_absolute_source_paths_leak_into_a_generated_project(tmp_path):
    """A project that embedded the checkout path would break the moment it moved."""
    from md_templates.openmm import input_gen
    # the stage launcher template must not bake in any absolute path
    body = input_gen._stage_launcher("min", Path("/somewhere"))
    assert "/data3" not in body
    assert str(REPO_ROOT) not in body


# --- RGDfV: dry generation on CPU, without paying 27 minutes of AM1-BCC ------------------------

RGDFV_SMILES = ("CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)"
                "[C@H](CCCNC(N)=[NH2+])NC1=O")


def test_rgdfv_system_generation_validates_on_cpu(tmp_path):
    """The SMILES route's routing and required build fields, without building."""
    smi = tmp_path / "cyclo_rgdfv.smi"
    smi.write_text(RGDFV_SMILES + "\n")
    config = tmp_path / "system_config.json"
    config.write_text((REPO_ROOT / "test" / "rgd" / "REST2" / "system_config.json").read_text())
    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(tmp_path / "rgd_system"),
                  "--config", str(config), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "system type  : ligand" in result.stdout
    assert "openff-2.2.0" in result.stdout and "am1bcc" in result.stdout


def test_the_rgdfv_smiles_shipped_with_the_example_is_the_vetted_one():
    """The example must not carry a SMILES that drifted from the vetted manifest."""
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load(
        (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
         / "cyclo_rgdfv.yaml").read_text())
    shipped = (REPO_ROOT / "test" / "rgd" / "REST2" / "cyclo_rgdfv.smi").read_text().strip()
    assert shipped == manifest["input"]["smiles"]


@pytest.mark.slow
def test_the_rgdfv_protocol_resolves_to_ten_replicas_on_cpu(prepared_system, tmp_path):
    """Dry generation of the RGDfV PROTOCOL.

    This uses the already-prepared alanine bundle rather than paying 27 minutes of AM1-BCC, because
    what is under test is that md_config.json resolves to TEN replicas. The RGDfV system route is
    covered separately by test_rgdfv_system_generation_validates_on_cpu.

    The pinned profile is dropped: profiles are route-bound, so `explicit-rest2-ligand-v1` (smiles)
    correctly refuses a pdb bundle. That refusal is desirable behaviour and is asserted below rather
    than worked around silently.
    """
    config = tmp_path / "rgd_md.json"
    rgd = json.loads((REPO_ROOT / "test" / "rgd" / "REST2" / "md_config.json").read_text())
    rgd.pop("profile", None)             # see the docstring: profiles are route-bound
    rgd["execution"]["platform"] = "CPU"
    rgd["execution"].pop("precision", None)
    config.write_text(json.dumps(rgd))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(tmp_path / "rgd_run"), "--config", str(config), "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "10 replicas" in result.stdout
    assert "1000 x 5 ps" in result.stdout


def test_both_shipped_md_configs_declare_their_replica_counts():
    for name, expected in (("ala", 6), ("rgd", 10)):
        config = json.loads(
            (REPO_ROOT / "test" / name / "REST2" / "md_config.json").read_text())
        assert config["protocol"]["production"]["tau_ladder"]["count"] == expected, name


@pytest.mark.slow
def test_a_ligand_profile_refuses_a_peptide_bundle(prepared_system, tmp_path):
    """Profiles are route-bound, and the mismatch must be refused rather than coerced.

    Discovered while writing the RGDfV dry-generation test: pinning the ligand REST2 profile
    against a pdb-route bundle fails, which is right -- a ligand profile carries small-molecule
    defaults that do not describe a peptide.
    """
    config = tmp_path / "mismatched.json"
    rgd = json.loads((REPO_ROOT / "test" / "rgd" / "REST2" / "md_config.json").read_text())
    assert rgd["profile"] == "explicit-rest2-ligand-v1"
    config.write_text(json.dumps(rgd))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(tmp_path / "nope"), "--config", str(config), "--dry-run")
    assert result.returncode != 0
    assert "is for route" in (result.stdout + result.stderr)


def test_the_system_front_end_sets_a_conformer_seed(tmp_path):
    """Regression: the SMILES route died on int(None) the first time it was actually run.

    The package DEFAULTS leave structure.etkdg.seed as None because the canonical pipeline fills it
    from the randomness block. This front end has no randomness block, so it must set one itself --
    a gap the dry-run tests could not reach, because they stop before anything is built.
    """
    from md_templates.openmm import system_prep
    smi = tmp_path / "x.smi"
    smi.write_text("CCO\n")
    cfg, smiles, pdb = system_prep._runtime_cfg_from_system_config(
        {"system": {"id": "x", "type": "ligand"}}, smi, "smi", "ligand")
    assert smiles == "CCO"
    assert isinstance(cfg["structure"]["etkdg"]["seed"], int)
    assert isinstance(cfg["run"]["seed"], int)


def test_the_conformer_seed_can_be_pinned(tmp_path):
    """Reproducing a specific conformer needs the seed to be an input, not a constant."""
    from md_templates.openmm import system_prep
    smi = tmp_path / "x.smi"
    smi.write_text("CCO\n")
    cfg, _, _ = system_prep._runtime_cfg_from_system_config(
        {"system": {"id": "x", "type": "ligand"}, "randomness": {"structure_seed": 4242}},
        smi, "smi", "ligand")
    assert cfg["structure"]["etkdg"]["seed"] == 4242
