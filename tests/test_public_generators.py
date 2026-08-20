"""The two public generators: routing, refusals, and the staged output contract.

These test the RESPONSIBILITY BOUNDARY as much as the mechanics. The split only holds if
MD_system_gen.py cannot run dynamics and MD_input_gen.py cannot change the chemistry, so those are
asserted rather than assumed.
"""
from __future__ import annotations

import json
import os
import pathlib
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


def test_a_destination_holding_an_existing_bundle_is_refused(tmp_path):
    """What blocks a run is a file we would WRITE, not merely a non-empty directory.

    This used to refuse any non-empty destination. That was both too strict -- a directory holding
    someone's notes is not a reason to refuse -- and too vague, because the error could not say what
    would have been clobbered. The rule is now stated in terms of the files being written.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "system_manifest.json").write_text("{}")
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(dest), "--config", str(config))
    assert result.returncode != 0
    assert "already holds" in result.stderr
    assert "system_manifest.json" in result.stderr


def test_a_destination_holding_only_unrelated_files_is_written_to(tmp_path):
    """And the unrelated files survive, which is the promise the refusal above implies."""
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "already_here").write_text("x")
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(dest), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "already_here").read_text() == "x"
    assert (dest / "system_manifest.json").is_file()


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
    assert STAGE_ORDER == ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1")
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
    expected = {"min": "MD_system_gen.py", "eq_nvt": "min", "eq_npt_1": "eq_nvt",
                "eq_npt_2": "eq_npt_1", "cMD_1": "eq_npt_2", "REST2_1": "cMD_1"}
    for stage, producer in expected.items():
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert payload["input"]["produced_by"] == producer, stage
        assert payload["input"]["system_xml"].endswith("system.xml")
        assert payload["input"]["state"].endswith(".xml")


@pytest.mark.slow
def test_a_state_carries_between_stages_not_a_pdb(generated_project):
    """Positions alone would discard velocities and box vectors at every boundary."""
    for stage in ("eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1"):
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert payload["input"]["state"].endswith("_final_state.xml"), stage


@pytest.mark.slow
def test_the_restrained_stages_carry_the_documented_restraint(generated_project):
    for stage in ("min", "eq_nvt", "eq_npt_1"):
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
    npt = json.loads((generated_project / "eq_npt_1" / "eq_npt_1.json").read_text())
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
def test_generation_refuses_a_project_it_would_overwrite(prepared_system, generated_project,
                                                        tmp_path):
    """Refused because stage files would be rewritten -- and the error names them."""
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(generated_project), "--config", str(config))
    assert result.returncode != 0
    assert "already holds 15 file(s)" in result.stderr
    assert "REST2_1/REST2_1.json" in result.stderr    # the listing names the colliding files
    assert "--overwrite" in result.stderr

    ok = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
              "-o", str(generated_project), "--config", str(config), "--overwrite")
    assert ok.returncode == 0, ok.stdout + ok.stderr


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
def test_number_of_segments_drives_the_rest2_loop_and_nothing_else(generated_project):
    """The segment count is an execution choice, so it must move the loop and NOT the hashes.

    It was previously exported by `run_all.sh` and consumed by nothing, so asking for more segments
    silently ran one. The REST2 stage is now invoked once per segment -- re-invoking the same
    launcher is what continues the chain, because the runner reads its own committed-generation
    record to find where the last one stopped.
    """
    body = (generated_project / "run_all.sh").read_text()
    assert 'run_segments REST2_1 "$NUMBER_OF_SEGMENTS"' in body
    assert "run_stage REST2_1" not in body, "REST2 must not be a single invocation"
    for single_shot in ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1"):
        assert f"run_stage {single_shot}" in body, single_shot

    # and it must not appear anywhere in the scientific configuration
    payload = json.loads((generated_project / "REST2_1" / "REST2_1.json").read_text())
    assert "NUMBER_OF_SEGMENTS" not in json.dumps(payload["rest2"]["exchange"])
    assert "n_segments" not in payload["rest2"]["exchange"]


@pytest.mark.slow
def test_inherit_can_select_the_stage_to_continue_from(prepared_system, executed_project,
                                                       tmp_path):
    """`--inherit <manifest>:<stage>` continues from an endpoint another project already reached.

    Without the `:<stage>` selector every child project re-runs equilibration it has no reason to
    repeat. With it, the named stage supplies the endpoint and the stages before it are recorded as
    skipped rather than silently omitted -- the record is what makes the shortcut auditable.
    """
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    dest = tmp_path / "from_npt2"
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(dest), "--config", str(config),
                  "--inherit", f"{executed_project / 'run_manifest.json'}:eq_npt_2")
    assert result.returncode == 0, result.stdout + result.stderr

    lineage = json.loads((dest / "run_manifest.json").read_text())["lineage"]
    assert lineage["inherited_stage"] == "eq_npt_2"
    # the inherited stage is skipped TOO: we take its endpoint, so the child never re-runs it
    assert lineage["skipped_stages"] == ["min", "eq_nvt", "eq_npt_1", "eq_npt_2"], \
        "the stages the parent already ran must be named, not quietly dropped"
    # and cMD_1 must consume the PARENT's endpoint, not a local file that was never produced
    payload = json.loads((dest / "cMD_1" / "cMD_1.json").read_text())
    assert "eq_npt_2_final_state.xml" in payload["input"]["state"]


@pytest.mark.slow
def test_inherit_rejects_a_stage_that_is_not_in_the_protocol(prepared_system, executed_project,
                                                             tmp_path):
    """A typo in the stage name must fail loudly, not inherit from nothing."""
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(tmp_path / "bad"), "--config", str(config),
                  "--inherit", f"{executed_project / 'run_manifest.json'}:eq_npt")
    assert result.returncode != 0
    assert "eq_npt" in (result.stdout + result.stderr)


@pytest.mark.slow
def test_every_generated_stage_validates_without_a_gpu(generated_project):
    from md_templates.openmm.stage import validate_stage
    for stage in ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1"):
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

    for name in ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1"):
        payload = json.loads((out / name / f"{name}.json").read_text())
        rc = stage_mod.execute_stage(out / name / f"{name}.json", payload)
        assert rc == 0, name
    return out


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1"])
def test_each_executed_stage_writes_its_endpoint_artifacts(executed_project, stage):
    """A stage that ran owns its outputs; that is what makes it independently rerunnable."""
    d = executed_project / stage
    for name in (f"{stage}_final_state.xml", f"{stage}_final.pdb",
                 f"{stage}.chk", f"{stage}_results.json"):
        assert (d / name).is_file(), f"{stage}/{name}"


@pytest.mark.slow
def test_the_restraint_reaches_the_solute_and_only_the_equilibration_stages(executed_project):
    """22 restrained atoms is the alanine solute; production must be unrestrained."""
    for stage in ("min", "eq_nvt", "eq_npt_1"):
        results = json.loads((executed_project / stage / f"{stage}_results.json").read_text())
        assert results["n_restrained_atoms"] == 22, stage
    # eq_npt_2 is the FREE equilibration: same barostat, no restraint
    for stage in ("eq_npt_2", "cMD_1"):
        results = json.loads((executed_project / stage / f"{stage}_results.json").read_text())
        assert results["n_restrained_atoms"] == 0, stage


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
def test_only_the_npt_stages_carry_a_barostat(executed_project):
    """Whether a barostat was applied is a fact; whether the volume moved is a sampling outcome.

    An earlier version of this test asserted that each NPT stage changed the box relative to the one
    before it. That is not an invariant: a MonteCarloBarostat proposes a move every 25 steps and can
    reject all of them, and in a short CPU stage it sometimes does -- `eq_npt_2` came back bitwise
    identical to `eq_npt_1`. The same mistake as asserting that minimisation lowers the reported
    energy. What the protocol actually guarantees is which stages carry a barostat.
    """
    results = {s: json.loads((executed_project / s / f"{s}_results.json").read_text())
               for s in ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1")}

    for stage in ("min", "eq_nvt"):
        assert results[stage]["barostat"] is None, f"{stage} must not carry a barostat"
    for stage in ("eq_npt_1", "eq_npt_2", "cMD_1"):
        assert results[stage]["barostat"] == "MonteCarloBarostat", stage

    # NVT constancy IS exact: with no barostat nothing can change the box.
    assert results["min"]["box_volume_nm3"] == pytest.approx(
        results["eq_nvt"]["box_volume_nm3"], rel=1e-12), "NVT must not change the box volume"


@pytest.mark.slow
def test_the_barostat_moves_the_box_over_the_whole_npt_sequence(executed_project):
    """Aggregated over every NPT stage, so one unlucky stage cannot fail it."""
    volumes = {s: json.loads((executed_project / s / f"{s}_results.json").read_text())["box_volume_nm3"]
               for s in ("eq_nvt", "cMD_1")}
    assert volumes["cMD_1"] != pytest.approx(volumes["eq_nvt"], rel=1e-9), \
        "three NPT stages must between them change the box volume"


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
def test_rest2_hands_the_runner_a_bundle_it_accepts(executed_project):
    """REST2 is delegated, and the delegation is only real if the runner accepts what we build.

    The stage layer never touches the committed-generation record -- it assembles the version-2
    bundle the runner already understands and calls `launch_rest2`. Validating that bundle here is
    what stops the hand-off from being decorative: `validate_bundle` is the same check the runner
    performs, including the config_hash recomputation that catches a bundle assembled by copying a
    hash instead of deriving one.
    """
    from md_templates.openmm.bundle import validate_bundle
    from md_templates.openmm.stage import _package_bundle_for_rest2

    here = executed_project / "REST2_1"
    payload = json.loads((here / "REST2_1.json").read_text())
    bundle = _package_bundle_for_rest2(here, payload)
    manifest = validate_bundle(bundle)
    # `bundle_schema_version` is the BUNDLE contract; `schema_version` is the manifest document's
    # own version and is 1. They are different numbers and asserting the wrong one passes nothing.
    assert manifest["bundle_schema_version"] == 2
    assert (bundle / "experiment.prepare.yaml").is_file()
    for role in ("system_xml", "topology_pdb", "topology_cif", "simbox", "equilibrated_state",
                 "canonical_configuration", "resolved_runtime_config", "forcefield_provenance",
                 "environment"):
        assert role in manifest["roles"], role


@pytest.mark.slow
def test_rest2_starts_from_the_predecessors_endpoint_not_the_prepared_system(executed_project):
    """The one substantive choice in the bundle, and the one that would silently discard cMD."""
    from md_templates.openmm.stage import _package_bundle_for_rest2

    here = executed_project / "REST2_1"
    payload = json.loads((here / "REST2_1.json").read_text())
    bundle = _package_bundle_for_rest2(here, payload)

    equilibrated = (bundle / "equilibrated_state.xml").read_bytes()
    assert equilibrated == (executed_project / "cMD_1" / "cMD_1_final_state.xml").read_bytes(), \
        "REST2 must start where conventional MD finished"
    assert equilibrated != (executed_project / "inputs" / "initial_state.xml").read_bytes(), \
        "starting from the prepared system would discard every stage that ran"


@pytest.mark.slow
def test_rest2_refuses_to_start_without_its_predecessors_endpoint(executed_project, tmp_path):
    """Missing coordinates must be an error, not a silent fall back to the prepared system."""
    import shutil

    from md_templates.openmm import stage as stage_mod

    copy = tmp_path / "no_cmd"
    shutil.copytree(executed_project, copy)
    (copy / "cMD_1" / "cMD_1_final_state.xml").unlink()

    here = copy / "REST2_1"
    payload = json.loads((here / "REST2_1.json").read_text())
    with pytest.raises(stage_mod.StageError, match="does not exist"):
        stage_mod._package_bundle_for_rest2(here, payload)


# ---------------------------------------------------------------------------------------------
# declared outputs must be real
# ---------------------------------------------------------------------------------------------

def _dcd_frame_count(path) -> int:
    """Frame count from the DCD header, without a trajectory library."""
    import struct

    with open(path, "rb") as handle:
        handle.seek(8)
        return struct.unpack("<i", handle.read(4))[0]


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["eq_nvt", "cMD_1"])
def test_a_dynamic_stage_writes_every_output_it_declares(executed_project, stage):
    """The defect this replaces: intervals were recorded and no reporter was ever attached.

    A manifest describing a trajectory that does not exist is worse than a missing feature, because
    nothing downstream notices until someone goes looking for the frames.
    """
    outputs = json.loads((executed_project / stage / f"{stage}.json").read_text())["output"]
    for key in ("trajectory_all_atoms", "trajectory_selected_atoms", "log",
                "final_state", "final_structure", "checkpoint", "results"):
        path = executed_project / stage / outputs[key]
        assert path.is_file(), f"{stage} declares {key}={outputs[key]} and did not write it"
        assert path.stat().st_size > 0, f"{stage}/{outputs[key]} is empty"


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["eq_nvt", "cMD_1"])
def test_trajectories_hold_frames_at_the_declared_cadence(executed_project, stage):
    """Frame COUNT, not merely file existence: a reporter at the wrong interval still writes a file."""
    results = json.loads((executed_project / stage / f"{stage}_results.json").read_text())
    outputs = json.loads((executed_project / stage / f"{stage}.json").read_text())["output"]
    steps = results["steps"]

    for key, filename in (("all_atom", outputs["trajectory_all_atoms"]),
                          ("selected_atoms", outputs["trajectory_selected_atoms"])):
        attached = results["reporters"][key]
        frames = _dcd_frame_count(executed_project / stage / filename)
        assert frames == steps // attached["interval_steps"], (
            f"{stage}/{filename}: {frames} frames for {steps} steps at interval "
            f"{attached['interval_steps']}")


@pytest.mark.slow
def test_the_selected_trajectory_holds_exactly_the_resolved_selection(executed_project):
    """22 atoms is the alanine solute -- the same selection the restraint uses."""
    results = json.loads((executed_project / "cMD_1" / "cMD_1_results.json").read_text())
    assert results["selected_atoms"]["type"] == "solute"
    assert results["selected_atoms"]["n_atoms"] == 22
    assert results["reporters"]["selected_atoms"]["n_atoms"] == 22
    assert results["selected_atoms"]["indices_sha256"], "the atom ORDER must be fingerprinted"

    # the restrained stages restrain the same 22 atoms: one definition of "solute", not two
    restrained = json.loads(
        (executed_project / "eq_nvt" / "eq_nvt_results.json").read_text())["n_restrained_atoms"]
    assert restrained == results["selected_atoms"]["n_atoms"]


@pytest.mark.slow
def test_the_all_atom_trajectory_is_wrapped_and_the_selection_is_not(executed_project):
    """A scientific choice, so it is asserted rather than left to whoever edits the call.

    Wrapping teleports the solute across the box whenever its centre crosses a face, which breaks
    every analysis that reads the trajectory as continuous.
    """
    results = json.loads((executed_project / "cMD_1" / "cMD_1_results.json").read_text())
    assert results["reporters"]["all_atom"]["wrapped"] is True
    assert results["reporters"]["selected_atoms"]["wrapped"] is False


@pytest.mark.slow
def test_the_state_log_has_one_header_and_the_expected_rows(executed_project):
    results = json.loads((executed_project / "cMD_1" / "cMD_1_results.json").read_text())
    lines = (executed_project / "cMD_1" / "cMD_1.log").read_text().splitlines()
    headers = [line for line in lines if line.lstrip('#"').startswith(('Step', "Step"))
               or line.startswith('#"Step')]
    assert len(headers) == 1, f"expected exactly one header, got {len(headers)}"
    rows = len(lines) - 1
    assert rows == results["steps"] // results["reporters"]["state_log"]["interval_steps"]


@pytest.mark.slow
def test_minimisation_declares_no_trajectory_because_it_takes_no_steps(executed_project):
    """A step-interval reporter on a stage that never steps writes an empty, broken-looking file."""
    results = json.loads((executed_project / "min" / "min_results.json").read_text())
    assert results["reporters"] == {}
    outputs = json.loads((executed_project / "min" / "min.json").read_text())["output"]
    assert not (executed_project / "min" / outputs["trajectory_all_atoms"]).exists()


def test_a_reporting_interval_must_be_a_whole_number_of_steps():
    """Rounding would silently change the sampling frequency."""
    from md_templates.openmm.reporting import exact_steps

    assert exact_steps(250, what="x") == 250
    with pytest.raises(ValueError, match="whole number of steps"):
        exact_steps(250.5, what="x")
    with pytest.raises(ValueError, match="at least one step"):
        exact_steps(0, what="x")


def test_a_reporting_interval_longer_than_the_stage_is_refused():
    """It would produce an empty trajectory, indistinguishable from a broken one."""
    from md_templates.openmm.reporting import attach_reporters

    class _Sim:
        reporters: list = []

    with pytest.raises(ValueError, match="no frame would ever be written"):
        attach_reporters(_Sim(), all_atom_path=pathlib.Path("x.dcd"),
                         all_atom_interval_steps=5000, total_steps=100)


def test_the_stage_refuses_a_selection_that_does_not_match_the_bundle(executed_project, tmp_path):
    """A project pointed at a different bundle must fail, not write a mislabelled trajectory."""
    from md_templates.openmm import stage as stage_mod

    payload = json.loads((executed_project / "cMD_1" / "cMD_1.json").read_text())
    payload["reporting"]["selected_atoms_expected_count"] = 999
    with pytest.raises(stage_mod.StageError, match="disagree"):
        stage_mod.execute_stage(executed_project / "cMD_1" / "cMD_1.json", payload)


# ---------------------------------------------------------------------------------------------
# seeds and velocity continuity
# ---------------------------------------------------------------------------------------------

def test_no_seed_literal_survives_in_the_package():
    """A seed compiled into the source makes two scientifically different runs share a stream."""
    import md_templates.openmm as pkg

    offenders = []
    for path in pathlib.Path(pkg.__file__).parent.rglob("*.py"):
        if path.name == "seeds.py":
            continue                      # documents the literal it replaced
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "20260820" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"hard-coded seed literal still reachable: {offenders}"


def test_seed_derivation_is_deterministic_nonzero_and_distinct():
    from md_templates.openmm.seeds import derive_seed, seed_map, stage_purpose

    purposes = [stage_purpose(s, r) for s in ("min", "eq_nvt", "cMD_1")
                for r in ("integrator", "barostat", "velocity")]
    first = seed_map(4242, purposes)
    second = seed_map(4242, purposes)
    assert first == second, "the same master seed must reproduce the same map"
    assert len(set(first["seeds"].values())) == len(purposes), "seeds must be distinct"
    assert all(0 < v < 2 ** 31 for v in first["seeds"].values()), "must be nonzero and 32-bit safe"
    assert first["derivation_version"] >= 1 and first["derivation"]

    # neighbouring master seeds must not produce overlapping sets; `master + offset` did
    a = set(seed_map(1000, purposes)["seeds"].values())
    b = set(seed_map(1001, purposes)["seeds"].values())
    assert not (a & b)


def test_seed_derivation_does_not_depend_on_python_hash_randomisation():
    """PEP 456 randomises `hash()` per process, so a seed built on it differs run to run."""
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, %r);"
            "from md_templates.openmm.seeds import derive_seed;"
            "print(derive_seed(7, 'stage/eq_nvt/integrator'))" % str(REPO_ROOT / "src"))
    seen = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(salt)}).stdout.strip()
            for salt in (0, 1, 12345)}
    assert len(seen) == 1, f"seed changed with PYTHONHASHSEED: {seen}"


@pytest.mark.slow
def test_the_generated_project_carries_a_seed_map_not_a_literal(generated_project):
    from md_templates.openmm.seeds import derive_seed

    manifest = json.loads((generated_project / "run_manifest.json").read_text())
    randomness = manifest["randomness"]
    master = randomness["master_seed"]
    assert randomness["derivation"] and randomness["derivation_version"] >= 1
    assert randomness["source"] in ("md_config.randomness.master_seed", "default")

    # every recorded seed must be recomputable from the master seed -- a bare list of integers
    # cannot be checked, and an unverifiable seed record is not provenance
    for purpose, seed in randomness["seeds"].items():
        assert seed == derive_seed(master, purpose), purpose

    for stage in ("min", "eq_nvt", "cMD_1"):
        payload = json.loads((generated_project / stage / f"{stage}.json").read_text())
        assert set(payload["seeds"]) == {"integrator", "barostat", "velocity"}
        assert payload["seeds"]["integrator"] == derive_seed(master, f"stage/{stage}/integrator")


@pytest.mark.slow
def test_a_stage_refuses_to_invent_a_seed(executed_project, tmp_path):
    """If the configuration carries no seeds the stage must stop, not choose one."""
    from md_templates.openmm import stage as stage_mod

    payload = json.loads((executed_project / "cMD_1" / "cMD_1.json").read_text())
    payload.pop("seeds")
    path = tmp_path / "cMD_1.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(stage_mod.StageError, match="no seeds"):
        stage_mod.execute_stage(executed_project / "cMD_1" / "cMD_1.json", payload)


@pytest.mark.slow
def test_velocities_are_initialised_once_and_inherited_afterwards(executed_project):
    """The defect this replaces: every dynamic stage re-drew velocities from Maxwell-Boltzmann.

    Re-drawing discards the equilibration the previous stage just paid for and hides it behind a
    plausible-looking temperature, so nothing downstream looks wrong.
    """
    results = {s: json.loads((executed_project / s / f"{s}_results.json").read_text())
               for s in ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1")}

    assert results["min"]["velocities"] == "not required", "minimisation does not integrate"
    assert results["eq_nvt"]["velocities"] == "initialized", "the first dynamics stage draws them"
    assert results["eq_nvt"]["velocity_seed"] is not None, "and the draw must be seeded"
    for stage in ("eq_npt_1", "eq_npt_2", "cMD_1"):
        assert results[stage]["velocities"] == "inherited", stage
        assert results[stage]["velocity_seed"] is None, stage

    initialised = [s for s, r in results.items() if r["velocities"] == "initialized"]
    assert initialised == ["eq_nvt"], f"velocities must be created exactly once, got {initialised}"


@pytest.mark.slow
def test_minimisation_hands_on_a_state_without_velocities(executed_project):
    """The absence IS the handshake, and writing zeros instead would break it.

    Velocities created before minimisation are constraint-projected for the pre-minimisation
    geometry; minimisation then moves every atom, so the next stage integrates from velocities that
    violate the constraints. Measured on the alanine OPC box, that is an immediate NaN.
    """
    from openmm import XmlSerializer

    state = XmlSerializer.deserialize(
        (executed_project / "min" / "min_final_state.xml").read_text())
    with pytest.raises(Exception):
        state.getVelocities()

    nvt = XmlSerializer.deserialize(
        (executed_project / "eq_nvt" / "eq_nvt_final_state.xml").read_text())
    assert nvt.getVelocities() is not None, "a dynamics stage must pass velocities on"


# ---------------------------------------------------------------------------------------------
# honest input support
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("suffix,label", [(".mol", "MOL"), (".mol2", "MOL2"), (".sdf", "SDF")])
def test_recognised_but_unimplemented_formats_are_refused(tmp_path, suffix, label):
    """These used to fall through to the PDB reader, which cannot read them.

    Advertising a format the execution path does not implement is how a user discovers the gap
    after paying for a parameterisation, so the refusal must be immediate and say the word.
    """
    molecule = tmp_path / f"ligand{suffix}"
    molecule.write_text("not really a molecule\n")
    config = tmp_path / "c.json"
    config.write_text(json.dumps(SYSTEM_CONFIG))
    result = _run(SYSTEM_GEN, "-i", str(molecule), "-o", str(tmp_path / "out"),
                  "--config", str(config), "--dry-run")
    assert result.returncode != 0
    assert "not implemented yet" in result.stderr
    assert label in result.stderr
    assert ".smi" in result.stderr, "must name a route that does work"


def test_a_protein_ligand_complex_is_refused(tmp_path):
    """Declared, recognised, and not built -- so it must not be accepted."""
    config = tmp_path / "c.json"
    config.write_text(json.dumps(dict(SYSTEM_CONFIG, system={"type": "protein-ligand"})))
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(tmp_path / "out"),
                  "--config", str(config), "--dry-run")
    assert result.returncode != 0
    assert "not implemented yet" in result.stderr


def test_ligand_build_values_are_checked_not_only_their_keys(tmp_path):
    """A block naming a charge model the build will not use is false provenance.

    It is copied into the manifest as a record of how the molecule was built, so an unsupported or
    contradictory value is recorded as if it were true.
    """
    config = json.loads((REPO_ROOT / "test" / "rgd" / "REST2" / "system_config.json").read_text())
    config["ligand_build"]["charge_model"] = "gasteiger"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config))
    smi = REPO_ROOT / "test" / "rgd" / "REST2" / "cyclo_rgdfv.smi"

    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(tmp_path / "out"),
                  "--config", str(path), "--dry-run")
    assert result.returncode != 0
    assert "gasteiger" in result.stderr and "am1bcc" in result.stderr, result.stderr


def test_an_unsupported_ligand_build_policy_is_refused(tmp_path):
    config = json.loads((REPO_ROOT / "test" / "rgd" / "REST2" / "system_config.json").read_text())
    config["ligand_build"]["conformer_generation"] = "guess"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config))
    smi = REPO_ROOT / "test" / "rgd" / "REST2" / "cyclo_rgdfv.smi"

    result = _run(SYSTEM_GEN, "-i", str(smi), "-o", str(tmp_path / "out"),
                  "--config", str(path), "--dry-run")
    assert result.returncode != 0
    assert "conformer_generation" in result.stderr


def test_the_route_check_also_protects_callers_that_skip_the_cli(tmp_path):
    """The library must not depend on the entry point for a scientific-safety check."""
    from md_templates.openmm.system_prep import check_ligand_build_matches_the_route

    declared = {"ligand_build": {"parameterization_route": "openff-1.0.0"}}
    resolved = {"forcefield": {"ligand": "openff-2.2.0", "ligand_charge_method": "am1bcc"}}
    with pytest.raises(ValueError, match="openff-1.0.0"):
        check_ligand_build_matches_the_route(declared, resolved)


# ---------------------------------------------------------------------------------------------
# refusing to overwrite
# ---------------------------------------------------------------------------------------------

def test_the_bundle_target_list_cannot_drift_from_the_required_files():
    """`destination` lists what a bundle contains without importing the scientific stack.

    That duplication is deliberate -- the entry point must be able to check a destination in
    milliseconds -- so the containment is asserted here instead of being hoped for.
    """
    from md_templates.openmm.destination import SYSTEM_BUNDLE_TARGETS
    from md_templates.openmm.system_prep import REQUIRED_BUNDLE_FILES

    missing = set(REQUIRED_BUNDLE_FILES) - set(SYSTEM_BUNDLE_TARGETS)
    assert not missing, f"required bundle files absent from the overwrite check: {sorted(missing)}"


def test_a_destination_is_only_blocked_by_files_the_generator_would_write(tmp_path):
    """Unrelated files in the destination are not this generator's business."""
    from md_templates.openmm.destination import check_destination

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "NOTES.md").write_text("mine")
    result = check_destination(dest, ("system.xml", "topology.pdb"), overwrite=False,
                               what="system bundle")
    assert result["colliding"] == []
    assert result["collateral"] == ["NOTES.md"]


def test_a_colliding_target_stops_the_run_and_names_what_collides(tmp_path):
    from md_templates.openmm.destination import DestinationExists, check_destination

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "system.xml").write_text("<System/>")
    with pytest.raises(DestinationExists) as caught:
        check_destination(dest, ("system.xml", "topology.pdb"), overwrite=False,
                          what="system bundle")
    message = str(caught.value)
    assert "system.xml" in message
    assert "topology.pdb" not in message, "must name what collides, not the whole target list"
    assert "--overwrite" in message


def test_overwrite_allows_the_collision(tmp_path):
    from md_templates.openmm.destination import check_destination

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "system.xml").write_text("<System/>")
    result = check_destination(dest, ("system.xml",), overwrite=True, what="system bundle")
    assert result["colliding"] == ["system.xml"]


def test_the_warning_counts_results_inside_stage_directories(tmp_path):
    """The files most worth warning about live INSIDE directories the generator owns.

    `min/` is written by the generator; `min/min_final_state.xml` is a result of running it, and
    --overwrite deletes it with the rest of the directory. A top-level-only scan reports neither.
    """
    from md_templates.openmm.destination import DestinationExists, check_destination

    dest = tmp_path / "project"
    (dest / "min").mkdir(parents=True)
    (dest / "min" / "min.json").write_text("{}")
    (dest / "min" / "min_final_state.xml").write_text("<State/>")
    (dest / "min" / "min.chk").write_bytes(b"")
    (dest / "inputs").mkdir()
    (dest / "inputs" / "system.xml").write_text("<System/>")

    with pytest.raises(DestinationExists) as caught:
        check_destination(dest, ("inputs", "min/min.json", "min/min.sh"), overwrite=False,
                          what="project")
    message = str(caught.value)
    assert "2 file(s) it did not write" in message, message
    assert "min/" in message
    # a directory target is ours in its entirety, so its contents are not collateral
    assert "inputs/" not in message.split("would also delete")[-1]


def test_publish_keeps_unrelated_files_and_overwrite_does_not(tmp_path):
    """The publish step must match what the check promised, or the check is a lie."""
    from md_templates.openmm.destination import publish

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "NOTES.md").write_text("mine")

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "system.xml").write_text("<System/>")
    publish(staging, dest, overwrite=False, what="system bundle")
    assert (dest / "NOTES.md").is_file(), "a non-colliding publish must not destroy other files"
    assert (dest / "system.xml").is_file()

    staging2 = tmp_path / "staging2"
    staging2.mkdir()
    (staging2 / "system.xml").write_text("<System2/>")
    publish(staging2, dest, overwrite=True, what="system bundle")
    assert not (dest / "NOTES.md").exists(), "--overwrite replaces the whole directory, as warned"
    assert (dest / "system.xml").read_text() == "<System2/>"


def test_overwrite_generated_replaces_our_files_and_keeps_the_rest(tmp_path):
    """The whole point: a project can be regenerated without discarding what running it produced."""
    from md_templates.openmm.destination import OVERWRITE_GENERATED, publish

    dest = tmp_path / "project"
    (dest / "min").mkdir(parents=True)
    (dest / "min" / "min.json").write_text("OLD")
    (dest / "min" / "min_final_state.xml").write_text("<State/>")   # a result
    (dest / "run.log").write_text("history")

    staging = tmp_path / "staging"
    (staging / "min").mkdir(parents=True)
    (staging / "min" / "min.json").write_text("NEW")
    (staging / "run_all.sh").write_text("#!/bin/sh\n")

    publish(staging, dest, overwrite=OVERWRITE_GENERATED, what="project")

    assert (dest / "min" / "min.json").read_text() == "NEW", "generated files are rewritten"
    assert (dest / "min" / "min_final_state.xml").is_file(), "results survive"
    assert (dest / "run.log").read_text() == "history", "logs survive"
    assert (dest / "run_all.sh").is_file()
    assert not staging.exists()


def test_the_three_overwrite_modes_are_distinct(tmp_path):
    """`--overwrite` and `--overwrite-generated` must not quietly do the same thing."""
    from md_templates.openmm.destination import (OVERWRITE_ALL, OVERWRITE_GENERATED,
                                                 OVERWRITE_NONE, publish, resolve_mode)

    assert resolve_mode(True) is OVERWRITE_ALL     # the older boolean keeps its meaning
    assert resolve_mode(False) is OVERWRITE_NONE

    for mode, result_survives in ((OVERWRITE_ALL, False), (OVERWRITE_GENERATED, True)):
        dest = tmp_path / f"dest_{mode}"
        dest.mkdir()
        (dest / "result.xml").write_text("mine")
        staging = tmp_path / f"staging_{mode}"
        staging.mkdir()
        (staging / "system.xml").write_text("<System/>")
        publish(staging, dest, overwrite=mode, what="project")
        assert (dest / "result.xml").exists() is result_survives, mode
        assert (dest / "system.xml").is_file(), mode


def test_an_unknown_overwrite_mode_is_rejected():
    from md_templates.openmm.destination import resolve_mode

    with pytest.raises(ValueError, match="unknown overwrite mode"):
        resolve_mode("sometimes")


@pytest.mark.slow
def test_overwrite_generated_refuses_when_the_protocol_changed(prepared_system, generated_project,
                                                               tmp_path):
    """Keeping results is right when the GENERATOR changed, wrong when the protocol did.

    Without this the surviving results would sit beside stage files that no longer describe them,
    and nothing in the directory would say so.
    """
    config = tmp_path / "md.json"
    config.write_text(json.dumps(MD_CONFIG))
    same = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                "-o", str(generated_project), "--config", str(config), "--overwrite-generated")
    assert same.returncode == 0, same.stdout + same.stderr

    changed = json.loads(json.dumps(MD_CONFIG))
    changed["protocol"]["production"]["exchange"]["n_exchange_per_segment"] += 7
    other = tmp_path / "other.json"
    other.write_text(json.dumps(changed))
    result = _run(INPUT_GEN, "--system", str(prepared_system / "system_manifest.json"),
                  "-o", str(generated_project), "--config", str(other), "--overwrite-generated")
    assert result.returncode != 0
    assert "different protocol" in result.stderr
    assert "--inherit" in result.stderr, "must name the correct alternative"


def test_publish_reports_a_target_that_appeared_while_we_worked(tmp_path):
    """Distinguishable from a user error, because it is a race and not a mistake."""
    from md_templates.openmm.destination import publish

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "system.xml").write_text("someone else")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "system.xml").write_text("<System/>")
    with pytest.raises(RuntimeError, match="appeared while"):
        publish(staging, dest, overwrite=False, what="system bundle")


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


def test_the_launcher_records_the_interpreter_that_generated_the_project(tmp_path):
    """The one absolute path a launcher may carry, and why it must carry it.

    `python -m md_templates.openmm.stage` with a bare `python` is how this broke twice in practice:
    the stack's activation script puts AmberTools' interpreter first on PATH, and that one has
    neither openmm nor md_templates, so every stage died with ModuleNotFoundError. So the launcher
    records the interpreter it was GENERATED with -- the one known to import both.

    It is recorded as an overridable default (`: "${PYTHON:=...}"`), not hard-coded, and it is
    preflighted, so a project that moves to a machine without that interpreter fails with an
    instruction rather than a traceback.
    """
    from md_templates.openmm import input_gen

    interpreter, pythonpath = input_gen._interpreter_defaults()
    body = input_gen._stage_launcher("min", interpreter, pythonpath)

    assert f': "${{PYTHON:={interpreter}}}"' in body, "must be an overridable default"
    assert "import md_templates, openmm" in body, "must preflight before running the stage"
    assert "PYTHON=" in body and "cannot import" in body, "must say how to fix it"


def test_no_source_checkout_path_leaks_except_the_recorded_pythonpath(tmp_path):
    """A project that embedded the checkout anywhere ELSE would break the moment it moved."""
    from md_templates.openmm import input_gen

    body = input_gen._stage_launcher("min", Path("/usr/bin/python3"), None)
    assert str(REPO_ROOT) not in body
    assert "/data3" not in body


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
