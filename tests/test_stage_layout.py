"""The generated directory contract: what is written, what reads what, and what refuses.

These tests execute the generated stages. A tree with the right filenames and a configuration that
reads well proves nothing about which state a stage actually started from.
"""
from __future__ import annotations

import json
import re
import stat

import pytest
import yaml

from .conftest import REPO_ROOT, run_stage, tiny_project

# Every test here minimises or integrates, so every test here runs on CUDA.
pytestmark = [pytest.mark.slow, pytest.mark.gpu]

EXPLICIT_STAGES = ["minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"]
IMPLICIT_STAGES = ["minimization", "eq/nvt_1kcal", "eq/nvt_free"]
STAGE_FILES = {"stage.log", "stage.csv", "checkpoint.chk", "final_state.xml", "final.pdb",
               "resolved_stage.yaml"}


@pytest.fixture(scope="module")
def explicit(tmp_path_factory):
    return tiny_project(tmp_path_factory.mktemp("explicit"), solvent="OPC")


@pytest.fixture(scope="module")
def implicit(tmp_path_factory):
    return tiny_project(tmp_path_factory.mktemp("implicit"), solvent="GBn2")


@pytest.fixture(scope="module")
def explicit_run(explicit):
    """One completed explicit workflow: every common stage, then both production branches."""
    for name in EXPLICIT_STAGES:
        result = run_stage(explicit / name)
        assert result.returncode == 0, f"{name}:\n{result.stdout[-2500:]}{result.stderr[-2500:]}"
    for directory, script in ((explicit / "cMD", "run.py"),
                              (explicit / "REST2", "equilibrate.py"),
                              (explicit / "REST2", "run.py")):
        result = run_stage(directory, script)
        assert result.returncode == 0, \
            f"{directory.name}/{script}:\n{result.stdout[-2500:]}{result.stderr[-2500:]}"
    return explicit


# --- 1. the exact trees ------------------------------------------------------

def test_the_explicit_tree_is_the_documented_one(explicit):
    top = {p.name for p in explicit.iterdir() if p.is_dir()}
    assert top == {"minimization", "eq", "cMD", "REST2"}, top
    assert {p.name for p in (explicit / "eq").iterdir() if p.is_dir()} == {
        "nvt_1kcal", "npt_1kcal", "npt_free"}
    for name in EXPLICIT_STAGES:
        assert {p.name for p in (explicit / name).iterdir()} >= {"run.py", "run.sh", "stage.yaml"}
    assert (explicit / "run_all.sh").is_file()
    # cMD carries the scaling module because it may run at tau > 0: a fixed-tau single walker on
    # one rung of the ladder. It is the SAME file REST2 gets, so the walker cannot drift from the
    # rung it is meant to match.
    assert {p.name for p in (explicit / "cMD").iterdir()} == {
        "run.py", "run.sh", "rest2_scaling.py"}
    assert {p.name for p in (explicit / "REST2").iterdir()} >= {
        "equilibrate.py", "equilibrate.sh", "run.py", "run.sh", "extend.sh", "rest2_scaling.py",
        "replica_00", "replica_01"}
    for replica in ("replica_00", "replica_01"):
        assert {p.name for p in (explicit / "REST2" / replica).iterdir()} == {
            "equilibration", "production"}


def test_the_implicit_tree_has_no_npt_stage(implicit):
    top = {p.name for p in implicit.iterdir() if p.is_dir()}
    assert top == {"minimization", "eq", "cMD", "REST2"}, top
    grouped = {p.name for p in (implicit / "eq").iterdir() if p.is_dir()}
    assert grouped == {"nvt_1kcal", "nvt_free"}, grouped
    assert not any("npt" in name for name in grouped), "implicit solvent has no box to control"


def test_the_folder_name_does_not_claim_1kcal_when_the_restraint_is_not_1(tmp_path):
    """A directory called eq1_nvt_1kcal holding a 5 kcal/mol/A^2 run is a lie told by a filename."""
    def stronger(protocol):
        protocol["equilibration"]["restraint_k_kcal_mol_a2"] = 5.0

    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",), edit=stronger)
    names = {p.name for p in (project / "eq").iterdir() if p.is_dir()}
    assert "nvt_restrained" in names, names
    assert not any("1kcal" in name for name in names), names


# --- 2. the dependency chain -------------------------------------------------

@pytest.mark.parametrize("stages,label", [(EXPLICIT_STAGES, "explicit"),
                                          (IMPLICIT_STAGES, "implicit")])
def test_each_stage_reads_only_its_parents_final_state(stages, label, explicit, implicit):
    project = explicit if label == "explicit" else implicit
    for index, name in enumerate(stages):
        stage = yaml.safe_load((project / name / "stage.yaml").read_text())
        if index == 0:
            assert stage["parent"] is None
            assert stage["input_state"].endswith("inputs/initial_state.xml"), stage["input_state"]
        else:
            import posixpath

            assert stage["parent_path"] == stages[index - 1]
            # The path must resolve, from this stage's OWN directory, to the parent's final state.
            landed = posixpath.normpath(posixpath.join(name, stage["input_state"]))
            assert landed == f"{stages[index - 1]}/final_state.xml", (name, stage["input_state"])
            assert (project / landed).parent.is_dir()
        assert "checkpoint" not in stage["input_state"], \
            "a stage must never take its parent's mid-run checkpoint as input"


def test_cmd_and_rest2_branch_from_the_same_finalized_state(explicit):
    config = yaml.safe_load((explicit / "md.config.yaml").read_text())
    assert config["paths"]["common_final_stage"] == EXPLICIT_STAGES[-1]
    assert config["paths"]["common_final_state"] == f"../{EXPLICIT_STAGES[-1]}/final_state.xml"
    # Both scripts read the same key, so neither can drift onto a different parent.
    for script in ((explicit / "cMD" / "run.py"), (explicit / "REST2" / "equilibrate.py")):
        assert 'CONFIG["paths"]["common_final_state"]' in script.read_text()


def test_a_missing_parent_state_names_the_file_and_the_command(tmp_path):
    """A stage must refuse to invent a starting point rather than run from nothing.

    Its own project, so this does not depend on whether another test has already run the chain.
    """
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))

    result = run_stage(project / "cMD")                    # nothing has run yet
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "cMD started without its parent state"
    assert "final_state.xml" in combined and "npt_free" in combined, combined[-800:]
    assert "./run.sh" in combined, combined[-800:]

    # and the same for a common stage whose parent has not run
    result = run_stage(project / "eq/npt_1kcal")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "nvt_1kcal" in combined and "final_state.xml" in combined, combined[-800:]


# --- 5/6. runtime force state, stage by stage --------------------------------

EXPECTED_EXPLICIT = {
    "minimization":  {"restrained": True,  "active": 0},
    "eq/nvt_1kcal":  {"restrained": True,  "active": 0},
    "eq/npt_1kcal":  {"restrained": True,  "active": 1},
    "eq/npt_free":   {"restrained": False, "active": 1},
}


def test_every_explicit_stage_has_the_required_restraint_and_barostat_state(explicit_run):
    for name, expected in EXPECTED_EXPLICIT.items():
        record = yaml.safe_load((explicit_run / name / "resolved_stage.yaml").read_text())
        strength = record["restraint_kj_mol_nm2"]
        assert (strength > 0) is expected["restrained"], (name, strength)
        assert record["barostats_active"] == expected["active"], (name, record)
        assert record["barostats_in_system"] == 1, "explicit systems keep one barostat throughout"


def test_the_implicit_chain_never_has_a_barostat(implicit):
    for name in IMPLICIT_STAGES:
        result = run_stage(implicit / name)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        record = yaml.safe_load((implicit / name / "resolved_stage.yaml").read_text())
        assert record["barostats_in_system"] == 0, name
        assert record["barostats_active"] == 0, name
        assert record["pressure_bar"] is None, name
    free = yaml.safe_load((implicit / "eq/nvt_free" / "resolved_stage.yaml").read_text())
    assert free["restraint_kj_mol_nm2"] == 0.0
    assert free["ensemble"] == "NVT"


# --- 3. the files a stage writes ---------------------------------------------

def test_every_common_stage_writes_the_complete_artifact_set(explicit_run):
    """Minimisation included: a stage missing half its files cannot be inspected like its peers."""
    for name in EXPLICIT_STAGES:
        present = {p.name for p in (explicit_run / name).iterdir()}
        assert STAGE_FILES <= present, (name, sorted(STAGE_FILES - present))


def test_resolved_stage_records_what_actually_ran(explicit_run):
    record = yaml.safe_load((explicit_run / "eq/npt_1kcal" / "resolved_stage.yaml").read_text())
    for key in ("kind", "ensemble", "duration_ps", "restraint_k_kcal_mol_a2",
                "temperature_kelvin", "pressure_bar", "timestep_fs", "integrator_seed",
                "input_state", "output_state", "template_commit"):
        assert key in record, key
    assert record["pressure_bar"] == 1.0
    minimization = yaml.safe_load(
        (explicit_run / "minimization" / "resolved_stage.yaml").read_text())
    assert minimization["max_iterations"] and minimization["duration_ps"] is None
    assert minimization["pressure_bar"] is None, "nothing controls pressure during minimisation"


# --- 8/9. REST2 replicas -----------------------------------------------------

def test_each_replica_equilibrates_and_produces_in_its_own_directory(explicit_run):
    for replica in range(2):
        base = explicit_run / "REST2" / f"replica_{replica:02d}"
        assert (base / "equilibration" / "final_state.xml").is_file()
        assert (base / "equilibration" / "resolved_stage.yaml").is_file()
        for name in ("production.chk", "final_state.xml", "replica.csv", "whole_system.dcd",
                     "solute.dcd"):
            assert (base / "production" / name).is_file(), (replica, name)


def test_replica_equilibration_uses_its_own_tau_and_distinct_seeds(explicit_run):
    seeds, taus = [], []
    for replica in range(2):
        record = yaml.safe_load(
            (explicit_run / "REST2" / f"replica_{replica:02d}" / "equilibration"
             / "resolved_stage.yaml").read_text())
        assert record["kind"] == "rest2_tau_equilibration"
        assert record["input_state"] == f"../{EXPLICIT_STAGES[-1]}/final_state.xml"
        assert record["kind"] == "rest2_tau_equilibration"
        seeds.extend([record["integrator_seed"], record["velocity_seed"],
                      record["barostat_seed"]])
        taus.append(record["tau"])
    assert len(set(seeds)) == len(seeds), f"seeds repeat across the ladder: {seeds}"
    assert taus[0] != taus[1], "every rung must have its own tau"


def test_rest2_production_totals_are_segment_times_exchanges(explicit_run):
    import csv

    config = yaml.safe_load((explicit_run / "md.config.yaml").read_text())
    rest2 = config["REST2"]
    per_segment = int(round(rest2["duration_per_segment_ps"] * 1000 /
                            config["common"]["timestep_fs"]))
    rows = list(csv.DictReader((explicit_run / "REST2" / "exchange_attempts.csv").open()))
    rounds = sorted({int(row["attempt_index"]) for row in rows})
    assert rounds == list(range(int(rest2["number_of_exchanges"])))
    assert max(int(row["step"]) for row in rows) == per_segment * int(rest2["number_of_exchanges"])


def test_extending_continues_exchange_production_without_repeating_equilibration(explicit_run):
    import csv

    def rows():
        return list(csv.DictReader((explicit_run / "REST2" / "exchange_attempts.csv").open()))

    before = rows()
    result = run_stage(explicit_run / "REST2", "run.py")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "resuming" in result.stdout
    assert "minimising" not in result.stdout

    after = rows()
    indices = sorted({int(row["attempt_index"]) for row in after})
    assert indices == list(range(len(indices))), "the exchange history restarted"
    assert len(after) > len(before)
    steps = [int(row["step"]) for row in after]
    assert steps == sorted(steps)


# --- 10. portability ---------------------------------------------------------

def test_no_generated_file_names_this_checkout_or_imports_the_package(explicit):
    offenders = {}
    for path in explicit.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        # An IMPORT of the package, not any mention of it: `md_templates_version` is a field
        # name in a runtime record and is not a dependency.
        if path.suffix == ".py" and re.search(r"^\s*(import|from)\s+md_templates\b", text,
                                              re.MULTILINE):
            offenders[str(path.relative_to(explicit))] = "imports md_templates"
        elif path.suffix == ".py" and str(REPO_ROOT) in text:
            offenders[str(path.relative_to(explicit))] = "names the checkout"
    assert not offenders, offenders


def test_every_launcher_is_executable(explicit):
    launchers = [explicit / "run_all.sh", explicit / "cMD" / "run.sh",
                 explicit / "REST2" / "run.sh", explicit / "REST2" / "equilibrate.sh",
                 explicit / "REST2" / "extend.sh"]
    launchers += [explicit / name / "run.sh" for name in EXPLICIT_STAGES]
    for path in launchers:
        assert path.stat().st_mode & stat.S_IXUSR, path


# --- restart accounting and the no-overwrite guard ---------------------------

def test_rerunning_a_completed_stage_changes_nothing_at_all(explicit_run):
    """Byte-for-byte, across every runtime output.

    Downstream stages have already consumed this final state. Rewriting any of these -- even the
    record -- would make the directory describe a run that is not the one downstream was built on.
    """
    import hashlib

    stage = explicit_run / "eq/npt_1kcal"
    outputs = ("stage.log", "stage.csv", "checkpoint.chk", "final_state.xml", "final.pdb",
               "resolved_stage.yaml")
    before = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in outputs}

    result = run_stage(stage)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "already complete" in result.stdout, result.stdout[-800:]

    after = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in outputs}
    assert after == before, {k: (before[k][:12], after[k][:12])
                             for k in outputs if before[k] != after[k]}


def test_the_runner_neither_advertises_nor_honours_a_redo_flag(explicit_run):
    """A completed dynamics stage still holds its terminal checkpoint.

    Anything that "reran" in place would load that checkpoint, find zero steps remaining, and
    rewrite the completion artifacts without integrating a single step from the parent. That is a
    redo in name only, so there is no flag for it.
    """
    stage = explicit_run / "eq/npt_1kcal"
    assert "MD_REDO" not in (stage / "run.py").read_text()

    result = run_stage(stage, extra_env={"MD_REDO": "1"})
    assert result.returncode == 0
    assert "already complete" in result.stdout, "MD_REDO was honoured"
    assert "MD_REDO" not in result.stdout, "the runner still advertises MD_REDO"
    assert "rm -f" in result.stdout and "run.py, run.sh and stage.yaml" in result.stdout
    assert "downstream" in result.stdout.lower(), "it must warn about downstream stages"


def test_the_recorded_signature_is_the_hash_of_the_current_stage_yaml(explicit_run):
    from .conftest import template_module

    md_stages = template_module("md_stages")
    for name in EXPLICIT_STAGES:
        request = yaml.safe_load((explicit_run / name / "stage.yaml").read_text())
        record = yaml.safe_load((explicit_run / name / "resolved_stage.yaml").read_text())
        recorded = record.get("stage_config_sha256")
        assert recorded, f"{name} recorded no stage_config_sha256"
        assert recorded == md_stages.stage_config_sha256(request), name


@pytest.mark.parametrize("field,value", [
    ("input_state", "../../minimization/final_state.xml"),
    ("system_pressure_bar", 50.0),
    ("integrator_seed", 123456789),
])
def test_changing_any_run_defining_field_makes_a_completed_stage_refuse(tmp_path, field, value):
    """The old check kept a list of fields that mattered, and these were not on it."""
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/npt_1kcal"
    assert run_stage(project / "eq/nvt_1kcal").returncode == 0
    assert run_stage(stage).returncode == 0

    request = yaml.safe_load((stage / "stage.yaml").read_text())
    assert field in request, f"{field} is not part of the stage request"
    assert request[field] != value, "the test must actually change something"
    request[field] = value
    (stage / "stage.yaml").write_text(yaml.safe_dump(request, sort_keys=False))

    result = run_stage(stage)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, f"a changed {field} was accepted as complete"
    assert "different stage request" in combined, combined[-800:]


def test_removing_the_runtime_outputs_starts_the_stage_fresh_from_its_parent(tmp_path):
    """Deliberate cleanup is the redo operation, and it must be a real one.

    The checkpoint is removed with the rest, so the stage cannot resume it: it loads the parent,
    resets its own step count, and integrates the configured number of steps.
    """
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/nvt_1kcal"
    assert run_stage(stage).returncode == 0

    request = yaml.safe_load((stage / "stage.yaml").read_text())
    configured = int(round(request["duration_ps"] * 1000 / request["timestep_fs"]))
    for name in ("stage.log", "stage.csv", "checkpoint.chk", "final_state.xml", "final.pdb",
                 "resolved_stage.yaml"):
        (stage / name).unlink()
    for kept in ("run.py", "run.sh", "stage.yaml"):
        assert (stage / kept).is_file(), f"cleanup must not remove {kept}"

    result = run_stage(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "started from" in result.stdout, result.stdout[-800:]
    assert "resuming" not in result.stdout, "it resumed a checkpoint that was removed"
    assert f"{configured:,} of {configured:,} steps" in result.stdout, result.stdout[-800:]
    record = yaml.safe_load((stage / "resolved_stage.yaml").read_text())
    assert record["steps"] == configured


def test_an_interrupted_stage_resumes_and_runs_exactly_the_missing_steps(tmp_path):
    """The bug this guards against, stated exactly.

    `Context.setState()` COPIES the parent's step count -- verified: a context stepped 37 times
    hands a State that sets a fresh context to 37, not 0. So a fresh stage that loaded its parent
    and then computed `configured - getStepCount()` was subtracting the PARENT's progress from its
    own target, giving too few remaining steps or a negative number.
    """
    import csv

    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/nvt_1kcal"
    request = yaml.safe_load((stage / "stage.yaml").read_text())
    configured = int(round(request["duration_ps"] * 1000 / request["timestep_fs"]))
    assert configured >= 4, "the smoke stage needs enough steps to interrupt in the middle"

    # Reach a genuinely PARTIAL checkpoint: run a copy of the stage configured for half the steps,
    # which leaves a checkpoint mid-way through the real stage's work.
    half = configured // 2
    partial = dict(request)
    partial["duration_ps"] = request["duration_ps"] * half / configured
    (stage / "stage.yaml").write_text(yaml.safe_dump(partial, sort_keys=False))
    assert run_stage(stage).returncode == 0
    interrupted = yaml.safe_load((stage / "resolved_stage.yaml").read_text())
    assert interrupted["steps"] == half

    # An interruption leaves the checkpoint and stage.csv, and NO completion artifacts -- a killed
    # process never reaches the point where those are written. Reproduce that exactly.
    (stage / "stage.yaml").write_text(yaml.safe_dump(request, sort_keys=False))
    (stage / "final_state.xml").unlink()
    (stage / "resolved_stage.yaml").unlink()
    assert (stage / "checkpoint.chk").is_file(), "the interruption must leave a checkpoint"
    rows_before = len(list(csv.reader((stage / "stage.csv").open())))

    result = run_stage(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    # It resumes AT the interrupted count and runs exactly what is missing -- not the whole stage
    # again, and not `configured` minus the parent's progress.
    assert f"resuming this stage from checkpoint.chk at step {half:,}" in result.stdout, \
        result.stdout[-800:]
    assert f"{configured - half:,} of {configured:,} steps" in result.stdout, result.stdout[-800:]
    record = yaml.safe_load((stage / "resolved_stage.yaml").read_text())
    assert record["steps"] == configured, "the stage did not finish at its configured length"
    rows_after = len(list(csv.reader((stage / "stage.csv").open())))
    assert rows_after > rows_before, "stage.csv restarted instead of appending"


def test_a_stage_with_only_half_its_completion_artifacts_refuses_to_run(tmp_path):
    """Neither reusing nor overwriting is safe when the two records disagree."""
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/nvt_1kcal"
    assert run_stage(stage).returncode == 0

    (stage / "resolved_stage.yaml").unlink()      # a final state with no record of what made it
    result = run_stage(stage)
    assert result.returncode != 0, result.stdout[-800:]
    combined = result.stdout + result.stderr
    assert "refusing to run" in combined and "resolved_stage.yaml" in combined, combined[-800:]
    assert "new output directory" in combined or "remove this stage" in combined


def test_a_completion_record_from_a_different_request_refuses_to_run(tmp_path):
    """The recorded signature IS the identity, so that is what gets tampered with here.

    The record's other fields are output, not input; `stage_config_sha256` is the one value that
    says which request produced these files.
    """
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/nvt_1kcal"
    assert run_stage(stage).returncode == 0

    record = yaml.safe_load((stage / "resolved_stage.yaml").read_text())
    record["stage_config_sha256"] = "0" * 64
    (stage / "resolved_stage.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    result = run_stage(stage)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-800:]
    assert "different stage request" in combined, combined[-800:]


def test_a_completion_record_with_no_signature_refuses_to_run(tmp_path):
    """A record written before signatures existed cannot vouch for what produced it."""
    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",))
    assert run_stage(project / "minimization").returncode == 0
    stage = project / "eq/nvt_1kcal"
    assert run_stage(stage).returncode == 0

    record = yaml.safe_load((stage / "resolved_stage.yaml").read_text())
    record.pop("stage_config_sha256")
    (stage / "resolved_stage.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    result = run_stage(stage)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "no stage_config_sha256" in combined, combined[-800:]


def test_a_fresh_stage_starts_its_own_step_count_at_zero(explicit_run):
    """Each stage counts its own steps; the parent's count is the parent's."""
    for name in ("eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        record = yaml.safe_load((explicit_run / name / "resolved_stage.yaml").read_text())
        stage = yaml.safe_load((explicit_run / name / "stage.yaml").read_text())
        expected = int(round(stage["duration_ps"] * 1000 / stage["timestep_fs"]))
        assert record["steps"] == expected, (name, record["steps"], expected)


# --- provenance --------------------------------------------------------------

def test_md_config_hash_is_the_hash_of_the_generated_md_config(explicit):
    """It hashed the INPUT document, which is not the file the project runs.

    Resolution fills in the stage paths, drops the solvent block that does not apply and
    reconciles the ensemble with the solvent, so the two differ in ways that matter.
    """
    import hashlib

    provenance = yaml.safe_load((explicit / "provenance.yaml").read_text())
    written = (explicit / "md.config.yaml").read_bytes()
    assert provenance["md_config_hash"] == hashlib.sha256(written).hexdigest()
    # and the lineage back to the prepared system it was generated from
    parent = provenance["parent_system"]
    inputs = (explicit / parent["inputs_path"]).resolve()
    for key, name in (("provenance_sha256", "provenance.yaml"),
                      ("forcefield_sha256", "forcefield.json"),
                      ("checksums_sha256", "SHA256SUMS")):
        assert parent[key] == hashlib.sha256((inputs / name).read_bytes()).hexdigest(), name


def test_the_template_commit_is_recorded_not_null(explicit_run):
    """Which revision of the run scripts produced this project."""
    for name in EXPLICIT_STAGES:
        record = yaml.safe_load((explicit_run / name / "resolved_stage.yaml").read_text())
        assert record["template_commit"], f"{name} recorded no template commit"
    equilibration = yaml.safe_load(
        (explicit_run / "REST2" / "replica_00" / "equilibration"
         / "resolved_stage.yaml").read_text())
    assert equilibration["template_commit"], "REST2 per-tau equilibration recorded no commit"


# --- GPU placement -----------------------------------------------------------

def test_rest2_equilibration_and_production_agree_on_device_placement(explicit_run):
    """Per-tau equilibration used to run one replica after another, idling every GPU but one."""
    devices = []
    for replica in range(2):
        record = yaml.safe_load(
            (explicit_run / "REST2" / f"replica_{replica:02d}" / "equilibration"
             / "resolved_stage.yaml").read_text())
        devices.append(record["device"])
    # Re-running equilibrate.py prints the placement and skips every replica that is done, so the
    # grouping can be read without integrating anything a second time.
    result = run_stage(explicit_run / "REST2", "equilibrate.py")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "already equilibrated, skipping" in result.stdout
    assert "replicas per device" in result.stdout or "no CUDA device" in result.stdout, \
        result.stdout[-600:]
    if any(device is not None for device in devices):
        assert len(set(devices)) == len(devices), \
            f"two replicas were pinned to the same device while others were idle: {devices}"


# --- FAIR runtime records ----------------------------------------------------

def test_every_common_stage_records_what_ran_at_runtime(explicit_run):
    """The record is written only after the final state exists, so `completed` cannot be early."""
    for name in EXPLICIT_STAGES:
        record = yaml.safe_load((explicit_run / name / "resolved_stage.yaml").read_text())
        assert record["format"] == "md-templates-runtime-record/v1"
        assert record["record_kind"] == "common_stage"
        assert record["status"] == "completed"
        assert record["started_utc"] and record["finished_utc"]
        assert record["platform"] == "CUDA", f"{name} did not run on CUDA"
        assert record["implementation"]["md_templates_version"], name
        assert record["outputs"]["final_state.xml"]["sha256"], name
        assert record["outputs"]["final_state.xml"]["bytes"] > 0
        if name != "minimization":
            assert record["completed_steps"] == record["configured_steps"], name


def test_a_stage_hashes_its_parents_final_state_and_not_a_checkpoint(explicit_run):
    record = yaml.safe_load((explicit_run / "eq/npt_1kcal" / "resolved_stage.yaml").read_text())
    assert record["input_state"].endswith("final_state.xml")
    assert record["input_state_sha256"], "the handoff it consumed must be identified"
    parent = yaml.safe_load((explicit_run / "eq/nvt_1kcal" / "resolved_stage.yaml").read_text())
    assert record["input_state_sha256"] == parent["outputs"]["final_state.xml"]["sha256"], \
        "the recorded input hash must be the parent's recorded output hash"


def test_cmd_writes_a_run_record_and_an_append_only_history(explicit_run):
    import json

    record = yaml.safe_load((explicit_run / "cMD" / "resolved_run.yaml").read_text())
    assert record["record_kind"] == "cmd_run" and record["status"] == "completed"
    assert record["platform"] == "CUDA"
    assert record["trajectories"]["solute"]["frames"] is not None
    # Trajectories are sized and counted, never hashed at runtime.
    assert "sha256" not in record["trajectories"]["solute"]
    assert record["outputs"]["final_state.xml"]["sha256"]

    lines = [json.loads(l) for l in
             (explicit_run / "cMD" / "invocations.jsonl").read_text().splitlines() if l.strip()]
    assert lines, "no invocation was recorded"
    assert [e["invocation_index"] for e in lines] == list(range(len(lines)))


def test_rest2_records_every_replica_and_the_exchange_lifetime(explicit_run):
    record = yaml.safe_load((explicit_run / "REST2" / "resolved_run.yaml").read_text())
    assert record["record_kind"] == "rest2_run" and record["status"] == "completed"
    thermo = record["thermodynamics"]
    assert thermo["common_beta_across_replicas"] is True
    assert thermo["positions_and_box_vectors_exchanged_together"] is True

    exchange = record["exchange"]
    assert exchange["lifetime_rounds"] >= exchange["rounds_this_invocation"]
    assert exchange["rng_provenance"]

    seeds = []
    for replica in record["replicas"]:
        assert replica["derived_scale_factor_s"] == pytest.approx((1 - replica["tau"]) ** 2)
        assert replica["ended_at_step"] >= replica["started_at_step"]
        seeds.extend(replica["seeds"].values())
    assert len(set(seeds)) == len(seeds), "replica seeds repeat"


def test_a_continuation_appends_and_keeps_indices_monotonic(explicit_run):
    import json

    path = explicit_run / "REST2" / "invocations.jsonl"
    before = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    result = run_stage(explicit_run / "REST2", "run.py")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]

    after = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(after) == len(before) + 1, "the continuation did not append one entry"
    assert after[:len(before)] == before, "earlier invocation history was rewritten"
    assert [e["invocation_index"] for e in after] == list(range(len(after)))
    assert after[-1]["exchange"]["first_round_index"] == before[-1]["exchange"]["lifetime_rounds"]
    assert after[-1]["exchange"]["lifetime_rounds"] > before[-1]["exchange"]["lifetime_rounds"]


def test_the_implicit_run_records_gbn2_and_no_pressure(implicit):
    for name in IMPLICIT_STAGES:
        result = run_stage(implicit / name)
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    record = yaml.safe_load((implicit / "eq/nvt_free" / "resolved_stage.yaml").read_text())
    assert record["pressure_bar"] is None
    assert record["barostats_in_system"] == 0

    forcefield = json.loads(
        (implicit.parent / "inputs" / "forcefield.json").read_text())
    # `applied_by` was added so the record says HOW the radii were set, not only which set.
    assert forcefield["implicit_solvent"]["model"] == "GBn2"
    assert forcefield["implicit_solvent"]["radii"] == "mbondi3"
    assert forcefield["protein"]["openmm_resource"] is None, "no OpenMM protein XML was loaded"
    assert forcefield["builder"]["route"] == "parmed.Structure.createSystem"
    assert forcefield["water"]["openmm_resource"] is None
    assert forcefield["explicit_solvent"] is None
