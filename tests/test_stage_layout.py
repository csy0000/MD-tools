"""The generated directory contract: what is written, what reads what, and what refuses.

These tests execute the generated stages. A tree with the right filenames and a configuration that
reads well proves nothing about which state a stage actually started from.
"""
from __future__ import annotations

import stat

import pytest
import yaml

from .conftest import REPO_ROOT, run_stage, tiny_project

pytestmark = pytest.mark.slow

EXPLICIT_STAGES = ["minimization", "eq1_nvt_1kcal", "eq2_npt_1kcal", "eq3_npt_free"]
IMPLICIT_STAGES = ["minimization", "eq1_nvt_1kcal", "eq2_nvt_free"]
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
    stages = [p.name for p in sorted(explicit.iterdir())
              if p.is_dir() and p.name not in ("cMD", "REST2")]
    assert stages == sorted(EXPLICIT_STAGES)
    for name in EXPLICIT_STAGES:
        assert {p.name for p in (explicit / name).iterdir()} >= {"run.py", "run.sh", "stage.yaml"}
    assert (explicit / "run_all.sh").is_file()
    assert {p.name for p in (explicit / "cMD").iterdir()} == {"run.py", "run.sh"}
    assert {p.name for p in (explicit / "REST2").iterdir()} >= {
        "equilibrate.py", "equilibrate.sh", "run.py", "run.sh", "extend.sh", "rest2_scaling.py",
        "replica_00", "replica_01"}
    for replica in ("replica_00", "replica_01"):
        assert {p.name for p in (explicit / "REST2" / replica).iterdir()} == {
            "equilibration", "production"}


def test_the_implicit_tree_has_no_npt_stage(implicit):
    stages = [p.name for p in sorted(implicit.iterdir())
              if p.is_dir() and p.name not in ("cMD", "REST2")]
    assert stages == sorted(IMPLICIT_STAGES)
    assert not any("npt" in name for name in stages), "implicit solvent has no box to control"


def test_the_folder_name_does_not_claim_1kcal_when_the_restraint_is_not_1(tmp_path):
    """A directory called eq1_nvt_1kcal holding a 5 kcal/mol/A^2 run is a lie told by a filename."""
    def stronger(protocol):
        protocol["equilibration"]["restraint_k_kcal_mol_a2"] = 5.0

    project = tiny_project(tmp_path, solvent="OPC", methods=("cMD",), edit=stronger)
    names = {p.name for p in project.iterdir() if p.is_dir()}
    assert "eq1_nvt_restrained" in names, names
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
            assert stage["parent"] == stages[index - 1]
            assert stage["input_state"] == f"../{stages[index - 1]}/final_state.xml"
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
    assert "final_state.xml" in combined and EXPLICIT_STAGES[-1] in combined, combined[-800:]
    assert "./run.sh" in combined, combined[-800:]

    # and the same for a common stage whose parent has not run
    result = run_stage(project / "eq2_npt_1kcal")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "eq1_nvt_1kcal" in combined and "final_state.xml" in combined, combined[-800:]


# --- 5/6. runtime force state, stage by stage --------------------------------

EXPECTED_EXPLICIT = {
    "minimization":  {"restrained": True,  "active": 0},
    "eq1_nvt_1kcal": {"restrained": True,  "active": 0},
    "eq2_npt_1kcal": {"restrained": True,  "active": 1},
    "eq3_npt_free":  {"restrained": False, "active": 1},
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
    free = yaml.safe_load((implicit / "eq2_nvt_free" / "resolved_stage.yaml").read_text())
    assert free["restraint_kj_mol_nm2"] == 0.0
    assert free["ensemble"] == "NVT"


# --- 3. the files a stage writes ---------------------------------------------

def test_each_common_stage_writes_the_required_files(explicit_run):
    for name in EXPLICIT_STAGES:
        present = {p.name for p in (explicit_run / name).iterdir()}
        expected = STAGE_FILES - ({"checkpoint.chk", "stage.csv"}
                                  if name == "minimization" else set())
        assert expected <= present, (name, sorted(expected - present))


def test_resolved_stage_records_what_actually_ran(explicit_run):
    record = yaml.safe_load((explicit_run / "eq2_npt_1kcal" / "resolved_stage.yaml").read_text())
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
        if path.suffix == ".py" and "md_templates" in text:
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
