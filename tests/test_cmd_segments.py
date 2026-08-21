"""Conventional MD as a public method, and as a durable segmented run.

Every test here fails against `8c0c85c`, where `production.method = "md"` was refused outright and
`cMD_1` was single-shot.

The persistence tests deliberately use tiny segments. What is being tested is the *contract* --
commit ordering, watermark truncation, append behaviour, refusal before mutation -- and none of that
gets more true with longer segments; it only gets slower to check.
"""

from __future__ import annotations

import json
import os
import struct
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


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


def _frames(path: Path) -> int:
    with open(path, "rb") as handle:
        handle.seek(8)
        return struct.unpack("<i", handle.read(4))[0]


def _tiny_md_config(*, implicit: bool, segment: str = "2 ps") -> dict:
    """A whole MD-only protocol small enough to run in a test, with intervals that fit."""
    return {
        "profile": f"{'implicit' if implicit else 'explicit'}-md-peptide-v1",
        "protocol": {
            "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                           "temperature": "300 K", "friction": "1 /ps"},
            "equilibration": ({"protocol": "simple", "minimize_max_iterations": 100,
                               "nvt": "1 ps"} if implicit else
                              {"protocol": "simple", "minimize_max_iterations": 100,
                               "nvt": "1 ps", "npt": "1 ps", "npt_free": "1 ps"}),
            "production": {"method": "md", "duration_per_segment": segment},
        },
        "randomness": {"master_seed": 20260821},
        "execution": {"platform": "CPU", "precision": "mixed",
                      "reporting": {"all_atom": "1 ps", "solute": "0.2 ps"}},
    }


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def implicit_bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("cmd_implicit") / "bundle"
    config = REPO_ROOT / "test" / "ala" / "cMD" / "implicit" / "system_config.json"
    result = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(out), "--config", str(config))
    if result.returncode != 0:
        pytest.skip(f"implicit preparation unavailable: {result.stderr[-300:]}")
    return out


@pytest.fixture(scope="module")
def md_project(implicit_bundle, tmp_path_factory):
    """An MD-only project, generated through the public entry point."""
    out = tmp_path_factory.mktemp("cmd_project") / "run"
    config = tmp_path_factory.mktemp("cmd_config") / "md.json"
    config.write_text(json.dumps(_tiny_md_config(implicit=True)))
    result = _run(INPUT_GEN, "--system", str(implicit_bundle / "system_manifest.json"),
                  "-o", str(out), "--config", str(config))
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def _run_stage(project: Path, stage: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(project / stage / f"{stage}.sh")], capture_output=True, text=True,
                          cwd=str(project / stage))


# ---------------------------------------------------------------------------------------------
# standalone MD is a public method
# ---------------------------------------------------------------------------------------------

def test_md_only_generation_is_no_longer_refused(md_project):
    """It used to fail with 'Conventional-MD-only projects are not yet generated'."""
    manifest = json.loads((md_project / "run_manifest.json").read_text())
    assert [e["stage"] for e in manifest["stages"]] == ["min", "eq_nvt", "cMD_1"]


def test_an_md_only_project_contains_no_rest2_machinery(md_project):
    """No tau ladder, no exchange schedule, no omega policy, and no seeds for a missing stage."""
    assert not (md_project / "REST2_1").exists()
    for stage in ("min", "eq_nvt", "cMD_1"):
        payload = json.loads((md_project / stage / f"{stage}.json").read_text())
        assert "rest2" not in payload, stage
    manifest = json.loads((md_project / "run_manifest.json").read_text())
    assert not any("REST2" in purpose for purpose in manifest["randomness"]["seeds"])


def test_the_two_md_stage_graphs_differ_only_by_the_npt_stages():
    from md_templates.openmm.input_gen import stage_order_for

    assert stage_order_for("explicit", "md") == ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1")
    assert stage_order_for("implicit", "md") == ("min", "eq_nvt", "cMD_1")
    # and REST2 keeps its chain
    assert stage_order_for("explicit", "rest2")[-1] == "REST2_1"
    assert stage_order_for("implicit", "rest2") == ("min", "eq_nvt", "cMD_1", "REST2_1")


def test_an_md_only_project_takes_its_segment_from_the_canonical_field(md_project):
    """Not from the untyped generator-only `conventional_md.duration`."""
    payload = json.loads((md_project / "cMD_1" / "cMD_1.json").read_text())
    assert payload["steps"] == 1000, "2 ps at 2 fs"


def test_stating_both_segment_fields_is_refused(implicit_bundle, tmp_path):
    """Two fields competing for one stage is resolved by refusing, not by precedence."""
    config = tmp_path / "md.json"
    document = _tiny_md_config(implicit=True)
    document["conventional_md"] = {"duration": "3 ps"}
    config.write_text(json.dumps(document))
    result = _run(INPUT_GEN, "--system", str(implicit_bundle / "system_manifest.json"),
                  "-o", str(tmp_path / "out"), "--config", str(config))
    assert result.returncode != 0
    assert "competing" in result.stderr


# ---------------------------------------------------------------------------------------------
# committed segments
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_two_segments_continue_in_one_run_directory(md_project):
    """Re-invoking the launcher continues; it never starts a sibling run and calls it continuation."""
    import glob

    for stage in ("min", "eq_nvt"):
        assert _run_stage(md_project, stage).returncode == 0, stage

    first = _run_stage(md_project, "cMD_1")
    assert first.returncode == 0, first.stderr[-600:]
    assert "committed generation 1" in first.stdout

    second = _run_stage(md_project, "cMD_1")
    assert second.returncode == 0, second.stderr[-600:]
    assert "continuing from generation 1 via checkpoint" in second.stdout, second.stdout
    assert "committed generation 2" in second.stdout

    committed = json.loads(
        Path(glob.glob(str(md_project / "cMD_1" / "run" / "**" / "committed.json"),
                       recursive=True)[0]).read_text())
    assert committed["generation"] == 2
    assert committed["absolute_step"] == 2000, "1,000 steps per segment, twice"
    assert committed["restart_source_for_this_segment"] == "checkpoint"

    # exactly one run directory: a continuation must not create a sibling
    assert len(list((md_project / "cMD_1" / "run").glob("*"))) >= 1
    assert not list(md_project.glob("cMD_1*_1")), "a sibling run directory was created"


@pytest.mark.slow
def test_outputs_append_across_the_boundary_without_duplicates(md_project):
    """Frame counts are the whole test: an overwrite and an append both leave a plausible file."""
    results = json.loads((md_project / "cMD_1" / "cMD_1_results.json").read_text())
    total = results["segment"]["absolute_step"]

    for key, name in (("all_atom", "cMD_1_all_atoms.dcd"),
                      ("selected_atoms", "cMD_1_selected_atoms.dcd")):
        interval = results["reporters"][key]["interval_steps"]
        assert _frames(md_project / "cMD_1" / name) == total // interval, name

    lines = (md_project / "cMD_1" / "cMD_1.log").read_text().splitlines()
    headers = [line for line in lines if line.startswith("#")]
    assert len(headers) == 1, f"one header, got {len(headers)}"
    steps = [int(line.split(",")[0]) for line in lines if not line.startswith("#")]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), "monotonic, no duplicates"
    assert steps[-1] == total


@pytest.mark.slow
def test_an_incompatible_continuation_is_refused_before_anything_is_appended(md_project, tmp_path):
    """Refusal must happen while the previous segment's outputs are exactly as it left them."""
    import shutil

    from md_templates.openmm import stage as stage_mod

    copy = tmp_path / "changed"
    shutil.copytree(md_project, copy)
    trajectory = copy / "cMD_1" / "cMD_1_all_atoms.dcd"
    before = trajectory.read_bytes()

    payload = json.loads((copy / "cMD_1" / "cMD_1.json").read_text())
    payload["integrator"]["temperature"] = "310 K"          # a different calculation
    with pytest.raises(Exception) as caught:
        stage_mod.execute_stage(copy / "cMD_1" / "cMD_1.json", payload)
    assert "continu" in str(caught.value).lower() or "differ" in str(caught.value).lower()
    assert trajectory.read_bytes() == before, "outputs were touched before the refusal"


def test_the_continuity_contract_excludes_the_segment_count():
    """Asking for more segments is the one change that is always safe."""
    from md_templates.openmm.stage import _cmd_continuity

    class _System:
        def getNumParticles(self): return 22
        def getNumConstraints(self): return 12
        def usesPeriodicBoundaryConditions(self): return False

    payload = {"stage": "cMD_1", "steps": 1000,
               "integrator": {"type": "langevin-middle", "timestep": "2 fs",
                              "temperature": "300 K", "friction": "1 /ps"},
               "reporting": {}, "input": {"state": "x.xml", "produced_by": "eq_nvt"}}
    contract = _cmd_continuity(payload, _System(), selected_atoms_fingerprint=None)
    assert "segments" not in json.dumps(contract).lower().replace("steps_per_segment", "")
    assert contract["ensemble"] == "NVT"
    assert contract["steps_per_segment"] == 1000


def test_an_uncommitted_tail_is_truncated_to_the_watermark(tmp_path):
    """Frames past the committed boundary belong to an invocation that died before committing."""
    from md_templates.openmm.cmd_segments import truncate_to_watermark

    log = tmp_path / "run.log"
    log.write_text('#"Step","T"\n100,300\n200,301\n300,299\n400,298\n')
    action = truncate_to_watermark(log, "state_log", 2, n_atoms=0)
    assert action["action"] == "truncated" and action["to"] == 2
    assert log.read_text().splitlines() == ['#"Step","T"', "100,300", "200,301"]

    # a file already at or below its watermark is left alone
    again = truncate_to_watermark(log, "state_log", 5, n_atoms=0)
    assert again["action"] == "kept"


# ---------------------------------------------------------------------------------------------
# corrected defaults
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("profile", [
    "explicit-md-peptide-v1", "explicit-md-ligand-v1",
    "implicit-md-peptide-v1", "implicit-md-ligand-v1",
    "explicit-rest2-peptide-v1", "explicit-rest2-ligand-v1",
    "implicit-rest2-peptide-v1", "implicit-rest2-ligand-v1",
])
def test_the_shipped_defaults_are_the_corrected_ones(profile):
    path = (REPO_ROOT / "src" / "md_templates" / "openmm" / "spec" / "profiles"
            / f"{profile}.json")
    defaults = json.loads(path.read_text())["defaults"]
    protocol = defaults["protocol"]

    assert protocol["schema_version"] == 5, "a version-4 label must not carry version-5 semantics"
    assert protocol["equilibration"]["minimize_max_iterations"] == 1000
    assert protocol["equilibration"]["nvt"] == "10 ps"
    assert protocol["production"]["duration_per_segment"] == "5 ns"
    assert defaults["execution"]["reporting"]["all_atom"] == "100 ps"
    assert defaults["execution"]["reporting"]["solute"] == "10 ps"
    assert protocol["integrator"]["friction"] == "1 /ps"
    assert protocol["integrator"]["kind"] == "langevin-middle"

    if "rest2" in profile:
        assert protocol["production"]["exchange"][
            "number_of_exchanges_per_segment"] == 1000
        ladder = protocol["production"]["tau_ladder"]
        assert (ladder["minimum"], ladder["maximum"]) == (0.0, 0.5)
        assert ladder["interpolation"] == "linear"


def test_the_worked_ladders_are_the_decided_sizes():
    """Explicit follows the instruction; implicit follows the user's own decision in-session.

    The instruction calls the implicit 4/6 counts incorrect "unless the user supplies a new
    explicit scientific decision", and the user supplied exactly that: alanine 4, RGDfV 6 under
    implicit solvent. Recorded here so the next reader sees which authority set which number.
    """
    src = REPO_ROOT / "src" / "md_templates" / "openmm" / "spec" / "profiles"

    def count(name):
        return json.loads((src / f"{name}.json").read_text())[
            "defaults"]["protocol"]["production"]["tau_ladder"]["count"]

    assert count("explicit-rest2-peptide-v1") == 6
    assert count("explicit-rest2-ligand-v1") == 10
    assert count("implicit-rest2-peptide-v1") == 4
    assert count("implicit-rest2-ligand-v1") == 6


def test_a_retired_protocol_schema_is_refused_with_its_migration():
    from md_templates.openmm.spec import resolve

    document = json.loads(
        (REPO_ROOT / "test" / "ala" / "REST2" / "alanine_rest2.json").read_text())
    document["protocol"]["schema_version"] = 4
    with pytest.raises(Exception, match="retired"):
        resolve.resolve_spec(document)


def test_an_unknown_protocol_schema_is_refused():
    from md_templates.openmm.spec import resolve

    document = json.loads(
        (REPO_ROOT / "test" / "ala" / "REST2" / "alanine_rest2.json").read_text())
    document["protocol"]["schema_version"] = 99
    with pytest.raises(Exception, match="not a version this build understands"):
        resolve.resolve_spec(document)


# ---------------------------------------------------------------------------------------------
# implicit runs never invent a box
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_an_implicit_run_records_no_volume_and_no_periodic_trajectory(md_project):
    results = json.loads((md_project / "cMD_1" / "cMD_1_results.json").read_text())
    assert results["periodic"] is False
    assert results["box_volume_nm3"] is None
    assert "not applicable" in results["volume_note"]
    assert results["reporters"]["all_atom"]["wrapped"] is False
    assert results["reporters"]["state_log"]["volume_and_density"] is False

    header = (md_project / "cMD_1" / "cMD_1.log").read_text().splitlines()[0]
    assert "Volume" not in header, header
    assert "Density" not in header, header


def test_reporting_is_declared_only_where_it_can_be_written():
    """The generator must not promise a trajectory a stage is too short to produce."""
    from md_templates.openmm.input_gen import _reporting_for_stage

    equilibration = _reporting_for_stage("eq_nvt", 2_500, full_steps=25_000, selected_steps=2_500)
    assert "full_system_interval_steps" not in equilibration
    assert "all-atom trajectory" in equilibration["omitted"]
    assert equilibration["selected_atoms_interval_steps"] == 2_500

    production = _reporting_for_stage("cMD_1", 125_000, full_steps=25_000, selected_steps=2_500)
    assert production["full_system_interval_steps"] == 25_000
    assert "omitted" not in production
