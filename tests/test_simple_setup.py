"""The Amber-like cMD workflow: a small request in, a runnable OpenMM directory out.

The properties these tests defend are the ones that make a generated directory worth having:
it runs without this package, every number in it is a literal, and nothing about one machine
leaks into it. A generated script that quietly re-acquired a `md_templates` import or an absolute
path would still run here and be useless on anyone else's machine.

Bounded and CPU-only unless marked `gpu`. The scientific defaults are asserted, not chosen, so a
change to force fields or the equilibration schedule fails here rather than silently reaching a
generated system.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_templates.openmm.config import ConfigError
from md_templates.openmm.simple import (SetupRequest, derive_seeds, format_preset, generate,
                                        origin_commit, package_version, refuse_unsafe_output, resolve,
                                        resolve_contributor, resolve_output_root)

from .conftest import ALA_PDB

SMOKE = {"production": "0.4 ps", "output_interval": "0.2 ps", "platform": "CPU"}


def _request(**overrides) -> SetupRequest:
    document = {"system": "T", "input": str(ALA_PDB), "type": "peptide",
                "solvent": "explicit", "protocol": "cMD", **SMOKE}
    document.update(overrides)
    return SetupRequest.from_document(document)


# --- the request ---------------------------------------------------------------

def test_the_minimal_request_is_seven_fields():
    """Everything else comes from the preset. If this grows, the interface has regressed."""
    request = SetupRequest.from_document({
        "system": "ALA", "input": "inputs/ALA.pdb", "type": "peptide", "solvent": "explicit",
        "protocol": "cMD", "production": "1 ns", "output_interval": "5 ps"})
    assert request.system == "ALA" and request.platform == "automatic"


def test_an_unknown_top_level_field_is_refused_rather_than_ignored():
    """A typo that silently becomes a default is worse than an error."""
    with pytest.raises(ConfigError, match="unknown field"):
        SetupRequest.from_document({"system": "A", "input": "x.pdb", "timestep": 4})


@pytest.mark.parametrize("text,expected_ps", [
    ("1 ns", 1000.0), ("250 ps", 250.0), ("0.5", 0.5), (2, 2.0), ("500 fs", 0.5)])
def test_durations_accept_the_units_a_person_writes(text, expected_ps):
    resolved = resolve(_request(production=text))
    assert resolved["production_ps"] == expected_ps


# --- the resolved preset -------------------------------------------------------

def test_the_peptide_preset_resolves_the_coupled_default_set():
    """ff14SB + TIP3P, coupled -- the repository's validated default and what every existing
    dataset used. `DEFAULT_SOLVENT` agrees, so `setup` and the older sys-config path resolve the
    same force fields. OPC is a supported selection, covered below, not a silent upgrade.
    """
    resolved = resolve(_request())
    assert resolved["sys_config"]["forcefield"] == {
        "protein": "amber14-all.xml", "water": "amber14/tip3p.xml"}
    assert resolved["sys_config"]["solvent"]["model"] == "TIP3P"
    assert resolved["temperature_K"] == 300.0
    assert resolved["timestep_fs"] == 2.0
    assert [s["name"] for s in resolved["equilibration_stages"]] == [
        "nvt_1kcal", "npt_1kcal", "npt_free"]


def test_the_ligand_preset_selects_sage_and_am1bcc():
    resolved = resolve(_request(type="ligand"))
    assert resolved["sys_config"]["solute"]["ligand_forcefield"] == "sage-2.2.1"
    assert resolved["sys_config"]["solute"]["ligand_charge_method"] == "am1bcc"


def test_implicit_solvent_has_no_barostat_and_a_shorter_chain():
    resolved = resolve(_request(solvent="implicit"))
    assert resolved["pressure_bar"] is None and resolved["barostat_interval"] is None
    assert [s["name"] for s in resolved["equilibration_stages"]] == ["nvt_1kcal", "nvt_free"]


def test_an_advanced_override_reaches_the_resolved_configuration():
    """An uncoupled field still overrides freely.

    This used to override `forcefield.water` alone, which is exactly the mixed configuration now
    refused -- the test was asserting the defect. Coupled fields move through `water:`; everything
    else is still reachable one at a time.
    """
    resolved = resolve(_request(advanced={"solvent.padding_nm": 1.2}))
    assert resolved["sys_config"]["solvent"]["padding_nm"] == 1.2


def test_an_advanced_setting_that_matches_nothing_is_refused():
    with pytest.raises(ConfigError, match="matches no field"):
        resolve(_request(advanced={"common.timestpe_fs": 4.0}))


def test_the_preset_is_shown_before_anything_is_written():
    text = format_preset(resolve(_request()))
    for expected in ("amber14-all.xml", "LangevinMiddle", "production", "output every"):
        assert expected in text


# --- generation-time safety ----------------------------------------------------

def test_a_timestep_above_3_fs_needs_hydrogen_mass_repartitioning():
    with pytest.raises(ConfigError, match="hydrogen-mass repartitioning"):
        resolve(_request(advanced={"common.timestep_fs": 4.0}))


def test_the_same_timestep_is_allowed_once_hydrogen_mass_is_repartitioned():
    resolved = resolve(_request(advanced={"common.timestep_fs": 4.0,
                                          "constraints.hydrogen_mass_amu": 4.0}))
    assert resolved["timestep_fs"] == 4.0


def test_an_output_interval_that_is_not_a_whole_number_of_steps_is_refused():
    """A reporter cannot write a third of a step, and rounding puts frames at times the
    header does not state."""
    with pytest.raises(ConfigError, match="not a whole number"):
        resolve(_request(output_interval="0.003 ps"))


def test_an_output_root_inside_a_git_worktree_is_refused(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        refuse_unsafe_output(repository / "data")


def test_a_root_outside_every_worktree_is_accepted(tmp_path):
    refuse_unsafe_output(tmp_path)          # raises if it objects


# --- the generated directory ---------------------------------------------------

CONTRIBUTOR = {"name": "Test Contributor", "email": "test@example.invalid"}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One ALA system, generated once. No dynamics."""
    root = tmp_path_factory.mktemp("systems")
    resolved = resolve(_request(system="ALA"))
    result = generate(resolved, input_path=ALA_PDB, output_root=root,
                      relative_project="project/2026-08", contributor=CONTRIBUTOR, echo=False)
    return result["system_dir"]


def _stage_scripts(system_dir: Path) -> list[Path]:
    return sorted([*system_dir.glob("min/*.py"), *system_dir.glob("eq/*/*.py"),
                   *system_dir.glob("cMD/*.py")])


def test_the_layout_is_the_documented_one(generated):
    assert {p.name for p in generated.iterdir()} == {
        "config.yaml", "paths.sh", "bin", "input", "min", "eq", "cMD", "run.sh"}
    assert not (generated / "common").exists(), "input/ replaced common/; two aliases would drift"
    assert {p.name for p in (generated / "eq").iterdir()} == {
        "nvt_1kcal", "npt_1kcal", "npt_free"}
    assert (generated / "cMD" / "cmd.py").is_file()


def test_no_generated_script_imports_md_templates(generated):
    """The property the whole design rests on: the directory outlives this package."""
    for script in _stage_scripts(generated):
        assert "md_templates" not in script.read_text(), script


def test_no_protocol_parses_configuration_or_asks_git(generated):
    """Checked on parsed imports, not substrings.

    `provenance: input/provenance.yaml` is a filename in the header, not a YAML parser, and a
    substring test cannot tell the difference -- it would force the reference out of the .out.
    """
    import ast

    for script in _stage_scripts(generated):
        tree = ast.parse(script.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("yaml", "subprocess", "md_templates"):
            assert forbidden not in imported, f"{script.name} imports {forbidden}"


def test_no_protocol_traverses_directories_or_names_a_concrete_path(generated):
    """Paths arrive through `files`. A protocol that computes one is not reusable."""
    for script in _stage_scripts(generated):
        text = script.read_text()
        for forbidden in ("Path(__file__)", "HERE =", "COMMON ="):
            assert forbidden not in text, f"{script.name} contains {forbidden!r}"


def test_no_generated_script_contains_an_absolute_path(generated):
    """An absolute path is a machine leaking into a directory meant to be moved."""
    for script in [*_stage_scripts(generated), generated / "run.sh"]:
        for number, line in enumerate(script.read_text().splitlines(), 1):
            assert not any(part in line for part in ("/home/", "/tmp/", "/data3/")), \
                f"{script.name}:{number}: {line.strip()}"


def test_every_generated_script_is_valid_python(generated):
    for script in _stage_scripts(generated):
        ast.parse(script.read_text())


def test_the_scientific_body_stays_small_once_the_mandated_header_is_excluded(generated):
    """The readability limit, measured against the part a reader is meant to read.

    The `.out` header grew when the output contract began requiring resolved paths, explicit seeds
    and a provenance reference -- about ten lines per stage that are output formatting, not
    protocol. Counting them against a limit on scientific code would push the science into a
    helper, which is the one outcome the design forbids. So the header is measured and excluded,
    and the remainder is held to the original targets.
    """
    targets = {"min.py": 40, "nvt_1kcal.py": 60, "npt_1kcal.py": 60, "npt_free.py": 60,
               "cmd.py": 60}
    for script in _stage_scripts(generated):
        lines = [l for l in script.read_text().splitlines() if l.strip()]
        inside, header = False, 0
        for line in lines:
            if 'print(f"""MD-OPENMM' in line:
                inside = True
            if inside:
                header += 1
            if inside and 'flush=True' in line:
                inside = False
        body = len(lines) - header
        assert body <= targets[script.name], (
            f"{script.name}: {body} lines of protocol (excluding a {header}-line header) "
            f"> {targets[script.name]}")


def test_the_resolved_parameters_are_literals_in_the_protocol(generated):
    """A reader sees the number that ran, not the expression that produced it."""
    text = (generated / "cMD" / "cmd.py").read_text()
    assert "300.0 * unit.kelvin" in text
    assert "2.0 * unit.femtoseconds" in text
    assert "simulation.step(200)" in text          # 0.4 ps at 2 fs


def test_the_system_config_is_small_and_records_its_origin(generated):
    document = yaml.safe_load((generated / "config.yaml").read_text())
    assert document["system_id"] == "ALA" and document["schema_version"] == 1
    assert document["contributor"]["name"] == "Test Contributor"
    assert document["created"]
    generated_by = document["generated_by"]
    # A wheel install legitimately has no checkout, but SOMETHING concrete must identify the
    # generator: a null commit is acceptable only beside a real package version.
    assert generated_by["md_templates_commit"] or generated_by["md_templates_version"]
    assert document["random_seed_base"]
    # It must not become a second copy of every parameter.
    assert "timestep_fs" not in yaml.safe_dump(document)


def test_generating_over_a_non_empty_directory_is_refused(generated):
    resolved = resolve(_request(system="ALA"))
    with pytest.raises(ConfigError, match="already exists and is not empty"):
        generate(resolved, input_path=ALA_PDB, output_root=generated.parent,
                 relative_project="project/2026-08", contributor=CONTRIBUTOR, echo=False)


def test_run_sh_only_calls_the_launchers(generated):
    """It must not reconstruct their commands, redirect their output, or hold a setting.

    The completion check moved into openmm-md, which is why run.sh no longer greps for the marker:
    a failed stage exits nonzero and `set -e` stops the chain.
    """
    text = (generated / "run.sh").read_text()
    assert "set -euo pipefail" in text
    assert "paths.sh" in text
    for leaked in ("-i ", "-p ", "--checkpoint", "temperature", "timestep", "> "):
        assert leaked not in text, f"run.sh contains {leaked!r}"
    assert len([l for l in text.splitlines() if l.strip()]) < 20


def test_every_launcher_names_its_parent_restart_and_its_own_outputs(generated):
    """The wiring is meant to be visible in the shell file, not inferred from a variable."""
    expected_parent = {
        "min.sh": "${INPUT_DIR}/initial_state.xml",
        "nvt_1kcal.sh": "${MIN_DIR}/min.state.xml",
        "npt_1kcal.sh": "${NVT_DIR}/nvt_1kcal.state.xml",
        "npt_free.sh": "${NPT_RESTRAINED_DIR}/npt_1kcal.state.xml",
        "cmd.sh": "${NPT_FREE_DIR}/npt_free.state.xml",
    }
    launchers = sorted([*generated.glob("min/*.sh"), *generated.glob("eq/*/*.sh"),
                        *generated.glob("cMD/*.sh")])
    assert {p.name for p in launchers} == set(expected_parent)
    for launcher in launchers:
        text = launcher.read_text()
        assert 'source "${STAGE_DIR}' in text and "paths.sh" in text
        assert f'-c "{expected_parent[launcher.name]}"' in text, launcher.name
        assert "${OPENMM_MD}" in text
        for required in ("-i ", "-p ", "-s ", "-o ", "-r "):
            assert required in text, f"{launcher.name} omits {required.strip()}"
        assert '"$@"' in text, f"{launcher.name} does not forward --force"


def test_paths_sh_contains_no_machine_path(generated):
    text = (generated / "paths.sh").read_text()
    assert 'MD_DATA:?' in text, "paths.sh must refuse an unset MD_DATA"
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        assert not any(part in line for part in ("/home/", "/tmp/", "/data3/")), line


# --- running it ----------------------------------------------------------------

@pytest.fixture(scope="module")
def without_md_templates(tmp_path_factory) -> Path:
    """A directory whose `md_templates` raises, so a stage that needs it cannot pass."""
    blocker = tmp_path_factory.mktemp("blocked")
    (blocker / "md_templates.py").write_text(
        'raise ImportError("md_templates is deliberately unavailable in this test")')
    return blocker


def _launch(system_dir: Path, relative: str, *, blocker: Path, extra=()):
    """Run a stage the way a user does: its own launcher, with $MD_DATA exported."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(blocker)
    # `generated` lives at <root>/ALA and paths.sh says $MD_DATA/project/2026-08/ALA, so the
    # managed root for these tests is the ancestor that makes that relative path resolve.
    environment["MD_DATA"] = str(system_dir.parent)
    launcher = system_dir / relative
    return subprocess.run(["bash", str(launcher), *extra], capture_output=True, text=True,
                          timeout=1800, env=environment)


@pytest.fixture(scope="module")
def runnable(tmp_path_factory):
    """A system whose paths.sh resolves: $MD_DATA/<relative>/ALA must be the real location."""
    managed = tmp_path_factory.mktemp("managed")
    root = managed / "project" / "2026-08"
    root.mkdir(parents=True)
    resolved = resolve(_request(system="ALA"))
    result = generate(resolved, input_path=ALA_PDB, output_root=root,
                      relative_project="project/2026-08", contributor=CONTRIBUTOR, echo=False)
    return managed, result["system_dir"]


def _run_stage(runnable, relative, blocker, extra=()):
    managed, system_dir = runnable
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(blocker)
    environment["MD_DATA"] = str(managed)
    return subprocess.run(["bash", str(system_dir / relative), *extra], capture_output=True,
                          text=True, timeout=1800, env=environment)


@pytest.mark.slow
def test_a_stage_runs_with_md_templates_unavailable(runnable, without_md_templates):
    """The claim, tested the only way it can be: make the import fail and run anyway."""
    _, system_dir = runnable
    result = _run_stage(runnable, "min/min.sh", without_md_templates)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    report = (system_dir / "min" / "min.out").read_text()
    assert "run_status: completed" in report
    assert (system_dir / "min" / "min.state.xml").is_file()


@pytest.mark.slow
def test_the_out_header_is_versioned_and_parses_as_key_value(runnable, without_md_templates):
    _, system_dir = runnable
    report_path = system_dir / "min" / "min.out"
    if not report_path.exists():
        assert _run_stage(runnable, "min/min.sh", without_md_templates).returncode == 0
    lines = report_path.read_text().splitlines()
    assert lines[0] == "MD-OPENMM OUTPUT VERSION: 1"
    header = {}
    for line in lines[1:]:
        if not line.strip():
            break
        key, _, value = line.partition(": ")
        header[key] = value
    for field in ("stage", "system", "openmm_version", "platform", "integrator",
                  "temperature_K", "timestep_fs", "integrator_seed", "topology", "system",
                  "coordinates", "restart", "provenance"):
        assert field in header, field
    assert header["stage"] == "min"
    assert int(header["integrator_seed"]) > 0, "the seed must be a concrete recorded number"


@pytest.mark.slow
def test_paths_sh_is_portable_under_a_moved_md_data(runnable, without_md_templates, tmp_path):
    """Copy the managed root elsewhere, export the new $MD_DATA, and the same launcher works.

    This is what "no machine path" buys: a generated system can be archived and restored under a
    different root without editing a single generated file.
    """
    managed, _ = runnable
    moved = tmp_path / "elsewhere"
    shutil.copytree(managed, moved)
    system_dir = moved / "project" / "2026-08" / "ALA"
    for stale in system_dir.rglob("*.out"):
        stale.unlink()
    for stale in list(system_dir.rglob("*.state.xml")) + list(system_dir.rglob("*.chk")):
        if stale.parent.name != "input":
            stale.unlink()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(without_md_templates)
    environment["MD_DATA"] = str(moved)
    result = subprocess.run(["bash", str(system_dir / "min" / "min.sh")], capture_output=True,
                            text=True, timeout=1800, env=environment)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "run_status: completed" in (system_dir / "min" / "min.out").read_text()


@pytest.mark.slow
def test_one_protocol_file_serves_two_different_path_sets(runnable, without_md_templates, tmp_path):
    """The protocol is reusable: change only the invocation, not the file.

    Same min.py, a second system's inputs, and outputs somewhere else entirely.
    """
    managed, system_dir = runnable
    if not (system_dir / "min" / "min.state.xml").exists():
        assert _run_stage(runnable, "min/min.sh", without_md_templates).returncode == 0

    elsewhere = tmp_path / "second"
    elsewhere.mkdir()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(without_md_templates)
    result = subprocess.run(
        [sys.executable, str(system_dir / "bin" / "openmm-md"),
         "-i", str(system_dir / "min" / "min.py"),
         "-p", str(system_dir / "input" / "topology.pdb"),
         "-s", str(system_dir / "input" / "system.xml"),
         "-c", str(system_dir / "input" / "initial_state.xml"),
         "-o", str(elsewhere / "again.out"),
         "-r", str(elsewhere / "again.state.xml")],
        capture_output=True, text=True, timeout=1800, env=environment)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "run_status: completed" in (elsewhere / "again.out").read_text()
    assert (elsewhere / "again.state.xml").is_file()


@pytest.mark.slow
def test_the_trajectory_and_restart_output_are_readable(runnable, without_md_templates):
    """A stage writing an unreadable DCD or unloadable state has produced no result."""
    managed, system_dir = runnable
    if not (system_dir / "min" / "min.state.xml").exists():
        assert _run_stage(runnable, "min/min.sh", without_md_templates).returncode == 0
    result = _run_stage(runnable, "eq/nvt_1kcal/nvt_1kcal.sh", without_md_templates)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    stage = system_dir / "eq" / "nvt_1kcal"
    from openmm import XmlSerializer

    state = XmlSerializer.deserialize((stage / "nvt_1kcal.state.xml").read_text())
    positions = state.getPositions(asNumpy=True)
    assert len(positions) > 0 and state.getPeriodicBoxVectors() is not None

    mdtraj = pytest.importorskip("mdtraj", reason="reading the DCD back needs MDTraj")
    trajectory = mdtraj.load(str(stage / "nvt_1kcal.dcd"),
                             top=str(system_dir / "input" / "topology.pdb"))
    assert trajectory.n_frames >= 1
    assert trajectory.n_atoms == len(positions), "the DCD and the state disagree on atom count"


@pytest.mark.slow
def test_rerunning_a_finished_stage_is_refused_and_changes_nothing(runnable, without_md_templates):
    """The launcher passes --force through, so a deliberate replacement is still possible."""
    managed, system_dir = runnable
    if not (system_dir / "min" / "min.out").exists():
        assert _run_stage(runnable, "min/min.sh", without_md_templates).returncode == 0
    report = system_dir / "min" / "min.out"
    before = report.read_bytes()

    refused = _run_stage(runnable, "min/min.sh", without_md_templates)
    assert refused.returncode != 0
    assert "already exist" in refused.stderr
    assert report.read_bytes() == before, "a refused rerun changed the .out"

    forced = _run_stage(runnable, "min/min.sh", without_md_templates, extra=("--force",))
    assert forced.returncode == 0, forced.stderr[-1500:]


@pytest.mark.slow
def test_the_ligand_route_generates_from_a_smiles_file(tmp_path):
    """phenol/IPH through Sage + AM1-BCC. The peptide example differs only in `type` and input."""
    smiles = tmp_path / "phenol_IPH.smi"
    smiles.write_text("Oc1ccccc1 IPH\n")
    root = tmp_path / "out" / "project" / "2026-08"
    root.mkdir(parents=True)
    resolved = resolve(_request(system="phenol-IPH", type="ligand", input=str(smiles)))
    result = generate(resolved, input_path=smiles, output_root=root,
                      relative_project="project/2026-08", contributor=CONTRIBUTOR, echo=False)

    system_dir = result["system_dir"]
    configuration = yaml.safe_load(
        (system_dir / "input" / "resolved_sys.config.yaml").read_text())
    assert configuration["solute"]["peptide"] is False
    assert configuration["solute"]["ligand_forcefield"] == "sage-2.2.1"
    assert not (configuration.get("forcefield") or {}).get("protein")
    for script in _stage_scripts(system_dir):
        assert "md_templates" not in script.read_text()


# --- the corrections this pass was for -----------------------------------------

def test_a_nested_missing_output_inside_a_worktree_is_still_refused(tmp_path):
    """The previous check looked only at the immediate parent.

    `$REPO/a/b/c` with none of a/b/c created answered "not a repository" -- because Git was asked
    about a directory that did not exist -- and was accepted, inside a worktree. It now walks up
    to the nearest existing ancestor before asking.
    """
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        refuse_unsafe_output(repository / "deeply" / "nested" / "missing")


def test_an_output_outside_md_data_is_refused(tmp_path, monkeypatch):
    """paths.sh resolves from $MD_DATA, so an output elsewhere could not be described portably."""
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("MD_DATA", str(managed))
    with pytest.raises(ConfigError, match="not beneath MD_DATA"):
        resolve_output_root(str(tmp_path / "somewhere-else"))


def test_an_unset_md_data_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("MD_DATA", raising=False)
    with pytest.raises(ConfigError, match="MD_DATA is not set"):
        resolve_output_root(str(tmp_path))


def test_the_relative_project_path_is_derived_from_md_data(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    (managed / "project" / "2026-08").mkdir(parents=True)
    monkeypatch.setenv("MD_DATA", str(managed))
    root, relative = resolve_output_root(str(managed / "project" / "2026-08"))
    assert relative == "project/2026-08"
    assert root == (managed / "project" / "2026-08").resolve()


def test_traversal_out_of_the_managed_root_is_refused(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("MD_DATA", str(managed))
    with pytest.raises(ConfigError):
        resolve_output_root(str(managed / ".." / "escaped"))


# --- contributor, seeds, version -----------------------------------------------

def test_a_contributor_is_never_invented(monkeypatch):
    monkeypatch.delenv("MD_CONTRIBUTOR", raising=False)
    with pytest.raises(ConfigError, match="no contributor"):
        resolve_contributor({}, interactive=False)


def test_the_contributor_comes_from_the_request_then_the_environment(monkeypatch):
    monkeypatch.setenv("MD_CONTRIBUTOR", "From Environment")
    assert resolve_contributor({}, interactive=False)["name"] == "From Environment"
    explicit = resolve_contributor({"contributor": "From Request <r@example.invalid>"},
                                   interactive=False)
    assert explicit == {"name": "From Request", "email": "r@example.invalid"}


def test_an_explicit_base_seed_is_used_rather_than_ignored():
    """`advanced.common.random_seed` was accepted and then silently dropped."""
    first = resolve(_request(advanced={"common.random_seed": 20260828}))
    again = resolve(_request(advanced={"common.random_seed": 20260828}))
    assert first["seeds"]["base"]["seed"] == 20260828
    assert first["seeds"]["cMD"] == again["seeds"]["cMD"], "derivation must be deterministic"


def test_stage_seeds_are_distinct_and_concrete():
    seeds = derive_seeds(None, ["min", "nvt_1kcal", "cMD"])
    values = [seeds[stage]["integrator"] for stage in ("min", "nvt_1kcal", "cMD")]
    assert len(set(values)) == 3, "each stage must get its own integrator seed"
    assert all(isinstance(v, int) and v > 0 for v in values)


def test_the_seeds_are_shown_in_the_resolved_preset():
    text = format_preset(resolve(_request(advanced={"common.random_seed": 7})))
    assert "base seed     7" in text


def test_a_package_version_is_available_even_without_git():
    """`md_templates_commit: null` is acceptable only beside a concrete version."""
    assert package_version() or origin_commit()


def test_the_protocol_sets_the_integrator_seed_explicitly(generated):
    text = (generated / "cMD" / "cmd.py").read_text()
    assert "setRandomNumberSeed(" in text
    assert "integrator_seed:" in text, "the seed must reach the .out header too"


# --- solvent coupling ----------------------------------------------------------
#
# The protein and water force fields were parameterised together: ff19SB's amino-acid CMAPs were
# fit in OPC, ff14SB's in TIP3P. A configuration naming one protein force field and the other
# water model is not a variant -- it is a System nobody parameterised, and it runs.

def test_the_default_explicit_selection_is_the_coupled_tip3p_set():
    resolved = resolve(_request())
    forcefield = resolved["sys_config"]["forcefield"]
    assert forcefield["protein"] == "amber14-all.xml"
    assert forcefield["water"] == "amber14/tip3p.xml"
    assert resolved["sys_config"]["solvent"]["model"] == "TIP3P"


def test_the_setup_default_matches_the_libraries_default_solvent():
    """`setup` and the older sys-config path must not disagree about the default force field."""
    from md_templates.openmm import defaults as D

    assert D.canonical_solvent(SetupRequest.__dataclass_fields__["water"].default) == \
        D.DEFAULT_SOLVENT


def test_opc_is_selectable_and_equally_self_consistent():
    """A supported selection, resolving its own coupled pair -- not a variant of the default."""
    resolved = resolve(_request(water="OPC"))
    forcefield = resolved["sys_config"]["forcefield"]
    assert forcefield["protein"] == "amber19-all.xml"
    assert forcefield["water"] == "amber19/opc.xml"
    assert resolved["sys_config"]["solvent"]["model"] == "OPC"


def test_one_field_selects_the_whole_coupled_set():
    """`water:` is the single user-facing choice; nothing else has to be named."""
    for name, protein, water in (("OPC", "amber19-all.xml", "amber19/opc.xml"),
                                 ("TIP3P", "amber14-all.xml", "amber14/tip3p.xml")):
        resolved = resolve(_request(water=name))
        assert (resolved["sys_config"]["forcefield"]["protein"],
                resolved["sys_config"]["forcefield"]["water"]) == (protein, water)


@pytest.mark.parametrize("override,description", [
    ({"forcefield.water": "amber19/opc.xml"}, "ff14SB protein with OPC water"),
    ({"forcefield.protein": "amber19-all.xml"}, "ff19SB protein with TIP3P water"),
    ({"solvent.model": "OPC"}, "a TIP3P pair labelled OPC"),
])
def test_a_contradictory_coupled_override_is_refused(override, description):
    """Overriding one of the three used to yield a silently mixed configuration."""
    with pytest.raises(ConfigError, match="not internally consistent"):
        resolve(_request(advanced=override))


def test_an_unsupported_water_model_is_refused_by_name():
    with pytest.raises(ConfigError, match="water must be one of"):
        resolve(_request(water="SPCE"))


def test_moving_the_whole_coupled_set_together_is_accepted():
    """The escape hatch still works when every coupled field moves -- though `water: OPC` is the
    supported way to say the same thing."""
    resolved = resolve(_request(advanced={"forcefield.protein": "amber19-all.xml",
                                          "forcefield.water": "amber19/opc.xml",
                                          "solvent.model": "OPC"}))
    assert resolved["sys_config"]["solvent"]["model"] == "OPC"


def test_the_resolved_water_model_is_shown_in_the_preset():
    assert "water model TIP3P" in format_preset(resolve(_request()))
    assert "water model OPC" in format_preset(resolve(_request(water="OPC")))


def test_the_generated_configuration_carries_the_coupled_set(generated):
    """What system generation was actually given, not what the request said."""
    configuration = yaml.safe_load(
        (generated / "input" / "resolved_sys.config.yaml").read_text())
    assert configuration["forcefield"]["protein"] == "amber14-all.xml"
    assert configuration["forcefield"]["water"] == "amber14/tip3p.xml"
    assert configuration["solvent"]["model"] == "TIP3P"


def test_the_provenance_record_names_the_force_fields_that_were_loaded(generated):
    forcefield = json.loads((generated / "input" / "forcefield.json").read_text())
    text = json.dumps(forcefield)
    assert "amber14-all.xml" in text and "amber14/tip3p.xml" in text


# --- contributor ---------------------------------------------------------------

def test_a_contributor_in_the_request_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("MD_CONTRIBUTOR", "Environment Person")
    resolved = resolve_contributor({"contributor": "Request Person"}, interactive=False)
    assert resolved["name"] == "Request Person"


def test_the_environment_is_used_when_the_request_is_silent(monkeypatch):
    monkeypatch.setenv("MD_CONTRIBUTOR", "Environment Person <e@example.invalid>")
    assert resolve_contributor({}, interactive=False) == {
        "name": "Environment Person", "email": "e@example.invalid"}


def test_a_prompt_is_used_when_prompting_is_allowed(monkeypatch):
    monkeypatch.delenv("MD_CONTRIBUTOR", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "Prompted Person")
    assert resolve_contributor({}, interactive=True)["name"] == "Prompted Person"


def test_no_contributor_and_no_prompt_is_an_error_not_a_guess(monkeypatch):
    """There is deliberately no fallback to $USER or the Git config."""
    monkeypatch.delenv("MD_CONTRIBUTOR", raising=False)
    with pytest.raises(ConfigError, match="no contributor"):
        resolve_contributor({}, interactive=False)
