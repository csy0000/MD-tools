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
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_templates.openmm.config import ConfigError
from md_templates.openmm.simple import (SetupRequest, format_preset, generate, refuse_unsafe_output,
                                        resolve)

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

def test_the_peptide_preset_is_this_repository_s_validated_default():
    """ff14SB + TIP3P. Not ff19SB/OPC, which is a selection and not the default."""
    resolved = resolve(_request())
    assert resolved["sys_config"]["forcefield"] == {
        "protein": "amber14-all.xml", "water": "amber14/tip3p.xml"}
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
    resolved = resolve(_request(advanced={"forcefield.water": "amber19/opc.xml"}))
    assert resolved["sys_config"]["forcefield"]["water"] == "amber19/opc.xml"


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

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One ALA system, generated once. No dynamics."""
    root = tmp_path_factory.mktemp("systems")
    resolved = resolve(_request(system="ALA"))
    result = generate(resolved, input_path=ALA_PDB, output_root=root, echo=False)
    return result["system_dir"]


def _stage_scripts(system_dir: Path) -> list[Path]:
    return sorted([*system_dir.glob("min/*.py"), *system_dir.glob("eq/*/*.py"),
                   *system_dir.glob("cMD/*.py")])


def test_the_layout_is_the_documented_one(generated):
    assert {p.name for p in generated.iterdir()} == {
        "config.yaml", "common", "min", "eq", "cMD", "run.sh"}
    assert {p.name for p in (generated / "eq").iterdir()} == {
        "nvt_1kcal", "npt_1kcal", "npt_free"}
    assert (generated / "cMD" / "cmd.py").is_file()


def test_no_generated_script_imports_md_templates(generated):
    """The property the whole design rests on: the directory outlives this package."""
    for script in _stage_scripts(generated):
        assert "md_templates" not in script.read_text(), script


def test_no_generated_script_parses_configuration_or_asks_git(generated):
    for script in _stage_scripts(generated):
        text = script.read_text()
        for forbidden in ("yaml", "subprocess", "git ", "sha256", "components.lock"):
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


def test_the_scripts_stay_within_the_readability_targets(generated):
    """Design targets from the milestone: 40 lines for minimisation, 60 for the rest.

    Not a style rule for its own sake. The previous generation's production script was 286 lines
    of which about twelve were OpenMM, and the Simulation was not even in it.
    """
    targets = {"min.py": 40, "nvt_1kcal.py": 60, "npt_1kcal.py": 60, "npt_free.py": 60,
               "cmd.py": 60}
    for script in _stage_scripts(generated):
        lines = len([l for l in script.read_text().splitlines() if l.strip()])
        assert lines <= targets[script.name], f"{script.name}: {lines} > {targets[script.name]}"


def test_the_resolved_parameters_are_literals_in_the_script(generated):
    """A reader sees the number that ran, not the expression that produced it."""
    text = (generated / "cMD" / "cmd.py").read_text()
    assert "300.0 * unit.kelvin" in text
    assert "2.0 * unit.femtoseconds" in text
    assert "simulation.step(200)" in text          # 0.4 ps at 2 fs


def test_the_system_config_is_small_and_records_its_origin(generated):
    document = yaml.safe_load((generated / "config.yaml").read_text())
    assert document["system_id"] == "ALA" and document["schema_version"] == 1
    assert "md_templates_commit" in document["generated_by"]
    # It must not become a second copy of every parameter.
    assert "timestep_fs" not in yaml.safe_dump(document)


def test_generating_over_a_non_empty_directory_is_refused(generated, tmp_path_factory):
    resolved = resolve(_request(system="ALA"))
    with pytest.raises(ConfigError, match="already exists and is not empty"):
        generate(resolved, input_path=ALA_PDB, output_root=generated.parent, echo=False)


def test_run_sh_is_short_and_stops_on_a_failed_stage(generated):
    text = (generated / "run.sh").read_text()
    assert "set -euo pipefail" in text
    assert "run_status: completed" in text          # it checks the marker, not just the exit code
    assert len([l for l in text.splitlines() if l.strip()]) < 30


# --- running it ----------------------------------------------------------------

def _run(script: Path, *, block_md_templates: Path) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(block_md_templates)
    return subprocess.run([sys.executable, script.name], cwd=str(script.parent),
                          capture_output=True, text=True, timeout=1800, env=environment)


@pytest.fixture(scope="module")
def without_md_templates(tmp_path_factory) -> Path:
    """A directory whose `md_templates` raises, so a stage that needs it cannot pass."""
    blocker = tmp_path_factory.mktemp("blocked")
    (blocker / "md_templates.py").write_text(
        'raise ImportError("md_templates is deliberately unavailable in this test")')
    return blocker


@pytest.mark.slow
def test_a_stage_runs_with_md_templates_unavailable(generated, without_md_templates):
    """The claim, tested the only way it can be: make the import fail and run anyway."""
    result = _run(generated / "min" / "min.py", block_md_templates=without_md_templates)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "run_status: completed" in result.stdout
    assert (generated / "min" / "min.state.xml").is_file()


@pytest.mark.slow
def test_the_out_header_is_versioned_and_parses_as_key_value(generated, without_md_templates):
    result = _run(generated / "min" / "min.py", block_md_templates=without_md_templates)
    lines = result.stdout.splitlines()
    assert lines[0] == "MD-OPENMM OUTPUT VERSION: 1"
    header = {}
    for line in lines[1:]:
        if not line.strip():
            break
        key, _, value = line.partition(": ")
        header[key] = value
    for field in ("stage", "system", "openmm_version", "platform", "integrator",
                  "temperature_K", "timestep_fs", "final_state"):
        assert field in header, field
    assert header["stage"] == "min" and header["system"] == "ALA"


@pytest.mark.slow
def test_a_failed_stage_leaves_no_completion_marker(generated, without_md_templates, tmp_path):
    """The marker is written last, after the state exists. A crash must not print it."""
    broken = tmp_path / "broken"
    broken.mkdir()
    text = (generated / "min" / "min.py").read_text().replace(
        'COMMON = HERE / ".." / "common"', 'COMMON = HERE / "does-not-exist"')
    (broken / "min.py").write_text(text)
    result = _run(broken / "min.py", block_md_templates=without_md_templates)
    assert result.returncode != 0
    assert "run_status: completed" not in result.stdout


@pytest.mark.slow
def test_the_ligand_route_generates_from_a_smiles_file(tmp_path):
    """phenol/IPH through Sage + AM1-BCC. The peptide example differs only in `type` and input.

    Marked slow because it charges the molecule; phenol is seven heavy atoms, so AM1-BCC finishes
    in seconds rather than the minutes a larger solute would take.
    """
    smiles = tmp_path / "phenol_IPH.smi"
    smiles.write_text("Oc1ccccc1 IPH\n")
    resolved = resolve(_request(system="phenol-IPH", type="ligand", input=str(smiles)))
    result = generate(resolved, input_path=smiles, output_root=tmp_path / "out", echo=False)

    system_dir = result["system_dir"]
    forcefield = yaml.safe_load((system_dir / "common" / "resolved_sys.config.yaml").read_text())
    assert forcefield["solute"]["peptide"] is False
    assert forcefield["solute"]["ligand_forcefield"] == "sage-2.2.1"
    # No protein force field is claimed on the ligand route.
    assert not (forcefield.get("forcefield") or {}).get("protein")
    for script in _stage_scripts(system_dir):
        assert "md_templates" not in script.read_text()
