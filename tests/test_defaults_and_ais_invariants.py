"""Scientific defaults and AIS invariants that outlived contract v1.

This file was `test_md_data_contract.py`, and most of it tested the v1 dataset contract, the
embedded `dataset:` block and the `sys-gen`/`md-gen` route -- all removed. What survives is what was
never about the contract at all:

  * the force fields the defaults ACTUALLY load, checked against a built System rather than against
    the label in a configuration;
  * the AIS invariants that hold whatever generates the run -- the source tau must be established
    rather than assumed, the runtime must stream the source instead of loading it whole, and a
    prepared source must not claim to hold velocities a DCD cannot carry.

Contract v2 and registration are tested in test_registration.py; AIS through the public command is
tested in test_ais_build_md.py.
"""
from __future__ import annotations

import contextlib
import csv
import os
import shutil
import struct
import subprocess
import sys

import pytest
import yaml


from .conftest import ALA_PDB, REPO_ROOT




def _roots(tmp_path, namespace="my-project", name="ALA-explicit"):
    """A synthetic $MD_DATA. Never a real one: nothing here may point at production storage."""
    root = tmp_path / "MD_DATA"
    local = root / namespace / current_month() / name
    local.mkdir(parents=True)
    return root, local


def test_no_fixture_hardcodes_a_dated_path_segment():
    """The regression guard for a bug that only fires on the first of a month.

    Every fixture here stamps `created_at` from the clock, and the contract requires the dated path
    segment to equal that month. A literal `2026-08` therefore passes for a few weeks and then
    fails at midnight on the 1st, in 24 tests at once, with nothing in the diff to explain it --
    which is exactly what happened.

    Asserting "no literal month appears" is what keeps the fix from being undone by the next person
    who writes a fixture by copying an existing one.
    """
    import ast
    import re
    from pathlib import Path

    pattern = re.compile(r"\b20\d\d-(0[1-9]|1[0-2])\b")
    offenders = []
    for path in (Path(__file__), Path(__file__).with_name("test_template_provenance.py")):
        source = path.read_text(encoding="utf-8")
        # Only string LITERALS that reach the code -- prose in a docstring or comment describing
        # the bug is not the bug, and a guard that cannot tell the difference gets deleted.
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = node.body[0] if node.body else None
                if (isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
                        and isinstance(doc.value.value, str)):
                    docstrings.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.lineno not in docstrings and pattern.search(node.value)):
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "a dated path segment is hardcoded; derive it with current_month() instead:\n  "
        + "\n  ".join(offenders))


# --- the manifest, against MD-data's own validator ---------------------------------------------





















# --- the preflight, without dynamics ------------------------------------------------------------

def _preflight():
    from .conftest import template_module

    return template_module("preflight")




def test_preflight_never_walks_the_storage_root_or_hashes_a_trajectory():
    """A bounded check is a safety property: pointing a recursive tool at production storage is
    the accident this must not enable."""
    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
              / "preflight.py").read_text()
    for forbidden in ("rglob", "os.walk", "glob.glob", "iterdir()"):
        assert forbidden not in source, f"preflight uses {forbidden}"
    # The only hashing it does is over the files SHA256SUMS names -- small preparation records,
    # never a production trajectory.
    assert "SHA256SUMS" in source
    assert ".dcd" not in source




# --- AIS source tau -----------------------------------------------------------------------------

def test_an_explicit_source_tau_is_required_when_there_is_no_companion_record():
    """An external trajectory has no runtime record, so the user has to say what it is."""
    from md_tools.openmm.config import ConfigError, resolve_md_config
    from md_tools.openmm.defaults import md_defaults

    document = md_defaults(methods=["AIS"])
    document["AIS"]["path"]["switching_duration_ps"] = 0.08
    document["AIS"]["source"].update({"trajectory": "external.dcd", "start_time_ps": 0.0,
                                      "end_time_ps": 10.0, "number_of_trajectories": 1,
                                      "first_frame_time_ps": 0.0, "frame_interval_ps": 0.5})
    # Valid without it: the runtime decides whether a companion record supplies it.
    resolve_md_config(document, implicit=False)

    document["AIS"]["source"]["source_tau"] = 0.25
    with pytest.raises(ConfigError, match="source_tau"):
        resolve_md_config(document, implicit=False)


def test_a_declared_source_tau_must_equal_the_path_start():
    from md_tools.openmm.config import ConfigError, resolve_md_config
    from md_tools.openmm.defaults import md_defaults

    document = md_defaults(methods=["AIS"])
    document["AIS"]["path"]["switching_duration_ps"] = 0.08
    document["AIS"]["source"].update({"trajectory": "external.dcd", "start_time_ps": 0.0,
                                      "end_time_ps": 10.0, "number_of_trajectories": 1,
                                      "source_tau": 0.5})
    resolved = resolve_md_config(document, implicit=False)
    assert resolved["AIS"]["source"]["source_tau"] == 0.5

    document["AIS"]["source"]["source_tau"] = 1.5
    with pytest.raises(ConfigError, match=r"in \[0, 1\)"):
        resolve_md_config(document, implicit=False)


# --- streamed AIS sources -----------------------------------------------------------------------

def test_the_ais_runtime_never_calls_mdtraj_load():
    """`mdtraj.load` reads an entire trajectory into memory. An AIS source is a production run."""
    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
              / "ais_run.py").read_text()
    import re

    calls = re.findall(r"mdtraj\.load\s*\(", source)
    assert not calls, "the AIS runtime calls mdtraj.load"
    assert "mdtraj.iterload" in source


def test_the_plan_json_carries_no_coordinates():
    """A JSON array of a solvated system's coordinates is enormous, and every worker reads it."""
    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
              / "ais_run.py").read_text()
    assert '"positions_nm"' not in source
    # Coordinates live in the prepared inputs; the plan carries only the index into them.
    assert '"dcd_frame_index"' in source and "SOURCES_DCD" in source


def test_the_prepared_inputs_never_claim_to_hold_velocities():
    """A directory of starting configurations is where someone would look for velocities."""
    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
              / "ais_run.py").read_text()
    assert '"stored": False' in source
    assert "NOT restart states" in source
    # The momenta are generated from a recorded seed, and that is what the record says.
    assert "setVelocitiesToTemperature(TEMPERATURE, int(entry[\"velocity_seed\"]))" in source


# --- the default force-field selection ----------------------------------------------------------

def test_the_generated_default_names_ff14sb_sage_221_and_tip3p():
    from md_tools.openmm.defaults import sys_defaults

    document = sys_defaults()
    assert document["forcefield"]["protein"] == "amber14-all.xml"
    assert document["forcefield"]["water"] == "amber14/tip3p.xml"
    assert document["solute"]["ligand_forcefield"] == "sage-2.2.1"


def test_no_cuda_acceptance_fixture_uses_the_optional_selection():
    """OPC stays a documented option, and stays out of the fixtures that run dynamics."""
    # Stated precisely, and derived rather than kept by hand: no test MARKED `gpu` may select the
    # optional combination. A hand-kept list of filenames went stale the moment a file was renamed
    # or removed; scanning for the marker cannot.
    import ast

    offenders = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            marks = {ast.unparse(d) for d in node.decorator_list}
            if not any("gpu" in mark for mark in marks):
                continue
            body = ast.unparse(node)
            if 'model": "OPC"' in body or "'OPC'" in body or '"OPC"' in body \
                    or "tip3pfb" in body.lower():
                offenders.append(f"{path.name}::{node.name}")
    assert not offenders, (
        f"these CUDA tests select the optional ff19SB/OPC combination: {offenders}. The CUDA lane "
        f"runs the combination users actually get.")


# --- CUDA acceptance: the default selection, and only the default selection ---------------------

@pytest.fixture(scope="module")
def peptide_default(tmp_path_factory):
    """A tiny ff14SB + TIP3P peptide system, built with NO configuration at all.

    Through the real `build-top`, because the point of the test is what a user who types the least
    actually gets. Small box: the force fields are what is under test, not the solvent count.
    """
    work = tmp_path_factory.mktemp("peptide-default")
    (work / "small.config").write_text(
        "solvent:\n  padding_nm: 0.5\n  cutoff_nm: 0.6\n", encoding="utf-8")
    built = _build_top(work, ALA_PDB, "small.config")
    assert built.returncode == 0, built.stdout + built.stderr
    return work


@pytest.fixture(scope="module")
def ligand_default(tmp_path_factory):
    """A tiny Sage 2.2.1 + TIP3P ligand system. Ethanol: small enough that AM1-BCC is seconds."""
    work = tmp_path_factory.mktemp("ligand-default")
    (work / "ligand.smi").write_text("CCO ETH\n", encoding="utf-8")
    (work / "small.config").write_text(
        "solute:\n  peptide: false\nsolvent:\n  padding_nm: 0.5\n  cutoff_nm: 0.6\n",
        encoding="utf-8")
    built = _build_top(work, work / "ligand.smi", "small.config")
    assert built.returncode == 0, built.stdout + built.stderr
    return work


def _build_top(work, structure, config):
    """Invoke the real command; nothing here reaches past the public interface."""
    import subprocess
    import sys as _sys

    return subprocess.run(
        [_sys.executable, "-m", "md_tools.cli.md_openmm", "build-top",
         "-i", str(structure), "-os", "built.xml", "-op", "built.pdb", "-log", "built.log",
         "--config", str(config)],
        cwd=work, capture_output=True, text=True, timeout=1800)




@pytest.mark.slow
def test_the_peptide_default_actually_loaded_ff14sb_and_tip3p(peptide_default):
    """From the BUILT record, not from sys.config.yaml.

    The configuration says what was requested; the build record says what was LOADED. Only the
    second one describes the Hamiltonian a trajectory belongs to.

    `forcefield.json` and the route that wrote it are retired; `build-top` carries the same record
    in built.log, which is where every other fact about the build now lives.
    """
    from md_tools.build.record import read_record

    record = read_record(peptide_default / "built.log")["forcefield_record"]
    assert record["route"] == "peptide"
    assert record["protein"]["openmm_resource"] == "amber14-all.xml"
    assert "amber14/protein.ff14SB.xml" in record["protein"]["openmm_resource_includes"]
    assert record["water"]["openmm_resource"] == "amber14/tip3p.xml"
    assert record["water"]["model"] == "TIP3P"
    assert record["builder"]["openmm_xml_loaded"] == ["amber14-all.xml", "amber14/tip3p.xml"]
    # Sage parameterised nothing here, and the record says so rather than naming it.
    assert record["ligand"]["openff_resource"] is None
    assert record["ligand"]["forcefield"] is None


@pytest.mark.slow
def test_the_ligand_default_actually_loaded_sage_221_and_tip3p(ligand_default):
    """The other half of the default selection, on the route that actually uses Sage.

    The generator builds a peptide route or a ligand route, not a protein-ligand complex carrying
    both parameter sets. So Sage is proved here, on a ligand, and is NOT claimed in the peptide
    fixture above.
    """
    from md_tools.build.record import read_record

    record = read_record(ligand_default / "built.log")["forcefield_record"]
    assert record["route"] == "ligand"
    assert record["ligand"]["openff_resource"] == "openff-2.2.1"
    assert record["ligand"]["requested_label"] == "sage-2.2.1"
    assert record["ligand"]["charge_method"] == "am1bcc"
    assert record["water"]["openmm_resource"] == "amber14/tip3p.xml"
    # ...and no protein force field was loaded, because none parameterised anything.
    assert record["protein"]["openmm_resource"] is None
    import json as _json
    assert "amber19" not in _json.dumps(record), "no OPC/ff19SB anywhere in a default build"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("fixture, expect", [
    ("peptide_default", "amber14-all.xml"),
    ("ligand_default", "openff-2.2.1"),
])
def test_the_default_selection_minimises_on_cuda(request, fixture, expect):
    """Both halves of the default selection integrate, on CUDA, with no fallback."""
    from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    work = request.getfixturevalue(fixture)
    system = XmlSerializer.deserialize((work / "built.xml").read_text())
    pdb = PDBFile(str(work / "built.pdb"))

    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(pdb.topology, system,
                            LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                                     2.0 * unit.femtoseconds),
                            platform, {"Precision": "mixed"})
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=25)
    simulation.step(20)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert simulation.context.getPlatform().getName() == "CUDA"
    assert energy.value_in_unit(unit.kilojoule_per_mole) == energy.value_in_unit(
        unit.kilojoule_per_mole)                      # not NaN
    import json

    from md_tools.build.record import read_record
    import json as _json
    assert expect in _json.dumps(read_record(work / "built.log")["forcefield_record"])


# --- a contract-managed dataset, end to end -----------------------------------------------------

@contextlib.contextmanager
def clean_checkout():
    """Present this checkout's raw identity as clean, without touching the working tree.

    The RAW observation is what is faked -- `implementation_identity()` -- so the resolution in
    `template_identity()` and every comparison downstream is the real one. The commit is this
    repository's actual HEAD, which is what `_dataset_block` claims, so the records still have to
    agree with each other for anything to pass.
    """
    from md_tools.openmm import provenance_min

    real = provenance_min.implementation_identity

    def clean():
        observed = dict(real())
        observed["git_commit"] = GENERATOR_COMMIT
        observed["git_dirty"] = False
        return observed

    patch = pytest.MonkeyPatch()
    patch.setattr(provenance_min, "implementation_identity", clean)
    try:
        yield
    finally:
        patch.undo()


def _generate_in_process(build, local, *, methods, stage):
    """`sys-gen` or `md-gen`, called directly so the patched identity applies."""
    from md_tools.openmm import mdgen, sysgen

    environment = dict(os.environ)
    os.environ.update({"MD_DATA": str(local.parents[2]), "MD_DATA_LOCAL": str(local)})
    try:
        if stage == "system":
            sysgen.generate_system(input_path=build / "ALA.pdb",
                                   config_path=build / "sys.config.yaml",
                                   output_folder=local / "common", echo=False)
        else:
            mdgen.generate_md(input_folder=local / "common",
                              config_path=build / "md.config.yaml", output_folder=local)
    finally:
        os.environ.clear()
        os.environ.update(environment)




def _launch(directory, script, environment, *args):
    return subprocess.run(["bash", script, *args], cwd=str(directory), capture_output=True,
                          text=True, env=environment, timeout=1800)
















# --- prepared AIS starting configurations -------------------------------------------------------







