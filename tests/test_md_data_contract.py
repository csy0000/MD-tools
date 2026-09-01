"""The MD-data v1 contract, the execution preflight, and streamed AIS sources.

Split by cost, as elsewhere:

    (unmarked)  manifests, validation, generated source. Nothing is built and nothing integrates.
    slow        one system built, one project generated.
    gpu         energies or dynamics. CUDA, and nowhere else.

The contract itself is validated by MD-data's OWN validator, imported. There is no copy of its
schema here and no test asserts against one: a second implementation of a contract is one that
drifts, and these tests would drift with it.
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

from md_tools.openmm import md_data_contract as MD

from .conftest import ALA_PDB, REPO_ROOT, run_cli
import datetime

md_data = pytest.importorskip("md_data", reason="the authoritative MD-data validator")

#: A pinned 40-hex commit for the synthetic fixtures. Not a real commit, and never derived from
#: one: the point of the field is that a caller supplies it.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


#: The commit this MD-tools can actually establish for itself. A fixture that claimed an
#: arbitrary 40-hex value would be refused now, and rightly: `templates.commit` must be the commit
#: that generated the dataset, not a syntactically valid string.
GENERATOR_COMMIT = MD.generator_commit()["commit"]


#: The dated path segment is the dataset's VERSION, and the contract requires it to equal the month
#: of `created_at` (MD-data `dataset_contract.py`: "it cannot disagree with when the dataset was
#: created"). A literal month here is a time bomb: these fixtures stamp `created_at` from the clock,
#: so every one of them started failing at 00:00 on the first of the month. Derive it from the same
#: clock the manifest is stamped from, and the fixture is correct on any day.
def current_month():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def _dataset_block(namespace="my-project", name="ALA-explicit", role="project"):
    return {
        "enabled": True,
        "dataset_id": f"{namespace}-{current_month()}-{name}".lower(),
        "namespace": namespace,
        "dataset_name": name,
        "role": role,
        "system": "Synthetic fixture: alanine dipeptide in TIP3P water.",
        "created_by": {"person_id": "shu-yu-chen", "name": "Shu-Yu Chen",
                       "affiliation": "University of Basel", "orcid": None},
        "origin": {"repository": "https://github.com/csy0000/example-project",
                   "commit": FAKE_COMMIT},
        "templates": {"repository": "https://github.com/csy0000/MD-tools",
                      "commit": GENERATOR_COMMIT or FAKE_COMMIT},
        "derived_from": [],
        "notes": None,
    }


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

def test_a_generated_manifest_passes_the_authoritative_validator(tmp_path):
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "common").mkdir()
    location = MD.resolve_roots(local / "common",
                                environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
    manifest = MD.build_manifest(
        dataset_block=_dataset_block(), location=location,
        components=[MD.component_entry("common", kind="shared-input")],
        templates_version="0.4.0.dev0")

    # MD-data's validator, not a local reimplementation of its rules. `--root` additionally checks
    # the declared layout, which needs the manifest to be on disk where it says it is.
    MD.write_manifest(local, manifest)
    report = MD.validate(manifest, root=str(root))
    assert report["dataset_id"] == f"my-project-{current_month()}-ala-explicit"
    assert report["role"] == "project" and report["status"] == "active"
    assert report["read_only"] is False
    assert report["validator"]["package"] == "md-data"
    assert report["validator"]["contract_version"] == md_data.CONTRACT_VERSION

    # ...and the path is relative, always. An absolute machine path in a manifest stops being true
    # the moment the tree moves, which is the whole reason $MD_DATA is a variable.
    assert manifest["path"] == f"my-project/{current_month()}/ALA-explicit"
    assert not os.path.isabs(manifest["path"])
    assert str(tmp_path) not in yaml.safe_dump(manifest)


def test_the_manifest_the_cli_writes_passes_md_datas_own_command(tmp_path):
    """End to end through the published CLI, which is what a user would run."""
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "common").mkdir()
    location = MD.resolve_roots(local / "common",
                                environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
    manifest = MD.build_manifest(
        dataset_block=_dataset_block(), location=location,
        components=[MD.component_entry("common", kind="shared-input")],
        templates_version="0.4.0.dev0")
    MD.write_manifest(local, manifest)

    result = subprocess.run([sys.executable, "-m", "md_data", "validate-dataset",
                             str(local / "dataset.yaml"), "--root", str(root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize("patch, expected", [
    ({"dataset_id": None}, "dataset_id"),
    ({"namespace": None}, "namespace"),
    ({"role": None}, "role"),
    ({"system": None}, "system"),
    ({"created_by": {"person_id": None, "name": "X"}}, "person_id"),
    ({"origin": {"repository": "https://x", "commit": None}}, "commit"),
    ({"origin": {"repository": "https://x", "commit": "abc1234"}}, "not an exact 40-hex commit"),
    ({"templates": {"repository": "https://x", "commit": "main"}}, "not an exact 40-hex commit"),
])
def test_a_field_this_package_must_not_guess_fails_by_name(tmp_path, patch, expected):
    """A version or an installed fingerprint is provenance; it is not a pin."""
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "common").mkdir()
    location = MD.resolve_roots(local / "common",
                                environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
    block = _dataset_block()
    block.update(patch)
    with pytest.raises(MD.ContractError) as error:
        MD.build_manifest(dataset_block=block, location=location,
                          components=[MD.component_entry("common", kind="shared-input")],
                          templates_version="0.4.0.dev0")
    assert expected in str(error.value)


@pytest.mark.parametrize("namespace, name, role, expected", [
    ("my-project", "ALA-explicit", "baseline", "reserved `baseline` namespace"),
    ("baseline", "ALA-explicit", "project", "reserved `baseline` namespace"),
])
def test_role_and_namespace_must_agree(tmp_path, namespace, name, role, expected):
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path, namespace=namespace, name=name)
    (local / "common").mkdir()
    location = MD.resolve_roots(local / "common",
                                environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
    block = _dataset_block(namespace=namespace, name=name, role=role)
    with pytest.raises(MD.ContractError) as error:
        MD.build_manifest(dataset_block=block, location=location,
                          components=[MD.component_entry("common", kind="shared-input")],
                          templates_version="0.4.0.dev0")
    assert expected in str(error.value)


def test_the_declared_identity_must_agree_with_where_the_dataset_actually_is(tmp_path):
    """The location on disk is the truth, and the manifest has to match it."""
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "common").mkdir()
    location = MD.resolve_roots(local / "common",
                                environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
    block = _dataset_block()
    block["dataset_name"] = "something-else"
    with pytest.raises(MD.ContractError, match="MD_DATA_LOCAL ends in"):
        MD.build_manifest(dataset_block=block, location=location,
                          components=[MD.component_entry("common", kind="shared-input")],
                          templates_version="0.4.0.dev0")


@pytest.mark.parametrize("environ, expected", [
    ({}, "MD_DATA is not set"),
    ({"MD_DATA": "__root__"}, "MD_DATA_LOCAL is not set"),
    ({"MD_DATA": "__root__", "MD_DATA_LOCAL": "/tmp"}, "not inside MD_DATA"),
    ({"MD_DATA": "__root__", "MD_DATA_LOCAL": "__root__/flat"}, "three segments"),
    ({"MD_DATA": "__root__", "MD_DATA_LOCAL": "__root__/ns/not-a-month/name"}, "not a real"),
])
def test_a_root_that_is_not_the_canonical_dataset_path_is_refused(tmp_path, environ, expected):
    from md_tools.openmm import md_data_contract as MD

    root = tmp_path / "MD_DATA"
    root.mkdir()
    resolved = {k: v.replace("__root__", str(root)) for k, v in environ.items()}
    target = resolved.get("MD_DATA_LOCAL")
    if target and target.startswith(str(root)):
        from pathlib import Path

        Path(target).mkdir(parents=True, exist_ok=True)
        output = Path(target) / "common"
        output.mkdir(exist_ok=True)
    else:
        output = tmp_path / "somewhere" / "common"
        output.mkdir(parents=True)
    with pytest.raises(MD.ContractError) as error:
        MD.resolve_roots(output, environ=resolved)
    assert expected in str(error.value)


def test_a_symlinked_dataset_root_is_refused_as_an_alias(tmp_path):
    """A symlink into another dataset is an ALIAS, and an alias resolves to somebody else's
    authoritative manifest. New output belongs in a dataset this project owns."""
    from md_tools.openmm import md_data_contract as MD

    root = tmp_path / "MD_DATA"
    real = root / "owner" / current_month() / "ALA"
    real.mkdir(parents=True)
    alias_parent = root / "consumer" / current_month()
    alias_parent.mkdir(parents=True)
    alias = alias_parent / "ALA"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(MD.ContractError, match="alias"):
        MD.resolve_roots(alias / "common",
                         environ={"MD_DATA": str(root), "MD_DATA_LOCAL": str(alias)})


def test_a_complete_dataset_is_never_overwritten(tmp_path):
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "dataset.yaml").write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "dataset_id": f"my-project-{current_month()}-ala-explicit",
        "path": f"my-project/{current_month()}/ALA-explicit", "namespace": "my-project",
        "dataset_name": "ALA-explicit", "role": "project", "status": "complete"}))
    with pytest.raises(MD.ContractError, match="read-only"):
        MD.write_manifest(local, {"dataset_id": f"my-project-{current_month()}-ala-explicit",
                                  "path": f"my-project/{current_month()}/ALA-explicit",
                                  "namespace": "my-project", "dataset_name": "ALA-explicit",
                                  "role": "project", "status": "active"})


def test_a_different_identity_never_replaces_an_existing_manifest(tmp_path):
    """Everything that referenced the old ID would silently resolve to something else."""
    from md_tools.openmm import md_data_contract as MD

    root, local = _roots(tmp_path)
    (local / "dataset.yaml").write_text(yaml.safe_dump({
        "schema_version": "1.0", "dataset_id": "someone-elses-dataset",
        "path": f"my-project/{current_month()}/ALA-explicit", "namespace": "my-project",
        "dataset_name": "ALA-explicit", "role": "project", "status": "active"}))
    with pytest.raises(MD.ContractError, match="identity of an existing dataset"):
        MD.write_manifest(local, {"dataset_id": f"my-project-{current_month()}-ala-explicit",
                                  "path": f"my-project/{current_month()}/ALA-explicit",
                                  "namespace": "my-project", "dataset_name": "ALA-explicit",
                                  "role": "project", "status": "active"})


def test_equilibration_stages_are_not_declared_as_components():
    """`eq/nvt_1kcal` is a stage INSIDE the `eq` component.

    Declaring one component per stage would turn one equilibration into four datasets, which is
    exactly the nesting the contract forbids.
    """
    from md_tools.openmm.mdgen import COMPONENT_OF_METHOD
    from md_tools.openmm.templates import __name__ as _   # noqa: F401

    assert set(COMPONENT_OF_METHOD) == {"cMD", "REST2", "AIS"}
    assert all("/" not in name for name in COMPONENT_OF_METHOD.values())


# --- the preflight, without dynamics ------------------------------------------------------------

def _preflight():
    from .conftest import template_module

    return template_module("preflight")


def test_an_unregistered_project_says_so_rather_than_pretending(tmp_path):
    preflight = _preflight()
    rows = preflight.check_dataset(tmp_path, tmp_path, {"dataset": {"contract_managed": False}})
    assert len(rows) == 1
    assert rows[0].status == preflight.SKIP
    assert "not MD-data compliant" in rows[0].detail


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


@pytest.mark.parametrize("directory, component", [
    ("minimization", "minimization"),
    ("eq/nvt_1kcal", "eq"),
    ("eq/npt_free", "eq"),
    ("cMD", "cMD"),
    ("REST2", "REST2"),
    ("AIS", "AIS"),
])
def test_each_generated_directory_maps_to_its_declared_component(tmp_path, directory, component):
    preflight = _preflight()
    here = tmp_path / directory
    here.mkdir(parents=True)
    assert preflight._component_for(here, tmp_path) == component


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
    for name in ("test_stage_layout.py", "test_system_generation.py", "test_md_generation.py",
                 "test_ais.py"):
        text = (REPO_ROOT / "tests" / name).read_text()
        assert 'solvent="OPC"' not in text, name
        assert '"--solvent", "OPC"' not in text, name
        assert "tip3pfb" not in text.lower(), name


# --- CUDA acceptance: the default selection, and only the default selection ---------------------

@pytest.fixture(scope="module")
def peptide_default(tmp_path_factory):
    """A tiny ff14SB + TIP3P peptide system, built with no `--solvent` flag at all."""
    work = tmp_path_factory.mktemp("peptide-default")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    assert run_cli("md_openmm", "sys-config", "--method", "cMD", cwd=work).returncode == 0
    path = work / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr
    return work


@pytest.fixture(scope="module")
def ligand_default(tmp_path_factory):
    """A tiny Sage 2.2.1 + TIP3P ligand system. Ethanol: small enough that AM1-BCC is seconds."""
    work = tmp_path_factory.mktemp("ligand-default")
    (work / "ligand.smi").write_text("CCO\n")
    assert run_cli("md_openmm", "sys-config", "--method", "cMD", "--peptide", "false",
                   cwd=work).returncode == 0
    path = work / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ligand.smi", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr
    return work


@pytest.mark.slow
def test_the_peptide_default_actually_loaded_ff14sb_and_tip3p(peptide_default):
    """From the BUILT record, not from sys.config.yaml.

    The configuration says what was requested; `forcefield.json` says what was loaded. Only the
    second one describes the Hamiltonian a trajectory belongs to.
    """
    import json

    record = json.loads((peptide_default / "inputs" / "forcefield.json").read_text())
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
    import json

    record = json.loads((ligand_default / "inputs" / "forcefield.json").read_text())
    assert record["route"] == "ligand"
    assert record["ligand"]["openff_resource"] == "openff-2.2.1"
    assert record["ligand"]["requested_label"] == "sage-2.2.1"
    assert record["ligand"]["charge_method"] == "am1bcc"
    assert record["water"]["openmm_resource"] == "amber14/tip3p.xml"
    # ...and no protein force field was loaded, because none parameterised anything.
    assert record["protein"]["openmm_resource"] is None
    assert "amber19" not in json.dumps(record), "no OPC/ff19SB anywhere in a default build"


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
    inputs = work / "inputs"
    system = XmlSerializer.deserialize((inputs / "system.xml").read_text())
    state = XmlSerializer.deserialize((inputs / "initial_state.xml").read_text())
    pdb = PDBFile(str(inputs / "topology.pdb"))

    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(pdb.topology, system,
                            LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                                     2.0 * unit.femtoseconds),
                            platform, {"Precision": "mixed"})
    simulation.context.setState(state)
    simulation.minimizeEnergy(maxIterations=25)
    simulation.step(20)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert simulation.context.getPlatform().getName() == "CUDA"
    assert energy.value_in_unit(unit.kilojoule_per_mole) == energy.value_in_unit(
        unit.kilojoule_per_mole)                      # not NaN
    import json

    assert expect in json.dumps(json.loads((inputs / "forcefield.json").read_text()))


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


@pytest.fixture(scope="module")
def managed(tmp_path_factory):
    """A contract-managed dataset with a multi-chunk AIS source, generated but not yet run."""
    work = tmp_path_factory.mktemp("managed")
    root = work / "MD_DATA"
    local = root / "proj" / current_month() / "ALA"
    local.mkdir(parents=True)
    build = work / "build"
    build.mkdir()
    shutil.copy2(ALA_PDB, build / "ALA.pdb")
    environment = dict(os.environ, MD_DATA=str(root), MD_DATA_LOCAL=str(local))

    def cli(*args):
        # Through conftest's run_cli, which routes the three retired subcommand names to the
        # generator API they used to reach. This fixture is exercising the generator and the
        # MD-data contract shape, not the argument parser.
        return run_cli("md_openmm", *args, cwd=build)

    assert cli("sys-config", "--method", "cMD", "AIS").returncode == 0
    path = build / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    block = _dataset_block(namespace="proj", name="ALA")
    document["dataset"] = block
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    # Generation runs IN-PROCESS with the raw identity observation presented as a clean checkout
    # of the real HEAD. Contract-managed generation refuses a dirty working tree by design, and a
    # development checkout is always dirty -- so a subprocess CLI call could never build this
    # fixture. Only the two generation calls change; every stage below still runs through its own
    # `run.sh` exactly as a user's would.
    with clean_checkout():
        _generate_in_process(build, local, methods=("cMD", "AIS"), stage="system")

    protocol_path = build / "md.config.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.02
    # 120 source frames at 0.002 ps spans three chunks of SOURCE_CHUNK_FRAMES = 50, so a
    # one-chunk implementation cannot pass this by accident.
    protocol["cMD"].update({"tau": 0.5, "duration_ns": 0.00024, "checkpoint_interval_ps": 0.24,
                            "whole_system_interval_ps": 0.002, "solute_interval_ps": 0.002})
    protocol["AIS"]["path"]["switching_duration_ps"] = 0.08
    protocol["AIS"]["source"].update({"trajectory": "cMD/whole_system.dcd",
                                      "start_time_ps": 0.002, "end_time_ps": 0.24,
                                      "number_of_trajectories": 2})
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    with clean_checkout():
        _generate_in_process(build, local, methods=("cMD", "AIS"), stage="md")
    return {"root": root, "local": local, "env": environment}


def _launch(directory, script, environment, *args):
    return subprocess.run(["bash", script, *args], cwd=str(directory), capture_output=True,
                          text=True, env=environment, timeout=1800)


@pytest.mark.slow
def test_the_generated_dataset_has_the_contract_shape(managed):
    local = managed["local"]
    components = {c["name"] for c in
                  yaml.safe_load((local / "dataset.yaml").read_text())["components"]}
    assert components == {"common", "minimization", "eq", "cMD", "AIS"}
    # The equilibration stages are directories INSIDE the eq component, not components.
    assert (local / "eq" / "nvt_1kcal").is_dir()
    assert not any(c.startswith("eq/") for c in components)
    assert (local / "dataset.yaml").is_file()
    # Exactly one authoritative manifest, at the dataset root.
    assert [p for p in local.rglob("dataset.yaml")] == [local / "dataset.yaml"]


@pytest.mark.slow
def test_generated_scripts_are_portable_and_carry_no_storage_root(managed):
    """Moving the whole tree and re-pointing MD_DATA must be enough to rerun it.

    Scope: the scripts and the files a rerun reads. NOT `provenance.yaml`, which records the
    command line verbatim -- the `-of` argument the user actually typed was absolute, and
    rewriting it to look relative would falsify the one record whose job is to say what happened.
    """
    import re

    local, root = managed["local"], managed["root"]
    offenders = {}
    for path in sorted(local.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".yaml"):
            continue
        if path.name == "provenance.yaml" or path.is_relative_to(local / "common"
                                                                 / "original_inputs"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(local))
        if str(root) in text:
            offenders[relative] = "contains the absolute MD_DATA root"
        if path.suffix == ".py" and re.search(r"^\s*(import|from)\s+md_tools\b", text,
                                              re.MULTILINE):
            offenders[relative] = "imports md_tools"
        if str(REPO_ROOT) in text:
            offenders[relative] = "names this checkout"
    assert not offenders, offenders

    # The manifest's own path field is relative to $MD_DATA, which is the contract's rule.
    manifest = yaml.safe_load((local / "dataset.yaml").read_text())
    assert manifest["path"] == f"proj/{current_month()}/ALA"
    assert not manifest["path"].startswith("/")


@pytest.mark.gpu
@pytest.mark.slow
def test_check_runs_the_whole_preflight_and_creates_nothing(managed):
    """`run_all.sh --check` is the same gate a real run runs, on its own and with zero steps."""
    local, environment = managed["local"], managed["env"]
    before = {p for p in local.rglob("*") if p.is_file()}

    result = _launch(local, "run_all.sh", environment, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight only" in result.stdout
    assert "md-data validator" in result.stdout

    after = {p for p in local.rglob("*") if p.is_file()}
    # Python's bytecode cache is unavoidable and is not an output of the run; everything else is.
    created = sorted(str(p.relative_to(local)) for p in after - before
                     if "__pycache__" not in p.parts)
    assert not created, f"--check created {created}"
    # ...and specifically none of the artifacts a run would make.
    for pattern in ("*.dcd", "*.chk", "final_state.xml", "resolved_stage.yaml", "resolved_run.yaml",
                    "stage.csv", "observations.csv"):
        assert not list(local.rglob(pattern)), pattern


@pytest.mark.gpu
@pytest.mark.slow
def test_an_invalid_contract_fails_before_any_context_exists(managed, tmp_path):
    """The manifest is broken, the run stops, and nothing at all is written."""
    local, environment = managed["local"], managed["env"]
    manifest = local / "dataset.yaml"
    original = manifest.read_text()
    document = yaml.safe_load(original)
    document["owner"] = "nobody"                     # the contract forbids undeclared fields
    manifest.write_text(yaml.safe_dump(document, sort_keys=False))
    before = {p for p in local.rglob("*") if p.is_file()}
    try:
        result = _launch(local / "minimization", "run.sh", environment)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "md-data validator" in combined and "FAIL" in combined
        assert "unknown key" in combined
        assert "no Context was created" in combined
        # The stage's own run log is the record that this attempt happened and why it stopped,
        # so it is written; nothing derived from a Context is.
        after = {p for p in local.rglob("*") if p.is_file()}
        wrote = {str(p.relative_to(local)) for p in after - before
                 if "__pycache__" not in p.parts and p.name != "run.log"}
        assert not wrote, f"a failed preflight wrote {wrote}"
        assert "FAIL" in (local / "minimization" / "run.log").read_text()
    finally:
        manifest.write_text(original)


@pytest.mark.gpu
@pytest.mark.slow
def test_the_ais_source_is_streamed_across_several_chunks(managed):
    """`iterload`, in bounded chunks, and never `mdtraj.load` -- proved by making it fail."""
    local, environment = managed["local"], managed["env"]
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free", "cMD"):
        result = _launch(local / stage, "run.sh", environment)
        assert result.returncode == 0, result.stdout + result.stderr

    source = yaml.safe_load((local / "cMD" / "resolved_run.yaml").read_text())
    assert source["trajectories"]["whole_system"]["frames"] == 120

    # A sitecustomize that makes mdtraj.load raise. If the runtime ever reaches for it the run
    # dies, so an implementation that quietly loaded the whole trajectory cannot pass this. The
    # sentinel is what makes the guard trustworthy: without it a guard that silently failed to
    # install would look exactly like a guard that was never triggered.
    ais = local / "AIS"
    sentinel = ais / "_guard_installed"
    guard = ais / "sitecustomize.py"
    guard.write_text(
        "import pathlib\n"
        "import mdtraj\n"
        "def _forbidden(*a, **k):\n"
        "    raise AssertionError('mdtraj.load was called on the AIS source')\n"
        "mdtraj.load = _forbidden\n"
        "pathlib.Path(__file__).with_name('_guard_installed').write_text('yes')\n")
    try:
        result = subprocess.run(["bash", "run.sh"], cwd=str(ais), capture_output=True, text=True,
                                env=dict(environment, PYTHONPATH=str(ais)), timeout=1800)
        assert sentinel.is_file(), "the guard never installed, so this test proved nothing"
        assert result.returncode == 0, result.stdout + result.stderr
        assert "mdtraj.load was called" not in result.stdout + result.stderr
    finally:
        guard.unlink(missing_ok=True)
        sentinel.unlink(missing_ok=True)
        shutil.rmtree(ais / "__pycache__", ignore_errors=True)

    record = yaml.safe_load((local / "AIS" / "resolved_run.yaml").read_text())["source"]
    assert record["loader"] == "mdtraj.iterload"
    assert record["chunk_frames"] == 50
    # 120 frames / 50 per chunk = 3: more than one, so streaming is genuinely exercised.
    assert record["chunks_read_survey"] >= 3, record
    assert record["n_frames"] == 120
    assert len(record["selected_frame_indices"]) == 2
    assert record["tau"] == 0.5 and record["tau_route"] == "companion record"

    # The plan is scratch and is cleaned up; the prepared inputs are not -- they are what the
    # paths started from.
    assert not (local / "AIS" / "_ais_plan.json").exists()
    assert not (local / "AIS" / "_ais_frames.npz").exists()
    assert (local / "AIS" / "inputs" / "sources.dcd").is_file()
    prepared = yaml.safe_load((local / "AIS" / "inputs" / "sources.yaml").read_text())
    assert prepared["n_paths"] == 2
    assert prepared["velocities"]["stored"] is False
    assert struct.unpack("<i", (local / "AIS" / "inputs" / "sources.dcd")
                         .read_bytes()[8:12])[0] == 2


@pytest.mark.gpu
@pytest.mark.slow
def test_21_dcd_frames_map_to_21_work_rows(managed):
    local = managed["local"]
    for index in range(2):
        directory = local / "AIS" / f"trajectory_{index:04d}"
        raw = (directory / "observations.dcd").read_bytes()[:12]
        frames = struct.unpack("<i", raw[8:12])[0]
        rows = list(csv.DictReader((directory / "observations.csv").open()))
        assert frames == 21 and len(rows) == 21, (index, frames, len(rows))
        assert [int(r["coordinate_frame_index"]) for r in rows] == list(range(21))
        assert float(rows[0]["tau"]) == 0.5
        assert float(rows[0]["cumulative_work_kj_mol"]) == 0.0
        assert float(rows[-1]["tau"]) == 0.0


@pytest.mark.gpu
@pytest.mark.slow
def test_a_truncated_or_missing_dcd_is_never_skipped_as_complete(managed):
    """The case a healthy JSON and CSV would otherwise hide completely."""
    local, environment = managed["local"], managed["env"]
    directory = local / "AIS" / "trajectory_0000"
    dcd = directory / "observations.dcd"
    original = dcd.read_bytes()

    # 1. missing entirely
    dcd.unlink()
    result = _launch(local / "AIS", "run.sh", environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "observations.dcd is missing" in result.stdout
    assert "replacing an incomplete directory" in result.stdout
    assert dcd.is_file()

    # 2. the header and the content disagree. This is NOT the truncation case -- the bytes are
    # all there and only NSET was edited -- and it is caught because the frames are read rather
    # than counted from the header. Genuine byte truncation, where the header is left intact and
    # the coordinates are missing, is in tests/test_integrity_corrections.py.
    raw = bytearray(dcd.read_bytes())
    struct.pack_into("<i", raw, 8, 7)
    dcd.write_bytes(bytes(raw))
    result = _launch(local / "AIS", "run.sh", environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replacing an incomplete directory" in result.stdout
    assert "header claims 7" in result.stdout or "claims 7" in result.stdout
    raw = (directory / "observations.dcd").read_bytes()[:12]
    assert struct.unpack("<i", raw[8:12])[0] == 21, "the rerun did not restore 21 frames"
    assert len(original) == len(dcd.read_bytes()), "a second path was appended to the old DCD"


# --- prepared AIS starting configurations -------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_ais_reruns_after_the_source_trajectory_is_gone(managed):
    """The point of preparing the inputs: an archived source is not a lost calculation."""
    local, environment = managed["local"], managed["env"]
    record = yaml.safe_load((local / "AIS" / "inputs" / "sources.yaml").read_text())
    frames = [int(p["source_frame_index"]) for p in record["paths"]]

    source = local / "cMD" / "whole_system.dcd"
    kept = source.read_bytes()
    shutil.rmtree(local / "AIS" / "trajectory_0000")
    source.unlink()                                    # the source is archived, or simply gone
    try:
        result = _launch(local / "AIS", "run.sh", environment)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "the source trajectory is not needed" in result.stdout
        assert "complete: 21 observations" in result.stdout
        # It started from the SAME configuration, not a re-drawn one.
        rerun = yaml.safe_load((local / "AIS" / "inputs" / "sources.yaml").read_text())
        assert [int(p["source_frame_index"]) for p in rerun["paths"]] == frames
        rows = list(csv.DictReader(
            (local / "AIS" / "trajectory_0000" / "observations.csv").open()))
        assert len(rows) == 21
        assert int(rows[0]["source_frame_index"]) == frames[0]
    finally:
        source.write_bytes(kept)


@pytest.mark.gpu
@pytest.mark.slow
def test_prepared_inputs_are_refused_when_they_describe_another_calculation(managed):
    """Silently reusing them would label this run with a configuration that did not produce it."""
    local, environment = managed["local"], managed["env"]
    path = local / "AIS" / "inputs" / "sources.yaml"
    original = path.read_text()
    record = yaml.safe_load(original)
    record["selection"]["window_ps"] = [0.1, 0.2]        # a different time window
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    try:
        result = _launch(local / "AIS", "run.sh", environment, "--check")
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] AIS inputs" in combined
        assert "prepared for a different calculation" in combined
        assert "window_ps" in combined
        assert "no Context was created" in combined
    finally:
        path.write_text(original)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_truncated_prepared_input_is_refused_rather_than_used(managed):
    local, environment = managed["local"], managed["env"]
    dcd = local / "AIS" / "inputs" / "sources.dcd"
    original = dcd.read_bytes()
    raw = bytearray(original)
    struct.pack_into("<i", raw, 8, 1)                    # the header claims one configuration
    dcd.write_bytes(bytes(raw))
    try:
        result = _launch(local / "AIS", "run.sh", environment, "--check")
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] AIS inputs" in combined
        # Read, not counted: the two frames are still there and the header is the thing lying.
        assert "reads 2 frame(s)" in combined and "claims 1" in combined
    finally:
        dcd.write_bytes(original)


@pytest.mark.gpu
@pytest.mark.slow
def test_the_prepared_record_is_relative_and_states_the_frame_spacing(managed):
    local = managed["local"]
    text = (local / "AIS" / "inputs" / "sources.yaml").read_text()
    assert str(managed["root"]) not in text, "the prepared record names this machine's storage"
    record = yaml.safe_load(text)
    assert record["topology"] == "common/topology.pdb"
    assert record["source"]["trajectory"] == "cMD/whole_system.dcd"
    assert not record["source"]["tau_evidence"].startswith("/")
    # Two paths starting a few femtoseconds apart are not two independent realisations, so how
    # close the closest pair came is recorded rather than left to be inferred.
    assert record["selection"]["minimum_frame_gap"] >= 1
    assert record["selection"]["minimum_time_gap_ps"] > 0
