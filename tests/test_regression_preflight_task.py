"""Regressions for the execution and recovery contracts review found still broken.

Written BEFORE the implementation. Each names a defect present at `1e2e5b9` -- the commit whose
release notes claim every acceptance criterion passed and whose CI run is red.

The theme is that a safe outer wrapper was hiding an unsafe runtime. `md-openmm md-run` grew
guards; `replica_main`, `ais_main` and the generated `REST2.py` / `AIS.py` that call the same
runtime directly did not get them, and two permissive MPI implementations survived outside
`md_tools.remd.mpi`:

    try:
        from mpi4py import MPI
    except ImportError:
        return                      # eight ranks, no coordination, one set of output paths

So these test the LOWER level wherever the lower level is reachable, and they assert on the
FILESYSTEM: a preflight that refuses must leave nothing behind, because a directory holding a
`resolved.config` and a `.log` is indistinguishable from a run that happened.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

#: A launcher environment for a world this process is not really part of. Every test that uses it
#: also sets MD_TOOLS_FORCE_NO_MPI4PY, so the question under test is "does it refuse", never
#: "does MPI work here".
FAKE_LAUNCH = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "4",
               "MD_TOOLS_FORCE_NO_MPI4PY": "1"}


def _written(text: str, name: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text, encoding="utf-8")
    return path


def _run(argv, *, cwd, environment=None, timeout=300):
    """A subprocess with the source tree importable, so generated scripts resolve md_tools."""
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(environment or {})
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=base)


# --- 1. the CI test itself ---------------------------------------------------------------------

def test_the_ci_workflow_tests_the_ng_refusal_with_the_current_flag_contract():
    """The red step passes an XML through `-x` and omits `-s`, so argparse refuses it first.

    It therefore exits non-zero for the wrong reason and never reaches the MPI check it is named
    after. A test that fails for the wrong reason is worse than no test: it is green-looking
    coverage of something nobody exercised.
    """
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    step = workflow[workflow.index("md-run refuses -ng outside an MPI launch"):]
    step = step[:step.index("\n      - name:")] if "\n      - name:" in step else step

    assert "-x missing.xml" not in step, "an XML is still passed through the trajectory flag"
    assert "-s " in step, "-s is mandatory and the step does not supply it"
    assert "not started by an MPI launcher" in step
    # And it must notice if argparse refused first, which is the failure it is masking today.
    assert "required" in step or "argparse" in step, \
        "the step cannot tell an argparse refusal from the MPI refusal it is testing"


def test_no_ci_text_still_says_three_commands():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "three public commands" not in workflow
    assert "three commands" not in workflow


# --- 2. one MPI authority ----------------------------------------------------------------------

def test_no_permissive_mpi_import_survives_outside_the_one_authority():
    """`except ImportError: return` in a collective is how eight ranks stop being a ladder.

    Asserted over the source because it is a property no successful run can demonstrate: the
    permissive path only runs when MPI is broken, which is exactly when nobody is watching.
    """
    offenders = []
    for path in sorted((REPO / "src" / "md_tools").rglob("*.py")):
        if path.name == "mpi.py" and path.parent.name == "remd":
            continue
        text = path.read_text(encoding="utf-8")
        if "from mpi4py import MPI" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        f"mpi4py is imported outside md_tools/remd/mpi.py in {offenders}. There must be one MPI "
        f"authority; a second one is a second policy.")


def test_the_driver_coordinator_cannot_represent_a_plural_world_as_serial():
    """`Coordinator()` returned rank 0 of 1 when mpi4py was missing, whatever the launcher said."""
    from md_tools.remd.driver import Coordinator

    environment = dict(os.environ, **FAKE_LAUNCH)
    previous = dict(os.environ)
    os.environ.update(environment)
    try:
        with pytest.raises(SystemExit) as refusal:
            Coordinator()
        assert "mpi4py" in str(refusal.value), refusal.value
    finally:
        os.environ.clear()
        os.environ.update(previous)


def test_the_executor_barrier_is_not_a_second_implementation():
    """`md_tools.remd.executor.barrier` had its own permissive copy of the same decision."""
    import inspect

    from md_tools.remd import executor

    if not hasattr(executor, "barrier"):
        return                                        # removed entirely, which is also correct
    source = inspect.getsource(executor.barrier)
    assert "except ImportError" not in source, source


# --- 3. generated wrappers are as fail-closed as md-run ----------------------------------------

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One generated project per protocol, so the wrappers can be called directly."""
    root = tmp_path_factory.mktemp("wrappers")
    for protocol in ("REST2", "AIS"):
        configuration = root / f"{protocol}.config"
        document = {"protocol": protocol, "solvent": "explicit"}
        if protocol == "AIS":
            document["ais_source"] = {"trajectory": "../source.dcd"}
        configuration.write_text(yaml.safe_dump(document), encoding="utf-8")
        done = subprocess.run(CLI + ["build-md", "-odir", str(root / protocol),
                                     "--config", str(configuration)],
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stdout + done.stderr
    return root


@pytest.mark.parametrize("protocol", ["REST2", "AIS"])
def test_a_generated_wrapper_refuses_a_plural_world_without_mpi4py(protocol, generated, tmp_path):
    """The wrapper calls the same runtime `md-run` does, so it needs the same guard.

    `md-run` grew one and the wrappers did not, which made the guard a property of one entry point
    rather than of the runtime -- and the runtime is what the public Python API calls.
    """
    project = generated / protocol
    destination = tmp_path / "never"
    argv = [sys.executable, str(project / f"{protocol}.py"),
            "-p", "missing.pdb", "-s", "missing.xml", "-odir", str(destination)]
    if protocol == "AIS":
        argv += ["-source-traj", "missing.dcd"]

    done = _run(argv, cwd=project, environment=FAKE_LAUNCH)
    assert done.returncode != 0, done.stdout
    assert "mpi4py" in done.stdout + done.stderr, done.stdout + done.stderr
    assert not destination.exists(), sorted(p.name for p in destination.iterdir())


# --- 4. absent configuration is not invalid configuration --------------------------------------

def _machine(document) -> Path:
    path = Path(tempfile.mkdtemp()) / "user.config"
    path.write_text(document if isinstance(document, str)
                    else yaml.safe_dump(document), encoding="utf-8")
    return path


VALID_USER = {"person_id": "t", "name": "T", "orcid": None, "affiliation": None}


def test_an_absent_configuration_resolves_to_the_built_in_defaults():
    from md_tools.registry.userconfig import machine_openmm_settings

    resolved = machine_openmm_settings(Path(tempfile.mkdtemp()) / "absent.config")
    assert resolved["platform"] == "CUDA" and resolved["origin"] == "built-in default"


def test_a_legacy_configuration_without_the_openmm_block_is_still_valid():
    from md_tools.registry.userconfig import machine_openmm_settings

    path = _machine({"schema_version": "1.0", "user": VALID_USER,
                     "machine": {"md_data": "/tmp"}})
    assert machine_openmm_settings(path)["platform"] == "CUDA"


@pytest.mark.parametrize("document, fragment", [
    ("schema_version: \"1.0\"\nuser: [not, a, mapping]\n", "mapping"),
    ("machine:\n  openmm:\n    platform: CUDA\n    platform: CPU\n", "platform"),
    ("machine:\n  md_data: /tmp\nmachine:\n  md_data: /other\n", "machine"),
    ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
     "machine: {openmm: {platform: Metal}}\n", "platform"),
    ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
     "machine: {openmm: {precision: quadruple}}\n", "precision"),
    ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
     "machine: {openmm: {device_policy: whatever}}\n", "device_policy"),
    ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
     "machine: {openmm: {nonsense: 1}}\n", "nonsense"),
    ("this: is: not: yaml:\n", "YAML"),
])
def test_an_invalid_configuration_is_fatal_rather_than_replaced_by_defaults(document, fragment):
    """Every one of these silently became CUDA / mixed / local_rank.

    A malformed configuration is a person's mistake in a file they edited. Answering it with the
    built-in defaults means the machine runs on settings nobody chose, and the file that was
    supposed to say otherwise is never mentioned again.
    """
    from md_tools.registry.errors import RegistrationError
    from md_tools.registry.userconfig import machine_openmm_settings

    path = _machine(document)
    with pytest.raises(RegistrationError) as refusal:
        machine_openmm_settings(path)
    assert fragment.lower() in str(refusal.value).lower(), refusal.value


def test_an_environment_variable_pointing_at_a_missing_file_is_a_broken_reference():
    """`MD_TOOLS_CONFIG=/gone` is not the same as not setting it: somebody meant that path."""
    from md_tools.registry.errors import RegistrationError
    from md_tools.registry.userconfig import machine_openmm_settings

    missing = Path(tempfile.mkdtemp()) / "gone.config"
    previous = os.environ.get("MD_TOOLS_CONFIG")
    os.environ["MD_TOOLS_CONFIG"] = str(missing)
    try:
        with pytest.raises(RegistrationError, match="MD_TOOLS_CONFIG"):
            machine_openmm_settings()
    finally:
        if previous is None:
            os.environ.pop("MD_TOOLS_CONFIG", None)
        else:
            os.environ["MD_TOOLS_CONFIG"] = previous


# --- 5. preflight precedes every authoritative output ------------------------------------------

@pytest.fixture
def project(tmp_path):
    """A generated cMD project, and an output directory that must stay absent."""
    configuration = tmp_path / "cMD.config"
    configuration.write_text(yaml.safe_dump({"protocol": "cMD", "solvent": "explicit"}),
                             encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(tmp_path / "md_script"),
                                 "--config", str(configuration)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return tmp_path / "md_script", tmp_path / "never"


@pytest.mark.parametrize("case, extra, environment", [
    ("an invalid machine configuration", [], "invalid-config"),
    ("a missing topology", ["-p", "absent.pdb"], None),
    ("colliding output paths", ["-o", "same.txt", "-log", "./same.txt"], None),
    ("a trajectory named .nc", ["-x", "run.nc"], None),
])
def test_a_preflight_failure_creates_no_output_at_all(case, extra, environment, project,
                                                      tmp_path):
    """The filesystem after a refusal must be exactly the filesystem before it.

    A `-odir` holding a `resolved.config` and a `.log` is indistinguishable from a run that
    happened, and the next person to look will read it as one.
    """
    script, destination = project
    environment_map = {}
    if environment == "invalid-config":
        broken = _machine("machine:\n  openmm:\n    platform: CUDA\n    platform: CPU\n")
        environment_map["MD_TOOLS_CONFIG"] = str(broken)

    argv = CLI + ["md-run", "-i", "cMD.in", "-p", "../built.pdb", "-s", "../built.xml",
                  "-odir", str(destination)]
    for index, value in enumerate(extra):
        argv.append(value)

    done = _run(argv, cwd=script, environment=environment_map)
    assert done.returncode != 0, f"{case}: accepted"
    assert not destination.exists(), \
        f"{case}: {sorted(p.name for p in destination.iterdir())}"


def test_paths_that_resolve_to_one_file_collide_however_they_are_spelled():
    """`-o sub/../run.out` and `-log ./run.out` are one file. String comparison misses it."""
    from md_tools.run.main import check_output_collisions

    check_output_collisions(output="a.out", log="b.log")            # accepted
    with pytest.raises(SystemExit):
        check_output_collisions(output="sub/../run.out", log="./run.out")


# --- 6. platform provenance -------------------------------------------------------------------

def test_a_machine_cpu_default_is_not_recorded_as_an_explicit_cli_request():
    """`machine.openmm.platform: CPU` is somebody's machine, not somebody's command line.

    The baseline recorded both as `requested_policy: explicit-cpu` with `explicit_cpu: true`,
    which makes a machine-wide default indistinguishable from a per-run override in the record.
    """
    from md_tools.openmm.platform_policy import PlatformRequest

    machine = PlatformRequest.from_machine({"platform": "CPU", "origin": "machine.openmm"})
    assert machine.name == "CPU"
    assert machine.platform_selection == "machine-config"
    assert machine.cli_cpu_override is False

    override = PlatformRequest.from_machine({"platform": "CUDA", "origin": "machine.openmm"},
                                            cpu=True)
    assert override.platform_selection == "cli-override"
    assert override.cli_cpu_override is True

    default = PlatformRequest.from_machine({})
    assert default.platform_selection == "built-in-default"
    assert default.cli_cpu_override is False


def test_the_record_carries_the_selection_fields():
    from md_tools.openmm.platform_policy import PlatformRequest, acceleration_record

    class _Resolution:
        request = PlatformRequest.from_machine({"platform": "CPU", "origin": "machine.openmm"})
        properties: dict = {}
        device_index = None
        visible_devices = ()
        name = "CPU"

    record = acceleration_record(_Resolution())
    assert record["platform_selection"] == "machine-config"
    assert record["cli_cpu_override"] is False
    assert record["requested_platform"] == "CPU"
    assert record["resolved_platform"] == "CPU"


# --- 7. device policy --------------------------------------------------------------------------

def test_both_advertised_device_policies_have_behaviour():
    """`openmm` was accepted by validation and never consulted: a field with no runtime effect."""
    from md_tools.openmm.platform_policy import device_index_for

    assert device_index_for(policy="openmm", rank=2, size=4, devices=["0", "1", "2", "3"]) is None
    assert device_index_for(policy="local_rank", rank=2, size=4,
                            devices=["0", "1", "2", "3"]) == "2"


def test_ais_reads_the_device_policy_before_choosing_a_device():
    """AIS picked a local-rank device unconditionally, before the machine setting was loaded."""
    import inspect

    from md_tools.ais import run

    source = inspect.getsource(run)
    placement = source.index("select_device_for_rank") if "select_device_for_rank" in source \
        else source.index("device_index_for")
    settings = source.index("machine_openmm_settings")
    assert settings < placement, "the device is chosen before the machine policy is read"


# --- 8. crash-atomic AIS checkpoints -----------------------------------------------------------

def test_the_checkpoint_transaction_is_generation_based_with_a_committed_pointer():
    from md_tools.ais import checkpoint

    assert hasattr(checkpoint, "commit_generation")
    assert hasattr(checkpoint, "read_committed")
    assert checkpoint.POINTER_NAME == "current_checkpoint.json"


def test_a_committed_generation_round_trips(tmp_path):
    from md_tools.ais.checkpoint import commit_generation, read_committed

    directory = tmp_path / "path_0000"
    directory.mkdir()
    binary = tmp_path / "state.chk"
    binary.write_bytes(b"opaque openmm state")

    commit_generation(directory, write_checkpoint=lambda p: p.write_bytes(binary.read_bytes()),
                     state={"protocol_step": 200, "work_rows": 3})
    committed = read_committed(directory)
    assert committed["state"]["protocol_step"] == 200
    assert committed["generation"] == 1
    assert Path(committed["checkpoint"]).is_file()


def test_an_uncommitted_newer_generation_is_never_selected(tmp_path):
    """A crash after writing a generation and before replacing the pointer must resume the old one.

    This is the whole point of the pointer. The baseline overwrote one checkpoint and then
    replaced one sidecar, so a crash between those two writes left a NEW Context checkpoint paired
    with OLD accumulated work -- a resume that looks successful and is measuring nothing.
    """
    from md_tools.ais.checkpoint import commit_generation, read_committed

    directory = tmp_path / "path_0000"
    directory.mkdir()
    commit_generation(directory, write_checkpoint=lambda p: p.write_bytes(b"first"),
                     state={"protocol_step": 100})
    # A generation that was written but never committed.
    orphan = directory / "checkpoints" / "generation_000009.chk"
    orphan.write_bytes(b"never committed")
    (directory / "checkpoints" / "generation_000009.json").write_text(
        json.dumps({"generation": 9, "state": {"protocol_step": 900}}), encoding="utf-8")

    committed = read_committed(directory)
    assert committed["state"]["protocol_step"] == 100, "an uncommitted generation was selected"


def test_a_corrupt_committed_checkpoint_is_refused_rather_than_loaded(tmp_path):
    from md_tools.ais.checkpoint import CheckpointError, commit_generation, read_committed

    directory = tmp_path / "path_0000"
    directory.mkdir()
    commit_generation(directory, write_checkpoint=lambda p: p.write_bytes(b"good"),
                     state={"protocol_step": 100})
    committed = read_committed(directory)
    Path(committed["checkpoint"]).write_bytes(b"corrupted after the fact")

    with pytest.raises(CheckpointError, match="sha256|digest"):
        read_committed(directory)


# --- 9. no parser abbreviates -----------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    "md_tools.run.main:md_run_parser",
    "md_tools.md.stage:stage_parser",
    "md_tools.ais.run:ais_parser",
    "md_tools.remd.generated:replica_parser",
])
@pytest.mark.parametrize("abbreviation", ["--traj", "--plat", "--dev", "--res"])
def test_no_execution_parser_expands_an_abbreviation(factory, abbreviation):
    """A misspelling argparse resolves RUNS, with a setting nobody wrote."""
    import importlib

    module_name, attribute = factory.split(":")
    module = importlib.import_module(module_name)
    parser = getattr(module, attribute)
    parser = parser("test") if attribute in ("stage_parser", "ais_parser") else parser()
    assert parser.allow_abbrev is False, f"{factory} abbreviates"


# --- 10. stale documentation -------------------------------------------------------------------

def test_no_stale_netcdf_ais_source_example_survives():
    """`-source-traj hot_cmd/production.nc` documents a source a fixed-tau cMD run cannot write."""
    text = (REPO / "src" / "md_tools" / "run" / "main.py").read_text(encoding="utf-8")
    assert "production.nc" not in text, text[text.index("production.nc") - 200:][:300]


def test_no_document_claims_cpu_can_only_come_from_the_command_line():
    """A machine-wide CPU default exists now, so `--cpu` is not the only way to reach the CPU."""
    for page in (REPO / "README.md", REPO / "CLAUDE.md", REPO / "docs" / "md-run.md"):
        text = page.read_text(encoding="utf-8")
        for claim in ("the only way to ask for a CPU run",
                      "is the only public CPU opt-in",
                      "the only public way to ask for a CPU run"):
            assert claim not in text, f"{page.name}: {claim!r}"
