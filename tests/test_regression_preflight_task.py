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
    # A LADDER READS -s ONLY FROM ITS GROUP FILE (0.5.4), so the step must pass `-groupfile` and
    # NOT `-s` on the md-run command line. The assertion here used to be `"-s " in step`, with the
    # reason "-s is mandatory and the step does not supply it" -- true when written, false from
    # 0.5.4, and it then REQUIRED the broken form. That is why a green test sat over a step that
    # could not pass: the test asserted the shape of the command, and the shape had gone stale.
    assert "-groupfile" in step, "a ladder needs a group file, and no -s on the command line"
    assert "-s missing.xml" not in step, "-s on the command line is refused by the ladder rule"
    assert "not started by an MPI launcher" in step
    # And it must notice if it refused for the wrong reason. TWO wrong reasons are possible and
    # both are checked: argparse (the 2024 shape) and a protocol rule refusing first (the 0.5.4
    # shape, which the argparse-only guard was structurally blind to).
    assert "required" in step or "argparse" in step, \
        "the step cannot tell an argparse refusal from the MPI refusal it is testing"
    assert "reads -s only from its group file" in step, \
        "the step cannot tell a ladder-flag refusal from the MPI refusal it is testing"


def test_no_ci_text_still_says_three_commands():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "three public commands" not in workflow
    assert "three commands" not in workflow


def _a_real_stage(tmp_path: Path) -> dict:
    """A cMD stage whose inputs are GENUINELY readable, so a protocol rule is what refuses.

    Three earlier attempts at this check refused for the wrong reason -- an Amber key the parser
    rejects by name, then an unreadable PDB, then a System that would not deserialise -- because
    `md-run` validates its inputs before it applies any protocol rule. A test whose refusal comes
    from an earlier gate proves nothing about the gate it is named for.
    """
    from openmm import System, XmlSerializer

    topology = tmp_path / "built.pdb"
    topology.write_text((REPO / "tests" / "data" / "ALA.pdb").read_text(encoding="utf-8"),
                        encoding="utf-8")
    particles = sum(1 for line in topology.read_text(encoding="utf-8").splitlines()
                    if line.startswith(("ATOM  ", "HETATM")))
    system = System()
    for _ in range(particles):
        system.addParticle(1.0)
    (tmp_path / "built.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    (tmp_path / "cmd.in").write_text("&cntrl\n  protocol = cMD,\n  stage    = min,\n/\n",
                                     encoding="utf-8")
    return {"cwd": tmp_path, "argv": ["-i", "cmd.in", "-p", "built.pdb", "-s", "built.xml"]}


def test_md_run_refuses_ng_on_a_cmd_stage_and_writes_nothing(tmp_path):
    """`-ng` was parsed by `md-run`, dropped, and never reached the rule written to refuse it.

    `preflight_stage` has always refused `-ng` for a stage, and the GENERATED stage script has
    always passed it there (`md/stage.py`). `md-run` called the same preflight without
    `number_of_groups`, which defaults to None -- so the refusal was handed nothing to refuse, and
    `md-openmm md-run -ng 4` on a cMD input ran ONE process and exited 0. That is the disagreement
    between `md-run` and a generated script this project forbids, with `md-run` unchecked.

    Both directions are asserted: a guard that has not been shown to stay quiet when the flag is
    absent is not a guard, it is a refusal that happens to be on.
    """
    fixture = _a_real_stage(tmp_path)

    refused = subprocess.run(CLI + ["md-run", "-ng", "4", *fixture["argv"], "-odir", "out"],
                             cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert refused.returncode != 0, refused.stdout
    assert "replicas to group" in refused.stderr, refused.stderr
    assert not (tmp_path / "out").exists(), \
        f"a refused preflight created -odir: {sorted(p.name for p in (tmp_path / 'out').iterdir())}"

    accepted = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "out2"],
                              cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert "replicas to group" not in accepted.stderr, \
        f"the rule fired without -ng: {accepted.stderr}"


def test_md_run_check_creates_nothing_on_a_cpu_run(tmp_path):
    """The device-free half of `--check` creates nothing: the same two cases, with `--cpu`.

    `--check` runs the WHOLE preflight, which ends in a real CUDA Context, so the ordinary form of
    this check cannot run without a card and lives in the gpu lane below. What it is really
    asserting -- that `md-run` does not create `-odir` or `resolved.config` before dispatching to
    a runtime -- is platform-independent, and `--cpu` is the project's one per-run override. So
    the coverage stays in the fast lane rather than moving wholesale to a lane most runs never
    reach, and the gpu test keeps the default CUDA path and the real run.
    """
    fixture = _a_real_stage(tmp_path)

    checked = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "fresh", "--check",
                                    "--cpu"],
                             cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "Nothing was created" in checked.stdout, checked.stdout
    assert not (tmp_path / "fresh").exists(), \
        f"--check created -odir: {sorted(p.name for p in (tmp_path / 'fresh').iterdir())}"

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "marker.txt").write_text("kept", encoding="utf-8")
    again = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "existing", "--check",
                                  "--cpu"],
                           cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stdout + again.stderr
    assert sorted(p.name for p in existing.iterdir()) == ["marker.txt"], \
        sorted(p.name for p in existing.iterdir())


@pytest.mark.gpu
def test_md_run_check_creates_nothing_not_even_the_output_directory(tmp_path):
    """`--check` printed "Nothing was created." and created `-odir` and `resolved.config`.

    The message was false as it was printed. Each RUNTIME already returned read-only for `--check`;
    `md-run` created both before dispatching to them, so the public surface produced what the
    runtimes decline to -- and a `-odir` holding a `resolved.config` is what this project's contract
    calls indistinguishable from a run that happened.

    Three cases, because the third is what a guard on the first two could silently break.

    MARKED `gpu`, which it was not: every case here goes through the whole preflight, and that
    ends in a real CUDA Context, so all three fail with CUDA_ERROR_NO_DEVICE on a machine with no
    card. Unmarked, it failed in every card-less lane instead of being deselected honestly -- and
    a lane with a standing known failure is a lane people stop reading. The device-free half is
    `test_md_run_check_creates_nothing_on_a_cpu_run` above, in the fast lane.
    """
    fixture = _a_real_stage(tmp_path)

    checked = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "fresh", "--check"],
                             cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "Nothing was created" in checked.stdout, checked.stdout
    assert not (tmp_path / "fresh").exists(), \
        f"--check created -odir: {sorted(p.name for p in (tmp_path / 'fresh').iterdir())}"

    # AN EXISTING `-odir` is the same violation with the directory already supplied: `--check` must
    # not add `resolved.config` to a tree it was asked only to report on.
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "marker.txt").write_text("kept", encoding="utf-8")
    again = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "existing", "--check"],
                           cwd=fixture["cwd"], capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stdout + again.stderr
    assert sorted(p.name for p in existing.iterdir()) == ["marker.txt"], \
        sorted(p.name for p in existing.iterdir())

    # AND A REAL RUN IS UNCHANGED. The fix routes `resolved.config` through a temporary directory
    # under `--check`; if that leaked into the ordinary path, nothing would be written at all.
    real = subprocess.run(CLI + ["md-run", *fixture["argv"], "-odir", "real"],
                          cwd=fixture["cwd"], capture_output=True, text=True, timeout=600)
    assert real.returncode == 0, real.stdout[-2000:] + real.stderr[-2000:]
    assert (tmp_path / "real" / "resolved.config").is_file(), \
        f"a real run wrote nothing: {sorted(p.name for p in (tmp_path / 'real').iterdir())}"


def test_md_run_check_on_a_chain_does_not_demand_what_only_running_produces(tmp_path):
    """A whole-workflow `--check` was refused BY CONSTRUCTION.

    Stage 2's `-c <odir>/min.xml` is stage 1's own output. Under `--check` nothing runs, so it
    cannot exist -- and the command that answers "would this start?" refused because the work had
    not been done. `_continuation_inputs` already carries the one legitimate exception, a NAMED
    earlier stage that writes the file, and `validate_generated_chain` states it at generation time.
    `md-run` never stated it.
    """
    fixture = _a_real_stage(tmp_path)
    (tmp_path / "chain.in").write_text("&cntrl\n  protocol = cMD,\n/\n", encoding="utf-8")

    done = subprocess.run(
        CLI + ["md-run", "-i", "chain.in", "-p", "built.pdb", "-s", "built.xml",
               "-odir", "chain", "--check"],
        cwd=fixture["cwd"], capture_output=True, text=True, timeout=600)

    assert "does not exist" not in done.stderr, \
        f"a chain --check still demands an artefact only running produces:\n{done.stderr}"
    assert not (tmp_path / "chain").exists(), "--check created -odir for a chain"


# --- 2. one MPI authority ----------------------------------------------------------------------

def test_no_permissive_mpi_import_survives_outside_the_one_authority():
    """`except ImportError: return` in a collective is how eight ranks stop being a ladder.

    Asserted over the source because it is a property no successful run can demonstrate: the
    permissive path only runs when MPI is broken, which is exactly when nobody is watching.
    """
    import ast

    offenders = []
    for path in sorted((REPO / "src" / "md_tools").rglob("*.py")):
        if path.name == "mpi.py" and path.parent.name == "remd":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # The AST, not the text: a docstring that QUOTES the retired code to explain why it was
        # retired is exactly the thing worth keeping, and a grep cannot tell it from the code.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mpi4py"):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
            if isinstance(node, ast.Import) and any(
                    alias.name.startswith("mpi4py") for alias in node.names):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        f"mpi4py is imported outside md_tools/remd/mpi.py at {offenders}. There must be one MPI "
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
    import ast
    import inspect

    from md_tools.remd import executor

    if not hasattr(executor, "barrier"):
        return                                        # removed entirely, which is also correct
    tree = ast.parse(inspect.getsource(executor.barrier).lstrip())
    handlers = [h for node in ast.walk(tree) if isinstance(node, ast.Try) for h in node.handlers]
    caught = [ast.unparse(h.type) for h in handlers if h.type is not None]
    assert "ImportError" not in caught, (
        f"executor.barrier still decides what to do about a missing mpi4py ({caught}); it must "
        f"delegate to md_tools.remd.mpi, which refuses.")


# --- 3. generated wrappers are as fail-closed as md-run ----------------------------------------

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One generated project per protocol, so the wrappers can be called directly."""
    from .conftest import make_dataset_root

    root = tmp_path_factory.mktemp("wrappers")
    # A ladder's rungs are scaled from `build/built.xml` at BUILD time now, so the dataset's
    # shared System has to exist before any run is generated.
    make_dataset_root(root, solvent="explicit")
    for protocol in ("REST2", "AIS"):
        configuration = root / f"{protocol}.config"
        document = {"protocol": protocol, "solvent": "explicit"}
        if protocol == "AIS":
            document["ais_source"] = {"trajectory": "../source.dcd"}
        configuration.write_text(yaml.safe_dump(document), encoding="utf-8")
        from .conftest import make_states_for

        make_states_for(root, configuration)
        done = subprocess.run(CLI + ["build-md", "-odir", str(root / f"{protocol}-run1"),
                                     "--config", str(configuration)],
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stdout + done.stderr
    # The state the ladder's group file continues from (a ladder reads its inputs only there).
    from .conftest import write_starting_state

    write_starting_state(root, root / "REST2-run1")
    return root


@pytest.mark.parametrize("protocol", ["REST2", "AIS"])
def test_a_generated_wrapper_refuses_a_plural_world_without_mpi4py(protocol, generated, tmp_path):
    """The wrapper calls the same runtime `md-run` does, so it needs the same guard.

    `md-run` grew one and the wrappers did not, which made the guard a property of one entry point
    rather than of the runtime -- and the runtime is what the public Python API calls.
    """
    project = generated / f"{protocol}-run1"
    destination = tmp_path / "never"
    # A ladder reads -s only from its group file (0.5.4), whose `-i` must name the protocol this
    # launch writes into `destination`; AIS still takes -s.
    if protocol == "REST2":
        from .conftest import ladder_group_file

        system = ["--groupfile", str(ladder_group_file(
            generated, destination.resolve(), run_dir="REST2-run1",
            start=generated / "REST2-run1" / "eq" / "eq_3.xml"))]
    else:
        system = ["-s", "missing.xml"]
    argv = [sys.executable, str(project / f"{protocol}.py"),
            "-p", "missing.pdb", *system, "-odir", str(destination)]
    if protocol == "AIS":
        # Both end states are named, so the refusal reached is the plural world rather than the
        # missing-flag refusal that precedes it.
        argv += ["-p2", "missing.pdb", "-s2", "missing2.xml", "-source-traj", "missing.dcd"]

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
    """Nobody has written one. That is legitimate and resolves to CUDA / mixed / local_rank.

    A genuine absence, via an empty XDG root -- not an explicit path that happens to be missing,
    which is a different thing and is tested separately as a broken reference.
    """
    from md_tools.registry.userconfig import machine_openmm_settings

    empty = Path(tempfile.mkdtemp())
    previous = {key: os.environ.get(key) for key in ("XDG_CONFIG_HOME", "MD_TOOLS_CONFIG")}
    os.environ["XDG_CONFIG_HOME"] = str(empty)
    os.environ.pop("MD_TOOLS_CONFIG", None)
    try:
        resolved = machine_openmm_settings()
        assert resolved["platform"] == "CUDA" and resolved["origin"] == "built-in default"
        assert resolved["config_path"] is None
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_an_explicit_path_that_does_not_exist_is_a_broken_reference_not_an_absence():
    """`--user-config /gone` names a file somebody meant. Defaults would hide the typo."""
    from md_tools.registry.errors import RegistrationError
    from md_tools.registry.userconfig import machine_openmm_settings

    with pytest.raises(RegistrationError, match="does not exist"):
        machine_openmm_settings(Path(tempfile.mkdtemp()) / "absent.config")


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
    from .conftest import make_dataset_root

    # IMPLICIT, because these cases INTEGRATE when they are not refused, and that makes this the
    # one fixture here whose System has to be physically viable rather than merely well shaped.
    # The explicit stand-in is a box with no water in it: a barostat collapses it on the first
    # step with "the periodic box size has decreased to less than twice the nonbonded cutoff",
    # which would end every case in this table with a non-zero exit for a reason that has nothing
    # to do with the refusal under test -- and a refusal test that passes because the run crashed
    # is testing the crash. GBn2 has no box and no barostat, so a case that is NOT refused runs.
    # The solvent is irrelevant to every assertion below; only the refusals are the subject.
    make_dataset_root(tmp_path, solvent="implicit")
    configuration = tmp_path / "cMD.config"
    configuration.write_text(yaml.safe_dump({"protocol": "cMD", "solvent": "implicit"}),
                             encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(tmp_path / "cMD-run1"),
                                 "--config", str(configuration)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return tmp_path / "cMD-run1", tmp_path / "never"


@pytest.mark.parametrize("case, extra, environment", [
    ("an invalid machine configuration", [], "invalid-config"),
    ("a missing topology", ["-p", "absent.pdb"], None),
    ("colliding output paths", ["-o", "same.txt", "-log", "./same.txt"], None),
    # `.xtc`, NOT `.nc`. This case named `.nc` for as long as a stage wrote through OpenMM's DCD
    # reporter, which had no honest way to produce NetCDF. The stage writes through mdtraj's
    # NetCDF writer now -- genuine AMBER NetCDF, and the only format carrying an atom subset, so
    # `.nc` is the DEFAULT name a stage chooses for itself and `check_trajectory_suffix` accepts
    # it deliberately. The case survived only because this fixture built no System until now, so
    # every row here was refused for a missing `-p` before its own flag was ever reached; with a
    # real System, `-x run.nc` is accepted, honoured, and writes the trajectory it names.
    #
    # The guard it was written for is unchanged -- a name claiming a format the stage cannot
    # produce -- so the case is repointed at one that still claims it.
    ("a trajectory named .xtc", ["-x", "run.xtc"], None),
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

    argv = CLI + ["md-run", "-i", "../input/cMD.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
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

    check_output_collisions(outputs={"o": "a.out", "log": "b.log"})          # accepted
    with pytest.raises(SystemExit):
        check_output_collisions(outputs={"o": "sub/../run.out", "log": "./run.out"})


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
    from md_tools.openmm.placement import MpsStatus, WorkerFacts, plan_launch

    absent = MpsStatus(requested=False, pipe_directory="/tmp/nvidia-mps", daemon="not-running")
    facts = [WorkerFacts(rank=r, hostname="h", cpus=tuple(range(4)), cpu_quota=None,
                         visible_devices=4, cuda_visible_devices=None, cuda_device_order=None,
                         launcher_local_rank=None, mps=absent) for r in range(4)]
    openmm = plan_launch(facts, platform="CUDA", device_policy="openmm")
    assert openmm.for_rank(2)["device"] is None
    placed = plan_launch(facts, platform="CUDA", device_policy="local_rank",
                         throughput={"h": [1.0, 1.0, 1.0, 1.0]})
    assert placed.for_rank(2)["device"] == 2


def test_ais_reads_the_device_policy_before_choosing_a_device():
    """AIS picked a local-rank device unconditionally, before the machine setting was loaded."""
    import inspect

    from md_tools.ais import run
    from md_tools.run import preflight

    # AIS no longer places a device at all. It asks preflight and uses the answer, so the
    # ordering cannot be got wrong here by editing this module: there is nothing left to order.
    for name in ("device_index_for", "plan_launch", "balanced_assignment"):
        assert name not in inspect.getsource(run), \
            f"AIS places its own device again ({name}); placement belongs to the shared preflight"

    # The ordering now lives in the one place that does the work, and is asserted there.
    source = inspect.getsource(preflight)
    settings = source.index("machine_openmm_settings(machine_config)")
    placement = source.index("placing.plan_launch(")
    assert settings < placement, "the device is chosen before the machine policy is read"

    # And the policy the call is given is the machine's, not a constant.
    call = source[placement:placement + 200]
    assert "device_policy=policy" in call, call


# --- 8. crash-atomic AIS checkpoints -----------------------------------------------------------

def test_the_checkpoint_transaction_is_generation_based_with_a_committed_pointer():
    from md_tools.openmm import checkpoint

    assert hasattr(checkpoint, "commit_generation")
    assert hasattr(checkpoint, "read_committed")
    assert checkpoint.POINTER_NAME == "current_checkpoint.json"


def test_a_committed_generation_round_trips(tmp_path):
    from md_tools.openmm.checkpoint import commit_generation, read_committed

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
    from md_tools.openmm.checkpoint import commit_generation, read_committed

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
    from md_tools.openmm.checkpoint import CheckpointError, commit_generation, read_committed

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
