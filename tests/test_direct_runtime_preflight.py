"""The generated scripts refuse before touching the filesystem, exactly as `md-run` does.

`md-openmm md-run` grew a shared preflight. The runtimes it calls did not have one, and the
generated `min.py`, `md.py`, `REST2.py`, `rREST2.py` and `AIS.py` call those runtimes DIRECTLY --
so a generated project could create its output directory, its `.out`, its `.log`, `solute.yaml`,
`_protocol.py` and a group file before discovering that the machine configuration was malformed or
that CUDA could not open a Context.

That is the fail-closed contract broken at the level where most people actually run things. A
directory holding a log and a resolved configuration is indistinguishable from a run that started.

So every test here invokes a GENERATED SCRIPT as a subprocess, snapshots the destination
byte-for-byte, and asserts the snapshot is unchanged -- or the directory still absent -- after the
refusal. And each asserts a diagnostic specific to its own condition: a test that is satisfied by
an argparse error proves nothing about the validation it is named for, because argparse runs first
and refuses everything equally.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

#: Deterministic seams. Neither is set in normal use; both exist so a refusal path can be exercised
#: by the real command instead of by a mock of it.
NO_MPI4PY = "MD_TOOLS_FORCE_NO_MPI4PY"
NO_CUDA = "MD_TOOLS_FORCE_NO_CUDA"


# --- helpers -------------------------------------------------------------------------------------

def _snapshot(path: Path) -> dict[str, str] | None:
    """Every file under `path`, by digest. None when the directory does not exist.

    Digests rather than names: a refusal that truncates or rewrites an existing file is as much a
    violation as one that creates a new file, and a name-only comparison would miss it.
    """
    if not path.exists():
        return None
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def _run(argv, *, cwd, environment=None, timeout=600):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(environment or {})
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=base)


def _refused(done, *, fragment: str):
    """Non-zero, and for the stated reason rather than an argparse complaint."""
    message = done.stdout + done.stderr
    assert done.returncode != 0, f"accepted:\n{message[-2000:]}"
    assert "the following arguments are required" not in message, (
        f"argparse refused first, so the check under test never ran:\n{message[-2000:]}")
    assert "unrecognized arguments" not in message, (
        f"argparse refused first:\n{message[-2000:]}")
    assert fragment.lower() in message.lower(), (
        f"refused, but not for {fragment!r}:\n{message[-3000:]}")
    return message


def _config(directory: Path, name: str, document) -> Path:
    path = directory / name
    path.write_text(document if isinstance(document, str) else yaml.safe_dump(document),
                    encoding="utf-8")
    return path


VALID_USER = {"person_id": "someone", "name": "Some One", "orcid": None, "affiliation": None}


# --- one built system and one generated project per protocol -------------------------------------

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A built System and a generated directory for each protocol, made once."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("direct")

    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    tiny = {"minimization_iterations": 5, "restrained_nvt_steps": 5,
            "restrained_npt_steps": 5, "unrestrained_npt_steps": 5, "production_steps": 5}
    reporting = {"solute_printout": 5, "system_printout": 5, "checkpoint_printout": 5}
    projects = {
        "split": {"protocol": "cMD", "solvent": "implicit", "stages": tiny,
                  "reporting": reporting},
        "allinone": {"protocol": "cMD", "solvent": "implicit", "stages": tiny,
                     "reporting": reporting},
        "REST2": {"protocol": "REST2", "solvent": "implicit", "stages": tiny,
                  "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                            "number_of_exchanges": 2},
                  "reporting": reporting},
        "rREST2": {"protocol": "rREST2", "solvent": "implicit", "stages": tiny,
                   "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                             "number_of_exchanges": 2},
                   "reservoir": {"enabled": True, "path": "../reservoir.nc"},
                   "reporting": reporting},
        "AIS": {"protocol": "AIS", "solvent": "implicit",
                "ais": {"number_of_paths": 2, "switching_steps": 10,
                        "observation_interval_steps": 5},
                "ais_source": {"trajectory": "../source.dcd"},
                "reporting": {"solute_printout": 5, "system_printout": 5,
                              "checkpoint_printout": 5}},
    }
    for name, document in projects.items():
        _config(root, f"{name}.config", document)
        extra = ["--all-in-one"] if name == "allinone" else []
        done = subprocess.run(
            CLI + ["build-md", "-odir", f"./{name}", "--config", str(root / f"{name}.config"),
                   *extra],
            cwd=root, capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, f"{name}: {done.stdout}{done.stderr}"

    # A GENUINE, READABLE DCD of this very system.
    #
    # This used to be 200 zero bytes behind a DCD magic number. That was enough while the source
    # was only sniffed for its format, and it stopped being enough the moment the preflight began
    # actually reading the source -- counting its frames and comparing its atom count -- which is
    # the whole point of moving those checks ahead of the output. A file that satisfies the
    # format check and cannot be opened tests the format check and nothing else.
    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


#: Which script each mode is launched through, and the flags it needs beyond -p/-s.
ENTRY = {
    "split": ("split/min.py", []),
    "allinone": ("allinone/md.py", []),
    "REST2": ("REST2/REST2.py", []),
    "rREST2": ("rREST2/rREST2.py", []),
    "AIS": ("AIS/AIS.py", ["-source-traj", "../source.dcd"]),
}
MODES = sorted(ENTRY)


def _launch(workspace, mode, destination: Path, *extra, environment=None):
    script, needed = ENTRY[mode]
    argv = [sys.executable, str(workspace / script),
            "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
            *needed, *extra]
    return _run(argv, cwd=workspace / Path(script).parent, environment=environment)


# --- the machine configuration -------------------------------------------------------------------

BROKEN_CONFIGS = {
    "malformed YAML": ("machine:\n  openmm:\n   platform: CUDA\n  bad indent\n", "YAML"),
    "duplicate keys": ("machine:\n  openmm:\n    platform: CUDA\n    platform: CPU\n",
                       "platform"),
    "unsupported platform": ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
                             "machine: {openmm: {platform: OpenCL}}\n", "platform"),
    "unsupported precision": ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
                              "machine: {openmm: {precision: quadruple}}\n", "precision"),
    "unsupported device policy": ("schema_version: \"1.0\"\nuser: {person_id: t, name: T}\n"
                                  "machine: {openmm: {device_policy: whatever}}\n",
                                  "device_policy"),
    "a blank person_id": ("schema_version: \"1.0\"\nuser: {person_id: \"\", name: T}\n",
                          "person_id"),
    "a missing name": ("schema_version: \"1.0\"\nuser: {person_id: t}\n", "name"),
    "a non-mapping user": ("schema_version: \"1.0\"\nuser: [a, b]\n", "mapping"),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(BROKEN_CONFIGS))
def test_a_broken_machine_configuration_stops_a_generated_script_before_any_output(
        mode, case, workspace, tmp_path):
    """Eight ways of being wrong, through five generated entry points, times nothing written.

    The configuration is a file a person edits by hand. Answering a mistake in it with the
    built-in defaults would run the machine on settings nobody chose; answering it after creating
    the output directory would leave a directory that reads as a started run.
    """
    document, fragment = BROKEN_CONFIGS[case]
    broken = _config(tmp_path, "user.config", document)
    destination = tmp_path / "never"

    done = _launch(workspace, mode, destination,
                   environment={"MD_TOOLS_CONFIG": str(broken)})
    _refused(done, fragment=fragment)
    assert _snapshot(destination) is None, sorted((_snapshot(destination) or {}))


@pytest.mark.parametrize("mode", MODES)
def test_an_explicit_configuration_path_that_does_not_exist_is_refused(mode, workspace, tmp_path):
    """`MD_TOOLS_CONFIG=/gone` names a file somebody meant. Defaults would hide the typo."""
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination,
                   environment={"MD_TOOLS_CONFIG": str(tmp_path / "absent.config")})
    _refused(done, fragment="does not exist")
    assert _snapshot(destination) is None


# --- the inputs ------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("flag, missing", [("-p", "gone.pdb"), ("-s", "gone.xml")])
def test_a_missing_input_stops_a_generated_script_before_any_output(mode, flag, missing,
                                                                    workspace, tmp_path):
    destination = tmp_path / "never"
    script, needed = ENTRY[mode]
    argv = [sys.executable, str(workspace / script),
            "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination), *needed]
    argv[argv.index(flag) + 1] = missing
    done = _run(argv, cwd=workspace / Path(script).parent)
    _refused(done, fragment="does not exist")
    assert _snapshot(destination) is None


@pytest.mark.parametrize("mode", MODES)
def test_a_topology_and_system_that_describe_different_particle_counts_are_refused(
        mode, workspace, tmp_path):
    """A cheap check that must happen before output: two files that cannot belong together.

    Reading a five-atom PDB against a thousand-particle System produces a Context that cannot be
    built, several layers down, after the log has been written.
    """
    stub = tmp_path / "two-atoms.pdb"
    stub.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "END\n", encoding="utf-8")
    destination = tmp_path / "never"
    script, needed = ENTRY[mode]
    done = _run([sys.executable, str(workspace / script),
                 "-p", str(stub), "-s", "../built.xml", "-odir", str(destination), *needed],
                cwd=workspace / Path(script).parent)
    _refused(done, fragment="particle")
    assert _snapshot(destination) is None


def test_an_ais_source_that_is_not_a_trajectory_is_refused_before_output(workspace, tmp_path):
    mislabelled = tmp_path / "source.dcd"
    mislabelled.write_bytes(b"this is not a trajectory at all" + b"\x00" * 100)
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / "AIS/AIS.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 "-source-traj", str(mislabelled)],
                cwd=workspace / "AIS")
    _refused(done, fragment="neither a DCD nor a NetCDF")
    assert _snapshot(destination) is None


def test_a_missing_ais_source_is_refused_before_output(workspace, tmp_path):
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / "AIS/AIS.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 "-source-traj", str(tmp_path / "absent.dcd")],
                cwd=workspace / "AIS")
    _refused(done, fragment="does not exist")
    assert _snapshot(destination) is None


# --- path collisions ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["split", "REST2", "rREST2", "AIS"])
def test_two_output_flags_that_resolve_to_one_file_are_refused(mode, workspace, tmp_path):
    """`sub/../run.out` and `./run.out` are one file. String comparison misses it.

    Opening it twice would have one writer truncate the other, and the reader would never know
    which half survived.
    """
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination,
                   "-o", "sub/../shared.txt", "-log", "./shared.txt")
    _refused(done, fragment="same file")
    assert _snapshot(destination) is None


@pytest.mark.parametrize("mode", ["split", "REST2", "rREST2", "AIS"])
def test_an_output_that_would_overwrite_an_input_is_refused(mode, workspace, tmp_path):
    """Writing the log over `built.xml` destroys the System the run needs to read."""
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination, "-log", "../built.xml")
    _refused(done, fragment="input")
    assert _snapshot(destination) is None


# --- the platform ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_cuda_that_cannot_open_a_context_stops_a_generated_script_before_any_output(
        mode, workspace, tmp_path):
    """Proved with a deterministic seam, so this runs on a machine with or without a GPU.

    Listing the CUDA platform is not the same as having a usable device -- conda-forge ships the
    plugin unconditionally -- so the resolver opens and discards a one-particle Context. What is
    asserted here is that the failure lands BEFORE the output directory, not after it.
    """
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination, environment={NO_CUDA: "1"})
    _refused(done, fragment="CUDA")
    assert _snapshot(destination) is None


@pytest.mark.parametrize("mode", MODES)
def test_a_plural_launch_without_mpi4py_stops_a_generated_script_before_any_output(
        mode, workspace, tmp_path):
    """A launcher claiming four ranks, and no way to coordinate them."""
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination,
                   environment={"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "4",
                                NO_MPI4PY: "1"})
    _refused(done, fragment="mpi4py")
    assert _snapshot(destination) is None


# --- an existing directory is not damaged either ---------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_a_refusal_does_not_modify_an_existing_output_directory(mode, workspace, tmp_path):
    """The other half of "creates nothing": it must not TOUCH what is already there.

    A rerun into a directory holding a previous run's results must leave them byte-for-byte
    identical when it refuses, or the refusal has destroyed the thing it was protecting.
    """
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "previous.log").write_text("an earlier run's record\n", encoding="utf-8")
    (destination / "previous.dcd").write_bytes(b"\x54\x00\x00\x00CORD" + b"\x00" * 64)
    before = _snapshot(destination)

    broken = _config(tmp_path, "user.config",
                     "machine:\n  openmm:\n    platform: CUDA\n    platform: CPU\n")
    done = _launch(workspace, mode, destination, environment={"MD_TOOLS_CONFIG": str(broken)})
    _refused(done, fragment="platform")
    assert _snapshot(destination) == before, "a refusal modified an existing directory"


# --- and the checks are reached in the right order --------------------------------------------------

def test_a_stage_reports_the_configuration_problem_rather_than_the_platform(workspace, tmp_path):
    """Ordering, asserted by which message comes out.

    A broken configuration and an unusable CUDA are both present here. The configuration must be
    what is reported: the platform cannot be resolved until the machine settings that name it have
    been read, so a platform error first would mean the platform was resolved from defaults the
    file was trying to override.
    """
    broken = _config(tmp_path, "user.config",
                     "schema_version: \"1.0\"\nuser: {person_id: \"\", name: T}\n")
    destination = tmp_path / "never"
    done = _launch(workspace, "split", destination,
                   environment={"MD_TOOLS_CONFIG": str(broken), NO_CUDA: "1"})
    message = _refused(done, fragment="person_id")
    assert "CUDA" not in message.split("person_id")[0][-400:], message


def test_the_check_flag_does_not_skip_validation_a_real_run_relies_on(workspace, tmp_path):
    """`--check` exists to validate a chain before running it, not to validate less of it."""
    broken = _config(tmp_path, "user.config",
                     "machine:\n  openmm:\n    platform: OpenCL\n")
    destination = tmp_path / "never"
    done = _launch(workspace, "split", destination, "--check",
                   environment={"MD_TOOLS_CONFIG": str(broken)})
    _refused(done, fragment="platform")
    assert _snapshot(destination) is None
