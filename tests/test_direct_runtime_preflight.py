"""The generated scripts refuse before touching the filesystem, exactly as `md-run` does.

`md-openmm md-run` grew a shared preflight. The runtimes it calls did not have one, and the
generated `min.py`, `md.py`, `REST2.py` and `AIS.py` call those runtimes DIRECTLY --
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

from .conftest import ladder_group_file

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

#: Passed by every test whose subject is the PROTOCOL rather than the platform.
#:
#: The preflight resolves the platform before it looks inside `-p` and `-s`, so on a machine with
#: no GPU -- every CI runner -- a CUDA refusal arrives first and the test then passes for entirely
#: the wrong reason, or fails, depending on which way it was written. Fourteen tests did exactly
#: that, and the workflow that ran them was reported green because its pytest was piped through
#: `tee` without `pipefail`.
#:
#: `--cpu` is the honest fix: it says "this test is about particle counts / trajectory formats /
#: HMR / collisions / `--check`, and the accelerator is not the variable". Tests of CUDA POLICY
#: never use it -- they are in the GPU lane and must fail if CUDA is missing, which is the point
#: of them.
PROTOCOL_ONLY = ["--cpu"]


# --- one built system and one generated project per protocol -------------------------------------

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A built System and a generated directory for each protocol, made once."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("direct")

    # THE DATASET ROOT. `build/`, `min/` and `input/` are shared by every run on this system, and
    # a ladder's rungs are scaled from `build/built.xml` at build time, so the System has to be
    # there before any run is generated. Real `build-top` -- and its tleap dependency -- is kept:
    # these tests launch generated scripts against this System, so it must be one a user would
    # actually have.
    (root / "build").mkdir(exist_ok=True)
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    # V1, THE SECOND END STATE AIS NEEDS. AIS transforms V0 (`-s`) into V1 (`-s2`), and the pair
    # is checked particle by particle before any output exists, so V1 has to be a genuine
    # parameter-only edit of this very System -- here the REST2 scaling at tau = 0.5, which
    # changes charges, epsilons, torsions and GB parameters and nothing structural. A copy of
    # `built.xml` would be refused as the same Hamiltonian, and the tests would pass for that.
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    solute = solute_atom_indices(PDBFile(str(root / "build" / "built.pdb")).topology)
    (root / "build" / "V1.xml").write_text(
        XmlSerializer.serialize(build_scaled_system(base, solute, 0.5)), encoding="utf-8")

    tiny = {"minimization_iterations": 5, "restrained_nvt_steps": 5,
            "restrained_npt_steps": 5, "unrestrained_npt_steps": 5, "production_steps": 5}
    reporting = {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5}
    projects = {
        "split": {"protocol": "cMD", "solvent": "implicit", "stages": tiny,
                  "reporting": reporting},
        "REST2": {"protocol": "REST2", "solvent": "implicit", "stages": tiny,
                  "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                            "number_of_exchanges": 2},
                  "reporting": reporting},
        "AIS": {"protocol": "AIS", "solvent": "implicit",
                "ais": {"number_of_paths": 2, "switching_steps": 10,
                        "observation_interval_steps": 5},
                "ais_source": {"trajectory": "../source.dcd"},
                "reporting": {"crd_printout_solute": 5, "info_printout": 5,
                              "checkpoint_printout": 5}},
    }
    for name, document in projects.items():
        _config(root, f"{name}.config", document)
        from .conftest import make_states_for

        make_states_for(root, root / f"{name}.config")
        done = subprocess.run(
            CLI + ["build-md", "-odir", f"./{name}-run1",
                   "--config", str(root / f"{name}.config")],
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

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))

    # THE STATE A LADDER'S GROUP FILE CONTINUES FROM. A ladder reads -s only from its group file
    # (0.5.4), and every line of the one `build-md` wrote names `-c eq/eq_3.xml`: without it each
    # launch below would be refused for the missing parent before reaching its own subject.
    from .conftest import write_starting_state

    write_starting_state(root, root / "REST2-run1")
    return root


#: The second end state, which every AIS invocation needs and nothing else accepts. Relative to a
#: run directory, as `run.sh` passes it.
V1 = ["-p2", "../build/built.pdb", "-s2", "../build/V1.xml"]

#: Which script each mode is launched through, and the flags it needs beyond -p/-s.
#: mode -> (the generated entry point, relative to the dataset root; extra flags it requires)
#:
#: `split` names the SHARED minimisation script. Minimisation draws no velocities and has no
#: seeded stochastic element, so every run on one system minimises to the same structure and the
#: script lives at `<system>/min/min.py` rather than in any one run. The others are per run.
ENTRY = {
    "split": ("min/min.py", []),
    # A ladder reads -s only from its group file (0.5.4), so it is launched with the one
    # `build-md` wrote rather than with -s.
    "REST2": ("REST2-run1/REST2.py", ["--groupfile", "remd_groupfile.1"]),
    "AIS": ("AIS-run1/AIS.py", ["-source-traj", "../source.dcd", *V1]),
}
MODES = sorted(ENTRY)


def _launch(workspace, mode, destination: Path, *extra, environment=None):
    """Run a generated entry point from its own directory, as a person would.

    `-p` and `-s` reach the dataset's shared `build/` -- one level up from a run, and from the
    shared `min/` -- which is what every generated `run.sh` types too.
    """
    script, needed = ENTRY[mode]
    system = [] if mode == "REST2" else ["-s", "../build/built.xml"]
    if mode == "REST2":
        # INTO `destination`, not the run directory, so the group file's `-i` must name the
        # `_protocol.py` the ladder writes THERE; a mismatch is refused before anything else.
        needed = ["--groupfile", str(ladder_group_file(
            workspace, Path(destination).resolve(), run_dir=f"{mode}-run1",
            start=workspace / f"{mode}-run1" / "eq" / "eq_3.xml"))]
    argv = [sys.executable, str(workspace / script),
            "-p", "../build/built.pdb", *system, "-odir", str(destination),
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
    ladder = mode == "REST2"
    if ladder:
        # A group file for a launch into `destination` (its `-i` must name the protocol written
        # there), and a ladder reads -s only from it (0.5.4): the missing System is on a line.
        group = ladder_group_file(workspace, destination.resolve(), run_dir=f"{mode}-run1",
                                  start=workspace / f"{mode}-run1" / "eq" / "eq_3.xml")
        needed = ["--groupfile", str(group)]
        if flag == "-s":
            state = str((workspace / "build" / "REST2" / "system_state0.xml").resolve())
            text = group.read_text(encoding="utf-8")
            assert state in text, text
            group.write_text(text.replace(state, str(tmp_path / missing)), encoding="utf-8")
    argv = [sys.executable, str(workspace / script),
            "-p", "../build/built.pdb", *([] if ladder else ["-s", "../build/built.xml"]),
            "-odir", str(destination), *needed]
    if not (ladder and flag == "-s"):
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
    system = [] if mode == "REST2" else ["-s", "../build/built.xml"]
    if mode == "REST2":
        # INTO `destination`, not the run directory, so the group file's `-i` must name the
        # `_protocol.py` the ladder writes THERE; a mismatch is refused before anything else.
        needed = ["--groupfile", str(ladder_group_file(
            workspace, Path(destination).resolve(), run_dir=f"{mode}-run1",
            start=workspace / f"{mode}-run1" / "eq" / "eq_3.xml"))]
    done = _run([sys.executable, str(workspace / script),
                 "-p", str(stub), *system, "-odir", str(destination),
                 *needed,
                 *PROTOCOL_ONLY],
                cwd=workspace / Path(script).parent)
    _refused(done, fragment="particle")
    assert _snapshot(destination) is None


def test_an_ais_source_that_is_not_a_trajectory_is_refused_before_output(workspace, tmp_path):
    mislabelled = tmp_path / "source.dcd"
    mislabelled.write_bytes(b"this is not a trajectory at all" + b"\x00" * 100)
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / ENTRY["AIS"][0]),
                 "-p", "../build/built.pdb", "-s", "../build/built.xml",
                 "-odir", str(destination),
                 "-source-traj", str(mislabelled), *V1, *PROTOCOL_ONLY],
                cwd=workspace / Path(ENTRY["AIS"][0]).parent)
    _refused(done, fragment="neither a DCD nor a NetCDF")
    assert _snapshot(destination) is None


def test_a_missing_ais_source_is_refused_before_output(workspace, tmp_path):
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / ENTRY["AIS"][0]),
                 "-p", "../build/built.pdb", "-s", "../build/built.xml",
                 "-odir", str(destination),
                 "-source-traj", str(tmp_path / "absent.dcd"), *V1],
                cwd=workspace / Path(ENTRY["AIS"][0]).parent)
    _refused(done, fragment="does not exist")
    assert _snapshot(destination) is None


@pytest.mark.parametrize("missing", ["-s2", "-p2"])
def test_ais_without_its_second_end_state_is_refused_by_name_before_output(missing, workspace,
                                                                           tmp_path):
    """AIS is a transformation V0 -> V1, and without V1 there is nothing to switch to.

    The refusal must name the flag. A run that fell back to something -- V0 against itself, or
    the retired tau switch -- would write a work table for a transformation nobody declared.
    """
    extra = list(V1)
    at = extra.index(missing)
    del extra[at:at + 2]
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / ENTRY["AIS"][0]),
                 "-p", "../build/built.pdb", "-s", "../build/built.xml",
                 "-odir", str(destination), "-source-traj", "../source.dcd",
                 *extra, *PROTOCOL_ONLY],
                cwd=workspace / Path(ENTRY["AIS"][0]).parent)
    _refused(done, fragment=f"AIS needs {missing}")
    assert _snapshot(destination) is None


@pytest.mark.parametrize("mode", ["split", "REST2"])
@pytest.mark.parametrize("flag, value", [("-s2", "../build/V1.xml"),
                                         ("-p2", "../build/built.pdb")])
def test_a_second_end_state_given_to_a_single_hamiltonian_run_is_refused_by_name(
        mode, flag, value, workspace, tmp_path):
    """A stage or a ladder has one Hamiltonian; `-s2` there is a file nothing would read.

    Accepting it silently is how a person believes they ran against V1 when they ran V0.
    """
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination, flag, value, *PROTOCOL_ONLY)
    _refused(done, fragment=f"{flag} names the second end state")
    assert _snapshot(destination) is None


# --- path collisions ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["split", "REST2", "AIS"])
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


@pytest.mark.parametrize("mode", ["split", "REST2", "AIS"])
def test_an_output_that_would_overwrite_an_input_is_refused(mode, workspace, tmp_path):
    """Writing the log over `built.pdb` destroys the topology the run needs to read.

    The topology, not `built.xml`: every mode reads `-p`, while a ladder reads its Systems from
    its group file (0.5.4) and never `built.xml`."""
    destination = tmp_path / "never"
    done = _launch(workspace, mode, destination, "-log", "../build/built.pdb")
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
