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
        "rREST2": {"protocol": "rREST2", "solvent": "implicit", "stages": tiny,
                   "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                             "number_of_exchanges": 2},
                   "reservoir": {"enabled": True, "path": "../reservoir.nc"},
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

    # A GENUINE PHASE-SPACE RESERVOIR, for exactly the reason the DCD above is genuine.
    #
    # The rREST2 declaration is now built and validated by the PREFLIGHT, before `-odir` exists,
    # rather than at helper-publication time with the run directory already created. That means
    # `--check` -- which is preflight and nothing else -- now opens this file, counts its frames
    # and reads its time axis. A path that merely does not exist tested nothing once the check
    # moved ahead of the output; it now correctly refuses.
    from md_tools.md.phase_space import PhaseSpaceWriter
    from md_tools.remd.generated import solute_document
    from md_tools.rest2 import identity as hamiltonian_identity
    from md_tools.run.preflight import check_scaling_plan, load_inputs

    # The identity has to be the TOP RUNG's, because that is the distribution a refresh would
    # draw from and the preflight now checks it. Built through the same helpers the ladder uses:
    # restating it by hand would only prove the fixture and the checker agree about a dictionary.
    loaded = load_inputs(str(root / "build" / "built.pdb"), str(root / "build" / "built.xml"))
    document = solute_document(loaded.pdb.topology, loaded.system, route=None)
    span = document.get("solute_atom_range")
    if span and document.get("solute_atom_indices_are_contiguous", False):
        indices = list(range(int(span[0]), int(span[1]) + 1))
    else:
        indices = list(range(int(document["n_solute_atoms"])))
    excluded = [tuple(int(a) for a in pair)
                for pair in (document.get("rest2") or {}).get("unscaled_central_bonds", [])]
    _audit, top_rung = check_scaling_plan(loaded, solute_indices=indices,
                                          excluded_bonds=excluded, tau=0.5, where="fixture")

    positions = frames.xyz[0]
    writer = PhaseSpaceWriter(
        root / "reservoir.nc", n_atoms=positions.shape[0], periodic=False,
        identity={"hamiltonian": hamiltonian_identity.identity_record(
            top_rung, tau=0.5, temperature_k=300.0, ensemble="NVT",
            solute_indices=indices, excluded_bonds=excluded)})
    try:
        for index in range(4):
            writer.append(positions=positions, velocities=positions * 0.0, box=None,
                          step=index * 5, time_ps=float(index))
    finally:
        writer.close()

    # THE STATE A LADDER'S GROUP FILE CONTINUES FROM. A ladder reads -s only from its group file
    # (0.5.4), and every line of the one `build-md` wrote names `-c eq/eq_3.xml`: without it each
    # launch below would be refused for the missing parent before reaching its own subject.
    from .conftest import write_starting_state

    for name in ("REST2", "rREST2"):
        write_starting_state(root, root / f"{name}-run1")
    return root


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
    "rREST2": ("rREST2-run1/rREST2.py", ["--groupfile", "remd_groupfile.1"]),
    "AIS": ("AIS-run1/AIS.py", ["-source-traj", "../source.dcd"]),
}
MODES = sorted(ENTRY)


def _launch(workspace, mode, destination: Path, *extra, environment=None):
    """Run a generated entry point from its own directory, as a person would.

    `-p` and `-s` reach the dataset's shared `build/` -- one level up from a run, and from the
    shared `min/` -- which is what every generated `run.sh` types too.
    """
    script, needed = ENTRY[mode]
    system = [] if mode in ("REST2", "rREST2") else ["-s", "../build/built.xml"]
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
    ladder = mode in ("REST2", "rREST2")
    argv = [sys.executable, str(workspace / script),
            "-p", "../build/built.pdb", *([] if ladder else ["-s", "../build/built.xml"]),
            "-odir", str(destination), *needed]
    if ladder and flag == "-s":
        # A ladder reads -s only from its group file (0.5.4): the missing System is on a line.
        group = workspace / Path(script).parent / "remd_groupfile.1"
        broken = tmp_path / "gone.group"
        broken.write_text(group.read_text(encoding="utf-8").replace(
            "../build/REST2/system_state0.xml", str(tmp_path / missing)), encoding="utf-8")
        argv[argv.index("remd_groupfile.1")] = str(broken)
    else:
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
    system = [] if mode in ("REST2", "rREST2") else ["-s", "../build/built.xml"]
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
                 "-source-traj", str(mislabelled), *PROTOCOL_ONLY],
                cwd=workspace / Path(ENTRY["AIS"][0]).parent)
    _refused(done, fragment="neither a DCD nor a NetCDF")
    assert _snapshot(destination) is None


def test_a_missing_ais_source_is_refused_before_output(workspace, tmp_path):
    destination = tmp_path / "never"
    done = _run([sys.executable, str(workspace / ENTRY["AIS"][0]),
                 "-p", "../build/built.pdb", "-s", "../build/built.xml",
                 "-odir", str(destination),
                 "-source-traj", str(tmp_path / "absent.dcd")],
                cwd=workspace / Path(ENTRY["AIS"][0]).parent)
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
