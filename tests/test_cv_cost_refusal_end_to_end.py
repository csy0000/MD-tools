"""A malformed stored cost record is refused BEFORE anything on disk changes.

WHY THIS FILE EXISTS SEPARATELY FROM THE PARSER MATRIX

    `test_cv_cost_schema.py` proves the parser refuses each malformed shape. That says nothing
    about *when* the refusal happens. A validator that runs after a run has truncated a series,
    replaced a pointer or rewritten a table is not protecting anything: it reports damage rather
    than preventing it, and the run that was refused has already destroyed the run it was
    continuing.

    So each case here doctors an authoritative record on disk, snapshots the entire output tree
    by digest, invokes the real command, and requires both that it refuses and that no scientific
    output moved.

WHAT MAY CHANGE, AND WHY THAT IS NOT A LOOPHOLE

    Three things: `<name>.out`, `<name>.log` and `<name>.runstate.json`. They are where the
    invocation reports that it refused and why -- the human record, the machine record, and the
    status field a later reader consults to learn the run failed. CLAUDE.md's "completion is read
    from a machine record" depends on them being written, and a tool that refused while leaving
    no trace would satisfy a stricter assertion here by being less useful.

    Everything else is held byte-identical: every CV series, work table, observation table,
    trajectory, checkpoint generation, pointer and completion manifest. That is the property
    worth having -- a refusal must not damage the run it declined to continue -- and one of these
    cases caught a real violation of it, where AIS truncated `observations.csv` and its staged
    trajectory some two hundred lines before validating the record it then refused.

    The refusal's own text is read from the subprocess's stderr, which lives in this process and
    not in the tree.

PLATFORM_POLICY_EXEMPTION: `--cpu` throughout. What is under test is whether a refusal happens
before a filesystem mutation, which no platform changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
"""


#: NO EXEMPTIONS. An earlier version of this file excused `.out`, `.log` and `runstate.json` on
#: the grounds that a refusal must be able to report itself. That was the wrong boundary: those
#: files are the prior run's machine-readable provenance and completion status, and a rejected
#: continuation that overwrites them has replaced the authoritative record of a run it never
#: started. "Completion is read from a machine record" is exactly why they may not move.
#:
#: The refusal reports itself on stderr, captured by the test OUTSIDE the protected tree. Once
#: preflight succeeds and execution begins, normal runtime logging resumes -- that is a different
#: phase, and the accepted-invocation tests cover it.
#:
#: Only filesystem ACCESS times are ignored: reading a file to validate it updates atime, and
#: turning that into a failure would make validation itself look like mutation.


def _tree(root: Path) -> dict:
    """Every file, directory entry and symlink target under the protected tree.

    Digest for regular files; the target for symlinks; a marker for directories -- so a refusal
    that deletes and recreates identical bytes, or replaces a file with a link, is still caught.
    `st_ino`/`st_mtime_ns` ride along so a rewrite of identical content is visible too.
    """
    state = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            state[key] = ("symlink", os.readlink(path))
        elif path.is_dir():
            state[key] = ("dir", None)
        else:
            stat = path.stat()
            state[key] = ("file", hashlib.sha256(path.read_bytes()).hexdigest(),
                          stat.st_ino, stat.st_mtime_ns)
    return state


def _assert_unchanged(root: Path, before: dict, what: str):
    """Nothing under the protected tree moved. Not one byte, entry, link or inode."""
    after = _tree(root)
    changed = sorted(key for key in set(before) | set(after)
                     if before.get(key) != after.get(key))
    assert not changed, (
        f"{what} modified the protected output tree before refusing: {changed}")


def _environment(root: Path, **extra):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base["OPENMM_CPU_THREADS"] = "1"
    base.update(extra)
    return base


def _build_top(root: Path):
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")


def _cv_reporting_checkpoints(destination: Path) -> Path:
    """The checkpoint directory of the stage that REPORTED collective variables.

    A cMD chain commits generations for its minimisation and equilibration stages too, and none
    of those carries a CV prefix; picking the last directory by name found one of them.
    """
    from md_tools.openmm.checkpoint import POINTER_NAME, read_committed

    for path in sorted(destination.rglob("*.checkpoints")):
        if not (path / POINTER_NAME).is_file():
            continue
        state = read_committed(path).get("state") or {}
        if state.get("cv_prefix"):
            return path
    raise AssertionError(f"no committed generation under {destination} carries a CV prefix")


def _doctor_checkpoint_cost(checkpoints: Path, mutate):
    """Change the committed generation's stored cost, leaving its digests self-consistent.

    The pointer and the sidecar's own checksum still agree afterwards, so the record reaches the
    cost validator rather than being caught by an unrelated integrity check first -- which would
    make the case pass for the wrong reason.
    """
    from md_tools.openmm.checkpoint import POINTER_NAME

    pointer = json.loads((checkpoints / POINTER_NAME).read_text(encoding="utf-8"))
    sidecar = checkpoints / "checkpoints" / pointer["sidecar"]
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    prefix = document["state"]["cv_prefix"]
    assert prefix and prefix.get("cost"), "no committed CV cost to doctor"
    mutate(prefix["cost"])
    sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- cMD continuation ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cmd_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("refusal-cmd")
    _build_top(root)
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 60},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260905},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./cMD-run1", "--config", str(root / "cMD.config"),
               "--all-in-one"],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0
    return root


def _run_cmd(root: Path, destination: Path, environment=None):
    return subprocess.run(
        [sys.executable, str(root / "cMD-run1" / "md.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-odir", str(destination), "--cpu"],
        cwd=root / "cMD-run1", capture_output=True, text=True, timeout=1800,
        env=_environment(root, **(environment or {})))


CMD_MUTATIONS = [
    ("fractional-counter", lambda cost: cost["cumulative"].__setitem__("cv_observations", 7.5)),
    ("numeric-string", lambda cost: cost["cumulative"].__setitem__("cv_observations", "7")),
    ("boolean-counter", lambda cost: cost["segment"].__setitem__("cv_evaluations", True)),
    ("negative-counter", lambda cost: cost["segment"].__setitem__("cv_observations", -1)),
    ("infinite-wall-time",
     lambda cost: cost["segment"].__setitem__("wall_seconds", float("inf"))),
    ("segment-exceeds-cumulative",
     lambda cost: cost["segment"].__setitem__("cv_observations", 9999)),
    ("unsupported-schema-version", lambda cost: cost.__setitem__("schema_version", 7)),
]


@pytest.mark.parametrize("mutate", [m for _id, m in CMD_MUTATIONS],
                         ids=[case_id for case_id, _m in CMD_MUTATIONS])
def test_cmd_continuation_refuses_a_malformed_cost_and_writes_nothing(cmd_project, tmp_path,
                                                                      mutate):
    """An interrupted cMD stage whose committed cost record is malformed.

    The stage continues automatically from its committed generation, so this is the ordinary
    resume path -- and the point at which it would truncate its appendable streams to the
    committed counts. A refusal has to come first.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "run"
    crashed = _run_cmd(cmd_project, destination,
                       {FAULT_ENVIRONMENT: "after-pointer-replace",
                        FAULT_AFTER_ENVIRONMENT: "4"})
    assert crashed.returncode != 0

    _doctor_checkpoint_cost(_cv_reporting_checkpoints(destination), mutate)

    before = _tree(destination)
    refused = _run_cmd(cmd_project, destination)
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused cMD continuation")


# --- AIS: partial-path continuation, completed-path skip, global aggregation --------------------

@pytest.fixture(scope="module")
def ais_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("refusal-ais")
    _build_top(root)
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 40,
                "observation_interval_steps": 10, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 5},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 10},
        "dynamics": {"seed": 20260905},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _run_ais(root: Path, destination: Path, *extra, environment=None):
    return subprocess.run(
        [sys.executable, str(root / "AIS-run1" / "AIS.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-source-traj", str(root / "source.dcd"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=root / "AIS-run1", capture_output=True, text=True, timeout=1800,
        env=_environment(root, **(environment or {})))


AIS_MANIFEST_MUTATIONS = [
    ("fractional-counter",
     lambda cost: cost["cumulative"].__setitem__("cv_observations", 5.0)),
    ("numeric-string", lambda cost: cost["cumulative"].__setitem__("cv_evaluations", "10")),
    ("boolean-counter", lambda cost: cost["segment"].__setitem__("cv_observations", True)),
    ("nan-wall-time", lambda cost: cost["cumulative"].__setitem__("wall_seconds",
                                                                 float("nan"))),
    ("missing-cumulative", lambda cost: cost.pop("cumulative")),
    ("missing-schema-version", lambda cost: cost.pop("schema_version")),
]


@pytest.mark.parametrize("mutate", [m for _id, m in AIS_MANIFEST_MUTATIONS],
                         ids=[case_id for case_id, _m in AIS_MANIFEST_MUTATIONS])
def test_ais_completed_path_skip_refuses_a_malformed_cost(ais_project, tmp_path, mutate):
    """A completed path whose manifest cost is malformed must not be skipped as though fine.

    This is the path that is never re-read again: skipping it silently is how a corrupt record
    becomes the permanent answer.
    """
    destination = tmp_path / "run"
    assert _run_ais(ais_project, destination).returncode == 0

    marker = destination / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    mutate(record["collective_variable_cost"])
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    before = _tree(destination)
    refused = _run_ais(ais_project, destination, "--resume")
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused AIS completed-path skip")


def test_ais_partial_path_continuation_refuses_a_malformed_checkpoint_cost(ais_project,
                                                                          tmp_path):
    """An interrupted path whose committed prefix cost is malformed.

    The continuation is about to truncate that series to the committed row count. Refusing
    afterwards would mean the refusal destroyed the rows it was complaining about.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "partial"
    crashed = _run_ais(ais_project, destination,
                       environment={FAULT_ENVIRONMENT: "after-work-row",
                                    FAULT_AFTER_ENVIRONMENT: "2"})
    assert crashed.returncode != 0

    _doctor_checkpoint_cost(
        destination / "path_0000",
        lambda cost: cost["cumulative"].__setitem__("cv_observations", 2.5))

    before = _tree(destination)
    refused = _run_ais(ais_project, destination, "--resume")
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused AIS partial-path continuation")


def test_ais_global_aggregation_refuses_a_manifest_it_cannot_parse(ais_project, tmp_path):
    """The aggregate is assembled from every completed manifest; one bad record stops it.

    Writing the table from the records it *could* read would produce a campaign total silently
    short of a path, which is the failure mode the per-path records exist to make impossible.
    """
    destination = tmp_path / "aggregate"
    assert _run_ais(ais_project, destination).returncode == 0

    marker = destination / "path_0001" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["collective_variable_cost"]["cumulative"]["cv_evaluations"] = 3
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    before = _tree(destination)
    refused = _run_ais(ais_project, destination, "--resume")
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused AIS aggregation")


# --- REST2 / rREST2 continuation -----------------------------------------------------------------

@pytest.fixture(scope="module")
def ladder_project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("refusal-ladder")
    _build_top(root)
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 10,
                  "number_of_exchanges": 4},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260905},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=900).returncode == 0

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    context = openmm.Context(system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


def _run_ladder(root: Path, destination: Path, *extra, environment=None):
    return subprocess.run(
        [sys.executable, str(root / "REST2-run1" / "REST2.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-c", str(root / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=root / "REST2-run1", capture_output=True, text=True, timeout=1800,
        env=_environment(root, **(environment or {})))


def test_ladder_continuation_refuses_a_malformed_per_state_cost(ladder_project, tmp_path):
    """An interrupted ladder whose committed per-state prefix cost is malformed.

    The ladder checkpoint is a NetCDF file whose `extra_json` carries the per-state prefixes, so
    the mutation goes in there. A continuation is about to truncate every state's series to the
    committed count; the refusal must precede that.
    """
    import netCDF4

    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT  # noqa: F401 - parity

    destination = tmp_path / "ladder"
    crashed = _run_ladder(ladder_project, destination,
                          environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                       "MD_TOOLS_FAIL_LADDER_AT": "after-cv-row",
                                       "MD_TOOLS_FAIL_PROPAGATION_AFTER": "2"})
    assert crashed.returncode != 0

    checkpoint = destination / "REST2_checkpoint.nc"
    dataset = netCDF4.Dataset(str(checkpoint), "a")
    try:
        extra = json.loads(dataset.extra_json)
        extra["cv_prefix"]["states"][0]["cost"]["cumulative"]["cv_observations"] = "5"
        dataset.extra_json = json.dumps(extra, sort_keys=True)
    finally:
        dataset.close()

    before = _tree(destination)
    refused = _run_ladder(ladder_project, destination, "--resume")
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused ladder continuation")


def test_ladder_extension_refuses_a_malformed_parent_cost(ladder_project, tmp_path):
    """An extension reads its parent read-only; a malformed parent must be refused before the
    extension directory holds anything."""
    destination = tmp_path / "parent"
    assert _run_ladder(ladder_project, destination).returncode == 0

    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    manifest["collective_variables"]["cost"]["per_state"][0]["cumulative"]["cv_evaluations"] = 3.0
    (destination / "restart.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    before = _tree(destination)
    extension = tmp_path / "extension"
    refused = _run_ladder(ladder_project, extension, "--extend", "2",
                          "--extend-from", str(destination))
    assert refused.returncode != 0, refused.stdout[-2000:]
    _assert_unchanged(destination, before, "a refused ladder extension")
    assert not list(extension.glob("cv_state*.csv")), (
        "the extension wrote CV outputs before refusing its parent")


# --- the PUBLIC entry point, at the same boundary as the generated wrappers ---------------------

def test_md_run_refuses_a_malformed_record_without_touching_the_tree(cmd_project, tmp_path):
    """`md-openmm md-run` must honour the same read-only boundary as a generated script.

    The two are different surfaces onto one runtime, and a boundary that held for only one of
    them would be a boundary nobody could rely on: `md-run` creates `-odir` and writes
    `resolved.config` itself, before dispatching to `stage_main`, so it is the surface with the
    most opportunity to touch the tree ahead of the validation.

    The generated wrapper is covered by the cases above; this covers the public command against
    the same doctored record.
    """
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    destination = tmp_path / "public"
    crashed = _run_cmd(cmd_project, destination,
                       {FAULT_ENVIRONMENT: "after-pointer-replace",
                        FAULT_AFTER_ENVIRONMENT: "4"})
    assert crashed.returncode != 0

    _doctor_checkpoint_cost(
        _cv_reporting_checkpoints(destination),
        lambda cost: cost["cumulative"].__setitem__("cv_observations", "9"))

    generated = cmd_project / "input"
    inputs = sorted(generated.glob("*.in"))
    assert inputs, f"no .in file in {generated}"
    stage_input = next((p for p in inputs if p.stem == "cMD"), inputs[-1])

    before = _tree(destination)
    refused = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
         "-i", str(stage_input), "-p", str(cmd_project / "build" / "built.pdb"),
         "-s", str(cmd_project / "build" / "built.xml"), "-odir", str(destination), "--cpu"],
        cwd=generated, capture_output=True, text=True, timeout=1800,
        env=_environment(cmd_project))
    assert refused.returncode != 0, refused.stdout[-3000:]
    _assert_unchanged(destination, before, "a refused md-run continuation")
    # The contract asks the refusal to identify the record, the scope and the FIELD -- so the
    # assertion is on the field name, not on the word "cost".
    combined = refused.stdout + refused.stderr
    assert "cv_observations" in combined, combined[-2000:]
    assert "cMD.cv.csv" in combined, combined[-2000:]
