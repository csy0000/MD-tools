"""Walker identity and the ladder permutation, refused before completion or extension.

WHY THIS FILE

    `walker_index` is what makes a state-centric CV series joinable to a walker-centric
    analysis: `cv_state2.csv` holds whatever occupied rung 2, and this column says who. Every
    other field in those files was validated -- the step grid, the state index, tau, the
    exchange phase, the frame reference, the finiteness of every value, the digest of the whole
    file -- and this one was not read at all. A walker of -1, of `n_states`, of "2.5", or the
    same walker occupying two rungs at once produced a completed run, an authoritative
    completion marker, and a set of files that read back perfectly.

    An exchange PERMUTES walkers among rungs. It never creates, destroys or duplicates one. So
    at every observation step the walkers across the ladder are exactly 0 .. n_states - 1, each
    once, and anything else describes a ladder that never ran.

WHY THE DIGESTS ARE REWRITTEN

    Every corruption here is applied to the CSV AND to the digest the manifest records for it.
    Leaving the digest stale would make each case pass for the wrong reason -- the digest check
    fires first and the walker logic never runs. Rewriting it is what a plausible bad file looks
    like: internally consistent, correctly checksummed, and scientifically wrong.

PLATFORM_POLICY_EXEMPTION: the ladder runs once under `--cpu` purely to produce a real,
correctly-shaped set of CV files and a real manifest to corrupt. Nothing here measures
dynamics, and the validation under test reads files and never touches a platform.
"""
from __future__ import annotations

import hashlib
import json
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

pytestmark = pytest.mark.slow

STATES = 3
EXCHANGE_EVERY = 10
EXCHANGES = 4
CV_EVERY = 10

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


from .conftest import ladder_group_file  # noqa: E402

@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One completed three-rung REST2 ladder. Three, so a permutation is not just a swap."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("walkers")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4): `build-md` refuses to generate one until
    # `build/REST2/` exists, as `md-openmm build-top --rest2-scaler` writes it.
    from .conftest import make_states_for

    make_states_for(root, root / "REST2.config")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr

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

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    run = root / "run-run1"
    ran = subprocess.run(
        [sys.executable, str(root / "REST2-run1" / "REST2.py"),
         "-p", str(root / "build" / "built.pdb"),
         "--groupfile", str(ladder_group_file(root, run)),
         "-odir", str(run), "--cpu"],
        cwd=root / "REST2-run1", capture_output=True, text=True, timeout=1800, env=base)
    assert ran.returncode == 0, ran.stdout[-4000:] + ran.stderr[-4000:]
    return root, run


def _copy(finished, tmp_path):
    _root, run = finished
    destination = tmp_path / "run"
    shutil.copytree(run, destination)
    return destination


def _block(destination: Path):
    return json.loads((destination / "restart.json").read_text(encoding="utf-8"))[
        "collective_variables"]


def _reseal(destination: Path, block):
    """Rewrite each recorded digest and size to match the file as it now stands.

    Without this every case below would be caught by the digest check instead of by the check
    it is meant to exercise, and the file would prove nothing about walker validation.
    """
    for entry in block["series"]:
        path = destination / entry["csv"]
        entry["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["csv_bytes"] = path.stat().st_size
    return block


def _edit(destination: Path, state: int, *, row: int, column: str, value: str):
    """Replace one field of one committed row, leaving the file otherwise byte-identical."""
    path = destination / f"cv_state{state}.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[row + 1].split(",")
    cells[header.index(column)] = value
    lines[row + 1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify(destination: Path, block):
    from md_tools.remd.cv_states import verify_manifest_entries

    return verify_manifest_entries(destination, block)


def test_a_healthy_ladder_passes(finished, tmp_path):
    """The control. Without it every case below could pass because the check refuses everything."""
    destination = _copy(finished, tmp_path)
    assert _verify(destination, _block(destination)) == []


def test_a_negative_walker_is_refused(finished, tmp_path):
    destination = _copy(finished, tmp_path)
    _edit(destination, 0, row=1, column="walker_index", value="-1")
    problems = _verify(destination, _reseal(destination, _block(destination)))
    assert any("walker -1" in problem for problem in problems), problems


def test_a_walker_equal_to_the_state_count_is_refused(finished, tmp_path):
    """`n_states` is the classic off-by-one: walkers are numbered 0 .. n-1."""
    destination = _copy(finished, tmp_path)
    _edit(destination, 0, row=1, column="walker_index", value=str(STATES))
    problems = _verify(destination, _reseal(destination, _block(destination)))
    assert any(f"walker {STATES}" in problem for problem in problems), problems


def test_a_non_integer_walker_is_refused(finished, tmp_path):
    """A walker is an identity. `1.5` is not a walker that half-occupied a rung."""
    destination = _copy(finished, tmp_path)
    _edit(destination, 0, row=1, column="walker_index", value="1.5")
    problems = _verify(destination, _reseal(destination, _block(destination)))
    assert any("not an integer" in problem for problem in problems), problems


def test_a_walker_duplicated_across_states_at_one_step_is_refused(finished, tmp_path):
    """Every field stays in range and every file stays self-consistent. Only the SET is wrong."""
    destination = _copy(finished, tmp_path)
    block = _block(destination)
    # Whatever occupies state 1 at row 1, make state 0 claim it too at the same step.
    lines = (destination / "cv_state1.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    theirs = lines[2].split(",")[header.index("walker_index")]
    _edit(destination, 0, row=1, column="walker_index", value=theirs)
    problems = _verify(destination, _reseal(destination, block))
    assert any("not a permutation" in problem and "more than one state" in problem
               for problem in problems), problems


def test_a_walker_missing_at_one_step_is_refused(finished, tmp_path):
    """One step short in one file: that step has a rung nobody occupied."""
    destination = _copy(finished, tmp_path)
    path = destination / "cv_state0.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    block = _reseal(destination, _block(destination))
    for entry in block["series"]:
        if entry["state_index"] == 0:
            entry["rows"] -= 1
            entry["expected_steps"] = entry["expected_steps"][:-1]
            entry["final_step"] = entry["expected_steps"][-1]
    problems = _verify(destination, block)
    assert any("record no observation" in problem for problem in problems), problems


def test_two_state_files_swapped_whole_are_refused(finished, tmp_path):
    """Syntactically perfect, scientifically wrong: every row valid, in the wrong file."""
    destination = _copy(finished, tmp_path)
    block = _block(destination)
    first = (destination / "cv_state0.csv").read_text(encoding="utf-8")
    second = (destination / "cv_state1.csv").read_text(encoding="utf-8")
    (destination / "cv_state0.csv").write_text(second, encoding="utf-8")
    (destination / "cv_state1.csv").write_text(first, encoding="utf-8")
    problems = _verify(destination, _reseal(destination, block))
    # Caught by the rows' own state index: a swapped file reports the other rung's identity.
    assert any("reports state_index" in problem for problem in problems), problems


def test_an_invalid_ladder_is_refused_an_extension_before_outputs_appear(finished, tmp_path):
    """The end-to-end promise: a bad parent is refused BEFORE the extension writes anything."""
    root, _run = finished
    destination = _copy(finished, tmp_path)
    _edit(destination, 0, row=1, column="walker_index", value=str(STATES))
    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    manifest["collective_variables"] = _reseal(destination, manifest["collective_variables"])
    (destination / "restart.json").write_text(json.dumps(manifest), encoding="utf-8")

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base["MD_TOOLS_CONFIG"] = str(root / "user.config")
    extension = tmp_path / "extended"
    done = subprocess.run(
        [sys.executable, str(root / "REST2-run1" / "REST2.py"),
         "-p", str(root / "build" / "built.pdb"),
         "--groupfile", str(ladder_group_file(root, extension)),
         "-odir", str(extension), "--cpu",
         "--extend", "2", "--extend-from", str(destination)],
        cwd=root / "REST2-run1", capture_output=True, text=True, timeout=1800, env=base)
    assert done.returncode != 0, done.stdout[-3000:]
    # The refusal itself is written to the run's own log, which is where a reader looking at a
    # failed extension goes; the launcher only points at it.
    reported = (extension / "REST2.out").read_text(encoding="utf-8")
    assert f"walker {STATES}" in reported, reported[-3000:]
    assert "cannot be extended" in reported, reported[-3000:]
    assert not list(extension.glob("cv_state*.csv")), (
        "the extension wrote CV outputs before refusing its parent")
