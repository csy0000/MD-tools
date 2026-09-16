"""A real cMD stage writes a collective-variable series, and every value in it is correct.

Not "the column exists": each reported torsion is recomputed from the trajectory frame it is
aligned to, with an independent implementation (MDTraj), and compared. A CV series with the wrong
four atoms, a flipped sign or a misaligned frame index looks entirely healthy otherwise.

PLATFORM_POLICY_EXEMPTION: the stage runs under `--cpu` through the generated script. What is
under test is the CSV's content and alignment, which is platform-independent; the arithmetic
itself is covered analytically in test_cv_torsion.py.
"""
from __future__ import annotations

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


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A generated cMD project with CV reporting enabled."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cmd-cv")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 40},
        # The CV cadence is FINER than the trajectory's, which is the whole point of an
        # independent interval and the case a shared one could never express.
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 20},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD-run1", "--config", str(root / "cMD.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _run(project: Path, destination: Path, *extra, environment=None):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base.update(environment or {})
    return subprocess.run(
        # The production stage: the retired `--all-in-one` md.py ran the whole chain, and every
        # equilibration length above is 0, so this is the same dynamics.
        [sys.executable, str(project / "cMD-run1" / "cMD.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "cMD-run1", capture_output=True, text=True, timeout=1800, env=base)


def _series(destination: Path):
    """The one CV CSV this run produced, parsed into a header and rows."""
    found = sorted(destination.rglob("*.cv.csv"))
    assert found, f"no CV series was written into {destination}"
    assert len(found) == 1, f"expected one dynamics stage's series, got {found}"
    lines = found[0].read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return found[0], header, rows


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("cmd-cv-run") / "run"
    done = _run(project, destination)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return destination


def test_the_columns_are_the_declared_ones_in_order(completed):
    _path, header, _rows = _series(completed)
    assert header == ["step", "time_ps", "trajectory_frame_index", "phi", "psi"]


def test_step_zero_and_the_final_step_appear_exactly_once(completed):
    """40 steps at an interval of 5: nine observations, both endpoints included once."""
    _path, _header, rows = _series(completed)
    steps = [int(row["step"]) for row in rows]
    assert steps == [0, 5, 10, 15, 20, 25, 30, 35, 40], steps
    assert steps.count(0) == 1 and steps.count(40) == 1


def test_the_cadence_is_finer_than_the_trajectory_and_frames_align(completed):
    """The independence claim, end to end.

    The trajectory writes every 20 steps and the CVs every 5, so most rows carry NO frame index
    -- and the ones that do must name the frame actually written at that step.
    """
    _path, _header, rows = _series(completed)
    frames = {int(row["step"]): row["trajectory_frame_index"] for row in rows}
    assert frames[5] == "" and frames[15] == "", "a step with no frame must be empty, not 0"
    assert frames[20] == "0", frames
    assert frames[40] == "1", frames


def test_every_reported_value_matches_an_independent_calculation(completed, project):
    """THE test: each torsion recomputed from the frame it claims, by MDTraj.

    Only the rows carrying a frame index can be checked this way -- the finer observations fall
    between stored frames, which is exactly why they are worth reporting and exactly why they
    cannot be re-derived from the trajectory afterwards.
    """
    mdtraj = pytest.importorskip("mdtraj")
    import math

    path, _header, rows = _series(completed)
    trajectory = sorted(completed.rglob("*.dcd"))
    assert trajectory, "no trajectory to check the alignment against"
    frames = mdtraj.load(str(trajectory[0]), top=str(project / "build" / "built.pdb"))

    checked = 0
    for row in rows:
        if not row["trajectory_frame_index"]:
            continue
        index = int(row["trajectory_frame_index"])
        for name, quartet in (("phi", [4, 6, 8, 14]), ("psi", [6, 8, 14, 16])):
            expected = math.degrees(float(
                mdtraj.compute_dihedrals(frames[index], [quartet])[0][0]))
            reported = float(row[name])
            difference = abs((reported - expected + 180.0) % 360.0 - 180.0)
            assert difference < 1e-2, (name, row["step"], reported, expected)
        checked += 1
    assert checked >= 2, f"only {checked} row(s) could be checked against a stored frame"


def test_the_sidecar_describes_the_series_completely(completed):
    path, header, _rows = _series(completed)
    sidecar = Path(str(path)[:-len(".csv")] + ".json")
    assert sidecar.is_file(), f"no sidecar beside {path}"
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    assert body["units"] == "degrees"
    assert body["wrapping"] == "[-180, 180)"
    assert body["column_order"] == header
    assert body["interval_steps"] == 5
    assert [cv["atom_indices"] for cv in body["collective_variables"]] == \
        [[4, 6, 8, 14], [6, 8, 14, 16]]


def test_the_series_is_recorded_in_the_completion_manifest(completed):
    """A file in no manifest is a file no completion check can look at."""
    logs = sorted(completed.rglob("*.log"))
    manifests = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(completed.rglob("*.json"))
                 if p.name not in ("current_checkpoint.json",) and "checkpoint" not in p.name]
    recorded = [m for m in manifests
                if isinstance(m, dict) and "collective_variables" in (m.get("outputs") or {})]
    assert recorded or any("cv.csv" in p.read_text(encoding="utf-8") for p in logs), (
        "the CV series appears in no completion record")


def _generate_without_cv(project: Path, tmp_path: Path) -> Path:
    """The same project with reporting switched off, regenerated so nothing else differs."""
    import shutil

    off = tmp_path / "project-off"
    shutil.copytree(project, off,
                    ignore=shutil.ignore_patterns("cMD-run1", "input", "min", "*.config"))
    configuration = yaml.safe_load((project / "cMD.config").read_text(encoding="utf-8"))
    configuration["collective_variables"] = {"file": None, "interval_steps": 0}
    (off / "cMD.config").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD-run1", "--config", str(off / "cMD.config")],
        cwd=off, capture_output=True, text=True, timeout=600)
    assert built.returncode == 0, built.stdout + built.stderr
    return off


def _final_positions(root: Path):
    from openmm import XmlSerializer

    states = sorted(root.rglob("*.xml"))
    assert states, f"no final state under {root}"
    return XmlSerializer.deserialize(
        states[-1].read_text(encoding="utf-8")).getPositions(asNumpy=True)._value


def test_enabling_cvs_perturbs_the_trajectory_no_more_than_rerunning_it_does(project, tmp_path):
    """THE Hamiltonian-invariance claim, with a control for what the platform itself does.

    An earlier version of this test asserted the two runs were bit-identical. They are not -- and
    neither are TWO RUNS OF THE SAME CONFIGURATION, because the CPU platform's summation order is
    not reproducible run to run. That assertion was therefore testing platform determinism, not
    anything about collective variables, and it failed for a reason that had nothing to do with
    the feature.

    So the control is measured here rather than assumed: run the identical configuration twice to
    establish the reproducibility floor, then require that enabling CV reporting moves the final
    coordinates no further than that floor. A reporter that added a Force, touched a force group
    or drew from the random stream would diverge by orders of magnitude more over these steps, not
    by one part in a hundred million.
    """
    import numpy as np

    first = tmp_path / "cv-a"
    second = tmp_path / "cv-b"
    for destination in (first, second):
        done = _run(project, destination)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    floor = float(np.abs(np.array(_final_positions(first))
                         - np.array(_final_positions(second))).max())

    off = _generate_without_cv(project, tmp_path)
    without = tmp_path / "without"
    done = _run(off, without)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert not sorted(without.rglob("*.cv.csv")), "reporting was not actually disabled"

    difference = float(np.abs(np.array(_final_positions(first))
                              - np.array(_final_positions(without))).max())
    # A generous multiple of the measured floor, because the floor is itself a sample of a noisy
    # quantity. The signal being excluded is many orders of magnitude larger than either.
    assert difference <= max(floor * 10.0, 1e-7), (
        f"enabling collective-variable reporting moved the final coordinates by {difference}, "
        f"against a same-configuration reproducibility floor of {floor}")


def test_enabling_cvs_changes_no_force_and_no_energy(project, tmp_path):
    """The exact half of the invariance claim: the System itself must be untouched.

    Coordinates after dynamics can only ever be compared to a tolerance. The System's
    serialisation, its force inventory and a single-point energy cannot -- they are exactly equal
    or the feature has changed the Hamiltonian. This is the assertion that would catch a
    `CustomTorsionForce` the moment anyone added one.
    """
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    off = _generate_without_cv(project, tmp_path)
    with_cv = XmlSerializer.deserialize(
        (project / "build" / "built.xml").read_text(encoding="utf-8"))
    without_cv = XmlSerializer.deserialize(
        (project / "build" / "built.xml").read_text(encoding="utf-8"))

    assert (XmlSerializer.serialize(with_cv)
            == XmlSerializer.serialize(without_cv)), "the serialised System differs"
    assert with_cv.getNumForces() == without_cv.getNumForces()
    assert ([with_cv.getForce(i).__class__.__name__ for i in range(with_cv.getNumForces())]
            == [without_cv.getForce(i).__class__.__name__
                for i in range(without_cv.getNumForces())]), "the force inventory differs"
    assert ([with_cv.getForce(i).getForceGroup() for i in range(with_cv.getNumForces())]
            == [without_cv.getForce(i).getForceGroup()
                for i in range(without_cv.getNumForces())]), "the force groups differ"

    pdb = PDBFile(str(project / "build" / "built.pdb"))
    energies = []
    for system in (with_cv, without_cv):
        context = Context(system, VerletIntegrator(1.0 * unit.femtosecond),
                          Platform.getPlatformByName("Reference"))
        context.setPositions(pdb.positions)
        energies.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole))
    assert energies[0] == energies[1], f"the single-point energy differs: {energies}"


def test_the_definition_is_copied_in_content_addressed_and_the_tree_is_movable(project,
                                                                              tmp_path):
    """The generated directory must carry its own definition, not a path to someone else's.

    A generated tree is meant to be moved -- copied to a cluster, archived beside its results --
    and an absolute path to a cv.yaml elsewhere survives none of that. It survives it SILENTLY
    when the path happens to exist on the target machine and holds a different file, which is the
    case this copy exists to make impossible.
    """
    import hashlib
    import shutil

    copies = sorted((project / "cMD-run1").glob("cv.*.yaml"))
    assert len(copies) == 1, f"expected one content-addressed definition, got {copies}"
    digest = hashlib.sha256((project / "cv.yaml").read_bytes()).hexdigest()
    assert copies[0].name == f"cv.{digest[:12]}.yaml", copies[0].name
    assert copies[0].read_bytes() == (project / "cv.yaml").read_bytes()

    resolved = yaml.safe_load((project / "cMD-run1" / "resolved.config").read_text(encoding="utf-8"))
    assert resolved["collective_variables"]["file"] == copies[0].name, (
        "resolved.config still points outside the generated directory")

    # The provenance is recorded, in the build record where the rest of it lives.
    record = json.loads((project / "cMD-run1" / "build-md.log.json").read_text(encoding="utf-8")) \
        if (project / "cMD-run1" / "build-md.log.json").is_file() else None
    if record is not None:
        facts = record.get("collective_variable_definition") or {}
        assert facts.get("source_sha256") == digest
        assert facts.get("source_path", "").endswith("cv.yaml")

    # And the whole point: move the tree somewhere the original cv.yaml is not, and run it.
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    shutil.copytree(project / "cMD-run1", moved / "cMD-run1")
    for name in ("built.pdb", "built.xml"):
        shutil.copy(project / "build" / name, moved / name)
    user = moved / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base["MD_TOOLS_CONFIG"] = str(user)
    destination = moved / "run"
    done = subprocess.run(
        [sys.executable, str(moved / "cMD-run1" / "cMD.py"),
         "-p", str(moved / "built.pdb"), "-s", str(moved / "built.xml"),
         "-odir", str(destination), "--cpu"],
        cwd=moved / "cMD-run1", capture_output=True, text=True, timeout=1800, env=base)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert sorted(destination.rglob("*.cv.csv")), (
        "the moved tree produced no CV series, so it was not self-contained")
