"""A completed ladder OWNS its CV series, and stops being complete if they change.

WHAT WAS WRONG

    The typed inventory named `remdN.cv.csv` and `remdN.cv.json`, so collisions and `--overwrite`
    governed them. The authoritative completion manifest did not. A run could therefore finish,
    record itself complete, and have any of its CV files deleted, truncated, edited or swapped
    afterwards -- and every validator would still call the run complete, because nothing had ever
    written down what those files were supposed to contain.

    An extension would then continue from that parent, and the extended series would be the old
    column concatenated with a new one, with nothing marking the join.

PLATFORM_POLICY_EXEMPTION: the ladder runs under `--cpu`. What is under test is whether a
manifest describes its own files and whether a validator re-reads them -- bookkeeping, identical
on every platform. The same runtime is exercised on CUDA in `test_cv_cuda_lanes.py`.
"""
from __future__ import annotations

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
CV_EVERY = 5
TOTAL = EXCHANGE_EVERY * EXCHANGES

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


def _project(root: Path, *, cv: bool):
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
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
        "collective_variables": ({"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY}
                                 if cv else {"file": None, "interval_steps": 0}),
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "built.pdb"))
    system = XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    return _project(tmp_path_factory.mktemp("ladder-cv-manifest"), cv=True)


def _run(project: Path, destination: Path, *extra, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    done = subprocess.run(
        [sys.executable, str(project / "REST2" / "REST2.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "REST2", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("ladder-cv-manifest-run") / "run"
    _run(project, destination)
    return destination


def _manifest(destination: Path):
    return json.loads((destination / "restart.json").read_text(encoding="utf-8"))


def test_the_manifest_records_every_series_with_its_identity(completed):
    block = _manifest(completed).get("collective_variables")
    assert block is not None, "the completion manifest does not record the CV series"
    assert len(block["series"]) == STATES
    for index, entry in enumerate(block["series"]):
        assert entry["state_index"] == index
        assert entry["csv"] == f"remd{index}.cv.csv"
        assert entry["sidecar"] == f"remd{index}.cv.json"
        assert len(entry["csv_sha256"]) == 64 and len(entry["sidecar_sha256"]) == 64
        assert entry["csv_bytes"] > 0 and entry["sidecar_bytes"] > 0
        assert entry["rows"] == TOTAL // CV_EVERY + 1
        assert entry["first_step"] == 0 and entry["final_step"] == TOTAL
        assert entry["interval_steps"] == CV_EVERY
        assert entry["expected_steps"] == list(range(0, TOTAL + 1, CV_EVERY))
        assert entry["units"] == "degrees" and entry["wrapping"] == "[-180, 180)"
        assert entry["exchange_phase"] == "pre-exchange"
        assert entry["atom_indices"] == [[4, 6, 8, 14], [6, 8, 14, 16]]
        assert entry["header"] == ",".join(entry["columns"])
        assert len(entry["definition_sha256"]) == 64


def test_the_manifest_records_the_cv_cost_separately(completed):
    """Both scopes, and a scalar count that is rows TIMES the definition width.

    This test used to assert `cv_evaluations == cv_rows`, under a comment noting that the
    definition holds two named torsions -- encoding the very misnomer it looked like it was
    guarding. The counter incremented once per reporter call whatever the definition held, so a
    two-torsion run reported half the scalar work it had done. `cv_observations` counts the
    calls; `cv_evaluations` counts the scalar values, and with two torsions the two cannot be
    equal.
    """
    cost = _manifest(completed)["collective_variables"].get("cost")
    assert cost, "no CV cost was recorded"
    rows = STATES * (TOTAL // CV_EVERY + 1)
    assert cost["cv_rows"] == rows
    assert cost["cumulative"]["cv_observations"] == rows
    assert cost["cumulative"]["cv_evaluations"] == rows * 2, (
        "two named torsions per observation, per state")
    assert cost["segment"] == cost["cumulative"], "an uninterrupted run's scopes are equal"
    assert cost["cumulative"]["wall_seconds"] >= 0.0
    assert cost["aggregation"] == "sum over thermodynamic states"
    assert len(cost["per_state"]) == STATES, "the per-state records make the total auditable"
    # Still deliberately apart from any energy counter: a position-only torsion is not an
    # energy evaluation, and folding it in would corrupt the number that says how expensive
    # the Hamiltonian is.
    assert not any("energy" in key for key in cost), sorted(cost)


def test_a_completed_run_validates_against_its_own_manifest(completed):
    """The control. Without it every refusal below could come from an unrelated fault."""
    from md_tools.remd import validate as replica_validate

    result = replica_validate.validate_replica_output(
        analysis=str(completed / "REST2.nc"), manifest=str(completed / "restart.json"))
    assert result.ok, replica_validate.format_report(result)


DAMAGE = ["delete_csv", "delete_sidecar", "truncate", "mutate_value", "mutate_step",
          "swap_two_states", "replace_sidecar"]


@pytest.mark.parametrize("damage", DAMAGE)
def test_a_damaged_series_stops_the_run_validating(completed, tmp_path, damage):
    """Every one of these leaves a file that still parses and still looks like a CV series."""
    from md_tools.remd import validate as replica_validate

    staged = tmp_path / damage
    shutil.copytree(completed, staged)
    csv0, csv1 = staged / "remd0.cv.csv", staged / "remd1.cv.csv"
    sidecar0 = staged / "remd0.cv.json"

    if damage == "delete_csv":
        csv0.unlink()
    elif damage == "delete_sidecar":
        sidecar0.unlink()
    elif damage == "truncate":
        lines = csv0.read_text(encoding="utf-8").splitlines()
        csv0.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif damage == "mutate_value":
        lines = csv0.read_text(encoding="utf-8").splitlines()
        parts = lines[-1].split(",")
        parts[-1] = f"{float(parts[-1]) + 7.5:.6f}"
        csv0.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "mutate_step":
        lines = csv0.read_text(encoding="utf-8").splitlines()
        parts = lines[-1].split(",")
        parts[0] = str(int(parts[0]) + 1)
        csv0.write_text("\n".join(lines[:-1] + [",".join(parts)]) + "\n", encoding="utf-8")
    elif damage == "swap_two_states":
        # The subtlest case: both files are intact, valid CV series -- just each other's.
        a, b = csv0.read_text(encoding="utf-8"), csv1.read_text(encoding="utf-8")
        csv0.write_text(b, encoding="utf-8")
        csv1.write_text(a, encoding="utf-8")
    else:
        sidecar0.write_text('{"schema_version": 1}', encoding="utf-8")

    result = replica_validate.validate_replica_output(
        analysis=str(staged / "REST2.nc"), manifest=str(staged / "restart.json"))
    assert not result.ok, (
        f"a {damage} left the run validating; the manifest claims files it does not check")
    report = replica_validate.format_report(result).lower()
    assert "collective variable" in report or "collective-variable" in report, \
        report[-2000:]


def test_an_extension_refuses_a_damaged_cv_parent(project, completed, tmp_path):
    """An extension must not build on a parent whose CV series no longer matches its manifest."""
    staged = tmp_path / "parent"
    shutil.copytree(completed, staged)
    lines = (staged / "remd0.cv.csv").read_text(encoding="utf-8").splitlines()
    (staged / "remd0.cv.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    extended = tmp_path / "extended"
    done = _run(project, extended, "--extend", "2", "--extend-from", str(staged), expect=1)
    # The refusal is written into the rank's own report, which the executor captures as the run's
    # record; only a pointer to it reaches the launcher's stdout.
    reports = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                        for path in sorted(extended.rglob("REST2.out*")))
    message = (done.stdout + done.stderr + reports).lower()
    assert "collective-variable" in message or "collective variable" in message, \
        message[-3000:]


def test_a_cv_disabled_run_records_that_it_has_no_series(tmp_path_factory, tmp_path):
    """`None` is a statement. Omission would be indistinguishable from an older manifest."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = _project(tmp_path_factory.mktemp("ladder-no-cv"), cv=False)
    destination = tmp_path / "run"
    _run(root, destination)
    record = _manifest(destination)
    assert "collective_variables" in record, "the field is absent, not stated"
    assert record["collective_variables"] is None
    assert not sorted(destination.glob("*.cv.csv"))
