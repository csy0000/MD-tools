"""The ladder's CV outputs in the OUTER machine record, and what a damaged one costs.

WHY THIS FILE

    `cv_state<N>.csv` and its sidecar were recorded in `restart.json` and, being ordinary files
    under the run root, hashed into the registry's `SHA256SUMS`. They were absent from the one
    place that connects a run to the runs around it: the `-log` machine record's `outputs`
    block, which `md_tools.registry.discovery.check_lineage` indexes by digest to match one
    stage's outputs against the next stage's inputs. A CV series was therefore invisible to
    every completion and lineage check that reads that record -- present on disk, vouched for by
    the run's own manifest, and unreferenced by the provenance a downstream reader consults.

WHY NOT A GLOB

    The inventory is built from the completion manifest, which is written only after the driver
    has validated every series. A `cv_state*.csv` glob over the directory would record a file
    left behind by an earlier run into the same place, or one belonging to a state this ladder
    does not have, as provenance for this one. The test for that is below: a foreign file
    dropped into the run root must not appear.

PLATFORM_POLICY_EXEMPTION: the ladders run under `--cpu`. What is under test is which files a
record names and what it says about them, which is identical on every platform.
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

STATES = 2
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


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cv-provenance")
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
        # `crd_printout_whole` must be set: a ladder's CV rows and its progress reconciliation
        # both count frames of the WHOLE state trajectory, and the key defaults to 0.
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY,
                      "crd_printout_whole": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
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
    return root


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base["OPENMM_CPU_THREADS"] = "1"
    done = subprocess.run(
        [sys.executable, str(project / "REST2-run1" / "REST2.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "REST2-run1", capture_output=True, text=True, timeout=1800,
        env={**base, **(environment or {})})
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _record(destination: Path) -> dict:
    """The machine block from the outer `-log`, as a downstream reader parses it."""
    from md_tools.build.record import read_record

    return read_record(destination / "REST2.log")


def _cv_roles(record: dict) -> dict:
    outputs = record.get("outputs") or {}
    return {role: facts for role, facts in outputs.items() if role.startswith("state_cv")}


def test_the_outer_record_names_every_cv_csv_and_sidecar(project, tmp_path):
    destination = tmp_path / "fresh"
    _run(project, destination)
    roles = _cv_roles(_record(destination))

    expected = set()
    for state in range(STATES):
        expected |= {f"state_cv_{state}", f"state_cv_definition_{state}"}
    assert set(roles) == expected, (
        "the machine record does not name every state's CV series and sidecar")

    manifest = json.loads(
        (destination / "restart.json").read_text(encoding="utf-8"))["collective_variables"]
    by_state = {entry["state_index"]: entry for entry in manifest["series"]}
    for state in range(STATES):
        entry = by_state[state]
        csv_facts = roles[f"state_cv_{state}"]
        side_facts = roles[f"state_cv_definition_{state}"]

        # The path is relative to the run root, so the record is readable after the directory
        # is moved or registered under another name.
        assert csv_facts["path"] == f"cv_state{state}.csv"
        assert side_facts["path"] == f"cv_state{state}.json"
        assert not Path(csv_facts["path"]).is_absolute()

        # Digest and size, and they are the manifest's -- the same numbers the run validated,
        # not a second opinion computed later.
        assert csv_facts["sha256"] == entry["csv_sha256"]
        assert csv_facts["bytes"] == entry["csv_bytes"]
        assert side_facts["sha256"] == entry["sidecar_sha256"]

        # What distinguishes this series from another identically shaped one.
        assert csv_facts["state_index"] == state
        assert csv_facts["tau"] == pytest.approx(entry["tau"])
        assert csv_facts["definition_sha256"] == entry["definition_sha256"]
        # Both roles cross-reference the same definition: two states are the same measurement.
        assert side_facts["definition_sha256"] == csv_facts["definition_sha256"]

    # CSV and sidecar are distinguished, not merged into one record.
    assert roles["state_cv_0"]["path"] != roles["state_cv_definition_0"]["path"]
    assert roles["state_cv_0"]["sha256"] != roles["state_cv_definition_0"]["sha256"]


def test_a_continued_run_carries_the_same_logical_inventory(project, tmp_path):
    """A resume must not produce a thinner record than an uninterrupted run."""
    reference = tmp_path / "reference"
    _run(project, reference)
    fresh = _cv_roles(_record(reference))

    resumed = tmp_path / "resumed"
    crashed = _run(project, resumed, expect=1,
                   environment={"MD_TOOLS_FAIL_PROPAGATION_ON_RANKS": "0",
                                "MD_TOOLS_FAIL_LADDER_AT": "after-checkpoint",
                                "MD_TOOLS_FAIL_PROPAGATION_AFTER": "1"})
    assert crashed.returncode != 0
    _run(project, resumed, "--resume")
    continued = _cv_roles(_record(resumed))

    # Non-empty FIRST. Two empty sets are equal, and a record that names no CV artefact at all
    # would otherwise satisfy every assertion below it.
    expected = set()
    for state in range(STATES):
        expected |= {f"state_cv_{state}", f"state_cv_definition_{state}"}
    assert set(fresh) == expected, "the fresh run's record already names no CV artefacts"
    assert set(continued) == set(fresh), (
        "the continued run's record names a different set of CV artefacts")
    for role, facts in fresh.items():
        for field in ("path", "bytes", "sha256", "state_index", "tau", "definition_sha256"):
            assert continued[role][field] == facts[field], (
                f"{role}.{field} differs between a fresh and a continued run")


def test_a_foreign_cv_file_in_the_run_root_is_not_recorded(project, tmp_path):
    """The reason this is built from the manifest and not from a glob.

    A file matching the naming convention, for a state this ladder does not have, dropped into
    the run root. A glob would record it as this run's provenance.
    """
    destination = tmp_path / "foreign"
    _run(project, destination)
    intruder = destination / f"remd{STATES + 5}.cv.csv"
    intruder.write_text("step,time_ps,phi\n0,0.0,-180.0\n", encoding="utf-8")

    # Re-derive the inventory the way the log writer does, now that the intruder exists.
    from md_tools.remd.generated import _cv_outputs

    roles = _cv_outputs(destination / "restart.json", destination)
    assert all(facts["path"] != intruder.name for facts in roles.values()), (
        "a foreign CV file was recorded as this run's provenance")
    assert len(roles) == 2 * STATES


def test_registration_refuses_a_cv_artefact_that_no_longer_matches(project, tmp_path):
    """A recorded digest is only worth what checking it is worth.

    Mutating one committed value in one series -- leaving the file readable, the row count
    right and the grid intact -- must be caught, both by the run's own verification and by the
    inventory a registration writes.
    """
    from md_tools.registry import inventory

    destination = tmp_path / "registered"
    _run(project, destination)
    entries = inventory.build(destination)
    listed = {entry["path"] for entry in entries}
    for state in range(STATES):
        assert f"cv_state{state}.csv" in listed
        assert f"cv_state{state}.json" in listed

    target = destination / "cv_state0.csv"
    lines = target.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[2].split(",")
    cells[header.index("phi")] = f"{float(cells[header.index('phi')]) + 5.0:.6f}"
    lines[2] = ",".join(cells)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The inventory taken before the edit no longer verifies.
    with pytest.raises(Exception) as refusal:
        inventory.verify(destination, entries)
    assert "cv_state0.csv" in str(refusal.value), refusal.value

    # And the run's own completion record refuses it by the same digest.
    from md_tools.remd.cv_states import verify_manifest_entries

    block = json.loads(
        (destination / "restart.json").read_text(encoding="utf-8"))["collective_variables"]
    problems = verify_manifest_entries(destination, block)
    assert any("cv_state0.csv" in problem for problem in problems), problems


def test_registration_refuses_a_missing_cv_artefact(project, tmp_path):
    from md_tools.registry import inventory

    destination = tmp_path / "missing"
    _run(project, destination)
    entries = inventory.build(destination)
    (destination / "cv_state1.json").unlink()

    with pytest.raises(Exception) as refusal:
        inventory.verify(destination, entries)
    assert "cv_state1.json" in str(refusal.value), refusal.value
