"""The AIS three-group decomposition survives interruption exactly, and stays self-checking.

WHAT MUST NOT DRIFT

    `U(tau, x) = U_non_scaled + sqrt(lambda) * U_sqrt_scaled + lambda * U_lin_scaled`, with
    `lambda = (1 - tau)^2`. The component works are accumulated across a whole path, so a resume
    that restored them wrongly -- or restored the evaluation counters wrongly -- produces a work
    value that is plausible, monotonic and wrong, and no single row reveals it.

    Two coordinates are deliberately distinct and must stay so: the work components are evaluated
    at the FROZEN PRE-SWITCH coordinate, where work is defined; the Hummer-Szabo observation
    potentials are evaluated at the coordinate actually saved and named by
    `coordinate_frame_index`. Confusing them is invisible in the output -- both are energies of
    the right magnitude.

    The identity is required to hold and is never used to DERIVE the total: the directly measured
    work and the sum of the components are computed independently, precisely so that their
    agreement tests something.

PLATFORM_POLICY_EXEMPTION: paths run under `--cpu`. What is under test is accumulator arithmetic
and row alignment, which is identical on every platform; the energies themselves are covered by
the CUDA lanes.
"""
from __future__ import annotations

import csv
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
"""

SWITCHING = 40
UPDATE_EVERY = 5
OBSERVE_EVERY = 10


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-decomp")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": SWITCHING,
                "observation_interval_steps": OBSERVE_EVERY,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": OBSERVE_EVERY, "system_printout": OBSERVE_EVERY,
                      "checkpoint_printout": OBSERVE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"),
                                 "interval_steps": UPDATE_EVERY * 2},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./AIS", "--config", str(root / "AIS.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _run(project: Path, destination: Path, *extra, environment=None, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    base.update(environment or {})
    done = subprocess.run(
        [sys.executable, str(project / "AIS" / "AIS.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-source-traj", str(project / "source.dcd"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "AIS", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("ais-decomp-run") / "run"
    _run(project, destination)
    return destination


GROUPS = ("non_scaled", "sqrt_scaled", "lin_scaled")


def test_the_three_group_names_are_intact(completed):
    """`lin_scaled` is never spelled `scaled`: three components carry a scaling."""
    rows = _rows(completed / "path_0000" / "observations.csv")
    for group in GROUPS:
        assert f"delta_work_{group}_kj_mol" in rows[0], sorted(rows[0])
        assert f"total_work_{group}_kj_mol" in rows[0], sorted(rows[0])
        assert f"potential_{group}_kj_mol" in rows[0], sorted(rows[0])
    assert not any(key in rows[0] for key in ("delta_work_scaled_kj_mol",
                                              "total_work_scaled_kj_mol",
                                              "potential_scaled_kj_mol")), (
        "a bare `scaled` column appeared: three components carry a scaling, so the name does "
        "not say which, and the ambiguity lands in the files a reweighting is built from")


#: The measured total and the three components that must sum to it. The two names differ
#: because the schema distinguishes the per-update increment from the running accumulation --
#: `delta_work_*` are the increments, `total_work_*` the cumulative ones.
IDENTITIES = {
    "incremental": ("incremental_work_kj_mol", "delta_work_{group}_kj_mol"),
    "cumulative": ("cumulative_work_kj_mol", "total_work_{group}_kj_mol"),
}


def _identity_holds(row, which, tolerance=1e-3):
    """The measured total equals the sum of its three groups, within tolerance.

    The total is MEASURED, never derived from the components -- that independence is what makes
    their agreement test anything at all.
    """
    total_column, component = IDENTITIES[which]
    total = float(row[total_column])
    parts = sum(float(row[component.format(group=group)]) for group in GROUPS)
    return abs(total - parts) <= max(tolerance, 1e-6 * abs(total))


def test_the_incremental_identity_holds_on_every_row(completed):
    rows = _rows(completed / "path_0000" / "observations.csv")
    for row in rows:
        assert _identity_holds(row, "incremental"), row


def test_the_cumulative_identity_holds_on_every_row(completed):
    rows = _rows(completed / "path_0000" / "observations.csv")
    for row in rows:
        assert _identity_holds(row, "cumulative"), row


def test_the_hs_table_is_the_frame_aligned_subset_with_group_potentials(completed):
    """Every HS row's potentials and its work must describe ONE saved configuration."""
    hs = _rows(completed / "AIS_hs.csv")
    assert hs, "no HS rows were written"
    for row in hs:
        assert row["coordinate_frame_index"] != "", (
            "an HS row without a saved coordinate puts empty potential cells in front of a "
            "reweighting with no way to notice")
        for group in GROUPS:
            assert row[f"potential_{group}_kj_mol"] != ""
        direct = float(row["potential_direct_kj_mol"])
        reconstructed = float(row["potential_reconstructed_kj_mol"])
        assert abs(direct - reconstructed) <= max(1e-2, 1e-6 * abs(direct)), row


@pytest.mark.parametrize("boundary", ["before-frame", "after-frame",
                                      "before-work-row", "after-work-row"])
def test_the_accumulators_survive_an_interruption_exactly(project, tmp_path, boundary):
    """THE resume test. A restored accumulator that drifts gives a plausible wrong work value."""
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    reference = tmp_path / f"reference-{boundary}"
    _run(project, reference)
    expected = _rows(reference / "path_0000" / "observations.csv")

    destination = tmp_path / f"resumed-{boundary}"
    crashed = _run(project, destination, expect=1,
                   environment={FAULT_ENVIRONMENT: boundary, FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0

    _run(project, destination, "--resume")
    rows = _rows(destination / "path_0000" / "observations.csv")

    assert len(rows) == len(expected), (
        f"{boundary}: resumed path wrote {len(rows)} observations, reference {len(expected)}")
    for resumed, clean in zip(rows, expected):
        assert int(resumed["protocol_step"]) == int(clean["protocol_step"])
        assert _identity_holds(resumed, "cumulative"), resumed
        for group in GROUPS:
            a = float(resumed[f"total_work_{group}_kj_mol"])
            b = float(clean[f"total_work_{group}_kj_mol"])
            assert abs(a - b) <= max(1e-3, 1e-6 * abs(b)), (
                f"{boundary}: {group} cumulative work drifted across the resume: {a} vs {b}")


def test_the_completion_manifest_records_the_decomposition_schema(completed):
    record = json.loads(
        (completed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    schema = record.get("decomposition_schema") or {}
    assert schema.get("name") and schema.get("version"), record.get("decomposition_schema")
    counters = record.get("evaluation_counters") or {}
    assert counters, "the evaluation counters were not recorded"
    # CV cost is accounted separately from energy evaluations.
    assert not any("cv" in key and "energy" in key for key in counters), sorted(counters)


def test_cv_rows_and_work_rows_are_counted_separately(completed):
    """A position-only torsion is not an energy evaluation and must not inflate that total."""
    record = json.loads(
        (completed / "path_0000" / "completed.json").read_text(encoding="utf-8"))
    assert "cv_rows" in record
    counters = record["evaluation_counters"]
    energy_like = [key for key in counters if "energy" in key]
    assert energy_like, sorted(counters)
    assert record["cv_rows"] > 0
