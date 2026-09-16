"""The committed CV prefix is protected by a digest, not merely by a row count.

WHY A COUNT IS NOT ENOUGH

    A checkpoint that records "12 rows were committed" detects a file SHORTER than 12. It says
    nothing about whether those 12 rows are still the rows that were committed. Edit a value in
    place, renumber a step, change a walker index -- the count still agrees, the continuation
    appends onto them, and the finished series is part measurement and part edit with nothing in
    the file marking the boundary.

    The generation therefore records a digest of exactly the header plus the committed rows.
    Not of the whole file: after a crash the file is legitimately LONGER than the checkpoint,
    because rows are flushed as they are written and the commit happens afterwards. Hashing the
    uncommitted tail would make every ordinary crash look like corruption.

Most of this file is in-process against the prefix module, so the whole matrix runs in the fast
lane. The end-to-end case drives a real interrupted cMD stage.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.cv import prefix as cv_prefix

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

COLUMNS = ["step", "time_ps", "state_index", "tau", "phi"]


def _series(path: Path, rows=4, *, state=1, tau=0.25, interval=5, tail=0):
    lines = [",".join(COLUMNS)]
    for n in range(rows + tail):
        lines.append(",".join([str(n * interval), f"{n * 0.01:.6f}", str(state),
                               f"{tau}", f"{10.0 + n:.6f}"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _entry(path, rows, *, cost="valid", sidecar=None):
    """A committed-prefix entry, carrying a valid cost unless a case is testing its absence.

    A CV-enabled prefix is now REQUIRED to record what it cost, so a fixture that omitted one
    made every case below refuse for that reason instead of the reason it is named for. The
    cost here is the honest one for this fixture: `rows` observations of the single `phi`
    column, so each case isolates the single thing it mutates.
    """
    from md_tools.cv.cost import CVCost, cost_record

    if cost == "valid":
        scope = CVCost(observations=rows, evaluations=rows, wall_seconds=0.001)
        cost = cost_record(scope, scope, rows=rows)
    return cv_prefix.record(path, rows=rows, sidecar=sidecar, cost=cost or None)


def _validate(path, entry, **kwargs):
    kwargs.setdefault("expect_columns", COLUMNS)
    kwargs.setdefault("value_columns", ["phi"])
    return cv_prefix.validate(path, entry, **kwargs)


def test_an_untouched_prefix_validates(tmp_path):
    """The control. Without it every refusal below could come from an unrelated fault."""
    path = _series(tmp_path / "a.cv.csv", rows=4)
    assert _validate(path, _entry(path, 4)) == 4


def test_an_uncommitted_tail_is_allowed_and_then_truncated(tmp_path):
    """A crash leaves rows past the commit. That is ordinary, not corruption."""
    path = _series(tmp_path / "a.cv.csv", rows=4, tail=3)
    entry = _entry(path, 4)
    assert _validate(path, entry) == 4
    assert cv_prefix.truncate(path, 4) == 4
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_a_mutated_committed_value_is_refused(tmp_path):
    """THE case a row count cannot see."""
    path = _series(tmp_path / "a.cv.csv", rows=4)
    entry = _entry(path, 4)
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[2].split(",")
    parts[-1] = f"{float(parts[-1]) + 3.0:.6f}"
    lines[2] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(cv_prefix.CVPrefixError, match="not the rows that were committed"):
        _validate(path, entry)


def test_a_renumbered_committed_step_is_refused(tmp_path):
    path = _series(tmp_path / "a.cv.csv", rows=4)
    entry = _entry(path, 4)
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[3].split(",")
    parts[0] = str(int(parts[0]) + 1)
    lines[3] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(cv_prefix.CVPrefixError, match="not the rows that were committed"):
        _validate(path, entry)


def test_a_file_shorter_than_committed_is_refused(tmp_path):
    """Fewer rows than committed is data loss, not a resumable state."""
    path = _series(tmp_path / "a.cv.csv", rows=4)
    entry = _entry(path, 4)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(cv_prefix.CVPrefixError, match="have been lost"):
        _validate(path, entry)


def test_a_missing_prefix_record_refuses_with_a_compatibility_message(tmp_path):
    """A legacy checkpoint cannot say which rows are durable, and must not guess."""
    path = _series(tmp_path / "a.cv.csv", rows=4)
    with pytest.raises(cv_prefix.CVPrefixError, match="does not record"):
        _validate(path, None)
    with pytest.raises(cv_prefix.CVPrefixError, match="does not record"):
        _validate(path, {"rows": 4})


def test_a_wrong_identifier_is_refused(tmp_path):
    """A series belonging to a different state is otherwise identically shaped."""
    path = _series(tmp_path / "a.cv.csv", rows=4, state=1)
    entry = _entry(path, 4)
    with pytest.raises(cv_prefix.CVPrefixError, match="different state or path"):
        _validate(path, entry, identifiers={"state_index": 2})
    assert _validate(path, entry, identifiers={"state_index": 1, "tau": 0.25}) == 4


def test_a_non_finite_value_is_refused(tmp_path):
    path = tmp_path / "a.cv.csv"
    path.write_text(",".join(COLUMNS) + "\n0,0.0,1,0.25,nan\n", encoding="utf-8")
    entry = _entry(path, 1)
    with pytest.raises(cv_prefix.CVPrefixError, match="non-finite"):
        _validate(path, entry)


def test_a_changed_column_set_is_refused(tmp_path):
    path = _series(tmp_path / "a.cv.csv", rows=2)
    entry = _entry(path, 2)
    with pytest.raises(cv_prefix.CVPrefixError, match="columns are"):
        _validate(path, entry, expect_columns=COLUMNS + ["psi"])


def test_a_gap_in_the_committed_grid_is_refused(tmp_path):
    """A missing row leaves a count that still matches if the tail made up the difference."""
    path = tmp_path / "a.cv.csv"
    path.write_text(",".join(COLUMNS) + "\n0,0.0,1,0.25,1.0\n10,0.02,1,0.25,2.0\n",
                    encoding="utf-8")
    entry = _entry(path, 2)
    with pytest.raises(cv_prefix.CVPrefixError, match="not on the declared"):
        _validate(path, entry, interval=5)


def test_a_changed_sidecar_is_refused(tmp_path):
    path = _series(tmp_path / "a.cv.csv", rows=2)
    sidecar = tmp_path / "a.cv.json"
    sidecar.write_text('{"units": "degrees"}', encoding="utf-8")
    entry = _entry(path, 2, sidecar=sidecar)
    sidecar.write_text('{"units": "radians"}', encoding="utf-8")
    with pytest.raises(cv_prefix.CVPrefixError, match="different interpretation"):
        _validate(path, entry, sidecar=sidecar)


# --- end to end: a real interrupted stage refuses a mutated committed prefix -------------------

@pytest.mark.slow
def test_a_real_stage_refuses_a_mutated_committed_prefix(tmp_path):
    """The whole point, through the runtime: the resume must refuse before it truncates."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    root = tmp_path / "project"
    root.mkdir()
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}\n", encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 40},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD", "--config", str(root / "cMD.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)

    destination = tmp_path / "run"

    def _launch(environment=None):
        return subprocess.run(
            [sys.executable, str(root / "cMD" / "md.py"),
             "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
             "-odir", str(destination), "--cpu"],
            cwd=root / "cMD", capture_output=True, text=True, timeout=1800,
            env={**base, **(environment or {})})

    crashed = _launch({FAULT_ENVIRONMENT: "after-pointer-replace",
                       FAULT_AFTER_ENVIRONMENT: "5"})
    assert crashed.returncode != 0

    series = sorted(destination.rglob("*.cv.csv"))
    assert series, "the interrupted stage wrote no CV series"
    lines = series[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) > 2, lines
    parts = lines[1].split(",")
    parts[-1] = f"{float(parts[-1]) + 12.0:.6f}"
    series[0].write_text("\n".join([lines[0], ",".join(parts)] + lines[2:]) + "\n",
                         encoding="utf-8")
    before = series[0].read_text(encoding="utf-8")

    refused = _launch()
    assert refused.returncode != 0, refused.stdout[-3000:] + refused.stderr[-3000:]
    message = (refused.stdout + refused.stderr).lower()
    assert "committed" in message, message[-2000:]
    assert series[0].read_text(encoding="utf-8") == before, (
        "the refused continuation modified the series it was refusing")


# --- CV cost accounting, persisted and accumulated ---------------------------------------------

@pytest.mark.slow
def test_cv_cost_is_persisted_and_survives_two_interruptions(tmp_path):
    """Cumulative cost survives REPEATED resume without loss or double counting.

    Two consecutive interruptions, not one: a single resume can pass while the accumulation is
    "restore the prefix and add this segment", which is right, or "take the prefix" / "take this
    segment", which are both wrong in ways one resume can hide. The second resume separates them.

    TWO named torsions throughout, so `cv_observations` and `cv_evaluations` are different
    numbers and a call-count implementation cannot pass.
    """
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    from md_tools.build.record import read_record
    from md_tools.openmm.checkpoint import FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT

    root = tmp_path / "project"
    root.mkdir()
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}\n"
        "  - {name: psi, type: torsion, atom_indices: [6, 8, 14, 16]}\n", encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 60},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./cMD", "--config", str(root / "cMD.config")],
        cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    # OpenMM's CPU platform sums its force reductions in thread-completion order and is
    # only reproducible at a fixed pool size, so comparing a resumed series to an
    # uninterrupted one value-by-value needs the pool pinned. A property of the platform;
    # nothing pins a thread count in production.
    base["OPENMM_CPU_THREADS"] = "1"

    def _launch(destination, environment=None):
        return subprocess.run(
            [sys.executable, str(root / "cMD" / "md.py"),
             "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
             "-odir", str(destination), "--cpu"],
            cwd=root / "cMD", capture_output=True, text=True, timeout=1800,
            env={**base, **(environment or {})})

    n_cv, interval, steps = 2, 5, 60
    expected_rows = steps // interval + 1

    clean = tmp_path / "clean"
    assert _launch(clean).returncode == 0
    reference = read_record(clean / "cMD.log")["collective_variable_cost"]
    assert reference["cumulative"]["cv_observations"] == expected_rows
    assert reference["cumulative"]["cv_evaluations"] == expected_rows * n_cv
    assert reference["segment"] == reference["cumulative"], "a fresh run's scopes are equal"

    # TWO interruptions, at different committed generations.
    resumed = tmp_path / "resumed"
    first = _launch(resumed, {FAULT_ENVIRONMENT: "after-pointer-replace",
                              FAULT_AFTER_ENVIRONMENT: "5"})
    assert first.returncode != 0
    second = _launch(resumed, {FAULT_ENVIRONMENT: "after-pointer-replace",
                               FAULT_AFTER_ENVIRONMENT: "2"})
    assert second.returncode != 0
    assert _launch(resumed).returncode == 0

    after = read_record(resumed / "cMD.log")["collective_variable_cost"]
    assert after["cumulative"]["cv_observations"] == expected_rows, (
        f"after two resumes the cumulative observations are "
        f"{after['cumulative']['cv_observations']}, not {expected_rows}: earlier work was lost "
        f"or counted twice")
    assert after["cumulative"]["cv_evaluations"] == expected_rows * n_cv
    assert after["segment"]["cv_observations"] < after["cumulative"]["cv_observations"], (
        "the segment equals the cumulative, so earlier segments were not carried")
    assert after["segment"]["cv_observations"] > 0
    assert after["cumulative"]["wall_seconds"] >= after["segment"]["wall_seconds"] >= 0.0

    # The series itself matches the uninterrupted reference exactly -- EVERY COLUMN, not only
    # the step grid. A continuation that restores coordinates but not the integrator's
    # pseudo-random stream writes a correct, statistically exact and completely different
    # trajectory onto a step grid that still lines up perfectly, and a comparison of step
    # numbers alone waves it through. That is precisely the defect this file exists to catch.
    a = (clean / "cMD.cv.csv").read_text(encoding="utf-8").splitlines()
    b = (resumed / "cMD.cv.csv").read_text(encoding="utf-8").splitlines()
    assert a[0] == b[0], "headers differ"
    assert a[1:] == b[1:], (
        "the resumed series differs from the uninterrupted reference: the continuation did not "
        "resume the trajectory, it started a new one")
    assert len(b) - 1 == expected_rows

    # Re-entering a completed run performs no new CV evaluation.
    before_reentry = after["cumulative"]["cv_evaluations"]
    assert _launch(resumed).returncode == 0
    again = read_record(resumed / "cMD.log")["collective_variable_cost"]
    assert again["cumulative"]["cv_evaluations"] == before_reentry, (
        "re-entering a completed run evaluated collective variables again")


def test_the_final_committed_generation_carries_the_cv_prefix_and_cost(tmp_path):
    """The LAST generation, not just the periodic ones.

    A stage commits generations periodically through its checkpoint reporter and once more at
    the end. The periodic path supplied the CV prefix; the final one did not -- despite a
    comment beside it saying the final commit goes through the same transaction as every
    periodic one. So the generation a later reader actually consults, the committed one,
    vouched for a CV row count in `streams` while carrying no digest of those rows and no
    cumulative counters. A continuation from it would refuse for want of a committed prefix,
    and nothing protected the committed rows of a finished stage from being edited in place.
    """
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    from md_tools.openmm.checkpoint import read_committed

    root = tmp_path / "project"
    root.mkdir()
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    # INTO `build/`: the System belongs to the DATASET, and `build-md` validates the chain it
    # generates against it.
    (root / "build").mkdir(exist_ok=True)
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0
    (root / "cv.yaml").write_text(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [4, 6, 8, 14]}\n"
        "  - {name: psi, type: torsion, atom_indices: [6, 8, 14, 16]}\n", encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 60},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-md", "-odir", "./cMD", "--config", str(root / "cMD.config")],
        cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)

    destination = tmp_path / "final"
    done = subprocess.run(
        # THE PRODUCTION STAGE, which is the one that reports collective variables. This used to
        # run the retired `--all-in-one` md.py for the whole chain; with every equilibration
        # length set to 0 above, production is the only stage carrying a CV stream either way.
        [sys.executable, str(root / "cMD" / "cMD.py"),
         "-p", str(root / "build" / "built.pdb"), "-s", str(root / "build" / "built.xml"),
         "-odir", str(destination), "--cpu"],
        cwd=root / "cMD", capture_output=True, text=True, timeout=1800, env=base)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # The stage that REPORTED collective variables, not simply the last directory: a chain
    # commits generations for its minimisation and equilibration stages too, and those carry no
    # CV stream at all.
    states = [read_committed(path)["state"] for path in sorted(destination.rglob("*.checkpoints"))]
    reporting = [state for state in states
                 if "collective_variables" in (state.get("streams") or {})]
    assert reporting, "no committed generation vouches for a CV stream"
    state = reporting[-1]
    rows = int(state["streams"]["collective_variables"])
    assert rows == 13

    entry = state.get("cv_prefix")
    assert entry is not None, (
        "the committed generation vouches for CV rows and carries no prefix record for them")
    assert int(entry["rows"]) == rows
    assert entry["prefix_sha256"], "no digest protects the committed rows"
    cost = entry["cost"]
    assert cost["cumulative"]["cv_observations"] == rows
    assert cost["cumulative"]["cv_evaluations"] == rows * 2, "two torsions per observation"

    # And it is a prefix a continuation would actually accept, by the same validator.
    series = sorted(destination.rglob("*.cv.csv"))[0]
    assert cv_prefix.validate(
        series, entry, sidecar=series.with_suffix(".json")) == rows


def test_a_cv_enabled_prefix_with_no_cost_is_refused(tmp_path):
    """The rule the fixture above now satisfies, asserted directly rather than by accident.

    An absent cost on a CV-enabled prefix is corruption or unsupported legacy data: restoring it
    as zero would discard the history the prefix exists to preserve. Every other case in this
    file supplies a valid cost so that it can test its own mutation; this one supplies none.
    """
    path = _series(tmp_path / "a.cv.csv", rows=3)
    entry = _entry(path, 3, cost=None)
    with pytest.raises(cv_prefix.CVPrefixError, match="no usable cost record"):
        _validate(path, entry)
