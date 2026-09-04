"""The CV schedule and the series it writes: exactness, no duplicates, and honest cost.

PLATFORM_POLICY_EXEMPTION: arithmetic over supplied coordinate arrays. Nothing is integrated and
no Context is created -- which is itself the property under test in `test_cv_evaluation_touches_no_openmm_object`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from md_tools.cv import (CVReportError, CVScheduleError, CVSeries, observation_steps,
                         check_divides, parse_cv_definition)

DEFINITION = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [0, 1, 2, 3]
"""


@pytest.fixture
def definition():
    return parse_cv_definition(DEFINITION, particles=4)


def _positions(angle_degrees: float) -> np.ndarray:
    import math

    theta = math.radians(angle_degrees)
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                     [math.cos(theta), math.sin(theta), 1.0]])


# --- the schedule --------------------------------------------------------------------------

def test_step_zero_and_the_final_step_are_each_present_exactly_once():
    steps = observation_steps(100, 25)
    assert steps == (0, 25, 50, 75, 100)
    assert steps.count(0) == 1 and steps.count(100) == 1


def test_an_interval_that_does_not_divide_the_span_is_refused():
    """A final partial gap breaks the uniform spacing every downstream analysis assumes."""
    with pytest.raises(CVScheduleError, match="does not divide"):
        observation_steps(100, 30)


def test_a_minimisation_produces_no_series():
    """Minimiser iterations are not dynamics: no timestep, so `time_ps` would be a fiction."""
    assert observation_steps(0, 25) == ()


def test_a_cadence_finer_than_the_trajectory_is_allowed():
    """THE independence claim: the CV series exists to be sampled more finely than frames are."""
    trajectory_every = 50
    cv_steps = observation_steps(100, 10)
    assert len(cv_steps) > 100 // trajectory_every + 1
    # And every trajectory step is still an observation step, so frames can always be aligned.
    assert set(range(0, 101, trajectory_every)) <= set(cv_steps)


def test_check_divides_names_the_span_it_is_talking_about():
    """REST2 states it against the exchange interval, AIS against the switching length."""
    check_divides(10, 100, where="REST2", what="exchange_interval_steps (100)")
    with pytest.raises(CVScheduleError, match="exchange_interval_steps"):
        check_divides(30, 100, where="REST2", what="exchange_interval_steps (100)")


# --- the series ----------------------------------------------------------------------------

def test_a_written_series_holds_the_values_an_independent_calculation_gives(definition, tmp_path):
    """Every value compared, not merely the column's existence."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step", "time_ps",
                                                   "trajectory_frame_index")).open() as series:
        for index, angle in enumerate((0.0, 45.0, -90.0, 179.0)):
            values = series.evaluate(_positions(angle))
            series.write((index * 10, index * 0.02, index if index % 2 == 0 else None), values)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,time_ps,trajectory_frame_index,phi"
    assert len(lines) == 5, "one header and four observations"
    for line, angle in zip(lines[1:], (0.0, 45.0, -90.0, 179.0)):
        assert float(line.split(",")[-1]) == pytest.approx(angle, abs=1e-6)


def test_an_absent_trajectory_frame_is_an_empty_field_not_a_number(definition, tmp_path):
    """0 and -1 are both valid frame indices; only empty can mean 'there is no frame here'."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition,
                  extra_columns=("step", "time_ps", "trajectory_frame_index")).open() as series:
        series.write((0, 0.0, None), series.evaluate(_positions(30.0)))
    assert path.read_text(encoding="utf-8").splitlines()[1].split(",")[2] == ""


def test_the_csv_carries_no_comment_lines(definition, tmp_path):
    """A `#` line is the specific thing that breaks the naive readers a CV series is read by."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        series.write((0,), series.evaluate(_positions(10.0)))
    assert not any(line.startswith("#")
                   for line in path.read_text(encoding="utf-8").splitlines())


def test_the_sidecar_carries_everything_needed_to_read_the_csv_alone(definition, tmp_path):
    """Provenance a comment line could not hold: units, wrapping, convention, resolved indices."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        pass
    body = json.loads(series.sidecar.read_text(encoding="utf-8"))
    assert body["units"] == "degrees"
    assert body["wrapping"] == "[-180, 180)"
    assert "minimum image" in body["periodic_convention"]
    assert body["definition_sha256"] == definition.digest
    assert body["collective_variables"][0]["atom_indices"] == [0, 1, 2, 3]
    assert body["column_order"] == ["step", "phi"]


def test_a_resume_truncates_to_the_committed_count_and_appends(definition, tmp_path):
    """No duplicate and no gap: rows after the committed count belong to repeated dynamics."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        for step in range(5):
            series.write((step,), series.evaluate(_positions(step * 10.0)))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 6

    # The checkpoint committed only three of them; the last two are about to be recomputed.
    with CVSeries(path, definition, extra_columns=("step",)).open(append_from=3) as series:
        for step in range(3, 6):
            series.write((step,), series.evaluate(_positions(step * 10.0)))

    lines = path.read_text(encoding="utf-8").splitlines()
    steps = [int(line.split(",")[0]) for line in lines[1:]]
    assert steps == [0, 1, 2, 3, 4, 5], "a resume duplicated or dropped an observation"


def test_a_series_shorter_than_the_checkpoint_claims_is_refused(definition, tmp_path):
    """Truncated or deleted rows must not be silently invented on continuation."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        series.write((0,), series.evaluate(_positions(10.0)))

    with pytest.raises(CVReportError, match="shorter than the run"):
        CVSeries(path, definition, extra_columns=("step",)).open(append_from=4)


def test_the_evaluation_cost_is_recorded_separately_from_energy_evaluations(definition, tmp_path):
    """A position-only torsion is not an energy evaluation, and must never be counted as one."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        for step in range(7):
            series.write((step,), series.evaluate(_positions(step * 5.0)))
    cost = series.cost()
    assert cost["cumulative"]["cv_observations"] == 7
    assert cost["cv_rows"] == 7
    assert cost["cumulative"]["wall_seconds"] >= 0.0
    assert not any("energy" in key for key in cost), sorted(cost)


def test_observations_and_scalar_evaluations_are_different_numbers(tmp_path):
    """THE misnomer this replaces.

    The counter incremented once per reporter call, so for a definition with two torsions it
    reported one where two scalar values had been computed: the name said "evaluations" and the
    number said "observations". They coincide only for the single-CV case -- which is the case
    the original test used, so the test passed and the misnomer became a contract.

    Two torsions here, deliberately, so a call-count implementation cannot pass.
    """
    two = parse_cv_definition(
        "schema_version: 1\ncollective_variables:\n"
        "  - {name: phi, type: torsion, atom_indices: [0, 1, 2, 3]}\n"
        "  - {name: psi, type: torsion, atom_indices: [1, 2, 3, 0]}\n", particles=4)
    path = tmp_path / "two.cv.csv"
    with CVSeries(path, two, extra_columns=("step",)).open() as series:
        for step in range(5):
            series.write((step,), series.evaluate(_positions(step * 7.0)))
    cost = series.cost()
    assert cost["cumulative"]["cv_observations"] == 5
    assert cost["cumulative"]["cv_evaluations"] == 10, (
        "five observations of a two-torsion definition are ten scalar evaluations")
    assert cost["cv_rows"] == 5


def test_a_fresh_run_reports_equal_segment_and_cumulative(definition, tmp_path):
    path = tmp_path / "fresh.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        for step in range(3):
            series.write((step,), series.evaluate(_positions(step * 10.0)))
    cost = series.cost()
    assert cost["segment"] == cost["cumulative"]
    assert cost["schema_version"] == 2


def test_a_restored_prefix_carries_cost_into_the_cumulative_scope(definition, tmp_path):
    """Rows and counters travel as ONE object, so a restore cannot drop half of it."""
    from md_tools.cv.cost import CVCost, CommittedPrefix

    path = tmp_path / "resumed.cv.csv"
    with CVSeries(path, definition, extra_columns=("step",)).open() as series:
        for step in range(4):
            series.write((step,), series.evaluate(_positions(step * 3.0)))
    earlier = series.cumulative_cost()
    assert earlier.observations == 4

    reopened = CVSeries(path, definition, extra_columns=("step",))
    reopened.open(committed=CommittedPrefix(rows=4, cumulative=earlier))
    reopened.write((4,), reopened.evaluate(_positions(12.0)))
    cost = reopened.cost()
    assert cost["segment"]["cv_observations"] == 1, "the segment covers only this invocation"
    assert cost["cumulative"]["cv_observations"] == 5, "the cumulative carries the earlier work"
    assert cost["cumulative"]["cv_evaluations"] == 5 * len(definition.variables)
    reopened.close()


def test_a_row_of_the_wrong_width_is_refused(definition, tmp_path):
    """A silently ragged CSV is read differently by every reader."""
    path = tmp_path / "cMD.cv.csv"
    with CVSeries(path, definition, extra_columns=("step", "time_ps")).open() as series:
        with pytest.raises(CVReportError, match="declared column"):
            series.write((0,), series.evaluate(_positions(10.0)))


def test_cv_evaluation_touches_no_openmm_object():
    """The whole design claim, asserted against the CODE rather than the prose.

    If these modules ever reach for a Force, a Context or an energy, a CV-enabled run stops being
    the same experiment as a CV-disabled one and every comparison between them becomes invalid.

    Checked through the AST with docstrings stripped: the modules DISCUSS `CustomTorsionForce` at
    length -- explaining why it is the wrong tool -- so a raw text search finds the explanation
    and reports the very thing the explanation says is not there.
    """
    import ast
    from pathlib import Path

    for name in ("reporter.py", "torsion.py", "definition.py", "schedule.py"):
        source = (Path(__file__).resolve().parents[1] / "src" / "md_tools" / "cv"
                  / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Drop docstrings: they are the first statement of a module, class or function.
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:]

        code = ast.unparse(tree)
        for forbidden in ("openmm", "CustomTorsionForce", "addForce", "getState",
                          "getPotentialEnergy", "Context"):
            assert forbidden not in code, f"{name} reaches for {forbidden}"
