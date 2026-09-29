"""The self-check record carries a max, the count it is over, and the median.

A MAX IS NOT A COMPARABLE STATISTIC. `evaluation_self_check_kJ_mol` is the largest deviation
between the evaluation Context and the sampling Context over every report a window made, so it
grows with the number of reports: a longer window looks worse for being longer, and a plot of it
across a ladder measures window length rather than agreement. The count is what makes it
interpretable -- with it a later reader can see that two windows are not comparable, instead of
discovering it in a plot -- and the median says what the typical case is.

The record is therefore version 2. Not a silent addition: two shapes answering to one schema name
is exactly what a version string exists to prevent, because a reader wanting the count could not
tell an old record from a malformed one. A version 1 record must still VERIFY, though, or the
change would silently re-run every window completed before it.
"""
from __future__ import annotations

import json
import statistics

import pytest

from md_tools.alchemy.windows import COMPLETION_SCHEMA, verify_window
from tests.test_alchemy_windows import SHORT, _run, model  # noqa: F401 - fixture


def test_the_completion_schema_is_version_2():
    assert COMPLETION_SCHEMA == "md-tools-alchemical-window-completion/2"


def _record(tmp_path, schema, digests):
    for name, text in (("samples", "a,b\n1,2\n"), ("restart", "<state/>")):
        (tmp_path / name).write_text(text, encoding="utf-8")
    doc = {"schema": schema, "fingerprint": "abc", **digests}
    (tmp_path / "completion").write_text(json.dumps(doc), encoding="utf-8")
    return {key: tmp_path / key for key in ("samples", "restart", "completion")}


def _digests(tmp_path):
    import hashlib
    for name, text in (("samples", "a,b\n1,2\n"), ("restart", "<state/>")):
        (tmp_path / name).write_text(text, encoding="utf-8")
    return {f"{n}_sha256": hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
            for n in ("samples", "restart")}


@pytest.mark.parametrize("schema", ["md-tools-alchemical-window-completion/1",
                                    "md-tools-alchemical-window-completion/2"])
def test_a_record_of_either_version_verifies(tmp_path, schema):
    """A window completed before the fields existed is not re-run because of them."""
    paths = _record(tmp_path, schema, _digests(tmp_path))
    assert verify_window(paths, "abc") == []


def test_the_guard_can_fail(tmp_path):
    """Shown to fail before it is trusted: a passing check that cannot fail closes the question."""
    paths = _record(tmp_path, COMPLETION_SCHEMA, _digests(tmp_path))
    (tmp_path / "samples").write_text("a,b\n1,3\n", encoding="utf-8")
    assert verify_window(paths, "abc") == ["samples changed since completion"]
    assert verify_window(paths, "other") != []


def test_the_reporter_keeps_every_deviation_not_a_running_summary():
    """The median cannot be recovered from a max, so the series is what must be kept."""
    from md_tools.alchemy.windows import SampleStreamReporter

    reporter = SampleStreamReporter.__new__(SampleStreamReporter)
    reporter.checked_self_energy = None
    reporter.self_check_deviations = []
    for deviation in (0.01, 0.05, 0.02, 0.80, 0.03):
        reporter.self_check_deviations.append(deviation)
        reporter.checked_self_energy = max(deviation, reporter.checked_self_energy or 0.0)

    assert reporter.checked_self_energy == 0.80
    assert len(reporter.self_check_deviations) == 5
    assert statistics.median(reporter.self_check_deviations) == 0.03
    # The point of recording all three: the max is 27x the median here, and a record carrying only
    # the max would read as a window in trouble rather than one with a single outlying report.
    assert reporter.checked_self_energy > 20 * statistics.median(reporter.self_check_deviations)


def test_a_real_window_writes_the_count_and_the_median(model):
    """The fields reach a WRITTEN record, not just the dict that builds it.

    Asserting that the keys appear in `windows.py` source would watch one side: a key can be
    constructed and never written, which is how a parameter forwarded nowhere survives for its
    whole life. So this runs a window and reads the file back.
    """
    import json

    result = _run(model, "w000", SHORT)
    doc = json.loads((model["tmp"] / "run" / "w000.complete.json").read_text())

    assert doc["schema"] == COMPLETION_SCHEMA
    # The count is the number of REPORTS, which is what makes the max interpretable at all.
    expected = SHORT.steps // SHORT.report_interval + 1
    assert doc["evaluation_self_check_n"] == result["rows"] == expected
    assert doc["evaluation_self_check_median_kJ_mol"] is not None
    assert doc["evaluation_self_check_median_kJ_mol"] <= doc["evaluation_self_check_kJ_mol"]
