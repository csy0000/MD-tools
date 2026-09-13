"""The four caller-named artefacts default by segment, like everything else in the directory."""
from __future__ import annotations

import pytest

from md_tools.md._stages import (energy_components_name, info_csv_name, stage_artifact_name)


@pytest.mark.parametrize("segment", [1, 2, 3, 17])
def test_it_follows_the_same_rule_as_the_state_table(segment):
    """One rule for the directory. A `mdout_prod2.csv` beside a `cMD.log` that segment 2
    overwrote is worse than either convention applied consistently."""
    assert stage_artifact_name("cMD", "csv", segment).replace("cMD", "mdout") == info_csv_name(
        "cMD", segment)
    assert stage_artifact_name("cMD", "csv", segment).replace(
        "cMD", "energy_components") == energy_components_name("cMD", segment)


def test_segment_one_keeps_the_plain_name():
    assert [stage_artifact_name("cMD", e) for e in ("out", "log", "xml", "chk")] == [
        "cMD.out", "cMD.log", "cMD.xml", "cMD.chk"]


def test_later_segments_cannot_collide_with_earlier_ones():
    seen = set()
    for segment in range(1, 12):
        names = {stage_artifact_name("cMD", e, segment) for e in ("out", "log", "xml", "chk")}
        assert not (names & seen), f"segment {segment} reuses a name from an earlier segment"
        seen |= names


def test_a_non_production_stage_never_takes_a_segment():
    """Equilibration runs once per chain; `eq_nvt_free_prod2.log` would describe nothing."""
    for segment in (1, 2, 9):
        assert stage_artifact_name("eq_nvt_free", "log", segment) == "eq_nvt_free.log"
        assert stage_artifact_name("min", "xml", segment) == "min.xml"


def test_the_extension_may_be_written_either_way():
    assert stage_artifact_name("cMD", ".out", 4) == stage_artifact_name("cMD", "out", 4)
