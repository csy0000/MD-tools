"""The `collective_variables` configuration section, and the guess it refuses to make.

Both keys default to off. Supplying only one is an error rather than a default, because both
readings of a half-statement are wrong: a file with no interval names torsions nobody asked to be
measured, and an interval with no file asks for observations of nothing. Silently choosing either
produces a run that disagrees with its own configuration -- and since CV output failures are
simulation failures, quietly reporting nothing is the worst of the available answers.
"""
from __future__ import annotations

import pytest

from md_tools.build.md import MD_SCHEMA
from md_tools.build.strict import ConfigError


def _resolve(**block):
    return MD_SCHEMA.resolve({"protocol": "cMD", "collective_variables": block})


def test_reporting_is_off_by_default():
    resolved = MD_SCHEMA.resolve({"protocol": "cMD"})["collective_variables"]
    assert resolved == {"file": None, "generate": None, "interval_steps": 0}


def test_both_keys_given_together_is_accepted():
    resolved = _resolve(file="cv.yaml", interval_steps=100)["collective_variables"]
    assert resolved == {"file": "cv.yaml", "generate": None, "interval_steps": 100}


def test_both_keys_explicitly_off_is_accepted():
    """Writing the defaults out is a deliberate statement, not a half one."""
    assert _resolve(file=None, interval_steps=0)["collective_variables"]["interval_steps"] == 0


def test_a_file_with_no_interval_is_refused():
    with pytest.raises(ConfigError, match="interval_steps is 0"):
        _resolve(file="cv.yaml")


def test_an_interval_with_no_file_is_refused():
    with pytest.raises(ConfigError, match="nothing to measure"):
        _resolve(interval_steps=100)


def test_an_unknown_field_in_the_section_is_refused():
    with pytest.raises(ConfigError, match="collective_variables"):
        _resolve(file="cv.yaml", interval_steps=100, wrap="degrees")


def test_a_negative_interval_is_refused():
    with pytest.raises(ConfigError, match="below the minimum"):
        _resolve(file="cv.yaml", interval_steps=-5)


def test_a_boolean_interval_is_refused_rather_than_read_as_one():
    """`true` is an int in Python; an accidental boolean must not become a 1-step cadence."""
    with pytest.raises(ConfigError, match="boolean"):
        _resolve(file="cv.yaml", interval_steps=True)


def test_a_duplicate_key_in_the_section_is_refused():
    """The whole document goes through the strict loader, this section included."""
    from md_tools.build.strict import load_yaml_strictly

    text = ("protocol: cMD\ncollective_variables:\n  file: a.yaml\n"
            "  interval_steps: 10\n  interval_steps: 20\n")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_yaml_strictly(text, source="test.config")
