"""rREST2 is archived (0.5.4), and every surface that could still ask for it refuses BY NAME.

The generic refusals would be wrong answers: an unknown enum value suggests a near-miss spelling and
an unknown key suggests a similar key, when neither was mistyped -- the method was withdrawn. So the
protocol name, the `reservoir` section, and the `.in` spellings of both are refused with a message
that says it is archived and where the last working version is (the tag `rREST2-final`).

PLATFORM_POLICY_EXEMPTION: configuration and input parsing only. Nothing is built or propagated.
"""
from __future__ import annotations

import pytest

from md_tools.build.md import resolve_md_config
from md_tools.build.strict import ConfigError
from md_tools.run.inputs import parse_run_input


def _refused(callable_, *args, **kwargs) -> str:
    with pytest.raises(ConfigError) as refusal:
        callable_(*args, **kwargs)
    message = str(refusal.value)
    assert "archived" in message and "rREST2-final" in message, message
    return message


def test_a_configuration_asking_for_rrest2_is_refused_by_name(tmp_path):
    path = tmp_path / "rREST2.config"
    path.write_text("protocol: rREST2\n", encoding="utf-8")
    assert "protocol: rREST2" in _refused(resolve_md_config, path)


def test_a_reservoir_section_is_refused_by_name_even_under_rest2(tmp_path):
    path = tmp_path / "REST2.config"
    path.write_text("protocol: REST2\nreservoir:\n  enabled: false\n", encoding="utf-8")
    assert "reservoir" in _refused(resolve_md_config, path)


def test_an_input_asking_for_rrest2_is_refused_by_name(tmp_path):
    path = tmp_path / "rREST2.in"
    path.write_text("&cntrl\n  protocol = rREST2,\n/\n", encoding="utf-8")
    _refused(parse_run_input, path)


def test_an_input_carrying_a_reservoir_key_is_refused_by_name(tmp_path):
    path = tmp_path / "REST2.in"
    path.write_text("&cntrl\n  protocol = REST2,\n/\n&remd\n  reservoir_enabled = false,\n/\n",
                    encoding="utf-8")
    assert "reservoir_enabled" in _refused(parse_run_input, path)
