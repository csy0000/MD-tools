"""`stages.number_of_segments` splits the production total, and every refusal is by name.

WHAT THIS FIELD IS FOR. A long run is written as several segments so each one is a complete,
comparable object on disk: `solute_state<i>_prod1.nc`, `..._prod2.nc`, and so on. The defect it
answers is real -- a five-chunk reference run wrote `_prod1` in every chunk, because nothing ever
passed a segment number and `state_trajectory_name` defaults to 1. Flattening those five chunks
into one directory would have collided four times.

IT SPLITS, IT DOES NOT MULTIPLY. The production total is what the configuration already states;
this decides how many files it is written as. So the only question is whether it divides exactly,
and a remainder is refused rather than rounded: a final short segment would make the last chunk
incomparable with the others, which is the comparison segments exist to enable.

THE QUANTITY DIVIDED DIFFERS BY PROTOCOL, and that is not cosmetic. A ladder has no
`production_steps` -- production is `number_of_exchanges * exchange_interval_steps` -- and its
indivisible unit is an EXCHANGE. Dividing its step count could put a boundary part-way through an
exchange interval, leaving a segment whose last interval was propagated but never attempted.

PLATFORM_POLICY_EXEMPTION: configuration resolution only. No Context, no device, no simulation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools.build.md import ConfigError, resolve_md_config


def _resolved(tmp_path, body):
    path = tmp_path / "t.config"
    path.write_text(body, encoding="utf-8")
    return resolve_md_config(path)


def _refusal(tmp_path, body):
    with pytest.raises(ConfigError) as failure:
        _resolved(tmp_path, body)
    return str(failure.value)


# -- the default -------------------------------------------------------------------------------

def test_the_default_is_one_segment(tmp_path):
    """One segment is the historical behaviour, and must stay the default."""
    assert _resolved(tmp_path, "protocol: cMD\n")["stages"]["number_of_segments"] == 1


def test_every_shipped_example_still_resolves_at_one_segment():
    """Adding a field must not change what any shipped configuration means."""
    root = Path(__file__).resolve().parents[1]
    for name in ("cMD", "REST2", "AIS", "umbrella"):
        for candidate in (root / "configs" / "md" / f"{name}.config",
                          root / "docs" / "openmm_methods" / name / "example.config"):
            if candidate.is_file():
                resolved = resolve_md_config(candidate)
                assert resolved["stages"]["number_of_segments"] == 1, candidate


# -- cMD and umbrella divide production_steps ---------------------------------------------------

@pytest.mark.parametrize("protocol", ["cMD", "umbrella"])
@pytest.mark.parametrize("segments", [1, 2, 4, 5, 10])
def test_a_divisor_of_the_production_length_is_accepted(tmp_path, protocol, segments):
    body = f"protocol: {protocol}\nstages:\n  number_of_segments: {segments}\n"
    if protocol == "umbrella":
        # An umbrella window restrains a collective variable and reports the same one, so the
        # resolver requires BOTH keys -- the window file alone is not enough.
        body += ("umbrella:\n  file: windows.yaml\n"
                 "collective_variables:\n  file: cv.yaml\n  interval_steps: 1000\n")
    resolved = _resolved(tmp_path, body)
    total = resolved["stages"]["production_steps"]
    assert total % segments == 0
    assert resolved["stages"]["number_of_segments"] == segments


def test_a_segment_count_that_does_not_divide_is_refused_with_the_arithmetic(tmp_path):
    message = _refusal(tmp_path, "protocol: cMD\nstages:\n  number_of_segments: 7\n")
    assert "does not divide" in message
    assert "stages.production_steps" in message
    assert "2500000 % 7" in message, f"the arithmetic that would fix it must be shown: {message}"
    assert "Divisors of" in message


def test_segments_of_a_zero_length_run_are_refused(tmp_path):
    """Several segments of nothing is not a thing to produce quietly."""
    message = _refusal(
        tmp_path, "protocol: cMD\nstages:\n  production_steps: 0\n  number_of_segments: 3\n")
    assert "is 0" in message and "hold nothing" in message


def test_one_segment_of_a_zero_length_run_is_still_fine(tmp_path):
    """A zero-length production run is legal; it is only SPLITTING one that is not."""
    resolved = _resolved(tmp_path, "protocol: cMD\nstages:\n  production_steps: 0\n")
    assert resolved["stages"]["production_steps"] == 0


# -- a ladder divides number_of_exchanges -------------------------------------------------------

def test_a_ladder_divides_its_exchange_count_not_its_step_count(tmp_path):
    body = "protocol: REST2\nstages:\n  number_of_segments: 5\n"
    resolved = _resolved(tmp_path, body)
    assert resolved["rest2"]["number_of_exchanges"] % 5 == 0


def test_a_ladder_refuses_a_segment_count_that_splits_an_exchange(tmp_path):
    """An exchange must never straddle a segment boundary."""
    message = _refusal(tmp_path, "protocol: REST2\nstages:\n  number_of_segments: 7\n")
    assert "rest2.number_of_exchanges" in message
    assert "does not divide" in message
    assert "500 % 7" in message


# -- AIS has nothing to split -------------------------------------------------------------------

def test_ais_refuses_segments_rather_than_ignoring_them(tmp_path):
    """An accepted-and-inert setting is worse than a refused one.

    AIS never reads `stages.production_steps`. Accepting a segment count would mean the run
    reported a number nothing honoured and wrote `prod1` regardless. A switching campaign is
    already divided by `ais.number_of_paths`.
    """
    message = _refusal(
        tmp_path,
        "protocol: AIS\nais_source:\n  trajectory: source.nc\n"
        "stages:\n  number_of_segments: 4\n")
    assert "protocol is AIS" in message
    assert "number_of_paths" in message, (
        f"the refusal must point at the setting that DOES divide a campaign: {message}")


def test_ais_at_one_segment_is_accepted(tmp_path):
    """The default must not refuse every AIS configuration."""
    resolved = _resolved(
        tmp_path, "protocol: AIS\nais_source:\n  trajectory: source.nc\n")
    assert resolved["stages"]["number_of_segments"] == 1
