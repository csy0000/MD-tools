"""The `alchemical` protocol's configuration: what it accepts, and what it refuses by name.

The window RUNTIME already exists (`md_tools.alchemy.windows.run_window`, which runs, continues
and verifies one fixed-lambda window). What is tested here is the entrance to it: a campaign
expressed as a configuration, resolved by the one authority for MD workflow configuration, and
refused when the two halves of a statement disagree.

THE PROPERTY THESE TESTS EXIST FOR is that a refusal arrives at RESOLUTION rather than on the
node. A lambda ladder is expensive: a window placement that misses a knot, a report interval that
does not divide the window, a softcore function that is not the one implemented -- each is a
configuration error or it is nothing, and it must not be a late one, discovered after
minimisation and three equilibration stages.

The arithmetic behind those refusals is NOT restated here. `md_tools.alchemy.paths`,
`md_tools.alchemy.softcore` and `md_tools.alchemy.windows` own it and refuse it at run time; the
configuration layer hands the resolved values to them, so what is asserted below is that the
refusal REACHES the person who wrote the file, under the name they wrote.

PLATFORM_POLICY_EXEMPTION: configuration resolution only. No Context, no device, no simulation.
"""
from __future__ import annotations

import textwrap

import pytest

from md_tools.build.md import ConfigError, MD_SCHEMA, resolve_md_config

#: A configuration that resolves. Every test below is a departure from this one.
COMPLETE = """
    protocol: alchemical
    solvent: explicit
    alchemical:
      cycle: RBFE
      leg: 1
      plan: transform/
      number_of_windows: 11
      window_steps: 100000
      report_interval_steps: 1000
      checkpoint_interval_steps: 10000
"""


def _config(tmp_path, body):
    path = tmp_path / "alchemical.config"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _block(**changes):
    """The resolved `alchemical` block of COMPLETE, with `changes` applied."""
    # `cycle` and `leg` are part of every valid alchemical block since 0.6.4: they name the run
    # directory `<cycle>-leg<i>-run<N>` and the shared input beside it, and a block without them
    # is refused. A test that omitted them would be testing a configuration nobody can run.
    block = {"cycle": "RBFE", "leg": 1,
             "plan": "transform/", "number_of_windows": 11, "window_steps": 100000,
             "report_interval_steps": 1000, "checkpoint_interval_steps": 10000}
    block.update(changes)
    return {key: value for key, value in block.items() if value is not None}


def _resolve(**changes):
    return MD_SCHEMA.resolve({"protocol": "alchemical", "alchemical": _block(**changes)})


# --- the protocol exists ----------------------------------------------------------------------

def test_alchemical_is_a_known_protocol():
    from md_tools.build.md import PROTOCOLS

    assert "alchemical" in PROTOCOLS


def test_a_complete_alchemical_configuration_resolves(tmp_path):
    resolved = resolve_md_config(_config(tmp_path, COMPLETE))
    assert resolved["protocol"] == "alchemical"
    assert resolved["alchemical"]["plan"] == "transform/"
    assert resolved["alchemical"]["number_of_windows"] == 11


# --- the section and the protocol are one statement -------------------------------------------

def test_a_plan_under_another_protocol_is_refused_by_name(tmp_path):
    """An alchemical ladder is a protocol, not a modifier another protocol can wear."""
    with pytest.raises(ConfigError, match=r"alchemical\.plan is set but protocol is cMD"):
        resolve_md_config(_config(tmp_path, """
            protocol: cMD
            alchemical: {plan: transform/}
        """))


def test_a_softcore_setting_under_another_protocol_is_refused_too(tmp_path):
    """Not only `plan`: any stated key is a statement about a run that is not happening.

    `sc: false` is the case worth naming. It is the setting whose silent acceptance would read as
    "softcore was deliberately disabled for this run" in a file describing a run that never had a
    softcore potential to disable.
    """
    with pytest.raises(ConfigError, match=r"alchemical\.sc is set but protocol is REST2"):
        resolve_md_config(_config(tmp_path, """
            protocol: REST2
            alchemical: {sc: false}
        """))


def test_the_section_left_at_its_defaults_does_not_make_a_cmd_run_alchemical(tmp_path):
    """The control for the two above: an untouched section states nothing, so it refuses nothing.

    Every schema Section resolves whether or not the document mentions it, so "the section is
    present" cannot be the test -- it is always present.
    """
    resolved = resolve_md_config(_config(tmp_path, "protocol: cMD\nsolvent: implicit\n"))
    assert resolved["alchemical"]["plan"] is None
    assert resolved["alchemical"]["softcore_function"] == "amber18"


def test_the_protocol_without_a_plan_is_refused(tmp_path):
    """`cycle` and `leg` are supplied so the MISSING PLAN is what this refuses on.

    Omitting all three tests only whichever check happens to run first, which is a statement about
    the order of the code rather than about the configuration. The cycle/leg refusal has its own
    test below.
    """
    with pytest.raises(ConfigError, match=r"alchemical\.plan is not set"):
        resolve_md_config(_config(tmp_path, """
            protocol: alchemical
            alchemical: {cycle: RBFE, leg: 1, number_of_windows: 11}
        """))


def test_the_protocol_without_a_cycle_and_leg_is_refused(tmp_path):
    """They name the run directory and the shared input; half of the name names nothing."""
    with pytest.raises(ConfigError, match=r"alchemical\.cycle and alchemical\.leg are not set"):
        resolve_md_config(_config(tmp_path, """
            protocol: alchemical
            alchemical: {plan: transform/, number_of_windows: 11}
        """))
    with pytest.raises(ConfigError, match=r"given together or not at all"):
        resolve_md_config(_config(tmp_path, """
            protocol: alchemical
            alchemical: {cycle: RBFE, plan: transform/, number_of_windows: 11}
        """))


# --- where the windows sit --------------------------------------------------------------------

def test_the_two_ways_of_placing_windows_are_exclusive():
    with pytest.raises(ConfigError, match="both place the windows"):
        _resolve(lambda_values="0.0, 0.5, 1.0")


def test_a_ladder_with_no_placement_at_all_is_refused():
    with pytest.raises(ConfigError, match="neither alchemical.number_of_windows nor"):
        _resolve(number_of_windows=None)


def test_written_out_lambda_values_are_accepted_and_kept_verbatim():
    resolved = _resolve(number_of_windows=None, lambda_values="0.0, 0.1, 0.4, 0.8, 1.0")
    assert resolved["alchemical"]["lambda_values"] == "0.0, 0.1, 0.4, 0.8, 1.0"


@pytest.mark.parametrize("values,message", [
    ("0.0, 0.9", "stops short of an end state"),
    ("0.1, 1.0", "stops short of an end state"),
    ("0.0, 0.6, 0.4, 1.0", "does not strictly increase"),
    ("0.0, 0.5, 0.5, 1.0", "does not strictly increase"),
    ("0.0, middle, 1.0", "is not a number"),
    ("0.0", "at least the two of them"),
])
def test_a_lambda_ladder_that_cannot_be_walked_is_refused_by_name(values, message):
    with pytest.raises(ConfigError, match=message):
        _resolve(number_of_windows=None, lambda_values=values)


def test_a_single_window_is_refused_whichever_way_it_is_written():
    """`0.5` alone parses as a number rather than a list, and must still be refused as a ladder."""
    with pytest.raises(ConfigError, match="names ONE window"):
        _resolve(number_of_windows=None, lambda_values=0.5)
    with pytest.raises(ConfigError, match="at least the two of them"):
        _resolve(number_of_windows=1)


# --- the knot of a staged path ------------------------------------------------------------------
#
# STAGED IS REFUSED SINCE X1, and these five keep their assertions rather than being deleted.
#
# They were written when the Hamiltonian moved two components. It moves THREE
# (`ALCHEMICAL_COMPONENTS`): `lambda_bonded` carries the common-core bonded terms that differ
# between the end states, and a staged path has no place to put it that somebody has chosen. Which
# stage carries it changes the path every window samples, so `alchemical_path` refuses `staged`
# and says why.
#
# `strict=True` deliberately: when the decision is recorded and staged is wired, these go from
# xfail to XPASS and the lane fails, which is how the next person finds the arithmetic that was
# already written for it instead of writing it again.
STAGED_PENDING = pytest.mark.xfail(
    strict=True, reason="alchemical.lambda_path: `staged` is refused until a scientific decision "
                        "places lambda_bonded relative to the knot (X1)")


def test_staged_is_refused_and_separates_what_is_settled_from_what_is_missing():
    """The refusal a person actually meets, and the only staged behaviour there is today.

    THE REASON CHANGED IN 0.6.4 and the test changed with it. `staged` used to be blocked on a
    scientific decision -- where the bonded components go -- and that is now settled: OpenFE's
    reference schedule moves them linearly across the whole path, unstaged, and this package
    adopts it (docs/scientific-defaults.md section 13).

    What blocks `staged` now is an implementation gap, and a different one: a faithful staged path
    is DIRECTIONAL. OpenFE discharges the disappearing region while growing the appearing one's
    core, then removes the old core while charging the new, which needs each region labelled as
    inserting or deleting. This layer moves one `lambda_electrostatics` and one `lambda_sterics`
    for both regions together, so a two-stage split would read as OpenFE's default without being
    it -- and would leave a charged region with no core at one end, the singularity staging exists
    to prevent.

    A refusal that still named the old reason would send the next person to re-decide something
    already decided, which is why this asserts the message does NOT name it.
    """
    with pytest.raises(ConfigError) as refusal:
        _resolve(lambda_path="staged", staged_knot=0.5)
    message = str(refusal.value)
    assert "not implemented" in message
    assert "SETTLED" in message and "scientific-defaults.md" in message
    assert "DIRECTIONAL" in message
    # The settled question must not still be presented as the blocker.
    assert "lambda_bonded" not in message
    # It points at what to do instead rather than only at what is wrong.
    assert "lambda_path: linear" in message


@STAGED_PENDING
def test_a_staged_path_needs_its_knot_and_a_linear_one_refuses_it():
    with pytest.raises(ConfigError, match=r"alchemical\.staged_knot is not set"):
        _resolve(lambda_path="staged")
    with pytest.raises(ConfigError, match="which has no knot"):
        _resolve(staged_knot=0.5)


@STAGED_PENDING
def test_a_window_count_that_lands_on_the_knot_is_accepted():
    """11 windows are spaced 1/10, and the knot at 0.5 is window 5."""
    resolved = _resolve(lambda_path="staged", staged_knot=0.5)
    assert resolved["alchemical"]["staged_knot"] == 0.5


@STAGED_PENDING
def test_a_window_count_that_misses_the_knot_is_refused_with_the_arithmetic():
    """NOT ROUNDED, and not moved to the nearest grid point.

    dU/ds has two values at a knot, so a trapezoid drawn across it is wrong by a finite amount no
    amount of sampling removes -- which is why `md_tools.alchemy.paths.require_knots_sampled`
    refuses it, and why this must be a configuration error rather than a result.
    """
    with pytest.raises(ConfigError) as refusal:
        _resolve(number_of_windows=12, lambda_path="staged", staged_knot=0.4)
    message = str(refusal.value)
    assert "no window at knot" in message
    # the arithmetic that would fix it, not merely the fact that it is wrong
    assert "1/11" in message and "4.4" in message
    assert "lambda_values" in message


@STAGED_PENDING
def test_written_out_values_may_place_a_window_on_an_awkward_knot():
    """The escape the refusal above points at, and proof that it is a real one."""
    resolved = _resolve(number_of_windows=None, lambda_path="staged", staged_knot=0.4,
                        lambda_values="0.0, 0.2, 0.4, 0.7, 1.0")
    assert resolved["alchemical"]["lambda_path"] == "staged"


@STAGED_PENDING
def test_a_knot_at_an_end_of_the_path_is_refused():
    """A stage of zero width is a linear path with a component that never moves; say that."""
    with pytest.raises(ConfigError, match="not strictly between 0 and 1"):
        _resolve(lambda_path="staged", staged_knot=0.0)


# --- the softcore settings, refused by the object that implements them ------------------------

def test_the_schema_defaults_are_the_softcore_settings_defaults():
    """Two copies of `scalpha = 0.5` would be one copy too many.

    The schema takes its defaults from `SoftcoreSettings`; this is what holds them together if
    somebody types a number into the `Field` instead.
    """
    from md_tools.alchemy.softcore import SoftcoreSettings

    defaults, section = SoftcoreSettings(), MD_SCHEMA.sections["alchemical"]
    for key in ("sc", "softcore_function", "scalpha", "scbeta", "sc_boundary_14"):
        assert section.fields[key].default == getattr(defaults, key), key


@pytest.mark.parametrize("changes,message", [
    ({"softcore_function": "gapsys"}, "Gapsys"),
    ({"softcore_function": "beutler"}, "Beutler"),
    ({"softcore_function": "amber20"}, "not Amber18"),
    ({"softcore_function": "smoothstep"}, "not Amber18"),
    ({"scalpha": 0.0}, r"scalpha must be a finite number > 0"),
    ({"scbeta": 0.0}, r"scbeta must be a finite number > 0"),
])
def test_a_softcore_potential_that_is_not_amber18_is_refused_by_name(changes, message):
    """A false label on a validated-looking number is what this refusal exists to prevent."""
    with pytest.raises(ConfigError, match=message):
        _resolve(**changes)


def test_the_boundary_14_rule_is_an_enumeration():
    assert _resolve(sc_boundary_14="unscaled")["alchemical"]["sc_boundary_14"] == "unscaled"
    with pytest.raises(ConfigError, match="is not one of"):
        _resolve(sc_boundary_14="gti_add_sc")


# --- the window schedule, refused by the object that runs it ----------------------------------

def test_the_window_keys_cover_every_window_setting_the_runtime_takes():
    """A `WindowSettings` field no configuration can state is a setting nobody can choose.

    The two names differ deliberately -- `window_steps` says which steps it means, beside
    `stages.production_steps` in the same document -- so this pins the MAPPING rather than the
    spelling. `timestep_fs`, `friction_per_ps`, `barostat_interval` and `seed` are absent on
    purpose: they are in `dynamics`, where every protocol reads them, and a second copy under
    `alchemical` would be a second answer to the same question.
    """
    import dataclasses

    from md_tools.alchemy.windows import WindowSettings
    from md_tools.build.md import _WINDOW_SETTING_NAMES

    runtime = {f.name for f in dataclasses.fields(WindowSettings)}
    in_dynamics = {"timestep_fs", "friction_per_ps", "barostat_interval", "seed"}
    fields = MD_SCHEMA.sections["alchemical"].fields
    for name in sorted(runtime - in_dynamics):
        assert name in _WINDOW_SETTING_NAMES, f"{name} is not mapped onto a configuration key"
        key = _WINDOW_SETTING_NAMES[name].split(".", 1)[1]
        assert key in fields, f"{_WINDOW_SETTING_NAMES[name]} is not a key of the section"


@pytest.mark.parametrize("changes,message", [
    ({"window_steps": 0}, r"alchemical\.window_steps must be a positive integer step count"),
    ({"report_interval_steps": 0},
     r"alchemical\.report_interval_steps must be a positive integer step count"),
    ({"checkpoint_interval_steps": 0},
     r"alchemical\.checkpoint_interval_steps must be a positive integer step count"),
    ({"report_interval_steps": 3000}, r"must be a multiple of alchemical\.report_interval_steps"),
    # 2500 divides the window and is not a multiple of the report interval: a generation would
    # be committed halfway through a report.
    ({"checkpoint_interval_steps": 2500},
     r"alchemical\.checkpoint_interval_steps 2500 must be a multiple of "
     r"alchemical\.report_interval_steps"),
    ({"equilibration_steps": 1500},
     r"alchemical\.equilibration_steps 1500 is not a multiple of "
     r"alchemical\.report_interval_steps"),
])
def test_a_schedule_that_would_have_to_be_rounded_is_refused(changes, message):
    """And the refusal names the key as the person WROTE it, not as the runtime spells it."""
    with pytest.raises(ConfigError, match=message):
        _resolve(**changes)


def test_the_refusal_never_quotes_the_runtime_spelling():
    """The translation is one substitution away from mangling `equilibration_steps` into
    `equilibration_alchemical.window_steps`, which is why it is asserted rather than assumed."""
    with pytest.raises(ConfigError) as refusal:
        _resolve(equilibration_steps=1500)
    message = str(refusal.value)
    assert "alchemical.equilibration_steps 1500" in message
    assert "alchemical.equilibration_alchemical" not in message
    assert "report_interval " not in message


# --- the `.in` language -----------------------------------------------------------------------

def test_every_alchemical_key_can_be_written_in_an_input():
    """A key the `.in` language cannot say is a key a generated input silently drops."""
    from md_tools.run.inputs import SECTION_KEYS

    reachable = {target for keys in SECTION_KEYS.values() for target in keys.values()}
    for key in MD_SCHEMA.sections["alchemical"].fields:
        assert f"alchemical.{key}" in reachable, f"alchemical.{key} cannot be written in a .in"


def test_an_alchemical_input_resolves_back_to_exactly_its_resolved_config(tmp_path):
    """THE ROUND TRIP. `resolved.config` is authoritative and the `.in` is what a person edits;
    they are the same document or one of them is a lie.

    Run over a configuration that departs from the defaults in every key of the section -- a
    round trip over defaults proves only that the defaults survive being left out.
    """
    from md_tools.build.md import in_file_text
    from md_tools.run.inputs import parse_run_input

    # `lambda_path` and `staged_knot` are the two keys this cannot depart on: `staged` is refused
    # (see STAGED_PENDING above), so `linear` is both the default and the only value. When staged
    # is wired, put it back here -- a round trip that only ever writes the default for a key has
    # not shown that the key survives the trip.
    resolved = _resolve(number_of_windows=None, lambda_values="0.0, 0.25, 0.5, 0.75, 1.0",
                        lambda_path="linear", sc=False, scalpha=0.25,
                        scbeta=6.0, sc_boundary_14="unscaled", equilibration_steps=5000,
                        minimize_iterations=100)
    written = tmp_path / "alchemical.in"
    written.write_text(in_file_text(resolved), encoding="utf-8")

    text = written.read_text(encoding="utf-8")
    assert "&alchemical" in text, text
    assert "lambda_values             = 0.0, 0.25, 0.5, 0.75, 1.0," in text, text

    assert parse_run_input(written).resolved == resolved


def test_the_lambda_values_survive_the_round_trip_as_a_string(tmp_path):
    """The one value whose TYPE the namelist could quietly change.

    `0.0, 0.25, 1.0` is neither an integer nor a float to the input parser, so it comes back as
    the string it went out as. A list that came back as a number would resolve to a different
    ladder, and the round-trip assertion above is what would notice.
    """
    from md_tools.build.md import in_file_text
    from md_tools.run.inputs import parse_run_input

    resolved = _resolve(number_of_windows=None, lambda_values="0.0, 0.25, 1.0")
    written = tmp_path / "alchemical.in"
    written.write_text(in_file_text(resolved), encoding="utf-8")
    back = parse_run_input(written).resolved
    assert back["alchemical"]["lambda_values"] == "0.0, 0.25, 1.0"


def test_a_preparation_input_carries_no_alchemical_block(tmp_path):
    """`input/min.in` is SHARED by every method on a system, so it holds no method identity."""
    from md_tools.build.md import in_file_text

    text = in_file_text(_resolve(), stage="min", preparation=True)
    assert "&alchemical" not in text, text
    assert "protocol" not in text, text


# --- the two surfaces, which now RUN one ------------------------------------------------------
#
# Until X1 both of these refused, and the two refusals were this branch's acceptance criteria:
# "when a window runs end to end from a generated directory, they come out." They came out. What
# each test asserts now is the behaviour that replaced its refusal, in the same place, so the
# transition is visible rather than a deletion.
#
# The END-TO-END evidence is `tests/test_alchemy_generated.py`, which needs a real plan. What is
# checked here is what this file has always checked: that the CONFIGURATION reaches the right
# place, and that a refusal still arrives before anything is created.

def test_build_md_generates_an_alchemical_run_and_needs_a_real_plan(tmp_path):
    """`alchemical.plan` is read at generation, and a directory is not created without one.

    `COMPLETE` names `transform/`, which does not exist. The refusal must be about THE PLAN -- the
    pair of end states -- and not about script generation being unimplemented, which is what it
    used to say.
    """
    from md_tools.build.md import build_scripts

    config = _config(tmp_path, COMPLETE)
    out_dir = tmp_path / "alchemical-run1"
    with pytest.raises(ConfigError, match="does not hold a plan.json"):
        build_scripts(config_path=config, out_dir=out_dir, echo=False)
    assert not out_dir.exists(), "a refused generation created its output directory"


def test_md_run_dispatches_an_alchemical_input_and_refuses_a_directory_with_no_leg(tmp_path,
                                                                                  capsys):
    """The input now REACHES the window dispatch, and is refused there by what is missing.

    `-odir` holding no `leg/` is a directory that declares a ladder whose leg was never prepared.
    That is a different statement from "this surface cannot run one", and it is the one a person
    who mistyped `-odir` needs.
    """
    from md_tools.build.md import in_file_text
    from md_tools.run.main import md_run_main

    written = tmp_path / "alchemical.in"
    written.write_text(in_file_text(_resolve()), encoding="utf-8")
    out_dir = tmp_path / "out"

    code = md_run_main(["-i", str(written), "-odir", str(out_dir), "--window", "w000", "--cpu"])
    assert code == 2
    message = capsys.readouterr().err
    assert "leg.json" in message and "never prepared" in message
    assert not out_dir.exists(), "a refused invocation created its output directory"


def test_md_run_refuses_a_second_system_for_the_windows(tmp_path, capsys):
    """`-p` and `-s` are refused BY NAME for a ladder, before -odir exists.

    They are not merely unused: a window's topology and System are the prepared leg, whose digest
    the runtime checks its rebuild against, so a second one named here would leave which of the
    two ran depending on which the runtime read.
    """
    from md_tools.build.md import in_file_text
    from md_tools.run.main import md_run_main

    written = tmp_path / "alchemical.in"
    written.write_text(in_file_text(_resolve()), encoding="utf-8")
    out_dir = tmp_path / "out"

    code = md_run_main(["-i", str(written), "-p", str(tmp_path / "built.pdb"),
                        "-odir", str(out_dir), "--window", "w000", "--cpu"])
    assert code == 2
    assert "refused for an alchemical ladder" in capsys.readouterr().err
    assert not out_dir.exists()
