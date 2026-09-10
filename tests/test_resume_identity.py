"""A resume must be blocked by a different CALCULATION, never by a different document.

The gate on `resolved.config` compared two resolved documents as dictionaries, so any difference
at all refused. That is the wrong instrument, and it failed in the field: `ais.work_measurement`
and `ais.verify_every_updates` were added to the schema with defaults, and every REST2
`resolved.config` written before that commit therefore differs from every one written after -- by
two settings a REST2 run never reads.

The consequence was not cosmetic. Every in-flight ladder became unresumable, the only exit being
`--overwrite`, which destroys the outputs. Three 500 ns ladders were restarted from zero. An
engine upgrade must never do that: the runs that most need a fix are the long ones already
running.

What must NOT be lost is the protection itself. A run whose frame cadence changed mid-flight must
still refuse -- that is a different calculation and the samples would not be uniformly spaced.
These tests hold both ends: the irrelevant field resumes, the load-bearing one refuses AND names
itself.

PLATFORM_POLICY_EXEMPTION: configuration comparison and argument parsing. Nothing is propagated.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from md_tools.run import resume_identity


def _rest2_document(**overrides):
    """A resolved REST2 document of the shape `md-run` writes."""
    document = {
        "schema_version": resume_identity.SCHEMA_VERSION,
        "protocol": "REST2",
        "solvent": "explicit",
        "dynamics": {"timestep_fs": 4.0, "temperature_K": 300.0, "seed": 7},
        "stages": {"production_steps": 1000},
        "reporting": {"crd_printout_solute": 1250, "info_printout": 12500,
                      "checkpoint_printout": 250000},
        "rest2": {"number_of_replicas": 4, "tau_max": 0.5,
                  "exchange_interval_steps": 2500, "number_of_exchanges": 100},
        "reservoir": {"enabled": False},
        "collective_variables": {},
    }
    document.update(overrides)
    return document


# --- the failure that cost a campaign three ladders ------------------------------------------

def test_a_defaulted_field_from_an_unrelated_protocol_does_not_block_a_resume():
    """The exact regression: two `ais:` settings, defaulted, on a REST2 run.

    This is the whole point. A REST2 ladder never reads `ais:`, so no value there can make its
    samples come from a different calculation.
    """
    stored = _rest2_document()
    stored.pop("schema_version")                      # written before the version existed
    upgraded = _rest2_document(ais={"work_measurement": "work", "verify_every_updates": 0})

    assert resume_identity.differences(stored, upgraded) == []


def test_an_unrelated_field_that_was_deliberately_set_is_still_not_this_run_s_business():
    """Not merely defaults: a REST2 run does not care what `ais:` says, chosen or not."""
    stored = _rest2_document()
    stored.pop("schema_version")
    other = _rest2_document(ais={"work_measurement": "components", "verify_every_updates": 25})
    assert resume_identity.differences(stored, other) == []


def test_a_new_field_in_a_RELEVANT_section_resumes_only_while_it_sits_at_its_default():
    """A defaulted addition is an upgrade artefact; a chosen value is a decision.

    `rest2.equilibration_steps` is load-bearing for a ladder, so its DEFAULT arriving by upgrade is
    not a change, and a non-default value is.
    """
    from md_tools.build.md import MD_SCHEMA

    default = MD_SCHEMA.sections["rest2"].fields["equilibration_steps"].default
    stored = _rest2_document()
    stored.pop("schema_version")

    at_default = _rest2_document()
    at_default["rest2"]["equilibration_steps"] = default
    assert resume_identity.differences(stored, at_default) == []

    chosen = _rest2_document()
    chosen["rest2"]["equilibration_steps"] = default + 5000
    assert resume_identity.differences(stored, chosen) == ["rest2.equilibration_steps"]


# --- and what must still refuse ---------------------------------------------------------------

@pytest.mark.parametrize("section,key,value", [
    ("reporting", "crd_printout_solute", 2500),
    ("reporting", "info_printout", 25000),
    ("reporting", "checkpoint_printout", 500000),
    ("rest2", "exchange_interval_steps", 5000),
    ("rest2", "number_of_replicas", 6),
    ("dynamics", "timestep_fs", 2.0),
])
def test_a_load_bearing_change_still_refuses_and_names_the_field(section, key, value):
    """Each of these changes the data. The refusal must say WHICH, not merely that one exists.

    The reporting intervals matter especially: after the ladder began honouring
    `crd_printout_solute` and `info_printout`, a run started before that legitimately differs in
    frame spacing, and refusing it is correct. Fixing the unrelated-field defect must not have
    loosened this.
    """
    stored = _rest2_document()
    changed = _rest2_document()
    changed[section][key] = value

    differing = resume_identity.differences(stored, changed)
    assert differing == [f"{section}.{key}"]

    message = resume_identity.explain(Path("resolved.config"), differing, stored, changed)
    assert f"{section}.{key}" in message
    assert repr(stored[section][key]) in message and repr(value) in message


def test_the_protocol_itself_changing_is_always_a_difference():
    stored = _rest2_document()
    assert "protocol" in resume_identity.differences(stored, _rest2_document(protocol="rREST2"))


def test_the_schema_version_is_never_itself_a_difference():
    """It is metadata about the document, not about the calculation."""
    stored = _rest2_document(schema_version=1)
    now = _rest2_document(schema_version=resume_identity.SCHEMA_VERSION)
    assert resume_identity.differences(stored, now) == []


def test_an_ais_run_compares_its_own_sections_and_a_rest2_run_does_not():
    """Section relevance is per protocol, and it comes from one place.

    `ais.work_measurement` is invisible to REST2 and decisive for AIS -- the same field, and the
    difference is which run is being resumed.
    """
    stored_ais = {"schema_version": resume_identity.SCHEMA_VERSION, "protocol": "AIS",
                  "solvent": "explicit", "dynamics": {}, "reporting": {},
                  "collective_variables": {},
                  "ais": {"work_measurement": "components"}, "ais_source": {}}
    changed = dict(stored_ais, ais={"work_measurement": "work"})
    assert resume_identity.differences(stored_ais, changed) == ["ais.work_measurement"]

    assert "rest2" not in resume_identity.sections_for("AIS")
    assert "ais" not in resume_identity.sections_for("REST2")
    assert "ais" in resume_identity.sections_for("AIS")


def test_an_unknown_protocol_is_compared_strictly():
    """A protocol this build does not know gets everything compared, not nothing.

    Failing safe: refusing a resume that would have been fine costs a rerun, permitting one that
    changes the calculation costs the result.
    """
    relevant = resume_identity.sections_for("SOMETHING-NEW")
    assert "ais" in relevant and "rest2" in relevant and "reporting" in relevant


# --- the second gate, which is what actually guards the physics --------------------------------

def test_the_schedule_identity_still_catches_a_cadence_change_independently():
    """The document gate is the OUTER one. Loosening it must not be the only thing standing.

    `ReplicaRun.compare_identity` compares the schedule stored inside the exchange record, and
    `schedule.describe()` puts all three intervals there. That gate is untouched by this change
    and refuses by name -- which is why scoping the document comparison is safe.
    """
    from md_tools.remd.driver import ReplicaRun

    before = {"format": "v2", "checkpoint_interval_ps": 10.0,
              "solute_output_interval_ps": 10.0, "whole_output_interval_ps": 10.0}
    now = {"format": "v2", "checkpoint_interval_ps": 1000.0,
           "solute_output_interval_ps": 5.0, "whole_output_interval_ps": 50.0}
    assert ReplicaRun.compare_identity(before, now) == [
        "checkpoint_interval_ps", "solute_output_interval_ps", "whole_output_interval_ps"]


def test_the_three_intervals_reach_the_stored_identity_at_all():
    """The test above is only meaningful if these keys genuinely appear in the identity."""
    from md_tools.remd.schedule import EventSchedule

    described = EventSchedule(
        timestep_fs=4.0, exchange_interval_ps=10.0, number_of_exchanges=10,
        whole_output_interval_ps=50.0, solute_output_interval_ps=5.0,
        checkpoint_interval_ps=1000.0).describe()
    assert described["solute_output_interval_ps"] == 5.0
    assert described["whole_output_interval_ps"] == 50.0
    assert described["checkpoint_interval_ps"] == 1000.0


# --- a message must not name a flag the command in hand does not have --------------------------

def test_no_message_tells_an_md_run_user_to_pass_a_flag_md_run_does_not_define():
    """`--force` is real on the generated ladder and on the executor, and NOT on `md-run`.

    Both messages that named it are reached from `md-run`, so a user following the instruction got
    `unrecognized arguments: --force` and fell back to deleting files by hand -- the more
    dangerous of the two remedies the message offered.

    This scans remedy instructions rather than every mention of a flag: a docstring describing
    another command is fine, an imperative telling THIS user to pass something is not.
    """
    from md_tools.run.main import md_run_parser

    defined = set()
    for action in md_run_parser()._actions:
        defined.update(o for o in action.option_strings if o.startswith("--"))

    # Modules on the `md-run` path. The executor and the generated script have their own parsers
    # and their own flags; these are the ones whose messages a plain `md-run` can print.
    on_the_md_run_path = ("run/main.py", "remd/generated.py", "remd/state_trajectories.py",
                          "md/stage.py", "ais/run.py")
    remedy = re.compile(r"(?:pass|Pass)\s+(--[a-z][a-z0-9-]+)")
    root = Path(__file__).resolve().parents[1] / "src" / "md_tools"

    offenders = []
    for relative in on_the_md_run_path:
        path = root / relative
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for flag in remedy.findall(node.value):
                    if flag not in defined:
                        offenders.append(f"{relative}:{node.lineno} says 'pass {flag}'")
    assert not offenders, (
        "these messages tell an md-run user to pass a flag md-run does not define:\n  "
        + "\n  ".join(offenders) + f"\n  md-run defines: {' '.join(sorted(defined))}")
