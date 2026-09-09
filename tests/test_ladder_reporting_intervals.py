"""The REST2 ladder must honour the reporting intervals its configuration resolved.

Every other protocol's stages have always carried `solute_printout`, `system_printout` and
`checkpoint_printout` through to the run. The ladder did not: `protocol_file_text` hard-wired both
output intervals to the exchange interval and passed no checkpoint interval at all, so
`ReplicaSchedule` fell back to its own default -- also the exchange interval.

The values were resolved, logged, and written into `resolved.config`, and then discarded without a
word. A campaign asking for 5 ps solute frames got 10 ps ones, asking for 50 ps whole-system
frames got 10 ps ones, and asking for a checkpoint every 1 ns got one every exchange: a hundred
times more often, each one regenerating `rem.log` in full (see `test_rem_log_bulk_read.py`).

That last one cost throughput. The first two changed the DATA, silently, in a way a user could
only discover by reading generated source -- which is why this file exists.

PLATFORM_POLICY_EXEMPTION: configuration resolution and source generation. Nothing is propagated.
"""
from __future__ import annotations

import pytest

from md_tools.remd.generated import protocol_file_text

#: 4 fs steps, so every interval below is a round number of picoseconds and the arithmetic in the
#: assertions is visible rather than incidental.
TIMESTEP_FS = 4.0
EXCHANGE_STEPS = 2500          # 10 ps
SOLUTE_STEPS = 1250            # 5 ps   -- FINER than the exchange interval
SYSTEM_STEPS = 12500           # 50 ps  -- COARSER than the exchange interval
CHECKPOINT_STEPS = 250000      # 1 ns   -- 100 exchange intervals


def _ladder(**overrides):
    """A ladder description of the shape `build-md` hands the generator."""
    ladder = {
        "protocol": "REST2",
        "solvent": "explicit",
        "n_states": 4,
        "tau_max": 0.5,
        "exchange_interval_steps": EXCHANGE_STEPS,
        "number_of_exchanges": 100,
        "equilibration_steps": 0,
        "dynamics": {"timestep_fs": TIMESTEP_FS, "temperature_K": 300.0,
                     "friction_per_ps": 1.0, "seed": 7, "platform": None},
        "reporting": {"solute_printout": SOLUTE_STEPS,
                      "system_printout": SYSTEM_STEPS,
                      "checkpoint_printout": CHECKPOINT_STEPS},
    }
    ladder.update(overrides)
    return ladder


def _protocol_values(text):
    """The keyword arguments of the generated `REST2Protocol(...)` call, as strings."""
    values = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_the_generated_protocol_carries_the_configured_intervals():
    """The three values, converted to picoseconds, and NOT the exchange interval three times.

    Each is deliberately different from the exchange interval and from the others, so a protocol
    that substituted any one of them fails on that one rather than passing by coincidence.
    """
    values = _protocol_values(protocol_file_text(_ladder()))

    assert float(values["exchange_interval_ps"]) == 10.0
    assert float(values["solute_output_interval_ps"]) == 5.0, (
        "the solute stream must be written at the configured solute_printout, not at the "
        "exchange interval")
    assert float(values["whole_output_interval_ps"]) == 50.0, (
        "the whole-system stream must be written at the configured system_printout")
    assert float(values["checkpoint_interval_ps"]) == 1000.0, (
        "the ladder must checkpoint at the configured checkpoint_printout, not once per exchange")


def test_a_disabled_interval_stays_disabled_rather_than_becoming_the_exchange_interval():
    """`0` means "no such stream". It must not be resurrected as a default."""
    ladder = _ladder()
    ladder["reporting"] = {"solute_printout": 0, "system_printout": SYSTEM_STEPS,
                           "checkpoint_printout": CHECKPOINT_STEPS}
    values = _protocol_values(protocol_file_text(ladder))
    assert values["solute_output_interval_ps"] == "None"
    assert float(values["whole_output_interval_ps"]) == 50.0


def test_a_checkpoint_interval_off_the_exchange_grid_is_refused_by_name():
    """Refused, never rounded, and the message names both numbers.

    A ladder checkpoints at exchange boundaries: that is where the state-to-walker mapping and the
    exchange RNG are jointly defined, and it is why `ReplicaSchedule` defaults the checkpoint
    interval to the exchange interval. Honouring a configured value must not give that up
    silently -- so a value that is not a whole number of exchange intervals stops the build.
    """
    ladder = _ladder()
    ladder["reporting"] = {"solute_printout": SOLUTE_STEPS, "system_printout": SYSTEM_STEPS,
                           "checkpoint_printout": 7000}      # 2.8 exchange intervals
    with pytest.raises(ValueError) as refusal:
        protocol_file_text(ladder)
    message = str(refusal.value)
    assert "7000" in message and str(EXCHANGE_STEPS) in message
    assert "checkpoint_printout" in message


def test_the_runtime_ladder_reconstruction_carries_the_reporting_block():
    """`ladder_from_resolved` is the description the RUN builds its protocol from.

    `build-md` assembles a ladder dict of its own for the log, but the generated script rebuilds
    one here when it starts, and that is the one `protocol_file_text` is handed. A field added to
    the build-time dict alone would be written into the log and still never reach the run -- the
    same trap `rest2.equilibration_steps` fell into.
    """
    from md_tools.remd.generated import ladder_from_resolved

    resolved = {
        "protocol": "REST2",
        "solvent": "explicit",
        "rest2": {"number_of_replicas": 4, "tau_max": 0.5,
                  "exchange_interval_steps": EXCHANGE_STEPS, "number_of_exchanges": 100,
                  "equilibration_steps": 0, "state_trajectory": True, "rem_log": True,
                  "neighbour_acceptance_report": True},
        "reservoir": {},
        "dynamics": {"timestep_fs": TIMESTEP_FS, "temperature_K": 300.0,
                     "friction_per_ps": 1.0, "seed": 7, "platform": None},
        "collective_variables": {},
        "reporting": {"solute_printout": SOLUTE_STEPS, "system_printout": SYSTEM_STEPS,
                      "checkpoint_printout": CHECKPOINT_STEPS},
    }
    ladder = ladder_from_resolved(resolved, "REST2")
    assert ladder["reporting"]["solute_printout"] == SOLUTE_STEPS
    assert ladder["reporting"]["system_printout"] == SYSTEM_STEPS
    assert ladder["reporting"]["checkpoint_printout"] == CHECKPOINT_STEPS

    # And through to the protocol the run executes.
    values = _protocol_values(protocol_file_text(ladder))
    assert float(values["solute_output_interval_ps"]) == 5.0
    assert float(values["whole_output_interval_ps"]) == 50.0
    assert float(values["checkpoint_interval_ps"]) == 1000.0


def test_a_ladder_with_no_reporting_block_keeps_the_historical_behaviour():
    """A `resolved.config` written before this change must not silently lose its streams.

    Absent reporting means an old description, and the honest reading of it is what it used to
    do -- both output streams at the exchange interval -- rather than `None`, which would turn
    two configured streams into one final frame each and look like a successful run.
    """
    ladder = _ladder()
    ladder.pop("reporting")
    values = _protocol_values(protocol_file_text(ladder))
    assert float(values["solute_output_interval_ps"]) == 10.0
    assert float(values["whole_output_interval_ps"]) == 10.0
    assert values["checkpoint_interval_ps"] == "None"
