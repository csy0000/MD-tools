"""A ladder must be able to relax each rung under its OWN Hamiltonian before it exchanges.

WHAT WAS UNREACHABLE

    `ReplicaDriver._equilibrate` propagates every owned rung for `protocol.equilibration_steps`
    against that rung's own scaled System, before the first exchange attempt and outside the
    production budget. It has always done that. `REST2Protocol` accepts `equilibration_ps`, and
    `Schedule` turns it into an exact step count.

    But `protocol_file_text` -- the function that writes the protocol module `build-md` emits --
    passed `equilibration_ps=0.0` as a literal, and the `rest2` configuration section had no field
    that could say otherwise. So through the public build path the capability did not exist.

WHY THAT MATTERS, AND WHY THE GROUP FILE IS NOT THE ANSWER

    A ladder starts from ONE coordinate file. That is deliberate: `_require_homogeneous_groups`
    refuses a group file whose lines name different `coordinates`, because N rungs must be states
    of the same system. So every rung begins from a configuration equilibrated at whatever tau
    produced the starting state -- in practice tau = 0.

    With no per-state equilibration, the hot rungs open by relaxing out of a distribution that is
    not theirs, and every one of those samples is counted and reported as production. The run
    looks healthy: exchanges are attempted, acceptance is finite, the logs are complete. The
    opening of each hot rung's series is simply not from the ensemble it is labelled with.

WHAT IS ASSERTED HERE

    That the number reaches the protocol, in the right units, by the same steps-to-picoseconds
    expression the exchange interval uses -- and that the driver then actually propagates each
    rung under its own Hamiltonian for that many steps, rather than the number merely appearing
    in a file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

TIMESTEP_FS = 2.0
EXCHANGE_STEPS = 5000
EQUILIBRATION_STEPS = 50_000
#: By hand, with the same expression the exchange interval uses: steps * fs / 1000.
EXPECTED_EQUILIBRATION_PS = EQUILIBRATION_STEPS * TIMESTEP_FS / 1000.0
EXPECTED_EXCHANGE_PS = EXCHANGE_STEPS * TIMESTEP_FS / 1000.0


def _ladder(**overrides):
    ladder = {
        "protocol": "REST2",
        "solvent": "implicit",
        "n_states": 6,
        "tau_max": 0.5,
        "exchange_interval_steps": EXCHANGE_STEPS,
        "number_of_exchanges": 100,
        "equilibration_steps": EQUILIBRATION_STEPS,
        "state_trajectory": True,
        "rem_log": True,
        "neighbour_acceptance_report": True,
        "reservoir": {},
        "dynamics": {"timestep_fs": TIMESTEP_FS, "temperature_K": 300.0,
                     "friction_per_ps": 1.0, "seed": 20260907, "platform": None},
        "collective_variables": {"interval_steps": 500},
    }
    ladder.update(overrides)
    return ladder


def test_the_configured_step_count_reaches_the_protocol_in_picoseconds():
    """Steps in the configuration, picoseconds in the protocol, one conversion."""
    from md_tools.remd.generated import protocol_file_text

    text = protocol_file_text(_ladder())
    assert f"equilibration_ps={EXPECTED_EQUILIBRATION_PS}" in text, text[:2000]
    # And it is not the exchange interval wearing another name.
    assert EXPECTED_EQUILIBRATION_PS != EXPECTED_EXCHANGE_PS
    assert f"exchange_interval_ps={EXPECTED_EXCHANGE_PS}" in text


def test_zero_stays_zero_and_is_still_written():
    """The default must remain 0.0, so an unchanged configuration keeps its behaviour."""
    from md_tools.remd.generated import protocol_file_text

    text = protocol_file_text(_ladder(equilibration_steps=0))
    assert "equilibration_ps=0.0" in text


def test_a_missing_key_does_not_crash_an_older_ladder_document():
    """`.get(...) or 0`, so a ladder dict assembled before this field existed still generates."""
    from md_tools.remd.generated import protocol_file_text

    ladder = _ladder()
    del ladder["equilibration_steps"]
    assert "equilibration_ps=0.0" in protocol_file_text(ladder)


def test_the_schedule_turns_those_picoseconds_back_into_the_same_step_count():
    """The round trip, because a conversion that does not invert is a silent budget change."""
    from md_tools.remd.schedule import EventSchedule

    schedule = EventSchedule(timestep_fs=TIMESTEP_FS,
                             exchange_interval_ps=EXPECTED_EXCHANGE_PS,
                             number_of_exchanges=100,
                             equilibration_ps=EXPECTED_EQUILIBRATION_PS)
    assert schedule.equilibration_steps == EQUILIBRATION_STEPS


def test_build_md_accepts_the_field_and_reports_it(tmp_path):
    """End to end through the public command, including the log line a reader relies on."""
    config = tmp_path / "REST2.config"
    config.write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "dynamics": {"timestep_fs": TIMESTEP_FS, "temperature_K": 300.0, "seed": 20260907},
        "stages": {"minimization_iterations": 100, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": 6, "tau_max": 0.5,
                  "exchange_interval_steps": EXCHANGE_STEPS, "number_of_exchanges": 100,
                  "equilibration_steps": EQUILIBRATION_STEPS},
        "reporting": {"solute_printout": EXCHANGE_STEPS, "system_printout": EXCHANGE_STEPS,
                      "checkpoint_printout": EXCHANGE_STEPS},
    }), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", "./REST2", "--config", str(config)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # `build-md` writes an entry point and a `resolved.config`; `_protocol.py` is materialised at
    # RUN time from that configuration. So the build-time assertion is on the configuration, and
    # the run-time reconstruction is asserted separately below -- that reconstruction is the path
    # that actually decides what the driver does, and it has its own way of dropping a field.
    import yaml as _yaml

    resolved = _yaml.safe_load(
        (tmp_path / "REST2" / "resolved.config").read_text(encoding="utf-8"))
    assert resolved["rest2"]["equilibration_steps"] == EQUILIBRATION_STEPS

    from md_tools.remd.generated import ladder_from_resolved, protocol_file_text

    ladder = ladder_from_resolved(resolved, "REST2")
    assert ladder["equilibration_steps"] == EQUILIBRATION_STEPS
    assert f"equilibration_ps={EXPECTED_EQUILIBRATION_PS}" in protocol_file_text(ladder)

    combined = done.stdout + "".join(
        p.read_text(encoding="utf-8") for p in sorted((tmp_path / "REST2").rglob("*.log")))
    assert "equilibration" in combined.lower()
    assert str(EQUILIBRATION_STEPS) in combined


@pytest.mark.slow
def test_the_driver_propagates_each_rung_under_its_own_hamiltonian(monkeypatch):
    """The number must MOVE each rung, and under its own scaled System -- not merely be written.

    A recorded step count that nothing propagates is exactly the failure this test exists for, so
    the engine is stubbed and the calls are counted per state index. The driver asks the engine
    for state `index`, which is the rung that owns that tau; asserting on the index is asserting
    that each rung was relaxed under its own Hamiltonian rather than six times under tau = 0.
    """
    from md_tools.remd.driver import ReplicaRun

    calls: list[tuple[int, int]] = []

    class _Engine:
        def set_configuration(self, index, configuration):
            pass

        def propagate(self, index, steps):
            calls.append((index, steps))

        def get_configuration(self, index):
            return "configuration"

    class _Coordinator:
        rank, size, is_root = 0, 1, True

    driver = ReplicaRun.__new__(ReplicaRun)
    driver.engine = _Engine()
    driver.owned = [0, 1, 2, 3, 4, 5]
    driver.coordinator = _Coordinator()

    class _Protocol:
        equilibration_steps = EQUILIBRATION_STEPS
        n_states = 6

    driver.protocol = _Protocol()
    driver._gather_configurations = lambda state: state["configurations"]

    state = {"configurations": ["c"] * 6, "state_to_walker": list(range(6))}
    driver._equilibrate(state)

    assert calls == [(index, EQUILIBRATION_STEPS) for index in range(6)], calls


@pytest.mark.slow
def test_zero_steps_propagates_nothing():
    """The control: with the default, no rung is relaxed at all and the driver returns early."""
    from md_tools.remd.driver import ReplicaRun

    calls = []

    class _Engine:
        def set_configuration(self, index, configuration):
            calls.append(index)

        def propagate(self, index, steps):
            calls.append((index, steps))

    driver = ReplicaRun.__new__(ReplicaRun)
    driver.engine = _Engine()
    driver.owned = [0, 1]

    class _Protocol:
        equilibration_steps = 0

    driver.protocol = _Protocol()
    driver._equilibrate({"configurations": ["c", "c"]})
    assert calls == []


def test_a_resolved_config_written_before_the_field_existed_still_reconstructs():
    """The run-time reconstruction reads configurations older than this change.

    `build-md`'s schema fills a default, but `ladder_from_resolved` also runs against
    `resolved.config` files already on disk in generated directories. A missing key there must
    mean "no per-state equilibration", exactly as before, and not a KeyError that makes an
    existing ladder unrunnable.
    """
    from md_tools.remd.generated import ladder_from_resolved

    resolved = {
        "protocol": "REST2", "solvent": "implicit",
        "rest2": {"number_of_replicas": 6, "tau_max": 0.5, "exchange_interval_steps": 5000,
                  "number_of_exchanges": 100, "state_trajectory": True, "rem_log": True,
                  "neighbour_acceptance_report": True},
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "friction_per_ps": 1.0,
                     "seed": 1},
        "reservoir": {}, "collective_variables": {"file": None, "interval_steps": 0},
    }
    assert ladder_from_resolved(resolved, "REST2")["equilibration_steps"] == 0
