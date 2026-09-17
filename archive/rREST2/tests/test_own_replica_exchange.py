"""rREST2 cases moved out of tests/test_own_replica_exchange.py when rREST2 was archived (0.5.4).

Kept as the record of what the method was tested against; not collected, and not expected to
run against current md-tools. See archive/rREST2/README.md and the tag rREST2-final.
"""
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

openmm = pytest.importorskip("openmm")

from md_tools.remd import rules as exchange_rules
from md_tools.remd import statistics as statistics
from md_tools.remd import reservoir as rrest2_reservoir
from md_tools.remd.protocol import REST2Protocol

#: At the time of archiving, reservoir.py and rrest2_exchange.py lived here.
TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"


class _Context:
    """The minimum an exchange rule may see, with a fixed reduced-potential matrix."""

    def __init__(self, matrix, *, n_states, seed, exchange_index=0, reservoir=None,
                 rule_state=None):
        self.matrix = np.asarray(matrix, dtype=float)
        self._n = n_states
        self.iteration = 1
        self.segment = 1
        self.exchange_index = exchange_index
        self.state_to_walker = list(range(n_states))
        self.rng = np.random.default_rng(seed)
        self.reservoir = reservoir
        self.rule_state = dict(rule_state or {})
        self.protocol = SimpleNamespace(n_states=n_states)

    @property
    def n_states(self):
        return self._n

    def reduced_potential(self, state, walker):
        return float(self.matrix[state][walker])

def test_a_loaded_rule_records_its_file_digest(tmp_path):
    path = tmp_path / "rule.py"
    shutil.copy2(TEMPLATES / "rrest2_exchange.py", path)
    rule, identity = exchange_rules.load_rule(path)
    assert identity["name"] == "rrest2-boltzmann"
    assert len(identity["sha256"]) == 64
    assert identity["describe"]["refresh_order"] == "after_neighbouring_sweep"



def test_a_reservoir_refresh_is_never_counted_as_an_exchange():
    accepted = np.zeros((3, 2, 2), dtype=np.int64)
    proposed = np.zeros((3, 2, 2), dtype=np.int64)
    # columns: state, frame, source step, outcome (-1 = no refresh at that row)
    events = np.array([[-1, -1, -1, -1], [1, 4, 2000, 1], [-1, -1, -1, -1]])
    stats = statistics.lifetime_statistics(accepted, proposed, tau=[0.0, 0.5],
                                           reservoir_events=events)
    assert stats["total_proposed"] == 0
    assert stats["reservoir"]["attempts"] == 1 and stats["reservoir"]["accepted"] == 1
    assert "NOT a swap" in stats["reservoir"]["note"]



def _declaration(tmp_path, **overrides):
    document = {
        "format": rrest2_reservoir.DECLARATION_FORMAT,
        "weighting": "boltzmann", "ensemble": "NVT", "prepared_directory": "reservoir",
        "refresh_interval_exchanges": 2, "random_seed": 1,
        "velocity_policy": "stored",
        "source": {"phase_space": "src/cmd.phase_space.nc", "start_time_ps": 0.0,
                   "end_time_ps": 10.0, "frames": 5},
    }
    document.update(overrides)
    path = tmp_path / "reservoir.yaml"
    path.write_text(yaml.safe_dump(document))
    return path



def _protocol():
    return REST2Protocol(tau=[0.0, 0.25, 0.5], temperature_k=300.0, timestep_fs=2.0,
                         exchange_interval_ps=2.0, number_of_exchanges=2)



def test_a_non_boltzmann_reservoir_is_refused_by_the_v1_rule(tmp_path):
    path = _declaration(tmp_path, weighting="clustered")
    with pytest.raises(rrest2_reservoir.ReservoirError) as raised:
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())
    message = str(raised.value)
    assert "not implemented" in message and "separately derived acceptance rule" in message



def test_an_npt_reservoir_is_refused(tmp_path):
    path = _declaration(tmp_path, ensemble="NPT")
    with pytest.raises(rrest2_reservoir.ReservoirError, match="distribution of volumes"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())



def test_solute_only_insertion_is_refused(tmp_path):
    path = _declaration(tmp_path, solute_only=True)
    with pytest.raises(rrest2_reservoir.ReservoirError, match="solute into an unrelated solvent"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())



def test_a_declaration_of_the_wrong_format_is_refused(tmp_path):
    path = _declaration(tmp_path, format="something-else/v9")
    with pytest.raises(rrest2_reservoir.ReservoirError, match="is 'something-else/v9'"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())



def test_the_rrest2_rule_refreshes_only_the_top_state():
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 1}, n_frames=4)
    matrix = np.zeros((4, 4))
    state = {}
    seen = []
    for attempt in range(4):
        context = _Context(matrix, n_states=4, seed=5, exchange_index=attempt,
                           reservoir=reservoir, rule_state=state)
        outcome = rule.propose(context)
        state = outcome.rule_state
        if outcome.reservoir_refresh:
            seen.append(outcome.reservoir_refresh["state"])
    assert seen and set(seen) == {3}, "a refresh must only ever touch the hottest state"



def test_the_rrest2_refresh_interval_is_honoured():
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 3}, n_frames=4)
    state, refreshes = {}, 0
    for attempt in range(9):
        outcome = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=2,
                                        exchange_index=attempt, reservoir=reservoir,
                                        rule_state=state))
        state = outcome.rule_state
        refreshes += outcome.reservoir_refresh is not None
    assert refreshes == 3, "an interval of 3 over 9 exchanges must refresh three times"



def test_the_refresh_order_is_recorded_not_left_to_scheduling():
    rule, identity = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")
    assert identity["describe"]["refresh_order"] == "after_neighbouring_sweep"
    source = (TEMPLATES / "rrest2_exchange.py").read_text()
    # The constant moved with the rule into `md_tools.remd.reservoir`; the plug-in file
    # is now only the worked example of the --exchange-rule contract.
    from md_tools.remd import reservoir as _reservoir

    assert "REFRESH_ORDER" in Path(_reservoir.__file__).read_text(encoding="utf-8")



def test_a_refresh_carries_a_recorded_velocity_seed():
    """A DCD has no velocities, so momenta are redrawn -- reproducibly."""
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 1}, n_frames=4)
    first = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=11, reservoir=reservoir))
    second = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=11, reservoir=reservoir))
    assert first.reservoir_refresh["velocity_seed"] == second.reservoir_refresh["velocity_seed"]
    assert first.reservoir_refresh["frame"] == second.reservoir_refresh["frame"]
