"""`rem.log` regeneration must not replay the exchange history record by record.

The log is a PROJECTION of `exchange.nc`, rebuilt in full at every checkpoint and replaced
atomically. That design is right and is not what is under test here: rebuilding in full is what
makes the log crash-safe, because a torn append would disagree with the authoritative record and
nothing in the file would say which half was true.

What was wrong is how the rebuild READ. `_write_rem_log` gathered the reduced potentials with
`reduced_potentials(i)` once per exchange, and netCDF4 charges per indexed access, so each
regeneration cost O(N) in the run's own length with a large constant. Measured on a real 4-state
record of 16,631 exchanges: 10.58 s for the per-record loop against 0.30 s for one slice of the
same data. A ladder therefore decayed as it ran, and because a configured `checkpoint_printout`
never reached the schedule (see `test_ladder_reporting_intervals.py`) the regeneration fired every
exchange instead of every thousandth.

Two properties are pinned:

  1. THE BYTES DO NOT CHANGE. The bulk path and a per-record reference must render identical logs.
     This is the test that lets a reader trust the change, because the fix is only worth having if
     it is invisible in the output.
  2. THE READ IS O(1) IN ACCESSES, not O(N). Counted, never timed: a timing assertion on a shared
     machine fails for reasons that have nothing to do with the code, and passes on a fast disk
     even when the loop came back.

PLATFORM_POLICY_EXEMPTION: NetCDF reads and text rendering. No dynamics are propagated and no
platform is selected -- what is under test is how many times a variable is indexed.
"""
from __future__ import annotations

import numpy as np
import pytest

from md_tools.remd import rem_log, storage

N_STATES = 4
#: Enough records that a per-record read is unmistakable in the access count, and few enough that
#: the file is written in well under a second.
N_EXCHANGES = 400


def _identity(n_states, n_exchanges, exchange_steps):
    return {"n_states": n_states, "tau": [0.0, 0.1667, 0.3333, 0.5][:n_states],
            "number_of_exchanges": n_exchanges,
            "schedule": {"exchange_steps": exchange_steps,
                         "total_steps": exchange_steps * n_exchanges},
            "temperature_k": 300.0, "is_temperature_remd": False}


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    """A synthetic `exchange.nc` whose reduced potentials all DIFFER from one another.

    Distinct values on purpose: a history of zeros would render identically however it was
    gathered, so an off-by-one or a repeated row would survive the byte comparison. Here every
    record carries its own numbers and any misgathering changes the log.
    """
    path = tmp_path_factory.mktemp("remlog") / "exchange.nc"
    identity = _identity(N_STATES, N_EXCHANGES, 2500)
    reporter = storage.ReplicaReporter.create(
        path, n_states=N_STATES, n_atoms=4, n_solute_atoms=1, has_box=True,
        identity=identity, metadata={"note": "bulk-read regression"})
    reporter.write_ladder(identity["tau"])

    rng = np.random.default_rng(20260909)
    for index in range(N_EXCHANGES):
        proposed = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        accepted = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        for lower in range(N_STATES - 1):
            proposed[lower, lower + 1] = proposed[lower + 1, lower] = 1
            if rng.random() < 0.5:
                accepted[lower, lower + 1] = accepted[lower + 1, lower] = 1
        u = rng.normal(-500.0, 25.0, size=(N_STATES, N_STATES))
        reporter.write_exchange(index, step=(index + 1) * 2500, time_ps=(index + 1) * 10.0,
                                state_to_walker=list(range(N_STATES)),
                                proposed=proposed, accepted=accepted, u=u,
                                u_evaluated=np.ones((N_STATES, N_STATES), dtype=np.int8))
    reporter.close() if hasattr(reporter, "close") else None
    return path


def _render(exchanges, n_states=N_STATES):
    blocks = rem_log.build(n_states=n_states, exchanges=exchanges, beta=1.0 / (0.0083144621 * 300.0),
                           temperature_k=300.0)
    return blocks


def test_the_bulk_gather_renders_the_same_bytes_as_a_per_record_gather(record, tmp_path):
    """The fix must be invisible in the output. Rendered both ways, compared as text."""
    reader = storage.ReplicaReporter(record, mode="r")
    last = reader.last_exchange()
    assert last == N_EXCHANGES - 1
    accepted, proposed = reader.statistics()

    # The OLD gather, kept here as the reference implementation rather than as shipped code.
    per_record = [(proposed[i], accepted[i], reader.reduced_potentials(i)[0])
                  for i in range(last + 1)]
    # The NEW gather.
    u_history, evaluated = reader.reduced_potentials_upto(last)
    bulk = [(proposed[i], accepted[i], u_history[i]) for i in range(last + 1)]

    left = tmp_path / "per_record.log"
    right = tmp_path / "bulk.log"
    rem_log.write(left, _render(per_record), remlog_name="rem.log")
    rem_log.write(right, _render(bulk), remlog_name="rem.log")
    assert left.read_text() == right.read_text()
    # Not merely equal text: a non-trivial log, so equality is a real comparison.
    assert len(right.read_text().splitlines()) > N_EXCHANGES


def test_the_bulk_gather_returns_exactly_what_the_per_record_reader_returns(record):
    """Value-for-value, including `u_evaluated`, which the log does not render but callers read."""
    reader = storage.ReplicaReporter(record, mode="r")
    last = reader.last_exchange()
    u_history, evaluated = reader.reduced_potentials_upto(last)
    assert u_history.shape == (last + 1, N_STATES, N_STATES)
    for index in range(last + 1):
        one_u, one_evaluated = reader.reduced_potentials(index)
        assert np.array_equal(u_history[index], one_u)
        assert np.array_equal(evaluated[index], one_evaluated)


def test_regenerating_the_log_reads_the_potentials_a_constant_number_of_times(record, monkeypatch):
    """COUNTED, not timed. The whole defect was an access count that grew with the history.

    A timing assertion would be the wrong instrument twice over: it fails on a loaded machine for
    reasons unrelated to this code, and on a fast disk it passes even if the per-record loop
    returns. The number of times the `u` variable is indexed is the thing that actually changed.
    """
    reader = storage.ReplicaReporter(record, mode="r")
    last = reader.last_exchange()

    class _Counting:
        """Wraps one netCDF variable and counts how many times it is indexed."""

        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.reads = 0

        def __getitem__(self, key):
            self.reads += 1
            return self._wrapped[key]

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    counted = {name: _Counting(reader.dataset.variables[name]) for name in ("u", "u_evaluated")}
    variables = dict(reader.dataset.variables)
    variables.update(counted)
    monkeypatch.setattr(type(reader.dataset), "variables",
                        property(lambda self: variables), raising=False)

    reader.reduced_potentials_upto(last)
    assert counted["u"].reads == 1, (
        f"one slice must index `u` once; it indexed it {counted['u'].reads} times over "
        f"{last + 1} exchanges")
    assert counted["u_evaluated"].reads == 1

    # And the per-record reader, for contrast: this is what the write path used to do.
    before = counted["u"].reads
    for index in range(last + 1):
        reader.reduced_potentials(index)
    assert counted["u"].reads - before == last + 1
