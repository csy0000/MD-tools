"""An interrupted ladder that stored no whole-system frame must still be resumable.

THE DEFECT THIS PINS. `validate_replica_output` is used for two different questions: "is this a
complete, coherent run" (`expect_completed=True`, the extension path) and "may this run be
continued" (`expect_completed=False`, the resume path, `driver.py:997`). The exchange-budget checks
honour that flag. The coordinate-stream check did not:

    if reporter.last_frame() < 0:
        result.fail("no whole-system frame was ever stored")

unguarded -- while the check three lines below it, for the solute stream, gets the same question
right by conditioning on whether an interval was configured at all:

    solute_interval = identity.get("solute_output_interval_steps")
    if solute_interval and reporter.last_solute_frame() < 0:

So a ladder whose configuration sets no whole-system cadence -- which is the DEFAULT; `REST2.config`
leaves it unset and a run records `whole_output_interval_steps: None` -- is refused continuation
after an interruption, for lacking a property it was never asked to produce. Found by interrupting a
real 4-state ladder at step 266500 of 1000000: the checkpoint and the run-state agreed on the step,
the exchange index and the walker count, and `--resume` was still refused with
`INVALID: 1 problem(s) - no whole-system frame was ever stored`.

Two things must remain true, which is why this is not simply "delete the check":

  * a COMPLETED run that stored no whole-system frame is still a broken run, because completion
    means every promised stream was written;
  * a run that CONFIGURED a whole-system interval and then stored nothing is broken either way --
    the promise was made and not kept, exactly as for the solute stream.

PLATFORM_POLICY_EXEMPTION: no Context and no propagation. Synthetic storage only; what is under
test is which of two questions the validator is answering.
"""
from __future__ import annotations

import numpy as np
import pytest

from md_tools.remd import storage, validate
from md_tools.remd.engine import Configuration

N_STATES = 2
N_ATOMS = 3


def _build(path, *, whole_interval, whole_frames, solute_interval=500, exchanges=4):
    """A minimal ladder storage, with the cadences under test recorded in its identity."""
    identity = {
        "tau": [0.0, 0.5],
        "n_states": N_STATES,
        "temperature_k": 300.0,
        "number_of_exchanges": exchanges,
        "whole_output_interval_steps": whole_interval,
        "solute_output_interval_steps": solute_interval,
        "exchange_interval_steps": 500,
    }
    reporter = storage.ReplicaReporter.create(
        path, n_states=N_STATES, n_atoms=N_ATOMS, n_solute_atoms=1, has_box=False,
        identity=identity, metadata={"note": "resume-precondition test"})
    reporter.write_ladder(identity["tau"])
    configurations = [Configuration(np.zeros((N_ATOMS, 3)), np.zeros((N_ATOMS, 3)), None)
                      for _ in range(N_STATES)]
    for index in range(exchanges):
        step = (index + 1) * 500
        proposed = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        accepted = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        proposed[0, 1] = proposed[1, 0] = 1
        reporter.write_exchange(
            index, step=step, time_ps=step * 0.002, state_to_walker=list(range(N_STATES)),
            proposed=proposed, accepted=accepted,
            u=np.zeros((N_STATES, N_STATES)),
            u_evaluated=np.ones((N_STATES, N_STATES), dtype=np.int8))
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=index,
                                   configurations=configurations, solute_indices=[0])
        if whole_frames:
            reporter.write_frame(step=step, time_ps=step * 0.002, exchange_index=index)
    reporter.close()
    return path


def _problems(result):
    return [str(p) for p in getattr(result, "problems", [])]


def test_an_interrupted_run_with_no_whole_cadence_may_be_continued(tmp_path):
    """THE DEFECT. No whole-system interval was configured, so no frame is owed."""
    analysis = _build(tmp_path / "run.nc", whole_interval=None, whole_frames=False)
    result = validate.validate_replica_output(analysis=str(analysis), expect_completed=False)
    assert result.ok, (
        "an interrupted ladder that never configured a whole-system cadence was refused "
        "continuation for lacking a whole-system frame:\n  " + "\n  ".join(_problems(result)))
    assert not any("whole-system frame" in p for p in _problems(result))


def test_a_completed_run_with_no_whole_cadence_is_still_coherent(tmp_path):
    """Completion does not require a stream the configuration never asked for either."""
    analysis = _build(tmp_path / "run.nc", whole_interval=None, whole_frames=False)
    result = validate.validate_replica_output(analysis=str(analysis), expect_completed=True)
    assert not any("whole-system frame" in p for p in _problems(result)), _problems(result)


def test_a_configured_whole_cadence_that_stored_nothing_is_still_refused(tmp_path):
    """The promise half, which must NOT be relaxed.

    A run that declared a whole-system interval and then wrote no frame is broken whether it
    finished or was interrupted -- the same rule the solute stream already follows.
    """
    analysis = _build(tmp_path / "run.nc", whole_interval=500, whole_frames=False)
    for expect_completed in (True, False):
        result = validate.validate_replica_output(
            analysis=str(analysis), expect_completed=expect_completed)
        assert not result.ok, f"expect_completed={expect_completed} should refuse"
        assert any("whole-system frame" in p for p in _problems(result)), _problems(result)


def test_a_run_that_stored_its_promised_whole_frames_passes(tmp_path):
    """The ordinary case, unchanged."""
    analysis = _build(tmp_path / "run.nc", whole_interval=500, whole_frames=True)
    result = validate.validate_replica_output(analysis=str(analysis), expect_completed=False)
    assert result.ok, _problems(result)


def test_the_two_stream_checks_are_conditioned_the_same_way():
    """The whole-system and solute checks answer the same question and must agree in shape.

    Asserted against the source because the asymmetry -- one guarded by its configured interval,
    the other unguarded -- is what produced the defect, and a future edit could reintroduce it
    without any of the cases above failing.
    """
    import inspect

    # `_validate` is where the checks live; `validate_replica_output` opens the file and
    # delegates, so introspecting the public name finds none of this.
    source = inspect.getsource(validate._validate)
    block = source[source.index("-- the coordinate streams"):source.index("-- the checkpoint")]
    assert "whole_output_interval_steps" in block, (
        "the whole-system check must condition on the configured interval, as the solute check "
        "does; an unconditional failure treats an interrupted run as though it had finished")
