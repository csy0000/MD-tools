"""The frame event writes N state trajectories and one marker, as a single commit.

After an accepted exchange the configuration occupying state 2 belongs to a different walker, and
it is that one which must appear in `remd2.nc`. These tests drive the real set rather than the
driver, so the mapping and the commit ordering are exercised without an OpenMM Context.

PLATFORM_POLICY_EXEMPTION: file-level bookkeeping on a synthetic four-atom system. No dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"
netCDF4 = pytest.importorskip("netCDF4")
sys.path.insert(0, str(TEMPLATES))

import amber_trajectory as amber                                   # noqa: E402
import replica_storage as storage                                  # noqa: E402
import state_trajectories as st                                    # noqa: E402
from replica_engine import Configuration                           # noqa: E402

TAUS = [0.0, 0.2, 0.4]
BOX = np.eye(3) * 2.5


def _configurations(marks):
    """One configuration per walker, each filled with its own recognisable number."""
    return [Configuration(np.full((4, 3), float(mark)), np.zeros((4, 3)), BOX) for mark in marks]


def _set(tmp_path):
    return st.StateTrajectorySet.create(tmp_path, taus=TAUS, n_atoms=4, temperature_k=300.0,
                                        periodic=True)


def test_each_state_gets_the_configuration_that_now_occupies_it(tmp_path):
    """The identity of a fixed-state trajectory: a walker moves, the file does not."""
    trajectories = _set(tmp_path)
    try:
        # walkers 0,1,2 carry marks 10,20,30; state->walker is the identity at first
        trajectories.write_frame(step=500, time_ps=1.0, state_to_walker=[0, 1, 2],
                                 configurations=_configurations([10, 20, 30]))
        # after an accepted 0<->1 swap the mapping changes; the walkers keep their marks
        trajectories.write_frame(step=1000, time_ps=2.0, state_to_walker=[1, 0, 2],
                                 configurations=_configurations([10, 20, 30]))
    finally:
        trajectories.close()

    for state, expected in ((0, [10.0, 20.0]), (1, [20.0, 10.0]), (2, [30.0, 30.0])):
        path = tmp_path / amber.state_trajectory_name(state)
        with netCDF4.Dataset(str(path)) as d:
            got = [float(np.array(d.variables["coordinates"][frame])[0][0]) / 10.0
                   for frame in range(2)]
        assert got == pytest.approx(expected), (
            f"remd{state}.nc must follow the STATE, taking whichever walker occupies it")


def test_the_files_carry_their_own_state_index_and_tau(tmp_path):
    trajectories = _set(tmp_path)
    trajectories.close()
    for state, tau in enumerate(TAUS):
        seen = amber.read_frames(tmp_path / amber.state_trajectory_name(state))
        assert seen["state_index"] == state
        assert seen["tau"] == pytest.approx(tau)
        assert seen["conventions"] == "AMBER"


def test_a_fresh_set_refuses_to_overwrite_existing_trajectories(tmp_path):
    _set(tmp_path).close()
    with pytest.raises(st.StateTrajectoryError, match="already exist"):
        _set(tmp_path)


# --- the commit protocol
# --------------------------------------------------------------------------

def test_a_complete_set_matches_its_marker(tmp_path):
    trajectories = _set(tmp_path)
    try:
        for frame in range(3):
            trajectories.write_frame(step=(frame + 1) * 500, time_ps=(frame + 1) * 1.0,
                                     state_to_walker=[0, 1, 2],
                                     configurations=_configurations([1, 2, 3]))
        trajectories.sync()
    finally:
        trajectories.close()
    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3, expect_frames=3, expect_taus=TAUS)
    assert seen["problems"] == [], seen["problems"]
    assert seen["facts"]["frames_per_state"] == {0: 3, 1: 3, 2: 3}


def test_rows_past_the_marker_are_uncommitted_not_corruption(tmp_path):
    """A crash between writing the set and moving the marker leaves rows nothing counts. That is
    the ordinary case, and a continuation ignores and overwrites them."""
    trajectories = _set(tmp_path)
    try:
        for frame in range(3):
            trajectories.write_frame(step=(frame + 1) * 500, time_ps=(frame + 1) * 1.0,
                                     state_to_walker=[0, 1, 2],
                                     configurations=_configurations([1, 2, 3]))
        trajectories.sync()
    finally:
        trajectories.close()

    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3, expect_frames=2)
    assert seen["problems"] == [], "an extra row is uncommitted, not a fault"
    assert seen["facts"]["uncommitted_rows"] == {0: 1, 1: 1, 2: 1}


def test_a_state_file_shorter_than_the_marker_is_corruption(tmp_path):
    """The marker only advances once every file has the row, so a short file cannot be an
    interrupted commit. It is refused, never padded."""
    trajectories = _set(tmp_path)
    try:
        trajectories.write_frame(step=500, time_ps=1.0, state_to_walker=[0, 1, 2],
                                 configurations=_configurations([1, 2, 3]))
    finally:
        trajectories.close()
    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3, expect_frames=5)
    assert any("fewer frames than the committed marker" in p for p in seen["problems"])
    assert any("corruption, not an interrupted commit" in p for p in seen["problems"])


def test_a_missing_state_file_is_refused_and_never_fabricated(tmp_path):
    trajectories = _set(tmp_path)
    trajectories.close()
    (tmp_path / amber.state_trajectory_name(1)).unlink()
    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3)
    assert any("remd1.nc is missing" in p for p in seen["problems"])
    assert any("fabricating" in p for p in seen["problems"])


def test_a_tau_that_disagrees_with_the_ladder_is_refused(tmp_path):
    trajectories = _set(tmp_path)
    trajectories.close()
    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3, expect_taus=[0.0, 0.9, 0.4])
    assert any("records tau" in p for p in seen["problems"])


def test_trajectories_written_by_different_runs_are_detected(tmp_path):
    """The committed prefix must agree across files; disagreeing times mean two runs."""
    trajectories = _set(tmp_path)
    try:
        trajectories.write_frame(step=500, time_ps=1.0, state_to_walker=[0, 1, 2],
                                 configurations=_configurations([1, 2, 3]))
    finally:
        trajectories.close()
    with netCDF4.Dataset(str(tmp_path / amber.state_trajectory_name(1)), "a") as d:
        d.variables["time"][0] = 99.0
    seen = st.StateTrajectorySet.inspect(tmp_path, n_states=3, expect_frames=1)
    assert any("disagree about their committed frame times" in p for p in seen["problems"])


# --- the superseded bundled layout
# ------------------------------------------------------------------

def test_the_schema_is_v3_and_carries_no_bundled_coordinates(tmp_path):
    path = tmp_path / "exchange.nc"
    reporter = storage.ReplicaReporter.create(
        path, n_states=3, n_atoms=4, n_solute_atoms=1, has_box=True,
        identity={"n_states": 3}, metadata={})
    reporter.close()
    assert storage.SCHEMA_VERSION.endswith("/v3")
    with netCDF4.Dataset(str(path)) as d:
        assert "positions" not in d.variables, "v3 keeps coordinates in the state trajectories"
        assert "box" not in d.variables
        assert "frame_step" in d.variables and "last_frame" in d.variables


def test_a_superseded_schema_is_refused_with_its_reason():
    with pytest.raises(storage.StorageError, match="different layouts of different things"):
        storage.refuse_superseded_schema("md-templates-replica-exchange/v2", path="old.nc")
    storage.refuse_superseded_schema(storage.SCHEMA_VERSION)


def test_no_migration_from_the_bundled_layout_is_offered():
    """Reindexing walkers into states after the fact would need the mapping history replayed, and
    a mistake there would attribute a configuration to the wrong Hamiltonian."""
    reason = storage.SUPERSEDED_SCHEMAS["md-templates-replica-exchange/v2"]
    assert "no migration is offered" in reason
    assert "wrong Hamiltonian" in reason
