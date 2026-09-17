"""ARCHIVED with rREST2 (0.5.4). Moved out of tests/test_cv_cuda_lanes.py unchanged; these depended on the
reservoir refresh and do not run against the current tree. See archive/rREST2/README.md.
"""


def test_rrest2_cv_on_cuda_holds_the_pre_refresh_configuration(ladder_project, tmp_path):
    """The scientific regression, on a device: the row must not hold the reservoir sample."""
    from md_tools.md.phase_space import PhaseSpaceReader
    from md_tools.remd import storage

    # Reuse the CPU file's reservoir builder: one definition of what a distinguishable
    # reservoir is, so the two lanes cannot drift apart about it.
    from tests.test_rrest2_pre_refresh_cv import _reservoir
    from md_tools.cv import torsion_degrees

    _reservoir(ladder_project)
    scripts = _generate(ladder_project, "rREST2",
                        _ladder_config(ladder_project, reservoir=True))
    destination = _ladder_directory(scripts, tmp_path, "rrest2")
    _run_ladder(scripts, destination, "rREST2")

    manifest = json.loads((destination / "restart.json").read_text(encoding="utf-8"))
    assert (manifest.get("execution") or {}).get("platform") == "CUDA", manifest.get("execution")

    reporter = storage.ReplicaReporter(destination / "rREST2.nc", mode="r")
    try:
        events = reporter.reservoir_events()
    finally:
        reporter.close()
    accepted = [(i, int(r[0]), int(r[1])) for i, r in enumerate(events)
                if int(r[3]) == 1 and int(r[0]) >= 0]
    assert accepted, "no reservoir refresh was accepted on CUDA"

    with PhaseSpaceReader(ladder_project / "reservoir.nc") as reader:
        samples = {frame: torsion_degrees(reader.frame(frame)[0], QUARTET)
                   for _e, _s, frame in accepted}

    checked = 0
    for exchange_index, state_index, frame in accepted:
        step = (exchange_index + 1) * 10
        rows = {int(r["step"]): r for r in _rows(destination / f"cv_state{state_index}.csv")}
        if step not in rows:
            continue
        reported = float(rows[step]["phi"])
        difference = abs((reported - samples[frame] + 180.0) % 360.0 - 180.0)
        assert difference > 1.0, (
            f"state {state_index} step {step} on CUDA reports {reported}, and the reservoir "
            f"sample has {samples[frame]}: the row holds the post-refresh coordinate")
        assert rows[step]["trajectory_frame_index"] == "", (
            "a refreshed state named a frame holding the reservoir sample")
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


def test_rrest2_cv_survives_two_interruptions_on_cuda(ladder_project, tmp_path):
    """rREST2 twice resumed on a device: a refresh REPLACES a walker mid-run.

    The one ladder path where a continuation has to reproduce not just its own dynamics but the
    reservoir draws that displaced them, and it had no resume coverage on CUDA at all.
    """
    from tests.test_rrest2_pre_refresh_cv import _reservoir

    _reservoir(ladder_project)
    scripts = _generate(ladder_project, "rREST2twice",
                        _ladder_config(ladder_project, reservoir=True))
    reference, resumed = _ladder_twice(scripts, "rREST2", tmp_path)
    _assert_ladder_matches(reference, resumed)

    from md_tools.remd import storage

    def _draws(directory):
        reporter = storage.ReplicaReporter(directory / "rREST2.nc", mode="r")
        try:
            return [(int(r[0]), int(r[1])) for r in reporter.reservoir_events()
                    if int(r[3]) == 1 and int(r[0]) >= 0]
        finally:
            reporter.close()

    want, got = _draws(reference), _draws(resumed)
    assert want, "no reservoir refresh was accepted, so nothing about refresh was exercised"
    assert got == want, (
        "the resumed run drew different reservoir frames: the refresh stream was not restored")
