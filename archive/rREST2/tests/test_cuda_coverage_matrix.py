"""ARCHIVED with rREST2 (0.5.4). Moved out of tests/test_cuda_coverage_matrix.py unchanged; these depended on the
reservoir refresh and do not run against the current tree. See archive/rREST2/README.md.
"""


def test_multi_rank_rrest2_with_a_real_reservoir(built, hardware, tmp_path):
    """rREST2 under real `mpirun -n 2` on CUDA, against a reservoir produced at the top rung.

    The serial rREST2 lane lives in `test_rrest2_cuda_smoke.py`. This is the coordinated one:
    the reservoir refresh is a COLLECTIVE operation -- rank 0 draws the sample and every rank has
    to agree about which exchange it happened at -- and a refresh that works in one process says
    nothing about one that has to be agreed across four.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the multi-rank rREST2 lane is an unmet criterion")
    if _visible() < 2:
        pytest.fail(f"2 states need 2 devices; {_visible()} visible")

    work = tmp_path / "rrest2-mpi"
    work.mkdir()
    # A ladder's rungs are scaled and serialised at BUILD time, so `build-md` reads
    # `<system>/build/built.xml` -- and `-odir work/project` makes `work` the system root. The
    # three other ladder lanes in this file copy it in; this one was missed and refused before
    # writing anything.
    shutil.copytree(built / "build", work / "build")
    tau_max = 0.5

    # The reservoir: a fixed-tau run AT THE LADDER'S TOP RUNG, streaming complete phase space.
    # Anything else is a sample of a different distribution.
    (work / "hot.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"tau": tau_max, "seed": 11, "phase_space_printout": 10},
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "production_steps": 40},
        "reporting": {"crd_printout_solute": 10, "info_printout": 20,
                      "checkpoint_printout": 40}}, sort_keys=False), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "hot"),
                                 "--config", str(work / "hot.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    hot_run = work / "hotrun"
    _equilibrate_chain(work / "hot", built / "build" / "built.pdb", built / "build" / "built.xml", work, hot_run,
                       last="cMD")
    phase_space = sorted(hot_run.glob("*phase*"))
    assert phase_space, f"no phase-space file: {sorted(p.name for p in hot_run.iterdir())}"
    reservoir = phase_space[0]

    # A SYSTEM ROOT OF ITS OWN FOR THE LADDER.
    #
    # The reservoir source above runs at `tau = tau_max` and the ladder at the default `tau = 0`,
    # so the two resolve the SHARED `input/eq_*.in` differently -- and `input/` belongs to the
    # system, so the second generation was refused by name: "give it a different `<system>` root".
    # They are deliberately different experiments over one built system, which is exactly what the
    # refusal advises. The reservoir reaches the ladder by the absolute path it already carries,
    # so splitting the roots costs nothing.
    ladder_root = work / "ladder"
    shutil.copytree(built / "build", ladder_root / "build")

    (ladder_root / "rrest2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "dynamics": {"seed": 13},
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "production_steps": 40},
        "reporting": {"crd_printout_solute": 10, "info_printout": 20,
                      "checkpoint_printout": 40},
        "rest2": {"number_of_replicas": 2, "tau_max": tau_max,
                  "exchange_interval_steps": 20, "number_of_exchanges": 2},
        "reservoir": {"enabled": True, "path": str(reservoir),
                      "refresh_interval_exchanges": 1, "velocities": "inherit"}},
        sort_keys=False), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(ladder_root / "project"),
                                 "--config", str(ladder_root / "rrest2.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    run = work / "run"
    start = _equilibrate(ladder_root / "project", built / "build" / "built.pdb",
                         built / "build" / "built.xml", work, run, script="rREST2.py")
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(ladder_root / "project" / "rREST2.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"),
         "-c", str(start), "-odir", str(run), "-ng", "2"],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    trajectories = sorted(run.glob("whole_state*_prod1.nc"))
    assert len(trajectories) == 2, [p.name for p in trajectories]
    report = (run / "rREST2.out").read_text(encoding="utf-8")
    assert "platform           : CUDA" in report, report[:1500]
    assert "reservoir" in report.lower(), report[:2000]
    _record("test_multi_rank_rrest2_with_a_real_reservoir",
            feature="rREST2 implicit, 2 states, real mpirun -n 2, real reservoir refresh",
            precision="mixed", device="0, 1",
            detail=f"reservoir {reservoir.name}; {len(trajectories)} state trajectories")
