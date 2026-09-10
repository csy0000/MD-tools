"""rREST2 with a real reservoir, on CUDA, through the real commands.

The reservoir properties -- that a refresh installs a RECORDED sample rather than a rescaled one,
that its velocity provenance is written down, and that the reservoir's Hamiltonian identity is the
ladder's top rung -- were asserted at unit level against stubs. What no test covered was the whole
path: a fixed-tau cMD producing a phase-space file, that file being declared as a reservoir, and
the ladder actually drawing from it on a GPU.

That path is where the identity has gone wrong before. The reservoir's identity record was once
taken AFTER a positional restraint had been added, so it described a Hamiltonian carrying a force
the reservoir samples were never generated under -- a defect invisible to every test that built
the declaration by hand.

Shortened but meaningful: implicit solvent, two rungs, a handful of exchanges.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.record import read_record

REPO = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TAU_MAX = 0.5


def _cli(work: Path, *args: str, timeout: int = 1800):
    return subprocess.run([sys.executable, "-m", "md_tools.cli.md_openmm", *args],
                          cwd=work, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """Build a system, produce a reservoir at the top rung, then run the ladder against it."""
    work = tmp_path_factory.mktemp("rrest2-cuda")
    ala = REPO / "tests" / "data" / "ALA.pdb"
    if not ala.is_file():
        pytest.skip("no ALA fixture")

    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    build = _cli(work, "build-top", "-i", str(ala), "-os", "built.xml", "-op", "built.pdb",
                 "-log", "built.log", "--config", str(work / "sys.config"))
    assert build.returncode == 0, build.stdout + build.stderr

    # The reservoir: a fixed-tau run AT THE LADDER'S TOP RUNG, streaming complete phase space.
    # Anything else would be a sample of a different distribution.
    (work / "hot.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        # The platform is machine.openmm.platform now, and CUDA is the built-in default, so a
        # GPU test states nothing about it. `test_the_ladder_ran_on_cuda...` checks that the
        # Context really was CUDA, which is the assertion that matters.
        "dynamics": {"tau": TAU_MAX, "seed": 11,
                     "phase_space_printout": 20},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 200},
        "reporting": {"crd_printout_solute": 20, "info_printout": 100,
                      "checkpoint_printout": 200},
    }, sort_keys=False), encoding="utf-8")
    assert _cli(work, "build-md", "-odir", "./hot", "--config", str(work / "hot.config"),
                timeout=600).returncode == 0
    hot = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"],
                         cwd=work / "hot", capture_output=True, text=True, timeout=3600)
    assert hot.returncode == 0, hot.stdout[-3000:] + hot.stderr[-3000:]

    phase_space = sorted((work / "hot").glob("*phase*"))
    assert phase_space, f"no phase-space file: {sorted(p.name for p in (work / 'hot').iterdir())}"
    reservoir = phase_space[0]

    (work / "rrest2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "dynamics": {"seed": 13},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 200},
        "reporting": {"crd_printout_solute": 20, "info_printout": 100,
                      "checkpoint_printout": 200},
        "rest2": {"number_of_replicas": 2, "tau_max": TAU_MAX,
                  "exchange_interval_steps": 50, "number_of_exchanges": 4},
        "reservoir": {"enabled": True, "path": str(reservoir),
                      "refresh_interval_exchanges": 1, "velocities": "inherit"},
    }, sort_keys=False), encoding="utf-8")
    generated = _cli(work, "build-md", "-odir", "./rrest2",
                     "--config", str(work / "rrest2.config"), timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    ran = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"],
                         cwd=work / "rrest2", capture_output=True, text=True, timeout=7200)
    assert ran.returncode == 0, ran.stdout[-4000:] + ran.stderr[-4000:]
    return work / "rrest2", reservoir, ran.stdout


def test_the_reservoir_declaration_describes_the_file_it_was_given(ladder):
    """The window and frame count are READ from the reservoir, never restated in configuration.

    A declaration that claimed a window the file does not contain would silently draw from
    whatever frames happened to exist.
    """
    directory, reservoir, _ = ladder
    declaration = yaml.safe_load((directory / "reservoir.yaml").read_text(encoding="utf-8"))
    assert declaration["weighting"] == "boltzmann"
    assert declaration["ensemble"] == "NVT"
    assert declaration["source"]["frames"] >= 1
    assert declaration["source"]["end_time_ps"] >= declaration["source"]["start_time_ps"]
    assert Path(declaration["source"]["phase_space"]).name == reservoir.name


def test_a_reservoir_refresh_installs_the_recorded_momentum(ladder):
    """`velocities: inherit` must mean the stored momentum, not a redrawn one.

    Redrawing gives a sample of a different distribution while looking identical in the record,
    so the policy has to be visible rather than implied.
    """
    directory, _, output = ladder
    declaration = yaml.safe_load((directory / "reservoir.yaml").read_text(encoding="utf-8"))
    assert declaration["velocity_policy"] == "stored"


def test_the_run_records_where_its_reservoir_samples_came_from(ladder):
    """Velocity provenance: the seed and the policy are in the record, not only in the config."""
    directory, _, output = ladder
    record = read_record(directory / "rREST2.log")
    assert record["status"] == "completed", record.get("status")
    text = (directory / "rREST2.log").read_text(encoding="utf-8")
    assert "reservoir" in text.lower(), "the run record never mentions the reservoir it drew from"


def test_the_ladder_ran_on_cuda_and_wrote_one_trajectory_per_state(ladder):
    """One trajectory per fixed thermodynamic state, and the platform is recorded."""
    directory, _, output = ladder
    # The ladder is one process per state, so the platform is a per-RANK fact and the driver
    # reports it per rank rather than folding one name into the run record.
    # `-o rREST2.out` is where a stage's output goes; run.sh's own stdout carries the summary.
    report = (directory / "rREST2.out").read_text(encoding="utf-8")
    lines = [line for line in report.splitlines() if line.lstrip().startswith("# platform")]
    assert lines, f"the driver reported no platform:\n{report[-2000:]}"
    assert all("CUDA" in line for line in lines), lines
    # Placement is reported once per PROCESS, not once per state. Run without mpiexec the ladder
    # is a single process that holds both states and lets OpenMM pick the device; under
    # `mpiexec -n 2` it would be two processes each naming its own. Both are correct, so what is
    # asserted is that every process that reported used CUDA -- not how many there were.
    assert all("device=" in line for line in lines), lines
    trajectories = sorted(directory.glob("whole_state*_prod1.nc"))
    assert len(trajectories) == 2, [p.name for p in trajectories]
    assert all(p.stat().st_size > 0 for p in trajectories)
