"""A REST2 ladder writes one CV series per THERMODYNAMIC STATE, with the walker recorded.

WHY PER STATE

    The same reason the trajectories are. A ladder's result is a property of a rung -- "the
    distribution at tau = 0.3" -- and a walker visits many rungs, so a per-walker series is a
    series over a changing Hamiltonian and is not an ensemble average of anything.

    `cv_state2.csv` therefore holds whatever configuration OCCUPIED state 2 at each step, and
    `walker_index` says which walker supplied it. Getting that backwards produces files that look
    perfect and describe the wrong ensembles, which is what the tests here exist to exclude.

PLATFORM_POLICY_EXEMPTION: the ladder runs under `--cpu`. What is under test is which
configuration is written to which file and when -- bookkeeping, identical on every platform.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


from .conftest import ladder_group_file  # noqa: E402

@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("remd-cv")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": 3, "exchange_interval_steps": 10,
                  "number_of_exchanges": 4},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10},
        # Finer than the exchange interval, and dividing it exactly.
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
    }), encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4): `build-md` refuses to generate one until
    # `build/REST2/` exists, as `md-openmm build-top --rest2-scaler` writes it.
    from .conftest import make_states_for

    make_states_for(root, root / "REST2.config")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./REST2-run1", "--config", str(root / "REST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    # WRITTEN HERE, by the fixture that owns this directory. It used to be written by `completed`,
    # while other tests read `project / "user.config"` having requested only `project` -- an
    # unstated dependency on a sibling fixture's side effect. It survived a serial run, where
    # something always created it first. Under `-n 24` xdist splits a module across workers and
    # each builds only the fixtures ITS tests need, so a worker running one of those readers and
    # no `completed`-dependent test found no file: the run refused with "MD_TOOLS_CONFIG points at
    # ... which does not exist" instead of the refusal the test was written for, and which of the
    # two happened depended on how the module was distributed.
    (root / "user.config").write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(project / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((project / "build" / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    initial = project / "initial_state.xml"
    initial.write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")

    destination = tmp_path_factory.mktemp("remd-cv-run") / "run"
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base["MD_TOOLS_CONFIG"] = str(project / "user.config")
    done = subprocess.run(
        [sys.executable, str(project / "REST2-run1" / "REST2.py"),
         "-p", str(project / "build" / "built.pdb"),
         "--groupfile", str(ladder_group_file(project, destination)),
         "-odir", str(destination), "--cpu"],
        cwd=project / "REST2-run1", capture_output=True, text=True, timeout=1800, env=base)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return destination


def _rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return header, [dict(zip(header, line.split(","))) for line in lines[1:]]


def test_there_is_one_series_per_state_named_for_the_state(completed):
    found = sorted(p.name for p in completed.glob("cv_state*.csv"))
    assert found == ["cv_state0.csv", "cv_state1.csv", "cv_state2.csv"], found


def test_the_columns_are_the_documented_ones(completed):
    header, _rows_ = _rows(completed / "cv_state0.csv")
    assert header == ["step", "time_ps", "exchange_attempt", "state_index", "tau",
                      "walker_index", "exchange_phase", "trajectory_frame_index", "phi"]


def test_each_file_reports_its_own_fixed_state_and_tau(completed):
    """A state file must never report another state's index or another rung's tau."""
    taus = {}
    for index in range(3):
        _header, rows = _rows(completed / f"cv_state{index}.csv")
        assert {row["state_index"] for row in rows} == {str(index)}
        tau_values = {row["tau"] for row in rows}
        assert len(tau_values) == 1, f"state {index} reported several taus: {tau_values}"
        taus[index] = float(tau_values.pop())
    # A linear ladder, cold first. If a file were written against a walker rather than a state,
    # its tau would change as the walker moved.
    assert taus[0] == 0.0 and taus[0] < taus[1] < taus[2]


def test_the_cadence_is_finer_than_the_exchange_interval_and_lands_on_the_grid(completed):
    """5-step observations inside a 10-step exchange interval, 4 exchanges: steps 0..40.

    THIS TEST USED TO ASSERT `[5, 10, ..., 40]`, which codified a defect rather than a contract.
    The ladder observed only at schedule events, and `events_at` never returns anything at step 0,
    so the initial configuration -- the one every later row is a displacement from -- was silently
    absent from every state file. The universal contract is that step 0 and the final step each
    appear exactly once.
    """
    _header, rows = _rows(completed / "cv_state0.csv")
    steps = [int(row["step"]) for row in rows]
    assert steps == [0, 5, 10, 15, 20, 25, 30, 35, 40], steps
    assert steps.count(0) == 1 and steps.count(40) == 1
    assert len(steps) == len(set(steps)), "a step was observed twice"


def test_the_step_zero_row_is_the_initial_configuration_before_any_attempt(completed):
    """Step 0 precedes every exchange attempt, and says so rather than claiming attempt 0."""
    for index in range(3):
        _header, rows = _rows(completed / f"cv_state{index}.csv")
        first = rows[0]
        assert int(first["step"]) == 0
        assert int(first["exchange_attempt"]) == -1, (
            "step 0 happens before any attempt has been made; reporting 0 would claim the first "
            "attempt had already occurred")
        assert int(first["walker_index"]) == index, (
            "a fresh run starts with the identity state-to-walker mapping")
        assert first["trajectory_frame_index"] == "", (
            "no state trajectory frame is written at step 0, so the field must be empty")
        assert first["exchange_phase"] == "pre-exchange"


def test_every_state_file_has_the_same_steps(completed):
    """They are written together; a file that drifted would misalign every later row."""
    reference = [row["step"] for row in _rows(completed / "cv_state0.csv")[1]]
    for index in (1, 2):
        assert [row["step"] for row in _rows(completed / f"cv_state{index}.csv")[1]] == reference


def test_the_walker_index_is_a_permutation_of_the_walkers_at_every_step(completed):
    """THE state/walker claim.

    At any step each walker occupies exactly one state, so reading the walker_index column across
    the three files at a fixed step must give a permutation of {0, 1, 2}. A bug that wrote the
    state index into the walker column, or that failed to follow an accepted swap, breaks this.
    """
    per_state = {index: _rows(completed / f"cv_state{index}.csv")[1] for index in range(3)}
    for position in range(len(per_state[0])):
        walkers = sorted(int(per_state[index][position]["walker_index"]) for index in range(3))
        assert walkers == [0, 1, 2], (
            f"at step {per_state[0][position]['step']} the walkers were {walkers}")


def test_the_walkers_actually_move_between_states(completed):
    """Otherwise the permutation test above would pass on a ladder that never exchanged.

    A ladder with no accepted exchange is a legitimate outcome of a short run, but it would make
    every state/walker assertion here vacuous -- so this states plainly whether the run under test
    exercised the case, and skips rather than passing silently if it did not.
    """
    _header, rows = _rows(completed / "cv_state0.csv")
    seen = {int(row["walker_index"]) for row in rows}
    if len(seen) == 1:
        pytest.skip(f"no accepted exchange in this short run; state 0 held walker {seen} "
                    f"throughout, so the mapping was never exercised")
    assert len(seen) > 1


def test_every_row_is_marked_pre_exchange(completed):
    """One convention, everywhere, written into the file rather than left to be inferred."""
    for index in range(3):
        _header, rows = _rows(completed / f"cv_state{index}.csv")
        assert {row["exchange_phase"] for row in rows} == {"pre-exchange"}


def test_the_sidecar_states_the_convention_and_what_the_columns_mean(completed):
    body = json.loads((completed / "cv_state1.json").read_text(encoding="utf-8"))
    assert body["state_index"] == 1
    assert body["series_follows"] == "thermodynamic state"
    assert body["exchange_phase"] == "pre-exchange"
    assert "before any swap" in body["exchange_phase_meaning"]
    assert "which walker supplied" in body["walker_index_meaning"]
    assert body["units"] == "degrees"


def test_a_cv_interval_that_does_not_divide_the_exchange_interval_is_refused(project, tmp_path):
    """The grid has to be exact, or observations sit at different offsets in each interval."""
    configuration = yaml.safe_load((project / "REST2.config").read_text(encoding="utf-8"))
    configuration["collective_variables"]["interval_steps"] = 3      # 10 % 3 != 0
    (tmp_path / "bad.config").write_text(yaml.safe_dump(configuration), encoding="utf-8")

    # A DATASET ROOT for the throwaway generation. The subject here is the CV schedule -- a
    # configuration error -- and a ladder's rungs are now scaled from `build/built.xml` at build
    # time, so without one build-md refuses for the missing System and never reaches the check
    # this test is named for.
    from .conftest import make_dataset_root, make_states_for

    make_dataset_root(tmp_path)
    # ...and its saved scaled states, without which build-md refuses a ladder for their absence
    # (0.5.4) before reaching the CV schedule either.
    make_states_for(tmp_path, tmp_path / "bad.config")
    done = subprocess.run(
        CLI + ["build-md", "-odir", str(tmp_path / "bad-run1"),
               "--config", str(tmp_path / "bad.config")],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    message = done.stdout + done.stderr
    if done.returncode == 0:
        # Refused at run time instead of build time is acceptable; refused nowhere is not.
        # Every generated group line continues from `eq/eq_3.xml`, and the chain is not run here.
        from .conftest import write_starting_state

        write_starting_state(tmp_path, tmp_path / "bad-run1")
        base = dict(os.environ)
        base["PYTHONPATH"] = str(REPO / "src")
        base["MD_TOOLS_CONFIG"] = str(project / "user.config")
        done = subprocess.run(
            [sys.executable, str(tmp_path / "bad-run1" / "REST2.py"),
             "-p", str(tmp_path / "build" / "built.pdb"),
             # Into its own directory, as `run.sh` launches it: the group file's `-i _protocol.py`
             # is that directory's helper, and any other -odir is refused for it.
             "--groupfile", "remd_groupfile.1", "-odir", ".", "--cpu", "--check"],
            cwd=tmp_path / "bad-run1", capture_output=True, text=True, timeout=900, env=base)
        message = done.stdout + done.stderr
    assert done.returncode != 0, message[-3000:]
    assert "does not divide" in message, message[-3000:]
