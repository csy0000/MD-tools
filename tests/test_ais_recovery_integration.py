"""AIS interruption and resume, through the REAL path runner rather than the commit function.

`test_checkpoint_transaction.py` proves the commit is atomic. That is necessary and not
sufficient: what a resume actually has to get right is the agreement between FOUR things that are
written by different code at different moments --

    the trajectory frames in the staged NetCDF
    the work rows in observations.csv
    the state rows in system.csv
    the accumulated bookkeeping inside the committed checkpoint sidecar

-- and the failure this guards against is a resume that reads a committed checkpoint from step
2000 and appends to streams that already reached step 3000, because a crash landed between a frame
becoming durable and the checkpoint that describes it. Nothing errors. The path completes. The
work integral is the sum of a trajectory nobody ran.

So these drive `run_one_path` itself with a stubbed Simulation: no CUDA, no molecular dynamics, a
few hundred milliseconds. What is under test is the recovery bookkeeping, and that is pure
Python -- running it against a real integrator would test OpenMM instead.

PLATFORM_POLICY_EXEMPTION: recovery bookkeeping on four free particles, Reference platform. No
molecular dynamics and no scientific result -- what is under test is which rows and frames survive
a crash, which is pure Python. The real CUDA evidence for AIS is `test_md_run_mpi_gpu.py`, and the
task asks for this one to run without a GPU so CI can exercise every fault boundary.

**The committed counters are the authority.** Recovery never infers progress from whichever file
happens to be longest; it truncates every stream back to what the committed sidecar vouches for.
That is why a crash after a frame is durable but before the checkpoint is committed must DROP the
frame rather than keep it.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy
import pytest

from md_tools.ais import run as ais_run
from md_tools.ais.checkpoint import (BOUNDARIES, STREAM_BOUNDARIES, CheckpointError,
                                     FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT, POINTER_NAME,
                                     read_committed)
from md_tools.ais.schedule import switching_schedule

#: Deliberately different cadences, so a stream that resumed to the wrong count is distinguishable
#: from the others rather than hidden by a shared number.
SWITCHING, UPDATE, OBSERVE, FRAME, STATE, CHECKPOINT = 200, 10, 20, 50, 100, 50
ATOMS = 4


#: The genuine class, bound before any test replaces the name in `openmm.app`.
from openmm.app import Simulation as _REAL_SIMULATION                     # noqa: E402


# --- a real OpenMM Simulation, four free particles wide -------------------------------------

class _TinySimulation:
    """A genuine `openmm.app.Simulation` over four free particles on the Reference platform.

    Not a mock. Every object the path runner touches -- the Context, the State, the checkpoint --
    is the real OpenMM type, so `XmlSerializer.serialize`, `saveCheckpoint` and `loadCheckpoint`
    exercise the same code a production run does. What is small is the SYSTEM, not the fidelity:
    four particles integrate in microseconds, need no CUDA, and are enough to prove that recovery
    restores the Context and the bookkeeping to the same instant.

    A mock would have been faster to write and would have tested the mock.
    """

    def __init__(self, topology, system, integrator, platform=None, properties=None):
        from openmm import LangevinMiddleIntegrator, Platform, System, unit

        real = System()
        for _ in range(ATOMS):
            real.addParticle(12.0 * unit.dalton)
        self._integrator = LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        # `_REAL_SIMULATION`, captured at import time: `openmm.app.Simulation` is monkeypatched
        # to this class for the duration of a test, so constructing it by name here would recurse.
        self._simulation = _REAL_SIMULATION(_FourAtomTopology(), real, self._integrator,
                                            Platform.getPlatformByName("Reference"))
        self._simulation.context.setPositions(
            numpy.zeros((ATOMS, 3)) * unit.nanometer)
        self.system = real
        self.topology = _FourAtomTopology()

    # -- what `run_one_path` uses ----------------------------------------------------------
    @property
    def context(self):
        return self._simulation.context

    def step(self, n):
        self._simulation.step(int(n))

    def saveCheckpoint(self, path):
        self._simulation.saveCheckpoint(str(path))

    def loadCheckpoint(self, path):
        self._simulation.loadCheckpoint(str(path))


class _FourAtomTopology:
    """A topology of the right size. `DCDFile`/NetCDF writers ask only for the atom count."""

    def __new__(cls):
        from openmm.app import Element, Topology

        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue("STB", chain)
        for index in range(ATOMS):
            topology.addAtom(f"A{index}", Element.getBySymbol("C"), residue)
        return topology


class _Switcher:
    """`TauSwitcher` reduced to what the path loop asks of it.

    The scaling mathematics is not under test here and is covered elsewhere; what this must do is
    hand back a System and accept a tau, so the recovery path can be driven deterministically.
    """

    def __init__(self):
        self.tau = None
        from openmm import System, unit

        self._system = System()
        for _ in range(ATOMS):
            self._system.addParticle(12.0 * unit.dalton)

    def prepared_system(self, tau):
        self.tau = tau
        return self._system

    def set_tau(self, context, system, tau):
        self.tau = tau


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A path directory, a schedule and the arguments `run_one_path` takes."""
    # `run_one_path` does `from openmm.app import Simulation` inside the function, so the name it
    # resolves is `openmm.app`'s -- patching the AIS module would do nothing at all.
    import openmm.app

    monkeypatch.setattr(openmm.app, "Simulation", _TinySimulation)
    # The integrator is left real: `run_one_path` only constructs one and hands it over, so a stub
    # would test less while looking like it tested the same thing.

    schedule = switching_schedule(
        tau_start=0.5, tau_end=0.0, switching_steps=SWITCHING,
        parameter_update_interval_steps=UPDATE, observation_interval_steps=OBSERVE,
        timestep_fs=2.0, trajectory_interval_steps=FRAME, state_interval_steps=STATE,
        checkpoint_interval_steps=CHECKPOINT)

    out = tmp_path / "AIS"
    out.mkdir()

    def call(*, resume=False, fault=None, after=1):
        """`after=1` by default: let the boundary pass once, so a generation is always committed.

        A fault at the FIRST occurrence leaves nothing committed, which is a real case but not the
        interesting one -- the interesting one is resuming from a good generation while a newer,
        half-written one lies beside it. `after=0` reaches the first case.
        """
        import md_tools.ais.checkpoint as _checkpoint

        _checkpoint._passed.clear()
        previous = os.environ.get(FAULT_ENVIRONMENT)
        previous_after = os.environ.get(FAULT_AFTER_ENVIRONMENT)
        if fault:
            os.environ[FAULT_ENVIRONMENT] = fault
            os.environ[FAULT_AFTER_ENVIRONMENT] = str(after)
        else:
            os.environ.pop(FAULT_ENVIRONMENT, None)
            os.environ.pop(FAULT_AFTER_ENVIRONMENT, None)
        try:
            return ais_run.run_one_path(
                index=0, chosen=[7, 9], out=out, schedule=schedule, taus=schedule["taus"],
                switcher=_Switcher(),
                simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                                   "mdtraj_top": object(), "implicit": False,
                                   "acceleration": _Acceleration()},
                dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                          "temperature_K": 300.0},
                ais={"tau_start": 0.5, "tau_end": 0.0}, beta=0.4, temperature=300.0,
                rank=0, resume=resume, fingerprint="fixed-fingerprint", log=lambda *a: None)
        finally:
            for key, value in ((FAULT_ENVIRONMENT, previous),
                               (FAULT_AFTER_ENVIRONMENT, previous_after)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return call, out, schedule


class _Acceleration:
    platform = None
    properties: dict = {}


def _read_frame_source(monkeypatch):
    """`_read_source_frame` reads a real trajectory; the stub returns fixed coordinates."""
    monkeypatch.setattr(ais_run, "_read_source_frame",
                        lambda *a, **k: (numpy.zeros((ATOMS, 3)), None))


@pytest.fixture(autouse=True)
def _stub_source(monkeypatch):
    _read_frame_source(monkeypatch)


def _counts(out: Path, path_id: int = 0):
    """What each stream actually holds, read independently of the checkpoint."""
    import mdtraj

    directory = out / f"path_{path_id:04d}"
    observations = list(csv.DictReader((directory / "observations.csv").open())) \
        if (directory / "observations.csv").is_file() else []
    states = list(csv.DictReader((directory / "system.csv").open())) \
        if (directory / "system.csv").is_file() else []

    staged = directory / "frames.partial.nc"
    published = out / "AIS_traj0000.nc"
    trajectory = staged if staged.is_file() else published
    frames = 0
    if trajectory.is_file():
        with mdtraj.formats.NetCDFTrajectoryFile(str(trajectory)) as handle:
            frames = len(handle)
    return {"work_rows": len(observations), "state_rows": len(states), "frames": frames,
            "observations": observations}


# --- an uninterrupted run, as the reference ------------------------------------------------------

def test_an_uninterrupted_path_writes_the_scheduled_counts(harness):
    call, out, schedule = harness
    record = call()

    assert record["status"] == "completed"
    counts = _counts(out)
    assert counts["work_rows"] == schedule["number_of_observations"] == 11
    assert counts["frames"] == schedule["number_of_frames"] == 5
    assert counts["state_rows"] == schedule["number_of_state_rows"] == 3
    assert [int(r["protocol_step"]) for r in counts["observations"]] == list(range(0, 201, 20))
    # A completed path keeps no transaction: leaving one invites a resume of finished work.
    assert not (out / "path_0000" / POINTER_NAME).exists()
    assert not (out / "path_0000" / "checkpoints").exists()


# --- a crash at every boundary the transaction exposes -------------------------------------------

@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_crash_at_a_checkpoint_boundary_resumes_from_the_last_committed_generation(
        boundary, harness):
    """Every stream is truncated to the COMMITTED counters, never to whichever file is longest.

    The frame case is the sharp one. Frames are flushed as they are written, so a crash after
    `after-checkpoint-write` leaves a trajectory that is longer than the checkpoint describing it.
    Keeping those extra frames would put the path's coordinates ahead of its work rows for the
    rest of the run, and every row after that would be attributed to the wrong configuration.
    """
    call, out, schedule = harness

    # The SECOND commit, so a fully committed generation always precedes the crash. The
    # first-commit case -- nothing committed, the path restarts from its source frame -- is
    # covered in `test_checkpoint_transaction.py`.
    with pytest.raises(RuntimeError):
        call(fault=boundary, after=1)

    committed = read_committed(out / "path_0000")
    assert committed is not None, f"{boundary}: no generation was committed before the crash"
    vouched = committed["state"]
    on_disk = _counts(out)
    assert on_disk["frames"] >= vouched["frames"], "frames went backwards before recovery"

    record = call(resume=True)
    assert record["status"] == "completed"
    assert record["resumed"] is True, boundary

    counts = _counts(out)
    assert counts["work_rows"] == schedule["number_of_observations"], boundary
    assert counts["frames"] == schedule["number_of_frames"], boundary
    assert counts["state_rows"] == schedule["number_of_state_rows"], boundary

    steps = [int(r["protocol_step"]) for r in counts["observations"]]
    assert steps == list(range(0, 201, 20)), f"{boundary}: {steps}"
    assert len(steps) == len(set(steps)), f"{boundary}: a step was written twice"

    running = 0.0
    for row in counts["observations"][1:]:
        running += float(row["incremental_work_kj_mol"])
        assert abs(running - float(row["cumulative_work_kj_mol"])) < 1e-9, boundary
    assert float(counts["observations"][0]["cumulative_work_kj_mol"]) == 0.0

    assert record["source_frame_index"] == 7, "the resumed path changed its source frame"
    assert record["trajectory"] == "AIS_traj0000.nc"
    assert not (out / "path_0000" / POINTER_NAME).exists()


# --- crashes between a stream write and its checkpoint -------------------------------------------

#: Which occurrence of each stream boundary to crash at, so the crash lands AFTER a checkpoint has
#: been committed. Frames fall at steps 0, 50 … 200 and checkpoints at 50 … 200, so crashing at the
#: first frame would leave nothing committed and test a different thing.
STREAM_OCCURRENCE = {"before-frame": 2, "after-frame": 2,
                     "before-work-row": 4, "after-work-row": 4,
                     "before-state-row": 1, "after-state-row": 1}


@pytest.mark.parametrize("boundary", STREAM_BOUNDARIES)
def test_a_crash_between_a_stream_write_and_its_checkpoint_drops_the_uncommitted_record(
        boundary, harness):
    """The case the committed counters exist for.

    A frame or a row that reached the disk after the last commit is NOT progress: nothing vouches
    for the Context that produced it. Recovery must drop it and rewrite it, and the completed path
    must be indistinguishable from one that never crashed.

    Not skipped when the timing is awkward -- the occurrence is CHOSEN so the crash lands after a
    committed generation. A skipped test here would be silence about the property it is named for.
    """
    call, out, schedule = harness

    with pytest.raises(RuntimeError):
        call(fault=boundary, after=STREAM_OCCURRENCE[boundary])

    committed = read_committed(out / "path_0000")
    assert committed is not None, (
        f"{boundary} crashed before any generation was committed; the chosen occurrence is wrong "
        f"and the test would prove nothing")
    vouched = committed["state"]

    # The streams may be AHEAD of the checkpoint at this moment. That is the whole point.
    on_disk = _counts(out)
    assert on_disk["work_rows"] >= vouched["work_rows"], boundary

    record = call(resume=True)
    assert record["status"] == "completed"
    assert record["resumed"] is True, boundary

    counts = _counts(out)
    assert counts["work_rows"] == schedule["number_of_observations"], boundary
    assert counts["frames"] == schedule["number_of_frames"], boundary
    assert counts["state_rows"] == schedule["number_of_state_rows"], boundary

    steps = [int(r["protocol_step"]) for r in counts["observations"]]
    assert steps == list(range(0, 201, 20)), f"{boundary}: {steps}"
    assert len(steps) == len(set(steps)), f"{boundary}: a step was written twice"

    running = 0.0
    for row in counts["observations"][1:]:
        running += float(row["incremental_work_kj_mol"])
        assert abs(running - float(row["cumulative_work_kj_mol"])) < 1e-9, boundary
    assert float(counts["observations"][0]["cumulative_work_kj_mol"]) == 0.0
    assert record["source_frame_index"] == 7, boundary


# --- what recovery must refuse -------------------------------------------------------------------

def test_an_uncommitted_newer_generation_is_ignored_by_the_runner(harness):
    """Debris from a crash is not a checkpoint, however new it is."""
    call, out, schedule = harness
    with pytest.raises(RuntimeError):
        call(fault="before-pointer-replace")

    committed = read_committed(out / "path_0000")
    if committed is None:
        pytest.skip("no generation was committed before the crash")
    generations = out / "path_0000" / "checkpoints"
    newer = generations / "generation_000099.chk"
    newer.write_text(json.dumps({"steps": 999999, "positions": [[9, 9, 9]] * ATOMS}),
                     encoding="utf-8")
    (generations / "generation_000099.json").write_text(
        json.dumps({"generation": 99, "checkpoint": newer.name,
                    "checkpoint_sha256": "0" * 64, "state": {"protocol_step": 99999}}),
        encoding="utf-8")

    record = call(resume=True)
    assert record["status"] == "completed"
    assert _counts(out)["work_rows"] == schedule["number_of_observations"]


def test_a_corrupt_committed_pair_is_refused_with_an_actionable_error(harness):
    call, out, schedule = harness
    with pytest.raises(RuntimeError):
        call(fault="after-pointer-replace")

    committed = read_committed(out / "path_0000")
    Path(committed["checkpoint"]).write_bytes(b"corrupted after commit")

    with pytest.raises(CheckpointError) as refusal:
        call(resume=True)
    message = str(refusal.value)
    assert "sha256" in message
    assert "delete the path directory" in message.lower(), "the refusal says nothing to do"
    assert "other paths are untouched" in message.lower(), (
        "the refusal does not say what survives, which is the first thing anyone asks")


def test_a_completed_path_is_skipped_and_cannot_be_resumed_as_incomplete(harness):
    """Its transaction is cleared, so there is nothing for a later `--resume` to pick up."""
    call, out, schedule = harness
    first = call()
    assert first["status"] == "completed"

    import hashlib

    published = out / "AIS_traj0000.nc"
    before = hashlib.sha256(published.read_bytes()).hexdigest()

    again = call(resume=True)
    assert again["status"] == "completed"
    assert hashlib.sha256(published.read_bytes()).hexdigest() == before, \
        "a completed path was rewritten by a resume"


def test_a_checkpoint_from_a_different_run_is_refused(harness, tmp_path):
    """The fingerprint is checked before anything loads: right-looking numbers, wrong simulation."""
    call, out, schedule = harness
    with pytest.raises(RuntimeError):
        call(fault="after-pointer-replace")

    sidecar = Path(read_committed(out / "path_0000")["sidecar"])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["state"]["fingerprint"] = "a different run entirely"
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    # The digest covers the checkpoint, not the sidecar, so this edit is the case a fingerprint
    # check exists for rather than one the digest would catch.
    with pytest.raises(SystemExit, match="does not belong to this run"):
        call(resume=True)
