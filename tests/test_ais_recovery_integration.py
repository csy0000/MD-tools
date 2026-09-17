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
from md_tools.openmm.checkpoint import (BOUNDARIES, STREAM_BOUNDARIES, CheckpointError,
                                     FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT, POINTER_NAME,
                                     read_committed)
from md_tools.ais.schedule import switching_schedule
from md_tools.ais.two_state import TwoStateHamiltonian

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
        from openmm import Platform, unit

        # The System AND the integrator the runner handed us, not fresh ones.
        #
        # The System matters because `TwoStateHamiltonian` reads its collective-variable values
        # through the `CustomCVForce` of the System the Context was built from, and moves lambda
        # as a parameter of that Context: a Context built from a different System of the same size
        # would have no `ais_lambda` at all, or report end-state energies nothing ran under.
        #
        # The integrator matters because `run_one_path` seeds the one it constructs. Building a
        # replacement here left it at OpenMM's seed 0, which means "choose a random seed" -- so
        # two runs of the same path took different random streams, and no test could compare a
        # resumed path against an uninterrupted one.
        self._integrator = integrator
        # `_REAL_SIMULATION`, captured at import time: `openmm.app.Simulation` is monkeypatched
        # to this class for the duration of a test, so constructing it by name here would recurse.
        self._simulation = _REAL_SIMULATION(_FourAtomTopology(), system, self._integrator,
                                            Platform.getPlatformByName("Reference"))
        self._simulation.context.setPositions(
            numpy.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.5, 0.0], [0.3, 0.3, 0.4]])
            * unit.nanometer)
        self.system = system
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


def _end_state(scale: float):
    """Four particles, one NonbondedForce and one bond; `scale` weakens particles 0 and 1.

    The bond is IDENTICAL in both end states, so the mixed System carries a shared force as well
    as a mixed one -- both halves of `pair_plan` are on the path under test.
    """
    from openmm import HarmonicBondForce, NonbondedForce, System, unit

    system = System()
    for _ in range(ATOMS):
        system.addParticle(12.0 * unit.dalton)
    nonbonded = NonbondedForce()
    for index in range(ATOMS):
        factor = scale if index in PERTURBED_ATOMS else 1.0
        nonbonded.addParticle((0.5 if index % 2 == 0 else -0.5) * factor, 0.3, 0.6 * factor)
    system.addForce(nonbonded)
    bond = HarmonicBondForce()
    bond.addBond(2, 3, 0.5, 1000.0)
    system.addForce(bond)
    return system


def _hamiltonian():
    """The REAL `TwoStateHamiltonian`, over four particles whose two end states genuinely differ.

    This used to be a tau switcher, and before that a stub that recorded the tau and did nothing.
    A stub stopped being enough the moment the path loop began measuring the Hamiltonian: one that
    changes nothing produces zero work, identical end-state potentials, and every identity in this
    file then holds without meaning anything. V1 halves the charges and epsilons of two
    particles, so V1 - V0 is tens of kJ/mol at the source configuration and the work a crash has
    to preserve is a genuinely non-zero number.
    """
    return TwoStateHamiltonian(_end_state(1.0), _end_state(0.5))


#: Which of the four particles V1 changes.
PERTURBED_ATOMS = (0, 1)


def _run_path(out: Path, tmp_path: Path, schedule, **kwargs):
    """`run_one_path` exactly as `ais_main` calls it, for this harness's four particles."""
    import md_tools.openmm.checkpoint as _checkpoint

    _checkpoint._passed.clear()
    return ais_run.run_one_path(
        index=0, chosen=[7, 9], out=out, schedule=schedule, lambdas=schedule["lambdas"],
        hamiltonian=_hamiltonian(),
        simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                           "mdtraj_top": object(), "implicit": False,
                           "acceleration": _Acceleration()},
        dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                  "temperature_K": 300.0},
        beta=0.4, temperature=300.0,
        rank=0, fingerprint="fixed-fingerprint", log=lambda *a: None, **kwargs)


def _schedule(**overrides):
    arguments = dict(
        switching_steps=SWITCHING, parameter_update_interval_steps=UPDATE,
        observation_interval_steps=OBSERVE, timestep_fs=2.0, trajectory_interval_steps=FRAME,
        state_interval_steps=STATE, checkpoint_interval_steps=CHECKPOINT)
    arguments.update(overrides)
    return switching_schedule(**arguments)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A path directory, a schedule and the arguments `run_one_path` takes."""
    # `run_one_path` does `from openmm.app import Simulation` inside the function, so the name it
    # resolves is `openmm.app`'s -- patching the AIS module would do nothing at all.
    import openmm.app

    monkeypatch.setattr(openmm.app, "Simulation", _TinySimulation)
    # The integrator is left real: `run_one_path` only constructs one and hands it over, so a stub
    # would test less while looking like it tested the same thing.

    schedule = _schedule()

    out = tmp_path / "AIS"
    out.mkdir()

    def call(*, resume=False, fault=None, after=1):
        """`after=1` by default: let the boundary pass once, so a generation is always committed.

        A fault at the FIRST occurrence leaves nothing committed, which is a real case but not the
        interesting one -- the interesting one is resuming from a good generation while a newer,
        half-written one lies beside it. `after=0` reaches the first case.
        """
        previous = os.environ.get(FAULT_ENVIRONMENT)
        previous_after = os.environ.get(FAULT_AFTER_ENVIRONMENT)
        if fault:
            os.environ[FAULT_ENVIRONMENT] = fault
            os.environ[FAULT_AFTER_ENVIRONMENT] = str(after)
        else:
            os.environ.pop(FAULT_ENVIRONMENT, None)
            os.environ.pop(FAULT_AFTER_ENVIRONMENT, None)
        try:
            return _run_path(out, tmp_path, schedule, resume=resume)
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


#: The source configuration every path in this file starts from. Four separated particles, not
#: four at the origin: the system carries a real NonbondedForce now, and coincident particles put
#: a Lennard-Jones singularity in the potential -- every energy becomes `inf`, every work becomes
#: `nan`, and every identity in this file compares `nan` with `nan` and passes.
SOURCE_POSITIONS = numpy.array([[0.0, 0.0, 0.0],
                                [0.4, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.3, 0.3, 0.4]])


def _read_frame_source(monkeypatch):
    """`_read_source_frame` reads a real trajectory; the stub returns fixed coordinates."""
    monkeypatch.setattr(ais_run, "_read_source_frame",
                        lambda *a, **k: (SOURCE_POSITIONS.copy(), None))


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


# --- the accumulated work survives an interruption ----------------------------------------------
#
# The work is accumulated in the same loop that propagates and committed by the same transaction.
# That makes it subject to exactly the failure the transaction exists to prevent -- a resume that
# restores a Context from one generation and the accumulator from another -- and every row after
# it would still satisfy "cumulative is the running sum of incremental" while the whole column is
# offset by the work before the crash.
#
# MIGRATED from the tau-basis component accumulators. Components mode and the three-group
# decomposition are deleted with the single-topology AIS; the one accumulator left is the total,
# and on the Reference platform a resumed path reproduces an uninterrupted one exactly, so the
# check is equality row for row rather than three sums.


def _final_row(out: Path):
    rows = _counts(out)["observations"]
    return rows[-1]


def _saved_positions(out: Path, frame_index: int):
    """The coordinate a frame-aligned row names, read back from the path's trajectory, in nm."""
    import mdtraj

    staged = out / "path_0000" / "frames.partial.nc"
    trajectory = staged if staged.is_file() else out / "AIS_traj0000.nc"
    with mdtraj.formats.NetCDFTrajectoryFile(str(trajectory)) as handle:
        xyz = handle.read()[0]
    return numpy.asarray(xyz[frame_index], dtype=float) / 10.0      # AMBER NetCDF is in angstrom


def _end_state_energy(system, positions_nm):
    from openmm import Context, Platform, VerletIntegrator, unit

    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setPositions(positions_nm * unit.nanometer)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_the_observation_potentials_are_the_end_states_at_the_saved_coordinate(harness):
    """Both identities, on every row that has a saved coordinate -- checked independently.

    `potential_direct_kj_mol` must be `(1 - lambda) V0 + lambda V1` of the row's own V0 and V1,
    and V0 and V1 must be what the two END-STATE Systems, in Contexts of their own, evaluate at the
    frame the row names. That second check reads the coordinate back from the trajectory, so a row
    carrying a neighbouring configuration's potentials fails it. A row with no saved coordinate
    carries empty potential cells by schema, never a neighbour's.

    MIGRATED from the three-group reconstruction of the single-topology AIS.
    """
    call, out, schedule = harness
    call()
    checked = 0
    for row in _counts(out)["observations"]:
        if row["coordinate_frame_index"] == "":
            for name in ("potential_v0_kj_mol", "potential_v1_kj_mol",
                         "potential_direct_kj_mol"):
                assert row[name] == "", (name, row["protocol_step"])
            continue
        lam = float(row["lambda"])
        v0, v1 = float(row["potential_v0_kj_mol"]), float(row["potential_v1_kj_mol"])
        mixed = (1.0 - lam) * v0 + lam * v1
        assert abs(mixed - float(row["potential_direct_kj_mol"])) < 1e-6, (
            f"step {row['protocol_step']}: mixed {mixed} against direct "
            f"{row['potential_direct_kj_mol']}")

        positions = _saved_positions(out, int(row["coordinate_frame_index"]))
        # The trajectory stores float32 angstroms, so the independent evaluation is at a
        # coordinate rounded to ~1e-5 A; a millijoule is far below any misattributed frame.
        for label, system, value in (("V0", _end_state(1.0), v0), ("V1", _end_state(0.5), v1)):
            independent = _end_state_energy(system, positions)
            assert abs(independent - value) < 1e-2, (
                f"step {row['protocol_step']}: {label} {value} in the row against {independent} "
                f"from the end-state System at frame {row['coordinate_frame_index']}")
        checked += 1
    assert checked >= 2, "no frame-aligned row was checked"


def test_the_work_and_the_end_state_difference_are_non_trivial_so_these_checks_can_fail(harness):
    """Guard on the guards: a V1 equal to V0 would satisfy every identity in this file."""
    call, out, schedule = harness
    record = call()
    row = _final_row(out)
    assert abs(float(row["cumulative_work_kj_mol"])) > 1e-3
    assert abs(record["total_work_kj_mol"]) > 1e-3
    assert abs(float(row["potential_v1_kj_mol"]) - float(row["potential_v0_kj_mol"])) > 1e-3


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_resumed_path_reproduces_the_uninterrupted_work(boundary, harness, tmp_path,
                                                          monkeypatch):
    """Crash, resume, and land on the same work as an uninterrupted run, row for row.

    Same seeds, same source frame, same schedule, and the Reference platform, where a restored
    committed generation continues bit-for-bit -- so the two paths are the same trajectory, and a
    resume that dropped or double-counted the work since the last observation would move the
    cumulative column from that row on. (On CUDA a resume is NOT bit-for-bit, by the decision of
    2026-09-16; that lane asserts exact restoration and a complete valid path instead.)

    MIGRATED from the three component totals of the single-topology AIS.
    """
    call, out, schedule = harness
    reference = call()
    reference_rows = _counts(out)["observations"]

    # A second, identical path in a fresh directory, interrupted at this boundary.
    second = tmp_path / "AIS-again"
    second.mkdir()
    _read_frame_source(monkeypatch)

    monkeypatch.setenv(FAULT_ENVIRONMENT, boundary)
    monkeypatch.setenv(FAULT_AFTER_ENVIRONMENT, "1")
    with pytest.raises(RuntimeError):
        _run_path(second, tmp_path, schedule, resume=False)
    monkeypatch.delenv(FAULT_ENVIRONMENT)
    monkeypatch.delenv(FAULT_AFTER_ENVIRONMENT)

    resumed = _run_path(second, tmp_path, schedule, resume=True)
    assert resumed["status"] == "completed" and resumed["resumed"] is True
    rows = _counts(second)["observations"]

    assert len(rows) == len(reference_rows), boundary
    for mine, theirs in zip(rows, reference_rows):
        for name in ("lambda", "incremental_work_kj_mol", "cumulative_work_kj_mol",
                     "potential_v0_kj_mol", "potential_v1_kj_mol", "potential_direct_kj_mol"):
            assert mine[name] == theirs[name], (
                f"{boundary}: {name} at step {mine['protocol_step']} differs between the "
                f"uninterrupted ({theirs[name]}) and the resumed ({mine[name]}) path")
    assert resumed["total_work_kj_mol"] == reference["total_work_kj_mol"], boundary


def _commit_then_edit_the_checkpoint_schema(call, out, edit):
    with pytest.raises(RuntimeError):
        call(fault="after-pointer-replace")
    sidecar = Path(read_committed(out / "path_0000")["sidecar"])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    edit(document["state"])
    sidecar.write_text(json.dumps(document), encoding="utf-8")


def test_a_checkpoint_without_an_ais_schema_is_refused_on_resume(harness):
    """A checkpoint that does not say what it measured is refused rather than continued.

    MIGRATED from the refusal of a checkpoint with no component-decomposition schema.
    """
    call, out, schedule = harness
    _commit_then_edit_the_checkpoint_schema(call, out, lambda state: state.pop("ais_schema"))
    with pytest.raises(SystemExit, match="records no AIS schema"):
        call(resume=True)


def test_a_checkpoint_from_another_schema_version_is_refused_on_resume(harness):
    call, out, schedule = harness
    _commit_then_edit_the_checkpoint_schema(
        call, out, lambda state: state["ais_schema"].update(version=99))
    with pytest.raises(SystemExit, match="not one this build implements"):
        call(resume=True)


def test_a_single_topology_checkpoint_is_refused_by_name_on_resume(harness):
    """A generation the tau-switching AIS committed is not continued by a two-state switch.

    Its accumulated work was measured along a different Hamiltonian path; adding linear two-state
    work onto it describes nothing. The refusal names the retired AIS, and leaves the committed
    generation exactly where it was.
    """
    call, out, schedule = harness

    def single_topology(state):
        state.pop("ais_schema")
        state.pop("lambda", None)
        state["tau"] = 0.25
        state["work_measurement"] = "components"
        state["decomposition_schema"] = {"name": "rest2-lambda-basis", "version": 2}

    _commit_then_edit_the_checkpoint_schema(call, out, single_topology)
    committed = read_committed(out / "path_0000")
    before = {name: Path(committed[name]).read_bytes() for name in ("checkpoint", "sidecar")}
    rows_before = (out / "path_0000" / "observations.csv").read_bytes()

    with pytest.raises(SystemExit) as refusal:
        call(resume=True)
    message = str(refusal.value)
    assert "rest2-lambda-basis/v2" in message and "single-topology" in message, message
    assert {name: Path(committed[name]).read_bytes()
            for name in ("checkpoint", "sidecar")} == before
    assert (out / "path_0000" / "observations.csv").read_bytes() == rows_before, (
        "the refused resume truncated a stream before refusing")


def test_the_completion_record_carries_the_schema_and_the_evaluation_count(harness):
    call, out, schedule = harness
    record = call()
    assert record["ais_schema"] == {"name": "two-state-linear", "version": 1}
    assert "decomposition_schema" not in record and "work_measurement" not in record
    counters = record["evaluation_counters"]
    # One WORK derivative per update, at the frozen pre-switch coordinate. The observation
    # potentials are per FRAME-ALIGNED OBSERVATION ROW, which is a third cadence again.
    #
    # This harness switches every 10 steps, observes every 20 and writes a frame every 50, over
    # 200 steps. Frames land at 0, 50, 100, 150, 200; observations at every multiple of 20. Only
    # 0, 100 and 200 are both, so three rows carry potentials and two frames (50, 150) exist in
    # the trajectory with no observation row naming them. That is not a defect -- it is what
    # independent cadences mean -- and it is why the count is neither the frame count nor the
    # observation count.
    aligned = [step for step in range(0, schedule["switching_steps"] + 1,
                                      schedule["observation_interval_steps"])
               if step % schedule["trajectory_interval_steps"] == 0]
    assert len(aligned) == 3, aligned
    assert counters["work_derivative_evaluations"] == schedule["number_of_updates"]
    # V and dV/dlambda in one call, plus the collective-variable values that check them: two.
    assert counters["observation_potential_energy_evaluations"] == 2 * len(aligned)
    assert counters["useful_total_energy_evaluations"] == (
        counters["work_derivative_evaluations"]
        + counters["observation_potential_energy_evaluations"]
        + counters["other_useful_energy_evaluations"])
    assert counters["paid_total_energy_evaluations"] == (
        counters["useful_total_energy_evaluations"]
        + counters["known_discarded_energy_evaluations"])
    # An uninterrupted path knows its whole cost.
    assert counters["known_discarded_energy_evaluations"] == 0
    assert counters["discarded_is_complete"] is True
    assert counters["parameter_updates"] == schedule["number_of_updates"]


def test_an_observation_interval_wider_than_the_update_interval_sums_several_updates(
        harness, tmp_path, monkeypatch):
    """The two cadences are independent, and a row must carry the work of EVERY update it covers.

    This harness observes every 20 steps and updates every 10, so each row after the first spans
    two parameter changes. A `since` accumulator that was reset per update instead of per
    observation would halve these numbers and still satisfy the running-sum identity -- so the
    same path is run again observing at EVERY update, and each coarse row must equal the sum of
    the two fine rows it spans. Observation reads energies and draws no random numbers, so on the
    Reference platform the two runs are the same trajectory.

    MIGRATED: it used to check per-component running sums.
    """
    call, out, schedule = harness
    assert schedule["observation_interval_steps"] > schedule["parameter_update_interval_steps"], \
        "this test is vacuous unless the observation cadence is the wider one"
    updates_per_row = (schedule["observation_interval_steps"]
                       // schedule["parameter_update_interval_steps"])
    assert updates_per_row == 2

    call()
    coarse = _counts(out)["observations"]
    running = 0.0
    for row in coarse[1:]:
        running += float(row["incremental_work_kj_mol"])
        assert abs(running - float(row["cumulative_work_kj_mol"])) < 1e-9, row["protocol_step"]

    fine_out = tmp_path / "AIS-fine"
    fine_out.mkdir()
    _read_frame_source(monkeypatch)
    _run_path(fine_out, tmp_path, _schedule(observation_interval_steps=UPDATE), resume=False)
    fine = {int(row["protocol_step"]): float(row["incremental_work_kj_mol"])
            for row in _counts(fine_out)["observations"]}
    for row in coarse[1:]:
        step = int(row["protocol_step"])
        spanned = fine[step] + fine[step - UPDATE]
        assert abs(float(row["incremental_work_kj_mol"]) - spanned) < 1e-9, (
            f"step {step}: the coarse row carries {row['incremental_work_kj_mol']} and the two "
            f"updates it spans sum to {spanned}")
    assert abs(float(coarse[-1]["cumulative_work_kj_mol"]) - sum(fine.values())) < 1e-6


# --- the run-level identity of an output directory ----------------------------------------------
#
# A path's checkpoint fingerprint protects that PATH. Nothing protected the DIRECTORY. A second
# invocation into the same `-odir` with a different source, schedule, seed or path count would
# skip the completed paths, run the rest under the new settings, and assemble one work table out
# of two different experiments -- a table that reads perfectly and describes neither.

def test_a_completed_path_is_verified_before_it_is_skipped(harness):
    """The record is only believed once the files it describes still hash to what it recorded."""
    call, out, schedule = harness
    first = call()
    assert first["status"] == "completed"
    assert "outputs" in first and "trajectory" in first["outputs"]

    # Unchanged: it is skipped, and the record comes back.
    again = call()
    assert again["status"] == "completed"
    assert again["total_work_kj_mol"] == first["total_work_kj_mol"]


def test_a_completed_path_whose_trajectory_changed_is_refused_rather_than_skipped(harness):
    """Truncated by a full disk, or rewritten by a second process. It reads as finished."""
    call, out, schedule = harness
    call()
    published = out / "AIS_traj0000.nc"
    published.write_bytes(published.read_bytes()[:-64])
    with pytest.raises(SystemExit, match="has changed since the path finished"):
        call()


def test_a_completed_path_whose_output_is_gone_is_refused(harness):
    call, out, schedule = harness
    call()
    (out / "AIS_traj0000.nc").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        call()


def test_a_completion_record_from_before_the_manifest_existed_is_refused(harness):
    """An old `completed.json` carries none of the fields a verification needs."""
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    marker.write_text(json.dumps({"status": "completed", "path_index": 0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="carries no"):
        call()


def test_a_completion_record_from_another_run_is_refused(harness):
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["fingerprint"] = "a different run entirely"
    marker.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="a path from a different run"):
        call()


def test_a_completion_record_written_under_another_schedule_is_refused(harness):
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["observations"] = int(record["observations"]) + 1
    marker.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="different schedule"):
        call()


def test_the_completion_manifest_lands_atomically(harness):
    """Written through a temporary: a half-written marker is read as a finished path."""
    call, out, schedule = harness
    call()
    assert (out / "path_0000" / "completed.json").is_file()
    leftovers = list((out / "path_0000").glob("*.partial"))
    assert not leftovers, f"a staging file survived the commit: {leftovers}"


def test_a_single_topology_completion_record_is_refused_by_name(harness):
    """A path the tau-switching AIS completed is not skipped as though this build had run it.

    Skipping it would put a work value measured along a different Hamiltonian path into this
    run's work table, where nothing distinguishes it from its neighbours.
    """
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record.pop("ais_schema")
    record["work_measurement"] = "components"
    record["decomposition_schema"] = {"name": "rest2-lambda-basis", "version": 2}
    marker.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        call()
    message = str(refusal.value)
    assert "rest2-lambda-basis/v2" in message and "single-topology" in message, message


def _identity(*, seed=3, paths=2, where=None, v1="u", schedule=None):
    from md_tools.ais.run import run_identity_document

    return run_identity_document(
        fingerprint="f",
        end_state_facts={"V0": {"system": "s", "topology": "t"},
                         "V1": {"system": v1, "topology": "t"}},
        source_facts={"sha256": "x"}, source_format="dcd",
        schedule=schedule or {"switching_steps": 200},
        ais={"number_of_paths": paths}, dynamics={"seed": seed}, chosen=list(range(paths)),
        reporting={"crd_printout_solute": 5}, resolved_config=where)


def test_the_run_identity_document_names_every_field_that_may_not_change():
    """A unit check on the document, so the fields are asserted rather than merely produced."""
    from md_tools.ais.run import (OBSERVATION_COLUMNS, RUN_IDENTITY_REQUIRED_KEYS,
                                  RUN_IDENTITY_VERSION)

    document = _identity(
        schedule={"switching_steps": 200, "lambdas": [0.0, 1.0], "observations": [], "note": "n"},
        where="/somewhere/resolved.config")
    assert document["schema_version"] == RUN_IDENTITY_VERSION
    for field in ("end_states", "source", "lambda", "schedule", "reporting", "seed_policy",
                  "number_of_paths", "selected_frames", "observation_columns", "ais_schema"):
        assert field in document, field
    assert RUN_IDENTITY_REQUIRED_KEYS <= set(document)
    assert set(document["end_states"]) == {"V0", "V1"}
    assert document["lambda"] == {"start": 0.0, "end": 1.0, "interpolation": "linear"}
    assert document["observation_columns"] == list(OBSERVATION_COLUMNS)
    assert "tau" not in document and "decomposition_schema" not in document
    # The derived, per-invocation parts of the schedule are excluded: `lambdas` is a list of
    # floats derived from the update count, and comparing it would report a difference twice.
    assert "lambdas" not in document["schedule"] and "observations" not in document["schedule"]


def test_a_directory_holding_another_run_is_refused_by_naming_what_differs(tmp_path):
    from md_tools.ais.run import RUN_IDENTITY, require_same_run

    (tmp_path / RUN_IDENTITY).write_text(json.dumps(_identity(seed=3, paths=2)),
                                         encoding="utf-8")
    require_same_run(tmp_path, _identity(seed=3, paths=2))              # unchanged: accepted

    with pytest.raises(SystemExit) as refusal:
        require_same_run(tmp_path, _identity(seed=4, paths=2))
    assert "seed_policy" in str(refusal.value)

    with pytest.raises(SystemExit) as refusal:
        require_same_run(tmp_path, _identity(seed=3, paths=5))
    message = str(refusal.value)
    assert "number_of_paths" in message and "selected_frames" in message

    # A different V1 is a different transformation, not a different setting of the same one.
    with pytest.raises(SystemExit) as refusal:
        require_same_run(tmp_path, _identity(v1="another V1"))
    assert "end_states" in str(refusal.value)


def test_an_older_run_identity_schema_is_refused_rather_than_compared(tmp_path):
    from md_tools.ais.run import RUN_IDENTITY, require_same_run

    document = _identity()
    old = dict(document, schema_version=0)
    (tmp_path / RUN_IDENTITY).write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema"):
        require_same_run(tmp_path, document)


def test_a_single_topology_run_identity_is_refused_by_name(tmp_path):
    """Run-identity v1 with a `tau` block is what the tau-switching AIS wrote.

    Refused, and refused as THAT -- "schema v1" alone sends a person to diff two JSON files for a
    reason the build already knows.
    """
    from md_tools.ais.run import RUN_IDENTITY, require_same_run

    document = _identity()
    old = {key: value for key, value in document.items()
           if key not in ("end_states", "lambda", "ais_schema")}
    old.update(schema_version=1, tau={"start": 0.5, "end": 0.0, "interpolation": "linear"},
               topology={"name": "t.pdb", "sha256": "t"}, system={"sha256": "s"},
               decomposition_schema={"name": "rest2-lambda-basis", "version": 2})
    (tmp_path / RUN_IDENTITY).write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="single-topology"):
        require_same_run(tmp_path, document)


def test_a_resolved_config_path_alone_does_not_make_it_a_different_run(tmp_path):
    """Where the file lives is not a property of the experiment. Moving a project is allowed."""
    from md_tools.ais.run import RUN_IDENTITY, require_same_run

    (tmp_path / RUN_IDENTITY).write_text(json.dumps(_identity(where="/old/resolved.config")),
                                         encoding="utf-8")
    require_same_run(tmp_path, _identity(where="/new/resolved.config"))


# --- the cost accounting, and what cannot be known after a crash --------------------------------

def test_a_resumed_path_restores_its_useful_counters_rather_than_calling_them_discarded(
        harness, tmp_path, monkeypatch):
    """The accounting bug, stated as its own test.

    Committed counters were being ADDED to `discarded`. Those evaluations produced the committed
    work, observations, frames and state rows the resume continues from -- they are the definition
    of useful. Calling them discarded made a resumed path report its own work as waste, and gave
    it a different useful total from an identical uninterrupted path, which is exactly the
    comparison the counters exist to support.
    """
    call, out, schedule = harness
    reference = call()
    reference_counters = reference["evaluation_counters"]
    assert reference_counters["known_discarded_energy_evaluations"] == 0
    assert reference_counters["discarded_is_complete"] is True

    second = tmp_path / "AIS-resumed"
    second.mkdir()
    _read_frame_source(monkeypatch)

    def run(**kwargs):
        return _run_path(second, tmp_path, schedule, **kwargs)

    monkeypatch.setenv(FAULT_ENVIRONMENT, "after-pointer-replace")
    monkeypatch.setenv(FAULT_AFTER_ENVIRONMENT, "1")
    with pytest.raises(RuntimeError):
        run(resume=False)
    monkeypatch.delenv(FAULT_ENVIRONMENT)
    monkeypatch.delenv(FAULT_AFTER_ENVIRONMENT)

    resumed = run(resume=True)
    assert resumed["resumed"] is True
    counters = resumed["evaluation_counters"]

    # THE POINT: the same useful total as the uninterrupted path, component by component.
    for name in ("work_derivative_evaluations", "observation_potential_energy_evaluations",
                 "useful_total_energy_evaluations", "parameter_updates"):
        assert counters[name] == reference_counters[name], (
            f"{name}: resumed {counters[name]} against uninterrupted "
            f"{reference_counters[name]}")

    # And the cost that genuinely cannot be recovered is reported as unknown, not as zero and not
    # by relabelling committed work.
    assert counters["discarded_is_complete"] is False, (
        "a resumed path claimed to know its complete discarded cost; a committed checkpoint "
        "cannot know how much work happened after it before the crash")
    assert counters["known_discarded_energy_evaluations"] == 0
    assert counters["paid_total_energy_evaluations"] == (
        counters["useful_total_energy_evaluations"]
        + counters["known_discarded_energy_evaluations"])


def test_the_counter_identities_hold_on_every_completion_record(harness):
    call, out, schedule = harness
    record = call()
    counters = record["evaluation_counters"]
    assert counters["useful_total_energy_evaluations"] == sum(
        counters[name] for name in ("work_derivative_evaluations",
                                    "observation_potential_energy_evaluations",
                                    "other_useful_energy_evaluations"))
    assert counters["paid_total_energy_evaluations"] == (
        counters["useful_total_energy_evaluations"]
        + counters["known_discarded_energy_evaluations"])
    assert "relationships" in counters and len(counters["relationships"]) == 2


# --- finalization is crash-atomic ----------------------------------------------------------------
#
# Publishing a path is several steps -- fsync, digest, link to the stable name, write the manifest,
# commit it, drop the checkpoints -- and every gap between them is a state a crash can leave.
#
# The sharp one is between the stable trajectory appearing and the completion manifest committing.
# The old code did `os.replace(staged, published)` there, so a crash left a PUBLISHED trajectory,
# no completion record, and no staged file for the resume to continue from. That state was not
# recoverable: the stable name existing was, by itself, a dead end.

from md_tools.openmm.checkpoint import FINALIZE_BOUNDARIES  # noqa: E402


@pytest.mark.parametrize("boundary", FINALIZE_BOUNDARIES)
def test_a_crash_at_a_finalization_boundary_finishes_identically_on_the_next_invocation(
        boundary, harness, tmp_path, monkeypatch):
    """Crash at each of the ten, then run again: identical outputs, or a clear refusal.

    Never a half-published path, never a duplicated row or frame, and never a failure whose only
    cause is that the stable trajectory had already been renamed.
    """
    call, out, schedule = harness
    reference = call()
    reference_rows = _counts(out)["observations"]
    reference_frames = _counts(out)["frames"]

    second = tmp_path / f"final-{boundary}"
    second.mkdir()
    _read_frame_source(monkeypatch)

    def run(**kwargs):
        return _run_path(second, tmp_path, schedule, **kwargs)

    monkeypatch.setenv(FAULT_ENVIRONMENT, boundary)
    monkeypatch.setenv(FAULT_AFTER_ENVIRONMENT, "0")
    crashed = None
    try:
        run(resume=False)
    except RuntimeError:
        crashed = True
    monkeypatch.delenv(FAULT_ENVIRONMENT)
    monkeypatch.delenv(FAULT_AFTER_ENVIRONMENT)
    assert crashed, f"{boundary}: the injected fault did not stop finalization"

    marker = second / "path_0000" / "completed.json"
    published = second / "AIS_traj0000.nc"
    committed = marker.is_file()
    if not committed:
        # Not committed: whatever is at the stable name is debris, and the next invocation must
        # say so rather than adopt it.
        assert (second / "path_0000" / "frames.partial.nc").is_file(), (
            f"{boundary}: the staged generation was destroyed before the commit, so the "
            f"interruption is unrecoverable")

    finished = run(resume=True)
    assert finished["status"] == "completed", boundary
    assert marker.is_file() and published.is_file(), boundary

    rows = _counts(second)["observations"]
    assert len(rows) == len(reference_rows), f"{boundary}: {len(rows)} rows"
    steps = [int(r["protocol_step"]) for r in rows]
    assert steps == sorted(set(steps)), f"{boundary}: a step is duplicated"
    assert _counts(second)["frames"] == reference_frames, boundary
    assert abs(float(rows[-1]["cumulative_work_kj_mol"])
               - float(reference_rows[-1]["cumulative_work_kj_mol"])) < 1e-6, boundary
    # And the staged generation is gone once the path is durably complete.
    assert not (second / "path_0000" / "frames.partial.nc").is_file() or committed


def test_a_stable_trajectory_without_a_committed_manifest_is_discarded(harness):
    """THE state the old finalization could not recover from, constructed directly."""
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    published = out / "AIS_traj0000.nc"
    assert published.is_file()

    # A published trajectory with no committed manifest: exactly what a crash between the two
    # used to leave, and what the code then had no way to continue from.
    record = json.loads(marker.read_text(encoding="utf-8"))
    marker.unlink()
    (out / "path_0000" / "frames.partial.nc").write_bytes(published.read_bytes())

    lines = []
    ais_run._discard_orphan_publication(published, marker, lines.append)
    assert not published.exists(), "the orphaned publication was kept"
    assert any("interrupted finalization" in line for line in lines), lines
    del record


def test_a_completion_manifest_missing_a_mandatory_key_is_refused(harness):
    """The exact key set, not a superset and not a subset."""
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert set(record["outputs"]) >= {"trajectory", "observations", "final_state"}

    for dropped in ("trajectory", "observations", "final_state"):
        broken = json.loads(json.dumps(record))
        broken["outputs"].pop(dropped)
        marker.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(SystemExit, match="mandatory output"):
            call()
    marker.write_text(json.dumps(record), encoding="utf-8")


def test_a_completion_manifest_with_an_unknown_key_is_refused(harness):
    call, out, schedule = harness
    call()
    marker = out / "path_0000" / "completed.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["outputs"]["something_else"] = {"sha256": "0" * 64}
    marker.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not produce"):
        call()


# =================================================================================================
#  ais.work_measurement -- RETIRED
#
#  The single-topology AIS offered two ways to measure a path's work: `work` (U at both endpoints
#  of every update) and `components` (a three-point basis probe fitted to a quadratic in 1 - tau,
#  verified every `verify_every_updates`). Both, the component columns, and the refusal to resume
#  under the other mode are gone with the tau switch: the two-state potential is LINEAR in lambda,
#  so one derivative per update IS the work and there is nothing to decompose. The tests that lived
#  here -- both modes agreeing, their 2N / 3N+2 costs, a work path carrying no component columns, a
#  components path naming itself, the cross-mode resume refusal, and the verification dial -- are
#  OBSOLETE with `ais.work_measurement`, `ais.verify_every_updates` and
#  `md_tools.ais.decomposition`. What survives of their intent is below.
# =================================================================================================

def test_the_observation_table_has_one_column_set_and_no_tau(harness):
    """One schema, whatever the run: no mode-dependent columns, no tau, no component columns.

    A reweighting script must fail on a key rather than read a column this build never measures,
    and the record must not claim a measurement mode that no longer exists.
    """
    call, out, _ = harness
    record = call()
    with (out / "path_0000" / "observations.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == ais_run.OBSERVATION_COLUMNS
    assert "tau" not in header
    assert not [name for name in header if "scaled" in name or "reconstructed" in name]
    assert "work_measurement" not in record
