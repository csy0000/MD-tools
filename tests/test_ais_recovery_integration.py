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
from md_tools.ais.decomposition import GROUPS

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
        # The System matters because `TauSwitcher.set_amplitude` pushes parameters through
        # `updateParametersInContext`, which writes through the Force objects the Context already
        # holds: a Context built from a different System of the same size would accept every call
        # and change nothing, and every component would read zero while the test reported success.
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


def _switcher():
    """The REAL `TauSwitcher`, over four particles carrying a real NonbondedForce.

    This used to be a stub that recorded the tau and did nothing. That was enough while the path
    loop only needed a System handed back -- and it stopped being enough the moment the loop began
    measuring the Hamiltonian, because a switcher that changes no parameters produces a potential
    that is zero at every amplitude, three identical probe energies, and components that are all
    zero. Every identity in this file would then hold, and none of them would mean anything.

    Two of the four particles are solute, so the three basis groups are all non-empty: the
    environment-environment pair is unscaled, the two cross pairs are linear, and the
    solute-solute pair is quadratic. The component accumulators a crash has to preserve are
    therefore genuinely different numbers rather than three zeros.
    """
    from openmm import NonbondedForce, System, unit

    from md_tools.rest2.scaler import TauSwitcher

    base = System()
    for _ in range(ATOMS):
        base.addParticle(12.0 * unit.dalton)
    nonbonded = NonbondedForce()
    for index in range(ATOMS):
        nonbonded.addParticle(0.5 if index % 2 == 0 else -0.5, 0.3, 0.6)
    base.addForce(nonbonded)
    return TauSwitcher(base, SOLUTE_ATOMS)


#: Which of the four particles are the enhanced region.
SOLUTE_ATOMS = (0, 1)


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

    def call(*, resume=False, fault=None, after=1, mode="components", verify_every=0):
        """`after=1` by default: let the boundary pass once, so a generation is always committed.

        A fault at the FIRST occurrence leaves nothing committed, which is a real case but not the
        interesting one -- the interesting one is resuming from a good generation while a newer,
        half-written one lies beside it. `after=0` reaches the first case.
        """
        import md_tools.openmm.checkpoint as _checkpoint

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
                switcher=_switcher(),
                simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                                   "mdtraj_top": object(), "implicit": False,
                                   "acceleration": _Acceleration()},
                dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                          "temperature_K": 300.0},
                ais={"work_measurement": mode, "verify_every_updates": verify_every,
                     "tau_start": 0.5, "tau_end": 0.0}, beta=0.4, temperature=300.0,
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


# --- the tau-basis component accumulators survive an interruption ------------------------------
#
# The components are accumulated in the same loop as the total and committed by the same
# transaction. That makes them subject to exactly the failure the transaction exists to prevent --
# a resume that restores a Context from one generation and accumulators from another -- with one
# extra way to go wrong: an accumulator that is simply not restored starts again from zero, and
# the resulting file still satisfies `dW_u + dW_l + dW_q == dW_total` on every INDIVIDUAL row
# while the cumulative columns are short by everything before the crash.

COMPONENT_TOTALS = ("total_work_non_scaled_kj_mol", "total_work_sqrt_scaled_kj_mol",
                    "total_work_lin_scaled_kj_mol")


def _final_row(out: Path):
    rows = _counts(out)["observations"]
    return rows[-1]


def test_the_components_sum_to_the_total_on_every_row_of_an_uninterrupted_path(harness):
    call, out, schedule = harness
    call()
    for row in _counts(out)["observations"]:
        parts = sum(float(row[name]) for name in
                    ("delta_work_non_scaled_kj_mol", "delta_work_sqrt_scaled_kj_mol",
                     "delta_work_lin_scaled_kj_mol"))
        assert abs(parts - float(row["incremental_work_kj_mol"])) < 1e-6, row["protocol_step"]
        cumulative = sum(float(row[name]) for name in COMPONENT_TOTALS)
        assert abs(cumulative - float(row["cumulative_work_kj_mol"])) < 1e-6, row["protocol_step"]


def test_the_unscaled_component_work_is_zero_on_every_row(harness):
    """The column a nonzero value would condemn. Written, so it can be read rather than assumed."""
    call, out, schedule = harness
    call()
    for row in _counts(out)["observations"]:
        assert float(row["delta_work_non_scaled_kj_mol"]) == 0.0, row["protocol_step"]
        assert float(row["total_work_non_scaled_kj_mol"]) == 0.0, row["protocol_step"]


def test_the_observation_potentials_reconstruct_and_match_a_direct_measurement(harness):
    """Both identities, on every row that has a saved coordinate.

    `potential_reconstructed_kj_mol` from the three groups, and `potential_direct_kj_mol` from a
    separate `getState(getEnergy=True)` at the same coordinate and tau. Two independent numbers in
    the file, so a reader checks the identity themselves instead of trusting that somebody did.
    """
    call, out, schedule = harness
    call()
    checked = 0
    for row in _counts(out)["observations"]:
        if row["coordinate_frame_index"] == "":
            # No saved coordinate: the potential cells are empty by schema, never a neighbour's.
            for name in ("potential_non_scaled_kj_mol", "potential_sqrt_scaled_kj_mol",
                         "potential_lin_scaled_kj_mol", "potential_reconstructed_kj_mol",
                         "potential_direct_kj_mol"):
                assert row[name] == "", (name, row["protocol_step"])
            continue
        tau = float(row["tau"])
        amplitude = 1.0 - tau
        reconstructed = (float(row["potential_non_scaled_kj_mol"])
                         + amplitude * float(row["potential_sqrt_scaled_kj_mol"])
                         + amplitude * amplitude * float(row["potential_lin_scaled_kj_mol"]))
        assert abs(reconstructed - float(row["potential_reconstructed_kj_mol"])) < 1e-6
        assert abs(reconstructed - float(row["potential_direct_kj_mol"])) < 1e-6, (
            f"step {row['protocol_step']}: reconstructed {reconstructed} against direct "
            f"{row['potential_direct_kj_mol']}")
        checked += 1
    assert checked >= 2, "no frame-aligned row was checked"


def test_the_components_are_non_trivial_so_these_checks_can_fail(harness):
    """Guard on the guards: three zeros would satisfy every identity above."""
    call, out, schedule = harness
    call()
    row = _final_row(out)
    assert abs(float(row["total_work_sqrt_scaled_kj_mol"])) > 1e-9
    assert abs(float(row["total_work_lin_scaled_kj_mol"])) > 1e-9
    assert abs(float(row["potential_lin_scaled_kj_mol"])) > 1e-9


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_resumed_path_reproduces_the_uninterrupted_component_totals(boundary, harness,
                                                                     tmp_path, monkeypatch):
    """Crash, resume, and land on the same three component totals as an uninterrupted run.

    Same seeds, same source frame, same schedule, so the paths are the same trajectory -- and a
    resume that dropped or double-counted a component would move exactly one of these three
    numbers while leaving the total right, because the total is measured independently.
    """
    call, out, schedule = harness
    reference = call()
    reference_row = _final_row(out)

    # A second, identical path in a fresh directory, interrupted at this boundary.
    second = tmp_path / "AIS-again"
    second.mkdir()
    _read_frame_source(monkeypatch)
    import md_tools.openmm.checkpoint as _checkpoint

    def run(**kwargs):
        _checkpoint._passed.clear()
        return ais_run.run_one_path(
            index=0, chosen=[7, 9], out=second, schedule=schedule, taus=schedule["taus"],
            switcher=_switcher(),
            simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                               "mdtraj_top": object(), "implicit": False,
                               "acceleration": _Acceleration()},
            dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                      "temperature_K": 300.0},
            ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0}, beta=0.4, temperature=300.0,
            rank=0, fingerprint="fixed-fingerprint", log=lambda *a: None, **kwargs)

    monkeypatch.setenv(FAULT_ENVIRONMENT, boundary)
    monkeypatch.setenv(FAULT_AFTER_ENVIRONMENT, "1")
    with pytest.raises(RuntimeError):
        run(resume=False)
    monkeypatch.delenv(FAULT_ENVIRONMENT)
    monkeypatch.delenv(FAULT_AFTER_ENVIRONMENT)

    resumed = run(resume=True)
    assert resumed["status"] == "completed" and resumed["resumed"] is True
    resumed_row = _final_row(second)

    for name in COMPONENT_TOTALS + ("cumulative_work_kj_mol",):
        assert abs(float(resumed_row[name]) - float(reference_row[name])) < 1e-6, (
            f"{boundary}: {name} differs between the uninterrupted and the resumed path")
    assert abs(sum(float(resumed_row[n]) for n in COMPONENT_TOTALS)
               - float(resumed_row["cumulative_work_kj_mol"])) < 1e-6


def test_a_checkpoint_without_a_decomposition_schema_is_refused_on_resume(harness):
    """An old checkpoint has no components to continue, and is refused rather than zero-filled."""
    call, out, schedule = harness
    with pytest.raises(RuntimeError):
        call(fault="after-pointer-replace")

    sidecar = Path(read_committed(out / "path_0000")["sidecar"])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["state"].pop("decomposition_schema")
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Exception, match="no component-decomposition schema"):
        call(resume=True)


def test_a_checkpoint_from_another_basis_version_is_refused_on_resume(harness):
    call, out, schedule = harness
    with pytest.raises(RuntimeError):
        call(fault="after-pointer-replace")

    sidecar = Path(read_committed(out / "path_0000")["sidecar"])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["state"]["decomposition_schema"]["version"] = 99
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Exception, match="not one this build implements"):
        call(resume=True)


def test_the_completion_record_carries_the_component_totals_and_the_evaluation_count(harness):
    call, out, schedule = harness
    record = call()
    assert record["decomposition_schema"]["version"] == 2
    counters = record["evaluation_counters"]
    # One WORK probe per update. The observation probes are per FRAME-ALIGNED OBSERVATION ROW,
    # which is a third cadence again -- and getting that arithmetic right is the correction.
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
    # Each probe is three evaluations; the WORK probe runs once per update, the OBSERVATION
    # probe once per frame-aligned row, and each observation adds one direct evaluation too.
    assert counters["work_basis_probe_energy_evaluations"] == 3 * schedule["number_of_updates"]
    assert counters["observation_potential_energy_evaluations"] == 4 * len(aligned)
    # TWO, for the whole path -- not two per update. The basis probe already gives U at both
    # endpoints, so `components` mode derives the work from the fit and measures it directly only
    # where it VERIFIES the fit: the first update of the path, and every `verify_every_updates`
    # after it. This harness leaves that at 0, so only the first update verifies, and the path
    # costs 3 evaluations per update instead of the 5 it used to.
    assert schedule.get("verify_every_updates", 0) == 0
    assert counters["direct_work_energy_evaluations"] == 2
    assert counters["useful_total_energy_evaluations"] == (
        counters["direct_work_energy_evaluations"]
        + counters["work_basis_probe_energy_evaluations"]
        + counters["observation_potential_energy_evaluations"]
        + counters["other_useful_energy_evaluations"])
    assert counters["paid_total_energy_evaluations"] == (
        counters["useful_total_energy_evaluations"]
        + counters["known_discarded_energy_evaluations"])
    # An uninterrupted path knows its whole cost.
    assert counters["known_discarded_energy_evaluations"] == 0
    assert counters["discarded_is_complete"] is True
    assert counters["parameter_updates"] > counters["work_basis_probe_energy_evaluations"]
    assert counters["probe_seconds"] > 0.0
    assert abs(sum(record[name] for name in COMPONENT_TOTALS)
               - record["total_work_kj_mol"]) < 1e-6


def test_an_observation_interval_wider_than_the_update_interval_sums_several_updates(harness):
    """The two cadences are independent, and a row must carry the work of EVERY update it covers.

    This harness observes every 20 steps and updates every 10, so each row after the first spans
    two parameter changes. A component accumulator that was reset per update instead of per
    observation would halve these numbers and still satisfy the per-row sum identity.
    """
    call, out, schedule = harness
    assert schedule["observation_interval_steps"] > schedule["parameter_update_interval_steps"], \
        "this test is vacuous unless the observation cadence is the wider one"
    updates_per_row = (schedule["observation_interval_steps"]
                       // schedule["parameter_update_interval_steps"])
    assert updates_per_row == 2

    call()
    rows = _counts(out)["observations"]
    # Reconstruct each row's linear work from the tau it spans. The linear component work over an
    # observation is sum_j (a_{j+1} - a_j) * U_linear(x_j), which is NOT (a_end - a_start) * U_l
    # unless the coordinates were frozen -- so the check here is the cheaper, sharper one: the
    # cumulative columns must be the running sum of the incremental ones, per component.
    running = {"non_scaled": 0.0, "sqrt_scaled": 0.0, "lin_scaled": 0.0}
    for row in rows[1:]:
        for name in running:
            running[name] += float(row[f"delta_work_{name}_kj_mol"])
            assert abs(running[name] - float(row[f"total_work_{name}_kj_mol"])) < 1e-9, (
                f"{name} at step {row['protocol_step']}")
    # And the last row's totals are the path's totals.
    assert abs(sum(running.values()) - float(rows[-1]["cumulative_work_kj_mol"])) < 1e-6


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


def test_the_run_identity_document_names_every_field_that_may_not_change():
    """A unit check on the document, so the fields are asserted rather than merely produced."""
    from md_tools.ais.run import RUN_IDENTITY_VERSION, run_identity_document

    document = run_identity_document(
        fingerprint="f", topology_facts={"sha256": "t"}, system_facts={"sha256": "s"},
        source_facts={"sha256": "x"}, source_format="dcd",
        schedule={"switching_steps": 200, "taus": [1, 2], "observations": [], "note": "n"},
        ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0, "number_of_paths": 2},
        dynamics={"seed": 3}, chosen=[7, 9],
        reporting={"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5},
        resolved_config="/somewhere/resolved.config")
    assert document["schema_version"] == RUN_IDENTITY_VERSION
    for field in ("source", "tau", "schedule", "reporting", "seed_policy", "number_of_paths",
                  "selected_frames", "observation_columns", "decomposition_schema"):
        assert field in document, field
    # The derived, per-invocation parts of the schedule are excluded: `taus` is a list of floats
    # derived from tau_start/tau_end/updates, and comparing it would report a difference twice.
    assert "taus" not in document["schedule"] and "observations" not in document["schedule"]


def test_a_directory_holding_another_run_is_refused_by_naming_what_differs(tmp_path):
    from md_tools.ais.run import RUN_IDENTITY, require_same_run, run_identity_document

    def document(seed, paths):
        return run_identity_document(
            fingerprint="f", topology_facts={"sha256": "t"}, system_facts={"sha256": "s"},
            source_facts={"sha256": "x"}, source_format="dcd",
            schedule={"switching_steps": 200},
            ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0, "number_of_paths": paths},
            dynamics={"seed": seed}, chosen=list(range(paths)),
            reporting={"crd_printout_solute": 5}, resolved_config=None)

    (tmp_path / RUN_IDENTITY).write_text(json.dumps(document(3, 2)), encoding="utf-8")
    require_same_run(tmp_path, document(3, 2))                    # unchanged: accepted

    with pytest.raises(SystemExit) as refusal:
        require_same_run(tmp_path, document(4, 2))
    assert "seed_policy" in str(refusal.value)

    with pytest.raises(SystemExit) as refusal:
        require_same_run(tmp_path, document(3, 5))
    message = str(refusal.value)
    assert "number_of_paths" in message and "selected_frames" in message


def test_an_older_run_identity_schema_is_refused_rather_than_compared(tmp_path):
    from md_tools.ais.run import RUN_IDENTITY, require_same_run, run_identity_document

    document = run_identity_document(
        fingerprint="f", topology_facts={"sha256": "t"}, system_facts={"sha256": "s"},
        source_facts={"sha256": "x"}, source_format="dcd", schedule={"switching_steps": 1},
        ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0, "number_of_paths": 1},
        dynamics={"seed": 1}, chosen=[0], reporting={}, resolved_config=None)
    old = dict(document, schema_version=0)
    (tmp_path / RUN_IDENTITY).write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema"):
        require_same_run(tmp_path, document)


def test_a_resolved_config_path_alone_does_not_make_it_a_different_run(tmp_path):
    """Where the file lives is not a property of the experiment. Moving a project is allowed."""
    from md_tools.ais.run import RUN_IDENTITY, require_same_run, run_identity_document

    def document(where):
        return run_identity_document(
            fingerprint="f", topology_facts={"sha256": "t"}, system_facts={"sha256": "s"},
            source_facts={"sha256": "x"}, source_format="dcd", schedule={"switching_steps": 1},
            ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0, "number_of_paths": 1},
            dynamics={"seed": 1}, chosen=[0], reporting={}, resolved_config=where)

    (tmp_path / RUN_IDENTITY).write_text(json.dumps(document("/old/resolved.config")),
                                         encoding="utf-8")
    require_same_run(tmp_path, document("/new/resolved.config"))


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
    import md_tools.openmm.checkpoint as _checkpoint

    def run(**kwargs):
        _checkpoint._passed.clear()
        return ais_run.run_one_path(
            index=0, chosen=[7, 9], out=second, schedule=schedule, taus=schedule["taus"],
            switcher=_switcher(),
            simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                               "mdtraj_top": object(), "implicit": False,
                               "acceleration": _Acceleration()},
            dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                      "temperature_K": 300.0},
            ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0}, beta=0.4, temperature=300.0,
            rank=0, fingerprint="fixed-fingerprint", log=lambda *a: None, **kwargs)

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
    for name in ("direct_work_energy_evaluations", "work_basis_probe_energy_evaluations",
                 "observation_potential_energy_evaluations", "useful_total_energy_evaluations"):
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
        counters[name] for name in ("direct_work_energy_evaluations",
                                    "work_basis_probe_energy_evaluations",
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
    import md_tools.openmm.checkpoint as _checkpoint

    def run(**kwargs):
        _checkpoint._passed.clear()
        return ais_run.run_one_path(
            index=0, chosen=[7, 9], out=second, schedule=schedule, taus=schedule["taus"],
            switcher=_switcher(),
            simulation_inputs={"topology": object(), "source_path": tmp_path / "source.dcd",
                               "mdtraj_top": object(), "implicit": False,
                               "acceleration": _Acceleration()},
            dynamics={"seed": 3, "friction_per_ps": 1.0, "timestep_fs": 2.0,
                      "temperature_K": 300.0},
            ais={"work_measurement": "components", "tau_start": 0.5, "tau_end": 0.0}, beta=0.4, temperature=300.0,
            rank=0, fingerprint="fixed-fingerprint", log=lambda *a: None, **kwargs)

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
#  ais.work_measurement -- the two ways a path may measure its work
#
#  `work` measures U at both endpoints of every update and subtracts: two evaluations, and the
#  work integral is all it produces. `components` takes the three-point basis probe, derives the
#  work from the fit, and additionally obtains the potential as a FUNCTION of tau -- which is what
#  reweighting onto an unvisited tau needs and a single work value cannot supply.
#
#  The two must agree on the work. That is the claim that makes the default safe, and it is the
#  first test below.
# =================================================================================================

def _work_rows(directory):
    with (directory / "path_0000" / "observations.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_both_modes_measure_the_same_work_on_the_same_path(harness):
    """The default may be cheaper, but it may not be a different measurement.

    Same seeds, same schedule, same source frame, so the two runs propagate the identical
    trajectory and differ only in how the work at each update is obtained -- subtraction of two
    measured potentials, or the quadratic fitted through three probe amplitudes. The three-group
    identity is exact, so these are two arithmetics for one number and they must agree to
    numerical precision rather than merely correlate.
    """
    call, out, _ = harness
    components = call(mode="components")
    (out / "path_0000").rename(out / "components_path")
    direct = call(mode="work")

    assert abs(components["total_work_kj_mol"] - direct["total_work_kj_mol"]) < 1e-6, (
        f"components {components['total_work_kj_mol']!r} vs work "
        f"{direct['total_work_kj_mol']!r}")

    # Not just the total: every row, so a pair of compensating per-update errors cannot pass.
    with (out / "components_path" / "observations.csv").open(newline="") as handle:
        component_rows = list(csv.DictReader(handle))
    for left, right in zip(component_rows, _work_rows(out)):
        assert abs(float(left["cumulative_work_kj_mol"])
                   - float(right["cumulative_work_kj_mol"])) < 1e-6


def test_a_work_path_costs_two_evaluations_per_update_and_a_component_path_three(harness):
    """The whole point of the setting, stated as arithmetic.

    `work` pays two direct evaluations per update and takes no probe at all. `components` pays
    three probe evaluations per update plus the two that verify the first update -- so it is
    3N + 2 against 2N, and it was 5N before the fit was trusted to supply the endpoints.
    """
    call, out, schedule = harness
    updates = schedule["number_of_updates"]

    components = call(mode="components")["evaluation_counters"]
    (out / "path_0000").rename(out / "components_path")
    direct = call(mode="work")["evaluation_counters"]

    assert direct["work_basis_probe_energy_evaluations"] == 0
    assert direct["direct_work_energy_evaluations"] == 2 * updates
    assert components["work_basis_probe_energy_evaluations"] == 3 * updates
    assert components["direct_work_energy_evaluations"] == 2

    # The observation rows differ too: a `work` observation is one potential, a `components`
    # observation is that potential plus a three-amplitude probe.
    assert components["observation_potential_energy_evaluations"] == (
        4 * direct["observation_potential_energy_evaluations"])


def test_a_work_path_carries_no_component_columns_at_all(harness):
    """Absent, not zero. A reweighting script must fail on a key, not read a fabricated nought."""
    call, out, _ = harness
    record = call(mode="work")

    header = set(_work_rows(out)[0])
    for group in GROUPS:
        assert f"delta_work_{group}_kj_mol" not in header
        assert f"total_work_{group}_kj_mol" not in header
        assert f"potential_{group}_kj_mol" not in header
    assert "potential_reconstructed_kj_mol" not in header
    # The one potential a direct run legitimately has: U at the coordinate the row names.
    assert "potential_direct_kj_mol" in header

    # And the record says which mode produced it, so nothing has to be inferred from the columns.
    assert record["work_measurement"] == "work"
    assert "decomposition_schema" not in record
    assert "total_work_non_scaled_kj_mol" not in record


def test_a_components_path_still_says_so_on_its_record(harness):
    call, _, _ = harness
    record = call(mode="components")
    assert record["work_measurement"] == "components"
    assert record["decomposition_schema"]["version"] == 2


def test_a_path_cannot_be_resumed_under_the_other_mode(harness):
    """Half a path measured each way is not a path. The resume is refused by name."""
    call, out, _ = harness
    with pytest.raises(RuntimeError):
        call(fault="after-checkpoint-sync", after=1)
    with pytest.raises(SystemExit) as refusal:
        call(resume=True, mode="work")
    message = str(refusal.value)
    assert "work_measurement" in message
    assert "'components'" in message and "'work'" in message


def test_verify_every_updates_buys_more_checks_and_costs_exactly_two_each(harness):
    """`verify_every_updates: N` is a dial on confidence with a stated price."""
    call, out, schedule = harness
    updates = schedule["number_of_updates"]

    once = call(mode="components")["evaluation_counters"]
    (out / "path_0000").rename(out / "once")
    often = call(mode="components", verify_every=5)["evaluation_counters"]

    assert once["direct_work_energy_evaluations"] == 2
    # Updates 0, 5, 10, 15 for a 20-update path: the first plus every fifth.
    expected = 2 * (1 + len([u for u in range(1, updates) if u % 5 == 0]))
    assert often["direct_work_energy_evaluations"] == expected
    # Verification costs evaluations and nothing else -- the probe count is untouched.
    assert often["work_basis_probe_energy_evaluations"] == \
        once["work_basis_probe_energy_evaluations"]

