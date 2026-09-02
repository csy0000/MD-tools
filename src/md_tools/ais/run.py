"""Annealed importance sampling: non-equilibrium switching paths from an equilibrium ensemble.

Driven by the `AIS.py` that `md-openmm build-md --config .../AIS.config` generates. The physics is
the implementation already validated on this branch -- `TauSwitcher`, and the same REST2
decomposition a fixed-tau walker or a ladder rung uses -- reached from the installed package
instead of from a copied-in script.

WHAT A PATH IS

The Hamiltonian is annealed from `tau_start` to `tau_end` while the coordinates propagate. The
temperature never changes: this is Hamiltonian switching, not temperature annealing. tau is the one
public, persisted protocol coordinate; the solute-solute and solute-environment scale factors are
derived from it inside the scaler and are deliberately not written as a second coordinate a reader
could take as authoritative.

THE WORK CONVENTION, STATED ONCE

    delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)

The parameters move FIRST, at frozen coordinates; the configuration then propagates under the new
Hamiltonian. Observation 0 is the source configuration under the source Hamiltonian, before any
parameter change and before any propagation, so its cumulative work is exactly zero by definition
rather than nearly zero.

FIXED VOLUME. No barostat is active during switching and no pressure-volume term enters the work,
even when the source ensemble was NPT. Each path keeps the box of the frame it started from.

Paths are independent: each has its own directory, its own deterministic seeds derived from the run
seed and the path index, and its own completion record. A completed path is skipped, never appended
to; an interrupted one is rerun from its source frame, because a switching path has no meaningful
mid-path restart -- the work integral is only defined along a whole path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from ..build.record import LogWriter, file_facts, openmm_platform_facts, read_record
from ..openmm.trajectory import check_trajectory_declaration

#: One row per observation. `s` and sqrt(s) are absent on purpose: see the module docstring.
OBSERVATION_COLUMNS = (
    "path_index", "observation_index", "coordinate_frame_index",
    "source_frame_index", "protocol_step", "switching_time_ps", "tau",
    "incremental_work_kj_mol", "cumulative_work_kj_mol", "cumulative_reduced_work",
    "temperature_kelvin", "integrator_seed", "velocity_seed",
)

COMPLETION_NAME = "completed.json"
OBSERVATIONS_CSV = "observations.csv"

#: The per-path thermodynamic state table, at `reporting.system_printout`. A different question
#: from the work: how the path is BEHAVING while the Hamiltonian moves, which is what tells you a
#: switch is too fast long before the work distribution does.
STATE_CSV = "system.csv"
STATE_COLUMNS = ("path_index", "protocol_step", "switching_time_ps", "tau",
                 "potential_energy_kj_mol", "kinetic_energy_kj_mol", "total_energy_kj_mol",
                 "temperature_kelvin", "volume_nm3", "density_g_per_ml")

#: Mid-path resume lives in `md_tools.ais.checkpoint`: a generation-based transaction whose
#: committed pointer is replaced last, so a crash never pairs a new Context with old bookkeeping.
#: `RESUME_SIDECAR` and `PATH_CHECKPOINT` were the two files of the previous, non-atomic design.
#: Frames are staged inside the path directory and published to the run root only when the path is
#: complete and validated. A half-written `AIS_trajNNNN.nc` at the root would look exactly like a
#: finished path to anyone globbing the directory.
STAGED_TRAJECTORY = "frames.partial.nc"

#: The global work table rank 0 writes once every path this run owns has finished.
#:
#: One row per (path, switching step), keyed by exactly that pair and sorted by it, so the table
#: is the same file whatever the worker count was and whichever rank produced which row -- see
#: `paths_for_rank`. It is assembled from the per-path `observations.csv` files rather than
#: gathered over MPI: a rank that died leaves its finished paths on disk, so the table describes
#: exactly what was measured, and a resumed run rebuilds it from the same records.
WORK_TABLE = "AIS_work.csv"
WORK_COLUMNS = ("path_id", "source_frame", "switch_step", "tau_before", "tau_after",
                "delta_work_kj_mol", "total_work_kj_mol", "total_reduced_work",
                "trajectory", "mpi_rank")

#: One row per path: the summary a reader wants when the question is about the work DISTRIBUTION
#: rather than about any individual path's trajectory through it.
WORK_SUMMARY = "AIS_paths.csv"
SUMMARY_COLUMNS = ("path_index", "source_frame_index", "observations",
                   "total_work_kj_mol", "total_reduced_work", "trajectory",
                   "integrator_seed", "velocity_seed", "mpi_rank")


def ais_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates (built.pdb)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System (built.xml)")
    parser.add_argument("-source-traj", "-src", "--source", dest="source", default=None,
                        metavar="TRAJ",
                        help="the equilibrium source trajectory these paths are drawn from; "
                             "overrides ais_source.trajectory. AIS consumes an ensemble you have "
                             "already produced -- it does not generate one")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="where the path directories are written (default: here)")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="the provenance record (machine-readable). Defaults to AIS.log")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable simulation output, Amber's mdout. A DIFFERENT file "
                             "from -log. Defaults to AIS.out")
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help="how many workers share the paths. Checked against the MPI "
                             "communicator; it never changes how many paths there are")
    parser.add_argument("--resume", action="store_true",
                        help="continue interrupted paths from their checkpoints. A completed "
                             "path is skipped either way; this affects only paths that stopped "
                             "part-way")
    parser.add_argument("--cpu", action="store_true",
                        help="run this invocation on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform. That setting can also select "
                             "CPU machine-wide; this flag is the per-run override, "
                             "and the record distinguishes the two")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. An execution placement option; rejected with "
                             "--cpu, which has no device to place")
    parser.add_argument("--paths", default=None,
                        help="run only these path indices, e.g. 0,1,2 or 0-9")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs, source and schedule, then exit without switching")
    return parser


def _selected(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    wanted: list[int] = []
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = piece.split("-", 1)
            wanted.extend(range(int(first), int(last) + 1))
        else:
            wanted.append(int(piece))
    out = sorted({index for index in wanted if 0 <= index < total})
    if not out:
        raise SystemExit(f"--paths {spec!r} selected no path in range 0..{total - 1}")
    return out


def choose_frames(*, eligible: list[int], count: int, selection: str, allow_repeats: bool,
                  seed: int) -> list[int]:
    """Which source frame each path starts from.

    Refuses rather than quietly reusing frames: two paths from the same configuration are not two
    independent realisations, and treating them as such understates the spread of the work
    distribution the whole method exists to measure.
    """
    import random

    if not eligible:
        raise SystemExit("the source window contains no eligible frames")
    if selection == "evenly_spaced":
        if count > len(eligible) and not allow_repeats:
            raise SystemExit(
                f"{count} paths were requested from {len(eligible)} eligible frames with "
                f"evenly_spaced selection. Widen the window, ask for fewer paths, or set "
                f"ais_source.allow_repeated_frames.")
        stride = max(1, len(eligible) // max(1, count))
        chosen = [eligible[min(index * stride, len(eligible) - 1)] for index in range(count)]
        return chosen
    generator = random.Random(seed)
    if allow_repeats:
        return [generator.choice(eligible) for _ in range(count)]
    if count > len(eligible):
        raise SystemExit(
            f"{count} independent paths were requested but the source window holds only "
            f"{len(eligible)} eligible frame(s). Two paths from one configuration are not two "
            f"independent realisations. Widen ais_source.first_frame/last_frame, ask for fewer "
            f"paths, or set ais_source.allow_repeated_frames if repeats are genuinely intended.")
    return sorted(generator.sample(eligible, count))


def _source_atom_count(path: Path) -> int | None:
    """How many atoms the trajectory FILE says it holds, or None when its format does not say.

    Deliberately reads the file's own header rather than trusting the topology it will be paired
    with: the whole point is to catch the case where those two disagree.
    """
    import mdtraj

    try:
        with mdtraj.open(str(path)) as handle:
            first = handle.read(1)
    except Exception:                                  # noqa: BLE001 - format cannot say; not fatal
        return None
    xyz = first[0] if isinstance(first, tuple) else first
    try:
        return int(xyz.shape[1])
    except (AttributeError, IndexError):
        return None


def write_work_table(out: Path, chosen: list[int]) -> dict[str, Any]:
    """Assemble the global work table and the per-path summary from what the paths wrote.

    Read from files rather than gathered over MPI, so an interrupted campaign still produces a
    table that describes exactly the paths that finished. A path with no completion record is
    absent: the table never invents a row for work that was not measured.
    """
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for path_id in range(len(chosen)):
        directory = out / f"path_{path_id:04d}"
        marker = directory / COMPLETION_NAME
        if not marker.is_file():
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        summary.append({name: record.get(name) for name in SUMMARY_COLUMNS})

        observations = directory / OBSERVATIONS_CSV
        if not observations.is_file():
            continue
        with observations.open(newline="") as handle:
            entries = list(csv.DictReader(handle))
        previous_tau = None
        for entry in entries:
            rows.append({
                "path_id": path_id,
                "source_frame": entry["source_frame_index"],
                "switch_step": int(entry["protocol_step"]),
                # `tau_before` on observation 0 is its own tau: the path has not moved yet, and
                # writing a blank there would make the first row the only one a reader has to
                # treat specially.
                "tau_before": previous_tau if previous_tau is not None else entry["tau"],
                "tau_after": entry["tau"],
                "delta_work_kj_mol": entry["incremental_work_kj_mol"],
                "total_work_kj_mol": entry["cumulative_work_kj_mol"],
                "total_reduced_work": entry["cumulative_reduced_work"],
                "trajectory": record.get("trajectory"),
                "mpi_rank": record.get("mpi_rank"),
            })
            previous_tau = entry["tau"]

    rows.sort(key=lambda row: (row["path_id"], row["switch_step"]))
    with (out / WORK_TABLE).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORK_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with (out / WORK_SUMMARY).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(summary)
    return {"rows": len(rows), "paths": len(summary), "requested": len(chosen)}


def write_selected_frames(path: Path, chosen: list[int], *, seed: int) -> None:
    """Which source frame each path starts from, and with which seeds. Written before dynamics."""
    from ..md._stages import derive_seed

    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_index", "source_frame_index", "integrator_seed", "velocity_seed"])
        for index, frame in enumerate(chosen):
            writer.writerow([index, frame,
                             derive_seed(seed, "ais", index, "integrator"),
                             derive_seed(seed, "ais", index, "velocity")])



# ---------------------------------------------------------------------------------------------
# One switching path
#
# Four cadences, a staged trajectory, and a checkpoint that restores the OpenMM Context and the
# bookkeeping OUTSIDE it together. Restoring one without the other resumes a simulation into
# somebody else's accounting, which is worse than restarting.
# ---------------------------------------------------------------------------------------------

def _truncate_csv(path: Path, keep: int, columns) -> None:
    """Cut a table back to `keep` data rows. A crash can leave a partial final line."""
    if not path.is_file():
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        return
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows[:keep])


def _open_staged_netcdf(path: Path, keep: int):
    """An open NetCDF writer positioned after exactly `keep` good frames.

    mdtraj's NetCDF writer offers 'r' and 'w' and no append mode, and that turns out to be the
    right constraint rather than an obstacle: a frame interrupted mid-write leaves a record that
    reads as present and is not complete, so appending onto an unverified tail would be wrong even
    where the format allowed it.

    So a resume REWRITES: the frames the sidecar vouches for are read back, written into a fresh
    file, and the handle is returned still open for the rest of the path. The count is bounded by
    `number_of_frames`, which is small by construction, and it happens once per interruption.
    """
    import os

    import mdtraj

    if not keep:
        return mdtraj.formats.NetCDFTrajectoryFile(str(path), "w")

    with mdtraj.formats.NetCDFTrajectoryFile(str(path)) as handle:
        available = len(handle)
        coordinates, time, lengths, angles = handle.read(min(keep, available))
    if available < keep:
        raise SystemExit(
            f"{path} holds {available} frame(s) but the checkpoint sidecar vouches for {keep}. "
            f"The staged trajectory and the record of it disagree, so neither can be trusted; "
            f"delete the path directory to rerun it from its source frame.")

    staged = path.with_name(path.name + ".rewrite")
    writer = mdtraj.formats.NetCDFTrajectoryFile(str(staged), "w")
    for frame in range(keep):
        writer.write(coordinates[frame],
                     time=None if time is None else time[frame],
                     cell_lengths=None if lengths is None else lengths[frame],
                     cell_angles=None if angles is None else angles[frame])
    writer.flush()
    # `os.replace` while the handle is open is safe on POSIX: the writer keeps writing to the same
    # inode, now reachable under the staged name.
    os.replace(staged, path)
    return writer


def _state_row(simulation, *, index, protocol_step, switching_time_ps, tau, implicit,
               temperature_unit) -> dict[str, Any]:
    """What the path is doing thermodynamically at this step."""
    from openmm import unit

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    # Degrees of freedom from the integrator's own view of the System, so constraints and any
    # centre-of-mass motion remover are accounted for rather than assumed away.
    dof = simulation.context.getIntegrator().computeSystemTemperature() \
        if hasattr(simulation.context.getIntegrator(), "computeSystemTemperature") else None
    if dof is not None:
        temperature_now = dof.value_in_unit(unit.kelvin)
    else:                                                 # older OpenMM: derive it
        particles = simulation.system.getNumParticles()
        constraints = simulation.system.getNumConstraints()
        free = max(3 * particles - constraints - 3, 1)
        gas = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        temperature_now = 2.0 * kinetic / (free * gas)

    volume = density = None
    if not implicit:
        box = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
        volume = box
        mass = sum(simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
                   for i in range(simulation.system.getNumParticles()))
        # g/mL from daltons per nm^3. One dalton is 1.66053906660e-24 g and one nm^3 is
        # 1e-21 mL, so the factor is their ratio: 1.66053906660e-3. Written out rather than
        # given as a bare constant, because the plausible wrong answer here is 1000x and water
        # at 947 g/mL looks like a number rather than like a mistake.
        density = mass / box * 1.66053906660e-3
    return {
        "path_index": index,
        "protocol_step": protocol_step,
        "switching_time_ps": switching_time_ps,
        "tau": tau,
        "potential_energy_kj_mol": potential,
        "kinetic_energy_kj_mol": kinetic,
        "total_energy_kj_mol": potential + kinetic,
        "temperature_kelvin": temperature_now,
        "volume_nm3": volume,
        "density_g_per_ml": density,
    }


def run_one_path(*, index: int, chosen: list[int], out: Path, schedule: dict[str, Any],
                 taus, switcher, simulation_inputs: dict[str, Any], dynamics: dict[str, Any],
                 ais: dict[str, Any], beta: float, temperature: float, rank: int,
                 resume: bool, fingerprint: str, log) -> dict[str, Any] | None:
    """Run (or finish) one switching path. Returns its completion record, or None if it failed."""
    import os

    import mdtraj
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import Simulation

    from ..md._stages import derive_seed
    from . import path_trajectory_name

    topology = simulation_inputs["topology"]
    source_path = simulation_inputs["source_path"]
    top = simulation_inputs["mdtraj_top"]
    implicit = simulation_inputs["implicit"]
    acceleration = simulation_inputs["acceleration"]

    directory = out / f"path_{index:04d}"
    marker = directory / COMPLETION_NAME
    trajectory_name = path_trajectory_name(index, len(chosen))
    published = out / trajectory_name

    if marker.is_file():
        log(f"  path {index:4d}: already completed; not rerun and never appended to")
        return json.loads(marker.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)

    frame = chosen[index]
    integrator_seed = derive_seed(int(dynamics["seed"]), "ais", index, "integrator")
    velocity_seed = derive_seed(int(dynamics["seed"]), "ais", index, "velocity")

    interval = schedule["parameter_update_interval_steps"]
    observe_every = schedule["observation_interval_steps"]
    frame_every = schedule["trajectory_interval_steps"]
    state_every = schedule["state_interval_steps"]
    checkpoint_every = schedule["checkpoint_interval_steps"]
    updates = schedule["number_of_updates"]

    staged = directory / STAGED_TRAJECTORY

    system = switcher.prepared_system(taus[0])
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(dynamics["friction_per_ps"]) / unit.picosecond,
        float(dynamics["timestep_fs"]) * unit.femtosecond)
    integrator.setRandomNumberSeed(int(integrator_seed))
    simulation = Simulation(topology, system, integrator,
                            acceleration.platform, acceleration.properties)

    # -- resume, or start ----------------------------------------------------------------------
    from .checkpoint import clear_committed, commit_generation, fault, read_committed

    state_of_path = None
    if resume:
        # ONLY the committed pointer. Never "the newest generation on disk": the newest file is
        # exactly what a crash leaves behind, and choosing it would pair a Context from step 3000
        # with bookkeeping from step 2000.
        committed = read_committed(directory)
        state_of_path = committed["state"] if committed else None

    # `--resume` over a path that never committed a generation is not an error: that path simply
    # starts from its source frame. Only a path that HAS a committed checkpoint is fingerprinted.
    if state_of_path is not None:
        # EVERY fingerprint before anything is loaded. A checkpoint carries positions and
        # velocities for one particular System, schedule and path; resuming it against another is
        # how a run continues with right-looking numbers and the wrong simulation.
        mismatched = [key for key, expected in
                      (("fingerprint", fingerprint), ("path_index", index),
                       ("source_frame_index", frame), ("integrator_seed", integrator_seed),
                       ("velocity_seed", velocity_seed), ("trajectory", trajectory_name))
                      if state_of_path.get(key) != expected]
        if mismatched:
            raise SystemExit(
                f"path {index}: the checkpoint in {directory} does not belong to this run "
                f"({', '.join(mismatched)} differ). Refusing to resume it: the numbers would look "
                f"right and describe a different simulation. Delete the path directory to rerun "
                f"it from its source frame.")

    if state_of_path is not None:
        simulation.loadCheckpoint(committed["checkpoint"])
        # The Context is back; now put the bookkeeping back to exactly the same instant. The
        # streams are cut to the counts the sidecar vouches for, so an interrupted final record
        # is dropped rather than appended to.
        updates_done = int(state_of_path["updates_completed"])
        cumulative = float(state_of_path["cumulative_work_kj_mol"])
        since = float(state_of_path["work_since_last_observation_kj_mol"])
        rows_emitted = int(state_of_path["work_rows"])
        frames_emitted = int(state_of_path["frames"])
        state_rows_emitted = int(state_of_path["state_rows"])
        switcher.set_tau(simulation.context, system, taus[updates_done])
        _truncate_csv(directory / OBSERVATIONS_CSV, rows_emitted, OBSERVATION_COLUMNS)
        _truncate_csv(directory / STATE_CSV, state_rows_emitted, STATE_COLUMNS)
        log(f"  path {index:4d}: resuming at step {updates_done * interval} of "
            f"{schedule['switching_steps']} ({rows_emitted} work row(s), {frames_emitted} "
            f"frame(s) kept)")
    else:
        positions, boxes = _read_source_frame(source_path, top, frame, implicit=implicit)
        system_box = boxes
        if system_box is not None:
            simulation.context.setPeriodicBoxVectors(*system_box)
        simulation.context.setPositions(positions)
        # A trajectory carries no velocities, so each path draws fresh Maxwell-Boltzmann momenta
        # at the run temperature with its own recorded seed.
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin,
                                                      int(velocity_seed))
        updates_done = 0
        cumulative = since = 0.0
        rows_emitted = frames_emitted = state_rows_emitted = 0
        for path, columns in ((directory / OBSERVATIONS_CSV, OBSERVATION_COLUMNS),
                              (directory / STATE_CSV, STATE_COLUMNS)):
            with path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        staged.unlink(missing_ok=True)

    # -- the streams ---------------------------------------------------------------------------
    netcdf = _open_staged_netcdf(staged, frames_emitted)

    def append(path: Path, columns, row: dict[str, Any]) -> None:
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writerow(row)

    def write_frame(protocol_step: int, switching_time_ps: float) -> None:
        nonlocal frames_emitted
        # A crash here, after the frame is durable and before the checkpoint that vouches for it,
        # is the case the committed counters exist for: the trajectory is then LONGER than the
        # bookkeeping, and keeping the extra frame would put the path's coordinates permanently
        # ahead of its work rows.
        fault("before-frame")
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        lengths = angles = None
        if not implicit:
            vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            # NOT unpacked as `a, b, c, alpha, beta, gamma`: `beta` is the reciprocal temperature
            # in this module, and binding the box's beta ANGLE to that name once made it local to
            # the writer and put the angle times the work into `cumulative_reduced_work`.
            box = mdtraj.utils.box_vectors_to_lengths_and_angles(
                vectors[0], vectors[1], vectors[2])
            lengths = [box[0] * 10.0, box[1] * 10.0, box[2] * 10.0]
            angles = [box[3], box[4], box[5]]
        netcdf.write(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                     time=switching_time_ps, cell_lengths=lengths, cell_angles=angles)
        netcdf.flush()
        frames_emitted += 1
        fault("after-frame")

    def write_observation(observation_index: int, protocol_step: int, switching_time_ps: float,
                          tau: float, incremental: float, wrote_frame: bool) -> None:
        nonlocal rows_emitted
        fault("before-work-row")
        append(directory / OBSERVATIONS_CSV, OBSERVATION_COLUMNS, {
            "path_index": index,
            "observation_index": observation_index,
            # The index into the path trajectory when this observation also wrote a frame, and
            # empty when it did not. The two cadences are independent now, so a row that claimed
            # a frame index it does not have would be a lie a reader could not detect.
            "coordinate_frame_index": frames_emitted - 1 if wrote_frame else "",
            "source_frame_index": frame,
            "protocol_step": protocol_step,
            "switching_time_ps": switching_time_ps,
            "tau": tau,
            "incremental_work_kj_mol": incremental,
            "cumulative_work_kj_mol": cumulative,
            "cumulative_reduced_work": beta * cumulative,
            "temperature_kelvin": temperature,
            "integrator_seed": integrator_seed,
            "velocity_seed": velocity_seed,
        })
        rows_emitted += 1
        fault("after-work-row")

    def write_state(protocol_step: int, switching_time_ps: float, tau: float) -> None:
        nonlocal state_rows_emitted
        fault("before-state-row")
        append(directory / STATE_CSV, STATE_COLUMNS,
               _state_row(simulation, index=index, protocol_step=protocol_step,
                          switching_time_ps=switching_time_ps, tau=tau, implicit=implicit,
                          temperature_unit=unit.kelvin))
        state_rows_emitted += 1
        fault("after-state-row")

    def save_checkpoint(updates_completed: int) -> None:
        """One crash-atomic generation: a new checkpoint, its sidecar, then the pointer.

        The committed pair is never overwritten and the pointer is replaced last, so a crash
        anywhere in here leaves the PREVIOUS generation committed and complete. The old design
        overwrote the binary state and then replaced the sidecar, and a crash between those two
        writes left a new Context paired with old accumulated work -- a resume that looks
        successful and is measuring a path that was never run.
        """
        netcdf.flush()
        commit_generation(
            directory,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={
                "fingerprint": fingerprint,
                "path_index": index,
                "source_frame_index": frame,
                "updates_completed": updates_completed,
                "protocol_step": updates_completed * interval,
                "tau": taus[updates_completed],
                "cumulative_work_kj_mol": cumulative,
                "work_since_last_observation_kj_mol": since,
                "work_rows": rows_emitted,
                "frames": frames_emitted,
                "state_rows": state_rows_emitted,
                "integrator_seed": integrator_seed,
                "velocity_seed": velocity_seed,
                "trajectory": trajectory_name,
            })

    def time_of(step: int) -> float:
        return round(step * float(dynamics["timestep_fs"]) / 1000.0, 9)

    # -- step 0: the source configuration, before any work ---------------------------------------
    if updates_done == 0 and rows_emitted == 0:
        wrote = False
        if 0 % frame_every == 0:
            write_frame(0, 0.0)
            wrote = True
        write_observation(0, 0, 0.0, taus[0], 0.0, wrote)
        if state_every:
            write_state(0, 0.0, taus[0])

    # -- the switch ------------------------------------------------------------------------------
    for update in range(updates_done, updates):
        before = simulation.context.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
        switcher.set_tau(simulation.context, system, taus[update + 1])
        after = simulation.context.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
        increment = after - before
        cumulative += increment
        since += increment
        simulation.step(interval)

        step = (update + 1) * interval
        wrote = False
        if step % frame_every == 0:
            write_frame(step, time_of(step))
            wrote = True
        if step % observe_every == 0:
            write_observation(step // observe_every, step, time_of(step), taus[update + 1],
                              since, wrote)
            since = 0.0
        if state_every and step % state_every == 0:
            write_state(step, time_of(step), taus[update + 1])
        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(update + 1)

    # -- finish: flush, validate, publish, and only then declare completion -----------------------
    final = simulation.context.getState(getPositions=True, getVelocities=True,
                                        getParameters=True, enforcePeriodicBox=False)
    (directory / "final_state.xml").write_text(XmlSerializer.serialize(final), encoding="utf-8")
    netcdf.close()

    if rows_emitted != schedule["number_of_observations"]:
        raise SystemExit(f"path {index} wrote {rows_emitted} observations, expected "
                         f"{schedule['number_of_observations']}")
    if frames_emitted != schedule["number_of_frames"]:
        raise SystemExit(f"path {index} wrote {frames_emitted} frames, expected "
                         f"{schedule['number_of_frames']}")
    if state_every and state_rows_emitted != schedule["number_of_state_rows"]:
        raise SystemExit(f"path {index} wrote {state_rows_emitted} state rows, expected "
                         f"{schedule['number_of_state_rows']}")

    with (directory / OBSERVATIONS_CSV).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if float(rows[0]["cumulative_work_kj_mol"]) != 0.0:
        raise SystemExit(f"path {index} observation 0 has non-zero work")
    if float(rows[0]["tau"]) != float(ais["tau_start"]) \
            or float(rows[-1]["tau"]) != float(ais["tau_end"]):
        raise SystemExit(f"path {index} does not span tau_start -> tau_end")

    # The staged trajectory becomes the published one only now, in one atomic move. Until this
    # point nothing at the run root could be mistaken for a finished path.
    os.replace(staged, published)
    with mdtraj.formats.NetCDFTrajectoryFile(str(published)) as handle:
        published_frames = len(handle)
    if published_frames != schedule["number_of_frames"]:
        raise SystemExit(f"path {index}: {published.name} holds {published_frames} frames, "
                         f"expected {schedule['number_of_frames']}")

    completion = {
        "status": "completed",
        "path_index": index,
        "source_frame_index": frame,
        "observations": rows_emitted,
        "frames": frames_emitted,
        "state_rows": state_rows_emitted,
        "total_work_kj_mol": cumulative,
        "total_reduced_work": beta * cumulative,
        "integrator_seed": integrator_seed,
        "velocity_seed": velocity_seed,
        "trajectory": trajectory_name,
        "mpi_rank": rank,
        "platform": simulation.context.getPlatform().getName(),
        "resumed": bool(state_of_path),
    }
    marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    # The transaction has served its purpose; leaving it would invite a resume of finished work.
    clear_committed(directory)
    log(f"  path {index:4d}: frame {frame}, {rows_emitted} observations, {frames_emitted} "
        f"frames, W = {cumulative:.4f} kJ/mol (reduced {beta * cumulative:.4f})")
    return completion


def _read_source_frame(source_path: Path, top, frame: int, *, implicit: bool):
    """One frame's positions and box, in the reduced form OpenMM requires."""
    import mdtraj
    from openmm import unit

    positions = boxes = None
    for offset, chunk in enumerate(mdtraj.iterload(str(source_path), top=top, chunk=50)):
        if offset * 50 <= frame < offset * 50 + chunk.n_frames:
            local = frame - offset * 50
            positions = chunk.xyz[local] * unit.nanometer
            if not implicit and chunk.unitcell_lengths is not None:
                # Rebuilt from lengths and angles rather than handed over as the vectors the file
                # happens to store. OpenMM requires REDUCED form, and a truncated octahedron
                # written by any of the usual tools is not in it -- `setPeriodicBoxVectors` then
                # refuses and every path dies at its first frame.
                import numpy
                from openmm.app.internal.unitcell import computePeriodicBoxVectors

                lengths = chunk.unitcell_lengths[local]
                angles = numpy.radians(chunk.unitcell_angles[local])
                boxes = computePeriodicBoxVectors(
                    float(lengths[0]), float(lengths[1]), float(lengths[2]),
                    float(angles[0]), float(angles[1]), float(angles[2]))
            break
    if positions is None:
        raise SystemExit(f"could not read frame {frame} from {source_path}")
    return positions, boxes


def ais_main(run: dict[str, Any], argv: list[str] | None = None) -> int:
    """Run the switching paths this generated script describes."""
    args = ais_parser(run.get("description", "AIS switching paths")).parse_args(argv)

    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    from .schedule import switching_schedule
    from ..openmm.timestep import resolve_timestep_fs
    from ..openmm.system import classify_omega_bonds
    from ..md._stages import derive_seed
    from ..rest2 import TauSwitcher
    from ..md.stage import solute_atom_indices

    from ..remd.mpi import Coordination
    from . import path_trajectory_name, paths_for_rank

    ais, source_cfg, dynamics = run["ais"], run["ais_source"], run["dynamics"]
    reporting = run["reporting"]

    # EVERYTHING is validated before `-odir` exists, here in the shared runtime rather than only
    # in `md-openmm md-run`: the generated `AIS.py` calls this function directly. Paths are
    # independent, so `-ng` is simply the worker count, and it must equal the communicator.
    topology_path = Path(args.topology)
    system_path = Path(args.system)
    source_path = Path(args.source or source_cfg["trajectory"] or "")
    out = Path(args.out_dir).resolve()

    from ..run.preflight import PreflightError, preflight_ais

    try:
        checked = preflight_ais(
            topology=topology_path, system=system_path, source=source_path,
            number_of_groups=args.number_of_groups,
            output=args.output or out / "AIS.out", log=args.log or out / "AIS.log",
            cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None)
    except PreflightError as refusal:
        print(f"AIS: {refusal}", file=sys.stderr)
        return 2

    coordination = checked.coordination
    rank, size = coordination.rank, coordination.size
    source_format = checked.source_format

    out.mkdir(parents=True, exist_ok=True)
    # Every rank keeps its own pair. An explicitly named -log or -o is suffixed the same way an
    # unnamed one is: without that, N ranks race to rename the same temporary and the run dies
    # with a FileNotFoundError that says nothing about the cause. A rank that failed to bind its
    # device is exactly what a multi-GPU run needs to be able to show, so the other ranks' files
    # are kept beside rank 0's rather than discarded.
    from ..remd.executor import report_path_for_rank

    log_path = Path(report_path_for_rank(str(Path(args.log) if args.log else out / "AIS.log"),
                                         rank))
    out_path = Path(report_path_for_rank(str(Path(args.output) if args.output else out / "AIS.out"),
                                         rank))

    from ..build.simout import SimulationOutput

    sim_out = SimulationOutput(out_path, title=f"AIS: {ais['number_of_paths']} switching paths",
                               log_path=log_path)
    sim_out.heading("Inputs")
    sim_out.field("topology", topology_path)
    sim_out.field("system", system_path)
    sim_out.field("source", f"{source_path} ({source_format.upper()})")
    sim_out.field("output directory", out)

    log = LogWriter(log_path, record_type="md-ais")
    log.update(simulation_output=str(out_path))
    log("md-openmm AIS")
    log("=" * 68)

    try:
        # The System is opened BEFORE the schedule is built, because the schedule reports a
        # duration and the duration needs the numerical timestep -- which under `auto` is a fact
        # about the masses in this System, not about the configuration.
        pdb = PDBFile(str(topology_path))
        base = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))

        timestep = resolve_timestep_fs(dynamics["timestep_fs"], base, pdb.topology)
        dynamics = dict(dynamics, timestep_fs=timestep["timestep_fs"])
        log.field("timestep", f"{timestep['timestep_fs']} fs (requested "
                              f"{timestep['requested']!r}, {timestep['basis']})")
        log.update(timestep=timestep)

        schedule = switching_schedule(
            tau_start=float(ais["tau_start"]), tau_end=float(ais["tau_end"]),
            switching_steps=int(ais["switching_steps"]),
            parameter_update_interval_steps=int(ais["parameter_update_interval_steps"]),
            observation_interval_steps=int(ais["observation_interval_steps"]),
            timestep_fs=float(dynamics["timestep_fs"]),
            # The three reporting cadences are settings a person wrote and they now do what they
            # say. `system_printout` and `checkpoint_printout` were validated and then dropped:
            # accepted, consequential-looking, and inert.
            trajectory_interval_steps=int(reporting["solute_printout"]),
            state_interval_steps=int(reporting["system_printout"]),
            checkpoint_interval_steps=int(reporting["checkpoint_printout"]))
        if pdb.topology.getNumAtoms() != base.getNumParticles():
            raise SystemExit(f"{topology_path} has {pdb.topology.getNumAtoms()} atoms but "
                             f"{system_path} has {base.getNumParticles()} particles")
        implicit = not base.usesPeriodicBoundaryConditions()

        # No barostat during switching, ever. A pressure-volume term would enter the work and the
        # path would no longer measure what the Jarzynski/Crooks relations are written for.
        from ..md._stages import count_barostats
        if count_barostats(base):
            raise SystemExit(
                "the prepared System carries a barostat. AIS switches at FIXED VOLUME: each path "
                "keeps the box of the frame it started from, and no pressure-volume term enters "
                "the work. Build the System without a barostat.")

        solute = solute_atom_indices(pdb.topology)
        omega = classify_omega_bonds(pdb.topology, solute, route="peptide", ligand_sdf=None)
        excluded = [tuple(int(a) for a in bond)
                    for bond in omega.get("omega_unscaled_bonds", [])]

        log.heading("Path")
        log.field("tau", f"{ais['tau_start']} -> {ais['tau_end']} (linear)")
        log.field("switching", f"{schedule['switching_steps']} steps "
                               f"= {schedule['switching_ps']:g} ps at {dynamics['timestep_fs']} fs")
        log.field("parameter updates", f"{schedule['number_of_updates']} "
                                       f"(every {schedule['parameter_update_interval_steps']} step)")
        log.field("observations", f"{schedule['number_of_observations']} "
                                  f"(work, every {schedule['observation_interval_steps']} steps, "
                                  f"both endpoints included)")
        log.field("frames", f"{schedule['number_of_frames']} "
                            f"(every {schedule['trajectory_interval_steps']} steps)")
        log.field("state rows", f"{schedule['number_of_state_rows']} "
                                f"(every {schedule['state_interval_steps']} steps)"
                  if schedule["state_interval_steps"] else "disabled (system_printout = 0)")
        log.field("checkpoints", f"{schedule['number_of_checkpoints']} "
                                 f"(every {schedule['checkpoint_interval_steps']} steps)"
                  if schedule["checkpoint_interval_steps"] else
                  "disabled (checkpoint_printout = 0): an interrupted path restarts from its "
                  "source frame")
        log.field("observation 0", "the source configuration, before any work")
        log.field("omega bonds", f"{len(excluded)} left unscaled")
        log.field("ensemble", "fixed volume; no barostat")

        # --- the source ensemble -----------------------------------------------------------
        import mdtraj

        top = mdtraj.Topology.from_openmm(pdb.topology)
        n_frames = sum(chunk.n_frames for chunk in
                       mdtraj.iterload(str(source_path), top=top, chunk=50))
        last = source_cfg["last_frame"]
        last = n_frames - 1 if last is None else int(last)
        first = int(source_cfg["first_frame"])
        if last >= n_frames:
            raise SystemExit(f"ais_source.last_frame = {last} but {source_path.name} holds "
                             f"{n_frames} frame(s) (0..{n_frames - 1})")
        stride = int(source_cfg["frame_stride"])
        eligible = list(range(first, last + 1, stride))

        # The source must describe the SAME particles as the System the paths run in. Read from
        # the FILE's own header rather than through `top`, which is derived from -p and would
        # therefore agree with itself. mdtraj would fail on a mismatch too, but obscurely and
        # several frames in; this fails here, with both counts and both filenames.
        # Hashed ONCE, here, and reused by both the record and the resume fingerprint. A
        # production trajectory is large and is already read frame by frame; digesting it twice
        # would double that for a number that has one value.
        source_facts = file_facts(source_path)
        source_atoms = _source_atom_count(source_path)
        if source_atoms is not None and source_atoms != pdb.topology.getNumAtoms():
            raise SystemExit(
                f"-source-traj {source_path.name} holds {source_atoms} atom(s) but "
                f"{topology_path.name} and {system_path.name} describe "
                f"{pdb.topology.getNumAtoms()}. The source ensemble must be of the same system "
                f"the paths are run in; a trajectory of a different one reads without error and "
                f"produces work values that mean nothing.")

        log.heading("Source ensemble")
        log.field("trajectory", f"{source_path}  ({source_format.upper()})")
        log.field("atoms", f"{source_atoms}, matching {topology_path.name}"
                  if source_atoms is not None else "not stated by the file format")
        log.field("frames", f"{n_frames} total, {len(eligible)} eligible "
                            f"(frames {first}..{last} inclusive"
                            + (f", every {stride}" if stride > 1 else "") + ")")
        log.field("selection", source_cfg["selection"])
        # Stated, not verified. Nothing in a coordinate trajectory records the Hamiltonian it was
        # sampled under, so this is an assertion the configuration makes and the log repeats --
        # written down precisely so that a reader can check it against the run that produced the
        # file rather than assume it was checked here.
        log.field("asserted ensemble", f"tau = {ais['tau_start']} (ASSERTED by ais.tau_start, "
                                       f"not verifiable from the trajectory itself)")
        log.field("velocities", "not read from the source; each path draws fresh "
                                f"Maxwell-Boltzmann momenta at {dynamics['temperature_K']} K "
                                f"with its own recorded seed")

        chosen = choose_frames(eligible=eligible, count=int(ais["number_of_paths"]),
                               selection=source_cfg["selection"],
                               allow_repeats=bool(source_cfg["allow_repeated_frames"]),
                               seed=int(dynamics["seed"]))
        log.field("paths", f"{len(chosen)} starting from frames {chosen[:8]}"
                           + (" ..." if len(chosen) > 8 else ""))

        log.update(
            schedule=schedule,
            source={"trajectory": source_path.name, "format": source_format,
                    "atoms": None if source_atoms is None else int(source_atoms),
                    "tau_asserted": float(ais["tau_start"]),
                    "tau_verified_from_file": False,
                    "velocity_policy": "resampled_maxwell_boltzmann",
                    "frames_total": n_frames,
                    "first_frame": first, "last_frame": last,
                    "frame_stride": stride,
                    "selection": source_cfg["selection"],
                    "allow_repeated_frames": bool(source_cfg["allow_repeated_frames"]),
                    "chosen_frames": chosen},
            thermodynamic_states={"source_tau": float(ais["tau_start"]),
                                  "target_tau": float(ais["tau_end"]),
                                  "temperature_K": float(dynamics["temperature_K"]),
                                  "ensemble": "fixed volume (NVT), no barostat"},
            work_convention="delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j); parameters move at "
                            "frozen coordinates, then the configuration propagates",
            inputs={"topology": file_facts(topology_path), "system": file_facts(system_path),
                    "source": source_facts},
            implicit=bool(implicit),
        )

        # Written BEFORE any dynamics, so which frame each path started from is recorded even if
        # the run is interrupted. Every rank computes the SAME table -- `choose_frames` is a pure
        # function of the window, the count and the run seed -- so only rank 0 writes it, and the
        # others would otherwise race to truncate the file rank 0 is writing.
        if rank == 0:
            write_selected_frames(out / "selected_source_frames.csv", chosen,
                                  seed=int(dynamics["seed"]))

        if args.check:
            log.heading("Preflight")
            log("  --check: schedule, source and Force layout validated; nothing was switched.")
            log.record["status"] = "checked"
            log.save()
            sim_out.heading("Preflight")
            sim_out.write("  --check: schedule, source and Force layout validated; nothing was "
                          "switched.")
            sim_out.completed("checked; no paths were run")
            return 0

        # --- run the paths -------------------------------------------------------------------
        # The preflight already resolved this -- machine settings, device policy, rank placement
        # and a proved Context -- before this function created anything. Consuming its result is
        # what keeps ONE platform policy in the package: resolving again here would be a second
        # implementation, and the one that runs is never the one that gets fixed.
        acceleration = checked.acceleration
        device = checked.device_index
        device_policy = checked.device_policy_detail
        log.field("platform", f"{acceleration.name}"
                              + (f" device {acceleration.device_index}"
                                 if acceleration.device_index is not None else ""))
        log.field("device policy", device_policy)
        log.update(acceleration=checked.record())

        # What a mid-path checkpoint has to match before it may be resumed from. The System, the
        # topology, the source ensemble and the whole schedule: a checkpoint carries positions and
        # velocities for one particular simulation, and resuming it against another produces
        # right-looking numbers for the wrong thing.
        import hashlib

        path_fingerprint = hashlib.sha256(json.dumps({
            "system": file_facts(system_path)["sha256"],
            "topology": file_facts(topology_path)["sha256"],
            "source": source_facts["sha256"],
            "schedule": {k: v for k, v in schedule.items()
                         if k not in ("observations", "taus", "note")},
            "temperature_K": float(dynamics["temperature_K"]),
            "friction_per_ps": float(dynamics["friction_per_ps"]),
            "seed": int(dynamics["seed"]),
            "resolved_config": run.get("resolved_config"),
        }, sort_keys=True).encode()).hexdigest()

        switcher = TauSwitcher(base, solute, excluded)
        taus = schedule["taus"]
        interval = schedule["parameter_update_interval_steps"]
        per_observation = schedule["updates_per_observation"]
        temperature = float(dynamics["temperature_K"])
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(
            unit.kilojoule_per_mole)

        # Which paths this worker runs. `paths_for_rank` is a pure function of
        # (rank, size, total): a given global path id always owns the same file name, so a
        # restart under a different worker count lands on the same trajectories rather than
        # silently reshuffling which path is which.
        wanted = _selected(args.paths, len(chosen))
        if size > 1:
            mine = set(paths_for_rank(rank, size, len(chosen)))
            wanted = [index for index in wanted if index in mine]
            log.field("mpi", f"rank {rank} of {size}: {len(wanted)} of {len(chosen)} path(s)")
            log.update(mpi={"rank": rank, "size": size, "paths": list(wanted)})

        sim_out.heading("Schedule")
        sim_out.field("tau", f"{ais['tau_start']} -> {ais['tau_end']} (linear)")
        sim_out.field("switching", f"{schedule['switching_steps']} steps "
                                   f"= {schedule['switching_ps']:g} ps")
        sim_out.field("work observations", f"{schedule['number_of_observations']} "
                                           f"(every {schedule['observation_interval_steps']})")
        sim_out.field("trajectory frames", f"{schedule['number_of_frames']} "
                                           f"(every {schedule['trajectory_interval_steps']})")
        sim_out.field("state rows", f"{schedule['number_of_state_rows']} "
                                    f"(every {schedule['state_interval_steps']})"
                      if schedule["state_interval_steps"] else "disabled")
        sim_out.field("checkpoints", f"{schedule['number_of_checkpoints']} "
                                     f"(every {schedule['checkpoint_interval_steps']})"
                      if schedule["checkpoint_interval_steps"] else "disabled")
        sim_out.field("platform", acceleration.name)
        if size > 1:
            sim_out.field("mpi", f"rank {rank} of {size}, {len(wanted)} path(s) of {len(chosen)}")

        sim_out.heading("Paths")
        sim_out.table_header(("path", "frame", "obs", "frames", "W kJ/mol", "reduced"),
                             (6, 8, 6, 8, 14, 12))

        log.heading("Paths")
        completed: list[dict[str, Any]] = []
        for index in wanted:
            record = run_one_path(
                index=index, chosen=chosen, out=out, schedule=schedule, taus=taus,
                switcher=switcher, simulation_inputs=dict(
                    topology=pdb.topology, source_path=source_path, mdtraj_top=top,
                    implicit=implicit, acceleration=acceleration),
                dynamics=dynamics, ais=ais, beta=beta, temperature=temperature,
                rank=rank, resume=bool(args.resume), fingerprint=path_fingerprint, log=log)
            if record is not None:
                completed.append(record)
                sim_out.table_row(
                    (record["path_index"], record["source_frame_index"], record["observations"],
                     record.get("frames", 0), float(record["total_work_kj_mol"]),
                     float(record["total_reduced_work"])), (6, 8, 6, 8, 14, 12))

        # The global work table. Every worker has written a completion record per path it owns;
        # rank 0 waits for all of them and assembles ONE table in global path order. The work
        # distribution is the result of the method, and it has to be readable as a single object
        # rather than as N per-rank fragments the reader is left to concatenate correctly.
        coordination.barrier()
        outputs = {"selected_source_frames": file_facts(out / "selected_source_frames.csv",
                                                        relative_to=out)}
        if rank == 0 and not args.paths:
            table = write_work_table(out, chosen)
            log.field("work table", f"{table['rows']} row(s) from {table['paths']} of "
                                    f"{table['requested']} path(s)")
            outputs["work_table"] = file_facts(out / WORK_TABLE, relative_to=out)
            outputs["work_summary"] = file_facts(out / WORK_SUMMARY, relative_to=out)

        log.heading("Outputs")
        log.field("paths completed", f"{len(completed)} of {len(chosen)}")
        log.update(
            platform=openmm_platform_facts(),
            paths=completed,
            outputs=outputs,
        )
        log.complete()
        log.heading("Summary")
        log(f"  {len(completed)} switching path(s), "
            f"{schedule['number_of_observations']} observations each")
        log("  status: completed")
        sim_out.completed(f"{len(completed)} switching path(s), "
                          f"{schedule['number_of_observations']} work observations and "
                          f"{schedule['number_of_frames']} frames each")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        sim_out.failed(f"{type(exc).__name__}: {exc}")
        print(f"AIS: {exc}", file=sys.stderr)
        return 1

    log.save()
    return 0


def run_generated_ais(script: str | Path, argv: list[str] | None = None) -> int:
    """Run the AIS paths described by the `resolved.config` beside this script.

    The whole body of a generated `AIS.py`:

        from md_tools.ais import run_generated_ais
        raise SystemExit(run_generated_ais(__file__))
    """
    from ..build.md import resolve_md_config
    from ..md.stage import resolved_config_beside

    config_path = resolved_config_beside(script)
    resolved = resolve_md_config(config_path)
    if resolved["protocol"] != "AIS":
        raise SystemExit(
            f"{config_path} declares protocol {resolved['protocol']!r}, but this is an AIS "
            f"script. The directory holds a script and a configuration that describe different "
            f"runs; regenerate it.")
    run = {
        "protocol": "AIS",
        "description": f"AIS: {resolved['ais']['number_of_paths']} switching paths",
        "ais": dict(resolved["ais"]),
        "ais_source": dict(resolved["ais_source"]),
        "dynamics": dict(resolved["dynamics"]),
        # The reporting block reaches the runtime. It used to be validated by `build-md` and then
        # left behind, so `system_printout` and `checkpoint_printout` were settings a person wrote
        # that nothing ever read.
        "reporting": dict(resolved["reporting"]),
        "resolved_config": str(config_path),
    }
    return ais_main(run, argv)
