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

#: One row per observation. `s` and sqrt(s) are absent on purpose: see the module docstring.
OBSERVATION_COLUMNS = (
    "path_index", "observation_index", "coordinate_frame_index",
    "source_frame_index", "protocol_step", "switching_time_ps", "tau",
    "incremental_work_kj_mol", "cumulative_work_kj_mol", "cumulative_reduced_work",
    "temperature_kelvin", "integrator_seed", "velocity_seed",
)

COMPLETION_NAME = "completed.json"
OBSERVATIONS_CSV = "observations.csv"

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
    parser = argparse.ArgumentParser(description=description)
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
                        help="readable log carrying this run's machine record")
    parser.add_argument("--cpu", action="store_true",
                        help="run the paths on the OpenMM CPU platform. CUDA is the default and "
                             "is mandatory; this is the only way to ask for a CPU run, and the "
                             "record says that you did")
    parser.add_argument("--platform", default=None,
                        help="force a named OpenMM platform. CUDA is the default; there is no "
                             "automatic fall back to anything else")
    parser.add_argument("--device", default=None, help="CUDA device index")
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

    from ..remd.executor import barrier, mpi_rank_and_size
    from . import path_trajectory_name, paths_for_rank

    ais, source_cfg, dynamics = run["ais"], run["ais_source"], run["dynamics"]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Paths are independent, so AIS parallelises by simply giving each worker its own paths. The
    # rank is read from the launcher's environment rather than by importing mpi4py: the split has
    # to be known before anything opens a Context, and a single-process run must work on a machine
    # with no MPI at all.
    rank, size = mpi_rank_and_size()
    log_path = Path(args.log) if args.log else out / (
        "AIS.log" if size == 1 else f"AIS.rank{rank:02d}.log")

    topology_path = Path(args.topology)
    system_path = Path(args.system)
    source_path = Path(args.source or source_cfg["trajectory"])
    for path, what in ((topology_path, "-p topology"), (system_path, "-s system")):
        if not path.is_file():
            print(f"AIS: {what} {path} does not exist", file=sys.stderr)
            return 2
    if not source_path.is_file():
        # `is_file`, deliberately, not `exists`. A path written with a trailing slash names a
        # DIRECTORY, and `exists` would accept `tau_0p5.nc/` and fail later, obscurely, inside
        # mdtraj rather than here with the path in the message.
        what = ("is a directory, not a file" if source_path.is_dir() else "does not exist")
        print(f"AIS: -source-traj {source_path} {what}. AIS consumes an equilibrium ensemble you "
              f"have already produced; it does not generate one.", file=sys.stderr)
        return 2

    log = LogWriter(log_path, record_type="md-ais")
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
            timestep_fs=float(dynamics["timestep_fs"]))
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
                                  f"(every {schedule['observation_interval_steps']} steps, "
                                  f"both endpoints included)")
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
        source_atoms = _source_atom_count(source_path)
        if source_atoms is not None and source_atoms != pdb.topology.getNumAtoms():
            raise SystemExit(
                f"-source-traj {source_path.name} holds {source_atoms} atom(s) but "
                f"{topology_path.name} and {system_path.name} describe "
                f"{pdb.topology.getNumAtoms()}. The source ensemble must be of the same system "
                f"the paths are run in; a trajectory of a different one reads without error and "
                f"produces work values that mean nothing.")

        log.heading("Source ensemble")
        log.field("trajectory", source_path)
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
            source={"trajectory": source_path.name,
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
                    "source": file_facts(source_path)},
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
            return 0

        # --- run the paths -------------------------------------------------------------------
        # ONE platform decision for the whole run, made before the first path opens a Context:
        # a CUDA device that cannot be initialised must fail here rather than 40 paths in. Under
        # MPI each rank takes its own device by local rank unless one was named.
        from ..openmm.platform_policy import (PlatformRequest, acceleration_record,
                                              resolve_platform_request)
        from ..remd.engine import select_device_for_rank, visible_cuda_devices

        device, device_policy = args.device, "named on the command line"
        if device is None:
            if size > 1 and not args.cpu:
                # Nothing binds ranks to devices automatically: without this every rank creates
                # its Context on the default device and the whole run sits on one GPU.
                device, device_policy = select_device_for_rank(
                    rank, size, visible_cuda_devices(probe=True))
                if device is None:
                    raise SystemExit(
                        "AIS was launched under MPI on CUDA but no CUDA device is visible to "
                        "this rank. Refusing rather than letting every rank fall onto one GPU.")
            else:
                device_policy = "single process: OpenMM selects the device"
        acceleration = resolve_platform_request(
            PlatformRequest.from_flags(cpu=bool(args.cpu),
                                       platform=args.platform or dynamics.get("platform")),
            device_index=device)
        log.field("platform", f"{acceleration.name}"
                              + (f" device {acceleration.device_index}"
                                 if acceleration.device_index is not None else ""))
        log.update(acceleration=dict(
            acceleration_record(acceleration, mpi_rank=rank, mpi_size=size, local_rank=device),
            device_policy=device_policy))

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

        log.heading("Paths")
        completed: list[dict[str, Any]] = []
        for index in wanted:
            directory = out / f"path_{index:04d}"
            marker = directory / COMPLETION_NAME
            if marker.is_file():
                log(f"  path {index:4d}: already completed; not rerun and never appended to")
                completed.append(json.loads(marker.read_text(encoding="utf-8")))
                continue
            directory.mkdir(parents=True, exist_ok=True)

            frame = chosen[index]
            integrator_seed = derive_seed(int(dynamics["seed"]), "ais", index, "integrator")
            velocity_seed = derive_seed(int(dynamics["seed"]), "ais", index, "velocity")

            positions = None
            boxes = None
            for offset, chunk in enumerate(mdtraj.iterload(str(source_path), top=top, chunk=50)):
                if offset * 50 <= frame < offset * 50 + chunk.n_frames:
                    local = frame - offset * 50
                    positions = chunk.xyz[local] * unit.nanometer
                    if not implicit and chunk.unitcell_lengths is not None:
                        # Rebuilt from lengths and angles rather than handed over as the vectors
                        # the file happens to store. OpenMM requires REDUCED form, and a
                        # truncated octahedron written by any of the usual tools is not in it --
                        # `setPeriodicBoxVectors` then refuses with "Periodic box vectors must be
                        # in reduced form" and every path of the run dies at its first frame.
                        # `computePeriodicBoxVectors` is OpenMM's own reduction, so the box the
                        # path keeps is the source frame's box, expressed the way OpenMM needs it.
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

            system = switcher.prepared_system(taus[0])
            integrator = LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                float(dynamics["friction_per_ps"]) / unit.picosecond,
                float(dynamics["timestep_fs"]) * unit.femtosecond)
            integrator.setRandomNumberSeed(int(integrator_seed))
            simulation = Simulation(pdb.topology, system, integrator,
                                    acceleration.platform, acceleration.properties)
            if boxes is not None:
                simulation.context.setPeriodicBoxVectors(*boxes)
            simulation.context.setPositions(positions)
            # A trajectory carries no velocities, so each path draws fresh Maxwell-Boltzmann
            # momenta at the run temperature with its own recorded seed.
            simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin,
                                                          int(velocity_seed))

            def energy() -> float:
                return simulation.context.getState(getEnergy=True).getPotentialEnergy(
                    ).value_in_unit(unit.kilojoule_per_mole)

            rows: list[dict[str, Any]] = []
            # One trajectory per PATH, named by its global path id, at the run root. The name is
            # a pure function of (path id, total): the same path writes the same file whatever
            # the worker count, so a campaign resumed on a different number of GPUs does not
            # reshuffle which trajectory is which. AMBER NetCDF rather than DCD because every
            # analysis tool in this stack (cpptraj, mdtraj, MDAnalysis) reads it and it carries
            # the box for a switching path that started from an NPT frame.
            trajectory_name = path_trajectory_name(index, len(chosen))
            netcdf = mdtraj.formats.NetCDFTrajectoryFile(str(out / trajectory_name), "w")

            def observe(observation, cumulative, incremental) -> None:
                state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
                lengths = angles = None
                if not implicit:
                    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer)
                    # NOT unpacked as `a, b, c, alpha, beta, gamma`. `beta` is the reciprocal
                    # temperature in this function, and binding the box's beta ANGLE to that name
                    # here made it local to this closure: `beta * cumulative` below then wrote the
                    # angle times the work into `cumulative_reduced_work` under explicit solvent,
                    # and raised UnboundLocalError under implicit, where the branch never ran.
                    box = mdtraj.utils.box_vectors_to_lengths_and_angles(
                        vectors[0], vectors[1], vectors[2])
                    lengths = [box[0] * 10.0, box[1] * 10.0, box[2] * 10.0]
                    angles = [box[3], box[4], box[5]]
                netcdf.write(
                    state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                    time=observation["switching_time_ps"],
                    cell_lengths=lengths, cell_angles=angles)
                rows.append({
                    "path_index": index,
                    "observation_index": observation["observation_index"],
                    # Appended in lockstep with the DCD, so the k-th row IS the k-th frame.
                    "coordinate_frame_index": len(rows),
                    "source_frame_index": frame,
                    "protocol_step": observation["protocol_step"],
                    "switching_time_ps": observation["switching_time_ps"],
                    "tau": observation["tau"],
                    "incremental_work_kj_mol": incremental,
                    "cumulative_work_kj_mol": cumulative,
                    "cumulative_reduced_work": beta * cumulative,
                    "temperature_kelvin": temperature,
                    "integrator_seed": integrator_seed,
                    "velocity_seed": velocity_seed,
                })

            # Observation 0: the source configuration under the source Hamiltonian, before any
            # parameter change and before any propagation. Its work is exactly zero.
            cumulative = 0.0
            since = 0.0
            observe(schedule["observations"][0], 0.0, 0.0)

            for update in range(schedule["number_of_updates"]):
                before = energy()
                switcher.set_tau(simulation.context, system, taus[update + 1])
                after = energy()
                increment = after - before
                cumulative += increment
                since += increment
                simulation.step(interval)
                if (update + 1) % per_observation == 0:
                    observe(schedule["observations"][(update + 1) // per_observation],
                            cumulative, since)
                    since = 0.0

            state = simulation.context.getState(getPositions=True, getVelocities=True,
                                                getParameters=True, enforcePeriodicBox=False)
            (directory / "final_state.xml").write_text(XmlSerializer.serialize(state),
                                                       encoding="utf-8")
            netcdf.close()
            with (directory / OBSERVATIONS_CSV).open("w", newline="") as table:
                writer = csv.DictWriter(table, fieldnames=list(OBSERVATION_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)

            # Validated before completion is declared: the row count, the endpoints, and that the
            # first row really is zero work at tau_start.
            if len(rows) != schedule["number_of_observations"]:
                raise SystemExit(f"path {index} wrote {len(rows)} observations, expected "
                                 f"{schedule['number_of_observations']}")
            if rows[0]["cumulative_work_kj_mol"] != 0.0:
                raise SystemExit(f"path {index} observation 0 has non-zero work")
            if rows[0]["tau"] != float(ais["tau_start"]) or rows[-1]["tau"] != float(ais["tau_end"]):
                raise SystemExit(f"path {index} does not span tau_start -> tau_end")

            completion = {
                "status": "completed",
                "path_index": index,
                "source_frame_index": frame,
                "observations": len(rows),
                "total_work_kj_mol": cumulative,
                "total_reduced_work": beta * cumulative,
                "integrator_seed": integrator_seed,
                "velocity_seed": velocity_seed,
                "trajectory": trajectory_name,
                "mpi_rank": rank,
                "platform": simulation.context.getPlatform().getName(),
            }
            marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
            completed.append(completion)
            log(f"  path {index:4d}: frame {frame}, {len(rows)} observations, "
                f"W = {cumulative:.4f} kJ/mol (reduced {beta * cumulative:.4f})")

        # The global work table. Every worker has written a completion record per path it owns;
        # rank 0 waits for all of them and assembles ONE table in global path order. The work
        # distribution is the result of the method, and it has to be readable as a single object
        # rather than as N per-rank fragments the reader is left to concatenate correctly.
        barrier(size)
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
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
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
        "resolved_config": str(config_path),
    }
    return ais_main(run, argv)
