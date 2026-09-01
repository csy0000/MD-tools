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
OBSERVATIONS_DCD = "observations.dcd"


def ais_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates (built.pdb)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System (built.xml)")
    parser.add_argument("-src", "--source", default=None, metavar="TRAJ",
                        help="the equilibrium source trajectory; overrides the configured path")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="where the path directories are written (default: here)")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="readable log carrying this run's machine record")
    parser.add_argument("--platform", default=None,
                        help="force an OpenMM platform (CUDA, OpenCL, CPU, Reference)")
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


def ais_main(run: dict[str, Any], argv: list[str] | None = None) -> int:
    """Run the switching paths this generated script describes."""
    args = ais_parser(run.get("description", "AIS switching paths")).parse_args(argv)

    from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import DCDFile, PDBFile, Simulation

    from ..openmm.ais import switching_schedule
    from ..openmm.system import classify_omega_bonds
    from ..openmm.templates.md_stages import derive_seed
    from ..openmm.templates.rest2_scaling import TauSwitcher
    from .stage import solute_atom_indices

    ais, source_cfg, dynamics = run["ais"], run["ais_source"], run["dynamics"]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "AIS.log"

    topology_path = Path(args.topology)
    system_path = Path(args.system)
    source_path = Path(args.source or source_cfg["trajectory"])
    for path, what in ((topology_path, "-p topology"), (system_path, "-s system")):
        if not path.is_file():
            print(f"AIS: {what} {path} does not exist", file=sys.stderr)
            return 2
    if not source_path.is_file():
        print(f"AIS: the source trajectory {source_path} does not exist. AIS consumes an "
              f"equilibrium ensemble you have already produced; it does not generate one.",
              file=sys.stderr)
        return 2

    log = LogWriter(log_path, record_type="md-ais")
    log("md-openmm AIS")
    log("=" * 68)

    try:
        schedule = switching_schedule(
            tau_start=float(ais["tau_start"]), tau_end=float(ais["tau_end"]),
            switching_steps=int(ais["switching_steps"]),
            parameter_update_interval_steps=int(ais["parameter_update_interval_steps"]),
            observation_interval_steps=int(ais["observation_interval_steps"]),
            timestep_fs=float(dynamics["timestep_fs"]))

        pdb = PDBFile(str(topology_path))
        base = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
        if pdb.topology.getNumAtoms() != base.getNumParticles():
            raise SystemExit(f"{topology_path} has {pdb.topology.getNumAtoms()} atoms but "
                             f"{system_path} has {base.getNumParticles()} particles")
        implicit = not base.usesPeriodicBoundaryConditions()

        # No barostat during switching, ever. A pressure-volume term would enter the work and the
        # path would no longer measure what the Jarzynski/Crooks relations are written for.
        from ..openmm.templates.md_stages import count_barostats
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
        eligible = list(range(first, last + 1))

        log.heading("Source ensemble")
        log.field("trajectory", source_path)
        log.field("frames", f"{n_frames} total, {len(eligible)} eligible "
                            f"(frames {first}..{last} inclusive)")
        log.field("selection", source_cfg["selection"])

        chosen = choose_frames(eligible=eligible, count=int(ais["number_of_paths"]),
                               selection=source_cfg["selection"],
                               allow_repeats=bool(source_cfg["allow_repeated_frames"]),
                               seed=int(dynamics["seed"]))
        log.field("paths", f"{len(chosen)} starting from frames {chosen[:8]}"
                           + (" ..." if len(chosen) > 8 else ""))

        log.update(
            schedule=schedule,
            source={"trajectory": source_path.name, "frames_total": n_frames,
                    "first_frame": first, "last_frame": last,
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
        # the run is interrupted.
        with (out / "selected_source_frames.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path_index", "source_frame_index", "integrator_seed",
                             "velocity_seed"])
            for index, frame in enumerate(chosen):
                writer.writerow([index, frame,
                                 derive_seed(int(dynamics["seed"]), "ais", index, "integrator"),
                                 derive_seed(int(dynamics["seed"]), "ais", index, "velocity")])

        if args.check:
            log.heading("Preflight")
            log("  --check: schedule, source and Force layout validated; nothing was switched.")
            log.record["status"] = "checked"
            log.save()
            return 0

        # --- run the paths -------------------------------------------------------------------
        switcher = TauSwitcher(base, solute, excluded)
        taus = schedule["taus"]
        interval = schedule["parameter_update_interval_steps"]
        per_observation = schedule["updates_per_observation"]
        temperature = float(dynamics["temperature_K"])
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(
            unit.kilojoule_per_mole)

        wanted = _selected(args.paths, len(chosen))
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
                    if not implicit and chunk.unitcell_vectors is not None:
                        boxes = chunk.unitcell_vectors[local] * unit.nanometer
                    break
            if positions is None:
                raise SystemExit(f"could not read frame {frame} from {source_path}")

            system = switcher.prepared_system(taus[0])
            integrator = LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                float(dynamics["friction_per_ps"]) / unit.picosecond,
                float(dynamics["timestep_fs"]) * unit.femtosecond)
            integrator.setRandomNumberSeed(int(integrator_seed))
            platform_name = args.platform or dynamics.get("platform")
            if platform_name:
                plat = Platform.getPlatformByName(platform_name)
                properties = {}
                if platform_name == "CUDA":
                    properties = {"Precision": "mixed"}
                    if args.device is not None:
                        properties["DeviceIndex"] = str(args.device)
                simulation = Simulation(pdb.topology, system, integrator, plat, properties)
            else:
                simulation = Simulation(pdb.topology, system, integrator)
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
            handle = open(directory / OBSERVATIONS_DCD, "wb")
            dcd = DCDFile(handle, pdb.topology, float(dynamics["timestep_fs"]) * unit.femtosecond)

            def observe(observation, cumulative, incremental) -> None:
                state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
                dcd.writeModel(state.getPositions(),
                               periodicBoxVectors=(None if implicit
                                                   else state.getPeriodicBoxVectors()))
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
            handle.close()
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
                "platform": simulation.context.getPlatform().getName(),
            }
            marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
            completed.append(completion)
            log(f"  path {index:4d}: frame {frame}, {len(rows)} observations, "
                f"W = {cumulative:.4f} kJ/mol (reduced {beta * cumulative:.4f})")

        log.heading("Outputs")
        log.field("paths completed", f"{len(completed)} of {len(chosen)}")
        log.update(
            platform=openmm_platform_facts(),
            paths=completed,
            outputs={"selected_source_frames": file_facts(out / "selected_source_frames.csv",
                                                          relative_to=out)},
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
