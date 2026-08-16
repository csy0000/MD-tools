from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform as _platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .config import rest2_ladder, write_manifest
from .equilibration import (_apply_coords, _load_bundle, _make_simulation,
                            _scaled_system, _steps)
from .md import (_assert_omega_classified, _attach_chunk_reporters, _close_chunk,
                 completed_prefix)

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Step 8 -- REST2 replica exchange
# ---------------------------------------------------------------------------------------------
def _pre_exchange_relaxation(simulations, coords: Path, cfg: dict, run_dir: Path,
                             scale_factors, temperature: float, dt_fs: float, seed: int) -> dict:
    """Propagate each replica under ITS OWN Hamiltonian before the first exchange, and discard it.

    Every replica starts from the same equilibrated coordinates, and those were equilibrated at
    ``s = 1``.  A replica at ``s < 1`` is therefore *not* in its own ensemble at step 0: its solute
    is suddenly softer than the configuration it holds.  Exchanging immediately would feed that
    transient straight into the swap statistics and into the trajectories.

    So each replica is run for ``production.remd.equilibration_ps`` with **no reporters and no
    exchange attempts**, and the result is thrown away.  This is *pre-exchange relaxation*, not a
    claim of equilibrium -- 10 ps settles the local solvent response to the scaled solute, nothing
    more.

    Restart-safe by construction: a completion record is written only after every replica's
    checkpoint is on disk, so a partial relaxation is detected and simply redone.  Redoing it is
    safe precisely because this runs only when no production output exists yet.
    """
    from openmm import unit

    relax_ps = float(cfg["production"]["remd"]["equilibration_ps"])
    relax_dir = run_dir / "_relaxation"
    record_path = relax_dir / "relaxation.json"
    checkpoints = [relax_dir / f"replica_{r:02d}.chk" for r in range(len(simulations))]

    if relax_ps <= 0:
        return {"performed": False, "reason": "production.remd.equilibration_ps = 0"}

    if record_path.exists() and all(c.exists() for c in checkpoints):
        for sim, chk in zip(simulations, checkpoints):
            sim.loadCheckpoint(str(chk))
        info = json.loads(record_path.read_text(encoding="utf-8"))
        info["reused"] = True
        print(f"[remd] reusing the stored {relax_ps:g} ps pre-exchange relaxation "
              f"({relax_dir}); it is not repeated", flush=True)
        return info

    if relax_dir.exists():
        print(f"[remd] {relax_dir} is incomplete (no completion record, or a checkpoint is "
              "missing); redoing the relaxation from scratch -- safe here because no production "
              "output exists yet", flush=True)
        for c in checkpoints:
            c.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)
    relax_dir.mkdir(parents=True, exist_ok=True)

    steps = _steps(relax_ps, dt_fs)
    per_replica, t0 = [], time.time()
    for r, sim in enumerate(simulations):
        _apply_coords(sim, coords)
        # deterministic and DISTINCT per replica: same positions, independent momenta
        vel_seed = int(seed) + 1000 + r
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, vel_seed)
        u0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.reporters.clear()                     # no production reporters during relaxation
        sim.step(steps)
        u1 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.saveCheckpoint(str(checkpoints[r]))
        per_replica.append({
            "replica": r, "scale_factor": float(scale_factors[r]),
            "velocity_seed": vel_seed,
            "potential_before_kj_mol": round(u0, 3), "potential_after_kj_mol": round(u1, 3),
            "delta_kj_mol": round(u1 - u0, 3),
            "checkpoint": str(checkpoints[r]),
        })
        print(f"[remd] relax replica {r:02d} s={scale_factors[r]:.4f}: "
              f"U {u0:.1f} -> {u1:.1f} kJ/mol", flush=True)

    info = {
        "performed": True, "reused": False,
        "discarded_ps": relax_ps, "discarded_steps_per_replica": steps,
        "timestep_fs": dt_fs, "temperature_k": temperature,
        "exchanges_attempted": 0, "reporters_attached": False,
        "wall_seconds": round(time.time() - t0, 1),
        "per_replica": per_replica,
        "note": "pre-exchange relaxation, DISCARDED; production time and step numbering are reset "
                "to zero afterwards, so production logs exclude it",
    }
    record_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def run_rest2_remd(cfg: dict, system_xml: Path, coords: Path, out_dir: Path,
                   suffix: str) -> dict:
    """Stage (d): REST2-REMD -- N replicas, neighbour exchange, chunked per replica.

    The ladder comes from ``production.remd.scale_factors`` if given, otherwise it is built from
    ``rest2.ladder`` (``s_cold``, ``s_hot``, ``n_rungs``, ``interp``).  Replica 0 is the physical
    rung (``s = 1``) by convention.

    The exchange criterion and the even/odd neighbour schedule are reused from
    ``md_templates.methods.md_run`` -- the same code the implicit-solvent references were produced
    with, so acceptance and round-trip statistics are comparable across solvents.

    ``<suffix>_exchange_attempts.csv`` carries both ``replica_i``/``replica_j`` (the schema
    ``md_run.run_remd`` writes) and ``i``/``j`` aliases, plus ``walker_at_replica_XX``, so
    ``analysis/remd_reliability.py`` computes the four reference-reliability probes (worst-pair
    acceptance, round trips, time-halves drift, per-replica spread) without a converter.

    All replicas share one process and one GPU.  ONE process per GPU: two processes sharing a card
    without CUDA MPS run ~3.7x slower each.
    """
    from openmm import unit


    out_dir = Path(out_dir)
    run_dir = out_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["production"]["remd"]
    if str(cfg["production"]["ensemble"]).upper() != "NVT":
        raise ValueError(
            "REST2-REMD in this baseline is NVT.  An NPT ladder needs a PV term in the "
            "acceptance criterion, which attempt_rest2_exchange does not include."
        )

    base, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])
    omega = _assert_omega_classified(bundle)

    requested = rcfg["scale_factors"]
    if requested is None:
        lad = cfg["rest2"]["ladder"]
        requested = rest2_ladder(
            float(lad["s_cold"]), float(lad["s_hot"]), int(lad["n_rungs"]), str(lad["interp"])
        )
    temperature = float(cfg["integrator"]["temperature_k"])
    scale_factors, t_eff = resolve_remd_scale_ladder(list(requested), temperature)
    n_replicas = len(scale_factors)
    dt_fs = float(cfg["integrator"]["timestep_fs"])

    simulations = []
    for r, sc in enumerate(scale_factors):
        system = _scaled_system(base, cfg, n_solute, sc, omega)
        simulations.append(_make_simulation(pdb.topology, system, cfg, int(rcfg["seed"]) + r))

    exchange_steps = _steps(float(rcfg["exchange_interval_ps"]), dt_fs)
    chunk_steps = _steps(float(rcfg["chunk_ns"]) * 1000.0, dt_fs)
    if chunk_steps % exchange_steps != 0:
        raise ValueError("remd.chunk_ns must be a whole number of exchange intervals")
    n_chunks = int(round(float(rcfg["total_ns_per_replica"]) / float(rcfg["chunk_ns"])))
    rounds_per_chunk = chunk_steps // exchange_steps

    replica_dirs = [run_dir / f"replica_{r:02d}" for r in range(n_replicas)]
    for d in replica_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # every replica must share the SAME completed prefix: they advance in lockstep between
    # exchange rounds, so a ragged set means one process died mid-round and the ladder's state is
    # not reconstructible from what survived
    prefixes = {r: completed_prefix(d, n_chunks) for r, d in enumerate(replica_dirs)}
    start_chunk = min(prefixes.values())
    if len(set(prefixes.values())) > 1:
        raise ValueError(
            f"replicas have different completed prefixes {prefixes}.  REST2 replicas advance in "
            "lockstep between exchange rounds, so this cannot be resumed consistently.  Delete "
            f"every replica's chunk_{start_chunk:04d} and later, then resume."
        )
    if start_chunk >= n_chunks:
        return {"status": "already-complete", "n_chunks": n_chunks, "output_dir": str(run_dir)}

    log_path = out_dir / f"{suffix}_exchange_attempts.csv"
    fieldnames = [
        "step", "time_ps", "phase", "replica_i", "replica_j", "i", "j",
        "effective_temperature_i_k", "effective_temperature_j_k",
        "scale_factor_i", "scale_factor_j",
        "energy_i_on_i_kj_mol", "energy_j_on_j_kj_mol",
        "energy_i_on_j_kj_mol", "energy_j_on_i_kj_mol",
        "delta_kj_mol", "log_acceptance", "accepted",
    ] + [f"walker_at_replica_{r:02d}" for r in range(n_replicas)]

    if start_chunk == 0:
        walker_by_replica = list(range(n_replicas))
        relaxation = _pre_exchange_relaxation(
            simulations, coords, cfg, run_dir, scale_factors, temperature, dt_fs,
            int(rcfg["seed"]),
        )
        log_mode = "w"
    else:
        for r, sim in enumerate(simulations):
            sim.loadCheckpoint(str(replica_dirs[r] / f"chunk_{start_chunk - 1:04d}" / "end.chk"))
        # the exchange log must end exactly at the production boundary the chunks reached, or a
        # resume would either duplicate rows or leave a silent hole in the swap history
        rows = list(csv.DictReader(log_path.open())) if log_path.exists() else []
        if not rows:
            raise ValueError(f"{log_path} has no rows; cannot restore the rung occupancy")
        boundary = start_chunk * chunk_steps
        keep = [r for r in rows if int(r["step"]) <= boundary]
        if len(keep) != len(rows):
            print(f"[remd] exchange log ran past the completed chunks ({len(rows)} rows, "
                  f"{len(keep)} at or before step {boundary}); truncating to the chunk boundary "
                  "so the resume neither duplicates nor skips a round", flush=True)
            with log_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(keep)
        if int(keep[-1]["step"]) != boundary:
            raise ValueError(
                f"{log_path} ends at step {keep[-1]['step']} but the completed chunks end at "
                f"{boundary}.  Refusing to resume from an inconsistent exchange history."
            )
        last = keep[-1]
        walker_by_replica = [int(last[f"walker_at_replica_{r:02d}"]) for r in range(n_replicas)]
        relaxation = {"performed": False, "reason": "resume: relaxation belongs to the fresh start"}
        log_mode = "a"

    for sim in simulations:
        sim.currentStep = start_chunk * chunk_steps
        sim.context.setTime(start_chunk * chunk_steps * dt_fs * 1e-3 * unit.picosecond)

    # exchange RNG: re-seeded on resume.  That is unbiased for Metropolis accept/reject, so a
    # continuation is faithful distributionally rather than bit-exact.
    rng = np.random.default_rng(int(rcfg["seed"]) + 977 * (start_chunk + 1))
    beta0 = 1.0 / (0.008314462618 * temperature)
    solute_atoms = list(range(n_solute))

    print(
        f"[remd] {n_replicas} replicas, s = {scale_factors}, T_eff = "
        f"{[round(t) for t in t_eff]} K, exchange every {rcfg['exchange_interval_ps']} ps, "
        f"{n_chunks - start_chunk} chunk(s) of {rcfg['chunk_ns']} ns to go",
        flush=True,
    )

    n_attempts = n_accepted = 0
    with log_path.open(log_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if log_mode == "w":
            writer.writeheader()
        phase = start_chunk * rounds_per_chunk
        for chunk in range(start_chunk, n_chunks):
            for r, sim in enumerate(simulations):
                _attach_chunk_reporters(
                    sim, replica_dirs[r] / f"chunk_{chunk:04d}", cfg, dt_fs, solute_atoms
                )
            t0 = time.time()
            for _ in range(rounds_per_chunk):
                for sim in simulations:
                    sim.step(exchange_steps)
                for i, j in exchange_pairs(n_replicas, phase):
                    result = attempt_rest2_exchange(simulations[i], simulations[j], beta0, rng)
                    n_attempts += 1
                    if result["accepted"]:
                        n_accepted += 1
                        walker_by_replica[i], walker_by_replica[j] = (
                            walker_by_replica[j], walker_by_replica[i],
                        )
                    step = simulations[0].currentStep
                    writer.writerow(
                        {
                            "step": step, "time_ps": step * dt_fs * 1e-3, "phase": phase % 2,
                            "replica_i": i, "replica_j": j, "i": i, "j": j,
                            "effective_temperature_i_k": t_eff[i],
                            "effective_temperature_j_k": t_eff[j],
                            "scale_factor_i": scale_factors[i],
                            "scale_factor_j": scale_factors[j],
                            **{k: v for k, v in result.items() if k != "accepted"},
                            "accepted": int(result["accepted"]),
                            **{
                                f"walker_at_replica_{r:02d}": w
                                for r, w in enumerate(walker_by_replica)
                            },
                        }
                    )
                phase += 1
                fh.flush()
            wall_s = time.time() - t0
            for r, sim in enumerate(simulations):
                _close_chunk(sim)
                cdir = replica_dirs[r] / f"chunk_{chunk:04d}"
                sim.saveCheckpoint(str(cdir / "end.chk"))
                (cdir / "done.json").write_text(
                    json.dumps(
                        {
                            "chunk": chunk, "replica": r, "scale_factor": scale_factors[r],
                            "effective_temperature_k": t_eff[r],
                            "wall_seconds": round(wall_s, 1),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(
                f"[remd] chunk {chunk + 1}/{n_chunks} in {wall_s / 3600:.2f} h; acceptance "
                f"{n_accepted / max(1, n_attempts):.3f}",
                flush=True,
            )

    info = {
        "suffix": suffix,
        "n_replicas": n_replicas,
        "scale_factors": scale_factors,
        "effective_temperatures_k": t_eff,
        "ladder_source": "production.remd.scale_factors" if rcfg["scale_factors"] is not None
                         else f"rest2.ladder ({cfg['rest2']['ladder']})",
        "exchange_interval_ps": float(rcfg["exchange_interval_ps"]),
        "pre_exchange_relaxation": relaxation,
        "n_chunks": n_chunks,
        "chunk_ns": float(rcfg["chunk_ns"]),
        "total_ns_per_replica": float(rcfg["total_ns_per_replica"]),
        "acceptance_fraction": n_accepted / max(1, n_attempts),
        "n_exchange_attempts": n_attempts,
        "exchange_log": str(log_path),
        "output_dir": str(run_dir),
    }
    (out_dir / f"{suffix}_rest2.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_rest2", cfg, {"result": info})
    return info


# ---------------------------------------------------------------------------------------------
# Replica exchange: neighbour schedule, Metropolis criterion, ladder validation
# ---------------------------------------------------------------------------------------------
def exchange_pairs(n_replicas: int, phase: int) -> list[tuple[int, int]]:
    """Return neighbor pairs for one REST2 exchange phase (even/odd alternation)."""
    start = phase % 2
    return [(i, i + 1) for i in range(start, n_replicas - 1, 2)]


def attempt_rest2_exchange(sim_i, sim_j, beta0: float, rng) -> dict:
    """Attempt a REST2 replica exchange between two simulations.

    Returns a dict with energy values, log_acceptance, and accepted flag.
    Exchange logic lifted from scripts/09_run_openmm_explicit_rest2.py.
    """
    from openmm import unit  # noqa: PLC0415

    state_i = sim_i.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    state_j = sim_j.context.getState(getPositions=True, getVelocities=True, getEnergy=True)

    pos_i = state_i.getPositions(asNumpy=True)
    pos_j = state_j.getPositions(asNumpy=True)
    vel_i = state_i.getVelocities(asNumpy=True)
    vel_j = state_j.getVelocities(asNumpy=True)
    box_i = state_i.getPeriodicBoxVectors()
    box_j = state_j.getPeriodicBoxVectors()

    E_ii = state_i.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    E_jj = state_j.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    sim_i.context.setPeriodicBoxVectors(*box_j)
    sim_i.context.setPositions(pos_j)
    E_ij = sim_i.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    sim_j.context.setPeriodicBoxVectors(*box_i)
    sim_j.context.setPositions(pos_i)
    E_ji = sim_j.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    delta = (E_ij + E_ji) - (E_ii + E_jj)
    log_accept = -beta0 * delta
    accepted = log_accept >= 0.0 or math.log(rng.random()) < log_accept

    if accepted:
        sim_i.context.setVelocities(vel_j)
        sim_j.context.setVelocities(vel_i)
    else:
        sim_i.context.setPeriodicBoxVectors(*box_i)
        sim_i.context.setPositions(pos_i)
        sim_j.context.setPeriodicBoxVectors(*box_j)
        sim_j.context.setPositions(pos_j)

    return {
        "energy_i_on_i_kj_mol": E_ii,
        "energy_j_on_j_kj_mol": E_jj,
        "energy_i_on_j_kj_mol": E_ij,
        "energy_j_on_i_kj_mol": E_ji,
        "delta_kj_mol": delta,
        "log_acceptance": log_accept,
        "accepted": accepted,
    }


def resolve_remd_scale_ladder(
    scale_factors: list[float],
    base_temperature_k: float = 300.0,
) -> tuple[list[float], list[float]]:
    """Validate and sort a user-supplied REST2 scale-factor list for REMD.

    Returns ``(scale_factors_descending, t_eff_descending)`` where
    ``scale_factors_descending[0] == 1.0`` (physical replica) by convention,
    matching the ordering produced by :func:`rest2_scale_factors`.

    Parameters
    ----------
    scale_factors:
        Raw scale factors (any order); must satisfy ``0 < s ≤ 1`` and
        ``len ≥ 2``.
    base_temperature_k:
        Bath / physical temperature in K.  ``T_eff = base_temperature_k / s``.

    Returns
    -------
    tuple of two lists:
      ``(scale_factors_desc, effective_temperatures_desc)``

    Raises
    ------
    ValueError
        If fewer than 2 replicas, or any scale factor is out of the ``(0, 1]``
        range.
    """
    if len(scale_factors) < 2:
        raise ValueError(
            f"REMD requires at least 2 replicas; got {len(scale_factors)} scale factors."
        )
    for s in scale_factors:
        if not (0.0 < s <= 1.0):
            raise ValueError(
                f"Every scale factor must be in the range (0, 1]; got {s!r}."
            )
    # Physical replica (s=1) must be replica 0 — descending order
    sorted_desc = sorted(scale_factors, reverse=True)
    t_eff_desc = [float(base_temperature_k / s) for s in sorted_desc]
    return sorted_desc, t_eff_desc


