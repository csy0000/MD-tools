from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform as _platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from . import runstate
from .config import resolve_chunk_plan, rest2_ladder, write_manifest
from .tau import map_replicas_to_devices
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


# ---------------------------------------------------------------------------------------------
# Durable exchange history
# ---------------------------------------------------------------------------------------------
#: Discovery has no lifetime ceiling: see the note where this is used.
_NO_CHUNK_CEILING = 1_000_000_000


def _rng_state(rng) -> dict:
    """Serialisable state of the exchange RNG, so a resume CONTINUES the sequence.

    Re-seeding from (seed, chunk) is unbiased for Metropolis accept/reject, but it restarts the
    stream: the same proposals recur across a resume boundary in a way that is not the sequence an
    uninterrupted run would have drawn. Persisting the generator state removes that difference.
    """
    return rng.bit_generator.state


def _restore_rng(state: Optional[dict]):
    """Rebuild the exchange RNG from a persisted state, or return None if there is nothing to use."""
    import numpy as _np

    if not isinstance(state, dict) or "bit_generator" not in state:
        return None
    rng = _np.random.default_rng()
    try:
        rng.bit_generator.state = state
    except (ValueError, KeyError, TypeError):
        return None
    return rng


def _assert_monotonic_attempts(rows: list, log_path) -> None:
    """Duplicate or out-of-order attempt indices are corruption, not something to average over.

    Silently double-counting them would inflate lifetime statistics in a way nothing downstream
    could detect, so this refuses instead.
    """
    seen = set()
    previous = -1
    for row in rows:
        raw = row.get("attempt_index")
        if raw in (None, ""):
            return                      # a log from before the index existed; nothing to check
        idx = int(raw)
        if idx in seen:
            raise ValueError(
                f"{log_path}: attempt_index {idx} appears more than once. The exchange history is "
                "corrupt; counting it would double-count that attempt."
            )
        if idx <= previous:
            raise ValueError(
                f"{log_path}: attempt_index {idx} follows {previous}, so the history is not "
                "monotonic. Refusing to continue from an inconsistent log."
            )
        seen.add(idx)
        previous = idx


def _lifetime_counts(log_path) -> tuple[int, int]:
    """(attempts, accepted) over the WHOLE durable log, for counters that must not restart at zero."""
    import csv as _csv
    from pathlib import Path as _Path

    path = _Path(log_path)
    if not path.is_file():
        return 0, 0
    attempts = accepted = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            attempts += 1
            accepted += int(str(row.get("accepted", "0")) in ("1", "True", "true"))
    return attempts, accepted


def summarise_exchange_log(run_dir) -> dict:
    """Rebuild lifetime statistics from the durable log alone.

    Idempotent by construction: it reads the log and nothing else, so deleting a summary and
    regenerating it gives the same numbers rather than adding to them.
    """
    import csv as _csv
    from collections import Counter
    from pathlib import Path as _Path

    run_dir = _Path(run_dir)
    path = run_dir / "exchange_attempts.csv"
    if not path.is_file():
        path = run_dir / "rest2_exchange_attempts.csv"
    if not path.is_file():
        return {"lifetime_exchange_attempts": 0, "lifetime_exchange_accepted": 0, "pairs": {}}
    rows = list(_csv.DictReader(path.open(encoding="utf-8", newline="")))
    _assert_monotonic_attempts(rows, path)
    attempts = Counter()
    accepts = Counter()
    for row in rows:
        key = f"{row.get('i')}-{row.get('j')}"
        attempts[key] += 1
        accepts[key] += int(str(row.get("accepted", "0")) in ("1", "True", "true"))
    total = sum(attempts.values())
    total_acc = sum(accepts.values())
    return {
        "source": str(path),
        "lifetime_exchange_attempts": total,
        "lifetime_exchange_accepted": total_acc,
        "lifetime_acceptance_fraction": total_acc / max(1, total),
        "pairs": {k: {"attempts": attempts[k], "accepted": accepts[k],
                      "acceptance": accepts[k] / max(1, attempts[k])}
                  for k in sorted(attempts)},
        "distinct_rounds": len({r.get("step") for r in rows}),
    }



def _resolve_replica_devices(cfg: dict, n_replicas: int) -> Optional[list[int]]:
    """Which device each replica runs on, or None when the platform has no devices.

    `production.device_indices` is the ordered list. When it is absent the single
    `production.device_index` is used for every replica, which reproduces the previous behaviour.
    The resolved mapping is returned so the caller can record it: a changed device list must be
    visible in the run manifest rather than silently re-dealt.
    """
    pcfg = cfg["production"]
    if str(pcfg["platform"]) not in ("CUDA", "OpenCL"):
        return None
    devices = pcfg.get("device_indices")
    if not devices:
        single = pcfg.get("device_index")
        return None if single is None else [int(single)] * n_replicas
    return map_replicas_to_devices(n_replicas, [int(d) for d in devices])


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
    # An empty suffix means "this IS the run directory". Writing into a subdirectory and moving the
    # results up afterwards is what broke resume: the second invocation looked for prior chunks in
    # the subdirectory, found none, and silently started over from chunk 0.
    run_dir = out_dir / suffix if suffix else out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["production"]["remd"]
    if str(cfg["production"]["ensemble"]).upper() != "NVT":
        raise ValueError(
            "REST2-REMD in this baseline is NVT.  An NPT ladder needs a PV term in the "
            "acceptance criterion, which attempt_rest2_exchange does not include."
        )

    base, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])
    omega = _assert_omega_classified(
        bundle, omega_exclusion=bool(cfg["rest2"]["omega_exclusion"]))

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

    # Replicas are dealt across the ordered device list. One replica per GPU is NOT hard-coded:
    # several replicas may share a device when they outnumber the devices, which is what makes a
    # ten-replica ladder possible on a four-GPU machine.
    device_map = _resolve_replica_devices(cfg, n_replicas)

    simulations = []
    for r, sc in enumerate(scale_factors):
        system = _scaled_system(base, cfg, n_solute, sc, omega)
        simulations.append(_make_simulation(pdb.topology, system, cfg, int(rcfg["seed"]) + r,
                                            device_index=(device_map[r] if device_map else None)))

    exchange_steps = _steps(float(rcfg["exchange_interval_ps"]), dt_fs)
    plan = resolve_chunk_plan(rcfg["n_chunks"], rcfg["chunk_ns"], timestep_fs=dt_fs,
                              where="production.remd",
                              exchange_interval_ps=float(rcfg["exchange_interval_ps"]))
    chunk_steps = plan["steps_per_chunk"]
    if chunk_steps % exchange_steps != 0:
        raise ValueError("remd.chunk_ns must be a whole number of exchange intervals")
    # `n_chunks` is what THIS invocation adds. A resume that reinterpreted it as a lifetime total
    # would do nothing at all once the first invocation had reached it.
    chunks_this_invocation = plan["n_chunks"]
    rounds_per_chunk = chunk_steps // exchange_steps

    replica_dirs = [run_dir / f"replica_{r:02d}" for r in range(n_replicas)]
    for d in replica_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # every replica must share the SAME completed prefix: they advance in lockstep between
    # exchange rounds, so a ragged set means one process died mid-round and the ladder's state is
    # not reconstructible from what survived
    # The commit record is the only thing that says what is durably finished. Read before any
    # decision about where to resume, and before a single output file is opened.
    committed = runstate.committed_record(run_dir)
    # The bound is only the "more chunks than configured" guard, and with n_chunks meaning
    # ADDITIONAL work there is no lifetime ceiling to check discovery against -- the completed
    # prefix is whatever the directory already holds.
    # THE COMMIT RECORD IS THE AUTHORITY, for every replica at once. A per-replica scan of
    # done.json can disagree with it, and the disagreement IS the crash window: a replica can have
    # written its chunk and its done.json while the generation that holds its state was never
    # committed.
    start_chunk = runstate.resume_boundary(run_dir)
    runstate.assert_committed_outputs(run_dir, start_chunk, required=["done.json", "end.chk"],
                                      subdirs=replica_dirs)
    # Every replica must be able to reach the committed boundary: they advance in lockstep between
    # exchange rounds, so a ragged set cannot be continued consistently.
    written = {r: completed_prefix(d, _NO_CHUNK_CEILING) for r, d in enumerate(replica_dirs)}
    behind = {r: n for r, n in written.items() if n < start_chunk}
    if behind:
        raise ValueError(
            f"the commit record says chunk {start_chunk - 1} is finished, but replicas {behind} "
            "have fewer completed chunks than that. The committed generation and the replica "
            "outputs disagree; refusing to resume."
        )
    quarantined = runstate.quarantine_uncommitted_tail(run_dir, start_chunk, subdirs=replica_dirs)
    if quarantined:
        print(f"[remd] {len(quarantined)} uncommitted chunk(s) moved to "
              f"{runstate.QUARANTINE_DIR}/: beyond the committed boundary, so not part of this "
              "run's history.", flush=True)
    n_chunks = start_chunk + chunks_this_invocation

    log_path = out_dir / (f"{suffix}_exchange_attempts.csv" if suffix
                          else "exchange_attempts.csv")
    fieldnames = [
        # A stable GLOBAL index across every invocation. Without it, "the third attempt" means
        # something different in each process that touched the run, and duplicate or non-monotonic
        # rows cannot be told apart from legitimate ones.
        "attempt_index",
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
        gen = runstate.committed_generation(run_dir)
        if gen is None:
            raise ValueError(
                f"{run_dir} has completed chunks but no committed restart generation, so there is "
                "no state that is known to be complete. Refusing to append output."
            )
        gdir = runstate.generation_dir(run_dir, gen)
        restored_from = {r: runstate.load_restart(sim, gdir, replica=r)
                         for r, sim in enumerate(simulations)}
        for r, sim in enumerate(simulations):
            st = sim.context.getState()
            runstate.assert_restart_consistent(
                loaded_step=int(sim.context.getStepCount()),
                loaded_time_ps=float(st.getTime().value_in_unit(unit.picosecond)),
                record=committed, timestep_fs=dt_fs,
            )
        if "state" in restored_from.values():
            print("[remd] one or more replicas restored from the portable State rather than a "
                  "binary checkpoint: the continuation is physically valid but NOT bitwise "
                  "identical to an uninterrupted run", flush=True)
        # the exchange log must end exactly at the production boundary the chunks reached, or a
        # resume would either duplicate rows or leave a silent hole in the swap history
        rows = list(csv.DictReader(log_path.open())) if log_path.exists() else []
        if not rows:
            raise ValueError(f"{log_path} has no rows; cannot restore the rung occupancy")
        # An attempt counts only once the restart generation holding its resulting state is
        # committed. Anything past that watermark is an uncommitted tail from a process that died
        # between writing rows and committing, and adopting it would double-count on the next run.
        watermark = (committed.get("attempts_committed")
                     if isinstance(committed.get("attempts_committed"), int) else None)
        if watermark is not None and len(rows) < watermark:
            raise ValueError(
                f"{log_path} holds {len(rows)} attempts but the commit record says {watermark} "
                "are committed. Rows that were committed have gone missing, so the exchange "
                "history is corrupt; refusing to continue."
            )
        if watermark is not None and len(rows) > watermark:
            print(f"[remd] exchange log has {len(rows)} rows but only {watermark} are committed; "
                  "discarding the uncommitted tail", flush=True)
            rows = rows[:watermark]
        _assert_monotonic_attempts(rows, log_path)
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
    # Prefer the persisted generator state so the exchange sequence continues rather than
    # restarting; fall back to the derived seed for a run committed before this was recorded.
    rng = _restore_rng(committed.get("rng_state")) if start_chunk else None
    rng_source = "restored"
    if rng is None:
        rng = np.random.default_rng(int(rcfg["seed"]) + 977 * (start_chunk + 1))
        rng_source = "reseeded" if start_chunk else "fresh"
    beta0 = 1.0 / (0.008314462618 * temperature)
    solute_atoms = list(range(n_solute))

    print(
        f"[remd] {n_replicas} replicas, s = {scale_factors}, T_eff = "
        f"{[round(t) for t in t_eff]} K, exchange every {rcfg['exchange_interval_ps']} ps, "
        f"{n_chunks - start_chunk} chunk(s) of {rcfg['chunk_ns']} ns to go",
        flush=True,
    )

    # LIFETIME counters, reconstructed from durable history. Starting them at zero is what made a
    # resumed run report only the last invocation's statistics.
    lifetime_attempts, lifetime_accepted = _lifetime_counts(log_path) if start_chunk else (0, 0)
    invocation_attempts = invocation_accepted = 0
    with log_path.open(log_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if log_mode == "w":
            writer.writeheader()
        # The scheduler phase is restored from the commit record when present; the derived value
        # is the fallback and agrees with it for an uninterrupted history.
        stored_phase = committed.get("exchange_phase")
        phase = int(stored_phase) if isinstance(stored_phase, int) and start_chunk else \
            start_chunk * rounds_per_chunk
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
                    lifetime_attempts += 1
                    invocation_attempts += 1
                    if result["accepted"]:
                        lifetime_accepted += 1
                        invocation_accepted += 1
                        walker_by_replica[i], walker_by_replica[j] = (
                            walker_by_replica[j], walker_by_replica[i],
                        )
                    step = simulations[0].currentStep
                    writer.writerow(
                        {
                            "attempt_index": lifetime_attempts - 1,
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
            fh.flush()
            os.fsync(fh.fileno())          # the log must be durable BEFORE the commit names it
            gdir = runstate.generation_dir(run_dir, chunk)
            members: list[str] = []
            for r, sim in enumerate(simulations):
                _close_chunk(sim)
                # Both restart forms, written to temporaries and renamed. Nothing points at this
                # generation until every replica's files exist.
                runstate.save_restart(sim, gdir, replica=r)
                members.extend(runstate.restart_members(r))
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
            # One atomic replacement makes the whole generation committed. Everything above is
            # written first; a crash before this line leaves the previous generation in force.
            runstate.commit_generation(
                run_dir, chunk, members=members,
                attempts_committed=lifetime_attempts,
                exchange_phase=phase,
                walker_by_replica=list(walker_by_replica),
                rng_state=_rng_state(rng),
                steps=int(simulations[0].currentStep),
            )
            print(
                f"[remd] chunk {chunk + 1}/{n_chunks} in {wall_s / 3600:.2f} h; acceptance "
                f"{lifetime_accepted / max(1, lifetime_attempts):.3f} (lifetime, "
                f"{lifetime_attempts} attempts)",
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
        "chunk_ns": plan["chunk_ns"],
        # derived from the plan, reported only
        "total_ns_per_replica": plan["total_ns"],
        # Both are reported and unambiguously labelled: lifetime describes the run directory,
        # invocation describes this process only.
        "lifetime_exchange_attempts": lifetime_attempts,
        "lifetime_exchange_accepted": lifetime_accepted,
        "lifetime_acceptance_fraction": lifetime_accepted / max(1, lifetime_attempts),
        "invocation_exchange_attempts": invocation_attempts,
        "invocation_exchange_accepted": invocation_accepted,
        "exchange_rng": rng_source,
        "restart_restored_from": (restored_from if start_chunk else None),
        "acceptance_fraction": lifetime_accepted / max(1, lifetime_attempts),
        "n_exchange_attempts": lifetime_attempts,
        "exchange_log": str(log_path),
        "output_dir": str(run_dir),
    }
    (out_dir / (f"{suffix}_rest2.json" if suffix else "rest2_summary.json")).write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_rest2" if suffix else "rest2", cfg, {"result": info})
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


