from __future__ import annotations

import copy
import csv
import hashlib
import json
import contextlib
import math
from concurrent.futures import ThreadPoolExecutor

from .seeds import as_openmm_seed, derive_seed, replica_purpose
from .ensembles import EXPLICIT_PRODUCTION_ENSEMBLE as ENSEMBLE_NPT, validate_ensemble
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
    t0 = time.time()

    # Each replica relaxes under its OWN tau Hamiltonian with no exchanges, so the replicas are
    # independent here exactly as they are between exchanges -- and this loop was serial for the
    # same reason the production loop was. On a six-replica ladder that is six times the
    # relaxation wall clock for no reason: 1 ns each, one GPU at a time.
    def _relax(r):
        sim = simulations[r]
        _apply_coords(sim, coords)
        # deterministic and DISTINCT per replica: same positions, independent momenta
        vel_seed = int(seed) + 1000 + r
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin,
                                               as_openmm_seed(int(vel_seed)))
        u0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.reporters.clear()                     # no production reporters during relaxation
        sim.step(steps)
        u1 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.saveCheckpoint(str(checkpoints[r]))
        return {
            "replica": r, "scale_factor": float(scale_factors[r]),
            "velocity_seed": vel_seed,
            "potential_before_kj_mol": round(u0, 3), "potential_after_kj_mol": round(u1, 3),
            "delta_kj_mol": round(u1 - u0, 3),
            "checkpoint": str(checkpoints[r]),
        }

    with replica_propagation_pool(len(simulations)) as pool:
        if pool is None:
            per_replica = [_relax(r) for r in range(len(simulations))]
        else:
            # Ordered by replica index, not by completion, so the record is reproducible.
            per_replica = list(pool.map(_relax, range(len(simulations))))
    for entry in per_replica:
        print(f"[remd] relax replica {entry['replica']:02d} "
              f"s={entry['scale_factor']:.4f}: "
              f"U {entry['potential_before_kj_mol']:.1f} -> "
              f"{entry['potential_after_kj_mol']:.1f} kJ/mol", flush=True)

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

    base, pdb, bundle = _load_bundle(system_xml)
    # This refused NPT outright, because the acceptance criterion carried no pV term. It does now
    # -- `attempt_rest2_exchange` evaluates all four reduced potentials u = beta*(U + p*V) -- so
    # the refusal has been replaced by the canonical check. Derived from the loaded System, so an
    # implicit ladder cannot be labelled NPT and an explicit one cannot claim a fixed box.
    solvation_mode = "explicit" if base.usesPeriodicBoundaryConditions() else "implicit"
    cfg["production"]["ensemble"] = validate_ensemble(
        cfg["production"].get("ensemble"), solvation_mode)
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

    # Explicit REST2 is NPT replica exchange: one physical pressure, one physical temperature, and
    # a Hamiltonian that differs between replicas. Each replica therefore carries its OWN barostat.
    # Sharing one seed across replicas would correlate their volume moves, which is precisely the
    # independence the exchange criterion assumes; the seeds are derived per replica and recorded.
    npt = (cfg["production"]["ensemble"] == ENSEMBLE_NPT)
    pressure_bar = float(cfg.get("equilibration", {}).get("pressure_bar", 1.0)) if npt else None
    barostat_seeds: list[int | None] = []

    simulations = []
    for r, sc in enumerate(scale_factors):
        system = _scaled_system(base, cfg, n_solute, sc, omega)
        if npt:
            from openmm import MonteCarloBarostat, unit as _u

            # Derived through the package's own seed machinery, not by arithmetic on the master
            # seed. `master + 100000 + r` overflowed OpenMM's 32-bit signed seed for a master of
            # 20260824003 and killed a six-replica ladder after its Contexts were built.
            # derive_seed is deterministic, nonzero and 32-bit safe by construction, and
            # replica_purpose keeps each replica's barostat stream independent of its integrator's.
            seed = derive_seed(int(rcfg["seed"]), replica_purpose(r, "barostat"))
            barostat = MonteCarloBarostat(pressure_bar * _u.bar,
                                          float(cfg["production"]["temperature_k"])
                                          if "temperature_k" in cfg["production"] else temperature,
                                          int(cfg.get("equilibration", {})
                                              .get("barostat_interval", 25)))
            barostat.setRandomNumberSeed(as_openmm_seed(seed))
            system.addForce(barostat)
            barostat_seeds.append(seed)
        else:
            barostat_seeds.append(None)
        simulations.append(_make_simulation(pdb.topology, system, cfg, int(rcfg["seed"]) + r,
                                            device_index=(device_map[r] if device_map else None)))

    # Exactly one barostat per Context, or none at all -- never a second one inherited from the
    # bundle's System, which would apply two independent volume moves per step.
    for r, sim in enumerate(simulations):
        n_baro = sum(1 for f in sim.system.getForces()
                     if "Barostat" in f.__class__.__name__)
        expected = 1 if npt else 0
        if n_baro != expected:
            raise ValueError(
                f"replica {r} has {n_baro} barostat(s), expected {expected} for ensemble "
                f"{cfg['production']['ensemble']!r}. Two barostats apply two independent volume "
                f"moves per step and sample no defined ensemble."
            )

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
        # The NPT half of the criterion. Recorded even when the pV terms cancel -- which they do
        # whenever every replica shares beta and p, as REST2 does -- because a log that omits them
        # cannot be used to check that they cancelled, only to assume it. `None` under implicit
        # solvent, where there is no volume rather than a volume of zero.
        "volume_i_nm3", "volume_j_nm3", "pv_i_kj_mol", "pv_j_kj_mol",
        "beta_i", "beta_j", "pressure_i_bar", "pressure_j_bar",
        # All four reduced potentials, so the acceptance arithmetic can be reproduced from the log
        # alone without re-running the simulation.
        "reduced_u_ii", "reduced_u_jj", "reduced_u_ij", "reduced_u_ji",
        "periodic",
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
    with log_path.open(log_mode, newline="") as fh, \
            replica_propagation_pool(n_replicas) as propagation_pool:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if log_mode == "w":
            writer.writeheader()
        # The scheduler phase is restored from the commit record when present; the derived value
        # is the fallback and agrees with it for an uninterrupted history.
        # One pool for the whole invocation. Creating it per round would pay thread setup 500
        # times; creating it per chunk would still pay it once per committed generation.
        # `None` when there is nothing to overlap, so a single-replica ladder takes the plain loop.
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
                propagate_replicas(simulations, exchange_steps, executor=propagation_pool)
                for i, j in exchange_pairs(n_replicas, phase):
                    result = attempt_rest2_exchange(simulations[i], simulations[j], beta0, rng,
                                             pressure_bar=pressure_bar)
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
        # Derived, reported only -- and derived from `n_chunks`, which is the LIFETIME count
        # (start_chunk + this invocation). It previously reported `plan["total_ns"]`, which is this
        # invocation's budget, so a resumed run recorded a lifetime chunk count beside a
        # single-invocation duration and the two silently disagreed.
        "total_ns_per_replica": n_chunks * plan["chunk_ns"],
        "invocation_ns_per_replica": chunks_this_invocation * plan["chunk_ns"],
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
@contextlib.contextmanager
def replica_propagation_pool(n_replicas: int):
    """A thread pool sized to the ladder, joined on every exit path.

    A context manager rather than a bare constructor so that an exception anywhere in the chunk
    loop still joins the workers instead of leaving threads attached to live CUDA Contexts. Yields
    `None` for a single-replica ladder, where there is nothing to overlap and the plain loop avoids
    the thread hand-off entirely.
    """
    if n_replicas < 2:
        yield None
        return
    pool = ThreadPoolExecutor(max_workers=n_replicas, thread_name_prefix="rest2-replica")
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def propagate_replicas(simulations, steps: int, executor=None) -> None:
    """Advance every replica by `steps`, concurrently when an executor is supplied.

    Replicas between exchanges are independent: each owns its System, Context, integrator and RNG
    stream, and none reads another's state until the exchange step that follows. So the loop is a
    fan-out with a barrier, not a sequence -- but it was written as `for sim in simulations:
    sim.step(...)`, which is blocking, so six replicas on six GPUs ran one at a time. Measured
    directly: exactly one of six devices at ~96 % at any instant, the rest holding a Context and
    idling. Six GPUs delivered one GPU's throughput.

    Threads rather than processes because OpenMM releases the GIL inside `step()` -- measured at
    2.80x on three devices against an ideal of 3x. A process pool would have to move Systems and
    States across a pipe at every exchange, which costs more than it saves.

    The barrier is the point: `map` is drained before returning, so no exchange is ever evaluated
    against a replica that is still integrating. Exceptions raised in a worker surface here, when
    the results are drained, rather than being swallowed into a thread.
    """
    if executor is None or len(simulations) < 2:
        for sim in simulations:
            sim.step(steps)
        return
    list(executor.map(lambda sim: sim.step(steps), simulations))


def exchange_pairs(n_replicas: int, phase: int) -> list[tuple[int, int]]:
    """Return neighbor pairs for one REST2 exchange phase (even/odd alternation)."""
    start = phase % 2
    return [(i, i + 1) for i in range(start, n_replicas - 1, 2)]


#: kJ/mol per (bar * nm^3). ``1 bar * 1 nm^3 = 1e5 Pa * 1e-27 m^3 = 1e-22 J``; times Avogadro and
#: divided by 1000 gives kJ/mol. Written as a named constant because a wrong pV conversion produces
#: an acceptance ratio that is merely *slightly* wrong, which is the hardest kind to notice.
BAR_NM3_TO_KJ_PER_MOL = 6.02214076e23 * 1e-22 / 1000.0     # = 0.0602214076


def reduced_potential(energy_kj_mol: float, beta: float,
                      pressure_bar: float | None = None,
                      volume_nm3: float | None = None) -> float:
    """The configurational reduced potential ``u = beta * (U + p*V)``.

    ``p*V`` is omitted when either the pressure or the volume is ``None``. That is not a shortcut
    for "assume zero": a nonperiodic system has no volume, so there is no ``pV`` term to compute,
    while a periodic system sampled at fixed volume has one that is constant and cancels. Passing
    ``None`` states the former; passing ``0.0`` would assert a real volume of zero.
    """
    u = energy_kj_mol
    if pressure_bar is not None and volume_nm3 is not None:
        u += pressure_bar * volume_nm3 * BAR_NM3_TO_KJ_PER_MOL
    return beta * u


def exchange_log_acceptance(u_ii: float, u_jj: float, u_ij: float, u_ji: float) -> float:
    """``-[u_i(x_j,V_j) + u_j(x_i,V_i) - u_i(x_i,V_i) - u_j(x_j,V_j)]``.

    All four reduced potentials enter. Writing the criterion in terms of energy differences alone
    is only valid when the two states share ``beta`` and ``p``; the four-term form stays correct
    when they do not, and reduces to the same number when they do.
    """
    return -((u_ij + u_ji) - (u_ii + u_jj))


def _volume_nm3(state) -> float | None:
    """Box volume, or ``None`` for a nonperiodic system -- which has no volume at all."""
    import numpy as np
    from openmm import unit

    box = state.getPeriodicBoxVectors(asNumpy=True)
    if box is None:
        return None
    vectors = np.asarray(box.value_in_unit(unit.nanometer), dtype=float)
    volume = float(abs(np.linalg.det(vectors)))
    # A nonperiodic Context still reports vectors; OpenMM's default is a unit box, and a genuine
    # simulation cell is never exactly 1 nm^3, so this is the cheapest reliable discriminator.
    return volume


def attempt_rest2_exchange(sim_i, sim_j, beta0: float, rng, *,
                           pressure_bar: float | None = None,
                           beta_i: float | None = None,
                           beta_j: float | None = None,
                           pressure_i_bar: float | None = None,
                           pressure_j_bar: float | None = None,
                           periodic: bool | None = None) -> dict:
    """Attempt a REST2 exchange between two replicas, in the general NPT form.

    REST2 is *Hamiltonian* replica exchange: every replica sits at the same physical temperature
    and, under NPT, the same physical pressure. What differs between replicas is the Hamiltonian,
    through the tau scaling. The "effective solute temperature" is an interpretation of that
    scaling, not a second thermostat.

    Because ``beta`` and ``p`` are shared, the ``pV`` terms cancel algebraically and the criterion
    collapses to the energy-difference form the previous implementation used. That cancellation is
    real, but it is a property of this protocol rather than of replica exchange, and relying on it
    silently is how a criterion survives into a protocol where it no longer holds. All four reduced
    potentials are therefore computed and recorded, and the cancellation is something the tests
    demonstrate rather than something the code assumes.

    At an exchange the two configurations are cross-evaluated **each under its own box**: a
    configuration carries its cell with it, and evaluating positions from one box inside another
    is a different physical state. On acceptance, positions, box vectors and velocities move
    together as one complete sampler state. Velocities are *not* rescaled -- the physical
    temperatures are identical, so rescaling would inject energy that the criterion never
    accounted for. On rejection both replicas are restored exactly.
    """
    from openmm import unit  # noqa: PLC0415

    beta_i = beta0 if beta_i is None else beta_i
    beta_j = beta0 if beta_j is None else beta_j
    p_i = pressure_bar if pressure_i_bar is None else pressure_i_bar
    p_j = pressure_bar if pressure_j_bar is None else pressure_j_bar

    state_i = sim_i.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    state_j = sim_j.context.getState(getPositions=True, getVelocities=True, getEnergy=True)

    pos_i = state_i.getPositions(asNumpy=True)
    pos_j = state_j.getPositions(asNumpy=True)
    vel_i = state_i.getVelocities(asNumpy=True)
    vel_j = state_j.getVelocities(asNumpy=True)
    box_i = state_i.getPeriodicBoxVectors()
    box_j = state_j.getPeriodicBoxVectors()

    is_periodic = (sim_i.system.usesPeriodicBoundaryConditions()
                   if periodic is None else bool(periodic))
    V_i = _volume_nm3(state_i) if is_periodic else None
    V_j = _volume_nm3(state_j) if is_periodic else None
    if not is_periodic:
        # No box means no pV term. Carrying a pressure into a nonperiodic exchange would multiply
        # it by a volume that does not exist.
        p_i = p_j = None

    E_ii = state_i.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    E_jj = state_j.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # Each configuration is evaluated under the OTHER Hamiltonian while keeping ITS OWN box.
    sim_i.context.setPeriodicBoxVectors(*box_j)
    sim_i.context.setPositions(pos_j)
    E_ij = sim_i.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)

    sim_j.context.setPeriodicBoxVectors(*box_i)
    sim_j.context.setPositions(pos_i)
    E_ji = sim_j.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)

    u_ii = reduced_potential(E_ii, beta_i, p_i, V_i)
    u_jj = reduced_potential(E_jj, beta_j, p_j, V_j)
    u_ij = reduced_potential(E_ij, beta_i, p_i, V_j)   # config j, evaluated by Hamiltonian i
    u_ji = reduced_potential(E_ji, beta_j, p_j, V_i)   # config i, evaluated by Hamiltonian j

    log_accept = exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji)
    accepted = log_accept >= 0.0 or math.log(rng.random()) < log_accept

    if accepted:
        # Positions and boxes are already crossed; velocities complete the swap. No rescaling:
        # both replicas are at the same physical temperature.
        sim_i.context.setVelocities(vel_j)
        sim_j.context.setVelocities(vel_i)
    else:
        sim_i.context.setPeriodicBoxVectors(*box_i)
        sim_i.context.setPositions(pos_i)
        sim_j.context.setPeriodicBoxVectors(*box_j)
        sim_j.context.setPositions(pos_j)

    def _pv(p, V):
        return None if (p is None or V is None) else p * V * BAR_NM3_TO_KJ_PER_MOL

    return {
        "energy_i_on_i_kj_mol": E_ii,
        "energy_j_on_j_kj_mol": E_jj,
        "energy_i_on_j_kj_mol": E_ij,
        "energy_j_on_i_kj_mol": E_ji,
        "volume_i_nm3": V_i,
        "volume_j_nm3": V_j,
        "pv_i_kj_mol": _pv(p_i, V_i),
        "pv_j_kj_mol": _pv(p_j, V_j),
        "beta_i": beta_i,
        "beta_j": beta_j,
        "pressure_i_bar": p_i,
        "pressure_j_bar": p_j,
        "reduced_u_ii": u_ii,
        "reduced_u_jj": u_jj,
        "reduced_u_ij": u_ij,
        "reduced_u_ji": u_ji,
        "periodic": is_periodic,
        "delta_kj_mol": (E_ij + E_ji) - (E_ii + E_jj),
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


