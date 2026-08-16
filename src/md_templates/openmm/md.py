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

from . import runstate
from .config import resolve_chunk_plan, write_manifest
from .equilibration import (_apply_coords, _load_bundle, _make_simulation,
                            _scaled_system, _steps)

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Chunked production
# ---------------------------------------------------------------------------------------------
def _attach_chunk_reporters(sim, chunk_dir: Path, cfg: dict, dt_fs: float,
                            solute_atoms: Sequence[int]) -> None:
    from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter

    rcfg = cfg["production"]["report"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sim.reporters.clear()
    # all atoms: WRAPPED into the box, the usual convention for a solvated trajectory
    sim.reporters.append(
        DCDReporter(
            str(chunk_dir / "traj_all.dcd"),
            _steps(float(rcfg["all_atom_ps"]), dt_fs),
            enforcePeriodicBox=True,
        )
    )
    # solute only: NOT wrapped.  OpenMM's enforcePeriodicBox wraps whole MOLECULES, not atoms, so
    # bonds are never broken either way (verified: a solute straddling a box face comes back with a
    # max bonded distance of 0.153 nm under both settings) -- torsions would be safe regardless.
    # What wrapping does do is teleport the whole solute to the other side of the box whenever its
    # centre crosses a face, which breaks every analysis that reads the trajectory as continuous
    # (RMSD without re-imaging, diffusion, Cartesian TICA) and makes the structure jump around in a
    # viewer.  Unwrapped, the solute may drift far from the origin, which nothing here cares about.
    sim.reporters.append(
        DCDReporter(
            str(chunk_dir / "traj_solute.dcd"),
            _steps(float(rcfg["solute_ps"]), dt_fs),
            enforcePeriodicBox=False,
            atomSubset=list(int(i) for i in solute_atoms),
        )
    )
    sim.reporters.append(
        StateDataReporter(
            str(chunk_dir / "state.csv"), _steps(float(rcfg["state_ps"]), dt_fs),
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
        )
    )
    sim.reporters.append(
        CheckpointReporter(
            str(chunk_dir / "mid.chk"), _steps(float(rcfg["checkpoint_ps"]), dt_fs)
        )
    )


def _close_chunk(sim) -> None:
    for reporter in list(sim.reporters):
        for attr in ("_out", "_traj_file", "_dcd"):
            handle = getattr(reporter, attr, None)
            close = getattr(handle, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
    sim.reporters.clear()


def completed_prefix(run_dir: Path, n_chunks: int, *, what: str = "chunk") -> int:
    """Length of the CONTIGUOUS run of completed chunks starting at 0.

    ``len(glob("chunk_*/done.json"))`` was wrong: with chunks 0 and 2 complete and 1 missing it
    returns 2 and the run silently restarts at chunk 2, leaving a hole that no later analysis can
    see.  A gap means something failed or was deleted, and continuing past it fabricates a
    trajectory that was never contiguous -- so it is rejected rather than repaired.
    """
    run_dir = Path(run_dir)
    done = {int(q.parent.name.split("_")[1]) for q in run_dir.glob("chunk_*/done.json")}
    prefix = 0
    while prefix in done:
        prefix += 1
    stray = sorted(c for c in done if c > prefix)
    if stray:
        raise ValueError(
            f"{run_dir}: {what}s {stray} are complete but {prefix} is missing.  The completed "
            f"prefix is {prefix}, so continuing would leave a gap.  Delete the stray "
            f"{what} directories to restart cleanly from {prefix}, or restore the missing one."
        )
    for c in range(prefix):
        chk = run_dir / f"chunk_{c:04d}" / "end.chk"
        if not chk.exists():
            raise ValueError(
                f"{run_dir}: chunk {c} is marked done but {chk.name} is missing, so the run "
                "cannot be continued from it."
            )
    if prefix > n_chunks:
        raise ValueError(f"{run_dir}: {prefix} completed chunks exceeds the configured {n_chunks}")
    return prefix

#: Discovery has no lifetime ceiling: n_chunks is additional work, not a total.
_NO_CHUNK_CEILING = 1_000_000_000


def _assert_omega_classified(bundle: dict, *, omega_exclusion: bool = True) -> list:
    """Refuse to run production while any amide candidate is unclassified.

    An unclassified candidate means a solute C-N bond was found that neither the residue rule nor
    the RDKit rule could name.  Scaling it or not scaling it are different Hamiltonians, so
    guessing would silently change the estimand.  Review it and either extend
    ``rest2.proline_like_residues`` / ``rest2.max_proline_ring_size`` or fix the input chemistry.

    The block applies only when ``rest2.omega_exclusion`` is enabled. With the exclusion off every
    eligible torsion is scaled and no bond is treated specially, so an amide the classifier could
    not name changes nothing -- blocking there would refuse a run over a distinction the
    Hamiltonian no longer makes.
    """
    if not omega_exclusion:
        return []
    unknown = bundle.get("omega_unclassified_candidates") or []
    if unknown:
        lines = "\n".join(
            f"    bond {c['bond']}  {c['carbon_residue']}-{c['nitrogen_residue']}  {c['ambiguous']}"
            for c in unknown
        )
        raise ValueError(
            f"{len(unknown)} amide candidate(s) could not be classified as ordinary or "
            f"proline-like, so the REST2 Hamiltonian is not defined:\n{lines}\n"
            "Production is blocked until these are reviewed."
        )
    return [tuple(b) for b in bundle.get("omega_unscaled_bonds",
                                         bundle.get("omega_central_bonds", []))]


def run_md(cfg: dict, system_xml: Path, coords: Path, out_dir: Path, suffix: str) -> dict:
    """Stage (c): one free walker at ``production.md.scale_factor``, written in chunks.

    ``scale_factor = 1`` is the cold walker; anything below 1 is a REST2-scaled hot walker
    (``s = 0.25`` -> ``T_eff = 1200 K`` on the solute).  Nothing else differs between them, which is
    why there is one script rather than two.

    Chunking makes a long run restartable and analysable while it is still going: each chunk is a
    self-contained directory with its own trajectories, state log and end-of-chunk checkpoint.  A
    resumed run picks up at the first chunk without a ``done.json`` and does **not** re-minimise --
    re-minimising a continuing walker quenches the structure it had reached.
    """
    from openmm import unit

    out_dir = Path(out_dir)
    run_dir = out_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    mcfg = cfg["production"]["md"]
    ensemble = str(cfg["production"]["ensemble"]).upper()
    if ensemble != "NVT":
        raise ValueError(
            f"production.ensemble is {ensemble!r}; only NVT is implemented for the baseline.  "
            "See the ensemble note in docs/implementation/explicit_solvent/baseline_setups.md."
        )

    base, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])
    scale = float(mcfg["scale_factor"])
    label = str(mcfg.get("label") or ("cold" if scale == 1.0 else "hot"))
    omega = _assert_omega_classified(
        bundle, omega_exclusion=bool(cfg["rest2"]["omega_exclusion"]))
    system = _scaled_system(base, cfg, n_solute, scale, omega)

    dt_fs = float(cfg["integrator"]["timestep_fs"])
    sim = _make_simulation(pdb.topology, system, cfg, int(mcfg["seed"]))

    plan = resolve_chunk_plan(mcfg["n_chunks"], mcfg["chunk_ns"],
                              timestep_fs=dt_fs, where="production.md")
    chunk_steps = plan["steps_per_chunk"]
    # `n_chunks` is what THIS invocation adds, so a resume extends the run instead of finding it
    # already finished and doing nothing.
    chunks_this_invocation = plan["n_chunks"]
    for name, interval_ps in (("all_atom_ps", cfg["production"]["report"]["all_atom_ps"]),
                              ("solute_ps", cfg["production"]["report"]["solute_ps"])):
        if chunk_steps % _steps(float(interval_ps), dt_fs) != 0:
            raise ValueError(
                f"chunk_ns is not a multiple of report.{name}; frames would not align to chunk "
                "boundaries and the first frame of each chunk would drift."
            )

    start_chunk = completed_prefix(run_dir, _NO_CHUNK_CEILING)
    n_chunks = start_chunk + chunks_this_invocation
    if start_chunk == 0:
        origin = _apply_coords(sim, coords, require_velocities=True)
    else:
        gen = runstate.committed_generation(run_dir)
        if gen is None:
            raise ValueError(
                f"{run_dir} has completed chunks but no committed restart generation, so no state "
                "is known to be complete. Refusing to append output."
            )
        gdir = runstate.generation_dir(run_dir, gen)
        kind = runstate.load_restart(sim, gdir)
        if kind == "state":
            print("[md] restored from the portable State rather than a binary checkpoint: the "
                  "continuation is physically valid but NOT bitwise identical to an "
                  "uninterrupted run", flush=True)
        origin = {"coords": str(gdir), "kind": kind, "generation": gen}
    sim.currentStep = start_chunk * chunk_steps
    sim.context.setTime(start_chunk * chunk_steps * dt_fs * 1e-3 * unit.picosecond)

    print(
        f"[md:{label}] {n_chunks - start_chunk} chunk(s) of {mcfg['chunk_ns']} ns at "
        f"s = {scale:g}, dt = {dt_fs} fs, from {origin['coords']}",
        flush=True,
    )
    solute_atoms = list(range(n_solute))
    times = []
    for chunk in range(start_chunk, n_chunks):
        chunk_dir = run_dir / f"chunk_{chunk:04d}"
        _attach_chunk_reporters(sim, chunk_dir, cfg, dt_fs, solute_atoms)
        t0 = time.time()
        sim.step(chunk_steps)
        wall_s = time.time() - t0
        _close_chunk(sim)
        sim.saveCheckpoint(str(chunk_dir / "end.chk"))
        gdir = runstate.generation_dir(run_dir, chunk)
        runstate.save_restart(sim, gdir)
        runstate.commit_generation(run_dir, chunk, members=list(runstate.restart_members()),
                                   steps=int(sim.currentStep))
        (chunk_dir / "done.json").write_text(
            json.dumps(
                {
                    "chunk": chunk, "steps": chunk_steps, "ns": float(mcfg["chunk_ns"]),
                    "scale_factor": scale, "label": label,
                    "simulated_time_ps": sim.context.getState().getTime().value_in_unit(
                        unit.picosecond
                    ),
                    "wall_seconds": round(wall_s, 1),
                    "ns_per_day": round(float(mcfg["chunk_ns"]) * 86400.0 / wall_s, 2),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        times.append(wall_s)
        print(
            f"[md:{label}] chunk {chunk + 1}/{n_chunks} done in {wall_s / 3600:.2f} h "
            f"({float(mcfg['chunk_ns']) * 86400.0 / wall_s:.1f} ns/day)",
            flush=True,
        )

    info = {
        "suffix": suffix, "label": label, "scale_factor": scale,
        "n_chunks": n_chunks,
        "chunks_this_invocation": chunks_this_invocation,
        "lifetime_chunks_completed": n_chunks,
        "chunk_ns": plan["chunk_ns"],
        # derived from the plan, reported only
        "total_ns": plan["total_ns"], "timestep_fs": dt_fs,
        "input_coords": origin, "output_dir": str(run_dir),
        "mean_ns_per_day": (
            round(float(mcfg["chunk_ns"]) * 86400.0 / float(np.mean(times)), 2) if times else None
        ),
    }
    (out_dir / f"{suffix}_md.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, f"{suffix}_md", cfg, {"result": info})
    return info


