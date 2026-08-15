"""OpenMM MD and REMD simulation drivers consuming build_topology output.

All inputs come from a `build_topology` output directory
(system.prmtop / system.inpcrd / config.json).

Solvent is read from config.json; positions and box vectors come directly
from the inpcrd file — no RDKit atom-mapping required.

Default protocol
----------------
1. Energy minimization (maxIterations=10 000 steps)
2. NVT equilibration (100 ps, no barostat)
3. NPT equilibration (100 ps, MonteCarloBarostat — explicit solvent only)
4. Production MD (user-specified duration)

Reporters
---------
trajectory.dcd       — every traj_interval_ps  (default 10 ps)
state.csv            — every traj_interval_ps
trajectory.chk       — every checkpoint_interval_ps (default 100 ps)

AIS input contract
------------------
The config.json written by run_single_md must contain:
  temperature_k, timestep_fs, traj_interval_ps, friction_per_ps,
  solvent ("implicit"/"explicit"), n_solute_atoms,
  prmtop_path, inpcrd_path, rest2_scale_factor (single)
  or rest2_scale_factors + effective_temperatures_k + n_replicas (REMD).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from escort_ais.common.paths import data_dir, ensure_dir

BOLTZMANN_KJ_PER_MOL_K: float = 0.00831446261815324

_VALID_ENSEMBLES = frozenset({"nvt", "npt"})
_VALID_SOLVENTS = frozenset({"implicit", "explicit"})  # as stored in md_run config.json

# ── pure helpers (unit-testable without OpenMM) ──────────────────────────────


def steps_from_time(time_ps: float, timestep_fs: float) -> int:
    """Convert a time in ps to an integer step count given a timestep in fs."""
    return int(round(time_ps / (timestep_fs / 1000.0)))


def map_topology_solvent(topology_solvent: str) -> str:
    """Map build_topology solvent ('opc'/'gbn2') to AIS-style ('explicit'/'implicit')."""
    mapping = {"opc": "explicit", "gbn2": "implicit"}
    if topology_solvent not in mapping:
        raise ValueError(f"Unknown topology solvent '{topology_solvent}'. Expected 'opc' or 'gbn2'.")
    return mapping[topology_solvent]


def resolve_md_output_dir(
    topology_dir: Path,
    temperature_k: float,
    duration_ns: float,
    ensemble: str,
    scale_factor: float = 1.0,
    replica: Optional[int] = None,
) -> Path:
    """Return the default output directory for a single-MD run."""
    topology_dir = Path(topology_dir)
    name = topology_dir.name
    temp_label = f"{temperature_k:g}K"
    dur_label = f"{duration_ns:g}".replace(".", "p") + "ns"
    scale_label = "" if abs(scale_factor - 1.0) < 1e-9 else f"_s{scale_factor:g}".replace(".", "p")
    rep_label = "" if replica is None else f"_rep{replica:02d}"
    slug = f"{name}_{temp_label}_{ensemble}_{dur_label}{scale_label}{rep_label}"
    return data_dir() / "md" / slug


def validate_ensemble_solvent(ensemble: str, solvent: str) -> None:
    """Raise ValueError if NPT is requested with implicit solvent."""
    if ensemble not in _VALID_ENSEMBLES:
        raise ValueError(f"ensemble must be one of {sorted(_VALID_ENSEMBLES)}, got '{ensemble}'")
    if ensemble == "npt" and solvent == "implicit":
        raise ValueError("ensemble='npt' is not compatible with implicit solvent (no periodic box).")


def build_md_config_dict(
    topology_dir: Path,
    output_dir: Path,
    prmtop_path: Path,
    inpcrd_path: Path,
    solvent: str,
    topology_solvent: str,
    n_solute_atoms: int,
    temperature_k: float,
    duration_ns: float,
    ensemble: str,
    timestep_fs: float,
    traj_interval_ps: float,
    checkpoint_interval_ps: float,
    friction_per_ps: float,
    pressure_bar: float,
    platform: str,
    seed: int,
    minimize_steps: int,
    equilibration_nvt_ps: float,
    equilibration_npt_ps: float,
    rest2_scale_factor: float = 1.0,
    **extra,
) -> dict:
    """Assemble the config.json dict satisfying the AIS input contract."""
    cfg: dict = {
        # AIS-required keys
        "temperature_k": temperature_k,
        "timestep_fs": timestep_fs,
        "traj_interval_ps": traj_interval_ps,
        "friction_per_ps": friction_per_ps,
        "solvent": solvent,
        "topology_solvent": topology_solvent,
        "n_solute_atoms": n_solute_atoms,
        "prmtop_path": str(prmtop_path.resolve()),
        "inpcrd_path": str(inpcrd_path.resolve()),
        "rest2_scale_factor": rest2_scale_factor,
        # run metadata
        "topology_dir": str(Path(topology_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "duration_ns": duration_ns,
        "ensemble": ensemble,
        "platform": platform,
        "seed": seed,
        "minimize_steps": minimize_steps,
        "equilibration_nvt_ps": equilibration_nvt_ps,
        "equilibration_npt_ps": equilibration_npt_ps,
        "pressure_bar": pressure_bar,
    }
    cfg.update(extra)
    return cfg


# ── OpenMM helpers ────────────────────────────────────────────────────────────


def _build_simulation(topology, system, temperature_k, friction_per_ps, timestep_fs, platform_name):
    from openmm import LangevinMiddleIntegrator, Platform, unit  # noqa: PLC0415
    from openmm import app  # noqa: PLC0415

    integrator = LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_per_ps / unit.picosecond,
        (timestep_fs / 1000.0) * unit.picoseconds,
    )
    platform = Platform.getPlatformByName(platform_name)
    props: dict[str, str] = {}
    if platform_name in {"CUDA", "OpenCL"}:
        props["Precision"] = "mixed"
    return app.Simulation(topology, system, integrator, platform, props)


def _attach_reporters(
    simulation,
    output_dir: Path,
    traj_interval_steps: int,
    checkpoint_interval_steps: int,
    production_steps: int,
) -> None:
    from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter  # noqa: PLC0415

    simulation.reporters.append(
        DCDReporter(str(output_dir / "trajectory.dcd"), traj_interval_steps)
    )
    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "state.csv"),
            traj_interval_steps,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, speed=True, progress=True,
            remainingTime=True, totalSteps=production_steps, separator=",",
        )
    )
    try:
        simulation.reporters.append(
            CheckpointReporter(str(output_dir / "trajectory.chk"), checkpoint_interval_steps)
        )
    except Exception:
        pass


def _write_pdb(path: Path, topology, positions) -> None:
    from openmm import app  # noqa: PLC0415

    with path.open("w", encoding="utf-8") as fh:
        app.PDBFile.writeFile(topology, positions, fh)


def _pick_platform(preferred: Optional[str]) -> str:
    from escort_ais.systems.openmm_system import available_platform_names  # noqa: PLC0415

    available = set(available_platform_names())
    if preferred:
        if preferred not in available:
            raise ValueError(f"Platform '{preferred}' not available. Available: {sorted(available)}")
        return preferred
    for candidate in ["CUDA", "OpenCL", "CPU", "Reference"]:
        if candidate in available:
            return candidate
    raise RuntimeError("No OpenMM platform available.")


def _load_topology(topology_dir: Path):
    """Load config.json + prmtop/inpcrd from a build_topology output dir."""
    config = json.loads((topology_dir / "config.json").read_text(encoding="utf-8"))
    # Prefer new scale-labelled name; fall back to legacy system.prmtop
    for prmtop_name in ("system_scale_1p0.prmtop", "system.prmtop"):
        prmtop_path = topology_dir / prmtop_name
        if prmtop_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No prmtop found in {topology_dir}. "
            "Expected system_scale_1p0.prmtop or system.prmtop."
        )
    for inpcrd_name in ("system_scale_1p0.inpcrd", "system.inpcrd"):
        inpcrd_path = topology_dir / inpcrd_name
        if inpcrd_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No inpcrd found in {topology_dir}. "
            "Expected system_scale_1p0.inpcrd or system.inpcrd."
        )
    return config, prmtop_path, inpcrd_path


def _create_base_system(
    prmtop,
    topology_solvent: str,
    temperature_k: float,
    pressure_bar: float,
    ensemble: str,
    topology_dir: Optional[Path] = None,
    gb_radii: Optional[str] = None,
    prmtop_path: Optional[Path] = None,
):
    """Create the base OpenMM System for the given solvent and ensemble.

    For OPC explicit solvent, loads the pre-serialized ``system.xml`` written by
    ``build_topology`` when it is present in *topology_dir*.  This preserves OPC 4-point-water
    virtual sites, which the Amber prmtop round-trip silently drops.

    For GBn2 implicit solvent the build goes through
    :func:`escort_ais.systems.openmm_system.build_implicit_system`, which always takes the parmed
    route.  Requires *prmtop_path*.

    Two things that are easy to conflate, and were conflated in this docstring until 2026-07-31:

    * the **builder** — ``parmed.Structure.createSystem`` vs ``app.AmberPrmtopFile.createSystem``.
      These differ by ~16 kJ/mol on GBn2 and that is what must be matched across runs.
    * the **radii** — ``changeRadii(st, "mbondi3")``. A no-op wherever tleap already wrote mbondi3
      (both alanine prmtops), load-bearing for the sage/openff macrocycles (no radii stored).

    The old text here claimed ``AmberPrmtopFile.createSystem`` "uses its own built-in radii and
    ignores the prmtop RADII array".  That is false: the per-particle GB radii are bit-identical
    between the two builders (0 of 22 atoms differ on alanine).  The 16 kJ/mol sits entirely in
    ``CustomGBForce`` at identical radii.
    """
    from openmm import MonteCarloBarostat, XmlSerializer, unit  # noqa: PLC0415
    from openmm import app  # noqa: PLC0415

    if topology_solvent == "opc":
        system_xml_path = (topology_dir / "system.xml") if topology_dir is not None else None
        if system_xml_path is not None and system_xml_path.exists():
            system = XmlSerializer.deserialize(
                system_xml_path.read_text(encoding="utf-8")
            )
        else:
            # Fallback — will lack virtual sites for OPC; left for back-compat but warned.
            if topology_dir is not None:
                import warnings
                warnings.warn(
                    f"system.xml not found in {topology_dir}; rebuilding OPC system from "
                    "prmtop which drops virtual sites.  Re-run build_topology to fix.",
                    stacklevel=3,
                )
            system = prmtop.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True,
            )
        if ensemble == "npt":
            system.addForce(MonteCarloBarostat(pressure_bar * unit.bar, temperature_k * unit.kelvin, 50))
    else:  # gbn2 — GBn2 + mbondi3 via parmed is enforced; see openmm_system.build_implicit_system
        from escort_ais.systems.openmm_system import (  # noqa: PLC0415
            DEFAULT_GB_RADII, build_implicit_system)
        if prmtop_path is None:
            raise ValueError(
                "implicit-solvent builds require prmtop_path: GBn2 + mbondi3 is applied through "
                "parmed, which needs the prmtop file (the AmberPrmtopFile GB branch differs by "
                "~16 kJ/mol and is no longer the fallback)")
        system = build_implicit_system(prmtop_path, gb_radii=gb_radii or DEFAULT_GB_RADII)
    return system


# ── public simulation runner ─────────────────────────────────────────────────


def run_single_md(
    topology_dir: Path,
    *,
    duration_ns: float,
    temperature_k: float = 300.0,
    ensemble: Optional[str] = None,
    timestep_fs: float = 2.0,
    platform: Optional[str] = None,
    seed: int = 42,
    scale_factor: float = 1.0,
    scaled_system_xml: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    friction_per_ps: float = 1.0,
    pressure_bar: float = 1.0,
    minimize_steps: int = 10_000,
    equilibration_nvt_ps: float = 100.0,
    equilibration_npt_ps: float = 100.0,
    traj_interval_ps: float = 10.0,
    checkpoint_interval_ps: float = 100.0,
    start_positions_nm: Optional[np.ndarray] = None,
    gb_radii: Optional[str] = None,
) -> dict:
    """Run a single MD simulation from a build_topology output directory.

    Parameters
    ----------
    topology_dir:
        Path to a build_topology output dir (system.prmtop/inpcrd/config.json).
    duration_ns:
        Production simulation length in nanoseconds. **Required** — no default.
    ensemble:
        'nvt' or 'npt'. Defaults to 'npt' for explicit solvent, 'nvt' for implicit.
    scale_factor:
        REST2 scale factor applied to solute atoms (1.0 = physical/unscaled).
    scaled_system_xml:
        If given, deserialize a pre-built System XML (from build_rest2_systems)
        instead of applying scale_factor on-the-fly.

    Returns
    -------
    dict  Summary of the run (output paths + final energy).
    """
    from openmm import XmlSerializer, app, unit  # noqa: PLC0415
    from escort_ais.systems.openmm_system import build_rest2_scaled_system  # noqa: PLC0415

    topology_dir = Path(topology_dir)
    topo_config, prmtop_path, inpcrd_path = _load_topology(topology_dir)

    topology_solvent = topo_config["solvent"]  # "opc" or "gbn2"
    solvent = map_topology_solvent(topology_solvent)  # "explicit" or "implicit"
    n_solute_atoms = int(topo_config["n_solute_atoms"])

    if ensemble is None:
        ensemble = "npt" if solvent == "explicit" else "nvt"
    validate_ensemble_solvent(ensemble, solvent)

    platform_name = _pick_platform(platform)

    if output_dir is None:
        output_dir = resolve_md_output_dir(
            topology_dir, temperature_k, duration_ns, ensemble, scale_factor
        )
    output_dir = ensure_dir(Path(output_dir))

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))

    base_system = _create_base_system(prmtop, topology_solvent, temperature_k, pressure_bar, ensemble,
                                      topology_dir=topology_dir, gb_radii=gb_radii, prmtop_path=prmtop_path)

    if scaled_system_xml is not None:
        system = XmlSerializer.deserialize(Path(scaled_system_xml).read_text(encoding="utf-8"))
    elif abs(scale_factor - 1.0) > 1e-9:
        system = build_rest2_scaled_system(base_system, np.arange(n_solute_atoms), scale_factor)
    else:
        system = base_system

    simulation = _build_simulation(
        prmtop.topology, system, temperature_k, friction_per_ps, timestep_fs, platform_name
    )
    if start_positions_nm is not None:
        simulation.context.setPositions(start_positions_nm * unit.nanometer)
    else:
        simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    t_start = time.time()

    # 1. Minimization
    simulation.minimizeEnergy(maxIterations=minimize_steps)

    # 2. NVT equilibration
    nvt_steps = steps_from_time(equilibration_nvt_ps, timestep_fs)
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    if nvt_steps > 0:
        simulation.step(nvt_steps)

    # 3. NPT equilibration (explicit only, and only when the production ensemble is NPT)
    npt_steps = 0
    if ensemble == "npt" and topology_solvent == "opc":
        npt_steps = steps_from_time(equilibration_npt_ps, timestep_fs)
        if npt_steps > 0:
            simulation.step(npt_steps)

    # Save start_structure after equilibration
    equil_state = simulation.context.getState(getPositions=True)
    _write_pdb(output_dir / "start_structure.pdb", prmtop.topology, equil_state.getPositions())

    # 4. Production
    production_steps = steps_from_time(duration_ns * 1000.0, timestep_fs)
    traj_steps = max(1, steps_from_time(traj_interval_ps, timestep_fs))
    chk_steps = max(1, steps_from_time(checkpoint_interval_ps, timestep_fs))

    _attach_reporters(simulation, output_dir, traj_steps, chk_steps, production_steps)
    simulation.step(production_steps)

    final_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    simulation.saveState(str(output_dir / "final_state.xml"))
    simulation.saveCheckpoint(str(output_dir / "final_checkpoint.chk"))
    _write_pdb(output_dir / "final_structure.pdb", prmtop.topology, final_state.getPositions())

    wall_time_s = time.time() - t_start

    # Write config.json
    config_dict = build_md_config_dict(
        topology_dir=topology_dir,
        output_dir=output_dir,
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        solvent=solvent,
        topology_solvent=topology_solvent,
        n_solute_atoms=n_solute_atoms,
        temperature_k=temperature_k,
        duration_ns=duration_ns,
        ensemble=ensemble,
        timestep_fs=timestep_fs,
        traj_interval_ps=traj_interval_ps,
        checkpoint_interval_ps=checkpoint_interval_ps,
        friction_per_ps=friction_per_ps,
        pressure_bar=pressure_bar,
        platform=platform_name,
        seed=seed,
        minimize_steps=minimize_steps,
        equilibration_nvt_ps=equilibration_nvt_ps,
        equilibration_npt_ps=equilibration_npt_ps,
        rest2_scale_factor=scale_factor,
        production_steps=production_steps,
        traj_interval_steps=traj_steps,
        checkpoint_interval_steps=chk_steps,
        wall_time_s=wall_time_s,
        # Recorded ALWAYS for implicit solvent, not only when overridden: the pre-2026-07-31 runs
        # omitted the key whenever it was left at the default, so the manifest could not say which
        # GB branch was used and only a direct energy comparison could settle it.
        **({"gb_radii": gb_radii or "mbondi3", "gb_model": "GBn2"}
           if topology_solvent == "gbn2" else {}),
    )
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "platform": platform_name,
        "rest2_scale_factor": scale_factor,
        "production_steps": production_steps,
        "duration_ns": duration_ns,
        "final_potential_energy_kj_mol": final_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        ),
        "wall_time_s": wall_time_s,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ── REMD helpers ─────────────────────────────────────────────────────────────


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


def run_remd(
    topology_dir: Path,
    *,
    duration_ns: float,
    temperature_k: float = 300.0,
    ensemble: Optional[str] = None,
    exchange_interval_ps: float = 1.0,
    timestep_fs: float = 2.0,
    platform: Optional[str] = None,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    friction_per_ps: float = 1.0,
    pressure_bar: float = 1.0,
    minimize_steps: int = 10_000,
    equilibration_nvt_ps: float = 100.0,
    equilibration_npt_ps: float = 100.0,
    traj_interval_ps: float = 10.0,
    checkpoint_interval_ps: float = 100.0,
) -> dict:
    """Run REST2 REMD from a build_topology output directory.

    Requires topology_dir/rest2/ladder.json produced by build_rest2_systems.

    Returns
    -------
    dict  Summary including acceptance fraction and per-replica info.
    """
    from openmm import XmlSerializer, app, unit  # noqa: PLC0415
    from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter  # noqa: PLC0415

    topology_dir = Path(topology_dir)
    ladder_path = topology_dir / "rest2" / "ladder.json"
    if not ladder_path.exists():
        raise FileNotFoundError(
            f"rest2/ladder.json not found in {topology_dir}. "
            "Run build_rest2_systems(..., mode='remd') first."
        )

    topo_config, prmtop_path, inpcrd_path = _load_topology(topology_dir)
    topology_solvent = topo_config["solvent"]
    solvent = map_topology_solvent(topology_solvent)
    n_solute_atoms = int(topo_config["n_solute_atoms"])

    if ensemble is None:
        ensemble = "npt" if solvent == "explicit" else "nvt"
    validate_ensemble_solvent(ensemble, solvent)

    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    n_replicas = len(ladder)
    scale_factors = [entry["scale_factor"] for entry in ladder]
    effective_temperatures_k = [entry["effective_temperature_k"] for entry in ladder]

    platform_name = _pick_platform(platform)

    if output_dir is None:
        dur_label = f"{duration_ns:g}".replace(".", "p") + "ns"
        output_dir = data_dir() / "md" / f"{topology_dir.name}_{n_replicas}rep_{temperature_k:g}K_remd_{dur_label}"
    output_dir = ensure_dir(Path(output_dir))

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))

    beta0 = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * temperature_k)

    # Build one simulation per replica from its XML
    simulations = []
    for entry in ladder:
        xml_path = Path(entry["system_xml"])
        system = XmlSerializer.deserialize(xml_path.read_text(encoding="utf-8"))
        # Guard: OPC systems must carry VirtualSite definitions.  The Amber prmtop round-trip
        # silently drops them.  If they are absent, the M-sites become free massless particles
        # and the very first minimize/step produces NaN coordinates.  Fail loudly here so the
        # error is obvious rather than surfacing as a mysterious NaN many hours later.
        if topology_solvent == "opc":
            n_vs = sum(1 for i in range(system.getNumParticles()) if system.isVirtualSite(i))
            if n_vs == 0:
                raise RuntimeError(
                    f"Replica {entry['replica']} system loaded from {xml_path} has 0 virtual "
                    "sites but solvent is OPC (4-point water).  The REST2 replica XMLs were "
                    "likely built from a prmtop that lost the VirtualSite frames.  Please "
                    "delete the topology directory and re-run build_topology + build_rest2_systems "
                    "with the current code to regenerate system.xml and the replica XMLs."
                )
        # Remove any existing barostat; we add ours
        if ensemble == "npt" and topology_solvent == "opc":
            from openmm import MonteCarloBarostat  # noqa: PLC0415
            system.addForce(MonteCarloBarostat(pressure_bar * unit.bar, temperature_k * unit.kelvin, 50))
        sim = _build_simulation(
            prmtop.topology, system, temperature_k, friction_per_ps, timestep_fs, platform_name
        )
        simulations.append(sim)

    # Minimize replica 0 and broadcast positions
    ref = simulations[0]
    ref.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        ref.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    ref.minimizeEnergy(maxIterations=minimize_steps)
    min_state = ref.context.getState(getPositions=True)
    min_pos = min_state.getPositions()
    min_box = min_state.getPeriodicBoxVectors()

    _write_pdb(output_dir / "start_structure_minimized.pdb", prmtop.topology, min_pos)

    replica_dirs = []
    for i, sim in enumerate(simulations):
        rdir = ensure_dir(output_dir / f"replica_{i:02d}")
        replica_dirs.append(rdir)
        if min_box is not None:
            sim.context.setPeriodicBoxVectors(*min_box)
        sim.context.setPositions(min_pos)
        sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed + i)

    # NVT equilibration
    nvt_steps = steps_from_time(equilibration_nvt_ps, timestep_fs)
    if nvt_steps > 0:
        for sim in simulations:
            sim.step(nvt_steps)

    # NPT equilibration (explicit only)
    npt_steps = steps_from_time(equilibration_npt_ps, timestep_fs) if ensemble == "npt" else 0
    if npt_steps > 0:
        for sim in simulations:
            sim.step(npt_steps)

    production_steps = steps_from_time(duration_ns * 1000.0, timestep_fs)
    traj_steps = max(1, steps_from_time(traj_interval_ps, timestep_fs))
    chk_steps = max(1, steps_from_time(checkpoint_interval_ps, timestep_fs))
    exchange_steps = max(1, steps_from_time(exchange_interval_ps, timestep_fs))

    for i, (sim, rdir) in enumerate(zip(simulations, replica_dirs)):
        sim.reporters.append(DCDReporter(str(rdir / "trajectory.dcd"), traj_steps))
        sim.reporters.append(
            StateDataReporter(
                str(rdir / "state.csv"), traj_steps,
                step=True, time=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True, speed=True, progress=True,
                remainingTime=True, totalSteps=production_steps, separator=",",
            )
        )
        try:
            sim.reporters.append(CheckpointReporter(str(rdir / "trajectory.chk"), chk_steps))
        except Exception:
            pass

    # Write top-level config
    config_dict = build_md_config_dict(
        topology_dir=topology_dir,
        output_dir=output_dir,
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        solvent=solvent,
        topology_solvent=topology_solvent,
        n_solute_atoms=n_solute_atoms,
        temperature_k=temperature_k,
        duration_ns=duration_ns,
        ensemble=ensemble,
        timestep_fs=timestep_fs,
        traj_interval_ps=traj_interval_ps,
        checkpoint_interval_ps=checkpoint_interval_ps,
        friction_per_ps=friction_per_ps,
        pressure_bar=pressure_bar,
        platform=platform_name,
        seed=seed,
        minimize_steps=minimize_steps,
        equilibration_nvt_ps=equilibration_nvt_ps,
        equilibration_npt_ps=equilibration_npt_ps,
        # REMD-specific (also satisfies AIS legacy replica-ladder path)
        n_replicas=n_replicas,
        rest2_scale_factors=scale_factors,
        effective_temperatures_k=effective_temperatures_k,
        exchange_interval_ps=exchange_interval_ps,
        production_steps=production_steps,
        traj_interval_steps=traj_steps,
        checkpoint_interval_steps=chk_steps,
        exchange_interval_steps=exchange_steps,
    )
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    # Production with exchange
    exchange_log_path = output_dir / "exchange_attempts.csv"
    fieldnames = [
        "step", "phase", "replica_i", "replica_j",
        "effective_temperature_i_k", "effective_temperature_j_k",
        "scale_factor_i", "scale_factor_j",
        "energy_i_on_i_kj_mol", "energy_j_on_j_kj_mol",
        "energy_i_on_j_kj_mol", "energy_j_on_i_kj_mol",
        "delta_kj_mol", "log_acceptance", "accepted",
    ] + [f"walker_at_replica_{r:02d}" for r in range(n_replicas)]

    t_start = time.time()
    attempted = 0
    accepted_count = 0
    walker_by_replica = list(range(n_replicas))
    rng = np.random.default_rng(seed)
    phase = 0

    with exchange_log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        remaining = production_steps
        while remaining > 0:
            block = min(exchange_steps, remaining)
            for sim in simulations:
                sim.step(block)
            current_step = simulations[0].currentStep

            for ri, rj in exchange_pairs(n_replicas, phase):
                result = attempt_rest2_exchange(simulations[ri], simulations[rj], beta0, rng)
                attempted += 1
                if result["accepted"]:
                    accepted_count += 1
                    walker_by_replica[ri], walker_by_replica[rj] = (
                        walker_by_replica[rj],
                        walker_by_replica[ri],
                    )
                row = {
                    "step": current_step, "phase": phase,
                    "replica_i": ri, "replica_j": rj,
                    "effective_temperature_i_k": effective_temperatures_k[ri],
                    "effective_temperature_j_k": effective_temperatures_k[rj],
                    "scale_factor_i": scale_factors[ri],
                    "scale_factor_j": scale_factors[rj],
                    **result,
                    **{f"walker_at_replica_{r:02d}": w for r, w in enumerate(walker_by_replica)},
                }
                writer.writerow(row)

            remaining -= block
            phase += 1

    wall_time_s = time.time() - t_start

    # Finalize replicas
    final_replica_summaries = []
    for i, (sim, rdir, t_eff, sf) in enumerate(
        zip(simulations, replica_dirs, effective_temperatures_k, scale_factors)
    ):
        final_state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        sim.saveState(str(rdir / "final_state.xml"))
        sim.saveCheckpoint(str(rdir / "final_checkpoint.chk"))
        _write_pdb(rdir / "final_structure.pdb", prmtop.topology, final_state.getPositions())
        final_replica_summaries.append({
            "replica_index": i,
            "effective_temperature_k": t_eff,
            "rest2_scale_factor": sf,
            "completed_steps": sim.currentStep,
            "final_potential_energy_kj_mol": final_state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            ),
        })

    summary = {
        "output_dir": str(output_dir),
        "platform": platform_name,
        "n_replicas": n_replicas,
        "temperature_k": temperature_k,
        "effective_temperatures_k": effective_temperatures_k,
        "rest2_scale_factors": scale_factors,
        "production_steps": production_steps,
        "duration_ns": duration_ns,
        "attempted_exchanges": attempted,
        "accepted_exchanges": accepted_count,
        "acceptance_fraction": (accepted_count / attempted) if attempted else 0.0,
        "wall_time_s": wall_time_s,
        "replicas": final_replica_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
