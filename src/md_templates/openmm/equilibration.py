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

from .config import write_manifest
from .system import build_system, initial_structure, protonate
from .solvation import solvate
from .hashing import sha256_text

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Integrator / Simulation helpers
# ---------------------------------------------------------------------------------------------
def make_integrator(cfg: dict, seed: int, timestep_fs: Optional[float] = None):
    """Return a seeded integrator.

    ``setRandomNumberSeed`` MUST be called before the ``Context`` exists -- afterwards OpenMM
    silently ignores it.  That is a project-wide engine invariant, and it is why this function
    returns the integrator rather than a Simulation built around one.
    """
    import openmm
    from openmm import unit

    icfg = cfg["integrator"]
    dt = float(icfg["timestep_fs"] if timestep_fs is None else timestep_fs) * unit.femtosecond
    temperature = float(icfg["temperature_k"]) * unit.kelvin
    friction = float(icfg["friction_per_ps"]) / unit.picosecond
    kind = str(icfg["kind"]).lower()

    if kind in ("leapfrog-langevin", "langevin", "leapfrog"):
        integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    elif kind in ("langevin-middle", "middle", "baoab"):
        integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
    elif kind == "verlet":
        integrator = openmm.VerletIntegrator(dt)
    else:
        raise ValueError(f"unknown integrator.kind {icfg['kind']!r}")
    integrator.setRandomNumberSeed(int(seed) % 2_147_483_647)
    return integrator


def _platform_and_properties(cfg: dict, device_index: Optional[int] = None):
    """Resolve the OpenMM platform and its properties.

    `device_index` overrides the configured one. REST2 uses it to place each replica on its own
    device: without a per-simulation override every replica would land on the same GPU, which is
    correct only when there is one.
    """
    import openmm

    pcfg = cfg["production"]
    name = str(pcfg["platform"])
    plat = openmm.Platform.getPlatformByName(name)
    props: dict[str, str] = {}
    if name in ("CUDA", "OpenCL"):
        # The platform default is single precision: ~25 % faster and it silently changes energies.
        props["Precision"] = str(pcfg["precision"])
        chosen = device_index if device_index is not None else pcfg["device_index"]
        if chosen is not None:
            props["DeviceIndex"] = str(chosen)
    return plat, props


#: Stages that do no dynamics. Minimisation walks downhill on the potential; it integrates no
#: equations of motion, so the platform changes only how long it takes, not what is sampled.
NON_DYNAMICS_STAGES = ("min",)

#: Platforms that use the GPU. "Reference" and "CPU" are correct but far too slow for production
#: dynamics -- a 5 ns explicit segment that would take minutes on CUDA takes days on CPU.
ACCELERATED_PLATFORMS = ("CUDA", "HIP", "OpenCL")


def assert_dynamics_platform(cfg: dict, stage: str, required: str | None = None) -> None:
    """Refuse to integrate dynamics on a platform nobody chose.

    `Platform.getPlatformByName` already raises when a named platform is absent, so a run
    configured for CUDA cannot silently land on the CPU. The gap this closes is upstream of that:
    `_runtime_cfg` defaulted `execution.platform` to "CPU", so a payload that lost the key would run
    correct dynamics on the wrong device at a thousandth of the speed and report success.

    An explicitly stated "CPU" is a legitimate choice -- the shipped smoke test and the CPU
    integration gate both want it -- so this does not hard-require a GPU. What it refuses is a
    platform that arrived by default rather than by decision.

    `required` (or `MD_REQUIRE_DYNAMICS_PLATFORM` in the environment) additionally pins the
    platform, so a campaign that must run on GPUs can assert that rather than trust it.

    Minimisation is exempt by name rather than by inspecting the step count, because "zero steps"
    is also what an interrupted dynamics stage looks like.
    """
    import os

    if stage in NON_DYNAMICS_STAGES:
        return
    pcfg = cfg["production"]
    name = pcfg.get("platform")
    if not name:
        raise RuntimeError(
            f"stage {stage!r} integrates dynamics but no execution.platform was resolved. The "
            f"protocol configuration must state it; it is not defaulted, because the default would "
            f"decide where the science runs."
        )
    required = required or os.environ.get("MD_REQUIRE_DYNAMICS_PLATFORM") or None
    if required and str(name) != str(required):
        raise RuntimeError(
            f"stage {stage!r} integrates dynamics on the {name!r} platform, but this run requires "
            f"{required!r} (MD_REQUIRE_DYNAMICS_PLATFORM). Minimisation is exempt; every stage that "
            f"integrates equations of motion is not."
        )


def actual_platform(simulation) -> dict:
    """What the Context is really running on, read back from the Context itself.

    The configured name and the realised platform are different facts. Recording only the request
    means a manifest asserts a device rather than reporting one.
    """
    platform = simulation.context.getPlatform()
    record = {"name": platform.getName()}
    for prop in ("DeviceIndex", "DeviceName", "Precision"):
        if prop in platform.getPropertyNames():
            try:
                record[prop] = platform.getPropertyValue(simulation.context, prop)
            except Exception:                      # noqa: BLE001
                pass
    return record


def _make_simulation(topology, system, cfg: dict, seed: int, timestep_fs: Optional[float] = None,
                     device_index: Optional[int] = None):
    from openmm import app

    integrator = make_integrator(cfg, seed, timestep_fs)
    plat, props = _platform_and_properties(cfg, device_index)
    return app.Simulation(topology, system, integrator, plat, props or None)


def _steps(time_ps: float, timestep_fs: float) -> int:
    n = time_ps * 1000.0 / timestep_fs
    if abs(n - round(n)) > 1e-6:
        raise ValueError(
            f"{time_ps} ps is not an integer number of {timestep_fs} fs steps "
            f"({n:.6f}).  Choose intervals divisible by the timestep -- a rounded reporter "
            "interval silently changes the sampling frequency."
        )
    return int(round(n))


def _scaled_system(system, cfg: dict, n_solute_atoms: int, scale_factor: float,
                   omega_bonds: Sequence[Sequence[int]]):
    """Apply the project's REST2 scaling to a copy of *system* (identity at ``s = 1``)."""
    from .system import build_rest2_scaled_system

    if abs(scale_factor - 1.0) < 1e-12:
        # tau = 0 is the unscaled physical Hamiltonian, so it takes a plain copy. It is still
        # audited: an unclassifiable force must fail at the cold rung too, otherwise the ladder
        # builds replica 0 happily and only refuses at replica 1.
        from .system import audit_force_classes
        audit_force_classes(system, where="REST2 scaling (tau = 0)")
        return copy.deepcopy(system)
    return build_rest2_scaled_system(
        system,
        np.arange(int(n_solute_atoms)),
        float(scale_factor),
        exclude_central_bonds=[tuple(b) for b in omega_bonds] or None,
    )


# ---------------------------------------------------------------------------------------------
# Step 5 -- minimise, NVT, NPT
# ---------------------------------------------------------------------------------------------
def solute_atom_indices(topology, selection: str = "solute") -> list:
    """Resolve a named selection to topology atom indices.

    One definition of "solute", used by the positional restraint and by the selected-atom
    trajectory. Two definitions would eventually disagree, and the disagreement would surface as a
    trajectory whose atom order does not match the restraint's -- silently, because both are
    plausible lists of integers.

    `solute` is everything that is not water or an ion; `solute-heavy` drops hydrogens as well.
    """
    from openmm.app import element as elem

    if selection not in ("solute", "solute-heavy"):
        raise ValueError(
            f"unknown atom selection {selection!r}; implemented: solute, solute-heavy")
    out = []
    for atom in topology.atoms():
        if atom.residue.name.upper() in WATER_RESIDUE_NAMES | ION_RESIDUE_NAMES:
            continue
        if selection == "solute-heavy" and atom.element == elem.hydrogen:
            continue
        out.append(int(atom.index))
    return out


#: The two restraint distance conventions, named so a bundle records which one it used.
RESTRAINT_MINIMUM_IMAGE = "minimum-image (periodicdistance)"
RESTRAINT_CARTESIAN = "cartesian (nonperiodic)"


def _add_positional_restraints(system, topology, selection: str, positions_nm: np.ndarray):
    """Add a flat harmonic positional restraint driven by the global parameter ``k_restraint``.

    Returns ``(force_index, restrained_indices, convention)``.

    The distance convention follows the SYSTEM, and getting this wrong is not cosmetic:

    * **Explicit, periodic** -- ``periodicdistance``, so an atom that wanders across a box face is
      still measured to its own reference point rather than to an image of it.
    * **Implicit, nonperiodic** -- plain Cartesian displacement. ``periodicdistance`` in a system
      with no box makes the restraint depend on OpenMM's default box vectors, and it makes the
      restraint Force itself report periodic boundary use -- so a protocol that claims to have no
      box acquires one through its own restraint.

    Periodicity is read BEFORE the Force is added. Reading it afterwards would let the Force being
    constructed change the answer used to construct it.

    The stiffness is a global parameter, so it can be lowered between stages with one
    ``setParameter`` call instead of rebuilding the Context.
    """
    from openmm import CustomExternalForce

    # decided from the unmodified System, before anything is added to it
    periodic = bool(system.usesPeriodicBoundaryConditions())
    if periodic:
        expression = "k_restraint*periodicdistance(x, y, z, x0, y0, z0)^2"
        convention = RESTRAINT_MINIMUM_IMAGE
    else:
        expression = "k_restraint*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        convention = RESTRAINT_CARTESIAN

    force = CustomExternalForce(expression)
    force.addGlobalParameter("k_restraint", 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)

    restrained = solute_atom_indices(topology, selection)
    for atom_index in restrained:
        force.addParticle(int(atom_index), [float(x) for x in positions_nm[atom_index]])
    index = system.addForce(force)

    if not periodic and system.usesPeriodicBoundaryConditions():
        raise RuntimeError(
            "adding the positional restraint made a nonperiodic System report periodic boundary "
            "conditions. The restraint expression must not reference a periodic function here; "
            "refusing rather than running an implicit protocol that has silently acquired a box."
        )
    return index, restrained, convention


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Heavy-atom RMSD after optimal superposition (nm)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum() / len(a)))


def build_simbox(cfg: dict, out_dir: Path, suffix: str, *, smiles: Optional[str] = None,
                 pdb: Optional[Path] = None) -> dict:
    """Stage (a): SMILES or PDB in, a parameterised solvated box out.

    Runs initial structure -> protonation -> solvation -> System in one go and emits a bundle whose
    names carry *suffix*:

        <suffix>_system.xml     the parameterised System (the "-p" of every later stage)
        <suffix>_topology.pdb   topology, solvated coordinates and box vectors
        <suffix>_simbox.json    n_solute_atoms, omega bonds, box geometry, force field, provenance

    The per-step intermediates stay in ``<suffix>_prep/`` so nothing is thrown away.
    """
    out_dir = Path(out_dir)
    work = out_dir / f"{suffix}_prep"
    work.mkdir(parents=True, exist_ok=True)

    if (smiles is None) == (pdb is None):
        raise ValueError("build_simbox needs exactly one of smiles= or pdb=")

    input_route = "smiles" if smiles is not None else "pdb"
    required = cfg["system"].get("require_input_route")
    if required and input_route != required:
        raise ValueError(
            f"this configuration requires the '{required}' input route; got '{input_route}'.  "
            "Supply --smiles for the cyclo_rgdfv preset."
        )
    if required == "smiles" and not (smiles or "").strip():
        raise ValueError("the required 'smiles' input route was selected but no SMILES was given")
    if smiles is not None:
        structure = initial_structure(smiles, work, cfg)
    else:
        import shutil

        shutil.copy(Path(pdb), work / "solute.pdb")
        structure = {"source": "pdb", "input_pdb": str(pdb),
                     "note": "ETKDG skipped: a structure was supplied"}
        (work / "initial_structure.json").write_text(
            json.dumps(structure, indent=2) + "\n", encoding="utf-8"
        )

    sdf = work / "solute.sdf"
    ligand = sdf if sdf.exists() else None
    prot = protonate(work / "solute.pdb", work, cfg, ligand_sdf=ligand,
                     input_route=input_route)
    solv = solvate(work / "solute_h.pdb", work, cfg, ligand_sdf=ligand, route=prot["route"])
    build = build_system(
        work / "solvated.pdb", work, cfg, solv["n_solute_atoms"], ligand_sdf=ligand,
        route=prot["route"],
    )

    system_xml = out_dir / f"{suffix}_system.xml"
    topology_pdb = out_dir / f"{suffix}_topology.pdb"
    system_xml.write_text(Path(build["system_xml"]).read_text(encoding="utf-8"), encoding="utf-8")
    topology_pdb.write_text(Path(build["solvated_pdb"]).read_text(encoding="utf-8"),
                            encoding="utf-8")

    info = {
        "suffix": suffix,
        "input_route": input_route,
        "route": prot["route"],
        "input_smiles": smiles,
        "input_pdb": None if smiles is not None else str(pdb),
        "system_xml": str(system_xml),
        "topology_pdb": str(topology_pdb),
        "prep_dir": str(work),
        "n_solute_atoms": int(build["n_solute_atoms"]),
        "omega_central_bonds": build["omega_central_bonds"],
        **{k: build[k] for k in
           ("omega_unscaled_bonds", "omega_proline_like_scaled_bonds",
            "omega_unclassified_candidates", "omega_detection_method", "omega_detail")},
        "n_particles": build["n_particles"],
        "n_constraints": build["n_constraints"],
        "degrees_of_freedom": build["degrees_of_freedom"],
        "hmr": build["hmr"],
        "nonbonded": build["nonbonded"],
        "geometry": solv["geometry"],
        # Which water model was SIMULATED (the force field decides) and which model's
        # pre-equilibrated box supplied the starting coordinates. They differ for models
        # OpenMM cannot build a box for, such as OPC; recording it keeps the substitution
        # visible instead of leaving a reader to infer it from the force-field name.
        "water": {
            "model": solv.get("water_model"),
            "packing_model": solv.get("water_packing_model"),
            "packing_substituted": solv.get("water_packing_substituted"),
        },
        "n_waters": solv["n_waters"],
        "ions": solv["ions"],
        "salt": solv.get("salt"),
        "forcefield": build["forcefield"],
        # explicit, unambiguous provenance of the Hamiltonian that actually ran
        "hamiltonian_provenance": {
            "solute_route": input_route,
            "small_molecule_forcefield": (cfg["forcefield"]["ligand"]
                                          if prot["route"] == "ligand" else None),
            "charge_method": (cfg["forcefield"]["ligand_charge_method"]
                              if prot["route"] == "ligand" else None),
            "protein_forcefield": build["forcefield"]["protein_forcefield"],
            "water_forcefield": cfg["forcefield"]["water"],
            "input_smiles": smiles,
            "input_smiles_sha256": (
                sha256_text(smiles) if smiles else None
            ),
            "formal_charge": (build["forcefield"]["ligand"] or {}).get("formal_charge"),
        },
        "structure": structure,
        "protonation": {k: prot[k] for k in
                        ("ph", "ph_applies", "route", "note", "n_hydrogens_after")},
    }
    (out_dir / f"{suffix}_simbox.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_simbox", cfg, {"result": info})
    return info


def _load_bundle(system_xml: Path):
    """Return ``(system, topology, simbox_info)`` for a ``-p`` argument.

    ``-p`` names the parameterised System.  A ``System`` XML carries parameters but no atom or
    residue names, so the topology is read from the sibling ``<stem>_topology.pdb`` that
    :func:`build_simbox` wrote next to it -- the pair together is the analogue of an Amber prmtop.
    """
    from openmm import XmlSerializer, app

    system_xml = Path(system_xml)
    if not system_xml.exists():
        raise FileNotFoundError(f"-p {system_xml} not found")
    name = system_xml.name
    base = name[: -len("_system.xml")] if name.endswith("_system.xml") else system_xml.stem
    topology_pdb = system_xml.with_name(f"{base}_topology.pdb")
    info_path = system_xml.with_name(f"{base}_simbox.json")
    for path, what in ((topology_pdb, "topology"), (info_path, "box metadata")):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing.  -p expects the bundle written by simbox-setup.py: "
                f"<stem>_system.xml, <stem>_topology.pdb and <stem>_simbox.json side by side "
                f"({what} comes from this file)."
            )
    system = XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
    pdb = app.PDBFile(str(topology_pdb))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return system, pdb, info


def _apply_coords(sim, coords: Path, *, require_velocities: bool = False,
                  velocity_seed: Optional[int] = None) -> dict:
    """Seed a Context from ``-c``: either a PDB or a serialised OpenMM ``State``.

    Deliberately not ``loadCheckpoint``: a checkpoint is only valid for the exact System that wrote
    it, and the equilibration System carries a barostat and a positional-restraint force that a
    production System does not.  Copying positions, velocities and box vectors across is
    System-agnostic and cannot silently mismatch.
    """
    from openmm import XmlSerializer, app, unit

    coords = Path(coords)
    if not coords.exists():
        raise FileNotFoundError(f"-c {coords} not found")
    if coords.suffix.lower() == ".pdb":
        pdb = app.PDBFile(str(coords))
        box = pdb.topology.getPeriodicBoxVectors()
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
        sim.context.setPositions(pdb.positions)
        if require_velocities:
            temperature = float(
                sim.integrator.getTemperature().value_in_unit(unit.kelvin)) * unit.kelvin
            if velocity_seed is None:
                sim.context.setVelocitiesToTemperature(temperature)
            else:
                sim.context.setVelocitiesToTemperature(temperature, int(velocity_seed))
        return {"coords": str(coords), "kind": "pdb", "velocities": "drawn from Maxwell-Boltzmann",
                "velocity_seed": velocity_seed}

    state = XmlSerializer.deserialize(coords.read_text(encoding="utf-8"))
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim.context.setPositions(state.getPositions())
    used_seed = None
    try:
        sim.context.setVelocities(state.getVelocities())
        vel = "carried over from the state"
    except Exception:
        # The state has no velocities. Drawing them here is only correct for a stage that is about
        # to integrate: a minimisation needs none, and velocities created before minimisation are
        # actively harmful. They are constraint-projected for the pre-minimisation geometry, and
        # minimisation then moves every atom, so by the time the next stage integrates they violate
        # the constraints badly enough to produce NaN on the first step. Measured on the alanine
        # OPC box: inheriting such velocities is an immediate NaN, a fresh draw is stable.
        if not require_velocities:
            vel = "state carried none; none drawn (this stage does not integrate)"
        elif velocity_seed is None:
            sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature())
            vel = "state carried none; drawn from Maxwell-Boltzmann"
        else:
            used_seed = int(velocity_seed)
            sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature(), used_seed)
            vel = "state carried none; drawn from Maxwell-Boltzmann"
    return {"coords": str(coords), "kind": "state-xml", "velocities": vel,
            "velocity_seed": used_seed}


def minimize_equilibrate(cfg: dict, system_xml: Path, coords: Path, out_dir: Path,
                         suffix: str) -> dict:
    """Stage (b): minimise and equilibrate; emit the state every production stage starts from.

    ``equilibration.protocol = "staged"`` (the default) is the standard protocol for a flexible
    solute in a freshly built water box:

        1. minimise with the solute heavy atoms restrained  -- lets the water shell relax around
           the conformer instead of the conformer deforming to fit a badly packed shell;
        2. minimise unrestrained;
        3. heat 50 K -> 300 K over 200 ps in NVT at 1 fs, restrained -- a Modeller water box is
           placed on a lattice, so the first picoseconds carry large local forces;
        4. NPT restrained, 200 ps -- the density collapses onto its equilibrium value here;
        5. release the restraint in steps (10 -> 2.5 -> 1 -> 0 kcal/mol/A^2), 200 ps each;
        6. unrestrained NPT, 1 ns, whose tail sets the production box.

    Total 2.0 ns, negligible against a 1 us production run.  ``protocol = "simple"`` is
    minimise -> NVT -> NPT with no restraints and no ramp; adequate for a small rigid solute such as
    alanine dipeptide and nothing larger.

    What equilibration does *not* fix: the starting conformer is one arbitrary point in the
    macrocycle's conformational space, and no equilibration protocol converges cis/trans amides,
    ring pucker or rotamers.  That is what the REST2 ladder and the production length are for.  The
    solute heavy-atom RMSD from the minimised structure is recorded per stage so it is visible how
    far the conformer moved.

    Writes ``<suffix>_state.xml`` (positions, velocities, box) -- the ``-c`` of stages (c) and (d).
    """
    from openmm import MonteCarloBarostat, XmlSerializer, app, unit
    from openmm.app import StateDataReporter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ecfg = cfg["equilibration"]
    protocol = str(ecfg["protocol"])
    if protocol not in ("staged", "simple"):
        raise ValueError(f"equilibration.protocol must be 'staged' or 'simple', got {protocol!r}")

    system, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])

    temperature = float(cfg["integrator"]["temperature_k"])
    t_target = float(ecfg["heat_to_k"] or temperature)

    ref_positions = np.array(pdb.positions.value_in_unit(unit.nanometer))
    restraint_index, restrained_atoms, _restraint_convention = _add_positional_restraints(
        system, pdb.topology, str(ecfg["restraint_selection"]), ref_positions
    )
    barostat = MonteCarloBarostat(
        float(ecfg["pressure_bar"]) * unit.bar, temperature * unit.kelvin,
        int(ecfg["barostat_interval"]),
    )
    barostat_index = system.addForce(barostat)

    dt0 = float(ecfg["heat_timestep_fs"] if protocol == "staged" else ecfg["timestep_fs"])
    sim = _make_simulation(pdb.topology, system, cfg, int(ecfg["seed"]), timestep_fs=dt0)
    origin = _apply_coords(sim, coords)
    _set_barostat(sim, barostat_index, 0)

    # the restraint reference must match the coordinates actually loaded, not the bundle PDB
    ref_positions = sim.context.getState(
        getPositions=True, enforcePeriodicBox=False
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    _retarget_restraints(sim, restraint_index, ref_positions)

    heavy = [
        a.index for a in pdb.topology.atoms()
        if a.index < n_solute and a.element is not None and a.element.symbol != "H"
    ]

    def solute_rmsd_nm() -> float:
        pos = sim.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(
            asNumpy=True
        ).value_in_unit(unit.nanometer)
        return _kabsch_rmsd(pos[heavy], ref_positions[heavy])

    k_strong = float(ecfg["restraint_k_kj_mol_nm2"])
    tol = float(ecfg["minimize_tolerance_kj_mol_nm"]) * unit.kilojoule_per_mole / unit.nanometer
    max_it = int(ecfg["minimize_max_iterations"])
    stages: list[dict] = []

    def record(name: str, **kw) -> None:
        state = sim.context.getState(getEnergy=True)
        vol = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
        entry = {
            "stage": name,
            "potential_kj_mol": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            "box_volume_nm3": round(vol, 4),
            "solute_heavy_rmsd_from_start_nm": round(solute_rmsd_nm(), 4),
            **kw,
        }
        stages.append(entry)
        print(
            f"[equil] {name:18s} U = {entry['potential_kj_mol']:12.1f} kJ/mol  "
            f"V = {vol:8.2f} nm^3  solute RMSD = {entry['solute_heavy_rmsd_from_start_nm']:.3f} nm",
            flush=True,
        )

    record("initial", restraint_k=0.0)
    csv_path = out_dir / f"{suffix}_equilibration.csv"

    if protocol == "staged":
        sim.context.setParameter("k_restraint", k_strong)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_restrained", restraint_k=k_strong)
        sim.context.setParameter("k_restraint", 0.0)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_free", restraint_k=0.0)

        with (out_dir / f"{suffix}_minimized.pdb").open("w") as fh:
            app.PDBFile.writeFile(
                sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
            )
        # the restraint reference is the MINIMISED structure, not the raw input
        ref_positions = sim.context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        _retarget_restraints(sim, restraint_index, ref_positions)

        sim.reporters.append(
            StateDataReporter(
                str(csv_path), _steps(10.0, dt0), step=True, time=True, potentialEnergy=True,
                kineticEnergy=True, temperature=True, volume=True, density=True, speed=True,
            )
        )
        sim.context.setParameter("k_restraint", k_strong)
        t_from = float(ecfg["heat_from_k"])
        sim.context.setVelocitiesToTemperature(t_from * unit.kelvin, int(ecfg["seed"]))
        n_win = max(1, int(ecfg["heat_n_windows"]))
        per_window = max(1, _steps(float(ecfg["heat_ps"]), dt0) // n_win)
        for w in range(n_win):
            t_w = t_from + (t_target - t_from) * (w + 1) / n_win
            sim.integrator.setTemperature(t_w * unit.kelvin)
            sim.step(per_window)
        record("nvt_heat", restraint_k=k_strong, temperature_k=t_target,
               ps=float(ecfg["heat_ps"]), timestep_fs=dt0)

        sim.integrator.setStepSize(float(ecfg["timestep_fs"]) * unit.femtosecond)
        dt = float(ecfg["timestep_fs"])
        _set_barostat(sim, barostat_index, int(ecfg["barostat_interval"]),
                      float(ecfg["pressure_bar"]))
        sim.step(_steps(float(ecfg["npt_restrained_ps"]), dt))
        record("npt_restrained", restraint_k=k_strong, ps=float(ecfg["npt_restrained_ps"]))

        for k in ecfg["release_schedule_kj_mol_nm2"]:
            sim.context.setParameter("k_restraint", float(k))
            sim.step(_steps(float(ecfg["release_ps_each"]), dt))
            record(f"npt_release_k{float(k):g}", restraint_k=float(k),
                   ps=float(ecfg["release_ps_each"]))
        sim.context.setParameter("k_restraint", 0.0)
        free_ps = float(ecfg["npt_free_ps"])
    else:
        sim.context.setParameter("k_restraint", 0.0)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_free", restraint_k=0.0)
        with (out_dir / f"{suffix}_minimized.pdb").open("w") as fh:
            app.PDBFile.writeFile(
                sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
            )
        sim.reporters.append(
            StateDataReporter(
                str(csv_path), _steps(10.0, dt0), step=True, time=True, potentialEnergy=True,
                kineticEnergy=True, temperature=True, volume=True, density=True, speed=True,
            )
        )
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, int(ecfg["seed"]))
        sim.integrator.setTemperature(temperature * unit.kelvin)
        dt = float(ecfg["timestep_fs"])
        sim.step(_steps(float(ecfg["nvt_ps"]), dt))
        record("nvt", ps=float(ecfg["nvt_ps"]))
        _set_barostat(sim, barostat_index, int(ecfg["barostat_interval"]),
                      float(ecfg["pressure_bar"]))
        free_ps = float(ecfg["npt_ps"])

    tail_ps = min(float(ecfg["box_average_last_ps"]), free_ps)
    head_steps = _steps(free_ps, dt) - _steps(tail_ps, dt)
    if head_steps > 0:
        sim.step(head_steps)
    tail_steps = _steps(tail_ps, dt)
    sample_every = max(1, min(_steps(10.0, dt), tail_steps // 20))
    n_samples = max(1, tail_steps // sample_every)
    # Keep the STATES, not only the box vectors.  Pasting a mean box onto the final instantaneous
    # configuration would pair coordinates equilibrated in one cell with a cell they never saw --
    # the solvent density and the solute's periodic separation would both be slightly wrong, and
    # nothing downstream would notice.  Instead the handoff is the sampled state whose own volume
    # is closest to the tail mean: a configuration and a box that actually occurred together.
    samples, boxes = [], []
    for _ in range(n_samples):
        sim.step(sample_every)
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        boxes.append(bv)
        samples.append((float(np.abs(np.linalg.det(bv))), st, bv))
    mean_box = np.mean(np.array(boxes), axis=0)
    record("npt_free", restraint_k=0.0, ps=free_ps)

    mean_vol = float(np.mean([v for v, _, _ in samples]))
    pick = int(np.argmin([abs(v - mean_vol) for v, _, _ in samples]))
    sel_vol, final, sel_box = samples[pick]
    box_handoff = {
        "method": "nearest-sampled-state",
        "mean_tail_volume_nm3": round(mean_vol, 5),
        "selected_volume_nm3": round(sel_vol, 5),
        "deviation_nm3": round(sel_vol - mean_vol, 6),
        "deviation_percent": round(100.0 * (sel_vol - mean_vol) / mean_vol, 4),
        "selected_sample_index": pick,
        "n_samples": len(samples),
        "note": "coordinates, velocities and box all come from ONE sampled state; the mean box is "
                "reported for reference and is deliberately NOT pasted onto other coordinates",
    }
    print(f"[equil] box handoff: sample {pick}/{len(samples)} at {sel_vol:.3f} nm^3, "
          f"{box_handoff['deviation_percent']:+.2f} % from the tail mean {mean_vol:.3f} nm^3",
          flush=True)
    mean_box = sel_box
    state_xml = out_dir / f"{suffix}_state.xml"
    state_xml.write_text(XmlSerializer.serialize(final), encoding="utf-8")

    # The human-readable structure must carry the SAME cell as the state it was taken from.
    # `sim.topology` still holds the box it was built with -- the pre-NPT one -- so writing through
    # it stamps a CRYST1 record describing a cell these coordinates were never equilibrated in.
    # Nothing downstream reads the PDB for production (the serialized State is authoritative), which
    # is exactly why the mismatch could sit here unnoticed and mislead anyone who opened the file.
    #
    # `sim.topology` is shared with the running Simulation, so its box is set for the write and
    # restored in a `finally`: the mutation window is this block only, and an exception inside it
    # cannot leave the shared topology carrying a box it never had.
    #
    # NOT copy.deepcopy(sim.topology). deepcopy produces new Atom objects while the copied Bond
    # tuples still reference the ORIGINALS, so PDBFile.writeFooter builds its atom index from the
    # new atoms and then dies with KeyError on the first bond. Found by running the pipeline; a
    # topology with no bonds -- which is what a unit test naturally builds -- never reaches it.
    #
    # PDB cannot round-trip a triclinic cell faithfully either: CRYST1 stores lengths and angles to
    # three decimals in angstrom, and reading it back yields OpenMM's REDUCED lattice form, which
    # for a dodecahedral box flips the sign of the third vector's x/y. Volume survives; the vectors
    # as written do not. So an mmCIF copy is written alongside and the authority order is stated:
    #
    #   {suffix}_state.xml  AUTHORITATIVE -- exact box, positions and velocities; production input
    #   {suffix}_equilibrated.cif         -- faithful cell for tools that read mmCIF
    #   {suffix}_equilibrated.pdb         -- for viewing; cell correct to PDB precision and form
    original_box = sim.topology.getPeriodicBoxVectors()
    try:
        sim.topology.setPeriodicBoxVectors(final.getPeriodicBoxVectors())
        with (out_dir / f"{suffix}_equilibrated.pdb").open("w") as fh:
            app.PDBFile.writeFile(sim.topology, final.getPositions(), fh)
        with (out_dir / f"{suffix}_equilibrated.cif").open("w") as fh:
            app.PDBxFile.writeFile(sim.topology, final.getPositions(), fh)
    finally:
        sim.topology.setPeriodicBoxVectors(original_box)

    box_handoff["selected_box_vectors_nm"] = [[round(float(x), 6) for x in row] for row in sel_box]
    box_handoff["structure_files"] = {
        "authoritative": f"{suffix}_state.xml",
        "mmcif": f"{suffix}_equilibrated.cif",
        "pdb": f"{suffix}_equilibrated.pdb",
        "note": "both structures carry the SELECTED NPT box, not the pre-NPT topology box; PDB "
                "CRYST1 is limited to 1e-4 nm and to OpenMM's reduced lattice form, so the "
                "serialized State is authoritative and the mmCIF is the faithful structure copy",
    }

    volumes = [float(np.abs(np.linalg.det(b))) for b in boxes]
    info = {
        "suffix": suffix,
        "protocol": protocol,
        "system_xml": str(system_xml),
        "input_coords": origin,
        "stages": stages,
        "n_restrained_atoms": len(restrained_atoms),
        "restraint_selection": str(ecfg["restraint_selection"]),
        "restraint_k_kj_mol_nm2": k_strong,
        "release_schedule_kj_mol_nm2": list(ecfg["release_schedule_kj_mol_nm2"]),
        "total_equilibration_ps": (
            float(ecfg["heat_ps"]) + float(ecfg["npt_restrained_ps"])
            + len(ecfg["release_schedule_kj_mol_nm2"]) * float(ecfg["release_ps_each"])
            + float(ecfg["npt_free_ps"])
            if protocol == "staged"
            else float(ecfg["nvt_ps"]) + float(ecfg["npt_ps"])
        ),
        "box_vectors_nm": mean_box.tolist(),
        "box_volume_nm3_mean": float(np.mean(volumes)),
        "box_volume_nm3_sd": float(np.std(volumes, ddof=1)) if len(volumes) > 1 else 0.0,
        "n_box_samples": len(volumes),
        "solute_heavy_rmsd_from_start_nm": stages[-1]["solute_heavy_rmsd_from_start_nm"],
        "box_handoff": box_handoff,
        "state_xml": str(state_xml),
        "production_ensemble": cfg["production"]["ensemble"],
    }
    (out_dir / f"{suffix}_min-eq.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_min-eq", cfg, {"result": info})
    return info


def _retarget_restraints(sim, force_index: int, positions_nm: np.ndarray) -> None:
    """Move every restraint's reference point onto *positions_nm*, in place."""
    force = sim.system.getForce(force_index)
    for particle in range(force.getNumParticles()):
        atom_index, _ = force.getParticleParameters(particle)
        force.setParticleParameters(
            particle, atom_index, [float(x) for x in positions_nm[atom_index]]
        )
    force.updateParametersInContext(sim.context)


def _set_barostat(sim, force_index: int, frequency: int, pressure_bar: float = 1.0) -> None:
    """Set the barostat frequency (0 = off for an NVT leg) and pressure, in place.

    The force stays in the System rather than being removed and re-added, so force indices the rest
    of the System refers to never move.
    """
    from openmm import unit

    force = sim.system.getForce(force_index)
    force.setFrequency(int(frequency))
    force.setDefaultPressure(pressure_bar * unit.bar)
    sim.context.reinitialize(preserveState=True)


