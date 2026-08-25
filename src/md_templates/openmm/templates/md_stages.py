"""Stage construction shared by the generated cMD and REST2 scripts.

Copied into every generated project, so it must depend on nothing but OpenMM, PyYAML and the
standard library.

The pipeline both methods run on a fresh start:

    restrained minimization
      -> restrained NVT   (no active barostat)
      -> restrained NPT   (exactly one active barostat; explicit solvent only)
      -> unrestrained production

Under implicit solvent there is no box, so there is no barostat and no NPT stage at all -- the
production ensemble is NVT.
"""
from concurrent.futures import ThreadPoolExecutor

from openmm import (CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
                    unit)
from openmm.app import Simulation

#: U = 1/2 k |r - r0|^2, with k given in kcal mol^-1 A^-2.
KCAL_PER_MOL_ANGSTROM2 = 418.4          # kJ mol^-1 nm^-2
RESTRAINT_PARAMETER = "restraint_k"
#: OpenMM seeds are 32-bit signed; 0 means "pick one at random", which is not reproducible.
MAX_SEED = 2 ** 31 - 1
#: Barostat attempt interval during NPT equilibration and NPT production.
PRODUCTION_BAROSTAT_FREQUENCY = 25


def derive_seed(base, *purpose):
    """A distinct, deterministic seed per (base, purpose).

    Every replica needs its own integrator, velocity and barostat seed. Sharing one seed across
    replicas correlates their trajectories, and a ladder whose rungs move together samples less
    than it appears to.
    """
    value = int(base)
    for part in purpose:
        text = str(part).encode("utf-8")
        for byte in text:
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    seed = value % MAX_SEED
    return seed if seed else 1              # never 0: OpenMM reads that as "choose randomly"


def add_positional_restraint(system, reference_positions, atom_indices):
    """A `CustomExternalForce` holding `atom_indices` near their reference coordinates.

    The Force stays in the System for the whole run so the checkpoint layout never changes; its
    strength is a global Context parameter, set to zero before production. Removing the Force
    instead would make a production checkpoint structurally incompatible with the equilibration
    that produced it.
    """
    force = CustomExternalForce(f"0.5*{RESTRAINT_PARAMETER}*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter(RESTRAINT_PARAMETER, 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    positions = reference_positions.value_in_unit(unit.nanometer)
    for index in atom_indices:
        x, y, z = positions[int(index)]
        force.addParticle(int(index), [x, y, z])
    system.addForce(force)
    return force


def add_barostat(system, pressure_bar, temperature, seed, frequency=PRODUCTION_BAROSTAT_FREQUENCY):
    """One barostat, created inactive.

    Frequency 0 means it never attempts a move, which is how the NVT stage runs without a barostat
    while the System keeps a stable Force layout for the checkpoint.
    """
    barostat = MonteCarloBarostat(float(pressure_bar) * unit.bar, temperature, frequency)
    barostat.setRandomNumberSeed(int(seed))
    system.addForce(barostat)
    return barostat


def count_barostats(system):
    return sum(1 for i in range(system.getNumForces())
               if "Barostat" in type(system.getForce(i)).__name__)


def set_barostat_frequency(simulation, frequency):
    """Turn the barostat on or off, and make the Context see it.

    A Force property changed after the Context exists is invisible until the Context is
    reinitialised; `preserveState=True` keeps positions, velocities and box.
    """
    system = simulation.context.getSystem()
    changed = False
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if "Barostat" in type(force).__name__:
            force.setFrequency(int(frequency))
            changed = True
    if changed:
        simulation.context.reinitialize(preserveState=True)
    return changed


def active_barostat_count(simulation):
    """Barostats that will actually attempt a move -- frequency 0 is inactive."""
    system = simulation.context.getSystem()
    return sum(1 for i in range(system.getNumForces())
               if "Barostat" in type(system.getForce(i)).__name__
               and system.getForce(i).getFrequency() > 0)


def set_restraint(simulation, k_kcal_mol_a2):
    simulation.context.setParameter(
        RESTRAINT_PARAMETER, float(k_kcal_mol_a2) * KCAL_PER_MOL_ANGSTROM2)


def restraint_strength(simulation):
    return simulation.context.getParameter(RESTRAINT_PARAMETER)


def resolve_platform(forced=None):
    """CUDA unless a different platform was asked for BY NAME.

    Simulations belong on a GPU. A silent fall back to the CPU still finishes, still writes a
    trajectory and still says "complete" -- two orders of magnitude later, on a machine whose GPU
    was simply not visible to this process. So the fallback has to be requested rather than
    inherited, and a missing CUDA platform is an error with the override spelled out.
    """
    available = sorted(Platform.getPlatform(i).getName()
                       for i in range(Platform.getNumPlatforms()))
    if forced:
        if forced not in available:
            raise SystemExit(f"MD_PLATFORM={forced} is not available in this OpenMM build. "
                             f"It has: {', '.join(available)}.")
        return forced
    if "CUDA" not in available:
        raise SystemExit(
            f"no CUDA platform in this OpenMM build (it has: {', '.join(available)}), and this "
            "run was not told to use anything else.\n"
            "Check that the GPU is visible to this process (nvidia-smi, CUDA_VISIBLE_DEVICES), "
            "or ask for another platform by name:\n"
            "    MD_PLATFORM=CPU ./run.sh")
    return "CUDA"


def make_simulation(topology, system, *, temperature, friction, timestep, seed,
                    platform_name=None, device=None):
    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
    integrator.setRandomNumberSeed(int(seed))
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name == "CUDA":
            properties = {"Precision": "mixed"}
            if device is not None:
                properties["DeviceIndex"] = str(device)
        return Simulation(topology, system, integrator, platform, properties)
    return Simulation(topology, system, integrator)


def steps_for(picoseconds, timestep_fs):
    """An exact number of steps, or a refusal. Rounding runs a different length than configured."""
    exact = float(picoseconds) * 1000.0 / float(timestep_fs)
    if abs(exact - round(exact)) > 1e-9:
        raise SystemExit(
            f"{picoseconds} ps is not a whole number of {timestep_fs} fs steps ({exact}). "
            "Choose a duration that divides exactly.")
    return int(round(exact))


def equilibrate(simulation, *, config, implicit, temperature, timestep_fs, velocity_seed,
                log=print, label=""):
    """restrained minimization -> restrained NVT -> restrained NPT -> production handoff.

    `minimization.restraint_k_kcal_mol_a2` holds the solute while the initial clashes are relieved;
    `equilibration.restraint_k_kcal_mol_a2` holds it while the solvent relaxes around it. They are
    two settings in the public configuration, so they are two strengths here rather than one of
    them being quietly ignored.

    Returns a record of what actually ran, so the caller can log it and a test can check it rather
    than trusting that a configured stage happened.
    """
    prefix = f"[{label}] " if label else ""
    record = {}

    equilibration = config.get("equilibration") or {}
    minimization_k = float(config["minimization"]["restraint_k_kcal_mol_a2"])
    equilibration_k = float(equilibration.get("restraint_k_kcal_mol_a2", minimization_k))

    set_restraint(simulation, minimization_k)
    record["restraint_kcal_minimization"] = minimization_k
    record["restraint_kj_nm2"] = restraint_strength(simulation)

    # Minimisation and NVT with the barostat inactive: a barostat during minimisation moves the box
    # against forces that are still enormous.
    set_barostat_frequency(simulation, 0)
    record["barostats_active_minimization"] = active_barostat_count(simulation)

    iterations = int(config["minimization"]["max_iterations"])
    log(f"{prefix}minimising, max {iterations} iterations, restraint "
        f"{minimization_k} kcal/mol/A^2 on the solute")
    simulation.minimizeEnergy(maxIterations=iterations)

    simulation.context.setVelocitiesToTemperature(temperature, int(velocity_seed))

    set_restraint(simulation, equilibration_k)
    record["restraint_kcal_equilibration"] = equilibration_k
    record["restraint_kj_nm2_equilibration"] = restraint_strength(simulation)

    nvt_ps = float(equilibration.get("nvt_duration_ps") or 0.0)
    npt_ps = float(equilibration.get("npt_duration_ps") or 0.0)

    nvt_steps = steps_for(nvt_ps, timestep_fs) if nvt_ps else 0
    if nvt_steps:
        record["barostats_active_nvt"] = active_barostat_count(simulation)
        log(f"{prefix}NVT {nvt_ps} ps, restraint {equilibration_k} kcal/mol/A^2, "
            f"{record['barostats_active_nvt']} active barostat(s)")
        simulation.step(nvt_steps)
    record.setdefault("barostats_active_nvt", active_barostat_count(simulation))

    npt_steps = steps_for(npt_ps, timestep_fs) if (npt_ps and not implicit) else 0
    if npt_steps:
        set_barostat_frequency(simulation, PRODUCTION_BAROSTAT_FREQUENCY)
        active = active_barostat_count(simulation)
        if active != 1:
            raise SystemExit(f"NPT equilibration needs exactly one active barostat; found {active}")
        record["barostats_active_npt"] = active
        log(f"{prefix}NPT {npt_ps} ps, restraint {equilibration_k} kcal/mol/A^2, "
            f"{active} active barostat")
        simulation.step(npt_steps)
    elif implicit:
        record["barostats_active_npt"] = 0

    record.update(enter_production(simulation, implicit=implicit))
    record["restraint_after_equilibration"] = record["restraint_kj_nm2_production"]
    simulation.context.setStepCount(0)
    log(f"{prefix}equilibration complete; restraint off, "
        f"{record['barostats_active_production']} active barostat(s) for production")
    return record


def enter_production(simulation, *, implicit):
    """The Force layout production runs in: restraint off, barostat active unless implicit.

    Production is unrestrained. The Force stays in the System -- only its strength goes to zero --
    so the checkpoint layout is the same before and after.
    """
    if not implicit:
        set_barostat_frequency(simulation, PRODUCTION_BAROSTAT_FREQUENCY)
    set_restraint(simulation, 0.0)
    return {"restraint_kj_nm2_production": restraint_strength(simulation),
            "barostats_active_production": active_barostat_count(simulation)}


def resume_production(simulation, checkpoint_path, *, implicit):
    """Load a production checkpoint into the production Force layout.

    Order matters. A barostat's frequency lives in the System, NOT in the checkpoint, so a resumed
    run that only loaded the checkpoint would keep the inactive barostat the fresh run used for
    minimisation: NPT production that is silently NVT. The layout is therefore set BEFORE loading,
    which also avoids a `reinitialize` afterwards -- that would restart the integrator's random
    stream, and the point of loading a checkpoint is that it restores it.
    """
    if not implicit:
        set_barostat_frequency(simulation, PRODUCTION_BAROSTAT_FREQUENCY)
    simulation.loadCheckpoint(str(checkpoint_path))
    # A Context parameter, so this needs no reinitialisation and disturbs nothing that was loaded.
    set_restraint(simulation, 0.0)
    return {"restraint_kj_nm2_production": restraint_strength(simulation),
            "barostats_active_production": active_barostat_count(simulation),
            "resumed_at_step": simulation.context.getStepCount()}


def device_groups(n_replicas, devices):
    """Replicas grouped by the device they run on, round-robin over at most n_replicas devices.

    Returns one list per device actually used, so `len(groups)` is
    `min(n_replicas, n_visible_devices)`. With no devices there is one group: the sequential path.
    """
    if not devices:
        return [list(range(n_replicas))]
    used = list(devices)[:min(n_replicas, len(devices))]
    groups = [[] for _ in used]
    for replica in range(n_replicas):
        groups[replica % len(used)].append(replica)
    return [group for group in groups if group]


def propagate_segment(groups, step_fn):
    """Advance every replica by one segment, then return.

    Replicas that share a device propagate sequentially -- interleaving them on one GPU makes both
    slower, not faster. Different devices propagate concurrently: OpenMM releases the GIL while
    stepping, so plain threads genuinely overlap. Every group is awaited before the caller attempts
    an exchange, because an exchange reads energies that a still-running replica has not produced.
    """
    if len(groups) <= 1:
        for replica in (groups[0] if groups else ()):
            step_fn(replica)
        return

    def run_group(group):
        for replica in group:
            step_fn(replica)

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(run_group, group) for group in groups]
        for future in futures:
            future.result()             # re-raise here, before any exchange is attempted
