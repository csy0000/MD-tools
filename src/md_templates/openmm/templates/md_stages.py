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
from pathlib import Path

from openmm import (CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
                    XmlSerializer, unit)
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


#: U = 1/2 k |r - r0|^2 under periodic boundaries: the minimum image, so an atom that crosses a
#: box face is pulled back to the nearest image of its reference rather than across the whole cell.
PERIODIC_RESTRAINT = "0.5*{k}*periodicdistance(x, y, z, x0, y0, z0)^2"
#: The same energy without a box. There is no minimum image to take, and asking for one is not
#: merely redundant -- see `add_positional_restraint`.
NONPERIODIC_RESTRAINT = "0.5*{k}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"


def add_positional_restraint(system, reference_positions, atom_indices):
    """A `CustomExternalForce` holding `atom_indices` near their reference coordinates.

    The energy expression depends on whether the System has a box:

        periodic      0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2
        non-periodic  0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)

    They are the same energy whenever an atom is far from any box face, so the difference is easy
    to miss. It matters because `periodicdistance` makes the Force report
    `usesPeriodicBoundaryConditions() == True`, and OpenMM answers that question for a System by
    asking its Forces: adding a periodic restraint to an implicit GBn2 system flips
    `System.usesPeriodicBoundaryConditions()` from False to True. The system then describes itself
    as periodic while having no meaningful box, which is a false statement about the physics being
    sampled and the kind of thing a later check reads and trusts.

    Periodicity is taken from the System, not from a file name or a solvent label: the System is
    what OpenMM will actually integrate. It must be read BEFORE the Force is added, because adding
    a periodic Force is precisely what would change the answer.

    The Force stays in the System for the whole run so the checkpoint layout never changes; its
    strength is a global Context parameter, set to zero before production. Removing the Force
    instead would make a production checkpoint structurally incompatible with the equilibration
    that produced it.
    """
    periodic = system.usesPeriodicBoundaryConditions()
    template = PERIODIC_RESTRAINT if periodic else NONPERIODIC_RESTRAINT
    force = CustomExternalForce(template.format(k=RESTRAINT_PARAMETER))
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


def require_parent_state(path, *, stage_name, command):
    """The finalized state of the parent stage, or a refusal that says how to produce it.

    A stage is never allowed to fall back to its parent's `checkpoint.chk`. A checkpoint is written
    while a stage is still running, so consuming one as input means starting from a partially
    completed parent while every artifact looks normal.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"missing {path.name}: this stage reads {path}, which stage '{stage_name}' writes "
            f"only when it finishes.\n"
            f"Run it first:\n"
            f"    cd {command} && ./run.sh")
    return XmlSerializer.deserialize(path.read_text(encoding="utf-8"))


def build_stage_system(inputs, *, implicit, restrained, barostat_active, pressure_bar,
                       temperature, barostat_seed, solute_indices):
    """The System a stage integrates, with its Force layout fixed before any state is loaded.

    The restraint Force is always present, at zero strength when the stage is unrestrained, so a
    State carrying a `restraint_k` parameter can be loaded into any stage's Context. The barostat
    is present only under explicit solvent, and its frequency -- not its presence -- is what makes
    a stage NVT or NPT.
    """
    system = XmlSerializer.deserialize((Path(inputs) / "system.xml").read_text(encoding="utf-8"))
    initial = XmlSerializer.deserialize(
        (Path(inputs) / "initial_state.xml").read_text(encoding="utf-8"))
    add_positional_restraint(system, initial.getPositions(), solute_indices)
    if not implicit:
        add_barostat(system, pressure_bar, temperature, barostat_seed,
                     frequency=PRODUCTION_BAROSTAT_FREQUENCY if barostat_active else 0)
    barostats = count_barostats(system)
    if implicit and barostats:
        raise SystemExit(f"implicit solvent must have no barostat in the System; found {barostats}")
    return system


def write_final_state(simulation, path):
    """The handoff to the next stage: positions, velocities, box, time and parameters.

    Written only once the stage has finished, and written atomically, so a downstream stage cannot
    read a half-written file and start from a state that never existed.
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True,
                                        getParameters=True, enforcePeriodicBox=False)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    temporary.replace(path)
    return state


def write_final_pdb(simulation, path, *, implicit):
    from openmm.app import PDBFile

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=not implicit)
    with Path(path).open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


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
