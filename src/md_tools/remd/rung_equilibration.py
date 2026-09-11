"""Equilibrate every rung of a ladder under its OWN Hamiltonian, before the first exchange.

`rest2.equilibration_per_tau: true`. The tau = 0 chain stops at its last pressure-coupled stage
(explicit solvent, which fixes the box) or at minimisation (implicit), and every rung -- tau = 0
included -- then runs the same fixed-volume equilibration stages the chain would have run, each
under that rung's own scaled System:

    eq_nvt_posres     restrained NVT      stages.restrained_nvt_steps
    eq_nvt_posres_2   restrained NVT      stages.restrained_npt_steps
    eq_nvt_free       unrestrained NVT    stages.unrestrained_npt_steps

These are exactly the stages a scaled cMD run gets, because a scaled Hamiltonian is fixed-volume
throughout, and a ladder's rungs must share one volume anyway.

EACH STAGE IS DONE THE WAY THE STAGE CHAIN DOES IT, NOT ONLY THE SAME LENGTH

  * a NEW Context and integrator per stage, seeded per stage and per rung --
    `derive_seed(seed, stage, "state<i>")` -- so no two rungs and no two stages share a stream;
  * the configuration the previous stage ended with is installed (box, positions, velocities),
    and the restraint strength is set AFTER it, as `stage_main` does after `setState`;
  * velocities are carried, never redrawn. The chain draws Maxwell velocities only for a first
    stage with no `-c`, and a ladder always has one: `min.xml` carries the minimiser's zero
    velocities and an explicit-solvent `eq_npt_free.xml` the thermostatted ones;
  * the positional restraint is `add_positional_restraint` below -- the same Force, on the same
    atoms, at the same reference coordinates (the topology's) -- on a CLONE of the rung. The
    rung Systems the ladder propagates never carry it: they are what the force audit, the
    identity and an exported bundle's `verify_rungs.py` describe.

The integrator's constraint tolerance is the LADDER's (1e-8), which is the tolerance the rung
will be propagated with a moment later.

COPIED VERBATIM INTO AN EXPORTED REFERENCE BUNDLE, with `engine.py` and `core.py`, so it must
import nothing from md_tools: OpenMM, numpy and the standard library only. The runner there calls
`equilibrate_rung` exactly as the driver does, and a test compares the two exchange for exchange.

`derive_seed` and the restraint Force are DEFINED here and re-exported by `md_tools.md._stages`,
which is what every cMD stage uses -- one implementation, reachable from a bundle.
"""
from openmm import Context, CustomExternalForce, LangevinMiddleIntegrator, XmlSerializer, unit

from .engine import Configuration

#: U = 1/2 k |r - r0|^2, with k given in kcal mol^-1 A^-2.
KCAL_PER_MOL_ANGSTROM2 = 418.4          # kJ mol^-1 nm^-2
RESTRAINT_PARAMETER = "restraint_k"
#: OpenMM seeds are 32-bit signed; 0 means "pick one at random", which is not reproducible.
MAX_SEED = 2 ** 31 - 1

#: The per-tau stages, in the order they run. Their lengths come from the stage block.
PER_TAU_STAGE_NAMES = ("eq_nvt_posres", "eq_nvt_posres_2", "eq_nvt_free")

#: The run-state `phase` of a ladder stopped during per-tau equilibration. Such a ladder has taken
#: no exchange step and committed no checkpoint, so `--resume` refuses it by this name.
INTERRUPTED_PHASE = "per_tau_equilibration"

#: The record a ladder writes beside its per-state handoff files, and their names.
RECORD_NAME = "per_tau_equilibration.json"


def handoff_name(state_index):
    """`per_tau_state<i>.xml`: where rung i's equilibrated state is kept. Named by state, never tau."""
    return f"per_tau_state{int(state_index)}.xml"


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


def stage_seed(seed, stage_name, state_index):
    """The integrator seed of one per-tau stage on one rung."""
    return derive_seed(int(seed), str(stage_name), f"state{int(state_index)}")


def restrained_clone(system, reference_positions, atom_indices):
    """A copy of a rung's System carrying the positional restraint, at zero strength.

    A copy, so the rung the ladder propagates is never modified. The restraint is added to the
    copy exactly as a cMD stage adds it to its own System.
    """
    clone = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    add_positional_restraint(clone, reference_positions, atom_indices)
    return clone


def run_stage(restrained, configuration, stage, *, state_index, seed, temperature_k,
              friction_per_ps, timestep_fs, constraint_tolerance, platform=None,
              properties=None):
    """One per-tau stage on one rung. Returns `(configuration, record, final_state_xml)`.

    `restrained` is `restrained_clone(rung)`; `stage` is `{"name", "steps",
    "restraint_kcal_per_mol_A2"}`. The Context lives for this stage only.
    """
    steps = int(stage["steps"])
    strength = float(stage["restraint_kcal_per_mol_A2"])
    seed_used = stage_seed(seed, stage["name"], state_index)
    integrator = LangevinMiddleIntegrator(float(temperature_k) * unit.kelvin,
                                          float(friction_per_ps) / unit.picosecond,
                                          float(timestep_fs) * unit.femtosecond)
    integrator.setConstraintTolerance(float(constraint_tolerance))
    integrator.setRandomNumberSeed(int(seed_used))
    if platform is None:
        context = Context(restrained, integrator)
    elif properties:
        context = Context(restrained, integrator, platform, dict(properties))
    else:
        context = Context(restrained, integrator, platform)
    try:
        if configuration.box is not None:
            context.setPeriodicBoxVectors(*[list(row) for row in configuration.box])
        context.setPositions(configuration.positions * unit.nanometer)
        context.setVelocities(configuration.velocities * unit.nanometer / unit.picosecond)
        # AFTER the configuration, as `stage_main` sets it after `setState`.
        context.setParameter(RESTRAINT_PARAMETER, strength * KCAL_PER_MOL_ANGSTROM2)
        integrator.step(steps)
        state = context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                 enforcePeriodicBox=False)
        box = None
        if configuration.box is not None:
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        finished = Configuration(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            box)
        text = XmlSerializer.serialize(state)
    finally:
        del context
        del integrator
    record = {"stage": str(stage["name"]), "steps": steps,
              "restraint_kcal_per_mol_A2": strength, "seed": int(seed_used)}
    return finished, record, text


def equilibrate_rung(system, configuration, stages, *, reference_positions, restrained_atoms,
                     state_index, seed, temperature_k, friction_per_ps, timestep_fs,
                     constraint_tolerance, platform=None, properties=None):
    """Every per-tau stage on one rung, in order. Returns `(configuration, records)`.

    The driver runs the same `run_stage` calls stage by stage across its rungs, so it can stop
    between stages on an interruption every rank agrees on; this is the serial form of it.
    """
    restrained = restrained_clone(system, reference_positions, restrained_atoms)
    current, records = configuration.copy(), []
    for stage in stages:
        current, record, _text = run_stage(
            restrained, current, stage, state_index=state_index, seed=seed,
            temperature_k=temperature_k, friction_per_ps=friction_per_ps,
            timestep_fs=timestep_fs, constraint_tolerance=constraint_tolerance,
            platform=platform, properties=properties)
        records.append(record)
    return current, records


def validate_plan(stages):
    """The per-tau stage list, checked. Returns it as plain dicts, or raises ValueError."""
    checked = []
    for stage in stages or ():
        name = str(stage.get("name") or "")
        if name not in PER_TAU_STAGE_NAMES:
            raise ValueError(f"per-tau equilibration stage {name!r} is not one of "
                             f"{', '.join(PER_TAU_STAGE_NAMES)}")
        steps = stage.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"per-tau equilibration stage {name}: steps must be a positive "
                             f"integer, got {steps!r}")
        strength = float(stage.get("restraint_kcal_per_mol_A2") or 0.0)
        if strength < 0.0:
            raise ValueError(f"per-tau equilibration stage {name}: the restraint strength must "
                             f"be >= 0, got {strength}")
        checked.append({"name": name, "steps": int(steps), "restraint_kcal_per_mol_A2": strength})
    names = [stage["name"] for stage in checked]
    if names != sorted(names, key=PER_TAU_STAGE_NAMES.index) or len(set(names)) != len(names):
        raise ValueError(f"per-tau equilibration stages must run in the order "
                         f"{', '.join(PER_TAU_STAGE_NAMES)}, each at most once; got {names}")
    return checked
