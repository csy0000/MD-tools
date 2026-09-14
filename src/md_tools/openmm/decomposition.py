"""The potential energy term by term, the way Amber's `mdout` prints it. POST-HOC ONLY.

WHY THIS IS NOT A RUNTIME REPORTER. OpenMM separates energies by FORCE GROUP, and a group can only
split what sits in different Force objects. `energy_components.csv` therefore reaches bonds, angles,
torsions, nonbonded direct space and -- for free -- PME reciprocal space, and stops there: every
1-4 pair is an *exception inside* the single `NonbondedForce`, which evaluates charge and dispersion
in one kernel, so `EELEC`, `VDWAALS`, `1-4 EEL` and `1-4 NB` are one number at run time whatever
the grouping. Amber prints them apart because its energy routines are written term by term.

Reaching them here needs the System RESTRUCTURED -- forces duplicated with charges or epsilons
zeroed -- and that must never happen to a System something integrates: it would change
`system_sha256`, invalidate every checkpoint fingerprint in flight, and make registered data
incomparable with new data. So this module does it on a COPY that is only ever asked for energies,
which is also why it is post-hoc: analysis belongs after the simulation, not inside it.

WHY THE SPLIT IS EXACT RATHER THAN APPROXIMATE. A nonbonded energy is additive in its two physical
parts, because each depends on disjoint parameters:

    U_nb(q, epsilon) = U_elec(q) + U_LJ(epsilon)

Zeroing every epsilon leaves electrostatics alone -- including the PME reciprocal sum and the Ewald
self-energy, both of which are functions of charge only. Zeroing every charge leaves Lennard-Jones
alone, including the long-range dispersion correction, which OpenMM computes from the stored
epsilons. Neither term borrows a parameter from the other, so the two evaluations sum to the
original to floating-point precision, and `decompose` checks exactly that rather than asserting it.

The same argument splits the 1-4 terms out: an exception REPLACES the standard interaction for its
pair, so a copy carrying only the exceptions -- every other interaction excluded -- is the 1-4
contribution and nothing else.

NO NEW COMMAND. `md-openmm` has its four public work commands and `export-reference`; this is an
import, used from a notebook or a script over frames already written. See CLAUDE.md.
"""
from __future__ import annotations

from typing import Any

__all__ = ["AMBER_TERM_ORDER", "decompose", "decomposition_systems"]

#: The order Amber's `mdout` prints, so a reader comparing two engines reads down one column.
#: `EHBOND` is absent: Amber has printed a constant 0.0 for it for over a decade.
AMBER_TERM_ORDER = (
    "BOND", "ANGLE", "DIHED", "1-4 NB", "1-4 EEL", "VDWAALS", "EELEC", "RESTRAINT",
)

#: Force class names that are stage machinery rather than terms of the molecular Hamiltonian.
#: A barostat carries no potential energy; a CM remover carries none either. The restraint DOES
#: carry energy and is reported under its own Amber name.
_MACHINERY = frozenset({"MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
                        "MonteCarloFlexibleBarostat", "MonteCarloMembraneBarostat",
                        "CMMotionRemover"})
_RESTRAINT_CLASSES = frozenset({"CustomExternalForce", "CustomTorsionForce"})


def _clone(system):
    """A deep copy, by the one route OpenMM guarantees."""
    from openmm import XmlSerializer

    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def decomposition_systems(system) -> dict[str, Any]:
    """Systems whose single-point energies are the individual Amber-style terms.

    Returned rather than evaluated so a caller may reuse them across many frames: building these
    is the expensive part, and a trajectory sweep should pay it once.

    Each is a separate copy. `system` is not touched, and none of these is ever integrated.
    """
    import openmm

    built: dict[str, Any] = {}

    # -- the bonded terms, which are already separate Force objects ----------------------------
    bonded = {"BOND": "HarmonicBondForce", "ANGLE": "HarmonicAngleForce",
              "DIHED": "PeriodicTorsionForce"}
    for term, class_name in bonded.items():
        copy = _clone(system)
        _keep_only(copy, lambda f, n=class_name: type(f).__name__ == n)
        built[term] = copy

    # CMAP joins DIHED where present: Amber folds it into the dihedral column too.
    if any(type(f).__name__ == "CMAPTorsionForce" for f in system.getForces()):
        copy = _clone(system)
        _keep_only(copy, lambda f: type(f).__name__ in ("PeriodicTorsionForce",
                                                        "CMAPTorsionForce"))
        built["DIHED"] = copy

    # -- the restraint, under its own name ------------------------------------------------------
    if any(type(f).__name__ in _RESTRAINT_CLASSES for f in system.getForces()):
        copy = _clone(system)
        _keep_only(copy, lambda f: type(f).__name__ in _RESTRAINT_CLASSES)
        built["RESTRAINT"] = copy

    # -- the four nonbonded terms ---------------------------------------------------------------
    #
    # Two independent splits, applied together: charge-only against epsilon-only, and
    # exceptions-only against everything-but-exceptions. The four combinations are exactly
    # Amber's EELEC, VDWAALS, 1-4 EEL and 1-4 NB.
    for term, keep_charges, exceptions_only in (
            ("EELEC", True, False), ("VDWAALS", False, False),
            ("1-4 EEL", True, True), ("1-4 NB", False, True)):
        copy = _clone(system)
        _keep_only(copy, lambda f: isinstance(f, openmm.NonbondedForce))
        if copy.getNumForces() == 0:
            continue
        _restrict_nonbonded(copy.getForce(0), keep_charges=keep_charges,
                            exceptions_only=exceptions_only)
        built[term] = copy

    return built


def _keep_only(system, predicate) -> None:
    """Remove every force the predicate rejects, plus all machinery. In place, on a copy."""
    for index in reversed(range(system.getNumForces())):
        force = system.getForce(index)
        if type(force).__name__ in _MACHINERY or not predicate(force):
            system.removeForce(index)
    for index in range(system.getNumForces()):
        # One group each, so a caller may also ask for them individually.
        system.getForce(index).setForceGroup(0)


def _restrict_nonbonded(force, *, keep_charges: bool, exceptions_only: bool) -> None:
    """Zero one half of a `NonbondedForce`, and optionally everything but its exceptions.

    Charges and epsilons are disjoint parameters, so zeroing one leaves the other's energy exactly
    -- with its reciprocal sum, self-energy or dispersion correction attached, since each of those
    is a function of that parameter alone.
    """
    from openmm import unit

    for particle in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(particle)
        if exceptions_only:
            # Every standard interaction removed; only the exception list can carry energy.
            force.setParticleParameters(particle, 0.0, sigma, 0.0)
            continue
        force.setParticleParameters(
            particle,
            charge if keep_charges else 0.0 * unit.elementary_charge,
            sigma,
            0.0 * unit.kilojoule_per_mole if keep_charges else epsilon)

    for index in range(force.getNumExceptions()):
        i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
        if not exceptions_only:
            # An exception replaces its pair's standard interaction; zeroing it removes that pair
            # from this term rather than double counting it.
            force.setExceptionParameters(index, i, j, 0.0, sigma, 0.0)
            continue
        force.setExceptionParameters(
            index, i, j,
            chargeprod if keep_charges else 0.0,
            sigma,
            0.0 if keep_charges else epsilon)


def decompose(system, positions, *, box_vectors=None, platform_name="CPU",
              systems=None, tolerance_kj_mol=1e-3) -> dict[str, Any]:
    """Every Amber-style term for one configuration, in kJ/mol, plus the total they sum to.

    `systems` may be a previously built `decomposition_systems(...)` mapping, to avoid rebuilding
    it per frame.

    THE SUM IS CHECKED, NOT ASSUMED. A decomposition that does not add up is worse than none: each
    individual number still looks plausible, so a missed force or a double-counted exception is
    invisible without this comparison. The residual is returned either way, and
    `tolerance_kj_mol` decides whether `exact` is True -- it is a floating-point comparison over a
    solvated system, not an equality.
    """
    from openmm import Context, LangevinMiddleIntegrator, Platform, unit

    prepared = systems if systems is not None else decomposition_systems(system)
    platform = Platform.getPlatformByName(platform_name)

    def energy(candidate):
        context = Context(candidate,
                          LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                          platform)
        if box_vectors is not None:
            context.setPeriodicBoxVectors(*box_vectors)
        context.setPositions(positions)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    terms = {name: energy(prepared[name]) for name in AMBER_TERM_ORDER if name in prepared}

    whole = _clone(system)
    _keep_only(whole, lambda f: True)
    total = energy(whole)
    summed = sum(terms.values())
    return {
        "terms": terms,
        "order": [name for name in AMBER_TERM_ORDER if name in terms],
        "sum_kj_mol": summed,
        "total_kj_mol": total,
        "residual_kj_mol": summed - total,
        "exact": abs(summed - total) <= float(tolerance_kj_mol),
        "units": "kJ/mol",
        "note": ("post-hoc decomposition on copies of the System; the integrated System is "
                 "untouched and its force groups are unchanged"),
    }
