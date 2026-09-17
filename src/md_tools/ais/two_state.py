"""The AIS Hamiltonian: a linear transformation between two end-state Systems.

    V(lambda, x) = (1 - lambda) * V0(x) + lambda * V1(x),        lambda in [0, 1]
    dV/dlambda   = V1(x) - V0(x)

V0 is `-s`/`-p`, the state the source ensemble was sampled from. V1 is `-s2`/`-p2`. They describe
the SAME particles -- same count, order, masses, constraints and virtual sites -- and differ only
in parameters. This is Amber's no-softcore linear mixing (Amber26 §27.1, Eq. 27.3); atom
mapping, softcore and dummy atoms are not implemented and a pair that would need them is refused.

WHY THIS AND NOT TAU

The previous implementation switched ONE System along a REST2 `tau`, which made the potential a
quadratic in `a = 1 - tau` and needed a three-point basis probe to decompose it, an analytic
dispersion-tail correction to switch explicit solvent cheaply, and a REST2 scaler inside AIS. Two
topology files and a straight line between them need none of that: the potential is LINEAR in
lambda at every coordinate, PME and the dispersion correction included, because each end state is
evaluated exactly as its own System would evaluate it.

THE WORK

    delta_W_j = V(lambda_{j+1}, x_j) - V(lambda_j, x_j)
              = (lambda_{j+1} - lambda_j) * [V1(x_j) - V0(x_j)]

The first line is the contract; the second is an identity for linear mixing, and it is how the
work is measured -- one evaluation of dV/dlambda at the frozen pre-switch coordinate. It is also
exactly Amber's Eq. 27.30, `(dU/dlambda) * dlambda`: the O(dlambda^2) difference between that
gradient convention and a finite difference vanishes for a linear path. The finite difference stays
the definition so a later non-linear schedule cannot silently change what `work` means.

HOW THE MIXTURE IS BUILT

Forces are paired by index. A pair whose two serialisations are identical is added once, unscaled
-- `(1 - lambda) F + lambda F = F` exactly. Every pair that differs becomes two collective variables
of ONE `CustomCVForce`, `(1 - lambda) * U0 + lambda * U1`, with an energy-parameter derivative.
Moving lambda is `Context.setParameter`: nothing is re-uploaded, for any System.

WHAT IT COSTS, measured (6232 particles, explicit TIP3P/PME, RTX 3080, mixed precision): a lambda
update with its work read and one step takes 1.40 ms, against 7.77 ms for the previous tau switch;
dynamics alone take 0.48 ms against 0.17 ms for plain V0, because the differing NonbondedForce is
evaluated twice -- as Amber computes the reciprocal sum twice.

WHAT IT CANNOT DO: resume bit-for-bit on CUDA. The inner Contexts of a `CustomCVForce` keep atom
ordering state that no checkpoint captures, so a force differs in its last bits the moment a run
resumes. A resume restores the committed generation exactly -- lambda, accumulated work, counters,
positions, velocities -- and continues the same switching process as a new realisation. Decided
with the user on 2026-09-16; see docs/amber-like-fix/AIS-two-topology.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["LAMBDA_PARAMETER", "TWO_STATE_SCHEMA", "HISTORICAL_SCHEMAS", "EndStateError",
           "SchemaError", "TwoStateHamiltonian", "EvaluationCounters",
           "OBSERVATION_POTENTIAL_COLUMNS", "HS_COLUMNS", "require_compatible_schema",
           "end_state_potentials", "identity_tolerance", "IDENTITY_TOLERANCES"]

#: The one global parameter the mixed System carries. Named for its owner so it cannot collide
#: with a parameter either end state already declares -- which is refused below regardless.
LAMBDA_PARAMETER = "ais_lambda"

#: What a record written by this build measured. On every checkpoint, manifest and run identity.
TWO_STATE_SCHEMA = {
    "name": "two-state-linear",
    "version": 1,
    "state_coordinate": "lambda",
    "lambda_start": 0.0,
    "lambda_end": 1.0,
    "hamiltonian": "V(lambda, x) = (1 - lambda) * V0(x) + lambda * V1(x)",
    "work_convention": ("delta_W_j = V(lambda_{j+1}, x_j) - V(lambda_j, x_j) "
                        "= (lambda_{j+1} - lambda_j) * (V1(x_j) - V0(x_j)); parameters move at "
                        "frozen coordinates, then the configuration propagates"),
    "end_states": {"V0": "-s / -p: the state the source ensemble was sampled from",
                   "V1": "-s2 / -p2"},
    "observation_potentials": ("potential_v0_kj_mol and potential_v1_kj_mol are the end-state "
                               "potentials at the coordinate the row SAVED; "
                               "potential_direct_kj_mol is V at that coordinate under the row's "
                               "lambda. Empty when the row saved no coordinate."),
    "units": "kJ/mol",
}

#: Records this build refuses to continue, and why. Nothing here rewrites them.
HISTORICAL_SCHEMAS = {
    ("rest2-lambda-basis", 2): (
        "it was written by the single-topology AIS, which switched one REST2 System along tau. "
        "Its work was measured along a different path -- linear in tau, quadratic in the "
        "Hamiltonian -- so no row of it can be continued under a linear two-state switch."),
    ("rest2-tau-quadratic-basis", 1): (
        "it was written by the single-topology AIS and additionally paired the pre-switch work "
        "basis with the wrong saved coordinate."),
}

#: The observation potentials, at the coordinate a row names. Empty when it named none: a row
#: with no saved configuration has no configuration for a potential to belong to.
OBSERVATION_POTENTIAL_COLUMNS = (
    "potential_v0_kj_mol",
    "potential_v1_kj_mol",
    "potential_direct_kj_mol",
)

#: The frame-aligned table a Hummer-Szabo reweighting reads. Only rows with a saved coordinate.
HS_COLUMNS = (
    "path_id", "source_frame", "observation_index", "switch_step",
    "coordinate_frame_index", "lambda", "total_work_kj_mol",
) + OBSERVATION_POTENTIAL_COLUMNS

#: `(relative, absolute floor)` in kJ/mol, by OpenMM precision, for the one identity checked at
#: run time: the end-state difference from the parameter derivative against the same difference
#: from the collective-variable values. Two kernels, one number.
IDENTITY_TOLERANCES = {
    "double": (1.0e-9, 1.0e-6),
    "mixed": (1.0e-6, 1.0e-3),
    "single": (5.0e-5, 1.0e-1),
}


def identity_tolerance(scale: float, *, precision: str = "mixed") -> float:
    """How far two measurements of the same energy may differ at this scale and precision."""
    relative, floor = IDENTITY_TOLERANCES.get(str(precision or "mixed").lower(),
                                              IDENTITY_TOLERANCES["mixed"])
    return max(abs(float(scale)) * relative, floor)


class EndStateError(ValueError):
    """Two Systems that are not a parameter-only pair."""


class SchemaError(ValueError):
    """A record written under a schema this build does not continue."""


def require_compatible_schema(recorded: Any, *, what: str) -> None:
    """Refuse a checkpoint, manifest or identity not written under `TWO_STATE_SCHEMA`."""
    schema = recorded.get("ais_schema") if isinstance(recorded, dict) else None
    if schema is None and isinstance(recorded, dict):
        schema = recorded.get("decomposition_schema")
    name = schema.get("name") if isinstance(schema, dict) else None
    version = schema.get("version") if isinstance(schema, dict) else None
    if (name, version) == (TWO_STATE_SCHEMA["name"], TWO_STATE_SCHEMA["version"]):
        return
    reason = HISTORICAL_SCHEMAS.get((name, version))
    if reason is None and name is None:
        reason = "it records no AIS schema at all, so what it measured cannot be established."
    elif reason is None:
        reason = "that schema is not one this build implements."
    raise SchemaError(
        f"{what} was written under {name}/v{version}, and this build writes "
        f"{TWO_STATE_SCHEMA['name']}/v{TWO_STATE_SCHEMA['version']}: {reason}\n"
        f"  The recorded run is left exactly as it is. Start a new -odir rather than continuing "
        f"one written by another AIS.")


def end_state_potentials(direct: float, difference: float, lam: float) -> tuple[float, float]:
    """`(V0, V1)` from `V(lambda)` and `V1 - V0` at the same coordinate. Exact for linear mixing."""
    lam = float(lam)
    return float(direct) - lam * float(difference), float(direct) + (1.0 - lam) * float(difference)


# ---------------------------------------------------------------------------------------------
# the pair
# ---------------------------------------------------------------------------------------------

_BAROSTATS = ("MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
              "MonteCarloMembraneBarostat", "MonteCarloFlexibleBarostat")


def _without_forces(system) -> str:
    """The System's XML with every Force removed: particles, masses, constraints, virtual sites,
    and the default box. Everything a parameter-only pair must share, compared in one string."""
    from openmm import XmlSerializer

    copy = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    for index in reversed(range(copy.getNumForces())):
        copy.removeForce(index)
    return XmlSerializer.serialize(copy)


def _force_xml(force) -> str:
    """A Force's serialisation with its group normalised. The group is a reporting label, not a
    parameter, and two builds of the same force may disagree about it."""
    from openmm import XmlSerializer

    copy = XmlSerializer.deserialize(XmlSerializer.serialize(force))
    copy.setForceGroup(0)
    return XmlSerializer.serialize(copy)


def _structural_differences(system0, system1) -> list[str]:
    """Why these two Systems are not the same particles, in words. Empty when they are."""
    from openmm import unit

    problems: list[str] = []
    n0, n1 = system0.getNumParticles(), system1.getNumParticles()
    if n0 != n1:
        return [f"V0 has {n0} particle(s) and V1 has {n1}"]
    masses = [i for i in range(n0)
              if system0.getParticleMass(i).value_in_unit(unit.dalton)
              != system1.getParticleMass(i).value_in_unit(unit.dalton)]
    if masses:
        problems.append(
            f"{len(masses)} particle mass(es) differ (first: particle {masses[0]}). The work "
            f"convention assumes masses that do not depend on lambda, so hydrogen mass "
            f"repartitioning must be the same in both builds")
    c0, c1 = system0.getNumConstraints(), system1.getNumConstraints()
    if c0 != c1:
        problems.append(f"V0 has {c0} constraint(s) and V1 has {c1}")
    else:
        for index in range(c0):
            a0, b0, d0 = system0.getConstraintParameters(index)
            a1, b1, d1 = system1.getConstraintParameters(index)
            if (a0, b0) != (a1, b1) or d0 != d1:
                problems.append(f"constraint {index} differs ({a0}-{b0} against {a1}-{b1})")
                break
    sites = [i for i in range(n0) if system0.isVirtualSite(i) != system1.isVirtualSite(i)]
    if sites:
        problems.append(f"virtual sites differ (first: particle {sites[0]})")
    if not problems and _without_forces(system0) != _without_forces(system1):
        problems.append("the default periodic box or a virtual-site definition differs")
    return problems


def _nonbonded_settings(force) -> dict[str, Any] | None:
    """The settings of a nonbonded force that are NOT parameters: a pair differing in these is two
    different treatments of long-range interactions, not two parameterisations of one."""
    from openmm import CustomGBForce, CustomNonbondedForce, NonbondedForce

    if isinstance(force, NonbondedForce):
        return {
            "method": force.getNonbondedMethod(),
            "cutoff": force.getCutoffDistance()._value,
            "switching": force.getUseSwitchingFunction(),
            "switching_distance": force.getSwitchingDistance()._value,
            "dispersion_correction": force.getUseDispersionCorrection(),
            "ewald_error_tolerance": force.getEwaldErrorTolerance(),
            "pme": tuple(float(v) if not hasattr(v, "_value") else float(v._value)
                         for v in force.getPMEParameters()),
            "exception_pairs": tuple(tuple(force.getExceptionParameters(i)[:2])
                                     for i in range(force.getNumExceptions())),
        }
    if isinstance(force, CustomNonbondedForce):
        return {
            "method": force.getNonbondedMethod(),
            "cutoff": force.getCutoffDistance()._value,
            "switching": force.getUseSwitchingFunction(),
            "exclusions": tuple(tuple(force.getExclusionParticles(i))
                                for i in range(force.getNumExclusions())),
        }
    if isinstance(force, CustomGBForce):
        return {
            "method": force.getNonbondedMethod(),
            "cutoff": force.getCutoffDistance()._value,
            "exclusions": tuple(tuple(force.getExclusionParticles(i))
                                for i in range(force.getNumExclusions())),
        }
    return None


def _global_parameter_names(force) -> set[str]:
    names = set()
    if hasattr(force, "getNumGlobalParameters"):
        for index in range(force.getNumGlobalParameters()):
            names.add(force.getGlobalParameterName(index))
    return names


def pair_plan(system0, system1) -> dict[str, Any]:
    """Which forces are shared and which are mixed, or `EndStateError` naming every problem.

    Checked, in order, because each is a way a pair produces a plausible work table and a wrong
    answer: the particles, masses, constraints and virtual sites; a barostat in either (switching
    is at fixed volume); the force list, class by class; the long-range settings of every
    nonbonded force; a global parameter named like ours; and a pair with nothing to switch.
    """
    problems = _structural_differences(system0, system1)

    for label, system in (("V0", system0), ("V1", system1)):
        barostats = [type(system.getForce(i)).__name__ for i in range(system.getNumForces())
                     if type(system.getForce(i)).__name__ in _BAROSTATS]
        if barostats:
            problems.append(
                f"{label} carries a barostat ({barostats[0]}). AIS switches at FIXED VOLUME: no "
                f"pressure-volume term enters the work. Build the System without one")

    f0, f1 = system0.getNumForces(), system1.getNumForces()
    classes0 = [type(system0.getForce(i)).__name__ for i in range(f0)]
    classes1 = [type(system1.getForce(i)).__name__ for i in range(f1)]
    if classes0 != classes1:
        problems.append(
            f"the force lists differ: V0 has {classes0} and V1 has {classes1}. Forces are paired "
            f"by position, so both end states must be built with the same force layout")

    shared: list[int] = []
    mixed: list[int] = []
    if classes0 == classes1:
        for index in range(f0):
            force0, force1 = system0.getForce(index), system1.getForce(index)
            if LAMBDA_PARAMETER in (_global_parameter_names(force0)
                                    | _global_parameter_names(force1)):
                problems.append(f"force {index} ({classes0[index]}) already declares a global "
                                f"parameter named {LAMBDA_PARAMETER!r}")
            settings0, settings1 = _nonbonded_settings(force0), _nonbonded_settings(force1)
            if settings0 != settings1:
                which = sorted(key for key in settings0 if settings0[key] != settings1[key])
                problems.append(
                    f"force {index} ({classes0[index]}) differs in {', '.join(which)}, which are "
                    f"not parameters: the two end states would treat interactions differently, "
                    f"not with different strengths")
            if _force_xml(force0) == _force_xml(force1):
                shared.append(index)
            else:
                mixed.append(index)
        if not problems and not mixed:
            problems.append(
                "V0 and V1 are the same Hamiltonian: every force is identical, so every path "
                "would measure exactly zero work")

    if problems:
        raise EndStateError(
            "the two end states are not a parameter-only pair:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\n  AIS mixes V(lambda) = (1 - lambda) V0 + lambda V1 over identical particles. "
              "Atom mapping, softcore and dummy atoms are not implemented.")
    return {"shared_forces": shared, "mixed_forces": mixed,
            "mixed_force_classes": [classes0[i] for i in mixed]}


class TwoStateHamiltonian:
    """V0 and V1 mixed into one System whose lambda is a Context parameter.

    Built ONCE per launch from the two deserialised end states, after `pair_plan` has accepted
    them. `system` is what every path's Context is created from.
    """

    def __init__(self, system0, system1):
        from openmm import CustomCVForce, XmlSerializer

        self.plan = pair_plan(system0, system1)
        mixed = XmlSerializer.deserialize(XmlSerializer.serialize(system0))
        for index in reversed(range(mixed.getNumForces())):
            mixed.removeForce(index)
        for index in self.plan["shared_forces"]:
            mixed.addForce(XmlSerializer.deserialize(
                XmlSerializer.serialize(system0.getForce(index))))

        names0 = [f"V0_force{index}" for index in self.plan["mixed_forces"]]
        names1 = [f"V1_force{index}" for index in self.plan["mixed_forces"]]
        expression = (f"(1 - {LAMBDA_PARAMETER}) * end0 + {LAMBDA_PARAMETER} * end1; "
                      f"end0 = {' + '.join(names0)}; end1 = {' + '.join(names1)}")
        switch = CustomCVForce(expression)
        switch.addGlobalParameter(LAMBDA_PARAMETER, 0.0)
        switch.addEnergyParameterDerivative(LAMBDA_PARAMETER)
        for index, name0, name1 in zip(self.plan["mixed_forces"], names0, names1):
            switch.addCollectiveVariable(name0, XmlSerializer.deserialize(
                XmlSerializer.serialize(system0.getForce(index))))
            switch.addCollectiveVariable(name1, XmlSerializer.deserialize(
                XmlSerializer.serialize(system1.getForce(index))))
        switch.setName("AIS two-state switch")
        self.switch_index = mixed.addForce(switch)
        self.system = mixed
        self._n_mixed = len(names0)

    def record(self) -> dict[str, Any]:
        """For the run log: what is shared, what is mixed, and how."""
        return {"schema": {"name": TWO_STATE_SCHEMA["name"],
                           "version": TWO_STATE_SCHEMA["version"]},
                "shared_forces": list(self.plan["shared_forces"]),
                "mixed_forces": list(self.plan["mixed_forces"]),
                "mixed_force_classes": list(self.plan["mixed_force_classes"]),
                "mechanism": f"CustomCVForce over the differing force pairs, global "
                             f"{LAMBDA_PARAMETER!r} with an energy-parameter derivative"}

    @staticmethod
    def set_lambda(context, value: float) -> None:
        context.setParameter(LAMBDA_PARAMETER, float(value))

    @staticmethod
    def difference(context) -> float:
        """`V1(x) - V0(x)` at the Context's coordinates, in kJ/mol: dV/dlambda. One evaluation."""
        state = context.getState(getParameterDerivatives=True)
        return float(state.getEnergyParameterDerivatives()[LAMBDA_PARAMETER])

    def observe(self, context, lam: float, *, precision: str = "mixed",
                where: str = "") -> dict[str, float]:
        """The observation potentials at the Context's coordinates, with the identity CHECKED.

        `V(lambda)` and `dV/dlambda` come from one `getState`; `V0` and `V1` follow exactly. The
        check is against an independent measurement of the same difference -- the collective
        variable values, evaluated by the inner Contexts rather than by the derivative kernel.
        If a force ever carried lambda-dependence outside the linear mixture, this is where it
        would show, at the first saved frame, rather than as a work table for a Hamiltonian
        nothing ran under.
        """
        from openmm import unit

        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        direct = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        difference = float(state.getEnergyParameterDerivatives()[LAMBDA_PARAMETER])
        values = list(context.getSystem().getForce(self.switch_index)
                      .getCollectiveVariableValues(context))
        end0 = sum(values[0::2])
        end1 = sum(values[1::2])
        allowed = identity_tolerance(max(abs(end0), abs(end1)), precision=precision)
        if abs((end1 - end0) - difference) > allowed:
            raise SystemExit(
                f"{where}the two-state identity does not hold at lambda = {lam}: dV/dlambda is "
                f"{difference:.6f} kJ/mol from the derivative kernel and {end1 - end0:.6f} from "
                f"the end-state energies of the mixed forces; the difference "
                f"{(end1 - end0) - difference:.3e} exceeds the {precision}-precision tolerance "
                f"{allowed:.3e}. Refusing rather than recording potentials the path did not run "
                f"under.")
        v0, v1 = end_state_potentials(direct, difference, lam)
        return {"potential_v0_kj_mol": v0, "potential_v1_kj_mol": v1,
                "potential_direct_kj_mol": direct}


@dataclass
class EvaluationCounters:
    """What a path's evaluations cost, split so no number double-counts another.

        useful_total = work_derivative + observation_potential + other_useful
        paid_total   = useful_total + known_discarded

    A committed checkpoint cannot know what a process evaluated after it and before dying, so a
    resumed path's discarded count is marked incomplete rather than reported as zero.
    """

    #: One `getState(getParameterDerivatives=True)` per switch, at the frozen pre-switch coordinate.
    work_derivative_evaluations: int = 0
    #: At each frame-aligned observation: V and dV/dlambda in one call, plus the collective
    #: variable values that check them. Counted as two.
    observation_potential_energy_evaluations: int = 0
    other_useful_energy_evaluations: int = 0
    known_discarded_energy_evaluations: int = 0
    discarded_is_complete: bool = True
    #: `Context.setParameter` pushes. Not energy evaluations, and cheap, but counted.
    parameter_updates: int = 0
    parameter_change_seconds: float = 0.0

    @property
    def useful_total(self) -> int:
        return (self.work_derivative_evaluations + self.observation_potential_energy_evaluations
                + self.other_useful_energy_evaluations)

    @property
    def paid_total(self) -> int:
        return self.useful_total + self.known_discarded_energy_evaluations

    def record(self) -> dict[str, Any]:
        return {
            "work_derivative_evaluations": self.work_derivative_evaluations,
            "observation_potential_energy_evaluations":
                self.observation_potential_energy_evaluations,
            "other_useful_energy_evaluations": self.other_useful_energy_evaluations,
            "useful_total_energy_evaluations": self.useful_total,
            "known_discarded_energy_evaluations": self.known_discarded_energy_evaluations,
            "paid_total_energy_evaluations": self.paid_total,
            "discarded_is_complete": bool(self.discarded_is_complete),
            "parameter_updates": self.parameter_updates,
            "parameter_change_seconds": round(self.parameter_change_seconds, 6),
            "relationships": [
                "useful_total = work_derivative + observation_potential + other_useful",
                "paid_total = useful_total + known_discarded",
            ],
        }

    @staticmethod
    def from_record(entry) -> "EvaluationCounters":
        """Restore committed counters. They come back as USEFUL, which is what they are."""
        entry = entry or {}
        return EvaluationCounters(
            work_derivative_evaluations=int(entry.get("work_derivative_evaluations", 0)),
            observation_potential_energy_evaluations=int(
                entry.get("observation_potential_energy_evaluations", 0)),
            other_useful_energy_evaluations=int(entry.get("other_useful_energy_evaluations", 0)),
            known_discarded_energy_evaluations=int(
                entry.get("known_discarded_energy_evaluations", 0)),
            discarded_is_complete=bool(entry.get("discarded_is_complete", True)),
            parameter_updates=int(entry.get("parameter_updates", 0)),
            parameter_change_seconds=float(entry.get("parameter_change_seconds", 0.0)))

    def __add__(self, other: "EvaluationCounters") -> "EvaluationCounters":
        return EvaluationCounters(
            work_derivative_evaluations=(self.work_derivative_evaluations
                                         + other.work_derivative_evaluations),
            observation_potential_energy_evaluations=(
                self.observation_potential_energy_evaluations
                + other.observation_potential_energy_evaluations),
            other_useful_energy_evaluations=(self.other_useful_energy_evaluations
                                             + other.other_useful_energy_evaluations),
            known_discarded_energy_evaluations=(self.known_discarded_energy_evaluations
                                                + other.known_discarded_energy_evaluations),
            discarded_is_complete=bool(self.discarded_is_complete
                                       and other.discarded_is_complete),
            parameter_updates=self.parameter_updates + other.parameter_updates,
            parameter_change_seconds=(self.parameter_change_seconds
                                      + other.parameter_change_seconds))
