"""The REST2 potential split into its three tau-basis components, for Hummer-Szabo reweighting.

WHY A DECOMPOSITION AT ALL

A switching path records one number per update: the total work. Reweighting a nonequilibrium
ensemble back onto a *different* Hamiltonian -- a different tau schedule, a different endpoint, or
a Jarzynski/Hummer-Szabo estimator evaluated at a tau the path never visited -- needs the potential
as a FUNCTION of tau, not its value at the one tau that was run. A single total is not enough
information to produce that function, and re-running the path at another tau is not reweighting.

THE EXACT BASIS

Write the coupling amplitude as

    a = 1 - tau

Every scale factor in this convention (`md_tools.rest2.scaler`) is a power of `a`:

    unscaled terms          a^0    bonds, angles, excluded omega torsions, environment-environment
    solute-environment      a^1    solute-environment nonbonded, the whole generalised-Born energy
    solute-solute           a^2    solute-solute nonbonded and 1-4, eligible solute torsions,
                                   solute CMAP

so at FROZEN COORDINATES the potential is exactly a quadratic polynomial in `a`:

    U(tau, x) = U_unscaled(x) + a U_linear(x) + a^2 U_quadratic(x)

This is an identity, not an approximation, and it holds for the whole `NonbondedForce` including
the PME reciprocal sum, the Ewald self-energy and the long-range dispersion correction. Each of
those is a quadratic form in the per-particle charge (or in sqrt(epsilon)), and the scaler
multiplies solute charges by `a` and solute epsilons by `a^2`; a pair term therefore carries
`a^(number of solute partners)`, which is exactly the three-way split above. Nothing needs to be
decomposed per-pair, and the validated force construction is not touched.

HOW IT IS MEASURED

Three potential-energy evaluations at three controlled amplitudes, at frozen coordinates, then one
quadratic fit. Three points determine a quadratic exactly, so the fit introduces no model error:
the only error is the floating-point error of the three evaluations themselves.

The amplitudes are `0`, `1/2` and `1`, chosen to be as far apart as the domain allows -- the fit
divides by their differences, so widely separated nodes are the difference between a stable
measurement and one dominated by cancellation. `a = 0` is worth having on its own: it switches
every scaled term off, so it reads `U_unscaled` directly rather than inferring it.

This costs three extra energy evaluations per switching update and no extra integration steps.
The count and its measured wall-clock cost are recorded in provenance
(`potential_energy_evaluations_per_update`, `switching_energy_evaluations`) rather than left for a
reader to guess.

WHAT IS CHECKED RATHER THAN ASSUMED

The total work stays what it always was: `U(tau_{j+1}, x_j) - U(tau_j, x_j)`, measured directly
from two evaluations of the real Hamiltonian. The component works are derived independently, from
the fit. The two are then required to agree:

    delta_W_total = delta_W_unscaled + delta_W_linear + delta_W_quadratic

Deriving the total FROM the components would make that identity true by construction and it would
test nothing. Measured separately, it is a genuine check on the whole chain -- and it fails if any
force ever acquires a tau dependence the three-group model does not describe.

`delta_W_unscaled` is identically zero: the unscaled component does not depend on tau, and a
parameter update happens at frozen coordinates. The column is kept anyway, because "this number is
zero" is a fact a reader should be able to confirm from the file rather than take on trust, and
because a nonzero value there is the signature of exactly the defect above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["BASIS_PROBE_AMPLITUDES", "POTENTIAL_ENERGY_EVALUATIONS_PER_UPDATE",
           "DECOMPOSITION_SCHEMA", "COMPONENT_OBSERVATION_COLUMNS", "COMPONENT_WORK_COLUMNS",
           "COMPONENT_SUMMARY_COLUMNS", "Components", "ComponentWork", "ComponentProbe",
           "quadratic_through", "reconstruction_tolerance", "require_compatible_schema",
           "RECONSTRUCTION_TOLERANCES", "DecompositionError"]


#: The three amplitudes probed, in the order they are applied. As far apart as the domain allows:
#: the fit divides by their pairwise differences.
BASIS_PROBE_AMPLITUDES = (0.0, 0.5, 1.0)

#: Extra potential-energy evaluations per switching update, beyond the two the total work already
#: needs. Recorded in provenance because it is a real cost, not an implementation detail.
POTENTIAL_ENERGY_EVALUATIONS_PER_UPDATE = 3

#: The schema of the component columns. Written into every record that carries them, and checked
#: on resume: accumulating components measured under one basis definition onto totals measured
#: under another would produce a work integral for a Hamiltonian nothing ever ran.
DECOMPOSITION_SCHEMA = {
    "name": "rest2-tau-quadratic-basis",
    "version": 1,
    "state_coordinate": "tau",
    "amplitude": "a = 1 - tau",
    "identity": "U(tau, x) = U_unscaled(x) + a*U_linear(x) + a^2*U_quadratic(x)",
    "work_identity": "dW_total = dW_unscaled + dW_linear + dW_quadratic",
    "unscaled_terms": ["bonds", "angles", "excluded omega torsions",
                       "environment-environment nonbonded"],
    "linear_terms": ["solute-environment nonbonded", "generalized Born"],
    "quadratic_terms": ["solute-solute nonbonded", "solute-solute 1-4",
                        "eligible solute torsions", "solute CMAP"],
    "probe_amplitudes": list(BASIS_PROBE_AMPLITUDES),
    "potential_energy_evaluations_per_update": POTENTIAL_ENERGY_EVALUATIONS_PER_UPDATE,
    "units": "kJ/mol",
}

#: Added to every per-path observation row. Appended after the existing columns, never inserted
#: among them: a reader keyed on position must keep reading the same numbers it always did.
COMPONENT_OBSERVATION_COLUMNS = (
    "potential_unscaled_kj_mol",
    "potential_linear_basis_kj_mol",
    "potential_quadratic_basis_kj_mol",
    "potential_linear_contribution_kj_mol",
    "potential_quadratic_contribution_kj_mol",
    "potential_total_reconstructed_kj_mol",
    "delta_work_unscaled_kj_mol",
    "delta_work_linear_kj_mol",
    "delta_work_quadratic_kj_mol",
    "total_work_unscaled_kj_mol",
    "total_work_linear_kj_mol",
    "total_work_quadratic_kj_mol",
)

#: The same twelve on the global work table, which is assembled from those rows.
COMPONENT_WORK_COLUMNS = COMPONENT_OBSERVATION_COLUMNS

#: Final per-path totals, on `AIS_paths.csv`.
COMPONENT_SUMMARY_COLUMNS = ("total_work_unscaled_kj_mol", "total_work_linear_kj_mol",
                             "total_work_quadratic_kj_mol", "decomposition_schema_version")


class DecompositionError(ValueError):
    """A component record that cannot be combined with this build's basis."""


def require_compatible_schema(recorded: Any, *, what: str = "this run") -> None:
    """Refuse to continue a path whose components were measured under another basis.

    An absent schema is an OLD record, not a compatible one: the columns did not exist, so there
    are no components to continue and the accumulators would start from a total that was never
    decomposed. Refused by name rather than defaulted.
    """
    if not isinstance(recorded, dict) or "decomposition_schema" not in recorded:
        raise DecompositionError(
            f"{what} carries no component-decomposition schema, so it was written before the "
            f"tau-basis columns existed. Its cumulative component work is unknown and cannot be "
            f"continued -- resuming would add components measured now onto a total measured under "
            f"no decomposition at all.\n"
            f"  Delete the path directory to rerun it from its source frame, or read the old run "
            f"as it stands; this build writes "
            f"{DECOMPOSITION_SCHEMA['name']}/v{DECOMPOSITION_SCHEMA['version']}.")
    schema = recorded["decomposition_schema"]
    name = schema.get("name") if isinstance(schema, dict) else None
    version = schema.get("version") if isinstance(schema, dict) else None
    if (name, version) == (DECOMPOSITION_SCHEMA["name"], DECOMPOSITION_SCHEMA["version"]):
        return
    raise DecompositionError(
        f"{what} records the decomposition schema {name}/v{version}, but this build implements "
        f"{DECOMPOSITION_SCHEMA['name']}/v{DECOMPOSITION_SCHEMA['version']}. The component "
        f"definitions differ, so the two sets of numbers are not summable.\n"
        f"  The recorded run is left exactly as it is. Start a new dataset rather than continuing "
        f"one written under a different basis.")


def quadratic_through(nodes, values):
    """The coefficients `(c0, c1, c2)` of the quadratic through three `(node, value)` points.

    Written as an explicit Lagrange expansion rather than a matrix solve so the whole computation
    is three divisions by pairwise differences, visible in the source. Three points determine a
    quadratic exactly; there is no fitting residual to report and none is invented.
    """
    if len(nodes) != 3 or len(values) != 3:
        raise DecompositionError(f"a quadratic needs exactly 3 points; got {len(nodes)}")
    (a0, a1, a2) = (float(n) for n in nodes)
    if len({a0, a1, a2}) != 3:
        raise DecompositionError(
            f"the three probe amplitudes must be distinct; got {(a0, a1, a2)}. Repeated nodes "
            f"leave the quadratic undetermined rather than merely ill-conditioned.")
    nodes = (a0, a1, a2)
    others = ((a1, a2), (a0, a2), (a0, a1))
    c0 = c1 = c2 = 0.0
    for position in range(3):
        own = nodes[position]
        other, third = others[position]
        value = float(values[position])
        denominator = (own - other) * (own - third)
        # (a - other)(a - third) / denominator, expanded: the Lagrange basis polynomial for
        # this node, weighted by its value.
        c2 += value / denominator
        c1 += -value * (other + third) / denominator
        c0 += value * other * third / denominator
    return c0, c1, c2


@dataclass(frozen=True)
class Components:
    """`U_unscaled`, `U_linear` and `U_quadratic` at one configuration, in kJ/mol.

    Coordinate-dependent and tau-independent: this IS the potential as a function of tau, which is
    the whole point of measuring it.
    """

    unscaled: float
    linear: float
    quadratic: float

    @staticmethod
    def from_probe(amplitudes, energies) -> "Components":
        c0, c1, c2 = quadratic_through(amplitudes, energies)
        return Components(unscaled=c0, linear=c1, quadratic=c2)

    def contributions_at(self, tau: float) -> tuple[float, float]:
        """The two scaled contributions AS THEY ENTER the potential at this tau.

        Distinct from the basis values: the basis is what multiplies the amplitude, the
        contribution is the product. A reader comparing energies wants the second; a reader
        reweighting to another tau wants the first. Both are written, because deriving one from
        the other requires knowing which convention produced the column.
        """
        amplitude = 1.0 - float(tau)
        return amplitude * self.linear, amplitude * amplitude * self.quadratic

    def total_at(self, tau: float) -> float:
        """`U(tau)` reconstructed from the components. Compared against the measured total."""
        linear, quadratic = self.contributions_at(tau)
        return self.unscaled + linear + quadratic

    def work_between(self, tau_before: float, tau_after: float) -> "ComponentWork":
        """The three incremental works for a parameter change at these frozen coordinates.

        `unscaled` is exactly `0.0`, and it is written as the literal it is. The unscaled
        component carries `a^0`, the coordinates did not move, so there is no arithmetic here that
        could produce anything else -- and computing it as `self.unscaled - self.unscaled` to make
        the column look measured would be dressing up a tautology as evidence.

        What actually has content is the SUM identity, checked by the caller against a total work
        measured independently from two evaluations of the real Hamiltonian:

            delta_W_total  ==  0 + delta_W_linear + delta_W_quadratic

        A force that secretly depended on tau -- the defect this column exists to expose -- would
        move the measured total away from the component sum and fail that check. The column is
        kept so the zero is visible in the file rather than implied by its absence.
        """
        before_linear, before_quadratic = self.contributions_at(tau_before)
        after_linear, after_quadratic = self.contributions_at(tau_after)
        return ComponentWork(unscaled=0.0,
                             linear=after_linear - before_linear,
                             quadratic=after_quadratic - before_quadratic)

    def record(self) -> dict[str, float]:
        return {"unscaled_kj_mol": self.unscaled, "linear_basis_kj_mol": self.linear,
                "quadratic_basis_kj_mol": self.quadratic}


@dataclass(frozen=True)
class ComponentWork:
    """Three works that must sum to the independently measured total."""

    unscaled: float
    linear: float
    quadratic: float

    @property
    def total(self) -> float:
        return self.unscaled + self.linear + self.quadratic

    def __add__(self, other: "ComponentWork") -> "ComponentWork":
        return ComponentWork(self.unscaled + other.unscaled, self.linear + other.linear,
                             self.quadratic + other.quadratic)

    @staticmethod
    def zero() -> "ComponentWork":
        return ComponentWork(0.0, 0.0, 0.0)


#: Documented numerical tolerance for both identities, as `(relative, absolute floor)` in kJ/mol,
#: by OpenMM precision mode. The reconstruction is a fixed linear combination of three energies
#: measured on one Context, so its error is that of the energies themselves, amplified by the
#: fit coefficients (whose magnitudes are 1, 3 and 4 for these nodes).
#:
#: `single` is deliberately loose. A solvated system's potential is order 1e5 kJ/mol and a
#: single-precision nonbonded sum carries roughly 1e-6 relative error, so a tolerance tighter than
#: this would report the platform's arithmetic as a defect in the decomposition.
RECONSTRUCTION_TOLERANCES = {
    "double": (1.0e-9, 1.0e-6),
    "mixed": (1.0e-6, 1.0e-3),
    "single": (5.0e-5, 1.0e-1),
}


def reconstruction_tolerance(scale: float, *, precision: str = "mixed") -> float:
    """How far the reconstructed total may sit from the measured one, at this energy scale."""
    relative, floor = RECONSTRUCTION_TOLERANCES.get(
        str(precision or "mixed").lower(), RECONSTRUCTION_TOLERANCES["mixed"])
    return max(abs(float(scale)) * relative, floor)


class ComponentProbe:
    """Measures the three basis components on a live Context.

    The probe drives the SAME switcher the path switches with, so a probe amplitude and a real tau
    reach the Context through one implementation. It always restores the amplitude it was told to
    restore, in a `finally`, because a probe that leaked its last amplitude would leave the path
    integrating under a Hamiltonian nobody chose -- and at `a = 1` or `a = 0` that failure is
    silent and enormous.
    """

    def __init__(self, switcher, system, *, amplitudes=BASIS_PROBE_AMPLITUDES):
        self.switcher = switcher
        self.system = system
        self.amplitudes = tuple(float(a) for a in amplitudes)
        if len(self.amplitudes) != POTENTIAL_ENERGY_EVALUATIONS_PER_UPDATE:
            raise DecompositionError(
                f"the basis needs exactly {POTENTIAL_ENERGY_EVALUATIONS_PER_UPDATE} probe "
                f"amplitudes; got {len(self.amplitudes)}")
        #: Every probe evaluation this object has performed. Reported in provenance.
        self.evaluations = 0
        #: Wall-clock seconds spent inside `measure`, including the parameter pushes and the
        #: restore. The overhead is stated as a measurement rather than as an estimate, which is
        #: the only form in which "this is acceptable" means anything.
        self.seconds = 0.0

    def measure(self, context, *, restore_tau: float) -> Components:
        """Three evaluations at frozen coordinates, then the exact quadratic through them."""
        import time

        from openmm import unit

        started = time.perf_counter()
        energies = []
        try:
            for amplitude in self.amplitudes:
                self.switcher.set_amplitude(context, self.system, amplitude)
                energies.append(context.getState(getEnergy=True).getPotentialEnergy(
                    ).value_in_unit(unit.kilojoule_per_mole))
                self.evaluations += 1
        finally:
            # Unconditional: an exception mid-probe must not leave the path at a probe amplitude.
            self.switcher.set_tau(context, self.system, restore_tau)
            self.seconds += time.perf_counter() - started
        return Components.from_probe(self.amplitudes, energies)
