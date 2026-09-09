"""The REST2 potential split into its three lambda-basis groups, for Hummer-Szabo reweighting.

WHY A DECOMPOSITION AT ALL

A switching path records one number per update: the total work. Reweighting a nonequilibrium
ensemble back onto a *different* Hamiltonian -- a different tau schedule, a different endpoint, or
a Hummer-Szabo estimator evaluated at a tau the path never visited -- needs the potential as a
FUNCTION of tau, not its value at the one tau that was run. A single total is not enough
information to produce that function, and re-running the path at another tau is not reweighting.

THE EXACT BASIS

Write the coupling amplitude and the REST2 scaling parameter as

    a      = 1 - tau
    lambda = a^2

Every scale factor in this convention (`md_tools.rest2.scaler`) is a power of `a`, and the group
names below say which power of LAMBDA that is -- because "linear" and "quadratic" are ambiguous
until you say linear in WHAT:

    non_scaled     lambda^0 = a^0     bonds, angles, excluded omega torsions,
                                      environment-environment nonbonded
    sqrt_scaled    sqrt(lambda) = a   solute-environment nonbonded, the whole generalised-Born
                                      energy
    lin_scaled     lambda = a^2       solute-solute nonbonded and 1-4, eligible solute torsions,
                                      solute CMAP

so at FROZEN COORDINATES the potential is exactly a quadratic polynomial in `a`:

    U(tau, x) = U_non_scaled(x)
              + sqrt(lambda) * U_sqrt_scaled(x)
              + lambda       * U_lin_scaled(x)

              = U_non_scaled(x) + a * U_sqrt_scaled(x) + a^2 * U_lin_scaled(x)

This is an identity, not an approximation, and it holds for the whole `NonbondedForce` including
the PME reciprocal sum, the Ewald self-energy and the long-range dispersion correction. Each of
those is a quadratic form in the per-particle charge (or in sqrt(epsilon)), and the scaler
multiplies solute charges by `a` and solute epsilons by `a^2`; a pair term therefore carries
`a^(number of solute partners)`, which is exactly the three-way split above. Nothing needs to be
decomposed per-pair, and the validated force construction is not touched.

TWO PROBES, AT TWO DIFFERENT COORDINATES

This is the distinction the first implementation got wrong, and it is the whole reason this module
has the shape it has.

A switching update does two things in sequence: it moves the PARAMETERS at frozen coordinates
`x_j`, and then it propagates the CONFIGURATION to `x_{j+1}`. Those are different coordinates, and
they answer different questions:

    WORK-BASIS PROBE          at the frozen pre-switch `x_j`, for the switch tau_j -> tau_{j+1}.
                              Decomposes the incremental and accumulated WORK. That is the right
                              coordinate for work precisely because the convention is
                              `dW_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)`, at frozen `x_j`.

    OBSERVATION-POTENTIAL     at the coordinate actually SAVED on the row, `x_t`, under the tau
    PROBE                     reported on that row. Supplies the energy basis a Hummer-Szabo
                              reweighting uses together with `W_t`.

The first implementation wrote a trajectory frame for `x_{j+1}` and put the work-basis values --
measured at `x_j`, one propagation earlier -- on that same row, under names that read like
potentials at that frame. Anyone reweighting from that file would have been pairing the work of
one configuration with the energy of another. Nothing raises; every number is plausible.

So: a row that names a `coordinate_frame_index` carries observation potentials recomputed AT THAT
FRAME. A row with no saved coordinate -- when the trajectory cadence is not the observation
cadence -- leaves those fields EMPTY rather than borrowing a neighbour's, and is excluded from the
frame-aligned HS table entirely.

HOW EACH PROBE IS MEASURED

Three potential-energy evaluations at three controlled amplitudes, at frozen coordinates, then one
quadratic fit. Three points determine a quadratic exactly, so the fit introduces no model error:
the only error is the floating-point error of the three evaluations themselves.

The amplitudes are `0`, `1/2` and `1`, as far apart as the domain allows -- the fit divides by
their differences, so widely separated nodes are the difference between a stable measurement and
one dominated by cancellation. `a = 0` is worth having on its own: it switches every scaled term
off, so it reads `U_non_scaled` directly rather than inferring it.

WHAT IS CHECKED RATHER THAN ASSUMED

The total work stays what it always was: `U(tau_{j+1}, x_j) - U(tau_j, x_j)`, measured directly
from two evaluations of the real Hamiltonian. The component works are derived independently, from
the work-basis fit. The two are then required to agree.

The observation potentials are checked the same way and separately: the reconstruction
`U_non + a U_sqrt + a^2 U_lin` is compared against a DIRECT `getState(getEnergy=True)` at the same
coordinate and tau. Both numbers are written -- `potential_reconstructed_kj_mol` and
`potential_direct_kj_mol` -- so a reader can check the identity in the file rather than trust that
somebody checked it once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["BASIS_PROBE_AMPLITUDES", "POTENTIAL_ENERGY_EVALUATIONS_PER_PROBE",
           "DECOMPOSITION_SCHEMA", "WORK_COMPONENT_COLUMNS", "OBSERVATION_POTENTIAL_COLUMNS",
           "COMPONENT_SUMMARY_COLUMNS", "HS_COLUMNS", "GROUPS",
           "Components", "ComponentWork", "ComponentProbe", "EvaluationCounters",
           "quadratic_through", "reconstruction_tolerance", "require_compatible_schema",
           "RECONSTRUCTION_TOLERANCES", "DecompositionError", "WORK_MEASUREMENT_MODES",
           "work_component_columns", "observation_potential_columns", "hs_columns",
           "component_summary_columns", "require_mode", "recorded_mode",
           "DIRECT_POTENTIAL_COLUMN"]


#: The three amplitudes probed, in the order they are applied. As far apart as the domain allows:
#: the fit divides by their pairwise differences.
BASIS_PROBE_AMPLITUDES = (0.0, 0.5, 1.0)

#: Energy evaluations one basis probe costs. Named for what it is: a probe is three evaluations,
#: and a run performs one probe per switching update and another per frame-aligned observation.
POTENTIAL_ENERGY_EVALUATIONS_PER_PROBE = 3

#: The three groups, by the power of LAMBDA they carry. One vocabulary for the work columns and
#: the observation-potential columns alike, so a reader learns it once.
GROUPS = ("non_scaled", "sqrt_scaled", "lin_scaled")

#: The schema of the component columns. Written into every record that carries them, and checked
#: on resume.
#:
#: VERSION 2, and version 1 is refused rather than continued. v1 wrote the pre-switch work basis
#: under names that read as potentials at the row's saved frame. That is not a renaming: the
#: numbers describe a different coordinate, so v1 rows and v2 rows cannot be mixed in one table
#: and a v1 checkpoint cannot be resumed into a v2 run.
DECOMPOSITION_SCHEMA = {
    "name": "rest2-lambda-basis",
    "version": 2,
    "state_coordinate": "tau",
    "amplitude": "a = 1 - tau",
    "lambda": "lambda = a^2",
    "identity": ("U(tau, x) = U_non_scaled(x) + sqrt(lambda)*U_sqrt_scaled(x) "
                 "+ lambda*U_lin_scaled(x)"),
    "work_identity": "dW_total = dW_non_scaled + dW_sqrt_scaled + dW_lin_scaled",
    "non_scaled_terms": ["bonds", "angles", "excluded omega torsions",
                         "environment-environment nonbonded"],
    "sqrt_scaled_terms": ["solute-environment nonbonded", "generalized Born"],
    "lin_scaled_terms": ["solute-solute nonbonded", "solute-solute 1-4",
                         "eligible solute torsions", "solute CMAP"],
    "probe_amplitudes": list(BASIS_PROBE_AMPLITUDES),
    "potential_energy_evaluations_per_probe": POTENTIAL_ENERGY_EVALUATIONS_PER_PROBE,
    "work_basis_coordinate": "x_j, frozen, before the parameter switch",
    "observation_potential_coordinate": "x_t, the coordinate saved on this row",
    "delta_work_meaning": ("the sum of every switch since the previous emitted work observation; "
                           "equal to one switch when the work and switching cadences agree"),
    "units": "kJ/mol",
}

#: Superseded schemas, refused by name with the reason. A record written under one of these is
#: left exactly as it is; nothing here rewrites history.
HISTORICAL_SCHEMAS = {
    ("rest2-tau-quadratic-basis", 1): (
        "v1 wrote the pre-switch WORK basis -- measured at x_j -- into columns named as if they "
        "were potentials at the row's saved coordinate x_t, one propagation later. Reweighting "
        "from those rows pairs the work of one configuration with the energy of another. The "
        "numbers are not convertible, so a v1 path cannot be continued into a v2 run."),
}

#: The incremental and cumulative WORK components, measured at the frozen pre-switch coordinate
#: where the work convention defines them.
WORK_COMPONENT_COLUMNS = tuple(
    [f"delta_work_{group}_kj_mol" for group in GROUPS]
    + [f"total_work_{group}_kj_mol" for group in GROUPS])

#: The OBSERVATION potentials, at the coordinate this row names. Empty when the row saved no
#: coordinate -- see the module docstring; they are never borrowed from a neighbouring frame.
OBSERVATION_POTENTIAL_COLUMNS = (
    "potential_non_scaled_kj_mol",
    "potential_sqrt_scaled_kj_mol",
    "potential_lin_scaled_kj_mol",
    "potential_reconstructed_kj_mol",
    "potential_direct_kj_mol",
)

#: Final per-path totals, on `AIS_paths.csv`.
COMPONENT_SUMMARY_COLUMNS = tuple(
    [f"total_work_{group}_kj_mol" for group in GROUPS] + ["decomposition_schema_version"])

#: The frame-aligned table a Hummer-Szabo reweighting reads. ONLY rows whose coordinate was
#: actually saved appear in it, so every row's potentials and work belong to the same
#: configuration and no reader has to filter first -- or forget to.
HS_COLUMNS = (
    "path_id", "source_frame", "observation_index", "switch_step",
    "coordinate_frame_index", "tau", "total_work_kj_mol",
    "potential_non_scaled_kj_mol", "potential_sqrt_scaled_kj_mol",
    "potential_lin_scaled_kj_mol", "potential_reconstructed_kj_mol",
    "potential_direct_kj_mol",
)


#: The two ways a run may measure the work at a parameter update. Mutually exclusive: `work` is
#: the direct difference and nothing else, `components` is the basis probe from which the work
#: follows. See `build.md`'s `ais.work_measurement` for which to choose.
WORK_MEASUREMENT_MODES = ("work", "components")

#: The one observation potential a `work`-mode run still has. It costs the evaluation the run
#: already makes, it belongs to the coordinate the row names, and it is U at the tau that was
#: VISITED -- which is the part of the reweighting story a direct measurement can honestly tell.
#: The basis columns beside it are the part it cannot, and they are absent rather than nought.
DIRECT_POTENTIAL_COLUMN = "potential_direct_kj_mol"


def require_mode(mode: Any, *, what: str = "work_measurement") -> str:
    """The mode, or a refusal naming what was given. Never a silent fallback to a default.

    A misspelled mode that quietly became `work` would drop the component columns from a run
    whose whole purpose was to produce them, and the run would look successful.
    """
    if mode not in WORK_MEASUREMENT_MODES:
        raise DecompositionError(
            f"{what} is {mode!r}; it must be one of {', '.join(WORK_MEASUREMENT_MODES)}.")
    return str(mode)


def recorded_mode(record: Any, *, what: str) -> str:
    """The mode a written record was produced under.

    An ABSENT `work_measurement` means `components`, and that is not a guess. Every build that
    wrote a record without this key took the basis probe unconditionally -- it was the only mode
    -- so the key's absence has one meaning and the columns beside it confirm it. This is the
    opposite of an absent `decomposition_schema`, which means the components were never measured
    at all; there, absence is refused. The difference is whether the old behaviour was knowable.
    """
    if isinstance(record, dict) and record.get("work_measurement") is None:
        return "components"
    return require_mode(record.get("work_measurement") if isinstance(record, dict) else record,
                        what=what)


def work_component_columns(mode: str) -> tuple[str, ...]:
    """The per-update work-component columns, which only `components` mode measures."""
    return WORK_COMPONENT_COLUMNS if require_mode(mode) == "components" else ()


def observation_potential_columns(mode: str) -> tuple[str, ...]:
    """The observation potentials this mode measured: all five, or the direct one alone."""
    return (OBSERVATION_POTENTIAL_COLUMNS if require_mode(mode) == "components"
            else (DIRECT_POTENTIAL_COLUMN,))


def component_summary_columns(mode: str) -> tuple[str, ...]:
    """The per-path component totals on `AIS_paths.csv`, absent outside `components` mode."""
    return COMPONENT_SUMMARY_COLUMNS if require_mode(mode) == "components" else ()


def hs_columns(mode: str) -> tuple[str, ...]:
    """The frame-aligned table's columns for this mode.

    `work` mode still gets the table: every row is a saved coordinate with the work that reached
    it and the potential AT it, which is what a Hummer-Szabo estimate at the schedule that ran
    needs. What it does not get is the basis, so an estimate at an UNVISITED tau cannot be formed
    from it -- and the reader discovers that from four missing columns rather than from four
    columns of zeros.
    """
    if require_mode(mode) == "components":
        return HS_COLUMNS
    return tuple(name for name in HS_COLUMNS
                 if name not in OBSERVATION_POTENTIAL_COLUMNS or name == DIRECT_POTENTIAL_COLUMN)


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
            f"lambda-basis columns existed. Its cumulative component work is unknown and cannot "
            f"be continued -- resuming would add components measured now onto a total measured "
            f"under no decomposition at all.\n"
            f"  Delete the path directory to rerun it from its source frame, or read the old run "
            f"as it stands; this build writes "
            f"{DECOMPOSITION_SCHEMA['name']}/v{DECOMPOSITION_SCHEMA['version']}.")
    schema = recorded["decomposition_schema"]
    name = schema.get("name") if isinstance(schema, dict) else None
    version = schema.get("version") if isinstance(schema, dict) else None
    if (name, version) == (DECOMPOSITION_SCHEMA["name"], DECOMPOSITION_SCHEMA["version"]):
        return
    reason = HISTORICAL_SCHEMAS.get((name, version))
    raise DecompositionError(
        f"{what} records the decomposition schema {name}/v{version}, but this build implements "
        f"{DECOMPOSITION_SCHEMA['name']}/v{DECOMPOSITION_SCHEMA['version']}.\n"
        f"  {reason or 'That basis is not one this build implements.'}\n"
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
    nodes_tuple = (a0, a1, a2)
    others = ((a1, a2), (a0, a2), (a0, a1))
    c0 = c1 = c2 = 0.0
    for position in range(3):
        own = nodes_tuple[position]
        other, third = others[position]
        value = float(values[position])
        denominator = (own - other) * (own - third)
        # (a - other)(a - third) / denominator, expanded: the Lagrange basis polynomial for this
        # node, weighted by its value.
        c2 += value / denominator
        c1 += -value * (other + third) / denominator
        c0 += value * other * third / denominator
    return c0, c1, c2


@dataclass(frozen=True)
class Components:
    """The three basis values at ONE configuration, in kJ/mol.

    Coordinate-dependent and tau-independent: this IS the potential as a function of tau at that
    configuration, which is the whole point of measuring it. WHICH configuration is the caller's
    business, and it is exactly what the two probes differ in.
    """

    non_scaled: float
    sqrt_scaled: float
    lin_scaled: float

    @staticmethod
    def from_probe(amplitudes, energies) -> "Components":
        c0, c1, c2 = quadratic_through(amplitudes, energies)
        return Components(non_scaled=c0, sqrt_scaled=c1, lin_scaled=c2)

    def contributions_at(self, tau: float) -> tuple[float, float]:
        """`sqrt(lambda)*U_sqrt_scaled` and `lambda*U_lin_scaled`, with `lambda = (1-tau)^2`."""
        amplitude = 1.0 - float(tau)
        return amplitude * self.sqrt_scaled, amplitude * amplitude * self.lin_scaled

    def total_at(self, tau: float) -> float:
        """`U(tau)` reconstructed from the components. Compared against a direct measurement."""
        sqrt_part, lin_part = self.contributions_at(tau)
        return self.non_scaled + sqrt_part + lin_part

    def work_between(self, tau_before: float, tau_after: float) -> "ComponentWork":
        """The three incremental works for a parameter change at these frozen coordinates.

        `non_scaled` is exactly `0.0`, and it is written as the literal it is. That group carries
        `lambda^0`, the coordinates did not move, so no arithmetic here could produce anything
        else -- and computing it as `self.non_scaled - self.non_scaled` to make the column look
        measured would be dressing up a tautology as evidence.

        What has content is the SUM identity, which the caller checks against a total work
        measured independently from two evaluations of the real Hamiltonian.
        """
        before_sqrt, before_lin = self.contributions_at(tau_before)
        after_sqrt, after_lin = self.contributions_at(tau_after)
        return ComponentWork(non_scaled=0.0,
                             sqrt_scaled=after_sqrt - before_sqrt,
                             lin_scaled=after_lin - before_lin)

    def row(self, tau: float, direct: float) -> dict[str, float]:
        """The five observation-potential columns for a row at this tau and this coordinate."""
        return {
            "potential_non_scaled_kj_mol": self.non_scaled,
            "potential_sqrt_scaled_kj_mol": self.sqrt_scaled,
            "potential_lin_scaled_kj_mol": self.lin_scaled,
            "potential_reconstructed_kj_mol": self.total_at(tau),
            "potential_direct_kj_mol": direct,
        }

    def record(self) -> dict[str, float]:
        return {f"{group}_kj_mol": getattr(self, group) for group in GROUPS}


@dataclass(frozen=True)
class ComponentWork:
    """Three works that must sum to the independently measured total."""

    non_scaled: float
    sqrt_scaled: float
    lin_scaled: float

    @property
    def total(self) -> float:
        return self.non_scaled + self.sqrt_scaled + self.lin_scaled

    def __add__(self, other: "ComponentWork") -> "ComponentWork":
        return ComponentWork(self.non_scaled + other.non_scaled,
                             self.sqrt_scaled + other.sqrt_scaled,
                             self.lin_scaled + other.lin_scaled)

    @staticmethod
    def zero() -> "ComponentWork":
        return ComponentWork(0.0, 0.0, 0.0)

    def row(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{group}_kj_mol": getattr(self, group) for group in GROUPS}

    def mapping(self) -> dict[str, float]:
        return {group: getattr(self, group) for group in GROUPS}

    @staticmethod
    def from_mapping(mapping) -> "ComponentWork":
        return ComponentWork(*(float(mapping[group]) for group in GROUPS))


#: Documented numerical tolerance for both identities, as `(relative, absolute floor)` in kJ/mol,
#: by OpenMM precision mode. The reconstruction is a fixed linear combination of three energies
#: measured on one Context, so its error is that of the energies themselves, amplified by the fit
#: coefficients (whose magnitudes are 1, 3 and 4 for these nodes).
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


@dataclass
class EvaluationCounters:
    """What a path's energy evaluations actually cost, split so no number double-counts another.

    THE ACCOUNTING BUG THIS REPLACES. The previous version added the counters restored from a
    committed checkpoint to `discarded_energy_evaluations`. Those evaluations are not discarded:
    they produced the committed work, observations, frames and state rows that the resume is
    continuing from. Calling them discarded made a resumed path report a cost profile in which
    its own useful work had vanished into waste -- and made the useful total of a resumed path
    differ from that of an identical uninterrupted one, which is exactly the comparison the
    counters exist to support.

    THE RELATIONSHIPS, and they are checked:

        useful_total = direct_work + work_basis_probe + observation_potential + other_useful
        paid_total   = useful_total + known_discarded

    WHAT CANNOT BE KNOWN. A committed checkpoint records what had been done WHEN IT COMMITTED. It
    cannot know how much further work a process performed before it died, because recording that
    would itself require a durable write after every evaluation. So the evaluations between the
    last commit and the crash are real, were paid for, and are UNOBSERVABLE.

    `known_discarded` counts only what durable evidence proves, and `discarded_is_complete` says
    whether that evidence is complete. Reporting an unknown as zero would understate the cost;
    reporting committed useful work as discarded -- the previous behaviour -- overstates it and
    corrupts the useful total as well. Saying "unknown" is the only honest option, and it is
    written down rather than left to be inferred from a suspiciously round number.
    """

    #: The two direct evaluations each switch performs: U(tau_j, x_j) and U(tau_{j+1}, x_j).
    direct_work_energy_evaluations: int = 0
    #: `getState(getEnergy=True)` inside the WORK-basis probe, at the frozen pre-switch coordinate.
    work_basis_probe_energy_evaluations: int = 0
    #: The direct evaluation at each frame-aligned observation, which the reconstruction is
    #: checked against, plus that row's own basis probe.
    observation_potential_energy_evaluations: int = 0
    #: Anything else with a stated scientific reason. Zero today; present so a future evaluation
    #: has somewhere honest to go instead of being folded into one of the three above.
    other_useful_energy_evaluations: int = 0
    #: Evaluations that durable evidence proves were performed and then thrown away.
    known_discarded_energy_evaluations: int = 0
    #: False once a resume has happened without attempt-level evidence: some cost is unobservable.
    discarded_is_complete: bool = True
    #: `updateParametersInContext` / `setParameter` pushes: the expensive half on a solvated
    #: system, and not an energy evaluation, so counted apart from all of the above.
    parameter_updates: int = 0
    #: Wall-clock seconds inside the probes.
    probe_seconds: float = 0.0

    @property
    def useful_total(self) -> int:
        return (self.direct_work_energy_evaluations
                + self.work_basis_probe_energy_evaluations
                + self.observation_potential_energy_evaluations
                + self.other_useful_energy_evaluations)

    @property
    def paid_total(self) -> int:
        return self.useful_total + self.known_discarded_energy_evaluations

    def record(self) -> dict[str, Any]:
        return {
            "direct_work_energy_evaluations": self.direct_work_energy_evaluations,
            "work_basis_probe_energy_evaluations": self.work_basis_probe_energy_evaluations,
            "observation_potential_energy_evaluations":
                self.observation_potential_energy_evaluations,
            "other_useful_energy_evaluations": self.other_useful_energy_evaluations,
            "useful_total_energy_evaluations": self.useful_total,
            "known_discarded_energy_evaluations": self.known_discarded_energy_evaluations,
            "paid_total_energy_evaluations": self.paid_total,
            "discarded_is_complete": bool(self.discarded_is_complete),
            "parameter_updates": self.parameter_updates,
            "probe_seconds": round(self.probe_seconds, 6),
            "relationships": [
                "useful_total = direct_work + work_basis_probe + observation_potential "
                "+ other_useful",
                "paid_total = useful_total + known_discarded",
            ],
            "counter_definitions": {
                "energy_evaluation": "one Context.getState(getEnergy=True) call",
                "parameter_update": ("one Force.updateParametersInContext or "
                                     "Context.setParameter push"),
                "known_discarded": ("evaluations that durable evidence proves were performed and "
                                    "thrown away"),
                "discarded_is_complete": ("false when a resume occurred: a committed checkpoint "
                                          "cannot know how much work happened after it before "
                                          "the crash, so that cost is real and unobservable"),
            },
        }

    @staticmethod
    def from_record(entry) -> "EvaluationCounters":
        """Restore committed counters. They come back as USEFUL, which is what they are."""
        entry = entry or {}
        return EvaluationCounters(
            direct_work_energy_evaluations=int(
                entry.get("direct_work_energy_evaluations", 0)),
            work_basis_probe_energy_evaluations=int(
                entry.get("work_basis_probe_energy_evaluations", 0)),
            observation_potential_energy_evaluations=int(
                entry.get("observation_potential_energy_evaluations", 0)),
            other_useful_energy_evaluations=int(
                entry.get("other_useful_energy_evaluations", 0)),
            known_discarded_energy_evaluations=int(
                entry.get("known_discarded_energy_evaluations", 0)),
            discarded_is_complete=bool(entry.get("discarded_is_complete", True)),
            parameter_updates=int(entry.get("parameter_updates", 0)),
            probe_seconds=float(entry.get("probe_seconds", 0.0)))

    def __add__(self, other: "EvaluationCounters") -> "EvaluationCounters":
        return EvaluationCounters(
            direct_work_energy_evaluations=(self.direct_work_energy_evaluations
                                            + other.direct_work_energy_evaluations),
            work_basis_probe_energy_evaluations=(self.work_basis_probe_energy_evaluations
                                                 + other.work_basis_probe_energy_evaluations),
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
            probe_seconds=self.probe_seconds + other.probe_seconds)


class ComponentProbe:
    """Measures the three basis components on a live Context, at whatever coordinate it holds.

    The probe drives the SAME switcher the path switches with, so a probe amplitude and a real tau
    reach the Context through one implementation. It always restores the amplitude it was told to
    restore, in a `finally`, because a probe that leaked its last amplitude would leave the path
    integrating under a Hamiltonian nobody chose -- and at `a = 1` or `a = 0` that failure is
    silent and enormous.

    It does not know or care WHICH coordinate it is measuring at. The caller decides that, and the
    caller is where the work/observation distinction lives.
    """

    def __init__(self, switcher, system, *, amplitudes=BASIS_PROBE_AMPLITUDES, counters=None):
        self.switcher = switcher
        self.system = system
        self.amplitudes = tuple(float(a) for a in amplitudes)
        if len(self.amplitudes) != POTENTIAL_ENERGY_EVALUATIONS_PER_PROBE:
            raise DecompositionError(
                f"the basis needs exactly {POTENTIAL_ENERGY_EVALUATIONS_PER_PROBE} probe "
                f"amplitudes; got {len(self.amplitudes)}")
        self.counters = counters if counters is not None else EvaluationCounters()

    def measure(self, context, *, restore_tau: float, observation: bool = False) -> Components:
        """Three evaluations at the CURRENT coordinates, then the exact quadratic through them.

        `observation` says which counter these three belong to. The work-basis probe and the
        observation-potential probe are the same arithmetic at different coordinates, and a cost
        report that merged them could not answer "what did the HS output cost?" -- which is the
        question the frame-aligned probe was added by, and therefore the one worth answering.
        """
        import time

        from openmm import unit

        started = time.perf_counter()
        energies = []
        try:
            for amplitude in self.amplitudes:
                self.switcher.set_amplitude(context, self.system, amplitude)
                self.counters.parameter_updates += 1
                energies.append(context.getState(getEnergy=True).getPotentialEnergy(
                    ).value_in_unit(unit.kilojoule_per_mole))
                if observation:
                    self.counters.observation_potential_energy_evaluations += 1
                else:
                    self.counters.work_basis_probe_energy_evaluations += 1
        finally:
            # Unconditional: an exception mid-probe must not leave the path at a probe amplitude.
            self.switcher.set_tau(context, self.system, restore_tau)
            self.counters.parameter_updates += 1
            self.counters.probe_seconds += time.perf_counter() - started
        return Components.from_probe(self.amplitudes, energies)
