"""Thermodynamic cycles: which legs a free energy needs, and the sign each one enters with.

A LEG is the free energy of one path, `dG = G(endpoint B) - G(endpoint A)`, in one environment,
plus a statement of what A and B are. A CYCLE names the legs it requires and combines them with
explicit signs; a cycle with a leg missing, in the wrong environment or the wrong orientation is
refused by name rather than summed. Legs are assumed statistically independent (separate
simulations), so uncertainties add in quadrature.

    absolute hydration      dG_hyd = dG(vacuum: coupled -> decoupled) - dG(solvent: coupled -> decoupled)
    relative hydration      ddG_hyd(A->B) = dG(solvent: A -> B) - dG(vacuum: A -> B)
    relative binding        ddG_bind(A->B) = dG(complex: A -> B) - dG(solvent: A -> B)
    absolute binding        dG_bind = dG(solvent: coupled -> decoupled)
                                      - dG(complex: unrestrained -> restrained, coupled)
                                      - dG(complex: coupled -> decoupled, restrained)
                                      - dG_release(restrained decoupled -> free at 1 M)

The absolute binding free energy is a STANDARD binding free energy: `dG_release` carries the
standard-state volume, and without it the number is a different quantity with the same name.
The cycle therefore refuses to assemble without it, and refuses a release term computed for a
different restraint than the one the restraint-attachment and complex legs were run with.

`coupled -> decoupled` is a decoupling of the ligand from its environment. Whether intramolecular
interactions are kept (decoupling) or removed (annihilation) is the Hamiltonian's decision; the
cycle is the same either way provided the vacuum or solvent leg uses the same choice as the
complex leg, which is checked through the recorded `alchemical_scheme`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from md_tools.alchemy.samples import KJ_PER_KCAL

CYCLE_SCHEMA = "md-tools-alchemical-cycle/1"

COUPLED = "coupled"
DECOUPLED = "decoupled"
UNRESTRAINED = "unrestrained"
RESTRAINED = "restrained"
ENVIRONMENTS = ("vacuum", "solvent", "complex")


class CycleError(ValueError):
    pass


@dataclass(frozen=True)
class Leg:
    """dG = G(endpoint_b) - G(endpoint_a) for one path in one environment, kJ/mol."""

    name: str
    environment: str
    endpoint_a: str
    endpoint_b: str
    delta_g_kj_mol: float
    sigma_kj_mol: float
    temperature_k: float
    estimator: str
    alchemical_scheme: str = ""
    restraint_digest: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    #: S2's digest of the LIGAND-side Hamiltonian of the plan this leg ran (packages, map, mode,
    #: dummy terms, junctions, exclusions, constraint policy, applied 1-4 scales). Two legs of one
    #: relative cycle must carry the same one, or the cycle would include the difference between
    #: two ligand Hamiltonians -- e.g. a solvent model's 1-4 scale -- as if it were hydration.
    ligand_hamiltonian_sha256: str | None = None

    def __post_init__(self):
        if self.environment not in ENVIRONMENTS:
            raise CycleError(f"leg {self.name}: environment {self.environment!r}; one of "
                             f"{ENVIRONMENTS}")
        if not (math.isfinite(self.delta_g_kj_mol) and math.isfinite(self.sigma_kj_mol)
                and self.sigma_kj_mol >= 0):
            raise CycleError(f"leg {self.name}: {self.delta_g_kj_mol} +- {self.sigma_kj_mol}")

    @classmethod
    def from_estimate(cls, name: str, environment: str, estimate: Mapping[str, Any], *,
                      path: Mapping[str, Any], temperature_k: float, alchemical_scheme: str = "",
                      restraint_digest: str | None = None,
                      ligand_hamiltonian_sha256: str | None = None) -> "Leg":
        """From one `estimators.analyze(...)['estimates'][X]` record and its path record."""
        return cls(name, environment, path["endpoint_a"], path["endpoint_b"],
                   float(estimate["delta_g_kJ_mol"]), float(estimate["sigma_kJ_mol"]),
                   temperature_k, estimate["estimator"], alchemical_scheme, restraint_digest,
                   {"path": dict(path), "estimate": dict(estimate)}, ligand_hamiltonian_sha256)

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "environment": self.environment,
                "endpoint_a": self.endpoint_a, "endpoint_b": self.endpoint_b,
                "delta_g_kJ_mol": self.delta_g_kj_mol, "sigma_kJ_mol": self.sigma_kj_mol,
                "temperature_k": self.temperature_k, "estimator": self.estimator,
                "alchemical_scheme": self.alchemical_scheme,
                "restraint_digest": self.restraint_digest,
                "ligand_hamiltonian_sha256": self.ligand_hamiltonian_sha256}


def _result(kind: str, terms: Sequence[tuple[float, Leg | Mapping[str, Any]]],
            temperature_k: float, extra: Mapping[str, Any]) -> dict[str, Any]:
    value = sum(sign * (t.delta_g_kj_mol if isinstance(t, Leg) else t["delta_g_kJ_mol"])
                for sign, t in terms)
    var = sum((t.sigma_kj_mol if isinstance(t, Leg) else t.get("sigma_kJ_mol", 0.0)) ** 2
              for _, t in terms)
    return {"schema": CYCLE_SCHEMA, "cycle": kind, "temperature_k": temperature_k,
            "delta_g_kJ_mol": value, "sigma_kJ_mol": math.sqrt(var),
            "delta_g_kcal_mol": value / KJ_PER_KCAL, "sigma_kcal_mol": math.sqrt(var) / KJ_PER_KCAL,
            "terms": [{"sign": sign, **(t.to_record() if isinstance(t, Leg) else dict(t))}
                      for sign, t in terms],
            "variance_rule": "legs independent; variances add", **extra}


def _require(leg: Leg | None, role: str, *, environment: str, a: str | None = None,
             b: str | None = None) -> Leg:
    if leg is None:
        raise CycleError(f"the {role} leg is missing")
    if leg.environment != environment:
        raise CycleError(f"the {role} leg ({leg.name}) runs in {leg.environment}; this cycle "
                         f"needs it in {environment}")
    if a is not None and (leg.endpoint_a, leg.endpoint_b) != (a, b):
        raise CycleError(
            f"the {role} leg ({leg.name}) runs {leg.endpoint_a} -> {leg.endpoint_b}; this cycle "
            f"needs {a} -> {b}. Reverse it explicitly with `reversed_leg` rather than letting the "
            f"sign be guessed")
    return leg


def _same_temperature(legs: Sequence[Leg]) -> float:
    temps = {round(leg.temperature_k, 9) for leg in legs}
    if len(temps) != 1:
        raise CycleError(f"legs at different temperatures {sorted(temps)}: one cycle, one T")
    return legs[0].temperature_k


def _same_scheme(legs: Sequence[Leg]) -> None:
    schemes = {leg.alchemical_scheme for leg in legs}
    if len(schemes) != 1:
        raise CycleError(f"legs use different alchemical schemes {sorted(schemes)}; the "
                         f"decoupled endpoints then differ and the cycle does not close")


def _same_ligand_hamiltonian(legs: Sequence[Leg]) -> None:
    digests = [leg.ligand_hamiltonian_sha256 for leg in legs]
    if any(d is None for d in digests):
        raise CycleError(
            f"legs {[leg.name for leg in legs]} carry no ligand_hamiltonian_sha256 (S2's plan "
            f"record): nothing shows the two legs share one ligand Hamiltonian, and a relative "
            f"cycle over two different ones reports their difference as a free energy")
    if len(set(digests)) != 1:
        raise CycleError(
            f"the legs' ligand Hamiltonians differ ({', '.join(d[:12] for d in digests)}): the "
            f"packages, map, dummy terms, constraints or applied 1-4 scales are not the same in "
            f"both environments. Run md_tools.alchemy.topology.matched_legs for the reason")


def reversed_leg(leg: Leg) -> Leg:
    """The same path traversed B -> A: the free energy changes sign, the uncertainty does not."""
    return Leg(leg.name + " (reversed)", leg.environment, leg.endpoint_b, leg.endpoint_a,
               -leg.delta_g_kj_mol, leg.sigma_kj_mol, leg.temperature_k, leg.estimator,
               leg.alchemical_scheme, leg.restraint_digest, {"reversed_from": leg.to_record()},
               leg.ligand_hamiltonian_sha256)


# ---------------------------------------------------------------------- the four cycles
def absolute_hydration(*, vacuum: Leg | None, solvent: Leg | None) -> dict[str, Any]:
    v = _require(vacuum, "vacuum decoupling", environment="vacuum", a=COUPLED, b=DECOUPLED)
    s = _require(solvent, "solvent decoupling", environment="solvent", a=COUPLED, b=DECOUPLED)
    _same_scheme([v, s])
    return _result("absolute_hydration", [(+1, v), (-1, s)], _same_temperature([v, s]),
                   {"quantity": "dG_hyd = G(solvated) - G(gas)"})


def relative_hydration(*, vacuum: Leg | None, solvent: Leg | None) -> dict[str, Any]:
    v = _require(vacuum, "vacuum A -> B", environment="vacuum")
    s = _require(solvent, "solvent A -> B", environment="solvent",
                 a=v.endpoint_a, b=v.endpoint_b)
    _same_scheme([v, s])
    _same_ligand_hamiltonian([v, s])
    return _result("relative_hydration", [(+1, s), (-1, v)], _same_temperature([v, s]),
                   {"quantity": f"ddG_hyd = dG_hyd({v.endpoint_b}) - dG_hyd({v.endpoint_a})"})


def relative_binding(*, solvent: Leg | None, complex: Leg | None) -> dict[str, Any]:
    s = _require(solvent, "solvent A -> B", environment="solvent")
    c = _require(complex, "complex A -> B", environment="complex",
                 a=s.endpoint_a, b=s.endpoint_b)
    _same_scheme([s, c])
    _same_ligand_hamiltonian([s, c])
    return _result("relative_binding", [(+1, c), (-1, s)], _same_temperature([s, c]),
                   {"quantity": f"ddG_bind = dG_bind({s.endpoint_b}) - dG_bind({s.endpoint_a})"})


def absolute_binding(*, solvent: Leg | None, restraint_attach: Leg | None,
                     complex: Leg | None, release: Mapping[str, Any] | None) -> dict[str, Any]:
    """The standard binding free energy with Boresch restraints.

    `release` is `BoreschRestraint.release_free_energy(T)` for the SAME restraint the
    `restraint_attach` and `complex` legs were run with (checked by digest).
    """
    s = _require(solvent, "solvent decoupling", environment="solvent", a=COUPLED, b=DECOUPLED)
    r = _require(restraint_attach, "restraint attachment", environment="complex",
                 a=UNRESTRAINED, b=RESTRAINED)
    c = _require(complex, "complex decoupling (restrained)", environment="complex",
                 a=COUPLED, b=DECOUPLED)
    if release is None:
        raise CycleError(
            "no restraint-release / standard-state term: without it the result is not a "
            "standard binding free energy, whatever it is called")
    t = _same_temperature([s, r, c])
    if abs(float(release["temperature_k"]) - t) > 1e-9:
        raise CycleError(f"release term computed at {release['temperature_k']} K; the legs ran "
                         f"at {t} K")
    digests = {r.restraint_digest, c.restraint_digest, release.get("restraint_digest")}
    if None in digests or len(digests) != 1:
        raise CycleError(
            f"restraint digests disagree or are missing (attach {r.restraint_digest}, complex "
            f"{c.restraint_digest}, release {release.get('restraint_digest')}): the release term "
            f"must be for the restraint the complex legs actually carried")
    _same_scheme([s, c])
    rel = {"name": "restraint release to standard state", "environment": "analytic",
           "delta_g_kJ_mol": float(release["delta_g_release_kJ_mol"]), "sigma_kJ_mol": 0.0,
           "standard_volume_nm3": release["standard_volume_nm3"],
           "standard_concentration": release["standard_concentration"],
           "restraint_digest": release["restraint_digest"], "method": release["method"]}
    return _result("absolute_binding", [(+1, s), (-1, r), (-1, c), (-1, rel)], t,
                   {"quantity": "standard binding free energy dG_bind at "
                                f"{release['standard_concentration']}",
                    "standard_state_included": True})
