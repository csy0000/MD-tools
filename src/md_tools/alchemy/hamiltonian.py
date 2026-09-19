"""The Amber18 softcore alchemical Hamiltonian, built from two end-state Systems.

INPUT

Two OpenMM Systems over ONE hybrid particle set, as the topology plan (S2,
`md-tools-topology-plan/1`) provides them: identical particle count, masses, constraints and force
layout; the particles are partitioned into common (C), A-only (disappearing) and B-only
(appearing). In System A the B-only particles are dummies -- charge 0, epsilon 0, every exception
touching them zero -- and System B mirrors that. A-only x B-only pairs are excluded in both.

THE HAMILTONIAN (Amber18 manual section 21.1.5, eqs. 21.3, 21.5-21.7, and the rules stated there)

    U(lam_e, lam_s, lam_b) =
        (1 - lam_e) El_A(lam_e) + lam_e El_B(lam_e)                      electrostatics
      + (1 - lam_s) LJ_A(lam_s) + lam_s LJ_B(lam_s)                      Lennard-Jones
      + (1 - lam_b) Bonded_A    + lam_b Bonded_B                         bonded, common core
      + U_unscaled                                                        softcore internal

* El_X is end state X's own Ewald (or vacuum Coulomb) energy over its own charges, except that
  (i) the direct-space term of a pair (X-only particle, common particle) is
  `q q erfc(kappa r) / sqrt(r^2 + beta l)` -- l = lam_e for A, 1 - lam_e for B -- which is eq. 21.7
  in the form pmemd evaluates under PME (see `softcore`), with the reciprocal, self and background
  terms untouched; and (ii) every pair internal to the X-only region, and every exception touching
  it, is removed from El_X exactly as an exclusion is (its reciprocal share subtracted).
* LJ_X is end state X's Lennard-Jones energy with (X-only, common) pairs replaced by eq. 21.5 (A)
  or 21.6 (B), softening argument lam_s or 1 - lam_s, and X-internal pairs and X-touching
  exceptions removed.
* U_unscaled is "the interactions among the disappearing atoms are not changed, and do not
  contribute to dV/dlambda" (manual 21.1.5): every non-excluded pair inside a region, as a plain
  vacuum Coulomb + LJ pair with no cutoff (pmemd `gti_cut = 1`), and every exception inside a
  region, at its full value from the end state where that region is physical.

EXCEPTIONS AND BONDED TERMS ON THE BOUNDARY -- where Amber18 and its successors differ

* A 1-4 exception between a region particle and a common one follows `sc_boundary_14`
  (`softcore.SoftcoreSettings`): "scaled" (default) mixes it between the end states like every
  other exception -- (1 - lam_e) qq_A/r + lam_e qq_B/r, (1 - lam_s) LJ_A + lam_s LJ_B, and the
  dummy end state's value is zero -- which is pmemd 20+'s `gti_add_sc = 1`; "unscaled" puts it in
  U_unscaled at full strength, the Amber18 manual's "any ... 1-4 term that involves at least one
  appearing or disappearing atom is not scaled by lambda" (`gti_add_sc = 0`). The Amber20+ manual
  calls the latter theoretically incorrect, and it couples a dummy to the physical coordinates, so
  it is not the default; it exists to reproduce an Amber18 run.
* A bonded term touching a region is whatever the topology plan's end states say. Identical in
  both -- a term the plan RETAINS at the dummy end -- it is unscaled, the Amber18 rule. A term the
  plan REMOVES at the dummy end (force constant 0 there, every other parameter equal) is mixed
  with lambda_bonded, as pmemd 20+'s `gti_bat_sc = 1` scales the junction terms it does not keep;
  that is what makes the plan's single-anchor dummy separable. Any other difference in a term
  touching a region is refused.

The Amber18 one-step transformation is the diagonal lam_e = lam_s = lam_b = lambda. The three
components exist so a path can stage them (decharge, then sterics); the functional form is the
same on and off the diagonal, and which path is run is a schedule's business, not this module's.

`sc = False` selects the same construction with no softcore region, which is ordinary linear
mixing and is only defined when no particle appears or disappears; otherwise it is refused.

WHAT IS NOT CARRIED OVER FROM AIS

`dU/dlambda = U1 - U0` holds for AIS because that path is linear. This one is not: the softening
argument moves with lambda. Derivatives here are computed per component -- see `derivatives` -- and
the tests check them against finite differences, never against that identity.

HOW IT IS BUILT, AND WHY THIS WAY

Amber mixes ENERGIES, not parameters: (1 - lam) q_A,i q_A,j + lam q_B,i q_B,j is not
q_i(lam) q_j(lam). Ewald energy is exactly quadratic in the charges, so end state A's electrostatics
are a NonbondedForce whose charges are q_A * sqrt(1 - lam_e), and B's one with q_B * sqrt(lam_e):
each is then exactly its end state's full Ewald energy, reciprocal sum and self term included, times
the mixing weight. The square roots are DERIVED Context parameters, which is why a caller sets a
state through `set_state` and never through `Context.setParameter` on a public name alone.

Everything whose Lennard-Jones depends on lambda is in CustomNonbondedForces, never in a
NonbondedForce. Measured on OpenMM 8.6 (Reference and CPU): a NonbondedForce's dispersion
correction is NOT recomputed when a parameter offset changes through a global parameter -- epsilon
scaled to 0.3 left the tail at its full-weight value -- while a CustomNonbondedForce's long-range
correction is. A NonbondedForce here therefore carries only LJ that is the same in both end states.

Forces, by force group (`FORCE_GROUPS`):

    0  static        every force identical in both end states, added once
    1  nonbonded_a   end state A electrostatics (weighted) + LJ common to both end states
    2  nonbonded_b   end state B electrostatics (weighted)
    3  lj_common_changing   common particles whose LJ differs: (1-lam_s) LJ_A + lam_s LJ_B
    4  softcore_a_elec, 5 softcore_a_lj, 6 softcore_b_elec, 7 softcore_b_lj
    8  softcore_internal    U_unscaled
    9  bonded_mixed  bonded forces that differ between end states, (1-lam_b) A + lam_b B
   10  dispersion    (1-lam_s) D_A(V) + lam_s D_B(V) - D_static(V), see below

THE DISPERSION CORRECTION is each end state's OWN NonbondedForce correction, mixed linearly in
lambda_sterics -- as pmemd weights each TI region's correction by its lambda weight and uses the
plain, unsoftened tail for softcore atoms. It cannot be assembled from per-subset corrections:
OpenMM's NonbondedForce correction counts pairs as N^2 / (N(N+1)/2) with self pairs, and a
CustomNonbondedForce with interaction groups returns exactly N/(N+1) times the distinct-pair tail
(both measured, OpenMM 8.6), so no split into static, changing and softcore subsets adds up to
either end state's number. D_A, D_B and the static part NonbondedForce A already carries are
therefore measured from OpenMM itself at construction, as coefficients of 1/V, and the remainder is
carried by one CustomNonbondedForce whose pair energy is identically zero (`step(r - cutoff)`
inside the cutoff) and whose long-range correction is the 1/V term, calibrated at construction.
Its cost is one pair. The result: U(0) and U(1) reproduce the end-state Systems' energies with
their dispersion corrections, at every box volume.

A softcore electrostatic force is a DELTA on the NonbondedForce's own direct-space term for the same
pair (see `softcore.pair_expressions`), because a NonbondedForce cannot skip a pair's direct space
without also removing its reciprocal share. The two cancel analytically; near an overlap they cancel
numerically, which costs precision in single-precision force evaluation and is tested as such.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from md_tools.alchemy.softcore import ONE_4PI_EPS0, SoftcoreSettings, pair_expressions
from md_tools.alchemy.topology import INTERNAL_FORCE_NAME

__all__ = [
    "HAMILTONIAN_SCHEMA", "PUBLIC_PARAMETERS", "FORCE_GROUPS", "AlchemicalHamiltonianError",
    "AlchemicalHamiltonian", "build_hamiltonian", "from_plan",
]

HAMILTONIAN_SCHEMA = "md-tools-alchemical-hamiltonian/1"

LAMBDA_ELECTROSTATICS = "lambda_electrostatics"
LAMBDA_STERICS = "lambda_sterics"
LAMBDA_BONDED = "lambda_bonded"
PUBLIC_PARAMETERS = (LAMBDA_ELECTROSTATICS, LAMBDA_STERICS, LAMBDA_BONDED)

# Derived Context parameters. Never set on their own; `context_parameters` is their one definition.
_QSCALE = {"A": "mdt_alchemy_qscale_a", "B": "mdt_alchemy_qscale_b"}   # sqrt of the elec weight
_W_EL = {"A": "mdt_alchemy_w_elec_a", "B": "mdt_alchemy_w_elec_b"}     # exception chargeProd weight
_W_LJ = {"A": "mdt_alchemy_w_lj_a", "B": "mdt_alchemy_w_lj_b"}         # exception LJ weight

FORCE_GROUPS: Mapping[str, int] = {
    "static": 0, "nonbonded_a": 1, "nonbonded_b": 2, "lj_common_changing": 3,
    "softcore_a_elec": 4, "softcore_a_lj": 5, "softcore_b_elec": 6, "softcore_b_lj": 7,
    "softcore_internal": 8, "bonded_mixed": 9, "dispersion": 10,
}

#: Sigma given to an epsilon-zero particle inside a custom softcore expression. The pair epsilon is
#: then zero whatever sigma is; the value only keeps (r/sigma)^6 finite so no 0 * inf appears in
#: the energy or its derivatives.
_SIGMA_FOR_ZERO_EPSILON = 0.1

#: The platform of the two CONSTRUCTION-time probes -- reading the PME parameters OpenMM chooses
#: (`_pme_parameters`) and calibrating the dispersion carrier (`_dispersion_force`). Pinned, not
#: the machine platform: they are deterministic single-point reads that fix numbers written into
#: the Hamiltonian's System, which must be the same whichever machine builds it, and nothing is
#: propagated. The Hamiltonian's RUN-time evaluations use whatever Context the caller made, on the
#: platform `md_tools.openmm.platform_policy` chose.
BUILD_PROBE_PLATFORM = "Reference"

_MIXABLE_BONDED = ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce")
_REFUSED_NONBONDED = ("CustomNonbondedForce", "GBSAOBCForce", "CustomGBForce", "AmoebaMultipoleForce",
                      "AmoebaVdwForce", "DrudeForce", "ATMForce")


class AlchemicalHamiltonianError(ValueError):
    """An end-state pair, or a setting, this Hamiltonian cannot represent. Always names the cause."""


def _mm():
    import openmm
    return openmm


def _type(force) -> str:
    return type(force).__name__


def _xml(obj) -> str:
    return _mm().XmlSerializer.serialize(obj)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


# --------------------------------------------------------------------------------------------------
# Reading the end states
# --------------------------------------------------------------------------------------------------

@dataclass
class _EndState:
    charge: list[float]
    sigma: list[float]
    epsilon: list[float]
    exceptions: dict[tuple[int, int], tuple[float, float, float]]


def _nonbonded_force(system, label: str):
    found = [f for f in system.getForces() if _type(f) == "NonbondedForce"]
    refused = sorted({_type(f) for f in system.getForces() if _type(f) in _REFUSED_NONBONDED})
    if refused:
        raise AlchemicalHamiltonianError(
            f"System {label} contains {refused}; the Amber18 softcore Hamiltonian is defined for "
            "a single NonbondedForce (vacuum or PME) and nothing else nonbonded")
    if len(found) != 1:
        raise AlchemicalHamiltonianError(
            f"System {label} has {len(found)} NonbondedForce objects; exactly one is required")
    nb = found[0]
    if nb.getNumGlobalParameters() or nb.getNumParticleParameterOffsets() \
            or nb.getNumExceptionParameterOffsets():
        raise AlchemicalHamiltonianError(
            f"System {label}'s NonbondedForce already has global parameters or parameter offsets; "
            "an end state must be a plain System, not an already-alchemical one")
    return nb


def _read_end_state(nb) -> _EndState:
    n = nb.getNumParticles()
    q, s, e = [], [], []
    for i in range(n):
        qi, si, ei = nb.getParticleParameters(i)
        q.append(qi._value); s.append(si._value); e.append(ei._value)
    exc = {}
    for k in range(nb.getNumExceptions()):
        i, j, qq, sg, ep = nb.getExceptionParameters(k)
        key = _pair(i, j)
        if key in exc:
            raise AlchemicalHamiltonianError(f"exception {key} is listed twice")
        exc[key] = (qq._value, sg._value, ep._value)
    return _EndState(q, s, e, exc)


_NB_METHODS = {0: "NoCutoff", 1: "CutoffNonPeriodic", 2: "CutoffPeriodic", 3: "Ewald", 4: "PME",
               5: "LJPME"}


def _nonbonded_settings(nb) -> dict[str, Any]:
    return {
        "method": _NB_METHODS.get(nb.getNonbondedMethod(), str(nb.getNonbondedMethod())),
        "cutoff_nm": nb.getCutoffDistance()._value,
        "ewald_error_tolerance": nb.getEwaldErrorTolerance(),
        "use_switching_function": nb.getUseSwitchingFunction(),
        "switching_distance_nm": nb.getSwitchingDistance()._value,
        "use_dispersion_correction": nb.getUseDispersionCorrection(),
        "exceptions_use_periodic": nb.getExceptionsUsePeriodicBoundaryConditions(),
        "include_direct_space": nb.getIncludeDirectSpace(),
    }


def _pme_parameters(system, nb) -> tuple[float, int, int, int]:
    """The (alpha, nx, ny, nz) OpenMM would choose, fixed once so both end states share them."""
    mm = _mm()
    alpha, nx, ny, nz = nb.getPMEParameters()
    if alpha._value > 0:
        return alpha._value, nx, ny, nz
    probe = mm.System()
    for i in range(system.getNumParticles()):
        probe.addParticle(system.getParticleMass(i))
    probe.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    probe.addForce(mm.XmlSerializer.deserialize(_xml(nb)))
    context = mm.Context(probe, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(BUILD_PROBE_PLATFORM))
    try:
        return tuple(context.getSystem().getForce(0).getPMEParametersInContext(context))  # type: ignore
    finally:
        del context


# --------------------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------------------

def from_plan(plan, settings: SoftcoreSettings | None = None) -> "AlchemicalHamiltonian":
    """Build from a topology plan exposing `system_a`, `system_b`, `a_only` and `b_only`."""
    return build_hamiltonian(plan.system_a, plan.system_b, plan.a_only, plan.b_only,
                             settings=settings)


def build_hamiltonian(system_a, system_b, a_only: Iterable[int], b_only: Iterable[int], *,
                      settings: SoftcoreSettings | None = None) -> "AlchemicalHamiltonian":
    settings = settings or SoftcoreSettings()
    mm = _mm()
    n = system_a.getNumParticles()
    a_set, b_set = frozenset(int(i) for i in a_only), frozenset(int(i) for i in b_only)
    problems = _structural_problems(system_a, system_b, a_set, b_set)
    if problems:
        raise AlchemicalHamiltonianError("the end states cannot be combined:\n  - "
                                         + "\n  - ".join(problems))
    if not settings.sc and (a_set or b_set):
        raise AlchemicalHamiltonianError(
            f"alchemical.sc is false but {len(a_set)} particle(s) disappear and {len(b_set)} "
            "appear. Without softcore that is linear insertion/removal, whose integrand diverges "
            "at the end point; it is refused rather than softened behind your back. Set sc: true, "
            "or use a map with no unique particles")
    common = frozenset(range(n)) - a_set - b_set
    boundary_scaled = settings.sc_boundary_14 == "scaled"

    nb_a, nb_b = _nonbonded_force(system_a, "A"), _nonbonded_force(system_b, "B")
    settings_a, settings_b = _nonbonded_settings(nb_a), _nonbonded_settings(nb_b)
    if settings_a != settings_b:
        diff = {k: (settings_a[k], settings_b[k]) for k in settings_a if settings_a[k] != settings_b[k]}
        raise AlchemicalHamiltonianError(f"the end states' NonbondedForce settings differ: {diff}")
    method = settings_a["method"]
    if method not in ("NoCutoff", "PME"):
        raise AlchemicalHamiltonianError(
            f"nonbonded method {method} is not supported: the Amber18 softcore electrostatics are "
            "defined here for vacuum (NoCutoff) and PME only")
    if not settings_a["include_direct_space"]:
        raise AlchemicalHamiltonianError("a NonbondedForce without direct space is not supported")
    periodic = method == "PME"

    A, B = _read_end_state(nb_a), _read_end_state(nb_b)
    problems = _end_state_problems(A, B, a_set, b_set, common)
    if problems:
        raise AlchemicalHamiltonianError("the end states' nonbonded parameters are not a valid "
                                         "alchemical pair:\n  - " + "\n  - ".join(problems))

    kappa = None
    pme = None
    if periodic:
        pme = _pme_parameters(system_a, nb_a)
        kappa = float(pme[0])

    # --- the new System: particles, masses, constraints, box --------------------------------
    system = mm.System()
    for i in range(n):
        system.addParticle(system_a.getParticleMass(i))
    for k in range(system_a.getNumConstraints()):
        i, j, d = system_a.getConstraintParameters(k)
        system.addConstraint(i, j, d)
    system.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())

    record: dict[str, Any] = {
        "schema": HAMILTONIAN_SCHEMA,
        "softcore": settings.record(),
        "rules": {
            "reference": "Amber18 manual section 21.1.5, eqs. 21.3, 21.5-21.7",
            "pme_direct_space": "q_i q_j erfc(kappa r)/sqrt(r^2 + beta l); reciprocal, self and "
                                "background per end state, unchanged (pmemd 26 pairs_calc_ti, "
                                "gti_nonBond_kernels eleGauss=0)",
            "softcore_internal_nonbonded": "unscaled, vacuum Coulomb + LJ, no cutoff",
            "softcore_internal_exceptions": "unscaled, from the end state where the region is "
                                            "physical",
            "boundary_14": settings.sc_boundary_14,
            "bonded_touching_softcore": "identical in both end states: unscaled; force constant 0 "
                                        "at the dummy end only: mixed with lambda_bonded; "
                                        "anything else refused",
            "a_only_x_b_only": "excluded",
            "dispersion_correction": "each end state's own NonbondedForce correction, mixed "
                                     "linearly in lambda_sterics (pmemd per-region weighting); "
                                     "softcore tails not softened",
            "amber18_path": "the diagonal lambda_electrostatics = lambda_sterics = lambda_bonded",
        },
        "nonbonded": dict(settings_a, kappa_nm_inv=kappa,
                          pme_grid=list(pme[1:]) if pme else None),
        "particles": {"total": n, "common": len(common), "a_only": sorted(a_set),
                      "b_only": sorted(b_set)},
        "end_states": {"system_a_sha256": _sha256(_xml(system_a)),
                       "system_b_sha256": _sha256(_xml(system_b))},
        "public_parameters": list(PUBLIC_PARAMETERS),
        "force_groups": dict(FORCE_GROUPS),
    }

    static_lj = frozenset(i for i in common
                          if (A.sigma[i], A.epsilon[i]) == (B.sigma[i], B.epsilon[i])
                          or (A.epsilon[i] == 0.0 and B.epsilon[i] == 0.0))
    changing_lj = common - static_lj
    record["particles"]["common_lj_changing"] = sorted(changing_lj)

    for label, region in (("A-only", a_set), ("B-only", b_set)):
        groups = _connected_groups(region, system_a, system_b)
        if len(groups) > 1:
            raise AlchemicalHamiltonianError(
                f"the {label} particles form {len(groups)} separate groups "
                f"{[sorted(g) for g in groups]}. Pairs BETWEEN two unique groups are zero at the "
                "dummy end (contract section 4), and how they are switched along lambda is not "
                "defined here yet; one unique group per side is supported")
    internal_pairs = {"A": _internal_pairs(a_set, A.exceptions), "B": _internal_pairs(b_set, B.exceptions)}
    checked = _check_internal_force(system_a, system_b, A, B, a_set, b_set, internal_pairs)
    # ONE exclusion set for every nonbonded force in the System. The CUDA platform refuses a
    # Context otherwise ("All Forces must have identical exceptions"); Reference and CPU do not
    # check, so a CPU-only suite passed with forces that differed (found by the first CUDA lane,
    # 2026-09-19). Every exception of either end state, and both regions' internal pairs.
    internal_all = sorted(set(internal_pairs["A"]) | set(internal_pairs["B"]))
    exclusions = set(A.exceptions) | set(B.exceptions) | set(internal_all)

    # --- bonded and every other non-nonbonded force ------------------------------------------
    mixed_bonded = _add_other_forces(system, system_a, system_b, a_set, b_set)

    # --- end-state electrostatics (+ static LJ in A) ------------------------------------------
    for label, state, region in (("A", A, a_set), ("B", B, b_set)):
        system.addForce(_end_state_nonbonded(
            label, state, region, a_set | b_set, static_lj, internal_all, settings_a, pme,
            boundary_scaled))

    # --- common-core LJ that changes ------------------------------------------------------------
    if changing_lj:
        system.addForce(_changing_lj_force(A, B, changing_lj, static_lj, exclusions, settings_a))

    # --- softcore regions -------------------------------------------------------------------------
    for label, state, region in (("A", A, a_set), ("B", B, b_set)):
        if not region:
            continue
        elec, lj = _softcore_forces(label, state, region, common, exclusions, settings_a, kappa,
                                    settings)
        system.addForce(elec)
        system.addForce(lj)

    internal = _internal_force(A, B, a_set, b_set, internal_pairs, periodic, boundary_scaled)
    if internal is not None:
        system.addForce(internal)

    if periodic and settings_a["use_dispersion_correction"] and settings_a["use_switching_function"]:
        raise AlchemicalHamiltonianError(
            "a dispersion correction together with a switching function is not implemented: the "
            "switched correction's integral is not reproduced here yet, and an approximate one "
            "would make neither end state recover its own energy")
    if periodic and settings_a["use_dispersion_correction"]:
        force, coefficients = _dispersion_force(system_a, A, B, static_lj, settings_a, pme,
                                                exclusions)
        if force is not None:
            system.addForce(force)
        record["nonbonded"]["dispersion_coefficients_kj_nm3_mol"] = coefficients
    record["bonded_mixed_forces"] = mixed_bonded
    record["plan_internal_pairs_checked"] = checked
    return AlchemicalHamiltonian(system=system, record=record)


def _structural_problems(sa, sb, a_set, b_set) -> list[str]:
    out: list[str] = []
    n = sa.getNumParticles()
    if sb.getNumParticles() != n:
        return [f"particle counts differ: A has {n}, B has {sb.getNumParticles()}"]
    if a_set & b_set:
        out.append(f"particles are both A-only and B-only: {sorted(a_set & b_set)}")
    bad = sorted(i for i in a_set | b_set if not 0 <= i < n)
    if bad:
        out.append(f"unique particle indices out of range 0..{n - 1}: {bad}")
    masses = [i for i in range(n) if sa.getParticleMass(i)._value != sb.getParticleMass(i)._value]
    if masses:
        out.append(f"particle masses differ at {masses[:10]}")
    vs = [i for i in range(n) if sa.isVirtualSite(i) or sb.isVirtualSite(i)]
    if vs:
        out.append(f"virtual sites are not supported yet (particles {vs[:10]})")

    def constraints(s):
        return sorted((*_pair(*s.getConstraintParameters(k)[:2]), s.getConstraintParameters(k)[2]._value)
                      for k in range(s.getNumConstraints()))
    if constraints(sa) != constraints(sb):
        out.append("the end states' constraints differ; the plan must make them identical")
    if sa.usesPeriodicBoundaryConditions() != sb.usesPeriodicBoundaryConditions():
        out.append("one end state is periodic and the other is not")
    elif sa.usesPeriodicBoundaryConditions():
        va = [v._value for v in sa.getDefaultPeriodicBoxVectors()]
        vb = [v._value for v in sb.getDefaultPeriodicBoxVectors()]
        if [list(v) for v in va] != [list(v) for v in vb]:
            out.append("the end states' default periodic box vectors differ")
    ta = [_type(f) for f in sa.getForces()]
    tb = [_type(f) for f in sb.getForces()]
    if ta != tb:
        out.append(f"the end states' force layouts differ: A {ta}, B {tb}")
    return out


def _end_state_problems(A: _EndState, B: _EndState, a_set, b_set, common) -> list[str]:
    out: list[str] = []
    for label, state, dummies in (("A", A, b_set), ("B", B, a_set)):
        for i in sorted(dummies):
            if state.charge[i] != 0.0 or state.epsilon[i] != 0.0:
                out.append(f"particle {i} is a dummy in System {label} but carries charge "
                           f"{state.charge[i]} / epsilon {state.epsilon[i]}")
    if set(A.exceptions) != set(B.exceptions):
        only_a = sorted(set(A.exceptions) - set(B.exceptions))[:5]
        only_b = sorted(set(B.exceptions) - set(A.exceptions))[:5]
        out.append(f"the exception pair sets differ (A only {only_a}, B only {only_b})")
    for label, state, dummies, physical in (("A", A, b_set, B), ("B", B, a_set, A)):
        for (i, j), (qq, sg, ep) in state.exceptions.items():
            if not (i in dummies or j in dummies):
                continue
            if i in dummies and j in dummies:
                # contract section 4: a unique group's own exceptions keep their physical values
                # at its dummy end (the Amber convention; the plan is the one definition)
                if not _same(state.exceptions[(i, j)], physical.exceptions[(i, j)]):
                    out.append(f"exception {(i, j)} is internal to the unique group that is a "
                               f"dummy in System {label}, and differs from its physical value "
                               f"{physical.exceptions[(i, j)]}; the plan keeps a group's own "
                               "exceptions physical at its dummy end")
            elif qq != 0.0 or ep != 0.0:
                out.append(f"exception {(i, j)} joins a dummy of System {label} to a particle "
                           "outside its group, but is not zero")
    for i in a_set:
        for j in b_set:
            if _pair(i, j) not in A.exceptions:
                out.append(f"A-only {i} and B-only {j} are not excluded from each other")
                break
    qa, qb = sum(A.charge), sum(B.charge)
    if abs(qa - qb) > 1e-6:
        out.append(f"net charge changes ({qa:+.6f} -> {qb:+.6f}); charge-changing transformations "
                   "need a finite-size correction and are not supported yet")
    for i in sorted(common):
        if (A.epsilon[i] > 0.0) != (B.epsilon[i] > 0.0):
            out.append(f"common particle {i} has Lennard-Jones in one end state only (epsilon "
                       f"{A.epsilon[i]} -> {B.epsilon[i]}); a particle losing its LJ must be in "
                       "a softcore region, not mixed linearly")
    return out


def _same(p, q, rel=1e-12) -> bool:
    return all(abs(a - b) <= rel * max(1.0, abs(a), abs(b)) for a, b in zip(p, q))


def _connected_groups(region, system_a, system_b) -> list[set[int]]:
    """The region's particles split into groups connected by bonds or constraints (both ends)."""
    mm = _mm()
    edges = set()
    for s in (system_a, system_b):
        for k in range(s.getNumConstraints()):
            i, j, _ = s.getConstraintParameters(k)
            edges.add((i, j))
        for f in s.getForces():
            if isinstance(f, mm.HarmonicBondForce):
                for k in range(f.getNumBonds()):
                    i, j, *_ = f.getBondParameters(k)
                    edges.add((i, j))
    parent = {i: i for i in region}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        if i in region and j in region:
            parent[root(i)] = root(j)
    groups: dict[int, set[int]] = {}
    for i in region:
        groups.setdefault(root(i), set()).add(i)
    return list(groups.values())


def _check_internal_force(system_a, system_b, A, B, a_set, b_set, internal_pairs) -> int:
    """S2's `UniqueGroupInternalNonbonded` force must be exactly the internal pairs this Hamiltonian
    keeps unscaled: physical values at the group's dummy end, zero at its physical end. It is then
    left out -- the Hamiltonian's own softcore-internal force carries those pairs at every lambda,
    and adding the plan's force too would count them twice. Returns the number of pairs checked."""
    mm = _mm()
    forces = []
    for s in (system_a, system_b):
        found = [f for f in s.getForces() if f.getName() == INTERNAL_FORCE_NAME]
        if len(found) > 1:
            raise AlchemicalHamiltonianError(f"more than one {INTERNAL_FORCE_NAME} force")
        forces.append(found[0] if found else None)
    fa, fb = forces
    expected = {**{p: ("B", A) for p in internal_pairs["A"]},     # A's pairs: dummy end is B
                **{p: ("A", B) for p in internal_pairs["B"]}}
    if fa is None and fb is None:
        if expected:
            raise AlchemicalHamiltonianError(
                f"the unique groups have {len(expected)} internal non-excluded pair(s), but the end "
                f"states carry no {INTERNAL_FORCE_NAME} force: by contract section 4 a group's "
                "internal pairs stay physical at its dummy end, so the end states are not the "
                "plan's")
        return 0
    if fa is None or fb is None or not isinstance(fa, mm.CustomBondForce) \
            or fa.getNumBonds() != fb.getNumBonds():
        raise AlchemicalHamiltonianError(f"{INTERNAL_FORCE_NAME} is not the same CustomBondForce "
                                         "layout in both end states")
    seen = set()
    for k in range(fa.getNumBonds()):
        i, j, pa = fa.getBondParameters(k)
        i2, j2, pb = fb.getBondParameters(k)
        key = _pair(i, j)
        if key != _pair(i2, j2) or key not in expected:
            raise AlchemicalHamiltonianError(
                f"{INTERNAL_FORCE_NAME} bond {k} on {key} is not an internal non-excluded pair of "
                "one unique group")
        dummy_end, state = expected[key]
        physical = (state.charge[i] * state.charge[j], 0.5 * (state.sigma[i] + state.sigma[j]),
                    math.sqrt(state.epsilon[i] * state.epsilon[j]))
        at_dummy, at_physical = (pb, pa) if dummy_end == "B" else (pa, pb)
        if not _same(at_dummy, physical) or at_physical[0] != 0.0 or at_physical[2] != 0.0:
            raise AlchemicalHamiltonianError(
                f"{INTERNAL_FORCE_NAME} pair {key} is {tuple(at_dummy)} at its dummy end and "
                f"{tuple(at_physical)} at its physical end; expected {physical} and zero")
        seen.add(key)
    if seen != set(expected):
        raise AlchemicalHamiltonianError(
            f"{INTERNAL_FORCE_NAME} omits internal pair(s) {sorted(set(expected) - seen)[:5]}")
    return len(seen)


def _internal_pairs(region, exceptions) -> list[tuple[int, int]]:
    """Non-excluded pairs inside a region (not already exceptions)."""
    members = sorted(region)
    return [(i, j) for x, i in enumerate(members) for j in members[x + 1:]
            if (i, j) not in exceptions]


def _add_other_forces(system, sa, sb, a_set, b_set) -> list[dict[str, Any]]:
    mm = _mm()
    mixed: list[dict[str, Any]] = []
    for k, (fa, fb) in enumerate(zip(sa.getForces(), sb.getForces())):
        kind = _type(fa)
        if kind == "NonbondedForce" or fa.getName() == INTERNAL_FORCE_NAME:
            continue              # the NonbondedForce is rebuilt; the plan's internal-pair force is
            #                       checked by `_check_internal_force` and carried by group 8
        xa, xb = _xml(fa), _xml(fb)
        if xa == xb:
            copy = mm.XmlSerializer.deserialize(xa)
            copy.setForceGroup(FORCE_GROUPS["static"])
            system.addForce(copy)
            continue
        if kind not in _MIXABLE_BONDED:
            raise AlchemicalHamiltonianError(
                f"force {k} ({kind}) differs between the end states and is not a force this "
                f"Hamiltonian mixes (it mixes {list(_MIXABLE_BONDED)})")
        terms_a, terms_b = _bonded_terms(fa), _bonded_terms(fb)
        if [t[0] for t in terms_a] != [t[0] for t in terms_b]:
            raise AlchemicalHamiltonianError(
                f"force {k} ({kind}): the end states list different atoms in the same term slots; "
                "the plan must give both end states one term layout")
        for slot, ((atoms, pa), (_, pb)) in enumerate(zip(terms_a, terms_b)):
            if pa != pb and set(atoms) & (a_set | b_set):
                problem = _boundary_term_problem(atoms, pa, pb, a_set, b_set)
                if problem:
                    raise AlchemicalHamiltonianError(
                        f"force {k} ({kind}) term {slot} on particles {atoms} touches a softcore "
                        f"particle and differs between the end states ({pa} vs {pb}): {problem}")
        system.addForce(_mixed_bonded_force(kind, terms_a, terms_b, fa.usesPeriodicBoundaryConditions()))
        mixed.append({"index": k, "type": kind, "terms": len(terms_a),
                      "differing_terms": sum(pa != pb for (_, pa), (_, pb) in zip(terms_a, terms_b))})
    return mixed


def _boundary_term_problem(atoms, pa, pb, a_set, b_set) -> str | None:
    """None if a term touching a region differs only by being removed at that region's dummy end.

    The force constant is the LAST parameter of every mixable term; everything before it
    (length, angle, periodicity, phase) must be equal.
    """
    touched_a, touched_b = bool(set(atoms) & a_set), bool(set(atoms) & b_set)
    if touched_a and touched_b:
        return "it spans both regions, which the plan must exclude"
    if pa[:-1] != pb[:-1]:
        return ("its geometry differs; only a force constant removed at the dummy end may differ "
                "for a term on a softcore particle")
    dummy_end_k = pb[-1] if touched_a else pa[-1]
    if dummy_end_k != 0.0:
        return ("its force constant differs at both ends; Amber18 does not scale a bonded term on "
                "an appearing or disappearing atom, and the only exception accepted is a term the "
                "plan removes at the dummy end (force constant 0 there)")
    return None


def _bonded_terms(force) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    kind = _type(force)
    out = []
    if kind == "HarmonicBondForce":
        for k in range(force.getNumBonds()):
            i, j, r0, kk = force.getBondParameters(k)
            out.append(((i, j), (r0._value, kk._value)))
    elif kind == "HarmonicAngleForce":
        for k in range(force.getNumAngles()):
            i, j, l, t0, kk = force.getAngleParameters(k)
            out.append(((i, j, l), (t0._value, kk._value)))
    else:
        for k in range(force.getNumTorsions()):
            i, j, l, m, per, ph, kk = force.getTorsionParameters(k)
            out.append(((i, j, l, m), (float(per), ph._value, kk._value)))
    return out


def _mixed_bonded_force(kind, terms_a, terms_b, periodic):
    mm = _mm()
    if kind == "HarmonicBondForce":
        f = mm.CustomBondForce("(1-lambda_bonded)*0.5*ka*(r-ra)^2 + lambda_bonded*0.5*kb*(r-rb)^2")
        for p in ("ra", "ka", "rb", "kb"):
            f.addPerBondParameter(p)
        add = lambda atoms, pa, pb: f.addBond(*atoms, [*pa, *pb])  # noqa: E731
    elif kind == "HarmonicAngleForce":
        f = mm.CustomAngleForce("(1-lambda_bonded)*0.5*ka*(theta-ta)^2 + lambda_bonded*0.5*kb*(theta-tb)^2")
        for p in ("ta", "ka", "tb", "kb"):
            f.addPerAngleParameter(p)
        add = lambda atoms, pa, pb: f.addAngle(*atoms, [*pa, *pb])  # noqa: E731
    else:
        f = mm.CustomTorsionForce("(1-lambda_bonded)*ka*(1+cos(na*theta-pa))"
                                  " + lambda_bonded*kb*(1+cos(nb*theta-pb))")
        for p in ("na", "pa", "ka", "nb", "pb", "kb"):
            f.addPerTorsionParameter(p)
        add = lambda atoms, pa, pb: f.addTorsion(*atoms, [*pa, *pb])  # noqa: E731
    f.addGlobalParameter(LAMBDA_BONDED, 0.0)
    f.addEnergyParameterDerivative(LAMBDA_BONDED)
    for (atoms, pa), (_, pb) in zip(terms_a, terms_b):
        add(atoms, pa, pb)
    f.setUsesPeriodicBoundaryConditions(periodic)
    f.setForceGroup(FORCE_GROUPS["bonded_mixed"])
    return f


def _configure_nonbonded(force, nb_settings, pme):
    mm = _mm()
    method = nb_settings["method"]
    force.setNonbondedMethod(mm.NonbondedForce.PME if method == "PME" else mm.NonbondedForce.NoCutoff)
    force.setCutoffDistance(nb_settings["cutoff_nm"])
    force.setEwaldErrorTolerance(nb_settings["ewald_error_tolerance"])
    force.setUseSwitchingFunction(nb_settings["use_switching_function"])
    force.setSwitchingDistance(nb_settings["switching_distance_nm"])
    force.setExceptionsUsePeriodicBoundaryConditions(nb_settings["exceptions_use_periodic"])
    if pme is not None:
        force.setPMEParameters(*pme)


def _end_state_nonbonded(label, state: _EndState, region, unique, static_lj, internal_pairs,
                         nb_settings, pme, boundary_scaled: bool):
    """End state `label`'s electrostatics times its weight, plus (in A) the LJ both states share."""
    mm = _mm()
    f = mm.NonbondedForce()
    _configure_nonbonded(f, nb_settings, pme)
    carries_static_lj = label == "A"
    f.setUseDispersionCorrection(bool(nb_settings["use_dispersion_correction"]) and carries_static_lj)
    default = {"A": 1.0, "B": 0.0}[label]     # the state lambda = 0
    for name in (_QSCALE[label], _W_EL[label], _W_LJ[label]):
        f.addGlobalParameter(name, default)
    if label == "A":
        for name in PUBLIC_PARAMETERS:        # so every public name exists in every Context
            f.addGlobalParameter(name, 0.0)
    for i, (q, s, e) in enumerate(zip(state.charge, state.sigma, state.epsilon)):
        eps = e if (carries_static_lj and i in static_lj) else 0.0
        f.addParticle(0.0, s, eps)
        if q != 0.0:
            f.addParticleParameterOffset(_QSCALE[label], i, q, 0.0, 0.0)
    for (i, j), (qq, sg, ep) in sorted(state.exceptions.items()):
        both_unique = i in unique and j in unique       # inside a region, or A-only x B-only
        if both_unique or ((i in unique or j in unique) and not boundary_scaled):
            f.addException(i, j, 0.0, sg, 0.0)          # removed like an exclusion; see U_unscaled
            continue
        static = i in static_lj and j in static_lj
        if static and carries_static_lj:
            k = f.addException(i, j, 0.0, sg, ep)
        else:
            k = f.addException(i, j, 0.0, sg, 0.0)
            if ep != 0.0 and not static:
                f.addExceptionParameterOffset(_W_LJ[label], k, 0.0, 0.0, ep)
        if qq != 0.0:
            f.addExceptionParameterOffset(_W_EL[label], k, qq, 0.0, 0.0)
    for i, j in internal_pairs:
        f.addException(i, j, 0.0, 1.0, 0.0)
    f.setForceGroup(FORCE_GROUPS[f"nonbonded_{label.lower()}"])
    return f


def _dispersion_coefficient(sigma, epsilon, cutoff: float) -> float:
    """C in OpenMM's NonbondedForce dispersion correction E = C / V, for these LJ parameters.

    OpenMM's formula (NonbondedForceImpl::calcDispersionCorrection, no switching function): group
    particles into (sigma, epsilon) classes; over class pairs i <= j with count n_i n_j, or
    n_i (n_i + 1) / 2 on the diagonal, accumulate count * eps_ij sigma_ij^12 and
    count * eps_ij sigma_ij^6; divide both by N (N + 1) / 2; then
    C = 8 pi N^2 (s12 / (9 rc^9) - s6 / (3 rc^3)). It is written out rather than measured because
    measuring it means subtracting two large pair energies; `test_dispersion_coefficient_is_openmms`
    holds this against OpenMM's own number.
    """
    classes: dict[tuple[float, float], int] = {}
    for s_i, e_i in zip(sigma, epsilon):
        classes[(s_i, e_i)] = classes.get((s_i, e_i), 0) + 1
    keys = list(classes)
    s12 = s6 = 0.0
    for a, (s_a, e_a) in enumerate(keys):
        for s_b, e_b in keys[a:]:
            n_a, n_b = classes[(s_a, e_a)], classes[(s_b, e_b)]
            count = n_a * (n_a + 1) / 2 if (s_a, e_a) == (s_b, e_b) else n_a * n_b
            sig = 0.5 * (s_a + s_b)
            eps = math.sqrt(e_a * e_b)
            sig6 = sig ** 6
            s12 += count * eps * sig6 * sig6
            s6 += count * eps * sig6
    n = len(sigma)
    interactions = n * (n + 1) / 2
    return 8 * n * n * math.pi * (s12 / interactions / (9 * cutoff ** 9)
                                  - s6 / interactions / (3 * cutoff ** 3))


def _dispersion_force(system_a, A, B, static_lj, nb_settings, pme, exclusions):
    """The part of (1-lam_s) D_A + lam_s D_B that NonbondedForce A's static correction lacks."""
    mm = _mm()
    n = len(A.charge)
    cutoff = float(nb_settings["cutoff_nm"])
    static_eps = [A.epsilon[i] if i in static_lj else 0.0 for i in range(n)]
    c_static = _dispersion_coefficient(A.sigma, static_eps, cutoff)
    c_a = _dispersion_coefficient(A.sigma, A.epsilon, cutoff)
    c_b = _dispersion_coefficient(B.sigma, B.epsilon, cutoff)
    coefficients = {"static": c_static, "A": c_a, "B": c_b}
    k_a, k_b = c_a - c_static, c_b - c_static
    if n < 2 or (k_a == 0.0 and k_b == 0.0):
        return None, coefficients
    # The carrier's one pair must not be excluded: it carries the System's exclusion set like every
    # other nonbonded force, and an excluded pair would not be an interaction at all.
    pair = next(((i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in exclusions),
                None)
    if pair is None:
        raise AlchemicalHamiltonianError("every particle pair is excluded; there is no pair to "
                                         "carry the dispersion correction")

    def force(ka: float, kb: float):
        # With ka == kb the term does not depend on lambda, and it must not SAY it does: OpenMM
        # differentiates the long-range correction for an energy-parameter derivative, finds the
        # derivative expression simplifies to 0, and aborts the process ("Cannot use long range
        # correction with a force that does not depend on r") -- an abort, not an exception.
        varies = ka != kb
        weight = (f"((1-lambda_sterics)*{ka!r} + lambda_sterics*{kb!r})" if varies else f"{ka!r}")
        f = mm.CustomNonbondedForce(f"{weight}*step(r-{cutoff!r})*({cutoff!r}/r)^6")
        for _ in range(n):
            f.addParticle([])
        f.addInteractionGroup([pair[0]], [pair[1]])
        for i, j in sorted(exclusions):
            f.addExclusion(i, j)
        f.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        f.setCutoffDistance(cutoff)
        f.setUseSwitchingFunction(False)
        f.setUseLongRangeCorrection(True)
        if varies:
            f.addGlobalParameter(LAMBDA_STERICS, 0.0)
            f.addEnergyParameterDerivative(LAMBDA_STERICS)
        f.setForceGroup(FORCE_GROUPS["dispersion"])
        return f

    # Calibrate: this force's correction per unit coefficient, times V, as OpenMM computes it.
    probe = mm.System()
    for i in range(n):
        probe.addParticle(1.0)
    box = system_a.getDefaultPeriodicBoxVectors()
    probe.setDefaultPeriodicBoxVectors(*box)
    probe.addForce(force(1.0, 2.0))        # at lambda_sterics = 0 the weight is exactly 1.0
    context = mm.Context(probe, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(BUILD_PROBE_PLATFORM))
    context.setPositions([[0.01 * i, 0.0, 0.0] for i in range(n)])    # the pair is inside the cutoff
    volume = box[0][0]._value * box[1][1]._value * box[2][2]._value
    unit = context.getState(getEnergy=True).getPotentialEnergy()._value * volume
    del context
    coefficients["unit"] = unit
    return force(k_a / unit, k_b / unit), coefficients


def _custom_nonbonded(expression, nb_settings):
    mm = _mm()
    f = mm.CustomNonbondedForce(expression)
    if nb_settings["method"] == "PME":
        f.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    else:
        f.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    f.setCutoffDistance(nb_settings["cutoff_nm"])
    f.setUseSwitchingFunction(nb_settings["use_switching_function"])
    f.setSwitchingDistance(nb_settings["switching_distance_nm"])
    return f


def _changing_lj_force(A, B, changing, static, exclusions, nb_settings):
    f = _custom_nonbonded(
        "(1-lambda_sterics)*4*ea*((sa/r)^12-(sa/r)^6) + lambda_sterics*4*eb*((sb/r)^12-(sb/r)^6);"
        " sa=0.5*(sa1+sa2); ea=sqrt(ea1*ea2); sb=0.5*(sb1+sb2); eb=sqrt(eb1*eb2)", nb_settings)
    for p in ("sa", "ea", "sb", "eb"):
        f.addPerParticleParameter(p)
    f.addGlobalParameter(LAMBDA_STERICS, 0.0)
    f.addEnergyParameterDerivative(LAMBDA_STERICS)
    for i in range(len(A.charge)):
        f.addParticle([A.sigma[i], A.epsilon[i], B.sigma[i], B.epsilon[i]])
    for i, j in sorted(exclusions):
        f.addExclusion(i, j)
    f.addInteractionGroup(sorted(changing), sorted(static))
    f.addInteractionGroup(sorted(changing), sorted(changing))
    f.setUseLongRangeCorrection(False)      # the tail is `_dispersion_force`'s, see there
    f.setForceGroup(FORCE_GROUPS["lj_common_changing"])
    return f


def _softcore_forces(label, state, region, common, exclusions, nb_settings, kappa, settings):
    """Region `label` x common: the electrostatic delta and the whole softcore LJ, as two forces."""
    elec_expr, lj_expr = pair_expressions(region=label, kappa=kappa, scalpha=settings.scalpha,
                                          scbeta_nm2=settings.scbeta_nm2)
    out = []
    for kind, expr, params in (("elec", elec_expr, ("q",)), ("lj", lj_expr, ("sig", "eps"))):
        f = _custom_nonbonded(expr, nb_settings)
        for p in params:
            f.addPerParticleParameter(p)
        public = LAMBDA_ELECTROSTATICS if kind == "elec" else LAMBDA_STERICS
        f.addGlobalParameter(public, 0.0)
        f.addEnergyParameterDerivative(public)
        for i in range(len(state.charge)):
            if kind == "elec":
                f.addParticle([state.charge[i]])
            else:
                sig = state.sigma[i] if state.epsilon[i] != 0.0 else _SIGMA_FOR_ZERO_EPSILON
                f.addParticle([sig, state.epsilon[i]])
        for i, j in sorted(exclusions):
            f.addExclusion(i, j)
        f.addInteractionGroup(sorted(region), sorted(common))
        f.setUseLongRangeCorrection(False)  # the tail is `_dispersion_force`'s, see there
        f.setForceGroup(FORCE_GROUPS[f"softcore_{label.lower()}_{kind}"])
        out.append(f)
    return out


def _internal_force(A, B, a_set, b_set, internal_pairs, periodic, boundary_scaled):
    mm = _mm()
    f = mm.CustomBondForce(f"{ONE_4PI_EPS0!r}*qq/r + 4*eps*((sig/r)^12 - (sig/r)^6)")
    for p in ("qq", "sig", "eps"):
        f.addPerBondParameter(p)
    count = 0
    for state, region, label in ((A, a_set, "A"), (B, b_set, "B")):
        for i, j in internal_pairs[label]:
            eps = math.sqrt(state.epsilon[i] * state.epsilon[j])
            f.addBond(i, j, [state.charge[i] * state.charge[j],
                             0.5 * (state.sigma[i] + state.sigma[j]), eps])
            count += 1
        for (i, j), (qq, sg, ep) in sorted(state.exceptions.items()):
            inside = i in region and j in region
            boundary = (i in region) != (j in region)
            if (inside or (boundary and not boundary_scaled)) and (qq != 0.0 or ep != 0.0):
                f.addBond(i, j, [qq, sg, ep])
                count += 1
    if not count:
        return None
    f.setUsesPeriodicBoundaryConditions(periodic)
    f.setForceGroup(FORCE_GROUPS["softcore_internal"])
    return f


# --------------------------------------------------------------------------------------------------
# The Hamiltonian object
# --------------------------------------------------------------------------------------------------

def _check_state(state: Mapping[str, float]) -> dict[str, float]:
    unknown = sorted(set(state) - set(PUBLIC_PARAMETERS))
    missing = sorted(set(PUBLIC_PARAMETERS) - set(state))
    if unknown or missing:
        raise AlchemicalHamiltonianError(
            f"a state names exactly {list(PUBLIC_PARAMETERS)}; unknown {unknown}, missing {missing}")
    out = {}
    for name in PUBLIC_PARAMETERS:
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise AlchemicalHamiltonianError(f"{name} must be a number in [0, 1], got {value!r}")
        out[name] = float(value)
    return out


@dataclass
class AlchemicalHamiltonian:
    system: Any
    record: dict[str, Any] = field(default_factory=dict)

    parameter_names = PUBLIC_PARAMETERS

    @staticmethod
    def context_parameters(state: Mapping[str, float]) -> dict[str, float]:
        """Every Context parameter value for `state`: the public ones and the derived ones.

        This is the ONE definition of the derived parameters; a record of a state is this dict.
        """
        s = _check_state(state)
        le, ls = s[LAMBDA_ELECTROSTATICS], s[LAMBDA_STERICS]
        return {
            **s,
            _QSCALE["A"]: math.sqrt(1.0 - le), _QSCALE["B"]: math.sqrt(le),
            _W_EL["A"]: 1.0 - le, _W_EL["B"]: le,
            _W_LJ["A"]: 1.0 - ls, _W_LJ["B"]: ls,
        }

    def set_state(self, context, state: Mapping[str, float]) -> None:
        """Set every Context parameter for `state`. Parameters only: nothing is reinitialised.

        A Context that lacks any of them was not built from this Hamiltonian's System, and is
        refused rather than half-set.
        """
        values = self.context_parameters(state)
        missing = sorted(set(values) - set(context.getParameters().keys()))
        if missing:
            raise AlchemicalHamiltonianError(
                f"this Context lacks the Hamiltonian parameter(s) {missing}; it was not created "
                "from this Hamiltonian's System")
        for name, value in values.items():
            context.setParameter(name, value)

    @staticmethod
    def _energy(context, groups) -> float:
        return context.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy()._value

    def energy(self, context, state: Mapping[str, float]) -> float:
        """Total potential energy in kJ/mol at the Context's coordinates, at `state`."""
        self.set_state(context, state)
        return context.getState(getEnergy=True).getPotentialEnergy()._value

    def energy_components(self, context, state: Mapping[str, float]) -> dict[str, float]:
        """Energy per force group, kJ/mol. The groups sum to `energy`."""
        self.set_state(context, state)
        return {name: self._energy(context, {g}) for name, g in FORCE_GROUPS.items()}

    def derivative_components(self, context, state: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """dU/dlambda_k per public parameter, split by the force group it comes from.

        * Custom forces: OpenMM energy-parameter derivatives. Each custom expression is written in
          the public names, so these are the complete partials of those forces.
        * The two NonbondedForces: exact algebra, not OpenMM derivatives. Their energy is
          qscale^2 Q + w_el X + w_lj Y + (terms independent of lambda), exactly, so Q, X and Y are
          read as energy differences at qscale, w_el and w_lj = 1 and 0 with the others held, and
          d(qscale^2)/dlambda_e = -1 (A) or +1 (B). Nothing is differenced numerically.

        The Context's parameters are restored to `state` before returning.
        """
        params = self.context_parameters(state)
        self.set_state(context, state)
        out: dict[str, dict[str, float]] = {name: {} for name in PUBLIC_PARAMETERS}
        custom = {g for name, g in FORCE_GROUPS.items()
                  if name not in ("static", "nonbonded_a", "nonbonded_b")}
        for name, g in FORCE_GROUPS.items():
            if g not in custom:
                continue
            derivs = context.getState(getParameterDerivatives=True, groups={g}).getEnergyParameterDerivatives()
            for p in PUBLIC_PARAMETERS:
                if p in derivs.keys():
                    out[p][name] = float(derivs[p])
        try:
            for label, sign in (("A", -1.0), ("B", +1.0)):
                group = {FORCE_GROUPS[f"nonbonded_{label.lower()}"]}
                name = f"nonbonded_{label.lower()}"

                def swing(parameter: str) -> float:
                    context.setParameter(parameter, 1.0)
                    high = self._energy(context, group)
                    context.setParameter(parameter, 0.0)
                    low = self._energy(context, group)
                    context.setParameter(parameter, params[parameter])
                    return high - low

                q_part, x_part, y_part = swing(_QSCALE[label]), swing(_W_EL[label]), swing(_W_LJ[label])
                out[LAMBDA_ELECTROSTATICS][name] = sign * (q_part + x_part)
                out[LAMBDA_STERICS][name] = sign * y_part
        finally:
            self.set_state(context, state)
        return out

    def derivatives(self, context, state: Mapping[str, float]) -> dict[str, float]:
        """The complete partial dU/dlambda_k, kJ/mol, for every public parameter."""
        return {p: math.fsum(parts.values())
                for p, parts in self.derivative_components(context, state).items()}
