"""The Amber18 softcore pair potential: settings, and the OpenMM expressions that implement it.

WHAT "amber18" MEANS HERE, AND NOTHING ELSE

Amber18 manual section 21.1.5 (R5), equations 21.5-21.7, for a softcore atom i of the region that
DISAPPEARS (present in V0, absent from V1) and an atom j it interacts with:

    V_V0,disappearing = 4 eps (1 - lam) [ 1 / (alpha lam + (r/sigma)^6)^2
                                          - 1 / (alpha lam + (r/sigma)^6) ]          (21.5)
    V_V1,appearing    = 4 eps lam       [ 1 / (alpha (1-lam) + (r/sigma)^6)^2
                                          - 1 / (alpha (1-lam) + (r/sigma)^6) ]      (21.6)
    V_V0,disappearing = (1 - lam) q_i q_j / (4 pi eps0 sqrt(beta lam + r^2))           (21.7)

with "replace lam by (1 - lam) and vice versa for the form for appearing atoms" for 21.7. The
(1 - lam) and lam in front are the V(lam) = (1 - lam) V0 + lam V1 mixing weights (eq. 21.3); the
lam inside is the softening. alpha = scalpha, beta = scbeta.

Under PME the manual gives only the vacuum form. What pmemd actually computes -- read from
pmemd 26 source, `pairs_calc_ti_AUTO.i` (CPU, `pairs_calc_ti_sc_common_*`) and
`cuda/gti_nonBond_kernels.cu` (GPU, eleGauss = 0, eleExp = 2) -- is the direct-space term

    q_i q_j erfc(kappa r) / sqrt(r^2 + beta lam)

with erfc evaluated at the TRUE distance and only 1/r softened, while the reciprocal sum of each end
state is computed with that end state's full charges and left alone. The softening therefore
touches the direct-space pair and nothing else; the end state's reciprocal, self and exclusion
terms are exactly those of the end state's own Ewald sum. That is the form implemented here.

WHAT IT IS NOT

* OpenFE's default Gapsys potential, and its Beutler LJ option, are different functions. So is
  Amber's later smoothstep variant (`gti_lam_sch = 1`, Amber20+ eqs. 27.8-27.14) and the
  `gti_scale_beta = 1` form with a sigma-scaled electrostatic softening. None of them may run under
  the name `amber18`, and each is refused by name below rather than silently mapped onto this one.
* `sc: true` with `scalpha = 0.5`, `scbeta = 12 A^2` is the MD-tools DEFAULT, chosen by the user.
  It is not a claim that AMBER's own `ifsc` defaults to 1 (it defaults to 0).

UNITS: scbeta is given in square angstroms, as in an Amber mdin, and used in nm^2 (12 A^2 =
0.12 nm^2). scalpha is dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "SOFTCORE_FUNCTION", "SoftcoreSettings", "SoftcoreError", "ONE_4PI_EPS0",
    "ANGSTROM2_TO_NM2", "REFUSED_FUNCTIONS", "BOUNDARY_14_RULES", "pair_expressions",
]

SOFTCORE_FUNCTION = "amber18"

#: OpenMM's Coulomb constant, kJ nm / (mol e^2). Measured, not typed: the energy of two unit
#: charges 1 nm apart under NoCutoff on the Reference platform, OpenMM 8.6. Every custom
#: expression uses the SAME number as NonbondedForce, or a softcore delta built to cancel a
#: NonbondedForce term would leave a residue proportional to q_i q_j / r.
ONE_4PI_EPS0 = 138.93545764438198

ANGSTROM2_TO_NM2 = 0.01

#: Names a user could plausibly type that are NOT this potential, each with what it actually is.
REFUSED_FUNCTIONS: Mapping[str, str] = {
    "gapsys": "OpenFE's default softcore (Gapsys et al. 2012) -- a different functional form",
    "beutler": "the Beutler et al. 1994 LJ softcore alone -- OpenFE's option, not Amber18 eqs. "
               "21.5-21.7, and it has no electrostatic counterpart",
    "openfe": "OpenFE's softcore, which is Gapsys by default and Beutler optionally; neither is "
              "Amber18",
    "smoothstep": "Amber20+ lambda scheduling with smoothstep functions (gti_lam_sch = 1) -- a "
                  "later AMBER variant, not Amber18",
    "amber20": "Amber20+ gti_lam_sch / gti_scale_beta variants -- not Amber18",
}


#: The two boundary 1-4 rules and the pmemd setting each one reproduces.
BOUNDARY_14_RULES: Mapping[str, str] = {
    "scaled": "gti_add_sc = 1 (pmemd 20+ default)",
    "unscaled": "gti_add_sc = 0 (Amber18 manual 21.1.5)",
}


#: The default boundary rule departs from the literal Amber18 text, so it was the user's decision;
#: they confirmed "scaled" on 2026-09-19, and every record says which rule ran and that it was. The
#: softcore FORM (`softcore_function`) and the boundary rule are two separate recorded facts:
#: nothing here calls their combination "amber18".
BOUNDARY_14_DEFAULT_STATUS = "scaled -- user-confirmed 2026-09-19"


class SoftcoreError(ValueError):
    """A softcore setting that is not the Amber18 potential, or is not a valid one."""


@dataclass(frozen=True)
class SoftcoreSettings:
    """`alchemical:` settings for the softcore potential. Defaults are the MD-tools defaults.

    `sc = False` selects ordinary linear mixing with NO softening, which is only defined when no
    particle appears or disappears; the Hamiltonian builder refuses it otherwise rather than
    softening behind the user's back.
    """

    sc: bool = True
    softcore_function: str = SOFTCORE_FUNCTION
    scalpha: float = 0.5
    scbeta: float = 12.0  # angstrom^2
    #: 1-4 exceptions between a softcore particle and a common one. "scaled": mixed between the end
    #: states like every other exception, so they vanish where the region is a dummy -- pmemd 20+
    #: `gti_add_sc = 1`, the AMBER default since Amber20, and what keeps a single-anchor dummy's
    #: partition function separable. "unscaled": present at full strength at every lambda -- the
    #: Amber18 manual 21.1.5 rule (`gti_add_sc = 0`), which the Amber20+ manual calls
    #: theoretically incorrect; kept so an Amber18 run can be reproduced. Pairs and exceptions
    #: INSIDE a region are unscaled under both.
    sc_boundary_14: str = "scaled"

    def __post_init__(self) -> None:
        if not isinstance(self.sc, bool):
            raise SoftcoreError(f"alchemical.sc must be true or false, got {self.sc!r}")
        name = str(self.softcore_function).strip().lower()
        if name in REFUSED_FUNCTIONS:
            raise SoftcoreError(
                f"alchemical.softcore_function {self.softcore_function!r} is refused: it is "
                f"{REFUSED_FUNCTIONS[name]}. MD-tools implements only {SOFTCORE_FUNCTION!r} "
                f"(Amber18 manual section 21.1.5, eqs. 21.5-21.7).")
        if name != SOFTCORE_FUNCTION:
            raise SoftcoreError(
                f"alchemical.softcore_function {self.softcore_function!r} is not known; the only "
                f"implemented function is {SOFTCORE_FUNCTION!r}.")
        if self.sc_boundary_14 not in BOUNDARY_14_RULES:
            raise SoftcoreError(f"alchemical.sc_boundary_14 must be one of "
                                f"{sorted(BOUNDARY_14_RULES)}, got {self.sc_boundary_14!r}")
        for key in ("scalpha", "scbeta"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SoftcoreError(f"alchemical.{key} must be a number, got {value!r}")
            if not value > 0.0 or value != value or value == float("inf"):
                raise SoftcoreError(f"alchemical.{key} must be a finite number > 0, got {value!r}")

    @property
    def scbeta_nm2(self) -> float:
        return float(self.scbeta) * ANGSTROM2_TO_NM2

    def record(self) -> dict[str, Any]:
        return {"sc": self.sc, "softcore_function": SOFTCORE_FUNCTION,
                "scalpha": float(self.scalpha), "scbeta_angstrom2": float(self.scbeta),
                "scbeta_nm2": self.scbeta_nm2, "sc_boundary_14": self.sc_boundary_14,
                "sc_boundary_14_amber": BOUNDARY_14_RULES[self.sc_boundary_14],
                "sc_boundary_14_default": BOUNDARY_14_DEFAULT_STATUS}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SoftcoreSettings":
        """Strict: unknown keys are refused, as every MD-tools configuration section is."""
        data = dict(data or {})
        known = {"sc", "softcore_function", "scalpha", "scbeta", "sc_boundary_14"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SoftcoreError(f"unknown alchemical softcore key(s) {unknown}; known: "
                                f"{sorted(known)}")
        return cls(**data)


def pair_expressions(*, region: str, kappa: float | None, scalpha: float,
                     scbeta_nm2: float) -> tuple[str, str]:
    """OpenMM energies of one softcore (region particle) x (common particle) pair: (elec, lj).

    * elec is a DELTA on top of what the region's end-state NonbondedForce already computes for
      the pair -- its ordinary direct-space term `w q q erfc(kappa r)/r`, and its reciprocal share
      -- so it adds `w q q erfc(kappa r) (1/sqrt(r^2 + beta l) - 1/r)`, leaving exactly pmemd's
      direct-space form. Without periodicity erfc is 1. Per-particle parameter: q.
    * lj is the whole softcore Lennard-Jones, eq. 21.5 (A) or 21.6 (B): the region particle
      carries epsilon 0 in every NonbondedForce. Per-particle parameters: sig, eps.

    `w` is the mixing weight (1 - lambda for the disappearing region A, lambda for the appearing
    region B) and `l` the softening argument (lambda for A, 1 - lambda for B), separately for the
    electrostatic and steric components.

    `kappa` is the Ewald splitting parameter in nm^-1, None without periodicity. It and the
    softcore constants are written into the expressions as NUMBERS, not global parameters: a
    Context parameter can be changed by anyone holding the Context, and these are part of the
    Hamiltonian's identity, not of its state.
    """
    if region == "A":
        w_el, l_el = "(1 - lambda_electrostatics)", "lambda_electrostatics"
        w_lj, l_lj = "(1 - lambda_sterics)", "lambda_sterics"
    elif region == "B":
        w_el, l_el = "lambda_electrostatics", "(1 - lambda_electrostatics)"
        w_lj, l_lj = "lambda_sterics", "(1 - lambda_sterics)"
    else:
        raise ValueError(f"region must be 'A' or 'B', got {region!r}")
    screen = f"erfc({float(kappa)!r}*r)" if kappa is not None else "1"
    a, b = f"{float(scalpha)!r}", f"{float(scbeta_nm2)!r}"
    elec = f"{w_el}*{ONE_4PI_EPS0!r}*q1*q2*{screen}*(1/sqrt(r^2 + {b}*{l_el}) - 1/r)"
    lj = (f"{w_lj}*4*eps*(1/({a}*{l_lj} + x6)^2 - 1/({a}*{l_lj} + x6));"
          " x6 = (r/sig)^6; sig = 0.5*(sig1 + sig2); eps = sqrt(eps1*eps2)")
    return elec, lj
