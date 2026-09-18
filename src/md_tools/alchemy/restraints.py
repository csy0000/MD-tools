"""Boresch orientational restraints and the standard-state correction for absolute binding.

SIX DEGREES OF FREEDOM (Boresch, Tettinger, Leitgeb and Karplus 2003, J Phys Chem B 107:9535)

    Three receptor anchors P1, P2, P3 and three ligand anchors L1, L2, L3:

        r        |P1 - L1|
        theta_A  angle P2-P1-L1          theta_B  angle P1-L1-L2
        phi_A    dihedral P3-P2-P1-L1    phi_B    dihedral P2-P1-L1-L2
        phi_C    dihedral P1-L1-L2-L3

    U = lambda_restraints * sum_i 1/2 K_i (q_i - q_i0)^2, with dihedral differences wrapped into
    (-pi, pi]. Units: nm and kJ/mol/nm^2 for r; radians and kJ/mol/rad^2 for angles.

THE RELEASE TERM

    For the DECOUPLED ligand, restrained, the free energy of releasing the restraints into a
    standard-state volume V0 (1 M, V0 = 1.66054 nm^3) is

        dG_release = G(free at V0) - G(restrained) = -kT ln( 8 pi^2 V0 / Z_r )

        Z_r = int r^2 exp(-b U_r) dr * int sin(tA) exp(-b U_tA) dtA * int sin(tB) exp(-b U_tB) dtB
              * prod_{A,B,C} int_{-pi}^{pi} exp(-b U_phi) dphi

    The integral separates because the six coordinates are independent for a non-interacting
    ligand. It is evaluated here by one-dimensional quadrature -- exact to quadrature precision,
    with no harmonic approximation -- and Boresch's closed form (their eq. 32, the Gaussian
    approximation) is computed alongside:

        dG_release ~ -kT ln[ 8 pi^2 V0 sqrt(K_r K_tA K_tB K_pA K_pB K_pC)
                             / ( r0^2 sin tA0 sin tB0 (2 pi kT)^3 ) ]

    The quadrature is the value used; the difference is reported. They disagree when a restraint
    is soft or an anchor angle is near 0 or pi, which is exactly when the closed form is wrong.

    dG_release is negative (the ligand gains the translational and rotational freedom of V0).
    `md_tools.alchemy.cycles` subtracts it with its sign; it is never folded in unnamed.

COLLINEAR ANCHORS ARE REFUSED

    When theta_A or theta_B approaches 0 or pi the dihedrals that share its bond are undefined and
    the restraint force diverges. An equilibrium angle within `MIN_ANCHOR_ANGLE_DEG` of either is
    refused at construction, before a force exists.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from md_tools.alchemy.samples import kt_kj_mol

BORESCH_SCHEMA = "md-tools-boresch-restraint/1"
#: 1 L / (N_A * 1 mol) in nm^3: the volume per molecule at 1 mol/L.
STANDARD_VOLUME_NM3 = 1.0e24 / 6.02214076e23
STANDARD_CONCENTRATION = "1 mol/L"
MIN_ANCHOR_ANGLE_DEG = 10.0
RESTRAINT_PARAMETER = "lambda_restraints"

_ANGLES = ("theta_a", "theta_b")
_DIHEDRALS = ("phi_a", "phi_b", "phi_c")


class RestraintError(ValueError):
    pass


@dataclass(frozen=True)
class BoreschRestraint:
    """Anchors are atom indices in the COMBINED topology (receptor P1..P3, ligand L1..L3)."""

    receptor_atoms: tuple[int, int, int]
    ligand_atoms: tuple[int, int, int]
    r0_nm: float
    theta_a0_rad: float
    theta_b0_rad: float
    phi_a0_rad: float
    phi_b0_rad: float
    phi_c0_rad: float
    k_r_kj_mol_nm2: float
    k_theta_a_kj_mol_rad2: float
    k_theta_b_kj_mol_rad2: float
    k_phi_a_kj_mol_rad2: float
    k_phi_b_kj_mol_rad2: float
    k_phi_c_kj_mol_rad2: float

    def __post_init__(self):
        atoms = tuple(self.receptor_atoms) + tuple(self.ligand_atoms)
        if len(atoms) != 6 or len(set(atoms)) != 6 or min(atoms) < 0:
            raise RestraintError(f"a Boresch restraint needs six distinct atoms; got {atoms}")
        if not self.r0_nm > 0:
            raise RestraintError(f"r0 = {self.r0_nm} nm")
        for name in _ANGLES:
            t = getattr(self, f"{name}0_rad")
            edge = math.degrees(min(t, math.pi - t))
            if not (0 < t < math.pi) or edge < MIN_ANCHOR_ANGLE_DEG:
                raise RestraintError(
                    f"{name}0 = {math.degrees(t):.1f} deg is within {MIN_ANCHOR_ANGLE_DEG} deg of "
                    f"collinear: the dihedrals sharing that bond are undefined there. Choose "
                    f"other anchors")
        for name in _DIHEDRALS:
            t = getattr(self, f"{name}0_rad")
            if not (-math.pi <= t <= math.pi):
                raise RestraintError(f"{name}0 = {t} rad outside [-pi, pi]")
        for k in ("k_r_kj_mol_nm2",) + tuple(f"k_{n}_kj_mol_rad2" for n in _ANGLES + _DIHEDRALS):
            if not getattr(self, k) > 0:
                raise RestraintError(f"{k} = {getattr(self, k)}: every force constant is positive")

    # ------------------------------------------------------------------ record
    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["receptor_atoms"] = list(self.receptor_atoms)
        d["ligand_atoms"] = list(self.ligand_atoms)
        return {"schema": BORESCH_SCHEMA, "parameter": RESTRAINT_PARAMETER, **d}

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> "BoreschRestraint":
        if r.get("schema") != BORESCH_SCHEMA:
            raise RestraintError(f"restraint schema {r.get('schema')!r}")
        d = {k: v for k, v in r.items() if k not in ("schema", "parameter")}
        d["receptor_atoms"] = tuple(d["receptor_atoms"])
        d["ligand_atoms"] = tuple(d["ligand_atoms"])
        return cls(**d)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_record(), sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------ energy
    def coordinates(self, positions_nm: np.ndarray) -> dict[str, float]:
        """The six restrained coordinates from positions (nm), independently of OpenMM."""
        x = np.asarray(positions_nm, dtype=float)
        p1, p2, p3 = (x[i] for i in self.receptor_atoms)
        l1, l2, l3 = (x[i] for i in self.ligand_atoms)
        return {"r": float(np.linalg.norm(l1 - p1)),
                "theta_a": _angle(p2, p1, l1), "theta_b": _angle(p1, l1, l2),
                "phi_a": _dihedral(p3, p2, p1, l1), "phi_b": _dihedral(p2, p1, l1, l2),
                "phi_c": _dihedral(p1, l1, l2, l3)}

    def energy_kj_mol(self, positions_nm: np.ndarray, lam: float = 1.0) -> float:
        q = self.coordinates(positions_nm)
        e = 0.5 * self.k_r_kj_mol_nm2 * (q["r"] - self.r0_nm) ** 2
        for n in _ANGLES:
            e += 0.5 * getattr(self, f"k_{n}_kj_mol_rad2") * (q[n] - getattr(self, f"{n}0_rad")) ** 2
        for n in _DIHEDRALS:
            d = _wrap(q[n] - getattr(self, f"{n}0_rad"))
            e += 0.5 * getattr(self, f"k_{n}_kj_mol_rad2") * d * d
        return lam * e

    def openmm_force(self, *, periodic: bool = False):
        """A CustomCompoundBondForce whose energy is `lambda_restraints * U`.

        `periodic` must be the System's own: in a periodic box the receptor and ligand anchors
        can be wrapped into different images, and a non-periodic restraint then measures the
        distance across the box.

        The energy is linear in `lambda_restraints`, so its parameter derivative -- requested
        here -- is the complete dU/d lambda_restraints for this force: U itself.
        """
        import openmm

        expr = ("lambda_restraints * 0.5 * (K_r*(distance(p1,p4)-r0)^2"
                " + K_tA*(angle(p2,p1,p4)-tA0)^2 + K_tB*(angle(p1,p4,p5)-tB0)^2"
                " + K_pA*dA^2 + K_pB*dB^2 + K_pC*dC^2);"
                " dA = dp - 6.283185307179586*floor((dp + 3.141592653589793)/6.283185307179586);"
                " dp = dihedral(p3,p2,p1,p4) - pA0;"
                " dB = dq - 6.283185307179586*floor((dq + 3.141592653589793)/6.283185307179586);"
                " dq = dihedral(p2,p1,p4,p5) - pB0;"
                " dC = dr - 6.283185307179586*floor((dr + 3.141592653589793)/6.283185307179586);"
                " dr = dihedral(p1,p4,p5,p6) - pC0")
        f = openmm.CustomCompoundBondForce(6, expr)
        f.addGlobalParameter(RESTRAINT_PARAMETER, 1.0)
        f.addEnergyParameterDerivative(RESTRAINT_PARAMETER)
        names = ["K_r", "r0", "K_tA", "tA0", "K_tB", "tB0", "K_pA", "pA0", "K_pB", "pB0",
                 "K_pC", "pC0"]
        for n in names:
            f.addPerBondParameter(n)
        f.addBond(list(self.receptor_atoms) + list(self.ligand_atoms),
                  [self.k_r_kj_mol_nm2, self.r0_nm,
                   self.k_theta_a_kj_mol_rad2, self.theta_a0_rad,
                   self.k_theta_b_kj_mol_rad2, self.theta_b0_rad,
                   self.k_phi_a_kj_mol_rad2, self.phi_a0_rad,
                   self.k_phi_b_kj_mol_rad2, self.phi_b0_rad,
                   self.k_phi_c_kj_mol_rad2, self.phi_c0_rad])
        f.setUsesPeriodicBoundaryConditions(bool(periodic))
        f.setName("BoreschRestraint")
        return f

    # ------------------------------------------------------------------ free energy
    def release_free_energy(self, temperature_k: float,
                            standard_volume_nm3: float = STANDARD_VOLUME_NM3) -> dict[str, Any]:
        """dG_release (kJ/mol) for the decoupled ligand: restrained -> free at V0."""
        from scipy.integrate import quad

        kt = kt_kj_mol(temperature_k)
        b = 1.0 / kt

        def gauss(k, x0):
            return lambda x: math.exp(-0.5 * b * k * (x - x0) ** 2)

        kr, r0 = self.k_r_kj_mol_nm2, self.r0_nm
        width = 12.0 * math.sqrt(kt / kr)
        z_r, _ = quad(lambda r: r * r * gauss(kr, r0)(r), max(0.0, r0 - width), r0 + width,
                      epsabs=0, epsrel=1e-12, limit=200)
        z = z_r
        for n in _ANGLES:
            k, t0 = getattr(self, f"k_{n}_kj_mol_rad2"), getattr(self, f"{n}0_rad")
            zi, _ = quad(lambda t: math.sin(t) * gauss(k, t0)(t), 0.0, math.pi, points=[t0],
                         epsabs=0, epsrel=1e-12, limit=200)
            z *= zi
        for n in _DIHEDRALS:
            k = getattr(self, f"k_{n}_kj_mol_rad2")
            zi, _ = quad(lambda d: math.exp(-0.5 * b * k * d * d), -math.pi, math.pi,
                         points=[0.0], epsabs=0, epsrel=1e-12, limit=200)
            z *= zi
        numerical = -kt * math.log(8.0 * math.pi ** 2 * standard_volume_nm3 / z)
        k_prod = (self.k_r_kj_mol_nm2 * self.k_theta_a_kj_mol_rad2 * self.k_theta_b_kj_mol_rad2
                  * self.k_phi_a_kj_mol_rad2 * self.k_phi_b_kj_mol_rad2 * self.k_phi_c_kj_mol_rad2)
        analytic = -kt * math.log(
            8.0 * math.pi ** 2 * standard_volume_nm3 * math.sqrt(k_prod)
            / (r0 ** 2 * math.sin(self.theta_a0_rad) * math.sin(self.theta_b0_rad)
               * (2.0 * math.pi * kt) ** 3))
        return {"delta_g_release_kJ_mol": numerical,
                "method": "separable one-dimensional quadrature (exact for the restraint)",
                "boresch_closed_form_kJ_mol": analytic,
                "closed_form_discrepancy_kJ_mol": analytic - numerical,
                "standard_volume_nm3": standard_volume_nm3,
                "standard_concentration": STANDARD_CONCENTRATION,
                "temperature_k": temperature_k, "restraint_digest": self.digest(),
                "sign": "G(free, V0) - G(restrained); decoupled ligand"}


def _angle(a, b, c) -> float:
    u, v = a - b, c - b
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(1.0, cos)))


def _dihedral(a, b, c, d) -> float:
    """Signed dihedral in (-pi, pi] with OpenMM's sign convention (checked against
    `CustomCompoundBondForce`'s `dihedral()` by test, because the two common conventions differ
    only in sign and a restraint built on the other would hold every phi at its mirror image)."""
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return -math.atan2(float(np.dot(m1, n2)), float(np.dot(n1, n2)))


def _wrap(d: float) -> float:
    return d - 2.0 * math.pi * math.floor((d + math.pi) / (2.0 * math.pi))
