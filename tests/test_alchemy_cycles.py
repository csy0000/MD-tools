"""S4: Boresch restraints, the standard-state release term, and the thermodynamic cycles."""
from __future__ import annotations

import math

import numpy as np
import pytest

from md_tools.alchemy import cycles as cy
from md_tools.alchemy.restraints import (BoreschRestraint, RestraintError, STANDARD_VOLUME_NM3,
                                         _dihedral)
from md_tools.alchemy.samples import kt_kj_mol

T = 300.0


def _restraint(**over):
    kw = dict(receptor_atoms=(0, 1, 2), ligand_atoms=(3, 4, 5), r0_nm=0.5, theta_a0_rad=1.6,
              theta_b0_rad=1.3, phi_a0_rad=0.5, phi_b0_rad=-2.0, phi_c0_rad=2.9,
              k_r_kj_mol_nm2=4184.0, k_theta_a_kj_mol_rad2=41.84, k_theta_b_kj_mol_rad2=41.84,
              k_phi_a_kj_mol_rad2=41.84, k_phi_b_kj_mol_rad2=41.84, k_phi_c_kj_mol_rad2=41.84)
    kw.update(over)
    return BoreschRestraint(**kw)


# ---------------------------------------------------------------------- restraint energy
def test_openmm_force_equals_the_independent_numpy_energy_and_derivative():
    openmm = pytest.importorskip("openmm")
    r = _restraint()
    system = openmm.System()
    for _ in range(6):
        system.addParticle(12.0)
    system.addForce(r.openmm_force())
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(7)
    for lam in (0.0, 0.3, 1.0):
        for _ in range(20):
            x = rng.normal(0, 0.4, (6, 3))
            ctx.setPositions(x)
            ctx.setParameter("lambda_restraints", lam)
            st = ctx.getState(getEnergy=True, getParameterDerivatives=True)
            e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert e == pytest.approx(r.energy_kj_mol(x, lam), abs=1e-9, rel=1e-10)
            d = st.getEnergyParameterDerivatives()["lambda_restraints"]
            assert d == pytest.approx(r.energy_kj_mol(x, 1.0), abs=1e-9, rel=1e-10)


def test_dihedral_sign_is_openmms():
    openmm = pytest.importorskip("openmm")
    f = openmm.CustomCompoundBondForce(4, "dihedral(p1,p2,p3,p4)")
    f.addBond([0, 1, 2, 3], [])
    s = openmm.System()
    for _ in range(4):
        s.addParticle(1.0)
    s.addForce(f)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.normal(0, 0.3, (4, 3))
        ctx.setPositions(x)
        assert ctx.getState(getEnergy=True).getPotentialEnergy()._value == pytest.approx(
            _dihedral(*x), abs=1e-10)


@pytest.mark.parametrize("angle", ["theta_a0_rad", "theta_b0_rad"])
@pytest.mark.parametrize("value", [0.05, math.pi - 0.05])
def test_collinear_anchors_are_refused(angle, value):
    with pytest.raises(RestraintError, match="collinear"):
        _restraint(**{angle: value})


def test_record_round_trip_and_digest():
    r = _restraint()
    assert BoreschRestraint.from_record(r.to_record()) == r
    assert _restraint(k_phi_c_kj_mol_rad2=10.0).digest() != r.digest()


# ---------------------------------------------------------------------- the release term
def test_release_matches_boresch_closed_form_for_stiff_restraints():
    """With stiff restraints the Gaussian approximation is exact to high order."""
    r = _restraint(k_r_kj_mol_nm2=4.184e5, **{f"k_{n}_kj_mol_rad2": 4184.0 for n in
                                              ("theta_a", "theta_b", "phi_a", "phi_b", "phi_c")})
    rel = r.release_free_energy(T)
    assert rel["delta_g_release_kJ_mol"] < 0
    assert abs(rel["closed_form_discrepancy_kJ_mol"]) < 0.01


def test_release_quadrature_is_the_cartesian_and_rotational_integral():
    """Independent of the r^2 sin(theta) Jacobian: integrate exp(-bU) over the Cartesian position
    of L1 on a grid, and over uniformly random rigid-body orientations of the ligand by Monte
    Carlo, with every coordinate computed from positions by `BoreschRestraint.coordinates`."""
    rot = pytest.importorskip("scipy.spatial.transform")
    r = _restraint(k_r_kj_mol_nm2=800.0, **{f"k_{n}_kj_mol_rad2": 8.0 for n in
                                            ("theta_a", "theta_b", "phi_a", "phi_b", "phi_c")})
    kt = kt_kj_mol(T)
    b = 1.0 / kt
    P = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.3, 0.5, 0.1]])  # P1, P2, P3

    # position part: U_r + U_thetaA + U_phiA over a Cartesian grid for L1
    h = 0.012
    ax = np.arange(-1.0, 1.0 + h / 2, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    L1 = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    rr = np.linalg.norm(L1 - P[0], axis=1)
    u1, u2 = P[1] - P[0], L1 - P[0]
    ta = np.arccos(np.clip((u2 @ u1) / (np.linalg.norm(u1) * rr), -1, 1))
    b1, b2, b3 = P[1] - P[2], P[0] - P[1], L1 - P[0]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    pa = -np.arctan2(n2 @ m1, n2 @ n1)
    dpa = pa - r.phi_a0_rad
    dpa -= 2 * np.pi * np.floor((dpa + np.pi) / (2 * np.pi))
    upos = 0.5 * (r.k_r_kj_mol_nm2 * (rr - r.r0_nm) ** 2
                  + r.k_theta_a_kj_mol_rad2 * (ta - r.theta_a0_rad) ** 2
                  + r.k_phi_a_kj_mol_rad2 * dpa ** 2)
    z_pos = float(np.sum(np.exp(-b * upos))) * h ** 3

    # orientation part: L1 fixed, rigid ligand frame rotated uniformly (Haar measure, 8 pi^2)
    l1 = np.array([0.5, 0.0, 0.0])  # theta_A = 90 deg: phi_B is defined only off the P2-P1 axis
    body = np.array([[0.15, 0.0, 0.0], [0.2, 0.12, 0.0]])  # L2 - L1, L3 - L1
    n = 2_000_000
    R = rot.Rotation.random(n, random_state=11).as_matrix()
    L2 = l1 + R @ body[0]
    L3 = l1 + R @ body[1]
    n = len(L2)
    l1b = np.broadcast_to(l1, (n, 3))
    tb = _vangle(np.broadcast_to(P[0], (n, 3)), l1b, L2)
    pb = _vdihedral(np.broadcast_to(P[1], (n, 3)), np.broadcast_to(P[0], (n, 3)), l1b, L2)
    pc = _vdihedral(np.broadcast_to(P[0], (n, 3)), l1b, L2, L3)
    # the vectorised geometry agrees with the class's own on a few frames
    x = np.zeros((6, 3))
    x[0:3], x[3] = P, l1
    for i in range(0, n, n // 5):
        x[4], x[5] = L2[i], L3[i]
        q = r.coordinates(x)
        assert (q["theta_b"], q["phi_b"], q["phi_c"]) == pytest.approx((tb[i], pb[i], pc[i]))
    uor = 0.5 * (r.k_theta_b_kj_mol_rad2 * (tb - r.theta_b0_rad) ** 2
                 + r.k_phi_b_kj_mol_rad2 * _wrap(pb - r.phi_b0_rad) ** 2
                 + r.k_phi_c_kj_mol_rad2 * _wrap(pc - r.phi_c0_rad) ** 2)
    w = np.exp(-b * uor)
    z_or = 8 * np.pi ** 2 * float(w.mean())
    z_or_se = 8 * np.pi ** 2 * float(w.std() / math.sqrt(n))
    independent = -kt * math.log(8 * np.pi ** 2 * STANDARD_VOLUME_NM3 / (z_pos * z_or))
    se = kt * z_or_se / z_or
    rel = r.release_free_energy(T)
    assert rel["delta_g_release_kJ_mol"] == pytest.approx(independent, abs=max(4 * se, 0.01))
    # and the soft restraint is where the closed form is visibly wrong
    assert abs(rel["closed_form_discrepancy_kJ_mol"]) > 0.3


def _wrap(d):
    return d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))


def _vangle(a, b, c):
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.arccos(np.clip(cos, -1, 1))


def _vdihedral(a, b, c, d):
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=1)[:, None])
    return -np.arctan2(np.sum(m1 * n2, axis=1), np.sum(n1 * n2, axis=1))


# ---------------------------------------------------------------------- cycles
def _leg(name, env, a, b, dg, sigma=0.2, **kw):
    return cy.Leg(name, env, a, b, dg, sigma, T, "MBAR", kw.pop("scheme", "decouple"), **kw)


def test_absolute_hydration_sign():
    out = cy.absolute_hydration(vacuum=_leg("v", "vacuum", "coupled", "decoupled", 10.0),
                                solvent=_leg("s", "solvent", "coupled", "decoupled", 30.0))
    assert out["delta_g_kJ_mol"] == pytest.approx(-20.0)
    assert out["sigma_kJ_mol"] == pytest.approx(math.sqrt(0.08))


def test_relative_cycles_signs_and_orientation():
    rh = cy.relative_hydration(vacuum=_leg("v", "vacuum", "A", "B", 5.0),
                               solvent=_leg("s", "solvent", "A", "B", 2.0))
    assert rh["delta_g_kJ_mol"] == pytest.approx(-3.0)
    rb = cy.relative_binding(solvent=_leg("s", "solvent", "A", "B", 2.0),
                             complex=_leg("c", "complex", "A", "B", 6.0))
    assert rb["delta_g_kJ_mol"] == pytest.approx(4.0)
    with pytest.raises(cy.CycleError, match="Reverse it explicitly"):
        cy.relative_binding(solvent=_leg("s", "solvent", "A", "B", 2.0),
                            complex=_leg("c", "complex", "B", "A", -6.0))
    ok = cy.relative_binding(solvent=_leg("s", "solvent", "A", "B", 2.0),
                             complex=cy.reversed_leg(_leg("c", "complex", "B", "A", -6.0)))
    assert ok["delta_g_kJ_mol"] == pytest.approx(4.0)


def test_leg_in_the_wrong_environment_is_refused():
    with pytest.raises(cy.CycleError, match="needs it in vacuum"):
        cy.absolute_hydration(vacuum=_leg("v", "solvent", "coupled", "decoupled", 1.0),
                              solvent=_leg("s", "solvent", "coupled", "decoupled", 1.0))


def test_mixed_schemes_are_refused():
    with pytest.raises(cy.CycleError, match="alchemical schemes"):
        cy.absolute_hydration(vacuum=_leg("v", "vacuum", "coupled", "decoupled", 1.0,
                                          scheme="annihilate"),
                              solvent=_leg("s", "solvent", "coupled", "decoupled", 1.0))


def _binding_legs(r):
    d = r.digest()
    return dict(
        solvent=_leg("s", "solvent", "coupled", "decoupled", 40.0),
        restraint_attach=_leg("r", "complex", "unrestrained", "restrained", 3.0,
                              restraint_digest=d),
        complex=_leg("c", "complex", "coupled", "decoupled", 70.0, restraint_digest=d),
        release=r.release_free_energy(T))


def test_absolute_binding_includes_the_standard_state_term():
    r = _restraint()
    legs = _binding_legs(r)
    out = cy.absolute_binding(**legs)
    rel = legs["release"]["delta_g_release_kJ_mol"]
    assert out["delta_g_kJ_mol"] == pytest.approx(40.0 - 3.0 - 70.0 - rel)
    assert out["standard_state_included"] is True
    assert any(t.get("standard_volume_nm3") == pytest.approx(STANDARD_VOLUME_NM3)
               for t in out["terms"])


def test_absolute_binding_without_standard_state_is_refused():
    legs = _binding_legs(_restraint())
    legs["release"] = None
    with pytest.raises(cy.CycleError, match="not a standard binding free energy"):
        cy.absolute_binding(**legs)


def test_absolute_binding_without_restraint_leg_is_refused():
    legs = _binding_legs(_restraint())
    legs["restraint_attach"] = None
    with pytest.raises(cy.CycleError, match="restraint attachment leg is missing"):
        cy.absolute_binding(**legs)


def test_release_for_a_different_restraint_is_refused():
    legs = _binding_legs(_restraint())
    legs["release"] = _restraint(k_r_kj_mol_nm2=1000.0).release_free_energy(T)
    with pytest.raises(cy.CycleError, match="restraint digests disagree"):
        cy.absolute_binding(**legs)


def test_legs_at_different_temperatures_are_refused():
    v = _leg("v", "vacuum", "coupled", "decoupled", 1.0)
    s = cy.Leg("s", "solvent", "coupled", "decoupled", 2.0, 0.1, 310.0, "MBAR", "decouple")
    with pytest.raises(cy.CycleError, match="one cycle, one T"):
        cy.absolute_hydration(vacuum=v, solvent=s)


def test_restraint_in_a_periodic_box_uses_the_minimum_image():
    """Anchors wrapped into different images: the periodic force sees the true 0.5 nm, a
    non-periodic one would see ~2.5 nm."""
    openmm = pytest.importorskip("openmm")
    r = _restraint()
    x = np.array([[0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.4, 0.6, 0.2],
                  [2.9, 0.1, 0.1], [2.9, 0.1, 0.25], [2.9, 0.2, 0.3]])   # L1 at -0.2 nm, wrapped
    s = openmm.System()
    s.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    for _ in range(6):
        s.addParticle(12.0)
    s.addForce(r.openmm_force(periodic=True))
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(x)
    unwrapped = x.copy()
    unwrapped[3:, 0] -= 3.0
    e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    assert e == pytest.approx(r.energy_kj_mol(unwrapped), rel=1e-9)
    assert e != pytest.approx(r.energy_kj_mol(x), rel=1e-3)
