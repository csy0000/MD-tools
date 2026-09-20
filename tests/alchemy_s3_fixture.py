"""S3's miniature end-state pair, and an independent numpy reference for the Amber18 Hamiltonian.

THE PAIR (hybrid indices), one atom substituted: an ethane-like molecule whose H on C0 (A) becomes
an F (B). Neutral in both end states; the common core's charges change (C0, H0a, H0b) and C0's
Lennard-Jones changes, so the linear mixing of common-core terms is exercised as well as softcore.

    0 C0   1 C1   2 H0a  3 H0b  (on C0)   4 H1a  5 H1b  6 H1c  (on C1)
    7 HA   A-only, on C0 (disappears)     8 FB   B-only, on C0 (appears)

followed by an environment: a Cl- close to the substituted site, a Na+, a neutral LJ probe, and
(periodic variant) TIP3P-like waters with flexible bonds. The end states follow the topology-plan
dummy convention of the topology plan (contract section 4): in System A the B-only particles have
charge 0 and epsilon 0 and every exception joining them to anything else is zero, while the group's
OWN exceptions keep their physical values and its non-excluded internal pairs sit in a
`UniqueGroupInternalNonbonded` CustomBondForce (zero at the physical end). System B mirrors that.
Bonded terms touching a unique particle are identical in both, and HA / FB (both on C0) are 1-3 to
each other, hence excluded in both.

This is S3's own hand-written fixture, kept beside S2's package fixtures because it is small enough
for an independent numpy reference. It is not a topology builder.

THE REFERENCE is written from the Amber18 rules, not from `md_tools.alchemy`: explicit pair loops,
plain Ewald with an explicit k-vector sum (not PME), and bonded energies from the force parameters.
It reads only the two end-state Systems' parameters and the positions.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import erf, erfc

K_COULOMB = 138.93545764438198   # measured in test_alchemy_softcore_pair.test_coulomb_constant_is_openmms
N_MOLECULE = 9
A_ONLY = {7}
B_ONLY = {8}

_Q = {  # charges: (A, B)
    0: (-0.27, 0.12), 1: (-0.27, -0.27), 2: (0.09, 0.06), 3: (0.09, 0.06),
    4: (0.09, 0.09), 5: (0.09, 0.09), 6: (0.09, 0.09), 7: (0.09, 0.0), 8: (0.0, -0.24),
}
_LJ = {  # (sigma nm, epsilon kJ/mol): (A, B)
    0: ((0.34, 0.457), (0.33, 0.43)), 1: ((0.34, 0.457), (0.34, 0.457)),
    **{i: ((0.265, 0.0657), (0.265, 0.0657)) for i in (2, 3, 4, 5, 6)},
    7: ((0.265, 0.0657), (0.265, 0.0)), 8: ((0.30, 0.0), (0.30, 0.255)),
}
_BONDS = [((0, 1), (0.153, 2.5e5), (0.153, 2.5e5)),
          ((0, 2), (0.109, 2.8e5), (0.110, 3.0e5)),   # a common-core bond that changes
          ((0, 3), (0.109, 2.8e5), (0.109, 2.8e5)),
          ((1, 4), (0.109, 2.8e5), (0.109, 2.8e5)), ((1, 5), (0.109, 2.8e5), (0.109, 2.8e5)),
          ((1, 6), (0.109, 2.8e5), (0.109, 2.8e5)),
          ((0, 7), (0.109, 2.8e5), (0.109, 2.8e5)),   # touches A-only: identical
          ((0, 8), (0.138, 3.0e5), (0.138, 3.0e5))]   # touches B-only: identical
_T = math.radians(109.5)
_ANGLES = [((1, 0, 2), (_T, 300.0), (_T, 360.0)),    # changes
           ((1, 0, 3), (_T, 300.0), (_T, 300.0)), ((2, 0, 3), (_T, 280.0), (_T, 280.0)),
           ((1, 0, 7), (_T, 300.0), (_T, 300.0)), ((2, 0, 7), (_T, 280.0), (_T, 280.0)),
           ((3, 0, 7), (_T, 280.0), (_T, 280.0)),
           ((1, 0, 8), (_T, 400.0), (_T, 400.0)), ((2, 0, 8), (_T, 350.0), (_T, 350.0)),
           ((3, 0, 8), (_T, 350.0), (_T, 350.0)),
           ((0, 1, 4), (_T, 300.0), (_T, 300.0)), ((0, 1, 5), (_T, 300.0), (_T, 300.0)),
           ((0, 1, 6), (_T, 300.0), (_T, 300.0)), ((4, 1, 5), (_T, 280.0), (_T, 280.0)),
           ((4, 1, 6), (_T, 280.0), (_T, 280.0)), ((5, 1, 6), (_T, 280.0), (_T, 280.0))]
_TORSIONS = []
for _h1 in (4, 5, 6):
    for _h0 in (2, 3, 7, 8):
        a = (3, 0.0, 0.6)
        b = (3, 0.0, 0.8) if (_h1, _h0) == (4, 2) else a    # one common-core torsion changes
        _TORSIONS.append(((_h1, 1, 0, _h0), a, b))

_ENV = [  # (charge, sigma, epsilon)
    (-1.0, 0.44, 0.418),   # Cl-, placed close to HA/FB
    (1.0, 0.333, 0.0116),  # Na+
    (0.0, 0.35, 0.6),      # neutral probe
]
_WATER = ((-0.834, 0.315, 0.636), (0.417, 0.1, 0.0), (0.417, 0.1, 0.0))
#: The `vsite` variant's 4-site water (TIP4P-Ew-like): O, H, H, and a massless M site built by a
#: ThreeParticleAverageSite, which carries the negative charge -- an environment virtual site, as
#: OPC's are in S2's complex fixture.
_WATER4 = ((0.0, 0.316435, 0.680946), (0.52422, 0.1, 0.0), (0.52422, 0.1, 0.0))
_M_SITE = (-1.04844, 0.1, 0.0)
_M_WEIGHTS = (0.786646558, 0.106676721, 0.106676721)


def molecule_positions() -> np.ndarray:
    c0, c1 = np.zeros(3), np.array([0.153, 0.0, 0.0])
    t = [np.array(v) / math.sqrt(3) for v in ((-1, 1, 1), (-1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    u = [np.array(v) / math.sqrt(3) for v in ((1, 1, 1), (1, -1, -1), (1, 1, -1))]
    h0a, h0b = c0 + 0.109 * t[1], c0 + 0.109 * t[2]
    ha = c0 + 0.109 * t[0]
    fb = c0 + 0.138 * (0.95 * t[0] + 0.31 * t[3]) / np.linalg.norm(0.95 * t[0] + 0.31 * t[3])
    h1 = [c1 + 0.109 * v for v in u]
    return np.array([c0, c1, h0a, h0b, *h1, ha, fb])


def _water_sites(box: float, avoid: np.ndarray, count: int, rng) -> list[np.ndarray]:
    sites = []
    grid = np.arange(0.2, box, 0.62)
    for x, y, z in itertools.product(grid, grid, grid):
        o = np.array([x, y, z]) + rng.uniform(-0.05, 0.05, 3)
        d = avoid - o
        d -= box * np.round(d / box)
        if np.min(np.linalg.norm(d, axis=1)) < 0.45:
            continue
        sites.append(o)
        if len(sites) == count:
            break
    return sites


#: The `tail` variant's appearing group: FB grows into a five-atom chain FB-T1-T2-T3-T4, so the
#: B region has internal 1-4 exceptions (FB-T3, T1-T4) and a non-excluded internal pair (FB-T4).
#: Charges (B end state) keep the net charge: -0.30 +0.10 -0.20 +0.25 -0.09 = -0.24 = F's.
_TAIL_Q = (-0.30, 0.10, -0.20, 0.25, -0.09)
_TAIL_LJ = (0.30, 0.30)


def _tail_positions(mol):
    c0, fb = mol[0], mol[8]
    axis = (fb - c0) / np.linalg.norm(fb - c0)
    side = np.cross(axis, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    out, last = [], fb
    for k in range(4):
        last = last + 0.15 * (0.82 * axis + (0.57 if k % 2 == 0 else -0.57) * side)
        out.append(last)
    return out


def build(periodic: bool, *, charges: bool = True, lj: bool = True, bonded: bool = True,
          dispersion: bool = False, n_water: int = 30, box: float = 2.4, cutoff: float = 0.9,
          ewald_tolerance: float = 1e-6, tail: bool = False, vsite: bool = False):
    """(system_a, system_b, a_only, b_only, positions). Component switches zero a term family."""
    import openmm
    rng = np.random.default_rng(20260919)
    mol = molecule_positions()
    offset = np.array([box / 2, box / 2, box / 2]) if periodic else np.zeros(3)
    mol = mol + offset
    site = mol[7]
    env_pos = [site + np.array([0.0, 0.26, 0.24]), mol[1] + np.array([0.0, -0.5, 0.35]),
               mol[1] + np.array([0.42, 0.0, 0.0])]
    positions = [*mol, *env_pos]
    masses = [12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 19.0, 35.45, 22.99, 40.0]
    bonds, angles, torsions = list(_BONDS), list(_ANGLES), list(_TORSIONS)
    b_only = set(B_ONLY)
    tail_q = {}
    if tail:
        first = len(positions)
        chain = [8, *range(first, first + 4)]
        positions += _tail_positions(mol)
        masses += [12.0] * 4
        b_only |= set(chain[1:])
        for k, idx in enumerate(chain):
            tail_q[idx] = _TAIL_Q[k]
        same = lambda p: (p, p)  # noqa: E731  -- every term touching the tail is identical
        for i, j in zip(chain, chain[1:]):
            bonds.append(((i, j), *same((0.15, 2.5e5))))
        for i, j, k in [(0, 8, chain[1]), *zip(chain, chain[1:], chain[2:])]:
            angles.append(((i, j, k), *same((math.radians(112.0), 450.0))))
        for i, j, k, m in [(1, 0, 8, chain[1]), (0, 8, chain[1], chain[2]),
                           *zip(chain, chain[1:], chain[2:], chain[3:])]:
            torsions.append(((i, j, k, m), *same((3, 0.0, 0.7))))
    waters = []
    if periodic:
        for o in _water_sites(box, np.array(positions), n_water, rng):
            waters.append(len(positions))
            positions += [o, o + np.array([0.09572, 0.0, 0.0]),
                          o + 0.09572 * np.array([math.cos(1.824), math.sin(1.824), 0.0])]
            masses += [16.0, 1.0, 1.0]
    m_sites = []
    if vsite:
        for w in waters:
            m_sites.append((len(positions), w))
            positions.append(sum(wt * positions[w + k] for k, wt in enumerate(_M_WEIGHTS)))
            masses.append(0.0)
    positions = np.array(positions)
    n = len(positions)

    graph = {i: set() for i in range(n)}
    for (i, j), *_ in bonds:
        graph[i].add(j); graph[j].add(i)
    for w in waters:
        for h in (w + 1, w + 2):
            graph[w].add(h); graph[h].add(w)
    for m, w in m_sites:
        graph[w].add(m); graph[m].add(w)
    pairs = {}
    for i in range(n):
        for j in graph[i]:
            pairs[min(i, j), max(i, j)] = 2
            for k in graph[j]:
                if k != i:
                    pairs.setdefault((min(i, k), max(i, k)), 3)
                    for m in graph[k]:
                        if m not in (i, j):
                            pairs.setdefault((min(i, m), max(i, m)), 4)
    for i in A_ONLY:                      # A-only x B-only: excluded in both, as the plan does it
        for j in b_only:
            pairs.setdefault((min(i, j), max(i, j)), 0)

    def particle_params(end: int):
        params = []
        for i in range(N_MOLECULE):
            q = tail_q.get(i, _Q[i][1]) if end == 1 and i in tail_q else _Q[i][end]
            sg, ep = _LJ[i][end]
            params.append((q, sg, ep))
        params += list(_ENV)
        for idx in range(len(params), len(params) + (4 if tail else 0)):
            params.append((tail_q[idx], *_TAIL_LJ) if end == 1 else (0.0, _TAIL_LJ[0], 0.0))
        for _ in waters:
            params += list(_WATER4 if vsite else _WATER)
        params += [_M_SITE] * len(m_sites)
        return params

    physical = {0: particle_params(0), 1: particle_params(1)}   # region A is physical at end 0
    region = {0: set(A_ONLY), 1: set(b_only)}
    internal_pairs = []                                          # (i, j, the end it is physical at)
    for end, members in region.items():
        members = sorted(members)
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if (i, j) not in pairs:
                    internal_pairs.append((i, j, end))

    def system(end: int):
        s = openmm.System()
        for m in masses:
            s.addParticle(m)
        nb = openmm.NonbondedForce()
        if periodic:
            s.setDefaultPeriodicBoxVectors([box, 0, 0], [0, box, 0], [0, 0, box])
            nb.setNonbondedMethod(openmm.NonbondedForce.PME)
            nb.setCutoffDistance(cutoff)
            nb.setEwaldErrorTolerance(ewald_tolerance)
            nb.setUseDispersionCorrection(dispersion)
        else:
            nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        dummy = b_only if end == 0 else A_ONLY
        params = physical[end]
        for q, sg, ep in params:
            nb.addParticle(q if charges else 0.0, sg, ep if lj else 0.0)
        for (i, j), kind in sorted(pairs.items()):
            # contract section 4: a unique group's OWN 1-4s keep their physical values at its dummy
            # end; every other exception touching a dummy is zero
            inside_dummy = i in dummy and j in dummy
            if kind < 4 or ((i in dummy or j in dummy) and not inside_dummy):
                nb.addException(i, j, 0.0, 0.1, 0.0)
            else:
                source = physical[1 - end] if inside_dummy else params
                qi, si, ei = source[i]
                qj, sj, ej = source[j]
                nb.addException(i, j, (qi * qj / 1.2) if charges else 0.0, 0.5 * (si + sj),
                                (0.5 * math.sqrt(ei * ej)) if lj else 0.0)
        s.addForce(nb)
        if internal_pairs:
            # ...and its non-excluded internal pairs live here at its dummy end, as S2's plan does
            from md_tools.alchemy.topology import COULOMB_CONSTANT, INTERNAL_FORCE_NAME
            f = openmm.CustomBondForce(
                f"{COULOMB_CONSTANT!r}*chargeprod/r + 4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
            for name in ("chargeprod", "sigma", "epsilon"):
                f.addPerBondParameter(name)
            f.setName(INTERNAL_FORCE_NAME)
            f.setUsesPeriodicBoundaryConditions(periodic)
            for i, j, phys_end in internal_pairs:
                qi, si, ei = physical[phys_end][i]
                qj, sj, ej = physical[phys_end][j]
                at_dummy = end != phys_end
                f.addBond(i, j, [qi * qj if (at_dummy and charges) else 0.0, 0.5 * (si + sj),
                                 math.sqrt(ei * ej) if (at_dummy and lj) else 0.0])
            s.addForce(f)
        hb, ha, ht = (openmm.HarmonicBondForce(), openmm.HarmonicAngleForce(),
                      openmm.PeriodicTorsionForce())
        for (i, j), pa, pb in bonds:
            r0, k = (pa, pb)[end]
            hb.addBond(i, j, r0, k if bonded else 0.0)
        for w in waters:
            hb.addBond(w, w + 1, 0.09572, 4.6e5 if bonded else 0.0)
            hb.addBond(w, w + 2, 0.09572, 4.6e5 if bonded else 0.0)
            ha.addAngle(w + 1, w, w + 2, 1.824, 836.8 if bonded else 0.0)
        for (i, j, k), pa, pb in angles:
            t0, kk = (pa, pb)[end]
            ha.addAngle(i, j, k, t0, kk if bonded else 0.0)
        for (i, j, k, m), pa, pb in torsions:
            per, ph, kk = (pa, pb)[end]
            ht.addTorsion(i, j, k, m, per, ph, kk if bonded else 0.0)
        for f in (hb, ha, ht):
            f.setUsesPeriodicBoundaryConditions(periodic)
            s.addForce(f)
        for m, w in m_sites:
            s.setVirtualSite(m, openmm.ThreeParticleAverageSite(w, w + 1, w + 2, *_M_WEIGHTS))
        return s

    return system(0), system(1), set(A_ONLY), set(b_only), positions


# ---------------------------------------------------------------------------------------------
# The independent reference
# ---------------------------------------------------------------------------------------------

def _nb_params(system):
    nb = next(f for f in system.getForces() if type(f).__name__ == "NonbondedForce")
    q, s, e = [], [], []
    for i in range(nb.getNumParticles()):
        a, b, c = nb.getParticleParameters(i)
        q.append(a._value); s.append(b._value); e.append(c._value)
    exc = {}
    for k in range(nb.getNumExceptions()):
        i, j, qq, sg, ep = nb.getExceptionParameters(k)
        exc[(min(i, j), max(i, j))] = (qq._value, sg._value, ep._value)
    return np.array(q), np.array(s), np.array(e), exc, nb


def _bonded_energy(system, x, box):
    def vec(i, j):
        d = x[j] - x[i]
        if box is not None:
            d = d - box * np.round(d / box)
        return d
    total = 0.0
    for f in system.getForces():
        kind = type(f).__name__
        if kind == "HarmonicBondForce":
            for k in range(f.getNumBonds()):
                i, j, r0, kk = f.getBondParameters(k)
                total += 0.5 * kk._value * (np.linalg.norm(vec(i, j)) - r0._value) ** 2
        elif kind == "HarmonicAngleForce":
            for k in range(f.getNumAngles()):
                i, j, m, t0, kk = f.getAngleParameters(k)
                a, b = vec(j, i), vec(j, m)
                theta = math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
                total += 0.5 * kk._value * (theta - t0._value) ** 2
        elif kind == "PeriodicTorsionForce":
            for k in range(f.getNumTorsions()):
                i, j, m, l, per, ph, kk = f.getTorsionParameters(k)
                b1, b2, b3 = vec(i, j), vec(j, m), vec(m, l)
                n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                phi = math.atan2(m1 @ n2, n1 @ n2)
                total += kk._value * (1 + math.cos(per * phi - ph._value))
    return total


_RECIP_CACHE: dict = {}


def _recip(q, x, box, kappa, nmax):
    """Ewald reciprocal energy by an explicit k-sum; memoised on (q, x, box, kappa, nmax), since a
    lambda scan re-uses the same two charge sets at the same coordinates."""
    key = (np.asarray(q).tobytes(), np.asarray(x).tobytes(), box, kappa, nmax)
    if key not in _RECIP_CACHE:
        _RECIP_CACHE.clear() if len(_RECIP_CACHE) > 64 else None
        _RECIP_CACHE[key] = _recip_sum(q, x, box, kappa, nmax)
    return _RECIP_CACHE[key]


def _recip_sum(q, x, box, kappa, nmax):
    V = box ** 3
    n = np.arange(-nmax, nmax + 1)
    kv = np.array(list(itertools.product(n, n, n)), dtype=float)
    kv = kv[np.any(kv != 0, axis=1)] * (2 * math.pi / box)
    k2 = np.sum(kv * kv, axis=1)
    keep = np.exp(-k2 / (4 * kappa ** 2)) > 1e-18
    kv, k2 = kv[keep], k2[keep]
    phase = x @ kv.T
    s_re, s_im = q @ np.cos(phase), q @ np.sin(phase)
    return K_COULOMB / (2 * V) * np.sum(4 * math.pi / k2 * np.exp(-k2 / (4 * kappa ** 2))
                                        * (s_re ** 2 + s_im ** 2))


def _pairs(n, x, box):
    """Every pair i < j: index arrays and minimum-image distances."""
    i, j = np.triu_indices(n, 1)
    d = x[j] - x[i]
    if box is not None:
        d = d - box * np.round(d / box)
    return i, j, np.linalg.norm(d, axis=1)


def _lj(r, s, e):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e != 0.0, 4 * e * ((s / r) ** 12 - (s / r) ** 6), 0.0)


def _mask(i, j, pairs, n):
    """Boolean over the (i, j) pair arrays: is the pair in `pairs`?"""
    lookup = np.zeros((n, n), dtype=bool)
    for a, b in pairs:
        lookup[a, b] = True
    return lookup[i, j]


def reference_energy(system_a, system_b, a_only, b_only, x, state, *, kappa=None, cutoff=None,
                     box=None, nmax=22, alpha=0.5, beta=0.12, boundary_14="scaled"):
    """Components and total, kJ/mol, of U(state) under the Amber18 rules. `box` None = vacuum.

    Per end state X with unique region R (X-only) and the other region O (dummies in X):
      excluded (every exception pair, and every R-R pair): Ewald share -erf(kappa r)/r removed;
          a C-C exception adds its own chargeProd/r and LJ, weighted with X
      R-C pairs: q q erfc(kappa r)/sqrt(r^2 + beta l) and 4 eps [1/(alpha l + (r/s)^6)^2 - ...]
      C-C pairs: q q erfc(kappa r)/r and plain LJ, within the cutoff
      + reciprocal and self terms of X's charges
    mixed as (1 - l_e) El_A + l_e El_B and (1 - l_s) LJ_A + l_s LJ_B, with l = l_e / l_s inside A's
    softcore and 1 - l_e / 1 - l_s inside B's. Unscaled, from X where R is physical: every R-R
    non-excluded pair as vacuum Coulomb + LJ, and every exception inside R.

    An exception between an R particle and a C particle ("boundary"): `boundary_14 = "scaled"`
    treats it as the C-C exceptions are (weighted with X, zero in the end state where R is a
    dummy); "unscaled" adds it to the unscaled sum instead (Amber18 manual 21.1.5).
    """
    le, ls, lb = (state["lambda_electrostatics"], state["lambda_sterics"], state["lambda_bonded"])
    qa, sa, ea, exa, _ = _nb_params(system_a)
    qb, sb, eb, exb, _ = _nb_params(system_b)
    n = len(qa)
    periodic = box is not None
    i, j, r = _pairs(n, x, box)
    a_mask, b_mask = np.zeros(n, bool), np.zeros(n, bool)
    a_mask[list(a_only)] = True
    b_mask[list(b_only)] = True
    unique = a_mask | b_mask
    in_cut = (r < cutoff) if periodic else np.ones_like(r, dtype=bool)
    screen = erfc(kappa * r) if periodic else np.ones_like(r)

    def end_state(q, s, e, exc, region, other, l_e, l_s):
        is_exc = _mask(i, j, exc, n)
        internal = region[i] & region[j]
        excluded = is_exc | internal
        touches_unique = unique[i] | unique[j]
        el = 0.0
        vdw = 0.0
        qq = q[i] * q[j]
        if periodic:
            el -= K_COULOMB * np.sum(qq[excluded] * erf(kappa * r[excluded]) / r[excluded])
        both_unique = unique[i] & unique[j]
        if boundary_14 == "scaled":
            cc_exc = is_exc & ~both_unique
        else:
            cc_exc = is_exc & ~touches_unique
        ex = np.array([exc[(a, b)] if (a, b) in exc else (0.0, 1.0, 0.0) for a, b in zip(i, j)])
        el += K_COULOMB * np.sum(ex[cc_exc, 0] / r[cc_exc])
        vdw += np.sum(_lj(r[cc_exc], ex[cc_exc, 1], ex[cc_exc, 2]))
        live = ~excluded & ~(other[i] | other[j]) & in_cut
        soft = live & (region[i] | region[j])
        plain = live & ~soft
        s_ij, e_ij = 0.5 * (s[i] + s[j]), np.sqrt(e[i] * e[j])
        el += K_COULOMB * np.sum(qq[plain] * screen[plain] / r[plain])
        vdw += np.sum(_lj(r[plain], s_ij[plain], e_ij[plain]))
        el += K_COULOMB * np.sum(qq[soft] * screen[soft] / np.sqrt(r[soft] ** 2 + beta * l_e))
        with np.errstate(divide="ignore", invalid="ignore"):
            x6 = np.where(e_ij[soft] > 0, (r[soft] / s_ij[soft]) ** 6, 1.0)
        vdw += np.sum(4 * e_ij[soft] * (1 / (alpha * l_s + x6) ** 2 - 1 / (alpha * l_s + x6)))
        if periodic:
            el += _recip(q, x, box, kappa, nmax) - K_COULOMB * kappa / math.sqrt(math.pi) * np.sum(q * q)
        return el, vdw

    el_a, lj_a = end_state(qa, sa, ea, exa, a_mask, b_mask, le, ls)
    el_b, lj_b = end_state(qb, sb, eb, exb, b_mask, a_mask, 1 - le, 1 - ls)

    unscaled = 0.0
    for q, s, e, exc, region in ((qa, sa, ea, exa, a_mask), (qb, sb, eb, exb, b_mask)):
        touches = region[i] | region[j]
        is_exc = _mask(i, j, exc, n)
        ex = np.array([exc[(a, b)] if (a, b) in exc else (0.0, 1.0, 0.0) for a, b in zip(i, j)])
        inside = region[i] & region[j]
        sel = (inside if boundary_14 == "scaled" else touches) & is_exc
        unscaled += np.sum(K_COULOMB * ex[sel, 0] / r[sel] + _lj(r[sel], ex[sel, 1], ex[sel, 2]))
        sel = region[i] & region[j] & ~is_exc
        unscaled += np.sum(K_COULOMB * q[i[sel]] * q[j[sel]] / r[sel]
                           + _lj(r[sel], 0.5 * (s[i[sel]] + s[j[sel]]), np.sqrt(e[i[sel]] * e[j[sel]])))

    bonded = (1 - lb) * _bonded_energy(system_a, x, box) + lb * _bonded_energy(system_b, x, box)
    out = {"electrostatics": float((1 - le) * el_a + le * el_b),
           "lennard_jones": float((1 - ls) * lj_a + ls * lj_b),
           "bonded": float(bonded), "softcore_internal": float(unscaled)}
    out["total"] = math.fsum(out.values())
    return out


def plain_energy(system, x, *, kappa=None, cutoff=None, box=None, nmax=22):
    """An ordinary System's energy by the same reference routines -- for calibrating them."""
    q, s, e, exc, _ = _nb_params(system)
    n = len(q)
    periodic = box is not None
    i, j, r = _pairs(n, x, box)
    is_exc = _mask(i, j, exc, n)
    ex = np.array([exc[(a, b)] if (a, b) in exc else (0.0, 1.0, 0.0) for a, b in zip(i, j)])
    qq = q[i] * q[j]
    total = np.sum(K_COULOMB * ex[is_exc, 0] / r[is_exc] + _lj(r[is_exc], ex[is_exc, 1], ex[is_exc, 2]))
    live = ~is_exc & ((r < cutoff) if periodic else True)
    screen = erfc(kappa * r[live]) if periodic else 1.0
    total += np.sum(K_COULOMB * qq[live] * screen / r[live])
    total += np.sum(_lj(r[live], 0.5 * (s[i[live]] + s[j[live]]), np.sqrt(e[i[live]] * e[j[live]])))
    if periodic:
        total -= K_COULOMB * np.sum(qq[is_exc] * erf(kappa * r[is_exc]) / r[is_exc])
        total += _recip(q, x, box, kappa, nmax) - K_COULOMB * kappa / math.sqrt(math.pi) * np.sum(q * q)
    return float(total + _bonded_energy(system, x, box))


# ---------------------------------------------------------------------------------------------
# Force / energy consistency, platform-agnostic (used on Reference and on CUDA)
# ---------------------------------------------------------------------------------------------

def fd_force_check(context, atoms, steps):
    """Per atom and Cartesian component: the force F, central finite differences -dE/dx of the
    Context's TOTAL energy at each step in `steps` (each half the previous), their errors, and the
    Richardson estimate from the two smallest steps, (4 D(h/2) - D(h)) / 3, and its error.

    A smooth energy whose force is its gradient shows errors falling ~4x per halving and a small
    Richardson error. A discontinuity or a kink inside the stencil shows neither. Positions are
    restored."""
    x0 = context.getState(getPositions=True).getPositions(asNumpy=True)._value.copy()
    forces = context.getState(getForces=True).getForces(asNumpy=True)._value
    rows = []
    for i in atoms:
        for k in range(3):
            fds = []
            for delta in steps:
                energies = []
                for sign in (1.0, -1.0):
                    x = x0.copy()
                    x[i, k] += sign * delta
                    context.setPositions(x)
                    energies.append(context.getState(getEnergy=True).getPotentialEnergy()._value)
                fds.append(-(energies[0] - energies[1]) / (2 * delta))
            f = float(forces[i, k])
            richardson = (4 * fds[-1] - fds[-2]) / 3
            rows.append({"atom": i, "component": "xyz"[k], "force": f,
                         "errors": [abs(f - d) for d in fds],
                         "richardson_error": abs(f - richardson)})
    context.setPositions(x0)
    return rows


def defective_system(h, group_name, r0, kind, size):
    """A copy of the Hamiltonian's System with one custom nonbonded force made non-smooth AT r0:
      "step": the energy jumps by `size` (relative) at r0, with no force for it;
      "kink": the energy gains size * |r - r0| (kJ/mol per nm): continuous, slope discontinuous.
    The defects a finite-difference force check must catch at the evaluated geometry."""
    import openmm
    from md_tools.alchemy.hamiltonian import FORCE_GROUPS
    s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(h.system))
    for f in s.getForces():
        if isinstance(f, openmm.CustomNonbondedForce) and f.getForceGroup() == FORCE_GROUPS[group_name]:
            head, tail = f.getEnergyFunction().split(";", 1)
            if kind == "step":
                head = f"({head})*(1+{size!r}*step({r0!r}-r))"
            elif kind == "kink":
                head = f"({head}) + {size!r}*abs(r-{r0!r})"
            else:
                raise ValueError(kind)
            f.setEnergyFunction(f"{head};{tail}")
    return s


def decoupling_pair(system_a, ligand):
    """(System A, System B) in the shape an ABFE decoupling plan has: `ligand` is the whole unique
    region, there is no mapped core and nothing appears.

    System A is the solvated System untouched. In System B every ligand particle has charge 0 and
    epsilon 0 and every ligand-environment exception is zero, while the ligand's OWN exceptions keep
    their physical values and its non-excluded internal pairs move into the
    `UniqueGroupInternalNonbonded` force (contract section 4). The ligand's bonded terms are
    identical in both. So at the decoupled end the ligand is a physical molecule in vacuum inside
    the box, and only its interactions with the environment have been switched off.
    """
    import openmm
    from md_tools.alchemy.topology import COULOMB_CONSTANT, INTERNAL_FORCE_NAME
    ligand = set(int(i) for i in ligand)
    pair = []
    for decoupled in (False, True):
        s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_a))
        for k in reversed(range(s.getNumForces())):       # the pair's own internal force replaces
            if s.getForce(k).getName() == INTERNAL_FORCE_NAME:   # any the fixture already wrote
                s.removeForce(k)
        nb = next(f for f in s.getForces() if isinstance(f, openmm.NonbondedForce))
        physical = [nb.getParticleParameters(i) for i in range(nb.getNumParticles())]
        excepted = set()
        for k in range(nb.getNumExceptions()):
            i, j, qq, sg, ep = nb.getExceptionParameters(k)
            excepted.add((min(i, j), max(i, j)))
            if decoupled and ((i in ligand) != (j in ligand)):
                nb.setExceptionParameters(k, i, j, 0.0, sg, 0.0)     # ligand-environment: off
        if decoupled:
            for i in sorted(ligand):
                nb.setParticleParameters(i, 0.0, physical[i][1], 0.0)
        f = openmm.CustomBondForce(
            f"{COULOMB_CONSTANT!r}*chargeprod/r + 4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
        for name in ("chargeprod", "sigma", "epsilon"):
            f.addPerBondParameter(name)
        f.setName(INTERNAL_FORCE_NAME)
        f.setUsesPeriodicBoundaryConditions(True)
        members = sorted(ligand)
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if (i, j) in excepted:
                    continue
                qi, si, ei = (v._value for v in physical[i])
                qj, sj, ej = (v._value for v in physical[j])
                params = [qi * qj, 0.5 * (si + sj), math.sqrt(ei * ej)] if decoupled else \
                    [0.0, 0.5 * (si + sj), 0.0]
                f.addBond(i, j, params)
        s.addForce(f)
        pair.append(s)
    return pair[0], pair[1]
