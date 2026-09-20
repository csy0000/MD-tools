"""Endpoint recovery of a topology plan, accounted term by term.

Comparing the raw energy of an endpoint System with the physical endpoint's energy is not an
endpoint test: the dummies' retained bonded terms are in one and not the other, so an exact
construction FAILS it and a broken one can be made to pass it by accident. What is checked here
instead:

``audit_plan``
    STRUCTURE. At each endpoint, the System restricted to that endpoint's physical ligand atoms
    carries exactly the package's parameter table -- every nonzero bonded term once, every
    exception, every charge and Lennard-Jones pair -- nothing missing, nothing duplicated; every
    environment term is untouched; every dummy is non-interacting; the two Systems agree in
    particles, masses, constraints and force layout. Run by the builder, so an inconsistent plan
    is never returned.

``factorization_check``
    SEPARABILITY, by arithmetic alone (no OpenMM). For every dummy group, the retained dummy
    energy is evaluated while the physical atoms are moved and the group is carried along at
    fixed coordinates in its (P1, P2, P3) frame. Constant means the dummy's configurational
    integral does not depend on the physical coordinates, i.e. it cancels between the legs of a
    cycle. The same measurement with EVERY term touching the group (the OpenFE default) is
    reported beside it, so the check is seen to be able to fail.

``endpoint_accounting``
    ENERGY, against an independent reference System for the physical endpoint:

        E_endpoint(x) = E_reference(x_phys) + E_dummy(x) + dE_dispersion + E_restraint

    per force class, with E_dummy (retained bonded terms plus each unique group's internal
    nonbonded energy) and E_restraint evaluated here in numpy from the plan's own record, and
    dE_dispersion the change in OpenMM's long-range dispersion correction caused by the
    zero-epsilon dummy particles (it averages over every particle). Nothing is left in a
    residual that the accounting does not name.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "FACTORIZATION_TOL_KJ",
    "RECOVERY_PLATFORM",
    "audit_plan",
    "dummy_energy",
    "endpoint_accounting",
    "factorization_check",
    "internal_nonbonded_energy",
    "restraint_energy",
]

#: Every Context this module creates is on OpenMM's Reference platform, by construction and with
#: no parameter to change it. Recovery is construction-time arithmetic, like build-top's hydrogen
#: relaxation: it must be float64 and bit-reproducible, and there is nothing to accelerate. A
#: `platform` argument here was a second platform policy beside
#: `md_tools.openmm.platform_policy` -- a caller could pass "CUDA" and bypass the machine
#: configuration, the device policy and --cpu -- so there is none. These Contexts never reach
#: platform_policy and are not CUDA evidence.
RECOVERY_PLATFORM = "Reference"

#: The retained dummy energy may vary by at most this under a physical move (kJ/mol): pure float64
#: arithmetic, so the floor is rounding. The unfiltered variation it is compared against is O(1).
FACTORIZATION_TOL_KJ = 1e-8

def _q(value) -> float:
    from openmm import unit

    if hasattr(value, "value_in_unit_system"):
        value = value.value_in_unit_system(unit.md_unit_system)
    return float(value)


# ------------------------------------------------------------------------------------------------
# arithmetic, independent of OpenMM's kernels
# ------------------------------------------------------------------------------------------------
def _angle(p0, p1, p2) -> float:
    u, v = p0 - p1, p2 - p1
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(1.0, c)))


def _dihedral(p0, p1, p2, p3) -> float:
    """IUPAC dihedral, the convention OpenMM's PeriodicTorsionForce uses."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.atan2(y, x)


def _term_energy(family: str, slot: dict, params: Sequence[float], x: np.ndarray) -> float:
    atoms = slot["atoms"]
    if family == "bonds":
        r = float(np.linalg.norm(x[atoms[0]] - x[atoms[1]]))
        return 0.5 * params[1] * (r - params[0]) ** 2
    if family == "angles":
        theta = _angle(*(x[a] for a in atoms))
        return 0.5 * params[1] * (theta - params[0]) ** 2
    phi = _dihedral(*(x[a] for a in atoms))
    return params[1] * (1.0 + math.cos(slot["periodicity"] * phi - params[0]))


def dummy_energy(record: dict, endpoint: str, positions_nm: np.ndarray, *,
                 atoms: Optional[set] = None, include_removed: bool = False) -> dict[str, float]:
    """The bonded energy of the dummies at *endpoint*, per family, from the plan record alone.

    A term counts if it touches a dummy of that endpoint (or, with *atoms*, one of those atoms).
    With *include_removed*, terms the junction rule removed are evaluated with their PHYSICAL
    parameters -- what the dummy energy would be had they been kept, as OpenFE keeps them.
    """
    endpoint = endpoint.upper()
    dummies = set(record["particles"]["b_only" if endpoint == "A" else "a_only"])
    if atoms is not None:
        dummies &= set(atoms)
    side = endpoint.lower()
    out = {"bonds": 0.0, "angles": 0.0, "torsions": 0.0}
    out.update(internal_nonbonded_energy(record, endpoint, positions_nm, atoms=dummies))
    for family, slots in record["terms"].items():
        for slot in slots:
            if not dummies & set(slot["atoms"]):
                continue
            if slot["at_dummy_end"] == "dummy-retained":
                out[family] += _term_energy(family, slot, slot[side], positions_nm)
            elif include_removed and slot["at_dummy_end"] == "dummy-removed":
                physical = slot["b" if endpoint == "A" else "a"]
                out[family] += _term_energy(family, slot, physical, positions_nm)
    return out


def _vacuum_pair(r: float, qq: float, sigma: float, epsilon: float) -> float:
    from .topology import COULOMB_CONSTANT

    return COULOMB_CONSTANT * qq / r + 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def internal_nonbonded_energy(record: dict, endpoint: str, positions_nm: np.ndarray, *,
                              atoms: Optional[set] = None) -> dict[str, float]:
    """The unique groups' INTERNAL nonbonded energy at the endpoint where they are dummies.

    `internal_exceptions` is carried by the NonbondedForce (the group's own 1-4 exceptions, kept
    physical), `internal_pairs` by the `UniqueGroupInternalNonbonded` CustomBondForce (its
    non-excluded pairs). Vacuum Coulomb and Lennard-Jones, over the group's internal distances
    only, so it separates from the physical system exactly as the retained bonded terms do.
    Molecules are taken as contiguous in the coordinates (no minimum image).
    """
    endpoint = endpoint.upper()
    dummies = set(record["particles"]["b_only" if endpoint == "A" else "a_only"])
    if atoms is not None:
        dummies &= set(atoms)
    side = endpoint.lower()
    x = positions_nm
    out = {"internal_exceptions": 0.0, "internal_pairs": 0.0}
    for slot in record["nonbonded"]["exceptions"]:
        i, j = slot["atoms"]
        if slot.get("unique_group_internal") and i in dummies and j in dummies:
            out["internal_exceptions"] += _vacuum_pair(float(np.linalg.norm(x[j] - x[i])),
                                                       *slot[side])
    for pair in record["nonbonded"]["unique_group_internal"]["pairs"]:
        i, j = pair["atoms"]
        if pair["dummy_at"] == endpoint and i in dummies and j in dummies:
            out["internal_pairs"] += _vacuum_pair(float(np.linalg.norm(x[j] - x[i])),
                                                  *pair["physical"])
    return out


def restraint_energy(record: dict, positions_nm: np.ndarray, box_nm: Optional[np.ndarray]) -> float:
    """The energy of the restraints this plan BUILDS: the alchemical-coupling ones.

    A standard-state restraint is recorded but built by the executor, so it is not evaluated here.
    """
    total = 0.0
    for restraint in record.get("restraints") or []:
        if restraint.get("role") != "alchemical-coupling":
            continue
        ca = positions_nm[restraint["group_a"]].mean(axis=0)
        cb = positions_nm[restraint["group_b"]].mean(axis=0)
        d = cb - ca
        if restraint["periodic"]:
            box = np.asarray(box_nm, dtype=float)
            if np.count_nonzero(box - np.diag(np.diag(box))):
                raise ValueError("restraint_energy evaluates rectangular boxes only")
            lengths = np.diag(box)
            d = d - lengths * np.round(d / lengths)
        total += 0.5 * restraint["k_kj_mol_nm2"] * float(np.dot(d, d))
    return total


# ------------------------------------------------------------------------------------------------
# separability
# ------------------------------------------------------------------------------------------------
def _frame(x: np.ndarray, p1: int, p2: Optional[int], p3: Optional[int]):
    origin = x[p1]
    if p2 is None:
        return origin, np.eye(3)
    e1 = x[p2] - origin
    e1 /= np.linalg.norm(e1)
    if p3 is not None:
        ref = x[p3] - x[p2]
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = ref - np.dot(ref, e1) * e1
    e2 /= np.linalg.norm(e2)
    return origin, np.stack([e1, e2, np.cross(e1, e2)])


def factorization_check(plan, *, trials: int = 8, displacement_nm: float = 0.03,
                        seed: int = 20260919) -> dict[str, Any]:
    """Refuse a plan whose retained dummy energy depends on physical coordinates."""
    from .topology import TopologyError

    record = plan.record
    if record["mode"] == "dual":
        return {"groups": [], "note": "dual topology: a dummy ligand has no bonded contact with "
                                      "the physical one; the centroid restraint separates "
                                      "analytically (see record.restraint)"}
    rng = np.random.default_rng(seed)
    x0 = np.array(plan.positions_nm, dtype=float)
    ligand = sorted(set(record["particles"]["common"]) | set(record["particles"]["a_only"])
                    | set(record["particles"]["b_only"]))
    results = []
    for group in record["dummy_groups"]:
        endpoint, members = group["dummy_at"], set(group["atoms"])
        frame = group["frame"]
        origin, axes = _frame(x0, frame["p1"], frame["p2"], frame["p3"])
        local = {a: axes @ (x0[a] - origin) for a in members}
        physical = [a for a in ligand if a not in members]
        base_kept = sum(dummy_energy(record, endpoint, x0, atoms=members).values())
        base_all = sum(dummy_energy(record, endpoint, x0, atoms=members,
                                    include_removed=True).values())
        worst_kept = worst_all = 0.0
        for _ in range(trials):
            x = x0.copy()
            x[physical] += rng.normal(scale=displacement_nm, size=(len(physical), 3))
            o, r = _frame(x, frame["p1"], frame["p2"], frame["p3"])
            for a in members:
                x[a] = o + r.T @ local[a]
            worst_kept = max(worst_kept, abs(sum(dummy_energy(
                record, endpoint, x, atoms=members).values()) - base_kept))
            worst_all = max(worst_all, abs(sum(dummy_energy(
                record, endpoint, x, atoms=members, include_removed=True).values()) - base_all))
        if worst_kept > FACTORIZATION_TOL_KJ:
            raise TopologyError(
                f"dummy group {group['local_names']} (dummy at {endpoint}): its retained energy "
                f"changes by {worst_kept:.3e} kJ/mol when only physical atoms move; its partition "
                f"function would not cancel")
        results.append({"dummy_at": endpoint, "atoms": sorted(members),
                        "retained_terms_max_variation_kj_mol": worst_kept,
                        "all_terms_max_variation_kj_mol": worst_all,
                        "terms_removed": group["terms_removed"]})
    return {"groups": results, "tolerance_kj_mol": FACTORIZATION_TOL_KJ, "trials": trials,
            "displacement_nm": displacement_nm, "seed": seed}


# ------------------------------------------------------------------------------------------------
# structure
# ------------------------------------------------------------------------------------------------
def _nonzero_terms(system, to_local: dict[int, int], bonds_local: set) -> dict[str, list]:
    """Every term of *system* lying wholly within *to_local*, in local indices, zeros dropped."""
    from openmm import (HarmonicAngleForce, HarmonicBondForce, NonbondedForce,
                        PeriodicTorsionForce)

    out: dict[str, list] = {"bonds": [], "angles": [], "proper_torsions": [],
                            "improper_torsions": [], "exceptions": [], "atoms": {}}
    for force in system.getForces():
        if isinstance(force, HarmonicBondForce):
            for k in range(force.getNumBonds()):
                i, j, r, kk = force.getBondParameters(k)
                if i in to_local and j in to_local and _q(kk) != 0.0:
                    a, b = sorted((to_local[i], to_local[j]))
                    out["bonds"].append((a, b, _q(r), _q(kk)))
        elif isinstance(force, HarmonicAngleForce):
            for k in range(force.getNumAngles()):
                i, j, l, t, kk = force.getAngleParameters(k)
                if {i, j, l} <= set(to_local) and _q(kk) != 0.0:
                    a, c = sorted((to_local[i], to_local[l]))
                    out["angles"].append((a, to_local[j], c, _q(t), _q(kk)))
        elif isinstance(force, PeriodicTorsionForce):
            for k in range(force.getNumTorsions()):
                i, j, kk_, l, n, phase, fk = force.getTorsionParameters(k)
                if {i, j, kk_, l} <= set(to_local) and _q(fk) != 0.0:
                    quartet = [to_local[i], to_local[j], to_local[kk_], to_local[l]]
                    chain = all(tuple(sorted(p)) in bonds_local for p in
                                ((quartet[0], quartet[1]), (quartet[1], quartet[2]),
                                 (quartet[2], quartet[3])))
                    if chain:
                        quartet = min(quartet, quartet[::-1])
                        out["proper_torsions"].append((*quartet, int(n), _q(phase), _q(fk)))
                    else:
                        out["improper_torsions"].append((*quartet, int(n), _q(phase), _q(fk)))
        elif isinstance(force, NonbondedForce):
            for h, local in to_local.items():
                out["atoms"][local] = tuple(_q(v) for v in force.getParticleParameters(h))
            for k in range(force.getNumExceptions()):
                i, j, qq, s, e = force.getExceptionParameters(k)
                if i in to_local and j in to_local:
                    a, b = sorted((to_local[i], to_local[j]))
                    out["exceptions"].append((a, b, _q(qq), _q(s), _q(e)))
    return out


def _table_nonzero(table: dict) -> dict[str, list]:
    out = {"bonds": [tuple(r) for r in table["bonds"] if r[3] != 0.0],
           "angles": [tuple(r) for r in table["angles"] if r[4] != 0.0],
           "proper_torsions": [tuple(r) for r in table["proper_torsions"] if r[6] != 0.0],
           "improper_torsions": [tuple(r) for r in table["improper_torsions"] if r[6] != 0.0],
           "exceptions": [tuple(r) for r in table["exceptions"]]}
    return out


def _same_multiset(a: list, b: list, width: int, rtol: float = 1e-9) -> Optional[str]:
    if len(a) != len(b):
        return f"{len(a)} terms vs {len(b)}"
    for x, y in zip(sorted(a), sorted(b)):
        if tuple(x[:width]) != tuple(y[:width]):
            return f"term {x[:width]} vs {y[:width]}"
        for p, q in zip(x[width:], y[width:]):
            if not math.isclose(float(p), float(q), rel_tol=rtol, abs_tol=1e-12):
                return f"term {x} vs {y}"
    return None


def audit_plan(plan, package_a, package_b, env_system) -> dict[str, Any]:
    """Structural endpoint recovery; raises `TopologyError` on the first discrepancy."""
    from openmm import NonbondedForce

    from .topology import TopologyError, scaled_table
    from .topology_mapping import _bonds

    record = plan.record
    constrained = set()
    system_a, system_b = plan.system_a, plan.system_b
    for k in range(system_a.getNumConstraints()):
        i, j, _ = system_a.getConstraintParameters(k)
        constrained.add(tuple(sorted((i, j))))

    # identical layout
    if system_a.getNumParticles() != system_b.getNumParticles():
        raise TopologyError("the endpoint Systems differ in particle count")
    for i in range(system_a.getNumParticles()):
        if _q(system_a.getParticleMass(i)) != _q(system_b.getParticleMass(i)):
            raise TopologyError(f"particle {i} has different masses in the endpoint Systems")
    if [system_a.getConstraintParameters(k) for k in range(system_a.getNumConstraints())] != \
            [system_b.getConstraintParameters(k) for k in range(system_b.getNumConstraints())]:
        raise TopologyError("the endpoint Systems differ in constraints")
    if [type(f).__name__ for f in system_a.getForces()] != \
            [type(f).__name__ for f in system_b.getForces()]:
        raise TopologyError("the endpoint Systems differ in forces")

    counts = {}
    for side, package, system in (("A", package_a, system_a), ("B", package_b, system_b)):
        to_hyb = record["endpoints"][side]["hybrid_index_of_local_atom"]
        if record["endpoints"][side].get("absent"):
            # Decoupling: endpoint B has no ligand at all, so there is no table to reproduce.
            # What must hold there -- every ligand particle inert, its own terms untouched -- is
            # checked as the dummy side of endpoint A below.
            counts[side] = {"absent": True}
            continue
        to_local = {h: i for i, h in enumerate(to_hyb)}
        bonds_local = _bonds(package.mol)
        got = _nonzero_terms(system, to_local, bonds_local)
        want = _table_nonzero(scaled_table(package, record["environment"]["nonbonded_applied"]))
        # a constrained bond has no term; the package table has one. Its length is checked in
        # constraints-consistent; here it is removed from the expectation.
        want["bonds"] = [r for r in want["bonds"]
                         if tuple(sorted((to_hyb[r[0]], to_hyb[r[1]]))) not in constrained]
        for key, width in (("bonds", 2), ("angles", 3), ("proper_torsions", 5),
                           ("improper_torsions", 5), ("exceptions", 2)):
            problem = _same_multiset(got[key], want[key], width)
            if problem:
                raise TopologyError(f"endpoint {side}: {key} restricted to package "
                                    f"{package.reference}'s atoms do not reproduce its table: "
                                    f"{problem}")
        for local, row in enumerate(package.table["atoms"]):
            if any(not math.isclose(p, q, rel_tol=1e-9, abs_tol=1e-12)
                   for p, q in zip(got["atoms"][local], row[2:5])):
                raise TopologyError(f"endpoint {side}: atom {package.atom_names[local]} carries "
                                    f"{got['atoms'][local]}, not the package's {row[2:5]}")
        counts[side] = {k: len(v) for k, v in want.items()}

        # dummies interact with nothing
        dummies = set(record["particles"]["b_only" if side == "A" else "a_only"])
        nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        for d in dummies:
            q, _, e = nb.getParticleParameters(d)
            if _q(q) != 0.0 or _q(e) != 0.0:
                raise TopologyError(f"endpoint {side}: dummy {d} has charge or epsilon")
        group_of = {h: n for n, g in enumerate(
            record["nonbonded"]["unique_group_internal"]["groups"]) for h in g["atoms"]}
        for k in range(nb.getNumExceptions()):
            i, j, qq, _, e = nb.getExceptionParameters(k)
            inside = i in group_of and group_of.get(j, -1) == group_of[i]
            if (i in dummies or j in dummies) and not inside and (_q(qq) != 0.0 or _q(e) != 0.0):
                raise TopologyError(f"endpoint {side}: exception {i}-{j} touches a dummy, is not "
                                    f"inside its unique group, and is not zero")

    # the environment is untouched outside the ligand
    ligand = set(record["particles"]["common"]) | set(record["particles"]["a_only"])
    n_env = record["particles"]["n_environment"]
    for label, system in (("A", system_a), ("B", system_b)):
        for f_env, f_sys in zip(env_system.getForces(), system.getForces()):
            _check_environment_terms(f_env, f_sys, ligand, n_env, label)
    return {"endpoint_tables_reproduced": counts,
            "zero_terms_ignored": "a term with force constant 0 contributes nothing and is not "
                                  "counted as present",
            "environment_terms": "identical outside the ligand in both Systems",
            "layout": "identical particles, masses, constraints and force classes"}


def _check_environment_terms(f_env, f_sys, ligand: set, n_env: int, label: str) -> None:
    from openmm import (HarmonicAngleForce, HarmonicBondForce, NonbondedForce,
                        PeriodicTorsionForce)

    from .topology import TopologyError

    def compare(n, get_env, get_sys, atoms_of, what):
        for k in range(n):
            e = get_env(k)
            if set(atoms_of(e)) & ligand:
                continue
            if get_sys(k) != e:
                raise TopologyError(f"endpoint {label}: environment {what} {k} changed")

    if isinstance(f_env, NonbondedForce):
        for i in range(n_env):
            if i not in ligand and f_env.getParticleParameters(i) != f_sys.getParticleParameters(i):
                raise TopologyError(f"endpoint {label}: environment particle {i} changed")
        compare(f_env.getNumExceptions(), f_env.getExceptionParameters,
                f_sys.getExceptionParameters, lambda e: e[:2], "exception")
    elif isinstance(f_env, HarmonicBondForce):
        compare(f_env.getNumBonds(), f_env.getBondParameters, f_sys.getBondParameters,
                lambda e: e[:2], "bond")
    elif isinstance(f_env, HarmonicAngleForce):
        compare(f_env.getNumAngles(), f_env.getAngleParameters, f_sys.getAngleParameters,
                lambda e: e[:3], "angle")
    elif isinstance(f_env, PeriodicTorsionForce):
        compare(f_env.getNumTorsions(), f_env.getTorsionParameters, f_sys.getTorsionParameters,
                lambda e: e[:4], "torsion")


# ------------------------------------------------------------------------------------------------
# energy
# ------------------------------------------------------------------------------------------------
def _energies_by_class(system, positions_nm) -> dict[str, float]:
    import openmm
    from openmm import unit

    work = openmm.XmlSerializer.clone(system)
    classes = []
    for k, force in enumerate(work.getForces()):
        name = type(force).__name__
        if name not in classes:
            classes.append(name)
        force.setForceGroup(classes.index(name))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(work, integrator,
                             openmm.Platform.getPlatformByName(RECOVERY_PLATFORM))
    context.setPositions(positions_nm)
    out = {}
    for g, name in enumerate(classes):
        state = context.getState(getEnergy=True, groups={g})
        out[name] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return out


def _dispersion(system, positions_nm) -> float:
    """OpenMM's long-range dispersion correction for *system*, as on minus off."""
    import openmm
    from openmm import NonbondedForce, unit

    nb = next((f for f in system.getForces() if isinstance(f, NonbondedForce)), None)
    if nb is None or not nb.getUseDispersionCorrection() \
            or not system.usesPeriodicBoundaryConditions():
        return 0.0
    values = []
    for flag in (True, False):
        work = openmm.System()
        for i in range(system.getNumParticles()):
            work.addParticle(system.getParticleMass(i))
        work.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        force = openmm.XmlSerializer.clone(nb)
        force.setUseDispersionCorrection(flag)
        work.addForce(force)
        context = openmm.Context(work, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName(RECOVERY_PLATFORM))
        context.setPositions(positions_nm)
        values.append(context.getState(getEnergy=True).getPotentialEnergy()
                      .value_in_unit(unit.kilojoule_per_mole))
    return values[0] - values[1]


def endpoint_accounting(plan, endpoint: str, reference_system, reference_to_hybrid: Sequence[int],
                        *, positions_nm: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Every term of the endpoint energy, named, against an independent physical reference.

    *reference_to_hybrid[i]* is the plan particle that reference particle i is. The reference
    System must describe exactly the physical endpoint -- built independently, e.g. by the force
    field from the endpoint's own package -- and the residual per force class is returned for
    the caller to hold to a tolerance.
    """
    endpoint = endpoint.upper()
    x = np.array(plan.positions_nm if positions_nm is None else positions_nm, dtype=float)
    system = plan.system(endpoint)
    hybrid = _energies_by_class(system, x)
    reference = _energies_by_class(reference_system, x[list(reference_to_hybrid)])
    dummy = dummy_energy(plan.record, endpoint, x)
    box = np.array([[_q(c) for c in v] for v in system.getDefaultPeriodicBoxVectors()])
    restraint = restraint_energy(plan.record, x, box)
    dispersion = _dispersion(system, x) - _dispersion(
        reference_system, x[list(reference_to_hybrid)])
    accounted = {
        "HarmonicBondForce": reference.get("HarmonicBondForce", 0.0) + dummy["bonds"],
        "HarmonicAngleForce": reference.get("HarmonicAngleForce", 0.0) + dummy["angles"],
        "PeriodicTorsionForce": reference.get("PeriodicTorsionForce", 0.0) + dummy["torsions"],
        "NonbondedForce": (reference.get("NonbondedForce", 0.0) + dispersion
                           + dummy["internal_exceptions"]),
        "CustomBondForce": dummy["internal_pairs"],
        "CustomCentroidBondForce": restraint,
    }
    for name, value in reference.items():
        accounted.setdefault(name, value)
    residual = {name: hybrid.get(name, 0.0) - accounted.get(name, 0.0)
                for name in sorted(set(hybrid) | set(accounted))}
    return {
        "endpoint": endpoint,
        "platform": RECOVERY_PLATFORM,
        "hybrid": hybrid,
        "reference": reference,
        "dummy": dummy,
        "dispersion_correction_shift": dispersion,
        "restraint": restraint,
        "raw_total_difference": sum(hybrid.values()) - sum(reference.values()),
        "residual": residual,
    }
