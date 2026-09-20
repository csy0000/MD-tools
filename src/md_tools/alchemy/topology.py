"""The alchemical topology plan: two endpoint Systems over ONE particle index space.

`build_topology_plan` takes two registered ligand parameter packages, an explicit atom map between
them (`alchemy.topology_mapping`), and ONE built environment holding endpoint A. It returns a
`TopologyPlan`: a record, independent of any MD runner, plus the two endpoint Systems it
describes. The Hamiltonian builder, the executor and the analysis all read the plan rather than
re-deriving a mapping each.

THE INDEX SPACE

Every environment particle -- solvent, ions, protein and the endpoint-A ligand -- keeps the index
it has in the environment System. The particles only endpoint B has are APPENDED, as one new
residue in a new chain. Nothing is renumbered, so an atom index or a one-based residue index
resolved against the environment before combination means the same atom after it.

    common   ligand particles physical at both endpoints (the mapped core; empty in dual)
    a_only   physical at A, dummy at B
    b_only   physical at B, dummy at A (the appended particles)

THE THREE REPRESENTATIONS (separated topology is deferred and refused)

    single   ONE evolving atom representation: every atom of one endpoint is mapped, a mapped
             atom may change element (a hydrogen may become a heavy atom where no constraint
             forbids it), and the other endpoint's surplus atoms are dummies.
    hybrid   a mapped core plus endpoint-unique atoms on BOTH sides, as separate particles. A
             hydrogen is never mapped to a heavy atom.
    dual     the two ligands share no particle. Each is a complete molecule, B appended whole;
             every A-B pair is excluded; a harmonic restraint between the centroids of the mapped
             atom groups keeps the dummy ligand with the physical one.

THE ENDPOINT SYSTEMS

System A is the environment System, unchanged, plus the appended particles and terms. System B
has the same particles, masses, constraints and force layout, term for term; only parameter
values differ. A dummy has charge 0 and epsilon 0 (sigma is kept), and every exception touching
it is zero, so it has no nonbonded interaction with anything, itself included. Every A-only x
B-only pair is an explicit zero exception in both Systems: they never interact, at any lambda.

A dummy keeps bonded terms, and that is where endpoint recovery is decided. A dummy group's
contribution cancels in a thermodynamic cycle only if its configurational integral does not
depend on the physical coordinates -- the partition function must FACTORIZE. OpenFE's hybrid
factory keeps every bonded term between the dummy region and the core and documents that this
can give systematic errors; this construction addresses it with the single-anchor rule of Fleck,
Wieder and Boresch (J. Chem. Theory Comput. 2021, 17, 4403):

    each unique group G hangs from exactly ONE bond D1-P1 to the core (refused otherwise, in
    `alchemy.topology_mapping`). Choose P2, a core neighbour of P1, and P3, a core neighbour of
    P2 other than P1 (heavy atoms first, then lowest package-local index: a property of the
    package, so every leg of a cycle chooses the same frame). At the endpoint where G is a
    dummy, RETAIN a term touching G only if its atoms lie in G + {P1, P2}, or it is the torsion
    D1-P1-P2-P3.

Every retained term is then a function of G's coordinates in the frame (P1, P2, P3) alone -- bond
length, one bend and one azimuth for D1, anything else internal to G -- and never of |P1-P2| or
of any other physical internal coordinate, so the dummy integral is a constant. Every other term
touching G (a second bend D1-P1-P2', a second azimuth D1-P1-P2-P3') couples the dummy to a
physical internal coordinate and is REMOVED at that endpoint (force constant 0), while staying at
full strength at the endpoint where G is physical. Each decision is recorded per term.

Dispersion correction: OpenMM's long-range correction averages over EVERY particle, zero-epsilon
dummies included, so adding a dummy shifts it by a small, volume-dependent amount. It is not
hidden: `alchemy.topology_recovery` reports it as its own term of the endpoint accounting.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..ligands.mapping import LigandSelector
from ..ligands.package import (LigandPackage, compare_parameter_tables, load_package,
                               subsystem_parameter_table)
from .topology_mapping import (MAP_SCHEMA, AtomMap, MapError, _bonds, _neighbours, _positions,
                               validate_map)

__all__ = [
    "PLAN_SCHEMA",
    "Environment",
    "applied_scales",
    "scaled_table",
    "TopologyError",
    "TopologyPlan",
    "PLAN_IDENTITY_FIELDS",
    "build_decoupling_plan",
    "build_topology_plan",
    "check_plan_digest",
    "ligand_hamiltonian",
    "load_plan",
    "matched_legs",
]

#: The plan record's schema. BUMPED whenever a field is added or removed, because every stored
#: plan_sha256 stops being reproducible when the record's shape changes -- which is how an
#: in-flight campaign's legs came to name a digest nothing could produce. Adding fields is
#: therefore batched and landed BEFORE a campaign starts (shared contracts, section 4).
#:
#: 1: the original record. 2: adds ligand_hamiltonian_sha256, environment.solvation,
#:    nonbonded.unique_group_internal and the per-slot unique_group_internal flag.
#: 3: `restraint` becomes `restraints`, a list whose entries carry a ROLE, and no restraint is
#:    part of ligand_hamiltonian_sha256 in any mode (S0 ruling, 2026-09-20).
PLAN_SCHEMA = "md-tools-topology-plan/3"
#: What a stored record's identity is compared on when its digest no longer matches: if these all
#: agree, only the record's shape moved and the physics is the same.
PLAN_IDENTITY_FIELDS = ("ligand_hamiltonian_sha256", "mode", "atom_map.sha256",
                        "endpoints.A.reference", "endpoints.A.package_sha256",
                        "endpoints.A.parameter_digest", "endpoints.B.reference",
                        "endpoints.B.package_sha256", "endpoints.B.parameter_digest")
PLAN_FILES = ("plan.json", "system_a.xml", "system_b.xml", "combined.pdb", "positions.npy")

#: Force classes an environment may carry. Only the first four may touch the ligand.
LIGAND_FORCES = ("NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
                 "PeriodicTorsionForce")
PASSIVE_FORCES = ("CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
                  "MonteCarloMembraneBarostat", "MonteCarloFlexibleBarostat", "CMAPTorsionForce")

#: Relative tolerance for "the environment carries package A's parameters": the package's own
#: comparison tolerance (`ligands.package.compare_parameter_tables`).
PARAMETER_RTOL = 1e-9
#: Two constraint lengths are the same constraint only if they agree to this (nm). Lengths from one
#: force-field parameter are bit-identical; a real difference is >= 1e-4 nm.
CONSTRAINT_TOL_NM = 1e-10
#: Mass agreement between the environment ligand and package A (amu).
MASS_TOL_AMU = 1e-6

DEFAULT_DUAL_RESTRAINT_K = 1000.0   # kJ/mol/nm^2

#: What an environment may be. The build record's `solvent.treatment` spells it.
SOLVATIONS = ("explicit", "implicit", "vacuum")

#: OpenMM's 1/(4 pi eps0) in kJ mol^-1 nm e^-2, as NonbondedForce evaluates an exception (measured
#: on the Reference platform: a unit-charge exception at 0.5 nm gives exactly half of it).
COULOMB_CONSTANT = 138.93545764438198
#: The force carrying a unique group's non-excluded internal pairs at its dummy end.
INTERNAL_FORCE_NAME = "UniqueGroupInternalNonbonded"


class TopologyError(ValueError):
    """A plan that cannot be built without misrepresenting an endpoint."""


def _canonical(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _positions_sha256(positions_nm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions_nm, dtype="<f8").tobytes()).hexdigest()


def _clone(obj):
    from openmm import XmlSerializer

    return XmlSerializer.clone(obj)


def _q(value) -> float:
    from openmm import unit

    if hasattr(value, "value_in_unit_system"):
        value = value.value_in_unit_system(unit.md_unit_system)
    return float(value)


# ------------------------------------------------------------------------------------------------
# the environment
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Environment:
    """ONE built System with endpoint A in it, and the residue that is endpoint A.

    Endpoint B is never given its own environment: two independently solvated boxes are not
    atom-matched, and treating them as if they were builds a plan that runs and is wrong.
    """

    system: Any
    topology: Any
    positions_nm: np.ndarray
    ligand: LigandSelector
    source: dict = field(default_factory=dict)
    #: The build's `nonbonded_compatibility` block: each package's own 1-4 scales beside the ones
    #: the System APPLIES (OpenMM applies its first NonbondedForce definition to every 1-4 pair,
    #: and the OPC water XMLs write 5/6 as 0.833333). Read from the build record, or stated by
    #: whoever built an in-memory System; never inferred. None refuses the plan.
    nonbonded_compatibility: Optional[dict] = None
    compatibility_source: Optional[str] = None
    #: "explicit", "implicit" or "vacuum": what the environment IS, from the build record's
    #: `solvent.treatment`, or stated by whoever built an in-memory System. Never inferred from
    #: periodicity. A vacuum leg and a solvated leg are different plans, so it is in plan_sha256;
    #: the window runner derives `vacuum_leg` from it.
    solvation: Optional[str] = None

    @classmethod
    def from_files(cls, system_xml: Path, pdb: Path, ligand: LigandSelector, *,
                   record: Path) -> "Environment":
        """A `build-top` pair, `built.xml` and `built.pdb`, and the build record that made them.

        The record (`built.log`) must describe THESE files -- its `outputs` sha256 of both are
        checked -- and must carry `forcefield_record.ligand.nonbonded_compatibility`.
        """
        from openmm import XmlSerializer, unit
        from openmm.app import PDBFile

        from ..build.record import RecordError, read_record

        system_xml, pdb, record = Path(system_xml), Path(pdb), Path(record)
        digests = {"system_xml": hashlib.sha256(system_xml.read_bytes()).hexdigest(),
                   "topology_pdb": hashlib.sha256(pdb.read_bytes()).hexdigest()}
        try:
            document = read_record(record)
        except RecordError as exc:
            raise TopologyError(f"environment build record: {exc}") from exc
        if document.get("record_type") != "build-top" or document.get("status") != "completed":
            raise TopologyError(f"{record}: not a completed build-top record "
                                f"({document.get('record_type')}, {document.get('status')})")
        outputs = document.get("outputs") or {}
        for key, digest in digests.items():
            recorded = (outputs.get(key) or {}).get("sha256")
            if recorded != digest:
                raise TopologyError(
                    f"{record} records {key} sha256 {recorded}, but the file given hashes to "
                    f"{digest}: the record describes another build")
        solvation = (document.get("solvent") or {}).get("treatment")
        if solvation not in SOLVATIONS:
            raise TopologyError(f"{record} records solvent.treatment {solvation!r}, not one of "
                                f"{SOLVATIONS}")
        compatibility = ((document.get("forcefield_record") or {}).get("ligand") or {}).get(
            "nonbonded_compatibility")
        if not compatibility:
            raise TopologyError(
                f"{record} carries no forcefield_record.ligand.nonbonded_compatibility, so it does "
                f"not say which 1-4 scales the System applies to the ligand. Rebuild the "
                f"environment with a build-top that records it (0.6.0 or later).")
        text = system_xml.read_text(encoding="utf-8")
        structure = PDBFile(str(pdb))
        positions = np.array(structure.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                             dtype=float)
        return cls(system=XmlSerializer.deserialize(text), topology=structure.topology,
                   positions_nm=positions, ligand=ligand,
                   source={"system_file": system_xml.name,
                           "system_file_sha256": digests["system_xml"],
                           "pdb_file": pdb.name,
                           "pdb_file_sha256": digests["topology_pdb"],
                           "record_file": record.name,
                           "record_file_sha256": hashlib.sha256(record.read_bytes()).hexdigest()},
                   nonbonded_compatibility=compatibility,
                   compatibility_source=f"build record {record.name}", solvation=solvation)

    def identity(self) -> dict[str, Any]:
        from openmm import XmlSerializer

        from ..rest2.selection import topology_digest

        return {"system_sha256": _sha256_text(XmlSerializer.serialize(self.system)),
                "topology_sha256": topology_digest(self.topology),
                "positions_sha256": _positions_sha256(self.positions_nm),
                "n_particles": self.system.getNumParticles(),
                "n_residues": self.topology.getNumResidues(),
                **self.source}


def _ligand_residue(environment: Environment, package: LigandPackage):
    residues = environment.ligand.residues(environment.topology)
    if len(residues) != 1:
        raise TopologyError(f"the ligand selector {environment.ligand.label()} names "
                            f"{len(residues)} residues; endpoint A is ONE instance")
    residue = residues[0]
    atoms = list(residue.atoms())
    names = tuple(a.name for a in atoms)
    if names != tuple(package.atom_names):
        raise TopologyError(f"residue {residue.name} {residue.id} carries atoms {names}, not "
                            f"package {package.reference}'s {tuple(package.atom_names)} in "
                            f"package order; it is not endpoint A")
    elements = [a.element.symbol for a in atoms]
    if elements != [a.GetSymbol() for a in package.mol.GetAtoms()]:
        raise TopologyError(f"residue {residue.name} {residue.id}: elements {elements} are not "
                            f"package {package.reference}'s")
    return residue, [a.index for a in atoms]


def _forces_by_name(system) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for k in range(system.getNumForces()):
        out.setdefault(type(system.getForce(k)).__name__, []).append(k)
    return out


def applied_scales(environment: Environment, package: LigandPackage) -> dict[str, Any]:
    """The 1-4 scales the environment's System applies to *package*, from its record; or refuse."""
    compatibility = environment.nonbonded_compatibility
    if not compatibility:
        raise TopologyError(
            "the environment does not say which 1-4 scales its System applies to the ligand "
            "(no nonbonded_compatibility). Give Environment.from_files the build record, or state "
            "it for an in-memory System; it is never inferred.")
    entries = [e for e in compatibility.get("packages", []) if e.get("reference") ==
               package.reference]
    if len(entries) != 1 or "applied" not in entries[0]:
        raise TopologyError(f"the environment's nonbonded_compatibility has no applied scales for "
                            f"package {package.reference}")
    applied = entries[0]["applied"]
    return {"coulomb14scale": float(applied["coulomb14scale"]),
            "lj14scale": float(applied["lj14scale"]),
            "source": environment.compatibility_source}


def scaled_table(package: LigandPackage, applied: dict[str, Any]) -> dict[str, Any]:
    """*package*'s parameter table with its 1-4 exceptions at the scales the System APPLIES.

    A package states its exceptions at its own convention (5/6 and 1/2 exactly). A System whose
    first NonbondedForce definition is OPC's applies 0.833333 to every 1-4 pair, the ligand's
    included, so the ligand in that System differs from the package table by ~4e-7 relative in
    each 1-4 chargeProd -- a fact about the environment, not a different parameter set. Every
    exception row is multiplied by applied/package: excluded pairs are zero and stay zero, and 1-4
    pairs are exactly the rows the convention scales. With equal scales the table is unchanged.
    """
    from ..ligands.parameters import copy_table

    table = copy_table(package.table)
    own = package.conventions
    fq = applied["coulomb14scale"] / own["coulomb14scale"]
    fl = applied["lj14scale"] / own["lj14scale"]
    if fq != 1.0 or fl != 1.0:
        table["exceptions"] = [[i, j, qq * fq, sigma, eps * fl]
                               for i, j, qq, sigma, eps in table["exceptions"]]
    table["conventions"] = {**own, "coulomb14scale": applied["coulomb14scale"],
                            "lj14scale": applied["lj14scale"]}
    return table


def _check_environment(environment: Environment, package_a: LigandPackage,
                       ligand_indices: list[int], checks: list) -> dict[str, Any]:
    """Endpoint A in the environment IS package A: parameters, masses, constraints, forces."""
    system = environment.system
    ligand = set(ligand_indices)
    forces = _forces_by_name(system)
    unknown = sorted(set(forces) - set(LIGAND_FORCES) - set(PASSIVE_FORCES))
    if unknown:
        raise TopologyError(f"the environment System carries {unknown}; 0.7.0 construction "
                            f"supports {LIGAND_FORCES} on the ligand and {PASSIVE_FORCES} "
                            f"elsewhere. Implicit solvent and custom forces are refused rather "
                            f"than carried unaltered through an alchemical change.")
    for name in LIGAND_FORCES:
        if len(forces.get(name, [])) != 1:
            raise TopologyError(f"the environment System has {len(forces.get(name, []))} "
                                f"{name}; exactly one is required")
    for k in forces.get("CMAPTorsionForce", []):
        cmap = system.getForce(k)
        for t in range(cmap.getNumTorsions()):
            if ligand & set(cmap.getTorsionParameters(t)[1:]):
                raise TopologyError("a CMAP term touches the ligand; refused")
    nb = system.getForce(forces["NonbondedForce"][0])
    if nb.getNumGlobalParameters() or nb.getNumParticleParameterOffsets() \
            or nb.getNumExceptionParameterOffsets():
        raise TopologyError("the environment NonbondedForce already has global parameters or "
                            "parameter offsets: a scaled or alchemical System is not an "
                            "environment")
    if any(system.isVirtualSite(i) for i in ligand_indices):
        raise TopologyError("a ligand atom is a virtual site; refused")

    applied = applied_scales(environment, package_a)
    expected_table = scaled_table(package_a, applied)
    c14, lj14 = applied["coulomb14scale"], applied["lj14scale"]
    try:
        table, constrained = subsystem_parameter_table(system, package_a.mol, ligand_indices,
                                                       c14, lj14)
    except ValueError as exc:
        raise TopologyError(f"the environment ligand cannot be read as package "
                            f"{package_a.reference}: {exc}") from exc
    problems = compare_parameter_tables(expected_table, table, where="environment ligand",
                               skip_masses=True, missing_bonds_ok=table["_constraint_lengths"],
                               rel=PARAMETER_RTOL)
    if problems:
        raise TopologyError(
            f"the environment's endpoint-A ligand does not carry package {package_a.reference}'s "
            f"parameters ({len(problems)} differences, first: {problems[:3]}). Registered "
            f"parameters are immutable; a plan is built only on the package as registered.")
    masses = [_q(system.getParticleMass(i)) for i in ligand_indices]
    expected = [row[1] for row in package_a.table["atoms"]]
    changed = [i for i, (m, e) in enumerate(zip(masses, expected)) if abs(m - e) > MASS_TOL_AMU]
    if changed:
        raise TopologyError(
            f"ligand masses differ from the package's for atoms "
            f"{[package_a.atom_names[i] for i in changed]} (hydrogen mass repartitioning?). "
            f"Appended endpoint-B atoms would need the same repartitioning, and 0.7.0 "
            f"construction does not apply it; build the environment without HMR.")

    bonds = _bonds(package_a.mol)
    hydrogen = {i for i, a in enumerate(package_a.mol.GetAtoms()) if a.GetAtomicNum() == 1}
    h_bonds = {b for b in bonds if hydrogen & set(b)}
    pairs = set(constrained)
    local = {p: i for i, p in enumerate(ligand_indices)}
    for k in range(system.getNumConstraints()):
        i, j, _ = system.getConstraintParameters(k)
        if (i in ligand) != (j in ligand):
            raise TopologyError("a constraint joins the ligand to another particle; refused")
        if i in ligand and tuple(sorted((local[i], local[j]))) not in bonds:
            raise TopologyError("the environment constrains a ligand atom pair that is not a "
                                "bond (HAngles?); only bond constraints are supported")
    if pairs and pairs == h_bonds == bonds:
        # every bond of this ligand is an X-H bond, so HBonds and AllBonds constrain the same
        # pairs here and the environment cannot say which policy it was built under
        policy = "HBonds-or-AllBonds"
    elif pairs == h_bonds:
        policy = "HBonds"
    elif pairs == bonds:
        policy = "AllBonds"
    elif not pairs:
        policy = "None"
    else:
        raise TopologyError(f"the ligand's constrained bonds {sorted(pairs)} are not HBonds, "
                            f"AllBonds or None; the policy cannot be applied to endpoint B")
    _check(checks, "environment-carries-package-a",
           {"reference": package_a.reference, "parameter_rtol": PARAMETER_RTOL,
            "masses": "package masses (no repartitioning)", "constraint_policy": policy,
            "applied_1_4_scales": applied,
            "package_1_4_scales": {k: package_a.conventions[k]
                                   for k in ("coulomb14scale", "lj14scale")},
            "compared_against": "package A's table with its 1-4 exceptions at the applied "
                                "scales"})

    if environment.solvation not in SOLVATIONS:
        raise TopologyError(
            f"the environment does not state its solvation ({environment.solvation!r}); it is "
            f"read from the build record, or stated as one of {SOLVATIONS} for an in-memory "
            f"System. It is never inferred from periodicity: a vacuum leg and an implicit-solvent "
            f"System are both non-periodic.")
    if environment.solvation == "explicit" and not system.usesPeriodicBoundaryConditions():
        raise TopologyError("the environment says explicit solvent but its System is not periodic")
    if environment.solvation != "explicit" and system.usesPeriodicBoundaryConditions():
        raise TopologyError(f"the environment says {environment.solvation} but its System is "
                            f"periodic")

    method = nb.getNonbondedMethod()
    return {
        "solvation": environment.solvation,
        "constraint_policy": policy,
        "nonbonded_applied": applied,
        "forces": sorted(forces),
        "periodic": system.usesPeriodicBoundaryConditions(),
        "nonbonded": {
            "method": int(method),
            "cutoff_nm": _q(nb.getCutoffDistance()),
            "ewald_error_tolerance": nb.getEwaldErrorTolerance(),
            "dispersion_correction": bool(nb.getUseDispersionCorrection()),
            "switching": bool(nb.getUseSwitchingFunction()),
            "switching_distance_nm": _q(nb.getSwitchingDistance()),
        },
    }


def _check(checks: list, name: str, detail: Any) -> None:
    checks.append({"check": name, "status": "pass", "detail": detail})


# ------------------------------------------------------------------------------------------------
# dummy junctions
# ------------------------------------------------------------------------------------------------
def _junction(mol, component: dict[str, Any], core: set[int]) -> dict[str, Any]:
    """The frame (P1, P2, P3) a dummy group is retained against, in its own package's indices."""
    neighbours = _neighbours(mol)
    heavy_first = (lambda i: (mol.GetAtomWithIdx(i).GetAtomicNum() == 1, i))
    (d1, p1), = component["attachments"]
    group = set(component["atoms"])
    p2_options = sorted((n for n in neighbours[p1] if n in core and n not in group),
                        key=heavy_first)
    p2 = p2_options[0] if p2_options else None
    p3 = None
    if p2 is not None:
        p3_options = sorted((n for n in neighbours[p2] if n in core and n != p1), key=heavy_first)
        p3 = p3_options[0] if p3_options else None
    return {"d1": d1, "p1": p1, "p2": p2, "p3": p3, "group": sorted(group)}


def _retained(atoms: Sequence[int], kind: str, junction: dict[str, Any]) -> bool:
    """Whether a term over package-local *atoms*, touching the group, is kept at its dummy end."""
    allowed = set(junction["group"]) | {junction["p1"]}
    if junction["p2"] is not None:
        allowed.add(junction["p2"])
    if set(atoms) <= allowed:
        return True
    frame = (junction["d1"], junction["p1"], junction["p2"], junction["p3"])
    return kind == "proper" and None not in frame and tuple(atoms) in (frame, frame[::-1])


# ------------------------------------------------------------------------------------------------
# the plan
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TopologyPlan:
    """The record, and the two endpoint Systems over one particle index space."""

    record: dict
    system_a: Any = field(repr=False)
    system_b: Any = field(repr=False)
    topology: Any = field(repr=False)
    positions_nm: np.ndarray = field(repr=False)
    #: The exact `combined.pdb` text `topology` was read from. `write` writes these bytes rather
    #: than re-serialising: an OpenMM PDB write/read round trip reorders bonds between two
    #: HETATM residues on every pass, so a re-written file would read back with another digest.
    pdb_text: Optional[str] = field(default=None, repr=False)

    @property
    def mode(self) -> str:
        return self.record["mode"]

    @property
    def common(self) -> frozenset:
        return frozenset(self.record["particles"]["common"])

    @property
    def a_only(self) -> frozenset:
        return frozenset(self.record["particles"]["a_only"])

    @property
    def b_only(self) -> frozenset:
        return frozenset(self.record["particles"]["b_only"])

    def dummies(self, endpoint: str) -> frozenset:
        """The particles that are dummies at *endpoint* ('A' or 'B')."""
        return {"A": self.b_only, "B": self.a_only}[_endpoint(endpoint)]

    def system(self, endpoint: str):
        return {"A": self.system_a, "B": self.system_b}[_endpoint(endpoint)]

    @property
    def sha256(self) -> str:
        return self.record["plan_sha256"]

    def write(self, directory: Path) -> Path:
        """Write the plan into a NEW directory, staged and renamed into place; never overwritten."""
        from openmm import XmlSerializer
        from openmm.app import PDBFile

        directory = Path(directory)
        if directory.exists():
            raise TopologyError(f"{directory} exists; a plan is written once, into a new "
                                f"directory")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
        blobs = {
            "system_a.xml": XmlSerializer.serialize(self.system_a).encode(),
            "system_b.xml": XmlSerializer.serialize(self.system_b).encode(),
        }
        if self.pdb_text is None:
            pdb = io.StringIO()
            PDBFile.writeFile(self.topology, self.positions_nm * 10.0, pdb, keepIds=True)
            text = pdb.getvalue()
        else:
            text = self.pdb_text
        blobs["combined.pdb"] = text.encode()
        npy = io.BytesIO()
        np.save(npy, np.ascontiguousarray(self.positions_nm, dtype="<f8"), allow_pickle=False)
        blobs["positions.npy"] = npy.getvalue()
        files = {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()}
        document = {**self.record, "files": files}
        blobs["plan.json"] = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()
        try:
            for name, data in blobs.items():
                with open(staging / name, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.rename(staging, directory)
        except BaseException:
            for name in blobs:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
            raise
        return directory


def _at(record: dict, path: str):
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_difference(stored: Any, current: Any, path: str = "") -> Optional[str]:
    """The first field whose value differs, as a dotted path. Depth-first, keys sorted."""
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in sorted(set(stored) | set(current)):
            if key in ("plan_sha256", "files"):
                continue
            where = f"{path}.{key}" if path else key
            if key not in stored or key not in current:
                return f"{where} (only in {'the stored plan' if key in stored else 'this plan'})"
            deeper = _first_difference(stored[key], current[key], where)
            if deeper:
                return deeper
        return None
    return None if stored == current else (path or "the record")


def check_plan_digest(stored_digest: str, stored_record: dict, current: "TopologyPlan") -> None:
    """Refuse a stored plan digest that this plan does not reproduce, saying WHICH case it is.

    A stored `plan_sha256` stops matching for two very different reasons, and a caller holding a
    leg or a campaign row needs to be told which:

    * the RECORD's schema changed and the plan's identity did not -- same ligand Hamiltonian, same
      endpoints, same map -- so the physics is unchanged and the plan only needs rebuilding;
    * anything else: this is a different plan, and the first differing field is named.
    """
    if stored_digest == current.sha256:
        return
    record = current.record
    same_identity = all(_at(stored_record, field) == _at(record, field)
                        for field in PLAN_IDENTITY_FIELDS)
    stored_schema, schema = stored_record.get("schema"), record.get("schema")
    if same_identity and stored_schema != schema:
        raise TopologyError(
            f"the plan record's schema changed from {stored_schema!r} to {schema!r}; the ligand "
            f"Hamiltonian and endpoints are unchanged, so rebuild the plan and re-prepare -- the "
            f"physics is the same. Stored plan_sha256 {stored_digest[:16]}..., this plan "
            f"{current.sha256[:16]}...")
    if same_identity:
        raise TopologyError(
            f"the stored plan_sha256 {stored_digest[:16]}... is not this plan's "
            f"{current.sha256[:16]}..., though the ligand Hamiltonian, endpoints and map are the "
            f"same: the environment or another recorded input differs. First field that differs: "
            f"{_first_difference(stored_record, record) or 'none found'}.")
    raise TopologyError(
        f"the stored plan_sha256 {stored_digest[:16]}... is not this plan's "
        f"{current.sha256[:16]}..., and they are different plans. First field that differs: "
        f"{_first_difference(stored_record, record) or 'none found'}.")


def _endpoint(endpoint: str) -> str:
    label = str(endpoint).upper()
    if label not in ("A", "B"):
        raise ValueError(f"endpoint is 'A' or 'B', not {endpoint!r}")
    return label


def _record_digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k not in ("plan_sha256", "files")}
    return _sha256_text(_canonical(body))


def load_plan(directory: Path, *, package_roots: Iterable[Path] = ()) -> TopologyPlan:
    """Read a plan and verify every file digest and the record's own digest.

    With *package_roots*, the two endpoint packages are also located, loaded (which re-verifies
    them) and checked against the identities the plan recorded.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    directory = Path(directory)
    missing = [name for name in PLAN_FILES if not (directory / name).is_file()]
    if missing:
        raise TopologyError(f"{directory}: not a topology plan (missing {missing})")
    extra = sorted(p.name for p in directory.iterdir() if p.name not in PLAN_FILES)
    if extra:
        raise TopologyError(f"{directory}: unexpected files {extra}")
    record = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if record.get("schema") != PLAN_SCHEMA:
        raise TopologyError(
            f"{directory}: plan schema {record.get('schema')!r} is not {PLAN_SCHEMA!r}. A record "
            f"written under another schema cannot reproduce its own plan_sha256, so it is "
            f"rebuilt rather than read: `combine-topology` again with the same configuration. "
            f"Its identity is unchanged if ligand_hamiltonian_sha256 "
            f"({_at(record, 'ligand_hamiltonian_sha256') or 'absent'}), the endpoints and the map "
            f"digest match the rebuilt plan's; `check_plan_digest` says which case it is.")
    for name, digest in record["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise TopologyError(f"{directory / name}: sha256 {actual} does not match the plan "
                                f"({digest}); the file was modified")
    if _record_digest(record) != record["plan_sha256"]:
        raise TopologyError(f"{directory}/plan.json: the record does not hash to its recorded "
                            f"plan_sha256; it was edited")
    record = {k: v for k, v in record.items() if k != "files"}
    for side in ("A", "B"):
        roots = list(package_roots)
        if not roots:
            break
        endpoint = record["endpoints"][side]
        compound, parameter = endpoint["reference"].split("/")
        found = [Path(r) / compound / parameter for r in roots
                 if (Path(r) / compound / parameter).is_dir()]
        if not found:
            raise TopologyError(f"package {endpoint['reference']} is in none of {roots}")
        package = load_package(found[0])
        if package.summary()["package_sha256"] != endpoint["package_sha256"] or \
                package.metadata["parameter_digest"] != endpoint["parameter_digest"]:
            raise TopologyError(f"package {endpoint['reference']} at {found[0]} is not the one "
                                f"this plan was built from")
    positions = np.load(directory / "positions.npy", allow_pickle=False)
    pdb_text = (directory / "combined.pdb").read_text(encoding="utf-8")
    topology = PDBFile(io.StringIO(pdb_text)).topology
    from ..rest2.selection import topology_digest

    if topology_digest(topology) != record["numbering"]["combined_topology_sha256"]:
        raise TopologyError(f"{directory}/combined.pdb does not read back as the topology the "
                            f"plan recorded")
    if positions.shape != (record["particles"]["n_total"], 3):
        raise TopologyError(f"{directory}/positions.npy has shape {positions.shape}")
    return TopologyPlan(
        record=record,
        system_a=XmlSerializer.deserialize((directory / "system_a.xml").read_text("utf-8")),
        system_b=XmlSerializer.deserialize((directory / "system_b.xml").read_text("utf-8")),
        topology=topology, positions_nm=positions, pdb_text=pdb_text)


# ------------------------------------------------------------------------------------------------
# construction
# ------------------------------------------------------------------------------------------------
def _kabsch(source: np.ndarray, target: np.ndarray):
    """Rotation R and translation so that (source - cs) @ R.T + ct best fits target."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cs, ct


def _torsion_key(quartet: Sequence[int], kind: str) -> tuple:
    q = tuple(quartet)
    return min(q, q[::-1]) if kind == "proper" else q


def _classify(quartet: Sequence[int], bonds: set) -> str:
    i, j, k, l = quartet
    chain = all(tuple(sorted(p)) in bonds for p in ((i, j), (j, k), (k, l)))
    return "proper" if chain else "improper"


def _standard_state_restraint(restraint: Optional[dict], absent_b: bool,
                              environment: Environment, hyb_a: list, n_env: int,
                              package_a: LigandPackage, checks: list) -> list:
    """Record the restraint a decoupling leg is run with, or refuse a leg that needs one.

    The plan does not build it: a Boresch restraint, its free energy and the standard-state
    correction are the executor's. What the plan owes a reader is that it EXISTS and which atoms
    it holds -- a decoupled ligand with no restraint wanders out of the site and the integral
    diverges, and nothing downstream could tell afterwards.
    """
    from ..md.stage import SOLVENT_RESIDUES

    if restraint is None and not absent_b:
        return []
    if restraint is not None and not absent_b:
        raise TopologyError("a standard-state restraint belongs to a decoupling plan; the other "
                            "modes transform one real molecule into another")
    ligand_residue = {atom.residue.index for atom in environment.topology.atoms()
                      if atom.index in set(hyb_a)}
    protein = sorted({residue.name for residue in environment.topology.residues()
                      if residue.index not in ligand_residue
                      and residue.name.strip().upper() not in SOLVENT_RESIDUES})
    if restraint is None:
        if protein:
            raise TopologyError(
                f"this environment holds {len(protein)} non-solvent residue kind(s) besides the "
                f"ligand ({', '.join(protein[:6])}...), so a decoupling leg in it needs a "
                f"standard-state restraint: a decoupled ligand with none wanders out of the site "
                f"and the free energy diverges. Pass restraint={{'kind': 'boresch', "
                f"'ligand_atoms': [...], 'environment_atoms': [...]}}; the executor builds the "
                f"force, computes its free energy and applies the standard-state correction.")
        _check(checks, "standard-state-restraint",
               {"required": False, "why": "no binding site to leave: the environment holds only "
                                          "the ligand and solvent"})
        return []
    unknown = sorted(set(restraint) - {"kind", "ligand_atoms", "environment_atoms", "parameters"})
    if unknown:
        raise TopologyError(f"unknown key(s) {unknown} in the restraint record")
    ligand_atoms = [int(i) for i in restraint.get("ligand_atoms") or []]
    environment_atoms = [int(i) for i in restraint.get("environment_atoms") or []]
    if not ligand_atoms or not environment_atoms:
        raise TopologyError("a standard-state restraint names ligand_atoms and environment_atoms")
    ligand = set(hyb_a)
    if not set(ligand_atoms) <= ligand:
        raise TopologyError(f"restraint ligand_atoms {sorted(set(ligand_atoms) - ligand)} are not "
                            f"the ligand's particles")
    if set(environment_atoms) & ligand or max(environment_atoms) >= n_env:
        raise TopologyError("restraint environment_atoms must be particles of the environment, "
                            "not the ligand")
    by_index = {atom.index: atom for atom in environment.topology.atoms()}
    local = {h: i for i, h in enumerate(hyb_a)}
    record = {
        "role": "standard-state",
        "kind": str(restraint.get("kind") or "boresch"),
        "built_by": "the executor (md_tools.alchemy.restraints): the plan records it, and does "
                    "not build it",
        "ligand_atoms": ligand_atoms,
        "ligand_atom_names": [package_a.atom_names[local[h]] for h in ligand_atoms],
        "environment_atoms": environment_atoms,
        # Named for review: an index means nothing to a person checking a binding site.
        "environment_atom_labels": [
            f"{by_index[h].residue.chain.id}:{by_index[h].residue.id}"
            f"{by_index[h].residue.name}:{by_index[h].name}" for h in environment_atoms],
        "parameters": restraint.get("parameters"),
    }
    _check(checks, "standard-state-restraint",
           {"required": bool(protein), "kind": record["kind"],
            "ligand_atoms": record["ligand_atom_names"],
            "environment_atoms": record["environment_atom_labels"]})
    return [record]


def build_decoupling_plan(package: LigandPackage, environment: Environment, *,
                          restraint: Optional[dict] = None) -> TopologyPlan:
    """The plan an absolute-binding leg needs: endpoint B is the ligand ABSENT.

    At lambda 0 this is the environment as built. At lambda 1 the ligand interacts with nothing
    outside itself -- every particle charge 0 and epsilon 0, every exception to the environment
    zero -- while its OWN Hamiltonian is retained unchanged: bonded terms, internal exceptions and
    internal pairs, physical at both ends. That is what the standard-state correction assumes, and
    it makes the ligand's intramolecular Hamiltonian lambda-independent.

    The whole ligand is the unique region, so the softcore covers exactly the ligand-environment
    pairs. `common` and `b_only` are empty: `common` here means "ligand particles physical at
    both endpoints", which is not the set a Hamiltonian computes as "everything that is not
    unique" -- those two are different by design.

    A ligand with a non-zero net formal charge is refused: decoupling it changes the box's net
    charge, and PME's neutralising background then contributes a free energy that needs a
    finite-size correction (Rocklin et al.) this release does not implement.
    """
    net = package.metadata["chemical_state"]["net_formal_charge"]
    if net != 0:
        raise TopologyError(
            f"{package.reference} has net formal charge {net}: decoupling it changes the box's "
            f"net charge, and PME's neutralising background then contributes a free energy that "
            f"needs a finite-size correction (Rocklin et al.) this release does not implement. "
            f"Not impossible -- not implemented.")
    return build_topology_plan(package, None, None, environment, mode="decoupling",
                               restraint=restraint)


def build_topology_plan(package_a: LigandPackage, package_b: Optional[LigandPackage],
                        atom_map: Optional[AtomMap], environment: Environment, *, mode: str,
                        dual_restraint_k: float = DEFAULT_DUAL_RESTRAINT_K,
                        b_positions_nm: Optional[np.ndarray] = None,
                        restraint: Optional[dict] = None) -> TopologyPlan:
    """Build and check the plan. Nothing is written; `TopologyPlan.write` writes it.

    *b_positions_nm* is endpoint B's pose in B package order (a docked pose, for example). Left
    None, package B's reference conformer is superposed on the mapped A atoms, which is what a
    reference conformer is for, and the core RMSD of that fit is recorded.

    In `mode: "decoupling"` there is no endpoint B and no map: pass `package_b=None` and
    `atom_map=None`, and `build_decoupling_plan` is the entry point that does.
    """
    from openmm import CustomBondForce, CustomCentroidBondForce, version
    from openmm.app import Element, Topology

    absent_b = mode == "decoupling"
    if absent_b:
        if package_b is not None or atom_map is not None:
            raise TopologyError("mode 'decoupling' has no endpoint B: package_b and atom_map "
                                "must be None")
        package_b = package_a          # every "B" table lookup below is over the empty B set
        map_report = {"mode": mode, "checks": [], "unique_components": {"A": [], "B": []},
                      "notes": {"decoupling": "no map: endpoint B is the ligand ABSENT"}}
    else:
        if package_b is None or atom_map is None:
            raise TopologyError(f"mode {mode!r} needs an endpoint B package and an atom map")
        try:
            map_report = validate_map(package_a, package_b, atom_map, mode)
        except MapError as exc:
            raise TopologyError(str(exc)) from exc
    checks: list[dict[str, Any]] = list(map_report["checks"])
    residue_a, ligand_env = _ligand_residue(environment, package_a)
    env_info = _check_environment(environment, package_a, ligand_env, checks)
    env_system = environment.system
    n_env = env_system.getNumParticles()
    dual = mode == "dual"
    #: Everything a whole-molecule dummy needs: no junction, every bonded term retained.
    whole_molecule_dummy = dual or absent_b

    a_to_b, b_to_a = ({}, {}) if whole_molecule_dummy else (atom_map.a_to_b, atom_map.b_to_a)
    n_a, n_b = len(package_a.atom_names), (0 if absent_b else len(package_b.atom_names))
    hyb_a = list(ligand_env)
    appended = [b for b in range(n_b) if b not in b_to_a]
    hyb_b = [0] * n_b
    for k, b in enumerate(appended):
        hyb_b[b] = n_env + k
    for b, a in b_to_a.items():
        hyb_b[b] = hyb_a[a]
    common = sorted(hyb_a[a] for a in a_to_b)
    a_only = sorted(hyb_a[a] for a in range(n_a) if a not in a_to_b)
    b_only = [hyb_b[b] for b in appended]
    dummy_at = {"A": set(b_only), "B": set(a_only)}
    n_total = n_env + len(appended)

    # -- positions ------------------------------------------------------------------------------
    mapped_pairs = [] if absent_b else sorted(atom_map.pairs)
    xa = environment.positions_nm[[hyb_a[a] for a, _ in mapped_pairs]]
    if absent_b:
        # Nothing to place: endpoint B adds no particle, and every coordinate is the
        # environment's own.
        placed = np.zeros((0, 3))
        placement = {"method": "none: endpoint B is the ligand absent"}
        rmsd = 0.0
    elif b_positions_nm is None:
        source_b = _positions(package_b) / 10.0
        rot, cs, ct = _kabsch(source_b[[b for _, b in mapped_pairs]], xa)
        placed = (source_b - cs) @ rot.T + ct
        placement = {"method": "package B reference conformer superposed on the mapped A atoms "
                               "(Kabsch)", "conformer_sha256":
                     package_b.metadata["artifacts"]["molecule.sdf"]["sha256"]}
    else:
        placed = np.asarray(b_positions_nm, dtype=float)
        if placed.shape != (n_b, 3):
            raise TopologyError(f"b_positions_nm must be ({n_b}, 3) in B package order")
        placement = {"method": "given", "positions_sha256": _positions_sha256(placed)}
    if not absent_b:
        rmsd = float(np.sqrt(np.mean(np.sum((placed[[b for _, b in mapped_pairs]] - xa) ** 2,
                                            axis=1))))
    placement["mapped_rmsd_nm"] = rmsd
    positions = np.vstack([environment.positions_nm, placed[appended]]) if appended \
        else environment.positions_nm.copy()

    # -- masses ---------------------------------------------------------------------------------
    masses = [_q(env_system.getParticleMass(i)) for i in range(n_env)] + \
             [package_b.table["atoms"][b][1] for b in appended]
    mass_changes = [{"particle": hyb_a[a], "a": package_a.table["atoms"][a][1],
                     "b": package_b.table["atoms"][b][1]}
                    for a, b in sorted(a_to_b.items())
                    if package_a.table["atoms"][a][1] != package_b.table["atoms"][b][1]]

    # -- dummy groups and their frames ----------------------------------------------------------
    groups = []
    junction_of: dict[tuple[str, int], dict] = {}   # (source side, local atom) -> junction
    if not whole_molecule_dummy:
        for side, package, components, core in (
                ("A", package_a, map_report["unique_components"]["A"], set(a_to_b)),
                ("B", package_b, map_report["unique_components"]["B"], set(b_to_a))):
            to_hyb = hyb_a if side == "A" else hyb_b
            for component in components:
                junction = _junction(package.mol, component, core)
                for atom in junction["group"]:
                    junction_of[(side, atom)] = junction
                groups.append({
                    "source_endpoint": side,
                    "dummy_at": "B" if side == "A" else "A",
                    "atoms": [to_hyb[i] for i in junction["group"]],
                    "local_atoms": junction["group"],
                    "local_names": [package.atom_names[i] for i in junction["group"]],
                    "anchor": {"d1": to_hyb[junction["d1"]], "p1": to_hyb[junction["p1"]]},
                    "frame": {k: (None if junction[k] is None else to_hyb[junction[k]])
                              for k in ("p1", "p2", "p3")},
                    "rule": "retain terms within G+{P1,P2} and the torsion D1-P1-P2-P3; remove "
                            "every other term touching G at the endpoint where G is a dummy",
                })

    # -- constraints ----------------------------------------------------------------------------
    policy = env_info["constraint_policy"]
    hydrogen_b = {i for i, a in enumerate(package_b.mol.GetAtoms()) if a.GetAtomicNum() == 1}

    if policy == "HBonds-or-AllBonds":
        if any(not (hydrogen_b & set(b)) for b in _bonds(package_b.mol)):
            raise TopologyError(
                "every bond of the environment ligand is an X-H bond, so the environment does not "
                "say whether it was built with HBonds or AllBonds, and endpoint B has bonds "
                "between heavy atoms whose treatment depends on which. Build the environment "
                "from the other endpoint, or from a ligand with a heavy-atom bond.")
        policy = "HBonds"

    def b_constrained(i: int, j: int) -> bool:
        if policy == "AllBonds":
            return True
        if policy == "HBonds":
            return bool(hydrogen_b & {i, j})
        return False

    env_constraints = {}
    for k in range(env_system.getNumConstraints()):
        i, j, d = env_system.getConstraintParameters(k)
        env_constraints[tuple(sorted((i, j)))] = _q(d)
    b_bond_rows = {} if absent_b else {tuple(sorted((r[0], r[1]))): r
                                       for r in package_b.table["bonds"]}
    added_constraints = []
    for (i, j), row in sorted(b_bond_rows.items()):
        key = tuple(sorted((hyb_b[i], hyb_b[j])))
        in_env = key in env_constraints
        wants = b_constrained(i, j)
        if key[0] < n_env and key[1] < n_env:   # a core bond: it exists in the environment
            if in_env != wants:
                raise TopologyError(
                    f"bond {package_b.atom_names[i]}-{package_b.atom_names[j]} is "
                    f"{'constrained' if in_env else 'flexible'} at A and "
                    f"{'constrained' if wants else 'flexible'} at B under the {policy} policy; "
                    f"a constraint cannot appear or vanish along the path")
            if in_env and abs(env_constraints[key] - row[2]) > CONSTRAINT_TOL_NM:
                raise TopologyError(
                    f"the constrained bond {package_b.atom_names[i]}-{package_b.atom_names[j]} is "
                    f"{env_constraints[key]!r} nm at A and {row[2]!r} nm at B. A constraint has "
                    f"one length; either endpoint's length would misrepresent the other.")
        elif wants:
            added_constraints.append([key[0], key[1], row[2]])
    constraint_record = {
        "policy": policy,
        "environment_constraints": len(env_constraints),
        "appended": added_constraints,
        "identical_in_both_systems": True,
    }
    _check(checks, "constraints-consistent",
           {"policy": policy, "core_constraints_same_length_tol_nm": CONSTRAINT_TOL_NM,
            "appended": len(added_constraints)})

    # -- the two Systems ------------------------------------------------------------------------
    system_a = _clone(env_system)
    for k, b in enumerate(appended):
        system_a.addParticle(masses[n_env + k])
    for i, j, d in added_constraints:
        system_a.addConstraint(i, j, d)
    forces = _forces_by_name(system_a)
    nb_a = system_a.getForce(forces["NonbondedForce"][0])
    bond_a = system_a.getForce(forces["HarmonicBondForce"][0])
    angle_a = system_a.getForce(forces["HarmonicAngleForce"][0])
    torsion_a = system_a.getForce(forces["PeriodicTorsionForce"][0])
    for b in appended:
        _, sigma, _ = package_b.table["atoms"][b][2:5]
        nb_a.addParticle(0.0, sigma, 0.0)
    system_b = _clone(system_a)
    nb_b = system_b.getForce(forces["NonbondedForce"][0])
    bond_b = system_b.getForce(forces["HarmonicBondForce"][0])
    angle_b = system_b.getForce(forces["HarmonicAngleForce"][0])
    torsion_b = system_b.getForce(forces["PeriodicTorsionForce"][0])

    ligand_hyb = set(hyb_a) | set(b_only)

    def role(atoms: Sequence[int]) -> str:
        s = set(atoms)
        if s & set(a_only):
            return "a-only"
        if s & set(b_only):
            return "b-only"
        return "core"

    # particles at B
    for a in range(n_a):
        h = hyb_a[a]
        if a in a_to_b:
            q, s, e = package_b.table["atoms"][a_to_b[a]][2:5]
            nb_b.setParticleParameters(h, q, s, e)
        else:
            q, s, e = nb_b.getParticleParameters(h)
            nb_b.setParticleParameters(h, 0.0, s, 0.0)
    for b in appended:
        q, s, e = package_b.table["atoms"][b][2:5]
        nb_b.setParticleParameters(hyb_b[b], q, s, e)

    # exceptions
    # Both endpoints' 1-4 exceptions at the scales THIS environment applies: endpoint B built
    # into the same box by build-top would get them, so System B must too.
    def exc_rows(package, to_hyb):
        table = scaled_table(package, env_info["nonbonded_applied"])
        return {tuple(sorted((to_hyb[r[0]], to_hyb[r[1]]))): tuple(r[2:5])
                for r in table["exceptions"]}

    exc_a = exc_rows(package_a, hyb_a)
    exc_b = {} if absent_b else exc_rows(package_b, hyb_b)
    env_slots = {}
    for k in range(nb_a.getNumExceptions()):
        i, j, *_ = nb_a.getExceptionParameters(k)
        if i in ligand_hyb:
            env_slots[tuple(sorted((i, j)))] = k
    if set(env_slots) != set(exc_a):
        raise TopologyError("the environment's ligand exceptions are not package A's")
    exception_slots = []
    local_a = {h: a for a, h in enumerate(hyb_a)}
    local_b = {h: b for b, h in enumerate(hyb_b)}

    # A unique group's INTERNAL nonbonded interactions -- its own exceptions and non-excluded pairs
    # -- stay at full physical strength at its dummy end (S0's ruling, the Amber convention): the
    # group is a decoupled but physical fragment, and a term over its internal coordinates alone
    # separates exactly as its retained bonded terms do. Per connected GROUP, never across two
    # groups: two dummy groups on different anchors are separated by physical coordinates, and a
    # pair between them would not separate. In dual topology each whole ligand is one group.
    if absent_b:
        # ONE group: the whole ligand, a dummy at B, bonded to nothing in the environment.
        unique_groups = [{"dummy_at": "B", "atoms": sorted(hyb_a)}]
    elif dual:
        unique_groups = [{"dummy_at": "B", "atoms": sorted(hyb_a)},
                         {"dummy_at": "A", "atoms": sorted(b_only)}]
    else:
        unique_groups = [{"dummy_at": g["dummy_at"], "atoms": g["atoms"]} for g in groups]
    group_of = {h: n for n, g in enumerate(unique_groups) for h in g["atoms"]}

    def internal(key) -> bool:
        return key[0] in group_of and group_of.get(key[1], -1) == group_of[key[0]]

    def zero(params):
        return (0.0, params[1], 0.0)

    for key in sorted(set(exc_a) | set(exc_b)):
        pa, pb = exc_a.get(key), exc_b.get(key)
        r = role(key)
        if pa is None:
            if r != "b-only":
                raise TopologyError(f"core pair {key} is an exception at B only; the core's "
                                    f"exclusion pattern changed")
            pa = pb if internal(key) else zero(pb)
        if pb is None:
            if r != "a-only":
                raise TopologyError(f"core pair {key} is an exception at A only; the core's "
                                    f"exclusion pattern changed")
            pb = pa if internal(key) else zero(pa)
        if key in env_slots:
            slot = env_slots[key]
            nb_b.setExceptionParameters(slot, key[0], key[1], *pb)
        else:
            slot = nb_a.addException(key[0], key[1], *pa)
            if nb_b.addException(key[0], key[1], *pb) != slot:
                raise AssertionError("exception layouts diverged")
        exception_slots.append({"slot": slot, "atoms": list(key), "role": r,
                                "a": list(pa), "b": list(pb),
                                "unique_group_internal": internal(key)})

    # the non-excluded internal pairs: at the physical end the NonbondedForce computes them; at the
    # dummy end the particles carry no charge or epsilon, so they are carried by one
    # CustomBondForce, present in both Systems, zero at the physical end.
    internal_pairs = []
    for n, g in enumerate(unique_groups):
        source = package_b if g["dummy_at"] == "A" else package_a
        local = local_b if g["dummy_at"] == "A" else local_a
        excepted = exc_b if g["dummy_at"] == "A" else exc_a
        members = sorted(g["atoms"])
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if (i, j) in excepted:
                    continue
                qi, si, ei = source.table["atoms"][local[i]][2:5]
                qj, sj, ej = source.table["atoms"][local[j]][2:5]
                internal_pairs.append({"atoms": [i, j], "group": n, "dummy_at": g["dummy_at"],
                                       "physical": [qi * qj, 0.5 * (si + sj),
                                                    math.sqrt(ei * ej)]})
    internal_force = None
    if internal_pairs:
        for system in (system_a, system_b):
            force = CustomBondForce(
                f"{COULOMB_CONSTANT!r}*chargeprod/r + 4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
            for name in ("chargeprod", "sigma", "epsilon"):
                force.addPerBondParameter(name)
            force.setName(INTERNAL_FORCE_NAME)
            # never periodic, whatever the environment: the term must be identical in every leg
            # of a cycle for it to cancel, so no box may enter it (contract section 4)
            force.setUsesPeriodicBoundaryConditions(False)
            endpoint = "A" if system is system_a else "B"
            for k, pair in enumerate(internal_pairs):
                params = pair["physical"] if pair["dummy_at"] == endpoint else \
                    [0.0, pair["physical"][1], 0.0]
                if force.addBond(*pair["atoms"], params) != k:
                    raise AssertionError("internal pair layouts diverged")
            system.addForce(force)
        internal_force = INTERNAL_FORCE_NAME
    exclusions = []
    a_ligand = sorted(hyb_a) if dual else a_only
    for i in a_ligand:
        for j in b_only:
            key = (min(i, j), max(i, j))
            if key in exc_a or key in exc_b:
                raise AssertionError("an A-only/B-only pair is already an exception")
            slot = nb_a.addException(key[0], key[1], 0.0, 1.0, 0.0)
            nb_b.addException(key[0], key[1], 0.0, 1.0, 0.0)
            exclusions.append({"slot": slot, "atoms": list(key), "reason": "a-only-x-b-only"})
    _check(checks, "exceptions-accounted",
           {"ligand_exception_slots": len(exception_slots), "a_only_x_b_only": len(exclusions),
            "internal_exceptions_kept": sum(e["unique_group_internal"]
                                            for e in exception_slots),
            "internal_pairs": len(internal_pairs),
            "dummy_particles": "charge 0, epsilon 0, sigma kept; every exception touching a "
                               "dummy is zero at its dummy endpoint except those inside its own "
                               "unique group"})

    # bonded terms
    bonds_a_graph = {tuple(sorted((hyb_a[i], hyb_a[j]))) for i, j in _bonds(package_a.mol)}
    bonds_b_graph = (set() if absent_b else
                     {tuple(sorted((hyb_b[i], hyb_b[j]))) for i, j in _bonds(package_b.mol)})

    def dummy_decision(atoms_hyb: Sequence[int], kind: str, source: str) -> bool:
        """Whether a term touching *source*'s unique atoms is kept where they are dummies."""
        if whole_molecule_dummy:
            return True     # a whole dummy ligand keeps every intramolecular bonded term
        local = local_a if source == "A" else local_b
        unique = set(a_only) if source == "A" else set(b_only)
        touched = next(h for h in atoms_hyb if h in unique)
        return _retained([local[h] for h in atoms_hyb], kind,
                         junction_of[(source, local[touched])])

    def lay_out(family: str, force_a, force_b, env_terms, rows_a, rows_b, *, width: int,
                add, setp, same=lambda pa, pb: True) -> list[dict]:
        """One bonded force: A's terms stay in their environment slots, B's extras are appended.

        A row is `(key, params, kind)`; `key[:width]` are its atoms and the rest (a torsion's
        periodicity) is part of its identity. An A term and a B term share a slot only if their
        keys agree and `same(params_a, params_b)`; otherwise each has its own slot and is zero at
        the other endpoint, so each endpoint is exactly its own package's terms.
        """
        def off(params):
            return (params[0], 0.0)

        def periodicity(key):
            return {"periodicity": key[width]} if len(key) > width else {}

        slots, used, pending = [], set(), {}
        for key, params, kind in rows_b:
            pending.setdefault(key, []).append((params, kind))
        for key, params, kind in rows_a:
            slot = next((k for k in env_terms.get(key, []) if k not in used), None)
            if slot is None:
                raise TopologyError(f"package A {family} term {key} has no slot in the "
                                    f"environment System")
            used.add(slot)
            atoms, r = key[:width], role(key[:width])
            match = next((n for n, (pb, _) in enumerate(pending.get(key, [])) if same(params, pb)),
                         None)
            if match is not None:
                pb, handling = pending[key].pop(match)[0], "both-physical"
            elif r == "a-only":
                keep = dummy_decision(atoms, kind, "A")
                pb, handling = (params, "dummy-retained") if keep else (off(params),
                                                                          "dummy-removed")
            else:
                pb, handling = off(params), "a-only-term"
            setp(force_b, slot, pb)
            slots.append({"slot": slot, "atoms": list(atoms), **periodicity(key),
                          "kind": kind, "role": r, "a": list(params), "b": list(pb),
                          "at_dummy_end": handling})
        for key, rest in sorted(pending.items()):
            for params, kind in rest:
                atoms, r = key[:width], role(key[:width])
                if r == "b-only":
                    keep = dummy_decision(atoms, kind, "B")
                    pa, handling = (params, "dummy-retained") if keep else (off(params),
                                                                              "dummy-removed")
                elif r == "core":
                    pa, handling = off(params), "b-only-term"
                else:
                    raise AssertionError(f"a B {family} term touches an A-only atom")
                slot = add(force_a, key, pa)
                if add(force_b, key, params) != slot:
                    raise AssertionError(f"{family} layouts diverged")
                slots.append({"slot": slot, "atoms": list(atoms), **periodicity(key),
                              "kind": kind, "role": r, "a": list(pa), "b": list(params),
                              "at_dummy_end": handling})
        return slots

    def env_slots_of(count, get, key_of):
        out: dict[tuple, list[int]] = {}
        for k in range(count):
            row = get(k)
            if row[0] in ligand_hyb:
                out.setdefault(key_of(row), []).append(k)
        return out

    # bonds: a constrained bond has no term in either System
    constrained_hyb = set(env_constraints) | {tuple(c[:2]) for c in added_constraints}

    def bond_rows(package, to_hyb):
        return [(key, (r[2], r[3]), "bond") for r in package.table["bonds"]
                if (key := tuple(sorted((to_hyb[r[0]], to_hyb[r[1]])))) not in constrained_hyb]

    term_slots = {"bonds": lay_out(
        "bond", bond_a, bond_b,
        env_slots_of(bond_a.getNumBonds(), bond_a.getBondParameters,
                     lambda row: tuple(sorted(row[:2]))),
        bond_rows(package_a, hyb_a), [] if absent_b else bond_rows(package_b, hyb_b), width=2,
        add=lambda f, key, p: f.addBond(key[0], key[1], p[0], p[1]),
        setp=lambda f, slot, p: f.setBondParameters(slot, *f.getBondParameters(slot)[:2], *p))}

    def angle_rows(package, to_hyb):
        rows = []
        for r in package.table["angles"]:
            i, j, l = to_hyb[r[0]], to_hyb[r[1]], to_hyb[r[2]]
            rows.append(((min(i, l), j, max(i, l)), (r[3], r[4]), "angle"))
        return rows

    term_slots["angles"] = lay_out(
        "angle", angle_a, angle_b,
        env_slots_of(angle_a.getNumAngles(), angle_a.getAngleParameters,
                     lambda row: (min(row[0], row[2]), row[1], max(row[0], row[2]))),
        angle_rows(package_a, hyb_a), [] if absent_b else angle_rows(package_b, hyb_b), width=3,
        add=lambda f, key, p: f.addAngle(*key, *p),
        setp=lambda f, slot, p: f.setAngleParameters(slot, *f.getAngleParameters(slot)[:3], *p))

    def torsion_rows(package, to_hyb, graph):
        rows = []
        for name, kind in (("proper_torsions", "proper"), ("improper_torsions", "improper")):
            for r in package.table[name]:
                quartet = tuple(to_hyb[x] for x in r[:4])
                if _classify(quartet, graph) != kind:
                    raise AssertionError("torsion classification disagrees with the package")
                rows.append(((*_torsion_key(quartet, kind), int(r[4])), (r[5], r[6]), kind))
        return rows

    # A torsion slot is shared only when periodicity AND phase agree: a phase change is two slots,
    # one switched off and one switched on.
    term_slots["torsions"] = lay_out(
        "torsion", torsion_a, torsion_b,
        env_slots_of(torsion_a.getNumTorsions(), torsion_a.getTorsionParameters,
                     lambda row: (*_torsion_key(row[:4], _classify(row[:4], bonds_a_graph)),
                                  int(row[4]))),
        torsion_rows(package_a, hyb_a, bonds_a_graph),
        [] if absent_b else torsion_rows(package_b, hyb_b, bonds_b_graph), width=4,
        add=lambda f, key, p: f.addTorsion(*key[:4], key[4], *p),
        setp=lambda f, slot, p: f.setTorsionParameters(
            slot, *f.getTorsionParameters(slot)[:5], *p),
        same=lambda pa, pb: pa[0] == pb[0])

    for group in groups:
        members = set(group["atoms"])
        kept = removed = 0
        for family in term_slots.values():
            for slot in family:
                if members & set(slot["atoms"]):
                    if slot["at_dummy_end"] == "dummy-retained":
                        kept += 1
                    elif slot["at_dummy_end"] == "dummy-removed":
                        removed += 1
        group["terms_retained"], group["terms_removed"] = kept, removed

    # -- restraints ---------------------------------------------------------------------------
    # An ALCHEMICAL-COUPLING restraint is built here, because it is part of the construction. A
    # STANDARD-STATE one (ABFE's Boresch) is only RECORDED: the executor builds it, computes its
    # free energy and applies the standard-state correction.
    standard_state = _standard_state_restraint(restraint, absent_b, environment, hyb_a, n_env,
                                               package_a, checks)
    coupling_restraint = None
    if dual:
        group_a = [hyb_a[a] for a, _ in mapped_pairs]
        group_b = [hyb_b[b] for _, b in mapped_pairs]
        for system in (system_a, system_b):
            force = CustomCentroidBondForce(2, "0.5*k_restraint*distance(g1,g2)^2")
            force.addPerBondParameter("k_restraint")
            force.addGroup(group_a, [1.0] * len(group_a))
            force.addGroup(group_b, [1.0] * len(group_b))
            force.addBond([0, 1], [float(dual_restraint_k)])
            force.setUsesPeriodicBoundaryConditions(bool(env_info["periodic"]))
            force.setName("DualTopologyCentroidRestraint")
            system.addForce(force)
        coupling_restraint = {
            "role": "alchemical-coupling",
            "kind": "harmonic centroid-centroid", "energy": "0.5*k*|c_A - c_B|^2",
            "k_kj_mol_nm2": float(dual_restraint_k), "weights": "geometric centroid (all 1)",
            "group_a": group_a, "group_b": group_b, "periodic": bool(env_info["periodic"]),
            "present_at": "both endpoints, identically",
            "why_it_cancels": "it depends only on the centroid separation; integrating the "
                              "dummy ligand's centroid gives (2*pi*kT/k)^(3/2), a constant "
                              "independent of the physical ligand, identical in every leg",
        }

    # -- the combined topology ------------------------------------------------------------------
    topology = Topology()
    topology.setPeriodicBoxVectors(environment.topology.getPeriodicBoxVectors())
    atom_of = {}
    for chain in environment.topology.chains():
        new_chain = topology.addChain(chain.id)
        for residue in chain.residues():
            new_residue = topology.addResidue(residue.name, new_chain, residue.id,
                                              residue.insertionCode)
            for atom in residue.atoms():
                atom_of[atom.index] = topology.addAtom(atom.name, atom.element, new_residue,
                                                       atom.id)
    for bond in environment.topology.bonds():
        topology.addBond(atom_of[bond.atom1.index], atom_of[bond.atom2.index])
    appended_residue = None
    if appended:
        chain_ids = {c.id for c in environment.topology.chains()}
        chain_id = next(c for c in "XYZWVUTSRQPONMLKJIHGFEDCBA" if c not in chain_ids)
        chain = topology.addChain(chain_id)
        residue = topology.addResidue(package_b.residue_name, chain, "1")
        for b in appended:
            atom_of[hyb_b[b]] = topology.addAtom(
                package_b.atom_names[b],
                Element.getByAtomicNumber(package_b.mol.GetAtomWithIdx(b).GetAtomicNum()),
                residue)
        for i, j in sorted(_bonds(package_b.mol)):
            hi, hj = hyb_b[i], hyb_b[j]
            if hi >= n_env or hj >= n_env:
                topology.addBond(atom_of[hi], atom_of[hj])
        appended_residue = {"index_1based": topology.getNumResidues(), "chain": chain_id,
                            "name": package_b.residue_name, "content": (
                                "endpoint B ligand" if dual else "endpoint-B-only atoms"),
                            "atoms": b_only}
    if [a.index for a in topology.atoms()] != list(range(n_total)):
        raise AssertionError("combined topology order does not match the index space")

    # Normalized through the file, as the record is through JSON: the plan keeps the PDB TEXT and
    # the Topology read from it, `write` writes that text verbatim and `load_plan` reads it, so a
    # built plan and a loaded one have the same bond order and one digest authority
    # (rest2.selection.topology_digest) serves both.
    from openmm.app import PDBFile

    from ..rest2.selection import topology_digest

    buffer = io.StringIO()
    PDBFile.writeFile(topology, positions * 10.0, buffer, keepIds=True)
    pdb_text = buffer.getvalue()
    topology = PDBFile(io.StringIO(pdb_text)).topology

    numbering = {
        "scheme": "topology-residue-index-1based",
        "source_topology_sha256": topology_digest(environment.topology),
        "combined_topology_sha256": topology_digest(topology),
        "source_residues_unchanged": environment.topology.getNumResidues(),
        "source_particles_unchanged": n_env,
        "appended_residue": appended_residue,
        "rule": "every environment residue and particle keeps its index; a mask or atom index "
                "resolved before combination selects the same atoms after it",
    }

    # -- identity -------------------------------------------------------------------------------
    def endpoint_record(package, side, to_hyb):
        return {**package.summary(), "side": side,
                "hybrid_index_of_local_atom": list(to_hyb),
                "molecule_sdf_sha256": package.metadata["artifacts"]["molecule.sdf"]["sha256"],
                "ffxml_sha256": package.metadata["artifacts"]["parameters.ffxml"]["sha256"]}

    from .. import __version__

    record = {
        "schema": PLAN_SCHEMA,
        "mode": mode,
        "created_by": {"md_tools": __version__, "openmm": version.full_version},
        "endpoints": {
            "A": {**endpoint_record(package_a, "A", hyb_a),
                  "environment_residue": {"chain": residue_a.chain.id, "resid": residue_a.id,
                                          "insertion_code": residue_a.insertionCode,
                                          "name": residue_a.name,
                                          "index_1based": residue_a.index + 1}},
            "B": ({"absent": True, "reference": None, "hybrid_index_of_local_atom": [],
                   "what": "endpoint B is the ligand ABSENT: its interactions with the "
                           "environment are removed and its own Hamiltonian is retained"}
                  if absent_b else endpoint_record(package_b, "B", hyb_b)),
        },
        "atom_map": ({"schema": MAP_SCHEMA, "package_a": package_a.reference, "package_b": None,
                      "a_to_b": {}, "b_to_a": {}, "by_name": [],
                      "note": "no map: endpoint B is the ligand absent",
                      "sha256": _sha256_text(_canonical(
                          {"decoupling": package_a.reference}))}
                     if absent_b else atom_map.record(package_a, package_b)),
        "map_validation": map_report,
        "environment": {**environment.identity(), **env_info},
        "particles": {"n_total": n_total, "n_environment": n_env, "common": common,
                      "a_only": a_only, "b_only": b_only,
                      "masses": {"policy": "environment masses for every existing particle "
                                           "(endpoint A's for the core); package B masses for "
                                           "appended particles. Masses do not enter the "
                                           "configurational free energy.",
                                 "core_mass_changes": mass_changes}},
        "dummy_groups": groups,
        # STATED, not inferred. A reader of a decoupling record must be able to see WHICH
        # convention produced it: "decoupled" and "annihilated" differ in what the other leg of
        # the cycle has to cancel, and a record that only omits the word leaves that to guesswork.
        **({"decoupling": {
            "intramolecular": "retained",
            "what_vanishes": "the ligand's interactions with the environment, and nothing else",
            "annihilation": "refused in v1: removing the ligand's internal nonbonded terms too "
                            "changes what the solvent leg must cancel and needs its own "
                            "derivation",
            "standard_state_correction": "the executor's, over the recorded restraint",
        }} if absent_b else {}),
        "dummy_nonbonded": "annihilated: charge 0 and epsilon 0 on the particle, zero on every "
                           "exception touching it; bonded terms per the junction rule",
        "nonbonded": {
            "exceptions": exception_slots, "exclusions": exclusions,
            "unique_group_internal": {
                "convention": "a unique group's own exceptions and non-excluded pairs keep "
                              "their physical values at its dummy end (Amber: interactions "
                              "among the disappearing atoms are not changed); vacuum Coulomb "
                              "and Lennard-Jones, per connected group, never across groups",
                "coulomb_constant_kj_mol_nm_e2": COULOMB_CONSTANT,
                "force": internal_force,
                "groups": unique_groups,
                "pairs": internal_pairs,
            },
        },
        "terms": term_slots,
        "constraints": constraint_record,
        # A LIST, and every entry carries its ROLE, because two different things are called a
        # restraint: an `alchemical-coupling` restraint is part of the construction and must be
        # identical in both legs of a cycle, while a `standard-state` one (ABFE's Boresch) is an
        # external term whose free energy is computed and corrected, deliberately present in one
        # leg and absent from the other. Neither is part of the ligand's own Hamiltonian.
        "restraints": ([coupling_restraint] if coupling_restraint else []) + standard_state,
        "coordinates": {"units": "nm", "placement": placement,
                        "positions_sha256": _positions_sha256(positions)},
        "numbering": numbering,
        "checks": checks,
    }

    plan = TopologyPlan(record=record, system_a=system_a, system_b=system_b, topology=topology,
                        positions_nm=positions)
    from .topology_recovery import audit_plan, factorization_check

    audit = audit_plan(plan, package_a, package_b, env_system)
    _check(checks, "term-accounting", audit)
    factor = factorization_check(plan)
    _check(checks, "dummy-factorization", factor)
    # The record is what a reader gets back from plan.json: tuples become lists here, not on
    # the first round trip, so an in-memory plan and a loaded one compare equal.
    record = json.loads(_canonical(record))
    record["ligand_hamiltonian_sha256"] = _sha256_text(_canonical(ligand_hamiltonian(record)))
    record["plan_sha256"] = _record_digest(record)
    return TopologyPlan(record=record, system_a=system_a, system_b=system_b, topology=topology,
                        positions_nm=positions, pdb_text=pdb_text)


# ------------------------------------------------------------------------------------------------
# legs of one cycle
# ------------------------------------------------------------------------------------------------
LIGAND_HAMILTONIAN_SCHEME = "md-tools-ligand-hamiltonian/2"


def ligand_hamiltonian(record: dict) -> dict[str, Any]:
    """Everything a plan says about the LIGANDS' Hamiltonian, independent of the environment.

    Plan indices differ between legs -- a vacuum leg and a solvated one number the ligand
    differently -- so every atom is named by its package-local identity: ("A", i) for an atom of
    endpoint A (the core included), ("B", j) for an atom only B has. Two legs of one cycle must
    agree on all of it: the same packages, map, mode, applied 1-4 scales and constraint policy,
    and the same dummy treatment term by term, or the dummy contributions and the ligand's own
    intramolecular energy do not cancel between them.

    NO RESTRAINT is in here, in any mode (S0 ruling, 2026-09-20): a restraint is not part of what
    the ligand IS. `matched_legs` checks restraints separately, by their role.
    """
    hyb_a = record["endpoints"]["A"]["hybrid_index_of_local_atom"]
    hyb_b = record["endpoints"]["B"]["hybrid_index_of_local_atom"]
    name = {h: ("A", i) for i, h in enumerate(hyb_a)}
    for j, h in enumerate(hyb_b):
        name.setdefault(h, ("B", j))

    def atoms(indices):
        return [list(name[h]) for h in indices]

    terms = {family: sorted(
        [atoms(s["atoms"]), s["kind"], s.get("periodicity"), s["role"], s["a"], s["b"],
         s["at_dummy_end"]] for s in slots) for family, slots in record["terms"].items()}
    exceptions = sorted([atoms(s["atoms"]), s["role"], s["a"], s["b"],
                         s.get("unique_group_internal", False)]
                        for s in record["nonbonded"]["exceptions"])
    internal = record["nonbonded"]["unique_group_internal"]
    groups = sorted([g["dummy_at"], g["local_atoms"],
                     {k: (None if v is None else name[v]) for k, v in g["frame"].items()}]
                    for g in record["dummy_groups"])
    body = {
        "scheme": LIGAND_HAMILTONIAN_SCHEME,
        "mode": record["mode"],
        # `.get`, because a decoupling plan's endpoint B is the ligand ABSENT and has no package.
        "endpoints": {side: {k: record["endpoints"][side].get(k) for k in
                             ("reference", "package_sha256", "parameter_digest", "absent")}
                      for side in ("A", "B")},
        "atom_map_sha256": record["atom_map"]["sha256"],
        "applied_1_4_scales": {k: record["environment"]["nonbonded_applied"][k]
                               for k in ("coulomb14scale", "lj14scale")},
        "constraint_policy": record["constraints"]["policy"],
        "dummy_groups": groups,
        "terms": terms,
        "exceptions": exceptions,
        "exclusions": sorted(atoms(s["atoms"]) for s in record["nonbonded"]["exclusions"]),
        "internal_pairs": sorted([atoms(p["atoms"]), p["dummy_at"], p["physical"]]
                                 for p in internal["pairs"]),
    }
    return json.loads(_canonical(body))


def matched_legs(first: "TopologyPlan", second: "TopologyPlan") -> dict[str, Any]:
    """Refuse two plans that cannot be the two legs of one thermodynamic cycle.

    Compares `ligand_hamiltonian` of both, component by component, and names every component that
    differs. Passing means the dummy groups' bonded and internal terms, the restraint and the
    ligands' own intramolecular Hamiltonian are identical in both legs, which is what makes them
    cancel -- the environment is all that differs.
    """
    coupling = _restraints_by_role(first.record, second.record, "alchemical-coupling")
    standard_state = _restraints_by_role(first.record, second.record, "standard-state")
    one, two = ligand_hamiltonian(first.record), ligand_hamiltonian(second.record)
    differing = sorted(k for k in set(one) | set(two) if one.get(k) != two.get(k))
    if "applied_1_4_scales" in differing:
        raise TopologyError(
            f"these plans apply different 1-4 scales to the ligand -- "
            f"{one['applied_1_4_scales']} and {two['applied_1_4_scales']} -- so the ligand's own "
            f"intramolecular Hamiltonian is a different function in the two legs, and a cycle "
            f"between them would fold that difference into the free energy, however small. An "
            f"OPC-solvated leg applies OPC's rounded 0.833333; a vacuum leg applies the package's "
            f"5/6. Pair a vacuum leg with a TIP3P-solvated one. (A vacuum build that applies the "
            f"solvent leg's stated scale is a deferred backlog item.)")
    if differing:
        named = {k: (one.get(k), two.get(k)) for k in differing
                 if k in ("mode", "constraint_policy", "atom_map_sha256")}
        raise TopologyError(
            f"these plans are not two legs of one cycle: {differing} differ"
            f"{' ' + str(named) if named else ''}. Build both legs from the same packages, map "
            f"and mode, in environments that apply the same 1-4 scales and constraint policy to "
            f"the ligand.")
    # ALCHEMICAL-COUPLING restraints shape the path: dual topology's centroid restraint cancels
    # only because both legs carry the same one.
    if _named_atoms(first.record, coupling[0]) != _named_atoms(second.record, coupling[1]):
        raise TopologyError(
            f"these legs carry different alchemical-coupling restraints ({len(coupling[0])} and "
            f"{len(coupling[1])}): such a restraint is part of the construction, and a cycle "
            f"cancels it only if both legs carry the same one.")
    # STANDARD-STATE restraints (ABFE's Boresch) are external terms whose free energy is computed
    # and corrected: deliberately in one leg and not the other, never silently ignored.
    if standard_state[0] and standard_state[1]:
        raise TopologyError(
            "both legs carry a standard-state restraint. Its free energy is computed and "
            "corrected for the leg that has one; two legs restrained at once is not a cycle this "
            "construction describes.")
    digest = _sha256_text(_canonical(one))
    return {"scheme": LIGAND_HAMILTONIAN_SCHEME, "ligand_hamiltonian_sha256": digest,
            "plans": [first.sha256, second.sha256],
            "environments": [first.record["environment"]["system_sha256"],
                             second.record["environment"]["system_sha256"]],
            "restraints": {
                "alchemical_coupling": "identical in both legs" if coupling[0] else "none",
                # Reported rather than compared: what it is worth is S4's to decide, and a
                # difference here is the point of an absolute-binding cycle, not a defect.
                "standard_state": [standard_state[0], standard_state[1]]}}


def _restraints_by_role(first: dict, second: dict, role: str) -> tuple[list, list]:
    return tuple([r for r in (record.get("restraints") or []) if r.get("role") == role]
                 for record in (first, second))


#: Fields of a restraint record that are NOT compared between two legs. `periodic` is a property
#: of the box, not of the restraint: a solvated leg takes the minimum image and a vacuum leg has
#: no images, while both evaluate the same centroid separation for a molecule that does not
#: straddle a boundary -- and the cancellation argument depends only on that separation. Requiring
#: it to match would make every vacuum/solvent dual cycle impossible.
RESTRAINT_FIELDS_NOT_COMPARED = ("periodic",)


def _named_atoms(record: dict, restraints: list) -> list:
    """Restraint records with their atoms in package-local identities, for comparing two legs."""
    hyb_a = record["endpoints"]["A"]["hybrid_index_of_local_atom"]
    hyb_b = record["endpoints"]["B"].get("hybrid_index_of_local_atom") or []
    name = {h: ["A", i] for i, h in enumerate(hyb_a)}
    for j, h in enumerate(hyb_b):
        name.setdefault(h, ["B", j])
    out = []
    for restraint in restraints:
        entry = {k: v for k, v in restraint.items()
                 if not k.startswith("group_") and k not in RESTRAINT_FIELDS_NOT_COMPARED}
        for key in ("group_a", "group_b"):
            if key in restraint:
                entry[key] = [name.get(h, ["environment", h]) for h in restraint[key]]
        out.append(entry)
    return out
