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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..ligands.mapping import LigandSelector
from ..ligands.package import LigandPackage, compare_parameter_tables, subsystem_parameter_table, load_package
from .topology_mapping import AtomMap, MapError, _bonds, _neighbours, _positions, validate_map

__all__ = [
    "PLAN_SCHEMA",
    "Environment",
    "TopologyError",
    "TopologyPlan",
    "build_topology_plan",
    "load_plan",
]

PLAN_SCHEMA = "md-tools-topology-plan/1"
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

    @classmethod
    def from_files(cls, system_xml: Path, pdb: Path, ligand: LigandSelector) -> "Environment":
        """A `build-top` pair: `built.xml` and `built.pdb`, particle order matching exactly."""
        from openmm import XmlSerializer, unit
        from openmm.app import PDBFile

        system_xml, pdb = Path(system_xml), Path(pdb)
        text = system_xml.read_text(encoding="utf-8")
        structure = PDBFile(str(pdb))
        positions = np.array(structure.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                             dtype=float)
        return cls(system=XmlSerializer.deserialize(text), topology=structure.topology,
                   positions_nm=positions, ligand=ligand,
                   source={"system_file": system_xml.name,
                           "system_file_sha256":
                               hashlib.sha256(system_xml.read_bytes()).hexdigest(),
                           "pdb_file": pdb.name,
                           "pdb_file_sha256": hashlib.sha256(pdb.read_bytes()).hexdigest()})

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

    c14, lj14 = package_a.conventions["coulomb14scale"], package_a.conventions["lj14scale"]
    try:
        table, constrained = subsystem_parameter_table(system, package_a.mol, ligand_indices, c14, lj14)
    except ValueError as exc:
        raise TopologyError(f"the environment ligand cannot be read as package "
                            f"{package_a.reference}: {exc}") from exc
    problems = compare_parameter_tables(package_a.table, table, where="environment ligand",
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
            "masses": "package masses (no repartitioning)", "constraint_policy": policy})

    method = nb.getNonbondedMethod()
    return {
        "constraint_policy": policy,
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
        raise TopologyError(f"{directory}: plan schema {record.get('schema')!r} is not "
                            f"{PLAN_SCHEMA!r}")
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


def build_topology_plan(package_a: LigandPackage, package_b: LigandPackage, atom_map: AtomMap,
                        environment: Environment, *, mode: str,
                        dual_restraint_k: float = DEFAULT_DUAL_RESTRAINT_K,
                        b_positions_nm: Optional[np.ndarray] = None) -> TopologyPlan:
    """Build and check the plan. Nothing is written; `TopologyPlan.write` writes it.

    *b_positions_nm* is endpoint B's pose in B package order (a docked pose, for example). Left
    None, package B's reference conformer is superposed on the mapped A atoms, which is what a
    reference conformer is for, and the core RMSD of that fit is recorded.
    """
    from openmm import CustomCentroidBondForce, version
    from openmm.app import Element, Topology

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

    a_to_b, b_to_a = ({}, {}) if dual else (atom_map.a_to_b, atom_map.b_to_a)
    n_a, n_b = len(package_a.atom_names), len(package_b.atom_names)
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
    mapped_pairs = sorted(atom_map.pairs)
    xa = environment.positions_nm[[hyb_a[a] for a, _ in mapped_pairs]]
    if b_positions_nm is None:
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
    rmsd = float(np.sqrt(np.mean(np.sum((placed[[b for _, b in mapped_pairs]] - xa) ** 2, axis=1))))
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
    if not dual:
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
    b_bond_rows = {tuple(sorted((r[0], r[1]))): r for r in package_b.table["bonds"]}
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
    def exc_rows(package, to_hyb):
        return {tuple(sorted((to_hyb[r[0]], to_hyb[r[1]]))): tuple(r[2:5])
                for r in package.table["exceptions"]}

    exc_a, exc_b = exc_rows(package_a, hyb_a), exc_rows(package_b, hyb_b)
    env_slots = {}
    for k in range(nb_a.getNumExceptions()):
        i, j, *_ = nb_a.getExceptionParameters(k)
        if i in ligand_hyb:
            env_slots[tuple(sorted((i, j)))] = k
    if set(env_slots) != set(exc_a):
        raise TopologyError("the environment's ligand exceptions are not package A's")
    exception_slots = []

    def zero(params):
        return (0.0, params[1], 0.0)

    for key in sorted(set(exc_a) | set(exc_b)):
        pa, pb = exc_a.get(key), exc_b.get(key)
        r = role(key)
        if pa is None:
            if r != "b-only":
                raise TopologyError(f"core pair {key} is an exception at B only; the core's "
                                    f"exclusion pattern changed")
            pa = zero(pb)
        if pb is None:
            if r != "a-only":
                raise TopologyError(f"core pair {key} is an exception at A only; the core's "
                                    f"exclusion pattern changed")
            pb = zero(pa)
        if key in env_slots:
            slot = env_slots[key]
            nb_b.setExceptionParameters(slot, key[0], key[1], *pb)
        else:
            slot = nb_a.addException(key[0], key[1], *pa)
            if nb_b.addException(key[0], key[1], *pb) != slot:
                raise AssertionError("exception layouts diverged")
        exception_slots.append({"slot": slot, "atoms": list(key), "role": r,
                                "a": list(pa), "b": list(pb)})
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
            "dummy_particles": "charge 0, epsilon 0, sigma kept; every exception touching a "
                               "dummy is zero at its dummy endpoint"})

    # bonded terms
    bonds_a_graph = {tuple(sorted((hyb_a[i], hyb_a[j]))) for i, j in _bonds(package_a.mol)}
    bonds_b_graph = {tuple(sorted((hyb_b[i], hyb_b[j]))) for i, j in _bonds(package_b.mol)}
    local_a = {h: a for a, h in enumerate(hyb_a)}
    local_b = {h: b for b, h in enumerate(hyb_b)}

    def dummy_decision(atoms_hyb: Sequence[int], kind: str, source: str) -> bool:
        """Whether a term touching *source*'s unique atoms is kept where they are dummies."""
        if dual:
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
        bond_rows(package_a, hyb_a), bond_rows(package_b, hyb_b), width=2,
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
        angle_rows(package_a, hyb_a), angle_rows(package_b, hyb_b), width=3,
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
        torsion_rows(package_b, hyb_b, bonds_b_graph), width=4,
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

    # -- the dual restraint -------------------------------------------------------------------
    restraint = None
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
        restraint = {
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
            "B": endpoint_record(package_b, "B", hyb_b),
        },
        "atom_map": atom_map.record(package_a, package_b),
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
        "dummy_nonbonded": "annihilated: charge 0 and epsilon 0 on the particle, zero on every "
                           "exception touching it; bonded terms per the junction rule",
        "nonbonded": {"exceptions": exception_slots, "exclusions": exclusions},
        "terms": term_slots,
        "constraints": constraint_record,
        "restraint": restraint,
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
    record["plan_sha256"] = _record_digest(record)
    return TopologyPlan(record=record, system_a=system_a, system_b=system_b, topology=topology,
                        positions_nm=positions, pdb_text=pdb_text)
