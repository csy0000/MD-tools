"""An explicit atom map between two DIFFERENT compounds, and the chemistry it must respect.

Nothing else in the package maps one compound onto another. `ligands.mapping.LigandInstance.
heavy_atom_map` joins a deposited residue to its OWN package, and `ais.two_state` refuses any
pair of Systems whose particles differ -- an identity map by construction. An alchemical
transformation needs a map between two parameter packages, and this module is that map.

A map is stated in PACKAGE-LOCAL atom indices, or names, of two registered packages. Those are
the only atom identities that survive a rebuild: a package's atom i is the i-th atom of its
`molecule.sdf`, forever, while a System index moves whenever solvent, ions or a protein are added.

The map is checked for what it IS (one-to-one, and its two directions exact inverses) and for
what it says about chemistry. A map that would need a ring to open, a stereocentre to invert, a
net charge to change, or a dummy group to hang from two anchors is REFUSED here, before anything
is built: each of those is a transformation the 0.7.0 construction does not represent correctly,
and a plan built from one would run and give a number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np

from ..ligands.package import LigandPackage

__all__ = [
    "MAP_SCHEMA",
    "MODES",
    "AtomMap",
    "MapError",
    "propose_map",
    "unique_components",
    "validate_map",
]

MAP_SCHEMA = "md-tools-alchemical-atom-map/1"

#: The topology representations 0.7.0 builds. Separated topology is deferred and refused by name.
MODES = ("single", "dual", "hybrid")
DEFERRED_MODES = {"separated": "separated topology is deferred beyond 0.7.0 and is not built, "
                               "partially or otherwise"}


class MapError(ValueError):
    """A map that is malformed, or that asks for a transformation this construction refuses."""


AtomRef = Union[int, str]


def _canonical(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _resolve(ref: AtomRef, package: LigandPackage, side: str) -> int:
    if isinstance(ref, bool):
        raise MapError(f"endpoint {side}: {ref!r} is not an atom index or name")
    if isinstance(ref, (int, np.integer)):
        index = int(ref)
        if not 0 <= index < len(package.atom_names):
            raise MapError(f"endpoint {side}: atom index {index} is outside "
                           f"{package.reference} (0..{len(package.atom_names) - 1})")
        return index
    if isinstance(ref, str):
        hits = [i for i, name in enumerate(package.atom_names) if name == ref]
        if len(hits) != 1:
            raise MapError(f"endpoint {side}: atom name {ref!r} names {len(hits)} atoms of "
                           f"{package.reference}; a name in a map must name exactly one")
        return hits[0]
    raise MapError(f"endpoint {side}: {ref!r} is not an atom index or name")


@dataclass(frozen=True)
class AtomMap:
    """Pairs `(a, b)` of package-local atom indices: atom a of endpoint A IS atom b of B."""

    package_a: str
    package_b: str
    pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_pairs(cls, package_a: LigandPackage, package_b: LigandPackage,
                   pairs: Union[Mapping[AtomRef, AtomRef], Iterable[Sequence[AtomRef]]]
                   ) -> "AtomMap":
        """Resolve indices or names, and refuse anything that is not one-to-one.

        A mapping type is accepted for convenience, but it cannot express a duplicated A atom, so
        a sequence of pairs is what a stored map is read from: a duplicate there is refused
        rather than silently collapsed by a dict.
        """
        raw = list(pairs.items()) if isinstance(pairs, Mapping) else [tuple(p) for p in pairs]
        resolved = []
        for pair in raw:
            if len(pair) != 2:
                raise MapError(f"a map entry is a pair (a, b), got {pair!r}")
            resolved.append((_resolve(pair[0], package_a, "A"), _resolve(pair[1], package_b, "B")))
        if not resolved:
            raise MapError("the map is empty; every construction needs at least one mapped pair")
        a_seen: dict[int, int] = {}
        b_seen: dict[int, int] = {}
        for a, b in resolved:
            if a in a_seen:
                raise MapError(f"A atom {package_a.atom_names[a]} ({a}) is mapped twice, to "
                               f"{a_seen[a]} and {b}; a map is one-to-one")
            if b in b_seen:
                raise MapError(f"B atom {package_b.atom_names[b]} ({b}) is mapped twice, from "
                               f"{b_seen[b]} and {a}; a map is one-to-one")
            a_seen[a], b_seen[b] = b, a
        return cls(package_a=package_a.reference, package_b=package_b.reference,
                   pairs=tuple(sorted(resolved)))

    @property
    def a_to_b(self) -> dict[int, int]:
        return {a: b for a, b in self.pairs}

    @property
    def b_to_a(self) -> dict[int, int]:
        return {b: a for a, b in self.pairs}

    def inverse(self) -> "AtomMap":
        return AtomMap(package_a=self.package_b, package_b=self.package_a,
                       pairs=tuple(sorted((b, a) for a, b in self.pairs)))

    def record(self, package_a: LigandPackage, package_b: LigandPackage) -> dict[str, Any]:
        """Both directions, by index AND by name, so either can be reviewed without the other."""
        if (package_a.reference, package_b.reference) != (self.package_a, self.package_b):
            raise MapError(f"this map is between {self.package_a} and {self.package_b}, not "
                           f"{package_a.reference} and {package_b.reference}")
        body = {
            "schema": MAP_SCHEMA,
            "package_a": self.package_a,
            "package_b": self.package_b,
            "a_to_b": {str(a): b for a, b in sorted(self.pairs)},
            "b_to_a": {str(b): a for b, a in sorted((b, a) for a, b in self.pairs)},
            "by_name": [[package_a.atom_names[a], package_b.atom_names[b]]
                        for a, b in sorted(self.pairs)],
        }
        return {**body, "sha256": hashlib.sha256(_canonical(body).encode()).hexdigest()}

    @classmethod
    def from_record(cls, record: Mapping[str, Any], package_a: LigandPackage,
                    package_b: LigandPackage) -> "AtomMap":
        """Read a stored map, and refuse one whose two directions or digest disagree."""
        if record.get("schema") != MAP_SCHEMA:
            raise MapError(f"atom map schema {record.get('schema')!r} is not {MAP_SCHEMA!r}")
        forward = {int(a): int(b) for a, b in record["a_to_b"].items()}
        backward = {int(b): int(a) for b, a in record["b_to_a"].items()}
        if {b: a for a, b in forward.items()} != backward or len(forward) != len(backward):
            raise MapError("the stored map's a_to_b and b_to_a are not inverses of each other")
        amap = cls.from_pairs(package_a, package_b, sorted(forward.items()))
        if amap.record(package_a, package_b) != dict(record):
            raise MapError("the stored map does not reproduce its own record (names, packages or "
                           "digest differ); it was edited or belongs to other packages")
        return amap


# ------------------------------------------------------------------------------------------------
# graph helpers
# ------------------------------------------------------------------------------------------------
def _bonds(mol) -> set[tuple[int, int]]:
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}


def _neighbours(mol) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {a.GetIdx(): [] for a in mol.GetAtoms()}
    for i, j in _bonds(mol):
        out[i].append(j)
        out[j].append(i)
    return {k: sorted(v) for k, v in out.items()}


def unique_components(mol, mapped: set[int]) -> list[dict[str, Any]]:
    """Connected components of the UNMAPPED atoms, each with the bonds that attach it to the core.

    Sorted by the smallest package-local index in each component, so the order is a property of
    the package and the map, never of a System.
    """
    neighbours = _neighbours(mol)
    unmapped = sorted(set(range(mol.GetNumAtoms())) - mapped)
    seen: set[int] = set()
    components = []
    for start in unmapped:
        if start in seen:
            continue
        stack, members = [start], set()
        while stack:
            atom = stack.pop()
            if atom in members:
                continue
            members.add(atom)
            stack.extend(n for n in neighbours[atom] if n not in mapped and n not in members)
        seen |= members
        attachments = sorted((d, p) for d in members for p in neighbours[d] if p in mapped)
        components.append({"atoms": sorted(members), "attachments": attachments})
    return sorted(components, key=lambda c: c["atoms"][0])


def _positions(package: LigandPackage) -> np.ndarray:
    conformer = package.mol.GetConformer()
    return np.array([list(conformer.GetAtomPosition(i)) for i in range(package.mol.GetNumAtoms())],
                    dtype=float)


def _signed_volume(points: np.ndarray, centre: np.ndarray) -> float:
    return float(np.linalg.det(np.stack([p - centre for p in points])))


def _stereocentres(mol) -> list[int]:
    from rdkit import Chem

    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work)
    return [i for i, _ in Chem.FindMolChiralCenters(work, includeUnassigned=True,
                                                     useLegacyImplementation=False)]


# ------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------
def _check(checks: list, name: str, detail: Any) -> None:
    checks.append({"check": name, "status": "pass", "detail": detail})


def validate_map(package_a: LigandPackage, package_b: LigandPackage, amap: AtomMap,
                 mode: str) -> dict[str, Any]:
    """Every chemistry check the construction depends on; a refusal raises `MapError`.

    Returns the report the plan stores: which checks ran, what each found, the unique components
    on each side with their attachment bonds, and the element, bond-order and aromaticity
    changes the map makes -- recorded, because they are allowed but must be reviewable.
    """
    if mode in DEFERRED_MODES:
        raise MapError(f"mode {mode!r}: {DEFERRED_MODES[mode]}")
    if mode not in MODES:
        raise MapError(f"mode {mode!r} is not one of {MODES}")
    if (amap.package_a, amap.package_b) != (package_a.reference, package_b.reference):
        raise MapError(f"the map is between {amap.package_a} and {amap.package_b}, not "
                       f"{package_a.reference} and {package_b.reference}")
    mol_a, mol_b = package_a.mol, package_b.mol
    a_to_b, b_to_a = amap.a_to_b, amap.b_to_a
    checks: list[dict[str, Any]] = []

    # -- the map as a function ------------------------------------------------------------------
    if any(b_to_a[a_to_b[a]] != a for a in a_to_b) or any(a_to_b[b_to_a[b]] != b for b in b_to_a):
        raise MapError("a_to_b and b_to_a are not inverses")
    _check(checks, "map-symmetry", {"pairs": len(amap.pairs),
                                    "a_to_b_then_b_to_a": "identity on the mapped A atoms",
                                    "b_to_a_then_a_to_b": "identity on the mapped B atoms"})

    # -- same force field, same charge model, same conventions ---------------------------------
    fa, fb = package_a.metadata["forcefield"], package_b.metadata["forcefield"]
    if fa != fb:
        raise MapError(f"{package_a.reference} is {fa}, {package_b.reference} is {fb}; the two "
                       f"endpoints of one transformation must come from ONE force field")
    ca, cb = package_a.metadata["charges"], package_b.metadata["charges"]
    for key in ("method", "backend_id"):
        if ca.get(key) != cb.get(key):
            raise MapError(f"charge {key} differs: {ca.get(key)!r} for {package_a.reference}, "
                           f"{cb.get(key)!r} for {package_b.reference}. Two charge models in one "
                           f"transformation measure the model difference along with the "
                           f"chemistry.")
    if package_a.conventions != package_b.conventions:
        raise MapError(f"nonbonded conventions differ: {package_a.conventions} vs "
                       f"{package_b.conventions}")
    _check(checks, "force-field-compatibility",
           {"forcefield": fa, "charge_method": ca.get("method"),
            "charge_backend_id": ca.get("backend_id"), "conventions": package_a.conventions})

    # -- net charge ---------------------------------------------------------------------------
    qa = package_a.metadata["chemical_state"]["net_formal_charge"]
    qb = package_b.metadata["chemical_state"]["net_formal_charge"]
    if qa != qb:
        raise MapError(f"net formal charge changes from {qa} to {qb}. A charge-changing "
                       f"transformation needs a finite-size/co-alchemical treatment that 0.7.0 "
                       f"does not implement; it is refused rather than run uncorrected.")
    _check(checks, "net-charge-preserved", {"net_formal_charge": qa})

    # -- elements -----------------------------------------------------------------------------
    element_changes = []
    for a, b in amap.pairs:
        ea, eb = mol_a.GetAtomWithIdx(a).GetSymbol(), mol_b.GetAtomWithIdx(b).GetSymbol()
        if ea == eb:
            continue
        hydrogen_swap = (ea == "H") != (eb == "H")
        if hydrogen_swap and mode != "single":
            raise MapError(f"{package_a.atom_names[a]} ({ea}) is mapped to "
                           f"{package_b.atom_names[b]} ({eb}): a hydrogen mapped to a heavy atom "
                           f"is a single-topology construction. In {mode} topology leave both "
                           f"unmapped, as unique atoms.")
        element_changes.append({"a": a, "b": b, "a_name": package_a.atom_names[a],
                                "b_name": package_b.atom_names[b], "a_element": ea,
                                "b_element": eb, "hydrogen_to_heavy": hydrogen_swap})

    bonds_a, bonds_b = _bonds(mol_a), _bonds(mol_b)
    components_a = unique_components(mol_a, set(a_to_b))
    components_b = unique_components(mol_b, set(b_to_a))
    notes: dict[str, Any] = {"element_changes": element_changes}

    if mode == "single" and components_a and components_b:
        raise MapError(
            f"single topology is ONE evolving atom representation: every atom of one endpoint "
            f"must be mapped. This map leaves {sum(len(c['atoms']) for c in components_a)} A "
            f"atoms and {sum(len(c['atoms']) for c in components_b)} B atoms unmapped; that is "
            f"a hybrid topology.")

    if mode in ("single", "hybrid"):
        # -- the mapped core keeps its bond graph, in both directions ---------------------------
        for a1, a2 in sorted(bonds_a):
            if a1 in a_to_b and a2 in a_to_b and tuple(sorted((a_to_b[a1], a_to_b[a2]))) \
                    not in bonds_b:
                raise MapError(f"A bond {package_a.atom_names[a1]}-{package_a.atom_names[a2]} "
                               f"maps onto two B atoms that are not bonded: the core's bond "
                               f"graph changes (a ring opening or a rearrangement), which is "
                               f"refused")
        for b1, b2 in sorted(bonds_b):
            if b1 in b_to_a and b2 in b_to_a and tuple(sorted((b_to_a[b1], b_to_a[b2]))) \
                    not in bonds_a:
                raise MapError(f"B bond {package_b.atom_names[b1]}-{package_b.atom_names[b2]} "
                               f"maps onto two A atoms that are not bonded: the core's bond "
                               f"graph changes (a ring closure or a rearrangement), which is "
                               f"refused")
        _check(checks, "core-bond-graph-preserved", {"core_bonds": sum(
            1 for a1, a2 in bonds_a if a1 in a_to_b and a2 in a_to_b)})

        ring_a, ring_b = mol_a.GetRingInfo(), mol_b.GetRingInfo()
        for a, b in amap.pairs:
            if ring_a.NumAtomRings(a) != ring_b.NumAtomRings(b):
                raise MapError(f"{package_a.atom_names[a]} is in {ring_a.NumAtomRings(a)} ring(s) "
                               f"and {package_b.atom_names[b]} in {ring_b.NumAtomRings(b)}: ring "
                               f"changes are refused in 0.7.0")
        _check(checks, "ring-membership-preserved", {"mapped_atoms": len(amap.pairs)})

        # -- every unique group hangs from exactly one core atom -------------------------------
        for side, components, package in (("A", components_a, package_a),
                                          ("B", components_b, package_b)):
            for component in components:
                if len(component["attachments"]) != 1:
                    names = [package.atom_names[i] for i in component["atoms"]]
                    raise MapError(
                        f"the unique {side} group {names} is attached to the mapped core by "
                        f"{len(component['attachments'])} bonds. A dummy group's partition "
                        f"function separates from the physical system only when it hangs from "
                        f"ONE bond to ONE core atom; a group bridging two core atoms (a ring "
                        f"growing or a partially mapped ring) is refused.")
        _check(checks, "single-attachment-dummy-groups",
               {"A": [c["attachments"][0] for c in components_a],
                "B": [c["attachments"][0] for c in components_b]})

        # -- bond order and aromaticity, recorded ---------------------------------------------
        order_changes = []
        for a1, a2 in sorted(bonds_a):
            if a1 in a_to_b and a2 in a_to_b:
                ba = mol_a.GetBondBetweenAtoms(a1, a2)
                bb = mol_b.GetBondBetweenAtoms(a_to_b[a1], a_to_b[a2])
                if ba.GetBondType() != bb.GetBondType():
                    order_changes.append({"a": [a1, a2], "b": [a_to_b[a1], a_to_b[a2]],
                                          "a_order": str(ba.GetBondType()),
                                          "b_order": str(bb.GetBondType())})
        notes["bond_order_changes"] = order_changes
        notes["aromaticity_changes"] = [
            {"a": a, "b": b} for a, b in amap.pairs
            if mol_a.GetAtomWithIdx(a).GetIsAromatic() != mol_b.GetAtomWithIdx(b).GetIsAromatic()]

        # -- stereochemistry ------------------------------------------------------------------
        xa, xb = _positions(package_a), _positions(package_b)
        nbr_a, nbr_b = _neighbours(mol_a), _neighbours(mol_b)
        centres = sorted({a for a in _stereocentres(mol_a) if a in a_to_b}
                         | {b_to_a[b] for b in _stereocentres(mol_b) if b in b_to_a})
        verified = []
        for a in centres:
            b = a_to_b[a]
            shared = [n for n in nbr_a[a] if n in a_to_b and a_to_b[n] in nbr_b[b]]
            if len(shared) < 3:
                raise MapError(
                    f"stereocentre {package_a.atom_names[a]} / {package_b.atom_names[b]} has "
                    f"{len(shared)} mapped neighbours in common; with fewer than three the map "
                    f"does not say which configuration B has relative to A. State a map that "
                    f"covers three of its neighbours, or refuse the pair.")
            va = _signed_volume(xa[shared[:3]], xa[a])
            vb = _signed_volume(xb[[a_to_b[n] for n in shared[:3]]], xb[b])
            if va * vb <= 0:
                raise MapError(f"the map inverts stereocentre {package_a.atom_names[a]} -> "
                               f"{package_b.atom_names[b]}; stereochemistry changes are refused")
            verified.append([a, b])
        _check(checks, "stereochemistry-preserved", {"centres": verified})
        notes["ambiguity"] = ("an explicit map in package-local indices names one pairing; "
                              "stereocentres are checked by the signed volume of three shared "
                              "neighbours in the package conformers, and a centre whose "
                              "configuration the map cannot fix is refused")
    else:
        # dual: the map only aligns B on A and chooses the restrained groups
        _check(checks, "dual-map-role", "the map places endpoint B and defines the restrained "
                                        "centroid groups; the ligands share no particle")

    return {
        "mode": mode,
        "checks": checks,
        "unique_components": {"A": components_a, "B": components_b},
        "notes": notes,
    }


# ------------------------------------------------------------------------------------------------
# a validated automatic map
# ------------------------------------------------------------------------------------------------
AUTOMATIC_METHOD = "rdkit-fmcs-heavy/1"
FMCS_TIMEOUT_S = 10


def _heavy(mol):
    from rdkit import Chem

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    stripped = Chem.RemoveHs(mol, sanitize=False)
    Chem.GetSymmSSSR(stripped)          # ring information, which an unsanitised copy lacks
    if [a.GetSymbol() for a in stripped.GetAtoms()] != \
            [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]:
        raise AssertionError("removing hydrogens reordered the heavy atoms")
    return stripped, heavy


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def _signature(mol_a, mol_b, pairs) -> tuple:
    """The map up to the symmetries of both molecules: canonical ranks with ties unbroken."""
    from rdkit import Chem

    ra = list(Chem.CanonicalRankAtoms(mol_a, breakTies=False))
    rb = list(Chem.CanonicalRankAtoms(mol_b, breakTies=False))
    return tuple(sorted((ra[a], rb[b]) for a, b in pairs))


def propose_map(package_a: LigandPackage, package_b: LigandPackage, mode: str) -> tuple[
        "AtomMap", dict[str, Any]]:
    """A maximum-common-substructure map, checked by `validate_map`, or a refusal.

    Heavy atoms by RDKit FMCS (elements equal, bond orders exact, rings match only rings and only
    complete rings); every placement of that substructure in A and in B is enumerated; hydrogens
    of each mapped heavy pair are paired by distance after superposing B's conformer on A's on
    the mapped heavy atoms. Each candidate must pass `validate_map`; the largest survive. When
    the survivors are not all related by a symmetry of A or of B -- compared by canonical ranks
    with ties unbroken -- the choice changes the transformation and nothing here can make it:
    that is REFUSED as chemically ambiguous, with the alternatives, and an explicit map is
    required. Single topology is explicit-map only.
    """
    from rdkit.Chem import rdFMCS

    if mode == "single":
        raise MapError("an automatic map is not offered for single topology, where a mapped atom "
                       "may change element; state the map explicitly")
    if mode not in MODES:
        validate_map(package_a, package_b, AtomMap(package_a.reference, package_b.reference,
                                                   ((0, 0),)), mode)   # raises by name
    mol_a, mol_b = package_a.mol, package_b.mol
    heavy_a, ia = _heavy(mol_a)
    heavy_b, ib = _heavy(mol_b)
    params = rdFMCS.MCSParameters()
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.Timeout = FMCS_TIMEOUT_S
    result = rdFMCS.FindMCS([heavy_a, heavy_b], params)
    if result.canceled:
        raise MapError(f"the maximum common substructure search did not finish in "
                       f"{FMCS_TIMEOUT_S} s; state the map explicitly")
    if not result.numAtoms:
        raise MapError("the two molecules share no heavy-atom substructure")
    from rdkit import Chem

    query = Chem.MolFromSmarts(result.smartsString)
    matches_a = heavy_a.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    matches_b = heavy_b.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    xa, xb = _positions(package_a), _positions(package_b)
    nbr_a, nbr_b = _neighbours(mol_a), _neighbours(mol_b)
    hydrogen_a = {i for i, a in enumerate(mol_a.GetAtoms()) if a.GetAtomicNum() == 1}
    hydrogen_b = {i for i, a in enumerate(mol_b.GetAtoms()) if a.GetAtomicNum() == 1}
    candidates, rejected, seen = [], [], set()
    for ma in matches_a:
        for mb in matches_b:
            heavy_pairs = sorted((ia[p], ib[q]) for p, q in zip(ma, mb))
            key = tuple(heavy_pairs)
            if key in seen:
                continue
            seen.add(key)
            a_idx = [a for a, _ in heavy_pairs]
            b_idx = [b for _, b in heavy_pairs]
            if len(heavy_pairs) >= 3:
                cs, ct = xb[b_idx].mean(axis=0), xa[a_idx].mean(axis=0)
                h = (xb[b_idx] - cs).T @ (xa[a_idx] - ct)
                u, _, vt = np.linalg.svd(h)
                d = np.sign(np.linalg.det(vt.T @ u.T))
                rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
                placed_b = (xb - cs) @ rot.T + ct
            else:
                placed_b = xb - xb[b_idx].mean(axis=0) + xa[a_idx].mean(axis=0)
            pairs = list(heavy_pairs)
            for a, b in heavy_pairs:
                ha = [n for n in nbr_a[a] if n in hydrogen_a]
                hb = [n for n in nbr_b[b] if n in hydrogen_b]
                if ha and hb:
                    cost = np.linalg.norm(xa[ha][:, None, :] - placed_b[hb][None, :, :], axis=-1)
                    pairs += [(ha[r], hb[c]) for r, c in _hungarian(cost)]
            amap = AtomMap.from_pairs(package_a, package_b, pairs)
            try:
                validate_map(package_a, package_b, amap, mode)
            except MapError as exc:
                rejected.append({"pairs": len(pairs), "reason": str(exc)[:200]})
                continue
            candidates.append(amap)
    if not candidates:
        raise MapError(f"no placement of the common substructure {result.smartsString} gives a "
                       f"map that validates; first refusals: {[r['reason'] for r in rejected[:3]]}")
    best = max(len(c.pairs) for c in candidates)
    top = [c for c in candidates if len(c.pairs) == best]
    signatures = {}
    for c in top:
        signatures.setdefault(_signature(mol_a, mol_b, c.pairs), c)
    if len(signatures) > 1:
        options = [[[package_a.atom_names[a], package_b.atom_names[b]] for a, b in c.pairs]
                   for c in signatures.values()]
        raise MapError(
            f"the automatic map is chemically ambiguous: {len(signatures)} maps of {best} atoms "
            f"validate and are not related by a symmetry of either molecule, so the choice "
            f"changes the transformation. State one explicitly. Alternatives: {options}")
    chosen = min(top, key=lambda c: c.pairs)
    report = {
        "method": AUTOMATIC_METHOD,
        "mcs_smarts": result.smartsString,
        "mcs_heavy_atoms": result.numAtoms,
        "fmcs": {"atoms": "CompareElements", "bonds": "CompareOrderExact",
                 "ring_matches_ring_only": True, "complete_rings_only": True,
                 "timeout_s": FMCS_TIMEOUT_S},
        "hydrogens": "paired per mapped heavy atom by distance after superposing B's conformer "
                     "on A's mapped heavy atoms",
        "placements_considered": len(seen),
        "valid_candidates": len(candidates),
        "rejected": rejected,
        "symmetry_equivalent_best": len(top),
        "mapped_atoms": best,
    }
    return chosen, report
