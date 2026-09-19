"""Map ligand instances in a structure onto parameter packages.

A deposited structure says where a ligand's heavy atoms are. It does not reliably say which bonds
are double, which atom carries a hydrogen, or whether its atom order matches anything. A package
says all of those things for ONE chemical state, in ITS atom order. This module joins the two,
explicitly and checkably:

  1. a SELECTOR names one residue: `{chain, resid, insertion_code}`, `chain` being the chain id
     the input structure carries (the AUTHOR chain for mmCIF, which is what OpenMM reads). A
     selector matching zero residues or more than one is refused;
  2. the residue's heavy atoms are matched to the package's heavy-atom graph by element and
     connectivity (connectivity perceived from the deposited coordinates, since a PDB carries no
     bond orders), and every match is enumerated;
  3. several matches are acceptable only when they are related by a symmetry of the package's
     FULL chemical graph -- bond orders, formal charges and hydrogens included -- so that choosing
     any of them assigns the same chemistry. A ring flip of a para-substituted phenyl is such a
     case. Two oxygens of a carboxylic acid are not: the skeleton cannot say which one carries
     the hydrogen, and the choice changes the molecule. That is refused unless the selector
     states the mapping. A resonance pair that the graph calls different but the PARAMETERS treat
     identically -- a carboxylate's oxygens, when charges and terms are symmetric -- is accepted,
     because either choice gives the same Hamiltonian;
  4. the package's stereochemistry is checked against the deposited 3D pose;
  5. hydrogens are taken from the package and placed on the deposited heavy atoms, which are not
     moved. The residue in the output topology carries the package's atoms, in package order,
     with package atom names; the deposited residue name, chain, number and insertion code are
     kept.

A ligand instance is bound to its residue IDENTITY and atom NAMES, never to atom indices: every
later preparation step (hydrogens on the protein, solvent, ions) inserts atoms and shifts indices.
`MappedStructure.resolve` finds the instances again in any later topology, and
`assert_instances_unchanged` proves a later step did not alter them.

A covalently attached ligand is refused. A metal within coordination distance is not a covalent
bond; it is recorded as a contact and its bond, if the file declared one, is not kept.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .package import LigandPackage

__all__ = [
    "LigandInstance",
    "entry_selector_and_reference",
    "LigandSelector",
    "MappedStructure",
    "MappingError",
    "assert_instances_unchanged",
    "check_forcefield_compatibility",
    "load_packages_into",
    "map_ligands",
    "unmapped_residues",
]

MAPPING_SCHEMA = "md-tools-ligand-mapping/1"

#: Elements treated as metal ions: a short contact to one is coordination, not a covalent bond.
METAL_ELEMENTS = frozenset({"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co",
                            "Ni", "Cu", "Zn", "Cd", "Hg"})

#: Heavy-atom pairs closer than covalent radii sum + this (nm) are bonded when perceiving the
#: connectivity of a deposited residue, and are a covalent attachment when between the ligand
#: and something else.
BOND_TOLERANCE_NM = 0.045

#: How far a frozen ligand atom may appear to move between preparation steps: the rounding of a
#: PDB coordinate (0.001 A, so at most 0.00005 nm), through which the preparation steps pass
#: structures. Anything larger is a real change.
POSITION_TOLERANCE_NM = 5.1e-5


class MappingError(ValueError):
    """A selector, a residue or a match that cannot be mapped without guessing."""


@dataclass(frozen=True)
class LigandSelector:
    """WHICH residues an entry is about, named the way the structure names them.

    Two ways to say it, and they differ in how many residues they may name:

      `{resname: TYL}`                 every residue with that name. Four copies of one ligand in
                                       an asymmetric unit is ordinary, and they take the same
                                       parameters, so this is the shape that says so once.
      `{chain: B, resid: "201"}`       exactly one residue. `resid` is an identifier, so it is a
                                       quoted string.

    They compose: any key that is stated must match, so `{resname: TYL, chain: B}` is every TYL of
    chain B. What decides whether several matches are allowed is whether `resid` was stated -- a
    residue number names ONE residue, and two residues answering to it is an ambiguous structure,
    not an instruction to map both. A selector that matches nothing is always refused.
    """

    chain: str = ""
    resid: str = ""
    insertion_code: str = ""
    resname: str = ""

    KEYS = ("resname", "chain", "resid", "insertion_code")

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any], *, where: str) -> "LigandSelector":
        unknown = sorted(set(block) - set(cls.KEYS))
        if unknown:
            raise MappingError(f"{where}: unknown selector key(s) {unknown}; a selector is "
                               f"{{resname}} or {{chain, resid, insertion_code}}, or any "
                               f"combination of them")
        if not any(block.get(key) for key in ("resname", "chain", "resid")):
            raise MappingError(f"{where}: selector needs `resname`, or `chain` and `resid`; an "
                               f"empty selector would name every residue in the structure")
        if "resid" in block and "chain" not in block:
            raise MappingError(f"{where}: `resid` without `chain` does not identify a residue -- "
                               f"the same number occurs in every chain. Add `chain`, or select by "
                               f"`resname`.")
        if "insertion_code" in block and "resid" not in block:
            raise MappingError(f"{where}: `insertion_code` distinguishes residues that share a "
                               f"number, so it belongs with `resid`.")
        resid = block.get("resid", "")
        if "resid" in block and not isinstance(resid, str):
            raise MappingError(
                f"{where}: resid must be a quoted string (\"201\"), got {resid!r}. Residue numbers "
                f"are identifiers, not integers: '0201' and '201' may be different residues in a "
                f"file, and a hybrid-36 number is not an integer at all.")
        return cls(chain=str(block.get("chain", "") or ""), resid=str(resid or ""),
                   insertion_code=str(block.get("insertion_code", "") or ""),
                   resname=str(block.get("resname", "") or ""))

    @property
    def names_one_residue(self) -> bool:
        """Whether this selector may match only one residue: a residue NUMBER was stated."""
        return bool(self.resid)

    def label(self) -> str:
        parts = []
        if self.resname:
            parts.append(f"resname {self.resname!r}")
        if self.chain:
            parts.append(f"chain {self.chain!r}")
        if self.resid:
            parts.append(f"resid {self.resid!r}")
            parts.append(f"icode {self.insertion_code!r}")
        return " ".join(parts)

    def matches(self, residue) -> bool:
        if self.resname and residue.name.strip().upper() != self.resname.strip().upper():
            return False
        if self.chain and residue.chain.id != self.chain:
            return False
        if self.resid:
            if str(residue.id).strip() != self.resid:
                return False
            if (residue.insertionCode or "").strip() != self.insertion_code:
                return False
        return True

    def residues(self, topology) -> list:
        """Every residue this selector names, or a refusal saying how many it found."""
        found = [r for r in topology.residues() if self.matches(r)]
        if not found:
            raise MappingError(
                f"{self.label()} matches no residue. Check the residue name, the chain id (the "
                f"AUTHOR chain for mmCIF) and the residue number.")
        if len(found) > 1 and self.names_one_residue:
            raise MappingError(
                f"{self.label()} matches {len(found)} residues; a stated residue number names "
                f"one. Chain ids must be unique across the structure -- an expanded assembly "
                f"whose copies reuse a chain id is ambiguous. Select by `resname` to map every "
                f"copy of a ligand with the same parameters.")
        return found

    def as_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in self.KEYS}


@dataclass
class LigandInstance:
    """One mapped ligand, bound to its residue identity and the package's atom names."""

    selector: LigandSelector
    #: THE RESIDUE this instance is, as the structure identifies it. Not the selector: a
    #: `resname` selector names several residues, so the selector cannot identify one of them,
    #: and everything downstream -- freezing, re-finding after a step that shifts indices,
    #: templates -- is about a residue rather than about the line that selected it.
    residue_key: tuple[str, str, str]
    residue_name: str
    package: LigandPackage = field(repr=False)
    heavy_atom_map: list[dict[str, Any]]
    hydrogen_source: str
    symmetry: dict[str, Any]
    coordination_contacts: list[dict[str, Any]]
    dropped_bonds: list[dict[str, Any]]
    positions_nm: np.ndarray = field(repr=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.residue_key

    def label(self) -> str:
        chain, resid, icode = self.residue_key
        return (f"{self.residue_name} chain {chain!r} resid {resid!r}"
                + (f" icode {icode!r}" if icode else ""))

    def is_this_residue(self, residue) -> bool:
        return (residue.chain.id, str(residue.id).strip(),
                (residue.insertionCode or "").strip()) == self.residue_key

    @property
    def template_name(self) -> str:
        return self.package.template_name

    def find(self, topology):
        """The residue carrying this instance in *topology*, verified by atom names; or a refusal."""
        found = [r for r in topology.residues() if self.is_this_residue(r)]
        if len(found) != 1:
            raise MappingError(f"ligand instance {self.label()}: {len(found)} residues "
                               f"match in this topology, expected exactly 1")
        residue = found[0]
        names = tuple(a.name for a in residue.atoms())
        if names != tuple(self.package.atom_names):
            raise MappingError(f"ligand instance {self.label()}: its atoms are {names}, "
                               f"not the package's {tuple(self.package.atom_names)}")
        return residue

    def record(self, topology=None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "selector": self.selector.as_dict(),
            # WHICH residue the selector resolved to. A `resname` selector names several, so the
            # selector alone does not say which one this instance is.
            "residue": {"chain": self.residue_key[0], "resid": self.residue_key[1],
                        "insertion_code": self.residue_key[2]},
            "residue_name": self.residue_name,
            "package": self.package.summary(),
            "heavy_atom_map": self.heavy_atom_map,
            "hydrogens": {"source": self.hydrogen_source,
                          "names": [n for n, a in zip(self.package.atom_names,
                                                      self.package.mol.GetAtoms())
                                    if a.GetAtomicNum() == 1]},
            "symmetry": self.symmetry,
            "coordination_contacts": self.coordination_contacts,
            "dropped_bonds": self.dropped_bonds,
        }
        if topology is not None:
            residue = self.find(topology)
            record["resolved"] = {
                "chain_index": residue.chain.index, "residue_index": residue.index,
                "atom_indices": [a.index for a in residue.atoms()],
                "package_atom_to_topology_index": {
                    str(i): a.index for i, a in enumerate(residue.atoms())},
            }
        return record


@dataclass
class MappedStructure:
    """The structure with every selected ligand replaced by its package's atoms."""

    topology: Any
    positions: Any
    instances: list[LigandInstance]

    @property
    def frozen_residues(self) -> set[tuple[str, str, str]]:
        """(chain, resid, insertion_code) of every mapped instance: never re-protonated."""
        return {instance.key for instance in self.instances}

    @property
    def packages(self) -> list[LigandPackage]:
        seen: dict[str, LigandPackage] = {}
        for instance in self.instances:
            seen.setdefault(instance.package.parameter_id, instance.package)
        return list(seen.values())

    def resolve(self, topology) -> dict[tuple[str, str, str], Any]:
        """Each instance's residue in *topology*, verified by atom names."""
        return {instance.key: instance.find(topology) for instance in self.instances}

    def residue_templates(self, topology) -> dict[Any, str]:
        """`{Residue: template name}` for ForceField.createSystem, addHydrogens and addSolvent."""
        return {residue: instance.template_name
                for instance, residue in zip(self.instances, self.resolve(topology).values())}

    def record(self, topology=None) -> dict[str, Any]:
        return {"schema": MAPPING_SCHEMA,
                "packages": [p.summary() for p in self.packages],
                "instances": [i.record(topology) for i in self.instances]}


# ------------------------------------------------------------------------------------------------
# graphs
# ------------------------------------------------------------------------------------------------
def _package_graph(mol, *, heavy_only: bool):
    import networkx as nx

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if heavy_only and atom.GetAtomicNum() == 1:
            continue
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol(),
                       formal_charge=atom.GetFormalCharge())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in graph and b in graph:
            graph.add_edge(a, b, order=bond.GetBondTypeAsDouble())
    return graph


def _perceived_graph(elements: Sequence[str], positions_nm: np.ndarray):
    """Connectivity from coordinates: covalent radii plus a tolerance. No bond orders."""
    import networkx as nx
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    graph = nx.Graph()
    for i, element in enumerate(elements):
        graph.add_node(i, element=element)
    radii = [table.GetRcovalent(table.GetAtomicNumber(e)) / 10.0 for e in elements]
    for i, j in itertools.combinations(range(len(elements)), 2):
        if elements[i] in METAL_ELEMENTS or elements[j] in METAL_ELEMENTS:
            continue
        if np.linalg.norm(positions_nm[i] - positions_nm[j]) < radii[i] + radii[j] + BOND_TOLERANCE_NM:
            graph.add_edge(i, j)
    return graph


def _all_matches(package_heavy, deposited) -> list[dict[int, int]]:
    """Every element-preserving isomorphism package-heavy-atom -> deposited-atom."""
    from networkx.algorithms.isomorphism import GraphMatcher

    matcher = GraphMatcher(package_heavy, deposited,
                           node_match=lambda a, b: a["element"] == b["element"])
    return [dict(m) for m in matcher.isomorphisms_iter()]


def _full_automorphisms_on_heavy(mol) -> set[tuple[int, ...]]:
    """Heavy-atom permutations that extend to a symmetry of the FULL chemical graph."""
    from networkx.algorithms.isomorphism import GraphMatcher

    full = _package_graph(mol, heavy_only=False)
    matcher = GraphMatcher(
        full, full,
        node_match=lambda a, b: (a["element"], a["formal_charge"]) == (b["element"], b["formal_charge"]),
        edge_match=lambda a, b: a["order"] == b["order"])
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return {tuple(m[i] for i in heavy) for m in matcher.isomorphisms_iter()}


def _parameters_invariant(package: LigandPackage, heavy_permutation: Mapping[int, int]) -> bool:
    """Whether relabelling the package's atoms by this permutation leaves its parameters unchanged.

    The graph test above is strict: a carboxylate's two oxygens are not graph-symmetric, because
    one is written with a double bond and a zero charge. Physically they are one resonance pair,
    and if the force field and the charges treat them identically then choosing either mapping
    gives the same Hamiltonian -- which is the question that actually matters. The permutation
    is extended to hydrogens parent by parent; a heavy atom mapped to one with a different number
    of hydrogens is never invariant (that is the carboxylic acid).
    """
    import math

    mol = package.mol
    full: dict[int, int] = dict(heavy_permutation)
    for source, target in heavy_permutation.items():
        hs = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(source).GetNeighbors()
                    if n.GetAtomicNum() == 1)
        ht = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(target).GetNeighbors()
                    if n.GetAtomicNum() == 1)
        if len(hs) != len(ht):
            return False
        full.update(zip(hs, ht))
    table = package.table

    def close(a, b):
        return all(x == y if isinstance(x, (str, int)) and not isinstance(x, float)
                   else math.isclose(float(x), float(y), rel_tol=1e-8, abs_tol=1e-10)
                   for x, y in zip(a, b))

    for i, row in enumerate(table["atoms"]):
        if not close(row, table["atoms"][full[i]]):
            return False

    def relabel(rows, width, symmetric):
        out = []
        for row in rows:
            atoms = [full[a] for a in row[:width]]
            if symmetric and atoms[::-1] < atoms:
                atoms = atoms[::-1]
            out.append([*atoms, *row[width:]])
        return sorted(out)

    for key, width, symmetric in (("bonds", 2, True), ("angles", 3, True),
                                  ("proper_torsions", 4, True), ("exceptions", 2, True)):
        mapped, original = relabel(table[key], width, symmetric), sorted(table[key])
        if len(mapped) != len(original) or not all(
                a[:width] == b[:width] and close(a[width:], b[width:])
                for a, b in zip(mapped, original)):
            return False
    impropers = table["improper_torsions"]
    if impropers:
        # An improper's atom order is not canonical, so compare the multiset of parameters per
        # central-atom set rather than per ordered quartet.
        def by_set(rows):
            grouped: dict[frozenset, list] = {}
            for row in rows:
                grouped.setdefault(frozenset(row[:4]), []).append(tuple(row[4:]))
            return {k: sorted(v) for k, v in grouped.items()}

        mapped = by_set([[full[a] for a in row[:4]] + list(row[4:]) for row in impropers])
        original = by_set(impropers)
        if set(mapped) != set(original) or not all(
                len(mapped[k]) == len(original[k]) and all(close(x, y) for x, y in
                                                           zip(mapped[k], original[k]))
                for k in original):
            return False
    return True


# ------------------------------------------------------------------------------------------------
# hydrogens and stereochemistry
# ------------------------------------------------------------------------------------------------
def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc, tc = source.mean(axis=0), target.mean(axis=0)
    h = (source - sc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, sc, tc


def _place_hydrogens(mol, heavy_positions: dict[int, np.ndarray]) -> np.ndarray:
    """Package hydrogens moved onto the deposited heavy atoms by local rigid superposition.

    For each hydrogen, the package conformer's heavy atoms within two bonds of its parent are
    superposed on their deposited positions and the hydrogen is carried along. Local, so a ligand
    whose deposited torsions differ from the package conformer still gets hydrogens in the right
    place on each fragment. Rotor positions (methyl, hydroxyl) are the package conformer's and are
    relaxed afterwards.
    """
    reference = mol.GetConformer().GetPositions() / 10.0
    positions = np.array(reference)
    for index, point in heavy_positions.items():
        positions[index] = point
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        near = {parent}
        frontier = {parent}
        for _ in range(2):
            frontier = {n.GetIdx() for f in frontier for n in mol.GetAtomWithIdx(f).GetNeighbors()
                        if n.GetAtomicNum() > 1} - near
            near |= frontier
        if len(near) < 3:
            near = set(heavy_positions)
        anchor = sorted(near)
        rotation, sc, tc = _kabsch(reference[anchor], np.array([heavy_positions[i] for i in anchor]))
        positions[atom.GetIdx()] = rotation @ (reference[atom.GetIdx()] - sc) + tc
    return positions


def _relax_hydrogens(package: LigandPackage, positions_nm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Minimise the ligand's hydrogens alone, heavy atoms fixed, on the Reference platform."""
    import openmm
    from openmm import unit

    from .parameters import ligand_system

    system = ligand_system(package.ffxml_text, package.mol, package.atom_names,
                           package.template_name)
    for atom in package.mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            system.setParticleMass(atom.GetIdx(), 0.0)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions_nm * unit.nanometer)
    before = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, 500)
    state = context.getState(getEnergy=True, getPositions=True)
    after = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    relaxed = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    heavy = [a.GetIdx() for a in package.mol.GetAtoms() if a.GetAtomicNum() > 1]
    moved = float(np.max(np.linalg.norm(relaxed[heavy] - positions_nm[heavy], axis=1)))
    if moved > 0.0:
        raise MappingError(f"internal: hydrogen relaxation moved a heavy atom by {moved} nm")
    return relaxed, {"method": "hydrogens only, heavy atoms fixed, ligand in vacuum, package "
                               "parameters, Reference platform, LocalEnergyMinimizer tol 1.0 "
                               "kJ/mol/nm, max 500 iterations",
                     "energy_before_kj_mol": before, "energy_after_kj_mol": after}


def _check_stereo(package: LigandPackage, positions_nm: np.ndarray, where: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    probe = Chem.Mol(package.mol)
    conformer = probe.GetConformer()
    for i, xyz in enumerate(positions_nm * 10.0):
        conformer.SetAtomPosition(i, Point3D(*map(float, xyz)))
    Chem.AssignStereochemistryFrom3D(probe)
    placed = Chem.MolToSmiles(probe)
    expected = Chem.MolToSmiles(package.mol)
    if placed != expected:
        raise MappingError(
            f"{where}: the deposited pose has different stereochemistry from package "
            f"{package.reference}: the pose reads {placed}, the package is {expected}. A package "
            f"describes one stereoisomer; this instance needs the package for the other.")
    return {"checked": True, "isomeric_smiles": expected}


# ------------------------------------------------------------------------------------------------
# mapping
# ------------------------------------------------------------------------------------------------
def entry_selector_and_reference(entry: Mapping[str, Any], *,
                                 where: str) -> tuple[LigandSelector, str]:
    """The selector and the package an entry names, in either spelling it may be written in.

    Two shapes, because they say the same thing:

        - select: {chain: B, resid: "201"}       a selector under `select`, with the package
          parameters: CHEMBL112/param_...        beside it as a catalog reference;
        - {resname: TYL, parameter: /path/...}   the selector keys written directly in the entry,
                                                 with the package as a path to its directory.

    `parameters` is a catalog reference, resolved against the configured catalogs; `parameter` is a
    filesystem path to a package directory, for a build that has one locally and no catalog.
    Exactly one of them, because two would have to agree and nothing would check that they do.
    """
    keys = set(entry)
    known = {"select", "parameters", "parameter", "atom_map", *LigandSelector.KEYS}
    unknown = sorted(keys - known)
    if unknown:
        raise MappingError(f"{where}: unknown key(s) {unknown}")
    references = [key for key in ("parameters", "parameter") if entry.get(key)]
    if len(references) != 1:
        raise MappingError(
            f"{where}: needs exactly one of `parameters` (a <compound>/param_<id> catalog "
            f"reference) or `parameter` (a path to a package directory); "
            f"{'both are set' if references else 'neither is set'}.")
    inline = {key: entry[key] for key in LigandSelector.KEYS if key in entry}
    if "select" in entry and inline:
        raise MappingError(f"{where}: the selector is written both under `select` and directly in "
                           f"the entry ({sorted(inline)}); write it once.")
    block = entry["select"] if "select" in entry else inline
    if not isinstance(block, Mapping):
        raise MappingError(f"{where}.select: expected a mapping of selector keys")
    selector = LigandSelector.from_mapping(block, where=f"{where}.select" if "select" in entry
                                           else where)
    return selector, str(entry[references[0]])


def unmapped_residues(topology, selectors: Iterable[LigandSelector], *,
                      known_residue_names: Iterable[str]) -> list[dict[str, str]]:
    """Residues no force-field template names and no selector covers: they cannot be built."""
    known = {n.upper() for n in known_residue_names}
    covered = list(selectors)
    out = []
    for residue in topology.residues():
        if residue.name.upper() in known or any(s.matches(residue) for s in covered):
            continue
        out.append({"chain": residue.chain.id, "resid": str(residue.id).strip(),
                    "insertion_code": (residue.insertionCode or "").strip(),
                    "residue_name": residue.name})
    return out


def _map_one(topology, positions_nm: np.ndarray, residue, selector: LigandSelector,
             package: LigandPackage, explicit_map: Optional[Mapping[str, str]],
             where: str) -> tuple[LigandInstance, np.ndarray, list]:
    atoms = list(residue.atoms())
    heavy = [a for a in atoms if a.element is not None and a.element.symbol != "H"]
    hydrogens = [a for a in atoms if a.element is not None and a.element.symbol == "H"]
    if any(a.element is None for a in atoms):
        raise MappingError(f"{where}: atoms without an element in residue {residue.name}; the "
                           f"element column is needed to match the chemical graph")
    mol = package.mol
    package_heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) != len(package_heavy_idx):
        raise MappingError(
            f"{where}: residue {residue.name} has {len(heavy)} heavy atoms, package "
            f"{package.reference} has {len(package_heavy_idx)}. Missing or extra atoms are not "
            f"filled in or dropped: rebuild the residue, or use a package for what is there.")

    elements = [a.element.symbol for a in heavy]
    heavy_xyz = np.array([positions_nm[a.index] for a in heavy])
    deposited = _perceived_graph(elements, heavy_xyz)
    package_heavy = _package_graph(mol, heavy_only=True)

    if explicit_map is not None:
        by_name = {a.name: i for i, a in enumerate(heavy)}
        package_by_name = {n: i for i, n in enumerate(package.atom_names)}
        unknown = sorted(set(explicit_map) - set(by_name)) + sorted(
            set(explicit_map.values()) - set(package_by_name))
        if unknown or len(explicit_map) != len(heavy):
            raise MappingError(f"{where}: atom_map must name every heavy atom once (deposited name "
                               f"-> package name); unknown or missing: {unknown or 'count'}")
        match = {package_by_name[p]: by_name[d] for d, p in explicit_map.items()}
        if not all(deposited.has_edge(match[a], match[b]) for a, b in package_heavy.edges()) or \
                deposited.number_of_edges() != package_heavy.number_of_edges() or \
                any(elements[match[i]] != mol.GetAtomWithIdx(i).GetSymbol() for i in match):
            raise MappingError(f"{where}: the stated atom_map is not an element- and "
                               f"bond-preserving map onto the deposited connectivity")
        matches = [match]
        symmetry = {"source": "atom_map stated in the configuration", "n_graph_matches": None}
    else:
        matches = _all_matches(package_heavy, deposited)
        if not matches:
            raise MappingError(
                f"{where}: the heavy atoms of residue {residue.name} do not have the connectivity "
                f"of package {package.reference} ({package.metadata['chemical_state']['canonical_smiles']}). "
                f"Deposited: {deposited.number_of_edges()} bonds perceived from coordinates, "
                f"package: {package_heavy.number_of_edges()}. Either this is a different "
                f"compound, or its coordinates are distorted enough that bonds are misperceived.")
        ordered = sorted(matches, key=lambda m: tuple(m[i] for i in package_heavy_idx))
        chosen = ordered[0]
        automorphisms = _full_automorphisms_on_heavy(mol)
        inverse = {v: k for k, v in chosen.items()}
        consequential = []
        for other in ordered[1:]:
            permutation = tuple(inverse[other[i]] for i in package_heavy_idx)
            if permutation not in automorphisms and not _parameters_invariant(
                    package, dict(zip(package_heavy_idx, permutation))):
                consequential.append({package.atom_names[i]: package.atom_names[inverse[other[i]]]
                                      for i in package_heavy_idx if inverse[other[i]] != i})
        if consequential:
            raise MappingError(
                f"{where}: residue {residue.name} matches package {package.reference} in "
                f"{len(matches)} ways that are NOT chemically equivalent: the skeleton is "
                f"symmetric but the package's bond orders, charges or hydrogens are not "
                f"(for example the swap {consequential[0]}). Choosing one would decide the "
                f"chemistry by atom order. State the mapping explicitly with `atom_map: "
                f"{{deposited name: package name}}` in this ligand's entry.")
        matches = [chosen]
        symmetry = {"source": "graph match", "n_graph_matches": len(ordered),
                    "all_matches_equivalent": True,
                    "equivalence": ("each alternative is a symmetry of the package's full "
                                    "chemical graph, or leaves every parameter of the package "
                                    "unchanged (a resonance pair such as a carboxylate)")}
    match = matches[0]

    heavy_positions = {p: heavy_xyz[d] for p, d in match.items()}
    placed = _place_hydrogens(mol, heavy_positions)
    hydrogen_source = "package hydrogens placed by local superposition, then relaxed"
    if hydrogens:
        placed, hydrogen_source = _adopt_deposited_hydrogens(
            mol, match, heavy, hydrogens, positions_nm, placed, where)
        relax_record = {"method": "not relaxed: deposited hydrogen positions kept"}
    else:
        placed, relax_record = _relax_hydrogens(package, placed)
    symmetry["stereochemistry"] = _check_stereo(package, placed, where)
    symmetry["hydrogen_relaxation"] = relax_record

    heavy_map = [{"package_index": p, "package_name": package.atom_names[p],
                  "deposited_name": heavy[d].name,
                  "deposited_serial": getattr(heavy[d], "id", None)}
                 for p, d in sorted(match.items())]
    instance = LigandInstance(
        selector=selector,
        residue_key=(residue.chain.id, str(residue.id).strip(),
                     (residue.insertionCode or "").strip()),
        residue_name=residue.name, package=package, heavy_atom_map=heavy_map,
        hydrogen_source=hydrogen_source, symmetry=symmetry, coordination_contacts=[],
        dropped_bonds=[], positions_nm=placed)
    return instance, placed, heavy


def _adopt_deposited_hydrogens(mol, match, heavy, hydrogens, positions_nm, placed, where):
    """Keep deposited hydrogens if -- and only if -- they are exactly the package's hydrogens."""
    deposited_parent: dict[int, list] = {}
    heavy_index = {a.index: i for i, a in enumerate(heavy)}
    for h in hydrogens:
        distances = [(np.linalg.norm(positions_nm[h.index] - positions_nm[a.index]), a.index)
                     for a in heavy]
        _, parent = min(distances)
        deposited_parent.setdefault(heavy_index[parent], []).append(h)
    inverse = {d: p for p, d in match.items()}
    out = np.array(placed)
    for p_idx in match:
        package_h = [n.GetIdx() for n in mol.GetAtomWithIdx(p_idx).GetNeighbors()
                     if n.GetAtomicNum() == 1]
        present = deposited_parent.get(match[p_idx], [])
        if len(present) != len(package_h):
            raise MappingError(
                f"{where}: deposited atom {heavy[match[p_idx]].name} carries {len(present)} "
                f"hydrogen(s) but the package gives it {len(package_h)}. That is a different "
                f"protonation or tautomeric state; a package's hydrogens are never changed to fit, "
                f"and this instance needs a package for its own state.")
        remaining = list(present)
        for h_idx in package_h:
            best = min(remaining, key=lambda h: np.linalg.norm(positions_nm[h.index] - placed[h_idx]))
            remaining.remove(best)
            out[h_idx] = positions_nm[best.index]
    return out, "deposited hydrogens, matched to the package's hydrogens by parent atom"


def map_ligands(topology, positions, entries: Sequence[Mapping[str, Any]],
                packages: Mapping[str, LigandPackage]) -> MappedStructure:
    """Replace each selected residue with its package's atoms. Refuses rather than guesses.

    `entries` are `{select: {...}, parameters: "<compound>/<param>", atom_map: {...}?}` blocks;
    `packages` maps each `parameters` reference to its loaded package.
    """
    from openmm import app, unit

    positions_nm = np.array(positions.value_in_unit(unit.nanometer)
                            if hasattr(positions, "value_in_unit") else positions, dtype=float)
    plans = []
    claimed: dict[int, str] = {}
    for n, entry in enumerate(entries):
        where = f"ligands[{n}]"
        selector, reference = entry_selector_and_reference(entry, where=where)
        if reference not in packages:
            raise MappingError(f"{where}: package {reference!r} is not loaded")
        # EVERY residue the selector names. A `resname` selector maps every copy of that ligand to
        # the same package, which is what four copies of one compound in an asymmetric unit need;
        # a stated residue number still names exactly one, and `residues` refuses anything else.
        for residue in selector.residues(topology):
            if residue.index in claimed:
                raise MappingError(
                    f"{where}: {selector.label()} names residue {residue.name} "
                    f"{residue.chain.id}:{str(residue.id).strip()}, which {claimed[residue.index]} "
                    f"already names. One residue takes its parameters from one entry.")
            claimed[residue.index] = where
            plans.append((selector, residue, packages[reference], entry.get("atom_map"), where))

    instances: dict[int, tuple[LigandInstance, np.ndarray, list]] = {}
    for selector, residue, package, atom_map, where in plans:
        instance, placed, heavy = _map_one(topology, positions_nm, residue, selector, package,
                                           atom_map, where)
        instance.coordination_contacts = _contacts(topology, positions_nm, residue, heavy,
                                                   instance, where)
        instances[residue.index] = (instance, placed, heavy)

    # Rebuild the topology in the original order, with each mapped residue replaced.
    new = app.Topology()
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    new_positions: list[np.ndarray] = []
    atom_lookup = {}
    for chain in topology.chains():
        new_chain = new.addChain(chain.id)
        for residue in chain.residues():
            new_residue = new.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            if residue.index in instances:
                instance, placed, _ = instances[residue.index]
                package_atoms = []
                for i, atom in enumerate(instance.package.mol.GetAtoms()):
                    element = app.Element.getByAtomicNumber(atom.GetAtomicNum())
                    package_atoms.append(new.addAtom(instance.package.atom_names[i], element,
                                                     new_residue))
                    new_positions.append(placed[i])
                for bond in instance.package.mol.GetBonds():
                    new.addBond(package_atoms[bond.GetBeginAtomIdx()],
                                package_atoms[bond.GetEndAtomIdx()])
            else:
                for atom in residue.atoms():
                    atom_lookup[atom.index] = new.addAtom(atom.name, atom.element, new_residue,
                                                          atom.id)
                    new_positions.append(positions_nm[atom.index])
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        a_mapped, b_mapped = a.residue.index in instances, b.residue.index in instances
        if not a_mapped and not b_mapped:
            new.addBond(atom_lookup[a.index], atom_lookup[b.index], bond.type, bond.order)
            continue
        if a_mapped and b_mapped and a.residue.index == b.residue.index:
            continue                                        # replaced by the package's own bonds
        mapped_atom, other = (a, b) if a_mapped else (b, a)
        instance = instances[mapped_atom.residue.index][0]
        if other.element is not None and other.element.symbol in METAL_ELEMENTS:
            instance.dropped_bonds.append({
                "ligand_atom": mapped_atom.name, "partner": _atom_label(other),
                "reason": "declared bond to a metal: coordination is not a covalent bond in a "
                          "single-residue package; the metal site model owns it"})
            continue
        raise MappingError(
            f"ligand {instance.label()}: the input declares a covalent bond from "
            f"{mapped_atom.name} to {_atom_label(other)}. A covalently attached ligand is not "
            f"supported by a single-residue package; it needs an explicit validated "
            f"representation of the linkage.")
    ordered = [instances[k][0] for k in sorted(instances)]
    return MappedStructure(topology=new, positions=np.array(new_positions) * unit.nanometer,
                           instances=ordered)


def _atom_label(atom) -> str:
    residue = atom.residue
    return (f"{atom.name} of {residue.name} chain {residue.chain.id!r} resid "
            f"{str(residue.id).strip()!r}{(' icode ' + repr(residue.insertionCode)) if (residue.insertionCode or '').strip() else ''}")


def _contacts(topology, positions_nm, residue, heavy, instance, where) -> list[dict[str, Any]]:
    """Metal contacts are recorded; any other covalent-distance contact is refused."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    own = {a.index for a in residue.atoms()}
    contacts = []
    others = [a for a in topology.atoms()
              if a.index not in own and a.element is not None and a.element.symbol != "H"]
    if not others:
        return contacts
    other_xyz = np.array([positions_nm[a.index] for a in others])
    for atom in heavy:
        distances = np.linalg.norm(other_xyz - positions_nm[atom.index], axis=1)
        for k in np.nonzero(distances < 0.30)[0]:
            other = others[int(k)]
            d = float(distances[k])
            if other.element.symbol in METAL_ELEMENTS:
                contacts.append({"ligand_atom": atom.name, "metal": _atom_label(other),
                                 "distance_nm": round(d, 4)})
                continue
            limit = (table.GetRcovalent(atom.element.atomic_number)
                     + table.GetRcovalent(other.element.atomic_number)) / 10.0 + BOND_TOLERANCE_NM
            if d < limit:
                raise MappingError(
                    f"{where}: ligand atom {atom.name} is {d:.3f} nm from {_atom_label(other)}, "
                    f"within covalent bonding distance. A covalently attached ligand is not "
                    f"supported by a single-residue package; if this is a clash or an alternate "
                    f"conformer, fix the input structure.")
    return contacts


# ------------------------------------------------------------------------------------------------
# force fields
# ------------------------------------------------------------------------------------------------
def check_forcefield_compatibility(forcefield, packages: Iterable[LigandPackage]) -> dict[str, Any]:
    """Refuse a package whose nonbonded conventions differ from the force field it joins.

    Checked before the package is loaded, within OpenMM's own merge tolerance
    (`NonbondedGenerator.SCALETOL`, 1e-5). This used to be exact, so that a near-miss would not be
    built QUIETLY under the protein's convention; but OpenMM merges any definitions within that
    tolerance and applies the FIRST one to the whole System, and several shipped water XMLs write
    5/6 as `0.833333`, so the exact check refused every package under OPC. A difference below
    1e-5 in a 1-4 scale is not a different convention, and it is no longer quiet: the report
    records, per package, its own scales beside the ones the System applies.
    """
    from openmm.app.forcefield import NonbondedGenerator

    generators = [g for g in forcefield._forces if isinstance(g, NonbondedGenerator)]
    custom = sorted({type(g).__name__ for g in forcefield._forces
                     if "CustomNonbonded" in type(g).__name__ or "LennardJones" in type(g).__name__})
    report = {"packages": [], "force_field_nonbonded": None}
    if custom:
        raise MappingError(
            f"the force field uses {custom}; ligand packages carry Lorentz-Berthelot "
            f"Lennard-Jones in a NonbondedForce, and mixing them with a different Lennard-Jones "
            f"representation is not supported")
    if generators:
        generator = generators[0]
        report["force_field_nonbonded"] = {"coulomb14scale": generator.coulomb14scale,
                                           "lj14scale": generator.lj14scale}
    # OPENMM'S OWN CRITERION, not bit-equality. Several shipped water XMLs write 5/6 rounded to
    # `0.833333` (amber14/opc.xml, amber19/opc.xml and their opc3 siblings), and whichever
    # NonbondedForce definition loads first is the one the whole System applies, the ligand's
    # exceptions included; OpenMM merges any later definition within `SCALETOL` into it. An exact
    # comparison therefore refused every package in every OPC build -- the documented ff19SB
    # alternative -- over a difference OpenMM itself treats as none. What is RECORDED is the
    # value actually applied, beside each package's own.
    tolerance = NonbondedGenerator.SCALETOL
    packages = list(packages)
    # WHAT THE SYSTEM WILL APPLY. With a force field already carrying a NonbondedForce definition,
    # that one. With none -- a vacuum build loads no protein and no water XML -- the first
    # package loaded defines it, and every later one merges into it within the tolerance, so its
    # own scales are the applied ones. Recorded either way, so a reader never has to infer it.
    if generators:
        applied_scales = (generators[0].coulomb14scale, generators[0].lj14scale)
    elif packages:
        first = packages[0].conventions
        applied_scales = (first["coulomb14scale"], first["lj14scale"])
    for package in packages:
        conventions = package.conventions
        entry = {"reference": package.reference, **{k: conventions[k] for k in
                                                   ("coulomb14scale", "lj14scale")}}
        entry["applied"] = {"coulomb14scale": applied_scales[0],
                            "lj14scale": applied_scales[1], "tolerance": tolerance,
                            "defined_by": ("the force field's first NonbondedForce definition"
                                           if generators else "the first package loaded")}
        report["packages"].append(entry)
        if (abs(conventions["coulomb14scale"] - applied_scales[0]) > tolerance
                or abs(conventions["lj14scale"] - applied_scales[1]) > tolerance):
            raise MappingError(
                f"package {package.reference} uses 1-4 scales coulomb "
                f"{conventions['coulomb14scale']!r} / LJ {conventions['lj14scale']!r}, the force "
                f"field it is combined with uses {applied_scales[0]!r} / "
                f"{applied_scales[1]!r} (compared within OpenMM's own {tolerance:g}). "
                f"One System has one convention; a package "
                f"parameterised for another one needs a package made for this force field.")
    return report


def load_packages_into(forcefield, packages: Iterable[LigandPackage]) -> dict[str, Any]:
    """Check compatibility, then load each distinct package's ffxml once."""
    import io

    distinct: dict[str, LigandPackage] = {}
    for package in packages:
        distinct.setdefault(package.parameter_id, package)
    report = check_forcefield_compatibility(forcefield, distinct.values())
    for package in distinct.values():
        forcefield.loadFile(io.StringIO(package.ffxml_text))
    return report


def assert_instances_unchanged(before: MappedStructure, topology, positions, *,
                               step: str) -> None:
    """Every mapped instance is still exactly as mapped: atoms, names, elements, bonds, positions.

    Called after every preparation step that rebuilds the topology. A step that re-added a
    hydrogen, renamed an atom or nudged a coordinate would otherwise be invisible: the package
    template still matches by graph, and the System builds.
    """
    from openmm import unit

    xyz = np.array(positions.value_in_unit(unit.nanometer) if hasattr(positions, "value_in_unit")
                   else positions, dtype=float)
    bonds_by_residue: dict[int, set] = {}
    for a, b in topology.bonds():
        if a.residue is b.residue:
            bonds_by_residue.setdefault(a.residue.index, set()).add(
                tuple(sorted((a.name, b.name))))
        else:
            for atom, other in ((a, b), (b, a)):
                bonds_by_residue.setdefault(atom.residue.index, set()).add(
                    ("EXTERNAL", atom.name, other.residue.name))
    for instance in before.instances:
        try:
            residue = instance.find(topology)
        except MappingError as exc:
            raise MappingError(f"after {step}: {exc}") from exc
        atoms = list(residue.atoms())
        expected_elements = [a.GetSymbol() for a in instance.package.mol.GetAtoms()]
        if [a.element.symbol for a in atoms] != expected_elements:
            raise MappingError(f"after {step}: ligand {instance.label()} elements changed")
        expected_bonds = {tuple(sorted((instance.package.atom_names[b.GetBeginAtomIdx()],
                                        instance.package.atom_names[b.GetEndAtomIdx()])))
                          for b in instance.package.mol.GetBonds()}
        if bonds_by_residue.get(residue.index, set()) != expected_bonds:
            raise MappingError(f"after {step}: ligand {instance.label()} bonds changed")
        moved = np.max(np.abs(xyz[[a.index for a in atoms]] - instance.positions_nm))
        if moved > POSITION_TOLERANCE_NM:
            raise MappingError(f"after {step}: ligand {instance.label()} atoms moved by "
                               f"up to {moved:.2e} nm")
