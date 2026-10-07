"""Symmetry-aware torsional distance: molecular symmetry is enumerated BEFORE clustering.

THE QUESTION THIS ANSWERS. A torsion vector describes a configuration through a particular
LABELLING of its atoms. A molecular symmetry -- an atom permutation that preserves the chemical
graph -- produces another labelling of the SAME configuration, and with it a different torsion
vector. Paracetamol's ring flip exchanges the two ortho carbons, so the `aryl` torsion measured to
one ortho carbon becomes the torsion measured to the other. Nothing moved: the atoms were
RELABELLED. A distance between configurations should therefore not depend on which of the
equivalent labellings each frame happens to be written in, and the one that does not is

    d_sym(x, y) = min_{g, h in G} || Phi(g x) - Phi(h y) ||

with `Phi` the periodic embedding `(..., sqrt(w_j) cos(theta_j), sqrt(w_j) sin(theta_j), ...)`.

WHEN THE ONE-SIDED FORM IS EXACT, AND WHY THIS MODULE CONSTRUCTS THAT CASE. If every `g` acts on
the descriptor as an ISOMETRY of the embedding, then `||Phi(g x) - Phi(h y)|| = ||Phi(x) -
Phi(g^-1 h y)||`, and since `g^-1 h` ranges over the whole group,

    d_sym(x, y) = min_{g in G} || Phi(x) - Phi(g y) ||

which costs |G| evaluations instead of |G|^2. The action is an isometry exactly when (1) every
operation maps the set of descriptor torsions ONTO itself, so that it acts as a permutation of
columns, and (2) the metric weights are constant on every orbit of that permutation group. An
arbitrary selection satisfies neither: paracetamol's `aryl` torsion (1,3,4,5) maps to (1,3,4,10),
which nobody selected, so the flip does not even act on the selected columns. `
SymmetricTorsionDescriptor` therefore CLOSES the selection under the group, adding each missing
image as a column recomputed from coordinates, and NORMALISES the weights so that every orbit
carries the total weight of the torsions selected in it, split equally among its members. Both
conditions are then checked, not assumed, and a failure raises.

WHAT IS NOT DONE, because it is wrong. Choosing one "canonical" labelling per frame and taking
ordinary Euclidean distances between the canonical representatives does NOT give d_sym: two frames
close to each other on opposite sides of the canonicalisation boundary get different
representatives and a large distance. The minimum is taken per PAIR, here. Nor is a symmetry
enumeration ever appended to the data as an extra frame: the enumerations are alternative
DESCRIPTIONS of one observation, and adding them would multiply frame counts and statistical
weights by the group order.

WHAT THE SYMMETRY IS, AND IS NOT. The operations come from the chemical GRAPH (element, bond order,
formal charge, isotope, specified stereochemistry), found by RDKit matching the molecule against
itself. That establishes STRUCTURAL equivalence of relabelled configurations. It does not
establish that symmetry-related basins carry equal thermodynamic population: an atom-specific
restraint, an atom-indexed parameter or a REST2 region covering one side breaks the energetic
symmetry and leaves the graph unchanged.

NEEDS THE `analysis` EXTRA (rdkit). Imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._t_symmetry import (MAX_AUTOMORPHISMS, TorsionDefinitions, _clean_for_automorphism,
                          _require_bonded_chain, dihedral, wrap)

#: Two squared distances closer than this (relative to the largest possible squared distance,
#: `4 W`) are a TIE, and the earlier operation wins. Without a tolerance, the minimising operation
#: of two exactly symmetry-related candidates would be decided by the last bit of a dot product,
#: and the answer reported for inspection would change between machines.
TIE_RELATIVE_TOLERANCE = 1e-12

#: Default agreement required between a STORED torsion column and the same quadruplet recomputed
#: from the coordinates, in radians. Its job is to catch a different frame order or a wrong atom
#: mapping, which produce disagreements of tenths of a radian; single-precision trajectories and
#: float32 dihedral code disagree with float64 at 1e-5.
STORED_AGREEMENT_TOLERANCE = 1e-3

__all__ = ["IncompleteSymmetryEnumeration", "MissingCoordinatesError", "SymmetryEnumeration",
           "DescriptorOperation", "SymmetricTorsionDescriptor", "enumerate_symmetry",
           "check_atom_mapping", "whole_molecule_coordinates", "minimum_image",
           "draw_symmetry_operation",
           "TIE_RELATIVE_TOLERANCE", "STORED_AGREEMENT_TOLERANCE"]


class IncompleteSymmetryEnumeration(RuntimeError):
    """The automorphism search stopped at its cap. Using what it found would make "no further
    symmetry" indistinguishable from "stopped looking", so nothing downstream accepts it."""


class MissingCoordinatesError(ValueError):
    """The symmetry-closed descriptor needs a torsion that is not in the stored table, and no
    coordinates were supplied to recompute it from. Never approximated by a constant shift."""


# ------------------------------------------------------------------------------- enumeration

@dataclass(frozen=True)
class SymmetryEnumeration:
    """Every stereochemistry-preserving atom permutation of one molecule. Identity FIRST.

    `permutations[k][i]` is the atom that atom `i` is mapped to. A configuration `x` RELABELLED by
    permutation `p` is the configuration whose atom `i` sits where atom `p[i]` sat in `x` -- the
    same geometry under another, chemically equivalent naming. Nothing in this object moves an
    atom.
    """

    permutations: Tuple[Tuple[int, ...], ...]
    n_atoms: int
    use_chirality: bool
    cap: int
    rejected_by_stereo_check: int

    @property
    def order(self) -> int:
        return len(self.permutations)

    def describe(self) -> Dict[str, Any]:
        return {"n_automorphisms": self.order, "n_atoms": self.n_atoms,
                "use_chirality": self.use_chirality, "cap": self.cap, "complete": True,
                "rejected_by_stereo_check": self.rejected_by_stereo_check,
                "method": "RDKit self-substructure match, uniquify=False, whole molecule "
                          "(not restricted to rings); every match re-verified as a bond- and "
                          "element-preserving bijection"}


def _graph_invariants(molecule):
    atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in molecule.GetAtoms()]
    bonds = {}
    for b in molecule.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds[(min(i, j), max(i, j))] = str(b.GetBondType())
    return atoms, bonds


def _is_automorphism(perm: Sequence[int], atoms, bonds) -> bool:
    n = len(atoms)
    if sorted(perm) != list(range(n)):
        return False
    if any(atoms[i] != atoms[perm[i]] for i in range(n)):
        return False
    for (i, j), kind in bonds.items():
        a, b = perm[i], perm[j]
        if bonds.get((min(a, b), max(a, b))) != kind:
            return False
    return True


def _stereo_labels(molecule):
    """CIP labels of every atom and bond that carries one, or None when nothing is specified.

    A graph automorphism preserves SPECIFIED stereochemistry exactly when it maps every labelled
    centre and double bond onto one with the same label: the label is a function of the graph and
    the local configuration, and an automorphism preserves the graph. That is the post-check
    applied on top of RDKit's own `useChirality` matching, so that a stereo-breaking permutation is
    refused even where the matcher's handling of a stereo element is incomplete.
    """
    from rdkit import Chem

    copy = Chem.Mol(molecule)
    has_atom = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in copy.GetAtoms())
    has_bond = any(b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                   for b in copy.GetBonds())
    if not (has_atom or has_bond):
        return None
    try:
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(copy)
    except Exception:                                            # pragma: no cover - old rdkit
        Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    atom = [a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "" for a in copy.GetAtoms()]
    bond = {}
    for b in copy.GetBonds():
        if b.HasProp("_CIPCode"):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond[(min(i, j), max(i, j))] = b.GetProp("_CIPCode")
    return atom, bond


def _stereo_preserved(perm, labels) -> bool:
    if labels is None:
        return True
    atom, bond = labels
    if any(atom[i] != atom[perm[i]] for i in range(len(atom))):
        return False
    for (i, j), code in bond.items():
        a, b = perm[i], perm[j]
        if bond.get((min(a, b), max(a, b))) != code:
            return False
    return True


def enumerate_symmetry(molecule, *, max_matches: int = MAX_AUTOMORPHISMS,
                       use_chirality: bool = True) -> SymmetryEnumeration:
    """Every whole-molecule atom permutation that preserves the chemical graph. Refuses if incomplete.

    RDKit matches the molecule against ITSELF with `uniquify=False`, which enumerates every
    isomorphism of the graph onto itself -- every automorphism, of the whole molecule, ring or not.
    On a COPY whose atom-map numbers are cleared, because a map number is bookkeeping a caller
    attached and the matcher would otherwise treat two identical atoms as different. Element,
    formal charge, isotope and bond order are re-verified on every returned match, and with
    `use_chirality` every match must also preserve each specified stereocentre's and stereo double
    bond's CIP label.

    TRUNCATION IS DETECTED, NEVER ACCEPTED. The matcher is asked for `max_matches + 1`; if it
    returns more than `max_matches`, the search was cut short and this raises
    `IncompleteSymmetryEnumeration`. A molecule with many equivalent hydrogens has a factorially
    large group (three methyls alone contribute 6^3 = 216), so the cap is reachable.
    """
    if int(max_matches) < 1:
        raise ValueError("max_matches must be at least 1")
    clean = _clean_for_automorphism(molecule)
    raw = clean.GetSubstructMatches(clean, uniquify=False, useChirality=bool(use_chirality),
                                    maxMatches=int(max_matches) + 1, useQueryQueryMatches=True)
    if len(raw) > int(max_matches):
        raise IncompleteSymmetryEnumeration(
            f"the automorphism search found more than max_matches={int(max_matches)} atom "
            f"permutations and stopped, so the enumeration is INCOMPLETE. A symmetry-aware "
            f"distance built from part of the group is not a quotient metric -- it would treat "
            f"some relabellings of one configuration as different configurations. Raise "
            f"max_matches, or (when the selected torsions involve no hydrogens) pass the "
            f"heavy-atom molecule, whose group is smaller because equivalent hydrogens no longer "
            f"multiply it.")
    atoms, bonds = _graph_invariants(clean)
    labels = _stereo_labels(molecule) if use_chirality else None
    kept, rejected = set(), 0
    for match in raw:
        perm = tuple(int(i) for i in match)
        if not _is_automorphism(perm, atoms, bonds):
            raise RuntimeError(f"RDKit returned a self-match that is not an automorphism: {perm}")
        if not _stereo_preserved(perm, labels):
            rejected += 1
            continue
        kept.add(perm)
    identity = tuple(range(clean.GetNumAtoms()))
    if identity not in kept:
        raise RuntimeError("the identity is missing from the automorphism enumeration")
    ordered = (identity,) + tuple(sorted(p for p in kept if p != identity))
    return SymmetryEnumeration(permutations=ordered, n_atoms=clean.GetNumAtoms(),
                               use_chirality=bool(use_chirality), cap=int(max_matches),
                               rejected_by_stereo_check=int(rejected))


# --------------------------------------------------------------------------- the descriptor

def _canonical(quad: Sequence[int]) -> Tuple[int, int, int, int]:
    """A dihedral is reversal-invariant: A-B-C-D and D-C-B-A are the same observable, exactly."""
    q = tuple(int(i) for i in quad)
    r = tuple(reversed(q))
    return q if q <= r else r


@dataclass(frozen=True)
class DescriptorOperation:
    """One distinct action of the symmetry group on the CLOSED descriptor.

    `sigma[c]` is the column of the ORIGINAL frame that becomes column `c` of the relabelled frame:
    `theta(g x)[:, c] == theta(x)[:, sigma[c]]`. Many atom permutations can share one `sigma` --
    paracetamol's six methyl-hydrogen permutations touch no descriptor torsion -- and they are one
    operation here, represented by the first of them in enumeration order.
    """

    index: int
    sigma: Tuple[int, ...]
    atom_permutation: Tuple[int, ...]
    n_atom_permutations: int
    is_identity: bool

    def moved_atoms(self) -> List[int]:
        return [i for i, j in enumerate(self.atom_permutation) if i != j]

    def describe(self) -> Dict[str, Any]:
        return {"index": self.index, "is_identity": self.is_identity,
                "sigma": list(self.sigma),
                "representative_atom_permutation": list(self.atom_permutation),
                "moved_atoms": self.moved_atoms(),
                "n_atom_permutations_collapsed": self.n_atom_permutations}


class SymmetricTorsionDescriptor:
    """The selected torsions CLOSED under the molecule's symmetry group, with orbit-normalised weights.

    ```python
    descriptor = SymmetricTorsionDescriptor(mol, torsion_definitions)
    theta = descriptor.closed_angles(stored, coordinates=traj, atom_mapping=mapping)
    D, op = descriptor.distance_matrix(descriptor.embed(theta), return_operation=True)
    ```

    COLUMNS. The selected torsions come first, in their given order and orientation. Each image of
    a selected quadruplet under some operation that is not already a column is APPENDED, named
    `<source>@<atoms>`, and must be recomputed from coordinates. `added_columns` lists them.

    WEIGHTS. Each orbit (a set of columns the group permutes among themselves) carries the total
    metric weight of the selected torsions in it, split EQUALLY among its columns. Paracetamol's
    `aryl` (weight 1) has the two-member orbit {(1,3,4,5), (1,3,4,10)}, so each carries 1/2: the
    ring orientation counts once in the distance, not twice. Selected torsions in one orbit must
    have equal weights; otherwise the selection itself breaks the symmetry it is asked to respect,
    and the constructor refuses.

    CHECKS. The column action is verified to be a GROUP (closed under composition, identity
    present) and an ISOMETRY (weights constant on orbits). Those two facts are what make the
    one-sided minimum equal to the two-sided one; `exhaustive_distance` is kept to demonstrate it.
    """

    def __init__(self, molecule, torsions: TorsionDefinitions, *, metric_weights=None,
                 enumeration: Optional[SymmetryEnumeration] = None,
                 max_matches: int = MAX_AUTOMORPHISMS, use_chirality: bool = True):
        if molecule is None:
            raise ValueError("a reference molecule is required; symmetry is a property of it")
        if not isinstance(torsions, TorsionDefinitions):
            raise TypeError("torsions must be a TorsionDefinitions (load_torsion_json, or "
                            "TorsionDefinitions(quads=..., names=..., indexing=..., units=...))")
        n_atoms = molecule.GetNumAtoms()
        outside = [(nm, q) for nm, q in zip(torsions.names, torsions.quads)
                   if any(int(i) >= n_atoms for i in q)]
        if outside:
            raise ValueError(f"torsions index atoms outside the molecule ({n_atoms} atoms): "
                             f"{outside}")
        _require_bonded_chain(molecule, torsions)
        canon_selected = [_canonical(q) for q in torsions.quads]
        if len(set(canon_selected)) != len(canon_selected):
            raise ValueError("two selected torsions are the same quadruplet (possibly reversed)")

        weights = (np.ones(torsions.n_torsions) if metric_weights is None
                   else np.asarray(metric_weights, dtype=np.float64).ravel())
        if weights.size != torsions.n_torsions or not np.all(np.isfinite(weights)) \
                or np.any(weights <= 0):
            raise ValueError("metric_weights must be finite, positive, one per selected torsion")

        self.molecule = molecule
        self.definitions = torsions
        self.enumeration = enumeration or enumerate_symmetry(
            molecule, max_matches=max_matches, use_chirality=use_chirality)
        if self.enumeration.n_atoms != n_atoms:
            raise ValueError("the enumeration belongs to a different molecule")
        perms = self.enumeration.permutations

        # ---- closure: every image of every selected quadruplet becomes a column
        quads: List[Tuple[int, int, int, int]] = [tuple(int(i) for i in q) for q in torsions.quads]
        names: List[str] = list(torsions.names)
        origin: List[Dict[str, Any]] = [{"kind": "selected", "selected": nm} for nm in names]
        where: Dict[Tuple[int, int, int, int], int] = {c: k for k, c in enumerate(canon_selected)}
        for s, quad in enumerate(list(quads[:torsions.n_torsions])):
            for k, perm in enumerate(perms):
                image = tuple(perm[i] for i in quad)
                if _canonical(image) in where:
                    continue
                where[_canonical(image)] = len(quads)
                quads.append(image)
                names.append(f"{torsions.names[s]}@{'-'.join(str(i) for i in image)}")
                origin.append({"kind": "added", "image_of": torsions.names[s],
                               "under_atom_permutation_index": k})
        self.quads = tuple(quads)
        self.names = tuple(names)
        self.origin = tuple(origin)
        self.n_selected = torsions.n_torsions
        self.n_columns = len(quads)

        # ---- the column action of every atom permutation, deduplicated
        sigmas: Dict[Tuple[int, ...], List[int]] = {}
        for k, perm in enumerate(perms):
            sigma = tuple(where[_canonical(tuple(perm[i] for i in q))] for q in self.quads)
            sigmas.setdefault(sigma, []).append(k)
        identity_sigma = tuple(range(self.n_columns))
        if identity_sigma not in sigmas:
            raise RuntimeError("the identity does not act as the identity on the descriptor")
        ordered = [identity_sigma] + sorted(s for s in sigmas if s != identity_sigma)
        self.operations = tuple(
            DescriptorOperation(index=i, sigma=s, atom_permutation=perms[sigmas[s][0]],
                                n_atom_permutations=len(sigmas[s]), is_identity=(i == 0))
            for i, s in enumerate(ordered))
        self._sigma_index = {op.sigma: op.index for op in self.operations}

        # ---- orbits and orbit-normalised weights
        parent = list(range(self.n_columns))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for op in self.operations:
            for c, s in enumerate(op.sigma):
                ra, rb = find(c), find(s)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        orbit_of = [find(c) for c in range(self.n_columns)]
        orbits: Dict[int, List[int]] = {}
        for c, o in enumerate(orbit_of):
            orbits.setdefault(o, []).append(c)
        self.orbits = tuple(tuple(v) for _, v in sorted(orbits.items()))
        column_weights = np.empty(self.n_columns)
        for members in self.orbits:
            selected = [c for c in members if c < self.n_selected]
            ws = weights[selected]
            if not np.allclose(ws, ws[0], rtol=0, atol=0):
                raise ValueError(
                    f"selected torsions {[self.names[c] for c in selected]} are related by "
                    f"molecular symmetry but carry different metric weights {ws.tolist()}. That "
                    f"weighting is itself asymmetric: no distance built on it can be invariant "
                    f"to relabelling. Give symmetry-equivalent torsions equal weights.")
            column_weights[list(members)] = float(ws.sum()) / len(members)
        self.selected_weights = weights
        self.column_weights = column_weights
        self.total_weight = float(column_weights.sum())

        self._check_group_and_isometry()
        self._embed_columns = [np.ravel(np.column_stack([2 * np.asarray(op.sigma),
                                                         2 * np.asarray(op.sigma) + 1]))
                               for op in self.operations]

    # ------------------------------------------------------------------------------ checks

    def _check_group_and_isometry(self) -> None:
        """The two conditions under which the one-sided minimum is exact. Raises if either fails."""
        sigmas = set(self._sigma_index)
        for a in self.operations:
            for b in self.operations:
                composed = tuple(a.sigma[b.sigma[c]] for c in range(self.n_columns))
                if composed not in sigmas:
                    raise RuntimeError(
                        f"the descriptor action is not closed under composition (operations "
                        f"{a.index} and {b.index}); the enumeration cannot be a whole group")
            if not np.array_equal(self.column_weights[list(a.sigma)], self.column_weights):
                raise RuntimeError(f"operation {a.index} does not preserve the column weights; "
                                   f"its action is not an isometry")
        self.isometry_verified = True
        self.group_closure_verified = True

    @property
    def added_columns(self) -> Tuple[int, ...]:
        return tuple(range(self.n_selected, self.n_columns))

    @property
    def is_trivial(self) -> bool:
        """True when the group acts as the identity on the descriptor: d_sym is then Euclidean."""
        return len(self.operations) == 1

    def inverse(self, op_index: int) -> int:
        sigma = self.operations[op_index].sigma
        inv = [0] * self.n_columns
        for c, s in enumerate(sigma):
            inv[s] = c
        return self._sigma_index[tuple(inv)]

    def compose(self, a: int, b: int) -> int:
        """The operation equal to applying `a` and then `b` to a frame: theta(b(a x))."""
        sa, sb = self.operations[a].sigma, self.operations[b].sigma
        return self._sigma_index[tuple(sa[sb[c]] for c in range(self.n_columns))]

    def describe(self) -> Dict[str, Any]:
        return {
            "selected": [{"name": n, "atoms": list(q)}
                         for n, q in zip(self.definitions.names, self.definitions.quads)],
            "columns": [{"name": n, "atoms": list(q), "weight": float(w), "origin": o}
                        for n, q, w, o in zip(self.names, self.quads, self.column_weights,
                                              self.origin)],
            "added_columns": [self.names[c] for c in self.added_columns],
            "orbits": [[self.names[c] for c in orb] for orb in self.orbits],
            "weight_normalisation": "each orbit carries the summed weight of the selected "
                                    "torsions in it, split equally among its columns",
            "total_weight": self.total_weight,
            "operations": [op.describe() for op in self.operations],
            "enumeration": self.enumeration.describe(),
            "group_closure_verified": bool(self.group_closure_verified),
            "isometry_verified": bool(self.isometry_verified),
            "one_sided_reduction": "exact: the action permutes columns within equal-weight "
                                   "orbits, so min over (g,h) equals min over g of "
                                   "||Phi(x) - Phi(g y)||",
        }

    # ------------------------------------------------------------------------ the angles

    def closed_angles(self, stored=None, *, coordinates=None, atom_mapping=None,
                      box_vectors=None, periodic: Optional[bool] = None, topology=None,
                      units: str = "radians",
                      tolerance: float = STORED_AGREEMENT_TOLERANCE) -> np.ndarray:
        """`(n_frames, n_columns)` radians: stored columns reused, every other column recomputed.

        `stored` is the selected-torsion table `(n_frames, n_selected)`, or None to compute every
        column from coordinates. Columns absent from it -- the added images -- are recomputed from
        `coordinates`, which may be an `mdtraj.Trajectory` (its unit cell and topology are used)
        or an `(n_frames, n_atoms, 3)` array with `box_vectors` (or `periodic=False`) and a
        `topology` for validating the mapping.

        `atom_mapping[i]` is the trajectory index of reference atom `i`. It is REQUIRED with
        coordinates, never defaulted to the identity, and checked twice: by graph (elements and
        every reference bond present among the mapped atoms) and by geometry (each stored column
        must agree with the same quadruplet recomputed under the mapping to `tolerance`).

        Molecules split across a periodic boundary are made whole along their bonds BEFORE any
        dihedral is computed; a dihedral of a broken molecule is a plausible-looking number that
        describes no geometry.
        """
        stored_rad = None
        if stored is not None:
            arr = np.asarray(stored, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != self.n_selected:
                raise ValueError(f"stored torsions must be (n_frames, {self.n_selected}) in the "
                                 f"order {list(self.definitions.names)}; got {arr.shape}")
            if units not in ("radians", "degrees"):
                raise ValueError("units must be 'radians' or 'degrees'")
            stored_rad = np.deg2rad(arr) if units == "degrees" else arr
            if not np.all(np.isfinite(stored_rad)):
                raise ValueError("stored torsions hold non-finite values")
            stored_rad = wrap(stored_rad)

        need_coordinates = bool(self.added_columns) or stored_rad is None
        if not need_coordinates:
            if coordinates is not None:
                xyz = self._mapped_whole(coordinates, atom_mapping, box_vectors, periodic,
                                         topology)
                self._check_stored(stored_rad, xyz, tolerance)
            return stored_rad
        if coordinates is None:
            missing = [f"{self.names[c]} {list(self.quads[c])}" for c in self.added_columns]
            raise MissingCoordinatesError(
                "the symmetry-closed descriptor needs torsions that are not in the stored table: "
                + "; ".join(missing) + ". They are the images of the selected torsions under "
                "the molecule's symmetry operations, and are recomputed from coordinates -- "
                "never approximated as a fixed 180-degree shift, because ring distortion and "
                "pyramidalisation move the outer atoms independently. Pass coordinates=<mdtraj "
                "trajectory or (n_frames, n_atoms, 3) array> and atom_mapping=<trajectory index "
                "of each reference atom>, or add those quadruplets to the torsion table.")
        xyz = self._mapped_whole(coordinates, atom_mapping, box_vectors, periodic, topology)
        if stored_rad is not None and xyz.shape[0] != stored_rad.shape[0]:
            raise ValueError(f"{xyz.shape[0]} coordinate frames for {stored_rad.shape[0]} stored "
                             f"torsion rows; they must be the same frames in the same order")
        out = np.empty((xyz.shape[0], self.n_columns))
        for c in range(self.n_columns):
            if c < self.n_selected and stored_rad is not None:
                out[:, c] = stored_rad[:, c]
            else:
                q = self.quads[c]
                out[:, c] = dihedral(xyz[:, q[0]], xyz[:, q[1]], xyz[:, q[2]], xyz[:, q[3]])
        if stored_rad is not None:
            self._check_stored(stored_rad, xyz, tolerance)
        return wrap(out)

    def _check_stored(self, stored_rad, xyz, tolerance) -> None:
        worst, column = 0.0, None
        for c in range(self.n_selected):
            q = self.quads[c]
            recomputed = dihedral(xyz[:, q[0]], xyz[:, q[1]], xyz[:, q[2]], xyz[:, q[3]])
            diff = float(np.max(np.abs(wrap(recomputed - stored_rad[:, c]))))
            if diff > worst:
                worst, column = diff, c
        self.stored_agreement_ = worst
        if worst > float(tolerance):
            raise ValueError(
                f"the stored torsion table disagrees with the coordinates under this atom mapping: "
                f"column {self.names[column]} differs by up to {worst:.4g} rad (tolerance "
                f"{tolerance}). Either the frames are not the same frames in the same order, or "
                f"the mapping names different atoms than the table was computed from.")

    def _mapped_whole(self, coordinates, atom_mapping, box_vectors, periodic, topology):
        xyz, box, top = _unpack_coordinates(coordinates, box_vectors, periodic, topology)
        n_ref = self.molecule.GetNumAtoms()
        if atom_mapping is None:
            raise ValueError(
                f"coordinates given with no atom_mapping. Reference (SDF) atom order and MD "
                f"topology order are not the same thing, and equal element counts do not show "
                f"that they are. Pass atom_mapping[i] = trajectory index of reference atom i "
                f"({n_ref} entries); `verify_atom_mapping` finds and checks the identity when it "
                f"holds.")
        mapping = np.asarray(atom_mapping, dtype=np.int64).ravel()
        if top is None:
            raise ValueError("a topology is needed to validate atom_mapping by graph; pass an "
                             "mdtraj.Trajectory as coordinates, or topology=<mdtraj.Topology>")
        check_atom_mapping(self.molecule, mapping, top, n_trajectory_atoms=xyz.shape[1])
        sub = xyz[:, mapping, :]
        if box is not None:
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in self.molecule.GetBonds()]
            sub = whole_molecule_coordinates(sub, bonds, box)
        return sub

    # ------------------------------------------------------------------------ the metric

    def embed(self, theta: np.ndarray) -> np.ndarray:
        """`Phi(theta)`: interleaved `sqrt(w_c) * (cos, sin)` per CLOSED column."""
        th = np.asarray(theta, dtype=np.float64)
        if th.ndim != 2 or th.shape[1] != self.n_columns:
            raise ValueError(f"expected (n, {self.n_columns}) closed angles; got {th.shape}")
        s = np.sqrt(self.column_weights)[None, :]
        X = np.empty((th.shape[0], 2 * self.n_columns))
        X[:, 0::2] = np.cos(th) * s
        X[:, 1::2] = np.sin(th) * s
        return X

    def transform(self, theta: np.ndarray, op_index: int) -> np.ndarray:
        """The closed angles of every frame RELABELLED by operation `op_index`. Moves no atom."""
        th = np.asarray(theta, dtype=np.float64)
        return th[:, list(self.operations[int(op_index)].sigma)]

    def _tie(self) -> float:
        return TIE_RELATIVE_TOLERANCE * 4.0 * self.total_weight

    def distance_matrix(self, XA: np.ndarray, XB: Optional[np.ndarray] = None, *,
                        return_operation: bool = False, block_bytes: int = 256 * 2 ** 20):
        """`D[i, j] = min_g ||XA[i] - Phi(g y_j)||`, with the minimising operation if asked.

        Exact, by the one-sided reduction the constructor verified. Every embedded point has
        squared norm `W` (the total weight) and every operation preserves it, so
        `||a - b||^2 = 2W - 2 a.b` and one matrix product per operation gives a whole block.

        TIES go to the EARLIER operation (identity first, then the fixed enumeration order): the
        minimum is the exact minimum, and the reported operation is the first whose squared
        distance is within `TIE_RELATIVE_TOLERANCE * 4W` of it.
        """
        same = XB is None
        XB = XA if same else XB
        nA, nB, nops = XA.shape[0], XB.shape[0], len(self.operations)
        W2 = 2.0 * self.total_weight
        rows = max(1, int(block_bytes // max(1, 8 * nB * (nops + 1))))
        D = np.empty((nA, nB))
        arg = np.zeros((nA, nB), dtype=np.int16) if return_operation else None
        tie = self._tie()
        transformed = [XB[:, cols] for cols in self._embed_columns]
        for lo in range(0, nA, rows):
            hi = min(nA, lo + rows)
            if not return_operation:
                # The smallest distance is the LARGEST dot product: a running maximum needs one
                # buffer, not one per operation.
                best = D[lo:hi]
                np.matmul(XA[lo:hi], transformed[0].T, out=best)
                buf = np.empty_like(best)
                for k in range(1, nops):
                    np.matmul(XA[lo:hi], transformed[k].T, out=buf)
                    np.maximum(best, buf, out=best)
                best *= -2.0
                best += W2
                np.maximum(best, 0.0, out=best)
                continue
            stack = np.empty((nops, hi - lo, nB))
            for k in range(nops):
                np.matmul(XA[lo:hi], transformed[k].T, out=stack[k])
                stack[k] *= -2.0
                stack[k] += W2
            np.maximum(stack, 0.0, out=stack)
            best = stack.min(axis=0)
            D[lo:hi] = best
            arg[lo:hi] = np.argmax(stack <= best[None] + tie, axis=0)
        np.sqrt(D, out=D)
        if same:
            # Exact symmetry and a zero diagonal. Mathematically D == D.T already (the group is
            # closed under inverses); numerically the two triangles can differ in the last bit,
            # and a precomputed-metric HDBSCAN must not see a distance that depends on argument
            # order.
            np.minimum(D, D.T, out=D)
            np.fill_diagonal(D, 0.0)
        return (D, arg) if return_operation else D

    def pair_table(self, theta_x: np.ndarray, theta_y: np.ndarray) -> List[Dict[str, Any]]:
        """Every operation applied to `y`, with the transformed angles and the distance to `x`.

        Direct differences rather than the norm identity, so a value of zero is zero.
        """
        tx = np.asarray(theta_x, dtype=np.float64).reshape(1, -1)
        ty = np.asarray(theta_y, dtype=np.float64).reshape(1, -1)
        ex = self.embed(tx)[0]
        rows = []
        for op in self.operations:
            gy = self.transform(ty, op.index)
            rows.append({"operation": op.index, "is_identity": op.is_identity,
                         "theta_gy": gy[0], "distance": float(np.linalg.norm(ex - self.embed(gy)[0]))})
        return rows

    def distance(self, theta_x: np.ndarray, theta_y: np.ndarray) -> Tuple[float, int]:
        """`(d_sym, minimising operation)` for one pair. Earlier operation on a tie."""
        rows = self.pair_table(theta_x, theta_y)
        values = np.array([r["distance"] ** 2 for r in rows])
        best = float(values.min())
        k = int(np.argmax(values <= best + self._tie()))
        return float(np.sqrt(best)), k

    def exhaustive_distance(self, theta_x: np.ndarray, theta_y: np.ndarray
                            ) -> Tuple[float, Tuple[int, int]]:
        """Two-sided `min_{g,h} ||Phi(g x) - Phi(h y)||` and the first minimising `(g, h)`.

        Kept as the reference the one-sided calculation is tested against, never used to fit.
        """
        tx = np.asarray(theta_x, dtype=np.float64).reshape(1, -1)
        ty = np.asarray(theta_y, dtype=np.float64).reshape(1, -1)
        best, pair = np.inf, (0, 0)
        values = {}
        for g in self.operations:
            egx = self.embed(self.transform(tx, g.index))[0]
            for h in self.operations:
                ehy = self.embed(self.transform(ty, h.index))[0]
                values[(g.index, h.index)] = float(np.sum((egx - ehy) ** 2))
        best = min(values.values())
        tie = self._tie()
        pair = min(k for k, v in values.items() if v <= best + tie)
        return float(np.sqrt(best)), pair


# --------------------------------------------------------------------------- coordinates

def _unpack_coordinates(coordinates, box_vectors, periodic, topology):
    """`(xyz, box_or_None, topology_or_None)` from an mdtraj Trajectory or arrays."""
    if hasattr(coordinates, "xyz") and hasattr(coordinates, "topology"):
        xyz = np.asarray(coordinates.xyz, dtype=np.float64)
        top = coordinates.topology if topology is None else topology
        cell = getattr(coordinates, "unitcell_vectors", None)
        if box_vectors is not None:
            box = np.asarray(box_vectors, dtype=np.float64)
        elif periodic is False:
            box = None
        elif cell is not None:
            box = np.asarray(cell, dtype=np.float64)
        else:
            if periodic:
                raise ValueError("periodic=True but the trajectory carries no unit cell")
            box = None
        return xyz, box, top
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"coordinates must be (n_frames, n_atoms, 3); got {xyz.shape}")
    if box_vectors is None:
        if periodic is not False:
            raise ValueError(
                "coordinates were given as an array with no box_vectors. Say which they are: "
                "box_vectors=(n_frames, 3, 3) or (3, 3) for a periodic trajectory, which is made "
                "whole before any dihedral is computed, or periodic=False for one that is not "
                "periodic (implicit solvent, gas phase).")
        return xyz, None, topology
    return xyz, np.asarray(box_vectors, dtype=np.float64), topology


def check_atom_mapping(molecule, mapping, topology, *, n_trajectory_atoms: Optional[int] = None
                       ) -> None:
    """Refuse a reference -> trajectory atom mapping that does not preserve the molecular graph.

    The trajectory may hold more atoms than the molecule (solvent, ions, a protein); the check is
    on the INDUCED subgraph: the mapped atoms must have the reference elements, and the bonds among
    them must be exactly the reference bonds. A swap of two carbons that keeps every element count
    is caught, because it moves a bond.
    """
    n = molecule.GetNumAtoms()
    m = np.asarray(mapping, dtype=np.int64).ravel()
    if m.size != n:
        raise ValueError(f"atom_mapping has {m.size} entries for {n} reference atoms")
    if len(set(m.tolist())) != m.size:
        raise ValueError("atom_mapping is not injective: two reference atoms share a trajectory atom")
    limit = n_trajectory_atoms if n_trajectory_atoms is not None else topology.n_atoms
    if m.min() < 0 or m.max() >= limit:
        raise ValueError(f"atom_mapping indexes outside the trajectory ({limit} atoms)")
    atoms = list(topology.atoms)
    for i, j in enumerate(m):
        ref = molecule.GetAtomWithIdx(i).GetSymbol()
        el = atoms[int(j)].element.symbol
        if ref != el:
            raise ValueError(f"atom_mapping: reference atom {i} is {ref} but trajectory atom "
                             f"{int(j)} is {el}")
    inverse = {int(j): i for i, j in enumerate(m)}
    mapped_bonds = set()
    for a, b in topology.bonds:
        if a.index in inverse and b.index in inverse:
            i, j = inverse[a.index], inverse[b.index]
            mapped_bonds.add((min(i, j), max(i, j)))
    reference = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                  max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in molecule.GetBonds()}
    if mapped_bonds != reference:
        raise ValueError(
            f"atom_mapping does not preserve the bond graph: reference bonds absent among the "
            f"mapped atoms {sorted(reference - mapped_bonds)[:4]}, extra bonds "
            f"{sorted(mapped_bonds - reference)[:4]}")


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Shortest periodic image of difference vectors, for orthorhombic AND reduced triclinic boxes.

    `delta` is `(n_frames, 3)` and `box` `(n_frames, 3, 3)` with rows a, b, c in OpenMM's reduced
    form (a along x, b in the xy plane). Subtracting c, then b, then a in that order is exact for
    such a box whenever the true separation is under half the shortest box height -- which every
    bonded pair is.
    """
    d = np.array(delta, dtype=np.float64, copy=True)
    for axis in (2, 1, 0):
        vec = box[:, axis, :]
        n = np.round(d[:, axis] / vec[:, axis])
        d -= n[:, None] * vec
    return d


def whole_molecule_coordinates(xyz: np.ndarray, bonds, box) -> np.ndarray:
    """Make a molecule whole along its bonds, frame by frame, before any dihedral is computed.

    Walks the bond graph from atom 0 and places each atom at the periodic image nearest its
    already-placed neighbour. Refuses a molecule whose bond graph is not connected, because the
    unreached atoms would keep whatever image the box left them in.
    """
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 2:
        box = np.broadcast_to(box, (xyz.shape[0], 3, 3)).copy()
    if box.shape != (xyz.shape[0], 3, 3):
        raise ValueError(f"box vectors must be (3, 3) or ({xyz.shape[0]}, 3, 3); got {box.shape}")
    neighbours: Dict[int, List[int]] = {}
    for i, j in bonds:
        neighbours.setdefault(int(i), []).append(int(j))
        neighbours.setdefault(int(j), []).append(int(i))
    seen, stack = {0}, [0]
    while stack:
        current = stack.pop()
        for other in sorted(neighbours.get(current, ())):
            if other in seen:
                continue
            xyz[:, other, :] = xyz[:, current, :] + minimum_image(
                xyz[:, other, :] - xyz[:, current, :], box)
            seen.add(other)
            stack.append(other)
    if len(seen) != xyz.shape[1]:
        raise ValueError(f"the bond graph is not connected ({len(seen)} of {xyz.shape[1]} atoms "
                         f"reachable from atom 0); a molecule cannot be made whole")
    return xyz


# ------------------------------------------------------------------------------- drawing

def draw_symmetry_operation(descriptor: SymmetricTorsionDescriptor, op_index: int, *, output,
                            size: Tuple[int, int] = (460, 380)) -> Dict[str, Any]:
    """A PNG of the molecule with the atoms one descriptor operation EXCHANGES coloured in pairs.

    Atom indices are printed, so each coloured pair can be read off as `i <-> p[i]`. The picture
    is of a RELABELLING: the coloured atoms swap names, none of them moves. The operation's
    representative atom permutation is drawn; the hydrogen permutations that collapse onto the
    same operation are listed in the returned record, not drawn.
    """
    from ._t_symmetry import _molecule_png, permutation_colours

    op = descriptor.operations[int(op_index)]
    colours = permutation_colours(op.atom_permutation)
    png = _molecule_png(descriptor.molecule, sorted(colours), size=size, colours=colours)
    with open(output, "wb") as handle:
        handle.write(png)
    swaps = sorted({tuple(sorted((i, j))) for i, j in enumerate(op.atom_permutation) if i != j})
    return {"output": str(output), "operation": op.index, "exchanged_pairs": swaps,
            "n_atom_permutations_collapsed": op.n_atom_permutations}
