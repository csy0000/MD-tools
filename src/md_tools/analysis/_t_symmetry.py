"""Molecular symmetry over torsional clusters: which clusters are the SAME conformer relabelled.

THE QUESTION. A density partition of a torsional ensemble (`md_tools.analysis.t_hdbscan`) returns
clusters of configurations. Some of those may be the same physical conformer seen through a
permutation of chemically identical atoms -- paracetamol's phenyl ring has a two-fold axis, so
flipping it maps one basin onto another with no physical change at all. Merging those is right.
Merging two genuinely different conformers because their populations happen to match is wrong, and
is the error everything below is arranged to prevent.

WHAT COUNTS AS EVIDENCE, and what does not:

* A graph AUTOMORPHISM of the molecule, found by RDKit, respecting element, bond order, formal
  charge, isotope and specified stereochemistry. That is a candidate operation.
* The operation's action on EVERY selected torsion at once, and then agreement of the two
  clusters' JOINT distributions in both directions.

Not evidence: equal populations, similar one-dimensional histograms, equal canonical ranks, a
force-field multiplicity, a bin count, or an unconstrained spatial reflection. Each of those is
either a consequence of symmetry rather than a demonstration of it, or a different claim
altogether. Canonical ranks are used BELOW only to find candidates quickly; nothing is accepted
on them.

THE COUPLED CASE IS WHY PER-TORSION FOLDING IS REFUSED. In a para-disubstituted benzene the
two-fold ring flip moves BOTH attachment torsions together. Folding each torsion into its own
period independently would also identify the configuration where only one substituent was rotated
-- a different relative orientation, and a different conformer. So a permutation is applied as one
object to the whole descriptor vector, never torsion by torsion.

WHAT THIS MODULE DOES NOT ESTABLISH. That two clusters are related by a chemical symmetry says
their conformations are structurally equivalent. It does NOT establish that their thermodynamic
populations should be equal: an atom-specific restraint, a REST2 scaling region that covers one
side and not the other, or any atom-indexed parameter breaks the equivalence while leaving the
graph automorphism intact. `fit` takes `broken_by` for exactly that declaration and reports it.

NEEDS THE `analysis` EXTRA (rdkit, scikit-learn). Both are imported lazily.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi

#: The angular units a definition file may declare. There is no default: a file that does not say
#: is refused, because radians and degrees differ by a factor that looks like a wide distribution
#: rather than like an error.
UNITS = {"radian": 1.0, "radians": 1.0, "rad": 1.0,
         "degree": math.pi / 180.0, "degrees": math.pi / 180.0, "deg": math.pi / 180.0}

#: How many automorphisms to enumerate before declaring the search truncated. A molecule with many
#: equivalent hydrogens has a factorially large automorphism group, and silently taking the first N
#: would make "no symmetry found" indistinguishable from "stopped looking".
MAX_AUTOMORPHISMS = 200_000


def _as_radians(values, unit: str) -> np.ndarray:
    try:
        scale = UNITS[str(unit).lower()]
    except KeyError:
        raise ValueError(
            f"unknown angle unit {unit!r}; one of {sorted(set(UNITS))}. There is no default: "
            f"degrees read as radians is a 57x error that looks like a broad distribution.")
    out = np.asarray(values, dtype=np.float64) * scale
    if not np.all(np.isfinite(out)):
        raise ValueError("angles must all be finite; found nan or inf")
    return out


def wrap(a) -> np.ndarray:
    """Wrap to ``[-pi, pi)``. The same convention as `md_tools.analysis._torsions`."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def circular_difference(a, b) -> np.ndarray:
    """Signed shortest difference `a - b` on the circle, in ``[-pi, pi)``."""
    return wrap(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


# ----------------------------------------------------------------------------- torsion definitions

@dataclass(frozen=True)
class TorsionDefinitions:
    """The selected torsions: quadruplets of atom indices, their names, and where they came from.

    `quads` are indices into the REFERENCE MOLECULE (the SDF), zero-based, after any one-based
    conversion the file declared. `indexing` records what the file said so a result can be read
    back against it.
    """

    quads: Tuple[Tuple[int, int, int, int], ...]
    names: Tuple[str, ...]
    indexing: str
    units: str
    source: Optional[str] = None
    sign_convention: Optional[str] = None

    def __post_init__(self):
        if len(self.quads) != len(self.names):
            raise ValueError(f"{len(self.quads)} quadruplets but {len(self.names)} names")
        if not self.quads:
            raise ValueError("no torsions defined")
        for name, quad in zip(self.names, self.quads):
            if len(quad) != 4:
                raise ValueError(f"torsion {name!r} has {len(quad)} atoms; a proper torsion has 4")
            if len(set(quad)) != 4:
                raise ValueError(f"torsion {name!r} repeats an atom: {quad}")
            if any(int(i) < 0 for i in quad):
                raise ValueError(f"torsion {name!r} has a negative index: {quad}")
        if len(set(self.names)) != len(self.names):
            raise ValueError("torsion names must be unique; they are how a column is addressed")

    @property
    def n_torsions(self) -> int:
        return len(self.quads)

    def central_bonds(self) -> Tuple[Tuple[int, int], ...]:
        """The B-C bond of each A-B-C-D, unordered. A torsion is named by its central bond."""
        return tuple(tuple(sorted(q[1:3])) for q in self.quads)

    def index_of(self, quad: Sequence[int]) -> Optional[int]:
        """The column holding this quadruplet, or None.

        A DIHEDRAL IS REVERSAL-INVARIANT: the angle A-B-C-D equals the angle D-C-B-A, exactly,
        for every geometry. So a transformed quadruplet that appears reversed in the table is the
        SAME observable and its stored column is reused. Nothing else is treated as equivalent --
        in particular a quadruplet sharing the central bond but with a different outer atom is a
        different observable, and assuming it differs by a constant is the error `_transform`
        refuses to make.
        """
        want, rev = tuple(int(i) for i in quad), tuple(int(i) for i in reversed(quad))
        for column, stored in enumerate(self.quads):
            if stored == want or stored == rev:
                return column
        return None


def load_torsion_json(path, *, molecule=None) -> TorsionDefinitions:
    """Read torsion definitions from JSON, handling the indexing convention EXPLICITLY.

    Accepted shapes, in order of preference:

    ```json
    {"indexing": "zero-based", "units": "radian",
     "torsions": [{"name": "omega", "atoms": [0, 1, 3, 4]}, ...]}
    ```

    ```json
    {"indexing": "one-based", "units": "degree",
     "columns": ["omega", ...], "quadruplets": [[1, 2, 4, 5], ...]}
    ```

    A BARE LIST of quadruplets is also read, and then `indexing` and `units` must be supplied by
    the file's sibling metadata or the call is refused -- guessing either is how a file silently
    describes different atoms than its author meant.

    NOT the md-tools `cv.yaml`, and not its `.cv.json` sidecar. Those identify atoms by SELECTOR
    (chain, residue, atom name) rather than by index, on purpose: an index is only meaningful
    against one topology and a rebuild renumbers. Resolving a selector needs that topology, so
    `md_tools.cv.definition` does it and this function does not pretend to. A sidecar's `columns`
    are names with no indices at all, and it is refused by name rather than half-read.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(document, dict) and "columns" in document and "quadruplets" not in document \
            and "torsions" not in document:
        raise ValueError(
            f"{path} looks like an md-tools CV sidecar: it carries `columns` (names) and no atom "
            f"indices. A sidecar records which series were written, not which atoms they measure "
            f"-- the atoms are in the `cv.yaml` it names in `source`, as SELECTORS. Resolve those "
            f"against the topology with md_tools.cv, or write a definition file with explicit "
            f"indices.")

    if isinstance(document, list):
        quads = document
        meta: Dict[str, Any] = {}
    elif "torsions" in document:
        entries = document["torsions"]
        quads = [e["atoms"] if isinstance(e, dict) else e for e in entries]
        names = [e.get("name") for e in entries if isinstance(e, dict)]
        meta = dict(document)
        if len(names) == len(quads) and all(n for n in names):
            meta["columns"] = names
    elif "quadruplets" in document:
        quads = document["quadruplets"]
        meta = dict(document)
    else:
        raise ValueError(
            f"{path} has none of `torsions`, `quadruplets` or a bare list of quadruplets")

    indexing = str(meta.get("indexing", "")).lower().replace("_", "-")
    if indexing in ("0", "zero-based", "zerobased", "zero"):
        offset = 0
    elif indexing in ("1", "one-based", "onebased", "one"):
        offset = 1
    else:
        raise ValueError(
            f"{path} does not declare `indexing`. It must say \"zero-based\" or \"one-based\": "
            f"an off-by-one over atom indices names DIFFERENT atoms and produces a torsion that "
            f"is perfectly well-formed and measures something else. There is no safe default, "
            f"and inferring it from whether 0 appears is a guess that fails on exactly the "
            f"molecules where it matters.")

    units = meta.get("units")
    if units is None:
        raise ValueError(f"{path} does not declare `units` (\"radian\" or \"degree\")")
    _as_radians([0.0], units)        # validate the unit now, not at first use

    quads_0 = tuple(tuple(int(i) - offset for i in q) for q in quads)
    columns = meta.get("columns")
    if columns is None:
        names = tuple(f"t{n}" for n in range(len(quads_0)))
    else:
        if len(columns) != len(quads_0):
            raise ValueError(f"{len(columns)} names for {len(quads_0)} quadruplets")
        names = tuple(str(c) for c in columns)

    definitions = TorsionDefinitions(
        quads=quads_0, names=names, indexing=("one-based" if offset else "zero-based"),
        units=str(units), source=str(path),
        sign_convention=meta.get("sign_convention"))

    if molecule is not None:
        n = molecule.GetNumAtoms()
        bad = [(nm, q) for nm, q in zip(definitions.names, definitions.quads)
               if any(i >= n for i in q)]
        if bad:
            raise ValueError(
                f"{path} indexes atoms outside the molecule ({n} atoms): {bad[:4]}. If the file is "
                f"one-based, say so in `indexing` -- this is what that field prevents.")
        _require_bonded_chain(molecule, definitions)
    return definitions


def _require_bonded_chain(molecule, definitions: TorsionDefinitions) -> None:
    """Every quadruplet must be a BONDED CHAIN A-B-C-D in the reference molecule.

    A proper torsion is three consecutive bonds. Four atoms that are not a chain still produce a
    number from the dihedral formula -- it is just not a torsion of this molecule, and no symmetry
    statement about it means anything.
    """
    broken = []
    for name, quad in zip(definitions.names, definitions.quads):
        for a, b in zip(quad, quad[1:]):
            if molecule.GetBondBetweenAtoms(int(a), int(b)) is None:
                broken.append((name, (int(a), int(b))))
    if broken:
        raise ValueError(
            f"these torsions are not bonded chains in the reference molecule: {broken[:6]}. A "
            f"proper torsion is three consecutive bonds; four unbonded atoms give a number that "
            f"is not a torsion of this molecule.")


# ------------------------------------------------------------------------------ cluster assembly

@dataclass
class ClusterEntry:
    """One cluster's frames. Labels are kept EXPLICIT and are never assumed consecutive."""

    label: int
    frames: np.ndarray          # (n_i,) indices into the original frame axis
    torsions: np.ndarray        # (n_i, n_torsions) radians, wrapped to [-pi, pi)
    weights: np.ndarray         # (n_i,) frame weights, unnormalised
    mass: float                 # weighted fraction of the WHOLE ensemble, noise included

    @property
    def n_frames(self) -> int:
        return int(self.frames.size)


@dataclass
class ClusterTorsions:
    """Existing cluster results, assembled for symmetry comparison.

    One entry per original cluster PLUS noise held separately. `labels` is the per-frame label
    array exactly as given, so nothing downstream has to reconstruct it.
    """

    entries: Dict[int, ClusterEntry]
    noise: ClusterEntry
    labels: np.ndarray
    torsions: np.ndarray
    weights: np.ndarray
    names: Tuple[str, ...]
    noise_label: int
    total_weight: float

    @property
    def cluster_labels(self) -> List[int]:
        """The original labels, sorted. NOT assumed to be 0..k-1 or to start at 1."""
        return sorted(self.entries)

    def mass_table(self) -> Dict[str, Any]:
        return {"clusters": {int(k): float(v.mass) for k, v in sorted(self.entries.items())},
                "noise": float(self.noise.mass),
                "sum": float(sum(v.mass for v in self.entries.values()) + self.noise.mass)}


def load_cluster_torsion(torsions, cluster_idx, *, noise_label: int = -1,
                         angle_unit: str = "radian", frame_weights=None,
                         names: Optional[Sequence[str]] = None) -> ClusterTorsions:
    """Assemble per-cluster torsion arrays from an EXISTING clustering. Nothing is re-fitted.

    `torsions` is `(n_frames, n_torsions)`, `cluster_idx` is `(n_frames,)`. Labels are used as
    given: they need not be consecutive, need not start at 0 or 1, and `noise_label` is kept out
    of the clusters rather than becoming one.

    `mass` is the weighted fraction of the WHOLE ensemble including noise, which is the same
    convention as `md_tools.analysis._torsions.cluster_mass`. That matters for merging: masses
    then sum to the original total and a merged group is the sum of its parts, with nothing
    renormalised and nothing multiplied by a symmetry order.
    """
    th = _as_radians(torsions, angle_unit)
    if th.ndim != 2:
        raise ValueError(f"torsions must be (n_frames, n_torsions); got shape {th.shape}")
    th = wrap(th)
    n_frames, n_torsions = th.shape

    lab = np.asarray(cluster_idx).ravel()
    if lab.size != n_frames:
        raise ValueError(f"{lab.size} labels for {n_frames} frames")
    if not np.issubdtype(lab.dtype, np.integer):
        if not np.all(np.equal(np.mod(lab.astype(np.float64), 1.0), 0.0)):
            raise ValueError("cluster labels must be integers")
        lab = lab.astype(np.int64)
    lab = lab.astype(np.int64)

    if frame_weights is None:
        w = np.ones(n_frames, dtype=np.float64)
    else:
        w = np.asarray(frame_weights, dtype=np.float64).ravel()
        if w.size != n_frames:
            raise ValueError(f"{w.size} frame weights for {n_frames} frames")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("frame weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("frame weights sum to zero")

    if names is None:
        column_names = tuple(f"t{i}" for i in range(n_torsions))
    else:
        if len(names) != n_torsions:
            raise ValueError(f"{len(names)} names for {n_torsions} torsion columns")
        column_names = tuple(str(x) for x in names)

    def _entry(label: int, mask: np.ndarray) -> ClusterEntry:
        frames = np.flatnonzero(mask)
        return ClusterEntry(label=int(label), frames=frames, torsions=th[frames],
                            weights=w[frames], mass=float(w[frames].sum() / total))

    entries = {int(c): _entry(int(c), lab == c)
               for c in np.unique(lab) if int(c) != int(noise_label)}
    if not entries:
        raise ValueError(f"every frame carries the noise label {noise_label}; nothing to compare")
    return ClusterTorsions(entries=entries, noise=_entry(int(noise_label), lab == noise_label),
                           labels=lab, torsions=th, weights=w, names=column_names,
                           noise_label=int(noise_label), total_weight=total)


# -------------------------------------------------------------------------- symmetry operations

@dataclass(frozen=True)
class SymmetryOperation:
    """One atom permutation and what it does to the selected torsions.

    `permutation[i]` is the atom that atom `i` is mapped TO. `column_source[c]` says where the
    transformed value of column `c` comes from: `("column", j)` to reuse stored column `j`,
    `("column_reversed", j)` likewise (a dihedral is reversal-invariant), or `("coordinates",)`
    when the transformed quadruplet is not in the table and must be recomputed.
    """

    index: int
    permutation: Tuple[int, ...]
    transformed_quads: Tuple[Tuple[int, int, int, int], ...]
    column_source: Tuple[Tuple[Any, ...], ...]
    is_identity: bool

    @property
    def needs_coordinates(self) -> bool:
        return any(src[0] == "coordinates" for src in self.column_source)

    @property
    def unresolved_columns(self) -> Tuple[int, ...]:
        return tuple(c for c, src in enumerate(self.column_source) if src[0] == "coordinates")

    def describe(self) -> Dict[str, Any]:
        return {"index": self.index, "is_identity": self.is_identity,
                "permutation": list(self.permutation),
                "moved_atoms": [i for i, j in enumerate(self.permutation) if i != j],
                "transformed_quadruplets": [list(q) for q in self.transformed_quads],
                "column_source": [list(s) for s in self.column_source],
                "needs_coordinates": self.needs_coordinates,
                "unresolved_columns": list(self.unresolved_columns)}


def _clean_for_automorphism(molecule):
    """A copy with atom-map numbers cleared, so they cannot act as chemical distinctions.

    An atom map number is bookkeeping a caller attached; RDKit's substructure matcher will happily
    treat two otherwise identical atoms as different because of it, which would hide a real
    symmetry. Everything that IS chemistry -- element, bond order, formal charge, isotope,
    specified stereochemistry -- is left exactly as it came.
    """
    from rdkit import Chem

    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return copy


def automorphisms(molecule, *, max_matches: int = MAX_AUTOMORPHISMS,
                  use_chirality: bool = True) -> Tuple[List[Tuple[int, ...]], bool]:
    """Every graph automorphism of `molecule`, and whether the enumeration was TRUNCATED.

    Found by matching the molecule against itself. `useChirality=True` keeps SPECIFIED
    stereochemistry: an operation that would swap the two faces of a defined E/Z double bond or
    invert an assigned centre is not an automorphism of the stereochemical graph and is refused
    here rather than filtered later.

    Truncation is RETURNED, never swallowed. A molecule with many equivalent hydrogens has a
    factorially large automorphism group; stopping at the cap and reporting "these are all of
    them" would make an incomplete search indistinguishable from a complete one.
    """
    clean = _clean_for_automorphism(molecule)
    matches = clean.GetSubstructMatches(clean, uniquify=False, useChirality=use_chirality,
                                        maxMatches=max_matches, useQueryQueryMatches=True)
    truncated = len(matches) >= max_matches
    return [tuple(int(i) for i in m) for m in matches], truncated


def _canonical_rank_partition(molecule) -> Dict[int, List[int]]:
    """Atoms grouped by canonical rank, with ties kept. A SCREEN, never evidence.

    Equal rank says two atoms have indistinguishable canonical environments, which is necessary
    for an automorphism to exchange them and nowhere near sufficient. It is computed so a result
    file can show which candidates were considered, and so a caller can see when a permutation
    that an automorphism search found maps across rank classes -- which would be a bug.
    """
    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(_clean_for_automorphism(molecule), breakTies=False))
    out: Dict[int, List[int]] = {}
    for atom, rank in enumerate(ranks):
        out.setdefault(int(rank), []).append(int(atom))
    return out


def _operation_for(permutation: Sequence[int], definitions: TorsionDefinitions,
                   index: int) -> SymmetryOperation:
    """Build the operation record: where each transformed column's value can come from."""
    perm = tuple(int(i) for i in permutation)
    transformed, sources = [], []
    for quad in definitions.quads:
        moved = tuple(perm[i] for i in quad)
        transformed.append(moved)
        column = definitions.index_of(moved)
        if column is None:
            sources.append(("coordinates",))
        elif tuple(definitions.quads[column]) == moved:
            sources.append(("column", column))
        else:
            sources.append(("column_reversed", column))
    return SymmetryOperation(
        index=index, permutation=perm, transformed_quads=tuple(transformed),
        column_source=tuple(sources),
        is_identity=all(i == j for i, j in enumerate(perm)))


def distinct_operations(molecule, definitions: TorsionDefinitions, *,
                        max_matches: int = MAX_AUTOMORPHISMS,
                        use_chirality: bool = True) -> Tuple[List[SymmetryOperation], bool]:
    """Automorphisms, DEDUPLICATED BY THEIR EFFECT ON THE SELECTED TORSIONS.

    Most of a molecule's automorphism group is permutations of equivalent hydrogens that leave
    every selected torsion's quadruplet untouched -- a methyl group alone contributes six. They
    are real automorphisms and they are identical as far as this comparison is concerned, so they
    collapse to one operation. Without this, paracetamol's handful of distinct operations arrive
    as hundreds of copies and every pairwise search pays for all of them.

    The identity is kept, FIRST, so `draw_symmetry(1)` means the first non-identity operation.
    """
    perms, truncated = automorphisms(molecule, max_matches=max_matches,
                                     use_chirality=use_chirality)
    seen: Dict[Tuple[Tuple[int, int, int, int], ...], SymmetryOperation] = {}
    for perm in perms:
        operation = _operation_for(perm, definitions, index=0)
        key = tuple(tuple(sorted((q, tuple(reversed(q))))[0]) for q in operation.transformed_quads)
        if key not in seen:
            seen[key] = operation
    ordered = sorted(seen.values(), key=lambda op: (not op.is_identity, op.permutation))
    return [SymmetryOperation(index=i, permutation=op.permutation,
                              transformed_quads=op.transformed_quads,
                              column_source=op.column_source, is_identity=op.is_identity)
            for i, op in enumerate(ordered)], truncated


# ------------------------------------------------------------------- comparing two distributions

def torsion_embedding(theta: np.ndarray) -> np.ndarray:
    """`(n, 2d)` of `(cos, sin)` per torsion. The periodic embedding the metric layer uses.

    Euclidean distance in this embedding is a monotone function of the chordal distance on the
    torus, so the -pi/+pi seam costs nothing. Comparing raw angles instead would make two frames
    at +179 and -179 degrees maximally far apart -- the defect the clustering metric exists to
    avoid, and it would be just as wrong here.
    """
    th = np.asarray(theta, dtype=np.float64)
    return np.concatenate([np.cos(th), np.sin(th)], axis=1)


def energy_distance(x: np.ndarray, y: np.ndarray, *, rng=None,
                    max_points: int = 1500) -> float:
    """The energy distance between two samples, on the periodic embedding.

    WHY THIS STATISTIC. It is a genuine distance between DISTRIBUTIONS that is zero if and only if
    they are equal, it uses the JOINT configuration of every torsion at once (so correlations are
    preserved, which per-torsion histograms destroy), and it needs no binning -- a binned test on
    d torsions needs exponentially many frames to fill the grid. It is symmetric by construction,
    so "agreement in both directions" is about COVERAGE rather than about the statistic.

    E = 2*mean|x-y| - mean|x-x'| - mean|y-y'|, with |.| the embedding's Euclidean distance.
    Subsampled to `max_points` per side because the computation is quadratic; the subsample is
    drawn once per call from `rng` so a reported number is reproducible from the configuration.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.shape[0] > max_points:
        a = a[rng.choice(a.shape[0], max_points, replace=False)]
    if b.shape[0] > max_points:
        b = b[rng.choice(b.shape[0], max_points, replace=False)]

    def _mean_dist(p, q, same):
        d = np.sqrt(np.maximum(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1), 0.0))
        if same:
            n = d.shape[0]
            if n < 2:
                return 0.0
            return float((d.sum() - np.trace(d)) / (n * (n - 1)))
        return float(d.mean())

    return float(2.0 * _mean_dist(a, b, False) - _mean_dist(a, a, True) - _mean_dist(b, b, True))


def _coverage(reference: np.ndarray, other: np.ndarray, radius: float) -> float:
    """Fraction of `reference` points with a neighbour of `other` within `radius`.

    THE DIRECTIONAL HALF. The energy distance is symmetric and can stay small when one cluster is
    a SUBSET of the other -- a narrow basin nested inside a broad one. Coverage is asked in each
    direction separately and both must pass, which is what distinguishes "these are the same
    distribution" from "this one fits inside that one".
    """
    if reference.size == 0 or other.size == 0:
        return 0.0
    d = np.sqrt(np.maximum(((reference[:, None, :] - other[None, :, :]) ** 2).sum(-1), 0.0))
    return float((d.min(axis=1) <= radius).mean())


@dataclass
class ComparisonTolerance:
    """Thresholds, CALIBRATED rather than invented. See `calibrate`.

    `energy_max` is the largest energy distance accepted as equivalence. `coverage_min` is the
    smallest fraction of either cluster that must be matched by the other. `radius` is the
    neighbourhood used for coverage, in embedding units.
    """

    energy_max: float
    coverage_min: float = 0.90
    radius: float = 0.35
    max_points: int = 1500
    calibration: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {"energy_max": self.energy_max, "coverage_min": self.coverage_min,
                "radius": self.radius, "max_points": self.max_points,
                "calibration": dict(self.calibration)}


def calibrate(cluster_torsions: ClusterTorsions, *, rng=None, margin: float = 3.0,
              max_points: int = 1500) -> ComparisonTolerance:
    """Set `energy_max` from WITHIN-cluster variability, measured on temporally separated halves.

    THE NUMBER HAS TO COME FROM THESE DATA. A universal threshold cannot exist: the energy
    distance has units of the embedding and its sampling floor depends on the number of frames and
    on how correlated they are. So each cluster is split into its FIRST and SECOND half by frame
    index -- temporally separated, therefore carrying the autocorrelation a random split would
    hide -- and the energy distance between the halves is measured. That is the distance two
    samples of the SAME distribution produce here. The threshold is `margin` times the largest
    such value.

    A split-half distance is a floor, not a full accounting of the error: it says nothing about
    slow motions neither half sampled. It is reported with the tolerance so a reader can see what
    the threshold was derived from.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    within = {}
    for label, entry in sorted(cluster_torsions.entries.items()):
        if entry.n_frames < 20:
            continue
        order = np.argsort(entry.frames)
        half = order.size // 2
        a = torsion_embedding(entry.torsions[order[:half]])
        b = torsion_embedding(entry.torsions[order[half:]])
        within[int(label)] = energy_distance(a, b, rng=rng, max_points=max_points)
    if not within:
        raise ValueError("no cluster has enough frames (>= 20) to calibrate a tolerance")
    # THE SCALE IS THE MAGNITUDE, and the sign is why. The energy distance is non-negative in
    # exact arithmetic, but the V-statistic estimated from finite samples of the SAME
    # distribution scatters about zero and comes back slightly NEGATIVE as often as not --
    # measured here at -2e-4 to -1.4e-3 on four clusters. Taking `max(values)` of a set of
    # negative numbers gives a threshold of ~0, which accepts only pairs that are numerically
    # identical and would reject a true symmetry that happened to be sampled a little
    # differently. The scale of the within-cluster noise is its MAGNITUDE, so that is what the
    # margin multiplies. A threshold derived from a set of negative numbers is not a tight
    # tolerance; it is a tolerance that cannot be met except by accident.
    worst = max(abs(v) for v in within.values())
    return ComparisonTolerance(
        energy_max=float(margin * max(worst, 1e-9)), max_points=max_points,
        calibration={"within_cluster_split_half_energy": within,
                     "worst_magnitude": float(worst), "margin": float(margin),
                     "basis": "first vs second half of each cluster by frame index, so the two "
                              "samples are temporally separated and carry the autocorrelation a "
                              "random split would hide",
                     "caveat": "a split-half distance is a FLOOR on the sampling error: it cannot "
                               "see a slow motion that neither half sampled"})


# ------------------------------------------------------------------------------- the pair result

@dataclass
class PairComparison:
    """One ordered attempt to show cluster `a` equivalent to cluster `b` under one operation."""

    a: int
    b: int
    verdict: str                 # "accepted" | "rejected" | "unresolved"
    operation: Optional[int]
    energy: Optional[float] = None
    coverage_a_in_b: Optional[float] = None
    coverage_b_in_a: Optional[float] = None
    reason: str = ""
    partial: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "verdict": self.verdict, "operation": self.operation,
                "energy": self.energy, "coverage_a_in_b": self.coverage_a_in_b,
                "coverage_b_in_a": self.coverage_b_in_a, "partial": self.partial,
                "reason": self.reason, "evidence": dict(self.evidence)}


@dataclass
class MergedClusters:
    """The result of `deduplicate`: merged arrays, labels, masses, and the mapping back."""

    groups: List[List[int]]
    original_to_merged: Dict[int, int]
    labels: np.ndarray                      # per-frame merged labels; noise unchanged
    original_labels: np.ndarray
    entries: Dict[int, ClusterEntry]        # merged, keyed by merged label
    noise: ClusterEntry
    aligned: Dict[int, np.ndarray]          # merged label -> symmetry-ALIGNED descriptors
    operations_used: Dict[int, Optional[int]]
    masses: Dict[int, float]
    noise_mass: float

    def describe(self) -> Dict[str, Any]:
        return {"groups": [list(g) for g in self.groups],
                "original_to_merged": {int(k): int(v)
                                       for k, v in sorted(self.original_to_merged.items())},
                "masses": {int(k): float(v) for k, v in sorted(self.masses.items())},
                "noise_mass": float(self.noise_mass),
                "operations_used": {int(k): v for k, v in sorted(self.operations_used.items())},
                "mass_sum_including_noise": float(sum(self.masses.values()) + self.noise_mass)}


class TorsionSymmetry:
    """Detect molecular symmetry and decide which existing clusters it merges.

    ```python
    detector = TorsionSymmetry(mol, torsion_idx)
    detector.fit(cluster_torsions)
    detector.cluster_groups          # [[1, 4], [2], [3]] in ORIGINAL labels
    result = detector.deduplicate(cluster_torsions)
    ```

    Nothing is re-clustered and no MD is re-run: the clusters come in as they are.
    """

    def __init__(self, molecule, torsion_idx: TorsionDefinitions, *,
                 max_matches: int = MAX_AUTOMORPHISMS, use_chirality: bool = True):
        if not isinstance(torsion_idx, TorsionDefinitions):
            raise TypeError(
                "torsion_idx must be a TorsionDefinitions (from load_torsion_json), so that the "
                "indexing convention and the angle units travel with the quadruplets rather than "
                "being supplied again, differently, at each call site")
        if molecule is None:
            raise ValueError("a reference molecule is required; symmetry is a property of it")
        n = molecule.GetNumAtoms()
        outside = [(nm, q) for nm, q in zip(torsion_idx.names, torsion_idx.quads)
                   if any(i >= n for i in q)]
        if outside:
            raise ValueError(f"torsions index atoms outside the molecule ({n} atoms): {outside}")
        _require_bonded_chain(molecule, torsion_idx)

        self.molecule = molecule
        self.definitions = torsion_idx
        self.use_chirality = bool(use_chirality)
        self.operations, self.truncated = distinct_operations(
            molecule, torsion_idx, max_matches=max_matches, use_chirality=use_chirality)
        self.rank_partition = _canonical_rank_partition(molecule)
        self.fitted_ = False

    # ------------------------------------------------------------------------------- the fit

    def fit(self, cluster_torsions: ClusterTorsions, *, coordinates=None, atom_mapping=None,
            tolerance: Optional[ComparisonTolerance] = None, rng=None,
            broken_by: Optional[Sequence[str]] = None) -> "TorsionSymmetry":
        """Compare every cluster pair under every operation and record accepted/rejected/unresolved.

        `coordinates` is `(n_frames, n_atoms, 3)` in the TRAJECTORY's atom order, or None.
        `atom_mapping[i]` is the trajectory index of reference-molecule atom `i`; required when
        coordinates are given, because SDF order and MD topology order are not the same thing and
        assuming they are silently measures different atoms.

        `broken_by` names anything atom-specific that breaks a chemical symmetry it does not
        change the graph of -- a restraint on one ring carbon, a REST2 region covering one side.
        It is RECORDED on every accepted pair rather than acted on: structural equivalence and
        equal populations are different claims, and this module only establishes the first.
        """
        if self.truncated:
            raise RuntimeError(
                f"the automorphism enumeration hit its cap of {MAX_AUTOMORPHISMS} and is "
                f"therefore INCOMPLETE. Accepting it would make 'no further symmetry' "
                f"indistinguishable from 'stopped looking'. Raise max_matches, or reduce the "
                f"molecule to the heavy-atom skeleton the selected torsions actually use.")
        rng = np.random.default_rng(0) if rng is None else rng
        self.broken_by = list(broken_by or [])
        self.coordinates_, self.atom_mapping_ = self._prepare_coordinates(
            coordinates, atom_mapping, cluster_torsions)
        self.tolerance = tolerance or calibrate(cluster_torsions, rng=rng)

        labels = cluster_torsions.cluster_labels
        self.comparisons: List[PairComparison] = []
        accepted: Dict[Tuple[int, int], PairComparison] = {}
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                best = self._compare_pair(cluster_torsions, a, b, rng=rng)
                self.comparisons.append(best)
                if best.verdict == "accepted":
                    accepted[(a, b)] = best

        self.accepted_pairs = accepted
        self.cluster_groups, self.group_operations = self._build_groups(
            cluster_torsions, labels, accepted, rng=rng)
        self.fitted_ = True
        return self

    def _prepare_coordinates(self, coordinates, atom_mapping, cluster_torsions):
        if coordinates is None:
            if atom_mapping is not None:
                raise ValueError("atom_mapping given without coordinates; nothing to map")
            return None, None
        xyz = np.asarray(coordinates, dtype=np.float64)
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError(f"coordinates must be (n_frames, n_atoms, 3); got {xyz.shape}")
        if xyz.shape[0] != cluster_torsions.labels.size:
            raise ValueError(
                f"{xyz.shape[0]} coordinate frames for {cluster_torsions.labels.size} labelled "
                f"frames; they must be the same frames in the same order")
        n_ref = self.molecule.GetNumAtoms()
        if atom_mapping is None:
            raise ValueError(
                f"coordinates given with no atom_mapping. SDF atom order and MD topology order "
                f"are not the same thing and matching element counts does not show that they "
                f"are -- two carbons can be exchanged with every count identical. Supply "
                f"atom_mapping[i] = trajectory index of reference atom i, or verify it with "
                f"`verify_atom_mapping`.")
        mapping = np.asarray(atom_mapping, dtype=np.int64).ravel()
        if mapping.size != n_ref:
            raise ValueError(f"atom_mapping has {mapping.size} entries for {n_ref} reference atoms")
        if len(set(mapping.tolist())) != mapping.size:
            raise ValueError("atom_mapping is not injective; two reference atoms share a frame atom")
        if mapping.min() < 0 or mapping.max() >= xyz.shape[1]:
            raise ValueError(
                f"atom_mapping indexes outside the trajectory ({xyz.shape[1]} atoms)")
        return xyz, mapping

    # -------------------------------------------------------------- transforming a descriptor set

    def transform(self, theta: np.ndarray, operation: SymmetryOperation,
                  frames: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ONE operation to every frame of `theta`, as a single object.

        Column `c` of the result is the transformed quadruplet's angle. It is REUSED from a stored
        column when that quadruplet is in the table (possibly reversed, which is exact), and
        otherwise RECOMPUTED from coordinates. It is never approximated: a transformed quadruplet
        that shares a central bond with a stored one does NOT differ from it by a constant, because
        ring distortion and pyramidalization move the outer atoms independently. Substituting an
        idealised 180-degree shift there would hide exactly the deviations the comparison is
        supposed to be sensitive to.
        """
        theta = np.asarray(theta, dtype=np.float64)
        out = np.empty_like(theta)
        for column, source in enumerate(operation.column_source):
            if source[0] in ("column", "column_reversed"):
                out[:, column] = theta[:, source[1]]
            else:
                if self.coordinates_ is None or frames is None:
                    raise RuntimeError(
                        f"column {column} ({self.definitions.names[column]}) maps to quadruplet "
                        f"{operation.transformed_quads[column]}, which is not in the torsion "
                        f"table. Recomputing it needs coordinates; this operation is unresolved.")
                out[:, column] = self._angles_from_coordinates(
                    operation.transformed_quads[column], frames)
        return wrap(out)

    def _angles_from_coordinates(self, quad: Sequence[int], frames: np.ndarray) -> np.ndarray:
        """The dihedral of `quad` over `frames`, from mapped coordinates.

        The quadruplet is in REFERENCE-molecule indices and is mapped through `atom_mapping` to
        the trajectory's own order before any geometry is touched.
        """
        mapped = [int(self.atom_mapping_[int(i)]) for i in quad]
        p = self.coordinates_[np.asarray(frames, dtype=np.int64)][:, mapped, :]
        return dihedral(p[:, 0], p[:, 1], p[:, 2], p[:, 3])

    # ---------------------------------------------------------------------------- one comparison

    def _compare_pair(self, data: ClusterTorsions, a: int, b: int, *, rng) -> PairComparison:
        """Search operations for one that maps cluster `a`'s distribution onto cluster `b`'s.

        Representatives SCREEN first: the circular mean of each cluster under the operation. A
        screen that is far away cannot become close under a full comparison, and the full
        comparison is quadratic. Nothing is accepted on the screen.
        """
        ea, eb = data.entries[a], data.entries[b]
        emb_b = torsion_embedding(eb.torsions)
        best: Optional[PairComparison] = None
        unresolved: List[PairComparison] = []

        for operation in self.operations:
            if operation.is_identity:
                continue
            if operation.needs_coordinates and self.coordinates_ is None:
                unresolved.append(PairComparison(
                    a=a, b=b, verdict="unresolved", operation=operation.index,
                    reason=(f"operation {operation.index} maps columns "
                            f"{list(operation.unresolved_columns)} "
                            f"({[self.definitions.names[c] for c in operation.unresolved_columns]}) "
                            f"to quadruplets not in the torsion table: "
                            f"{[list(operation.transformed_quads[c]) for c in operation.unresolved_columns]}. "
                            f"Supply trajectory coordinates and an atom_mapping, or add those "
                            f"torsions to the definition file."),
                    evidence={"missing_quadruplets":
                              [list(operation.transformed_quads[c])
                               for c in operation.unresolved_columns]}))
                continue
            try:
                moved = self.transform(ea.torsions, operation, frames=ea.frames)
            except RuntimeError as error:
                unresolved.append(PairComparison(a=a, b=b, verdict="unresolved",
                                                 operation=operation.index, reason=str(error)))
                continue

            emb_a = torsion_embedding(moved)
            energy = energy_distance(emb_a, emb_b, rng=rng, max_points=self.tolerance.max_points)
            cov_ab = _coverage(emb_a, emb_b, self.tolerance.radius)
            cov_ba = _coverage(emb_b, emb_a, self.tolerance.radius)
            ok_energy = energy <= self.tolerance.energy_max
            ok_cover = (cov_ab >= self.tolerance.coverage_min
                        and cov_ba >= self.tolerance.coverage_min)
            partial = ok_energy and not ok_cover and max(cov_ab, cov_ba) >= \
                self.tolerance.coverage_min

            candidate = PairComparison(
                a=a, b=b, verdict=("accepted" if (ok_energy and ok_cover) else "rejected"),
                operation=operation.index, energy=float(energy),
                coverage_a_in_b=float(cov_ab), coverage_b_in_a=float(cov_ba), partial=partial,
                reason=("" if (ok_energy and ok_cover) else
                        f"energy {energy:.4f} vs max {self.tolerance.energy_max:.4f}; "
                        f"coverage {cov_ab:.3f}/{cov_ba:.3f} vs min "
                        f"{self.tolerance.coverage_min:.3f}"
                        + (". ONE CLUSTER COVERS ONLY PART OF THE OTHER: this is a subset "
                           "relationship, not an equivalence, and merging whole clusters would "
                           "be wrong." if partial else "")),
                evidence={"permutation": list(operation.permutation),
                          "moved_atoms": [i for i, j in enumerate(operation.permutation) if i != j],
                          "broken_by": list(self.broken_by)})
            if candidate.verdict == "accepted":
                return candidate
            if best is None or (candidate.energy or np.inf) < (best.energy or np.inf):
                best = candidate

        if best is not None:
            return best
        if unresolved:
            return unresolved[0]
        return PairComparison(a=a, b=b, verdict="rejected", operation=None,
                              reason="no non-identity symmetry operation exists for this molecule "
                                     "and torsion selection")

    # --------------------------------------------------------------------------- building groups

    def _build_groups(self, data: ClusterTorsions, labels: List[int],
                      accepted: Dict[Tuple[int, int], PairComparison], *, rng):
        """Group accepted pairs, VERIFYING each group against a common reference.

        TRANSITIVITY IS NOT ASSUMED. Under a loose threshold A~B and B~C can both pass while A and
        C are plainly different -- equivalence by chaining is how a merge swallows a molecule. So
        a connected component is only a CANDIDATE group: every member is then re-tested against
        the group's reference (its lowest label) under ONE operation each, and any member that
        fails that direct test is split back out into its own group. The operation that related it
        to the reference is recorded per member, so the alignment is reproducible.
        """
        parent = {label: label for label in labels}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b) in accepted:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        components: Dict[int, List[int]] = {}
        for label in labels:
            components.setdefault(find(label), []).append(label)

        groups, operations = [], {}
        for root, members in sorted(components.items()):
            members = sorted(members)
            reference = members[0]
            operations[reference] = None
            confirmed = [reference]
            rejected = []
            for member in members[1:]:
                direct = self._compare_pair(data, reference, member, rng=rng)
                if direct.verdict == "accepted":
                    confirmed.append(member)
                    operations[member] = direct.operation
                    self.comparisons.append(
                        PairComparison(a=reference, b=member, verdict="accepted",
                                       operation=direct.operation, energy=direct.energy,
                                       coverage_a_in_b=direct.coverage_a_in_b,
                                       coverage_b_in_a=direct.coverage_b_in_a,
                                       reason="group verification against the common reference",
                                       evidence=dict(direct.evidence, role="group-verification"))) 
                else:
                    rejected.append(member)
                    self.comparisons.append(
                        PairComparison(a=reference, b=member, verdict="rejected",
                                       operation=direct.operation, energy=direct.energy,
                                       coverage_a_in_b=direct.coverage_a_in_b,
                                       coverage_b_in_a=direct.coverage_b_in_a,
                                       reason=("TRANSITIVE MERGE REFUSED: this cluster was linked "
                                               "into the group through another member, but does "
                                               "not match the group's reference directly. "
                                               + direct.reason),
                                       evidence=dict(direct.evidence, role="transitivity-guard")))
            groups.append(confirmed)
            for member in rejected:
                groups.append([member])
                operations[member] = None
        return sorted((sorted(g) for g in groups), key=lambda g: g[0]), operations

    # ---------------------------------------------------------------------------- deduplication

    def deduplicate(self, cluster_torsions: ClusterTorsions) -> MergedClusters:
        """Merge the accepted groups. Masses SUM; nothing is multiplied by a symmetry order.

        Frames are concatenated without duplication, weights travel with them, and the merged mass
        is the sum of the originals -- once. Multiplying by the size of the symmetry group would
        count the same frames several times; the group's order says how many labels the ensemble
        was split into, not how much probability it holds.

        Noise keeps its own label and is returned unchanged. The ensemble normalisation is
        preserved exactly: merged masses plus noise mass equal the original total.

        The symmetry-ALIGNED descriptors are returned separately, in `.aligned`, never written
        over the original observations: an aligned array is a derived view under a specific
        operation, and a reader who cannot tell it from the measurement cannot check either.
        """
        if not self.fitted_:
            raise RuntimeError("call fit() before deduplicate()")
        groups = self.cluster_groups
        original_to_merged, entries, aligned, operations_used, masses = {}, {}, {}, {}, {}

        for merged_label, group in enumerate(groups):
            frames, torsions, weights, aligned_parts, mass = [], [], [], [], 0.0
            reference = group[0]
            for member in group:
                entry = cluster_torsions.entries[member]
                frames.append(entry.frames)
                torsions.append(entry.torsions)
                weights.append(entry.weights)
                mass += entry.mass                      # SUMMED ONCE, never scaled
                operation_index = self.group_operations.get(member)
                if operation_index is None or member == reference:
                    aligned_parts.append(entry.torsions)
                else:
                    aligned_parts.append(
                        self.transform(entry.torsions, self.operations[operation_index],
                                       frames=entry.frames))
                original_to_merged[member] = merged_label
                operations_used[member] = operation_index
            order = np.argsort(np.concatenate(frames))
            merged_frames = np.concatenate(frames)[order]
            if merged_frames.size != np.unique(merged_frames).size:
                raise RuntimeError("a frame appears in two clusters; the partition is not one")
            entries[merged_label] = ClusterEntry(
                label=merged_label, frames=merged_frames,
                torsions=np.concatenate(torsions)[order],
                weights=np.concatenate(weights)[order], mass=float(mass))
            aligned[merged_label] = np.concatenate(aligned_parts)[order]
            masses[merged_label] = float(mass)

        merged_labels = np.full(cluster_torsions.labels.shape, cluster_torsions.noise_label,
                                dtype=np.int64)
        for original, merged in original_to_merged.items():
            merged_labels[cluster_torsions.labels == original] = merged

        total = sum(masses.values()) + cluster_torsions.noise.mass
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(
                f"merged masses plus noise come to {total!r}, not 1. The ensemble normalisation "
                f"must be preserved exactly by a relabelling.")
        return MergedClusters(
            groups=[list(g) for g in groups], original_to_merged=original_to_merged,
            labels=merged_labels, original_labels=cluster_torsions.labels.copy(),
            entries=entries, noise=cluster_torsions.noise, aligned=aligned,
            operations_used=operations_used, masses=masses,
            noise_mass=float(cluster_torsions.noise.mass))

    # ----------------------------------------------------------------------------------- report

    def report(self) -> Dict[str, Any]:
        """Everything a result file should carry: operations, comparisons, groups, provenance."""
        if not self.fitted_:
            raise RuntimeError("call fit() before report()")
        import sys

        versions = {"python": sys.version.split()[0], "numpy": np.__version__}
        try:
            import rdkit
            versions["rdkit"] = rdkit.__version__
        except Exception:
            pass
        try:
            import sklearn
            versions["scikit-learn"] = sklearn.__version__
        except Exception:
            pass
        try:
            from md_tools import __version__ as md_version
            versions["md-tools"] = md_version
        except Exception:
            pass
        return {
            "schema_version": 1,
            "torsions": {"names": list(self.definitions.names),
                         "quadruplets": [list(q) for q in self.definitions.quads],
                         "central_bonds": [list(b) for b in self.definitions.central_bonds()],
                         "indexing": self.definitions.indexing, "units": self.definitions.units,
                         "source": self.definitions.source},
            "operations": [op.describe() for op in self.operations],
            "automorphism_search": {"truncated": self.truncated, "cap": MAX_AUTOMORPHISMS,
                                    "use_chirality": self.use_chirality,
                                    "distinct_on_selected_torsions": len(self.operations)},
            "canonical_rank_partition": {int(k): v for k, v in self.rank_partition.items()},
            "tolerance": self.tolerance.describe(),
            "comparisons": [c.describe() for c in self.comparisons],
            "cluster_groups": [list(g) for g in self.cluster_groups],
            "group_operations": {int(k): v for k, v in self.group_operations.items()},
            "coordinates_available": self.coordinates_ is not None,
            "symmetry_broken_by": list(self.broken_by),
            "caveats": [
                "a graph automorphism establishes STRUCTURAL equivalence; it does not establish "
                "that two clusters should hold equal population. An atom-specific restraint or "
                "scaling region breaks the thermodynamic symmetry while leaving the graph intact "
                "-- declare those with broken_by.",
                "tolerances are calibrated from within-cluster split-half variability on THESE "
                "data; they are not universal constants and must not be copied to another system.",
                "failure to reject is not evidence of equivalence: an underpowered comparison "
                "accepts everything. Coverage and effect size are reported for that reason.",
            ],
            "versions": versions,
        }


# --------------------------------------------------------------------------------- geometry

def dihedral(p0, p1, p2, p3) -> np.ndarray:
    """The A-B-C-D dihedral, in radians, IUPAC/MDTraj sign: positive is clockwise along B -> C.

    The same convention the CV reporter records in its sidecar, so a recomputed angle and a stored
    one are directly comparable. Coordinates must already be whole across periodic boundaries --
    see `make_whole`.
    """
    p0, p1, p2, p3 = (np.asarray(x, dtype=np.float64) for x in (p0, p1, p2, p3))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def make_whole(coordinates, bonds: Iterable[Tuple[int, int]], box_vectors) -> np.ndarray:
    """Unwrap a molecule broken across periodic boundaries, bond by bond.

    A molecule that straddles the box edge has atoms on both sides, and a dihedral computed from
    those raw positions is nonsense -- a plausible-looking number that describes no geometry. This
    walks the bond graph from atom 0 and shifts each atom by the box vector that brings it nearest
    its bonded neighbour.
    """
    xyz = np.array(coordinates, dtype=np.float64, copy=True)
    box = np.asarray(box_vectors, dtype=np.float64)
    if box.ndim == 2:
        box = np.broadcast_to(box, (xyz.shape[0],) + box.shape).copy()
    lengths = np.stack([np.linalg.norm(box[:, i, :], axis=-1) for i in range(3)], axis=-1)

    neighbours: Dict[int, List[int]] = {}
    for i, j in bonds:
        neighbours.setdefault(int(i), []).append(int(j))
        neighbours.setdefault(int(j), []).append(int(i))

    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for other in neighbours.get(current, ()):
            if other in seen:
                continue
            delta = xyz[:, other, :] - xyz[:, current, :]
            xyz[:, other, :] -= lengths * np.round(delta / lengths)
            seen.add(other)
            stack.append(other)
    return xyz


def verify_atom_mapping(molecule, topology_bonds: Iterable[Tuple[int, int]],
                        elements: Sequence[str], mapping: Optional[Sequence[int]] = None):
    """Check (or find) the reference-molecule -> trajectory atom correspondence BY GRAPH.

    Matching element counts is not a verification: two carbons can be exchanged with every count
    identical. This compares the BOND GRAPH with elements attached, so a swap that preserves
    counts but not connectivity is caught.

    With `mapping`, returns `(True, reason)` or `(False, reason)`. Without, searches for one and
    returns `(mapping_or_None, reason)`.
    """
    n = molecule.GetNumAtoms()
    reference_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                       for b in molecule.GetBonds()}
    reference_elements = [a.GetSymbol() for a in molecule.GetAtoms()]
    other_bonds = {tuple(sorted((int(i), int(j)))) for i, j in topology_bonds}
    other_elements = [str(e) for e in elements]

    if len(other_elements) != n:
        return (False, f"{n} reference atoms, {len(other_elements)} trajectory atoms")

    def _ok(candidate) -> Tuple[bool, str]:
        for i, j in enumerate(candidate):
            if reference_elements[i] != other_elements[j]:
                return False, (f"reference atom {i} is {reference_elements[i]} but maps to "
                               f"trajectory atom {j}, which is {other_elements[j]}")
        mapped = {tuple(sorted((candidate[i], candidate[j]))) for i, j in reference_bonds}
        if mapped != other_bonds:
            missing = sorted(mapped - other_bonds)[:4]
            extra = sorted(other_bonds - mapped)[:4]
            return False, (f"the bond graphs disagree under this mapping; mapped-but-absent "
                           f"{missing}, present-but-unmapped {extra}")
        return True, "elements and the full bond graph agree under this mapping"

    if mapping is not None:
        candidate = [int(i) for i in mapping]
        if sorted(candidate) != list(range(n)):
            return (False, "mapping is not a permutation of the trajectory's atom indices")
        return _ok(candidate)

    identity_ok, reason = _ok(list(range(n)))
    if identity_ok:
        return (list(range(n)), "identity verified: " + reason)
    return (None, "the identity mapping does not hold (" + reason
            + "); supply an explicit atom_mapping")


# ----------------------------------------------------------------------------------- drawing

#: The two sides of a swap. Chosen to be distinguishable in greyscale as well as in colour, so a
#: printed figure still shows which half maps to which.
SWAP_COLOURS = ((0.85, 0.33, 0.10), (0.00, 0.45, 0.70))      # orange-red, blue
#: An atom the operation leaves where it is.
FIXED_COLOUR = (0.80, 0.80, 0.80)
#: An atom whose INDEX changed under the operation. Marked inside the transformed torsion rather
#: than instead of it. Gold rather than black: a dark fill hides the atom INDEX printed on it, and
#: the index is the only way to check a quadruplet against the picture.
CHANGED_COLOUR = (0.98, 0.78, 0.12)


def permutation_colours(permutation: Sequence[int], *, include_fixed: bool = False):
    """Colour the atoms by WHICH SIDE of the operation they are on.

    A single highlight colour says "these atoms move" and stops there. For a two-fold flip the
    useful statement is which atom goes to which: the permutation decomposes into cycles, and for
    a flip every non-trivial cycle is a transposition {i, j}. One member gets the first colour and
    its image the second, so the picture shows the two halves being exchanged rather than a single
    undifferentiated blob.

    Longer cycles (a three-fold axis, say) alternate through the same two colours by position, and
    an odd-length cycle therefore repeats a colour -- unavoidable with two colours, and harmless,
    because the pairing it would otherwise assert is not a pairing in that case.

    Returns `{atom_index: (r, g, b)}`, covering only the atoms that MOVE unless `include_fixed`.
    """
    perm = [int(i) for i in permutation]
    colours: Dict[int, Tuple[float, float, float]] = {}
    seen = set()
    for start in range(len(perm)):
        if start in seen:
            continue
        cycle, current = [], start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = perm[current]
        if len(cycle) == 1:
            if include_fixed:
                colours[cycle[0]] = FIXED_COLOUR
            continue
        for position, atom in enumerate(cycle):
            colours[atom] = SWAP_COLOURS[position % 2]
    return colours


def _drawing_molecule(molecule, *, keep_hydrogens: bool):
    """A 2D-depicted copy. Hydrogens are KEPT by default because torsions often name them."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    copy = Chem.Mol(molecule)
    if not keep_hydrogens:
        copy = Chem.RemoveHs(copy)
    if copy.GetNumConformers() == 0 or not copy.GetConformer().Is3D():
        AllChem.Compute2DCoords(copy)
    else:
        copy = Chem.Mol(copy)
        AllChem.Compute2DCoords(copy)
    return copy


def draw_torsions(molecule, torsion_idx: TorsionDefinitions, *, output,
                  keep_hydrogens: bool = True, panel: Tuple[int, int] = (300, 260),
                  columns: int = 3) -> Dict[str, Any]:
    """An SVG: the molecule with atom indices and numbered central bonds, then one panel each.

    Panel `k` highlights ALL FOUR atoms of torsion `k` and its central bond, so a reader can check
    the definition against the molecule rather than against the index list that produced it.

    Returns the table linking torsion id, name, quadruplet and central bond -- the same table the
    JSON report carries, so the picture and the record cannot disagree.
    """
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = _drawing_molecule(molecule, keep_hydrogens=keep_hydrogens)
    width, height = panel
    n = torsion_idx.n_torsions
    rows = 1 + (n + columns - 1) // columns
    drawer = rdMolDraw2D.MolDraw2DSVG(width * columns, height * rows,
                                      width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.legendFontSize = 18

    mols, legends, highlight_atoms, highlight_bonds = [], [], [], []
    centres = []
    for name, quad in zip(torsion_idx.names, torsion_idx.quads):
        bond = mol.GetBondBetweenAtoms(int(quad[1]), int(quad[2]))
        centres.append(bond.GetIdx() if bond is not None else None)

    mols.append(mol)
    legends.append(f"{mol.GetNumAtoms()} atoms; central bonds numbered by torsion id")
    highlight_atoms.append([int(i) for q in torsion_idx.quads for i in q])
    highlight_bonds.append([c for c in centres if c is not None])

    for k, (name, quad) in enumerate(zip(torsion_idx.names, torsion_idx.quads)):
        mols.append(mol)
        legends.append(f"{k}: {name}  [{'-'.join(str(i) for i in quad)}]")
        highlight_atoms.append([int(i) for i in quad])
        highlight_bonds.append([centres[k]] if centres[k] is not None else [])

    svg = Draw.MolsToGridImage(
        mols, molsPerRow=columns, subImgSize=panel, legends=legends,
        highlightAtomLists=highlight_atoms, highlightBondLists=highlight_bonds,
        useSVG=True)
    text = svg.data if hasattr(svg, "data") else str(svg)
    Path(output).write_text(text, encoding="utf-8")

    return {"output": str(output),
            "table": [{"id": k, "name": name, "quadruplet": list(quad),
                       "central_bond": list(sorted(quad[1:3]))}
                      for k, (name, quad) in enumerate(zip(torsion_idx.names,
                                                           torsion_idx.quads))]}


def _draw_operation(self, which: int = 1, *, output, **kwargs) -> Dict[str, Any]:
    """Draw ONE symmetry OPERATION: `detector.draw_operation(1, output=...)`.

    A thin name over `draw_operation_effect`, kept because the original specification called this
    `draw_symmetry`. There is ONE operation drawing, not two: an earlier version rendered a grid
    with a panel per torsion, which showed the same facts in more space and drifted out of step
    with the before/after one as soon as both existed.

    INDEXING: operation 0 is the IDENTITY, so `1` is the first non-identity operation. Asking for
    an index the molecule does not have is refused with the count rather than wrapping around to
    the identity and drawing a picture in which nothing moves.
    """
    if which >= len(self.operations):
        raise IndexError(
            f"operation {which} does not exist: this molecule has {len(self.operations)} "
            f"operations distinct on the selected torsions, index 0 being the identity"
            + (" -- so it has NO non-identity symmetry here" if len(self.operations) == 1 else
               f", so the valid non-identity indices are 1..{len(self.operations) - 1}"))
    return self.draw_operation_effect(which, output=output, **kwargs)


#: The molecule-level drawing, by OPERATION index. `draw_symmetry` is kept as its original name
#: from the specification; `draw_operation` says what it draws.
TorsionSymmetry.draw_operation = _draw_operation
TorsionSymmetry.draw_symmetry = _draw_operation


#: The name used at the call site, matching `t_hdbscan` and `t_mi`.
t_symmetry = TorsionSymmetry


# -------------------------------------------------------------------- drawing a symmetry GROUP

#: One colour per drawn torsion, reused between the molecule panel and its histogram row so the
#: two can be read against each other without counting.
TORSION_COLOURS = ((0.85, 0.33, 0.10), (0.00, 0.45, 0.70), (0.47, 0.67, 0.19),
                   (0.49, 0.18, 0.56), (0.93, 0.69, 0.13))


def _molecule_png(molecule, highlight_atoms, size=(420, 360), colours=None,
                  bonds=None, bond_colours=None, atom_indices=True) -> bytes:
    """A PNG of the molecule with `highlight_atoms` marked, for embedding in a figure.

    `colours` is the `{atom: rgb}` from `permutation_colours`, so the two sides of a swap are
    drawn differently rather than as one undifferentiated highlight.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = _drawing_molecule(molecule, keep_hydrogens=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.addAtomIndices = bool(atom_indices)
    options.highlightBondWidthMultiplier = 18
    atoms = [int(i) for i in highlight_atoms]
    kwargs = {}
    if colours:
        kwargs["highlightAtomColors"] = {int(k): tuple(v) for k, v in colours.items()
                                         if int(k) in atoms}
    if bonds:
        kwargs["highlightBonds"] = [int(b) for b in bonds]
        if bond_colours:
            kwargs["highlightBondColors"] = {int(k): tuple(v) for k, v in bond_colours.items()}
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atoms, **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _circular_mean(theta: np.ndarray) -> float:
    """The mean direction. An arithmetic mean of angles is wrong across the seam: two frames at
    +179 and -179 average to 0, which is the opposite side of the circle."""
    return float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))


def _draw_sym_group(self, group, *, output, cluster_torsions=None,
                    bins: int = 48) -> Dict[str, Any]:
    """THE MAIN DRAWING: one merged symmetry GROUP -- the operation relating its members, their
    angles, and which torsions move together.

    `group` is the MERGED label (0, 1, 2, ... in the order of `cluster_groups`), or a list of
    original labels. Indexing is the merged one because that is what `deduplicate` returns and
    what a population table is keyed by.

    NOT THE SAME INDEX as `draw_operation` (also available as `draw_symmetry`), which numbers the
    molecule's OPERATIONS with the identity at 0. A group is a set of clusters; an operation is a
    permutation of atoms. They are different objects, so they are numbered separately rather than
    sharing one index that would mean two things -- and `draw_sym_group(1)` is the second merged
    group, not the first non-identity operation.

    Three rows:

    * the molecule, with the atoms the relating operation moves;
    * each torsion PAIR as a joint scatter, coloured by ORIGINAL cluster label, BEFORE alignment
      -- this is where a symmetry-related pair shows up as two separated lobes;
    * the same AFTER applying the operation, where they should coincide. A pair that does not
      coincide here was merged on a tolerance that is too loose, and the picture says so more
      directly than any number.

    The MI between torsion columns is computed over the merged group and annotated, because
    "which torsions are correlated" is a statement about the joint distribution and the scatter
    alone can hide a dependence that the marginals do not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import io
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if not self.fitted_:
        raise RuntimeError("call fit() before draw_sym_group()")
    if isinstance(group, (list, tuple, set)):
        members = sorted(int(x) for x in group)
        merged_label = next((i for i, g in enumerate(self.cluster_groups)
                             if sorted(g) == members), None)
    else:
        merged_label = int(group)
        if merged_label >= len(self.cluster_groups):
            raise IndexError(
                f"merged group {merged_label} does not exist: the fit produced "
                f"{len(self.cluster_groups)} groups, {self.cluster_groups}. Note that this is the "
                f"MERGED label, not an operation index -- draw_operation() takes the latter.")
        members = list(self.cluster_groups[merged_label])

    if cluster_torsions is None:
        raise ValueError("pass cluster_torsions= so the member angles can be drawn")

    names = list(self.definitions.names)
    reference = members[0]

    before, after, used = {}, {}, {}
    for member in members:
        entry = cluster_torsions.entries[member]
        before[member] = entry.torsions
        operation_index = self.group_operations.get(member)
        used[member] = operation_index
        after[member] = (entry.torsions if operation_index is None or member == reference
                         else self.transform(entry.torsions, self.operations[operation_index],
                                             frames=entry.frames))

    operation = None
    for member in members:
        if used[member]:
            operation = self.operations[used[member]]
            break
    moved_atoms = ([i for i, j in enumerate(operation.permutation) if i != j]
                   if operation is not None else [])

    # ONLY THE TORSIONS THE OPERATION ACTUALLY CHANGES. A column the operation maps to itself is
    # invariant under it: both clusters hold the same distribution there by construction, so a
    # panel for it shows two curves on top of each other and says nothing about the symmetry.
    # Drawing every torsion -- or every PAIR of torsions -- buries the one or two that carry the
    # relation among panels that cannot distinguish anything. So the panel count IS the number of
    # torsions the operation moves: one for a single two-fold bond, two for a coupled pair.
    if operation is None:
        relevant = list(range(len(names)))
        relevance = "single cluster: no operation, so every torsion is shown"
    else:
        relevant = [c for c, source in enumerate(operation.column_source)
                    if source != ("column", c)]
        relevance = (f"{len(relevant)} of {len(names)} torsions move under operation "
                     f"{operation.index}")
        if not relevant:
            relevant = list(range(len(names)))
            relevance = ("the operation leaves every selected torsion invariant; showing all, "
                         "and note that it cannot have related these clusters")

    # TWO HISTOGRAMS PER RELEVANT TORSION: the merge, before and after.
    #
    # LEFT is what the clustering produced -- one colour per original cluster, sitting in separate
    # lobes. RIGHT is what the merge asserts: every member mapped through the operation that
    # relates it to the reference, pooled into ONE distribution in one colour. The claim being
    # made is that the right panel is a single state, and the eye checks that directly. An
    # overlay of dashed outlines on one axis, which is what this drew first, asks the reader to
    # do the merge in their head.
    n_rows = len(relevant)
    row_colours = {torsion: TORSION_COLOURS[k % len(TORSION_COLOURS)]
                   for k, torsion in enumerate(relevant)}

    # THE MOLECULE GETS ITS OWN COLUMN AND ROOM TO BE READ. It was a third the width of a
    # histogram and letterboxed inside that, which makes the atom indices -- the only way to
    # check a quadruplet against the picture -- illegible.
    figure = plt.figure(figsize=(7.2 + 9.4, max(4.6, 3.6 * n_rows) + 1.2),
                        layout="constrained")
    grid = figure.add_gridspec(n_rows, 3, width_ratios=[1.55, 1.0, 1.0])

    edges = np.linspace(-np.pi, np.pi, int(bins) + 1)
    # A SEPARATE PALETTE FROM THE TORSION COLOURS, deliberately. The row colours identify which
    # TORSION a row is about; these identify which CLUSTER a bar belongs to. Sharing one palette
    # made orange mean "the aryl torsion" in the molecule and "cluster 3" in the histogram beside
    # it -- two meanings for one colour in one figure.
    cluster_palette = ("0.25", "0.62", "#7F3FBF", "#2E8B8B", "#B8860B")

    # EACH DRAWN TORSION IS HIGHLIGHTED, in the colour of its own row. A panel of histograms
    # labelled `aryl` and `phenol` beside an unmarked molecule asks the reader to resolve the
    # names against a quadruplet list somewhere else; highlighting the three bonds each torsion
    # is measured over answers it in the picture.
    mol_for_bonds = _drawing_molecule(self.molecule, keep_hydrogens=True)
    bond_highlights, bond_colour_map, atom_highlights, atom_colour_map = [], {}, [], {}
    for torsion in relevant:
        quad = self.definitions.quads[torsion]
        for bond in _bond_path(mol_for_bonds, quad):
            bond_highlights.append(bond)
            bond_colour_map[bond] = row_colours[torsion]
        for atom in quad:
            atom_highlights.append(int(atom))
            atom_colour_map.setdefault(int(atom), row_colours[torsion])

    axis = figure.add_subplot(grid[:, 0])
    axis.imshow(mpimg.imread(io.BytesIO(_molecule_png(
        self.molecule, atom_highlights, size=(900, 760), colours=atom_colour_map,
        bonds=bond_highlights, bond_colours=bond_colour_map)), format="png"))
    axis.axis("off")
    axis.set_title(f"group {merged_label}: clusters {members}\n"
                   + (f"operation {operation.index}, {len(moved_atoms)} atoms move; "
                      f"each torsion in its row colour"
                      if operation is not None else "no operation applied"), fontsize=11)

    pooled_mass = sum(cluster_torsions.entries[m].mass for m in members)
    for row, torsion in enumerate(relevant):
        left = figure.add_subplot(grid[row, 1])
        for k, member in enumerate(members):
            left.hist(before[member][:, torsion], bins=edges, density=True, alpha=0.72,
                      color=cluster_palette[k % len(cluster_palette)],
                      label=f"cluster {member} ({cluster_torsions.entries[member].mass:.3f})")
        left.set_title(f"{names[torsion]} -- BEFORE merge", fontsize=11,
                       color=row_colours[torsion])
        left.legend(fontsize=8)

        right = figure.add_subplot(grid[row, 2], sharey=left)
        right.hist(np.concatenate([after[m][:, torsion] for m in members]), bins=edges,
                   density=True, color=row_colours[torsion], alpha=0.72,
                   label=f"merged group {merged_label} ({pooled_mass:.3f})")
        right.set_title(f"{names[torsion]} -- AFTER merge (symmetry-aligned)", fontsize=11,
                        color=row_colours[torsion])
        right.legend(fontsize=8)

        for panel in (left, right):
            panel.set_xlim(-np.pi, np.pi)
            panel.set_xlabel(f"{names[torsion]} (rad)")
        left.set_ylabel("density")

    figure.suptitle(relevance, fontsize=10)
    figure.savefig(output, dpi=140)
    plt.close(figure)

    correlations = []
    if len(relevant) > 1:
        from ._t_mi import torsional_mi

        stacked = np.vstack([after[m] for m in members])[:, relevant]
        try:
            report = torsional_mi(stacked, names=[names[c] for c in relevant],
                                  bins=min(24, int(bins)), n_null=6)
            correlations = [{"a": pair["a"], "b": pair["b"], "mi_nats": pair["mi_nats"],
                             "mi_nats_debiased": pair["mi_nats_debiased"],
                             "excess_over_null_in_sd": pair.get("excess_over_null_in_sd")}
                            for pair in report["pairs"]]
        except Exception as error:                       # pragma: no cover - diagnostic only
            correlations = [{"error": f"{type(error).__name__}: {error}"}]

    return {"output": str(output), "merged_label": merged_label, "members": members,
            "operations_used": {int(k): v for k, v in used.items()},
            "moved_atoms": list(moved_atoms),
            "relevant_torsions": [names[c] for c in relevant],
            "invariant_torsions": [n for c, n in enumerate(names) if c not in relevant],
            "circular_means": {int(m): {names[c]: _circular_mean(before[m][:, c])
                                        for c in range(len(names))} for m in members},
            "circular_means_aligned": {int(m): {names[c]: _circular_mean(after[m][:, c])
                                                for c in range(len(names))} for m in members},
            "correlations": correlations}


TorsionSymmetry.draw_sym_group = _draw_sym_group


def _indexed_draw_options():
    """Draw options with atom indices on, for any figure whose legend names atom numbers."""
    from rdkit.Chem.Draw import rdMolDraw2D

    options = rdMolDraw2D.MolDrawOptions()
    options.addAtomIndices = True
    options.legendFontSize = 15
    options.highlightBondWidthMultiplier = 16
    return options


def _bond_path(molecule, quad) -> List[int]:
    """The three bond indices of a torsion A-B-C-D, so the MEASURED torsion can be drawn."""
    out = []
    for a, b in zip(quad, quad[1:]):
        bond = molecule.GetBondBetweenAtoms(int(a), int(b))
        if bond is not None:
            out.append(bond.GetIdx())
    return out


def _draw_operation_effect(self, operation=None, *, output, group=None, torsions=None,
                           size=(560, 470)) -> Dict[str, Any]:
    """ONE figure, two panels: the molecule BEFORE and AFTER the symmetry operation.

    What a reader needs to see is the PRUNING -- why two clusters are one. So each relevant
    torsion is drawn as the bonds that actually carry it, and the atom that the operation moves is
    coloured on BOTH sides: the bond holding it is red in the before panel and blue in the after
    panel, with its partner the other way round. Watching those two colours exchange IS the
    operation; a single highlight colour over "the atoms that move" cannot show it, which is what
    the first version of this drawing got wrong.

    Atom indices are on, because a legend reading `1-3-4-5` is unreadable against a picture that
    does not number its atoms.
    """
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    # Atom indices ON for this drawing specifically: the legends name quadruplets like 1-3-4-5,
    # and those numbers mean nothing against an unnumbered picture.
    Draw.rdDepictor.SetPreferCoordGen(True)

    if not getattr(self, "operations", None):
        raise RuntimeError("no operations; construct the detector first")

    # WHICH OPERATION, said explicitly. This used to take the first non-identity one, which is
    # right only for a molecule that has exactly one -- on any other it silently drew an
    # operation that related nothing, with a legend that looked perfectly correct.
    if group is not None:
        if not self.fitted_:
            raise RuntimeError("call fit() before asking for the operation of a group")
        members = (sorted(int(x) for x in group) if isinstance(group, (list, tuple, set))
                   else list(self.cluster_groups[int(group)]))
        indices = {self.group_operations.get(m) for m in members} - {None}
        if not indices:
            raise ValueError(
                f"group {group} is a single cluster, or its members were not related by any "
                f"operation, so there is no operation to draw")
        chosen = self.operations[sorted(indices)[0]]
    elif operation is None:
        candidates = [op for op in self.operations if not op.is_identity]
        if not candidates:
            raise RuntimeError(
                "this molecule has no non-identity operation on the selected torsions")
        if len(candidates) > 1:
            raise ValueError(
                f"this molecule has {len(candidates)} non-identity operations, so which one to "
                f"draw is not obvious. Pass an operation index, or group=<merged label> to draw "
                f"the one that actually related that group's members.")
        chosen = candidates[0]
    else:
        chosen = self.operations[int(operation)]
    operation = chosen

    relevant = [c for c, source in enumerate(operation.column_source)
                if source != ("column", c)] if torsions is None else list(torsions)
    mol = _drawing_molecule(self.molecule, keep_hydrogens=True)
    names = list(self.definitions.names)

    mol = _drawing_molecule(self.molecule, keep_hydrogens=True)
    names = list(self.definitions.names)

    # BEFORE shows the torsion AS MEASURED: its four atoms and the three bonds the dihedral is
    # taken over, in that torsion's own colour.
    #
    # AFTER shows ONLY WHAT THE PERMUTATION CHANGED -- the positions where the quadruplet's
    # number is different, and nothing else. Re-highlighting the whole image quadruplet buries
    # the one atom that moved among three that did not, and the reader is left comparing two
    # four-number strings by eye. Here `1-3-4-5 -> 1-3-4-10` is drawn as: atom 10, highlighted.
    before_atoms, before_bonds = {}, {}
    after_atoms, after_bonds = {}, {}
    changes, swapped = {}, []
    for position, column in enumerate(relevant):
        colour = TORSION_COLOURS[position % len(TORSION_COLOURS)]
        quad = tuple(int(i) for i in self.definitions.quads[column])
        image = tuple(int(i) for i in operation.transformed_quads[column])
        for atom in quad:
            before_atoms[atom] = colour
        for bond in _bond_path(mol, quad):
            before_bonds[bond] = colour
        differing = [(a, b) for a, b in zip(quad, image) if a != b]
        changes[names[column]] = [[a, b] for a, b in differing]
        swapped.extend(differing)

        # THE WHOLE TRANSFORMED TORSION, in the same colour as the BEFORE panel, so the two
        # panels show the SAME OBJECT in two places rather than a torsion beside a fragment.
        # Drawing only the atoms that changed left the reader with nothing to compare: a lone
        # atom says which number moved but not what the measured torsion became.
        for atom in image:
            after_atoms[atom] = colour
        for bond in _bond_path(mol, image):
            after_bonds[bond] = colour
        # ...and the atoms whose NUMBER changed are then marked out of that torsion, so the
        # difference is still immediately visible inside the whole.
        for _, new_atom in differing:
            after_atoms[new_atom] = CHANGED_COLOUR

    legend_before = "BEFORE -- the torsions as measured: " + ", ".join(
        f"{names[c]} = {'-'.join(map(str, self.definitions.quads[c]))}" for c in relevant)
    legend_after = (f"AFTER operation {operation.index} -- the transformed torsions; "
                    f"GOLD marks the atom whose index changed: ") + ", ".join(
        f"{name} {', '.join(f'{a} to {b}' for a, b in pairs)}"
        for name, pairs in changes.items() if pairs)

    image = Draw.MolsToGridImage(
        [mol, mol], molsPerRow=2, subImgSize=size, legends=[legend_before, legend_after],
        highlightAtomLists=[sorted(before_atoms), sorted(after_atoms)],
        highlightAtomColors=[before_atoms, after_atoms],
        highlightBondLists=[sorted(before_bonds), sorted(after_bonds)],
        highlightBondColors=[before_bonds, after_bonds], useSVG=False,
        drawOptions=_indexed_draw_options())
    data = image.data if hasattr(image, "data") else image
    if hasattr(data, "save"):
        data.save(str(output))
    else:
        Path(output).write_bytes(data)

    return {"output": str(output), "operation": operation.index,
            "relevant_torsions": [names[c] for c in relevant],
            "changed_positions": changes,
            "swapped_atom_pairs": sorted(set(swapped)),
            "quadruplets": {names[c]: {"before": list(self.definitions.quads[c]),
                                       "after": list(operation.transformed_quads[c])}
                            for c in relevant}}


TorsionSymmetry.draw_operation_effect = _draw_operation_effect


# -------------------------------------------------------------- representative configurations

#: Groups are named A, B, C ... in `cluster_groups` order. A merged group is no longer "cluster 2"
#: -- it is a set of original labels -- so reusing an integer for it invites the two to be
#: confused in exactly the place where the difference matters.
def group_name(index: int) -> str:
    """`0 -> 'A'`, `1 -> 'B'`, ... `26 -> 'AA'`."""
    index = int(index)
    if index < 0:
        raise ValueError("group index must be non-negative")
    letters = ""
    while True:
        letters = chr(ord("A") + index % 26) + letters
        index = index // 26 - 1
        if index < 0:
            return letters


def representative_by_vote(fit, labels=None) -> Dict[int, int]:
    """The frame of each cluster with the HIGHEST classifier vote margin; first one on a tie.

    THE REPRESENTATIVE IS AN ACTUAL FRAME, never an average. The mean of two Cartesian structures
    is not a structure, and the circular mean of a bimodal torsion points at the barrier between
    its basins -- the one place the molecule is never found.

    "Highest vote" means the k-NN margin `classify_to_clusters` reports: the frame its neighbours
    agree about most strongly, which is the one furthest from any boundary. Ties are broken by
    taking the FIRST, deterministically, because the alternative -- picking arbitrarily among
    equals -- makes a figure that changes between runs of the same analysis.
    """
    from ._torsions import classify_to_clusters

    labels = fit.labels_ if labels is None else np.asarray(labels)
    out = classify_to_clusters(fit.train_theta_, fit.train_labels_, fit.theta_, units="radians",
                               metric_weights=fit.metric_weights,
                               multiplicities=fit.multiplicities, k=fit.k,
                               min_vote_margin=fit.min_vote)
    margins = np.asarray(out["vote_margin"], dtype=np.float64)
    representatives = {}
    for cluster in sorted({int(c) for c in np.unique(labels) if int(c) >= 0}):
        members = np.flatnonzero(labels == cluster)
        if members.size == 0:
            continue
        best = margins[members]
        representatives[cluster] = int(members[int(np.argmax(best))])   # argmax -> first on ties
    return representatives


def strip_nonpolar_hydrogens(trajectory):
    """Atom indices of everything except hydrogens bonded to CARBON.

    POLAR hydrogens are kept: an O-H or N-H orientation is part of the conformation a reader is
    being shown, and dropping it would hide the phenol rotation this analysis is partly about.
    Nonpolar hydrogens are removed because they trebles the line count of a picture without
    adding a degree of freedom anyone is looking at.
    """
    topology = trajectory.topology
    drop = set()
    for atom in topology.atoms:
        if atom.element.symbol != "H":
            continue
        heavy = [b[0] if b[1].index == atom.index else b[1]
                 for b in topology.bonds if atom.index in (b[0].index, b[1].index)]
        if heavy and heavy[0].element.symbol == "C":
            drop.add(atom.index)
    return [a.index for a in topology.atoms if a.index not in drop]


def draw_representative_structures(trajectory, frames: Dict[str, int], *, output,
                                   atom_indices=None, align: bool = True,
                                   label_atoms: bool = True) -> Dict[str, Any]:
    """Draw one ACTUAL frame per group, PRE-ALIGNED, as a static 3D figure.

    `frames` maps a group name to a frame index. The structures are superposed on each other
    before drawing, so what differs between the panels is CONFORMATION rather than the arbitrary
    position and orientation the box happened to leave the molecule in. Without that, two
    identical conformers look different and two different ones can look alike.

    Static matplotlib rather than an interactive viewer, deliberately: an executed notebook is
    read on GitHub as often as it is run, and a JavaScript viewer renders as nothing there. The
    aligned coordinates are also returned so a caller can write them out for a real viewer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        raise ValueError("no frames given")
    indices = (list(range(trajectory.n_atoms)) if atom_indices is None
               else [int(i) for i in atom_indices])
    subset = trajectory.atom_slice(indices)

    ordered = list(frames.items())
    picked = subset[[int(i) for _, i in ordered]]
    if align:
        picked.superpose(picked, 0)          # onto the FIRST group, so panels share a frame

    topology = picked.topology
    bonds = [(b[0].index, b[1].index) for b in topology.bonds]
    colours = {"C": "0.35", "N": "#2060C0", "O": "#C02020", "H": "0.70", "S": "#C0A020"}

    figure = plt.figure(figsize=(4.8 * len(ordered), 4.4), layout="constrained")
    for position, (name, frame) in enumerate(ordered):
        axis = figure.add_subplot(1, len(ordered), position + 1, projection="3d")
        xyz = picked.xyz[position] * 10.0                      # nm -> angstrom
        for i, j in bonds:
            axis.plot(*zip(xyz[i], xyz[j]), color="0.45", lw=2.0, zorder=1)
        for atom in topology.atoms:
            element = atom.element.symbol
            axis.scatter(*xyz[atom.index], s=(90 if element != "H" else 40),
                         color=colours.get(element, "0.5"), depthshade=False, zorder=2,
                         edgecolors="k", linewidths=0.4)
            if label_atoms and element != "H":
                axis.text(*xyz[atom.index], f" {atom.name}", fontsize=7, color="0.2")
        axis.set_title(f"group {name}  (frame {frame})", fontsize=12)
        axis.set_axis_off()
        # ONE cubic box around the molecule, sized from the structure rather than left to
        # matplotlib: a 3D axes defaults to a lot of padding, and three independently scaled
        # axes would stretch the geometry differently in each panel -- which is the one thing a
        # figure comparing CONFORMATIONS must not do.
        centre = xyz.mean(axis=0)
        span = float(np.abs(xyz - centre).max()) * 1.02
        for setter, value in ((axis.set_xlim, centre[0]), (axis.set_ylim, centre[1]),
                              (axis.set_zlim, centre[2])):
            setter(value - span, value + span)
        axis.set_box_aspect((1, 1, 1), zoom=1.45)

    figure.suptitle("Representative structures, superposed. Nonpolar hydrogens removed; "
                    "polar H kept because its orientation IS part of the conformation.",
                    fontsize=9)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return {"output": str(output), "frames": dict(frames),
            "n_atoms_drawn": int(picked.n_atoms), "aligned": bool(align),
            "aligned_xyz_nm": picked.xyz}
