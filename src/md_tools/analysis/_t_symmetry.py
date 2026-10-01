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


def draw_symmetry_operation(molecule, torsion_idx: TorsionDefinitions,
                            operation: SymmetryOperation, *, output,
                            keep_hydrogens: bool = True,
                            panel: Tuple[int, int] = (340, 300)) -> Dict[str, Any]:
    """An SVG of one operation: which atoms move where, and what it does to each torsion."""
    from rdkit.Chem import Draw

    mol = _drawing_molecule(molecule, keep_hydrogens=keep_hydrogens)
    moved = [i for i, j in enumerate(operation.permutation) if i != j]
    mols, legends, highlights = [], [], []

    mols.append(mol)
    legends.append(f"operation {operation.index}: {len(moved)} atoms move")
    highlights.append(moved)

    for k, (name, quad) in enumerate(zip(torsion_idx.names, torsion_idx.quads)):
        image = operation.transformed_quads[k]
        source = operation.column_source[k]
        # NOT a dict lookup: a dict literal evaluates EVERY value before indexing, and
        # `source[1]` does not exist for ("coordinates",). The lookup form raised IndexError on
        # exactly the operations this drawing is most useful for -- the ones needing coordinates.
        if source[0] == "column":
            where = f"= column {source[1]}"
        elif source[0] == "column_reversed":
            where = f"= column {source[1]} reversed"
        else:
            where = "NOT in the table -- needs coordinates"
        mols.append(mol)
        legends.append(f"{name}: {'-'.join(map(str, quad))} -> "
                       f"{'-'.join(map(str, image))}  {where}")
        highlights.append(sorted({*map(int, quad), *map(int, image)}))

    svg = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=panel, legends=legends,
                               highlightAtomLists=highlights, useSVG=True)
    text = svg.data if hasattr(svg, "data") else str(svg)
    Path(output).write_text(text, encoding="utf-8")
    return {"output": str(output), "operation": operation.describe()}


def _draw_operation(self, which: int = 1, *, output, **kwargs) -> Dict[str, Any]:
    """Draw ONE symmetry OPERATION of the molecule: `detector.draw_operation(1, output=...)`.

    This is the molecule-level view and needs no clusters, so it works before `fit`. For the
    cluster-level view -- which clusters merged, their angles, and which torsions move together --
    use `draw_sym_group`, which is what most callers want and is indexed differently.

    INDEXING: operation 0 is the IDENTITY, always present and always first, so `1` is the first
    non-identity operation -- which is what a caller asking to see "the symmetry" means. Asking
    for an index the molecule does not have is refused with the count, rather than wrapping
    around to the identity and drawing a picture in which nothing moves.
    """
    if which >= len(self.operations):
        raise IndexError(
            f"operation {which} does not exist: this molecule has {len(self.operations)} "
            f"operations distinct on the selected torsions, index 0 being the identity"
            + (" -- so it has NO non-identity symmetry here" if len(self.operations) == 1 else
               f", so the valid non-identity indices are 1..{len(self.operations) - 1}"))
    return draw_symmetry_operation(self.molecule, self.definitions, self.operations[which],
                                   output=output, **kwargs)


#: The molecule-level drawing, by OPERATION index. `draw_symmetry` is kept as its original name
#: from the specification; `draw_operation` says what it draws.
TorsionSymmetry.draw_operation = _draw_operation
TorsionSymmetry.draw_symmetry = _draw_operation


#: The name used at the call site, matching `t_hdbscan` and `t_mi`.
t_symmetry = TorsionSymmetry


# -------------------------------------------------------------------- drawing a symmetry GROUP

def _molecule_png(molecule, highlight_atoms, size=(420, 360)) -> bytes:
    """A PNG of the molecule with `highlight_atoms` marked, for embedding in a figure."""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = _drawing_molecule(molecule, keep_hydrogens=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                       highlightAtoms=[int(i) for i in highlight_atoms])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _circular_mean(theta: np.ndarray) -> float:
    """The mean direction. An arithmetic mean of angles is wrong across the seam: two frames at
    +179 and -179 average to 0, which is the opposite side of the circle."""
    return float(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()))


def _draw_sym_group(self, group, *, output, cluster_torsions=None, merged=None,
                    bins: int = 48, max_points: int = 4000) -> Dict[str, Any]:
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
                f"MERGED label, not an operation index -- draw_symmetry() takes the latter.")
        members = list(self.cluster_groups[merged_label])

    if cluster_torsions is None:
        raise ValueError("pass cluster_torsions= so the member angles can be drawn")

    names = list(self.definitions.names)
    n_torsions = len(names)
    pairs = [(i, j) for i in range(n_torsions) for j in range(i + 1, n_torsions)]
    reference = members[0]

    # the angles, before and after the operation that relates each member to the reference
    before, after, used = {}, {}, {}
    for member in members:
        entry = cluster_torsions.entries[member]
        before[member] = entry.torsions
        operation_index = self.group_operations.get(member)
        used[member] = operation_index
        after[member] = (entry.torsions if operation_index in (None,) or member == reference
                         else self.transform(entry.torsions, self.operations[operation_index],
                                             frames=entry.frames))

    moved = []
    for member in members:
        index = used[member]
        if index:
            moved = [i for i, j in enumerate(self.operations[index].permutation) if i != j]
            break

    n_cols = max(len(pairs), 1)
    figure, axes = plt.subplots(3, n_cols, figsize=(4.6 * n_cols, 12.4), squeeze=False)

    image = mpimg.imread(io.BytesIO(_molecule_png(self.molecule, moved)), format="png")
    axes[0][0].imshow(image)
    axes[0][0].axis("off")
    axes[0][0].set_title(
        f"group {merged_label}: original clusters {members}\n"
        + (f"related by operation {used[members[-1]]}, {len(moved)} atoms move"
           if moved else "single cluster, no operation applied"), fontsize=10)
    for column in range(1, n_cols):
        axes[0][column].axis("off")

    colours = plt.get_cmap("tab10")
    for column, (i, j) in enumerate(pairs or [(0, 0)]):
        for row, (data, label) in enumerate(((before, "as measured"), (after, "symmetry-aligned")),
                                            start=1):
            axis = axes[row][column]
            for k, member in enumerate(members):
                values = data[member]
                step = max(1, values.shape[0] // max_points)
                axis.scatter(values[::step, i], values[::step, j], s=3, alpha=0.35,
                             color=colours(k % 10), label=f"cluster {member}")
            axis.set_xlabel(f"{names[i]} (rad)")
            axis.set_ylabel(f"{names[j]} (rad)")
            axis.set_xlim(-np.pi, np.pi)
            axis.set_ylim(-np.pi, np.pi)
            axis.set_title(f"{names[i]} vs {names[j]} -- {label}", fontsize=10)
            if column == 0 and row == 1:
                axis.legend(markerscale=4, fontsize=8, loc="upper right")

    # which torsions move together, over the merged group
    correlations = []
    if n_torsions > 1:
        from ._t_mi import torsional_mi

        stacked = np.vstack([after[m] for m in members])
        try:
            report = torsional_mi(stacked, names=names, bins=min(24, bins), n_null=6)
            for pair in report["pairs"]:
                correlations.append({
                    "a": pair["a"], "b": pair["b"], "mi_nats": pair["mi_nats"],
                    "mi_nats_debiased": pair["mi_nats_debiased"],
                    "excess_over_null_in_sd": pair.get("excess_over_null_in_sd")})
        except Exception as error:                       # pragma: no cover - diagnostic only
            correlations.append({"error": f"{type(error).__name__}: {error}"})
    for column, (i, j) in enumerate(pairs or []):
        entry = next((c for c in correlations
                      if c.get("a") == names[i] and c.get("b") == names[j]), None)
        if entry and entry.get("mi_nats_debiased") is not None:
            axes[2][column].annotate(
                f"MI {entry['mi_nats_debiased']:+.4f} nats "
                f"({entry['excess_over_null_in_sd']:+.1f} sd over null)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    figure.suptitle(
        "An MI excess near zero means these torsions move INDEPENDENTLY within the group; "
        "it is not evidence that the merge was wrong.", fontsize=9, y=0.005)
    figure.tight_layout()
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)

    return {"output": str(output), "merged_label": merged_label, "members": members,
            "operations_used": {int(k): v for k, v in used.items()},
            "moved_atoms": list(moved),
            "circular_means": {int(m): {names[c]: _circular_mean(before[m][:, c])
                                        for c in range(n_torsions)} for m in members},
            "circular_means_aligned": {int(m): {names[c]: _circular_mean(after[m][:, c])
                                                for c in range(n_torsions)} for m in members},
            "correlations": correlations}


TorsionSymmetry.draw_sym_group = _draw_sym_group
