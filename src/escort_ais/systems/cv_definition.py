"""THE collective-variable definition for a system: declared once, in JSON, read everywhere.

Every system carries ``systems/<slug>/cv_definition.json`` listing its CVs as named atom quartets.
Simulation writers and analysis both load it, so the CV set is declared in exactly one place and is
never re-derived downstream.

Why this exists. The chunked AIS writer recorded its CVs by calling ``mdtraj.compute_phi/psi``.
For ACE-ALA-NME that works. For a cyclic sage/openff macrocycle mdtraj finds **no** phi/psi at all,
so the writer stored NaN -- silently. ``data/rgd/ess50_5ps`` carries 183 000 Hummer-Szabo rows whose
``phi``/``psi`` columns are entirely NaN, and because the intermediate-slice coordinates are not
kept either, that multi-slice cloud can never be reconstructed. A whole partition analysis was run
on 1-frame-per-trajectory endpoints as a result.

The CV set by system class:

* **dipeptide** -- (phi, psi) of the single residue.
* **macrocycle / peptide** -- every NON-omega backbone torsion. Omega torsions are excluded because
  they are the cis/trans amide coordinate, handled separately by the REST2 omega exclusion; the
  partition CVs are the flexible backbone dihedrals.

Angles are returned in RADIANS, wrapped to (-pi, pi], in the file's declared order. That order is
part of the definition: cell arcs are indexed by CV position, so reordering a definition silently
reinterprets every stored partition.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2
#: Versions this module can read. v1 predates the provenance fields and is upgraded on load
#: with the provenance left blank -- an unverifiable definition, not an invalid one.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

#: Only torsions today. Distances and bond angles are NOT supported and must not be added by
#: widening this: the partition machinery is periodic by construction -- rectangular_flatbottom
#: uses ``acos(clamp(cos(theta-mu)))`` and PeriodicInterval wraps at +/-pi -- so a distance
#: (unbounded) or a bond angle (bounded, reflecting at 0 and pi) fed through it would produce a
#: plausible-looking partition that is silently wrong, exactly like the radians/degrees mix-up
#: that once gave pi_m = 0.71/0.13/0.16 against a true 0.97/0.007/0.027. Supporting them means
#: giving each CV an explicit geometry and teaching the walls a linear branch.
CV_ARITY = 4


@dataclass(frozen=True)
class CVDefinition:
    """Named atom quartets defining a system's collective variables."""

    slug: str
    kind: str                      # "dipeptide" | "macrocycle"
    names: tuple[str, ...]
    quartets: np.ndarray           # (n_cv, 4) int
    source: str = ""
    # ── provenance (schema v2) ─────────────────────────────────────────────────────────────
    # Atom INDICES are topology-order dependent: rebuild the prmtop with a different tleap or
    # parmed and the indices shift, silently reinterpreting every partition ever stored against
    # this definition. These two fields make that detectable instead of silent.
    topology_sha256: str = ""      # checksum of the topology the indices were read from
    atom_labels: tuple = ()        # (n_cv, 4) of "RES<resSeq>-<atomname>", order-independent

    def __len__(self) -> int:
        return len(self.names)

    def compute(self, traj) -> np.ndarray:
        """(n_frames, n_cv) torsions in RADIANS, in the declared order."""
        import mdtraj as md  # noqa: PLC0415

        ang = md.compute_dihedrals(traj, self.quartets)
        return (np.asarray(ang, float) + np.pi) % (2 * np.pi) - np.pi

    def to_dict(self) -> dict:
        return dict(schema_version=SCHEMA_VERSION, slug=self.slug, kind=self.kind,
                    n_cv=len(self), names=list(self.names),
                    quartets=[[int(x) for x in q] for q in self.quartets], source=self.source,
                    topology_sha256=self.topology_sha256,
                    atom_labels=[list(q) for q in self.atom_labels])

    def column_names(self) -> list[str]:
        """Parquet/CSV column names for the CV values -- ``cv0_phi`` style, position-prefixed.

        The position prefix is deliberate: it makes a reordered definition a visible schema change
        rather than a silent reinterpretation.
        """
        return ["cv%d_%s" % (i, n) for i, n in enumerate(self.names)]


def cv_definition_path(slug: str, root: Optional[Path] = None) -> Path:
    from escort_ais.common.paths import project_root  # noqa: PLC0415

    return (root or project_root()) / "systems" / slug / "cv_definition.json"


def load_cv_definition(slug_or_path, root: Optional[Path] = None) -> CVDefinition:
    """Load a CV definition by system slug or explicit path."""
    p = Path(slug_or_path)
    if not p.suffix == ".json":
        p = cv_definition_path(str(slug_or_path), root)
    if not p.exists():
        raise FileNotFoundError(
            f"no CV definition at {p}. Every macrocycle/peptide must declare its CVs in "
            f"systems/<slug>/cv_definition.json -- deriving them downstream is what produced the "
            f"all-NaN phi/psi columns in the RGD Hummer-Szabo clouds.")
    d = json.loads(p.read_text())
    ver = int(d.get("schema_version", 0))
    if ver not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"{p}: schema_version {ver} not in {SUPPORTED_SCHEMA_VERSIONS}")
    q = np.asarray(d["quartets"], int)
    if q.ndim != 2 or q.shape[1] != CV_ARITY:
        raise ValueError(
            f"{p}: quartets must be (n_cv, {CV_ARITY}); got {q.shape}. Only torsions are "
            f"supported -- see CV_ARITY.")
    if len(d["names"]) != len(q):
        raise ValueError(f"{p}: {len(d['names'])} names for {len(q)} quartets")
    return CVDefinition(slug=d["slug"], kind=d["kind"], names=tuple(d["names"]),
                        quartets=q, source=d.get("source", ""),
                        topology_sha256=d.get("topology_sha256", ""),
                        atom_labels=tuple(tuple(x) for x in d.get("atom_labels", ())))


def topology_sha256(path) -> str:
    """Checksum of the topology file the atom indices refer to."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atom_labels_for(topology_pdb, quartets) -> tuple:
    """``(n_cv, 4)`` of ``"<resName><resSeq>-<atomName>"`` for a human-readable cross-check.

    Indices alone cannot tell you whether a definition still means what it meant: an atom-order
    change keeps them valid-looking. Labels survive reordering and say what went wrong.
    """
    import mdtraj as md  # noqa: PLC0415

    top = md.load(str(topology_pdb)).topology
    out = []
    for q in np.asarray(quartets, int):
        row = []
        for i in q:
            a = top.atom(int(i))
            row.append(f"{a.residue.name}{a.residue.resSeq}-{a.name}")
        out.append(tuple(row))
    return tuple(out)


def verify_against_topology(cv: CVDefinition, topology_pdb, strict: bool = True) -> dict:
    """Check a definition still describes ``topology_pdb``. Returns a report; raises when strict.

    Two independent checks, because they fail differently:

    * ``topology_sha256`` -- the file is byte-identical to the one the indices were read from.
      A mismatch is not automatically wrong (a re-solvated box, a re-minimised structure), so on
      its own it is a WARNING.
    * ``atom_labels`` -- the indices still point at the same named atoms. A mismatch here IS an
      error: the definition now selects different atoms than it did, and every partition stored
      against it silently means something else.
    """
    report = {"slug": cv.slug, "topology": str(topology_pdb),
              "sha_recorded": cv.topology_sha256, "sha_actual": topology_sha256(topology_pdb),
              "sha_match": None, "labels_match": None, "label_diff": []}
    report["sha_match"] = bool(cv.topology_sha256) and report["sha_recorded"] == report["sha_actual"]

    if cv.atom_labels:
        actual = atom_labels_for(topology_pdb, cv.quartets)
        diff = [f"{cv.names[i]}: recorded {list(rec)} but topology has {list(act)}"
                for i, (rec, act) in enumerate(zip(cv.atom_labels, actual)) if tuple(rec) != act]
        report["labels_match"] = not diff
        report["label_diff"] = diff
        if diff and strict:
            raise ValueError(
                f"cv_definition for '{cv.slug}' no longer matches {topology_pdb}: the atom "
                f"indices select different atoms than when it was written.\n  "
                + "\n  ".join(diff[:10])
                + "\nEvery partition stored against this definition means something else now.")
    return report


def write_cv_definition(cv: CVDefinition, root: Optional[Path] = None) -> Path:
    p = cv_definition_path(cv.slug, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cv.to_dict(), indent=2) + "\n")
    return p


def from_backbone_quartets(slug: str, npz_path, source: str = "") -> CVDefinition:
    """Macrocycle CVs: the NON-omega backbone torsions from a stored quartet file.

    The ``is_omega`` mask is required, not optional -- including the omega torsions would mix the
    cis/trans amide coordinate into the partition CVs, and the REST2 ladder already excludes those
    bonds from scaling.
    """
    z = np.load(npz_path)
    if "quartets" not in z or "is_omega" not in z:
        raise KeyError(f"{npz_path} must contain 'quartets' and 'is_omega'")
    keep = ~z["is_omega"].astype(bool)
    q = np.asarray(z["quartets"], int)[keep]
    names = tuple("t%d" % i for i in range(len(q)))
    return CVDefinition(slug=slug, kind="macrocycle", names=names, quartets=q,
                        source=source or str(npz_path))


def from_dipeptide(slug: str, pdb_path, source: str = "") -> CVDefinition:
    """Dipeptide CVs: (phi, psi) of the single residue, via mdtraj."""
    import mdtraj as md  # noqa: PLC0415

    t = md.load(str(pdb_path))
    qp, _ = md.compute_phi(t)
    qs, _ = md.compute_psi(t)
    if len(qp) != 1 or len(qs) != 1:
        raise ValueError(f"{pdb_path}: expected exactly one phi and one psi, "
                         f"got {len(qp)} and {len(qs)}")
    q = np.vstack([np.asarray(qp[0], int), np.asarray(qs[0], int)])
    return CVDefinition(slug=slug, kind="dipeptide", names=("phi", "psi"), quartets=q,
                        source=source or str(pdb_path))


def with_provenance(cv: CVDefinition, topology_pdb) -> CVDefinition:
    """Attach the topology checksum and atom labels to a definition."""
    return replace(cv, topology_sha256=topology_sha256(topology_pdb),
                   atom_labels=atom_labels_for(topology_pdb, cv.quartets))


def ensure_cv_definition(slug: str, kind: str, topology_pdb, *,
                         quartet_npz=None, root: Optional[Path] = None,
                         verify: bool = True) -> CVDefinition:
    """Generate ``systems/<slug>/cv_definition.json`` if absent; verify it if present.

    Called at system-construction time so that declaring the CVs is not a manual step that can
    be forgotten -- forgetting it is what let the chunked writer fall back to
    ``mdtraj.compute_phi/psi`` and store 183 000 all-NaN rows for a cyclic macrocycle.

    ORDER IS NOT RE-SORTED. Cell arcs are indexed by CV position, so imposing a canonical sort
    here would silently reinterpret every partition already stored against an existing
    definition. Order is taken from the source (the npz for a macrocycle, phi-then-psi for a
    dipeptide) and its stability is *verified* through ``atom_labels`` rather than enforced by
    re-sorting.

    An existing definition is never rewritten -- it is checked and left alone. A definition that
    no longer matches its topology is an error to be resolved deliberately, not by regeneration.
    """
    path = cv_definition_path(slug, root)
    if path.exists():
        cv = load_cv_definition(path)
        if verify:
            verify_against_topology(cv, topology_pdb, strict=True)
        return cv

    if kind == "dipeptide":
        cv = from_dipeptide(slug, topology_pdb)
    elif kind == "macrocycle":
        if quartet_npz is None:
            raise ValueError(
                f"'{slug}' is a macrocycle: pass quartet_npz (the stored backbone torsion "
                f"quartets). mdtraj finds no phi/psi on a cyclic topology, so they cannot be "
                f"derived here.")
        cv = from_backbone_quartets(slug, quartet_npz)
    else:
        raise ValueError(f"unknown CV kind {kind!r}; expected 'dipeptide' or 'macrocycle'")

    cv = with_provenance(cv, topology_pdb)
    write_cv_definition(cv, root)
    return cv


def cv_frame(traj, cv: CVDefinition, degrees: bool = True):
    """A DataFrame of CV values with the definition's column names -- for parquet/CSV writers."""
    import pandas as pd  # noqa: PLC0415

    a = cv.compute(traj)
    if degrees:
        a = np.degrees(a)
    return pd.DataFrame({c: a[:, i] for i, c in enumerate(cv.column_names())})


def assert_finite(values, cv: CVDefinition, where: str = "") -> None:
    """Fail loudly on all-NaN CVs.

    The RGD Hummer-Szabo clouds stored 183 000 rows of NaN because nothing checked. A writer that
    cannot compute its CVs must stop, not record nothing.
    """
    a = np.asarray(values, float)
    if a.size and not np.isfinite(a).any():
        raise ValueError(
            f"all {a.size} CV values are non-finite{' at ' + where if where else ''} for system "
            f"'{cv.slug}'. The quartets in its cv_definition.json do not match this topology.")
