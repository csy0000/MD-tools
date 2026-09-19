"""Which atoms are scaled, and which torsions are left alone — as a file that can be checked.

The REST2 Hamiltonian is not determined by tau alone. It also depends on *which* atoms count as
solute and *which* torsions are excluded from scaling, and those are decisions taken by a
classifier over a particular topology. Two runs at the same tau over the same PDB can be different
Hamiltonians if the selection differed, and nothing in a trajectory records that.

So the selection is written to `solute.yaml` beside the run, with the digest of the topology it
was derived against, and it is *validated* rather than trusted when supplied.

**Exclusions are stored as central bonds, not as torsion indices.** An unscaled bond is not one
term: several torsions share the same C–N bond, and a force field may enumerate them differently
from one build to the next. Naming the bond excludes every torsion across it, which is the
physical statement — "this rotation keeps its barrier" — rather than an accident of enumeration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["ScalingSelection", "SelectionError", "SELECTION_FORMAT", "LEGACY_SELECTION_FORMAT",
           "LEGACY_MODE", "EXPLICIT_MODE"]

#: Versioned, because a reader that cannot tell which rules produced a file cannot use it.
#: 2.0 (0.6.1): the record says HOW the selection was made -- `selection_mode`, the masks as
#: written, the residue map, the nonbonded set, the selected torsion central bonds with their
#: owners, per-term CMAP decisions and the ligand instances with their exclusion files' contents --
#: because selective REST2 makes the how part of the Hamiltonian.
SELECTION_FORMAT = "md-tools-solute-selection/2.0"
#: What every file before 0.6.1 was. Still read, as the legacy full-solute selection it describes.
LEGACY_SELECTION_FORMAT = "md-tools-solute-selection/1.0"

LEGACY_MODE = "legacy-full-solute"
EXPLICIT_MODE = "explicit"


class SelectionError(ValueError):
    """A selection file that does not describe the topology it is being used with."""


#: How `topology_digest` is computed, recorded in every 2.0 record beside the digest.
TOPOLOGY_DIGEST_SCHEME = "atoms-in-index-order+sorted-bond-set/1"


def _hash_atoms(hasher, topology) -> None:
    for atom in topology.atoms():
        element = atom.element.symbol if atom.element is not None else "?"
        hasher.update(f"{atom.index}:{atom.name}:{element}:"
                      f"{atom.residue.index}:{atom.residue.name}\n".encode())


def topology_digest(topology) -> str:
    """A digest of the atom and bond identity a selection is only valid against
    (`TOPOLOGY_DIGEST_SCHEME`).

    Over element, name, residue and index -- the things an atom index MEANS -- rather than over the
    file's bytes, so the same chemistry written by a different writer still matches, and a
    renumbered or re-ordered topology does not.

    The bonds are hashed as a SORTED SET of sorted pairs. They used to be hashed in the order the
    Topology iterates them, and that order comes from the file: OpenMM's PDB writer emits CONECT
    records for a non-standard residue in its own order, so a read-write-read cycle of a structure
    with a cross-residue CONECT bond made the digest alternate between two values forever
    (shared contract §2, 2026-09-19). The bond SET is the identity; its order is an accident.
    """
    hasher = hashlib.sha256()
    _hash_atoms(hasher, topology)
    for first, second in sorted(tuple(sorted((bond.atom1.index, bond.atom2.index)))
                                for bond in topology.bonds()):
        hasher.update(f"bond:{first}-{second}\n".encode())
    return hasher.hexdigest()


def _legacy_iteration_order_digest(topology) -> str:
    """The 1.0 digest, bonds in iteration order. Used ONLY by `_legacy_digest_agrees`, so a
    0.6.0 record keeps validating; nothing new is ever written with it."""
    hasher = hashlib.sha256()
    _hash_atoms(hasher, topology)
    for bond in topology.bonds():
        first, second = sorted((bond.atom1.index, bond.atom2.index))
        hasher.update(f"bond:{first}-{second}\n".encode())
    return hasher.hexdigest()


def _legacy_digest_agrees(stored: str, topology) -> bool:
    """THE COMPATIBILITY BRANCH for a 1.0 record's digest (shared contract §2): it validates when
    it equals EITHER the canonical digest of this topology or the legacy iteration-order one."""
    return stored in (topology_digest(topology), _legacy_iteration_order_digest(topology))


@dataclass(frozen=True)
class ScalingSelection:
    """The resolved answer to "what is scaled here", loadable, writable and checkable."""

    #: The NONBONDED hot set. Legacy: every solute atom. Explicit: `selected_nonbonded_atoms`.
    solute_atoms: tuple[int, ...]
    excluded_bonds: tuple[tuple[int, int], ...] = ()
    topology_sha256: str | None = None
    labels: tuple[dict[str, Any], ...] = field(default=())
    detection: str | None = None
    mode: str = LEGACY_MODE
    #: Explicit mode only: the scaled torsion central bonds, and the scaled CMAP term indices.
    #: None in legacy mode, where both follow the whole-solute rule.
    torsion_bonds: tuple[tuple[int, int], ...] | None = None
    cmap_terms: tuple[int, ...] | None = None
    unscaled_impropers: bool = True
    detector_version: int | None = None
    #: The rest of the 2.0 record (policy, masks, residue map, owners, CMAP decisions, ligand
    #: instances), kept verbatim: it is provenance, and part of what the identity hashes.
    details: tuple[tuple[str, Any], ...] = field(default=())
    #: The format of the record this was READ from: 1.0 digests may use the legacy scheme. Not
    #: part of what the selection IS, so it takes no part in equality.
    record_format: str = field(default=SELECTION_FORMAT, compare=False)

    @property
    def explicit(self) -> bool:
        return self.mode == EXPLICIT_MODE

    # -- deriving -------------------------------------------------------------------------------

    @classmethod
    def derive(cls, topology, solute_atoms: Iterable[int],
               ligand_sdf=None) -> "ScalingSelection":
        """Use the validated classifier to decide the exclusions for this topology.

        *ligand_sdf* is the SDF `build-top` retains beside the System for a molecular input. It is
        optional because a peptide has none and needs none: the classifier chooses its evidence
        per candidate from the residue name, and only reaches for bond orders when a residue
        cannot answer. Without it, a non-standard residue is refused rather than guessed.
        """
        from ..openmm.system import UnclassifiedTorsionError, unscaled_torsions

        atoms = tuple(sorted({int(i) for i in solute_atoms}))
        try:
            classified = unscaled_torsions(topology, atoms, ligand_sdf=ligand_sdf)
        except UnclassifiedTorsionError as refusal:
            raise SelectionError(str(refusal)) from None

        # ONLY the unscaled central bonds. A proline-like peptide bond stays eligible for ordinary
        # scaling -- its nitrogen carries no amide hydrogen, so the cis/trans argument that
        # protects an ordinary omega does not apply. An unclassified candidate never reaches
        # here: `unscaled_torsions` refused it above. Both decisions are recorded below so the
        # file says what was decided rather than only what was excluded.
        bonds = tuple(sorted({tuple(sorted(int(i) for i in pair))
                              for pair in classified["unscaled_central_bonds"]}))
        labels = []
        reasons = {"amide_omega": "ordinary amide omega: left unscaled so a hot rung cannot "
                                  "isomerise it",
                   "aromatic_ring": "aromatic ring bond: left unscaled so a hot rung cannot "
                                    "pucker the ring",
                   "double_bond": "double bond: left unscaled so a hot rung cannot twist it"}
        for entry in classified.get("central_bonds") or []:
            labels.append({
                "bond": [int(i) for i in entry["bond"]],
                "residues": f"{entry.get('residue')}{entry.get('residue_index')}",
                "reason": reasons[entry["class"]],
            })
        for pair in classified.get("proline_like_scaled_bonds") or []:
            labels.append({"bond": [int(i) for i in pair], "residues": None,
                           "reason": "proline-like: SCALED, no amide hydrogen to protect"})
        return cls(solute_atoms=atoms, excluded_bonds=bonds,
                   topology_sha256=topology_digest(topology),
                   labels=tuple(labels), detection=str(classified["detection_method"]),
                   unscaled_impropers=bool(classified.get("unscaled_impropers", True)),
                   detector_version=classified.get("detector_version"))

    # -- persisting -----------------------------------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        details = dict(self.details)
        document = {
            "format": SELECTION_FORMAT,
            "selection_mode": self.mode,
            "topology_sha256": self.topology_sha256,
            "topology_digest_scheme": TOPOLOGY_DIGEST_SCHEME,
            # 1.0's name, kept: the atoms carrying the nonbonded factors.
            "solute_atoms": list(self.solute_atoms),
            "selected_nonbonded_atoms": list(self.solute_atoms),
            "selected_torsion_central_bonds": (None if self.torsion_bonds is None
                                               else [list(b) for b in self.torsion_bonds]),
            "scaled_cmap_terms": None if self.cmap_terms is None else list(self.cmap_terms),
            "unscaled_torsion_central_bonds": [list(b) for b in self.excluded_bonds],
            "excluded_central_bonds": [list(b) for b in self.excluded_bonds],
            "improper_policy": {"unscaled_impropers": bool(self.unscaled_impropers)},
            "detector_policy_version": self.detector_version,
            "labels": [dict(entry) for entry in self.labels],
            "detection_method": self.detection,
        }
        for key in ("policy", "masks", "residue_map", "torsion_bond_owners", "cmap_decisions",
                    "ligand_instances", "partially_owned_central_bonds", "notes"):
            document[key] = details.get(key)
        return document

    def digest(self) -> str:
        """sha256 of the Hamiltonian-determining projection: what the identity hashes. Two
        selections differing only in provenance (mask spelling, labels, paths) share it."""
        from .identity import selection_identity_sha256

        return selection_identity_sha256(self.to_document())

    def provenance_digest(self) -> str:
        """sha256 over the WHOLE 2.0 document, provenance included. Never an identity."""
        import json

        return hashlib.sha256(json.dumps(self.to_document(), sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def from_document(cls, document: dict[str, Any], *, source: str = "the selection"
                      ) -> "ScalingSelection":
        """The inverse of `to_document`, for a 2.0 document; a 1.0 one is read as legacy."""
        fmt = document.get("format")
        if fmt not in (SELECTION_FORMAT, LEGACY_SELECTION_FORMAT):
            raise SelectionError(
                f"{source}: format is {fmt!r}, not {SELECTION_FORMAT!r}. A selection written by "
                f"a different version of these rules is not interchangeable.")
        atoms = tuple(int(i) for i in document.get("solute_atoms") or ())
        bonds = tuple(tuple(sorted(int(i) for i in pair))
                      for pair in document.get("unscaled_torsion_central_bonds") or ())
        if len(set(atoms)) != len(atoms):
            raise SelectionError(f"{source}: solute_atoms contains duplicates")
        if len(set(bonds)) != len(bonds):
            raise SelectionError(f"{source}: unscaled_torsion_central_bonds contains duplicates")
        mode = document.get("selection_mode", LEGACY_MODE) if fmt == SELECTION_FORMAT \
            else LEGACY_MODE
        if mode not in (LEGACY_MODE, EXPLICIT_MODE):
            raise SelectionError(f"{source}: selection_mode {mode!r} is neither "
                                 f"{LEGACY_MODE!r} nor {EXPLICIT_MODE!r}")
        torsion = document.get("selected_torsion_central_bonds")
        cmap = document.get("scaled_cmap_terms")
        if (mode == EXPLICIT_MODE) != (torsion is not None and cmap is not None):
            raise SelectionError(
                f"{source}: selection_mode is {mode}, but selected_torsion_central_bonds and "
                f"scaled_cmap_terms are {'absent' if mode == EXPLICIT_MODE else 'present'}. An "
                f"explicit selection names both; a legacy one names neither.")
        if mode == EXPLICIT_MODE and set(atoms) != set(
                int(i) for i in document.get("selected_nonbonded_atoms") or ()):
            raise SelectionError(f"{source}: solute_atoms and selected_nonbonded_atoms disagree")
        if fmt == SELECTION_FORMAT and document.get("topology_digest_scheme") != \
                TOPOLOGY_DIGEST_SCHEME:
            raise SelectionError(
                f"{source}: topology_digest_scheme is {document.get('topology_digest_scheme')!r}, "
                f"not {TOPOLOGY_DIGEST_SCHEME!r}. A {SELECTION_FORMAT} record carries only the "
                f"canonical digest.")
        policy = document.get("improper_policy") or {}
        details = tuple((key, document.get(key)) for key in (
            "policy", "masks", "residue_map", "torsion_bond_owners", "cmap_decisions",
            "ligand_instances", "partially_owned_central_bonds", "notes"))
        return cls(solute_atoms=tuple(sorted(atoms)), excluded_bonds=tuple(sorted(bonds)),
                   topology_sha256=document.get("topology_sha256"),
                   labels=tuple(document.get("labels") or ()),
                   detection=document.get("detection_method"), mode=mode,
                   torsion_bonds=(None if torsion is None else tuple(
                       tuple(sorted(int(i) for i in pair)) for pair in torsion)),
                   cmap_terms=None if cmap is None else tuple(int(i) for i in cmap),
                   unscaled_impropers=bool(policy.get("unscaled_impropers", True)),
                   detector_version=document.get("detector_policy_version"),
                   details=details if fmt == SELECTION_FORMAT else (), record_format=fmt)

    def write(self, path: str | Path) -> Path:
        from ..openmm.yaml_io import write_yaml

        return write_yaml(Path(path), self.to_document(), header=(
            "# Which atoms REST2 scales, and which torsions keep their physical barrier.\n"
            "# Written by md-tools. Exclusions are CENTRAL BONDS: every torsion across the bond\n"
            "# is excluded, so a force field that enumerates torsions differently cannot change\n"
            "# which rotations are softened.\n"))

    # -- loading, with the checks that make a supplied file usable ------------------------------

    @classmethod
    def load(cls, path: str | Path, topology=None) -> "ScalingSelection":
        """Read a selection file and refuse one that does not describe this topology.

        Every check here exists because its absence would produce a *plausible* run at the wrong
        Hamiltonian rather than an error.
        """
        import yaml

        path = Path(path)
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        selection = cls.from_document(document, source=str(path))
        if topology is not None:
            selection.validate_against(topology, source=str(path))
        return selection

    def validate_against(self, topology, *, source: str = "the selection") -> None:
        """Refuse a selection that cannot be true of this topology."""
        actual = topology_digest(topology)
        if self.topology_sha256 and self.record_format == LEGACY_SELECTION_FORMAT:
            agrees = _legacy_digest_agrees(self.topology_sha256, topology)
        else:
            agrees = self.topology_sha256 == actual
            if not agrees and self.topology_sha256 == _legacy_iteration_order_digest(topology):
                raise SelectionError(
                    f"{source}: a {SELECTION_FORMAT} record carries the LEGACY iteration-order "
                    f"topology digest ({self.topology_sha256[:12]}...). {SELECTION_FORMAT} records "
                    f"carry only the canonical digest ({TOPOLOGY_DIGEST_SCHEME}); this one was not "
                    f"written by the code that defines that format.")
        if self.topology_sha256 and not agrees:
            raise SelectionError(
                f"{source}: topology_sha256 {self.topology_sha256[:12]}... does not match the "
                f"topology being used ({actual[:12]}...). The selection was derived against a "
                f"different structure, so its atom indices mean something else here.")

        n_atoms = topology.getNumAtoms()
        out_of_range = [i for i in self.solute_atoms if not 0 <= i < n_atoms]
        if out_of_range:
            raise SelectionError(
                f"{source}: solute atom indices outside 0..{n_atoms - 1}: {out_of_range[:8]}")

        present = {tuple(sorted((b.atom1.index, b.atom2.index))) for b in topology.bonds()}
        missing = [b for b in self.excluded_bonds if b not in present]
        if missing:
            raise SelectionError(
                f"{source}: these excluded central bonds do not exist in this topology: "
                f"{missing[:8]}. An exclusion that names no bond would silently scale a torsion "
                f"the file says is protected.")

        if self.explicit:
            missing = [b for b in (self.torsion_bonds or ()) if b not in present]
            if missing:
                raise SelectionError(
                    f"{source}: these selected central bonds do not exist in this topology: "
                    f"{missing[:8]}")
            overlap = sorted(set(self.torsion_bonds or ()) & set(self.excluded_bonds))
            if overlap:
                raise SelectionError(
                    f"{source}: central bonds both selected for scaling and protected: "
                    f"{overlap[:8]}. A protected bond is never scaled; the record contradicts "
                    f"itself.")
            return
        solute = set(self.solute_atoms)
        outside = [b for b in self.excluded_bonds if not (b[0] in solute and b[1] in solute)]
        if outside:
            raise SelectionError(
                f"{source}: these excluded bonds are not entirely inside the solute: "
                f"{outside[:8]}. Only solute torsions are scaled, so excluding an environment "
                f"bond protects nothing and means the selection was built for another system.")

    def check_matches_torsions(self, system, *, source: str = "the selection") -> None:
        """Refuse an exclusion that matches no torsion in the System.

        Silently ignoring it is the failure: the file would claim a rotation keeps its barrier,
        the run would soften it anyway, and nothing downstream could tell.
        """
        from openmm import PeriodicTorsionForce

        if not self.excluded_bonds:
            return
        central = set()
        for force in (system.getForce(i) for i in range(system.getNumForces())):
            if not isinstance(force, PeriodicTorsionForce):
                continue
            for index in range(force.getNumTorsions()):
                _, second, third, *_ = force.getTorsionParameters(index)
                central.add(tuple(sorted((second, third))))
        unmatched = [b for b in self.excluded_bonds if b not in central]
        if unmatched:
            raise SelectionError(
                f"{source}: these excluded central bonds match no torsion in the System: "
                f"{unmatched[:8]}. The exclusion would do nothing, and the file would say "
                f"otherwise.")

    # -- what the scaler consumes ---------------------------------------------------------------

    def as_scaler_arguments(self) -> dict[str, Any]:
        """Exactly the keyword arguments `hamiltonian.build_scaled_system` takes besides tau."""
        return {"solute_indices": list(self.solute_atoms),
                "excluded_bonds": [tuple(b) for b in self.excluded_bonds],
                "unscaled_impropers": bool(self.unscaled_impropers),
                "torsion_central_bonds": (None if self.torsion_bonds is None
                                          else [tuple(b) for b in self.torsion_bonds]),
                "cmap_terms": None if self.cmap_terms is None else list(self.cmap_terms)}


def resolve_selection(path: str | Path, topology, solute_atoms: Sequence[int],
                      system=None) -> ScalingSelection:
    """Load the selection beside a run, or derive and write it if there is none.

    The resolved file is always written, so a run that started from defaults still records the
    Hamiltonian it actually used.
    """
    path = Path(path)
    if path.is_file():
        selection = ScalingSelection.load(path, topology=topology)
    else:
        selection = ScalingSelection.derive(topology, solute_atoms)
        selection.write(path)
    if system is not None:
        selection.check_matches_torsions(system, source=str(path))
    return selection
