"""Which atoms are scaled, and which torsions are left alone — as a file that can be checked.

The REST2 Hamiltonian is not determined by tau alone. It also depends on *which* atoms count as
solute and *which* torsions are excluded from scaling, and those are decisions taken by a
classifier over a particular topology. Two runs at the same tau over the same PDB can be different
Hamiltonians if the selection differed, and nothing in a trajectory records that.

So the selection is written to `solute.yaml` beside the run, with the digest of the topology it
was derived against, and it is *validated* rather than trusted when supplied.

**Exclusions are stored as central bonds, not as torsion indices.** A peptide omega is not one
term: several torsions share the same C–N bond, and a force field may enumerate them differently
from one build to the next. Naming the bond excludes every torsion across it, which is the
physical statement — "this rotation keeps its barrier" — rather than an accident of enumeration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["ScalingSelection", "SelectionError", "SELECTION_FORMAT"]

#: Versioned, because a reader that cannot tell which rules produced a file cannot use it.
SELECTION_FORMAT = "md-tools-solute-selection/1.0"


class SelectionError(ValueError):
    """A selection file that does not describe the topology it is being used with."""


def topology_digest(topology) -> str:
    """A digest of the atom and bond identity a selection is only valid against.

    Over element, name, residue and index -- the things an atom index MEANS -- rather than over the
    file's bytes, so the same chemistry written by a different writer still matches, and a
    renumbered or re-ordered topology does not.
    """
    hasher = hashlib.sha256()
    for atom in topology.atoms():
        element = atom.element.symbol if atom.element is not None else "?"
        hasher.update(f"{atom.index}:{atom.name}:{element}:"
                      f"{atom.residue.index}:{atom.residue.name}\n".encode())
    for bond in topology.bonds():
        first, second = sorted((bond.atom1.index, bond.atom2.index))
        hasher.update(f"bond:{first}-{second}\n".encode())
    return hasher.hexdigest()


@dataclass(frozen=True)
class ScalingSelection:
    """The resolved answer to "what is scaled here", loadable, writable and checkable."""

    solute_atoms: tuple[int, ...]
    excluded_bonds: tuple[tuple[int, int], ...] = ()
    topology_sha256: str | None = None
    labels: tuple[dict[str, Any], ...] = field(default=())
    detection: str | None = None

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
        from ..openmm.system import UnclassifiedOmegaError, omega_exclusions

        atoms = tuple(sorted({int(i) for i in solute_atoms}))
        try:
            classified = omega_exclusions(topology, atoms, ligand_sdf=ligand_sdf)
        except UnclassifiedOmegaError as refusal:
            raise SelectionError(str(refusal)) from None

        # ONLY `omega_unscaled_bonds`. A proline-like peptide bond stays eligible for ordinary
        # scaling -- its nitrogen carries no amide hydrogen, so the cis/trans argument that
        # protects an ordinary omega does not apply. An unclassified candidate never reaches
        # here: `omega_exclusions` refused it above. Both decisions are recorded below so the
        # file says what was decided rather than only what was excluded.
        bonds = tuple(sorted({tuple(sorted(int(i) for i in pair))
                              for pair in classified["omega_unscaled_bonds"]}))
        labels = []
        for entry in (classified.get("omega_detail") or {}).get("unscaled", []):
            labels.append({
                "bond": [int(entry["carbon"]), int(entry["nitrogen"])],
                "residues": f"{entry.get('carbon_residue')}-{entry.get('nitrogen_residue')}",
                "reason": "ordinary amide omega: left unscaled so a hot rung cannot isomerise it",
            })
        for pair in classified.get("omega_proline_like_scaled_bonds") or []:
            labels.append({"bond": [int(i) for i in pair], "residues": None,
                           "reason": "proline-like: SCALED, no amide hydrogen to protect"})
        return cls(solute_atoms=atoms, excluded_bonds=bonds,
                   topology_sha256=topology_digest(topology),
                   labels=tuple(labels), detection=str(classified["omega_detection_method"]))

    # -- persisting -----------------------------------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        return {
            "format": SELECTION_FORMAT,
            "topology_sha256": self.topology_sha256,
            "solute_atoms": list(self.solute_atoms),
            "unscaled_torsion_central_bonds": [list(b) for b in self.excluded_bonds],
            "labels": [dict(entry) for entry in self.labels],
            "detection_method": self.detection,
        }

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
        if document.get("format") != SELECTION_FORMAT:
            raise SelectionError(
                f"{path}: format is {document.get('format')!r}, not {SELECTION_FORMAT!r}. A "
                f"selection written by a different version of these rules is not interchangeable.")

        atoms = tuple(int(i) for i in document.get("solute_atoms") or ())
        bonds = tuple(tuple(sorted(int(i) for i in pair))
                      for pair in document.get("unscaled_torsion_central_bonds") or ())
        digest = document.get("topology_sha256")

        if len(set(atoms)) != len(atoms):
            raise SelectionError(f"{path}: solute_atoms contains duplicates")
        if len(set(bonds)) != len(bonds):
            raise SelectionError(f"{path}: unscaled_torsion_central_bonds contains duplicates")

        selection = cls(solute_atoms=tuple(sorted(atoms)), excluded_bonds=tuple(sorted(bonds)),
                        topology_sha256=digest,
                        labels=tuple(document.get("labels") or ()),
                        detection=document.get("detection_method"))
        if topology is not None:
            selection.validate_against(topology, source=str(path))
        return selection

    def validate_against(self, topology, *, source: str = "the selection") -> None:
        """Refuse a selection that cannot be true of this topology."""
        actual = topology_digest(topology)
        if self.topology_sha256 and self.topology_sha256 != actual:
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
        return {"solute_indices": list(self.solute_atoms),
                "excluded_bonds": [tuple(b) for b in self.excluded_bonds]}


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
