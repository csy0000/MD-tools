"""Which residues line a ligand: printed for a person to read, paste and keep.

    python -m md_tools.rest2.pocket build/built.pdb --ligand :13 --cutoff-nm 0.5

It PRINTS a mask and a table. It never resolves anything at run time, and the configuration still
carries the explicit mask the user pasted, so the record says exactly which residues were hot
rather than "whatever was within 5 A of something on the day". That is the same reason
`build-top --rest2-scaler` resolves the region once and writes it down.

THE CRITERION, stated because a distance is meaningless without one (`POCKET_CRITERION`):

* **Heavy atoms only**, on both sides. Hydrogen positions come from the build and a hydrogen makes
  a residue look 0.1 nm closer than its chemistry is.
* **Minimum distance** over all heavy-atom pairs, ligand to residue: the closest approach, not a
  centroid distance, which would miss a long residue reaching into the site.
* **Strictly less than the cutoff.** A residue exactly at the cutoff is OUT. Floating-point
  equality is not a decision anybody should make, so the boundary is a strict one and it is
  written here; the table additionally lists what lies just outside, with its distance, so a
  borderline residue is visible rather than silently dropped.
* **Minimum image** when the topology carries a box, so a residue that the builder wrapped to the
  far side is measured where it is, not where its coordinates happen to sit.
* **Solvent and ions are excluded** by default: the water within 5 A of a ligand is not a pocket.

The mask is over ONE-BASED TOPOLOGY RESIDUE INDICES, the same numbering the selectors use, and it
stays meaningful under any renumbering only by being resolved against the topology it was printed
from -- which is why the printed header names the structure and its digest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = ["POCKET_CRITERION", "DEFAULT_CUTOFF_NM", "DEFAULT_MARGIN_NM", "PocketError",
           "pocket_residues", "residue_mask", "pocket_report", "main"]

#: Name and version of the criterion, printed with every report.
POCKET_CRITERION = "md-tools-pocket-selection/1"
#: 5 A: the conventional first-shell cutoff for "lining the site".
DEFAULT_CUTOFF_NM = 0.5
#: How far beyond the cutoff the report still lists a residue, marked as NOT selected.
DEFAULT_MARGIN_NM = 0.05


class PocketError(ValueError):
    """A ligand that cannot be named, or a structure without the coordinates to measure."""


def _heavy(residue) -> list:
    return [atom for atom in residue.atoms()
            if atom.element is not None and atom.element.symbol != "H"]


def _minimum_image(deltas, box):
    """Wrap displacements into the nearest image of a (possibly triclinic) box."""
    import numpy as np

    if box is None:
        return deltas
    box = np.asarray(box, dtype=float)
    for axis in (2, 1, 0):
        deltas = deltas - np.outer(np.round(deltas[:, axis] / box[axis][axis]), box[axis])
    return deltas


def pocket_residues(topology, positions, *, ligand, cutoff_nm: float = DEFAULT_CUTOFF_NM,
                    margin_nm: float = DEFAULT_MARGIN_NM,
                    include_solvent: bool = False) -> dict[str, Any]:
    """Every residue with a heavy atom strictly within `cutoff_nm` of the ligand's heavy atoms.

    `ligand` is a one-based topology residue index, or a mask naming exactly one residue.
    Returns the ligand's own description, the selected residues and the near misses, each with its
    minimum distance in nm.
    """
    import numpy as np
    from openmm import unit

    from .masks import parse_residue_mask
    from .regions import residue_kind
    from .selection import topology_digest

    residues = list(topology.residues())
    if isinstance(ligand, str):
        numbers = parse_residue_mask(ligand, where="--ligand")
        if len(numbers) != 1:
            raise PocketError(f"--ligand {ligand!r} names {len(numbers)} residues; a pocket is "
                              f"measured around ONE ligand instance")
        number = numbers[0]
    else:
        number = int(ligand)
    if not 1 <= number <= len(residues):
        raise PocketError(f"residue {number} is out of range; this topology has {len(residues)} "
                          f"residues")
    target = residues[number - 1]
    if hasattr(positions, "value_in_unit"):
        positions = positions.value_in_unit(unit.nanometer)
    xyz = np.asarray([[float(v) for v in row] for row in positions], dtype=float)
    if xyz.shape[0] != topology.getNumAtoms():
        raise PocketError(f"{xyz.shape[0]} positions for {topology.getNumAtoms()} atoms")

    box = topology.getPeriodicBoxVectors()
    if box is not None:
        box = np.asarray([[float(v.value_in_unit(unit.nanometer)) for v in row] for row in box])
    ligand_atoms = _heavy(target)
    if not ligand_atoms:
        raise PocketError(f"residue {number} ({target.name}) has no heavy atom to measure from")
    ligand_xyz = xyz[[atom.index for atom in ligand_atoms]]

    selected, near = [], []
    for other in residues:
        if other.index == target.index:
            continue
        kind = residue_kind(other)
        if kind == "solvent" and not include_solvent:
            continue
        atoms = _heavy(other)
        if not atoms:
            continue
        deltas = (xyz[[a.index for a in atoms]][:, None, :] - ligand_xyz[None, :, :]).reshape(-1, 3)
        distance = float(np.linalg.norm(_minimum_image(deltas, box), axis=1).min())
        entry = {"residue": other.index + 1, "chain": other.chain.id,
                 "residue_id": str(other.id).strip(),
                 "insertion_code": (other.insertionCode or "").strip(),
                 "residue_name": other.name, "kind": kind,
                 "minimum_distance_nm": round(distance, 4)}
        if distance < float(cutoff_nm):
            selected.append(entry)
        elif distance < float(cutoff_nm) + float(margin_nm):
            near.append(entry)
    return {"criterion": POCKET_CRITERION, "cutoff_nm": float(cutoff_nm),
            "margin_nm": float(margin_nm), "periodic": box is not None,
            "topology_sha256": topology_digest(topology),
            "ligand": {"residue": number, "residue_name": target.name,
                       "chain": target.chain.id, "residue_id": str(target.id).strip(),
                       "heavy_atoms": len(ligand_atoms)},
            "residues": sorted(selected, key=lambda e: e["residue"]),
            "near_misses": sorted(near, key=lambda e: e["minimum_distance_nm"])}


def residue_mask(numbers: Sequence[int]) -> str:
    """`[45, 46, 47, 59]` -> `":45-47,59"`: the compact form of the mask grammar."""
    numbers = sorted({int(n) for n in numbers})
    if not numbers:
        return ""
    parts, start, previous = [], numbers[0], numbers[0]
    for number in numbers[1:] + [None]:
        if number == previous + 1:
            previous = number
            continue
        parts.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = number
    return ":" + ",".join(parts)


def pocket_report(found: dict[str, Any], *, source: str | None = None) -> str:
    """The printed block: what was measured, the table, and the mask to paste."""
    ligand = found["ligand"]
    lines = [
        f"pocket of {ligand['residue_name']} at topology residue {ligand['residue']} "
        f"(chain {ligand['chain']!r}, id {ligand['residue_id']}), "
        f"{ligand['heavy_atoms']} heavy atoms",
        f"  criterion : {found['criterion']}: heavy-atom minimum distance, strictly < "
        f"{found['cutoff_nm']} nm"
        + (", minimum image (the topology has a box)" if found["periodic"] else ", no box"),
        f"  structure : {source or 'the given topology'}  "
        f"topology_sha256 {found['topology_sha256'][:12]}...",
        "",
        "  index  chain  resid   name   min distance (nm)",
    ]
    for entry in found["residues"]:
        lines.append(f"  {entry['residue']:>5}  {entry['chain']:>5}  {entry['residue_id']:>5}"
                     f"{entry['insertion_code']:<1}  {entry['residue_name']:<5}  "
                     f"{entry['minimum_distance_nm']:>8.3f}")
    if not found["residues"]:
        lines.append("  (none)")
    mask = residue_mask([entry["residue"] for entry in found["residues"]])
    lines += ["", "  paste into the scaler configuration, unchanged:",
              f"    sidechain_scaling_list: \"{mask}\"" if mask else
              "    (nothing within the cutoff)"]
    if found["near_misses"]:
        lines += ["", f"  NOT selected, within {found['margin_nm']} nm beyond the cutoff "
                      f"(the boundary is strict):"]
        for entry in found["near_misses"]:
            lines.append(f"    {entry['residue']:>5} {entry['residue_name']:<5} "
                         f"{entry['minimum_distance_nm']:>8.3f} nm")
    lines += ["", "  The mask is one-based TOPOLOGY residue indices of this structure. It is a "
                  "decision, not a", "  query: nothing resolves a pocket at run time."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from openmm.app import PDBFile, PDBxFile

    parser = argparse.ArgumentParser(
        prog="python -m md_tools.rest2.pocket", allow_abbrev=False,
        description="Print the residues lining a ligand, as a mask to paste into a "
                    "scaler.config. Resolves nothing at run time.")
    parser.add_argument("structure", help="the built structure (build/built.pdb or .cif)")
    parser.add_argument("--ligand", required=True,
                        help="the ligand instance: a mask naming ONE residue, e.g. \":201\"")
    parser.add_argument("--cutoff-nm", type=float, default=DEFAULT_CUTOFF_NM)
    parser.add_argument("--margin-nm", type=float, default=DEFAULT_MARGIN_NM)
    parser.add_argument("--include-solvent", action="store_true",
                        help="measure water and ions too; off by default")
    arguments = parser.parse_args(argv)

    path = Path(arguments.structure)
    reader = PDBxFile if path.suffix.lower() in (".cif", ".pdbx") else PDBFile
    structure = reader(str(path))
    try:
        found = pocket_residues(structure.topology, structure.positions,
                                ligand=arguments.ligand, cutoff_nm=arguments.cutoff_nm,
                                margin_nm=arguments.margin_nm,
                                include_solvent=arguments.include_solvent)
    except (PocketError, ValueError) as refusal:
        print(f"pocket: {refusal}", file=__import__("sys").stderr)
        return 2
    print(pocket_report(found, source=path.name))
    return 0


if __name__ == "__main__":                                        # pragma: no cover - entry point
    raise SystemExit(main())
