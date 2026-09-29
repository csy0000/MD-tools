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
           "interface_residues", "pocket_residues", "residue_mask", "pocket_report", "main"]

#: Name and version of the criterion, printed with every report. Version 2 reports a SIDE per
#: entry, because the two regions are no longer a ligand and everything else.
POCKET_CRITERION = "md-tools-pocket-selection/2"
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


def _resolve_side(spec, residues, *, where: str) -> list[int]:
    """A mask or a residue index -> the one-based numbers it names, checked against the topology."""
    from .masks import parse_residue_mask

    if isinstance(spec, str):
        numbers = parse_residue_mask(spec, where=where)
    elif isinstance(spec, (list, tuple, set, frozenset)):
        numbers = [int(n) for n in spec]
    else:
        numbers = [int(spec)]
    if not numbers:
        raise PocketError(f"{where} names no residue")
    for number in numbers:
        if not 1 <= number <= len(residues):
            raise PocketError(f"{where}: residue {number} is out of range; this topology has "
                              f"{len(residues)} residues")
    return sorted(set(numbers))


def interface_residues(topology, positions, *, int2, int1=None,
                       cutoff_nm: float = DEFAULT_CUTOFF_NM,
                       margin_nm: float = DEFAULT_MARGIN_NM,
                       include_solvent: bool = False) -> dict[str, Any]:
    """Residues of each region with a heavy atom strictly within `cutoff_nm` of the OTHER region.

    Two regions, because an interface has two sides and a ligand site is only the special case
    where one of them is a single residue. `--int2` is that side; `int1` defaults to every
    non-solvent residue NOT in `int2`, which reproduces what `--ligand` meant.

    The result reports the two sides SEPARATELY, and that separation is the point. For a ligand
    site the int1 side is the pocket and belongs in `sidechain_scaling_list`, while the ligand
    itself belongs in `ligand_scaling_dict` and must NOT go in the sidechain list; for a
    protein-protein interface both sides are sidechains and the combined mask is what you want.
    Merging them unconditionally would be wrong for exactly one of those, silently.

    Only cross-region pairs are measured. Contacts WITHIN a region are not contacts across an
    interface, so a residue is never selected because of its own neighbours.
    """
    import numpy as np
    from openmm import unit

    from .regions import residue_kind
    from .selection import topology_digest

    residues = list(topology.residues())
    side2 = _resolve_side(int2, residues, where="--int2")
    if int1 is None:
        chosen = set(side2)
        side1 = [r.index + 1 for r in residues
                 if r.index + 1 not in chosen
                 and (include_solvent or residue_kind(r) != "solvent")]
    else:
        side1 = _resolve_side(int1, residues, where="--int1")
    overlap = sorted(set(side1) & set(side2))
    if overlap:
        raise PocketError(f"--int1 and --int2 share residue(s) {overlap[:6]}; an interface has "
                          f"two SIDES, and a residue cannot be on both")

    if hasattr(positions, "value_in_unit"):
        positions = positions.value_in_unit(unit.nanometer)
    xyz = np.asarray([[float(v) for v in row] for row in positions], dtype=float)
    if xyz.shape[0] != topology.getNumAtoms():
        raise PocketError(f"{xyz.shape[0]} positions for {topology.getNumAtoms()} atoms")
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        box = np.asarray([[float(v.value_in_unit(unit.nanometer)) for v in row] for row in box])

    def heavy_indices(numbers):
        out = []
        for number in numbers:
            out.extend(atom.index for atom in _heavy(residues[number - 1]))
        return out

    def describe(number, distance, side):
        residue = residues[number - 1]
        return {"residue": number, "chain": residue.chain.id,
                "residue_id": str(residue.id).strip(),
                "insertion_code": (residue.insertionCode or "").strip(),
                "residue_name": residue.name, "kind": residue_kind(residue), "side": side,
                "minimum_distance_nm": round(distance, 4)}

    # BOTH sides are measured when both were stated, because an interface has two of them.
    # When int1 was defaulted the caller asked "what lines this?", and the answer is the other
    # side only: reporting the int2 residues back would say a ligand lines its own pocket, and
    # would put it in a sidechain list where it must not appear.
    sides = [(side1, side2, "int1")]
    if int1 is not None:
        sides.append((side2, side1, "int2"))

    selected, near = [], []
    for mine, theirs, side in sides:
        other_xyz = xyz[heavy_indices(theirs)]
        if not len(other_xyz):
            raise PocketError("the opposite region has no heavy atom to measure against")
        for number in mine:
            atoms = _heavy(residues[number - 1])
            if not atoms:
                continue                       # glycine, an ion: nothing to measure or to heat
            deltas = (xyz[[a.index for a in atoms]][:, None, :]
                      - other_xyz[None, :, :]).reshape(-1, 3)
            distance = float(np.linalg.norm(_minimum_image(deltas, box), axis=1).min())
            if distance < float(cutoff_nm):
                selected.append(describe(number, distance, side))
            elif distance < float(cutoff_nm) + float(margin_nm):
                near.append(describe(number, distance, side))

    return {"criterion": POCKET_CRITERION, "cutoff_nm": float(cutoff_nm),
            "margin_nm": float(margin_nm), "periodic": box is not None,
            "topology_sha256": topology_digest(topology),
            "int1": {"residues": side1, "stated": int1 is not None},
            "int2": {"residues": side2, "stated": True},
            "residues": sorted(selected, key=lambda e: e["residue"]),
            "near_misses": sorted(near, key=lambda e: e["minimum_distance_nm"])}


def pocket_residues(topology, positions, *, ligand, cutoff_nm: float = DEFAULT_CUTOFF_NM,
                    margin_nm: float = DEFAULT_MARGIN_NM,
                    include_solvent: bool = False) -> dict[str, Any]:
    """The ligand case of `interface_residues`: one residue against everything else.

    Kept because it is the common call and reads as what it does. `--ligand X` is exactly
    `--int2 X` with `int1` left to default, so this adds no second criterion.
    """
    residues = list(topology.residues())
    numbers = _resolve_side(ligand, residues, where="--ligand")
    if len(numbers) != 1:
        raise PocketError(f"--ligand {ligand!r} names {len(numbers)} residues; a pocket is "
                          f"measured around ONE ligand instance. For a region with several "
                          f"residues, or for a protein-protein interface, use --int1/--int2.")
    return interface_residues(topology, positions, int2=numbers, int1=None,
                              cutoff_nm=cutoff_nm, margin_nm=margin_nm,
                              include_solvent=include_solvent)


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
    """The printed block: what was measured, the table, and the mask(s) to paste."""
    side1 = [e for e in found["residues"] if e["side"] == "int1"]
    side2 = [e for e in found["residues"] if e["side"] == "int2"]
    stated = found["int1"]["stated"]
    lines = [
        (f"interface between --int1 ({len(found['int1']['residues'])} residue(s)) and "
         f"--int2 ({len(found['int2']['residues'])} residue(s))" if stated else
         f"pocket around --int2 ({len(found['int2']['residues'])} residue(s)), measured against "
         f"every other non-solvent residue"),
        f"  criterion : {found['criterion']}: heavy-atom minimum distance ACROSS the two "
        f"regions, strictly < {found['cutoff_nm']} nm"
        + (", minimum image (the topology has a box)" if found["periodic"] else ", no box"),
        f"  structure : {source or 'the given topology'}  "
        f"topology_sha256 {found['topology_sha256'][:12]}...",
        "",
        "   side  index  chain  resid   name   min distance (nm)",
    ]
    for entry in found["residues"]:
        lines.append(f"  {entry['side']:>5}  {entry['residue']:>5}  {entry['chain']:>5}  "
                     f"{entry['residue_id']:>5}{entry['insertion_code']:<1}  "
                     f"{entry['residue_name']:<5}  {entry['minimum_distance_nm']:>8.3f}")
    if not found["residues"]:
        lines.append("  (none)")

    mask1 = residue_mask([e["residue"] for e in side1])
    mask2 = residue_mask([e["residue"] for e in side2])
    both = residue_mask([e["residue"] for e in found["residues"]])
    lines += ["", "  paste into the scaler configuration, unchanged:"]
    if stated:
        # Both sides are protein: the combined mask is the sidechain list.
        lines += [f"    sidechain_scaling_list: \"{both}\"" if both else
                  "    (nothing within the cutoff)"]
        if mask1 and mask2:
            lines += [f"      --int1 side only: \"{mask1}\"",
                      f"      --int2 side only: \"{mask2}\""]
    else:
        # A ligand site: the pocket is the int1 side ONLY. The ligand goes in
        # ligand_scaling_dict, and putting it in the sidechain list too would scale it twice.
        lines += [f"    sidechain_scaling_list: \"{mask1}\"" if mask1 else
                  "    (nothing within the cutoff)"]
        lines += ["      the --int2 residue(s) are NOT in that list: a ligand belongs in",
                  "      ligand_scaling_dict, and naming it in both would scale it twice."]
    if found["near_misses"]:
        lines += ["", f"  NOT selected, within {found['margin_nm']} nm beyond the cutoff "
                      f"(the boundary is strict):"]
        for entry in found["near_misses"]:
            lines.append(f"    {entry['side']:>5} {entry['residue']:>5} "
                         f"{entry['residue_name']:<5} "
                         f"{entry['minimum_distance_nm']:>8.3f} nm")
    what = "an interface" if stated else "a pocket"
    lines += ["", "  The mask is one-based TOPOLOGY residue indices of this structure. It is a "
                  "decision, not a", f"  query: nothing resolves {what} at run time."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from openmm.app import PDBFile, PDBxFile

    parser = argparse.ArgumentParser(
        prog="python -m md_tools.rest2.pocket", allow_abbrev=False,
        description="Print the residues lining an interface -- a ligand site or a "
                    "protein-protein contact -- as a mask to paste into a scaler.config. "
                    "Resolves nothing at run time.")
    parser.add_argument("structure", help="the built structure (build/built.pdb or .cif)")
    parser.add_argument("--int2", "--ligand", dest="int2", required=True,
                        help="one side of the interface, as a mask. For a ligand site this is "
                             "the ligand, e.g. \":291\"; --ligand is the same option under its "
                             "older name.")
    parser.add_argument("--int1", default=None,
                        help="the other side, as a mask, e.g. a protein chain. Left out, it is "
                             "every non-solvent residue NOT in --int2, which is what a ligand "
                             "pocket means.")
    parser.add_argument("--cutoff-nm", type=float, default=DEFAULT_CUTOFF_NM)
    parser.add_argument("--margin-nm", type=float, default=DEFAULT_MARGIN_NM)
    parser.add_argument("--include-solvent", action="store_true",
                        help="measure water and ions too; off by default")
    arguments = parser.parse_args(argv)

    path = Path(arguments.structure)
    reader = PDBxFile if path.suffix.lower() in (".cif", ".pdbx") else PDBFile
    structure = reader(str(path))
    try:
        found = interface_residues(structure.topology, structure.positions,
                                   int2=arguments.int2, int1=arguments.int1,
                                   cutoff_nm=arguments.cutoff_nm,
                                   margin_nm=arguments.margin_nm,
                                   include_solvent=arguments.include_solvent)
    except (PocketError, ValueError) as refusal:
        print(f"pocket: {refusal}", file=__import__("sys").stderr)
        return 2
    print(pocket_report(found, source=path.name))
    return 0


if __name__ == "__main__":                                        # pragma: no cover - entry point
    raise SystemExit(main())
