"""Missing heavy atoms and missing residues in a deposited structure: found, reported, decided.

A crystal structure routinely lacks the side-chain atoms of a disordered surface residue (1BRS
chain D GLN 58 has no OE1 or NE2) and the residues of a disordered terminus. Force-field template
matching then fails deep inside hydrogen addition ("No template found for residue 310 (GLN)"),
which names neither the residue nor the choice that has to be made.

`inspect_structure` finds both with PDBFixer and returns a record. The policy decides:

  missing_atoms: refuse   (default) -- a build with incomplete residues is refused, listing each
                           residue and the atoms it lacks by chain, residue id and insertion code.
  missing_atoms: add      -- PDBFixer.addMissingAtoms builds them (heavy atoms, and a terminal OXT);
                           every added atom is recorded. Built atoms carry no crystallographic
                           evidence, and the record says which they are.

MISSING RESIDUES ARE NEVER BUILT. A gap INSIDE a chain would need loop modelling, which is not a
preparation md-tools makes up: it is refused, naming the gap. Residues missing from a chain's
ends are left unbuilt and recorded -- the chain simply starts or ends at its first or last
observed residue, with its terminus capped as that residue.

Only standard residues are completed (PDBFixer has templates for nothing else); ligands, waters
and ions are left exactly as they are.
"""
from __future__ import annotations

from typing import Any

POLICIES = ("refuse", "add")


class CompletionError(ValueError):
    """A structure that cannot be built under the stated policy. Refused before any output."""


def _key(residue) -> dict[str, str]:
    return {"chain": residue.chain.id, "resid": str(residue.id).strip(),
            "insertion_code": (residue.insertionCode or "").strip(), "residue": residue.name}


#: A peptide C-N distance above this between consecutive residues of one chain is a break.
BREAK_ANGSTROM = 2.0


def _chain_breaks(topology, positions) -> list[dict[str, Any]]:
    """Consecutive protein residues of one chain whose C and N are not bonded-close."""
    import numpy as np
    from openmm import unit

    from .system import PROTEIN_RESIDUES

    xyz = np.asarray(positions.value_in_unit(unit.angstrom))
    breaks = []
    for chain in topology.chains():
        residues = [r for r in chain.residues() if r.name.upper() in PROTEIN_RESIDUES]
        for previous, following in zip(residues, residues[1:]):
            c = next((a.index for a in previous.atoms() if a.name == "C"), None)
            n = next((a.index for a in following.atoms() if a.name == "N"), None)
            if c is None or n is None:
                continue
            distance = float(np.linalg.norm(xyz[c] - xyz[n]))
            if distance > BREAK_ANGSTROM:
                breaks.append({"chain": chain.id, "previous": _key(previous),
                               "next": _key(following), "distance_angstrom": round(distance, 2)})
    return breaks


def _renamed_not_missing(fixer, flagged: list) -> list:
    """The flagged residues whose atoms are PRESENT under names the template does not define.

    A template is matched by atom NAME, so a residue written with another program's names reads as
    one missing exactly those atoms -- which is what a prepared structure with Maestro cap names
    (ACE `C1 O1 C2 H2_*`) does. The old refusal said "missing heavy atoms" and pointed at
    `input.missing_atoms: add`, which would have added a SECOND copy of every one of them. A
    residue is reported here instead when it carries at least as many atoms as the template
    defines and at least as many unrecognised names as the atoms said to be missing.
    """
    by_key = {}
    for residue in fixer.topology.residues():
        by_key[(residue.chain.id, str(residue.id).strip(),
                (residue.insertionCode or "").strip())] = residue
    renamed = []
    for entry in flagged:
        template = fixer.templates.get(entry["residue"])
        residue = by_key.get((entry["chain"], entry["resid"], entry["insertion_code"]))
        if template is None or residue is None:
            continue
        # HEAVY ATOMS ONLY, on both sides: PDBFixer's templates define no hydrogens, so counting
        # a residue's hydrogens against them made every incomplete residue look renamed -- an
        # alanine missing its CB "has H HA HB1 HB2 HB3 where the template has CB".
        def heavy(atoms):
            return [a.name for a in atoms if (a.element is None or a.element.symbol != "H")]

        defined = set(heavy(template.topology.atoms()))
        present = heavy(residue.atoms())
        missing = [name for name in entry["missing_atoms"] if name in defined]
        unmatched = sorted(name for name in present if name not in defined)
        if unmatched and len(present) >= len(defined) and len(unmatched) >= len(missing):
            renamed.append({**entry, "unmatched_names": unmatched,
                            "template_atom_count": len(defined),
                            "present_atom_count": len(present)})
    return renamed


def inspect_structure(topology, positions, *, missing_atoms: str = "refuse"):
    """(topology, positions, record) with missing heavy atoms handled under `missing_atoms`."""
    from pdbfixer import PDBFixer

    if missing_atoms not in POLICIES:
        raise CompletionError(f"input.missing_atoms must be one of {POLICIES}, got "
                              f"{missing_atoms!r}")
    import io

    from openmm import app

    buffer = io.StringIO()
    app.PDBFile.writeFile(topology, positions, buffer, keepIds=True)
    buffer.seek(0)
    fixer = PDBFixer(pdbfile=buffer)
    internal = _chain_breaks(fixer.topology, fixer.positions)
    if internal:
        shown = "; ".join(
            f"chain {g['chain']} between {g['previous']['residue']} {g['previous']['resid']}"
            f"{g['previous']['insertion_code']} and {g['next']['residue']} {g['next']['resid']}"
            f"{g['next']['insertion_code']} (C-N {g['distance_angstrom']} A)" for g in internal)
        raise CompletionError(
            f"the structure has {len(internal)} break(s) INSIDE a chain ({shown}): residues are "
            f"missing there. Building them would be loop modelling, which md-tools does not "
            f"invent; model the gap with a dedicated tool, or build a structure that does not need "
            f"it. Nothing was written.")
    # Residues missing from a chain's ENDS are not built. PDBFixer reads them from SEQRES, which a
    # prepared or expanded file may not carry, so they are not claimed here either way.
    fixer.missingResidues = {}
    terminal: list = []

    fixer.findMissingAtoms()
    incomplete = [{**_key(residue), "missing_atoms": sorted(atom.name for atom in atoms)}
                  for residue, atoms in fixer.missingAtoms.items()]
    terminals = [{**_key(residue), "missing_atoms": sorted(names)}
                 for residue, names in fixer.missingTerminals.items()]
    record: dict[str, Any] = {
        "missing_atoms_policy": missing_atoms,
        "chain_breaks": [],
        "residues_missing_atoms": incomplete,
        "missing_terminal_atoms": terminals,
        "atoms_added": [],
    }
    if not incomplete and not terminals:
        return topology, positions, record
    renamed = _renamed_not_missing(fixer, incomplete + terminals)
    record["residues_with_unmatched_atom_names"] = renamed
    if missing_atoms == "refuse":
        if renamed:
            shown = "; ".join(
                f"{r['chain']}:{r['resid']}{r['insertion_code']} {r['residue']} has "
                f"{' '.join(r['unmatched_names'])} where the template has "
                f"{' '.join(r['missing_atoms'])}" for r in renamed[:12])
            more = len(renamed) - 12
            raise CompletionError(
                f"{len(renamed)} residue(s) carry the right number of atoms under names the "
                f"force-field template does not define ({shown}"
                f"{f'; and {more} more' if more > 0 else ''}). These atoms are PRESENT: what does "
                f"not match is their NAMES, which is how templates are matched. Rename them to "
                f"the template's names in your structure -- md-tools does not rename atoms in "
                f"someone's file -- and do NOT set input.missing_atoms: add, which would add a "
                f"second copy of each. Nothing was written.")
        shown = "; ".join(f"{r['chain']}:{r['resid']}{r['insertion_code']} {r['residue']} lacks "
                          f"{' '.join(r['missing_atoms'])}" for r in (incomplete + terminals)[:12])
        more = len(incomplete) + len(terminals) - 12
        raise CompletionError(
            f"{len(incomplete) + len(terminals)} residue(s) are missing heavy atoms ({shown}"
            f"{f'; and {more} more' if more > 0 else ''}). Force-field templates need complete "
            f"residues. Set input.missing_atoms: add to build them with PDBFixer (every added atom "
            f"is recorded), or supply a completed structure. Nothing was written.")
    before = {(a.residue.chain.id, str(a.residue.id).strip(), a.residue.insertionCode or "",
               a.name) for a in fixer.topology.atoms()}
    fixer.addMissingAtoms()
    record["atoms_added"] = [
        {"chain": a.residue.chain.id, "resid": str(a.residue.id).strip(),
         "insertion_code": (a.residue.insertionCode or "").strip(), "residue": a.residue.name,
         "atom": a.name}
        for a in fixer.topology.atoms()
        if (a.residue.chain.id, str(a.residue.id).strip(), a.residue.insertionCode or "",
            a.name) not in before]
    return fixer.topology, fixer.positions, record
