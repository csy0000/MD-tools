"""Expand an mmCIF biological assembly into one structure, with every copy distinguishable.

An asymmetric unit and a biological assembly are different molecules: 1TYL's asymmetric unit is an
insulin dimer, its assembly 3 the T3R3 hexamer. `expand_assembly` applies the assembly's operators
(`_pdbx_struct_assembly_gen`, `_pdbx_struct_oper_list`) to the chains it names and writes the result
as a PDB file build-top reads like any other.

IDENTITY. Every expanded copy of an author chain gets its OWN chain id, assigned in a fixed order
(operators in assembly order, author chains in file order within each). The record keeps, per
output chain, the author chain it came from, the label_asym ids it carries (mmCIF's label and
author chain ids are different things and are never conflated here), and the operator id, name,
matrix and vector. Ligand selectors name the OUTPUT chain id; the record maps it back.

SPECIAL POSITIONS. An ion or water on a symmetry axis is deposited once at partial occupancy (1/3
on a three-fold axis) and every operator puts a copy on top of it. Those copies are one physical
atom, so a residue of an ion or water whose atoms coincide within `tolerance_angstrom` of an
already kept copy of the same residue name is dropped -- and each drop is RECORDED with both
identities, the distance and the summed occupancy. Nothing else is merged: coinciding protein or
ligand atoms mean the structure or the assembly choice is wrong, and that is refused, not repaired.

ALTERNATE CONFORMATIONS keep the first alternate location (gemmi's
`remove_alternative_conformations`), and every residue affected is recorded.

gemmi (conda-forge) does the mmCIF reading and the writing.
"""
from __future__ import annotations

import hashlib
import string
from pathlib import Path
from typing import Any

#: Residues that may legitimately sit on a symmetry axis and be deduplicated.
DEDUPLICABLE = frozenset({"HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB",
                          "CS", "CU", "FE", "MN", "CO", "NI", "CD"})

#: Default distance under which two copies of an ion or water are the same atom.
DEFAULT_TOLERANCE_ANGSTROM = 0.1

#: Output chain ids, in order. A PDB chain field holds one character.
CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits


class AssemblyError(ValueError):
    """An assembly that cannot be expanded as asked. Refused before any output."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_assembly(source: Path, assembly_id: str, out_pdb: Path, *,
                    tolerance_angstrom: float = DEFAULT_TOLERANCE_ANGSTROM) -> dict[str, Any]:
    """Write assembly `assembly_id` of mmCIF `source` to `out_pdb`. Returns the record."""
    import gemmi
    import numpy as np

    source, out_pdb = Path(source), Path(out_pdb)
    if source.suffix.lower() not in (".cif", ".mmcif"):
        raise AssemblyError(f"{source}: a biological assembly is read from mmCIF (.cif); PDB "
                            f"BIOMT remarks are not supported")
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    names = [assembly.name for assembly in structure.assemblies]
    assembly = next((a for a in structure.assemblies if a.name == str(assembly_id)), None)
    if assembly is None:
        raise AssemblyError(f"{source.name} has no assembly {assembly_id!r}; it defines "
                            f"{', '.join(names) or 'none'}")

    altlocs = sorted({(chain.name, str(residue.seqid.num), residue.seqid.icode.strip(),
                       residue.name)
                      for chain in structure[0] for residue in chain
                      if any(atom.altloc != "\0" for atom in residue)})
    structure.remove_alternative_conformations()
    model = structure[0]

    out = gemmi.Structure()
    out.cell = gemmi.UnitCell()
    out.spacegroup_hm = "P 1"
    out_model = gemmi.Model("1")
    chains: list[dict[str, Any]] = []
    kept: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    deduplicated: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    next_id = iter(CHAIN_IDS)

    for generator in assembly.generators:
        subchains = set(generator.subchains)
        for operator in generator.operators:
            matrix = np.array(operator.transform.mat.tolist(), dtype=float)
            vector = np.array(operator.transform.vec.tolist(), dtype=float)
            for chain in model:
                residues = [r for r in chain if r.subchain in subchains]
                if not residues:
                    continue
                try:
                    chain_id = next(next_id)
                except StopIteration:
                    raise AssemblyError(
                        f"assembly {assembly_id} of {source.name} expands to more than "
                        f"{len(CHAIN_IDS)} chains, which a PDB chain field cannot name") from None
                new_chain = gemmi.Chain(chain_id)
                labels = []
                for residue in residues:
                    copy = gemmi.Residue()
                    copy.name, copy.seqid, copy.subchain = residue.name, residue.seqid, residue.subchain
                    copy.het_flag = residue.het_flag
                    for atom in residue:
                        moved = gemmi.Atom()
                        moved.name, moved.element, moved.occ, moved.b_iso = \
                            atom.name, atom.element, atom.occ, atom.b_iso
                        xyz = matrix @ np.array([atom.pos.x, atom.pos.y, atom.pos.z]) + vector
                        moved.pos = gemmi.Position(*xyz)
                        copy.add_atom(moved)
                    identity = {"chain": chain_id, "resid": str(residue.seqid.num),
                                "insertion_code": residue.seqid.icode.strip(),
                                "residue": residue.name, "author_chain": chain.name,
                                "operator": operator.name}
                    duplicate = _coincident(copy, kept.get(residue.name, ()), tolerance_angstrom)
                    if duplicate is not None:
                        if residue.name.upper() not in DEDUPLICABLE:
                            raise AssemblyError(
                                f"{residue.name} {chain.name}:{residue.seqid.num} under operator "
                                f"{operator.name} lands on {duplicate[1]['residue']} "
                                f"{duplicate[1]['chain']}:{duplicate[1]['resid']} "
                                f"({duplicate[0]:.3f} A). Only ions and waters on a symmetry axis "
                                f"are merged; coinciding protein or ligand atoms mean the assembly "
                                f"or the structure is wrong. Nothing was written.")
                        deduplicated.append({
                            "dropped": identity, "kept": duplicate[1],
                            "distance_angstrom": round(duplicate[0], 3),
                            "occupancy_each": round(float(residue[0].occ), 3)})
                        continue
                    kept.setdefault(residue.name, []).append((copy, identity))
                    if residue.subchain not in labels:
                        labels.append(residue.subchain)
                    counts[residue.name] = counts.get(residue.name, 0) + 1
                    new_chain.add_residue(copy)
                if len(new_chain):
                    out_model.add_chain(new_chain)
                    chains.append({"chain_id": chain_id, "author_chain": chain.name,
                                   "label_asym_ids": labels, "assembly_id": assembly.name,
                                   "operator_id": operator.name, "operator_type": operator.type,
                                   "matrix": matrix.round(10).tolist(),
                                   "vector": vector.round(6).tolist(),
                                   "n_residues": len(new_chain)})
    out.add_model(out_model)
    out.setup_entities()
    for entry in deduplicated:
        copies = 1 + sum(1 for other in deduplicated if other["kept"] == entry["kept"])
        entry["occupancy_summed"] = round(entry["occupancy_each"] * copies, 3)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    # TER records stay: OpenMM starts a new chain at each, which is what makes a protein's last
    # residue its C terminus when waters and ions of the same author chain follow it.
    options = gemmi.PdbWriteOptions()
    options.minimal_file = True
    out.write_pdb(str(out_pdb), options)
    return {
        "source": source.name,
        "source_sha256": _sha256(source),
        "assembly_id": assembly.name,
        "assemblies_available": names,
        "output_pdb": out_pdb.name,
        "output_sha256": _sha256(out_pdb),
        "chains": chains,
        "residue_counts": dict(sorted(counts.items())),
        "deduplicated": deduplicated,
        "deduplication_tolerance_angstrom": tolerance_angstrom,
        "alternate_conformations": {
            "policy": "first alternate location kept (gemmi remove_alternative_conformations)",
            "residues": [{"author_chain": c, "resid": r, "insertion_code": i, "residue": n}
                         for c, r, i, n in altlocs]},
    }


def _coincident(residue, candidates, tolerance: float):
    """(distance, identity) of an already kept residue every atom of `residue` sits on, or None."""
    for other, identity in candidates:
        if len(other) != len(residue):
            continue
        worst = max(other[i].pos.dist(residue[i].pos) for i in range(len(residue)))
        if worst <= tolerance:
            return worst, identity
    return None
