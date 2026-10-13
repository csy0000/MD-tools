"""Which bonds does the REST2 scaler protect, and which does it leave on the lambda path?

    python scaler_report.py build/built.pdb build/REST2/scaler.yaml

Reads the topology the scaler was given and the `scaler.yaml` it wrote, and sorts every bond a
torsion can turn about (a solute bond whose two atoms each have another neighbour) into:

  amide omega    unscaled  -- a protein or peptide-like residue's ordinary amide C-N
  aromatic/double unscaled -- no other conformer to reach (planarity, cis/trans is a chemical isomer)
  scaled                   -- everything else: phi, psi, side-chain torsions, and a LIGAND's amide

Impropers are unscaled by convention and are not bonds, so they are counted, not listed.
"""
import sys
from collections import defaultdict

import yaml
from openmm.app import PDBFile


def main(pdb: str, scaler: str) -> None:
    topology = PDBFile(pdb).topology
    record = yaml.safe_load(open(scaler, encoding="utf-8"))
    solute = set(record["solute"]["atom_indices"])
    unscaled = {tuple(sorted(e["bond"])): e for e in record["unscaled_torsions"]["central_bonds"]}

    neighbours = defaultdict(set)
    for bond in topology.bonds():
        neighbours[bond.atom1.index].add(bond.atom2.index)
        neighbours[bond.atom2.index].add(bond.atom1.index)
    atoms = list(topology.atoms())

    def central(a, b):
        return a in solute and b in solute and len(neighbours[a]) >= 2 and len(neighbours[b]) >= 2

    rows = {"amide omega": [], "aromatic / double": [], "scaled": []}
    for bond in topology.bonds():
        a, b = sorted((bond.atom1.index, bond.atom2.index))
        if not central(a, b):
            continue
        label = f"{atoms[a].residue.name}:{atoms[a].name}-{atoms[b].name} ({a}-{b})"
        entry = unscaled.get((a, b))
        if entry is None:
            rows["scaled"].append(label)
        elif entry["class"] == "amide_omega":
            rows["amide omega"].append(label)
        else:
            rows["aromatic / double"].append(label)

    for name, labels in rows.items():
        print(f"{name:18s} {len(labels):3d}")
        for label in labels:
            print(f"    {label}")
    print(f"{'improper terms':18s} {record['unscaled_torsions']['counts']['improper_terms']:3d}  (unscaled)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
