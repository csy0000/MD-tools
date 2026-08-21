#!/usr/bin/env python
"""Stage (a) -- build the simulation box.

    simbox-setup.py --smiles "$SMILES" --out-suffix "$SUFFIX" --config simbox-config.json
    simbox-setup.py --pdb    "$PDB"    --out-suffix "$SUFFIX" --config simbox-config.json

Initial structure -> hydrogens at pH 7 -> rhombic-dodecahedron solvation (OPC, 0.15 M NaCl)
-> OpenMM System with PME at 1.0 nm, HBonds constraints and hydrogen mass repartitioning.

--smiles is the LIGAND route: RDKit ETKDGv3 + MMFF94s, parameterised with Sage 2.2 + AM1BCC.
--pdb    is the PEPTIDE route: ff19SB, which matches by residue template and therefore cannot be
         used on a SMILES-built structure (RDKit yields a single UNL residue).

Outputs
    <SUFFIX>_system.xml     the "-p" of every later stage
    <SUFFIX>_topology.pdb   topology, solvated coordinates, box vectors -- the first "-c"
    <SUFFIX>_simbox.json    solute atom count, omega bonds, box geometry, force field, provenance
    <SUFFIX>_prep/          per-step intermediates
"""

from __future__ import annotations

from md_templates.openmm import build_simbox

from _stage_cli import resolve, stage_parser


def main() -> None:
    args = stage_parser(__doc__, needs_input=True).parse_args()
    cfg, out_dir = resolve(args)
    info = build_simbox(cfg, out_dir, args.out_suffix, smiles=args.smiles, pdb=args.pdb)
    g = info["geometry"]
    print(f"[simbox] {info['n_particles']} particles = {info['n_solute_atoms']} solute + "
          f"{info['n_waters']} waters + ions {info['ions']}")
    print(f"[simbox] {g['box_shape']} box, width {g['box_width_nm']:.3f} nm, solute-image gap "
          f"{g['solute_image_gap_nm']:.3f} nm, max legal cutoff {g['max_legal_cutoff_nm']:.3f} nm")
    print(f"[simbox] HMR: {info['hmr']['n_hydrogens_repartitioned']} hydrogens -> "
          f"{info['hmr']['target_hydrogen_mass_amu']} amu, total mass conserved")
    print(f"[simbox] omega bonds left unscaled by REST2: {info['omega_central_bonds']}")
    print(f"[simbox] -> {info['system_xml']}\n[simbox] -> {info['topology_pdb']}")


if __name__ == "__main__":
    main()
