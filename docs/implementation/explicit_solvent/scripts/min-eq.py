#!/usr/bin/env python
"""Stage (b) -- minimisation and equilibration.

    min-eq.py --p $topology --c $crd --out-suffix "$SUFFIX" --config min-eq-config.json

Default protocol ("staged") is the standard one for a flexible solute in a freshly packed water
box: restrained minimisation, free minimisation, 50->300 K ramp under restraint, NPT restrained,
the restraint released in steps, then free NPT whose tail fixes the production box.  2.0 ns total.
"simple" (minimise -> NVT -> NPT, unrestrained) is kept for small rigid solutes only.

Production then runs NVT at that box: the exchange criterion carries no PV term, and the free cold
walker must be the same ensemble as the REMD s=1 rung it is compared against.

Outputs
    <SUFFIX>_state.xml         positions + velocities + box -- the "-c" of stages (c) and (d)
    <SUFFIX>_equilibrated.pdb  <SUFFIX>_minimized.pdb
    <SUFFIX>_equilibration.csv <SUFFIX>_min-eq.json  (per-stage energy, volume and solute RMSD)
"""

from __future__ import annotations

from escort_ais.systems.explicit_baseline import minimize_equilibrate

from _stage_cli import resolve, stage_parser


def main() -> None:
    args = stage_parser(__doc__, needs_topology=True).parse_args()
    cfg, out_dir = resolve(args)
    info = minimize_equilibrate(cfg, args.topology, args.coords, out_dir, args.out_suffix)
    print(f"[min-eq] protocol '{info['protocol']}', {len(info['stages'])} stages, "
          f"{info['total_equilibration_ps']:g} ps, {info['n_restrained_atoms']} atoms restrained")
    print(f"[min-eq] solute heavy-atom RMSD from the minimised structure: "
          f"{info['solute_heavy_rmsd_from_start_nm']:.3f} nm")
    print(f"[min-eq] production box {info['box_volume_nm3_mean']:.2f} +/- "
          f"{info['box_volume_nm3_sd']:.2f} nm^3 over {info['n_box_samples']} samples")
    print(f"[min-eq] -> {info['state_xml']}")


if __name__ == "__main__":
    main()
