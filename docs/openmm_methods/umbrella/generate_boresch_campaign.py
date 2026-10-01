#!/usr/bin/env python
"""Generate a Boresch dissociation campaign: one window per separation, seeded along the path.

    python generate_boresch_campaign.py <built.pdb> <campaign dir> \\
        --receptor 1503 2260 625 --ligand 4687 4673 4675 \\
        --from 0.628 --to 2.0 --spacing 0.05

Writes, per window, a system root holding `cv.yaml`, `umbrella.yaml` and a `umbrella.config`, plus
`run_campaign.sh` that runs them IN ORDER OF SEPARATION, each seeded from the previous window's
final state.

WHY THE WINDOWS MUST BE RUN IN ORDER AND SEEDED

    Not an optimisation -- the alternative does not run. A stiff restraint switched on at a
    configuration far from its centre is an enormous force: on the alanine dipeptide barrier,
    launching a k=1000 kJ/mol/rad^2 window 1.47 rad from its centre put 1080 kJ/mol of bias into
    the system and 7 of 12 windows died with `ValueError: Energy is NaN` in 1.9 seconds.

    A dissociation profile is that situation by construction. Every window starts from a BOUND
    complex, so a window at r = 2.0 nm starts about 1.4 nm from its centre; at
    2000 kJ/mol/nm^2 that is 0.5*2000*1.4^2 = 1960 kJ/mol of bias with a 2800 kJ/mol/nm force,
    applied to a ligand wedged in a pocket. It would not integrate, and if it did it would tear
    the ligand through whatever is in the way in a few steps.

    Seeding each window from the previous one's endpoint makes the initial displacement ONE
    SPACING -- 0.05 nm, so 0.5*2000*0.05^2 = 2.5 kJ/mol, which is kT. The ligand then leaves
    along a path the restraints walked it down, which is also what makes the five held
    orientational terms meaningful: they hold the pose the previous window ended in.

WHAT THIS DOES NOT DECIDE FOR YOU

    The anchors. Pass them, and `md_tools.umbrella.check_anchors` refuses a choice whose
    coordinates are not well defined before any window is generated -- every window shares the
    choice, so a bad one wastes the campaign rather than a run. Choosing them well (rigid
    backbone atoms, away from collinear, with the R1->L1 vector pointing out of the site) is
    discussed in boresch-dissociation.md.

    And it does not decide that the resulting profile is an unbinding free energy. It is the work
    along the path the restraints walked, with the orientation confined and hysteresis unmeasured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("built_pdb", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--receptor", type=int, nargs=3, required=True, metavar=("R1", "R2", "R3"))
    parser.add_argument("--ligand", type=int, nargs=3, required=True, metavar=("L1", "L2", "L3"))
    parser.add_argument("--from", dest="start_nm", type=float, default=None,
                        help="first window centre in nm (default: the bound separation)")
    parser.add_argument("--to", dest="end_nm", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.05, help="nm between windows")
    parser.add_argument("--distance-k", type=float, default=2000.0, help="kJ/mol/nm^2")
    parser.add_argument("--angular-k", type=float, default=100.0, help="kJ/mol/rad^2")
    parser.add_argument("--production-steps", type=int, default=5000000, help="10 ns at 2 fs")
    parser.add_argument("--card", type=int, default=0, help="the ONE card this campaign may use")
    arguments = parser.parse_args(argv)

    from openmm.app import PDBFile
    from openmm import unit

    from md_tools.umbrella import (BoreschAnchors, BoreschError, boresch_cv_document,
                                   boresch_window_document, check_anchors, standard_state_note)

    structure = PDBFile(str(arguments.built_pdb))
    positions = np.array([[c.value_in_unit(unit.nanometer) for c in v]
                          for v in structure.positions])
    anchors = BoreschAnchors(receptor=tuple(arguments.receptor), ligand=tuple(arguments.ligand))
    try:
        measured = check_anchors(positions, anchors)
    except BoreschError as refusal:
        print(str(refusal), file=sys.stderr)
        return 2

    print("anchors accepted; bound pose measured from " + arguments.built_pdb.name)
    for name, value in measured.items():
        print(f"  {name:4s} {value:9.3f} {'nm' if name == 'r' else 'deg'}")

    start = arguments.start_nm if arguments.start_nm is not None else measured["r"]
    centres = list(np.round(np.arange(start, arguments.end_nm + 1e-9, arguments.spacing), 4))
    if len(centres) < 2:
        print(f"a profile needs at least two windows; got {centres}", file=sys.stderr)
        return 2

    campaign = arguments.campaign
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "cv.yaml").write_text(
        yaml.safe_dump(boresch_cv_document(anchors), sort_keys=False), encoding="utf-8")

    names = []
    for centre in centres:
        name = f"r{int(round(centre * 1000)):04d}"       # r0628 = 0.628 nm, sorts correctly
        names.append(name)
        root = campaign / name
        root.mkdir(exist_ok=True)
        (root / "cv.yaml").write_text((campaign / "cv.yaml").read_text(encoding="utf-8"),
                                      encoding="utf-8")
        (root / "umbrella.yaml").write_text(
            yaml.safe_dump(boresch_window_document(
                measured, centre_nm=float(centre),
                distance_force_constant=arguments.distance_k,
                angular_force_constant=arguments.angular_k), sort_keys=False), encoding="utf-8")
        (root / "umbrella.config").write_text(yaml.safe_dump({
            "protocol": "umbrella", "solvent": "explicit",
            "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0},
            "stages": {"minimization_iterations": 5000, "restrained_nvt_steps": 25000,
                       "restrained_npt_steps": 25000, "unrestrained_npt_steps": 50000,
                       "production_steps": int(arguments.production_steps)},
            "reporting": {"crd_printout_solute": 5000, "info_printout": 25000,
                          "checkpoint_printout": 100000},
            "collective_variables": {"file": "cv.yaml", "interval_steps": 500},
            "umbrella": {"file": "umbrella.yaml"},
        }, sort_keys=False), encoding="utf-8")

    # The runner. One card, in order, each window seeded from the previous one's endpoint.
    script = campaign / "run_campaign.sh"
    script.write_text(f"""#!/usr/bin/env bash
# Boresch dissociation campaign: {len(names)} windows, {centres[0]:.3f} -> {centres[-1]:.3f} nm.
#
# ONE CARD, IN ORDER, SEEDED. See generate_boresch_campaign.py for why the order and the seeding
# are not optional: a window launched from the bound complex at its own centre would switch on
# ~2000 kJ/mol of bias and fail to integrate.
set -uo pipefail
CARD={arguments.card}
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # without this, device 0 is NOT the card you mean
export CUDA_VISIBLE_DEVICES="$CARD"
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
WINDOWS=({' '.join(names)})

previous=""
for name in "${{WINDOWS[@]}}"; do
  dir="$HERE/$name"
  if [ -f "$dir/run1/umbrella.cv.csv" ] && grep -q "status *completed" "$dir/run1/umbrella.log" 2>/dev/null; then
    echo "[$name] already complete"; previous="$dir"; continue
  fi
  ln -sfn ../build "$dir/build"
  ( cd "$dir" && md-openmm build-md -odir ./run1 --config umbrella.config >/dev/null ) \\
      || {{ echo "[$name] build-md FAILED"; exit 2; }}
  if [ -z "$previous" ]; then
    # The first window sits at the bound separation, so the ordinary chain is correct for it.
    ( cd "$dir/run1" && ./run.sh ) || {{ echo "[$name] FAILED"; exit 2; }}
  else
    # Every later window continues from its neighbour's endpoint, skipping equilibration: a
    # configuration already equilibrated one spacing away beats an unbiased bound one.
    ( cd "$dir/run1" && md-openmm md-run -i ../input/umbrella.in \\
        -p ../build/built.pdb -s ../build/built.xml \\
        -c "$previous/run1/umbrella.xml" -odir . ) || {{ echo "[$name] FAILED"; exit 2; }}
  fi
  echo "[$name] done, r = $(tail -1 "$dir/run1/umbrella.cv.csv" | cut -d, -f4)"
  previous="$dir"
done
echo "campaign finished: {len(names)} windows"
""", encoding="utf-8")
    script.chmod(0o755)

    print(f"\n{len(names)} windows written to {campaign}")
    print(f"  centres {centres[0]:.3f} .. {centres[-1]:.3f} nm, spacing {arguments.spacing} nm")
    print(f"  run with: {script}")
    print()
    print(standard_state_note(arguments.distance_k, arguments.angular_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
