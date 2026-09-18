#!/usr/bin/env python
"""What the REST2 ladder bought: walker round trips, and the torsions it did and did not heat.

Run it from the dataset root (the directory holding `build/` and `REST2-run1/`):

    python paracetamol_analysis.py [REST2-run1]

TWO MEASUREMENTS, and they answer different questions.

`round trips` counts, per WALKER, a full journey from the physical state (0) to the hottest
state (3) and back. It is the honest measure of whether a ladder is doing anything: a high
exchange acceptance with no round trips means neighbouring states swap busily and nothing ever
crosses the ladder, which buys no sampling at all.

`circular sd` is measured per STATE, on `solute_state<i>_prod1.nc`, which follows the
thermodynamic state rather than a walker -- so state 0 is the physical ensemble and needs no
demultiplexing. A torsion left unscaled must look the SAME in state 3 as in state 0 (that is
what "unscaled" means); a scaled one should visibly loosen. Ordinary standard deviation is wrong
for an angle -- 179 deg and -179 deg are 2 deg apart, not 358 -- so this uses the circular
standard deviation, sqrt(-2 ln R) with R the mean resultant length.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mdtraj as md
import numpy as np

#: Paracetamol atoms, by package index (see build/parameter/molecule.sdf).
#: C1-C2(=O1)-N1-C3: the amide. C2-N1-C3-C4: the ring's rotation about the N-C bond.
AMIDE = (0, 1, 3, 4)
RING = (1, 3, 4, 5)


def circular_sd_degrees(radians: np.ndarray) -> float:
    resultant = np.abs(np.mean(np.exp(1j * radians)))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))


def round_trips(exchange_csv: Path) -> dict[int, int]:
    """Full 0 -> top -> 0 journeys per walker, from the committed exchange rows."""
    rows = np.genfromtxt(exchange_csv, delimiter=",", names=True)
    walkers = rows["walker"].astype(int)
    states = rows["state"].astype(int)
    top = int(states.max())
    trips: dict[int, int] = {}
    for walker in sorted(set(walkers)):
        visited = states[walkers == walker]
        # Keep only the ends of the ladder: the journey between them is what a trip is.
        ends = visited[(visited == 0) | (visited == top)]
        if ends.size == 0:
            trips[walker] = 0
            continue
        changes = ends[np.insert(np.diff(ends) != 0, 0, True)]
        # 0, top, 0 is one round trip; count completed returns to 0 after reaching the top.
        count, seen_top = 0, False
        for state in changes:
            if state == top:
                seen_top = True
            elif state == 0 and seen_top:
                count += 1
                seen_top = False
        trips[walker] = count
    return trips


def main(argv: list[str]) -> int:
    run = Path(argv[1] if len(argv) > 1 else "REST2-run1")
    topology = Path("build") / "built.pdb"
    if not run.is_dir() or not topology.is_file():
        print(f"run this from the dataset root: {topology} and {run}/ must exist", file=sys.stderr)
        return 2

    trips = round_trips(run / "exchange.csv")
    print(f"round trips (0 -> top -> 0), per walker: "
          f"{', '.join(str(trips[w]) for w in sorted(trips))}")

    solute = md.load(str(Path("build") / "built.solute.pdb"))
    states = sorted(run.glob("solute_state*_prod1.nc"))
    print(f"\n{'torsion':<26}{'scaled?':<10}" + "".join(f"state {n:<10}" for n, _ in
                                                        enumerate(states)))
    for label, atoms, scaled in (("amide omega 0-1-3-4", AMIDE, "no"),
                                 ("ring about N-C 1-3-4-5", RING, "yes")):
        line = f"{label:<26}{scaled:<10}"
        for path in states:
            traj = md.load(str(path), top=solute.topology)
            angles = md.compute_dihedrals(traj, [list(atoms)])[:, 0]
            line += f"{circular_sd_degrees(angles):>8.1f} deg  "
        print(line)
        if scaled == "no":
            # An unscaled amide must not isomerise in ANY state: a cis frame in the hot state
            # would be a geometry state 0 never visits, which is what leaving it unscaled prevents.
            for n, path in enumerate(states):
                traj = md.load(str(path), top=solute.topology)
                angles = md.compute_dihedrals(traj, [list(atoms)])[:, 0]
                cis = int(np.sum(np.abs(np.degrees(angles)) < 90.0))
                print(f"    state {n}: {cis} cis frame(s) of {len(angles)} "
                      f"(|omega| < 90 deg)")
    print(f"\n{len(md.load(str(states[0]), top=solute.topology))} frames per state, "
          f"{len(states)} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
