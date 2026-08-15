#!/usr/bin/env python
"""Stage (c) -- one free walker, written in chunks.

    md.py --p $topology --c $crd --out-suffix "$SUFFIX" --config md-config.json

production.md.scale_factor selects the walker: s = 1 is the cold walker (1 us in 100 ns chunks by
default), s = 0.25 the hot one (200 ns in 1 ns chunks -- short chunks because that trajectory is an
AIS seed source and the chunk boundary should coincide with a cBAR generation).  Nothing else
differs between them, which is why there is one script and not two.

Resumable: re-running continues at the first chunk without a done.json and does NOT re-minimise.
A finished run reports already-complete and does nothing.

Outputs
    <SUFFIX>/chunk_0000../   traj_all.dcd (10 ps, wrapped), traj_solute.dcd (2 ps, UNwrapped),
                             state.csv, mid.chk, end.chk, done.json (wall time and ns/day)
    <SUFFIX>_md.json
"""

from __future__ import annotations

from escort_ais.systems.explicit_baseline import run_md

from _stage_cli import resolve, stage_parser


def main() -> None:
    args = stage_parser(__doc__, needs_topology=True).parse_args()
    cfg, out_dir = resolve(args)
    print(run_md(cfg, args.topology, args.coords, out_dir, args.out_suffix))


if __name__ == "__main__":
    main()
