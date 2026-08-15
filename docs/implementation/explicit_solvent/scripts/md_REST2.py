#!/usr/bin/env python
"""Stage (d) -- REST2 replica exchange.

    md_REST2.py --p $topology --c $crd --out-suffix "$SUFFIX" --config rest2-config.json

N replicas in one process, neighbour exchange every 10 ps, 100 ns chunks per replica.  Replica 0 is
the physical rung (s = 1).  The ladder comes from production.remd.scale_factors, or is built from
rest2.ladder (s_cold, s_hot, n_rungs, interp) when that is null.

The exchange criterion and the even/odd schedule are reused from escort_ais.methods.md_run, so
acceptance and round-trip statistics are comparable with the implicit-solvent references, and
<SUFFIX>_exchange_attempts.csv is written in the schema analysis/remd_reliability.py reads.

ONE process per GPU: two processes sharing a card without CUDA MPS run ~3.7x slower each.

Outputs
    <SUFFIX>/replica_00../chunk_0000../    as for md.py, per replica
    <SUFFIX>_exchange_attempts.csv         <SUFFIX>_rest2.json
"""

from __future__ import annotations

from escort_ais.systems.explicit_baseline import run_rest2_remd

from _stage_cli import resolve, stage_parser


def main() -> None:
    args = stage_parser(__doc__, needs_topology=True).parse_args()
    cfg, out_dir = resolve(args)
    print(run_rest2_remd(cfg, args.topology, args.coords, out_dir, args.out_suffix))


if __name__ == "__main__":
    main()
