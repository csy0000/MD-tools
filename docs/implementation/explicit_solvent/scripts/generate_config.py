#!/usr/bin/env python
"""Write the per-stage JSON config files the explicit-solvent stages take.

Each stage reads `--config <stage>-config.json`, a JSON of values that DIFFER from the baseline in
`escort_ais.systems.explicit_baseline.DEFAULTS`. All four files are slices of that one schema, so
nothing can drift between stages, and an unknown key is rejected rather than silently ignored.

  generate_config.py --all                          every stage, macrocycle defaults (8 rungs)
  generate_config.py --all --system cyclo_rgdfv     every stage, RGD defaults (10 rungs, Sage 2.2)
  generate_config.py --all --system alanine         every stage, alanine defaults (6 rungs)
  generate_config.py --simbox --smiles-route        just simbox-config.json
  generate_config.py --md --walker hot              just md-config.json, the 200 ns hot walker
  generate_config.py --rest2 --s_cold 1 --s_hot 0.25 --N_rungs 6 --interp sqrt

The REST2 ladder is spaced evenly in sqrt(s) by default. Exchange acceptance is governed by the
overlap of two rungs' energy distributions, whose width scales roughly as sqrt(s), so even spacing
there gives roughly even acceptance along the chain -- and the chain is only as good as its weakest
link. With s_hot = 0.25 and 6 rungs this reproduces the ladder every existing reference in this
project used, which is now the alanine default: [1.0, 0.81, 0.64, 0.49, 0.36, 0.25].

  --system alanine      6 rungs   [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]
  --system macrocycle   8 rungs   [1.0, 0.8622, 0.7347, 0.6173, 0.5102, 0.4133, 0.3265, 0.25]
  --system cyclo_rgdfv 10 rungs   [1.0, 0.8920, 0.7901, 0.6944, 0.6049, 0.5216, 0.4444, 0.3735,
                                   0.3086, 0.25]   (selected by matched pilots; RGD only)

Both are UNVALIDATED STARTING LADDERS.  Eight rungs does not guarantee adequate macrocycle
exchange; only a pilot that measures the worst neighbouring pair can say.

Stages (e), md_cBAR.py and pREST2.py, do not exist yet; `--cbar` / `--prest2` say so and write
nothing rather than emitting a config for a script that cannot read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from escort_ais.systems.explicit_baseline import DEFAULTS, load_config, rest2_ladder

# Per-system defaults. The ladder length is the only thing that differs today, and it differs
# because REST2's solute-solvent cross term grows with solute size: a dipeptide's rungs overlap far
# more readily than a macrocycle's, so alanine needs fewer of them to keep every neighbouring pair
# exchanging. These are STARTING POINTS -- verify with a short acceptance pilot before a long run.
SYSTEM_PRESETS: dict[str, dict] = {
    "alanine": {
        "n_rungs": 6,
        # "auto", NOT "peptide": the force-field route follows the INPUT actually supplied.
        # Alanine dipeptide can legitimately arrive either way -- as a residue-named PDB (ff19SB)
        # or as SMILES (Sage 2.2 + AM1BCC) -- and pinning "peptide" here made the preset contradict
        # a --smiles invocation, which then failed deep inside template matching.
        "solute_kind": "auto",
        "equilibration_protocol": "simple",
        "note": "small rigid solute: the unrestrained equilibration is adequate; "
                "force-field route follows the input (--pdb -> ff19SB, --smiles -> Sage 2.2)",
    },
    # RGD ONLY.  Selected by three matched 2 ns/replica pilots against the 8-rung control
    # (median worst-pair 0.140 -> 0.240, pooled minimum 0.173 -> 0.277).  Deliberately NOT applied
    # to "macrocycle": the evidence is one molecule, and a ladder that suits cyclo-RGDfV is not
    # thereby established for cyclosporin or the RGD derivatives.
    "cyclo_rgdfv": {
        "n_rungs": 10,
        # ENFORCED, not merely documented: ligand route + required SMILES input, so a PDB
        # invocation fails before any force field is built rather than quietly becoming ff19SB.
        "solute_kind": "ligand",
        "require_input_route": "smiles",
        "equilibration_protocol": "staged",
        "note": "Sage 2.2/AM1-BCC cyclo-RGDfV; 10-rung working ladder adopted from matched pilots",
    },
    "macrocycle": {
        "n_rungs": 8,
        "solute_kind": "auto",
        "equilibration_protocol": "staged",
        "note": "flexible solute from a gas-phase conformer: staged equilibration",
    },
}

WALKER_PRESETS: dict[str, dict] = {
    "cold": {"scale_factor": 1.0, "total_ns": 1000.0, "chunk_ns": 100.0, "label": "cold"},
    # 1 ns chunks: this trajectory is an AIS seed source, so a chunk should be one cBAR generation
    "hot": {"scale_factor": 0.25, "total_ns": 200.0, "chunk_ns": 1.0, "label": "hot"},
}


def simbox_config(args, preset: dict) -> dict:
    cfg: dict = {
        "run": {"seed": args.seed},
        "system": {"slug": args.slug, "solute_kind": preset["solute_kind"],
                   "require_input_route": preset.get("require_input_route")},
        "solvation": {
            "padding_nm": args.padding_nm,
            "box_shape": args.box_shape,
            "ionic_strength_molar": args.salt_molar,
        },
        "system_build": {
            "nonbonded_cutoff_nm": args.cutoff_nm,
            "hydrogen_mass_amu": args.hydrogen_mass_amu,
        },
        "protonation": {"ph": args.ph},
        "rest2": {"omega_selective": not args.no_omega_selective},
    }
    if args.water is not None:
        cfg["forcefield"] = {"water": args.water}
    return cfg


def min_eq_config(args, preset: dict) -> dict:
    return {
        "run": {"seed": args.seed},
        "integrator": {
            "kind": args.integrator,
            "temperature_k": args.temperature_k,
            "timestep_fs": args.timestep_fs,
        },
        "equilibration": {"protocol": args.equilibration or preset["equilibration_protocol"]},
        "production": {"platform": args.platform, "precision": args.precision},
    }


def md_config(args, preset: dict) -> dict:
    walker = dict(WALKER_PRESETS[args.walker])
    if args.total_ns is not None:
        walker["total_ns"] = args.total_ns
    if args.chunk_ns is not None:
        walker["chunk_ns"] = args.chunk_ns
    if args.s_hot is not None and args.walker == "hot":
        walker["scale_factor"] = args.s_hot
    return {
        "run": {"seed": args.seed},
        "integrator": {
            "kind": args.integrator,
            "temperature_k": args.temperature_k,
            "timestep_fs": args.timestep_fs,
        },
        "production": {
            "platform": args.platform,
            "precision": args.precision,
            "report": {"all_atom_ps": args.f_all_ps, "solute_ps": args.f_solute_ps},
            "md": walker,
        },
    }


def rest2_config(args, preset: dict) -> dict:
    n_rungs = args.N_rungs if args.N_rungs is not None else preset["n_rungs"]
    s_cold = args.s_cold if args.s_cold is not None else 1.0
    s_hot = args.s_hot if args.s_hot is not None else 0.25
    ladder = rest2_ladder(s_cold, s_hot, n_rungs, args.interp)
    return {
        "run": {"seed": args.seed},
        "integrator": {
            "kind": args.integrator,
            "temperature_k": args.temperature_k,
            "timestep_fs": args.timestep_fs,
        },
        "rest2": {
            "ladder": {
                "s_cold": s_cold, "s_hot": s_hot, "n_rungs": n_rungs, "interp": args.interp,
            }
        },
        "production": {
            "platform": args.platform,
            "precision": args.precision,
            "report": {"all_atom_ps": args.f_all_ps, "solute_ps": args.f_solute_ps},
            "remd": {
                # written out explicitly as well as implied by rest2.ladder, so the rungs a run
                # used are visible in its own config without re-deriving them
                "scale_factors": ladder,
                "total_ns_per_replica": (
                    args.total_ns if args.total_ns is not None else 1000.0
                ),
                "chunk_ns": args.chunk_ns if args.chunk_ns is not None else 100.0,
                "exchange_interval_ps": args.exchange_ps,
            },
        },
    }


BUILDERS = {
    "simbox": ("simbox-config.json", simbox_config),
    "min-eq": ("min-eq-config.json", min_eq_config),
    "md": ("md-config.json", md_config),
    "rest2": ("rest2-config.json", rest2_config),
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = p.add_argument_group("which config to write")
    which.add_argument("--all", action="store_true",
                       help="write every stage's config with defaults (cold AND hot md configs)")
    which.add_argument("--simbox", action="store_true", help="stage (a) simbox-setup.py")
    which.add_argument("--min-eq", action="store_true", dest="min_eq",
                       help="stage (b) min-eq.py")
    which.add_argument("--md", action="store_true", help="stage (c) md.py")
    which.add_argument("--rest2", action="store_true", help="stage (d) md_REST2.py")
    which.add_argument("--cbar", action="store_true", help="stage (e) md_cBAR.py -- NOT YET BUILT")
    which.add_argument("--prest2", action="store_true", help="stage (e) pREST2.py -- NOT YET BUILT")

    p.add_argument("--system", choices=sorted(SYSTEM_PRESETS), default="macrocycle",
                   help="per-system defaults, chiefly the ladder length (default: macrocycle)")
    p.add_argument("--slug", default=None, help="system slug recorded in every manifest")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--prefix", default="", help="prefix for the written filenames")
    p.add_argument("--seed", type=int, default=20260814)

    ladder = p.add_argument_group("REST2 ladder (--rest2)")
    ladder.add_argument("--s_cold", type=float, default=None, help="cold end (default 1.0)")
    ladder.add_argument("--s_hot", type=float, default=None, help="hot end (default 0.25)")
    ladder.add_argument("--N_rungs", type=int, default=None,
                        help="number of replicas (default: 6 alanine, 8 macrocycle, 10 cyclo_rgdfv)")
    ladder.add_argument("--interp", choices=("sqrt", "linear", "geometric"), default="sqrt",
                        help="what is spaced evenly between the ends (default: sqrt)")
    ladder.add_argument("--exchange-ps", type=float, default=10.0, dest="exchange_ps")

    run = p.add_argument_group("run settings")
    run.add_argument("--walker", choices=sorted(WALKER_PRESETS), default="cold",
                     help="--md only: which free walker (default: cold)")
    run.add_argument("--total-ns", type=float, default=None, dest="total_ns")
    run.add_argument("--chunk-ns", type=float, default=None, dest="chunk_ns")
    run.add_argument("--temperature-k", type=float, default=300.0, dest="temperature_k")
    run.add_argument("--timestep-fs", type=float, default=4.0, dest="timestep_fs")
    run.add_argument("--integrator", default="langevin-middle",
                     choices=("langevin-middle", "leapfrog-langevin", "verlet"))
    run.add_argument("--equilibration", choices=("staged", "simple"), default=None)
    run.add_argument("--f-all-ps", type=float, default=10.0, dest="f_all_ps")
    run.add_argument("--f-solute-ps", type=float, default=2.0, dest="f_solute_ps")
    run.add_argument("--platform", default="CUDA")
    run.add_argument("--precision", default="mixed")

    box = p.add_argument_group("box and force field")
    box.add_argument("--padding-nm", type=float, default=1.2, dest="padding_nm")
    box.add_argument("--cutoff-nm", type=float, default=1.0, dest="cutoff_nm")
    box.add_argument("--box-shape", default="dodecahedron", dest="box_shape",
                     choices=("dodecahedron", "cube", "octahedron"))
    box.add_argument("--salt-molar", type=float, default=0.15, dest="salt_molar")
    box.add_argument("--ph", type=float, default=7.0)
    box.add_argument("--water", default=None, help="water XML (default amber19/tip3pfb.xml)")
    box.add_argument("--hydrogen-mass-amu", type=float, default=3.024, dest="hydrogen_mass_amu")
    box.add_argument("--no-omega-selective", action="store_true",
                     help="scale the amide omega torsions too (default: leave them unscaled)")
    box.add_argument("--smiles-route", action="store_true",
                     help="documentation only: --smiles implies Sage 2.2, --pdb implies ff19SB")

    args = p.parse_args()

    for flag, name in ((args.cbar, "md_cBAR.py"), (args.prest2, "pREST2.py")):
        if flag:
            sys.exit(
                f"{name} does not exist yet, so there is no config schema to write for it.  "
                "Nothing written -- a config for a script that cannot read it would only rot."
            )

    preset = SYSTEM_PRESETS[args.system]
    if args.slug is None:
        args.slug = {"alanine": "alanine_dipeptide",
                     "cyclo_rgdfv": "cyclo_rgdfv_sage_explicit"}.get(args.system)

    selected = [k for k, f in (("simbox", args.simbox), ("min-eq", args.min_eq),
                               ("md", args.md), ("rest2", args.rest2)) if f]
    if args.all:
        selected = list(BUILDERS)
    if not selected:
        p.error("choose at least one of --all / --simbox / --min-eq / --md / --rest2")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in selected:
        filename, builder = BUILDERS[key]
        variants = [(filename, args)]
        if key == "md" and args.all:
            # --all writes both walkers, since a campaign needs each
            import copy as _copy

            variants = []
            for walker in ("cold", "hot"):
                a = _copy.copy(args)
                a.walker = walker
                variants.append((f"md-{walker}-config.json", a))
        for name, a in variants:
            payload = builder(a, preset)
            load_config_check(payload)          # fail here, not at run time
            path = args.out_dir / f"{args.prefix}{name}"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written.append(path)

    print(f"system preset '{args.system}': {preset['note']}")
    if "rest2" in selected:
        lad = rest2_config(args, preset)["production"]["remd"]["scale_factors"]
        t = [round(args.temperature_k / s) for s in lad]
        print(f"REST2 ladder ({args.interp}, {len(lad)} rungs): "
              f"{[round(x, 4) for x in lad]}")
        print(f"  T_eff on the solute (K): {t}")
        print("  acceptance along this chain is UNVERIFIED in explicit solvent -- run a short "
              "pilot and check the WORST neighbouring pair before committing a long run.")
    for path in written:
        print(f"wrote {path}")


def load_config_check(payload: dict) -> None:
    """Merge the payload into DEFAULTS so a bad key fails here rather than at run time."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        tmp = Path(fh.name)
    try:
        load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
