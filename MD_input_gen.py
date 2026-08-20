#!/usr/bin/env python3
"""Generate a staged simulation protocol from a prepared system.

    python MD_input_gen.py --system alanine_dipeptide_system/system_manifest.json \\
                           -o alanine_run --config md_config.json

This is the protocol front end. It consumes a system bundle that `MD_system_gen.py` prepared,
verifies its checksums, resolves one canonical MD configuration, and writes an inspectable project:

    alanine_run/
        inputs/            immutable references to the prepared system
        min/               min.json + min.sh
        eq_nvt/            eq_nvt.json + eq_nvt.sh
        eq_npt/            eq_npt.json + eq_npt.sh
        cMD_1/             cMD_1.json + cMD_1.sh
        REST2_1/           REST2_1.json + REST2_1.sh
        run_all.sh         runs each stage's launcher in order
        run_manifest.json  machine-readable description of what was generated
        run.log            what has actually been executed

What it never does: reparameterise the molecule, resolvate it, change a force field, or rebuild the
system. Those decisions belong to the bundle and are already fixed; touching them here would mean a
protocol silently changed the chemistry it claims to be simulating.

It also does not run production. Generation is cheap and GPU-free; execution is the launchers' job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "src" / "md_templates").is_dir():
    sys.path.insert(0, str(REPO_ROOT / "src"))


class InputError(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"MD_input_gen: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MD_input_gen.py",
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Examples:\n"
            "  python MD_input_gen.py --system sys/system_manifest.json -o run --config md.json\n"
            "  python MD_input_gen.py --system sys/system_manifest.json -o run --config md.json \\\n"
            "                         --dry-run          # validate only, no files written\n"
            "\n"
            "--inherit records LINEAGE from a previous generated run. It is not a checkpoint\n"
            "resume: continuing a REST2 run happens inside that run's own directory, through the\n"
            "committed-generation contract, not by regenerating a project."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system", required=True, metavar="MANIFEST",
                        help="system_manifest.json from MD_system_gen.py")
    parser.add_argument("-o", "-odir", "--outdir", dest="outdir", required=True, metavar="DIR",
                        help="destination project directory")
    parser.add_argument("--config", required=True, metavar="JSON",
                        help="md_config.json: protocol, reporting and execution choices only")
    parser.add_argument("--inherit", default=None, metavar="RUN_MANIFEST",
                        help="run_manifest.json of a previous generated run, for lineage only")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a non-empty destination (refused by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and validate everything, write nothing; needs no GPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    manifest_path = Path(args.system).resolve()
    if not manifest_path.is_file():
        raise InputError(f"system manifest not found: {manifest_path}")
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise InputError(f"config not found: {config_path}")

    outdir = Path(args.outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise InputError(
            f"destination {outdir} exists and is not empty. Refusing to write into it: a project "
            "half-overwritten with stages from two different configurations would run without "
            "complaint and mean nothing. Use --overwrite, or choose another -o."
        )

    inherit_path = Path(args.inherit).resolve() if args.inherit else None
    if inherit_path is not None and not inherit_path.is_file():
        raise InputError(f"--inherit manifest not found: {inherit_path}")

    from md_templates.openmm import input_gen

    result = input_gen.generate_project(
        system_manifest=manifest_path,
        md_config=json.loads(config_path.read_text()),
        outdir=outdir,
        inherit=inherit_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print(f"  system      : {result['system_id']}  ({result['n_solute_atoms']} solute atoms)")
    print(f"  stages      : {', '.join(result['stages'])}")
    for line in result["summary"]:
        print(f"    {line}")
    if args.dry_run:
        print("  dry run: configuration resolved and validated; no files were written.")
        return 0
    print(f"  project     : {outdir}")
    print(f"  run it with : cd {outdir} && ./run_all.sh")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    try:
        raise SystemExit(main())
    except InputError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
