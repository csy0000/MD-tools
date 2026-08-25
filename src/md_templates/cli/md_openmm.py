"""`md-openmm` -- write configuration, build a System, generate run scripts.

Four subcommands and no state between them: each reads files and writes files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..openmm import defaults as D
from ..openmm.config import ConfigError, write_yaml

SYS_HEADER = """\
# System preparation for OpenMM. Edit this file, then run:
#     md-openmm sys-gen -i <structure> --config sys.config.yaml -of ./inputs/
#
# Only the solvent block that applies is present: an explicit water box (solvent:) or an implicit
# GB model (implicit_solvent:), never both.
"""

MD_HEADER = """\
# Simulation protocol for OpenMM. Edit this file, then run:
#     md-openmm md-gen -if ./inputs/ --config md.config.yaml -of ./MD/
#
# `common` applies to every method; each method block adds only what is specific to it.
# Durations are in the units named by each key.
"""


def _bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in ("true", "yes", "1", "t"):
        return True
    if text in ("false", "no", "0", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def cmd_sys_config(args) -> int:
    try:
        methods = [D.canonical_method(m) for m in args.method]
        solvent = D.canonical_solvent(args.solvent)
    except ValueError as error:
        raise SystemExit(str(error))

    out = Path(args.output_dir).resolve()
    sys_doc = D.sys_defaults(peptide=args.peptide, solvent=solvent)
    md_doc = D.md_defaults(methods=methods, solvent=solvent)

    sys_path = write_yaml(out / "sys.config.yaml", sys_doc, header=SYS_HEADER)
    md_path = write_yaml(out / "md.config.yaml", md_doc, header=MD_HEADER)

    print(f"  methods      : {', '.join(methods)}")
    print(f"  solute       : {'peptide' if args.peptide else 'non-peptide'}")
    print(f"  solvent      : {solvent}"
          + ("  (implicit: no barostat, production ensembles are NVT)"
             if D.is_implicit(solvent) else ""))
    print(f"  wrote        : {sys_path}")
    print(f"                 {md_path}")
    print()
    print("Edit those, then:")
    print("    md-openmm sys-gen -i <structure> --config sys.config.yaml -of ./inputs/")
    return 0


def cmd_show_default(args) -> int:
    try:
        document = D.default_document(args.name)
    except ValueError as error:
        raise SystemExit(str(error))
    sys.stdout.write(yaml.safe_dump(document, sort_keys=False, default_flow_style=False,
                                    width=88))
    return 0


def cmd_sys_gen(args) -> int:
    from ..openmm.sysgen import generate_system

    try:
        result = generate_system(input_path=Path(args.input), config_path=Path(args.config),
                                 output_folder=Path(args.output_folder))
    except (ConfigError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"sys-gen: {error}")
    print()
    print(f"  {result['n_particles']} particles, {result['n_solute_atoms']} in the solute")
    print(f"  next: md-openmm md-gen -if {args.output_folder} --config md.config.yaml -of ./MD/")
    return 0


def cmd_md_gen(args) -> int:
    from ..openmm.mdgen import generate_md

    try:
        result = generate_md(input_folder=Path(args.input_folder), config_path=Path(args.config),
                             output_folder=Path(args.output_folder))
    except (ConfigError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"md-gen: {error}")
    print()
    for method in result["methods"]:
        print(f"  {method:6} -> {Path(args.output_folder) / method}")
    print(f"  run with: cd {Path(args.output_folder) / result['methods'][0]} && ./run.sh")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-openmm",
        description="Write OpenMM configuration, build a system, generate run scripts.")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("sys-config", help="write sys.config.yaml and md.config.yaml")
    config.add_argument("--method", nargs="+", default=["cMD", "REST2"],
                        help="cMD and/or REST2 (case-insensitive)")
    config.add_argument("--peptide", type=_bool, default=True, help="true or false")
    config.add_argument("--solvent", default="OPC", help="OPC or GBn2 (case-insensitive)")
    config.add_argument("--output-dir", "--output_dir", dest="output_dir", default=".")
    config.set_defaults(func=cmd_sys_config)

    show = sub.add_parser("show-default", help="print a default configuration block")
    show.add_argument("name", help="sys, cMD, REST2 or all (case-insensitive)")
    show.set_defaults(func=cmd_show_default)

    sysgen = sub.add_parser("sys-gen", help="build the OpenMM system")
    sysgen.add_argument("-i", "--input", required=True,
                        help="structure (.pdb) or a file containing a SMILES string")
    sysgen.add_argument("--config", required=True, help="sys.config.yaml")
    sysgen.add_argument("-of", "--output-folder", "--output_folder", dest="output_folder",
                        required=True)
    sysgen.set_defaults(func=cmd_sys_gen)

    mdgen = sub.add_parser("md-gen", help="generate run scripts")
    mdgen.add_argument("-if", "--input-folder", "--input_folder", dest="input_folder",
                       required=True, help="the folder sys-gen wrote")
    mdgen.add_argument("--config", required=True, help="md.config.yaml")
    mdgen.add_argument("-of", "--output-folder", "--output_folder", dest="output_folder",
                       required=True)
    mdgen.set_defaults(func=cmd_md_gen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
