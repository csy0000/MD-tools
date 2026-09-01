"""`md-openmm` -- the only executable MD-tools installs.

Three public work commands, and no state carried between them: each reads files and writes files.

    md-openmm build-top        one input structure  -> built.xml + built.pdb + built.log
    md-openmm build-md         a protocol config    -> readable run scripts in ./md_script/
    md-openmm data-register    a finished directory -> a verified dataset under $MD_DATA

The option spellings, including the single-dash multi-character ones (`-os`, `-op`, `-log`,
`-odir`, `-idata`, `-project_name`, `-data_name`, `-year`), are contractual: they are what the
documented examples type, so they are tested rather than left to argparse's prefix matching.
Double-dash aliases exist alongside them where they read better in a script.

`-h` must work on a machine with no CUDA and no OpenMM context. Nothing in this module imports
OpenMM at parser-construction time; the heavy imports live inside the command functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("md-tools")
    except PackageNotFoundError:                      # pragma: no cover - source checkout only
        return "0.0.0+unknown"


# ---------------------------------------------------------------------------------------------
# build-top
# ---------------------------------------------------------------------------------------------

def cmd_build_top(args) -> int:
    from ..build.strict import ConfigError
    from ..build.top import build_topology

    try:
        build_topology(
            input_path=Path(args.input),
            config_path=Path(args.config) if args.config else None,
            out_system=Path(args.out_system),
            out_pdb=Path(args.out_pdb),
            out_log=Path(args.out_log),
            overwrite=bool(args.overwrite),
        )
    except ConfigError as exc:
        print(f"build-top: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                          # the log has already recorded the failure
        print(f"build-top: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------
# build-md
# ---------------------------------------------------------------------------------------------

def cmd_build_md(args) -> int:
    from ..build.md import build_scripts
    from ..build.strict import ConfigError

    try:
        build_scripts(
            config_path=Path(args.config) if args.config else None,
            out_dir=Path(args.out_dir),
            all_in_one=bool(args.all_in_one),
            overwrite=bool(args.overwrite),
        )
    except ConfigError as exc:
        print(f"build-md: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"build-md: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------
# data-register
# ---------------------------------------------------------------------------------------------

def cmd_data_register(args) -> int:
    from ..registry.errors import RegistrationError
    from ..registry.register import register_dataset
    from ..registry.userconfig import init_user_config

    if args.init:
        try:
            path = init_user_config(explicit=args.user_config, force=bool(args.force),
                                    noninteractive=bool(args.noninteractive),
                                    name=args.name, person_id=args.person_id,
                                    orcid=args.orcid, md_data=args.md_data)
        except RegistrationError as exc:
            print(f"data-register --init: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    missing = [flag for flag, value in (("-idata", args.idata),
                                        ("-project_name", args.project_name),
                                        ("-data_name", args.data_name),
                                        ("-year", args.year)) if not value]
    if missing:
        print(f"data-register: missing required option(s) {', '.join(missing)}. "
              f"Use `md-openmm data-register -h`, or `--init` to create the user configuration.",
              file=sys.stderr)
        return 2

    try:
        result = register_dataset(
            source=Path(args.idata),
            project_name=args.project_name,
            data_name=args.data_name,
            year=args.year,
            common=bool(args.common_data),
            dry_run=bool(args.dry_run),
            verify_only=bool(args.verify_only),
            user_config=args.user_config,
            md_data_override=args.md_data,
        )
    except RegistrationError as exc:
        print(f"data-register: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"data-register: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result["canonical_path"])
    return 0


# ---------------------------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-openmm",
        description="Build OpenMM topologies, generate MD run scripts, and register finished data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log\n"
            "  md-openmm build-md -odir ./md_script/ --config cMD.config\n"
            "  md-openmm data-register -idata ./data/ALA -project_name ALA "
            "-data_name ALA-cMD -year 2026\n"),
    )
    parser.add_argument("--version", action="version", version=f"md-tools {_version()}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # -- build-top ----------------------------------------------------------------------------
    top = sub.add_parser(
        "build-top", help="build an OpenMM System and topology from one structure",
        description="Build one OpenMM System from one input structure. Writes a serialised "
                    "System, the matching PDB, and a readable log that carries a machine record.")
    top.add_argument("-i", "--input", required=True, metavar="INPUT",
                     help="input structure: an existing .pdb (peptide) or .smi (single molecule)")
    top.add_argument("-os", "--out-system", default="./built.xml", metavar="PATH",
                     help="serialised OpenMM System (default: ./built.xml)")
    top.add_argument("-op", "--out-pdb", default="./built.pdb", metavar="PATH",
                     help="final coordinates and topology (default: ./built.pdb)")
    top.add_argument("-log", "--out-log", default="./built.log", metavar="PATH",
                     help="readable build log with the machine record (default: ./built.log)")
    top.add_argument("--config", default=None, metavar="PATH",
                     help="build configuration (YAML syntax, .config suffix); built-in defaults "
                          "are used when omitted")
    top.add_argument("--overwrite", action="store_true",
                     help="replace existing outputs instead of refusing")
    top.set_defaults(func=cmd_build_top)

    # -- build-md -----------------------------------------------------------------------------
    md = sub.add_parser(
        "build-md", help="generate readable OpenMM run scripts",
        description="Generate small, readable run scripts that import the installed md_tools "
                    "runtime. The generated scripts contain no absolute path and no reference to "
                    "a source checkout.")
    md.add_argument("-odir", "--out-dir", default="./md_script/", metavar="DIR",
                    help="directory to write the scripts into (default: ./md_script/)")
    md.add_argument("--config", default=None, metavar="PATH",
                    help="protocol configuration (YAML syntax, .config suffix); the default is "
                         "explicit-solvent cMD at a 2 fs timestep")
    md.add_argument("--all-in-one", action="store_true",
                    help="emit a single md.py running every stage, instead of one script per "
                         "stage; the resolved settings and stage boundaries are identical")
    md.add_argument("--overwrite", action="store_true",
                    help="replace an existing script directory instead of refusing")
    md.set_defaults(func=cmd_build_md)

    # -- data-register ------------------------------------------------------------------------
    reg = sub.add_parser(
        "data-register", help="verify a finished data directory and register it under $MD_DATA",
        description="Transactionally register one finished data directory into the managed "
                    "storage root, then replace it locally with a symlink to the verified "
                    "destination. The source is preserved unless every check passes.")
    reg.add_argument("-idata", "--idata", default=None, metavar="DIR",
                     help="the finished local data directory to register")
    reg.add_argument("-project_name", "--project-name", default=None, metavar="NAME",
                     help="project segment of the canonical path; one safe path segment")
    reg.add_argument("-data_name", "--data-name", default=None, metavar="NAME",
                     help="dataset segment of the canonical path; one safe path segment")
    reg.add_argument("-year", "--year", default=None, metavar="YYYY",
                     help="four digits; the year the data were COMPLETED (see the migration note)")
    reg.add_argument("--common-data", action="store_true",
                     help="register under {year}/common/{project}/{data} instead of "
                          "{year}/{project}/{data}")
    reg.add_argument("--dry-run", action="store_true",
                     help="run every check and print the destination, writing nothing at all")
    reg.add_argument("--verify-only", action="store_true",
                     help="verify an already-registered destination against its inventory")
    reg.add_argument("--init", action="store_true",
                     help="create the user configuration (identity and $MD_DATA) and exit")
    reg.add_argument("--user-config", default=None, metavar="PATH",
                     help="explicit user configuration path; highest precedence")
    reg.add_argument("--md-data", default=None, metavar="DIR",
                     help="override the managed storage root for this invocation")
    reg.add_argument("--force", action="store_true",
                     help="with --init, overwrite an existing configuration")
    reg.add_argument("--noninteractive", action="store_true",
                     help="with --init, never prompt; every value must be supplied by flag")
    reg.add_argument("--name", default=None, help="with --init, the user's name")
    reg.add_argument("--person-id", default=None, help="with --init, a stable person identifier")
    reg.add_argument("--orcid", default=None, help="with --init, an ORCID, or omit for none")
    reg.set_defaults(func=cmd_data_register)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
