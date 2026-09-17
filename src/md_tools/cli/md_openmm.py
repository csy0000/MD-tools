"""`md-openmm` -- the only executable MD-tools installs.

Four public work commands, and no state carried between them: each reads files and writes files.

    md-openmm build-top        one input structure  -> built.xml + built.pdb + built.log
    md-openmm build-md         a protocol config    -> readable run scripts and .in files
    md-openmm md-run           an Amber-like .in    -> a stage, a ladder, or AIS paths
    md-openmm data-register    a finished directory -> a verified dataset under $MD_DATA

`md-run` is a SUBCOMMAND, not a second executable. It is the Amber-like way to execute what
`build-md` resolved -- `-i`, `-p`, `-c`, `-o`, `-x`, `-r`, and `-ng` under `mpirun` -- and it
delegates to the same installed runtime a generated script calls.

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

    # TWO MODES, and each refuses the other's flags by name rather than ignoring them.
    #   build-top -i STRUCTURE ...                         builds a System
    #   build-top --rest2-scaler -s SYSTEM -p PDB --config  scales a built System
    # The output paths have documented defaults, so "given" means "not the default".
    structure_flags = {"-i": args.input,
                       "-os": None if args.out_system == "./built.xml" else args.out_system,
                       "-op": None if args.out_pdb == "./built.pdb" else args.out_pdb,
                       "-log": None if args.out_log == "./built.log" else args.out_log}
    scaler_flags = {"-s": args.system, "-p": args.topology}
    if args.rest2_scaler:
        given = [flag for flag, value in structure_flags.items() if value is not None]
        missing = [flag for flag, value in scaler_flags.items() if value is None]
        if given or missing or not args.config:
            print("build-top: --rest2-scaler scales a BUILT System, so it takes -s, -p and "
                  "--config and nothing that builds one"
                  + (f"; refused: {' '.join(given)}" if given else "")
                  + (f"; missing: {' '.join(missing + ([] if args.config else ['--config']))}"
                     if missing or not args.config else ""), file=sys.stderr)
            return 2
        from ..build.scaler import build_scaled_states

        try:
            build_scaled_states(system_path=Path(args.system), topology_path=Path(args.topology),
                                config_path=Path(args.config), overwrite=bool(args.overwrite),
                                check=bool(args.check))
        except ConfigError as exc:
            print(f"build-top: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"build-top: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    given = [flag for flag, value in scaler_flags.items() if value is not None]
    if given or args.check:
        print(f"build-top: {' '.join(given + (['--check'] if args.check else []))} belong(s) to "
              f"--rest2-scaler, which scales a built System; without it build-top builds one "
              f"from -i", file=sys.stderr)
        return 2
    if args.input is None:
        print("build-top: -i INPUT is required (or --rest2-scaler -s SYSTEM -p PDB --config)",
              file=sys.stderr)
        return 2

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
# md-run
# ---------------------------------------------------------------------------------------------

def cmd_md_run(args) -> int:
    """Parsed by `md_tools.run` itself, so its Amber-like flags live with its implementation."""
    from ..run.main import md_run_main

    return md_run_main(args.md_run_argv)


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

    if args.ligand_package or args.find_ligand:
        return _ligand_catalog_command(args)

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
            project_repo=args.project_repo,
            notes=args.notes,
        )
    except RegistrationError as exc:
        print(f"data-register: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"data-register: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result["canonical_path"])
    return 0


def _ligand_catalog_command(args) -> int:
    """`data-register --ligand-package DIR` and `--find-ligand TEXT`: the parameter catalog.

    The catalog is `$MD_DATA/parameters/ligands`, beside the registered datasets and not one of
    them. Registration is write-once; an identity already present is kept, never replaced.
    """
    from ..ligands.catalog import catalog_root_for, register_package, search_catalog
    from ..ligands.package import PackageError, load_package
    from ..registry.errors import RegistrationError
    from ..registry.userconfig import load_user_config, resolve_md_data

    dataset_flags = [flag for flag, value in (("-idata", args.idata),
                                              ("-project_name", args.project_name),
                                              ("-data_name", args.data_name), ("-year", args.year),
                                              ("--common-data", args.common_data),
                                              ("--init", args.init)) if value]
    if dataset_flags or (args.ligand_package and args.find_ligand):
        print(f"data-register: --ligand-package and --find-ligand act on the ligand parameter "
              f"catalog and take no dataset options, and not each other"
              + (f" (got {', '.join(dataset_flags)})" if dataset_flags else ""), file=sys.stderr)
        return 2
    try:
        try:
            document, _, _ = load_user_config(args.user_config)
        except RegistrationError:
            if args.user_config:
                raise
            document = {}
        md_data, origin = resolve_md_data(document, override=args.md_data)
        catalog = catalog_root_for(md_data)
        if args.find_ligand:
            for hit in search_catalog(args.find_ligand, catalog):
                print(f"{hit['reference']}  {hit['canonical_smiles']}  "
                      f"{hit['forcefield']}/{hit['charge_method']}  "
                      f"aliases: {', '.join(hit['aliases']) or '-'}")
            return 0
        package = load_package(Path(args.ligand_package), expected_directory_name=False)
        destination = catalog / package.compound_id / package.parameter_id
        if args.dry_run or args.verify_only:
            if args.verify_only:
                load_package(destination)
                print(f"verified {destination}")
            else:
                print(f"would register {package.reference} at {destination} "
                      f"({'already present' if destination.exists() else 'new'}; $MD_DATA from "
                      f"{origin})")
            return 0
        package, placed, written = register_package(Path(args.ligand_package), catalog)
        print(f"{'registered' if written else 'already registered, kept'} {package.reference} "
              f"at {placed}")
        return 0
    except (PackageError, RegistrationError) as exc:
        print(f"data-register: {exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-openmm",
        description="Build OpenMM topologies, generate MD run scripts, and register finished data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            # THE LAYOUT, in the one place every user reads first. Every line here taught the
            # retired flat tree -- `-odir ./md_script/`, `-i md_script/min.in`, a bare
            # `built.pdb`, and a ladder with `-s`, which is now refused outright because each
            # rung is its own pre-scaled System named on its own group-file line.
            "examples:\n"
            "  # the system: build/ holds it, and every run beside it shares it\n"
            "  md-openmm build-top -i ALA.pdb -os build/built.xml -op build/built.pdb \\\n"
            "      -log build/built.log\n"
            "\n"
            "  # a run of its own, beside build/, min/ and input/\n"
            "  md-openmm build-md -odir ./cMD-run1 --config cMD.config\n"
            "  cd cMD-run1 && ./run.sh\n"
            "\n"
            "  # or one stage at a time: the inputs are SHARED, each stage names its own -odir\n"
            "  md-openmm md-run -i ../input/min.in -p ../build/built.pdb \\\n"
            "      -s ../build/built.xml -odir ../min\n"
            "\n"
            "  # a ladder: one rank per state, the rungs named per group-file line, NO -s\n"
            "  mpirun -n 4 md-openmm md-run -ng 4 -i ../input/REST2.in \\\n"
            "      -p ../build/built.pdb --groupfile remd_groupfile.1 -odir . \\\n"
            "      -o remd_records/REST2_prod1.out -log remd_records/REST2_prod1.log \\\n"
            "      -r remd_records/restart_prod1.json\n"
            "\n"
            "  # the registered unit is the RUN, checked by digest against what it ran against\n"
            "  md-openmm data-register -idata ./cMD-run1 -project_name ALA "
            "-data_name ALA-cMD -year 2026\n"),
    )
    parser.add_argument("--version", action="version", version=f"md-tools {_version()}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # -- build-top ----------------------------------------------------------------------------
    top = sub.add_parser(
        "build-top", help="build an OpenMM System and topology from one structure",
        description="Build one OpenMM System from one input structure. Writes a serialised "
                    "System, the matching PDB, and a readable log that carries a machine record.")
    top.add_argument("-i", "--input", default=None, metavar="INPUT",
                     help="input structure: an existing .pdb or a .seq of residue names "
                          "(peptide; tleap builds a .seq extended), or a .smi or .sdf holding "
                          "one molecule (.smi is embedded and minimised; .sdf supplies its own "
                          "coordinates)")
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
                     help="replace existing outputs instead of refusing (with --rest2-scaler, the "
                          "previous build/<method>/ is moved aside, never deleted)")
    top.add_argument("--rest2-scaler", action="store_true",
                     help="scale a BUILT System into build/<method>/system_state<n>.xml with "
                          "scaler.yaml, scaler.log and <RESNAME>-unscaled.png; takes -s, -p and "
                          "--config (a scaler.config)")
    top.add_argument("-s", "--system", default=None, metavar="XML",
                     help="with --rest2-scaler: the built System to scale")
    top.add_argument("-p", "--topology", default=None, metavar="PDB",
                     help="with --rest2-scaler: its topology")
    top.add_argument("--check", action="store_true",
                     help="with --rest2-scaler: validate everything and create nothing")
    top.set_defaults(func=cmd_build_top)

    # -- build-md -----------------------------------------------------------------------------
    md = sub.add_parser(
        "build-md", help="generate readable OpenMM run scripts",
        description="Generate small, readable run scripts that import the installed md_tools "
                    "runtime. The generated scripts contain no absolute path and no reference to "
                    "a source checkout.")
    md.add_argument("-odir", "--out-dir", required=True, metavar="DIR",
                    help="THE RUN DIRECTORY, named `<method>-run<N>` beside the dataset's "
                         "build/, min/ and input/ -- for example `-odir REST2-run1`. It must not "
                         "already exist: a second generation into a run that has output would "
                         "leave scripts and results produced by different settings side by side. "
                         "There is no default, because the old `./md_script/` put a run's scripts "
                         "in a directory that said nothing about which run it was")
    md.add_argument("--config", default=None, metavar="PATH",
                    help="protocol configuration (YAML syntax, .config suffix); the default is "
                         "explicit-solvent cMD at `timestep_fs: auto` -- resolved from the masses "
                         "in built.xml when the run starts, so it is 2 fs ordinarily and 4 fs "
                         "only when the System proves hydrogen mass repartitioning")
    md.add_argument("--overwrite", action="store_true",
                    help="replace an existing script directory instead of refusing")
    md.set_defaults(func=cmd_build_md)

    # -- md-run ------------------------------------------------------------------------------
    # Its own parser owns the flags, because they are Amber's spellings rather than this CLI's and
    # they belong beside the code that acts on them. `parse_known_args` is not enough here: `-i`,
    # `-p` and `-c` collide with nothing above only by accident today, and forwarding the argv
    # verbatim keeps that an accident that cannot start mattering.
    run = sub.add_parser(
        "md-run", help="run a stage, a REST2 ladder or AIS paths from an Amber-like input",
        add_help=False, prefix_chars="\0",
        description="Run what an Amber-like .in file describes. See `md-openmm md-run -h`.")
    run.add_argument("md_run_argv", nargs=argparse.REMAINDER,
                     help="the md-run flags: -i -p -s -c -x -r -o -log -odir -ng "
                          "-source-traj --cpu")
    run.set_defaults(func=cmd_md_run)

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
                     help="register under common/{year}/{project}/{data} instead of "
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
    reg.add_argument("--project-repo", default=None, metavar="DIR",
                     help="the project repository that produced this data, when the data are "
                          "not inside it. Normally omitted: the repository is resolved from "
                          "-idata, never from the shell's working directory.")
    reg.add_argument("--notes", default=None, metavar="TEXT",
                     help="free text stored in the manifest. The only field asserted by you "
                          "rather than derived from the run's own records, so say how you know "
                          "whatever you put here.")
    reg.add_argument("--ligand-package", default=None, metavar="DIR",
                     help="register one ligand parameter package (<compound>/<param_...>/ with "
                          "molecule.sdf, parameters.ffxml, metadata.json) into the catalog "
                          "$MD_DATA/parameters/ligands, verified and write-once; with --dry-run "
                          "or --verify-only, check without writing")
    reg.add_argument("--find-ligand", default=None, metavar="TEXT",
                     help="list catalog packages whose compound id, alias, residue name or "
                          "SMILES contains TEXT, with the reference a configuration names")
    reg.set_defaults(func=cmd_data_register)

    ref = sub.add_parser(
        "export-reference",
        help="export a finished stage as a standalone bundle that needs only OpenMM",
        description="Write a reference bundle: the System that was integrated, the topology, a "
                    "standalone run.py and run.sh, the settings read from the run's own record, "
                    "provenance, and a SHA256SUMS inventory. Nothing in the bundle imports "
                    "md_tools, so it outlives this package.")
    ref.add_argument("-idata", "--idata", required=True, metavar="DIR",
                     help="a finished run directory")
    ref.add_argument("-odir", "--odir", required=True, metavar="DIR",
                     help="where to write the bundle")
    ref.add_argument("--stage", default="cMD", metavar="NAME",
                     help="which finished stage or ladder to export (default: cMD; use REST2 "
                          "for a ladder). The bundle's shape follows the record type, not this "
                          "name.")
    ref.set_defaults(func=cmd_export_reference)



    return parser


# ------------------------------------------------------------------------------------------------
# export-reference
# ------------------------------------------------------------------------------------------------

def cmd_export_reference(args) -> int:
    """A finished run -> a bundle that runs on OpenMM alone.

    Which exporter is decided by the RECORD, not by a flag. `md-stage:*` is one Context driven
    for a fixed number of steps; `md-replica:*` is a ladder, a different control flow with a
    different bundle. Asking the caller to say which would let them say the wrong one.
    """
    from ..build.record import RecordError, read_record
    from ..reference import export_reference, export_rest2_reference

    log = Path(args.idata) / f"{args.stage}.log"
    try:
        kind = str(read_record(log).get("record_type") or "")
    except (RecordError, OSError) as refusal:
        print(f"export-reference: {refusal}", file=sys.stderr)
        return 2

    try:
        if kind.startswith("md-replica:"):
            manifest = export_rest2_reference(Path(args.idata), Path(args.odir), stage=args.stage)
            settings = manifest["settings"]
            per_rung_ns = (settings["exchange_interval_steps"] * settings["number_of_exchanges"]
                           * settings["timestep_fs"] * 1e-6)
            print(f"  exported {manifest['files']} file(s) to {args.odir}")
            print(f"  {settings['name']}: {settings['n_states']} rungs, tau "
                  f"{settings['tau'][0]:g}..{settings['tau'][-1]:g}, "
                  f"{settings['number_of_exchanges']} exchange(s), {per_rung_ns:g} ns per rung")
            print(f"  {len(manifest['vendored'])} module(s) copied verbatim into ladder/")
            print("  needs OpenMM and numpy only -- run it with ./run.sh")
            return 0
        manifest = export_reference(Path(args.idata), Path(args.odir), stage=args.stage)
    except (FileNotFoundError, ValueError, KeyError) as refusal:
        print(f"export-reference: {refusal}", file=sys.stderr)
        return 2
    settings = manifest["settings"]
    nanoseconds = settings["steps"] * settings["timestep_fs"] * 1e-6
    print(f"  exported {manifest['files']} file(s) to {args.odir}")
    print(f"  {settings['name']}: {nanoseconds:g} ns {settings['ensemble']}, "
          f"tau = {settings['tau']:g}, seed {settings['seed']}")
    print("  needs OpenMM only -- run it with ./run.sh")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
