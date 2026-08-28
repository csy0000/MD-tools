"""`md-openmm` -- write configuration, build a System, generate run scripts.

Four subcommands and no state between them: each reads files and writes files.
"""
from __future__ import annotations

import argparse
import os
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
#
# Defaults, and where to change them (docs/md-defaults-scientific-rationale.md gives the evidence):
#   forcefield.protein/water   ff14SB + TIP3P is the default; ff19SB + OPC is `--solvent OPC`
#   solute.ligand_forcefield   OpenFF Sage 2.2.1, standard AM1-BCC through AmberTools sqm
#   solvent.padding_nm         1.5 nm requested solute-to-box clearance; use 2.0 for an unfolded
#                              or unusually flexible solute, or when sampling should expand it
#   solvent.cutoff_nm          1.0 nm real-space cutoff, with PME beyond it
#   constraints                HBonds + rigid water, hydrogen masses unmodified (2 fs baseline).
#                              For HMR at 4 fs see docs/examples/hmr-4fs.yaml.
"""

MD_HEADER = """\
# Simulation protocol for OpenMM. Edit this file, then run:
#     md-openmm md-gen -if ./inputs/ --config md.config.yaml -of ./MD/
#
# `common` applies to every method; each method block adds only what is specific to it.
# Durations are in the units named by each key.
#
#   common.timestep_fs                2.0 with unmodified hydrogen masses -- the baseline. 4.0 is
#                                     an explicit performance option and needs HMR in sys.config
#                                     (docs/examples/hmr-4fs.yaml); md-gen refuses it otherwise.
#   common.friction_per_ps            LangevinMiddleIntegrator collision rate, 1.0 ps^-1
#   common.barostat_frequency_steps   MonteCarloBarostat attempt interval in STEPS (OpenMM's own
#                                     default, 25); null under implicit solvent, which has none
#
# An AIS block, if present, carries null fields that are REQUIRED USER INPUT, not defaults:
# the switching duration, the source trajectory, its inclusive time window, and how many
# independent paths to run. `md-gen` names any that are still null. See docs/examples/ais-ala.yaml.
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
    if D.is_implicit(solvent):
        print(f"  solvent      : {solvent}  (implicit: no barostat, production ensembles are NVT)")
        print(f"  forcefield   : {sys_doc['forcefield']['protein']} + "
              f"{sys_doc['implicit_solvent']['model']}/{sys_doc['implicit_solvent']['radii']}, "
              f"nonpolar_sasa={sys_doc['implicit_solvent']['nonpolar_sasa']}")
    else:
        print(f"  solvent      : {solvent}")
        print(f"  forcefield   : {sys_doc['forcefield']['protein']} + "
              f"{sys_doc['forcefield']['water']}")
        print(f"  box          : {sys_doc['solvent']['box_shape']}, "
              f"{sys_doc['solvent']['padding_nm']} nm padding, "
              f"{sys_doc['solvent']['cutoff_nm']} nm cutoff (PME)")
    if not args.peptide:
        print(f"  ligand       : {sys_doc['solute']['ligand_forcefield']}, "
              f"{sys_doc['solute']['ligand_charge_method']}")
    if "AIS" in methods:
        path = md_doc["AIS"]["path"]
        print(f"  AIS          : tau {path['tau_start']} -> {path['tau_end']}, "
              f"{path['interpolation']}, "
              f"{md_doc['AIS']['output']['number_of_observations']} observations")
        print("                 fill in AIS.path.switching_duration_ps and the AIS.source block; "
              "AIS is not in MD/run_all.sh")
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


def _ask(prompt: str, default: str, choices: tuple[str, ...] = ()) -> str:
    """One interactive question. Enter takes the default, which is always shown."""
    hint = f" [{'/'.join(choices)}]" if choices else ""
    while True:
        answer = input(f"  {prompt}{hint} ({default}): ").strip() or default
        if not choices or answer.lower() in [c.lower() for c in choices]:
            return answer
        print(f"    choose one of {', '.join(choices)}")


def cmd_setup(args) -> int:
    """One small request in, a runnable OpenMM directory out."""
    from ..openmm.simple import (SetupRequest, format_preset, generate, resolve,
                                 resolve_contributor, resolve_output_root)

    interactive = not args.config
    if args.config:
        document = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    else:
        print("md-openmm setup -- press Enter to accept each default\n")
        document = {
            "input": _ask("input structure", "inputs/ALA.pdb"),
            "system": _ask("system name", "ALA"),
            "type": _ask("system type", "peptide", ("peptide", "ligand")),
            "solvent": _ask("solvent", "explicit", ("explicit", "implicit")),
            "protocol": _ask("protocol", "cMD", ("cMD",)),
            "production": _ask("production duration", "1 ns"),
            "output_interval": _ask("trajectory output interval", "5 ps"),
            "platform": _ask("platform", "automatic", ("CUDA", "CPU", "automatic")),
            "output": _ask("output location (under $MD_DATA)", os.environ.get("MD_DATA", "")),
        }

    contributor_field = document.pop("contributor", None)
    try:
        request = SetupRequest.from_document(document)
        resolved = resolve(request)
        contributor = resolve_contributor({"contributor": contributor_field},
                                          interactive=interactive and not args.yes)
        output_root, relative_project = resolve_output_root(args.output or request.output)
    except ConfigError as error:
        raise SystemExit(f"setup: {error}")

    if args.advanced:
        print("\nAdvanced settings are set through `advanced:` in the request, dotted by path:\n")
        for line in ("advanced:",
                     "  solvent.padding_nm: 1.2",
                     "  common.timestep_fs: 4.0",
                     "  constraints.hydrogen_mass_amu: 4.0     # 4 fs is refused without this",
                     "  common.temperature_kelvin: 310",
                     "  common.random_seed: 20260828",
                     "",
                     "ff19SB/OPC is a COUPLED selection -- the protein and water force fields must",
                     "change together, and setting only the water XML would pair ff14SB with OPC:",
                     "  forcefield.protein: amber19-all.xml",
                     "  forcefield.water: amber19/opc.xml"):
            print(f"  {line}")
        print()

    print("\nResolved preset:\n")
    print(format_preset(resolved))
    print(f"\n  contributor   {contributor['name']}"
          + (f" <{contributor['email']}>" if contributor.get("email") else ""))
    print(f"  output        {output_root / resolved['system_id']}")
    print(f"  portable as   $MD_DATA/{relative_project}/{resolved['system_id']}\n")

    # `--yes` is what makes a run unattended. Without it BOTH modes ask, config-driven included:
    # a config file says what to build, not that nobody wants to see it first.
    if not args.yes:
        if input("  generate? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  nothing written")
            return 0

    try:
        result = generate(resolved, input_path=Path(request.input), output_root=output_root,
                          relative_project=relative_project, contributor=contributor,
                          overwrite=args.overwrite)
    except ConfigError as error:
        raise SystemExit(f"setup: {error}")

    print(f"\n  wrote {result['system_dir']}")
    for name in result["written"]:
        print(f"    {name}")
    print(f"\n  run every stage:  cd {result['system_dir']} && ./run.sh")
    print(f"  or one at a time: {result['system_dir']}/min/min.sh")
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
    out = Path(args.output_folder)
    print()
    print("  common stages, in the order they depend on each other:")
    for name in result["common_stages"]:
        print(f"    {out / name}")
    print("  production, both branching from the last common stage:")
    for method in result["methods"]:
        print(f"    {out / method}")
    print()
    print("  run every stage in order:")
    print(f"    cd {out} && ./run_all.sh")
    print("  or one stage at a time, which is the authoritative way:")
    print(f"    cd {out / result['common_stages'][0]} && ./run.sh")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-openmm",
        description="Write OpenMM configuration, build a system, generate run scripts.")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("sys-config", help="write sys.config.yaml and md.config.yaml")
    config.add_argument("--method", nargs="+", default=["cMD", "REST2"],
                        help="any of cMD, REST2, AIS (case-insensitive). AIS anneals the REST2 "
                             "Hamiltonian from tau=0.5 to tau=0 and records the nonequilibrium "
                             "work; it starts from an equilibrium trajectory you already ran, so "
                             "it is not part of MD/run_all.sh.")
    config.add_argument("--peptide", type=_bool, default=True, help="true or false")
    config.add_argument(
        "--solvent", default=D.DEFAULT_SOLVENT,
        help="TIP3P (default: ff14SB + Sage 2.2.1 + TIP3P), OPC (ff19SB + Sage 2.2.1 + OPC) or "
             "GBn2 (implicit: ff14SB + GBn2/mbondi3, no SASA). Case-insensitive.")
    config.add_argument("--output-dir", "--output_dir", dest="output_dir", default=".")
    config.set_defaults(func=cmd_sys_config)

    show = sub.add_parser("show-default", help="print a default configuration block")
    show.add_argument("name",
                      help="sys, dataset, cMD, REST2, AIS or all (case-insensitive). AIS prints "
                           "the annealed-importance-sampling block: the tau path, the "
                           "source-trajectory contract and the observation schedule. dataset "
                           "prints the MD-data identity block on its own. In both, a null field "
                           "is required user input, not a default -- an identity or a pinned "
                           "commit this package invented would be a fabrication, not a "
                           "convenience.")
    show.set_defaults(func=cmd_show_default)

    setup = sub.add_parser(
        "setup", help="one small request -> a runnable OpenMM directory",
        description="Resolve a small request against this repository's validated defaults and "
                    "write a runnable OpenMM directory: one readable script per stage, run "
                    "directly, writing a .out beside it.")
    setup.add_argument("--config", help="setup.yaml; omit for the interactive prompts")
    setup.add_argument("--output", help="storage root; defaults to $MD_DATA")
    setup.add_argument("--advanced", action="store_true",
                       help="show how to override force field, box, timestep and the rest")
    setup.add_argument("--overwrite", action="store_true",
                       help="allow generating over a non-empty system directory")
    setup.add_argument("-y", "--yes", action="store_true",
                   help="do not ask before writing; required for unattended use")
    setup.set_defaults(func=cmd_setup)

    sysgen = sub.add_parser("sys-gen", help="build the OpenMM system (advanced; `setup` is the "
                                            "normal entry point)")
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
