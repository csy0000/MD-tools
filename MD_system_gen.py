#!/usr/bin/env python3
"""Prepare a molecular system. Stop before any dynamics.

    python MD_system_gen.py -i ALA.pdb -o alanine_dipeptide_system --config system_config.json

This is the molecular front end. It reads a structure, decides the parameterisation route,
assigns force fields, solvates, ionises, builds the box and writes a portable, immutable system
bundle. It does **not** minimise, equilibrate, or run dynamics of any kind -- that is
`MD_input_gen.py`'s territory, and the separation is the point:

    MD_system_gen.py   WHAT the molecule is       -> a system bundle
    MD_input_gen.py    WHAT is done to it         -> stage directories and launchers

Keeping them apart means a protocol can be regenerated without re-deriving AM1-BCC charges (27
minutes for a macrocycle), and a system can be reused by several protocols without any chance of one
of them quietly re-solvating it.

The script is a thin entry point. Every scientific decision is made by `md_templates.openmm`; this
file resolves arguments, refuses ambiguous input, and calls the package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "src" / "md_templates").is_dir():          # running from a checkout
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Extensions we accept, and the input route each implies. The extension chooses the READER; it
#: does not decide what kind of system this is -- a PDB may be a ligand, a protein, or a complex,
#: and guessing that from a filename is how a peptide silently becomes a small-molecule run.
_FORMAT_BY_SUFFIX = {
    ".pdb": "pdb",
    ".mol": "mol",
    ".mol2": "mol",
    ".sdf": "mol",
    ".smi": "smi",
    ".smiles": "smi",
}

#: System types this pipeline can currently prepare.
_SUPPORTED_TYPES = ("ligand", "protein", "protein-ligand")

#: What a SMILES input must state explicitly. None of these can be inferred from a SMILES string
#: without choosing chemistry on the user's behalf.
_REQUIRED_SMI_FIELDS = (
    "formal_charge",
    "stereochemistry_policy",
    "protonation_policy",
    "conformer_generation",
    "charge_model",
    "parameterization_route",
)


class InputError(SystemExit):
    """A usage error, reported without a traceback."""

    def __init__(self, message: str):
        super().__init__(f"MD_system_gen: {message}")


def detect_format(path: Path) -> str:
    """Reader format from the extension. Never the system type."""
    suffix = path.suffix.lower()
    if suffix not in _FORMAT_BY_SUFFIX:
        raise InputError(
            f"unsupported input extension {suffix!r}. Supported: "
            f"{', '.join(sorted(_FORMAT_BY_SUFFIX))}"
        )
    return _FORMAT_BY_SUFFIX[suffix]


def classify_system(fmt: str, config: dict, input_path: Path) -> str:
    """Decide ligand / protein / protein-ligand, or refuse.

    A declared `system.type` always wins. Only the unambiguous cases are inferred: a MOL/SDF or
    SMILES input is a single small molecule by construction. A PDB is genuinely ambiguous -- it may
    hold a peptide, a ligand, or both -- so it must be declared.
    """
    declared = (config.get("system") or {}).get("type")
    if declared is not None:
        if declared not in _SUPPORTED_TYPES:
            raise InputError(
                f"system.type {declared!r} is not supported. Supported: "
                f"{', '.join(_SUPPORTED_TYPES)}"
            )
        return declared

    if fmt in ("mol", "smi"):
        return "ligand"          # a single small molecule, by construction of the format

    raise InputError(
        f"{input_path.name} is a PDB, and a PDB does not say what kind of system it holds -- it may "
        "be a peptide, a ligand, or a complex. Declare it explicitly:\n"
        '    "system": {"type": "protein"}      (or "ligand", or "protein-ligand")\n'
        "The extension chooses the reader, never the chemistry: inferring the type here is how a "
        "peptide silently becomes a small-molecule parameterisation under the same name."
    )


def require_smi_build_fields(config: dict) -> dict:
    """A SMILES input must state its build decisions; none of them are guessable."""
    ligand = (config.get("ligand_build") or {})
    missing = [f for f in _REQUIRED_SMI_FIELDS if f not in ligand]
    if missing:
        raise InputError(
            "a SMILES input requires an explicit ligand_build block; missing: "
            f"{', '.join(missing)}.\n"
            "None of these can be inferred from a SMILES string without choosing chemistry on your "
            "behalf -- protonation state and formal charge in particular change the molecule.\n"
            'Example:\n'
            '    "ligand_build": {\n'
            '        "formal_charge": 0,\n'
            '        "stereochemistry_policy": "from_smiles",\n'
            '        "protonation_policy": "as_given",\n'
            '        "conformer_generation": "etkdgv3",\n'
            '        "charge_model": "am1bcc",\n'
            '        "parameterization_route": "openff-2.2.0"\n'
            "    }"
        )
    return ligand


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MD_system_gen.py",
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Examples:\n"
            "  python MD_system_gen.py -i ALA.pdb -o alanine_system --config system_config.json\n"
            "  python MD_system_gen.py -i ligand.smi -o lig_system --config system_config.json\n"
            "\n"
            "This command never minimises or runs dynamics. Use MD_input_gen.py for protocols."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="molecular input: .pdb, .mol/.mol2/.sdf, or .smi/.smiles")
    # -o, -odir and --outdir are aliases for the same destination.
    parser.add_argument("-o", "-odir", "--outdir", dest="outdir", required=True, metavar="DIR",
                        help="destination directory for the prepared system bundle")
    parser.add_argument("--config", required=True, metavar="JSON",
                        help="system_config.json: chemistry and system-building choices only")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a non-empty destination (refused by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate input, config and routing, then stop without building")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise InputError(f"input not found: {input_path}")
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise InputError(f"config not found: {config_path}")
    config = json.loads(config_path.read_text())

    fmt = detect_format(input_path)
    system_type = classify_system(fmt, config, input_path)
    ligand_build = require_smi_build_fields(config) if fmt == "smi" else None

    outdir = Path(args.outdir).resolve()
    # The destination check itself lives in the library, so it also protects callers that never go
    # through this entry point, and so --overwrite means the same thing in both generators.

    print(f"  input        : {input_path.name}  (format {fmt})")
    print(f"  system type  : {system_type}")
    print(f"  destination  : {outdir}")
    if ligand_build:
        print(f"  ligand build : {ligand_build.get('parameterization_route')} / "
              f"{ligand_build.get('charge_model')}, formal charge "
              f"{ligand_build.get('formal_charge')}")

    # A dry run validates the invocation as given, and an occupied destination is a property of the
    # invocation. Reporting it here is the whole point: parameterisation can cost half an hour, and
    # nobody wants to spend it and then be told the destination was never writable.
    from md_templates.openmm.destination import (DestinationExists, SYSTEM_BUNDLE_TARGETS,
                                                 check_destination)

    try:
        check_destination(outdir, SYSTEM_BUNDLE_TARGETS,
                          overwrite=args.overwrite, what="system bundle")
    except DestinationExists as error:
        raise InputError(str(error))

    if args.dry_run:
        print("  dry run: input, configuration and routing validated; nothing was built.")
        return 0

    from md_templates.openmm import system_prep      # imported late: heavy scientific deps
    try:
        result = system_prep.prepare_system(
            input_path=input_path,
            input_format=fmt,
            system_type=system_type,
            config=config,
            outdir=outdir,
            overwrite=args.overwrite,
        )
    except DestinationExists as error:
        raise InputError(str(error))

    print(f"  system bundle: {result['bundle_dir']}")
    print(f"  manifest     : {result['system_manifest']}")
    print(f"  solute atoms : {result['n_solute_atoms']}  particles {result['n_particles']}")
    print("  NOT minimised or equilibrated -- generate a protocol with MD_input_gen.py")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    try:
        raise SystemExit(main())
    except InputError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
