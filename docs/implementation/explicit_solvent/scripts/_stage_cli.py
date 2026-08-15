"""Shared argument parsing for the explicit-solvent stage scripts.

Every stage takes ``--config <stage>-config.json`` (only the values that differ from the baseline;
unknown keys are rejected) and ``--out-suffix``, which names its outputs.  Stages after the first
also take ``--p`` (the parameterised System) and ``--c`` (coordinates).

No scientific logic lives in this folder; it is all in ``escort_ais.systems.explicit_baseline``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from escort_ais.systems.explicit_baseline import load_config


def stage_parser(description: str, *, needs_input: bool = False,
                 needs_topology: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None, metavar="CONFIG.json",
                   help="JSON of NON-DEFAULT values; make one with generate_config.py")
    p.add_argument("--out-suffix", required=True, metavar="SUFFIX",
                   help="every output of this stage is named <SUFFIX>_*")
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="where those outputs go (default: the current directory)")
    if needs_input:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--smiles", type=str, help="SMILES -> ETKDGv3 + MMFF94s + Sage 2.2")
        g.add_argument("--pdb", type=Path, help="structure as given -> ff19SB (peptide route)")
    if needs_topology:
        p.add_argument("--p", dest="topology", type=Path, required=True, metavar="SYSTEM.xml",
                       help="parameterised System from simbox-setup.py; its sibling "
                            "<stem>_topology.pdb and <stem>_simbox.json are read alongside")
        p.add_argument("--c", dest="coords", type=Path, required=True, metavar="COORDS",
                       help="coordinates: a .pdb, or a serialised OpenMM State .xml "
                            "(which also carries velocities)")
    return p


def resolve(args) -> tuple[dict, Path]:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir
