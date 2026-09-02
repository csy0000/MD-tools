"""The Amber-like execution surface: `md-openmm md-run`.

    md_tools.run.parse_run_input   read one `.in` file and resolve it, refusing what it cannot
    md_tools.run.md_run_main       the subcommand's body

A surface over the runtime, never a second copy of it. Each protocol is handed to the same
function a generated script calls, so a run started here and a run started from `md_script/` are
the same run.
"""
from __future__ import annotations

from .inputs import RunInput, parse_run_input
from .main import md_run_main, md_run_parser

__all__ = ["RunInput", "parse_run_input", "md_run_main", "md_run_parser"]
