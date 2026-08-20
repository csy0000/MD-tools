#!/usr/bin/env python3
"""Prepare a molecular system for OpenMM. Runs no dynamics.

A thin shim. The implementation lives in `md_templates.openmm.cli_system_gen` so that it ships in the
wheel and can be run from an installed package with no checkout in sight -- these entry points are
public, and a public entry point that only exists in a source tree is not installable.

Running this file from a checkout uses the checkout's sources; running the console script
`md-system-gen` uses the installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "src" / "md_templates").is_dir():          # running from a checkout
    sys.path.insert(0, str(REPO_ROOT / "src"))

from md_templates.openmm.cli_system_gen import InputError, main  # noqa: E402

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
