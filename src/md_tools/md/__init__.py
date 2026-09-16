"""Ordinary molecular dynamics: the stable API a generated stage script calls.

A generated script is an entry point, not an implementation. It says which stage it is and where it
lives, and everything else — restraints, integrator, barostat, platform, reporting, restart — is
here, in the installed package:

    #!/usr/bin/env python
    from md_tools.md import run_generated_stage
    raise SystemExit(run_generated_stage(__file__, "min"))

That shape is deliberate. A generated file that carried the implementation was a *copy*: fixing a
defect meant regenerating every project ever produced, and two projects generated a month apart ran
different code while claiming the same protocol. Now the run scripts are small enough to read in
one glance and the behaviour has one version — the installed one, recorded in every log.

`resolved.config` beside the script is the single resolved declaration of the workflow. The helpers
find it relative to ``__file__``, so a generated directory can be moved anywhere and still runs,
and it is strictly validated at execution and bound into the checkpoint fingerprint.
"""
from __future__ import annotations

from .phase_space import PhaseSpaceReader, PhaseSpaceWriter
from .reporting import ReportingConfig
from .restraints import PositionalRestraint
from .simulation import (active_barostat_count, add_barostat, count_barostats, derive_seed,
                         resolve_platform)
from .stage import run_generated_stage, run_stage

__all__ = [
    # what a generated script calls
    "run_generated_stage",
    "run_stage",
    # what a caller composing a run by hand needs
    "PositionalRestraint",
    "ReportingConfig",
    # shared with md_tools.remd, so a ladder and a stage cannot disagree about these
    "resolve_platform",
    "derive_seed",
    "add_barostat",
    "count_barostats",
    "active_barostat_count",
    "PhaseSpaceReader",
    "PhaseSpaceWriter",
]
