"""Portable explicit-solvent REST2: manifests, bundles, immutable runs, and one installed CLI.

The scientific implementation lives in `escort_ais.systems.explicit_baseline`. This package is the
portability layer around it — everything needed to run the same calculation from another
repository or another machine, with provenance sufficient to prove what was run.

Entry point: `escort-explicit` (see `cli.py`).
"""
from __future__ import annotations

from .schemas import (  # noqa: F401
    SCHEMA_VERSION,
    ExperimentManifest,
    ManifestError,
    SystemManifest,
    config_hash,
    load_experiment,
    load_system,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExperimentManifest",
    "ManifestError",
    "SystemManifest",
    "config_hash",
    "load_experiment",
    "load_system",
]
