"""Explicit-water molecular dynamics and omega-selective REST2 on OpenMM.

The pipeline is four stages, each resumable and each writing its own provenance:

    build_simbox           SMILES/PDB -> addHydrogens(pH 7) -> solvation -> a System with PME,
                           HBonds and hydrogen mass repartitioning
    minimize_equilibrate   staged: restrained minimisation, 50->300 K ramp, NPT with the
                           restraint released in steps, free NPT whose tail fixes the box
    run_md                 one free walker at a fixed scale factor, in chunks
    run_rest2_remd         N replicas, neighbour exchange, chunked per replica

Modules, in dependency order -- each imports only from those above it, so the chain is acyclic:

    config          defaults, manifest resolution, run manifests, the REST2 ladder rule
    system          structure generation, force fields, omega classification, System building,
                    and the REST2 Hamiltonian scaling
    solvation       box geometry and solvation
    equilibration   integrators, the prepared box, and the staged equilibration
    md              chunked conventional MD
    rest2           replica exchange: neighbour schedule, Metropolis criterion, ladder validation

    schemas / bundle / fingerprint / provenance / runner / envcheck / cli
                    the portability layer: hash-checked manifests, transportable bundles,
                    run directories, and the installed `md-openmm` command.

Units: nm, ps, kJ/mol, kelvin, amu (OpenMM's MD unit system). ``s = T_base / T_eff in (0, 1]``
is the REST2 convention used throughout; ``s = 1`` is cold.
"""
from __future__ import annotations

from .config_legacy import (  # noqa: F401
    DEFAULTS,
    dump_defaults,
    exchange_rounds,
    load_config,
    resolve_config,
    rest2_ladder,
    write_manifest,
)
from .equilibration import build_simbox, make_integrator, minimize_equilibrate  # noqa: F401
from .md import completed_prefix, run_md  # noqa: F401
from .rest2 import (  # noqa: F401
    attempt_rest2_exchange,
    exchange_pairs,
    resolve_remd_scale_ladder,
    run_rest2_remd,
)
from .schemas import (  # noqa: F401
    SCHEMA_VERSION,
    ExperimentManifest,
    ManifestError,
    SystemManifest,
    config_hash,
    load_experiment,
    load_system,
)
from .solvation import solvate  # noqa: F401
from .system import (  # noqa: F401
    build_forcefield,
    build_rest2_scaled_system,
    build_system,
    classify_omega_bonds,
    initial_structure,
    omega_central_bonds,
    protonate,
    repartition_hydrogen_mass,
    resolve_route,
)

__all__ = [
    # configuration
    "DEFAULTS", "load_config", "dump_defaults", "resolve_config", "exchange_rounds",
    "rest2_ladder", "write_manifest",
    # preparation
    "initial_structure", "protonate", "build_forcefield", "solvate", "build_system",
    "repartition_hydrogen_mass", "resolve_route",
    "classify_omega_bonds", "omega_central_bonds", "build_rest2_scaled_system",
    # running
    "build_simbox", "minimize_equilibrate", "make_integrator", "run_md", "run_rest2_remd",
    "completed_prefix", "exchange_pairs", "attempt_rest2_exchange", "resolve_remd_scale_ladder",
    # manifests
    "SCHEMA_VERSION", "ExperimentManifest", "ManifestError", "SystemManifest",
    "config_hash", "load_experiment", "load_system",
]
