"""The OpenMM provider: the single reusable implementation for this engine.

Phase 4 of the migration moved the implementation here from `md_templates.openmm`, which is now a
thin compatibility namespace. There is exactly one copy of every module; nothing was duplicated.

Boundaries, in dependency order -- each layer imports only from those above it:

    schemas / config      legacy manifest reading and the runtime configuration this engine expects
    adapter               canonical spec -> that runtime configuration
    system / solvation    force fields, structure generation, omega classification, System building
    equilibration         integrators, the prepared box, staged equilibration
    methods/md            one free walker at a fixed Hamiltonian
    methods/rest2         replica exchange: ladder, neighbour schedule, Metropolis criterion
    bundle / bundlecheck  preparing, validating and inspecting a transportable bundle
    bundleinfo            counts and force-field provenance that need a built System
    restart               checkpoint and serialized-State I/O for a running Simulation
    platform              platform and device selection
    provenance / runner   run directories, manifests, and the top-level orchestration
    cli                   the `md-openmm` command

Everything engine-neutral lives in `md_templates.core` and is imported from there: the canonical
configuration model, the bundle file format, persistence layout, hashing, and the template catalog.
"""
