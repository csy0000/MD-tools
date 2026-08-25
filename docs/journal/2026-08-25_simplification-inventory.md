# Simplification inventory

Date: 2026-08-25
Branch: `dev`

20,216 lines across 55 Python modules. This records what is scientifically load-bearing and must
survive, and what is infrastructure built for engines, workflows and guarantees this repository was
never asked to provide.

## Preserve — validated scientific implementation

| what | where it lives now |
|---|---|
| REST2 tau scaling, AMBER convention: solute-solute `(1-tau)^2`, solute-environment `(1-tau)` | `openmm/tau.py` (`TauScaling`, `scaling_for_tau`, `linear_tau_ladder`), applied in `openmm/system.py:build_rest2_scaled_system` |
| REST2 exchange: pairing, NPT reduced potential `u_k(x,V) = b_k(U_k + p_k V)`, acceptance, state swap | `openmm/rest2.py` (`exchange_pairs`, `reduced_potential`, `exchange_log_acceptance`, `run_rest2_remd`) |
| omega exclusion: amide detection, ring-size bound, proline-like classification | `openmm/system.py` (`classify_omega_bonds`, `_amide_candidates`, `_ligand_ring_nitrogens`, `omega_central_bonds`) |
| GBn2 + mbondi3, built through ParmEd rather than `AmberPrmtopFile` (the two differ by ~16 kJ/mol in `CustomGBForce`) | `openmm/implicit.py` |
| dodecahedral box, and the separation of lattice translation / reduced height / cutoff fit | `openmm/solvation.py` (`_box_vectors`, `shortest_lattice_translation`, `minimum_reduced_box_height`, `solvate`, `salt_accounting`) |
| minimisation, restrained equilibration, positional restraints, integrator construction | `openmm/equilibration.py` |
| hydrogen mass repartitioning with its conservation and floor checks | `openmm/system.py` (`repartition_hydrogen_mass`, `verify_hydrogen_mass_repartitioning`) |
| checkpoint / restart continuation | `openmm/rest2.py` relaxation + chunk loop; cMD counterpart in `openmm/cmd_segments.py` |
| force field construction, protonation, ligand parameterisation | `openmm/system.py` (`build_forcefield`, `protonate`, `initial_structure`) |

## Remove — infrastructure for problems this repository does not have

| what | why |
|---|---|
| `core/registry.py`, `core/template.py`, `core/packaged.py` | plugin registry and template catalog for hypothetical engines |
| `openmm/spec/` (models, resolve, migrate, diffs, canonical, adapter, units, 21 profiles) | a canonical-configuration layer with schema migration and profile inheritance, replacing what two readable YAML files should say |
| `openmm/input_gen.py`, `openmm/runtime_export.py` | generated a project that vendored the entire package as a runtime snapshot |
| `openmm/lockfile.py`, `build_backend/`, `build_support/catalog.py`, `openmm/fingerprint.py`, `openmm/hashing.py` | build-time provenance framework, wheel identity stamping, source-tree digests, multiple overlapping identity files |
| `openmm/bundle.py`, `bundlev2.py`, `destination.py`, `schemas.py`, `system_prep.py` | bundle abstraction with schema versions and checksum manifests over what is a directory of four files |
| `openmm/stage.py`, `runner.py`, `segments.py`, `cmd_segments.py`, `runstate.py`, `faults.py`, `dcdtail.py` | a staged workflow engine with a committed-generation contract, quarantine, and crash boundaries |
| `openmm/gpus.py` (busy detection, MIG, PCI mapping) | reduced to `min(n_replicas, n_visible)` over `CUDA_VISIBLE_DEVICES` |
| `openmm/water_policy.py`, `solvation_mode.py`, `manifests/`, `templates/` catalog | policy layers and packaged manifests over settings that belong in the user's YAML |
| `openmm/cli.py`, `cli_system_gen.py`, `cli_input_gen.py` | replaced by two small entry points |

## Consequence for generated projects

Generated `run.py` must not import `md_templates` at simulation time. So the classification work --
solute indices, the REST2 enhanced region, omega-excluded bonds -- is done ONCE at `sys-gen` time
and written to `solute.yaml`. The generated script reads that file and applies the scaling
arithmetic directly, which is about thirty lines. The chemistry stays in the package where it is
tested; the runtime script stays readable and standalone.
