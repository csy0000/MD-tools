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

Generated `run.py` must not import `md_tools` at simulation time. So the classification work --
solute indices, the REST2 enhanced region, omega-excluded bonds -- is done ONCE at `sys-gen` time
and written to `solute.yaml`. The generated script reads that file and applies the scaling
arithmetic directly, which is about thirty lines. The chemistry stays in the package where it is
tested; the runtime script stays readable and standalone.

---

## Outcome

Implemented on `dev` at `ca29fcd`.

```
source          20,216 lines / 55 modules   ->   4,211 lines / 23 files
tests           41 files / 1,372 tests      ->   8 files / 32 tests
commands        3 entry points, 9 subcommands  ->  2 entry points, 6 subcommands
```

### The scaling was checked, not assumed

`templates/rest2_scaling.py` is standalone code, so the transfer was verified against the
implementation it replaces rather than trusted. Over every scaled parameter — charges, sigmas,
epsilons, exception parameters, torsion force constants, CMAP map energies — at tau = 0.1, 0.3,
0.5:

```
max |standalone - validated| = 0.000e+00
```

### A physics bug the refactor exposed

Writing the exchange loop as a plain script made a defect visible that the old layered runner had
hidden: the cross-energy step put replica *i*'s positions into replica *j*'s context **without**
`j` adopting `i`'s box. Under NPT each replica has its own volume, so the configuration was
evaluated in a cell it does not fit — overlapping images, enormous energies, acceptance pinned at
zero.

```
before   log_acceptance  -89.3, -472.8, -152.5, -1176.6     0/4 accepted
after    log_acceptance   -0.078, +0.441                    2/2 accepted
```

with `tau_max = 0.15` (s = 0.72…1.0). A configuration is positions *and* the cell they are periodic
in.

### What deletion taught

`seeds.py` was deleted and had to be restored: it is imported **lazily inside `protonate()`**, so a
top-level import scan does not see it. The suite found it; reading did not. Any further deletion
should be driven by running, not by static analysis.

### Deliberately not done

`md-template install` was exercised with `--dry-run` only. A real install downloads roughly a
gigabyte from conda-forge, and the environment it would create already exists on this machine. The
command's package-manager detection, environment path construction, platform probe and
`machine.yaml` recording were all exercised directly against the existing environment: `mamba`
detected, OpenMM 8.6.0, Python 3.12.13, platforms Reference/CPU/CUDA/OpenCL, `cuda_check: ok`.
