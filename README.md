# Explicit-solvent MD and REST2 — OpenMM template

A standard, self-contained template for building and running **explicit-water** simulations of
peptide-like compounds and small-molecule ligands with OpenMM: parameterise from SMILES or a PDB,
solvate, equilibrate, and run REST2 replica exchange from hash-checked, transportable bundles.

Two force-field routes, both exercised end to end on this machine:

| route | input | force field | worked example |
|---|---|---|---|
| ligand | `smiles` | **Sage / openff-2.2.0** + AM1-BCC | `cyclo_rgdfv`, `small_macrocycle_smoke` |
| peptide | `pdb` | **ff19SB** (+ CMAP) | `ace_ala_nme` |

The route is **declared, never inferred**: a ligand manifest may not name a protein force field and
a peptide manifest may not name a small-molecule one, because that is precisely how a peptide
silently becomes a Sage run with the same system name. Water is TIP3P-FB throughout.

Nothing here depends on the project it came from: no annealed-importance-sampling machinery, no implicit-solvent
work.

## Install

```bash
conda env create -f docs/implementation/explicit_solvent/environment.yml   # -> md-templates
conda activate md-templates
pip install -e . --no-deps          # from source, for development
```

`--no-deps` is deliberate: OpenMM, OpenFF, AmberTools and RDKit come from conda, and letting pip
resolve them produces a different and usually broken stack. `environment.yml` therefore lists every
runtime dependency.

To consume the pipeline **without** a checkout, use the shipped package instead — see
[`docs/implementation/explicit_solvent/HANDOFF_PACKAGE.md`](docs/implementation/explicit_solvent/HANDOFF_PACKAGE.md).

```bash
md-openmm validate-env    --platform CUDA --device 0
md-openmm validate-system --system cyclo_rgdfv --experiment rgd_rest2_10rung
md-openmm smoke           --system small_macrocycle_smoke --out-root ./runs --platform CPU
md-openmm smoke           --system ace_ala_nme           --out-root ./runs --platform CPU
md-openmm prepare         --system cyclo_rgdfv --experiment rgd_rest2_10rung \
                                --out-root ./runs --platform CUDA --device 0
md-openmm rest2           --bundle ./runs/<TIMESTAMP>__cyclo_rgdfv__bundle__<HASH> \
                                --out-root ./runs --platform CUDA --device 0
```

## What is here

```text
src/md_templates/
  explicit/                    the portable pipeline: CLI, config, bundles, runner, manifests
  systems/explicit_baseline.py the explicit-solvent build and MD driver
  systems/openmm_system.py     build_rest2_scaled_system -- the REST2 Hamiltonian
  systems/topology_prep.py     resolve_remd_scale_ladder, used by run_rest2_remd
  systems/cv_definition.py     reached lazily from topology_prep
  methods/md_run.py            attempt_rest2_exchange, exchange_pairs
  common/paths.py              data_dir / ensure_dir / project_root
docs/implementation/explicit_solvent/
  PORTABLE_REST2.md            the design and its limits
  HANDOFF_PACKAGE.md           the shipped package, its revision, and the source gap below
  environment.yml              the conda environment
  scripts/                     the pre-CLI staged scripts (md.py, md_REST2.py, simbox-setup.py, ...)
tests/                         test_explicit_baseline.py, test_explicit_portable.py
scripts/rest2_pilot_acceptance.py
```

The source tree is 23 modules: `md_templates.openmm.*`, `systems/{explicit_baseline, openmm_system,
topology_prep, cv_definition}.py`, `methods/md_run.py` and `common/paths.py`.

That closure was established by **running the pipeline from a pristine checkout**, not by reading
imports. An import trace of the explicit package loads only eleven modules and would have you
believe `methods/` is unnecessary — but `explicit_baseline.run_rest2_remd` imports
`methods.md_run` and `systems.topology_prep` at lines 2186–2187, *inside the function*, so they
appear only once an exchange is actually attempted. The first cut of this branch imported cleanly,
prepared a bundle cleanly, and then failed at the first exchange with
`ModuleNotFoundError: No module named 'md_templates.openmm.md'`. The CPU smoke is the check that
matters; it now passes from a pristine checkout of this branch (`status: completed`, 4/4 rounds).

## State of the science — read before quoting anything

* `cyclo_rgdfv` + `rgd_rest2_10rung` is **`ladder_status: pilot_supported`**. That is a claim about
  one molecule on declared pilot evidence. It is *not* `validated`, not converged, and not
  production-ready; the predeclared conjunctive acceptance rule was never formally satisfied.
* Any other macrocycle needs its own pair-resolved acceptance pilot — start from
  `macrocycle_pilot_8rung.yaml`, which is `ladder_status: unvalidated` by design.
* **The 2 fs / 4 fs hydrogen-mass-repartitioning equivalence gate is open.** Long production is
  blocked on it.
* Restart safety and convergence are not established.
* The smokes prove installability and mechanical execution. Nothing more.
* **No ladder is validated for any peptide.** `ace_ala_nme` proves the ff19SB route runs; it says
  nothing about rung spacing for a peptide of any size.

## Source provenance — the wheel and this tree agree

The distributed wheel `md_templates-0.1.0-py3-none-any.whl`
(sha256 `50e9a1a51ecbd3680d43985786813727186d534c486a9501ebd4895fe639a183`) was built from commit
`10809c7` of the originating repository. That source is the source in this tree: all nine modules of
`md_templates/openmm/` are **byte-identical** between `10809c7` and the shipped wheel, verified by
comparison rather than assumed from a version string.

Two fixes sit on top of it, both found by running the pipeline rather than reading it:

* `environment.yml` asked for the conda package `build`, which conda-forge packages as
  `python-build`. With `channels: [conda-forge, nodefaults]` pinned, `conda env create` was
  unsatisfiable — the package could not be installed by following its own instructions.
* a `pdb`-route bundle did not copy the structure its own manifest points at, so it could not be
  re-validated and the peptide route could build a system but never use one.

## Historical validation evidence

The 2026-08-15 target-machine validation, the ff19SB peptide-route report and the ladder pilots
were produced by the package as it was BEFORE this refactor: a different distribution name, a
different console script, experiment schema v1, and the retired salt fields. They are accurate
about what they tested and are preserved in git history at `f885be5` -- `git show f885be5:reports/` --
but they are not kept in the working tree, because every command in them names an entry point this
template no longer has, and rewriting those names would turn a true record into a false one.

Re-running them against the current CLI is the way to restore an evidence directory here.

## Provenance

Extracted from the research repository, where this pipeline was developed alongside
implicit-solvent work. Published as a single initial commit: the tree is what matters for a
template, and the development history remains in the originating repository.
