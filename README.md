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

Nothing here depends on the project it came from: no AIS, no cBAR, no pREST2, no implicit-solvent
work.

## Install

```bash
conda env create -f docs/implementation/explicit_solvent/environment.yml   # -> escort-ais-explicit
conda activate escort-ais-explicit
pip install -e . --no-deps          # from source, for development
```

`--no-deps` is deliberate: OpenMM, OpenFF, AmberTools and RDKit come from conda, and letting pip
resolve them produces a different and usually broken stack. `environment.yml` therefore lists every
runtime dependency.

To consume the pipeline **without** a checkout, use the shipped package instead — see
[`docs/implementation/explicit_solvent/HANDOFF_PACKAGE.md`](docs/implementation/explicit_solvent/HANDOFF_PACKAGE.md).

```bash
escort-explicit validate-env    --platform CUDA --device 0
escort-explicit validate-system --system cyclo_rgdfv --experiment rgd_rest2_10rung
escort-explicit smoke           --system small_macrocycle_smoke --out-root ./runs --platform CPU
escort-explicit smoke           --system ace_ala_nme           --out-root ./runs --platform CPU
escort-explicit prepare         --system cyclo_rgdfv --experiment rgd_rest2_10rung \
                                --out-root ./runs --platform CUDA --device 0
escort-explicit rest2           --bundle ./runs/<TIMESTAMP>__cyclo_rgdfv__bundle__<HASH> \
                                --out-root ./runs --platform CUDA --device 0
```

## What is here

```text
src/escort_ais/
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
docs/journal/                  four explicit-solvent working journals, 2026-08-14/15
reports/
  explicit_solvent/            target-machine validations + the shipped package artefact
  explicit_solvent_validation/ the 2026-08-14 ladder pilots (10-rung vs 8-rung, three repeats each)
  alanine/20260814_explicit_solvent_validation/   the alanine and RGD pilot summaries
tests/                         test_explicit_baseline.py, test_explicit_portable.py
scripts/rest2_pilot_acceptance.py
```

The source tree is 23 modules: `escort_ais.explicit.*`, `systems/{explicit_baseline, openmm_system,
topology_prep, cv_definition}.py`, `methods/md_run.py` and `common/paths.py`.

That closure was established by **running the pipeline from a pristine checkout**, not by reading
imports. An import trace of the explicit package loads only eleven modules and would have you
believe `methods/` is unnecessary — but `explicit_baseline.run_rest2_remd` imports
`methods.md_run` and `systems.topology_prep` at lines 2186–2187, *inside the function*, so they
appear only once an exchange is actually attempted. The first cut of this branch imported cleanly,
prepared a bundle cleanly, and then failed at the first exchange with
`ModuleNotFoundError: No module named 'escort_ais.methods'`. The CPU smoke is the check that
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

## Known gaps

**The shipped wheel is ahead of this source.** The distributed package was built from commit
`10809c7`, which exists in no clone on `<host>`; this branch's `src/` lacks
`explicit/fingerprint.py` and still carries `box_shape: cube`, `ladder_status: validated` and
six-decimal scale factors, where the wheel has dodecahedron, `pilot_supported` and exact values.
**Do not rebuild the wheel from this branch** — it would silently regress all four under the same
version number. Details and the recovery path are in `HANDOFF_PACKAGE.md`.

**A cited evidence file is missing.** `src/escort_ais/explicit/manifests/systems/cyclo_rgdfv.yaml`
names `reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json`
as the durable repository copy to cite for the dodecahedron box geometry — and that file is not
tracked anywhere in this repository. The geometry it records (box width 3.66182 nm, minimum-image
distance 2.5893 nm, 1047 waters, 3226 particles) is corroborated independently: a bundle prepared
from scratch on 2026-08-15 reproduced every one of those values exactly. See
`reports/explicit_solvent/20260815_target_machine_validation_run2/README.md` §7.

## Evidence

Every claim above has its artefacts in `reports/`, not just prose:

* `explicit_solvent/20260815_target_machine_validation_run2/` — the ligand route installed and run
  from a clean environment on a machine that had never seen the source: CPU and CUDA smokes, a
  from-scratch RGD preparation, and a ten-rung REST2 run, with all console logs, environment
  exports and bundle manifests.
* `explicit_solvent/20260815_ff19sb_peptide_route/` — the peptide route's first end-to-end run,
  including the defect it exposed (a `pdb`-route bundle did not carry its own structure, so it
  could not be re-validated) and the passing run after the fix.
* `explicit_solvent_validation/20260814_v2/` — the ladder pilots behind `rgd_rest2_10rung`:
  ten-rung against eight-rung, three matched repeats each, pair-resolved.

## Provenance

Extracted from the `escort-ais` research repository, where this pipeline was developed alongside
implicit-solvent work. Published as a single initial commit: the tree is what matters for a
template, and the development history remains in the originating repository.
