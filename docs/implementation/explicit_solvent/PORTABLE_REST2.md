# Portable explicit-solvent REST2

How to run this project's explicit-solvent REST2 from **another repository or another machine**,
through an installed package, without the source checkout, without `/home/...` paths, without
fixed GPU numbers, and without undocumented molecular inputs.

Everything below is executed through one console script, `escort-explicit`, which the wheel puts
on `PATH`. No command needs `docs/implementation/explicit_solvent/scripts/`.

---

## What this does and does not establish

Read this before quoting any number produced by these commands.

* The portable smoke demonstrates **installability and mechanical execution**: the package
  installs from a wheel, the toolchain is complete, a System can be built, every replica
  propagates, exchanges are attempted, and a complete run directory is written.
* It **does not validate a new macrocycle ladder**. The smoke is three rungs and two picoseconds
  on a cyclic tripeptide. Nothing about acceptance, convergence or overlap can be read from it.
* **Generic macrocycles require a new pair-resolved acceptance pilot.** Use
  `macrocycle_pilot_8rung` (`ladder_status: unvalidated`), read `exchange_attempts.csv` per
  neighbouring pair, and size the real ladder from the worst pair.
* **cyclo-(RGDfV) retains the ten-rung working ladder** (`rgd_rest2_10rung`,
  `ladder_status: validated`). That status is specific to RGD: it comes from three matched
  2 ns/replica pilots against the 8-rung control (median worst-pair acceptance 0.140 → 0.240,
  pooled minimum 0.173 → 0.277). One molecule is the whole evidence base; it does not transfer to
  cyclosporin or to the RGD derivatives.
* **Long production is still gated.** Two questions are open and neither is closed by this work:
  the 2 fs / 4 fs hydrogen-mass-repartitioning equivalence, and restart correctness. Until both
  are resolved, treat these runs as pilots.

---

## Install

```bash
conda env create -f environment.yml          # name: escort-ais-explicit
conda activate escort-ais-explicit
python -m build                              # writes dist/escort_ais-<version>-py3-none-any.whl
python -m pip install dist/escort_ais-*.whl --no-deps
```

`--no-deps` is deliberate: the heavy scientific stack (OpenMM, OpenFF, AmberTools, RDKit) comes
from conda, and letting pip resolve it produces a different and usually broken stack. Every
runtime dependency is therefore listed in `environment.yml`.

An editable install is **not** required. The consuming repository installs the wheel and never
sees this source tree.

Check the machine before paying for anything:

```bash
escort-explicit validate-env --platform CUDA --device 0
```

It reports the Python version, the importability and version of every package that can change a
number, the OpenMM platforms actually available, `sqm` (without which AM1-BCC is unobtainable),
the CUDA runtime and the requested device, the precision mode, and where the installed package
lives. Any **FAIL** line blocks `prepare` and `rest2`, which run the same checks.

---

## Workflow 1 — build locally from a versioned system manifest

Use when the consuming machine should construct its own solvated box.

```bash
escort-explicit validate-system --system cyclo_rgdfv

escort-explicit prepare \
  --system     cyclo_rgdfv \
  --experiment rgd_rest2_10rung \
  --out-root   ./runs \
  --platform   CUDA --device 0

escort-explicit rest2 \
  --bundle   ./runs/<TIMESTAMP>__cyclo_rgdfv__bundle__<HASH> \
  --out-root ./runs \
  --platform CUDA --device 0
```

`cyclo_rgdfv` and `rgd_rest2_10rung` are manifests **shipped inside the wheel**; that is what
keeps these commands free of machine-local paths. Either argument also accepts a path to your own
file.

**This route is scientifically consistent but not bitwise identical** to a box built elsewhere.
AM1-BCC charges, ETKDG embedding and water placement all depend on library versions and on host
floating point, so the System and the starting state differ in their last digits and trajectories
diverge immediately.

## Workflow 2 — consume an already prepared bundle

Use when the System and the starting state must be **identical** — which is the case whenever runs
from several machines will be pooled.

```bash
# on the machine that has the bundle
tar czf bundle.tgz <TIMESTAMP>__cyclo_rgdfv__bundle__<HASH>

# on the consuming machine
tar xzf bundle.tgz
escort-explicit validate-bundle --bundle <TIMESTAMP>__cyclo_rgdfv__bundle__<HASH>

escort-explicit rest2 \
  --bundle   <TIMESTAMP>__cyclo_rgdfv__bundle__<HASH> \
  --out-root ./runs \
  --platform CUDA --device 0
```

`validate-bundle` recomputes the SHA-256 of every file and re-derives the config hash from the
manifests inside the bundle. A truncated copy, an edited manifest, or a missing file fails here
rather than after a week of compute.

---

## The two manifests

### System manifest — what the molecule is

```yaml
schema_version: 1
system_id: cyclo_rgdfv
display_name: cyclo-RGDfV

input:
  route: smiles                                  # smiles | pdb — never `auto`
  smiles: "<exact stereochemical and protonation-state SMILES>"
  canonical_isomeric_smiles: "<RDKit canonical isomeric SMILES>"
  canonical_smiles_sha256: "<sha256 of the line above>"
  expected_formal_charge: 0

parameterization:
  small_molecule_forcefield: openff-2.2.0
  charge_method: am1bcc
  protein_forcefield: null
  water_forcefield: amber19/tip3pfb.xml
```

For a PDB-based peptide macrocycle the contract is different, and the difference is enforced:

```yaml
input:
  route: pdb
  pdb: path/to/input.pdb                         # relative to the manifest
  pdb_sha256: "<sha256>"

parameterization:
  protein_forcefield: amber19/protein.ff19SB.xml
  small_molecule_forcefield: null
```

Validated before any expensive work, in this order: the input exists; the route and the
parameterisation agree; the SMILES parses; the canonical isomeric SMILES and its hash agree;
stereochemistry is fully assigned; the computed formal charge equals `expected_formal_charge`;
the PDB hash agrees.

Two prohibitions are structural rather than advisory. **A ligand route cannot load a protein force
field**, and **a peptide route cannot also name a small-molecule force field** — that is how an
ff19SB calculation silently becomes a Sage one under the same name. Neither can `route` be `auto`:
it is resolvable only against a specific invocation, which is the opposite of portable.

**`cyclo_rgdfv` is pinned by identity, not by name.** The canonical SMILES hash must be
`59d4422635f77292ed94c609a76808df5c97110be09cd9779819270363143159` and the formal charge exactly
`0` — the zwitterion, C26H38N8O7, Asp deprotonated and Arg guanidinium protonated. A neutral
macrocycle carrying that `system_id` is rejected, because "a neutral macrocycle called RGD" is
precisely what a copy-paste error produces.

> Note: `systems/cyclo_rgdfv/system.yaml` in the source tree records `forcefield: ff19SB`. That is
> the **scientific system definition** for the implicit-solvent AIS work and its REMD reference; it
> is a different artifact for a different calculation and is not interchangeable with the portable
> manifest above, which parameterises the same molecule as a Sage/AM1-BCC ligand.

### Experiment manifest — how it is run

```yaml
schema_version: 1
experiment_id: rgd_rest2_10rung
master_seed: 20260814
ladder_status: validated          # validated | unvalidated; absent means unvalidated

integrator:
  kind: langevin-middle
  temperature_k: 300.0
  timestep_fs: 4.0

rest2:
  scale_factors: [1.0, 0.891975, ..., 0.25]
  exchange_interval_ps: 10.0
  relaxation_ps: 10.0
  total_ns_per_replica: 2.0
  chunk_ns: 1.0

platform:
  precision: mixed
```

Optional `equilibration:` and `overrides:` blocks reach the rest of the baseline configuration
tree. Unknown keys are rejected in both, so a typo cannot leave a default in force while the file
appears to change it.

Three are shipped: `rgd_rest2_10rung` (validated, RGD only), `macrocycle_pilot_8rung`
(**unvalidated** template for a new acceptance pilot), and `smoke` (three rungs, picoseconds).

---

## Run directories

```
RUN_ROOT/YYYYMMDDTHHMMSSZ__SYSTEM_ID__rest2__CONFIG_HASH/
  run_manifest.json      identity, ladder, platform, full environment block
  system.yaml            the system manifest, as run
  experiment.yaml        the experiment manifest, as run
  resolved_config.json   every resolved baseline knob
  bundle_manifest.json   the bundle this run consumed
  status.json            running | completed | failed | interrupted
  stdout.log stderr.log
  replica_00/ ...        one directory per rung
  exchange_attempts.csv  every attempt, with per-pair scale factors and energies
```

The config hash derives from the canonical serialisation of the two manifests, so the same
calculation gets the same hash on any machine and any change to a seed or a scientific setting
changes it. It deliberately excludes the platform and device: the same calculation on CPU and CUDA
is the same configuration.

**Runs are immutable.** An existing directory is refused rather than reused or extended.

**`completed` means the requested budget was reached.** A smoke run, an interrupted run and a
crashed run write `interrupted` or `failed` and a non-zero exit code. A preempted job gets
`interrupted` rather than being left at `running` for ever, because `SIGTERM` is handled.

Exit codes are deterministic, so a scheduler can branch without parsing text: `0` ok, `2` usage,
`3` manifest, `4` environment, `5` bundle, `6` run directory exists, `7` runtime, `130`
interrupted.

---

## Launching, and what changed

One process controls its own configured replicas on one selected device. Multi-job scheduling is
the consuming repository's problem.

```bash
escort-explicit rest2 --bundle BUNDLE --out-root ROOT --platform CUDA --device 1
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set at launch so `--device 1` is the card `nvidia-smi` calls 1.
Without it the CUDA runtime may order devices differently.

`docs/implementation/explicit_solvent/scripts/run_all.sh` survives as a thin wrapper around these
commands. Its previous version hardcoded `CUDA_VISIBLE_DEVICES=0/1/2`, assumed three GPUs,
backgrounded three jobs with `nohup`, and wrote logs to a repository-relative `logs/`. On a
one-GPU machine the second and third jobs landed back on device 0, where two REST2 processes
sharing a card without CUDA MPS run about **3.7× slower each** — wrong, but not visibly wrong.

---

## Verify an installation

```bash
cd "$(mktemp -d)"                      # anywhere outside the checkout
escort-explicit --help
escort-explicit validate-env
escort-explicit validate-system --system cyclo_rgdfv
escort-explicit smoke --system small_macrocycle_smoke --out-root ./test-runs --platform CPU
```

The smoke must exit `0` with `status: completed` and at least two exchange rounds. It takes about
a minute on CPU. Again: it proves the install, not the science.
