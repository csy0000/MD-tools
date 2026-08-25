# Portable explicit-solvent REST2

> **HISTORICAL — not current usage.**
>
> This describes the explicit-solvent baseline as it stood before the repository was reduced to the
> six-command CLI in `ca29fcd`. Its runnable scripts have been deleted: they were a second,
> unmaintained copy of the run scripts, and they never received the REST2 duration, equilibration,
> seed and trajectory corrections of 2026-08-25. Running them would reproduce those bugs.
>
> It is kept because `reports/explicit_solvent/` cites it, and a retained validation report needs
> the document that explains what was run. For current usage see the root `README.md`.


How to run this project's explicit-solvent REST2 from **another repository or another machine**,
through an installed package, without the source checkout, without `/home/...` paths, without
fixed GPU numbers, and without undocumented molecular inputs.

Everything below is executed through one console script, `md-openmm`, which the wheel puts
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
  `ladder_status: pilot_supported`). That status is specific to RGD: it comes from three matched
  2 ns/replica pilots against the 8-rung control (median worst-pair acceptance 0.140 → 0.240,
  pooled minimum 0.173 → 0.277). One molecule is the whole evidence base; it does not transfer to
  cyclosporin or to the RGD derivatives.
* **Long production is still gated.** Two questions are open and neither is closed by this work:
  the 2 fs / 4 fs hydrogen-mass-repartitioning equivalence, and restart correctness. Until both
  are resolved, treat these runs as pilots.

---

## Ladder status vocabulary

Two values, and deliberately no third:

| status | meaning |
|---|---|
| `unvalidated` | no system-specific pair-resolved evidence. An **absent** status resolves here — silence is not evidence. |
| `pilot_supported` | adopted as the working ladder on declared, **system-specific** pilot evidence. |

`pilot_supported` is the strongest status this vocabulary offers, and it is deliberately weaker
than "validated". It does **not** assert convergence, production readiness, or a formal pass of
every predeclared acceptance condition — cyclo-(RGDfV)'s ladder was adopted after three matched
pilots while the predeclared conjunctive rule was *not* formally satisfied, because condition 5
was under-specified. It is also a claim about one system only: copying a `pilot_supported`
manifest to another molecule silently transfers a claim that was never made about it.

An unknown value fails manifest validation rather than being carried as a free-text note, and run
and bundle manifests record the status verbatim.

## Install

```bash
conda env create -f environment.yml          # name: md-templates
conda activate md-templates
python -m build                              # writes dist/md_templates-<version>-py3-none-any.whl
python -m pip install dist/md_templates-*.whl --no-deps
```

`--no-deps` is deliberate: the heavy scientific stack (OpenMM, OpenFF, AmberTools, RDKit) comes
from conda, and letting pip resolve it produces a different and usually broken stack. Every
runtime dependency is therefore listed in `environment.yml`.

An editable install is **not** required. The consuming repository installs the wheel and never
sees this source tree.

Check the machine before paying for anything:

```bash
md-openmm validate-env --platform CUDA --device 0
```

It reports the Python version, the importability and version of every package that can change a
number, the OpenMM platforms actually available, `sqm` (without which AM1-BCC is unobtainable),
the CUDA runtime and the requested device, the precision mode, and where the installed package
lives. Any **FAIL** line blocks `prepare` and `rest2`, which run the same checks.

---

## Workflow 1 — build locally from a versioned system manifest

Use when the consuming machine should construct its own solvated box.

```bash
md-openmm validate-system --system cyclo_rgdfv

md-openmm prepare \
  --system     cyclo_rgdfv \
  --experiment rgd_rest2_10rung \
  --out-root   ./runs \
  --platform   CUDA --device 0

md-openmm rest2 \
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
md-openmm validate-bundle --bundle <TIMESTAMP>__cyclo_rgdfv__bundle__<HASH>

md-openmm rest2 \
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
  water_forcefield: amber19/opc.xml
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

**The RGD box shape is evidence, not a default.** `box_shape: dodecahedron` is proven from the
Phase-A prepared system that fed all six matched ladder pilots — they copy its `rgd_simbox.json`
(see `reports/explicit_solvent_validation/20260814_v2/raw_exchange/pilots6_command.sh`). Preserved
in-repo, because the original lived in a job temp directory:

```
reports/explicit_solvent_validation/20260814_v2/phaseA_prepared_system/rgd_simbox.json
sha256 ea1c14edb7c9fc949d892fb41fb733750179555c235b3d6a0a54c7bda821ac61
  geometry.box_shape              dodecahedron
  geometry.box_width_nm           3.66182       (grown_for_cutoff: false)
  geometry.min_image_distance_nm  2.5893        at a 1.0 nm cutoff
  realised_ionic_strength_molar   0.15903       (Na+ 3 / Cl- 3, 1047 waters, 3226 particles)
```

An earlier revision of this manifest said `cube`, which would have attached the ten-rung
acceptance statistics to a box geometry the pilots never ran in. A regression test pins the
manifest to this artifact.

> Note: `systems/cyclo_rgdfv/system.yaml` in the source tree records `forcefield: ff19SB`. That is
> the **scientific system definition** for the originating project's implicit-solvent work and its REMD reference; it
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

**A run is created fresh or continued in place.** `--run-name NAME` creates exactly
`RUN_ROOT/NAME` with no timestamp appended; omitting it gives a timestamped default that also
carries the method, so an MD run and a REST2 run of the same configuration cannot collide. A fresh
run refuses an existing directory rather than reusing it, and says to resume explicitly.

**`--resume-run` continues a run in the SAME directory** -- no sibling, no child. `n_chunks` means
the chunks this invocation ADDS, so a resume extends the run. Before anything is loaded or
appended, the continuity contract recorded in `run_state.json` is compared with the requested
configuration, and a mismatch is refused with the differing fields listed. Chunk length, force
field, integrator, constraints, ladder and the omega-exclusion setting are all part of that
contract; the number of chunks deliberately is not.

**Restarts are committed, not merely written.** At each chunk boundary both an OpenMM checkpoint
and a portable serialized State are written into a generation directory, and only then does a
single atomic replacement of `restart/committed.json` make that generation current. A crash
part-way through leaves the previous generation in force. On resume the binary checkpoint is
preferred; if it is missing, corrupt or from another platform the State is used instead, which
preserves positions, velocities, box, time and parameters but NOT the stochastic integrator's
stream -- reported, never silent.

**REST2 statistics are lifetime statistics.** The exchange log carries a global `attempt_index`;
counters are rebuilt from it when a run is opened rather than starting at zero, and the summary
reports lifetime and this-invocation figures separately. An attempt counts only once the generation
holding its resulting state is committed, so an uncommitted tail is discarded on recovery.

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
md-openmm rest2 --bundle BUNDLE --out-root ROOT --platform CUDA --device 1
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set at launch so `--device 1` is the card `nvidia-smi` calls 1.
Without it the CUDA runtime may order devices differently.

`docs/implementation/explicit_solvent/scripts/run_all.sh` survives as a thin wrapper around these
commands. Its previous version hardcoded `CUDA_VISIBLE_DEVICES=0/1/2`, assumed three GPUs,
backgrounded three jobs with `nohup`, and wrote logs to a repository-relative `logs/`. On a
one-GPU machine the second and third jobs landed back on device 0, where two REST2 processes
sharing a card without CUDA MPS run about **3.7× slower each** — wrong, but not visibly wrong.

---

## Running a different experiment against a prepared bundle

`rest2 --bundle B --experiment E` lets a new experiment run against an already-prepared bundle.
That is the point of a transferable bundle — run longer, or on another device, from the identical
starting state. It is also the obvious way to produce a silently wrong result, because nothing in
the file layout stops a new experiment from declaring a different force field or a different box,
neither of which can take effect: `system.xml` and `equilibrated_state.xml` were built under the
old settings.

So a bundle records a **build-defining projection** of its resolved configuration and that
projection's SHA-256 (`prepared_system.fingerprint` in `bundle_manifest.json`). An override
recomputes the projection and must match it exactly.

| accepted — the prepared System is still right | rejected — the prepared System would be wrong |
|---|---|
| total duration, chunk length | force field, charge method, water model |
| exchange interval, relaxation | box shape, padding, salt, neutralisation |
| reporting intervals | cutoff, PME settings, constraints, rigid water, HMR |
| platform, device, precision | minimum-image margin |
| production timestep, run seeds | omega selection and classification |
| the REST2 ladder itself | anything that produced `equilibrated_state.xml` |

Rejection happens **before** the run directory or any OpenMM context is created, and names the
differing dotted paths with both values. The comparison is against the recorded projection, not
against the fingerprint alone, so rehashing an edited manifest does not bypass it. Both the bundle
fingerprint and the run's own config hash are recorded in `run_manifest.json`. Exit code `8`.

## Box geometry: the minimum-image margin

OpenMM refuses a cutoff larger than half the minimum image distance, and aborts mid-run if the box
contracts past it. When a requested box is too small for the cutoff, the preparation grows it —
and growing it to *exactly* twice the cutoff leaves it sitting on the limit, so the first NPT step
crosses it and the run dies with "The periodic box size has decreased to less than twice the
nonbonded cutoff".

`system_build.minimum_image_margin_nm` (default **0.10 nm**) is the headroom required when growth
is necessary:

```
minimum image distance  >=  2 * nonbonded_cutoff + minimum_image_margin_nm
```

A box that is already large enough is left exactly as requested — the rule only ever grows. A
negative margin is rejected before any box is built, and `0.0` restores the old
grow-to-the-limit behaviour for anyone who needs it deliberately. The `refuse` policy reports the
padding that would satisfy the same margin. Every number is recorded in the simbox and bundle
manifests: cutoff, requested margin, minimum required image distance, the actual image distance,
and whether growth occurred.

This is the general fix. The smoke's small cutoff is a **speed** choice, not the remedy — the
margin protects any preparation, including production ones with a tight box at a 1.0 nm cutoff.

## Verify an installation

```bash
cd "$(mktemp -d)"                      # anywhere outside the checkout
md-openmm --help
md-openmm validate-env
md-openmm validate-system --system cyclo_rgdfv
md-openmm smoke --system small_macrocycle_smoke --out-root ./test-runs --platform CPU
```

The smoke must exit `0` with `status: completed` and at least two exchange rounds. It takes about
a minute on CPU. Again: it proves the install, not the science.
