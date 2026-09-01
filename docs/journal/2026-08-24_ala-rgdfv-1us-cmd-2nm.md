# Three box quantities that were one, and a 2.0 nm default

Instruction: `claudecode-instructions/20260824_ala-rgdfv-1us-explicit-implicit-2nm.md`, from
`d0e6ec3`. Commits `9a18e77` (geometry and defaults) and `9e70cb1` (charge method).

## The conflation

Three different lengths were being computed by one expression and reported under one name.

For an OpenMM reduced triclinic box of width `w`:

| quantity | cube | dodecahedron | trunc. octahedron | what it is for |
|---|---|---|---|---|
| shortest lattice translation | `w` | `w` | `w` | how far a periodic copy actually is |
| minimum reduced-box height | `w` | `w/√2` | `√6/3·w` | what OpenMM compares the cutoff against |
| solute bounding radius `R` | — | — | — | a property of the molecule, not the box |

The shortest lattice translation is `w` for **all three shapes**. That is not obvious, so it is
established by enumeration rather than by formula: `shortest_lattice_translation()` searches
`(i,j,k) ∈ [-2,2]³` and takes the smallest nonzero `‖i·a + j·b + k·c‖`. The heights come from
`min(a_x, b_y, c_z)` — the diagonal of the reduced vectors, which is the comparison OpenMM's own
periodic-box check performs.

Clearance between the solute and its nearest copy is therefore

```
solute_image_clearance = shortest_lattice_translation − 2R
```

and **not** `min_reduced_box_height − 2R`, which is what the code computed. For the dodecahedron
that understates the clearance by `w(1 − 1/√2) ≈ 0.29·w` — 1.17 nm at `w = 4.0`.

This reconciles a measurement that had not made sense. The 2026-08-21 rep01 box gave a
height-based figure of 3.965 − 1.604·2 = 0.674 nm, which would have been alarming, against a
measured nearest-image atom distance of 2.134 nm. The lattice-translation figure is 2.361 nm —
tight against the measurement, as a bounding argument should be, and on the correct side of it.

The bound is a bound, not an estimate: it is a sphere argument, and it was pessimistic by
1.3–1.5 nm against the true atom–atom minimum in the 2026-08-21 runs. That is why the runtime
check measures atom–atom distances over the 26 neighbouring translations rather than trusting the
sphere. Reported separately, the three quantities stop being interchangeable, and a validation
message can say which one failed.

Manifests now record all of `shortest_lattice_translation_nm`, `solute_image_clearance_nm`,
`min_reduced_box_height_nm`, `max_legal_cutoff_nm`, `required_cutoff_height_nm` and
`solute_bounding_radius_nm`. The three legacy keys (`min_image_distance_nm`,
`solute_image_gap_nm`, `minimum_image_required_nm`) are kept for readers of older bundles, next to
a `legacy_field_note` stating in the file itself that they are not the solute-to-copy distance.

`tests/test_box_geometry.py` — 35 tests, covering the eight properties the instruction asked for:
fallback vectors against installed OpenMM's, brute-force shortest translation equal to `w` for each
shape, the three height factors, the OpenMM width formula and the clearance derived from it, cutoff
growth against a real `Context`, a 1.0 nm cutoff surviving NPT contraction, the representative
ALA and RGDfV boxes, and the volume ratios 1.0 / 0.7071 / 0.7698 at equal copy spacing.

## 2.0 nm, with OpenMM semantics

OpenMM has **no numeric `addSolvent` padding default** — `Modeller.addSolvent` requires either
`padding` or explicit box vectors. 2.0 nm is this repository's choice, expressed in OpenMM's
semantics:

```
width = max(2R + padding, 2·padding)
```

which is what `Modeller._computeBoxVectors` does. For both systems here the `2·padding` term wins
(ALA `2R + p = 2.94`, RGDfV `3.48`, both below `4.0`), so both boxes are 4.0 nm wide. That is a
property of small solutes at this padding, not a coincidence to rely on.

Set in all four `explicit-*-v2` profiles: dodecahedron, `padding: "2.0 nm"`,
`padding_semantics: "openmm"`, `cutoff_fit_policy: "grow"`, 0.15 M NaCl, PME at 1.0 nm,
0.10 nm minimum-image margin. The v1 profiles stay at 1.2 nm as the historical record.

The default does not reach implicit mode: the implicit profiles carry no `solvation` block, and
`test_implicit_gbn2.py` asserts the resolved implicit configuration has no padding key.

## Ensemble and equilibration

Production is now **NPT**, not fixed-box — `"ensemble": "NPT"` in `config.py`. The barostat was
already being applied to `cMD_1` through `BAROSTAT_STAGES`; the label said NVT. The label was the
error.

Equilibration default is now 250 ps NVT + 250 ps NPT (restrained) + 500 ps NPT free, replacing
200 ps + 1000 ps.

## am1bcc_nagl

AmberTools `am1bcc` is a **single-conformer** AM1 calculation — the wrapper reports
`{'min_confs': 1, 'max_confs': 1, 'rec_confs': 1}` — so its charges depend on whichever conformer
ETKDG produced. It took ~40 minutes for the 79-atom macrocycle. `am1bccelf10` removes that
dependence at ten times the cost.

OpenFF NAGL reproduces AM1-BCC ELF10 from the molecular graph alone, with no conformer entering
the calculation at any point. Measured on cyclo-(RGDfV) against the AM1-BCC charges in this
campaign's bundle: **RMSD 0.0247 e, correlation 0.9976**, largest deviations on two carbonyl
oxygens (0.096 e) and a guanidinium hydrogen (0.051 e). 1.28 s for the charges; 5.4 s through
`build_forcefield`.

Adding it exposed a live bug. The pre-existing `nagl` branch hard-coded
`openff-gnn-am1bcc-0.1.0-rc.3.pt` — a release candidate, pinned in source, with 1.0.0 installed
beside it. Nothing in the provenance would have shown this, because only the string `"nagl"` was
ever recorded. Resolution now asks the package for production models specifically
(`production_only=True`, excluding the shipped alpha and three release candidates) and takes the
newest.

Because a trained model can be upgraded underneath an unchanged configuration file, the method name
alone does not identify a Hamiltonian. `forcefield.json` records the model file and its digest:

```json
"charge_method": "am1bcc_nagl",
"nagl_model_file": "openff-gnn-am1bcc-1.0.0.pt",
"nagl_model_sha256": "7981e7f5b0b1e424c9e10a40d9e7606d96dcd3dd2b095cb4eeff6829f92238ee"
```

`am1bcc` records its conformer scheme (`am1bcc` or `am1bccelf10`, depending on whether OpenEye is
present) for the same reason.

Offered, not defaulted. No profile changed; `ligand_charge_method` remains `am1bcc`, which is what
the runs below used. `nagl` is kept as an alias and now resolves to the same production model.

## Scope

The instruction specified twelve 1 µs runs (ALA and RGDfV × explicit and implicit × 3 replicates).
Partway through, that was **superseded by explicit direction**: production always NPT, the
250/250/500 equilibration default, and "for now just do the equilibration + 5 ns simulation for
both compounds using dodecahedron". What is recorded below is that reduced scope, delivered in
full. The 1 µs campaign, the implicit arm and replicates 2–3 were not run.

Alchemical transformations and Deeptime/MSM analysis were out of scope throughout and were not
performed.

## What ran

Data root `/path/to/MD-analysis-data/20260824_cmd_2nm/`, manifest
`campaign_manifest.json`. Nothing from the 2026-08-21 campaign was touched.

| | ALA | cyclo-(RGDfV) |
|---|---|---|
| route / profile | peptide, `explicit-md-peptide-v2` | ligand, `explicit-md-ligand-v2` |
| force field | ff19SB + OPC | OpenFF Sage 2.2.0 + TIP3P |
| charge method | — | `am1bcc` (single conformer) |
| master seed | 20260824001 | 20260824101 |
| integrator / velocity / barostat seed | 1613025927 / 1988172834 / 1091332350 | 1271759473 / 60071551 / 1442851464 |
| solute bounding radius | 0.4717 nm | 0.73923 nm |
| box width | 4.0 nm | 4.0 nm |
| shortest lattice translation | 4.0 nm | 4.0 nm |
| solute-image clearance | 3.0566 nm | 2.52154 nm |
| min reduced-box height | 2.82843 nm | 2.82843 nm |
| max legal cutoff | 1.41421 nm | 1.41421 nm |
| grown for cutoff | no | no |
| box volume (built) | 45.25483 nm³ | 45.25483 nm³ |
| particles / waters | 5722 / 1423 | 4278 / 1397 |
| `system.xml` sha256 | `dd5fa428…78abb` | `723db5c5…1ce3f57` |
| GPU | 0, `GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1` | 1, `GPU-96ce533d-9d42-acf6-8384-5e27150e9a85` |
| wall clock | 14:01:18 → 14:12:43 | 14:40:06 → 14:47:00 |

Water models differ by force-field family and are chosen by policy, not per run: ff19SB → OPC,
OpenFF Sage → plain TIP3P. RGDfV's `forcefield.json` records `protein_forcefield: null`, confirming
the ligand route does not load ff19SB into a Sage calculation.

## Runtime validation

Both runs, all checks:

| check | ALA | RGDfV |
|---|---|---|
| committed | 5000.0 ps, gen 1 | 5000.0 ps, gen 1 |
| watermarks (all-atom / solute / state) | 50 / 500 / 50 | 50 / 500 / 50 |
| restrained atoms in production | 0 | 0 |
| barostat | MonteCarloBarostat | MonteCarloBarostat |
| reduced-box height over trajectory | 2.771 – 2.797 nm | 2.770 – 2.805 nm |
| min solute–image **atom** distance | **3.062 nm** (frame 42) | **2.274 nm** (frame 15) |
| NPT volume | 42.56 – 43.74 nm³ | 42.51 – 44.13 nm³ |
| finite energies and box | yes | yes |
| verdict | PASS | PASS |

Every reduced-box height stays above `2 × cutoff = 2.0 nm`, and every solute–image atom distance
above the 1.0 nm cutoff, in every frame of both runs. The distances are true atom–atom minima over
all 26 neighbouring lattice translations, not the sphere bound — the sphere bound is what was
pessimistic by 1.3–1.5 nm before.

Volume moves in both runs, which is the check that the barostat is acting rather than merely
configured. The built box contracts about 5% on equilibration in both cases, as `addSolvent`
packing is slightly under-dense.

Neither simulation environment has mdtraj or MDAnalysis, so the validator reads the DCD directly
and reconstructs box vectors from the lengths and angle cosines the way `openmm/app/dcdfile.py`
writes them, so the periodic geometry checked is the one that ran.


## Frames, watermarks and continuity

| stream | interval | ALA | RGDfV |
|---|---|---|---|
| all-atom DCD | 25,000 steps (100 ps) | 50 declared / 50 complete, 5722 atoms | 50 / 50, 4278 atoms |
| solute DCD | 2,500 steps (10 ps) | 500 / 500, 22 atoms | 500 / 500, 79 atoms |
| state log | 25,000 steps | 50 | 50 |
| trailing bytes | — | 0 | 0 |

`declared_frames == complete_frames` with zero trailing bytes on every stream: no partial frame was
written, so no tail recovery was needed. Both runs record `tail_recovery: absent` for all three
streams, `generation: 1`, `continued: false`, `restart_source: "predecessor state"`.

**No failures, retries or checkpoint fallbacks occurred in either run.** Each ran start to finish
in one process. Continuation from a checkpoint was therefore never exercised here; the mechanism is
checkpoint-preferred with portable-State fallback, and remains covered by the test suite rather
than by this campaign.

## Throughput and disk

Measured on production (`cMD_1`) alone, one run per GPU, no sharing:

| | ALA | RGDfV |
|---|---|---|
| particles | 5,722 | 4,278 |
| `cMD_1` wall clock (5 ns) | 568 s | 336 s |
| throughput | **760 ns/day** | **1,286 ns/day** |
| minimisation + equilibration (1 ns) | 117 s | 78 s |
| all-atom trajectory | 0.687 MB/ns | 0.514 MB/ns |
| solute trajectory | 0.034 MB/ns | 0.103 MB/ns |
| total trajectory | 0.721 MB/ns | 0.617 MB/ns |
| whole run directory | 22 MB | 18 MB |
| bundle | 5.4 MB | 3.3 MB |

Extrapolated to the 1 µs runs the instruction originally specified, at these reporting intervals:
about **721 MB** and **1.3 GPU-days** per ALA replicate, **617 MB** and **0.8 GPU-days** per RGDfV
replicate. Six explicit replicates would be roughly 4.0 GB and 6.3 GPU-days. On nine GPUs that is
well under a day of wall clock, and negligible against the 6.9 TB free. The implicit arm is not
estimated here, because no implicit run long enough to measure has been performed — see *State*.

## Software

OpenMM 8.5.2.dev-36a30cb, openff-toolkit 0.19.0, openmmforcefields 0.16.0,
openff-interchange 0.5.4, openff-nagl 0.5.5, openff-nagl-models 2025.9.0. AmberTools 26 for `sqm`.

## Reproducing and extending

```bash
R=/path/to/MD-analysis-data/20260824_cmd_2nm

# regenerate a run project from its bundle (does not re-derive charges)
PYTHONPATH=/path/to/MD-tools/src \
  python MD_input_gen.py --system $R/bundles/rgdfv_explicit/system_manifest.json \
                         -o $R/explicit/rgdfv/replicate_01/run \
                         --config $R/configs/rgdfv_explicit_md.json

# launch one run on one idle GPU, under an exclusive per-GPU lock
bash $R/bin_launch.sh <name> <run_dir> <gpu_index> <segments>

# monitor
tail -f $R/logs/<name>.log
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# resume: the launcher refuses a GPU that already has a compute process, and cMD restart is
# checkpoint-preferred, so re-running the same command continues from cMD_1.chk
```

Extending to more segments is `CMD_NUMBER_OF_SEGMENTS`; the committed-generation contract makes
each segment an atomic commit, so an interrupted extension resumes at the last committed
generation rather than restarting.

## State

`dev` at `9e70cb1`. 906 fast tests pass. Both 5 ns runs complete and validated; all nine GPUs idle.
The 2026-08-21 campaign is preserved untouched.

Two decisions are open and deliberately not taken here:

* **Charge method.** `am1bcc_nagl` is available and measured. Switching it means regenerating both
  bundles, after which these runs become the `am1bcc` reference rather than the baseline.
* **Implicit arm.** No implicit run longer than 1.0 ns has ever been performed in this repository,
  and the implicit *ligand* route has never been exercised end to end. `changeRadii(st, "mbondi3")`
  runs unconditionally, which assigns mbondi3 radii to a Sage-typed macrocycle whose Amber files
  carry no GB radii at all. That is a scientific question about the model, not a plumbing gap, and
  it should be answered before an implicit production campaign, not during one.
