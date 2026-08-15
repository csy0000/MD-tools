# Explicit-solvent baseline: review fixes

**Date:** 2026-08-14 · **Scope:** `docs/implementation/explicit_solvent/`,
`src/escort_ais/systems/explicit_baseline.py` · **Status:** implemented, tested on CPU and in a
three-GPU functional smoke; the acceptance pilot that would validate the ladders has NOT been run

The explicit-solvent baseline written earlier the same day (commits `af8c795`, `edd86af`, `163b360`)
went through review, and the review found seven classes of problem. This records what changed and,
where a judgement call was needed, why it went the way it did.

## The Hamiltonian this is all in service of

```
U_s = s·U_solute-solute + √s·U_solute-solvent + U_solvent-solvent
```

with eligible solute torsions and CMAP scaled by `s`, and **bonds and angles never scaled**. That
was already what `openmm_system.build_rest2_scaled_system` implemented; what was missing was any
test that could tell the middle term from the first, because every existing REST2 test ran on an
all-solute implicit system where `U_s = s·U` and any consistent scaling passes.

## What changed

### 1. Environment

`environment.yml` gained `numba` — declared in `pyproject.toml`, and the install is
`pip install -e . --no-deps`, so pip would never have pulled it in — plus `pip`, `setuptools>=64`,
`pytest`, `git`, and `nodefaults` after `conda-forge` to stop channel mixing. `cuda-version=13.0` is
now labelled as the Linux/CUDA line with the CPU-only alternative spelled out.

The file no longer describes itself as reproducing the validation environment exactly. It pins
primary package versions; build strings and transitive dependencies are unpinned, so it is
"validated primary package versions", not a lock. Calling it a lock would have been the kind of
claim that is only discovered to be false when someone needs it to be true.

### 2. Pre-exchange relaxation (`production.remd.equilibration_ps`, default 10 ps)

Every replica starts from the same equilibrated coordinates, and those were equilibrated at `s = 1`.
A replica at `s < 1` therefore holds a configuration its own Hamiltonian did not produce: its solute
is suddenly softer than the structure it is sitting in. Exchanging immediately feeds that transient
straight into the swap statistics and the trajectories.

Each replica is now propagated for 10 ps under **its own** Hamiltonian with no reporters and no
exchange attempts, and the result is discarded. Production time and step numbering reset to zero
afterwards, so production logs exclude it. It is restart-safe: the completion record is written only
after every replica's checkpoint is on disk, so a partial relaxation is detected and redone — safe
precisely because this runs only when no production output exists. It is not repeated on resume.

**This is pre-exchange relaxation, not a claim of equilibrium.** 10 ps settles the local solvent
response to a scaled solute and nothing more.

### 3. Omega classification — the substantive change

The previous detector took every solute C–N bond whose carbon carried one oxygen and excluded it
from torsion scaling. Two things are wrong with that. It treats an X–PRO peptide bond as an ordinary
omega, when a proline nitrogen is ring-locked and its torsion is not the near-planar two-state
coordinate the exclusion exists to protect. And "a carbon next to one oxygen and one nitrogen" also
describes carbamates, ureas and carbamic acids.

`classify_omega_bonds` replaces it with two auditable routes and four recorded outputs
(`omega_unscaled_bonds`, `omega_proline_like_scaled_bonds`, `omega_unclassified_candidates`,
`omega_detection_method`).

**Peptide route** — residue-aware. Proline-like character is read from the residue containing the
amide *nitrogen*, against a configurable name set (`rest2.proline_like_residues`, `["PRO"]`).

**Ligand route** — bond-order aware, from the retained SDF, because a SMILES-built solute is one
`UNL` residue with no residue evidence at all. Ordinary amides match `[CX3](=[OX1])[NX3]`.

Two judgement calls worth recording:

* **The ring-size bound is load-bearing.** The obvious proline-like test, `[NX3;R]`, is wrong for
  this project: every backbone nitrogen of a cyclic peptide is "in a ring", so an unbounded test
  would free *every macrocyclic omega* for scaling — on the systems this baseline exists for. A
  nitrogen counts as proline-like only in a ring of at most `rest2.max_proline_ring_size` (7) atoms.
  Verified both ways: cyclo-tetraglycine (12-ring) gives 4 unscaled and 0 proline-like;
  N-methyl-2-pyrrolidinone (5-ring) gives 1 proline-like and 0 unscaled.
* **An unrecognised residue blocks rather than defaulting.** The first implementation treated any
  inter-residue amide as ordinary, on the reasoning that a peptide bond is a peptide bond. That
  silently mis-handles HYP and every other non-standard residue — exactly the cases that most need a
  human. Unknown nitrogen residues now land in `omega_unclassified_candidates`, and
  **any non-empty unclassified list blocks production**: scaling a torsion and not scaling it are
  different Hamiltonians, so guessing would change the estimand without a trace.

The RDKit→OpenMM atom mapping is **asserted**, not assumed: element sequence and the full bond graph
must agree, and a mismatch raises. They coincide today because the SDF and the PDB are written from
the same RDKit molecule in the same order, but that is a property of the pipeline, not a guarantee,
and a silent off-by-one would scale the wrong torsions.

### 4. Default ladders

Alanine **6** rungs, macrocycles **8**, both spaced evenly in `√s` from 1 to 0.25:

```
alanine     [1.0, 0.81, 0.64, 0.49, 0.36, 0.25]
macrocycle  [1.0, 0.862245, 0.734694, 0.617347, 0.510204, 0.413265, 0.326531, 0.25]
```

These remain **unvalidated starting ladders**. Eight rungs does not guarantee adequate macrocycle
exchange, and `generate_config.py` prints that warning every time it writes a REST2 config.

### 5–6. Everything else the review caught

* **Route classification follows the input.** The alanine preset pinned `solute_kind: peptide`, which
  contradicted a `--smiles` invocation and then failed deep inside template matching with
  "No template found for residue 0 (UNL)". The preset now says `auto`; SMILES implies Sage,
  a residue-named PDB implies ff19SB, and a declared route that contradicts the input is rejected
  **before** any hydrogen is deleted or any force field is built.
* **Ligand protonation is documented for what it is.** `addHydrogens(pH=7)` chooses a variant per
  *supported residue template*; it has no rules for a ligand and does not titrate one. The ligand
  route uses the protomer and formal charges encoded in the SMILES, which is the only place that
  information exists. The input SMILES and the resulting net charge are recorded.
* **Restart integrity.** `start_chunk = len(done_files)` was wrong: with chunks 0 and 2 complete and
  1 missing it returns 2 and silently restarts past a hole. `completed_prefix` validates a
  contiguous prefix and rejects gaps and missing checkpoints. REST2 additionally requires the same
  prefix on every replica — they advance in lockstep between exchange rounds, so a ragged set is not
  reconstructible — and the exchange log must end exactly at the production boundary, with rows past
  it truncated so a resume neither duplicates nor skips a round.
* **Box handoff.** Pasting a mean box onto the final instantaneous configuration pairs coordinates
  equilibrated in one cell with a cell they never saw. The handoff is now the sampled state whose
  own volume is closest to the tail mean — a configuration and a box that actually occurred
  together — with the method, mean volume, selected volume and deviation recorded. Measured on
  alanine: sample 25 of 50, −0.02 % from the mean.

## Evidence

`tests/test_explicit_baseline.py`, 46 tests, CPU-only, seconds to run.

Three-sector scaling is checked numerically at `s ∈ {1.0, 0.64, 0.25}` on a small periodic system
built with 4 solute and 4 solvent particles, PME, one exception per sector, bonds, angles, torsions
and CMAP: solute–solute pair energy ×`s`, solute–solvent ×`√s`, solvent–solvent ×1; the three
exception classes likewise; sigma never scaled; PME compared against an independently hand-built
"intended" system rather than against the implementation; dispersion correction preserved; bonds and
angles identical parameter-by-parameter *and* in isolated energy; eligible torsions and CMAP ×`s`
while excluded omegas and solvent-straddling torsions do not move.

Two traps found while writing those tests, both worth remembering: a `Force` proxy obtained from a
temporary `System` dangles once the System is collected and silently returns uninitialised doubles
(`1.3e-313`); and two torsions sharing a central bond are both excluded by one
`exclude_central_bonds` entry, which made a "must scale" assertion fail for the right reason.

**Test results.** Focused set (`test_rest2_scaler`, `test_rest2_lambda_system`,
`test_explicit_baseline`): 63 passed. Full `-m "not slow"` suite: **759 passed, 3 skipped, 1
deselected, 0 failed**. No pre-existing failures, no regressions.

**Three-GPU functional smoke, alanine** (`~/.claude/jobs/29daa285/tmp/phaseA_alanine/`):

| Method | Replicas | Relaxation | Production | GPU | Result | ns/day |
|---|---:|---:|---:|---:|---|---:|
| REST2-REMD | 6 | 10 ps | 40 ps | 0 (A4500) | rc=0 | — |
| cold MD | 1 | — | 40 ps | 1 (A5000) | rc=0 | 2854 |
| hot MD, s=0.25 | 1 | — | 40 ps | 2 (A5000) | rc=0 | 2740 |

Box 1817 atoms, solute–image gap 1.200 nm, largest legal cutoff 1.073 nm. Omega: 2 unscaled, 0
proline-like, 0 unclassified. The relaxation recorded 0 exchanges and the first exchange row is at
step 2500, so production starts at zero and excludes it. GPU 0 is the newly-installed A4500 still on
probation and completed without incident.

## Not done

* **Phase B acceptance pilots (2 ns) on either system.** This is the step that would produce a
  pair-resolved acceptance table, and it is the one thing worth doing before any long run. Phase A's
  acceptance over ~10 attempts at 40 ps is not interpretable and is not reported as a result.
* **Phase A for `cyclo_rgdfv`.** Its `system.yaml` declares ff19SB; the explicit route needs the
  SMILES/Sage path, and rewriting historical system metadata to suit a new test would be the wrong
  fix. ~~Whether a usable SMILES is present there has not been checked.~~
  **Corrected 2026-08-14:** that last sentence was wrong — `systems/cyclo_rgdfv/system.yaml` does
  contain a usable zwitterionic SMILES, and it should have been read before the claim was written.
  Phase A has since been run on it through the Sage 2.2 + AM1BCC route, giving the predicted
  5 / 0 / 0 omega classification. See
  [`2026-08-14_explicit_solvent_validation.md`](2026-08-14_explicit_solvent_validation.md).
  **Settled since:** explicit-solvent cyclo-RGDfV is Sage 2.2 + AM1-BCC; ff19SB is not an
  alternative Hamiltonian for a macrocycle.
* **`conda env create` against the pinned file.** A dry-run solve ran over an hour without
  finishing. The pins come from the live environment so they are mutually consistent, but "conda can
  solve this today" is unverified.
* **ff19SB + TIP3P-FB**, the 4 fs timestep against 2 fs, and the staged equilibration on an actual
  macrocycle all remain unvalidated, as recorded in the setup's own README.

Related: `docs/implementation/explicit_solvent/README.md` (reviewer's guide, kept current),
`docs/implementation/explicit_solvent/baseline_setups.md` (authoritative for this setup).
