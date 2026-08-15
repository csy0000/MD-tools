# Explicit-solvent validation: unit tests, functional smokes, and 2 ns ladder pilots

**Date:** 2026-08-14 · **Commit under test:** `70f253e` (branch `flatbottom-basin-cft-mixture`) ·
**Scope:** validating the implementation fixed in
[`2026-08-14_explicit_solvent_fix.md`](2026-08-14_explicit_solvent_fix.md)

> **SUPERSEDED for cyclo-RGDfV, 2026-08-14.** Everything below about the RGD 8-rung ladder is
> **historical evidence only**: it is one pilot. Explicit-solvent cyclo-RGDfV uses **Sage 2.2 +
> AM1-BCC** — that is settled, not an open choice. The 8-rung `QUALIFIED` verdict below is
> superseded and the 8-rung ladder is **not** to be measured further.
>
> Under three matched 2 ns/replica repeats, ten rungs is adopted as the working cyclo-RGDfV ladder
> for the remaining validation gates. Six predeclared conditions were met; condition 5 was
> under-specified and therefore received no formal verdict. The conjunctive seven-condition rule was
> not formally satisfied. This operational adoption does not establish equilibrium mixing,
> convergence, or long-production readiness.
>
> See [`2026-08-14_rgd_ladder_8_vs_10.md`](2026-08-14_rgd_ladder_8_vs_10.md). The alanine
> conclusions here stand.

**Go/no-go: GO for alanine. The cyclo-RGDfV `QUALIFIED` verdict below is SUPERSEDED** (historical). Everything mechanical passes. The alanine
6-rung ladder is provisionally acceptable; the cyclo-RGDfV 8-rung ladder is **borderline and its
cold-end pair is not securely above the failure threshold**. Detail in §3.

The four kinds of evidence below are deliberately not mixed. A functional smoke says the code runs;
it says nothing about whether a ladder works, and a 2 ns pilot says nothing about whether a
reference is converged.

---

## 1. Unit-test evidence

```
python -m pytest tests/test_rest2_scaler.py tests/test_rest2_lambda_system.py \
                 tests/test_explicit_baseline.py -q      ->  63 passed
```

No source was changed during validation, so the full suite was not re-run; it passed
(759 passed / 3 skipped / 0 failed) on this commit when the fix landed.

All eight required scaling assertions are covered, at `s ∈ {1.0, 0.64, 0.25}` on a periodic system
with 4 solute and 4 solvent particles — solute–solute `×s`, solute–solvent `×√s`, solvent–solvent
unchanged, the three exception sectors, sigma unchanged, PME reciprocal against an independently
hand-built intended system, bonds and angles unchanged in parameters *and* isolated energy, eligible
torsions and CMAP `×s`, excluded omega unchanged.

CUDA contexts were created on all three GPUs. **Provenance note:** without
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, CUDA's enumeration does not match `nvidia-smi` — logical 0 returned
the A5000, not the A4500. Every run script pins it, and the GPU assignments below are physical.

## 2. Functional-smoke evidence

Environment: python 3.11.15, OpenMM 8.5.1 (Reference/CPU/CUDA/OpenCL), `sqm` at
`.../envs/escort-ais/bin/sqm`, driver 580.173.02, GPUs 0 = RTX A4500 (20 GB), 1,2 = RTX A5000
(24 GB).

| System | FF route | Atoms | Box gap / max cutoff | Replicas | Result |
|---|---|---:|---:|---:|---|
| alanine | ff19SB (`--pdb`) | 1817 | 1.200 nm / 1.073 nm | 6 | all rc=0 |
| `cyclo_rgdfv_sage_explicit` | **Sage 2.2 + AM1BCC** (`--smiles`) | 3226 | 1.200 nm / 1.295 nm | 8 | all rc=0 |

**Force-field provenance, stated plainly:** `systems/cyclo_rgdfv/system.yaml` records `ff19SB`. The
explicit run does **not** use it. The SMILES route implies OpenFF Sage 2.2 with AM1BCC charges and
the protomer encoded in the SMILES, and that is what ran. The historical metadata was not rewritten.
The SMILES is the stored zwitterion (carboxylate + guanidinium); the assigned net charge came out at
1.4e-15 e, i.e. neutral as intended.

Both smokes passed every listed criterion: no NaNs or CUDA errors, all checkpoints/manifests/
completion records present, production time starting at zero after relaxation, the first exchange
attempt one full interval into production, relaxation recorded as 10 ps per replica with **zero
exchanges and no reporters**, and no relaxation frames in production trajectories.

**cyclo-RGDfV omega classification — matches the predicted 5 / 0 / 0 exactly:**

```
omega_unscaled_bonds              5   [(5,4) (16,15) (24,23) (28,27) (39,38)]
omega_proline_like_scaled_bonds   0
omega_unclassified_candidates     0
method: ligand/RDKit SMARTS [CX3](=[OX1])[NX3]; proline-like = amide N in a ring of <= 7 atoms
```

This is the case the ring-size bound was written for. All five backbone nitrogens sit in the
15-membered macrocycle, so an unbounded `[NX3;R]` proline-like test would have marked every one of
them proline-like and freed all five omegas for scaling. Bounded at 7 atoms, none qualifies.
Alanine's smoke gave 2 / 0 / 0, also as expected.

Box handoff (nearest sampled state to the tail mean): alanine sample 25/50, 18.404 nm³ vs mean
18.408, **−0.02 %**; cyclo-RGDfV 32.20 ± 0.27 nm³ over 50 samples, staged protocol, 9 stages,
2000 ps, 41 atoms restrained, solute heavy-atom RMSD 0.123 nm from the minimised structure.

## 3. Ladder-acceptance evidence — 2 ns/replica pilots

Defaults exactly as configured; neither ladder was tuned after seeing results. Each pilot carried
its own 10 ps pre-exchange relaxation, excluded from the 2 ns accounting. 200 exchange rounds each;
the per-pair attempt counts match the even/odd schedule exactly (alanine 500 total, RGD 700, 100 per
pair in both, no missing adjacent pairs).

| System | FF route | Replicas | Relaxation | Production/replica | Worst-pair acceptance | Classification | Result |
|---|---|---:|---:|---:|---:|---|---|
| alanine | ff19SB | 6 | 10 ps | 2 ns | **0.400** [0.309, 0.498] | provisionally acceptable | rc=0, 374 s |
| `cyclo_rgdfv_sage_explicit` | Sage 2.2 | 8 | 10 ps | 2 ns | **0.130** [0.078, 0.210] | borderline | rc=0, 924 s |

Pair-resolved CSVs: `reports/alanine/20260814_explicit_solvent_validation/{alanine,cyclo_rgdfv}_pilot_pairs.csv`.

**Alanine, overall 0.458.** All five pairs fall in 0.40–0.50 with Wilson lower bounds ≥ 0.309 — the
chain is uniform, which is what even `√s` spacing is supposed to buy. 31 traversals and **13
end-to-end round trips** in 2 ns.

**cyclo-RGDfV, overall 0.196.** In THIS pilot the weakest pair was 0–1 (`s = 1.000 → 0.862`) at
0.130. ⚠️ **That did not reproduce.** Across three matched repeats the pooled 8-rung minimum is
pair **6–7** at 0.173, at the hot end of the ladder. The mechanistic explanation once given here —
that the weakness sits where the solute–solvent cross term is largest — is **withdrawn as
unsupported**; it predicts a cold-end bottleneck and the pooled data do not show one. 4 traversals
and **1 round trip**.

**The borderline verdict is itself not secure, and that matters more than the label.** Pair 0–1's
Wilson interval is [0.078, 0.210]: it straddles the 0.10 boundary, so this pilot cannot distinguish
"borderline" from "failed" for that pair. Reporting it as *borderline* on the point estimate alone
would overstate what 100 attempts support. The honest statement is that the cold end of the 8-rung
RGD ladder is the thing to fix or measure harder before any long run.

Classification thresholds (≥ 0.20 / 0.10–0.20 / < 0.10) are the predeclared **pilot** decision from
the instruction, not universal REST2 criteria, and are recorded as such in the summary JSON.

## 4. Work not performed

* **No 1 µs production**, by instruction.
* **Restart regression tests (§6) not written**: the direct test of the relaxation path (per-replica
  propagation, distinct deterministic velocity seeds, checkpoint reuse, partial-relaxation redo) and
  the interrupted-REST2-resume test. The underlying code paths are exercised indirectly — the smokes
  show relaxation running, being recorded and being excluded from production, and
  `completed_prefix` has unit tests for contiguous prefixes, gaps and missing checkpoints — but the
  end-to-end resume was not driven.
* **Alanine Phase A was reused, not re-run**; its artifacts were sufficient.
* **`conda env create` against the pinned environment.yml still unsolved** (a dry run exceeded an
  hour). Everything here ran in the pre-existing `escort-ais` environment.

## 5. Remaining scientific risks

1. **The RGD weak pair — at the HOT end, not the cold end** (pooled pair 6–7). Historical: the
   working ladder is now ten rungs, and the 8-rung ladder is not measured further —
   changing the ladder after seeing 100 attempts is fitting to noise. A longer pilot on the current
   ladder is the cheaper and more defensible next step.
2. **One round trip in 2 ns on RGD** is not evidence the ladder cannot mix. It is "not observed
   beyond once within the pilot"; the traversal count (4) is the more informative number at this
   length.
3. **ff19SB + TIP3P-FB** on alanine remains unvalidated against ff19SB + OPC, its parameterisation
   partner.
4. **RESOLVED: explicit-solvent cyclo-RGDfV is Sage 2.2 + AM1-BCC.** `ff19SB` is a protein force
   field and is not an alternative Hamiltonian for this macrocycle. The legacy field in
   `systems/cyclo_rgdfv/system.yaml` is historical metadata and is neither read nor rewritten.
5. **4 fs against 2 fs**, and the staged equilibration on a macrocycle, are still unmeasured — the
   RGD staged run completed cleanly with a 0.123 nm solute RMSD, which is reassuring but is not a
   test of the timestep.

## Reproduce

```bash
conda activate escort-ais
python -m pytest tests/test_rest2_scaler.py tests/test_rest2_lambda_system.py \
                 tests/test_explicit_baseline.py -q
bash ~/.claude/jobs/29daa285/tmp/phaseA_rgd.sh     # Phase A, cyclo-RGDfV
bash ~/.claude/jobs/29daa285/tmp/phaseB.sh         # both 2 ns pilots
python scripts/rest2_pilot_acceptance.py --exchange-log <run>/pilot_exchange_attempts.csv \
       --out-prefix <prefix> --system <name> --expected-rounds 200
```

Run directories (not committed; trajectories and checkpoints stay there):

```
~/.claude/jobs/29daa285/tmp/phaseA_alanine/
~/.claude/jobs/29daa285/tmp/20260814T1500_phaseA_cyclo_rgdfv/
~/.claude/jobs/29daa285/tmp/20260814T1520_phaseB_alanine/
~/.claude/jobs/29daa285/tmp/20260814T1520_phaseB_cyclo_rgdfv/
```
