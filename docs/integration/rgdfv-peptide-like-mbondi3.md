# c(RGDfV) with verified mbondi3 corrections: `solute.kind: peptide-like`

**Date:** 2026-09-08
**Engine pin:** `a0bf0cab9b3c999d23521ecd2e7b7bc66fac7cb4` (MD-tools `dev`)
**Project:** `../RGDfV-REST2/`, sibling of the MD-tools checkout
**Supersedes the MODEL, not the record.** The earlier run stands as history:
[`rgdfv-rest2.md`](rgdfv-rest2.md) documents the uncorrected Hamiltonian and its results, and is
not rewritten here.

---

## 1. The defect, measured before anything changed

`mbondi3` is `mbondi2` plus two side-chain corrections, and ParmEd selects them by **residue and
atom name** — verified in the installed `parmed 4.3.1`
(`parmed/tools/changeradii.py::mbondi3`): `OD*`/`OE*` in `GLU/ASP/GL4/AS4` → 1.4 Å, `HH*`/`HE*`
in `ARG` → 1.17 Å, plus an `OXT` C-terminal rule that a head-to-tail cycle never triggers.

A solute built from SMILES is **one residue with one invented name**, so none of those rules
match. Measured on the registered historical System:

| atoms | intrinsic radius | mbondi3 wants |
|---|---|---|
| 2 × Asp carboxylate O (21, 22) | **1.5000 Å** | 1.40 |
| 5 × Arg guanidinium H (73–77) | **1.3000 Å** | 1.17 |

**0 of 7 corrected**, while the build reported `radii: mbondi3`.

---

## 2. What was implemented

### `solute.kind`

```yaml
solute:
  kind: peptide-like          # peptide | peptide-like | ligand
  ligand_forcefield: sage-2.2.1
  ligand_charge_method: am1bcc
solvent:
  model: GBn2
constraints:
  type: HBonds
hydrogen_mass_repartitioning:
  enabled: false
```

`peptide-like` is the **ligand route plus a map** — same force field, same charge method, same
builder. It never loads a protein force field and never replaces Sage's charges or bonded terms.

The retired `solute.peptide` boolean is folded into `kind` on the **raw document, before
defaults**. That ordering is the whole of the compatibility rule: after defaults, every
configuration carries `kind: peptide` whether or not anyone wrote it, and a stated
`peptide: false` would look like a conflict with a value nobody chose. `true` → `peptide`,
`false` → `ligand`; both stated is accepted only when they agree; `peptide-like` has no boolean
spelling, so either value contradicts it and is refused with a migration message.

### The molecular map

`md_tools/openmm/peptide_map.py`. Residues are read off the graph: side chains are compared as
**canonical SMILES of the side-chain fragment** with a `*` at the attachment point, so leucine
and isoleucine are distinguished and atom order and names are irrelevant. Protonation is part of
the identity — `ASH` is not `ASP`, `ARN` is not `ARG` — and handedness is derived per residue
because **L-Cys is R** while every other L residue is S.

It refuses rather than guessing: a linear peptide, an unsupported side chain, unspecified
stereochemistry, or a disconnected input each raise naming the residue and pointing at
`solute.kind: ligand`.

### Where the correction lands, and why that IS the correctness argument

Between `changeRadii(...)` and `createSystem(...)`.

`createSystem` derives, from each radius: the offset radius `or = radius − offset`, the scaled
offset radius `sr = screen × or`, and `radindex`, an index into the GBn2 neck lookup table.
Measured on the corrected build, `radindex` changed on **all 43 atoms** of a test molecule
because that table is keyed on the *set* of distinct radii — adding 1.40 and 1.17 shifts every
index. A radius corrected after `createSystem` would leave `sr` and `radindex` describing the old
one, and the System would disagree with itself inside a single force.

**Stricter than ParmEd in one place, deliberately:** ParmEd also corrects `AS4`/`GL4`, the
*protonated* variants, because their names start the same way. Chemistry does not: a neutral
carboxylic acid is not a carboxylate and is not corrected.

---

## 3. Test results

| group | file | result |
|---|---|---|
| A — configuration | `test_solute_kind_config.py` | **27 passed** |
| B — chemistry | `test_peptide_map.py` | **23 passed** |
| C — radii | `test_peptide_like_radii.py` | **9 passed** |
| D/E — Hamiltonian and integration | `test_peptide_like_hamiltonian.py` | **9 passed** |
| fast suite | `pytest tests -m "not slow and not gpu"` | **1444 passed** |
| CUDA/MPI lanes | `test_cv_mpi_cuda_lanes`, `test_cv_mpi_cuda_rrest2`, `test_cv_cuda_lanes`, `test_cmd_cuda_smoke`, `test_rrest2_cuda_smoke` | **52 passed** (real CUDA, real `mpiexec`) |

The radius expectations are **not** produced by the new mapper. `changeRadii("mbondi3")` is
called on a ParmEd structure whose residues are named `ASP` and `ARG` — the situation the rule
was written for — and the values it assigns are what the peptide-like build must match, having
reached them from chemistry.

Measured on cyclo(Gly-Asp-Arg) built both ways: exactly **2 O 1.5000 → 1.4000 Å** and **5 H
1.3000 → 1.1700 Å**, every other intrinsic radius unchanged, screen factors untouched, and an
explicit-solvent build acquiring **no GB force at all**.

---

## 4. The corrected integration run

Built with the installed wheel from `a0bf0ca`:
`.../RGDfV-REST2/data/engine/venv/lib/python3.12/site-packages/md_tools/__init__.py`

### Chemistry and charges

| | |
|---|---|
| mapped sequence | VAL(L) – ARG(L) – GLY(achiral) – ASP(L) – PHE(**D**) |
| map digest | `fcfceff4620402ea9d268aaf5e5e16a8a70acac443fc8b28078d85e1b5b7b09e` |
| cyclic links | identical to the historical map, atom for atom |
| charge method | recomputed **AmberTools AM1-BCC** (`sqm`), not ELF10 or NAGL |
| charge digest | `6d9e6ed6104f8ff290a92f6240deed45209fc5987560f0bf0a8434e262073988` |
| **charges vs historical** | **identical, 0 of 79 atoms differ** |
| net charge | −2.54e−08 e |

Because the charges reproduced exactly, this is a **controlled comparison**: the corrected
dataset differs from the historical one in the seven radii and in nothing else. That was not
guaranteed in advance — see *Limitations*.

### The corrections actually applied

| atom | group | before | after |
|---|---|---|---|
| 21 | Asp carboxylate O | 1.5000 Å | **1.4000 Å** |
| 22 | Asp carboxylate O | 1.5000 Å | **1.4000 Å** |
| 73 | Arg guanidinium H | 1.3000 Å | **1.1700 Å** |
| 74 | Arg guanidinium H | 1.3000 Å | **1.1700 Å** |
| 75 | Arg guanidinium H | 1.3000 Å | **1.1700 Å** |
| 76 | Arg guanidinium H | 1.3000 Å | **1.1700 Å** |
| 77 | Arg guanidinium H | 1.3000 Å | **1.1700 Å** |

Intrinsic radii, reconstructed as `or + 0.0195141 nm` with the offset read from the System's own
GB expressions. Every other intrinsic radius is unchanged from the historical build.

### Force-level omega exclusion, on the corrected System

The automatic classifier and the map name the **same five bonds** — `(4,5) (15,16) (23,24)
(27,28) (38,39)` — with 0 proline-like and 0 unclassified.

| tau | omega terms unchanged | eligible terms scaled | violations |
|---|---|---|---|
| 0.00 | 40 | 326 | 0 |
| 0.25 | 40 | 326 | 0 |
| 0.50 | 40 | 326 | 0 |

Tolerance `rel = 1e-9`, recorded before inspection.

### The run

Six states τ = 0, 0.1, 0.2, 0.3, 0.4, 0.5 at 300 K; 2 fs, HMR off, HBonds, nonperiodic GBn2, no
SASA; minimise then **100 ps per-state equilibration at each state's own Hamiltonian**; 1 ns
production per state; exchange every 10 ps; 15 mapped CVs every 1 ps; trajectory and checkpoint
every 10 ps. Six MPI ranks, one per GPU on devices 0–5, all otherwise idle.

**Interrupted** with SIGINT at 251 ps of production. Committed checkpoint: step **128000**,
exchange 24, `cv_rows` **257** = 128000/500 + 1 — progress and CV record agreeing exactly.
**Resumed** to the original budget: `continuation: bitwise`, `steps completed 500000 of 500000`.

| check | result |
|---|---|
| CV grid | **1001 rows per state**, steps 0…500000 every 500, exactly once, all six states |
| tau per series | each series carries only its own state's tau |
| mapped CVs vs coordinates | **450** torsions recomputed from matching frames, worst **3.27e−05°** |
| pre-exchange convention | respected: only rows naming a trajectory frame were recomputed |
| cyclic integrity | longest backbone link **0.1467 nm** across sampled frames of all six states |
| completion | `run_status: completed` |

Old checkpoints were never resumed under the changed Hamiltonian, and no historical file was
altered.

### Descriptive diagnostics — no convergence claim, no threshold

Acceptance, cumulative over 100 exchanges:

| pair | accepted | rate |
|---|---|---|
| 0 ↔ 1 | 0/50 | 0.000 |
| 1 ↔ 2 | 13/50 | 0.260 |
| 2 ↔ 3 | 16/50 | 0.320 |
| 3 ↔ 4 | 2/50 | 0.040 |
| 4 ↔ 5 | 13/50 | 0.260 |
| overall | 44/250 | 0.176 |

Arg guanidinium N ⋯ Asp carboxylate O minimum distance, per state over 100 frames (nm):

| state | min | mean of per-frame minimum |
|---|---|---|
| 0 | 0.2934 | 0.8738 |
| 1 | 0.6741 | 1.2340 |
| 2 | 0.7904 | 1.2825 |
| 3 | 1.0324 | 1.2798 |
| 4 | 0.9808 | 1.2708 |
| 5 | 0.8961 | 1.2436 |

Reported as measured. The ladder was not tuned during the test, and nothing here supports a claim
about which model is more correct.

### Registration

| | |
|---|---|
| dataset id | `rgdfv-rest2-rgdfv-sage221-gbn2-rest2-corrected-mbondi3` |
| destination | `/path/to/DATA/2026/RGDfV-REST2/RGDfV-Sage221-GBn2-REST2-corrected-mbondi3` |
| inventory | **54 files, 5.8 MB, all digests verified** |
| creator | `chen` |

Dry run → transaction → `--verify-only`, all passing. The project path is now a symlink into
managed storage, and trajectories and CV series reopen through it. **The historical dataset was
re-verified afterwards and is intact: 57 files, manifest valid, all digests match.**

---

## 5. Limitations

- **This is still an exploratory Sage + GBn2 hybrid.** Applying mbondi3's corrections makes the
  radii what the model asks for; it does **not** establish that GB-Neck2 — fit in the
  ff99SB/ff14SB lineage — is appropriate for OpenFF valence and vdW parameters. Conformational
  accuracy is not established by this change and no comparison here bears on it.
- **1 ns per state is integration behaviour, not converged populations.** The acceptance and
  contact numbers above are observations of one short run.
- **The clean charge comparison was luck, not a guarantee.** AM1-BCC charges are not reproducible
  between builds of the same input in this build — 41 of 43 atoms differed on a 43-atom test
  molecule, sequentially — because the OpenFF toolkit discards the seeded ETKDG conformer and
  generates its own. RGDfV happened to reproduce exactly. Recorded as `docs/backlog.md` entry 6
  and **not fixed**: passing `use_conformers=` would change every future build's charges, which is
  a scientific decision rather than a bug fix.
- **Scope.** v1 supports an unambiguous head-to-tail α-peptide cycle of canonical side chains.
  Linear peptides, modified side chains and terminal `OXT` corrections are out of scope and are
  refused rather than approximated.

## 6. Reproducing it

```bash
md-openmm build-top -i RGDfV.smi -os built.xml -op built.pdb -log built.log \
    --config build.corrected-mbondi3.config      # solute.kind: peptide-like
md-openmm build-md -odir ./REST2 --config REST2.corrected-mbondi3.config
python REST2/min.py -p built.pdb -s built.xml -log min.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 mpiexec -n 6 python REST2/REST2.py \
    -p built.pdb -s built.xml -c min.xml -odir ./prod
```

Project files, configurations and scripts are in the sibling project; the data are in the
registered dataset above.
