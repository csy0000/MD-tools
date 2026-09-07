# Integration experiment: c(RGDfV) six-state REST2, Sage 2.2.1 + AM1-BCC, GBn2

**Date:** 2026-09-07
**Project:** `../RGDfV-REST2/`, a sibling of the MD-tools checkout (here
`/path/to/scheme/RGDfV-REST2`)
**Engine pin used:** `8e95a589aebddb945fe160764dc34999fddade75`
**Engine pin requested:** `443fc736f3a8f426414e244a50a1a438d2d8a497` — changed by two blockers,
both fixed here and both with focused regression tests.

The engine was installed into the project from an isolated checkout and wheel
(`data/engine/venv`). The user's MD-tools checkout was never reset or modified by the experiment.

```
import origin: .../RGDfV-REST2/data/engine/venv/lib/python3.12/site-packages/md_tools/__init__.py
```

---

## 1. The molecule

Head-to-tail **cyclo(L-Arg–Gly–L-Asp–D-Phe–L-Val)**, non-N-methylated.

```
CC(C)[C@@H]1NC(=O)[C@@H](Cc2ccccc2)NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)[C@H](CCCNC(N)=[NH2+])NC1=O
```

**Origin.** Constructed for the 2026-08-28 MD-projects pipeline test, where no cyclo(RGDfV)
structure was available outside the archive. It was **re-validated here from the molecular graph**
rather than trusted: `scripts/validate_molecule.py` is the validation, not a restatement of it.

| claim | result |
|---|---|
| one connected molecule, cyclic | 1 fragment |
| backbone ring | **15-membered**, found as the one ring containing all five amide nitrogens |
| backbone amide links in that ring | **5**, including the **Val→Arg ring closure** |
| sequence, read off the graph via side chains | ARG(L) – GLY(achiral) – ASP(L) – PHE(**D**) – VAL(L) |
| formula / mass / net charge | C26H38N8O7 / 574.64 / **0** |
| charged groups | Arg guanidinium **+1**, Asp carboxylate **−1** (checked as groups, not as a total) |
| free backbone termini | **0** |
| atoms | 41 heavy, **79** with hydrogens |

L/D is derived per residue from the CIP code and substituent pattern, not reported as a raw CIP
letter — L-Cys is R while every other L-amino acid is S, and conflating the two is how a
D-residue passes review.

**Atom mapping.** `scripts/map_and_define.py` matches the validated molecule onto the built SDF
with `useChirality=True` and then *asserts* the correspondence — same element at every mapped
position, same bond between every mapped pair. Every downstream index (five omega bonds, fifteen
torsions, the charge table) comes from that map; none is hand-copied.

| residue | N | CA | C | O |
|---|---|---|---|---|
| ARG0 (L) | 38 | 30 | 28 | 29 |
| GLY1 | 27 | 26 | 24 | 25 |
| ASP2 (L) | 23 | 18 | 16 | 17 |
| PHE3 (D) | 15 | 7 | 5 | 6 |
| VAL4 (L) | 4 | 3 | 39 | 40 |

---

## 2. Parameterisation — what was *actually* assigned

`solute.peptide: false` with a `.smi` input, which is the line that makes this a Sage test: a
peptide PDB would load the protein force field and never exercise Sage.

| | |
|---|---|
| intramolecular force field | **openff-2.2.1** (requested `sage-2.2.1`) |
| charge method | **`am1bcc`** — AmberTools `sqm`, ~25 min of real semi-empirical calculation |
| charge scheme recorded | `charge_scheme: am1bcc` |
| toolkits registered | NAGL, RDKit, **AmberTools**, BuiltIn — **no OpenEye** |
| openff-toolkit | 0.19.0 |
| constraints / timestep | HBonds (38 constraints) / 2 fs |
| HMR | off; masses are the force field's own |

**The ELF10 hazard did not fire, and that was verified rather than assumed.** The engine selects
`am1bccelf10` when OpenEye is registered (`scheme = "am1bccelf10" if _has_openeye() else
"am1bcc"`). OpenEye is not in this environment's toolkit registry, `sqm` was observed running, and
the record says `am1bcc`. This is the AmberTools AM1-BCC route the experiment required.

**Verification** (`reports/parameterization.json`, from the serialised System):

| check | result |
|---|---|
| particles | 79 = the molecule's atom count |
| net charge | **−2.53e−08 e** (tolerance 1e−05 e) |
| charges finite, sigmas positive | yes — assignment is complete |
| bonds / angles / torsions | 80 / 141 / 366 |
| nonbonded exceptions | 424 |
| Arg0 group charge | **+0.892 e** |
| Asp2 group charge | **−0.873 e** |
| stereochemistry and ring closure | preserved (the mapping match is chirality-aware) |

### GBn2 is a separate modelling choice, and mbondi3 was a no-op

The build log says so itself:

```
[implicit] note: no GLU/ASP/GL4/AS4/ARG residue is present, so mbondi3's residue-specific
adjustments cannot apply and the radii are exactly mbondi2.
```

Confirmed independently from the serialised System — this is what the dynamics actually used:

| atoms | GB radius (nm) |
|---|---|
| Asp2 carboxylate oxygens | 0.130486 |
| **every other oxygen** | 0.130486 |
| Arg0 guanidinium hydrogens | 0.110486 |
| every other hydrogen | 0.100486, 0.110486 |

`asp_oxygens_differ_from_other_oxygens: false`, `arg_hydrogens_differ_from_other_hydrogens:
false`. The single-residue ligand representation prevents the residue-name-keyed corrections from
firing, so **the requested radius change was a no-op and the radii are mbondi2** — on precisely
the two groups mbondi3's corrections were written for.

Nothing was relabelled and no radius was invented to make a check pass. GB coverage is complete
(79/79 particles, all radii positive), so this is an applicability limitation, not a missing
parameter. **This is an explicitly labelled exploratory Sage+GBn2 integration test and makes no
claim of conventional peptide mbondi3 parity.**

---

## 3. Omega exclusion, proved at the force level

The **automatic** classifier was run unprimed — given the built System and the retained SDF and
nothing else — and its output compared with the five bonds derived independently from the source
chemical graph.

```
detection: ligand/RDKit SMARTS [CX3](=[OX1])[NX3];
           proline-like = amide N in a ring of <= 7 atoms; SDF built.sdf
```

| from → to | central bond | torsion terms protected | |
|---|---|---|---|
| ARG0 → GLY1 | C28–N27 | 8 | |
| GLY1 → ASP2 | C24–N23 | 8 | |
| ASP2 → PHE3 | C16–N15 | 8 | |
| PHE3 → VAL4 | C5–N4 | 8 | |
| VAL4 → ARG0 | C39–N38 | 8 | **ring closure** |

- resolved set **equals** the graph-derived set — exactly five ordinary backbone amides;
- the ring-closing bond **is** included;
- **0 proline-like**: every backbone nitrogen of this macrocycle lies in a ring, and the 7-atom
  bound is what stops a 15-membered ring being mistaken for a pyrrolidine;
- **0 unclassified**, so nothing was ambiguous and nothing unrelated was excluded;
- the SDF→topology index mapping is demonstrated by the asserted atom map above.

### Term-by-term comparison of the scaled Systems

Tolerances were recorded in `reports/tolerances.md` **before** any of this ran: `rel = 1e-9` (the
repository's own value, from `tests/test_gb_linear_scaling.py`), and exact integer equality for
periodicities and term counts. Every PeriodicTorsionForce term is matched in either atom order —
a torsion may be stored `i-j-k-l` or `l-k-j-i`, and matching one direction checks half of them.

| tau | (1−tau)² | omega terms **unchanged** | eligible terms **scaled** |
|---|---|---|---|
| 0.00 | 1.0000 | 40 | 326 |
| 0.25 | 0.5625 | 40 | 326 |
| 0.50 | 0.2500 | 40 | 326 |

40 + 326 = 366 = every torsion in the System.

For all 40 omega terms, at every rung: **periodicity, phase and force constant unchanged**, and
the term set is identical to the original — the torsion is **retained**, not deleted, restrained
or dropped.

Positive and negative controls, all at `rel = 1e-9`:

- eligible non-omega solute torsions scale by exactly **(1−tau)²**;
- nonbonded charges by **√s**, epsilons by **s**, 1-4 exceptions (chargeprod and epsilon) by **s**;
- the CustomGBForce survives scaling (GB scales by **(1−tau)**, per `REST2_IMPLEMENTATION` v2);
- **bonds and angles unchanged**, parameter by parameter;
- **tau = 0 reproduces the original System's serialisation byte for byte.**

An omega trajectory is a useful diagnostic but is not the proof, and is not used as one here.

---

## 4. Two integration blockers, found and fixed

Both were found by trying to run the experiment, and both are fixed upstream with focused
regression tests. Neither is a workaround inside the project.

### 4.1 Per-state equilibration was unreachable — `b555f2d`

`ReplicaRun._equilibrate` has always relaxed each rung against **its own scaled System**, before
the first exchange and outside the production budget. But `protocol_file_text` passed
`equilibration_ps=0.0` as a literal and the `rest2` config section had no field to say otherwise.

This matters because a ladder takes **one** coordinate file — `_require_homogeneous_groups`
refuses a group file naming different `coordinates` per line, deliberately, since rungs must be
states of the same system. So without it every rung opens from a tau = 0 configuration, the hot
rungs spend their first exchanges relaxing out of a distribution that is not theirs, and those
samples are production by every record that describes them. Nothing looks wrong.

`rest2.equilibration_steps` now exists, defaults to 0, and is logged whether or not it was asked
for. Two things the first draft missed, both caught by existing contracts: the *run-time*
reconstruction (`ladder_from_resolved`) needed the field separately from the build-time one, and
`test_md_run_inputs.py` requires every resolved field to be expressible in the Amber-style `.in`
language.

### 4.2 A ligand's omega bonds could not be classified at all — `8e95a58`

A REST2 ladder could not be run over a Sage-parameterised molecule. Both routes were closed:

- the **peptide** route refuses every backbone amide of a SMILES-built solute, because the whole
  molecule is one residue whose name is not a known protein residue — observed here as
  *"5 amide candidate(s) could not be classified as ordinary or proline-like"*;
- the **ligand** route raised before it started: `solute_document` passed `ligand_sdf=None` as a
  literal, and there was no way to supply one — *"the ligand omega route needs the SDF … but none
  was supplied"*.

Underneath both: `build-top` wrote `solute.sdf` into its staging directory, used it to assign
charges and parameters, and **deleted it with the staging directory**. Bond orders are not
recoverable from a topology, and the ligand route needs them.

The SDF is now retained as `<system stem>.sdf` beside the System — where a later run already has
a path to it — and the preflight resolves the route from its presence. A peptide build writes
none, so existing peptide ladders are untouched and the absence is as informative as the presence.

Both fixes were validated with the fast suite (1393 passed) and the affected real CUDA/MPI lanes
(`test_cv_mpi_cuda_lanes.py`, `test_cv_mpi_cuda_rrest2.py`, `test_rrest2_cuda_smoke.py`:
**37 passed**).

---

## 5. The run

| setting | value |
|---|---|
| states | 6 at tau = 0, 0.1, 0.2, 0.3, 0.4, 0.5 |
| temperature | 300 K, one physical temperature for every rung |
| timestep / HMR | 2 fs / off |
| ensemble | fixed-volume, nonperiodic implicit solvent |
| equilibration | 50,000 steps = **100 ps per state, at that state's own Hamiltonian** |
| production | 500,000 steps = **1 ns per state** |
| exchange | every 5,000 steps = 10 ps, 100 attempts |
| CVs | every 500 steps = 1 ps → 1001 rows per state |
| trajectory / checkpoint | every 5,000 steps = 10 ps |
| seed | 20260907, Langevin friction 1.0 /ps |

**Hardware.** GPUs 0–4 were running unrelated jobs and were left alone. Six ranks over the four
free devices (5, 6, 7, 8) via the documented `local_rank` round-robin, one state per rank:

| rank | process-local device | physical GPU | state |
|---|---|---|---|
| 0 | 0 | 5 | 0 |
| 1 | 1 | 6 | 1 |
| 2 | 2 | 7 | 2 |
| 3 | 3 | 8 | 3 |
| 4 | 0 | 5 (shared) | 4 |
| 5 | 1 | 6 (shared) | 5 |

### Interrupted resume

Interrupted with SIGINT at **250 ps of production**, past the 200 ps required. The committed
checkpoint held:

```
committed step   : 126500
exchange index   : 24
committed cv_rows: 254
```

254 = 126500/500 + 1 — the checkpoint's progress and its CV record agree exactly, which is the
reconciliation added in `443fc73` holding on a real run.

Resumed with `--resume` to the **original** budget; it was not extended.

```
# interrupted at step 126500 of 500000; a checkpoint was committed and --resume will continue it
# continuation : bitwise: every rung's OpenMM context checkpoint was restored
# steps completed : 500000 of 500000
```

### Verification (`reports/run_verification.json`)

| check | result |
|---|---|
| grid | 1001 rows per state, steps 0…500000 every 500, **exactly once**, all six states |
| tau and state index per series | each series carries only its own state's tau and index |
| **no lost or duplicated observations** | the resumed run is **identical row-for-row and value-for-value** to an uninterrupted reference run of the same ladder — 6006 rows compared |
| independent recomputation | **720** reported torsion values recomputed from the saved coordinates with a dihedral formula written for the check; worst difference **2.18e−05°** (tolerance 0.5°) |
| pre-exchange convention | respected: only rows naming a trajectory frame were recomputed. An empty `trajectory_frame_index` marks a state the exchange moved, whose frame holds another walker's configuration |
| ring integrity | longest backbone bond **0.1673 nm** across sampled frames of all six states (limit 0.25 nm) |

The interrupted-and-resumed run also reproduced the reference's **final state→walker mapping
`[3, 0, 2, 1, 4, 5]`** and its acceptance statistics exactly.

### Observations, not targets

Neighbouring-pair acceptance, cumulative over all 100 exchanges:

| pair | accepted | fraction |
|---|---|---|
| 0 ↔ 1 | 2/50 | 0.040 |
| 1 ↔ 2 | 11/50 | 0.220 |
| 2 ↔ 3 | 1/50 | 0.020 |
| 3 ↔ 4 | 10/50 | 0.200 |
| 4 ↔ 5 | 8/50 | 0.160 |
| **overall** | **32/250** | **0.128** |

Acceptance is low and uneven, with a near-impassable 2↔3 gap. This is reported as measured; the
agreed ladder was **not** changed mid-experiment to improve it.

All fifteen backbone torsions were reported for every state. Every omega remained trans in every
state over the nanosecond (cis fraction 0.0000 at all six rungs), which is consistent with the
exclusion but — as stated above — is a diagnostic, not the proof.

---

## 6. Registration: blocked, and why

**Not performed.** `md-openmm data-register` resolves identity and storage independently, and
neither exists on this machine:

```
$ md-openmm data-register -idata <dataset> -project_name RGDfV-REST2 \
      -data_name RGDfV-Sage221-GBn2-REST2 -year 2026 --dry-run
data-register: no user configuration at ~/.config/md-tools/user.config (from XDG default).
Create one with:
    md-openmm data-register --init
```

`$MD_DATA` is unset, `~/.config/md-tools/user.config` does not exist, and no documented override
supplies a root. Every `data-register` path — including `--dry-run` — needs both. **No storage
root was invented and no user configuration was created**, since `--init` requires the user's own
name and `person_id` and would write a machine-wide file that is theirs to own.

**This is the deposition blocker to clear**, and it is a one-time action by the user:

```bash
md-openmm data-register --init          # asks for name, person_id and $MD_DATA
```

The dataset is assembled, self-contained and ready at
`../RGDfV-REST2/data/dataset/RGDfV-Sage221-GBn2-REST2/` (5.7 MB). Its expected destination is
`$MD_DATA/2026/RGDfV-REST2/RGDfV-Sage221-GBn2-REST2/`. It contains molecular identity and the
atom map, the SDF with bond orders, the serialised System and topology, the build log, every
configuration and seed, all six state trajectories and CV series with sidecars, the exchange
history, the checkpoint, the run records, and the validation reports — with **no reference to any
temporary build directory** (checked). The run records report `run_status: completed`.

No DOI, no public deposit and no licence are claimed; none exists.

---

## 7. Limitations, and the next scientific step

- **1 ns per state establishes integration behaviour, not converged populations.** No
  conformational preference, no free energy and no kinetic statement follows from this run.
- **Sage + GBn2 is not a validated combination for peptides.** GB-Neck2 was fit in the
  ff99SB/ff14SB lineage and has not been reparameterised against OpenFF valence and vdW
  parameters. The build labels it experimental and claims no Amber igb=8 parity.
- **mbondi3's Arg/Asp corrections were a no-op** (§2). The Hamiltonian is complete and internally
  consistent, but it is a hybrid whose implicit-solvent component is unvalidated for this
  chemistry — a limitation, not evidence of broken software.
- **Exchange acceptance is low and uneven** (§5), with a 2% pair. A production study would need a
  ladder chosen from measured acceptance rather than a uniform six-rung tau spacing.
- **Registration is unexecuted**, pending the user's one-time `--init` (§6).

**Next scientific step.** Before any longer study, re-tune the ladder: run a short acceptance
scan to place the rungs by measured overlap rather than uniform tau spacing, paying particular
attention to the 2↔3 gap. Separately, decide the implicit-solvent question deliberately — either
accept the exploratory Sage+GBn2 hybrid and say so in every result, or move the study to explicit
solvent where Sage is validated. Neither is a software question, and neither was decided here.

---

## Files

In the sibling project (`../RGDfV-REST2/`):

| path | what |
|---|---|
| `inputs/molecule.identity.json` | the validated chemistry |
| `inputs/atom_map.json` | residue/backbone table, five omega bonds, fifteen CVs |
| `config/` | build, REST2 and CV configurations, exactly as run |
| `reports/tolerances.md` | tolerances, recorded before any result was inspected |
| `reports/omega_exclusion.json` | the exclusion set and the term-by-term comparison |
| `reports/parameterization.json`, `reports/charge_table.json` | charges, valence, GB radii |
| `reports/run_verification.json` | grid, resume, recomputation, ring integrity |
| `scripts/` | every check above, runnable in order (see the project README) |
