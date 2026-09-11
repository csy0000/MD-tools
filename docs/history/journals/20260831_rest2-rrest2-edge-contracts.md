# Closing the REST2/rREST2 edge contracts

**Date** 2026-08-31
**Instruction** `docs/claudecode-instructions/20260831_rest2-rrest2-edge-contracts.md`
**Branches** `MD-templates: fix/openmm-md-rest2-rrest2-edge-contracts` · `MD-project: dev7-openmm-md-rest2-rrest2-edge-contracts`

## SHAs

| repository | branch | starting SHA | final SHA |
|---|---|---|---|
| MD-templates | `fix/openmm-md-rest2-rrest2-correctness` → `fix/openmm-md-rest2-rrest2-edge-contracts` | `6cb693f748a2b63e5a0a49a9500ae295433fc2e0` | `b60184f63adc5f745834f1cfcabf06a8b2c942d0` |
| MD-project | `dev6-openmm-md-rest2-rrest2-correctness` → `dev7-openmm-md-rest2-rrest2-edge-contracts` | `fc89212ea05298013a1f71e13467f1f5c111d925` | see the final commit of this branch |

Both starting heads were fetched and verified before anything changed. Nothing was rebased and no
history was rewritten.

The architecture is unchanged: one `openmm-md` executor, `--groupfile`/`-ng` coordinated
execution, a pluggable `--exchange-rule`, an optional `--reservoir`, Hamiltonian REST2 scaling at
one physical temperature, independent event schedules, rank-0 writable storage, exact System
fingerprints, and the current ALA defaults. No `openmm-rest2`, no OpenMMTools execution, no
temperature REMD, no kinetic reservoir, no non-Boltzmann acceptance, no NPT REST2.

---

## 1. `velocity_policy` is one contract, end to end

**What was wrong.** `PreparedReservoir.open()` parsed `velocity_policy` and then called
`_prepare()` without it. The helper's own default `"stored"` decided how the source was validated,
so a declaration that explicitly asked for `maxwell` was prepared under `stored` rules and refused
for velocities it never intended to use. A helper default overrode the declaration.

**What it is now.** The parsed policy is passed through preparation, validation, materialisation,
the prepared manifest and the completion record. `_prepare` takes it as a **required** argument, so
no future caller can drop it back into a default.

The narrow contract:

- Both policies read the **same** source, the versioned phase-space NetCDF. It carries positions,
  box, absolute source step and time, the Hamiltonian identity, the atom identity and the
  completion marker in one auditable file. The policy decides what happens to the recorded
  momenta; it does not widen the source format.
- `stored` remains the default: finite, correctly shaped, **nonzero** recorded velocities,
  installed unchanged.
- `maxwell` uses the source positions and box and **deliberately ignores** the recorded velocity
  values, redrawing at the one common temperature from a seed recorded per refresh.
- No automatic fallback exists. The only way to redraw is to ask.

**A misleading claim, corrected.** The module docstring described `maxwell` as "an EXPLICIT opt-in
for a coordinate-only source". That is wrong: a DCD carries no Hamiltonian identity, no absolute
source step and no completion marker, so it satisfies this contract under **neither** policy. A
coordinate-only source would need its own versioned, separately validated contract, and none is
implemented. The docstring, the emitted declaration and the MD-project examples now say so.

**Provenance.** The prepared manifest records the policy it was materialised under — not merely
the ones the format supports — and reusing a prepared file across policies is refused with the
reason. Storage gained a `reservoir_velocity_seed` column so any single Maxwell draw is
reproducible from the file alone rather than by replaying the rule RNG from the beginning.

Observed on the Maxwell smoke: per-exchange seeds
`[-1, 876448373, -1, 1740819207, -1, 400947573, -1, 1955835690, -1, 1850551831]` — a seed exactly
where a refresh happened, `-1` elsewhere — and the completion record's
`lifetime_statistics.reservoir.velocity_seeds_used` lists all five. Both `stored` runs recorded
**no** seed at all, which is the honest record of a policy that draws nothing.

---

## 2. Sampling with replacement is refused, not written and then rejected

**The contradiction.** The declaration exposed `allow_sampling_with_replacement`, so selection
could draw one source frame twice. Materialisation sorted the indices and
`PhaseSpaceReader.validate()` — correctly requiring strictly increasing absolute steps — rejected
the file this code had just finished writing. The repository refused its own output, after writing
it.

**Why the invariant stays.** A repeated draw gives one empirical configuration extra statistical
weight in a reservoir meant to be a uniform sample of the top rung's distribution, and it makes
`frames` overstate the effective reservoir size. The finite-reservoir approximation is already the
weakest assumption here. Silently de-duplicating would be worse: it changes both the requested
count and the weights and records neither.

**What it is now.** Refused at the rREST2 boundary before a directory exists, and
`allow_replacement=False` is a literal at the selection call rather than a value read back out of
the declaration. Asking for more frames than the window holds gets a reservoir-specific message
with the requested count, the eligible distinct count, the window and the source spacing — the
shared selector's own message offers replacement as a way out, which is right for AIS and wrong
here.

AIS is untouched: it keeps its separately documented replacement contract and still reaches that
path. The inclusive-window rule was extracted as `eligible_frames()` and is called by both, so the
count an error message reports cannot drift from the count the selector used.

Observed: all three prepared reservoirs hold 12 frames with unique, strictly increasing source
frame indices and steps, `allow_replacement: false`.

---

## 3. Continuation flags are grouped-only

**What was wrong.** The parser accepted `--resume` and `--extend` in both modes, but only the
grouped replica runtime consumes them. The conventional path calls `load_protocol(...).run(files)`,
and generated cMD scripts reset time and step state and create their reporters as a fresh run — so
`--resume` promised a continuation nothing implements and would have written over the DCD,
checkpoint and phase-space outputs of the run the user meant to continue.

**What it is now.** No partial conventional-MD restart was attempted. The refusal lives in the
pure validation pass, which returns before a directory is created, a report is opened, or the
protocol is imported. The message says what is not implemented and offers a fresh stage in a clean
location; it does not offer to delete anything. `--force` remains a deliberate fresh-run
replacement and the CLI help now says it is not continuation.

Exercised against a real co-generated fixed-`tau` source stage with existing outputs:

```
$ ./cMD_tau0p5/cmd.sh --resume     → exit 2
$ ./cMD_tau0p5/cmd.sh --extend 5   → exit 2
```

SHA-256 of `cmd.chk`, `cmd.dcd`, `cmd.out`, `cmd.phase_space.nc` and `cmd.state.xml` before and
after both refusals: **byte-identical**. The `.out` report was not even opened.

Grouped continuation is unchanged: `--resume`, then `--extend 4`, then `--extend 3` took the
Maxwell ladder from 10 → 14 → 17 exchanges, with 5 whole and 34 solute frames, all three streams in
proportion, and 8 Maxwell draws recorded.

---

## Two defects the verification pass turned up

Neither is one of the three contracts. Both were pre-existing and were found by running
`--verify-only` against a real twice-extended run, which nothing had done before.

- **A descriptive manifest field was checked as a filename.** `storage` mixes filenames with
  description: `schema` names a format, `authoritative` names another key of the same block, and
  `coordinate_indexing` says how coordinates are indexed. Only the first two were skipped, so
  validation failed with `storage.coordinate_indexing=walker does not exist beside the manifest`.
- **The identity budget was compared for equality.** The scientific identity records what was
  *originally* requested; `--extend` legitimately runs past it. Comparing stored rows against that
  count rejected every extended run — `the storage holds 17 exchange row(s) but the run promised
  10 attempts`. The identity count is now a floor, and the authoritative budget is the schedule the
  run actually finished under.

After both fixes, `--verify-only` returns `VALID` on all seven generated projects, including the
twice-extended one.

---

## Environment

| | |
|---|---|
| conda env | `$DATA_ROOT/software/md-stack/envs/openmm-rest2` |
| python | 3.12.14 |
| openmm | 8.6.0 |
| netCDF4 / numpy / mdtraj | 1.7.4 / 2.4.6 / 1.11.1 |
| mpi4py | 4.1.2 (OpenMPI) |
| snakemake | 9.26.1 (separate venv, not the conda env) |
| GPUs | 9 visible: device 0 RTX A5000, devices 1–8 RTX 3080; driver 580.173.02 |
| `MD_DATA` | `$DATA_ROOT/MD_DATA_amber` |

`openmmtools 0.26.0` is installed and **unused** by this runtime.

**Linting, honestly.** The repository defines no lint or format configuration, and `ruff`,
`flake8` and `black` are **not installed** in the project environment. `ruff 0.16.5` was installed
into a throwaway venv purely as an external check. Reformatting was **not** applied: the codebase
was never formatted with it, and `ruff format` would have rewritten 8 files into an unrelated diff.
The meaningful measurement is the delta — on the files this branch touches, ruff reports **47
findings before and 47 after, with none added and none removed**. The new test file triggers only
`I001` and `RUF100`, the same two rules the repository's existing test files trigger under ruff's
defaults.

---

## Test suites

| suite | result |
|---|---|
| MD-templates `tests/` | **653 passed**, 0 failed, 0 skipped (6m42s) |
| MD-project `tests/` | **355 passed**, 0 failed, 0 skipped |

The MD-project suite skipped 2 before the MD-templates branch was pushed: `test_component_pin.py`
checks that the remote actually carries the requested ref, which it could not until the push. They
were re-run afterwards and **pass** — the 355 above is that final run. No check was left
unexercised.

`tests/test_rest2_edge_contracts.py` is new: 24 tests, **8 of which fail on `6cb693f7`**, the head
this branch starts from — the two policy-plumbing tests, the two replacement tests, and the four
non-grouped continuation tests. The rest pin behaviour that was already correct, so that it stays
correct.

---

## Smoke matrix

Picosecond smokes. **These validate mechanism only. They are not convergence evidence, and the
Maxwell case is not a recommended scientific default.**

All six states, `tau = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`, one thermostat at 300 K, 2 fs, 20 ps per
replica, exchange every 2 ps, whole every 10 ps, solute every 1 ps.

| system | solvent | force field | policy | serial | 6-rank MPI | verify |
|---|---|---|---|---|---|---|
| `ALA_imp_REST2` | implicit | ff14SB + GBn2/mbondi3, no SASA | — | completed | completed | VALID |
| `ALA_exp_REST2` | explicit NVT | ff14SB + TIP3P | — | completed | completed | VALID |
| `ALA_imp_rREST2` | implicit | ff14SB + GBn2/mbondi3, no SASA | `stored` | completed | completed | VALID |
| `ALA_exp_rREST2` | explicit NVT | ff14SB + TIP3P | `stored` | completed | completed | VALID |
| `ALA_imp_rREST2_maxwell` | implicit | ff14SB + GBn2/mbondi3, no SASA | **`maxwell`** | completed | completed | VALID |

Every run: `exchanges 10 (every 1000 steps = 2.0 ps)`, `whole frames 2 (every 5000 steps)`,
`solute frames 20 (every 500 steps)`. Every reservoir run: `5/5 refreshes at state(s) [5]` — the
top rung only.

ff19SB/OPC remains an optional compatibility case (`ALA_exp_REST2_opc`), not the default.

### Serial versus six-rank MPI

| system | platform | result |
|---|---|---|
| `ALA_imp_REST2` | CUDA | identical |
| `ALA_imp_rREST2` (`stored`) | CUDA | identical |
| `ALA_imp_rREST2_maxwell` | CUDA | identical |
| `ALA_exp_REST2` | CUDA | differs |
| `ALA_exp_rREST2` | CUDA | differs |
| `ALA_exp_CPU` (explicit REST2, 4 ps) | **CPU** | **identical** |
| `ALA_imp_maxwell_CPU` (rREST2 `maxwell`, 8 ps) | **CPU** | **identical** |

The explicit CUDA differences are **not** an ownership defect. Under MPI each rank binds its own
device and this host is heterogeneous — one RTX A5000 and eight RTX 3080s — so for an
explicit-solvent system with PME the block decomposition differs between architectures, forces
differ in the last bits, and a Langevin trajectory diverges from there. On the **deterministic CPU
platform** the same explicit ladder agrees exactly between one process and six ranks, which is what
isolates the ownership logic from the arithmetic. The implicit systems agree even on CUDA because a
22-particle system is small enough to be device-independent.

That the **Maxwell** case is identical serial and six-rank, on both CUDA and CPU, is the direct
evidence that only the owning rank draws and the resulting array is shared: every rank installs the
same momenta from the same seed.

---

## Workflow

`snakemake -s workflow/rest2.smk --configfile workflow/rest2.config.yaml -n` builds an acyclic DAG:

```
generate 8   equilibrate 8   run_source_stage 3   run_stage 8   verify 8   replica_all 1   total 36
```

`run_source_stage` fires three times, once per rREST2 system, now including the Maxwell case. The
workflow passes **no** continuation flag anywhere: every stage it launches is a fresh run.

Generated projects carry no absolute machine path in anything executable — `paths.sh`, launchers,
protocols and group files all resolve from `$MD_DATA`. Absolute paths appear only in
`input/provenance.yaml`, which exists to record exactly that.

---

## Pin

`components.yaml` requests `fix/openmm-md-rest2-rrest2-edge-contracts`;
`components.lock.yaml` pins `b60184f63adc5f745834f1cfcabf06a8b2c942d0`. Every
`configs/systems/*.sys.config.yaml` carries the same full 40-character SHA, and
`tests/devtest/test_component_pin.py` holds it as an independent second copy so the lock and the
acceptance evidence must agree rather than merely restate each other.

---

## Limitations

- **Picosecond smokes validate mechanism, not convergence.** Nothing here is evidence about ladder
  quality, mixing efficiency or sampling adequacy.
- **`maxwell` is a mechanism check, not a recommendation.** `stored` is the default because the
  probability-one acceptance is stated for a phase-space sample installed unchanged; replacing its
  momenta is a different operation whose consequences are not characterised here.
- **The finite-reservoir approximation is real.** Twelve configurations are not the top rung's
  equilibrium distribution, and the assumption that the selected window represents it is an
  assumption.
- **Conventional cMD continuation is still not implemented.** This milestone refuses it clearly; it
  does not provide it.
- **Heterogeneous-GPU MPI runs are not bitwise comparable to serial ones.** Reproducibility holds
  per device, not across this pool. The CPU comparison is what establishes ownership correctness.
- **No linter is configured or installed in the project environment.** The ruff measurement above
  is an external check with a delta of zero, not a project standard being met.
- **NVT only.** NPT REST2 exchange remains refused, not approximated.
- Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
