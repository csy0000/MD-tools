# REST2/rREST2 correctness: exact scheduling, phase space, and one Hamiltonian

**Date** 2026-08-30
**Instruction** `docs/claudecode-instructions/20260830_rest2-rrest2-correctness-phase-space.md`
**Branches** `MD-templates: fix/openmm-md-rest2-rrest2-correctness` · `MD-project: dev6-openmm-md-rest2-rrest2-correctness`

**Heads verified before any work**

| repository | branch | SHA |
|---|---|---|
| MD-templates | `feat/openmm-md-rest2-rrest2` | `5c4edfba40a9724e5737c60ca3da9cee55dbf5b0` |
| MD-project | `dev5-openmm-md-rest2-rrest2` | `32aadcd4b7b74c096b39693dc85dee1dd79b6598` |

Both matched. Nothing was rebased and nothing ran against another revision.

The architecture is unchanged: one `openmm-md` executor, `--groupfile`/`-ng` coordinated
execution, a pluggable `--exchange-rule`, an optional `--reservoir`. No `openmm-rest2`, no
OpenMMTools in the runtime, no temperature REMD.

---

## What was wrong, and what it is now

### 1. `segment_ps` bundled four schedules into one number

It was the propagation quantum AND the coordinate-storage interval AND, in effect, the checkpoint
period. A solute stream could not be denser than the exchange period -- which is the stream a
REST2 run is usually for.

Exchange, whole-system output, solute output and checkpointing are now four independent schedules
(`replica_schedule.py`). Each is a physical time, each converts to an exact whole number of
integration steps, and one that does not is refused rather than rounded. Propagation advances to
the next event and never past it. Simultaneous events resolve in one fixed order --
`exchange, whole, solute, checkpoint` -- so a frame written at an exchange step is the
post-exchange state and the checkpoint describes everything already written.

`segment_ps` now raises an error naming the two fields that replaced it, because its old meaning
does not map onto any single new one.

Observed in the shipped runs: 10 exchanges, 2 whole frames, **20 solute frames** -- the solute
stream is ten times denser than the whole one and twice as dense as the exchange period.

### 2/3. The reservoir stores phase space, in files of its own

A DCD cannot store velocities, so a DCD was never a phase-space reservoir. The source is a NetCDF
stream of positions AND velocities AND boxes (`phase_space.py`), written by the fixed-tau run with
the completion marker committed last.

The storage was split to match: the solute stream is its own file (`rest2.solute.nc`) with its own
dimension and its own marker, and every stream carries absolute steps.

### 4/5. Stored velocity is the default; Maxwell is an explicit opt-in

`velocity_policy: stored` installs the recorded momentum unchanged, which is what the
probability-one Boltzmann acceptance assumes. `maxwell` redraws from a recorded seed and must be
asked for. **There is no silent fallback.** Velocities that are missing, the wrong shape,
non-finite, or *identically zero* are each a hard error when the source is opened.

The zero case was a real gap found while testing: identically zero passes every shape and
finiteness check, so a file converted from a coordinate-only source would have looked valid and
then been installed as zero momentum, which is not a sample from any Boltzmann distribution.

### 6. The MPI top-rung velocity loss

The refresh was installed by the rank owning the top rung, and rank 0's configuration list was
broadcast **afterwards**. Whenever the top rung was not rank 0's, the fresh velocities were
overwritten by stale ones on the way out: the refresh appeared to happen and silently did not. The
refresh is now applied by index on every rank from the same prepared sample.

### 7. Writable NetCDF is rank 0 only

On every path -- new, resume and extend. No non-root rank opens the analysis file in any mode; the
others adopt a broadcast decision.

### 8. Interruption, resume and extension are coordinated

A signal sets a flag and the ranks agree on it at an event boundary, so no rank stops
mid-propagation while another writes.

Two defects surfaced here and were fixed:

- **`--extend` shrank the budget.** It lengthened the *protocol file's* budget, not the one the
  run reached. A run extended once stores a larger budget than the protocol describes, so a second
  extension produced a total *below* the steps already run: the loop propagated nothing and the
  summary reported `220000 of 210000` while exiting successfully. It now extends the budget the
  run actually reached, and refuses a total that is not beyond the steps already run.
- **An interruption was reported as a failure.** The executor's promise check said a deliberate,
  cleanly checkpointed stop had "finished" and then failed it for not writing a completion
  manifest. Interruption now exits 130 -- neither success nor failure -- and the promise check
  applies only to a run that claims to have finished.

Before a continuation opens anything for writing, rank 0 **reads and validates** the stored output.
Nothing trusts the previous process to have exited cleanly.

### 9. One Hamiltonian, recomputed on both sides

The comparison is a SHA-256 over the canonically serialized OpenMM System plus a digest of the
enhanced-region selection (`hamiltonian_identity.py`). The current side is always **recomputed**;
two stored claims are never compared with each other. tau and temperature agreeing is necessary
and not sufficient -- ff14SB/TIP3P and ff19SB/OPC agree on both.

Two defects were found by making this check real:

- The driver fingerprinted the **unscaled reference** system while labelling the record with
  `tau_max`, so the digest and the tau described different objects. As written it rejected every
  correctly prepared source and would have *accepted* one recorded at tau = 0 -- the pairing that
  actually breaks the acceptance rule. It now compares the **top rung's scaled** system, and a
  reference is recorded with `tau: null`, a different claim from the rung at `tau: 0.0`.
- The emitted declaration named a sibling `cMD_tau<max>/` directory **that no invocation of the
  generator could produce**, so the default reservoir path could only ever dangle. An rREST2
  request now emits that fixed-tau stage itself, ahead of the ladder in `run.sh`. That makes the
  same-Hamiltonian contract true by construction -- one input, one force field, one solvation
  model, one equilibrated box -- and the identity is still recomputed and compared, because
  agreeing by construction is not the same as checking.

### 10. The exchange-rule boundary is preserved, and nothing new was implemented

`--exchange-rule` still takes a file; the rREST2 refresh is one. No new variant was added.

---

## Storage-integrity invariants the validator did not have

- **A completion marker cannot lead its own data.** The marker is committed after the rows it
  describes, so a crash leaves it *behind* them. A marker ahead of the rows the file holds means
  the file disagrees with itself, and a resume would read rows that were never written.
- **The stored exchange steps must be the scheduled ones.** A dropped attempt in the middle leaves
  strictly increasing steps and can still reach the budget, so no other check here would see it.

---

## Two defects found outside the ten

- **The `AIS` defaults had been deleted** along with a duplicated `rREST2` block on the parent
  branch, so `md_defaults(methods=(..., "AIS"))` returned a document with no `AIS` key and
  `sys-config --method cMD AIS` failed with a `KeyError`. Restored.
- **A stated platform never reached the replica runtime.** `platform: CPU` was accepted at setup,
  printed in the preset, and then dropped: every replica run silently took whatever OpenMM picked.
  It is now carried on the protocol and deliberately kept *out* of the scientific identity, so
  resuming on another machine is not refused for it. This one was found because it made the
  serial-vs-MPI comparison below impossible to interpret.

---

## Environment

| | |
|---|---|
| conda env | `$DATA_ROOT/software/md-stack/envs/openmm-rest2` |
| python | 3.12.14 |
| openmm | 8.6.0 (CUDA, mixed precision) |
| netCDF4 / numpy / mdtraj | 1.7.4 / 2.4.6 / 1.11.1 |
| mpi4py | 4.1.2 (OpenMPI) |
| snakemake | 9.26.1 (separate venv) |
| GPUs | 9 visible: device 0 RTX A5000, devices 1-8 RTX 3080; driver 580.173.02 |
| `MD_DATA` | `$DATA_ROOT/MD_DATA_amber` |

`openmmtools 0.26.0` is installed and **unused** by this runtime.

---

## Acceptance runs

All four on the required defaults. 6 states, tau `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`, 300 K, 2 fs,
20 ps production per replica, exchange every 2 ps, whole every 10 ps, solute every 1 ps.

| system | solvent | resolved force field | reservoir | serial | 6-rank MPI |
|---|---|---|---|---|---|
| `ALA_imp_REST2` | implicit | `leaprc.protein.ff14SB` + GBn2/mbondi3, no SASA | -- | completed | completed |
| `ALA_exp_REST2` | explicit NVT | `amber14-all.xml` + `amber14/tip3p.xml` (**TIP3P**) | -- | completed | completed |
| `ALA_imp_rREST2` | implicit | `leaprc.protein.ff14SB` + GBn2/mbondi3, no SASA | 12 stored-velocity samples | completed | completed |
| `ALA_exp_rREST2` | explicit NVT | `amber14-all.xml` + `amber14/tip3p.xml` (**TIP3P**) | 12 stored-velocity samples | completed | completed |

The explicit default is **ff14SB + TIP3P**. ff19SB/OPC is present as an additional compatibility
case (`examples/ALA/setup.rest2.explicit.opc.yaml`, system `ALA_exp_REST2_opc`) and does not
replace it.

Every stream, every run: `exchanges 10 (every 1000 steps = 2.0 ps)`, `whole frames 2 (every 5000
steps)`, `solute frames 20 (every 500 steps)`. Both rREST2 runs: `reservoir refreshes 5/5 at
state(s) [5]`, `velocity policy: stored`. The reservoirs were real -- prepared from a
co-generated fixed-tau cMD run whose phase-space file held 20 frames of finite, non-zero
velocities.

### Commands

```bash
export MD_DATA=$DATA_ROOT/MD_DATA_amber
md-openmm setup --config examples/ALA/setup.rest2.implicit.yaml  --output $MD_DATA/final --yes
cd $MD_DATA/final/ALA_imp_REST2 && ./run.sh              # serial
mpiexec -n 6 ./REST2/rest2.sh                            # one rank per state
```

---

## Runtime failure modes, exercised against a real run

`ALA_imp_LONG`: implicit ALA, 200 exchanges, 400 ps per replica.

| scenario | result |
|---|---|
| `SIGTERM` mid-run | `interrupted at step 54500 of 200000; a checkpoint was committed and --resume will continue it`; exit 130 |
| `--resume` | completed to 200 exchanges / 40 whole / 400 solute frames |
| `--extend 20` | 220 exchanges / 44 whole / 440 solute |
| `--extend 10` again | 230 exchanges / 46 whole / 460 solute (the budget bug above) |
| analysis NetCDF truncated to half | refused: `could not be opened as NetCDF ... A truncated or corrupt file fails here` |
| `last_exchange` set to 9999 | refused: `the marker is committed after the data it describes, so it can lag behind but never lead` |
| reservoir source at tau = 0.3 for a tau = 0.5 rung | refused on both the System digest and `tau: was 0.3, now 0.5` |

---

## MPI ownership: serial and 6-rank compared

| system | platform | serial vs 6-rank |
|---|---|---|
| `ALA_imp_rREST2` | CUDA | **byte-identical** summaries |
| `ALA_imp_REST2` | CUDA | **byte-identical** summaries |
| `ALA_exp_REST2` | CUDA | differ |
| `ALA_exp_REST2` (4 ps) | **CPU** | **byte-identical** summaries |

The explicit CUDA difference is **not** an MPI defect. Under MPI each rank binds its own device,
and this host is heterogeneous (one RTX A5000, eight RTX 3080). For an explicit-solvent system
with PME the block decomposition differs between architectures, forces differ in the last bits,
and a Langevin trajectory diverges from there. Re-running the same explicit ladder on the
deterministic **CPU** platform gave summaries that agree exactly between one process and six
ranks -- which is what isolates the ownership logic from the arithmetic. The implicit systems
agree even on CUDA because a 22-particle system is small enough to be device-independent.

Finding this required fixing the dropped-platform bug first: before that, asking for `CPU` silently
ran on CUDA and the comparison was meaningless.

**Limitation, stated plainly:** on this heterogeneous host, a CUDA run under MPI is not bitwise
comparable to the same run in one process. Reproducibility holds per device, not across the pool.

---

## Test evidence

| suite | result |
|---|---|
| MD-templates `tests/` | **629 passed**, 0 failed, 0 skipped (6m39s) |
| MD-project `tests/` | **355 passed**, 2 skipped, 0 failed |

`tests/test_rest2_phase_space_corrections.py` is new: 16 tests, each written so that it fails
against the previous implementation. `tests/test_own_replica_exchange.py` was updated to the
current contract rather than deleted -- the segment/stride vocabulary, the v1 reservoir
declaration and the single-frame-dimension storage are gone, and the tests that pinned them now
pin what replaced them.

The two MD-project skips are the ligand routes, which need a toolkit not installed here.

---

## Workflow integration

`snakemake -s workflow/rest2.smk --configfile workflow/rest2.config.yaml -n` builds an acyclic DAG:

```
generate  7   equilibrate  7   run_source_stage  2   run_stage  7   verify  7   total  31
```

`run_source_stage` fires exactly twice -- once per rREST2 system -- because each now generates its
reservoir source into its own project rather than depending on a separate system. The two
standalone fixed-tau systems remain: they are the AIS source route and are independently useful.

---

## Pin

`components.lock.yaml` pins MD-templates at
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0`, resolved from
`fix/openmm-md-rest2-rrest2-correctness`. Every `configs/systems/*.sys.config.yaml` carries the
same full 40-character SHA, and `tests/devtest/test_component_pin.py` holds it as an independent
second copy so the lock and the acceptance evidence must agree.

---

## Limitations

- Picosecond smoke sizes. These validate mechanism, **not** ladder quality and not convergence.
- The finite-reservoir approximation is real: 12 configurations are not the top rung's equilibrium
  distribution, and the assumption that the selected window represents it is an assumption.
- NVT only. NPT exchange is refused, not approximated.
- On a heterogeneous GPU host, MPI and serial CUDA runs are not bitwise comparable (above).
- Implicit-solvent **ligand** REST2 remains scientifically unvalidated.
- ff19SB/OPC is exercised as a compatibility case only; it has not been validated as a production
  route.
