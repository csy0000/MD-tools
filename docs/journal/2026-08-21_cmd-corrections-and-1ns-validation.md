# 2026-08-21 — Standalone cMD, committed segments, and two 1 ns alanine validations

> **Superseded by [`2026-08-21_cmd-review-fixes-and-opc-default.md`](2026-08-21_cmd-review-fixes-and-opc-default.md).**
> Seven findings were raised against this round. The **implicit** results and hashes below no longer
> describe the code on `dev`: the positional restraint used `periodicdistance` in implicit solvent,
> so it read box vectors that System should not have had, and correcting it changed that
> Hamiltonian. The explicit results stand, but TIP3P-FB is no longer the active default — OPC is,
> via the `explicit-*-v2` profiles.

Instruction: `claudecode-instructions/20260821_cmd-corrections-and-1ns-validation.md`
Starting commit: `8fa39cb` (instruction) on `dev`, itself a descendant of `8c0c85c`

**Both 1 ns validations completed on CUDA.** Explicit ff19SB/OPC NPT and implicit
ff19SB/GBn2/mbondi3 NVT, each as two committed 500 ps segments in one run directory.

## Review findings and their resolutions

| # | finding | resolution |
|---|---|---|
| 1 | `production.method = "md"` refused: "Conventional-MD-only projects are not yet generated" | four method-and-mode-specific stage graphs; MD-only ends at `cMD_1` |
| 2 | cMD single-shot: reporters without append, no committed boundary | `cmd_segments.py` on the existing `runstate` primitives |
| 3 | cMD segment length taken from the untyped `conventional_md.duration` | MD-only uses canonical `duration_per_segment`; stating both is refused |
| 4 | `minimize_max_iterations = 0` in every profile | `1000` |
| 5 | no NVT duration in any profile | `10 ps` (and 10 ps restrained/free NPT where the explicit graph has them) |
| 6 | cMD default 100 ps, REST2 default 2 ns / 200 exchanges | `5 ns` and `5 ns / 1000` |
| 7 | reporting 10 ps / 2 ps | `100 ps` all-atom, `10 ps` selected |
| 8 | explicit ladders 10/10 | peptide 6, ligand 10 |
| 9 | `PROTOCOL_SCHEMA_VERSION` 5 but every profile embedded schema 4 with v5 fields | all profiles emit 5; the field is validated, not carried |
| 10 | the migrator emitted schema 1, which this build then refuses | it emits the current version |
| 11 | implicit runs recorded a numerical `box_volume_nm3` and requested volume/density columns | periodicity is read from the System; implicit records null with a reason |
| 12 | `enforcePeriodicBox=True` on implicit all-atom trajectories | wrapping follows periodicity |
| 13 | HCT/OBC/GBn and five radius sets publicly accepted | public acceptance is GBn2 + mbondi3 only |
| 14 | `--devices` honoured only by REST2 | honoured by every stage |
| 15 | CI did not run on `dev` | `dev` added to both workflow push triggers |

### A second user decision, mid-task: implicit equilibration is not "NVT"

The user confirmed the replica counts and added: *"implicit doesn't need nvt or npt, just positional
restraints is fine"*, then clarified: *"implicit solvent also needs to undergo the same equilibration
time (20ps in total) but no nvt or npt because it doesn't have simulation box"*.

The first reading dropped the equilibration stage entirely, which was wrong. The clarification is
narrower and better: implicit solvent still equilibrates under restraints for the same kind of time;
what it cannot have is an *ensemble label*. "NVT" fixes a volume, and there is no volume here.

So the implicit graphs are:

```
implicit md    : min -> eq -> cMD_1
implicit rest2 : min -> eq -> cMD_1 -> REST2_1
```

`eq` runs 20 ps of restrained constant-temperature dynamics with no barostat, configured by
`protocol.equilibration.restrained` -- a field named for what it is rather than for an ensemble it
cannot have. A stated `nvt`, `npt` or `npt_free` under implicit solvent is refused, and a *missing*
`restrained` is refused too: the solute still has to settle before production, so its absence is an
omission rather than a choice.

Two things this exposed:

* `EXECUTION_STATUS` did not know the new stage name, so `eq.sh` failed with "unknown stage 'eq'".
  That is the second time a new stage name has needed registering there; it is a list the stage
  runner treats as authoritative, which is the right design and a step easy to forget.
* Velocity creation moved with the graph. `eq` is now the first stage that integrates, so it is
  where velocities are initialised, and `cMD_1` inherits them. Measured on the final run:
  `min` "not required", `eq` "initialized", `cMD_1` "inherited" then "restored from the committed
  generation" on its second segment.

The implicit 1 ns validation was re-run on this final graph and reproduces every total.

### The replica-count conflict, resolved toward the user

The instruction says the implicit 4/6 counts are "incorrect for this repository instruction history"
and should become 6/10 — **"unless the user supplies a new explicit scientific decision."** The user
supplied exactly that in the session that produced them: *"for implicit solvent alanine only needs 4
replicas and rgd needs 6 replicas."*

So implicit stays **4 (peptide) / 6 (ligand)** and explicit becomes **6 / 10**. The comments that
described the implicit counts as "by instruction" were misleading and now cite the decision itself,
and a test records which authority set which number so this does not have to be reconstructed from
commit archaeology later. This deviation is deliberate and flagged rather than silently resolved in
either direction.

## The cMD persistence design

At each committed segment boundary, in this order:

1. reporters are closed, so the files on disk end exactly at the boundary;
2. the binary checkpoint and the serialized State are written through temporary files;
3. the generation is committed atomically, naming its members and the per-stream watermarks.

A crash anywhere before step 3 leaves an uncommitted tail, which the next invocation truncates before
opening anything for append. Frame counts therefore come from the **commit record**, not from
measuring the files: a file length is a fact about the last crash, not about the last committed
boundary.

Watermarks are **per stream**. An all-atom trajectory at 100 ps and a solute trajectory at 10 ps
reach a 500 ps boundary with 5 and 50 frames; one number would corrupt one of them.

The continuity contract covers particle and constraint counts, periodicity, ensemble, barostat
pressure, integrator kind/timestep/temperature/friction, restraint stiffness, steps per segment,
reporting intervals, selection fingerprint, and the predecessor identity. It deliberately **excludes**
the segment count: asking for more segments is the one change that is always safe, which is why it
lives in Bash and never in the hash.

Continuation is decided before the Context exists, so an incompatible continuation is refused while
the previous segment's outputs are still exactly as it left them. A test asserts the trajectory bytes
are unchanged after the refusal.

## Hardware

Nine GPUs, all idle, no compute processes at selection time.

| | explicit | implicit |
|---|---|---|
| device (nvidia-smi index) | 0 | 1 |
| model | NVIDIA RTX A5000 | NVIDIA GeForce RTX 3080 |
| UUID | `GPU-7a14ba65-b0a3-66bd-536d-881e08b55da1` | `GPU-96ce533d-9d42-acf6-8384-5e27150e9a85` |
| platform / precision | CUDA / mixed | CUDA / mixed |

Driver 580.173.02, OpenMM 8.5.2, ParmEd 4.3.1, Python 3.12.13.

**`CUDA_DEVICE_ORDER=PCI_BUS_ID` is load-bearing here.** By default CUDA orders devices
`FASTEST_FIRST`, so OpenMM's device 0 was the RTX 3080 while `nvidia-smi` index 0 is the A5000.
Without pinning the ordering, a recorded device identity is approximately right and actually wrong.

The two runs executed concurrently on different idle GPUs, which the instruction permits.

## Commands

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# explicit
python MD_system_gen.py -i ace_ala_nme.pdb -o explicit_bundle \
    --config test/ala/cMD/explicit/system_config.json
python MD_input_gen.py --system explicit_bundle/system_manifest.json -o explicit_run \
    --config test/ala/cMD/explicit/md_config.json
cd explicit_run && MD_DEVICES=0 CMD_NUMBER_OF_SEGMENTS=2 ./run_all.sh

# implicit — same, with the implicit configs and MD_DEVICES=1
```

## Resolved protocols

| | explicit | implicit |
|---|---|---|
| stages | min, eq_nvt, eq_npt_1, eq_npt_2, cMD_1 | min, eq, cMD_1 |
| minimisation | 1000 iterations, solute restrained 1 kcal/mol/Å² | same |
| equilibration | 10 ps NVT, 10 ps restrained NPT, 10 ps free NPT | 20 ps restrained `eq`, no ensemble label |
| timestep | 4 fs with HMR to 3.024 amu | 2 fs, no HMR |
| ensemble | NPT, 300 K, 1 bar | NVT, 300 K |
| segment | 500 ps = 125,000 steps | 500 ps = 250,000 steps |
| all-atom | every 25,000 steps (100 ps) | every 50,000 steps (100 ps) |
| selected | every 2,500 steps (10 ps) | every 5,000 steps (10 ps) |

`duration_per_segment: "500 ps"` is a worked-test override. The named profile default remains 5 ns.

## Results

| | explicit | implicit | expected |
|---|---|---|---|
| committed generations | 2 | 2 | 2 |
| absolute step | 250,000 | 500,000 | 250,000 / 500,000 |
| absolute time | 1000.0 ps | 1000.0 ps | 1000.0 ps |
| restart source | checkpoint | checkpoint | checkpoint preferred |
| barostat | MonteCarloBarostat | none | NPT / NVT |
| periodic | true | **false** | true / false |
| `box_volume_nm3` | 18.364 | **null** | value / null |
| all-atom frames | 10 (2438 atoms) | 10 (22 atoms) | 10 |
| selected frames | 100 (22 atoms) | 100 (22 atoms) | 100 |
| log header / rows | 1 / 10, monotonic | 1 / 10, monotonic | 1 header |
| NaN or Inf | none | none | none |

Descriptive ranges over the production log — recorded, not asserted, because these are stochastic
observables:

```
explicit   T 292.6 .. 309.9 K    U -33205.4 .. -32364.7 kJ/mol   V 18.0 .. 18.8 nm^3
implicit   T 225.0 .. 289.8 K    U   -138.2 ..   -103.9 kJ/mol   V n/a (nonperiodic)
```

The implicit temperature range is wider than the explicit one because the system is 22 atoms rather
than 2438; the instantaneous temperature of a small system fluctuates strongly, and that is expected
rather than a defect. The implicit log has no Volume or Density column at all -- confirmed from the
header -- which is the point of the mode-aware reporting.

Implicit figures and hashes are from the re-run on the final graph (min -> eq -> cMD_1). Two earlier
runs, on the original and the intermediate graphs, reproduced the same step, time and frame totals --
the graph change moved the trajectory, as it must, but not the accounting.

Wall time for production only: explicit 38 s, implicit 34 s, giving roughly 2,274 and 2,541 ns/day.

Output hashes (sha256, first 16 hex):

```
explicit  cMD_1_all_atoms.dcd       e446485749bb2229
explicit  cMD_1_selected_atoms.dcd  11a8a34e6ea357bb
implicit  cMD_1_all_atoms.dcd       618316bc98d7babe
implicit  cMD_1_selected_atoms.dcd  3f149df55867bd8c
```

Both restart forms reload in a fresh Context at t = 1000.0 ps with identical energy
(explicit −32659.7, implicit −129.5 kJ/mol).

**A correction worth recording.** A first reload attempt reported both forms failing for both runs.
That was the test's fault: it rebuilt a bare System on the **CPU** platform, while an OpenMM
checkpoint is specific to the platform *and* to the exact System — barostat and restraint forces
included. Reloading through the stage's own construction succeeds. A checkpoint's platform
specificity is exactly why the portable State exists.

## A defect the corrected defaults exposed

The first launch stopped at `eq_nvt`:

```
ValueError: all-atom trajectory interval is 25000 steps but this stage runs 2500,
so no frame would ever be written.
```

The reporting cadence is a **production** cadence, and a 10 ps equilibration stage cannot produce a
frame at a 100 ps interval. The reporter guard caught it correctly, but the fault was the generator
*declaring* a stream that could not be written — the same defect this repository already fixed in
the other direction. Reporting is now declared per stage, with omissions and their reasons recorded,
and the state log falls back to a cadence that fits so equilibration stays observable.

## Tests

```
tests/test_cmd_segments.py                      23 passed
  same file against 8c0c85c in a worktree       16 failed, 7 errors, 0 passed
complete non-slow suite                        755 passed, 99 deselected
complete suite including slow markers          (recorded below)
wheel + sdist build                            ok; 9 profiles, cmd_segments packaged,
                                               6 cMD example files in the sdist
installed wheel outside the checkout           md-system-gen + md-input-gen generated an
                                               MD-only project; two cMD segments ran with
                                               checkpoint continuation
relocated bundle                               regenerates and validates
goldens                                        regenerated; moved values are exactly the
                                               instructed default corrections
```

Golden movement, stated so no hash is hand-waved: `duration_per_segment` 100 ps → 5 ns for the cMD
profiles and 2 ns → 5 ns for the REST2 profiles, and the explicit peptide tau ladder going from ten
rungs to six. Nothing else moved.

## CI

`.github/workflows/fast.yml` and `integration-cpu.yml` triggered on `openmm` and `main` only. `dev`
is now in both push trigger lists. The `openmm` entry is stale — that branch was renamed to `main` in
earlier work — and is left in place rather than removed as an unrelated change.

The remote run result is **not** reported here: this environment has no authenticated GitHub API
access (`gh` is not installed), so the workflow outcome on `dev` could not be read. The trigger
change is verifiable in the workflow files; the run itself is not verified.

## Limitations

- **A 1 ns run is engineering validation, not scientific validation.** It shows the chain executes,
  commits, continues and accounts for every frame and step. It says nothing about convergence,
  ensemble quality, or whether 1 ns is adequate for any question about alanine dipeptide.
- The implicit temperature range reflects a 22-atom system; no thermodynamic claim is made from it.
- Public implicit solvent is GBn2/mbondi3 only. Other pairs are refused as unvalidated.
- Implicit REST2 remains full-solute only.
- No forced State-fallback test ran against the 1 ns runs themselves; the fallback path is exercised
  by the contract in `runstate.load_restart` and announced when taken. A dedicated tiny forced-
  fallback test is listed as deferred below.
- The explicit HMR/4 fs decision is unchanged, as instructed.

## Deferred

- A tiny forced-State-fallback continuation test and a process-interruption test around the atomic
  commit. The truncation and refusal paths are covered; the deliberate corruption path is not.
- Reading the actual CI run on `dev`.
