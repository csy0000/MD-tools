# AIS — annealed importance sampling

Non-equilibrium switching between two end states. Many independent paths start from configurations
drawn from an equilibrium ensemble of **V0**, move the Hamiltonian to **V1** while the coordinates
propagate, and accumulate the **work** done along the way [@neal2001ais]. The work distribution is
what the method produces; the Jarzynski equality relates its exponential average to a free-energy
difference [@jarzynski1997equality].

AIS is not a sampling protocol that produces one long trajectory. It produces *N* short ones and a
number for each.

```text
V(λ, x) = (1 − λ)·V0(x) + λ·V1(x),        λ: 0 → 1, linear
```

V0 is `-s`/`-p`, the state the source ensemble was sampled from. V1 is `-s2`/`-p2`. This is Amber's
linear (no-softcore) mixing of two end states [Amber26 §27.1, Eq. 27.3], driven along a path as in
its Jarzynski protocol [§27.8]. Until 0.5.4 AIS instead switched one System along a REST2 τ; that
implementation is retired, and a REST2 switch is now expressed as a pair of files.


## The two example files beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm build-md` |
| [`example.in`](example.in) | what `md-run` then reads — one switching campaign |

`build-md` generates the `.in` from the `.config`; you do not normally write one by
hand. It is shipped here because it is the file the run actually reads, and because
every setting in it carries the schema's own description as a comment — so the meaning
of a key can be looked up where it is used rather than in the source.

`resolved.config`, written beside the `.in` at run time, stays AUTHORITATIVE: the `.in`
is resolved into it, and that resolved document is what the run reads.

Both are checked by `tests/test_method_example_inputs.py`, which regenerates the `.in`
from the `.config` and fails if they have drifted — an example that no longer matches
the engine is worse than none.

The end states are **files, not configuration**, exactly as `-s` and `-p` are for every other
protocol.

## The two end states

V0 and V1 must hold the same particles in the same order, with the same masses, constraints,
virtual sites, force layout and long-range treatment (nonbonded method, cutoff, switching,
dispersion correction, PME parameters, exception pairs). They differ in **parameters only**. The
preflight refuses a pair that differs in anything else and names every difference at once, and it
refuses a barostat in either. Atom mapping, softcore potentials and dummy atoms are not
implemented; a pair that would need them cannot be expressed, which is the point of refusing.

For a REST2 switch from a scaled state to the physical Hamiltonian, V0 is the scaled System the hot
run integrated and V1 is the unmodified `build/built.xml`. A reverse switch exchanges the two files;
the schedule never runs downhill, so the Jarzynski average is always over the V0 ensemble.

## The work convention

For each parameter update *j*:

```
ΔW_j = V(λ_{j+1}, x_j) − V(λ_j, x_j) = (λ_{j+1} − λ_j)·(V1 − V0)(x_j)
```

**The parameters move at frozen coordinates, and only then does the configuration propagate.** That
ordering is the definition, not an implementation detail: evaluating the two energies at different
coordinates would compute a different quantity that happens to have the same units.

The second equality holds exactly because V is linear in λ, and it is how the work is measured —
one evaluation of dV/dλ per switch. It is also exactly Amber's Eq. 27.30, `(∂U/∂λ)·Δλ`. The finite
difference stays the definition, so the schedule below changes which λ values a path visits and
never what `work` means.

## Schedule

`ais.lambda_schedule` says how λ follows the switching progress `t = update / number_of_updates`:

| schedule | λ(t) |
|---|---|
| `linear` (default) | `t` |
| `tau-linear` | `[(1 − τ₀ + τ₀t)² − (1 − τ₀)²] / [1 − (1 − τ₀)²]` |

`tau-linear` is for V0 a saved REST2 state at τ₀ and V1 its unscaled source. The mixture then scales
the solute–solute terms (nonbonded, 1-4 and eligible torsions) by `(1 − λ)(1 − τ₀)² + λ = (1 − τ)²`
along `τ = τ₀(1 − t)` — what a switch linear in τ would do to them. **It does not make the
solute–environment scaling `(1 − τ)`**: one mixing coefficient cannot follow two different powers
of `1 − τ`. In vacuum, where every atom is solute, the mixture is exactly the REST2 System at that τ,
and a test holds it to that.

`ais.lambda_schedule_tau0` is τ₀. When `ais_source.generate` is true it defaults to `dynamics.tau`
and must equal it, and build-md writes the number into `resolved.config`; otherwise it must be
stated. The run reads λ from that number, never from a REST2 record, but it CHECKS the claim before
any output: `-s` must be a saved state at τ₀ and `-s2` the System its `scaler.yaml` names as the
source, or tau-linear is refused. The schedule, τ₀ and a sha256 of the λ table are part of
`AIS_run.json` and the checkpoint fingerprint, so a directory cannot be continued under another
schedule.

**Observation 0 precedes all work.** The first recorded observation is the source configuration
under V0, before any parameter change and before any propagation, so its cumulative work is exactly
zero. A path whose first row is not zero work at λ = 0 is refused.

## Ensemble

**Fixed volume.** Switching is at constant volume, and a barostat in either end state is refused. A
volume move during a switching path would do work that the convention above does not account for.

## What MD-tools implements

* an exact integer-step schedule: `switching_steps`, `parameter_update_interval_steps` and
  `observation_interval_steps` must divide consistently, and a schedule that would have to be
  rounded is refused with the arithmetic that would fix it;
* both endpoints included in the schedule;
* **one evaluation per switch.** Forces identical in V0 and V1 are added once; each differing force
  pair becomes a collective variable of one `CustomCVForce`, and λ is a Context parameter. Nothing
  is re-uploaded when λ moves, for any System, explicit solvent included;
* at every observation that saves a frame, `V0`, `V1` and `V(λ)` at that frame, with `V1 − V0`
  checked against an independent evaluation before the row is written;
* independent per-path seeds derived from the run seed, so paths are independent and the whole set
  is reproducible;
* frame selection from an explicitly identified source ensemble, without repetition by default.
  `evenly_spaced` spreads the paths over the WHOLE eligible window, first and last frame included
  (before this release its stride was rounded down, so 64 paths from 95 frames used only the first
  64);
* a source ensemble generated by the same run (`ais_source.generate: true`): minimisation of the
  unscaled System, equilibration and a `source` stage on V0, then the paths;
* four INDEPENDENT reporting cadences -- work observations, trajectory frames, a per-path state
  table, and checkpoints -- each of which must divide `switching_steps` on its own;
* per-path directories, each with its own observation table and `system.csv`, and one published
  NetCDF trajectory per global path id;
* restart: a completed path is skipped rather than appended to, and an interrupted one resumes
  from its checkpoint without duplicating a work row, a state row or a frame;
* validation before completion: a path's CV output is validated in full -- row count against the
  schedule, switching-step grid, path and source-frame identity on every row, protocol column
  order, λ schedule and endpoints, the empty-or-index semantics of the frame references, finite
  values, the sidecar, and the cumulative CV cost against the row count -- *before* `completed.json`
  is committed, by the same rules that refuse a completed path when a later invocation skips it.

AIS scales nothing itself. The REST2 scaler, and the omega exclusion that belongs to it, act where
a scaled state is BUILT; AIS consumes the result.

## What it does not implement

* no free-energy estimator. MD-tools produces the work values; computing ΔF from them — Jarzynski,
  BAR, or anything else — is the analysis, and is deliberately outside this package;
* no bidirectional (Crooks) protocol in one run — run the reverse as a second campaign;
* no schedule other than `linear` and `tau-linear` (for example Amber's smoothstep `S_p(λ)`), no
  partial λ window;
* no atom mapping, softcore or dummy atoms;

## Required inputs

V0 and V1 as System/topology pairs, **and a source trajectory** of V0:

```bash
md-openmm build-md -odir ./hot/ --config hot.config   # a run of the V0 System
cd hot && ./run.sh
```

**The source is verified when it can be.** A stage's whole-system AMBER NetCDF records the
`system_sha256` of the System it integrated when that System is its `-s` unmodified, and AIS
refuses a source whose recorded digest is not `sha256(-s)`. A DCD, a foreign file, or a stage that
built its System in memory records none, and the log then says the ensemble was ASSERTED, in those
words. What is always checked is the file itself: its atom count against `-p` and `-s`, and that
its contents are a genuine DCD or NetCDF rather than something with the right suffix.

## Minimal sequence

```bash
md-openmm build-md -odir ./AIS-run1 --config example.config
cd AIS-run1 && ./run.sh V0.pdb V0.xml ../hot-run1/whole_prod1.nc
```

`run.sh` takes V0 and the source as arguments and V1 from `V1_TOPOLOGY` / `V1_SYSTEM`, which default
to `../build/built.pdb` and `../build/built.xml`. It **requires** the source explicitly: a wrong
source is not a slower run, it is a different measurement.

`./run.sh ... --check` runs the preflight only — the end-state pair, schedule divisibility, the
barostat refusal, the source digest, the eligible-frame count — and writes nothing. Run it before
committing to a long set of paths.

## Generated files

```text
AIS-run1/
├── resolved.config
├── AIS.py           the compact entry point
├── run.sh
└── build-md.log
```

AIS carries **no minimisation or equilibration chain**: it consumes an ensemble that already exists.

Running produces:

```text
selected_source_frames.csv    which source frame each path starts from, and with which seeds,
                              written BEFORE any dynamics
AIS_traj0000.nc …             one genuine NetCDF trajectory per GLOBAL path id
AIS_work.csv                  one row per (path, switching step), sorted by that pair
AIS_paths.csv                 one row per path: the work distribution
AIS_hs.csv                    the frame-aligned subset, for Hummer–Szabo reweighting
AIS_run.json                  the run's identity: both end-state digests, source, schedule, seeds
path_0000/observations.csv    the work rows for one path
path_0000/system.csv          its state table
path_0000/completed.json      the machine record that says the path finished
AIS.out / AIS.log             the readable output and the provenance record
```

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `ais.number_of_paths` | 100 | independent realisations. The spread of the work distribution is the result |
| `ais.switching_steps` | 50000 | **work is path-length dependent**: a faster switch does more dissipative work |
| `ais.observation_interval_steps` | 2500 | how often the WORK is measured. This one is the method |
| `ais.parameter_update_interval_steps` | 1 | how often λ moves |
| `reporting.crd_printout_solute` | follows the observations | frames in `AIS_trajNNNN.nc`. Set it larger for a smaller file |
| `reporting.info_printout` | follows the observations | rows in `path_NNNN/system.csv`. 0 disables the table |
| `reporting.checkpoint_printout` | follows the observations | how often a path becomes resumable. 0 means an interrupted path restarts from its source frame |
| `ais_source.trajectory` | null | required. The equilibrium ensemble of V0 the paths start from |
| `ais_source.allow_repeated_frames` | false | two paths from one configuration are not two independent realisations |

`ais.tau_start`, `ais.tau_end`, `ais.work_measurement` and `ais.verify_every_updates` are retired and
refused with the migration.

## Reporting and registration

Each path's `observations.csv` carries λ, the incremental and cumulative work, the reduced work,
the temperature and the seeds. λ is the only persisted switching coordinate.

`system.csv` is a different question, at its own cadence: protocol step, λ, potential, kinetic and
total energy, temperature, and volume and density where the box makes them meaningful. It is how
the path is *behaving* while the Hamiltonian moves, which shows a switch is too fast long before
the work distribution does.

The four cadences are independent and each must divide `switching_steps` exactly. Work every 10
steps with frames every 50 is an ordinary thing to want. A fifth, collective-variable observations,
is described in [Collective variables](../../basics/collective-variables.md); it is on the
parameter-update grid and also divides `switching_steps`. See
[Running](../../basics/md-run.md#ais) for the resume contract.

### The observation potentials

| column | measured at |
|---|---|
| `potential_v0_kj_mol` | V0 at the coordinate the row SAVED |
| `potential_v1_kj_mol` | V1 at the same coordinate |
| `potential_direct_kj_mol` | V(λ) at the same coordinate, under the row's λ |

They are filled only on rows that name a `coordinate_frame_index`, and are EMPTY otherwise — never a
neighbouring frame's values. `potential_direct = (1 − λ)·potential_v0 + λ·potential_v1` holds on
every row, so the identity is checkable from the file.

`AIS_hs.csv` is the **frame-aligned subset**: only rows whose coordinate was actually saved, so
every HS row's potentials and its work describe *one* configuration. A row without a stored frame
has empty potential cells by schema, and including it would put those empty cells in front of a
reweighting with no way to notice.

Two probes, never confused: the work is measured at the frozen **pre-switch** coordinate `x_j`,
where work is defined; the observation potentials at the **saved** coordinate `x_t`. They are the
same arithmetic at different configurations, and the failure they guard against — attaching one's
numbers to the other's coordinate — is invisible in the output.

## Limitations

* Work values depend on the switching length. Two sets run at different `switching_steps` are not
  comparable.
* Work values from 0.5.3's tau switch and 0.5.4's linear two-state switch are not comparable path by
  path: the endpoints, and so ΔF, are the same; the path, and so the work distribution, is not.
* The estimator is not provided; an exponential average over too few paths is dominated by rare
  low-work realisations, and this package does not protect you from that.
* Source quality is your responsibility: paths starting from a poorly equilibrated ensemble produce
  a work distribution for that ensemble, not for the one you meant.
* Work values are not bit-reproducible across GPUs or worker counts, and on CUDA a resumed path is
  not bit-for-bit the path an uninterrupted run would have taken: the mixing force's inner Contexts
  keep state no checkpoint captures. Path IDENTITY is reproducible: path *n* starts from the same
  frame with the same seeds and writes `AIS_traj000n.nc` whatever the world size, and a resume
  restores the committed state exactly.
* Each differing force is evaluated for both end states, so a step costs more than plain dynamics
  of either — about 2.8× on a 6232-particle explicit system where only the nonbonded force and the
  torsions differ.

## References

Annealed importance sampling [@neal2001ais]; the Jarzynski equality
[@jarzynski1997equality]; REST2 scaling [@wang2011rest2]; Amber's alchemical end-state mixing and
Jarzynski protocol (Amber 2026 manual, §27.1 and §27.8).
