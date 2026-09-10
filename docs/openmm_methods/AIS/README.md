# AIS — annealed importance sampling

Non-equilibrium switching. Many independent paths start from configurations drawn from an
equilibrium ensemble at τ_start, anneal the Hamiltonian to τ_end while the coordinates propagate,
and accumulate the **work** done along the way [@neal2001ais]. The work distribution is what the
method produces; the Jarzynski equality relates its exponential average to a free-energy difference
[@jarzynski1997equality].

AIS is not a sampling protocol that produces one long trajectory. It produces *N* short ones and a
number for each.


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

## The work convention

For each parameter update *j*:

```
ΔW_j = U(τ_{j+1}, x_j) − U(τ_j, x_j)
```

**The parameters move at frozen coordinates, and only then does the configuration propagate.** That
ordering is the definition, not an implementation detail: evaluating the two energies at different
coordinates would compute a different quantity that happens to have the same units.

**Observation 0 precedes all work.** The first recorded observation is the source configuration
under the source Hamiltonian, before any parameter change and before any propagation, so its
cumulative work is exactly zero. A path whose first row is not zero work at τ_start is refused.

## Ensemble

**Fixed volume.** Switching is at constant volume, and a barostat in the prepared System is
refused. A volume move during a switching path would do work that the convention above does not
account for.

## What MD-tools implements

* an exact integer-step schedule: `switching_steps`, `parameter_update_interval_steps` and
  `observation_interval_steps` must divide consistently, and a schedule that would have to be
  rounded is refused with the arithmetic that would fix it;
* both endpoints included in the schedule;
* **two ways to measure the work, chosen with `ais.work_measurement`, and they cost different
  amounts.** `work` (the DEFAULT) evaluates `U(tau_k)` and `U(tau_k+1)` and subtracts: two energy
  evaluations per update, and the work integral is what you get. `components` instead probes the
  three-group basis at amplitudes (0, 0.5, 1) and derives the work from the fit -- three
  evaluations per update -- and additionally records the potential as a *function* of tau, which
  is what reweighting onto a different schedule, a different endpoint, or a Hummer-Szabo estimate
  at an unvisited tau requires. A single work value cannot produce that function, and re-running
  at another tau is not reweighting, so this is a decision to take **before** the run rather than
  after it. In `components` mode `ais.verify_every_updates` schedules the independent direct
  measurement the fit is checked against; 0, the default, checks the first update of every path;
* component columns are **absent** from a `work` run's tables rather than zero, and every
  `completed.json`, `AIS_paths.csv` row and run-identity document records which mode produced it
  -- so a reweighting script meets a missing column, and a directory cannot mix the two;
* independent per-path seeds derived from the run seed, so paths are independent and the whole set
  is reproducible;
* frame selection from an explicitly identified source ensemble, without repetition by default;
* four INDEPENDENT reporting cadences -- work observations, trajectory frames, a per-path state
  table, and checkpoints -- each of which must divide `switching_steps` on its own;
* per-path directories, each with its own observation table and `system.csv`, and one published
  NetCDF trajectory per global path id;
* restart: a completed path is skipped rather than appended to, and an interrupted one resumes
  from its checkpoint without duplicating a work row, a state row or a frame.
* validation before completion: a path's CV output is validated in full -- row count against the
  schedule, switching-step grid, path and source-frame identity on every row, protocol column
  order, tau schedule and endpoints, the empty-or-index semantics of the frame references, finite
  values, the sidecar, and the cumulative CV cost against the row count -- *before* `completed.json`
  is committed, by the same rules that refuse a completed path when a later invocation skips it.
  Everything before that commit can be undone, so a refusal there costs a rerun rather than a
  wrong answer recorded as finished.

The Hamiltonian is scaled by the **same** `md_tools.rest2.REST2Scaler` that REST2 uses. There is no
second AIS scaler.

## What it does not implement

* no free-energy estimator. MD-tools produces the work values; computing ΔF from them — Jarzynski,
  BAR, or anything else — is the analysis, and is deliberately outside this package;
* no bidirectional (Crooks) protocol;
* no adaptive schedule;
* it cannot generate its own source ensemble. AIS anneals *away from* an equilibrium ensemble and
  needs one to exist first.

## Required inputs

`built.pdb`, `built.xml`, **and a source trajectory** produced by a fixed-τ cMD run at
`ais.tau_start`:

```bash
md-openmm build-md -odir ./hot/ --config hot.config   # dynamics.tau: 0.5
cd hot && ./run.sh
```

`ais.tau_start` **asserts** what ensemble the source represents; nothing can verify it. A
coordinate trajectory does not record the Hamiltonian it was sampled under, so the log says so in
those words and records `tau_verified_from_file: false`. Check it against the run that produced the
file. What IS checked is the file itself: its atom count against `-p` and `-s`, and that its
contents are a genuine DCD or NetCDF rather than something with the right suffix.

A fixed-τ cMD run writes its whole-system stream as AMBER NetCDF, so `../hot/whole_prod1.nc` is the ordinary source. An earlier AIS or REST2
NetCDF works too; the format is read from the file's leading bytes, and a file whose suffix and
contents disagree is refused with both named.

## Minimal sequence

```bash
md-openmm build-md -odir ./md_script/ --config example.config
cd md_script && ./run.sh ../built.pdb ../built.xml ../hot/whole_prod1.nc
```

`run.sh` **requires** the source explicitly. There is no default for it: a wrong source is not a
slower run, it is a different measurement.

`./run.sh --check` runs the preflight only — schedule divisibility, the barostat refusal, the source
τ, the eligible-frame count — and writes nothing. Run it before committing to a long set of paths.

## Generated files

```text
md_script/
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
path_0000/observations.csv    the work rows for one path
path_0000/system.csv          its state table
path_0000/completed.json      the machine record that says the path finished
AIS.out / AIS.log             the readable output and the provenance record
```

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `ais.number_of_paths` | 100 | independent realisations. The spread of the work distribution is the result |
| `ais.tau_start` / `tau_end` | 0.5 / 0.0 | the path. They must differ, or every work value is zero |
| `ais.switching_steps` | 50000 | **work is path-length dependent**: a faster switch does more dissipative work |
| `ais.observation_interval_steps` | 2500 | how often the WORK is measured. This one is the method |
| `reporting.crd_printout_solute` | follows the observations | frames in `AIS_trajNNNN.nc`. Set it larger for a smaller file |
| `reporting.info_printout` | follows the observations | rows in `path_NNNN/system.csv`. 0 disables the table |
| `reporting.checkpoint_printout` | follows the observations | how often a path becomes resumable. 0 means an interrupted path restarts from its source frame |
| `ais_source.trajectory` | null | required. The equilibrium ensemble the paths start from |
| `ais_source.allow_repeated_frames` | false | two paths from one configuration are not two independent realisations |

## Reporting and registration

Each path's `observations.csv` carries τ, the incremental and cumulative work, the reduced work,
the temperature and the seeds. τ is the only persisted scaling coordinate — neither `s` nor `√s` is
ever written.

`system.csv` is a different question, at its own cadence: protocol step, τ, potential, kinetic and
total energy, temperature, and volume and density where the box makes them meaningful. It is how
the path is *behaving* while the Hamiltonian moves, which shows a switch is too fast long before
the work distribution does.

The four cadences are independent and each must divide `switching_steps` exactly. Work every 10
steps with frames every 50 is an ordinary thing to want. A fifth, collective-variable observations,
is described in [Collective variables](../../collective_variables/README.md); it is on the
parameter-update grid and also divides `switching_steps`. See
[Running](../../md-run.md#ais) for the resume contract.

### The λ-basis decomposition, and the columns it is read from

**`ais.work_measurement: components` only.** A `work` run has none of the columns below; see
*What MD-tools implements* above for the two modes, and
[the release note](../../release-notes/20260909-ais-work-measurement.md) for why `work` is the
default.

`U(τ, x) = U_non_scaled + √λ · U_sqrt_scaled + λ · U_lin_scaled`, with `λ = (1 − τ)²`. The three
components are recovered from three energy evaluations at amplitudes `a ∈ {0, ½, 1}`. Because the
fit gives `U` at *any* τ, both endpoints of an update come from it and the work needs no direct
evaluation at all — which is why a component update costs three evaluations rather than five.

The fit is checked against a directly measured potential on the updates
`ais.verify_every_updates` schedules — the first of every path by default — and a run refuses
rather than record components that do not reproduce `U(τ)`. Checking every update instead would
cost two more evaluations each time to re-establish something that cannot vary along a path: the
identity holds for a *system*, and a force whose τ-dependence falls outside the basis falls
outside it at every coordinate. Observation potentials, by contrast, are checked every time they
are taken, because they cost the direct evaluation anyway — it is one of their columns.

The components are written as separate columns rather than folded into a total, because that is
what makes a reweighting at a τ the run never visited possible without rerunning it:

| column group | columns |
|---|---|
| incremental work | `delta_work_non_scaled_kj_mol`, `delta_work_sqrt_scaled_kj_mol`, `delta_work_lin_scaled_kj_mol` |
| cumulative work | `total_work_non_scaled_kj_mol`, `total_work_sqrt_scaled_kj_mol`, `total_work_lin_scaled_kj_mol` |
| observation potential | `potential_non_scaled_kj_mol`, `potential_sqrt_scaled_kj_mol`, `potential_lin_scaled_kj_mol` |
| the check | `potential_reconstructed_kj_mol`, `potential_direct_kj_mol` |

**`lin_scaled` is never spelled `scaled`.** Three components carry a scaling and a bare "scaled"
does not say which, so the ambiguity would land in exactly the files a reweighting is built from.

`AIS_hs.csv` is written in both modes and is the **frame-aligned subset**: only rows whose
coordinate was actually saved, so
every HS row's potentials and its work describe *one* configuration. A row without a stored frame
has empty potential cells by schema, and including it would put those empty cells in front of a
reweighting with no way to notice.

Two probes, never confused: the work-basis probe is taken at the frozen **pre-switch** coordinate
`x_j`, where work is defined; the observation-potential probe is taken at the **saved** coordinate
`x_t`. They are the same arithmetic at different configurations, and the failure they guard
against — attaching one's numbers to the other's coordinate — is invisible in the output.

## Limitations

* Work values depend on the switching length. Two sets run at different `switching_steps` are not
  comparable.
* The estimator is not provided; an exponential average over too few paths is dominated by rare
  low-work realisations, and this package does not protect you from that.
* Source quality is your responsibility: paths starting from a poorly equilibrated ensemble produce
  a work distribution for that ensemble, not for the one you meant. `ais.tau_start` is asserted,
  not verified — nothing in a coordinate trajectory records the Hamiltonian it was sampled under.
* Work values are not bit-reproducible across GPUs or worker counts. Path IDENTITY is: path *n*
  starts from the same frame with the same seeds and writes `AIS_traj000n.nc` whatever the world
  size. The values differ by CUDA's reduction order, amplified along a chaotic trajectory.

## References

Annealed importance sampling [@neal2001ais]; the Jarzynski equality
[@jarzynski1997equality]; REST2 scaling [@wang2011rest2].
