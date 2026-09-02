# AIS — annealed importance sampling

Non-equilibrium switching. Many independent paths start from configurations drawn from an
equilibrium ensemble at τ_start, anneal the Hamiltonian to τ_end while the coordinates propagate,
and accumulate the **work** done along the way [@neal2001ais]. The work distribution is what the
method produces; the Jarzynski equality relates its exponential average to a free-energy difference
[@jarzynski1997equality].

AIS is not a sampling protocol that produces one long trajectory. It produces *N* short ones and a
number for each.

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
* independent per-path seeds derived from the run seed, so paths are independent and the whole set
  is reproducible;
* frame selection from an explicitly identified source ensemble, without repetition by default;
* per-path directories, each with its own trajectory and observation table;
* restart: a completed path is skipped rather than appended to.

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

The source's own record is checked against `ais.tau_start`: a source generated at a different τ is
refused. There is no second, user-declared "source tau" to disagree with it.

## Minimal sequence

```bash
md-openmm build-md -odir ./md_script/ --config example.config
cd md_script && ./run.sh ../built.pdb ../built.xml ../hot/cMD.dcd
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

Running produces `selected_initial_frames.csv` (which source frame each path started from, written
*before* any dynamics), one `trajectory_NNNN/` per path with its observation table, and `AIS.log`.

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `ais.number_of_paths` | 100 | independent realisations. The spread of the work distribution is the result |
| `ais.tau_start` / `tau_end` | 0.5 / 0.0 | the path. They must differ, or every work value is zero |
| `ais.switching_steps` | 50000 | **work is path-length dependent**: a faster switch does more dissipative work |
| `ais.observation_interval_steps` | 2500 | how finely the path is observed |
| `ais_source.trajectory` | null | required. The equilibrium ensemble the paths start from |
| `ais_source.allow_repeated_frames` | false | two paths from one configuration are not two independent realisations |

## Reporting and registration

Each path's observation table carries τ, the incremental and cumulative work, the reduced work, the
temperature and the seeds. τ is the only persisted scaling coordinate — neither `s` nor `√s` is ever
written.

## Limitations

* Work values depend on the switching length. Two sets run at different `switching_steps` are not
  comparable.
* The estimator is not provided; an exponential average over too few paths is dominated by rare
  low-work realisations, and this package does not protect you from that.
* Source quality is your responsibility: paths starting from a poorly equilibrated ensemble produce
  a work distribution for that ensemble, not for the one you meant.

## References

Annealed importance sampling [@neal2001ais]; the Jarzynski equality
[@jarzynski1997equality]; REST2 scaling [@wang2011rest2].
