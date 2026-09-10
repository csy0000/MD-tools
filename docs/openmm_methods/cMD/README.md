# cMD — conventional molecular dynamics

Ordinary MD: one system, one Hamiltonian, one trajectory. It is also the method that produces the
*source ensembles* the other protocols consume — a fixed-τ cMD run at the ladder's top rung is how
an rREST2 reservoir and an AIS source are generated.


## The two example files beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm build-md` |
| [`example.in`](example.in) | what `md-run` then reads — the production stage of a conventional MD chain |

`build-md` generates the `.in` from the `.config`; you do not normally write one by
hand. It is shipped here because it is the file the run actually reads, and because
every setting in it carries the schema's own description as a comment — so the meaning
of a key can be looked up where it is used rather than in the source.

`resolved.config`, written beside the `.in` at run time, stays AUTHORITATIVE: the `.in`
is resolved into it, and that resolved document is what the run reads.

Both are checked by `tests/test_method_example_inputs.py`, which regenerates the `.in`
from the `.config` and fails if they have drifted — an example that no longer matches
the engine is worse than none.

## Ensemble

| solvent | minimisation | equilibration | production |
|---|---|---|---|
| explicit (TIP3P, OPC) | NVT | restrained NVT → restrained NPT → free NPT | **NPT** |
| implicit (GBn2) | NVT | restrained NVT → restrained NVT → free NVT | **NVT** |

Implicit solvent has no box, so it has no volume to control: there is no barostat anywhere in the
System and no stage claims NPT. The implicit stages are *renamed* (`eq_nvt_posres_2.py`,
`eq_nvt_free.py`) rather than being NPT stages with the pressure quietly ignored. See
[scientific defaults §5](../../scientific-defaults.md).

## What MD-tools implements

* the five-stage chain above, each stage a separate restartable script;
* positional restraints on solute heavy atoms during the restrained stages, released in one step;
* Langevin-middle integration at 300 K with 1.0 ps⁻¹ friction;
* a Monte Carlo barostat in explicit solvent, active only in the NPT stages;
* a fixed-τ mode (`dynamics.tau > 0`) that runs cMD at one rung of the REST2 ladder.

## What it does not implement

* no free-energy estimator, no umbrella sampling, no metadynamics;
* no Berendsen barostat — [documented and deliberately absent](../../scientific-defaults.md#101-berendsen-pressure-coupling-documented-not-implemented);
* no automatic equilibration detection. The stage lengths are yours to choose.

## Required inputs

`built.pdb` and `built.xml` from `build-top`. Nothing else.

## Minimal sequence

```bash
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log
md-openmm build-md  -odir ./md_script/ --config example.config
cd md_script && ./run.sh
```

`run.sh` runs the chain in order. To drive one stage yourself:

```bash
python min.py -p ../built.pdb -s ../built.xml -o min.out -r min.xml -log min.log

# or, the Amber-like way -- the same run, reaching the same installed code:
md-openmm md-run -i min.in -p ../built.pdb -s ../built.xml \
    -o min.out -r min.xml -log min.log
```

## Generated files

```text
md_script/
├── resolved.config     the single resolved declaration of this workflow
├── build-md.log        what was resolved, and from which defaults
├── min.py              five compact entry points, one per stage
├── eq_nvt_posres.py
├── eq_npt_posres.py    (explicit)  /  eq_nvt_posres_2.py (implicit)
├── eq_npt_free.py      (explicit)  /  eq_nvt_free.py     (implicit)
├── cMD.py
└── run.sh
```

`--all-in-one` emits a single `md.py` carrying the same chain instead of five scripts.

Running produces, per stage: `<stage>.log` (readable log **and** the machine record),
`<stage>.xml` (final state), `<stage>.chk` + `.chk.json` (checkpoint and its fingerprint),
`mdout.csv` (the state table: energy, temperature, volume, density) and the coordinate
streams `solute_prod<N>.nc` and `whole_prod<N>.nc` -- the solute alone and every atom,
each at its own interval. Equilibration stages write `solute_<stage>.nc` and
`whole_<stage>.nc`, so no two stages share a filename.

## Restart and continuation

A stage that already reports `status: completed` in its own machine record is skipped rather than
silently rerun. An interrupted stage resumes from its checkpoint **only if** the checkpoint
fingerprint still matches the resolved configuration, the System and the topology. Editing
`resolved.config` after a checkpoint exists is therefore a refusal, not a silent restart.

Each stage records the parent state it consumed with that file's SHA-256, so a parent regenerated
after a child consumed it is caught rather than becoming false ancestry.

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `stages.production_steps` | 2500000 | the length of the run. An integer step count, always |
| `dynamics.timestep_fs` | `auto` | `auto` reads the built System's masses: 2 fs ordinary, 4 fs if HMR is present |
| `dynamics.tau` | 0.0 | > 0 runs at one fixed rung of the REST2 ladder, which is how a reservoir or AIS source is made. A scaled run is NVT by construction |
| `dynamics.phase_space_printout` | 0 | > 0 writes positions **and velocities**; required to generate an rREST2 reservoir |
| `reporting.crd_printout_solute` | 1000 | solute trajectory interval, in steps |

## Reporting and registration

The stage logs are the machine records `data-register` reads. Completion is taken from
`status: completed` in that record, never from prose. Register a finished directory with:

```bash
md-openmm data-register -idata ./data/ALA-cMD -project_name ALA -data_name ALA-cMD -year 2026
```

See [data registration](../../data_register/README.md).

## Limitations

* Stage lengths are defaults, not recommendations: 2.5 M steps is 5 ns at 2 fs, which converges
  nothing in particular.
* The restraint is released in one step between the last restrained stage and the first free one.
  A system that needs a gradual release needs more stages than this chain provides.
* Fixed-τ cMD samples a *scaled* Hamiltonian. It is not a physical ensemble unless τ = 0.

## References

Langevin-middle integration [@zhang2019lfmiddle], Monte Carlo barostat
[@chow1995mcbarostat; @aqvist2004mcbarostat], and the force fields in
[scientific defaults](../../scientific-defaults.md).
