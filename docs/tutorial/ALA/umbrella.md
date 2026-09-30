# Umbrella sampling: alanine dipeptide φ, one window per run

**Tested against md-tools `0.6.5`.** Every command, every configuration and every block of output
on this page was executed as written, in implicit solvent, on the CPU.

!!! warning "The profile itself is not measured yet"
    [Section 6](#6-what-a-profile-needs-that-this-page-does-not-yet-have) is a **gap, not a
    result**: the per-window means and the PMF need a CUDA run, and this page will not carry
    numbers from the CPU runs that verified its commands. Everything up to section 5 — the build,
    the resolved stages, the generated tree, the refusals, the window digests — is measured, and
    is what a reader needs to start a profile. What a converged profile looks like is not here
    yet, and nothing below pretends otherwise.

Umbrella sampling here is **conventional MD with a bias on named collective variables**, and those
variables reported as the run goes. It produces a biased trajectory and the CV series that goes
with it. It does not produce a free energy: WHAM, MBAR and any estimator over a set of windows are
analysis, and they live in the project asking the question rather than in the engine generating
the samples ([section 7](#7-the-estimator-is-not-in-the-engine)).

**Starts from a built system.** This page builds its own, because it uses implicit solvent — for a
22-atom peptide that is often enough, and it has no box, no barostat and no salt to equilibrate.

## 1. What is measured, and what is biased

Two files, and the separation is the whole design. `cv.yaml` says **what is measured**;
`umbrella.yaml` says **which of those are biased, how, and where**. A restraint names a variable
**by name** and carries no atom indices of its own, so a run cannot bias one torsion and report
another.

```yaml
# cv.yaml -- what is measured. Indices are 0-based into the built topology.
schema_version: 1
collective_variables:
  - {name: phi_ALA, type: torsion, atom_indices: [4, 6, 8, 14]}
  - {name: psi_ALA, type: torsion, atom_indices: [6, 8, 14, 16]}
```

Those two quartets are φ and ψ of the capped dipeptide. They are worth checking rather than
trusting, because a CV series naming the wrong four atoms cannot be told apart from one naming the
right four — both are plausible numbers in the right range under the right column heading. Against
the built topology:

```text
phi_ALA  [4, 6, 8, 14]   ->  ACE:C  ALA:N  ALA:CA  ALA:C
psi_ALA  [6, 8, 14, 16]  ->  ALA:N  ALA:CA  ALA:C  NME:N
```

which is φ = C(−1)–N–CA–C and ψ = N–CA–C–N(+1), as they should be. The same indices hold for an
**explicit** build of this structure too: the solute is written first and its ordering is
preserved, so a box does not renumber these four atoms. That was checked both ways rather than
assumed.

```yaml
# umbrella.yaml -- what is biased. One window: hold phi near -60 deg.
schema_version: 1
restraints:
  - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 100.0}
```

**The two forms answer different questions.**

| form | energy | what it is |
|---|---|---|
| `harmonic` | `0.5*k*dtheta^2` | an umbrella **WINDOW** — biased everywhere, including at its own centre, so **every** sample needs reweighting |
| `flat_bottom` | `0.5*k*max(0, abs(dtheta) - w)^2` | a **BOUND** — exactly zero inside the window, so samples there are unbiased and need no correction |

A flat-bottom restraint is the right tool for keeping a molecule in a basin and the wrong one for
measuring across it. This page builds a profile along φ, so φ gets a harmonic window. `dtheta` is
wrapped onto the circle, so a restraint at 170° pulls a torsion at −170° the short way round.

`collective_variables.file` and `interval_steps` are both **required** under this protocol, and
required together: a window that biased a variable and never recorded it would produce a trajectory
nobody can reweight.

## 2. Build the system

The structure ships with the package as `tests/data/ALA.pdb`.

```yaml
# build.config
solvent: {model: GBn2}
```

```bash
md-openmm build-top -i ALA.pdb --config build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log
```

```text
Counts
------
  atoms                       22
  residues                    3
  solute atoms                22
  waters                      0 (implicit solvent)
  ions                        none

Summary
-------
  built 22 particles, 3 residues, implicit GBn2
  status: completed
```

An implicit build has no box, no barostat, no counter-ions and no NPT stage anywhere downstream.

## 3. Generate one window

```yaml
# umbrella.config
protocol: umbrella
solvent: implicit

dynamics: {timestep_fs: 2.0, temperature_K: 300.0}

stages:
  minimization_iterations: 5000
  restrained_nvt_steps: 25000      # 50 ps
  restrained_npt_steps: 25000      # 50 ps -- renamed under implicit solvent, see below
  unrestrained_npt_steps: 50000    # 100 ps
  production_steps: 1000000        # 2 ns at 2 fs

reporting: {crd_printout_solute: 500, info_printout: 5000, checkpoint_printout: 50000}

collective_variables: {file: cv.yaml, interval_steps: 100}
umbrella: {file: umbrella.yaml}
```

```bash
md-openmm build-md -odir ./run1 --config umbrella.config
```

```text
Stages
------
  min                         5000 iterations
  eq_nvt_posres               25000 steps = 50 ps (0.05 ns), NVT
  eq_nvt_posres_2             25000 steps = 50 ps (0.05 ns), NVT
  eq_nvt_free                 50000 steps = 100 ps (0.1 ns), NVT
  umbrella                    1000000 steps = 2000 ps (2 ns), NVT
```

**The stage names are NVT and the keys are not.** `restrained_npt_steps` keeps its name in the
configuration while the stage it generates is called `eq_nvt_posres_2` and runs at fixed volume:
under implicit solvent there is no box to couple a barostat to, so the stages are *renamed* rather
than run at a pressure that would mean nothing. Writing `unrestrained_nvt_steps` in the
configuration is refused, by name, with the key it meant:

```text
build-md: umbrella.config: stages: unknown key(s) 'unrestrained_nvt_steps'
  (did you mean 'unrestrained_npt_steps'?)
```

**The CV interval must divide every stage it applies to**, not just production. An interval of
1000 against these 25000-step stages is fine; against a 500-step stage it is refused rather than
rounded, because a final partial gap breaks the uniform spacing every downstream time-series
analysis assumes and none can detect:

```text
CVScheduleError: stage eq_nvt_posres: collective_variables.interval_steps = 1000 does not divide
the stage step count (500). It would leave a final gap of 500 steps ...
```

`build-md` copies both YAML definitions into the generated directory under content-addressed
names, so a run never depends on a path outside it and a definition that changed cannot quietly
replace the one a previous run used. It also records the restraint in `build-md.log`, resolved:

```yaml
# umbrella_definition:
#   copied_as: umbrella.ac66d59f044b.yaml
#   restraints:
#   - cv: phi_ALA
#     form: harmonic
#     centre_deg: -60.0
#     force_constant_kj_mol_rad2: 100.0
#     atom_indices: [4, 6, 8, 14]
```

That last line is the point of resolving by name: the output says which four atoms were biased,
without the reader opening two files and trusting that they match.

## 4. Run the window

```bash
cd run1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./run.sh
```

Without `PCI_BUS_ID`, CUDA may number the cards differently from `nvidia-smi`. CUDA is the default
and is mandatory; `--cpu` is the one per-run override, and `./run.sh --cpu` passes it through to
every stage. A 22-atom implicit system is one of the few cases where a CPU run is not absurd —
that is how this page's commands were verified — but a CPU run is **not** evidence for a CUDA
result and none of its numbers appear here.

`run.sh` is five `md-run` calls in order: `min`, three equilibration stages, then the biased
production stage. A stage that already reports completion in its own machine record is skipped;
an interrupted one resumes from its last committed checkpoint.

## 5. A profile: one system root per window

**Two different windows cannot share a system root.** `input/umbrella.in` carries the window's
restraint as `umbrella_file = umbrella.<digest>.yaml`, and `input/` is shared by every run on a
root, so a second window under the same root is refused:

```text
build-md: input/umbrella.in already exists and is not what this configuration resolves to.
  `input/` is shared by every run on this system: the runs already beside it read THIS file, so
  replacing it would make their inputs describe a different experiment than the one they ran.
```

The refusal is right, and it is exactly one file wide: across three centres, `min/resolved.config`,
`input/min.in` and `input/eq_1.in` come out byte-identical, and only `umbrella.in` differs. So
`-odir ./umbrella-run1`, `./umbrella-run2` — which is what indices are for, namely repeats of one
window differing only in their seed — is the wrong shape for a profile. Give each window its own
root, over one shared `build/`:

```bash
for centre in -150 -120 -90 -60 -30; do
  mkdir -p w${centre}
  ln -sfn ../build w${centre}/build          # relative, so the tree stays movable
  cp cv.yaml umbrella.config w${centre}/

  cat > w${centre}/umbrella.yaml <<YAML
schema_version: 1
restraints:
  - {cv: phi_ALA, form: harmonic, centre_deg: ${centre}.0, force_constant: 100.0}
YAML

  ( cd w${centre} && md-openmm build-md -odir ./run1 --config umbrella.config \
      && cd run1 && ./run.sh )
done
```

**Then check that the windows actually differ, before spending a GPU on them.** The copied
definition is content-addressed, so one `ls` answers it:

```bash
$ ls w*/run1/umbrella.*.yaml
w-120/run1/umbrella.0fe7a1060b0b.yaml
w-150/run1/umbrella.8ce7c8747e0b.yaml
w-90/run1/umbrella.e0ae5f575a75.yaml
...
```

Five windows, five digests. This is worth a line of script because the opposite failure is
invisible: edit the restraint into a file the configuration does not name, and every window
resolves the same definition, every run reports `status: completed`, and the profile is five
copies of one window with nothing anywhere saying so. An earlier version of the method README did
exactly that.

## 6. What a profile needs that this page does not yet have

Everything above is measured. The following is **not**, and is what remains before this page can
claim a PMF:

| what | status |
|---|---|
| per-window φ distributions, 2 ns × 5 windows on CUDA | **not run** — needs a card |
| window overlap (adjacent histograms must overlap, or no estimator can join them) | **not measured** |
| the PMF along φ, and its uncertainty | **not measured** |
| how the profile compares with the [1 µs cMD reference](cMD.md#4-what-10-ns-sampled-against-a-microsecond) | **not measured** — this is the check worth making, since that run's φ histogram is the truth a profile should reproduce |

The last row is why alanine dipeptide is the right system for this page. Its barriers are low
enough that a long unbiased run gives a reference distribution, so an umbrella profile here can be
checked against the answer rather than against another biased method. A profile that disagrees
with the microsecond is wrong, and that is a rarer thing to be able to say than it sounds.

Spacing and force constant are also a choice this page has not yet earned the right to
recommend: 30° spacing at k = 100 kJ/mol/rad² is a starting point, and whether adjacent windows
overlap at that spacing is exactly what the first run has to show.

## 7. The estimator is not in the engine

What a window produces is the biased series and the restraint that produced it, which is what an
estimator needs as input. The estimator itself — WHAM, MBAR, anything over a set of windows —
belongs in the project asking the question. Keeping that boundary is what lets md-tools stay a
3.3 MB wheel: the estimators bring dependencies the sampling does not need.

[`umbrella_analysis.py`](umbrella_analysis.py) beside this page reads the windows this page
generates, takes each window's restraint from **its own** content-addressed record rather than
from a command line, reports what each one sampled and whether adjacent windows overlap, and
produces the PMF by WHAM:

```bash
python umbrella_analysis.py w-150 w-120 w-90 w-60 w-30
```

Three things in it are worth knowing before you trust a curve it prints.

**It refuses a PMF across a gap.** WHAM given windows that do not overlap still converges — to a
curve whose relative offsets across the gap no data constrains, which is a smooth, plausible PMF
containing invented barrier heights with nothing in the output saying which parts are unsupported.
So overlap is measured first and a gap is an exit, naming the pair:

```text
adjacent overlap (fraction of the sparser neighbour's samples in the shared range)
       w-120 : w-60          0.0%   <-- GAP
REFUSING a PMF: 1 adjacent pair(s) do not overlap.
```

**It discards the leading 10% of each window** (`--discard-fraction`). The bias applies to the
production stage, so a window begins wherever *unbiased* equilibration left the molecule — for a
centre far from that point, outside the window entirely — and the first stretch is the relaxation
into the window rather than sampling of it. Keeping it puts weight at values the window never
equilibrated at, and WHAM turns weight into free energy. On the short windows used to check this
page, all three began at −161°, and that single retained sample in 101 became the global minimum
of the PMF, about 19 kJ/mol below the region the windows actually sampled.

**Its estimator is checked against a known answer**, because an estimator that is never checked
returns a plausible curve whichever way it is wrong:

```bash
$ python umbrella_analysis.py --self-test
self-test: 18 synthetic windows over a known double well
  bins recovered        72 of 72
  max deviation         0.77 kJ/mol
  rms deviation         0.40 kJ/mol
  OK: the recovered PMF matches the one that generated the samples
```

The script's docstring records what that self-test catches when broken on purpose — a force
constant wrong by 2× gives 21.7 kJ/mol, one window's centre wrong by 20° gives 7.5 — and the one
error it cannot see: shifting *every* centre by exactly one window spacing, which on a uniform
periodic set merely relabels each window as its neighbour. That limit is in the file rather than
left to be assumed away.

## 8. What it wrote

```text
ALA/
├── build/                     built.xml, built.pdb, built.log   (shared by every window)
└── w-60/
    ├── build -> ../build
    ├── input/                 min.in, eq_1.in, eq_2.in, eq_3.in, umbrella.in
    │                          cv.<digest>.yaml, umbrella.<digest>.yaml
    ├── min/                   min.xml, min.log, min.out, min.py
    └── run1/
        ├── eq/                eq_1.xml … eq_3.xml, their logs and CV series
        ├── solute_prod1.nc    solute trajectory
        ├── mdout.csv          the state table
        ├── energy_components.csv
        ├── umbrella.cv.csv    the CV series -- what an estimator reads
        ├── umbrella.cv.json   its sidecar: definition digest, resolved indices, units,
        │                      wrapping and sign conventions, column order
        ├── umbrella.xml       the final state
        ├── umbrella.checkpoints/
        └── umbrella.log       the machine-readable record
```

`umbrella.cv.csv` carries `step`, `time_ps`, `trajectory_frame_index` and one column per named
variable. **Steps are absolute** — the step axis continues through the run, so the production
stage of this window begins at the step equilibration ended on, not at 0. A
`trajectory_frame_index` is present only on rows whose configuration was actually written to
`solute_prod1.nc`, and empty otherwise: never `-1`, which would invite a reader to index from the
end of the file.

## Next

* the unbiased reference this profile should reproduce: [cMD: alanine dipeptide](cMD.md)
* the same barrier crossed by tempering instead of biasing: [REST2](REST2.md)
* the method reference, including both restraint forms and the energy checks:
  [umbrella sampling](../../openmm_methods/umbrella/README.md)
* [the system page](index.md) — the other methods for alanine dipeptide
