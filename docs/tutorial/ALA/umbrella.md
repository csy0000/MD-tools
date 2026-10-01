# Umbrella sampling: alanine dipeptide φ, one window per run

**Tested against md-tools `0.6.5`.** Every command, every configuration, every number and every
block of output on this page comes from a run executed as written: 36 windows and a 200 ns
unbiased reference, implicit GBn2, on one NVIDIA RTX A5000. **Total GPU time: about 2 hours**, of
which the reference is 1.9 h and all 36 windows together are 11 minutes.

The profile is checked against an unbiased 200 ns run of the **same** Hamiltonian. Which number
is the method's error takes some care, because not every check is a prediction:

| check | result | what it is worth |
|---|---|---|
| the **ψ** marginal vs the unbiased run | **rms 0.23 kJ/mol**, max 0.76 | **the method's error** — ψ was never biased, so this is a genuine prediction |
| the φ marginal vs the unbiased run | rms 0.15 kJ/mol, max 0.37 | a consistency check: the windows were biased *along* φ, so reproducing it is nearly circular |
| `pymbar` MBAR on the same samples | rms 0.13 kJ/mol | the estimator, not the sampling — two estimators over one dataset |
| the estimator vs a known synthetic double well | max 0.77 kJ/mol | arithmetic only; no simulation involved |

**So quote 0.23, not 0.15.** `kT` at 300 K is 2.49 kJ/mol, so the method's error here is about
`kT`/11. The φ agreement is reported because a *disagreement* there would have been damning, not
because agreeing proves much.

!!! warning "What two agreeing marginals cannot tell you"
    A marginal integrates over the other coordinate, so it is insensitive to exactly the failure a
    joint density partition catches: getting both basin populations right while putting the wrong
    joint structure between them. A reference can look converged in every 1D projection and still
    be several percentage points out in a cluster population — measured elsewhere on this host at
    6.27 pp spread for a density partition against 1.05 pp for a single-CV arc on the same runs.

    The full 2D φ–ψ surface is the honest target for this molecule and **it is not computed here**.
    Two 1D marginals agreeing is weaker, and is all this page claims.

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
every stage. A 22-atom implicit system is one of the few cases where a CPU run is not absurd, and
it is a reasonable way to check that a command chain works before spending a card on it — but a
CPU run is **not** evidence for a CUDA result. Every number on this page comes from the A5000.

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

## 6. The profile, and what it is checked against

36 windows: 24 at k = 100 kJ/mol/rad² on a 15° grid, and 12 at k = 1000 on a 5° grid across the
steep barrier (section 6.1 explains why the second set exists). 5 ns of production each, 180 ns in
windows altogether.

```text
phi      PMF (kJ/mol)   feature
 -72.5        0.00       global minimum, the C7eq/alpha region
-152.5        3.46       the extended/beta basin
 +57.5        7.77       the alphaL basin
  +2.5       30.22       the eclipsed barrier between the negative-phi basins and alphaL
+127.5       60.13       the highest barrier on the circle
```

**The checks that could have failed.** A separate **unbiased** 200 ns run of the same system, same
implicit GBn2 Hamiltonian, no bias anywhere: its φ histogram gives `F = −kT ln p` directly, with
no reweighting to get wrong. It crosses φ = 0 sixty-six times, so it is a reference rather than
another under-sampled run, and it spends 1.92% of its time in the αL basin.

```text
$ python umbrella_vs_unbiased.py ../ala-phi-reference/run1/cMD.cv.csv w* s*
coverage: 72 of 72 bins sampled; thinnest sampled bin 266, median 2146
overlap graph: 36 windows, largest component 36  connected
36 windows on phi_ALA; unbiased reference 180001 observations, 66 sign changes across 0

comparable bins (>= 50 unbiased samples): 38 of 72
  rms deviation   0.15 kJ/mol
  max deviation   0.37 kJ/mol at phi = 72.5 deg
  (kT at 300 K is 2.49 kJ/mol)
```

**The ψ marginal is the half that was a prediction.** The windows biased φ and merely *reported*
ψ, so reweighting the same samples to the unbiased ensemble and binning on ψ asks the windows
about a dimension they never controlled. If ψ relaxed slowly against a 5 ns window, this is where
it would show:

```text
psi marginal, reweighted from the phi-biased windows vs the unbiased run
  30 comparable bins, rms 0.23 kJ/mol, max 0.76 at psi = -145 deg
```

It passed, so ψ relaxation within a 5 ns window was adequate at this force constant. Had only the
φ comparison been run, this page would have quoted 0.15 kJ/mol for a profile whose other
coordinate had never been checked at all.

**Only 38 of 72 bins are comparable, and that is the point rather than a shortfall.** The unbiased
run is silent exactly where the profile is most valuable — it never visits the top of a 60 kJ/mol
barrier — so those bins are reported as *unchecked* rather than averaged into the agreement. An
umbrella profile that agreed with an unbiased run everywhere would be a profile that had bought
nothing.

**Do not compare this against the explicit-solvent runs on this system.** The
[1 µs cMD reference](cMD.md#4-what-10-ns-sampled-against-a-microsecond) is explicit TIP3P and this
profile is implicit GBn2. Those are two different Hamiltonians with genuinely different free-energy
surfaces, so agreement would be luck and disagreement uninterpretable. An earlier draft of this
page promised exactly that comparison; it was replaced with the implicit reference above, which is
the only version of the check that can fail for the right reason.

### 6.1 Why twelve of the windows are ten times stiffer

The first 24 windows left a hole at the top of the barrier, and the arithmetic says why. A window
on a slope does not sit at its centre: it settles where the restraint balances the gradient, a
displacement of `slope / k`. Around φ = 120° the PMF rises about 52 kJ/mol/rad, so at
k = 100 kJ/mol/rad² the expected displacement is 0.52 rad — **30°**. Measured: the window centred
at 120° sampled a mean of 89.1°, and the one at 135° sampled 167.1°. They slid off the barrier in
opposite directions and left its top unsampled.

k = 1000 puts the displacement near 3° and σ near 2.9°, which is why those windows are spaced 5°
rather than 15°. Mixing force constants across one profile is fine: each window carries its own
`k` in its own record, and WHAM uses each window's own bias.

**Stiff windows cannot be started from an unbiased equilibration.** Launched that way, 7 of the 12
died with `ValueError: Energy is NaN` in 1.9 s — the equilibration ends near φ = −161°, so a
window centred at 115° starts 84° (1.47 rad) from its centre and the bias alone is
0.5 · 1000 · 1.47² ≈ 1080 kJ/mol with a 1470 kJ/mol/rad force. The ones that survived were simply
those whose centre happened to fall within ~60° of the equilibrated value.

The remedy is to seed each window from its neighbour's endpoint, so the initial displacement is
one 5° spacing instead of most of a circle:

```bash
# marching along the barrier: each window starts where the previous one finished
md-openmm md-run -i ../input/umbrella.in -p ../build/built.pdb -s ../build/built.xml \
    -c ../../sp140k1000/run1/umbrella.xml -odir .
```

The equilibration chain is skipped deliberately: a configuration already equilibrated under a bias
5° away is a better starting point than an unbiased one 80° away.

### 6.2 What makes a set of windows sufficient

Not pairwise overlap between neighbours. That is the obvious test and it fails exactly here: sorted
by centre, `wp120` and `wp135` look adjacent, but one sampled 89° and the other 167°, so "next by
centre" is not "next in sampled space". Sorting this profile by centre and demanding neighbourly
overlap reported five gaps between windows that in fact overlap other windows perfectly well.

What WHAM requires is that the sampled range is **covered** and that the overlap graph is
**connected** — an island of windows joined to the rest by nothing has a free-energy offset no data
constrains. Both are independent of where the centres sit and of how far any window slid, and both
are reported above. `umbrella_analysis.py` refuses a PMF when either fails.

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

**It refuses a PMF the windows do not support.** WHAM given an uncovered region still converges —
to a curve whose offsets across that region no data constrains, which is a smooth, plausible PMF
containing invented barrier heights with nothing in the output saying which parts are unsupported.
So coverage and connectivity are checked first (section 6.2) and a failure is an exit, naming what
is wrong:

```text
coverage: 68 of 72 bins sampled; thinnest sampled bin 266, median 2146
overlap graph: 24 windows, largest component 19
REFUSING a PMF.
  4 bin(s) inside the range have NO samples from any window, at phi = 112.5, 117.5, 122.5, 127.5
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
