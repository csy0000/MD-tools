# Umbrella sampling

Conventional MD with a bias on named collective variables, and those variables reported as the
run goes. One window per run.

## The example files beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm build-md` |
| [`example.in`](example.in) | what `md-run` then reads — one biased window |
| [`cv.yaml`](cv.yaml) | what is measured |
| [`umbrella.yaml`](umbrella.yaml) | what is biased |

`build-md` generates the `.in` and copies both YAML definitions into the generated
directory under content-addressed names, so a run never depends on a path outside it and a
definition that changed cannot quietly replace one a previous run used.

`tests/test_method_example_inputs.py` regenerates the `.in` from the `.config` and fails if
they have drifted.

## What this protocol produces, and what it deliberately does not

It produces **a biased trajectory and the CV series that goes with it**, plus a record of exactly
which restraint was applied. That is what a free-energy estimator needs as input.

It does **not** produce a free energy. WHAM, MBAR, any estimator over a set of windows — that is
analysis, and it belongs in the project asking the question rather than in the engine generating
the samples. Keeping the boundary there is what lets MD-tools stay a 3.3 MB wheel: the estimators
bring dependencies the sampling does not need.

## The restraint names a collective variable; it never redefines one

Two files, and the separation is the point:

| file | says |
|---|---|
| [`cv.yaml`](cv.yaml) | **what is measured** — named torsions, by explicit atom indices |
| [`umbrella.yaml`](umbrella.yaml) | **what is biased** — which of those names, which form, where |

A restraint refers to a variable by name. It carries no atom indices of its own, and that is
deliberate. `cv/definition.py` explains the hazard for measurement:

> a CV series that names the wrong four atoms cannot be told apart from one that names the right
> four, because both are plausible numbers in the right range with the right column heading

A restraint carrying its own indices would add a second, worse way to be wrong: the run would bias
one torsion and report another, and every column would still look correct. Resolving by name — and
loading the definition **once**, shared between the restraint and the reporter — makes that
failure inexpressible rather than merely unlikely.

For the same reason, `collective_variables.file` and `interval_steps` are both **required**: a
window that biases a variable and never records it produces a trajectory nobody can reweight.

## The two forms

```
harmonic      0.5*k*dtheta^2                 an umbrella WINDOW
flat_bottom   0.5*k*max(0, |dtheta| - w)^2   a BOUND
```

A harmonic window is biased everywhere, including at its own centre, so **every** sample needs
reweighting. A flat-bottom restraint is **exactly zero** inside its window, so samples there come
from the unbiased ensemble and need no correction at all — which makes it the right choice for
keeping a molecule in a basin, and the wrong one for measuring across it.

`dtheta` is wrapped onto the circle, so a restraint at 170° pulls a torsion at −170° the short way
round. Getting that wrong biases by an amount that depends on where the molecule happens to be;
it is checked by energy rather than by sampling, in `tests/test_torsion_restraints.py`.

## Running a set of windows

One window per run. There is no window scheduler: a campaign is a loop in the project's own
script, which is where the choice of centres and spacings belongs.

**One system root per window, over one shared `build/`.** That is not a style preference. `build-md`
writes the window's restraint into `input/umbrella.in` as
`umbrella_file = umbrella.<digest>.yaml`, and `input/` is shared by every run on a system root —
so a second window under the same root is refused:

```text
build-md: input/umbrella.in already exists and is not what this configuration resolves to.
  `input/` is shared by every run on this system: the runs already beside it read THIS file, so
  replacing it would make their inputs describe a different experiment than the one they ran.
```

The refusal is right, and it is exactly one file wide: across three centres, `min/resolved.config`,
`input/min.in` and `input/eq_1.in` come out byte-identical and only `umbrella.in` differs. So give
each window its own root and let them share the built system, which is genuinely the same system:

```bash
# Build once, at the top. Every window reads this one system.
md-openmm build-top -i ALA.pdb --config build.config \
    -os build/built.xml -op build/built.pdb -log build/built.log

for centre in -150 -120 -90 -60 -30; do
  mkdir -p w${centre}
  ln -sfn ../build w${centre}/build          # relative, so the tree stays movable
  cp cv.yaml w${centre}/

  # The window's OWN restraint file, and a configuration that names it. Both, together:
  # a configuration that still points at a shared `umbrella.yaml` biases every window at
  # whatever that file says, whatever the loop variable is.
  cat > w${centre}/umbrella.yaml <<YAML
schema_version: 1
restraints:
  - {cv: phi_ALA, form: harmonic, centre_deg: ${centre}.0, force_constant: 100.0}
YAML
  cp example.config w${centre}/

  ( cd w${centre} && md-openmm build-md -odir ./run1 --config example.config \
      && cd run1 && ./run.sh )
done
```

**Check that the windows differ before running them.** The copied definition is
content-addressed, so one `ls` answers it — five windows must show five digests:

```bash
ls w*/run1/umbrella.*.yaml
```

This is worth a line of script because the failure is invisible in the output. An earlier version
of this page edited the restraint into a file the configuration did not name; every window then
resolved the same definition, every run reported `status: completed`, and the profile was five
copies of one window at −60° with nothing anywhere saying so.

Each run also records its own restraint in `build-md.log` under `umbrella_definition`, including
the resolved atom indices — so a reader of the output does not have to open two files to learn
which four atoms were biased:

```yaml
# umbrella_definition:
#   source_sha256: ac66d59f044beef42fa85280a38b76470780a0f5760ef302e90aeb93a275b2d7
#   copied_as: umbrella.ac66d59f044b.yaml
#   restraints:
#   - cv: phi_ALA
#     form: harmonic
#     centre_deg: -60.0
#     force_constant_kj_mol_rad2: 100.0
#     atom_indices: [4, 6, 8, 14]
```

## Measured behaviour

Alanine dipeptide, ff14SB + GBn2, 2 ps of biased production at k = 500 kJ/mol/rad², reported
every 100 steps:

| CV | restraint | mean | target |
|---|---|---|---|
| `phi_ALA` | harmonic | −63.6° | −60° |
| `psi_ALA` | flat-bottom ±30° | 140.9° | 140° (95% of samples inside the window) |

A harmonic window's mean sits a few degrees off its centre wherever the underlying free energy has
a slope. That is expected at finite force constant, and it is exactly why windows are reweighted
rather than read directly.
