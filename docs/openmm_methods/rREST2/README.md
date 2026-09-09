# rREST2 — REST2 with a Boltzmann reservoir

REST2 whose **top rung is refreshed from a pre-generated equilibrium reservoir** instead of being
propagated. It is a REST2 ladder plus a different transition rule, not a separate sampling method,
and it reuses the REST2 block for everything about the ladder itself.

Read [the REST2 page](../REST2/README.md) first. Everything there — the scaling, the NVT runtime,
one trajectory per state, the four schedules, restart and extension — applies unchanged.


## The two example files beside this README

| file | what it is |
|---|---|
| [`example.config`](example.config) | what you hand to `md-openmm build-md` |
| [`example.in`](example.in) | what `md-run` then reads — the ladder's production stage, with the reservoir block |

`build-md` generates the `.in` from the `.config`; you do not normally write one by
hand. It is shipped here because it is the file the run actually reads, and because
every setting in it carries the schema's own description as a comment — so the meaning
of a key can be looked up where it is used rather than in the source.

`resolved.config`, written beside the `.in` at run time, stays AUTHORITATIVE: the `.in`
is resolved into it, and that resolved document is what the run reads.

Both are checked by `tests/test_method_example_inputs.py`, which regenerates the `.in`
from the `.config` and fails if they have drifted — an example that no longer matches
the engine is worse than none.

## What the reservoir changes

The hottest rung's configurations are drawn from a reservoir that was generated *separately, at
that rung's Hamiltonian*, rather than being produced by propagating the ladder. The rung therefore
stops being a bottleneck: the cost of decorrelating the hot state is paid once, up front, and the
ladder's job is only to carry those configurations down to τ = 0.

## Ensemble

NVT, exactly as REST2. The reservoir is an **NVT** reservoir; an NPT one is refused.

## Phase space, not configuration

A reservoir stores **positions, velocities and boxes**. A DCD cannot store velocities, so a DCD is
not a phase-space reservoir and is refused as a source — there is no silent fallback. Velocities
that are missing, wrongly shaped, non-finite or identically zero are a hard error when the source is
opened.

**Velocity policy.** `stored` (the default here, written as `velocities: inherit`) installs the
recorded momentum unchanged, which is what the probability-one acceptance assumes. `maxwell`
(`velocities: resample`) uses the source positions and box and deliberately redraws the momenta at
the one common temperature from a seed recorded per refresh, so any single draw is reproducible from
the storage alone. Under MPI only the owning rank draws, and the array is shared, so one process and
N ranks install identical momenta.

The reservoir is materialised **without replacement**. A repeated draw would give one configuration
extra statistical weight and make `frames` overstate the effective reservoir size.

## One Hamiltonian, recomputed on both sides

The reservoir and the rung it refreshes must be the **same complete Hamiltonian**. The comparison is
a SHA-256 over the canonically serialised OpenMM System plus a digest of the enhanced-region
selection, and the current side is always **recomputed** — two stored claims are never compared with
each other.

τ and temperature agreeing is necessary and not sufficient: ff14SB/TIP3P and ff19SB/OPC agree on
both and are different Hamiltonians. The system compared is the **top rung's scaled** system, not
the unscaled reference the ladder is built from; the reference is recorded with `tau: null`, which
is a different claim from a rung at `tau: 0.0`.

## Required inputs

Everything REST2 needs, plus a phase-space reservoir generated at the ladder's top rung:

```bash
# 1. generate the reservoir: fixed-tau cMD at tau = tau_max, streaming phase space
md-openmm build-md -odir ./hot/ --config hot.config      # dynamics.tau: 0.5
                                                          # dynamics.phase_space_printout: 500
cd hot && ./run.sh
```

Then point `reservoir.path` at the phase-space file that run produced. Generating the reservoir
*with* the ladder means one input, one force field, one solvation model and one equilibrated box are
shared by construction rather than by two projects happening to agree.

## Minimal sequence

```bash
md-openmm build-md -odir ./md_script/ --config example.config
cd md_script && ./run.sh
```

## Generated files

As REST2, plus `reservoir.yaml` (the resolved declaration — frame count and time window are read
**from the reservoir file**, never restated in configuration) and a `reservoir/` working directory.
`rREST2.py` replaces `REST2.py`.

## Configuration fields that matter

| field | default | why you would change it |
|---|---|---|
| `reservoir.enabled` | false | true selects the reservoir transition rule |
| `reservoir.path` | null | the phase-space file. Required when enabled |
| `reservoir.refresh_interval_exchanges` | 1 | how often the top rung is refreshed |
| `reservoir.velocities` | `resample` | `inherit` installs the stored momenta; `resample` redraws them |

## Limitations

* The reservoir must be generated at exactly the top rung's Hamiltonian. This is checked, not
  assumed — but generating it is your job.
* A refresh is **not** an exchange and is never counted as one in the acceptance statistics.
* Reservoir size bounds the run: the draw is without replacement, so a long ladder needs a large
  reservoir.
* The generic reservoir rule is exposed as `md_tools.remd.reservoir.ReservoirRefreshRule` so future
  reservoir-REMD work can reuse it; only the rREST2 composition is validated today.

## References

REST2 [@wang2011rest2]; see also [the scientific defaults](../../scientific-defaults.md).
