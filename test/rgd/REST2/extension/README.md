# Extension — cyclo-RGDfV (ten replicas)

Request **additional 5 ns REST2 segments** from the committed checkpoints of the parent run.

## This directory is a request, not a run

Continuation writes into the **original** run directory, `../outputs/rest2/<run-name>/`.

Copying checkpoints into this directory and running beside them would create a scientifically
separate sibling run while calling it a continuation — the exchange sequence would restart, the
lifetime statistics would be wrong, and the two halves would not be one trajectory. The repository's
same-directory resume contract exists to prevent exactly that, and `extend.sh` uses the same run
name and output root as the parent.

## Command

```bash
ADDITIONAL_SEGMENTS=1 MD_DEVICES=1,2,3,4 ./extend.sh
```

The same command pattern works a second time; running it twice adds two more segments.

## What a correct extension demonstrates

| property | why it matters |
|---|---|
| binary checkpoint preferred when compatible | best same-environment continuation |
| serialized State fallback is **announced and recorded** | physically valid but generally not bitwise identical for stochastic dynamics, so it must never be silent |
| physical time, steps, frames continue monotonically | a reset index makes the trajectory unusable for any time-resolved analysis |
| exchange attempt indices, phase, RNG state and walker mapping continue | the second segment must continue the exchange sequence, not restart it |
| lifetime vs invocation statistics reported separately | an acceptance rate computed over one invocation is not the run's acceptance rate |
| trajectory and state-data output append without duplicated headers | a duplicated header silently corrupts downstream parsing |
| incompatible changes are refused **before output is opened** | rejecting after opening for append has already damaged the run |

An incompatible tau ladder, enhanced region, omega policy, topology, integrator or nonbonded setting
must be rejected before any file is opened for append.

## Status

**Not executed.** The extension script is generated and its command pattern is the same as the
parent run's, but no extension has been run, because the parent run has not been run. See the
journal for what was and was not executed.
