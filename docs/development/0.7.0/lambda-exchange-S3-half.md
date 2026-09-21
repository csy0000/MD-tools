# Replica exchange over lambda (A3b): the Hamiltonian's half

**S3's half of a joint recommendation with S4.** S0 asked for one recommendation, not two; this is
the part that belongs to the Hamiltonian, written so S4 can fold it into the execution design. It
answers S0's four questions in order. Nothing here is implemented and nothing is authorised.

## 1. What an exchange attempt needs, and what it costs

**It needs nothing beyond `energy(context, state)`.** An attempt between neighbouring windows `j`
and `j+1` needs the four reduced potentials `u_j(x_j)`, `u_j(x_{j+1})`, `u_{j+1}(x_j)`,
`u_{j+1}(x_{j+1})`. Each rank already holds the two at its own configuration:

```python
here  = h.energy(context, state_j)        # the rank's own lambda
there = h.energy(context, state_j_plus_1) # the neighbour's, same coordinates
h.set_state(context, state_j)             # restored; energy() leaves the state it was given
```

`set_state` sets Context parameters and nothing else -- no `reinitialize()`, no second System, no
second Context, no coordinate copy. That is the whole reason a lambda ladder is cheaper than a tau
ladder, and it is question 4's asymmetry in one line.

Three properties of this that the design should lean on, and one it must not:

* `energy()` is defined to `set_state` first, so a caller cannot read an energy at a state the
  Context is not actually in. The two evaluations above cannot silently swap.
* `context_parameters()` is the ONE definition of a state, public and derived parameters together.
  A state recorded in an exchange log should be exactly that dict, not the three public numbers,
  because the derived ones (`sqrt(1-lambda_e)` and the weights) are what the forces actually read.
* `derivative_components()` is NOT needed for an exchange. It is TI's consumer. An exchange needs
  energies at two states; a derivative at one state is a different quantity and pairing them would
  be the two-probe confusion AIS already has an invariant about.

**Cost, measured, with its own limits stated** (`scripts/s3_exchange_cost_probe.py`, CPU, the
102-particle fixture, median of 15):

| | ms | in steps |
|---|---|---|
| one MD step (hybrid) | 36.3 | 1.00 |
| energy at the CURRENT lambda | 39.4 | 1.09 |
| energy at a NEIGHBOUR lambda + restore | 42.6 | 1.17 |
| ... electrostatics only (dispersion carrier untouched) | 42.1 | 1.16 |
| one attempt = two cross evaluations | 85.3 | 2.35 |

**The qualitative finding transfers; the ratio does not, and I would rather say so than let a
cadence be chosen from it.** A neighbour evaluation costs what a same-lambda one costs, and an
electrostatics-only move costs what a full move costs, so nothing is REBUILT on a parameter change
-- in particular the dispersion carrier's quadrature does not dominate. But the plain end-state
System A also measures 17.7 ms per step on this fixture, and 102 particles do not need 17 ms of
arithmetic: both numbers are dominated by fixed per-call overhead, so "1.17 steps" is mostly two
overheads being equal. What a cadence needs is one measurement on a card with a campaign-sized
system. The hybrid costs 2.0x a plain end state per step here (11 forces against 4), which is the
one ratio I would carry forward as an order of magnitude.

## 2. Does the recorded identity distinguish two lambda states of one System?

**Not by itself, and it should not be made to. Add a per-rung state record instead.**

`h.record` identifies the HAMILTONIAN: both end states by sha256, the topology plan's digests, the
softcore settings including `sc_boundary_14`, kappa and the PME grid, the force-group map, the
public parameter names. Every window of a lambda ladder shares all of it, by construction -- that
is the point. So the record answers "are these rungs the same experiment?" exactly, and "which rung
is this?" not at all.

The clean split, which matches how the package already treats REST2:

* **Ladder-level (one record, every rung must agree):** `h.record`. This is the direct analogue of
  a REST2 ladder requiring one `scaler.yaml` whose taus equal the ladder's. A rung whose
  `sc_boundary_14`, PME grid or end-state digests differ is a different experiment and must be
  refused before output, exactly as a mismatched `scaler.yaml` is.
* **Rung-level (one per state):** `AlchemicalHamiltonian.context_parameters(state)` plus the state
  index. That dict IS the rung's identity, it is already the one definition, and it is what the
  forces read.

**No lambda-aware form of the identity is needed, and inventing one would be a defect**: it would
create a second authority on what a state is, and the existing one is used by `set_state`,
`energy`, `energy_components` and `derivative_components` alike. The scientific invariant that a
scaled Hamiltonian is never re-derived at run time is satisfied differently here and genuinely: a
lambda window re-derives nothing because there is nothing to derive -- one System, built once,
read at recorded parameter values.

One caveat I would write into the ladder's preflight: the per-rung record must be compared against
the lambda the rung actually runs, not the lambda a config file requested. A rung whose recorded
state and whose Context parameters disagree is the lambda analogue of "tau 0 on a hot state", and
the existing tau rule shows the shape of the refusal.

## 3. Can A3b and 0.7.1's tent path share one implementation?

**Yes, and A3b should be written as the `tau = 0` case of the 0.7.1 ladder rather than as its own
mechanism** -- but only if one thing is settled now.

The 0.7.1 design (`docs/development/0.7.1/rest2-alchemical-composition.md`, section 3) is a 1-D
ladder along a path through `(lambda, tau)` with `tau = 0` at both ends, nearest-neighbour swaps in
the state index `j`, CONFIGURATIONS exchanged rather than states. A3b is that ladder with `tau_j =
0` for every `j`. The exchange topology, the acceptance test, the per-state trajectory contract and
the CV conventions are identical; only the per-rung Hamiltonian differs.

**The thing to settle now: a rung must be addressed by its INDEX `j`, with `(lambda_j, tau_j)` as
its content.** If A3b is written with `lambda` as the rung's identity -- the obvious shortcut when
`tau` is absent -- then 0.7.1 has to either re-index every rung or carry two addressing schemes,
and the second is how a walker ends up filed under the wrong state. This costs nothing today: it is
one field in a record and one argument in a signature.

With that, doing A3b first CONSTRAINS 0.7.1 helpfully rather than badly: it lands the exchange
machinery, the per-state trajectories and the estimator plumbing against a Hamiltonian that is
already validated, leaving 0.7.1 to add only the composed Hamiltonian and the tau half of the
identity. The reverse order would mean debugging a new exchange layer and a new Hamiltonian at
once.

## 4. The asymmetry, stated so it cannot be flattened

`tau` and `lambda` are both ladder coordinates and they are not the same kind of thing.

| | `tau` (REST2) | `lambda` (alchemical) |
|---|---|---|
| where it lives | BAKED into a serialised System | a Context PARAMETER of one System |
| how a rung is made | `build-top --rest2-scaler` writes `system_state<i>.xml` before the run | nothing is written; the state is a dict of parameter values |
| what an exchange across it needs | the neighbour's SYSTEM to evaluate the neighbour's potential | the same System at other parameter values |
| cost of evaluating a neighbour | a second System in memory, or a second Context | `setParameter` calls and one energy |
| what the record must pin | `scaler.yaml`: source digest, every state's tau and sha256 | the shared `h.record`, plus each rung's parameter dict |
| re-derivation at run time | FORBIDDEN, and impossible: the file is the Hamiltonian | not applicable: there is nothing to derive |
| what a mismatch looks like | a rung integrating a System that is not the tau it claims | a rung running a lambda its record does not name |

**The operational consequence for A3b: a lambda ladder needs no group file.** The REST2 ladder
reads `-s` only from `remd_groupfile.<segment>`, one saved state per line, because each rung is a
different file and the mapping from rung to file is the thing most worth making explicit. A lambda
ladder has ONE System for every rung, so a per-rung `-s` would be the same path repeated K times --
and a line that can only ever hold one value is a line that will eventually hold a wrong one
without anybody noticing. The rung's lambda belongs in the resolved configuration, not in a group
file column.

**And the consequence for 0.7.1: the tent path needs BOTH mechanisms at once**, since it moves in
tau and lambda together. That is the real reason to keep them distinguished in the design rather
than calling them both "the ladder coordinate": on the tent path, rung `j` and rung `j+1` differ by
a Context parameter AND by a serialised System, and only the second requires the neighbour's file.

## Open, for S4 and me to close together

1. **Who evaluates the neighbour's potential?** Section 3 of the 0.7.1 note prefers exchanging
   CONFIGURATIONS (preserving the per-state trajectory contract). For lambda that choice is nearly
   free either way; for the tent path it is not. One answer for both, chosen now.
2. **Cadence.** Needs the card measurement above, on a campaign-sized system, before a number goes
   into any config.
3. **MBAR's `u_k(x_n)` at every state**, not only neighbours: with lambda a parameter, a rank can
   produce a whole row by looping `set_state` over every state at one configuration. Cheap here,
   expensive on the tent path. S4's estimator layer should say what it wants stored.
4. **The CV contract carries over unchanged**, and I think it must: pre-exchange rows, one series
   per STATE, step 0 and the final step exactly once. Worth S4 confirming against `md_tools.remd`
   rather than assuming.
