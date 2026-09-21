# The ladder coordinate record: `ladder_coordinates()` and schema v4

**Design, on paper. Nothing here is implemented**, and nothing lands until the TYK2 campaign has
run (S0's freeze; the ruling is at 0.7.0 `295afe2`). Written now so that when the campaign ends the
decision is made once and quickly, rather than re-argued at the moment everyone wants to land it.

**A correction first, because both S0 and I have been repeating the wrong number.** This has been
called a "v2 -> v3" change in several messages. The storage is ALREADY v3:
`md_tools.remd.storage.SCHEMA_VERSION == "md-tools-replica-exchange/v3"`. What follows is therefore
**v3 -> v4**. The mistake is harmless in conversation and would not have been harmless in a
migration note, since `SUPERSEDED_SCHEMAS` keys on the exact string.

## 1. The problem, precisely

A REST2 rung's coordinate is `tau`, baked into a serialised System. A lambda rung's coordinate is
`lambda`, a set of Context parameter values on ONE System. A 0.7.1 FEP-REST2 rung on the tent path
has **both**. The driver reads `protocol.tau` in seventeen places -- the reporter, the per-state
trajectories, the CV series, `restart.json`, the exchange and lifetime statistics -- and the NetCDF
analysis file carries exactly one variable:

```text
tau(state)  "tau[state] is the REST2 source parameter of that thermodynamic state"
```

Every one of those seventeen readers wants "the coordinate of rung i". `tau` is that only for one
protocol.

## 2. The runtime half: `ladder_coordinates()`

One accessor on the protocol, replacing every `protocol.tau` read:

```python
def ladder_coordinates(self) -> LadderCoordinates: ...
```

returning an object with

```text
names    ("tau",)                      REST2
         ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")        a lambda ladder
         ("tau", "lambda_electrostatics", "lambda_sterics", "lambda_bonded") FEP-REST2
values   values[state][name] -> float, every state, every name, no default
kind     "rest2" | "lambda" | "fep-rest2"
```

Three rules that are not negotiable, each because of a defect this project has already had:

* **No implicit zero.** A name absent from `names` is absent from the record; it is never written
  as 0. A lambda ladder writing `tau = 0` at every rung is indistinguishable from a REST2 ladder
  that never heated (S0's ruling, refusing that option explicitly). The same argument forbids the
  mirror: a REST2 ladder must not write `lambda = 0`, which would read as a ladder sitting at its
  own end state.
* **`kind` is recorded, not inferred.** A reader must not decide what it is holding by asking
  which variables are present, because that makes a FUTURE coordinate's absence indistinguishable
  from this protocol's. The CV contract already works this way: a CV-disabled run records
  `collective_variables: null` explicitly, since omission cannot be told from a manifest predating
  the field.
* **`values` is dense.** Every state carries every name in `names`. A sparse table would let a
  rung silently inherit a neighbour's coordinate, which is the failure the AIS two-probe rows
  avoid by leaving a row EMPTY rather than borrowing.

`protocol.tau` does not survive as an alias. A REST2 protocol keeps it -- it is that protocol's own
field and its scientific meaning is exact -- but nothing in the driver reads it, so there is one
path rather than one path and a shim that agree until somebody edits one.

## 3. The file half: schema v4

```text
dataset.schema         = "md-tools-replica-exchange/v4"
dataset.ladder_kind    = "rest2" | "lambda" | "fep-rest2"      REQUIRED, never inferred
dimension  coordinate  = len(names)
coordinate_name(coordinate)          str    the names, in order
coordinate_value(state, coordinate)  f8     the dense table
```

`tau(state)` is **retained in v4 and written only by a `rest2` ladder**, with its existing meaning
and long_name. Not for the runtime -- which reads `coordinate_value` for every kind -- but because
`cpptraj`-adjacent tooling, the reference bundles and `verify_rungs.py` read `tau` by name, and
dropping it would break readers that are correct today for a gain of one variable. A `lambda` or
`fep-rest2` file does not create it at all; it is absent, not zero.

**What a reader does, in one rule:** read `ladder_kind`; read `coordinate_name`/`coordinate_value`.
Nothing else distinguishes the three cases, and in particular the presence or absence of `tau` does
not.

## 4. What an old file reads as, and what an old reader does with a new one

Both directions, because only one of them is safe by construction.

**A v3 file read by a v4 runtime: REFUSED BY NAME, and migrated deliberately or not at all.**
`validate.py` already fails on `schema != SCHEMA_VERSION` and reports `SUPERSEDED_SCHEMAS[schema]`
when it knows the version -- "identified, explained, and refused, not silently reinterpreted as if
it were current". So v4 adds one entry:

```python
SUPERSEDED_SCHEMAS["md-tools-replica-exchange/v3"] = (
    "v3 carries `tau` alone and no ladder_kind, so a reader cannot tell which coordinate it "
    "names. Every v3 file IS a REST2 ladder -- no other kind could be written by a v3 runtime -- "
    "so migration is exact: ladder_kind = 'rest2', coordinate_name = ['tau'], "
    "coordinate_value[:, 0] = tau[:]. `md-openmm ...` migrates in place, recording the event in "
    "`storage_migrations_json` as the existing migrations do.")
```

That migration is exact **only because of a fact that will stop being true**: today a v3 file can
only be a REST2 ladder, since no runtime could write anything else. The migration must therefore
be written now, while it is provable, and not left for a version in which somebody has to guess.

**A v4 file read by a v3 runtime: the dangerous direction, and NetCDF does not protect us.**
An old reader opening a v4 file finds `tau` missing on a lambda ladder -- an error, which is fine --
but on a `fep-rest2` file it finds `tau` PRESENT and correct, ignores `coordinate_*` entirely, and
produces a perfectly plausible analysis of a ladder whose lambda it never saw. Unknown variables
are silently ignored by every NetCDF reader; that is the format's design.

Two mitigations, and I would take both:

1. v3 runtimes already refuse `schema != "md-tools-replica-exchange/v3"`, so a released v3 runtime
   will refuse a v4 file for the right reason. This is the real protection, and it exists today.
2. A v4 `fep-rest2` file names its tau variable `tau` **and** sets `tau.incomplete_coordinate = 1`
   with a long_name saying the state is not described by tau alone.

**`incomplete_coordinate` IS A LABEL, NOT A MECHANISM, and must stay labelled as one.** It changes
nothing about what any reader does: a program that ignores `ladder_kind` will ignore this too, since
both are attributes and NetCDF hands out neither unless asked. It exists for the person reading
`ncdump` output, and writing it must never be mistaken for having closed the hole. The mechanism is
mitigation 1 -- a released v3 runtime refusing a schema it does not recognise -- and there is no
second one. If this line is ever read as "v4 protects old readers", the failure it invites is the
one with no trace: a correct `tau`, a lambda that was never seen, and a plausible analysis of a
ladder the reader did not understand.

## 5. What this does to the other sixteen readers

Mechanical, and listed so the landing is not a discovery exercise:

| reader | today | v4 |
|---|---|---|
| `reporter.write_ladder(tau)` | writes `tau(state)` | writes the dense table + `ladder_kind` |
| per-state trajectories (`taus=`) | names files, records the rung | takes `LadderCoordinates` |
| CV series (`taus=` in six calls) | sidecar field per state | the rung's full coordinate dict |
| `restart.json` state entries | `{"state_index", "tau", ...}` | `{"state_index", "coordinates": {...}}` |
| exchange / lifetime statistics | `tau=` for labelling | `coordinates=`, label from `names` |

The per-state trajectory NAME is unaffected and must stay unaffected: `state_trajectory_name(i)`
comes from the state INDEX and never from the coordinate, so that two ladders with different
schedules at the same index name the same file. A rung is addressed by its index; this record says
what that rung IS, not where it is filed.

## 6. What is deliberately NOT here

* **No `lambda` scalar.** A lambda ladder has three public components and the diagonal is a
  schedule choice, not a property of the Hamiltonian. Collapsing them to one number would make an
  off-diagonal path unrepresentable and would be discovered by whoever first needs one.
* **No coordinate-derived effective temperature** for a lambda ladder. REST2 reports one for tau
  and calls it "for reporting"; there is no equivalent for lambda, and inventing one would put a
  number in the record that nothing thermostats at.
* **No migration from v4 back to v3.** A v4 lambda file has no v3 meaning.
