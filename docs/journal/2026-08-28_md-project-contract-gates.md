# Fix the contract gates MD-project depends on

| | |
|---|---|
| Date | 2026-08-28 |
| Branch | `fix/md-project-contract-gates` |
| Base | `1a54f45ebfe126e3aab55b97e2a040a97174bf32` (`dev` at the time, unchanged) |
| Reported by | MD-project `dev-test-ala-rgdfv`, `docs/upstream/20260828_md-templates-preflight-and-charge-provenance.md` |
| Scope | five blocking properties. Charge caching deliberately excluded. |

Five safety and provenance properties failed against `1a54f45`, found by running the real
MD-project examples rather than by reading the code. Each was reproduced before being fixed.

---

## What was wrong, and what it cost

### 1. `am1bcc_nagl` could not import its own dependency

```python
# system.py:146
from md_templates.openmm.hashing import sha256_file      # no such module
```

`resolve_nagl_am1bcc_model()` raised `ModuleNotFoundError` for anyone selecting a method the
package advertises in `NAGL_AM1BCC_METHODS`. Every other call site uses
`from .provenance_min import sha256_file`, so this was a single wrong import — but the *route* was
dead, and a consumer spent 26 minutes in `sqm` on the supported route with the documented faster
one unavailable.

**Fixed** by using the package's existing hashing implementation. The two imports that were failing
as one are now separated: `openff.nagl_models` absent is the genuine optional-dependency case and
raises a `RuntimeError` naming the install and the alternative; a broken internal import propagates
as itself. Telling a user to install a package they already have is worse than the original error.

No silent fallback to AmberTools AM1-BCC. A user who asked for NAGL and got Sage AM1-BCC would have
a different Hamiltonian with no configuration change to show for it.

### 2. The executed charge scheme never reached `forcefield.json`

`system.py` records `charge_scheme` — `am1bcc` or `am1bccelf10`, decided at run time by
`_has_openeye()` — and for NAGL the model file and its digest. `_ligand_record` then built a fixed
dictionary that copied none of it, and looked for `nagl_model` where the builder writes
`nagl_model_file`.

Observed in every ligand dataset generated at the base commit:

```text
ligand: charge_method=am1bcc  charge_scheme=None  charge_model=None
```

**Fixed.** The record carries `charge_scheme`, `charge_model` and `charge_model_sha256`, and
`_null_ligand()` declares all of them so a reader can tell absent from unrecorded. The scheme is
deliberately **not** defaulted to the requested method: absent means the builder did not report one,
and substituting the label is how a record comes to claim `am1bcc` for a run whose scheme was never
established. Over a flexible molecule those are different quantities.

### 3, 4. Production never reached the shared preflight

`preflight.check_parent` and `check_own_completion` were correct. They were never called for
production, because of `run()`'s guard:

```python
if stage is not None:
    results += check_parent(here, stage, dynamics=dynamics)
    results += check_own_completion(here, stage)
```

and production stages had **no `stage.yaml` at all** to pass. `stage_run.py` passes `stage=STAGE`,
which is exactly why the equilibration chain was protected and cMD, REST2 and REST2 equilibration
were not. A missing parent surfaced only in the run path, after `simulation` — and therefore its
Context — existed; a changed production request was never checked.

**Fixed** by giving production stages the same thing the common chain has. `md-gen` writes
`stage.yaml` from `production_stage_document`, ONE derivation that the generated launcher reuses,
loaded from the same file `md-gen` copies into the project. Deriving it twice is how a run comes to
write one hash and a check to compare another.

AIS is excluded: it does not start from the common chain and resolves its own request in
`path_definition.yaml`. REST2 declares no `output_state`, because it keeps a state per replica and
naming a file that never appears would report it as never having run.

### 5. Runtime configuration bypassed the timestep/HMR invariant

`md-gen` refuses a timestep above ~3 fs without hydrogen mass repartitioning. Generated launchers
reread `md.config.yaml` at run time — which is what makes extension work — and nothing re-applied
that invariant. Editing the file to 4 fs and raising the duration appended **500 steps at 4 fs to a
trajectory produced at 2 fs**, under a system with `HBonds` and no HMR.

This was the only one of the five that altered existing scientific output.

**Fixed** by `check_runtime_request`, which recomputes the request from the configuration on disk
now and compares it with what `stage.yaml` recorded, naming the field:

```text
[FAIL] runtime request  md.config.yaml no longer describes the stage this directory was
generated for (timestep_fs: 2.0 -> 4.0). Continuing would apply the new setting to output
produced under the old one ... raising the requested length alone is a legitimate extension
and is allowed.
```

---

## The design decision worth arguing with

**Two fingerprints, because production extends and common stages do not.**

A common stage is fixed at generation and refuses to rerun, so hashing its whole document is right
and `stage_config_sha256` already did that. Running longer is the documented way to extend an active
dataset, and it changes the requested length and nothing else. Hashing the whole document for
production would have forbidden **every extension** in order to catch one unsafe edit — a fix worse
than the defect.

So `stage_invariant_sha256` removes `PRODUCTION_EXTENDABLE_FIELDS` and reuses the same
canonicalisation. It covers timestep, temperature, ensemble, tau, seeds, solvent mode, hydrogen
mass and the parent handoff. A changed invariant means the checkpoint on disk was produced under
different physics, and continuing from it would splice two Hamiltonians into one trajectory.

Both are recorded in `resolved_stage.yaml` and they answer different questions: the full one
identifies the exact request that produced these outputs, the invariant one is what a continuation
is checked against.

`check_own_completion` branches for production into consistency **only**. Whether a stage finished,
and whether it should run again, is the launcher's own resume logic against `resolved_run.yaml`.
Two tests hold this line: one proves a 0.002 → 0.004 ns extension still resumes and completes, the
other proves a timestep change is refused with the tree byte-identical afterwards.

---

## Tests

```console
$ pytest tests/test_md_project_contract_gates.py -q
24 passed in 6.82s

$ pytest tests/test_md_data_contract.py tests/test_template_provenance.py \
      tests/test_fair_provenance.py tests/test_integrity_corrections.py \
      tests/test_installer_capability.py -q
172 passed in 82.09s            # unchanged from the base commit

$ pytest -q -m "not gpu"
351 passed, 71 deselected in 123.96s
```

**71 GPU tests were deselected, not passed.** They need a CUDA platform under the `gpu` marker and
were not run in this pass; the behavioural refusal tests above build a real system and do run.

The 24 new tests all fail at `1a54f45`. Two exist to stop the fix being undone in the obvious ways:
one asserts a longer run is still allowed, the other that preflight stays bounded — no `rglob`,
`os.walk`, `glob.glob`, `iterdir` or `.dcd` — because the new check reads configuration and could
have grown into a walk.

---

## Not in this change

**Charge caching.** The upstream report proposes caching AM1-BCC work keyed on canonical SMILES and
conformer scheme. It changes behaviour and provenance beyond the blocking defects and belongs in its
own review.

**Existing datasets.** Datasets generated at `1a54f45` keep that commit in their `templates.commit`.
They were generated by that code and rewriting the record would be false provenance rather than
imprecise provenance.
