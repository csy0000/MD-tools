# Preserve the unevaluated MD-data capability state

## Purpose

Correct one semantic inconsistency left after
`20260827_dry-run-and-validation-shape-finalization.md`.

The canonical dry-run MD-data record correctly says:

    contract_support_ready: null

But `capability_summary()` currently computes:

    "md_data_contract_support_ready": bool(md_data.get("contract_support_ready"))

In Python, `bool(None)` is `False`. This silently changes the meaning from “readiness was not
evaluated” to “contract support is unavailable”.

The CLI currently hides this inconsistency by checking `md_data.attempted is False` first, and a
dry run is not persisted to `machine.yaml`, so no simulation result is wrong. Nevertheless, the
structured return value contradicts its own canonical record and the documented three-state model.
A caller using only `result["capabilities"]` would receive the wrong answer.

This is a one-function semantic correction plus focused unit tests. Do not redesign the installer,
the capability schema, the MD-data contract, provenance, system generation, MD execution, cMD,
REST2, AIS, or the six-command architecture.

Work on the current `dev` branch of `csy0000/MD-templates`. The inspected starting head was
`a72d6c693cd5c0eb890c349b5aa516e4f42a2ab7`.

Read before editing:

- `CLAUDE.md`
- `claudecode-instructions/20260827_dry-run-and-validation-shape-finalization.md`
- `docs/journal/2026-08-27_dry-run-and-validation-shape-finalization.md`
- `src/md_templates/install/openmm.py`
- `src/md_templates/cli/md_template.py`
- `tests/test_installer_capability.py`

Write the report to:

    docs/journal/2026-08-27_preserve-unevaluated-capability-state.md

Commit and push the correction to `dev`. Do not merge into `main` and do not create or move a
release tag.

## Required correction

Change `capability_summary()` so it preserves all three readiness states instead of coercing them
through `bool()`:

| MD-data record value | Capability value | Meaning |
|---|---|---|
| `True` | `True` | contract support is ready |
| `False` | `False` | contract support was evaluated and is unavailable |
| `None` | `None` | contract support was not evaluated |

A suitable implementation is conceptually:

    readiness = md_data.get("contract_support_ready")
    if readiness not in (True, False, None):
        # normalize or reject deliberately; do not silently apply Python truthiness
    return {
        "openmm_runtime_ready": True,
        "md_data_contract_support_ready": readiness,
        "md_data_unavailable_reasons": ...,
    }

Use the simplest implementation consistent with the existing data contract. Do not add a new enum,
class, schema framework, command, or compatibility layer.

Keep these invariants:

- a dry run returns `md_data.contract_support_ready: null` and
  `capabilities.md_data_contract_support_ready: null`;
- a verified ready installation or validation returns `True` in both places;
- a verified unavailable installation or validation returns `False` in both places;
- the real dry-run CLI still prints only `not evaluated (dry run)`;
- it prints neither `ready` nor `UNAVAILABLE`;
- unavailable real installation/validation still prints its reasons;
- dry-run reasons may explain that readiness was not evaluated, but must not be relabelled as
  unavailability reasons.

Review `warnings_for()` for the same tri-state mistake. It must emit the unavailable-contract
warning only when readiness is explicitly `False`, not when it is `None`. Do not change unrelated
warnings.

## Tests

Add or adjust focused tests proving:

1. `capability_summary({"contract_support_ready": None, ...})` preserves `None`;
2. `True` remains `True`;
3. `False` remains `False`;
4. the real `md-template install --dry-run` result has `None` in both the canonical record and
   the capability summary;
5. its CLI output says `not evaluated (dry run)` and contains neither `ready` nor
   `UNAVAILABLE`;
6. `warnings_for()` emits no MD-data-unavailable warning for readiness `None`;
7. `warnings_for()` still emits the warning and reasons for readiness `False`.

Run only the focused installer-capability and CLI tests, followed by the normal non-GPU unit suite
once. Do not run GPU tests, AIS, MD, REST2, environment creation, pip installation, or network
probes.

## Documentation and index

Document concisely:

- the exact expression that caused the bug: `bool(None) is False`;
- why the CLI looked correct despite the structured result being wrong;
- the corrected three-state examples;
- exact tests and results;
- confirmation that no scientific runtime changed.

Add this instruction exactly once to `claudecode-instructions/README.md` as pending. When complete,
replace that same entry with one executed entry. Do not create a duplicate.

## Required report

Do not return PASS until the report includes:

- pushed commit SHA;
- exact files changed;
- exact test commands and pass counts;
- before/after dry-run structured values;
- ready, unavailable and unevaluated capability test evidence;
- dry-run CLI output;
- warning behavior for `False` versus `None`;
- confirmation that no GPU, MD, installation, or network operation ran;
- confirmation that no scientific runtime file changed.

Continue through routine implementation choices without stopping for questions. Inspect the final
diff for scope creep, commit, and push to `dev`.
