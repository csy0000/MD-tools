# 2026-08-27 — preserving the unevaluated MD-data capability state

Instruction: `claudecode-instructions/20260827_preserve-unevaluated-capability-state.md`.
Branch `dev`. No merge to `main`, no tag. **No scientific runtime file changed. No GPU test, MD
run, AIS execution, REST2, environment creation, pip install or network probe was performed.**

One expression, one function.

## The bug

`capability_summary()` computed:

```python
"md_data_contract_support_ready": bool(md_data.get("contract_support_ready")),
```

`bool(None) is False`. A dry run, whose canonical record correctly says
`contract_support_ready: null`, therefore produced a capability summary saying contract support was
**unavailable**. The record and the summary derived from it contradicted each other:

```yaml
md_data:
  contract_support_ready: null     # not evaluated
capabilities:
  md_data_contract_support_ready: false   # evaluated, and unavailable
```

The function's own docstring said "the three states". The implementation had two.

## Why the CLI looked right

`_report_md_data()` tests `md_data.get("attempted") is False` *before* it reads readiness, so the
dry-run path printed `not evaluated (dry run)` and never consulted the wrong value. And a dry run
is never persisted, so no `machine.yaml` recorded the contradiction and no simulation result was
affected.

That is the whole reason this was worth fixing rather than shrugging at: the output was right by a
coincidence of ordering, and a caller reading `result["capabilities"]` on its own — which is what a
structured return value is *for* — got the wrong answer. It is the fourth in this sequence of
defects where something looked verified because the path that happened to be exercised avoided it.

My own test asserted the contradiction, on the line directly below the one asserting `None`:

```python
assert result["md_data"]["contract_support_ready"] is None
assert result["capabilities"]["md_data_contract_support_ready"] is False   # wrong
```

## The correction

Readiness is passed through rather than coerced, and a value outside the contract is refused:

```python
readiness = md_data.get("contract_support_ready")
if not any(readiness is state for state in (True, False, None)):
    raise ValueError(...)
```

Identity, not membership: `1 == True` in Python, so `readiness not in (True, False, None)` would
let `1` and `0` through and store them as themselves. A record that says `1` instead of `true` is a
record someone has to interpret.

`warnings_for()` had the same tri-state mistake — `not capabilities.get(...)` is true for `None` —
and now tests `is False` explicitly. "Not evaluated" is not a failure, and warning about it reports
a problem nobody looked for.

| record | capability | warning | meaning |
|---|---|---|---|
| `True` | `True` | no | ready |
| `False` | `False` | yes, with reasons | evaluated, unavailable |
| `None` | `None` | no | not evaluated |

Dry-run reasons still explain that readiness was not evaluated, and are never relabelled as reasons
it is unavailable.

## Evidence

### Before and after, dry run

```text
before   md_data.contract_support_ready       None
         capabilities.md_data_..._ready       False      <- contradiction

after    md_data.contract_support_ready       None
         capabilities.md_data_..._ready       None
```

### All three states

```text
  record True  -> capability True   warning: no
  record False -> capability False  warning: yes
  record None  -> capability None   warning: no
  bad value refused: contract_support_ready must be True, False or None, not 'maybe'
```

### The real dry-run command

```text
$ md-template install -e openmm -ev 8.6.0 --dry-run --target-dir <stack>
  environment   : <stack>/envs/openmm-8.6.0
  log           : <stack>/logs/install-openmm-20260827T191602Z.log
  dry run: nothing was installed
  md-data contract: not evaluated (dry run)

'not evaluated (dry run)': True | 'ready': False | 'UNAVAILABLE': False
```

`subprocess.run` was patched to raise for the whole run and was never called: no pip, no
environment, no import, no network.

An evaluated result still reports its outcome and its reasons — `ready` for `True`,
`UNAVAILABLE` plus every reason for `False`, and `not evaluated` for neither.

### Tests

```text
pytest tests/test_installer_capability.py -q -p no:randomly                33 passed
pytest tests/test_installer_capability.py tests/test_cli.py -q -p no:randomly
                                                                           37 passed
pytest tests/ -q -p no:randomly -m "not gpu"                              327 passed, 71 deselected
```

### Scope

Changed: `src/md_tools/install/openmm.py` (+27) and `tests/test_installer_capability.py`
(+90), plus the index and this journal. A grep of the diff for `ais_run`, `rest2`, `sysgen`,
`mdgen`, `stage_run`, `md_stages`, `system.py`, `solvation`, `forcefield_record`, `implicit`,
`preflight`, `openmm/config.py`, `openmm/defaults.py`, `md_data_contract` and `provenance_min`
returns nothing. No MD equation, force field, integration schedule, REST2 exchange, AIS work
accumulation, provenance contract or command was touched.

## Remaining external limitation

Unchanged and not probed for this task: `csy0000/MD-data` is private and `md-data` is not
published, so anonymous installation of the pinned validator remains impossible. That result stands
from the earlier journal; no network operation was performed here.
