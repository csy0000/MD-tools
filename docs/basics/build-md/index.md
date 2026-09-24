# build-md

`build-md` turns one protocol configuration into a **run directory**: the entry-point scripts, the
Amber-like `.in` files, the resolved configuration and `run.sh`. It generates; it does not simulate.

```bash
md-openmm build-md -odir cMD-run1 --config cMD.config
```

## What it needs first

A **built System**. `build-md` validates the whole stage chain as it generates it — lengths,
reporting cadences, the ensemble of each stage, the platform policy — and that validation reads
`build/built.xml`, so the system has to exist before the run can be generated.

For a hot REST2 ladder it also needs the **saved scaled states**: every rung a ladder integrates is
a file written by `build-top --rest2-scaler` before the run is generated, never derived at run
time. `build-md` refuses until `build/<method>/scaler.yaml` exists, and names the command that
produces it.

## What it writes

```text
<method>-run<N>/
    <method>.py         the production entry point: one import, one call
    eq/                 the equilibration entry points, filed by position
    run.sh              the launcher, with the right rank count for the protocol
    resolved.config     the authoritative resolved declaration
    run.config          the configuration as you wrote it
    build-md.log        the build record
```

Plus, for a ladder, `remd_groupfile.<segment>`, `remd_records/` and a `rank/` tree.

`min/` and `input/` are written as **siblings** of the run directory, not inside it: they belong to
the system rather than to the run, and two runs on one system share them. See
[the run layout](../run-layout.md).

## Generated scripts are entry points

A generated `.py` is three lines — an import of the stable API and one call. No function
definitions, no `argparse`, no OpenMM import, no reporter or restraint constructed there:

```python
#!/usr/bin/env python
from md_tools.md import run_generated_stage
raise SystemExit(run_generated_stage(__file__, "min"))
```

Everything the run needs is in `resolved.config` beside the script, located from `__file__` so the
directory can be moved, re-validated by the strict resolver at execution, and bound into the
checkpoint fingerprint. That is what stops a generated directory becoming a private fork of the
implementation.

## The configuration

One file, YAML despite the `.config` suffix, with **unknown keys refused by name** — a misspelled
key is an error rather than silent metadata. Lengths are integer step counts everywhere; a schedule
that would have to be rounded is refused with the arithmetic that would fix it.

Every key, with its default and which protocol reads it:
[the configuration reference](configuration.md).

## Round-tripping

Every `.in` file `build-md` writes resolves back to exactly the `resolved.config` beside it. That
round trip is a test, not a convention — it is what makes the two descriptions of a run one
description.
