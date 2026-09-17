# rREST2 — archived

rREST2 is REST2 whose top rung is periodically refreshed from a pre-generated Boltzmann
reservoir of phase-space samples. It is **archived as of md-tools 0.5.4** and is no longer a
protocol md-tools runs.

## Why it was archived

0.5.4 moved every scaled Hamiltonian into files built once by
`md-openmm build-top --rest2-scaler`, and made a ladder read its states only from its group file.
The reservoir refresh would have had to be re-derived on top of that: its source is a fixed-τ
phase-space stream whose Hamiltonian identity must match the top rung exactly, and every piece of
that contract (the declaration, the identity check, the velocity policy, the CV pre-refresh
snapshot, the refresh columns in the ladder NetCDF) was written against run-time scaling. The
supported methods for 0.5.4 are cMD, fixed-τ cMD, REST2, AIS and umbrella sampling, and rREST2 was
withdrawn rather than carried forward untested.

## What refuses it now

* `protocol: rREST2` in a `build-md` configuration, and any `reservoir:` section — refused by
  name, pointing here.
* `protocol = rREST2` and the `&remd` keys `reservoir_enabled`, `reservoir_path`,
  `refresh_interval_exchanges`, `reservoir_velocities` in an `.in` file — the same refusal.
* `--reservoir` is no longer an executor flag, and the exchange-rule interface is
  `md-tools-exchange-rule/v2`: its context has no `reservoir` and its outcome no
  `reservoir_refresh`.
* `export-reference` still refuses a ladder record that ran with a reservoir.

The ladder NetCDF keeps its `reservoir_*` columns, always written as −1, so the v2 file format
does not change. `data-register` still recognises an existing `rREST2` run directory: registering
describes finished data and runs nothing.

## What is here

```text
src/md_tools/remd/reservoir.py        the refresh rule, the prepared reservoir, the source checks
src/md_tools/remd/rrest2_exchange.py  the `--exchange-rule` plug-in a generated project used
configs/md/rREST2.config              the shipped example configuration
docs/rREST2/                          the method page, example.config and example.in
tests/                                the tests that were only about rREST2 or the reservoir
```

Nothing under `archive/` is packaged, imported by md-tools, or collected by pytest (`testpaths`
is `tests/`). The files are kept as they were at the last working commit and will not run against
the current tree: the driver branches that called them were removed.

## Getting it back

The git tag **`rREST2-final`** is the last commit where rREST2 is a working protocol, with its
driver branches, preflight, configuration schema and tests all in place:

```bash
git checkout rREST2-final
```

Restoring it on a newer tree is a port, not a revert: bring back the refresh as one design — the
rule, the driver's `_apply_reservoir` and CV pre-refresh snapshot, the preflight's source
validation and the schema section — built against saved scaled states. A partial return is a
reservoir that opens and never draws, which is the failure rREST2 already had once.
