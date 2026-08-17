# Architecture

Three layers, each of which can be understood without the one below it.

```
    md-templates                 md-openmm
         │                           │
         ▼                           │
  md_templates.cli                   │      generic commands: list, inspect, identity,
         │                           │      and prepare/run/resume by dispatch
         ▼                           │
  md_templates.core                  │      ENGINE-NEUTRAL. Imports YAML and pydantic and
    ├── config/     typed model,     │      nothing heavier, at any level.
    │               resolution,      │
    │               canonical bytes, │
    │               hash projections │
    ├── bundle      file format      │
    ├── persistence layout, atomic   │
    │               writes, commits  │
    ├── registry    the catalog      │
    ├── template    descriptors      │
    ├── identity    SHA + path       │
    ├── packaged    catalog in a     │
    │               distribution     │
    └── dispatch    (method, engine) │
                    -> provider      │
         │                           │
         ▼                           ▼
  md_templates.engines.openmm               THE implementation. One copy.
    ├── schemas, config, adapter            legacy manifests; runtime configuration
    ├── system, solvation, equilibration    force fields, building, staged equilibration
    ├── methods/md, methods/rest2           the two methods
    ├── bundle, bundlecheck, bundleinfo     preparing and validating bundles
    ├── restart, platform, provenance       Simulation I/O, device selection, run dirs
    ├── runner, cli, provider               orchestration, `md-openmm`, generic dispatch
    └── manifests/                          shipped system and experiment manifests

  md_templates.openmm                       COMPATIBILITY ONLY. Aliases, no behaviour.
```

## The boundary that matters

**Core must be importable where no engine is installed.** Not "should avoid" — must. Listing
templates, inspecting a descriptor, resolving an identity and validating a configuration all work on
a laptop with no OpenMM, no OpenFF and no RDKit, and a test proves it by making those packages
*unimportable* in a subprocess and then doing real work.

That is what lets a catalog command run on a machine that could never run a simulation, and it is why
`core` never imports `engines` — in either direction, at any level, deferred or not. A test walks the
AST of every core module to enforce it, because the violation that actually happened was a deferred
import inside a function, invisible to an import-time check.

The complement is equally deliberate: engine-specific work stays in the engine even when it looks
like plumbing. `save_restart` takes a running `Simulation`; `topology_counts` needs a built `System`
to know that a virtual site is not a topology atom. Those live with the provider, and core owns only
the layout and the bookkeeping around them.

## Compatibility

`md_templates.openmm.*` and `md-openmm` are published interfaces and are not going away. Every
historical import path resolves to the **same module object** as its new home — aliased in
`sys.modules`, not re-exported into a copy. A copying shim would let `monkeypatch.setattr` land on a
shadow while the code under test reads the original: the test passes and tests nothing.

`tests/test_engine_provider.py` holds the compatibility matrix as data, and asserts identity (`is`)
for all 26 entries.

## Templates

A template is a directory:

```
templates/<method>/<engine>/<variant>/
    template.yaml     the descriptor: method, engine, routes, capabilities, status, provider
    profiles/         its versioned default profiles -- the scientific defaults it owns
```

The tracked directory is the only editable copy. The build stages it into the distribution, so an
installed wheel carries the same bytes; nothing is duplicated in source. Profile discovery is rooted
at the **package's own location** — the packaged catalog if there is one, otherwise the repository the
package is imported from — and never at the working directory, so a wheel-installed run resolves the
same profiles from any directory.

## Dispatch

`md_templates.core.dispatch` is a table from `(method, engine)` to a provider module. No entry-point
scanning, no plugin protocol, no import-time side effects. The provider translates a generic action
into exactly one engine command and passes the user's arguments through unchanged — which is what
makes "the generic and legacy routes are equivalent" a checkable claim rather than an intention.

Adding an engine means adding a line to a table and writing a provider. It does not mean satisfying
a framework.

## What is not here

Template identity is **not** written into bundles or run manifests, and no template metadata enters
any scientific or continuity hash. The catalog describes; it does not stamp.
