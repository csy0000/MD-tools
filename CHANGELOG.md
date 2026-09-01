# Changelog

## 0.5.0.dev0 — unreleased

**Breaking.** MD-templates became MD-tools: a standalone, pip-installable package with one
executable and three commands. Every detail, with the tests that verify it, is in
[`docs/release-notes/v0.5.0.md`](docs/release-notes/v0.5.0.md).

### The interface

- one installed executable, `md-openmm`, with exactly `build-top`, `build-md` and `data-register`
- AIS is a `build-md` protocol, not a fourth command
- **removed**: `sys-config`, `sys-gen`, `md-gen`, `setup`, `show-default`, the `openmm-md`
  executable, and the `md-template` environment installer. `pip install md-tools` replaces the
  installer; nothing replaces the rest, because the three commands cover what they did.

### Configuration

- `configs/{machine,sys,md}/` are ordinary browsable files at the repository root, one copy each,
  shipped as wheel data files and located through distribution metadata
- every length is an exact integer step count; logs derive ps/ns
- examples must resolve, through the real resolver, to the model's own defaults

### The dataset contract

- contract **v2**, owned by MD-tools: `$MD_DATA/{year}/{project}/{data}/`, no month segment
- `extension.yaml` ported with a newly required checkpoint digest
- `dataset-v2.0` and `extension-v2.0` schemas generated from the models and drift-checked
- **`md_data` is no longer imported at runtime**, and contract v1 is gone

### Fixed

- `src/md_tools/build/` — the package implementing `build-top` and `build-md` — had never been
  committed, silently excluded by an unanchored `build/` ignore rule. A clone of this repository
  did not contain two of its three commands.
- a stage now records the parent state it continued from **with its digest**, so a parent rewritten
  after a child consumed it is caught rather than becoming false ancestry
- a 4 fs timestep is refused unless the masses in the System prove hydrogen mass repartitioning

---

Earlier releases are in Git history. The state immediately before the v0.5.0 cleanup is preserved
at the tag `pre-v0.5-doc-cleanup`; this file no longer carries hundreds of lines about commands and
an installer that no longer exist.
