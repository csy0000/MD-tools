# Registering finished data

`md-openmm data-register` takes a directory you have finished writing and moves it into managed
storage as a **verified, immutable dataset**. It is the third of the three commands, and the only
one that touches anything outside your working directory.

Three example files sit beside this page. They are documentation, not new schemas: each one is
validated in the test suite against the same model the command itself uses.

| file | what it is |
|---|---|
| [`user.config.example`](user.config.example) | the file `--init` writes: who you are, and where your storage root is |
| [`dataset.yaml.example`](dataset.yaml.example) | the manifest registration **writes**, shown so you know what to expect |
| [`extension.yaml.example`](extension.yaml.example) | the declaration **you write** when a dataset continues another |

The schema-level authority is [project data](../../structure/project-data.md). This page is how to
use it.

## First, once per machine

```bash
md-openmm data-register --init
```

Interactive on a terminal; fully flag-driven otherwise
(`--noninteractive --name ... --person-id ... --md-data ...`). It writes:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/md-tools/user.config
```

Never into the package, the repository, or `site-packages`. The storage root is resolved in this
order, and the command always prints which source supplied it:

1. `--md-data PATH`
2. `$MD_DATA`
3. `machine.md_data` in the user configuration

## Where a dataset ends up

```text
$MD_DATA/{year}/{project_name}/{data_name}/          # role: project
$MD_DATA/common/{year}/{project_name}/{data_name}/   # role: common
```

**No month segment.** `year` is the year the dataset was *completed*, and it is checked against the
machine records inside the directory rather than taken on trust.

`common/` is for data several projects legitimately share — a prepared system, a reference
trajectory — so that the second project to need it links rather than copies.

## The ordinary flow

Generate into a local, git-ignored `data/` directory, then register:

```bash
# 1. produce it
md-openmm build-top -i ALA.pdb -os built.xml -op built.pdb -log built.log
md-openmm build-md  -odir ./data/ALA-cMD/cMD-run1 --config cMD.config
cd data/ALA-cMD/cMD-run1 && ./run.sh && cd -

# 2. look before you leap: dry run writes nothing, anywhere
md-openmm data-register -idata ./data/ALA-cMD \
    -project_name ALA -data_name ALA-cMD -year 2026 --dry-run

# 3. register
md-openmm data-register -idata ./data/ALA-cMD \
    -project_name ALA -data_name ALA-cMD -year 2026
```

Afterwards `./data/ALA-cMD` is a **symlink** to the registered location, so paths in your notes keep
working.

## What is checked before anything moves

* every machine record parses, carries a known schema version, and reports `status: completed` —
  **prose in a log is never evidence of completion**;
* nothing is still being written to (an actively-written directory is refused);
* no symlink inside the dataset, no path traversal in any name, `year` exactly four digits;
* the data has not changed since the records were written;
* the destination does not already hold different bytes under the same name.

## How the move is done

Registration is a transaction with seven durable states:

```text
planned → staged → verified → committed → source-removed → linked → complete
```

Staging is a **sibling of the destination**, so the commit is an atomic rename rather than a copy
across a filesystem boundary. Every destination file is re-hashed and compared with the source
**before** the source is removed — the source is the only other copy until that point.

Interrupted, it is safe to run again:

* before `committed` — the source is untouched; re-running finishes the job;
* after `committed` — the data is already at its final path, complete and verifiable. Re-running
  reports that the dataset is already registered and, in the narrow window where the source was
  removed but the link not yet made, tells you the command to restore the link.

Re-registering identical bytes is a no-op. Re-registering *different* bytes under a name that
already exists is refused, and the source is preserved.

## Collective-variable artefacts

Two independent things record a run's CV output, and they answer different questions.

`SHA256SUMS` lists every file under the dataset root, so `cv_state<N>.csv` and its sidecar are
hashed and re-checkable like any other file — that is what `--verify-only` re-reads.

The **provenance record** is separate: registration reads each run's machine `-log`, and
`check_lineage` indexes that record's `outputs` block by digest to connect one stage's outputs to
the next stage's inputs. A file absent from `outputs` is invisible to lineage checking even while
sitting in `SHA256SUMS`, which is exactly what happened to the ladder's CV series. Each per-state
CSV and sidecar is now named there as its own role, carrying the path relative to the run root,
the digest and byte size the completion manifest recorded, the state index and its tau, and the
CV definition digest. The inventory is built from that validated manifest rather than from a
`cv_state*.csv` glob, so a file left behind by an earlier run into the same directory is not
recorded as this run's provenance.

## Verifying later

```bash
md-openmm data-register -idata $MD_DATA/2026/ALA/ALA-cMD --verify-only
```

Re-reads the destination and re-hashes every file against `SHA256SUMS`, naming any difference.

## Extending a completed dataset

A completed dataset is immutable. Continuing one produces a **new** dataset holding only the new
segment, with the parent pinned by content and left byte-for-byte unchanged.

Write an [`extension.yaml`](extension.yaml.example) into the new directory before registering it.
Registration then:

* refuses an unfinished parent;
* re-hashes the copied source checkpoint against `source_checkpoint_sha256` — an in-place extension
  without a copied checkpoint is refused outright;
* cross-checks the declaration against the manifest's `derived_from` and refuses a disagreement;
* never opens the parent dataset to do any of it.

The chain is therefore a sequence of immutable segments. `cpptraj` concatenates them with no gap
and no duplicated boundary frame; the tests assert exactly that.

## Contract versions

Dataset **v2** and extension **v2**. The generated schemas ship with the package and are
drift-checked against the models that actually validate, so a schema and its validator cannot
disagree. `md_data` is never imported at run time.

## Related

* [The dataset contract](../../structure/project-data.md) — the schema-level authority
* [Method pages](../../openmm_methods/README.md) — what produces the data in the first place
