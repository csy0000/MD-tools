# Registering a dataset

A finished run is a directory. Registration turns it into a **verified, immutable dataset** under
`$MD_DATA`, and replaces the local directory with a symlink to it.

## Before you start

* the run is **finished** — completion is read from the machine records, not from prose in a log;
* you have decided the `data_name`, because registration is **write-once** and the name is
  permanent;
* everything a later tool needs is **inside the tree**. A dataset that lacks the inputs its own
  build record cites cannot be exported afterwards, and cannot be repaired in place.

## Dry run first

```bash
md-openmm data-register -idata <run-dir> \
    -project_name <project> -data_name <name> -year <YYYY> --dry-run
```

`--dry-run` verifies everything and **writes nothing** — not even the destination directory. It
prints the canonical path, the file count and total size, the machine records it found and the
lineage it verified.

## Register

```bash
md-openmm data-register -idata <run-dir> \
    -project_name <project> -data_name <name> -year <YYYY>
```

`year` is the year the data were **completed**, not the year you registered them.

The command stages the tree at its canonical path, re-verifies every digest at the destination, and
only then replaces the local directory with a relative symlink. **The source is preserved unless
every check passes.**

## Afterwards

```bash
md-openmm data-register -idata <path> --verify-only
```

Re-hashes the registered tree against its manifest. Worth running once after registering, and again
before citing the dataset anywhere.

## What you get

```text
$MD_DATA/{year}/{project_name}/{data_name}/
    dataset.yaml        the manifest: every file, its digest, the software and the lineage
    ...                 the run tree, byte for byte
```

What each field of `dataset.yaml` means: [project data](../../structure/project-data.md). What the
move does to your working directory: [registered data](../../structure/registered-data.md).

!!! warning "Naming is the part you cannot take back"
    Two runs on one system differ in what was *changed*, not in what they *were*. A name that says
    `tyk2-rest2` twice is two datasets nobody can tell apart; `tyk2-ejm31-rest2-ligand-8x5ns` says
    the ligand, the region, the rungs and the length.
