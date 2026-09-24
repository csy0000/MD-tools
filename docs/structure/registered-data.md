# Registered data

Registration is a **move**, not a copy. `md-openmm data-register` verifies a finished directory,
transactionally moves it into the managed storage root, and **replaces the local directory with a
symlink** to the verified destination. Scripts that referred to the old path keep working; the
bytes are in one place, under the contract.

## What happens, in order

1. **Verify.** Every file is hashed, the machine records are read, completion is checked from those
   records rather than from prose in a log, and the manifest is built.
2. **Move.** The tree is staged into `$MD_DATA` at its canonical path and renamed into place.
3. **Re-verify.** Every digest is checked again at the destination.
4. **Symlink.** The original location becomes a relative symlink to the registered tree.

**The source is preserved unless every check passes.** A registration that fails leaves the local
directory exactly as it was.

## The canonical path

```text
$MD_DATA/{year}/{project_name}/{data_name}/
$MD_DATA/common/{year}/{project_name}/{data_name}/     # role: common
```

`year` is the year the data were **completed**, not the year they were registered. There is no
month segment. See [project data](project-data.md) for what a dataset is and what `dataset.yaml`
records.

## Registration is write-once

A registered dataset cannot be renamed, moved or added to. Two consequences worth knowing before
you run the command:

* **The `data_name` is permanent.** Choose one that distinguishes this run from a later one on the
  same system — what changed, not just what it was.
* **Anything a later tool needs must be inside the tree at registration time.** A dataset that does
  not carry the inputs its own build record cites cannot be exported afterwards, and cannot be
  repaired in place.

## Checking without writing

```bash
md-openmm data-register -idata <dir> ... --dry-run      # verify and print, write nothing
md-openmm data-register -idata <dir> ... --verify-only  # re-verify an already registered tree
```

`--dry-run` creates nothing, including the destination directory.
