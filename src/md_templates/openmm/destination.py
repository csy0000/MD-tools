"""What a generator refuses to overwrite, and how it says so.

Both public generators publish transactionally: they build into a staging directory and then
*replace* the destination. That is what keeps a failed run from leaving a half-written bundle
behind, but it also means `--overwrite` deletes the whole destination directory, not just the files
being rewritten. Anything else living there goes with it.

So the guard here does two things, and the second is the one that is easy to forget:

1. It refuses when a file the generator is **about to write** already exists, naming those files, so
   the answer to "what would this clobber?" is in the error rather than in the reader's head.
2. When `--overwrite` is given, it reports what else is in the destination -- files the generator
   does not produce and would nonetheless destroy.

One implementation, imported by both generators, so `--overwrite` means exactly the same thing in
each and there is a single place to audit the rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = ["DestinationExists", "check_destination", "publish", "SYSTEM_BUNDLE_TARGETS",
           "project_targets"]

#: Everything `prepare_system` publishes into a system bundle.
#:
#: Kept here rather than in `system_prep` so that the entry point can check a destination without
#: importing the scientific stack -- the whole reason that import is deferred is that it is slow.
#: It is a superset of `system_prep.REQUIRED_BUNDLE_FILES`; `tests/test_public_generators.py`
#: asserts that containment, so the two cannot drift apart silently.
SYSTEM_BUNDLE_TARGETS = (
    "system.xml",
    "topology.pdb",
    "topology.cif",
    "initial_state.xml",
    "forcefield.json",
    "system_manifest.json",
    "system.yaml",
    "checksums.json",
    "config.json",
    "original_inputs",
    "system_prep",
    "system_simbox.json",
    "manifest_system_simbox.json",
)

#: What a generated project holds regardless of which stages it contains.
PROJECT_TARGETS = ("run_all.sh", "run_manifest.json", "inputs")


def project_targets(stages) -> tuple[str, ...]:
    """Every file `generate_project` writes, for the stages it will actually generate.

    Stage-dependent because `--inherit <manifest>:<stage>` generates fewer stage directories, and a
    fixed list would report collisions for directories that were never going to be written.
    """
    return (*PROJECT_TARGETS,
            *(f"{s}/{s}.json" for s in stages),
            *(f"{s}/{s}.sh" for s in stages))

#: How many colliding paths to list before summarising the rest.
_MAX_SHOWN = 12


class DestinationExists(Exception):
    """Raised when writing would overwrite files that already exist."""


def _listed(paths: list[str]) -> str:
    shown = "\n".join(f"    {p}" for p in paths[:_MAX_SHOWN])
    if len(paths) > _MAX_SHOWN:
        shown += f"\n    ... and {len(paths) - _MAX_SHOWN} more"
    return shown


def _grouped(paths: list[str]) -> str:
    """Summarise by top-level entry rather than listing every file.

    An executed project has hundreds of result files, and an alphabetical sample of them says far
    less than "REST2_1/ -- 213 files" does. What a reader needs from this warning is which parts of
    their run are about to disappear, not the first twelve filenames.
    """
    groups: dict[str, int] = {}
    for path in paths:
        head = path.split("/", 1)[0] if "/" in path else path
        groups[head] = groups.get(head, 0) + 1
    lines = []
    for head in sorted(groups):
        count = groups[head]
        lines.append(f"    {head}/  ({count} files)" if count > 1 or head not in paths
                     else f"    {head}")
    return "\n".join(lines)


def _collateral(outdir: Path, targets: list) -> list[str]:
    """Files `--overwrite` would delete that this generator does not produce.

    Looking only at the top level is not enough, and on an executed project it misses everything
    that matters: `min/` is a directory we write into, but `min/min_final_state.xml` is a *result*,
    and `--overwrite` deletes it along with the rest of the directory. Those are exactly the files
    someone needs warned about before they lose a run.

    Directory targets (`inputs/`, `original_inputs/`) are ours in their entirety, so their contents
    are not collateral and are not listed.
    """
    owned_files = set(targets)
    owned_trees = tuple(f"{t}/" for t in targets if (outdir / t).is_dir())
    found = []
    for path in outdir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(outdir).as_posix()
        if rel in owned_files or rel.startswith(owned_trees):
            continue
        found.append(rel)
    return sorted(found)


def check_destination(outdir: Path, targets: Iterable[str], *, overwrite: bool,
                      what: str) -> dict:
    """Refuse to write over existing files unless ``overwrite`` is set.

    ``targets`` are paths relative to ``outdir`` -- the specific things this generator creates, not
    the whole directory. A destination that merely contains unrelated files is not a reason to stop:
    the reason to stop is that a file we are about to write is already there, because a destination
    half-rewritten from a different configuration runs without complaint and means nothing.

    Returns ``{"colliding": [...], "collateral": [...]}``. ``collateral`` is what ``--overwrite``
    would additionally delete, and is empty unless the destination holds files this generator does
    not produce.

    Raises `DestinationExists` when there are collisions and ``overwrite`` is false.
    """
    outdir = Path(outdir)
    targets = list(dict.fromkeys(targets))
    if not outdir.exists():
        return {"colliding": [], "collateral": []}
    if not outdir.is_dir():
        raise DestinationExists(
            f"destination {outdir} exists and is not a directory, so no {what} can be written "
            f"there. Choose another destination."
        )

    colliding = sorted(t for t in targets if (outdir / t).exists())
    collateral = _collateral(outdir, targets)

    if colliding and not overwrite:
        extra = ""
        if collateral:
            extra = (
                f"\n\n  --overwrite replaces the WHOLE destination directory, which would also "
                f"delete {len(collateral)} file(s) it did not write:\n{_grouped(collateral)}"
            )
        raise DestinationExists(
            f"destination {outdir} already holds {len(colliding)} file(s) this {what} would "
            f"write:\n{_listed(colliding)}\n\n"
            f"  Refusing to write: a destination half-rewritten from a different configuration "
            f"would run without complaint and mean nothing.\n"
            f"  Use --overwrite to replace it, or choose another destination."
            f"{extra}"
        )
    return {"colliding": colliding, "collateral": collateral}


def publish(staging: Path, outdir: Path, *, overwrite: bool, what: str) -> None:
    """Move a completed staging directory into place.

    Three cases, and the middle one is why this is not a one-liner:

    * the destination does not exist -- a single rename, fully atomic.
    * the destination exists and holds nothing we would write -- the staged entries are moved in
      individually. Replacing the directory wholesale would delete files that are not ours and that
      `check_destination` explicitly declined to complain about; refusing outright would make that
      check a lie.
    * ``overwrite`` -- the destination is replaced entirely, which is what the warning in
      `check_destination` says it does.

    A target that exists here but did not exist at the pre-flight check appeared while we worked, so
    it is a race and not a user error, and it is reported as such.
    """
    import shutil

    staging, outdir = Path(staging), Path(outdir)
    if not outdir.exists():
        staging.replace(outdir)
        return
    if overwrite:
        shutil.rmtree(outdir)
        staging.replace(outdir)
        return

    for entry in list(staging.iterdir()):
        target = outdir / entry.name
        if target.exists():
            raise RuntimeError(
                f"{target} appeared while the {what} was being written; refusing to overwrite it"
            )
        entry.replace(target)
    shutil.rmtree(staging, ignore_errors=True)
