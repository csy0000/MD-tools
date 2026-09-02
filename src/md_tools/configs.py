"""Locating the shipped configuration examples.

The examples live at the repository root, under `configs/{machine,sys,md}/`, as ordinary browsable
files. There is exactly one copy of each: no symlink, and no second hand-edited mirror inside the
package. GitHub shows them as a directory tree, which a `mode-120000` symlink blob did not.

They reach an installed environment as **wheel data files**, which pip unpacks under `sys.prefix`:

    <prefix>/share/md-tools/configs/machine/user.config.example
    <prefix>/share/md-tools/configs/sys/build-top.config
    <prefix>/share/md-tools/configs/md/{cMD,REST2,rREST2,AIS}.config

so `example_root()` asks the installed distribution where they went rather than assuming a layout.
`importlib.resources` is not the right tool here: these are not package data, and making them package
data is exactly the duplication this arrangement exists to avoid.

The examples are documentation of the schemas, never their source. Runtime defaults live in the
validated models in `md_tools.build`; a test resolves every shipped example and asserts it produces
those same defaults, so an example cannot drift into describing a default that is not applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Where the data files are installed, relative to the environment prefix.
INSTALLED_PREFIX = Path("share") / "md-tools" / "configs"

#: Every example that must exist, relative to the configs root.
EXAMPLES = (
    Path("machine") / "user.config.example",
    Path("sys") / "build-top.config",
    Path("md") / "cMD.config",
    Path("md") / "REST2.config",
    Path("md") / "rREST2.config",
    Path("md") / "AIS.config",
)


class ExamplesNotFound(RuntimeError):
    """The shipped examples could not be located in this environment."""


def _from_distribution() -> Path | None:
    """Ask the installed distribution where its data files went.

    A wheel records data files in RECORD as paths relative to site-packages, typically
    `../../share/md-tools/configs/...`. Reading them from the metadata means a non-standard prefix,
    a virtualenv, or a `--target` install are all found without guessing.
    """
    from importlib.metadata import PackageNotFoundError, files as distribution_files

    try:
        recorded = distribution_files("md-tools")
    except PackageNotFoundError:
        return None
    if not recorded:
        return None
    for entry in recorded:
        text = str(entry).replace("\\", "/")
        if "share/md-tools/configs/" not in text:
            continue
        resolved = Path(entry.locate()).resolve()
        # Walk back up to the `configs` directory itself.
        for parent in resolved.parents:
            if parent.name == "configs" and parent.parent.name == "md-tools":
                return parent
    return None


def _from_prefix() -> Path | None:
    for base in (Path(sys.prefix), Path(getattr(sys, "base_prefix", sys.prefix))):
        candidate = base / INSTALLED_PREFIX
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _from_source_tree() -> Path | None:
    """A source checkout: `src/md_tools/configs.py` sits two levels below the repository root."""
    candidate = Path(__file__).resolve().parents[2] / "configs"
    return candidate.resolve() if candidate.is_dir() else None


def _holds_the_examples(candidate: Path) -> bool:
    """A candidate counts only if the examples are actually IN it.

    "The directory exists" is not "the examples are there", and the difference is not academic:
    `pip uninstall` removes a wheel's data files but leaves the directories they lived in, and
    `pip install -e .` does not install data files at all (setuptools does not, under PEP 660).
    An install-then-editable cycle therefore leaves an EMPTY `<prefix>/share/md-tools/configs`,
    which the old test accepted -- so the source-tree fallback below it was never reached, and
    every caller got `ExamplesNotFound: <that empty path> does not exist` from one level deeper.
    """
    return candidate.is_dir() and all((candidate / relative).is_file() for relative in EXAMPLES)


def example_root() -> Path:
    """The directory holding `machine/`, `sys/` and `md/`, wherever this is running."""
    tried = []
    for locate in (_from_distribution, _from_prefix, _from_source_tree):
        found = locate()
        if found is None:
            continue
        if _holds_the_examples(found):
            return found
        tried.append(found)
    where = ("\n  ".join(f"{path} (present, but does not hold all {len(EXAMPLES)} examples)"
                         for path in tried)) if tried else "nowhere"
    raise ExamplesNotFound(
        "the shipped configuration examples were not found. They install as wheel data files "
        f"under <prefix>/{INSTALLED_PREFIX.as_posix()}; in a source checkout they are at the "
        f"repository root under configs/.\n  Looked at: {where}")


def example(relative: str | Path) -> Path:
    """One shipped example, by its path relative to the configs root."""
    path = example_root() / Path(relative)
    if not path.is_file():
        raise ExamplesNotFound(f"{path} does not exist")
    return path


def all_examples() -> list[Path]:
    root = example_root()
    return [root / relative for relative in EXAMPLES]
