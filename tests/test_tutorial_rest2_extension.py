"""The paracetamol REST2 tutorial's extension section, checked against the parsers it instructs.

Section 8 once told the reader to extend a ladder with `md-openmm md-run --extend-from`, which
fails in argparse: `md-run` defines neither `--extend` nor `--extend-from`. Nothing caught it,
because no test compared the prose with the code. These do, for the two claims that would silently
go stale: which flags the generated entry point accepts, and which ones `md-run` does NOT.

They assert the parsers, never the prose's wording, so rephrasing the section is free and
renaming a flag is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SECTION = "## 8. Extending it"
PAGE = Path(__file__).resolve().parents[1] / "docs/tutorial/paracetamol/REST2.md"


def _extension_section():
    text = PAGE.read_text(encoding="utf-8")
    start = text.index(SECTION)
    rest = text[start + len(SECTION):]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _flags_in(text):
    """Every long flag the section's shell blocks pass. Prose mentions are excluded on purpose:
    the section NAMES `--resume` and `--force` to say they are refused, and a flag the tutorial
    says is refused must not be required to exist."""
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert blocks, "the extension section has no shell block to check"
    return {flag for block in blocks for flag in re.findall(r"(--[a-z][a-z-]+)", block)}


def test_every_flag_the_section_passes_is_one_the_generated_ladder_accepts():
    from md_tools.remd.generated import replica_parser

    parser = replica_parser()
    known = {option for action in parser._actions for option in action.option_strings}
    used = _flags_in(_extension_section())
    assert used, "no flags found -- the parse is broken, not the page"
    assert used <= known, f"the tutorial passes flags the ladder rejects: {sorted(used - known)}"


def test_the_section_passes_the_flags_the_feature_needs():
    """A guard that cannot fail is worth nothing: if `_flags_in` ever stopped finding anything,
    the test above would pass on an empty set. These are the flags the section exists to show."""
    used = _flags_in(_extension_section())
    for flag in ("--extend", "--extend-from", "--extend-manifest", "--groupfile"):
        assert flag in used, f"section 8 no longer shows {flag}"


def test_md_run_still_does_not_define_the_extension_flags():
    """The section tells the reader to use the generated script BECAUSE `md-run` cannot do this.
    If `md-run` ever gains these flags that advice becomes wrong, and this says so."""
    from md_tools.run.main import md_run_parser

    known = {option for action in md_run_parser()._actions for option in action.option_strings}
    for flag in ("--extend", "--extend-from", "--extend-manifest"):
        assert flag not in known, (
            f"md-run now defines {flag}, so the tutorial's reason for using the generated "
            f"REST2.py is no longer true. Update section 8 rather than deleting this test.")


@pytest.mark.parametrize("name", ["remd_records/restart_prod1.json", "restart.json"])
def test_both_manifest_names_the_section_tables_are_the_real_ones(name):
    """The table says a `run.sh` parent files its manifest under `remd_records/` and an extension
    segment writes `restart.json` at its root. The second is the driver's assumed default; the
    first is what `build-md` writes, and the whole point of `--extend-manifest` is that they
    differ."""
    from md_tools.remd.driver import ReplicaRun

    section = _extension_section()
    assert name in section, f"the section no longer names {name}"
    if name == "restart.json":
        assert ReplicaRun.PARENT_MANIFEST == name
    else:
        assert ReplicaRun.PARENT_MANIFEST != name, (
            "the default is now the generated name, so the flag -- and this table -- are obsolete")
