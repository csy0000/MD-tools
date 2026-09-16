"""Every method's `example.in` must be what `build-md` would actually generate.

`docs/openmm_methods/<method>/` carries two example files with different jobs:

    example.config   what you HAND to `md-openmm build-md`
    example.in       what `md-run` then READS -- generated from that config

The `.in` is documentation of a generated artifact, which is the kind of documentation that rots
fastest: a field renamed in the schema leaves a hand-written example describing an input the
parser no longer accepts, and nothing notices until a user copies it. So it is not hand-written.
Its values come from running `build-md` on the `example.config` beside it, and its comments come
from the schema's own field docs.

This test regenerates both and compares. It fails when the two drift, which is exactly when the
example has become a lie.

Comments are excluded from the comparison and the VALUES are not: an annotation may be reworded
freely, but if `build-md` starts emitting a different field or a different default, the example
must be regenerated.

PLATFORM_POLICY_EXEMPTION: generates input files and parses them. Nothing is propagated.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

METHODS = ("cMD", "REST2", "rREST2", "AIS", "umbrella")
#: The production stage each method's `example.in` shows -- the one that does the sampling.
PRODUCTION_STAGE = {"cMD": "cMD.in", "REST2": "REST2.in", "rREST2": "rREST2.in",
                    "AIS": "AIS.in", "umbrella": "umbrella.in"}


def _significant(text: str) -> list[str]:
    """The lines that carry meaning: no comments, no blank lines, whitespace normalised."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        out.append(re.sub(r"\s+", " ", stripped))
    return out


@pytest.mark.parametrize("method", METHODS)
def test_every_method_ships_both_example_files(method):
    directory = REPO_ROOT / "docs" / "openmm_methods" / method
    assert (directory / "example.config").is_file(), f"{method} has no example.config"
    assert (directory / "example.in").is_file(), (
        f"{method} has no example.in -- the file a user actually passes to md-run")


@pytest.mark.parametrize("method", METHODS)
def test_the_example_input_parses(method):
    """It must be a valid input, not merely a plausible-looking one."""
    from md_tools.run.inputs import parse_run_input

    parsed = parse_run_input(REPO_ROOT / "docs" / "openmm_methods" / method / "example.in")
    assert parsed.protocol == method


@pytest.mark.parametrize("method", METHODS)
def test_the_example_input_matches_what_build_md_generates(method, tmp_path):
    """The values in `example.in` are `build-md`'s, not a human's recollection of them."""
    from .conftest import make_dataset_root

    directory = REPO_ROOT / "docs" / "openmm_methods" / method
    # A dataset root, because a ladder's rungs are scaled from `build/built.xml` at BUILD time
    # now. The inputs this test compares are generated text and do not depend on which System
    # produced them -- see `make_dataset_root` on what that fixture may and may not stand for.
    make_dataset_root(tmp_path, solvent="explicit")
    out = tmp_path / f"{method}-run1"
    done = subprocess.run(
        [sys.executable, "-c",
         "import sys;from md_tools.cli.md_openmm import main;"
         f"sys.argv=['md-openmm','build-md','-odir',{str(out)!r},"
         f"'--config',{str(directory / 'example.config')!r}];main()"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert done.returncode == 0, f"build-md failed for {method}:\n{done.stderr[-2000:]}"

    # THE INPUT IS SHARED, so it is at the dataset root rather than inside the run: `input/` is
    # read by every repeat of a method on one system.
    generated = tmp_path / "input" / PRODUCTION_STAGE[method]
    assert generated.is_file(), f"build-md produced no input/{PRODUCTION_STAGE[method]}"

    shipped = _significant((directory / "example.in").read_text(encoding="utf-8"))
    fresh = _significant(generated.read_text(encoding="utf-8"))
    assert shipped == fresh, (
        f"docs/openmm_methods/{method}/example.in no longer matches what build-md generates "
        f"from the example.config beside it.\n"
        f"  shipped has {len(shipped)} significant lines, generated has {len(fresh)}.\n"
        f"  Regenerate it rather than editing the values by hand.")


@pytest.mark.parametrize("method", METHODS)
def test_the_example_input_is_actually_annotated(method):
    """An example with no comments is just a generated file checked in.

    The point of shipping it is that a reader learns what each key means without opening the
    schema, so the presence of the annotation is part of the contract.
    """
    text = (REPO_ROOT / "docs" / "openmm_methods" / method / "example.in").read_text(
        encoding="utf-8")
    comments = [l for l in text.splitlines() if l.strip().startswith("!")]
    settings = [l for l in text.splitlines() if re.match(r"^\s*[a-z_0-9]+\s*=", l)]
    assert len(comments) > len(settings), (
        f"{method}/example.in has {len(comments)} comment lines for {len(settings)} settings; "
        f"it is not meaningfully annotated")
