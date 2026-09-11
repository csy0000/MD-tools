"""The commit a wheel was built from, and the rule for when it may be trusted.

WHY THIS MATTERS ENOUGH TO TEST

`md_tools_commit` was `null` in every run record, every dataset manifest and every exported
reference bundle, because a wheel is built from a DIRECTORY and nothing in the build consults
git. Recovering the commit for an already-finished campaign meant hashing all 99 installed files
against the git trees. It worked, and it is not a provenance system.

Two rules are asserted here, and both exist because the plausible alternative is wrong:

  * a DIRTY tree bakes None. The commit would not describe what is in the wheel, and a provenance
    record that is confidently wrong is worse than one that is honestly empty.
  * a checkout reads LIVE GIT, not the baked file, and decides "am I a checkout" by file
    IDENTITY. `git rev-parse` walks upward from wherever it starts, so a virtual environment
    sitting inside some unrelated repository would otherwise report that repository's commit as
    the engine's -- a wrong answer indistinguishable from a right one.
"""
from __future__ import annotations

import subprocess

import pytest

from md_tools.build.record import source_commit

pytestmark = pytest.mark.slow


def _repo(path):
    """A throwaway git repository shaped like this one, so the policy is tested on a real tree."""
    (path / "src" / "md_tools").mkdir(parents=True)
    (path / "src" / "md_tools" / "__init__.py").write_text("", encoding="utf-8")
    for argv in (["init", "-q"], ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "T"], ["add", "-A"],
                 ["commit", "-qm", "initial"]):
        assert subprocess.run(["git", "-C", str(path), *argv],
                              capture_output=True).returncode == 0, argv
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def test_a_clean_tree_bakes_the_commit_being_built(tmp_path):
    import _build_backend

    head = _repo(tmp_path)
    assert _build_backend.bake(tmp_path) == head

    namespace = {}
    exec((tmp_path / "src" / "md_tools" / "_commit.py").read_text(encoding="utf-8"), namespace)
    assert namespace["COMMIT"] == head


def test_a_dirty_tree_bakes_nothing_rather_than_a_commit_that_does_not_describe_it(tmp_path):
    """The build must not claim a commit for a wheel that does not contain that commit's code."""
    import _build_backend

    head = _repo(tmp_path)
    (tmp_path / "src" / "md_tools" / "__init__.py").write_text("# edited\n", encoding="utf-8")
    assert _build_backend.bake(tmp_path) is None

    namespace = {}
    exec((tmp_path / "src" / "md_tools" / "_commit.py").read_text(encoding="utf-8"), namespace)
    assert namespace["COMMIT"] is None, f"a dirty tree claimed {head}"


def test_a_checkout_reports_live_git_and_not_a_stale_baked_value():
    """Running from a checkout, the answer must track `git rev-parse HEAD` as it moves."""
    from md_tools.build import record

    toplevel = subprocess.run(
        ["git", "-C", str(record.__file__.rsplit("/", 1)[0]), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True)
    if toplevel.returncode != 0:
        pytest.skip("not running from a checkout; the wheel path is covered by the bake tests")
    head = subprocess.run(["git", "-C", toplevel.stdout.strip(), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert source_commit() == head
