"""Does this suite run the checkout it belongs to?

Every other test here assumes the answer is yes. When it is not, they do not fail -- they pass,
against different code, and the run reads as evidence for a commit it never touched. That happened:
a full green GPU suite generated every project with an editable install of another clone sitting on
the base commit, and the result was reported on a pull request.

`tests/conftest.py` pins both halves -- `sys.path` for this process, `os.environ["PYTHONPATH"]` for
everything it spawns. These two tests are the alarm on that pin, and they live in a real test
module rather than in conftest because **pytest does not collect test functions from conftest.py**.
Written there, they never ran, which is the same silence they exist to break.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .conftest import CHECKOUT_SRC


def test_this_process_imports_the_checkout_under_test():
    """In-process half: `from md_tools...` in a test module must resolve here."""
    import md_tools

    resolved = Path(md_tools.__file__).resolve().parent
    assert resolved == (CHECKOUT_SRC / "md_tools").resolve(), (
        f"this test process imports md_tools from {resolved}, not from the checkout under "
        f"test ({CHECKOUT_SRC / 'md_tools'}). Every in-process assertion in this suite would "
        f"be about different code, and would look like a pass."
    )


def test_subprocesses_run_the_checkout_under_test():
    """Subprocess half: generation is spawned as `python -m md_tools.cli.md_openmm`.

    A subprocess resolves the INSTALLED package regardless of what this process imported, so the
    pin has to reach it through the environment.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import md_tools, os; print(os.path.dirname(md_tools.__file__))"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip()).resolve()
    assert resolved == (CHECKOUT_SRC / "md_tools").resolve(), (
        f"subprocesses in this suite import md_tools from {resolved}, not from the checkout "
        f"under test. Every generated project below would be built by different code."
    )


def test_no_source_file_is_hidden_from_git_by_an_ignore_rule():
    """Every file under src/, tests/ and configs/ must be visible to git.

    This exists because it happened. `.gitignore` carried a bare `build/` rule for Python build
    artifacts, and a bare pattern matches a directory of that name at ANY depth -- so the whole of
    `src/md_tools/build/`, the package implementing `build-top` and `build-md`, was silently
    excluded from every commit. It passed every test on the machine that wrote it, because the
    files were there; a clone of the repository did not contain the two commands the project is
    for.

    Anchoring the rule (`/build/`) fixed it. This test makes the class of failure impossible to
    reintroduce quietly: an ignore pattern that swallows source now fails here rather than in
    somebody else's clone.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    tracked_roots = [d for d in ("src", "tests", "configs") if (repo / d).is_dir()]
    candidates = []
    for root in tracked_roots:
        for path in (repo / root).rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.relative_to(repo).parts)
            if "__pycache__" in parts or path.suffix == ".pyc":
                continue
            if any(part.endswith(".egg-info") for part in parts):
                continue
            candidates.append(path.relative_to(repo))

    assert candidates, "found no source files to check, which means this test is not testing"
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=repo, text=True,
                            input="\n".join(str(c) for c in candidates),
                            capture_output=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these source files are invisible to git because an ignore rule matches them:\n  "
        + "\n  ".join(ignored)
        + "\n\nAnchor the rule to the repository root (`/build/`, not `build/`).")


def test_no_test_hard_codes_a_path_on_one_machine():
    """An absolute path to somebody's home or scratch volume is a silent skip everywhere else.

    This is not hypothetical. `cpptraj` and Amber's reference `rem.log` were both read from an
    absolute path under one workstation's software tree. Locally everything ran; in CI -- where
    AmberTools IS installed -- eight tests skipped, including the only two that check our Amber
    trajectory and H-REMD log against the real parser. The suite reported success for code no
    machine but one had actually exercised.

    Tools are found on PATH. Fixtures live in `tests/data/`.
    """
    import re

    # Only a path used to LOCATE something counts. A test may legitimately mention an absolute
    # path as data -- a synthetic value in a contract fixture, or the very string a generated
    # script is asserted NOT to contain -- and neither is a machine dependency.
    forbidden = re.compile(
        r'(Path\(|open\(|which\(|run\(\[)[^)]*["\'](/home/|/data\d*/|/scratch/|/Users/)')
    offenders = []
    tests = Path(__file__).resolve().parent
    for path in sorted(tests.glob("test_*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            # A path inside a comment or docstring is explanation, not a lookup.
            if stripped.startswith("#") or stripped.startswith("#:"):
                continue
            if forbidden.search(line):
                offenders.append(f"{path.name}:{number}: {stripped[:90]}")
    assert not offenders, (
        "tests must not hard-code a path on one machine:\n  " + "\n  ".join(offenders))
