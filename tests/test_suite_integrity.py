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
