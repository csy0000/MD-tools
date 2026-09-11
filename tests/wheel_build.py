"""Build the package's wheel from a COPY of the working tree. One implementation, for every test.

`pip wheel <repo>` has setuptools write into `<repo>/build/`, which every pytest-xdist worker
shares: several of them building at once corrupt each other's intermediate tree. It surfaced twice
in two costumes -- as a module that errored at setup with nothing but "Processing ./." to show for
it, and as `No such file or directory: 'build/bdist.linux-x86_64/wheel'` turned into a SKIP by a
fixture that read any failure to build as "the wheel could not be built here". Every affected test
passes serially, so the failure looks like a packaging bug and is a concurrency bug.

The copy is of the WORKING TREE, not of HEAD -- a wheel test that silently ignored uncommitted
changes would pass on a checkout whose packaging is broken.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

#: Dropped ONLY at the repository root. `shutil.ignore_patterns` matches on basename at every
#: depth, so a bare "build" also drops `src/md_tools/build/` -- a real package -- and the wheel
#: then installs cleanly and raises `ModuleNotFoundError: md_tools.build` at import.
ROOT_ONLY = {".git", "build", "dist", ".conda-env", "tests"}
ANYWHERE = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def build_wheel_from_copy(repo: Path, work: Path) -> Path:
    """Copy `repo` into `work/checkout`, build its wheel into `work/dist`, return the wheel.

    A failure to build is an assertion, never a skip: a package that cannot be packaged is the
    defect these tests exist to find.
    """
    repo, work = Path(repo), Path(work)

    def ignore(directory, names):
        dropped = {name for name in names if name in ANYWHERE or name.endswith(".egg-info")}
        if Path(directory) == repo:
            dropped |= {name for name in names if name in ROOT_ONLY}
        return dropped

    source = work / "checkout"
    shutil.copytree(repo, source, ignore=ignore)
    built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                            "--wheel-dir", str(work / "dist"), str(source)],
                           capture_output=True, text=True, timeout=900)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    wheels = list((work / "dist").glob("md_tools-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]
