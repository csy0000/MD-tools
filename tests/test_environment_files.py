"""`environment.yml` must carry the CUDA pin, and must be able to run the suite it documents.

THERE USED TO BE TWO FILES. `environment-ci.yml` described a CPU-only, single-rank environment and
`environment-cuda.yml` was that list plus `cuda-version`, `mpi4py` and `openmpi`; most of this
module existed to stop them drifting apart. They were merged into one `environment.yml` on
2026-09-16, so those comparisons have nothing left to compare and are gone rather than rewritten
into something weaker.

Merging cost no coverage, which is the part worth recording. The tests that prove a plural launch
without working MPI refuses and writes nothing do not rely on the environment lacking `mpi4py` --
they force its absence (`MD_TOOLS_FORCE_NO_MPI4PY`, `NO_MPI4PY`, a shadowed module that raises on
import), so they behave identically in an environment that has it. CI installs a CUDA stack it
cannot use on a driverless runner, which is a slower solve and nothing else: no test claims CUDA
evidence there, and `ci.yml` says so where a reader would look.

The pin itself is not a preference. OpenMM compiles kernels at run time: nvrtc emits PTX and the
driver assembles it, PTX is backward compatible only, and CUDA's minor-version-compatibility rule
exempts the PTX JIT. cuda-nvrtc 13.3 against a CUDA 13.0 driver therefore installs cleanly and
fails every kernel it must compile, with CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222) -- while
anything already in the kernel cache keeps working, so it looks like flaky hardware rather than a
bad solve. Losing this pin costs a campaign; a test is cheap.

PLATFORM_POLICY_EXEMPTION: reads one YAML file. Nothing is propagated and no platform is chosen.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "environment.yml"
PYPROJECT = ROOT / "pyproject.toml"


def _dependencies(path: Path) -> dict[str, str]:
    """`{name: full spec}` for every dependency, ignoring the comments around them."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    found = {}
    for entry in document["dependencies"]:
        if not isinstance(entry, str):
            continue                      # a nested pip: block, if one is ever added
        name = entry.split("=")[0].split(">")[0].split("<")[0].strip()
        found[name] = entry.strip()
    return found


def test_the_environment_file_exists_and_is_the_only_one():
    """One file. A second would be the thing this module used to spend five tests policing."""
    assert ENVIRONMENT.is_file()
    strays = sorted(p.name for p in ROOT.glob("environment-*.yml"))
    assert not strays, f"a second environment file is back: {strays}"


def test_it_pins_cuda_version():
    """Pinned, and pinned to a specific minor -- `cuda-version` alone would solve to the newest."""
    spec = _dependencies(ENVIRONMENT).get("cuda-version")
    assert spec is not None, "environment.yml must pin cuda-version"
    assert "=" in spec, f"cuda-version must name a version; got {spec!r}"


def test_the_reason_for_the_pin_is_written_down_where_someone_would_remove_it():
    """A pin whose reason is not beside it gets 'cleaned up' by the next person to read the file.

    The failure it prevents is invisible on a warm kernel cache, so a maintainer who removes the
    pin will very likely see everything pass.
    """
    text = ENVIRONMENT.read_text(encoding="utf-8")
    assert "UNSUPPORTED_PTX_VERSION" in text
    assert "OPENMM_CACHE_DIR" in text, "the file must say how to reproduce the failure on purpose"


def test_it_can_run_the_suite_the_project_configures():
    """`addopts` is passed to EVERY pytest invocation, so the environment has to honour it.

    This is not hypothetical. `pyproject.toml` has set `--dist loadgroup` while both environment
    files listed a bare `pytest`, so CI's test lane exited 4 with `unrecognized arguments: --dist`
    before collecting anything -- and nobody saw it, because the step before it failed first and
    skipped the lane. A flag in `addopts` whose plugin is not installed is a suite that cannot run.
    """
    addopts = re.search(r'addopts\s*=\s*"([^"]*)"', PYPROJECT.read_text(encoding="utf-8"))
    assert addopts, "pyproject.toml no longer sets addopts; this test needs rewriting"
    installed = _dependencies(ENVIRONMENT)
    if "--dist" in addopts.group(1) or "-n" in addopts.group(1).split():
        assert "pytest-xdist" in installed, (
            f"addopts is {addopts.group(1)!r}, which needs pytest-xdist; environment.yml has "
            f"{sorted(installed)}")


def test_it_carries_the_test_runner_at_all():
    """Obvious, and worth asserting: the file documents the environment the suite runs in."""
    installed = _dependencies(ENVIRONMENT)
    assert "pytest" in installed
