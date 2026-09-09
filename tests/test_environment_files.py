"""The two environment files must not drift apart, and the GPU one must carry the CUDA pin.

`environment-ci.yml` used to claim that `tests/test_install.py` asserted it agreed with
`src/md_tools/install/openmm.py`. Neither file exists -- they were lost and never restored -- so
the cross-check the header promised had silently stopped happening. This is the part of that
promise which can be kept.

The pin itself is not a preference. OpenMM compiles kernels at run time: nvrtc emits PTX and the
driver assembles it, PTX is backward compatible only, and CUDA's minor-version-compatibility rule
exempts the PTX JIT. cuda-nvrtc 13.3 against a CUDA 13.0 driver therefore installs cleanly and
fails every kernel it must compile, with CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222) -- while
anything already in the kernel cache keeps working, so it looks like flaky hardware rather than a
bad solve. Losing this pin costs a campaign; a test is cheap.

PLATFORM_POLICY_EXEMPTION: reads two YAML files. Nothing is propagated and no platform is chosen.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "environment-ci.yml"
CUDA = ROOT / "environment-cuda.yml"

#: In the GPU file and deliberately not in the CPU one. CI has no second device and no driver.
GPU_ONLY = {"cuda-version", "mpi4py", "openmpi"}


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


def test_both_environment_files_exist():
    assert CI.is_file() and CUDA.is_file()


def test_the_gpu_file_pins_cuda_version():
    """Pinned, and pinned to a specific minor -- `cuda-version` alone would solve to the newest."""
    spec = _dependencies(CUDA).get("cuda-version")
    assert spec is not None, "environment-cuda.yml must pin cuda-version"
    assert "=" in spec, f"cuda-version must name a version; got {spec!r}"


def test_the_cpu_file_does_not_pull_a_cuda_stack_onto_a_runner_with_no_gpu():
    assert "cuda-version" not in _dependencies(CI)


def test_every_shared_package_is_pinned_identically_in_both_files():
    """The GPU environment is the CI one plus CUDA and MPI -- not a separately drifting list.

    Two files describing 'the environment MD-tools is tested against' that disagree about, say,
    the openmmtools version would mean CI validates something nobody runs.
    """
    ci, cuda = _dependencies(CI), _dependencies(CUDA)
    shared = set(ci) & set(cuda)
    differing = {name: (ci[name], cuda[name]) for name in sorted(shared)
                 if ci[name] != cuda[name]}
    assert not differing, f"these packages are pinned differently in the two files: {differing}"


def test_the_gpu_file_adds_only_what_a_gpu_machine_needs():
    """Anything else appearing in one and not the other is drift, and is named."""
    ci, cuda = _dependencies(CI), _dependencies(CUDA)
    extra_in_cuda = set(cuda) - set(ci) - GPU_ONLY
    missing_from_cuda = set(ci) - set(cuda)
    assert not extra_in_cuda, f"environment-cuda.yml has packages CI does not: {extra_in_cuda}"
    assert not missing_from_cuda, (
        f"environment-cuda.yml is missing packages CI has: {missing_from_cuda}")


def test_the_reason_for_the_pin_is_written_down_where_someone_would_remove_it():
    """A pin whose reason is not beside it gets 'cleaned up' by the next person to read the file.

    The failure it prevents is invisible on a warm kernel cache, so a maintainer who removes the
    pin will very likely see everything pass.
    """
    text = CUDA.read_text(encoding="utf-8")
    assert "UNSUPPORTED_PTX_VERSION" in text
    assert "OPENMM_CACHE_DIR" in text, "the file must say how to reproduce the failure on purpose"
