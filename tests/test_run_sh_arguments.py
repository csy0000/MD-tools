"""`run.sh` reads leading path arguments only up to the first flag.

The generated header took `$1` as the topology whatever it was, so `./run.sh --cpu` -- the form its
own header comment recommends for a CPU run -- stopped with "no topology at --cpu" on every
protocol. A flag now ends the positional paths and is forwarded to each `md-run`.

The script is executed for real, against a stand-in `md-openmm` on PATH that records its arguments
and exits 0, so what is under test is the generated bash, not a reading of it.

PLATFORM_POLICY_EXEMPTION: no Context is created; `md-openmm` is a recording shim.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from .conftest import make_dataset_root


@pytest.fixture
def generated(tmp_path):
    from md_tools.build.md import build_scripts

    root = make_dataset_root(tmp_path / "ALA")
    config = tmp_path / "cMD.config"
    config.write_text("protocol: cMD\nsolvent: implicit\n", encoding="utf-8")
    build_scripts(config_path=config, out_dir=root / "cMD-run1", echo=False)

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    record = tmp_path / "calls.txt"
    shim = shim_dir / "md-openmm"
    shim.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{record}"\n', encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return root, shim_dir, record


def _run(root: Path, shim_dir: Path, *arguments):
    environment = dict(os.environ)
    environment["PATH"] = f"{shim_dir}{os.pathsep}{environment['PATH']}"
    return subprocess.run(["bash", str(root / "cMD-run1" / "run.sh"), *arguments],
                          capture_output=True, text=True, timeout=60, env=environment)


def test_a_leading_flag_uses_the_default_paths_and_is_forwarded(generated):
    root, shim_dir, record = generated
    done = _run(root, shim_dir, "--cpu")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = record.read_text(encoding="utf-8").splitlines()
    assert calls and all(call.split()[-1] == "--cpu" for call in calls), calls
    assert all("/build/built.pdb" in call for call in calls), calls


def test_named_paths_are_still_taken_and_not_forwarded(generated):
    root, shim_dir, record = generated
    build = root / "build"
    done = _run(root, shim_dir, str(build / "built.pdb"), str(build / "built.xml"), "--cpu")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = record.read_text(encoding="utf-8").splitlines()
    assert calls and all(call.split()[-1] == "--cpu" for call in calls), calls
    assert not any(call.count("built.pdb") > 1 for call in calls), (
        "a consumed path was forwarded again as an argument")


def test_a_missing_named_topology_is_still_refused(generated):
    root, shim_dir, _record = generated
    done = _run(root, shim_dir, str(root / "nowhere.pdb"))
    assert done.returncode == 2
    assert "no topology at" in done.stderr


def test_a_grouped_ladder_launch_with_a_trajectory_flag_is_not_a_crash():
    """`_check_file_roles` compared `-x` with `-s` whenever `-x` was given, but a grouped ladder has
    no `-s` (0.5.4), so `md-run --groupfile ... -x REST2.nc` raised UnboundLocalError."""
    from argparse import Namespace

    from md_tools.run.main import _check_file_roles

    common = {"source_traj": None, "output": None, "log": None}
    _check_file_roles(Namespace(system=None, trajectory="REST2.nc", groupfile="remd_groupfile.1",
                                **common))
    with pytest.raises(SystemExit, match="same path"):
        _check_file_roles(Namespace(system="same.bin", trajectory="same.bin", groupfile=None,
                                    **common))
