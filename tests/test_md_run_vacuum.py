"""md-run and a vacuum System: refused by PROVENANCE, not by geometry (S0 ruling, 2026-09-19).

A vacuum build and an ordinary non-periodic test System look alike from inside. Only a build
record beside the System whose `outputs.system_xml.sha256` is that exact file may say "vacuum",
and then every ordinary path -- md-run, a generated stage script -- refuses it before anything is
written, while the alchemical window runner passes `vacuum_leg=True`. With no matching record,
nothing changes.

All on the CPU platform: nothing here is CUDA evidence, and nothing here touches a GPU.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")

VACUUM = Path(__file__).resolve().parent / "data" / "alchemy" / "vacuum-v1"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
MIN_IN = "&cntrl\n  protocol = cMD,\n  stage    = min,\n/\n"


def _dataset(tmp_path: Path, *, record: bool = True, damage_record: bool = False) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    shutil.copy(VACUUM / "built.xml", build / "built.xml")
    shutil.copy(VACUUM / "built.pdb", build / "built.pdb")
    if record:
        text = (VACUUM / "built.log").read_text()
        if damage_record:
            # the record now names a different file: it must be ignored, not trusted
            lines = text.splitlines(keepends=True)
            start = next(i for i, l in enumerate(lines) if l.startswith("#   system_xml:"))
            k = next(i for i in range(start, len(lines)) if "sha256:" in lines[i])
            lines[k] = lines[k].split("sha256:")[0] + "sha256: " + "0" * 64 + "\n"
            text = "".join(lines)
        (build / "built.log").write_text(text)
    return build


def _md_run(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "min.in").write_text(MIN_IN)
    result = subprocess.run(
        CLI + ["md-run", "-i", "min.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
               "-o", "min.out", "-r", "min.xml", "-log", "min.log", "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=900)
    return run, result


def test_md_run_refuses_a_vacuum_build_and_writes_nothing(tmp_path):
    _dataset(tmp_path)
    run, result = _md_run(tmp_path)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "is a vacuum build" in output
    assert "Vacuum builds are alchemical legs; ordinary MD in vacuum is not supported." in output
    assert sorted(p.name for p in run.iterdir()) == ["min.in"]


def test_the_same_system_with_no_record_runs_as_before(tmp_path):
    _dataset(tmp_path, record=False)
    run, result = _md_run(tmp_path)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (run / "min.xml").is_file()


def test_a_record_that_names_another_file_is_ignored_not_trusted(tmp_path):
    _dataset(tmp_path, damage_record=True)
    run, result = _md_run(tmp_path)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def _preflight(tmp_path, build, **kwargs):
    from md_tools.run.preflight import preflight_stage

    return preflight_stage(
        topology=build / "built.pdb", system=build / "built.xml",
        output=tmp_path / "w.out", log=tmp_path / "w.log", restart=tmp_path / "w.xml",
        cpu=True, protocol="alchemical window test", timestep_fs=2.0, ensemble="NVT",
        **kwargs)


def test_the_window_runner_path_proceeds_and_knows_it_is_vacuum(tmp_path):
    from md_tools.run.preflight import PreflightError

    build = _dataset(tmp_path)
    with pytest.raises(PreflightError, match="ordinary MD in vacuum is not supported"):
        _preflight(tmp_path, build)
    checked = _preflight(tmp_path, build, vacuum_leg=True)
    assert checked.loaded.solvation == "vacuum"
    assert checked.loaded.build_record.endswith("built.log")
    assert not any(tmp_path.glob("w.*"))


def test_with_no_record_the_label_is_the_old_one(tmp_path):
    build = _dataset(tmp_path, record=False)
    checked = _preflight(tmp_path, build)
    assert checked.loaded.solvation == "implicit" and checked.loaded.build_record is None
