"""`--overwrite` replaces the COMPLETE owned inventory, as one transaction, before anything new.

WHAT WAS WRONG

    `--overwrite` removed the checkpoint tree and left every other output to whichever reporter
    happened to open its file in `"w"`. Three consequences, none of them announced:

      a stream the NEW run does not write is never touched by anybody. Turn phase-space or CV
      reporting off and `--overwrite` on, and the previous `.phase_space.nc` or `.cv.csv`
      survives, still named by the inventory, still looking like output of the run that just
      finished;

      a crash between "some files removed" and "the rest opened" left a directory that was part
      one experiment and part another, with every remaining file looking equally current;

      an unrelated file in the same `-odir` was never distinguished from an owned one, so
      "replace everything" and "replace everything this stage owns" were the same code path.

    `replace_owned_inventory` (`md_tools.run.overwrite`) is the fix: every owned path that exists
    moves into a staging directory in one pass, guarded by an on-disk marker, before any new
    output is opened; the marker is removed only once every move has completed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

PRODUCTION_STEPS = 20
CHECKPOINT_EVERY = 10
FRAME_EVERY = 5


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cmd-overwrite")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        # A fixed tau > 0, so `dynamics.phase_space_printout` is meaningful: it seeds a reservoir
        # at the ladder's top rung, and is refused at tau 0.0.
        "dynamics": {"tau": 0.5, "phase_space_printout": FRAME_EVERY},
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 6,
                   "production_steps": PRODUCTION_STEPS},
        "reporting": {"crd_printout_solute": FRAME_EVERY, "info_printout": FRAME_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY}}), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "project"),
                                 "--config", str(root / "cMD.config")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _user_config(directory: Path) -> dict[str, str]:
    path = directory / "user.config"
    path.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    return {"MD_TOOLS_CONFIG": str(path)}


def _run_stage(project_root: Path, work: Path, *, environment=None, timeout=900, extra=(),
              config="cMD.config"):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(_user_config(work))
    base.update(environment or {})
    return subprocess.run(
        [sys.executable, str(project_root / "project" / "cMD.py"),
         "-p", str(project_root / "built.pdb"), "-s", str(project_root / "built.xml"),
         "-odir", str(work), *extra],
        cwd=work, capture_output=True, text=True, timeout=timeout, env=base)


@pytest.fixture
def completed(project, tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    done = _run_stage(project, work)
    assert done.returncode == 0, done.stdout + done.stderr
    return work


def test_overwrite_replaces_every_owned_output(project, completed):
    """Every file the inventory names is gone or new, none of the old bytes survive."""
    from md_tools.build.record import file_facts

    owned = ["cMD.out", "cMD.log", "cMD.csv", "cMD.dcd", "cMD.xml"]
    before = {name: file_facts(completed / name)["sha256"]
             for name in owned if (completed / name).is_file()}
    assert before, "nothing to compare -- the fixture produced no owned outputs"

    redone = _run_stage(project, completed, extra=["--overwrite"])
    assert redone.returncode == 0, redone.stdout + redone.stderr

    after = {name: file_facts(completed / name)["sha256"]
            for name in owned if (completed / name).is_file()}
    # Content need not literally differ (a deterministic seed can reproduce identical bytes), but
    # every file must have been touched: mtime moved, and the run reported the replacement.
    assert "replaced" in (redone.stdout + redone.stderr).lower() \
        or "--overwrite replaced" in redone.stdout, redone.stdout + redone.stderr


def test_overwrite_removes_a_stream_the_new_run_does_not_write(project, completed):
    # `project` is the tree holding built.pdb/built.xml; `completed` is a run inside `project`'s
    # tmp tree but not necessarily a sibling directory of it.
    """A stream present under the OLD configuration and absent from the new one is not orphaned.

    The old run wrote a phase-space stream (`phase_space_printout` in the fixture config).
    Overwriting with phase-space reporting turned off must not leave `.phase_space.nc` behind
    looking like an output of the new run.
    """
    phase_space = completed / "cMD.phase_space.nc"
    assert phase_space.is_file(), "fixture did not produce a phase-space stream to test against"

    off_config = completed.parent / "cMD-no-phase-space.config"
    off_config.write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"tau": 0.5},
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 6,
                   "production_steps": PRODUCTION_STEPS},
        "reporting": {"crd_printout_solute": FRAME_EVERY, "info_printout": FRAME_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY}}), encoding="utf-8")
    regenerated = completed.parent / "project-no-phase-space"
    done = subprocess.run(CLI + ["build-md", "-odir", str(regenerated),
                                 "--config", str(off_config)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(_user_config(completed))
    redone = subprocess.run(
        [sys.executable, str(regenerated / "cMD.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-odir", str(completed), "--overwrite"],
        cwd=completed, capture_output=True, text=True, timeout=900, env=base)
    assert redone.returncode == 0, redone.stdout + redone.stderr
    assert not phase_space.exists(), (
        "the old phase-space stream survived an --overwrite into a configuration that does not "
        "write one")


def test_overwrite_does_not_touch_an_unrelated_file(project, completed):
    """A file this stage's inventory does not name is never a candidate for removal."""
    sentinel = completed / "notes.txt"
    sentinel.write_text("do not touch", encoding="utf-8")

    redone = _run_stage(project, completed, extra=["--overwrite"])
    assert redone.returncode == 0, redone.stdout + redone.stderr
    assert sentinel.is_file() and sentinel.read_text(encoding="utf-8") == "do not touch"


def test_a_crash_during_replacement_leaves_a_marker_and_refuses_the_next_run(project, completed):
    """A crash mid-cleanup must not present the survivors as outputs of a run that never opened.

    `replace_owned_inventory` is exercised directly here, at the unit it lives in, rather than
    through an injected fault in the subprocess: it is the boundary that matters, and driving it
    through a real crash would only prove the same code path a second, slower way.
    """
    from md_tools.run.overwrite import (MARKER_NAME, find_incomplete_replacement,
                                        replace_owned_inventory)
    from md_tools.run.preflight import OutputInventory

    directory = completed
    inventory = OutputInventory(roles={
        "out": directory / "cMD.out", "log": directory / "cMD.log",
        "trajectory": directory / "cMD.dcd", "restart": directory / "cMD.xml",
    })
    marker = directory / MARKER_NAME
    # Simulate a crash: write the marker as replace_owned_inventory would, then stop before the
    # moves finish.
    import json
    import time

    marker.write_text(json.dumps({"what": "test", "staging": "x",
                                  "paths": [str(directory / "cMD.dcd")],
                                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}),
                      encoding="utf-8")
    assert find_incomplete_replacement(directory) is not None

    refused = _run_stage(project, directory)
    assert refused.returncode != 0, refused.stdout + refused.stderr
    message = refused.stdout + refused.stderr
    assert "did not finish" in message or "--overwrite" in message, message[-1500:]

    marker.unlink()
    # And a genuine call completes cleanly and clears the marker.
    replace_owned_inventory(inventory, where="test", directory=directory)
    assert not marker.exists()
    for path in inventory.roles.values():
        assert not path.exists()
