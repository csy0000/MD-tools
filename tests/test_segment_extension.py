"""A fresh REST2 run, then two extensions, in the SAME run directory.

This is the test the segment contract exists for. Everything else about segments is arithmetic;
this is the part that can be wrong at runtime while every unit test passes -- a second invocation
that restarts the exchange sequence instead of continuing it produces a run that looks healthy,
reports a plausible acceptance rate, and is not one trajectory.

Marked slow: it prepares a real solvated system and integrates it on CPU.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("pydantic")

REPO_ROOT = Path(__file__).resolve().parents[1]
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
               / "systems" / "ace_ala_nme.pdb")

pytestmark = pytest.mark.slow


def _cli(*args, cwd: Path, timeout: int = 3600):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "md_templates.openmm.cli", *args],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)


#: The alanine worked example shrunk to CPU size. Chemistry, HMR, timestep and ladder SHAPE are the
#: example's; only the durations and the box are small enough to integrate in a test.
def _smoke_config() -> dict:
    document = json.loads((REPO_ROOT / "test" / "ala" / "REST2"
                           / "alanine_rest2.json").read_text())
    document["execution"]["platform"] = "CPU"
    document["execution"].pop("precision", None)
    document["execution"]["reporting"] = {"all_atom": "2 ps", "solute": "1 ps",
                                          "state": "1 ps", "checkpoint": "2 ps"}
    document["protocol"]["equilibration"] = {"protocol": "simple", "minimize_max_iterations": 100,
                                             "nvt": "2 ps", "npt": "2 ps", "npt_free": "2 ps"}
    document["protocol"]["production"]["tau_ladder"]["count"] = 3
    # 2 exchanges x 2 ps = a 4 ps segment, derived. duration_per_segment is not an input for REST2.
    document["protocol"]["production"]["exchange"] = {"n_exchange_per_segment": 2,
                                                      "exchange_interval": "2 ps"}
    document["build"]["solvation"]["padding"] = "0.9 nm"
    document["build"]["nonbonded"]["cutoff"] = "0.7 nm"
    return document


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> tuple[Path, Path, Path]:
    work = tmp_path_factory.mktemp("segment_extension")
    (work / "ace_ala_nme.pdb").write_text(ALANINE_PDB.read_text())
    config = work / "smoke.json"
    config.write_text(json.dumps(_smoke_config(), indent=2))

    result = _cli("prepare", "--config", "smoke.json", "--out-root", "./bundle", cwd=work)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    bundle = next((work / "bundle").iterdir())
    return work, config, bundle


def _exchange_rows(run_dir: Path) -> list[dict]:
    return list(csv.DictReader((run_dir / "exchange_attempts.csv").open()))


def test_hmr_and_the_four_site_water_survive_preparation(prepared):
    """The prepared bundle must record what was actually built, not what was requested."""
    work, _, bundle = prepared
    simbox = json.loads((bundle / "simbox.json").read_text())

    # HMR really ran: alanine dipeptide has 12 hydrogens, all in the solute.
    assert simbox["hmr"]["target_hydrogen_mass_amu"] == pytest.approx(3.024)
    assert simbox["hmr"]["n_hydrogens_repartitioned"] == 12
    assert simbox["hmr"]["scope"] == "solute"

    # OPC has no pre-equilibrated box in OpenMM, so it is packed with TIP4P-Ew geometry and
    # parameterised by its own force field. The substitution must be recorded, not inferred.
    assert simbox["water"]["model"] == "opc"
    assert simbox["water"]["packing_model"] == "tip4pew"
    assert simbox["water"]["packing_substituted"] is True
    assert simbox["forcefield"]["water"] == "amber19/opc.xml"


def test_a_fresh_run_then_two_extensions_form_one_continuous_trajectory(prepared):
    """Three invocations, one run directory, one exchange sequence."""
    work, config, bundle = prepared
    run_root = work / "rest2"

    first = _cli("rest2", "--bundle", str(bundle), "--config", str(config),
                 "--out-root", str(run_root), "--run-name", "ext", "--platform", "CPU", cwd=work)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    run_dir = run_root / "ext"
    assert len(_exchange_rows(run_dir)) == 2

    # A FRESH run must refuse to overwrite an existing directory: silently reusing it would
    # restart the exchange sequence while looking like a continuation.
    clash = _cli("rest2", "--bundle", str(bundle), "--config", str(config),
                 "--out-root", str(run_root), "--run-name", "ext", "--platform", "CPU", cwd=work)
    assert clash.returncode != 0
    assert "already exists" in (clash.stdout + clash.stderr)

    # two extensions, the same command pattern each time
    for expected_rows in (4, 6):
        extension = _cli("rest2", "--bundle", str(bundle), "--config", str(config),
                         "--out-root", str(run_root), "--resume-run", "ext",
                         "--platform", "CPU", cwd=work)
        assert extension.returncode == 0, extension.stdout[-3000:] + extension.stderr[-3000:]
        assert len(_exchange_rows(run_dir)) == expected_rows

    rows = _exchange_rows(run_dir)
    attempts = [int(r["attempt_index"]) for r in rows]
    steps = [int(r["step"]) for r in rows]
    times = [float(r["time_ps"]) for r in rows]

    assert attempts == list(range(6)), "attempt indices must continue, not restart"
    assert len(set(attempts)) == len(attempts), "duplicate attempt indices"
    assert all(b > a for a, b in zip(steps, steps[1:])), "step counter must increase monotonically"
    assert all(b > a for a, b in zip(times, times[1:])), "physical time must increase monotonically"

    # alternating nearest-neighbour phases must continue across the segment boundary
    assert [r["phase"] for r in rows] == ["0", "1", "0", "1", "0", "1"]


def test_the_exchange_history_is_appended_not_rewritten(prepared):
    """A duplicated header silently corrupts every downstream parse of this file."""
    work, _, _ = prepared
    text = (work / "rest2" / "ext" / "exchange_attempts.csv").read_text()
    assert text.count("attempt_index,step,time_ps") == 1


def test_the_committed_record_tracks_every_generation(prepared):
    """The committed generation record is the sole authority for the restart boundary."""
    work, _, _ = prepared
    restart = work / "rest2" / "ext" / "restart"
    assert (restart / "committed.json").is_file()
    generations = sorted(p.name for p in restart.glob("gen_*"))
    assert len(generations) >= 2, generations


def test_the_walker_mapping_is_persisted_across_segments(prepared):
    """Without the walker mapping, a resumed run cannot say which trajectory is which."""
    work, _, _ = prepared
    rows = _exchange_rows(work / "rest2" / "ext")
    walker_columns = [c for c in rows[0] if c.startswith("walker_at_replica_")]
    assert len(walker_columns) == 3
    for row in rows:
        assert sorted(int(row[c]) for c in walker_columns) == [0, 1, 2]
