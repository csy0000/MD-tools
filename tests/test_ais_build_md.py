"""AIS through the real `md-openmm build-md` interface.

These tests invoke the actual command, not a helper that bypasses it. AIS used to be reachable only
through `sys-config` + `md-gen`, and that retired route was the sole reason contract v1 and the old
generator survived; a test that reached past the CLI would have let the same gap reopen quietly.

The scientific contract asserted here is the one the previous implementation established:
tau-only coordinates, an exact integer-step schedule including both endpoints, observation 0 before
any work, independent per-path seeds, fixed volume, and a completed path that is never appended to.
"""
from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.md import resolve_md_config, stage_plan
from md_tools.build.strict import ConfigError

REPO = Path(__file__).resolve().parents[1]


def _build_md(work: Path, config: dict, *extra: str):
    """Run the REAL command, the way the documentation tells a user to."""
    path = work / "AIS.config"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md",
         "-odir", "./md_script/", "--config", str(path), *extra],
        cwd=work, capture_output=True, text=True, timeout=600)


def _base(**patch) -> dict:
    config = {
        "protocol": "AIS",
        "solvent": "implicit",
        "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "seed": 7},
        "ais": {"number_of_paths": 3, "tau_start": 0.5, "tau_end": 0.0,
                "switching_steps": 200, "observation_interval_steps": 40,
                "parameter_update_interval_steps": 1},
        "ais_source": {"trajectory": "../source.dcd"},
    }
    for key, value in patch.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    return config


# --- the public interface ---------------------------------------------------------------------

def test_ais_is_a_build_md_protocol_and_not_a_command_of_its_own():
    """The whole point of the migration: no `build-ais`, and no `ais-run`.

    `md-run` joined the surface later and is not a counterexample: it runs whatever protocol its
    input declares, AIS included, so AIS still has no command that is only for AIS.
    """
    from md_tools.cli.md_openmm import build_parser

    names = set()
    for action in build_parser()._actions:
        if getattr(action, "choices", None):
            names |= {str(k) for k in action.choices}
    assert names == {"build-top", "build-md", "md-run", "data-register"}, names
    assert not [name for name in names if "ais" in name.lower()], names

    from md_tools.build.md import PROTOCOLS
    assert "AIS" in PROTOCOLS


def test_build_md_generates_an_ais_bundle(tmp_path):
    result = _build_md(tmp_path, _base())
    assert result.returncode == 0, result.stdout + result.stderr
    written = {p.name for p in (tmp_path / "md_script").iterdir()}
    assert {"AIS.py", "run.sh", "resolved.config"} <= written, written


def test_ais_has_no_equilibration_chain(tmp_path):
    """AIS consumes an ensemble that already exists; generating one would run it too early."""
    result = _build_md(tmp_path, _base())
    assert result.returncode == 0, result.stdout + result.stderr
    written = {p.name for p in (tmp_path / "md_script").iterdir()}
    for stage in ("min.py", "eq_nvt_posres.py", "eq_npt_posres.py", "eq_npt_free.py", "cMD.py"):
        assert stage not in written, f"AIS generated {stage}; it has no stage chain"
    assert stage_plan(resolve_md_config(None) | {"protocol": "AIS"}) is not None


def test_the_generated_script_imports_the_installed_runtime_and_names_no_absolute_path(tmp_path):
    _build_md(tmp_path, _base())
    text = (tmp_path / "md_script" / "AIS.py").read_text(encoding="utf-8")
    assert "from md_tools.ais import run_generated_ais" in text
    assert "run_generated_ais(__file__)" in text
    assert str(REPO) not in text, "the generated script names the source checkout"
    for line in text.splitlines():
        if line.strip().startswith("#") or '"""' in line:
            continue
        assert "/data3" not in line and "/home/" not in line, line
    compile(text, "AIS.py", "exec")


def test_run_sh_requires_the_source_explicitly(tmp_path):
    """A wrong source is not a slower run; it is a different measurement."""
    _build_md(tmp_path, _base())
    run_sh = (tmp_path / "md_script" / "run.sh").read_text(encoding="utf-8")
    assert "SOURCE" in run_sh

    # Real topology and system files, so the SOURCE check is the one under test rather than the
    # earlier existence checks.
    (tmp_path / "a.pdb").write_text("END\n")
    (tmp_path / "b.xml").write_text("<System/>\n")
    result = subprocess.run(["bash", "run.sh", "../a.pdb", "../b.xml"],
                            cwd=tmp_path / "md_script", capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "SOURCE_TRAJECTORY" in combined, combined
    assert "cannot generate one" in combined, combined


# --- coordinates and the schedule ---------------------------------------------------------------

def test_the_public_coordinate_is_tau_and_there_is_no_second_one(tmp_path):
    """`s` may be derived inside the scaler; it is not an alternative persisted coordinate."""
    _build_md(tmp_path, _base())
    resolved = yaml.safe_load((tmp_path / "md_script" / "resolved.config").read_text())
    assert "tau_start" in resolved["ais"] and "tau_end" in resolved["ais"]
    text = yaml.safe_dump(resolved["ais"])
    for forbidden in ("\ns:", "sqrt_s", "lambda"):
        assert forbidden not in text, f"the AIS config exposes {forbidden!r} as a coordinate"

    from md_tools.ais.run import OBSERVATION_COLUMNS
    assert "tau" in OBSERVATION_COLUMNS
    assert "s" not in OBSERVATION_COLUMNS and "sqrt_s" not in OBSERVATION_COLUMNS


def test_every_length_is_an_integer_step_count(tmp_path):
    _build_md(tmp_path, _base())
    resolved = yaml.safe_load((tmp_path / "md_script" / "resolved.config").read_text())
    for key in ("switching_steps", "observation_interval_steps",
                "parameter_update_interval_steps"):
        assert isinstance(resolved["ais"][key], int), key
    text = yaml.safe_dump(resolved["ais"])
    assert "_ps" not in text and "_ns" not in text, (
        "an AIS length is expressed as a duration; every length is an exact step count")


def test_the_schedule_includes_both_endpoints_and_observation_zero_precedes_work():
    from md_tools.ais import switching_schedule

    schedule = switching_schedule(tau_start=0.5, tau_end=0.0, switching_steps=200,
                                  parameter_update_interval_steps=1,
                                  observation_interval_steps=40, timestep_fs=2.0)
    assert schedule["number_of_observations"] == 6
    assert schedule["observations"][0]["tau"] == 0.5
    assert schedule["observations"][-1]["tau"] == 0.0
    assert schedule["observation_zero_precedes_all_work"] is True


@pytest.mark.parametrize("patch, expected", [
    ({"ais": {"tau_start": 0.5, "tau_end": 0.5}}, "never changes"),
    ({"ais": {"switching_steps": 15, "observation_interval_steps": 2}},
     "observation_interval_steps"),
    ({"ais": {"switching_steps": 12, "parameter_update_interval_steps": 2,
              "observation_interval_steps": 3}}, "would not land on the"),
    ({"ais_source": {"first_frame": 90, "last_frame": 10}}, "before first_frame"),
])
def test_an_inconsistent_schedule_is_refused_rather_than_rounded(tmp_path, patch, expected):
    result = _build_md(tmp_path, _base(**patch))
    assert result.returncode != 0
    assert expected in result.stdout + result.stderr


def test_a_missing_source_is_refused(tmp_path):
    config = _base()
    config["ais_source"] = {"trajectory": None}
    result = _build_md(tmp_path, config)
    assert result.returncode != 0
    assert "required for AIS" in result.stdout + result.stderr


def test_a_source_on_a_non_ais_protocol_is_refused(tmp_path):
    config = _base(protocol="cMD")
    result = _build_md(tmp_path, config)
    assert result.returncode != 0
    assert "only consumed by AIS" in result.stdout + result.stderr


def test_asking_for_more_paths_than_frames_is_refused_not_silently_reused():
    """Two paths from one configuration are not two independent realisations."""
    from md_tools.ais.run import choose_frames

    with pytest.raises(SystemExit, match="independent realisations"):
        choose_frames(eligible=[0, 1, 2], count=10, selection="uniform_random",
                      allow_repeats=False, seed=1)
    repeated = choose_frames(eligible=[0, 1, 2], count=10, selection="uniform_random",
                             allow_repeats=True, seed=1)
    assert len(repeated) == 10


def test_paths_get_independent_deterministic_seeds():
    from md_tools.md import derive_seed

    seeds = {(derive_seed(7, "ais", i, "integrator"), derive_seed(7, "ais", i, "velocity"))
             for i in range(20)}
    assert len(seeds) == 20, "two paths share a seed pair"
    assert derive_seed(7, "ais", 3, "integrator") == derive_seed(7, "ais", 3, "integrator")


# --- the scientific contract, on CUDA -------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_ais_runs_through_the_real_cli_and_keeps_its_work_contract(tmp_path):
    """A whole AIS run, driven exactly as the documentation says, on CUDA.

    Shortened but meaningful: real switching, real work accumulation, real independent paths. What
    is asserted is the contract, not a number -- the work value depends on the system.
    """
    from md_tools.build.record import read_record

    source = REPO / "tests" / "data"
    ala = source / "ALA.pdb"
    if not ala.is_file():
        pytest.skip("no ALA fixture")

    work = tmp_path
    build = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ala),
         "-os", "built.xml", "-op", "built.pdb", "-log", "built.log",
         "--config", str(_implicit_config(work))],
        cwd=work, capture_output=True, text=True, timeout=1800)
    assert build.returncode == 0, build.stdout + build.stderr

    # A short fixed-tau source ensemble at tau = 0.5: the ensemble AIS anneals away from.
    hot = work / "hot.config"
    hot.write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "dynamics": {"platform": "CUDA", "tau": 0.5, "seed": 3},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
                   "production_steps": 200},
        "reporting": {"solute_printout": 20, "system_printout": 100,
                      "checkpoint_printout": 200},
    }, sort_keys=False), encoding="utf-8")
    assert subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", "./hot",
         "--config", str(hot)], cwd=work, capture_output=True, text=True).returncode == 0
    ran = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml"],
                         cwd=work / "hot", capture_output=True, text=True, timeout=3600)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]

    result = _build_md(work, _base(ais_source={"trajectory": "../hot/cMD.dcd"},
                                   dynamics={"platform": "CUDA"}))
    assert result.returncode == 0, result.stdout + result.stderr

    ais = subprocess.run(["bash", "run.sh", "../built.pdb", "../built.xml", "../hot/cMD.dcd"],
                         cwd=work / "md_script", capture_output=True, text=True, timeout=3600)
    assert ais.returncode == 0, ais.stdout[-3000:] + ais.stderr[-3000:]

    record = read_record(work / "md_script" / "AIS.log")
    assert record["status"] == "completed"
    assert record["platform"]["name"] in (None, "CUDA")
    assert record["thermodynamic_states"]["source_tau"] == 0.5
    assert record["thermodynamic_states"]["target_tau"] == 0.0
    assert "no barostat" in record["thermodynamic_states"]["ensemble"]
    assert "U(tau_{j+1}, x_j) - U(tau_j, x_j)" in record["work_convention"]

    totals = []
    for index in range(3):
        directory = work / "md_script" / f"path_{index:04d}"
        rows = list(csv.DictReader((directory / "observations.csv").open()))
        assert len(rows) == 6, rows

        # Observation 0 is the source configuration, before any parameter change: exactly zero.
        assert float(rows[0]["cumulative_work_kj_mol"]) == 0.0
        assert float(rows[0]["tau"]) == 0.5
        assert float(rows[-1]["tau"]) == 0.0

        # tau decreases monotonically and the cumulative work is the running sum of increments.
        taus = [float(r["tau"]) for r in rows]
        assert taus == sorted(taus, reverse=True)
        running = 0.0
        for row in rows[1:]:
            running += float(row["incremental_work_kj_mol"])
            assert float(row["cumulative_work_kj_mol"]) == pytest.approx(running, rel=1e-9)

        # tau is the only persisted coordinate.
        assert "s" not in rows[0] and "sqrt_s" not in rows[0]

        completion = json.loads((directory / "completed.json").read_text())
        assert completion["status"] == "completed"
        totals.append(completion["total_work_kj_mol"])

    # Independent paths from different source frames give different work: identical values would
    # mean the paths were not independent realisations.
    assert len({round(value, 6) for value in totals}) == len(totals), totals


def _implicit_config(work: Path) -> Path:
    path = work / "build.config"
    path.write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    return path
