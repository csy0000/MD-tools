"""`ais_source.generate: true` -- an AIS run that produces its own source ensemble.

One configuration, one `run.sh`: minimisation of the UNSCALED built System (the shared `min/`),
equilibration and a `source` production stage on V0 -- the saved scaled state
`build/AIS/system_state0.xml`, passed as `SCALED_SYSTEM` like every hot stage's -- then the
switching paths from V0 to V1 = the built System, starting from the source stage's whole-system
stream.

Also pinned here, because this chain is what exposed them: a CV-enabled run's minimisation used to
refuse at its first stage (its shared declaration named CVs its input omits), and an AIS run through
`md-run` used to resolve a CV definition and then report nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .conftest import make_dataset_root

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
REPO = Path(__file__).resolve().parents[1]

CHAIN = {
    "protocol": "AIS", "solvent": "implicit",
    "dynamics": {"timestep_fs": 2.0, "tau": 0.5},
    "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 200,
               "restrained_npt_steps": 200, "unrestrained_npt_steps": 200,
               "production_steps": 2000},
    "reporting": {"crd_printout_whole": 200},
    "ais": {"number_of_paths": 3, "switching_steps": 200, "observation_interval_steps": 20},
    "ais_source": {"generate": True},
    "collective_variables": {"generate": "all_solute_torsions", "interval_steps": 20},
}


def _resolve(tmp_path, document):
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "AIS.config"
    path.write_text(yaml.safe_dump(document))
    return resolve_md_config(path)


def _build(root, document, name="AIS-run1"):
    from .conftest import make_states_for

    make_states_for(root, document)
    config = root / f"{name}.config"
    config.write_text(yaml.safe_dump(document))
    return subprocess.run(CLI + ["build-md", "-odir", str(root / name), "--config", str(config)],
                          capture_output=True, text=True, timeout=600)


# -- configuration ------------------------------------------------------------------------------

def test_generate_with_a_named_trajectory_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    document = dict(CHAIN, ais_source={"generate": True, "trajectory": "../hot/whole_prod1.nc"})
    with pytest.raises(ConfigError, match="Give one"):
        _resolve(tmp_path, document)


def test_generate_without_a_whole_system_stream_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    document = dict(CHAIN, reporting={"crd_printout_whole": 0})
    with pytest.raises(ConfigError, match="crd_printout_whole is 0"):
        _resolve(tmp_path, document)


def test_a_source_run_that_would_not_end_on_a_frame_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    document = dict(CHAIN, reporting={"crd_printout_whole": 300})
    with pytest.raises(ConfigError, match="not a whole number"):
        _resolve(tmp_path, document)


def test_without_generate_the_source_is_still_required_and_the_refusal_names_the_option(tmp_path):
    from md_tools.build.strict import ConfigError

    document = dict(CHAIN, ais_source={})
    with pytest.raises(ConfigError, match="generate: true"):
        _resolve(tmp_path, document)


def test_tau_is_accepted_for_ais_only_as_the_claim_about_a_generated_source(tmp_path):
    from md_tools.build.strict import ConfigError

    resolved = _resolve(tmp_path, CHAIN)
    assert resolved["dynamics"]["tau"] == 0.5
    with pytest.raises(ConfigError, match="protocol is AIS"):
        _resolve(tmp_path, dict(CHAIN, ais_source={"trajectory": "source.nc"}))


def test_a_generated_source_with_no_tau_claim_is_refused(tmp_path):
    """At tau 0 the source stages would run the built System, which is V1: a switch to itself."""
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="dynamics.tau is 0"):
        _resolve(tmp_path, dict(CHAIN, dynamics={"timestep_fs": 2.0}))


def test_the_plan_is_min_equilibration_and_a_fixed_volume_source_under_a_tau_claim(tmp_path):
    from md_tools.build.md import stage_plan

    resolved = _resolve(tmp_path, dict(CHAIN, solvent="explicit"))
    plan = stage_plan(resolved)
    assert [s["name"] for s in plan] == ["min", "eq_nvt_posres", "eq_nvt_posres_2",
                                         "eq_nvt_free", "source"]
    assert {s["ensemble"] for s in plan} == {"NVT"}
    assert plan[-1]["whole_interval_steps"] == 200


def test_a_named_source_still_has_no_chain(tmp_path):
    from md_tools.build.md import stage_plan

    assert stage_plan(_resolve(tmp_path, dict(CHAIN, dynamics={"timestep_fs": 2.0},
                                              ais_source={"trajectory": "s.nc"}))) == []


# -- generation ---------------------------------------------------------------------------------

@pytest.fixture
def generated(tmp_path):
    root = make_dataset_root(tmp_path)
    done = _build(root, CHAIN)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def test_one_run_sh_minimises_the_built_system_and_runs_everything_else_on_v0(generated):
    text = (generated / "AIS-run1" / "run.sh").read_text()
    calls = [block for block in text.split('echo "== ')[1:]]
    names = [block.split(" ==")[0] for block in calls]
    assert names == ["min", "eq_1", "eq_2", "eq_3", "source", "AIS"]
    assert '-s "${SYSTEM}"' in calls[0] and "SCALED_SYSTEM" not in calls[0]
    for block in calls[1:5]:
        assert '-s "${SCALED_SYSTEM}"' in block
    assert '-s "${SCALED_SYSTEM}" -p2 "${TOPOLOGY}" -s2 "${SYSTEM}"' in calls[5]
    assert "-source-traj" not in calls[5]
    assert 'SCALED_SYSTEM="${HERE}/../build/AIS/system_state0.xml"' in text


def test_the_resolved_configuration_names_the_source_stage_stream(generated):
    from md_tools.build.md import GENERATED_SOURCE_TRAJECTORY

    stored = yaml.safe_load((generated / "AIS-run1" / "resolved.config").read_text())
    assert stored["ais_source"] == dict(stored["ais_source"], generate=True,
                                        trajectory=GENERATED_SOURCE_TRAJECTORY)


@pytest.mark.parametrize("name", ["AIS.in", "source.in", "eq_3.in"])
def test_every_generated_input_resolves_back_to_its_declaration(generated, name):
    from md_tools.run.inputs import parse_run_input

    run = generated / "AIS-run1"
    declaration = run / ("eq/resolved.config" if name.startswith("eq_") else "resolved.config")
    stored = yaml.safe_load(declaration.read_text())
    config = run / ("eq/run.config" if name.startswith("eq_") else "run.config")
    parsed = parse_run_input(generated / "input" / name, run_config=config)
    assert {k: v for k, v in parsed.resolved.items() if k != "schema_version"} == \
        {k: v for k, v in stored.items() if k != "schema_version"}


def test_the_shared_minimisation_declares_no_collective_variables(generated):
    stored = yaml.safe_load((generated / "min" / "resolved.config").read_text())
    assert stored["collective_variables"]["file"] is None
    assert stored["collective_variables"]["interval_steps"] == 0


# -- the chain, run -----------------------------------------------------------------------------

@pytest.mark.slow
def test_the_whole_chain_runs_and_the_paths_start_from_a_verified_v0(tmp_path):
    """CPU, because this checks the CHAIN's wiring, not a GPU claim: every stage completes on the
    saved tau-0.5 state, the source is confirmed as V0 by its recorded digest, and the AIS CV table
    reports every torsion."""
    root = make_dataset_root(tmp_path)
    done = _build(root, CHAIN)
    assert done.returncode == 0, done.stdout + done.stderr

    run = root / "AIS-run1"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO / "src"), os.environ.get("PYTHONPATH", "")]))
    ran = subprocess.run(["./run.sh", "--cpu"],
                         cwd=run, capture_output=True, text=True, timeout=1500, env=env)
    assert ran.returncode == 0, ran.stdout[-4000:] + ran.stderr[-4000:]

    log = (run / "AIS.log").read_text()
    assert "V0, CONFIRMED" in log
    header = (run / "AIS_cv.csv").read_text().splitlines()[0].split(",")
    definition = yaml.safe_load(next(run.glob("cv.*.yaml")).read_text())
    assert len(header) == 7 + len(definition["collective_variables"])
    rows = (run / "AIS_cv.csv").read_text().splitlines()[1:]
    assert len(rows) == 3 * 11
