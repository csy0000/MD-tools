"""AIS lambda schedules (`linear`, `tau-linear`) and evenly spaced source frames.

`tau-linear` is pinned by the physics it claims rather than by its formula alone. For alanine
dipeptide in VACUUM every atom is solute, so there is no solute-environment term, and the mixture
`(1 - lambda(t)) V0 + lambda(t) V1` with V0 the REST2 System at tau0 must EQUAL the REST2 System at
`tau = tau0 (1 - t)` at any coordinate. In solvent it cannot, and the docs say so.

Reference platform, double precision.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.ais.run import choose_frames
from md_tools.ais.schedule import check_schedule, lambda_at, switching_schedule

ALA = Path(__file__).parent / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]


def _schedule(**extra):
    return switching_schedule(switching_steps=200, parameter_update_interval_steps=1,
                              observation_interval_steps=20, timestep_fs=2.0, **extra)


# -- the lambda tables -------------------------------------------------------------------------

def test_linear_is_the_default_and_its_lambdas_are_unchanged():
    schedule = _schedule()
    assert schedule["lambda_schedule"] == "linear" and schedule["lambda_schedule_tau0"] is None
    expected = [j / 200 for j in range(201)]
    expected[0], expected[-1] = 0.0, 1.0
    assert schedule["lambdas"] == expected


@pytest.mark.parametrize("tau0", [0.2, 0.5, 0.8])
def test_tau_linear_follows_the_solute_solute_scaling_of_a_linear_tau(tau0):
    lambdas = _schedule(lambda_schedule="tau-linear", lambda_schedule_tau0=tau0)["lambdas"]
    assert lambdas[0] == 0.0 and lambdas[-1] == 1.0
    assert all(a < b for a, b in zip(lambdas, lambdas[1:]))
    for j, lam in enumerate(lambdas):
        tau = tau0 * (1.0 - j / 200)
        assert (1.0 - lam) * (1.0 - tau0) ** 2 + lam == pytest.approx((1.0 - tau) ** 2, abs=1e-12)


def test_tau_linear_at_half_is_the_documented_value():
    assert lambda_at(0.5, kind="tau-linear", tau0=0.5) == pytest.approx(5.0 / 12.0, abs=1e-15)


def test_the_schedule_digest_changes_with_the_lambdas_and_only_then():
    linear = _schedule()["lambda_sha256"]
    assert _schedule()["lambda_sha256"] == linear
    assert _schedule(lambda_schedule="tau-linear", lambda_schedule_tau0=0.5)["lambda_sha256"] \
        != linear
    assert _schedule(lambda_schedule="tau-linear", lambda_schedule_tau0=0.4)["lambda_sha256"] \
        != _schedule(lambda_schedule="tau-linear", lambda_schedule_tau0=0.5)["lambda_sha256"]


@pytest.mark.parametrize("kind, tau0, match", [
    ("linear", 0.5, "does not use it"),
    ("tau-linear", None, "needs V0's tau"),
    ("tau-linear", 0.0, "0 < tau0 < 1"),
    ("cubic", None, "the schedules are"),
])
def test_a_schedule_that_cannot_define_a_path_is_refused(kind, tau0, match):
    with pytest.raises(ValueError, match=match):
        check_schedule(kind, tau0)


# -- the physics: vacuum, where tau-linear is exact ---------------------------------------------

def test_in_vacuum_the_tau_linear_mixture_is_the_rest2_system_at_the_linear_tau():
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm.app import ForceField, HBonds, NoCutoff, PDBFile

    from md_tools.ais.two_state import TwoStateHamiltonian
    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    pdb = PDBFile(str(ALA))
    base = ForceField("amber14-all.xml").createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                                      constraints=HBonds)
    solute = solute_atom_indices(pdb.topology)
    assert len(solute) == base.getNumParticles()          # vacuum: nothing is environment
    tau0 = 0.5
    hamiltonian = TwoStateHamiltonian(build_scaled_system(base, solute, tau0), base)
    reference = Platform.getPlatformByName("Reference")
    mixed = Context(hamiltonian.system, VerletIntegrator(0.001), reference)
    mixed.setPositions(pdb.positions)
    kj = unit.kilojoule_per_mole
    lambdas = _schedule(lambda_schedule="tau-linear", lambda_schedule_tau0=tau0)["lambdas"]
    for j in (0, 37, 100, 163, 200):
        hamiltonian.set_lambda(mixed, lambdas[j])
        mixture = mixed.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
        direct = Context(build_scaled_system(base, solute, tau0 * (1 - j / 200)),
                         VerletIntegrator(0.001), reference)
        direct.setPositions(pdb.positions)
        rest2 = direct.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
        assert mixture == pytest.approx(rest2, abs=1e-6), j


# -- evenly spaced frames ----------------------------------------------------------------------

def test_evenly_spaced_covers_the_whole_window_with_distinct_frames():
    eligible = list(range(5, 100))
    chosen = choose_frames(eligible=eligible, count=64, selection="evenly_spaced",
                           allow_repeats=False, seed=1)
    assert len(set(chosen)) == 64
    assert chosen[0] == 5 and chosen[-1] == 99
    assert chosen == sorted(chosen)
    assert max(b - a for a, b in zip(chosen, chosen[1:])) <= 2


@pytest.mark.parametrize("count", [1, 2, 7, 95])
def test_evenly_spaced_is_exact_at_the_edges(count):
    eligible = list(range(5, 100))
    chosen = choose_frames(eligible=eligible, count=count, selection="evenly_spaced",
                           allow_repeats=False, seed=1)
    assert len(set(chosen)) == count and chosen[0] == 5
    if count > 1:
        assert chosen[-1] == 99
    if count == len(eligible):
        assert chosen == eligible


def test_more_paths_than_frames_is_still_refused_unless_repeats_are_allowed():
    with pytest.raises(SystemExit, match="evenly_spaced"):
        choose_frames(eligible=[0, 1, 2], count=5, selection="evenly_spaced",
                      allow_repeats=False, seed=1)
    repeated = choose_frames(eligible=[0, 1, 2], count=6, selection="evenly_spaced",
                             allow_repeats=True, seed=1)
    assert repeated == [0, 0, 1, 1, 2, 2]


# -- configuration and preflight ---------------------------------------------------------------

CHAIN = {
    "protocol": "AIS", "solvent": "implicit",
    "dynamics": {"timestep_fs": 2.0, "tau": 0.5},
    "stages": {"minimization_iterations": 50, "restrained_nvt_steps": 200,
               "restrained_npt_steps": 200, "unrestrained_npt_steps": 200,
               "production_steps": 2000},
    "reporting": {"crd_printout_whole": 200},
    "ais": {"number_of_paths": 3, "switching_steps": 200, "observation_interval_steps": 20,
            "lambda_schedule": "tau-linear"},
    "ais_source": {"generate": True},
}


def _resolve(tmp_path, document):
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "AIS.config"
    path.write_text(yaml.safe_dump(document))
    return resolve_md_config(path)


def test_a_generated_source_gives_tau_linear_its_tau0_and_a_different_one_is_refused(tmp_path):
    from md_tools.build.md import ais_lambda_schedule
    from md_tools.build.strict import ConfigError

    assert ais_lambda_schedule(_resolve(tmp_path, CHAIN)) == ("tau-linear", 0.5)
    with pytest.raises(ConfigError, match="dynamics.tau is 0.5"):
        _resolve(tmp_path, dict(CHAIN, ais=dict(CHAIN["ais"], lambda_schedule_tau0=0.4)))


def test_a_named_source_must_state_tau0(tmp_path):
    from md_tools.build.strict import ConfigError

    document = dict(CHAIN, dynamics={"timestep_fs": 2.0},
                    ais_source={"trajectory": "source.nc"})
    with pytest.raises(ConfigError, match="needs V0's tau"):
        _resolve(tmp_path, document)
    stated = dict(document, ais=dict(CHAIN["ais"], lambda_schedule_tau0=0.5))
    assert _resolve(tmp_path, stated)["ais"]["lambda_schedule_tau0"] == 0.5


def test_build_md_writes_tau0_into_the_resolved_configuration_and_the_input(tmp_path):
    from md_tools.run.inputs import parse_run_input

    from .conftest import make_dataset_root, make_states_for

    root = make_dataset_root(tmp_path)
    make_states_for(root, CHAIN)
    config = root / "AIS-run1.config"
    config.write_text(yaml.safe_dump(CHAIN))
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "AIS-run1"),
                                 "--config", str(config)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    stored = yaml.safe_load((root / "AIS-run1" / "resolved.config").read_text())
    assert stored["ais"]["lambda_schedule"] == "tau-linear"
    assert stored["ais"]["lambda_schedule_tau0"] == 0.5
    text = (root / "input" / "AIS.in").read_text()
    assert "lambda_schedule_tau0" in text
    parsed = parse_run_input(root / "input" / "AIS.in", run_config=root / "AIS-run1" / "run.config")
    assert parsed.resolved["ais"] == stored["ais"]


@pytest.fixture
def saved_state(tmp_path):
    from .conftest import make_dataset_root, make_scaled_state

    root = make_dataset_root(tmp_path)
    return root, make_scaled_state(root, tau=0.5, method="AIS")


def _claim(system0, system1, *, tau0=0.5, kind="tau-linear"):
    from md_tools.run.preflight import _ais_schedule_claim

    return _ais_schedule_claim(
        ais={"lambda_schedule": kind, "lambda_schedule_tau0": tau0},
        source_config={"generate": False}, dynamics={"tau": 0.0}, where="AIS",
        system0=system0, system1=system1)


def test_tau_linear_accepts_a_saved_state_switched_to_its_own_source(saved_state):
    root, state = saved_state
    assert _claim(state, root / "build" / "built.xml") == ("tau-linear", 0.5)


def test_tau_linear_refuses_a_v0_that_is_not_a_saved_state(saved_state):
    from md_tools.run.preflight import PreflightError

    root, _ = saved_state
    with pytest.raises(PreflightError, match="not a saved scaled state"):
        _claim(root / "build" / "built.xml", root / "build" / "built.xml")


def test_tau_linear_refuses_a_tau0_the_saved_state_does_not_have(saved_state):
    from md_tools.run.preflight import PreflightError

    root, state = saved_state
    with pytest.raises(PreflightError, match="at tau 0.5"):
        _claim(state, root / "build" / "built.xml", tau0=0.3)


def test_tau_linear_refuses_a_v1_that_is_not_the_scaled_source(saved_state, tmp_path):
    from md_tools.run.preflight import PreflightError

    root, state = saved_state
    other = tmp_path / "other.xml"
    other.write_text((root / "build" / "built.xml").read_text() + "\n")
    with pytest.raises(PreflightError, match="not the System -s was scaled from"):
        _claim(state, other)


def test_linear_never_reads_the_state_record(tmp_path):
    missing = tmp_path / "nowhere.xml"
    assert _claim(missing, missing, tau0=None, kind="linear") == ("linear", None)


def test_a_v2_run_directory_is_refused_with_the_reason(tmp_path):
    import json

    from md_tools.ais.run import require_same_run

    (tmp_path / "AIS_run.json").write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(SystemExit, match="predates lambda schedules"):
        require_same_run(tmp_path, {"schema_version": 3})
