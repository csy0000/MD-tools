"""`mode: decoupling` in `combine-topology`: the plan absolute hydration needs, from the CLI.

`md_tools.alchemy.topology.build_decoupling_plan` has existed since the plan builder did, but
until 0.6.4 `combine-topology` offered only single, hybrid and dual -- so the simplest alchemical
calculation there is, decoupling one small molecule from water, was reachable only from Python. The
nine-leg hydration campaign behind `docs/openmm_methods/alchemy/validation.md` was driven that way.

THE TEST THAT MATTERS is `test_the_cli_plan_is_the_same_plan_the_python_api_builds`: the published
hydration numbers were produced through the Python entry point, so a CLI that built a merely
similar plan would make the tutorial's commands describe a different calculation from the one the
results came from. The plans must be equal by `plan_sha256`, not by spot-checking their contents.

Endpoint B here is the ligand ABSENT, so every key that presupposes a second ligand is refused BY
NAME. Each of those refusals is asserted, because a setting accepted and inert is worse than one
refused: a `map` silently ignored under `mode: decoupling` is a mutation somebody wrote and set the
wrong mode on, and it would run as a decoupling without a word.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from md_tools.build.combine import MODES, PAIRED_MODES, combine_topology
from md_tools.build.strict import ConfigError

FX = Path(__file__).parent / "data" / "alchemy"
ENV = FX / "ethane-tip3p-v2"
ETHANE = FX / "v1/packages/LOCAL-OTMSDBZUPAUEDD/param_cf41a2bd76f4"
CHLOROETHANE = FX / "v1/packages/LOCAL-HRYZWHHZPQKTII/param_5930c10b0577"
ACETATE_ENV = FX / "charged-v1" / "acetate-tip3p"
ACETATE = FX / "charged-v1/packages/LOCAL-QTBSBXVTEAMEQO/param_c565813e02ae"

pytestmark = pytest.mark.skipif(not ENV.is_dir(), reason="alchemy fixtures are not present")


def _config(tmp_path: Path, **changes) -> Path:
    document = {
        "format": "md-tools-combine-topology/1",
        "mode": "decoupling",
        "endpoints": {"A": {"parameters": str(ETHANE.resolve())}},
        "environment": {
            "system": str((ENV / "built.xml").resolve()),
            "topology": str((ENV / "built.pdb").resolve()),
            "record": str((ENV / "built.log").resolve()),
            "ligand": {"resname": "ETA"},
        },
    }
    for dotted, value in changes.items():
        section, _, field = dotted.partition(".")
        if field:
            document.setdefault(section, {})[field] = value
        else:
            document[section] = value
    path = tmp_path / "decouple.config"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _plan(tmp_path: Path, **changes) -> dict:
    return combine_topology(config_path=_config(tmp_path, **changes),
                            out_dir=tmp_path / "plan", check=True, echo=lambda *a: None)


# ------------------------------------------------------------------ the mode exists and resolves
def test_decoupling_is_a_mode_and_is_not_a_paired_one():
    assert "decoupling" in MODES
    assert "decoupling" not in PAIRED_MODES
    assert set(PAIRED_MODES) == {"single", "hybrid", "dual"}


def test_a_decoupling_configuration_resolves_and_builds_a_plan(tmp_path):
    summary = _plan(tmp_path)
    assert summary["mode"] == "decoupling"
    assert len(summary["plan_sha256"]) == 64


def test_no_second_package_and_no_pair_count_are_reported_as_none_not_zero(tmp_path):
    """Zero mapped pairs is a true statement about a mutation sharing nothing.

    A decoupling has no map to count. Reporting 0 would let a reader compare the two as though
    they were the same measurement, which is the class of error that folding two different
    statistics into one column always is.
    """
    summary = _plan(tmp_path)
    assert summary["package_b"] is None
    assert summary["n_pairs"] is None


# ------------------------------------------------------------------------------- the real check
def test_the_cli_plan_is_the_same_plan_the_python_api_builds(tmp_path):
    """The published hydration numbers came from the Python entry point.

    If the CLI built a different plan the tutorial's commands would describe a different
    calculation from the one the results came from, and nothing would say so. Equal by digest.
    """
    from md_tools.alchemy.topology import (Environment, LigandSelector,
                                           build_decoupling_plan)
    from md_tools.ligands.package import load_package

    environment = Environment.from_files(
        ENV / "built.xml", ENV / "built.pdb", LigandSelector(resname="ETA"),
        record=ENV / "built.log")
    direct = build_decoupling_plan(load_package(ETHANE), environment)

    assert _plan(tmp_path)["plan_sha256"] == direct.sha256


def test_the_digest_discriminates(tmp_path):
    """Shown to tell two calculations apart, so the equality above is not vacuous.

    Without this, the equality would pass just as well if `plan_sha256` were a constant. A
    different ligand in a different box must give a different digest.
    """
    if not ACETATE_ENV.is_dir():
        pytest.skip("the charged fixture is not present")
    from md_tools.alchemy.topology import (Environment, LigandSelector,
                                           build_topology_plan)
    from md_tools.ligands.package import load_package

    environment = Environment.from_files(
        ACETATE_ENV / "built.xml", ACETATE_ENV / "built.pdb", LigandSelector(resname="ACT"),
        record=ACETATE_ENV / "built.log")
    # `build_topology_plan` directly, NOT `build_decoupling_plan`: acetate carries a net charge and
    # the entry point refuses it (the next test). The plan itself still builds, which is what makes
    # it usable here as a second digest.
    other = build_topology_plan(load_package(ACETATE), None, None, environment,
                                mode="decoupling")

    assert _plan(tmp_path)["plan_sha256"] != other.sha256


def test_a_charged_ligand_is_refused_through_the_cli(tmp_path):
    """The net formal charge refusal must survive the trip through the configuration layer.

    Decoupling a charged ligand changes the box's net charge, and PME's neutralising background
    then contributes a free energy needing a finite-size correction this release does not
    implement. That refusal lives in `build_decoupling_plan`, which is precisely why the CLI calls
    that entry point rather than `build_topology_plan(mode="decoupling")` -- reaching past it would
    build the plan and lose the check, and the test above shows the plan does build.
    """
    if not ACETATE_ENV.is_dir():
        pytest.skip("the charged fixture is not present")
    with pytest.raises(ConfigError) as refusal:
        combine_topology(
            config_path=_config(
                tmp_path,
                **{"endpoints.A": {"parameters": str(ACETATE.resolve())},
                   "environment.system": str((ACETATE_ENV / "built.xml").resolve()),
                   "environment.topology": str((ACETATE_ENV / "built.pdb").resolve()),
                   "environment.record": str((ACETATE_ENV / "built.log").resolve()),
                   "environment.ligand": {"resname": "ACT"}}),
            out_dir=tmp_path / "plan", check=True, echo=lambda *a: None)
    message = str(refusal.value)
    assert "net formal charge" in message
    assert "not implemented" in message


# --------------------------------------------------------------- what presupposes a second ligand
@pytest.mark.parametrize("changes, named", [
    ({"endpoints.B": {"parameters": str(CHLOROETHANE.resolve())}}, "endpoints.B"),
    ({"b_pose": "pose.sdf"}, "b_pose"),
    ({"map.file": "map.yaml"}, "map.file"),
    ({"map.automatic": True}, "map.automatic"),
    ({"dual.restraint_k_kj_mol_nm2": 1000.0}, "dual.restraint_k_kj_mol_nm2"),
])
def test_every_key_that_presupposes_a_second_ligand_is_refused_by_name(tmp_path, changes, named):
    with pytest.raises(ConfigError) as refusal:
        _plan(tmp_path, **changes)
    message = str(refusal.value)
    assert named in message
    assert "the ligand ABSENT" in message
    # The refusal says what to do instead, naming the modes that DO transform one into another.
    assert all(mode in message for mode in PAIRED_MODES)


def test_a_paired_mode_without_endpoint_b_is_refused_and_says_which_mode_has_one(tmp_path):
    """The other direction of the same rule, which the nullable field opened up.

    Making `endpoints.B` nullable for decoupling means a mutation can now omit it, so the omission
    has to be caught here rather than by a confusing failure deeper in the builder.
    """
    with pytest.raises(ConfigError) as refusal:
        _plan(tmp_path, mode="hybrid")
    message = str(refusal.value)
    assert "endpoints.B" in message
    assert "decoupling" in message
