"""`md-openmm combine-topology`: the command surface over S2's callable topology layer.

The command adds no construction of its own: it resolves the two packages through the one package
resolver, reads the environment, takes or proposes the map, and hands all of it to
`build_topology_plan`. So the load-bearing assertions here are that it produces exactly the plan
the callable layer produces, that `--check` creates nothing, that a proposed map written for review
reproduces its plan, and that every accepted key does its job or is refused by name.

The machine's $MD_DATA is never read: the catalog is the fixture's own package directory, named
explicitly, and MD_DATA is SET to an empty temporary root (unsetting it would fall back to the user
configuration and find the machine's root again).

PLATFORM_POLICY_EXEMPTION: plan construction only; its checks run on Reference by constant.
"""
from __future__ import annotations

import yaml
import pytest

from tests.alchemy_fixtures import (CHLOROETHANE, CORE, ETHANE, FIXTURE_ROOT, PACKAGES,
                                    core_map, package, water_environment)


@pytest.fixture(autouse=True)
def _isolated_md_data(tmp_path, monkeypatch):
    root = tmp_path / "md_data"
    root.mkdir()
    monkeypatch.setenv("MD_DATA", str(root))
    yield
    assert not any(root.iterdir()), "combine-topology wrote into $MD_DATA"


def _config(tmp_path, **overrides):
    document = {
        "format": "md-tools-combine-topology/1",
        "mode": "hybrid",
        "endpoints": {"A": {"parameters": ETHANE}, "B": {"parameters": CHLOROETHANE}},
        "environment": {"system": str(FIXTURE_ROOT / "ethane-tip3p" / "built.xml"),
                        "topology": str(FIXTURE_ROOT / "ethane-tip3p" / "built.pdb"),
                        "record": str(FIXTURE_ROOT / "ethane-tip3p" / "built.log"),
                        "ligand": {"resname": "ETA"}},
        "map": {"file": "core.map.yaml"},
        "ligand_catalog": {"path": str(PACKAGES)},
    }
    for key, value in overrides.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    (tmp_path / "core.map.yaml").write_text(
        yaml.safe_dump({"pairs": [[name, name] for name in CORE]}), encoding="utf-8")
    path = tmp_path / "combine.config"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _run(config, out_dir, check=False):
    from md_tools.build.combine import combine_topology

    return combine_topology(config_path=config, out_dir=out_dir, check=check,
                            echo=lambda *_: None)


def test_the_command_writes_exactly_the_plan_the_callable_layer_builds(tmp_path):
    from md_tools.alchemy.topology import build_topology_plan, load_plan

    a, b = package(ETHANE), package(CHLOROETHANE)
    direct = build_topology_plan(a, b, core_map(a, b), water_environment(), mode="hybrid")
    summary = _run(_config(tmp_path), tmp_path / "plan")
    assert summary["plan_sha256"] == direct.sha256
    assert load_plan(tmp_path / "plan", package_roots=[PACKAGES]).sha256 == direct.sha256


def test_check_builds_the_plan_and_creates_nothing(tmp_path):
    out = tmp_path / "not-yet" / "plan"
    summary = _run(_config(tmp_path), out, check=True)
    assert summary["check"] and summary["plan_sha256"]
    assert not (tmp_path / "not-yet").exists(), "--check must not even create the parent"


def test_a_proposed_map_is_written_beside_the_plan_and_reproduces_it(tmp_path):
    config = _config(tmp_path, map={"automatic": True})
    first = _run(config, tmp_path / "plan")
    sidecar = tmp_path / "plan.map.yaml"
    assert first["proposed_map"] == str(sidecar) and sidecar.is_file()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("plan")) == \
        ["plan", "plan.map.yaml"]
    again = _run(_config(tmp_path, map={"file": str(sidecar)}), tmp_path / "plan2")
    assert again["plan_sha256"] == first["plan_sha256"]


def test_an_edited_proposal_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    _run(_config(tmp_path, map={"automatic": True}), tmp_path / "plan")
    sidecar = tmp_path / "plan.map.yaml"
    document = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    first = next(iter(document["map"]["a_to_b"]))
    document["map"]["a_to_b"].pop(first)
    sidecar.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigError, match="combine-topology"):
        _run(_config(tmp_path, map={"file": str(sidecar)}), tmp_path / "plan2")
    assert not (tmp_path / "plan2").exists()


@pytest.mark.parametrize("overrides, words", [
    ({"mode": "separated"}, "separated is not implemented"),
    ({"mode": "single", "map": {"automatic": True}}, "refused for mode: single"),
    ({"map": {"file": "core.map.yaml", "automatic": True}}, "exactly one of"),
    ({"map": {}}, "exactly one of"),
    ({"dual": {"restraint_k_kj_mol_nm2": 500.0}}, "but mode is hybrid"),
    ({"format": "md-tools-combine-topology/0"}, "format"),
    ({"surprise": 1}, "surprise"),
    ({"endpoints": {"A": {"parameters": ETHANE, "aliases": ["x"]},
                    "B": {"parameters": CHLOROETHANE}}}, "exactly one key"),
])
def test_every_refusal_names_its_cause_and_writes_nothing(tmp_path, overrides, words):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match=words):
        _run(_config(tmp_path, **overrides), tmp_path / "plan")
    assert not (tmp_path / "plan").exists()
    assert not (tmp_path / "plan.map.yaml").exists()


def test_an_existing_output_or_proposal_is_never_overwritten(tmp_path):
    from md_tools.build.strict import ConfigError

    (tmp_path / "plan").mkdir()
    with pytest.raises(ConfigError, match="already exists"):
        _run(_config(tmp_path), tmp_path / "plan")
    (tmp_path / "other.map.yaml").write_text("keep me\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="already exists"):
        _run(_config(tmp_path, map={"automatic": True}), tmp_path / "other")
    assert (tmp_path / "other.map.yaml").read_text(encoding="utf-8") == "keep me\n"
    assert not (tmp_path / "other").exists()


def test_the_cli_exit_codes(tmp_path, capsys):
    from md_tools.cli.md_openmm import main

    assert main(["combine-topology", "--config", str(_config(tmp_path)),
                 "-odir", str(tmp_path / "plan")]) == 0
    assert (tmp_path / "plan").is_dir()
    assert main(["combine-topology", "--config", str(_config(tmp_path, mode="separated")),
                 "-odir", str(tmp_path / "plan2")]) == 2
    assert "separated is not implemented" in capsys.readouterr().err
    assert main(["combine-topology", "--config", str(_config(tmp_path)),
                 "-odir", str(tmp_path / "plan3"), "--check"]) == 0
    assert not (tmp_path / "plan3").exists()
