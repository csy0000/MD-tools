"""The selective-REST2 keys of a REST2 configuration are CLAIMS about the saved states.

`md-openmm build-top --rest2-scaler` is the one place a scaled Hamiltonian is made, and the region
it scaled is recorded in `build/REST2/scaler.yaml`. `build-md` scales nothing; like
`rest2.number_of_replicas` and `rest2.tau_max`, a selector key in the REST2 configuration is
checked against that record -- through `rest2.regions.claimed_region_differences`, the one
comparison -- and refused when it disagrees. Leaving the keys out accepts the recorded region, and
`build-md.log` then says which region that is, so nobody runs a selective ladder unawares.

PLATFORM_POLICY_EXEMPTION: configuration resolution and System construction only; no Context.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .conftest import make_dataset_root, make_scaled_ladder


def _explicit_states(root: Path, **selectors) -> None:
    """Saved states whose region was CHOSEN by the scaler, from a selective scaler.config."""
    from md_tools.build.scaler import build_scaled_states

    build = root / "build"
    config = build / "scaler-REST2.config"
    config.write_text(yaml.safe_dump({
        "method": "REST2",
        "schedule": {"n_states": 4, "tau_min": 0.0, "tau_max": 0.5},
        **selectors}), encoding="utf-8")
    build_scaled_states(system_path=build / "built.xml", topology_path=build / "built.pdb",
                        config_path=config, echo=False)


def _config(root: Path, *, protocol: str = "REST2", **rest2) -> Path:
    path = root / f"{protocol}.config"
    document = {
        "protocol": protocol, "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                   "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                   "production_steps": 10},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5},
    }
    if protocol == "REST2":
        document["rest2"] = {"number_of_replicas": 4, "tau_max": 0.5,
                             "exchange_interval_steps": 5, "number_of_exchanges": 2, **rest2}
    else:
        document["rest2"] = rest2
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _generate(root: Path, **kwargs) -> Path:
    from md_tools.build.md import build_scripts

    config = _config(root, **kwargs)
    run = root / f"{config.stem}-run1"
    build_scripts(config_path=config, out_dir=run, echo=False)
    return run


def _refused(root: Path, match: str, **kwargs) -> str:
    from md_tools.build.md import ConfigError

    with pytest.raises(ConfigError, match=match) as refused:
        _generate(root, **kwargs)
    assert not list(root.glob("*-run1")), "a refused build-md must write no run directory"
    return str(refused.value)


@pytest.fixture
def explicit_root(tmp_path):
    make_dataset_root(tmp_path, solvent="explicit")
    _explicit_states(tmp_path, backbone_scaling_list=":2", sidechain_scaling_list=":2")
    return tmp_path


def test_a_claim_that_resolves_to_the_recorded_region_is_accepted(explicit_root):
    run = _generate(explicit_root, backbone_scaling_list=":2", sidechain_scaling_list=":2")
    log = (run / "build-md.log").read_text(encoding="utf-8")
    assert "explicit" in log and "ALA" in log


def test_a_claim_naming_a_different_region_is_refused_with_the_rebuild_command(explicit_root):
    message = _refused(explicit_root, "not the one", backbone_scaling_list=":2")
    assert "build-top --rest2-scaler" in message


def test_no_claim_accepts_the_record_and_the_log_names_the_region(explicit_root):
    run = _generate(explicit_root)
    log = (run / "build-md.log").read_text(encoding="utf-8")
    assert "hot region" in log
    assert "explicit" in log and "ALA" in log, "a selective ladder must not run unannounced"


def test_an_explicit_claim_against_whole_solute_states_is_refused(tmp_path):
    make_dataset_root(tmp_path, solvent="explicit")
    make_scaled_ladder(tmp_path)
    _refused(tmp_path, "legacy", backbone_scaling_list=":2")


def test_whole_solute_states_without_a_claim_say_so(tmp_path):
    make_dataset_root(tmp_path, solvent="explicit")
    make_scaled_ladder(tmp_path)
    run = _generate(tmp_path)
    assert "the whole solute (legacy)" in (run / "build-md.log").read_text(encoding="utf-8")


def test_an_invalid_claim_is_refused_as_a_claim(explicit_root):
    _refused(explicit_root, "rest2 selector claim", backbone_scaling_list=":2&:3")


def test_the_compact_ligand_form_is_refused(explicit_root):
    _refused(explicit_root, "rest2 selector claim", ligand_scaling_dict={"L01": "L01.yaml"})


def test_a_claim_under_another_protocol_is_refused_by_name(tmp_path):
    from md_tools.build.md import ConfigError

    make_dataset_root(tmp_path, solvent="explicit")
    with pytest.raises(ConfigError, match="rest2.backbone_scaling_list is set but protocol is cMD"):
        _generate(tmp_path, protocol="cMD", backbone_scaling_list=":2")
    assert not list(tmp_path.glob("*-run1"))
