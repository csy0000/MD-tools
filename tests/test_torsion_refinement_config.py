"""`solute.torsion_refinement` and its environment keys: what resolves, and what is refused.

Each refusal below can fail by a different mistake. A configuration that names a method with no
environment must not be accepted (the fit would then have nothing to run, and the alternative -- the
stock force field -- would be a silent substitution); an environment named for the other family, or
beside `none`, is a mistake the user can fix; and a reused package must not be combined with a fit
that could only be ignored.

PLATFORM_POLICY_EXEMPTION: configuration resolution only. Nothing on disk is read and no
environment is run.
"""
from __future__ import annotations

import pytest
import yaml

from md_tools.build.strict import ConfigError


def _resolve(tmp_path, solute):
    from md_tools.build.top import resolve_build_config

    path = tmp_path / "build-top.config"
    path.write_text(yaml.safe_dump({"solute": solute}), encoding="utf-8")
    return resolve_build_config(path)


def test_the_default_is_no_refinement_and_names_no_environment(tmp_path):
    solute = _resolve(tmp_path, {"kind": "ligand"})["solute"]
    assert solute["torsion_refinement"] == "none"
    assert solute["bespokefit_env"] is None and solute["mace_env"] is None
    assert solute["mace_model"] is None and solute["refinement_offxml"] is None


@pytest.mark.parametrize("method, env_key", [
    ("bespokefit", "bespokefit_env"),
    ("mace-omol-bespokefit", "mace_env"),
    ("mace-off23-bespokefit", "mace_env"),
])
@pytest.mark.parametrize("kind", ["ligand", "peptide-like"])
def test_each_method_resolves_with_its_own_environment(tmp_path, method, env_key, kind):
    solute = _resolve(tmp_path, {"kind": kind, "torsion_refinement": method,
                                 env_key: "/opt/envs/bespoke"})["solute"]
    assert solute["torsion_refinement"] == method and solute[env_key] == "/opt/envs/bespoke"


@pytest.mark.parametrize("solute, words", [
    ({"kind": "ligand", "torsion_refinement": "bespokefit"}, "bespokefit_env"),
    ({"kind": "ligand", "torsion_refinement": "mace-omol-bespokefit"}, "mace_env"),
    ({"kind": "ligand", "torsion_refinement": "mace-off23-bespokefit", "mace_env": " "},
     "mace_env"),
    ({"kind": "ligand", "torsion_refinement": "mace-omol-bespokefit", "mace_env": "/e",
      "bespokefit_env": "/p"}, "bespokefit_env is set"),
    ({"kind": "ligand", "torsion_refinement": "bespokefit", "bespokefit_env": "/e",
      "mace_env": "/m"}, "mace_env is set"),
    ({"kind": "ligand", "mace_env": "/m"}, "torsion_refinement is 'none'"),
    ({"kind": "ligand", "refinement_offxml": "bespoke.offxml"}, "torsion_refinement is 'none'"),
    ({"kind": "peptide", "torsion_refinement": "bespokefit", "bespokefit_env": "/e"},
     "single molecule"),
    ({"kind": "ligand", "torsion_refinement": "bespokefit", "bespokefit_env": "/e",
      "parameters": "CHEMBL112/param_e932f4c4f371"}, "keeps its own parameters"),
    ({"kind": "ligand", "torsion_refinement": "bespokefit", "bespokefit_env": "/e",
      "mace_model": "large"}, "only to the mace"),
    ({"kind": "ligand", "torsion_refinement": "mace-omol-bespokefit", "mace_env": "/e",
      "mace_model": "large"}, "not a MACE-OMOL model"),
    ({"kind": "ligand", "torsion_refinement": "mace-off23-bespokefit", "mace_env": "/e",
      "mace_model": "extra_large"}, "not a MACE-OFF23 model"),
    ({"kind": "ligand", "torsion_refinement": "mace-omol-gap", "mace_env": "/e"},
     "torsion_refinement"),
])
def test_a_setting_that_cannot_apply_is_refused(tmp_path, solute, words):
    with pytest.raises(ConfigError, match=words):
        _resolve(tmp_path, solute)


def test_a_fitted_offxml_stands_in_for_the_environment(tmp_path):
    solute = _resolve(tmp_path, {"kind": "ligand", "torsion_refinement": "mace-omol-bespokefit",
                                 "refinement_offxml": "bespoke.offxml"})["solute"]
    assert solute["refinement_offxml"] == "bespoke.offxml" and solute["mace_env"] is None


def test_a_model_file_is_accepted_for_either_mace_reference(tmp_path):
    for method in ("mace-omol-bespokefit", "mace-off23-bespokefit"):
        solute = _resolve(tmp_path, {"kind": "ligand", "torsion_refinement": method,
                                     "mace_env": "/e", "mace_model": "/m/custom.model"})["solute"]
        assert solute["mace_model"] == "/m/custom.model"
