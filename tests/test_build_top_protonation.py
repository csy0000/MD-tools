"""build-top's `protonation:` and `input.assembly` settings: refusals, and a real PROPKA build.

The refusals run through `resolve_build_config` and `build_topology`, so each is shown to stop the
build before any output exists. The slow test builds ACE-ALA-NME in explicit water with
`protonation.method: propka` and reads the machine record back; the default build is shown to
record `method: openmm`, which is what every earlier build meant.

PLATFORM_POLICY_EXEMPTION: configuration resolution, and a build-top run whose hydrogen relaxation
uses the Reference platform by design. No dynamics.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.strict import ConfigError

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]


def _config(tmp_path, document) -> Path:
    path = tmp_path / "build-top.config"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _resolve(tmp_path, document):
    from md_tools.build.top import resolve_build_config

    return resolve_build_config(_config(tmp_path, document))


@pytest.mark.parametrize("document, words", [
    ({"solute": {"kind": "ligand"}, "input": {"assembly": "1"}}, "input.assembly"),
    ({"solute": {"kind": "ligand"}, "protonation": {"method": "propka"}}, "nothing titrates it"),
    ({"solute": {"kind": "peptide"}, "solvent": {"model": "GBn2"},
      "protonation": {"method": "propka"}}, "implicit"),
    ({"solute": {"kind": "peptide"},
      "protonation": {"overrides": [{"select": {"chain": "A", "resid": "1"}, "variant": "HIZ"}]}},
     "not one addHydrogens can build"),
    ({"solute": {"kind": "peptide"}, "protonation": {"method": "pka-guess"}}, "method"),
])
def test_a_setting_that_cannot_apply_is_refused(tmp_path, document, words):
    with pytest.raises(ConfigError, match=words):
        _resolve(tmp_path, document)


def test_the_defaults_mean_what_every_earlier_build_meant(tmp_path):
    resolved = _resolve(tmp_path, {"solute": {"kind": "peptide"}})
    assert resolved["protonation"]["method"] == "openmm"
    assert resolved["protonation"]["ph"] == 7.0
    assert resolved["input"]["assembly"] is None


def test_an_assembly_of_a_pdb_file_is_refused_before_any_output(tmp_path):
    from md_tools.build.top import build_topology

    config = _config(tmp_path, {"solute": {"kind": "peptide"}, "input": {"assembly": "1"}})
    out = tmp_path / "out"
    with pytest.raises(ConfigError, match="must be a .cif"):
        build_topology(input_path=ALA, config_path=config, out_system=out / "built.xml",
                       out_pdb=out / "built.pdb", out_log=out / "built.log", echo=False)
    assert not out.exists()


def test_an_unknown_assembly_is_refused_before_any_output(tmp_path):
    from md_tools.build.top import build_topology

    from .test_assembly_expansion import _cif

    config = _config(tmp_path, {"solute": {"kind": "peptide"}, "input": {"assembly": "7"}})
    out = tmp_path / "out"
    with pytest.raises(ConfigError, match="has no assembly '7'"):
        build_topology(input_path=_cif(tmp_path), config_path=config,
                       out_system=out / "built.xml", out_pdb=out / "built.pdb",
                       out_log=out / "built.log", echo=False)
    assert not out.exists()


def test_a_ligand_refusal_names_the_input_the_caller_typed(tmp_path):
    """Not the temporary copy: `input.assembly` and `input.missing_atoms` read a staged file in a
    private directory that is gone by the time anybody reads the refusal, and a refusal nobody can
    trace back to their own `-i` is close to useless. This is the refusal a person meets when
    reusing a parameter package.
    """
    pytest.importorskip("pdbfixer")
    from md_tools.build.top import build_topology

    from .test_assembly_expansion import _cif

    source = _cif(tmp_path)
    config = _config(tmp_path, {
        "solute": {"kind": "complex"},
        "ligands": [{"select": {"chain": "A", "resid": "31"},
                     "parameters": "CHEMBL999/param_000000000000"}],
        "input": {"assembly": "1", "missing_atoms": "add"}})
    out = tmp_path / "out"
    with pytest.raises(ConfigError) as refusal:
        build_topology(input_path=source, config_path=config, out_system=out / "built.xml",
                       out_pdb=out / "built.pdb", out_log=out / "built.log", echo=False)
    message = str(refusal.value)
    assert f"-i {source}" in message
    assert "after input.assembly: 1" in message
    assert "build-top-assembly-" not in message
    assert not out.exists()


@pytest.mark.slow
@pytest.mark.parametrize("method", ["openmm", "propka"])
def test_the_build_records_how_the_protein_was_protonated(tmp_path, method):
    pytest.importorskip("propka")
    from md_tools.build.record import read_record

    config = _config(tmp_path, {"solute": {"kind": "peptide"},
                                "protonation": {"method": method}})
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "out/built.xml", "-op", "out/built.pdb",
               "-log", "out/built.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    record = read_record(tmp_path / "out" / "built.log")
    protonation = record["protonation"]
    assert protonation["method"] == method
    if method == "propka":
        assert protonation["propka"]["version"]
        assert protonation["propka"]["input_sha256"]
    else:
        assert protonation["propka"] is None
    assert "protonation  : method " + method in (tmp_path / "out" / "built.log").read_text()
