"""The selective-REST2 examples in the documentation run through the real resolvers.

`docs/openmm_methods/REST2/selective-scaler.config` is loaded by the strict `SCALER_SCHEMA` and
then builds real states from ACE-ALA-NME, the structure it says it is written for. The YAML blocks
of the REST2 page's "Selective REST2" section go through the same parsers a user's file would:
the selectors through `parse_selectors`, the exclusion file through `load_torsion_exclusions`.
An example that does not resolve is copied, fails, and reads as a broken tool.

PLATFORM_POLICY_EXEMPTION: configuration parsing and System construction only; no Context.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from md_tools.build.scaler import SCALER_SCHEMA, build_scaled_states
from md_tools.rest2.regions import has_selectors, load_torsion_exclusions, parse_selectors

from .conftest import make_dataset_root

REST2 = Path(__file__).resolve().parents[1] / "docs" / "openmm_methods" / "REST2"
EXAMPLE = REST2 / "selective-scaler.config"


def test_the_example_resolves_strictly_and_is_explicit():
    resolved = SCALER_SCHEMA.load(EXAMPLE)
    assert has_selectors(resolved)
    assert resolved["backbone_scaling_list"] == ":2"
    assert resolved["sidechain_scaling_list"] == ":2"
    assert resolved["ligand_scaling_dict"] is None


def test_the_example_builds_states_for_the_structure_it_names(tmp_path):
    root = make_dataset_root(tmp_path, solvent="explicit")
    build = root / "build"
    shutil.copy(EXAMPLE, build / "scaler.config")
    record = build_scaled_states(system_path=build / "built.xml",
                                 topology_path=build / "built.pdb",
                                 config_path=build / "scaler.config", echo=False)
    selection = record["selection"]
    assert selection["selection_mode"] == "explicit"
    assert selection["residue_map"]["2"]["residue_name"] == "ALA"
    assert selection["residue_map"]["2"]["categories"] == ["backbone", "sidechain"]
    assert len(record["states"]) == 4
    assert "ALA" in (build / "REST2" / "scaler.log").read_text(encoding="utf-8")


def _section_blocks() -> list[str]:
    page = (REST2 / "README.md").read_text(encoding="utf-8")
    start = page.index("## Selective REST2")
    end = page.index("\n## ", start + 1)
    return re.findall(r"```yaml\n(.*?)```", page[start:end], flags=re.S)


def test_the_pages_selector_block_parses():
    blocks = [yaml.safe_load(b) for b in _section_blocks()]
    selectors = next(b for b in blocks if "backbone_scaling_list" in b)
    parsed = parse_selectors(selectors)
    assert parsed["backbone"]["residues"] == (45, 46, 59)
    assert parsed["sidechain"]["residues"] == (45, 46, 47, 48, 49, 50)
    assert parsed["ligands"]["L01"]["residues"] == (201,)


def test_the_pages_exclusion_file_loads(tmp_path):
    block = next(b for b in _section_blocks()
                 if b.startswith("format: md-tools-torsion-exclusions/1"))
    path = tmp_path / "L01-exclusions.yaml"
    path.write_text(block, encoding="utf-8")
    loaded = load_torsion_exclusions(path, where="the REST2 page")
    assert loaded["parameters"] == "CHEMBL112/param_0123456789ab"
    assert loaded["central_bonds"] == [["C4", "N1"]]
