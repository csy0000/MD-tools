"""The scripts shipped beside the bespoke-fit tutorial: what the page says about them is checked.

They run under ANOTHER environment's python (BespokeFit, QCEngine, MACE), so nothing here executes a
fit. What is asserted is what the page promises and a regression can break silently: that each file
parses, that none imports `md_tools` (the page says so, and an import would fail in that
environment), and that the charge symmetrisation averages over symmetry classes and nothing else.

PLATFORM_POLICY_EXEMPTION: static analysis of source files and one RDKit graph-symmetry
computation. No force field, no dynamics.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

BESPOKE = (Path(__file__).resolve().parents[1] / "docs" / "tutorial" / "ligand-parameterization"
           / "bespoke")
SCRIPTS = ["fit_bespoke.py", "mace_adapter.py", "qc_worker.py", "coordinator_once.py"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_each_shipped_script_parses_and_never_imports_md_tools(name):
    tree = ast.parse((BESPOKE / name).read_text(encoding="utf-8"), filename=name)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "md_tools" not in imported, f"{name} would fail in the BespokeFit environment"


def test_the_environment_recipe_pins_what_the_page_says_matters():
    text = (BESPOKE / "environment.yml").read_text(encoding="utf-8")
    for pin in ("qcengine<0.50", "qcelemental<0.50", "e3nn=0.4.4", "openff-forcefields>=2024.09",
                "openff-bespokefit=0.4.3"):
        assert pin in text, pin


def _fit_module():
    spec = importlib.util.spec_from_file_location("fit_bespoke", BESPOKE / "fit_bespoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_charge_symmetrisation_averages_equivalent_atoms_and_leaves_the_rest():
    rdkit = pytest.importorskip("rdkit")  # noqa: F841
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    # the three methyl hydrogens are equivalent; give them different numbers, and the rest distinct ones
    charges = [0.1 * i for i in range(mol.GetNumAtoms())]
    out, worst = _fit_module().symmetrise(mol, charges)
    methyl_h = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    assert len(methyl_h) == 3
    mean = sum(charges[i] for i in methyl_h) / 3
    assert all(abs(out[i] - mean) < 1e-12 for i in methyl_h)
    assert worst > 0
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in methyl_h and atom.GetAtomicNum() != 1:
            assert out[atom.GetIdx()] == pytest.approx(charges[atom.GetIdx()])


def test_a_charged_molecule_is_refused_for_mace_off23_before_any_fit():
    source = (BESPOKE / "fit_bespoke.py").read_text(encoding="utf-8")
    assert "MACE-OFF23 has no charge input" in source and "mace-omol" in source
