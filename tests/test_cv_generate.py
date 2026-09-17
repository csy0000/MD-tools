"""`collective_variables.generate: all_solute_torsions` -- build-md writes the cv.yaml itself.

The generated file is an ordinary definition: explicit atom selectors, parsed by the same loader,
copied in content-addressed, and named in `resolved.config`'s `file`. These tests pin the
enumeration, the configuration rules, and that what a run reads is that ordinary file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.cv.generate import cv_document, cv_yaml_text, solute_torsions

from .conftest import make_dataset_root

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]


def _atoms(names, residue="LIG", resid="1", chain="A"):
    return [{"name": n, "residue_name": residue, "residue_id": resid, "chain_id": chain}
            for n in names]


# -- the enumeration ------------------------------------------------------------------------------

def test_a_butane_chain_has_exactly_one_heavy_atom_torsion():
    bonds = [(0, 1), (1, 2), (2, 3)]
    assert solute_torsions(_atoms("ABCD"), bonds, range(4)) == [(0, 1, 2, 3)]


def test_each_dihedral_is_listed_once_not_once_per_direction():
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
    found = solute_torsions(_atoms("ABCDE"), bonds, range(5))
    assert found == [(0, 1, 2, 3), (1, 2, 3, 4)]
    assert len({frozenset(q) for q in found}) == len(found)


def test_every_substituent_combination_about_a_bond_is_a_torsion():
    # C1(H,H) - C2(H,H): 2 x 2 torsions about the central bond, and no others.
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    found = solute_torsions(_atoms(["C1", "C2", "H1", "H2", "H3", "H4"]), bonds, range(6))
    assert len(found) == 4
    assert all((q[1], q[2]) == (0, 1) for q in found)


def test_atoms_outside_the_solute_are_never_part_of_a_torsion():
    bonds = [(0, 1), (1, 2), (2, 3)]
    assert solute_torsions(_atoms("ABCD"), bonds, range(3)) == []


def test_a_ring_does_not_produce_a_degenerate_torsion():
    bonds = [(0, 1), (1, 2), (2, 0)]                       # a three-membered ring
    assert solute_torsions(_atoms("ABC"), bonds, range(3)) == []


def test_names_are_valid_csv_headings_and_unique():
    atoms = _atoms(["C", "C", "C'", "O-1"], residue="T-L")
    document = cv_document(atoms, [(0, 1, 2, 3), (3, 2, 1, 0)], source="all_solute_torsions")
    names = [entry["name"] for entry in document["collective_variables"]]
    assert len(set(names)) == 2
    assert all(name[0].isalpha() and name.replace("_", "").isalnum() for name in names)


def test_the_generated_text_is_a_loadable_cv_definition(tmp_path):
    from openmm.app import PDBFile

    from md_tools.cv import load_cv_definition

    pdb = PDBFile(str(Path(__file__).parent / "data" / "ALA.pdb"))
    atoms = [{"name": a.name, "residue_name": a.residue.name, "residue_id": a.residue.id,
              "chain_id": a.residue.chain.id} for a in pdb.topology.atoms()]
    bonds = [(b[0].index, b[1].index) for b in pdb.topology.bonds()]
    torsions = solute_torsions(atoms, bonds, range(len(atoms)))
    path = tmp_path / "cv.yaml"
    path.write_text(cv_yaml_text(cv_document(atoms, torsions, source="all_solute_torsions")))
    definition = load_cv_definition(path, topology=pdb.topology, particles=len(atoms))
    assert len(definition.names) == len(torsions)
    assert [cv.indices for cv in definition.variables] == torsions


# -- the configuration --------------------------------------------------------------------------

def _resolve(tmp_path, block):
    from md_tools.build.md import resolve_md_config

    path = tmp_path / "c.config"
    path.write_text(yaml.safe_dump({"protocol": "cMD", "solvent": "implicit",
                                    "collective_variables": block}))
    return resolve_md_config(path)


def test_file_and_generate_together_are_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="Give one"):
        _resolve(tmp_path, {"file": "cv.yaml", "generate": "all_solute_torsions",
                            "interval_steps": 100})


def test_generate_without_an_interval_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="interval_steps is 0"):
        _resolve(tmp_path, {"generate": "all_solute_torsions"})


def test_an_unknown_generator_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError):
        _resolve(tmp_path, {"generate": "everything", "interval_steps": 100})


# -- build-md -----------------------------------------------------------------------------------

@pytest.fixture
def generated(tmp_path):
    root = make_dataset_root(tmp_path)
    config = root / "cMD.config"
    config.write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "collective_variables": {"generate": "all_solute_torsions", "interval_steps": 100}}))
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "cMD-run1"),
                                 "--config", str(config)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def test_build_md_writes_one_content_addressed_definition_and_names_it(generated):
    run = generated / "cMD-run1"
    copies = sorted(run.glob("cv.*.yaml"))
    assert [p.name for p in copies] != [] and len(copies) == 1
    assert not (run / "cv.generated.yaml").exists()
    resolved = yaml.safe_load((run / "resolved.config").read_text())
    assert resolved["collective_variables"]["file"] == copies[0].name
    assert resolved["collective_variables"]["generate"] is None
    # Beside every declaration that names it, as a named definition is.
    for directory in (generated / "input", generated / "min", run / "eq"):
        assert (directory / copies[0].name).is_file(), directory


def test_the_generated_definition_is_every_proper_torsion_of_the_built_solute(generated):
    from openmm import HarmonicBondForce, XmlSerializer
    from openmm.app import PDBFile

    from md_tools.cv import load_cv_definition
    from md_tools.md.stage import solute_atom_indices

    run = generated / "cMD-run1"
    topology = PDBFile(str(generated / "build" / "built.pdb")).topology
    system = XmlSerializer.deserialize((generated / "build" / "built.xml").read_text())
    definition = load_cv_definition(next(run.glob("cv.*.yaml")), topology=topology)
    solute = set(solute_atom_indices(topology))
    bonds = set()
    for force in system.getForces():
        if isinstance(force, HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a, b, *_ = force.getBondParameters(i)
                bonds.add(frozenset((a, b)))
    for i in range(system.getNumConstraints()):
        a, b, _ = system.getConstraintParameters(i)
        bonds.add(frozenset((a, b)))
    for cv in definition.variables:
        i, j, k, l = cv.indices
        assert set(cv.indices) <= solute
        assert {frozenset((i, j)), frozenset((j, k)), frozenset((k, l))} <= bonds
    reversed_pairs = {tuple(reversed(cv.indices)) for cv in definition.variables}
    assert not reversed_pairs & {cv.indices for cv in definition.variables}


def test_the_generated_input_resolves_back_to_the_resolved_config(generated):
    from md_tools.run.inputs import parse_run_input

    run = generated / "cMD-run1"
    parsed = parse_run_input(generated / "input" / "cMD.in", run_config=run / "run.config")
    stored = yaml.safe_load((run / "resolved.config").read_text())
    assert parsed.resolved["collective_variables"] == stored["collective_variables"]
