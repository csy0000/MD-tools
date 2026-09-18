"""`build-top --parameterize`: one molecule to one package directory, with readable copies.

The directory it writes IS a package -- it loads, it registers, and a catalog search finds it --
and it also holds the files a person and a later command line point at. A loose folder of
`{TYL.sdf, TYL.pdb}` would satisfy the eye and neither of the other two.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")

from tests.test_ligand_mapping import _package  # noqa: E402
from tests.test_ligand_packages import _molecule  # noqa: E402

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top"]


def _sdf(mol, path: Path) -> Path:
    from rdkit import Chem

    Chem.MolToMolFile(Chem.Mol(mol), str(path))
    return path


def _mol2(mol, path: Path) -> Path:
    """A minimal Tripos mol2 for the same molecule: RDKit reads mol2 but does not write one."""
    from rdkit import Chem

    order = {Chem.BondType.SINGLE: "1", Chem.BondType.DOUBLE: "2",
             Chem.BondType.TRIPLE: "3", Chem.BondType.AROMATIC: "ar"}
    conformer = mol.GetConformer()
    lines = ["@<TRIPOS>MOLECULE", "LIG", f" {mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
             "SMALL", "NO_CHARGES", "", "@<TRIPOS>ATOM"]
    for index, atom in enumerate(mol.GetAtoms(), start=1):
        point = conformer.GetAtomPosition(index - 1)
        element = atom.GetSymbol()
        kind = f"{element}.ar" if atom.GetIsAromatic() and element in ("C", "N") else element
        lines.append(f"{index:7d} {element}{index:<6} {point.x:9.4f} {point.y:9.4f} "
                     f"{point.z:9.4f} {kind:<6} 1 LIG 0.0000")
    lines.append("@<TRIPOS>BOND")
    for n, bond in enumerate(mol.GetBonds(), start=1):
        lines.append(f"{n:6d} {bond.GetBeginAtomIdx() + 1:5d} {bond.GetEndAtomIdx() + 1:5d} "
                     f"{order[bond.GetBondType()]:>4}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(work: Path, catalog: Path, compound: str = "CHEMBL112") -> Path:
    path = work / "para.config"
    path.write_text(f"solute:\n  kind: ligand\n  compound_id: {compound}\n"
                    f"ligand_catalog:\n  path: {catalog}\n", encoding="utf-8")
    return path


def _run(args, work: Path, env=None):
    return subprocess.run(CLI + args, cwd=work, capture_output=True, text=True, timeout=1800,
                          env=env or os.environ)


def _parameterize(work: Path, structure: Path, catalog: Path, *, out="build/parameter",
                  resname="TYL", extra_args=(), env=None, compound="CHEMBL112"):
    _config(work, catalog, compound)
    return _run(["--parameterize", "-i", str(structure), "--config", "para.config",
                 "-op", f"{out}/{resname}.pdb", "-os", f"{out}/{resname}.xml",
                 "-log", "parameterize.log", "--resname", resname, *extra_args], work, env)


@pytest.mark.slow
@pytest.mark.parametrize("source", ["sdf", "mol2"])
def test_it_writes_a_real_package_with_the_readable_copies_beside_it(tmp_path, source):
    from md_tools.build.record import read_record
    from md_tools.ligands import load_package

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    work = tmp_path / "work"
    work.mkdir()
    mol = _molecule("CC(=O)Nc1ccc(O)cc1")
    structure = (_sdf(mol, work / "in.sdf") if source == "sdf" else _mol2(mol, work / "in.mol2"))

    result = _parameterize(work, structure, tmp_path / "catalog")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    directory = work / "build" / "parameter"
    assert sorted(p.name for p in directory.iterdir()) == [
        "TYL.pdb", "TYL.sdf", "TYL.xml", "metadata.json", "molecule.sdf", "parameter.config",
        "parameters.ffxml"]

    # It IS a package: it loads from a directory that is not named param_<id>, which is the shape
    # a local folder has, and the catalog's shape is not required of it.
    package = load_package(directory, expected_directory_name=False)
    assert package.reference == tyl.reference
    assert sorted(package.metadata["readable_copies"]) == ["TYL.pdb", "TYL.sdf", "TYL.xml"]

    # The copies are the package's own contents, not a second opinion about them.
    assert (directory / "TYL.sdf").read_bytes() == (directory / "molecule.sdf").read_bytes()
    from openmm import XmlSerializer, app

    system = XmlSerializer.deserialize((directory / "TYL.xml").read_text())
    assert system.getNumParticles() == mol.GetNumAtoms()
    assert system.getNumConstraints() == 0 and not system.usesPeriodicBoundaryConditions()
    topology = app.PDBFile(str(directory / "TYL.pdb")).topology
    assert [r.name for r in topology.residues()] == ["TYL"]
    assert [a.name for a in topology.atoms()] == list(package.atom_names)

    record = read_record(work / "parameterize.log")
    assert record["mode"] == "parameterize"
    assert record["outputs"]["package"]["reference"] == tyl.reference
    # The catalog already held this exact state, so nothing was charged again.
    assert record["ligand_packages"]["attached"]["charges_generated_in_this_build"] is False


@pytest.mark.slow
def test_a_modified_readable_copy_is_refused(tmp_path):
    """The copies are declared with their digests, so "a package holds exactly these" still holds."""
    from md_tools.ligands import PackageError, load_package

    _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    work = tmp_path / "work"
    work.mkdir()
    result = _parameterize(work, _sdf(_molecule("CC(=O)Nc1ccc(O)cc1"), work / "in.sdf"),
                           tmp_path / "catalog")
    assert result.returncode == 0, result.stderr[-2000:]
    directory = work / "build" / "parameter"

    (directory / "TYL.pdb").write_text("ATOM      1  C1  TYL     1       0.0   0.0   0.0\nEND\n")
    with pytest.raises(PackageError, match="the copy was modified"):
        load_package(directory, expected_directory_name=False)

    (directory / "stray.txt").write_text("x")
    with pytest.raises(PackageError, match="unexpected files"):
        load_package(directory, expected_directory_name=False)


@pytest.mark.slow
def test_register_puts_it_in_the_catalog_through_the_one_registration_path(tmp_path):
    """`--register` is `data-register --ligand-package`, not a second implementation of it."""
    from md_tools.ligands import load_package

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    work = tmp_path / "work"
    work.mkdir()
    md_data = tmp_path / "MD_DATA"
    md_data.mkdir()
    env = {**os.environ, "MD_DATA": str(md_data), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    env.pop("MD_TOOLS_CONFIG", None)

    result = _parameterize(work, _sdf(_molecule("CC(=O)Nc1ccc(O)cc1"), work / "in.sdf"),
                           tmp_path / "catalog", extra_args=["--register"], env=env)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    destination = md_data / "parameters" / "ligands" / "CHEMBL112" / tyl.parameter_id
    assert load_package(destination).reference == tyl.reference
    # The catalog copy carries the readable copies too, and still verifies under its own name.
    assert (destination / "TYL.xml").is_file()

    found = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "data-register", "--find-ligand", "TYL"],
        capture_output=True, text=True, env=env, timeout=600)
    assert found.returncode == 0 and tyl.reference in found.stdout


@pytest.mark.slow
def test_a_molecule_the_catalog_does_not_hold_is_parameterised(tmp_path):
    """The search decides; `--parameterize` is not a synonym for "always charge"."""
    from md_tools.build.record import read_record

    _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")     # a different compound
    work = tmp_path / "work"
    work.mkdir()
    result = _parameterize(work, _sdf(_molecule("CCO"), work / "in.sdf"), tmp_path / "catalog",
                           resname="EOH", compound="CHEMBL545")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    attached = read_record(work / "parameterize.log")["ligand_packages"]["attached"]
    assert attached["how"] == "created"
    assert attached["charges_generated_in_this_build"] is True
    assert (work / "build" / "parameter" / "EOH.xml").is_file()


@pytest.mark.parametrize("case, expected", [
    ("two directories", "different directories"),
    ("no resname", "missing: --resname"),
    ("reserved resname", "already names water"),
    ("log inside", "inside the package directory"),
    ("with scaler", "two different jobs"),
    ("resname without the flag", "belongs to --parameterize"),
    ("smiles input", "must be .sdf or .mol2"),
])
def test_the_refusals(tmp_path, case, expected):
    work = tmp_path / "work"
    work.mkdir()
    (work / "in.sdf").write_text("x\n", encoding="utf-8")
    (work / "in.smi").write_text("CCO ethanol\n", encoding="utf-8")
    args = {
        "two directories": ["--parameterize", "-i", "in.sdf", "-op", "a/TYL.pdb",
                            "-os", "b/TYL.xml", "--resname", "TYL"],
        "no resname": ["--parameterize", "-i", "in.sdf", "-op", "c/TYL.pdb", "-os", "c/TYL.xml"],
        "reserved resname": ["--parameterize", "-i", "in.sdf", "-op", "d/x.pdb", "-os", "d/x.xml",
                             "--resname", "HOH"],
        # A configuration is needed here: this refusal is about paths, and it must arrive before
        # the configuration is read rather than after it.
        "log inside": ["--parameterize", "-i", "in.sdf", "-op", "e/TYL.pdb", "-os", "e/TYL.xml",
                       "-log", "e/built.log", "--resname", "TYL"],
        "with scaler": ["--parameterize", "--rest2-scaler", "-i", "in.sdf", "--resname", "TYL"],
        "resname without the flag": ["-i", "in.sdf", "--resname", "TYL"],
        "smiles input": ["--parameterize", "-i", "in.smi", "-op", "f/T.pdb", "-os", "f/T.xml",
                         "--resname", "TYL"],
    }[case]
    result = _run(args, work)
    assert result.returncode == 2, result.stdout[-1000:] + result.stderr[-1000:]
    assert expected in result.stderr, result.stderr
    assert not any(work.glob("*/TYL.pdb")) and not any(work.glob("*/*.xml"))
