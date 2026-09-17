"""`solute.residue_name` is APPLIED, and the prepared molecule is `<RESNAME>.sdf` (backlog 18).

Until 0.5.4 the name was resolved and recorded and applied to nothing: the topology said `UNL`
while the record said `TYL`, and the molecule was written as `built.sdf`. Now the residue in
`built.pdb`, `built.solute.pdb` and the serialised topology carries the name, stated or assigned,
on both input routes and both solvent routes; the molecule is written as `<RESNAME>.sdf` with that
name as its title; and `preflight._ligand_sdf_beside` finds either layout.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def _build(work: Path, structure: Path, config_text: str):
    config = work / "sys.config"
    config.write_text(config_text, encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", "build/built.xml",
               "-op", "build/built.pdb", "-log", "build/built.log", "--config", str(config)],
        cwd=work, capture_output=True, text=True, timeout=1800)


def _solute_residue_names(pdb: Path) -> list[str]:
    from openmm.app import PDBFile

    from md_tools.md.stage import SOLVENT_RESIDUES

    return sorted({r.name for r in PDBFile(str(pdb)).topology.residues()
                   if r.name.upper() not in SOLVENT_RESIDUES})


# --- _ligand_sdf_beside: both layouts ---------------------------------------------------------

def _pdb(path: Path, residues: list[tuple[str, int]]) -> Path:
    lines = []
    serial = 1
    for name, number in residues:
        lines.append(f"HETATM{serial:5d}  C1  {name:>3} A{number:4d}       0.000   0.000   0.000"
                     f"  1.00  0.00           C  ")
        serial += 1
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


def test_the_preflight_finds_a_pre_0_5_4_system_stem_sdf(tmp_path):
    from md_tools.run.preflight import _ligand_sdf_beside

    (tmp_path / "built.xml").write_text("<System/>", encoding="utf-8")
    _pdb(tmp_path / "built.pdb", [("UNL", 1), ("HOH", 2), ("NA", 3)])
    (tmp_path / "built.sdf").write_text("x", encoding="utf-8")
    assert _ligand_sdf_beside(tmp_path / "built.xml") == tmp_path / "built.sdf"


def test_the_preflight_finds_the_residue_named_sdf(tmp_path):
    from md_tools.run.preflight import _ligand_sdf_beside

    (tmp_path / "built.xml").write_text("<System/>", encoding="utf-8")
    _pdb(tmp_path / "built.pdb", [("TYL", 1), ("HOH", 2), ("HOH", 3), ("Cl-", 4), ("CL", 5)])
    assert _ligand_sdf_beside(tmp_path / "built.xml") is None      # named, but not written
    (tmp_path / "TYL.sdf").write_text("x", encoding="utf-8")
    assert _ligand_sdf_beside(tmp_path / "built.xml") == tmp_path / "TYL.sdf"
    # The old spelling, when both exist, is the one read: it is what that build paired.
    (tmp_path / "built.sdf").write_text("x", encoding="utf-8")
    assert _ligand_sdf_beside(tmp_path / "built.xml") == tmp_path / "built.sdf"


def test_the_preflight_finds_nothing_for_a_peptide(tmp_path):
    from md_tools.run.preflight import _ligand_sdf_beside

    (tmp_path / "built.xml").write_text("<System/>", encoding="utf-8")
    _pdb(tmp_path / "built.pdb", [("ACE", 1), ("ALA", 2), ("NME", 3), ("HOH", 4)])
    # Even a stray SDF named for one of its residues is not a molecule build-top wrote.
    (tmp_path / "ALA.sdf").write_text("x", encoding="utf-8")
    assert _ligand_sdf_beside(tmp_path / "built.xml") is None
    assert _ligand_sdf_beside(None) is None
    assert _ligand_sdf_beside(tmp_path / "missing.xml") is None


# --- the name itself --------------------------------------------------------------------------

def test_an_assigned_name_passes_over_one_that_already_means_something(tmp_path):
    from md_tools.build.top import _assigned_residue_name

    assert _assigned_residue_name(None, "paracetamol", tmp_path / "x.smi") == "PAR"
    assert _assigned_residue_name(None, None, tmp_path / "ab.sdf") == "ABX"
    # `ALA` would be read as a protein residue, `HOH` as water: the next candidate is taken.
    assert _assigned_residue_name(None, "ala", tmp_path / "ligand.smi") == "LIG"
    assert _assigned_residue_name(None, None, tmp_path / "hoh.sdf") == "LIG"


@pytest.mark.parametrize("kind, name, message", [
    ("peptide", "TYL", "applied to nothing"),
    ("ligand", "TY", "not three letters or digits"),
    ("ligand", "TYL1", "not three letters or digits"),
    ("ligand", "T-L", "not three letters or digits"),
    ("ligand", "HOH", "already names water, an ion or a protein residue"),
    ("ligand", "ala", "already names water, an ion or a protein residue"),
])
def test_a_stated_name_that_cannot_be_applied_is_refused_with_nothing_created(kind, name,
                                                                             message, tmp_path):
    structure = tmp_path / "in.smi"
    structure.write_text("CCO\n", encoding="utf-8")
    done = _build(tmp_path, structure,
                  f"solute:\n  kind: {kind}\n  residue_name: {name}\nsolvent:\n  model: GBn2\n")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "solute.residue_name" in done.stderr and message in done.stderr, done.stderr
    assert not (tmp_path / "build").exists()


def test_a_legacy_system_stem_sdf_beside_the_outputs_is_refused(tmp_path):
    """The preflight reads `<stem>.sdf` first, so a new System must not be written beside one."""
    structure = tmp_path / "in.smi"
    structure.write_text("CCO\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "built.sdf").write_text("stale", encoding="utf-8")
    done = _build(tmp_path, structure, "solute:\n  kind: ligand\nsolvent:\n  model: GBn2\n")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "built.sdf" in done.stderr, done.stderr
    assert sorted(p.name for p in (tmp_path / "build").iterdir()) == ["built.sdf"]


# --- real builds ------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_stated_name_is_applied_on_the_explicit_smiles_route(tmp_path):
    from rdkit import Chem

    from md_tools.build.record import read_record
    from md_tools.run.preflight import _ligand_sdf_beside

    structure = tmp_path / "paracetamol.smi"
    structure.write_text(f"{PARACETAMOL} paracetamol\n", encoding="utf-8")
    done = _build(tmp_path, structure,
                  "solute:\n  kind: ligand\n  residue_name: TYL\n  ligand_charge_method: nagl\n"
                  "solvent:\n  model: TIP3P\n  padding_nm: 0.6\n  cutoff_nm: 0.6\n")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    build = tmp_path / "build"
    assert _solute_residue_names(build / "built.pdb") == ["TYL"]
    assert _solute_residue_names(build / "built.solute.pdb") == ["TYL"]
    assert (build / "TYL.sdf").is_file() and not (build / "built.sdf").exists()
    mol = Chem.MolFromMolFile(str(build / "TYL.sdf"), removeHs=False)
    assert mol.GetProp("_Name") == "TYL" and mol.GetNumAtoms() == 20
    record = read_record(build / "built.log")
    assert record["interpretation"]["residue_name"] == "TYL"
    assert record["outputs"]["solute_sdf"]["residue_name"] == "TYL"
    assert record["outputs"]["solute_sdf"]["path"] == "TYL.sdf"
    assert _ligand_sdf_beside(build / "built.xml") == build / "TYL.sdf"


@pytest.mark.slow
def test_an_assigned_name_is_applied_on_the_implicit_sdf_route(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from md_tools.build.record import read_record
    from md_tools.run.preflight import _ligand_sdf_beside

    mol = Chem.AddHs(Chem.MolFromSmiles(PARACETAMOL))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    structure = tmp_path / "paracetamol.sdf"
    Chem.MolToMolFile(mol, str(structure))
    done = _build(tmp_path, structure,
                  "solute:\n  kind: ligand\n  ligand_charge_method: nagl\nsolvent:\n  model: GBn2\n")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    build = tmp_path / "build"
    assert _solute_residue_names(build / "built.pdb") == ["PAR"]
    assert _solute_residue_names(build / "built.solute.pdb") == ["PAR"]
    assert (build / "PAR.sdf").is_file() and not (build / "built.sdf").exists()
    assert Chem.MolFromMolFile(str(build / "PAR.sdf"), removeHs=False).GetProp("_Name") == "PAR"
    record = read_record(build / "built.log")
    assert record["interpretation"]["residue_name"] == "PAR"
    assert _ligand_sdf_beside(build / "built.xml") == build / "PAR.sdf"
