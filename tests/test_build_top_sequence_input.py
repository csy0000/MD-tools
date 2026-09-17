"""`build-top -i X.seq`: a peptide from one line of residue names, built by tleap's `sequence`.

WHAT IS PROMISED

    A `.seq` holds exactly one record of whitespace-separated residue names (`ACE ALA NME`), with
    `#` comments and blank lines allowed. tleap builds the chain -- extended, library geometry --
    with the leaprc matching `forcefield.protein`, and from that PDB on the build is exactly the
    `.pdb` peptide route, explicit and implicit alike. Only `solute.kind: peptide` accepts it; the
    kind-by-suffix refusals live with the rest of the matrix in `test_build_top_input_formats.py`.

A REFUSAL LEAVES NOTHING BEHIND

    Malformed files and a residue tleap does not know are both statements about the input, so
    both exit 2 with no output directory -- tleap runs in a private temporary directory BEFORE
    any output parent is created, precisely so that its refusal can arrive this way.

THE COMPARISON

    `tests/data/ALA.pdb` is what tleap writes for `ACE ALA NME`, so building it the `.pdb` way and
    the `.seq` way must give the same residues and atom names in the same order, the same particle
    count, and the same masses. Coordinates are not compared: explicit solvation seeds from the
    solute, and nothing here claims the two routes place the same waters.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

needs_tleap = pytest.mark.skipif(
    __import__("shutil").which("tleap") is None,
    reason="tleap (AmberTools) is not on PATH; activate openmm-env")


def _build(work: Path, structure: Path, config_text: str, out: str = "out"):
    config = work / f"{out}.config"
    config.write_text(config_text, encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", f"{out}/built.xml",
               "-op", f"{out}/built.pdb", "-log", f"{out}/built.log", "--config", str(config)],
        cwd=work, capture_output=True, text=True, timeout=1800)


def _seq(work: Path, text: str, name: str = "ALA.seq") -> Path:
    path = work / name
    path.write_text(text, encoding="utf-8")
    return path


# --- the file's shape -------------------------------------------------------------------------

@pytest.mark.parametrize("text, message", [
    ("", "contains no sequence record"),
    ("# only a comment\n\n", "contains no sequence record"),
    ("ACE ALA NME\nACE GLY NME\n", "contains 2 sequence records"),
])
def test_a_malformed_seq_is_refused_before_anything_exists(text, message, tmp_path):
    done = _build(tmp_path, _seq(tmp_path, text), "solvent:\n  model: GBn2\n")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert message in done.stderr, done.stderr
    assert "ALA.seq" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


@needs_tleap
def test_a_residue_tleap_does_not_know_is_refused_by_tleap_with_nothing_created(tmp_path):
    """tleap is the authority on residue names; its refusal is surfaced, with its log."""
    done = _build(tmp_path, _seq(tmp_path, "ACE XYZ NME\n"), "solvent:\n  model: GBn2\n")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "tleap refused the sequence" in done.stderr, done.stderr
    assert "XYZ" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


# --- real builds ------------------------------------------------------------------------------

def _residues_and_atoms(pdb: Path):
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices

    topology = PDBFile(str(pdb)).topology
    solute = set(solute_atom_indices(topology))
    atoms = [a for a in topology.atoms() if a.index in solute]
    residues = []
    for atom in atoms:
        if not residues or residues[-1] is not atom.residue:
            residues.append(atom.residue)
    return [r.name for r in residues], [(a.residue.name, a.name) for a in atoms]


def _masses(xml: Path, n: int):
    from openmm import XmlSerializer, unit

    system = XmlSerializer.deserialize(xml.read_text(encoding="utf-8"))
    return system.getNumParticles(), [round(system.getParticleMass(i).value_in_unit(unit.dalton), 6)
                                      for i in range(n)]


@needs_tleap
@pytest.mark.slow
def test_an_explicit_hmr_build_from_a_sequence_is_the_pdb_build_of_the_same_peptide(tmp_path):
    """TIP3P, 1.5 nm padding, HMR on: residues ACE ALA NME, hydrogens repartitioned, and the
    same solute atoms, names and masses as building `ALA.pdb` the `.pdb` way."""
    from md_tools.build.record import read_record

    config = ("solvent:\n  model: TIP3P\n  padding_nm: 1.5\n"
              "hydrogen_mass_repartitioning:\n  enabled: true\n")
    seq = _build(tmp_path, _seq(tmp_path, "# alanine dipeptide\n\nACE ALA NME\n"), config, "seq")
    assert seq.returncode == 0, seq.stdout[-3000:] + seq.stderr[-3000:]
    pdb = _build(tmp_path, ALA, config, "pdb")
    assert pdb.returncode == 0, pdb.stdout[-3000:] + pdb.stderr[-3000:]

    for out in ("seq", "pdb"):
        assert (tmp_path / out / "built.xml").is_file() and (tmp_path / out / "built.pdb").is_file()
    residues, atoms = _residues_and_atoms(tmp_path / "seq" / "built.pdb")
    assert residues == ["ACE", "ALA", "NME"]
    assert atoms == _residues_and_atoms(tmp_path / "pdb" / "built.pdb")[1]
    assert _residues_and_atoms(tmp_path / "seq" / "built.solute.pdb")[0] == ["ACE", "ALA", "NME"]

    n_solute = len(atoms)
    _n_seq, seq_masses = _masses(tmp_path / "seq" / "built.xml", n_solute)
    _n_pdb, pdb_masses = _masses(tmp_path / "pdb" / "built.xml", n_solute)
    assert seq_masses == pdb_masses
    hydrogen_masses = {m for (_, name), m in zip(atoms, seq_masses) if name.startswith("H")}
    assert hydrogen_masses == {3.024}, hydrogen_masses

    record = read_record(tmp_path / "seq" / "built.log")
    assert record["interpretation"]["input_format"] == "seq"
    assert record["interpretation"]["sequence"] == ["ACE", "ALA", "NME"]
    assert record["hmr"]["applied"] is True and 3.024 in record["hmr"]["distinct_hydrogen_masses_amu"]
    assert record["solvent"]["treatment"] == "explicit"
    block = record["sequence"]
    assert block["residues"] == ["ACE", "ALA", "NME"]
    assert block["leaprc"] == "leaprc.protein.ff14SB"
    assert "mol = sequence { ACE ALA NME }" in block["tleap_commands"]
    assert len(block["tleap_log"]["sha256"]) == 64 and len(block["generated_pdb"]["sha256"]) == 64
    log = (tmp_path / "seq" / "built.log").read_text(encoding="utf-8")
    assert "EXTENDED" in log and "Errors = 0" in log
    # No machine path from tleap's library search leaks into the record.
    assert "/dat/leap/" not in log
    assert sorted(p.name for p in (tmp_path / "seq").glob("*.sdf")) == []


@needs_tleap
@pytest.mark.slow
def test_an_implicit_build_from_a_sequence_is_the_pdb_build_of_the_same_peptide(tmp_path):
    """GBn2: the whole System is the solute, so the particle counts and masses match exactly."""
    config = "solvent:\n  model: GBn2\n"
    seq = _build(tmp_path, _seq(tmp_path, "ACE ALA NME\n"), config, "seq")
    assert seq.returncode == 0, seq.stdout[-3000:] + seq.stderr[-3000:]
    pdb = _build(tmp_path, ALA, config, "pdb")
    assert pdb.returncode == 0, pdb.stdout[-3000:] + pdb.stderr[-3000:]

    residues, atoms = _residues_and_atoms(tmp_path / "seq" / "built.pdb")
    assert residues == ["ACE", "ALA", "NME"]
    assert atoms == _residues_and_atoms(tmp_path / "pdb" / "built.pdb")[1]
    n_seq, seq_masses = _masses(tmp_path / "seq" / "built.xml", len(atoms))
    n_pdb, pdb_masses = _masses(tmp_path / "pdb" / "built.xml", len(atoms))
    assert n_seq == n_pdb == len(atoms) == 22
    assert seq_masses == pdb_masses


def test_the_leaprc_follows_every_protein_forcefield():
    """A force field added to `PROTEIN_FORCEFIELDS` without a residue library would make a
    `.seq` build under it fail with a KeyError rather than a refusal."""
    from md_tools.build.top import PROTEIN_FORCEFIELDS, SEQUENCE_LEAPRC

    assert set(SEQUENCE_LEAPRC) == set(PROTEIN_FORCEFIELDS)
    assert SEQUENCE_LEAPRC["ff19SB"] == "leaprc.protein.ff19SB"
