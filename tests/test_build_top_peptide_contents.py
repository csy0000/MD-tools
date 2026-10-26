"""`kind: peptide` builds protein chains and their crystal waters, and nothing else.

A water is the chain's own and is kept unless `--strip-water` is passed. An ion, a crystallisation
additive (DMSO, EDO) or a ligand is refused by name, with exit 2 and nothing created: deleting it
is the user's edit to their own file, and carrying it along would build a different system from
the one `kind: peptide` names. `--strip-water` is refused where it would do nothing.

COMMAND-LEVEL: the real `md-openmm build-top` in a subprocess, because the exit code and the
absence of an output directory are the contract.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

def _row(serial, atom, resname, resid, xyz, element):
    """One fixed-column HETATM record (PDB columns, not whitespace-separated fields)."""
    return (f"HETATM{serial:5d} {atom:<4s} {resname:>3s} B{resid:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}")


HETATM = {
    "HOH": [_row(23, " O  ", "HOH", 1, (8.0, 8.0, 8.0), "O"),
            _row(24, " O  ", "HOH", 2, (10.0, 8.0, 8.0), "O")],
    "DMS": [_row(23, " S  ", "DMS", 1, (8.0, 8.0, 8.0), "S")],
    "NA": [_row(23, "NA  ", " NA", 1, (8.0, 8.0, 8.0), "NA")],
}


def _structure(work: Path, *extra: str) -> Path:
    lines = ALA.read_text().splitlines()
    end = lines.index("TER   ")
    lines[end:end] = [row for name in extra for row in HETATM[name]]
    target = work / "in.pdb"
    target.write_text("\n".join(lines) + "\n")
    return target


def _build(work: Path, structure: Path, *flags: str, kind="peptide", solvent="GBn2"):
    config = work / "sys.config"
    config.write_text(f"solute:\n  kind: {kind}\nsolvent:\n  model: {solvent}\n", encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", "out/built.xml", "-op", "out/built.pdb",
               "-log", "out/built.log", "--config", str(config), *flags],
        cwd=work, capture_output=True, text=True, timeout=1800)


@pytest.mark.parametrize("name", ["DMS", "NA"])
def test_anything_but_protein_and_water_is_refused_and_leaves_nothing(name, tmp_path):
    done = _build(tmp_path, _structure(tmp_path, name))
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "kind is 'peptide'" in done.stderr and f"{name} chain 'B'" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


def test_a_refusal_is_not_lifted_by_strip_water(tmp_path):
    """Stripping removes water only; a DMSO is still refused."""
    done = _build(tmp_path, _structure(tmp_path, "HOH", "DMS"), "--strip-water")
    assert done.returncode == 2 and "DMS chain 'B'" in done.stderr and "HOH" not in done.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["ligand", "peptide-like"])
def test_strip_water_is_refused_where_it_does_nothing(kind, tmp_path):
    smi = tmp_path / "in.smi"
    smi.write_text("CCO MOL\n")
    done = _build(tmp_path, smi, "--strip-water", kind=kind)
    assert done.returncode == 2 and "--strip-water" in done.stderr and "kind: peptide" in done.stderr
    assert not (tmp_path / "out").exists()


def test_strip_water_removes_the_crystal_waters_and_says_so(tmp_path):
    done = _build(tmp_path, _structure(tmp_path, "HOH"), "--strip-water")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "HOH" not in (tmp_path / "out" / "built.pdb").read_text()
    assert "2 water residue(s) removed (--strip-water)" in (tmp_path / "out" / "built.log").read_text()


@pytest.mark.slow
def test_crystal_waters_are_kept_by_default(tmp_path):
    done = _build(tmp_path, _structure(tmp_path, "HOH"), solvent="TIP3P")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    built = (tmp_path / "out" / "built.pdb").read_text()
    assert built.count(" HOH B   1") >= 1 and built.count(" HOH B   2") >= 1
