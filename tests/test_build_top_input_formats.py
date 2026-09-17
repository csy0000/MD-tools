"""`build-top`'s `-i`: which file formats it accepts, for which `solute.kind`, and what it refuses.

WHY THIS FILE EXISTS

    Every refusal here was unguarded. `build/top.py` has always refused a `.pdb` for a ligand and
    a `.smi` for a peptide by name, and a repository-wide grep for those message texts matched
    only the source lines that raise them -- so the rules could have been deleted and the suite
    would have stayed green. Adding `.sdf` as a third input made that gap worth closing rather
    than widening.

THE MATRIX IS THE POINT

    Four suffixes and three kinds is twelve cases, and they are not symmetric: `peptide` takes a
    PDB or a residue sequence (`.seq`, built by tleap) and nothing else, while `ligand` and
    `peptide-like` take either molecular-graph input.
    Stating all nine means a future fourth format has an obvious place to be declared, and means
    "accepted" is written down rather than inferred from the absence of a refusal.

A REFUSAL LEAVES NOTHING BEHIND

    Checked on every refusing case, not once. `-odir` holding a `built.log` is indistinguishable
    from a build that was attempted and died halfway, and the checks that can refuse therefore
    run before the output parents are created. That ordering is invisible from the outside and
    easy to undo, which is exactly what makes it worth a test.

THESE ARE COMMAND-LEVEL TESTS

    They run the real `md-openmm build-top` in a subprocess, the way `test_pairing_policy_through_
    build_top.py` does, because what is under test is the surface a user types -- including the
    exit code, which is 2 for "your input is wrong" and 1 for an internal failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

#: Small, rigid, and cheap to parameterise. Ethanol keeps AM1-BCC to seconds; none of these tests
#: needs a molecule whose conformer search is interesting.
ETHANOL = "CCO"

#: THE MOLECULE HAS TO SUIT THE KIND. `peptide-like` is not "a ligand with a flag on it": the
#: build applies `map_cyclic_peptide` over the result and refuses a molecule that is not a
#: head-to-tail cyclic peptide -- `PeptideMapError: no amide bond was found`. Ethanol is therefore
#: a valid `ligand` and an invalid `peptide-like`, and a matrix that used one molecule for both
#: would be testing the mapper rather than the input formats.
#:
#: Glycine anhydride -- cyclo(Gly-Gly) -- is the smallest thing that satisfies it: 8 heavy atoms,
#: two amide bonds, and it maps to ['GLY', 'GLY'] with 6 torsions. The alternative is the
#: cyclo(Gly-Asp-Arg) of example 5, which is 23 heavy atoms and is the suite's designated slow
#: build precisely because AM1-BCC dominates it.
CYCLO_GLY_GLY = "O=C1CNC(=O)CN1"

#: What each kind is built from here. `peptide` is the only one that takes a residue-named PDB.
MOLECULE = {"ligand": ETHANOL, "peptide-like": CYCLO_GLY_GLY}


def _write_sdf(path: Path, smiles: str = ETHANOL, *, hydrogens: bool = True,
               three_d: bool = True, count: int = 1):
    """An SDF built to order, so each refusal has exactly one thing wrong with it."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols = []
    for each in ([smiles, "CCC"][:count] if count > 1 else [smiles]):
        mol = Chem.MolFromSmiles(each)
        if hydrogens:
            mol = Chem.AddHs(mol)
        if three_d:
            AllChem.EmbedMolecule(mol, randomSeed=1)
        else:
            AllChem.Compute2DCoords(mol)
        mols.append(mol)
    if count > 1:
        writer = Chem.SDWriter(str(path))
        for mol in mols:
            writer.write(mol)
        writer.close()
    else:
        Chem.MolToMolFile(mols[0], str(path))
    return path


def _input(work: Path, suffix: str, kind: str = "ligand") -> Path:
    """A valid input of each format, so a refusal is about the MATRIX and not about the file."""
    if suffix == ".pdb":
        target = work / "in.pdb"
        target.write_bytes(ALA.read_bytes())
        return target
    if suffix == ".seq":
        target = work / "in.seq"
        target.write_text("ACE ALA NME\n", encoding="utf-8")
        return target
    smiles = MOLECULE.get(kind, ETHANOL)
    if suffix == ".smi":
        target = work / "in.smi"
        target.write_text(f"{smiles} MOL\n", encoding="utf-8")
        return target
    return _write_sdf(work / "in.sdf", smiles)


def _build(work: Path, structure: Path, kind: str):
    config = work / "sys.config"
    # GBn2 throughout: an implicit build takes seconds, and nothing here is about solvation.
    config.write_text(f"solute:\n  kind: {kind}\nsolvent:\n  model: GBn2\n", encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", "out/built.xml", "-op", "out/built.pdb",
               "-log", "out/built.log", "--config", str(config)],
        cwd=work, capture_output=True, text=True, timeout=1800)


# --- the matrix -------------------------------------------------------------------------------

#: (kind, suffix) -> accepted. `peptide` is residue-aware and needs a PDB; the two whole-molecule
#: kinds are built from a molecular graph and take either of the two that carry one.
MATRIX = {
    ("peptide", ".pdb"): True,
    ("peptide", ".seq"): True,
    ("peptide", ".smi"): False,
    ("peptide", ".sdf"): False,
    ("ligand", ".pdb"): False,
    ("ligand", ".seq"): False,
    ("ligand", ".smi"): True,
    ("ligand", ".sdf"): True,
    ("peptide-like", ".pdb"): False,
    ("peptide-like", ".seq"): False,
    ("peptide-like", ".smi"): True,
    ("peptide-like", ".sdf"): True,
}


@pytest.mark.parametrize("kind, suffix",
                         [pair for pair, ok in MATRIX.items() if not ok])
def test_a_kind_refuses_the_formats_it_cannot_be_built_from(kind, suffix, tmp_path):
    """Refused by name, with exit 2, and with nothing created."""
    done = _build(tmp_path, _input(tmp_path, suffix, kind), kind)
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert suffix in done.stderr and kind in done.stderr, done.stderr
    # A refusal that leaves a half-built directory behind is not a refusal.
    assert not (tmp_path / "out").exists(), sorted(p.name for p in (tmp_path / "out").iterdir())


@pytest.mark.slow
@pytest.mark.parametrize("kind, suffix", [pair for pair, ok in MATRIX.items() if ok])
def test_a_kind_builds_from_every_format_it_accepts(kind, suffix, tmp_path):
    """The positive half. Slow: these build real Systems, and the ligand ones run AM1-BCC."""
    done = _build(tmp_path, _input(tmp_path, suffix, kind), kind)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert (tmp_path / "out" / "built.xml").is_file()
    assert (tmp_path / "out" / "built.pdb").is_file()


# --- the SDF's own shape ----------------------------------------------------------------------

def test_a_two_dimensional_sdf_is_refused(tmp_path):
    """A flat molecule is not a starting structure: no force field considers that geometry real."""
    structure = _write_sdf(tmp_path / "in.sdf", three_d=False)
    done = _build(tmp_path, structure, "ligand")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "two-dimensional" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


def test_an_sdf_without_explicit_hydrogens_is_refused(tmp_path):
    """The ligand route parameterises what it is given and adds none, so this would build the
    heavy-atom skeleton alone -- silently, and with a plausible-looking System."""
    structure = _write_sdf(tmp_path / "in.sdf", hydrogens=False)
    done = _build(tmp_path, structure, "ligand")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "explicit hydrogens" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


def test_a_multi_record_sdf_is_refused_and_says_how_many(tmp_path):
    """Picking the first record would silently drop the rest. Mirrors `read_single_smiles`."""
    structure = _write_sdf(tmp_path / "in.sdf", count=2)
    done = _build(tmp_path, structure, "ligand")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "2 molecule records" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


def test_an_empty_sdf_is_refused_as_input_not_as_a_crash(tmp_path):
    """RDKit raises OSError on CONSTRUCTION for a file it cannot open, rather than yielding zero
    records, so this arrives by a different route than every other case here. It must still be a
    refusal of the user's input -- exit 2 -- and not an internal error."""
    structure = tmp_path / "in.sdf"
    structure.write_text("", encoding="utf-8")
    done = _build(tmp_path, structure, "ligand")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "-i" in done.stderr and str(structure) in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


def test_an_unknown_suffix_is_refused_by_the_format_gate(tmp_path):
    """The outermost gate, and the one that names all four accepted formats."""
    structure = _write_sdf(tmp_path / "in.mol2")
    done = _build(tmp_path, structure, "ligand")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert ".pdb, .cif, .seq, .smi or .sdf" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()


# --- what a build from each molecular-graph input leaves beside the System ---------------------

@pytest.mark.slow
@pytest.mark.parametrize("solvent", ["GBn2", "TIP3P"])
@pytest.mark.parametrize("suffix", [".smi", ".sdf"])
def test_every_ligand_build_writes_its_residue_named_sdf_beside_the_system(solvent, suffix,
                                                                          tmp_path):
    """Bond orders are not recoverable from a topology, and three consumers need them.

    MIGRATED in 0.5.4 from `..._writes_built_sdf_...`: the file is now `<RESNAME>.sdf`, named for
    the residue the topology carries -- `MOL` from the .smi's name field, `INX` from the .sdf's
    stem `in` -- and `built.sdf` is no longer written. The regression below is unchanged.

    THE EXPLICIT CASE IS THE REGRESSION. `initial_structure` wrote into the staging root on the
    explicit route and into `staging/structure` on the implicit one, while `build/top.py` copied
    `built.sdf` out of the latter -- so an explicit-solvent ligand build emitted no `built.sdf` at
    all. Nothing caught it because every assertion about that file ran under GBn2. Without it,
    `classify_unscaled_torsions`'s ligand route has nothing to perceive amides from, `map_from_sdf` has
    no input, and `preflight._ligand_sdf_beside` finds nothing beside the System.
    """
    structure = _input(tmp_path, suffix)
    config = tmp_path / "sys.config"
    text = f"solute:\n  kind: ligand\nsolvent:\n  model: {solvent}\n"
    if solvent != "GBn2":
        text += "  padding_nm: 0.6\n  cutoff_nm: 0.6\n"
    config.write_text(text, encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", "out/built.xml", "-op", "out/built.pdb",
               "-log", "out/built.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    name = "MOL" if suffix == ".smi" else "INX"
    assert (tmp_path / "out" / f"{name}.sdf").is_file(), (
        f"a {solvent} ligand build from {suffix} wrote no {name}.sdf")
    assert not (tmp_path / "out" / "built.sdf").exists()

    # It must be readable as what it claims to be, not merely present.
    from rdkit import Chem

    mol = Chem.MolFromMolFile(str(tmp_path / "out" / f"{name}.sdf"), removeHs=False)
    assert mol is not None and mol.GetNumConformers() == 1
    assert mol.GetProp("_Name") == name


@pytest.mark.slow
@pytest.mark.parametrize("suffix", [".pdb", ".seq"])
def test_a_peptide_build_writes_no_sdf(suffix, tmp_path):
    """Its absence is the signal that the ligand route does not apply, so it is asserted too.

    MIGRATED in 0.5.4: `built.sdf` is no longer the name a molecule would have, so ANY `.sdf`
    beside the System is what must be absent -- and the .seq peptide input is covered too."""
    done = _build(tmp_path, _input(tmp_path, suffix), "peptide")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.sdf")) == []


# --- the record says which input it was, and what that implied --------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("suffix, expected_format", [(".smi", "smi"), (".sdf", "sdf")])
def test_the_record_names_the_input_format_it_actually_read(suffix, expected_format, tmp_path):
    """`interpretation.input_format` is what the reference exporter reads to choose a reader, so
    a build that recorded the wrong one would export a bundle that rebuilds a different molecule.
    """
    from md_tools.build.record import read_record

    done = _build(tmp_path, _input(tmp_path, suffix), "ligand")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    record = read_record(tmp_path / "out" / "built.log")
    interpretation = record["interpretation"]
    assert interpretation["input_format"] == expected_format
    assert interpretation["route"] == "ligand"
    # The SMILES route records the string it embedded; the SDF route has no such claim to make,
    # because the coordinates came from the file rather than from a string.
    if suffix == ".smi":
        assert interpretation["smiles"] == ETHANOL
    else:
        assert interpretation["smiles"] is None


@pytest.mark.slow
def test_an_sdf_build_says_in_its_log_that_the_coordinates_were_not_generated(tmp_path):
    """The one fact that distinguishes the two ligand routes for a reader a year later."""
    done = _build(tmp_path, _input(tmp_path, ".sdf"), "ligand")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    log = (tmp_path / "out" / "built.log").read_text(encoding="utf-8")
    assert "no embedding, no minimisation" in log, log[:2000]
