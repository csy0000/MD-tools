"""`build-top` with ligand parameter packages: reuse in water and in a complex, and the refusals.

The acceptance criterion is the plan's: one package, the SAME per-atom, bonded and nonbonded
parameters in a ligand-only build and in a protein-ligand build, compared term by term on the
ligand atoms of each built System -- never whole-system energies against each other -- and no
charge calculation when a package is reused.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")
pytest.importorskip("networkx")

from tests.test_ligand_mapping import _package, _write_structure  # noqa: E402

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]


def _build(work: Path, structure: Path, config: str, *, out="build"):
    (work / "top.config").write_text(config, encoding="utf-8")
    return subprocess.run(
        CLI + ["build-top", "-i", str(structure), "-os", f"{out}/built.xml",
               "-op", f"{out}/built.pdb", "-log", f"{out}/built.log", "--config", "top.config"],
        cwd=work, capture_output=True, text=True, timeout=1800)


def _record(log: Path) -> dict:
    from md_tools.build.record import read_record

    return read_record(log)


def _ligand_table(system_xml: Path, pdb: Path, residue_key, package):
    from openmm import XmlSerializer, app

    from md_tools.ligands.package import _subsystem_table

    system = XmlSerializer.deserialize(system_xml.read_text())
    topology = app.PDBFile(str(pdb)).topology
    residue = next(r for r in topology.residues()
                   if (r.chain.id, str(r.id).strip()) == residue_key or
                   (residue_key is None and r.name == package.residue_name))
    indices = [a.index for a in residue.atoms()]
    assert [a.name for a in residue.atoms()] == list(package.atom_names)
    table, constrained = _subsystem_table(system, package.mol, indices,
                                          package.conventions["coulomb14scale"],
                                          package.conventions["lj14scale"])
    table.pop("_constraint_lengths")
    return table, constrained


def _write_sdf(package, path: Path, *, shuffle_seed=None):
    """The package molecule as an input SDF, optionally with its atoms in another order."""
    import numpy as np
    from rdkit import Chem

    mol = Chem.Mol(package.mol)
    for prop in list(mol.GetPropNames()):
        mol.ClearProp(prop)
    if shuffle_seed is not None:
        order = list(range(mol.GetNumAtoms()))
        np.random.default_rng(shuffle_seed).shuffle(order)
        mol = Chem.RenumberAtoms(mol, order)
    Chem.MolToMolFile(mol, str(path))
    return path


@pytest.mark.slow
def test_one_package_gives_identical_ligand_parameters_in_water_and_in_a_complex(tmp_path):
    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    eth = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    catalog = tmp_path / "catalog"

    # Ligand-only, from an SDF whose atoms are in a DIFFERENT order from the package's.
    alone = tmp_path / "alone"
    alone.mkdir()
    sdf = _write_sdf(tyl, alone / "tyl.sdf", shuffle_seed=3)
    result = _build(alone, sdf, f"""
solute:
  kind: ligand
  residue_name: TYL
  parameters: {tyl.reference}
ligand_catalog:
  path: {catalog}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stderr[-3000:]
    record = _record(alone / "build" / "built.log")
    attached = record["ligand_packages"]["attached"]
    assert attached["how"] == "reused (stated reference)"
    assert attached["charges_generated_in_this_build"] is False
    assert (alone / "build" / "ligands" / "CHEMBL112" / tyl.parameter_id / "parameters.ffxml").is_file()

    # A complex: a peptide, two copies of the same package and a second species.
    complex_dir = tmp_path / "complex"
    complex_dir.mkdir()
    structure = _write_structure(complex_dir / "complex.pdb", [
        ("B", "201", "TYL", tyl, (1.4, 0.2, 0.1)),
        ("C", "201", "TYL", tyl, (0.1, 1.5, 0.2)),
        ("C", "202", "EOH", eth, (0.2, 0.1, 1.5)),
    ])
    result = _build(complex_dir, Path("complex.pdb"), f"""
solute:
  kind: complex
ligands:
  - select: {{chain: B, resid: "201"}}
    parameters: {tyl.reference}
  - select: {{chain: C, resid: "201"}}
    parameters: {tyl.reference}
  - select: {{chain: C, resid: "202"}}
    parameters: {eth.reference}
ligand_catalog:
  path: {catalog}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stderr[-3000:]
    build = complex_dir / "build"
    readable = (build / "built.log").read_text()
    assert "protein-ligand complex (.pdb), 3 ligand instance(s)" in readable
    assert "single-molecule" not in readable and "supplied by the SDF" not in readable
    mapping = json.loads((build / "ligand_mapping.json").read_text())
    assert [i["package"]["reference"] for i in mapping["instances"]] == [
        tyl.reference, tyl.reference, eth.reference]

    reference, reference_constrained = _ligand_table(alone / "build" / "built.xml",
                                                     alone / "build" / "built.pdb", None, tyl)
    for key in (("B", "201"), ("C", "201")):
        table, constrained = _ligand_table(build / "built.xml", build / "built.pdb", key, tyl)
        assert constrained == reference_constrained
        for section in ("atoms", "bonds", "angles", "proper_torsions", "improper_torsions",
                        "exceptions"):
            assert sorted(table[section]) == sorted(reference[section]), (key, section)
    # And both equal the package itself, term by term (masses aside: no HMR here, but the
    # System's masses are compared by the package tests).
    from md_tools.ligands.package import _compare_tables

    table, constrained = _ligand_table(build / "built.xml", build / "built.pdb", ("C", "202"), eth)
    table["conventions"] = eth.table["conventions"]
    # Package first: a bond constrained in the built System has no force term there.
    assert _compare_tables(eth.table, table, where="EOH", skip_masses=True,
                           missing_bonds_ok=constrained) == []


@pytest.mark.slow
@pytest.mark.parametrize("case", ["unmapped-residue", "unknown-package", "ambiguous-selector",
                                  "implicit-complex", "wrong-state"])
def test_refusals_leave_nothing_behind(tmp_path, case):
    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    catalog = tmp_path / "catalog"
    work = tmp_path / "work"
    work.mkdir()
    ligands = [("B", "201", "TYL", tyl, (1.4, 0.2, 0.1)), ("C", "201", "TYL", tyl, (0.1, 1.5, 0.2))]
    entries = f"""
  - select: {{chain: B, resid: "201"}}
    parameters: {tyl.reference}
  - select: {{chain: C, resid: "201"}}
    parameters: {tyl.reference}"""
    solvent = "TIP3P\n  padding_nm: 1.0"
    structure = work / "complex.pdb"
    expected = None
    if case == "unmapped-residue":
        entries = entries.split("  - select: {chain: C")[0]
        expected = "no `ligands` entry maps them"
    elif case == "unknown-package":
        entries = entries.replace(tyl.parameter_id, "param_000000000000", 1)
        expected = "was not found"
    elif case == "ambiguous-selector":
        ligands[1] = ("B", "201", "TYL", tyl, (0.1, 1.5, 0.2))
        entries = entries.split("  - select: {chain: C")[0]
        expected = "matches 2 residues"
    elif case == "implicit-complex":
        solvent = "GBn2"
        expected = "implicit"
    _write_structure(structure, ligands)
    config = f"""
solute:
  kind: complex
ligands:{entries}
ligand_catalog:
  path: {catalog}
solvent:
  model: {solvent}
"""
    if case == "wrong-state":
        phenolate = _package(tmp_path, "CC(=O)Nc1ccc([O-])cc1", "CHEMBL112", "TYL")
        structure = work / "tyl.sdf"
        _write_sdf(tyl, structure)
        config = f"""
solute:
  kind: ligand
  residue_name: TYL
  parameters: {phenolate.reference}
ligand_catalog:
  path: {catalog}
"""
        expected = "not the chemical state of package"
    result = _build(work, structure.relative_to(work), config)
    assert result.returncode != 0
    assert expected in result.stderr, result.stderr[-2000:]
    if case != "wrong-state":
        # Refused while resolving the inputs: not even the output directory exists.
        assert not (work / "build").exists()


@pytest.mark.slow
def test_a_ligand_build_creates_its_package_once_and_a_rebuild_from_it_is_identical(tmp_path):
    """The charge-generating build, then the reusing one: same System, no second AM1-BCC."""
    first = tmp_path / "first"
    first.mkdir()
    (first / "paracetamol.smi").write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n")
    result = _build(first, Path("paracetamol.smi"), """
solute:
  kind: ligand
  residue_name: TYL
  compound_id: CHEMBL112
  aliases: [paracetamol, acetaminophen]
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stderr[-3000:]
    attached = _record(first / "build" / "built.log")["ligand_packages"]["attached"]
    assert attached["how"] == "created" and attached["charges_generated_in_this_build"] is True
    reference = attached["reference"]

    second = tmp_path / "second"
    second.mkdir()
    (second / "paracetamol.smi").write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n")
    result = _build(second, Path("paracetamol.smi"), f"""
solute:
  kind: ligand
  residue_name: TYL
  parameters: {reference}
ligand_catalog:
  path: {first / 'build' / 'ligands'}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stderr[-3000:]
    attached = _record(second / "build" / "built.log")["ligand_packages"]["attached"]
    assert attached["how"] == "reused (stated reference)"
    assert ((second / "build" / "built.xml").read_bytes()
            == (first / "build" / "built.xml").read_bytes())


@pytest.mark.slow
def test_build_top_searches_the_catalog_and_reuses_or_parameterises(tmp_path):
    """The three-way rule, through the command: search, reuse on a match, parameterise on a difference.

    The difference used is a protomer of the same compound: same topology, same charge method,
    same force field, different protonation state. Reusing there would be the silent failure the
    comparison exists to prevent, and nothing downstream could detect it.
    """
    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    catalog = tmp_path / "catalog"

    def build(name, smiles, parameters=None, aliases=None):
        work = tmp_path / name
        work.mkdir()
        (work / "in.smi").write_text(f"{smiles} molecule\n", encoding="utf-8")
        result = _build(work, Path("in.smi"), f"""
solute:
  kind: ligand
  residue_name: TYL
  compound_id: CHEMBL112
{f'  parameters: {parameters}' if parameters else ''}
{f'  aliases: {aliases}' if aliases else ''}
ligand_catalog:
  path: {catalog}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
        assert result.returncode == 0, result.stderr[-3000:]
        attached = _record(work / "build" / "built.log")["ligand_packages"]["attached"]
        return {**attached, "stderr": result.stderr}

    # 1. The catalog holds exactly this molecule, state, charge implementation and force field.
    # Aliases are names, not identity: they never prevent the match, and the reused package
    # keeps its own -- which the record and stderr both say, instead of dropping them silently.
    reused = build("reuse", "CC(=O)Nc1ccc(O)cc1", aliases=["paracetamol"])
    assert reused["how"] == "reused (catalog search)"
    assert reused["stated_aliases_not_applied"] == ["paracetamol"]
    assert "build-top: NOTE: solute.aliases ['paracetamol'] were NOT applied" in reused["stderr"]
    assert reused["reference"] == tyl.reference
    assert reused["charges_generated_in_this_build"] is False
    assert reused["matched_on"] == {"topology": "same", "protonation": "same",
                                    "charges": "same", "forcefield": "same"}
    assert reused["search"]["decision"] == "reuse"

    # 2. The phenolate: the same compound in another protonation state. No reuse.
    made = build("differs", "CC(=O)Nc1ccc([O-])cc1")
    assert made["how"] == "created"
    assert made["charges_generated_in_this_build"] is True
    assert made["reference"] != tyl.reference
    assert made["search"]["decision"] == "parameterise"
    considered = {row["reference"]: row for row in made["search"]["considered"]}
    assert considered[tyl.reference]["compared"]["topology"] == "same"
    assert "different chemical state" in considered[tyl.reference]["compared"]["protonation"]

    # 3. `generate` parameterises whatever the catalog holds, and searches nothing.
    forced = build("forced", "CC(=O)Nc1ccc(O)cc1", parameters="generate")
    assert forced["how"] == "created" and forced["search"] is None
    assert forced["reference"] == tyl.reference or forced["parameter_id"].startswith("param_")


@pytest.mark.slow
def test_a_stated_package_without_a_recorded_backend_builds_and_the_record_says_so(tmp_path):
    """A package written before the implementation was recorded: usable when NAMED, never searched.

    The build proceeds -- its numbers are unchanged -- and built.log states that the package does
    not record what produced its charges, so a reader is never told this was reuse of known ones.
    """
    import json

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    metadata = json.loads((tyl.path / "metadata.json").read_text())
    metadata["charges"].pop("backend_id")
    metadata["charges"].pop("backend", None)
    (tyl.path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (tyl.path / "parameter.config").unlink()

    work = tmp_path / "work"
    work.mkdir()
    (work / "in.smi").write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n", encoding="utf-8")
    result = _build(work, Path("in.smi"), f"""
solute:
  kind: ligand
  residue_name: TYL
  parameters: {tyl.reference}
ligand_catalog:
  path: {tmp_path / 'catalog'}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    attached = _record(work / "build" / "built.log")["ligand_packages"]["attached"]
    assert attached["how"] == "reused (stated reference)"
    assert attached["charges_generated_in_this_build"] is False
    assert attached["charge_backend_recorded"] is False
    assert attached["charge_backend_id"] is None

    # The same package is never the answer to a search.
    searched = tmp_path / "searched"
    searched.mkdir()
    (searched / "in.smi").write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n", encoding="utf-8")
    result = _build(searched, Path("in.smi"), f"""
solute:
  kind: ligand
  residue_name: TYL
  compound_id: CHEMBL112
  parameters: search
ligand_catalog:
  path: {tmp_path / 'catalog'}
solvent:
  model: TIP3P
  padding_nm: 1.0
""")
    assert result.returncode == 0, result.stderr[-2000:]
    attached = _record(searched / "build" / "built.log")["ligand_packages"]["attached"]
    assert attached["how"] == "created"
    row = next(c for c in attached["search"]["considered"] if c["reference"] == tyl.reference)
    assert "does not record which implementation" in row["compared"]["charges"]
