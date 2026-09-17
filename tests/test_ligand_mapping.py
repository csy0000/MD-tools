"""Ligand instances in a structure, mapped onto packages: selectors, symmetry, hydrogens, reuse.

The structure is synthetic and written through PDBFile, so the mapping reads exactly what a
deposited file gives it: heavy atoms with element columns, no bond orders, no hydrogens on the
ligands, and atom order and names that have nothing to do with the package's.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")
pytest.importorskip("networkx")

from test_ligand_packages import _charges, _import, _molecule, _system_with_charges  # noqa: E402

ALA_PDB = Path(__file__).resolve().parent / "data" / "ALA.pdb"


def _package(tmp_path, smiles, compound, residue, *, perturb=0.0):
    from md_tools.ligands import import_package_from_system

    mol = _molecule(smiles)
    charges = _charges(mol, perturb=perturb)
    return import_package_from_system(
        mol, system=_system_with_charges(mol, charges), atom_indices=range(mol.GetNumAtoms()),
        compound_id=compound, residue_name=residue, out_root=tmp_path / "catalog",
        forcefield="sage-2.2.1", charge_method="am1bcc")


def _rotation(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _write_structure(path: Path, ligands, *, peptide=True, hydrogens=False, seed=7):
    """ALA dipeptide (chain A) plus ligand heavy atoms: (chain, resid, resname, package, offset)."""
    from openmm import app, unit

    ala = app.PDBFile(str(ALA_PDB))
    topology = app.Topology()
    positions = []
    if peptide:
        chain = topology.addChain("A")
        for residue in ala.topology.residues():
            new = topology.addResidue(residue.name, chain, residue.id)
            for atom in residue.atoms():
                topology.addAtom(atom.name, atom.element, new)
                positions.append(ala.positions[atom.index].value_in_unit(unit.nanometer))
    rng = np.random.default_rng(seed)
    name_maps = []
    for n, (chain_id, resid, resname, package, offset) in enumerate(ligands):
        mol = package.mol
        xyz = mol.GetConformer().GetPositions() / 10.0
        xyz = (xyz - xyz.mean(axis=0)) @ _rotation(seed + n).T + np.asarray(offset)
        order = [a.GetIdx() for a in mol.GetAtoms() if hydrogens or a.GetAtomicNum() > 1]
        rng.shuffle(order)
        chain = topology.addChain(chain_id)
        residue = topology.addResidue(resname, chain, resid)
        counts = {}
        name_map = {}
        for index in order:
            atom = mol.GetAtomWithIdx(index)
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
            # Deliberately NOT the package's names: deposited names are someone else's convention.
            name = f"{symbol}{counts[symbol] + 10}"
            topology.addAtom(name, app.Element.getBySymbol(symbol), residue)
            name_map[name] = package.atom_names[index]
            positions.append(xyz[index])
        name_maps.append(name_map)
    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, np.array(positions) * 10.0 * unit.angstrom, handle,
                              keepIds=True)
    pdb = app.PDBFile(str(path))
    pdb.name_maps = name_maps          # deposited name -> package name, per ligand, for the tests
    return pdb


def test_repeated_copies_and_a_second_species_map_with_equivalent_parameters(tmp_path):
    from openmm import app, unit

    from md_tools.ligands.mapping import (assert_instances_unchanged, load_packages_into,
                                          map_ligands)
    from md_tools.ligands.package import _subsystem_table, _compare_tables
    from md_tools.ligands.parameters import ligand_system

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    eth = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    pdb = _write_structure(tmp_path / "complex.pdb", [
        ("B", "201", "TYL", tyl, (2.5, 0, 0)),
        ("C", "201", "TYL", tyl, (0, 2.5, 0)),
        ("B", "202", "EOH", eth, (0, 0, 2.5)),
    ])
    entries = [
        {"select": {"chain": "B", "resid": "201"}, "parameters": tyl.reference},
        {"select": {"chain": "C", "resid": "201", "insertion_code": ""}, "parameters": tyl.reference},
        {"select": {"chain": "B", "resid": "202"}, "parameters": eth.reference},
    ]
    mapped = map_ligands(pdb.topology, pdb.positions, entries,
                         {tyl.reference: tyl, eth.reference: eth})
    assert [i.key for i in mapped.instances] == [("B", "201", ""), ("C", "201", ""),
                                                 ("B", "202", "")]
    assert [p.reference for p in mapped.packages] == [tyl.reference, eth.reference]
    assert mapped.topology.getNumAtoms() == 22 + 2 * 20 + 9

    forcefield = app.ForceField("amber14-all.xml")
    report = load_packages_into(forcefield, mapped.packages)
    assert report["force_field_nonbonded"]["coulomb14scale"] == 5.0 / 6.0
    templates = mapped.residue_templates(mapped.topology)
    system = forcefield.createSystem(mapped.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=None, residueTemplates=templates)

    positions = np.array(mapped.positions.value_in_unit(unit.nanometer))
    for instance, residue in zip(mapped.instances, mapped.resolve(mapped.topology).values()):
        indices = [a.index for a in residue.atoms()]
        package = instance.package
        # Every parameter of this instance in the complex System is the package's own.
        table, _ = _subsystem_table(system, package.mol, indices,
                                    package.conventions["coulomb14scale"],
                                    package.conventions["lj14scale"])
        assert _compare_tables(table, package.table, where=instance.selector.label(),
                               skip_masses=True) == []
        # And the isolated-ligand energy at these coordinates is the package's own.
        from openmm import Context, Platform, VerletIntegrator

        def energy(sys_, xyz):
            context = Context(sys_, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
            context.setPositions(xyz * unit.nanometer)
            return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)

        modeller = app.Modeller(mapped.topology, mapped.positions)
        modeller.delete([r for r in mapped.topology.residues() if r.index != residue.index])
        alone = forcefield.createSystem(
            modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None,
            residueTemplates={next(iter(modeller.topology.residues())): package.template_name})
        reference = ligand_system(package.ffxml_text, package.mol, package.atom_names,
                                  package.template_name)
        assert energy(alone, positions[indices]) == pytest.approx(
            energy(reference, positions[indices]), rel=1e-9, abs=1e-9)
        assert instance.symmetry["stereochemistry"]["checked"]
        assert [m["package_index"] for m in instance.heavy_atom_map] == sorted(
            a.GetIdx() for a in package.mol.GetAtoms() if a.GetAtomicNum() > 1)

    # Nothing about the instances changes through a PDB round trip.
    buffer = io.StringIO()
    app.PDBFile.writeFile(mapped.topology, mapped.positions, buffer, keepIds=True)
    reread = app.PDBFile(io.StringIO(buffer.getvalue()))
    # PDBFile re-derives ligand bonds only from CONECT, so compare against the mapped topology.
    assert_instances_unchanged(mapped, mapped.topology, reread.positions, step="PDB round trip")
    record = mapped.record(mapped.topology)
    assert record["instances"][1]["resolved"]["atom_indices"][0] == 22 + 20


def test_a_ring_flip_is_equivalent_but_a_carboxylic_acid_is_ambiguous(tmp_path):
    from md_tools.ligands.mapping import MappingError, map_ligands

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    pdb = _write_structure(tmp_path / "tyl.pdb", [("B", "1", "TYL", tyl, (0, 0, 0))],
                           peptide=False)
    mapped = map_ligands(pdb.topology, pdb.positions,
                         [{"select": {"chain": "B", "resid": "1"}, "parameters": tyl.reference}],
                         {tyl.reference: tyl})
    assert mapped.instances[0].symmetry["n_graph_matches"] == 2      # the phenyl flip

    acid = _package(tmp_path, "CC(=O)O", "CHEMBL539", "ACY")
    pdb = _write_structure(tmp_path / "acid.pdb", [("B", "1", "ACY", acid, (0, 0, 0))],
                           peptide=False)
    entry = {"select": {"chain": "B", "resid": "1"}, "parameters": acid.reference}
    with pytest.raises(MappingError, match="NOT chemically equivalent"):
        map_ligands(pdb.topology, pdb.positions, [entry], {acid.reference: acid})
    # Stating the mapping resolves it, and a wrong one is refused.
    names = pdb.name_maps[0]
    mapped = map_ligands(pdb.topology, pdb.positions, [{**entry, "atom_map": names}],
                         {acid.reference: acid})
    assert mapped.instances[0].symmetry["source"].startswith("atom_map")
    assert {m["deposited_name"]: m["package_name"]
            for m in mapped.instances[0].heavy_atom_map} == names
    swapped = dict(names)
    carbons = [k for k, v in names.items() if v.startswith("C")]
    swapped[carbons[0]], swapped[carbons[1]] = names[carbons[1]], names[carbons[0]]
    with pytest.raises(MappingError, match="not an element- and bond-preserving"):
        map_ligands(pdb.topology, pdb.positions, [{**entry, "atom_map": swapped}],
                    {acid.reference: acid})


def test_a_carboxylate_resonance_pair_is_equivalent_when_its_parameters_are(tmp_path):
    from md_tools.ligands.mapping import map_ligands

    acetate = _package(tmp_path, "CC(=O)[O-]", "CHEMBL1200", "ACT")
    pdb = _write_structure(tmp_path / "act.pdb", [("B", "1", "ACT", acetate, (0, 0, 0))],
                           peptide=False)
    mapped = map_ligands(pdb.topology, pdb.positions,
                         [{"select": {"chain": "B", "resid": "1"}, "parameters": acetate.reference}],
                         {acetate.reference: acetate})
    assert mapped.instances[0].symmetry["n_graph_matches"] == 2

def test_selectors_that_do_not_name_exactly_one_residue_are_refused(tmp_path):
    from md_tools.ligands.mapping import MappingError, map_ligands

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    pdb = _write_structure(tmp_path / "dup.pdb", [("B", "201", "TYL", tyl, (2.5, 0, 0)),
                                                  ("B", "201", "TYL", tyl, (0, 2.5, 0))])
    packages = {tyl.reference: tyl}
    with pytest.raises(MappingError, match="matches 2 residues"):
        map_ligands(pdb.topology, pdb.positions,
                    [{"select": {"chain": "B", "resid": "201"}, "parameters": tyl.reference}],
                    packages)
    with pytest.raises(MappingError, match="matches 0 residues"):
        map_ligands(pdb.topology, pdb.positions,
                    [{"select": {"chain": "Z", "resid": "201"}, "parameters": tyl.reference}],
                    packages)
    with pytest.raises(MappingError, match="quoted string"):
        map_ligands(pdb.topology, pdb.positions,
                    [{"select": {"chain": "B", "resid": 201}, "parameters": tyl.reference}],
                    packages)


def test_a_different_chemical_state_or_compound_is_refused(tmp_path):
    from md_tools.ligands.mapping import MappingError, map_ligands

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    eth = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    pdb = _write_structure(tmp_path / "eoh.pdb", [("B", "1", "EOH", eth, (0, 0, 0))],
                           peptide=False)
    with pytest.raises(MappingError, match="heavy atoms"):
        map_ligands(pdb.topology, pdb.positions,
                    [{"select": {"chain": "B", "resid": "1"}, "parameters": tyl.reference}],
                    {tyl.reference: tyl})
    # Deposited hydrogens that disagree with the package's: the phenolate is a different state.
    phenolate = _package(tmp_path, "CC(=O)Nc1ccc([O-])cc1", "CHEMBL112", "TYL")
    pdb = _write_structure(tmp_path / "withh.pdb", [("B", "1", "TYL", tyl, (0, 0, 0))],
                           peptide=False, hydrogens=True)
    with pytest.raises(MappingError, match="hydrogen"):
        map_ligands(pdb.topology, pdb.positions,
                    [{"select": {"chain": "B", "resid": "1"}, "parameters": phenolate.reference}],
                    {phenolate.reference: phenolate})
    # With the right package, deposited hydrogens are kept where they were.
    mapped = map_ligands(pdb.topology, pdb.positions,
                         [{"select": {"chain": "B", "resid": "1"}, "parameters": tyl.reference}],
                         {tyl.reference: tyl})
    assert mapped.instances[0].hydrogen_source.startswith("deposited hydrogens")


def test_the_mirror_image_pose_is_refused(tmp_path):
    from md_tools.ligands.mapping import MappingError, map_ligands

    lactate = _package(tmp_path, "C[C@H](O)C(=O)[O-]", "CHEMBL1200559", "LAC")
    pdb = _write_structure(tmp_path / "lac.pdb", [("B", "1", "LAC", lactate, (0, 0, 0))],
                           peptide=False)
    entry = [{"select": {"chain": "B", "resid": "1"}, "parameters": lactate.reference}]
    map_ligands(pdb.topology, pdb.positions, entry, {lactate.reference: lactate})
    from openmm import app, unit

    mirrored = np.array(pdb.positions.value_in_unit(unit.nanometer)) * np.array([-1.0, 1.0, 1.0])
    with pytest.raises(MappingError, match="stereochemistry"):
        map_ligands(pdb.topology, mirrored * unit.nanometer, entry, {lactate.reference: lactate})


def test_a_covalent_attachment_is_refused_and_a_metal_contact_is_recorded(tmp_path):
    from openmm import app, unit

    from md_tools.ligands.mapping import MappingError, map_ligands

    eth = _package(tmp_path, "CCO", "CHEMBL545", "EOH")
    pdb = _write_structure(tmp_path / "eoh.pdb", [("B", "1", "EOH", eth, (0, 0, 0))],
                           peptide=False)
    xyz = np.array(pdb.positions.value_in_unit(unit.nanometer))
    residue = next(iter(pdb.topology.residues()))
    oxygen = next(a for a in residue.atoms() if a.element.symbol == "O")
    carbon = [a for a in residue.atoms() if a.element.symbol == "C"]
    direction = xyz[oxygen.index] - np.mean([xyz[c.index] for c in carbon], axis=0)
    direction /= np.linalg.norm(direction)

    def with_extra(element, distance):
        modeller = app.Modeller(pdb.topology, pdb.positions)
        extra = app.Topology()
        chain = extra.addChain("M")
        res = extra.addResidue("ZN" if element == "Zn" else "CYS", chain, "900")
        extra.addAtom("ZN" if element == "Zn" else "SG", app.Element.getBySymbol(element), res)
        modeller.add(extra, [(xyz[oxygen.index] + direction * distance) * 10.0] * unit.angstrom)
        return modeller

    entry = [{"select": {"chain": "B", "resid": "1"}, "parameters": eth.reference}]
    metal = with_extra("Zn", 0.20)
    mapped = map_ligands(metal.topology, metal.positions, entry, {eth.reference: eth})
    assert mapped.instances[0].coordination_contacts[0]["distance_nm"] == pytest.approx(0.2, abs=1e-3)
    sulfur = with_extra("S", 0.17)
    with pytest.raises(MappingError, match="covalent"):
        map_ligands(sulfur.topology, sulfur.positions, entry, {eth.reference: eth})


def test_an_incompatible_one_four_convention_is_refused_before_loading(tmp_path):
    from openmm import app

    from md_tools.ligands.mapping import MappingError, load_packages_into

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    other = app.ForceField(io.StringIO(
        '<ForceField><NonbondedForce coulomb14scale="1.0" lj14scale="1.0"/></ForceField>'))
    with pytest.raises(MappingError, match="1-4 scales"):
        load_packages_into(other, [tyl])


def test_unmapped_non_standard_residues_are_listed(tmp_path):
    from md_tools.ligands.mapping import LigandSelector, unmapped_residues

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    pdb = _write_structure(tmp_path / "c.pdb", [("B", "201", "TYL", tyl, (2.5, 0, 0)),
                                                ("C", "201", "TYL", tyl, (0, 2.5, 0))])
    known = {"ACE", "ALA", "NME", "HOH"}
    left = unmapped_residues(pdb.topology, [LigandSelector("B", "201")], known_residue_names=known)
    assert left == [{"chain": "C", "resid": "201", "insertion_code": "", "residue_name": "TYL"}]
