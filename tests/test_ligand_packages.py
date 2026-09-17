"""Reusable ligand parameter packages: identity, verification, and no regeneration on reuse.

The fast tests never run a charge calculation. They parameterise a molecule with Sage using
charges the test supplies, build the System that a real build would have produced, and recover a
package from it -- the route the existing paracetamol dataset takes. The charge-generating route
(`create_package`, AM1-BCC through sqm) is `slow`, and its test checks the property the whole
design rests on: generating from the molecule and recovering from a System built with those
charges give the SAME parameter identity.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def _molecule(smiles: str = PARACETAMOL, seed: int = 20260917):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mol


def _charges(mol, *, perturb: float = 0.0) -> list[float]:
    """Deterministic test charges summing exactly to the formal charge (NOT a real charge model)."""
    from rdkit.Chem import rdPartialCharges

    work = type(mol)(mol)
    rdPartialCharges.ComputeGasteigerCharges(work)
    values = [round(float(a.GetProp("_GasteigerCharge")), 4) for a in work.GetAtoms()]
    values[0] += perturb
    formal = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    values[-1] += formal - sum(values)
    return values


def _system_with_charges(mol, charges, *, hmr: bool = False, constraints: bool = True):
    """What build-top would have serialised for this molecule: Sage, given charges, HBonds, HMR."""
    import numpy as np
    from openff.toolkit import Molecule
    from openff.units import unit as off_unit
    from openmm import app, unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    offmol = Molecule.from_rdkit(mol, hydrogens_are_explicit=True)
    offmol.partial_charges = np.asarray(charges) * off_unit.elementary_charge
    generator = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield="openff-2.2.1")
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(generator.generator)
    topology = offmol.to_topology().to_openmm()
    return forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds if constraints else None,
        **({"hydrogenMass": 3.024 * unit.amu} if hmr else {}))


@pytest.fixture
def no_charge_generation(monkeypatch):
    from openff.toolkit import Molecule

    def refuse(*args, **kwargs):
        raise AssertionError("a charge calculation ran; reuse must never generate charges")

    monkeypatch.setattr(Molecule, "assign_partial_charges", refuse)


def _import(mol, charges, root: Path, **kwargs):
    from md_tools.ligands import import_package_from_system

    system = kwargs.pop("system", None) or _system_with_charges(mol, charges, hmr=True)
    kwargs.setdefault("charge_provenance", {"scheme": "am1bcc", "backend_id": "ambertools-sqm"})
    return import_package_from_system(
        mol, system=system, atom_indices=range(mol.GetNumAtoms()), compound_id="CHEMBL112",
        residue_name="TYL", out_root=root, forcefield="sage-2.2.1", charge_method="am1bcc",
        aliases=["paracetamol", "acetaminophen"], **kwargs)


def test_a_recovered_package_verifies_and_its_identity_comes_from_its_contents(
        tmp_path, no_charge_generation):
    from md_tools.ligands import load_package

    mol = _molecule()
    package = _import(mol, _charges(mol), tmp_path / "catalog")
    assert package.path == tmp_path / "catalog" / "CHEMBL112" / package.parameter_id
    assert package.parameter_id.startswith("param_") and len(package.parameter_id) == 18
    assert package.template_name == f"MDT_{package.parameter_id}"
    assert sorted(p.name for p in package.path.iterdir()) == [
        "metadata.json", "molecule.sdf", "parameter.config", "parameters.ffxml"]
    reloaded = load_package(package.path)
    assert reloaded.package_sha256 == package.package_sha256
    assert reloaded.metadata["charges"]["source"] == "imported"
    # The same inputs give the same identity, into a different root.
    again = _import(mol, _charges(mol), tmp_path / "elsewhere")
    assert again.parameter_id == package.parameter_id


def test_hydrogen_mass_repartitioning_in_the_source_does_not_change_the_package(
        tmp_path, no_charge_generation):
    mol = _molecule()
    charges = _charges(mol)
    with_hmr = _import(mol, charges, tmp_path / "a",
                       system=_system_with_charges(mol, charges, hmr=True),
                       charge_provenance={"scheme": "am1bcc", "backend_id": "ambertools-sqm",
                                          "backend": "stated by the test"})
    assert with_hmr.metadata["charges"]["backend"] == "stated by the test"
    assert with_hmr.metadata["charges"]["source"] == "imported"
    without = _import(mol, charges, tmp_path / "b",
                      system=_system_with_charges(mol, charges, hmr=False, constraints=False))
    assert with_hmr.parameter_id == without.parameter_id
    masses = {row[0]: row[1] for row in with_hmr.table["atoms"]}
    assert masses["H"] == pytest.approx(1.007947, abs=1e-3)


def test_a_scaled_system_is_refused_as_a_parameter_source(tmp_path, no_charge_generation):
    from openmm import PeriodicTorsionForce

    from md_tools.ligands import PackageError

    mol = _molecule()
    charges = _charges(mol)
    system = _system_with_charges(mol, charges)
    torsions = next(f for f in system.getForces() if isinstance(f, PeriodicTorsionForce))
    for k in range(torsions.getNumTorsions()):
        *atoms, n, phase, force_k = torsions.getTorsionParameters(k)
        torsions.setTorsionParameters(k, *atoms, n, phase, force_k * 0.25)
    with pytest.raises(PackageError, match="do not reproduce"):
        _import(mol, charges, tmp_path / "catalog", system=system)
    assert not (tmp_path / "catalog" / "CHEMBL112").exists() or not any(
        p for p in (tmp_path / "catalog" / "CHEMBL112").iterdir() if not p.name.startswith("."))


@pytest.mark.parametrize("damage", ["ffxml-charge", "metadata-charge", "renamed-directory",
                                    "wrong-compound-directory", "extra-file"])
def test_a_modified_package_does_not_load(tmp_path, damage, no_charge_generation):
    from md_tools.ligands import PackageError, load_package

    mol = _molecule()
    package = _import(mol, _charges(mol), tmp_path / "catalog")
    path = package.path
    if damage == "ffxml-charge":
        text = (path / "parameters.ffxml").read_text()
        first = text.index('charge="') + len('charge="')
        (path / "parameters.ffxml").write_text(text[:first] + "9" + text[first:])
    elif damage == "metadata-charge":
        metadata = json.loads((path / "metadata.json").read_text())
        metadata["atoms"][0]["partial_charge_e"] += 0.1
        (path / "metadata.json").write_text(json.dumps(metadata))
    elif damage == "renamed-directory":
        path = path.rename(path.with_name("param_000000000000"))
    elif damage == "wrong-compound-directory":
        target = tmp_path / "catalog" / "CHEMBL999" / package.parameter_id
        target.parent.mkdir(parents=True)
        path = path.rename(target)
    elif damage == "extra-file":
        (path / "notes.txt").write_text("x")
    with pytest.raises(PackageError):
        load_package(path)


def test_two_packages_of_one_molecule_load_together_without_name_collisions(
        tmp_path, no_charge_generation):
    from openmm import app

    from md_tools.ligands.parameters import topology_for_molecule

    mol = _molecule()
    first = _import(mol, _charges(mol), tmp_path / "catalog")
    second = _import(mol, _charges(mol, perturb=0.01), tmp_path / "catalog")
    assert first.parameter_id != second.parameter_id
    assert first.metadata["chemical_state"]["digest"] == second.metadata["chemical_state"]["digest"]
    forcefield = app.ForceField(io.StringIO(first.ffxml_text), io.StringIO(second.ffxml_text))
    topology = topology_for_molecule(mol, first.atom_names, "TYL")
    residue = next(iter(topology.residues()))
    for package in (first, second):
        system = forcefield.createSystem(topology, residueTemplates={residue: package.template_name})
        assert system.getNumParticles() == mol.GetNumAtoms()


def test_a_package_loads_beside_amber_force_fields_with_the_same_one_four_scales(
        tmp_path, no_charge_generation):
    """Loaded after ff14SB, OpenMM keeps the FIRST 1-4 scale; the package must already agree."""
    from openmm import NonbondedForce, app

    from md_tools.ligands.parameters import topology_for_molecule

    mol = _molecule()
    package = _import(mol, _charges(mol), tmp_path / "catalog")
    assert package.conventions["coulomb14scale"] == 5.0 / 6.0
    topology = topology_for_molecule(mol, package.atom_names, "TYL")
    residue = next(iter(topology.residues()))
    alone = app.ForceField(io.StringIO(package.ffxml_text)).createSystem(
        topology, residueTemplates={residue: package.template_name})
    beside = app.ForceField("amber14-all.xml", "amber14/tip3p.xml",
                            io.StringIO(package.ffxml_text)).createSystem(
        topology, residueTemplates={residue: package.template_name})

    def exceptions(system):
        force = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        return [force.getExceptionParameters(k) for k in range(force.getNumExceptions())]

    assert [str(e) for e in exceptions(alone)] == [str(e) for e in exceptions(beside)]


def test_different_protonation_states_are_different_chemical_states():
    from md_tools.ligands import chemical_state

    acid = chemical_state(_molecule("CC(=O)O"))
    base = chemical_state(_molecule("CC(=O)[O-]"))
    assert acid["digest"] != base["digest"]
    assert acid["fixed_h_inchikey"] != base["fixed_h_inchikey"]
    assert base["net_formal_charge"] == -1
    # A local compound id groups them: it is the substance, not the state.
    from md_tools.ligands import local_compound_id

    assert local_compound_id(_molecule("CC(=O)O")) == local_compound_id(_molecule("CC(=O)[O-]"))


def test_tautomers_are_different_chemical_states():
    from md_tools.ligands import chemical_state

    assert (chemical_state(_molecule("O=C1C=CC=CN1"))["digest"]
            != chemical_state(_molecule("Oc1ccccn1"))["digest"])


def test_refusals_before_anything_is_written(tmp_path, no_charge_generation):
    from rdkit import Chem

    from md_tools.ligands import CompoundIdError, PackageError

    mol = _molecule()
    charges = _charges(mol)
    system = _system_with_charges(mol, charges)
    from md_tools.ligands import import_package_from_system

    sqm = {"scheme": "am1bcc", "backend_id": "ambertools-sqm"}
    with pytest.raises(PackageError, match="backend_id"):
        import_package_from_system(mol, system=system, atom_indices=range(mol.GetNumAtoms()),
                                   compound_id="CHEMBL112", residue_name="TYL",
                                   out_root=tmp_path / "c", forcefield="sage-2.2.1",
                                   charge_method="am1bcc")
    with pytest.raises(CompoundIdError, match="aliases"):
        import_package_from_system(mol, system=system, atom_indices=range(mol.GetNumAtoms()),
                                   compound_id="paracetamol", residue_name="TYL",
                                   out_root=tmp_path / "c", forcefield="sage-2.2.1",
                                   charge_method="am1bcc", charge_provenance=sqm)

    with pytest.raises(PackageError, match="GAFF"):
        import_package_from_system(mol, system=system, atom_indices=range(mol.GetNumAtoms()),
                                   compound_id="CHEMBL112", residue_name="TYL",
                                   out_root=tmp_path / "c", forcefield="gaff2",
                                   charge_method="am1bcc", charge_provenance=sqm)
    with pytest.raises(PackageError, match="implicit hydrogens"):
        import_package_from_system(Chem.MolFromSmiles(PARACETAMOL), system=system,
                                   atom_indices=range(11), compound_id="CHEMBL112",
                                   residue_name="TYL", out_root=tmp_path / "c",
                                   forcefield="sage-2.2.1", charge_method="am1bcc",
                                   charge_provenance=sqm)
    chiral = _molecule("CC(N)C(=O)O")                    # alanine with no stereo stated
    flat = Chem.Mol(chiral)
    flat.RemoveAllConformers()
    for atom in flat.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    with pytest.raises(PackageError, match="stereo"):
        import_package_from_system(flat, system=system, atom_indices=range(flat.GetNumAtoms()),
                                   compound_id="CHEMBL112", residue_name="ALX",
                                   out_root=tmp_path / "c", forcefield="sage-2.2.1",
                                   charge_method="am1bcc", charge_provenance=sqm)
    assert not (tmp_path / "c").exists()


@pytest.mark.slow
def test_generating_once_and_recovering_from_the_built_system_give_one_identity(tmp_path):
    """The reason packages are worth having: one parameter id however the parameters arrived."""
    from openmm import NonbondedForce, unit

    from md_tools.ligands import create_package

    mol = _molecule()
    generated = create_package(mol, compound_id="CHEMBL112", residue_name="TYL",
                               out_root=tmp_path / "generated", aliases=["paracetamol"])
    assert generated.metadata["charges"]["source"] == "generated"
    charges = [atom["partial_charge_e"] for atom in generated.metadata["atoms"]]
    recovered = _import(mol, charges, tmp_path / "recovered")
    assert recovered.parameter_id == generated.parameter_id
    assert recovered.metadata["parameter_digest"] == generated.metadata["parameter_digest"]
