"""What `build-top` does with packages: create or reuse one for a ligand, and put its atoms in order.

A single-molecule build (`solute.kind: ligand` or `peptide-like`) either

  * REUSES a package named by `solute.parameters`: the prepared molecule must be the package's
    exact chemical state, its atoms are put into package order with package names, and no charge
    is generated; or
  * CREATES one from the prepared molecule, charges generated once, before any force field is
    built, so every later step of the same build loads the saved parameters instead of charging
    the molecule again.

Either way the package is copied into the build (`<build dir>/ligands/<compound>/<parameter>/`),
so the build does not depend on the catalog staying where it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .package import LigandPackage, PackageError

__all__ = ["LIGANDS_DIRNAME", "attach_ligand_package", "order_like_package",
           "write_prepared_molecule"]

#: Where a build keeps the packages it used, beside built.xml.
LIGANDS_DIRNAME = "ligands"


def order_like_package(mol, package: LigandPackage, *, where: str) -> tuple[np.ndarray, list[int]]:
    """Coordinates of *mol* (ANGSTROM, as read) in package atom order, iff it IS the package's state.

    The match is on the full chemical graph -- elements, formal charges, bond orders, every
    hydrogen -- so a different tautomer, protomer or compound has no match and is refused. Any
    match that exists is a symmetry of that graph and assigns identical chemistry; the first in a
    deterministic order is taken. Stereochemistry is then checked from the coordinates.

    Returns the ordered coordinates and the permutation: element i is the input atom that became
    package atom i. The permutation is recorded, so a rebuild outside md-tools applies exactly
    this one instead of searching for its own among the symmetric alternatives.
    """
    from networkx.algorithms.isomorphism import GraphMatcher

    from .identity import chemical_state
    from .mapping import MappingError, _check_stereo, _package_graph

    source = _package_graph(mol, heavy_only=False)
    target = _package_graph(package.mol, heavy_only=False)
    matcher = GraphMatcher(
        target, source,
        node_match=lambda a, b: (a["element"], a["formal_charge"]) == (b["element"], b["formal_charge"]),
        edge_match=lambda a, b: a["order"] == b["order"])
    match = next(matcher.isomorphisms_iter(), None)
    if match is None or mol.GetNumAtoms() != package.mol.GetNumAtoms():
        raise PackageError(
            f"{where}: the input molecule is {chemical_state(mol)['canonical_smiles']}, not the "
            f"chemical state of package {package.reference} "
            f"({package.metadata['chemical_state']['canonical_smiles']}). A package's parameters "
            f"belong to one exact state; a different tautomer, protomer or charge needs its own "
            f"package.")
    # Kept in the file's own unit: a round trip through nanometres changes the last digit of a
    # coordinate written back at the SDF's and PDB's precision.
    xyz = mol.GetConformer().GetPositions()
    permutation = [int(match[i]) for i in range(package.mol.GetNumAtoms())]
    ordered = np.array([xyz[j] for j in permutation])
    try:
        _check_stereo(package, ordered / 10.0, where)
    except MappingError as exc:
        raise PackageError(str(exc)) from exc
    return ordered, permutation


def write_prepared_molecule(package: LigandPackage, positions_angstrom: np.ndarray,
                            residue_name: str, sdf_path: Path, pdb_path: Path) -> None:
    """The prepared molecule, rewritten in package atom order with package atom names.

    Both files through RDKit, as the preparers write them, so a build that reuses a package
    produces the same prepared files as the build that created it.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(package.mol)
    conformer = mol.GetConformer()
    for i, xyz in enumerate(positions_angstrom):
        conformer.SetAtomPosition(i, Point3D(*map(float, xyz)))
    for atom, name in zip(mol.GetAtoms(), package.atom_names):
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(f"{name:<4}"[:4], residueName=residue_name,
                                                    residueNumber=1, isHeteroAtom=True))
    mol.SetProp("_Name", residue_name)
    for prop in list(mol.GetPropNames()):
        if prop != "_Name":
            mol.ClearProp(prop)
    Chem.MolToMolFile(mol, str(sdf_path))
    Chem.MolToPDBFile(mol, str(pdb_path))


def attach_ligand_package(*, prepared_sdf: Path, prepared_pdb: Path, settings: dict[str, Any],
                          staging: Path) -> Optional[dict[str, Any]]:
    """Create or reuse the package for a prepared single molecule; rewrite the prepared files.

    Returns None only for a force-field family packages do not support (GAFF), with the build then
    parameterising as before. Everything else either yields a package or refuses.
    """
    from rdkit import Chem

    from ..openmm.ligand_forcefield import is_gaff
    from .catalog import find_package
    from .identity import local_compound_id
    from .package import create_package, load_package

    resource = settings["forcefield"]
    if is_gaff(resource):
        if settings.get("parameters"):
            raise PackageError("solute.parameters names a package, but solute.ligand_forcefield "
                               "is GAFF; a package carries its own force field, so leave "
                               "ligand_forcefield at its default when reusing one")
        return None
    mol = Chem.MolFromMolFile(str(prepared_sdf), removeHs=False)
    if mol is None:
        raise PackageError(f"{prepared_sdf}: the prepared molecule could not be re-read")
    residue_name = settings["residue_name"]
    root = Path(staging) / LIGANDS_DIRNAME
    reference = settings.get("parameters")
    if reference:
        package = find_package(reference, settings.get("catalog_roots") or [])
        positions, permutation = order_like_package(mol, package, where="solute.parameters")
        package = load_package(package.copy_into(root))
        how = "reused"
    else:
        compound_id = settings.get("compound_id") or local_compound_id(mol)
        package = create_package(
            mol, compound_id=compound_id, residue_name=residue_name, out_root=root,
            forcefield=resource, charge_method=settings["charge_method"],
            aliases=settings.get("aliases") or (),
            source={"input": settings.get("input_name"), "input_sha256": settings.get("input_sha256")})
        positions, permutation = order_like_package(mol, package, where="the prepared molecule")
        how = "created"
    rewritten = not _already_in_package_order(mol, package, Path(prepared_pdb))
    if rewritten:
        write_prepared_molecule(package, positions, residue_name, Path(prepared_sdf),
                                Path(prepared_pdb))
    return {"package": package, "how": how,
            "record": {**package.summary(), "how": how,
                       "copied_to": f"{LIGANDS_DIRNAME}/{package.compound_id}/{package.parameter_id}",
                       "charges_generated_in_this_build": how == "created",
                       "prepared_atom_for_package_atom": permutation,
                       "prepared_files_rewritten_in_package_order": rewritten}}


def _already_in_package_order(mol, package: LigandPackage, prepared_pdb: Path) -> bool:
    """Whether the prepared files already ARE package order and names, and need no rewrite.

    Rewriting them anyway would re-derive the PDB from the SDF, whose coordinates carry one
    decimal fewer, and move the last digit of a coordinate the preparer wrote at full precision.
    """
    from openmm import app

    same_elements = [a.GetSymbol() for a in mol.GetAtoms()] == [
        a.GetSymbol() for a in package.mol.GetAtoms()]
    same_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()} \
        == {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in package.mol.GetBonds()}
    if not (same_elements and same_bonds):
        return False
    names = [a.name for a in app.PDBFile(str(prepared_pdb)).topology.atoms()]
    return names == list(package.atom_names)


def packages_from_dirs(paths: Sequence[str]) -> list[LigandPackage]:
    from .package import load_package

    return [load_package(Path(p)) for p in paths]


def attach_for_build(cfg: dict[str, Any], staging: Path, *, solute_sdf: Path,
                     solute_pdb: Path) -> Optional[dict[str, Any]]:
    """The builders' one call: attach the package `cfg["ligand_package"]` asks for, if any.

    Sets `cfg["forcefield"]["ligand_packages"]`, which `build_forcefield` reads on every later
    step of the build, and returns the record for built.log. A cfg without `ligand_package`
    settings (a builder called outside build-top) is left exactly as it was.
    """
    settings = cfg.get("ligand_package")
    if not settings:
        return None
    attached = attach_ligand_package(prepared_sdf=Path(solute_sdf), prepared_pdb=Path(solute_pdb),
                                     settings=settings, staging=Path(staging))
    if attached is None:
        cfg["forcefield"]["ligand_packages"] = []
        return {"how": "not packaged",
                "reason": f"{settings['forcefield']}: GAFF packages are not supported yet; the "
                          f"ligand was parameterised through the template generator as before"}
    cfg["forcefield"]["ligand_packages"] = [str(attached["package"].path)]
    return attached["record"]
