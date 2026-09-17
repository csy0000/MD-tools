"""Compound identity and chemical-state identity.

A COMPOUND identifier names a substance: `CHEMBL112` is paracetamol whichever tautomer or protomer
of it is simulated. It organises the catalog and is what a person searches for, together with the
aliases. It never authorises reuse on its own.

A CHEMICAL STATE is the exact graph that was parameterised: every atom including every hydrogen,
formal charges, bond orders, isotopes and stereochemistry. Two protomers of one compound are two
states, even though ChEMBL, a standard InChI or a normalising toolkit would merge them. The state
is what a ligand instance is checked against before a package's parameters are applied to it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

__all__ = [
    "CHEMICAL_STATE_SCHEME",
    "COMPOUND_ID_PATTERN",
    "CompoundIdError",
    "chemical_state",
    "check_compound_id",
    "local_compound_id",
]

#: The external identifier forms accepted, and the local form for a compound no database lists.
#: Both are single safe path segments, because the compound id is a catalog directory name.
COMPOUND_ID_PATTERN = re.compile(r"CHEMBL[1-9][0-9]*|LOCAL-[A-Z]{14}")

#: Versioned so that a change to what a state digest covers is a new scheme, not a silent change
#: of meaning for every package already written.
CHEMICAL_STATE_SCHEME = "md-tools-chemical-state/1"


class CompoundIdError(ValueError):
    """A compound identifier that is not one of the accepted forms."""


def check_compound_id(text: str) -> str:
    """The identifier as given, or a refusal naming the accepted forms."""
    value = str(text)
    if not COMPOUND_ID_PATTERN.fullmatch(value):
        raise CompoundIdError(
            f"compound id {value!r} is not an accepted form. Use a ChEMBL identifier "
            f"(`CHEMBL112`), or for a compound no database lists the local form "
            f"`LOCAL-<first block of its standard InChIKey>` -- `local_compound_id` computes it. "
            f"Names such as 'paracetamol' belong in `aliases`, where they are searchable; they "
            f"are not identifiers, because two people spell them differently.")
    return value


def local_compound_id(mol) -> str:
    """`LOCAL-` plus the connectivity block of the standard InChIKey.

    The first block encodes the skeleton with mobile hydrogens, so the tautomers and protomers of
    one substance share it -- exactly the grouping a compound identifier is for. The chemical
    state, not this, tells the states apart.
    """
    from rdkit import Chem

    key = Chem.MolToInchiKey(Chem.RemoveHs(mol))
    if not key:
        raise CompoundIdError("RDKit produced no InChIKey for this molecule, so no local compound "
                              "id can be derived; supply an external identifier")
    return f"LOCAL-{key.split('-')[0]}"


def unassigned_stereo(mol) -> list[dict[str, Any]]:
    """Stereo elements the molecule has but does not specify.

    A parameter package must describe ONE stereoisomer: charges differ between diastereomers, and
    an instance whose 3D pose is the other enantiomer must be refused rather than given parameters
    fitted to something else.
    """
    from rdkit import Chem

    missing = []
    for element in Chem.FindPotentialStereo(Chem.Mol(mol)):
        if element.specified == Chem.StereoSpecified.Unspecified:
            missing.append({"type": str(element.type), "centered_on": int(element.centeredOn)})
    return missing


def chemical_state(mol) -> dict[str, Any]:
    """The identity of one exact chemical graph.

    `mol` must carry every hydrogen explicitly and have its stereochemistry assigned (from 3D
    coordinates when it has them). The digest covers RDKit's canonical isomeric SMILES of the
    explicit-hydrogen graph, which encodes elements, formal charges, isotopes, bond orders
    (aromaticity perceived), hydrogen placement and stereochemistry. That string depends on the
    RDKit version, so the fixed-hydrogen InChI -- which does not merge tautomers and is stable
    across toolkit releases -- is recorded beside it as the version-independent cross-check.
    """
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    if any(atom.GetAtomicNum() > 1 and atom.GetNumImplicitHs() for atom in mol.GetAtoms()):
        raise ValueError("the molecule has implicit hydrogens; a chemical state is defined on an "
                         "explicit-hydrogen graph, because hydrogens are what tautomers and "
                         "protomers differ by")
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    fixed_h = Chem.MolToInchi(mol, options="/FixedH")
    standard = Chem.MolToInchi(mol)
    digest = hashlib.sha256(f"{CHEMICAL_STATE_SCHEME}|{smiles}".encode()).hexdigest()
    return {
        "scheme": CHEMICAL_STATE_SCHEME,
        "digest": digest,
        "canonical_smiles": smiles,
        "fixed_h_inchi": fixed_h,
        "fixed_h_inchikey": Chem.InchiToInchiKey(fixed_h) if fixed_h else None,
        "standard_inchikey": Chem.InchiToInchiKey(standard) if standard else None,
        "formula": CalcMolFormula(mol),
        "net_formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "n_atoms": int(mol.GetNumAtoms()),
        "n_heavy_atoms": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)),
    }
