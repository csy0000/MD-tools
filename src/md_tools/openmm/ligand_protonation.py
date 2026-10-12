"""Protonation of a small molecule BEFORE it is parameterised: Open Babel's pH model.

WHY THIS EXISTS. A ligand is built from the molecule its input encodes, and nothing used to say
whether that molecule is the one present at the pH being simulated: `CC(=O)O` was built as the
neutral acid whatever the pH, and the charges, the parameter package and every later result
described a species that is mostly absent at 7. The protein half has had PROPKA for this
(`md_tools.openmm.protonation`); the molecule half now has Open Babel.

WHAT IT DOES. `obabel -p <pH>` is `OBMol.AddHydrogens(polaronly=False, correctForPH=True, pH)`:
a table of functional-group transformations (carboxylic and sulfonic acids to anions, aliphatic
amines and amidines/guanidines to cations, ...) followed by hydrogen perception. This module makes
that call, so the CLI and the library cannot disagree, and records which version made it.

WHAT IT DOES NOT DO, and the record says so rather than leaving it to be assumed:
  * It is a TABLE, not a pKa predictor. It does not consider the molecule's environment, so a
    phosphate, an imidazole, a phenol next to a nitro group or a zwitterion's second pKa are left
    as they came. `ligand_method: none` plus a protomer you chose is the answer where it matters.
  * It never changes the HEAVY-ATOM graph or the stereochemistry. A result whose element graph or
    number of specified stereo elements differs from the input is REFUSED, not used: a different
    molecule parameterised under the same name is the one error this step must not add.
  * It does not choose a tautomer.

THE MODES, deliberately named like the protein ones:

  openbabel   the default. pH is `protonation.ph` (7.0).
  none        the molecule is parameterised exactly as its input encodes it. Recorded as a choice.

Open Babel missing, or failing, is an ERROR and never a silent fall back to `none`: the build would
parameterise a different species than the record says was asked for.

A 3D input (SDF, mol2) keeps its heavy-atom coordinates: when the pH model changes nothing the
molecule is returned untouched, hydrogens and all; when it changes something, only the hydrogens
that were removed or added differ, and those Open Babel places. Nothing is minimised or re-embedded.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["DEFAULT_PH", "DEFAULT_LIGAND_PROTONATION", "LIGAND_PROTONATION_METHODS",
           "LigandProtonationError", "ligand_protonation_settings", "openbabel_version",
           "protonate_molecule", "protonate_smiles"]

#: THE defaults, in one place. `build-top`'s schema, the builders' default table and the functions
#: below all read these; a literal repeated in any of them is a second value that drifts. The pH is
#: the build's one `protonation.ph`, which the protein half has always used too.
DEFAULT_LIGAND_PROTONATION = "openbabel"
DEFAULT_PH = 7.0

LIGAND_PROTONATION_METHODS = ("openbabel", "none")

class LigandProtonationError(ValueError):
    """A ligand protonation request that cannot be carried out as stated. Refused before output."""


def openbabel_version() -> str:
    """The Open Babel library version, or a refusal that says what to do about its absence."""
    try:
        from openbabel import openbabel
    except ImportError:
        raise LigandProtonationError(
            "protonation.ligand_method is openbabel, but Open Babel is not installed in this "
            "environment. Install it (`conda install -c conda-forge openbabel`, listed in "
            "environment.yml) or set protonation.ligand_method: none deliberately, so that the "
            "molecule is parameterised exactly as its input encodes it. Nothing was built."
        ) from None
    return str(openbabel.OBReleaseVersion())


def protonation_summary(record: Mapping[str, Any]) -> str:
    """One line for a log: what was asked, and what the pH model did to the molecule."""
    if record["method"] == "none":
        return "none -- the molecule is parameterised exactly as its input encodes it"
    head = f"Open Babel {record.get('openbabel_version')} at pH {record['ph']}"
    if not record["changed"]:
        return f"{head}: no change"
    charge = record["net_formal_charge"]
    where = ""
    if "smiles_out" in record:
        where = f"; {record['smiles_in']} -> {record['smiles_out']}"
    elif record.get("changes"):
        where = f"; {len(record['changes'])} heavy atom(s) changed"
    return f"{head}: net charge {charge[0]:+d} -> {charge[1]:+d}{where}"


def ligand_protonation_settings(cfg: Mapping[str, Any] | None) -> tuple[str, float]:
    """`(method, pH)` from a builder configuration, with the module defaults for what it lacks.

    A direct caller that hands the builders a configuration with no `protonation` section gets the
    default -- Open Babel at pH 7.0 -- and not "nothing": the default IS the behaviour.
    """
    section = (cfg or {}).get("protonation") or {}
    method = section.get("ligand_method") or DEFAULT_LIGAND_PROTONATION
    ph = section.get("ph")
    ph = DEFAULT_PH if ph is None else float(ph)
    if method not in LIGAND_PROTONATION_METHODS:
        raise LigandProtonationError(
            f"protonation.ligand_method must be one of {LIGAND_PROTONATION_METHODS}, got "
            f"{method!r}")
    return str(method), ph


def _disabled(ph: float, route: str) -> dict[str, Any]:
    return {"method": "none", "ph": ph, "input_route": route, "changed": False,
            "note": "protonation.ligand_method is none: the molecule is parameterised exactly as "
                    "its input encodes it"}


def _skeleton(mol) -> str:
    """The heavy-atom ELEMENT GRAPH as a canonical string: elements and connectivity, nothing else.

    Charges, hydrogens, bond orders, aromaticity and stereo are all removed, because those are
    exactly what a protomer may differ in. (`rdMolHash`'s ElementGraph was tried first and is not
    this: it keeps stereo, so a lost stereocentre read as a changed graph.)
    """
    from rdkit import Chem

    plain = Chem.RWMol(Chem.RemoveHs(Chem.Mol(mol), sanitize=False))
    for atom in plain.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumRadicalElectrons(0)
    for bond in plain.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(plain, canonical=True, isomericSmiles=False)


def _stereo_elements(mol) -> int:
    """How many stereo elements are SPECIFIED: assigned chiral centres plus defined double bonds."""
    from rdkit import Chem

    plain = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    Chem.AssignStereochemistry(plain, cleanIt=True, force=True)
    centres = len(Chem.FindMolChiralCenters(plain, includeUnassigned=False,
                                            useLegacyImplementation=False))
    bonds = sum(1 for b in plain.GetBonds()
                if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY))
    return centres + bonds


def _changes(before, after) -> list[dict[str, Any]]:
    """Per heavy atom (in input order): formal charge and hydrogen count, where either moved."""
    def table(mol):
        hydrogens: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                for neighbour in atom.GetNeighbors():
                    hydrogens[neighbour.GetIdx()] = hydrogens.get(neighbour.GetIdx(), 0) + 1
        return [(a.GetIdx(), a.GetSymbol(), a.GetFormalCharge(), hydrogens.get(a.GetIdx(), 0))
                for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    out = []
    for position, (old, new) in enumerate(zip(table(before), table(after))):
        if old[2] != new[2] or old[3] != new[3]:
            out.append({"heavy_atom": position, "element": old[1],
                        "formal_charge": [old[2], new[2]], "n_hydrogens": [old[3], new[3]]})
    return out


def _require_same_molecule(before, after, *, what: str) -> None:
    """Refuse a result that is a different MOLECULE, as opposed to a different protomer of it."""
    skeleton_before, stereo_before = _skeleton(before), _stereo_elements(before)
    skeleton_after, stereo_after = _skeleton(after), _stereo_elements(after)
    if skeleton_before != skeleton_after:
        raise LigandProtonationError(
            f"Open Babel's pH model changed the heavy-atom graph of {what}. Protonation may move "
            f"hydrogens and formal charges and nothing else, so this result is not used. Supply "
            f"the protomer you mean and set protonation.ligand_method: none.")
    if stereo_before != stereo_after:
        raise LigandProtonationError(
            f"Open Babel's pH model changed the number of specified stereo elements of {what} "
            f"({stereo_before} -> {stereo_after}). Stereochemistry is not Open Babel's to alter, "
            f"so this result is not used. Supply the protomer you mean and set "
            f"protonation.ligand_method: none.")


def _record(method: str, ph: float, route: str, before, after, *,
            changes: list[dict[str, Any]] | None, changed: bool) -> dict[str, Any]:
    """*changes* is per heavy atom where the two share an atom order (a 3D molecule) and None where
    they do not (a SMILES, which Open Babel re-writes in its own order)."""
    record = {"method": method, "ph": ph, "openbabel_version": openbabel_version(),
              "call": "OBMol.AddHydrogens(polaronly=False, correctForPH=True, pH) -- obabel -p",
              "input_route": route, "changed": bool(changed),
              "net_formal_charge": [int(sum(a.GetFormalCharge() for a in before.GetAtoms())),
                                    int(sum(a.GetFormalCharge() for a in after.GetAtoms()))],
              "not_a_pka_predictor": "a table of functional-group rules: phosphates, imidazoles "
                                     "and any environment-dependent pKa are left as they came"}
    if changes is not None:
        record["changes"] = changes
    return record


def protonate_smiles(smiles: str, ph: float, *, method: str = DEFAULT_LIGAND_PROTONATION
                     ) -> tuple[str, dict[str, Any]]:
    """The SMILES of the species Open Babel's pH model gives at *ph*, and the record of the step.

    Returns *smiles* itself, untouched, for `method: none` and whenever the model changes nothing,
    so a molecule that needed no change keeps the exact string it was written with.
    """
    from rdkit import Chem

    if method == "none":
        return smiles, _disabled(ph, "smiles")
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise LigandProtonationError(f"RDKit could not parse SMILES: {smiles!r}")
    openbabel_version()
    from openbabel import openbabel

    openbabel.obErrorLog.SetOutputLevel(0)
    conversion = openbabel.OBConversion()
    if not conversion.SetInAndOutFormats("smi", "smi"):
        raise LigandProtonationError("Open Babel has no SMILES format in this installation")
    molecule = openbabel.OBMol()
    if not conversion.ReadString(molecule, smiles):
        raise LigandProtonationError(f"Open Babel could not read SMILES: {smiles!r}")
    molecule.AddHydrogens(False, True, float(ph))
    produced = conversion.WriteString(molecule).split()
    if not produced:
        raise LigandProtonationError(f"Open Babel wrote nothing for SMILES {smiles!r}")
    result = Chem.MolFromSmiles(produced[0])
    if result is None:
        raise LigandProtonationError(
            f"RDKit could not parse the SMILES Open Babel wrote for {smiles!r}: {produced[0]!r}")
    _require_same_molecule(parsed, result, what=f"SMILES {smiles!r}")
    changed = Chem.MolToSmiles(parsed) != Chem.MolToSmiles(result)
    record = _record(method, ph, "smiles", parsed, result, changes=None, changed=changed)
    record["smiles_in"] = smiles
    if not changed:
        record["smiles_out"] = smiles
        return smiles, record
    record["smiles_out"] = produced[0]
    return produced[0], record


def protonate_molecule(mol, ph: float, *, route: str, method: str = DEFAULT_LIGAND_PROTONATION):
    """`(molecule, record)` for a 3D molecule with explicit hydrogens (an SDF or a mol2).

    The molecule that comes back is the INPUT object when nothing changed, so a docked pose keeps
    every hydrogen it arrived with. When something changed it is Open Babel's molecule read back
    through RDKit, and its heavy atoms are checked to sit where they were: same elements in the
    same order, no coordinate more than 1e-3 A away.
    """
    import numpy as np
    from rdkit import Chem

    if method == "none":
        return mol, _disabled(ph, route)
    openbabel_version()
    from openbabel import openbabel

    openbabel.obErrorLog.SetOutputLevel(0)
    conversion = openbabel.OBConversion()
    conversion.SetInAndOutFormats("mol", "mol")
    molecule = openbabel.OBMol()
    if not conversion.ReadString(molecule, Chem.MolToMolBlock(mol)):
        raise LigandProtonationError("Open Babel could not read the molecule it was handed")
    molecule.AddHydrogens(False, True, float(ph))
    result = Chem.MolFromMolBlock(conversion.WriteString(molecule), removeHs=False, sanitize=True)
    if result is None or result.GetNumConformers() == 0:
        raise LigandProtonationError(
            "RDKit could not read back the molecule Open Babel wrote after its pH model")
    _require_same_molecule(mol, result, what="the supplied molecule")

    heavy_in = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    heavy_out = [a.GetIdx() for a in result.GetAtoms() if a.GetAtomicNum() > 1]
    if [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_in] != \
            [result.GetAtomWithIdx(i).GetSymbol() for i in heavy_out]:
        raise LigandProtonationError(
            "Open Babel returned the heavy atoms in a different order, so the pose cannot be "
            "carried across. Nothing was built.")
    moved = float(np.abs(mol.GetConformer().GetPositions()[heavy_in]
                         - result.GetConformer().GetPositions()[heavy_out]).max())
    if moved > 1e-3:
        raise LigandProtonationError(
            f"Open Babel moved a heavy atom by {moved:.4f} A; protonation may only add and "
            f"remove hydrogens. Nothing was built.")
    changes = _changes(mol, result)
    record = _record(method, ph, route, mol, result, changes=changes, changed=bool(changes))
    if not changes:
        return mol, record
    record["heavy_atom_max_displacement_angstrom"] = round(moved, 6)
    return result, record
