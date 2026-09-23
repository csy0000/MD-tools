"""Which sidechain bonds a selective REST2 region may scale, per amino acid, written out.

The amino acids are a small fixed set, so which of a sidechain's bonds are rotatable is a LOOKUP,
not a perception problem (the user, 2026-09-20). This table says, for every residue the package
supports, every central bond inside the sidechain -- including CA-CB, which the sidechain owns
because it carries chi1 -- and for each one either the chi torsion it turns or the reason it is
never scaled.

`SIDECHAIN_BONDS[name]["bonds"]` is a tuple of entries:

    {"bond": ("CA", "CB"), "kind": "chi", "label": "chi1", "torsion": ("N","CA","CB","CG")}
    {"bond": ("CG", "ND2"), "kind": "fixed", "label": "amide C-N", "class": "amide_omega"}

`kind` is one of:

  `chi`       a named side-chain dihedral, scaled;
  `rotation`  a terminal group's own rotation -- a methyl, a hydroxyl, a thiol, an ammonium --
              scaled, and listed because it IS a central bond and a table that omitted it would
              be incomplete rather than silent;
  `fixed`     never scaled, with the `class` the torsion classifier gives it.

THE TABLE AND THE CLASSIFIER MUST AGREE, and that is enforced twice. A test compares them residue
by residue against the force field's own residue templates, and `md_tools.rest2.regions` checks
every selected sidechain against this table while it resolves a region, so a disagreement refuses
a build rather than producing a Hamiltonian nobody predicted. The classes are the classifier's
(`openmm.system.classify_unscaled_torsions`): `amide_omega`, `aromatic_ring`, `double_bond`.

WHAT IS NOT HERE, deliberately:

* A carboxylate (ASP CG-OD1/OD2, GLU CD-OE1/OE2) is NOT a central bond: both oxygens are terminal,
  so no proper torsion runs across it and there is nothing to scale or protect. It is recorded per
  residue under `not_central` so the omission is a statement rather than a gap.
* The backbone. N-CA and CA-C belong to the backbone (`regions.central_bond_owner`), and the
  peptide bond to the residue that owns omega.
* A disulfide. CYX's SG-SG is an INTER-RESIDUE bond owned jointly by both sidechains; CYX's own
  CB-SG becomes a central bond only once that bond exists, which is why CYX lists it as a chi with
  `needs_partner`.
* Anything not in `PROTEIN_RESIDUES`. A modified or unsupported residue is refused by name, as it
  already is.
"""
from __future__ import annotations

from typing import Any

__all__ = ["SIDECHAIN_BONDS", "SIDECHAIN_TABLE_VERSION", "SidechainTableError",
           "sidechain_entry", "scalable_bonds", "fixed_bonds", "describe_residue"]

#: Bumped when a residue's rotatability changes. It is part of what a selection means.
SIDECHAIN_TABLE_VERSION = "md-tools-sidechain-rotatability/1"


class SidechainTableError(KeyError):
    """A residue this table does not define."""


def _chi(number, torsion, *, needs_partner=False):
    entry = {"bond": tuple(sorted(torsion[1:3])), "kind": "chi", "label": f"chi{number}",
             "torsion": tuple(torsion)}
    if needs_partner:
        entry["needs_partner"] = True
    return entry


def _rotation(bond, label):
    return {"bond": tuple(sorted(bond)), "kind": "rotation", "label": label}


def _fixed(bond, label, kind):
    return {"bond": tuple(sorted(bond)), "kind": "fixed", "label": label, "class": kind}


_BENZENE_RING = (("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"),
                 ("CD2", "CG"))
_IMIDAZOLE_RING = (("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG"))
_INDOLE_RING = (("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CG"),
                ("CD2", "CE3"), ("CE3", "CZ3"), ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2"))


def _ring(pairs, label):
    return [_fixed(pair, label, "aromatic_ring") for pair in pairs]


#: name -> {bonds, not_central, note}. Every central bond of the sidechain, and nothing else.
SIDECHAIN_BONDS: dict[str, dict[str, Any]] = {
    "ALA": {"bonds": (_rotation(("CA", "CB"), "methyl rotation (chi1 has no heavy fourth atom)"),),
            "note": "one methyl; no chi"},
    "ARG": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD")),
                      _chi(3, ("CB", "CG", "CD", "NE")), _chi(4, ("CG", "CD", "NE", "CZ")),
                      _fixed(("NE", "CZ"), "guanidinium", "double_bond"),
                      _fixed(("CZ", "NH1"), "guanidinium", "double_bond"),
                      _fixed(("CZ", "NH2"), "guanidinium", "double_bond")),
            "note": "the guanidinium's three partial double bonds are unscaled (user, 2026-09-16)"},
    "ASN": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "OD1")),
                      _fixed(("CG", "ND2"), "amide C-N", "amide_omega")),
            "note": "the sidechain amide isomerises like a peptide bond, so it is unscaled"},
    "ASP": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "OD1"))),
            "not_central": (("CG", "OD1"), ("CG", "OD2")),
            "note": "the carboxylate oxygens are terminal: no torsion runs across them"},
    "CYS": {"bonds": (_chi(1, ("N", "CA", "CB", "SG")),
                      _rotation(("CB", "SG"), "thiol rotation")),
            "note": "free cysteine; a disulfide is CYX"},
    "CYX": {"bonds": (_chi(1, ("N", "CA", "CB", "SG")),
                      {"bond": ("CB", "SG"), "kind": "chi", "label": "chi2 (across the "
                                                                     "disulfide)",
                       "torsion": ("CA", "CB", "SG", "SG'"), "needs_partner": True}),
            "note": "chi1 (CA-CB) is always there. CB-SG carries a torsion only once the SG-SG "
                    "bond exists, which is why chi2 needs a partner; SG-SG itself is "
                    "inter-residue and owned by BOTH sidechains (regions.central_bond_owner)"},
    "GLN": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD")),
                      _chi(3, ("CB", "CG", "CD", "OE1")),
                      _fixed(("CD", "NE2"), "amide C-N", "amide_omega")),
            "note": "as ASN, one carbon further out"},
    "GLU": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD")),
                      _chi(3, ("CB", "CG", "CD", "OE1"))),
            "not_central": (("CD", "OE1"), ("CD", "OE2")),
            "note": "as ASP, one carbon further out"},
    "GLY": {"bonds": (), "note": "no sidechain at all"},
    "HID": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "ND1")),
                      *_ring(_IMIDAZOLE_RING, "imidazole ring")),
            "note": "delta-protonated"},
    "HIE": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "ND1")),
                      *_ring(_IMIDAZOLE_RING, "imidazole ring")),
            "note": "epsilon-protonated"},
    "HIP": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "ND1")),
                      *_ring(_IMIDAZOLE_RING, "imidazole ring")),
            "note": "doubly protonated, +1"},
    "HIS": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "ND1")),
                      *_ring(_IMIDAZOLE_RING, "imidazole ring")),
            "note": "the ambiguous name. A built system carries HID/HIE/HIP; this entry exists so "
                    "a structure that still says HIS resolves to the same ring"},
    "ILE": {"bonds": (_chi(1, ("N", "CA", "CB", "CG1")), _chi(2, ("CA", "CB", "CG1", "CD1")),
                      _rotation(("CB", "CG2"), "methyl rotation"),
                      _rotation(("CG1", "CD1"), "methyl rotation")),
            "note": "beta-branched"},
    "LEU": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD1")),
                      _rotation(("CG", "CD1"), "methyl rotation"),
                      _rotation(("CG", "CD2"), "methyl rotation")),
            "note": ""},
    "LYS": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD")),
                      _chi(3, ("CB", "CG", "CD", "CE")), _chi(4, ("CG", "CD", "CE", "NZ")),
                      _rotation(("CE", "NZ"), "ammonium rotation")),
            "note": ""},
    "MET": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "SD")),
                      _chi(3, ("CB", "CG", "SD", "CE")),
                      _rotation(("SD", "CE"), "methyl rotation")),
            "note": ""},
    "PHE": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD1")),
                      *_ring(_BENZENE_RING, "benzene ring")),
            "note": ""},
    "PRO": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD")),
                      _chi(3, ("CB", "CG", "CD", "N")), _chi(4, ("CG", "CD", "N", "CA"))),
            "note": "the ring closes through the backbone N, so chi4's CD-N is the sidechain's "
                    "(regions gives a bond with any sidechain atom to the sidechain); proline's "
                    "omega is the proline-like exception and stays scalable"},
    "SER": {"bonds": (_chi(1, ("N", "CA", "CB", "OG")),
                      _rotation(("CB", "OG"), "hydroxyl rotation")),
            "note": ""},
    "THR": {"bonds": (_chi(1, ("N", "CA", "CB", "OG1")),
                      _rotation(("CB", "OG1"), "hydroxyl rotation"),
                      _rotation(("CB", "CG2"), "methyl rotation")),
            "note": "beta-branched"},
    "TRP": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD1")),
                      *_ring(_INDOLE_RING, "indole ring")),
            "note": ""},
    "TYR": {"bonds": (_chi(1, ("N", "CA", "CB", "CG")), _chi(2, ("CA", "CB", "CG", "CD1")),
                      *_ring(_BENZENE_RING, "benzene ring"),
                      _rotation(("CZ", "OH"), "hydroxyl rotation")),
            "note": "the ring is fixed; the hydroxyl turns"},
    "VAL": {"bonds": (_chi(1, ("N", "CA", "CB", "CG1")),
                      _rotation(("CB", "CG1"), "methyl rotation"),
                      _rotation(("CB", "CG2"), "methyl rotation")),
            "note": "beta-branched"},
    # The caps have no sidechain. Their own methyl rotation belongs to the cap, which is backbone
    # throughout (`regions.atom_category`), and is listed so the table covers every supported name.
    "ACE": {"bonds": (), "cap": True, "note": "acetyl cap: backbone throughout"},
    "NME": {"bonds": (), "cap": True, "note": "N-methyl cap: backbone throughout"},
    "NHE": {"bonds": (), "cap": True, "note": "amide cap: no heavy sidechain"},
    "NMA": {"bonds": (), "cap": True, "note": "N-methylamide cap, tleap's spelling of NME"},
}


def sidechain_entry(residue_name: str) -> dict[str, Any]:
    """The table entry for a residue, by name. Refuses a residue it does not define."""
    name = str(residue_name).strip().upper()
    try:
        return SIDECHAIN_BONDS[name]
    except KeyError:
        raise SidechainTableError(
            f"{residue_name}: no sidechain rotatability is defined for this residue "
            f"({SIDECHAIN_TABLE_VERSION}). Defined: {sorted(SIDECHAIN_BONDS)}. A modified or "
            f"unsupported residue is refused rather than guessed.") from None


def scalable_bonds(residue_name: str) -> set[tuple[str, str]]:
    """The sidechain central bonds a selective region may scale, by atom name."""
    return {entry["bond"] for entry in sidechain_entry(residue_name)["bonds"]
            if entry["kind"] in ("chi", "rotation")}


def fixed_bonds(residue_name: str) -> dict[tuple[str, str], str]:
    """The sidechain central bonds that are never scaled -> the classifier class of each."""
    return {entry["bond"]: entry["class"] for entry in sidechain_entry(residue_name)["bonds"]
            if entry["kind"] == "fixed"}


def describe_residue(residue_name: str) -> str:
    """One readable block per residue, for a person deciding what to select."""
    entry = sidechain_entry(residue_name)
    lines = [f"{str(residue_name).upper()}  ({SIDECHAIN_TABLE_VERSION})"]
    if entry.get("note"):
        lines.append(f"  {entry['note']}")
    for bond in entry["bonds"]:
        mark = "scaled  " if bond["kind"] in ("chi", "rotation") else "UNSCALED"
        extra = f" [{bond['class']}]" if bond["kind"] == "fixed" else ""
        torsion = f"  {'-'.join(bond['torsion'])}" if bond.get("torsion") else ""
        lines.append(f"  {mark} {bond['bond'][0]:>4}-{bond['bond'][1]:<4} {bond['label']}{extra}"
                     f"{torsion}")
    for bond in entry.get("not_central", ()):
        lines.append(f"  none     {bond[0]:>4}-{bond[1]:<4} terminal: no torsion runs across it")
    return "\n".join(lines)
