"""Which water model a solute's force fields were actually fitted with.

A protein force field and a small-molecule force field are parameterised against water, and not
against the same water. Choosing one default for "explicit solvent" therefore mispairs one of them
by construction. The default here is derived from the **resolved force-field family**, not from a
broad input label like `peptide` or `ligand`: the label says which reader parsed the input, while
the family says which parameters will actually be assigned, and it is the parameters that were fit.

## The pairings, and where they come from

* **ff19SB -> OPC.** ff19SB replaced generic backbone dihedrals with amino-acid-specific CMAP terms
  trained against QM energy surfaces computed *in solution*, and the paper recommends OPC; with
  TIP3P the amino-acid-dependent helical propensities the CMAPs exist to reproduce come out wrong.
  Tian, Kasavajhala, Belfon, Raguette, Huang, Migues, Bickel, Wang, Pincay, Wu and Simmerling,
  "ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy
  Surfaces in Solution", J. Chem. Theory Comput. 2020, 16, 528-552, doi:10.1021/acs.jctc.9b00591.

* **ff14SB -> TIP3P.** The predecessor was developed and validated in TIP3P. Maier, Martinez,
  Kasavajhala, Wickstrom, Hauser and Simmerling, "ff14SB: Improving the Accuracy of Protein Side
  Chain and Backbone Parameters from ff99SB", J. Chem. Theory Comput. 2015, 11, 3696-3713,
  doi:10.1021/acs.jctc.5b00255.

* **OpenFF Sage (openff-2.x) -> TIP3P.** Sage refit the Lennard-Jones parameters against
  condensed-phase properties, and that training used TIP3P; the OpenFF force fields also ship TIP3P
  water parameters themselves, so TIP3P is the model the release is built around. Boothroyd, Behara,
  Madin, Hahn, Jang, Gapsys, Wagner, Horton, Dotson, Thompson, Maat, Gokey, Wang, Cole, Gilson,
  Chodera, Bayly, Shirts and Mobley, "Development and Benchmarking of Open Force Field 2.0.0 -- the
  Sage Small Molecule Force Field", J. Chem. Theory Comput. 2023, 19, 3251-3275,
  doi:10.1021/acs.jctc.3c00039. See also the openff-forcefields release notes, which distribute
  `tip3p.offxml` alongside each Sage release.
  Parsley (openff-1.x) is treated the same way; it was likewise developed with TIP3P.

## The mixed case

A protein-ligand complex has one box and therefore one water model, so one of the two force fields
must run away from its fitting partner. It resolves to **OPC**, ff19SB's partner, because the
protein backbone is normally the dominant error term and Sage's Lennard-Jones refit is the less
water-sensitive of the two. This is a compatibility choice, not a validated pairing, and it is
recorded as one: the resolved provenance carries the rationale so a report built on such a system
can state it rather than imply that both force fields are being used as published.

## Ions

Ion parameters are water-model specific -- the Joung-Cheatham sets are fitted per water model -- and
OpenMM ships them **inside each water force-field file**. Loading `amber19/opc.xml` therefore brings
OPC-matched ions and loading `amber19/tip3p.xml` brings TIP3P-matched ones, with no separate choice
to make and no way for them to drift apart. Measured through `ForceField.createSystem`, Na+ epsilon
differs by roughly a factor of three between the two, so this is not a formality. The source file is
recorded in provenance so the pairing is stated rather than assumed.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "AmbiguousWaterPolicy",
    "OPC",
    "TIP3P",
    "forcefield_families",
    "resolve_default_water",
]

#: `(force-field resource, modeller packing model)`.
OPC = ("amber19/opc.xml", "opc")
TIP3P = ("amber19/tip3p.xml", "tip3p")

#: Substring in the resolved protein force-field resource -> (family, its fitting water).
#: Matched against the resolved resource name rather than a label, so `amber19/protein.ff19SB.xml`
#: and `leaprc.protein.ff19SB` -- the OpenMM and tleap spellings of the same choice -- agree.
_PROTEIN_FAMILIES = (
    ("ff19sb", ("ff19SB", OPC)),
    ("ff14sb", ("ff14SB", TIP3P)),
)

#: Prefix of the resolved small-molecule force field -> (family, its fitting water).
_LIGAND_FAMILIES = (
    ("openff-2", ("OpenFF Sage (openff-2.x)", TIP3P)),
    ("openff-1", ("OpenFF Parsley (openff-1.x)", TIP3P)),
)


class AmbiguousWaterPolicy(ValueError):
    """The default water cannot be derived from the force fields that were actually resolved.

    Raised rather than guessed. Picking a water model for an unrecognised force field would pair
    parameters with a solvent nobody checked, and the resulting run would look entirely normal.
    """


def _match(value: Optional[str], table) -> Optional[tuple]:
    if not value:
        return None
    text = str(value).lower()
    for needle, result in table:
        if needle in text:
            return result
    return None


def forcefield_families(forcefield: dict, *, solute_kind: Optional[str] = None) -> dict:
    """Which recognised families the resolved force-field block names, and their fitting water.

    `solute_kind` narrows which fields are consulted, because a peptide-route bundle may carry a
    package-default `ligand` entry it never uses. It is a filter on the fields, never the source of
    the answer.
    """
    protein_field = forcefield.get("protein")
    ligand_field = forcefield.get("ligand")
    if solute_kind == "peptide":
        ligand_field = None
    elif solute_kind == "ligand":
        protein_field = None

    protein = _match(protein_field, _PROTEIN_FAMILIES)
    ligand = _match(ligand_field, _LIGAND_FAMILIES)
    return {
        "protein_forcefield": protein_field,
        "ligand_forcefield": ligand_field,
        "protein_family": protein[0] if protein else None,
        "ligand_family": ligand[0] if ligand else None,
        "protein_water": protein[1] if protein else None,
        "ligand_water": ligand[1] if ligand else None,
        "unrecognised": [
            name for name, value, hit in
            (("forcefield.protein", protein_field, protein),
             ("forcefield.ligand", ligand_field, ligand))
            if value and not hit
        ],
    }


def resolve_default_water(forcefield: dict, *, solute_kind: Optional[str] = None) -> dict:
    """The default water for these force fields, with the reasoning that produced it.

    Returns `water`, `water_model`, `basis` (which family decided it), `mixed` and a `rationale`
    sentence, all of which belong in the resolved provenance. Raises `AmbiguousWaterPolicy` when the
    families present do not determine an answer.
    """
    families = forcefield_families(forcefield, solute_kind=solute_kind)
    protein_water = families["protein_water"]
    ligand_water = families["ligand_water"]

    if protein_water and ligand_water:
        water, model = OPC
        mixed = protein_water != ligand_water
        return {
            "water": water, "water_model": model,
            "basis": f"{families['protein_family']} + {families['ligand_family']}",
            "mixed": mixed,
            "rationale": (
                f"protein-ligand complex: {families['protein_family']} was fitted with "
                f"{protein_water[1].upper()} and {families['ligand_family']} with "
                f"{ligand_water[1].upper()}. One box carries one water model, so this defaults to "
                f"{model.upper()}, the protein force field's partner, because the protein backbone "
                "is normally the dominant error term. This is a mixed-force-field compatibility "
                "choice and not a validated pairing for the small molecule."
                if mixed else
                f"protein-ligand complex: both {families['protein_family']} and "
                f"{families['ligand_family']} were fitted with {model.upper()}."
            ),
        }

    for role, fitted in (("protein", protein_water), ("ligand", ligand_water)):
        if fitted:
            water, model = fitted
            family = families[f"{role}_family"]
            return {
                "water": water, "water_model": model,
                "basis": family, "mixed": False,
                "rationale": (f"{family} was parameterised with {model.upper()}, so it is the "
                              "default here."),
            }

    stated = ", ".join(
        f"{k}={v!r}" for k, v in (("forcefield.protein", families["protein_forcefield"]),
                                  ("forcefield.ligand", families["ligand_forcefield"])) if v
    ) or "no solute force field is named"
    raise AmbiguousWaterPolicy(
        "cannot choose a default water model from the resolved force fields: "
        f"{stated}. Recognised pairings are "
        f"{', '.join(f for f, _ in (v for _, v in _PROTEIN_FAMILIES))} and "
        f"{', '.join(f for f, _ in (v for _, v in _LIGAND_FAMILIES))}. "
        "A water model chosen for an unrecognised force field would pair parameters with a solvent "
        "nobody validated them against, and the run would look entirely normal. Set "
        "`forcefield.water` and `solvation.water_model` explicitly to state the pairing you intend; "
        "the choice is recorded as user input in the bundle provenance."
    )
