"""System-construction configuration: the one YAML file `md-openmm build-top` resolves.

This module resolves the SYSTEM half of a build -- force fields, solvent model, box, ions,
constraints -- and nothing else. MD workflow configuration (`protocol:`, stage lengths in integer
steps, reporting intervals) has one authority, `md_tools.build.md`, and it is not here.

That was not always true. This file used to carry a second, retired MD model built around
`methods:` with `duration_ns` and `switching_duration_ps`, alongside the system half. Two callable
resolvers both looked authoritative, which is exactly the ambiguity that makes a reader -- or an
agent -- pick the wrong one. The retired half is gone; see `docs/release-notes/v0.5.0.md`.

Deliberately thin. There is no schema-migration layer, no profile inheritance and no canonical
intermediate model: the YAML is the configuration, and resolving it means dropping the solvent
block that does not apply and checking the handful of combinations that are physically wrong.
"""
from __future__ import annotations

from typing import Any

import yaml

from ..build.strict import ConfigError
from .system_defaults import canonical_solvent, is_implicit, is_vacuum

__all__ = ["resolve_sys_config", "openff_resource", "pairing_warnings", "ConfigError"]

def resolve_sys_config(document: dict[str, Any]) -> dict[str, Any]:
    """The effective system settings: the irrelevant solvent block removed, and checks applied."""
    resolved = yaml.safe_load(yaml.safe_dump(document))       # deep copy via the same serialiser
    solvent_block = resolved.get("solvent") or {}
    implicit_block = resolved.get("implicit_solvent") or {}

    model = solvent_block.get("model")
    implicit_model = implicit_block.get("model")
    # Which treatment applies is decided by which block the user left in place. Both present is the
    # default file's shape, in which case `solvent.model` decides.
    if model and implicit_model:
        chosen = canonical_solvent(model)
    elif implicit_model:
        chosen = canonical_solvent(implicit_model)
    elif model:
        chosen = canonical_solvent(model)
    else:
        raise ConfigError("neither solvent.model nor implicit_solvent.model is set")

    resolved["solvation"] = ("implicit" if is_implicit(chosen)
                             else "vacuum" if is_vacuum(chosen) else "explicit")
    if resolved["solvation"] == "vacuum":
        # No water parameterises anything and no box exists; the solvent block keeps its model.
        resolved.pop("implicit_solvent", None)
        resolved["solvent"] = {"model": "vacuum"}
        resolved["forcefield"] = dict(resolved.get("forcefield") or {})
        resolved["forcefield"]["water"] = None
        resolved.setdefault("constraints", {})["rigid_water"] = False
    elif resolved["solvation"] == "implicit":
        resolved.pop("solvent", None)
        resolved["forcefield"] = dict(resolved.get("forcefield") or {})
        # No water model participates in an implicit build, and recording one would name a force
        # field that never loaded.
        resolved["forcefield"]["water"] = None
        constraints = resolved.setdefault("constraints", {})
        constraints["rigid_water"] = False
    else:
        resolved.pop("implicit_solvent", None)

    # The ligand force field is recorded as the resource that will actually be loaded, decided
    # here rather than left as a label for a reader to map later. `sage-2.2.1` is what a user
    # writes; `openff-2.2.1` is what the toolkit resolves, and preflight compares the record
    # against this without needing a second copy of the mapping.
    solute = resolved.get("solute") or {}
    if not bool(solute.get("peptide", True)):
        resolved["forcefield"] = dict(resolved.get("forcefield") or {})
        resolved["forcefield"]["ligand"] = openff_resource(solute.get("ligand_forcefield"))
        resolved["forcefield"]["ligand_charge_method"] = solute.get("ligand_charge_method")
        # No protein force field participates in a ligand build -- the ligand route loads only
        # the water XML and the SMIRNOFF template generator -- and recording one would name a
        # force field that never loaded. Same rule as water under implicit solvent.
        resolved["forcefield"]["protein"] = None

    _check_constraints(resolved)
    _check_protein_solvation_pairing(resolved)
    return resolved


def openff_resource(name):
    """The exact small-molecule resource a written label resolves to.

    `sage-2.2.1` is what a user writes and `openff-2.2.1` is what the toolkit loads; `gaff2` is an
    alias for the newest installed GAFF 2.x. The resolution -- and the refusal of a GAFF version
    that is not installed -- lives in `ligand_forcefield`, which owns it for every caller.

    Kept under this name because `builders.py` and the resolved system document already use it.
    """
    from .ligand_forcefield import resolve_ligand_forcefield

    return resolve_ligand_forcefield(name)


def pairing_warnings(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured warnings for an explicit protein/water pairing outside the two supported ones.

    **A warning, not a refusal.** ff14SB + OPC and ff19SB + TIP3P are combinations a competent user
    may deliberately want -- to reproduce someone else's published setup, or to measure exactly the
    water-model sensitivity this pairing exposes. Refusing them made MD-tools the arbiter of
    somebody else's experiment. What the tool owes the user is that the choice is never made
    silently or by accident.

    So the two supported pairs are quiet, and a crossed pair is loud: named on stderr and recorded
    structurally in `built.log` and its machine record, so a reader of the data a year later can
    see the combination was chosen rather than inferred.

    The two pairs are not interchangeable halves. ff14SB's backbone adjustment is an empirical
    correction fit in TIP3P, and its authors caution that transferring it to another water model
    needs evaluation; ff19SB's amino-acid-specific CMAPs were trained for a better water model and
    its authors recommend OPC. Crossing them discards the reason either pair works. See
    `docs/scientific-defaults.md` section 3.

    This is about the PROTEIN/WATER pairing only. A ligand force field -- Sage or GAFF -- is
    orthogonal to it and never triggers a warning here.
    """
    from .system_defaults import EXPLICIT_COMBINATIONS, PROTEIN_FAMILY_MARKERS, WATER_FAMILY_MARKERS

    if resolved.get("solvation") != "explicit":
        return []
    forcefield = resolved.get("forcefield") or {}
    solvent = resolved.get("solvent") or {}
    model = canonical_solvent(solvent.get("model"))
    expected = EXPLICIT_COMBINATIONS[model]
    protein = str(forcefield.get("protein") or "").strip()
    water = str(forcefield.get("water") or "").strip()

    def family(value: str, markers: dict[str, tuple[str, ...]]) -> str:
        """Which supported family a resource name belongs to, or "" if it is neither."""
        lowered = value.lower()
        for name, tokens in markers.items():
            if any(token in lowered for token in tokens):
                return name
        return ""

    crossed = []
    if protein and family(protein, PROTEIN_FAMILY_MARKERS) not in ("", model):
        crossed.append(("forcefield.protein", protein, expected["protein"]))
    if water and family(water, WATER_FAMILY_MARKERS) not in ("", model):
        crossed.append(("forcefield.water", water, expected["water"]))
    if not crossed:
        return []

    named = ", ".join(f"{field}={value!r}" for field, value, _ in crossed)
    other = ", ".join(f"{name} = {values['protein']} + {values['water']}"
                      for name, values in EXPLICIT_COMBINATIONS.items())
    return [{
        "code": "crossed_explicit_pair",
        "severity": "warning",
        "message": (
            f"CROSSED PROTEIN/WATER PAIR: solvent.model={model!r} with {named}. "
            f"The protein force field and the water model are one coupled selection. The pairs "
            f"that were developed and validated together are: {other}. This combination will "
            f"build and run; it is not one anyone has validated, and the backbone correction is "
            f"being used with a water model it was not fit for. See "
            f"docs/scientific-defaults.md section 3."),
        "solvent_model": model,
        "fields": [{"field": field, "value": value, "supported_pair_expects": want}
                   for field, value, want in crossed],
        "supported_pairs": {name: dict(values) for name, values in EXPLICIT_COMBINATIONS.items()},
    }]


def _check_protein_solvation_pairing(resolved: dict[str, Any]) -> None:
    """Refuse a protein force field that was not parameterised for this solvation model.

    ff19SB's amino-acid-specific CMAPs were fit in explicit OPC water, and no GB model has been
    reparameterised against them; GBn2 was developed and validated in the ff99SB/ff14SB lineage.
    Running the pair produces numbers, which is exactly the problem -- nothing fails, and the
    result silently describes a Hamiltonian nobody validated.

    This is refused rather than warned about because a warning in a log is not read by whoever
    reads the trajectory a year later.
    """
    from .system_defaults import GB_INCOMPATIBLE_PROTEIN, IMPLICIT_PROTEIN_FORCEFIELD

    if resolved.get("solvation") != "implicit":
        return
    protein = str((resolved.get("forcefield") or {}).get("protein") or "")
    model = (resolved.get("implicit_solvent") or {}).get("model")
    if not protein:
        return
    if any(marker.lower() in protein.lower() for marker in GB_INCOMPATIBLE_PROTEIN):
        raise ConfigError(
            f"forcefield.protein = {protein!r} is not parameterised for implicit_solvent.model = "
            f"{model!r}.\n"
            f"  ff19SB's amino-acid-specific CMAP corrections were fit in explicit OPC water, and "
            f"no GB model has been reparameterised against them. GBn2 was developed and validated "
            f"with the ff99SB/ff14SB lineage, so this pair mixes a backbone trained in explicit "
            f"solvent with a solvation model tuned for a different one.\n"
            f"  Use the matched pair:\n"
            f"      forcefield.protein: {IMPLICIT_PROTEIN_FORCEFIELD}\n"
            f"  or switch to explicit solvent, where ff19SB belongs.")


def _check_constraints(resolved: dict[str, Any]) -> None:
    constraints = resolved.get("constraints") or {}
    mass = constraints.get("hydrogen_mass_amu")
    if mass is not None and float(mass) < 1.008:
        raise ConfigError(
            f"constraints.hydrogen_mass_amu is {mass}, lighter than a hydrogen. Repartitioning "
            "moves mass INTO hydrogens from the heavy atoms they are bonded to.")
    # The strings `build-top`'s schema actually offers, plus an absent key. `"None"` is one of
    # them and used to be refused, because this compared against Python `None` and never the
    # string a YAML file can hold. `HAngles` used to be accepted here and is not in the enum:
    # scientific-defaults.md states that no angle is ever constrained by this option, so the enum
    # was the intended policy and this was the disagreement.
    if constraints.get("type") not in ("HBonds", "AllBonds", "None", None):
        raise ConfigError(f"constraints.type {constraints.get('type')!r} is not supported")
