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
from .system_defaults import canonical_solvent, is_implicit

__all__ = ["resolve_sys_config", "openff_resource", "ConfigError"]

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

    resolved["solvation"] = "implicit" if is_implicit(chosen) else "explicit"
    if resolved["solvation"] == "implicit":
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
    _check_explicit_pairing(resolved)
    return resolved


def openff_resource(name):
    """`sage-2.2.1` is what a user writes; `openff-2.2.1` is what the toolkit loads.

    The installed `openforcefields` package ships the file as `openff-2.2.1.offxml`, and
    `SMIRNOFFTemplateGenerator` resolves the name with or without the suffix. A name that does not
    resolve raises there, at the point the parameters would have been assigned.
    """
    if not name:
        return None
    text = str(name).strip().lower()
    if text.startswith("sage-"):
        return "openff-" + text[len("sage-"):]
    return text


def _check_explicit_pairing(resolved: dict[str, Any]) -> None:
    """Refuse a hand-edited explicit configuration that crosses the two supported pairs.

    the build configuration writes a coupled selection -- ff14SB with TIP3P, ff19SB with OPC -- but the file it
    writes is ordinary editable YAML, so generating it correctly is not the same as building it
    correctly. Changing `solvent.model` to OPC and leaving `forcefield.protein` at ff14SB produces
    a System, runs to completion, and reports a Hamiltonian nobody validated.

    The two pairs are not interchangeable halves. ff14SB's backbone adjustment is an empirical
    correction fit in TIP3P and its authors caution that transferring it to another solvent model
    needs evaluation; ff19SB's amino-acid-specific CMAPs were trained for a better water model and
    its authors recommend OPC. Crossing them discards the reason either pair works. See
    `docs/scientific-defaults.md` section 3.

    Both halves are checked, in both directions, because either one alone can be the edited field.
    """
    from .system_defaults import EXPLICIT_COMBINATIONS, PROTEIN_FAMILY_MARKERS, WATER_FAMILY_MARKERS

    if resolved.get("solvation") != "explicit":
        return
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

    wrong = []
    if protein and family(protein, PROTEIN_FAMILY_MARKERS) not in ("", model):
        wrong.append(("forcefield.protein", protein, expected["protein"]))
    if water and family(water, WATER_FAMILY_MARKERS) not in ("", model):
        wrong.append(("forcefield.water", water, expected["water"]))
    if not wrong:
        return

    named = "\n".join(f"      {field}: {value}   -> should be {want}" for field, value, want in wrong)
    raise ConfigError(
        f"solvent.model = {model!r} does not match the force field this configuration names.\n"
        f"{named}\n"
        f"  The protein force field and the water model are ONE selection, not two independent\n"
        f"  keys. {model} is supported only as:\n"
        f"      forcefield.protein: {expected['protein']}\n"
        f"      forcefield.water:   {expected['water']}\n"
        f"      solvent.model:      {model}\n"
        f"  and the other supported explicit selection is\n"
        + "".join(f"      {other}: {values['protein']} + {values['water']}\n"
                  for other, values in EXPLICIT_COMBINATIONS.items() if other != model)
        + "  Crossing them combines a backbone with a water model it was not corrected for; both\n"
          "  run and neither is a combination anyone has validated. See\n"
          "  docs/scientific-defaults.md section 3.\n"
          "  Set solvent.model in your build configuration: "
        + f"{model}` if you did not mean to change it.")


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
    if constraints.get("type") not in ("HBonds", "AllBonds", "HAngles", None):
        raise ConfigError(f"constraints.type {constraints.get('type')!r} is not supported")
