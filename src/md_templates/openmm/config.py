"""Read the two YAML files a user edits, and resolve them into what will actually be built.

Deliberately thin. There is no schema-migration layer, no profile inheritance and no canonical
intermediate model: the YAML is the configuration, and resolving it means dropping the solvent
block that does not apply and checking the handful of combinations that are physically wrong.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .defaults import canonical_method, canonical_solvent, is_implicit

__all__ = ["load_yaml", "write_yaml", "resolve_sys_config", "resolve_md_config",
           "sha256_of_document", "ConfigError"]


class ConfigError(ValueError):
    """A configuration that cannot be built, with the reason stated."""


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    return document


def write_yaml(path: Path, document: dict[str, Any], *, header: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=88)
    path.write_text((header + text) if header else text, encoding="utf-8")
    return path


def sha256_of_document(document: dict[str, Any]) -> str:
    """A stable hash of a configuration, for provenance."""
    canonical = yaml.safe_dump(document, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    _check_constraints(resolved)
    _check_protein_solvation_pairing(resolved)
    return resolved


def _check_protein_solvation_pairing(resolved: dict[str, Any]) -> None:
    """Refuse a protein force field that was not parameterised for this solvation model.

    ff19SB's amino-acid-specific CMAPs were fit in explicit OPC water, and no GB model has been
    reparameterised against them; GBn2 was developed and validated in the ff99SB/ff14SB lineage.
    Running the pair produces numbers, which is exactly the problem -- nothing fails, and the
    result silently describes a Hamiltonian nobody validated.

    This is refused rather than warned about because a warning in a log is not read by whoever
    reads the trajectory a year later.
    """
    from .defaults import GB_INCOMPATIBLE_PROTEIN, IMPLICIT_PROTEIN_FORCEFIELD

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


def resolve_md_config(document: dict[str, Any], *, implicit: bool) -> dict[str, Any]:
    """The effective protocol: ensembles reconciled with the solvent, and timestep checked."""
    resolved = yaml.safe_load(yaml.safe_dump(document))
    methods = [canonical_method(m) for m in (resolved.get("methods") or [])]
    if not methods:
        raise ConfigError("md config lists no methods")
    resolved["methods"] = methods

    common = resolved.setdefault("common", {})
    if implicit:
        # A non-periodic system has no volume to control. Saying NPT here would name an ensemble
        # the run cannot sample.
        common["pressure_bar"] = None
        for method in methods:
            block = resolved.get(method) or {}
            if str(block.get("ensemble", "")).upper() == "NPT":
                block["ensemble"] = "NVT"
            resolved[method] = block
    else:
        if common.get("pressure_bar") is None:
            raise ConfigError(
                "explicit solvent needs common.pressure_bar for the barostat; it is null. "
                "Set it (1.0 bar is the default) or switch the system config to GBn2.")

    _check_timestep(resolved)
    _check_tau(resolved)
    return resolved


def _check_tau(resolved: dict[str, Any]) -> None:
    """`cMD.tau` selects one fixed rung of the REST2 ladder; validate it before anything is built.

    Refused here rather than deep inside a generated script, where the failure would arrive after
    the System had been constructed and the run directory opened. tau = 1 is excluded because
    s = (1 - tau)^2 would be zero: a solute with no intramolecular Hamiltonian at all is not a rung
    of the ladder, it is a different calculation.
    """
    block = resolved.get("cMD") or {}
    if "tau" not in block:
        return
    tau = block["tau"]
    try:
        tau = float(tau)
    except (TypeError, ValueError):
        raise ConfigError(f"cMD.tau must be a number in [0, 1); got {block['tau']!r}") from None
    if not 0.0 <= tau < 1.0:
        raise ConfigError(
            f"cMD.tau must be in [0, 1); got {tau}. tau = 0 is ordinary conventional MD on the "
            "unscaled Hamiltonian, and tau -> 1 removes the solute Hamiltonian entirely.")
    resolved["cMD"]["tau"] = tau


def _check_timestep(resolved: dict[str, Any]) -> None:
    """4 fs is only defensible with repartitioned hydrogens; this is where that pair is checked.

    The system config carries the mass, so this check needs both files and is applied by the
    caller through `check_timestep_against_masses`.
    """
    timestep = float((resolved.get("common") or {}).get("timestep_fs", 2.0))
    if timestep <= 0:
        raise ConfigError(f"common.timestep_fs must be positive; got {timestep}")


def check_timestep_against_masses(md_resolved: dict[str, Any],
                                  sys_resolved: dict[str, Any]) -> None:
    """Refuse 4 fs on unrepartitioned hydrogens, naming both values.

    This is the one cross-file check worth making: the two settings live in different files and are
    individually reasonable, so nothing else would catch the combination.
    """
    timestep = float((md_resolved.get("common") or {}).get("timestep_fs", 2.0))
    mass = (sys_resolved.get("constraints") or {}).get("hydrogen_mass_amu")
    if timestep > 3.0 and mass is None:
        raise ConfigError(
            f"md.config.yaml sets common.timestep_fs = {timestep} but sys.config.yaml leaves "
            "constraints.hydrogen_mass_amu null, so hydrogens keep their real mass. A timestep "
            "above ~3 fs needs hydrogen mass repartitioning; set hydrogen_mass_amu to 3.024, or "
            "lower the timestep to 2.0.")
