"""Translate portable manifests into the `explicit_baseline` configuration tree.

The science lives in `escort_ais.systems.explicit_baseline`, which is driven by one nested dict of
every knob it has. The portable manifests are deliberately much smaller than that dict: they carry
identity, the handful of controls that define an experiment, and nothing else, so that a manifest
stays readable and reviewable.

This module is the single place the two representations meet. Keeping it here — rather than
letting the CLI reach into the config tree in six places — means there is exactly one answer to
"which manifest field set this baseline knob", and the resolved tree is written into every bundle
and run directory so the answer is recoverable after the fact.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from escort_ais.systems.explicit_baseline import DEFAULTS, _deep_merge

from .schemas import ExperimentManifest, ManifestError, SystemManifest

#: Which `solute_kind` each portable route implies. The mapping is one-way and total: a portable
#: manifest never says `auto`, so the baseline's route-resolution never has to guess.
ROUTE_TO_SOLUTE_KIND = {"smiles": "ligand", "pdb": "peptide"}


def resolve_config(
    system: SystemManifest,
    experiment: ExperimentManifest,
    *,
    platform: str = "CPU",
    device: Optional[str] = None,
) -> dict[str, Any]:
    """The fully resolved `explicit_baseline` config for this (system, experiment, machine).

    Machine choices (`platform`, `device`) are applied last and are deliberately NOT part of the
    config hash: the same calculation on CPU and on CUDA is the same configuration.
    """
    cfg = copy.deepcopy(DEFAULTS)
    sdoc, edoc = system.doc, experiment.doc

    # ---- identity and route -------------------------------------------------------------------
    cfg["system"]["slug"] = system.system_id
    cfg["system"]["solute_kind"] = ROUTE_TO_SOLUTE_KIND[system.route]
    # enforced, not documented: a pdb handed to a smiles-route manifest must fail before any force
    # field is built, rather than quietly becoming an ff19SB peptide calculation
    cfg["system"]["require_input_route"] = system.route

    par = sdoc["parameterization"]
    cfg["forcefield"]["water"] = par["water_forcefield"]
    if system.route == "smiles":
        cfg["forcefield"]["ligand"] = par["small_molecule_forcefield"]
        cfg["forcefield"]["ligand_charge_method"] = par["charge_method"]
    else:
        cfg["forcefield"]["protein"] = par["protein_forcefield"]

    for key, value in (sdoc.get("solvation") or {}).items():
        if key not in cfg["solvation"]:
            raise ManifestError(
                f"{system.source}: solvation.{key} is not a recognised solvation setting; "
                f"known keys are {sorted(cfg['solvation'])}"
            )
        cfg["solvation"][key] = value

    # ---- experiment controls -------------------------------------------------------------------
    cfg["run"]["seed"] = experiment.master_seed
    cfg["run"]["name"] = system.system_id

    integ = edoc["integrator"]
    cfg["integrator"]["kind"] = integ["kind"]
    cfg["integrator"]["temperature_k"] = float(integ["temperature_k"])
    cfg["integrator"]["timestep_fs"] = float(integ["timestep_fs"])

    rest2 = edoc["rest2"]
    remd = cfg["production"]["remd"]
    remd["scale_factors"] = [float(v) for v in rest2["scale_factors"]]
    remd["exchange_interval_ps"] = float(rest2["exchange_interval_ps"])
    remd["equilibration_ps"] = float(rest2["relaxation_ps"])
    remd["total_ns_per_replica"] = float(rest2["total_ns_per_replica"])
    remd["chunk_ns"] = float(rest2["chunk_ns"])
    cfg["production"]["precision"] = edoc["platform"]["precision"]

    for key, value in (edoc.get("equilibration") or {}).items():
        if key not in cfg["equilibration"]:
            raise ManifestError(
                f"{experiment.source}: equilibration.{key} is not a recognised setting"
            )
        cfg["equilibration"][key] = value

    # An explicit escape hatch for the rest of the baseline tree. Unknown keys raise (deep_merge
    # rejects them), so a typo cannot leave a baseline value in force while the manifest claims
    # otherwise -- which is the failure this whole layer exists to prevent.
    overrides = edoc.get("overrides")
    if overrides:
        cfg = _deep_merge(cfg, copy.deepcopy(overrides))

    # ---- machine choices, applied last and excluded from the config hash -----------------------
    cfg["production"]["platform"] = platform
    cfg["production"]["device_index"] = device

    _resolve_seeds(cfg)
    _check_exchange_divisibility(cfg, experiment)
    return cfg


def _resolve_seeds(cfg: dict) -> None:
    """Derive the per-stage seeds from the master seed, exactly as `load_config` does.

    Duplicated deliberately rather than calling `load_config`: that function's entry point is a
    JSON file, and routing a manifest through a temporary file to obtain seed derivation would put
    a filesystem round-trip in the middle of a pure transformation.
    """
    master = int(cfg["run"]["seed"])
    for offset, (section, key) in enumerate(
        [
            (cfg["structure"]["etkdg"], "seed"),
            (cfg["equilibration"], "seed"),
            (cfg["production"]["md"], "seed"),
            (cfg["production"]["remd"], "seed"),
        ]
    ):
        if section.get(key) is None:
            section[key] = master + offset


def _check_exchange_divisibility(cfg: dict, experiment: ExperimentManifest) -> None:
    """Fail here, with the arithmetic shown, rather than deep inside the REMD driver.

    `run_rest2_remd` requires a chunk to be a whole number of exchange intervals so that every
    replica reaches a chunk boundary on the same exchange round. Violating it raises there too,
    but by then the System has been built and the user has paid for parameterisation.
    """
    dt_fs = float(cfg["integrator"]["timestep_fs"])
    remd = cfg["production"]["remd"]
    exchange_steps = round(float(remd["exchange_interval_ps"]) * 1000.0 / dt_fs)
    chunk_steps = round(float(remd["chunk_ns"]) * 1e6 / dt_fs)
    if exchange_steps <= 0 or chunk_steps <= 0:
        raise ManifestError(
            f"{experiment.source}: exchange interval and chunk must each be at least one step at "
            f"{dt_fs} fs"
        )
    if chunk_steps % exchange_steps != 0:
        raise ManifestError(
            f"{experiment.source}: rest2.chunk_ns must be a whole number of exchange intervals.\n"
            f"  chunk    {remd['chunk_ns']} ns = {chunk_steps} steps at {dt_fs} fs\n"
            f"  exchange {remd['exchange_interval_ps']} ps = {exchange_steps} steps\n"
            f"  {chunk_steps} / {exchange_steps} = {chunk_steps / exchange_steps:.4f}, not an "
            "integer"
        )


def exchange_rounds(cfg: dict) -> int:
    """How many exchange attempts the configured budget will make, in total."""
    dt_fs = float(cfg["integrator"]["timestep_fs"])
    remd = cfg["production"]["remd"]
    exchange_steps = round(float(remd["exchange_interval_ps"]) * 1000.0 / dt_fs)
    total_steps = round(float(remd["total_ns_per_replica"]) * 1e6 / dt_fs)
    return int(total_steps // exchange_steps)
