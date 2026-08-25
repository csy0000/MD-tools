"""The default system and protocol settings, defined once.

`sys-config`, `md-config` and `show-default` all read these. There is no second copy: the reason
this repository accumulated drift guards is that the same fact used to be declared in five places,
and a change would update two of them.

Values are plain Python data that serialise to the YAML a user edits. Nothing here validates -- see
`config.py` -- and nothing here is versioned per-profile. A user who wants different settings edits
the YAML.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1
ENGINE = "openmm"
ENGINE_VERSION = "8.6.0"

#: Canonical spellings. Input is accepted case-insensitively and written back in these forms.
METHODS = ("cMD", "REST2")
SOLVENTS = ("OPC", "GBn2")


def canonical_method(name: str) -> str:
    for canonical in METHODS:
        if str(name).lower() == canonical.lower():
            return canonical
    raise ValueError(f"unknown method {name!r}; expected one of {', '.join(METHODS)}")


def canonical_solvent(name: str) -> str:
    for canonical in SOLVENTS:
        if str(name).lower() == canonical.lower():
            return canonical
    raise ValueError(f"unknown solvent {name!r}; expected one of {', '.join(SOLVENTS)}")


def is_implicit(solvent: str) -> bool:
    return canonical_solvent(solvent) == "GBn2"


def sys_defaults(*, peptide: bool = True, solvent: str = "OPC") -> dict[str, Any]:
    """System preparation settings.

    Both the explicit and implicit blocks are written so the file documents what the other option
    would look like, but only the one matching `solvent` is used when the configuration is
    resolved -- `config.resolve_sys_config` drops the other and says so.
    """
    solvent = canonical_solvent(solvent)
    implicit = is_implicit(solvent)
    document = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "solute": {
            "peptide": bool(peptide),
            # Sage 2.2 and standard AM1-BCC (AmberTools sqm). `am1bcc_nagl` is a graph network
            # TRAINED to predict AM1-BCC ELF10 charges -- close but not that calculation -- and is
            # selected explicitly or not at all.
            "ligand_forcefield": "sage-2.2.0",
            "ligand_charge_method": "am1bcc",
        },
        "forcefield": {
            "protein": "amber19-all.xml",
            "water": "opc.xml",
        },
        "solvent": {
            "model": solvent if not implicit else "OPC",
            # OpenMM's padding semantics: width = max(2R + padding, 2 * padding), so this is a
            # LOWER bound on the solute-to-periodic-image separation, not the box width.
            "padding_nm": 2.0,
            "box_shape": "dodecahedron",
            "ionic_strength_molar": 0.15,
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
            "cutoff_nm": 1.0,
        },
        "implicit_solvent": {
            "model": "GBn2",
            "radii": "mbondi3",
        },
        "constraints": {
            "type": "HBonds",
            "rigid_water": True,
            # null: hydrogens keep the masses the force field gave them, and the timestep stays at
            # 2 fs. Set to 3.024 for HMR, and raise timestep_fs to 4.0 with it -- 4 fs on
            # unrepartitioned hydrogens is the unstable combination.
            "hydrogen_mass_amu": None,
        },
    }
    # Only the block that applies is written. The file describes ONE system; carrying both an
    # explicit water box and a GB model would leave a reader -- and the resolver -- to guess which
    # one the run uses, which is exactly the ambiguity `resolve_sys_config` exists to avoid.
    document.pop("implicit_solvent" if not implicit else "solvent")
    if implicit:
        document["forcefield"]["water"] = None
        document["constraints"]["rigid_water"] = False
    return document


def md_defaults(*, methods=("cMD", "REST2"), solvent: str = "OPC") -> dict[str, Any]:
    """Simulation protocol settings.

    Under implicit solvent there is no box, so there is no barostat and no pressure: the production
    ensembles become NVT and `pressure_bar` is written as null with a note. Saying "NPT" for a
    non-periodic system would be a false record of the ensemble that ran.
    """
    methods = [canonical_method(m) for m in methods]
    implicit = is_implicit(solvent)
    ensemble = "NVT" if implicit else "NPT"

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "methods": methods,
        "common": {
            "temperature_kelvin": 300,
            "pressure_bar": None if implicit else 1.0,
            "timestep_fs": 2.0,
            "friction_per_ps": 1.0,
            "random_seed": None,
        },
        "minimization": {
            "max_iterations": 1000,
            "restraint_k_kcal_mol_a2": 1.0,
        },
        # One directory per stage is generated from this block, in this order. Every duration is
        # the length of that stage; a null one means the stage does not apply to this solvent.
        "equilibration": {
            "nvt_restrained_duration_ps": 10.0,
            "npt_restrained_duration_ps": None if implicit else 10.0,
            "npt_free_duration_ps": None if implicit else 10.0,
            "nvt_free_duration_ps": 10.0 if implicit else None,
            "restraint_k_kcal_mol_a2": 1.0,
        },
    }
    if implicit:
        document["common"]["pressure_note"] = (
            "implicit solvent has no periodic box, so pressure is not applicable and no barostat "
            "is added")
        document["equilibration"]["npt_note"] = (
            "not applicable without a box: the free stage is NVT")

    if "cMD" in methods:
        document["cMD"] = {
            "ensemble": ensemble,
            # A single walker on ONE fixed rung of the REST2 ladder. tau = 0 is ordinary cMD: the
            # unscaled, physical Hamiltonian, and the System is left byte-identical. tau > 0 keeps
            # the same thermostat temperature and scales only the solute Hamiltonian, exactly as
            # the matching REST2 rung does -- it is NOT high-temperature MD, and beta is unchanged.
            # tau is the SOURCE parameter; s = (1 - tau)^2 is a derived diagnostic.
            "tau": 0.0,
            # Only meaningful at tau > 0, and matched to the REST2 default so a fixed-tau walker
            # and the ladder rung at the same tau are the same Hamiltonian.
            "omega_exclusion": True,
            "duration_ns": 5,
            "checkpoint_interval_ps": 100,
            "whole_system_interval_ps": 100,
            "solute_interval_ps": 10,
        }
    if "REST2" in methods:
        document["REST2"] = {
            "ensemble": ensemble,
            # tau is the SOURCE parameter. The scaling is derived: solute-solute (1-tau)^2,
            # solute-environment (1-tau). See openmm/tau.py.
            "tau_min": 0.0,
            "tau_max": 0.5,
            "number_of_replicas": 6,
            "tau_interpolation": "linear",
            # Time BETWEEN consecutive exchange rounds. Total production is
            # duration_per_segment_ps * number_of_exchanges, so this default is 10 ns per replica.
            # Per-tau equilibration, run once per replica by `REST2/equilibrate.py` before any
            # exchange. It is NOT production and is not counted in the totals below.
            "equilibration_duration_ps": 1000.0,
            "duration_per_segment_ps": 10,
            "number_of_exchanges": 1000,
            "enhanced_region": "solute",
            "omega_exclusion": True,
            "checkpoint_interval_ps": 100,
            "whole_system_interval_ps": 100,
            "solute_interval_ps": 10,
        }
    return document


def default_document(name: str) -> dict[str, Any]:
    """`show-default <name>` -- the same definitions `sys-config` writes."""
    key = str(name).lower()
    if key == "sys":
        return sys_defaults()
    if key == "cmd":
        return {k: v for k, v in md_defaults(methods=["cMD"]).items() if k != "REST2"}
    if key == "rest2":
        return {k: v for k, v in md_defaults(methods=["REST2"]).items() if k != "cMD"}
    if key == "all":
        return {"sys": sys_defaults(), "md": md_defaults()}
    raise ValueError(f"unknown default {name!r}; expected sys, cMD, REST2 or all")
