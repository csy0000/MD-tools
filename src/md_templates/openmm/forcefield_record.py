"""`inputs/forcefield.json` -- how the OpenMM System was parameterised.

One record, written from the values the builder actually selected at the point the System was
constructed. It is not derived by reading prose out of `tleap.log`: a log line is a description of
what happened, and this is the thing itself.

Every field that does not apply to a route is present and `null` rather than absent, so a reader
can tell "not applicable here" from "nobody recorded it". That distinction is the whole point of
the file: an implicit system has no water model, and saying so is information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

FORMAT = "md-templates-forcefield/v1"


def build_forcefield_record(*, resolved: dict[str, Any], route: str, record: dict[str, Any],
                            inputs_dir: Path, artifacts: dict[str, str]) -> dict[str, Any]:
    """Assemble the record from the resolved configuration and the builder's own report.

    `artifacts` maps a role name to a path relative to `inputs/`; each is checksummed here so the
    record identifies the parameterisation inputs it names.
    """
    from .provenance_min import environment_versions, sha256_file

    implicit = resolved.get("solvation") == "implicit"
    forcefield = resolved.get("forcefield") or {}
    solute = resolved.get("solute") or {}
    build = resolved.get("system_build") or {}
    constraints = resolved.get("constraints") or {}
    solvent = resolved.get("solvent") or {}
    implicit_solvent = resolved.get("implicit_solvent") or {}

    is_ligand = route != "peptide"
    versions = environment_versions()

    checksums = {}
    for role, relative in sorted(artifacts.items()):
        path = inputs_dir / relative
        if path.is_file():
            checksums[role] = {"path": relative, "sha256": sha256_file(path)}

    return {
        "format": FORMAT,
        "route": route,
        "solvation": "implicit" if implicit else "explicit",

        "protein": {
            # The exact OpenMM resource name, not a family label: `amber19-all.xml` is a file
            # ForceField() resolves, and a different file is a different force field.
            "forcefield": forcefield.get("protein"),
            "openmm_resource": forcefield.get("protein"),
        },
        "ligand": {
            "forcefield": solute.get("ligand_forcefield") if is_ligand else None,
            "charge_method": solute.get("ligand_charge_method") if is_ligand else None,
            "prepared_artifact": (checksums.get("ligand_sdf") or {}).get("path"),
            "prepared_artifact_sha256": (checksums.get("ligand_sdf") or {}).get("sha256"),
        } if is_ligand else _null_ligand(),
        "water": {
            "model": solvent.get("model") if not implicit else None,
            "openmm_resource": forcefield.get("water") if not implicit else None,
            "rigid": constraints.get("rigid_water"),
        },

        "explicit_solvent": None if implicit else {
            "box_shape": solvent.get("box_shape"),
            "padding_nm": solvent.get("padding_nm"),
            "ionic_strength_molar": solvent.get("ionic_strength_molar"),
            "positive_ion": solvent.get("positive_ion"),
            "negative_ion": solvent.get("negative_ion"),
            "cutoff_nm": solvent.get("cutoff_nm", build.get("nonbonded_cutoff_nm")),
            "n_waters": record.get("n_waters"),
            "ions": record.get("ions"),
        },
        "implicit_solvent": {
            "model": implicit_solvent.get("model"),
            "radii": implicit_solvent.get("radii"),
        } if implicit else None,

        "nonbonded": {
            # Implicit systems are non-periodic, so PME does not apply and saying "PME" would
            # describe an interaction that is not being computed.
            "method": "NoCutoff" if implicit else "PME",
            "cutoff_nm": None if implicit else solvent.get(
                "cutoff_nm", build.get("nonbonded_cutoff_nm")),
        },
        "constraints": {
            "type": constraints.get("type"),
            "rigid_water": constraints.get("rigid_water"),
            "hydrogen_mass_amu": constraints.get("hydrogen_mass_amu",
                                                 build.get("hydrogen_mass_amu")),
            "hmr_scope": (record.get("hmr") or {}).get("scope",
                                                       build.get("hmr_scope") or "none"),
            "n_hydrogens_repartitioned": (record.get("hmr") or {}).get(
                "n_hydrogens_repartitioned"),
        },

        "builder": {
            # Which code path produced the System. The implicit route goes through
            # ParmEd.Structure.createSystem rather than AmberPrmtopFile, and the two differ
            # measurably in CustomGBForce -- so the route is part of what the numbers mean.
            "route": ("parmed.Structure.createSystem" if implicit
                      else "openmm.app.ForceField.createSystem"),
            "notes": ("GBn2/mbondi3 built through ParmEd, not AmberPrmtopFile" if implicit
                      else None),
        },
        "package_versions": versions,
        "artifact_checksums": checksums,
    }


def _null_ligand() -> dict[str, Optional[str]]:
    return {"forcefield": None, "charge_method": None,
            "prepared_artifact": None, "prepared_artifact_sha256": None}
