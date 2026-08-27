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
    """What the builder ACTUALLY loaded, not what the configuration asked for.

    These are not the same thing, and the differences are the ones that matter:

    * the user writes `opc.xml`; `ForceField()` is given `amber19/opc.xml`, the qualified file that
      also carries the Na+/Cl- templates. Recording the short label describes a file that would
      have failed to solvate this box.
    * the ligand-only route deliberately does NOT load a protein force field. Recording ff19SB
      there names a Hamiltonian that parameterised nothing.
    * the implicit route builds through tleap and ParmEd, not `app.ForceField`. Recording
      `amber19-all.xml` claims an OpenMM XML constructed a System that no OpenMM XML touched.

    So the builder's own report is the source, and the resolved configuration is used only for
    values the builder does not report (box shape, requested padding, the human-facing labels).
    """
    from .provenance_min import environment_versions, sha256_file

    implicit = resolved.get("solvation") == "implicit"
    requested = resolved.get("forcefield") or {}
    solute = resolved.get("solute") or {}
    build = resolved.get("system_build") or {}
    constraints = resolved.get("constraints") or {}
    solvent = resolved.get("solvent") or {}
    implicit_solvent = resolved.get("implicit_solvent") or {}
    is_ligand = route != "peptide"

    # The builder's report. Explicit: `build_system(...)["forcefield"]`. Implicit: the tleap and
    # ParmEd record. Absent only if a builder stopped reporting, which the assertions below catch.
    omega = record.get("omega") or {}
    implicit_report = (record.get("implicit_report") or omega.get("implicit")
                       or record.get("implicit") or {})
    # Explicit: build_system reports under "forcefield". Implicit: the ligand's OpenFF report is
    # carried through the tleap/ParmEd record as "forcefield_info".
    reported = (record.get("forcefield")
                or omega.get("forcefield")
                or implicit_report.get("forcefield_info")
                or {})

    checksums = {}
    for role, relative in sorted(artifacts.items()):
        path = inputs_dir / relative
        if path.is_file():
            checksums[role] = {"path": relative, "sha256": sha256_file(path)}

    protein = _protein_record(implicit=implicit, is_ligand=is_ligand, reported=reported,
                              requested=requested, implicit_report=implicit_report)
    ligand = _ligand_record(is_ligand=is_ligand, reported=reported, requested=solute,
                            checksums=checksums)
    water = _water_record(implicit=implicit, reported=reported, solvent=solvent,
                          constraints=constraints)

    hmr = record.get("hmr") or (reported.get("hmr") if isinstance(reported, dict) else None) or {}

    return {
        "format": FORMAT,
        "route": route,
        "solvation": "implicit" if implicit else "explicit",
        "source": "builder report (resources actually loaded), with requested values labelled",

        "protein": protein,
        "ligand": ligand,
        "water": water,

        "explicit_solvent": None if implicit else {
            "box_shape": record.get("box_shape", solvent.get("box_shape")),
            "padding_nm": solvent.get("padding_nm"),
            "ionic_strength_molar": solvent.get("ionic_strength_molar"),
            "positive_ion": solvent.get("positive_ion"),
            "negative_ion": solvent.get("negative_ion"),
            "cutoff_nm": solvent.get("cutoff_nm", build.get("nonbonded_cutoff_nm")),
            "n_waters": record.get("n_waters"),
            "ions": record.get("ions"),
            "box_volume_nm3": record.get("box_volume_nm3"),
        },
        "implicit_solvent": {
            "model": implicit_report.get("implicit_model", implicit_solvent.get("model")),
            "radii": implicit_report.get("radii", implicit_solvent.get("radii")),
            "applied_by": implicit_report.get("radii_applied_by",
                                              "parmed.tools.changeRadii + tleap PBRadii"),
        } if implicit else None,

        "nonbonded": _nonbonded_record(implicit=implicit, reported=reported, solvent=solvent,
                                       build=build),
        "constraints": {
            "type": constraints.get("type"),
            "rigid_water": (reported.get("rigid_water") if "rigid_water" in reported
                            else constraints.get("rigid_water")),
            "hydrogen_mass_amu": (hmr.get("target_hydrogen_mass_amu")
                                  or constraints.get("hydrogen_mass_amu")
                                  or build.get("hydrogen_mass_amu")),
            "hmr_scope": hmr.get("scope", build.get("hmr_scope") or "none"),
            "n_hydrogens_repartitioned": hmr.get("n_hydrogens_repartitioned"),
            "hmr_performed_by": hmr.get("performed_by"),
        },

        "builder": {
            # Which code actually built the System. The implicit route goes through
            # ParmEd.Structure.createSystem, not AmberPrmtopFile: the two differ measurably in
            # CustomGBForce, so the route is part of what the numbers mean.
            "route": ("parmed.Structure.createSystem" if implicit
                      else "openmm.app.ForceField.createSystem"),
            "openmm_xml_loaded": (None if implicit else list(reported.get("xml") or [])),
            "tleap_used": bool(implicit),
            "notes": ("GBn2/mbondi3 built through ParmEd, not AmberPrmtopFile" if implicit
                      else None),
        },
        "package_versions": environment_versions(),
        "artifact_checksums": checksums,
    }


def _protein_record(*, implicit, is_ligand, reported, requested, implicit_report):
    """Null where no protein force field was loaded; the exact resource where one was."""
    if is_ligand:
        # The ligand route does not load a protein force field at all. Naming one here would
        # attribute parameters to a file that contributed none.
        return {"forcefield": None, "openmm_resource": None, "tleap_resource": None,
                "note": "the ligand-only route loads no protein force field"}
    if implicit:
        return {
            "forcefield": implicit_report.get("protein_forcefield"),
            # No OpenMM protein XML is loaded on this route; tleap writes the topology.
            "openmm_resource": None,
            "tleap_resource": implicit_report.get("protein_forcefield"),
        }
    loaded = reported.get("protein_forcefield")
    return {"forcefield": loaded or requested.get("protein"),
            "openmm_resource": loaded,
            "tleap_resource": None,
            "requested_label": requested.get("protein")}


def _ligand_record(*, is_ligand, reported, requested, checksums):
    if not is_ligand:
        return _null_ligand()
    ligand = reported.get("ligand") or {}
    prepared = checksums.get("ligand_sdf") or checksums.get("ligand_mol2") or {}
    return {
        # The exact SMIRNOFF/OpenFF resource the template generator selected, kept separate from
        # the human-facing "sage-2.2.0" label the user typed.
        "forcefield": ligand.get("forcefield") or ligand.get("smirnoff"),
        "openff_resource": ligand.get("forcefield") or ligand.get("smirnoff"),
        "requested_label": requested.get("ligand_forcefield"),
        "charge_method": ligand.get("charge_method") or requested.get("ligand_charge_method"),
        "charge_model": ligand.get("nagl_model") or ligand.get("charge_model"),
        "toolkit_registry": ligand.get("toolkit_registry"),
        "prepared_artifact": prepared.get("path"),
        "prepared_artifact_sha256": prepared.get("sha256"),
    }


def _water_record(*, implicit, reported, solvent, constraints):
    if implicit:
        # There is no water. Saying so explicitly is information; omitting the key is not.
        return {"model": None, "openmm_resource": None, "requested_label": None, "rigid": None,
                "note": "implicit solvent has no water model"}
    loaded = reported.get("water")
    return {"model": solvent.get("model"),
            # `amber19/opc.xml`, not `opc.xml`: the qualified file also defines the ion templates
            # this box needed.
            "openmm_resource": loaded,
            "requested_label": solvent.get("model"),
            "rigid": constraints.get("rigid_water")}


def _nonbonded_record(*, implicit, reported, solvent, build):
    if implicit:
        return {"method": "NoCutoff", "cutoff_nm": None,
                "note": "non-periodic implicit solvent; PME does not apply"}
    nonbonded = reported.get("nonbonded") if isinstance(reported, dict) else None
    method = (nonbonded or {}).get("method", "PME")
    return {"method": method,
            "cutoff_nm": solvent.get("cutoff_nm", build.get("nonbonded_cutoff_nm"))}


def _null_ligand() -> dict[str, Optional[str]]:
    """No ligand on this route. Every key present and null, so a reader can tell."""
    return {"forcefield": None, "openff_resource": None, "requested_label": None,
            "charge_method": None, "charge_model": None, "toolkit_registry": None,
            "prepared_artifact": None, "prepared_artifact_sha256": None}
