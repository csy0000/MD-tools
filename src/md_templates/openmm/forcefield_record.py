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
                            inputs_dir: Path, artifacts: dict[str, str],
                            builder: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """What the builder ACTUALLY loaded, not what the configuration asked for.

    These are not the same thing, and the differences are the ones that matter:

    * the water label is short; `ForceField()` is given the qualified resource
      (`amber14/tip3p.xml`, `amber19/opc.xml`) that also carries the Na+/Cl- templates. Recording
      the short label describes a file that would have failed to solvate this box.
    * `amber14-all.xml` is a manifest of includes. The protein parameters come from
      `amber14/protein.ff14SB.xml`, and the record names both.
    * the ligand-only route deliberately does NOT load a protein force field. Recording ff14SB
      there names a Hamiltonian that parameterised nothing.
    * the implicit route builds through tleap and ParmEd, not `app.ForceField`. Recording an
      OpenMM XML claims a file constructed a System that no OpenMM XML touched.

    So the builder's own report is the source, and the resolved configuration is used only for
    values the builder does not report (box shape, requested padding, the human-facing labels).

    `builder` is the resolved builder parameter dict `sysgen._legacy_cfg` produced. The user's
    `sys.config.yaml` has no `system_build` block -- it is an internal shape -- so settings like the
    requested nonbonded method and the minimum-image margin are only available from there.
    """
    from .provenance_min import environment_versions, sha256_file

    implicit = resolved.get("solvation") == "implicit"
    requested = resolved.get("forcefield") or {}
    solute = resolved.get("solute") or {}
    build = dict((builder or {}).get("system_build") or {})
    build.update(resolved.get("system_build") or {})
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
    # `build_system` reports the force-field resources under "forcefield" and the built
    # NonbondedForce separately under "nonbonded", so the two come from different levels of the
    # same record. Reading the second out of the first is what silently produced "unknown".
    nonbonded_report = (record.get("nonbonded") or omega.get("nonbonded")
                        or (reported.get("nonbonded") if isinstance(reported, dict) else None)
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

    # Implicit surfaces it at the top level; explicit leaves it inside the `build_system` record
    # that `record["omega"]` carries. Looking in only one place published a null HMR block for a
    # System whose hydrogens really had been repartitioned.
    hmr = (record.get("hmr") or omega.get("hmr")
           or (reported.get("hmr") if isinstance(reported, dict) else None) or {})

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
            # Added salt and neutralising counterions, separated. A box whose only ions are
            # counterions is not a 0.15 M salt solution, and calling it one is a false record.
            "salt": record.get("salt"),
            "box_volume_nm3": record.get("box_volume_nm3"),
            "box_vectors_nm": record.get("box_vectors_nm"),
            # The four quantities that "padding" gets conflated with, kept apart and named. See
            # `solvation._resolve_box`: OpenMM's padding is a requested solute-to-BOX clearance,
            # the solute-to-periodic-COPY distance is set by the shortest lattice translation, and
            # the cutoff legality check uses the reduced-box height, which is smaller than the
            # width for any non-cubic shape.
            "box_geometry": _box_geometry_record(record.get("box_geometry") or {}),
            # The packing box supplies only the starting coordinates; the force field assigns the
            # parameters. When the two differ the substitution is recorded rather than inferred.
            "water_packing_model": record.get("water_packing_model"),
            "water_packing_substituted": record.get("water_packing_substituted"),
            "water_model_reconciled": record.get("water_model_reconciled"),
        },
        "implicit_solvent": {
            "model": implicit_report.get("implicit_model", implicit_solvent.get("model")),
            "radii": implicit_report.get("radii", implicit_solvent.get("radii")),
            "applied_by": implicit_report.get("radii_applied_by",
                                              "parmed.tools.changeRadii + tleap PBRadii"),
            # Whether the ACE surface-area nonpolar term is in the Hamiltonian. Recorded because
            # the two choices differ by ~16 kJ/mol and because ParmEd and OpenMM default
            # differently: a bundle that does not say is a bundle nobody can reproduce.
            "nonpolar_sasa": implicit_report.get(
                "nonpolar_sasa", implicit_solvent.get("nonpolar_sasa", False)),
            "nonpolar_model": implicit_report.get("nonpolar_model"),
            "polar_reference": "GB-Neck2 (Nguyen, Roe & Simmerling, JCTC 2013); Amber igb=8",
            # Measured on the built CustomGBForce: which atoms carry parameters from the GB-Neck2
            # fit and which carry ParmEd's generic fallback. See implicit.gb_parameter_coverage.
            "parameter_coverage": implicit_report.get("parameter_coverage"),
            **_implicit_support_status(is_ligand=is_ligand,
                                       coverage=implicit_report.get("parameter_coverage") or {}),
        } if implicit else None,

        "nonbonded": _nonbonded_record(implicit=implicit, nonbonded=nonbonded_report,
                                       solvent=solvent, build=build),
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
            "hmr_verified_by": hmr.get("verified_by"),
            # Mass conservation checked per heavy-atom/hydrogen group against the same System built
            # without HMR, not only on the box total where two errors of opposite sign cancel.
            "hmr_group_conservation": hmr.get("group_conservation"),
        } if _hmr_consistent(constraints, build, hmr) else _hmr_inconsistent(constraints, build,
                                                                             hmr),

        "builder": {
            # Which code actually built the System. The implicit route goes through
            # ParmEd.Structure.createSystem, not AmberPrmtopFile: the two differ measurably in
            # CustomGBForce, so the route is part of what the numbers mean.
            "route": ("parmed.Structure.createSystem" if implicit
                      else "openmm.app.ForceField.createSystem"),
            "openmm_xml_loaded": (None if implicit else list(reported.get("xml") or [])),
            "openmm_xml_includes": (None if implicit else dict(reported.get("xml_includes") or {})),
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
    includes = (reported.get("xml_includes") or {}).get(loaded) or []
    return {"forcefield": loaded or requested.get("protein"),
            "openmm_resource": loaded,
            # `amber14-all.xml` includes `amber14/protein.ff14SB.xml` and three others. Naming the
            # wrapper alone does not say which file carried the protein parameters.
            "openmm_resource_includes": list(includes),
            "tleap_resource": None,
            "requested_label": requested.get("protein")}


def _ligand_record(*, is_ligand, reported, requested, checksums):
    if not is_ligand:
        return _null_ligand()
    ligand = reported.get("ligand") or {}
    prepared = checksums.get("ligand_sdf") or checksums.get("ligand_mol2") or {}
    return {
        # The exact SMIRNOFF/OpenFF resource the template generator selected, kept separate from
        # the human-facing "sage-2.2.1" label the user typed.
        "forcefield": ligand.get("forcefield") or ligand.get("smirnoff"),
        "openff_resource": ligand.get("forcefield") or ligand.get("smirnoff"),
        "requested_label": requested.get("ligand_forcefield"),
        "charge_method": ligand.get("charge_method") or requested.get("ligand_charge_method"),
        # WHAT WAS ACTUALLY EXECUTED, not what was asked for. The method name alone does not
        # identify a Hamiltonian: `am1bcc` resolves to `am1bcc` or `am1bccelf10` depending on
        # whether OpenEye is available, and those are different quantities over a flexible
        # molecule. The builder reports the scheme it ran; this record must carry it through
        # rather than let a reader infer it from the label.
        #
        # It is deliberately NOT defaulted to the requested method: an absent scheme means the
        # builder did not report one, and saying so is information. Quietly substituting the
        # label here is how the record came to claim `am1bcc` for runs whose scheme was never
        # established.
        "charge_scheme": ligand.get("charge_scheme"),
        # The NAGL model file is a trained artefact that can be upgraded underneath an unchanged
        # configuration, so it is recorded by name and digest. `nagl_model_file` is the key the
        # builder writes; `nagl_model` was looked for here and never existed, which is why the
        # model identity was dropped alongside the scheme.
        "charge_model": ligand.get("nagl_model_file") or ligand.get("charge_model"),
        "charge_model_sha256": ligand.get("nagl_model_sha256"),
        "toolkit_registry": ligand.get("toolkit_registry"),
        # What the charged molecule actually was. The builder computes these from the assigned
        # charges and the parsed molecule, and they were being dropped alongside the scheme. They
        # are what lets a reader check that two builds of the same input produced the same
        # charges -- without them the record says which METHOD ran and nothing about its result.
        "net_charge_e": ligand.get("net_charge_e"),
        "formal_charge": ligand.get("formal_charge"),
        "n_atoms": ligand.get("n_atoms"),
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


#: What this repository claims for an implicit-solvent build, and why. Written into the record so
#: a reader of the archive does not have to reconstruct the scope from the documentation.
_IMPLICIT_SUPPORTED = (
    "supported: peptide/protein through tleap with ff14SB. GBn2 was developed and validated in "
    "the ff99SB/ff14SB lineage, mbondi3's residue-specific adjustments apply, and every atom "
    "carries parameters from the GB-Neck2 fit.")
_IMPLICIT_EXPERIMENTAL_LIGAND = (
    "EXPERIMENTAL: a Sage-parameterised small molecule. GB-Neck2 was fit on peptides and proteins "
    "in the ff99SB/ff14SB lineage; no published work validates it for arbitrary drug-like "
    "chemistry with OpenFF valence and vdW parameters. mbondi3's adjustments are keyed on "
    "GLU/ASP/ARG residue names and the OXT atom name, so for a one-residue ligand mbondi3 reduces "
    "to mbondi2. Use it for exploration, not for a quantitative solvation or binding claim.")
_IMPLICIT_UNFITTED = (
    "EXPERIMENTAL: some atoms are outside the GB-Neck2 element fit and carry ParmEd's generic "
    "alpha/beta/gamma = (1.0, 0.8, 4.85) with screen = 0.5. Those atoms' solvation is not "
    "GB-Neck2 and this System is not Amber igb=8 parity.")


def _hmr_consistent(constraints, build, hmr) -> bool:
    """Was a requested repartitioning actually reported by the builder?

    The configuration asking for 3.024 amu and the record saying nothing is the exact shape of the
    failure this repository has already had once: a profile requesting HMR coexisting with a System
    carrying 1.008 amu hydrogens, with every configuration-only check passing.
    """
    requested = constraints.get("hydrogen_mass_amu", build.get("hydrogen_mass_amu"))
    if requested is None:
        return True
    return bool(hmr) and hmr.get("target_hydrogen_mass_amu") is not None


def _hmr_inconsistent(constraints, build, hmr):
    requested = constraints.get("hydrogen_mass_amu", build.get("hydrogen_mass_amu"))
    raise ValueError(
        f"constraints.hydrogen_mass_amu = {requested} was requested, but the builder reported no "
        f"hydrogen mass repartitioning (got {hmr!r}). Writing this record would claim a "
        f"repartitioned System without evidence that one was built.")


def _implicit_support_status(*, is_ligand: bool, coverage: dict[str, Any]) -> dict[str, Any]:
    """`supported` or `experimental`, decided by what was measured, not by the route label."""
    measured = bool(coverage.get("measured"))
    fully_covered = bool(coverage.get("all_atoms_covered_by_gbn2_fit"))
    if measured and not fully_covered:
        status, note = "experimental", _IMPLICIT_UNFITTED
    elif is_ligand:
        status, note = "experimental", _IMPLICIT_EXPERIMENTAL_LIGAND
    else:
        status, note = "supported", _IMPLICIT_SUPPORTED
    return {
        "support_status": status,
        "support_note": note,
        # The exact-parity claim, stated only where the evidence supports it.
        "amber_igb8_parity_claimed": bool(status == "supported"),
        "amber_igb8_parity_basis": (
            "ff14SB topology from tleap with PBRadii mbondi3, GBn2 with useSASA=False, matching "
            "igb=8 with gbsa=0" if status == "supported" else
            "not claimed: see support_note and parameter_coverage"),
    }


def _box_geometry_record(geometry: dict[str, Any]) -> dict[str, Any]:
    """The box as resolved, under names that say which distance each number is.

    Every key is present and null when the geometry was not reported, so "not recorded" is
    distinguishable from "does not apply". The legacy aliases `solvation._resolve_box` still emits
    are deliberately not copied here: they were the reduced-box height under a name that claimed to
    be the minimum image distance, and a new record should not carry that forward.
    """
    keys = (
        "padding_nm_requested",             # what was asked of Modeller.addSolvent
        "padding_semantics",
        "box_width_nm",                     # what was built
        "box_width_requested_nm",
        "grown_for_cutoff",
        "solute_bounding_radius_nm",
        "shortest_lattice_translation_nm",  # distance to the nearest periodic COPY
        "solute_image_clearance_nm",        # that, minus the solute diameter
        "min_reduced_box_height_nm",        # what OpenMM's cutoff check compares against
        "required_cutoff_height_nm",        # 2*cutoff + minimum-image margin
        "max_legal_cutoff_nm",
        "minimum_image_margin_nm",
    )
    return {key: geometry.get(key) for key in keys}


def _nonbonded_record(*, implicit, nonbonded, solvent, build):
    """The nonbonded treatment as BUILT: every term that changes the energy, not just the cutoff.

    A record that says "PME, 1.0 nm" leaves the switching function, the long-range dispersion
    correction and the Ewald error tolerance unstated, and all three change the numbers.
    """
    if implicit:
        return {"method": "NoCutoff", "cutoff_nm": None, "switching": False,
                "switch_distance_nm": None, "dispersion_correction": False,
                "ewald_error_tolerance": None,
                "note": "non-periodic implicit solvent; PME does not apply"}
    nonbonded = nonbonded or {}
    if not nonbonded.get("method"):
        # The builder always reports this off the built NonbondedForce. Reaching here means the
        # report was not plumbed through, and writing "unknown" would publish a record that says
        # nobody knows how the electrostatics were treated -- for a System that certainly does.
        raise ValueError(
            "the builder reported no nonbonded treatment for an explicit-solvent System. "
            "`build_system` reads it off the built NonbondedForce and returns it under "
            "'nonbonded'; that value did not reach the force-field record.")
    return {
        # Read back off the built NonbondedForce, as a name. `requested_method` is the string the
        # configuration asked for; they agree unless a builder substituted one.
        "method": nonbonded.get("method", "unknown"),
        "requested_method": build.get("nonbonded_method"),
        "cutoff_nm": nonbonded.get(
            "cutoff_nm", solvent.get("cutoff_nm", build.get("nonbonded_cutoff_nm"))),
        "switching": nonbonded.get("switching"),
        "switch_distance_nm": nonbonded.get("switch_distance_nm"),
        "dispersion_correction": nonbonded.get("dispersion_correction"),
        "ewald_error_tolerance": nonbonded.get("ewald_error_tolerance"),
        "minimum_image_margin_nm": build.get("minimum_image_margin_nm"),
    }


def _null_ligand() -> dict[str, Optional[str]]:
    """No ligand on this route. Every key present and null, so a reader can tell."""
    return {"forcefield": None, "openff_resource": None, "requested_label": None,
            "charge_method": None, "charge_scheme": None, "charge_model": None,
            "charge_model_sha256": None, "toolkit_registry": None,
            "net_charge_e": None, "formal_charge": None, "n_atoms": None,
            "prepared_artifact": None, "prepared_artifact_sha256": None}
