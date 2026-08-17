"""What makes a prepared System *that* System, and nothing else.

`md-openmm rest2 --bundle B --experiment E` lets a new experiment run against an
already-prepared bundle. That is genuinely useful — running longer, or on another device, against
the identical starting state is the whole point of a transferable bundle. It is also the obvious
way to produce a silently wrong result: nothing in the file layout stops a new experiment from
declaring a different force field, a different box, or a different constraint scheme, none of
which can take effect, because `system.xml` and `equilibrated_state.xml` were built under the old
settings. The run would then be *described* by one configuration and *performed* under another.

So the bundle records a **build-defining projection** of its resolved configuration — every
setting that changed the serialised System or the meaning of the stored starting state — and its
SHA-256. A launch-time override recomputes the same projection and must match.

The split is the substance here. Something belongs in the projection if changing it would make
the stored `system.xml`/`equilibrated_state.xml` the wrong artifacts; it belongs outside if the
prepared System is still exactly right and only the run differs. Comparing experiment IDs or
filenames would not distinguish those cases at all.
"""
from __future__ import annotations

from typing import Any, Optional

from .hashing import canonical_json, sha256_text

#: Dotted paths whose values define the prepared System. Changing any of them invalidates a
#: prepared bundle, because the stored System or starting state was built under the old value.
BUILD_DEFINING_PATHS: tuple[str, ...] = (
    # --- identity and route: which molecule, and which force-field pathway ---------------------
    "system.slug",
    "system.solute_kind",
    "system.require_input_route",
    # --- force field and charges ---------------------------------------------------------------
    "forcefield.protein",
    "forcefield.water",
    "forcefield.ligand",
    "forcefield.ligand_charge_method",
    "forcefield.extra_xml",
    # --- the prepared solute: embedding and protonation ----------------------------------------
    "structure.etkdg.version",
    "structure.etkdg.n_conformers",
    "structure.etkdg.seed",
    "structure.etkdg.use_random_coords",
    "structure.etkdg.prune_rms_thresh",
    "structure.mmff.variant",
    "structure.mmff.max_iterations",
    "protonation.ph",
    "protonation.delete_existing_hydrogens",
    "protonation.variants",
    "protonation.skip_for_ligand",
    # --- solvation: box, water, salt ------------------------------------------------------------
    "solvation.water_model",
    "solvation.box_shape",
    "solvation.padding_nm",
    "solvation.padding_semantics",
    "solvation.cutoff_fit_policy",
    "solvation.ionic_strength_molar",
    "solvation.positive_ion",
    "solvation.negative_ion",
    "solvation.neutralize",
    # --- the System object itself ---------------------------------------------------------------
    "system_build.nonbonded_method",
    "system_build.nonbonded_cutoff_nm",
    "system_build.switch_distance_nm",
    "system_build.use_dispersion_correction",
    "system_build.ewald_error_tolerance",
    "system_build.constraints",
    "system_build.rigid_water",
    "system_build.hydrogen_mass_amu",
    "system_build.hmr_scope",
    "system_build.remove_cm_motion",
    "system_build.minimum_image_margin_nm",
    # --- REST2 omega selection: decides which torsions the stored simbox metadata excludes ------
    "rest2.omega_exclusion",
    "rest2.proline_like_residues",
    "rest2.max_proline_ring_size",
    # --- what produced equilibrated_state.xml ---------------------------------------------------
    "equilibration.protocol",
    "equilibration.minimize_max_iterations",
    "equilibration.minimize_tolerance_kj_mol_nm",
    "equilibration.restraint_k_kj_mol_nm2",
    "equilibration.restraint_selection",
    "equilibration.heat_from_k",
    "equilibration.heat_to_k",
    "equilibration.heat_ps",
    "equilibration.heat_timestep_fs",
    "equilibration.heat_n_windows",
    "equilibration.npt_restrained_ps",
    "equilibration.release_schedule_kj_mol_nm2",
    "equilibration.release_ps_each",
    "equilibration.npt_free_ps",
    "equilibration.timestep_fs",
    "equilibration.nvt_ps",
    "equilibration.npt_ps",
    "equilibration.pressure_bar",
    "equilibration.barostat_interval",
    "equilibration.box_average_last_ps",
    "equilibration.seed",
    # --- the integrator the equilibration ran under ---------------------------------------------
    "integrator.kind",
    "integrator.temperature_k",
    "integrator.friction_per_ps",
)

#: Deliberately EXCLUDED, with the reason. These change the run, not the prepared System: the
#: stored `system.xml` and `equilibrated_state.xml` remain exactly the right artifacts.
RUNTIME_ONLY_PATHS: dict[str, str] = {
    "run.root": "output location",
    "run.name": "output naming",
    "production.platform": "which OpenMM platform executes the same System",
    "production.device_index": "which device executes it",
    "production.precision": "arithmetic precision of the propagation, not of the System",
    "production.report.all_atom_ps": "reporting cadence",
    "production.report.solute_ps": "reporting cadence",
    "production.report.state_ps": "reporting cadence",
    "production.report.checkpoint_ps": "reporting cadence",
    "production.remd.n_chunks": "how long to run the same ladder",
    "production.remd.chunk_ns": "restart granularity",
    "production.remd.scale_factors": "the REST2 ladder is applied at run time to the prepared "
                                     "System; it does not change the System that was prepared",
    "production.remd.exchange_interval_ps": "exchange cadence",
    "production.remd.equilibration_ps": "pre-exchange relaxation, performed at run time",
    "production.remd.seed": "run-time RNG",
    "production.md.n_chunks": "unused by REST2",
    "production.md.chunk_ns": "unused by REST2",
    "production.md.seed": "unused by REST2",
    "production.md.scale_factor": "unused by REST2",
    "production.md.label": "unused by REST2",
    "production.ensemble": "checked separately by the REMD driver",
    "run.seed": "the derived stage seeds are compared individually; the master alone is a label",
    "integrator.timestep_fs": "the production timestep; the equilibration timestep is compared "
                              "separately as equilibration.timestep_fs",
}


def _get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def build_projection(cfg: dict) -> dict[str, Any]:
    """The build-defining subset of a resolved config, as a flat dotted mapping."""
    return {path: _get(cfg, path) for path in BUILD_DEFINING_PATHS}


def fingerprint(cfg: dict) -> str:
    """SHA-256 of the build-defining projection. Same prepared System <=> same fingerprint.

    Takes the RESOLVED CONFIG, not a projection. Passing an already-built projection used to
    succeed and return a constant: every dotted lookup missed on the flat mapping, so the hash was
    of `{path: None, ...}` and two different systems compared equal. A wrong answer that looks like
    a hash is worse than an error, so this refuses.
    """
    if cfg and all("." in str(k) for k in cfg):
        raise TypeError(
            "fingerprint() takes a resolved configuration, but was given what looks like a "
            "build-defining projection (every key is dotted). Passing a projection would hash a "
            "mapping of Nones and make unrelated systems compare equal. Call fingerprint(cfg), or "
            "sha256_text(canonical_json(projection)) if you really have a projection."
        )
    return sha256_text(canonical_json(build_projection(cfg)))


def diff_projections(bundle_proj: dict, proposed_proj: dict) -> list[tuple[str, Any, Any]]:
    """Differing `(dotted_path, bundle_value, proposed_value)`, sorted by path.

    Includes paths present in one projection and not the other, so a bundle built by an older
    version with fewer recorded paths is reported rather than silently accepted.
    """
    out: list[tuple[str, Any, Any]] = []
    for path in sorted(set(bundle_proj) | set(proposed_proj)):
        a, b = bundle_proj.get(path, "<absent>"), proposed_proj.get(path, "<absent>")
        if a != b:
            out.append((path, a, b))
    return out


def describe_incompatibility(diffs: list[tuple[str, Any, Any]]) -> str:
    lines = [
        "the requested experiment describes a DIFFERENT prepared System than this bundle contains.",
        "",
        "The bundle's system.xml and equilibrated_state.xml were built under the first value; the",
        "new setting cannot take effect, so the run would be described by one configuration and",
        "performed under another. Re-run `prepare` with the new experiment instead.",
        "",
        f"{'setting':<52} {'bundle':<24} proposed",
    ]
    for path, old, new in diffs:
        lines.append(f"{path:<52} {str(old)[:23]:<24} {new}")
    return "\n".join(lines)


def check_compatible(bundle_manifest: dict, proposed_cfg: dict) -> Optional[str]:
    """`None` when the override may run against this bundle, else a human-readable reason.

    Compares the recomputed projection against the bundle's *recorded projection*, not against its
    fingerprint alone. Rehashing an edited manifest therefore cannot bypass the check: the values
    themselves are compared, and a manifest whose fingerprint disagrees with its own projection is
    reported as tampered.
    """
    recorded = (bundle_manifest.get("prepared_system") or {})
    proj = recorded.get("projection")
    stored_fp = recorded.get("fingerprint")
    if not isinstance(proj, dict) or not stored_fp:
        return (
            "this bundle records no prepared-system fingerprint, so a experiment override cannot "
            "be checked against it. It was built by an older version; re-run `prepare`, or launch "
            "without --experiment to use the bundle's own experiment."
        )
    recomputed = sha256_text(canonical_json(proj))
    if recomputed != stored_fp:
        return (
            "this bundle's recorded prepared-system projection does not hash to its recorded "
            f"fingerprint (recorded {stored_fp}, recomputed {recomputed}). The bundle manifest has "
            "been edited; do not run from it."
        )
    diffs = diff_projections(proj, build_projection(proposed_cfg))
    return describe_incompatibility(diffs) if diffs else None
