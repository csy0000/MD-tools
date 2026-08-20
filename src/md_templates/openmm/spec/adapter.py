"""The single bridge from the canonical model into the runtime configuration.

There is one configuration engine. The canonical `SimulationSpec` is the source of truth; this
module projects it into the flat dictionary the runners already consume, and the legacy
system+experiment manifests reach that same dictionary by being migrated into the canonical model
first. That ordering is the point -- a second engine would mean two places where a scientific
default can live and disagree.

    manifests --(migrate)--> SimulationSpec --(here)--> runtime cfg --> build/run
    YAML/JSON ---(load)-----^

Quantities are converted once, here, into the unit each runtime key's NAME declares
(`timestep_fs`, `chunk_ns`, `padding_nm`). Those names are the runtime's convention and are not
changed in this pass; the canonical model is what carries units explicitly.
"""
from __future__ import annotations

import copy
from typing import Any

from .. import segments
from .. import tau as tau_module
from .models import SimulationSpec

__all__ = ["spec_to_runtime_cfg"]


def spec_to_runtime_cfg(spec: SimulationSpec, *, base: dict | None = None) -> dict[str, Any]:
    """Project a validated spec onto the runtime configuration tree.

    `base` is the package DEFAULTS, used only for keys the canonical model does not yet model --
    equilibration staging detail and reporter plumbing. Every scientific value that the model does
    carry overwrites it, so the model wins wherever the two overlap.
    """
    from ..config import DEFAULTS

    cfg = copy.deepcopy(base if base is not None else DEFAULTS)

    system, build, protocol, execution = spec.system, spec.build, spec.protocol, spec.execution
    prod = protocol.production

    cfg["system"]["slug"] = system.system_id
    cfg["system"]["solute_kind"] = "ligand" if system.route == "smiles" else "peptide"
    cfg["system"]["require_input_route"] = system.route

    ff = build.forcefield
    cfg["forcefield"]["ligand"] = ff.small_molecule
    cfg["forcefield"]["ligand_charge_method"] = ff.charge_method
    cfg["forcefield"]["protein"] = ff.protein
    cfg["forcefield"]["water"] = ff.water

    sol = build.solvation
    if sol is not None:
        cfg["solvation"].update({
            "mode": "explicit",
            "water_model": sol.water_model,
            "box_shape": sol.box_shape,
            "padding_nm": sol.padding.value,
            "padding_semantics": sol.padding_semantics,
            "ionic_strength_molar": sol.ionic_strength_molar,
            "positive_ion": sol.positive_ion,
            "negative_ion": sol.negative_ion,
            "neutralize": sol.neutralize,
            "cutoff_fit_policy": sol.cutoff_fit_policy,
        })
    else:
        # Implicit: the runtime solvation block carries the GB model and nothing about water. It is
        # REPLACED rather than updated, so no explicit default survives into a configuration that
        # has no water to apply it to.
        cfg["solvation"] = {
            "mode": "implicit",
            "implicit_model": build.implicit.model,
            "radii": build.implicit.radii,
            "salt_concentration_molar": 0.0,
        }

    nb = build.nonbonded
    cfg["system_build"].update({
        "nonbonded_method": nb.method,
        "nonbonded_cutoff_nm": (nb.cutoff.value if nb.cutoff else None),
        "switch_distance_nm": (nb.switch_distance.value if nb.switch_distance else None),
        "use_dispersion_correction": nb.use_dispersion_correction,
        "ewald_error_tolerance": nb.ewald_error_tolerance,
        "minimum_image_margin_nm": (nb.minimum_image_margin.value
                                    if nb.minimum_image_margin else None),
        "constraints": build.constraints,
        "rigid_water": build.rigid_water,
        "hydrogen_mass_amu": (build.hydrogen_mass.value if build.hydrogen_mass else None),
        "hmr_scope": build.hmr_scope,
        "remove_cm_motion": build.remove_cm_motion,
    })

    integ = protocol.integrator
    cfg["integrator"].update({
        "kind": integ.kind,
        "timestep_fs": integ.timestep.value * 1000.0,          # ps -> fs, the runtime's unit
        "temperature_k": integ.temperature.value,
        "friction_per_ps": integ.friction.value,
    })
    cfg["equilibration"]["protocol"] = protocol.equilibration.protocol
    cfg["equilibration"]["minimize_max_iterations"] = protocol.equilibration.minimize_max_iterations
    eq = protocol.equilibration
    if eq.npt_free is not None:
        cfg["equilibration"]["npt_free_ps"] = eq.npt_free.value
    else:
        # Implicit solvent has no NPT stage at all. Leaving a stale default here would describe an
        # equilibration the generated project does not contain.
        cfg["equilibration"].pop("npt_free_ps", None)
    for spec_field, runtime_key, scale in (("timestep", "timestep_fs", 1000.0),
                                           ("nvt", "nvt_ps", 1.0),
                                           ("npt", "npt_ps", 1.0),
                                           ("box_average_last", "box_average_last_ps", 1.0)):
        q = getattr(eq, spec_field)
        if q is not None:
            cfg["equilibration"][runtime_key] = q.value * scale
    if eq.seed is not None:
        cfg["equilibration"]["seed"] = eq.seed

    rep = execution.reporting
    cfg["production"]["report"].update({
        "all_atom_ps": rep.all_atom.value,
        "solute_ps": rep.solute.value,
        "state_ps": rep.state.value,
        "checkpoint_ps": rep.checkpoint.value,
    })
    cfg["production"]["platform"] = execution.platform
    cfg["production"]["device_index"] = execution.device
    cfg["production"]["precision"] = execution.precision

    # ONE segment per invocation. The canonical model no longer carries a count: the driver script
    # decides how many segments to request and the run manifest records how many committed, so the
    # runtime's `n_chunks` here means "this invocation runs one segment" and is no longer a
    # scientific input.
    if prod.method == "md":
        steps_per_segment = segments.steps_for_duration(
            prod.duration_per_segment.value, integ.timestep.value,
            duration_source=prod.duration_per_segment.source,
            timestep_source=integ.timestep.source,
            duration_label="production.duration_per_segment",
        )

        cfg["production"]["md"].update({
            "n_chunks": 1,
            "chunk_ns": prod.duration_per_segment.value / 1000.0,   # ps -> ns
            "scale_factor": prod.scale_factor,
            "seed": prod.seed,
        })
    else:
        # tau is the source parameter; `s` reaches the System builder unchanged. The exchange
        # interval is DERIVED from the segment length and the exchange count rather than stated a
        # second time, so the two can never disagree.
        plan = segments.plan_segment_from_duration_and_exchanges(
            prod.duration_per_segment.value,
            prod.exchange.number_of_exchanges_per_segment,
            integ.timestep.value,
            duration_source=prod.duration_per_segment.source,
            timestep_source=integ.timestep.source,
        )
        exchange_interval_ps = plan.steps_per_exchange * integ.timestep.value
        tau_values = prod.tau_ladder.tau_values()

        cfg["production"]["remd"].update({
            "n_chunks": 1,
            "chunk_ns": prod.duration_per_segment.value / 1000.0,
            "scale_factors": list(prod.scale_factors()),
            "exchange_interval_ps": exchange_interval_ps,
            "equilibration_ps": (prod.relaxation.value if prod.relaxation is not None else 0.0),
            "seed": prod.seed,
            "steps_per_exchange": plan.steps_per_exchange,
            "number_of_exchanges_per_segment": plan.number_of_exchanges_per_segment,
        })
        cfg["rest2"].update({
            "omega_exclusion": prod.omega_exclusion.enabled,
            "proline_like_residues": list(prod.omega_exclusion.proline_like_residues),
            "max_proline_ring_size": prod.omega_exclusion.max_proline_ring_size,
        })
        # tau and its derived columns are recorded so a run directory can be read without
        # recomputing the ladder, and so a changed ladder shows up in a diff.
        cfg["rest2"]["tau_ladder"] = {
            "minimum": prod.tau_ladder.minimum,
            "maximum": prod.tau_ladder.maximum,
            "count": prod.tau_ladder.count,
            "interpolation": prod.tau_ladder.interpolation,
            "tau_values": tau_values,
            "derived": tau_module.ladder_diagnostics(tau_values, integ.temperature.value),
        }
        steps_per_segment = plan.steps_per_segment
        cfg["rest2"]["enhanced_region"] = {
            "type": prod.enhanced_region.type,
            "atom_indices": (list(prod.enhanced_region.atom_indices)
                             if prod.enhanced_region.atom_indices else None),
            "amber_mask": prod.enhanced_region.amber_mask,
        }

    cfg["production"]["steps_per_segment"] = steps_per_segment

    # Seeds come from the canonical randomness block, which reproduces the legacy derivation
    # (master + stage offset) exactly. The runtime tree is filled from the RESOLVED values, so the
    # legacy `_resolve_seeds` has nothing left to derive.
    seeds = spec.randomness.resolve()
    cfg["run"]["seed"] = spec.randomness.master_seed
    cfg["structure"]["etkdg"]["seed"] = seeds["structure"]
    cfg["equilibration"]["seed"] = seeds["equilibration"]
    cfg["production"]["md"]["seed"] = seeds["md"]
    cfg["production"]["remd"]["seed"] = seeds["rest2"]
    # A production seed stated on the production block is the same setting by another name; the
    # randomness block is canonical, so it is folded in rather than allowed to disagree.
    if prod.seed is not None:
        key = "md" if prod.method == "md" else "remd"
        cfg["production"][key]["seed"] = prod.seed

    # Where every value came from, so a run directory can answer it without the source document.
    cfg["_canonical"] = {
        "method": prod.method,
        "route": system.route,
        "master_seed": spec.randomness.master_seed,
        "stage_seeds": seeds,
        "stage_seed_sources": spec.randomness.sources(),
        "schema_versions": {
            "system": system.schema_version, "build": build.schema_version,
            "protocol": protocol.schema_version, "execution": execution.schema_version,
        },
    }
    return cfg
