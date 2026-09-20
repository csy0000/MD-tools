"""The public surface of `md_tools.alchemy`, in ONE place.

A guard, not a description, and the same shape as `tests/public_commands.py`: a name here is a
name other sessions, the CLI or a generated script may import, so removing one is a deliberate act
with a reviewer attached.

WHY IT EXISTS. On 2026-09-20 a rewrite of `combine_repeats` replaced everything from its own `def`
to the end of `campaign.py`, deleting `matched_leg_report` and `relative_hydration_from_legs` with
it. Nothing failed: the suite does not import them, and the loss surfaced only because the next
lane written that afternoon happened to import one and could not. Every test still passed while
two public functions had ceased to exist.

Signatures are listed for the entry points another session calls, because a parameter that
silently disappears is the same defect wearing a smaller hat: `run_window(prepared=...)` quietly
becoming an unknown keyword would send a caller back through the whole preflight instead of
consuming the one it was handed.
"""
from __future__ import annotations

#: module -> the names it exports.
ALCHEMY_SURFACE = {
    "md_tools.alchemy.paths": (
        "AlchemicalPath", "Knot", "PathError", "Segment", "PATH_SCHEMA", "INTERPOLATION",
        "S_TOLERANCE", "check_component_name", "linear_path", "staged_path"),
    "md_tools.alchemy.samples": (
        "SampleSet", "SampleRecordError", "ThermodynamicState", "SAMPLES_SCHEMA", "UNITS",
        "BOLTZMANN_KJ_MOL_K", "BAR_NM3_TO_KJ_MOL", "KJ_PER_KCAL", "concatenate", "kt_kj_mol",
        "window_states"),
    "md_tools.alchemy.estimators": (
        "Estimate", "EstimatorError", "MissingAnalysisDependency", "OVERLAP_WARNING",
        "EXP_ESS_WARNING", "GATE_ABSOLUTE_KCAL_MOL", "GATE_SIGMAS",
        "GATE_MAX_COMBINED_SIGMA_KCAL_MOL", "agreement_gate", "analyze", "bar_estimate",
        "bar_from_work", "bar_path", "decorrelate", "exp_estimate", "mbar_estimate",
        "statistical_inefficiency", "subsample_indices", "ti_estimate", "ti_weights"),
    "md_tools.alchemy.restraints": (
        "BoreschRestraint", "RestraintError", "BORESCH_SCHEMA", "RESTRAINT_PARAMETER",
        "STANDARD_VOLUME_NM3", "STANDARD_CONCENTRATION", "MIN_ANCHOR_ANGLE_DEG"),
    "md_tools.alchemy.cycles": (
        "Leg", "CycleError", "CYCLE_SCHEMA", "ENVIRONMENTS", "COUPLED", "DECOUPLED",
        "RESTRAINED", "UNRESTRAINED", "absolute_binding", "absolute_hydration",
        "relative_binding", "relative_hydration", "reversed_leg"),
    "md_tools.alchemy.windows": (
        "ComposedHamiltonian", "CrossStateEvaluator", "ParametricHamiltonian",
        "SampleStreamReporter", "WindowError", "WindowSettings", "WINDOW_SCHEMA",
        "COMPLETION_SCHEMA", "RESTRAINT_FORCE_GROUP", "SOLVATIONS", "SELF_CHECK_ABS_KJ_MOL",
        "SELF_CHECK_REL", "check_hamiltonian_matches_path", "read_window_samples", "run_window",
        "self_check_relative", "stream_columns", "validate_stream_prefix", "verify_window",
        "window_identity", "window_paths"),
    "md_tools.alchemy.campaign": (
        "LEG_SCHEMA", "analyze_leg", "combine_repeats", "matched_leg_report", "prepare_leg",
        "read_leg", "relative_hydration_from_legs", "run_leg"),
}

#: entry point -> parameters another session passes by name.
ALCHEMY_SIGNATURES = {
    "md_tools.alchemy.windows.run_window": (
        "topology", "system", "hamiltonian", "path", "states", "window_id", "out_dir",
        "settings", "coordinates", "cpu", "device", "machine_config", "overwrite", "restraint",
        "prepared", "check", "solvation", "vacuum_leg"),
    "md_tools.alchemy.campaign.prepare_leg": (
        "directory", "plan", "hamiltonian", "path", "s_values", "temperature_k", "pressure_bar",
        "environment", "endpoint_a", "endpoint_b", "scheme"),
    "md_tools.alchemy.campaign.run_leg": (
        "directory", "hamiltonian", "settings", "windows", "repeat"),
    "md_tools.alchemy.campaign.analyze_leg": ("directory", "repeat", "estimator", "name"),
    "md_tools.alchemy.estimators.agreement_gate": (
        "value_kcal", "sigma_kcal", "reference_kcal", "reference_sigma_kcal",
        "integration_sigma_kcal", "numerical_floor_kcal"),
}
