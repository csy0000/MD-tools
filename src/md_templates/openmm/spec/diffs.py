"""Semantic diff and field explanation.

A difference between two configurations is not merely a changed value; what matters is what the
change *costs*. Each is classified into one of four consequences:

    bundle-defining     the prepared System is wrong -> a new bundle is required
    continuity-defining the physical run is different -> a new run is required
    extension-only      more work of the same kind -> a resume may proceed
    execution-only      performance or output volume -> nothing scientific changes

Classifying by consequence rather than by section is the point: a reader wants to know whether they
must re-prepare, restart, or simply carry on.
"""
from __future__ import annotations

from typing import Any

from .canonical import EXTENSION_ONLY, dump_model, to_plain
from .models import SimulationSpec

__all__ = ["classify", "diff_specs", "explain_field", "CATEGORIES"]

CATEGORIES = ("bundle-defining", "continuity-defining", "extension-only",
              "execution-only", "seed-label")


def classify(dotted: str) -> str:
    if dotted in EXTENSION_ONLY:
        return "extension-only"
    if dotted.startswith("randomness."):
        # The master seed is a label; what changes physics is the RESOLVED stage seed. Structure
        # and equilibration seeds shape the prepared artifacts, the production seed shapes the run,
        # and a master seed whose stage seeds are all pinned changes neither.
        if dotted.endswith(("stage_seeds.structure", "stage_seeds.equilibration")):
            return "bundle-defining"
        if dotted.endswith(("stage_seeds.md", "stage_seeds.rest2")):
            return "continuity-defining"
        return "seed-label"
    if dotted.startswith("execution."):
        return "execution-only"
    if dotted.startswith(("system.", "build.")):
        return "bundle-defining"
    if dotted.startswith("protocol."):
        return "continuity-defining"
    return "continuity-defining"


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = data
    return out


def diff_specs(a: SimulationSpec, b: SimulationSpec) -> list[dict]:
    """Differences between two resolved configurations, each with its consequence."""
    fa = _flatten(to_plain(dump_model(a)))
    fb = _flatten(to_plain(dump_model(b)))
    rows = []
    for key in sorted(set(fa) | set(fb)):
        left, right = fa.get(key, "<absent>"), fb.get(key, "<absent>")
        if left != right:
            rows.append({"field": key, "a": left, "b": right, "consequence": classify(key)})
    return rows


#: What a field means, beyond its type. Only entries that carry a real scientific consequence are
#: listed; anything absent falls back to the model's own type and constraints.
_EFFECTS: dict[str, str] = {
    "protocol.integrator.timestep":
        "integration step. Larger steps need hydrogen-mass repartitioning and constraints; the "
        "2 fs / 4 fs equivalence gate is NOT closed in this template.",
    "protocol.integrator.temperature": "thermostat set point for the whole system.",
    "protocol.production.scale_factors":
        "the REST2 ladder. Rung spacing sets exchange acceptance; changing it changes the run.",
    "protocol.production.omega_exclusion":
        "leaves ordinary amide omega torsions unscaled by REST2. Enabled by default. Changing it "
        "changes the Hamiltonian and invalidates a bundle.",
    "protocol.production.duration_per_segment":
        "length of ONE production segment; also the restart granularity. How many segments run is "
        "an execution choice made by the driver, not a field here.",
    "protocol.production.tau_ladder":
        "the REST2 ladder in tau. s = (1 - tau)^2 and the solute-environment coupling is "
        "sqrt(s) = 1 - tau. Rung spacing sets exchange acceptance; changing it changes the run.",
    "protocol.production.exchange":
        "how many exchange rounds one segment contains. The interval is derived from it.",
    "build.solvation.padding": "minimum solvent between the solute and its periodic image.",
    "build.solvation.ionic_strength_molar":
        "requested SALT concentration. Neutralising counterions are counted separately.",
    "build.nonbonded.cutoff": "real-space cutoff; the box must exceed twice this.",
    "build.hydrogen_mass": "hydrogen mass after repartitioning; enables a longer timestep.",
    "build.constraints": "which bonds are constrained; changes the accessible timestep.",
    "execution.platform": "OpenMM platform. Performance only; not part of any compatibility hash.",
    "execution.precision": "CUDA/OpenCL precision. Single silently changes energies; mixed is the "
                           "default here.",
    "randomness.master_seed":
        "the seed stage seeds DERIVE from (master + offset: structure 0, equilibration 1, md 2, "
        "rest2 3). Changing it normally changes every derived seed and therefore both physical "
        "hashes -- but if every affected stage seed is explicitly pinned to its old value, the "
        "hashes are unchanged, because what is hashed is the resolved stage seed, not the label.",
}


def explain_field(spec: SimulationSpec, dotted: str, sources: dict) -> dict:
    """Type, unit, where the value came from, allowed values, and what it does."""
    flat = _flatten(to_plain(dump_model(spec)))
    if dotted not in flat and f"{dotted}.value" not in flat:
        near = [k for k in flat if dotted.split(".")[-1] in k]
        raise KeyError(f"unknown field {dotted!r}" + (f"; did you mean {near[:5]}?" if near else ""))

    value = flat.get(dotted, flat.get(f"{dotted}.value"))
    unit = flat.get(f"{dotted}.unit")
    model, field = _locate(spec, dotted)
    allowed = None
    if model is not None and field in getattr(model, "model_fields", {}):
        info = model.model_fields[field]
        annotation = info.annotation
        allowed = getattr(annotation, "__args__", None)
        allowed = [a for a in allowed] if allowed and all(isinstance(a, str) for a in allowed) \
            else None
    return {
        "field": dotted,
        "value": value,
        "unit": unit,
        "source": sources.get(dotted, sources.get(f"{dotted}.value", "unknown")),
        "consequence": classify(dotted),
        "allowed_values": allowed,
        "effect": _EFFECTS.get(dotted, "see the model definition for type and constraints"),
    }


def _locate(spec: SimulationSpec, dotted: str):
    node: Any = spec
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = getattr(node, key, None)
        if node is None:
            return None, parts[-1]
    return node, parts[-1]
