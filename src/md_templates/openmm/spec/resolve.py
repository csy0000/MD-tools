"""Loading, profile resolution, and source attribution.

Resolution precedence is exactly three layers, and the resolved output records which layer supplied
every final value:

    1. the versioned profile
    2. the user's input document
    3. explicit CLI overrides (`--set dotted.path=value`)

"Where did this number come from?" is the question a reproducibility argument turns on, so it is
answered by data rather than by rereading files in the right order.

Format is a front end. `.yaml`, `.yml` and `.json` go through the same loader registry into the
same typed model, so the same configuration in either format produces byte-identical canonical
JSON and identical hashes. A future Amber-style namelist reader registers here and adapts INTO the
model; it must never become a second set of semantics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .canonical import hashes
from .models import SimulationSpec

__all__ = ["load_document", "load_profile", "list_profiles", "resolve_spec", "ResolutionError",
           "apply_overrides", "LOADERS", "register_loader", "PROFILE_DIR"]

PROFILE_DIR = Path(__file__).parent / "profiles"


class ResolutionError(ValueError):
    """A document could not be read, or a profile could not be selected."""


# ---------------------------------------------------------------------------------------------
# the loader registry -- format is a front end
# ---------------------------------------------------------------------------------------------

def _load_yaml(text: str) -> dict:
    import yaml

    return yaml.safe_load(text) or {}


def _load_json(text: str) -> dict:
    return json.loads(text)


LOADERS: dict[str, Callable[[str], dict]] = {
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
    ".json": _load_json,
}


def register_loader(suffix: str, loader: Callable[[str], dict]) -> None:
    """Register a front end for a file suffix.

    The contract for any future adapter -- an Amber-style namelist reader is the motivating case --
    is exactly this: parse the foreign syntax, return plain data shaped like this model, and reject
    every key it cannot map. It must not add defaults, must not reinterpret values, and must not
    grow its own validation. Unsupported keys FAIL; silently dropping them would run a calculation
    the author did not ask for.
    """
    LOADERS[suffix.lower()] = loader


def load_document(path: Path) -> dict:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ResolutionError(
            f"{path}: no loader for suffix {path.suffix!r}. Supported: {sorted(LOADERS)}."
        )
    try:
        data = loader(path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 - reported with its own type
        raise ResolutionError(f"{path}: {type(exc).__name__}: {exc}") from None
    if not isinstance(data, dict):
        raise ResolutionError(f"{path}: expected a mapping at the top level, got "
                              f"{type(data).__name__}")
    return data


# ---------------------------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------------------------

def list_profiles() -> list[dict]:
    out = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["_path"] = str(path)
        out.append(doc)
    return out


def load_profile(profile_id: str) -> dict:
    path = PROFILE_DIR / f"{profile_id}.json"
    if not path.is_file():
        available = sorted(p["profile_id"] for p in list_profiles())
        raise ResolutionError(f"unknown profile {profile_id!r}. Available: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_profile(route: str, method: str) -> dict:
    """The `default` alias: choose by DECLARED route and method, never by file contents.

    Only a profile marked `is_default` is eligible. Without that, selection was "first match in
    sorted order", which silently chose the CPU smoke profile -- picoseconds of unvalidated
    settings -- for any ligand REST2 document. Ambiguity is refused rather than broken by sorting.
    """
    candidates = [d for d in list_profiles()
                  if d.get("route") == route and d.get("method") == method and d.get("is_default")]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        available = sorted(d["profile_id"] for d in list_profiles())
        raise ResolutionError(
            f"no default profile for route {route!r} with method {method!r}. Name one explicitly "
            f"with `profile:`. Available: {available}"
        )
    raise ResolutionError(
        f"{len(candidates)} profiles claim to be the default for route {route!r} and method "
        f"{method!r}: {[d['profile_id'] for d in candidates]}. Exactly one may."
    )


# ---------------------------------------------------------------------------------------------
# merging with attribution
# ---------------------------------------------------------------------------------------------

def _merge(base: dict, overlay: dict, sources: dict, layer: str, prefix: str = "") -> dict:
    """Recursive overlay that records which layer supplied each leaf."""
    out = dict(base)
    for key, value in overlay.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value, sources, layer, prefix=f"{dotted}.")
        elif isinstance(value, dict):
            # a whole subtree the profile did not define (e.g. `system`). Attribute every LEAF,
            # not just the subtree: "where did system.route come from" must have an answer.
            out[key] = value
            _flatten_sources(value, layer, sources, prefix=f"{dotted}.")
        else:
            out[key] = value
            sources[dotted] = layer
    return out


def _flatten_sources(data: Any, layer: str, sources: dict, prefix: str = "") -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            _flatten_sources(value, layer, sources, prefix=f"{prefix}{key}.")
    else:
        sources[prefix.rstrip(".")] = layer


def _coerce(text: str) -> Any:
    """Type a `--set` value without eval: JSON first, then plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(data: dict, overrides: list[str], sources: dict) -> dict:
    """Apply `dotted.path=value` overrides, validated later by the same model as everything else."""
    out = json.loads(json.dumps(data))
    for item in overrides or []:
        if "=" not in item:
            raise ResolutionError(f"--set {item!r} must be of the form dotted.path=value")
        dotted, raw = item.split("=", 1)
        dotted = dotted.strip()
        node = out
        parts = dotted.split(".")
        for key in parts[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
        sources[dotted] = "cli"
    return out


#: Inputs retired when segment count left the scientific configuration. Each maps to the precise
#: instruction a user needs, because "unknown field" alone would send them looking for a typo.
_RETIRED_PRODUCTION_KEYS = {
    "n_chunks": (
        "protocol.production.n_chunks has been REMOVED from the scientific configuration. How many "
        "segments to run is an execution choice, not a property of the calculation: putting it in "
        "the JSON meant that asking for a longer run changed the configuration hash and made an "
        "extended run look like a different calculation.\n"
        "  Migration: delete `n_chunks`, keep the segment length as "
        "`protocol.production.duration_per_segment`, and control repetition from the driver script "
        "(for example NUMBER_OF_SEGMENTS in the generated Bash). Completed segments are recorded in "
        "the run manifest."
    ),
    "chunk_ns": (
        "protocol.production.chunk_ns has been RENAMED and re-typed. Segment length is now "
        "`protocol.production.duration_per_segment` and carries explicit units.\n"
        "  Migration: replace `\"chunk_ns\": 5.0` with `\"duration_per_segment\": \"5 ns\"`."
    ),
    "chunk": (
        "protocol.production.chunk has been RENAMED to "
        "`protocol.production.duration_per_segment`.\n"
        "  Migration: rename the field; the value and units are unchanged."
    ),
    "scale_factors": (
        "protocol.production.scale_factors has been REPLACED by the tau parameterisation. tau is "
        "now the source parameter and s is derived as s = (1 - tau)^2, with the solute-environment "
        "coupling sqrt(s) = 1 - tau. The Hamiltonian is unchanged; only the way the ladder is "
        "written has changed.\n"
        "  Migration: replace the explicit s list with "
        "`\"tau_ladder\": {\"minimum\": 0.0, \"maximum\": 0.5, \"count\": N, "
        "\"interpolation\": \"linear\"}`. For an existing ladder, tau = 1 - sqrt(s) for each rung."
    ),
    "number_of_exchanges_per_segment": (
        "protocol.production.number_of_exchanges_per_segment has been REPLACED. The exchange "
        "schedule is now stated as a count AND an interval, and the segment length is derived "
        "from their product:\n"
        "      duration_per_segment = n_exchange_per_segment * exchange_interval\n"
        "  A product is exact; the previous form made the interval a quotient of the segment "
        "duration, which could fail to divide.\n"
        "  Migration: replace it with "
        "`\"exchange\": {\"n_exchange_per_segment\": 1000, \"exchange_interval\": \"5 ps\"}`, "
        "and DELETE protocol.production.duration_per_segment -- for REST2 it is derived, not an "
        "input."
    ),
}


def _refuse_retired_chunk_inputs(document: dict) -> None:
    """Reject retired production inputs with the exact migration, never a silent reinterpretation.

    Silently mapping an old `n_chunks` onto the new contract would be the worst option available:
    the run would proceed under a plan the author did not write and believes they did.
    """
    production = ((document.get("protocol") or {}).get("production") or {})
    if not isinstance(production, dict):
        return
    found = [key for key in _RETIRED_PRODUCTION_KEYS if key in production]

    # The exchange block is now a count only; the interval is derived from the segment duration.
    # Both retired fields are named individually so the message can say what replaces each.
    exchange = production.get("exchange")
    if isinstance(exchange, dict):
        retired_exchange = [k for k in ("n_exchange_per_segment", "exchange_interval")
                            if k in exchange]
        if retired_exchange:
            raise ResolutionError(
                "protocol.production.exchange states "
                f"{', '.join(sorted(retired_exchange))}, which "
                f"{'have' if len(retired_exchange) > 1 else 'has'} been REPLACED.\n"
                "  A segment is now stated by its DURATION and the number of exchanges in it; the "
                "interval is derived:\n"
                "      steps_per_segment  = duration_per_segment / timestep\n"
                "      steps_per_exchange = steps_per_segment / number_of_exchanges_per_segment\n"
                "      exchange_interval  = steps_per_exchange * timestep\n"
                "  Both divisions must be exact and are refused otherwise, never rounded.\n"
                "  Migration: replace\n"
                '      "exchange": {"n_exchange_per_segment": 1000, "exchange_interval": "5 ps"}\n'
                "  with\n"
                '      "duration_per_segment": "5 ns",\n'
                '      "exchange": {"number_of_exchanges_per_segment": 1000}\n'
                "  which is the same schedule: 1,250,000 steps at 4 fs, 1250 steps per round."
            )

    if not found:
        return
    detail = "\n\n".join(_RETIRED_PRODUCTION_KEYS[key] for key in found)
    raise ResolutionError(
        f"This configuration uses {len(found)} retired production field(s): "
        f"{', '.join(sorted(found))}.\n\n{detail}"
    )


def resolve_spec(document: dict, *, overrides: Optional[list[str]] = None,
                 profile_id: Optional[str] = None) -> dict:
    """Resolve one document into a validated `SimulationSpec` plus provenance.

    Returns `{"spec", "profile", "sources", "hashes", "resolved"}`. `sources` maps every dotted
    field to the layer that supplied its final value.
    """
    document = json.loads(json.dumps(document))          # never mutate the caller's data
    _refuse_retired_chunk_inputs(document)
    system = document.get("system")
    if not isinstance(system, dict) or "route" not in system:
        raise ResolutionError(
            "system.route is required and is never inferred from file contents: it decides whether "
            "this is a peptide or a small-molecule parameterisation."
        )
    declared_profile = profile_id or document.pop("profile", None)
    method = (document.get("protocol") or {}).get("production", {}).get("method")
    if method is None:
        raise ResolutionError(
            "protocol.production.method is required ('md' or 'rest2'); it selects which settings "
            "are meaningful and is not guessed."
        )

    if declared_profile in (None, "default"):
        profile = select_profile(system["route"], method)
    else:
        profile = load_profile(str(declared_profile))
        if profile.get("method") != method:
            raise ResolutionError(
                f"profile {profile['profile_id']!r} is for method {profile.get('method')!r} but "
                f"this document declares {method!r}"
            )
        if profile.get("route") != system["route"]:
            raise ResolutionError(
                f"profile {profile['profile_id']!r} is for route {profile.get('route')!r} but "
                f"this document declares {system['route']!r}"
            )

    sources: dict[str, str] = {}
    merged = json.loads(json.dumps(profile["defaults"]))
    _flatten_sources(merged, f"profile:{profile['profile_id']}", sources)
    merged = _merge(merged, document, sources, "document")
    merged = apply_overrides(merged, overrides or [], sources)

    spec = SimulationSpec.model_validate(merged)
    return {
        "spec": spec,
        "profile": {"profile_id": profile["profile_id"],
                    "profile_schema_version": profile["profile_schema_version"],
                    "description": profile.get("description"),
                    "sha256": _profile_hash(profile)},
        "sources": sources,
        "hashes": hashes(spec),
        "resolved": merged,
    }


def _profile_hash(profile: dict) -> str:
    from .canonical import sha256_of

    return sha256_of({k: v for k, v in profile.items() if k != "_path"})
