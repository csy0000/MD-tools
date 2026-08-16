"""Canonical serialisation, projections, and the three hashes that decide compatibility.

A hash here is a promise about what may change without invalidating something else, so each is
computed from an explicit projection rather than from the whole document:

    system/build hash        changing it means the prepared bundle is wrong -> NEW BUNDLE
    protocol/continuity hash changing it means the physical run is different -> NEW RUN
    execution projection     platform, device, reporting -- recorded, never compared

Canonicalisation is what makes those hashes comparable across file formats and machines: sorted
keys, no insignificant whitespace, quantities as their normalised value plus unit convention, and
floats formatted by `repr` so the same double never serialises two ways.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import SimulationSpec
from .units import Quantity

__all__ = ["canonical_json", "sha256_of", "system_build_projection", "protocol_projection",
           "execution_projection", "hashes", "to_plain"]

#: Fields that are execution-only: they may change performance or output volume, never the
#: Hamiltonian, so they are recorded in provenance and excluded from every compatibility hash.
EXECUTION_ONLY = ("execution",)

#: Extension-only: more work of the same kind. Excluded from the continuity hash so a run can be
#: extended, which is the entire point of resuming.
EXTENSION_ONLY = ("protocol.production.n_chunks",)


def to_plain(value: Any) -> Any:
    """Plain JSON-able data, with quantities in canonical form.

    A Quantity serialises to BOTH its normalised value and its unit convention, so the canonical
    document states what the number means instead of leaving it to a reader's assumption.
    """
    if isinstance(value, Quantity):
        return {"value": value.value, "unit": value.unit}
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, tight separators, one trailing newline."""
    return json.dumps(to_plain(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def dump_model(model: Any) -> Any:
    """Plain data that PRESERVES quantities.

    `model_dump` flattens a NamedTuple to a bare list, so a canonical document would carry
    `[0.004, "ps", "time", "4 fs"]` -- ordering-dependent, and unreadable as a quantity by anything
    downstream. Walking the model instead keeps Quantity objects intact until `to_plain` renders
    them as an explicit value/unit pair.
    """
    from pydantic import BaseModel

    if isinstance(model, Quantity):
        return model
    if isinstance(model, BaseModel):
        return {name: dump_model(getattr(model, name)) for name in type(model).model_fields}
    if isinstance(model, dict):
        return {k: dump_model(v) for k, v in model.items()}
    if isinstance(model, (list, tuple)):
        return [dump_model(v) for v in model]
    return model


def _dump(spec: SimulationSpec) -> dict:
    return to_plain(dump_model(spec))


def system_build_projection(spec: SimulationSpec) -> dict:
    """Everything that decides the prepared System. A change here needs a new bundle."""
    data = _dump(spec)
    return {"system": data["system"], "build": data["build"]}


def protocol_projection(spec: SimulationSpec) -> dict:
    """Everything that decides the physical run, minus extension-only fields.

    `n_chunks` is removed deliberately: asking for more chunks extends a run rather than redefining
    it, and including it here would make every continuation look incompatible.
    """
    data = _dump(spec)
    protocol = json.loads(json.dumps(data["protocol"]))
    protocol.get("production", {}).pop("n_chunks", None)
    return {"protocol": protocol}


def execution_projection(spec: SimulationSpec) -> dict:
    """Machine choices. Recorded in provenance; never part of a compatibility decision."""
    return {"execution": _dump(spec)["execution"]}


def hashes(spec: SimulationSpec) -> dict[str, str]:
    return {
        "system_build_sha256": sha256_of(system_build_projection(spec)),
        "protocol_sha256": sha256_of(protocol_projection(spec)),
        "execution_sha256": sha256_of(execution_projection(spec)),
    }
