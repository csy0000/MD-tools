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
           "prepared_state_projection", "protocol_at_prepare_projection",
           "execution_projection", "hashes", "to_plain", "dump_model"]

#: Fields that are execution-only: they may change performance or output volume, never the
#: Hamiltonian, so they are recorded in provenance and excluded from every compatibility hash.
EXECUTION_ONLY = ("execution",)

#: Extension-only: more work of the same kind. Excluded from the continuity hash so a run can be
#: extended, which is the entire point of resuming.
#: Empty by construction, and that is the point. The only field that was ever extension-only
#: was `protocol.production.n_chunks`, and segment count is no longer a scientific input at
#: all -- the driver script decides how many segments to request and the run manifest records
#: how many committed. Nothing remains in the canonical configuration that a user can change
#: and still resume: every remaining field is bundle-, continuity-, or execution-defining.
EXTENSION_ONLY: tuple[str, ...] = ()


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
    """Molecular identity and parameterisation: what the System is made of."""
    data = _dump(spec)
    return {"system": data["system"], "build": data["build"]}


def prepared_state_projection(spec: SimulationSpec) -> dict:
    """Everything that decides the prepared ARTIFACTS -- system.xml AND equilibrated_state.xml.

    Wider than system/build, deliberately. A field belongs here if changing it would make the
    stored starting state the wrong one, regardless of which model section it happens to live in:
    the equilibration protocol and its duration, the integrator used to reach that state, and the
    structure/equilibration seeds all qualify. Classifying by section rather than by consequence
    would leave a bundle reusable across an equilibration change that produced a different state.
    """
    data = _dump(spec)
    seeds = spec.randomness.resolve()
    return {
        "system": data["system"],
        "build": data["build"],
        "equilibration": data["protocol"]["equilibration"],
        "integrator": data["protocol"]["integrator"],
        # the seeds that shaped the stored artifacts; the production seed does not belong here
        "seeds": {"structure": seeds["structure"], "equilibration": seeds["equilibration"]},
    }


def protocol_at_prepare_projection(spec: SimulationSpec) -> dict:
    """The protocol as it stood when the bundle was prepared, recorded for comparison later."""
    return {"protocol": _dump(spec)["protocol"]}


def protocol_projection(spec: SimulationSpec) -> dict:
    """Everything that decides the physical run.

    Nothing is stripped any more. Segment count used to be removed here so that a continuation did
    not look incompatible; it is no longer a configuration field at all, so the projection is now
    simply the whole protocol. `duration_per_segment` IS hashed: it is the restart granularity, and
    a resume that silently changed it would produce segments of two different lengths in one run.
    """
    data = _dump(spec)
    protocol = json.loads(json.dumps(data["protocol"]))
    # The resolved PRODUCTION seed is continuity-defining: the same state advanced under a
    # different seed is a different trajectory. The master seed is not hashed as a label.
    seeds = spec.randomness.resolve()
    method = spec.protocol.production.method
    protocol["production_seed"] = seeds["md" if method == "md" else "rest2"]
    return {"protocol": protocol}


def execution_projection(spec: SimulationSpec) -> dict:
    """Machine choices. Recorded in provenance; never part of a compatibility decision."""
    return {"execution": _dump(spec)["execution"]}


def hashes(spec: SimulationSpec) -> dict[str, str]:
    return {
        "system_build_sha256": sha256_of(system_build_projection(spec)),
        "prepared_state_sha256": sha256_of(prepared_state_projection(spec)),
        "protocol_sha256": sha256_of(protocol_projection(spec)),
        "protocol_at_prepare_sha256": sha256_of(protocol_at_prepare_projection(spec)),
        "execution_sha256": sha256_of(execution_projection(spec)),
    }
