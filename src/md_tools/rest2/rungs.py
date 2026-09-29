"""Where every rung of a ladder came from: declared, verified against the saved states, recorded.

A scaled Hamiltonian is built ONCE, as a file, by `md-openmm build-top --rest2-scaler`, and never
re-derived at run time. `md-run` and the generated scripts read every rung from a saved state named
in the group file. The DIRECT Python API (`ReplicaRun` constructed with a hand-built
`LadderPreflight`) is the other door, and this module is what it is checked with (shared contract
§3, "The ladder's direct Python API", decided by the user 2026-09-19):

  * `rung_source` says how the rungs were made. `saved-states` is set only by the saved-state
    preflight; a direct caller declares `caller-supplied` WITH a reason, or its rungs are refused.
  * Each rung's ORIGIN is decided by its canonical System digest (`identity.system_fingerprint`,
    the serialised XML with file-level attributes stripped -- stable under a round trip, not a hash
    of Python objects), compared with every state the `scaler.yaml` lists: `saved-state <i>
    (verified)` on a match, `caller-modified` otherwise. A caller-modified rung is allowed -- a
    biased auxiliary rung is a legitimate method -- but it is never presented as a saved state.
  * Particle count, masses and constraints are checked against the saved states' System.
  * The declaration and the per-rung origins are part of the ladder's identity (`provenance`),
    so a resume with other rungs or another declaration is refused.

Nothing here writes a file; the record lives in `restart.json` through the identity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["SAVED_STATES", "CALLER_SUPPLIED", "RUNG_SOURCES", "RungProvenanceError",
           "check_declaration", "find_saved_states_record", "saved_state_digests",
           "rung_origins", "check_rung_structure", "provenance", "RUNGS_FORMAT"]

SAVED_STATES = "saved-states"
CALLER_SUPPLIED = "caller-supplied"
RUNG_SOURCES = (SAVED_STATES, CALLER_SUPPLIED)
#: The shape of the `rungs` identity entry.
RUNGS_FORMAT = "md-tools-ladder-rungs/1"


class RungProvenanceError(ValueError):
    """Rungs whose origin is undeclared, unverifiable, or structurally not this system's."""


def check_declaration(*, rung_systems: Sequence, rung_source: str, rung_source_reason: str,
                      sealed: bool) -> None:
    """The declaration rules, applied when a plan is made. `sealed` is True only for the plan the
    saved-state preflight builds itself."""
    if rung_source not in ("",) + RUNG_SOURCES:
        raise RungProvenanceError(
            f"rung_source {rung_source!r} is not one of {list(RUNG_SOURCES)}")
    if rung_source == SAVED_STATES and not sealed:
        raise RungProvenanceError(
            "rung_source='saved-states' is set by md-tools' own saved-state preflight (md-run and "
            "the generated scripts) and cannot be claimed by a caller: a hand-built plan's rungs "
            "are never presented as saved states on its own word. Declare "
            "rung_source='caller-supplied' with a rung_source_reason; each rung is then compared "
            "with the saved states by digest and recorded as 'saved-state <i> (verified)' where "
            "it matches.")
    if rung_systems and not rung_source:
        raise RungProvenanceError(
            f"this plan supplies {len(rung_systems)} rung System(s) without saying where they come "
            f"from. A ladder's rungs are saved states built by `md-openmm build-top "
            f"--rest2-scaler`; a direct caller that supplies its own must declare "
            f"rung_source='caller-supplied' and a non-empty rung_source_reason, and every rung's "
            f"origin is then recorded.")
    if rung_source == CALLER_SUPPLIED and not str(rung_source_reason or "").strip():
        raise RungProvenanceError(
            "rung_source='caller-supplied' needs a non-empty rung_source_reason: the record says "
            "why these rungs are not simply the saved states, and an empty reason says nothing.")


def find_saved_states_record(*, explicit=None, system_path=None) -> Path | None:
    """The `scaler.yaml` a caller's rungs are checked against, in the contract's order:

      (a) `explicit` (LadderPreflight.saved_states_record);
      (b) the record naming `system_path`, when that is itself a saved state;
      (c) `<dir of system_path>/REST2/scaler.yaml`, only if its source System's sha256 is
          `system_path`'s.

    None when none applies; the caller refuses by name.
    """
    from .states import RECORD_NAME, load_scaler_record, scaled_state_identity

    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise RungProvenanceError(f"saved_states_record {path} does not exist")
        load_scaler_record(path)
        return path
    if not system_path:
        return None
    system_path = Path(system_path)
    if not system_path.is_file():
        return None
    identity = scaled_state_identity(system_path)
    if identity is not None:
        return Path(identity["record"])
    beside = system_path.parent / "REST2" / RECORD_NAME
    if beside.is_file():
        record = load_scaler_record(beside)
        digest = hashlib.sha256(system_path.read_bytes()).hexdigest()
        if (record.get("source") or {}).get("system_sha256") == digest:
            return beside
    return None


def saved_state_digests(record_path: Path) -> tuple[list[str], list[float], Any]:
    """Canonical digest and tau of every state the record lists, in state order, verified against
    the record's file sha256; plus the System of state 0 for the structural checks."""
    from openmm import XmlSerializer

    from .identity import system_fingerprint
    from .states import ScaledStateError, load_scaler_record

    record_path = Path(record_path)
    record = load_scaler_record(record_path)
    digests, taus, reference = [], [], None
    for entry in sorted(record["states"], key=lambda e: int(e["state"])):
        path = record_path.parent / entry["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ScaledStateError(
                f"{path} does not have the sha256 {record_path} recorded for it; the saved states "
                f"changed after they were built, so nothing can be verified against them.")
        system = XmlSerializer.deserialize(raw.decode("utf-8"))
        digests.append(system_fingerprint(system))
        taus.append(float(entry["tau"]))
        if reference is None:
            reference = system
    return digests, taus, reference


def rung_origins(rungs: Sequence, saved_digests: Sequence[str], *,
                 restraints_added: bool = False) -> list[dict[str, Any]]:
    """Per rung: its canonical digest and its origin against the saved states."""
    from .identity import system_fingerprint

    out = []
    for index, system in enumerate(rungs):
        digest = system_fingerprint(system)
        origin = (f"saved-state {saved_digests.index(digest)} (verified)"
                  if digest in saved_digests else "caller-modified")
        if restraints_added and origin != "caller-modified":
            origin += "; ladder restraints added"
        out.append({"rung": index, "system_digest": digest, "origin": origin})
    return out


def check_rung_structure(rungs: Sequence, reference) -> None:
    """Refuse a rung that is not a Hamiltonian OF this system: particles, masses, constraints."""
    from openmm import unit

    def masses(system):
        return [system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(system.getNumParticles())]

    def constraints(system):
        out = []
        for i in range(system.getNumConstraints()):
            a, b, d = system.getConstraintParameters(i)
            out.append((int(a), int(b), float(d.value_in_unit(unit.nanometer))))
        return sorted(out)

    reference_masses, reference_constraints = masses(reference), constraints(reference)
    for index, system in enumerate(rungs):
        if system.getNumParticles() != reference.getNumParticles():
            raise RungProvenanceError(
                f"rung {index} has {system.getNumParticles()} particles and the saved states have "
                f"{reference.getNumParticles()}: it is not a Hamiltonian of this system")
        if masses(system) != reference_masses:
            raise RungProvenanceError(
                f"rung {index}'s particle masses differ from the saved states': an exchange "
                f"between them would move configurations between two different systems")
        if constraints(system) != reference_constraints:
            raise RungProvenanceError(
                f"rung {index}'s constraints differ from the saved states'; the configuration "
                f"space of the ladder would not be one space")


def provenance(*, rung_source: str, rung_source_reason: str, origins: Iterable[dict],
               record: Path | None) -> dict[str, Any]:
    """The `rungs` entry of the ladder's identity, and what `restart.json` records."""
    return {"format": RUNGS_FORMAT, "rung_source": rung_source,
            "rung_source_reason": str(rung_source_reason or ""),
            "saved_states_record": Path(record).name if record else None,
            "rungs": [dict(entry) for entry in origins]}


def all_verified_saved_states(entry: dict | None) -> bool:
    """Whether a `rungs` entry describes a plain saved-state ladder: what every ladder before
    0.6.1 ran, and the ONLY case the compatibility branch accepts in place of a missing entry."""
    return bool(entry) and entry.get("rung_source") == SAVED_STATES and all(
        str(r.get("origin", "")).startswith("saved-state ") for r in entry.get("rungs") or ())
