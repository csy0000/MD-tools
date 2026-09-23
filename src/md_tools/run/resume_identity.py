"""What makes a stored `resolved.config` the SAME run as the one being resumed.

A resumed run must not append samples from a different calculation. That is the whole purpose of
the check, and it is not negotiable. What was wrong was the instrument: the gate compared the two
resolved documents as dictionaries, so ANY difference refused -- including a field that has
nothing to do with the protocol being run.

The concrete failure: `ais.work_measurement` and `ais.verify_every_updates` were added to the
schema with defaults. Every REST2 `resolved.config` written before that commit lacks them and
every one written after has them, so every in-flight REST2 ladder in the world became
unresumable -- by two settings a REST2 run never reads. The only exit was `--overwrite`, which
destroys the outputs, so in practice a long campaign was abandoned rather than resumed. An engine
upgrade must not do that; the runs that most need a fix are the long ones already going.

So identity is defined by what determines the CALCULATION, on two axes:

  WHICH SECTIONS   A protocol's identity covers the sections that protocol actually reads. The
                   map is `_IN_SECTIONS` in `build.md` -- the same one that decides what its `.in`
                   file can express -- so "what this run depends on" has one definition and the
                   input language and the resume gate cannot drift apart. A REST2 resume does not
                   care what `ais:` says, because a REST2 run never looks at it.

  WHICH CHANGES    A field ABSENT from the stored document whose new value is the schema default
                   was added by an upgrade and was never chosen by anyone. That is not a change of
                   calculation. A field present in both with different values is.

WHAT THIS DOES NOT RELAX. A genuinely load-bearing difference still refuses, and now names the
fields rather than saying only that something differs. Reporting intervals in particular remain
part of a REST2 run's identity: after the ladder began honouring `crd_printout_solute` and
`info_printout`, a run started before that legitimately differs in frame spacing and MUST be
refused, because the cadence would change mid-run.

That protection does not rest on this gate alone. `ReplicaRun.compare_identity` checks the
schedule stored inside the exchange record itself -- `checkpoint_interval_ps`,
`solute_output_interval_ps`, `whole_output_interval_ps` all reach it through
`schedule.describe()` -- and refuses with `IdentityError` naming each field. Two independent
gates, and only the outer, blunter one is being sharpened here.
"""
from __future__ import annotations

from typing import Any

#: Written into every `resolved.config` and never compared. It exists so that a field missing from
#: a stored document can be read as "written before this field existed" rather than "deliberately
#: removed": absence alone cannot distinguish the two, and the difference decides whether a resume
#: is safe. Bump it when a section or field is ADDED or REMOVED, not when a default changes.
SCHEMA_VERSION = 4

#: Never part of any run's identity: metadata about the document, not about the calculation.
METADATA_KEYS = ("schema_version",)


def sections_for(protocol: str) -> frozenset[str]:
    """The configuration sections `protocol` actually reads. `""` means the top-level fields.

    Derived from `build.md._IN_SECTIONS` rather than restated, so that a protocol which gains a
    section gains it here too and nobody has to remember this file exists.
    """
    from ..build.md import _IN_SECTIONS

    if protocol not in _IN_SECTIONS:
        # An unknown protocol gets the strictest reading: everything matters. Better to refuse a
        # resume that would have been fine than to permit one that changes the calculation.
        return frozenset({""} | set(_all_section_names()))
    names = {""}
    for _namelist, sections in _IN_SECTIONS[protocol]:
        names.update(sections)
    return frozenset(names)


def _all_section_names() -> frozenset[str]:
    from ..build.md import MD_SCHEMA

    return frozenset(MD_SCHEMA.sections)


def _default_of(section: str, key: str) -> tuple[bool, Any]:
    """`(has_default, default)` for one schema field. `(False, None)` when there is no such field."""
    from ..build.md import MD_SCHEMA
    from ..build.strict import _MISSING

    fields = MD_SCHEMA.fields if section == "" else (
        MD_SCHEMA.sections[section].fields if section in MD_SCHEMA.sections else {})
    field = fields.get(key)
    if field is None:
        return (False, None)
    if getattr(field, "default", _MISSING) is _MISSING:
        return (False, None)
    return (True, field.default)


def differences(stored: dict[str, Any], resolved: dict[str, Any], *,
                protocol: str | None = None) -> list[str]:
    """Dotted paths that differ in a way that changes the calculation. Empty means resumable.

    `protocol` defaults to the one the NEW document declares. A stored document that declares a
    different protocol is itself a difference, and is reported as `protocol` rather than silently
    compared under one of the two.
    """
    protocol = protocol or resolved.get("protocol")
    relevant = sections_for(protocol)
    stored_version = stored.get("schema_version")
    # An upgrade may have ADDED fields; the same version cannot have. When the stored document was
    # written by this very schema, a field present in one and not the other is a real difference
    # rather than an upgrade artefact, and is reported.
    upgraded = stored_version != SCHEMA_VERSION

    found: list[str] = []

    def compare(section: str, before: Any, after: Any) -> None:
        prefix = f"{section}." if section else ""
        if not isinstance(before, dict) or not isinstance(after, dict):
            if before != after:
                found.append(section or "<document>")
            return
        for key in sorted(set(before) | set(after)):
            if section == "" and key in METADATA_KEYS:
                continue
            if section == "" and key in _all_section_names():
                continue                      # sections are walked separately
            in_before, in_after = key in before, key in after
            if in_before and in_after:
                if before[key] != after[key]:
                    found.append(f"{prefix}{key}")
            elif in_after and upgraded:
                # Added by an upgrade. Only a value left AT ITS DEFAULT is an upgrade artefact --
                # one that was actually set is a choice this run was never made under.
                has_default, default = _default_of(section, key)
                if not (has_default and after[key] == default):
                    found.append(f"{prefix}{key}")
            else:
                found.append(f"{prefix}{key}")

    compare("", stored, resolved)
    for section in sorted(relevant - {""}):
        before, after = stored.get(section), resolved.get(section)
        if before is None and after is None:
            continue
        if before is None:
            # A whole section absent from the stored document. Under an upgrade that is a new
            # section; if every field in it sits at its default, nothing was chosen.
            if upgraded and isinstance(after, dict) and all(
                    _default_of(section, k) == (True, v) for k, v in after.items()):
                continue
            found.append(section)
            continue
        if after is None:
            found.append(section)
            continue
        compare(section, before, after)
    return sorted(set(found))


def explain(path, differing: list[str], stored: dict[str, Any],
            resolved: dict[str, Any]) -> str:
    """The refusal, naming every field and both of its values.

    Modelled on `IdentityError`'s message, which says exactly what changed. "Describes a different
    run" is true and useless: the reader has to diff two documents by hand to find out what the
    engine already knows.
    """
    def value_at(document, dotted):
        section, _, key = dotted.rpartition(".")
        block = document.get(section, {}) if section else document
        return block.get(key, "<absent>") if isinstance(block, dict) else "<absent>"

    lines = "\n".join(
        f"    {name}: was {value_at(stored, name)!r}, now {value_at(resolved, name)!r}"
        for name in differing)
    return (
        f"{path} describes a different run:\n{lines}\n"
        f"  These settings determine the calculation, so continuing would append samples from "
        f"one run to another. Use a different -odir, or --overwrite if the outputs there are "
        f"meant to be replaced.\n"
        f"  Fields belonging to protocols this run does not use, and fields added by an engine "
        f"upgrade that are still at their defaults, are NOT compared -- so an upgrade alone never "
        f"makes a run unresumable.")
