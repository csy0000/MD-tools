"""The configuration manual, rendered FROM the schemas that enforce it.

WHY THIS IS GENERATED AND THE EXAMPLES ARE NOT.

`configs/` holds ordinary, browsable, hand-editable example files, and that is deliberate: it is
what lets someone read a complete configuration on GitHub without installing anything, and the
guarantee protecting them is that every example resolves, through the real resolver, to the
model's own defaults. They may be reworded and reordered freely.

A MANUAL is a different artefact. It documents every accepted key once, and the authoritative text
for each key already exists exactly once -- in the `Field(doc=...)` string the schema declares.
Hand-copying that into a document makes a second copy that can drift, and the four comprehensive
configs showed what that costs: 97% of `cMD.config` and `REST2.config` were identical lines, 98%
between `REST2.config` and `rREST2.config`, and `umbrella.config` had decayed into a copy of its
own minimal example documenting 12 of 52 keys. Nobody can keep five hand-written references in
step, so this one is rendered and drift-checked instead -- the same treatment
`md_tools.data_contract` gives its published JSON schemas.

`render()` is the whole interface. `docs/md-configuration.md` is its output, and a test asserts
the two agree, so the document cannot go stale without the suite saying so.
"""
from __future__ import annotations

from typing import Any

from .md import _IN_SECTIONS, MD_SCHEMA
from .strict import Field, Schema, Section
from .top import BUILD_SCHEMA

#: How a Python type reads in prose. A tuple of types is joined with "or".
_TYPE_NAMES = {int: "integer", float: "number", str: "string", bool: "boolean",
               list: "list", dict: "mapping"}


def _type_of(field: Field) -> str:
    kinds = field.kind if isinstance(field.kind, tuple) else (field.kind,)
    names = [_TYPE_NAMES.get(kind, getattr(kind, "__name__", str(kind))) for kind in kinds]
    rendered = " or ".join(dict.fromkeys(names))
    return f"{rendered} or null" if field.nullable and "null" not in rendered else rendered


def _default_of(field: Field) -> str:
    if field.required:
        return "**required**"
    value = field.default
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, str):
        return f"`{value}`" if value else "`\"\"`"
    return f"`{value}`"


def _constraints_of(field: Field) -> str:
    parts: list[str] = []
    if field.enum is not None:
        parts.append("one of " + ", ".join(f"`{option}`" for option in field.enum))
    if field.minimum is not None:
        parts.append(f"minimum {field.minimum}")
    if field.maximum is not None:
        parts.append(f"maximum {field.maximum}")
    if field.unit:
        parts.append(f"unit: {field.unit}")
    return "; ".join(parts)


def _doc_paragraphs(text: str) -> list[str]:
    """A `Field` doc is prose with embedded newlines. Keep its paragraphs, drop its wrapping."""
    if not text:
        return []
    out, current = [], []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if current:
                out.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        out.append(" ".join(current))
    return out


def _render_field(field: Field, *, path: str) -> list[str]:
    lines = [f"#### `{path}`", ""]
    facts = [f"type: {_type_of(field)}", f"default: {_default_of(field)}"]
    constraints = _constraints_of(field)
    if constraints:
        facts.append(constraints)
    lines.append(" · ".join(facts))
    lines.append("")
    for paragraph in _doc_paragraphs(field.doc):
        lines.append(paragraph)
        lines.append("")
    return lines


def _render_section(section: Section, *, prefix: str = "") -> list[str]:
    name = f"{prefix}{section.name}"
    lines = [f"### `{name}`", ""]
    if section.required:
        lines += ["**Required section.**", ""]
    for paragraph in _doc_paragraphs(section.doc):
        lines.append(paragraph)
        lines.append("")
    for field in section.fields.values():
        lines += _render_field(field, path=f"{name}.{field.name}")
    return lines


def _render_schema(schema: Schema, *, heading: str, command: str) -> list[str]:
    lines = [f"## {heading}", ""]
    for paragraph in _doc_paragraphs(schema.doc):
        lines.append(paragraph)
        lines.append("")
    lines += [f"Resolved by `{command}`. Unknown keys are refused by name rather than ignored, "
              f"so a typo is an error and never silent metadata.", ""]
    if schema.fields:
        lines += ["### Top-level keys", ""]
        for field in schema.fields.values():
            lines += _render_field(field, path=field.name)
    for section in schema.sections.values():
        lines += _render_section(section)
    return lines


def _render_applicability() -> list[str]:
    """Which sections each protocol actually reads, from the mapping the `.in` writer uses.

    This is the table the four hand-written configs were expressing by copying each other: the
    shared sections are the duplication, and the protocol-specific ones are the difference.
    """
    lines = ["## Which keys apply to which protocol", "",
             "A key that a protocol does not read is refused rather than ignored, so this table "
             "is a constraint and not a convenience.", "",
             "| protocol | sections it reads |", "|---|---|"]
    for protocol, blocks in _IN_SECTIONS.items():
        sections: list[str] = []
        for _, names in blocks:
            for name in names:
                label = "top-level" if name == "" else name
                if label not in sections:
                    sections.append(label)
        lines.append(f"| `{protocol}` | " + ", ".join(f"`{s}`" for s in sections) + " |")
    lines.append("")
    return lines


def render() -> str:
    """The whole manual, as Markdown. `docs/md-configuration.md` is exactly this."""
    lines = [
        "# Configuration reference",
        "",
        "**Generated from the schemas by `md_tools.build.manual.render()`. Do not hand-edit:**",
        "a test asserts this file and the schemas agree, so an edit here is a failure rather than",
        "a change. The authoritative text for every key is the `Field(doc=...)` string in",
        "`md_tools.build.top` (for `build-top`) and `md_tools.build.md` (for `build-md`).",
        "",
        "Every configuration file is YAML despite the `.config` suffix. Unknown keys are refused",
        "by name, so a misspelled key is an error and never silently becomes metadata. Every key",
        "below has a default unless it says **required**, and the defaults here are the ones the",
        "resolver actually applies -- the shipped examples in `configs/` are held to them by test.",
        "",
        "Lengths are **integer step counts**, everywhere. A duration in picoseconds would have to",
        "divide by a timestep the file may not have been written against; logs derive ps and ns",
        "for the reader instead.",
        "",
    ]
    lines += _render_applicability()
    lines += _render_schema(BUILD_SCHEMA, heading="`build-top`: topology and System construction",
                            command="md-openmm build-top")
    lines += _render_schema(MD_SCHEMA, heading="`build-md`: protocol, stages and reporting",
                            command="md-openmm build-md")
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip("\n") + "\n"


def documented_keys() -> set[str]:
    """Every dotted key the manual documents. Used by the drift test."""
    keys: set[str] = set()
    for schema in (BUILD_SCHEMA, MD_SCHEMA):
        for field in schema.fields.values():
            keys.add(field.name)
        for section in schema.sections.values():
            for field in section.fields.values():
                keys.add(f"{section.name}.{field.name}")
    return keys
