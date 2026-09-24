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

`render()` is the whole interface. `docs/basics/build-md/configuration.md` is its output, and a test asserts
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


#: What a run WRITES. Not a setting, so it is not derivable from the schema -- but it belongs in
#: the manual for the same reason the keys do: it existed as a hand-maintained block in four
#: separate config files, which is four copies of one explanation.
_OUTPUTS = """
## What a run writes

Nothing in this section is a setting. It is what the files you get MEAN, because a reader should
not have to run something to find out.

### The `.out` census

Every stage's `.out` opens with what it actually ran, read off the **serialised System** rather
than off the configuration that asked for it. That distinction is the point: a configuration
asking for a 1.0 nm cutoff and a System carrying 0.8 nm are different runs, and only one of them
integrates.

| section | what it reports |
|---|---|
| `System` | atoms, residues by name, net charge, degrees of freedom, box and volume |
| `Method` | nonbonded treatment and cutoff, Ewald tolerance, dispersion correction, switching, 1-4 exception count, constraints, barostat present, force inventory |
| `Selections` | the solute atom count, and which omega bonds are left unscaled |

The same facts enter the `.log` as structured fields, because prose is not a database. A box is
reported only when the System is genuinely **periodic**: an OpenMM System defaults to a 2 nm cube,
so an implicit run would otherwise print an invented box and an 8 nm³ volume in the same voice as
the measurement above it.

### The per-term energy decomposition

`energy_components_<stage>.csv` carries the potential energy term by term beside `mdout*.csv` --
except the production stage itself, which is `energy_components.csv` on its first segment and
`energy_components_prod<N>.csv` on later ones. `md_tools.md._stages.energy_components_name` is the
one authority, and only `cMD` and `umbrella` count as production stage names.

Where the System has usable force groups -- the implicit route, through ParmEd -- they are read
directly. Where it does not, which is every explicit-solvent System built through OpenMM's
`ForceField.createSystem`, a group-separated **copy** is probed instead: `built.xml`, the run's own
Context and every digest taken from them stay untouched, because a force group is part of the
serialised System and regrouping the integrated one would change `system_sha256` and invalidate
every checkpoint fingerprint in flight.

`EELEC` from `VDWAALS`, and the 1-4 terms from either, are **not** reachable that way -- every 1-4
pair is an exception inside the single `NonbondedForce`, which evaluates charge and dispersion in
one kernel -- so that split lives in post-hoc `md_tools.openmm.decomposition`.

### Fluctuations over a single sample

An rms fluctuation over one report reads `n/a (single sample)` rather than `0`. Over one sample
`sqrt(<x^2> - <x>^2)` is exactly zero, which reads as "this did not move" when it means "there was
nothing to compare it against" -- and a stage shorter than `info_printout` produces exactly one
row, so this is the ordinary case rather than the corner one.

### A ladder additionally writes

The Hamiltonian it integrated, so the ladder is checkable from its own output rather than from the
configuration that requested it: the tau ladder, the scaling laws applied and the terms left
unscaled, the solute region and its excluded omega bonds, `system_sha256`, that velocities are
never rescaled on a swap (one beta across the ladder), and a TIMINGS block with elapsed time,
per-replica and aggregate throughput, and cost per step.

Per-state and per-segment filenames are documented in `docs/basics/run-layout.md`.
"""


def render() -> str:
    """The whole manual, as Markdown. `docs/basics/build-md/configuration.md` is exactly this."""
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
    lines += _OUTPUTS.strip("\n").split("\n")
    lines.append("")
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
