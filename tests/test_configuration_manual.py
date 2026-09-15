"""The configuration manual is rendered from the schemas, and cannot drift from them.

WHY A GENERATED MANUAL AND HAND-EDITABLE EXAMPLES ARE BOTH RIGHT.

The shipped examples in `configs/` are deliberately NOT generated: they are browsable,
hand-editable files, which is what lets someone read a complete configuration on GitHub without
installing anything, and the guarantee protecting them is that each one resolves through the real
resolver to the model's own defaults (`test_config_generation.py`).

A manual is the other artefact -- every accepted key, documented once -- and the authoritative
text for each key already exists exactly once, in its `Field(doc=...)`. Copying that into prose
creates a second copy that drifts, which is precisely what happened to the four comprehensive
configs: 97% of `cMD.config` and `REST2.config` were identical lines, 98% between `REST2.config`
and `rREST2.config`, and `umbrella.config` had decayed into a copy of its own minimal example
covering 12 of 52 keys. So the manual is rendered and held to the schemas here.

PLATFORM_POLICY_EXEMPTION: string rendering only. No Context, no device, no simulation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools.build.manual import documented_keys, render
from md_tools.build.md import MD_SCHEMA
from md_tools.build.top import BUILD_SCHEMA

MANUAL = Path(__file__).resolve().parents[1] / "docs" / "md-configuration.md"


def test_the_shipped_manual_is_exactly_what_the_renderer_emits():
    """THE DRIFT TEST. An edit to the document is a failure, not a change."""
    assert MANUAL.is_file(), f"{MANUAL} is missing; regenerate it from md_tools.build.manual"
    shipped = MANUAL.read_text(encoding="utf-8")
    fresh = render()
    if shipped != fresh:
        shipped_lines, fresh_lines = shipped.splitlines(), fresh.splitlines()
        first = next((i for i, (a, b) in enumerate(zip(shipped_lines, fresh_lines)) if a != b),
                     min(len(shipped_lines), len(fresh_lines)))
        pytest.fail(
            f"docs/md-configuration.md no longer matches md_tools.build.manual.render().\n"
            f"  shipped has {len(shipped_lines)} lines, rendered has {len(fresh_lines)}.\n"
            f"  first difference at line {first + 1}:\n"
            f"    shipped:  {shipped_lines[first] if first < len(shipped_lines) else '<end>'!r}\n"
            f"    rendered: {fresh_lines[first] if first < len(fresh_lines) else '<end>'!r}\n"
            f"  Regenerate it; do not edit it by hand.")


def test_every_accepted_key_is_documented():
    """A key the resolver accepts and the manual omits is an undocumented interface."""
    text = MANUAL.read_text(encoding="utf-8")
    missing = sorted(key for key in documented_keys() if f"`{key}`" not in text)
    assert not missing, f"accepted keys absent from the manual: {missing}"


def test_the_manual_documents_every_field_of_both_schemas():
    """Counted against the schemas themselves, so adding a Field cannot be forgotten."""
    expected = sum(len(schema.fields) for schema in (BUILD_SCHEMA, MD_SCHEMA))
    expected += sum(len(section.fields)
                    for schema in (BUILD_SCHEMA, MD_SCHEMA)
                    for section in schema.sections.values())
    assert len(documented_keys()) == expected, (
        "documented_keys() and the schemas disagree about how many keys exist")


def test_every_field_carries_a_doc_string():
    """An accepted key with no documentation is the gap the manual exists to close."""
    undocumented = []
    for label, schema in (("build-top", BUILD_SCHEMA), ("build-md", MD_SCHEMA)):
        for field in schema.fields.values():
            if not field.doc.strip():
                undocumented.append(f"{label}: {field.name}")
        for section in schema.sections.values():
            for field in section.fields.values():
                if not field.doc.strip():
                    undocumented.append(f"{label}: {section.name}.{field.name}")
    assert not undocumented, f"Field()s with no doc=: {undocumented}"


def test_the_manual_states_that_it_is_generated():
    """A reader who does not know it is generated will edit it."""
    text = MANUAL.read_text(encoding="utf-8")
    assert "Do not hand-edit" in text
    assert "md_tools.build.manual.render()" in text


def test_the_protocol_table_names_every_protocol():
    """The table is the duplication the four hand-written configs expressed by copying."""
    from md_tools.build.md import PROTOCOLS

    text = MANUAL.read_text(encoding="utf-8")
    for protocol in PROTOCOLS:
        assert f"| `{protocol}` |" in text, f"{protocol} missing from the applicability table"
