"""Deterministic serialisation and hashing, shared by every layer that must agree on bytes.

These three helpers decide what "the same configuration" means. They were defined alongside the
legacy OpenMM manifest reader, which made them look engine-specific; they are not, and both the
engine-neutral fingerprint projection and the manifest reader need exactly the same answer. Two
implementations that drifted by a separator or a sort order would produce two different hashes for
one document, and the first symptom would be a bundle that refuses its own configuration.

One definition, imported by everything, including the legacy `md_templates.openmm.schemas` path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["canonical_json", "sha256_text", "sha256_file"]


def canonical_json(doc: Any) -> str:
    """Deterministic serialisation used for hashing.

    Sorted keys, no insignificant whitespace, UTF-8. Two documents that differ only in key order or
    formatting must hash identically; two that differ in any *value* must not.
    """
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
