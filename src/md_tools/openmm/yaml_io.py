"""Write a YAML document the way every file this package emits is written.

One helper, one place. `builders.py` writes the resolved system document beside a build and
`runtime/replica.py` writes a reservoir declaration; both must produce the same shape of file --
block style, keys in the order the writer chose rather than alphabetised, UTF-8 -- because these
files are read back by tooling and diffed by people.

It lived in `openmm/config.py`, which is now specifically the build-top system resolver. A YAML
writer is not configuration, and the runtime importing a module called `config` to write a
reservoir file was misleading about what depends on what.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["write_yaml"]

def write_yaml(path: Path, document: dict[str, Any], *, header: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=88)
    path.write_text((header + text) if header else text, encoding="utf-8")
    return path
