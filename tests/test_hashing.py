"""One hash function, and every caller reaching it.

Content hashes are how this package answers every identity question -- whether two bundles are the
same system, whether a resumed run is the same calculation, whether a relocated bundle is intact.
The same eight-line function had been written seven times under five names. They agreed on the day
they were consolidated; the risk was that one would later acquire a different block size or a
different encoding and two bundles would disagree for a reason nobody could find.

These tests keep the consolidation real: every surviving public name must produce the canonical
digest, so a future edit to one of them fails here rather than in a provenance record.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from md_templates.openmm import hashing  # noqa: E402


def _named_hashers():
    """Every public spelling of "hash this file" that still exists in the package."""
    from md_templates.openmm import (bundlev2, cli, cli_system_gen, input_gen, schemas, stage,
                                     system_prep)
    return {
        "hashing.sha256_file": hashing.sha256_file,
        "bundlev2.sha256_file": bundlev2.sha256_file,
        "cli_system_gen.sha256_of": cli_system_gen.sha256_of,
        "input_gen._sha256": input_gen._sha256,
        "system_prep._sha256": system_prep._sha256,
        "schemas.sha256_file": schemas.sha256_file,
        "stage._file_sha256": stage._file_sha256,
        "cli._sha256_of_file": cli._sha256_of_file,
    }


@pytest.mark.parametrize("relative", [
    "pyproject.toml",
    "README.md",
    "src/md_templates/openmm/config.py",
    "src/md_templates/openmm/manifests/systems/ace_ala_nme.pdb",
])
def test_every_spelling_gives_the_canonical_digest(relative):
    path = REPO_ROOT / relative
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, fn in _named_hashers().items():
        assert fn(path) == expected, f"{name} disagrees on {relative}"


def test_the_digest_is_of_bytes_not_of_parsed_content(tmp_path):
    """Two documents that parse identically but differ in bytes must hash differently.

    This is the property the whole provenance story rests on: a reformatted JSON file is a changed
    file, and hashing parsed content would call it unchanged.
    """
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1, "y": 2}')
    b.write_text('{\n  "x": 1,\n  "y": 2\n}\n')
    import json
    assert json.loads(a.read_text()) == json.loads(b.read_text())
    assert hashing.sha256_file(a) != hashing.sha256_file(b)


def test_block_reading_matches_a_single_shot_hash(tmp_path):
    """A file several blocks long must hash the same as if it were read in one go."""
    payload = bytes(range(256)) * (5 * hashing.BLOCK_SIZE // 256 + 7)
    path = tmp_path / "large.bin"
    path.write_bytes(payload)
    assert len(payload) > 5 * hashing.BLOCK_SIZE
    assert hashing.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_an_empty_file_hashes_to_the_empty_digest(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert hashing.sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_text_hashing_pins_utf8(tmp_path):
    """The encoding is a decision, and every caller must make the same one."""
    text = "cyclo-(RGDfV) — café åäö"
    assert hashing.sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert hashing.sha256_text(text) != hashlib.sha256(text.encode("utf-16")).hexdigest()

    from md_templates.openmm import schemas
    assert schemas.sha256_text(text) == hashing.sha256_text(text)


def test_text_and_bytes_agree_for_the_same_content():
    assert hashing.sha256_bytes(b"abc") == hashing.sha256_text("abc")


def test_only_the_seed_derivation_still_uses_hashlib_directly():
    """A guard against the duplication growing back.

    `seeds.py` is the one legitimate exception: it needs the raw digest BYTES to derive an integer
    seed, not a hex string, so it is a different operation rather than another copy of this one.
    """
    import re

    offenders = {}
    for path in (REPO_ROOT / "src" / "md_templates" / "openmm").glob("*.py"):
        if path.name in ("hashing.py", "seeds.py"):
            continue
        hits = [line.strip() for line in path.read_text().splitlines()
                if re.search(r"hashlib\.sha256", line)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        "these modules hash directly instead of using md_templates.openmm.hashing: "
        f"{offenders}")
