"""One definition of "the hash of this file".

Content hashes are how this package answers every question about identity: whether two bundles are
the same system, whether a resumed run is the same calculation, whether a relocated bundle is
intact. That made it worth noticing that the same eight-line function had been written seven times
across seven modules, under five different names -- `sha256_file`, `_sha256`, `_file_sha256`,
`sha256_of`, plus one wrapper that already delegated.

They were byte-identical when this module was written, and verified so against four files before the
consolidation. The risk was never that they disagreed; it was that one of them would later acquire a
different block size, a different error path, or a `text` mode, and two bundles would then disagree
about a file for a reason nobody could find. A provenance system with seven hash functions has seven
places to introduce that.

Everything hashes **bytes**. Parsing and re-serialising a document before hashing would make a
changed file hash the same, which defeats the purpose.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["BLOCK_SIZE", "sha256_bytes", "sha256_file", "sha256_text"]

#: Read granularity. Large enough that the loop is not the bottleneck on a multi-gigabyte
#: trajectory, small enough that a checkpoint does not have to fit in memory to be hashed.
BLOCK_SIZE = 1 << 20


def sha256_file(path) -> str:
    """Hex SHA-256 of a file's bytes, read in blocks so size does not matter."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a string, encoded UTF-8.

    Separate from `sha256_file` because the encoding is a decision: hashing a `str` requires
    choosing one, and every caller must choose the same one or two records of the same text will
    disagree.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of an in-memory buffer, for content that was never written to disk."""
    return hashlib.sha256(data).hexdigest()
