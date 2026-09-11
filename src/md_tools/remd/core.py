"""Values and derivations the ladder shares with any reference bundle exported from it.

Everything here is plain arithmetic over plain numbers: no OpenMM, no md_tools, no filesystem.
That is the point. `md_tools.reference.rest2_export` copies this module, and the four beside it
that have the same property, VERBATIM into an exported bundle -- so the exchange criterion, the
sweep schedule and the seed derivation in a bundle are not a reimplementation that has to be
kept in agreement, they are the same bytes.

Two things live here rather than where they used to, and both moved for that reason:

  * `BAR_NM3_TO_KJ_PER_MOL` was in `protocol.py`, which imports `md_tools.rest2` and so drags the
    whole package in behind it. `engine.py` needed one float from it and was therefore not
    importable with numpy and OpenMM alone.
  * `stream_seed` was `driver._stream_seed`, and `driver.py` is 1993 lines of resume, MPI
    coordination, checkpointing and provenance -- none of which belongs in a frozen artefact.
    The seed derivation does: it decides which random stream an exchange draws from, so a bundle
    that derived it differently would propose the same swaps and accept different ones.
"""
from __future__ import annotations

#: bar * nm^3 -> kJ/mol. The pV term of the reduced potential, for a barostatted state.
BAR_NM3_TO_KJ_PER_MOL = 0.0602214076


def stream_seed(base, name):
    """A named, reproducible substream from one recorded seed.

    Distinct streams for distinct purposes: the exchange acceptance draw must not share a
    generator with anything else, or adding an unrelated draw somewhere would silently change
    which swaps a recorded seed produces.
    """
    value = int(base)
    for byte in str(name).encode("utf-8"):
        value = (value * 1000003 + byte) & 0xFFFFFFFF
    return (value % (2 ** 31 - 1)) or 1
