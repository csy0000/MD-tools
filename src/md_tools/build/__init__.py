"""Topology construction and MD-script generation for the `md-openmm` command surface.

`build-top` turns one input structure into a serialised OpenMM `System` and the matching PDB.
`build-md` turns a resolved protocol configuration into small, readable run scripts that import
this installed package. Neither reads a sibling checkout, and neither assumes a repository root:
the installed configuration examples are located through the distribution's own metadata, by
`md_tools.configs.example_root()`.
"""

from __future__ import annotations

__all__ = ["strict", "record", "top", "md"]
