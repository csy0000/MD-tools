"""The runtime that generated MD scripts import.

A script written by `md-openmm build-md` is a small, readable entry point: it declares the stage's
resolved settings as a literal dict and calls into this package. The physics lives here, in the
installed distribution, so a generated directory can be copied anywhere, contains no absolute
path, and never imports a source checkout.
"""

from __future__ import annotations

__all__ = ["stage"]
