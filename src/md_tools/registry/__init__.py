"""Finish detection, inventory, transactional registration, and the local symlink.

Ported into MD-tools from MD-project's `finish.py`, `finish_gate.py`, `register.py`,
`cli_finish.py` and `cli_register.py`, which are removed there. The safety behaviour is the point
of the port: every rule below exists because losing or corrupting finished simulation data is
unrecoverable in a way that a failed run is not.
"""

from __future__ import annotations

__all__ = ["errors", "userconfig", "discovery", "inventory", "transaction", "register"]
