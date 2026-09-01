"""One error type for everything registration refuses.

Registration says no far more often than it says yes, and every refusal has to leave the source
untouched and explain itself well enough that the person can decide what to do. A single type
keeps the CLI's handling honest: anything else escaping is a bug, not a refusal.
"""

from __future__ import annotations


class RegistrationError(Exception):
    """Registration refused. The source is unchanged."""
