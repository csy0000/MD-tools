"""A small, safe parser for physical quantities.

Configuration accepts explicit quantities -- ``2 fs``, ``300 K``, ``1 /ps``, ``1.0 nm``, ``1 bar``,
``10 ps`` -- and normalises each into one canonical unit before anything is hashed or compared. Two
inputs that mean the same thing must produce the same canonical number, or the hashes that decide
bundle and continuity compatibility would disagree over spelling.

**No ``eval``.** The grammar is a number, optional whitespace, and one unit token from a fixed
table. Anything else is an error naming what was found. A configuration language that evaluates
arbitrary Python is a remote-code-execution surface in a file people copy between machines.

Canonical units are OpenMM's MD unit system, so a canonical value can be handed to OpenMM without
a second conversion:

    time         picosecond      ps
    length       nanometre       nm
    temperature  kelvin          K
    pressure     bar             bar
    mass         dalton          amu
    rate         per picosecond  /ps
    energy       kilojoule/mole  kJ/mol
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

__all__ = ["Quantity", "parse_quantity", "CANONICAL_UNITS", "UnitError", "DIMENSIONS"]


class UnitError(ValueError):
    """A quantity could not be parsed, or carries the wrong dimension for its field."""


#: dimension -> canonical unit symbol
CANONICAL_UNITS: dict[str, str] = {
    "time": "ps",
    "length": "nm",
    "temperature": "K",
    "pressure": "bar",
    "mass": "amu",
    "rate": "/ps",
    "energy": "kJ/mol",
    "dimensionless": "",
}

#: unit symbol -> (dimension, factor to the canonical unit)
_UNITS: dict[str, tuple[str, float]] = {
    # time
    "fs": ("time", 1e-3), "femtosecond": ("time", 1e-3), "femtoseconds": ("time", 1e-3),
    "ps": ("time", 1.0), "picosecond": ("time", 1.0), "picoseconds": ("time", 1.0),
    "ns": ("time", 1e3), "nanosecond": ("time", 1e3), "nanoseconds": ("time", 1e3),
    "us": ("time", 1e6), "microsecond": ("time", 1e6), "microseconds": ("time", 1e6),
    # length
    "pm": ("length", 1e-3), "picometer": ("length", 1e-3), "picometre": ("length", 1e-3),
    "angstrom": ("length", 0.1), "angstroms": ("length", 0.1), "A": ("length", 0.1),
    "nm": ("length", 1.0), "nanometer": ("length", 1.0), "nanometre": ("length", 1.0),
    "nanometers": ("length", 1.0), "nanometres": ("length", 1.0),
    # temperature
    "K": ("temperature", 1.0), "kelvin": ("temperature", 1.0),
    # pressure
    "bar": ("pressure", 1.0), "atm": ("pressure", 1.01325), "atmosphere": ("pressure", 1.01325),
    # mass
    "amu": ("mass", 1.0), "dalton": ("mass", 1.0), "Da": ("mass", 1.0),
    # rate (inverse time)
    "/ps": ("rate", 1.0), "ps^-1": ("rate", 1.0), "1/ps": ("rate", 1.0),
    "/ns": ("rate", 1e-3), "ns^-1": ("rate", 1e-3), "1/ns": ("rate", 1e-3),
    "/fs": ("rate", 1e3), "fs^-1": ("rate", 1e3), "1/fs": ("rate", 1e3),
    # energy
    "kJ/mol": ("energy", 1.0), "kj/mol": ("energy", 1.0),
    "kcal/mol": ("energy", 4.184),
}

#: dimensions a field may declare, for error messages
DIMENSIONS = tuple(sorted(CANONICAL_UNITS))

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_PATTERN = re.compile(rf"^\s*(?P<value>{_NUMBER})\s*(?P<unit>[^\s].*?)?\s*$")


class Quantity(NamedTuple):
    """A parsed quantity: the canonical magnitude, its canonical unit, and what was written."""

    value: float           # in the canonical unit for its dimension
    unit: str              # canonical unit symbol
    dimension: str
    source: str            # exactly what the input said, for provenance

    def as_dict(self) -> dict[str, Any]:
        """Canonical form: both the normalised value and the convention it is expressed in."""
        return {"value": self.value, "unit": self.unit, "dimension": self.dimension,
                "as_written": self.source}


def parse_quantity(raw: Any, *, dimension: str, field: str = "value") -> Quantity:
    """Parse `raw` into the canonical unit for `dimension`.

    A bare number is accepted ONLY for a dimensionless field. Elsewhere it is refused: a unitless
    `2` for a timestep meant femtoseconds in the old manifests and picoseconds in OpenMM's own unit
    system, and silently choosing one is a factor of a thousand. Migration supplies the unit
    explicitly instead -- see `spec.migrate`.
    """
    if dimension not in CANONICAL_UNITS:
        raise UnitError(f"{field}: unknown dimension {dimension!r}; expected one of {DIMENSIONS}")

    if isinstance(raw, Quantity):
        if raw.dimension != dimension:
            raise UnitError(f"{field}: expected a {dimension} quantity, got {raw.dimension}")
        return raw

    if isinstance(raw, bool):
        raise UnitError(f"{field}: expected a {dimension} quantity, got the boolean {raw!r}")

    if isinstance(raw, (int, float)):
        if dimension != "dimensionless":
            raise UnitError(
                f"{field}: {raw!r} has no unit. Write it explicitly, e.g. "
                f"'{raw} {CANONICAL_UNITS[dimension]}'. A unitless number is refused because the "
                "same figure means different things in different conventions."
            )
        return Quantity(float(raw), "", "dimensionless", str(raw))

    if not isinstance(raw, str):
        raise UnitError(f"{field}: expected a quantity such as '2 fs', got {type(raw).__name__}")

    match = _PATTERN.match(raw)
    if not match:
        raise UnitError(
            f"{field}: cannot read {raw!r} as a quantity. Expected a number and a unit, "
            f"e.g. '2 fs' or '300 K'."
        )
    value = float(match.group("value"))
    symbol = (match.group("unit") or "").strip()

    if not symbol:
        if dimension == "dimensionless":
            return Quantity(value, "", "dimensionless", raw)
        raise UnitError(
            f"{field}: {raw!r} has no unit; expected a {dimension} unit such as "
            f"'{CANONICAL_UNITS[dimension]}'."
        )

    entry = _UNITS.get(symbol)
    if entry is None:
        known = sorted({s for s, (d, _) in _UNITS.items() if d == dimension})
        raise UnitError(f"{field}: unknown unit {symbol!r}. For {dimension}, use one of {known}.")

    got_dimension, factor = entry
    if got_dimension != dimension:
        raise UnitError(
            f"{field}: {raw!r} is a {got_dimension} quantity but this field is {dimension}. "
            f"Expected a unit such as '{CANONICAL_UNITS[dimension]}'."
        )
    return Quantity(value * factor, CANONICAL_UNITS[dimension], dimension, raw)
