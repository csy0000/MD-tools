"""Alchemical paths: a scalar progress coordinate `s` mapped to NAMED lambda components.

A Hamiltonian state is a set of named coordinates (shared contracts, section 5):

    lambda_sterics: 0.4
    lambda_electrostatics: 0.0

A PATH maps `s in [0, 1]` onto those components and states the orientation: `s = 0` is endpoint
A, `s = 1` is endpoint B, and the record says which physical end each one is. It is piecewise
linear between KNOTS, which is what a staged schedule is ("electrostatics first, then sterics" has
a knot where the first stage ends).

Why the knots matter to thermodynamic integration. Along a staged path `d lambda_k / ds` jumps at
a knot, so `dU/ds = sum_k (dU/d lambda_k)(d lambda_k/ds)` has DIFFERENT left and right values
there. Trapezoidal integration across a knot with one value of `dU/ds` is wrong by a finite amount
that no amount of sampling removes. This module therefore gives each segment its own slopes, and
refuses a set of windows that does not place a window on every knot (`require_knots_sampled`), so
TI can integrate segment by segment with one-sided derivatives at the ends of each.

Names are checked here, once. `lambda` alone is the AIS mixing parameter
(`md_tools.ais.two_state.LAMBDA_PARAMETER`) and `tau` is REST2 solute scaling; neither may name an
alchemical component, because a name means one thing.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

PATH_SCHEMA = "md-tools-alchemical-path/1"
INTERPOLATION = "piecewise-linear"

#: Component names are `lambda_<word>`. The bare `lambda` and `tau` are owned elsewhere.
_COMPONENT = re.compile(r"^lambda_[a-z][a-z0-9_]*$")
_RESERVED = {
    "lambda": "`lambda` is the two-state AIS mixing parameter (md_tools.ais.two_state); an "
              "alchemical component needs its own name, e.g. lambda_sterics",
    "tau": "`tau` is REST2 solute scaling; it is not an alchemical component",
}

#: Two floats closer than this are the same progress coordinate. The schedule is written by a
#: person to a few decimals; this only absorbs binary representation error.
S_TOLERANCE = 1e-9


class PathError(ValueError):
    """A path or window set that cannot be integrated correctly."""


def check_component_name(name: str) -> str:
    if name in _RESERVED:
        raise PathError(f"alchemical component {name!r} refused: {_RESERVED[name]}")
    if not _COMPONENT.match(name):
        raise PathError(f"alchemical component {name!r} refused: a component is named "
                        f"lambda_<word> in lower case, e.g. lambda_sterics")
    return name


@dataclass(frozen=True)
class Knot:
    s: float
    components: Mapping[str, float]


@dataclass(frozen=True)
class Segment:
    index: int
    s_start: float
    s_end: float
    #: d lambda_k / ds, constant on the segment.
    slopes: Mapping[str, float]


@dataclass(frozen=True)
class AlchemicalPath:
    """A piecewise-linear path from endpoint A (`s = 0`) to endpoint B (`s = 1`)."""

    knots: tuple[Knot, ...]
    endpoint_a: str
    endpoint_b: str
    #: What the path does, in words a reader can check, e.g. "decouple ligand L from its
    #: environment: A is fully interacting, B is non-interacting".
    description: str = ""
    components: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if len(self.knots) < 2:
            raise PathError("a path needs at least two knots, s = 0 and s = 1")
        names = tuple(sorted(self.knots[0].components))
        if not names:
            raise PathError("a path must move at least one named lambda component")
        for name in names:
            check_component_name(name)
        for knot in self.knots:
            if tuple(sorted(knot.components)) != names:
                raise PathError(f"knot s={knot.s} names components {sorted(knot.components)}, "
                                f"the first knot names {list(names)}: every knot must set the "
                                f"same components")
            for name, value in knot.components.items():
                if not (0.0 <= float(value) <= 1.0):
                    raise PathError(f"{name} = {value} at s={knot.s} lies outside [0, 1]")
        s = [k.s for k in self.knots]
        if abs(s[0]) > S_TOLERANCE or abs(s[-1] - 1.0) > S_TOLERANCE:
            raise PathError(f"a path runs from s = 0 to s = 1; these knots run {s[0]} -> {s[-1]}")
        if any(b - a <= S_TOLERANCE for a, b in zip(s, s[1:])):
            raise PathError(f"knot positions must strictly increase; got {s}")
        if not self.endpoint_a or not self.endpoint_b:
            raise PathError("a path names both endpoints: which physical state is A (s = 0) "
                            "and which is B (s = 1)")
        object.__setattr__(self, "components", names)

    # ------------------------------------------------------------------ geometry
    @property
    def segments(self) -> tuple[Segment, ...]:
        out = []
        for i, (a, b) in enumerate(zip(self.knots, self.knots[1:])):
            width = b.s - a.s
            slopes = {n: (float(b.components[n]) - float(a.components[n])) / width
                      for n in self.components}
            out.append(Segment(i, a.s, b.s, slopes))
        return tuple(out)

    def components_at(self, s: float) -> dict[str, float]:
        s = float(s)
        if s < -S_TOLERANCE or s > 1.0 + S_TOLERANCE:
            raise PathError(f"s = {s} lies outside [0, 1]")
        for a, b in zip(self.knots, self.knots[1:]):
            if s <= b.s + S_TOLERANCE:
                t = min(max((s - a.s) / (b.s - a.s), 0.0), 1.0)
                return {n: float(a.components[n]) + t * (float(b.components[n])
                                                          - float(a.components[n]))
                        for n in self.components}
        raise AssertionError("unreachable")

    def segments_touching(self, s: float) -> list[Segment]:
        """The segments a window at `s` belongs to: one in a segment's interior, two on an
        interior knot, one at either end of the path."""
        return [seg for seg in self.segments
                if seg.s_start - S_TOLERANCE <= s <= seg.s_end + S_TOLERANCE]

    def require_knots_sampled(self, s_values: Iterable[float]) -> None:
        s_values = sorted(float(s) for s in s_values)
        missing = [k.s for k in self.knots
                   if not any(abs(k.s - s) <= S_TOLERANCE for s in s_values)]
        if missing:
            raise PathError(
                f"no window at knot(s) s = {missing}. The path changes slope there, so dU/ds has "
                f"different values on either side and cannot be integrated across it from "
                f"neighbouring windows; add a window at every knot")

    # ------------------------------------------------------------------ record
    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PATH_SCHEMA,
            "interpolation": INTERPOLATION,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "orientation": "s = 0 is endpoint_a, s = 1 is endpoint_b",
            "description": self.description,
            "components": list(self.components),
            "knots": [{"s": float(k.s),
                       "components": {n: float(k.components[n]) for n in self.components}}
                      for k in self.knots],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AlchemicalPath":
        if record.get("schema") != PATH_SCHEMA:
            raise PathError(f"path record schema {record.get('schema')!r}; this reader reads "
                            f"{PATH_SCHEMA!r}")
        if record.get("interpolation") != INTERPOLATION:
            raise PathError(f"path interpolation {record.get('interpolation')!r} is not "
                            f"implemented; only {INTERPOLATION!r}")
        return cls(knots=tuple(Knot(float(k["s"]), dict(k["components"]))
                               for k in record["knots"]),
                   endpoint_a=record["endpoint_a"], endpoint_b=record["endpoint_b"],
                   description=record.get("description", ""))

    def digest(self) -> str:
        blob = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def linear_path(components: Sequence[str], *, endpoint_a: str, endpoint_b: str,
                description: str = "") -> AlchemicalPath:
    """Every component 0 -> 1 together."""
    return AlchemicalPath(
        knots=(Knot(0.0, {c: 0.0 for c in components}), Knot(1.0, {c: 1.0 for c in components})),
        endpoint_a=endpoint_a, endpoint_b=endpoint_b, description=description)


def staged_path(stages: Sequence[tuple[str, float]], *, endpoint_a: str, endpoint_b: str,
                description: str = "") -> AlchemicalPath:
    """Components moved one after another: `[("lambda_electrostatics", 0.5), ("lambda_sterics",
    1.0)]` moves electrostatics 0 -> 1 over s in [0, 0.5], then sterics over [0.5, 1]."""
    names = [n for n, _ in stages]
    if len(set(names)) != len(names):
        raise PathError(f"a staged path moves each component once; got {names}")
    current = {n: 0.0 for n in names}
    knots = [Knot(0.0, dict(current))]
    for name, s_end in stages:
        current[name] = 1.0
        knots.append(Knot(float(s_end), dict(current)))
    return AlchemicalPath(tuple(knots), endpoint_a=endpoint_a, endpoint_b=endpoint_b,
                          description=description)
