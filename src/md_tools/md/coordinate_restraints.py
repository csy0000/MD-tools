"""Biases on a distance and on an angle, for umbrella windows that are not torsions.

WHY THIS IS A SEPARATE MODULE FROM `torsion_restraints`

    Not because a torsion is special, but because each of these is a DIFFERENT OpenMM force with
    its own per-term parameters and its own units: `CustomBondForce` over one distance in
    nanometres, `CustomAngleForce` over one angle in radians, `CustomTorsionForce` over one
    dihedral in radians and wrapped onto the circle. A single class spanning all three would
    branch on kind in its constructor, its adder and its energy expression, which is three
    switches pretending to be one abstraction.

    What IS shared is the design, and it is shared deliberately: one global parameter carrying
    "are the biases on", the strength riding on a per-term `scale`, and the centre stored per
    term. A reader who understands one of these modules understands the other.

UNITS ARE PER KIND AND ARE NEVER IMPLIED

    A distance restraint's centre is in NANOMETRES and its force constant in kJ/mol/nm^2; an
    angle's centre is in DEGREES at the boundary and radians inside the force, with a constant in
    kJ/mol/rad^2. These are not interchangeable and a number alone cannot say which it is: 0.5 is
    a plausible distance in nm and a plausible angle in radians and a very weak force constant.
    So `md_tools.umbrella` refuses `centre_deg` on a distance and `centre_nm` on an angle BY NAME
    rather than accepting whichever it finds -- the failure mode being a window centred somewhere
    nobody chose, sampling happily, with every column in range.

WHAT DOES NOT WRAP

    An angle on [0, 180] and a distance on [0, inf) have no second branch, so neither needs the
    wrapping the torsion form exists to get right. An angle restraint near 0 or 180 is instead
    ILL-CONDITIONED in a way wrapping cannot fix: the gradient of the angle with respect to the
    positions vanishes as the triple becomes collinear, so a restraint there pushes with almost no
    force and the dihedrals measured around that axis become undefined. `md_tools.md.boresch`
    refuses an anchor choice that sits in that region rather than producing a window whose
    orientational terms mean nothing.
"""
from __future__ import annotations

import math
from typing import Sequence

from openmm import CustomAngleForce, CustomBondForce

__all__ = ["AngleRestraint", "DistanceRestraint", "ANGLE_RESTRAINT_PARAMETER",
           "DISTANCE_RESTRAINT_PARAMETER", "HARMONIC", "FLAT_BOTTOM", "RESTRAINT_FORMS"]

#: One name each, so a reader of a serialised System can find them and `set_strength` cannot
#: drift from what the constructor built. Mirrors `TORSION_RESTRAINT_PARAMETER`.
DISTANCE_RESTRAINT_PARAMETER = "distance_restraint_k"
ANGLE_RESTRAINT_PARAMETER = "angle_restraint_k"

HARMONIC = "harmonic"
FLAT_BOTTOM = "flat_bottom"
RESTRAINT_FORMS = (HARMONIC, FLAT_BOTTOM)

_DISTANCE_ENERGY = {
    HARMONIC: f"0.5*{DISTANCE_RESTRAINT_PARAMETER}*scale*(r - r0)^2",
    # From the window EDGE, as in the torsion form: `max(0, |r - r0| - width)` and not an offset
    # harmonic, so the bias is exactly zero inside the window and those samples need no
    # correction at all.
    FLAT_BOTTOM: (f"0.5*{DISTANCE_RESTRAINT_PARAMETER}*scale*"
                  f"max(0, abs(r - r0) - width)^2"),
}

_ANGLE_ENERGY = {
    HARMONIC: f"0.5*{ANGLE_RESTRAINT_PARAMETER}*scale*(theta - theta0)^2",
    FLAT_BOTTOM: (f"0.5*{ANGLE_RESTRAINT_PARAMETER}*scale*"
                  f"max(0, abs(theta - theta0) - width)^2"),
}


class _Restraint:
    """The shape both of these share: one global switch, per-term centre, scale and width."""

    parameter: str = ""

    def __init__(self, system, form: str = HARMONIC) -> None:
        if form not in RESTRAINT_FORMS:
            raise ValueError(
                f"unknown restraint form {form!r}; expected one of {', '.join(RESTRAINT_FORMS)}")
        self.form = form
        force = self._build(form)
        # Zero, so a System carrying this force is unbiased until something asks otherwise. A
        # restraint that began at full strength would bias the equilibration that precedes it.
        force.addGlobalParameter(self.parameter, 0.0)
        self._add_parameters(force, form)
        self._force = force
        self.force_index = system.addForce(force)
        self._system = system

    def _build(self, form):                                   # pragma: no cover - overridden
        raise NotImplementedError

    def _add_parameters(self, force, form):                   # pragma: no cover - overridden
        raise NotImplementedError

    def set_strength(self, simulation, value: float) -> None:
        """Turn the biases on (or off) without rebuilding the System or the Context."""
        simulation.context.setParameter(self.parameter, float(value))

    def __repr__(self) -> str:                                # pragma: no cover - diagnostics
        return f"{type(self).__name__}(form={self.form!r}, terms={self.n_terms})"


class DistanceRestraint(_Restraint):
    """A bias on one or more interatomic distances, in nanometres.

    The dissociation coordinate of a Boresch restraint is one of these: the distance between a
    receptor anchor and a ligand anchor, scanned across windows to build a profile.
    """

    parameter = DISTANCE_RESTRAINT_PARAMETER

    def _build(self, form):
        return CustomBondForce(_DISTANCE_ENERGY[form])

    def _add_parameters(self, force, form):
        force.addPerBondParameter("r0")
        force.addPerBondParameter("scale")
        if form == FLAT_BOTTOM:
            force.addPerBondParameter("width")

    def add_distance(self, atoms: Sequence[int], centre_nm: float,
                     half_width_nm: float = 0.0, scale: float = 1.0) -> int:
        """One restrained distance. `centre_nm` and `half_width_nm` are NANOMETRES."""
        pair = [int(a) for a in atoms]
        if len(pair) != 2:
            raise ValueError(f"a distance restraint needs exactly two atoms; got {pair}")
        if len(set(pair)) != 2:
            raise ValueError(f"a distance restraint needs two distinct atoms; got {pair}")
        if float(centre_nm) < 0.0:
            raise ValueError(
                f"a distance restraint centre must not be negative; got {centre_nm} nm. A "
                f"negative separation is not a geometry, and squaring it would centre the window "
                f"on its absolute value without saying so")
        values = [float(centre_nm), float(scale)]
        if self.form == FLAT_BOTTOM:
            values.append(float(half_width_nm))
        return self._force.addBond(pair[0], pair[1], values)

    @property
    def n_terms(self) -> int:
        return self._force.getNumBonds()


class AngleRestraint(_Restraint):
    """A bias on one or more angles. Centres are given in DEGREES and stored in radians.

    Two of these hold the orientation of a Boresch restraint. Converted at the boundary rather
    than in the caller, so a configuration is written in the units a person thinks in while the
    force does its arithmetic in the units OpenMM defines.
    """

    parameter = ANGLE_RESTRAINT_PARAMETER

    def _build(self, form):
        return CustomAngleForce(_ANGLE_ENERGY[form])

    def _add_parameters(self, force, form):
        force.addPerAngleParameter("theta0")
        force.addPerAngleParameter("scale")
        if form == FLAT_BOTTOM:
            force.addPerAngleParameter("width")

    def add_angle(self, atoms: Sequence[int], centre_degrees: float,
                  half_width_degrees: float = 0.0, scale: float = 1.0) -> int:
        """One restrained angle, `atoms` in order i-j-k with j the vertex."""
        triple = [int(a) for a in atoms]
        if len(triple) != 3:
            raise ValueError(f"an angle restraint needs exactly three atoms; got {triple}")
        if len(set(triple)) != 3:
            raise ValueError(f"an angle restraint needs three distinct atoms; got {triple}")
        centre = float(centre_degrees)
        if not 0.0 <= centre <= 180.0:
            raise ValueError(
                f"an angle restraint centre must lie on [0, 180] degrees; got {centre}. An angle "
                f"between three atoms is unsigned and bounded, so a centre outside that range "
                f"names no geometry -- unlike a torsion, it does not wrap onto one")
        values = [math.radians(centre), float(scale)]
        if self.form == FLAT_BOTTOM:
            values.append(math.radians(float(half_width_degrees)))
        return self._force.addAngle(triple[0], triple[1], triple[2], values)

    @property
    def n_terms(self) -> int:
        return self._force.getNumAngles()
