"""Biasing restraints on a torsion: the force side of umbrella sampling.

`md_tools.cv.torsion` MEASURES a torsion and deliberately never touches the integrator -- that
separation is what keeps a reported collective variable an observation rather than something the
simulation was pushed towards. This module is the other half, and it is deliberately a different
file: a force that biases the dynamics is not a measurement, and the two must not be reachable
through one import by accident.

TWO FORMS, and they answer different questions.

    harmonic      0.5*k*dtheta^2                    an umbrella WINDOW: hold the torsion near
                                                    theta0 and reweight afterwards
    flat-bottom   0.5*k*max(0, |dtheta| - w)^2      a BOUND: leave the torsion alone inside a
                                                    window and stop it leaving

A harmonic window biases everywhere, including at its own centre, so every sample needs
reweighting. A flat-bottom restraint contributes exactly zero inside its window, so samples there
are from the unbiased ensemble and need no correction -- which is what makes it the right choice
for keeping a molecule in a basin rather than for measuring a profile across one.

THE PERIODIC WRAP IS THE PART THAT GOES WRONG. A torsion lives on a circle, so `theta - theta0` is
only meaningful modulo 2*pi. Without the wrap below, a restraint at +170 degrees pulls a torsion
at -170 degrees the long way round -- through 340 degrees of rotation it should never take -- and
the bias is wrong by a factor that depends on where the molecule happens to be. The wrap is

    dtheta = theta - theta0 - floor((theta - theta0)/(2*pi) + 0.5)*2*pi

which maps the difference into (-pi, pi]. `tests/test_torsion_restraints.py` checks it by energy
rather than by sampling: at theta0 = 170 degrees, E(150) and E(-170) are both 20 degrees away and
must be equal.

A SIGN ERROR HERE SURVIVES THE OBVIOUS TEST. Alanine dipeptide's phi starts at 180 degrees, and
180 is its own negative -- so a dihedral convention checked only at the starting structure agrees
with a negated one. The tests use asymmetric geometries for exactly this reason.

The force follows `PositionalRestraint`'s pattern: added ONCE, before any state is loaded, with
its strength carried in a global parameter. Adding or removing a Force changes the System's force
layout and therefore the checkpoint layout, so a chain whose biased and unbiased stages had
different layouts could not hand a checkpoint from one stage to the next. Present-at-zero is what
keeps the layout identical along the whole chain.
"""
from __future__ import annotations

import math
from typing import Sequence

from openmm import CustomTorsionForce

__all__ = ["TorsionRestraint", "TORSION_RESTRAINT_PARAMETER", "HARMONIC", "FLAT_BOTTOM",
           "RESTRAINT_FORMS"]

#: The global parameter carrying the force constant, in kJ/mol/rad^2. One name, so a reader of a
#: serialised System can find it and `set_strength` cannot drift from what the constructor built.
TORSION_RESTRAINT_PARAMETER = "torsion_restraint_k"

HARMONIC = "harmonic"
FLAT_BOTTOM = "flat_bottom"
RESTRAINT_FORMS = (HARMONIC, FLAT_BOTTOM)

#: Wraps `theta - theta0` into (-pi, pi]. See the module docstring for why this is not optional.
_WRAP = (f"dtheta = theta - theta0 - floor((theta - theta0)/(2*{math.pi}) + 0.5)*2*{math.pi}")

_ENERGY = {
    HARMONIC: f"0.5*{TORSION_RESTRAINT_PARAMETER}*scale*dtheta^2; {_WRAP}",
    # `max(0, ...)` and not an `abs` inside a square: outside the window the penalty must grow
    # from the window EDGE, not from theta0, or the form is just a harmonic with an offset.
    FLAT_BOTTOM: (f"0.5*{TORSION_RESTRAINT_PARAMETER}*scale*max(0, abs(dtheta) - width)^2; "
                  f"{_WRAP}"),
}


class TorsionRestraint:
    """A bias on one or more torsions, whose strength can be changed without rebuilding.

    Every torsion added shares the one global force constant, and carries its own centre
    (`theta0`), its own per-torsion `scale`, and -- for the flat-bottom form -- its own half-width.
    `scale` exists so a single window can weight torsions differently without needing a force each;
    it defaults to 1.0 and most callers never touch it.
    """

    def __init__(self, system, form: str = HARMONIC) -> None:
        if form not in RESTRAINT_FORMS:
            raise ValueError(
                f"unknown torsion restraint form {form!r}; expected one of "
                f"{', '.join(RESTRAINT_FORMS)}")
        self.form = form
        force = CustomTorsionForce(_ENERGY[form])
        # Zero, so a System carrying this force is unbiased until something asks otherwise. A
        # restraint that began at full strength would bias the equilibration that precedes it.
        force.addGlobalParameter(TORSION_RESTRAINT_PARAMETER, 0.0)
        force.addPerTorsionParameter("theta0")
        force.addPerTorsionParameter("scale")
        if form == FLAT_BOTTOM:
            force.addPerTorsionParameter("width")
        self._force = force
        self.force_index = system.addForce(force)
        self._system = system

    def add_torsion(self, atoms: Sequence[int], centre_degrees: float,
                    half_width_degrees: float = 0.0, scale: float = 1.0) -> int:
        """Restrain the torsion through `atoms` towards `centre_degrees`.

        Degrees in, radians stored: every configuration and report in this project speaks degrees
        for torsions, and converting at one boundary is what stops a radian reaching a file that
        claims degrees.
        """
        indices = [int(i) for i in atoms]
        if len(indices) != 4:
            raise ValueError(f"a torsion needs exactly 4 atoms; got {len(indices)}")
        if len(set(indices)) != 4:
            raise ValueError(f"a torsion needs 4 DISTINCT atoms; got {indices}")
        if self.form == FLAT_BOTTOM and half_width_degrees <= 0.0:
            raise ValueError(
                f"a flat-bottom restraint needs a positive half-width; got "
                f"{half_width_degrees}. A zero width is a harmonic restraint -- ask for that "
                f"form instead of expressing it as a degenerate bound.")
        if self.form == HARMONIC and half_width_degrees:
            raise ValueError(
                f"a harmonic restraint has no half-width, but {half_width_degrees} was given. "
                f"It would be silently ignored, so it is refused.")
        parameters = [math.radians(float(centre_degrees)), float(scale)]
        if self.form == FLAT_BOTTOM:
            parameters.append(math.radians(float(half_width_degrees)))
        return self._force.addTorsion(*indices, parameters)

    @property
    def n_torsions(self) -> int:
        return self._force.getNumTorsions()

    def set_strength(self, simulation, kj_per_mol_rad2: float) -> None:
        """Change the force constant on a live Context. 0.0 releases it without removing it."""
        strength = float(kj_per_mol_rad2)
        if strength < 0.0:
            raise ValueError(
                f"a restraint force constant must be >= 0; got {strength}. A negative constant "
                f"pushes the torsion AWAY from its centre, which is not a restraint.")
        simulation.context.setParameter(TORSION_RESTRAINT_PARAMETER, strength)

    def __repr__(self) -> str:                                # pragma: no cover - diagnostics
        return f"TorsionRestraint(form={self.form!r}, torsions={self.n_torsions})"
