"""What ensemble a run is in, decided by solvation mode rather than by a global default.

The tree this replaces was internally contradictory in four places at once: the active default said
production was NPT, the generated stage path attached a barostat to ``cMD_1`` and ran NPT, legacy
``run_md()`` rejected every ensemble except NVT, and ``run_rest2_remd()`` rejected NPT. Each of the
four was individually defensible and no two agreed.

The reason a single default cannot fix that is physical, not organisational. **An ensemble label is
a property of the system, not a preference.** An explicit-solvent box has a volume and a pressure,
so NPT is meaningful and is what the barostat implements. An implicit-solvent system has no box, no
volume, and no pressure: there is nothing for a barostat to act on and nothing for ``V`` to denote.
Labelling it "NVT" is not a conservative choice, it is a false statement -- NVT fixes a volume, and
this system has none. Labelling it "NPT" is worse. So implicit runs are named for what they are:
nonperiodic constant-temperature dynamics.

That is why the resolution here is a function of ``solvation_mode`` and not a new default value. A
default is a value that applies until overridden; this is a fact that differs between two modes and
is never a matter of choice in either.
"""

from __future__ import annotations

from typing import Iterable, Optional

__all__ = [
    "EXPLICIT_PRODUCTION_ENSEMBLE",
    "IMPLICIT_PRODUCTION_ENSEMBLE",
    "ENSEMBLE_BY_SOLVATION_MODE",
    "EnsembleError",
    "canonical_ensemble",
    "ensemble_is_barostatted",
    "validate_ensemble",
]

#: Explicit solvent: constant particle number, pressure and temperature. One `MonteCarloBarostat`
#: at the physical pressure; the box fluctuates.
EXPLICIT_PRODUCTION_ENSEMBLE = "NPT"

#: Implicit solvent: constant temperature, no volume and no pressure. Deliberately not spelled
#: "NVT" -- the V would be naming a quantity the system does not have. Written out in words rather
#: than as an acronym so that it cannot be mistaken for one of the standard three.
IMPLICIT_PRODUCTION_ENSEMBLE = "nonperiodic-constant-temperature"

ENSEMBLE_BY_SOLVATION_MODE = {
    "explicit": EXPLICIT_PRODUCTION_ENSEMBLE,
    "implicit": IMPLICIT_PRODUCTION_ENSEMBLE,
}

#: Accepted spellings for the implicit ensemble, so an older configuration or a hand-written one
#: does not fail on punctuation. "NVT" is NOT among them: silently accepting it would let a
#: configuration keep asserting a fixed volume for a system that has none.
_IMPLICIT_ALIASES = frozenset({
    "nonperiodic-constant-temperature",
    "nonperiodic_constant_temperature",
    "nonperiodic constant temperature",
    "constant-temperature",
})

_EXPLICIT_ALIASES = frozenset({"npt"})


class EnsembleError(ValueError):
    """An ensemble that the solvation mode cannot have.

    A `ValueError` subclass so existing `except ValueError` handlers keep working, and a distinct
    type so a caller that wants to report this specific confusion can catch it precisely.
    """


def canonical_ensemble(solvation_mode: str) -> str:
    """The one ensemble this solvation mode can be in.

    Not "the default for" -- the only one. There is no explicit-solvent production protocol in this
    repository that fixes the box, and no implicit protocol that has a box to fix.
    """
    mode = str(solvation_mode).strip().lower()
    try:
        return ENSEMBLE_BY_SOLVATION_MODE[mode]
    except KeyError:
        raise EnsembleError(
            f"unknown solvation mode {solvation_mode!r}; expected one of "
            f"{sorted(ENSEMBLE_BY_SOLVATION_MODE)}"
        ) from None


def ensemble_is_barostatted(solvation_mode: str) -> bool:
    """Whether a Context in this mode carries a barostat.

    The single place that answers "should this stage get a barostat?", so that a stage list and a
    force-construction path cannot drift apart -- which is precisely how explicit REST2 production
    came to run at fixed volume while its configuration said otherwise.
    """
    return canonical_ensemble(solvation_mode) == EXPLICIT_PRODUCTION_ENSEMBLE


def validate_ensemble(ensemble: Optional[str], solvation_mode: str, *,
                      field: str = "production.ensemble") -> str:
    """Return the canonical ensemble, or raise explaining the impossible combination.

    `None` resolves to the canonical value rather than raising: an absent field is a configuration
    that has not stated an opinion, and there is only one correct answer per mode. A *stated* wrong
    value is a different matter -- it is an assertion about physics that the system contradicts, and
    it is refused with both the dotted path and the value, before any System or run directory
    exists.
    """
    expected = canonical_ensemble(solvation_mode)
    if ensemble is None:
        return expected

    given = str(ensemble).strip()
    normalised = given.lower().replace("_", "-").replace(" ", "-")
    aliases = _EXPLICIT_ALIASES if expected == EXPLICIT_PRODUCTION_ENSEMBLE else _IMPLICIT_ALIASES
    if normalised in {a.lower().replace("_", "-").replace(" ", "-") for a in aliases}:
        return expected

    if expected == IMPLICIT_PRODUCTION_ENSEMBLE and normalised in {"nvt", "npt"}:
        raise EnsembleError(
            f"{field} is {given!r}, but the solvation mode is implicit. An implicit-solvent System "
            f"has no periodic box, so it has neither a volume to hold fixed nor a pressure to hold "
            f"constant. Use {expected!r} -- or omit the field, which resolves to it. This is "
            f"refused rather than ignored because the label would otherwise assert a physical "
            f"quantity the System does not have."
        )
    if expected == EXPLICIT_PRODUCTION_ENSEMBLE and normalised == "nvt":
        raise EnsembleError(
            f"{field} is {given!r}, but explicit-solvent production in this repository is "
            f"{expected!r}: a barostat is attached to every production stage and the box "
            f"fluctuates. Fixing the box would freeze a volume estimated from a finite "
            f"equilibration window. Use {expected!r}, or omit the field."
        )
    raise EnsembleError(
        f"{field} is {given!r}, which is not a recognised ensemble for solvation mode "
        f"{solvation_mode!r}; expected {expected!r}"
    )


def assert_no_pressure_settings(keys: Iterable[str], solvation_mode: str) -> None:
    """Refuse pressure/barostat settings on a mode that cannot use them.

    Separate from `validate_ensemble` because a configuration can name the right ensemble and still
    carry a stray `pressure`, which would otherwise be silently dropped.
    """
    if ensemble_is_barostatted(solvation_mode):
        return
    offenders = sorted(k for k in keys
                       if any(t in str(k).lower() for t in ("pressure", "barostat")))
    if offenders:
        raise EnsembleError(
            f"solvation mode is implicit, so {offenders} cannot apply: there is no periodic volume "
            f"for a barostat to act on. Remove them rather than leaving them to be ignored."
        )
