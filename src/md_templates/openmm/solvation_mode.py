"""Explicit water or implicit solvent, discriminated once.

Two solvent treatments share almost no configuration. Explicit water needs a box, a water model,
ions, a salt concentration, PME and a real-space cutoff; implicit solvent needs a GB model and a
radius set and has *none* of those. A single flat block that accepts both would silently ignore
whichever half does not apply, and "silently ignored" is the failure this module exists to prevent:
a configuration stating `ionic_strength_molar` under implicit solvent describes an experiment that
was never run, and nothing in the output would say so.

So `solvation.mode` discriminates, and each mode rejects the other's fields by name.

## Implicit defaults, and why they are what they are

`GBn2` with `mbondi3`. The radius set is not a free choice: GBn2 was parameterised against mbondi3,
and pairing it with another set is a different Hamiltonian that still runs. Both are canonicalised
to exactly these spellings on the way in, so a bundle records one name for one thing.

## No barostat, ever

Implicit solvent has no box, so there is no volume to control and pressure is undefined. An NPT
stage or a `pressure` setting is refused before generation rather than ignored at run time.
"""

from __future__ import annotations

__all__ = [
    "EXPLICIT",
    "IMPLICIT",
    "DEFAULT_IMPLICIT_MODEL",
    "DEFAULT_IMPLICIT_RADII",
    "SUPPORTED_IMPLICIT_MODELS",
    "SUPPORTED_RADII",
    "RECOGNISED_BUT_UNVALIDATED_MODELS",
    "RECOGNISED_BUT_UNVALIDATED_RADII",
    "EXPLICIT_ONLY_FIELDS",
    "IMPLICIT_ONLY_FIELDS",
    "SolvationError",
    "resolve_solvation",
]

EXPLICIT = "explicit"
IMPLICIT = "implicit"

DEFAULT_IMPLICIT_MODEL = "GBn2"
DEFAULT_IMPLICIT_RADII = "mbondi3"

#: What the PUBLIC schema accepts. OpenMM exposing HCT/OBC/GBn and ParmEd accepting several radius
#: sets does not make every pair a validated template: the energy identity in this repository is
#: pinned for GBn2 with mbondi3 and for nothing else. Public acceptance is an evidence boundary, so
#: it is drawn where the evidence is, not where the library's capability is.
#:
#: The internal builder in `implicit.py` remains extensible on purpose -- validating another pair
#: should mean adding evidence and widening this tuple, not rewriting the construction.
SUPPORTED_IMPLICIT_MODELS = ("GBn2",)
SUPPORTED_RADII = ("mbondi3",)

#: Recognised by OpenMM/ParmEd but not validated here. Named separately so the refusal can say
#: "not validated" rather than "unknown", which would be untrue and would send a reader looking for
#: a typo.
RECOGNISED_BUT_UNVALIDATED_MODELS = ("HCT", "OBC1", "OBC2", "GBn")
RECOGNISED_BUT_UNVALIDATED_RADII = ("bondi", "mbondi", "mbondi2", "amber6")

#: Fields that only mean something with explicit water. Under implicit solvent each one describes a
#: quantity that does not exist.
EXPLICIT_ONLY_FIELDS = (
    "water_model",
    "box_shape",
    "padding_nm",
    "padding_semantics",
    "cutoff_fit_policy",
    "ionic_strength_molar",
    "positive_ion",
    "negative_ion",
    "neutralize",
    "nonbonded_cutoff_nm",
    "ewald_error_tolerance",
    "minimum_image_margin_nm",
)

#: Fields that only mean something without water.
IMPLICIT_ONLY_FIELDS = ("implicit_model", "radii", "salt_concentration_molar")


class SolvationError(ValueError):
    """A solvation configuration that cannot describe one experiment."""


def _canonical(value, allowed, field: str, recognised=()) -> str:
    """Match case-insensitively, store the canonical spelling.

    `gbn2`, `GBN2` and `GBn2` are the same model, and a bundle that records three spellings for one
    Hamiltonian cannot be compared against another bundle by equality.
    """
    lowered = {item.lower(): item for item in allowed}
    key = str(value).strip().lower()
    if key in lowered:
        return lowered[key]
    if key in {item.lower() for item in recognised}:
        raise SolvationError(
            f"solvation.{field} = {value!r} is recognised but NOT VALIDATED in this repository.\n"
            f"  The implicit-solvent energy identity is pinned for "
            f"{DEFAULT_IMPLICIT_MODEL}/{DEFAULT_IMPLICIT_RADII} and for no other combination, so "
            "accepting this one\n"
            "  would ship a template with no evidence behind it. Public support follows the "
            "evidence, not the library's\n"
            f"  capability. Implemented: {', '.join(allowed)}."
        )
    raise SolvationError(
        f"solvation.{field} = {value!r} is not a recognised {field}; "
        f"implemented: {', '.join(allowed)}")


def resolve_solvation(block: dict | None, *, defaults: dict | None = None) -> dict:
    """Validate and resolve a solvation block into one mode's settings.

    Returns `{"mode", ...}` plus that mode's resolved fields and a `sources` map. Raises
    `SolvationError` naming the offending fields rather than dropping them.
    """
    block = dict(block or {})
    mode = str(block.pop("mode", EXPLICIT)).strip().lower()
    if mode not in (EXPLICIT, IMPLICIT):
        raise SolvationError(
            f"solvation.mode = {mode!r} is not recognised; expected {EXPLICIT!r} or {IMPLICIT!r}")

    if mode == EXPLICIT:
        intruders = sorted(f for f in IMPLICIT_ONLY_FIELDS if f in block)
        if intruders:
            raise SolvationError(
                f"solvation.mode is 'explicit' but the block states {', '.join(intruders)}, which "
                "only apply to implicit solvent.\n"
                "  Set \"mode\": \"implicit\" if that is what you meant. These are not ignored, "
                "because a configuration\n"
                "  that states them describes an experiment that would not be run."
            )
        resolved = dict(defaults or {})
        resolved.update(block)
        sources = {k: ("user input" if k in block else "package default") for k in resolved}
        return {"mode": EXPLICIT, **resolved, "sources": sources}

    intruders = sorted(f for f in EXPLICIT_ONLY_FIELDS if f in block)
    if intruders:
        raise SolvationError(
            f"solvation.mode is 'implicit' but the block states {', '.join(intruders)}.\n"
            "  Implicit solvent has no water, no box, no ions, no PME and no real-space cutoff, so "
            "these describe\n"
            "  quantities that do not exist. Remove them, or use \"mode\": \"explicit\".\n"
            "  Implicit solvent takes only: "
            f"{', '.join(IMPLICIT_ONLY_FIELDS)}."
        )
    unknown = sorted(f for f in block if f not in IMPLICIT_ONLY_FIELDS)
    if unknown:
        raise SolvationError(
            f"solvation states unknown field(s) under implicit solvent: {', '.join(unknown)}.\n"
            f"  Implemented: {', '.join(IMPLICIT_ONLY_FIELDS)}."
        )

    model = block.get("implicit_model", DEFAULT_IMPLICIT_MODEL)
    radii = block.get("radii", DEFAULT_IMPLICIT_RADII)
    resolved = {
        "implicit_model": _canonical(model, SUPPORTED_IMPLICIT_MODELS, "implicit_model",
                                     RECOGNISED_BUT_UNVALIDATED_MODELS),
        "radii": _canonical(radii, SUPPORTED_RADII, "radii", RECOGNISED_BUT_UNVALIDATED_RADII),
        "salt_concentration_molar": float(block.get("salt_concentration_molar", 0.0)),
    }
    if resolved["salt_concentration_molar"] != 0.0:
        raise SolvationError(
            "solvation.salt_concentration_molar is not supported under implicit solvent in this "
            "release.\n"
            "  A Debye-Huckel screening term is a different Hamiltonian from the GBn2/mbondi3 one "
            "this bundle\n"
            "  is validated against, so accepting it would silently change the energy the "
            "validation pins."
        )
    sources = {
        "implicit_model": "user input" if "implicit_model" in block else "package default",
        "radii": "user input" if "radii" in block else "package default",
        "salt_concentration_molar": "package default: no screening",
    }
    return {"mode": IMPLICIT, **resolved, "sources": sources}
