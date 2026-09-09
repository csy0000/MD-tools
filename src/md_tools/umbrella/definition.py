"""Reading `umbrella.yaml`: the schema, and every way it is refused.

The same strictness `cv.definition` applies, for the same reason. A restraint that is silently
misread does not fail; it produces a window biased somewhere other than where its author wrote,
and every downstream number is plausible. So there is no first-match, no fallback, and no default
for anything that decides where the bias sits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: The only `schema_version` this build reads.
SCHEMA_VERSION = 1

#: The forms a restraint may take. Mirrors `md.torsion_restraints.RESTRAINT_FORMS`; a test pins
#: the two together so a form the configuration accepts is always one the runtime can build.
RESTRAINT_FORMS = ("harmonic", "flat_bottom")

_REQUIRED = ("cv", "centre_deg", "force_constant")
_OPTIONAL = ("form", "half_width_deg")


class UmbrellaError(ValueError):
    """An umbrella definition that cannot be used as written."""


@dataclass(frozen=True)
class UmbrellaRestraint:
    """One resolved restraint: which CV, which form, and the numbers that place it."""

    cv: str
    form: str
    centre_deg: float
    force_constant: float
    half_width_deg: float | None
    atom_indices: tuple[int, ...]

    def record(self) -> dict[str, Any]:
        """What the run writes down about this restraint.

        The atom indices are included even though the name determines them: a reader of the
        output should not have to open two files to learn which four atoms were biased.
        """
        out = {"cv": self.cv, "form": self.form, "centre_deg": self.centre_deg,
               "force_constant_kj_mol_rad2": self.force_constant,
               "atom_indices": list(self.atom_indices)}
        if self.half_width_deg is not None:
            out["half_width_deg"] = self.half_width_deg
        return out


def load_umbrella_definition(path, cv_definition) -> tuple[UmbrellaRestraint, ...]:
    """Read `path` and resolve every restraint against `cv_definition`.

    `cv_definition` is the ALREADY-LOADED collective-variable definition the run will report from
    -- passed in rather than read here, so the restraint and the series provably resolve against
    one object rather than two reads of one file that could differ.
    """
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UmbrellaError(f"{path} does not exist") from None
    except yaml.YAMLError as failure:
        raise UmbrellaError(f"{path} is not readable YAML: {failure}") from None
    if not isinstance(document, dict):
        raise UmbrellaError(f"{path} must be a mapping; got {type(document).__name__}")

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UmbrellaError(
            f"{path} declares schema_version {version!r}; this build reads {SCHEMA_VERSION}. "
            f"Refused rather than read on a guess: a future schema may place the same keys "
            f"differently.")

    entries = document.get("restraints")
    if not isinstance(entries, list) or not entries:
        raise UmbrellaError(
            f"{path} has no `restraints` list. Umbrella sampling IS the restraint; a definition "
            f"with none is a cMD run and should say so.")

    known = {variable.name: variable for variable in cv_definition.variables}
    resolved: list[UmbrellaRestraint] = []
    seen: dict[str, int] = {}

    for position, entry in enumerate(entries):
        where = f"{path}: restraints[{position}]"
        if not isinstance(entry, dict):
            raise UmbrellaError(f"{where} must be a mapping; got {type(entry).__name__}")
        unknown = set(entry) - set(_REQUIRED) - set(_OPTIONAL)
        if unknown:
            raise UmbrellaError(
                f"{where} has unknown key(s) {', '.join(sorted(unknown))}. Refused rather than "
                f"ignored: a misspelled `centre_deg` would silently restrain to 0 degrees.")
        for required in _REQUIRED:
            if entry.get(required) is None:
                raise UmbrellaError(f"{where} has no {required}")

        name = str(entry["cv"])
        if name not in known:
            raise UmbrellaError(
                f"{where} restrains {name!r}, which is not defined in the collective-variable "
                f"file. Known: {', '.join(sorted(known)) or '(none)'}. A restraint names a CV; it "
                f"does not define one, so that the biased and the reported quantity cannot "
                f"differ.")
        if name in seen:
            raise UmbrellaError(
                f"{where} restrains {name!r}, which restraints[{seen[name]}] already restrains. "
                f"Two restraints on one variable sum into a bias neither describes; write the "
                f"combined force constant once instead.")
        seen[name] = position

        form = str(entry.get("form") or "harmonic")
        if form not in RESTRAINT_FORMS:
            raise UmbrellaError(
                f"{where} has form {form!r}; expected one of {', '.join(RESTRAINT_FORMS)}")

        force_constant = float(entry["force_constant"])
        if force_constant <= 0.0:
            raise UmbrellaError(
                f"{where} has force_constant {force_constant}, which applies no bias. A window "
                f"with no restraint is an unbiased run wearing a window's name.")

        width = entry.get("half_width_deg")
        if form == "flat_bottom":
            if width is None or float(width) <= 0.0:
                raise UmbrellaError(
                    f"{where} is flat_bottom and needs a positive half_width_deg; got {width!r}. "
                    f"A zero-width bound is a harmonic restraint -- ask for that form instead of "
                    f"expressing it as a degenerate bound.")
            width = float(width)
        elif width is not None:
            raise UmbrellaError(
                f"{where} is {form} and has no half-width, but half_width_deg = {width} was "
                f"given. It would be silently ignored, so it is refused.")

        resolved.append(UmbrellaRestraint(
            cv=name, form=form, centre_deg=float(entry["centre_deg"]),
            force_constant=force_constant, half_width_deg=width,
            atom_indices=tuple(int(i) for i in known[name].indices)))
    return tuple(resolved)


def definition_digest(path) -> str:
    """The sha256 of the definition as read, for the run record."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
