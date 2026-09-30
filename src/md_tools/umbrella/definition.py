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

#: A restraint is written in the units of the variable it restrains, and the KEY NAME carries
#: them. An angular centre is `centre_deg`; a distance's is `centre_nm`. They are not
#: interchangeable and a bare number cannot say which it is -- 0.5 is a plausible distance in
#: nanometres, a plausible angle in radians, and a very weak force constant -- so the wrong
#: spelling for a kind is refused BY NAME rather than accepted and reinterpreted.
_CENTRE_KEY = {"degrees": "centre_deg", "nanometers": "centre_nm"}
_WIDTH_KEY = {"degrees": "half_width_deg", "nanometers": "half_width_nm"}
#: The force constant's units follow the same rule, and are recorded under a key that states them.
_CONSTANT_KEY = {"degrees": "force_constant_kj_mol_rad2",
                 "nanometers": "force_constant_kj_mol_nm2"}

_REQUIRED = ("cv", "force_constant")
_OPTIONAL = ("form", *_CENTRE_KEY.values(), *_WIDTH_KEY.values())


class UmbrellaError(ValueError):
    """An umbrella definition that cannot be used as written."""


@dataclass(frozen=True)
class UmbrellaRestraint:
    """One resolved restraint: which CV, which form, and the numbers that place it."""

    cv: str
    form: str
    #: In the restrained variable's OWN units -- degrees for an angle or torsion, nanometres for
    #: a distance. `units` says which, and every spelling below derives from it.
    centre: float
    force_constant: float
    half_width: float | None
    atom_indices: tuple[int, ...]
    kind: str = "torsion"
    units: str = "degrees"

    @property
    def centre_deg(self) -> float | None:
        """The centre when it is an angular one, else None. The name predates the other kinds."""
        return self.centre if self.units == "degrees" else None

    @property
    def half_width_deg(self) -> float | None:
        return self.half_width if self.units == "degrees" else None

    @property
    def centre_nm(self) -> float | None:
        return self.centre if self.units == "nanometers" else None

    @property
    def half_width_nm(self) -> float | None:
        return self.half_width if self.units == "nanometers" else None

    def record(self) -> dict[str, Any]:
        """What the run writes down about this restraint.

        The atom indices are included even though the name determines them: a reader of the
        output should not have to open two files to learn which atoms were biased. Every numeric
        key names its own units, so a record cannot be read under the wrong ones.
        """
        out = {"cv": self.cv, "kind": self.kind, "form": self.form,
               _CENTRE_KEY[self.units]: self.centre,
               _CONSTANT_KEY[self.units]: self.force_constant,
               "atom_indices": list(self.atom_indices)}
        if self.half_width is not None:
            out[_WIDTH_KEY[self.units]] = self.half_width
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

        # THE UNITS COME FROM THE VARIABLE, and the key must be spelled for them. A `centre_deg`
        # on a distance is not a centre this build can interpret: accepting it would place the
        # window at that number of nanometres, which is nobody's intent and samples perfectly
        # happily. So the right key is required and the wrong one is refused BY NAME.
        variable = known[name]
        units = variable.units
        centre_key = _CENTRE_KEY[units]
        width_key = _WIDTH_KEY[units]
        for other_units, other_key in _CENTRE_KEY.items():
            if other_units != units and entry.get(other_key) is not None:
                raise UmbrellaError(
                    f"{where} restrains {name!r}, which is a {variable.kind} measured in {units}, "
                    f"and gives {other_key} = {entry[other_key]}. Write {centre_key} instead: "
                    f"{other_key} would have to be reinterpreted as {units} to be used at all, "
                    f"and a number alone cannot say which units it was written in.")
        for other_units, other_key in _WIDTH_KEY.items():
            if other_units != units and entry.get(other_key) is not None:
                raise UmbrellaError(
                    f"{where} restrains a {variable.kind} measured in {units} and gives "
                    f"{other_key} = {entry[other_key]}; write {width_key} instead.")
        if entry.get(centre_key) is None:
            raise UmbrellaError(
                f"{where} restrains {name!r}, a {variable.kind} measured in {units}, and has no "
                f"{centre_key}")
        centre = float(entry[centre_key])

        width = entry.get(width_key)
        if form == "flat_bottom":
            if width is None or float(width) <= 0.0:
                raise UmbrellaError(
                    f"{where} is flat_bottom and needs a positive {width_key}; got {width!r}. "
                    f"A zero-width bound is a harmonic restraint -- ask for that form instead of "
                    f"expressing it as a degenerate bound.")
            width = float(width)
        elif width is not None:
            raise UmbrellaError(
                f"{where} is {form} and has no half-width, but {width_key} = {width} was "
                f"given. It would be silently ignored, so it is refused.")

        # Bounds the force can enforce anyway, checked here so the refusal names the
        # configuration rather than surfacing from inside a force three stages later.
        if units == "nanometers" and centre < 0.0:
            raise UmbrellaError(
                f"{where} centres a distance window at {centre} nm. A negative separation is not "
                f"a geometry, and squaring it would place the window at its absolute value "
                f"without saying so.")
        if variable.kind == "angle" and not 0.0 <= centre <= 180.0:
            raise UmbrellaError(
                f"{where} centres an angle window at {centre} degrees. An angle between three "
                f"atoms is unsigned and bounded by [0, 180], so this names no geometry -- unlike "
                f"a torsion, it does not wrap onto one.")

        resolved.append(UmbrellaRestraint(
            cv=name, form=form, centre=centre, force_constant=force_constant,
            half_width=width, kind=variable.kind, units=units,
            atom_indices=tuple(int(i) for i in variable.indices)))
    return tuple(resolved)


def definition_digest(path) -> str:
    """The sha256 of the definition as read, for the run record."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
