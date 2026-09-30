"""Reading `cv.yaml`: the schema, the selectors, and every way it is refused.

WHY SO STRICT

    A collective variable is the thing the run is FOR. Every other output can be recomputed from
    the trajectory; a CV series that names the wrong four atoms cannot be told apart from one that
    names the right four, because both are plausible numbers in the right range with the right
    column heading. There is no downstream check that catches it.

    So there is no first-match, no fallback, and no default. A selector that matches two atoms is
    refused rather than resolved to the first; a name that repeats is refused rather than
    suffixed; an index out of range is refused rather than clamped. Every one of those is a case
    where guessing produces a file that looks exactly like the correct one.

WHAT A `type` MAY BE, AND WHERE THAT IS WRITTEN DOWN

    `torsion`, `angle` and `distance`. `type` is required and any other value is refused BY NAME
    with the schema version it was refused under -- so an entry written against a future schema
    fails loudly on an old build instead of being ignored into a silently shorter CSV.

    Everything that follows from the choice of kind -- how many atoms it needs, what its units
    are, whether it wraps -- is in `CV_KINDS` and nowhere else. `SUPPORTED_TYPES` is derived from
    that table rather than listed beside it, because a list of names that has to be kept in step
    with a table of properties is two statements of one fact.

    `distance` and `angle` arrived for Boresch-type restraints, where a dissociation coordinate
    is a distance and the five orientational terms that hold the pathway are angles and torsions.
    A torsion-only `cv.yaml` resolves to exactly the same variables, on the same atoms, with
    the same digest -- the digest is over the FILE, so no record that binds it moves. Its sidecar
    is NOT byte-identical: every column now states its own units, and its periodic ones their own
    wrapping, because the per-column value is authoritative and the file-level one is a derived
    convenience. A reader that consulted only the file-level keys still finds them, unchanged,
    whenever they are true of every column.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The only `schema_version` this build reads.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CVKind:
    """What one kind of collective variable IS: its arity, its units, its wrapping.

    ONE TABLE, ASKED BY EVERYONE. `units` and `wrapping` used to be literals written in three
    places -- the definition record, a ladder's per-state sidecar, and the ladder's
    append-compatibility guard, which compared them against `"degrees"` and `"[-180, 180)"` BY
    VALUE. Three writers of one field with nothing comparing them is the shape that has already
    cost this project a doubled step count and a completion record missing a key the periodic
    path supplied; with a second kind of variable it would cost a campaign that samples correctly
    and is then refused at completion, or mid-campaign on the next append.

    `wrapping` is None for a quantity with no periodic convention. A distance does not wrap, and
    the honest record of that is an ABSENT key rather than a null or an empty string, both of
    which read as a convention whose name happens to be blank.
    """

    atoms: int
    units: str
    wrapping: str | None
    #: What the slots MEAN, quoted in refusals so a message says more than how many are needed.
    order: str

    @property
    def periodic(self) -> bool:
        return self.wrapping is not None


#: Every `type` a `cv.yaml` may name, and everything that follows from the choice.
CV_KINDS: dict[str, CVKind] = {
    "torsion": CVKind(atoms=4, units="degrees", wrapping="[-180, 180)",
                      order="bonding order i-j-k-l"),
    "angle": CVKind(atoms=3, units="degrees", wrapping=None,
                    order="bonding order i-j-k, with j the vertex"),
    "distance": CVKind(atoms=2, units="nanometers", wrapping=None,
                       order="the two atoms it is measured between"),
}

#: The types this build evaluates. Derived from the table, so the two cannot disagree.
SUPPORTED_TYPES = tuple(CV_KINDS)

#: A CV name is a CSV column header and a key in a resolved sidecar. Restricting it to this keeps
#: it from needing quoting in one format and escaping in another -- a comma or a newline inside a
#: column name is a file that different readers disagree about.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_ENTRY_KEYS = frozenset({"name", "type", "atom_indices", "atoms"})
_SELECTOR_KEYS = frozenset({"chain", "residue", "atom"})
_DOCUMENT_KEYS = frozenset({"schema_version", "collective_variables"})


class CVDefinitionError(ValueError):
    """A `cv.yaml` was refused. The message is addressed to whoever wrote it."""


@dataclass(frozen=True)
class CollectiveVariable:
    """One resolved variable: a name, its kind, and the atom indices it is measured on."""

    name: str
    indices: tuple[int, ...]
    #: A key of `CV_KINDS`. Defaulted to `torsion` so the many places that build one of these
    #: positionally, from when a torsion was the only kind, keep meaning what they meant.
    kind: str = "torsion"
    #: The selectors as written, when the entry used them. Kept for the sidecar so a reader can
    #: see WHY these indices, not merely which -- an index list alone cannot be checked against a
    #: topology by a person reading the output later.
    selectors: tuple[dict[str, str], ...] | None = None

    @property
    def spec(self) -> CVKind:
        return CV_KINDS[self.kind]

    @property
    def units(self) -> str:
        return self.spec.units

    @property
    def wrapping(self) -> str | None:
        return self.spec.wrapping


#: The name this class had when a torsion was the only kind there was. Kept because it is part of
#: the import API and appears in tests and in other sessions' code; it is the same class, not a
#: torsion-only subset of it.
TorsionCV = CollectiveVariable


@dataclass(frozen=True)
class CVDefinition:
    """A whole validated `cv.yaml`, with its digest and its resolved atom mapping.

    Frozen and content-addressed on purpose: this object is bound into run fingerprints, so two
    runs that resolve the same definition must produce byte-identical provenance, and a run whose
    definition changed must be unable to continue into the same output.
    """

    schema_version: int
    variables: tuple[TorsionCV, ...]
    #: sha256 of the exact bytes that were parsed -- not of a re-serialisation of the parsed
    #: document, which would erase a difference in the file that this build happens to ignore.
    digest: str
    source: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(cv.name for cv in self.variables)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(cv.kind for cv in self.variables)

    def agreed(self, field: str):
        """The one value of `field` every column shares, or None when they differ.

        THE PER-COLUMN VALUE IS AUTHORITATIVE and this is a derived convenience. A file-level
        `units` is written only when it is true of every column, so a reader consulting either
        one cannot be silently wrong -- which is a stronger property than writing both and
        trusting that they agree.
        """
        if not self.variables:
            return None
        values = {getattr(cv, field) for cv in self.variables}
        return values.pop() if len(values) == 1 else None

    def resolved(self) -> dict[str, Any]:
        """The sidecar body: everything a reader needs to interpret the CSV without the topology."""
        agreed_units = self.agreed("units")
        agreed_wrapping = self.agreed("wrapping")
        body: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "definition_sha256": self.digest,
        }
        # The file-level keys, when and only when they describe every column: a torsion-only
        # definition still carries both, with the values it always had, and a mixed one omits
        # them rather than picking a winner. They are DERIVED -- the per-column values below are
        # authoritative -- so a reader consulting either cannot be silently wrong. `wrapping` is
        # omitted for an unwrapped quantity for the same reason it is None in the table: there is
        # no convention to name, and a null would read as one whose name is blank.
        if agreed_units is not None:
            body["units"] = agreed_units
        if agreed_wrapping is not None:
            body["wrapping"] = agreed_wrapping
        if any(cv.spec.periodic for cv in self.variables):
            body["periodic_convention"] = (
                "triclinic minimum image applied to the three sequential bond vectors")
            body["sign_convention"] = (
                "IUPAC/MDTraj: positive is clockwise looking along j -> k")
        body["columns"] = list(self.names)
        body["collective_variables"] = [
            {"name": cv.name, "type": cv.kind, "atom_indices": list(cv.indices),
             "units": cv.units,
             # ABSENT, not null, for a quantity that does not wrap.
             **({"wrapping": cv.wrapping} if cv.wrapping is not None else {}),
             **({"atoms": [dict(s) for s in cv.selectors]} if cv.selectors else {})}
            for cv in self.variables]
        return body


def _require_mapping(value, *, what: str) -> dict:
    if not isinstance(value, dict):
        raise CVDefinitionError(f"{what}: expected a mapping, got {type(value).__name__}")
    return value


def _unknown(keys, allowed, *, what: str) -> None:
    extra = sorted(set(keys) - set(allowed))
    if extra:
        raise CVDefinitionError(
            f"{what}: unknown field(s) {extra}. Accepted here: {sorted(allowed)}. A field that is "
            f"silently ignored is a definition whose author believes it took effect")


def _selector_text(selector: dict[str, str]) -> str:
    return f"chain {selector['chain']}, residue {selector['residue']}, atom {selector['atom']}"


def resolve_selector(topology, selector: dict[str, str], *, where: str) -> int:
    """The ONE atom this selector names. Refuses zero matches and refuses two.

    `residue` is compared as a STRING against the topology's own residue id, because a PDB residue
    identifier is not an integer -- it may carry an insertion code, and `"52A"` and `52` are
    different residues that `int()` would conflate. An insertion code is therefore supported
    exactly to the extent the topology represents it unambiguously, and no further.
    """
    chain_wanted = str(selector["chain"])
    residue_wanted = str(selector["residue"])
    atom_wanted = str(selector["atom"])

    matches: list[int] = []
    for chain in topology.chains():
        if str(getattr(chain, "id", "")) != chain_wanted:
            continue
        for residue in chain.residues():
            if str(getattr(residue, "id", "")) != residue_wanted:
                continue
            for atom in residue.atoms():
                if str(atom.name) == atom_wanted:
                    matches.append(int(atom.index))
    if not matches:
        raise CVDefinitionError(
            f"{where}: no atom in the topology matches {_selector_text(selector)}. Chain ids and "
            f"residue ids are compared exactly as the topology spells them")
    if len(matches) > 1:
        raise CVDefinitionError(
            f"{where}: {_selector_text(selector)} matches {len(matches)} atoms "
            f"(indices {matches[:6]}). A selector must name exactly one atom -- resolving this to "
            f"the first match would silently measure a torsion nobody asked for")
    return matches[0]


def _parse_entry(entry: Any, *, position: int, topology, particles: int | None) -> TorsionCV:
    where = f"collective_variables[{position}]"
    entry = _require_mapping(entry, what=where)
    _unknown(entry, _ENTRY_KEYS, what=where)

    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise CVDefinitionError(
            f"{where}: name must be a string of letters, digits and underscores beginning with a "
            f"letter (it becomes a CSV column heading); got {name!r}")
    where = f"collective_variables[{position}] ({name})"

    kind = entry.get("type")
    if kind not in SUPPORTED_TYPES:
        raise CVDefinitionError(
            f"{where}: type {kind!r} is not supported by cv.yaml schema version {SCHEMA_VERSION}, "
            f"which defines {list(SUPPORTED_TYPES)} only. A later schema version may add it; this "
            f"build refuses it rather than omitting the column")
    spec = CV_KINDS[kind]

    has_indices = "atom_indices" in entry
    has_atoms = "atoms" in entry
    if has_indices == has_atoms:
        raise CVDefinitionError(
            f"{where}: give exactly one of atom_indices or atoms"
            + (" -- both were given, and they could disagree" if has_indices else
               " -- neither was given, so there is nothing to measure"))

    selectors = None
    if has_indices:
        raw = entry["atom_indices"]
        if not isinstance(raw, list) or len(raw) != spec.atoms:
            raise CVDefinitionError(
                f"{where}: a {kind} needs exactly {spec.atoms} zero-based atom_indices, in "
                f"{spec.order}; got {raw!r}")
        indices = []
        for index in raw:
            if isinstance(index, bool) or not isinstance(index, int):
                raise CVDefinitionError(
                    f"{where}: atom_indices must be integers; got {index!r}")
            if index < 0:
                raise CVDefinitionError(
                    f"{where}: atom_indices are zero-based, so {index} is out of range")
            if particles is not None and index >= particles:
                raise CVDefinitionError(
                    f"{where}: atom index {index} is out of range for a topology with "
                    f"{particles} atoms (valid: 0..{particles - 1})")
            indices.append(int(index))
    else:
        raw = entry["atoms"]
        if not isinstance(raw, list) or len(raw) != spec.atoms:
            raise CVDefinitionError(
                f"{where}: a {kind} needs exactly {spec.atoms} selectors, in {spec.order}; got "
                f"{len(raw) if isinstance(raw, list) else type(raw).__name__}")
        if topology is None:
            raise CVDefinitionError(
                f"{where}: uses atom selectors, which can only be resolved against a topology, "
                f"and none was supplied to resolve them against")
        selectors_built = []
        indices = []
        for slot, selector in enumerate(raw):
            at = f"{where}.atoms[{slot}]"
            selector = _require_mapping(selector, what=at)
            _unknown(selector, _SELECTOR_KEYS, what=at)
            missing = sorted(_SELECTOR_KEYS - set(selector))
            if missing:
                raise CVDefinitionError(
                    f"{at}: a selector must give all of chain, residue and atom explicitly; "
                    f"missing {missing}. A partial selector would match by accident")
            for key in sorted(_SELECTOR_KEYS):
                if not isinstance(selector[key], (str, int)) or isinstance(selector[key], bool):
                    raise CVDefinitionError(
                        f"{at}.{key}: expected a string identifier, got {selector[key]!r}")
            selector = {key: str(selector[key]) for key in sorted(_SELECTOR_KEYS)}
            selectors_built.append(selector)
            indices.append(resolve_selector(topology, selector, where=at))
        selectors = tuple(selectors_built)

    if len(set(indices)) != spec.atoms:
        raise CVDefinitionError(
            f"{where}: the {spec.atoms} atoms of a {kind} must be distinct; resolved to "
            f"{indices}")
    return CollectiveVariable(name=name, indices=tuple(indices), kind=kind, selectors=selectors)


def parse_cv_definition(text: str, *, source: str = "cv.yaml", topology=None,
                        particles: int | None = None) -> CVDefinition:
    """Validate `cv.yaml` text into a `CVDefinition`, or refuse it.

    `topology` is required only when an entry uses `atoms` selectors. `particles` bounds
    `atom_indices`; pass it whenever it is known, since an out-of-range index is otherwise not
    discovered until the first evaluation, by which time output exists.
    """
    from ..build.strict import ConfigError, load_yaml_strictly

    try:
        document = load_yaml_strictly(text, source=source)
    except ConfigError as refusal:
        raise CVDefinitionError(str(refusal)) from None

    document = _require_mapping(document, what=source)
    _unknown(document, _DOCUMENT_KEYS, what=source)

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CVDefinitionError(
            f"{source}: schema_version must be {SCHEMA_VERSION}; got {version!r}. This build "
            f"reads version {SCHEMA_VERSION} only, and refuses rather than guessing which fields "
            f"of another version it would understand")

    entries = document.get("collective_variables")
    if not isinstance(entries, list) or not entries:
        raise CVDefinitionError(
            f"{source}: collective_variables must be a non-empty list. A definition file that "
            f"defines nothing would enable reporting and produce a CSV with no columns")

    variables = [_parse_entry(entry, position=position, topology=topology, particles=particles)
                 for position, entry in enumerate(entries)]

    seen: dict[str, int] = {}
    for position, cv in enumerate(variables):
        if cv.name in seen:
            raise CVDefinitionError(
                f"{source}: two collective variables are both named {cv.name!r} (entries "
                f"{seen[cv.name]} and {position}). Names are CSV column headings and must be "
                f"unique -- one would overwrite the other in every reader")
        seen[cv.name] = position

    return CVDefinition(
        schema_version=SCHEMA_VERSION, variables=tuple(variables),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(), source=source)


def load_cv_definition(path, *, topology=None, particles: int | None = None) -> CVDefinition:
    """`parse_cv_definition` over a file, digesting the exact bytes on disk."""
    path = Path(path)
    if not path.is_file():
        raise CVDefinitionError(f"{path}: no such collective-variable definition file")
    return parse_cv_definition(path.read_text(encoding="utf-8"), source=str(path),
                               topology=topology, particles=particles)
