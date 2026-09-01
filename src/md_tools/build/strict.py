"""Strict configuration schemas.

Every configuration this package accepts is YAML, is validated against a declared schema, and
**rejects what it does not recognise**. A configuration key that is silently ignored is the most
expensive kind of bug in this domain: the run completes, the log looks healthy, and the number it
produces answers a different question than the one that was asked. A misspelled `timestep_fs` that
falls back to a default costs the whole simulation, and nothing in the output says so.

So the rules here are deliberately unforgiving:

  * an unknown key is an error, and the message names the nearest known key;
  * a value of the wrong type is an error, never a coercion;
  * an out-of-range or non-enumerated value is an error;
  * a declared incompatibility between two otherwise valid values is an error.

The schema doubles as the documentation source: `Field` carries the unit, the default, and the
scientific consequence, and the shipped `.config` examples are generated from the same objects, so
a comment in an example cannot drift away from the rule the code enforces.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """A configuration was rejected. The message is addressed to the person who wrote it."""


_MISSING = object()


class Field:
    """One accepted configuration key.

    `unit` is documentation, not arithmetic: this package stores durations as integer step counts
    and physical constants in their own explicit units, and never converts between them silently.
    """

    def __init__(self, name: str, kind: type | tuple[type, ...], *, default: Any = _MISSING,
                 enum: Sequence[Any] | None = None, minimum: Any = None, maximum: Any = None,
                 unit: str | None = None, doc: str = "", nullable: bool = False) -> None:
        self.name = name
        self.kind = kind
        self.default = default
        self.enum = tuple(enum) if enum is not None else None
        self.minimum = minimum
        self.maximum = maximum
        self.unit = unit
        self.doc = doc
        self.nullable = nullable

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    def check(self, value: Any, *, where: str) -> Any:
        at = f"{where}.{self.name}" if where else self.name
        if value is None:
            if self.nullable:
                return None
            raise ConfigError(f"{at}: null is not allowed here")
        # bool is a subclass of int in Python; an accidental `true` where a count belongs must not
        # be accepted as 1.
        if self.kind is not bool and isinstance(value, bool):
            raise ConfigError(f"{at}: expected {_name_of(self.kind)}, got a boolean")
        if not isinstance(value, self.kind):
            raise ConfigError(f"{at}: expected {_name_of(self.kind)}, got "
                              f"{type(value).__name__} ({value!r})")
        if self.enum is not None and value not in self.enum:
            raise ConfigError(f"{at}: {value!r} is not one of {list(self.enum)}")
        if self.minimum is not None and value < self.minimum:
            raise ConfigError(f"{at}: {value!r} is below the minimum {self.minimum}"
                              + (f" {self.unit}" if self.unit else ""))
        if self.maximum is not None and value > self.maximum:
            raise ConfigError(f"{at}: {value!r} is above the maximum {self.maximum}"
                              + (f" {self.unit}" if self.unit else ""))
        return value


def _name_of(kind: type | tuple[type, ...]) -> str:
    if isinstance(kind, tuple):
        return " or ".join(k.__name__ for k in kind)
    return kind.__name__


class Section:
    """A named mapping of fields. Unknown keys inside it are refused."""

    def __init__(self, name: str, fields: Iterable[Field], *, doc: str = "",
                 required: bool = False) -> None:
        self.name = name
        self.fields = {f.name: f for f in fields}
        self.doc = doc
        self.required = required

    def resolve(self, block: Any, *, where: str = "") -> dict[str, Any]:
        at = f"{where}.{self.name}" if where else self.name
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ConfigError(f"{at}: expected a mapping of keys, got {type(block).__name__}")
        reject_unknown(block, self.fields, where=at)
        out: dict[str, Any] = {}
        for name, field in self.fields.items():
            if name in block:
                out[name] = field.check(block[name], where=at)
            elif field.required:
                raise ConfigError(f"{at}.{name} is required and has no default"
                                  + (f" -- {field.doc}" if field.doc else ""))
            else:
                out[name] = field.default
        return out


def reject_unknown(block: Mapping[str, Any], known: Mapping[str, Any], *, where: str) -> None:
    """Refuse keys the schema does not declare, suggesting the nearest one that it does.

    This is the whole point of a strict parser. `difflib` turns "you wrote something wrong" into
    "you wrote `timestep_ps`, did you mean `timestep_fs`", which is the difference between a
    useful error and an annoying one.
    """
    unknown = [k for k in block if k not in known]
    if not unknown:
        return
    parts = []
    for key in sorted(unknown):
        near = difflib.get_close_matches(str(key), list(known), n=1, cutoff=0.6)
        parts.append(f"{key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    raise ConfigError(
        f"{where}: unknown key(s) {', '.join(parts)}. "
        f"Known keys here: {', '.join(sorted(known))}. "
        f"Unknown keys are refused rather than ignored, because a configuration value that is "
        f"silently dropped produces a completed run that answers a different question.")


class Schema:
    """A whole configuration document: top-level fields plus named sections."""

    def __init__(self, name: str, *, fields: Iterable[Field] = (),
                 sections: Iterable[Section] = (), doc: str = "",
                 checks: Sequence[Callable[[dict[str, Any]], None]] = ()) -> None:
        self.name = name
        self.fields = {f.name: f for f in fields}
        self.sections = {s.name: s for s in sections}
        self.doc = doc
        self.checks = tuple(checks)

    def resolve(self, document: Any) -> dict[str, Any]:
        if document is None:
            document = {}
        if not isinstance(document, Mapping):
            raise ConfigError(f"{self.name}: the document must be a mapping, got "
                              f"{type(document).__name__}")
        known = {**self.fields, **self.sections}
        reject_unknown(document, known, where=self.name)
        out: dict[str, Any] = {}
        for name, field in self.fields.items():
            if name in document:
                out[name] = field.check(document[name], where="")
            elif field.required:
                raise ConfigError(f"{name} is required and has no default"
                                  + (f" -- {field.doc}" if field.doc else ""))
            else:
                out[name] = field.default
        for name, section in self.sections.items():
            if section.required and name not in document:
                raise ConfigError(f"the section {name!r} is required"
                                  + (f" -- {section.doc}" if section.doc else ""))
            out[name] = section.resolve(document.get(name), where="")
        # Cross-field incompatibilities are refused here, after every individual value is known to
        # be well formed, so their messages can talk about the combination rather than the key.
        for check in self.checks:
            check(out)
        return out

    def load(self, path) -> dict[str, Any]:
        from pathlib import Path
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"{path}: no such configuration file")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML -- {exc}") from None
        try:
            return self.resolve(document)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None
