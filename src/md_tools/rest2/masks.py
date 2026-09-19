"""AMBER residue masks, the documented subset: which residues a selective REST2 region names.

    ":45"          one residue
    ":45,46,59"    a list
    ":45-50"       an inclusive range
    ":45,48-52"    a mixture

That is the whole grammar (`MASK_GRAMMAR`, version 1). Numbers are ONE-BASED TOPOLOGY RESIDUE
INDICES -- the position of the residue in the built topology, counted across every chain without
reset -- which is what AMBER's `:` selector means. They are not Python indices and not PDB author
residue numbers: a structure numbered from 201 still has residue 1.

Everything else AMBER accepts is REFUSED BY NAME, never approximated: residue names (`:ALA`), atom
selectors (`@CA`), chain or molecule selectors (`::A`, `^1`), the boolean and distance operators
(`&`, `|`, `!`, `<:`, `>:`), wildcards and `=`. A parser that quietly read `:45@CA` as `:45` would
heat a whole residue when an atom was asked for, and the run would look exactly as intended.

This module parses. Resolving the numbers against a topology -- range checks, what each residue
is -- is `md_tools.rest2.regions`, so the grammar can be tested without a System.
"""
from __future__ import annotations

import re

from .selection import SelectionError

__all__ = ["MASK_GRAMMAR", "MASK_NUMBERING", "MaskError", "parse_residue_mask",
           "SUPPORTED_FORMS"]

#: The grammar's name and version, recorded beside every mask it parsed.
MASK_GRAMMAR = "md-tools-residue-mask/1"
#: What a number in a mask means, recorded beside every mask.
MASK_NUMBERING = "topology-residue-index-1based"

SUPPORTED_FORMS = ('":45" (one residue)', '":45,46,59" (a list)', '":45-50" (an inclusive range)',
                   '":45,48-52" (a mixture)')


class MaskError(SelectionError):
    """A mask outside the supported subset, or one that does not describe this topology."""


def _supported() -> str:
    return ("the supported subset of the AMBER mask grammar (" + MASK_GRAMMAR + ") is "
            + ", ".join(SUPPORTED_FORMS) + "; numbers are one-based topology residue indices")


#: Refused constructs, checked in this order, each with what it would have meant in AMBER.
_REFUSED = (
    ("@", "an atom selector (`@`)"),
    ("::", "a chain selector (`::`)"),
    ("^", "a molecule selector (`^`)"),
    ("&", "the `&` (and) operator"),
    ("|", "the `|` (or) operator"),
    ("!", "the `!` (not) operator"),
    ("<", "a distance selector (`<:` / `<@`)"),
    (">", "a distance selector (`>:` / `>@`)"),
    ("*", "a wildcard (`*`)"),
    ("=", "a wildcard (`=`)"),
    ("%", "a type selector (`%`)"),
    ("(", "parentheses"),
    (")", "parentheses"),
)

_NUMBER = re.compile(r"[0-9]+")


def parse_residue_mask(mask, *, where: str = "mask") -> tuple[int, ...]:
    """The one-based residue indices *mask* names, sorted and unique. Refuses anything else.

    Overlapping entries (`":45-50,48"`) name the same set and are accepted. A descending range, a
    zero, an empty item and every construct outside the subset are refused, each by name.
    """
    if not isinstance(mask, str):
        hint = (f" An unquoted YAML value such as {mask!r} is not a mask: write it quoted, "
                f"\":{mask}\"." if isinstance(mask, int) and not isinstance(mask, bool) else "")
        raise MaskError(f"{where}: a residue mask is a quoted string, got {type(mask).__name__} "
                        f"{mask!r}.{hint} {_supported()}.")
    text = mask.strip()
    if not text:
        raise MaskError(f"{where}: the mask is empty. {_supported()}.")
    for token, meaning in _REFUSED:
        if token in text:
            raise MaskError(f"{where}: {mask!r} uses {meaning}, which is not supported. "
                            f"{_supported()}.")
    if not text.startswith(":"):
        raise MaskError(f"{where}: {mask!r} does not start with `:`, the residue selector. "
                        f"{_supported()}.")
    body = text[1:]
    if not body.strip():
        raise MaskError(f"{where}: {mask!r} names no residue. {_supported()}.")
    if ":" in body:
        raise MaskError(f"{where}: {mask!r} has more than one `:`; several residue selections are "
                        f"written as one list, \":45,48-52\". {_supported()}.")
    if re.search(r"\s", body):
        raise MaskError(f"{where}: {mask!r} contains whitespace; write the list without spaces, "
                        f"e.g. \":45,48-52\". {_supported()}.")

    selected: set[int] = set()
    for item in body.split(","):
        if not item:
            raise MaskError(f"{where}: {mask!r} has an empty list item (a stray `,`). "
                            f"{_supported()}.")
        bounds = item.split("-")
        if len(bounds) > 2 or not all(bounds):
            raise MaskError(f"{where}: {mask!r}: {item!r} is not a residue number or a range "
                            f"`first-last`. {_supported()}.")
        if not all(_NUMBER.fullmatch(bound) for bound in bounds):
            raise MaskError(
                f"{where}: {mask!r}: {item!r} is not numeric. Residue NAMES (\":ALA\") are not "
                f"supported: name the residues by their one-based topology index. {_supported()}.")
        numbers = [int(bound) for bound in bounds]
        if any(number == 0 for number in numbers):
            raise MaskError(f"{where}: {mask!r}: residue 0 does not exist -- the numbering is "
                            f"one-based, so the first residue is 1. {_supported()}.")
        if len(numbers) == 2 and numbers[0] > numbers[1]:
            raise MaskError(f"{where}: {mask!r}: the range {item!r} is descending; write it "
                            f"\"{numbers[1]}-{numbers[0]}\".")
        selected.update(range(numbers[0], numbers[-1] + 1))
    return tuple(sorted(selected))
