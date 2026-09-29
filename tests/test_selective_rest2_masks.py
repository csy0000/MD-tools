"""The residue-mask grammar selective REST2 accepts, and every form it refuses by name.

The table IS the specification (0.6.1 AIMS, S1-A): a documented subset of AMBER's mask grammar,
one-based topology residue indices, and a refusal that names what was not supported rather than a
parser that quietly reads `:45@CA` as `:45`.
"""
from __future__ import annotations

import pytest

from md_tools.rest2.masks import MASK_GRAMMAR, MaskError, parse_residue_mask
from md_tools.rest2.selection import SelectionError

ACCEPTED = [
    (":45", (45,)),
    (":45,46,59", (45, 46, 59)),
    (":45-50", (45, 46, 47, 48, 49, 50)),
    (":45,48-52", (45, 48, 49, 50, 51, 52)),
    (":1", (1,)),
    (" :3-4 ", (3, 4)),                     # surrounding whitespace is YAML's, not the mask's
    (":45-50,48", (45, 46, 47, 48, 49, 50)),  # overlap names the same set
    (":7-7", (7,)),
]


@pytest.mark.parametrize("mask, expected", ACCEPTED)
def test_every_supported_form_parses_to_one_based_indices(mask, expected):
    assert parse_residue_mask(mask) == expected


REFUSED = [
    (":45@CA", "atom selector"),
    ("@CA", "atom selector"),
    ("::A", "chain selector"),
    (":45::A", "chain selector"),
    ("^1", "molecule selector"),
    (":45&:46", "`&`"),
    (":45|:46", "`|`"),
    ("!:45", "`!`"),
    (":45<:5.0", "distance selector"),
    (":45>:5.0", "distance selector"),
    (":*", "wildcard"),
    (":4=", "wildcard"),
    ("@%CT", "atom selector"),
    (":ALA", "Residue NAMES"),
    (":45a", "not numeric"),
    ("45", "does not start with `:`"),
    (":", "names no residue"),
    ("", "empty"),
    (":45,,46", "empty list item"),
    (":45,", "empty list item"),
    (":45-", "not a residue number or a range"),
    (":45-50-52", "not a residue number or a range"),
    (":50-45", "descending"),
    (":0", "one-based"),
    (":0-3", "one-based"),
    (":45, 46", "whitespace"),
    (":45:46", "more than one `:`"),
    (":(45)", "parentheses"),
]


@pytest.mark.parametrize("mask, words", REFUSED)
def test_every_unsupported_form_is_refused_by_name(mask, words):
    with pytest.raises(MaskError) as refusal:
        parse_residue_mask(mask, where="backbone_scaling_list")
    message = str(refusal.value)
    assert words in message
    assert "backbone_scaling_list" in message
    assert isinstance(refusal.value, SelectionError)


@pytest.mark.parametrize("mask", REFUSED[:14])
def test_a_refused_construct_prints_the_supported_subset(mask):
    with pytest.raises(MaskError) as refusal:
        parse_residue_mask(mask)
    assert MASK_GRAMMAR in str(refusal.value)
    assert '":45,48-52"' in str(refusal.value)


@pytest.mark.parametrize("value", [45, 4.5, None, [45], True])
def test_a_mask_must_be_a_quoted_string(value):
    with pytest.raises(MaskError, match="quoted string"):
        parse_residue_mask(value)


def test_an_unquoted_integer_is_told_how_to_quote_it():
    with pytest.raises(MaskError, match='":45"'):
        parse_residue_mask(45)
