"""`solute.kind`, and the retired boolean it replaces.

WHY THE ORDER MATTERS

    `kind` has a default and the legacy `peptide` boolean does not. If the alias were folded in
    AFTER schema resolution, every configuration would arrive carrying `kind: peptide` whether or
    not anyone wrote it, and a stated `peptide: false` would look like a conflict with a value
    nobody chose. So the fold happens on the RAW document, where "stated" still means stated --
    and these tests exist because that ordering is invisible from the outside and easy to undo.

WHAT `peptide-like` IS

    The ligand route, plus a map. Not a third parameterisation. The test that matters here is the
    negative one: a peptide-like configuration must resolve to the same force-field selection as
    a ligand configuration, because if it ever started loading a protein force field the whole
    point of the classification would be lost while every log still said Sage.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from md_tools.build.strict import ConfigError
from md_tools.build.top import resolve_build_config


def _resolve(document):
    with tempfile.NamedTemporaryFile("w", suffix=".config", delete=False) as handle:
        yaml.safe_dump(document, handle)
        path = Path(handle.name)
    try:
        return resolve_build_config(path)
    finally:
        path.unlink(missing_ok=True)


# --- the three kinds --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["peptide", "peptide-like", "ligand"])
def test_each_kind_resolves_to_itself(kind):
    assert _resolve({"solute": {"kind": kind}})["solute"]["kind"] == kind


def test_absent_fields_keep_the_historical_default():
    """A configuration written before any of this existed must build what it always built."""
    assert _resolve({})["solute"]["kind"] == "peptide"
    assert _resolve({"solvent": {"model": "GBn2"}})["solute"]["kind"] == "peptide"


# --- the retired boolean ----------------------------------------------------------------------

@pytest.mark.parametrize("legacy,expected", [(True, "peptide"), (False, "ligand")])
def test_the_legacy_boolean_maps_to_a_kind(legacy, expected):
    assert _resolve({"solute": {"peptide": legacy}})["solute"]["kind"] == expected


@pytest.mark.parametrize("kind,legacy", [("peptide", True), ("ligand", False)])
def test_the_compatible_alias_pair_is_accepted(kind, legacy):
    """Saying the same thing twice is redundant, not wrong."""
    assert _resolve({"solute": {"kind": kind, "peptide": legacy}})["solute"]["kind"] == kind


@pytest.mark.parametrize("kind,legacy", [
    ("peptide", False), ("ligand", True),
    # `peptide-like` postdates the boolean and has no truthful spelling in it, so EITHER value
    # contradicts it. Both directions are refused rather than one being quietly preferred.
    ("peptide-like", True), ("peptide-like", False),
])
def test_a_contradictory_pair_is_refused_with_a_migration_message(kind, legacy):
    with pytest.raises(ConfigError) as refusal:
        _resolve({"solute": {"kind": kind, "peptide": legacy}})
    message = str(refusal.value)
    assert "solute.kind" in message and "solute.peptide" in message
    # It must say what to DO, not merely that something is wrong.
    assert "delete solute.peptide" in message


def test_an_injected_boolean_cannot_override_a_stated_kind():
    """The ordering property, stated as a behaviour.

    If aliases were resolved after defaults, `kind` would always be present and this case would
    be indistinguishable from the compatible-pair case above. It is refused because the two
    stated values genuinely disagree.
    """
    with pytest.raises(ConfigError):
        _resolve({"solute": {"kind": "peptide-like", "peptide": True}})
    # And the reverse: a stated kind with no boolean is untouched.
    assert _resolve({"solute": {"kind": "peptide-like"}})["solute"]["kind"] == "peptide-like"


# --- bad input --------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["protein", "Peptide", "peptide_like", "", "small-molecule"])
def test_an_unknown_kind_is_refused(value):
    with pytest.raises(ConfigError) as refusal:
        _resolve({"solute": {"kind": value}})
    assert "solute.kind" in str(refusal.value)


@pytest.mark.parametrize("value", ["yes", 1, 0, None, "true"])
def test_a_non_boolean_legacy_value_is_refused(value):
    with pytest.raises(ConfigError) as refusal:
        _resolve({"solute": {"peptide": value}})
    assert "solute.peptide" in str(refusal.value)


# --- round trip and the force field -----------------------------------------------------------

def test_the_resolved_configuration_round_trips():
    """What is written back must resolve to the same classification when read again."""
    for kind in ("peptide", "peptide-like", "ligand"):
        once = _resolve({"solute": {"kind": kind}})
        again = _resolve({"solute": {"kind": once["solute"]["kind"]}})
        assert again["solute"]["kind"] == kind


def test_a_saved_configuration_carrying_the_old_boolean_still_reads():
    """`resolved.config` files already on disk carry `peptide`, and must keep working."""
    for legacy, expected in ((True, "peptide"), (False, "ligand")):
        saved = {"solute": {"peptide": legacy, "ligand_forcefield": "sage-2.2.1",
                            "ligand_charge_method": "am1bcc"},
                 "solvent": {"model": "GBn2"}}
        assert _resolve(saved)["solute"]["kind"] == expected


def test_peptide_like_selects_sage_and_never_a_protein_force_field():
    """THE property that makes peptide-like a classification rather than a second route."""
    from md_tools.openmm.system_defaults import sys_defaults

    like = sys_defaults(peptide=False, kind="peptide-like", solvent="GBn2")
    ligand = sys_defaults(peptide=False, kind="ligand", solvent="GBn2")

    assert like["solute"]["ligand_forcefield"] == ligand["solute"]["ligand_forcefield"]
    assert like["solute"]["ligand_charge_method"] == ligand["solute"]["ligand_charge_method"]
    # The derived compatibility field agrees with the classification and is never a second
    # opinion: peptide-like is not a protein build.
    assert like["solute"]["peptide"] is False
    assert like["solute"]["kind"] == "peptide-like"


def test_the_derived_boolean_is_recomputed_from_kind_rather_than_stored_independently():
    """A derived field that could drift from the authoritative one is a second source of truth."""
    from md_tools.openmm.system_defaults import sys_defaults

    for kind, expected in (("peptide", True), ("peptide-like", False), ("ligand", False)):
        document = sys_defaults(peptide=(kind == "peptide"), kind=kind, solvent="GBn2")
        assert document["solute"]["kind"] == kind
        assert document["solute"]["peptide"] is expected
