"""Missing heavy atoms are refused or built under `input.missing_atoms`; chain breaks are refused.

ACE-ALA-NME with its alanine CB deleted, and with its NME moved 5 A away (a break inside the chain).
The 1BRS case this was written for: assemblies 1 and 2 lack barstar residues 64-65 (a break), and
assembly 3 is complete but lacks 44 side-chain atoms over 15 residues.

PLATFORM_POLICY_EXEMPTION: structure inspection only; no Context.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools.openmm.completion import CompletionError, inspect_structure

ALA = Path(__file__).resolve().parents[1] / "tests" / "data" / "ALA.pdb"


def _structure(*, drop=None, shift_residue=None):
    from openmm import Vec3, app, unit

    pdb = app.PDBFile(str(ALA))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if drop is not None:
        modeller.delete([a for a in modeller.topology.atoms()
                         if (a.residue.name, a.name) == drop])
    positions = modeller.positions
    if shift_residue is not None:
        moved = {a.index for a in modeller.topology.atoms() if a.residue.name == shift_residue}
        raw = positions.value_in_unit(unit.nanometer)
        positions = unit.Quantity([p + Vec3(0.5, 0, 0) if i in moved else p
                                   for i, p in enumerate(raw)], unit.nanometer)
    return modeller.topology, positions


def test_a_complete_structure_passes_unchanged():
    topology, positions = _structure()
    same_topology, _, record = inspect_structure(topology, positions)
    assert same_topology is topology
    assert record["residues_missing_atoms"] == [] and record["atoms_added"] == []


def test_missing_side_chain_atoms_are_refused_by_default_and_named():
    topology, positions = _structure(drop=("ALA", "CB"))
    with pytest.raises(CompletionError, match=r"ALA lacks CB.*input.missing_atoms: add"):
        inspect_structure(topology, positions)


def test_missing_side_chain_atoms_are_built_and_recorded_when_asked():
    topology, positions = _structure(drop=("ALA", "CB"))
    completed, _, record = inspect_structure(topology, positions, missing_atoms="add")
    assert [(a["residue"], a["atom"]) for a in record["atoms_added"]] == [("ALA", "CB")]
    assert any(a.name == "CB" and a.residue.name == "ALA" for a in completed.atoms())


def test_a_break_inside_a_chain_is_refused_whatever_the_policy():
    topology, positions = _structure(shift_residue="NME")
    for policy in ("refuse", "add"):
        with pytest.raises(CompletionError, match="INSIDE a chain.*loop modelling"):
            inspect_structure(topology, positions, missing_atoms=policy)


def test_an_unknown_policy_is_refused():
    topology, positions = _structure()
    with pytest.raises(CompletionError, match="missing_atoms must be one of"):
        inspect_structure(topology, positions, missing_atoms="guess")


def test_an_old_configuration_naming_input_remove_is_refused(tmp_path):
    """`input.remove` was deleted: removing an additive is the reader's edit to their own file.

    An old configuration must REFUSE rather than be ignored -- a build that silently kept the
    EDO and SCN a configuration asked to delete would parameterise residues nobody meant to keep,
    or fail much later with a template error naming neither.
    """
    from md_tools.build.strict import ConfigError
    from md_tools.build.top import resolve_build_config

    config = tmp_path / "c.config"
    config.write_text("solute:\n  kind: peptide\ninput:\n  remove:\n"
                      "    - select: {chain: B, resid: \"301\"}\n      reason: additive\n",
                      encoding="utf-8")
    with pytest.raises(ConfigError, match="input.remove is retired"):
        resolve_build_config(config)


def _renamed(residue_name: str, renames: dict[str, str]):
    """ALA.pdb with one residue's atoms renamed: every atom present, under other names."""
    from openmm import app

    pdb = app.PDBFile(str(ALA))
    for atom in pdb.topology.atoms():
        if atom.residue.name == residue_name and atom.name in renames:
            atom.name = renames[atom.name]
    return pdb.topology, pdb.positions


def test_a_residue_whose_atoms_are_renamed_is_not_reported_as_missing_them():
    """The TYK2 case: upstream caps carry Maestro names, and every atom is present.

    The old message said "missing heavy atoms" and pointed at `input.missing_atoms: add`, which
    would have added a second copy of each. It must name the template mismatch instead, and must
    not suggest adding anything.
    """
    topology, positions = _renamed("ACE", {"C": "C1", "O": "O1", "CH3": "C2",
                                           "H1": "H2_1", "H2": "H2_2", "H3": "H2_3"})
    with pytest.raises(CompletionError) as refusal:
        inspect_structure(topology, positions)
    message = str(refusal.value)
    assert "under names the force-field template does not define" in message
    assert "ACE has" in message and "where the template has" in message
    assert "do NOT set input.missing_atoms: add" in message
    assert "are missing heavy atoms" not in message


def test_a_genuinely_incomplete_residue_still_gets_the_old_message():
    """The other branch: an atom that really is absent keeps the refusal that fits it."""
    topology, positions = _structure(drop=("ALA", "CB"))
    with pytest.raises(CompletionError) as refusal:
        inspect_structure(topology, positions)
    message = str(refusal.value)
    assert "are missing heavy atoms" in message and "ALA lacks CB" in message
    assert "Set input.missing_atoms: add" in message
    assert "template does not define" not in message
