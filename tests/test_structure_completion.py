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


# --- input.remove ---------------------------------------------------------------------------------

def _with_additive():
    """ACE-ALA-NME plus one ethylene glycol residue (EDO) in chain B, as a deposited additive."""
    from openmm import Vec3, app, unit
    from openmm.app import element

    topology, positions = _structure()
    modeller = app.Modeller(topology, positions)
    extra = app.Topology()
    chain = extra.addChain("B")
    residue = extra.addResidue("EDO", chain, "301")
    for name, symbol in (("C1", element.carbon), ("O1", element.oxygen)):
        extra.addAtom(name, symbol, residue)
    modeller.add(extra, [Vec3(2.0, 2.0, 2.0), Vec3(2.14, 2.0, 2.0)] * unit.nanometer)
    return modeller.topology, modeller.positions


def test_a_named_additive_is_removed_and_recorded():
    from md_tools.openmm.completion import remove_residues

    topology, positions = _with_additive()
    kept, _, record = remove_residues(topology, positions, [
        {"select": {"chain": "B", "resid": "301"}, "reason": "crystallisation additive"}])
    assert "EDO" not in {r.name for r in kept.residues()}
    assert record == [{"chain": "B", "resid": "301", "insertion_code": "", "residue": "EDO",
                       "n_atoms": 2, "reason": "crystallisation additive"}]


@pytest.mark.parametrize("select, words", [
    ({"chain": "B", "resid": "999"}, "matches 0 residues"),
    ("ALA", "standard protein residue"),
])
def test_a_removal_that_is_not_one_non_protein_residue_is_refused(select, words):
    from md_tools.openmm.completion import remove_residues

    topology, positions = _with_additive()
    if select == "ALA":
        alanine = next(r for r in topology.residues() if r.name == "ALA")
        select = {"chain": alanine.chain.id, "resid": str(alanine.id)}
    with pytest.raises(CompletionError, match=words):
        remove_residues(topology, positions, [{"select": select, "reason": "x"}])


def test_a_removal_without_a_reason_is_refused_at_resolution(tmp_path):
    from md_tools.build.strict import ConfigError
    from md_tools.build.top import resolve_build_config

    config = tmp_path / "c.config"
    config.write_text("solute:\n  kind: peptide\ninput:\n  remove:\n"
                      "    - select: {chain: B, resid: \"301\"}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no reason"):
        resolve_build_config(config)
