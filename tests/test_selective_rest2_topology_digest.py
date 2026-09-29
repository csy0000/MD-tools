"""The topology digest is canonical from selection 2.0 on (shared contract §2, 2026-09-19).

`topology_digest` used to hash bonds in iteration order, and that order comes from the file:
OpenMM's PDB writer orders CONECT records for a non-standard residue its own way, so reading and
rewriting a structure with a cross-residue CONECT bond made the digest alternate forever. The
fixture reproduces it with the writer's own continuation lines -- a ligand (LGA) covalently linked
by CONECT to another residue (LGB) -- and the first test shows the OLD scheme oscillating on it
before anything is claimed about the new one.

PLATFORM_POLICY_EXEMPTION: topology digests and PDB text only; no Context is created.
"""
from __future__ import annotations

import io

import pytest
import yaml

from md_tools.rest2.selection import (LEGACY_SELECTION_FORMAT, TOPOLOGY_DIGEST_SCHEME,
                                      ScalingSelection, SelectionError,
                                      _legacy_iteration_order_digest, topology_digest)

from .selective_rest2_fixture import RESIDUE, build_fixture


@pytest.fixture(scope="module")
def cycles():
    """A structure with a cross-residue CONECT bond, read and rewritten four times."""
    from openmm import app

    fixture = build_fixture(padding_nm=0.5)
    topology, positions = fixture.topology, fixture.positions
    first = next(a for a in fixture.residue(RESIDUE["LGA#1"]).atoms() if a.name == "C4")
    second = next(a for a in fixture.residue(RESIDUE["LGB"]).atoms() if a.name == "O1")
    topology.addBond(first, second)
    topologies = []
    for _ in range(4):
        text = io.StringIO()
        app.PDBFile.writeFile(topology, positions, text, keepIds=True)
        pdb = app.PDBFile(io.StringIO(text.getvalue()))
        topology, positions = pdb.topology, pdb.positions
        topologies.append(topology)
    return topologies


def test_the_old_scheme_oscillates_on_this_fixture(cycles):
    """The control: without it, a stable digest below could just mean the fixture is too easy."""
    legacy = [_legacy_iteration_order_digest(t) for t in cycles]
    assert len(set(legacy)) == 2 and legacy[0] == legacy[2] and legacy[1] == legacy[3], legacy


def test_the_canonical_digest_is_stable_through_a_conect_round_trip(cycles):
    assert len({topology_digest(t) for t in cycles}) == 1


def _record(fmt, digest, **extra):
    document = {"format": fmt, "topology_sha256": digest, "solute_atoms": [0, 1],
                "unscaled_torsion_central_bonds": []}
    document.update(extra)
    return document


def test_a_1_0_record_with_the_legacy_digest_still_validates(cycles, tmp_path):
    topology = cycles[1]
    path = tmp_path / "solute.yaml"
    path.write_text(yaml.safe_dump(_record(LEGACY_SELECTION_FORMAT,
                                           _legacy_iteration_order_digest(topology))),
                    encoding="utf-8")
    assert ScalingSelection.load(path, topology=topology).mode == "legacy-full-solute"
    # ...and one with the canonical digest too, since 1.0 never said which it meant.
    path.write_text(yaml.safe_dump(_record(LEGACY_SELECTION_FORMAT, topology_digest(topology))),
                    encoding="utf-8")
    ScalingSelection.load(path, topology=topology)


def test_a_1_0_record_of_another_topology_is_still_refused(cycles, tmp_path):
    path = tmp_path / "solute.yaml"
    path.write_text(yaml.safe_dump(_record(LEGACY_SELECTION_FORMAT, "0" * 64)), encoding="utf-8")
    with pytest.raises(SelectionError, match="does not match"):
        ScalingSelection.load(path, topology=cycles[1])


def test_a_2_0_record_carrying_a_legacy_digest_is_refused(cycles, tmp_path):
    topology = cycles[1]
    legacy = _legacy_iteration_order_digest(topology)
    assert legacy != topology_digest(topology), "this topology must tell the two schemes apart"
    selection = ScalingSelection(solute_atoms=(0, 1), topology_sha256=legacy)
    path = selection.write(tmp_path / "selection.yaml")
    with pytest.raises(SelectionError, match="LEGACY iteration-order"):
        ScalingSelection.load(path, topology=topology)


def test_a_2_0_record_names_its_scheme_and_one_without_it_is_refused(cycles, tmp_path):
    topology = cycles[1]
    selection = ScalingSelection(solute_atoms=(0, 1), topology_sha256=topology_digest(topology))
    document = selection.to_document()
    assert document["topology_digest_scheme"] == TOPOLOGY_DIGEST_SCHEME
    del document["topology_digest_scheme"]
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(SelectionError, match="topology_digest_scheme"):
        ScalingSelection.load(path, topology=topology)
