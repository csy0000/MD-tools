"""The pocket helper: which residues line a ligand, printed as a mask to paste.

`md_tools.rest2.pocket` resolves nothing at run time; it prints. What is tested is the criterion
it states (`POCKET_CRITERION`): heavy atoms only, minimum distance, STRICTLY inside the cutoff,
minimum image when there is a box, solvent excluded by default -- and that the mask it prints is
one the mask parser accepts.

PLATFORM_POLICY_EXEMPTION: distances over coordinates; no System and no Context.
"""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import Vec3, unit                                                   # noqa: E402
from openmm.app import Topology, element                                        # noqa: E402

from md_tools.rest2.masks import parse_residue_mask                             # noqa: E402
from md_tools.rest2.pocket import (DEFAULT_CUTOFF_NM, POCKET_CRITERION, PocketError,  # noqa: E402
                                   main, pocket_report, pocket_residues, residue_mask)


def _structure(residues, *, box=None):
    """A topology of one-atom-or-more residues at given positions: `[(name, [(element, xyz)])]`."""
    topology = Topology()
    chain = topology.addChain("A")
    positions = []
    for index, (name, atoms) in enumerate(residues, start=1):
        residue = topology.addResidue(name, chain, id=str(index))
        for symbol, xyz in atoms:
            topology.addAtom(symbol, element.get_by_symbol(symbol), residue)
            positions.append(Vec3(*xyz))
    if box is not None:
        topology.setPeriodicBoxVectors([Vec3(box, 0, 0), Vec3(0, box, 0), Vec3(0, 0, box)]
                                       * unit.nanometer)
    return topology, positions * unit.nanometer


#: Two heavy atoms, the second pointing AWAY from the residues below, so a distance quoted here
#: is the distance from the ligand atom at the origin.
LIGAND = ("LIG", [("C", (0.0, 0.0, 0.0)), ("N", (-0.1, 0.0, 0.0))])


def test_the_cutoff_is_strict_and_a_residue_exactly_on_it_is_listed_but_not_selected():
    topology, positions = _structure([
        LIGAND,
        ("ALA", [("C", (0.4, 0.0, 0.0))]),      # inside
        ("SER", [("C", (0.5, 0.0, 0.0))]),      # EXACTLY at the cutoff: out, by the stated rule
        ("LEU", [("C", (0.54, 0.0, 0.0))]),     # just outside: listed as a near miss
        ("VAL", [("C", (0.9, 0.0, 0.0))]),      # far: not mentioned at all
    ])
    found = pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5)
    assert [entry["residue"] for entry in found["residues"]] == [2]
    assert [entry["residue"] for entry in found["near_misses"]] == [3, 4]
    assert found["residues"][0]["minimum_distance_nm"] == pytest.approx(0.4)
    assert found["criterion"] == POCKET_CRITERION


def test_hydrogens_are_not_measured():
    topology, positions = _structure([
        LIGAND,
        ("ALA", [("C", (0.7, 0.0, 0.0)), ("H", (0.3, 0.0, 0.0))]),
    ])
    found = pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5)
    assert found["residues"] == [], "a hydrogen must not pull a residue into the pocket"
    assert found["near_misses"] == []


def test_the_minimum_distance_is_over_all_pairs_not_a_centroid():
    topology, positions = _structure([
        LIGAND,
        # A long residue: its centroid is 1.0 nm away, one atom is 0.3 nm away.
        ("ARG", [("C", (0.3, 0.0, 0.0)), ("C", (1.0, 0.0, 0.0)), ("N", (1.7, 0.0, 0.0))]),
    ])
    found = pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5)
    assert [entry["residue"] for entry in found["residues"]] == [2]
    assert found["residues"][0]["minimum_distance_nm"] == pytest.approx(0.3)


def test_a_residue_wrapped_to_the_far_side_of_the_box_is_measured_where_it_is():
    # ONE ligand atom here, at the origin: with the two-atom LIGAND the second atom would itself
    # wrap to 2.9 and the nearest pair would be a different one.
    single = ("LIG", [("C", (0.0, 0.0, 0.0))])
    topology, positions = _structure([
        single,
        ("ALA", [("C", (2.7, 0.0, 0.0))]),     # 0.3 nm away through the periodic boundary
    ], box=3.0)
    found = pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5)
    assert found["periodic"] is True
    assert [entry["residue"] for entry in found["residues"]] == [2]
    assert found["residues"][0]["minimum_distance_nm"] == pytest.approx(0.3)
    # ...and without a box the same coordinates are far away.
    flat, flat_positions = _structure([single, ("ALA", [("C", (2.7, 0.0, 0.0))])])
    assert pocket_residues(flat, flat_positions, ligand=":1", cutoff_nm=0.5)["residues"] == []


def test_solvent_and_ions_are_excluded_unless_asked_for():
    topology, positions = _structure([
        LIGAND,
        ("HOH", [("O", (0.25, 0.0, 0.0))]),
        ("NA", [("Na", (0.3, 0.0, 0.0))]),
        ("ALA", [("C", (0.35, 0.0, 0.0))]),
    ])
    assert [e["residue"] for e in pocket_residues(
        topology, positions, ligand=":1", cutoff_nm=0.5)["residues"]] == [4]
    everything = pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5,
                                 include_solvent=True)
    assert [e["residue"] for e in everything["residues"]] == [2, 3, 4]


@pytest.mark.parametrize("numbers, mask", [
    ([45], ":45"), ([45, 46, 47], ":45-47"), ([45, 47], ":45,47"),
    ([45, 46, 59, 60, 61, 70], ":45-46,59-61,70"), ([], ""),
])
def test_the_printed_mask_is_compact_and_the_parser_accepts_it(numbers, mask):
    assert residue_mask(numbers) == mask
    if mask:
        assert parse_residue_mask(mask) == tuple(sorted(numbers))


def test_the_ligand_must_name_exactly_one_residue():
    topology, positions = _structure([LIGAND, ("ALA", [("C", (0.4, 0.0, 0.0))])])
    with pytest.raises(PocketError, match="names 2 residues"):
        pocket_residues(topology, positions, ligand=":1,2", cutoff_nm=0.5)
    with pytest.raises(PocketError, match="out of range"):
        pocket_residues(topology, positions, ligand=":9", cutoff_nm=0.5)


def test_the_report_states_the_criterion_the_structure_and_what_was_left_out():
    topology, positions = _structure([
        LIGAND, ("ALA", [("C", (0.4, 0.0, 0.0))]), ("LEU", [("C", (0.52, 0.0, 0.0))])])
    text = pocket_report(pocket_residues(topology, positions, ligand=":1", cutoff_nm=0.5),
                         source="built.pdb")
    assert POCKET_CRITERION in text and "strictly <" in text
    assert "built.pdb" in text and "topology_sha256" in text
    assert 'sidechain_scaling_list: ":2"' in text
    assert "NOT selected" in text and "LEU" in text
    assert "nothing resolves a pocket at run time" in text


# --- against the real fixture, checked independently -----------------------------------------------

def test_on_the_boundary_fixture_every_answer_matches_a_brute_force_check(tmp_path, capsys):
    from .selective_rest2_fixture import RESIDUE, build_fixture

    fixture = build_fixture(padding_nm=0.5)
    cutoff = 1.6
    found = pocket_residues(fixture.topology, fixture.positions,
                            ligand=f":{RESIDUE['LGA#1']}", cutoff_nm=cutoff)
    assert found["residues"], "the fixture must have something in range at this cutoff"

    # Independently: brute force over heavy atoms, minimum image, strict comparison.
    xyz = np.array(fixture.positions.value_in_unit(unit.nanometer))
    box = np.array([[float(v.value_in_unit(unit.nanometer)) for v in row]
                    for row in fixture.topology.getPeriodicBoxVectors()])
    residues = list(fixture.topology.residues())
    target = residues[RESIDUE["LGA#1"] - 1]
    heavy = lambda r: [a.index for a in r.atoms()                              # noqa: E731
                       if a.element is not None and a.element.symbol != "H"]
    expected = []
    for other in residues:
        if other.index == target.index or other.name in ("HOH", "NA", "CL"):
            continue
        atoms = heavy(other)
        if not atoms:
            continue
        deltas = (xyz[atoms][:, None, :] - xyz[heavy(target)][None, :, :]).reshape(-1, 3)
        for axis in (2, 1, 0):
            deltas = deltas - np.outer(np.round(deltas[:, axis] / box[axis][axis]), box[axis])
        if float(np.linalg.norm(deltas, axis=1).min()) < cutoff:
            expected.append(other.index + 1)
    assert [entry["residue"] for entry in found["residues"]] == sorted(expected)

    # ...and the printed mask resolves back to exactly those residues.
    mask = residue_mask([entry["residue"] for entry in found["residues"]])
    assert parse_residue_mask(mask) == tuple(sorted(expected))


def test_the_command_line_prints_a_mask_and_refuses_a_bad_ligand(tmp_path, capsys):
    from .selective_rest2_fixture import RESIDUE, build_fixture

    build = build_fixture(padding_nm=0.5).write(tmp_path / "build", sdfs=False)
    argv = [str(build / "built.pdb"), "--ligand", f":{RESIDUE['LGA#1']}", "--cutoff-nm", "1.6"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert POCKET_CRITERION in printed and "sidechain_scaling_list" in printed
    assert main([str(build / "built.pdb"), "--ligand", ":99999"]) == 2
    assert "out of range" in capsys.readouterr().err
    assert main([str(build / "built.pdb"), "--ligand", ":13@CA"]) == 2
    assert "atom selector" in capsys.readouterr().err
