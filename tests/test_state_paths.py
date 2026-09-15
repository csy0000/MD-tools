"""The per-state directory layout, and the one authority that names it.

`md_tools.remd.amber_trajectory` decides every per-state name. This file pins the layout itself,
because the layout is a contract with whoever reads a finished run:

    remd<i>/  build_state<i>.xml
              whole_state<i>_prod<N>.nc     solute_state<i>_prod<N>.nc
              cv_state<i>_prod<N>.dat       cv_state<i>_prod<N>.json
              restart_state<i>_prod<N>.json restart_state<i>_prod<N>.xml
    rank/     the rank-indexed process reports
    <root>    rem.log, exchange.csv, and the ladder's own record

TWO THINGS HERE ARE CORRECTNESS RATHER THAN TIDINESS.

The rank directory is separate from the state directories because a rank is a PROCESS and a state
is a THERMODYNAMIC STATE, and they are not in bijection -- the reference ladder this work is
migrating ran 6 states on 5 ranks. Filing `REST2.out.rank01` under `remd1/` would assert a
correspondence that does not exist.

The segment is part of every per-state filename because a second segment written into the same
state directory would otherwise collide with the first. The CV series carried no segment at all,
which was safe only while the trajectories beside it were equally unsegmented -- and they were,
which is exactly the defect: five chunks of one reference run each wrote `_prod1`.

PLATFORM_POLICY_EXEMPTION: pure string and path arithmetic. No Context, no device, no file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools.remd import amber_trajectory as amber


# -- the state directory ------------------------------------------------------------------------

@pytest.mark.parametrize("index,expected", [(0, "remd0"), (2, "remd2"), (12, "remd12")])
def test_a_state_directory_is_named_for_its_state_index(index, expected):
    assert amber.state_directory(index) == expected


def test_a_negative_state_index_is_refused_everywhere_it_appears():
    """One guard, applied by every name, so no artefact can be written to `remd-1/`."""
    for call in (amber.state_directory,
                 amber.state_system_name,
                 amber.state_cv_name,
                 amber.state_restart_name,
                 lambda i: amber.state_trajectory_name(i, content="whole")):
        with pytest.raises(ValueError, match="cannot be negative"):
            call(-1)


# -- the per-state filenames --------------------------------------------------------------------

def test_the_rung_system_is_named_for_its_state_and_carries_no_segment():
    """The Hamiltonian does not change between segments, so a segment in the name would lie."""
    assert amber.state_system_name(3) == "build_state3.xml"


@pytest.mark.parametrize("segment", [1, 2, 5])
def test_every_per_state_stream_carries_the_segment(segment):
    """A second segment must not be able to collide with the first."""
    names = {
        amber.state_trajectory_name(1, content="whole", segment=segment),
        amber.state_trajectory_name(1, content="solute", segment=segment),
        amber.state_cv_name(1, segment=segment),
        amber.state_cv_name(1, segment=segment, suffix="json"),
        amber.state_restart_name(1, segment=segment),
        amber.state_restart_name(1, segment=segment, suffix="xml"),
    }
    assert len(names) == 6, f"two streams share a name at segment {segment}: {sorted(names)}"
    for name in names:
        assert f"_prod{segment}." in name, name


def test_the_cv_series_is_dat_with_a_json_sidecar():
    assert amber.state_cv_name(0) == "cv_state0_prod1.dat"
    assert amber.state_cv_name(0, suffix="json") == "cv_state0_prod1.json"


def test_the_per_state_restart_is_a_json_record_and_an_xml_state():
    assert amber.state_restart_name(4) == "restart_state4_prod1.json"
    assert amber.state_restart_name(4, suffix="xml") == "restart_state4_prod1.xml"


def test_segments_of_one_state_never_collide():
    """THE DEFECT THIS EXISTS FOR. Five chunks of the reference run all wrote `_prod1`."""
    produced = [amber.state_trajectory_name(0, content="solute", segment=n)
                for n in range(1, 6)]
    assert len(set(produced)) == 5, produced
    assert produced[1] == "solute_state0_prod2.nc"


# -- the rank directory -------------------------------------------------------------------------

def test_the_rank_directory_is_not_a_state_directory():
    """6 states on 5 ranks: a rank report belongs to no single state."""
    assert amber.RANK_DIRECTORY == "rank"
    assert amber.RANK_DIRECTORY != amber.state_directory(1)
    for index in range(8):
        assert amber.state_directory(index) != amber.RANK_DIRECTORY


# -- reading an index back out ------------------------------------------------------------------

def test_a_bare_state_trajectory_name_still_yields_its_index():
    """Group files name bare outputs today; that must keep working."""
    assert amber.state_index_from_name("whole_state3_prod1.nc") == 3
    assert amber.state_index_from_name("solute_state0_prod7.nc") == 0


def test_a_name_inside_its_own_state_directory_is_accepted():
    assert amber.state_index_from_name("remd3/whole_state3_prod1.nc") == 3
    assert amber.state_index_from_name(Path("run") / "remd0" / "solute_state0_prod2.nc") == 0


def test_a_directory_and_a_filename_that_disagree_are_refused():
    """Picking either would be a guess, and both readings look authoritative."""
    with pytest.raises(ValueError, match="disagrees with itself"):
        amber.state_index_from_name("remd2/whole_state3_prod1.nc")


def test_a_parent_that_is_not_a_state_directory_is_ignored_not_refused():
    """The index comes from the FILENAME; the layout is not a precondition for reading it.

    This function is handed absolute paths and paths under arbitrary run directories. An earlier
    version of the directory check refused anything whose parent was not `remd<i>/`, which broke
    `/somewhere/whole_state3_prod1.nc` for no reason -- a real case, asserted in
    `test_amber_trajectory.py::test_the_name_is_the_state_index`.
    """
    assert amber.state_index_from_name("/somewhere/whole_state3_prod1.nc") == 3
    assert amber.state_index_from_name("trajectories/whole_state3_prod1.nc") == 3


def test_something_that_is_not_a_state_trajectory_is_still_refused_by_name():
    for wrong in ("remd0.nc", "whole_prod1.nc", "cv_state0_prod1.dat", "REST2.nc"):
        with pytest.raises(ValueError, match="not a state trajectory name"):
            amber.state_index_from_name(wrong)


def test_the_groupfile_validator_accepts_names_in_state_directories():
    """`validate_state_outputs` is the preflight's check, and it reads through the same parser."""
    names = [f"remd{i}/whole_state{i}_prod1.nc" for i in range(4)]
    assert amber.validate_state_outputs(names) == [0, 1, 2, 3]


def test_the_groupfile_validator_still_accepts_bare_names():
    names = [f"whole_state{i}_prod1.nc" for i in range(3)]
    assert amber.validate_state_outputs(names) == [0, 1, 2]
