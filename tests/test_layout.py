"""The run directory layout, pinned against the migrated reference dataset.

`md_tools.layout` is the one authority for where a file goes. Seven modules used to assemble
output paths themselves -- reservoir, continuation, ais/run, openmm/checkpoint,
state_trajectories, overwrite and md/stage -- which is seven places to disagree, and a reader of
a finished run cannot tell which was right.

EVERY NAME HERE WAS READ OFF A REAL MIGRATED RUN, not designed in the abstract:
`data/reference/{ALA-explicit-HMR,ALA-implicit-HMR,RGDfV-implicit-HMR}/`. That matters because two
of the shapes were surprises -- the equilibration coordinate streams are solvent-dependent, and
the rank count does not equal the state count.

PLATFORM_POLICY_EXEMPTION: pure path arithmetic. No Context, no device, no file is opened.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools import layout as L


@pytest.fixture
def dataset(tmp_path):
    return L.DatasetLayout(tmp_path / "ALA-explicit-HMR")


@pytest.fixture
def run(dataset):
    return dataset.run("REST2", 1)


# -- the shared dataset root --------------------------------------------------------------------

def test_the_three_shared_directories_sit_at_the_dataset_root(dataset):
    assert dataset.build.name == "build"
    assert dataset.min.name == "min"
    assert dataset.input.name == "input"
    for directory in (dataset.build, dataset.min, dataset.input):
        assert directory.parent == dataset.root


def test_a_run_is_a_sibling_of_the_shared_directories(dataset, run):
    assert run.root.parent == dataset.root
    assert run.root.name == "REST2-run1"
    assert dataset.run("cMD", 2).root.name == "cMD-run2"
    assert dataset.run("AIS", 3).root.name == "AIS-run3"


def test_the_built_system_keeps_build_tops_own_names(dataset):
    """`-os` defaults to built.xml, `-op` to built.pdb, `-log` to built.log."""
    assert dataset.built("xml").name == "built.xml"
    assert dataset.built("pdb").name == "built.pdb"
    assert dataset.built("built.solute.pdb").name == "built.solute.pdb"
    assert dataset.built("log").name == "built.log"
    # A ligand system carries an SDF; an implicit one has no build-top.out. Asked for, not assumed.
    assert dataset.built("sdf").name == "built.sdf"
    assert dataset.built("build-top.out").name == "build-top.out"


def test_the_minimised_structure_and_the_inputs(dataset):
    assert dataset.minimised("xml").name == "min.xml"
    assert dataset.minimised("xml").parent == dataset.min
    assert dataset.stage_input("min").name == "min.in"
    assert dataset.stage_input("eq_1").name == "eq_1.in"
    assert dataset.stage_input("REST2.in").name == "REST2.in"
    assert dataset.stage_input("REST2").parent == dataset.input


# -- per-state output ---------------------------------------------------------------------------

def test_a_state_directory_holds_everything_for_that_state(run):
    assert run.state(3).name == "remd3"
    assert run.state(3).parent == run.root
    for path in (run.state_system(3), run.state_trajectory(3), run.state_cv(3),
                 run.state_restart(3)):
        assert path.parent == run.state(3), path


def test_the_whole_system_stream_is_remd_state_and_the_solute_stream_is_solute_state(run):
    """Read off the migrated tree: remd0/ holds remd_state0_prod<x>.nc and solute_state0_...."""
    assert run.state_trajectory(0, content="whole", segment=2).name == "remd_state0_prod2.nc"
    assert run.state_trajectory(0, content="solute", segment=2).name == "solute_state0_prod2.nc"


@pytest.mark.parametrize("segment", [1, 2, 5])
def test_every_per_state_stream_carries_the_segment_and_none_collide(run, segment):
    names = {run.state_trajectory(1, content="whole", segment=segment).name,
             run.state_trajectory(1, content="solute", segment=segment).name,
             run.state_cv(1, segment=segment).name,
             run.state_cv(1, segment=segment, suffix="json").name,
             run.state_restart(1, segment=segment).name,
             run.state_restart(1, segment=segment, suffix="xml").name}
    assert len(names) == 6, sorted(names)
    for name in names:
        assert f"_prod{segment}." in name, name


def test_the_rung_system_carries_no_segment(run):
    """The Hamiltonian does not change between segments, so a segment would be a lie."""
    assert run.state_system(2).name == "build_state2.xml"


def test_five_segments_of_one_state_never_collide(run):
    """THE DEFECT THIS LAYOUT ANSWERS: every chunk of the reference run wrote _prod1."""
    produced = {run.state_trajectory(0, content="solute", segment=n).name for n in range(1, 6)}
    assert len(produced) == 5


# -- the ladder's own records -------------------------------------------------------------------

def test_the_records_share_one_directory_with_the_segment_in_each_name(run):
    """Indexed by segment ALONE, so a directory per segment would scatter one subject."""
    assert run.records.name == "remd_records"
    for kind, expected in (("ledger", "ledger_prod2.nc"),
                           ("ledger_solute", "ledger_prod2.solute.nc"),
                           ("checkpoint", "checkpoint_prod2.nc"),
                           ("restart", "restart_prod2.json"),
                           ("runstate", "runstate_prod2.json"),
                           ("rem_log", "rem_prod2.log"),
                           ("exchange", "exchange_prod2.csv")):
        path = run.record(kind, segment=2)
        assert path.name == expected, kind
        assert path.parent == run.records


def test_the_protocol_named_records_carry_the_protocol(run):
    for kind, expected in (("out", "REST2_prod3.out"), ("log", "REST2_prod3.log"),
                           ("group", "REST2_prod3.group"),
                           ("stage_stdout", "REST2_prod3.stage.stdout")):
        assert run.record(kind, segment=3).name == expected, kind


def test_the_ledger_is_not_named_like_a_trajectory(run):
    """It has no coordinates and no Conventions attribute; `REST2.nc` read as the trajectory."""
    assert run.ledger(segment=1).name == "ledger_prod1.nc"
    assert run.ledger(segment=1, solute=True).name == "ledger_prod1.solute.nc"
    assert "state" not in run.ledger().name


def test_an_unknown_record_kind_is_refused_by_name(run):
    with pytest.raises(ValueError, match="unknown record kind"):
        run.record("trajectory")


def test_a_segment_below_one_is_refused(run):
    with pytest.raises(ValueError, match="segment number starts at 1"):
        run.record("ledger", segment=0)


# -- rank reports -------------------------------------------------------------------------------

def test_rank_reports_are_not_filed_under_a_state(run):
    """6 states on 5 ranks, and 4 on 3: a rank belongs to no single state."""
    path = run.rank_report(1, segment=2)
    assert path.name == "REST2_prod2.out.rank01"
    assert path.parent == run.rank
    assert run.rank.name == "rank"
    assert run.rank != run.state(1)
    assert run.rank_report("03", segment=1, what="log").name == "REST2_prod1.log.rank03"


# -- equilibration ------------------------------------------------------------------------------

def test_equilibration_is_numbered_by_position_not_by_ensemble(run):
    """Implicit solvent renames its stages; the position is what both shapes share."""
    assert L.eq_stage_key(1) == "eq_1"
    assert run.eq_artifact(2, "xml").name == "eq_2.xml"
    assert run.eq_artifact(3, "out").parent == run.eq
    assert run.eq_checkpoints(1).name == "eq_1.checkpoints"


def test_the_equilibration_coordinate_streams_are_solvent_dependent(run):
    """Read off both migrated shapes, and this one surprised me.

    Explicit solvent writes `eq_<k>.whole.nc` AND `eq_<k>.solute.nc`; implicit writes
    `eq_<k>.whole.nc` AND `eq_<k>.dcd`. Ordinary MD writes genuine DCD and OpenMM has no native
    NetCDF writer, so the pair is whatever the stage produced rather than a fixed two.
    """
    assert run.eq_artifact(1, "whole.nc").name == "eq_1.whole.nc"
    assert run.eq_artifact(1, "solute.nc").name == "eq_1.solute.nc"
    assert run.eq_artifact(1, "dcd").name == "eq_1.dcd"
    assert run.eq_artifact(1, "energy_components.csv").name == "eq_1.energy_components.csv"
    assert run.eq_artifact(1, "mdout.csv").name == "eq_1.mdout.csv"


def test_an_equilibration_position_below_one_is_refused():
    with pytest.raises(ValueError, match="position starts at 1"):
        L.eq_stage_key(0)


# -- the layout does not redefine what amber_trajectory owns ------------------------------------

def test_the_per_state_names_come_from_the_one_module_that_owns_them():
    """Two modules naming one file is the defect this module exists to remove."""
    from md_tools.remd import amber_trajectory

    assert L.state_directory is amber_trajectory.state_directory
    assert L.state_system_name is amber_trajectory.state_system_name
    assert L.state_cv_name is amber_trajectory.state_cv_name
    assert L.state_restart_name is amber_trajectory.state_restart_name
    assert L.RANK_DIRECTORY == amber_trajectory.RANK_DIRECTORY


# -- the shape matches the migrated reference dataset -------------------------------------------

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference"


@pytest.mark.reference_data
@pytest.mark.skipif(not REFERENCE.is_dir(),
                    reason="the migrated reference dataset is not present (data/ is gitignored)")
@pytest.mark.parametrize("system,states,ranks,segments", [
    ("ALA-explicit-HMR", 6, 5, 5),
    ("ALA-implicit-HMR", 4, 3, 1),
    ("RGDfV-implicit-HMR", 6, 5, 1),
])
def test_the_layout_addresses_every_file_of_a_real_migrated_run(system, states, ranks, segments):
    """THE CHECK THAT MATTERS: the paths this module computes are the files that exist."""
    dataset = L.DatasetLayout(REFERENCE / system)
    if not dataset.root.is_dir():
        pytest.skip(f"{system} not migrated here")
    run = dataset.run("REST2", 1)

    assert dataset.built("xml").is_file(), dataset.built("xml")
    assert dataset.minimised("xml").is_file()
    assert dataset.stage_input("REST2").is_file()

    for state in range(states):
        for segment in range(1, segments + 1):
            for content in ("whole", "solute"):
                path = run.state_trajectory(state, content=content, segment=segment)
                assert path.is_file(), path
    for segment in range(1, segments + 1):
        for kind in ("ledger", "checkpoint", "restart", "runstate", "rem_log", "exchange"):
            assert run.record(kind, segment=segment).is_file(), (kind, segment)
    for rank in range(1, ranks + 1):
        assert run.rank_report(rank, segment=1).is_file(), rank
    for order in (1, 2, 3):
        assert run.eq_artifact(order, "xml").is_file(), order
