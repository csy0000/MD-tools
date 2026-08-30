"""The REST2 storage contract: NetCDF, walker/state views, resume, extension, refusal.

These integrate a molecular system, so they carry the `gpu` marker and run on CUDA, as every
propagating test in this repository does. They are picoseconds long: they check that the storage
and restart CONTRACT holds, and none of them is evidence about ladder quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"

openmm = pytest.importorskip("openmm")
openmmtools = pytest.importorskip("openmmtools")

sys.path.insert(0, str(TEMPLATES))
import rest2_runtime as runtime               # noqa: E402

from openmm import XmlSerializer, unit        # noqa: E402
from openmm.app import PDBFile                # noqa: E402

pytestmark = pytest.mark.gpu


def _cuda_or_skip():
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.skip("no CUDA platform; propagating tests are deselected, not passed, without one")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A small solvated peptide with the four files `openmm-rest2` takes, written once."""
    _cuda_or_skip()
    from openmmtools import testsystems

    directory = tmp_path_factory.mktemp("rest2_inputs")
    ts = testsystems.AlanineDipeptideExplicit(constraints=openmm.app.HBonds)
    system, topology, positions = ts.system, ts.topology, ts.positions

    with open(directory / "topology.pdb", "w") as handle:
        PDBFile.writeFile(topology, positions, handle)
    (directory / "system.xml").write_text(XmlSerializer.serialize(system))

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                 2 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 20260829)
    integrator.step(250)
    state = context.getState(getPositions=True, getVelocities=True)
    (directory / "coords.xml").write_text(XmlSerializer.serialize(state))
    del context, integrator

    solute_atoms = 22
    (directory / "solute.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "route": "peptide", "n_solute_atoms": solute_atoms,
        "solute_atom_range": [0, solute_atoms - 1],
        "solute_atom_indices_are_contiguous": True, "residues": [],
        "rest2": {"enhanced_region": "solute", "omega_excluded_bonds": [[6, 8]],
                  "omega_proline_like_scaled_bonds": [],
                  "omega_detection_method": "test fixture"},
    }, sort_keys=False))
    return directory, solute_atoms


def files_in(directory, prepared_dir, *, resume=False, extend=0):
    return SimpleNamespace(
        input=str(directory / "rest2.py"),
        topology=str(prepared_dir / "topology.pdb"),
        system=str(prepared_dir / "system.xml"),
        coordinates=str(prepared_dir / "coords.xml"),
        solute=str(prepared_dir / "solute.yaml"),
        output=str(directory / "rest2.out"),
        trajectory=str(directory / "rest2.nc"),
        restart=str(directory / "restart.json"),
        checkpoint=str(directory / "rest2_checkpoint.nc"),
        resume=resume, extend=extend)


def make(files, **overrides):
    settings = dict(
        tau_min=0.0, tau_max=0.5, number_of_replicas=3,
        exchange_interval_ps=0.02, solute_output_interval_ps=0.004,
        whole_output_interval_ps=0.02, equilibration_duration_ps=0.0,
        temperature_k=300.0, pressure_bar=1.0, timestep_fs=2.0, platform="CUDA")
    settings.update(overrides)
    return runtime.REST2(files, **settings)


@pytest.fixture(scope="module")
def completed(prepared, tmp_path_factory):
    """One finished 3-replica run, reused by the read-only assertions below."""
    prepared_dir, _ = prepared
    directory = tmp_path_factory.mktemp("rest2_run")
    (directory / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(directory, prepared_dir)
    record = make(files).run(number_of_exchanges=4)
    return directory, files, record


# --- the NetCDF contract ------------------------------------------------------------------------

def test_the_run_writes_both_netcdf_files_and_a_completed_manifest(completed):
    directory, files, record = completed
    assert Path(files.trajectory).is_file()
    assert Path(files.checkpoint).is_file()
    assert record["run_status"] == "completed"
    assert json.loads(Path(files.restart).read_text())["run_status"] == "completed"
    # 4 exchanges at a stride of 5 iterations.
    assert record["iterations_completed"] == 20
    assert record["exchange_attempts_recorded"] == 4


def test_the_manifest_references_the_netcdf_rather_than_duplicating_it(completed):
    _, files, record = completed
    storage = record["storage"]
    assert storage["analysis_netcdf"] == Path(files.trajectory).name
    assert storage["checkpoint_netcdf"] == Path(files.checkpoint).name
    assert storage["authoritative"] == "analysis_netcdf"
    # No coordinate or energy array is copied into the manifest.
    text = Path(files.restart).read_text()
    assert len(text) < 20000, "the manifest has grown to the size of the data it should reference"


def test_solute_frames_are_stored_every_iteration_and_are_all_distinct(completed, prepared):
    """The whole point of the strided model: real intermediate coordinates, never repeated.

    A design that wrote the exchange-boundary configuration five times would pass a frame-count
    check and fail this one.
    """
    from openmmtools.multistate import MultiStateReporter
    _, files, record = completed
    _, solute_atoms = prepared
    reporter = MultiStateReporter(files.trajectory, open_mode="r")
    try:
        frames = []
        for iteration in range(1, record["iterations_completed"] + 1):
            sampler_states = reporter.read_sampler_states(iteration=iteration,
                                                          analysis_particles_only=True)
            frames.append(np.array(
                sampler_states[0].positions.value_in_unit(unit.nanometer)))
        assert len(frames) == record["iterations_completed"]
        assert frames[0].shape == (solute_atoms, 3)
        displacements = [np.abs(frames[i + 1] - frames[i]).max() for i in range(len(frames) - 1)]
        assert all(d > 0 for d in displacements), (
            "consecutive solute frames were identical, which means a boundary configuration was "
            "written more than once instead of a real intermediate one")
    finally:
        reporter.close()


def test_full_coordinates_appear_only_on_checkpoint_iterations(completed):
    from openmmtools.multistate import MultiStateReporter
    _, files, record = completed
    reporter = MultiStateReporter(files.trajectory, open_mode="r")
    try:
        stride = record["scientific_identity"]["checkpoint_interval_iterations"]
        present, absent = [], []
        for iteration in range(1, record["iterations_completed"] + 1):
            states = reporter.read_sampler_states(iteration=iteration)
            (present if states is not None else absent).append(iteration)
        assert present == [i for i in range(1, record["iterations_completed"] + 1)
                           if i % stride == 0]
        assert absent, "every iteration held full coordinates, so the checkpoint interval did not apply"
    finally:
        reporter.close()


# --- walker and state views ---------------------------------------------------------------------

def test_walker_and_state_views_are_both_derivable_from_the_stored_mapping(completed):
    """OpenMMTools stores walkers plus a mapping; both views come from that, not a second copy."""
    from openmmtools.multistate import MultiStateReporter
    _, files, record = completed
    reporter = MultiStateReporter(files.trajectory, open_mode="r")
    try:
        mapping = np.asarray(reporter.read_replica_thermodynamic_states())
        n_iterations, n_replicas = mapping.shape
        assert n_replicas == record["scientific_identity"]["number_of_replicas"]

        # Every iteration is a PERMUTATION: each thermodynamic state is occupied exactly once.
        for row in mapping:
            assert sorted(row.tolist()) == list(range(n_replicas))

        # walker view: replica r's own trajectory is simply column r.
        walker = mapping[:, 0]
        assert walker.shape == (n_iterations,)

        # state view: which walker occupied state s at each iteration.
        for state in range(n_replicas):
            occupants = [int(np.where(row == state)[0][0]) for row in mapping]
            assert len(occupants) == n_iterations
            assert all(0 <= o < n_replicas for o in occupants)
    finally:
        reporter.close()


# --- resume and extension -----------------------------------------------------------------------

def test_extension_adds_exactly_the_requested_attempts(prepared, tmp_path):
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(tmp_path, prepared_dir)
    first = make(files).run(number_of_exchanges=3)
    assert first["exchange_attempts_recorded"] == 3

    extended_files = files_in(tmp_path, prepared_dir, extend=2)
    second = make(extended_files).run(number_of_exchanges=3)
    assert second["exchange_attempts_recorded"] == 5, "extension did not add exactly 2 attempts"
    assert second["iterations_completed"] == first["iterations_completed"] + 2 * 5


def test_resume_preserves_the_stored_mapping_history(prepared, tmp_path):
    """A resume must continue the SAME storage: the earlier iterations stay where they were."""
    from openmmtools.multistate import MultiStateReporter
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(tmp_path, prepared_dir)
    make(files).run(number_of_exchanges=2)

    reporter = MultiStateReporter(files.trajectory, open_mode="r")
    before = np.asarray(reporter.read_replica_thermodynamic_states()).copy()
    reporter.close()

    make(files_in(tmp_path, prepared_dir, extend=2)).run(number_of_exchanges=2)

    reporter = MultiStateReporter(files.trajectory, open_mode="r")
    after = np.asarray(reporter.read_replica_thermodynamic_states())
    reporter.close()
    assert after.shape[0] > before.shape[0]
    assert np.array_equal(after[:before.shape[0]], before), (
        "resuming rewrote iterations that had already been recorded")


def test_a_changed_scientific_configuration_is_refused_before_appending(prepared, tmp_path):
    """Running longer is a legitimate extension; changing the physics is not."""
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    make(files_in(tmp_path, prepared_dir)).run(number_of_exchanges=2)

    changed = make(files_in(tmp_path, prepared_dir, extend=1), tau_max=0.6)
    with pytest.raises(ValueError) as raised:
        changed.run(number_of_exchanges=2)
    message = str(raised.value)
    assert "scientific configuration changed" in message
    assert "taus" in message


def test_a_continuation_no_longer_needs_the_completion_manifest(prepared, tmp_path):
    """This asserted the opposite, and the opposite was the bug.

    Requiring `restart.json` to continue made an INTERRUPTED run unrecoverable, because an
    interrupted run is precisely the one that never wrote a manifest. The identity a continuation
    is checked against now lives in reporter metadata inside the authoritative storage, written
    before propagation began, so the manifest is evidence of completion and not a precondition for
    continuing.
    """
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(tmp_path, prepared_dir)
    make(files).run(number_of_exchanges=2)

    Path(files.restart).unlink()
    Path(runtime.run_state_path(files.trajectory)).unlink()   # only the NetCDF is left

    record = make(files_in(tmp_path, prepared_dir, extend=1)).run(number_of_exchanges=2)
    assert record["run_status"] == "completed"
    assert record["iterations_completed"] == 3 * 5           # 2 exchanges + 1, stride 5


def test_a_continuation_is_refused_when_no_identity_can_be_established(prepared, tmp_path,
                                                                      monkeypatch):
    """Storage whose identity cannot be read is refused rather than appended to on trust."""
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(tmp_path, prepared_dir)
    make(files).run(number_of_exchanges=2)
    Path(files.restart).unlink()
    Path(runtime.run_state_path(files.trajectory)).unlink()

    ladder = make(files_in(tmp_path, prepared_dir, extend=1))
    monkeypatch.setattr(type(ladder), "stored_identity",
                        lambda self, reporter: (None, "deliberately unavailable"))
    with pytest.raises(runtime.Rest2IdentityError, match="cannot be continued on trust"):
        ladder.run(number_of_exchanges=2)


def test_corrupt_storage_fails_clearly(prepared, tmp_path):
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    files = files_in(tmp_path, prepared_dir)
    make(files).run(number_of_exchanges=2)
    Path(files.trajectory).write_bytes(b"this is not a NetCDF file")

    with pytest.raises(Exception) as raised:
        make(files_in(tmp_path, prepared_dir, extend=1)).run(number_of_exchanges=2)
    assert not isinstance(raised.value, AssertionError)


def test_a_run_that_never_started_cannot_be_resumed(prepared, tmp_path):
    prepared_dir, _ = prepared
    (tmp_path / "rest2.py").write_text("# protocol placeholder\n")
    with pytest.raises(FileNotFoundError, match="nothing to"):
        make(files_in(tmp_path, prepared_dir, resume=True)).run(number_of_exchanges=2)


# --- the recorded provenance ----------------------------------------------------------------------

def test_the_manifest_records_exchange_ownership_and_the_omega_convention(completed):
    _, _, record = completed
    exchange = record["exchange"]
    # The default is stock OpenMMTools, so the manifest must say OpenMMTools decided the swaps.
    assert exchange["replica_mixing_scheme"] == "swap-all"
    assert exchange["decision_owner"] == "openmmtools"
    # What this repository DOES own is the schedule, and it says so separately.
    assert exchange["schedule_owner"] == "md-templates"
    assert exchange["exchange_stride_iterations"] == 5

    omega = record["omega_convention"]
    assert omega["name"] == "omega-selective REST2"
    assert omega["omega_bonds_left_unscaled"] == 1
    assert "not an unmodified standard REST2" in omega["statement"]

    assert record["hamiltonian"]["is_temperature_remd"] is False
    assert record["hamiltonian"]["single_temperature_k"] == 300.0
    assert record["move"]["integrator"] == "openmm.LangevinMiddleIntegrator"

    versions = record["versions"]
    assert versions["openmmtools"] == "0.26.0"
    assert versions["netCDF4"] and versions["openmm"]
