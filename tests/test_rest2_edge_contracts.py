"""The three edge contracts: velocity policy, reservoir replacement, and continuation mode.

Each test names the gap it pins, and each fails against
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0` -- the head this branch starts from.

PLATFORM_POLICY_EXEMPTION: declaration parsing, selection arithmetic, storage refusals and CLI
argument validation. Nothing here propagates dynamics; the runtime evidence stays in the
CUDA-marked files and in the smoke matrix recorded in the MD-project journal.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"
SRC = Path(__file__).resolve().parents[1] / "src"

openmm = pytest.importorskip("openmm")
sys.path.insert(0, str(TEMPLATES))
sys.path.insert(0, str(SRC))

import openmm_md                                                   # noqa: E402
import phase_space                                                 # noqa: E402
import hamiltonian_identity                                        # noqa: E402
import replica_storage as storage                                  # noqa: E402
import rrest2_reservoir                                            # noqa: E402
import source_ensemble                                             # noqa: E402
from replica_protocol import REST2Protocol                         # noqa: E402


def _protocol():
    return REST2Protocol(tau=[0.0, 0.25, 0.5], temperature_k=300.0, timestep_fs=2.0,
                         exchange_interval_ps=2.0, number_of_exchanges=4, random_seed=7)


def _declaration(tmp_path, **overrides):
    source = {"phase_space": "src/cmd.phase_space.nc", "start_time_ps": 0.0,
              "end_time_ps": 10.0, "frames": 3}
    source.update(overrides.pop("source", {}))
    document = {
        "format": rrest2_reservoir.DECLARATION_FORMAT,
        "weighting": "boltzmann", "ensemble": "NVT", "prepared_directory": "reservoir",
        "velocity_policy": "stored", "refresh_interval_exchanges": 2, "random_seed": 11,
        "source": source,
    }
    document.update(overrides)
    # The declaration lives in the STAGE directory (`<system>/rREST2/reservoir.yaml`), and
    # `source.phase_space` is relative to the system root above it. The fixture mirrors that.
    stage = tmp_path / "rREST2"
    stage.mkdir(exist_ok=True)
    path = stage / "reservoir.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _system(n_atoms=4):
    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0)
    force = openmm.NonbondedForce()
    for _ in range(n_atoms):
        force.addParticle(0.0, 0.1, 0.1)
    system.addForce(force)
    return system


def _phase_space_file(path, *, n_frames=4, n_atoms=4, velocity=1.0, periodic=False,
                      system=None, protocol=None):
    """A source in the versioned phase-space format -- the ONE source format, both policies."""
    identity = {"hamiltonian": {"system_sha256": "x"}}
    if system is not None:
        protocol = protocol or _protocol()
        identity = {"hamiltonian": hamiltonian_identity.identity_record(
            system, tau=protocol.tau[-1], temperature_k=protocol.temperature_k,
            ensemble="NVT", solute_indices=(), excluded_bonds=())}
    writer = phase_space.PhaseSpaceWriter(path, n_atoms=n_atoms, periodic=periodic,
                                          identity=identity)
    for index in range(n_frames):
        writer.append(positions=np.full((n_atoms, 3), 0.1 * (index + 1)),
                      velocities=np.full((n_atoms, 3), velocity),
                      box=(np.eye(3) * 2.0 if periodic else None),
                      step=(index + 1) * 500, time_ps=(index + 1) * 1.0)
    writer.close()
    return path


# --- 1. velocity_policy is ONE contract, end to end -------------------------------------------

def test_open_passes_the_requested_policy_into_preparation(tmp_path, monkeypatch):
    """The gap: `open()` parsed `velocity_policy` and then called `_prepare()` without it, so the
    helper's own default `"stored"` decided how the source was validated. An explicit `maxwell`
    declaration was prepared under `stored` rules and refused for velocities it never intended to
    use."""
    seen = {}

    @classmethod
    def spy(cls, *args, **kwargs):
        seen.update(kwargs)
        raise rrest2_reservoir.ReservoirError("stop here")

    monkeypatch.setattr(rrest2_reservoir.PreparedReservoir, "_prepare", spy)
    path = _declaration(tmp_path, velocity_policy="maxwell")
    with pytest.raises(rrest2_reservoir.ReservoirError):
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    assert seen.get("policy") == "maxwell", (
        "preparation was not told which velocity policy the declaration asked for")


def test_omitting_the_field_selects_stored(tmp_path, monkeypatch):
    seen = {}

    @classmethod
    def spy(cls, *args, **kwargs):
        seen.update(kwargs)
        raise rrest2_reservoir.ReservoirError("stop here")

    monkeypatch.setattr(rrest2_reservoir.PreparedReservoir, "_prepare", spy)
    path = _declaration(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.pop("velocity_policy")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(rrest2_reservoir.ReservoirError):
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    assert seen.get("policy") == "stored"


def test_an_unknown_policy_is_refused_and_never_defaulted(tmp_path):
    path = _declaration(tmp_path, velocity_policy="thermostat")
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    message = str(caught.value)
    assert "thermostat" in message and "stored" in message and "maxwell" in message


def test_zero_velocities_are_refused_under_stored_and_accepted_under_maxwell(tmp_path):
    """`maxwell` uses the source POSITIONS and BOX and deliberately ignores the recorded velocity
    values, so a source whose velocities are identically zero is usable under it and is not usable
    under `stored`. This is the whole reason the policy has to reach the validator."""
    path = _phase_space_file(tmp_path / "zero.nc", velocity=0.0)
    with phase_space.PhaseSpaceReader(path) as reader:
        under_stored = reader.validate(expect_atoms=4, expect_periodic=False,
                                       require_velocities=True)
        under_maxwell = reader.validate(expect_atoms=4, expect_periodic=False,
                                        require_velocities=False)
    assert under_stored, "a configuration-only frame is not a phase-space sample under `stored`"
    assert under_maxwell == [], "under `maxwell` the recorded velocity values are not used"


def test_no_code_path_turns_stored_into_maxwell(tmp_path):
    """There is no automatic fallback. The only way to redraw is to ask for it."""
    source = (TEMPLATES / "rrest2_reservoir.py").read_text(encoding="utf-8")
    driver = (TEMPLATES / "replica_driver.py").read_text(encoding="utf-8")
    for text, name in ((source, "rrest2_reservoir.py"), (driver, "replica_driver.py")):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert 'policy = "maxwell"' not in stripped, name
            assert "velocity_policy = 'maxwell'" not in stripped, name
            assert 'velocity_policy = "maxwell"' not in stripped, name


def test_the_source_format_is_the_same_for_both_policies():
    """`maxwell` is not DCD support. A DCD carries no Hamiltonian identity, no absolute source
    step, and no completion marker, so it cannot satisfy this reservoir contract under either
    policy."""
    source = (TEMPLATES / "rrest2_reservoir.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for claim in ("maxwell allows a dcd", "dcd is allowed under maxwell",
                  "coordinate-only source is supported"):
        assert claim not in lowered


# --- 2. no replacement when materialising an rREST2 reservoir ----------------------------------

def test_replacement_is_refused_before_anything_is_written(tmp_path):
    """The contradiction this closes: selection could draw one frame twice, materialisation sorted
    the indices, and the reader's own strictly-increasing-step invariant then rejected the file it
    had just written. Duplicates also give one configuration extra statistical weight while the
    declared frame count overstates the effective reservoir size."""
    directory = tmp_path / "src"
    directory.mkdir()
    _phase_space_file(directory / "cmd.phase_space.nc", n_frames=4)
    path = _declaration(tmp_path, source={"phase_space": "src/cmd.phase_space.nc",
                                          "start_time_ps": 0.0, "end_time_ps": 10.0,
                                          "frames": 3, "allow_sampling_with_replacement": True})
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=openmm.System())
    message = str(caught.value)
    assert "allow_sampling_with_replacement" in message
    assert not (tmp_path / "rREST2" / "reservoir" / "reservoir.nc").exists(), (
        "a prepared file was written for a request that must be refused")
    assert not (tmp_path / "rREST2" / "reservoir" / "reservoir.yaml").exists(), (
        "a manifest was left behind describing a reservoir that was never materialised")


def test_asking_for_more_frames_than_exist_says_what_is_available(tmp_path):
    directory = tmp_path / "src"
    directory.mkdir()
    system = _system(4)
    _phase_space_file(directory / "cmd.phase_space.nc", n_frames=3, system=system)
    path = _declaration(tmp_path, source={"phase_space": "src/cmd.phase_space.nc",
                                          "start_time_ps": 0.0, "end_time_ps": 10.0,
                                          "frames": 99})
    with pytest.raises(rrest2_reservoir.ReservoirError) as caught:
        rrest2_reservoir.PreparedReservoir.open(
            path, protocol=_protocol(), topology_path=tmp_path / "t.pdb", periodic=False,
            system=system)
    message = str(caught.value)
    assert "99" in message, "the message does not say what was requested"
    assert "3" in message, "the message does not say how many distinct frames are eligible"


def test_ais_replacement_behaviour_is_untouched():
    """The prohibition belongs at the rREST2 reservoir boundary, not in the shared selector. AIS
    has its own separately documented contract and still reaches the replacement path."""
    signature = source_ensemble.SourceRequest.__init__.__code__.co_varnames
    assert "allow_replacement" in signature
    text = (TEMPLATES / "source_ensemble.py").read_text(encoding="utf-8")
    assert "replace=request.allow_replacement" in text, (
        "the shared selector no longer honours replacement at all, which would change AIS")


# --- 3. continuation flags belong to the grouped runtime ---------------------------------------

class _Arguments:
    def __init__(self, **kwargs):
        self.groupfile = kwargs.get("groupfile")
        self.number_of_groups = kwargs.get("number_of_groups")
        self.exchange_rule = None
        self.reservoir = None
        self.force = kwargs.get("force", False)


class _Files:
    def __init__(self, tmp_path, *, resume=False, extend=0):
        self.input = str(tmp_path / "protocol.py")
        self.topology = str(tmp_path / "top.pdb")
        self.system = str(tmp_path / "system.xml")
        self.coordinates = str(tmp_path / "start.xml")
        self.output = str(tmp_path / "run.out")
        self.trajectory = str(tmp_path / "run.dcd")
        self.restart = None
        self.checkpoint = None
        self.solute_x = None
        self.rem = None
        self.resume = resume
        self.extend = extend


def _refusal(problems):
    return [p for p in problems if "groupfile" in p and ("resume" in p or "extend" in p)]


def test_non_grouped_resume_is_refused(tmp_path):
    """Only the grouped replica runtime consumes these flags. The conventional protocol path
    builds a fresh run -- it resets time and step state and creates its reporters from scratch --
    so accepting `--resume` there promised a continuation nothing implements."""
    problems = openmm_md.validate(_Files(tmp_path, resume=True), _Arguments(), rank=0, groups=None)
    assert _refusal(problems), problems


def test_non_grouped_extend_is_refused(tmp_path):
    problems = openmm_md.validate(_Files(tmp_path, extend=5), _Arguments(), rank=0, groups=None)
    assert _refusal(problems), problems


def test_the_refusal_never_suggests_deleting_data(tmp_path):
    problems = openmm_md.validate(_Files(tmp_path, resume=True), _Arguments(), rank=0, groups=None)
    assert _refusal(problems), "there is no refusal to inspect"
    text = " ".join(problems).lower()
    assert "rm " not in text and "delete" not in text and "overwrite" not in text


def test_grouped_resume_and_extend_are_still_accepted(tmp_path):
    """The supported path is unchanged: this milestone narrows the mode boundary, it does not
    take anything away from the replica runtime."""
    trajectory = tmp_path / "rest2.nc"
    trajectory.write_bytes(b"not empty")
    files = _Files(tmp_path, resume=True)
    files.trajectory = str(trajectory)
    arguments = _Arguments(groupfile=str(tmp_path / "rest2.group"), number_of_groups=2)
    groups = [{"group_index": 0}, {"group_index": 1}]
    assert not _refusal(openmm_md.validate(files, arguments, rank=0, groups=groups))

    files = _Files(tmp_path, extend=20)
    files.trajectory = str(trajectory)
    assert not _refusal(openmm_md.validate(files, arguments, rank=0, groups=groups))


def test_validation_happens_before_any_output_is_touched(tmp_path):
    """The refusal must be side-effect free: no directory created, no report opened, no protocol
    imported. `validate()` is a pure function over the parsed arguments, and `main()` returns on
    its problems before it makes a single directory."""
    outputs = tmp_path / "deep" / "nested"
    files = _Files(tmp_path, resume=True)
    files.output = str(outputs / "run.out")
    files.trajectory = str(outputs / "run.dcd")

    assert _refusal(openmm_md.validate(files, _Arguments(), rank=0, groups=None))
    assert not outputs.exists(), "validation created an output directory"

    source = (TEMPLATES / "openmm_md.py").read_text(encoding="utf-8")
    refusal_index = source.index("def validate(")
    mkdir_index = source.index("Path(value).parent.mkdir")
    import_index = source.index("load_protocol(files.input)")
    assert refusal_index < mkdir_index < import_index


def test_an_existing_output_is_untouched_by_a_refused_continuation(tmp_path):
    """The bytes a refused run must not disturb."""
    sentinel = tmp_path / "run.dcd"
    sentinel.write_bytes(b"authoritative bytes")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    files = _Files(tmp_path, resume=True)
    files.trajectory = str(sentinel)
    assert _refusal(openmm_md.validate(files, _Arguments(), rank=0, groups=None))
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


# --- Maxwell draws are deterministic and shared, not drawn per rank ----------------------------

def test_a_seeded_maxwell_draw_is_reproducible():
    """What the recorded seed is worth: the same seed gives the same momenta, so a refresh can be
    reproduced from the storage alone rather than by replaying the whole rule RNG."""
    from openmm import unit

    system = _system(8)
    drawn = []
    for _ in range(2):
        integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        context = openmm.Context(system, integrator,
                                 openmm.Platform.getPlatformByName("Reference"))
        # Spread out: eight particles stacked at the origin give an infinite
        # nonbonded energy, and the drawn state comes back as NaN.
        context.setPositions(np.arange(24, dtype=float).reshape(8, 3) * unit.nanometer)
        context.setVelocitiesToTemperature(300.0 * unit.kelvin, 4242)
        drawn.append(context.getState(getVelocities=True).getVelocities(asNumpy=True)
                     .value_in_unit(unit.nanometer / unit.picosecond))
    assert np.array_equal(drawn[0], drawn[1])
    assert np.any(drawn[0]), "a Maxwell draw at 300 K is not identically zero"


def test_only_the_owning_rank_draws_and_the_array_is_shared():
    """Serial and MPI must install the SAME momenta. Every rank drawing for itself would give
    each a different array from the same seed once the contexts differ, so the owning rank draws
    and the result is shared."""
    driver = (TEMPLATES / "replica_driver.py").read_text(encoding="utf-8")
    block = driver[driver.index("if installed_velocities is None:"):
                   driver.index("replacement = Configuration(")]
    assert "state_index in self.owned" in block, "any rank could draw"
    assert "allgather" in block, "the drawn array is not shared with the other ranks"
    assert block.index("state_index in self.owned") < block.index("allgather")


def test_the_storage_records_the_seed_each_maxwell_refresh_drew_from(tmp_path):
    import replica_storage as storage
    from replica_engine import Configuration

    path = tmp_path / "rest2.nc"
    reporter = storage.ReplicaReporter.create(
        path, n_states=2, n_atoms=4, n_solute_atoms=1, has_box=False,
        identity={"n_states": 2}, metadata={})
    try:
        zeros = np.zeros((2, 2), dtype=np.int64)
        reporter.write_exchange(0, step=500, time_ps=1.0, state_to_walker=[0, 1],
                                proposed=zeros, accepted=zeros, u=np.zeros((2, 2)),
                                u_evaluated=zeros.astype(np.int8),
                                reservoir=(1, 3, 2000, 1, 987654))
        reporter.write_exchange(1, step=1000, time_ps=2.0, state_to_walker=[0, 1],
                                proposed=zeros, accepted=zeros, u=np.zeros((2, 2)),
                                u_evaluated=zeros.astype(np.int8), reservoir=None)
    finally:
        reporter.close()

    reader = storage.ReplicaReporter(path, mode="r")
    try:
        seeds = reader.reservoir_velocity_seeds()
        assert list(seeds) == [987654, -1], (
            "the seed of a Maxwell refresh is not recoverable from the storage")
    finally:
        reader.close()
    assert Configuration is not None


def test_the_statistics_report_the_seeds_that_were_actually_used():
    import replica_statistics as statistics

    accepted = np.zeros((3, 2, 2), dtype=np.int64)
    proposed = np.zeros((3, 2, 2), dtype=np.int64)
    events = np.array([[-1, -1, -1, -1], [1, 4, 2000, 1], [1, 2, 3000, 1]])
    seeds = np.array([-1, 555, 556])
    stats = statistics.lifetime_statistics(accepted, proposed, tau=[0.0, 0.5],
                                           reservoir_events=events,
                                           reservoir_velocity_seeds=seeds)
    assert stats["reservoir"]["velocity_seeds_used"] == [555, 556]

    stored_mode = statistics.lifetime_statistics(
        accepted, proposed, tau=[0.0, 0.5], reservoir_events=events,
        reservoir_velocity_seeds=np.array([-1, -1, -1]))
    assert stored_mode["reservoir"]["velocity_seeds_used"] == [], (
        "under `stored` nothing is drawn, so no seed may be reported as used")


def test_the_prepared_manifest_names_the_policy_it_was_materialised_under():
    source = (TEMPLATES / "rrest2_reservoir.py").read_text(encoding="utf-8")
    manifest_block = source[source.index('"format": "md-templates-prepared-reservoir/v1"'):
                            source.index('"citations"')]
    assert '"velocity_policy": policy' in manifest_block, (
        "the manifest records only which policies the FORMAT supports, not the one selected")


# --- the co-generated source stage is not append-safe cMD -------------------------------------

def test_the_generated_cmd_launcher_passes_no_continuation_flag():
    """The fixed-tau stage an rREST2 project generates is an ordinary single-protocol cMD run. It
    must never be launched as though it could be continued."""
    from md_templates.openmm import emit

    launcher = emit.launcher("cmd", directory_var="CMD_TAU0P5_DIR", depth=1,
                             parent_restart="eq/nvt_free/nvt_free.state.xml",
                             trajectory=True, checkpoint=True)
    assert "--resume" not in launcher
    assert "--extend" not in launcher


def test_continuing_the_source_stage_is_refused_like_any_other_cmd(tmp_path):
    """Even though it lives inside an rREST2 project, the source stage is non-grouped, so the
    mode boundary applies to it unchanged."""
    files = _Files(tmp_path, resume=True)
    files.trajectory = str(tmp_path / "cMD_tau0p5" / "cmd.dcd")
    problems = openmm_md.validate(files, _Arguments(), rank=0, groups=None)
    assert _refusal(problems), problems


# --- validation must accept a correctly EXTENDED run -------------------------------------------
#
# Both of these were found by running `--verify-only` against a real twice-extended rREST2 run.
# Neither is one of this milestone's three contracts; both are pre-existing and were never
# exercised, because nothing had validated a run after `--extend`.

def test_a_descriptive_manifest_field_is_not_checked_as_a_filename(tmp_path):
    """`storage` mixes filenames with description. `coordinate_indexing: walker` says how the
    coordinates are indexed; treating it as a file failed every run with "walker does not exist
    beside the manifest"."""
    import replica_validate

    manifest = tmp_path / "restart.json"
    (tmp_path / "run.nc").write_bytes(b"x")
    record = {"storage": {"analysis_netcdf": "run.nc",
                          "schema": storage.SCHEMA_VERSION,
                          "authoritative": "analysis_netcdf",
                          "coordinate_indexing": "walker"}}
    result = replica_validate.ValidationResult()
    replica_validate._check_manifest_paths(manifest, record, result)
    assert result.ok, result.problems

    # And a key that IS a filename is still checked.
    record["storage"]["analysis_netcdf"] = "missing.nc"
    strict = replica_validate.ValidationResult()
    replica_validate._check_manifest_paths(manifest, record, strict)
    assert not strict.ok


def test_the_identity_budget_is_a_floor_not_an_equality(tmp_path):
    """`--extend` legitimately runs past the count the protocol was created with, so comparing the
    stored rows against the ORIGINAL request rejected every extended run. The authoritative budget
    is the schedule the run actually finished under."""
    source = (TEMPLATES / "replica_validate.py").read_text(encoding="utf-8")
    block = source[source.index('expected = identity.get("number_of_exchanges")'):
                   source.index('if stats["reservoir"]:')]
    assert "schedule" in block, "the actual budget is never consulted"
    assert 'stats["exchanges_committed"] != int(\n                expected)' not in block, (
        "the original request is still compared for equality")
    assert "< int(" in block, "the identity count is not treated as a floor"
