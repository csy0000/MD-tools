"""The owned replica-exchange runtime: one executor, group files, rules, reservoirs, storage.

PLATFORM_POLICY_EXEMPTION: these are parsing, arithmetic and storage contract tests. The few that
propagate use a 22-particle vacuum peptide for a handful of steps to exercise bookkeeping, not to
produce scientific evidence; the task contract requires them to run without a GPU. The suite's
scientific runtime evidence stays in the CUDA-marked files.
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "openmm" / "templates"
SRC = Path(__file__).resolve().parents[1] / "src"

openmm = pytest.importorskip("openmm")
sys.path.insert(0, str(TEMPLATES))

import exchange_rules                     # noqa: E402
import replica_executor                          # noqa: E402
import replica_statistics as statistics   # noqa: E402
import replica_storage as storage         # noqa: E402
import replica_validate as validate       # noqa: E402
import rrest2_reservoir                   # noqa: E402
import source_ensemble                    # noqa: E402
from replica_engine import Configuration, exchange_log_acceptance   # noqa: E402
from replica_protocol import ProtocolError, REST2Protocol           # noqa: E402


# --- one executor -----------------------------------------------------------------------------

def test_openmm_rest2_is_gone_from_the_package():
    """There is one simulation executor. A second one is a second thing to keep consistent."""
    assert not (TEMPLATES / "openmm_rest2.py").exists()
    for stale in ("rest2_runtime.py", "rest2_openmmtools.py", "rest2_validate.py",
                  "rest2_statistics.py"):
        assert not (TEMPLATES / stale).exists(), f"{stale} belonged to the retired executor"


def test_no_source_file_still_advertises_openmm_rest2():
    offenders = []
    for path in list(SRC.rglob("*.py")) + [Path(__file__).parents[1] / "README.md"]:
        if "egg-info" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "openmm-rest2" in text or "openmm_rest2" in text:
            offenders.append(path.name)
    assert not offenders, f"these still name the retired executor: {offenders}"


def test_the_packaged_console_scripts_do_not_add_a_method_command():
    """`md-openmm` is the ONLY executable this distribution installs.

    The migration to MD-tools retired `md-template` (the environment installer) and never
    introduced `openmm-md`; a method-named executable would put the protocol in the command
    surface instead of in the configuration, which is what this has always guarded against.
    """
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert "md-openmm =" in pyproject
    assert "md-template =" not in pyproject
    assert "openmm-md" not in pyproject
    assert "openmm-rest2" not in pyproject


# --- the group file ---------------------------------------------------------------------------

def _group_file(tmp_path, lines):
    path = tmp_path / "x.group"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_group_file_is_parsed_with_shlex_and_never_evaluated(tmp_path):
    """Quoting is shell-LIKE; execution is not. A data file that can run commands is a hole."""
    marker = tmp_path / "should_not_exist"
    path = _group_file(tmp_path, [
        f'-i "a b.py" -p t.pdb -s s.xml -c "$(touch {marker})" --group-index 0'])
    groups = replica_executor.parse_group_file(path)
    assert groups[0]["input"] == "a b.py", "quoted values must survive parsing"
    # shlex leaves the substitution as literal text; a shell would have run it.
    assert "$(touch" in groups[0]["coordinates"]
    assert not marker.exists(), "the group file was evaluated by a shell"


def test_group_parsing_uses_shlex_in_the_source():
    source = (TEMPLATES / "replica_executor.py").read_text()
    tree = ast.parse(source)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "shlex" in imported
    for forbidden in ("os.system", "subprocess", "eval(", "shell=True"):
        assert forbidden not in source, f"the executor uses {forbidden}"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = _group_file(tmp_path, [
        "# a comment", "",
        "-i a.py -p t.pdb -s s.xml -c c.xml --group-index 0",
        "   ",
        "-i a.py -p t.pdb -s s.xml -c c.xml --group-index 1"])
    assert len(replica_executor.parse_group_file(path)) == 2


@pytest.mark.parametrize("line,fragment", [
    ("-i a.py -p t.pdb -s s.xml -c c.xml -o out.txt --group-index 0", "belongs on the outer"),
    ("-i a.py -p t.pdb -s s.xml -c c.xml -x run.nc --group-index 0", "belongs on the outer"),
    ("-i a.py -p t.pdb -s s.xml -c c.xml --wat 1 --group-index 0", "unknown group field"),
    ("-i a.py -p t.pdb -s s.xml -c c.xml --group-index", "has no value"),
    ("-i a.py -i b.py -p t.pdb -s s.xml -c c.xml --group-index 0", "more than once"),
    ("-i a.py -p t.pdb -s s.xml -c c.xml", "no --group-index"),
    ("-p t.pdb -s s.xml -c c.xml --group-index 0", "no input"),
])
def test_a_bad_group_line_is_refused_by_line_number(tmp_path, line, fragment):
    path = _group_file(tmp_path, [line])
    with pytest.raises(replica_executor.GroupFileError) as raised:
        replica_executor.parse_group_file(path)
    message = str(raised.value)
    assert fragment in message
    assert ":1:" in message or str(path) in message


def test_group_indices_must_be_unique_and_contiguous(tmp_path):
    path = _group_file(tmp_path, [
        "-i a.py -p t.pdb -s s.xml -c c.xml --group-index 0",
        "-i a.py -p t.pdb -s s.xml -c c.xml --group-index 2"])
    with pytest.raises(replica_executor.GroupFileError, match="unique, contiguous and zero-based"):
        replica_executor.parse_group_file(path)


def test_an_empty_group_file_is_refused(tmp_path):
    path = _group_file(tmp_path, ["# nothing here"])
    with pytest.raises(replica_executor.GroupFileError, match="no group lines"):
        replica_executor.parse_group_file(path)


# --- mode validation ---------------------------------------------------------------------------

def _files(**overrides):
    base = dict(input=None, topology=None, system=None, coordinates=None, output="run.out",
                trajectory="run.nc", restart="restart.json", checkpoint="run_chk.nc",
                solute_x=None, resume=False, extend=0)
    base.update(overrides)
    return SimpleNamespace(**base)


def _arguments(**overrides):
    base = dict(groupfile=None, number_of_groups=None, exchange_rule=None, reservoir=None,
                resume=False, extend=0, force=False, verify_only=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_groupfile_requires_ng():
    problems = replica_executor.validate(_files(), _arguments(groupfile="g.group"), groups=[{}, {}])
    assert any("--groupfile requires -ng" in p for p in problems)


def test_ng_without_a_groupfile_is_refused():
    problems = replica_executor.validate(_files(input="a.py"), _arguments(number_of_groups=4))
    assert any("-ng describes a group file" in p for p in problems)


def test_ng_must_equal_the_number_of_group_lines():
    groups = [{"group_index": i, "input": "a.py"} for i in range(3)]
    problems = replica_executor.validate(_files(), _arguments(groupfile="g", number_of_groups=6),
                                  groups=groups)
    assert any("but the group file has 3 group line(s)" in p for p in problems)


def test_grouped_only_flags_are_refused_in_single_mode():
    problems = replica_executor.validate(_files(input="a.py"), _arguments(exchange_rule="r.py"))
    assert any("--exchange-rule applies to a coordinated run" in p for p in problems)
    problems = replica_executor.validate(_files(input="a.py"), _arguments(reservoir="r.yaml"))
    assert any("--reservoir applies to a coordinated run" in p for p in problems)


def test_a_grouped_run_requires_the_analysis_storage():
    groups = [{"group_index": 0, "input": "a.py"}]
    problems = replica_executor.validate(_files(trajectory=None), _arguments(groupfile="g",
                                                                     number_of_groups=1),
                                  groups=groups)
    assert any("-x is required in grouped mode" in p for p in problems)


def test_force_never_combines_with_a_continuation():
    problems = replica_executor.validate(_files(resume=True), _arguments(groupfile="g",
                                                                  number_of_groups=1, force=True),
                                  groups=[{"group_index": 0}])
    assert any("--force replaces a new run's outputs" in p for p in problems)


def test_a_group_input_that_is_also_an_output_is_refused(tmp_path):
    shared = str(tmp_path / "same.nc")
    groups = [{"group_index": 0, "input": "a.py", "topology": shared}]
    problems = replica_executor.validate(_files(trajectory=shared),
                                  _arguments(groupfile="g", number_of_groups=1), groups=groups)
    assert any("are the same file" in p for p in problems)


def test_two_outputs_may_not_be_the_same_file(tmp_path):
    shared = str(tmp_path / "same.nc")
    problems = replica_executor.validate(_files(trajectory=shared, checkpoint=shared),
                                  _arguments(groupfile="g", number_of_groups=1),
                                  groups=[{"group_index": 0}])
    assert any("are the same file" in p for p in problems)


def test_the_mpi_world_must_match_the_group_count(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    groups = [{"group_index": i} for i in range(6)]
    problems = replica_executor.validate(_files(), _arguments(groupfile="g", number_of_groups=6),
                                  groups=groups)
    assert any("one rank per group" in p for p in problems)


# --- the neighbouring rule ---------------------------------------------------------------------

def test_the_pairing_covers_every_adjacent_pair_over_two_phases():
    for n_states in (2, 3, 4, 6, 7):
        covered = set()
        for attempt in range(2):
            phase = exchange_rules.deterministic_phase(n_states, attempt)
            covered.update(exchange_rules.alternating_pairs(n_states, phase))
        assert covered == {(i, i + 1) for i in range(n_states - 1)}, n_states


def test_a_two_state_ladder_never_gets_an_empty_sweep():
    for attempt in range(20):
        phase = exchange_rules.deterministic_phase(2, attempt)
        assert exchange_rules.alternating_pairs(2, phase), "a two-state sweep proposed nothing"


def test_the_phase_alternates_on_exchange_attempts_not_iterations():
    """Exchanges land every `stride` iterations; an even stride freezes the iteration parity."""
    phases = [exchange_rules.deterministic_phase(6, attempt) for attempt in range(6)]
    assert phases == [0, 1, 0, 1, 0, 1]


def test_the_log_acceptance_matches_the_analytical_criterion():
    rng = np.random.RandomState(20260830)
    for _ in range(200):
        u_ii, u_jj, u_ij, u_ji = rng.normal(size=4) * 5.0
        analytical = (u_ii + u_jj) - (u_ij + u_ji)
        assert exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji) == pytest.approx(analytical)


class _Context:
    """The minimum an exchange rule may see, with a fixed reduced-potential matrix."""

    def __init__(self, matrix, *, n_states, seed, exchange_index=0, reservoir=None,
                 rule_state=None):
        self.matrix = np.asarray(matrix, dtype=float)
        self._n = n_states
        self.iteration = 1
        self.segment = 1
        self.exchange_index = exchange_index
        self.state_to_walker = list(range(n_states))
        self.rng = np.random.default_rng(seed)
        self.reservoir = reservoir
        self.rule_state = dict(rule_state or {})
        self.protocol = SimpleNamespace(n_states=n_states)

    @property
    def n_states(self):
        return self._n

    def reduced_potential(self, state, walker):
        return float(self.matrix[state][walker])


def test_a_certain_swap_is_always_accepted_and_an_impossible_one_never_is():
    rule = exchange_rules.NeighbouringExchangeRule()
    favourable = np.array([[0.0, 100.0], [100.0, 0.0]])
    outcome = rule.propose(_Context(favourable, n_states=2, seed=1))
    assert outcome.swaps == [], "log_alpha is negative here; the swap must be rejected"

    swapped = np.array([[100.0, 0.0], [0.0, 100.0]])
    outcome = rule.propose(_Context(swapped, n_states=2, seed=1))
    assert outcome.swaps == [(0, 1)], "log_alpha is positive here; the swap must be accepted"


def test_exchange_decisions_repeat_exactly_under_a_fixed_seed():
    rng = np.random.RandomState(7)
    matrix = rng.normal(size=(6, 6))
    rule = exchange_rules.NeighbouringExchangeRule()
    first = rule.propose(_Context(matrix, n_states=6, seed=99))
    second = rule.propose(_Context(matrix, n_states=6, seed=99))
    assert first.proposals == second.proposals
    assert first.swaps == second.swaps
    different = rule.propose(_Context(matrix, n_states=6, seed=100))
    assert isinstance(different.proposals, list)


def test_every_pair_in_one_sweep_is_disjoint():
    """A state cannot be swapped twice in one sweep, so the mapping stays a permutation."""
    for n_states in (4, 6, 7):
        for phase in (0, 1):
            pairs = exchange_rules.alternating_pairs(n_states, phase)
            touched = [state for pair in pairs for state in pair]
            assert len(touched) == len(set(touched)), (n_states, phase)


def test_the_rule_contract_is_not_a_registry():
    source = (TEMPLATES / "exchange_rules.py").read_text()
    for forbidden in ("entry_points", "register(", "REGISTRY", "importlib.metadata"):
        assert forbidden not in source, f"exchange_rules grew a {forbidden}"


def test_an_exchange_rule_file_must_define_a_rule(tmp_path):
    path = tmp_path / "empty_rule.py"
    path.write_text("# nothing here\n")
    with pytest.raises(exchange_rules.ExchangeRuleError, match="defines neither"):
        exchange_rules.load_rule(path)


def test_a_missing_exchange_rule_file_is_refused(tmp_path):
    with pytest.raises(exchange_rules.ExchangeRuleError, match="does not exist"):
        exchange_rules.load_rule(tmp_path / "absent.py")


def test_a_loaded_rule_records_its_file_digest(tmp_path):
    path = tmp_path / "rule.py"
    shutil.copy2(TEMPLATES / "rrest2_exchange.py", path)
    rule, identity = exchange_rules.load_rule(path)
    assert identity["name"] == "rrest2-boltzmann"
    assert len(identity["sha256"]) == 64
    assert identity["describe"]["refresh_order"] == "after_neighbouring_sweep"


# --- the protocol -------------------------------------------------------------------------------

def test_the_protocol_refuses_a_descending_or_duplicated_ladder():
    common = dict(temperature_k=300.0, timestep_fs=2.0,
                  exchange_interval_ps=2.0, number_of_exchanges=1)
    with pytest.raises(ProtocolError, match="ascending order"):
        REST2Protocol(tau=[0.5, 0.0], **common)
    with pytest.raises(ProtocolError, match="distinct"):
        REST2Protocol(tau=[0.0, 0.0], **common)


def test_the_protocol_refuses_a_pressure_because_this_runtime_is_nvt():
    protocol = REST2Protocol(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=2.0,
                             exchange_interval_ps=2.0, number_of_exchanges=1, pressure_bar=1.0)
    with pytest.raises(ProtocolError, match="NVT and installs no barostat"):
        protocol.build_systems(openmm.System(), [0])


def test_intervals_must_convert_exactly():
    common = dict(tau=[0.0, 0.5], temperature_k=300.0, timestep_fs=4.0, number_of_exchanges=1)
    # Each schedule is independent now, so each is checked on its own terms: an interval that is
    # not a whole number of integration steps is refused rather than rounded into one.
    with pytest.raises(ProtocolError, match="whole number"):
        REST2Protocol(exchange_interval_ps=0.006, **common)
    with pytest.raises(ProtocolError, match="whole number"):
        REST2Protocol(exchange_interval_ps=2.0, solute_output_interval_ps=0.003, **common)
    with pytest.raises(ProtocolError, match="whole number"):
        REST2Protocol(exchange_interval_ps=2.0, whole_output_interval_ps=0.003, **common)


def test_the_protocol_records_that_it_is_not_temperature_remd():
    described = REST2Protocol(tau=[0.0, 0.25, 0.5], temperature_k=300.0, timestep_fs=2.0,
                              exchange_interval_ps=2.0,
                              number_of_exchanges=3).describe()
    assert described["is_temperature_remd"] is False
    assert described["single_temperature"] is True
    assert described["velocity_rescaling_on_exchange"] is False


def test_no_runtime_module_rescales_velocities_on_exchange():
    """Velocity rescaling belongs to temperature REMD and would inject energy here."""
    for name in ("replica_driver.py", "exchange_rules.py", "replica_engine.py"):
        source = (TEMPLATES / name).read_text()
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        assert "setVelocitiesToTemperature" not in code or name == "replica_engine.py", name
    # The one place it appears is the reservoir path, where a DCD supplies no velocities at all.
    driver = (TEMPLATES / "replica_driver.py").read_text()
    assert "set_velocities_to_temperature" in driver
    assert driver.index("set_velocities_to_temperature") > driver.index("_apply_reservoir")


# --- statistics and views -------------------------------------------------------------------------

def test_the_walker_and_state_views_invert_each_other():
    mapping = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]])
    inverse = statistics.walker_view(mapping)
    assert np.array_equal(statistics.walker_view(inverse), mapping)


def test_every_mapping_row_must_be_a_permutation():
    good = np.array([[0, 1, 2], [2, 0, 1]])
    bad = np.array([[0, 1, 2], [1, 1, 2]])
    assert statistics.mapping_is_permutation_every_iteration(good, n_states=3)[0]
    ok, offenders = statistics.mapping_is_permutation_every_iteration(bad, n_states=3)
    assert not ok and offenders == [1]


def test_lifetime_statistics_sum_the_whole_history():
    accepted = np.zeros((5, 2, 2), dtype=np.int64)
    proposed = np.zeros((5, 2, 2), dtype=np.int64)
    for index, (a, p) in enumerate([(0, 0), (1, 4), (0, 0), (3, 6), (0, 0)]):
        accepted[index, 0, 1] = accepted[index, 1, 0] = a
        proposed[index, 0, 1] = proposed[index, 1, 0] = p
    stats = statistics.lifetime_statistics(accepted, proposed, tau=[0.0, 0.5])
    pair = stats["by_state_pair"][0]
    assert pair["proposed"] == 10 and pair["accepted"] == 4
    # Separate schedules mean every stored row IS an attempt: no zero rows stand for skipped
    # propagation, and nothing is inferred from a stride.
    assert stats["exchanges_committed"] == 5


def test_a_reservoir_refresh_is_never_counted_as_an_exchange():
    accepted = np.zeros((3, 2, 2), dtype=np.int64)
    proposed = np.zeros((3, 2, 2), dtype=np.int64)
    # columns: state, frame, source step, outcome (-1 = no refresh at that row)
    events = np.array([[-1, -1, -1, -1], [1, 4, 2000, 1], [-1, -1, -1, -1]])
    stats = statistics.lifetime_statistics(accepted, proposed, tau=[0.0, 0.5],
                                           reservoir_events=events)
    assert stats["total_proposed"] == 0
    assert stats["reservoir"]["attempts"] == 1 and stats["reservoir"]["accepted"] == 1
    assert "NOT a swap" in stats["reservoir"]["note"]


def test_round_trip_counting_is_not_offered_here():
    """It is downstream analysis, and it was deliberately removed from this layer.

    The committed mapping is preserved in the authoritative NetCDF, so anyone who wants round
    trips can compute them from it with their own burn-in and window choices -- which is the point:
    those choices are not ours to make silently.
    """
    for gone in ("count_round_trips", "round_trip_report"):
        assert not hasattr(statistics, gone), f"{gone} belongs downstream, not here"


# --- the source ensemble, shared by AIS and rREST2 ------------------------------------------------

def test_source_tau_never_comes_from_the_directory_name(tmp_path):
    """`cMD_tau0p5` is a name a reader finds helpful. It is not evidence."""
    directory = tmp_path / "cMD_tau0p5"
    directory.mkdir()
    trajectory = directory / "cmd.dcd"
    trajectory.write_bytes(b"")
    request = source_ensemble.SourceRequest(
        project=tmp_path, prepared_topology=tmp_path / "t.pdb", trajectory=str(trajectory),
        start_time_ps=0.0, end_time_ps=1.0, count=1, seed=1, implicit=True, required_tau=0.5)
    with pytest.raises(source_ensemble.SourceError) as raised:
        source_ensemble.source_tau(request, trajectory)
    message = str(raised.value)
    assert "is NOT evidence" in message and "cMD_tau0p5" in message


def test_a_conflicting_declared_and_recorded_tau_is_refused(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "resolved_run.yaml").write_text(json.dumps({"tau": 0.5}))
    trajectory = directory / "cmd.dcd"
    trajectory.write_bytes(b"")
    request = source_ensemble.SourceRequest(
        project=tmp_path, prepared_topology=tmp_path / "t.pdb", trajectory=str(trajectory),
        start_time_ps=0.0, end_time_ps=1.0, count=1, seed=1, implicit=True, required_tau=0.4,
        declared_tau=0.4)
    with pytest.raises(source_ensemble.SourceError, match="These must agree"):
        source_ensemble.source_tau(request, trajectory)


def test_frame_timing_is_refused_without_evidence(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    trajectory = directory / "cmd.dcd"
    trajectory.write_bytes(b"")
    request = source_ensemble.SourceRequest(
        project=tmp_path, prepared_topology=tmp_path / "t.pdb", trajectory=str(trajectory),
        start_time_ps=0.0, end_time_ps=1.0, count=1, seed=1, implicit=True, required_tau=0.5)
    with pytest.raises(source_ensemble.SourceError, match="A frame index is not a time"):
        source_ensemble.source_frame_timing(request, trajectory, 10)


def test_the_window_is_inclusive_at_both_ends():
    times = [1.0 + index for index in range(10)]
    request = source_ensemble.SourceRequest(
        project=Path("."), prepared_topology=Path("t.pdb"), trajectory="x.dcd",
        start_time_ps=1.0, end_time_ps=10.0, count=3, seed=1, implicit=True, required_tau=0.5)
    selected, eligible = source_ensemble.select_source_frames(
        request, times, {"frame_interval_ps": 1.0, "source": "test"})
    assert eligible[0] == 0 and eligible[-1] == 9 and len(eligible) == 10
    assert len(set(selected)) == 3


def test_ais_and_rrest2_select_the_same_frames_for_the_same_request_and_seed():
    """One implementation, so two callers asking the same question get the same answer."""
    times = [1.0 + index for index in range(50)]
    evidence = {"frame_interval_ps": 1.0, "source": "test"}
    shared = dict(project=Path("."), prepared_topology=Path("t.pdb"), trajectory="x.dcd",
                  start_time_ps=1.0, end_time_ps=50.0, count=8, seed=20260830, implicit=True,
                  required_tau=0.5)
    ais = source_ensemble.SourceRequest(**shared, purpose="shared",
                                        selection_purpose=("shared", "selection"))
    rrest2 = source_ensemble.SourceRequest(**shared, purpose="shared",
                                           selection_purpose=("shared", "selection"))
    first, _ = source_ensemble.select_source_frames(ais, times, evidence)
    second, _ = source_ensemble.select_source_frames(rrest2, times, evidence)
    assert first == second


def test_the_shared_seed_derivation_is_the_repositorys_own():
    """AIS projects must keep selecting the configurations they always did."""
    def historical(base, *purpose):
        value = int(base)
        for part in purpose:
            for byte in str(part).encode("utf-8"):
                value = (value * 1000003 + byte) & 0xFFFFFFFF
        return (value % (2 ** 31 - 1)) or 1

    for args in [(20260827, "AIS", "source-selection"), (1, "x"), (99, "a", "b")]:
        assert source_ensemble.derive_seed(*args) == historical(*args)


def test_a_production_source_is_read_with_bounded_iterload_and_never_hashed():
    source = (TEMPLATES / "source_ensemble.py").read_text()
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    assert "mdtraj.iterload" in code
    assert "mdtraj.load(" not in code, "mdtraj.load reads the whole production trajectory"
    # sha256_file exists for SMALL prepared files; it must not be applied to the source survey.
    survey = code[code.index("def survey_source"):code.index("def collect_frames")]
    assert "sha256" not in survey


def test_the_prepared_manifest_records_that_the_source_was_not_hashed():
    source = (TEMPLATES / "source_ensemble.py").read_text()
    assert '"hashed": False' in source
    assert "never hashed" in source


def test_two_box_representations_of_one_lattice_compare_equal():
    a = np.array([[3.0, 0, 0], [0, 3.0, 0], [1.5, 1.5, 2.12]])
    b = np.array([[3.0, 0, 0], [0, 3.0, 0], [-1.5, -1.5, 2.12]])
    assert source_ensemble.same_lattice(a, b)
    assert not source_ensemble.same_lattice(a, np.array([[3.1, 0, 0], [0, 3.1, 0],
                                                         [-1.55, -1.55, 2.2]]))


# --- the reservoir ---------------------------------------------------------------------------------

def _declaration(tmp_path, **overrides):
    document = {
        "format": rrest2_reservoir.DECLARATION_FORMAT,
        "weighting": "boltzmann", "ensemble": "NVT", "prepared_directory": "reservoir",
        "refresh_interval_exchanges": 2, "random_seed": 1,
        "velocity_policy": "stored",
        "source": {"phase_space": "src/cmd.phase_space.nc", "start_time_ps": 0.0,
                   "end_time_ps": 10.0, "frames": 5},
    }
    document.update(overrides)
    path = tmp_path / "reservoir.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def _protocol():
    return REST2Protocol(tau=[0.0, 0.25, 0.5], temperature_k=300.0, timestep_fs=2.0,
                         exchange_interval_ps=2.0, number_of_exchanges=2)


def test_a_non_boltzmann_reservoir_is_refused_by_the_v1_rule(tmp_path):
    path = _declaration(tmp_path, weighting="clustered")
    with pytest.raises(rrest2_reservoir.ReservoirError) as raised:
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())
    message = str(raised.value)
    assert "not implemented" in message and "separately derived acceptance rule" in message


def test_an_npt_reservoir_is_refused(tmp_path):
    path = _declaration(tmp_path, ensemble="NPT")
    with pytest.raises(rrest2_reservoir.ReservoirError, match="distribution of volumes"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())


def test_solute_only_insertion_is_refused(tmp_path):
    path = _declaration(tmp_path, solute_only=True)
    with pytest.raises(rrest2_reservoir.ReservoirError, match="solute into an unrelated solvent"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())


def test_a_declaration_of_the_wrong_format_is_refused(tmp_path):
    path = _declaration(tmp_path, format="something-else/v9")
    with pytest.raises(rrest2_reservoir.ReservoirError, match="is 'something-else/v9'"):
        rrest2_reservoir.PreparedReservoir.open(path, protocol=_protocol(),
                                                topology_path=tmp_path / "t.pdb", periodic=False,
                                                system=openmm.System())


def test_the_rrest2_rule_refreshes_only_the_top_state():
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 1}, n_frames=4)
    matrix = np.zeros((4, 4))
    state = {}
    seen = []
    for attempt in range(4):
        context = _Context(matrix, n_states=4, seed=5, exchange_index=attempt,
                           reservoir=reservoir, rule_state=state)
        outcome = rule.propose(context)
        state = outcome.rule_state
        if outcome.reservoir_refresh:
            seen.append(outcome.reservoir_refresh["state"])
    assert seen and set(seen) == {3}, "a refresh must only ever touch the hottest state"


def test_the_rrest2_refresh_interval_is_honoured():
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 3}, n_frames=4)
    state, refreshes = {}, 0
    for attempt in range(9):
        outcome = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=2,
                                        exchange_index=attempt, reservoir=reservoir,
                                        rule_state=state))
        state = outcome.rule_state
        refreshes += outcome.reservoir_refresh is not None
    assert refreshes == 3, "an interval of 3 over 9 exchanges must refresh three times"


def test_the_refresh_order_is_recorded_not_left_to_scheduling():
    rule, identity = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")
    assert identity["describe"]["refresh_order"] == "after_neighbouring_sweep"
    source = (TEMPLATES / "rrest2_exchange.py").read_text()
    assert "REFRESH_ORDER" in source


def test_a_refresh_carries_a_recorded_velocity_seed():
    """A DCD has no velocities, so momenta are redrawn -- reproducibly."""
    rule = exchange_rules.load_rule(TEMPLATES / "rrest2_exchange.py")[0]
    reservoir = SimpleNamespace(declaration={"refresh_interval_exchanges": 1}, n_frames=4)
    first = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=11, reservoir=reservoir))
    second = rule.propose(_Context(np.zeros((3, 3)), n_states=3, seed=11, reservoir=reservoir))
    assert first.reservoir_refresh["velocity_seed"] == second.reservoir_refresh["velocity_seed"]
    assert first.reservoir_refresh["frame"] == second.reservoir_refresh["frame"]


# --- storage and validation -----------------------------------------------------------------------

def _identity(n_states=3, n_exchanges=4, exchange_steps=500):
    """The scientific identity in the vocabulary the storage now uses: absolute steps and a
    per-stream schedule, with no `segment` and no exchange stride to infer anything from."""
    return {"n_states": n_states, "tau": [0.0, 0.25, 0.5][:n_states],
            "number_of_exchanges": n_exchanges,
            "schedule": {"exchange_steps": exchange_steps,
                         "total_steps": exchange_steps * n_exchanges},
            "temperature_k": 300.0, "is_temperature_remd": False}


def build_storage(tmp_path, *, n_exchanges=4, n_states=3, n_atoms=4, has_box=True,
                  mapping=None, identity=None, omit_exchange_at=(), exchange_steps=500,
                  whole_every=2):
    """A synthetic run in the current schema.

    Every stored exchange row IS an attempt -- there are no placeholder rows -- and the whole and
    solute streams are written on their own schedules into their own dimensions.
    """
    path = tmp_path / "run.nc"
    identity = identity or _identity(n_states, n_exchanges, exchange_steps)
    reporter = storage.ReplicaReporter.create(
        path, n_states=n_states, n_atoms=n_atoms, n_solute_atoms=1, has_box=has_box,
        identity=identity, metadata={"note": "synthetic"})
    reporter.write_ladder(identity["tau"])
    configurations = [Configuration(np.zeros((n_atoms, 3)), np.zeros((n_atoms, 3)),
                                    np.eye(3) if has_box else None) for _ in range(n_states)]
    written = -1
    for attempt in range(n_exchanges):
        if (attempt + 1) in omit_exchange_at:
            continue
        step = (attempt + 1) * exchange_steps
        proposed = np.zeros((n_states, n_states), dtype=np.int64)
        accepted = np.zeros((n_states, n_states), dtype=np.int64)
        u = np.zeros((n_states, n_states))
        evaluated = np.ones((n_states, n_states), dtype=np.int8)
        proposed[0, 1] = proposed[1, 0] = 1
        row = (mapping[attempt] if mapping is not None else list(range(n_states)))
        written += 1
        reporter.write_exchange(written, step=step, time_ps=step * 0.002,
                                state_to_walker=row, proposed=proposed, accepted=accepted,
                                u=u, u_evaluated=evaluated)
        reporter.write_solute_frame(step=step, time_ps=step * 0.002, exchange_index=written,
                                    configurations=configurations, solute_indices=[0])
        if (attempt + 1) % whole_every == 0:
            frame = reporter.write_frame(step=step, time_ps=step * 0.002,
                                         exchange_index=written)
            storage.ReplicaCheckpoint(tmp_path / "run_checkpoint.nc").write(
                step=step, exchange_index=written, frame_index=frame,
                solute_frame_index=written, configurations=configurations,
                state_to_walker=row, rng_states={"exchange": {}}, rule_state={},
                schedule=identity["schedule"], identity=identity)
    reporter.close()
    storage.write_run_state(path, "completed", identity=identity,
                            step=n_exchanges * exchange_steps,
                            total_steps=identity["schedule"]["total_steps"])
    return path


def test_a_complete_storage_validates(tmp_path):
    analysis = build_storage(tmp_path)
    result = validate.validate_replica_output(
        analysis=analysis, checkpoint=tmp_path / "run_checkpoint.nc")
    assert result.ok, result.problems
    assert result.facts["last_committed_exchange"] == 3


def test_a_truncated_analysis_file_is_rejected(tmp_path):
    analysis = build_storage(tmp_path)
    with open(analysis, "r+b") as handle:
        handle.truncate(1024)
    result = validate.validate_replica_output(analysis=analysis)
    assert not result.ok and any("NetCDF" in p for p in result.problems)


def test_a_truncated_checkpoint_is_rejected(tmp_path):
    analysis = build_storage(tmp_path)
    checkpoint = tmp_path / "run_checkpoint.nc"
    with open(checkpoint, "r+b") as handle:
        handle.truncate(512)
    result = validate.validate_replica_output(analysis=analysis, checkpoint=checkpoint)
    assert not result.ok


def test_a_mapping_row_that_is_not_a_permutation_is_rejected(tmp_path):
    analysis = build_storage(tmp_path, mapping=[[0, 1, 2], [1, 1, 2], [0, 1, 2], [0, 1, 2]])
    result = validate.validate_replica_output(analysis=analysis)
    assert not result.ok and any("permutation" in p for p in result.problems)


def test_a_missing_scheduled_exchange_is_rejected(tmp_path):
    # A dropped attempt in the MIDDLE: the remaining steps still increase, and the last one still
    # reaches the budget, so only a comparison against the schedule can see it.
    analysis = build_storage(tmp_path, n_exchanges=4, omit_exchange_at=(3,))
    result = validate.validate_replica_output(analysis=analysis)
    assert not result.ok
    assert any("not the scheduled ones" in problem for problem in result.problems), result.problems


def test_an_incomplete_budget_is_rejected(tmp_path):
    analysis = build_storage(tmp_path, n_exchanges=4,
                             identity=_identity(n_exchanges=10))
    storage.write_run_state(analysis, "completed", identity=_identity(n_exchanges=10),
                            step=2000, total_steps=5000)
    result = validate.validate_replica_output(analysis=analysis)
    assert not result.ok and any("short of the" in p for p in result.problems)


def test_a_manifest_may_not_name_a_path_outside_its_directory(tmp_path):
    analysis = build_storage(tmp_path)
    manifest = tmp_path / "restart.json"
    manifest.write_text(json.dumps({
        "run_status": "completed", "iterations_committed": 4,
        "storage": {"analysis_netcdf": "../elsewhere/run.nc"}}))
    result = validate.validate_replica_output(analysis=analysis, manifest=manifest)
    assert not result.ok and any("bare filename" in p for p in result.problems)


def test_a_manifest_from_a_different_run_is_rejected(tmp_path):
    analysis = build_storage(tmp_path)
    other = _identity()
    other["temperature_k"] = 310.0
    manifest = tmp_path / "restart.json"
    manifest.write_text(json.dumps({
        "run_status": "completed", "iterations_committed": 4,
        "storage": {"analysis_netcdf": "run.nc"}, "scientific_identity": other}))
    result = validate.validate_replica_output(analysis=analysis, manifest=manifest)
    assert not result.ok and any("disagrees with the storage" in p for p in result.problems)


def test_the_exchange_marker_is_written_after_the_row_it_describes():
    """An interrupted write must leave the counter on the previous, complete row."""
    source = (TEMPLATES / "replica_storage.py").read_text()
    block = source[source.index("def write_exchange"):source.index("def write_frame")]
    assert block.rindex("last_exchange") > block.index("u_evaluated")


def test_the_reporter_can_rewind_to_a_checkpoint(tmp_path):
    analysis = build_storage(tmp_path, n_exchanges=4)
    reporter = storage.ReplicaReporter(analysis, mode="a")
    try:
        assert reporter.last_exchange() == 3
        reporter.rewind(exchange=1, frame=0, solute_frame=1)
        assert reporter.last_exchange() == 1
        with pytest.raises(storage.StorageError, match="cannot rewind"):
            reporter.rewind(exchange=3, frame=1, solute_frame=3)
    finally:
        reporter.close()


def test_the_run_state_sidecar_sits_beside_the_storage():
    assert storage.run_state_path("/a/b/run.nc").name == "run.runstate.json"




def _ladder(protocol="REST2", platform=None, states=4, tau_max=0.5):
    """A ladder description of the shape `build-md` writes into a generated script."""
    return {"protocol": protocol, "solvent": "explicit", "n_states": states,
            "tau_max": tau_max, "exchange_interval_steps": 1000, "number_of_exchanges": 10,
            "state_trajectory": True, "rem_log": True, "neighbour_acceptance_report": True,
            "reservoir": {"enabled": False, "path": None, "velocities": "resample",
                          "refresh_interval_exchanges": 1},
            "dynamics": {"timestep_fs": 2.0, "temperature_K": 300.0, "pressure_bar": 1.0,
                         "friction_per_ps": 1.0, "barostat_interval_steps": 25,
                         "restraint_kcal_per_mol_A2": 1.0, "seed": 1, "platform": platform,
                         "tau": 0.0, "phase_space_printout": 0}}


def test_generated_replica_inputs_contain_no_concrete_path():
    """A generated protocol file must name no path, so the directory it sits in can be moved.

    Ported from the retired `emit.replica_protocol_file`: `build-md` writes the ladder, and
    `md_tools.runtime.replica` renders the protocol module the executor loads.
    """
    import ast as _ast

    from md_tools.runtime.replica import protocol_file_text

    for protocol in ("REST2", "rREST2"):
        text = protocol_file_text(_ladder(protocol=protocol))
        _ast.parse(text)
        body = text.split('"""')[2]
        for forbidden in ("/", "$MD_DATA", "..", "topology.pdb", "system.xml", "built.pdb"):
            assert forbidden not in body, f"{protocol} protocol body names a path: {forbidden}"
        assert "REST2Protocol(" in text
