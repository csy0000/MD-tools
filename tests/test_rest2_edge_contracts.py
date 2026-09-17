"""Edge contracts of the REST2 runtime: continuation mode and validation of extended runs.

rREST2 is archived (0.5.4). Its velocity-policy, reservoir-replacement and Maxwell-draw contracts
moved to archive/rREST2/tests/test_rest2_edge_contracts.py.

Each test names the gap it pins, and each fails against
`6cb693f748a2b63e5a0a49a9500ae295433fc2e0` -- the head this branch starts from.

PLATFORM_POLICY_EXEMPTION: declaration parsing, selection arithmetic, storage refusals and CLI
argument validation. Nothing here propagates dynamics; the runtime evidence stays in the
CUDA-marked files and in the smoke matrix recorded in
`docs/history/journals/20260831_rest2-rrest2-edge-contracts.md` and
`docs/history/journals/20260831_rest2-state-trajectories.md`.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd"
SRC = Path(__file__).resolve().parents[1] / "src"

openmm = pytest.importorskip("openmm")
sys.path.insert(0, str(SRC))

from md_tools.remd import executor as replica_executor
from md_tools.md import phase_space                                                 # noqa: E402
from md_tools.rest2 import identity as hamiltonian_identity                                        # noqa: E402
from md_tools.remd import storage as storage
from md_tools.remd import source_ensemble as source_ensemble
from md_tools.remd.protocol import REST2Protocol                         # noqa: E402


def test_ais_replacement_behaviour_is_untouched():
    """Replacement is AIS's own separately documented contract, and the shared selector still
    honours it. (The prohibition that sat beside this belonged to rREST2's reservoir, archived.)"""
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
    problems = replica_executor.validate(_Files(tmp_path, resume=True), _Arguments(), rank=0, groups=None)
    assert _refusal(problems), problems


def test_non_grouped_extend_is_refused(tmp_path):
    problems = replica_executor.validate(_Files(tmp_path, extend=5), _Arguments(), rank=0, groups=None)
    assert _refusal(problems), problems


def test_the_refusal_never_suggests_deleting_data(tmp_path):
    problems = replica_executor.validate(_Files(tmp_path, resume=True), _Arguments(), rank=0, groups=None)
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
    assert not _refusal(replica_executor.validate(files, arguments, rank=0, groups=groups))

    files = _Files(tmp_path, extend=20)
    files.trajectory = str(trajectory)
    assert not _refusal(replica_executor.validate(files, arguments, rank=0, groups=groups))


def test_validation_happens_before_any_output_is_touched(tmp_path):
    """The refusal must be side-effect free: no directory created, no report opened, no protocol
    imported. `validate()` is a pure function over the parsed arguments, and `main()` returns on
    its problems before it makes a single directory."""
    outputs = tmp_path / "deep" / "nested"
    files = _Files(tmp_path, resume=True)
    files.output = str(outputs / "run.out")
    files.trajectory = str(outputs / "run.dcd")

    assert _refusal(replica_executor.validate(files, _Arguments(), rank=0, groups=None))
    assert not outputs.exists(), "validation created an output directory"

    source = (TEMPLATES / "executor.py").read_text(encoding="utf-8")
    refusal_index = source.index("def validate(")
    mkdir_index = source.index("Path(value).parent.mkdir")
    run_index = source.index("status = run_grouped(")
    assert refusal_index < mkdir_index < run_index


def test_an_ungrouped_launch_is_refused():
    """There is ONE route through the executor, and it carries a ladder plan.

    Without a group file the executor imported `files.input` and called `.run(files)` on it: no
    plan, no preflight, no platform resolution. A second contract for the same runtime, reachable
    by calling this module directly, and the one nothing validated.
    """
    files = _Files(Path("/nonexistent"))
    problems = replica_executor.validate(files, _Arguments(), rank=0, groups=None)
    assert any("--groupfile" in problem for problem in problems), problems
    assert not hasattr(replica_executor, "load_protocol"), (
        "the ungrouped protocol loader is still reachable")


def test_an_existing_output_is_untouched_by_a_refused_continuation(tmp_path):
    """The bytes a refused run must not disturb."""
    sentinel = tmp_path / "run.dcd"
    sentinel.write_bytes(b"authoritative bytes")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    files = _Files(tmp_path, resume=True)
    files.trajectory = str(sentinel)
    assert _refusal(replica_executor.validate(files, _Arguments(), rank=0, groups=None))
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


# --- the co-generated source stage is not append-safe cMD -------------------------------------


def test_continuing_the_source_stage_is_refused_like_any_other_cmd(tmp_path):
    """Even though it lives inside an rREST2 project, the source stage is non-grouped, so the
    mode boundary applies to it unchanged."""
    files = _Files(tmp_path, resume=True)
    files.trajectory = str(tmp_path / "cMD_tau0p5" / "cmd.dcd")
    problems = replica_executor.validate(files, _Arguments(), rank=0, groups=None)
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
    from md_tools.remd import validate as replica_validate

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
    source = (TEMPLATES / "validate.py").read_text(encoding="utf-8")
    block = source[source.index('expected = identity.get("number_of_exchanges")'):
                   source.index("except Exception as failure:", source.index(
                       'expected = identity.get("number_of_exchanges")'))]
    assert "schedule" in block, "the actual budget is never consulted"
    assert 'stats["exchanges_committed"] != int(\n                expected)' not in block, (
        "the original request is still compared for equality")
    assert "< int(" in block, "the identity count is not treated as a floor"


def test_the_generated_cmd_launcher_passes_no_continuation_flag(tmp_path):
    """A fixed-tau cMD stage is an ordinary run and must never be launched as continuable.

    Ported from the retired `emit.launcher`. `build-md` now generates the stage scripts and
    run.sh, so the property is asserted against what it actually writes.
    """
    from md_tools.build.md import build_scripts

    from .conftest import make_dataset_root, make_scaled_state

    make_dataset_root(tmp_path)
    make_scaled_state(tmp_path, tau=0.5)            # a hot stage runs on its saved state
    config = tmp_path / "hot.config"
    config.write_text("protocol: cMD\nsolvent: implicit\n"
                      "dynamics:\n  tau: 0.5\n"
                      "stages:\n  production_steps: 1000\n", encoding="utf-8")
    out = tmp_path / "cMD-run1"
    build_scripts(config_path=config, out_dir=out, echo=False)

    run_sh = (out / "run.sh").read_text()
    assert "--resume" not in run_sh
    assert "--extend" not in run_sh
    cmd = (out / "cMD.py").read_text()
    assert "--resume" not in cmd and "--extend" not in cmd
    # The tau reaches the stage through `resolved.config`, which is the one declaration; the
    # script names the stage and nothing else. What matters is that the value survives the round
    # trip to every stage that claims it -- all but minimisation, which scales nothing and claims
    # nothing because `min/` is shared by every method (step 3).
    from md_tools.build.md import resolve_md_config, stage_plan

    plan = stage_plan(resolve_md_config(out / "resolved.config"))
    assert plan[0]["name"] == "min" and float(plan[0]["tau"]) == 0.0
    assert all(stage["tau"] == 0.5 for stage in plan[1:]), \
        "the fixed tau must reach every stage that claims it"
