"""The replica executor's file interface: flags, precedence, refusals, and what it must never do.

This is `md_tools/remd/executor.py`, reached as a FUNCTION from `md_tools.remd.generated`.
It was once installed as a separate `openmm-md` executable and the file was named after it; both
are retired, and the package installs exactly one executable, `md-openmm`.

The runner is generic on purpose. It resolves paths, refuses to destroy results, runs one ladder
with its output captured, and reports. There is ONE route through it, and it carries a ladder
plan: without a group file the launch is refused, because the ungrouped route it used to take
imported an arbitrary protocol and called `run(files)` on it with no plan and no preflight. It decides nothing scientific, and these tests are largely
about proving the negative: that it creates no Context while validating, writes no completion
marker of its own, and invents no path.

Most run without OpenMM doing any work -- validation happens before a Context exists, which is what
makes it testable in milliseconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "src" / "md_tools" / "remd" / "executor.py"
#: The checkout's `src`, for subprocesses. NOT the package directory: putting that on PYTHONPATH
#: would let `remd/statistics.py` shadow the standard library.
SRC = Path(__file__).resolve().parents[1] / "src"

pytest.importorskip("openmm")

from md_tools.remd import executor as replica_executor  # noqa: E402

TRIVIAL = textwrap.dedent('''
    from pathlib import Path

    def run(files):
        print("stage: trivial")
        Path(files.restart).write_text("<state/>")
        if files.checkpoint:
            Path(files.checkpoint).write_bytes(b"chk")
        if files.trajectory:
            Path(files.trajectory).write_bytes(b"dcd")
        if files.solute_x:
            Path(files.solute_x).write_bytes(b"dcd")
        print("run_status: completed")
''')


@pytest.fixture
def workspace(tmp_path):
    """A protocol and the four inputs it needs. Contents are irrelevant to the interface."""
    (tmp_path / "protocol.py").write_text(TRIVIAL)
    for name in ("topology.pdb", "system.xml", "start.xml"):
        (tmp_path / name).write_text("x")
    return tmp_path


def _run(workspace, *args, env=None):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(env or {})
    return subprocess.run([sys.executable, "-m", "md_tools.remd.executor", *args], capture_output=True, text=True,
                          cwd=str(workspace), env=environment, timeout=300)


def _complete(workspace, **overrides):
    arguments = {"-i": "protocol.py", "-p": "topology.pdb", "-s": "system.xml",
                 "-c": "start.xml", "-o": "stage.out", "-r": "stage.state.xml"}
    arguments.update(overrides)
    return [item for pair in arguments.items() for item in pair if item]


# --- option parsing and precedence ---------------------------------------------

@pytest.mark.parametrize("missing", ["-i", "-p", "-s", "-c", "-o", "-r"])
def test_a_missing_required_path_fails_before_anything_is_created(workspace, missing):
    arguments = _complete(workspace)
    index = arguments.index(missing)
    del arguments[index:index + 2]
    result = _run(workspace, *arguments)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (workspace / "stage.out").exists(), "an .out was created despite invalid arguments"
    assert not (workspace / "stage.state.xml").exists()


def test_an_input_that_does_not_exist_is_named(workspace):
    result = _run(workspace, *_complete(workspace, **{"-p": "absent.pdb"}))
    assert result.returncode == 2
    assert "absent.pdb" in result.stderr and "does not exist" in result.stderr
    assert not (workspace / "stage.out").exists()


# --- refusals ------------------------------------------------------------------

def test_an_existing_output_is_refused_and_left_untouched(workspace):
    (workspace / "stage.out").write_text("previous result")
    result = _run(workspace, *_complete(workspace))
    assert result.returncode == 2
    assert "already exist" in result.stderr
    assert (workspace / "stage.out").read_text() == "previous result", "the old .out was clobbered"


def test_force_replaces_an_existing_output_deliberately(workspace):
    (workspace / "stage.out").write_text("previous result")
    result = _run(workspace, *_complete(workspace), "--force")
    assert result.returncode == 0, result.stderr
    assert "previous result" not in (workspace / "stage.out").read_text()


def test_an_output_that_is_also_an_input_is_refused(workspace):
    """Writing the trajectory over the topology would destroy the thing being read."""
    result = _run(workspace, *_complete(workspace, **{"-x": "topology.pdb"}))
    assert result.returncode == 2
    assert "same file" in result.stderr
    assert (workspace / "topology.pdb").read_text() == "x"


def test_two_outputs_naming_the_same_file_are_refused(workspace):
    result = _run(workspace, *_complete(workspace, **{"-x": "same.bin", "--checkpoint": "same.bin"}))
    assert result.returncode == 2 and "same file" in result.stderr


# --- what it does with the protocol ------------------------------------------
#
# DIRECT GROUPED ENTRY. These once ran a one-line `protocol.py` through an ungrouped route that
# took no ladder plan, ran no preflight and resolved no platform -- a second contract for the same
# runtime, reachable only by calling this module directly, and the one nothing validated. It is
# gone, and the behaviour it was carrying (the report capture, the completion gate, the promised-
# output check) belongs to `main` and is reached the same way by the only route there is.
#
# So these run a real two-state ladder on the CPU, directly through `python -m
# md_tools.remd.executor`, as §8 asks: direct-entry tests, not `md-run` standing in for them.

def test_output_directories_are_created_only_after_validation(workspace):
    """A nested output directory must not appear when the arguments were never valid."""
    result = _run(workspace, "-i", "protocol.py", "-p", "topology.pdb", "-s", "system.xml",
                  "-c", "start.xml", "-o", "deep/nested/stage.out")
    assert result.returncode == 2                      # -r missing
    assert not (workspace / "deep").exists(), "a directory was created for an invalid invocation"


def test_the_runner_imports_nothing_beyond_the_standard_library_and_openmm():
    """Checked on the parsed imports, not on the text.

    A prose mention of `md_tools` in the docstring -- saying it deliberately does not use it --
    is not a dependency, and a test that cannot tell the difference would force the explanation
    out of the file.
    """
    import ast

    tree = ast.parse(RUNNER.read_text())
    module_level, deferred = set(), set()
    for node in ast.walk(tree):
        names = set()
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = {node.module.split(".")[0]}
        if not names:
            continue
        # An import inside a function body is paid for only when that path runs.
        inside_function = any(
            isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            for parent in ast.walk(tree)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node in ast.walk(parent))
        (deferred if inside_function else module_level).update(names)

    allowed = {"argparse", "contextlib", "importlib", "json", "os", "shlex", "sys", "traceback",
               "pathlib", "types", "__future__", "openmm"}
    assert module_level <= allowed, (
        f"the replica executor imports beyond the standard library and OpenMM at module level: "
        f"{module_level - allowed}. A single generated stage must still run once md_tools is "
        f"gone, so anything else has to be deferred into the path that needs it.")
    assert "md_tools" not in module_level | deferred
    # Grouped mode legitimately imports the runtime modules copied beside the protocol -- but only
    # inside the grouped path, so a single-stage run never touches them.
    assert "yaml" not in module_level, "yaml is a grouped-mode dependency and must stay deferred"




def test_the_ungrouped_route_is_gone(workspace):
    """Reaching the runtime, not argparse: the arguments below are complete and well-formed.

    They name an existing protocol, topology, system and coordinates, and every output path is
    free. The old route would have imported `protocol.py` and called `run(files)` on it. It is
    refused now because it carries no group file, and therefore no ladder plan.
    """
    result = _run(workspace, *_complete(workspace))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--groupfile" in result.stderr, result.stderr
    assert not (workspace / "stage.out").exists(), (
        "the refused ungrouped launch created an .out")


@pytest.fixture
def ladder(prepared, tmp_path):
    """A private copy of the real two-state ladder, so a run may mutate it."""
    import shutil

    source, _atoms, _unit = prepared
    work = tmp_path / "ladder"
    shutil.copytree(source, work)
    for stale in ("run.out", "run.out.rank01", "exchange.nc", "restart.json",
                  "checkpoint.nc", "rem.log"):
        (work / stale).unlink(missing_ok=True)
    return work


def _grouped(work, *extra, env=None):
    environment = dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1")
    environment.pop("MD_TOOLS_FORCE_NO_CUDA", None)
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", "2",
         "-x", "exchange.nc", "-r", "restart.json", "--checkpoint", "checkpoint.nc",
         "-o", "run.out", *extra],
        cwd=str(work), capture_output=True, text=True, timeout=1800, env=environment)


@pytest.mark.slow
def test_a_direct_grouped_entry_runs_and_writes_every_promised_output(ladder):
    """The whole point of the flag contract: what was promised is what appears."""
    result = _grouped(ladder, "--rem", "rem.log")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for name in ("exchange.nc", "restart.json", "checkpoint.nc", "rem.log"):
        assert (ladder / name).exists(), f"{name} was promised and not written"
    assert "run_status: completed" in (ladder / "run.out").read_text(encoding="utf-8")


@pytest.mark.slow
def test_the_environment_supplies_a_path_when_the_flag_is_absent(ladder):
    """`OPENMM_*` exists for running a ladder by hand, not for hiding generated wiring."""
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.remd.executor",
         "--groupfile", "ladder.group", "-ng", "2", "-x", "exchange.nc",
         "-r", "restart.json", "--checkpoint", "checkpoint.nc"],
        cwd=str(ladder), capture_output=True, text=True, timeout=1800,
        env=dict(os.environ, PYTHONPATH=str(SRC), OPENMM_CPU_THREADS="1",
                 OPENMM_OUTPUT="from_environment.out"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert (ladder / "from_environment.out").exists(), sorted(
        path.name for path in ladder.iterdir())


@pytest.mark.slow
def test_the_flag_wins_over_the_environment(ladder):
    result = _grouped(ladder, env={"OPENMM_OUTPUT": str(ladder / "from_environment.out")})
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert (ladder / "run.out").exists()
    assert not (ladder / "from_environment.out").exists()


@pytest.mark.slow
def test_the_protocols_output_lands_in_the_rank_report_not_the_terminal(ladder):
    """`-o` is where a run's output goes. A launcher's terminal is not a record."""
    result = _grouped(ladder)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    report = (ladder / "run.out").read_text(encoding="utf-8")
    assert "grouped mode" in report, report[:2000]
    assert "grouped mode" not in result.stdout


@pytest.mark.slow
def test_force_replaces_an_existing_output_deliberately(ladder):
    """The refusal above is the default; this is the deliberate override, and it must run."""
    (ladder / "exchange.nc").write_bytes(b"previous result")
    refused = _grouped(ladder)
    assert refused.returncode == 2, refused.stderr
    assert (ladder / "exchange.nc").read_bytes() == b"previous result"

    result = _grouped(ladder, "--force")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert (ladder / "exchange.nc").read_bytes() != b"previous result"


@pytest.mark.slow
def test_a_failing_protocol_returns_nonzero_and_keeps_its_report(ladder):
    """The report must survive a failure, or there is nothing to diagnose it with."""
    text = (ladder / "protocol.py").read_text(encoding="utf-8")
    (ladder / "protocol.py").write_text(
        text + '\nraise RuntimeError("the physics disagreed")\n', encoding="utf-8")
    result = _grouped(ladder)
    assert result.returncode != 0
    report = (ladder / "run.out").read_text(encoding="utf-8")
    assert "the physics disagreed" in report, "the traceback must be in the report"
    assert "run_status: completed" not in report


#: Appended to the real protocol module, which is imported inside the run before anything starts.
#: The ladder below is a REAL two-state run through the real driver -- only the record it returns,
#: or the file it leaves behind, is disturbed, which is exactly the shape these two checks in
#: `main` exist to catch: a run that believed it finished.
DISTURB = """

import md_tools.remd.driver as _driver

_original_run = _driver.ReplicaRun.run


def _disturbed(self, *arguments, **keywords):
    record = _original_run(self, *arguments, **keywords)
    {disturbance}
    return record


_driver.ReplicaRun.run = _disturbed
"""


def _disturb(ladder, disturbance):
    text = (ladder / "protocol.py").read_text(encoding="utf-8")
    (ladder / "protocol.py").write_text(
        text + DISTURB.format(disturbance=disturbance), encoding="utf-8")


@pytest.mark.slow
def test_a_marker_over_missing_outputs_is_not_accepted(ladder):
    """The marker means the outputs exist. Printing it without them is a false report.

    The ladder runs for real and completes; the checkpoint it promised is then gone by the time
    `main` looks. That is the shape of the failure: the run believed it finished, and one of the
    files it named is not there.
    """
    _disturb(ladder, 'from pathlib import Path as _P; '
                     '_P(self.files.checkpoint).unlink(missing_ok=True)')
    result = _grouped(ladder)
    assert result.returncode != 0, "a completion marker over a missing output was accepted"
    assert "did not write" in result.stderr, result.stderr
    assert "checkpoint.nc" in result.stderr, result.stderr


@pytest.mark.slow
def test_a_run_that_finishes_silently_has_not_completed(ladder):
    """The runner must not write the marker on the protocol's behalf.

    A real run, returning without claiming completion. Every promised output is on disk, so only
    the missing claim can fail it -- and it must.
    """
    _disturb(ladder, 'record = dict(record, run_status="ran, but said nothing")')
    result = _grouped(ladder)
    assert result.returncode != 0
    assert "no 'run_status: completed' line" in result.stderr, result.stderr
    for name in ("exchange.nc", "restart.json", "checkpoint.nc"):
        assert (ladder / name).exists(), (
            f"{name} is missing, so this failed for the wrong reason")
    assert "run_status: completed" not in (
        ladder / "run.out").read_text(encoding="utf-8")


# --- the completion marker -----------------------------------------------------
#
# The marker is a claim that the requested outputs exist. A substring test would accept a protocol
# that merely mentioned it, and report a failed stage as finished.
#
# These test `_has_completion_line` on the report text DIRECTLY. That is honest about what they
# are: the decision is a line-matching rule, and the run above proves the rule actually gates the
# exit code. Driving twelve real two-state ladders to vary one printed line would be twelve
# minutes of CPU to re-measure one `str.strip`.

def _report(tmp_path, line: str) -> Path:
    path = tmp_path / "run.out"
    path.write_text(f"# grouped mode: 2 groups\n{line}\n", encoding="utf-8")
    return path


def test_the_exact_line_counts_as_completion(tmp_path):
    assert replica_executor._has_completion_line(_report(tmp_path, "run_status: completed"))


def test_a_missing_marker_is_not_completion(tmp_path):
    assert not replica_executor._has_completion_line(_report(tmp_path, "finished, I think"))


@pytest.mark.parametrize("misleading", [
    'checking whether run_status: completed applies here',
    'run_status: completed_with_warnings',
    'not run_status: completed',
    '# run_status: completed',
    'previous run_status: completed (from an earlier stage)',
])
def test_a_line_merely_containing_the_marker_is_not_completion(tmp_path, misleading):
    """Each of these contains the marker as a substring and claims nothing."""
    assert not replica_executor._has_completion_line(_report(tmp_path, misleading)), misleading


@pytest.mark.parametrize("padded", ["  run_status: completed", "run_status: completed   ",
                                    "\trun_status: completed"])
def test_surrounding_whitespace_is_tolerated(tmp_path, padded):
    """A reporter that indents its final line has still made the claim."""
    assert replica_executor._has_completion_line(_report(tmp_path, padded)), padded
