"""The `openmm-md` file interface: flags, precedence, refusals, and what it must never do.

The runner is generic on purpose. It resolves paths, refuses to destroy results, runs one protocol
with its output captured, and reports. It decides nothing scientific, and these tests are largely
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

RUNNER = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates" / "openmm_md.py"

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
    return subprocess.run([sys.executable, str(RUNNER), *args], capture_output=True, text=True,
                          cwd=str(workspace), env=environment, timeout=300)


def _complete(workspace, **overrides):
    arguments = {"-i": "protocol.py", "-p": "topology.pdb", "-s": "system.xml",
                 "-c": "start.xml", "-o": "stage.out", "-r": "stage.state.xml"}
    arguments.update(overrides)
    return [item for pair in arguments.items() for item in pair if item]


# --- option parsing and precedence ---------------------------------------------

def test_every_documented_flag_is_accepted(workspace):
    result = _run(workspace, *_complete(workspace, **{"-x": "t.dcd", "--checkpoint": "s.chk",
                                                      "--solute-x": "solute.dcd"}))
    assert result.returncode == 0, result.stderr
    for name in ("stage.out", "stage.state.xml", "t.dcd", "s.chk"):
        assert (workspace / name).exists(), name


def test_environment_supplies_a_path_when_the_flag_is_absent(workspace):
    """`OPENMM_*` exists for running a protocol by hand, not for hiding generated wiring."""
    result = _run(workspace, "-i", "protocol.py", "-p", "topology.pdb", "-s", "system.xml",
                  env={"OPENMM_COORDINATES": "start.xml", "OPENMM_OUTPUT": "e.out",
                       "OPENMM_RESTART": "e.state.xml"})
    assert result.returncode == 0, result.stderr
    assert (workspace / "e.out").exists()


def test_the_flag_wins_over_the_environment(workspace):
    result = _run(workspace, *_complete(workspace, **{"-o": "flag.out"}),
                  env={"OPENMM_OUTPUT": str(workspace / "environment.out")})
    assert result.returncode == 0, result.stderr
    assert (workspace / "flag.out").exists()
    assert not (workspace / "environment.out").exists()


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


# --- what it does with the protocol --------------------------------------------

def test_stdout_and_stderr_both_land_in_the_out(workspace):
    (workspace / "protocol.py").write_text(textwrap.dedent('''
        import sys
        from pathlib import Path

        def run(files):
            print("this is stdout")
            print("this is stderr", file=sys.stderr)
            Path(files.restart).write_text("<state/>")
            print("run_status: completed")
    '''))
    result = _run(workspace, *_complete(workspace))
    assert result.returncode == 0, result.stderr
    report = (workspace / "stage.out").read_text()
    assert "this is stdout" in report and "this is stderr" in report


def test_a_failing_protocol_returns_nonzero_and_keeps_its_out(workspace):
    (workspace / "protocol.py").write_text(textwrap.dedent('''
        def run(files):
            print("got this far")
            raise RuntimeError("the physics disagreed")
    '''))
    result = _run(workspace, *_complete(workspace))
    assert result.returncode != 0
    report = (workspace / "stage.out").read_text()
    assert "got this far" in report, "the .out must survive for diagnosis"
    assert "the physics disagreed" in report, "the traceback must be in the .out"
    assert "run_status: completed" not in report


def test_the_runner_never_writes_the_completion_marker_itself(workspace):
    """A protocol that finishes silently has not completed, and the runner must not pretend."""
    (workspace / "protocol.py").write_text(textwrap.dedent('''
        from pathlib import Path

        def run(files):
            Path(files.restart).write_text("<state/>")
            print("done quietly")
    '''))
    result = _run(workspace, *_complete(workspace))
    assert result.returncode != 0
    assert "run_status: completed" not in (workspace / "stage.out").read_text()


def test_a_marker_over_missing_outputs_is_not_accepted(workspace):
    """The marker means the outputs exist. Printing it without them is a false report."""
    (workspace / "protocol.py").write_text(textwrap.dedent('''
        def run(files):
            print("run_status: completed")
    '''))
    result = _run(workspace, *_complete(workspace, **{"-x": "t.dcd"}))
    assert result.returncode != 0
    assert "did not write" in result.stderr


def test_a_protocol_without_run_is_reported_clearly(workspace):
    (workspace / "protocol.py").write_text("value = 1\n")
    result = _run(workspace, *_complete(workspace))
    assert result.returncode != 0
    assert "run(files)" in (workspace / "stage.out").read_text() + result.stderr


def test_output_directories_are_created_only_after_validation(workspace):
    """A nested output directory must not appear when the arguments were never valid."""
    result = _run(workspace, "-i", "protocol.py", "-p", "topology.pdb", "-s", "system.xml",
                  "-c", "start.xml", "-o", "deep/nested/stage.out")
    assert result.returncode == 2                      # -r missing
    assert not (workspace / "deep").exists(), "a directory was created for an invalid invocation"


def test_the_runner_imports_nothing_beyond_the_standard_library_and_openmm():
    """Checked on the parsed imports, not on the text.

    A prose mention of `md_templates` in the docstring -- saying it deliberately does not use it --
    is not a dependency, and a test that cannot tell the difference would force the explanation
    out of the file.
    """
    import ast

    tree = ast.parse(RUNNER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    allowed = {"argparse", "contextlib", "importlib", "os", "sys", "traceback", "pathlib",
               "types", "__future__", "openmm"}
    assert imported <= allowed, f"openmm-md imports beyond the allowed set: {imported - allowed}"
    assert "md_templates" not in imported and "yaml" not in imported
