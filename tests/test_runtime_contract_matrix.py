"""The protocol x flag matrix: every accepted flag does its job, or is refused before any output.

A flag that a protocol parses and then ignores is worse than one it rejects. The run completes,
the provenance record shows the flag was given, and nothing anywhere did what it says -- so the
person who typed it believes it took effect, and the only way to find out otherwise is to notice
that a result is wrong.

These go through the GENERATED SCRIPTS, as subprocesses, for the reason
`test_direct_runtime_preflight.py` states: `md-run` is one entry point and the generated runtimes
are four more, and a refusal that lives only in `md-run` leaves those four open. Where a refusal
is claimed for both layers, both layers are tested.

Every case asserts the specific diagnostic AND that the destination is untouched, and `_refused`
rejects an argparse complaint: a test satisfied by "unrecognized arguments" proves nothing about
the validation it is named for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `tests` is a package, so the harness is imported relatively. Shared rather than copied: two
# copies of `_refused` would drift, and the copy that stopped rejecting argparse errors would be
# the one whose tests still passed.
from .test_direct_runtime_preflight import (ENTRY, MODES, PROTOCOL_ONLY,  # noqa: F401
                                            VALID_USER, _config, _launch, _refused, _run,
                                            _snapshot, workspace)

CONFIG = "MD_TOOLS_CONFIG"


@pytest.fixture
def good_config(tmp_path):
    """A complete, valid user configuration, so a refusal is never about this file."""
    return {CONFIG: str(_config(tmp_path, "user.config",
                                {"schema_version": "1.0", "user": VALID_USER}))}


def _untouched(destination: Path, before):
    assert _snapshot(destination) == before, (
        f"the refusal wrote to {destination}: "
        f"{sorted((_snapshot(destination) or {}).keys())}")


# --- flags that belong to another protocol -------------------------------------------------------

#: mode -> (flag and value, the fragment its refusal must contain)
MISAPPLIED = {
    "split": [(["-ng", "4"], "-ng"), (["-groupfile", "g.txt"], "-groupfile"),
              (["-source-traj", "../source.dcd"], "-source-traj")],
    "allinone": [(["-ng", "4"], "-ng"), (["-groupfile", "g.txt"], "-groupfile"),
                 (["-source-traj", "../source.dcd"], "-source-traj")],
    "AIS": [(["-x", "one.nc"], "-x"), (["-c", "prev.xml"], "-c"),
            (["-r", "out.xml"], "-r"), (["-chk", "a.chk"], "-chk"),
            (["-groupfile", "g.txt"], "-groupfile")],
}


@pytest.mark.parametrize("mode", sorted(MISAPPLIED))
def test_a_flag_from_another_protocol_is_refused_by_name_before_any_output(
        mode, workspace, tmp_path, good_config):
    """Not "unrecognized arguments": the flag is accepted by the parser and refused by the rule.

    Leaving these off the parser would also refuse them, but argparse's refusal names the flag and
    explains nothing -- and, worse, it puts the rule in argparse rather than in the shared
    preflight, where `md-run` and the generated script can then disagree about it. They did: only
    `md-run` ever checked AIS `-x`.
    """
    for flags, fragment in MISAPPLIED[mode]:
        destination = tmp_path / f"out-{mode}-{fragment.strip('-')}"
        before = _snapshot(destination)
        done = _launch(workspace, mode, destination, *flags, environment=good_config)
        _refused(done, fragment=fragment)
        _untouched(destination, before)


# --- a continuation that is not there ------------------------------------------------------------

@pytest.mark.parametrize("mode", ["split", "REST2", "rREST2"])
def test_an_explicitly_named_missing_continuation_is_refused(mode, workspace, tmp_path,
                                                             good_config):
    """`-c` naming a file that does not exist used to be silently dropped.

    The old rule was `if Path(c).exists()`, which reads as leniency for the all-in-one `--check`
    chain and is complete leniency for a typo. A real run given `-c eq_npt_fre.xml` found nothing
    to check, started from the coordinates in `-p`, and finished reporting success -- having
    continued nothing.
    """
    destination = tmp_path / f"missing-c-{mode}"
    before = _snapshot(destination)
    done = _launch(workspace, mode, destination, "-c", "no_such_restart.xml",
                   environment=good_config)
    _refused(done, fragment="no_such_restart.xml")
    _untouched(destination, before)


def test_the_all_in_one_check_still_allows_the_parent_a_later_stage_will_write(
        workspace, tmp_path, good_config):
    """The ONE legitimate missing continuation, and it must keep working.

    Under `--check` nothing has run, so every stage after the first is missing its parent by
    construction. That case is now stated by the chain -- it names the stage that produces the
    file -- rather than inferred from the file's absence, which is what made the rule above
    impossible to enforce.
    """
    destination = tmp_path / "check-chain"
    before = _snapshot(destination)
    done = _launch(workspace, "allinone", destination, "--check", *PROTOCOL_ONLY,
                   environment=good_config)
    assert done.returncode == 0, done.stdout + done.stderr
    # `--check` on the whole chain must also write nothing at all.
    _untouched(destination, before)


def test_a_first_stage_continuation_must_exist_even_under_check(workspace, tmp_path,
                                                                good_config):
    """`-c` for the FIRST stage names a file from outside this chain. Nothing here will write it."""
    destination = tmp_path / "check-first-c"
    before = _snapshot(destination)
    done = _launch(workspace, "allinone", destination, "--check", "-c", "not_here.xml",
                   *PROTOCOL_ONLY, environment=good_config)
    _refused(done, fragment="not_here.xml")
    _untouched(destination, before)


# --- --check is read-only ------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_check_writes_nothing_at_all(mode, workspace, tmp_path, good_config):
    """A `--check` that creates its output directory has already broken the contract it validates.

    An existing `-odir` holding a log or a `resolved.config` is indistinguishable from a run that
    started, and a person who ran `--check` to find out whether a run WOULD work now has a
    directory that says one did.
    """
    destination = tmp_path / f"check-{mode}"
    before = _snapshot(destination)
    # `--cpu`, because the subject is what `--check` WRITES, not which accelerator it would have
    # used. Without it this asserted `returncode == 0` on a machine with no GPU and failed for a
    # reason that has nothing to do with the contract under test.
    done = _launch(workspace, mode, destination, "--check", *PROTOCOL_ONLY,
                   environment=good_config)
    assert done.returncode == 0, f"{mode}: --check failed:\n{done.stdout}{done.stderr}"
    _untouched(destination, before)


# --- abbreviation is off everywhere --------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_no_generated_parser_resolves_an_abbreviation(mode, workspace, tmp_path, good_config):
    """`--dev 3` must not silently become `--device 3`, and `--cp` must not become `--cpu`.

    argparse resolves unambiguous prefixes by default. A misspelling it resolves RUNS, which for
    a placement or platform flag means the run happens somewhere nobody asked for.
    """
    destination = tmp_path / f"abbrev-{mode}"
    before = _snapshot(destination)
    done = _launch(workspace, mode, destination, "--dev", "0", environment=good_config)
    assert done.returncode != 0, "an abbreviation was resolved"
    assert "unrecognized arguments" in (done.stdout + done.stderr), (
        f"{mode}: refused, but not as an unknown flag:\n{done.stdout}{done.stderr}")
    _untouched(destination, before)


# --- the complete output inventory ----------------------------------------------------------------

def test_the_inventory_names_every_artefact_a_ladder_writes():
    """`--overwrite` governed `resolved.config` alone; everything else was replaced silently.

    The per-state trajectories are the scientific result and were in no inventory at all.
    """
    from md_tools.run.preflight import _ladder_inventory

    inventory = _ladder_inventory(protocol="REST2", replicas=4, output="run/REST2.out",
                                  log="run/REST2.log", trajectory="run/REST2.nc",
                                  restart="run/restart.json", checkpoint="run/REST2_chk.nc",
                                  groupfile=None)
    for role in ("out", "log", "trajectory", "restart", "checkpoint", "solute",
                 "protocol_helper", "group_file", "rem_log"):
        assert role in inventory.roles, role
    for state in range(4):
        assert f"state_trajectory_{state}" in inventory.roles


def test_a_named_group_file_is_an_input_and_not_part_of_the_inventory():
    """It is read, not written. Listing it as an output made it collide with itself."""
    from md_tools.run.preflight import _ladder_inventory

    inventory = _ladder_inventory(protocol="REST2", replicas=2, output="run/o", log="run/l",
                                  trajectory=None, restart=None, checkpoint=None,
                                  groupfile="ladder.group")
    assert "group_file" not in inventory.roles


def test_the_inventory_names_every_artefact_an_ais_run_writes():
    from md_tools.run.preflight import _ais_inventory

    inventory = _ais_inventory(output="run/AIS.out", log="run/AIS.log", paths=3)
    for role in ("out", "log", "run_identity", "work_table", "work_summary", "selected_frames"):
        assert role in inventory.roles, role
    for index in range(3):
        assert f"path_{index:04d}" in inventory.roles
        assert f"trajectory_{index:04d}" in inventory.roles


def test_a_stage_inventory_includes_the_outputs_it_derives_rather_than_is_given():
    """The phase-space stream and the checkpoint tree arrive as no flag and are still written."""
    from md_tools.run.preflight import _stage_inventory

    inventory = _stage_inventory(output="d/min.out", log="d/min.log", trajectory="d/min.dcd",
                                 restart="d/min.xml", checkpoint="d/min.chk")
    assert inventory.roles["phase_space"].name == "min.phase_space.nc"
    assert inventory.roles["checkpoints"].name == "min.checkpoints"
    # The checkpoint tree is what `--resume` reads, so its presence is never itself the objection.
    assert "checkpoints" in inventory.resumable


def test_an_existing_output_is_refused_unless_overwrite_or_resume_says_otherwise(tmp_path):
    from md_tools.run.preflight import (OutputInventory, PreflightError,
                                        check_existing_outputs)

    existing = tmp_path / "min.dcd"
    existing.write_bytes(b"frames")
    inventory = OutputInventory(roles={"trajectory": existing, "log": tmp_path / "absent.log"})

    with pytest.raises(PreflightError) as refusal:
        check_existing_outputs(inventory, where="stage min")
    assert "min.dcd" in str(refusal.value) and "trajectory" in str(refusal.value)

    check_existing_outputs(inventory, overwrite=True)      # explicitly asked for
    check_existing_outputs(inventory, resume=True)         # continuing that run


def test_a_run_refuses_to_write_over_an_existing_one_and_says_which_files(workspace, tmp_path,
                                                                          good_config):
    """End to end, through `md-run`: the second invocation must not half-overwrite the first."""
    destination = tmp_path / "twice"
    first = _run([sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
                  "-i", "min.in", "-p", "../built.pdb", "-s", "../built.xml",
                  "-odir", str(destination), *PROTOCOL_ONLY],
                 cwd=workspace / "split", environment=good_config)
    assert first.returncode == 0, first.stdout + first.stderr

    # Rerunning a stage that COMPLETED under this exact configuration is the idempotent case and
    # stays a success: it is how a chain is safely re-driven.
    second = _run([sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
                   "-i", "min.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-odir", str(destination), *PROTOCOL_ONLY],
                  cwd=workspace / "split", environment=good_config)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already completed" in second.stdout + second.stderr

    # Remove the log and the outputs are suddenly a previous run nobody claimed. THAT is the case
    # `--overwrite` exists for, and it must be asked for rather than assumed.
    for log in destination.glob("*.log"):
        log.unlink()
    third = _run([sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
                  "-i", "min.in", "-p", "../built.pdb", "-s", "../built.xml",
                  "-odir", str(destination), *PROTOCOL_ONLY],
                 cwd=workspace / "split", environment=good_config)
    assert third.returncode != 0, third.stdout + third.stderr
    message = third.stdout + third.stderr
    assert "already exist" in message and "--overwrite" in message, message

    fourth = _run([sys.executable, "-m", "md_tools.cli.md_openmm", "md-run",
                   "-i", "min.in", "-p", "../built.pdb", "-s", "../built.xml",
                   "-odir", str(destination), "--overwrite", *PROTOCOL_ONLY],
                  cwd=workspace / "split", environment=good_config)
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr


# --- ladder helpers are content-addressed and written once ------------------------------------

def test_a_stale_protocol_helper_is_refused_rather_than_reused_or_replaced(tmp_path):
    from md_tools.remd.generated import HELPER_FINGERPRINT, _write_helper_if_compatible

    helper = tmp_path / "_protocol.py"
    _write_helper_if_compatible(helper, "n_states = 4\n")
    assert helper.read_text(encoding="utf-8").startswith(HELPER_FINGERPRINT)

    # The same content again is a no-op, not a rewrite.
    before = helper.stat().st_mtime_ns
    _write_helper_if_compatible(helper, "n_states = 4\n")
    assert helper.stat().st_mtime_ns == before

    with pytest.raises(SystemExit, match="generated from different content"):
        _write_helper_if_compatible(helper, "n_states = 8\n")
    assert "n_states = 4" in helper.read_text(encoding="utf-8"), "the refusal replaced it anyway"

    _write_helper_if_compatible(helper, "n_states = 8\n", force=True)
    assert "n_states = 8" in helper.read_text(encoding="utf-8")


def test_a_helper_without_a_fingerprint_is_refused_rather_than_assumed_current(tmp_path):
    from md_tools.remd.generated import _write_helper_if_compatible

    helper = tmp_path / "_protocol.py"
    helper.write_text("n_states = 4\n", encoding="utf-8")   # hand-written, or from an old build
    with pytest.raises(SystemExit, match="no fingerprint"):
        _write_helper_if_compatible(helper, "n_states = 4\n")


# --- group files ---------------------------------------------------------------------------------

def test_group_lines_that_disagree_about_their_inputs_are_refused(tmp_path):
    """A homogeneous ladder is N rungs of ONE system, differing only in tau."""
    from md_tools.remd.executor import GroupFileError, parse_group_file

    path = tmp_path / "ladder.group"
    path.write_text(
        "-i p.py -p a.pdb -s s.xml -c c.xml --solute solute.yaml --group-index 0\n"
        "-i p.py -p OTHER.pdb -s s.xml -c c.xml --solute solute.yaml --group-index 1\n",
        encoding="utf-8")
    with pytest.raises(GroupFileError, match="different values for topology"):
        parse_group_file(path)


def test_group_file_paths_resolve_against_the_group_file_not_the_working_directory(tmp_path):
    """Amber reads them this way, and so does everyone who writes one.

    Resolved against `os.getcwd()`, `mpirun` from one directory with a group file in another gave
    every rank a different idea of where its inputs were -- usually "nowhere", which is loud, and
    occasionally a DIFFERENT built.pdb, which is not.
    """
    from md_tools.remd.executor import parse_group_file

    elsewhere = tmp_path / "ladder"
    elsewhere.mkdir()
    path = elsewhere / "ladder.group"
    path.write_text(
        "-i p.py -p built.pdb -s s.xml -c c.xml --solute solute.yaml --group-index 0\n",
        encoding="utf-8")
    group = parse_group_file(path)[0]
    assert Path(group["topology"]) == elsewhere / "built.pdb"
    assert Path(group["solute"]) == elsewhere / "solute.yaml"


def test_the_group_file_is_written_with_paths_relative_to_itself(tmp_path):
    """The writer and the parser must agree about what a relative path is relative to.

    Found by running an installed wheel from OUTSIDE the checkout, which is the only place the
    two conventions come apart: the parser resolves against the group file (Amber's rule), the
    writer emitted `-p` and `-s` exactly as they arrived on the command line, and those are
    relative to the working directory. They agreed for as long as every ladder happened to be
    launched from its own output directory -- which every test in this suite did, and which a
    person running `md-openmm md-run -odir ./run` from their project root does not.

    Run from anywhere else, every rank looked for `built.pdb` beside the group file. Not finding
    it is the good case.
    """
    import argparse
    import os

    from md_tools.remd.executor import parse_group_file
    from md_tools.remd.generated import _group_file_text

    project = tmp_path / "project"
    run = tmp_path / "project" / "run"
    run.mkdir(parents=True)
    (project / "built.pdb").write_text("END\n", encoding="utf-8")
    (project / "built.xml").write_text("<System/>\n", encoding="utf-8")
    (run / "eq.xml").write_text("<State/>\n", encoding="utf-8")
    (run / "solute.yaml").write_text("n_solute_atoms: 1\n", encoding="utf-8")
    (run / "_protocol.py").write_text("n_states = 2\n", encoding="utf-8")

    # Exactly as a caller standing in `project` would spell them.
    args = argparse.Namespace(topology="built.pdb", system="built.xml",
                              continue_from="run/eq.xml")
    here = Path.cwd()
    os.chdir(project)
    try:
        text = _group_file_text("REST2", 2, {"tau_max": 0.5}, args,
                                run / "_protocol.py", run / "solute.yaml")
    finally:
        os.chdir(here)
    (run / "REST2.group").write_text(text, encoding="utf-8")

    # Parsed with the working directory somewhere else entirely, which is the whole point.
    os.chdir(tmp_path)
    try:
        groups = parse_group_file(run / "REST2.group")
    finally:
        os.chdir(here)

    assert len(groups) == 2
    for group in groups:
        assert Path(group["topology"]) == (project / "built.pdb").resolve(), group["topology"]
        assert Path(group["system"]) == (project / "built.xml").resolve(), group["system"]
        assert Path(group["coordinates"]) == (run / "eq.xml").resolve(), group["coordinates"]
    # Relative, not absolute: the directory has to stay movable, for the same reason a generated
    # script contains no absolute path.
    assert "/tmp" not in text and str(tmp_path) not in text, text


# --- a serial protocol under a plural launch -------------------------------------------------

@pytest.mark.slow
def test_a_cmd_stage_launched_under_mpirun_is_refused(workspace, tmp_path, good_config):
    """N ranks running one serial stage is N simulations over ONE set of output paths.

    Nothing partitions the work and nothing coordinates the writers, so the trajectory, the state
    table and the checkpoint are interleaved from N walkers -- and no file says so. The run
    "completes". This is the failure that is worst to discover late, because the output looks
    exactly like a successful run.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH. This test is in the slow lane precisely because it needs "
                    "a real launcher; a missing one there is an unmet criterion, not a neutral "
                    "absence.")

    destination = tmp_path / "plural-cmd"
    before = _snapshot(destination)
    done = _run(["mpirun", "-n", "2", sys.executable, str(workspace / "split" / "min.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 *PROTOCOL_ONLY],
                cwd=workspace / "split", environment=good_config)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "serial protocol" in message, message[-2000:]
    _untouched(destination, before)


@pytest.mark.slow
def test_the_all_in_one_workflow_is_refused_under_a_plural_launch(workspace, tmp_path,
                                                                   good_config):
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; see the note on the test above.")

    destination = tmp_path / "plural-chain"
    before = _snapshot(destination)
    done = _run(["mpirun", "-n", "2", sys.executable, str(workspace / "allinone" / "md.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 *PROTOCOL_ONLY],
                cwd=workspace / "allinone", environment=good_config)
    assert done.returncode != 0
    assert "serial protocol" in (done.stdout + done.stderr)
    _untouched(destination, before)


# --- AIS --overwrite is a complete fresh-run transaction ---------------------------------------

def test_ais_overwrite_removes_every_artefact_of_the_previous_run(tmp_path):
    """Removing only `AIS_run.json` was strictly worse than refusing.

    It deleted the one record saying which run the directory held and left every path directory,
    trajectory, checkpoint and aggregate table in place -- so the next invocation adopted them
    under a NEW identity, skipped them as "completed", and assembled one table out of two
    experiments. The refusal at least said something was wrong.
    """
    from md_tools.ais.run import clear_run_directory

    out = tmp_path / "AIS"
    (out / "path_0000" / "checkpoints").mkdir(parents=True)
    (out / "path_0000" / "completed.json").write_text("{}", encoding="utf-8")
    (out / "path_0000" / "checkpoints" / "generation_000001.chk").write_bytes(b"x")
    (out / "path_0001").mkdir()
    for name in ("AIS_run.json", "AIS_work.csv", "AIS_paths.csv", "AIS_hs.csv",
                 "selected_source_frames.csv", "AIS_traj0000.nc", "AIS_traj0001.nc"):
        (out / name).write_text("stale", encoding="utf-8")
    # A straggler from a previous, LONGER run: it looks exactly like this run's own output.
    (out / "AIS_traj0099.nc").write_text("stale", encoding="utf-8")
    # Something this run did not write, which must survive: --overwrite is not a delete button.
    (out / "resolved.config").write_text("protocol: AIS\n", encoding="utf-8")
    (out / "notes.txt").write_text("mine", encoding="utf-8")

    removed = clear_run_directory(out, paths=2)
    assert removed >= 9, removed
    assert not (out / "path_0000").exists() and not (out / "path_0001").exists()
    for name in ("AIS_run.json", "AIS_work.csv", "AIS_paths.csv", "AIS_hs.csv",
                 "selected_source_frames.csv", "AIS_traj0000.nc", "AIS_traj0001.nc",
                 "AIS_traj0099.nc"):
        assert not (out / name).exists(), name
    assert (out / "resolved.config").is_file(), "--overwrite deleted a file it did not write"
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


# --- a supplied group file is an input ----------------------------------------------------------

def test_a_supplied_group_file_does_not_cause_a_default_one_to_be_written(workspace, tmp_path,
                                                                          good_config):
    """It is read, not written. A default beside it is a file nothing reads and a later run adopts."""
    from md_tools.remd.executor import GroupFileError, parse_group_file  # noqa: F401

    destination = tmp_path / "supplied-group"
    destination.mkdir()
    supplied = destination / "mine.group"
    supplied.write_text(
        "-i _protocol.py -p ../built.pdb -s ../built.xml -c start.xml "
        "--solute solute.yaml --group-index 0\n"
        "-i _protocol.py -p ../built.pdb -s ../built.xml -c start.xml "
        "--solute solute.yaml --group-index 1\n", encoding="utf-8")

    done = _launch(workspace, "REST2", destination, "--groupfile", str(supplied),
                   *PROTOCOL_ONLY, environment=good_config)
    # It will refuse for some reason (the group file names inputs that do not exist here); what
    # matters is that it did not manufacture a REST2.group beside the one it was handed.
    assert not (destination / "REST2.group").exists(), (
        "a default group file was written although one was supplied:\n"
        + (done.stdout + done.stderr)[-1500:])


def test_resume_and_overwrite_together_are_refused(workspace, tmp_path, good_config):
    """One says continue, the other says start over. Precedence is the wrong way to settle that.

    The two outcomes are "your previous work continues" and "your previous work is gone", and a
    person who typed both should be told rather than given whichever the implementation happened
    to check first.
    """
    for mode in ("split", "AIS"):
        destination = tmp_path / f"contradiction-{mode}"
        before = _snapshot(destination)
        done = _launch(workspace, mode, destination, "--resume", "--overwrite", *PROTOCOL_ONLY,
                       environment=good_config)
        _refused(done, fragment="contradict")
        _untouched(destination, before)


# --- AIS identity is decided before a single byte is written -------------------------------------

def test_an_incompatible_ais_source_refuses_without_creating_the_directory(workspace, tmp_path,
                                                                           good_config):
    """The identity used to be checked AFTER `-odir` and both rank reports existed.

    So an incompatible source was refused by a run that had already produced a directory which,
    to anyone looking at it afterwards, is indistinguishable from a run that started. The check is
    in the preflight now: the source digest, the fingerprint, the identity document and the
    decision about what this invocation is all happen before anything is created.
    """
    import mdtraj

    destination = tmp_path / "identity"
    destination.mkdir()

    # A completed run in the directory, then the same command with a DIFFERENT source.
    first = _launch(workspace, "AIS", destination, *PROTOCOL_ONLY, environment=good_config)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    before = _snapshot(destination)
    assert before and any(name == "AIS_run.json" for name in before)

    other = tmp_path / "other_source.dcd"
    frames = mdtraj.load(str(workspace / "built.pdb"))
    mdtraj.join([frames] * 5).save_dcd(str(other))          # a different length, so a different digest

    done = _run([sys.executable, str(workspace / "AIS" / "AIS.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 "-source-traj", str(other), *PROTOCOL_ONLY],
                cwd=workspace / "AIS", environment=good_config)
    _refused(done, fragment="different AIS run")
    _untouched(destination, before)


def test_an_ais_refusal_on_a_fresh_directory_leaves_it_absent(workspace, tmp_path, good_config):
    """No directory, no reports, no frame table -- the refusal happens before `mkdir`."""
    import mdtraj

    destination = tmp_path / "never-made"
    # A source of the wrong system: refused by the atom-count check, which is inside the plan.
    wrong = tmp_path / "wrong.dcd"
    import numpy

    import mdtraj.core.element as element  # noqa: F401

    frames = mdtraj.load(str(workspace / "built.pdb"))
    sliced = frames.atom_slice(list(range(frames.n_atoms - 1)))
    mdtraj.join([sliced] * 4).save_dcd(str(wrong))

    done = _run([sys.executable, str(workspace / "AIS" / "AIS.py"),
                 "-p", "../built.pdb", "-s", "../built.xml", "-odir", str(destination),
                 "-source-traj", str(wrong), *PROTOCOL_ONLY],
                cwd=workspace / "AIS", environment=good_config)
    assert done.returncode != 0, done.stdout + done.stderr
    assert not destination.exists(), sorted(p.name for p in destination.iterdir())


def test_overwrite_deletes_only_names_this_run_owns(tmp_path):
    """`path_notes/` is not `path_0000/`, and a wildcard cannot tell them apart.

    The previous cleanup globbed `path_*` and removed every directory that matched -- so a working
    directory named `path_notes` beside the run would have gone with it. `--overwrite` is not a
    licence to delete a directory because its name is suggestive.
    """
    from md_tools.ais.run import clear_run_directory

    out = tmp_path / "AIS"
    (out / "path_0000" / "checkpoints").mkdir(parents=True)
    (out / "path_0001").mkdir()
    (out / "path_notes").mkdir()                            # NOT owned: five characters in common
    (out / "path_0000" / "completed.json").write_text("{}", encoding="utf-8")
    (out / "path_notes" / "todo.md").write_text("mine", encoding="utf-8")
    for name in ("AIS_run.json", "AIS_work.csv", "AIS_paths.csv", "AIS_hs.csv",
                 "selected_source_frames.csv", "AIS_traj0000.nc", "AIS_traj0099.nc",
                 "AIS.out", "AIS.out.rank05", "AIS.log", "AIS.log.rank05"):
        (out / name).write_text("stale", encoding="utf-8")
    (out / "AIS_traj_notes.nc").write_text("not the schema", encoding="utf-8")
    (out / "resolved.config").write_text("protocol: AIS\n", encoding="utf-8")

    removed = clear_run_directory(out, paths=2, ranks=2)

    assert not (out / "path_0000").exists() and not (out / "path_0001").exists()
    assert (out / "path_notes" / "todo.md").read_text(encoding="utf-8") == "mine", (
        "--overwrite deleted a directory it does not own")
    assert (out / "AIS_traj_notes.nc").is_file(), (
        "--overwrite deleted a file whose name does not match the trajectory schema")
    assert (out / "resolved.config").is_file()
    for name in ("AIS_run.json", "AIS_work.csv", "AIS_hs.csv", "AIS_traj0000.nc",
                 "AIS_traj0099.nc", "AIS.out", "AIS.log"):
        assert not (out / name).exists(), name
    # Rank reports from a LARGER previous world go too: an overwrite from six ranks to two left
    # rank05's report beside the new one, equally current-looking.
    assert not (out / "AIS.out.rank05").exists()
    assert not (out / "AIS.log.rank05").exists()
    assert removed >= 11, removed


# --- the all-in-one chain is planned in full before stage 1 runs --------------------------------

def test_an_invalid_last_stage_stops_the_chain_before_stage_one_writes_anything(
        workspace, tmp_path, good_config):
    """Only the FIRST stage used to be preflighted.

    So a chain whose later stages are invalid ran every earlier stage to completion and then
    refused -- leaving those stages' output on disk and no way to finish. That is exactly what
    `--check` on a chain is supposed to make impossible, and the run itself had none of it.

    The invalid stage here is produced by real configuration rather than by corrupting one: an
    EXPLICIT-solvent project (whose later stages are NPT) run against the IMPLICIT System this
    workspace built. `min` is NVT and fine; the first NPT stage has no volume to control.
    """
    import yaml as _yaml

    project = tmp_path / "explicit-plan"
    (tmp_path / "explicit.config").write_text(_yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                   "production_steps": 5},
        "reporting": {"solute_printout": 5, "system_printout": 5,
                      "checkpoint_printout": 5}}), encoding="utf-8")
    built = _run([sys.executable, "-m", "md_tools.cli.md_openmm", "build-md",
                  "-odir", str(project), "--config", str(tmp_path / "explicit.config"),
                  "--all-in-one"], cwd=tmp_path, environment=good_config)
    assert built.returncode == 0, built.stdout + built.stderr

    destination = tmp_path / "chain-out"
    before = _snapshot(destination)
    done = _run([sys.executable, str(project / "md.py"),
                 "-p", str(workspace / "built.pdb"), "-s", str(workspace / "built.xml"),
                 "-odir", str(destination), *PROTOCOL_ONLY],
                cwd=project, environment=good_config)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "no volume to control" in message, message[-2500:]
    # THE POINT: `min` is a perfectly valid stage and it produced nothing, because the chain was
    # planned in full before any of it ran.
    assert _snapshot(destination) == before, (
        f"the chain ran before validating its later stages: "
        f"{sorted((_snapshot(destination) or {}).keys())}")


def test_two_stages_writing_one_path_are_refused(tmp_path):
    """A cross-stage collision no single stage's own inventory can see.

    Exercised directly: the generator gives every stage a distinct name, so this state cannot
    arise from a valid project, and a test that corrupted a configuration to reach it would be
    testing the corruption rather than the guard.
    """
    from md_tools.md.stage import cross_stage_collision
    from md_tools.run.preflight import OutputInventory

    class _Plan:
        def __init__(self, roles):
            self.inventory = OutputInventory(roles=roles)

    plans = {
        "min": _Plan({"trajectory": tmp_path / "a.dcd", "restart": tmp_path / "min.xml"}),
        "prod": _Plan({"trajectory": tmp_path / "b.dcd", "restart": tmp_path / "prod.xml"}),
    }
    assert cross_stage_collision(plans) is None

    plans["prod"].inventory.roles["trajectory"] = tmp_path / "a.dcd"
    complaint = cross_stage_collision(plans)
    assert complaint is not None
    assert "min" in complaint and "prod" in complaint and "a.dcd" in complaint


# --- the ladder's overwrite is one policy --------------------------------------------------------

def test_generated_overwrite_reaches_the_executor():
    """`--overwrite` regenerated the helpers and then did not reach the run they were for.

    The executor refuses existing outputs unless told otherwise, and only `--force` told it. So
    `--overwrite` rewrote `solute.yaml`, `_protocol.py` and the group file and was then turned
    away -- leaving the user to pass a second flag for the same intent, with the first already
    partially applied.
    """
    import inspect

    from md_tools.remd import generated

    source = inspect.getsource(generated.replica_main)
    assert "args.force or args.overwrite" in source, (
        "--overwrite does not reach the executor, so it can update the helpers and then be "
        "refused by the run they were prepared for")


def test_verify_only_is_read_only_on_an_absent_target(workspace, tmp_path, good_config):
    """A verification that creates what it was asked to verify has verified nothing."""
    destination = tmp_path / "verify-absent"
    before = _snapshot(destination)
    done = _launch(workspace, "REST2", destination, "--verify-only", *PROTOCOL_ONLY,
                   environment=good_config)
    assert done.returncode != 0, done.stdout + done.stderr
    assert _snapshot(destination) == before, (
        f"--verify-only created {sorted((_snapshot(destination) or {}).keys())} in a directory "
        f"that held no ladder")


def test_verify_only_is_read_only_on_a_corrupt_target(workspace, tmp_path, good_config):
    """And on a directory holding a damaged ladder, which is when it is actually used."""
    destination = tmp_path / "verify-corrupt"
    destination.mkdir()
    (destination / "solute.yaml").write_text("n_solute_atoms: 1\n", encoding="utf-8")
    (destination / "_protocol.py").write_text("n_states = 2\n", encoding="utf-8")
    (destination / "REST2.group").write_text("# truncated\n", encoding="utf-8")
    (destination / "REST2.nc").write_bytes(b"not a netcdf file at all")
    before = _snapshot(destination)

    done = _launch(workspace, "REST2", destination, "--verify-only", *PROTOCOL_ONLY,
                   environment=good_config)
    assert done.returncode != 0, done.stdout + done.stderr
    assert _snapshot(destination) == before, (
        "--verify-only modified the damaged ladder it was inspecting")


# --- the ladder inventory names everything a ladder writes ---------------------------------------

def test_the_ladder_inventory_names_every_artefact_a_ladder_writes(tmp_path):
    """An output nothing names is an output nothing can check.

    A ladder writes far more than `-x`, `-r` and `-o`: a helper protocol, a solute selection, a
    group file, a reservoir declaration, one report PER RANK, one trajectory per state, a
    checkpoint generation tree, a rem log and a provenance record. Each one missing from the
    inventory is a file `--overwrite` leaves behind and existing-output checking cannot see -- so
    a two-state ladder run over a six-state one keeps four stale state trajectories and four
    stale rank reports, all looking equally current.
    """
    from md_tools.run.preflight import _ladder_inventory

    out = tmp_path / "REST2.out"
    inventory = _ladder_inventory(
        protocol="rREST2", replicas=4, output=out, log=tmp_path / "REST2.log",
        trajectory=tmp_path / "REST2.nc", restart=tmp_path / "restart.json",
        checkpoint=tmp_path / "REST2.chk", groupfile=None, reservoir=True)
    named = {path.name for path in inventory.roles.values()}

    for required in ("reservoir.yaml", "solute.yaml", "_protocol.py", "rREST2.group",
                     "rem.log", "machine.yaml", "restart.json", "REST2.chk",
                     "REST2.checkpoints", "REST2.out", "REST2.log", "REST2.nc"):
        assert required in named, f"{required} is written and is in no inventory: {sorted(named)}"
    for state in range(4):
        assert f"remd{state}.nc" in named, f"state {state}'s trajectory is unnamed"
    for rank in range(1, 4):
        assert f"REST2.out.rank{rank:02d}" in named, (
            f"rank {rank}'s report is unnamed -- it is the file a rank that failed to bind its "
            f"device writes into")

    # The checkpoint tree is READ by a resume, so its presence must never be the refusal.
    assert "checkpoints" in inventory.resumable


def test_a_ladder_without_a_reservoir_does_not_claim_one(tmp_path):
    """The other direction: naming a file the run never writes refuses a directory for nothing."""
    from md_tools.run.preflight import _ladder_inventory

    inventory = _ladder_inventory(
        protocol="REST2", replicas=2, output=tmp_path / "REST2.out", log=None,
        trajectory=None, restart=None, checkpoint=None, groupfile=None, reservoir=False)
    assert "reservoir_declaration" not in inventory.roles
