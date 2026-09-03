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
