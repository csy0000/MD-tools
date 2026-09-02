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
from .test_direct_runtime_preflight import (ENTRY, MODES, VALID_USER, _config,  # noqa: F401
                                            _launch, _refused, _run, _snapshot, workspace)

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
    done = _launch(workspace, "allinone", destination, "--check", environment=good_config)
    assert done.returncode == 0, done.stdout + done.stderr
    # `--check` on the whole chain must also write nothing at all.
    _untouched(destination, before)


def test_a_first_stage_continuation_must_exist_even_under_check(workspace, tmp_path,
                                                                good_config):
    """`-c` for the FIRST stage names a file from outside this chain. Nothing here will write it."""
    destination = tmp_path / "check-first-c"
    before = _snapshot(destination)
    done = _launch(workspace, "allinone", destination, "--check", "-c", "not_here.xml",
                   environment=good_config)
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
    extra = ["--check"]
    done = _launch(workspace, mode, destination, *extra, environment=good_config)
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
