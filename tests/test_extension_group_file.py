"""An extension must be able to run the group file an extension generates.

`--extend-from` takes the physical state from the parent's checkpoint, so it correctly writes no
`-c` on any group line -- and the parser required one on every line, so every rank aborted with
`no coordinates given` on a run that had asked for nothing wrong. Nothing exercised the generated
file against the parser that has to read it, which is why the two halves could disagree.
"""
from __future__ import annotations

import pathlib

import pytest

from md_tools.remd.executor import GroupFileError, parse_group_file

LINE = ("-i _protocol.py -p built.pdb -s built.xml --solute solute.yaml --group-index {i}")
LINE_WITH_C = ("-i _protocol.py -p built.pdb -s built.xml -c eq.xml --solute solute.yaml "
               "--group-index {i}")


def _write(tmp_path, template, n=4):
    path = tmp_path / "REST2.group"
    path.write_text("# a generated group file\n"
                    + "\n".join(template.format(i=i) for i in range(n)) + "\n")
    return path


def test_a_fresh_run_still_requires_coordinates(tmp_path):
    """The rule is not removed, only made conditional: a fresh ladder has nothing to start from."""
    with pytest.raises(GroupFileError, match="no coordinates given"):
        parse_group_file(_write(tmp_path, LINE))


def test_the_refusal_says_when_coordinates_are_optional(tmp_path):
    with pytest.raises(GroupFileError, match="only with --extend-from"):
        parse_group_file(_write(tmp_path, LINE))


def test_an_extension_parses_a_group_file_with_no_coordinates(tmp_path):
    groups = parse_group_file(_write(tmp_path, LINE), extending=True)
    assert len(groups) == 4
    assert all("coordinates" not in g for g in groups)
    assert [g["group_index"] for g in groups] == [0, 1, 2, 3]


def test_an_extension_still_accepts_coordinates_if_present(tmp_path):
    """A caller may pass -c anyway -- the value is inert, not forbidden."""
    groups = parse_group_file(_write(tmp_path, LINE_WITH_C), extending=True)
    # Resolved against the group file's directory, as every path on a group line is.
    assert all(pathlib.Path(g["coordinates"]).name == "eq.xml" for g in groups)
    assert all(pathlib.Path(g["coordinates"]).is_absolute() for g in groups)


def test_the_other_required_fields_are_still_required_when_extending(tmp_path):
    path = tmp_path / "REST2.group"
    path.write_text("-i _protocol.py -p built.pdb --group-index 0\n")
    with pytest.raises(GroupFileError, match="no system given"):
        parse_group_file(path, extending=True)


def test_homogeneity_is_unaffected_by_absent_coordinates(tmp_path):
    """Lines that all omit -c agree about it, so the homogeneous-ladder check must not fire."""
    parse_group_file(_write(tmp_path, LINE), extending=True)      # would raise if it did


def test_the_executor_can_ANNOUNCE_a_group_file_with_no_coordinates(tmp_path, capsys):
    """Parsing it is half the job; the executor then PRINTS what it is about to run.

    `_announce` read `group['coordinates']` unconditionally, so an extension died with
    `KeyError: 'coordinates'` immediately after writing its header -- every rank, before any
    dynamics, on a run that had asked for nothing wrong. The parser had been made conditional and
    this half had not, which is the same defect one step further along: the two halves of one
    change have to be exercised together, and only a test that gets past the parser does that.
    """
    from types import SimpleNamespace

    from md_tools.remd.executor import _announce

    groups = parse_group_file(_write(tmp_path, LINE, n=2), extending=True)
    arguments = SimpleNamespace(groupfile=str(tmp_path / "REST2.group"), exchange_rule=None,
                                reservoir=None)
    files = SimpleNamespace(trajectory="REST2.nc", checkpoint="REST2_checkpoint.nc",
                            restart="restart.json")
    _announce(arguments, files, groups, 0, 2)                     # must not raise
    printed = capsys.readouterr().out
    assert "group 0" in printed and "group 1" in printed
    assert "continued from the parent's checkpoint" in printed, \
        "an extension's announcement must say where its state comes from, not omit it"


def test_every_consumer_of_a_group_reads_its_coordinates_conditionally():
    """Parsing it is not enough: whatever READS a parsed group must tolerate an absent `-c`.

    Two consumers did not -- `_announce` and `run_grouped` -- so an extension died with
    `KeyError: 'coordinates'` twice over, at the second one only after the first was fixed. A
    source check is the honest guard here: reaching `run_grouped` needs a real ladder, and the
    defect is textual, in a file this test can read.
    """
    import re

    from md_tools.remd import executor

    source = pathlib.Path(executor.__file__).read_text(encoding="utf-8")
    offenders = [line.strip() for line in source.splitlines()
                 if re.search(r"""\[['"]coordinates['"]\]""", line)
                 # A read GUARDED on the same line is the fix, not the defect: the announcement
                 # prints the name only when there is one.
                 and 'get("coordinates")' not in line and "get('coordinates')" not in line]
    assert not offenders, (
        "a parsed group's coordinates must be read with .get(): an extension's group file has "
        f"none. Offending line(s): {offenders}")


def test_a_generated_extension_group_file_is_parseable_by_this_parser(tmp_path):
    """The regression in one line: generate the shape --extend-from writes, then parse it."""
    generated = tmp_path / "REST2.group"
    generated.write_text(
        "# md-tools-helper-sha256:0000\n"
        "# REST2: 2 states.\n"
        "# One group per line, inputs only.\n\n"
        "-i _protocol.py -p ../../build/built.pdb -s ../../build/built.xml "
        "--solute solute.yaml --group-index 0\n"
        "-i _protocol.py -p ../../build/built.pdb -s ../../build/built.xml "
        "--solute solute.yaml --group-index 1\n")
    assert len(parse_group_file(generated, extending=True)) == 2
