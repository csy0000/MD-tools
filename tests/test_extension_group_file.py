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
