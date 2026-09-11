"""The public command surface, with contractual option spellings.

The single-dash multi-character options (`-os`, `-op`, `-log`, `-odir`, `-idata`, `-project_name`,
`-data_name`, `-year`, `-p`, `-s`) are what the documented examples type. argparse's prefix
matching would let several near-misses through silently, so each spelling is asserted rather than
assumed.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from md_tools.cli.md_openmm import build_parser

from tests.public_commands import PUBLIC_COMMANDS, RETIRED_COMMANDS


def test_the_help_lists_every_public_command(md_openmm):
    result = md_openmm("--help")
    assert result.returncode == 0
    for name in PUBLIC_COMMANDS:
        assert name in result.stdout, f"{name} is missing from the top-level help"


def test_the_retired_commands_are_gone():
    """Not hidden, not deprecated: gone. A legacy route left in place becomes the real one.

    Deliberately invokes the REAL command rather than the `md_openmm` fixture. conftest routes
    the retired subcommand names to the surviving generator API so that the scientific tests
    written against them keep running; this test is about the command surface, so it must not go
    through that shim.
    """
    for name in RETIRED_COMMANDS:
        result = subprocess.run([sys.executable, "-m", "md_tools.cli.md_openmm", name, "--help"],
                                capture_output=True, text=True)
        assert result.returncode != 0, f"{name} still runs"
        assert "invalid choice" in result.stderr or "usage:" in result.stderr


def test_the_subparsers_are_exactly_the_public_commands():
    actions = [a for a in build_parser()._actions if hasattr(a, "choices") and a.choices]
    names = set()
    for action in actions:
        names |= {str(k) for k in action.choices}
    assert names == set(PUBLIC_COMMANDS), names


@pytest.mark.parametrize("command", PUBLIC_COMMANDS)
def test_each_command_has_help_without_importing_cuda(command):
    """`-h` must work on a machine with no GPU and no OpenMM context.

    Run in a subprocess with OpenMM poisoned: if the parser imports it at construction time, this
    fails, which is the whole point.
    """
    program = (
        "import sys\n"
        "class _Poison:\n"
        "    def __getattr__(self, name):\n"
        "        raise AssertionError('OpenMM must not be imported to print help')\n"
        "sys.modules['openmm'] = _Poison()\n"
        "from md_tools.cli.md_openmm import main\n"
        f"raise SystemExit(main([{command!r}, '-h']))\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("argv, attribute, expected", [
    (["build-top", "-i", "x.pdb"], "input", "x.pdb"),
    (["build-top", "-i", "x.pdb", "-os", "s.xml"], "out_system", "s.xml"),
    (["build-top", "-i", "x.pdb", "-op", "p.pdb"], "out_pdb", "p.pdb"),
    (["build-top", "-i", "x.pdb", "-log", "b.log"], "out_log", "b.log"),
    (["build-md", "-odir", "./scripts/"], "out_dir", "./scripts/"),
    (["data-register", "-idata", "./data/ALA"], "idata", "./data/ALA"),
    (["data-register", "-project_name", "ALA"], "project_name", "ALA"),
    (["data-register", "-data_name", "ALA-cMD"], "data_name", "ALA-cMD"),
    (["data-register", "-year", "2026"], "year", "2026"),
])
def test_the_contractual_single_dash_spellings_are_accepted(argv, attribute, expected):
    args = build_parser().parse_args(argv)
    assert getattr(args, attribute) == expected


def test_the_documented_defaults_are_what_the_parser_applies():
    args = build_parser().parse_args(["build-top", "-i", "x.pdb"])
    assert (args.out_system, args.out_pdb, args.out_log) == ("./built.xml", "./built.pdb",
                                                             "./built.log")
    assert build_parser().parse_args(["build-md"]).out_dir == "./md_script/"


def test_version_reports_the_distribution(md_openmm):
    result = md_openmm("--version")
    assert result.returncode == 0
    assert "md-tools" in result.stdout
