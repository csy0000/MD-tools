"""The command surface of `md-openmm`, in ONE place.

This list is a guard, not a description. Two tests assert that the parser offers exactly these
names -- `tests/test_cli.py` because the surface is the contract, and `tests/test_ais_build_md.py`
because AIS must never acquire a command of its own. Adding a name here is therefore a deliberate
act with a reviewer attached, which is the whole point: a CLI grows a command at a time, and each
one is cheap to add and permanent to keep.

It lives in its own module because the list used to be written out twice, once in each of those
files. Adding `export-reference` then failed two tests in two files for one reason, and the
obvious repair -- edit both literals -- is exactly the repair that keeps them able to disagree.

The count is deliberately not in any test name. It was: `test_the_help_lists_exactly_the_three_
public_commands` was still called "three" long after `data-register` made it four, so the name had
stopped describing the assertion while the assertion was still correct. A name that has to be
maintained alongside the thing it counts eventually lies about it.
"""
from __future__ import annotations

#: Every subcommand `md-openmm` offers.
PUBLIC_COMMANDS = ("build-top", "build-md", "md-run", "data-register", "export-reference")

#: Spellings that existed once and must not come back. Not hidden, not deprecated: gone.
RETIRED_COMMANDS = ("sys-config", "sys-gen", "md-gen", "setup", "show-default")
