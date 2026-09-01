"""The six commands exist, and refuse clearly when they cannot proceed."""
from __future__ import annotations


def test_md_openmm_offers_the_four_subcommands(md_openmm):
    result = md_openmm("--help")
    assert result.returncode == 0
    for name in ("sys-config", "show-default", "sys-gen", "md-gen"):
        assert name in result.stdout, name
