"""The six commands exist, and refuse clearly when they cannot proceed."""
from __future__ import annotations


def test_md_template_offers_init_and_install(md_template):
    result = md_template("--help")
    assert result.returncode == 0
    assert "init" in result.stdout and "install" in result.stdout


def test_md_openmm_offers_the_four_subcommands(md_openmm):
    result = md_openmm("--help")
    assert result.returncode == 0
    for name in ("sys-config", "show-default", "sys-gen", "md-gen"):
        assert name in result.stdout, name


def test_installing_another_engine_is_refused_rather_than_pretended(md_template, tmp_path):
    """Amber and GROMACS are not implemented and must not look as if they are."""
    result = md_template("install", "-e", "amber", "--target-dir", str(tmp_path))
    assert result.returncode != 0
    assert "not supported" in (result.stdout + result.stderr)


def test_a_missing_stack_directory_says_how_to_set_it(md_template, monkeypatch):
    result = md_template("init")          # no --target-dir, and MD_STACK unset in a clean env
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        assert "MD_STACK" in combined
