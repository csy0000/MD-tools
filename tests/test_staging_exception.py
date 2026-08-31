"""The one exception to "generated data never lives inside a repository".

The rule exists because an ignore rule is not protection: it is one `git add -f` from being wrong.
That reasoning stands. What it did not allow for is the staging step the registration pipeline is
built on -- a dataset is generated project-locally, finished, verified, and only then moved under
$MD_DATA by `md-data-register`.

The exception is therefore narrow and CHECKED, not asserted. All three of an opt-in variable, a
visible marker file, and git's own opinion of whether the path is ignored must agree. Any one of
them missing and the original refusal stands, because each covers a different way of being wrong:
the variable is someone's intent, the marker is that intent left where the next reader can see it,
and `check-ignore` is the only one of the three that is a fact.

PLATFORM_POLICY_EXEMPTION: git and the filesystem. Nothing runs.
"""
from __future__ import annotations

import subprocess

import pytest

from md_templates.openmm.simple import ConfigError, resolve_output_root


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A repository whose `data/` contents are ignored, generated into as staging would be."""
    repository = tmp_path / "repo"
    (repository / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("/data/**\n", encoding="utf-8")
    monkeypatch.setenv("MD_DATA", str(repository / "data"))
    monkeypatch.delenv("MD_TEMPLATES_ALLOW_STAGING", raising=False)
    return repository / "data"


def _resolve(root):
    return resolve_output_root(str(root))


def test_a_declared_ignored_staging_area_is_accepted(staged, monkeypatch):
    monkeypatch.setenv("MD_TEMPLATES_ALLOW_STAGING", "1")
    (staged / ".md-staging").write_text("", encoding="utf-8")
    root, relative = _resolve(staged)
    assert root == staged.resolve()
    assert relative == "."


def test_without_the_opt_in_the_refusal_stands(staged):
    (staged / ".md-staging").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        _resolve(staged)


def test_without_the_marker_file_the_refusal_stands(staged, monkeypatch):
    """The environment variable alone is one person's shell. The marker is what the next reader
    of the tree sees."""
    monkeypatch.setenv("MD_TEMPLATES_ALLOW_STAGING", "1")
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        _resolve(staged)


def test_a_directory_that_is_not_actually_ignored_is_refused(tmp_path, monkeypatch):
    """The case the whole rule exists for: someone declares a staging area inside a repository
    where nothing is ignored. The declaration does not make it safe, and git is asked rather
    than believed."""
    repository = tmp_path / "repo"
    (repository / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    # no .gitignore at all
    monkeypatch.setenv("MD_DATA", str(repository / "data"))
    monkeypatch.setenv("MD_TEMPLATES_ALLOW_STAGING", "1")
    (repository / "data" / ".md-staging").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        _resolve(repository / "data")


def test_an_ignore_rule_that_covers_only_some_children_is_refused(tmp_path, monkeypatch):
    """What must be ignored is where the DATA lands, whatever it ends up being called.

    Ignoring `data/REST2` and then generating `data/ALA` leaves the new system fully trackable.
    The check asks about a system directory rather than about the root, so this is caught; a
    check on the root alone would have passed it.
    """
    repository = tmp_path / "repo"
    (repository / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("/data/REST2/**\n", encoding="utf-8")
    monkeypatch.setenv("MD_DATA", str(repository / "data"))
    monkeypatch.setenv("MD_TEMPLATES_ALLOW_STAGING", "1")
    (repository / "data" / ".md-staging").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="inside the Git working tree"):
        _resolve(repository / "data")


def test_a_root_outside_any_repository_needs_none_of_this(tmp_path, monkeypatch):
    monkeypatch.setenv("MD_DATA", str(tmp_path))
    monkeypatch.delenv("MD_TEMPLATES_ALLOW_STAGING", raising=False)
    root, _ = _resolve(tmp_path)
    assert root == tmp_path.resolve()


def test_the_refusal_says_exactly_what_would_make_it_pass(staged):
    """A refusal that does not say how to proceed gets worked around rather than satisfied."""
    with pytest.raises(ConfigError) as raised:
        _resolve(staged)
    message = str(raised.value)
    assert ".md-staging" in message
    assert "MD_TEMPLATES_ALLOW_STAGING=1" in message
    assert "check-ignore" in message
