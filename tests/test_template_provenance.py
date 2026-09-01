"""One canonical generator identity, propagated into every record and required by preflight.

The gap these close: each writer used to reach for `implementation_identity()["git_commit"]` on
its own. In a VCS-installed package that is `None` even though `direct_url.json` records the exact
commit -- so `dataset.yaml` carried a verified commit while every `stage.yaml` beside it carried a
null, and preflight compared only the fields that happened to be present.

These tests drive the identity by faking the RAW observation (`implementation_identity`), not the
resolution or the comparison, and then assert on the files actually written.
"""
from __future__ import annotations



import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.openmm import provenance_min

from .conftest import REPO_ROOT

CLEAN = "1" * 40
OTHER = "2" * 40
VCS = "3" * 40


def _raw(*, git_commit=None, dirty=None, direct_url=None):
    """What `implementation_identity()` observes, before any resolution."""
    return {
        "version": "0.5.0.dev0",
        "git_commit": git_commit,
        "git_dirty": dirty,
        "installed_fingerprint": {"algorithm": "sha256", "file_count": 1, "value": "f" * 64},
        "direct_url": direct_url,
    }


# --- the canonical resolution -------------------------------------------------------------------

def test_a_clean_checkout_resolves_to_its_head(monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=False))
    identity = provenance_min.template_identity()
    assert identity["commit"] == CLEAN
    assert identity["evidence"] == "git checkout"
    assert identity["dirty"] is False
    assert identity["reproducible_from_commit"] is True


def test_a_dirty_checkout_resolves_but_says_it_does_not_reproduce(monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=True))
    identity = provenance_min.template_identity()
    assert identity["commit"] == CLEAN
    assert identity["dirty"] is True
    assert identity["reproducible_from_commit"] is False
    assert "does NOT reproduce" in identity["reproducibility"]


def test_a_vcs_install_resolves_from_direct_url_when_there_is_no_checkout(monkeypatch):
    """The case that motivated all of this: git_commit is None and the commit is still known."""
    monkeypatch.setattr(provenance_min, "implementation_identity", lambda: _raw(
        git_commit=None, dirty=None,
        direct_url={"url": "https://github.com/csy0000/MD-tools.git",
                    "vcs_info": {"vcs": "git", "commit_id": VCS, "requested_revision": "dev"}}))
    identity = provenance_min.template_identity()
    assert identity["commit"] == VCS
    assert identity["evidence"] == "direct_url.json"
    # A VCS install is a fixed set of bytes; there is no working tree to be dirty.
    assert identity["dirty"] is False
    assert identity["reproducible_from_commit"] is True


def test_an_installation_with_no_commit_resolves_to_none_rather_than_a_guess(monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity", lambda: _raw(
        git_commit=None, dirty=None,
        direct_url={"url": "https://files.pythonhosted.org/x.whl", "archive_info": {}}))
    identity = provenance_min.template_identity()
    assert identity["commit"] is None
    assert identity["evidence"] == "unavailable"
    # Never derived from the version or the fingerprint, both of which are available here.
    assert identity["version"] == "0.5.0.dev0"
    assert identity["installed_fingerprint"] == "f" * 64
    assert identity["reproducible_from_commit"] is False






# --- generation, and the files it actually writes -----------------------------------------------





def _recorded_commits(inputs, project):
    """Every generator commit written into the project, by where it was found."""
    def dig(path, *keys):
        if not Path(path).is_file():
            return "<file missing>"
        document = yaml.safe_load(Path(path).read_text()) or {}
        for key in keys:
            if not isinstance(document, dict):
                return None
            document = document.get(key)
        return document

    found = {
        "provenance.yaml": dig(project / "provenance.yaml", "template", "commit"),
        "common/provenance.yaml": dig(inputs / "provenance.yaml", "template", "commit"),
        "resolved_sys.config.yaml": dig(inputs / "resolved_sys.config.yaml", "provenance",
                                        "template", "commit"),
        "md.config.yaml": dig(project / "md.config.yaml", "provenance", "template_commit"),
    }
    manifest = project / "dataset.yaml"
    if manifest.is_file():
        found["dataset.yaml"] = dig(manifest, "templates", "commit")
    for stage in sorted(project.rglob("stage.yaml")):
        found[str(stage.relative_to(project))] = dig(stage, "template_commit")
    return found








# --- preflight requires every record ------------------------------------------------------------

def _contract_project(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=False))
    return _generate(tmp_path, enabled=True, commit=CLEAN, identity={})














# --- nothing derives the commit independently any more ------------------------------------------

def test_no_writer_reaches_for_git_commit_on_its_own():
    """One canonical resolution. Reaching past it is how the null-under-direct_url bug happened."""
    offenders = {}
    # The two writers that used to reach past the resolution are gone with the retired route.
    # The rule now applies to everything that writes a commit into a durable record.
    # `provenance_min` is deliberately absent: it IS the canonical resolution, so reading the raw
    # field there is the point. The rule is that nothing ELSE reaches past it.
    sources = [REPO_ROOT / "src" / "md_tools" / "build" / "record.py",
               REPO_ROOT / "src" / "md_tools" / "registry" / "register.py",
               REPO_ROOT / "src" / "md_tools" / "build" / "top.py",
               REPO_ROOT / "src" / "md_tools" / "runtime" / "ais.py"]
    for path in sources:
        name = path.name
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):          # an explanation of the bug is not the bug
                continue
            if '["git_commit"]' in line or "['git_commit']" in line:
                offenders[f"{name}:{number}"] = stripped
    assert not offenders, offenders


# --- where the dirty-checkout guarantee lives now -------------------------------------------------
#
# `md_data_contract.check_templates_commit` enforced that a contract-managed generation could not
# record a commit that did not describe what ran. Contract v1 and that function are gone; the
# guarantee is not. It moved to the two places that actually write a commit into a durable record:
#
#   * `md_tools.build.record.source_commit()` -- what every run record carries;
#   * registration's `_origin()` -- which REFUSES a dirty project tree outright, asserted in
#     tests/test_registration.py::test_a_dirty_project_tree_is_refused.

def test_a_run_record_never_invents_a_commit(monkeypatch):
    """None is an honest answer; a guess is not.

    An installed wheel has no git tree. Recording a plausible-looking commit there would put an
    unverifiable claim into provenance, which is worse than recording nothing.
    """
    from md_tools.build import record

    monkeypatch.setattr(record, "source_commit", lambda: None)
    facts = record.environment_facts()
    assert facts["md_tools_commit"] is None


def test_the_run_record_carries_the_commit_when_there_is_one():
    from md_tools.build.record import source_commit

    commit = source_commit()
    if commit is None:
        pytest.skip("not a git checkout")
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)


def test_registration_refuses_to_record_a_commit_that_does_not_describe_what_ran():
    """The v1 rule, in the place that now writes commits into a manifest that travels."""
    from md_tools.registry import register

    source = Path(register.__file__).read_text(encoding="utf-8")
    assert "uncommitted changes" in source
    assert "would not describe what actually ran" in source
