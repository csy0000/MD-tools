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

from md_templates.openmm import md_data_contract as MD
from md_templates.openmm import provenance_min

from .conftest import ALA_PDB, REPO_ROOT, template_module

CLEAN = "1" * 40
OTHER = "2" * 40
VCS = "3" * 40


def _raw(*, git_commit=None, dirty=None, direct_url=None):
    """What `implementation_identity()` observes, before any resolution."""
    return {
        "version": "0.4.0.dev0",
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
        direct_url={"url": "https://github.com/csy0000/MD-templates.git",
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
    assert identity["version"] == "0.4.0.dev0"
    assert identity["installed_fingerprint"] == "f" * 64
    assert identity["reproducible_from_commit"] is False


def test_the_contract_refuses_a_dirty_checkout(monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=True))
    with pytest.raises(MD.ContractError) as error:
        MD.check_templates_commit(CLEAN)
    message = str(error.value)
    assert "UNCOMMITTED CHANGES" in message
    assert "false provenance" in message
    assert "dataset.enabled false" in message


def test_the_contract_accepts_a_vcs_install(monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity", lambda: _raw(
        git_commit=None, direct_url={"url": "https://github.com/csy0000/MD-templates.git",
                                     "vcs_info": {"commit_id": VCS}}))
    assert MD.check_templates_commit(VCS)["commit"] == VCS
    with pytest.raises(MD.ContractError):
        MD.check_templates_commit(OTHER)


# --- generation, and the files it actually writes -----------------------------------------------

def _sys_config(work, *, enabled, commit):
    from .conftest import run_cli

    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    assert run_cli("md_openmm", "sys-config", "--method", "cMD", cwd=work).returncode == 0
    path = work / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    if enabled:
        document["dataset"].update({
            "enabled": True, "dataset_id": "proj-2026-08-ala", "namespace": "proj",
            "dataset_name": "ala", "role": "project", "system": "ALA in TIP3P",
            "created_by": {"person_id": "chen", "name": "Chen", "affiliation": None,
                           "orcid": None},
            "origin": {"repository": "https://github.com/csy0000/example", "commit": commit}})
        document["dataset"]["templates"]["commit"] = commit
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _generate(work, *, enabled, commit, identity):
    """Generate a whole project IN-PROCESS, with the raw identity observation faked.

    In-process because the identity has to be steered and a subprocess cannot be monkeypatched;
    the assertions are all on the files this actually writes, not on the patched function.
    """
    from md_templates.openmm import mdgen, sysgen

    _sys_config(work, enabled=enabled, commit=commit)
    environment = dict(os.environ)
    if enabled:
        root = work / "MD_DATA"
        local = root / "proj" / "2026-08" / "ala"
        local.mkdir(parents=True)
        os.environ.update({"MD_DATA": str(root), "MD_DATA_LOCAL": str(local)})
        inputs, project = local / "common", local
    else:
        inputs, project = work / "inputs", work / "MD"
    try:
        sysgen.generate_system(input_path=work / "ALA.pdb", config_path=work / "sys.config.yaml",
                               output_folder=inputs, echo=False)
        protocol = work / "md.config.yaml"
        document = yaml.safe_load(protocol.read_text())
        document["minimization"]["max_iterations"] = 25
        for key, value in list(document["equilibration"].items()):
            if key.endswith("_duration_ps") and value is not None:
                document["equilibration"][key] = 0.02
        document["cMD"].update({"duration_ns": 0.00004, "checkpoint_interval_ps": 0.02,
                                "whole_system_interval_ps": 0.02, "solute_interval_ps": 0.02})
        protocol.write_text(yaml.safe_dump(document, sort_keys=False))
        mdgen.generate_md(input_folder=inputs, config_path=protocol, output_folder=project)
    finally:
        os.environ.clear()
        os.environ.update(environment)
    return inputs, project


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


@pytest.mark.slow
@pytest.mark.parametrize("identity, label", [
    (dict(git_commit=CLEAN, dirty=False), "clean git checkout"),
    (dict(git_commit=None, dirty=None,
          direct_url={"url": "https://github.com/csy0000/MD-templates.git",
                      "vcs_info": {"vcs": "git", "commit_id": VCS}}), "direct_url.json"),
])
def test_the_same_commit_reaches_every_record(tmp_path, monkeypatch, identity, label):
    """Contract-managed generation writes ONE commit into every record that names a generator."""
    monkeypatch.setattr(provenance_min, "implementation_identity", lambda: _raw(**identity))
    expected = identity.get("git_commit") or VCS
    inputs, project = _generate(tmp_path, enabled=True, commit=expected, identity=identity)

    recorded = _recorded_commits(inputs, project)
    assert recorded, "nothing recorded a generator commit"
    assert set(recorded.values()) == {expected}, f"{label}: {recorded}"
    # The stage files are the ones that used to be null under direct_url.json.
    assert any(name.endswith("stage.yaml") for name in recorded)
    assert "dataset.yaml" in recorded


@pytest.mark.slow
def test_contract_generation_from_a_dirty_checkout_fails_before_the_system_is_built(
        tmp_path, monkeypatch):
    from md_templates.openmm import sysgen

    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=True))
    _sys_config(tmp_path, enabled=True, commit=CLEAN)
    root = tmp_path / "MD_DATA"
    local = root / "proj" / "2026-08" / "ala"
    local.mkdir(parents=True)
    monkeypatch.setenv("MD_DATA", str(root))
    monkeypatch.setenv("MD_DATA_LOCAL", str(local))

    with pytest.raises(Exception) as error:
        sysgen.generate_system(input_path=tmp_path / "ALA.pdb",
                               config_path=tmp_path / "sys.config.yaml",
                               output_folder=local / "common", echo=False)
    assert "UNCOMMITTED CHANGES" in str(error.value)
    # Nothing was built: the refusal is before construction, not after it.
    assert not (local / "common" / "system.xml").exists()
    assert not (local / "common" / "topology.pdb").exists()
    assert not (local / "dataset.yaml").exists()


@pytest.mark.slow
def test_unregistered_generation_from_a_dirty_checkout_is_permitted_and_says_so(
        tmp_path, monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=True))
    inputs, project = _generate(tmp_path, enabled=False, commit=CLEAN, identity={})

    assert (inputs / "system.xml").is_file(), "unregistered development must still work"
    for path in (inputs / "provenance.yaml", project / "provenance.yaml"):
        template = yaml.safe_load(path.read_text())["template"]
        assert template["commit"] == CLEAN
        assert template["dirty"] is True
        assert template["reproducible_from_commit"] is False
        assert "does NOT reproduce" in template["reproducibility"]
        assert template["installed_fingerprint"] == "f" * 64
    # And it is not presented as contract-verified.
    assert not (project / "dataset.yaml").exists()
    assert yaml.safe_load((project / "provenance.yaml").read_text())["dataset"][
        "contract_managed"] is False


# --- preflight requires every record ------------------------------------------------------------

def _contract_project(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance_min, "implementation_identity",
                        lambda: _raw(git_commit=CLEAN, dirty=False))
    return _generate(tmp_path, enabled=True, commit=CLEAN, identity={})


@pytest.mark.slow
def test_preflight_passes_when_every_record_agrees(tmp_path, monkeypatch):
    inputs, project = _contract_project(tmp_path, monkeypatch)
    preflight = template_module("preflight")
    row, = preflight.check_template_provenance(project / "minimization", project, inputs=inputs,
                                               contract_managed=True)
    assert row.status == preflight.PASS, row.detail
    assert CLEAN[:12] in row.detail


@pytest.mark.slow
@pytest.mark.parametrize("target, keys", [
    ("provenance.yaml", ("template", "commit")),
    ("md.config.yaml", ("provenance", "template_commit")),
    ("minimization/stage.yaml", ("template_commit",)),
])
def test_a_missing_provenance_record_fails_preflight(tmp_path, monkeypatch, target, keys):
    """Comparing only what is present meant deleting the field was enough to pass."""
    inputs, project = _contract_project(tmp_path, monkeypatch)
    preflight = template_module("preflight")
    path = project / target
    document = yaml.safe_load(path.read_text())
    node = document
    for key in keys[:-1]:
        node = node[key]
    node.pop(keys[-1])
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    row, = preflight.check_template_provenance(project / "minimization", project, inputs=inputs,
                                               contract_managed=True)
    assert row.status == preflight.FAIL, row.detail
    assert "must name the generator" in row.detail


@pytest.mark.slow
def test_a_missing_common_provenance_file_fails_preflight(tmp_path, monkeypatch):
    inputs, project = _contract_project(tmp_path, monkeypatch)
    preflight = template_module("preflight")
    (inputs / "provenance.yaml").unlink()
    row, = preflight.check_template_provenance(project / "minimization", project, inputs=inputs,
                                               contract_managed=True)
    assert row.status == preflight.FAIL
    assert "does not exist" in row.detail


@pytest.mark.slow
@pytest.mark.parametrize("target, keys", [
    ("dataset.yaml", ("templates", "commit")),
    ("provenance.yaml", ("template", "commit")),
    ("minimization/stage.yaml", ("template_commit",)),
])
def test_a_mismatching_provenance_record_fails_preflight(tmp_path, monkeypatch, target, keys):
    inputs, project = _contract_project(tmp_path, monkeypatch)
    preflight = template_module("preflight")
    path = project / target
    document = yaml.safe_load(path.read_text())
    node = document
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = OTHER
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    row, = preflight.check_template_provenance(project / "minimization", project, inputs=inputs,
                                               contract_managed=True)
    assert row.status == preflight.FAIL, row.detail
    assert "not generated by the same MD-templates" in row.detail


@pytest.mark.slow
def test_a_record_that_disagrees_with_itself_fails_preflight(tmp_path, monkeypatch):
    """`git_commit` and `direct_url.vcs_info.commit_id` in one file must not differ."""
    inputs, project = _contract_project(tmp_path, monkeypatch)
    preflight = template_module("preflight")
    path = project / "provenance.yaml"
    document = yaml.safe_load(path.read_text())
    document["implementation"]["git_commit"] = CLEAN
    document["implementation"]["direct_url"] = {"url": "https://example.invalid",
                                                "vcs_info": {"commit_id": OTHER}}
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    row, = preflight.check_template_provenance(project / "minimization", project, inputs=inputs,
                                               contract_managed=True)
    assert row.status == preflight.FAIL
    assert "disagree" in row.detail


def test_an_unregistered_project_is_skipped_not_failed(tmp_path):
    preflight = template_module("preflight")
    (tmp_path / "minimization").mkdir()
    row, = preflight.check_template_provenance(tmp_path / "minimization", tmp_path)
    assert row.status == preflight.SKIP
    assert "unregistered" in row.detail


# --- nothing derives the commit independently any more ------------------------------------------

def test_no_writer_reaches_for_git_commit_on_its_own():
    """One canonical resolution. Reaching past it is how the null-under-direct_url bug happened."""
    offenders = {}
    for name in ("sysgen.py", "mdgen.py"):
        path = REPO_ROOT / "src" / "md_templates" / "openmm" / name
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):          # an explanation of the bug is not the bug
                continue
            if '["git_commit"]' in line or "['git_commit']" in line:
                offenders[f"{name}:{number}"] = stripped
    assert not offenders, offenders
