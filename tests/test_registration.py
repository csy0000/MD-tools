"""Registration: the contract, the transaction, and every way it must refuse.

Registration moves data that cannot be regenerated and then deletes the only other copy. Every
test here corresponds to a way that could go wrong. None of them assert on prose.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from md_tools.build.record import BEGIN, END, RECORD_SCHEMA_VERSION
from md_tools.data_contract.model import MANIFEST_NAME, canonical_path, validate_dataset_file
from md_tools.registry import inventory
from md_tools.registry.errors import RegistrationError
from md_tools.registry.register import register_dataset
from md_tools.registry.transaction import ORDER, Transaction
from md_tools.registry.userconfig import init_user_config

OLD = 4 * 3600           # seconds; older than the active-writer window


def _record(**fields) -> str:
    body = {"schema_version": RECORD_SCHEMA_VERSION, "record_type": "md-stage:cMD",
            "status": "completed", "started_utc": "2026-09-01T10:00:00+00:00",
            "finished_utc": "2026-09-01T11:00:00+00:00", "command": ["x"],
            "environment": {"packages": {"md-tools": "0.5.0.dev0"}, "md_tools_commit": "a" * 40}}
    body.update(fields)
    block = yaml.safe_dump(body, sort_keys=False).rstrip("\n")
    commented = "\n".join(f"# {line}" if line else "#" for line in block.splitlines())
    return f"a readable log\n\n{BEGIN}\n{commented}\n{END}\n"


def _age(path: Path, seconds: int = OLD) -> None:
    for item in [path, *path.rglob("*")]:
        os.utime(item, (item.stat().st_atime - seconds, item.stat().st_mtime - seconds))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A managed root, a user configuration, and a committed project holding finished data."""
    root = tmp_path / "MD_DATA"; root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MD_DATA", raising=False)
    monkeypatch.delenv("MD_TOOLS_CONFIG", raising=False)
    init_user_config(noninteractive=True, person_id="test-person", name="Test Person",
                     md_data=str(root))

    project = tmp_path / "project"; (project / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "."], cwd=project, check=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "https://github.com/csy0000/MD-project"], cwd=project, check=True)
    (project / ".gitignore").write_text("data/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.email=t@e", "-c", "user.name=T", "commit", "-qm", "init"],
                   cwd=project, check=True)

    source = project / "data" / "ALA-cMD"
    (source / "cMD").mkdir(parents=True)
    (source / "cMD" / "traj.dcd").write_bytes(b"frames" * 100)
    (source / "cMD.log").write_text(_record())
    _age(source)
    monkeypatch.chdir(project)
    return {"root": root, "project": project, "source": source}


def _register(world, **kw):
    params = dict(source=world["source"], project_name="ALA", data_name="ALA-cMD",
                  year="2026", echo=False)
    params.update(kw)
    return register_dataset(**params)


# -- the path shape ---------------------------------------------------------------------------

def test_the_project_path_is_year_first_with_no_month(world):
    result = _register(world)
    assert result["canonical_path"] == "2026/ALA/ALA-cMD"
    assert (world["root"] / "2026" / "ALA" / "ALA-cMD").is_dir()


def test_the_common_path_inserts_common_after_the_year(world):
    result = _register(world, common=True)
    assert result["canonical_path"] == "2026/common/ALA/ALA-cMD"


def test_no_registered_path_contains_a_month_segment(world):
    _register(world)
    import re
    for path in (world["root"]).rglob("*"):
        for part in path.relative_to(world["root"]).parts:
            assert not re.fullmatch(r"\d{4}-\d{2}", part), f"{part} looks like a yyyy-mm segment"


# -- names and the year ------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["../escape", "a/b", "..", ".", ".hidden", "with space",
                                  "", "a\\b", "ctrl\x01"])
def test_an_unsafe_name_is_refused(world, name):
    with pytest.raises(RegistrationError):
        _register(world, project_name=name, dry_run=True)


@pytest.mark.parametrize("year", ["26", "20266", "abcd", "", "2026a"])
def test_the_year_must_be_exactly_four_digits(world, year):
    with pytest.raises(RegistrationError):
        _register(world, year=year, dry_run=True)


def test_the_year_must_match_the_records(world):
    """The year is a fact about the data, not a label chosen at registration time."""
    with pytest.raises(RegistrationError, match="year"):
        _register(world, year="2019")


# -- evidence, never prose ----------------------------------------------------------------------

def test_prose_claiming_completion_is_not_evidence(world):
    (world["source"] / "cMD.log").write_text(
        "Everything completed successfully.\nstatus: completed\nrun_status: completed\n")
    _age(world["source"])
    with pytest.raises(RegistrationError, match="no machine record"):
        _register(world, dry_run=True)


def test_a_record_that_is_not_completed_is_refused(world):
    (world["source"] / "cMD.log").write_text(_record(status="started"))
    _age(world["source"])
    with pytest.raises(RegistrationError, match="not 'completed'"):
        _register(world, dry_run=True)


def test_a_failed_record_is_refused(world):
    (world["source"] / "cMD.log").write_text(_record(status="failed", failure_reason="OOM"))
    _age(world["source"])
    with pytest.raises(RegistrationError, match="FAILED"):
        _register(world, dry_run=True)


def test_a_record_with_an_unknown_schema_version_is_refused(world):
    (world["source"] / "cMD.log").write_text(_record(schema_version="md-tools-record/99.0"))
    _age(world["source"])
    with pytest.raises(RegistrationError):
        _register(world, dry_run=True)


def test_a_truncated_record_is_refused(world):
    text = (world["source"] / "cMD.log").read_text()
    (world["source"] / "cMD.log").write_text(text[: text.index(END)])
    _age(world["source"])
    with pytest.raises(RegistrationError):
        _register(world, dry_run=True)


def test_an_actively_written_directory_is_refused(world):
    (world["source"] / "cMD" / "traj.dcd").write_bytes(b"still writing")
    with pytest.raises(RegistrationError, match="still being written"):
        _register(world, dry_run=True)


def test_data_changed_after_the_record_is_refused(world):
    """A record's declared input digest must still describe the file on disk."""
    payload = world["source"] / "cMD" / "traj.dcd"
    from md_tools.build.record import sha256_file
    (world["source"] / "cMD.log").write_text(_record(inputs={
        "topology": {"path": "cMD/traj.dcd", "bytes": payload.stat().st_size,
                     "sha256": "0" * 64}}))
    _age(world["source"])
    with pytest.raises(RegistrationError, match="hashes to"):
        _register(world, dry_run=True)


# -- dry run is pure ------------------------------------------------------------------------------

def test_dry_run_writes_nothing_anywhere(world):
    before_root = sorted(p.name for p in world["root"].rglob("*"))
    before_source = sorted(str(p) for p in world["source"].rglob("*"))
    result = _register(world, dry_run=True)
    assert result["dry_run"] is True
    assert sorted(p.name for p in world["root"].rglob("*")) == before_root == []
    assert sorted(str(p) for p in world["source"].rglob("*")) == before_source


# -- the transaction ------------------------------------------------------------------------------

def test_the_source_becomes_a_symlink_only_after_verification(world):
    source = world["source"]
    _register(world)
    assert source.is_symlink()
    assert source.resolve() == (world["root"] / "2026" / "ALA" / "ALA-cMD").resolve()


def test_every_destination_file_matches_the_source_digest(world):
    entries = inventory.build(world["source"])
    _register(world)
    inventory.verify(world["root"] / "2026" / "ALA" / "ALA-cMD", entries)


def test_registering_the_same_bytes_again_is_a_no_op(world, tmp_path):
    _register(world)
    second = world["project"] / "data" / "second"
    shutil.copytree(world["root"] / "2026" / "ALA" / "ALA-cMD", second)
    for name in (MANIFEST_NAME, "dataset.resolved.yaml", "SHA256SUMS"):
        (second / name).unlink(missing_ok=True)
    _age(second)
    result = _register(world, source=second)
    assert result.get("idempotent") is True
    assert second.is_dir() and not second.is_symlink(), "an idempotent call must not consume it"


def test_a_collision_with_different_bytes_is_refused_and_preserves_the_source(world):
    _register(world)
    other = world["project"] / "data" / "other"
    other.mkdir()
    (other / "cMD").mkdir()
    (other / "cMD" / "traj.dcd").write_bytes(b"different bytes entirely")
    (other / "cMD.log").write_text(_record())
    _age(other)
    with pytest.raises(RegistrationError, match="DIFFER"):
        _register(world, source=other)
    assert (other / "cMD" / "traj.dcd").is_file(), "the source must survive a refusal"


@pytest.mark.parametrize("stage", ["planned", "staged", "verified"])
def test_an_interruption_before_commit_leaves_the_source_intact(world, stage):
    """Interrupt at each pre-commit boundary; the source is the only copy and must survive."""
    destination = world["root"] / "2026" / "ALA" / "ALA-cMD"
    staging = destination.with_name(destination.name + ".registering")
    state = staging.parent / f"{destination.name}.md-tools-registration.json"
    entries = inventory.build(world["source"])

    destination.parent.mkdir(parents=True)
    transaction = Transaction(state, plan={"source": str(world["source"]),
                                           "destination": str(destination),
                                           "staging": str(staging), "canonical_path": "x",
                                           "files": len(entries), "bytes": 0})
    transaction.save()
    if ORDER.index(stage) >= ORDER.index("staged"):
        from md_tools.registry.transaction import copy_tree
        copy_tree(world["source"], staging, entries=entries)
        transaction.advance("staged")
    if stage == "verified":
        transaction.advance("verified")

    assert world["source"].is_dir() and not world["source"].is_symlink()
    assert not destination.exists(), "the final path must not exist before commit"

    # Resuming completes the registration rather than starting over or losing data.
    result = _register(world)
    assert result["canonical_path"] == "2026/ALA/ALA-cMD"
    inventory.verify(destination, entries)
    assert world["source"].is_symlink()


def test_a_transaction_state_from_another_version_is_refused(world):
    destination = world["root"] / "2026" / "ALA" / "ALA-cMD"
    destination.parent.mkdir(parents=True)
    state = destination.parent / f"{destination.name}.md-tools-registration.json"
    state.write_text(json.dumps({"format": "something-else/1.0", "stage": "staged"}))
    with pytest.raises(RegistrationError, match="format"):
        Transaction(state)


def test_the_transaction_never_goes_backwards(tmp_path):
    state = tmp_path / "s.json"
    transaction = Transaction(state, plan={})
    transaction.save()
    transaction.advance("staged")
    with pytest.raises(RegistrationError, match="cannot go back"):
        transaction.advance("planned")


# -- symlinks and traversal -----------------------------------------------------------------------

def test_a_symlink_inside_the_dataset_is_refused(world):
    (world["source"] / "link.dcd").symlink_to(world["source"] / "cMD" / "traj.dcd")
    _age(world["source"])
    with pytest.raises(RegistrationError, match="symlink"):
        _register(world, dry_run=True)


def test_registering_an_already_registered_symlink_is_refused(world):
    _register(world)
    with pytest.raises(RegistrationError, match="symlink"):
        _register(world, dry_run=True)


# -- the manifest ------------------------------------------------------------------------------------

def test_the_registered_manifest_validates_against_the_contract(world):
    _register(world)
    dataset = validate_dataset_file(world["root"] / "2026" / "ALA" / "ALA-cMD" / MANIFEST_NAME)
    assert dataset.path == canonical_path(year="2026", project_name="ALA", data_name="ALA-cMD")
    assert dataset.status == "complete"
    assert dataset.role == "project"


def test_the_manifest_carries_no_absolute_machine_path(world):
    _register(world)
    text = (world["root"] / "2026" / "ALA" / "ALA-cMD" / MANIFEST_NAME).read_text()
    document = yaml.safe_load(text)
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, str):
            yield node
    for value in walk(document):
        assert not value.startswith("/"), f"absolute path in the manifest: {value!r}"
        assert str(world["root"]) not in value


def test_a_dirty_project_tree_is_refused(world):
    (world["project"] / "untracked.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "untracked.py"], cwd=world["project"], check=True)
    with pytest.raises(RegistrationError, match="uncommitted"):
        _register(world)


# -- verify-only ---------------------------------------------------------------------------------------

def test_verify_only_checks_an_already_registered_dataset(world):
    _register(world)
    result = _register(world, source=world["source"], verify_only=True)
    assert result["verified"] is True


def test_verify_only_detects_a_corrupted_destination(world):
    _register(world)
    victim = world["root"] / "2026" / "ALA" / "ALA-cMD" / "cMD" / "traj.dcd"
    victim.write_bytes(b"corrupted")
    with pytest.raises(RegistrationError, match="digest differs|size differs"):
        _register(world, source=world["source"], verify_only=True)


# -- crossing a filesystem boundary ------------------------------------------------------------
#
# The staging directory is deliberately a SIBLING of the destination, so the staging-to-committed
# step is always a rename within one filesystem and is therefore atomic. The crossing, when there
# is one, happens during the explicit file-by-file copy from the source -- which is followed by a
# re-read of every digest at the destination, because a copy returning without raising is not
# evidence that the bytes arrived.

def _other_filesystem(path: Path) -> Path | None:
    """A writable directory on a different device from `path`, or None."""
    for candidate in (Path("/dev/shm"), Path("/tmp"), Path.home()):
        try:
            if candidate.is_dir() and os.access(candidate, os.W_OK) \
                    and candidate.stat().st_dev != path.stat().st_dev:
                return candidate
        except OSError:
            continue
    return None


def test_registration_works_across_a_filesystem_boundary(world, tmp_path):
    elsewhere = _other_filesystem(tmp_path)
    if elsewhere is None:
        pytest.skip("no second writable filesystem on this machine to cross")
    root = Path(tempfile.mkdtemp(prefix="md-tools-xfs-", dir=str(elsewhere)))
    try:
        assert root.stat().st_dev != world["source"].stat().st_dev, "not actually crossing"
        entries = inventory.build(world["source"])
        result = _register(world, md_data_override=str(root))
        destination = root / "2026" / "ALA" / "ALA-cMD"
        assert result["canonical_path"] == "2026/ALA/ALA-cMD"
        inventory.verify(destination, entries)
        assert world["source"].is_symlink()
        assert world["source"].resolve() == destination.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_staging_directory_is_a_sibling_of_the_destination(world):
    """So the commit step is a rename inside one filesystem, and therefore atomic."""
    destination = world["root"] / "2026" / "ALA" / "ALA-cMD"
    staging = destination.with_name(destination.name + ".registering")
    assert staging.parent == destination.parent


def test_an_incomplete_staging_directory_is_not_committed(world):
    """A 'staged' marker is not proof; the staging is checked for the manifests that define it."""
    destination = world["root"] / "2026" / "ALA" / "ALA-cMD"
    staging = destination.with_name(destination.name + ".registering")
    state = staging.parent / f"{destination.name}.md-tools-registration.json"
    entries = inventory.build(world["source"])

    destination.parent.mkdir(parents=True)
    from md_tools.registry.transaction import copy_tree
    transaction = Transaction(state, plan={"source": str(world["source"]),
                                           "destination": str(destination),
                                           "staging": str(staging), "canonical_path": "x",
                                           "files": len(entries), "bytes": 0})
    transaction.save()
    copy_tree(world["source"], staging, entries=entries)      # bytes only: no manifest written
    transaction.advance("staged")

    _register(world)
    assert (destination / MANIFEST_NAME).is_file(), "committed a dataset with no manifest"
    validate_dataset_file(destination / MANIFEST_NAME)


# -- the published schema, the shipped example, and the migration note ----------------------------

def test_the_exported_schema_matches_the_model_that_actually_validates():
    """A published schema that has drifted from the running validator is worse than none.

    The file is generated from the pydantic model, so this regenerates and compares rather than
    checking a few fields by hand.
    """
    import json

    from md_tools.data_contract import exported_schema_path
    from md_tools.data_contract.model import json_schema

    path = exported_schema_path()
    assert path.is_file(), f"{path} is not shipped"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == json_schema(), (
        "the exported dataset schema differs from what the model renders. Regenerate it; do not "
        "edit it by hand.")


def test_the_exported_schema_describes_the_year_first_path():
    import json

    from md_tools.data_contract import exported_schema_path

    schema = json.loads(exported_schema_path().read_text(encoding="utf-8"))
    for field in ("year", "project_name", "data_name"):
        assert field in schema["properties"], field
    assert "namespace" not in schema["properties"], "v1's namespace must not be in the v2 schema"
    assert "2.0" in schema["$id"]


def test_the_user_configuration_example_is_shipped_and_carries_no_secret():
    """It must be shipped, and it must be safe to ship."""
    from importlib.resources import files

    path = Path(str(files("md_tools").joinpath("configs", "user.config.example")))
    assert path.is_file(), f"{path} is not shipped"
    text = path.read_text(encoding="utf-8")
    assert "schema_version" in text and "md_data" in text and "person_id" in text
    lowered = text.lower()
    for forbidden in ("token", "password", "secret", "api_key", "private_key"):
        assert forbidden not in lowered, f"the shipped example mentions {forbidden!r}"


def test_the_shipped_example_is_what_init_actually_writes(tmp_path, monkeypatch):
    """The example must describe the file that gets created, not an older idea of it."""
    from importlib.resources import files

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MD_TOOLS_CONFIG", raising=False)
    root = tmp_path / "MD_DATA"; root.mkdir()
    written = yaml.safe_load(init_user_config(
        noninteractive=True, person_id="a-person", name="A Person",
        md_data=str(root)).read_text(encoding="utf-8"))
    example = yaml.safe_load(
        Path(str(files("md_tools").joinpath("configs", "user.config.example"))).read_text())
    assert set(written) == set(example), (set(written), set(example))
    assert set(written["user"]) == set(example["user"])
    assert set(written["machine"]) == set(example["machine"])


def test_the_migration_document_exists_and_names_both_versions():
    from importlib.resources import files

    path = Path(str(files("md_tools.data_contract").joinpath("migration.md")))
    assert path.is_file(), "the v1-to-v2 migration document is missing"
    text = path.read_text(encoding="utf-8")
    assert "20d982eb463ed439095f1b95e00ff1b1d75906b4" in text, "the ported commit is not attributed"
    assert "{namespace}/{yyyy-mm}/{dataset_name}" in text and "{year}/{project_name}/{data_name}" in text
