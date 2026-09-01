"""`md-openmm data-register` -- verify finished data, then move it into managed storage.

The fifteen steps, in the order they must happen, and what each one refuses:

  1  resolve and validate the managed storage root
  2  resolve -idata without letting an untrusted name build the destination
  3  discover structured records recursively; treat free text as display only
  4  reject missing, malformed, failed, incomplete or actively-written records
  5  validate lineage and exact input/output digests
  6  inventory every file: relative path, size, SHA-256
  7  derive the dataset manifest and the resolved provenance manifest
  8  compute the year-first destination, with no month segment
  9  refuse a collision unless the existing destination is provably identical
 10  stage transactionally, beside the destination, working across filesystems
 11  reopen the destination and verify its inventory
 12  remove the source ONLY after the destination verifies
 13  replace the source with a symlink to the verified destination
 14  record durable transaction state, so interruption resumes safely and idempotently
 15  validate once more and print the canonical relative destination

If any check fails the source is preserved, unchanged. Two datasets are never merged in place,
completed data are never overwritten, a success line in a log is never trusted, and
`shutil.move` is never treated as proof that a cross-filesystem transfer arrived intact.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..data_contract.model import (INVENTORY_NAME, MANIFEST_NAME, RESOLVED_NAME, Dataset,
                                   DatasetError, canonical_path, check_segment,
                                   validate_dataset)
from . import discovery, inventory
from .errors import RegistrationError
from .transaction import ORDER, Transaction, copy_tree
from .userconfig import load_user_config, resolve_md_data

STATE_NAME = ".md-tools-registration.json"

#: Which component type and method a directory name implies. A directory whose name is not here
#: is still registered, as a `reference` component with no method claim -- the alternative is to
#: invent a method, and an invented method in a manifest is worse than an honest absence.
COMPONENT_KINDS: dict[str, tuple[str, str | None, str]] = {
    "input": ("shared-input", None, "Prepared system and topology shared by every method."),
    "common": ("shared-input", None, "Prepared system and topology shared by every method."),
    "min": ("simulation", "minimization", "Restrained energy minimisation."),
    "eq": ("simulation", "equilibration", "The equilibration chain before production."),
    "md_script": ("shared-input", None, "The generated run scripts and resolved configuration."),
    "cMD": ("simulation", "cMD", "Production molecular dynamics."),
    "REST2": ("simulation", "REST2", "Replica-exchange solute tempering ladder."),
    "rREST2": ("simulation", "rREST2", "Reservoir REST2 ladder."),
    "analysis": ("analysis", None, "Analysis derived from this dataset's own simulation output."),
}


def register_dataset(*, source: Path, project_name: str, data_name: str, year: str,
                     common: bool = False, dry_run: bool = False, verify_only: bool = False,
                     user_config: str | None = None, md_data_override: str | None = None,
                     echo: bool = True) -> dict[str, Any]:
    def say(message: str = "") -> None:
        if echo:
            print(message)

    # -- 1, 2: the root and the names, before anything touches the filesystem ---------------
    document, config_file, config_origin = load_user_config(user_config)
    root, root_origin = resolve_md_data(document, override=md_data_override)
    say(f"managed storage root : {root}   (from {root_origin})")
    say(f"user configuration   : {config_file}   (from {config_origin})")

    # `check_segment` and `canonical_path` speak the contract's error type. Registration presents
    # ONE refusal type, so the CLI's handling stays honest: anything else reaching the top is a
    # bug rather than a refusal, and should not be reported as if the user had mistyped something.
    try:
        check_segment("project_name", project_name)
        check_segment("data_name", data_name)
        relative = canonical_path(year=year, project_name=project_name, data_name=data_name,
                                  common=common)
    except DatasetError as exc:
        raise RegistrationError(str(exc)) from None

    source = Path(source).expanduser()
    if not source.exists() and not source.is_symlink():
        raise RegistrationError(f"-idata {source}: does not exist")
    # Checked BEFORE resolving. A registered source has been replaced by a symlink to its
    # destination, so resolving first would turn "you already registered this" into the far less
    # helpful "that would copy a directory into itself".
    if source.is_symlink() and not verify_only:
        raise RegistrationError(
            f"-idata {source} is a symlink to {os.readlink(source)}. It has already been "
            f"registered -- registration replaces the source with a link to its destination. "
            f"There is nothing further to do.")
    source = source.resolve()
    destination = (root / relative).resolve()
    # The destination is built from validated segments joined to a validated root, and then
    # checked to be inside it. Building a path from an untrusted name and trusting the result is
    # how `..` in a name escapes the root.
    if root not in destination.parents and destination != root:
        raise RegistrationError(f"the computed destination {destination} is not inside {root}")
    say(f"canonical path       : {relative}")
    say(f"destination          : {destination}")

    # --verify-only reads the destination and nothing else, so the source-versus-destination
    # checks below do not apply to it: pointing -idata at the registered symlink is the normal
    # way to name which dataset to verify.
    if verify_only:
        return _verify_only(destination, relative, say)

    if source == destination or source in destination.parents:
        raise RegistrationError(f"-idata {source} contains or equals the destination "
                                f"{destination}; that would copy a directory into itself")

    # -- 3, 4: records, and only records ----------------------------------------------------
    found = discovery.classify(source)
    say(f"machine records      : {len(found['records'])} accepted, all reporting completion")

    # -- 5: lineage --------------------------------------------------------------------------
    notes = discovery.check_lineage(found["records"], source)
    say(f"lineage              : {len(notes)} stage handoff(s) verified by digest")

    # -- 6: inventory -------------------------------------------------------------------------
    entries = inventory.build(source)
    total = sum(entry["bytes"] for entry in entries)
    say(f"inventory            : {len(entries)} files, {total / 1e6:.1f} MB, all hashed")

    # -- 7: the manifests ----------------------------------------------------------------------
    manifest = _manifest(source=source, relative=relative, year=year,
                         project_name=project_name, data_name=data_name, common=common,
                         records=found["records"], user=document["user"])
    try:
        dataset = validate_dataset(manifest)
    except DatasetError as exc:
        raise RegistrationError(
            f"the manifest derived from this data violates the dataset contract:\n{exc}\n\n"
            f"Nothing has been moved or changed.") from None
    say(f"contract             : dataset.yaml validates against contract v2 "
        f"({len(dataset.components)} components)")

    resolved = {
        "format": "md-tools-resolved-provenance/2.0",
        "registered_utc": _now(),
        "source_recorded_as": str(source.name),
        "records": [{"log": entry["log"], "record": entry["record"]}
                    for entry in found["records"]],
        "lineage": notes,
        "inventory": entries,
    }

    if dry_run:
        say("")
        say("--dry-run: every check above passed. NOTHING was written, moved or removed.")
        return {"canonical_path": relative, "destination": str(destination),
                "dry_run": True, "files": len(entries), "bytes": total}

    # -- 8, 9: collision ------------------------------------------------------------------------
    staging = destination.with_name(destination.name + ".registering")
    state_path = staging.parent / f"{destination.name}{STATE_NAME}"
    existing = Transaction.find(state_path)

    if destination.exists():
        if existing is not None and existing.stage in ("committed", "source-removed", "linked",
                                                       "complete"):
            say("an earlier registration of this dataset reached "
                f"{existing.stage!r}; finishing it rather than starting again")
            return _resume(existing, source=source, destination=destination, relative=relative,
                           entries=entries, say=say)
        _refuse_collision(destination, entries, manifest, say)
        say("the existing destination is byte-identical and contract-identical; "
            "re-registration is a no-op")
        return {"canonical_path": relative, "destination": str(destination),
                "idempotent": True, "files": len(entries), "bytes": total}

    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction = existing or Transaction(state_path, plan={
        "source": str(source), "destination": str(destination), "staging": str(staging),
        "canonical_path": relative, "files": len(entries), "bytes": total})
    if existing is None:
        transaction.save()

    # -- 10: stage ------------------------------------------------------------------------------
    #
    # A transaction that claims "staged" is not taken at its word: the staging directory is
    # checked for the manifests that make it a dataset. A process killed between writing the
    # bytes and writing the manifest can leave the marker ahead of the reality, and committing
    # that would publish a dataset with no dataset.yaml.
    if (ORDER.index(transaction.stage) >= ORDER.index("staged")
            and not _staging_is_complete(staging)):
        say("the recorded stage claims the data were staged, but the staging directory is "
            "incomplete; staging again from the source")
        transaction.state["stage"] = "planned"
        transaction.save()

    if ORDER.index(transaction.stage) < ORDER.index("staged"):
        if staging.exists():
            shutil.rmtree(staging)
        copy_tree(source, staging, entries=entries)
        inventory.write(staging, entries)
        (staging / MANIFEST_NAME).write_text(
            yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False,
                           allow_unicode=True), encoding="utf-8")
        (staging / RESOLVED_NAME).write_text(
            yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False,
                           allow_unicode=True), encoding="utf-8")
        transaction.advance("staged")
        say(f"staged               : {staging}")

    # -- 11: verify the staged bytes, by re-reading them ----------------------------------------
    if ORDER.index(transaction.stage) < ORDER.index("verified"):
        inventory.verify(staging, entries)
        transaction.advance("verified")
        say("verified             : every staged file re-read, all digests match")

    # -- 12 (commit first): the atomic rename ---------------------------------------------------
    if ORDER.index(transaction.stage) < ORDER.index("committed"):
        os.replace(staging, destination)
        transaction.advance("committed")
        say(f"committed            : {destination}")

    return _resume(transaction, source=source, destination=destination, relative=relative,
                   entries=entries, say=say)


def _staging_is_complete(staging: Path) -> bool:
    """Does the staging directory hold everything a committed dataset must have?"""
    staging = Path(staging)
    return (staging.is_dir() and (staging / MANIFEST_NAME).is_file()
            and (staging / RESOLVED_NAME).is_file() and (staging / INVENTORY_NAME).is_file())


def _resume(transaction: Transaction, *, source: Path, destination: Path, relative: str,
            entries: list[dict[str, Any]], say) -> dict[str, Any]:
    """Finish a transaction from whatever stage it reached. Safe to call repeatedly."""
    # Re-verify at the FINAL path, not just at the staging path. The rename is atomic, but the
    # claim being made is about the destination, and it costs one pass to make it about the
    # thing it names.
    if ORDER.index(transaction.stage) < ORDER.index("source-removed"):
        inventory.verify(destination, entries)
        say("re-verified          : the committed destination matches the inventory")
        if Path(source).exists() and not Path(source).is_symlink():
            shutil.rmtree(source)
        transaction.advance("source-removed")
        say(f"source removed       : {source}   (only after the destination verified)")

    if ORDER.index(transaction.stage) < ORDER.index("linked"):
        link = Path(source)
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(_relative_link(link, destination), target_is_directory=True)
        transaction.advance("linked")
        say(f"linked               : {link} -> {destination}")

    if ORDER.index(transaction.stage) < ORDER.index("complete"):
        from ..data_contract.model import validate_dataset_file
        validate_dataset_file(destination / MANIFEST_NAME)
        transaction.advance("complete")
        say("validated            : the registered manifest passes the contract")

    try:
        Path(transaction.path).unlink()
    except OSError:
        pass
    say("")
    return {"canonical_path": relative, "destination": str(destination),
            "files": len(entries), "bytes": sum(e["bytes"] for e in entries)}


def _relative_link(link: Path, destination: Path) -> str:
    """A relative symlink where one is possible, so a moved project still resolves."""
    try:
        return os.path.relpath(destination, link.parent)
    except ValueError:                                    # different drives on Windows
        return str(destination)


def _refuse_collision(destination: Path, entries: list[dict[str, Any]],
                      manifest: dict[str, Any], say) -> None:
    """A destination that already exists is refused unless it is provably the same dataset."""
    existing_manifest = destination / MANIFEST_NAME
    if not existing_manifest.is_file():
        raise RegistrationError(
            f"{destination} already exists but holds no {MANIFEST_NAME}. Refusing to write into "
            f"it: registration never merges two datasets in place, and this directory's contents "
            f"are unexplained.")
    try:
        inventory.verify(destination, entries)
    except RegistrationError as exc:
        raise RegistrationError(
            f"{destination} already exists and its contents DIFFER from what would be "
            f"registered:\n{exc}\n\n"
            f"Completed data are immutable. Choose a different -data_name, or remove the "
            f"existing dataset deliberately. Nothing has been changed.") from None
    try:
        existing = yaml.safe_load(existing_manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistrationError(f"{existing_manifest}: not valid YAML -- {exc}") from None
    ignore = {"created_at", "completed_at"}
    a = {k: v for k, v in (existing or {}).items() if k not in ignore}
    b = {k: v for k, v in manifest.items() if k not in ignore}
    if a != b:
        differing = sorted({k for k in set(a) | set(b) if a.get(k) != b.get(k)})
        raise RegistrationError(
            f"{destination} already exists with byte-identical data but a DIFFERENT manifest "
            f"(fields: {differing}). Refusing to overwrite it.")


def _verify_only(destination: Path, relative: str, say) -> dict[str, Any]:
    from ..data_contract.model import validate_dataset_file

    if not destination.is_dir():
        raise RegistrationError(f"--verify-only: {destination} does not exist")
    manifest = destination / MANIFEST_NAME
    validate_dataset_file(manifest)
    listing = destination / INVENTORY_NAME
    if not listing.is_file():
        raise RegistrationError(f"--verify-only: {destination} has no {INVENTORY_NAME}")
    entries = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        full = destination / path
        entries.append({"path": path, "sha256": digest,
                        "bytes": full.stat().st_size if full.is_file() else -1})
    inventory.verify(destination, entries)
    say(f"verified             : {len(entries)} files, manifest valid, all digests match")
    return {"canonical_path": relative, "destination": str(destination),
            "verified": True, "files": len(entries)}


def _manifest(*, source: Path, relative: str, year: str, project_name: str, data_name: str,
              common: bool, records: list[dict[str, Any]], user: dict[str, Any]) -> dict[str, Any]:
    """Derive `dataset.yaml` from the records, never from the directory looking finished."""
    times = [entry["record"].get("finished_utc") or entry["record"].get("started_utc")
             for entry in records]
    times = sorted(t for t in times if t)
    started = sorted(entry["record"].get("started_utc") for entry in records
                     if entry["record"].get("started_utc"))
    created_at = started[0] if started else times[0]
    completed_at = times[-1]

    components = []
    for child in sorted(Path(source).iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
            continue
        kind, method, description = COMPONENT_KINDS.get(
            child.name, ("reference", None,
                         "Directory registered as part of this dataset. No method is claimed for "
                         "it, because none could be established from the records."))
        components.append({"name": child.name, "type": kind, "path": child.name,
                           "status": "complete", "linked": False, "method": method,
                           "description": description})
    if not components:
        # A flat dataset: the files are the dataset. One component, honestly described.
        components.append({
            "name": data_name, "type": "simulation",
            "method": _method_from(records) or "cMD", "path": ".", "status": "complete",
            "linked": False,
            "description": "The dataset's own simulation output, written directly at its root."})

    software = None
    for entry in records:
        environment = entry["record"].get("environment") or {}
        commit = environment.get("md_tools_commit")
        if commit:
            software = {"repository": "https://github.com/csy0000/MD-tools", "commit": commit,
                        "version": (environment.get("packages") or {}).get("md-tools")}
            break

    return {
        "schema_version": "2.0",
        "dataset_id": _dataset_id(project_name, data_name),
        "path": relative,
        "year": year,
        "project_name": project_name,
        "data_name": data_name,
        "role": "common" if common else "project",
        "system": _system_description(records, data_name),
        "created_at": created_at,
        "created_by": {"person_id": user["person_id"], "name": user["name"],
                       "orcid": user.get("orcid"), "affiliation": user.get("affiliation")},
        "status": "complete",
        "completed_at": completed_at,
        "origin": _origin(),
        "software": software,
        "components": components,
        "derived_from": [],
        "notes": None,
    }


def _dataset_id(project_name: str, data_name: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", f"{project_name}-{data_name}".lower()).strip("-")
    return slug or "dataset"


def _method_from(records: list[dict[str, Any]]) -> str | None:
    for entry in records:
        kind = str(entry["record"].get("record_type", ""))
        if kind.startswith("md-replica:"):
            return kind.split(":", 1)[1]
    for entry in records:
        kind = str(entry["record"].get("record_type", ""))
        if kind == "md-stage:cMD":
            return "cMD"
    return None


def _system_description(records: list[dict[str, Any]], data_name: str) -> str:
    for entry in records:
        record = entry["record"]
        if record.get("record_type") == "build-top":
            counts = record.get("counts") or {}
            solvent = record.get("solvent") or {}
            return (f"{counts.get('atoms', '?')} particles, "
                    f"{solvent.get('treatment', 'unknown')} {solvent.get('model', '')}, "
                    f"{counts.get('solute_atoms', '?')} solute atoms. "
                    f"Registered as {data_name}.")
    return f"Registered as {data_name}. No build record was present to describe the system."


def _origin() -> dict[str, Any]:
    """The project repository this data came from, pinned at an exact commit.

    Refused rather than guessed when the working tree is dirty: a commit that does not describe
    what actually ran is a false provenance claim, which is worse than none.
    """
    import subprocess

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit:
        raise RegistrationError(
            "cannot establish the project commit: this is not a git repository, or git is "
            "unavailable. A dataset manifest pins the repository and commit that produced it; "
            "registering without one would record a provenance nobody can check.")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RegistrationError(
            "the project working tree has uncommitted changes, so the commit recorded in the "
            "manifest would not describe what actually ran:\n"
            + "\n".join(f"  {line}" for line in dirty.splitlines()[:10])
            + "\n\nCommit or stash them, then register.")
    url = git("remote", "get-url", "origin") or ""
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    if url.endswith(".git"):
        url = url[: -len(".git")]
    if not url.startswith("https://"):
        raise RegistrationError(
            f"the project's origin remote ({url or 'unset'}) is not an https URL, and the "
            f"manifest records one so that a reader can reach it.")
    return {"repository": url, "commit": commit, "version": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
