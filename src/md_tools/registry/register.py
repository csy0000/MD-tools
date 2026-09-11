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

from ..data_contract.extension import (EXTENSION_NAME, check_dataset_extension,
                                       validate_extension_file)
from ..data_contract.model import (INVENTORY_NAME, MANIFEST_NAME, RESOLVED_NAME, Dataset,
                                   DatasetError, canonical_path, check_data_path,
                                   check_segment,
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


def _missing_source_reason(source: Path, destination: Path, relative: str) -> str:
    """Why `-idata` is not there -- which is not always "you typed the wrong path".

    Registration commits by renaming the staged tree into place, THEN removes the source, THEN
    replaces it with a symlink. Interrupted between the last two, the source is gone and the link
    does not exist yet, so a re-run sees exactly what a typo produces. The difference matters: in
    one case nothing has happened, in the other the dataset is registered and complete and the
    only thing missing is the convenience link. Saying "does not exist" for both sends someone
    looking for data that is already safely in the store.
    """
    if not (destination / MANIFEST_NAME).is_file():
        return f"-idata {source}: does not exist"

    state_path = destination.parent / f"{destination.name}{STATE_NAME}"
    interrupted = ""
    if state_path.is_file():
        try:
            stage = Transaction(state_path).stage
        except RegistrationError:
            stage = None
        if stage and stage != ORDER[-1]:
            interrupted = (f"\nAn interrupted registration is recorded at stage {stage!r}; the "
                           f"commit itself had already happened.")
    return (f"-idata {source}: does not exist, but {relative} IS registered at {destination}.\n"
            f"Registration removes the source and then replaces it with a symlink, so an "
            f"interruption between those two steps leaves exactly this state: the dataset is "
            f"committed and complete, and only the link back is missing.{interrupted}\n"
            f"Verify it with:\n"
            f"    md-openmm data-register -idata {destination} --verify-only\n"
            f"and recreate the link with:\n"
            f"    ln -s {destination} {source}")


def register_dataset(*, source: Path, project_name: str, data_name: str, year: str,
                     common: bool = False, dry_run: bool = False, verify_only: bool = False,
                     user_config: str | None = None, md_data_override: str | None = None,
                     project_repo: str | None = None, echo: bool = True) -> dict[str, Any]:
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
        # MAY be several segments: a reference set is browsable as
        # `<yyyy-mm>/<system>/<method>/run<n>` rather than one flattened name. Every component is
        # still validated individually -- see `check_data_path`.
        check_data_path("data_name", data_name)
        relative = canonical_path(year=year, project_name=project_name, data_name=data_name,
                                  common=common)
    except DatasetError as exc:
        raise RegistrationError(str(exc)) from None

    source = Path(source).expanduser()
    missing_source = not source.exists() and not source.is_symlink()
    # A missing source is reported below, once the destination is known. Registration removes the
    # source before it links it, so "-idata does not exist" is also what an interruption in that
    # window looks like -- and there the dataset IS registered, which the user needs to be told.
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
    if missing_source:
        raise RegistrationError(_missing_source_reason(source, destination, relative))

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
    # If this data continues something, the manifest has to SAY so: `derived_from` is where a
    # reader learns the provenance, and the extension record is cross-checked against it below.
    # Read here, before the manifest is derived; validated in full once the dataset exists.
    parent = _declared_parent(source)
    manifest = _manifest(source=source, relative=relative, year=year,
                         project_name=project_name, data_name=data_name, common=common,
                         records=found["records"], user=document["user"],
                         project_repo=project_repo,
                         derived_from=[parent] if parent else [])
    try:
        dataset = validate_dataset(manifest)
    except DatasetError as exc:
        raise RegistrationError(
            f"the manifest derived from this data violates the dataset contract:\n{exc}\n\n"
            f"Nothing has been moved or changed.") from None
    say(f"contract             : dataset.yaml validates against contract v2 "
        f"({len(dataset.components)} components)")

    # -- the extension record, if this data continues something ---------------------------
    extension = _validate_extension(source, dataset, say)

    resolved = {
        "format": "md-tools-resolved-provenance/2.0",
        "registered_utc": _now(),
        "source_recorded_as": str(source.name),
        "records": [{"log": entry["log"], "record": entry["record"]}
                    for entry in found["records"]],
        "lineage": notes,
        "inventory": entries,
        "extension": extension,
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


def _validate_extension(source: Path, dataset: Dataset, say) -> dict[str, Any] | None:
    """Validate `extension.yaml` if the data carries one, WITHOUT touching the parent.

    The parent of a continuation is named, never opened. Loading it would make registering this
    dataset depend on another dataset still being present and unchanged -- and a completed parent
    is immutable precisely so that nothing needs to reach into it.

    The checkpoint the continuation restarted from IS verified, because that is the join point:
    the record states its digest, and if the file is here it must still hash to that.
    """
    path = Path(source) / EXTENSION_NAME
    if not path.is_file():
        return None
    try:
        record = validate_extension_file(path)
    except DatasetError as exc:
        raise RegistrationError(
            f"this data carries an {EXTENSION_NAME} that violates the contract:\n{exc}\n\n"
            f"Nothing has been moved or changed.") from None

    problems = check_dataset_extension(dataset, record)
    if problems:
        raise RegistrationError(
            f"{EXTENSION_NAME} and {MANIFEST_NAME} disagree:\n"
            + "\n".join(f"  {problem}" for problem in problems)
            + "\n\nNothing has been moved or changed.")

    checkpoint = Path(source) / record.target.source_checkpoint
    if checkpoint.is_file():
        from ..build.record import sha256_file
        actual = sha256_file(checkpoint)
        if actual != record.target.source_checkpoint_sha256:
            raise RegistrationError(
                f"{EXTENSION_NAME} records the restart checkpoint "
                f"{record.target.source_checkpoint} as "
                f"{record.target.source_checkpoint_sha256[:12]}..., but the file present hashes "
                f"to {actual[:12]}.... The join point is not the one the record describes.")
        say(f"extension            : {record.mode}, restart verified against "
            f"{record.target.source_checkpoint}")
    else:
        say(f"extension            : {record.mode}, parent "
            f"{record.target.dataset_id!r} (checkpoint not carried here)")

    return {"mode": record.mode, "parent_dataset_id": record.target.dataset_id,
            "parent_component": record.target.component,
            "output_component": record.output_component,
            "restart_step": record.restart_step,
            "source_checkpoint": record.target.source_checkpoint,
            "source_checkpoint_sha256": record.target.source_checkpoint_sha256,
            "combined_length_ns": record.combined_length_ns,
            "status": record.status}


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


def _declared_parent(source: Path) -> str | None:
    """The parent dataset id an `extension.yaml` names, if there is one.

    Read without validating: the full contract check needs the derived dataset, and the dataset
    cannot be derived until `derived_from` is known. A malformed record is caught moments later by
    `_validate_extension`, which refuses the registration outright.
    """
    path = Path(source) / EXTENSION_NAME
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    target = document.get("target") or {}
    parent = target.get("dataset_id")
    return parent if isinstance(parent, str) and parent else None


def _manifest(*, source: Path, relative: str, year: str, project_name: str, data_name: str,
              project_repo: str | None = None,
              common: bool, records: list[dict[str, Any]], user: dict[str, Any],
              derived_from: list[str] | None = None) -> dict[str, Any]:
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
        "origin": _origin(source, project_repo=project_repo),
        "software": software,
        "components": components,
        "derived_from": list(derived_from or []),
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


def _origin(source: Path, *, project_repo: str | None = None) -> dict[str, Any]:
    """The project repository THIS DATA came from, pinned at an exact commit.

    The repository is resolved from the DATA, not from the shell's working directory.

    It used to be the shell's. Every git command ran with no `-C`, so the field meant "whichever
    repository the person happened to be standing in", and three reference datasets were
    registered claiming MD-tools' HEAD as the project that produced a campaign belonging to
    MD-project. The two guards around it -- refuse a dirty tree, refuse a non-https remote --
    both passed, which is what made the wrong answer convincing.

    Resolution is by `rev-parse --show-toplevel` from the data directory, which works whether or
    not the data are tracked: `data/**` is routinely gitignored and ignore status has nothing to
    do with which worktree a path is in.

    When the data are not inside a repository at all, this REFUSES and names `--project-repo`.
    The silent fall back to the cwd is precisely what produced the wrong field, so there is no
    fall back; an explicit answer can be checked, a guess cannot.
    """
    import subprocess

    def git_in(where, *args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(where), *args],
                                    capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    if project_repo is not None:
        start = Path(project_repo)
        if not start.is_dir():
            raise RegistrationError(f"--project-repo {start}: not a directory")
    else:
        start = Path(source)
        start = start if start.is_dir() else start.parent

    toplevel = git_in(start, "rev-parse", "--show-toplevel")
    if not toplevel:
        raise RegistrationError(
            f"{start} is not inside a git repository, so the project that produced this data "
            f"cannot be established. A dataset manifest pins the repository and commit that "
            f"produced it, and registering without one would record a provenance nobody can "
            f"check.\n\n"
            f"If the data legitimately live outside the project tree, name the project "
            f"explicitly:\n"
            f"    --project-repo /path/to/the/project")

    def git(*args: str) -> str | None:
        return git_in(toplevel, *args)

    commit = git("rev-parse", "HEAD")
    if not commit:
        raise RegistrationError(
            f"cannot establish the commit of {toplevel}: git reports no HEAD there. A dataset "
            f"manifest pins the repository and commit that produced it; registering without one "
            f"would record a provenance nobody can check.")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RegistrationError(
            f"the project working tree at {toplevel} has uncommitted changes, so the commit "
            f"recorded in the manifest would not describe what actually ran:\n"
            + "\n".join(f"  {line}" for line in dirty.splitlines()[:10])
            + "\n\nCommit or stash them, then register.")
    url = git("remote", "get-url", "origin") or ""
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    if url.endswith(".git"):
        url = url[: -len(".git")]
    if not url.startswith("https://"):
        raise RegistrationError(
            f"the origin remote of {toplevel} ({url or 'unset'}) is not an https URL, and the "
            f"manifest records one so that a reader can reach it.")
    return {"repository": url, "commit": commit, "version": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
