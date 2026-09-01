"""Write and validate a dataset manifest that MD-data's own validator accepts.

MD-data owns the contract, the schema, the identity rules, the lifecycle, the catalogue, aliases,
extensions and archival policy (`csy0000/MD-data`, `docs/contracts/dataset-v1.md`). This module
owns none of that. It does exactly two things:

1. assembles a `dataset.yaml` from the user's editable `dataset:` block plus the components this
   generation actually produced, and
2. hands it to `md_data.validate_dataset` -- the published validator, imported, never vendored.

There is deliberately no copy of MD-data's Pydantic models or JSON schema here. A second
implementation of a contract is a contract that drifts, and the failure would be silent: this
repository would keep writing manifests that its own copy accepts and MD-data rejects.

**The boundary.** MD-tools may generate simulation files and a contract-valid manifest inside
the ONE explicitly selected active dataset. It does not register, copy, move, archive, delete or
catalogue anything, does not walk `$MD_DATA`, does not open another dataset, and does not hash a
production trajectory.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST_NAME = "dataset.yaml"
SCHEMA_VERSION = "1.0"
#: The prepared-system component every generated dataset carries. `sys-gen` writes into it and
#: every method component reads from it.
COMMON_COMPONENT = "common"
#: An exact 40-hex commit. A branch, a tag or an abbreviated SHA is not a pin.
COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: `{namespace}/{yyyy-mm}/{dataset_name}` -- the canonical path, checked here before MD-data sees
#: it so the failure names the dotted field a user can fix rather than a schema location.
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class ContractError(ValueError):
    """This generation cannot produce a contract-valid dataset, with the reason and the field."""


def md_data_identity() -> dict[str, Any]:
    """Which validator is in use, or why there is none. Recorded in provenance either way."""
    try:
        import md_data
    except ImportError as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}",
                "package": "md-data", "version": None, "contract_version": None}
    return {
        "available": True,
        "package": "md-data",
        "version": getattr(md_data, "__version__", "unknown"),
        "contract_version": getattr(md_data, "CONTRACT_VERSION", "unknown"),
        "supported_schema_versions": list(getattr(md_data, "SUPPORTED_SCHEMA_VERSIONS", ())),
        "module": getattr(md_data, "__file__", None),
    }


#: The ONE place the MD-data compatibility identity is maintained. Everything that installs,
#: documents or reports the dependency reads it from here rather than repeating a URL.
#:
#: HTTPS and an exact 40-hex commit, never SSH and never a branch. `git+ssh://.../dev` needs a key
#: agent and moves under the user's feet: two people following the same documented command on the
#: same day can end up validating against different contracts, which is precisely the drift this
#: whole arrangement exists to prevent.
MD_DATA_REPOSITORY = "https://github.com/csy0000/MD-data"
MD_DATA_COMMIT = "48628f9a5d3ace6c6398a63bc3905cd58d542de3"
MD_DATA_CONTRACT_VERSION = "1.0"


def md_data_requirement() -> str:
    """The pip requirement specifier for the compatible validator."""
    return f"md-data @ git+{MD_DATA_REPOSITORY}.git@{MD_DATA_COMMIT}"


def md_data_install_hint() -> str:
    return f"      pip install '{md_data_requirement()}'"


def require_md_data():
    """The published validator, or a refusal that says how to install it."""
    try:
        import md_data
    except ImportError as error:
        raise ContractError(
            "dataset.enabled is true, but the `md-data` package is not installed, so this "
            "manifest cannot be validated against the contract that owns it.\n"
            f"  ({type(error).__name__}: {error})\n"
            "  Install the pinned compatible validator into this environment:\n"
            f"{md_data_install_hint()}\n"
            "  This package deliberately carries no copy of MD-data's schema: a second "
            "implementation of a contract is one that drifts, silently.") from None
    return md_data


# ---------------------------------------------------------------------------------------------
# Which MD-tools generated this
# ---------------------------------------------------------------------------------------------

def generator_commit() -> dict[str, Any]:
    """The canonical template identity, as this module's callers expect it.

    A thin adaptor over `provenance_min.template_identity()` so there is exactly ONE resolution of
    "which MD-tools is this" in the package. Kept as a name because the contract code and its
    tests speak in terms of the generating commit.
    """
    from .provenance_min import template_identity

    identity = template_identity()
    return {"commit": identity["commit"], "route": identity["evidence"],
            "dirty": identity["dirty"], "detail": identity["detail"], "identity": identity}


def check_templates_commit(claimed: str) -> dict[str, Any]:
    """`dataset.templates.commit` must be the commit that is actually generating this dataset.

    Three refusals, and a dirty checkout is one of them. A working tree with uncommitted changes is
    NOT reproducible from its HEAD: someone who checks that commit out gets different code. For
    unregistered local work that is a limitation worth recording; for a dataset that will be
    registered, archived and cited it is a FALSE provenance rather than an imprecise one, so
    contract-managed generation stops here -- before the System is built, so the refusal is cheap.
    """
    from .defaults import MD_TOOLS_REPOSITORY

    established = generator_commit()
    if established["commit"] is None:
        raise ContractError(
            f"dataset.templates.commit is {claimed!r}, but this MD-tools cannot establish its "
            f"own exact commit, so the claim cannot be verified.\n"
            f"  {established['detail']}.\n"
            f"  Contract-managed generation refuses rather than record an unverified pin. Run "
            f"from a clean Git checkout, or install with "
            f"`pip install 'md-tools @ git+{MD_TOOLS_REPOSITORY}.git@<commit>'`, or set "
            f"dataset.enabled false for an unregistered local project.")
    if established["dirty"]:
        raise ContractError(
            f"the MD-tools generating this dataset is a Git checkout at "
            f"{established['commit'][:12]} with UNCOMMITTED CHANGES, so that commit does not "
            f"describe the code that would run.\n"
            f"  A registered dataset records templates.commit as the way to reproduce it. "
            f"Checking out {established['commit'][:12]} would give someone different code than "
            f"this ran, which is a false provenance rather than an imprecise one.\n"
            f"  Commit or stash the changes, or set dataset.enabled false for unregistered local "
            f"generation -- which records git_dirty and states plainly that the commit alone does "
            f"not reproduce it.")
    if str(claimed).strip().lower() != established["commit"].lower():
        raise ContractError(
            f"dataset.templates.commit is {claimed!r}, but the MD-tools actually generating "
            f"this dataset is at {established['commit']!r} ({established['route']}: "
            f"{established['detail']}).\n"
            f"  A 40-hex string that is not the generating commit is worse than none: it records "
            f"a provenance that can be checked out and will not reproduce this run.")
    return established


# ---------------------------------------------------------------------------------------------
# Where the dataset is
# ---------------------------------------------------------------------------------------------

def resolve_roots(output_folder: Path, *, environ: Optional[dict] = None,
                  component: Optional[str] = COMMON_COMPONENT) -> dict[str, Any]:
    """`MD_DATA`, `MD_DATA_LOCAL`, and the canonical path of the dataset being written into.

    `output_folder` is where this command was told to write, and `component` says what that folder
    IS: a named component inside the dataset (`sys-gen -of "$MD_DATA_LOCAL/common/"`), or `None`
    for the dataset root itself (`md-gen -of "$MD_DATA_LOCAL/"`). Either way the DATASET ROOT is
    what must equal `MD_DATA_LOCAL`, and `MD_DATA_LOCAL` must sit inside `MD_DATA` at exactly
    `{namespace}/{yyyy-mm}/{dataset_name}`.

    Nothing is walked. Two environment variables are read and three paths are compared.
    """
    environ = os.environ if environ is None else environ
    output = Path(output_folder).resolve()
    dataset_root = output.parent if component else output

    raw_root = environ.get("MD_DATA")
    if not raw_root:
        raise ContractError(
            "dataset.enabled is true, but MD_DATA is not set. It is the managed-storage root that "
            "every canonical dataset path is relative to.\n"
            "      export MD_DATA=/managed/storage")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise ContractError(f"MD_DATA={raw_root!r} is not an existing directory.")
    root = root.resolve()

    raw_local = environ.get("MD_DATA_LOCAL")
    if not raw_local:
        raise ContractError(
            "dataset.enabled is true, but MD_DATA_LOCAL is not set. It is the ONE dataset root "
            "this generation may write into.\n"
            '      export MD_DATA_LOCAL="$MD_DATA/{namespace}/{yyyy-mm}/{dataset_name}"')
    local = Path(raw_local).expanduser().resolve()

    try:
        relative = local.relative_to(root)
    except ValueError:
        raise ContractError(
            f"MD_DATA_LOCAL={str(local)!r} is not inside MD_DATA={str(root)!r}.\n"
            "  Output must stay inside the explicitly supplied storage root.") from None
    if local.is_symlink() or Path(raw_local).expanduser().is_symlink():
        raise ContractError(
            f"MD_DATA_LOCAL={raw_local!r} is a symlink. A symlink into another dataset is an "
            "ALIAS, and an alias is read-only: it resolves to somebody else's authoritative "
            "dataset.yaml. New output belongs in a dataset this project owns.")

    parts = relative.parts
    if len(parts) != 3:
        raise ContractError(
            f"MD_DATA_LOCAL resolves to {relative.as_posix()!r} under MD_DATA, but the canonical "
            "path is exactly {namespace}/{yyyy-mm}/{dataset_name} -- three segments.\n"
            '      export MD_DATA_LOCAL="$MD_DATA/my-project/2026-08/ALA-explicit"')
    if not MONTH.match(parts[1]):
        raise ContractError(
            f"the middle segment of the canonical path is {parts[1]!r}, which is not a real "
            "month. It is the creation month of this dataset version, as yyyy-mm.")

    if dataset_root != local:
        where = "the parent of the output folder" if component else "the output folder"
        raise ContractError(
            f"{where} is {str(dataset_root)!r}, but MD_DATA_LOCAL is {str(local)!r}.\n"
            f"  A contract-managed generation writes into exactly one dataset root. For sys-gen "
            f"that means `-of \"$MD_DATA_LOCAL/{COMMON_COMPONENT}/\"`; for md-gen, "
            f"`-of \"$MD_DATA_LOCAL/\"`.")

    return {
        "md_data": str(root),
        "md_data_local": str(local),
        # RELATIVE, always. An absolute machine path in a committed manifest stops being true the
        # moment the tree moves, which is the whole reason $MD_DATA is a variable.
        "path": relative.as_posix(),
        "namespace": parts[0],
        "month": parts[1],
        "dataset_name": parts[2],
        "dataset_root": str(local),
    }


# ---------------------------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------------------------

def _required(block: dict[str, Any], key: str, *, where: str, meaning: str) -> Any:
    value = block.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ContractError(
            f"{where}.{key} is not set, and this package must not guess it. {meaning}\n"
            f"  Set it in sys.config.yaml and run `md-openmm sys-gen` again.")
    return value


def _pinned_repository(block: dict[str, Any], where: str, *, what: str) -> dict[str, str]:
    """A repository and its exact 40-hex commit. Naming one without the other is not a pin."""
    repository = _required(block, "repository", where=where,
                           meaning=f"It is the {what} repository URL.")
    commit = _required(
        block, "commit", where=where,
        meaning=(f"It is the exact 40-hex commit of the {what} repository. A version or an "
                 f"installed fingerprint is useful generation provenance but is NOT a pin: "
                 f"naming a repository without its commit records where to look, not what ran."))
    if not COMMIT.match(str(commit).strip()):
        raise ContractError(
            f"{where}.commit is {commit!r}, which is not an exact 40-hex commit. A branch name, a "
            f"tag or an abbreviated SHA is not a pin.")
    return {"repository": str(repository).strip(), "commit": str(commit).strip()}


def build_manifest(*, dataset_block: dict[str, Any], location: dict[str, Any],
                   components: list[dict[str, Any]],
                   templates_version: str) -> dict[str, Any]:
    """Assemble the manifest. Every unguessable value comes from the user; the rest is derived.

    Derived here because each is a fact about THIS generation rather than a choice: the schema
    version (the contract's), `created_at`, `status: active` (a dataset being generated into is
    active by definition), `path` (where the dataset actually is under `MD_DATA`), the component
    list (what was actually written), and `templates.version`.
    """
    where = "dataset"
    block = dict(dataset_block or {})

    namespace = _required(block, "namespace", where=where,
                          meaning="It is the owning project, or `baseline` for shared data.")
    dataset_name = _required(block, "dataset_name", where=where,
                             meaning="It is the human-facing dataset name, and the last segment "
                                     "of the canonical path.")
    role = _required(block, "role", where=where,
                     meaning="It is `project` or `baseline`. The role is stated in the manifest; "
                             "the path alone is never the only evidence.")

    # The location on disk is the truth, and the declared identity has to agree with it. Checked
    # here rather than left to the validator so the message names the field a user edits.
    if str(namespace) != location["namespace"]:
        raise ContractError(
            f"dataset.namespace is {namespace!r} but MD_DATA_LOCAL puts this dataset under "
            f"{location['namespace']!r} ({location['path']}). They must agree.")
    if str(dataset_name) != location["dataset_name"]:
        raise ContractError(
            f"dataset.dataset_name is {dataset_name!r} but MD_DATA_LOCAL ends in "
            f"{location['dataset_name']!r} ({location['path']}). They must agree.")
    if str(role) == "baseline" and location["namespace"] != "baseline":
        raise ContractError(
            f"dataset.role is 'baseline' but the namespace is {location['namespace']!r}. "
            "`role: baseline` requires the reserved `baseline` namespace.")
    if str(role) != "baseline" and location["namespace"] == "baseline":
        raise ContractError(
            f"dataset.role is {role!r} but the dataset sits in the reserved `baseline` namespace. "
            "A project-owned dataset cannot live there.")

    creator = dict(block.get("created_by") or {})
    person = {
        "person_id": _required(creator, "person_id", where=f"{where}.created_by",
                               meaning="It is a scientific identity -- not the machine login or "
                                       "service account a job was submitted under."),
        "name": _required(creator, "name", where=f"{where}.created_by",
                          meaning="It is the creator's name."),
    }
    for optional in ("affiliation", "orcid"):
        if creator.get(optional):
            person[optional] = creator[optional]

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": _required(block, "dataset_id", where=where,
                                meaning="It is the stable machine-facing identifier, a lowercase "
                                        "slug. Deriving it from the path would make it stop being "
                                        "stable the moment the path changed."),
        "path": location["path"],
        "namespace": location["namespace"],
        "dataset_name": location["dataset_name"],
        "role": role,
        "system": _required(block, "system", where=where,
                            meaning="It is prose describing what was simulated."),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_by": person,
        # A dataset being generated into is active. Marking it complete is a deliberate MD-data
        # operation by its owner, and nothing here does it.
        "status": "active",
        "origin": _pinned_repository(dict(block.get("origin") or {}), f"{where}.origin",
                                     what="originating project"),
        "templates": {
            **_pinned_repository(dict(block.get("templates") or {}), f"{where}.templates",
                                 what="MD-tools"),
            "version": str(templates_version),
        },
        "components": components,
    }
    if block.get("derived_from"):
        manifest["derived_from"] = list(block["derived_from"])
    if block.get("notes"):
        manifest["notes"] = block["notes"]
    return manifest


def component_entry(name: str, *, kind: str, method: Optional[str] = None,
                    description: Optional[str] = None) -> dict[str, Any]:
    """One component of this dataset. The path is the name: components are not nested.

    `eq/nvt_1kcal` is a stage INSIDE the `eq` component, not a component of its own -- declaring
    one per stage would make four datasets out of one equilibration.
    """
    entry: dict[str, Any] = {"name": name, "type": kind, "path": name, "status": "active",
                             "linked": False}
    if method:
        entry["method"] = method
    if description:
        entry["description"] = description
    return entry


def validate(manifest: dict[str, Any], *, root: Optional[str] = None,
             dataset_root: Optional[str] = None) -> dict[str, Any]:
    """Hand the manifest to MD-data's validator. Metadata only unless `root` is given.

    With `root`, MD-data additionally checks the declared layout on disk: that the dataset root
    exists where the manifest says, and that each declared component directory exists and agrees
    with its `linked` flag. It opens no component contents and hashes nothing.
    """
    md_data = require_md_data()
    try:
        dataset = md_data.validate_dataset(manifest)
    except md_data.ContractViolation as error:
        raise ContractError(
            "the generated dataset.yaml does not satisfy the MD-data v1 contract:\n  "
            + "\n  ".join(error.problems)) from None

    layout: list[str] = []
    if root:
        from md_data.storage import check_dataset_tree

        try:
            layout = check_dataset_tree(Path(root), dataset)
        except md_data.ContractViolation as error:
            raise ContractError("; ".join(error.problems)) from None
        if layout:
            raise ContractError(
                f"the dataset layout under {root} does not match its manifest:\n  "
                + "\n  ".join(layout))
    # Deliberately NO absolute path. `path` is relative to $MD_DATA and the root itself is an
    # environment variable: recording its value would bake this machine's storage location into a
    # record that has to survive the whole tree being moved.
    return {
        "validator": md_data_identity(),
        "dataset_id": dataset.dataset_id,
        "role": dataset.role,
        "status": dataset.status,
        "read_only": dataset.read_only,
        "path": dataset.path,
        "components": [c.name for c in dataset.components],
        "layout_verified": bool(root),
    }


def write_manifest(dataset_root: Path, manifest: dict[str, Any], *,
                   header: str = "") -> Path:
    """Write `dataset.yaml`, refusing to overwrite a manifest with a different identity.

    Regenerating the same dataset is ordinary; silently replacing the identity of a dataset that
    already exists is not, because everything that referenced the old ID would now resolve to
    something else.
    """
    import yaml

    path = Path(dataset_root) / MANIFEST_NAME
    if path.is_file():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as error:
            raise ContractError(
                f"{path} exists but cannot be read ({type(error).__name__}: {error}). Refusing to "
                f"overwrite a manifest whose contents are unknown.") from None
        for field in ("dataset_id", "path", "namespace", "dataset_name", "role"):
            if existing.get(field) and existing.get(field) != manifest.get(field):
                raise ContractError(
                    f"{path} already exists with {field} = {existing[field]!r}, and this "
                    f"generation would write {manifest.get(field)!r}. Refusing to change the "
                    f"identity of an existing dataset: everything that referenced the old one "
                    f"would silently resolve to a different thing. Write into a new dataset root, "
                    f"or correct sys.config.yaml to match.")
        if str(existing.get("status")) in ("complete", "archived"):
            raise ContractError(
                f"{path} says status = {existing['status']!r}. A complete or archived dataset is "
                f"read-only: every comparison already made against it depends on it not moving. "
                f"Create a new dated dataset instead.")
        # Preserve the identity fields a human may legitimately have edited, and the creation time
        # of the dataset -- this generation is not its creation.
        for field in ("created_at", "notes", "derived_from"):
            if existing.get(field) and not manifest.get(field):
                manifest[field] = existing[field]
        if existing.get("created_at"):
            manifest["created_at"] = existing["created_at"]

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, width=88)
    path.write_text((header + text) if header else text, encoding="utf-8")
    return path


def merge_components(existing: list[dict[str, Any]],
                     new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the components this generation produced, keeping the ones already declared.

    `md-gen` adds method components to a dataset whose `common` was declared by `sys-gen`. A
    component already present keeps its recorded status: marking a finished component active again
    would be this package overriding a decision its owner made.
    """
    by_name = {entry["name"]: dict(entry) for entry in (existing or [])}
    order = [entry["name"] for entry in (existing or [])]
    for entry in new:
        if entry["name"] not in by_name:
            by_name[entry["name"]] = dict(entry)
            order.append(entry["name"])
    return [by_name[name] for name in order]
