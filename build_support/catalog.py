"""Build-time staging of the template catalog into the distribution.

The tracked root `registry.yaml` and `templates/**/template.yaml` are the **only** editable source
of the catalog. Nothing here creates a second tracked copy under `src/`; the packaged copy is
generated into the build tree (`build/lib/md_templates/core/_packaged/`) on the way to a wheel and
is never written back into the working tree.

That single-source rule is the whole design. Two tracked copies would be edited independently
roughly once, after which the repository would describe two different catalogs and the identity —
whose entire job is to say "these bytes, that commit" — would be ambiguous about which bytes.

What this module does, in order:

1. validate the source catalog with the *same* strict loader the runtime uses, so a broken catalog
   fails the build rather than shipping;
2. copy the exact bytes of the registry and every registered descriptor into the staging directory,
   under their logical repository paths;
3. write a deterministic resource manifest hashing those exact bytes;
4. decide the provenance of the source being built, and write it bound to that manifest by hash.

It imports `md_templates.core` from the source tree rather than from an installed copy, because at
build time there may not be an installed copy, and because validating with a *different* version of
the loader than the one being shipped would prove nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _import_core():
    """Import the catalog modules from the source tree being built."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from md_templates.core import identity, registry, resources  # noqa: WPS433

    return registry, identity, resources


class CatalogBuildError(RuntimeError):
    """The source catalog could not be validated or staged. The build must stop."""


#: The digest below is CLOSED-WORLD: every regular file in the tree is hashed, and only the entries
#: named here are skipped. An allowlist was the previous design and it was the wrong shape -- it
#: covered `src/`, `build_support/`, `templates/` and four root files, so README.md, a build-
#: configuration file, a descriptor's referenced documentation and any newly added file were all
#: invisible. A digest whose coverage has to be remembered will eventually be forgotten; the only
#: maintainable version is "everything, minus a short list of things that are generated".
#:
#: Each exclusion earns its place by being REGENERATED during a build. Including any of them would
#: make the digest depend on whether something had been built before, and two builds of one commit
#: must produce identical metadata.
EXCLUDED_COMPONENTS = frozenset({"__pycache__", ".git"})       # bytecode caches, VCS internals
EXCLUDED_COMPONENT_SUFFIXES = (".egg-info",)                    # rewritten by every egg_info run
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
EXCLUDED_TOP_LEVEL_DIRS = frozenset({"build", "dist"})          # `python -m build` output


def _is_excluded(relative: Path, provenance_filename: str) -> bool:
    parts = relative.parts
    if relative.as_posix() == provenance_filename:
        # The record cannot describe itself: its own digest field is written from this value.
        return True
    if parts and parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
        return True
    for part in parts:
        if part in EXCLUDED_COMPONENTS:
            return True
        if part.endswith(EXCLUDED_COMPONENT_SUFFIXES):
            return True
    return relative.suffix in EXCLUDED_FILE_SUFFIXES


def _digest_entries(root: Path) -> dict:
    """`{normalised logical path: content digest}` for every regular file that is not generated."""
    _, _, resources_mod = _import_core()
    root = Path(root)
    entries: dict[str, str] = {}
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if _is_excluded(relative, resources_mod.SOURCE_PROVENANCE_FILENAME):
            continue
        key = relative.as_posix()
        if candidate.is_symlink():
            # Hash the link target, not what it points at: replacing a file with a symlink is a
            # change to the tree and must move the digest.
            entries[key] = resources_mod.sha256_bytes(
                b"symlink:" + os.readlink(candidate).encode("utf-8"))
        elif candidate.is_file():
            entries[key] = resources_mod.sha256_bytes(candidate.read_bytes())
    return entries


def source_tree_digest(root: Path) -> str:
    """One digest over every regular file in the source tree, keyed by normalised logical path."""
    _, _, resources_mod = _import_core()
    entries = _digest_entries(root)
    return resources_mod.sha256_bytes(
        resources_mod.canonical_json_bytes({"files": dict(sorted(entries.items()))}))


def decide_provenance(source_root: Path, manifest_sha256: str, canonical_url: str,
                      source_sha256: str | None = None):
    """Decide what this source tree may claim about itself.

    Five outcomes, all closed values -- see `SOURCE_STATES`. The rules the instruction fixes:

    * a clean Git checkout contributes its exact full HEAD;
    * a failed or timed-out `git status` is **unresolved**, never clean;
    * untracked non-ignored files count as dirty, because they are part of what was built;
    * no environment variable overrides any of this, and there is deliberately no
      "allow dirty but trust HEAD" escape hatch -- that option is the defect PR 1 removed, and
      adding it back at build time would be the same mistake one layer down.

    The remaining case is an sdist: no `.git`, but a provenance record written when the archive was
    made. It is honoured only if the catalog bytes still hash to what that record described, so an
    archive whose templates were edited after unpacking cannot inherit a clean commit.
    """
    _, identity_mod, resources_mod = _import_core()
    BuildProvenance = resources_mod.BuildProvenance

    digest = source_sha256 if source_sha256 is not None else source_tree_digest(source_root)

    def record(resolved, state, sha=None):
        return BuildProvenance(
            schema_version=resources_mod.BUILD_PROVENANCE_SCHEMA_VERSION,
            resolved=resolved,
            source_state=state,
            commit_sha=sha,
            canonical_url=canonical_url,
            resource_manifest_sha256=manifest_sha256,
            source_tree_sha256=digest,
        )

    git = identity_mod.inspect_provenance(source_root)

    if git.commit_sha is not None:
        if not git.status_known:
            return record(False, "unverifiable-git-status")
        if git.dirty:
            return record(False, "dirty-source-tree")
        return record(True, "clean-git-checkout", git.commit_sha)

    archive = source_root / resources_mod.SOURCE_PROVENANCE_FILENAME
    if archive.is_file() and not archive.is_symlink():
        try:
            carried = BuildProvenance.model_validate(
                json.loads(archive.read_text(encoding="utf-8")))
        except Exception:
            return record(False, "no-verifiable-git-provenance")
        # The archive may be taken at its word only if BOTH digests still agree: the catalog bytes
        # and every build- and runtime-relevant source file. Binding the catalog alone would let an
        # unpacked archive have `packaged.py`, an engine module or `build_support/catalog.py`
        # rewritten and still inherit the commit -- a commit naming code it never contained.
        if (carried.resolved
                and carried.resource_manifest_sha256 == manifest_sha256
                and carried.source_tree_sha256 is not None
                and carried.source_tree_sha256 == digest):
            return record(True, "verified-source-archive", carried.commit_sha)
        return record(False, "no-verifiable-git-provenance")

    return record(False, "no-verifiable-git-provenance")


def build_manifest(source_root: Path):
    """Validate the source catalog and describe exactly what will be packaged.

    Returns `(manifest, {logical path: bytes})`. Validation is the shipped strict loader, so every
    guarantee the runtime relies on -- schema, uniqueness, normalised non-escaping paths, no
    symlinks, registry/descriptor agreement, reference existence -- is a build gate too.
    """
    registry_mod, _, resources_mod = _import_core()

    try:
        catalog = registry_mod.load_catalog(source_root)
    except Exception as exc:
        raise CatalogBuildError(
            f"the source catalog at {source_root} does not validate, so it will not be packaged: "
            f"{exc}"
        ) from exc

    payloads: dict[str, bytes] = {}
    registry_path = source_root / registry_mod.REGISTRY_FILENAME
    payloads["registry.yaml"] = registry_path.read_bytes()

    references: set[str] = set()
    for entry in catalog.entries:
        payloads[entry.template_path] = (source_root / entry.template_path).read_bytes()
        references.update(catalog.descriptor(entry.template_id).repository_references)

    manifest = resources_mod.ResourceManifest(
        schema_version=resources_mod.RESOURCE_MANIFEST_SCHEMA_VERSION,
        resources={path: resources_mod.sha256_bytes(data)
                   for path, data in sorted(payloads.items())},
        # Recorded, not shipped. These name documentation and source paths that legitimately do not
        # travel in a wheel; `load_catalog` above already proved each one exists in this source
        # tree, and the installed loader reports them as validated-at-build rather than pretending
        # a `docs/` path exists in site-packages.
        references_validated_at_build=sorted(references),
    )
    return manifest, payloads


def stage_catalog(source_root: Path, package_root: Path) -> Path:
    """Write the packaged catalog into `package_root/md_templates/core/_packaged/`.

    `package_root` is a build tree (`build/lib`), never the working tree.
    """
    registry_mod, _, resources_mod = _import_core()

    manifest, payloads = build_manifest(source_root)
    manifest_bytes = manifest.to_bytes()
    manifest_sha256 = resources_mod.sha256_bytes(manifest_bytes)

    canonical_url = registry_mod.load_catalog(source_root).canonical_url
    provenance = decide_provenance(source_root, manifest_sha256, canonical_url)

    target = Path(package_root) / "md_templates" / "core" / resources_mod.PACKAGED_SUBDIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for relative, data in sorted(payloads.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    (target / resources_mod.RESOURCE_MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (target / resources_mod.BUILD_PROVENANCE_FILENAME).write_bytes(provenance.to_bytes())

    print(f"catalog staged: {len(payloads)} resource(s), provenance {provenance.source_state}"
          + (f" @ {provenance.commit_sha}" if provenance.resolved else " (unresolved)"))
    return target


def sdist_extra_files(source_root: Path) -> list[str]:
    """Everything an sdist must carry for its catalog to still validate after unpacking.

    Derived from the descriptors rather than hard-coded in `MANIFEST.in`, so the declaration stays in
    one place: a descriptor that grows a `repository_references` entry ships that path automatically.
    Without this an unpacked sdist would fail its own build with "reference does not exist", which is
    the correct check firing on an incomplete archive rather than a false alarm.
    """
    registry_mod, _, _ = _import_core()
    catalog = registry_mod.load_catalog(source_root)

    wanted: list[str] = [registry_mod.REGISTRY_FILENAME]
    for entry in catalog.entries:
        wanted.append(entry.template_path)
        for ref in catalog.descriptor(entry.template_id).repository_references:
            target = source_root / ref
            if target.is_dir():
                wanted.extend(
                    p.relative_to(source_root).as_posix()
                    for p in sorted(target.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts
                )
            elif target.is_file():
                wanted.append(ref)
    return sorted(set(wanted))


def copy_referenced_sources(source_root: Path, release_tree: Path,
                            relative_paths: list[str]) -> None:
    """Ensure every listed path exists in the release tree, copying what the manifest missed."""
    for relative in relative_paths:
        source = Path(source_root) / relative
        target = Path(release_tree) / relative
        if target.exists() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def decide_source_state(source_root: Path):
    """The Git half of the decision, taken before the sdist machinery touches anything.

    `sdist` creates its release tree inside the project directory, so judging the tree afterwards
    would see the build's own scratch space and call a clean checkout dirty. The digests, by
    contrast, must be computed from the finished release tree -- that is the tree the archive will
    actually contain -- so the two halves are taken at different moments and combined at the end.
    """
    _, identity_mod, _ = _import_core()
    git = identity_mod.inspect_provenance(source_root)
    if git.commit_sha is None:
        return False, "no-verifiable-git-provenance", None
    if not git.status_known:
        return False, "unverifiable-git-status", None
    if git.dirty:
        return False, "dirty-source-tree", None
    return True, "clean-git-checkout", git.commit_sha


def write_source_provenance(release_tree: Path, source_state) -> None:
    """Record the archive's provenance, with digests taken over the finished release tree.

    Written into the sdist staging tree only. The working tree is never touched -- a build that
    modified tracked source would make the next `git status` dirty and quietly poison the provenance
    of every subsequent build.
    """
    registry_mod, _, resources_mod = _import_core()
    if source_state is None:
        raise CatalogBuildError("sdist provenance was not decided before the release tree was made")

    resolved, state, sha = source_state
    release_tree = Path(release_tree)
    manifest, _ = build_manifest(release_tree)
    provenance = resources_mod.BuildProvenance(
        schema_version=resources_mod.BUILD_PROVENANCE_SCHEMA_VERSION,
        resolved=resolved,
        source_state=state,
        commit_sha=sha,
        canonical_url=registry_mod.load_catalog(release_tree).canonical_url,
        resource_manifest_sha256=resources_mod.sha256_bytes(manifest.to_bytes()),
        source_tree_sha256=source_tree_digest(release_tree),
    )
    (release_tree / resources_mod.SOURCE_PROVENANCE_FILENAME).write_bytes(provenance.to_bytes())
    print(f"sdist provenance: {provenance.source_state}"
          + (f" @ {provenance.commit_sha}" if provenance.resolved else " (unresolved)"))
