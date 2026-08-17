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


def decide_provenance(source_root: Path, manifest_sha256: str, canonical_url: str):
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

    def record(resolved, state, sha=None):
        return BuildProvenance(
            schema_version=resources_mod.BUILD_PROVENANCE_SCHEMA_VERSION,
            resolved=resolved,
            source_state=state,
            commit_sha=sha,
            canonical_url=canonical_url,
            resource_manifest_sha256=manifest_sha256,
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
        if carried.resolved and carried.resource_manifest_sha256 == manifest_sha256:
            # The archive named a commit, and the catalog bytes in it still hash to exactly what
            # that record described. Anything else and the archive cannot be taken at its word.
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


def source_provenance_bytes(source_root: Path) -> bytes:
    """Decide this source tree's provenance and serialise it, without writing anything.

    Called before the sdist machinery runs, because `sdist` creates its release tree inside the
    project directory: judging the tree afterwards would see the build's own scratch space and call
    a clean checkout dirty.
    """
    registry_mod, _, resources_mod = _import_core()

    manifest, _ = build_manifest(source_root)
    manifest_sha256 = resources_mod.sha256_bytes(manifest.to_bytes())
    canonical_url = registry_mod.load_catalog(source_root).canonical_url
    provenance = decide_provenance(source_root, manifest_sha256, canonical_url)
    print(f"sdist provenance: {provenance.source_state}"
          + (f" @ {provenance.commit_sha}" if provenance.resolved else " (unresolved)"))
    return provenance.to_bytes()


def write_source_provenance(release_tree: Path, payload: bytes) -> None:
    """Record the decided provenance inside an sdist so a later wheel can inherit it.

    Written into the sdist staging tree only. The working tree is never touched -- a build that
    modified tracked source would make the next `git status` dirty and quietly poison the provenance
    of every subsequent build.
    """
    _, _, resources_mod = _import_core()
    if payload is None:
        raise CatalogBuildError("sdist provenance was not decided before the release tree was made")
    (Path(release_tree) / resources_mod.SOURCE_PROVENANCE_FILENAME).write_bytes(payload)
