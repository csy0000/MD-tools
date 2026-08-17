"""The packaged catalog and its build provenance.

Every wheel here is built from a temporary Git repository this file creates, never from the
developer's checkout — which is dirty by definition while the implementation is being edited, and
would make a clean-build test pass or fail for reasons unrelated to the code.

Three wheels are built in total and shared across the tests that need them: one from a clean
committed tree, one from the same tree made dirty, one with an untracked file added. Wheel builds
are the expensive part, so they are module-scoped.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from md_templates.core import load_catalog
from md_templates.core.packaged import (
    BuildProvenanceError,
    PackagedCatalogError,
    ResourceIntegrityError,
    UnresolvedBuildProvenanceError,
    resolve_packaged_identity,
    verify_packaged_catalog,
)
from md_templates.core.resources import (
    BUILD_PROVENANCE_FILENAME,
    BUILD_PROVENANCE_SCHEMA_VERSION,
    PACKAGED_SUBDIR,
    RESOURCE_MANIFEST_FILENAME,
    RESOURCE_MANIFEST_SCHEMA_VERSION,
    BuildProvenance,
    canonical_json_bytes,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/csy0000/MD-templates"
MD_ID = "conventional-md/openmm/explicit-water"
REST2_ID = "rest2/openmm/explicit-water"
REST2_PATH = f"templates/{REST2_ID}/template.yaml"

#: Everything a build needs, and nothing else.
#: `.gitignore` is not optional here: a build writes `build/`, `dist/` and `*.egg-info/` into the
#: source root, and without the ignore rules those artifacts would make the tree look dirty and
#: every clean-build test would fail for a reason that has nothing to do with the catalog.
SOURCE_ITEMS = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", ".gitignore",
                "registry.yaml", "templates", "src", "build_support")


# ------------------------------------------------------------------------------------------------
# building and installing, in isolation from the developer's tree
# ------------------------------------------------------------------------------------------------

def run(args, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=900, **kwargs)
    if result.returncode != 0:
        raise AssertionError(f"{args} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return result


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(root), *args])


def make_source_tree(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for item in SOURCE_ITEMS:
        source = REPO_ROOT / item
        target = dest / item
        if source.is_dir():
            shutil.copytree(source, target,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)

    # Whatever the descriptors reference must exist, because the build validates references against
    # the source tree before packaging. Copied from the real repository rather than faked, so a
    # descriptor that grows a reference does not silently stop being checked here.
    catalog = load_catalog(REPO_ROOT)
    for entry in catalog.entries:
        for ref in catalog.descriptor(entry.template_id).repository_references:
            source, target = REPO_ROOT / ref, dest / ref
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(source, target)
    return dest


def make_clean_repo(dest: Path) -> Path:
    make_source_tree(dest)
    git(dest.parent, "init", "--quiet", str(dest))
    git(dest, "config", "user.email", "test@example.invalid")
    git(dest, "config", "user.name", "Packaging Test")
    git(dest, "config", "commit.gpgsign", "false")
    git(dest, "add", "-A")
    git(dest, "commit", "--quiet", "-m", "packaged catalog source")
    assert not git(dest, "status", "--porcelain").stdout.strip()
    return dest


def build_wheel(source_root: Path, outdir: Path) -> Path:
    """Build with `--no-isolation`: this environment already has setuptools, pyyaml and pydantic.

    Isolation would fetch them from the network, which would make these tests need it.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(outdir), str(source_root)])
    wheels = sorted(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def install_wheel(wheel: Path, target: Path) -> Path:
    """A real install into an empty directory, offline and without dependencies."""
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-index",
         "--target", str(target), str(wheel)])
    return target


def in_install(target: Path, code: str, cwd: Path, extra_env: dict | None = None):
    """Run code against the installed package, from a directory with no checkout in sight."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(target)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=900, cwd=str(cwd), env=env)


@pytest.fixture(scope="module")
def clean_install(tmp_path_factory):
    """A wheel built from a clean committed checkout, installed into an empty directory."""
    base = tmp_path_factory.mktemp("clean")
    repo = make_clean_repo(base / "repo")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    wheel = build_wheel(repo, base / "dist")
    target = install_wheel(wheel, base / "site")
    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    return {"repo": repo, "head": head, "wheel": wheel, "target": target,
            "elsewhere": elsewhere,
            "packaged": target / "md_templates" / "core" / PACKAGED_SUBDIR}


@pytest.fixture(scope="module")
def dirty_install(tmp_path_factory):
    """The same tree with a tracked catalog file edited but not committed."""
    base = tmp_path_factory.mktemp("dirty")
    repo = make_clean_repo(base / "repo")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    descriptor = repo / REST2_PATH
    descriptor.write_text(descriptor.read_text() + "\n# edited, not committed\n", encoding="utf-8")
    assert git(repo, "status", "--porcelain").stdout.strip()
    wheel = build_wheel(repo, base / "dist")
    target = install_wheel(wheel, base / "site")
    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    return {"repo": repo, "head": head, "target": target, "elsewhere": elsewhere,
            "packaged": target / "md_templates" / "core" / PACKAGED_SUBDIR}


@pytest.fixture(scope="module")
def untracked_install(tmp_path_factory):
    """A tree that is clean except for an untracked, non-ignored source file."""
    base = tmp_path_factory.mktemp("untracked")
    repo = make_clean_repo(base / "repo")
    (repo / "src" / "md_templates" / "core" / "scratch_helper.py").write_text(
        "# not committed\n", encoding="utf-8")
    wheel = build_wheel(repo, base / "dist")
    target = install_wheel(wheel, base / "site")
    return {"repo": repo, "target": target,
            "packaged": target / "md_templates" / "core" / PACKAGED_SUBDIR}


# ================================================================================================
# 1. source and wheel equivalence
# ================================================================================================

def projection(catalog) -> dict:
    """A canonical serialisation of everything a consumer can observe about the catalog."""
    return {
        "canonical_url": catalog.canonical_url,
        "schema_version": catalog.registry.schema_version,
        "identity_scheme": catalog.registry.identity.scheme,
        "entries": [
            {
                "template_id": e.template_id,
                "template_path": e.template_path,
                "method": e.method,
                "engine": e.engine,
                "method_aliases": list(e.method_aliases),
                "tags": list(e.tags),
            }
            for e in catalog.entries
        ],
        "descriptors": {
            tid: catalog.descriptors[tid].model_dump(mode="json")
            for tid in sorted(catalog.descriptors)
        },
    }


def test_1_source_and_packaged_projections_are_identical(clean_install):
    source = projection(load_catalog(clean_install["repo"]))

    result = in_install(clean_install["target"], """
        import json
        from md_templates.core.packaged import load_packaged_catalog
        catalog = load_packaged_catalog()
        print(json.dumps({
            "canonical_url": catalog.canonical_url,
            "schema_version": catalog.registry.schema_version,
            "identity_scheme": catalog.registry.identity.scheme,
            "entries": [
                {"template_id": e.template_id, "template_path": e.template_path,
                 "method": e.method, "engine": e.engine,
                 "method_aliases": list(e.method_aliases), "tags": list(e.tags)}
                for e in catalog.entries
            ],
            "descriptors": {t: catalog.descriptors[t].model_dump(mode="json")
                            for t in sorted(catalog.descriptors)},
        }))
        """, cwd=clean_install["elsewhere"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == source


def test_1b_packaged_loader_runs_with_no_checkout_on_the_path(clean_install):
    result = in_install(clean_install["target"], """
        import md_templates, pathlib, sys
        p = pathlib.Path(md_templates.__file__).resolve()
        assert 'site' in p.parts, p
        from md_templates.core.packaged import load_packaged_catalog
        print(sorted(load_packaged_catalog().descriptors))
        """, cwd=clean_install["elsewhere"])
    assert result.returncode == 0, result.stderr
    assert "rest2/openmm/explicit-water" in result.stdout


def test_1c_packaged_catalog_reports_its_origin_and_reference_policy(clean_install):
    packaged = verify_packaged_catalog(clean_install["packaged"])
    assert packaged.catalog.origin == "packaged-distribution"
    assert packaged.catalog.reference_policy == "validated-at-build"
    assert load_catalog(clean_install["repo"]).reference_policy == "source-tree-existence"
    # the references themselves are recorded, so nothing pretends a docs/ path ships in a wheel
    assert "docs/support-matrix.md" in packaged.manifest.references_validated_at_build
    assert not (clean_install["packaged"] / "docs").exists()


# ================================================================================================
# 2. clean build identity
# ================================================================================================

def test_2_clean_build_records_the_exact_head(clean_install):
    packaged = verify_packaged_catalog(clean_install["packaged"])
    provenance = packaged.provenance
    assert provenance.resolved is True
    assert provenance.source_state == "clean-git-checkout"
    assert provenance.commit_sha == clean_install["head"]
    assert len(provenance.commit_sha) == 40 and provenance.commit_sha.islower()
    assert provenance.schema_version == BUILD_PROVENANCE_SCHEMA_VERSION
    assert packaged.manifest.schema_version == RESOURCE_MANIFEST_SCHEMA_VERSION


@pytest.mark.parametrize("ref", [MD_ID, REST2_ID])
def test_2b_both_packaged_identities_use_that_sha_and_logical_paths(clean_install, ref):
    identity = resolve_packaged_identity(ref, root=clean_install["packaged"])
    head = clean_install["head"]
    assert identity.commit_sha == head
    assert identity.canonical_url == URL
    assert identity.template_path == f"templates/{ref}/template.yaml"
    assert identity.canonical == f"{URL}@{head}#templates/{ref}/template.yaml"


def test_2c_no_branch_version_timestamp_or_wheel_path_appears_in_the_identity(clean_install):
    identity = resolve_packaged_identity(REST2_ID, root=clean_install["packaged"])
    text = identity.canonical
    for banned in ("main", "master", "0.1.0", "site-packages", PACKAGED_SUBDIR, ".whl",
                   str(clean_install["target"]), "2026"):
        assert banned not in text, banned
    assert text.startswith(f"{URL}@") and "#templates/" in text


def test_2d_build_metadata_is_deterministic(tmp_path_factory):
    """Two builds of the same commit must produce byte-identical metadata."""
    base = tmp_path_factory.mktemp("determinism")
    repo = make_clean_repo(base / "repo")
    records = []
    for i in (1, 2):
        wheel = build_wheel(repo, base / f"dist{i}")
        target = install_wheel(wheel, base / f"site{i}")
        packaged = target / "md_templates" / "core" / PACKAGED_SUBDIR
        records.append(((packaged / RESOURCE_MANIFEST_FILENAME).read_bytes(),
                        (packaged / BUILD_PROVENANCE_FILENAME).read_bytes()))
    assert records[0] == records[1]
    blob = records[0][0] + records[0][1]
    for machine_local in (str(repo).encode(), b"tmp", b"T00:00", b"Z\"", b"hostname"):
        assert machine_local not in blob, machine_local


def test_2e_building_leaves_the_working_tree_unmodified(clean_install):
    """A build that dirtied its own source would poison the provenance of the next one."""
    assert git(clean_install["repo"], "status", "--porcelain").stdout.strip() == ""


# ================================================================================================
# 3. dirty build refusal
# ================================================================================================

def test_3_dirty_build_does_not_claim_resolved_provenance(dirty_install):
    packaged = verify_packaged_catalog(dirty_install["packaged"])
    assert packaged.provenance.resolved is False
    assert packaged.provenance.source_state == "dirty-source-tree"
    assert packaged.provenance.commit_sha is None


def test_3b_dirty_head_is_not_written_anywhere_in_the_metadata(dirty_install):
    blob = (dirty_install["packaged"] / BUILD_PROVENANCE_FILENAME).read_bytes()
    assert dirty_install["head"].encode() not in blob


def test_3c_dirty_build_still_lists_its_catalog(dirty_install):
    """Integrity is intact even though provenance is not; listing stays useful."""
    catalog = verify_packaged_catalog(dirty_install["packaged"]).catalog
    assert sorted(catalog.descriptors) == sorted([MD_ID, REST2_ID])


def test_3d_dirty_build_refuses_identity_with_a_typed_actionable_error(dirty_install):
    with pytest.raises(UnresolvedBuildProvenanceError) as exc:
        resolve_packaged_identity(REST2_ID, root=dirty_install["packaged"])
    message = str(exc.value)
    assert "dirty-source-tree" in message
    assert "uncommitted" in message and "Rebuild" in message


def test_3e_the_same_head_cannot_be_pushed_back_in_through_a_convenience_argument(dirty_install):
    """No caller-supplied SHA reaches the packaged API — the PR 1 defect stays closed."""
    import inspect

    params = inspect.signature(resolve_packaged_identity).parameters
    assert set(params) == {"template_ref", "root"}
    with pytest.raises(TypeError):
        resolve_packaged_identity(REST2_ID, commit_sha=dirty_install["head"])
    with pytest.raises(TypeError):
        resolve_packaged_identity(REST2_ID, trusted=dirty_install["head"])


def test_3f_an_untracked_source_file_makes_the_build_dirty(untracked_install):
    provenance = verify_packaged_catalog(untracked_install["packaged"]).provenance
    assert provenance.resolved is False
    assert provenance.source_state == "dirty-source-tree"
    with pytest.raises(UnresolvedBuildProvenanceError):
        resolve_packaged_identity(MD_ID, root=untracked_install["packaged"])


def test_3g_a_failed_git_status_is_unresolved_never_clean(clean_install, monkeypatch):
    """A tree that cannot be shown clean has not been shown clean."""
    from build_support import catalog as build_catalog
    from md_templates.core import identity as identity_mod

    real = identity_mod._git

    def fake(root, *args):
        if args and args[0] == "status":
            return subprocess.CompletedProcess(args, 128, "", "fatal: broken index")
        return real(root, *args)

    monkeypatch.setattr(identity_mod, "_git", fake)
    provenance = build_catalog.decide_provenance(
        clean_install["repo"], sha256_bytes(b"whatever"), URL)
    assert provenance.resolved is False
    assert provenance.source_state == "unverifiable-git-status"
    assert provenance.commit_sha is None


# ================================================================================================
# 4. no-Git installed operation
# ================================================================================================

def test_4_installed_operation_needs_no_checkout_no_git_and_no_cwd_assumption(clean_install,
                                                                             tmp_path):
    """The source repository is deleted outright before this runs."""
    base = tmp_path / "standalone"
    repo = make_clean_repo(base / "repo")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    wheel = build_wheel(repo, base / "dist")
    target = install_wheel(wheel, base / "site")

    shutil.rmtree(repo)
    assert not repo.exists()

    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    result = in_install(target, """
        import pathlib, sys
        from md_templates.core.packaged import (
            load_packaged_catalog, resolve_packaged_identity, packaged_catalog_root)
        assert not (packaged_catalog_root().parent / '.git').exists()
        catalog = load_packaged_catalog()
        print(len(catalog.entries))
        print(resolve_packaged_identity('rest2/openmm/explicit-water').canonical)
        """, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "2"
    assert lines[1] == f"{URL}@{head}#{REST2_PATH}"


def test_4b_the_installed_tree_contains_no_git_metadata(clean_install):
    assert not list(clean_install["target"].rglob(".git"))


# ================================================================================================
# 5. tamper and truncation detection
# ================================================================================================

@pytest.fixture
def tampered(clean_install, tmp_path):
    """A private copy of the packaged catalog, safe to corrupt."""
    target = tmp_path / PACKAGED_SUBDIR
    shutil.copytree(clean_install["packaged"], target)
    return target


def expect_refusal(root, exception=Exception):
    """Whatever the case, no identity may come out."""
    with pytest.raises(exception):
        resolve_packaged_identity(REST2_ID, root=root)


def test_5a_changed_registry_bytes(tampered):
    path = tampered / "registry.yaml"
    path.write_text(path.read_text() + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(ResourceIntegrityError, match="hash to"):
        verify_packaged_catalog(tampered)
    expect_refusal(tampered, ResourceIntegrityError)


def test_5b_changed_descriptor_bytes(tampered):
    path = tampered / REST2_PATH
    path.write_text(path.read_text().replace("api_version: 1", "api_version: 2"), encoding="utf-8")
    with pytest.raises(ResourceIntegrityError, match="hash to"):
        verify_packaged_catalog(tampered)


def test_5c_missing_descriptor(tampered):
    (tampered / REST2_PATH).unlink()
    with pytest.raises(ResourceIntegrityError, match="does not contain it"):
        verify_packaged_catalog(tampered)


def test_5d_missing_provenance(tampered):
    (tampered / BUILD_PROVENANCE_FILENAME).unlink()
    with pytest.raises(BuildProvenanceError, match="no build_provenance.json"):
        verify_packaged_catalog(tampered)


def test_5e_malformed_provenance(tampered):
    (tampered / BUILD_PROVENANCE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(BuildProvenanceError):
        verify_packaged_catalog(tampered)


def test_5f_unsupported_provenance_schema_version(tampered):
    document = json.loads((tampered / BUILD_PROVENANCE_FILENAME).read_text())
    document["schema_version"] = 99
    (tampered / BUILD_PROVENANCE_FILENAME).write_bytes(canonical_json_bytes(document))
    with pytest.raises(BuildProvenanceError, match="schema_version"):
        verify_packaged_catalog(tampered)


@pytest.mark.parametrize("bad_sha", ["4d21838", "not-a-sha", "Z" * 40, "", None])
def test_5g_malformed_or_abbreviated_sha(tampered, bad_sha):
    document = json.loads((tampered / BUILD_PROVENANCE_FILENAME).read_text())
    document["commit_sha"] = bad_sha
    (tampered / BUILD_PROVENANCE_FILENAME).write_bytes(canonical_json_bytes(document))
    with pytest.raises(BuildProvenanceError):
        verify_packaged_catalog(tampered)


def test_5h_resource_manifest_path_traversal(tampered):
    document = json.loads((tampered / RESOURCE_MANIFEST_FILENAME).read_text())
    document["resources"]["../../../../etc/passwd"] = "0" * 64
    (tampered / RESOURCE_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(document))
    # the manifest hash also moves, so this must fail whichever check fires first
    with pytest.raises(PackagedCatalogError):
        verify_packaged_catalog(tampered)


def test_5i_resource_hash_mismatch_recorded_in_the_manifest(tampered):
    document = json.loads((tampered / RESOURCE_MANIFEST_FILENAME).read_text())
    document["resources"][REST2_PATH] = "0" * 64
    (tampered / RESOURCE_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(document))
    with pytest.raises(BuildProvenanceError, match="mixed"):
        verify_packaged_catalog(tampered)


def test_5j_provenance_repaired_to_match_a_tampered_manifest_still_fails(tampered):
    """Rewriting both records consistently is the interesting case, not the naive one."""
    document = json.loads((tampered / RESOURCE_MANIFEST_FILENAME).read_text())
    document["resources"][REST2_PATH] = "0" * 64
    manifest_bytes = canonical_json_bytes(document)
    (tampered / RESOURCE_MANIFEST_FILENAME).write_bytes(manifest_bytes)

    provenance = json.loads((tampered / BUILD_PROVENANCE_FILENAME).read_text())
    provenance["resource_manifest_sha256"] = sha256_bytes(manifest_bytes)
    (tampered / BUILD_PROVENANCE_FILENAME).write_bytes(canonical_json_bytes(provenance))

    # binding now passes, so integrity has to catch it
    with pytest.raises(ResourceIntegrityError, match="hash to"):
        verify_packaged_catalog(tampered)


def test_5k_an_unclaimed_resource_is_refused(tampered):
    (tampered / "templates" / "rogue.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ResourceIntegrityError, match="absent from"):
        verify_packaged_catalog(tampered)


def test_5l_a_registered_but_unpackaged_descriptor_is_refused(tampered):
    """Drop a descriptor from both the tree and the manifest, keeping the two consistent."""
    (tampered / REST2_PATH).unlink()
    document = json.loads((tampered / RESOURCE_MANIFEST_FILENAME).read_text())
    document["resources"].pop(REST2_PATH)
    manifest_bytes = canonical_json_bytes(document)
    (tampered / RESOURCE_MANIFEST_FILENAME).write_bytes(manifest_bytes)
    provenance = json.loads((tampered / BUILD_PROVENANCE_FILENAME).read_text())
    provenance["resource_manifest_sha256"] = sha256_bytes(manifest_bytes)
    (tampered / BUILD_PROVENANCE_FILENAME).write_bytes(canonical_json_bytes(provenance))

    with pytest.raises(PackagedCatalogError, match="registered but not packaged"):
        verify_packaged_catalog(tampered)


def test_5m_a_symlinked_packaged_resource_is_refused(tampered, tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_bytes((tampered / REST2_PATH).read_bytes())
    (tampered / REST2_PATH).unlink()
    (tampered / REST2_PATH).symlink_to(outside)
    with pytest.raises(ResourceIntegrityError, match="symlink"):
        verify_packaged_catalog(tampered)


def test_5n_provenance_naming_a_different_repository_is_refused(tampered):
    document = json.loads((tampered / BUILD_PROVENANCE_FILENAME).read_text())
    document["canonical_url"] = "https://github.com/someone-else/MD-templates"
    (tampered / BUILD_PROVENANCE_FILENAME).write_bytes(canonical_json_bytes(document))
    with pytest.raises(BuildProvenanceError, match="names repository"):
        resolve_packaged_identity(REST2_ID, root=tampered)


def test_5o_a_record_claiming_resolution_without_a_sha_is_rejected_by_the_model():
    with pytest.raises(Exception, match="no commit_sha"):
        BuildProvenance(schema_version=1, resolved=True, source_state="clean-git-checkout",
                        commit_sha=None, canonical_url=URL,
                        resource_manifest_sha256="a" * 64)


def test_5p_a_record_carrying_a_sha_while_unresolved_is_rejected_by_the_model():
    with pytest.raises(Exception, match="must not carry a SHA|commit_sha"):
        BuildProvenance(schema_version=1, resolved=False, source_state="dirty-source-tree",
                        commit_sha="a" * 40, canonical_url=URL,
                        resource_manifest_sha256="a" * 64)


def test_5q_no_packaged_catalog_at_all_is_a_clear_error(tmp_path):
    with pytest.raises(PackagedCatalogError, match="no resource_manifest.json"):
        verify_packaged_catalog(tmp_path)


# ================================================================================================
# 6. no network, and the dependency boundary
# ================================================================================================

def test_6_packaged_use_makes_no_network_call_and_loads_no_heavy_dependency(clean_install):
    result = in_install(clean_install["target"], """
        import socket, sys

        def refuse(*a, **k):
            raise RuntimeError("packaged catalog attempted a network operation")

        socket.socket = refuse
        socket.create_connection = refuse
        socket.getaddrinfo = refuse

        from md_templates.core.packaged import load_packaged_catalog, resolve_packaged_identity
        catalog = load_packaged_catalog()
        identity = resolve_packaged_identity('rest2/openmm/explicit-water')
        heavy = sorted(m for m in sys.modules
                       if m.split('.')[0] in {'openmm', 'openff', 'rdkit', 'mdtraj', 'parmed',
                                              'openmmtools', 'numpy', 'scipy', 'pandas'})
        print(identity.canonical)
        print('HEAVY:' + ','.join(heavy))
        """, cwd=clean_install["elsewhere"])
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"{URL}@{clean_install['head']}#{REST2_PATH}"
    assert lines[1] == "HEAVY:", lines[1]


def test_6b_the_packaged_module_imports_no_engine_dependency():
    text = (REPO_ROOT / "src" / "md_templates" / "core" / "packaged.py").read_text()
    for banned in ("import openmm", "import openff", "import rdkit", "import numpy",
                   "from openmm", "md_templates.openmm"):
        assert banned not in text, banned


# ================================================================================================
# the sdist route: provenance survives an unmodified archive, and only an unmodified one
# ================================================================================================

def build_sdist(source_root: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "build", "--sdist", "--no-isolation",
         "--outdir", str(outdir), str(source_root)])
    archives = sorted(outdir.glob("*.tar.gz"))
    assert len(archives) == 1, archives
    return archives[0]


def unpack(archive: Path, into: Path) -> Path:
    import tarfile

    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(into)
    roots = [p for p in into.iterdir() if p.is_dir()]
    assert len(roots) == 1, roots
    return roots[0]


@pytest.fixture(scope="module")
def sdist_tree(tmp_path_factory):
    base = tmp_path_factory.mktemp("sdist")
    repo = make_clean_repo(base / "repo")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    archive = build_sdist(repo, base / "sdist")
    unpacked = unpack(archive, base / "unpacked")
    return {"base": base, "repo": repo, "head": head, "unpacked": unpacked}


def test_sdist_carries_source_provenance_and_no_git(sdist_tree):
    from md_templates.core.resources import SOURCE_PROVENANCE_FILENAME

    unpacked = sdist_tree["unpacked"]
    assert not (unpacked / ".git").exists()
    record = json.loads((unpacked / SOURCE_PROVENANCE_FILENAME).read_text())
    assert record["resolved"] is True
    assert record["source_state"] == "clean-git-checkout"
    assert record["commit_sha"] == sdist_tree["head"]
    assert (unpacked / "registry.yaml").is_file()


def test_wheel_from_an_unmodified_sdist_inherits_the_commit(sdist_tree, tmp_path):
    wheel = build_wheel(sdist_tree["unpacked"], tmp_path / "dist")
    target = install_wheel(wheel, tmp_path / "site")
    packaged = verify_packaged_catalog(target / "md_templates" / "core" / PACKAGED_SUBDIR)
    assert packaged.provenance.resolved is True
    assert packaged.provenance.source_state == "verified-source-archive"
    assert packaged.provenance.commit_sha == sdist_tree["head"]
    identity = resolve_packaged_identity(
        REST2_ID, root=target / "md_templates" / "core" / PACKAGED_SUBDIR)
    assert identity.canonical == f"{URL}@{sdist_tree['head']}#{REST2_PATH}"


def test_wheel_from_a_modified_sdist_must_not_inherit_the_commit(sdist_tree, tmp_path):
    """The archive's word is only good while its catalog bytes still hash to what it described."""
    modified = tmp_path / "modified"
    shutil.copytree(sdist_tree["unpacked"], modified)
    descriptor = modified / REST2_PATH
    descriptor.write_text(descriptor.read_text() + "\n# edited after unpacking\n", encoding="utf-8")

    wheel = build_wheel(modified, tmp_path / "dist")
    target = install_wheel(wheel, tmp_path / "site")
    root = target / "md_templates" / "core" / PACKAGED_SUBDIR
    packaged = verify_packaged_catalog(root)
    assert packaged.provenance.resolved is False
    assert packaged.provenance.source_state == "no-verifiable-git-provenance"
    with pytest.raises(UnresolvedBuildProvenanceError):
        resolve_packaged_identity(REST2_ID, root=root)


def test_a_source_tree_with_neither_git_nor_an_archive_record_is_unresolved(tmp_path):
    tree = make_source_tree(tmp_path / "bare")
    wheel = build_wheel(tree, tmp_path / "dist")
    target = install_wheel(wheel, tmp_path / "site")
    provenance = verify_packaged_catalog(
        target / "md_templates" / "core" / PACKAGED_SUBDIR).provenance
    assert provenance.resolved is False
    assert provenance.source_state == "no-verifiable-git-provenance"


# ================================================================================================
# review finding 1: provenance must never be inherited from an enclosing repository
#
# `git -C <dir>` walks upwards. An unpacked source tree dropped anywhere inside an unrelated
# repository would otherwise report THAT repository's HEAD -- and report it clean, because the
# enclosing tree genuinely is clean while ignoring the copy. The identity would name a commit from a
# different project.
# ================================================================================================

@pytest.fixture(scope="module")
def enclosed_install(tmp_path_factory):
    """A source tree unpacked into an ignored directory of an unrelated, clean Git repository."""
    base = tmp_path_factory.mktemp("enclosed")
    outer = base / "unrelated-project"
    outer.mkdir()
    (outer / "README.md").write_text("an unrelated project\n", encoding="utf-8")
    (outer / ".gitignore").write_text("vendor/\n", encoding="utf-8")

    git(base, "init", "--quiet", str(outer))
    git(outer, "config", "user.email", "test@example.invalid")
    git(outer, "config", "user.name", "Unrelated Project")
    git(outer, "config", "commit.gpgsign", "false")
    git(outer, "add", "-A")
    git(outer, "commit", "--quiet", "-m", "unrelated project")

    vendored = make_source_tree(outer / "vendor" / "md-templates")
    # the decisive setup: the enclosing repository is CLEAN, because it ignores the vendored copy
    assert git(outer, "status", "--porcelain").stdout.strip() == ""

    wheel = build_wheel(vendored, base / "dist")
    target = install_wheel(wheel, base / "site")
    return {"outer": outer, "outer_head": git(outer, "rev-parse", "HEAD").stdout.strip(),
            "vendored": vendored, "target": target,
            "packaged": target / "md_templates" / "core" / PACKAGED_SUBDIR}


def test_f1_git_reports_the_enclosing_repository_from_the_vendored_directory(enclosed_install):
    """The hazard is real: plain `git -C` does resolve to the outer repository from there."""
    result = subprocess.run(["git", "-C", str(enclosed_install["vendored"]), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert result.stdout.strip() == enclosed_install["outer_head"]


def test_f1_inspect_provenance_refuses_a_directory_that_is_not_the_worktree_root(enclosed_install):
    from md_templates.core import inspect_provenance

    provenance = inspect_provenance(enclosed_install["vendored"])
    assert provenance.commit_sha is None
    assert provenance.resolved is False
    assert "not the root of a Git worktree" in provenance.detail
    assert provenance.worktree_root is not None


def test_f1_source_identity_refuses_an_enclosed_checkout(enclosed_install):
    from md_templates.core import NoProvenanceError, load_catalog, resolve_identity

    catalog = load_catalog(enclosed_install["vendored"])
    with pytest.raises(NoProvenanceError):
        resolve_identity(catalog, REST2_ID)


def test_f1_a_wheel_built_there_does_not_inherit_the_outer_commit(enclosed_install):
    packaged = verify_packaged_catalog(enclosed_install["packaged"])
    assert packaged.provenance.resolved is False
    assert packaged.provenance.source_state == "no-verifiable-git-provenance"
    assert packaged.provenance.commit_sha is None

    blob = (enclosed_install["packaged"] / BUILD_PROVENANCE_FILENAME).read_bytes()
    assert enclosed_install["outer_head"].encode() not in blob

    with pytest.raises(UnresolvedBuildProvenanceError):
        resolve_packaged_identity(REST2_ID, root=enclosed_install["packaged"])


# ================================================================================================
# review finding 2: archive provenance must bind the implementation, not just the catalog
#
# Binding registry.yaml and the descriptors alone leaves the code that reads and verifies them
# unbound: unpack, rewrite the loader or the build hook, rebuild, and the wheel would inherit a
# commit that never contained that code.
# ================================================================================================

@pytest.mark.parametrize("relative,edit", [
    ("src/md_templates/core/packaged.py", "\n# tampered loader\n"),
    ("src/md_templates/openmm/bundlev2.py", "\n# tampered engine module\n"),
    ("build_support/catalog.py", "\n# tampered build hook\n"),
])
def test_f2_editing_any_relevant_source_in_an_archive_breaks_inheritance(sdist_tree, tmp_path,
                                                                        relative, edit):
    modified = tmp_path / "modified"
    shutil.copytree(sdist_tree["unpacked"], modified)
    target = modified / relative
    assert target.is_file(), target
    target.write_text(target.read_text() + edit, encoding="utf-8")

    wheel = build_wheel(modified, tmp_path / "dist")
    installed = install_wheel(wheel, tmp_path / "site")
    root = installed / "md_templates" / "core" / PACKAGED_SUBDIR

    packaged = verify_packaged_catalog(root)
    assert packaged.provenance.resolved is False, relative
    assert packaged.provenance.source_state == "no-verifiable-git-provenance"
    assert sdist_tree["head"].encode() not in (root / BUILD_PROVENANCE_FILENAME).read_bytes()
    with pytest.raises(UnresolvedBuildProvenanceError):
        resolve_packaged_identity(REST2_ID, root=root)


# ================================================================================================
# review finding 3: the supported gate must require resolved provenance, not accept either branch
# ================================================================================================

GATE = REPO_ROOT / "scripts" / "ci" / "check_packaged_catalog.py"


def run_gate(target: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(target)
    return subprocess.run([sys.executable, str(GATE), *args],
                          capture_output=True, text=True, timeout=900, cwd=str(cwd), env=env)


def test_f3_the_gate_fails_on_an_unresolved_build_by_default(dirty_install):
    result = run_gate(dirty_install["target"], dirty_install["elsewhere"])
    assert result.returncode != 0, result.stdout
    assert "build provenance is unresolved" in result.stdout + result.stderr
    assert "dirty-source-tree" in result.stdout + result.stderr


def test_f3_local_development_mode_must_be_selected_explicitly(dirty_install):
    result = run_gate(dirty_install["target"], dirty_install["elsewhere"], "--allow-unresolved")
    assert result.returncode == 0, result.stderr
    assert "local-development" in result.stdout
    assert "identity refused" in result.stdout


def test_f3_the_gate_passes_on_a_clean_build_with_the_matching_commit(clean_install):
    result = run_gate(clean_install["target"], clean_install["elsewhere"],
                      "--expect-commit", clean_install["head"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mode:               strict" in result.stdout
    assert "(matches)" in result.stdout
    assert clean_install["head"] in result.stdout


def test_f3_the_gate_fails_when_the_wheel_was_built_from_a_different_commit(clean_install):
    """A stale wheel next to a moved checkout is the realistic version of this."""
    other = "0" * 39 + "1"
    result = run_gate(clean_install["target"], clean_install["elsewhere"],
                      "--expect-commit", other)
    assert result.returncode != 0
    assert "built from different source" in result.stdout + result.stderr


def test_f3_the_gate_rejects_a_malformed_expected_commit(clean_install):
    result = run_gate(clean_install["target"], clean_install["elsewhere"],
                      "--expect-commit", "4d21838")
    assert result.returncode != 0
    assert "not a full 40-character hex SHA" in result.stdout + result.stderr


def test_f3_the_shipped_fast_gate_does_not_hardcode_the_development_escape():
    script = (REPO_ROOT / "scripts" / "ci" / "fast_checks.sh").read_text()
    assert "--expect-commit" in script
    assert "FAST_CHECKS_ALLOW_UNRESOLVED:-0" in script, "the escape must default to off"
    assert "--allow-unresolved" in script
    # the flag may only be added inside the opt-in branch
    unconditional = [line for line in script.splitlines()
                     if "--allow-unresolved" in line and "FAST_CHECKS_ALLOW_UNRESOLVED" not in line]
    assert all("PACKAGED_ARGS+=" in line for line in unconditional), unconditional


def test_f2_the_archive_record_carries_a_source_tree_digest(sdist_tree):
    from md_templates.core.resources import SOURCE_PROVENANCE_FILENAME

    record = json.loads((sdist_tree["unpacked"] / SOURCE_PROVENANCE_FILENAME).read_text())
    assert len(record["source_tree_sha256"]) == 64
    assert record["source_tree_sha256"] != record["resource_manifest_sha256"]


def test_f2_the_source_digest_covers_more_than_the_catalog(sdist_tree, tmp_path):
    """It must move when implementation code moves, not only when a descriptor does."""
    from build_support.catalog import source_tree_digest

    baseline = source_tree_digest(sdist_tree["unpacked"])
    copy = tmp_path / "copy"
    shutil.copytree(sdist_tree["unpacked"], copy)
    assert source_tree_digest(copy) == baseline

    edited = copy / "src" / "md_templates" / "core" / "packaged.py"
    edited.write_text(edited.read_text() + "\n# moved\n", encoding="utf-8")
    assert source_tree_digest(copy) != baseline


def test_f2_build_artifacts_do_not_disturb_the_source_digest(sdist_tree, tmp_path):
    """`.egg-info` and `__pycache__` appear during a build; the digest must ignore them."""
    from build_support.catalog import source_tree_digest

    copy = tmp_path / "copy"
    shutil.copytree(sdist_tree["unpacked"], copy)
    baseline = source_tree_digest(copy)

    (copy / "src" / "md_templates.egg-info").mkdir(parents=True, exist_ok=True)
    (copy / "src" / "md_templates.egg-info" / "SOURCES.txt").write_text("x\n", encoding="utf-8")
    (copy / "src" / "md_templates" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (copy / "src" / "md_templates" / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\x00")
    assert source_tree_digest(copy) == baseline


# ================================================================================================
# 7. single source of truth
# ================================================================================================

def test_7_no_second_tracked_catalog_exists_under_src():
    """One editable catalog. Two would be edited independently exactly once."""
    tracked = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"],
                             capture_output=True, text=True, timeout=120).stdout.split()
    registries = [p for p in tracked if p.endswith("registry.yaml")]
    descriptors = [p for p in tracked if p.endswith("template.yaml")]
    assert registries == ["registry.yaml"], registries
    assert sorted(descriptors) == sorted([
        f"templates/{MD_ID}/template.yaml", f"templates/{REST2_ID}/template.yaml"]), descriptors
    assert not any(p.startswith("src/") and PACKAGED_SUBDIR in p for p in tracked)


def test_7b_the_staged_copy_is_never_written_into_the_source_tree():
    assert not (REPO_ROOT / "src" / "md_templates" / "core" / PACKAGED_SUBDIR).exists()
