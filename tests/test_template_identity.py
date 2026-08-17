"""Immutable template identity: full commit SHA plus exact template path.

Every Git case runs in an isolated temporary repository created by this file. The developer's own
checkout is never inspected, never made dirty, and never cleaned — so `test_28` means the same thing
whether or not there is uncommitted work in progress, which is exactly when a dirty-state test is
most likely to be wrong.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from md_templates.core import (
    DirtyWorkingTreeError,
    IdentityError,
    NoProvenanceError,
    TemplateIdentity,
    TrustedProvenance,
    UnknownTemplateError,
    build_identity,
    inspect_provenance,
    load_catalog,
    normalise_commit_sha,
    resolve_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/csy0000/MD-templates"
MD_ID = "conventional-md/openmm/explicit-water"
REST2_ID = "rest2/openmm/explicit-water"
MD_PATH = f"templates/{MD_ID}/template.yaml"
REST2_PATH = f"templates/{REST2_ID}/template.yaml"

SHA_A = "4d21838009804a47047348e80c8ba352e3d8539e"
SHA_B = "1e80a2061613ac40a10143c28044c661e1d5d4ac"

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(REPO_ROOT)


# ------------------------------------------------------------------------------------------------
# isolated temporary repositories
# ------------------------------------------------------------------------------------------------

def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(root), *args],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result


def make_catalog_tree(root: Path) -> None:
    """Copy the shipped catalog files, plus stand-ins for everything they reference."""
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    (root / "registry.yaml").write_bytes((REPO_ROOT / "registry.yaml").read_bytes())
    for tid in (MD_ID, REST2_ID):
        src = REPO_ROOT / "templates" / tid / "template.yaml"
        dst = root / "templates" / tid / "template.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        for ref in yaml.safe_load(src.read_text()).get("repository_references", []):
            target = root / ref
            if (REPO_ROOT / ref).is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    """A temporary Git repository containing a committed copy of the catalog."""
    root = tmp_path / "clean"
    make_catalog_tree(root)
    git(root.parent, "init", "--quiet", str(root))
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Catalog Test")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "catalog")
    return root


@pytest.fixture
def dirty_repo(clean_repo: Path) -> Path:
    """The descriptor is edited but still valid — the interesting case.

    A dirty tree whose descriptor no longer parses fails for an obvious reason. A dirty tree whose
    descriptor validates perfectly is the one where silently resolving to HEAD would look right and
    be wrong, so that is what the refusal has to catch.
    """
    path = clean_repo / "templates" / REST2_ID / "template.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited, not committed\n",
                    encoding="utf-8")
    return clean_repo


# ================================================================================================
# 22-26. construction and normalisation
# ================================================================================================

def test_22_same_sha_and_path_is_deterministic():
    a = build_identity(URL, SHA_A, REST2_PATH)
    b = build_identity(URL, SHA_A.upper(), REST2_PATH)
    assert a == b
    assert a.canonical == b.canonical == f"{URL}@{SHA_A}#{REST2_PATH}"
    assert a.as_dict() == b.as_dict()
    assert hash(a) == hash(b)


def test_22b_structured_fields_are_preserved_alongside_the_string():
    identity = build_identity(URL, SHA_A, REST2_PATH)
    assert isinstance(identity, TemplateIdentity)
    assert identity.canonical_url == URL
    assert identity.commit_sha == SHA_A
    assert identity.template_path == REST2_PATH
    assert identity.as_dict()["scheme"] == "git-commit-plus-template-path"


def test_23_different_sha_changes_identity():
    assert build_identity(URL, SHA_A, REST2_PATH) != build_identity(URL, SHA_B, REST2_PATH)


def test_24_different_template_path_changes_identity():
    assert build_identity(URL, SHA_A, REST2_PATH) != build_identity(URL, SHA_A, MD_PATH)


@pytest.mark.parametrize("bad", ["4d21838", "4d218380098", SHA_A[:39], SHA_A + "a"])
def test_25_abbreviated_or_overlong_sha_fails(bad):
    with pytest.raises(IdentityError, match="full 40-character"):
        normalise_commit_sha(bad)


@pytest.mark.parametrize("bad", [
    "z" * 40,
    "4d21838009804a47047348e80c8ba352e3d8539g",
    "main",
    "v0.1.0",
    "0.1.0",
    "2026-08-17",
    "",
    None,
    123,
])
def test_26_non_hex_or_non_sha_identity_fails(bad):
    with pytest.raises(IdentityError):
        normalise_commit_sha(bad)


def test_sha_is_normalised_to_lowercase_and_stripped():
    assert normalise_commit_sha(f"  {SHA_A.upper()}\n") == SHA_A


@pytest.mark.parametrize("bad", [
    "/templates/rest2/openmm/explicit-water/template.yaml",
    "templates/rest2/openmm/../openmm/explicit-water/template.yaml",
    "templates/./rest2/openmm/explicit-water/template.yaml",
    "templates\\rest2\\openmm\\explicit-water\\template.yaml",
    "../template.yaml",
])
def test_path_is_normalised_before_identity_is_constructed(bad):
    """Rejected, never repaired: two spellings must not collapse into one identity."""
    with pytest.raises(Exception) as exc:
        build_identity(URL, SHA_A, bad)
    assert "template_path" in str(exc.value)


def test_27_unregistered_template_path_fails(catalog):
    with pytest.raises(UnknownTemplateError, match="not a registered template"):
        resolve_identity(catalog, "templates/rest2/openmm/implicit-water/template.yaml",
                         commit_sha=SHA_A)
    with pytest.raises(UnknownTemplateError):
        resolve_identity(catalog, "umbrella-sampling/gromacs/explicit-water", commit_sha=SHA_A)


def test_registered_template_resolves_by_id_or_by_path(catalog):
    by_id = resolve_identity(catalog, REST2_ID, commit_sha=SHA_A)
    by_path = resolve_identity(catalog, REST2_PATH, commit_sha=SHA_A)
    assert by_id == by_path
    assert by_id.canonical == f"{URL}@{SHA_A}#{REST2_PATH}"


# ================================================================================================
# 28-30. provenance policy, in isolated repositories
# ================================================================================================

def test_29_clean_temporary_checkout_resolves_its_full_sha(clean_repo):
    catalog = load_catalog(clean_repo)
    expected = git(clean_repo, "rev-parse", "HEAD").stdout.strip()
    assert len(expected) == 40

    provenance = inspect_provenance(clean_repo)
    assert provenance.dirty is False and provenance.resolved is True
    assert provenance.commit_sha == expected

    identity = resolve_identity(catalog, REST2_ID)
    assert identity.commit_sha == expected
    assert identity.canonical == f"{URL}@{expected}#{REST2_PATH}"


def test_28_dirty_temporary_checkout_cannot_resolve_immutable_identity(dirty_repo):
    catalog = load_catalog(dirty_repo)  # loads the COMMITTED registry; the edit is to a descriptor
    head = git(dirty_repo, "rev-parse", "HEAD").stdout.strip()

    provenance = inspect_provenance(dirty_repo)
    assert provenance.dirty is True
    assert provenance.resolved is False
    assert provenance.commit_sha == head, "HEAD is still reported for development inspection"

    with pytest.raises(DirtyWorkingTreeError) as exc:
        resolve_identity(catalog, MD_ID)
    message = str(exc.value)
    assert "uncommitted" in message
    assert head in message, "the refusal names HEAD rather than hiding it"


def test_28b_an_untracked_file_also_makes_the_tree_dirty(clean_repo):
    (clean_repo / "templates" / "scratch-notes.txt").write_text("wip\n", encoding="utf-8")
    assert inspect_provenance(clean_repo).dirty is True
    with pytest.raises(DirtyWorkingTreeError):
        resolve_identity(load_catalog(clean_repo), MD_ID)


def test_28c_committing_the_change_resolves_again(dirty_repo):
    git(dirty_repo, "checkout", "--", ".")
    (dirty_repo / "NOTES.md").write_text("note\n", encoding="utf-8")
    git(dirty_repo, "add", "-A")
    git(dirty_repo, "commit", "--quiet", "-m", "note")
    head = git(dirty_repo, "rev-parse", "HEAD").stdout.strip()
    assert resolve_identity(load_catalog(dirty_repo), MD_ID).commit_sha == head


def test_30_non_git_directory_without_provenance_fails_clearly(tmp_path):
    root = tmp_path / "installed"
    make_catalog_tree(root)
    catalog = load_catalog(root)

    provenance = inspect_provenance(root)
    assert provenance.commit_sha is None and provenance.resolved is False

    with pytest.raises(NoProvenanceError) as exc:
        resolve_identity(catalog, MD_ID)
    assert "no Git metadata" in str(exc.value)
    assert "package version" in str(exc.value), "the refusal names what will NOT be substituted"


def test_30b_explicit_trusted_provenance_resolves_without_git(tmp_path):
    """The hook a packaged catalog needs in PR 2: a full SHA supplied by build metadata."""
    root = tmp_path / "installed"
    make_catalog_tree(root)
    catalog = load_catalog(root)
    identity = resolve_identity(catalog, MD_ID, trusted=TrustedProvenance(commit_sha=SHA_A))
    assert identity.canonical == f"{URL}@{SHA_A}#{MD_PATH}"


def test_30c_trusted_provenance_must_still_be_a_full_sha(tmp_path):
    root = tmp_path / "installed"
    make_catalog_tree(root)
    catalog = load_catalog(root)
    for bad in ("0.1.0", "4d21838", "main"):
        with pytest.raises(IdentityError, match="trusted.commit_sha"):
            resolve_identity(catalog, MD_ID, trusted=TrustedProvenance(commit_sha=bad))


def test_dirty_state_is_assumed_when_it_cannot_be_disproved(clean_repo, monkeypatch):
    """A `git status` that fails must not be read as 'clean'."""
    import md_templates.core.identity as identity_mod

    real = identity_mod._git

    def fake(root, *args):
        if args and args[0] == "status":
            return subprocess.CompletedProcess(args, 128, "", "fatal: broken index")
        return real(root, *args)

    monkeypatch.setattr(identity_mod, "_git", fake)
    assert inspect_provenance(clean_repo).dirty is True
    with pytest.raises(DirtyWorkingTreeError):
        resolve_identity(load_catalog(clean_repo), MD_ID)


def test_git_is_invoked_without_a_shell_and_with_a_timeout(clean_repo, monkeypatch):
    import md_templates.core.identity as identity_mod

    seen = []
    real_run = subprocess.run

    def spy(args, **kwargs):
        seen.append((args, kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(identity_mod.subprocess, "run", spy)
    inspect_provenance(clean_repo)

    assert seen, "no git call was made"
    for args, kwargs in seen:
        assert isinstance(args, list), "arguments must be a list, never a shell string"
        assert kwargs.get("shell") is False
        assert kwargs.get("timeout") == identity_mod.GIT_TIMEOUT_SECONDS
        assert kwargs.get("check") is False


# ================================================================================================
# 31. no network
# ================================================================================================

def test_31_identity_resolution_makes_no_network_request(clean_repo):
    """Run in a subprocess with sockets disabled: validation must work offline.

    An offline compute node and a locked-down runner are exactly where reproducibility matters, so a
    check that needed GitHub to be reachable would fail where it is most needed. The canonical URL
    is a name, not an endpoint.
    """
    code = r"""
import socket, sys

class Blocked(RuntimeError):
    pass

def refuse(*a, **k):
    raise Blocked("identity resolution attempted a network operation")

socket.socket = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse

from md_templates.core import load_catalog, resolve_identity, inspect_provenance
root = sys.argv[1]
print(inspect_provenance(__import__("pathlib").Path(root)).commit_sha)
print(resolve_identity(load_catalog(root), "rest2/openmm/explicit-water").canonical)
"""
    result = subprocess.run([sys.executable, "-c", code, str(clean_repo)],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    head = git(clean_repo, "rev-parse", "HEAD").stdout.strip()
    assert result.stdout.strip().splitlines() == [head, f"{URL}@{head}#{REST2_PATH}"]


# ================================================================================================
# scope: PR 1 does not persist identity anywhere
# ================================================================================================

def test_identity_is_not_written_into_bundles_or_runs():
    """The catalog is metadata in PR 1. Nothing in the runtime references it."""
    runtime = REPO_ROOT / "src" / "md_templates" / "openmm"
    offenders = []
    for path in sorted(runtime.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "md_templates.core" in text or "from ..core" in text or "template_identity" in text:
            offenders.append(path.relative_to(REPO_ROOT))
    assert offenders == [], f"runtime modules reference the catalog: {offenders}"
