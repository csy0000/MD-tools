"""Immutable template identity: canonical URL, full commit SHA, exact template path.

A template is identified by the tuple

    (canonical repository URL, 40-character Git commit SHA, normalised template.yaml path)

whose canonical string form is

    https://github.com/csy0000/MD-templates@<40 hex>#templates/<method>/<engine>/<variant>/template.yaml

Nothing else is admitted as identity. A branch name moves, a tag can be re-pointed, an abbreviated
SHA is ambiguous by construction and grows more ambiguous as the repository does, a package version
covers many commits, a semantic template version is a human claim about compatibility rather than a
statement about content, and a timestamp identifies nothing. A commit SHA plus a path names exactly
one sequence of bytes forever.

**A dirty checkout has no immutable identity, and this module says so rather than approximating
one.** HEAD describes what was committed; the file on disk may be something else entirely. Silently
resolving to HEAD would stamp an artifact with an identity that does not reproduce it, which is
worse than having no identity at all -- the wrong answer is indistinguishable from the right one
later. `resolve_identity` raises `DirtyWorkingTreeError`, and `inspect_provenance` remains available
for development, clearly labelled unresolved.

Identity resolution never touches the network. It reads Git metadata from a local directory and
formats a string; it does not contact GitHub, and the canonical URL is a name, not an endpoint. A
validation step that needed GitHub to be reachable would fail in exactly the environments -- an
offline compute node, a locked-down runner -- where reproducibility matters most.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import assert_under_templates

__all__ = [
    "IdentityError", "DirtyWorkingTreeError", "NoProvenanceError", "UnknownTemplateError",
    "TemplateIdentity", "GitProvenance", "TrustedProvenance",
    "normalise_commit_sha", "build_identity", "inspect_provenance", "resolve_identity",
    "GIT_TIMEOUT_SECONDS",
]

#: Full SHA-1 object name. Abbreviations are refused however long they are: a prefix that is unique
#: today may collide after the next commit, so a stored short SHA is a latent ambiguity.
_FULL_SHA = re.compile(r"\A[0-9a-fA-F]{40}\Z")

#: Git is invoked without a shell, with an explicit argument list, and bounded. A hung `git` in a
#: broken checkout must fail the call, not the process.
GIT_TIMEOUT_SECONDS = 10.0


class IdentityError(ValueError):
    """An immutable template identity could not be constructed from what was supplied."""


class DirtyWorkingTreeError(IdentityError):
    """The checkout has uncommitted changes, so no commit describes the files on disk."""


class NoProvenanceError(IdentityError):
    """No Git metadata and no explicitly supplied trusted commit provenance."""


class UnknownTemplateError(IdentityError):
    """The requested path is not a registered template."""


# ------------------------------------------------------------------------------------------------
# the identity itself
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateIdentity:
    """The structured identity and its canonical string form.

    Both are kept. The string is what gets written into a report or quoted in an issue; the fields
    are what a consumer compares, so nobody has to parse the string back apart to ask "same
    repository, different commit?".
    """

    canonical_url: str
    commit_sha: str
    template_path: str

    @property
    def canonical(self) -> str:
        return f"{self.canonical_url}@{self.commit_sha}#{self.template_path}"

    def as_dict(self) -> dict:
        return {
            "scheme": "git-commit-plus-template-path",
            "canonical_url": self.canonical_url,
            "commit_sha": self.commit_sha,
            "template_path": self.template_path,
            "canonical": self.canonical,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.canonical


@dataclass(frozen=True)
class GitProvenance:
    """What a local checkout can say about itself.

    `commit_sha` is HEAD whether or not the tree is dirty, because a developer inspecting a branch
    still wants to see it. `resolved` is the field that decides whether an identity may be built.
    """

    commit_sha: Optional[str]
    dirty: bool
    source: str
    detail: str = ""
    #: False when `git status` failed or timed out. Such a tree is reported dirty -- one that cannot
    #: be shown clean has not been shown clean -- but a caller that must record *why* can tell the
    #: two apart without parsing `detail`.
    status_known: bool = True
    #: The Git worktree root that was found, when one was found but was NOT this directory. Set only
    #: in that case, which is the enclosing-repository refusal.
    worktree_root: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.commit_sha is not None and not self.dirty

    def as_dict(self) -> dict:
        return {"commit_sha": self.commit_sha, "dirty": self.dirty, "source": self.source,
                "resolved": self.resolved, "detail": self.detail,
                "status_known": self.status_known, "worktree_root": self.worktree_root}


@dataclass(frozen=True)
class TrustedProvenance:
    """A full commit SHA supplied by something other than a checkout.

    The hook a packaged catalog needs: a wheel built from a clean commit can record that commit in
    its build metadata and hand it here, and identity resolution then works with no `.git` present.
    PR 1 defines the shape and honours it; PR 2 is what actually writes such metadata into a wheel.
    Nothing infers this -- it is passed in explicitly, so an unprovenanced install cannot acquire an
    identity by accident.
    """

    commit_sha: str
    source: str = "explicit-trusted-provenance"


# ------------------------------------------------------------------------------------------------
# normalisation
# ------------------------------------------------------------------------------------------------

def normalise_commit_sha(value: object, *, field: str = "commit_sha") -> str:
    """Return the canonical lowercase 40-character hexadecimal form, or raise `IdentityError`."""
    if not isinstance(value, str):
        raise IdentityError(f"{field}: must be a string, got {type(value).__name__}")
    text = value.strip()
    if not _FULL_SHA.match(text):
        raise IdentityError(
            f"{field}: {value!r} is not a full 40-character hexadecimal Git commit SHA. "
            f"Abbreviated SHAs, branch names, tags, package versions and template semantic versions "
            f"are all refused as immutable identity."
        )
    return text.lower()


def build_identity(canonical_url: str, commit_sha: object, template_path: object) -> TemplateIdentity:
    """Construct an identity from parts, normalising the path **before** the identity is formed.

    Purely syntactic. It checks that the SHA is a full 40-hex string and that the path is a
    normalised template path; it does **not** and cannot check that the commit contains those bytes.
    Use it to parse, format and compare identities. To *mint* one for a working tree, use
    `resolve_identity`, which proves provenance instead of taking the caller's word for it.
    """
    path = assert_under_templates(
        template_path if isinstance(template_path, str) else str(template_path),
        field="template_path",
    )
    if not isinstance(canonical_url, str) or not canonical_url.strip():
        raise IdentityError("canonical_url: must be a non-empty string")
    return TemplateIdentity(canonical_url=canonical_url.strip(),
                            commit_sha=normalise_commit_sha(commit_sha),
                            template_path=path)


# ------------------------------------------------------------------------------------------------
# local Git provenance
# ------------------------------------------------------------------------------------------------

def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False, shell=False,
    )


def inspect_provenance(root: Path) -> GitProvenance:
    """Describe a directory's Git state without deciding whether it may be used as identity.

    Safe to call on a dirty tree; that is what it is for. `resolve_identity` applies the policy.
    """
    root = Path(root)
    if not root.is_dir():
        return GitProvenance(None, False, "absent", f"{root} is not a directory")

    # The worktree root must BE this directory, not merely contain it. `git -C <dir>` walks upwards,
    # so an unpacked source tree sitting anywhere inside an unrelated repository -- vendored, or
    # dropped in an ignored directory -- would otherwise report that repository's HEAD, and report it
    # clean, because the enclosing tree genuinely is clean while ignoring the copy. The identity would
    # then name a commit from a different project entirely. Checked before HEAD so the refusal says
    # what actually happened.
    try:
        toplevel = _git(root, "rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return GitProvenance(None, False, "absent", "git executable not found on PATH")
    except subprocess.TimeoutExpired:
        return GitProvenance(None, False, "absent",
                             f"git timed out after {GIT_TIMEOUT_SECONDS} s")
    if toplevel.returncode != 0:
        return GitProvenance(None, False, "absent",
                             (toplevel.stderr or "git rev-parse --show-toplevel failed").strip())

    worktree = toplevel.stdout.strip()
    if not worktree or os.path.realpath(worktree) != os.path.realpath(root):
        return GitProvenance(
            None, False, "absent",
            f"{root} is not the root of a Git worktree: it lies inside the repository at "
            f"{worktree or '<unknown>'}, whose commits describe different content. Provenance is "
            f"never inherited from an enclosing repository.",
            worktree_root=worktree or None,
        )

    try:
        head = _git(root, "rev-parse", "HEAD")
    except subprocess.TimeoutExpired:
        return GitProvenance(None, False, "absent",
                             f"git timed out after {GIT_TIMEOUT_SECONDS} s")
    if head.returncode != 0:
        return GitProvenance(None, False, "absent",
                             (head.stderr or "git rev-parse HEAD failed").strip())

    try:
        sha = normalise_commit_sha(head.stdout.strip(), field="git rev-parse HEAD")
    except IdentityError as exc:
        return GitProvenance(None, False, "absent", str(exc))

    try:
        status = _git(root, "status", "--porcelain")
    except subprocess.TimeoutExpired:
        return GitProvenance(sha, True, "git-checkout",
                             "git status timed out; treated as dirty", status_known=False)
    if status.returncode != 0:
        return GitProvenance(sha, True, "git-checkout",
                             (status.stderr or "git status failed").strip()
                             + "; treated as dirty because it could not be shown to be clean",
                             status_known=False)

    changed = [line for line in status.stdout.splitlines() if line.strip()]
    return GitProvenance(sha, bool(changed), "git-checkout",
                         f"{len(changed)} uncommitted path(s)" if changed else "clean",
                         worktree_root=worktree)


# ------------------------------------------------------------------------------------------------
# the policy
# ------------------------------------------------------------------------------------------------

def resolve_identity(catalog, template_ref: str, *,
                     root: Optional[Path] = None,
                     trusted: Optional[TrustedProvenance] = None) -> TemplateIdentity:
    """Resolve one registered template to its immutable identity, proving provenance.

    `template_ref` may be a `template_id` or a `template_path`; either way it must name a template
    the registry lists, so an identity is never minted for a file outside the catalog.

    **The caller cannot supply the commit.** There is deliberately no `commit_sha` parameter: an
    identity asserts "these bytes are in that commit", and a caller-supplied SHA asserts nothing.
    The earlier shortcut let a dirty checkout name its own HEAD and receive a resolved identity, and
    let a clean checkout be stamped with any unrelated 40-hex string — in both cases producing an
    identity that does not reproduce the files on disk. `build_identity` remains available for
    syntactic construction, where nothing is being claimed about a working tree.

    The commit therefore comes from exactly one place, decided by whether `root` is a Git checkout:

    * **In a checkout** the tree must be clean, and the commit is HEAD. If `trusted` is also given
      it must *equal* HEAD — it is a cross-check on build metadata, never an override.
    * **Outside a checkout** only explicitly trusted build provenance is accepted, and absent that
      this raises rather than inventing an identity.
    """
    entry = catalog.require(template_ref)
    where = Path(root) if root is not None else catalog.root
    provenance = inspect_provenance(where)

    if provenance.commit_sha is None:
        if trusted is None:
            raise NoProvenanceError(
                f"no Git metadata under {where} and no trusted commit provenance supplied, so "
                f"there is no immutable identity for {entry.template_id!r}. An installed copy must "
                f"carry an explicit full commit SHA from its build metadata; nothing is inferred "
                f"from a package version or a timestamp. ({provenance.detail})"
            )
        return build_identity(
            catalog.canonical_url,
            normalise_commit_sha(trusted.commit_sha, field="trusted.commit_sha"),
            entry.template_path,
        )

    if provenance.dirty:
        raise DirtyWorkingTreeError(
            f"the checkout at {where} has uncommitted changes ({provenance.detail}), so no commit "
            f"describes the files on disk and {entry.template_id!r} has no immutable identity. "
            f"HEAD is {provenance.commit_sha}, which is deliberately NOT used: stamping it here "
            f"would claim an identity that does not reproduce these files. Commit the tree, or "
            f"call inspect_provenance() for an explicitly unresolved development view."
        )

    if trusted is not None:
        declared = normalise_commit_sha(trusted.commit_sha, field="trusted.commit_sha")
        if declared != provenance.commit_sha:
            raise IdentityError(
                f"trusted.commit_sha {declared} does not match HEAD {provenance.commit_sha} at "
                f"{where}. Trusted provenance cross-checks a checkout; it never overrides one. A "
                f"mismatch means the build metadata and the tree describe different commits, and "
                f"which of them the files on disk belong to is exactly what is in doubt."
            )

    return build_identity(catalog.canonical_url, provenance.commit_sha, entry.template_path)
