"""Exercise the packaged catalog of the INSTALLED distribution.

Run from a directory outside the repository, against a wheel that has been installed:

    cd "$WORKDIR" && python "$REPO_ROOT/scripts/ci/check_packaged_catalog.py" \
        --expect-commit "$(git -C "$REPO_ROOT" rev-parse HEAD)"

**Resolved provenance is required by default, and must match `--expect-commit` exactly.** A gate that
accepted an unresolved build would pass on precisely the wheels that cannot name their own source,
which is the failure it exists to catch -- and it would do so silently, because an unresolved build
is otherwise indistinguishable from a healthy one until someone asks for an identity.

`--allow-unresolved` exists for local development from a dirty tree. It must be selected explicitly;
CI never sets it, and the script says which mode it is in on every run.

Nothing here reads the checkout. `sys.path[0]` is this script's own directory, so `md_templates`
resolves to the installed package; a catalog that only worked next to its source would not be
packaged at all. Sockets are replaced by raising stubs before the first import, because a catalog
that needed the network would fail exactly where reproducibility matters most.

Exits non-zero with a specific message on the first thing that does not hold.
"""
from __future__ import annotations

import argparse
import pathlib
import socket
import sys


def _refuse(*args, **kwargs):
    raise RuntimeError("the packaged catalog attempted a network operation")


socket.socket = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse

import md_templates  # noqa: E402
from md_templates.core.packaged import (  # noqa: E402
    UnresolvedBuildProvenanceError,
    load_packaged_catalog,
    load_packaged_provenance,
    packaged_catalog_root,
    resolve_packaged_identity,
)

EXPECTED = ["conventional-md/openmm/explicit-water", "rest2/openmm/explicit-water"]


def fail(message: str) -> None:
    sys.exit(f"packaged-catalog gate FAILED: {message}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", metavar="SHA",
                        help="require the packaged build provenance to name exactly this full "
                             "40-character commit SHA")
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="local development only: accept an unresolved build and check that it "
                             "refuses identity. CI must never pass this.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    package = pathlib.Path(md_templates.__file__).resolve()
    repo_marker = pathlib.Path(__file__).resolve().parents[2] / "src"
    if str(repo_marker) in str(package):
        fail(f"md_templates resolves into the checkout ({package}); install the wheel first")
    print(f"  package:            {package}")
    print(f"  packaged catalog:   {packaged_catalog_root()}")

    provenance = load_packaged_provenance()
    print(f"  provenance schema:  {provenance.schema_version}")
    print(f"  source state:       {provenance.source_state} (resolved={provenance.resolved})")

    catalog = load_packaged_catalog()
    listed = sorted(catalog.descriptors)
    if listed != EXPECTED:
        fail(f"expected {EXPECTED}, got {listed}")
    print(f"  templates listed:   {listed}")
    print(f"  registry schema:    {catalog.registry.schema_version}")
    print(f"  reference policy:   {catalog.reference_policy}")

    mode = "local-development (unresolved builds accepted)" if args.allow_unresolved else "strict"
    print(f"  mode:               {mode}")

    if not provenance.resolved:
        if not args.allow_unresolved:
            fail(
                f"build provenance is unresolved ({provenance.source_state}); the supported gate "
                f"requires a wheel built from a clean checkout that can name its own commit. Build "
                f"from a committed tree, or pass --allow-unresolved for local development."
            )
        # Explicitly selected development mode: the refusal itself is what gets checked.
        try:
            resolve_packaged_identity(listed[0])
        except UnresolvedBuildProvenanceError:
            print(f"  identity refused:   {provenance.source_state} (expected in this mode)")
            return 0
        fail("unresolved build provenance produced an identity")

    if args.expect_commit:
        expected = args.expect_commit.strip().lower()
        if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
            fail(f"--expect-commit {args.expect_commit!r} is not a full 40-character hex SHA")
        if provenance.commit_sha != expected:
            fail(f"packaged provenance names {provenance.commit_sha}, but the checkout under test "
                 f"is at {expected}. The wheel was built from different source.")
        print(f"  expected commit:    {expected} (matches)")
    else:
        print("  expected commit:    not supplied (pass --expect-commit in CI)")

    for template_id in listed:
        identity = resolve_packaged_identity(template_id)
        if identity.commit_sha != provenance.commit_sha:
            fail(f"{template_id} resolved to {identity.commit_sha}, "
                 f"not the build commit {provenance.commit_sha}")
        expected_path = f"templates/{template_id}/template.yaml"
        if identity.template_path != expected_path:
            fail(f"{template_id} used {identity.template_path!r}, not the logical root path "
                 f"{expected_path!r}")
        for banned in ("_packaged", ".whl", "site-packages"):
            if banned in identity.canonical:
                fail(f"a distribution path leaked into an identity: {identity.canonical}")
        print(f"  identity:           {identity.canonical}")

    heavy = sorted(m for m in sys.modules
                   if m.split(".")[0] in {"openmm", "openff", "rdkit", "mdtraj", "parmed",
                                          "openmmtools", "numpy", "scipy", "pandas"})
    if heavy:
        fail(f"the packaged catalog pulled in heavy dependencies: {heavy}")
    print("  ok: offline, no heavy dependencies, no checkout, no Git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
