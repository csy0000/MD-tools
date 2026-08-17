#!/usr/bin/env bash
# Fast repository gate. Runs identically here and in GitHub Actions -- the workflow calls this
# script rather than repeating its steps, so "what CI does" cannot drift from "what I ran".
#
#   scripts/ci/fast_checks.sh [WORKDIR]
#
# Requires: an environment with the package's runtime dependencies (see
# docs/implementation/explicit_solvent/environment.yml) and `build`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-$(mktemp -d)}"
mkdir -p "$WORKDIR"

step() { printf '\n=== %s ===\n' "$1"; }

step "0. the expected commit"
# The supported gate REQUIRES resolved packaged provenance naming exactly this checkout's HEAD, so
# the SHA must be acquired BEFORE anything is built -- a checkout that cannot name itself should fail
# in a second, not after a wheel build and an install.
#
# FAST_CHECKS_ALLOW_UNRESOLVED=1 is the local-development escape: it permits an unresolved build and
# a missing SHA. It must be selected explicitly, it defaults to off, and it is announced here. CI
# must never set it.
ALLOW_UNRESOLVED="${FAST_CHECKS_ALLOW_UNRESOLVED:-0}"
EXPECT_SHA=""
if SHA_OUTPUT="$(bash "$REPO_ROOT/scripts/ci/expect_commit.sh" "$REPO_ROOT" 2>&1)"; then
    EXPECT_SHA="$SHA_OUTPUT"
    echo "  expected commit: $EXPECT_SHA"
elif [ "$ALLOW_UNRESOLVED" = "1" ]; then
    echo "  NOTE: no expected commit ($SHA_OUTPUT)"
else
    echo "fast checks FAILED: $SHA_OUTPUT" >&2
    echo "The supported gate requires a checkout whose full commit SHA can be determined, so the" >&2
    echo "packaged wheel can be checked against it. Run from a Git checkout with at least one" >&2
    echo "commit, or set FAST_CHECKS_ALLOW_UNRESOLVED=1 for local development." >&2
    exit 1
fi

if [ "$ALLOW_UNRESOLVED" = "1" ]; then
    echo "  NOTE: FAST_CHECKS_ALLOW_UNRESOLVED=1 -- local development mode, not the supported gate"
fi

step "1. build the wheel"
cd "$REPO_ROOT"
rm -rf dist
python -m build --wheel --outdir dist
WHEEL="$(ls dist/*.whl | head -1)"
echo "built: $WHEEL"

step "2. inspect wheel contents"
# A wheel missing its profiles ships a `config` command that cannot resolve anything, so the
# packaged resources are checked rather than assumed.
python - "$WHEEL" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
required = {
    # Profiles are template-local since Phase 5/6 and travel inside the packaged catalog.
    "profiles":            "_packaged/templates/conventional-md/openmm/explicit-water/profiles/",
    "rest2 profiles":      "_packaged/templates/rest2/openmm/explicit-water/profiles/",
    # The engine's shipped assets moved under engines/ in Phase 4.
    "system manifests":    "md_templates/engines/openmm/manifests/systems/",
    "experiment manifests":"md_templates/engines/openmm/manifests/experiments/",
    "peptide structure":   "manifests/systems/ace_ala_nme.pdb",
    # The canonical configuration package moved to core in Phase 3.
    "config package":      "md_templates/core/config/models.py",
    "bundle contract":     "md_templates/core/bundle.py",
    "engine provider":     "md_templates/engines/openmm/provider.py",
    "generic cli":         "md_templates/cli.py",
    # A wheel whose catalog is missing ships a packaged loader that cannot resolve anything, so the
    # catalog resources and BOTH metadata records are checked rather than assumed.
    "packaged registry":   "md_templates/core/_packaged/registry.yaml",
    "packaged templates":  "md_templates/core/_packaged/templates/",
    "resource manifest":   "md_templates/core/_packaged/resource_manifest.json",
    "build provenance":    "md_templates/core/_packaged/build_provenance.json",
}
missing = []
for label, pattern in required.items():
    hits = [n for n in names if pattern in n]
    print(f"  {label:22} {len(hits)} file(s)")
    if not hits:
        missing.append(label)
if missing:
    sys.exit(f"wheel is missing required package data: {missing}")
PY

step "3. install the wheel into this environment"
python -m pip uninstall -y md-templates >/dev/null 2>&1 || true
python -m pip install --no-deps "$WHEEL"

step "4. run public commands from OUTSIDE the checkout"
cd "$WORKDIR"
md-openmm --version
md-openmm --help >/dev/null
md-openmm config list-profiles >/dev/null
md-openmm bundle --help >/dev/null
echo "  ok: commands work with no checkout on the path"

step "4b. the packaged catalog, from outside the checkout"
# Still in WORKDIR: this exercises the INSTALLED distribution. The script sets no sys.path of its
# own, blocks sockets before importing, and refuses if md_templates resolves into the checkout.
#
# The expected commit was acquired in step 0, before anything was built, so a checkout that cannot
# name itself fails in a second rather than after a wheel build.
PACKAGED_ARGS=()
if [ -n "$EXPECT_SHA" ]; then
    PACKAGED_ARGS+=(--expect-commit "$EXPECT_SHA")
fi
if [ "$ALLOW_UNRESOLVED" = "1" ]; then
    PACKAGED_ARGS+=(--allow-unresolved)
fi
python "$REPO_ROOT/scripts/ci/check_packaged_catalog.py" "${PACKAGED_ARGS[@]}"

step "5. validate every shipped profile"
python - <<'PY'
from md_templates.openmm.spec import canonical, resolve
profiles = resolve.list_profiles()
assert profiles, "no profiles are packaged"
defaults = {}
for doc in profiles:
    assert doc["profile_id"] and doc["description"]
    assert isinstance(doc["profile_schema_version"], int)
    canonical.sha256_of({k: v for k, v in doc.items() if k != "_path"})
    if doc.get("is_default"):
        key = (doc["route"], doc["method"])
        assert key not in defaults, f"two defaults for {key}"
        defaults[key] = doc["profile_id"]
    print(f"  {doc['profile_id']:28} default={bool(doc.get('is_default'))}")
PY

step "6. YAML and JSON must produce identical canonical hashes"
python - <<'PY'
import json, tempfile, pathlib, yaml
from md_templates.openmm.spec import resolve
doc = {"system": {"system_id": "eq", "route": "smiles", "smiles": "CCO"},
       "protocol": {"production": {"method": "md", "n_chunks": 2, "chunk": "10 ps"}}}
d = pathlib.Path(tempfile.mkdtemp())
(d / "a.yaml").write_text(yaml.safe_dump(doc))
(d / "a.json").write_text(json.dumps(doc))
h1 = resolve.resolve_spec(resolve.load_document(d / "a.yaml"))["hashes"]
h2 = resolve.resolve_spec(resolve.load_document(d / "a.json"))["hashes"]
assert h1 == h2, f"YAML and JSON disagree:\n  {h1}\n  {h2}"
print(f"  identical across formats: {list(h1)}")
PY

step "7. non-slow test suite"
cd "$REPO_ROOT"
python -m pytest tests/ -q -m "not slow"

echo
echo "fast checks: PASSED"
