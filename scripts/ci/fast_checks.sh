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
    "profiles":            "md_templates/openmm/spec/profiles/",
    "system manifests":    "md_templates/openmm/manifests/systems/",
    "experiment manifests":"md_templates/openmm/manifests/experiments/",
    "peptide structure":   "manifests/systems/ace_ala_nme.pdb",
    "spec package":        "md_templates/openmm/spec/models.py",
    "bundle contract":     "md_templates/openmm/bundlev2.py",
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
