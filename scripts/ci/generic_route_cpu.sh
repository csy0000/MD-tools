#!/usr/bin/env bash
# Phase 5 gate: the generic `md-templates` route and the legacy `md-openmm` route must produce the
# same scientific artifacts, and the generic route must work from an installed wheel outside the
# checkout.
#
#   scripts/ci/generic_route_cpu.sh [WORKDIR]
#
# CPU only, picoseconds only. This proves the ROUTES agree and that prepare/run/resume work; it is
# engineering evidence and not scientific validation of anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-$(mktemp -d -t md-generic-XXXX)}"
mkdir -p "$WORKDIR"/{generic,legacy}

step() { printf '\n=== %s ===\n' "$1"; }
fail() { echo "GENERIC ROUTE FAILED: $1" >&2; exit 1; }

step "0. the package under test is the INSTALLED wheel"
python -c "import md_templates, pathlib, sys
p = pathlib.Path(md_templates.__file__).resolve()
if '$REPO_ROOT/src' in str(p):
    sys.exit(f'md_templates resolves to the checkout ({p}); install the wheel first')
print(f'  package: {p}')"

# The same document for both routes. Tiny on purpose: this measures agreement, not physics.
# No `profile:` line: the conventional-MD default profile is what this gate should exercise, and it
# is now template-local. cpu-smoke-v1 is a REST2 profile and naming it here is refused -- correctly,
# and that refusal is itself covered by the unit tests.
read -r -d '' CONFIG <<'YAML' || true
system: {system_id: ggg, route: smiles, smiles: "O=C1CNC(=O)CNC(=O)CN1"}
protocol:
  # The template-local default profile supplies the force field, water model, integrator, cutoff,
  # constraints and HMR -- every scientific setting. Only the equilibration SCHEDULE is shortened:
  # the default `staged` protocol runs a 200 ps ramp, 200 ps restrained NPT, four 200 ps release
  # stages and 1 ns of free NPT, which is right for production and hopeless for a CPU gate. Both
  # routes read the same document, so the comparison is unaffected by the shortening.
  equilibration:
    protocol: simple
    minimize_max_iterations: 200
    timestep: 1 fs
    nvt: 2 ps
    npt: 2 ps
    npt_free: 2 ps
    box_average_last: 1 ps
  production:
    method: md
    n_chunks: 2
    chunk: 0.001 ns
execution:
  platform: CPU
  # 1 ps chunks need reporting intervals that divide 1 ps, or the engine refuses -- frames would not
  # align to chunk boundaries and the first frame of each chunk would drift. Refusing is right; the
  # gate just has to ask for something coherent.
  report:
    all_atom_ps: 0.5
    solute_ps: 0.5
    log_ps: 0.5
YAML
printf '%s\n' "$CONFIG" > "$WORKDIR/generic/sim.yaml"
printf '%s\n' "$CONFIG" > "$WORKDIR/legacy/sim.yaml"

step "1. prepare through BOTH routes"
cd "$WORKDIR/generic"
md-templates prepare --template conventional-md/openmm/explicit-water -- \
    --config sim.yaml --out-root ./runs --platform CPU
cd "$WORKDIR/legacy"
md-openmm prepare --config sim.yaml --out-root ./runs --platform CPU

GENERIC_BUNDLE="$(find "$WORKDIR/generic/runs" -maxdepth 1 -name '*bundle*' | head -1)"
LEGACY_BUNDLE="$(find "$WORKDIR/legacy/runs" -maxdepth 1 -name '*bundle*' | head -1)"
[ -n "$GENERIC_BUNDLE" ] || fail "the generic route produced no bundle"
[ -n "$LEGACY_BUNDLE" ] || fail "the legacy route produced no bundle"

step "2. the two bundles must carry identical canonical hashes"
python - "$GENERIC_BUNDLE" "$LEGACY_BUNDLE" <<'PY'
import json, sys, pathlib

generic, legacy = (json.loads((pathlib.Path(p) / "canonical_configuration.json").read_text())
                   for p in sys.argv[1:3])
gh, lh = generic["hashes"], legacy["hashes"]
for key in sorted(set(gh) | set(lh)):
    mark = "ok " if gh.get(key) == lh.get(key) else "DIFFERS"
    print(f"  {mark} {key}: {gh.get(key)}")
if gh != lh:
    sys.exit("the generic and legacy routes produced different canonical hashes")
if generic["configuration"] != legacy["configuration"]:
    sys.exit("the generic and legacy routes produced different canonical configurations")
if generic["profile"] != legacy["profile"]:
    sys.exit("the two routes selected different profiles")
if generic["sources"] != legacy["sources"]:
    sys.exit("the two routes attributed values to different layers")
print(f"  identical canonical configuration, profile "
      f"({generic['profile']['profile_id']}) and source attribution")
print("  identical on all five projection hashes")
PY

step "3. run and resume through the GENERIC route"
cd "$WORKDIR/generic"
md-templates run --template conventional-md/openmm/explicit-water -- \
    --bundle "$GENERIC_BUNDLE" --out-root ./runs --run-name m1 --platform CPU
md-templates resume --template conventional-md/openmm/explicit-water -- \
    --bundle "$GENERIC_BUNDLE" --out-root ./runs --resume-run m1 --platform CPU

step "4. the resumed run is contiguous and committed"
python - "$WORKDIR/generic/runs/m1" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1])
committed = json.loads((run / "restart" / "committed.json").read_text())
print(f"  committed generation: {committed.get('generation')}")
state = json.loads((run / "run_state.json").read_text())
invocations = state.get("invocations", [])
print(f"  invocations recorded: {len(invocations)}")
if len(invocations) < 2:
    sys.exit("resume did not record a second invocation")
csv = run / "md_chunks.csv"
if csv.is_file():
    rows = [r for r in csv.read_text().splitlines() if r.strip()]
    headers = [r for r in rows if r.lower().startswith("chunk")]
    print(f"  chunk rows: {len(rows) - len(headers)}, headers: {len(headers)}")
    if len(headers) != 1:
        sys.exit("the resumed run wrote more than one CSV header")
PY

step "5. resume refuses to guess which run to continue"
cd "$WORKDIR/generic"
if md-templates resume --template conventional-md/openmm/explicit-water -- \
       --bundle "$GENERIC_BUNDLE" --out-root ./runs --platform CPU 2>/dev/null; then
    fail "resume without --resume-run should have been refused"
fi
echo "  ok: refused without --resume-run"

echo
echo "generic route: PASSED"
