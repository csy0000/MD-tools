#!/usr/bin/env bash
# Portable CPU integration gate: prepare, relocate, validate, run and resume BOTH canonical routes
# from a wheel-installed package, outside the checkout, with the original sources deleted.
#
#   scripts/ci/integration_cpu.sh [WORKDIR]
#
# CPU only. Nothing here assumes CUDA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-$(mktemp -d -t md-integration-XXXX)}"
mkdir -p "$WORKDIR"/{src_smiles,src_pdb,elsewhere}

step() { printf '\n=== %s ===\n' "$1"; }
fail() { echo "INTEGRATION FAILED: $1" >&2; exit 1; }

step "0. the package under test is the INSTALLED wheel, not the checkout"
python -c "import md_templates, pathlib, sys
p = pathlib.Path(md_templates.__file__).resolve()
if '$REPO_ROOT' in str(p):
    sys.exit(f'md_templates resolves to the checkout ({p}); install the wheel first')
print(f'  package: {p}')"

cat > "$WORKDIR/src_smiles/sim.yaml" <<'YAML'
profile: cpu-smoke-v1
system: {system_id: ggg, route: smiles, smiles: "O=C1CNC(=O)CNC(=O)CN1"}
protocol:
  production:
    method: rest2
    n_chunks: 2
    chunk: 0.001 ns
    scale_factors: [1.0, 0.5625, 0.25]
    exchange_interval: 0.5 ps
    relaxation: 1 ps
execution: {platform: CPU}
YAML

python -c "
import shutil, pathlib
from md_templates.openmm.schemas import shipped_system
src = shipped_system('ace_ala_nme').parent / 'ace_ala_nme.pdb'
shutil.copy2(src, pathlib.Path('$WORKDIR/src_pdb/ace_ala_nme.pdb'))"

cat > "$WORKDIR/src_pdb/sim.yaml" <<'YAML'
system: {system_id: ace_ala_nme, route: pdb, pdb: ace_ala_nme.pdb}
build:
  forcefield: {small_molecule: null, charge_method: null,
               protein: amber19/protein.ff19SB.xml, water: amber19/opc.xml}
  solvation: {box_shape: cube, padding: 1.2 nm, ionic_strength_molar: 0.0}
  nonbonded: {cutoff: 0.7 nm}
protocol:
  equilibration: {protocol: simple, minimize_max_iterations: 200, timestep: 1 fs,
                  nvt: 2 ps, npt: 2 ps, npt_free: 2 ps, box_average_last: 1 ps}
  production: {method: md, n_chunks: 2, chunk: 0.001 ns}
execution:
  platform: CPU
  reporting: {all_atom: 0.5 ps, solute: 0.5 ps, state: 0.5 ps, checkpoint: 0.5 ps}
YAML

step "1. prepare a tiny canonical SMILES bundle"
(cd "$WORKDIR/src_smiles" && md-openmm prepare --config sim.yaml --out-root ./runs --platform CPU)

step "2. prepare a tiny canonical PDB bundle"
(cd "$WORKDIR/src_pdb" && md-openmm prepare --config sim.yaml --out-root ./runs --platform CPU)

step "3. relocate both, then DELETE the originating directories"
cp -r "$WORKDIR"/src_smiles/runs/*__bundle__* "$WORKDIR/elsewhere/smiles_bundle"
cp -r "$WORKDIR"/src_pdb/runs/*__bundle__*    "$WORKDIR/elsewhere/pdb_bundle"
# only inside this run's own temporary directory
rm -rf "$WORKDIR/src_smiles" "$WORKDIR/src_pdb"
[ -d "$WORKDIR/src_smiles" ] && fail "source directory survived removal"

cd "$WORKDIR/elsewhere"

step "4. validate and inspect the relocated bundles"
for b in smiles_bundle pdb_bundle; do
  md-openmm bundle validate "$b" --deep || fail "$b failed validation after relocation"
  md-openmm bundle inspect "$b" >/dev/null || fail "$b failed inspection"
  md-openmm bundle relocate-check "$b" | grep -q '^relocatable' || fail "$b is not relocatable"
done

step "5. REST2 from the relocated bundle: fresh, then resume in the SAME directory"
md-openmm rest2 --bundle smiles_bundle --out-root ./runs --run-name r2 --platform CPU
md-openmm rest2 --bundle smiles_bundle --out-root ./runs --resume-run r2 --platform CPU

step "6. conventional MD from the relocated bundle: fresh, then resume"
md-openmm md --bundle pdb_bundle --out-root ./runs --run-name m1 --platform CPU
md-openmm md --bundle pdb_bundle --out-root ./runs --resume-run m1 --platform CPU

step "7. continuity of the resumed runs"
python - <<'PY'
import csv, json, pathlib, sys
r2 = pathlib.Path("runs/r2")
rows = list(csv.DictReader((r2 / "exchange_attempts.csv").open()))
idx = [int(r["attempt_index"]) for r in rows]
steps = [int(r["step"]) for r in rows]
if idx != list(range(len(idx))):
    sys.exit(f"REST2 attempt indices are not contiguous: {idx[:20]}")
if steps != sorted(steps):
    sys.exit("REST2 steps went backwards across the resume")
header_count = (r2 / "exchange_attempts.csv").read_text().count("attempt_index,")
if header_count != 1:
    sys.exit(f"exchange log has {header_count} headers; a resume duplicated one")
print(f"  REST2 lifetime attempts {len(rows)}, indices contiguous, one header")

m1 = pathlib.Path("runs/m1")
chunks = sorted(int(p.name.split("_")[1]) for p in m1.glob("chunk_*") if p.is_dir())
if chunks != list(range(len(chunks))):
    sys.exit(f"MD chunk sequence has a hole or repeat: {chunks}")
print(f"  MD chunks contiguous: {chunks}")

state = json.loads((r2 / "run_state.json").read_text())
if len(state.get("invocations", [])) < 2:
    sys.exit("the resume was not recorded in the invocation history")
print(f"  invocations recorded: {len(state['invocations'])}")
PY

step "8. State fallback when the checkpoint is unusable"
python - <<'PY'
import pathlib, sys
from md_templates.openmm import runstate
run = pathlib.Path("runs/r2")
gen = runstate.committed_generation(run)
gdir = runstate.generation_dir(run, gen)
chk = gdir / "replica_00.chk"
chk.write_bytes(b"not a checkpoint")          # corrupt, State beside it stays valid
print(f"  corrupted {chk.name} in committed generation {gen}")
PY
md-openmm rest2 --bundle smiles_bundle --out-root ./runs --resume-run r2 --platform CPU 2>&1 \
  | tee /tmp/fallback.log | tail -3
grep -q "portable State" /tmp/fallback.log \
  || fail "the State fallback was not announced when the checkpoint was unusable"
echo "  State fallback used and announced"

step "9. offline: validation and inspection must not touch the network"
python - <<'PY'
import socket, sys
def refuse(*a, **k):
    raise AssertionError("network access attempted")
socket.socket.connect = refuse
socket.create_connection = refuse
from md_templates.openmm import bundlecheck
report = bundlecheck.validate_bundle_v2("smiles_bundle", deep=True)
if not report.ok:
    sys.exit(f"offline validation failed: {report['errors']}")
info = bundlecheck.inspect_bundle("pdb_bundle")
if not info["validation"]["ok"]:
    sys.exit("offline inspection failed")
print("  validated and inspected both bundles with the network disabled")
PY

echo
echo "CPU integration: PASSED"
