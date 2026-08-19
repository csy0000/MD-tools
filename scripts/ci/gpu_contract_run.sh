#!/usr/bin/env bash
# Real-GPU evidence for the platform contract. NOT part of CPU CI -- run deliberately on a machine
# with CUDA, and record what it produces.
#
#   scripts/ci/gpu_contract_run.sh [WORKDIR] [DEVICE]
#
# What this establishes, and what it does not:
#
#   establishes   the CUDA path actually runs: prepare, conventional MD, REST2, and resume, on a
#                 named device, with the precision that was asked for; and that the generic and
#                 legacy routes produce identical canonical hashes on an accelerator, not only on
#                 CPU. Device selection is checked against PCI bus order, which only means anything
#                 on a machine with mixed hardware.
#   does NOT      validate any science. These are picoseconds. A GPU run that completes is
#                 engineering evidence exactly as a CPU run that completes is.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-$(mktemp -d -t md-gpu-XXXX)}"
DEVICE="${2:-1}"
mkdir -p "$WORKDIR"/{generic,legacy,rest2}

step() { printf '\n=== %s ===\n' "$1"; }
fail() { echo "GPU CONTRACT FAILED: $1" >&2; exit 1; }

step "0. hardware, and the device this run selects"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: this script needs a real GPU"
# PCI bus order is what makes an index stable when the devices are not identical.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader | sed 's/^/  /'
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEVICE")"
GPU_BUS="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$DEVICE")"
echo "  selected: device $DEVICE -> $GPU_NAME at $GPU_BUS"

step "1. the environment agrees CUDA is usable on that device"
md-openmm validate-env --platform CUDA --device "$DEVICE"

# Same document for both routes. Tiny: this measures agreement and execution, not physics.
read -r -d '' CONFIG <<YAML || true
system: {system_id: ggg, route: smiles, smiles: "O=C1CNC(=O)CNC(=O)CN1"}
protocol:
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
  platform: CUDA
  device: "$DEVICE"
  precision: mixed
  reporting: {all_atom: 0.5 ps, solute: 0.5 ps, state: 0.5 ps, checkpoint: 0.5 ps}
YAML
printf '%s\n' "$CONFIG" > "$WORKDIR/generic/sim.yaml"
printf '%s\n' "$CONFIG" > "$WORKDIR/legacy/sim.yaml"

step "2. prepare on CUDA through BOTH routes"
cd "$WORKDIR/generic"
md-templates prepare --template conventional-md/openmm/explicit-water -- \
    --config sim.yaml --out-root ./runs --platform CUDA --device "$DEVICE"
cd "$WORKDIR/legacy"
md-openmm prepare --config sim.yaml --out-root ./runs --platform CUDA --device "$DEVICE"

GENERIC_BUNDLE="$(find "$WORKDIR/generic/runs" -maxdepth 1 -name '*bundle*' | head -1)"
LEGACY_BUNDLE="$(find "$WORKDIR/legacy/runs" -maxdepth 1 -name '*bundle*' | head -1)"
[ -n "$GENERIC_BUNDLE" ] && [ -n "$LEGACY_BUNDLE" ] || fail "a route produced no bundle"

step "3. identical canonical hashes on the accelerator too"
python - "$GENERIC_BUNDLE" "$LEGACY_BUNDLE" <<'PY'
import json, pathlib, sys

g, l = (json.loads((pathlib.Path(p) / "canonical_configuration.json").read_text())
        for p in sys.argv[1:3])
for key in sorted(g["hashes"]):
    mark = "ok " if g["hashes"][key] == l["hashes"].get(key) else "DIFFERS"
    print(f"  {mark} {key}: {g['hashes'][key]}")
if g["hashes"] != l["hashes"] or g["configuration"] != l["configuration"]:
    sys.exit("the generic and legacy CUDA routes disagree")
print(f"  precision recorded: {g['configuration']['execution']['precision']}"
      f"   device: {g['configuration']['execution']['device']}")
print("  identical canonical configuration and all five projection hashes")
PY

step "4. conventional MD: run and resume on CUDA, generic route"
cd "$WORKDIR/generic"
md-templates run --template conventional-md/openmm/explicit-water -- \
    --bundle "$GENERIC_BUNDLE" --out-root ./runs --run-name g1 --platform CUDA --device "$DEVICE"
md-templates resume --template conventional-md/openmm/explicit-water -- \
    --bundle "$GENERIC_BUNDLE" --out-root ./runs --resume-run g1 --platform CUDA --device "$DEVICE"

step "5. REST2: run and resume on ONE device, three replicas, one process"
cd "$WORKDIR/rest2"
cat > sim.yaml <<YAML
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
execution: {platform: CUDA, device: "$DEVICE", precision: mixed}
YAML
md-templates prepare --template rest2/openmm/explicit-water -- \
    --config sim.yaml --out-root ./runs --platform CUDA --device "$DEVICE"
REST2_BUNDLE="$(find "$WORKDIR/rest2/runs" -maxdepth 1 -name '*bundle*' | head -1)"
[ -n "$REST2_BUNDLE" ] || fail "no REST2 bundle"
md-templates run --template rest2/openmm/explicit-water -- \
    --bundle "$REST2_BUNDLE" --out-root ./runs --run-name r1 --platform CUDA --device "$DEVICE"
md-templates resume --template rest2/openmm/explicit-water -- \
    --bundle "$REST2_BUNDLE" --out-root ./runs --resume-run r1 --platform CUDA --device "$DEVICE"

step "6. what the runs committed"
python - "$WORKDIR/generic/runs/g1" "$WORKDIR/rest2/runs/r1" <<'PY'
import json, pathlib, sys

for label, path in (("conventional MD", sys.argv[1]), ("REST2", sys.argv[2])):
    run = pathlib.Path(path)
    committed = json.loads((run / "restart" / "committed.json").read_text())
    state = json.loads((run / "run_state.json").read_text())
    members = committed.get("members", [])
    replicas = sorted({m.split('.')[0] for m in members if m.startswith("replica_")})
    print(f"  {label}: generation {committed.get('generation')}, "
          f"{len(state.get('invocations', []))} invocation(s)"
          + (f", replicas {replicas}" if replicas else ""))
    if len(state.get("invocations", [])) < 2:
        sys.exit(f"{label} did not record a resume")
PY

step "7. an unavailable device is refused, not silently downgraded"
if md-openmm validate-env --platform CUDA --device 99 >/dev/null 2>&1; then
    fail "device 99 should not have validated"
fi
echo "  ok: device 99 refused"

echo
echo "GPU contract run: PASSED on device $DEVICE ($GPU_NAME, $GPU_BUS), precision mixed"
