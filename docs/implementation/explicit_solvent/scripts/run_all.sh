#!/bin/bash
# Thin compatibility wrapper around the installed CLI. The pipeline lives in the package
# (`md_templates.openmm`), not here; this file exists so an existing muscle-memory invocation still
# works and so there is one obvious place that shows the portable command.
#
#   ./run_all.sh --system cyclo_rgdfv --experiment rgd_rest2_10rung --out-root /path/to/runs \
#                --platform CUDA --device 0
#   ./run_all.sh --system ./my_system.yaml --experiment ./my_experiment.yaml --out-root ./runs \
#                --platform CPU
#
# You do not need this script. It is exactly equivalent to:
#
#   md-openmm prepare --system SYS --experiment EXP --out-root ROOT [--platform P] [--device D]
#   md-openmm rest2   --bundle BUNDLE --out-root ROOT [--platform P] [--device D]
#
# WHAT CHANGED, AND WHY. The previous version hardcoded CUDA_VISIBLE_DEVICES=0/1/2, assumed the
# machine had three GPUs, backgrounded three jobs with `nohup`, and wrote logs to a
# repository-relative `logs/`. On a one-GPU machine the second and third jobs silently landed back
# on device 0, where two REST2 processes sharing a card without CUDA MPS run ~3.7x slower each --
# wrong, but not visibly wrong. Multi-job scheduling now belongs to the consuming repository's
# scheduler: this launches ONE process, controlling its own replicas, on ONE selected device.
set -euo pipefail

SYSTEM=""; EXPERIMENT=""; OUT_ROOT=""; PLATFORM="CUDA"; DEVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --system)     SYSTEM="$2";     shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --out-root)   OUT_ROOT="$2";   shift 2 ;;
    --platform)   PLATFORM="$2";   shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    -h|--help)    sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for required in SYSTEM EXPERIMENT OUT_ROOT; do
  if [ -z "${!required}" ]; then
    echo "usage: run_all.sh --system SYS --experiment EXP --out-root ROOT [--platform CPU|CUDA|OpenCL] [--device ID]" >&2
    exit 2
  fi
done

if ! command -v md-openmm >/dev/null 2>&1; then
  echo "md-openmm is not on PATH. Activate the environment and install the package:" >&2
  echo "  conda activate md-templates" >&2
  echo "  python -m build && python -m pip install dist/md_templates-*.whl --no-deps" >&2
  exit 4
fi

# So that --device N means the card nvidia-smi calls N. Without it the CUDA runtime is free to
# order devices by its own heuristic.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then DEVICE_ARGS=(--device "$DEVICE"); fi

md-openmm validate-env --platform "$PLATFORM" "${DEVICE_ARGS[@]}"

# `prepare` prints the bundle path as the first line of its output; capture it rather than
# guessing the timestamped directory name.
BUNDLE=$(md-openmm prepare \
  --system "$SYSTEM" --experiment "$EXPERIMENT" --out-root "$OUT_ROOT" \
  --platform "$PLATFORM" "${DEVICE_ARGS[@]}" | sed -n 's/^bundle: //p' | head -1)
if [ -z "$BUNDLE" ]; then
  echo "prepare did not report a bundle path" >&2
  exit 5
fi
echo "bundle: $BUNDLE"

# No nohup and no `&`: one process, in the foreground, so the caller's scheduler owns the job and
# a non-zero exit is actually observed. stdout/stderr are also written inside the run directory.
exec md-openmm rest2 \
  --bundle "$BUNDLE" --out-root "$OUT_ROOT" \
  --platform "$PLATFORM" "${DEVICE_ARGS[@]}"
