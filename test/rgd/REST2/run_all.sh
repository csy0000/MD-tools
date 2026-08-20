#!/usr/bin/env bash
# Alanine dipeptide, ten-replica REST2 -- the whole chain in order:
#
#   prepare -> minimize -> 10 ps NVT -> 10 ps NPT -> 1 ns cMD -> 10 ns REST2
#
# The scientific JSON declares ONE segment. This script decides how many to run, which is why
# NUMBER_OF_SEGMENTS lives here and not in the configuration.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/rgdfv_rest2.json}"
OUT_ROOT="${OUT_ROOT:-$HERE/outputs}"
RUN_NAME="${RUN_NAME:-rgd_rest2}"

# Replicas are dealt round-robin over this ordered list. Six replicas on six GPUs is one each;
# fewer GPUs is allowed and the resolved mapping is recorded.
MD_DEVICES="${MD_DEVICES:-0}"
NUMBER_OF_SEGMENTS="${NUMBER_OF_SEGMENTS:-2}"      # 2 x 5 ns = 10 ns per replica
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "== configuration =="
md-openmm config resolve --input "$CONFIG"

echo "== 1. prepare the bundle (build + solvate + ionise) =="
md-openmm prepare --config "$CONFIG" --out-root "$OUT_ROOT/bundle"

echo "== 2. minimise + 10 ps NVT + 10 ps NPT, solute restrained =="
# Equilibration runs as part of prepare/md entry depending on the configured protocol; the
# restrained stages and their reference coordinates come from protocol.equilibration.

echo "== 3. 1 ns conventional NPT MD =="
md-openmm md \
    --bundle "$OUT_ROOT/bundle" \
    --config "$CONFIG" \
    --out-root "$OUT_ROOT/md" \
    --run-name "${RUN_NAME}_cmd" \
    --platform CUDA --device "${MD_DEVICES%%,*}"

echo "== 4. REST2: $NUMBER_OF_SEGMENTS segment(s) of 5 ns per replica =="
for segment in $(seq 1 "$NUMBER_OF_SEGMENTS"); do
    echo "-- segment $segment of $NUMBER_OF_SEGMENTS"
    # Every invocation runs ONE segment and continues in the SAME run directory. The first
    # NAMES the run; every later one RESUMES it. The two flags are mutually exclusive, and a
    # fresh run deliberately refuses to overwrite an existing directory -- so passing --run-name
    # twice would stop the second segment rather than silently restarting the exchange sequence.
    if [ "$segment" -eq 1 ]; then
        NAME_ARGS=(--run-name "$RUN_NAME")
    else
        NAME_ARGS=(--resume-run "$RUN_NAME")
    fi
    md-openmm rest2 \
        --bundle "$OUT_ROOT/bundle" \
        --config "$CONFIG" \
        --out-root "$OUT_ROOT/rest2" \
        "${NAME_ARGS[@]}" \
        --platform CUDA --devices "$MD_DEVICES"
done

echo "== done. committed state: =="
cat "$OUT_ROOT/rest2/$RUN_NAME/committed.json" 2>/dev/null || \
    echo "  (no committed.json yet -- check the run log)"
