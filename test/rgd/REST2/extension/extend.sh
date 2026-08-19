#!/usr/bin/env bash
# Request ADDITIONAL 5 ns REST2 segments for the parent run.
#
# This directory holds the extension REQUEST, not a copy of the run. Continuation writes into the
# ORIGINAL run directory: copying checkpoints here and running beside them would produce a
# scientifically separate sibling run wearing the word "continuation".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"

CONFIG="${CONFIG:-$PARENT/rgdfv_rest2.json}"
OUT_ROOT="${OUT_ROOT:-$PARENT/outputs}"
RUN_NAME="${RUN_NAME:-rgd_rest2}"
MD_DEVICES="${MD_DEVICES:-0}"
ADDITIONAL_SEGMENTS="${ADDITIONAL_SEGMENTS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN_DIR="$OUT_ROOT/rest2/$RUN_NAME"
if [ ! -d "$RUN_DIR" ]; then
    echo "no parent run at $RUN_DIR -- run ../run_all.sh first" >&2
    exit 1
fi

echo "== committed state BEFORE extension =="
cat "$RUN_DIR/committed.json"

for segment in $(seq 1 "$ADDITIONAL_SEGMENTS"); do
    echo "-- additional segment $segment of $ADDITIONAL_SEGMENTS"
    # Same command pattern as the parent run, same run name, same output root. The committed
    # generation record decides where this resumes from; nothing here derives the boundary from
    # file counts or log length.
    md-openmm rest2 \
        --bundle "$OUT_ROOT/bundle" \
        --config "$CONFIG" \
        --out-root "$OUT_ROOT/rest2" \
        --resume-run "$RUN_NAME" \
        --platform CUDA --devices "$MD_DEVICES"
done

echo "== committed state AFTER extension =="
cat "$RUN_DIR/committed.json"
