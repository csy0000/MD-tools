#!/bin/bash
# M3 driver: one window per process, one thread each, every process pinned INSIDE the cpuset
# agreed with hpREST2 (39-47). Width = number of cores in the set; never wider.
#
#   ./driver.sh <root> "<cores>" <policy> [<policy> ...]
#
# Windows are run in a fixed order; a finished window is skipped by run_window itself, so the
# driver can be re-run after an interruption and continues.
set -u
ROOT="$1"; CORES="$2"; shift 2
REPO=/data3/data/chen/scheme/MD-tools-S4-execution
IFS=',' read -ra CORE_LIST <<< "$(echo "$CORES" | sed 's/-/../' | xargs -I{} bash -c 'eval echo {}' | tr ' ' ',')"
WIDTH=${#CORE_LIST[@]}
echo "$(date +%T) driver: root=$ROOT cores=${CORE_LIST[*]} width=$WIDTH policies=$*"

jobs_list=()
for policy in "$@"; do
  for leg in vacuum vacuum_ba solvent_v2; do
    n=$(python - "$leg" <<'PY'
import sys
sys.path.insert(0, "/data3/data/chen/scheme/MD-tools-S4-execution")
from tests.test_alchemy_junction_policy import S_VALUES
print(len(S_VALUES[sys.argv[1]]))
PY
)
    for r in r1 r2 r3; do
      for ((i=0; i<n; i++)); do
        jobs_list+=("$policy $leg $r $(printf 'w%03d' "$i")")
      done
    done
  done
done
echo "$(date +%T) driver: ${#jobs_list[@]} windows queued"

i=0
for job in "${jobs_list[@]}"; do
  core=${CORE_LIST[$((i % WIDTH))]}
  while [ "$(jobs -rp | wc -l)" -ge "$WIDTH" ]; do sleep 5; done
  set -- $job
  log="$ROOT/logs/$1_$2_$3_$4.log"
  mkdir -p "$ROOT/logs"
  taskset -c "$core" env OPENMM_CPU_THREADS=1 CUDA_VISIBLE_DEVICES="" PYMBAR_DISABLE_JAX=true \
      MD_TOOLS_S4_M3_ROOT="$ROOT" PYTHONPATH="$REPO" \
      python "$REPO/tests/test_alchemy_junction_policy.py" "$1" "$2" "$3" "$4" \
      > "$log" 2>&1 &
  i=$((i + 1))
done
wait
echo "$(date +%T) driver: all ${#jobs_list[@]} windows finished"
