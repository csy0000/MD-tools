#!/bin/bash
# Matched RGD ladder comparison: 3 independent repeats each of the 8-rung control and the
# 10-rung candidate, 2 ns/replica, everything else identical.  Seeds frozen here, before results.
export PATH=~/software/miniforge3/envs/escort-ais/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPO=~/projects/escort-ais-alanine
S=$REPO/docs/implementation/explicit_solvent/scripts
SRC=~/.claude/jobs/29daa285/tmp/20260814T1500_phaseA_cyclo_rgdfv
BASE=~/.claude/jobs/29daa285/tmp/20260814T1600_rgd_ladder

one () {  # $1 rungs  $2 repeat  $3 gpu
  local OUT=$BASE/rungs$1_rep$2; mkdir -p $OUT; cd $OUT
  local SEED=$(( 20260814 + $2 * 1000 + $1 ))
  python $S/generate_config.py --rest2 --system macrocycle --N_rungs $1 --seed $SEED \
      --slug cyclo_rgdfv_sage_explicit > /dev/null 2>&1
  python - <<PY
import json
d=json.load(open("rest2-config.json"))
d["production"]["remd"].update(total_ns_per_replica=2.0, chunk_ns=0.5, equilibration_ps=10.0,
                               exchange_interval_ps=10.0)
json.dump(d,open("rest2-config.json","w"),indent=2)
PY
  cp $SRC/rgd_system.xml $SRC/rgd_topology.pdb $SRC/rgd_simbox.json .
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$3 python $S/md_REST2.py --p rgd_system.xml --c $SRC/eq_state.xml \
      --out-suffix pilot --config rest2-config.json > pilot.log 2>&1
  echo "rungs=$1 rep=$2 seed=$SEED gpu=$3 rc=$? wall_s=$(( $(date +%s) - t0 ))" >> $BASE/results.txt
}

mkdir -p $BASE
for R in 1 2 3; do one 8 $R $((R-1)) & done; wait
for R in 1 2 3; do one 10 $R $((R-1)) & done; wait
echo "### PILOTS6_DONE"; cat $BASE/results.txt
