#!/usr/bin/env bash
# hot_explicit: a SCALED cMD (tau = 0.5), which is fixed-volume throughout -- equilibration
# included, because the barostat's volume move is defined for the unscaled Hamiltonian and the
# runtime refuses it at a scaled one. So the density must be equilibrated BEFORE the scaling is
# applied, exactly as a REST2 ladder does it: its eq stages run at tau = 0.0 and only the ladder
# scales.
#
# `run.sh` would start this chain from `build-top`'s box, which on this system is 0.947 g/cm^3
# against TIP3P's ~0.985 -- 3.8% under, and nothing in a fixed-volume run will ever relax it. So
# the chain starts from cold_explicit_r1's NPT-equilibrated state (18.204 nm^3, 0.9934 g/cm^3).
#
# Placed in `hot_explicit/` after `md-openmm build-md -odir hot_explicit --config
# configs/hot_explicit.config`, with the campaign root as its parent. Recovered verbatim from the
# campaign's run history; it was the launcher of the 2026-09-11 hot_explicit run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
EQ=../cold_explicit_r1/eq_npt_free.xml
[[ -f $EQ ]] || { echo "no equilibrated structure at $EQ" >&2; exit 2; }
prev="$EQ"
for stage in eq_nvt_posres eq_nvt_posres_2 eq_nvt_free cMD; do
  [[ -f ${stage}.in ]] || continue
  echo "== ${stage} (from ${prev}) =="
  md-openmm md-run -i ${stage}.in -p ../explicit.pdb -s ../explicit.xml \
      -c "${prev}" -odir . || exit $?
  prev="${stage}.xml"
done
echo "hot_explicit complete"
