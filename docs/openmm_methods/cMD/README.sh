#!/usr/bin/env bash
#
# =============================================================================
#  cMD -- the runnable form of README.md beside this file.
#
#      ./README.sh                     the documented example, on a GPU
#      ./README.sh --quick --cpu       a small implicit-solvent version, anywhere
#      ./README.sh -o /tmp/x           write somewhere else
#      ./README.sh -i my_protein.pdb   your own structure
#
#  README.md explains what cMD is and what the generated directory contains.
#  This file is the same three commands, executable, so they cannot drift from
#  what actually works.
#
#  DEFAULT is the documented example: explicit TIP3P water and example.config's
#  100,000 production steps (200 ps at 2 fs). Minutes on a GPU, far longer on a
#  CPU -- which is what --quick is for.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# This script NEVER deletes anything. If the output directory exists it stops
# and says so, rather than clearing a path it was handed.
PLATFORM=""; QUICK=""; OUT="./run"; STRUCTURE="../../../tests/data/ALA.pdb"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)   PLATFORM="--cpu"; shift ;;
    --quick) QUICK=1; shift ;;
    -o)      OUT="$2"; shift 2 ;;
    -i)      STRUCTURE="$2"; shift 2 ;;
    *) echo "usage: $0 [--cpu] [--quick] [-o OUTDIR] [-i STRUCTURE.pdb]" >&2; exit 2 ;;
  esac
done
if [[ -e "${OUT}" ]]; then
  echo "'${OUT}' already exists. This script does not delete anything." >&2
  echo "Remove it yourself, or pass -o SOMEWHERE_ELSE." >&2
  exit 2
fi

STRUCTURE="$(cd "$(dirname "${STRUCTURE}")" && pwd)/$(basename "${STRUCTURE}")"
mkdir -p "${OUT}"
cp example.config "${OUT}/md.config"

if [[ -n "${QUICK}" ]]; then
  # Implicit solvent and a few thousand steps, so the whole chain finishes on a
  # CPU in about a minute. Everything else -- the commands, the generated files,
  # the order -- is identical to the documented example.
  cat > "${OUT}/sys.config" <<'YAML'
solute: {kind: peptide}
solvent: {model: GBn2}
constraints: {type: HBonds}
hydrogen_mass_repartitioning: {enabled: false}
YAML
  cat > "${OUT}/md.config" <<'YAML'
protocol: cMD
solvent: implicit
dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 20260908}
stages:
  minimization_iterations: 100
  restrained_nvt_steps: 250
  restrained_npt_steps: 250
  unrestrained_npt_steps: 250
  production_steps: 2000
reporting: {crd_printout_solute: 250, info_printout: 250, checkpoint_printout: 250}
YAML
fi
cd "${OUT}"

# -----------------------------------------------------------------------------
# 1. BUILD THE SYSTEM -- the physics.
#
# Force field, solvation, charges, radii, constraints. Writes built.xml, which
# IS the Hamiltonian, plus the topology and a machine-readable record.
#
# With no --config, build-top uses its documented defaults: a peptide PDB and
# TIP3P explicit water. --quick writes a sys.config to choose implicit instead.
# -----------------------------------------------------------------------------
echo "== 1. build-top =="
if [[ -n "${QUICK}" ]]; then
  md-openmm build-top -i "${STRUCTURE}" -os built.xml -op built.pdb -log built.log \
      --config sys.config
else
  md-openmm build-top -i "${STRUCTURE}" -os built.xml -op built.pdb -log built.log
fi

# -----------------------------------------------------------------------------
# 2. GENERATE THE WORKFLOW -- the experiment.
#
# Stage lengths and reporting intervals. build-md never opens built.xml: the
# timestep is resolved when the run starts, from the masses actually serialised
# there, because a configuration claiming HMR is a request and the System is the
# fact.
#
# Into ./md_script/ :  run.sh, one .in and one .py per stage, and
# resolved.config -- which is AUTHORITATIVE and is what the run reads.
# -----------------------------------------------------------------------------
echo "== 2. build-md =="
md-openmm build-md -odir ./md_script --config md.config

# -----------------------------------------------------------------------------
# 3. RUN IT.
#
# run.sh calls each stage in order and wires each stage's final state into the
# next one's -c. Arguments after the two paths reach every stage, which is how
# --cpu gets there.
#
# One stage at a time instead, if you prefer -- the same run, same installed code:
#
#   md-openmm md-run -i min.in -p ../built.pdb -s ../built.xml \
#       -o min.out -r min.xml -log min.log
#   python min.py    -p ../built.pdb -s ../built.xml \
#       -o min.out -r min.xml -log min.log
# -----------------------------------------------------------------------------
echo "== 3. run.sh =="
cd md_script
# shellcheck disable=SC2086
./run.sh ../built.pdb ../built.xml ${PLATFORM}

# -----------------------------------------------------------------------------
# WHAT YOU GET, per stage: <stage>.dcd, .xml (the state the next stage takes),
# .out, .log, .csv, and <stage>.checkpoints/.
#
# IF IT IS INTERRUPTED: a cMD chain currently CANNOT be resumed. Re-running is
# refused because its outputs exist, and --resume is rejected as "not a cMD
# flag". Today the options are --overwrite, which discards the finished stages,
# or a fresh -odir. See ../../backlog.md entry 7. A REST2 ladder is unaffected.
# -----------------------------------------------------------------------------
echo
echo "done. outputs are in $(pwd)"
ls -1 whole_prod1.nc solute_prod1.nc mdout.csv cMD.out cMD.log 2>/dev/null || true
