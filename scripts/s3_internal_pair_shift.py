#!/usr/bin/env python
"""How big is the intra-ligand cutoff shift, as a function of ligand extent?

Development evidence for 0.7.0, not package code. S2's plan currently treats a unique group's
internal non-excluded pairs asymmetrically -- uncut at the dummy end (a CustomBondForce), cut at
0.9 nm at the physical end (the NonbondedForce) -- so those pairs are lambda-DEPENDENT and the
decoupling derivation's premise that the ligand's intramolecular Hamiltonian cancels between legs
is false by exactly their size. Under the symmetric construction they become one
modified-but-lambda-independent function whose offset is identical in both legs and cancels in
ddG, provided both legs are built the same way (`matched_legs`, `ligand_hamiltonian_sha256`).

This measures that offset per ligand, so the record carries its SCALE rather than one number:
the ligand's atom count and extent, how many internal pairs fall beyond the cutoff, and what they
are worth in Lennard-Jones and in real-space electrostatics.

Usage (builds the TYK2 fixture into --work, about a minute per ligand, no GPU):
    python scripts/s3_internal_pair_shift.py --work <dir> [--ligand ejm_31 ...]
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

TYK2 = REPO / "tests" / "data" / "alchemy" / "tyk2-v1"
LIGANDS = {"ejm_31": ("L31", "LOCAL-DKNAYSZNMZIMIZ/param_bd1388e5fe3e"),
           "ejm_42": ("L42", "LOCAL-CEJFWHOOEUEYOE/param_c1d8d1147233"),
           "ejm_43": ("L43", "LOCAL-HKQHWWQCRDXFGC/param_c66a3e91a5dd")}


def measure(ligand: str, work: Path):
    import numpy as np
    import openmm
    from md_tools.alchemy.topology import Environment, build_decoupling_plan
    from md_tools.ligands import load_package
    from md_tools.ligands.mapping import LigandSelector
    from md_tools.alchemy.softcore import ONE_4PI_EPS0

    resname, reference = LIGANDS[ligand]
    build = work / ligand / "solvated" / "build"
    if not (build / "built.xml").is_file():
        # the fixture script shells out to `md-openmm`; pin that subprocess to THIS checkout, or it
        # resolves the installed md_tools and refuses the package with an older validator
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(REPO / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        subprocess.run([sys.executable, str(TYK2 / "build_tyk2_fixture.py"), "--out", str(work),
                        "--ligand", ligand, "--kind", "solvated"],
                       check=True, timeout=3600, cwd=str(REPO / "src"), env=environment)
    env = Environment.from_files(build / "built.xml", build / "built.pdb",
                                 LigandSelector(resname=resname), record=build / "built.log")
    plan = build_decoupling_plan(load_package(TYK2 / "packages" / reference), env)
    x, group = plan.positions_nm, sorted(plan.a_only)
    nb = next(f for f in plan.system_a.getForces() if isinstance(f, openmm.NonbondedForce))
    cutoff = nb.getCutoffDistance()._value
    context = openmm.Context(plan.system_a, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(x)
    kappa = nb.getPMEParametersInContext(context)[0]
    del context
    excepted = set()
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        excepted.add((min(i, j), max(i, j)))

    internal = beyond = 0
    lj = electrostatics = 0.0
    extent = 0.0
    for i, j in itertools.combinations(group, 2):
        r = float(np.linalg.norm(x[j] - x[i]))
        extent = max(extent, r)
        if (i, j) in excepted:
            continue
        internal += 1
        if r < cutoff:
            continue
        beyond += 1
        qi, si, ei = (v._value for v in nb.getParticleParameters(i))
        qj, sj, ej = (v._value for v in nb.getParticleParameters(j))
        s, e = 0.5 * (si + sj), math.sqrt(ei * ej)
        lj += 4 * e * ((s / r) ** 12 - (s / r) ** 6)
        electrostatics += ONE_4PI_EPS0 * qi * qj * math.erfc(kappa * r) / r
    return {"ligand": ligand, "atoms": len(group), "extent_nm": extent, "cutoff_nm": cutoff,
            "internal_pairs": internal, "beyond_cutoff": beyond, "lj_kj_mol": lj,
            "real_space_electrostatics_kj_mol": electrostatics, "shift_kj_mol": lj + electrostatics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ligand", action="append", choices=sorted(LIGANDS))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = [measure(name, work) for name in (args.ligand or sorted(LIGANDS))]
    print(f"\n{'ligand':8s} {'atoms':>5s} {'extent':>7s} {'internal':>9s} {'beyond':>7s} "
          f"{'LJ':>10s} {'elec':>10s} {'shift':>10s}")
    for r in rows:
        print(f"{r['ligand']:8s} {r['atoms']:5d} {r['extent_nm']:7.3f} {r['internal_pairs']:9d} "
              f"{r['beyond_cutoff']:7d} {r['lj_kj_mol']:10.6f} "
              f"{r['real_space_electrostatics_kj_mol']:10.2e} {r['shift_kj_mol']:10.6f}")
    print("\nextent is the largest intra-ligand distance, nm; shift is what moving those pairs into "
          "the internal force changes the physical end state by, kJ/mol.")


if __name__ == "__main__":
    main()
