#!/usr/bin/env python
"""S3 cross-engine gate: the softcore Hamiltonian against AMBER's pmemd, at fixed coordinates.

Development evidence for 0.7.0 milestone A2 (validation gate 4), not package code. It

1. builds S3's miniature end-state pair (`tests/alchemy_s3_fixture.py`, PME, no dispersion
   correction) and, from it, pmemd's single-topology-file TI layout: a V0 copy of the molecule
   (core + A-only, end state A's parameters), a V1 copy (core + B-only, end state B's), then the
   environment. timask1/timask2 select the copies and scmask1/scmask2 the unique atoms; pmemd
   removes every V0 x V1 interaction itself;
2. writes it with ParmEd, after proving the writer on single-copy Systems: each round-trips
   through OpenMM's AmberPrmtopFile to the OpenMM System it was written from;
3. runs pmemd (the CPU build, double precision) at clambda = 0, 0.25, 0.5, 0.75, 1, with the
   PME splitting, grid and spline order shared with OpenMM, a dense erfc table, no long-range LJ
   correction,
   and MBAR energies at all five lambdas;
4. compares U(lambda_k) and dU/dlambda with `md_tools.alchemy.hamiltonian` on the diagonal path.

Usage (AMBERHOME must hold bin/pmemd):
    python scripts/s3_amber_crossengine.py --amberhome /path/to/amber26 --out <dir> [--boundary scaled|unscaled]
Writes <dir>/report.json and the pmemd inputs/outputs. Needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

KJ_PER_KCAL = 4.184
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ELEMENTS = {"C": "carbon", "H": "hydrogen", "F": "fluorine", "Cl": "chlorine", "Na": "sodium",
            "Ar": "argon", "O": "oxygen"}


#: One PME grid both engines accept: pmemd needs nfft with prime factors 2, 3, 5 only, and OpenMM
#: picks 103 for this box at ewaldErrorTolerance 1e-6. Both are given OpenMM's alpha and this grid.
PME_GRID = 100


def fixture(tail=False):
    import openmm
    from tests import alchemy_s3_fixture as fx
    sa, sb, a, b, x = fx.build(True, dispersion=False, tail=tail)
    probe = openmm.Context(sa, openmm.VerletIntegrator(0.001),
                           openmm.Platform.getPlatformByName("Reference"))
    nb = next(f for f in probe.getSystem().getForces() if isinstance(f, openmm.NonbondedForce))
    alpha = nb.getPMEParametersInContext(probe)[0]
    for s in (sa, sb):
        next(f for f in s.getForces() if isinstance(f, openmm.NonbondedForce)).setPMEParameters(
            alpha, PME_GRID, PME_GRID, PME_GRID)
    return sa, sb, a, b, x


# --------------------------------------------------------------------------------------------
# The dual-copy layout
# --------------------------------------------------------------------------------------------

MOLECULE_NAMES = ["C0", "C1", "H0A", "H0B", "H1A", "H1B", "H1C"]
MOLECULE_ELEMENTS = ["C", "C", "H", "H", "H", "H", "H"]


def layout(sa, sb, a_only, b_only, x, copies=("A", "B")):
    """(topology, system, positions, index map) for the requested copies plus the environment.

    Copy A: hybrid 0..6 (core) and the A-only atom, with System A's parameters and terms.
    Copy B: hybrid 0..6 and the B-only atom, with System B's. Environment: hybrid 9.., System A's
    (identical in both). Terms and exceptions are taken from each copy's own end state, among that
    copy's atoms only.
    """
    import openmm
    from openmm import app
    n = sa.getNumParticles()
    unique = set(a_only) | set(b_only)
    env = [i for i in range(9, n) if i not in unique]
    core = [(i, MOLECULE_NAMES[i], MOLECULE_ELEMENTS[i]) for i in range(7)]
    tail_names = iter(["T1", "T2", "T3", "T4"])
    unique_atoms = {7: ("HA", "H"), 8: ("FB", "F")}
    for i in sorted(unique - {7, 8}):
        unique_atoms[i] = (next(tail_names), "C")
    groups = []                      # (residue name, [(hybrid index, atom name, element)], system)
    if "A" in copies:
        groups.append(("MLA", core + [(i, *unique_atoms[i]) for i in sorted(a_only)], sa))
    if "B" in copies:
        groups.append(("MLB", core + [(i, *unique_atoms[i]) for i in sorted(b_only)], sb))
    topology = app.Topology()
    chain = topology.addChain()
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*sa.getDefaultPeriodicBoxVectors())
    positions, atoms = [], []
    maps = []                        # per group: hybrid index -> new index
    for resname, members, source in groups:
        res = topology.addResidue(resname, chain)
        m = {}
        for hyb, name, el in members:
            m[hyb] = len(atoms)
            atoms.append(topology.addAtom(name, app.Element.getBySymbol(el), res))
            system.addParticle(source.getParticleMass(hyb))
            positions.append(x[hyb])
        maps.append((m, source))
    env_map = {}
    names = {9: ("CL", "CL", "Cl"), 10: ("NA", "NA", "Na"), 11: ("PRB", "AR", "Ar")}
    for hyb in env:
        if hyb in names:
            resname, name, el = names[hyb]
            res = topology.addResidue(resname, chain)
        else:
            k = (hyb - 12) % 3
            if k == 0:
                res = topology.addResidue("WTR", chain)      # not WAT: no SETTLE anywhere
            name, el = ("O", "O") if k == 0 else (f"H{k}", "H")
        env_map[hyb] = len(atoms)
        atoms.append(topology.addAtom(name, app.Element.getBySymbol(el), res))
        system.addParticle(sa.getParticleMass(hyb))
        positions.append(x[hyb])
    maps.append((env_map, sa))

    nb_src = {id(s): next(f for f in s.getForces() if isinstance(f, openmm.NonbondedForce))
              for s in (sa, sb)}
    ref_nb = nb_src[id(sa)]
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(ref_nb.getCutoffDistance())
    nb.setEwaldErrorTolerance(ref_nb.getEwaldErrorTolerance())
    nb.setUseDispersionCorrection(False)
    params = [None] * len(atoms)
    for m, source in maps:
        src = nb_src[id(source)]
        for hyb, new in m.items():
            params[new] = src.getParticleParameters(hyb)
    for q, s, e in params:
        nb.addParticle(q, s, e)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsions = openmm.PeriodicTorsionForce()
    for m, source in maps:
        src = nb_src[id(source)]
        for k in range(src.getNumExceptions()):
            i, j, qq, sg, ep = src.getExceptionParameters(k)
            if i in m and j in m:
                nb.addException(m[i], m[j], qq, sg, ep)
        for f in source.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                for k in range(f.getNumBonds()):
                    i, j, r0, kk = f.getBondParameters(k)
                    if i in m and j in m:
                        bonds.addBond(m[i], m[j], r0, kk)
                        topology.addBond(atoms[m[i]], atoms[m[j]])
            elif isinstance(f, openmm.HarmonicAngleForce):
                for k in range(f.getNumAngles()):
                    i, j, l, t0, kk = f.getAngleParameters(k)
                    if i in m and j in m and l in m:
                        angles.addAngle(m[i], m[j], m[l], t0, kk)
            elif isinstance(f, openmm.PeriodicTorsionForce):
                for k in range(f.getNumTorsions()):
                    i, j, l, o, per, ph, kk = f.getTorsionParameters(k)
                    if all(a in m for a in (i, j, l, o)):
                        torsions.addTorsion(m[i], m[j], m[l], m[o], per, ph, kk)
    for f in (nb, bonds, angles, torsions):
        system.addForce(f)
    topology.setPeriodicBoxVectors(sa.getDefaultPeriodicBoxVectors())
    return topology, system, np.array(positions), maps


def write_amber(topology, system, positions, stem: Path):
    import parmed
    structure = parmed.openmm.load_topology(topology, system=system, xyz=positions * 10.0)
    structure.box = [v * 10 for v in (positions.max() * 0 + np.array(
        [system.getDefaultPeriodicBoxVectors()[i][i]._value for i in range(3)]))] + [90.0, 90.0, 90.0]
    structure.save(str(stem.with_suffix(".parm7")), format="amber", overwrite=True)
    structure.save(str(stem.with_suffix(".rst7")), format="rst7", overwrite=True)
    return stem.with_suffix(".parm7"), stem.with_suffix(".rst7")


def openmm_energy(system, positions):
    import openmm
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(positions)
    return c.getState(getEnergy=True).getPotentialEnergy()._value


def prmtop_energy(parm7, rst7, like_system):
    import openmm
    from openmm import app
    prmtop, inpcrd = app.AmberPrmtopFile(str(parm7)), app.AmberInpcrdFile(str(rst7))
    ref_nb = next(f for f in like_system.getForces() if isinstance(f, openmm.NonbondedForce))
    s = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=ref_nb.getCutoffDistance(),
                            ewaldErrorTolerance=ref_nb.getEwaldErrorTolerance(),
                            constraints=None, rigidWater=False, removeCMMotion=False)
    for f in s.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setUseDispersionCorrection(False)
    return openmm_energy(s, inpcrd.getPositions(asNumpy=True)._value)


# --------------------------------------------------------------------------------------------
# pmemd
# --------------------------------------------------------------------------------------------

def mdin(clambda, kappa_nm, grid, boundary, scmask1="HA", scmask2="FB"):
    gti_add_sc = {"scaled": 1, "unscaled": 0}[boundary]
    lambdas = ",".join(f"{v:.4f}" for v in LAMBDAS)
    return f"""S3 cross-engine single point, clambda = {clambda}
 &cntrl
  imin = 0, irest = 0, ntx = 1, nstlim = 1, dt = 0.000001,
  ntt = 0, tempi = 0.0, ntc = 1, ntf = 1, ntb = 1, cut = 9.0,
  ntpr = 1, ntwx = 0, ntwr = 1, iwrap = 0, ig = 1,
  icfe = 1, ifsc = 1, clambda = {clambda:.6f},
  timask1 = ':MLA', timask2 = ':MLB',
  scmask1 = ':MLA@{scmask1}', scmask2 = ':MLB@{scmask2}',
  scalpha = 0.5, scbeta = 12.0, gti_add_sc = {gti_add_sc},
  ifmbar = 1, bar_intervall = 1, mbar_states = {len(LAMBDAS)},
  mbar_lambda = {lambdas},
 /
 &ewald
  ew_coeff = {kappa_nm / 10.0:.10f}, nfft1 = {grid[0]}, nfft2 = {grid[1]}, nfft3 = {grid[2]},
  order = 5, eedmeth = 1, eedtbdns = 20000, vdwmeth = 0, netfrc = 0,
 /
"""


def parse_mdout(text):
    """Step-1 energies (kcal/mol), DV/DL, and the MBAR energies at each lambda."""
    block = text.split(" NSTEP =        1")[1] if " NSTEP =        1" in text else text
    def grab(key):
        m = re.search(rf"{key}\s*=\s*(-?\d+\.\d+)", block)
        return float(m.group(1)) if m else None
    out = {k: grab(k) for k in ("EPtot", "BOND", "ANGLE", "DIHED", "1-4 NB", "1-4 EEL",
                                "VDWAALS", "EELEC", "DV/DL")}
    mbar = re.findall(r"Energy at\s+(\d+\.\d+)\s*=\s*(-?\d+\.\d+)", text)
    out["mbar"] = {float(l): float(e) for l, e in mbar[:len(LAMBDAS)]}
    # pmemd reports every term on a softcore atom that it does not scale -- here the unique atoms'
    # angles and torsions -- as the region's "Softcore part", OUTSIDE EPtot and the MBAR energies.
    out["sc_eptot"] = {}
    for region in (1, 2):
        m = re.search(rf"\| TI region  {region}.*?SC_EPtot\s*=\s*(-?\d+\.\d+)", text, re.S)
        out["sc_eptot"][region] = float(m.group(1)) if m else None
    return out


def plain_mdin(kappa_nm, grid):
    return f"""S3 cross-engine calibration: an ordinary end state, no TI
 &cntrl
  imin = 0, irest = 0, ntx = 1, nstlim = 1, dt = 0.000001,
  ntt = 0, tempi = 0.0, ntc = 1, ntf = 1, ntb = 1, cut = 9.0,
  ntpr = 1, ntwx = 0, ntwr = 1, iwrap = 0, ig = 1,
 /
 &ewald
  ew_coeff = {kappa_nm / 10.0:.10f}, nfft1 = {grid[0]}, nfft2 = {grid[1]}, nfft3 = {grid[2]},
  order = 5, eedmeth = 1, eedtbdns = 20000, vdwmeth = 0, netfrc = 0,
 /
"""


#: pmemd prints EPtot, SC_EPtot and DV/DL to 1e-4 kcal/mol; its MBAR energies to 1e-8.
PRINT_RESOLUTION_KJ = 1e-4 * KJ_PER_KCAL


def verdict(calibration, rows):
    """The acceptance rule, written down before it was applied to the `unscaled` and `tail` runs.

    What a TI comparison may legitimately differ by is what the two engines already differ by on
    the ordinary end states (the calibration), and only its lambda-DEPENDENT part matters to a
    free energy: a constant offset cancels in every difference and in dU/dlambda. The calibration
    differs between A and B by |cal_B - cal_A|, each known to one EPtot print resolution. So:

      dU/dlambda:         |pmemd - md-tools| <= |cal_B - cal_A| + 2 x resolution   (DV/DL printed
                                                                                   at resolution)
      U(lambda_k) shape:  |D_k - D_0| <= |cal_B - cal_A| + 2 x resolution,
                          D_k = pmemd MBAR U(lambda_k) - md-tools U(lambda_k), per run

    The absolute offset (all terms, SC parts included) is reported and must be within
    max |cal| + 2 x resolution; it is not a free-energy quantity.
    """
    spread = abs(calibration["B"]["difference_kj"] - calibration["A"]["difference_kj"])
    tol = spread + 2 * PRINT_RESOLUTION_KJ
    worst_slope = max(abs(r["dUdl_pmemd"] - r["dUdl_md_tools"]) for r in rows)
    worst_shape = max(abs(d - r["mbar_pmemd_minus_md_tools"]["0.00"])
                      for r in rows for d in r["mbar_pmemd_minus_md_tools"].values())
    worst_abs = max(abs(r["EPtot_plus_SC_pmemd"] - r["U_md_tools"]) for r in rows)
    abs_tol = max(abs(c["difference_kj"]) for c in calibration.values()) + 2 * PRINT_RESOLUTION_KJ
    return {"tolerance_kj": tol, "absolute_tolerance_kj": abs_tol,
            "worst_dUdl_kj": worst_slope, "worst_shape_kj": worst_shape,
            "worst_absolute_kj": worst_abs,
            "PASS": bool(worst_slope <= tol and worst_shape <= tol and worst_abs <= abs_tol)}


def main():
    import openmm
    from md_tools.alchemy.hamiltonian import build_hamiltonian
    from md_tools.alchemy.softcore import SoftcoreSettings

    ap = argparse.ArgumentParser()
    ap.add_argument("--amberhome", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundary", default="scaled", choices=("scaled", "unscaled"))
    ap.add_argument("--tail", action="store_true",
                    help="the five-atom appearing chain: softcore-internal pairs and 1-4s")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pmemd = Path(args.amberhome) / "bin" / "pmemd"

    sa, sb, a_only, b_only, x = fixture(args.tail)
    scmask2 = "FB" + ("".join(",T%d" % k for k in range(1, 5)) if args.tail else "")
    report = {"fixture": f"tests/alchemy_s3_fixture.py build(True, dispersion=False, tail={args.tail})",
              "boundary_14": args.boundary, "units": "kJ/mol (pmemd kcal/mol x 4.184)"}

    # 1. the writer, proved on single copies
    writer = {}
    for copies in (("A",), ("B",)):
        top, sys_, pos, _ = layout(sa, sb, a_only, b_only, x, copies)
        parm7, rst7 = write_amber(top, sys_, pos, out / f"single_{copies[0]}")
        writer[copies[0]] = {"openmm_kj": openmm_energy(sys_, pos),
                             "prmtop_roundtrip_kj": prmtop_energy(parm7, rst7, sys_)}
        writer[copies[0]]["difference_kj"] = (writer[copies[0]]["prmtop_roundtrip_kj"]
                                              - writer[copies[0]]["openmm_kj"])
    report["writer_roundtrip"] = writer

    # 2. the dual-copy topology
    top, sys_, pos, _ = layout(sa, sb, a_only, b_only, x)
    parm7, rst7 = write_amber(top, sys_, pos, out / "ti")

    h = build_hamiltonian(sa, sb, a_only, b_only,
                          settings=SoftcoreSettings(sc_boundary_14=args.boundary))
    kappa, grid = h.record["nonbonded"]["kappa_nm_inv"], h.record["nonbonded"]["pme_grid"]
    c = openmm.Context(h.system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(x)
    state = lambda v: dict(lambda_electrostatics=v, lambda_sterics=v, lambda_bonded=v)  # noqa: E731
    mdt = {v: {"U_kj": h.energy(c, state(v)),
               "dUdl_kj": sum(h.derivatives(c, state(v)).values())} for v in LAMBDAS}

    # 3a. calibration, BEFORE the comparison: pmemd and OpenMM on the ordinary single-copy end
    # states (no TI). Their difference is the cross-engine discrepancy of everything the TI
    # comparison shares -- PME implementation, erfc table, bonded arithmetic -- and sets the scale
    # a TI difference is judged against.
    calibration = {}
    for label in ("A", "B"):
        stem = out / f"plain_{label}"
        stem.with_suffix(".mdin").write_text(plain_mdin(kappa, grid))
        subprocess.run([str(pmemd), "-O", "-i", str(stem.with_suffix(".mdin")),
                        "-p", str(out / f"single_{label}.parm7"), "-c", str(out / f"single_{label}.rst7"),
                        "-o", str(stem.with_suffix(".mdout")), "-r", str(stem.with_suffix(".restrt")),
                        "-inf", str(stem.with_suffix(".mdinfo"))], check=True, cwd=out)
        eptot = parse_mdout(stem.with_suffix(".mdout").read_text())["EPtot"] * KJ_PER_KCAL
        calibration[label] = {"pmemd_kj": eptot, "openmm_kj": writer[label]["openmm_kj"],
                              "difference_kj": eptot - writer[label]["openmm_kj"]}
    report["calibration_plain_end_states"] = calibration

    # 3b. pmemd at each clambda
    runs = {}
    for v in LAMBDAS:
        stem = out / f"clambda_{round(v * 100):03d}"
        stem.with_suffix(".mdin").write_text(mdin(v, kappa, grid, args.boundary, "HA", scmask2))
        cmd = [str(pmemd), "-O", "-i", str(stem.with_suffix(".mdin")), "-p", str(parm7),
               "-c", str(rst7), "-o", str(stem.with_suffix(".mdout")),
               "-r", str(stem.with_suffix(".restrt")), "-inf", str(stem.with_suffix(".mdinfo"))]
        subprocess.run(cmd, check=True, cwd=out)
        runs[v] = parse_mdout(stem.with_suffix(".mdout").read_text())
    report["pmemd"] = {"binary": "$AMBERHOME/bin/pmemd",   # no machine path in a report
                       "runs": {f"{v:.2f}": r for v, r in runs.items()}}

    # 4. compare. U(lambda) is compared as a difference from U(0): pmemd's EPtot and OpenMM's
    # total differ by a lambda-independent constant only if the conventions agree everywhere
    # else; that constant is reported, not hidden.
    rows = []
    for v in LAMBDAS:
        r = runs[v]
        sc = sum(r["sc_eptot"].values()) * KJ_PER_KCAL
        amber_u = {k: e * KJ_PER_KCAL + sc for k, e in r["mbar"].items()}
        rows.append({
            "lambda": v,
            "dUdl_md_tools": mdt[v]["dUdl_kj"],
            "dUdl_pmemd": None if r["DV/DL"] is None else r["DV/DL"] * KJ_PER_KCAL,
            "U_md_tools": mdt[v]["U_kj"],
            "EPtot_plus_SC_pmemd": None if r["EPtot"] is None else r["EPtot"] * KJ_PER_KCAL + sc,
            "mbar_pmemd_minus_md_tools": {f"{k:.2f}": amber_u[k] - mdt[k]["U_kj"]
                                         for k in amber_u if k in mdt},
        })
    report["comparison"] = rows
    report["verdict"] = verdict(calibration, rows)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({"writer_roundtrip": writer, "calibration": calibration,
                      "verdict": report["verdict"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
