"""M2.6: the ethane -> chloroethane edge in AMBER 26 pmemd, against MD-tools, CPU.

Rows M2.6a-M2.6e are declared in `docs/development/0.7.0/handoffs/S4-acceptance-matrix.md`
BEFORE any number here existed.

WHY THE COMPARISON RUNS UNCONSTRAINED, AND WHAT THAT COSTS

    M2 sampled with `constraints=HBonds` at 2 fs. pmemd cannot SHAKE a bond involving a softcore
    atom -- AMBER requires `noshakemask` over the TI region -- and the unique atoms of this edge
    are exactly a hydrogen (A's H6) and a chlorine (B's Cl1), so the C-H6 bond is constrained in
    OpenMM and cannot be in pmemd. Comparing across that difference would compare two
    Hamiltonians, and any disagreement would be unattributable.

    So the cross-engine rows run BOTH engines unconstrained at 1 fs, matched in everything else.
    That makes M2.6b a comparison of engines rather than of M2's own number, and the bridge back
    to M2 is M2.6e: the same MD-tools leg constrained and unconstrained, which measures what the
    constraint is worth. Two comparisons, each of one difference, instead of one comparison of
    two.

LAYOUT

    AMBER TI is dual topology: both end-state copies are present in one prmtop and never see each
    other (`timask1`/`timask2`), with the unique atoms softcore (`scmask1`/`scmask2`). Copy A
    carries the core plus A's unique atom with System A's parameters; copy B the core plus B's
    with System B's; the environment (none, in vacuum) comes from A. Generalised from S3's
    `scripts/s3_amber_crossengine.py`, which did this for its own fixture at fixed coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

#: M2.0's vacuum grid: 16 uniform windows plus the two the pilot added at the stiff end.
LAMBDAS = sorted([k / 15 for k in range(16)] + [1 / 60, 1 / 30])
TEMPERATURE_K = 300.0
KJ_PER_KCAL = 4.184


def _edge():
    """The plan, its two end-state Systems (UNCONSTRAINED, vacuum), and the two packages."""
    from tests import alchemy_fixtures as af
    from md_tools.alchemy.topology import build_topology_plan

    eth, cle = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    env = af.vacuum_environment(eth, constraints=None)      # no HBonds: see the module note
    plan = build_topology_plan(eth, cle, af.core_map(eth, cle), env, mode="hybrid")
    return plan, {"A": eth, "B": cle}


def _names(plan, packages, side: str):
    """hybrid index -> (atom name, element symbol), from the PACKAGE.

    The plan record maps local atom index to hybrid index; the names and elements live in the
    package (`atom_names`, and the RDKit molecule for elements), which is where a ligand's atom
    identity is defined.
    """
    package = packages[side]
    hyb = plan.record["endpoints"][side]["hybrid_index_of_local_atom"]
    names = list(package.atom_names)
    elements = [a.GetSymbol() for a in package.mol.GetAtoms()]
    return {h: (names[i], elements[i]) for i, h in enumerate(hyb)}


def layout(plan, packages):
    """One OpenMM System holding both copies, for ParmEd to write as a dual-topology prmtop."""
    import openmm
    from openmm import app

    sa, sb = plan.system_a, plan.system_b
    a_only, b_only = sorted(plan.a_only), sorted(plan.b_only)
    common = sorted(plan.common)
    topology = app.Topology()
    chain = topology.addChain()
    system = openmm.System()
    positions, atoms, maps = [], [], []
    for resname, members, source in (("MLA", common + a_only, sa), ("MLB", common + b_only, sb)):
        side = "A" if resname == "MLA" else "B"
        naming = _names(plan, packages, side)
        res = topology.addResidue(resname, chain)
        m = {}
        for hyb in members:
            name, element = naming.get(hyb, (f"X{hyb}", "C"))
            m[hyb] = len(atoms)
            atoms.append(topology.addAtom(name, app.Element.getBySymbol(element), res))
            system.addParticle(source.getParticleMass(hyb))
            positions.append(plan.positions_nm[hyb])
        maps.append((m, source))

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    params = [None] * len(atoms)
    for m, source in maps:
        src = next(f for f in source.getForces() if isinstance(f, openmm.NonbondedForce))
        for hyb, new in m.items():
            params[new] = src.getParticleParameters(hyb)
    for q, s, e in params:
        nb.addParticle(q, s, e)
    bonds, angles, torsions = (openmm.HarmonicBondForce(), openmm.HarmonicAngleForce(),
                               openmm.PeriodicTorsionForce())
    for m, source in maps:
        src = next(f for f in source.getForces() if isinstance(f, openmm.NonbondedForce))
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
    return topology, system, np.array(positions), maps


def end_state_system(plan, packages, side: str):
    """One physical end state alone (core + that side's unique atoms), for calibration."""
    import openmm
    from openmm import app

    source = plan.system_a if side == "A" else plan.system_b
    members = sorted(plan.common) + sorted(plan.a_only if side == "A" else plan.b_only)
    naming = _names(plan, packages, side)
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("MOL", chain)
    system = openmm.System()
    m, atoms, positions = {}, [], []
    for hyb in members:
        name, element = naming.get(hyb, (f"X{hyb}", "C"))
        m[hyb] = len(atoms)
        atoms.append(topology.addAtom(name, app.Element.getBySymbol(element), res))
        system.addParticle(source.getParticleMass(hyb))
        positions.append(plan.positions_nm[hyb])
    src = next(f for f in source.getForces() if isinstance(f, openmm.NonbondedForce))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for hyb in members:
        nb.addParticle(*src.getParticleParameters(hyb))
    for k in range(src.getNumExceptions()):
        i, j, qq, sg, ep = src.getExceptionParameters(k)
        if i in m and j in m:
            nb.addException(m[i], m[j], qq, sg, ep)
    bonds, angles, torsions = (openmm.HarmonicBondForce(), openmm.HarmonicAngleForce(),
                               openmm.PeriodicTorsionForce())
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
    return topology, system, np.array(positions), m


def write_amber(topology, system, positions, stem: Path):
    import parmed

    structure = parmed.openmm.load_topology(topology, system=system, xyz=positions * 10.0)
    parm7, rst7 = stem.with_suffix(".parm7"), stem.with_suffix(".rst7")
    structure.save(str(parm7), overwrite=True)
    structure.save(str(rst7), overwrite=True)
    return parm7, rst7


def openmm_energy(system, positions):
    import openmm

    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    return context.getState(getEnergy=True).getPotentialEnergy()._value


def prmtop_energy(parm7, rst7):
    """The prmtop read back through OpenMM: proves ParmEd wrote what we meant."""
    from openmm import app
    import openmm

    prmtop, inpcrd = app.AmberPrmtopFile(str(parm7)), app.AmberInpcrdFile(str(rst7))
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None,
                                 removeCMMotion=False)
    return openmm_energy(system, np.array(inpcrd.positions.value_in_unit(
        __import__("openmm").unit.nanometer)))


SINGLE_POINT = """single point
&cntrl
  imin = 0, irest = 0, ntx = 1, nstlim = 1, dt = 0.000001,
  ntb = 0, ntc = 1, ntf = 1, cut = 999.0, igb = 0,
  ntt = 0, ntpr = 1, ntwx = 0, ntwr = 0, ioutfm = 1,
&end
"""

TI_TEMPLATE = """TI at clambda = {clambda}
&cntrl
  imin = 0, irest = {irest}, ntx = {ntx}, nstlim = {nstlim}, dt = 0.001,
  ntb = 0, ntc = 1, ntf = 1, cut = 999.0, igb = 0,
  ntt = 3, gamma_ln = 1.0, temp0 = {temperature}, tempi = {temperature}, ig = {seed},
  ntpr = {ntpr}, ntwx = 0, ntwr = {nstlim}, ioutfm = 1,
  icfe = 1, ifsc = 1, clambda = {clambda}, scalpha = 0.5, scbeta = 12.0,
  timask1 = ':MLA', timask2 = ':MLB',
  scmask1 = ':MLA@{sc1}', scmask2 = ':MLB@{sc2}',
  ifmbar = 1, bar_intervall = {ntpr}, mbar_states = {n_states},
  mbar_lambda = {mbar_lambda},
&end
"""


def run_pmemd(work: Path, stem: str, mdin: str, parm7: Path, rst7: Path, amberhome: Path):
    (work / f"{stem}.mdin").write_text(mdin)
    out = work / f"{stem}.mdout"
    cmd = [str(amberhome / "bin" / "pmemd"), "-O", "-i", str(work / f"{stem}.mdin"),
           "-p", str(parm7), "-c", str(rst7), "-o", str(out),
           "-r", str(work / f"{stem}.rst7"), "-x", str(work / f"{stem}.nc"),
           "-inf", str(work / f"{stem}.mdinfo")]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if result.returncode != 0:
        raise SystemExit(f"pmemd failed for {stem}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return out


def parse_energy(mdout: str, key: str = "EPtot") -> float:
    """The first printed value of `key`, in kJ/mol."""
    for line in mdout.splitlines():
        if key in line:
            parts = line.split(key)[1].split("=")[1].split()
            return float(parts[0]) * KJ_PER_KCAL
    raise SystemExit(f"no {key} in mdout")


def calibrate(work: Path, amberhome: Path) -> dict:
    """M2.6a: the engines on the plain end states, at fixed coordinates."""
    plan, packages = _edge()
    rows = {"unique_atoms": {side: sorted(_names(plan, packages, side)[h][0]
                                          for h in (plan.a_only if side == "A" else plan.b_only))
                             for side in ("A", "B")}}
    for side in ("A", "B"):
        topology, system, positions, _ = end_state_system(plan, packages, side)
        stem = work / f"end_{side}"
        parm7, rst7 = write_amber(topology, system, positions, stem)
        omm = openmm_energy(system, positions)
        round_trip = prmtop_energy(parm7, rst7)
        mdout = run_pmemd(work, f"end_{side}", SINGLE_POINT, parm7, rst7, amberhome)
        amber = parse_energy(mdout.read_text())
        rows[side] = {"openmm_kJ": omm, "prmtop_roundtrip_kJ": round_trip,
                      "writer_difference_kJ": round_trip - omm,
                      "pmemd_kJ": amber, "difference_kJ": amber - omm}
    rows["lambda_dependent_part_kJ"] = abs(rows["B"]["difference_kJ"]
                                           - rows["A"]["difference_kJ"])
    rows["print_resolution_kJ"] = 2 * 1e-4 * KJ_PER_KCAL
    rows["tolerance_kJ"] = rows["lambda_dependent_part_kJ"] + rows["print_resolution_kJ"]
    return rows


def parse_mbar_blocks(mdout: str, lambdas) -> np.ndarray:
    """(n_reports, n_states) potentials in kJ/mol from pmemd's `MBAR Energy analysis` blocks.

    pmemd prints, at every `ntpr`, the potential of the current configuration at every
    `mbar_lambda` -- exactly the cross-state rows MBAR consumes, which is why this comparison can
    run BOTH engines' data through the same estimator code in `md_tools.alchemy.estimators`.
    """
    rows, current = [], None
    for line in mdout.splitlines():
        if line.startswith("MBAR Energy analysis"):
            current = []
        elif current is not None and line.startswith("Energy at"):
            current.append(float(line.split("=")[1]) * KJ_PER_KCAL)
        elif current is not None and line.strip().startswith("---"):
            if len(current) == len(lambdas):
                rows.append(current)
            current = None
    return np.array(rows)


def parse_dvdl_series(mdout: str) -> np.ndarray:
    """dV/dlambda in kJ/mol at every print step, one value per NSTEP block (the TI total)."""
    out, seen = [], None
    for line in mdout.splitlines():
        if "NSTEP =" in line:
            if seen is not None:
                out.append(seen)
            seen = None
        elif line.strip().startswith("DV/DL") and "AVERAGES" not in line and seen is None:
            seen = float(line.split("=")[1]) * KJ_PER_KCAL
    if seen is not None:
        out.append(seen)
    return np.array(out)


def pmemd_window(work: Path, amberhome: Path, parm7, rst7, clambda, *, sc1, sc2, seed,
                 equil_steps: int, prod_steps: int, ntpr: int):
    """Equilibrate, then produce, at one lambda. Returns (potentials, dvdl) for production."""
    tag = f"l{clambda:.5f}"
    mbar = ", ".join(f"{v:.5f}" for v in LAMBDAS)
    equil = TI_TEMPLATE.format(clambda=f"{clambda:.5f}", irest=0, ntx=1, nstlim=equil_steps,
                               temperature=TEMPERATURE_K, seed=seed, ntpr=equil_steps,
                               sc1=sc1, sc2=sc2, n_states=len(LAMBDAS), mbar_lambda=mbar)
    run_pmemd(work, f"{tag}_eq", equil, parm7, rst7, amberhome)
    prod = TI_TEMPLATE.format(clambda=f"{clambda:.5f}", irest=1, ntx=5, nstlim=prod_steps,
                              temperature=TEMPERATURE_K, seed=seed + 1, ntpr=ntpr,
                              sc1=sc1, sc2=sc2, n_states=len(LAMBDAS), mbar_lambda=mbar)
    out = run_pmemd(work, f"{tag}_prod", prod, parm7, work / f"{tag}_eq.rst7", amberhome)
    text = out.read_text()
    return parse_mbar_blocks(text, LAMBDAS), parse_dvdl_series(text)


def pmemd_sample_set(potentials, dvdl):
    """pmemd's rows as an md-tools sample record, so ONE estimator analyses both engines.

    The path carries a single component, `lambda_diagonal`: pmemd's `clambda` IS the diagonal
    progress coordinate of the Amber18 one-step path, and `DV/DL` is dU/ds along it. Naming it
    once here is honest; splitting one reported derivative into three components would invent a
    decomposition pmemd never reported.
    """
    from md_tools.alchemy.paths import linear_path
    from md_tools.alchemy.samples import SampleSet, window_states

    path = linear_path(["lambda_diagonal"], endpoint_a="ethane", endpoint_b="chloroethane",
                       description="pmemd clambda, the Amber18 diagonal")
    states = window_states(path, LAMBDAS, temperature_k=TEMPERATURE_K)
    ids, origin, steps, pots, ders = [], [], [], [], []
    for state, (rows, series) in zip(states, zip(potentials, dvdl)):
        n = min(len(rows), len(series))
        for i in range(n):
            ids.append(f"{state.state_id}:{i:06d}")
            origin.append(state.state_id)
            steps.append(i)
            pots.append(rows[i])
            ders.append(series[i])
    return SampleSet(states, path, ids, origin, steps, np.array(pots), None,
                     {"lambda_diagonal": np.array(ders)},
                     provenance={"engine": "AMBER 26 pmemd (CPU)", "row": "M2.6"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "ti"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--amberhome", default=os.environ.get("AMBERHOME",
                                                              "/data3/data/chen/software/amber26"))
    parser.add_argument("--ns", type=float, default=1.0, help="production per window, ns")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    amberhome = Path(args.amberhome)

    if args.phase == "calibrate":
        rows = calibrate(work, amberhome)
        (work / "calibration.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
        for side in ("A", "B"):
            r = rows[side]
            print(f"end state {side}: OpenMM {r['openmm_kJ']:.6f}  pmemd {r['pmemd_kJ']:.6f}  "
                  f"difference {r['difference_kJ']:+.3e} kJ/mol  "
                  f"(ParmEd round trip {r['writer_difference_kJ']:+.1e})")
        print(f"lambda-dependent part {rows['lambda_dependent_part_kJ']:.3e}, "
              f"tolerance {rows['tolerance_kJ']:.3e} kJ/mol")
    else:
        from md_tools.alchemy import estimators as est

        plan, packages = _edge()
        topology, system, positions, _ = layout(plan, packages)
        parm7, rst7 = write_amber(topology, system, positions, work / "dual")
        sc1 = ",".join(sorted(_names(plan, packages, "A")[h][0] for h in plan.a_only))
        sc2 = ",".join(sorted(_names(plan, packages, "B")[h][0] for h in plan.b_only))
        prod_steps = int(args.ns * 1_000_000)
        potentials, dvdl = [], []
        for i, clambda in enumerate(LAMBDAS):
            rows, series = pmemd_window(work, amberhome, parm7, rst7, clambda, sc1=sc1, sc2=sc2,
                                        seed=args.seed + 13 * i, equil_steps=50_000,
                                        prod_steps=prod_steps, ntpr=1000)
            potentials.append(rows)
            dvdl.append(series)
            print(f"  lambda {clambda:.5f}: {len(rows)} MBAR rows, {len(series)} dV/dl, "
                  f"mean dV/dl {np.mean(series):+.3f} kJ/mol", flush=True)
        samples = pmemd_sample_set(potentials, dvdl)
        result = est.analyze(samples)
        (work / "pmemd_analysis.json").write_text(json.dumps(result, indent=1, sort_keys=True,
                                                             default=str))
        for name, e in result["estimates"].items():
            print(f"{name:12s} {e['delta_g_kcal_mol']:+.4f} +- {e['sigma_kcal_mol']:.4f} kcal/mol")


if __name__ == "__main__":
    main()
