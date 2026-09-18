"""S4: fixed-lambda windows through the shared preflight and checkpoint transaction.

The Hamiltonian is an OpenMM harmonic model with the same staged, nonlinear lambda dependence as
the frozen estimator fixture, so a window campaign has an EXACT free energy to be judged against.
Everything here runs on the CPU platform with an explicit `--cpu`: it exercises the runtime and
is NOT CUDA evidence (see the S4 acceptance matrix).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")   # else pymbar -> JAX takes every GPU
openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from md_tools.alchemy import estimators as est  # noqa: E402
from md_tools.alchemy.paths import staged_path  # noqa: E402
from md_tools.alchemy.restraints import BoreschRestraint  # noqa: E402
from md_tools.alchemy.samples import KJ_PER_KCAL, concatenate, kt_kj_mol, window_states  # noqa: E402
from md_tools.alchemy.windows import (ComposedHamiltonian, CrossStateEvaluator,  # noqa: E402
                                      ParametricHamiltonian, WindowError, WindowSettings,
                                      read_window_samples, run_window, window_paths,
                                      RESTRAINT_FORCE_GROUP)
from md_tools.run.preflight import PreflightError  # noqa: E402

N = 6
K0, K1, A, C = 100.0, 400.0, 0.1, 5.0
T = 300.0
NAMES = ("lambda_electrostatics", "lambda_sterics")


def harmonic_system(n=N):
    s = openmm.System()
    # CustomCompoundBondForce, one particle per bond: CustomExternalForce cannot request
    # parameter derivatives, and the whole point is a complete dU/d lambda from OpenMM.
    f = openmm.CustomCompoundBondForce(
        1, "0.5*kk*((x1-mu)^2 + (y1-mu)^2 + (z1-mu)^2) + cc*lambda_electrostatics^2;"
        f" kk = {K0} + ({K1} - {K0})*lambda_sterics; mu = {A}*lambda_electrostatics;"
        f" cc = {C}/{n}")
    for name in NAMES:
        f.addGlobalParameter(name, 0.0)
        f.addEnergyParameterDerivative(name)
    for i in range(n):
        s.addParticle(40.0)
        f.addBond([i], [])
    s.addForce(f)
    return s


def exact_kj_mol(n=N):
    return 1.5 * n * kt_kj_mol(T) * math.log(K1 / K0) + C


@pytest.fixture()
def model(tmp_path):
    system = harmonic_system()
    ham = ParametricHamiltonian(system, NAMES)
    top = app.Topology()
    chain = top.addChain()
    for i in range(N):
        res = top.addResidue("AR", chain)
        top.addAtom("AR", app.Element.getBySymbol("Ar"), res)
    pos = [openmm.Vec3(0.01 * i, 0.0, 0.0) for i in range(N)] * unit.nanometer
    pdb = tmp_path / "model.pdb"
    with pdb.open("w") as h:
        app.PDBFile.writeFile(top, pos, h)
    sys_xml = tmp_path / "model.xml"
    sys_xml.write_text(openmm.XmlSerializer.serialize(system))
    path = staged_path([("lambda_electrostatics", 0.5), ("lambda_sterics", 1.0)],
                       endpoint_a="harmonic K0 at 0", endpoint_b="harmonic K1 at A")
    states = window_states(path, [0.0, 0.25, 0.5, 0.75, 1.0], temperature_k=T)
    return dict(ham=ham, pdb=pdb, xml=sys_xml, path=path, states=states, tmp=tmp_path)


def _run(model, wid, settings, out=None, **kw):
    return run_window(topology=model["pdb"], system=model["xml"], hamiltonian=model["ham"],
                      path=model["path"], states=model["states"], window_id=wid,
                      out_dir=out or model["tmp"] / "run", settings=settings, cpu=True, **kw)


SHORT = WindowSettings(steps=2000, report_interval=100, checkpoint_interval=500,
                       equilibration_steps=200, timestep_fs=2.0, seed=11)


# ---------------------------------------------------------------------- evaluation
def test_cross_state_energies_and_derivatives_are_the_model(model):
    ev = CrossStateEvaluator(model["ham"], model["states"],
                             platform=openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (N, 3))
    for origin in model["states"]:
        e, d = ev.evaluate(x, None, origin)
        for k, st in enumerate(model["states"]):
            le, ls = st.components["lambda_electrostatics"], st.components["lambda_sterics"]
            kk = K0 + (K1 - K0) * ls
            ref = 0.5 * kk * np.sum((x - A * le) ** 2) + C * le * le
            assert e[k] == pytest.approx(ref, rel=1e-10)
        le, ls = origin.components["lambda_electrostatics"], origin.components["lambda_sterics"]
        kk = K0 + (K1 - K0) * ls
        assert d["lambda_sterics"] == pytest.approx(0.5 * (K1 - K0) * np.sum((x - A * le) ** 2))
        assert d["lambda_electrostatics"] == pytest.approx(
            -kk * A * np.sum(x - A * le) + 2 * C * le, rel=1e-9, abs=1e-9)


def test_derivative_matches_central_differences_at_several_steps(model):
    """dU/d lambda_k against central differences in the interior, one-sided at the ends."""
    ctx = openmm.Context(model["ham"].system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.random.default_rng(1).normal(0, 0.1, (N, 3)))
    ham = model["ham"]

    def energy(state):
        ham.set_state(ctx, state)
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    for base in ({"lambda_electrostatics": 0.3, "lambda_sterics": 0.6},
                 {"lambda_electrostatics": 0.0, "lambda_sterics": 0.0},
                 {"lambda_electrostatics": 1.0, "lambda_sterics": 1.0}):
        d = ham.derivatives(ctx, base)
        for name in NAMES:
            errs = []
            for h in (1e-2, 1e-3, 1e-4):
                v = base[name]
                if 0 < v - h and v + h < 1:
                    fd = (energy({**base, name: v + h}) - energy({**base, name: v - h})) / (2 * h)
                elif v - h < 0:
                    fd = (-3 * energy(base) + 4 * energy({**base, name: v + h})
                          - energy({**base, name: v + 2 * h})) / (2 * h)
                else:
                    fd = (3 * energy(base) - 4 * energy({**base, name: v - h})
                          + energy({**base, name: v - 2 * h})) / (2 * h)
                errs.append(abs(fd - d[name]))
            assert min(errs) < 1e-5 * max(1.0, abs(d[name])), (name, base, errs)


def test_path_and_hamiltonian_must_name_the_same_parameters(model, tmp_path):
    bad = staged_path([("lambda_sterics", 1.0)], endpoint_a="a", endpoint_b="b")
    states = window_states(bad, [0.0, 1.0], temperature_k=T)
    with pytest.raises(WindowError, match="Missing from the path"):
        run_window(topology=model["pdb"], system=model["xml"], hamiltonian=model["ham"],
                   path=bad, states=states, window_id="w000", out_dir=tmp_path / "x",
                   settings=SHORT, cpu=True)
    assert not (tmp_path / "x").exists()


# ---------------------------------------------------------------------- refusal before output
def test_a_system_file_that_is_not_the_hamiltonians_is_refused(model, tmp_path):
    other = harmonic_system()
    other.getForce(0).setGlobalParameterDefaultValue(0, 0.5)
    f = tmp_path / "other.xml"
    f.write_text(openmm.XmlSerializer.serialize(other))
    with pytest.raises(WindowError, match="not the Hamiltonian's System"):
        run_window(topology=model["pdb"], system=f, hamiltonian=model["ham"], path=model["path"],
                   states=model["states"], window_id="w000", out_dir=tmp_path / "out",
                   settings=SHORT, cpu=True)
    assert not (tmp_path / "out").exists()


def test_npt_without_a_box_is_refused_by_the_shared_preflight(model, tmp_path):
    states = window_states(model["path"], [0.0, 0.25, 0.5, 0.75, 1.0], temperature_k=T,
                           pressure_bar=1.0)
    with pytest.raises(PreflightError, match="no periodic box"):
        run_window(topology=model["pdb"], system=model["xml"], hamiltonian=model["ham"],
                   path=model["path"], states=states, window_id="w000",
                   out_dir=tmp_path / "npt", settings=SHORT, cpu=True)
    assert not (tmp_path / "npt").exists()


def test_check_creates_nothing(model, tmp_path, capsys):
    res = _run(model, "w000", SHORT, out=tmp_path / "chk", check=True)
    assert res["disposition"] == "checked"
    assert "--check passed. Nothing was created." in capsys.readouterr().out
    assert not (tmp_path / "chk").exists()


def test_a_prepared_preflight_is_consumed_not_replanned(model, tmp_path, monkeypatch):
    from md_tools.run import preflight as pf
    out = tmp_path / "prep"
    p = window_paths(out, "w001")
    checked = pf.preflight_stage(
        topology=model["pdb"], system=model["xml"], output=p["out"], log=p["record"],
        restart=p["restart"], checkpoint=p["checkpoint"], cpu=True,
        protocol="alchemical window w001", timestep_fs=2.0, ensemble="NVT")
    monkeypatch.setattr(pf, "preflight_stage",
                        lambda **kw: pytest.fail("re-planned a prepared preflight"))
    assert _run(model, "w001", SHORT, out=out, prepared=checked)["rows"] == 21


def test_settings_that_would_round_are_refused():
    with pytest.raises(WindowError, match="rounded"):
        WindowSettings(steps=1050, report_interval=100, checkpoint_interval=500)
    with pytest.raises(WindowError, match="splits a report"):
        WindowSettings(steps=3000, report_interval=200, checkpoint_interval=500)


# ---------------------------------------------------------------------- one window, and resume
def test_window_runs_writes_samples_and_verifies(model):
    res = _run(model, "w002", SHORT)
    assert res["disposition"] == "fresh" and res["rows"] == 21
    paths = window_paths(model["tmp"] / "run", "w002")
    record = json.loads(paths["record"].read_text())
    assert record["acceleration"]["platform_selection"] == "cli-override"
    s = read_window_samples(paths["samples"], record)
    assert set(s.origin_state) == {"w002"}
    assert list(s.step) == list(range(200, 2201, 100))
    assert "rank" not in paths["samples"].read_text().splitlines()[0]
    before = {p: p.stat().st_mtime_ns for p in paths.values() if p.exists()}
    again = _run(model, "w002", SHORT)
    assert again["disposition"] == "verified-complete"
    assert {p: p.stat().st_mtime_ns for p in paths.values() if p.exists()} == before


def test_interrupted_window_resumes_to_the_same_rows(model):
    whole = _run(model, "w001", SHORT, out=model["tmp"] / "whole")
    part = _run(model, "w001", SHORT, out=model["tmp"] / "part", stop_after_steps=1200)
    assert part["disposition"] == "interrupted"
    done = _run(model, "w001", SHORT, out=model["tmp"] / "part")
    assert done["disposition"] == "resumed" and done["rows"] == whole["rows"]
    a = window_paths(model["tmp"] / "whole", "w001")["samples"].read_text().splitlines()
    b = window_paths(model["tmp"] / "part", "w001")["samples"].read_text().splitlines()
    assert [r.split(",")[2] for r in a] == [r.split(",")[2] for r in b]   # the same step grid
    # CPU platform: a restored checkpoint continues the same realisation
    assert a == b


def test_a_changed_committed_row_is_refused_without_touching_anything(model):
    out = model["tmp"] / "tamper"
    _run(model, "w003", SHORT, out=out, stop_after_steps=1200)
    paths = window_paths(out, "w003")
    lines = paths["samples"].read_text().splitlines()
    fields = lines[2].split(",")
    fields[-1] = repr(float(fields[-1]) + 1.0)
    lines[2] = ",".join(fields)
    paths["samples"].write_text("\n".join(lines) + "\n")
    snapshot = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(WindowError, match="not the rows the checkpoint committed"):
        _run(model, "w003", SHORT, out=out)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == snapshot


def test_a_changed_definition_cannot_continue_a_checkpoint(model):
    out = model["tmp"] / "redefine"
    _run(model, "w000", SHORT, out=out, stop_after_steps=1200)
    other = WindowSettings(steps=2000, report_interval=100, checkpoint_interval=500,
                           equilibration_steps=200, timestep_fs=2.0, seed=12)
    snapshot = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(WindowError, match="different window definition"):
        _run(model, "w000", other, out=out)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == snapshot


def test_overwrite_moves_the_previous_window_aside(model):
    out = model["tmp"] / "ow"
    _run(model, "w004", SHORT, out=out)
    res = _run(model, "w004", SHORT, out=out, overwrite=True)
    assert res["disposition"] == "fresh"
    assert list(out.glob("w004.samples.csv.replaced-*"))


# ---------------------------------------------------------------------- the campaign
@pytest.mark.slow
def test_window_campaign_recovers_the_exact_free_energy(model):
    """Five windows, Langevin sampling, every estimator, judged by the 0.7.0 gate against the
    closed-form answer. CPU platform: runtime evidence, not CUDA evidence."""
    settings = WindowSettings(steps=100_000, report_interval=200, checkpoint_interval=20_000,
                              equilibration_steps=2000, timestep_fs=2.0, seed=7)
    out = model["tmp"] / "campaign"
    parts = []
    for st in model["states"]:
        _run(model, st.state_id, settings, out=out)
        p = window_paths(out, st.state_id)
        parts.append(read_window_samples(p["samples"], json.loads(p["record"].read_text())))
    samples = concatenate(parts)
    result = est.analyze(samples)
    exact = exact_kj_mol() / KJ_PER_KCAL
    print(f"\nS4 W7 campaign: exact {exact:.4f} kcal/mol, samples {samples.counts()}")
    for name, e in result["estimates"].items():
        print(f"  {name:12s} {e['delta_g_kcal_mol']:.4f} +- {e['sigma_kcal_mol']:.4f} kcal/mol")
    # W7 as recorded in the acceptance matrix (2026-09-19): five windows are too coarse for 18
    # coordinates. MBAR, BAR and EXP forward pass the gate; EXP reverse fails it and TI is
    # inconclusive. The gate is not weakened. What is asserted about those two is W7b: neither
    # failure is SILENT -- each is flagged by its own diagnostics.
    verdicts = _verdicts(result, exact)
    for name in ("MBAR", "BAR", "EXP_forward"):
        assert verdicts[name]["verdict"] == "PASS", (name, verdicts[name])
    _assert_no_silent_failure(result, verdicts)


@pytest.mark.slow
def test_w8_seventeen_windows_pass_every_estimator(model):
    """W8, defined after W7 and before sampling: the same model and gate on 17 windows."""
    states = window_states(model["path"], [k / 16 for k in range(17)], temperature_k=T)
    model = dict(model, states=states)
    settings = WindowSettings(steps=100_000, report_interval=200, checkpoint_interval=20_000,
                              equilibration_steps=2000, timestep_fs=2.0, seed=8)
    out = model["tmp"] / "w8"
    parts = []
    for st in states:
        _run(model, st.state_id, settings, out=out)
        p = window_paths(out, st.state_id)
        parts.append(read_window_samples(p["samples"], json.loads(p["record"].read_text())))
    result = est.analyze(concatenate(parts))
    exact = exact_kj_mol() / KJ_PER_KCAL
    verdicts = _verdicts(result, exact)
    print(f"\nS4 W8 campaign: exact {exact:.4f} kcal/mol")
    for name, v in verdicts.items():
        e = result["estimates"][name]
        print(f"  {name:12s} {e['delta_g_kcal_mol']:.4f} +- {e['sigma_kcal_mol']:.4f} "
              f"kcal/mol  {v['verdict']}")
    for name, v in verdicts.items():
        assert v["verdict"] == "PASS", (name, v)


def _verdicts(result, exact):
    out = {}
    for name in ("MBAR", "BAR", "EXP_forward", "EXP_reverse", "TI"):
        e = result["estimates"][name]
        sigma_int = (e["diagnostics"]["quadrature_discrepancy_kJ_mol"] / KJ_PER_KCAL
                     if name == "TI" else 0.0)
        out[name] = est.agreement_gate(e["delta_g_kcal_mol"], e["sigma_kcal_mol"], exact,
                                       integration_sigma_kcal=sigma_int)
    return out


def _assert_no_silent_failure(result, verdicts):
    for name, v in verdicts.items():
        if v["verdict"] == "PASS":
            continue
        d = result["estimates"][name]["diagnostics"]
        flagged = d.get("poor_overlap") or (
            name == "TI" and v["verdict"] == "INCONCLUSIVE"
            and d["quadrature_discrepancy_kJ_mol"] / KJ_PER_KCAL > est.GATE_MAX_COMBINED_SIGMA_KCAL_MOL / 3)
        assert flagged, (f"{name} missed the gate ({v}) and nothing in its diagnostics says so")


# ---------------------------------------------------------------------- the restraint layer
def test_composed_hamiltonian_owns_lambda_restraints(model):
    r = BoreschRestraint((0, 1, 2), (3, 4, 5), 0.5, 1.6, 1.3, 0.5, -2.0, 2.9,
                         4184.0, 41.84, 41.84, 41.84, 41.84, 41.84)
    ham = ComposedHamiltonian(model["ham"], r)
    assert ham.parameter_names == NAMES + ("lambda_restraints",)
    groups = {f.getForceGroup() for f in ham.system.getForces()}
    assert RESTRAINT_FORCE_GROUP in groups
    ctx = openmm.Context(ham.system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    x = np.random.default_rng(3).normal(0, 0.3, (N, 3))
    ctx.setPositions(x)
    state = {"lambda_electrostatics": 0.2, "lambda_sterics": 0.4, "lambda_restraints": 0.7}
    d = ham.derivatives(ctx, state)
    assert d["lambda_restraints"] == pytest.approx(r.energy_kj_mol(x, 1.0), rel=1e-10)
    ham.set_state(ctx, state)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    inner = model["ham"]
    c2 = openmm.Context(inner.system, openmm.VerletIntegrator(0.001),
                        openmm.Platform.getPlatformByName("Reference"))
    c2.setPositions(x)
    inner.set_state(c2, {k: state[k] for k in NAMES})
    e_inner = c2.getState(getEnergy=True).getPotentialEnergy()._value
    assert e == pytest.approx(e_inner + r.energy_kj_mol(x, 0.7), rel=1e-10)
    with pytest.raises(WindowError, match="exactly"):
        ham.set_state(ctx, {k: state[k] for k in NAMES})


# ---------------------------------------------------------------------- NPT, periodic, PME water
def _npt_model(tmp_path):
    """Two dummy particles joined by a lambda-dependent harmonic bond (r0 = 0) in TIP3P water
    under PME. The dummies carry no charge and no LJ, so the bond's free energy is exactly
    (3/2) kT ln(K1/K0) + C whatever the solvent, the box or its fluctuations do."""
    ff = app.ForceField("amber14/tip3p.xml")
    m = app.Modeller(app.Topology(), [])
    m.addSolvent(ff, boxSize=openmm.Vec3(2.2, 2.2, 2.2) * unit.nanometer, model="tip3p")
    system = ff.createSystem(m.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometer, constraints=app.HBonds)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    top = app.Topology()
    top.setPeriodicBoxVectors(m.topology.getPeriodicBoxVectors())
    atoms = {}
    for chain in m.topology.chains():
        c = top.addChain()
        for res in chain.residues():
            r = top.addResidue(res.name, c)
            for a in res.atoms():
                atoms[a] = top.addAtom(a.name, a.element, r)
    for bond in m.topology.bonds():
        top.addBond(atoms[bond[0]], atoms[bond[1]])
    dc = top.addChain()
    first = system.getNumParticles()
    for _ in range(2):
        top.addAtom("D", app.Element.getBySymbol("C"), top.addResidue("DUM", dc))
        system.addParticle(12.0)
        nb.addParticle(0.0, 0.1, 0.0)
    bond = openmm.CustomBondForce(
        f"0.5*kk*r^2 + {C}*lambda_electrostatics^2; kk = {K0} + ({K1} - {K0})*lambda_sterics")
    for name in NAMES:
        bond.addGlobalParameter(name, 0.0)
        bond.addEnergyParameterDerivative(name)
    bond.addBond(first, first + 1, [])
    bond.setUsesPeriodicBoundaryConditions(True)
    system.addForce(bond)
    pos = list(m.positions.value_in_unit(unit.nanometer)) + [openmm.Vec3(1.1, 1.1, 1.1),
                                                             openmm.Vec3(1.15, 1.1, 1.1)]
    pdb = tmp_path / "npt.pdb"
    with pdb.open("w") as h:
        app.PDBFile.writeFile(top, pos * unit.nanometer, h)
    xml = tmp_path / "npt.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    path = staged_path([("lambda_electrostatics", 0.5), ("lambda_sterics", 1.0)],
                       endpoint_a="bond K0", endpoint_b="bond K1")
    states = window_states(path, [0.0, 0.25, 0.5, 0.75, 1.0], temperature_k=T,
                           pressure_bar=1.01325)
    return dict(ham=ParametricHamiltonian(system, NAMES), pdb=pdb, xml=xml, path=path,
                states=states, tmp=tmp_path)


@pytest.mark.slow
def test_npt_windows_carry_pv_resume_and_recover_the_exact_free_energy(tmp_path):
    model = _npt_model(tmp_path)
    settings = WindowSettings(steps=10_000, report_interval=50, checkpoint_interval=2_500,
                              equilibration_steps=1_000, timestep_fs=2.0, seed=5)
    out = tmp_path / "npt"
    parts = []
    for st in model["states"]:
        if st.state_id == "w002":
            assert _run(model, st.state_id, settings, out=out,
                        stop_after_steps=4_000)["disposition"] == "interrupted"
        _run(model, st.state_id, settings, out=out)
        p = window_paths(out, st.state_id)
        record = json.loads(p["record"].read_text())
        parts.append(read_window_samples(p["samples"], record))
    samples = concatenate(parts)
    assert samples.ensemble == "NPT" and np.ptp(samples.volume_nm3) > 0.05   # the box moved
    u = samples.reduced_potential()
    pv = 1.01325 * 0.0602214076 * samples.volume_nm3 / samples.kt
    np.testing.assert_allclose(u - samples.potential_kj_mol / samples.kt, pv[:, None] *
                               np.ones_like(u), rtol=1e-12)
    result = est.analyze(samples)
    exact = (1.5 * kt_kj_mol(T) * math.log(K1 / K0) + C) / KJ_PER_KCAL
    print(f"\nS4 N1 NPT campaign: exact {exact:.4f} kcal/mol, samples {samples.counts()}")
    for name, e in result["estimates"].items():
        print(f"  {name:12s} {e['delta_g_kcal_mol']:.4f} +- {e['sigma_kcal_mol']:.4f} kcal/mol")
    for name in ("MBAR", "BAR"):
        e = result["estimates"][name]
        gate = est.agreement_gate(e["delta_g_kcal_mol"], e["sigma_kcal_mol"], exact)
        assert gate["verdict"] == "PASS", (name, gate)
