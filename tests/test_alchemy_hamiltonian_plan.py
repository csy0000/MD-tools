"""The softcore Hamiltonian over S2's real topology plans (fixture `alchemy-endpoints/1`).

Real AM1-BCC packages -- ethane, chloroethane, ethanol -- in TIP3P under PME with the dispersion
correction on, combined by `md_tools.alchemy.topology.build_topology_plan` (hybrid). What only a
real plan can show:

* the plan's dummy-removed junction terms are accepted (these fixtures are built with
  `junction_policy="separable"` on purpose; the default retain-all leaves nothing differing, which
  `test_retain_all_leaves_no_bonded_term_on_the_lambda_path` covers), and nothing else about a
  unique atom's bonded terms differs;
* under the default boundary rule the Hamiltonian's end states ARE the plan's end-state Systems,
  energy for energy, dispersion correction included -- for pentane too, whose appearing propyl
  group has internal 1-4s and 1-5 pairs that the plan keeps physical at its dummy end;
* the derivatives are the finite-difference limits of the energy, and the Amber18 path is not
  linear (the AIS identity dU/dlambda = U1 - U0 fails);
* S4's window layer accepts the Hamiltonian for a path over its three components.

Reference platform, float64: fixed-coordinate evaluation, nothing propagated.
"""
from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_fixtures as af  # noqa: E402
from tests import alchemy_s3_fixture as fx3  # noqa: E402
from md_tools.alchemy.hamiltonian import build_hamiltonian, from_plan  # noqa: E402
from md_tools.alchemy.softcore import SoftcoreSettings  # noqa: E402

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")
#: Tolerance fixed before comparing: S2's end-state recovery holds these Systems to 1e-7 kJ/mol
#: per force class on Reference (handoffs/S2.md), and the Hamiltonian adds float64 arithmetic
#: over |E| ~ 1e4, ~1e-11 relative.
ENDPOINT_TOL = 1e-7


def _state(v):
    return dict(zip(NAMES, v))


def _context(system, x):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(x)
    return c


def _energy(system, x):
    return _context(system, x).getState(getEnergy=True).getPotentialEnergy()._value


@pytest.fixture(scope="module", params=["chloroethane", "ethanol", "pentane", "complex-cmap"])
def plan(request):
    """`pentane` (fixture internal-v1) leaves a propyl group appearing, with internal 1-4s and 1-5
    pairs: the plan keeps them physical at its dummy end (contract section 4), in the NonbondedForce
    and in its UniqueGroupInternalNonbonded force. `complex-cmap` (complex-cmap-v1) is ethane ->
    chloroethane beside a capped alanine (ff19SB, CMAP) in OPC water: 180 environment virtual
    sites, and the environment's 1-4 scale applied to both endpoints' ligand exceptions."""
    from md_tools.alchemy.topology import build_topology_plan
    a = af.package(af.ETHANE)
    name = request.param
    b = af.package({"chloroethane": af.CHLOROETHANE, "ethanol": af.ETHANOL,
                    "pentane": af.PENTANE, "complex-cmap": af.CHLOROETHANE}[name])
    env = af.complex_environment(af.CMAP_ROOT) if name == "complex-cmap" else af.water_environment()
    # `separable` DELIBERATELY: under the default retain-all no bonded term differs between the end
    # states, which is the point of that policy, and these tests are what covers the mixed-force
    # machinery on terms that DO differ. The mirror case is
    # `test_retain_all_leaves_no_bonded_term_on_the_lambda_path`.
    return build_topology_plan(a, b, af.core_map(a, b), env, mode="hybrid",
                               junction_policy="separable")


def test_the_end_states_are_the_plans_systems(plan):
    x = plan.positions_nm
    h = from_plan(plan)
    internal = plan.record["nonbonded"].get("unique_group_internal", {}).get("pairs", [])
    assert h.record["plan_internal_pairs_checked"] == len(internal)
    c = _context(h.system, x)
    assert any(f["differing_terms"] for f in h.record["bonded_mixed_forces"])   # dummy-removed
    assert h.record["nonbonded"]["use_dispersion_correction"] is True
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(_energy(plan.system_a, x), abs=ENDPOINT_TOL)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(_energy(plan.system_b, x), abs=ENDPOINT_TOL)


def test_retain_all_leaves_no_bonded_term_on_the_lambda_path():
    """The default policy's mirror of the fixture above: with every junction term retained at both
    ends, NO bonded term differs, so the mixed bonded force carries nothing that moves with lambda
    and dU/dlambda_bonded is exactly zero. Together the two cover both policies, and neither can
    silently stop testing its own."""
    from md_tools.alchemy.topology import build_topology_plan
    a, b = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    plan = build_topology_plan(a, b, af.core_map(a, b), af.water_environment(), mode="hybrid",
                               junction_policy="retain-all")
    assert plan.record["junction_policy"] == "retain-all"
    h = from_plan(plan)
    assert not any(f["differing_terms"] for f in h.record["bonded_mixed_forces"])
    x = plan.positions_nm
    c = _context(h.system, x)
    parts = h.derivative_components(c, _state((0.5, 0.5, 0.5)))["lambda_bonded"]
    assert sum(parts.values()) == 0.0, parts
    box = plan.system_a.getDefaultPeriodicBoxVectors()[0][0]._value
    split = fx3.bonded_derivative_split(plan.system_a, plan.system_b, set(plan.a_only) | set(plan.b_only),
                                        x, box=box, policy="retain-all")
    assert split["unique_touching"] == 0.0 and split["warning"] is None, split
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(_energy(plan.system_a, x), abs=ENDPOINT_TOL)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(_energy(plan.system_b, x), abs=ENDPOINT_TOL)


def test_the_amber18_boundary_rule_changes_the_end_states_and_says_so(plan):
    """`unscaled` keeps a unique atom's 1-4s with the core at full strength: the end states are
    then NOT the plan's Systems, by exactly those 1-4s."""
    x = plan.positions_nm
    h = build_hamiltonian(plan.system_a, plan.system_b, plan.a_only, plan.b_only,
                          settings=SoftcoreSettings(sc_boundary_14="unscaled"))
    assert h.record["softcore"]["sc_boundary_14"] == "unscaled"
    c = _context(h.system, x)
    assert abs(h.energy(c, _state((0, 0, 0))) - _energy(plan.system_a, x)) > 1e-3
    assert abs(h.energy(c, _state((1, 1, 1))) - _energy(plan.system_b, x)) > 1e-3


def _fd(f, lam, h):
    if lam - h < 0:
        return (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
    if lam + h > 1:
        return (3 * f(lam) - 4 * f(lam - h) + f(lam - 2 * h)) / (2 * h)
    return (f(lam + h) - f(lam - h)) / (2 * h)


#: OpenMM re-integrates a custom long-range correction by quadrature whenever lambda_sterics
#: changes, so the dispersion carrier's energy departs from exact linearity by a small amount
#: (4.9e-8 kJ/mol, 6e-8 relative, on the chloroethane plan). Its derivative is the exact slope. The
#: carrier is therefore checked on its own, against its OWN measured non-linearity: an absolute
#: bound fixed on one plan (5e-8 kJ/mol) did not transfer to pentane's larger carrier, and was
#: replaced after that failure by this measurement.
DISPERSION = "dispersion"
QUADRATURE_RELATIVE = 1e-6


def test_derivatives_are_the_limit_of_the_energy_and_the_path_is_not_linear(plan):
    from md_tools.alchemy.hamiltonian import FORCE_GROUPS
    x = plan.positions_nm
    h = from_plan(plan)
    c = _context(h.system, x)
    rest = set(FORCE_GROUPS.values()) - {FORCE_GROUPS[DISPERSION]}

    def energy(t, groups):
        h.set_state(c, _state(t))
        return c.getState(getEnergy=True, groups=groups).getPotentialEnergy()._value
    for base in (0.0, 0.5, 1.0):
        parts = h.derivative_components(c, _state((base,) * 3))
        for k, name in enumerate(NAMES):
            def f(v, k=k, groups=rest):
                t = [base] * 3
                t[k] = v
                return energy(t, groups)
            want = sum(v for g, v in parts[name].items() if g != DISPERSION)
            # three steps and Richardson on each adjacent pair: S2 places a dummy group without a
            # clash check, so at its physical end it can overlap water (pentane's propyl gives
            # dU/dlambda_sterics ~ 7e3 kJ/mol at lambda = 1), and there the O(h^2) truncation
            # needs h = 1e-4 -- as in S3's own tail fixture
            fds = [_fd(f, base, h) for h in (1e-2, 1e-3, 1e-4)]
            richardson = [(100 * fine - coarse) / 99 for coarse, fine in zip(fds, fds[1:])]
            best = min(abs(want - r) for r in richardson)
            assert best < 2e-5 * max(1.0, abs(want)), (base, name, want, richardson)
        def g(v):
            return energy([base, v, base], {FORCE_GROUPS[DISPERSION]})
        grid = [k / 10 for k in range(11)]
        values = [g(v) for v in grid]
        residual = max(abs(e - ((1 - v) * values[0] + v * values[-1])) for v, e in zip(grid, values))
        assert residual <= QUADRATURE_RELATIVE * max(abs(values[0]), abs(values[-1])), residual
        slope = parts["lambda_sterics"][DISPERSION]
        assert abs(slope - (values[-1] - values[0])) <= 2 * residual + 1e-12, (base, slope)
    # Along a linear path dU/dlambda is the same at every lambda (AIS's U1 - U0). Here it is not.
    # (Comparing the midpoint slope with the secant would be no test: a curved U(lambda) can
    # have its midpoint slope equal to its secant, and at these coordinates it nearly does.)
    slope_0 = sum(h.derivatives(c, _state((0, 0, 0))).values())
    slope_1 = sum(h.derivatives(c, _state((1, 1, 1))).values())
    assert abs(slope_1 - slope_0) > 1.0, (slope_0, slope_1)


def test_the_dispersion_slope_is_the_end_states_own(plan):
    """d/dlambda_sterics of the dispersion term is D_B - D_A, with D_X the correction OpenMM applies
    to the plan's end-state System X -- measured with the correction on and off."""
    x = plan.positions_nm

    def correction(system):
        s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        on = _energy(s, x)
        for f in s.getForces():
            if isinstance(f, openmm.NonbondedForce):
                f.setUseDispersionCorrection(False)
        return on - _energy(s, x)
    h = from_plan(plan)
    c = _context(h.system, x)
    for base in (0.0, 0.5, 1.0):
        slope = h.derivative_components(c, _state((base,) * 3))["lambda_sterics"][DISPERSION]
        assert slope == pytest.approx(correction(plan.system_b) - correction(plan.system_a), abs=1e-7)


def test_the_window_layer_accepts_it_for_a_three_component_path(plan):
    from md_tools.alchemy.paths import AlchemicalPath, Knot
    from md_tools.alchemy.windows import check_hamiltonian_matches_path
    h = from_plan(plan)
    path = AlchemicalPath(knots=(Knot(0.0, _state((0, 0, 0))), Knot(1.0, _state((1, 1, 1)))),
                          endpoint_a="ethane", endpoint_b="the substituted molecule")
    check_hamiltonian_matches_path(h, path)


# ------------------------------------------------------------------------------------------------
# the real ABFE leg: TYK2's ejm_31 decoupled from water
# ------------------------------------------------------------------------------------------------
TYK2 = __import__("pathlib").Path(__file__).resolve().parent / "data" / "alchemy" / "tyk2-v1"


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "OPEN RULING (S3 -> S2/S0, 2026-09-20): the plan treats a unique group's internal non-excluded "
    "pairs asymmetrically. At the dummy end they are in UniqueGroupInternalNonbonded, a "
    "CustomBondForce with NO cutoff; at the physical end they are in the NonbondedForce, which "
    "CUTS them. This Hamiltonian is uniform (uncut at every lambda, Amber's gti_cut = 1), so U(1) "
    "equals System B exactly and U(0) misses System A by the intra-ligand energy beyond the "
    "cutoff: -0.076484 kJ/mol here, from 74 pairs of a 32-atom ligand (-0.0685 of it LJ). Every "
    "earlier fixture's ligand fits inside 0.9 nm, which is why nothing caught it. strict=True: "
    "this flips to a failure the moment the construction is symmetric."))
def test_the_tyk2_solvated_decoupling_leg(tmp_path):
    """The campaign's own ABFE solvent leg, built by the fixture's script: ejm_31 decoupled from
    water, 32 ligand atoms as one unique region, nothing appearing, no mapped core.

    The Reference twin of what a GPU window would integrate. It checks what the measurement rests
    on: the end states ARE the plan's Systems, the ligand's intramolecular Hamiltonian does not
    move along lambda (so the decoupled end is the ligand intact in vacuum inside the box), and no
    junction term rides on the lambda path.
    """
    import subprocess
    import sys

    import md_tools
    from md_tools.alchemy.topology import Environment, build_decoupling_plan
    from md_tools.ligands import load_package
    from md_tools.ligands.mapping import LigandSelector

    root = str(__import__("pathlib").Path(md_tools.__file__).resolve().parents[1])
    subprocess.run([sys.executable, str(TYK2 / "build_tyk2_fixture.py"), "--out", str(tmp_path),
                    "--ligand", "ejm_31", "--kind", "solvated"], check=True, timeout=3600, cwd=root)
    build = tmp_path / "ejm_31" / "solvated" / "build"
    env = Environment.from_files(build / "built.xml", build / "built.pdb",
                                 LigandSelector(resname="L31"), record=build / "built.log")
    package = load_package(TYK2 / "packages" / "LOCAL-DKNAYSZNMZIMIZ" / "param_bd1388e5fe3e")
    plan = build_decoupling_plan(package, env)
    x = plan.positions_nm
    h = from_plan(plan)
    assert len(plan.a_only) == 32 and not plan.b_only
    assert h.record["plan_internal_pairs_checked"] == len(
        plan.record["nonbonded"]["unique_group_internal"]["pairs"])

    c = _context(h.system, x)
    assert h.energy(c, _state((0, 0, 0))) == pytest.approx(_energy(plan.system_a, x), abs=1e-7)
    assert h.energy(c, _state((1, 1, 1))) == pytest.approx(_energy(plan.system_b, x), abs=1e-7)

    internal = {v: h.energy_components(c, _state((v, v, v)))["softcore_internal"]
                for v in (0.0, 0.5, 1.0)}
    assert len(set(round(e, 9) for e in internal.values())) == 1, internal

    box = plan.system_a.getDefaultPeriodicBoxVectors()[0][0]._value
    policy = (plan.record.get("terms", {}) or {}).get("junction_policy")
    split = fx3.bonded_derivative_split(plan.system_a, plan.system_b, plan.a_only, x, box=box,
                                        policy=policy)
    assert split["unique_touching"] == 0.0, split      # a decoupling leg has no junction at all
    assert h.derivative_components(c, _state((0.5, 0.5, 0.5)))["lambda_bonded"] == {} \
        or sum(h.derivative_components(c, _state((0.5, 0.5, 0.5)))["lambda_bonded"].values()) == 0.0
