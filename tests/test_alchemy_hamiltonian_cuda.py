"""The softcore Hamiltonian's energies, forces and derivatives on CUDA, at fixed coordinates.

`test_alchemy_hamiltonian_fixture.py` validates the Hamiltonian on the Reference platform against
an independent reference. This file asks the question only a device can answer: does the SAME
System give the same numbers on CUDA, per force group, in mixed and in double precision? It
propagates nothing -- that is `test_alchemy_hamiltonian_cuda_dynamics.py`, which names no other
platform -- so comparing against Reference here is what the platform audit allows.

TOLERANCES are set before the comparison, from a calibration inside each test: the ordinary
end-state System A is evaluated on CUDA and Reference at the same coordinates, and its error e_A
sets the scale. A force group may differ by at most max(10 e_A, floor) where the floor is the
documented precision of the mode -- 1e-6 relative for mixed (single-precision forces, energies
accumulated in 64-bit), 1e-10 relative for double -- times the group's magnitude. The same rule
is applied to derivatives, which are energy differences.

DESELECTED on a machine without CUDA (the `gpu` marker), never skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import FORCE_GROUPS, build_hamiltonian  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

NAMES = ("lambda_electrostatics", "lambda_sterics", "lambda_bonded")
STATES = [(v, v, v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)] + [(1.0, 0.5, 0.3), (0.4, 1.0, 0.0)]
FLOOR = {"mixed": 1e-6, "double": 1e-10}


def _context(system, x, platform, precision=None):
    props = {"Precision": precision} if precision else {}
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName(platform), props)
    c.setPositions(x)
    return c


def _energy(c, groups=None):
    kw = {"groups": groups} if groups is not None else {}
    return c.getState(getEnergy=True, **kw).getPotentialEnergy()._value


@pytest.mark.parametrize("precision", ["mixed", "double"])
@pytest.mark.parametrize("tail", [False, True], ids=["one-atom", "tail"])
def test_cuda_matches_reference_per_force_group(precision, tail):
    sa, sb, a, b, x = fx.build(True, dispersion=True, tail=tail)
    calib = abs(_energy(_context(sa, x, "CUDA", precision)) - _energy(_context(sa, x, "Reference")))
    h = build_hamiltonian(sa, sb, a, b)
    cuda, ref = _context(h.system, x, "CUDA", precision), _context(h.system, x, "Reference")
    platform_name = cuda.getPlatform().getName()
    assert platform_name == "CUDA", platform_name
    report = []
    for t in STATES:
        state = dict(zip(NAMES, t))
        h.set_state(cuda, state)
        h.set_state(ref, state)
        for name, g in FORCE_GROUPS.items():
            e_c, e_r = _energy(cuda, {g}), _energy(ref, {g})
            tol = max(10 * calib, FLOOR[precision] * max(1.0, abs(e_r)))
            report.append((t, name, e_r, e_c - e_r, tol))
            assert abs(e_c - e_r) <= tol, (precision, t, name, e_c, e_r, tol)
        f_c = cuda.getState(getForces=True).getForces(asNumpy=True)._value
        f_r = ref.getState(getForces=True).getForces(asNumpy=True)._value
        rms = np.sqrt(np.mean(f_r ** 2))
        assert np.max(np.abs(f_c - f_r)) <= {"mixed": 1e-3, "double": 1e-7}[precision] * rms + 1e-6, t
        d_c, d_r = h.derivatives(cuda, state), h.derivatives(ref, state)
        for p in NAMES:
            tol = max(20 * calib, FLOOR[precision] * max(1.0, abs(d_r[p]), abs(_energy(ref))))
            assert abs(d_c[p] - d_r[p]) <= tol, (precision, t, p, d_c[p], d_r[p], tol)
    print(f"\n{precision} {'tail' if tail else 'one-atom'}: end-state calibration {calib:.2e} kJ/mol; "
          f"worst group |dE| = {max(abs(r[3]) for r in report):.2e}")
