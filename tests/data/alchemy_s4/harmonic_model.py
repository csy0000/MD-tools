"""S4's analytic estimator fixture: a harmonic model with an exact free energy.

NOT the shared-contracts section 7 fixture (frozen energies of the miniature endpoint pair). That
one does not exist yet; this one lets estimator and uncertainty work proceed without a real
Hamiltonian, and it keeps a closed-form answer, which a molecular fixture never has.

MODEL (D independent coordinates, x in nm)

    U(x; le, ls) = 1/2 K(ls) sum_d (x_d - mu(le))^2 + C le^2
    K(ls) = K0 + (K1 - K0) ls            mu(le) = A le

    le = lambda_electrostatics, ls = lambda_sterics -- names only; nothing electrostatic here.

    F(le, ls) - F(0, 0) = (D kT / 2) ln(K(ls) / K0) + C le^2          (exact)

    dU/d le = -K(ls) A sum_d (x_d - mu) + 2 C le
    dU/d ls = 1/2 (K1 - K0) sum_d (x_d - mu)^2

The C le^2 term makes U NONLINEAR in lambda, so dU/dlambda is not U1 - U0 and an estimator that
used that identity would get the wrong answer here.

PATH: staged -- electrostatics over s in [0, 0.5], then sterics over [0.5, 1] -- so s = 0.5 is a
knot and dU/ds jumps there. Windows at s = 0, 0.25, 0.5, 0.75, 1.

SAMPLING is exact: each window is a stationary AR(1) chain with the Boltzmann marginal
N(mu, kT/K) per coordinate, correlation RHO, so statistical inefficiency is (1+RHO)/(1-RHO) for
linear observables. Regenerate with `python tests/data/alchemy_s4/harmonic_model.py`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

FIXTURE_NAME = "s4-harmonic-staged"
FIXTURE_VERSION = 1
FIXTURE_FILE = Path(__file__).with_name(f"harmonic_staged_v{FIXTURE_VERSION}.json")

D = 3
K0, K1 = 100.0, 400.0        # kJ/mol/nm^2
A = 0.1                      # nm
C = 5.0                      # kJ/mol
TEMPERATURE_K = 300.0
S_WINDOWS = (0.0, 0.25, 0.5, 0.75, 1.0)
SAMPLES_PER_WINDOW = 1000
RHO = 0.8
SEED = 20260919


def path():
    from md_tools.alchemy.paths import staged_path
    return staged_path([("lambda_electrostatics", 0.5), ("lambda_sterics", 1.0)],
                       endpoint_a="harmonic K0 at mu=0", endpoint_b="harmonic K1 at mu=A",
                       description="S4 analytic harmonic model; electrostatics then sterics")


def stiffness(ls):
    return K0 + (K1 - K0) * ls


def potential(x, le, ls):
    return 0.5 * stiffness(ls) * np.sum((x - A * le) ** 2, axis=-1) + C * le * le


def d_electrostatics(x, le, ls):
    return -stiffness(ls) * A * np.sum(x - A * le, axis=-1) + 2.0 * C * le


def d_sterics(x, le, ls):
    return 0.5 * (K1 - K0) * np.sum((x - A * le) ** 2, axis=-1)


def exact_delta_f_kj_mol(kt, le=1.0, ls=1.0):
    return 0.5 * D * kt * math.log(stiffness(ls) / K0) + C * le * le


def generate(samples_per_window=SAMPLES_PER_WINDOW, rho=RHO, seed=SEED, pressure_bar=None):
    from md_tools.alchemy.samples import SampleSet, kt_kj_mol, window_states

    p = path()
    states = window_states(p, S_WINDOWS, temperature_k=TEMPERATURE_K, pressure_bar=pressure_bar)
    kt = kt_kj_mol(TEMPERATURE_K)
    rng = np.random.default_rng(seed)
    ids, origin, steps, pot, dle, dls, vol = [], [], [], [], [], [], []
    for st in states:
        le, ls = st.components["lambda_electrostatics"], st.components["lambda_sterics"]
        sigma = math.sqrt(kt / stiffness(ls))
        x = np.empty((samples_per_window, D))
        x[0] = rng.normal(0.0, sigma, D)
        noise = rng.normal(0.0, sigma * math.sqrt(1 - rho * rho), (samples_per_window, D))
        for t in range(1, samples_per_window):
            x[t] = rho * x[t - 1] + noise[t]
        x += A * le
        for t in range(samples_per_window):
            ids.append(f"{st.state_id}:{t:05d}")
            origin.append(st.state_id)
            steps.append(t * 500)
        pot.append(np.stack([potential(x, o.components["lambda_electrostatics"],
                                       o.components["lambda_sterics"]) for o in states], axis=1))
        dle.append(d_electrostatics(x, le, ls))
        dls.append(d_sterics(x, le, ls))
        if pressure_bar is not None:
            vol.append(rng.uniform(26.0, 28.0, samples_per_window))
    return SampleSet(
        states, p, ids, origin, steps, np.concatenate(pot),
        np.concatenate(vol) if vol else None,
        {"lambda_electrostatics": np.concatenate(dle), "lambda_sterics": np.concatenate(dls)},
        provenance={"fixture": FIXTURE_NAME, "fixture_version": FIXTURE_VERSION,
                    "generator": "tests/data/alchemy_s4/harmonic_model.py", "seed": seed,
                    "rho": rho, "samples_per_window": samples_per_window,
                    "exact_delta_f_kJ_mol": exact_delta_f_kj_mol(kt),
                    "not_the_shared_fixture": "shared-contracts section 7 fixture is separate"})


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    digest = generate().write_json(FIXTURE_FILE)
    print(FIXTURE_FILE, digest)
