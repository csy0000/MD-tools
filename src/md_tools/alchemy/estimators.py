"""Free-energy estimators over ONE sample record: EXP, BAR, MBAR and TI.

All four read `md_tools.alchemy.samples.SampleSet`. EXP, BAR and MBAR consume the cross-state
reduced potentials; TI consumes the derivative components of the same samples along the same
path. None of them evaluates a potential.

CONVENTIONS

    `delta_f` is f(to) - f(from) in kT, the reduced free energy of the `to` state minus that of
    the `from` state. Along a path from endpoint A (s = 0) to endpoint B (s = 1) the reported
    value is F(B) - F(A). Converted to kJ/mol and kcal/mol with the record's single kT.

    Uncertainties are one standard error. EXP and BAR use the asymptotic (delta-method)
    variances with the sample count reduced by the statistical inefficiency, or computed on
    decorrelated samples; MBAR uses pymbar's asymptotic covariance on decorrelated samples. The
    path total from pairwise BAR sums neighbour variances, which ignores the correlation through
    the window shared by two neighbouring pairs -- the usual approximation, stated rather than
    hidden.

EXP IS COMPUTED STABLY

    -ln <exp(-w)> = -(logsumexp(-w) - ln N). A naive mean of exp(-w) overflows or underflows for
    the |w| of tens of kT a poorly overlapping pair produces, and returns inf or 0 without error.

OVERLAP IS REPORTED, NOT ASSUMED

    Each estimate carries its diagnostics: EXP the Kish effective sample size of its weights and
    the forward/reverse disagreement, MBAR the overlap matrix and its smallest neighbour element.
    Below `OVERLAP_WARNING` a result is marked `poor_overlap`. A number with poor overlap is
    returned -- it is still what the data say -- but it is never unmarked.

TI INTEGRATES SEGMENT BY SEGMENT

    dU/ds = sum_k (dU/d lambda_k)(d lambda_k/ds) with the slopes of the SEGMENT, so a window on a
    knot contributes with its left-hand slopes to one segment and its right-hand slopes to the
    next (see `md_tools.alchemy.paths`). The estimate is linear in per-sample values, so each
    window contributes one combined observable and the variance is exact for the quadrature
    weights: no double counting of a knot window's samples in two segments. The quadrature is
    trapezoidal; a cubic-spline value is computed alongside, and their difference is reported as
    the integration-error indicator (unknown for a segment with only two windows).

`dU/dlambda = U1 - U0` IS NOT USED ANYWHERE

    It is an identity for AIS's linear mixing, not for a softcore path. TI here reads the
    derivative components the Hamiltonian reported; it never differences endpoint energies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from md_tools.alchemy.samples import KJ_PER_KCAL, SampleSet

#: Minimum acceptable off-diagonal neighbour overlap (Klimovich, Shirts and Mobley 2015,
#: J Comput Aided Mol Des 29:397 -- "at least 0.03").
OVERLAP_WARNING = 0.03
#: EXP is unreliable when its weights are dominated by a handful of samples.
EXP_ESS_WARNING = 50.0
#: pymbar's `mintime` for statistical inefficiency.
_MINTIME = 3


class EstimatorError(ValueError):
    """Data that no estimator here can analyse honestly."""


class MissingAnalysisDependency(ImportError):
    """The optional alchemy analysis dependency is not installed."""


def _pymbar():
    # pymbar 4 imports JAX when it can, and JAX preallocates most of the memory of EVERY visible
    # GPU -- on a shared machine an analysis step then holds cards other jobs were placed on. MBAR
    # on a few thousand samples needs no accelerator, so pymbar's OWN switch is set: pymbar then
    # never imports JAX. JAX_PLATFORMS is deliberately not touched -- that would be a process-wide
    # device decision for every other JAX user. Which solver actually ran is recorded with the
    # estimate (`pymbar_backend`), because an earlier import may already have chosen JAX.
    import os
    os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
    try:
        import pymbar  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingAnalysisDependency(
            "MBAR needs pymbar (>=4,<5), part of the optional `alchemy` extra; EXP, BAR and TI "
            "do not") from exc
    return pymbar


def _pymbar_backend() -> str:
    try:
        from pymbar import mbar_solvers
    except ImportError:  # pragma: no cover
        return "unknown"
    return "numpy/scipy" if getattr(mbar_solvers, "force_no_jax", False) else "jax"


@dataclass
class Estimate:
    estimator: str
    from_state: str
    to_state: str
    delta_f_kt: float
    sigma_kt: float
    kt_kj_mol: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_g_kj_mol(self) -> float:
        return self.delta_f_kt * self.kt_kj_mol

    @property
    def sigma_kj_mol(self) -> float:
        return self.sigma_kt * self.kt_kj_mol

    @property
    def delta_g_kcal_mol(self) -> float:
        return self.delta_g_kj_mol / KJ_PER_KCAL

    @property
    def sigma_kcal_mol(self) -> float:
        return self.sigma_kj_mol / KJ_PER_KCAL

    def to_record(self) -> dict[str, Any]:
        return {"estimator": self.estimator, "from_state": self.from_state,
                "to_state": self.to_state, "delta_f_kT": self.delta_f_kt,
                "sigma_kT": self.sigma_kt, "kT_kJ_mol": self.kt_kj_mol,
                "delta_g_kJ_mol": self.delta_g_kj_mol, "sigma_kJ_mol": self.sigma_kj_mol,
                "delta_g_kcal_mol": self.delta_g_kcal_mol,
                "sigma_kcal_mol": self.sigma_kcal_mol, "diagnostics": _plain(self.diagnostics)}


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


# ---------------------------------------------------------------------- correlation
def statistical_inefficiency(series: Sequence[float]) -> float:
    """g = 1 + 2 sum_t (1 - t/N) C(t), truncated at the first non-positive C(t) after `mintime`.

    The same definition as pymbar's `timeseries.statistical_inefficiency` (non-fast mode), written
    here so EXP, BAR and TI do not need pymbar. A constant series has g = 1.
    """
    a = np.asarray(series, dtype=float)
    n = a.size
    if n < 2:
        return 1.0
    d = a - a.mean()
    sigma2 = float(np.mean(d * d))
    if sigma2 == 0.0:
        return 1.0
    g = 1.0
    t = 1
    while t < n - 1:
        c = float(np.sum(d[: n - t] * d[t:])) / (float(n - t) * sigma2)
        if c <= 0.0 and t > _MINTIME:
            break
        g += 2.0 * c * (1.0 - float(t) / float(n))
        t += 1
    return max(g, 1.0)


def subsample_indices(n: int, g: float) -> np.ndarray:
    """pymbar's rule: indices round(k g) for k = 0, 1, ... while < n."""
    if g <= 1.0:
        return np.arange(n)
    out, k = [], 0
    while True:
        i = int(round(k * g))
        if i >= n:
            break
        if not out or i != out[-1]:
            out.append(i)
        k += 1
    return np.asarray(out, dtype=int)


def decorrelate(samples: SampleSet) -> tuple[SampleSet, dict[str, float]]:
    """Subsample each origin state by the statistical inefficiency of its own reduced potential.

    Returns the decorrelated record and g per state. The observable is the potential of the
    state the sample was drawn from, the one every state has; it is recorded with the result.
    """
    u = samples.reduced_potential()
    keep, gs = [], {}
    for k, sid in enumerate(samples.state_ids):
        idx = samples.from_state(sid)
        if idx.size == 0:
            continue
        g = statistical_inefficiency(u[idx, k])
        gs[sid] = g
        keep.append(idx[subsample_indices(idx.size, g)])
    return samples.subset(np.concatenate(keep)), gs


# ---------------------------------------------------------------------- EXP
def _logmeanexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


def exp_estimate(samples: SampleSet, from_state: str, to_state: str, *,
                 statistical_inefficiency_g: float | None = None) -> Estimate:
    """Zwanzig: f(to) - f(from) = -ln < exp(-(u_to - u_from)) >_from."""
    i, j = samples.state_index(from_state), samples.state_index(to_state)
    idx = samples.from_state(from_state)
    if idx.size < 2:
        raise EstimatorError(f"EXP from {from_state} needs at least 2 samples; it has {idx.size}")
    u = samples.reduced_potential()
    w = u[idx, j] - u[idx, i]
    df = -_logmeanexp(-w)
    g = statistical_inefficiency(w) if statistical_inefficiency_g is None \
        else statistical_inefficiency_g
    n_eff = idx.size / g
    x = np.exp(-(w - w.min()))
    sigma = math.sqrt(float(np.var(x)) / (n_eff * float(np.mean(x)) ** 2))
    ess = float(x.sum() ** 2 / np.sum(x * x))
    return Estimate("EXP", from_state, to_state, df, sigma, samples.kt,
                    {"samples": int(idx.size), "statistical_inefficiency": g,
                     "kish_effective_samples": ess, "poor_overlap": ess < EXP_ESS_WARNING})


# ---------------------------------------------------------------------- BAR
def _fermi(x: np.ndarray) -> np.ndarray:
    # 1 / (1 + exp(x)) without overflow
    return np.where(x > 0, np.exp(-np.clip(x, 0, None)) / (1.0 + np.exp(-np.clip(x, 0, None))),
                    1.0 / (1.0 + np.exp(np.clip(x, None, 0))))


def bar_from_work(w_f: np.ndarray, w_r: np.ndarray, *, n_eff_f: float | None = None,
                  n_eff_r: float | None = None) -> tuple[float, float]:
    """Bennett acceptance ratio from reduced forward work (samples of state 0, u1 - u0) and
    reverse work (samples of state 1, u0 - u1). Returns (f1 - f0, sigma) in kT.

    Solves  sum_F f(M + w_F - df) = sum_R f(-M + w_R + df),  M = ln(n_F / n_R),  f(x) = 1/(1+e^x)
    by bracketed bisection on a function that is monotone in df, so it cannot converge to a wrong
    root. Uncertainty: var(f_F)/(n_F <f_F>^2) + var(f_R)/(n_R <f_R>^2) (pymbar's 'BAR' method).
    """
    w_f, w_r = np.asarray(w_f, float), np.asarray(w_r, float)
    nf, nr = w_f.size, w_r.size
    if nf < 2 or nr < 2:
        raise EstimatorError(f"BAR needs samples on both sides; have {nf} forward, {nr} reverse")
    m = math.log(nf / nr)

    def imbalance(df: float) -> float:
        return float(np.sum(_fermi(m + w_f - df)) - np.sum(_fermi(-m + w_r + df)))

    lo = min(float(np.min(w_f)), float(-np.max(w_r))) - 50.0
    hi = max(float(np.max(w_f)), float(-np.min(w_r))) + 50.0
    f_lo, f_hi = imbalance(lo), imbalance(hi)
    if not (f_lo < 0 < f_hi):
        raise EstimatorError(f"BAR could not bracket its root ({f_lo}, {f_hi})")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if imbalance(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    df = 0.5 * (lo + hi)
    ff = _fermi(m + w_f - df)
    fr = _fermi(-m + w_r + df)
    nf_e = nf if n_eff_f is None else n_eff_f
    nr_e = nr if n_eff_r is None else n_eff_r
    var = float(np.var(ff)) / (nf_e * float(np.mean(ff)) ** 2) \
        + float(np.var(fr)) / (nr_e * float(np.mean(fr)) ** 2)
    return df, math.sqrt(var)


def bar_estimate(samples: SampleSet, state_0: str, state_1: str, *,
                 correlated: bool = True) -> Estimate:
    i, j = samples.state_index(state_0), samples.state_index(state_1)
    u = samples.reduced_potential()
    a, b = samples.from_state(state_0), samples.from_state(state_1)
    w_f = u[a, j] - u[a, i]
    w_r = u[b, i] - u[b, j]
    gf = statistical_inefficiency(w_f) if correlated else 1.0
    gr = statistical_inefficiency(w_r) if correlated else 1.0
    df, sigma = bar_from_work(w_f, w_r, n_eff_f=a.size / gf, n_eff_r=b.size / gr)
    return Estimate("BAR", state_0, state_1, df, sigma, samples.kt,
                    {"forward_samples": int(a.size), "reverse_samples": int(b.size),
                     "statistical_inefficiency": {"forward": gf, "reverse": gr}})


def bar_path(samples: SampleSet, *, correlated: bool = True) -> Estimate:
    """Sum of neighbour BAR estimates along the path, first state to last."""
    ids = _ordered_ids(samples)
    parts = [bar_estimate(samples, a, b, correlated=correlated) for a, b in zip(ids, ids[1:])]
    return Estimate("BAR", ids[0], ids[-1], sum(p.delta_f_kt for p in parts),
                    math.sqrt(sum(p.sigma_kt ** 2 for p in parts)), samples.kt,
                    {"pairs": [p.to_record() for p in parts],
                     "variance_rule": "sum of neighbour variances; correlation through shared "
                                      "windows ignored"})


# ---------------------------------------------------------------------- MBAR
def mbar_estimate(samples: SampleSet) -> Estimate:
    """MBAR over every state (pymbar). Pass DECORRELATED samples: its covariance assumes them."""
    pymbar = _pymbar()
    ids = _ordered_ids(samples)
    order = np.concatenate([samples.from_state(s) for s in ids])
    u = samples.reduced_potential()[order]
    cols = [samples.state_index(s) for s in ids]
    u_kn = u[:, cols].T
    n_k = np.array([samples.from_state(s).size for s in ids])
    mbar = pymbar.MBAR(u_kn, n_k, solver_protocol="robust")
    res = mbar.compute_free_energy_differences(compute_uncertainty=True)
    overlap = np.asarray(mbar.compute_overlap()["matrix"])
    neighbour = [float(overlap[k, k + 1]) for k in range(len(ids) - 1)]
    return Estimate("MBAR", ids[0], ids[-1], float(res["Delta_f"][0, -1]),
                    float(res["dDelta_f"][0, -1]), samples.kt,
                    {"states": ids, "samples_per_state": n_k.tolist(),
                     "delta_f_matrix_kT": np.asarray(res["Delta_f"]),
                     "d_delta_f_matrix_kT": np.asarray(res["dDelta_f"]),
                     "overlap_matrix": overlap, "neighbour_overlap": neighbour,
                     "min_neighbour_overlap": min(neighbour),
                     "poor_overlap": min(neighbour) < OVERLAP_WARNING,
                     "pymbar_version": getattr(pymbar, "__version__", "unknown"),
                     "pymbar_backend": _pymbar_backend()})


# ---------------------------------------------------------------------- TI
def ti_weights(samples: SampleSet) -> dict[str, dict[str, float]]:
    """Per window, the coefficient c_wk of dU/d lambda_k in the trapezoidal TI sum.

    TI = sum_w mean_n( sum_k c_wk dU/d lambda_k (x_n) ), with c_wk = sum over the segments the
    window closes of (trapezoid weight in s) * (d lambda_k/ds on that segment).
    """
    path = samples.path
    s_of = {st.state_id: st.s for st in samples.states}
    path.require_knots_sampled(s_of.values())
    coeff = {sid: {c: 0.0 for c in path.components} for sid in s_of}
    for seg in path.segments:
        inside = sorted((s, sid) for sid, s in s_of.items()
                        if seg.s_start - 1e-9 <= s <= seg.s_end + 1e-9)
        for n, (s, sid) in enumerate(inside):
            left = s - inside[n - 1][0] if n > 0 else 0.0
            right = inside[n + 1][0] - s if n + 1 < len(inside) else 0.0
            weight = 0.5 * (left + right)
            for c in path.components:
                coeff[sid][c] += weight * seg.slopes[c]
    return coeff


def _segment_spline_value(samples: SampleSet, means: dict[str, dict[str, float]]):
    """Cubic-spline TI per segment, from per-window means of each dU/d lambda_k. None where a
    segment has fewer than three windows."""
    from scipy.interpolate import CubicSpline

    s_of = {st.state_id: st.s for st in samples.states}
    total, unknown = 0.0, []
    trap_total = 0.0
    for seg in samples.path.segments:
        inside = sorted((s, sid) for sid, s in s_of.items()
                        if seg.s_start - 1e-9 <= s <= seg.s_end + 1e-9)
        x = np.array([s for s, _ in inside])
        y = np.array([sum(means[sid][c] * seg.slopes[c] for c in seg.slopes)
                      for _, sid in inside])
        trap = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
        trap_total += trap
        if len(x) >= 3:
            total += float(CubicSpline(x, y, bc_type="not-a-knot").integrate(x[0], x[-1]))
        else:
            total += trap
            unknown.append(seg.index)
    return total, trap_total, unknown


def ti_estimate(samples: SampleSet) -> Estimate:
    """Thermodynamic integration of the complete dU/ds along the path, in kJ/mol then kT."""
    comps = samples.path.components
    missing = [c for c in comps if c not in samples.derivatives]
    if missing:
        raise EstimatorError(
            f"TI needs dU/d lambda for every component the path moves; {missing} absent. A "
            f"missing component is a missing term in dU/ds, not a zero one")
    coeff = ti_weights(samples)
    value, var = 0.0, 0.0
    per_window, means = {}, {}
    for sid in _ordered_ids(samples):
        idx = samples.from_state(sid)
        if idx.size < 2:
            raise EstimatorError(f"TI window {sid} has {idx.size} samples")
        y = sum(coeff[sid][c] * samples.derivatives[c][idx] for c in comps)
        g = statistical_inefficiency(y)
        mean = float(np.mean(y))
        sem2 = float(np.var(y, ddof=1)) * g / idx.size
        value += mean
        var += sem2
        means[sid] = {c: float(np.mean(samples.derivatives[c][idx])) for c in comps}
        per_window[sid] = {"samples": int(idx.size), "statistical_inefficiency": g,
                           "contribution_kJ_mol": mean, "sem_kJ_mol": math.sqrt(sem2),
                           "mean_dU_dlambda_kJ_mol": means[sid]}
    spline, trap_check, unknown = _segment_spline_value(samples, means)
    assert abs(trap_check - value) <= 1e-6 * max(1.0, abs(value)), (trap_check, value)
    kt = samples.kt
    return Estimate("TI", _ordered_ids(samples)[0], _ordered_ids(samples)[-1], value / kt,
                    math.sqrt(var) / kt, kt,
                    {"quadrature": "trapezoidal, per path segment",
                     "cubic_spline_kJ_mol": spline,
                     "quadrature_discrepancy_kJ_mol": abs(spline - value),
                     "segments_without_quadrature_check": unknown,
                     "windows": per_window, "path_digest": samples.path.digest()})


# ---------------------------------------------------------------------- everything
def _ordered_ids(samples: SampleSet) -> list[str]:
    return [st.state_id for st in sorted(samples.states, key=lambda st: st.s)]


def analyze(samples: SampleSet, *, estimators: Sequence[str] = ("EXP", "BAR", "MBAR", "TI"),
            ) -> dict[str, Any]:
    """Every requested estimator over one record, end to end (s = 0 -> s = 1).

    TI and the correlated BAR/EXP variances use every sample with their statistical inefficiency;
    MBAR runs on the record decorrelated by `decorrelate`. The per-state g is reported.
    """
    ids = _ordered_ids(samples)
    missing = [s for s in ids if samples.from_state(s).size == 0]
    if missing:
        raise EstimatorError(f"states {missing} have no samples; an estimator over the whole "
                             f"path needs every window")
    out: dict[str, Any] = {"states": ids, "samples_per_state": samples.counts(),
                           "kT_kJ_mol": samples.kt, "ensemble": samples.ensemble,
                           "path": samples.path.to_record(), "estimates": {}}
    if "EXP" in estimators:
        fwd = [exp_estimate(samples, a, b) for a, b in zip(ids, ids[1:])]
        rev = [exp_estimate(samples, b, a) for a, b in zip(ids, ids[1:])]
        f = sum(e.delta_f_kt for e in fwd)
        r = -sum(e.delta_f_kt for e in rev)
        for name, parts, val in (("EXP_forward", fwd, f), ("EXP_reverse", rev, r)):
            out["estimates"][name] = Estimate(
                "EXP", ids[0], ids[-1], val, math.sqrt(sum(e.sigma_kt ** 2 for e in parts)),
                samples.kt, {"pairs": [e.to_record() for e in parts],
                             "poor_overlap": any(e.diagnostics["poor_overlap"] for e in parts)})
        out["exp_forward_reverse_gap_kT"] = abs(f - r)
    if "BAR" in estimators:
        out["estimates"]["BAR"] = bar_path(samples)
    if "MBAR" in estimators:
        dec, g = decorrelate(samples)
        est = mbar_estimate(dec)
        est.diagnostics["statistical_inefficiency"] = g
        est.diagnostics["decorrelation_observable"] = "reduced potential at the origin state"
        out["estimates"]["MBAR"] = est
    if "TI" in estimators:
        out["estimates"]["TI"] = ti_estimate(samples)
    out["estimates"] = {k: v.to_record() for k, v in out["estimates"].items()}
    return _plain(out)


# ---------------------------------------------------------------------- the agreement gate
#: The 0.7.0 free-energy agreement gate (AIMS.md, "Acceptance criteria"), fixed before sampling.
GATE_ABSOLUTE_KCAL_MOL = 0.5
GATE_SIGMAS = 3.0
#: Above this combined standard error the gate cannot tell a 0.5 kcal/mol error from noise, and a
#: pass would be luck: the verdict is INCONCLUSIVE, which is not a pass.
GATE_MAX_COMBINED_SIGMA_KCAL_MOL = 0.25


def agreement_gate(value_kcal: float, sigma_kcal: float, reference_kcal: float, *,
                   reference_sigma_kcal: float = 0.0, integration_sigma_kcal: float = 0.0,
                   numerical_floor_kcal: float = 0.0) -> dict[str, Any]:
    """PASS only if |value - reference| is below BOTH 0.5 kcal/mol AND three combined standard
    errors (statistical, reference and integration uncertainty in quadrature).

    `numerical_floor_kcal` is for effectively exact fixtures, where three combined standard errors
    of ~0 would fail on round-off; it must be justified where it is set (the acceptance matrix)
    and it never raises the 0.5 kcal/mol ceiling.
    """
    combined = math.sqrt(sigma_kcal ** 2 + reference_sigma_kcal ** 2
                         + integration_sigma_kcal ** 2)
    error = abs(value_kcal - reference_kcal)
    tolerance = min(GATE_ABSOLUTE_KCAL_MOL, max(GATE_SIGMAS * combined, numerical_floor_kcal))
    if not all(math.isfinite(v) for v in (value_kcal, sigma_kcal, reference_kcal)):
        verdict = "INCONCLUSIVE"
    elif combined > GATE_MAX_COMBINED_SIGMA_KCAL_MOL:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if error < tolerance else "FAIL"
    return {"verdict": verdict, "error_kcal_mol": error, "combined_sigma_kcal_mol": combined,
            "tolerance_kcal_mol": tolerance, "absolute_ceiling_kcal_mol": GATE_ABSOLUTE_KCAL_MOL,
            "sigmas": GATE_SIGMAS, "max_combined_sigma_kcal_mol": GATE_MAX_COMBINED_SIGMA_KCAL_MOL,
            "numerical_floor_kcal_mol": numerical_floor_kcal}
