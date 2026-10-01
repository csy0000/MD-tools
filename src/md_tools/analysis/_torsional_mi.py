"""`torsional_mi` -- pairwise mutual information between torsions, normalised and not.

INTENDED HOME, as for `t_hdbscan`: `md_tools.analysis.torsional_mi`. numpy only -- no sklearn, no
free-energy estimator, no file layout. Takes angles and optional per-frame weights and knows
nothing about where they came from.

    out = torsional_mi(torsions)                      # uniform weights
    out = torsional_mi(torsions, weights=w, names=names)
    out["mi_nats"]            # (n, n) UNNORMALISED mutual information, nats
    out["mi_normalized"]      # (n, n) symmetric, I / sqrt(Hi Hj), in [0, 1]
    out["entropy_nats"]       # (n,) marginal entropies of the same binning

WHY BOTH, since the choice changes which pairs look coupled. Unnormalised MI in nats is the
information shared, and it is bounded by the SMALLER marginal entropy -- so a sharply peaked
torsion with little entropy of its own can never show a large MI however completely it is
determined by its partner. The normalised form divides that out and answers "what FRACTION of what
could be shared, is", which is the question a redundancy decision needs. Three normalisations are
returned because they answer different questions and the literature uses all three:

    mi_normalized          I / sqrt(Hi Hj)    symmetric, the usual "NMI", in [0, 1]
    mi_fraction_of_min     I / min(Hi, Hj)    the fraction of the ACHIEVABLE maximum
    mi_fraction_of_row     I / Hi             asymmetric: of torsion i's entropy, how much
                                              torsion j explains. This is the one a "drop the
                                              redundant torsion" rule must use, because dropping
                                              i is only safe if i is determined by what remains.

THE PLUG-IN ESTIMATOR IS BIASED UPWARD AND THAT IS NOT A DETAIL. A histogram MI of independent
variables is positive, not zero: with `bins` x `bins` cells and a finite effective sample the bias
is of order (bins-1)^2 / (2 * ESS) nats. At 24 bins and an ESS of 2000 that is about 0.07 nats,
which is the size of a weak real coupling. So `mi_bias_estimate` is measured per pair by shuffling
one torsion against the other -- destroying the dependence while keeping both marginals and the
binning exactly -- and `mi_nats_debiased` is reported beside the raw value. A pair whose raw MI
does not clear its own null is not evidence of coupling.

NOTHING HERE IS CAUSAL OR DYNAMICAL. MI is a statement about a joint distribution of frames. It
does not say which torsion drives which, and it does not say the two interconvert on any timescale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# THE TWO HELPERS THIS NEEDS, inlined rather than imported. hpREST2's `mi.py` also holds the
# dependence graph, the FDR correction and the Fourier diagnostics, none of which this module
# reaches; the transitive closure of these two is exactly `TWO_PI`, `wrap_to`, `joint_table` and
# `mi_from_table`, so inlining them is what keeps `md_tools.analysis` from acquiring a
# mutual-information *framework* alongside the one function that was asked for.
# ---------------------------------------------------------------------------------------------

TWO_PI = 2.0 * np.pi


def wrap_to(theta, origin: float = -np.pi) -> np.ndarray:
    """Angles into `[origin, origin + 2pi)`. The explicit periodic convention for this module."""
    t = np.asarray(theta, dtype=np.float64)
    return origin + np.mod(t - origin, TWO_PI)


def joint_table(t1, t2, w=None, bins: int = 24, origin: float = -np.pi) -> np.ndarray:
    """Normalised weighted joint histogram on the torus. Rows index `t1`, columns `t2`."""
    a = wrap_to(t1, origin); b = wrap_to(t2, origin)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    w = np.ones(a.size, dtype=np.float64) / a.size if w is None else \
        np.asarray(w, dtype=np.float64).ravel()
    if w.size != a.size:
        raise ValueError(f"{w.size} weights for {a.size} frames")
    if np.any(w < 0):
        raise ValueError("negative weights are not importance weights")
    edges = np.linspace(origin, origin + TWO_PI, bins + 1)
    ia = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    ib = np.clip(np.digitize(b, edges) - 1, 0, bins - 1)
    P = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(P, (ia, ib), w)
    tot = P.sum()
    if tot <= 0:
        raise ValueError("total weight is zero")
    return P / tot                       # GLOBAL RESCALING OF w CANCELS HERE, exactly


def mi_from_table(P: np.ndarray) -> float:
    """MI in nats from a normalised joint table, marginals taken from that same table."""
    pj = P.sum(axis=1, keepdims=True)
    pk = P.sum(axis=0, keepdims=True)
    denom = pj @ pk
    m = (P > 0) & (denom > 0)            # zero cells contribute zero; nothing is fabricated
    return float(np.sum(P[m] * np.log(P[m] / denom[m])))

# ---------------------------------------------------------------------------------------------


__all__ = ["torsional_mi"]


def _entropy_from_table(P: np.ndarray, axis: int) -> float:
    """Marginal entropy in nats, from the SAME normalised joint table the MI came from.

    Taken from the joint rather than computed independently so that `I <= min(Hi, Hj)` holds by
    construction; two separately binned estimates can violate it and make a normalised MI exceed 1.
    """
    p = P.sum(axis=1 - axis)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def torsional_mi(torsions, *, weights=None, names: Optional[Sequence[str]] = None,
                 bins: int = 24, units: str = "radians", n_null: int = 8,
                 null_seed: int = 0) -> Dict[str, Any]:
    """Pairwise MI between every torsion pair, raw and normalised, with a measured null.

    `torsions` is `(n_frames, n_torsions)`; `weights` is `(n_frames,)` and defaults to 1. Returns
    symmetric `(n, n)` matrices with the diagonal set to NaN (self-MI is the marginal entropy and
    is returned separately as `entropy_nats`, never ranked as a coupling), plus a `pairs` list
    ranked by debiased MI.

    `n_null` shuffles per pair estimate the plug-in bias. Set it to 0 to skip -- but then
    `mi_nats_debiased` is None rather than silently equal to the raw value.

    NO PASS/FAIL IS RETURNED. Each pair carries `mi_bias_estimate` and `mi_bias_sd` from its own
    null and `excess_over_null_in_sd`; where the line falls is the caller's choice. A boolean would
    collapse an excess of 34 sd and an excess of 0.5 sd into the same token.
    """
    th = np.asarray(torsions, dtype=np.float64)
    if th.ndim != 2:
        raise ValueError(f"torsions must be (n_frames, n_torsions), got shape {th.shape}")
    if str(units).lower() in ("deg", "degrees"):
        th = np.deg2rad(th)
    elif str(units).lower() not in ("rad", "radians"):
        raise ValueError(f"units must be 'radians' or 'degrees', got {units!r}")
    nF, n = th.shape
    if n < 2:
        raise ValueError(f"MI needs at least 2 torsions, got {n}")
    if weights is None:
        w = np.ones(nF, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != nF:
            raise ValueError(f"{w.size} weights for {nF} frames")
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
    wn = w / w.sum()
    ess = float(1.0 / np.sum(wn * wn))
    nm = ([f"tor{j}" for j in range(n)] if names is None else [str(x) for x in names])
    if len(nm) != n:
        raise ValueError(f"{len(nm)} names for {n} torsions")

    mi = np.full((n, n), np.nan)
    nmi = np.full((n, n), np.nan)
    fmin = np.full((n, n), np.nan)
    frow = np.full((n, n), np.nan)
    bias = np.full((n, n), np.nan)
    H = np.empty(n)
    rng = np.random.default_rng(null_seed)

    # marginal entropies from a self-joint, so they share the binning with every pair
    for j in range(n):
        P = joint_table(th[:, j], th[:, j], w=w, bins=bins)
        H[j] = _entropy_from_table(P, axis=0)

    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            P = joint_table(th[:, i], th[:, j], w=w, bins=bins)
            I = float(mi_from_table(P))
            Hi, Hj = _entropy_from_table(P, 0), _entropy_from_table(P, 1)
            b, b_sd = np.nan, None
            if n_null > 0:
                # SHUFFLE ONE AXIS ONLY: the dependence is destroyed, both marginals and the
                # binning are untouched, so what is left is the estimator's own bias at this ESS.
                vals = []
                for _ in range(int(n_null)):
                    perm = rng.permutation(nF)
                    vals.append(float(mi_from_table(
                        joint_table(th[:, i], th[perm, j], w=w, bins=bins))))
                b = float(np.mean(vals))
                b_sd = (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
            mi[i, j] = mi[j, i] = I
            bias[i, j] = bias[j, i] = b
            denom = float(np.sqrt(Hi * Hj))
            nmi[i, j] = nmi[j, i] = (I / denom) if denom > 0 else np.nan
            mn = float(min(Hi, Hj))
            fmin[i, j] = fmin[j, i] = (I / mn) if mn > 0 else np.nan
            frow[i, j] = (I / Hi) if Hi > 0 else np.nan
            frow[j, i] = (I / Hj) if Hj > 0 else np.nan
            pairs.append(dict(
                i=i, j=j, a=nm[i], b=nm[j], mi_nats=I,
                mi_nats_debiased=(None if n_null == 0 else I - b),
                mi_bias_estimate=(None if n_null == 0 else b),
                mi_bias_sd=(None if n_null == 0 else b_sd),
                # NO BOOLEAN VERDICT. A threshold on "does this clear its null" is the caller's
                # scientific choice, not the estimator's: a bool collapses 1.53-against-0.045 and
                # 0.050-against-0.045 into the same answer, and the second is precisely the case
                # where a reader needs the numbers. The null's value and spread are returned so
                # the comparison can be made explicitly.
                excess_over_null_in_sd=(
                    None if (n_null == 0 or not b_sd) else float((I - b) / b_sd)),
                mi_normalized=float(nmi[i, j]), mi_fraction_of_min=float(fmin[i, j]),
                fraction_of_a=float(frow[i, j]), fraction_of_b=float(frow[j, i])))

    key = (lambda d: (-(d["mi_nats_debiased"] if d["mi_nats_debiased"] is not None
                        else d["mi_nats"])))
    pairs.sort(key=key)
    return dict(
        names=nm, n_frames=int(nF), n_torsions=int(n), bins=int(bins), ess=ess,
        weights_uniform=bool(np.allclose(w, w[0])),
        mi_nats=mi, mi_normalized=nmi, mi_fraction_of_min=fmin, mi_fraction_of_row=frow,
        mi_bias_estimate=(None if n_null == 0 else bias),
        mi_nats_debiased=(None if n_null == 0 else mi - bias),
        entropy_nats=H, pairs=pairs, n_null=int(n_null),
        bias_note=(
            "the plug-in histogram estimator is biased UPWARD: independent torsions give a "
            f"positive MI of order (bins-1)^2/(2*ESS) ~ {((bins - 1) ** 2) / (2 * ess):.4f} nats "
            f"at bins={bins} and ESS={ess:.0f}. mi_bias_estimate is measured per pair by shuffling "
            "one axis, which keeps both marginals and the binning. A raw MI that does not clear "
            "its own null is not evidence of coupling."
            if n_null > 0 else
            "n_null=0, so NO bias was estimated. The raw plug-in MI is positively biased and "
            "mi_nats_debiased is None rather than silently equal to mi_nats."),
        caveats=[
            "the diagonal of every matrix is NaN: self-MI is the marginal entropy, returned as "
            "entropy_nats, and ranking it as a coupling is meaningless",
            "normalisations answer different questions -- mi_normalized is symmetric I/sqrt(HiHj), "
            "mi_fraction_of_row is asymmetric I/Hi and is the one a redundancy rule needs",
            "MI is a property of the joint distribution of frames: it is NOT causal and says "
            "nothing about which torsion drives which",
            "MI is NOT dynamical: it does not say the two torsions interconvert on any timescale",
            "no pass/fail is returned: compare mi_nats against mi_bias_estimate and mi_bias_sd "
            "yourself, because where that line falls is a scientific choice and not the "
            "estimator's to make",
        ])
