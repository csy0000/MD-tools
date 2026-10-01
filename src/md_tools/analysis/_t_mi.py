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

from dataclasses import dataclass, field
from itertools import combinations

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


__all__ = ["t_mi", "torsional_mi", "mi_matrix", "dependence_graph", "mi_pair"]


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



# ------------------------------------------------------------------ correlation detection
#
# The PAIRWISE estimator above answers "are these two torsions dependent". Everything from here
# answers the question a user actually arrives with -- WHICH torsions are coupled, across the
# whole set -- and that is a different question statistically, because asking it of d torsions
# means d(d-1)/2 simultaneous comparisons. Without a multiplicity correction, a handful of
# "significant" pairs is the expected yield of pure noise, so `benjamini_hochberg` controls the
# false discovery rate and the uncorrected values are kept beside the corrected ones.
#
# The verdicts are deliberately three-valued and NONE of them says "independent": a pairwise test
# that does not detect dependence at this binning and this sample size has not shown independence,
# and a vocabulary that let it say so would turn an underpowered comparison into a result.

#: verdicts, and none of them says "independent"
SUPPORTED = "supported_dependence"


NOT_DETECTED = "no_dependence_detected_at_this_resolution"


UNRESOLVED = "unresolved_sampling_or_estimator_sensitivity"


def mi_pair(t1, t2, w=None, bins: int = 24, origin: float = -np.pi) -> float:
    return mi_from_table(joint_table(t1, t2, w, bins, origin))


def support_diagnostics(P: np.ndarray, n_frames: int, ess: float) -> dict:
    """How much of the joint the sample actually covers. MI is meaningless without this."""
    occ = int(np.sum(P > 0))
    cells = int(P.size)
    return {"cells": cells, "cells_occupied": occ, "frac_occupied": occ / cells,
            "n_frames": int(n_frames), "ess_weight": float(ess),
            "ess_per_occupied_cell": float(ess / max(occ, 1)),
            "naive_bias_nats_estimate": float((np.sqrt(cells) - 1) ** 2 / (2.0 * max(ess, 1.0))),
            "naive_bias_note": "(B-1)^2/(2*ESS), the leading independent-data histogram bias; "
                               "an order-of-magnitude guide, not a correction"}


# ----------------------------------------------------------------------------- the null
def null_product_marginals(t1, t2, w=None, *, bins: int = 24, origin: float = -np.pi,
                           n_draws: int, n_independent: float,
                           seed: int = 0) -> np.ndarray:
    """MI of samples drawn from `p_j (x) p_k`, the TARGET-ENSEMBLE independence null.

    WHY NOT SHUFFLE A COLUMN. The obvious null -- permute one angular column and keep the weights
    -- does not represent independence in the TARGET ensemble when the weights are
    configuration-dependent. After the permutation, frame `t`'s weight was computed from a
    configuration whose other coordinate is no longer present, so the weighted marginals of the
    shuffled data are not the target marginals, and the null is a null for a different question.

    WHAT THIS DOES INSTEAD. It draws independently from the two WEIGHTED marginals, which are
    estimates of the target marginals, and so realises exactly "these two torsions, with their
    real target-ensemble marginals, and no dependence". `tests/hprest2/test_mi.py` validates it
    against constructions whose independence is known by design, including concentrated marginals.

    ASSUMPTIONS, stated because they bound every p-value derived from this:
      * the weighted marginals are well enough estimated to be resampled from -- if a marginal
        rests on a handful of high-weight frames, so does the null;
      * `n_independent` is the effective number of INDEPENDENT samples, which the caller must
        supply from a block analysis. The weight ESS alone overstates it for an autocorrelated
        trajectory, and the finite-sample MI bias scales like `1/N`, so using the wrong `N` moves
        the whole null. This is the single largest assumption in the module.
      * the null is unweighted by construction (draws are already from the target marginals), so
        it does not reproduce weight-induced variance. That makes it mildly optimistic; the block
        bootstrap on the real data is what carries the weight-variance information.
    """
    P = joint_table(t1, t2, w, bins, origin)
    pj = P.sum(axis=1); pk = P.sum(axis=0)
    rng = np.random.default_rng(seed)
    n = int(max(2, round(n_independent)))
    centres = np.linspace(origin, origin + TWO_PI, bins + 1)[:-1] + TWO_PI / (2 * bins)
    out = np.empty(n_draws)
    for d in range(n_draws):
        a = rng.choice(bins, size=n, p=pj)
        b = rng.choice(bins, size=n, p=pk)
        out[d] = mi_pair(centres[a], centres[b], None, bins=bins, origin=origin)
    return out


# ----------------------------------------------------------------------------- matrix + verdicts
@dataclass
class MIResult:
    names: List[str]
    bins: int
    origin: float
    mi_raw: np.ndarray                                  # (n, n) nats, symmetric, diagonal = self
    pairs: List[dict] = field(default_factory=list)     # ranked, diagonal excluded
    provenance: Dict[str, object] = field(default_factory=dict)

    def ranked(self, verdict: Optional[str] = None) -> List[dict]:
        rows = [p for p in self.pairs if verdict is None or p["verdict"] == verdict]
        return sorted(rows, key=lambda p: -p["mi_raw_nats"])


def mi_matrix(theta: np.ndarray, w=None, bins: int = 24, origin: float = -np.pi) -> np.ndarray:
    """Symmetric MI matrix. Diagonal holds the self-MI (the marginal entropy), never ranked."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"expected (n_frames, n_torsions); got shape {theta.shape}")
    n = theta.shape[1]
    M = np.zeros((n, n))
    for j in range(n):
        M[j, j] = mi_pair(theta[:, j], theta[:, j], w, bins, origin)
        for k in range(j + 1, n):
            M[j, k] = M[k, j] = mi_pair(theta[:, j], theta[:, k], w, bins, origin)
    return M


def n_unique_pairs(n_torsions: int) -> int:
    """`n(n-1)/2`. For 15 torsions that is 105; the count is derived, never hard-coded."""
    return n_torsions * (n_torsions - 1) // 2


def benjamini_hochberg(p: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """BH step-up. Returns the reject mask and the adaptive threshold actually used.

    Applied because a 15-torsion analysis tests 105 pairs at once: at alpha = 0.05 an uncorrected
    scan expects about 5 false positives, which is the same order as the number of real couplings
    this campaign has ever found.
    """
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if not np.any(passed):
        return np.zeros(m, dtype=bool), 0.0
    kmax = int(np.max(np.nonzero(passed)[0]))
    cut = float(p[order][kmax])
    return p <= cut, cut


def dependence_graph(res: MIResult) -> List[List[str]]:
    """Connected components over SUPPORTED pairs. Overlapping membership is allowed.

    A COMPONENT IS A CANDIDATE ANALYSIS GROUP, NOT PROOF OF IRREDUCIBLE MANY-BODY COUPLING, and a
    MISSING edge rules nothing out: pairwise MI is blind to dependence that appears only in three
    or more variables jointly, and a macrocycle closure constraint is exactly that kind of object.
    Nothing here forces disjoint pairs.
    """
    edges = [(p["name_i"], p["name_j"]) for p in res.pairs if p["verdict"] == SUPPORTED]
    adj: Dict[str, set] = {n: set() for n in res.names}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for n in res.names:
        if n in seen or not adj[n]:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    return comps


def normalised_fourier_correlation(t1, t2, m: int, n: int, w=None) -> dict:
    """`R_mn = |C_mn| / sqrt((1-|a_m|^2)(1-|b_n|^2))`, a genuine correlation coefficient in [0,1].

    `C_mn` is a COVARIANCE and so carries the marginals' own concentration: a pair whose marginals
    are sharply peaked has small `Var(e^{i m theta}) = 1 - |a_m|^2`, and a modest covariance
    between them then looks small on the raw scale while being nearly complete on the normalised
    one. Dividing by the geometric mean of the two variances removes that, exactly as Pearson's
    r does for real variables.

    `R_mn = 0` iff the two Fourier components are uncorrelated at that order; `R_mn = 1` means one
    determines the other. IT REMAINS AN ORDER-BY-ORDER STATEMENT: `R_mn = 0` for every small
    `(m, n)` does NOT imply independence, which is why MI stays the dependence metric and this is
    a descriptor of a chosen direction.

    Degenerate case: a marginal with `|a_m| = 1` is a delta function at that order, its variance is
    zero and `R_mn` is undefined; reported as `None` rather than divided through.
    """
    a = np.asarray(t1, dtype=np.float64).ravel()
    b = np.asarray(t2, dtype=np.float64).ravel()
    wn = (np.ones(a.size) / a.size) if w is None else \
        np.asarray(w, dtype=np.float64).ravel() / np.sum(w)
    am = complex(np.sum(wn * np.exp(1j * m * a)))
    bn = complex(np.sum(wn * np.exp(1j * n * b)))
    c = complex(np.sum(wn * np.exp(1j * (m * a + n * b))) - am * bn)
    va, vb = 1.0 - abs(am) ** 2, 1.0 - abs(bn) ** 2
    if va <= 1e-12 or vb <= 1e-12:
        # m = 0 or n = 0 lands here by construction: <exp(i*0*x)> = 1, variance 0. A correlation
        # with a constant is undefined, not zero, so the bare-torsion directions (1,0) and (0,1)
        # have no R_mn -- which is correct and worth seeing rather than silently coercing.
        return {"C_mn": c, "abs_C_mn": abs(c), "a_m": am, "b_n": bn,
                "abs_a_m": abs(am), "abs_b_n": abs(bn), "R_mn": None,
                "why": (f"a marginal variance is ~0 (|a_m|={abs(am):.6f}, |b_n|={abs(bn):.6f}); "
                        "that Fourier component is deterministic and R_mn is undefined")}
    return {"C_mn": c, "abs_C_mn": abs(c), "a_m": am, "b_n": bn,
            "abs_a_m": abs(am), "abs_b_n": abs(bn),
            "R_mn": float(abs(c) / np.sqrt(va * vb))}


def centred_fourier_moment(t1, t2, m: int, n: int, w=None) -> complex:
    """`C_mn = <e^{i(m t1 + n t2)}> - <e^{i m t1}><e^{i n t2}>`.

    CENTRED, which is the whole point: the raw moment `<e^{i(m t1 + n t2)}>` is large for
    INDEPENDENT concentrated marginals, and subtracting the product of the marginal moments
    removes exactly that. Use it only to SUGGEST an interpretable sum or difference for a pair MI
    has already flagged -- never as the dependence score, and never optimised over `m, n`.
    """
    a = np.asarray(t1, dtype=np.float64).ravel(); b = np.asarray(t2, dtype=np.float64).ravel()
    wn = (np.ones(a.size) / a.size) if w is None else \
        np.asarray(w, dtype=np.float64).ravel() / np.sum(w)
    j = np.sum(wn * np.exp(1j * (m * a + n * b)))
    return complex(j - np.sum(wn * np.exp(1j * m * a)) * np.sum(wn * np.exp(1j * n * b)))


# --------------------------------------------- the orchestrator, and the weighting it needs
#
# `analyse` is what makes `dependence_graph` and `benjamini_hochberg` REACHABLE. Without it the
# port had both of them as public names that nothing could call: `dependence_graph` takes an
# `MIResult` and no function in the module produced one. That is dead public API -- accepted,
# importable, and inert -- which is the same defect as a flag that does nothing, and it also made
# this module's own comment about controlling the false discovery rate false, since nothing
# applied the correction.
#
# The block bootstrap below comes with it because `analyse` needs it: a correlated trajectory has
# far fewer independent samples than frames, and an error bar computed as if every frame were
# independent is too small by the square root of the correlation time.

def logsumexp(a: np.ndarray) -> float:
    """Stable log(sum(exp(a))). Local so this module has no scipy dependency."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def normalized_weights(log_w) -> np.ndarray:
    """`exp(log_w - logsumexp(log_w))`, summing to 1 and never overflowing.

    Large-but-finite log weights are handled by the subtraction, so `log_w = 700` is fine where
    `exp(700)` would not be.
    """
    lw = np.asarray(log_w, dtype=np.float64).ravel()
    if lw.size == 0:
        return lw.copy()
    if not np.all(np.isfinite(lw)):
        raise ValueError(
            f"{int(np.sum(~np.isfinite(lw)))} of {lw.size} log weights are not finite. A "
            "non-finite weight is a bug upstream (an empty bias table, a NaN CV), not something "
            "to normalise away.")
    return np.exp(lw - logsumexp(lw))


def ess_weight(log_w) -> float:
    """`1 / sum_t w_t^2` on NORMALISED weights -- the weight-only effective sample size.

    THIS IS NOT THE NUMBER OF INDEPENDENT CONFIGURATIONS. It measures only how unevenly the
    weights are spread and is blind to temporal autocorrelation: a trajectory stuck in one basin
    for its whole length can have `ess_weight` equal to its frame count. Treating it as a sample
    count is a documented error of this campaign -- it was used for a kappa uncertainty and gave
    an answer a visit-block bootstrap contradicted. Pair it with `block_bootstrap` or with
    between-run spread, always.
    """
    w = normalized_weights(log_w)
    return float(1.0 / np.sum(w ** 2)) if w.size else 0.0


def block_bootstrap(n_frames: int, n_blocks: int, n_resamples: int = 500,
                    seed: int = 0) -> np.ndarray:
    """Indices for a moving-block bootstrap over COMPLETE RECORDS.

    Returns `(n_resamples, n_frames)` integer indices. The caller applies them to coordinates AND
    weights with the same array, which is the point: resampling a frame must carry its weight with
    it. Bootstrapping isolated frames as if independent is the error this replaces -- it treats an
    autocorrelated trajectory as `n_frames` independent draws and shrinks every error bar.

    A block count this small is normal for MD and is itself a limit on what can be claimed: with
    fewer than ~10 blocks the percentile interval is not trustworthy and the honest verdict is
    `unresolved`, not a tight interval.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    n_blocks = int(max(1, min(n_blocks, n_frames)))
    L = n_frames // n_blocks
    if L < 1:
        raise ValueError(f"{n_blocks} blocks do not fit in {n_frames} frames")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_frames - L + 1, size=(n_resamples, n_blocks))
    offs = np.arange(L)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_resamples, -1)
    if idx.shape[1] < n_frames:            # keep length exact for aligned application
        pad = np.repeat(idx[:, -1:], n_frames - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
    return idx[:, :n_frames] % n_frames


def enough_blocks(n_blocks: int, minimum: int = 10) -> bool:
    """Whether a percentile interval over this many blocks should be quoted at all."""
    return int(n_blocks) >= int(minimum)

def analyse(theta: np.ndarray, names: Sequence[str], *, w=None,
            bins_sweep: Sequence[int] = (18, 24, 36),
            origin_sweep: Sequence[float] = (-np.pi, -np.pi + np.pi / 24),
            n_blocks: int = 20, n_boot: int = 300, n_null: int = 300,
            alpha: float = 0.05, seed: int = 0,
            n_independent: Optional[float] = None,
            target_ensemble: str = "unspecified",
            phase: str = "unspecified") -> MIResult:
    """Full pairwise analysis with uncertainty, null, correction and explicit verdicts.

    `n_independent` should come from a block analysis or from between-run spread. Left `None` it
    falls back to the WEIGHT ESS and the provenance records that this is an upper bound, because
    the weight ESS is blind to autocorrelation.
    """
    theta = np.asarray(theta, dtype=np.float64)
    names = list(names)
    if theta.ndim != 2 or theta.shape[1] != len(names):
        raise ValueError(f"theta {theta.shape} does not match {len(names)} names")
    nF, nT = theta.shape
    wn = (np.ones(nF) / nF) if w is None else normalized_weights(np.log(
        np.maximum(np.asarray(w, dtype=np.float64).ravel(), 1e-300)))
    ess = float(1.0 / np.sum(wn ** 2))
    n_ind = float(ess if n_independent is None else n_independent)

    base_bins, base_origin = bins_sweep[0], origin_sweep[0]
    M = mi_matrix(theta, wn, base_bins, base_origin)
    boot_idx = block_bootstrap(nF, n_blocks, n_boot, seed=seed)
    blocks_ok = enough_blocks(n_blocks)

    rows = []
    for j, k in combinations(range(nT), 2):
        tj, tk = theta[:, j], theta[:, k]
        raw = float(M[j, k])

        # --- robustness across resolution AND bin origin -----------------------------------
        sweep = [mi_pair(tj, tk, wn, b, o) for b in bins_sweep for o in origin_sweep]
        sweep = np.asarray(sweep)

        # --- uncertainty: block bootstrap over COMPLETE records ----------------------------
        # coordinates and weights are indexed by the SAME array, so a resampled frame keeps its
        # own weight. Bootstrapping frames independently would treat an autocorrelated trajectory
        # as nF independent draws and shrink this interval to nothing.
        bs = np.empty(n_boot)
        for r in range(n_boot):
            ix = boot_idx[r]
            bs[r] = mi_pair(tj[ix], tk[ix], wn[ix], base_bins, base_origin)
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if blocks_ok else (np.nan, np.nan))

        # --- null, then a corrected score kept as its OWN field ----------------------------
        nul = null_product_marginals(tj, tk, wn, bins=base_bins, origin=base_origin,
                                     n_draws=n_null, n_independent=n_ind, seed=seed + j * 97 + k)
        p_emp = float((1.0 + np.sum(nul >= raw)) / (1.0 + len(nul)))
        corrected = float(raw - np.mean(nul))
        P = joint_table(tj, tk, wn, base_bins, base_origin)
        rows.append({
            "i": j, "j": k, "name_i": names[j], "name_j": names[k],
            "mi_raw_nats": raw,
            "mi_null_mean_nats": float(np.mean(nul)),
            "mi_null_p97_5_nats": float(np.percentile(nul, 97.5)),
            "mi_corrected_nats": corrected,
            "mi_boot_lo_nats": float(lo), "mi_boot_hi_nats": float(hi),
            "mi_sweep_min_nats": float(sweep.min()), "mi_sweep_max_nats": float(sweep.max()),
            "mi_sweep_spread_nats": float(sweep.max() - sweep.min()),
            "p_empirical": p_emp,
            "support": support_diagnostics(P, nF, ess),
        })

    pvals = [r["p_empirical"] for r in rows]
    reject, cut = benjamini_hochberg(pvals, alpha)
    for r, rej in zip(rows, reject):
        # A pair is SUPPORTED only if it survives the null AFTER correction, its bootstrap
        # interval is quotable, and the whole resolution/origin sweep stays above the null.
        sweep_above = r["mi_sweep_min_nats"] > r["mi_null_p97_5_nats"]
        if not blocks_ok or np.isnan(r["mi_boot_lo_nats"]):
            r["verdict"] = UNRESOLVED
            r["why"] = (f"only {n_blocks} blocks; a percentile interval over that few is not "
                        "trustworthy, so neither is any significance statement")
        elif rej and sweep_above and r["mi_boot_lo_nats"] > r["mi_null_mean_nats"]:
            r["verdict"] = SUPPORTED
            r["why"] = "survives the null after BH correction, and across the whole sweep"
        elif rej and not sweep_above:
            r["verdict"] = UNRESOLVED
            r["why"] = (f"significant at the base resolution but the sweep spans "
                        f"{r['mi_sweep_min_nats']:.4f}-{r['mi_sweep_max_nats']:.4f} nats and "
                        "crosses the null; histogram MI is not origin-invariant at finite bins")
        else:
            r["verdict"] = NOT_DETECTED
            r["why"] = ("not distinguishable from the independence null at this resolution and "
                        "sample size. THIS IS NOT PROVEN INDEPENDENCE -- a coupling below the "
                        "resolution of this estimator, or needing more samples, would look "
                        "identical")
    return MIResult(
        names=names, bins=base_bins, origin=float(base_origin), mi_raw=M, pairs=rows,
        provenance={
            "estimator": "weighted periodic histogram, nats, marginals from the joint table",
            "n_frames": int(nF), "n_torsions": int(nT),
            "n_unique_pairs": n_unique_pairs(nT), "n_pairs_analysed": len(rows),
            "bins_sweep": list(bins_sweep), "origin_sweep": [float(o) for o in origin_sweep],
            "base_bins": int(base_bins), "base_origin": float(base_origin),
            "ess_weight": ess,
            "ess_weight_note": "weights only; excludes temporal autocorrelation",
            "n_independent_used": n_ind,
            "n_independent_source": ("weight ESS (UPPER BOUND -- autocorrelation ignored)"
                                     if n_independent is None else "caller-supplied"),
            "n_blocks": int(n_blocks), "blocks_sufficient_for_interval": bool(blocks_ok),
            "n_boot": int(n_boot), "n_null": int(n_null),
            "multiple_testing": f"Benjamini-Hochberg at alpha={alpha}",
            "bh_threshold_used": cut,
            "target_ensemble": target_ensemble, "phase": phase,
            "no_universal_cutoff": "MI has no absolute significance threshold; verdicts come "
                                   "from the calibrated null, not from a fixed value",
        })

#: The name used at the call site, matching `t_hdbscan` and `t_symmetry`. `torsional_mi` is the
#: original name and stays as an alias: it is what the hpREST2 session's own callers use, and
#: breaking it would make the two trees disagree about a function they share.
t_mi = torsional_mi
