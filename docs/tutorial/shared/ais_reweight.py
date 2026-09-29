#!/usr/bin/env python
"""Reweight AIS path endpoints into a torsion distribution of V1.

Each path starts from a frame of the V0 ensemble and is switched to V1 in finite time, so its
endpoint is NOT a sample of V1. Weighting each endpoint by exp(-beta W) (Jarzynski / annealed
importance sampling) turns the set of endpoints into an estimate of V1's equilibrium distribution:

    w_i = exp(-beta W_i) / sum_j exp(-beta W_j)

This script reads, from an AIS run directory:

    AIS_paths.csv    total_reduced_work (= beta W) per path
    AIS_cv.csv       every torsion, per path, per observation -- the endpoint is the last step
    source.cv.csv    the V0 ensemble itself (present when ais_source.generate was used)

and writes one PNG per torsion comparing the three distributions, plus a summary of the weights
(the Kish effective sample size) and the free-energy estimate dF = -kT ln <exp(-beta W)>.

    python ais_reweight.py AIS-run1 ALA2_C_N_CA_C ALA2_N_CA_C_N

Nothing here is part of md-tools: the estimator is the analysis, deliberately outside the package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def weights_from_reduced_work(reduced_work: np.ndarray) -> np.ndarray:
    """Normalised exp(-beta W), computed stably (log-sum-exp)."""
    shifted = -(reduced_work - reduced_work.min())
    w = np.exp(shifted - np.log(np.exp(shifted).sum()))
    return w


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the AIS run directory, e.g. AIS-run1")
    parser.add_argument("torsions", nargs="+", help="CV names from the run's cv.<digest>.yaml")
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--out", type=Path, default=None, help="where the PNGs go (default: run)")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = args.run
    out = args.out or run
    resolved = yaml.safe_load((run / "resolved.config").read_text())
    temperature = float(resolved["dynamics"]["temperature_K"])
    kt = 8.31446261815324e-3 * temperature                      # kJ/mol

    paths = pd.read_csv(run / "AIS_paths.csv")
    cv = pd.read_csv(run / "AIS_cv.csv")
    missing = [name for name in args.torsions if name not in cv.columns]
    if missing:
        names = [c for c in cv.columns if c not in (
            "path_index", "source_frame_index", "protocol_step", "time_ps", "lambda",
            "observation_index", "coordinate_frame_index")]
        sys.exit(f"not in AIS_cv.csv: {missing}. Available: {', '.join(names)}")

    # The endpoint of every path: its last observed step, at lambda = 1.
    last = cv["protocol_step"].max()
    ends = cv[cv["protocol_step"] == last].set_index("path_index")
    assert (ends["lambda"] == 1.0).all(), "the last CV row of a path must be at lambda = 1"
    paths = paths.set_index("path_index").loc[ends.index]
    reduced = paths["total_reduced_work"].to_numpy(dtype=float)

    weights = weights_from_reduced_work(reduced)
    ess = 1.0 / np.sum(weights ** 2)
    # dF = -kT ln <exp(-beta W)>, again through log-sum-exp.
    delta_f_kt = -(np.log(np.mean(np.exp(-(reduced - reduced.min())))) - reduced.min())
    print(f"paths                    {len(reduced)}")
    print(f"work  mean / min / max   {np.mean(reduced) * kt:.2f} / {reduced.min() * kt:.2f} / "
          f"{reduced.max() * kt:.2f} kJ/mol")
    print(f"dF (Jarzynski, V0 -> V1) {delta_f_kt * kt:.2f} kJ/mol  ({delta_f_kt:.2f} kT)")
    print(f"Kish effective samples   {ess:.1f} of {len(reduced)}"
          + ("   <-- few paths carry the estimate; switch slower or run more paths"
             if ess < 0.1 * len(reduced) else ""))

    source = run / "source.cv.csv"
    source_cv = pd.read_csv(source) if source.is_file() else None
    edges = np.linspace(-180.0, 180.0, args.bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for name in args.torsions:
        figure, axis = plt.subplots(figsize=(6, 4))
        if source_cv is not None:
            density, _ = np.histogram(source_cv[name], bins=edges, density=True)
            axis.plot(centres, density, color="0.6", label="V0 source ensemble (hot MD)")
        density, _ = np.histogram(ends[name], bins=edges, density=True)
        axis.plot(centres, density, color="tab:blue", linestyle="--",
                  label="path endpoints, unweighted (not an ensemble)")
        density, _ = np.histogram(ends[name], bins=edges, weights=weights, density=True)
        axis.plot(centres, density, color="tab:red", linewidth=2,
                  label=f"endpoints reweighted by exp(-βW): V1 (ESS {ess:.0f})")
        axis.set_xlim(-180, 180)
        axis.set_xticks(range(-180, 181, 60))
        axis.set_xlabel(f"{name} (degrees)")
        axis.set_ylabel("probability density")
        axis.legend(fontsize=8)
        figure.tight_layout()
        target = out / f"reweighted_{name}.png"
        figure.savefig(target, dpi=150)
        plt.close(figure)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
