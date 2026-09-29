#!/usr/bin/env python
"""Does the ladder's state 0 sample what a plain run samples?

REST2 state 0 integrates the UNSCALED Hamiltonian at the same temperature as an ordinary cMD run
of the same system. At infinite sampling the two must agree on every equilibrium property, and the
ladder's only claim over cMD is that it gets there sooner. This script puts the two histograms of
one slow coordinate on top of each other, counts how often each run crosses the barrier, and
measures the difference with a two-sample Kolmogorov-Smirnov statistic.

READ THE CROSSING COUNT BEFORE THE KS STATISTIC. A disagreement has two possible causes and they
point in opposite directions:

* **Neither run has converged, and the one that crossed the barrier fewer times is the wrong one.**
  This is the usual case, and it is the whole reason the ladder exists. A plain run trapped in one
  basin produces a narrow, smooth, entirely convincing histogram of a distribution it has not
  sampled.
* **The ladder is broken.** Exchanges that rescaled velocities, walker-indexed trajectories
  reported as state-indexed, a `tau` claim that did not match the saved state -- each shows up as
  state 0 disagreeing with a WELL-CONVERGED unbiased run, and nowhere in the completion records,
  which would all still say `completed`.

The crossing counts tell you which you are looking at. If state 0 crossed the barrier many times
and cMD barely did, a difference is cMD's fault. If both crossed freely and they still disagree,
suspect the ladder.

    python compare_cmd_rest2.py --system ALA --cmd <cMD run dir> --rest2 <REST2 run dir> \
        --cmd-build <build dir> --rest2-build <build dir> --out <png>

Reads solute trajectories only (`solute_prod1.nc`, `solute_state0_prod1.nc`), so it needs the
solute topology, which it slices out of `built.pdb` by dropping solvent and ions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
import mdtraj as md                                                          # noqa: E402
import numpy as np                                                          # noqa: E402
from scipy import stats                                                      # noqa: E402

#: Residue names that are NOT solute. `solute_*.nc` holds everything else, in topology order.
SOLVENT = {"HOH", "WAT", "NA", "CL", "K", "MG", "SOD", "CLA"}

#: One slow coordinate per system: the thing REST2 is supposed to help with.
#:
#: Each entry is (label, kind, selector). `torsion` takes four atom selectors and returns degrees;
#: `rmsd` takes a reference and returns nm. Atom selectors are mdtraj DSL strings resolved against
#: the SOLUTE topology, so indices are solute-relative and stable under a rebuild.
OBSERVABLES = {
    "ALA": ("phi (C-N-CA-C)", "torsion",
            ["resname ACE and name C", "resid 1 and name N",
             "resid 1 and name CA", "resid 1 and name C"]),
    "paracetamol": ("ring rotation about N-C", "torsion",
                    ["name C2", "name N1", "name C3", "name C4"]),
    "chignolin": ("backbone RMSD to the NMR model", "rmsd", None),
}


def solute_trajectory(built_pdb: Path, trajectory: Path) -> md.Trajectory:
    """`solute_*.nc` read against a solute-only topology sliced out of built.pdb."""
    reference = md.load(str(built_pdb))
    keep = [a.index for a in reference.topology.atoms if a.residue.name not in SOLVENT]
    solute = reference.atom_slice(keep)
    frames = md.load(str(trajectory), top=solute.topology)
    if frames.n_atoms != solute.n_atoms:
        raise SystemExit(
            f"{trajectory.name} has {frames.n_atoms} atoms and the solute of {built_pdb.name} has "
            f"{solute.n_atoms}. These are not the same system.")
    return frames, solute


def observable(frames: md.Trajectory, solute: md.Trajectory, system: str):
    label, kind, selector = OBSERVABLES[system]
    if kind == "torsion":
        indices = []
        for expression in selector:
            hit = frames.topology.select(expression)
            if len(hit) != 1:
                raise SystemExit(f"{expression!r} selects {len(hit)} atoms; it must select one")
            indices.append(int(hit[0]))
        values = md.compute_dihedrals(frames, [indices])[:, 0]
        return label, np.degrees(values), "degrees"
    if kind == "rmsd":
        backbone = frames.topology.select("backbone")
        values = md.rmsd(frames, solute, frame=0, atom_indices=backbone)
        return label, values, "nm"
    raise SystemExit(f"unknown observable kind {kind!r}")


def occupancy(values: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Boolean: is each frame inside the basin `window`?

    A WINDOW, not a threshold, because a torsion is periodic. `values > 0` also flips when the
    series wraps from -179 to +179, which never passed through zero -- that counts a wrap as a
    barrier crossing and inflates the number for whichever run wanders nearest the branch cut.
    Asking whether the coordinate is inside a basin has no branch cut to trip over.
    """
    low, high = window
    return (values > low) & (values < high)


def crossings(values: np.ndarray, *, window: tuple[float, float]) -> int:
    """How many times the series enters or leaves the basin.

    The count, not the rate: two runs of different length are compared per nanosecond by the
    caller, which knows the lengths. A coordinate that never leaves its starting basin has not
    been sampled, whatever its histogram looks like.

    A COARSER frame interval sees FEWER crossings of the same dynamics, because a there-and-back
    excursion between two saved frames is invisible. So when the ladder is saved less often than
    the plain run and still crosses more, the comparison understates it.
    """
    inside = occupancy(values, window)
    return int(np.sum(inside[1:] != inside[:-1]))


def independent_sample(values: np.ndarray) -> tuple[np.ndarray, float]:
    """`values` thinned to uncorrelated samples, and the statistical inefficiency g.

    MD frames 1 ps apart are not independent draws, and every test that assumes they are reports a
    p-value computed against a sample size the run does not have. pymbar's timeseries module
    estimates g -- the number of correlated frames per independent one -- so N_eff = N / g and the
    series can be thinned to that before testing. On a torsion that rarely crosses its barrier g
    is large, which is the same fact the crossing count reports from the other side.
    """
    from pymbar import timeseries

    g = float(timeseries.statistical_inefficiency(values))
    indices = timeseries.subsample_correlated_data(values, g=g)
    return values[indices], g


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(OBSERVABLES))
    parser.add_argument("--cmd", required=True, type=Path, help="the cMD run directory")
    parser.add_argument("--cmd-build", required=True, type=Path)
    parser.add_argument("--rest2", type=Path, help="the REST2 run directory; omit for cMD alone")
    parser.add_argument("--rest2-build", type=Path)
    parser.add_argument("--window", type=float, nargs=2, default=(0.0, 120.0),
                        metavar=("LOW", "HIGH"),
                        help="the basin whose entries and exits are counted (default 0 120, the "
                             "alpha-L basin of a peptide phi)")
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)

    cmd_frames, cmd_solute = solute_trajectory(
        arguments.cmd_build / "built.pdb", arguments.cmd / "solute_prod1.nc")
    label, cmd_values, unit = observable(cmd_frames, cmd_solute, arguments.system)
    series = [("cMD", cmd_values)]

    if arguments.rest2 is not None:
        build = arguments.rest2_build or arguments.rest2.parent / "build"
        rest_frames, rest_solute = solute_trajectory(
            build / "built.pdb", arguments.rest2 / "solute_state0_prod1.nc")
        _, rest_values, _ = observable(rest_frames, rest_solute, arguments.system)
        series.append(("REST2 state 0", rest_values))

    # --- the numbers, printed so a page can quote them -------------------------------------
    print(f"{arguments.system}: {label}  [{unit}]")
    for name, values in series:
        half = len(values) // 2
        print(f"  {name:14s} n={len(values):6d}  mean={values.mean():8.2f}  sd={values.std():7.2f}"
              f"  in-basin={100*occupancy(values, tuple(arguments.window)).mean():5.2f}%"
              f"  crossings={crossings(values, window=tuple(arguments.window)):5d}"
              f"  first/second half mean={values[:half].mean():8.2f} / {values[half:].mean():8.2f}")
    if len(series) == 2:
        thinned = {}
        for name, values in series:
            sample, g = independent_sample(values)
            thinned[name] = sample
            print(f"  {name:14s} statistical inefficiency g={g:8.1f}  ->  {len(sample):5d} "
                  f"independent sample(s) of {len(values)}")
        a, b = thinned[series[0][0]], thinned[series[1][0]]
        statistic, pvalue = stats.ks_2samp(a, b)
        raw_d = stats.ks_2samp(series[0][1], series[1][1]).statistic
        print(f"  two-sample KS on the INDEPENDENT samples: D={statistic:.4f}  p={pvalue:.3g}")
        print(f"  (on every frame it would read D={raw_d:.4f}, with a p-value computed against a")
        print("   sample size neither run has -- correlated frames make that p meaningless.)")
        print("  D is the largest gap between the two cumulative distributions. These sample the")
        print("  same Hamiltonian, so at infinite sampling D would be 0; read it together with the")
        print("  crossing counts above, which say WHICH run is the unconverged one.")
        counts = {name: crossings(values, window=tuple(arguments.window)) for name, values in series}
        low = min(counts, key=counts.get)
        if counts[low] * 4 < max(counts.values()):
            print(f"  {low} crossed the barrier {counts[low]} time(s) against "
                  f"{max(counts.values())}, so it has not sampled this coordinate and any")
            print("  difference here is ITS shortfall, not evidence against the other.")
        print("  NOTE a smaller g does NOT mean better sampling. A run trapped in one basin has a")
        print("  SHORT correlation time within that basin and a small g, and has still not seen the")
        print("  free-energy surface. g measures the series it was given, not the one it missed.")

    # --- the figure -------------------------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    colours = {"cMD": "#4C4C4C", "REST2 state 0": "#C1453C"}
    for name, values in series:
        axes[0].hist(values, bins=72, density=True, histtype="stepfilled", alpha=0.45,
                     color=colours[name], label=name)
        axes[0].hist(values, bins=72, density=True, histtype="step", lw=1.6, color=colours[name])
    axes[0].set_xlabel(f"{label} / {unit}")
    axes[0].set_ylabel("probability density")
    axes[0].set_title(f"{arguments.system}: equilibrium distribution")
    axes[0].legend(frameon=False)

    low, high = arguments.window
    axes[0].axvspan(low, high, color="#C1453C", alpha=0.07, zorder=0)
    for name, values in series:
        inside = occupancy(values, (low, high))
        events = np.concatenate([[0], np.cumsum(inside[1:] != inside[:-1])])
        axes[1].plot(np.linspace(0, 1, len(values)), events, lw=1.8,
                     color=colours[name], label=f"{name} ({events[-1]})")
    axes[1].set_xlabel("fraction of the production run")
    axes[1].set_ylabel(f"cumulative entries and exits of {low:.0f}-{high:.0f} {unit}")
    axes[1].set_title("barrier crossings, which is what the ladder buys" if len(series) == 2
                      else "barrier crossings")
    axes[1].legend(frameon=False, loc="upper left")

    figure.tight_layout()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out, dpi=130)
    print(f"  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
