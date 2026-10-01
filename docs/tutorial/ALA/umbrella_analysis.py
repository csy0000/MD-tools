#!/usr/bin/env python
"""What a set of umbrella windows sampled, whether they overlap, and the PMF along phi.

Run it from the dataset root, naming the window directories in any order:

    python umbrella_analysis.py w-150 w-120 w-90 w-60 w-30
    python umbrella_analysis.py --self-test          # check the estimator against a known answer

WHY THIS IS A SCRIPT BESIDE A TUTORIAL AND NOT A SUBCOMMAND

    The engine produces the biased series and the restraint that produced it, which is what an
    estimator needs. The estimator belongs in the project asking the question: WHAM and MBAR
    bring dependencies the sampling does not need, and there is no one right binning, tolerance
    or error estimate to build in.

OVERLAP IS CHECKED BEFORE A PMF IS PRODUCED, AND A FAILURE REFUSES

    WHAM joins windows through the region where their histograms overlap. Given windows that do
    not overlap it still converges, to a curve whose relative offsets are unconstrained by any
    data -- a smooth, plausible PMF containing invented barrier heights, with nothing in the
    output saying so. So the overlap is measured first and a gap is a refusal, naming the pair.

    The reported overlap is the fraction of the SPARSER neighbour's samples that fall in the
    shared range, which is the quantity that matters: two windows whose ranges touch but whose
    populations do not are not joined by the touching.

THE BIAS IS READ FROM EACH WINDOW'S OWN RECORD, NEVER PASSED IN

    Every window carries the definition it ran under, content-addressed, as
    `umbrella.<digest>.yaml` beside its `resolved.config`. Taking centres from a command line
    instead would let the analysis describe a set of windows that was never run -- and the
    numbers would look no different.

CIRCULAR THROUGHOUT

    phi is an angle: -179 deg and 179 deg are 2 deg apart, not 358. Displacements from a window
    centre are wrapped onto (-180, 180], the same convention the restraint itself uses, and
    window means are circular means rather than arithmetic ones.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml

#: Bins across the full circle for the PMF. 5 deg is fine for a 30 deg window spacing; finer
#: bins than the data support produce a jagged curve, not a better one.
BIN_WIDTH_DEG = 5.0

#: kT at 300 K, in kJ/mol. The temperature a window ran at is in its `resolved.config`; this is
#: the default the tutorial's configuration sets, and `--temperature` overrides it.
GAS_CONSTANT_KJ = 0.0083144621


def wrap_deg(values):
    """Displacements onto (-180, 180], the convention the restraint uses."""
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values):
    radians = np.radians(np.asarray(values, dtype=float))
    return math.degrees(math.atan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))


class Window:
    """One window: the samples it drew, and the restraint it drew them under."""

    def __init__(self, name, values, centre_deg, force_constant, cv_name):
        self.name = name
        self.values = np.asarray(values, dtype=float)
        self.centre_deg = float(centre_deg)
        self.force_constant = float(force_constant)
        self.cv_name = cv_name
        #: How many leading observations were dropped, and how many the series held. Reported,
        #: because "how much of the window was thrown away" must not be silent.
        self.discarded = 0
        self.total_observations = len(self.values)

    @property
    def mean_deg(self):
        return circular_mean_deg(self.values)

    def bias_kj(self, grid_deg):
        """The restraint energy at each point of `grid_deg`, in kJ/mol.

        0.5*k*dtheta^2 with dtheta in RADIANS, because `force_constant` is per rad^2 -- the
        record names its own units (`force_constant_kj_mol_rad2`) for exactly this reason.
        """
        displacement = np.radians(wrap_deg(grid_deg - self.centre_deg))
        return 0.5 * self.force_constant * displacement ** 2


def _one_definition(directory: Path, cv_name: str | None):
    """The restraint this window actually ran under, from its own generated directory."""
    found = sorted(directory.rglob("umbrella.*.yaml"))
    if not found:
        raise SystemExit(
            f"{directory}: no umbrella.<digest>.yaml -- this is not a generated umbrella run "
            f"directory, or `build-md` was never run in it")
    digests = {path.name for path in found}
    if len(digests) > 1:
        # Several DIFFERENT definitions under one window root means two windows were generated
        # over one set of paths, and which one a given output belongs to is no longer knowable.
        raise SystemExit(
            f"{directory}: {len(digests)} different restraint definitions under one window "
            f"({', '.join(sorted(digests))}). A window is one restraint; this directory holds "
            f"more than one experiment.")
    document = yaml.safe_load(found[0].read_text(encoding="utf-8"))
    restraints = [entry for entry in document.get("restraints") or []
                  if entry.get("form") == "harmonic"]
    if not restraints:
        raise SystemExit(
            f"{found[0]}: no harmonic restraint. A flat-bottom restraint is a BOUND, not a "
            f"window: it is zero inside its range, so it defines no window to reweight.")
    if cv_name is not None:
        restraints = [entry for entry in restraints if entry["cv"] == cv_name]
        if not restraints:
            raise SystemExit(f"{found[0]}: no harmonic restraint on {cv_name}")
    if len(restraints) > 1:
        raise SystemExit(
            f"{found[0]}: {len(restraints)} harmonic restraints. This script reads a "
            f"one-dimensional profile; a multidimensional one needs a different estimator.")
    return restraints[0]


def load_window(directory: Path, cv_name: str | None = None,
                discard_fraction: float = 0.1) -> Window:
    """One window's production CV series, and the restraint it ran under.

    THE LEADING TRANSIENT IS DISCARDED, and that is not a convenience.

    The bias is applied to the PRODUCTION stage. Equilibration before it is unbiased (apart from
    positional restraints), so production begins wherever equilibration left the molecule --
    which for a window centred far from that point is outside the window entirely. The run then
    spends its first stretch being pulled in, and those configurations are not samples from the
    biased ensemble at all: they are a relaxation towards it.

    Keeping them puts weight at values the window never equilibrated at, and WHAM turns weight
    into free energy. Measured on three 10 ps windows whose equilibration ended at -161 deg: one
    such sample in 101 became the global minimum of the PMF, some 19 kJ/mol below the region the
    windows actually sampled. A longer window dilutes this without fixing it.
    """
    directory = Path(directory)
    restraint = _one_definition(directory, cv_name)
    name = restraint["cv"]

    series = sorted(path for path in directory.rglob("umbrella.cv.csv"))
    if not series:
        raise SystemExit(
            f"{directory}: no umbrella.cv.csv. The window has not been run, or was run with "
            f"collective-variable reporting disabled -- in which case there is nothing to "
            f"reweight and the run needs repeating.")
    if len(series) > 1:
        raise SystemExit(f"{directory}: {len(series)} production CV series: {series}")

    with series[0].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if name not in (rows[0] if rows else {}):
        raise SystemExit(
            f"{series[0]}: the biased variable {name!r} is not a column in the series it was "
            f"reported into (columns: {', '.join(rows[0]) if rows else 'none'}). The run biased "
            f"one variable and recorded another.")
    values = [float(row[name]) for row in rows if row[name] != ""]
    if not values:
        raise SystemExit(f"{series[0]}: no {name} values")

    discarded = int(len(values) * float(discard_fraction))
    kept = values[discarded:]
    if not kept:
        raise SystemExit(
            f"{series[0]}: discarding {discard_fraction:.0%} of {len(values)} observation(s) "
            f"leaves nothing. The window is too short to analyse.")

    window = Window(directory.name, kept, restraint["centre_deg"],
                    restraint.get("force_constant_kj_mol_rad2", restraint.get("force_constant")),
                    name)
    window.discarded = discarded
    window.total_observations = len(values)
    return window


def overlap_fraction(left: Window, right: Window, bin_width_deg: float = BIN_WIDTH_DEG) -> float:
    """The histogram overlap of two windows: sum over bins of the smaller normalised count.

    NOT a comparison of ranges. An earlier version took `max(min)` and `min(max)` of the two
    windows' values and asked what fraction of each fell inside that interval. That is wrong on a
    PERIODIC coordinate: a window straddling +/-180 holds samples at both -179 and +179, so its
    linear range is the whole circle and it "overlaps" everything. Measured on this page's own
    profile, four windows near +/-180 reported ranges of exactly -180.0 to 180.0 and mutual
    overlaps of 99.2% -- the number was highest exactly where it meant least, which is the worst
    way for a check to fail.

    The overlap coefficient has none of that. It is computed per bin on a fixed grid covering the
    circle, so -179 and +179 are neighbours only if the grid says so, and it lies in [0, 1] with
    1 for identical distributions. It also still catches what the range version was written for:
    two windows whose supports merely touch share no bin and score 0.
    """
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    left_hist, _ = np.histogram(left.values, bins=edges)
    right_hist, _ = np.histogram(right.values, bins=edges)
    if left_hist.sum() == 0 or right_hist.sum() == 0:
        return 0.0
    left_p = left_hist / left_hist.sum()
    right_p = right_hist / right_hist.sum()
    return float(np.minimum(left_p, right_p).sum())


def coverage_report(windows, *, bin_width_deg=BIN_WIDTH_DEG, minimum_overlap=0.03,
                    minimum_count=50):
    """Does this set of windows support a PMF? Returns (lines, problems, thin).

    ONE implementation, called by every script that computes a profile. It lived in two places
    for about an hour and the second copy still held the superseded pairwise test, so the two
    disagreed about the same windows -- one printed a PMF while the other refused it. Two readers
    of one policy is the defect, whatever either one says.

    WHAT WHAM ACTUALLY REQUIRES, and it is not pairwise adjacency. The first version walked the
    windows sorted by CENTRE and demanded each neighbouring pair overlap. That breaks exactly
    where a profile is hardest: a window on a steep slope slides off its centre by slope/k, so
    "next by centre" stops meaning "next in sampled space". Measured here, a window centred at
    120 deg sampled 89 deg, and the by-centre chain reported five gaps between windows that
    overlap other windows perfectly well -- while the union covered every bin and formed one
    connected graph.

    So: the sampled range must be COVERED, and the overlap graph CONNECTED. An island of windows
    joined to the rest by nothing has a free-energy offset no data constrains. Both tests are
    independent of where the centres sit and of how far any window slid.
    """
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    totals = np.zeros(len(edges) - 1, dtype=int)
    for window in windows:
        totals += np.histogram(window.values, bins=edges)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    empty = [c for c, n in zip(centres, totals) if n == 0]
    thin = [(c, n) for c, n in zip(centres, totals) if 0 < n < minimum_count]

    neighbours = {i: set() for i in range(len(windows))}
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            if overlap_fraction(windows[i], windows[j], bin_width_deg) >= minimum_overlap:
                neighbours[i].add(j)
                neighbours[j].add(i)
    reached, stack = {0}, [0]
    while stack:
        here = stack.pop()
        for other in neighbours[here] - reached:
            reached.add(other)
            stack.append(other)
    islands = sorted(set(range(len(windows))) - reached)

    lines = [
        f"coverage: {int((totals > 0).sum())} of {len(totals)} bins sampled; thinnest sampled "
        f"bin {int(totals[totals > 0].min()) if (totals > 0).any() else 0}, "
        f"median {int(np.median(totals))}",
        f"overlap graph: {len(windows)} windows, largest component {len(reached)}"
        + ("  connected" if not islands else ""),
    ]
    problems = []
    if empty:
        problems.append(
            f"{len(empty)} bin(s) inside the range have NO samples from any window, at phi = "
            + ", ".join(f"{c:.1f}" for c in empty[:8]) + (" ..." if len(empty) > 8 else ""))
    if islands:
        problems.append("these windows are joined to the rest by nothing: "
                        + ", ".join(windows[i].name for i in islands))
    return lines, problems, thin


REFUSAL_ADVICE = (
    "  WHAM would still converge, to a curve whose offsets across the unsampled region no data\n"
    "  constrains -- a smooth PMF with invented barrier heights and nothing in it saying which\n"
    "  parts are unsupported. Add windows there, or stiffen them: a window on a slope sits\n"
    "  slope/k away from its centre, so on a steep barrier a loose restraint slides off and\n"
    "  leaves the top unsampled."
)

def wham(windows, temperature_k=300.0, bin_width_deg=BIN_WIDTH_DEG,
         tolerance=1e-7, max_iterations=100000):
    """Binned WHAM on a periodic coordinate. Returns (bin centres, PMF in kJ/mol).

    The standard self-consistent pair: the unbiased distribution from every window's samples
    weighted by the biases they ran under, and each window's free-energy offset from that
    distribution, iterated until the offsets stop moving.
    """
    kt = GAS_CONSTANT_KJ * float(temperature_k)
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts = np.array([np.histogram(window.values, bins=edges)[0] for window in windows],
                      dtype=float)
    samples = counts.sum(axis=1, keepdims=True)
    # exp(-bias/kT) for every window at every bin: the weight each window's samples carry.
    bias = np.array([window.bias_kj(centres) for window in windows], dtype=float)
    boltzmann = np.exp(-bias / kt)

    offsets = np.zeros(len(windows))
    total = counts.sum(axis=0)
    for _ in range(max_iterations):
        weights = (samples * np.exp(offsets[:, None] / kt) * boltzmann).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            probability = np.where(weights > 0, total / weights, 0.0)
        probability = probability / probability.sum()
        with np.errstate(divide="ignore"):
            updated = -kt * np.log((probability * boltzmann).sum(axis=1))
        updated -= updated[0]
        if np.max(np.abs(updated - offsets)) < tolerance:
            offsets = updated
            break
        offsets = updated
    else:
        print("warning: WHAM did not converge to the tolerance; the PMF below is not final",
              file=sys.stderr)

    with np.errstate(divide="ignore"):
        pmf = -kt * np.log(probability)
    occupied = np.isfinite(pmf)
    pmf[occupied] -= pmf[occupied].min()
    return centres, pmf


def self_test() -> int:
    """The estimator against an answer known in advance.

    Windows are sampled from a KNOWN PMF plus each window's own bias, analytically rather than by
    simulation, so what comes back can be compared with the curve that generated it. An estimator
    that is never checked this way returns a plausible curve whichever way it is wrong.

    WHAT IT CATCHES, measured by breaking it on purpose rather than assumed:

        force constant wrong by 2x          max deviation 21.7 kJ/mol   caught
        one window's centre wrong by 20 deg               7.5 kJ/mol    caught
        one window's centre wrong by 40 deg              15.1 kJ/mol    caught
        EVERY centre shifted by 20 deg                    0.4 kJ/mol    NOT caught

    The last line is a property of this test's geometry, not a hidden strength of the estimator:
    the synthetic windows are uniformly spaced 20 deg apart around the whole circle, so shifting
    all of them by one spacing relabels each window as its neighbour and the SET of biases is
    unchanged. A uniform shift is invisible to any check built on a uniform, periodic, fully
    covering set. Real windows are read from their own records rather than typed, which is what
    makes that case unreachable in practice -- but it is a limit of the test and is written down
    rather than left for someone to assume away.
    """
    rng = np.random.default_rng(20260930)
    kt = GAS_CONSTANT_KJ * 300.0
    force_constant = 100.0

    # A double well in phi, in kJ/mol: minima near -120 and +60 deg.
    def truth(phi):
        phi = wrap_deg(phi)
        return 0.5 * 0.004 * (phi + 120.0) ** 2 * np.exp(-((phi + 120.0) / 90.0) ** 2) + \
            8.0 * (1.0 - np.cos(np.radians(2.0 * (phi - 60.0))))

    grid = np.arange(-180.0, 180.0, 1.0)
    windows = []
    for centre in range(-180, 180, 20):
        # Sample the biased distribution of this window directly, by inverse transform over a
        # fine grid: exp(-(V + bias)/kT).
        displacement = np.radians(wrap_deg(grid - centre))
        biased = truth(grid) + 0.5 * force_constant * displacement ** 2
        weights = np.exp(-(biased - biased.min()) / kt)
        weights /= weights.sum()
        drawn = rng.choice(grid, size=4000, p=weights)
        windows.append(Window(f"w{centre}", drawn, centre, force_constant, "phi_ALA"))

    centres, pmf = wham(windows, temperature_k=300.0, bin_width_deg=5.0)
    expected = truth(centres)
    finite = np.isfinite(pmf)
    expected = expected - expected[finite].min()
    deviation = np.abs(pmf[finite] - expected[finite])

    print(f"self-test: {len(windows)} synthetic windows over a known double well")
    print(f"  bins recovered        {finite.sum()} of {len(centres)}")
    print(f"  max deviation         {deviation.max():.2f} kJ/mol")
    print(f"  rms deviation         {math.sqrt(float(np.mean(deviation ** 2))):.2f} kJ/mol")
    # Generous, because the comparison includes bins in the barrier top where both curves are
    # steep and the sampled one is noisy. It is nowhere near loose enough to pass a PMF that is
    # wrong in shape, which is what it is for.
    if deviation.max() > 2.0:
        print("  FAILED: the recovered PMF does not match the one that generated the samples")
        return 1
    print("  OK: the recovered PMF matches the one that generated the samples")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("windows", nargs="*", type=Path, help="window directories")
    parser.add_argument("--cv", default=None, help="which variable, when a window biases several")
    parser.add_argument("--temperature", type=float, default=300.0, help="kelvin")
    parser.add_argument("--bin-width", type=float, default=BIN_WIDTH_DEG, help="degrees")
    parser.add_argument("--minimum-overlap", type=float, default=0.03,
                        help="two windows count as joined above this histogram overlap")
    parser.add_argument("--minimum-count", type=int, default=50,
                        help="warn about a sampled bin holding fewer samples than this")
    parser.add_argument("--discard-fraction", type=float, default=0.1,
                        help="leading fraction of each window discarded as the relaxation into "
                             "the window rather than sampling of it (default 0.1)")
    parser.add_argument("--self-test", action="store_true",
                        help="check the estimator against a known PMF and exit")
    arguments = parser.parse_args(argv)

    if arguments.self_test:
        return self_test()
    if not arguments.windows:
        parser.error("name the window directories, or pass --self-test")

    windows = sorted((load_window(path, arguments.cv, arguments.discard_fraction)
                      for path in arguments.windows),
                     key=lambda window: window.centre_deg)

    print(f"{len(windows)} window(s) on {windows[0].cv_name}, "
          f"leading {arguments.discard_fraction:.0%} of each discarded as relaxation")
    print(f"{'window':>10}  {'centre':>8}  {'k':>7}  {'kept':>6}  {'dropped':>7}  {'mean':>8}  "
          f"{'offset':>7}  {'range':>18}")
    for window in windows:
        offset = wrap_deg(window.mean_deg - window.centre_deg)
        print(f"{window.name:>10}  {window.centre_deg:>7.1f}d  {window.force_constant:>7.1f}  "
              f"{len(window.values):>6}  {window.discarded:>7}  {window.mean_deg:>7.1f}d  "
              f"{float(offset):>6.1f}d  "
              f"{window.values.min():>8.1f} to {window.values.max():>6.1f}")

    print("\na window's mean sits off its centre wherever the free energy has a slope; that is")
    print("expected at finite force constant, and is why windows are reweighted, not read.")

    lines, problems, thin = coverage_report(
        windows, bin_width_deg=arguments.bin_width, minimum_overlap=arguments.minimum_overlap,
        minimum_count=arguments.minimum_count)
    print()
    for line in lines:
        print(line)
    if problems:
        print("REFUSING a PMF.", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(REFUSAL_ADVICE, file=sys.stderr)
        return 2
    if thin:
        print(f"  WARNING: {len(thin)} sampled bin(s) hold fewer than {arguments.minimum_count} "
              f"samples; the PMF there rests on very little", file=sys.stderr)

    centres, pmf = wham(windows, temperature_k=arguments.temperature,
                        bin_width_deg=arguments.bin_width)
    print(f"\nPMF along {windows[0].cv_name}, kJ/mol, zeroed at its minimum "
          f"({arguments.bin_width:g} deg bins, {arguments.temperature:g} K)")
    for centre, value in zip(centres, pmf):
        if math.isfinite(value):
            print(f"  {centre:>7.1f}d  {value:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
