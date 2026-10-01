#!/usr/bin/env python
"""Check an umbrella PMF against an UNBIASED run of the same Hamiltonian.

    python umbrella_vs_unbiased.py <reference cv.csv> <window dir> [<window dir> ...]

WHY THIS IS THE CHECK THAT COUNTS

An umbrella profile is a reweighted quantity: a wrong force constant, a window centred in the
wrong units, a misaligned frame index or a gap between windows all produce a smooth, plausible
curve. Nothing inside the calculation objects. The only thing that settles whether it is right is
an independent measurement of the same quantity -- and for a system whose barriers a plain run
can cross, that measurement exists: the Boltzmann histogram of an unbiased trajectory,
`F = -kT ln p`, with no reweighting anywhere in it.

THE HAMILTONIAN MUST MATCH, WHICH IS EASY TO GET WRONG

The reference must be the SAME potential, not merely the same molecule. The alanine dipeptide
tutorial very nearly compared an implicit-solvent (GBn2) umbrella profile against the system's
1 microsecond EXPLICIT TIP3P run, because both were "the ALA phi distribution". Those are two
different Hamiltonians with genuinely different free-energy surfaces, so the comparison could not
have failed for the right reason: agreement would have been luck and disagreement would have been
uninterpretable. A check that cannot fail correctly is not a check.

WHERE THE COMPARISON IS MEANINGFUL

Only where the unbiased run actually sampled. That is the whole reason the umbrella profile exists
-- it covers barriers the plain run never crosses -- so the reference is silent exactly where the
profile is most valuable. This script therefore compares only bins whose unbiased population
exceeds a floor, reports how many bins that leaves, and says plainly that the rest is unchecked
rather than quietly averaging over it.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umbrella_analysis import (BIN_WIDTH_DEG, GAS_CONSTANT_KJ, REFUSAL_ADVICE,  # noqa: E402
                               coverage_report, load_window, wham)


def unbiased_profile(path: Path, column: str, *, bin_width_deg: float, temperature_k: float,
                     discard_fraction: float):
    """`F = -kT ln p` from an unbiased series. No reweighting, nothing to get wrong."""
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if column not in (rows[0] if rows else {}):
        raise SystemExit(f"{path}: no column {column!r}; has {', '.join(rows[0] or [])}")
    values = np.array([float(r[column]) for r in rows if r[column] != ""])
    values = values[int(len(values) * discard_fraction):]

    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    counts, _ = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    probability = counts / counts.sum()
    kt = GAS_CONSTANT_KJ * temperature_k
    with np.errstate(divide="ignore"):
        free_energy = -kt * np.log(probability)
    crossings = int((np.diff(np.where(values < 0, -1, 1)) != 0).sum())
    return centres, free_energy, counts, len(values), crossings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="an UNBIASED run's *.cv.csv")
    parser.add_argument("windows", nargs="+", type=Path)
    parser.add_argument("--cv", default=None)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bin-width", type=float, default=BIN_WIDTH_DEG)
    parser.add_argument("--discard-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-overlap", type=float, default=0.03)
    parser.add_argument("--minimum-count", type=int, default=50,
                        help="bins with fewer unbiased samples are UNCHECKED, not compared")
    arguments = parser.parse_args(argv)

    windows = sorted((load_window(p, arguments.cv, arguments.discard_fraction)
                      for p in arguments.windows), key=lambda w: w.centre_deg)
    name = windows[0].cv_name

    # THE SAME GATE the profile script applies, imported rather than restated. This file carried
    # its own copy of the superseded pairwise test for an hour, so the two scripts disagreed
    # about the same windows: one printed a PMF and the other refused it.
    lines, problems, _thin = coverage_report(
        windows, bin_width_deg=arguments.bin_width, minimum_overlap=arguments.minimum_overlap,
        minimum_count=arguments.minimum_count)
    for line in lines:
        print(line)
    if problems:
        print("REFUSING: these windows do not support a PMF.", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(REFUSAL_ADVICE, file=sys.stderr)
        return 2

    centres, pmf = wham(windows, temperature_k=arguments.temperature,
                        bin_width_deg=arguments.bin_width)
    ref_centres, reference, counts, n, crossings = unbiased_profile(
        arguments.reference, name, bin_width_deg=arguments.bin_width,
        temperature_k=arguments.temperature, discard_fraction=arguments.discard_fraction)
    assert np.allclose(centres, ref_centres), "the two profiles are on different grids"

    print(f"{len(windows)} windows on {name}; unbiased reference {n} observations, "
          f"{crossings} sign changes across 0")
    if crossings < 10:
        print("  WARNING: the reference barely crosses, so it is not a reference for the "
              "barrier it does not cross", file=sys.stderr)

    checkable = (counts >= arguments.minimum_count) & np.isfinite(pmf)
    if not checkable.any():
        print("no bin has enough unbiased samples to compare", file=sys.stderr)
        return 2

    # Both curves are free energies up to a constant, so they are aligned on their SHARED
    # support before being compared -- not on each curve's own minimum, which may sit in a
    # different bin and would import an offset the data does not contain.
    shift = np.mean(pmf[checkable] - reference[checkable])
    deviation = (pmf[checkable] - shift) - reference[checkable]

    print(f"\ncomparable bins (>= {arguments.minimum_count} unbiased samples): "
          f"{int(checkable.sum())} of {len(centres)}")
    print(f"  rms deviation   {math.sqrt(float(np.mean(deviation ** 2))):.2f} kJ/mol")
    print(f"  max deviation   {float(np.max(np.abs(deviation))):.2f} kJ/mol "
          f"at phi = {float(centres[checkable][np.argmax(np.abs(deviation))]):.1f} deg")
    print(f"  (kT at {arguments.temperature:g} K is "
          f"{GAS_CONSTANT_KJ * arguments.temperature:.2f} kJ/mol)")

    print(f"\n{'phi':>8} {'umbrella':>9} {'unbiased':>9} {'diff':>7} {'n_unbiased':>11}")
    for i in np.where(checkable)[0]:
        print(f"{centres[i]:8.1f} {pmf[i] - shift:9.2f} {reference[i]:9.2f} "
              f"{deviation[list(np.where(checkable)[0]).index(i)]:7.2f} {counts[i]:11d}")

    unchecked = int((~checkable & np.isfinite(pmf)).sum())
    print(f"\n{unchecked} bin(s) carry an umbrella PMF the reference cannot check -- it did not "
          f"sample them.\nThat is what the windows were run FOR, and it is reported as unchecked "
          f"rather than averaged into the agreement above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
