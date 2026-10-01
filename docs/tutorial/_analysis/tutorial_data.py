"""Where the tutorial runs are, for the analysis notebooks. NO MACHINE PATH IS COMMITTED HERE.

A notebook needs the trajectories a tutorial produced, and those live wherever the reader put
them. So the root comes from the environment:

```bash
export MD_TUTORIAL_RUNS=/path/to/your/tutorial/runs
```

and a notebook that cannot find it says so with the name of the variable and the layout it
expects, rather than failing twenty cells later on an empty array. The alternative -- a default
path -- would be a machine path in a committed file, which this project forbids for the reason
that it works for exactly one person and fails confusingly for everyone else.
"""
from __future__ import annotations

import os
from pathlib import Path

#: What each notebook asks for, by name, so a missing run is reported as a missing RUN rather
#: than as a missing file.
LAYOUT = {
    "ALA-1us": "cMD-run1/solute_prod1.nc  -- 1 us of alanine dipeptide",
    "ALA-REST2": "REST2-run1/solute_state<i>_prod1.nc  -- the four-state ladder",
    "PARA-1us": "cMD-run1/solute_prod1.nc  -- 1 us of paracetamol",
    "PARA-REST2": "REST2-run1/solute_state<i>_prod1.nc  + build/TYL.sdf",
}

ENV = "MD_TUTORIAL_RUNS"


def runs_root() -> Path:
    root = os.environ.get(ENV)
    if not root:
        raise RuntimeError(
            f"set {ENV} to the directory holding the tutorial runs, e.g.\n"
            f"    export {ENV}=/path/to/tutorial-runs/tut-0.6.1\n"
            f"It is not defaulted because a default would be one machine's path.")
    path = Path(root).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{ENV}={root} is not a directory")
    return path


def run_dir(name: str) -> Path:
    """One run directory, with what it should contain named if it is absent."""
    path = runs_root() / name
    if not path.is_dir():
        raise RuntimeError(
            f"{path} does not exist. This notebook needs the `{name}` run: {LAYOUT.get(name, '')}\n"
            f"Run that tutorial first, or point {ENV} at a tree that has it.")
    return path


def solute(run: str, topology_run: str | None = None, stride: int = 1,
           trajectory: str = "cMD-run1/solute_prod1.nc"):
    """Load a SOLUTE-ONLY trajectory against its solvated topology.

    The `solute_*.nc` files hold the solute alone while `built.pdb` is the whole box, so mdtraj
    refuses the pair unless the topology is sliced to the same atoms first. Doing that here keeps
    every notebook from re-deriving the same selection slightly differently.
    """
    import mdtraj as md

    directory = run_dir(run)
    reference = md.load(str(run_dir(topology_run or run) / "build/built.pdb"))
    selection = reference.topology.select("not water and not (resname NA or resname CL)")
    traj = md.load(str(directory / trajectory), top=str(run_dir(topology_run or run)
                                                        / "build/built.pdb"),
                   atom_indices=selection)
    return traj[::stride] if stride > 1 else traj


def pmf(values, bins: int = 72, temperature_k: float = 300.0, periodic: bool = True):
    """A 1D PMF in kJ/mol from a sample of an angle, zeroed at its minimum.

    `-kT ln p`, with empty bins left as NaN rather than as a large finite number: a bin nothing
    visited has no free energy estimate, and filling it in invents a barrier height.
    """
    import numpy as np

    kT = 0.008314462618 * float(temperature_k)
    span = (-np.pi, np.pi) if periodic else (float(np.min(values)), float(np.max(values)))
    density, edges = np.histogram(np.asarray(values), bins=bins, range=span, density=True)
    free = -kT * np.log(np.where(density > 0, density, np.nan))
    return 0.5 * (edges[:-1] + edges[1:]), free - np.nanmin(free)


def pmf2d(x, y, bins: int = 72, temperature_k: float = 300.0):
    """The same on a pair of angles. Returns `(x_centres, y_centres, F)` with F zeroed."""
    import numpy as np

    kT = 0.008314462618 * float(temperature_k)
    density, xe, ye = np.histogram2d(np.asarray(x), np.asarray(y), bins=bins,
                                     range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True)
    free = -kT * np.log(np.where(density > 0, density, np.nan))
    return 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), free - np.nanmin(free)
