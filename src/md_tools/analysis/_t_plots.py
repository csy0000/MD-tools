"""Figures for a torsional clustering: the compound with its torsions, and per-cluster profiles.

`show_compound` draws the molecule with each torsion's four atoms circled in that torsion's colour
and its central bond labelled `t<k>`, polar hydrogens shown and carbon-bound ones hidden (the
convention of the reference drawings). `plot_hist` draws, for the clusters of a
`hierarchical_clustering` result, the circular profile of every torsion in each cluster: as a free
energy (default; each row zero at its own minimum, a dashed line at `kT_cut`) or as a density.

Both import matplotlib and rdkit when called, never at import.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

__all__ = ["show_compound", "plot_hist"]


def _colour(k: int):
    from matplotlib import colors as mcolors

    return mcolors.to_rgb(f"C{k % 10}")


def _drawing(molecule, quads, labels, size: int):
    """RGBA image of the molecule, torsions highlighted; labels is None or one text per torsion."""
    import io

    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen
    from rdkit.Chem.Draw import rdMolDraw2D

    hide = sorted((a.GetIdx() for a in molecule.GetAtoms()
                   if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() == 6), reverse=True)
    keep = [a.GetIdx() for a in molecule.GetAtoms() if a.GetIdx() not in set(hide)]
    new = {o: i for i, o in enumerate(keep)}
    m = Chem.RWMol(molecule)
    for a in hide:
        m.RemoveAtom(a)
    m = m.GetMol()
    Chem.SanitizeMol(m)
    rdCoordGen.AddCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = d.drawOptions()
    opts.bondLineWidth = 2
    opts.annotationFontScale = 0.8
    atom_col, bond_col, radius = {}, {}, {}
    for k, q in enumerate(quads):
        col = _colour(k)
        qi = [new[int(x)] for x in q]
        for x in qi:
            atom_col[x], radius[x] = col, 0.32
        for u, v in zip(qi[:-1], qi[1:]):
            bond_col[m.GetBondBetweenAtoms(u, v).GetIdx()] = col
        if labels is not None:
            m.GetBondBetweenAtoms(qi[1], qi[2]).SetProp("bondNote", labels[k])
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=list(atom_col),
                                       highlightAtomColors=atom_col, highlightAtomRadii=radius,
                                       highlightBonds=list(bond_col), highlightBondColors=bond_col)
    d.FinishDrawing()
    return plt.imread(io.BytesIO(d.GetDrawingText()))


def show_compound(molecule, quads: Sequence[Sequence[int]], *, torsion_enumerate: bool = True,
                  names: Optional[Sequence[str]] = None, size: int = 1600, ax=None, output=None,
                  dpi: int = 300):
    """Draw `molecule` with each torsion highlighted; returns the matplotlib Axes.

    `quads` are reference-molecule atom indices (a `hierarchical_clustering` result carries them as
    `result["quads"]` with the molecule as `result["molecule"]`). With `torsion_enumerate=True` each
    central bond is labelled `t1`, `t2`, ... in the order given; `names` replaces the label text.
    """
    import matplotlib.pyplot as plt

    quads = [tuple(int(a) for a in q) for q in quads]
    if names is not None and len(names) != len(quads):
        raise ValueError(f"{len(names)} names for {len(quads)} torsions")
    labels = None
    if torsion_enumerate:
        labels = list(names) if names is not None else [f"t{k + 1}" for k in range(len(quads))]
    img = _drawing(molecule, quads, labels, int(size))
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.axis("off")
    if output is not None:
        (fig or ax.figure).savefig(output, dpi=dpi, bbox_inches="tight")
    return ax


def _select_torsions(spec, names) -> list:
    if isinstance(spec, str):
        if spec != "all":
            raise ValueError("torsion_show must be 'all', a list of torsion names or indices, or a "
                             "1D numpy array of them")
        return list(range(len(names)))
    arr = np.asarray(spec)
    if arr.ndim != 1:
        raise ValueError("torsion_show must be 1D")
    out = []
    for v in arr.tolist():
        if isinstance(v, str):
            hit = [i for i, n in enumerate(names) if n == v or n.split()[0] == v]
            if len(hit) != 1:
                raise ValueError(f"torsion {v!r} is not one of {names}")
            out.append(hit[0])
        else:
            i = int(v)
            if not 0 <= i < len(names):
                raise ValueError(f"torsion index {i} outside 0..{len(names) - 1}")
            out.append(i)
    return out


def plot_hist(result: Dict[str, Any], *, torsion_show: Union[str, Sequence] = "all",
              n_cluster: int = 10, kind: str = "free_energy", kT: float = 0.0019872 * 300.0,
              n_bins: int = 72, show_molecule: bool = True, output=None, dpi: int = 200,
              ylim_kcal: float = 5.0):
    """Per-cluster circular profiles of the selected torsions; returns the matplotlib Figure.

    `result` is the output of `hierarchical_clustering`. The first row is every frame, then the
    `n_cluster` most populated clusters (fewer if fewer exist; they are numbered by mass, so these
    are clusters 1..n). `torsion_show` is `'all'`, a list of names or indices, or a 1D array of them.
    `kind='free_energy'` plots F = -kT ln p in kcal/mol, zero at each row's own minimum, with a
    dashed line at the cut barrier (`result['settings']['kT_cut']` kT, at the temperature `kT` is
    given for: pass `kT` in kcal/mol, the default is 300 K, an assumption); `kind='density'` plots
    the normalised probability density, each row on one scale.
    """
    import matplotlib.pyplot as plt

    if kind not in ("free_energy", "density"):
        raise ValueError("kind must be 'free_energy' or 'density'")
    if int(n_cluster) < 1:
        raise ValueError("n_cluster must be at least 1")
    names = list(result["torsion_names"])
    cols = _select_torsions(torsion_show, names)
    if not cols:
        raise ValueError("no torsion selected")
    labels = np.asarray(result["labels"])
    ang = np.asarray(result["aligned_angles"])
    clusters = list(range(min(int(n_cluster), int(result["n_clusters"]))))
    rows = [None] + clusters
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    barrier = float(result["settings"]["kT_cut"]) * float(kT)
    n = len(cols)
    ratios = ([5] if show_molecule else []) + [1] * n
    fig = plt.figure(figsize=(1.9 * n + (7 if show_molecule else 0), 1.35 * len(rows) + 1.6))
    gs = fig.add_gridspec(len(rows), len(ratios), width_ratios=ratios)
    off = 1 if show_molecule else 0
    if show_molecule:
        ax0 = fig.add_subplot(gs[:, 0])
        show_compound(result["molecule"], [result["quads"][j] for j in cols], ax=ax0,
                      names=[names[j].split()[0] for j in cols])
        ax0.set_title("tk marks the central bond of torsion k")
    total = float(len(labels))
    ymax = {}
    for ri, sel in enumerate(rows):
        pick = np.ones(len(labels), bool) if sel is None else labels == sel
        ymax[ri] = max(np.histogram(ang[pick, j], edges)[0].max() / max(1, pick.sum()) / w
                       for j in cols) * 1.08
    for cj, j in enumerate(cols):
        for ri, sel in enumerate(rows):
            ax = fig.add_subplot(gs[ri, off + cj])
            pick = np.ones(len(labels), bool) if sel is None else labels == sel
            p = np.histogram(ang[pick, j], edges)[0] / max(1, pick.sum()) / w
            if kind == "free_energy":
                with np.errstate(divide="ignore"):
                    F = -kT * np.log(p)
                F = F - np.nanmin(F[np.isfinite(F)])
                F[~np.isfinite(F)] = np.nan
                ax.plot(c, F, color=f"C{j % 10}", lw=1.5)
                ax.axhline(0, color="0.85", lw=0.6)
                ax.axhline(barrier, color="k", ls="--", lw=1.0, alpha=0.4)
                ax.set_ylim(-0.3, ylim_kcal)
            else:
                ax.fill_between(c, p, color=f"C{j % 10}", alpha=0.35, lw=0)
                ax.plot(c, p, color=f"C{j % 10}", lw=1.3)
                ax.set_ylim(0, ymax[ri])
                ax.set_yticks([])
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels(["-π", "0", "π"] if ri == len(rows) - 1 else ["", "", ""], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            if kind == "free_energy" and cj > 0:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(names[j].replace(" ", "\n", 1), fontsize=11, color=f"C{j % 10}")
            if cj == 0:
                what = "F (kcal/mol)" if kind == "free_energy" else "density"
                head = "all frames" if sel is None else (
                    f"cluster {sel + 1} ({100 * (labels == sel).sum() / total:.1f}%)")
                ax.set_ylabel(f"{head}\n{what}", fontsize=11)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=dpi)
    return fig
