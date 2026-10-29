"""Figures for a torsional clustering: the compound with its torsions, and per-cluster profiles.

`show_compound` draws the molecule with each torsion's four atoms circled in that torsion's colour
and its central bond labelled `t<k>`, polar hydrogens shown and carbon-bound ones hidden (the
convention of the reference drawings). `plot_hist` draws, for the clusters of a
`hierarchical_clustering` result, the circular profile of every torsion in each cluster: as a free
energy (default; each row zero at its own minimum, a dashed line at `kT_cut`) or as a density.
`plot_split_tree` draws the tree of a `cut_clustering` result, one panel per cell.

Both import matplotlib and rdkit when called, never at import.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

__all__ = ["show_compound", "plot_hist", "plot_split_tree"]


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
              ylim_kcal: float = 5.0, mark: Optional[Sequence[int]] = None):
    """Per-cluster circular profiles of the selected torsions; returns the matplotlib Figure.

    `result` is the output of `hierarchical_clustering`. The first row is every frame, then the
    `n_cluster` most populated clusters (fewer if fewer exist; they are numbered by mass, so these
    are clusters 1..n). `torsion_show` is `'all'`, a list of names or indices, or a 1D array of them.
    `kind='free_energy'` plots F = -kT ln p in kcal/mol, zero at each row's own minimum, with a
    dashed line at the cut barrier (`result['settings']['kT_cut']` kT, at the temperature `kT` is
    given for: pass `kT` in kcal/mol, the default is 300 K, an assumption); `kind='density'` plots
    the normalised probability density, each row on one scale. `mark` is a collection of cluster
    indices (0-based, as in `result['labels']`) whose rows get a tinted background and a star in
    the row label, for example the clusters a later step can split further.
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
    marked = set(int(m) for m in (mark or ()))
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
            if sel is not None and sel in marked:
                ax.set_facecolor("#fff1b8")
            pick = np.ones(len(labels), bool) if sel is None else labels == sel
            p = np.histogram(ang[pick, j], edges)[0] / max(1, pick.sum()) / w
            if kind == "free_energy":
                with np.errstate(divide="ignore"):
                    F = -kT * np.log(p)
                F = F - np.nanmin(F[np.isfinite(F)])
                F[~np.isfinite(F)] = np.nan
                ax.plot(c, F, color=f"C{j % 10}", lw=1.5)
                ax.axhline(0, color="0.85", lw=0.6)
                if np.isfinite(barrier):
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
                ax.set_ylabel(f"{head}{' ★' if sel in marked and sel is not None else ''}\n{what}",
                              fontsize=11)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=dpi)
    return fig


def plot_split_tree(result: Dict[str, Any], *, pair: Optional[Sequence[str]] = None,
                    leaf_torsions: Optional[Sequence[str]] = None, output=None, dpi: int = 150,
                    ylim_kT: float = 9.0, title: Optional[str] = None):
    """The splitting tree of a `cut_clustering` result, one panel per cell; returns the Figure.

    Every cell sits under its parent, leaves left to right in depth-first order. Without `pair`,
    a cell that was cut shows F = -ln p (kT, zero at its minimum) of its own frames along the
    coordinate cut there, the cut in red and the boundaries the cell inherited on that coordinate
    dashed grey; a leaf shows its cluster number (1-based, by mass), its mass and the circular mean
    of each torsion named in `leaf_torsions`, in the labelling of `result["aligned_angles"]`. With
    `pair=(a, b)`, two cut-coordinate names, every cell shows the 2D histogram of its frames over
    them (log counts) with the 1D densities on the top and right edges, the cut and the inherited
    boundaries drawn on whichever of the two they lie.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if "coordinate_values" not in result or "tree" not in result:
        raise ValueError("plot_split_tree needs a cut_clustering result")
    tree = result["tree"]
    names = list(result["coordinates"])
    Y = np.degrees(np.asarray(result["coordinate_values"], float))
    valid = np.asarray(result["coordinate_valid"], bool)
    labels = np.asarray(result["labels"])
    N = len(labels)
    if pair is not None:
        pair = [names.index(str(v)) if str(v) in names else None for v in pair]
        if len(pair) != 2 or None in pair:
            raise ValueError(f"pair must name two cut coordinates of {names}")
    tnames = [n.split()[0] for n in result["torsion_names"]]
    show = []
    for v in (leaf_torsions or ()):
        if str(v) not in tnames:
            raise ValueError(f"leaf torsion {v!r} is not one of {tnames}")
        show.append(tnames.index(str(v)))
    ang = np.asarray(result["aligned_angles"], float)

    def leaves(i):
        n = tree[i]
        return [n["cluster"]] if not n["children"] else sum((leaves(c) for c in n["children"]), [])

    order, depth, X = [], {}, {}

    def walk(i, d):
        depth[i] = d
        if not tree[i]["children"]:
            order.append(i)
        for c in tree[i]["children"]:
            walk(c, d + 1)

    walk(0, 0)
    for k, i in enumerate(order):
        X[i] = float(k)

    def place(i):
        if tree[i]["children"]:
            X[i] = float(np.mean([place(c) for c in tree[i]["children"]]))
        return X[i]

    place(0)
    nx, ny = len(order), max(depth.values()) + 1
    cw, ch = (3.4, 3.6) if pair is not None else (2.6, 2.6)
    fig = plt.figure(figsize=(cw * nx, ch * ny))
    B = np.linspace(-180.0, 180.0, int(result["settings"]["n_bins"]) + 1)
    cen = 0.5 * (B[1:] + B[:-1])
    anchor = {}
    for i, n in enumerate(tree):
        rows = np.flatnonzero(np.isin(labels, leaves(i)))
        x0, y0 = X[i] / nx, 1.0 - (depth[i] + 1) / ny
        cut = n["cut"]
        head = (f"cluster {n['cluster'] + 1}: " if "cluster" in n else "") + \
            f"{len(rows)} frames ({100.0 * len(rows) / N:.2f}%)"
        if pair is not None:
            w, h = 0.62 / nx, 0.56 / ny
            left, bot = x0 + 0.13 / nx, y0 + 0.12 / ny
            ax = fig.add_axes([left, bot, w, h])
            top = fig.add_axes([left, bot + h, w, 0.16 / ny], sharex=ax)
            right = fig.add_axes([left + w, bot, 0.16 / nx, h], sharey=ax)
            a, b = pair
            H2 = np.histogram2d(Y[rows, a], Y[rows, b], [B, B])[0]
            ax.pcolormesh(B, B, np.where(H2 > 0, H2, np.nan).T, cmap="viridis", rasterized=True,
                          norm=LogNorm(vmin=1, vmax=max(2.0, H2.max())))
            for j, side, vertical in ((a, top, True), (b, right, False)):
                p = np.histogram(Y[rows, j], B)[0] / max(1, len(rows)) / (B[1] - B[0])
                if vertical:
                    side.fill_between(cen, p, color="C0", alpha=0.4, lw=0)
                else:
                    side.fill_betweenx(cen, p, color="C1", alpha=0.4, lw=0)
                side.axis("off")
            for j, axis, edge in ((a, ax.axvline, top.axvline), (b, ax.axhline, right.axhline)):
                box = n["box_degrees"].get(names[j])
                for v in (box or ()):
                    axis(v, color="0.6", ls="--", lw=1)
                if cut and cut["coordinate"] == names[j]:
                    for v in cut["positions_degrees"]:
                        axis(v, color="r", lw=1.6)
                        edge(v, color="r", lw=1.6)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-180, -90, 0, 90, 180])
            ax.tick_params(labelsize=7)
            ax.set_xlabel(names[a], fontsize=9, labelpad=0)
            ax.set_ylabel(names[b], fontsize=9, labelpad=0)
            ax.set_title(head, fontsize=9, pad=26, fontweight="bold" if "cluster" in n else None)
            anchor[i] = (left + w / 2, bot + h + 0.18 / ny, bot - 0.10 / ny)
        else:
            w, h = 0.7 / nx, 0.45 / ny
            left, bot = x0 + 0.15 / nx, y0 + 0.22 / ny
            ax = fig.add_axes([left, bot, w, h])
            if cut:
                j = names.index(cut["coordinate"])
                r = rows[valid[rows, j]]
                cnt = np.histogram(Y[r, j], B)[0].astype(float)
                with np.errstate(divide="ignore"):
                    F = -np.log(cnt / max(1.0, cnt.sum()))
                F -= F[np.isfinite(F)].min()
                F[~np.isfinite(F)] = np.nan
                ax.plot(cen, F, color="k", lw=1.2)
                for v in cut["positions_degrees"]:
                    ax.axvline(v, color="r", lw=1.5)
                for v in (n["box_degrees"].get(cut["coordinate"]) or ()):
                    ax.axvline(v, color="0.6", ls="--", lw=1)
                ax.set_ylim(-0.3, ylim_kT)
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, 0, 180])
                ax.tick_params(labelsize=7)
                ax.set_ylabel("F (kT)", fontsize=7, labelpad=0)
                ax.set_title(head, fontsize=8)
            else:
                ax.axis("off")
                ax.text(0.5, 1.0, head.replace(": ", "\n"), ha="center", va="top",
                        fontsize=10, fontweight="bold", color=_colour(n["cluster"]))
                if show:
                    z = np.exp(1j * ang[rows][:, show]).mean(axis=0)
                    text = "\n".join(f"{tnames[k]} {np.degrees(np.angle(v)):6.0f}"
                                     for k, v in zip(show, z))
                    ax.text(0.5, 0.6, text, ha="center", va="top", fontsize=7.5,
                            family="monospace")
            anchor[i] = (left + w / 2, bot + h + 0.12 / ny, bot - 0.04 / ny)
    for i, n in enumerate(tree):
        if not n["children"]:
            continue
        px, _, pb = anchor[i]
        for c in n["children"]:
            cx, ct, _ = anchor[c]
            fig.add_artist(plt.Line2D([px, cx], [pb, ct], transform=fig.transFigure,
                                      color="0.3", lw=1))
        cut = n["cut"]
        what = ("empty bins" if cut["barrier_kT"] is None else
                f"{cut['barrier_kT']:.2f} kT at E* = {cut['level_kT']:.1f}")
        where = "/".join(f"{v:.0f}°" for v in cut["positions_degrees"])
        fig.text(px, pb - 0.01 / ny,
                 f"{cut['coordinate']} {where}\n{what}" + ("\nbounded" if cut["kind"] == "bounded"
                                                         else ""),
                 ha="center", va="top", fontsize=7.5, bbox=dict(fc="w", ec="0.7", lw=0.5))
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    if output is not None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return fig
