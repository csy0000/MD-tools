from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform as _platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .config import write_manifest
from .system import build_forcefield

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Step 3 -- solvation
# ---------------------------------------------------------------------------------------------
def _box_vectors(width_nm: float, shape: str) -> np.ndarray:
    """Reduced box vectors for a cube / rhombic dodecahedron / truncated octahedron of *width*.

    These are OpenMM's own reduced forms.  In a reduced triclinic box the minimum image distance is
    ``min(a_x, b_y, c_z)``, i.e. the smallest diagonal element -- which for a dodecahedron is
    ``width/sqrt(2)``, not ``width``.  That factor is the whole reason the padding needs care.
    """
    w = float(width_nm)
    if shape == "cube":
        return np.array([[w, 0.0, 0.0], [0.0, w, 0.0], [0.0, 0.0, w]])
    if shape == "dodecahedron":
        return np.array(
            [[w, 0.0, 0.0], [0.0, w, 0.0], [w / 2, w / 2, w * math.sqrt(2) / 2]]
        )
    if shape == "octahedron":
        return np.array(
            [
                [w, 0.0, 0.0],
                [w / 3, w * 2 * math.sqrt(2) / 3, 0.0],
                [-w / 3, w * math.sqrt(2) / 3, w * math.sqrt(6) / 3],
            ]
        )
    raise ValueError(f"unsupported solvation.box_shape {shape!r}")


def _resolve_box(modeller, cfg: dict) -> dict:
    """Decide the periodic box, reconciling the requested padding with the nonbonded cutoff.

    Two things go wrong if the box is taken straight from ``addSolvent(padding=...)``:

    1. OpenMM's padding is ``width = max(2*radius + padding, 2*padding)`` and a rhombic
       dodecahedron's minimum image distance is ``width/sqrt(2)``.  For a compact solute the
       ``2*padding`` branch wins, so ``padding = 1.2 nm`` gives alanine dipeptide a solute-to-image
       gap of **0.50 nm**, not 1.2 nm.  ``padding_semantics = "solute-image-gap"`` instead solves
       for the width that delivers the requested gap, which is what "padded by 1.2 nm from the
       solute" means.
    2. OpenMM refuses a cutoff larger than half the minimum image distance.  A 1.0 nm cutoff
       therefore needs a minimum image distance of 2.0 nm, which the raw-padding box does not
       reach for a small solute -- the run dies at ``Context`` construction.

    Both are resolved here, before any water is placed, and every number is recorded.
    """
    from openmm import unit

    scfg = cfg["solvation"]
    shape = str(scfg["box_shape"])
    padding = float(scfg["padding_nm"])
    cutoff = float(cfg["system_build"]["nonbonded_cutoff_nm"])

    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    radius = float(np.linalg.norm(positions - centre, axis=1).max())

    # the shortest diagonal element as a fraction of the width
    frac = float(np.min(np.diag(_box_vectors(1.0, shape))))

    semantics = str(scfg["padding_semantics"])
    if semantics == "openmm":
        width = max(2 * radius + padding, 2 * padding)
    elif semantics == "solute-image-gap":
        # min image distance - solute diameter >= padding
        width = (2 * radius + padding) / frac
    else:
        raise ValueError(f"unknown solvation.padding_semantics {semantics!r}")

    margin = float(cfg["system_build"]["minimum_image_margin_nm"])
    if margin < 0.0:
        raise ValueError(
            f"system_build.minimum_image_margin_nm must be >= 0, got {margin}.  A negative margin "
            "would ask for a box below OpenMM's hard minimum-image limit."
        )

    width_requested = width
    min_image = frac * width
    hard_limit = 2.0 * cutoff
    # A grown box must clear the hard limit BY THE MARGIN.  Growing to exactly 2*cutoff leaves no
    # room for the NPT contraction that immediately follows, and OpenMM's check is a hard abort.
    needed = hard_limit + margin
    policy = str(scfg["cutoff_fit_policy"])
    grown = False
    if min_image < needed:
        required_width = needed / frac
        if policy == "grow":
            # Only ever grow.  A box that is already large enough is left exactly as requested --
            # shrinking it to the threshold would silently change a system the user sized.
            width = required_width
            grown = True
        elif policy == "refuse":
            raise ValueError(
                f"a {shape} box with padding {padding} nm gives a minimum image distance of "
                f"{min_image:.3f} nm, but a {cutoff} nm cutoff with a "
                f"{margin:.3f} nm margin needs {needed:.3f} nm.  Increase solvation.padding_nm to "
                f"at least {frac * required_width - 2 * radius:.3f} nm, lower "
                f"system_build.nonbonded_cutoff_nm to {(min_image - margin) / 2:.3f} nm, reduce "
                f"system_build.minimum_image_margin_nm, or set "
                "solvation.cutoff_fit_policy='grow'."
            )
        else:
            raise ValueError(f"unknown solvation.cutoff_fit_policy {policy!r}")

    vectors = _box_vectors(width, shape)
    min_image = frac * width
    info = {
        "box_shape": shape,
        "box_width_nm": round(width, 5),
        "box_width_requested_nm": round(width_requested, 5),
        "grown_for_cutoff": grown,
        "solute_bounding_radius_nm": round(radius, 5),
        "padding_nm_requested": padding,
        "padding_semantics": semantics,
        "solute_image_gap_nm": round(min_image - 2 * radius, 5),
        "min_image_distance_nm": round(min_image, 5),
        "nonbonded_cutoff_nm": cutoff,
        "minimum_image_margin_nm": margin,
        "minimum_image_required_nm": round(needed, 5),
        "max_legal_cutoff_nm": round(min_image / 2, 5),
        "box_vectors_nm": vectors.tolist(),
    }
    if grown:
        print(
            f"[solvate] box grown for the cutoff: minimum image {width_requested * frac:.3f} -> "
            f"{min_image:.3f} nm so a {cutoff} nm cutoff fits with a {margin:.3f} nm margin "
            f"(needs {needed:.3f} nm, hard limit {hard_limit:.3f}).  The "
            f"solute-to-image gap is now {info['solute_image_gap_nm']:.3f} nm, above the "
            f"{padding} nm requested.",
            flush=True,
        )
    return info


def solvate(pdb_in: Path, out_dir: Path, cfg: dict, ligand_sdf: Optional[Path] = None,
            route: Optional[str] = None) -> dict:
    """Solvate in a rhombic-dodecahedron box with ``padding_nm`` of water and NaCl at 0.15 M.

    The solute atom indices are recorded here, before any water exists.  Modeller appends solvent,
    so the solute keeps indices ``0 .. n_solute-1``; that is asserted rather than assumed, because
    every REST2 scaling downstream is defined by that index set.
    """
    from openmm import Vec3, app, unit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scfg = cfg["solvation"]

    pdb = app.PDBFile(str(pdb_in))
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf, route=route)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    n_solute = modeller.topology.getNumAtoms()
    solute_residues = [r.name for r in modeller.topology.residues()]

    geometry = _resolve_box(modeller, cfg)
    modeller.addSolvent(
        forcefield,
        model=scfg["water_model"],
        boxVectors=unit.Quantity(
            tuple(Vec3(*v) for v in geometry["box_vectors_nm"]), unit.nanometer
        ),
        positiveIon=scfg["positive_ion"],
        negativeIon=scfg["negative_ion"],
        ionicStrength=float(scfg["ionic_strength_molar"]) * unit.molar,
        neutralize=bool(scfg["neutralize"]),
    )

    topology = modeller.topology
    # the solute must still be the first n_solute atoms
    for atom in list(topology.atoms())[:n_solute]:
        if atom.residue.name.upper() in WATER_RESIDUE_NAMES | ION_RESIDUE_NAMES:
            raise RuntimeError(
                f"atom {atom.index} ({atom.residue.name}) is solvent but falls inside the first "
                f"{n_solute} indices.  Modeller.addSolvent is expected to APPEND solvent; the "
                "solute index set that every REST2 scaling depends on is therefore wrong.  "
                "Refusing to continue."
            )

    out_pdb = out_dir / "solvated.pdb"
    with out_pdb.open("w") as fh:
        app.PDBFile.writeFile(topology, modeller.positions, fh, keepIds=True)

    box = topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    n_water = sum(1 for r in topology.residues() if r.name.upper() in WATER_RESIDUE_NAMES)
    ions: dict[str, int] = {}
    for r in topology.residues():
        if r.name.upper() in ION_RESIDUE_NAMES:
            ions[r.name] = ions.get(r.name, 0) + 1
    volume_nm3 = float(np.abs(np.linalg.det(np.array(box))))

    info = {
        "input_pdb": str(pdb_in),
        "output_pdb": str(out_pdb),
        "n_solute_atoms": n_solute,
        "solute_residues": solute_residues,
        "n_atoms_total": topology.getNumAtoms(),
        "n_waters": n_water,
        "ions": ions,
        "geometry": geometry,
        "box_shape": scfg["box_shape"],
        "padding_nm": float(scfg["padding_nm"]),
        "box_vectors_nm": [[float(x) for x in v] for v in box],
        "box_volume_nm3": volume_nm3,
        # 55.5 mol/L is the concentration of pure water; this is the salt molarity actually built
        "realised_ionic_strength_molar": (
            round(min(ions.values()) * 55.5 / n_water, 5) if ions and n_water else 0.0
        ),
        "water_model_template": scfg["water_model"],
        "forcefield": ff_info,
    }
    (out_dir / "solvation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, "step3_solvate", cfg, {"result": info})
    return info


