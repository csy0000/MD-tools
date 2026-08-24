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
import contextlib
import random
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

    Delegated to OpenMM's own `Modeller._computeBoxVectors`, so there is one definition of these
    shapes rather than two that can drift apart. In a reduced triclinic box the minimum image
    distance is `min(a_x, b_y, c_z)` -- the smallest diagonal element, which for a dodecahedron is
    `width/sqrt(2)` and not `width`. That factor is the whole reason padding needs care.

    `_computeBoxVectors` is private, so `test_box_vectors_match_openmm` asserts the local fallback
    below still reproduces it exactly. If OpenMM ever moves or changes it, that test fails loudly
    and the fallback keeps working rather than the build breaking.
    """
    w = float(width_nm)
    if shape not in _SUPPORTED_BOX_SHAPES:
        raise ValueError(f"unsupported solvation.box_shape {shape!r}")
    try:
        from openmm.app import Modeller

        vectors = Modeller._computeBoxVectors(None, w, shape)
        return np.array([[v.x, v.y, v.z] for v in vectors])
    except (ImportError, AttributeError, ValueError, TypeError):
        return _box_vectors_fallback(w, shape)


#: The shapes this package supports, which is the set OpenMM's `_computeBoxVectors` accepts.
_SUPPORTED_BOX_SHAPES = ("cube", "dodecahedron", "octahedron")


def _box_vectors_fallback(width_nm: float, shape: str) -> np.ndarray:
    """OpenMM's reduced forms, written out. Only used if the private helper is unavailable."""
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


def shortest_lattice_translation(vectors) -> float:
    """The shortest nonzero lattice translation, by enumeration rather than by formula.

    This is the distance between periodic images of a *point*, and therefore the quantity a
    solute-to-periodic-copy clearance is measured against. For OpenMM's reduced cube, rhombic
    dodecahedron and truncated octahedron it equals the box width -- verified here by enumeration
    rather than asserted, because it is easy to reach for the wrong number.

    It is NOT the quantity OpenMM's cutoff check uses. See `minimum_reduced_box_height`.
    """
    import itertools

    vectors = np.asarray(vectors, dtype=float)
    best = np.inf
    for i, j, k in itertools.product((-2, -1, 0, 1, 2), repeat=3):
        if (i, j, k) == (0, 0, 0):
            continue
        best = min(best, float(np.linalg.norm(i * vectors[0] + j * vectors[1] + k * vectors[2])))
    return best


def minimum_reduced_box_height(vectors) -> float:
    """The smallest perpendicular height of the reduced cell: `min(a_x, b_y, c_z)`.

    This is what OpenMM's periodic-box check compares the cutoff against -- it refuses a cutoff
    larger than half of it -- because a cutoff must not reach beyond the slab the minimum-image
    convention can resolve. For a cube it equals the width; for a rhombic dodecahedron it is
    `width/sqrt(2)`; for a truncated octahedron `sqrt(6)/3 * width`.

    Using this as the solute-image distance understates the real separation badly: for a
    dodecahedron it is 29% smaller than the shortest lattice translation, which is the difference
    between "0.7 nm of clearance" and "2.4 nm of clearance" on a real macrocycle box.
    """
    return float(np.min(np.diag(np.asarray(vectors, dtype=float))))


def _resolve_box(modeller, cfg: dict) -> dict:
    """Decide the periodic box, reconciling the requested padding with the nonbonded cutoff.

    Three quantities are involved and they are NOT interchangeable. Conflating the first two is the
    error this function was previously written around:

    * **shortest lattice translation** -- how far a point is from its own periodic image. Equal to
      the box width for all three shapes OpenMM builds. This is what a solute-image clearance is
      measured against.
    * **minimum reduced-box height** -- `min(a_x, b_y, c_z)`, which is what OpenMM's cutoff legality
      check uses. Smaller than the width for any non-cubic shape.
    * **solute bounding radius** -- half the solute's diameter, from OpenMM's own definition.

    So the conservative clearance is `shortest_lattice_translation - 2*radius`, while the cutoff must
    satisfy `minimum_reduced_box_height >= 2*cutoff + margin`. Both are enforced and both are
    recorded, under names that say which is which.

    OpenMM itself has no numeric `addSolvent` padding default; the default in this repository is its
    own choice, expressed in OpenMM's semantics.
    """
    from openmm import unit

    scfg = cfg["solvation"]
    shape = str(scfg["box_shape"])
    padding = float(scfg["padding_nm"])
    cutoff = float(cfg["system_build"]["nonbonded_cutoff_nm"])

    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    radius = float(np.linalg.norm(positions - centre, axis=1).max())

    # per unit width, for this shape
    unit_vectors = _box_vectors(1.0, shape)
    height_fraction = minimum_reduced_box_height(unit_vectors)
    translation_fraction = shortest_lattice_translation(unit_vectors)

    semantics = str(scfg["padding_semantics"])
    if semantics == "openmm":
        # exactly Modeller.addSolvent
        width = max(2 * radius + padding, 2 * padding)
    elif semantics == "solute-image-gap":
        # solve for the width that delivers `padding` of clearance to the nearest periodic COPY,
        # which is set by the shortest lattice translation and not by the reduced-box height
        width = (2 * radius + padding) / translation_fraction
    else:
        raise ValueError(f"unknown solvation.padding_semantics {semantics!r}")

    margin = float(cfg["system_build"]["minimum_image_margin_nm"])
    if margin < 0.0:
        raise ValueError(
            f"system_build.minimum_image_margin_nm must be >= 0, got {margin}.  A negative margin "
            "would ask for a box below OpenMM's hard cutoff-height limit."
        )

    width_requested = width
    height = height_fraction * width
    hard_limit = 2.0 * cutoff
    # A grown box must clear the hard limit BY THE MARGIN.  Growing to exactly 2*cutoff leaves no
    # room for the NPT contraction that immediately follows, and OpenMM's check is a hard abort.
    needed = hard_limit + margin
    policy = str(scfg["cutoff_fit_policy"])
    grown = False
    if height < needed:
        required_width = needed / height_fraction
        if policy == "grow":
            # Only ever grow.  A box that is already large enough is left exactly as requested --
            # shrinking it to the threshold would silently change a system the user sized.
            width = required_width
            grown = True
        elif policy == "refuse":
            raise ValueError(
                f"a {shape} box with padding {padding} nm gives a reduced-box height of "
                f"{height:.3f} nm, but a {cutoff} nm cutoff with a "
                f"{margin:.3f} nm margin needs {needed:.3f} nm.  Increase solvation.padding_nm to "
                f"at least {required_width - 2 * radius:.3f} nm, lower "
                f"system_build.nonbonded_cutoff_nm to {(height - margin) / 2:.3f} nm, reduce "
                f"system_build.minimum_image_margin_nm, or set "
                "solvation.cutoff_fit_policy='grow'."
            )
        else:
            raise ValueError(f"unknown solvation.cutoff_fit_policy {policy!r}")

    vectors = _box_vectors(width, shape)
    height = minimum_reduced_box_height(vectors)
    translation = shortest_lattice_translation(vectors)
    clearance = translation - 2 * radius
    info = {
        "box_shape": shape,
        "box_width_nm": round(width, 5),
        "box_width_requested_nm": round(width_requested, 5),
        "box_volume_nm3": round(float(abs(np.linalg.det(vectors))), 5),
        "grown_for_cutoff": grown,
        "solute_bounding_radius_nm": round(radius, 5),
        "padding_nm_requested": padding,
        "padding_semantics": semantics,
        # --- distance to the nearest periodic COPY -------------------------------------------
        "shortest_lattice_translation_nm": round(translation, 5),
        "solute_image_clearance_nm": round(clearance, 5),
        # --- the quantity OpenMM's cutoff check uses -----------------------------------------
        "min_reduced_box_height_nm": round(height, 5),
        "nonbonded_cutoff_nm": cutoff,
        "minimum_image_margin_nm": margin,
        "required_cutoff_height_nm": round(needed, 5),
        "max_legal_cutoff_nm": round(height / 2, 5),
        # --- LEGACY, retained so old readers do not break ------------------------------------
        # `min_image_distance_nm` historically held the reduced-box HEIGHT while being described as
        # the minimum image distance, and `solute_image_gap_nm` was that height minus the solute
        # diameter. Neither is the solute-to-periodic-copy distance; both are kept only so existing
        # manifests and tooling still parse. Read the two blocks above instead.
        "min_image_distance_nm": round(height, 5),
        "solute_image_gap_nm": round(height - 2 * radius, 5),
        "minimum_image_required_nm": round(needed, 5),
        "legacy_field_note": (
            "min_image_distance_nm and solute_image_gap_nm are the REDUCED-BOX HEIGHT and that "
            "height minus the solute diameter. They are not the solute-to-periodic-copy distance; "
            "use shortest_lattice_translation_nm and solute_image_clearance_nm."
        ),
        "box_vectors_nm": vectors.tolist(),
    }
    if grown:
        print(
            f"[solvate] box grown for the cutoff: reduced-box height "
            f"{width_requested * height_fraction:.3f} -> {height:.3f} nm so a {cutoff} nm cutoff "
            f"fits with a {margin:.3f} nm margin (needs {needed:.3f} nm, hard limit "
            f"{hard_limit:.3f}).  Solute-to-copy clearance is now {clearance:.3f} nm.",
            flush=True,
        )
    return info



#: OpenMM's `Modeller.addSolvent` can only BUILD a pre-equilibrated box for the handful of models it
#: ships boxes for. The scientific water model is decided by the FORCE FIELD, not by the packing
#: geometry, so a model without its own box is packed using a same-topology stand-in and then
#: parameterised by its own force field. OpenMM's own documentation sanctions this: "a box of
#: TIP4P-Ew water can be used for most four site water models".
#:
#: Only same-site-count substitutions appear here. Packing a four-site model into a three-site box
#: would leave the virtual sites unplaced, so those are refused rather than approximated.
_PACKING_MODEL = {
    "opc": "tip4pew",       # 4-site, like TIP4P-Ew
    "opc3": "tip3p",        # 3-site
    "tip4pfb": "tip4pew",   # 4-site
    "tip3pfb": "tip3p",     # 3-site
}

#: Models `addSolvent` can build a box for directly.
_NATIVE_PACKING_MODELS = frozenset({"tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"})

#: Site count per water model. The packing geometry and the parameters must agree on this number.
#: A 4-site model packed into a 3-site box leaves its virtual sites unplaced; a 3-site force field
#: handed a 4-site box has no template for the extra site and OpenMM fails with an opaque
#: "No template found for residue (HOH) ... contains extra sites".
_WATER_SITES = {
    "tip3p": 3, "tip3pfb": 3, "opc3": 3, "spce": 3,
    "tip4pew": 4, "tip4pfb": 4, "opc": 4,
    "tip5p": 5, "swm4ndp": 5,
}


def water_model_for_forcefield(water_xml) -> Optional[str]:
    """The water model an OpenMM water force-field resource parameterises, or None if unknown.

    `amber19/opc.xml` -> `opc`. Returning None rather than guessing keeps an unrecognised resource
    from being reconciled against a model it may not actually parameterise.
    """
    if not water_xml:
        return None
    stem = str(water_xml).replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".xml"):
        stem = stem[:-4]
    stem = stem.lower()
    return stem if stem in _WATER_SITES else None


def reconcile_water_model(water_xml, water_model) -> tuple[str, Optional[dict]]:
    """Make the packing model agree with the force field about how many sites water has.

    `forcefield.water` and `solvation.water_model` are separate settings and either can be set
    without the other -- a manifest that names a water force field but no packing model leaves the
    packing model at its default, and since the default moved to OPC (4-site) that silently paired
    a 3-site force field with a 4-site box.

    The force field is the authority: it assigns the parameters, so it decides which model is being
    simulated. When the two disagree on site count the packing model follows the force field, and
    the substitution is returned so it lands in the recorded provenance rather than happening
    quietly. Models of the same site count are left alone -- `tip3pfb` parameters packed from a
    `tip3p` box is the documented, correct arrangement.
    """
    implied = water_model_for_forcefield(water_xml)
    asked = str(water_model).lower() if water_model else None
    if implied is None or asked is None or asked not in _WATER_SITES:
        return water_model, None
    if _WATER_SITES[implied] == _WATER_SITES[asked]:
        return water_model, None
    note = {
        "requested_water_model": water_model,
        "resolved_water_model": implied,
        "water_forcefield": str(water_xml),
        "reason": (
            f"solvation.water_model={water_model!r} is {_WATER_SITES[asked]}-site but "
            f"forcefield.water={water_xml!r} parameterises {implied!r}, which is "
            f"{_WATER_SITES[implied]}-site. The force field assigns the parameters, so it decides "
            f"the model; packing follows it."
        ),
    }
    return implied, note


def resolve_packing_model(water_model: str) -> tuple[str, bool]:
    """Return `(model addSolvent can build, whether a stand-in box was substituted)`.

    The returned model chooses only the starting COORDINATES. The force field named in the
    configuration is what assigns parameters, so the simulated model is the one that was asked for.
    """
    model = str(water_model)
    if model in _NATIVE_PACKING_MODELS:
        return model, False
    if model in _PACKING_MODEL:
        return _PACKING_MODEL[model], True
    raise ValueError(
        f"water model {model!r} has no pre-equilibrated box in OpenMM and no same-topology "
        f"stand-in is declared for it. Known directly: {sorted(_NATIVE_PACKING_MODELS)}; "
        f"known by substitution: {sorted(_PACKING_MODEL)}. Add an explicit entry rather than "
        "letting a different water model be packed silently."
    )


#: `Modeller.addSolvent` chooses WHICH water molecules to replace with ions using Python's global
#: `random`, and neither OpenMM 8.5.2 nor 8.6.0 exposes a `randomSeed` parameter to control it
#: (`addSolvent` still takes no such argument in 8.6.0). Unseeded, two builds
#: from byte-identical configuration produce different bundles: the same water count and the same
#: Hamiltonian in form, but different coordinates and different ion sites, so `system.xml` and
#: `topology.pdb` never match. That silently undercuts every bundle hash, relocation check and
#: provenance comparison in this package -- "which value did I choose?" cannot be answered by
#: comparing two bundles if rebuilding the same one gives a different answer each time.
#:
#: Seeding the global module is the only lever available, so it is taken deliberately and put back:
#: the previous state is restored on the way out, so this changes solvation and nothing else.
@contextlib.contextmanager
def _seeded_global_random(seed: int):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


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
    water_model, water_note = reconcile_water_model(
        (cfg.get("forcefield") or {}).get("water"), scfg["water_model"])
    if water_note is not None:
        print(f"  solvation: {water_note['reason']}", flush=True)
    packing_model, substituted = resolve_packing_model(water_model)
    # Derived from the run's master seed like every other stream, so solvation is part of the
    # seed map rather than an unrecorded source of variation.
    from .seeds import DEFAULT_MASTER_SEED, derive_seed

    master = (cfg.get("run") or {}).get("seed")
    solvation_seed = derive_seed(int(master if master is not None else DEFAULT_MASTER_SEED),
                                 "structure/solvation")
    with _seeded_global_random(solvation_seed):
        modeller.addSolvent(
            forcefield,
            model=packing_model,
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
        # The model that was SIMULATED (assigned by the force field) and the model whose
        # pre-equilibrated box supplied the starting coordinates. When these differ the
        # substitution is recorded rather than left for a reader to infer.
        "solvation_seed": solvation_seed,
        "water_model": water_model,
        "water_model_requested": scfg["water_model"],
        "water_model_reconciled": water_note,
        "water_packing_model": packing_model,
        "water_packing_substituted": substituted,
        "box_shape": scfg["box_shape"],
        "padding_nm": float(scfg["padding_nm"]),
        "box_vectors_nm": [[float(x) for x in v] for v in box],
        "box_volume_nm3": volume_nm3,
        # Added salt and neutralising counterions are separated and named; see salt_accounting.
        # This field used to be min(ion counts) * 55.5 / n_water, which reported a salt
        # concentration for a box whose only ions were neutralising counterions.
        "salt": salt_accounting(
            ions,
            n_water=n_water,
            volume_nm3=volume_nm3,
            solute_formal_charge=(cfg.get("system") or {}).get("expected_formal_charge"),
            positive_ion=scfg.get("positive_ion"),
            negative_ion=scfg.get("negative_ion"),
            requested_molar=scfg.get("ionic_strength_molar"),
        ),
        "water_model_template": water_model,
        "forcefield": ff_info,
    }
    (out_dir / "solvation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, "step3_solvate", cfg, {"result": info})
    return info




# ---------------------------------------------------------------------------------------------
# Salt accounting
#
# Neutralising counterions are NOT salt. A solvated box of a -1 solute at zero requested salt
# contains one Na+ and no Cl-; calling that "0.15 M ionic strength" because some ion exists, or
# because half the ions were assumed to be pairs, misdescribes the system. The two quantities are
# reported separately and named for what they are.
# ---------------------------------------------------------------------------------------------

#: Ions OpenMM's `Modeller.addSolvent` can place, by element symbol -> valence. All monovalent.
MONOVALENT_IONS = {"NA": +1, "K": +1, "LI": +1, "RB": +1, "CS": +1,
                   "CL": -1, "BR": -1, "F": -1, "I": -1}

#: Recognised but unsupported here: the monovalent salt-pair formula does not describe them, and
#: guessing is worse than refusing. Listed so the error can say what was found.
MULTIVALENT_IONS = {"MG": +2, "CA": +2, "ZN": +2}

AVOGADRO = 6.02214076e23
#: Molarity of pure water, for the per-water convention reported alongside the volume-based one.
WATER_MOLAR = 55.5


def salt_accounting(ion_counts: dict, *, n_water: int, volume_nm3: float,
                    solute_formal_charge: Optional[int], positive_ion: Optional[str],
                    negative_ion: Optional[str], requested_molar: Optional[float]) -> dict:
    """Separate added salt from neutralising counterions, and report both.

    `ion_counts` maps element symbol (any case) to count. A species that is absent is zero, not
    missing: `min()` over "whatever happens to be there" is what made a counterion-only box report
    a salt concentration.

    Added salt pairs are `min(n_positive, n_negative)`. Whatever is left over is attributed to
    charge neutralisation, and its signed charge is checked against the solute's declared formal
    charge -- if those disagree, the box is not what the manifest says it is.
    """
    counts = {str(k).upper().strip("+-0123456789"): int(v) for k, v in (ion_counts or {}).items()}
    unsupported = {k: v for k, v in counts.items() if k in MULTIVALENT_IONS and v}
    if unsupported:
        raise ValueError(
            f"multivalent ions present ({unsupported}); this template's salt accounting is "
            f"monovalent only. Applying the monovalent formula would misreport ionic strength, so "
            f"it refuses instead. Supported: {sorted(MONOVALENT_IONS)}."
        )
    unknown = {k: v for k, v in counts.items() if k not in MONOVALENT_IONS and v}
    if unknown:
        raise ValueError(
            f"unrecognised ion species {sorted(unknown)}; valence is unknown so ionic strength "
            f"cannot be computed. Supported: {sorted(MONOVALENT_IONS)}."
        )

    n_pos = sum(v for k, v in counts.items() if MONOVALENT_IONS.get(k, 0) > 0)
    n_neg = sum(v for k, v in counts.items() if MONOVALENT_IONS.get(k, 0) < 0)
    n_salt_pairs = min(n_pos, n_neg)
    n_neutralising = abs(n_pos - n_neg)
    neutralising_charge = (n_pos - n_neg)          # signed, in elementary charges

    volume_l = float(volume_nm3) * 1e-24 if volume_nm3 else 0.0
    salt_molar = (n_salt_pairs / AVOGADRO / volume_l) if volume_l > 0 else 0.0
    # I = 0.5 * sum(c_i z_i^2); for a monovalent salt pair this equals the salt-pair molarity, but
    # it is computed from the valences rather than assumed, so multivalent support is a data change
    ionic_strength = 0.0
    if volume_l > 0:
        for sym, n in counts.items():
            z = MONOVALENT_IONS.get(sym, 0)
            ionic_strength += (n / AVOGADRO / volume_l) * z * z
        ionic_strength *= 0.5

    balanced = None
    if solute_formal_charge is not None:
        balanced = (neutralising_charge + int(solute_formal_charge)) == 0

    return {
        "requested_salt_molar": (float(requested_molar) if requested_molar is not None else None),
        "realized_salt_pair_molar": round(salt_molar, 6),
        "ionic_strength_molar": round(ionic_strength, 6),
        "n_salt_pairs": int(n_salt_pairs),
        "n_neutralizing_ions": int(n_neutralising),
        "neutralizing_charge_e": int(neutralising_charge),
        "solute_formal_charge": (int(solute_formal_charge)
                                 if solute_formal_charge is not None else None),
        "charge_balanced": balanced,
        "n_positive_ions": int(n_pos),
        "n_negative_ions": int(n_neg),
        "n_ions_total": int(n_pos + n_neg),
        "ion_counts": {k: int(v) for k, v in counts.items() if v},
        "positive_ion_requested": positive_ion,
        "negative_ion_requested": negative_ion,
        "n_water_molecules": int(n_water),
        "box_volume_nm3": (round(float(volume_nm3), 6) if volume_nm3 else 0.0),
        "salt_pairs_per_55p5_mol_water": (
            round(n_salt_pairs * WATER_MOLAR / n_water, 6) if n_water else 0.0
        ),
        "note": "realized_salt_pair_molar counts ADDED salt only: min(n_positive, n_negative) over "
                "the box volume. Excess ions are neutralising counterions and are reported "
                "separately; they are not salt and are not ionic strength.",
    }
