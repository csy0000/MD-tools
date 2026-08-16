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

from openmm import (CMAPTorsionForce, CustomGBForce, NonbondedForce,
                    PeriodicTorsionForce, XmlSerializer)

from .config import write_manifest

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Step 1 -- initial structure from SMILES (ETKDGv3 + MMFF)
# ---------------------------------------------------------------------------------------------
def initial_structure(smiles: str, out_dir: Path, cfg: dict) -> dict:
    """Embed *smiles* with ETKDGv3, MMFF-minimise, and write the lowest-energy conformer.

    Writes ``solute.sdf`` (bond orders and formal charges, which the OpenFF path needs) and
    ``solute.pdb`` (what Modeller reads).  Every conformer's energy and convergence flag is
    recorded, so "the minimisation converged" is a value in a file rather than an assumption.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    ecfg, mcfg = cfg["structure"]["etkdg"], cfg["structure"]["mmff"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    if ecfg["version"] != "ETKDGv3":
        raise ValueError(f"structure.etkdg.version must be 'ETKDGv3', got {ecfg['version']!r}")
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = int(ecfg["seed"])
    params.useRandomCoords = bool(ecfg["use_random_coords"])
    params.pruneRmsThresh = float(ecfg["prune_rms_thresh"])
    params.numThreads = int(ecfg["num_threads"])
    conf_ids = list(rdDistGeom.EmbedMultipleConfs(mol, int(ecfg["n_conformers"]), params))
    if not conf_ids:
        raise RuntimeError(f"ETKDGv3 produced no conformers for {smiles!r}")

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mcfg["variant"])
    if props is None:
        raise RuntimeError(
            f"MMFF ({mcfg['variant']}) has no parameters for this molecule; it cannot be "
            "minimised with MMFF.  Supply a --pdb structure instead."
        )
    records = []
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
        if ff is None:
            raise RuntimeError(f"could not build the MMFF force field for conformer {cid}")
        converged = ff.Minimize(
            maxIts=int(mcfg["max_iterations"]),
            energyTol=float(mcfg["energy_tolerance"]),
            forceTol=float(mcfg["force_tolerance"]),
        )
        records.append(
            {
                "conformer_id": int(cid),
                "mmff_energy_kcal_mol": float(ff.CalcEnergy()),
                # RDKit returns 0 on convergence, 1 if the iteration cap was hit first
                "converged": converged == 0,
                "max_iterations": int(mcfg["max_iterations"]),
            }
        )
    records.sort(key=lambda r: r["mmff_energy_kcal_mol"])
    best = records[0]

    keep = Chem.Mol(mol)
    keep.RemoveAllConformers()
    keep.AddConformer(mol.GetConformer(best["conformer_id"]), assignId=True)
    Chem.MolToMolFile(keep, str(out_dir / "solute.sdf"))
    Chem.MolToPDBFile(keep, str(out_dir / "solute.pdb"))

    info = {
        "smiles": smiles,
        "n_conformers_embedded": len(conf_ids),
        "mmff_variant": mcfg["variant"],
        "selected_conformer": best,
        "all_conformers": records,
        "n_unconverged": sum(1 for r in records if not r["converged"]),
        "solute_sdf": str(out_dir / "solute.sdf"),
        "solute_pdb": str(out_dir / "solute.pdb"),
    }
    (out_dir / "initial_structure.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, "step1_initial_structure", cfg, {"result": info})
    return info


# ---------------------------------------------------------------------------------------------
# Force field construction (shared by steps 2, 3, 4)
# ---------------------------------------------------------------------------------------------
def build_forcefield(cfg: dict, ligand_sdf: Optional[Path] = None,
                    route: Optional[str] = None):
    """Return ``(ForceField, info)`` for the baseline: ff19SB + TIP3P-FB (+ Sage for a ligand).

    A ligand is parameterised through :class:`openmmforcefields.generators.SMIRNOFFTemplateGenerator`,
    which registers a residue template on the fly.  ``ligand_charge_method='am1bcc'`` runs
    AmberTools' ``sqm`` -- if the AmberTools binaries are not on ``PATH`` the OpenFF toolkit
    registry silently lacks its wrapper and AM1BCC becomes unavailable, so this checks and fails
    loudly instead.
    """
    from openmm import app

    ff_cfg = cfg["forcefield"]
    # On the LIGAND route the protein force field is not loaded at all.  It parameterises nothing
    # here -- a SMILES-built solute is one UNL residue handled by SMIRNOFF, and the water XML
    # already carries the Na+/Cl-/HOH templates addSolvent needs -- but leaving it in the list put
    # "ff19SB" into the manifest of a Sage calculation, which misdescribes the Hamiltonian that
    # actually ran.  Route selection therefore overrides the legacy protein-force-field default.
    protein_xml = None if route == "ligand" else ff_cfg["protein"]
    xmls = ([protein_xml] if protein_xml else []) + [ff_cfg["water"], *ff_cfg["extra_xml"]]
    info: dict[str, Any] = {
        "xml": list(xmls),
        "route": route,
        "protein_forcefield": protein_xml,
        "water": ff_cfg["water"],
        "ligand": None,
    }
    forcefield = app.ForceField(*xmls)

    if ligand_sdf is not None:
        from openff.toolkit import Molecule
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        method = str(ff_cfg["ligand_charge_method"]).lower()
        wrappers = [t.__class__.__name__ for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]
        if method == "am1bcc" and "AmberToolsToolkitWrapper" not in wrappers:
            raise RuntimeError(
                "forcefield.ligand_charge_method='am1bcc' needs AmberTools' sqm, but the OpenFF "
                f"toolkit registry only has {wrappers}.  Activate the md-templates environment "
                "(conda activate md-templates) so antechamber/sqm are on PATH, or set "
                "forcefield.ligand_charge_method to 'nagl' / 'espaloma' if that is intended.  "
                "This is checked here because otherwise the charges would silently fall back to "
                "a different method and the run would be mislabelled."
            )
        offmol = Molecule.from_file(str(ligand_sdf))
        if method == "am1bcc":
            offmol.assign_partial_charges("am1bccelf10" if _has_openeye() else "am1bcc")
        elif method == "nagl":
            offmol.assign_partial_charges("openff-gnn-am1bcc-0.1.0-rc.3.pt")
        else:
            raise ValueError(f"unsupported ligand_charge_method {method!r}")
        generator = SMIRNOFFTemplateGenerator(
            molecules=[offmol], forcefield=ff_cfg["ligand"]
        )
        forcefield.registerTemplateGenerator(generator.generator)
        info["ligand"] = {
            "forcefield": ff_cfg["ligand"],
            "charge_method": method,
            "net_charge_e": float(sum(c.m for c in offmol.partial_charges)),
            "formal_charge": int(round(sum(a.formal_charge.m for a in offmol.atoms))),
            "n_atoms": offmol.n_atoms,
        }
    return forcefield, info


def _has_openeye() -> bool:
    try:
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

        return any(
            "OpenEye" in t.__class__.__name__
            for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------
# Step 2 -- protonation at pH
# ---------------------------------------------------------------------------------------------
def protonate(pdb_in: Path, out_dir: Path, cfg: dict, ligand_sdf: Optional[Path] = None,
              input_route: Optional[str] = None) -> dict:
    """Delete existing hydrogens and re-add them with ``Modeller.addHydrogens(pH=...)``.

    Deleting first is deliberate: ``addHydrogens`` only *adds* what is missing, so a structure that
    already carries hydrogens in the wrong protonation state keeps them.  Deleting makes the
    hydrogen set a function of the pH and the force field, not of whatever wrote the input file.
    """
    from openmm import app
    from openmm.app import element as elem

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["protonation"]

    pdb = app.PDBFile(str(pdb_in))
    route = resolve_route(cfg, app.PDBFile(str(pdb_in)).topology, input_route=input_route)
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf, route=route)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    n_h_before = sum(1 for a in modeller.topology.atoms() if a.element == elem.hydrogen)
    is_ligand_only = route == "ligand"

    if is_ligand_only and pcfg["skip_for_ligand"]:
        added: list = []
        note = (
            "skipped -- addHydrogens(pH) only applies to SUPPORTED RESIDUE TEMPLATES.  It chooses "
            "a protonation variant per standard residue from the force field's template set; it "
            "has no rules for a ligand and would not titrate one.  The ligand route therefore "
            "uses the protomer/tautomer and formal charges EXPLICITLY ENCODED IN THE SMILES, "
            "which is the only place that information exists.  If a different protonation state "
            "is wanted at pH 7, write it into the SMILES."
        )
    else:
        if pcfg["delete_existing_hydrogens"]:
            modeller.delete(
                [a for a in modeller.topology.atoms() if a.element == elem.hydrogen]
            )
        added = modeller.addHydrogens(
            forcefield, pH=float(pcfg["ph"]), variants=pcfg["variants"]
        )
        note = f"addHydrogens at pH {pcfg['ph']}"

    out_pdb = out_dir / "solute_h.pdb"
    with out_pdb.open("w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)

    n_h_after = sum(1 for a in modeller.topology.atoms() if a.element == elem.hydrogen)
    info = {
        "input_pdb": str(pdb_in),
        "output_pdb": str(out_pdb),
        "route": route,
        "input_route": input_route,
        "ph": float(pcfg["ph"]),
        "ph_applies": route != "ligand",
        "note": note,
        "n_hydrogens_before": n_h_before,
        "n_hydrogens_after": n_h_after,
        "n_atoms": modeller.topology.getNumAtoms(),
        # which variant addHydrogens chose for each residue -- the record of what pH 7 meant here
        "variants": [None if v is None else str(v) for v in added],
        "forcefield": ff_info,
    }
    (out_dir / "protonation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, "step2_protonate", cfg, {"result": info})
    return info


PROTEIN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME", "NHE", "NMA",
})


def resolve_route(cfg: dict, topology=None, *, input_route: Optional[str] = None) -> str:
    """Decide the force-field route: ``peptide`` (ff19SB) or ``ligand`` (SMIRNOFF/Sage).

    **The route follows the input actually supplied**, not a system preset.  ff19SB assigns
    parameters per residue TEMPLATE, so it can only see a peptide in a residue-named PDB; an RDKit
    structure built from SMILES is a single ``UNL`` residue and must go through SMIRNOFF.  A preset
    that pinned ``solute_kind: peptide`` while the user passed ``--smiles`` used to fail deep inside
    template matching with "No template found for residue 0 (UNL)"; incompatible combinations are
    now rejected up front, before any hydrogen is deleted or any force field is built.

    ``system.solute_kind`` overrides the decision when it is not ``"auto"``, but an override that
    contradicts the input route is an error rather than a silent reinterpretation.
    """
    required = cfg["system"].get("require_input_route")
    if required and input_route is not None and input_route != required:
        raise ValueError(
            f"this configuration requires the '{required}' input route but was invoked with "
            f"'{input_route}'.  For the cyclo_rgdfv preset that means --smiles: it is a Sage 2.2 / "
            "AM1-BCC calculation and a PDB invocation would silently become a different Hamiltonian."
        )
    declared = str(cfg["system"]["solute_kind"])
    inferred = None
    if input_route == "smiles":
        inferred = "ligand"
    elif topology is not None:
        names = {r.name.upper() for r in topology.residues()}
        solute = names - WATER_RESIDUE_NAMES - ION_RESIDUE_NAMES
        if solute & PROTEIN_RESIDUES:
            inferred = "complex" if solute - PROTEIN_RESIDUES else "peptide"
        else:
            inferred = "ligand"

    if declared == "auto":
        if inferred is None:
            raise ValueError(
                "system.solute_kind is 'auto' but neither an input route nor a topology was "
                "given, so the force-field route cannot be inferred."
            )
        return inferred

    if inferred is not None and declared != inferred:
        if input_route == "smiles" and declared in ("peptide", "complex"):
            raise ValueError(
                f"system.solute_kind='{declared}' contradicts --smiles input.  ff19SB matches by "
                "residue template and an RDKit structure built from SMILES is a single 'UNL' "
                "residue, so the peptide route cannot succeed here.  Supply a residue-named PDB "
                "with --pdb for the ff19SB route, or set system.solute_kind to 'auto'/'ligand'."
            )
        print(
            f"[route] system.solute_kind='{declared}' overrides the inferred '{inferred}'; "
            "proceeding as declared",
            flush=True,
        )
    return declared


def _classify_solute(topology, cfg: dict) -> str:
    """Return ``peptide`` / ``ligand`` / ``complex`` from the residue names.

    The decision is made on residue names because that is what the force fields key off:
    ``ff19SB`` matches per residue, so a solute whose residues it recognises is a peptide, and
    anything else must go through SMIRNOFF (Sage).  A structure built by RDKit from SMILES is one
    residue called ``UNL`` and is therefore always a ligand -- which is why a *peptide* has to be
    supplied as a residue-named PDB, not as a SMILES string.
    """
    kind = cfg["system"]["solute_kind"]
    if kind != "auto":
        return kind
    names = {r.name.upper() for r in topology.residues()}
    protein_like = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "ACE", "NME", "NHE",
    }
    if names & protein_like:
        return "complex" if names - protein_like - WATER_RESIDUE_NAMES else "peptide"
    return "ligand"


# ---------------------------------------------------------------------------------------------
# Step 4 -- System, hydrogen mass repartitioning
# ---------------------------------------------------------------------------------------------
def repartition_hydrogen_mass(
    system, topology, target_h_mass_amu: float, solute_atoms: Optional[Iterable[int]] = None
) -> dict:
    """Move mass from heavy atoms onto their bonded hydrogens (HMR), in place.

    ``m_H -> target``, and the heavy partner loses exactly what the hydrogen gained, so the total
    mass is conserved and the centre-of-mass dynamics are untouched.  This is what buys the 4 fs
    timestep: it lowers the frequency of the bond-angle motions involving hydrogen, which are the
    fastest unconstrained degrees of freedom left once ``constraints=HBonds`` has removed the bond
    stretches.

    **Water is never repartitioned.**  Rigid water is fully constrained, so its hydrogen masses do
    not limit the timestep, and changing them would change water's rotational dynamics (and so its
    diffusion constant and dielectric relaxation) for no benefit.
    """
    from openmm import unit
    from openmm.app import element as elem

    solute = None if solute_atoms is None else {int(i) for i in solute_atoms}
    total_before = sum(
        system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())
    )
    changed = 0
    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.element == elem.hydrogen and b.element != elem.hydrogen:
            h, heavy = a, b
        elif b.element == elem.hydrogen and a.element != elem.hydrogen:
            h, heavy = b, a
        else:
            continue
        if h.residue.name.upper() in WATER_RESIDUE_NAMES:
            continue
        if solute is not None and (h.index not in solute or heavy.index not in solute):
            continue
        m_h = system.getParticleMass(h.index).value_in_unit(unit.amu)
        if m_h <= 0.0:  # a virtual site or an already-massless particle
            continue
        delta = float(target_h_mass_amu) - m_h
        m_heavy = system.getParticleMass(heavy.index).value_in_unit(unit.amu)
        if m_heavy - delta <= 1.0:
            raise ValueError(
                f"repartitioning {delta:.3f} amu onto H {h.index} would leave heavy atom "
                f"{heavy.index} with {m_heavy - delta:.3f} amu.  Lower "
                "system_build.hydrogen_mass_amu."
            )
        system.setParticleMass(h.index, target_h_mass_amu * unit.amu)
        system.setParticleMass(heavy.index, (m_heavy - delta) * unit.amu)
        changed += 1
    total_after = sum(
        system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())
    )
    if not math.isclose(total_before, total_after, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(
            f"HMR changed the total mass: {total_before:.6f} -> {total_after:.6f} amu"
        )
    return {
        "target_hydrogen_mass_amu": float(target_h_mass_amu),
        "n_hydrogens_repartitioned": changed,
        "total_mass_amu": round(total_after, 6),
        "scope": "solute" if solute is not None else "all-non-water",
    }


def _amide_candidates(topology, solute: set[int]) -> list[dict]:
    """Every solute C-N bond whose carbon carries a carbonyl oxygen, with the evidence found.

    A topology has no bond orders, so "carbonyl" is established structurally: an oxygen bonded to
    the carbon and to nothing else heavy, and carrying no hydrogen.  That distinguishes C=O from
    the C-OH of a carboxylic acid and from an ester's bridging oxygen.  Anything that does not fit
    cleanly is returned with ``ambiguous`` set rather than being quietly treated as an amide --
    "a carbon next to one oxygen and one nitrogen" also describes carbamates, ureas and carbamic
    acids, and those are not peptide omega bonds.
    """
    from openmm.app import element as elem

    neigh: dict[int, list] = {}
    for bond in topology.bonds():
        neigh.setdefault(bond.atom1.index, []).append(bond.atom2)
        neigh.setdefault(bond.atom2.index, []).append(bond.atom1)

    out = []
    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.index not in solute or b.index not in solute:
            continue
        if {a.element, b.element} != {elem.carbon, elem.nitrogen}:
            continue
        c, n = (a, b) if a.element == elem.carbon else (b, a)
        oxy = [o for o in neigh.get(c.index, []) if o.element == elem.oxygen]
        if not oxy:
            continue                                     # not an amide at all; not a candidate
        carbonyl, hydroxyl = [], []
        for o in oxy:
            others = [x for x in neigh.get(o.index, []) if x.index != c.index]
            (hydroxyl if others else carbonyl).append(o)
        nitrogens = [x for x in neigh.get(c.index, []) if x.element == elem.nitrogen]
        reason = None
        if len(carbonyl) != 1:
            reason = f"carbon {c.index} has {len(carbonyl)} carbonyl oxygens (expected 1)"
        elif hydroxyl:
            reason = f"carbon {c.index} also carries a hydroxyl/ester oxygen: not a plain amide"
        elif len(nitrogens) > 1:
            reason = f"carbon {c.index} is bonded to {len(nitrogens)} nitrogens (urea-like)"
        out.append({
            "bond": (int(c.index), int(n.index)),
            "carbon": int(c.index), "nitrogen": int(n.index),
            "carbon_residue": c.residue.name, "nitrogen_residue": n.residue.name,
            "inter_residue": c.residue.index != n.residue.index,
            "ambiguous": reason,
        })
    return out


def classify_omega_bonds(topology, solute_atoms: Iterable[int], *, route: str = "peptide",
                         ligand_sdf: Optional[Path] = None,
                         proline_like_residues: Iterable[str] = ("PRO",),
                         max_proline_ring_size: int = 7) -> dict:
    """Split the solute's amide C-N bonds into REST2-unscaled, proline-like-scaled, unclassified.

    REST2 here is *omega-selective*: an ORDINARY amide omega torsion is left unscaled, because the
    REMD ladder leaves it unscaled and a hot rung that isomerises cis/trans samples states the
    reference never does.  A **proline-like** peptide bond is the exception and stays ELIGIBLE for
    normal scaling: its nitrogen is locked into a small ring, so the torsion is not the near-planar
    two-state coordinate the exclusion exists to protect.

    Two routes, both auditable:

    * ``peptide`` -- residue-aware.  Proline-like character is read from the residue containing the
      amide NITROGEN, against a configurable name set (``PRO`` by default).  An X-PRO peptide bond
      is therefore *not* excluded.
    * ``ligand`` -- bond-order aware, from the retained SDF, because a SMILES-built solute is one
      ``UNL`` residue and has no residue evidence at all.  Ordinary amides are matched with
      ``[CX3](=[OX1])[NX3]``; proline-like nitrogens with ``[NX3;R]`` restricted to rings of at most
      *max_proline_ring_size* atoms.  **The ring-size bound is what makes this correct for
      macrocycles**: every backbone nitrogen of a cyclic peptide is "in a ring", but a 15-30
      membered macrocycle does not constrain the amide the way a pyrrolidine does, so an unbounded
      ``;R`` test would wrongly free every macrocyclic omega for scaling.

    N-methylated amides are ordinary amides under both routes: an N-methyl nitrogen is neither
    proline-like nor ring-locked, and those bonds isomerise readily, so they must stay unscaled.

    Returns a dict with ``omega_unscaled_bonds``, ``omega_proline_like_scaled_bonds``,
    ``omega_unclassified_candidates`` and ``omega_detection_method``.  **A non-empty unclassified
    list must block production** -- it means a candidate was found that neither rule could name, and
    guessing would silently change the Hamiltonian.
    """
    solute = {int(i) for i in solute_atoms}
    candidates = _amide_candidates(topology, solute)
    pro_names = {str(x).upper() for x in proline_like_residues}

    unscaled, proline, unknown = [], [], []
    if route == "ligand":
        ring_info = _ligand_ring_nitrogens(ligand_sdf, topology, solute, max_proline_ring_size)
        method = (f"ligand/RDKit SMARTS [CX3](=[OX1])[NX3]; proline-like = amide N in a ring of "
                  f"<= {max_proline_ring_size} atoms; SDF {Path(ligand_sdf).name}")
        for cand in candidates:
            if cand["ambiguous"]:
                unknown.append(cand); continue
            if cand["bond"] not in ring_info["amide_bonds"]:
                cand = dict(cand, ambiguous="RDKit found no ordinary-amide match for this C-N bond")
                unknown.append(cand); continue
            if cand["nitrogen"] in ring_info["small_ring_nitrogens"]:
                cand = dict(cand, ring_sizes=ring_info["ring_sizes"].get(cand["nitrogen"], []))
                proline.append(cand)
            else:
                unscaled.append(cand)
    else:
        method = (f"peptide/residue-aware; proline-like residue names = {sorted(pro_names)} "
                  "applied to the residue containing the amide NITROGEN")
        for cand in candidates:
            if cand["ambiguous"]:
                unknown.append(cand); continue
            if cand["nitrogen_residue"].upper() in pro_names:
                proline.append(cand)
            elif cand["nitrogen_residue"].upper() in PROTEIN_RESIDUES:
                unscaled.append(cand)
            else:
                # An unrecognised residue is NOT assumed to be an ordinary amide.  HYP, and any
                # other proline-like or non-standard residue, would otherwise be silently excluded
                # from scaling on the strength of nothing but "it is a peptide bond".  Blocking
                # forces the name into rest2.proline_like_residues, or into review.
                unknown.append(dict(cand, ambiguous=(
                    f"nitrogen residue '{cand['nitrogen_residue']}' is neither a known protein "
                    "residue nor listed in rest2.proline_like_residues, so the peptide route "
                    "cannot say whether this omega is ordinary or proline-like")))
    return {
        "omega_unscaled_bonds": [c["bond"] for c in unscaled],
        "omega_proline_like_scaled_bonds": [c["bond"] for c in proline],
        "omega_unclassified_candidates": unknown,
        "omega_detection_method": method,
        "omega_detail": {"unscaled": unscaled, "proline_like_scaled": proline},
    }


def _ligand_ring_nitrogens(ligand_sdf, topology, solute: set[int], max_ring: int) -> dict:
    """RDKit amide perception on the retained SDF, mapped onto OpenMM indices.

    The mapping is ASSERTED, never assumed: element sequence and the full bond graph must agree
    between the SDF and the solute part of the topology.  They coincide today because the SDF and
    the PDB are written from the same RDKit molecule in the same atom order, but that is a property
    of the pipeline, not a guarantee -- and a silent off-by-one here would scale the wrong torsions.
    """
    from rdkit import Chem

    if ligand_sdf is None:
        raise ValueError(
            "the ligand omega route needs the SDF written by simbox-setup.py (bond orders are "
            "not recoverable from a topology), but none was supplied"
        )
    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {ligand_sdf}")

    atoms = [a for a in topology.atoms() if a.index in solute]
    atoms.sort(key=lambda a: a.index)
    if mol.GetNumAtoms() != len(atoms):
        raise ValueError(
            f"atom-count mismatch: SDF has {mol.GetNumAtoms()}, the topology's solute has "
            f"{len(atoms)}.  The RDKit->OpenMM mapping cannot be established."
        )
    offset = atoms[0].index
    for i, atom in enumerate(atoms):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if atom.element is None or atom.element.symbol != sym:
            raise ValueError(
                f"element mismatch at solute index {i}: SDF says {sym}, topology says "
                f"{None if atom.element is None else atom.element.symbol}.  Refusing to guess a "
                "mapping between the SDF and the topology."
            )
    rd_bonds = {frozenset((b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset))
                for b in mol.GetBonds()}
    top_bonds = {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()
                 if b.atom1.index in solute and b.atom2.index in solute}
    if rd_bonds != top_bonds:
        raise ValueError(
            f"bond-graph mismatch between the SDF and the topology "
            f"({len(rd_bonds ^ top_bonds)} differing bonds).  Refusing to guess a mapping."
        )

    amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
    amide_bonds, small_ring_n, ring_sizes = set(), set(), {}
    ri = mol.GetRingInfo()
    for c_i, _o_i, n_i in mol.GetSubstructMatches(amide):
        amide_bonds.add((c_i + offset, n_i + offset))
        sizes = sorted(len(r) for r in ri.AtomRings() if n_i in r)
        if sizes:
            ring_sizes[n_i + offset] = sizes
            if min(sizes) <= max_ring:
                small_ring_n.add(n_i + offset)
    return {"amide_bonds": amide_bonds, "small_ring_nitrogens": small_ring_n,
            "ring_sizes": ring_sizes}


def omega_central_bonds(topology, solute_atoms: Iterable[int]) -> list[tuple[int, int]]:
    """DEPRECATED structural detector: every amide C-N bond, with no proline-like exception.

    Kept so pre-2026-08-14 bundles can be re-derived.  It treats an X-PRO peptide bond as an
    ordinary omega and excludes it from scaling, which
    :func:`classify_omega_bonds` deliberately does not.  New code must use that function.
    """
    solute = {int(i) for i in solute_atoms}
    return sorted({c["bond"] for c in _amide_candidates(topology, solute) if not c["ambiguous"]})


def build_system(solvated_pdb: Path, out_dir: Path, cfg: dict, n_solute_atoms: int,
                 ligand_sdf: Optional[Path] = None, route: str = "peptide") -> dict:
    """Create the OpenMM ``System`` (PME, 1.0 nm, HBonds, HMR) and serialise it to XML."""
    from openmm import XmlSerializer, app, unit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcfg = cfg["system_build"]

    pdb = app.PDBFile(str(solvated_pdb))
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf, route=route)

    method = {"PME": app.PME, "LJPME": app.LJPME, "CutoffPeriodic": app.CutoffPeriodic}[
        bcfg["nonbonded_method"]
    ]
    constraints = {
        "HBonds": app.HBonds, "AllBonds": app.AllBonds, "HAngles": app.HAngles, "None": None,
    }[str(bcfg["constraints"])]

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        nonbondedCutoff=float(bcfg["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=constraints,
        rigidWater=bool(bcfg["rigid_water"]),
        removeCMMotion=bool(bcfg["remove_cm_motion"]),
        ewaldErrorTolerance=float(bcfg["ewald_error_tolerance"]),
    )

    from openmm import NonbondedForce

    nb_info = {}
    for force in (system.getForce(i) for i in range(system.getNumForces())):
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(bool(bcfg["use_dispersion_correction"]))
            if bcfg["switch_distance_nm"] is not None:
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(
                    float(bcfg["switch_distance_nm"]) * unit.nanometer
                )
            nb_info = {
                "method": force.getNonbondedMethod(),
                "cutoff_nm": force.getCutoffDistance().value_in_unit(unit.nanometer),
                "switching": force.getUseSwitchingFunction(),
                "dispersion_correction": force.getUseDispersionCorrection(),
                "ewald_error_tolerance": force.getEwaldErrorTolerance(),
            }

    scope = None if bcfg["hmr_scope"] == "all" else range(n_solute_atoms)
    hmr = repartition_hydrogen_mass(
        system, pdb.topology, float(bcfg["hydrogen_mass_amu"]), scope
    )

    rcfg = cfg["rest2"]
    if rcfg["omega_selective"]:
        omega_info = classify_omega_bonds(
            pdb.topology, range(n_solute_atoms), route=route, ligand_sdf=ligand_sdf,
            proline_like_residues=rcfg["proline_like_residues"],
            max_proline_ring_size=int(rcfg["max_proline_ring_size"]),
        )
    else:
        omega_info = {
            "omega_unscaled_bonds": [], "omega_proline_like_scaled_bonds": [],
            "omega_unclassified_candidates": [],
            "omega_detection_method": "disabled (rest2.omega_selective = false): every torsion "
                                      "is scaled, including ordinary amide omegas",
            "omega_detail": {"unscaled": [], "proline_like_scaled": []},
        }
    omega = [tuple(b) for b in omega_info["omega_unscaled_bonds"]]

    (out_dir / "system.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    info = {
        "solvated_pdb": str(solvated_pdb),
        "system_xml": str(out_dir / "system.xml"),
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "n_solute_atoms": int(n_solute_atoms),
        "constraints": str(bcfg["constraints"]),
        "rigid_water": bool(bcfg["rigid_water"]),
        "nonbonded": nb_info,
        "hmr": hmr,
        "omega_central_bonds": omega,          # back-compat alias for omega_unscaled_bonds
        **{k: omega_info[k] for k in
           ("omega_unscaled_bonds", "omega_proline_like_scaled_bonds",
            "omega_unclassified_candidates", "omega_detection_method", "omega_detail")},
        "forcefield": ff_info,
        "degrees_of_freedom": (
            3 * system.getNumParticles() - system.getNumConstraints()
            - (3 if bcfg["remove_cm_motion"] else 0)
        ),
    }
    (out_dir / "system_build.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, "step4_build_system", cfg, {"result": info})
    return info




# ---------------------------------------------------------------------------------------------
# REST2 Hamiltonian scaling
#
# Solute charges scale by sqrt(s) and solute epsilons by s, which gives
#   U_s = s*U_solute + sqrt(s)*U_solute-solvent + U_solvent
# for a pairwise nonbonded force. PeriodicTorsion and CMAP (ff19SB) are handled too.
# ---------------------------------------------------------------------------------------------
def _clone_system(system):
    """Deep-copy an OpenMM System via XML serialization."""
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _scale_nonbonded_force(force: NonbondedForce, solute_atom_indices: set[int], scale_factor: float) -> None:
    sqrt_scale = math.sqrt(scale_factor)
    for atom_index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(atom_index)
        if atom_index in solute_atom_indices:
            force.setParticleParameters(atom_index, charge * sqrt_scale, sigma, epsilon * scale_factor)
    for exception_index in range(force.getNumExceptions()):
        atom_i, atom_j, charge_prod, sigma, epsilon = force.getExceptionParameters(exception_index)
        n_solute = int(atom_i in solute_atom_indices) + int(atom_j in solute_atom_indices)
        if n_solute == 2:
            force.setExceptionParameters(exception_index, atom_i, atom_j,
                                         charge_prod * scale_factor, sigma, epsilon * scale_factor)
        elif n_solute == 1:
            force.setExceptionParameters(exception_index, atom_i, atom_j,
                                         charge_prod * sqrt_scale, sigma, epsilon * sqrt_scale)


def _scale_torsion_force(force: PeriodicTorsionForce, solute_atom_indices: set[int], scale_factor: float,
                         exclude_central_bonds: set | None = None) -> None:
    exclude = exclude_central_bonds or set()
    for torsion_index in range(force.getNumTorsions()):
        atom_i, atom_j, atom_k, atom_l, periodicity, phase, k_value = force.getTorsionParameters(torsion_index)
        if all(atom in solute_atom_indices for atom in [atom_i, atom_j, atom_k, atom_l]) \
                and frozenset((int(atom_j), int(atom_k))) not in exclude:
            force.setTorsionParameters(torsion_index, atom_i, atom_j, atom_k, atom_l,
                                       periodicity, phase, k_value * scale_factor)


def _scale_cmap_force(force: CMAPTorsionForce, solute_atom_indices: set[int], scale_factor: float) -> None:
    solute_map_indices: set[int] = set()
    non_solute_map_indices: set[int] = set()
    for torsion_index in range(force.getNumTorsions()):
        map_index, a1, a2, a3, a4, b1, b2, b3, b4 = force.getTorsionParameters(torsion_index)
        atoms = [a1, a2, a3, a4, b1, b2, b3, b4]
        if all(atom in solute_atom_indices for atom in atoms):
            solute_map_indices.add(map_index)
        else:
            non_solute_map_indices.add(map_index)
    shared = solute_map_indices & non_solute_map_indices
    if shared:
        raise RuntimeError("Cannot selectively scale CMAP terms because a CMAP map is shared "
                           "between solute and non-solute torsions.")
    for map_index in solute_map_indices:
        size, energy = force.getMapParameters(map_index)
        force.setMapParameters(map_index, size, [v * scale_factor for v in energy])


def build_rest2_scaled_system(base_system, solute_atom_indices: np.ndarray, scale_factor: float,
                              exclude_central_bonds=None):
    """Return a deep copy of *base_system* with REST2 Hamiltonian scaling applied.

    Parameters
    ----------
    base_system:
        Unscaled OpenMM System (scale_factor == 1.0 corresponds to no scaling).
    solute_atom_indices:
        Integer array of solute atom indices (0-based).  For alanine dipeptide
        implicit runs this is ``np.arange(22)``.
    scale_factor:
        REST2 scale factor ``s = T_bath / T_effective``.  Use 1.0 for the
        physical (U0) system and the actual replica value for higher replicas.
    exclude_central_bonds:
        Optional iterable of {i, j} atom-index pairs whose torsion terms are left
        unscaled (e.g. the omega bond of non-proline-like amides).  Torsion-only.

    Returns
    -------
    openmm.System
        Scaled copy; the original *base_system* is not modified.
    """
    system = _clone_system(base_system)
    solute_set = {int(i) for i in solute_atom_indices}
    exclude = ({frozenset((int(a), int(b))) for a, b in exclude_central_bonds}
               if exclude_central_bonds is not None else set())
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, NonbondedForce):
            _scale_nonbonded_force(force, solute_set, scale_factor)
        elif isinstance(force, PeriodicTorsionForce):
            _scale_torsion_force(force, solute_set, scale_factor, exclude)
        elif isinstance(force, CMAPTorsionForce):
            _scale_cmap_force(force, solute_set, scale_factor)
        elif isinstance(force, CustomGBForce):
            raise ValueError(
                "CustomGBForce found: this system uses implicit solvent, which this "
                "explicit-water template does not support. Build the system with PME in "
                "a solvated box instead."
            )
    return system


