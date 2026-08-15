"""Explicit-solvent baseline setup: the production preparation and sampling path.

This module is the single implementation of the explicit-solvent baseline described in
``docs/implementation/explicit_solvent/baseline_setups.md``.  The scripts under
``docs/implementation/explicit_solvent/scripts/`` are thin CLIs over the functions here; no
scientific logic lives in them.

The pipeline is four stages, each resumable and each writing its own provenance:

    (a) build_simbox           ETKDGv3/PDB -> addHydrogens(pH 7) -> dodecahedron solvation ->
                               System with PME, HBonds and hydrogen mass repartitioning
    (b) minimize_equilibrate   staged: restrained minimisation, 50->300 K ramp, NPT with the
                               restraint released in steps, free NPT whose tail fixes the box
    (c) run_md                 one free walker at production.md.scale_factor, in chunks
    (d) run_rest2_remd         N replicas, neighbour exchange, chunked per replica

    (e) conditional BAR and partitioned REST2 are future work and are not here.

Reused from the rest of the package rather than reimplemented -- these are the parts that carry
scientific meaning and are already validated:

* :func:`escort_ais.systems.openmm_system.build_rest2_scaled_system` -- the REST2 Hamiltonian.
  It scales solute charges by ``sqrt(s)`` and solute epsilons by ``s``, which gives
  ``U_s = s*U_solute + sqrt(s)*U_solute-solvent + U_solvent`` for a pairwise nonbonded force, and
  it handles PeriodicTorsion / CMAP (ff19SB) as well.
* :func:`escort_ais.systems.topology_prep.resolve_remd_scale_ladder` -- ladder validation and the
  ``replica 0 == s = 1`` ordering convention.
* :func:`escort_ais.methods.md_run.exchange_pairs` and
  :func:`escort_ais.methods.md_run.attempt_rest2_exchange` -- the neighbour-swap schedule and the
  Metropolis criterion.  ``attempt_rest2_exchange`` already exchanges periodic box vectors, so it
  is correct for an explicit-solvent box.

Units: nm, ps, kJ/mol, kelvin, amu (OpenMM's MD unit system).  ``s = T_base / T_eff in (0, 1]`` is
the project's REST2 convention; ``s = 1`` is cold.
"""

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

__all__ = [
    "DEFAULTS",
    "load_config",
    "dump_defaults",
    "write_manifest",
    "rest2_ladder",
    "build_simbox",
    "minimize_equilibrate",
    "run_md",
    "run_rest2_remd",
    "initial_structure",
    "protonate",
    "solvate",
    "build_system",
    "repartition_hydrogen_mass",
    "make_integrator",
    "classify_omega_bonds",
    "omega_central_bonds",
]

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
# Every value the baseline uses is here.  A user-supplied ``--parameters CONFIG.json`` is merged
# on top of this tree (recursively, key by key), so a config file need only carry the values that
# differ from the baseline.  The merged tree is copied into every step's output directory, so a run
# is reproducible from its own outputs and a later change to this file cannot silently reinterpret
# an old run.
DEFAULTS: dict[str, Any] = {
    "run": {
        "name": None,                 # output directory basename; defaults to the system slug
        "root": None,                 # parent directory; defaults to <data_root>/explicit_baseline
        "seed": 20260814,             # master seed; per-step seeds derive from it unless set
    },
    "system": {
        "slug": None,                 # e.g. "alanine_dipeptide"; required for CV-aware analysis
        "solute_kind": "auto",        # auto | peptide | ligand | complex
        # When set, the run MUST be launched through this input route ("smiles" or "pdb").  A named
        # preset uses it to make its force-field choice an enforced invariant rather than a comment:
        # cyclo_rgdfv is a Sage 2.2/AM1-BCC calculation, and a PDB invocation must fail up front
        # instead of being silently reinterpreted as a peptide.
        "require_input_route": None,
    },
    # ---- step 1: initial structure (SMILES input only; a --pdb input skips this step) --------
    "structure": {
        "etkdg": {
            "version": "ETKDGv3",
            "n_conformers": 10,       # embed N, minimise all, keep the lowest-MMFF one
            "seed": None,             # None -> run.seed
            "use_random_coords": False,
            "prune_rms_thresh": 0.5,  # nm-free RDKit units (Angstrom)
            "num_threads": 0,         # 0 -> all cores
        },
        "mmff": {
            "variant": "MMFF94s",     # RDKit implements MMFF94 / MMFF94s only -- see the doc
            "max_iterations": 1000,
            "energy_tolerance": 1e-6,
            "force_tolerance": 1e-4,
        },
    },
    # ---- step 2: protonation ----------------------------------------------------------------
    "protonation": {
        "ph": 7.0,
        "delete_existing_hydrogens": True,   # delete, then let addHydrogens place them at pH
        "variants": None,                    # explicit per-residue variants; None -> pH decides
        "skip_for_ligand": True,             # an ETKDG ligand is already fully protonated
    },
    # ---- force field -------------------------------------------------------------------------
    "forcefield": {
        "protein": "amber19/protein.ff19SB.xml",
        "water": "amber19/tip3pfb.xml",
        "ligand": "openff-2.2.0",            # Sage 2.2
        "ligand_charge_method": "am1bcc",    # requires AmberTools (sqm) on PATH
        "extra_xml": [],
    },
    # ---- step 3: solvation -------------------------------------------------------------------
    "solvation": {
        "water_model": "tip3p",              # 3-site geometry template; parameters come from
                                             # forcefield.water (tip3pfb) -- see the doc
        "box_shape": "dodecahedron",
        "padding_nm": 1.2,
        # "solute-image-gap": padding means what it says -- at least padding_nm of solvent between
        #   the solute and its nearest periodic image.  "openmm": Modeller.addSolvent's own
        #   padding semantics, width = max(2*radius + padding, 2*padding), which for a compact
        #   solute in a dodecahedron delivers far less separation than padding_nm.  See the doc.
        "padding_semantics": "solute-image-gap",
        # the minimum image distance must be at least 2x the nonbonded cutoff (an OpenMM hard
        # requirement).  "grow" enlarges the box to satisfy it and records both values; "refuse"
        # stops and reports the padding that would work.
        "cutoff_fit_policy": "grow",
        "ionic_strength_molar": 0.15,
        "positive_ion": "Na+",
        "negative_ion": "Cl-",
        "neutralize": True,
    },
    # ---- step 4: System ----------------------------------------------------------------------
    "system_build": {
        "nonbonded_method": "PME",
        "nonbonded_cutoff_nm": 1.0,          # electrostatics AND vdW
        "switch_distance_nm": None,          # None -> hard vdW cutoff at nonbonded_cutoff_nm
        "use_dispersion_correction": True,
        "ewald_error_tolerance": 5.0e-4,
        "constraints": "HBonds",
        "rigid_water": True,
        "hydrogen_mass_amu": 3.024,          # 3x1.008; enables the 4 fs timestep
        "hmr_scope": "solute",               # solute | all -- never touches rigid water
        "remove_cm_motion": True,
    },
    # ---- integrator --------------------------------------------------------------------------
    "integrator": {
        # "langevin-middle" is openmm.LangevinMiddleIntegrator (BAOAB): the position update sits
        # between two half-kicks, which is what makes a 4 fs timestep defensible with HMR + HBonds.
        # "leapfrog-langevin" is openmm.LangevinIntegrator, the legacy leapfrog scheme; it samples a
        # slightly hot configurational distribution at long timesteps and is kept for reproducing
        # runs made before 2026-08-14.
        "kind": "langevin-middle",           # langevin-middle | leapfrog-langevin | verlet
        "timestep_fs": 4.0,
        "friction_per_ps": 1.0,
        "temperature_k": 300.0,
    },
    # ---- step 5: equilibration ---------------------------------------------------------------
    # "staged" is the standard protocol for a flexible solute (macrocycle, peptide) dropped into a
    # freshly packed water box: restrained minimisation, gradual heating, then NPT with the
    # restraint released in steps.  "simple" is minimise -> NVT -> NPT with no restraints, which is
    # appropriate for a small rigid solute (alanine dipeptide) and nothing larger.
    "equilibration": {
        "protocol": "staged",                # staged | simple
        "minimize_max_iterations": 0,        # 0 == minimise to convergence (NOT "skip")
        "minimize_tolerance_kj_mol_nm": 10.0,
        "restraint_k_kj_mol_nm2": 4184.0,    # 10 kcal/mol/A^2, the usual starting stiffness
        "restraint_selection": "solute-heavy",
        "heat_from_k": 50.0,
        "heat_to_k": None,                   # None -> integrator.temperature_k
        "heat_ps": 200.0,
        "heat_timestep_fs": 1.0,             # gentle while the water shell is still bad
        "heat_n_windows": 25,
        "npt_restrained_ps": 200.0,
        # restraint stiffness for each release window, in kJ/mol/nm^2 (~2.5, 1, 0 kcal/mol/A^2)
        "release_schedule_kj_mol_nm2": [1046.0, 418.0, 0.0],
        "release_ps_each": 200.0,
        "npt_free_ps": 1000.0,
        "timestep_fs": 2.0,                  # after heating, and for the "simple" protocol
        "nvt_ps": 200.0,                     # "simple" protocol only
        "npt_ps": 1000.0,                    # "simple" protocol only
        "pressure_bar": 1.0,
        "barostat_interval": 50,
        "box_average_last_ps": 500.0,        # NPT window averaged to fix the production box
        "seed": None,
    },
    # ---- steps 6-8: production ---------------------------------------------------------------
    "production": {
        "ensemble": "NVT",                   # NVT at the equilibrated box; see the doc
        "platform": "CUDA",
        "precision": "mixed",                # explicit: the CUDA default is single
        "device_index": None,
        "report": {
            "all_atom_ps": 10.0,             # f_all
            "solute_ps": 2.0,                # f_solute
            "state_ps": 10.0,
            "checkpoint_ps": 100.0,
        },
        # ONE walker per invocation of md.py; the scale factor selects cold (s = 1) or hot.
        "md": {
            "scale_factor": 1.0,
            "total_ns": 1000.0,
            "chunk_ns": 100.0,
            "seed": None,
            "label": "cold",                 # free-text, recorded and used in log lines
        },
        "remd": {
            "total_ns_per_replica": 1000.0,
            "chunk_ns": 100.0,
            "scale_factors": None,           # None -> built from rest2.ladder
            "exchange_interval_ps": 10.0,
            # PRE-EXCHANGE RELAXATION, discarded.  Every replica starts from the same equilibrated
            # coordinates, which were equilibrated at s = 1; a replica at s < 1 is therefore not in
            # its own ensemble at step 0, and exchanging immediately would mix in that transient.
            # This propagates each replica under ITS OWN Hamiltonian first, with no reporters and
            # no exchanges, and throws the result away.  It is NOT a claim of equilibrium.
            "equilibration_ps": 10.0,
            "seed": None,
        },
    },
    "rest2": {
        "omega_selective": True,             # leave ORDINARY amide omega torsions unscaled
        # A proline-like peptide bond stays ELIGIBLE for scaling: its nitrogen is ring-locked, so
        # the torsion is not the near-planar two-state coordinate the exclusion protects.
        "proline_like_residues": ["PRO"],
        # ligand route only: an amide N counts as proline-like when it sits in a ring of at most
        # this many atoms.  Load-bearing for macrocycles -- every backbone N of a cyclic peptide is
        # "in a ring", and an unbounded test would free every macrocyclic omega for scaling.
        "max_proline_ring_size": 7,
        # the REMD ladder, when production.remd.scale_factors is not given explicitly
        "ladder": {
            "s_cold": 1.0,
            "s_hot": 0.25,
            "n_rungs": 6,                    # 6 for alanine, 8 for macrocycles
            "interp": "sqrt",                # sqrt | linear | geometric
        },
    },
}


def rest2_ladder(s_cold: float = 1.0, s_hot: float = 0.25, n_rungs: int = 6,
                 interp: str = "sqrt") -> list[float]:
    """Return the REST2 scale-factor ladder, descending, with ``result[0] == s_cold``.

    ``interp`` fixes what is spaced evenly between the two ends:

    * ``sqrt``      -- linear in ``sqrt(s)``.  The project default, and what
      ``topology_prep.effective_temperature_ladder`` already uses.  Exchange acceptance depends on
      the overlap of the two rungs' energy distributions, whose width scales roughly as
      ``sqrt(s)``, so even spacing in ``sqrt(s)`` gives roughly even acceptance along the chain.
      With ``s_hot = 0.25`` and 6 rungs it reproduces the ladder used for every existing reference:
      ``[1.0, 0.81, 0.64, 0.49, 0.36, 0.25]``.
    * ``linear``    -- linear in ``s`` itself; bunches the rungs at the hot end.
    * ``geometric`` -- linear in ``ln s``, i.e. a geometric temperature ladder, the usual choice for
      *parallel tempering*.  It is offered for comparison; REST2 scales only the solute, so the
      argument for a geometric ladder does not transfer unchanged.
    """
    if n_rungs < 2:
        raise ValueError(f"n_rungs must be >= 2, got {n_rungs}")
    if not (0.0 < s_hot <= s_cold <= 1.0):
        raise ValueError(f"require 0 < s_hot <= s_cold <= 1, got s_hot={s_hot}, s_cold={s_cold}")
    if interp == "sqrt":
        values = np.linspace(math.sqrt(s_cold), math.sqrt(s_hot), n_rungs) ** 2
    elif interp == "linear":
        values = np.linspace(s_cold, s_hot, n_rungs)
    elif interp == "geometric":
        values = np.exp(np.linspace(math.log(s_cold), math.log(s_hot), n_rungs))
    else:
        raise ValueError(f"unknown interp {interp!r}; use sqrt | linear | geometric")
    return [float(round(v, 12)) for v in values]


def dump_defaults(path: Optional[Path] = None) -> str:
    """Serialise DEFAULTS deterministically; the single source for ``config_defaults.json``.

    The dumped file and the runtime tree drifted once already -- the dump was missing
    ``production.remd.equilibration_ps`` -- because they were maintained by hand in two places.
    This makes the file a function of the tree, and a test compares them.
    """
    text = json.dumps(DEFAULTS, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = dict(base)
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            raise KeyError(
                f"unknown configuration key '{where}'.  Valid keys at this level: "
                f"{sorted(base)}.  Typos are rejected rather than ignored so a misspelled "
                f"parameter cannot silently leave the baseline value in force."
            )
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(base[key], value, where)
        else:
            out[key] = value
    return out


def load_config(parameters: Optional[Path] = None) -> dict[str, Any]:
    """Return DEFAULTS with ``parameters`` (a JSON file of non-default values) merged on top.

    Unknown keys raise.  A silently ignored typo in a config file is the failure mode this
    guards: it would leave the baseline value in force while the run's own config.json claims
    otherwise.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if parameters is not None:
        raw = json.loads(Path(parameters).read_text(encoding="utf-8"))
        cfg = _deep_merge(cfg, raw)
    master = int(cfg["run"]["seed"])
    # derive any unset seed from the master seed, deterministically and distinctly per step
    for offset, (section, key) in enumerate(
        [
            (cfg["structure"]["etkdg"], "seed"),
            (cfg["equilibration"], "seed"),
            (cfg["production"]["md"], "seed"),
            (cfg["production"]["remd"], "seed"),
        ]
    ):
        if section.get(key) is None:
            section[key] = master + offset
    return cfg


def _git_commit() -> dict[str, str]:
    try:
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=20,
        )
        return {
            "commit": commit.stdout.strip() or "unknown",
            "dirty": "true" if dirty.stdout.strip() else "false",
        }
    except Exception:  # pragma: no cover - provenance must never break a run
        return {"commit": "unknown", "dirty": "unknown"}


def write_manifest(out_dir: Path, step: str, cfg: dict, extra: Optional[dict] = None) -> Path:
    """Write ``<out_dir>/manifest_<step>.json`` and a copy of the resolved config."""
    import openmm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    payload = {
        "step": step,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_commit(),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "openmm": openmm.__version__,
        "host": _platform.node(),
    }
    if extra:
        payload.update(extra)
    path = out_dir / f"manifest_{step}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


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
                f"toolkit registry only has {wrappers}.  Activate the escort-ais environment "
                "(conda activate escort-ais) so antechamber/sqm are on PATH, or set "
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

    width_requested = width
    min_image = frac * width
    needed = 2.0 * cutoff
    policy = str(scfg["cutoff_fit_policy"])
    grown = False
    if min_image < needed:
        required_width = needed / frac
        if policy == "grow":
            width = required_width
            grown = True
        elif policy == "refuse":
            raise ValueError(
                f"a {shape} box with padding {padding} nm gives a minimum image distance of "
                f"{min_image:.3f} nm, but a {cutoff} nm cutoff needs {needed:.3f} nm.  Increase "
                f"solvation.padding_nm to at least {frac * required_width - 2 * radius:.3f} nm, "
                f"lower system_build.nonbonded_cutoff_nm to {min_image / 2:.3f} nm, or set "
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
        "max_legal_cutoff_nm": round(min_image / 2, 5),
        "box_vectors_nm": vectors.tolist(),
    }
    if grown:
        print(
            f"[solvate] box grown for the cutoff: minimum image {width_requested * frac:.3f} -> "
            f"{min_image:.3f} nm so a {cutoff} nm cutoff fits (needs {needed:.3f} nm).  The "
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
# Integrator / Simulation helpers
# ---------------------------------------------------------------------------------------------
def make_integrator(cfg: dict, seed: int, timestep_fs: Optional[float] = None):
    """Return a seeded integrator.

    ``setRandomNumberSeed`` MUST be called before the ``Context`` exists -- afterwards OpenMM
    silently ignores it.  That is a project-wide engine invariant, and it is why this function
    returns the integrator rather than a Simulation built around one.
    """
    import openmm
    from openmm import unit

    icfg = cfg["integrator"]
    dt = float(icfg["timestep_fs"] if timestep_fs is None else timestep_fs) * unit.femtosecond
    temperature = float(icfg["temperature_k"]) * unit.kelvin
    friction = float(icfg["friction_per_ps"]) / unit.picosecond
    kind = str(icfg["kind"]).lower()

    if kind in ("leapfrog-langevin", "langevin", "leapfrog"):
        integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    elif kind in ("langevin-middle", "middle", "baoab"):
        integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
    elif kind == "verlet":
        integrator = openmm.VerletIntegrator(dt)
    else:
        raise ValueError(f"unknown integrator.kind {icfg['kind']!r}")
    integrator.setRandomNumberSeed(int(seed) % 2_147_483_647)
    return integrator


def _platform_and_properties(cfg: dict):
    import openmm

    pcfg = cfg["production"]
    name = str(pcfg["platform"])
    plat = openmm.Platform.getPlatformByName(name)
    props: dict[str, str] = {}
    if name in ("CUDA", "OpenCL"):
        # The platform default is single precision: ~25 % faster and it silently changes energies.
        props["Precision"] = str(pcfg["precision"])
        if pcfg["device_index"] is not None:
            props["DeviceIndex"] = str(pcfg["device_index"])
    return plat, props


def _make_simulation(topology, system, cfg: dict, seed: int, timestep_fs: Optional[float] = None):
    from openmm import app

    integrator = make_integrator(cfg, seed, timestep_fs)
    plat, props = _platform_and_properties(cfg)
    return app.Simulation(topology, system, integrator, plat, props or None)


def _steps(time_ps: float, timestep_fs: float) -> int:
    n = time_ps * 1000.0 / timestep_fs
    if abs(n - round(n)) > 1e-6:
        raise ValueError(
            f"{time_ps} ps is not an integer number of {timestep_fs} fs steps "
            f"({n:.6f}).  Choose intervals divisible by the timestep -- a rounded reporter "
            "interval silently changes the sampling frequency."
        )
    return int(round(n))


def _scaled_system(system, cfg: dict, n_solute_atoms: int, scale_factor: float,
                   omega_bonds: Sequence[Sequence[int]]):
    """Apply the project's REST2 scaling to a copy of *system* (identity at ``s = 1``)."""
    from escort_ais.systems.openmm_system import build_rest2_scaled_system

    if abs(scale_factor - 1.0) < 1e-12:
        return copy.deepcopy(system)
    return build_rest2_scaled_system(
        system,
        np.arange(int(n_solute_atoms)),
        float(scale_factor),
        exclude_central_bonds=[tuple(b) for b in omega_bonds] or None,
    )


# ---------------------------------------------------------------------------------------------
# Step 5 -- minimise, NVT, NPT
# ---------------------------------------------------------------------------------------------
def _add_positional_restraints(system, topology, selection: str, positions_nm: np.ndarray):
    """Add a flat harmonic positional restraint driven by the global parameter ``k_restraint``.

    ``periodicdistance`` is used so an atom that wanders across a box face is still measured to its
    own reference point rather than to an image of it.  The stiffness is a global parameter, so it
    can be lowered between stages with one ``setParameter`` call instead of rebuilding the Context.
    """
    from openmm import CustomExternalForce
    from openmm.app import element as elem

    force = CustomExternalForce(
        "k_restraint*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    force.addGlobalParameter("k_restraint", 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)

    atoms = list(topology.atoms())
    restrained = []
    for atom in atoms:
        if atom.residue.name.upper() in WATER_RESIDUE_NAMES | ION_RESIDUE_NAMES:
            continue
        if selection == "solute-heavy" and atom.element == elem.hydrogen:
            continue
        force.addParticle(int(atom.index), [float(x) for x in positions_nm[atom.index]])
        restrained.append(int(atom.index))
    index = system.addForce(force)
    return index, restrained


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Heavy-atom RMSD after optimal superposition (nm)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum() / len(a)))


def build_simbox(cfg: dict, out_dir: Path, suffix: str, *, smiles: Optional[str] = None,
                 pdb: Optional[Path] = None) -> dict:
    """Stage (a): SMILES or PDB in, a parameterised solvated box out.

    Runs initial structure -> protonation -> solvation -> System in one go and emits a bundle whose
    names carry *suffix*:

        <suffix>_system.xml     the parameterised System (the "-p" of every later stage)
        <suffix>_topology.pdb   topology, solvated coordinates and box vectors
        <suffix>_simbox.json    n_solute_atoms, omega bonds, box geometry, force field, provenance

    The per-step intermediates stay in ``<suffix>_prep/`` so nothing is thrown away.
    """
    out_dir = Path(out_dir)
    work = out_dir / f"{suffix}_prep"
    work.mkdir(parents=True, exist_ok=True)

    if (smiles is None) == (pdb is None):
        raise ValueError("build_simbox needs exactly one of smiles= or pdb=")

    input_route = "smiles" if smiles is not None else "pdb"
    required = cfg["system"].get("require_input_route")
    if required and input_route != required:
        raise ValueError(
            f"this configuration requires the '{required}' input route; got '{input_route}'.  "
            "Supply --smiles for the cyclo_rgdfv preset."
        )
    if required == "smiles" and not (smiles or "").strip():
        raise ValueError("the required 'smiles' input route was selected but no SMILES was given")
    if smiles is not None:
        structure = initial_structure(smiles, work, cfg)
    else:
        import shutil

        shutil.copy(Path(pdb), work / "solute.pdb")
        structure = {"source": "pdb", "input_pdb": str(pdb),
                     "note": "ETKDG skipped: a structure was supplied"}
        (work / "initial_structure.json").write_text(
            json.dumps(structure, indent=2) + "\n", encoding="utf-8"
        )

    sdf = work / "solute.sdf"
    ligand = sdf if sdf.exists() else None
    prot = protonate(work / "solute.pdb", work, cfg, ligand_sdf=ligand,
                     input_route=input_route)
    solv = solvate(work / "solute_h.pdb", work, cfg, ligand_sdf=ligand, route=prot["route"])
    build = build_system(
        work / "solvated.pdb", work, cfg, solv["n_solute_atoms"], ligand_sdf=ligand,
        route=prot["route"],
    )

    system_xml = out_dir / f"{suffix}_system.xml"
    topology_pdb = out_dir / f"{suffix}_topology.pdb"
    system_xml.write_text(Path(build["system_xml"]).read_text(encoding="utf-8"), encoding="utf-8")
    topology_pdb.write_text(Path(build["solvated_pdb"]).read_text(encoding="utf-8"),
                            encoding="utf-8")

    info = {
        "suffix": suffix,
        "input_route": input_route,
        "route": prot["route"],
        "input_smiles": smiles,
        "input_pdb": None if smiles is not None else str(pdb),
        "system_xml": str(system_xml),
        "topology_pdb": str(topology_pdb),
        "prep_dir": str(work),
        "n_solute_atoms": int(build["n_solute_atoms"]),
        "omega_central_bonds": build["omega_central_bonds"],
        **{k: build[k] for k in
           ("omega_unscaled_bonds", "omega_proline_like_scaled_bonds",
            "omega_unclassified_candidates", "omega_detection_method", "omega_detail")},
        "n_particles": build["n_particles"],
        "n_constraints": build["n_constraints"],
        "degrees_of_freedom": build["degrees_of_freedom"],
        "hmr": build["hmr"],
        "nonbonded": build["nonbonded"],
        "geometry": solv["geometry"],
        "n_waters": solv["n_waters"],
        "ions": solv["ions"],
        "realised_ionic_strength_molar": solv["realised_ionic_strength_molar"],
        "forcefield": build["forcefield"],
        # explicit, unambiguous provenance of the Hamiltonian that actually ran
        "hamiltonian_provenance": {
            "solute_route": input_route,
            "small_molecule_forcefield": (cfg["forcefield"]["ligand"]
                                          if prot["route"] == "ligand" else None),
            "charge_method": (cfg["forcefield"]["ligand_charge_method"]
                              if prot["route"] == "ligand" else None),
            "protein_forcefield": build["forcefield"]["protein_forcefield"],
            "water_forcefield": cfg["forcefield"]["water"],
            "input_smiles": smiles,
            "input_smiles_sha256": (
                hashlib.sha256(smiles.encode("utf-8")).hexdigest() if smiles else None
            ),
            "formal_charge": (build["forcefield"]["ligand"] or {}).get("formal_charge"),
        },
        "structure": structure,
        "protonation": {k: prot[k] for k in
                        ("ph", "ph_applies", "route", "note", "n_hydrogens_after")},
    }
    (out_dir / f"{suffix}_simbox.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_simbox", cfg, {"result": info})
    return info


def _load_bundle(system_xml: Path):
    """Return ``(system, topology, simbox_info)`` for a ``-p`` argument.

    ``-p`` names the parameterised System.  A ``System`` XML carries parameters but no atom or
    residue names, so the topology is read from the sibling ``<stem>_topology.pdb`` that
    :func:`build_simbox` wrote next to it -- the pair together is the analogue of an Amber prmtop.
    """
    from openmm import XmlSerializer, app

    system_xml = Path(system_xml)
    if not system_xml.exists():
        raise FileNotFoundError(f"-p {system_xml} not found")
    name = system_xml.name
    base = name[: -len("_system.xml")] if name.endswith("_system.xml") else system_xml.stem
    topology_pdb = system_xml.with_name(f"{base}_topology.pdb")
    info_path = system_xml.with_name(f"{base}_simbox.json")
    for path, what in ((topology_pdb, "topology"), (info_path, "box metadata")):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing.  -p expects the bundle written by simbox-setup.py: "
                f"<stem>_system.xml, <stem>_topology.pdb and <stem>_simbox.json side by side "
                f"({what} comes from this file)."
            )
    system = XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
    pdb = app.PDBFile(str(topology_pdb))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return system, pdb, info


def _apply_coords(sim, coords: Path, *, require_velocities: bool = False) -> dict:
    """Seed a Context from ``-c``: either a PDB or a serialised OpenMM ``State``.

    Deliberately not ``loadCheckpoint``: a checkpoint is only valid for the exact System that wrote
    it, and the equilibration System carries a barostat and a positional-restraint force that a
    production System does not.  Copying positions, velocities and box vectors across is
    System-agnostic and cannot silently mismatch.
    """
    from openmm import XmlSerializer, app, unit

    coords = Path(coords)
    if not coords.exists():
        raise FileNotFoundError(f"-c {coords} not found")
    if coords.suffix.lower() == ".pdb":
        pdb = app.PDBFile(str(coords))
        box = pdb.topology.getPeriodicBoxVectors()
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
        sim.context.setPositions(pdb.positions)
        if require_velocities:
            sim.context.setVelocitiesToTemperature(
                float(sim.integrator.getTemperature().value_in_unit(unit.kelvin)) * unit.kelvin
            )
        return {"coords": str(coords), "kind": "pdb", "velocities": "drawn from Maxwell-Boltzmann"}

    state = XmlSerializer.deserialize(coords.read_text(encoding="utf-8"))
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim.context.setPositions(state.getPositions())
    try:
        sim.context.setVelocities(state.getVelocities())
        vel = "carried over from the state"
    except Exception:
        sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature())
        vel = "state carried none; drawn from Maxwell-Boltzmann"
    return {"coords": str(coords), "kind": "state-xml", "velocities": vel}


def minimize_equilibrate(cfg: dict, system_xml: Path, coords: Path, out_dir: Path,
                         suffix: str) -> dict:
    """Stage (b): minimise and equilibrate; emit the state every production stage starts from.

    ``equilibration.protocol = "staged"`` (the default) is the standard protocol for a flexible
    solute in a freshly built water box:

        1. minimise with the solute heavy atoms restrained  -- lets the water shell relax around
           the conformer instead of the conformer deforming to fit a badly packed shell;
        2. minimise unrestrained;
        3. heat 50 K -> 300 K over 200 ps in NVT at 1 fs, restrained -- a Modeller water box is
           placed on a lattice, so the first picoseconds carry large local forces;
        4. NPT restrained, 200 ps -- the density collapses onto its equilibrium value here;
        5. release the restraint in steps (10 -> 2.5 -> 1 -> 0 kcal/mol/A^2), 200 ps each;
        6. unrestrained NPT, 1 ns, whose tail sets the production box.

    Total 2.0 ns, negligible against a 1 us production run.  ``protocol = "simple"`` is
    minimise -> NVT -> NPT with no restraints and no ramp; adequate for a small rigid solute such as
    alanine dipeptide and nothing larger.

    What equilibration does *not* fix: the starting conformer is one arbitrary point in the
    macrocycle's conformational space, and no equilibration protocol converges cis/trans amides,
    ring pucker or rotamers.  That is what the REST2 ladder and the production length are for.  The
    solute heavy-atom RMSD from the minimised structure is recorded per stage so it is visible how
    far the conformer moved.

    Writes ``<suffix>_state.xml`` (positions, velocities, box) -- the ``-c`` of stages (c) and (d).
    """
    from openmm import MonteCarloBarostat, XmlSerializer, app, unit
    from openmm.app import StateDataReporter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ecfg = cfg["equilibration"]
    protocol = str(ecfg["protocol"])
    if protocol not in ("staged", "simple"):
        raise ValueError(f"equilibration.protocol must be 'staged' or 'simple', got {protocol!r}")

    system, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])

    temperature = float(cfg["integrator"]["temperature_k"])
    t_target = float(ecfg["heat_to_k"] or temperature)

    ref_positions = np.array(pdb.positions.value_in_unit(unit.nanometer))
    restraint_index, restrained_atoms = _add_positional_restraints(
        system, pdb.topology, str(ecfg["restraint_selection"]), ref_positions
    )
    barostat = MonteCarloBarostat(
        float(ecfg["pressure_bar"]) * unit.bar, temperature * unit.kelvin,
        int(ecfg["barostat_interval"]),
    )
    barostat_index = system.addForce(barostat)

    dt0 = float(ecfg["heat_timestep_fs"] if protocol == "staged" else ecfg["timestep_fs"])
    sim = _make_simulation(pdb.topology, system, cfg, int(ecfg["seed"]), timestep_fs=dt0)
    origin = _apply_coords(sim, coords)
    _set_barostat(sim, barostat_index, 0)

    # the restraint reference must match the coordinates actually loaded, not the bundle PDB
    ref_positions = sim.context.getState(
        getPositions=True, enforcePeriodicBox=False
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    _retarget_restraints(sim, restraint_index, ref_positions)

    heavy = [
        a.index for a in pdb.topology.atoms()
        if a.index < n_solute and a.element is not None and a.element.symbol != "H"
    ]

    def solute_rmsd_nm() -> float:
        pos = sim.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(
            asNumpy=True
        ).value_in_unit(unit.nanometer)
        return _kabsch_rmsd(pos[heavy], ref_positions[heavy])

    k_strong = float(ecfg["restraint_k_kj_mol_nm2"])
    tol = float(ecfg["minimize_tolerance_kj_mol_nm"]) * unit.kilojoule_per_mole / unit.nanometer
    max_it = int(ecfg["minimize_max_iterations"])
    stages: list[dict] = []

    def record(name: str, **kw) -> None:
        state = sim.context.getState(getEnergy=True)
        vol = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
        entry = {
            "stage": name,
            "potential_kj_mol": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            "box_volume_nm3": round(vol, 4),
            "solute_heavy_rmsd_from_start_nm": round(solute_rmsd_nm(), 4),
            **kw,
        }
        stages.append(entry)
        print(
            f"[equil] {name:18s} U = {entry['potential_kj_mol']:12.1f} kJ/mol  "
            f"V = {vol:8.2f} nm^3  solute RMSD = {entry['solute_heavy_rmsd_from_start_nm']:.3f} nm",
            flush=True,
        )

    record("initial", restraint_k=0.0)
    csv_path = out_dir / f"{suffix}_equilibration.csv"

    if protocol == "staged":
        sim.context.setParameter("k_restraint", k_strong)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_restrained", restraint_k=k_strong)
        sim.context.setParameter("k_restraint", 0.0)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_free", restraint_k=0.0)

        with (out_dir / f"{suffix}_minimized.pdb").open("w") as fh:
            app.PDBFile.writeFile(
                sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
            )
        # the restraint reference is the MINIMISED structure, not the raw input
        ref_positions = sim.context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        _retarget_restraints(sim, restraint_index, ref_positions)

        sim.reporters.append(
            StateDataReporter(
                str(csv_path), _steps(10.0, dt0), step=True, time=True, potentialEnergy=True,
                kineticEnergy=True, temperature=True, volume=True, density=True, speed=True,
            )
        )
        sim.context.setParameter("k_restraint", k_strong)
        t_from = float(ecfg["heat_from_k"])
        sim.context.setVelocitiesToTemperature(t_from * unit.kelvin, int(ecfg["seed"]))
        n_win = max(1, int(ecfg["heat_n_windows"]))
        per_window = max(1, _steps(float(ecfg["heat_ps"]), dt0) // n_win)
        for w in range(n_win):
            t_w = t_from + (t_target - t_from) * (w + 1) / n_win
            sim.integrator.setTemperature(t_w * unit.kelvin)
            sim.step(per_window)
        record("nvt_heat", restraint_k=k_strong, temperature_k=t_target,
               ps=float(ecfg["heat_ps"]), timestep_fs=dt0)

        sim.integrator.setStepSize(float(ecfg["timestep_fs"]) * unit.femtosecond)
        dt = float(ecfg["timestep_fs"])
        _set_barostat(sim, barostat_index, int(ecfg["barostat_interval"]),
                      float(ecfg["pressure_bar"]))
        sim.step(_steps(float(ecfg["npt_restrained_ps"]), dt))
        record("npt_restrained", restraint_k=k_strong, ps=float(ecfg["npt_restrained_ps"]))

        for k in ecfg["release_schedule_kj_mol_nm2"]:
            sim.context.setParameter("k_restraint", float(k))
            sim.step(_steps(float(ecfg["release_ps_each"]), dt))
            record(f"npt_release_k{float(k):g}", restraint_k=float(k),
                   ps=float(ecfg["release_ps_each"]))
        sim.context.setParameter("k_restraint", 0.0)
        free_ps = float(ecfg["npt_free_ps"])
    else:
        sim.context.setParameter("k_restraint", 0.0)
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_it)
        record("min_free", restraint_k=0.0)
        with (out_dir / f"{suffix}_minimized.pdb").open("w") as fh:
            app.PDBFile.writeFile(
                sim.topology, sim.context.getState(getPositions=True).getPositions(), fh
            )
        sim.reporters.append(
            StateDataReporter(
                str(csv_path), _steps(10.0, dt0), step=True, time=True, potentialEnergy=True,
                kineticEnergy=True, temperature=True, volume=True, density=True, speed=True,
            )
        )
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, int(ecfg["seed"]))
        sim.integrator.setTemperature(temperature * unit.kelvin)
        dt = float(ecfg["timestep_fs"])
        sim.step(_steps(float(ecfg["nvt_ps"]), dt))
        record("nvt", ps=float(ecfg["nvt_ps"]))
        _set_barostat(sim, barostat_index, int(ecfg["barostat_interval"]),
                      float(ecfg["pressure_bar"]))
        free_ps = float(ecfg["npt_ps"])

    tail_ps = min(float(ecfg["box_average_last_ps"]), free_ps)
    head_steps = _steps(free_ps, dt) - _steps(tail_ps, dt)
    if head_steps > 0:
        sim.step(head_steps)
    tail_steps = _steps(tail_ps, dt)
    sample_every = max(1, min(_steps(10.0, dt), tail_steps // 20))
    n_samples = max(1, tail_steps // sample_every)
    # Keep the STATES, not only the box vectors.  Pasting a mean box onto the final instantaneous
    # configuration would pair coordinates equilibrated in one cell with a cell they never saw --
    # the solvent density and the solute's periodic separation would both be slightly wrong, and
    # nothing downstream would notice.  Instead the handoff is the sampled state whose own volume
    # is closest to the tail mean: a configuration and a box that actually occurred together.
    samples, boxes = [], []
    for _ in range(n_samples):
        sim.step(sample_every)
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        boxes.append(bv)
        samples.append((float(np.abs(np.linalg.det(bv))), st, bv))
    mean_box = np.mean(np.array(boxes), axis=0)
    record("npt_free", restraint_k=0.0, ps=free_ps)

    mean_vol = float(np.mean([v for v, _, _ in samples]))
    pick = int(np.argmin([abs(v - mean_vol) for v, _, _ in samples]))
    sel_vol, final, sel_box = samples[pick]
    box_handoff = {
        "method": "nearest-sampled-state",
        "mean_tail_volume_nm3": round(mean_vol, 5),
        "selected_volume_nm3": round(sel_vol, 5),
        "deviation_nm3": round(sel_vol - mean_vol, 6),
        "deviation_percent": round(100.0 * (sel_vol - mean_vol) / mean_vol, 4),
        "selected_sample_index": pick,
        "n_samples": len(samples),
        "note": "coordinates, velocities and box all come from ONE sampled state; the mean box is "
                "reported for reference and is deliberately NOT pasted onto other coordinates",
    }
    print(f"[equil] box handoff: sample {pick}/{len(samples)} at {sel_vol:.3f} nm^3, "
          f"{box_handoff['deviation_percent']:+.2f} % from the tail mean {mean_vol:.3f} nm^3",
          flush=True)
    mean_box = sel_box
    state_xml = out_dir / f"{suffix}_state.xml"
    state_xml.write_text(XmlSerializer.serialize(final), encoding="utf-8")
    with (out_dir / f"{suffix}_equilibrated.pdb").open("w") as fh:
        app.PDBFile.writeFile(sim.topology, final.getPositions(), fh)

    volumes = [float(np.abs(np.linalg.det(b))) for b in boxes]
    info = {
        "suffix": suffix,
        "protocol": protocol,
        "system_xml": str(system_xml),
        "input_coords": origin,
        "stages": stages,
        "n_restrained_atoms": len(restrained_atoms),
        "restraint_selection": str(ecfg["restraint_selection"]),
        "restraint_k_kj_mol_nm2": k_strong,
        "release_schedule_kj_mol_nm2": list(ecfg["release_schedule_kj_mol_nm2"]),
        "total_equilibration_ps": (
            float(ecfg["heat_ps"]) + float(ecfg["npt_restrained_ps"])
            + len(ecfg["release_schedule_kj_mol_nm2"]) * float(ecfg["release_ps_each"])
            + float(ecfg["npt_free_ps"])
            if protocol == "staged"
            else float(ecfg["nvt_ps"]) + float(ecfg["npt_ps"])
        ),
        "box_vectors_nm": mean_box.tolist(),
        "box_volume_nm3_mean": float(np.mean(volumes)),
        "box_volume_nm3_sd": float(np.std(volumes, ddof=1)) if len(volumes) > 1 else 0.0,
        "n_box_samples": len(volumes),
        "solute_heavy_rmsd_from_start_nm": stages[-1]["solute_heavy_rmsd_from_start_nm"],
        "box_handoff": box_handoff,
        "state_xml": str(state_xml),
        "production_ensemble": cfg["production"]["ensemble"],
    }
    (out_dir / f"{suffix}_min-eq.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_min-eq", cfg, {"result": info})
    return info


def _retarget_restraints(sim, force_index: int, positions_nm: np.ndarray) -> None:
    """Move every restraint's reference point onto *positions_nm*, in place."""
    force = sim.system.getForce(force_index)
    for particle in range(force.getNumParticles()):
        atom_index, _ = force.getParticleParameters(particle)
        force.setParticleParameters(
            particle, atom_index, [float(x) for x in positions_nm[atom_index]]
        )
    force.updateParametersInContext(sim.context)


def _set_barostat(sim, force_index: int, frequency: int, pressure_bar: float = 1.0) -> None:
    """Set the barostat frequency (0 = off for an NVT leg) and pressure, in place.

    The force stays in the System rather than being removed and re-added, so force indices the rest
    of the System refers to never move.
    """
    from openmm import unit

    force = sim.system.getForce(force_index)
    force.setFrequency(int(frequency))
    force.setDefaultPressure(pressure_bar * unit.bar)
    sim.context.reinitialize(preserveState=True)


# ---------------------------------------------------------------------------------------------
# Chunked production
# ---------------------------------------------------------------------------------------------
def _attach_chunk_reporters(sim, chunk_dir: Path, cfg: dict, dt_fs: float,
                            solute_atoms: Sequence[int]) -> None:
    from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter

    rcfg = cfg["production"]["report"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sim.reporters.clear()
    # all atoms: WRAPPED into the box, the usual convention for a solvated trajectory
    sim.reporters.append(
        DCDReporter(
            str(chunk_dir / "traj_all.dcd"),
            _steps(float(rcfg["all_atom_ps"]), dt_fs),
            enforcePeriodicBox=True,
        )
    )
    # solute only: NOT wrapped.  OpenMM's enforcePeriodicBox wraps whole MOLECULES, not atoms, so
    # bonds are never broken either way (verified: a solute straddling a box face comes back with a
    # max bonded distance of 0.153 nm under both settings) -- torsions would be safe regardless.
    # What wrapping does do is teleport the whole solute to the other side of the box whenever its
    # centre crosses a face, which breaks every analysis that reads the trajectory as continuous
    # (RMSD without re-imaging, diffusion, Cartesian TICA) and makes the structure jump around in a
    # viewer.  Unwrapped, the solute may drift far from the origin, which nothing here cares about.
    sim.reporters.append(
        DCDReporter(
            str(chunk_dir / "traj_solute.dcd"),
            _steps(float(rcfg["solute_ps"]), dt_fs),
            enforcePeriodicBox=False,
            atomSubset=list(int(i) for i in solute_atoms),
        )
    )
    sim.reporters.append(
        StateDataReporter(
            str(chunk_dir / "state.csv"), _steps(float(rcfg["state_ps"]), dt_fs),
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
        )
    )
    sim.reporters.append(
        CheckpointReporter(
            str(chunk_dir / "mid.chk"), _steps(float(rcfg["checkpoint_ps"]), dt_fs)
        )
    )


def _close_chunk(sim) -> None:
    for reporter in list(sim.reporters):
        for attr in ("_out", "_traj_file", "_dcd"):
            handle = getattr(reporter, attr, None)
            close = getattr(handle, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
    sim.reporters.clear()


def completed_prefix(run_dir: Path, n_chunks: int, *, what: str = "chunk") -> int:
    """Length of the CONTIGUOUS run of completed chunks starting at 0.

    ``len(glob("chunk_*/done.json"))`` was wrong: with chunks 0 and 2 complete and 1 missing it
    returns 2 and the run silently restarts at chunk 2, leaving a hole that no later analysis can
    see.  A gap means something failed or was deleted, and continuing past it fabricates a
    trajectory that was never contiguous -- so it is rejected rather than repaired.
    """
    run_dir = Path(run_dir)
    done = {int(q.parent.name.split("_")[1]) for q in run_dir.glob("chunk_*/done.json")}
    prefix = 0
    while prefix in done:
        prefix += 1
    stray = sorted(c for c in done if c > prefix)
    if stray:
        raise ValueError(
            f"{run_dir}: {what}s {stray} are complete but {prefix} is missing.  The completed "
            f"prefix is {prefix}, so continuing would leave a gap.  Delete the stray "
            f"{what} directories to restart cleanly from {prefix}, or restore the missing one."
        )
    for c in range(prefix):
        chk = run_dir / f"chunk_{c:04d}" / "end.chk"
        if not chk.exists():
            raise ValueError(
                f"{run_dir}: chunk {c} is marked done but {chk.name} is missing, so the run "
                "cannot be continued from it."
            )
    if prefix > n_chunks:
        raise ValueError(f"{run_dir}: {prefix} completed chunks exceeds the configured {n_chunks}")
    return prefix


def _assert_omega_classified(bundle: dict) -> list:
    """Refuse to run production while any amide candidate is unclassified.

    An unclassified candidate means a solute C-N bond was found that neither the residue rule nor
    the RDKit rule could name.  Scaling it or not scaling it are different Hamiltonians, so
    guessing would silently change the estimand.  Review it and either extend
    ``rest2.proline_like_residues`` / ``rest2.max_proline_ring_size`` or fix the input chemistry.
    """
    unknown = bundle.get("omega_unclassified_candidates") or []
    if unknown:
        lines = "\n".join(
            f"    bond {c['bond']}  {c['carbon_residue']}-{c['nitrogen_residue']}  {c['ambiguous']}"
            for c in unknown
        )
        raise ValueError(
            f"{len(unknown)} amide candidate(s) could not be classified as ordinary or "
            f"proline-like, so the REST2 Hamiltonian is not defined:\n{lines}\n"
            "Production is blocked until these are reviewed."
        )
    return [tuple(b) for b in bundle.get("omega_unscaled_bonds",
                                         bundle.get("omega_central_bonds", []))]


def run_md(cfg: dict, system_xml: Path, coords: Path, out_dir: Path, suffix: str) -> dict:
    """Stage (c): one free walker at ``production.md.scale_factor``, written in chunks.

    ``scale_factor = 1`` is the cold walker; anything below 1 is a REST2-scaled hot walker
    (``s = 0.25`` -> ``T_eff = 1200 K`` on the solute).  Nothing else differs between them, which is
    why there is one script rather than two.

    Chunking makes a long run restartable and analysable while it is still going: each chunk is a
    self-contained directory with its own trajectories, state log and end-of-chunk checkpoint.  A
    resumed run picks up at the first chunk without a ``done.json`` and does **not** re-minimise --
    re-minimising a continuing walker quenches the structure it had reached.
    """
    from openmm import unit

    out_dir = Path(out_dir)
    run_dir = out_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    mcfg = cfg["production"]["md"]
    ensemble = str(cfg["production"]["ensemble"]).upper()
    if ensemble != "NVT":
        raise ValueError(
            f"production.ensemble is {ensemble!r}; only NVT is implemented for the baseline.  "
            "See the ensemble note in docs/implementation/explicit_solvent/baseline_setups.md."
        )

    base, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])
    scale = float(mcfg["scale_factor"])
    label = str(mcfg.get("label") or ("cold" if scale == 1.0 else "hot"))
    omega = _assert_omega_classified(bundle)
    system = _scaled_system(base, cfg, n_solute, scale, omega)

    dt_fs = float(cfg["integrator"]["timestep_fs"])
    sim = _make_simulation(pdb.topology, system, cfg, int(mcfg["seed"]))

    chunk_steps = _steps(float(mcfg["chunk_ns"]) * 1000.0, dt_fs)
    n_chunks = int(round(float(mcfg["total_ns"]) / float(mcfg["chunk_ns"])))
    if not math.isclose(n_chunks * float(mcfg["chunk_ns"]), float(mcfg["total_ns"]), rel_tol=1e-9):
        raise ValueError(
            f"total_ns ({mcfg['total_ns']}) is not an integer multiple of chunk_ns "
            f"({mcfg['chunk_ns']})"
        )
    for name, interval_ps in (("all_atom_ps", cfg["production"]["report"]["all_atom_ps"]),
                              ("solute_ps", cfg["production"]["report"]["solute_ps"])):
        if chunk_steps % _steps(float(interval_ps), dt_fs) != 0:
            raise ValueError(
                f"chunk_ns is not a multiple of report.{name}; frames would not align to chunk "
                "boundaries and the first frame of each chunk would drift."
            )

    start_chunk = completed_prefix(run_dir, n_chunks)
    if start_chunk >= n_chunks:
        return {"status": "already-complete", "n_chunks": n_chunks, "output_dir": str(run_dir)}
    if start_chunk == 0:
        origin = _apply_coords(sim, coords, require_velocities=True)
    else:
        prev = run_dir / f"chunk_{start_chunk - 1:04d}" / "end.chk"
        sim.loadCheckpoint(str(prev))          # same System within a run: a checkpoint is valid
        origin = {"coords": str(prev), "kind": "checkpoint"}
    sim.currentStep = start_chunk * chunk_steps
    sim.context.setTime(start_chunk * chunk_steps * dt_fs * 1e-3 * unit.picosecond)

    print(
        f"[md:{label}] {n_chunks - start_chunk} chunk(s) of {mcfg['chunk_ns']} ns at "
        f"s = {scale:g}, dt = {dt_fs} fs, from {origin['coords']}",
        flush=True,
    )
    solute_atoms = list(range(n_solute))
    times = []
    for chunk in range(start_chunk, n_chunks):
        chunk_dir = run_dir / f"chunk_{chunk:04d}"
        _attach_chunk_reporters(sim, chunk_dir, cfg, dt_fs, solute_atoms)
        t0 = time.time()
        sim.step(chunk_steps)
        wall_s = time.time() - t0
        _close_chunk(sim)
        sim.saveCheckpoint(str(chunk_dir / "end.chk"))
        (chunk_dir / "done.json").write_text(
            json.dumps(
                {
                    "chunk": chunk, "steps": chunk_steps, "ns": float(mcfg["chunk_ns"]),
                    "scale_factor": scale, "label": label,
                    "simulated_time_ps": sim.context.getState().getTime().value_in_unit(
                        unit.picosecond
                    ),
                    "wall_seconds": round(wall_s, 1),
                    "ns_per_day": round(float(mcfg["chunk_ns"]) * 86400.0 / wall_s, 2),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        times.append(wall_s)
        print(
            f"[md:{label}] chunk {chunk + 1}/{n_chunks} done in {wall_s / 3600:.2f} h "
            f"({float(mcfg['chunk_ns']) * 86400.0 / wall_s:.1f} ns/day)",
            flush=True,
        )

    info = {
        "suffix": suffix, "label": label, "scale_factor": scale,
        "n_chunks": n_chunks, "chunk_ns": float(mcfg["chunk_ns"]),
        "total_ns": float(mcfg["total_ns"]), "timestep_fs": dt_fs,
        "input_coords": origin, "output_dir": str(run_dir),
        "mean_ns_per_day": (
            round(float(mcfg["chunk_ns"]) * 86400.0 / float(np.mean(times)), 2) if times else None
        ),
    }
    (out_dir / f"{suffix}_md.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_manifest(out_dir, f"{suffix}_md", cfg, {"result": info})
    return info


# ---------------------------------------------------------------------------------------------
# Step 8 -- REST2 replica exchange
# ---------------------------------------------------------------------------------------------
def _pre_exchange_relaxation(simulations, coords: Path, cfg: dict, run_dir: Path,
                             scale_factors, temperature: float, dt_fs: float, seed: int) -> dict:
    """Propagate each replica under ITS OWN Hamiltonian before the first exchange, and discard it.

    Every replica starts from the same equilibrated coordinates, and those were equilibrated at
    ``s = 1``.  A replica at ``s < 1`` is therefore *not* in its own ensemble at step 0: its solute
    is suddenly softer than the configuration it holds.  Exchanging immediately would feed that
    transient straight into the swap statistics and into the trajectories.

    So each replica is run for ``production.remd.equilibration_ps`` with **no reporters and no
    exchange attempts**, and the result is thrown away.  This is *pre-exchange relaxation*, not a
    claim of equilibrium -- 10 ps settles the local solvent response to the scaled solute, nothing
    more.

    Restart-safe by construction: a completion record is written only after every replica's
    checkpoint is on disk, so a partial relaxation is detected and simply redone.  Redoing it is
    safe precisely because this runs only when no production output exists yet.
    """
    from openmm import unit

    relax_ps = float(cfg["production"]["remd"]["equilibration_ps"])
    relax_dir = run_dir / "_relaxation"
    record_path = relax_dir / "relaxation.json"
    checkpoints = [relax_dir / f"replica_{r:02d}.chk" for r in range(len(simulations))]

    if relax_ps <= 0:
        return {"performed": False, "reason": "production.remd.equilibration_ps = 0"}

    if record_path.exists() and all(c.exists() for c in checkpoints):
        for sim, chk in zip(simulations, checkpoints):
            sim.loadCheckpoint(str(chk))
        info = json.loads(record_path.read_text(encoding="utf-8"))
        info["reused"] = True
        print(f"[remd] reusing the stored {relax_ps:g} ps pre-exchange relaxation "
              f"({relax_dir}); it is not repeated", flush=True)
        return info

    if relax_dir.exists():
        print(f"[remd] {relax_dir} is incomplete (no completion record, or a checkpoint is "
              "missing); redoing the relaxation from scratch -- safe here because no production "
              "output exists yet", flush=True)
        for c in checkpoints:
            c.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)
    relax_dir.mkdir(parents=True, exist_ok=True)

    steps = _steps(relax_ps, dt_fs)
    per_replica, t0 = [], time.time()
    for r, sim in enumerate(simulations):
        _apply_coords(sim, coords)
        # deterministic and DISTINCT per replica: same positions, independent momenta
        vel_seed = int(seed) + 1000 + r
        sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, vel_seed)
        u0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.reporters.clear()                     # no production reporters during relaxation
        sim.step(steps)
        u1 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        sim.saveCheckpoint(str(checkpoints[r]))
        per_replica.append({
            "replica": r, "scale_factor": float(scale_factors[r]),
            "velocity_seed": vel_seed,
            "potential_before_kj_mol": round(u0, 3), "potential_after_kj_mol": round(u1, 3),
            "delta_kj_mol": round(u1 - u0, 3),
            "checkpoint": str(checkpoints[r]),
        })
        print(f"[remd] relax replica {r:02d} s={scale_factors[r]:.4f}: "
              f"U {u0:.1f} -> {u1:.1f} kJ/mol", flush=True)

    info = {
        "performed": True, "reused": False,
        "discarded_ps": relax_ps, "discarded_steps_per_replica": steps,
        "timestep_fs": dt_fs, "temperature_k": temperature,
        "exchanges_attempted": 0, "reporters_attached": False,
        "wall_seconds": round(time.time() - t0, 1),
        "per_replica": per_replica,
        "note": "pre-exchange relaxation, DISCARDED; production time and step numbering are reset "
                "to zero afterwards, so production logs exclude it",
    }
    record_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def run_rest2_remd(cfg: dict, system_xml: Path, coords: Path, out_dir: Path,
                   suffix: str) -> dict:
    """Stage (d): REST2-REMD -- N replicas, neighbour exchange, chunked per replica.

    The ladder comes from ``production.remd.scale_factors`` if given, otherwise it is built from
    ``rest2.ladder`` (``s_cold``, ``s_hot``, ``n_rungs``, ``interp``).  Replica 0 is the physical
    rung (``s = 1``) by convention.

    The exchange criterion and the even/odd neighbour schedule are reused from
    ``escort_ais.methods.md_run`` -- the same code the implicit-solvent references were produced
    with, so acceptance and round-trip statistics are comparable across solvents.

    ``<suffix>_exchange_attempts.csv`` carries both ``replica_i``/``replica_j`` (the schema
    ``md_run.run_remd`` writes) and ``i``/``j`` aliases, plus ``walker_at_replica_XX``, so
    ``analysis/remd_reliability.py`` computes the four reference-reliability probes (worst-pair
    acceptance, round trips, time-halves drift, per-replica spread) without a converter.

    All replicas share one process and one GPU.  ONE process per GPU: two processes sharing a card
    without CUDA MPS run ~3.7x slower each.
    """
    from openmm import unit

    from escort_ais.methods.md_run import attempt_rest2_exchange, exchange_pairs
    from escort_ais.systems.topology_prep import resolve_remd_scale_ladder

    out_dir = Path(out_dir)
    run_dir = out_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["production"]["remd"]
    if str(cfg["production"]["ensemble"]).upper() != "NVT":
        raise ValueError(
            "REST2-REMD in this baseline is NVT.  An NPT ladder needs a PV term in the "
            "acceptance criterion, which attempt_rest2_exchange does not include."
        )

    base, pdb, bundle = _load_bundle(system_xml)
    n_solute = int(bundle["n_solute_atoms"])
    omega = _assert_omega_classified(bundle)

    requested = rcfg["scale_factors"]
    if requested is None:
        lad = cfg["rest2"]["ladder"]
        requested = rest2_ladder(
            float(lad["s_cold"]), float(lad["s_hot"]), int(lad["n_rungs"]), str(lad["interp"])
        )
    temperature = float(cfg["integrator"]["temperature_k"])
    scale_factors, t_eff = resolve_remd_scale_ladder(list(requested), temperature)
    n_replicas = len(scale_factors)
    dt_fs = float(cfg["integrator"]["timestep_fs"])

    simulations = []
    for r, sc in enumerate(scale_factors):
        system = _scaled_system(base, cfg, n_solute, sc, omega)
        simulations.append(_make_simulation(pdb.topology, system, cfg, int(rcfg["seed"]) + r))

    exchange_steps = _steps(float(rcfg["exchange_interval_ps"]), dt_fs)
    chunk_steps = _steps(float(rcfg["chunk_ns"]) * 1000.0, dt_fs)
    if chunk_steps % exchange_steps != 0:
        raise ValueError("remd.chunk_ns must be a whole number of exchange intervals")
    n_chunks = int(round(float(rcfg["total_ns_per_replica"]) / float(rcfg["chunk_ns"])))
    rounds_per_chunk = chunk_steps // exchange_steps

    replica_dirs = [run_dir / f"replica_{r:02d}" for r in range(n_replicas)]
    for d in replica_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # every replica must share the SAME completed prefix: they advance in lockstep between
    # exchange rounds, so a ragged set means one process died mid-round and the ladder's state is
    # not reconstructible from what survived
    prefixes = {r: completed_prefix(d, n_chunks) for r, d in enumerate(replica_dirs)}
    start_chunk = min(prefixes.values())
    if len(set(prefixes.values())) > 1:
        raise ValueError(
            f"replicas have different completed prefixes {prefixes}.  REST2 replicas advance in "
            "lockstep between exchange rounds, so this cannot be resumed consistently.  Delete "
            f"every replica's chunk_{start_chunk:04d} and later, then resume."
        )
    if start_chunk >= n_chunks:
        return {"status": "already-complete", "n_chunks": n_chunks, "output_dir": str(run_dir)}

    log_path = out_dir / f"{suffix}_exchange_attempts.csv"
    fieldnames = [
        "step", "time_ps", "phase", "replica_i", "replica_j", "i", "j",
        "effective_temperature_i_k", "effective_temperature_j_k",
        "scale_factor_i", "scale_factor_j",
        "energy_i_on_i_kj_mol", "energy_j_on_j_kj_mol",
        "energy_i_on_j_kj_mol", "energy_j_on_i_kj_mol",
        "delta_kj_mol", "log_acceptance", "accepted",
    ] + [f"walker_at_replica_{r:02d}" for r in range(n_replicas)]

    if start_chunk == 0:
        walker_by_replica = list(range(n_replicas))
        relaxation = _pre_exchange_relaxation(
            simulations, coords, cfg, run_dir, scale_factors, temperature, dt_fs,
            int(rcfg["seed"]),
        )
        log_mode = "w"
    else:
        for r, sim in enumerate(simulations):
            sim.loadCheckpoint(str(replica_dirs[r] / f"chunk_{start_chunk - 1:04d}" / "end.chk"))
        # the exchange log must end exactly at the production boundary the chunks reached, or a
        # resume would either duplicate rows or leave a silent hole in the swap history
        rows = list(csv.DictReader(log_path.open())) if log_path.exists() else []
        if not rows:
            raise ValueError(f"{log_path} has no rows; cannot restore the rung occupancy")
        boundary = start_chunk * chunk_steps
        keep = [r for r in rows if int(r["step"]) <= boundary]
        if len(keep) != len(rows):
            print(f"[remd] exchange log ran past the completed chunks ({len(rows)} rows, "
                  f"{len(keep)} at or before step {boundary}); truncating to the chunk boundary "
                  "so the resume neither duplicates nor skips a round", flush=True)
            with log_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(keep)
        if int(keep[-1]["step"]) != boundary:
            raise ValueError(
                f"{log_path} ends at step {keep[-1]['step']} but the completed chunks end at "
                f"{boundary}.  Refusing to resume from an inconsistent exchange history."
            )
        last = keep[-1]
        walker_by_replica = [int(last[f"walker_at_replica_{r:02d}"]) for r in range(n_replicas)]
        relaxation = {"performed": False, "reason": "resume: relaxation belongs to the fresh start"}
        log_mode = "a"

    for sim in simulations:
        sim.currentStep = start_chunk * chunk_steps
        sim.context.setTime(start_chunk * chunk_steps * dt_fs * 1e-3 * unit.picosecond)

    # exchange RNG: re-seeded on resume.  That is unbiased for Metropolis accept/reject, so a
    # continuation is faithful distributionally rather than bit-exact.
    rng = np.random.default_rng(int(rcfg["seed"]) + 977 * (start_chunk + 1))
    beta0 = 1.0 / (0.008314462618 * temperature)
    solute_atoms = list(range(n_solute))

    print(
        f"[remd] {n_replicas} replicas, s = {scale_factors}, T_eff = "
        f"{[round(t) for t in t_eff]} K, exchange every {rcfg['exchange_interval_ps']} ps, "
        f"{n_chunks - start_chunk} chunk(s) of {rcfg['chunk_ns']} ns to go",
        flush=True,
    )

    n_attempts = n_accepted = 0
    with log_path.open(log_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if log_mode == "w":
            writer.writeheader()
        phase = start_chunk * rounds_per_chunk
        for chunk in range(start_chunk, n_chunks):
            for r, sim in enumerate(simulations):
                _attach_chunk_reporters(
                    sim, replica_dirs[r] / f"chunk_{chunk:04d}", cfg, dt_fs, solute_atoms
                )
            t0 = time.time()
            for _ in range(rounds_per_chunk):
                for sim in simulations:
                    sim.step(exchange_steps)
                for i, j in exchange_pairs(n_replicas, phase):
                    result = attempt_rest2_exchange(simulations[i], simulations[j], beta0, rng)
                    n_attempts += 1
                    if result["accepted"]:
                        n_accepted += 1
                        walker_by_replica[i], walker_by_replica[j] = (
                            walker_by_replica[j], walker_by_replica[i],
                        )
                    step = simulations[0].currentStep
                    writer.writerow(
                        {
                            "step": step, "time_ps": step * dt_fs * 1e-3, "phase": phase % 2,
                            "replica_i": i, "replica_j": j, "i": i, "j": j,
                            "effective_temperature_i_k": t_eff[i],
                            "effective_temperature_j_k": t_eff[j],
                            "scale_factor_i": scale_factors[i],
                            "scale_factor_j": scale_factors[j],
                            **{k: v for k, v in result.items() if k != "accepted"},
                            "accepted": int(result["accepted"]),
                            **{
                                f"walker_at_replica_{r:02d}": w
                                for r, w in enumerate(walker_by_replica)
                            },
                        }
                    )
                phase += 1
                fh.flush()
            wall_s = time.time() - t0
            for r, sim in enumerate(simulations):
                _close_chunk(sim)
                cdir = replica_dirs[r] / f"chunk_{chunk:04d}"
                sim.saveCheckpoint(str(cdir / "end.chk"))
                (cdir / "done.json").write_text(
                    json.dumps(
                        {
                            "chunk": chunk, "replica": r, "scale_factor": scale_factors[r],
                            "effective_temperature_k": t_eff[r],
                            "wall_seconds": round(wall_s, 1),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(
                f"[remd] chunk {chunk + 1}/{n_chunks} in {wall_s / 3600:.2f} h; acceptance "
                f"{n_accepted / max(1, n_attempts):.3f}",
                flush=True,
            )

    info = {
        "suffix": suffix,
        "n_replicas": n_replicas,
        "scale_factors": scale_factors,
        "effective_temperatures_k": t_eff,
        "ladder_source": "production.remd.scale_factors" if rcfg["scale_factors"] is not None
                         else f"rest2.ladder ({cfg['rest2']['ladder']})",
        "exchange_interval_ps": float(rcfg["exchange_interval_ps"]),
        "pre_exchange_relaxation": relaxation,
        "n_chunks": n_chunks,
        "chunk_ns": float(rcfg["chunk_ns"]),
        "total_ns_per_replica": float(rcfg["total_ns_per_replica"]),
        "acceptance_fraction": n_accepted / max(1, n_attempts),
        "n_exchange_attempts": n_attempts,
        "exchange_log": str(log_path),
        "output_dir": str(run_dir),
    }
    (out_dir / f"{suffix}_rest2.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(out_dir, f"{suffix}_rest2", cfg, {"result": info})
    return info
