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
        "nonbonded_cutoff_nm": 1.0,
        # Headroom above OpenMM's hard minimum-image limit, in nm, applied ONLY when the box has
        # to be grown to fit the cutoff.  Growing to exactly 2*cutoff leaves the box sitting on
        # the limit, and the first NPT contraction then crosses it: OpenMM aborts with "The
        # periodic box size has decreased to less than twice the nonbonded cutoff".  Measured on a
        # 12-heavy-atom macrocycle, that abort happened on the first NPT step, twice, with two
        # different cutoffs -- the box is grown to the limit by construction, so the failure is
        # deterministic rather than unlucky.  0.10 nm is ~5 % of a 2.0 nm requirement, which
        # comfortably exceeds equilibration-scale contraction.  0.0 restores the old behaviour and
        # must be set deliberately.
        "minimum_image_margin_nm": 0.10,          # electrostatics AND vdW
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
            # Both are INPUTS. total_ns is derived (n_chunks * chunk_ns) and reported; it is never
            # an input and is never read back to recover n_chunks.
            "n_chunks": 10,
            "chunk_ns": 100.0,
            "seed": None,
            "label": "cold",                 # free-text, recorded and used in log lines
        },
        "remd": {
            "n_chunks": 10,
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
        "omega_exclusion": True,             # leave ORDINARY amide omega torsions unscaled
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


# ---------------------------------------------------------------------------------------------
# Chunk planning
#
# The number of chunks is an INPUT. It used to be int(round(total_ns / chunk_ns)), which silently
# accepted a total that was not a whole number of chunks and then ran a different length than the
# manifest declared. Totals are now derived from the plan and are outputs only -- nothing ever
# reconstructs n_chunks from them.
# ---------------------------------------------------------------------------------------------

def resolve_chunk_plan(n_chunks: Any, chunk_ns: Any, *, timestep_fs: float, where: str,
                       exchange_interval_ps: Optional[float] = None) -> dict[str, Any]:
    """Validate an explicit (n_chunks, chunk_ns) plan and derive its totals.

    `bool` is rejected explicitly: `True` is an `int` in Python and would otherwise be accepted as
    "one chunk", which is a silent misreading of a configuration error.
    """
    if isinstance(n_chunks, bool) or not isinstance(n_chunks, int):
        raise ValueError(
            f"{where}.n_chunks must be an integer, got {n_chunks!r} ({type(n_chunks).__name__}). "
            "The number of chunks is an input, not a rounded quotient."
        )
    if n_chunks <= 0:
        raise ValueError(f"{where}.n_chunks must be greater than zero, got {n_chunks}")

    try:
        chunk = float(chunk_ns)
    except (TypeError, ValueError):
        raise ValueError(f"{where}.chunk_ns must be a number, got {chunk_ns!r}") from None
    if not math.isfinite(chunk) or chunk <= 0:
        raise ValueError(f"{where}.chunk_ns must be finite and positive, got {chunk_ns!r}")

    steps_per_chunk = chunk * 1e6 / float(timestep_fs)
    if abs(steps_per_chunk - round(steps_per_chunk)) > 1e-6:
        raise ValueError(
            f"{where}.chunk_ns ({chunk} ns) is not a whole number of {timestep_fs} fs steps "
            f"({steps_per_chunk:.6f}). A rounded chunk runs a different length than it declares."
        )
    steps_per_chunk = int(round(steps_per_chunk))

    if exchange_interval_ps is not None:
        per_chunk = chunk * 1000.0 / float(exchange_interval_ps)
        if abs(per_chunk - round(per_chunk)) > 1e-6:
            raise ValueError(
                f"{where}.chunk_ns ({chunk} ns) is not a whole number of exchange intervals "
                f"({exchange_interval_ps} ps): {per_chunk:.6f}. A chunk boundary that falls "
                "mid-interval would drop or duplicate an exchange attempt across a resume."
            )

    return {
        "n_chunks": int(n_chunks),
        "chunk_ns": chunk,
        "steps_per_chunk": steps_per_chunk,
        # DERIVED, and reported only. Never read back to recover n_chunks.
        "total_ns": round(n_chunks * chunk, 12),
    }


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
# Manifest -> resolved configuration
# ---------------------------------------------------------------------------------------------

from .schemas import ExperimentManifest, ManifestError, SystemManifest

#: Which `solute_kind` each portable route implies. The mapping is one-way and total: a portable
#: manifest never says `auto`, so the baseline's route-resolution never has to guess.
ROUTE_TO_SOLUTE_KIND = {"smiles": "ligand", "pdb": "peptide"}


def resolve_config(
    system: SystemManifest,
    experiment: ExperimentManifest,
    *,
    platform: str = "CPU",
    device: Optional[str] = None,
    omega_exclusion: Optional[bool] = None,
) -> dict[str, Any]:
    """The fully resolved config for this (system, experiment, machine).

    Machine choices (`platform`, `device`) are applied last and are deliberately NOT part of the
    config hash: the same calculation on CPU and on CUDA is the same configuration.

    `omega_exclusion` is the opposite: it selects which torsions REST2 scales, so it changes the
    Hamiltonian. It is a build-defining setting, it enters the config hash and the continuation
    contract, and a run cannot be resumed across a change to it. `None` means "leave whatever the
    manifests resolved to"; True/False is an explicit CLI override and is recorded as one.
    """
    cfg = copy.deepcopy(DEFAULTS)
    sdoc, edoc = system.doc, experiment.doc

    # ---- identity and route -------------------------------------------------------------------
    cfg["system"]["slug"] = system.system_id
    cfg["system"]["solute_kind"] = ROUTE_TO_SOLUTE_KIND[system.route]
    # enforced, not documented: a pdb handed to a smiles-route manifest must fail before any force
    # field is built, rather than quietly becoming an ff19SB peptide calculation
    cfg["system"]["require_input_route"] = system.route

    par = sdoc["parameterization"]
    cfg["forcefield"]["water"] = par["water_forcefield"]
    if system.route == "smiles":
        cfg["forcefield"]["ligand"] = par["small_molecule_forcefield"]
        cfg["forcefield"]["ligand_charge_method"] = par["charge_method"]
    else:
        cfg["forcefield"]["protein"] = par["protein_forcefield"]

    for key, value in (sdoc.get("solvation") or {}).items():
        if key not in cfg["solvation"]:
            raise ManifestError(
                f"{system.source}: solvation.{key} is not a recognised solvation setting; "
                f"known keys are {sorted(cfg['solvation'])}"
            )
        cfg["solvation"][key] = value

    # ---- experiment controls -------------------------------------------------------------------
    cfg["run"]["seed"] = experiment.master_seed
    cfg["run"]["name"] = system.system_id

    integ = edoc["integrator"]
    cfg["integrator"]["kind"] = integ["kind"]
    cfg["integrator"]["temperature_k"] = float(integ["temperature_k"])
    cfg["integrator"]["timestep_fs"] = float(integ["timestep_fs"])

    rest2 = edoc["rest2"]
    remd = cfg["production"]["remd"]
    remd["scale_factors"] = [float(v) for v in rest2["scale_factors"]]
    remd["exchange_interval_ps"] = float(rest2["exchange_interval_ps"])
    remd["equilibration_ps"] = float(rest2["relaxation_ps"])
    # The external manifest carries the same two canonical fields; they map straight through, so
    # there is one source of truth rather than a manifest total and a config total to keep in step.
    remd["n_chunks"] = int(rest2["n_chunks"])
    remd["chunk_ns"] = float(rest2["chunk_ns"])
    cfg["production"]["precision"] = edoc["platform"]["precision"]

    for key, value in (edoc.get("equilibration") or {}).items():
        if key not in cfg["equilibration"]:
            raise ManifestError(
                f"{experiment.source}: equilibration.{key} is not a recognised setting"
            )
        cfg["equilibration"][key] = value

    # An explicit escape hatch for the rest of the baseline tree. Unknown keys raise (deep_merge
    # rejects them), so a typo cannot leave a baseline value in force while the manifest claims
    # otherwise -- which is the failure this whole layer exists to prevent.
    overrides = edoc.get("overrides")
    if overrides:
        cfg = _deep_merge(cfg, copy.deepcopy(overrides))

    # ---- the omega-exclusion override ----------------------------------------------------------
    # Applied after the manifests and before the seeds, and recorded as an override so the resolved
    # configuration says where the value came from. This is NOT a machine choice: it decides which
    # torsions are scaled, so two runs differing in it are different Hamiltonians.
    if omega_exclusion is not None:
        cfg["rest2"]["omega_exclusion"] = bool(omega_exclusion)
        cfg.setdefault("_overrides", {})["rest2.omega_exclusion"] = {
            "value": bool(omega_exclusion),
            "source": "command line",
        }

    # ---- machine choices, applied last and excluded from the config hash -----------------------
    cfg["production"]["platform"] = platform
    cfg["production"]["device_index"] = device

    _resolve_seeds(cfg)
    _check_exchange_divisibility(cfg, experiment)
    return cfg


def _resolve_seeds(cfg: dict) -> None:
    """Derive the per-stage seeds from the master seed, exactly as `load_config` does.

    Duplicated deliberately rather than calling `load_config`: that function's entry point is a
    JSON file, and routing a manifest through a temporary file to obtain seed derivation would put
    a filesystem round-trip in the middle of a pure transformation.
    """
    master = int(cfg["run"]["seed"])
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


def _check_exchange_divisibility(cfg: dict, experiment: ExperimentManifest) -> None:
    """Fail here, with the arithmetic shown, rather than deep inside the REMD driver.

    `run_rest2_remd` requires a chunk to be a whole number of exchange intervals so that every
    replica reaches a chunk boundary on the same exchange round. Violating it raises there too,
    but by then the System has been built and the user has paid for parameterisation.
    """
    dt_fs = float(cfg["integrator"]["timestep_fs"])
    remd = cfg["production"]["remd"]
    exchange_steps = round(float(remd["exchange_interval_ps"]) * 1000.0 / dt_fs)
    chunk_steps = round(float(remd["chunk_ns"]) * 1e6 / dt_fs)
    if exchange_steps <= 0 or chunk_steps <= 0:
        raise ManifestError(
            f"{experiment.source}: exchange interval and chunk must each be at least one step at "
            f"{dt_fs} fs"
        )
    if chunk_steps % exchange_steps != 0:
        raise ManifestError(
            f"{experiment.source}: rest2.chunk_ns must be a whole number of exchange intervals.\n"
            f"  chunk    {remd['chunk_ns']} ns = {chunk_steps} steps at {dt_fs} fs\n"
            f"  exchange {remd['exchange_interval_ps']} ps = {exchange_steps} steps\n"
            f"  {chunk_steps} / {exchange_steps} = {chunk_steps / exchange_steps:.4f}, not an "
            "integer"
        )


def exchange_rounds(cfg: dict) -> int:
    """How many exchange attempts the configured plan will make, in total.

    Derived from the plan (n_chunks x steps_per_chunk), never from a stored total: a total that had
    drifted out of step with the plan would silently change this count.
    """
    dt_fs = float(cfg["integrator"]["timestep_fs"])
    remd = cfg["production"]["remd"]
    exchange_steps = round(float(remd["exchange_interval_ps"]) * 1000.0 / dt_fs)
    plan = resolve_chunk_plan(remd["n_chunks"], remd["chunk_ns"], timestep_fs=dt_fs,
                              where="production.remd",
                              exchange_interval_ps=float(remd["exchange_interval_ps"]))
    return int(plan["n_chunks"] * plan["steps_per_chunk"] // exchange_steps)
