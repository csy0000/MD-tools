"""The default system and protocol settings, defined once.

`sys-config`, `md-config` and `show-default` all read these. There is no second copy: the reason
this repository accumulated drift guards is that the same fact used to be declared in five places,
and a change would update two of them.

Values are plain Python data that serialise to the YAML a user edits. Nothing here validates -- see
`config.py` -- and nothing here is versioned per-profile. A user who wants different settings edits
the YAML.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1
ENGINE = "openmm"
ENGINE_VERSION = "8.6.0"

#: Canonical spellings. Input is accepted case-insensitively and written back in these forms.
METHODS = ("cMD", "REST2", "AIS")
SOLVENTS = ("TIP3P", "OPC", "GBn2")

#: What `--solvent` selects when nothing is asked for. TIP3P is the method-development default:
#: see `docs/md-defaults-scientific-rationale.md`.
DEFAULT_SOLVENT = "TIP3P"


def canonical_method(name: str) -> str:
    for canonical in METHODS:
        if str(name).lower() == canonical.lower():
            return canonical
    raise ValueError(f"unknown method {name!r}; expected one of {', '.join(METHODS)}")


def canonical_solvent(name: str) -> str:
    for canonical in SOLVENTS:
        if str(name).lower() == canonical.lower():
            return canonical
    raise ValueError(f"unknown solvent {name!r}; expected one of {', '.join(SOLVENTS)}")


def is_implicit(solvent: str) -> bool:
    return canonical_solvent(solvent) == "GBn2"


#: The forward AIS path, and how many points along it are observed. tau = 0.5 is the scaled end
#: (s = 0.25) and tau = 0 is the physical Hamiltonian; 21 observations are endpoint-inclusive.
AIS_TAU_START = 0.5
AIS_TAU_END = 0.0
AIS_OBSERVATIONS = 21

#: The two explicit-solvent combinations this repository supports, named by the OpenMM resources
#: that are actually loaded rather than by a family label.
#:
#: This is a two-entry lookup, not a profile registry: there is no inheritance, no versioning and
#: no third layer. `--solvent` picks one row, the row is written into ordinary editable YAML, and
#: everything after that is the user's file.
#:
#: TIP3P is the default. ff14SB was developed and benchmarked in TIP3P, OpenFF Sage's valence and
#: vdW parameters were fit to condensed-phase and gas-phase physical-property data (not to protein
#: binding affinities, and not exclusively against TIP3P), and the combination ff14SB + Sage + TIP3P
#: is the one with published workflow-level protein-ligand use. ff19SB + OPC is the alternative:
#: ff19SB's amino-acid-specific CMAPs were fit with OPC, so the pair is internally consistent, but
#: it is a heavier and slower water model and the joint ff19SB/Sage/OPC combination has no
#: published combination-level benchmark. See `docs/md-defaults-scientific-rationale.md`.
EXPLICIT_COMBINATIONS = {
    "TIP3P": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"},
    "OPC": {"protein": "amber19-all.xml", "water": "amber19/opc.xml"},
}
#: Kept for readers of older records: the resource the 0.3.x default named.
EXPLICIT_PROTEIN_FORCEFIELD = EXPLICIT_COMBINATIONS["OPC"]["protein"]

#: Substrings that identify which supported family a hand-written resource name belongs to.
#:
#: `sys-config` writes the qualified resource, but the file is editable YAML and a user may write
#: `amber14/protein.ff14SB.xml`, `ff14SB.xml` or `amber/ff14SB.xml` instead. Matching on family
#: markers rather than on an exact string means `config._check_explicit_pairing` catches a crossed
#: pair however it was spelled, while a name belonging to NEITHER family is left alone -- somebody
#: loading a force field this repository does not ship is doing something deliberate, and refusing
#: it here would be refusing a choice this table has no opinion about.
PROTEIN_FAMILY_MARKERS = {
    "TIP3P": ("ff14sb", "amber14"),
    "OPC": ("ff19sb", "amber19"),
}
WATER_FAMILY_MARKERS = {
    "TIP3P": ("tip3p",),
    "OPC": ("opc",),
}
#: Implicit GBn2: ff14SB, the force field GBn2 was developed and validated against. A tleap
#: resource, because the implicit route builds its topology with tleap rather than an OpenMM XML.
IMPLICIT_PROTEIN_FORCEFIELD = "leaprc.protein.ff14SB"
#: Protein force fields known to be mismatched with a GB implicit-solvent model.
GB_INCOMPATIBLE_PROTEIN = ("ff19SB", "amber19")

#: The small-molecule force field, by the label a user writes and the resource the toolkit loads.
#: Sage 2.2.1 is the current Sage release in the pinned environment; `sysgen._openff_name` maps the
#: label onto `openff-2.2.1`, and `openforcefields` ships `openff-2.2.1.offxml`.
LIGAND_FORCEFIELD = "sage-2.2.1"

#: Solute-to-box clearance requested from `Modeller.addSolvent`. 1.5 nm is the default; 2.0 nm is
#: the conservative option for unfolded or unusually flexible solutes and for enhanced sampling
#: expected to expand the solute. Neither number is a guarantee about a future conformation -- the
#: built-system cutoff/minimum-image gate in `solvation._resolve_box` is what is actually enforced.
DEFAULT_PADDING_NM = 1.5
CONSERVATIVE_PADDING_NM = 2.0

#: `MonteCarloBarostat` volume-move attempt interval, in integration steps. OpenMM's own default.
#: 25 steps is 0.05 ps at the 2 fs baseline and 0.10 ps with the optional 4 fs timestep.
DEFAULT_BAROSTAT_FREQUENCY_STEPS = 25

#: The hydrogen mass the optional performance setting repartitions to, with the timestep it is
#: paired with. Never a default: `constraints.hydrogen_mass_amu` stays null and `timestep_fs` 2.0.
HMR_HYDROGEN_MASS_AMU = 3.024
HMR_TIMESTEP_FS = 4.0


def ais_defaults() -> dict[str, Any]:
    """The AIS block: a switching path, where its configurations come from, and what is observed.

    AIS anneals the REST2 Hamiltonian from `tau_start` to `tau_end` while the coordinates propagate,
    and records the nonequilibrium work. The scaling is the SAME decomposition REST2 uses, so a
    fixed-tau cMD walker, a REST2 rung and an AIS path at the same tau are the same Hamiltonian:
    `s = (1 - tau)^2` for solute-solute terms and `sqrt(s) = 1 - tau` for solute-environment terms,
    with torsions about an omega bond left unscaled. Temperature and beta come from
    `common.temperature_kelvin` and do not change along the path -- this is Hamiltonian switching,
    not temperature annealing.

    The `null` fields are REQUIRED USER INPUT, not silent defaults. AIS starts from an existing
    equilibrium trajectory, and there is no defensible guess for which one, which part of it, or how
    long the switch should take. `md-gen` names the missing field rather than choosing.
    """
    return {
        "path": {
            # The only path type implemented. Named so a configuration asking for something else is
            # refused rather than quietly getting this one.
            "type": "rest2_tau",
            # Linear in tau, so sqrt(s) is linear and s is quadratic along the path.
            "tau_start": AIS_TAU_START,
            "tau_end": AIS_TAU_END,
            "interpolation": "linear",
            # The same omega bonds and the same enhanced region cMD and REST2 use, read from
            # inputs/solute.yaml. Changing either makes the path a different Hamiltonian from the
            # equilibrium ensemble that seeded it.
            "omega_exclusion": True,
            "enhanced_region": "solute",
            # REQUIRED. How long the switch takes. Work is path-length dependent, so there is no
            # default that is not a scientific choice. Its step count must divide exactly by
            # parameter_update_interval_steps, and the resulting update count by
            # output.number_of_observations - 1.
            "switching_duration_ps": None,
            # How often the Hamiltonian changes, in integration STEPS. 1 is the finest schedule
            # this can express.
            "parameter_update_interval_steps": 1,
        },
        "source": {
            # REQUIRED. An existing equilibrium trajectory at tau_start: a fixed-tau cMD run or one
            # REST2 rung. Resolved relative to the generated MD/ project, and the resolved path is
            # recorded.
            "trajectory": None,
            # null uses inputs/topology.pdb, the topology the System was built from. The resolved
            # choice is recorded either way.
            "topology": None,
            # REQUIRED, and INCLUSIVE at both ends. Frames outside [start, end] are not eligible.
            "start_time_ps": None,
            "end_time_ps": None,
            # Only needed when the trajectory has no companion runtime record to take frame times
            # from. A frame INDEX is never treated as a time.
            "first_frame_time_ps": None,
            "frame_interval_ps": None,
            # REQUIRED. Each is one independent switching path with its own directory and DCD.
            "number_of_trajectories": None,
            "selection": "uniform_random",
            "allow_sampling_with_replacement": False,
        },
        "output": {
            # 21 points = 20 equal intervals plus the starting configuration.
            "number_of_observations": AIS_OBSERVATIONS,
            "include_start": True,
            "include_end": True,
            "coordinates": "whole_system",
        },
        "execution": {
            "platform": "CUDA",
            # `auto` uses every visible CUDA device; a list pins particular ones.
            "gpu_devices": "auto",
        },
    }


def sys_defaults(*, peptide: bool = True, solvent: str = DEFAULT_SOLVENT) -> dict[str, Any]:
    """System preparation settings.

    Both the explicit and implicit blocks are written so the file documents what the other option
    would look like, but only the one matching `solvent` is used when the configuration is
    resolved -- `config.resolve_sys_config` drops the other and says so.
    """
    solvent = canonical_solvent(solvent)
    implicit = is_implicit(solvent)
    # An implicit file still documents what the explicit block would look like, and it documents
    # the DEFAULT explicit combination rather than whichever one happens to sort first.
    combination = EXPLICIT_COMBINATIONS[DEFAULT_SOLVENT if implicit else solvent]
    document = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "solute": {
            "peptide": bool(peptide),
            # Sage 2.2.1 and standard AM1-BCC (AmberTools sqm). `am1bcc_nagl` is a graph network
            # TRAINED to predict AM1-BCC ELF10 charges -- close but not that calculation -- and is
            # selected explicitly or not at all.
            "ligand_forcefield": LIGAND_FORCEFIELD,
            "ligand_charge_method": "am1bcc",
        },
        "forcefield": {
            # The protein force field is chosen WITH the solvation model, not independently.
            #
            # Explicit: ff14SB with TIP3P is the default -- ff14SB's dihedral refit and its
            # published benchmarks are TIP3P work -- and ff19SB with OPC is the alternative, the
            # pairing ff19SB's amino-acid-specific CMAPs were fit alongside.
            #
            # Implicit: ff14SB again, because GBn2 was developed and validated in the
            # ff99SB/ff14SB lineage (Nguyen, Roe & Simmerling, JCTC 2013) and no GB model has been
            # reparameterised against ff19SB's CMAPs.
            #
            # The value is also in the namespace the builder for this route consumes: an OpenMM
            # XML for the explicit route, a tleap leaprc for the implicit one.
            "protein": (IMPLICIT_PROTEIN_FORCEFIELD if implicit else combination["protein"]),
            # The QUALIFIED OpenMM resource, which is the file `ForceField()` is given. The
            # amber14/amber19 copies carry the Na+/Cl- ion templates `addSolvent` needs; the
            # top-level `tip3p.xml` does not.
            "water": None if implicit else combination["water"],
        },
        "solvent": {
            "model": solvent if not implicit else DEFAULT_SOLVENT,
            # OpenMM's padding semantics: width = max(2R + padding, 2 * padding), so this is a
            # requested solute-to-BOX clearance, not the box width and not the solute-to-periodic-
            # copy distance. `sys-gen` records all four quantities; raise this to 2.0 for an
            # unfolded or unusually flexible solute, or when enhanced sampling is expected to
            # expand it.
            "padding_nm": DEFAULT_PADDING_NM,
            "box_shape": "dodecahedron",
            "ionic_strength_molar": 0.15,
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
            "cutoff_nm": 1.0,
        },
        "implicit_solvent": {
            "model": "GBn2",
            "radii": "mbondi3",
            # The ACE surface-area nonpolar term. False matches Amber's igb=8 with gbsa=0, which
            # is the context GBn2's parameters were fit in; OpenMM's implicit/gbn2.xml turns it on
            # by default. The two differ by ~16 kJ/mol (~6 kT) on ACE-ALA-NME, so this is a
            # modelling choice and is stated rather than inherited.
            "nonpolar_sasa": False,
        },
        "constraints": {
            "type": "HBonds",
            "rigid_water": True,
            # null: hydrogens keep the masses the force field gave them, and the timestep stays at
            # 2 fs. This is the baseline, not a placeholder. Set to 3.024 for HMR, and raise
            # timestep_fs to 4.0 with it -- 4 fs on unrepartitioned hydrogens is the unstable
            # combination, and `md-gen` refuses it. See docs/examples/hmr-4fs.yaml.
            "hydrogen_mass_amu": None,
        },
    }
    # Only the block that applies is written. The file describes ONE system; carrying both an
    # explicit water box and a GB model would leave a reader -- and the resolver -- to guess which
    # one the run uses, which is exactly the ambiguity `resolve_sys_config` exists to avoid.
    document.pop("implicit_solvent" if not implicit else "solvent")
    if implicit:
        document["forcefield"]["water"] = None
        document["constraints"]["rigid_water"] = False
    return document


def md_defaults(*, methods=("cMD", "REST2"), solvent: str = DEFAULT_SOLVENT) -> dict[str, Any]:
    """Simulation protocol settings.

    Under implicit solvent there is no box, so there is no barostat and no pressure: the production
    ensembles become NVT and `pressure_bar` is written as null with a note. Saying "NPT" for a
    non-periodic system would be a false record of the ensemble that ran.
    """
    methods = [canonical_method(m) for m in methods]
    implicit = is_implicit(solvent)
    ensemble = "NVT" if implicit else "NPT"

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "methods": methods,
        "common": {
            "temperature_kelvin": 300,
            "pressure_bar": None if implicit else 1.0,
            # `MonteCarloBarostat` volume-move attempt interval, in STEPS -- OpenMM's own default,
            # 0.05 ps at 2 fs and 0.10 ps at the optional 4 fs. null under implicit solvent, where
            # there is no box and no barostat exists in the System at all.
            "barostat_frequency_steps": (None if implicit
                                         else DEFAULT_BAROSTAT_FREQUENCY_STEPS),
            "timestep_fs": 2.0,
            # OpenMM's LangevinMiddleIntegrator collision rate, in ps^-1. 1.0 ps^-1 is a nominal
            # 1 ps damping time: weak enough not to dominate the dynamics, strong enough to
            # thermostat. It still affects real-time dynamical and transport observables -- see
            # docs/md-defaults-scientific-rationale.md.
            "friction_per_ps": 1.0,
            "random_seed": None,
        },
        "minimization": {
            "max_iterations": 1000,
            "restraint_k_kcal_mol_a2": 1.0,
        },
        # One directory per stage is generated from this block, in this order. Every duration is
        # the length of that stage; a null one means the stage does not apply to this solvent.
        "equilibration": {
            "nvt_restrained_duration_ps": 10.0,
            "npt_restrained_duration_ps": None if implicit else 10.0,
            "npt_free_duration_ps": None if implicit else 10.0,
            "nvt_free_duration_ps": 10.0 if implicit else None,
            "restraint_k_kcal_mol_a2": 1.0,
        },
    }
    if implicit:
        document["common"]["pressure_note"] = (
            "implicit solvent has no periodic box, so pressure is not applicable and no barostat "
            "is added")
        document["equilibration"]["npt_note"] = (
            "not applicable without a box: the free stage is NVT")

    if "cMD" in methods:
        document["cMD"] = {
            "ensemble": ensemble,
            # A single walker on ONE fixed rung of the REST2 ladder. tau = 0 is ordinary cMD: the
            # unscaled, physical Hamiltonian, and the System is left byte-identical. tau > 0 keeps
            # the same thermostat temperature and scales only the solute Hamiltonian, exactly as
            # the matching REST2 rung does -- it is NOT high-temperature MD, and beta is unchanged.
            # tau is the SOURCE parameter; s = (1 - tau)^2 is a derived diagnostic.
            "tau": 0.0,
            # Only meaningful at tau > 0, and matched to the REST2 default so a fixed-tau walker
            # and the ladder rung at the same tau are the same Hamiltonian.
            "omega_exclusion": True,
            "duration_ns": 5,
            "checkpoint_interval_ps": 100,
            "whole_system_interval_ps": 100,
            "solute_interval_ps": 10,
        }
    if "AIS" in methods:
        document["AIS"] = ais_defaults()
    if "REST2" in methods:
        document["REST2"] = {
            "ensemble": ensemble,
            # tau is the SOURCE parameter. The scaling is derived: solute-solute (1-tau)^2,
            # solute-environment (1-tau). See openmm/tau.py.
            "tau_min": 0.0,
            "tau_max": 0.5,
            "number_of_replicas": 6,
            "tau_interpolation": "linear",
            # Time BETWEEN consecutive exchange rounds. Total production is
            # duration_per_segment_ps * number_of_exchanges, so this default is 10 ns per replica.
            # Per-tau equilibration, run once per replica by `REST2/equilibrate.py` before any
            # exchange. It is NOT production and is not counted in the totals below.
            "equilibration_duration_ps": 1000.0,
            "duration_per_segment_ps": 10,
            "number_of_exchanges": 1000,
            "enhanced_region": "solute",
            "omega_exclusion": True,
            "checkpoint_interval_ps": 100,
            "whole_system_interval_ps": 100,
            "solute_interval_ps": 10,
        }
    return document


def default_document(name: str) -> dict[str, Any]:
    """`show-default <name>` -- the same definitions `sys-config` writes."""
    key = str(name).lower()
    if key == "sys":
        return sys_defaults()
    if key == "cmd":
        return {k: v for k, v in md_defaults(methods=["cMD"]).items() if k != "REST2"}
    if key == "rest2":
        return {k: v for k, v in md_defaults(methods=["REST2"]).items() if k != "cMD"}
    if key == "ais":
        return {k: v for k, v in md_defaults(methods=["AIS"]).items()
                if k not in ("cMD", "REST2")}
    if key == "all":
        return {"sys": sys_defaults(), "md": md_defaults(methods=METHODS)}
    raise ValueError(f"unknown default {name!r}; expected sys, cMD, REST2, AIS or all")
