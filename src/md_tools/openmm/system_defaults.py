"""The system-construction defaults `md-openmm build-top` resolves against, defined once.

`build/top.py` and the shipped `configs/sys/build-top.config` both read these, so a supported
force-field pairing or a default box padding is declared in exactly one place. The reason this
repository grew drift guards at all is that the same fact used to live in five.

Values are plain Python data that serialise to the YAML a user edits. Nothing here validates --
see `system_config.py` -- and nothing here is versioned per-profile. A user who wants different
settings edits the YAML.

**No MD workflow settings live here.** They used to: `md_defaults()` and `ais_defaults()` described
the retired `methods:` model in `duration_ns` and `switching_duration_ps`, sitting beside the system
defaults so that two different models both looked authoritative. Protocol, stage lengths -- always
integer step counts -- and reporting intervals have one authority, `md_tools.build.md`.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1
ENGINE = "openmm"
ENGINE_VERSION = "8.6.0"

SOLVENTS = ("TIP3P", "OPC", "GBn2")

#: What `--solvent` selects when nothing is asked for. TIP3P is the method-development default:
#: see `docs/scientific-defaults.md`.
DEFAULT_SOLVENT = "TIP3P"




def canonical_solvent(name: str) -> str:
    for canonical in SOLVENTS:
        if str(name).lower() == canonical.lower():
            return canonical
    raise ValueError(f"unknown solvent {name!r}; expected one of {', '.join(SOLVENTS)}")


def is_implicit(solvent: str) -> bool:
    return canonical_solvent(solvent) == "GBn2"



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
#: published combination-level benchmark. See `docs/scientific-defaults.md`.
EXPLICIT_COMBINATIONS = {
    "TIP3P": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"},
    "OPC": {"protein": "amber19-all.xml", "water": "amber19/opc.xml"},
}

#: Substrings that identify which supported family a hand-written resource name belongs to.
#:
#: the build configuration writes the qualified resource, but the file is editable YAML and a user may write
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
#: The implicit route's counterpart of `PROTEIN_FORCEFIELDS`: the label a user writes, mapped to
#: the tleap resource. Stated as a mapping rather than as a single default so that what an
#: implicit build uses is what the configuration asked for. It used to be the default
#: unconditionally, which meant a request for a force field with no GB parameterisation was not
#: refused -- it was quietly built as this one.
IMPLICIT_PROTEIN_FORCEFIELDS = {"ff14SB": IMPLICIT_PROTEIN_FORCEFIELD}
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




#: The MD-data repository, named once. Its exact commit is NEVER filled in here: a commit this
#: package could guess is not a pin, and MD-data's contract exists to prevent exactly that.
MD_TOOLS_REPOSITORY = "https://github.com/csy0000/MD-tools"


def dataset_defaults() -> dict[str, Any]:
    """The MD-data dataset identity, as editable YAML with every unguessable field left null.

    MD-tools owns the dataset contract now: the v2 model and its generated schema live in
    `md_tools.data_contract`, and `docs/data-contract.md` states where the FAIR boundary falls.
    (MD-data owned v1, at `docs/contracts/dataset-v1.md`; nothing here validates against it.) This
    block is the smallest input `build-top` needs to WRITE a manifest that the validator accepts. It
    lives in `sys.config.yaml` rather than in both files because a dataset has one identity, and
    `build-md` reads it back from `common/resolved_sys.config.yaml`.

    Every `null` is a required value that this package must not invent:

    * a person is a scientific identity, not the account the job ran under;
    * a repository without its exact 40-hex commit records where to look but not what ran, which
      is the failure MD-data's contract exists to prevent -- so a version or an installed
      fingerprint is useful generation provenance and does NOT satisfy `commit`;
    * a dataset ID that this package derived from a path stops being stable the moment the path
      changes.

    Leave `enabled: false` for an unregistered local `inputs/ + MD/` generation. Such a tree is
    NOT MD-data compliant and is labelled that way in its own provenance.

    Derived rather than asked for, because they are facts about this generation rather than
    choices: `schema_version`, `created_at`, `status`, `path` (from where the dataset actually is
    under `MD_DATA`), the component list, and `templates.version`.
    """
    return {
        # false: write a plain inputs/ + MD/ tree, unregistered and not contract-managed.
        # true: write a contract-managed dataset, and refuse to generate until the fields below
        # are filled in.
        "enabled": False,
        "dataset_id": None,
        "namespace": None,
        "dataset_name": None,
        # `project` or `baseline`. The `baseline` namespace is reserved for `role: baseline`.
        "role": None,
        # Prose. What was actually simulated.
        "system": None,
        "created_by": {
            "person_id": None,
            "name": None,
            "affiliation": None,
            # Optional in the contract, and optional here.
            "orcid": None,
        },
        # The project repository this dataset was produced for, pinned.
        "origin": {
            "repository": None,
            "commit": None,
        },
        # This repository. The commit is required and is never filled in automatically: an
        # installed wheel has no checkout to read one from, and inventing one would defeat the pin.
        "templates": {
            "repository": MD_TOOLS_REPOSITORY,
            "commit": None,
        },
        "derived_from": [],
        "notes": None,
    }


def sys_defaults(*, peptide: bool = True, solvent: str = DEFAULT_SOLVENT,
                 kind: str | None = None) -> dict[str, Any]:
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
            # The authoritative classification. `peptide` below is derived from it and kept only
            # so readers written before `kind` existed still find what they expect.
            "kind": str(kind) if kind else ("peptide" if peptide else "ligand"),
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
            # copy distance. `build-top` records all four quantities; raise this to 2.0 for an
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
        # The MD-data dataset identity. Disabled by default: a plain inputs/ + MD/ tree needs no
        # manifest, and a manifest cannot be written from values this package would have to guess.
        "dataset": dataset_defaults(),
        "constraints": {
            "type": "HBonds",
            "rigid_water": True,
            # null: hydrogens keep the masses the force field gave them, and the timestep stays at
            # 2 fs. This is the baseline, not a placeholder. Set to 3.024 for HMR, and raise
            # timestep_fs to 4.0 with it -- 4 fs on unrepartitioned hydrogens is the unstable
            # combination, and build-md refuses it. See the HMR section of
            # configs/sys/build-top.config.
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




