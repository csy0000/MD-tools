"""Molecular preparation, ending before any dynamics.

This is the implementation behind ``MD_system_gen.py``. It reuses the package's existing
construction path (``equilibration.build_simbox``, which despite its module name performs no
dynamics) and stops there, then writes the portable system bundle that ``MD_input_gen.py`` consumes.

What "stops there" means precisely: the bundle contains an **initial** State -- positions and
periodic box vectors as built, with no velocities, because nothing has been integrated. A bundle
from this module has never seen a minimiser.

The bundle is deliberately more than ``System.xml``. A serialized OpenMM System carries parameters
but no atom or residue names, so it is not a topology and cannot be read back into something a human
or a downstream tool can index. The bundle therefore always carries a topology-bearing structure
alongside it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["prepare_system", "SYSTEM_MANIFEST_SCHEMA_VERSION"]

#: Bumped when the manifest's shape changes. `MD_input_gen.py` validates it on read rather than
#: assuming, so an old bundle fails loudly instead of being silently misread.
SYSTEM_MANIFEST_SCHEMA_VERSION = 1

#: Files every system bundle must contain. The list is checked after writing, so a bundle that is
#: missing a piece is caught here rather than by a run three stages later.
REQUIRED_BUNDLE_FILES = (
    "system.xml",
    "topology.pdb",
    "topology.cif",
    "initial_state.xml",
    "forcefield.json",
    "system_manifest.json",
    "system.yaml",
    "checksums.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Atomic write: temp file in the destination filesystem, fsync, replace."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _runtime_cfg_from_system_config(config: dict, input_path: Path, input_format: str,
                                    system_type: str) -> tuple[dict, Optional[str], Optional[Path]]:
    """Project system_config.json onto the package's runtime configuration tree.

    system_config.json owns chemistry and system-building choices ONLY. Anything protocol-shaped
    (minimisation, equilibration, production, reporting) is rejected here rather than silently
    ignored, because a user who wrote it there believes it took effect.
    """
    from .config import DEFAULTS
    from .solvation_mode import resolve_solvation
    import copy

    protocol_keys = {"minimization", "equilibration", "production", "protocol", "reporting",
                     "integrator", "execution"}
    intruders = sorted(protocol_keys & set(config))
    if intruders:
        raise ValueError(
            f"system_config.json contains protocol settings: {', '.join(intruders)}. "
            "Chemistry and system building live here; minimisation, equilibration, production, "
            "reporting and execution belong in md_config.json and are consumed by "
            "MD_input_gen.py. Leaving them here would mean they are never applied."
        )

    cfg = copy.deepcopy(DEFAULTS)

    # Conformer generation needs an explicit seed. The package DEFAULTS leave it None because the
    # canonical pipeline fills it from the randomness block; this front end has no such block, so it
    # derives one. Without a seed the SMILES route dies on int(None) inside ETKDG -- a failure the
    # dry-run path cannot reach, since it never builds anything.
    #
    # Derived rather than hard-coded: the conformer seed decides which starting structure the whole
    # bundle is built around, so it belongs to the run's seed map like every other stream.
    from .seeds import DEFAULT_MASTER_SEED, derive_seed

    randomness = config.get("randomness") or {}
    master = int(randomness.get("master_seed", DEFAULT_MASTER_SEED))
    explicit = randomness.get("structure_seed")
    seed = int(explicit) if explicit is not None else derive_seed(master, "structure/conformer")
    cfg["structure"]["etkdg"]["seed"] = seed
    cfg["run"]["seed"] = master
    for stage in ("equilibration",):
        if stage in cfg and isinstance(cfg[stage], dict):
            cfg[stage].setdefault("seed", seed)

    system_block = config.get("system") or {}
    cfg["system"]["slug"] = system_block.get("id") or input_path.stem.lower().replace("-", "_")
    cfg["system"]["solute_kind"] = "ligand" if system_type == "ligand" else "peptide"

    #: Where every build-defining value came from. A manifest that records values without their
    #: origin cannot answer the only question that matters when two bundles differ: which of these
    #: did I choose, and which did the package choose for me?
    sources: dict = {}
    for section in ("forcefield", "system_build"):
        for key in cfg.get(section, {}):
            sources[f"{section}.{key}"] = "package default"
        if section in config:
            for key, value in config[section].items():
                sources[f"{section}.{key}"] = "user input"
            cfg[section].update(config[section])
    # The default water is derived from the force fields that were ACTUALLY RESOLVED just above,
    # not from the input label. `peptide` and `ligand` say which reader parsed the input; the
    # force-field fields say which parameters will be assigned, and it is the parameters that were
    # fitted against a particular water model. Deriving from the label would also give the wrong
    # answer for a complex, where both are present.
    from .water_policy import AmbiguousWaterPolicy, resolve_default_water

    user_forcefield = config.get("forcefield") or {}
    user_solvation = config.get("solvation") or {}
    mode = (user_solvation.get("mode") or "explicit")
    water_policy = None
    if mode != "implicit":
        try:
            water_policy = resolve_default_water(cfg["forcefield"],
                                                 solute_kind=cfg["system"]["solute_kind"])
        except AmbiguousWaterPolicy:
            # Only fatal if the package would have to choose. A configuration that states BOTH
            # water fields has already answered the question, and refusing it would reject the very
            # escape hatch the error message recommends.
            if "water" not in user_forcefield or "water_model" not in user_solvation:
                raise

    solvation_defaults = copy.deepcopy(DEFAULTS["solvation"])
    if "water" in user_forcefield:
        sources["forcefield.water"] = "user input"
    elif water_policy is not None:
        cfg["forcefield"]["water"] = water_policy["water"]
        sources["forcefield.water"] = f"package default: {water_policy['basis']}"
    if water_policy is not None:
        solvation_defaults["water_model"] = water_policy["water_model"]
        # Ion parameters are water-model specific and OpenMM ships them INSIDE the water
        # force-field file, so they cannot drift from the water they were fitted with. Recorded
        # because "which ion parameters did this run use?" otherwise has no answer in the bundle.
        cfg["_water_policy"] = {
            **water_policy,
            "ion_parameters": {
                "source": cfg["forcefield"]["water"],
                "note": ("Joung-Cheatham ion parameters are fitted per water model and ship inside "
                         "the water force-field file, so they follow the water model above."),
            },
        }

    # One discriminated solvation contract, resolved once. Explicit mode keeps the existing
    # defaults; implicit mode rejects every field that describes water it does not have.
    solvation = resolve_solvation(config.get("solvation"), defaults=solvation_defaults)
    cfg["solvation"] = {k: v for k, v in solvation.items() if k != "sources"}
    for key, origin in solvation["sources"].items():
        sources[f"solvation.{key}"] = origin
    sources["solvation.mode"] = ("user input" if (config.get("solvation") or {}).get("mode")
                                 else "package default: explicit")

    sources["system.slug"] = ("user input" if (config.get("system") or {}).get("id")
                              else "route-derived: input filename")
    sources["system.solute_kind"] = "route-derived: system.type"
    sources["structure.etkdg.seed"] = ("user input" if explicit is not None
                                       else f"derived from master seed {master}")
    cfg["_value_sources"] = sources

    if input_format == "smi":
        check_ligand_build_matches_the_route(config, cfg)

    smiles = None
    pdb: Optional[Path] = None
    if input_format == "smi":
        text = input_path.read_text().strip().splitlines()
        if not text:
            raise ValueError(f"{input_path} is empty")
        smiles = text[0].split()[0]
    else:
        pdb = input_path

    return cfg, smiles, pdb


#: The sections of the resolved runtime configuration that DEFINE the built System. Changing any of
#: them requires a new bundle, not a new protocol, which is exactly why they are the ones persisted.
_BUILD_DEFINING_SECTIONS = ("forcefield", "solvation", "system_build", "structure", "system")


#: OpenMM serialises a nonbonded method as an enum integer, so `forcefield.json` records `4` where
#: everything else records `"PME"`. Comparing those directly reports a disagreement that is only a
#: difference of representation. The mapping is written out rather than guessed.
_NONBONDED_METHOD_NAMES = {0: "NoCutoff", 1: "CutoffNonPeriodic", 2: "CutoffPeriodic",
                           3: "Ewald", 4: "PME", 5: "LJPME"}


def _nonbonded_method_name(value):
    """A nonbonded method as a name, whichever representation it arrives in."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return _NONBONDED_METHOD_NAMES.get(value)
    return str(value)


def _named_nonbonded(nonbonded):
    """The nonbonded record with the method's NAME added beside its enum.

    Added rather than replaced: existing bundles record the integer, and a reader of either
    generation should find what they expect. `4` alone is not a description of a Hamiltonian.
    """
    if not isinstance(nonbonded, dict):
        return nonbonded
    out = dict(nonbonded)
    name = _nonbonded_method_name(out.get("method"))
    if name is not None:
        out["method_name"] = name
    return out


def _check_the_three_files_describe_one_hamiltonian(bundle: Path) -> None:
    """`forcefield.json`, `system.yaml` and `system_manifest.json` must agree.

    Three files describe the same System for three different readers. Nothing forces them to agree,
    and a disagreement is invisible: each file is internally plausible, so a bundle can claim ff19SB
    in one place and something else in another, and only a run would reveal it -- if anyone looked.
    Checked at build time rather than only in a test, because the bundle is the thing that travels.
    """
    import json as _json

    import yaml as _yaml

    manifest = _json.loads((bundle / "system_manifest.json").read_text())
    forcefield = _json.loads((bundle / "forcefield.json").read_text())
    system_yaml = _yaml.safe_load((bundle / "system.yaml").read_text())

    resolved = manifest["resolved_system_config"]["forcefield"]
    parameterization = system_yaml.get("parameterization") or {}
    # `forcefield.json` names the protein field `protein_forcefield`; the resolved configuration and
    # `system.yaml` use their own spellings. The mapping is written out rather than assumed, because
    # comparing a key that does not exist compares None to None and can never fail.
    # On the ligand route `forcefield.json` records the small molecule as a BLOCK -- the force
    # field name plus the charge method, net and formal charge and atom count -- while the manifest
    # and `system.yaml` record the force field as a bare name. Comparing the block against the name
    # put a dict into a set and raised `TypeError: unhashable type: 'dict'`, so this check crashed
    # on every explicit ligand bundle instead of running. Take the name out of the block first.
    ff_ligand = forcefield.get("ligand")
    ff_charge_method = forcefield.get("ligand_charge_method")
    if isinstance(ff_ligand, dict):
        ff_charge_method = ff_charge_method or ff_ligand.get("charge_method")
        ff_ligand = ff_ligand.get("forcefield")

    checks = (
        ("protein force field", resolved.get("protein"),
         parameterization.get("protein_forcefield"), forcefield.get("protein_forcefield")),
        ("water force field", resolved.get("water"),
         parameterization.get("water_forcefield"), forcefield.get("water")),
        ("small-molecule force field", resolved.get("ligand"),
         parameterization.get("small_molecule_forcefield"), ff_ligand),
        ("charge method", resolved.get("ligand_charge_method"),
         parameterization.get("charge_method"), ff_charge_method),
    )
    problems = []

    # Build settings, not just force fields. The first version of this check compared only the
    # latter and so did not notice a manifest claiming PME and HMR for a System that had neither.
    build = manifest["resolved_system_config"].get("system_build") or {}
    ff_nonbonded = (forcefield.get("nonbonded") or {})
    ff_method = _nonbonded_method_name(ff_nonbonded.get("method"))
    if ff_method and build.get("nonbonded_method") and ff_method != build["nonbonded_method"]:
        problems.append(
            f"    nonbonded method: system_manifest.json={build['nonbonded_method']!r} "
            f"forcefield.json={ff_method!r}")
    ff_hmr = (forcefield.get("hmr") or {})
    if ff_hmr.get("scope") and build.get("hmr_scope") and ff_hmr["scope"] != build["hmr_scope"]:
        problems.append(
            f"    hydrogen mass repartitioning: system_manifest.json={build['hmr_scope']!r} "
            f"forcefield.json={ff_hmr['scope']!r}")

    for label, in_manifest, in_yaml, in_ff in checks:
        # compared by their canonical text, so a value this check did not anticipate is REPORTED as
        # a disagreement rather than crashing the build that was about to be written
        stated = {json.dumps(v, sort_keys=True, default=str)
                  for v in (in_manifest, in_yaml, in_ff) if v is not None}
        if len(stated) > 1:
            problems.append(
                f"    {label}: system_manifest.json={in_manifest!r} system.yaml={in_yaml!r} "
                f"forcefield.json={in_ff!r}")
    if problems:
        raise RuntimeError(
            "the bundle's three descriptions of the Hamiltonian disagree, refusing to publish it:\n"
            + "\n".join(problems))


def _implicit_forcefield_record(cfg: dict, built: dict) -> dict:
    """The force-field block for an implicit bundle, in the same shape the explicit path uses."""
    ff = cfg.get("forcefield") or {}
    build = built["build"]
    return {
        "protein_forcefield": (build.get("protein_forcefield") if built["route"] == "peptide"
                               else None),
        "water": None,
        "ligand": build.get("small_molecule_forcefield") if built["route"] == "ligand" else None,
        "ligand_charge_method": (build.get("charge_method") if built["route"] == "ligand"
                                 else ff.get("ligand_charge_method")),
        "xml": [],
        "implicit_model": build["implicit_model"],
        "radii": build["radii"],
    }


def _write_initial_state_from_amber(system_xml: Path, rst7: Path, out: Path) -> None:
    """An initial State from full-precision Amber coordinates. No velocities, nothing integrated."""
    import openmm
    from openmm import XmlSerializer, app, unit

    system = XmlSerializer.deserialize(Path(system_xml).read_text(encoding="utf-8"))
    positions = app.AmberInpcrdFile(str(rst7)).positions
    context = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    state = context.getState(getPositions=True)
    Path(out).write_text(XmlSerializer.serialize(state), encoding="utf-8")


def _build_defining(cfg: dict) -> dict:
    """The resolved, build-defining configuration, with nothing execution-only in it.

    Platform, device and precision are deliberately excluded: they are machine choices recorded in
    provenance elsewhere, and including them here would make an identical System built on a
    different machine look like a different System.
    """
    import copy

    out = {}
    for section in _BUILD_DEFINING_SECTIONS:
        value = cfg.get(section)
        if isinstance(value, dict):
            out[section] = copy.deepcopy(
                {k: v for k, v in value.items() if not k.startswith("_")})
    return out


def check_ligand_build_matches_the_route(config: dict, cfg: dict) -> None:
    """The declared chemistry must be the chemistry that will actually be used.

    `ligand_build` is copied into the bundle manifest as a record of how the molecule was built. If
    it names a charge model or parameterisation route that the resolved force field does not use,
    that record is false -- and it is exactly the kind of falsehood nobody notices, because both
    values look plausible in isolation. Checked here rather than in the entry point so that callers
    which never touch the command line are covered too.
    """
    declared = config.get("ligand_build") or {}
    ff = cfg.get("forcefield") or {}
    pairs = (
        ("parameterization_route", ff.get("ligand"), "forcefield.ligand"),
        ("charge_model", ff.get("ligand_charge_method"), "forcefield.ligand_charge_method"),
    )
    problems = [
        f"    ligand_build.{field}={declared[field]!r} but the resolved {source}={resolved!r}"
        for field, resolved, source in pairs
        if field in declared and resolved is not None and declared[field] != resolved
    ]
    if problems:
        raise ValueError(
            "the declared ligand_build does not describe the parameterisation that would run:\n"
            + "\n".join(problems)
            + "\n  Change the declaration, or state the force field you meant under `forcefield`. "
              "This block is\n  persisted as provenance, so a mismatch would record chemistry that "
              "never happened."
        )


def prepare_system(*, input_path: Path, input_format: str, system_type: str, config: dict,
                   outdir: Path, overwrite: bool = False) -> dict:
    """Build a portable system bundle. Runs no dynamics.

    Transactional: everything is written into a temporary directory in the same filesystem and
    moved into place only once the required-file check passes, so an interrupted run leaves either
    the previous bundle or nothing -- never a half-written one whose checksums describe a mixture.
    """
    from .destination import SYSTEM_BUNDLE_TARGETS, check_destination, publish
    from .solvation_mode import IMPLICIT
    from .equilibration import build_simbox

    outdir = Path(outdir)
    # Checked BEFORE any work: parameterising a ligand can cost half an hour, and discovering the
    # destination was occupied only at the publish step would throw all of it away.
    check_destination(outdir, SYSTEM_BUNDLE_TARGETS, overwrite=overwrite, what="system bundle")
    staging = outdir.parent / f".{outdir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        cfg, smiles, pdb = _runtime_cfg_from_system_config(
            config, input_path, input_format, system_type)

        solvation = cfg["solvation"]
        if solvation["mode"] == IMPLICIT:
            # A different construction entirely: no water, no box, no ions. The Amber files it
            # writes are how the GBn2 System is built and are kept as provenance, not because
            # anything here runs Amber.
            from .implicit import build_implicit_bundle_inputs, implicit_provenance

            built = build_implicit_bundle_inputs(
                route=("ligand" if system_type == "ligand" else "peptide"),
                cfg=cfg, staging=staging, pdb=pdb, smiles=smiles,
                implicit_model=solvation["implicit_model"], radii=solvation["radii"])
            # The implicit peptide route is parameterised by tleap, so the force field it used is
            # the leaprc -- not the OpenMM XML the explicit defaults name. Recording both names for
            # one force field is exactly what the agreement check exists to catch.
            # The package defaults name a force field for EVERY component -- protein, water and
            # small molecule -- because an explicit-water build may need all three. A given route
            # parameterises one of them, and carrying the others records chemistry that never ran.
            # This has now been the shape of three separate defects (water on the ligand route, a
            # protein force field on the ligand route, small-molecule parameters on the peptide
            # route), so the rule is written once: keep what this route used, null the rest.
            used_by_route = {
                "peptide": {"protein": built["build"].get("protein_forcefield",
                                                          "leaprc.protein.ff19SB")},
                "ligand": {"ligand": built["build"].get("small_molecule_forcefield"),
                           "ligand_charge_method": built["build"].get("charge_method")},
            }[built["route"]]
            for field in ("protein", "water", "ligand", "ligand_charge_method"):
                cfg["forcefield"][field] = used_by_route.get(field)
                cfg.setdefault("_value_sources", {})[f"forcefield.{field}"] = (
                    f"route-derived: {built['route']} route, implicit solvent")
            # The build settings must describe the System that was built. The package defaults are
            # explicit-water ones -- PME, a 1.0 nm cutoff, HMR to 3.024 amu, rigid water -- and
            # leaving them here made the manifest claim a Hamiltonian this bundle does not have.
            cfg["system_build"].update({
                "nonbonded_method": "NoCutoff",
                "nonbonded_cutoff_nm": None,
                "switch_distance_nm": None,
                "use_dispersion_correction": False,
                "ewald_error_tolerance": None,
                "minimum_image_margin_nm": None,
                "constraints": "HBonds",
                "rigid_water": False,
                "hydrogen_mass_amu": None,
                "hmr_scope": "none",
                "remove_cm_motion": True,
            })
            for key in ("nonbonded_method", "nonbonded_cutoff_nm", "hydrogen_mass_amu",
                        "hmr_scope", "rigid_water"):
                cfg.setdefault("_value_sources", {})[f"system_build.{key}"] = (
                    "route-derived: implicit solvent")
                cfg.setdefault("_value_sources", {})["forcefield.protein"] = (
                    "route-derived: tleap leaprc for the implicit route")
            info = {
                "route": built["route"],
                "input_route": input_format,
                "n_solute_atoms": built["n_solute_atoms"],
                "n_particles": built["n_particles"],
                "forcefield": _implicit_forcefield_record(cfg, built),
                "nonbonded": {"method": "NoCutoff", "cutoff_nm": None,
                              "note": "implicit solvent has no periodic box and no cutoff"},
                "hmr": {"scope": "none", "target_hydrogen_mass_amu": None,
                        "note": ("the pinned reference does not repartition hydrogen mass; the "
                                 "implicit profile does not either, so the GBn2 energy comparison "
                                 "is against an unrepartitioned System")},
                "implicit": implicit_provenance(built["build"]),
                "box": None,
                "salt": None,
            }
            system_xml = staging / "system.xml"
            topology_pdb = staging / "topology.pdb"

            # Positions come from the rst7, not the PDB: PDB coordinates are rounded to three
            # decimals in angstrom, which shifts every force by ~0.02 kJ/mol against the
            # reference construction. Measured; small, and pointless to accept.
            initial_state = staging / "initial_state.xml"
            _write_initial_state_from_amber(
                system_xml, staging / "system.rst7", initial_state)
        else:
            # The construction path, reused unchanged. It solvates, ionises and parameterises; it
            # integrates nothing.
            info = build_simbox(cfg, staging, "system", smiles=smiles, pdb=pdb)

            system_xml = staging / "system.xml"
            topology_pdb = staging / "topology.pdb"
            Path(info["system_xml"]).replace(system_xml)
            Path(info["topology_pdb"]).replace(topology_pdb)

            # An initial State: positions and box vectors as built, no velocities, nothing
            # integrated.
            initial_state = staging / "initial_state.xml"
            _write_initial_state(system_xml, topology_pdb, initial_state)
        _write_topology_cif(topology_pdb, staging / "topology.cif")

        forcefield = {
            "route": info.get("route"),
            "input_route": info.get("input_route"),
            "system_type": system_type,
            **(info.get("forcefield") or {}),
            "nonbonded": _named_nonbonded(info.get("nonbonded")),
            "hmr": info.get("hmr"),
            "implicit": info.get("implicit"),
            "constraints_note": (
                "constraints and hydrogen mass are properties of the built System and are recorded "
                "here; changing either requires a new system bundle, not a new protocol."
            ),
        }
        _write_json(staging / "forcefield.json", forcefield)

        # copy the exact molecular input so the bundle is self-contained and relocatable
        original = staging / "original_inputs"
        original.mkdir(exist_ok=True)
        shutil.copy2(input_path, original / input_path.name)

        manifest = {
            "schema_version": SYSTEM_MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "MD_system_gen.py",
            "prepared_through": "minimisation NOT run; this bundle has never been integrated",
            "system": {
                "id": cfg["system"]["slug"],
                "type": system_type,
                "input_file": input_path.name,
                "input_format": input_format,
                "input_sha256": _sha256(input_path),
                "smiles": smiles,
            },
            "composition": {
                "n_solute_atoms": info["n_solute_atoms"],
                "n_particles": info["n_particles"],
                "n_waters": info.get("n_waters"),
                "ions": info.get("ions"),
                "degrees_of_freedom": info.get("degrees_of_freedom"),
                "n_constraints": info.get("n_constraints"),
            },
            # Neutralising counterions and added salt pairs are DIFFERENT quantities and are
            # reported separately; collapsing them misstates the ionic strength.
            "salt": info.get("salt"),
            "geometry": info.get("geometry"),
            "water": info.get("water"),
            # WHY this water model, not merely which one. For a mixed-force-field complex the
            # rationale states that the choice is a compatibility decision rather than a validated
            # pairing, so a report built on the bundle can say so instead of implying otherwise.
            "water_policy": cfg.get("_water_policy"),
            # Present only for implicit bundles. Everything needed to rebuild the exact GBn2
            # System: the construction branch, both library versions, the tleap commands where
            # they were used, and what the radius change actually did.
            "implicit": info.get("implicit"),
            "amber_files": ({"topology": "system.prmtop", "coordinates": "system.rst7",
                             "role": ("construction intermediates and provenance for the OpenMM "
                                      "System; this repository has no Amber execution engine")}
                            if info.get("implicit") else None),
            "omega": {
                "central_bonds": info.get("omega_central_bonds"),
                "detection_method": info.get("omega_detection_method"),
            },
            "files": {name: name for name in REQUIRED_BUNDLE_FILES},
            # The configuration that was actually USED, after defaults and route decisions -- not
            # the document the user wrote. Recording the input as if it were the resolution is how a
            # manifest comes to omit every value the package chose, which is most of them.
            "stated_system_config": config,
            "resolved_system_config": _build_defining(cfg),
            "value_sources": cfg.get("_value_sources", {}),
            "outputs_are_amber_or_gromacs": False,
            "adapter_status": {
                "openmm": "implemented",
                "amber": "not implemented -- would emit prmtop/rst7",
                "gromacs": "not implemented -- would emit top/gro; a tpr is stage-specific and "
                           "belongs to MD input generation, not molecular preparation",
            },
        }
        _write_json(staging / "system_manifest.json", manifest)

        # The package's legacy system manifest, written from the SAME chemistry. The runner reads
        # it, so emitting it here is what lets a generated project be handed to the existing REST2
        # runner instead of teaching that runner a second bundle format. It carries chemistry only
        # -- no protocol -- so it stays inside this generator's remit.
        _write_system_yaml(staging / "system.yaml", config, cfg, manifest, info)

        checksums = {p.name: _sha256(p) for p in sorted(staging.iterdir())
                     if p.is_file() and p.name != "checksums.json"}
        _write_json(staging / "checksums.json", {
            "algorithm": "sha256",
            "note": "covers every bundle file except this one, which cannot contain its own hash",
            "files": checksums,
        })

        _check_the_three_files_describe_one_hamiltonian(staging)

        required = list(REQUIRED_BUNDLE_FILES)
        solvation = cfg["solvation"]
        if solvation["mode"] == IMPLICIT:
            # The Amber files ARE the construction path for GBn2, so a bundle without them cannot
            # be rebuilt or checked. Required members, not incidental leftovers.
            required += ["system.prmtop", "system.rst7"]
        missing = [f for f in required if not (staging / f).is_file()]
        if missing:
            raise RuntimeError(
                f"system bundle is incomplete, refusing to publish it: missing {missing}"
            )

        publish(staging, outdir, overwrite=overwrite, what="system bundle")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "bundle_dir": str(outdir),
        "system_manifest": str(outdir / "system_manifest.json"),
        "n_solute_atoms": info["n_solute_atoms"],
        "n_particles": info["n_particles"],
    }


def _write_initial_state(system_xml: Path, topology_pdb: Path, out_state: Path) -> None:
    """Serialize positions and box vectors as an OpenMM State. No velocities: nothing has moved."""
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile
    import openmm

    system = XmlSerializer.deserialize(system_xml.read_text())
    pdb = PDBFile(str(topology_pdb))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)   # required to make a Context
    platform = openmm.Platform.getPlatformByName("Reference")      # never touches a GPU
    context = openmm.Context(system, integrator, platform)
    context.setPositions(pdb.positions)
    if pdb.topology.getPeriodicBoxVectors() is not None:
        context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    state = context.getState(getPositions=True)
    out_state.write_text(XmlSerializer.serialize(state))
    del context, integrator


def _write_topology_cif(topology_pdb: Path, out_cif: Path) -> None:
    from openmm.app import PDBFile, PDBxFile

    pdb = PDBFile(str(topology_pdb))
    with out_cif.open("w") as handle:
        PDBxFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)


def _write_system_yaml(path: Path, config: dict, cfg: dict, manifest: dict, info: dict) -> None:
    """Emit the package's legacy system manifest from the prepared chemistry.

    Chemistry only. Protocol belongs to md_config.json and is never written here.

    Reads the RESOLVED configuration, not the user's document: this file used to describe whatever
    the user happened to write, so a bundle whose force field came from a default recorded `null`.
    """
    import yaml

    system = manifest["system"]
    ff = cfg.get("forcefield") or {}
    solv = cfg.get("solvation") or {}
    doc = {
        "schema_version": 1,
        "system_id": system["id"],
        "display_name": (config.get("system") or {}).get("display_name", system["id"]),
        "input": {
            "route": "smiles" if system["input_format"] == "smi" else "pdb",
            "expected_formal_charge": (config.get("ligand_build") or {}).get("formal_charge", 0),
        },
        "parameterization": {
            "small_molecule_forcefield": ff.get("ligand"),
            "charge_method": ff.get("ligand_charge_method"),
            "protein_forcefield": ff.get("protein"),
            "water_forcefield": ff.get("water"),
        },
        # The solvation block is written per MODE. An implicit system emitting water-shaped keys
        # full of nulls reads as a system whose water settings were forgotten, rather than one that
        # has no water.
        "solvation": ({
            "mode": "implicit",
            "implicit_model": solv.get("implicit_model"),
            "radii": solv.get("radii"),
        } if solv.get("mode") == "implicit" else {
            "mode": "explicit",
            "water_model": solv.get("water_model"),
            "box_shape": solv.get("box_shape"),
            "padding_nm": solv.get("padding_nm"),
            "ionic_strength_molar": solv.get("ionic_strength_molar"),
            "positive_ion": solv.get("positive_ion", "Na+"),
            "negative_ion": solv.get("negative_ion", "Cl-"),
        }),
    }
    if doc["input"]["route"] == "smiles":
        from .schemas import sha256_text

        canonical = system.get("smiles")
        doc["input"]["smiles"] = canonical
        doc["input"]["canonical_isomeric_smiles"] = canonical
        # The hash of the string this file itself declares. It catches an edited SMILES even in an
        # environment that cannot parse chemistry, which is the point of storing it rather than
        # re-deriving it on read.
        doc["input"]["canonical_smiles_sha256"] = sha256_text(canonical or "")
    else:
        doc["input"]["pdb"] = system["input_file"]
        doc["input"]["pdb_sha256"] = system["input_sha256"]
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
