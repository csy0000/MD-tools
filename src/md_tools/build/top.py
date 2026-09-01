"""`md-openmm build-top` -- one input structure to one serialised System and its PDB.

The chemistry is not reimplemented here. Solvation, parameterisation, the GBn2/mbondi3 implicit
path and the omega classification all stay in `md_tools.openmm`, where they are already covered by
the scientific tests. This module owns the *interface* around them: a strict configuration, a
deterministic reading of the input, validation that the two outputs actually describe the same
system, atomic replacement of the outputs, and a `built.log` that a person can read and a machine
can trust.

Two outputs, and they are a pair:

  built.pdb   final coordinates and topology, after hydrogens, parameterisation, solvation or
              implicit setup, ions, and box construction;
  built.xml   an OpenMM-serialised `System` whose particle order matches `built.pdb` exactly.

If those two ever disagree, every downstream force, restraint index and trajectory frame is
meaningless, so the agreement is checked before either file is put in place rather than asserted
in a comment.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..openmm.defaults import DEFAULT_PADDING_NM, SOLVENTS, canonical_solvent, is_implicit
from .record import LogWriter, file_facts, openmm_platform_facts
from .strict import ConfigError, Field, Schema, Section

#: Protein force fields this command will load. `ff14SB` is the default; `ff19SB` is offered
#: because it is the one that pairs with OPC, and pairing is checked rather than assumed.
PROTEIN_FORCEFIELDS = {
    "ff14SB": "amber14-all.xml",
    "ff19SB": "amber19-all.xml",
}

BUILD_SCHEMA = Schema(
    "md_build.config",
    doc="Topology and System construction for `md-openmm build-top`.",
    sections=[
        Section("solute", [
            Field("peptide", bool, default=True,
                  doc="true reads -i as a peptide/protein PDB and parameterises it with the "
                      "protein force field. false reads a .smi and parameterises the molecule "
                      "with the small-molecule force field. Sage never touches a peptide-only "
                      "input, and the record must not claim it did."),
            Field("ligand_forcefield", str, default="sage-2.2.1",
                  doc="Small-molecule force field, used only when peptide is false."),
            Field("ligand_charge_method", str, default="am1bcc",
                  enum=("am1bcc", "am1bccelf10", "gasteiger", "nagl"),
                  doc="Partial-charge method for the small molecule. am1bcc is the validated "
                      "default and runs on CPU; it is the slowest part of a ligand build."),
            Field("residue_name", str, default=None, nullable=True,
                  doc="Three-character residue name for a molecule read from .smi. Left null, a "
                      "deterministic name is assigned from the file and recorded, so the same "
                      ".smi always produces the same residue identity."),
        ], doc="What the input is, and how it is parameterised."),
        Section("forcefield", [
            Field("protein", str, default="ff14SB", enum=tuple(PROTEIN_FORCEFIELDS),
                  doc="Protein force field. ff19SB is intended to be paired with OPC water; the "
                      "pairing is checked."),
        ], doc="Force-field selection. The resolved resource names are recorded, not these labels."),
        Section("solvent", [
            Field("model", str, default="TIP3P", enum=tuple(SOLVENTS),
                  doc="TIP3P or OPC give an explicit, periodic, solvated system. GBn2 is IMPLICIT "
                      "solvent: no water, no box, no ions, no barostat and no NPT stage anywhere "
                      "downstream. Choosing GBn2 changes what the rest of this file may say."),
            Field("padding_nm", float, default=DEFAULT_PADDING_NM, minimum=0.5, maximum=5.0,
                  unit="nm",
                  doc="Minimum distance from the solute to the box boundary. The built box may be "
                      "grown beyond this if the nonbonded cutoff requires it; both the requested "
                      "and achieved clearances are recorded."),
            Field("box_shape", str, default="dodecahedron",
                  enum=("dodecahedron", "cube", "octahedron"),
                  doc="A rhombic dodecahedron holds ~71% of the water a cube needs for the same "
                      "clearance, so it is the default. Explicit solvent only."),
            Field("ionic_strength_molar", float, default=0.15, minimum=0.0, maximum=2.0,
                  unit="mol/L",
                  doc="Salt added AFTER neutralising the solute charge, so the final ionic "
                      "strength is this value and the box is neutral. 0.15 M is physiological."),
            Field("positive_ion", str, default="Na+", enum=("Na+", "K+", "Li+", "Cs+", "Rb+"),
                  doc="Cation used both to neutralise and to reach the ionic strength."),
            Field("negative_ion", str, default="Cl-", enum=("Cl-", "Br-", "F-", "I-"),
                  doc="Anion used both to neutralise and to reach the ionic strength."),
            Field("cutoff_nm", float, default=1.0, minimum=0.6, maximum=2.0, unit="nm",
                  doc="Nonbonded real-space cutoff. The box must be at least twice this in its "
                      "smallest reduced height; if it is not, the box is grown and that is logged."),
        ], doc="Solvent treatment. Under GBn2 every key here except `model` is inapplicable."),
        Section("constraints", [
            Field("type", str, default="HBonds", enum=("HBonds", "AllBonds", "None"),
                  doc="HBonds constrains X-H and permits the 2 fs default timestep."),
            Field("rigid_water", bool, default=True,
                  doc="Forced to false under implicit solvent, where there is no water to hold "
                      "rigid."),
            Field("hydrogen_mass_amu", float, default=None, nullable=True, minimum=1.0,
                  maximum=6.0, unit="amu",
                  doc="Hydrogen mass repartitioning. Null (the default) means HMR is NOT applied "
                      "and the serialised masses are the force field's own. Setting this rewrites "
                      "particle masses in built.xml, which is a property of the System and cannot "
                      "be inferred later from a config that merely asks for 4 fs. Whether HMR was "
                      "applied, the target mass, and a mass summary are all recorded."),
        ], doc="Constraints and mass treatment. These change the serialised System."),
    ],
)


def _check_pairings(resolved: dict[str, Any]) -> None:
    """Refuse combinations that are individually valid and jointly wrong."""
    solvent = canonical_solvent(resolved["solvent"]["model"])
    implicit = is_implicit(solvent)
    if implicit:
        # Not a warning. An implicit build with a box request in the file means the person who
        # wrote it expected a box, and would read the resulting log as if they had got one.
        for key in ("padding_nm", "box_shape", "ionic_strength_molar", "cutoff_nm",
                    "positive_ion", "negative_ion"):
            if key in resolved.get("_explicit_keys", {}).get("solvent", ()):
                raise ConfigError(
                    f"solvent.{key} is set, but solvent.model is {solvent}, which is implicit: "
                    f"there is no box, no periodic boundary and no salt. Remove the key or "
                    f"choose an explicit water model.")
    if resolved["forcefield"]["protein"] == "ff19SB" and not implicit and solvent != "OPC":
        raise ConfigError(
            f"forcefield.protein is ff19SB but solvent.model is {solvent}. ff19SB was "
            f"parameterised against OPC water; pairing it with {solvent} is a combination neither "
            f"force field was validated for. Use ff14SB with {solvent}, or OPC with ff19SB.")


def resolve_build_config(path: Path | None) -> dict[str, Any]:
    """Resolve a build configuration, recording which keys the user actually stated.

    The set of stated keys matters: `solvent.padding_nm: 1.5` written out under GBn2 is a
    misunderstanding worth refusing, while the same value arriving as a default is not. The
    cross-field check therefore runs here, where both the resolved values and the stated keys are
    known, rather than inside the schema, which only ever sees the resolved half.
    """
    if path is None:
        document: dict[str, Any] = {}
    else:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"{path}: no such configuration file")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML -- {exc}") from None
        if not isinstance(document, dict):
            raise ConfigError(f"{path}: the document must be a mapping")
    stated = {name: tuple(block) for name, block in document.items() if isinstance(block, dict)}
    resolved = BUILD_SCHEMA.resolve(document)
    resolved["_explicit_keys"] = stated
    _check_pairings(resolved)
    resolved.pop("_explicit_keys")
    resolved["_stated"] = stated
    return resolved


def _sys_document(resolved: dict[str, Any]) -> dict[str, Any]:
    """Map the strict build config onto the internal system-configuration document.

    One direction, one place. The builders downstream are the validated ones; this does not
    reinterpret any chemistry, it only renames.
    """
    from ..openmm.defaults import sys_defaults

    solvent = canonical_solvent(resolved["solvent"]["model"])
    peptide = bool(resolved["solute"]["peptide"])
    document = sys_defaults(peptide=peptide, solvent=solvent)
    document["solute"]["peptide"] = peptide
    document["solute"]["ligand_forcefield"] = resolved["solute"]["ligand_forcefield"]
    document["solute"]["ligand_charge_method"] = resolved["solute"]["ligand_charge_method"]
    if not is_implicit(solvent):
        document["forcefield"]["protein"] = PROTEIN_FORCEFIELDS[resolved["forcefield"]["protein"]]
        document["solvent"].update({
            "model": solvent,
            "padding_nm": resolved["solvent"]["padding_nm"],
            "box_shape": resolved["solvent"]["box_shape"],
            "ionic_strength_molar": resolved["solvent"]["ionic_strength_molar"],
            "positive_ion": resolved["solvent"]["positive_ion"],
            "negative_ion": resolved["solvent"]["negative_ion"],
            "cutoff_nm": resolved["solvent"]["cutoff_nm"],
        })
    document["constraints"] = {
        "type": resolved["constraints"]["type"],
        "rigid_water": (False if is_implicit(solvent)
                        else bool(resolved["constraints"]["rigid_water"])),
        "hydrogen_mass_amu": resolved["constraints"]["hydrogen_mass_amu"],
    }
    document.pop("dataset", None)          # registration is a separate command now
    return document


def read_single_smiles(path: Path) -> tuple[str, str | None]:
    """Exactly one SMILES record, or a refusal naming how many were found.

    A `.smi` with several molecules is a perfectly good file; it is just not something this phase
    can build one System from, and picking the first line would silently drop the rest.
    """
    records: list[tuple[str, str | None]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append((parts[0], parts[1] if len(parts) > 1 else None))
    if not records:
        raise ConfigError(
            f"{path}: contains no SMILES record. Expected one non-comment line holding a SMILES "
            f"string, optionally followed by a name.")
    if len(records) > 1:
        raise ConfigError(
            f"{path}: contains {len(records)} SMILES records; this phase builds exactly one "
            f"System from one molecule. Split the file, or keep the record you mean to build.")
    return records[0]


def _assigned_residue_name(stated: str | None, name_field: str | None, source: Path) -> str:
    """A deterministic residue identity, so the same input always yields the same name."""
    import re
    for candidate in (stated, name_field, source.stem):
        if not candidate:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(candidate)).upper()[:3]
        if cleaned:
            return cleaned.ljust(3, "X")
    return "LIG"


def build_topology(*, input_path: Path, config_path: Path | None = None,
                   out_system: Path, out_pdb: Path, out_log: Path,
                   overwrite: bool = False, echo: bool = True) -> dict[str, Any]:
    """Build one System. Returns the machine record that was written into `out_log`."""
    from openmm import XmlSerializer, app

    input_path = Path(input_path)
    out_system, out_pdb, out_log = Path(out_system), Path(out_pdb), Path(out_log)

    if not input_path.is_file():
        raise ConfigError(f"-i {input_path}: no such file")
    suffix = input_path.suffix.lower()
    if suffix not in (".pdb", ".smi"):
        raise ConfigError(
            f"-i {input_path}: expected a .pdb or .smi FILE. `-i` names a file so that the input "
            f"is unambiguous and can be hashed into the record; an inline structure or SMILES "
            f"string is not accepted.")

    existing = [p for p in (out_system, out_pdb) if p.exists()]
    if existing and not overwrite:
        raise ConfigError(
            f"refusing to replace {', '.join(str(p) for p in existing)}. Pass --overwrite to "
            f"replace them deliberately. A build that silently overwrites a System leaves any "
            f"trajectory already produced against the old one unexplainable.")

    # The outputs may name a directory that does not exist yet -- `-op data/ALA-cMD/built.pdb` is
    # the documented shape -- and the staging directory is created INSIDE the output directory so
    # that the final move is a rename within one filesystem. Both parents therefore have to exist
    # before anything else happens.
    for target in (out_system, out_pdb, out_log):
        target.parent.mkdir(parents=True, exist_ok=True)

    resolved = resolve_build_config(config_path)
    stated = resolved.pop("_stated", {})
    solvent = canonical_solvent(resolved["solvent"]["model"])
    implicit = is_implicit(solvent)
    peptide = bool(resolved["solute"]["peptide"])
    route = "peptide" if peptide else "ligand"

    log = LogWriter(out_log, record_type="build-top", echo=echo)
    log("md-openmm build-top")
    log("=" * 68)
    log.heading("Command")
    log.field("input", input_path)
    log.field("config", config_path if config_path else "(none -- built-in defaults)")
    log.field("system out", out_system)
    log.field("topology out", out_pdb)

    log.heading("Resolved configuration")
    for section in ("solute", "forcefield", "solvent", "constraints"):
        for key, value in resolved[section].items():
            if implicit and section == "solvent" and key != "model":
                continue
            origin = "set" if key in stated.get(section, ()) else "default"
            log.field(f"{section}.{key}", f"{value}   ({origin})")

    log.heading("Input interpretation")
    smiles = residue_name = None
    if peptide:
        if suffix != ".pdb":
            raise ConfigError(f"-i {input_path}: solute.peptide is true, so the input must be a "
                              f".pdb file, not {suffix}")
        log.field("interpreted as", "peptide/protein PDB")
        log.field("small-molecule FF", "not used (peptide-only input)")
    else:
        if suffix != ".smi":
            raise ConfigError(f"-i {input_path}: solute.peptide is false, so the input must be a "
                              f".smi file, not {suffix}")
        smiles, name_field = read_single_smiles(input_path)
        residue_name = _assigned_residue_name(resolved["solute"]["residue_name"], name_field,
                                              input_path)
        log.field("interpreted as", "single-molecule SMILES")
        log.field("smiles", smiles)
        log.field("residue name", f"{residue_name}   "
                                  f"({'stated' if resolved['solute']['residue_name'] else 'assigned deterministically'})")
        log.field("small-molecule FF", resolved["solute"]["ligand_forcefield"])

    log.field("solvent treatment", "implicit (no box, no ions, no barostat)" if implicit
                                   else f"explicit {solvent}, periodic")

    document = _sys_document(resolved)

    from ..openmm.config import resolve_sys_config
    from ..openmm.sysgen import Log as _BuilderLog, _build_explicit, _build_implicit, _legacy_cfg

    # Record what RESOLVES, not what was requested. `resolve_sys_config` nulls the force fields
    # that do not participate: a ligand-only build loads no protein force field, and an implicit
    # build loads no water model. Recording the pre-resolution document would put a force field
    # into the provenance that never loaded.
    sys_resolved = resolve_sys_config(document)
    cfg = _legacy_cfg(sys_resolved)

    log.update(
        input=file_facts(input_path),
        resolved_config=sys_resolved,
        stated_keys={k: list(v) for k, v in stated.items()},
        interpretation={"route": route, "smiles": smiles, "residue_name": residue_name,
                        "input_format": suffix.lstrip(".")},
    )
    log.field("protein FF", sys_resolved["forcefield"].get("protein") or "not used")
    log.field("water FF", sys_resolved["forcefield"].get("water") or "not used")

    staging = Path(tempfile.mkdtemp(prefix=".build-top-", dir=str(out_pdb.parent.resolve())))
    try:
        log.heading("Preparation")
        # echo=False: this module re-emits the builder's lines into its own log below, and
        # letting the builder echo too prints every preparation line twice.
        builder_log = _BuilderLog(staging / "builder.log", echo=False)
        builder = _build_implicit if implicit else _build_explicit
        record = builder(input_path, cfg, staging, route=route, log=builder_log)
        for line in builder_log.lines:
            log(f"  {line}")

        built_pdb = Path(record["topology_pdb"])
        built_xml = Path(record["system_xml"])
        pdb = app.PDBFile(str(built_pdb))
        system = XmlSerializer.deserialize(built_xml.read_text())

        # --- the pair must describe one system -------------------------------------------
        log.heading("Validation")
        n_pdb = pdb.topology.getNumAtoms()
        n_sys = system.getNumParticles()
        if n_pdb != n_sys:
            raise ConfigError(
                f"built.pdb has {n_pdb} atoms but built.xml has {n_sys} particles. These two "
                f"files are a pair and every downstream index depends on their order agreeing; "
                f"neither output has been written.")
        log.field("particle agreement", f"{n_pdb} atoms == {n_sys} particles  OK")

        periodic = system.usesPeriodicBoundaryConditions()
        if implicit and periodic:
            raise ConfigError("an implicit-solvent System must not be periodic, but this one is")
        if not implicit and not periodic:
            raise ConfigError(f"an explicit-solvent ({solvent}) System must be periodic, but this "
                              f"one is not")
        log.field("periodicity", f"{'periodic' if periodic else 'non-periodic'}  OK "
                                 f"({'explicit' if not implicit else 'implicit'})")

        box = None
        if periodic:
            vectors = system.getDefaultPeriodicBoxVectors()
            box = [[float(v.x), float(v.y), float(v.z)] for v in vectors]
            log.field("box vectors (nm)", "; ".join(f"({v[0]:.3f} {v[1]:.3f} {v[2]:.3f})"
                                                    for v in box))

        masses = [system.getParticleMass(i).value_in_unit_system(
            __import__("openmm").unit.md_unit_system) for i in range(n_sys)]
        hmr_requested = resolved["constraints"]["hydrogen_mass_amu"]
        hydrogen_masses = sorted({round(m, 4) for atom, m in zip(pdb.topology.atoms(), masses)
                                  if atom.element is not None and atom.element.symbol == "H"})
        hmr = {
            "applied": hmr_requested is not None,
            "target_hydrogen_mass_amu": hmr_requested,
            "distinct_hydrogen_masses_amu": hydrogen_masses[:12],
            "total_mass_amu": round(sum(masses), 4),
            "recommended_timestep_fs": 4.0 if hmr_requested is not None else 2.0,
        }
        log.field("HMR", ("applied, target "
                          f"{hmr_requested} amu, recommend {hmr['recommended_timestep_fs']} fs"
                          if hmr_requested is not None else
                          "NOT applied; masses are the force field's own, recommend 2.0 fs"))
        log.field("distinct H masses", hydrogen_masses[:6])

        residues = list(pdb.topology.residues())
        waters = sum(1 for r in residues if r.name in ("HOH", "WAT", "SOL"))
        log.heading("Counts")
        log.field("atoms", n_pdb)
        log.field("residues", len(residues))
        log.field("solute atoms", record["n_solute_atoms"])
        log.field("waters", waters if not implicit else "0 (implicit solvent)")
        _ions = record.get("ions") or {}
        log.field("ions", _ions.get("counts", _ions) if not implicit else "none")

        log.update(
            counts={"atoms": n_pdb, "particles": n_sys, "residues": len(residues),
                    "solute_atoms": record["n_solute_atoms"], "waters": waters},
            periodic=bool(periodic),
            box_vectors_nm=box,
            box_geometry=record.get("box_geometry"),
            solvent={"treatment": "implicit" if implicit else "explicit",
                     "model": solvent,
                     "water_model": record.get("water_model"),
                     "ions": record.get("ions") if not implicit else None},
            forcefield=sys_resolved.get("forcefield"),
            hmr=hmr,
            platform=openmm_platform_facts(),
            validation={"particle_counts_match": True,
                        "periodicity_matches_solvent": True},
        )

        # --- atomic placement, only now that everything above passed ---------------------
        log.heading("Outputs")
        for source, target in ((built_xml, out_system), (built_pdb, out_pdb)):
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            shutil.copy2(source, tmp)
            os.replace(tmp, target)
            log.field(target.name, target)

        # Re-read what was actually placed. Hashing the staging copy would prove nothing about
        # the file that now exists at the destination.
        reread = app.PDBFile(str(out_pdb))
        resystem = XmlSerializer.deserialize(out_system.read_text())
        if reread.topology.getNumAtoms() != resystem.getNumParticles():
            raise ConfigError("the written outputs disagree when re-read; refusing to report "
                              "completion")
        log.field("re-read check", f"{reread.topology.getNumAtoms()} atoms == "
                                   f"{resystem.getNumParticles()} particles  OK")
        log.update(outputs={"system_xml": file_facts(out_system),
                            "topology_pdb": file_facts(out_pdb)})
        log.complete()
        log.heading("Summary")
        log(f"  built {n_pdb} particles, {len(residues)} residues, "
            f"{'implicit ' + solvent if implicit else 'explicit ' + solvent}")
        log(f"  status: completed")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.save()
    return log.record
