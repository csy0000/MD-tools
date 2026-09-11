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
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..openmm.system_defaults import DEFAULT_PADDING_NM, SOLVENTS, canonical_solvent, is_implicit
from .record import LogWriter, file_facts, openmm_platform_facts
from .strict import ConfigError, Field, Schema, Section, load_yaml_strictly

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
            Field("kind", str, default="peptide",
                  enum=("peptide", "peptide-like", "ligand"),
                  doc="What the solute IS, which decides how it is parameterised and what "
                      "chemistry may be read from it. This is the authoritative "
                      "classification.\n"
                      "  peptide       -- read -i as a peptide/protein PDB and parameterise it "
                      "with the protein force field. Sage never touches it.\n"
                      "  ligand        -- read -i as a .smi and parameterise the whole molecule "
                      "with the small-molecule force field. One residue, no peptide chemistry "
                      "is claimed or read.\n"
                      "  peptide-like  -- the SAME whole-molecule route as `ligand`, with the "
                      "same force field and the same charges, PLUS a validated peptide-chemistry "
                      "map over the result. It exists for a head-to-tail cyclic peptide built "
                      "from SMILES, whose residues are real amino acids but which a "
                      "single-residue ligand representation cannot describe -- so "
                      "residue-keyed corrections such as mbondi3's silently miss it. It never "
                      "loads a protein force field and never replaces Sage's charges or bonded "
                      "terms."),
            Field("peptide", bool, default=None, nullable=True,
                  doc="RETIRED spelling of `kind`, kept so configurations written before `kind` "
                      "existed -- including every `resolved.config` already on disk -- still "
                      "read. true means kind: peptide, false means kind: ligand, and there is no "
                      "boolean for peptide-like. Resolved into `kind` BEFORE defaults are "
                      "applied, so a stated boolean is never compared against a `kind` nobody "
                      "wrote. Stating both is accepted only when they agree exactly; anything "
                      "else is refused with a migration message rather than silently preferring "
                      "one."),
            Field("ligand_forcefield", str, default="sage-2.2.1",
                  doc="Small-molecule force field, used only when peptide is false. Two families "
                      "are supported. `sage-2.2.1` (the default) is OpenFF Sage, applied through "
                      "SMIRNOFF. `gaff2` selects the newest installed GAFF 2.x and is resolved to "
                      "its exact version -- `gaff-2.2.20` here -- because GAFF2 has been "
                      "distributed as several different parameter sets and the ambiguous label "
                      "does not identify a Hamiltonian. An exact version such as `gaff-2.11` may "
                      "be written instead; one that is not installed is refused, naming what is."),
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
                  doc="HBonds constrains the LENGTH of every bond to a hydrogen, and permits the "
                      "2 fs default timestep. It constrains NO ANGLE -- not even H-X-H. That is "
                      "the point: with the X-H stretches frozen the fastest remaining motions "
                      "are the hydrogen bond-angle vibrations at ~10 fs, which 2 fs resolves. "
                      "OpenMM's HAngles would constrain those too and is deliberately not "
                      "offered. AllBonds additionally freezes heavy-atom bond lengths.\n"
                      "  HOW the constraints are solved is OpenMM's choice, not a setting here "
                      "and not selectable: CCMA for the general case, which is every X-H "
                      "constraint in an ordinary solute, and SETTLE for rigid three-site water. "
                      "There is no SHAKE in OpenMM. See docs/scientific-defaults.md section 11.2."),
            Field("rigid_water", bool, default=True,
                  doc="Hold water rigid -- the case OpenMM solves with SETTLE, which needs a real "
                      "rigid triangle (both O-H bonds and the H-H distance). Forced to false "
                      "under implicit solvent, where there is no water to hold rigid, and "
                      "recorded as the resolved value rather than the requested one; such a run "
                      "therefore uses CCMA and nothing else."),
        ], doc="Constraints. These change the serialised System."),
        Section("hydrogen_mass_repartitioning", [
            Field("enabled", bool, default=False,
                  doc="Whether to repartition hydrogen masses. FALSE by default: the serialised "
                      "masses are then the force field's own. Repartitioning rewrites particle "
                      "masses in built.xml, which is a property of the System -- it cannot be "
                      "inferred later from a configuration that merely asks for a 4 fs timestep, "
                      "which is exactly why it is stated here and verified from the System at "
                      "run time."),
            Field("hydrogen_mass_amu", float, default=3.024, minimum=1.0, maximum=6.0,
                  unit="amu",
                  doc="Target hydrogen mass when enabled. Mass is moved FROM the bonded heavy "
                      "atom, so the total is conserved; water is never repartitioned, because "
                      "rigid water's hydrogen masses do not limit the timestep. Ignored, and "
                      "recorded as ignored, when enabled is false."),
        ], doc="Hydrogen mass repartitioning, stated explicitly rather than implied by a null."),
    ],
)


def _check_hmr_constraints(resolved: dict[str, Any]) -> None:
    """Repartitioning without X-H constraints is the combination that silently does nothing good.

    HMR buys a longer timestep by slowing the X-H stretch. If those bonds are NOT constrained the
    stretch is still the fastest motion in the system, so the repartitioning has moved mass around
    -- changing the dynamics -- without removing the thing that limits the step. Worse, the user
    asked for it in order to run at 4 fs, and 4 fs on unconstrained X-H does not integrate.

    Refused rather than warned: there is no reading of this configuration under which it does what
    the person who wrote it wanted.
    """
    if not resolved["hydrogen_mass_repartitioning"]["enabled"]:
        return
    kind = str(resolved["constraints"]["type"])
    if kind == "None":
        raise ConfigError(
            "hydrogen_mass_repartitioning.enabled is true but constraints.type is 'None'.\n"
            "  Repartitioning lengthens the stable timestep by slowing the X-H stretch. With the "
            "X-H bonds unconstrained that stretch is still the fastest motion in the system, so "
            "the masses would be changed -- altering the dynamics -- without buying the timestep "
            "the change is for.\n"
            "  Set constraints.type to HBonds (or AllBonds), or set "
            "hydrogen_mass_repartitioning.enabled to false.")


def _check_pairings(resolved: dict[str, Any]) -> None:
    """Refuse combinations that are individually valid and jointly wrong."""
    solvent = canonical_solvent(resolved["solvent"]["model"])
    implicit = is_implicit(solvent)
    if implicit:
        # ff19SB with GBn2 is refused HERE, where the protein force field the file asked for is
        # still visible. Downstream it is not: the implicit branch of `_sys_document` takes the
        # protein from `sys_defaults`, so an ff19SB request became an ff14SB build silently and
        # `_check_protein_solvation_pairing` -- written for exactly this pair -- never saw a
        # document that still said ff19SB. A refusal placed where the evidence is already gone is
        # not a refusal.
        #
        # Which force fields have no GB parameterisation is NOT decided again here: the list is
        # `GB_INCOMPATIBLE_PROTEIN`, and this reads it.
        from ..openmm.system_defaults import (GB_INCOMPATIBLE_PROTEIN,
                                              IMPLICIT_PROTEIN_FORCEFIELD)

        protein = str(resolved["forcefield"]["protein"])
        if any(marker.lower() in protein.lower() for marker in GB_INCOMPATIBLE_PROTEIN):
            raise ConfigError(
                f"forcefield.protein = {protein!r} is not parameterised for solvent.model = "
                f"{solvent!r}.\n"
                f"  ff19SB's amino-acid-specific CMAP corrections were fit in explicit OPC water, "
                f"and no GB model has been reparameterised against them. GBn2 was developed and "
                f"validated with the ff99SB/ff14SB lineage, so this pair mixes a backbone trained "
                f"in explicit solvent with a solvation model tuned for a different one. It would "
                f"produce numbers, which is the problem: nothing fails, and the result describes "
                f"a Hamiltonian nobody validated.\n"
                f"  This is one of the two combinations that is REFUSED rather than warned about. "
                f"The crossed explicit pairs -- ff14SB/OPC and ff19SB/TIP3P -- build and warn.\n"
                f"  Use the matched pair:\n"
                f"      forcefield.protein: ff14SB   ({IMPLICIT_PROTEIN_FORCEFIELD})\n"
                f"  or switch to explicit solvent, where ff19SB belongs.")

        # Not a warning. An implicit build with a box request in the file means the person who
        # wrote it expected a box, and would read the resulting log as if they had got one.
        for key in ("padding_nm", "box_shape", "ionic_strength_molar", "cutoff_nm",
                    "positive_ion", "negative_ion"):
            if key in resolved.get("_explicit_keys", {}).get("solvent", ()):
                raise ConfigError(
                    f"solvent.{key} is set, but solvent.model is {solvent}, which is implicit: "
                    f"there is no box, no periodic boundary and no salt. Remove the key or "
                    f"choose an explicit water model.")
    # ff19SB with an explicit water model other than OPC used to be REFUSED here. It is not any
    # more: a crossed explicit pair builds and warns. Refusing it made this tool the arbiter of
    # somebody else's experiment -- reproducing a published ff19SB/TIP3P setup, or measuring the
    # water-model sensitivity the pairing exposes, are things a competent user may deliberately
    # want -- and the refusal fired BEFORE `pairing_warnings()` could see the combination, so the
    # documented warning policy was unreachable for half the combinations it described.
    #
    # The warning is raised in `md_tools.openmm.system_config.pairing_warnings` and recorded by
    # `build_topology`. Incoherent physics is still refused, elsewhere and deliberately: ff19SB
    # with GBn2 has no parameterisation at all and `resolve_sys_config` refuses it, and explicit
    # box or salt keys under GBn2 are refused above.


def _refuse_retired_hmr_key(document: dict[str, Any]) -> None:
    """`constraints.hydrogen_mass_amu` was the old way to ask for repartitioning. Say so.

    The generic unknown-key refusal would name the key and suggest `rigid_water`, which is worse
    than useless here: the value has not moved, it has changed MEANING. It used to be
    null-means-off, where the number and the on/off switch were the same field, so a file could
    not distinguish "HMR off" from "HMR on at the default mass" without knowing that convention.

    A migration error is worth more than a suggestion, because the old file is otherwise valid and
    the user's intent is unambiguous.
    """
    constraints = document.get("constraints")
    if not isinstance(constraints, dict) or "hydrogen_mass_amu" not in constraints:
        return
    value = constraints["hydrogen_mass_amu"]
    if value is None:
        wanted = "  hydrogen_mass_repartitioning:\n    enabled: false"
        meaning = "`null` meant repartitioning was OFF"
    else:
        wanted = (f"  hydrogen_mass_repartitioning:\n    enabled: true\n"
                  f"    hydrogen_mass_amu: {value}")
        meaning = f"a value meant repartitioning was ON with a target of {value} amu"
    raise ConfigError(
        f"constraints.hydrogen_mass_amu has been replaced. It used to carry two facts in one "
        f"field -- {meaning} -- so a file could not say 'on, at the default mass' at all. "
        f"Repartitioning is now stated explicitly:\n\n{wanted}\n\n"
        f"Move the value into that block and delete it from `constraints`. Nothing about the "
        f"repartitioning itself changed: mass still comes from the bonded heavy atom, water is "
        f"still never repartitioned, and the result is still verified against the built System.")


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
        document = load_yaml_strictly(path.read_text(encoding="utf-8"), source=str(path)) or {}
        if not isinstance(document, dict):
            raise ConfigError(f"{path}: the document must be a mapping")
    _refuse_retired_hmr_key(document)
    stated = {name: tuple(block) for name, block in document.items() if isinstance(block, dict)}
    # ALIASES BEFORE DEFAULTS. `kind` has a default; the legacy boolean does not have one that
    # could be compared against it. Resolving here, against what the document actually STATES,
    # is what stops an injected `peptide: true` from being weighed against a `kind` nobody wrote
    # -- and stops the default `kind: peptide` from looking like a conflict with a stated
    # `peptide: false`.
    document = _resolve_solute_kind(document)
    resolved = BUILD_SCHEMA.resolve(document)
    resolved["_explicit_keys"] = stated
    _check_pairings(resolved)
    _check_hmr_constraints(resolved)
    resolved.pop("_explicit_keys")
    resolved["_stated"] = stated
    return resolved



#: The only boolean/kind pairs that mean the same thing. `peptide-like` is deliberately absent:
#: it postdates the boolean and has no truthful spelling in it, so stating both is always wrong.
_KIND_FOR_LEGACY = {True: "peptide", False: "ligand"}


def _resolve_solute_kind(document: dict[str, Any]) -> dict[str, Any]:
    """Fold a legacy `solute.peptide` into `solute.kind`, on the RAW document.

    Runs before schema resolution, so "stated" means stated and "absent" means absent. After
    defaults have been applied the two are indistinguishable, and every rule below depends on
    telling them apart.

    Returns a copy; the caller's document is not mutated.
    """
    solute = document.get("solute")
    if not isinstance(solute, dict):
        return document
    has_kind = "kind" in solute
    has_legacy = "peptide" in solute
    if not has_legacy:
        return document                       # nothing to fold; `kind` defaults if also absent

    legacy = solute["peptide"]
    if not isinstance(legacy, bool):
        raise ConfigError(
            f"solute.peptide must be true or false, got {legacy!r}. It is the retired spelling "
            f"of solute.kind; write `kind: peptide`, `kind: peptide-like` or `kind: ligand` "
            f"instead.")
    implied = _KIND_FOR_LEGACY[legacy]

    if has_kind:
        stated_kind = solute["kind"]
        if stated_kind != implied:
            raise ConfigError(
                f"solute.kind is {stated_kind!r} and the retired solute.peptide is {legacy!r}, "
                f"which means {implied!r}. They describe different solutes, and guessing which "
                f"one was meant would parameterise the molecule the other way round.\n"
                f"  Keep solute.kind and delete solute.peptide. The boolean has no spelling for "
                f"'peptide-like', so a peptide-like solute must not carry it at all.")
        # They agree. Keep the authoritative key and drop the alias, so exactly one field
        # decides the route from here on.
    updated = dict(document)
    solute = dict(solute)
    solute.pop("peptide", None)
    solute["kind"] = implied if not has_kind else solute["kind"]
    updated["solute"] = solute
    return updated


def _sys_document(resolved: dict[str, Any]) -> dict[str, Any]:
    """Map the strict build config onto the internal system-configuration document.

    One direction, one place. The builders downstream are the validated ones; this does not
    reinterpret any chemistry, it only renames.
    """
    from ..openmm.system_defaults import sys_defaults

    solvent = canonical_solvent(resolved["solvent"]["model"])
    kind = str(resolved["solute"]["kind"])
    # THE ROUTE, from the classification. `peptide-like` takes the ligand route deliberately and
    # completely: same force field, same charges, same builder. What it adds is a validated map
    # over the result, not a different parameterisation.
    peptide = kind == "peptide"
    document = sys_defaults(peptide=peptide, solvent=solvent, kind=kind)
    # DERIVED, never independently authoritative. Downstream readers that predate `kind` still
    # find the boolean they expect, but it is recomputed from `kind` at every crossing rather
    # than stored as a second opinion that could drift from it.
    document["solute"]["peptide"] = peptide
    document["solute"]["ligand_forcefield"] = resolved["solute"]["ligand_forcefield"]
    document["solute"]["ligand_charge_method"] = resolved["solute"]["ligand_charge_method"]
    protein = resolved["forcefield"]["protein"]
    if is_implicit(solvent):
        # Carried through rather than defaulted. `sys_defaults` used to decide this on its own on
        # the implicit branch, so whatever the configuration asked for became ff14SB with nothing
        # said about it. Anything with no GB parameterisation is refused in `_check_pairings`,
        # where the request is still visible; anything else has to be mapped here explicitly.
        from ..openmm.system_defaults import IMPLICIT_PROTEIN_FORCEFIELDS

        if protein not in IMPLICIT_PROTEIN_FORCEFIELDS:
            raise ConfigError(
                f"forcefield.protein = {protein!r} has no tleap resource for the implicit route. "
                f"Known: {', '.join(sorted(IMPLICIT_PROTEIN_FORCEFIELDS))}.")
        document["forcefield"]["protein"] = IMPLICIT_PROTEIN_FORCEFIELDS[protein]
    else:
        document["forcefield"]["protein"] = PROTEIN_FORCEFIELDS[protein]
    if not is_implicit(solvent):
        document["solvent"].update({
            "model": solvent,
            "padding_nm": resolved["solvent"]["padding_nm"],
            "box_shape": resolved["solvent"]["box_shape"],
            "ionic_strength_molar": resolved["solvent"]["ionic_strength_molar"],
            "positive_ion": resolved["solvent"]["positive_ion"],
            "negative_ion": resolved["solvent"]["negative_ion"],
            "cutoff_nm": resolved["solvent"]["cutoff_nm"],
        })
    hmr_block = resolved["hydrogen_mass_repartitioning"]
    document["constraints"] = {
        "type": resolved["constraints"]["type"],
        "rigid_water": (False if is_implicit(solvent)
                        else bool(resolved["constraints"]["rigid_water"])),
        # The builders take a scalar-or-null; the CONFIGURATION states it explicitly. `enabled:
        # false` collapses to null here, which is what "the force field's own masses" means to
        # `createSystem`. The user-facing block is never a null-with-a-meaning.
        "hydrogen_mass_amu": (float(hmr_block["hydrogen_mass_amu"])
                              if hmr_block["enabled"] else None),
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


def _resolution(config_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The build configuration, the keys the user stated, and what the record calls resolved.

    Record what RESOLVES, not what was requested. `resolve_sys_config` nulls the force fields that
    do not participate: a ligand-only build loads no protein force field, and an implicit build
    loads no water model. Recording the pre-resolution document would put a force field into the
    provenance that never loaded.
    """
    from ..openmm.system_config import resolve_sys_config

    resolved = resolve_build_config(config_path)
    stated = resolved.pop("_stated", {})
    return resolved, stated, resolve_sys_config(_sys_document(resolved))


def recorded_configuration(config_path: Path | None) -> tuple[dict[str, Any], dict[str, list]]:
    """What a build-top record would hold as `resolved_config` and `stated_keys` for this file.

    The same resolution `build_topology` records, so a checker can prove a configuration file is
    the one a recorded build used -- by resolving it now and comparing -- without a second copy of
    the rules. The reference exporter uses it to decide what goes into a bundle's `input/`.
    """
    _resolved, stated, sys_resolved = _resolution(config_path)
    return sys_resolved, {k: list(v) for k, v in stated.items()}


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

    resolved, stated, sys_resolved = _resolution(config_path)
    solvent = canonical_solvent(resolved["solvent"]["model"])
    implicit = is_implicit(solvent)
    kind = str(resolved["solute"]["kind"])
    peptide = kind == "peptide"
    # Two names for two different questions. `route` is how the System is BUILT and has exactly
    # two values, because there are exactly two parameterisation paths; `kind` is what the solute
    # IS and has three. `peptide-like` is a ligand build whose chemistry is then mapped.
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
            raise ConfigError(f"-i {input_path}: solute.kind is 'peptide', so the input must be "
                              f"a .pdb file, not {suffix}")
        log.field("interpreted as", "peptide/protein PDB")
        log.field("small-molecule FF", "not used (peptide-only input)")
    else:
        if suffix != ".smi":
            raise ConfigError(f"-i {input_path}: solute.kind is {kind!r}, which is built from a "
                              f"molecular graph, so the input must be a .smi file, not "
                              f"{suffix}")
        smiles, name_field = read_single_smiles(input_path)
        residue_name = _assigned_residue_name(resolved["solute"]["residue_name"], name_field,
                                              input_path)
        log.field("interpreted as",
                  "single-molecule SMILES" if kind == "ligand"
                  else "single-molecule SMILES, mapped as a peptide-like solute")
        log.field("smiles", smiles)
        log.field("residue name", f"{residue_name}   "
                                  f"({'stated' if resolved['solute']['residue_name'] else 'assigned deterministically'})")
        log.field("small-molecule FF", resolved["solute"]["ligand_forcefield"])

    log.field("solvent treatment", "implicit (no box, no ions, no barostat)" if implicit
                                   else f"explicit {solvent}, periodic")

    from ..openmm.system_config import pairing_warnings
    from ..openmm.builders import (Log as _BuilderLog, _build_explicit, _build_implicit,
                               _legacy_cfg)

    # `sys_resolved` came from `_resolution`, which records what RESOLVES; see there.
    cfg = _legacy_cfg(sys_resolved)

    # A supported-but-unvalidated combination is allowed and never silent. Emitted on stderr so a
    # person watching sees it, and recorded structurally so a reader of the DATA sees it too --
    # a warning that exists only in a terminal that has since been closed is not provenance.
    warnings = pairing_warnings(sys_resolved)
    for warning in warnings:
        print(f"build-top: WARNING: {warning['message']}", file=sys.stderr)
        log(f"  WARNING: {warning['message']}")

    log.update(
        input=file_facts(input_path),
        resolved_config=sys_resolved,
        warnings=warnings,
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
        hmr_block = resolved["hydrogen_mass_repartitioning"]
        hmr_enabled = bool(hmr_block["enabled"])
        hmr_requested = float(hmr_block["hydrogen_mass_amu"]) if hmr_enabled else None
        hydrogen_masses = sorted({round(m, 4) for atom, m in zip(pdb.topology.atoms(), masses)
                                  if atom.element is not None and atom.element.symbol == "H"})
        hmr = {
            "enabled": hmr_enabled,
            "applied": hmr_requested is not None,
            "target_hydrogen_mass_amu": hmr_requested,
            # Present and explained rather than merely absent: a reader must be able to tell
            # "repartitioning was off" from "nobody recorded whether it was on".
            "configured_hydrogen_mass_amu": float(hmr_block["hydrogen_mass_amu"]),
            "ignored_because_disabled": (None if hmr_enabled
                                         else "enabled is false; the force field's own masses "
                                              "are serialised"),
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

        # What was ACTUALLY loaded, not what the configuration asked for. The two differ in ways
        # that matter: the water label is short but ForceField() is given the qualified resource
        # that also carries the ion templates, `amber14-all.xml` is a manifest whose protein
        # parameters come from `amber14/protein.ff14SB.xml`, and a ligand-only route loads no
        # protein force field at all. Carried in the record since `forcefield.json` and the route
        # that wrote it were retired.
        from ..openmm.forcefield_record import build_forcefield_record
        forcefield_record = build_forcefield_record(
            resolved=sys_resolved, route=route, record=record, inputs_dir=out_pdb.parent,
            artifacts={}, builder=cfg)
        log.field("forcefield", forcefield_record["protein"]["openmm_resource"]
                  or forcefield_record["ligand"]["openff_resource"])

        log.update(
            forcefield_record=forcefield_record,
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
        # THE SDF, for a molecule built from SMILES. It was written into the staging directory,
        # used to assign charges and parameters, and then deleted with the staging directory --
        # so the bond orders it carries existed only for the duration of the build.
        #
        # Bond orders are not recoverable from a topology, and the omega classifier's ligand route
        # needs them: a SMILES-built solute is ONE residue, so there is no residue evidence to
        # read and RDKit perception on the SDF is the only thing that can say which C-N bonds are
        # amides. Without this file a REST2 ladder over such a system cannot be run at all -- the
        # peptide route refuses every candidate as unclassifiable, and the ligand route has
        # nothing to read.
        #
        # Placed beside the System, with the System's stem, because that is what a later run has
        # in hand: `-s built.xml` is given, and `built.sdf` is then discoverable without a second
        # path to keep in step. A peptide build writes none, which is itself the signal that the
        # ligand route does not apply.
        # THE SOLUTE TOPOLOGY, beside the full one.
        #
        # A solute-only trajectory -- `solute_prod<N>.nc`, and the per-state streams a ladder
        # writes -- has one row per solute atom, so it cannot be opened against `built.pdb`.
        # mdtraj refuses the pair outright ("the topology and the trajectory files might not
        # contain the same atoms"), which is correct of it and useless without this file.
        #
        # Written from the SAME `solute_atom_indices` every other part of the run means by
        # "solute", so the topology and the trajectory cannot disagree about which atoms those
        # are. For an implicit system it is the whole thing, and writing it anyway costs a few
        # kilobytes and removes a special case from every analysis script.
        from ..md.stage import solute_atom_indices

        solute_indices = solute_atom_indices(pdb.topology)
        staged_solute = staging / "solute_topology.pdb"
        solute_pdb = app.Modeller(pdb.topology, pdb.positions)
        keep = {int(i) for i in solute_indices}
        solute_pdb.delete([a for a in pdb.topology.atoms() if a.index not in keep])
        with staged_solute.open("w") as handle:
            app.PDBFile.writeFile(solute_pdb.topology, solute_pdb.positions, handle,
                                  keepIds=True)
        out_solute = out_pdb.with_name(out_pdb.stem + ".solute.pdb")

        outputs = [(built_xml, out_system), (built_pdb, out_pdb),
                   (staged_solute, out_solute)]
        staged_sdf = staging / "structure" / "solute.sdf"
        out_sdf = out_system.with_suffix(".sdf") if staged_sdf.is_file() else None
        if out_sdf is not None:
            outputs.append((staged_sdf, out_sdf))
        for source, target in outputs:
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
        written_outputs = {"system_xml": file_facts(out_system),
                           "topology_pdb": file_facts(out_pdb),
                           "solute_topology_pdb": file_facts(out_solute)}
        if out_sdf is not None:
            written_outputs["solute_sdf"] = file_facts(out_sdf)
        log.update(outputs=written_outputs)
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
