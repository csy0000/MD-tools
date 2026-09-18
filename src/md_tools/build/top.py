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

import json
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

#: The tleap residue library a `.seq` peptide is built from, for each protein force field above.
#: tleap only makes the COORDINATES and NAMES here -- the parameters still come from the resource
#: in `PROTEIN_FORCEFIELDS` on the explicit route, and from tleap with the same leaprc on the
#: implicit one -- but the library is chosen to match, so the atom names it writes are the names
#: that force field's residue templates carry.
SEQUENCE_LEAPRC = {
    "ff14SB": "leaprc.protein.ff14SB",
    "ff19SB": "leaprc.protein.ff19SB",
}
assert set(SEQUENCE_LEAPRC) == set(PROTEIN_FORCEFIELDS)

#: Every `-i` suffix build-top reads, and the `solute.kind`s each may be built as.
INPUT_SUFFIXES = (".pdb", ".cif", ".seq", ".smi", ".sdf")

BUILD_SCHEMA = Schema(
    "md_build.config",
    doc="Topology and System construction for `md-openmm build-top`.",
    fields=[
        Field("ligands", list, default=[],
              doc="The ligand instances of a `kind: complex` build, each mapped explicitly onto a "
                  "reusable parameter package. One entry per instance:\n"
                  "  - select: {chain: B, resid: \"201\", insertion_code: \"\"}\n"
                  "    parameters: CHEMBL112/param_e932f4c4f371\n"
                  "`chain` is the chain id the input file carries (the AUTHOR chain for mmCIF); "
                  "`resid` is a quoted string. Each selector must name exactly one residue. The "
                  "residue's heavy atoms are matched to the package's chemical graph, its "
                  "hydrogens come from the package, and its deposited pose is kept. A match that "
                  "is ambiguous in a way that changes the chemistry (the two oxygens of a "
                  "carboxylic acid) is refused unless the entry adds `atom_map: {deposited atom "
                  "name: package atom name}` for every heavy atom. Repeated copies name the same "
                  "package. Every residue that is not a standard protein residue, water or ion "
                  "must be listed; nothing is guessed. Empty, and refused if set, for every other "
                  "kind."),
    ],
    sections=[
        Section("solute", [
            Field("kind", str, default="peptide",
                  enum=("peptide", "peptide-like", "ligand", "complex"),
                  doc="What the solute IS, which decides how it is parameterised and what "
                      "chemistry may be read from it. This is the authoritative "
                      "classification.\n"
                      "  peptide       -- read -i as a peptide/protein .pdb, or as a .seq holding "
                      "one line of residue names that tleap's `sequence` builds (extended), and "
                      "parameterise it with the protein force field. Sage never touches it.\n"
                      "  ligand        -- read -i as a .smi or .sdf and parameterise the whole "
                      "molecule with the small-molecule force field. One residue, no peptide "
                      "chemistry is claimed or read. A .smi states the chemistry and the "
                      "conformer is generated (ETKDGv3, then MMFF); a .sdf carries the "
                      "coordinates too and they are used as given.\n"
                      "  peptide-like  -- the SAME whole-molecule route as `ligand`, with the "
                      "same force field and the same charges, PLUS a validated peptide-chemistry "
                      "map over the result. It exists for a head-to-tail cyclic peptide built "
                      "from SMILES, whose residues are real amino acids but which a "
                      "single-residue ligand representation cannot describe -- so "
                      "residue-keyed corrections such as mbondi3's silently miss it. It never "
                      "loads a protein force field and never replaces Sage's charges or bonded "
                      "terms.\n"
                      "  complex       -- read -i as a .pdb or .cif holding protein chains and "
                      "ligand instances. The protein takes the protein force field; each ligand "
                      "instance listed under `ligands` takes the parameters of an existing "
                      "package, loaded as saved -- no charge is generated. Explicit solvent "
                      "only."),
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
            Field("compound_id", str, default=None, nullable=True,
                  doc="Catalog identity of the molecule for kind: ligand or peptide-like: a "
                      "ChEMBL id (`CHEMBL112`) or `LOCAL-<first block of the standard InChIKey>`. "
                      "The build writes the parameter package it creates under this compound. "
                      "Left null, the LOCAL form is derived from the molecule and recorded."),
            Field("aliases", list, default=[],
                  doc="Searchable names stored with a package this build creates: "
                      "`[paracetamol, acetaminophen, TYL]`. Names, not identities."),
            Field("parameters", str, default="search", nullable=True,
                  doc="Where this molecule's parameters come from. Three values:\n"
                      "  search (the default) -- look in the catalog for a package whose declared "
                      "criteria match this build: the molecule's topology, its protonation state, "
                      "and the charge method INCLUDING the implementation that would run here "
                      "(AM1-BCC through AmberTools' sqm is not AM1-BCC through OpenEye, and "
                      "neither is NAGL's graph model of it), together with the small-molecule "
                      "force field. Reuse on a match, parameterise on any difference; a near "
                      "match is a difference. Which package matched, on what, and what else was "
                      "considered, are recorded in built.log.\n"
                      "  <compound id>/param_<12 hex> -- reuse exactly that package, and search "
                      "nothing.\n"
                      "  generate -- parameterise the molecule whatever the catalog holds.\n"
                      "On either kind of reuse the prepared molecule must be the package's exact "
                      "chemical state (every hydrogen, charge and bond order, and the "
                      "stereochemistry of its coordinates); its atoms are put into package order "
                      "and no charge is generated. A reused package brings its own force field "
                      "and charges, so ligand_forcefield and ligand_charge_method may not be "
                      "stated beside a reference. Catalogs are searched in order: "
                      "ligand_catalog.path, then $MD_DATA/parameters/ligands."),
            Field("residue_name", str, default=None, nullable=True,
                  doc="Three-character residue name for a molecule read from .smi or .sdf. It is "
                      "APPLIED: the molecule's residue in built.pdb, built.solute.pdb and the "
                      "topology carries it, and the prepared molecule is written beside the "
                      "System as `<residue_name>.sdf`. Left null, a deterministic name is "
                      "assigned from the file (the .smi name field, else the file stem) and "
                      "recorded, so the same input always produces the same residue identity. "
                      "Three letters or digits; a name that already means water, an ion or a "
                      "protein residue is refused, because solvent selection and the omega "
                      "classifier read residue names. Refused for kind: peptide, whose residues "
                      "are named by the input."),
        ], doc="What the input is, and how it is parameterised."),
        Section("input", [
            Field("assembly", str, default=None, nullable=True,
                  doc="Build this BIOLOGICAL ASSEMBLY of an mmCIF input (the `_pdbx_struct_assembly` "
                      "id, quoted: \"3\"), not its asymmetric unit. They are different molecules: "
                      "1TYL's asymmetric unit is an insulin dimer, its assembly 3 the T3R3 "
                      "hexamer. Every copy of a chain gets its own chain id (A, B, C ... in "
                      "operator order), and build/assembly.json maps each back to its author "
                      "chain, label_asym ids and operator. An ion or water every operator places "
                      "on the same symmetry-axis position is kept once, and each dropped copy is "
                      "recorded; coinciding protein or ligand atoms are refused. `ligands` "
                      "selectors and `protonation.overrides` name the EXPANDED chain ids. Null "
                      "builds the file as deposited. Only for a .cif input and kind: peptide or "
                      "complex."),
            Field("missing_atoms", str, default="refuse", enum=("refuse", "add"),
                  doc="What to do when a standard residue lacks heavy atoms -- a disordered "
                      "surface side chain, a missing terminal OXT. `refuse` (the default) stops "
                      "the build and lists every such residue and the atoms it lacks. `add` builds "
                      "them with PDBFixer and records every added atom in built.log; they carry "
                      "no crystallographic evidence. Missing RESIDUES inside a chain (a C-N "
                      "break above 2 A) are always refused: that is loop modelling. Only for "
                      "kind: peptide or complex built from a .pdb or .cif."),
        ], doc="How the structure file is read."),
        Section("protonation", [
            Field("method", str, default="openmm", enum=("openmm", "propka"),
                  doc="How titratable protein residues get their protonation states.\n"
                      "  openmm -- Modeller.addHydrogens(pH) chooses, as md-tools always did.\n"
                      "  propka -- PROPKA3 predicts pKa values on the prepared structure (ligands "
                      "and ions kept), and md-tools assigns variants by one stated rule: ASP/GLU "
                      "protonated (ASH/GLH) when pKa > pH, LYS neutral (LYN) when pKa < pH, HIS "
                      "doubly protonated (HIP) when pKa > pH and otherwise NEUTRAL, with OpenMM's "
                      "hydrogen-bond heuristic choosing HID or HIE -- PROPKA does not resolve that "
                      "tautomer -- and CYX for disulfides. A predicted state no supported variant "
                      "builds (deprotonated CYS, tyrosinate, neutral ARG, a changed terminus) is "
                      "REPORTED and the standard state kept. PROPKA missing or failing is an "
                      "error, never a fall back. Predictions within `near_ph_window` of the pH "
                      "are flagged. Everything is recorded in build/protonation.json.\n"
                      "  Either way the states are held fixed for the run: this is not "
                      "constant-pH MD."),
            Field("ph", float, default=7.0, minimum=0.0, maximum=14.0,
                  doc="Target pH."),
            Field("overrides", list, default=[],
                  doc="Explicit per-residue variants, which win over any prediction and are "
                      "reported when they do:\n"
                      "  - select: {chain: A, resid: \"102\", insertion_code: \"\"}\n"
                      "    variant: HIE\n"
                      "One of ASP ASH GLU GLH HID HIE HIP LYS LYN CYS CYX, of the residue's own "
                      "family. An unknown selector or another family's variant is refused."),
            Field("histidine_proximity_angstrom", float, default=5.0, minimum=0.0, maximum=20.0,
                  unit="A",
                  doc="A histidine with a heavy atom within this distance of a ligand or ion "
                      "heavy atom is printed as a WARNING: chain/resid/icode, the neighbour, the "
                      "distance, the predicted pKa, the final variant and where it came from, and "
                      "for an ion the ND1-ion and NE2-ion distances separately. Screening only: "
                      "proximity is not coordination, and no tautomer is imposed; set an override "
                      "if it matters."),
            Field("near_ph_window", float, default=1.0, minimum=0.0, maximum=7.0,
                  doc="A predicted pKa within this many units of the pH is flagged as "
                      "near-pH, so its assigned state reads as uncertain."),
        ], doc="Protonation of titratable protein residues."),
        Section("ligand_catalog", [
            Field("path", str, default=None, nullable=True,
                  doc="A directory laid out as `<compound id>/<parameter id>/`, searched FIRST "
                      "for `solute.parameters` and `ligands[].parameters` -- a registered "
                      "catalog, or the `ligands/` directory of an earlier build. Relative to the "
                      "configuration file. The shared catalog $MD_DATA/parameters/ligands is "
                      "searched after it when $MD_DATA is known. A build only reads a catalog; "
                      "registration writes to it."),
        ], doc="Where reusable ligand parameter packages are looked up."),
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


def _refuse_retired_remove_key(document: dict[str, Any]) -> None:
    """`input.remove` deleted crystallisation additives. Removing them is the reader's own edit.

    The generic unknown-key refusal would say `remove` is not a key of `input` and list the two
    that are, which tells a reader holding a working configuration nothing about what to do. The
    feature existed, it was removed deliberately, and the replacement is one line of shell -- so
    the refusal carries the migration, as the retired HMR key below does.
    """
    section = document.get("input")
    if not isinstance(section, dict) or "remove" not in section:
        return
    raise ConfigError(
        "input.remove is retired. Deleting a crystallisation additive (EDO, SCN, a cryoprotectant) "
        "is an edit to your own structure file, not a build setting. Strip the residues before "
        "building, for example\n"
        "    grep -v -E \"^HETATM .* (EDO|SCN) \" deposited.cif > prepared.cif\n"
        "and pass the edited file to -i. That edits the DEPOSITED ids, before any input.assembly "
        "expansion, so a residue present in several copies is removed from all of them. Record in "
        "your own notes what you removed and why: the build no longer does.")


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
    _refuse_retired_remove_key(document)
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
    _check_residue_name(resolved)
    _check_ligand_settings(resolved)
    _check_input_and_protonation(resolved)
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
    # A complex loads the protein force field exactly as a peptide does; its ligands come from
    # packages, which the builder adds, so the ligand-only nulling of the protein must not apply.
    peptide = kind in ("peptide", "complex")
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
    document.setdefault("protonation", {}).update({
        "method": resolved["protonation"]["method"],
        "ph": float(resolved["protonation"]["ph"]),
        "overrides": list(resolved["protonation"]["overrides"]),
        "histidine_proximity_angstrom": float(resolved["protonation"]["histidine_proximity_angstrom"]),
        "near_ph_window": float(resolved["protonation"]["near_ph_window"]),
    })
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


def reserved_residue_names() -> frozenset[str]:
    """Residue names a molecule must not be given, because something already reads them.

    `solute_atom_indices` decides what is solvent by residue name, so a molecule called `HOH` or
    `NA` would silently vanish from the solute; the omega classifier decides residue evidence
    against `PROTEIN_RESIDUES`, so a molecule called `ALA` or `ACE` would have its amides read as
    a protein backbone's instead of from its bond orders.
    """
    from ..md.stage import SOLVENT_RESIDUES
    from ..openmm.system import ION_RESIDUE_NAMES, PROTEIN_RESIDUES, WATER_RESIDUE_NAMES

    return frozenset(name.upper() for name in (*SOLVENT_RESIDUES, *WATER_RESIDUE_NAMES,
                                               *ION_RESIDUE_NAMES, *PROTEIN_RESIDUES))


def _check_residue_name(resolved: dict[str, Any]) -> None:
    """A stated `solute.residue_name` is applied as written, so it has to be one that can be."""
    import re

    stated = resolved["solute"].get("residue_name")
    if stated is None:
        return
    kind = str(resolved["solute"]["kind"])
    if kind in ("peptide", "complex"):
        raise ConfigError(
            f"solute.residue_name = {stated!r} is set, but solute.kind is 'peptide': a peptide's "
            f"residues are named by its .pdb or .seq input, so this name would be applied to "
            f"nothing. Remove the key, or use kind: ligand / peptide-like for a molecule read "
            f"from .smi or .sdf.")
    text = str(stated)
    if not re.fullmatch(r"[A-Za-z0-9]{3}", text):
        raise ConfigError(
            f"solute.residue_name = {stated!r} is not three letters or digits. It is written into "
            f"the PDB residue-name column and used as the file name of the prepared molecule "
            f"(`<residue_name>.sdf`), so it is applied exactly rather than trimmed or padded.")
    if text.upper() in reserved_residue_names():
        raise ConfigError(
            f"solute.residue_name = {stated!r} already names water, an ion or a protein residue. "
            f"Solvent selection and the omega classifier read residue names, so a molecule called "
            f"{text.upper()} would be treated as one. Choose another name.")


def _check_ligand_settings(resolved: dict[str, Any]) -> None:
    """Package and instance settings, refused where they cannot apply. Nothing is loaded here."""
    from ..ligands.catalog import parse_reference
    from ..ligands.identity import CompoundIdError, check_compound_id
    from ..ligands.package import PackageError

    solute = resolved["solute"]
    kind = str(solute["kind"])
    entries = resolved["ligands"]
    stated = resolved.get("_explicit_keys", {}).get("solute", ())
    single = kind in ("ligand", "peptide-like")
    if kind == "complex":
        if is_implicit(canonical_solvent(resolved["solvent"]["model"])):
            raise ConfigError(
                "solute.kind is 'complex' with an implicit solvent. The complex route builds "
                "through ForceField templates and explicit solvent only; a GBn2 protein-ligand "
                "System would need the ligand packages carried into the tleap/ParmEd route, "
                "which is not implemented.")
        if not entries:
            raise ConfigError(
                "solute.kind is 'complex' but `ligands` is empty. A structure with no ligand "
                "instances is `kind: peptide`; a complex names every ligand instance and the "
                "package that parameterises it.")
    elif entries:
        raise ConfigError(
            f"`ligands` lists {len(entries)} instance(s), but solute.kind is {kind!r}. Ligand "
            f"instances are mapped only in a `kind: complex` build.")
    # STATED, not merely resolved: `parameters` has a default ("search"), so a peptide build that
    # never mentions it must not be refused for carrying it.
    for key in ("compound_id", "parameters"):
        # `parameters: search` is the default and says nothing about a ligand, so a peptide
        # configuration that spells it out -- the shipped example does -- is not a mistake.
        if key == "parameters" and solute.get(key) in (None, "search"):
            continue
        if key in stated and solute.get(key) is not None and not single:
            raise ConfigError(f"solute.{key} is set, but solute.kind is {kind!r}; it describes the "
                              f"single molecule of a kind: ligand or peptide-like build.")
    if solute.get("aliases") and not single:
        raise ConfigError(f"solute.aliases is set, but solute.kind is {kind!r}.")
    if not all(isinstance(a, str) for a in solute.get("aliases") or []):
        raise ConfigError("solute.aliases must be a list of strings")
    stated_reference = solute.get("parameters") not in (None, "search", "generate")
    if solute.get("aliases") and stated_reference:
        raise ConfigError("solute.aliases describe a package this build CREATES; with a stated "
                          "solute.parameters reference the package already exists and keeps its "
                          "aliases.")
    if stated_reference and solute.get("compound_id"):
        raise ConfigError("solute.compound_id and a stated solute.parameters reference are both "
                          "set; the reference already names the compound.")
    try:
        if solute.get("compound_id") is not None:
            check_compound_id(solute["compound_id"])
        if stated_reference:
            parse_reference(solute["parameters"])
            for key in ("ligand_forcefield", "ligand_charge_method"):
                if key in stated:
                    raise ConfigError(
                        f"solute.{key} is stated together with a solute.parameters "
                        f"reference. A reused package brings its own force field and "
                        f"charges; a second statement could only agree with it or be "
                        f"ignored. Remove solute.{key}.")
        for n, entry in enumerate(entries):
            if not isinstance(entry, dict) or "parameters" not in entry:
                raise ConfigError(f"ligands[{n}] must be a mapping with `select` and `parameters`")
            parse_reference(entry["parameters"])
    except (CompoundIdError, PackageError) as exc:
        raise ConfigError(str(exc)) from exc


def ligand_entries(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """The `ligands:` block, normalised: selector, package reference and atom map, IN ORDER.

    Order is part of it. Two instances whose selectors are swapped are a different assignment of
    packages to residues, and comparing unordered sets would call them the same configuration.
    """
    from ..ligands.mapping import LigandSelector

    entries = []
    for n, entry in enumerate(resolved.get("ligands") or []):
        selector = LigandSelector.from_mapping(entry.get("select") or {},
                                               where=f"ligands[{n}].select")
        atom_map = entry.get("atom_map")
        entries.append({"selector": selector.as_dict(),
                        "parameters": str(entry.get("parameters")),
                        "atom_map": dict(atom_map) if atom_map else None})
    return entries


def recorded_ligands(config_path: Path | None) -> list[dict[str, Any]]:
    """What a build-top record would hold as `ligand_packages.instances` selectors for this file.

    The counterpart of `recorded_configuration` for the one block that resolution does not carry.
    """
    return ligand_entries(resolve_build_config(config_path))


def _check_input_and_protonation(resolved: dict[str, Any]) -> None:
    """`input.assembly` and `protonation` settings, refused where they cannot apply.

    The suffix half of `input.assembly` (a .cif) is checked in `build_topology`, where the input
    path is known. Overrides are parsed strictly here, so a malformed one refuses before anything
    is read; whether its residue exists is known only once the structure is.
    """
    from ..openmm.protonation import ProtonationError, parse_overrides

    kind = str(resolved["solute"]["kind"])
    implicit = is_implicit(canonical_solvent(resolved["solvent"]["model"]))
    stated = resolved.get("_explicit_keys", {}).get("protonation", ())
    protein = kind in ("peptide", "complex")
    if resolved["input"]["assembly"] is not None and not protein:
        raise ConfigError(
            f"input.assembly = {resolved['input']['assembly']!r}, but solute.kind is {kind!r}. A "
            f"biological assembly is expanded from a protein structure: kind: peptide or complex.")
    protonation = resolved["protonation"]
    if not protein and stated:
        raise ConfigError(
            f"protonation.{', protonation.'.join(stated)} is set, but solute.kind is {kind!r}. A "
            f"molecule built from .smi/.sdf keeps the protomer its input encodes; nothing titrates "
            f"it.")
    if implicit and (protonation["method"] == "propka" or protonation["overrides"]):
        raise ConfigError(
            "protonation.method: propka and protonation.overrides apply to the explicit-solvent "
            "route, where hydrogens are added by OpenMM. The implicit (GBn2) route builds the "
            "protein with tleap, which assigns its own residue states.")
    try:
        parse_overrides(protonation["overrides"])
    except ProtonationError as exc:
        raise ConfigError(str(exc)) from None


def catalog_roots(resolved: dict[str, Any], config_path: Path | None) -> list[Path]:
    """Where packages are looked up, in order: `ligand_catalog.path`, then $MD_DATA's catalog.

    `ligand_catalog.path` stays in the configuration as written, so no record carries this
    machine's absolute path; a relative one is resolved against the configuration file here, when
    the catalog is actually searched.
    """
    from ..ligands.catalog import default_catalog_root

    roots = []
    stated = resolved["ligand_catalog"]["path"]
    if stated is not None:
        path = Path(stated).expanduser()
        if not path.is_absolute() and config_path is not None:
            path = Path(config_path).parent / path
        roots.append(path)
    default = default_catalog_root()
    if default is not None:
        roots.append(default)
    return roots


def _assigned_residue_name(stated: str | None, name_field: str | None, source: Path) -> str:
    """A deterministic residue identity, so the same input always yields the same name.

    A candidate that cleans to a reserved name (`hoh.sdf`, `ALA` as a SMILES name) is passed
    over for the next one rather than applied: see `reserved_residue_names`. A STATED name has
    already been validated, so it arrives here unchanged apart from case.
    """
    import re

    reserved = reserved_residue_names()
    for candidate in (stated, name_field, source.stem):
        if not candidate:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(candidate)).upper()[:3]
        if cleaned:
            cleaned = cleaned.ljust(3, "X")
            if cleaned not in reserved:
                return cleaned
    return "LIG"


def read_single_sequence(path: Path) -> list[str]:
    """Exactly one record of whitespace-separated residue names, or a refusal saying what is wrong.

    `#` comment lines and blank lines are allowed. Which names exist is tleap's to decide, not
    this reader's: an unknown one is refused by tleap with its log.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"-i {path}: is not a text file ({exc}); a .seq holds one line of "
                          f"residue names, such as `ACE ALA NME`.") from exc
    records = [line.split() for line in (raw.strip() for raw in text.splitlines())
               if line and not line.startswith("#")]
    if not records:
        raise ConfigError(
            f"-i {path}: contains no sequence record. Expected one non-comment line of "
            f"whitespace-separated residue names, such as `ACE ALA NME`.")
    if len(records) > 1:
        raise ConfigError(
            f"-i {path}: contains {len(records)} sequence records; this phase builds exactly one "
            f"System from one chain, written on ONE line. Join the residues onto one line, or "
            f"keep the record you mean to build.")
    return records[0]


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
    if suffix not in INPUT_SUFFIXES:
        raise ConfigError(
            f"-i {input_path}: expected a .pdb, .cif, .seq, .smi or .sdf FILE. `-i` names a file so "
            f"that the input is unambiguous and can be hashed into the record; an inline "
            f"structure, sequence or SMILES string is not accepted.")

    # RESOLVED FIRST, and once. Every check that can refuse runs before anything is created: the
    # output parents below, and then the log. A refusal must leave nothing behind, because a
    # `-odir` holding a `built.log` is indistinguishable from a build that was attempted and
    # failed halfway.
    resolved, stated, sys_resolved = _resolution(config_path)
    kind = str(resolved["solute"]["kind"])
    peptide = kind == "peptide"

    # WHAT THE INPUT IS, AND WHETHER THIS KIND MAY BE BUILT FROM IT. Stated once, here; the
    # "Input interpretation" section below reports what was decided rather than deciding it
    # again. These used to be checked down there, so every refusal of them created the output
    # directory first.
    complex_build = kind == "complex"
    assembly_id = resolved["input"]["assembly"]
    if assembly_id is not None and suffix != ".cif":
        raise ConfigError(f"-i {input_path}: input.assembly = {assembly_id!r} expands an mmCIF "
                          f"biological assembly, so the input must be a .cif file, not {suffix}")
    if peptide and suffix not in (".pdb", ".seq") and not (suffix == ".cif" and assembly_id):
        raise ConfigError(f"-i {input_path}: solute.kind is 'peptide', so the input must be "
                          f"a .pdb or .seq file (or a .cif with input.assembly), not {suffix}")
    if complex_build and suffix not in (".pdb", ".cif"):
        raise ConfigError(f"-i {input_path}: solute.kind is 'complex', so the input must be a "
                          f".pdb or .cif structure, not {suffix}")
    if not peptide and not complex_build and suffix not in (".smi", ".sdf"):
        raise ConfigError(f"-i {input_path}: solute.kind is {kind!r}, which is built from a "
                          f"molecular graph, so the input must be a .smi or .sdf file, not "
                          f"{suffix}")
    # THE SDF'S SHAPE, here rather than only in the preparer, for the same reason. The same
    # function runs again inside the preparer, for the paths that do not come through this
    # command.
    if suffix == ".sdf":
        from ..openmm.system import read_single_sdf_molecule

        read_single_sdf_molecule(input_path)
    # A .SEQ'S SHAPE and a .SMI'S, likewise: one record each, refused before anything exists.
    sequence = read_single_sequence(input_path) if suffix == ".seq" else None
    smiles = name_field = residue_name = None
    if suffix == ".smi":
        smiles, name_field = read_single_smiles(input_path)
    if not peptide and not complex_build:
        residue_name = _assigned_residue_name(resolved["solute"]["residue_name"], name_field,
                                              input_path)

    # THE PREPARED MOLECULE'S FILE, named for its residue: `<RESNAME>.sdf` beside the System.
    out_solute = out_pdb.with_name(out_pdb.stem + ".solute.pdb")
    out_sdf = out_system.parent / f"{residue_name}.sdf" if residue_name is not None else None
    if out_sdf is not None and out_sdf in (out_system, out_pdb, out_log, out_solute):
        raise ConfigError(
            f"the prepared molecule is written to {out_sdf}, which is also named as another "
            f"output of this build. Choose different -os/-op/-log names.")
    # A `<system stem>.sdf` from a build before 0.5.4 is what the run-time preflight reads FIRST,
    # so leaving one beside a new System would pair it with a molecule it was not built from --
    # or turn a peptide build into a ligand one. Not replaced under --overwrite either: this
    # build would not write that file, so nothing would replace it.
    legacy_sdf = out_system.with_suffix(".sdf")
    if legacy_sdf != out_sdf and legacy_sdf.exists():
        raise ConfigError(
            f"refusing to build beside {legacy_sdf}. That file is the molecule an earlier build "
            f"wrote under the System's name, and the run-time preflight reads `<system "
            f"stem>.sdf` before `<residue name>.sdf`, so it would be paired with the System this "
            f"build writes. Move it aside (or build into a new directory) and run again.")

    # THE LIGAND INSTANCES of a complex, resolved and mapped before anything exists: an unknown
    # package, an ambiguous selector, an unmapped residue or a mismatched chemical state is a
    # statement about the inputs, and refuses with nothing created.
    # THE BIOLOGICAL ASSEMBLY, expanded before anything else reads the structure: ligand selectors
    # and protonation overrides name the EXPANDED chain ids. Into a private temporary directory,
    # so an assembly refusal -- or a mapping refusal that follows it -- leaves nothing behind.
    assembly_record = None
    prepared_pdb_bytes = None
    completion_record = None
    assembly_scratch = None
    # What was done to `-i` before the structure was read, for the refusals that follow: they name
    # the file the caller typed, not the temporary copy, which is gone by the time they read it.
    prepared_by: list[str] = []
    out_assembly = out_system.parent / "assembly.json"
    # THE PREPARED STRUCTURE: the expanded and/or completed coordinates protonation starts from.
    # Saved beside the System so a rebuild -- and a reference bundle -- can start from exactly it.
    out_prepared = out_pdb.with_name(out_pdb.stem + ".prepared.pdb")
    structure_input = input_path
    if assembly_id is not None:
        from ..openmm.assembly import AssemblyError, expand_assembly

        assembly_scratch = tempfile.TemporaryDirectory(prefix="build-top-assembly-")
        expanded = Path(assembly_scratch.name) / "assembly.pdb"
        try:
            assembly_record = expand_assembly(input_path, assembly_id, expanded)
        except AssemblyError as exc:
            assembly_scratch.cleanup()
            raise ConfigError(f"-i {input_path}: {exc}") from None
        prepared_pdb_bytes = expanded.read_bytes()
        structure_input = expanded
        prepared_by.append(f"input.assembly: {assembly_record['assembly_id']}")

    # MISSING HEAVY ATOMS AND CHAIN BREAKS, decided before anything reads the structure for real:
    # a break is refused, and incomplete residues are refused or built under input.missing_atoms.
    if (peptide or complex_build) and suffix in (".pdb", ".cif"):
        from openmm import app as _app

        from ..openmm.completion import CompletionError, inspect_structure

        reader = _app.PDBxFile if structure_input.suffix.lower() == ".cif" else _app.PDBFile
        try:
            original = reader(str(structure_input))
            completed_topology, completed_positions, completion_record = inspect_structure(
                original.topology, original.positions,
                missing_atoms=resolved["input"]["missing_atoms"])
        except CompletionError as exc:
            if assembly_scratch is not None:
                assembly_scratch.cleanup()
            raise ConfigError(f"-i {input_path}: {exc}") from None
        if completion_record["atoms_added"]:
            if assembly_scratch is None:
                assembly_scratch = tempfile.TemporaryDirectory(prefix="build-top-assembly-")
            completed = Path(assembly_scratch.name) / "completed.pdb"
            with completed.open("w") as handle:
                _app.PDBFile.writeFile(completed_topology, completed_positions, handle, keepIds=True)
            structure_input = completed
            prepared_pdb_bytes = completed.read_bytes()
            prepared_by.append(f"input.missing_atoms: add, "
                               f"{len(completion_record['atoms_added'])} atom(s)")

    mapped = None
    out_mapping = out_system.parent / "ligand_mapping.json"
    if complex_build:
        mapped = _map_complex_ligands(
            structure_input, resolved, config_path, input_path=input_path,
            prepared_by="after " + ", ".join(prepared_by) if prepared_by else None)
    elif not peptide and str(resolved["solute"]["parameters"] or "search") not in ("search",
                                                                                  "generate"):
        # A STATED package reference, loaded and verified here, where a refusal still leaves
        # nothing behind. The builder loads it again when it attaches it; without this the first
        # thing to read the package was several steps into the build, so a package that could not
        # be loaded -- one written before it recorded which implementation charged it, say --
        # failed after the output directory and its log already existed.
        from ..ligands.catalog import find_package
        from ..ligands.package import PackageError

        try:
            find_package(resolved["solute"]["parameters"],
                         catalog_roots(resolved, config_path))
        except PackageError as exc:
            raise ConfigError(f"solute.parameters: {exc}") from exc
    out_ligands = out_system.parent / "ligands"

    existing = [p for p in (out_system, out_pdb, *([out_sdf] if out_sdf else []),
                            *([out_mapping] if complex_build else []),
                            *([out_assembly] if assembly_record else []),
                            *([out_prepared] if prepared_pdb_bytes is not None else []))
                if p.exists()]
    if existing and not overwrite:
        raise ConfigError(
            f"refusing to replace {', '.join(str(p) for p in existing)}. Pass --overwrite to "
            f"replace them deliberately. A build that silently overwrites a System leaves any "
            f"trajectory already produced against the old one unexplainable.")

    # The outputs may name a directory that does not exist yet -- `-op data/ALA-cMD/built.pdb` is
    # the documented shape -- and the staging directory is created INSIDE the output directory so
    # that the final move is a rename within one filesystem. Both parents therefore have to exist
    # before anything else happens.
    # THE SEQUENCE'S STRUCTURE, made before any output exists. tleap is the authority on which
    # residue names are real, and its refusal is a statement about the input file -- so it has to
    # arrive the way the other input refusals do, with nothing created. It runs in a private
    # temporary directory, and the PDB it wrote is carried into staging below.
    sequence_build = None
    if sequence is not None:
        from ..openmm.implicit import build_structure_from_sequence

        leaprc = SEQUENCE_LEAPRC[str(resolved["forcefield"]["protein"])]
        with tempfile.TemporaryDirectory(prefix="build-top-sequence-") as scratch:
            made = build_structure_from_sequence(sequence, Path(scratch),
                                                 protein_forcefield=leaprc)
            sequence_build = {**made, "pdb_bytes": Path(made["pdb"]).read_bytes()}

    for target in (out_system, out_pdb, out_log):
        target.parent.mkdir(parents=True, exist_ok=True)

    solvent = canonical_solvent(resolved["solvent"]["model"])
    implicit = is_implicit(solvent)
    # Two names for two different questions. `route` is how the System is BUILT and has exactly
    # two values, because there are exactly two parameterisation paths; `kind` is what the solute
    # IS and has three. `peptide-like` is a ligand build whose chemistry is then mapped.
    route = "peptide" if peptide else ("complex" if complex_build else "ligand")

    log = LogWriter(out_log, record_type="build-top", echo=echo)
    log("md-openmm build-top")
    log("=" * 68)
    log.heading("Command")
    log.field("input", input_path)
    if assembly_record is not None:
        log.field("assembly", f"{assembly_record['assembly_id']}: "
                              f"{len(assembly_record['chains'])} chain(s), "
                              f"{len(assembly_record['deduplicated'])} on-axis copy(ies) dropped")
    if completion_record is not None and completion_record["atoms_added"]:
        added_residues = {(a["chain"], a["resid"], a["insertion_code"])
                          for a in completion_record["atoms_added"]}
        log.field("missing atoms", f"{len(completion_record['atoms_added'])} atom(s) added to "
                                   f"{len(added_residues)} residue(s) (input.missing_atoms: add)")
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
    if complex_build:
        log.field("interpreted as",
                  f"protein-ligand complex ({suffix}), {len(mapped.instances)} ligand instance(s) "
                  f"mapped to parameter packages")
        for instance in mapped.instances:
            log.field(f"ligand {instance.residue_name}",
                      f"{instance.selector.label()} -> {instance.package.reference}")
        log.field("coordinates", "the deposited pose; ligand hydrogens from each package")
        log.field("small-molecule FF", "not run: parameters loaded from the packages")
    elif peptide and sequence_build is not None:
        log.field("interpreted as", "peptide sequence, built by tleap `sequence`")
        log.field("sequence", " ".join(sequence))
        log.field("residue library", sequence_build["protein_forcefield"])
        # Said because it is the thing a reader would otherwise assume was chosen: nothing was.
        log.field("conformation", "EXTENDED -- tleap's library geometry, not sampled and not "
                                  "minimised; minimisation and equilibration start from it")
        if sequence_build["built_residues"] != sequence:
            log.field("residues written", " ".join(sequence_build["built_residues"]))
        log.field("small-molecule FF", "not used (peptide-only input)")
    elif peptide:
        log.field("interpreted as", "peptide/protein PDB")
        log.field("small-molecule FF", "not used (peptide-only input)")
    else:
        # The two molecular-graph inputs differ in exactly one thing: where the coordinates come
        # from. A `.smi` states the chemistry and the conformer is generated (ETKDGv3, then MMFF);
        # a `.sdf` carries both, and is used as given. Everything after this branch is shared.
        graph = "SMILES" if suffix == ".smi" else "SDF"
        log.field("interpreted as",
                  f"single-molecule {graph}" if kind == "ligand"
                  else f"single-molecule {graph}, mapped as a peptide-like solute")
        if smiles is not None:
            log.field("smiles", smiles)
        else:
            # Which build was free to choose its own conformer is the thing a reader comparing
            # two of them needs to know, so it is a field rather than an absence.
            log.field("coordinates", "supplied by the SDF, used as given "
                                     "(no embedding, no minimisation)")
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
    # THE RESIDUE NAME, handed to the one place that writes the prepared molecule. Both preparers
    # read it, so the .smi and .sdf routes and both solvent routes carry it into the topology.
    cfg["solute"]["residue_name"] = residue_name
    roots = catalog_roots(resolved, config_path) if (complex_build or not peptide) else []
    if not peptide and not complex_build:
        # A single molecule gets a package: reused if `solute.parameters` names one, created once
        # otherwise. The builder attaches it before its first force field is built.
        cfg["ligand_package"] = {
            "forcefield": sys_resolved["forcefield"].get("ligand"),
            "charge_method": sys_resolved["forcefield"].get("ligand_charge_method"),
            "compound_id": resolved["solute"]["compound_id"],
            "aliases": list(resolved["solute"]["aliases"] or []),
            "parameters": resolved["solute"]["parameters"],
            "catalog_roots": roots,
            "residue_name": residue_name,
            "input_name": input_path.name,
            "input_sha256": file_facts(input_path)["sha256"],
        }

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
                        "input_format": suffix.lstrip("."),
                        **({"sequence": list(sequence)} if sequence is not None else {})},
    )
    if sequence_build is not None:
        log.update(sequence={
            "residues": list(sequence),
            "residues_written": sequence_build["built_residues"],
            "leaprc": sequence_build["protein_forcefield"],
            "tleap_commands": sequence_build["tleap_commands"],
            # Not retained whole: tleap's log names every file of this machine's AmberTools
            # installation, and a record must not carry a machine path. Its lines that name no
            # path -- the verdict, the warnings, the file it wrote -- are copied into this log
            # under "Sequence (tleap)"; the digest identifies the whole text.
            "tleap_log": {"name": "tleap.log", "sha256": sequence_build["tleap_log_sha256"],
                          "bytes": len(sequence_build["tleap_log_text"].encode("utf-8")),
                          "retained": "lines naming no path, under 'Sequence (tleap)'"},
            "generated_pdb": {"name": "sequence.pdb", "sha256": sequence_build["pdb_sha256"]},
            "conformation": sequence_build["conformation"],
        })
        log.heading("Sequence (tleap)")
        for command in sequence_build["tleap_commands"]:
            log(f"  > {command}")
        for line in sequence_build["tleap_log_text"].splitlines():
            if line.strip() and "/" not in line:
                log(f"  | {line.strip()}")
    log.field("protein FF", sys_resolved["forcefield"].get("protein") or "not used")
    log.field("water FF", sys_resolved["forcefield"].get("water") or "not used")

    staging = Path(tempfile.mkdtemp(prefix=".build-top-", dir=str(out_pdb.parent.resolve())))
    try:
        log.heading("Preparation")
        # echo=False: this module re-emits the builder's lines into its own log below, and
        # letting the builder echo too prints every preparation line twice.
        builder_log = _BuilderLog(staging / "builder.log", echo=False)
        builder = _build_implicit if implicit else _build_explicit
        structure_path = input_path
        if complex_build:
            from ..openmm.builders import _build_complex

            staged_ligands = staging / "ligands"
            placed = [package.copy_into(staged_ligands) for package in mapped.packages]
            cfg["forcefield"]["ligand_packages"] = [str(path) for path in placed]

            def builder(path, cfg, staging, *, route, log):          # noqa: F811
                return _build_complex(path, cfg, staging, mapped=mapped, log=log)
        if prepared_pdb_bytes is not None:
            # From here the expanded and/or completed structure IS a PDB, built as the `.pdb`
            # route would build it.
            structure_path = staging / "prepared" / "prepared.pdb"
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_bytes(prepared_pdb_bytes)
            if assembly_scratch is not None:
                assembly_scratch.cleanup()
        if sequence_build is not None:
            # From here the sequence IS a PDB, and the build is the `.pdb` peptide route exactly.
            structure_path = staging / "sequence" / "sequence.pdb"
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_bytes(sequence_build["pdb_bytes"])
        record = builder(structure_path, cfg, staging, route=route, log=builder_log)
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

        if residue_name is not None:
            # APPLIED, and checked on the file that is about to be placed rather than trusted from
            # the preparer: a recorded name the topology does not carry is the defect this closes.
            from ..md.stage import SOLVENT_RESIDUES

            solute_names = sorted({r.name for r in pdb.topology.residues()
                                   if r.name.strip().upper() not in SOLVENT_RESIDUES})
            if solute_names != [residue_name]:
                raise RuntimeError(
                    f"the solute residue in the built topology is {solute_names}, not "
                    f"[{residue_name!r}] as resolved; neither output has been written.")
            log.field("solute residue", f"{residue_name}  OK")

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
        # THE SDF, for a molecule built from SMILES or SDF. It was written into the staging directory,
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
        # Placed beside the System and named for the solute's RESIDUE (`TYL.sdf`), which the
        # topology beside it carries: a later run has `-s built.xml` in hand, reads the one
        # non-solvent residue name from `built.pdb`, and finds the molecule without a second path
        # to keep in step (`preflight._ligand_sdf_beside`). Until 0.5.4 it was `<system
        # stem>.sdf`; that name is still read. A peptide build writes none, which is itself the
        # signal that the ligand route does not apply.
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
        outputs = [(built_xml, out_system), (built_pdb, out_pdb),
                   (staged_solute, out_solute)]
        if complex_build:
            from ..ligands.mapping import assert_instances_unchanged

            final = app.PDBFile(str(built_pdb))
            assert_instances_unchanged(mapped, final.topology, final.positions,
                                       step="the whole build")
            mapping_record = mapped.record(final.topology)
            staged_mapping = staging / "ligand_mapping.json"
            staged_mapping.write_text(json.dumps(mapping_record, indent=2) + "\n", encoding="utf-8")
            outputs.append((staged_mapping, out_mapping))
        staged_sdf = staging / "structure" / "solute.sdf"
        if (out_sdf is not None) != staged_sdf.is_file():
            raise RuntimeError(
                f"the {route} route {'did not prepare' if out_sdf is not None else 'prepared'} a "
                f"molecule SDF, which it {'must' if out_sdf is not None else 'must not'}; "
                f"refusing to report completion")
        if out_sdf is not None:
            outputs.append((staged_sdf, out_sdf))
        if prepared_pdb_bytes is not None:
            staged_prepared = staging / "prepared" / "prepared.pdb"
            outputs.append((staged_prepared, out_prepared))
        if assembly_record is not None:
            staged_assembly = staging / "assembly.json"
            staged_assembly.write_text(json.dumps(assembly_record, indent=2) + "\n",
                                       encoding="utf-8")
            outputs.append((staged_assembly, out_assembly))
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
        placed_packages = _place_packages(staging / "ligands", out_ligands)
        if placed_packages:
            log.field("ligand packages", ", ".join(
                f"{p['reference']} ({p['how']})" for p in placed_packages))
        written_outputs = {"system_xml": file_facts(out_system),
                           "topology_pdb": file_facts(out_pdb),
                           "solute_topology_pdb": file_facts(out_solute)}
        if out_sdf is not None:
            written_outputs["solute_sdf"] = {**file_facts(out_sdf), "residue_name": residue_name}
        if complex_build:
            written_outputs["ligand_mapping"] = file_facts(out_mapping)
        if assembly_record is not None:
            written_outputs["assembly"] = file_facts(out_assembly)
        if prepared_pdb_bytes is not None:
            written_outputs["prepared_structure"] = file_facts(out_prepared)
        if completion_record is not None:
            log.update(structure_completion=completion_record)
        protonation = (record.get("protonation") or {}).get("protonation")
        if protonation is not None:
            log.update(protonation=protonation)
        log.update(outputs=written_outputs,
                   ligand_packages={
                       "catalog_searched": [str(r) for r in ([resolved["ligand_catalog"]["path"]]
                                                             if resolved["ligand_catalog"]["path"]
                                                             else [])],
                       "placed_in": "ligands",
                       "packages": placed_packages,
                       "attached": record.get("ligand_package"),
                       # WHICH PACKAGE EACH INSTANCE GOT, in the order the configuration listed
                       # them, with the atom map when one was stated. This is what proves that a
                       # `ligands:` block found later is the one that produced this build:
                       # `ligands` is a top-level list, so it reaches neither `resolved_config`
                       # nor `stated_keys`, and an export comparing only those would accept an
                       # edited one. See `ligand_entries` and `reference.export.user_inputs_plan`.
                       "instances": ([{"selector": i.selector.as_dict(),
                                       "residue_name": i.residue_name,
                                       "parameters": i.package.reference,
                                       "atom_map": entry.get("atom_map")}
                                      for i, entry in zip(mapped.instances, resolved["ligands"])]
                                     if complex_build else None),
                   })
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


def _map_complex_ligands(structure_path: Path, resolved: dict[str, Any], config_path: Path | None,
                         *, input_path: Path, prepared_by: str | None = None):
    """Load every package the configuration names and map every ligand instance. Writes nothing.

    `structure_path` is the file READ, which after assembly expansion or missing-atom completion
    lives in a private temporary directory that no longer exists by the time anybody reads a
    refusal. Every refusal here therefore names `input_path` -- the `-i` the caller typed -- and
    says what was done to it (`prepared_by`): a refusal that cannot be traced back to the reader's
    own input is close to useless, and this is the refusal a person meets when reusing a parameter
    package. `built.prepared.pdb` beside the System is the durable copy of what was read.
    """
    from openmm import app

    from ..ligands.catalog import find_package
    from ..ligands.mapping import LigandSelector, MappingError, map_ligands, unmapped_residues
    from ..ligands.package import PackageError
    from ..openmm.system import ION_RESIDUE_NAMES, PROTEIN_RESIDUES, WATER_RESIDUE_NAMES

    where = f"-i {input_path}" + (f" ({prepared_by})" if prepared_by else "")
    try:
        reader = app.PDBxFile if structure_path.suffix.lower() == ".cif" else app.PDBFile
        structure = reader(str(structure_path))
    except Exception as exc:
        raise ConfigError(f"{where}: OpenMM could not read this structure ({exc})") from exc
    roots = catalog_roots(resolved, config_path)
    entries = resolved["ligands"]
    try:
        packages = {}
        for entry in entries:
            reference = entry["parameters"]
            if reference not in packages:
                packages[reference] = find_package(reference, roots)
        selectors = [LigandSelector.from_mapping(entry.get("select") or {},
                                                 where=f"ligands[{n}].select")
                     for n, entry in enumerate(entries)]
        known = set(PROTEIN_RESIDUES) | set(WATER_RESIDUE_NAMES) | set(ION_RESIDUE_NAMES)
        left = unmapped_residues(structure.topology, selectors, known_residue_names=known)
        if left:
            shown = "; ".join(f"{r['residue_name']} chain {r['chain']!r} resid {r['resid']!r}"
                              + (f" icode {r['insertion_code']!r}" if r["insertion_code"] else "")
                              for r in left[:10])
            raise ConfigError(
                f"{where}: {len(left)} residue(s) are neither standard protein residues, "
                f"water nor ions, and no `ligands` entry maps them: {shown}"
                + (" ..." if len(left) > 10 else "") + ". Each needs an entry naming its "
                f"parameter package; nothing is parameterised by guessing, and nothing is "
                f"silently deleted.")
        return map_ligands(structure.topology, structure.positions, entries, packages)
    except (PackageError, MappingError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _place_packages(staged_root: Path, out_root: Path) -> list[dict[str, Any]]:
    """Put the packages a build used beside built.xml. An identical identity already there is kept."""
    from ..ligands.catalog import register_package

    placed = []
    if not staged_root.is_dir():
        return placed
    for package_dir in sorted(staged_root.glob("*/param_*")):
        package, destination, written = register_package(package_dir, out_root)
        placed.append({**package.summary(),
                       "path": str(destination.relative_to(out_root.parent)),
                       "how": "written" if written else "already present with this identity"})
    return placed
