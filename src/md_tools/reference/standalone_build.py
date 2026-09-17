#!/usr/bin/env python
"""Rebuild a run's System and topology from its structure, without md-tools.

Copied into a reference bundle as `input/build_system.py`, beside `build_settings.json`, which holds
the values the recorded `md-openmm build-top` used. This file imports OpenMM and the chemistry
libraries `build-top` itself calls -- RDKit, the OpenFF toolkit, openmmforcefields, ParmEd, AmberTools'
tleap -- and nothing from md-tools, so it runs in `openmm-env` without the `md-openmm` command.

Every step below is one step of `build-top`, in its order, with its seeds:

    peptide, explicit   PDB -> hydrogens at pH (seeded) -> box -> water and ions (seeded) -> System
                        or a .seq -> tleap `sequence` (extended) -> PDB -> the same steps
    ligand,  explicit   SMILES -> 3D (ETKDGv3 + MMFF, seeded) -> charges -> box -> solvent -> System
                        or SDF -> coordinates as given -> charges -> box -> solvent -> System
    peptide, implicit   PDB (or a .seq, through tleap `sequence`) -> tleap (mbondi3 radii) -> ParmEd GBn2 System
    ligand,  implicit   SMILES -> 3D -> charges -> OpenFF System -> Amber files -> ParmEd GBn2 System
                        or SDF -> coordinates as given -> charges -> ... -> ParmEd GBn2 System

The result is compared with the System and topology the run used. Exit 0 means the System is
byte-identical and the topology identical apart from the date OpenMM writes into its first line.

    python build_system.py --out rebuilt

A test builds each route with `build-top` and with this file and requires the same bytes, so the two
cannot drift apart silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})
PROTEIN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME", "NHE", "NMA",
})

#: Models `Modeller.addSolvent` has a pre-equilibrated box for, and the same-site-count stand-in
#: used to pack a model it has none for. The force field, not the box, decides the water model.
NATIVE_PACKING_MODELS = frozenset({"tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"})
PACKING_STAND_IN = {"opc": "tip4pew", "opc3": "tip3p", "tip4pfb": "tip4pew", "tip3pfb": "tip3p"}
WATER_SITES = {"tip3p": 3, "tip3pfb": 3, "opc3": 3, "spce": 3, "tip4pew": 4, "tip4pfb": 4,
               "opc": 4, "tip5p": 5, "swm4ndp": 5}

#: The line `PDBFile.writeFile` dates. Two builds on different days differ in it and nowhere else.
DATED_PDB_LINE = "REMARK   1 CREATED WITH OPENMM"


def say(step: str, text: str) -> None:
    print(f"[{step}] {text}", flush=True)


# ---------------------------------------------------------------------------------------------
# Seeds. Each random step draws from its own seed, derived from one master seed.
# ---------------------------------------------------------------------------------------------
def derive_build_seed(master_seed: int, purpose: str) -> int:
    """`1 + sha256("md-tools/seed/v1|<master>|<purpose>")[:8] mod (2**31 - 2)`."""
    digest = hashlib.sha256(f"md-tools/seed/v1|{int(master_seed)}|{purpose}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (2 ** 31 - 2)


class seeded_global_random:
    """`addHydrogens` and `addSolvent` draw from Python's global `random` and take no seed."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *exc):
        random.setstate(self.state)


# ---------------------------------------------------------------------------------------------
# Step: a peptide PDB from a residue sequence, with tleap.
# ---------------------------------------------------------------------------------------------
def structure_from_sequence(path: Path, work: Path, sequence: dict) -> Path:
    """tleap's `sequence { ... }`, with the library build-top used, written as a PDB.

    The residues are read from the `.seq` itself and must be the ones build-top recorded; the
    leaprc is the recorded one. tleap places library geometry, so there is nothing to seed.
    """
    records = [line.split() for line in (raw.strip() for raw in
                                         path.read_text(encoding="utf-8").splitlines())
               if line and not line.startswith("#")]
    if len(records) != 1:
        raise SystemExit(f"{path}: expected exactly one sequence record, found {len(records)}")
    if records[0] != list(sequence["residues"]):
        raise SystemExit(f"{path}: holds {records[0]}, but build-top recorded "
                         f"{sequence['residues']}")
    if shutil.which("tleap") is None:
        raise SystemExit("tleap (AmberTools) is not on PATH; activate openmm-env")
    work.mkdir(parents=True, exist_ok=True)
    script = work / "sequence.leap"
    script.write_text("\n".join(sequence["tleap_commands"]) + "\n", encoding="utf-8")
    result = subprocess.run(["tleap", "-f", script.name], capture_output=True, text=True,
                            cwd=str(work))
    (work / "tleap.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    pdb = work / "sequence.pdb"
    if result.returncode != 0 or not pdb.is_file():
        raise SystemExit(f"tleap failed; see {work / 'tleap.log'}")
    say("structure", f"{path.name}: tleap sequence {{ {' '.join(records[0])} }} with "
                     f"{sequence['leaprc']} (extended conformation)")
    return pdb


def peptide_structure(settings: dict, work: Path) -> Path:
    """The peptide PDB build-top parameterised: the supplied file, or the one tleap made."""
    structure = HERE / settings["structure_file"]
    if str(settings.get("input_format", "pdb")).lower() == "seq":
        return structure_from_sequence(structure, work / "sequence", settings["sequence"])
    return structure


def name_molecule(mol, residue_name):
    """Apply build-top's `solute.residue_name`: RDKit's own atom names kept, only the residue renamed."""
    if not residue_name:
        return mol
    from rdkit import Chem

    names = [line[12:16] for line in Chem.MolToPDBBlock(mol).splitlines()
             if line.startswith(("HETATM", "ATOM"))]
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(
            name, residueName=str(residue_name), residueNumber=1, isHeteroAtom=True))
    mol.SetProp("_Name", str(residue_name))
    return mol


# ---------------------------------------------------------------------------------------------
# Step: a 3D structure from a SMILES string.
# ---------------------------------------------------------------------------------------------
def read_smiles(path: Path) -> str:
    records = [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]
    if len(records) != 1:
        raise SystemExit(f"{path}: expected exactly one SMILES record, found {len(records)}")
    return records[0]


def structure_from_sdf(path: Path, work: Path, residue_name=None) -> tuple[Path, Path]:
    """`(solute.sdf, solute.pdb)` from a supplied SDF, whose coordinates are used AS GIVEN.

    The counterpart of `structure_from_smiles`, and deliberately the shorter one: build-top did
    not embed or minimise this molecule either, so neither does the rebuild. It takes no
    `builder`, unlike its counterpart, because none of the structure settings apply here -- there
    is no seed, no ETKDG and no MMFF on this route to configure.
    """
    from rdkit import Chem

    work.mkdir(parents=True, exist_ok=True)
    records = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True))
    if len(records) != 1 or records[0] is None:
        raise SystemExit(f"{path}: expected exactly one readable molecule record")
    mol = records[0]
    if mol.GetNumConformers() == 0:
        raise SystemExit(f"{path}: carries no conformer, so it supplies no coordinates")
    Chem.AssignStereochemistryFrom3D(mol)
    say("structure", f"{path.name}: coordinates used as given (no embedding, no minimisation)")
    name_molecule(mol, residue_name)
    Chem.MolToMolFile(mol, str(work / "solute.sdf"))
    Chem.MolToPDBFile(mol, str(work / "solute.pdb"))
    return work / "solute.sdf", work / "solute.pdb"


def solute_structure(settings: dict, work: Path) -> tuple[Path, Path]:
    """The prepared solute, from whichever molecular-graph input build-top was given.

    One place decides it, because the two readers are not interchangeable: handing an SDF to
    `read_smiles` takes the molfile's title line as a SMILES string and builds some other
    molecule, or nothing at all.
    """
    structure = HERE / settings["structure_file"]
    residue_name = settings.get("residue_name")
    if str(settings.get("input_format", "smi")).lower() == "sdf":
        return structure_from_sdf(structure, work, residue_name)
    return structure_from_smiles(read_smiles(structure), work, settings["builder"], residue_name)


def structure_from_smiles(smiles: str, work: Path, builder: dict,
                          residue_name=None) -> tuple[Path, Path]:
    """Embed with ETKDGv3, MMFF-minimise every conformer, keep the lowest in energy.

    Returns `(solute.sdf, solute.pdb)`. The SDF carries the bond orders and formal charges the
    charge model needs; the PDB is what OpenMM's Modeller reads.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    etkdg, mmff = builder["structure"]["etkdg"], builder["structure"]["mmff"]
    work.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    seed = etkdg.get("seed")
    params.randomSeed = int(seed if seed is not None else builder["run"]["seed"])
    params.useRandomCoords = bool(etkdg["use_random_coords"])
    params.pruneRmsThresh = float(etkdg["prune_rms_thresh"])
    params.numThreads = int(etkdg["num_threads"])
    conformers = list(rdDistGeom.EmbedMultipleConfs(mol, int(etkdg["n_conformers"]), params))
    if not conformers:
        raise SystemExit(f"ETKDGv3 produced no conformers for {smiles!r}")

    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff["variant"])
    energies = []
    for cid in conformers:
        field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=int(cid))
        field.Minimize(maxIts=int(mmff["max_iterations"]),
                       energyTol=float(mmff["energy_tolerance"]),
                       forceTol=float(mmff["force_tolerance"]))
        energies.append((float(field.CalcEnergy()), int(cid)))
    energies.sort(key=lambda item: item[0])
    best = energies[0][1]
    say("structure", f"ETKDGv3 seed {params.randomSeed}: {len(conformers)} conformer(s), "
                     f"{mmff['variant']}-minimised, kept conformer {best} "
                     f"({energies[0][0]:.4f} kcal/mol)")

    keep = Chem.Mol(mol)
    keep.RemoveAllConformers()
    keep.AddConformer(mol.GetConformer(best), assignId=True)
    name_molecule(keep, residue_name)
    Chem.MolToMolFile(keep, str(work / "solute.sdf"))
    Chem.MolToPDBFile(keep, str(work / "solute.pdb"))
    return work / "solute.sdf", work / "solute.pdb"


# ---------------------------------------------------------------------------------------------
# The force field: OpenMM XML files, plus a template generator for a molecule built from SMILES.
# ---------------------------------------------------------------------------------------------
def forcefield(builder: dict, ligand_sdf: Path | None, *, ligand_only: bool):
    from openmm import app

    ff = builder["forcefield"]
    protein = None if ligand_only else ff["protein"]
    files = [x for x in ([protein] if protein else []) + [ff["water"], *ff["extra_xml"]] if x]
    forcefield = app.ForceField(*files)
    if ligand_sdf is None:
        return forcefield

    from openff.toolkit import Molecule
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

    method = str(ff["ligand_charge_method"]).lower()
    molecule = Molecule.from_file(str(ligand_sdf))
    if method == "am1bcc":
        openeye = any("OpenEye" in t.__class__.__name__
                      for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits)
        molecule.assign_partial_charges("am1bccelf10" if openeye else "am1bcc")
    elif method in ("am1bcc_nagl", "nagl"):
        from openff.nagl_models import get_models_by_type

        models = list(get_models_by_type("am1bcc", production_only=True))
        if not models:
            raise SystemExit("no production NAGL AM1-BCC model is installed")
        molecule.assign_partial_charges(str(models[-1]))
    else:
        raise SystemExit(f"unsupported ligand_charge_method {method!r}")

    resource = ff["ligand"]
    if str(resource).lower().startswith("gaff"):
        from openmmforcefields.generators import GAFFTemplateGenerator

        generator = GAFFTemplateGenerator(molecules=[molecule], forcefield=resource)
    else:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        generator = SMIRNOFFTemplateGenerator(molecules=[molecule], forcefield=resource)
    forcefield.registerTemplateGenerator(generator.generator)
    return forcefield


def write_pdb(topology, positions, path: Path) -> Path:
    """Written and read back between steps, as build-top does: the next step sees PDB precision."""
    from openmm import app

    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
    return path


def constraint_option(name):
    from openmm import app

    return None if name is None else {"HBonds": app.HBonds, "AllBonds": app.AllBonds,
                                      "None": None}[str(name)]


# ---------------------------------------------------------------------------------------------
# Explicit solvent: hydrogens, box, water and ions, System.
# ---------------------------------------------------------------------------------------------
def protonate(source: Path, work: Path, builder: dict, ligand_sdf: Path | None,
              from_smiles: bool) -> Path:
    """Delete the hydrogens and add them back at the configured pH, on the Reference platform.

    `addHydrogens` places each new hydrogen from a random direction drawn from Python's global
    `random`, then relaxes it; seeding the one and using the single-threaded Reference platform for
    the other is what makes this step reproducible. A molecule built from SMILES keeps the
    protonation its SMILES states: `addHydrogens` has templates for standard residues only.
    """
    from openmm import Platform, app
    from openmm.app import element

    pdb = app.PDBFile(str(source))
    names = {r.name.upper() for r in pdb.topology.residues()}
    ligand_only = from_smiles or not ((names - WATER_RESIDUE_NAMES - ION_RESIDUE_NAMES)
                                      & PROTEIN_RESIDUES)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    settings = builder["protonation"]
    if ligand_only and settings["skip_for_ligand"]:
        say("hydrogens", "kept as the SMILES states them")
    else:
        forcefield_ = forcefield(builder, ligand_sdf, ligand_only=ligand_only)
        if settings["delete_existing_hydrogens"]:
            modeller.delete([a for a in modeller.topology.atoms() if a.element == element.hydrogen])
        seed = derive_build_seed(int(builder["run"]["seed"]), "structure/protonation")
        with seeded_global_random(seed):
            modeller.addHydrogens(forcefield_, pH=float(settings["ph"]),
                                  variants=settings["variants"],
                                  platform=Platform.getPlatformByName("Reference"))
        say("hydrogens", f"added at pH {settings['ph']}, seed {seed}, Reference platform")
    return write_pdb(modeller.topology, modeller.positions, work / "solute_h.pdb")


def box_vectors(width: float, shape: str) -> np.ndarray:
    """OpenMM's reduced cube, rhombic dodecahedron or truncated octahedron of this width."""
    from openmm.app import Modeller

    try:
        vectors = Modeller._computeBoxVectors(None, float(width), shape)
        return np.array([[v.x, v.y, v.z] for v in vectors])
    except (AttributeError, TypeError):
        w = float(width)
        return {"cube": np.array([[w, 0, 0], [0, w, 0], [0, 0, w]], dtype=float),
                "dodecahedron": np.array([[w, 0, 0], [0, w, 0],
                                          [w / 2, w / 2, w * math.sqrt(2) / 2]]),
                "octahedron": np.array([[w, 0, 0], [w / 3, w * 2 * math.sqrt(2) / 3, 0],
                                        [-w / 3, w * math.sqrt(2) / 3,
                                         w * math.sqrt(6) / 3]])}[shape]


def box_for(modeller, builder: dict) -> np.ndarray:
    """The periodic box: the requested padding, then grown if the cutoff would not fit.

    Width is `max(2 r + padding, 2 padding)` with `r` the solute's bounding radius (OpenMM's own
    padding semantics), or the width that leaves `padding` to the nearest periodic copy. OpenMM
    refuses a cutoff above half the smallest reduced-box height, so the box grows until that
    height is `2 cutoff + margin`.
    """
    from openmm import unit

    solvation, system_build = builder["solvation"], builder["system_build"]
    shape = str(solvation["box_shape"])
    padding = float(solvation["padding_nm"])
    cutoff = float(system_build["nonbonded_cutoff_nm"])
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    radius = float(np.linalg.norm(positions - centre, axis=1).max())

    unit_cell = box_vectors(1.0, shape)
    height_fraction = float(np.min(np.diag(unit_cell)))
    if solvation["padding_semantics"] == "openmm":
        width = max(2 * radius + padding, 2 * padding)
    else:                                                    # "solute-image-gap"
        import itertools

        translation = min(float(np.linalg.norm(i * unit_cell[0] + j * unit_cell[1]
                                               + k * unit_cell[2]))
                          for i, j, k in itertools.product((-2, -1, 0, 1, 2), repeat=3)
                          if (i, j, k) != (0, 0, 0))
        width = (2 * radius + padding) / translation
    needed = 2.0 * cutoff + float(system_build["minimum_image_margin_nm"])
    if height_fraction * width < needed:
        if solvation["cutoff_fit_policy"] != "grow":
            raise SystemExit(f"a {shape} box of width {width:.3f} nm is too small for a "
                             f"{cutoff} nm cutoff")
        width = needed / height_fraction
    say("box", f"{shape}, width {width:.5f} nm (solute radius {radius:.5f} nm, padding "
               f"{padding} nm, cutoff {cutoff} nm)")
    return box_vectors(width, shape)


def solvate(source: Path, work: Path, builder: dict, ligand_sdf: Path | None,
            ligand_only: bool) -> tuple[Path, int]:
    from openmm import Vec3, app, unit

    pdb = app.PDBFile(str(source))
    forcefield_ = forcefield(builder, ligand_sdf, ligand_only=ligand_only)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    n_solute = modeller.topology.getNumAtoms()
    vectors = box_for(modeller, builder)

    solvation = builder["solvation"]
    water_file = builder["forcefield"]["water"]
    stem = str(water_file or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    implied = stem[:-4] if stem.endswith(".xml") else stem
    model = str(solvation["water_model"]).lower()
    if implied in WATER_SITES and model in WATER_SITES and WATER_SITES[implied] != WATER_SITES[model]:
        model = implied                               # the force field decides the model
    packing = model if model in NATIVE_PACKING_MODELS else PACKING_STAND_IN[model]

    seed = derive_build_seed(int(builder["run"]["seed"]), "structure/solvation")
    with seeded_global_random(seed):
        modeller.addSolvent(
            forcefield_, model=packing,
            boxVectors=unit.Quantity(tuple(Vec3(*v) for v in vectors.tolist()), unit.nanometer),
            positiveIon=solvation["positive_ion"], negativeIon=solvation["negative_ion"],
            ionicStrength=float(solvation["ionic_strength_molar"]) * unit.molar,
            neutralize=bool(solvation["neutralize"]))
    waters = sum(1 for r in modeller.topology.residues()
                 if r.name.upper() in WATER_RESIDUE_NAMES)
    say("solvent", f"{waters} {model} waters packed from a {packing} box, "
                   f"{solvation['positive_ion']}/{solvation['negative_ion']} at "
                   f"{solvation['ionic_strength_molar']} M, neutralised; seed {seed}")
    return write_pdb(modeller.topology, modeller.positions, work / "solvated.pdb"), n_solute


def repartition(system, topology, target: float, solute: set[int]) -> None:
    """Hydrogen mass repartitioning by hand, for flexible water, where OpenMM's own does not run."""
    from openmm import unit
    from openmm.app import element

    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.element == element.hydrogen and b.element != element.hydrogen:
            h, heavy = a, b
        elif b.element == element.hydrogen and a.element != element.hydrogen:
            h, heavy = b, a
        else:
            continue
        if h.residue.name.upper() in WATER_RESIDUE_NAMES:
            continue
        if h.index not in solute or heavy.index not in solute:
            continue
        m_h = system.getParticleMass(h.index).value_in_unit(unit.amu)
        if m_h <= 0.0:
            continue
        delta = float(target) - m_h
        m_heavy = system.getParticleMass(heavy.index).value_in_unit(unit.amu)
        system.setParticleMass(h.index, float(target) * unit.amu)
        system.setParticleMass(heavy.index, (m_heavy - delta) * unit.amu)


def explicit_system(solvated: Path, builder: dict, ligand_sdf: Path | None, ligand_only: bool,
                    n_solute: int):
    from openmm import NonbondedForce, app, unit

    b = builder["system_build"]
    pdb = app.PDBFile(str(solvated))
    forcefield_ = forcefield(builder, ligand_sdf, ligand_only=ligand_only)
    method = {"PME": app.PME, "LJPME": app.LJPME,
              "CutoffPeriodic": app.CutoffPeriodic}[b["nonbonded_method"]]
    scope = str(b["hmr_scope"] or "none")
    mass = b["hydrogen_mass_amu"]
    # OpenMM repartitions hydrogen mass itself and skips rigid water, which is exactly the
    # solute-only repartitioning wanted -- so it does the work whenever water is rigid.
    delegate = bool(b["rigid_water"]) and scope in ("solute", "all")
    system = forcefield_.createSystem(
        pdb.topology, nonbondedMethod=method,
        nonbondedCutoff=float(b["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=constraint_option(b["constraints"]), rigidWater=bool(b["rigid_water"]),
        removeCMMotion=bool(b["remove_cm_motion"]),
        ewaldErrorTolerance=float(b["ewald_error_tolerance"]),
        **({"hydrogenMass": float(mass) * unit.amu} if delegate else {}))
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(bool(b["use_dispersion_correction"]))
            if b["switch_distance_nm"] is not None:
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(float(b["switch_distance_nm"]) * unit.nanometer)
    if scope != "none" and not delegate:
        repartition(system, pdb.topology, float(mass),
                    set(range(system.getNumParticles())) if scope == "all"
                    else set(range(n_solute)))
    say("system", f"{b['nonbonded_method']}, cutoff {b['nonbonded_cutoff_nm']} nm, "
                  f"{b['constraints']}, rigid water {b['rigid_water']}, hydrogen mass "
                  f"{mass if scope != 'none' else 'as parameterised'}")
    return system


def build_explicit(settings: dict, work: Path) -> tuple[object, Path]:
    builder = settings["builder"]
    ligand_sdf = None
    if settings["route"] == "ligand":
        ligand_sdf, source = solute_structure(settings, work / "structure")
    else:
        source = work / "input.pdb"
        shutil.copy2(peptide_structure(settings, work), source)
    ligand_only = settings["route"] == "ligand"
    protonated = protonate(source, work, builder, ligand_sdf,
                           from_smiles=settings["route"] == "ligand")
    solvated, n_solute = solvate(protonated, work, builder, ligand_sdf, ligand_only)
    return explicit_system(solvated, builder, ligand_sdf, ligand_only, n_solute), solvated


# ---------------------------------------------------------------------------------------------
# Implicit solvent: Amber files, then ParmEd's GBn2 System.
# ---------------------------------------------------------------------------------------------
def tleap_files(structure: Path, work: Path, builder: dict, radii: str) -> tuple[Path, Path, Path]:
    """tleap writes the topology with the GB radii already in it."""
    if shutil.which("tleap") is None:
        raise SystemExit("tleap (AmberTools) is not on PATH; activate openmm-env")
    leaprc = builder["forcefield"]["protein"] or "leaprc.protein.ff14SB"
    script = work / "tleap.in"
    script.write_text("\n".join([
        f"source {leaprc}",
        f"set default PBRadii {radii}",
        f"mol = loadPdb {structure.resolve()}",
        f"saveAmberParm mol {(work / 'system.prmtop').resolve()} "
        f"{(work / 'system.rst7').resolve()}",
        f"savePdb mol {(work / 'tleap_out.pdb').resolve()}",
        "quit"]) + "\n", encoding="utf-8")
    result = subprocess.run(["tleap", "-f", str(script)], capture_output=True, text=True,
                            cwd=str(work))
    (work / "tleap.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not (work / "system.prmtop").is_file():
        raise SystemExit(f"tleap failed; see {work / 'tleap.log'}")
    say("amber", f"tleap: {leaprc}, PBRadii {radii}")
    return work / "system.prmtop", work / "system.rst7", work / "tleap_out.pdb"


def ligand_amber_files(settings: dict, work: Path) -> tuple[Path, Path, Path]:
    """The molecule's OpenFF (or GAFF) parameters, written to Amber files through ParmEd.

    Built with no constraints: `createSystem` applies them on the way back out.
    """
    import parmed
    from openmm import app

    builder = settings["builder"]
    sdf, pdb_path = solute_structure(settings, work / "structure")
    forcefield_ = forcefield(builder, sdf, ligand_only=True)
    solute = app.PDBFile(str(pdb_path))
    bare = forcefield_.createSystem(solute.topology, nonbondedMethod=app.NoCutoff,
                                    constraints=None)
    structure = parmed.openmm.load_topology(solute.topology, system=bare, xyz=solute.positions)
    structure.save(str(work / "system.prmtop"), overwrite=True)
    structure.save(str(work / "system.rst7"), format="rst7", overwrite=True)
    topology_pdb = write_pdb(solute.topology, solute.positions, work / "ligand_topology.pdb")
    say("amber", f"{builder['forcefield']['ligand']} with "
                 f"{builder['forcefield']['ligand_charge_method']} charges, through ParmEd")
    return work / "system.prmtop", work / "system.rst7", topology_pdb


def implicit_system(prmtop: Path, rst7: Path, settings: dict):
    """`changeRadii`, then ParmEd's `createSystem` with GBn2 -- not `AmberPrmtopFile`.

    `useSASA` is the ACE nonpolar term, stated rather than inherited: the two choices differ by
    about 16 kJ/mol on alanine dipeptide.
    """
    import parmed
    from openmm import app, unit
    from parmed.tools import changeRadii

    implicit, b = settings["implicit"], settings["builder"]["system_build"]
    structure = parmed.load_file(str(prmtop), xyz=str(rst7))
    changeRadii(structure, str(implicit["radii"])).execute()
    scope = str(b["hmr_scope"] or "none")
    system = structure.createSystem(
        nonbondedMethod=app.NoCutoff, constraints=constraint_option(b["constraints"]),
        implicitSolvent=getattr(app, implicit["model"]),
        useSASA=bool(implicit["nonpolar_sasa"]),
        removeCMMotion=bool(implicit["remove_cm_motion"]),
        **({"hydrogenMass": float(b["hydrogen_mass_amu"]) * unit.dalton}
           if scope != "none" else {}))
    say("system", f"{implicit['model']}, {implicit['radii']} radii, no cutoff, "
                  f"{b['constraints']}, SASA term {implicit['nonpolar_sasa']}, hydrogen mass "
                  f"{b['hydrogen_mass_amu'] if scope != 'none' else 'as parameterised'}")
    return system


def build_implicit(settings: dict, work: Path) -> tuple[object, Path]:
    from openmm import app

    if settings["route"] == "peptide":
        prmtop, rst7, leap_pdb = tleap_files(peptide_structure(settings, work), work,
                                             settings["builder"], settings["implicit"]["radii"])
    else:
        prmtop, rst7, leap_pdb = ligand_amber_files(settings, work)
    system = implicit_system(prmtop, rst7, settings)
    pdb = app.PDBFile(str(leap_pdb))
    return system, write_pdb(pdb.topology, pdb.positions, work / "topology.pdb")


# ---------------------------------------------------------------------------------------------
def undated(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(DATED_PDB_LINE)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="rebuilt", help="directory for the rebuilt files")
    parser.add_argument("--settings", default=str(HERE / "build_settings.json"))
    args = parser.parse_args(argv)

    from openmm import XmlSerializer

    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    if settings.get("unsupported"):
        raise SystemExit(settings["unsupported"])
    out = Path(args.out).resolve()                  # tleap runs in the work directory
    expected = settings["expected"]
    targets = {role: out / Path(expected[role]["file"]).name for role in ("system", "topology")}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing:
        raise SystemExit(f"refusing to replace {', '.join(existing)}")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    say("input", f"{settings['structure_file']}: {settings['route']} route, "
                 f"{settings['solvent']} solvent")
    builder = build_implicit if settings["solvent"] == "implicit" else build_explicit
    system, topology = builder(settings, work)
    targets["system"].write_text(XmlSerializer.serialize(system), encoding="utf-8")
    shutil.copy2(topology, targets["topology"])

    same = True
    for role, path in targets.items():
        reference = HERE / expected[role]["file"]
        if role == "system":
            match = hashlib.sha256(path.read_bytes()).hexdigest() == expected[role]["sha256"]
        else:
            match = undated(path) == undated(reference)
        same = same and match
        say("check", f"{path} {'IDENTICAL to' if match else 'DIFFERS from'} "
                     f"{reference.name}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
