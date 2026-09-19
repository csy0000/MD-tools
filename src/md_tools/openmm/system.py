from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform as _platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from openmm import (CMAPTorsionForce, CustomGBForce, NonbondedForce,
                    PeriodicTorsionForce, XmlSerializer)


WATER_RESIDUE_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"})
ION_RESIDUE_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"})

# ---------------------------------------------------------------------------------------------
# Step 1 -- initial structure from SMILES (ETKDGv3 + MMFF)
# ---------------------------------------------------------------------------------------------
def _stated_residue_name(cfg: dict) -> Optional[str]:
    """The residue name build-top resolved for this molecule, or None for RDKit's own `UNL`."""
    name = (cfg.get("solute") or {}).get("residue_name")
    return str(name) if name else None


def _name_molecule(mol, residue_name: Optional[str]):
    """Give the prepared molecule its residue name, in the PDB records and as the SDF title.

    `solute.residue_name` used to be resolved, recorded -- and applied to nothing, so the built
    topology said `UNL` while the record said something else. It is applied HERE, on the one
    molecule both preparers write, so both input routes and both solvent routes inherit it from
    the same two files.

    RDKit writes `UNL` and names the atoms by element and count when an atom carries no residue
    information. Those default atom names are read back off RDKit's own PDB block and kept
    exactly: only the residue name changes, so nothing downstream sees different atom names. The
    small-molecule template generators match a residue by its bonded graph, not by its name.
    """
    if residue_name is None:
        return mol
    from rdkit import Chem

    names = [line[12:16] for line in Chem.MolToPDBBlock(mol).splitlines()
             if line.startswith(("HETATM", "ATOM"))]
    if len(names) != mol.GetNumAtoms():
        raise RuntimeError(
            f"RDKit wrote {len(names)} atom records for a {mol.GetNumAtoms()}-atom molecule; the "
            f"residue name {residue_name!r} cannot be applied atom by atom")
    for atom, name in zip(mol.GetAtoms(), names):
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(
            name, residueName=str(residue_name), residueNumber=1, isHeteroAtom=True))
    mol.SetProp("_Name", str(residue_name))
    return mol


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
    # A null seed means "unset", and RDKit's own default is a RANDOM embedding -- two runs of the
    # same SMILES would give different starting coordinates, which is the one thing a prepared
    # system must not do. Fall back to the run seed so the conformer is reproducible.
    seed = ecfg.get("seed")
    if seed is None:
        seed = (cfg.get("run") or {}).get("seed")
    if seed is None:
        raise ValueError(
            "structure.etkdg.seed and run.seed are both null, so the SMILES embedding would be "
            "random and this prepared system would not be reproducible. Set one of them.")
    params.randomSeed = int(seed)
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
    residue_name = _stated_residue_name(cfg)
    _name_molecule(keep, residue_name)
    Chem.MolToMolFile(keep, str(out_dir / "solute.sdf"))
    Chem.MolToPDBFile(keep, str(out_dir / "solute.pdb"))

    info = {
        "smiles": smiles,
        "n_conformers_embedded": len(conf_ids),
        "mmff_variant": mcfg["variant"],
        "selected_conformer": best,
        "all_conformers": records,
        "n_unconverged": sum(1 for r in records if not r["converged"]),
        "residue_name": residue_name or "UNL",
        "solute_sdf": str(out_dir / "solute.sdf"),
        "solute_pdb": str(out_dir / "solute.pdb"),
    }
    (out_dir / "initial_structure.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


# ---------------------------------------------------------------------------------------------
# Step 1, the other way in -- initial structure from a SUPPLIED SDF (coordinates as given)
# ---------------------------------------------------------------------------------------------
def read_single_sdf_molecule(sdf_path: Path):
    """The one molecule in *sdf_path*, or a `ConfigError` naming exactly what is wrong with it.

    ONE implementation, called from two places, and that is deliberate. `build-top` calls it
    beside the suffix check so that a malformed input refuses before any directory is created;
    :func:`initial_structure_from_sdf` calls it again because the generated entry points and the
    implicit route reach the preparer directly. A gate that lived only in the command would be a
    safe outer surface over an unsafe runtime, which is worse than no gate: it makes the
    unchecked path look tested.

    Raises `ConfigError` rather than `ValueError`, because every one of these is a statement
    about the file the user named on the command line -- the same class `-i`'s suffix and
    existence checks raise, and the one the CLI maps to exit 2.
    """
    from rdkit import Chem

    from ..build.strict import ConfigError

    sdf_path = Path(sdf_path)

    # `removeHs=False` is not a preference. RDKit strips explicit hydrogens by default, and the
    # hydrogens in this file are the ones being parameterised -- a silently de-protonated solute
    # would be given Sage's parameters for a different molecule while every log named this one.
    # `sanitize=True` is what the parameterisation will do anyway: a record that cannot be
    # sanitised here would fail later inside charge assignment, further from the file that caused
    # it, and it arrives as `None` and is refused below.
    #
    # The `try` is for a different failure: RDKit raises on CONSTRUCTION for a file it cannot open
    # at all -- an empty one included -- rather than yielding zero records, so "no records" and
    # "no readable file" arrive by two routes and both must become a refusal that names `-i`.
    try:
        records = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True))
    except OSError as exc:
        raise ConfigError(
            f"-i {sdf_path}: RDKit could not open this file as an SDF ({exc}). An empty file or "
            f"one whose contents are not a molfile reaches here; check that it is the file you "
            f"meant to pass.") from exc
    if not records:
        raise ConfigError(
            f"-i {sdf_path}: contains no molecule record. Expected one molecule with 3D "
            f"coordinates and explicit hydrogens.")
    if len(records) > 1:
        raise ConfigError(
            f"-i {sdf_path}: contains {len(records)} molecule records; this phase builds exactly "
            f"one System from one molecule. Split the file, or keep the record you mean to "
            f"build.")
    mol = records[0]
    if mol is None:
        raise ConfigError(
            f"-i {sdf_path}: RDKit could not read this SDF. A molecule that fails sanitisation "
            f"here would fail again inside the charge assignment, with a less useful message.")
    if mol.GetNumConformers() == 0:
        raise ConfigError(
            f"-i {sdf_path}: carries no conformer, so it supplies no coordinates. An SDF input "
            f"IS the structure; to have one generated instead, supply a .smi and the ETKDG/MMFF "
            f"route will build it.")
    if not mol.GetConformer().Is3D():
        raise ConfigError(
            f"-i {sdf_path}: its conformer is flagged two-dimensional. A flat molecule is not a "
            f"starting structure -- solvating and integrating it would begin from a geometry no "
            f"force field considers physical. Supply 3D coordinates, or a .smi to have them "
            f"generated.")
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ConfigError(
            f"-i {sdf_path}: carries no explicit hydrogens. The small-molecule route "
            f"parameterises the molecule exactly as given and adds none, so an "
            f"implicit-hydrogen SDF would be built as the heavy-atom skeleton alone. Add "
            f"hydrogens before supplying it.")
    return mol


def initial_structure_from_sdf(sdf_path: Path, out_dir: Path, cfg: dict) -> dict:
    """Take over where :func:`initial_structure` finishes, from a molecule the caller supplies.

    The SMILES route *generates* coordinates -- ETKDGv3 embeds, MMFF minimises every conformer,
    the lowest in energy is kept -- and writes exactly two files that everything downstream
    consumes: ``solute.sdf`` (bond orders and formal charges, which the OpenFF path needs) and
    ``solute.pdb`` (what Modeller reads). An SDF already carries all three, so this route writes
    those same two files and stops.

    **It does not embed and it does not minimise.** The conformer in the file is the pose the
    caller chose -- docked, crystallographic, or the output of some other pipeline -- and
    replacing it with an MMFF minimum would be a different experiment reported under the same
    name. `structure.etkdg` and `structure.mmff` are therefore not read here, and a configuration
    that sets them has not been ignored: they describe a step this route does not take.

    The provenance says so too. There is no seed and no conformer table, because nothing was
    sampled; what makes this build reproducible is the identity of the input file, so its digest
    is recorded in their place.
    """
    from rdkit import Chem

    from ..build.record import sha256_file

    out_dir = Path(out_dir)
    sdf_path = Path(sdf_path)

    mol = read_single_sdf_molecule(sdf_path)

    # Only now. Everything above can refuse, and a refusal must not leave a directory behind.
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3D is the authority for stereochemistry once coordinates exist: an SDF's parity flags and
    # its geometry can disagree, and the geometry is what will be integrated.
    Chem.AssignStereochemistryFrom3D(mol)

    # Written through RDKit rather than copied, so that both input routes hand downstream code a
    # file from ONE writer. A copied SDF would carry whatever dialect its producer used, and the
    # first thing to disagree would be `map_from_sdf`, which re-reads this file for the peptide
    # chemistry.
    residue_name = _stated_residue_name(cfg)
    _name_molecule(mol, residue_name)
    Chem.MolToMolFile(mol, str(out_dir / "solute.sdf"))
    Chem.MolToPDBFile(mol, str(out_dir / "solute.pdb"))

    info = {
        # THE SAME KEY AS THE SMILES ROUTE, AND NOT THE SAME FACT. There it is the string the user
        # supplied and the coordinates were built from; here the coordinates came first and this is
        # read back off them by RDKit. The key is shared because callers log it as "which molecule
        # is this" -- `builders._build_explicit` does -- and `smiles_origin` is what keeps the two
        # meanings apart for anyone reading the file.
        "smiles": Chem.MolToSmiles(mol),
        "smiles_origin": "derived from the supplied coordinates, not a supplied string",
        "coordinate_source": "supplied SDF, used as given (no embedding, no minimisation)",
        "source_file": {"path": sdf_path.name, "sha256": sha256_file(sdf_path)},
        "n_conformers_embedded": 0,
        "mmff_variant": None,
        "selected_conformer": None,
        "all_conformers": [],
        "n_unconverged": 0,
        "residue_name": residue_name or "UNL",
        "solute_sdf": str(out_dir / "solute.sdf"),
        "solute_pdb": str(out_dir / "solute.pdb"),
    }
    (out_dir / "initial_structure.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


# ---------------------------------------------------------------------------------------------
# Force field construction (shared by steps 2, 3, 4)
# ---------------------------------------------------------------------------------------------
#: Charge methods that route to OpenFF NAGL's graph model of AM1-BCC.  ``nagl`` is the older
#: spelling and is kept working; ``am1bcc_nagl`` is preferred because it says which quantity the
#: model reproduces.  Both resolve to the same model -- see :func:`resolve_nagl_am1bcc_model`.
NAGL_AM1BCC_METHODS = ("am1bcc_nagl", "nagl")


def resolve_nagl_am1bcc_model() -> dict:
    """Return the newest *production* NAGL AM1-BCC model, as ``{name, path, sha256}``.

    Resolved at call time rather than pinned in this source file.  An earlier version hard-coded
    ``openff-gnn-am1bcc-0.1.0-rc.3.pt``, which meant that installing a newer `openff-nagl-models`
    kept using a release candidate that the release notes had superseded -- and that nothing in the
    recorded provenance revealed, because only the string ``"nagl"`` was ever written down.

    ``production_only=True`` is the point of the query: the package also ships an alpha and three
    release candidates, and picking the newest file overall would silently prefer a pre-release the
    moment one is published.  The list is ordered oldest to newest, so the last entry is current.

    The returned digest is what makes a NAGL run reproducible.  The method name alone does not
    identify a Hamiltonian -- upgrading the models package would change the charges without
    changing any configuration file -- so callers record the file and its hash alongside the name.
    """
    # The optional dependency, and the package's own hashing implementation. These two imports
    # fail for completely different reasons and must not be reported as the same thing: the first
    # means "NAGL is not installed here", the second would mean "MD-tools is broken". Keeping
    # them apart is why the first is caught and the second is not.
    try:
        from openff.nagl_models import get_models_by_type
    except ImportError as error:                      # the genuine optional-dependency case
        raise RuntimeError(
            "ligand_charge_method='am1bcc_nagl' needs `openff-nagl-models`, which is not "
            f"installed in this environment ({error}). Install it, or use "
            "ligand_charge_method='am1bcc' to charge through AmberTools instead. This is a "
            "deliberate refusal: falling back to a different charge model silently would change "
            "the Hamiltonian without changing any configuration file."
        ) from error

    from .provenance_min import sha256_file

    models = list(get_models_by_type("am1bcc", production_only=True))
    if not models:
        raise RuntimeError(
            "no production NAGL AM1-BCC model is installed, so ligand_charge_method="
            "'am1bcc_nagl' cannot be honoured.  `openff-nagl-models` is present but ships only "
            "pre-release models in this environment.  Install a release that carries "
            "openff-gnn-am1bcc-1.0.0.pt or later, or use ligand_charge_method='am1bcc'."
        )
    newest = Path(str(models[-1]))
    return {"name": newest.name, "path": str(newest), "sha256": sha256_file(newest)}


def build_forcefield(cfg: dict, ligand_sdf: Optional[Path] = None,
                    route: Optional[str] = None):
    """Return ``(ForceField, info)`` for the configured combination (+ Sage for a ligand).

    The default is ff14SB + TIP3P; ``--solvent OPC`` selects ff19SB + OPC. Which one is loaded is
    the user's ``forcefield`` block, not a constant here, and the resources that were actually
    given to ``ForceField()`` are reported back in ``info``.

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
    # `water` is None under implicit solvent -- there is no water to parameterise -- and passing
    # None into ForceField() raised "expected str, bytes or os.PathLike object, not NoneType". The
    # implicit LIGAND route had never been exercised end to end, so this only surfaced the first
    # time a macrocycle was built with GBn2: the implicit peptide route always names a protein XML,
    # which masked it.
    xmls = [x for x in ([protein_xml] if protein_xml else [])
            + [ff_cfg["water"], *ff_cfg["extra_xml"]] if x]
    info: dict[str, Any] = {
        "xml": list(xmls),
        # `amber14-all.xml` is a manifest of includes, not a parameter file. A record that names
        # only the wrapper cannot say which protein XML supplied the dihedrals, so the includes are
        # expanded and recorded beside it.
        "xml_includes": {name: _xml_includes(name) for name in xmls},
        "route": route,
        "protein_forcefield": protein_xml,
        "water": ff_cfg["water"],
        "ligand": None,
    }
    forcefield = app.ForceField(*xmls)

    package_dirs = list(ff_cfg.get("ligand_packages") or [])
    if package_dirs:
        # SAVED PARAMETERS, loaded as they are. No charge is assigned and no template generator
        # is registered: the package was created (or verified) once, before this function was
        # first called, and every step of the build -- hydrogens, solvent, System -- loads the
        # same file. `ligand_sdf` is not read for parameters on this path.
        from ..ligands.build import packages_from_dirs
        from ..ligands.mapping import load_packages_into

        packages = packages_from_dirs(package_dirs)
        compatibility = load_packages_into(forcefield, packages)
        first = packages[0]
        info["ligand"] = {
            "forcefield": first.metadata["forcefield"]["resource"],
            "charge_method": first.metadata["charges"]["method"],
            "charge_scheme": first.metadata["charges"].get("scheme"),
            "charge_source": first.metadata["charges"].get("source"),
            "net_charge_e": float(first.metadata["net_partial_charge_e"]),
            "formal_charge": first.metadata["chemical_state"]["net_formal_charge"],
            "n_atoms": len(first.atom_names),
            "family": first.metadata["forcefield"]["family"],
            "packages": [p.summary() for p in packages],
            "nonbonded_compatibility": compatibility,
        }
        return forcefield, info

    if ligand_sdf is not None:
        from openff.toolkit import Molecule
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        method = str(ff_cfg["ligand_charge_method"]).lower()
        wrappers = [t.__class__.__name__ for t in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits]
        if method == "am1bcc" and "AmberToolsToolkitWrapper" not in wrappers:
            raise RuntimeError(
                "forcefield.ligand_charge_method='am1bcc' needs AmberTools' sqm, but the OpenFF "
                f"toolkit registry only has {wrappers}.  Activate the environment "
                "(micromamba activate openmm-env) so antechamber/sqm are on PATH, or set "
                "forcefield.ligand_charge_method to 'am1bcc_nagl' if that is intended -- NAGL is "
                "a graph network TRAINED to predict AM1-BCC ELF10 charges and needs no sqm, but it "
                "is not that calculation and not numerically identical to it, so it is a different "
                "Hamiltonian rather than a drop-in substitute.  "
                "This is checked here because otherwise the charges would silently fall back to "
                "a different method and the run would be mislabelled."
            )
        if method in NAGL_AM1BCC_METHODS and "NAGLToolkitWrapper" not in wrappers:
            raise RuntimeError(
                f"forcefield.ligand_charge_method={method!r} needs OpenFF NAGL, but the OpenFF "
                f"toolkit registry only has {wrappers}.  Install `openff-nagl` and "
                "`openff-nagl-models` into the active environment.  Checked for the same reason as "
                "the AmberTools case above: without it the charges come from somewhere else and "
                "the run is mislabelled."
            )
        offmol = Molecule.from_file(str(ligand_sdf))
        # What actually identifies the charges, beyond the method name. For AM1-BCC the name plus
        # the conformer scheme is enough; for NAGL the model file is a trained artefact that can be
        # upgraded underneath an unchanged configuration, so it is recorded explicitly.
        charge_provenance: dict[str, Any] = {}
        if method == "am1bcc":
            scheme = "am1bccelf10" if _has_openeye() else "am1bcc"
            offmol.assign_partial_charges(scheme)
            charge_provenance["charge_scheme"] = scheme
        elif method in NAGL_AM1BCC_METHODS:
            model = resolve_nagl_am1bcc_model()
            offmol.assign_partial_charges(model["path"])
            charge_provenance = {"charge_scheme": model["name"],
                                 "nagl_model_file": model["name"],
                                 "nagl_model_sha256": model["sha256"]}
        else:
            supported = ", ".join(repr(m) for m in ("am1bcc", *NAGL_AM1BCC_METHODS))
            raise ValueError(
                f"unsupported ligand_charge_method {method!r}; supported: {supported}")
        # Two families, one already-resolved resource name. `ligand_forcefield` turned whatever
        # the user wrote into an exact version before the box was built, so nothing here has to
        # interpret a label -- it only has to pick the generator that loads this kind of file.
        from .ligand_forcefield import gaff_provenance, is_gaff

        resource = ff_cfg["ligand"]
        family_provenance: dict[str, Any] = {}
        if is_gaff(resource):
            from openmmforcefields.generators import GAFFTemplateGenerator

            # The charges are already on the molecule, assigned above by the requested method.
            # GAFFTemplateGenerator uses them rather than recomputing, so the charge method
            # recorded in the manifest is the one that actually produced these numbers.
            generator = GAFFTemplateGenerator(molecules=[offmol], forcefield=resource)
            family_provenance = gaff_provenance(resource)
        else:
            generator = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield=resource)
            family_provenance = {"family": "smirnoff"}
        forcefield.registerTemplateGenerator(generator.generator)
        info["ligand"] = {
            "forcefield": resource,
            "charge_method": method,
            "net_charge_e": float(sum(c.m for c in offmol.partial_charges)),
            "formal_charge": int(round(sum(a.formal_charge.m for a in offmol.atoms))),
            "n_atoms": offmol.n_atoms,
            **charge_provenance,
            **family_provenance,
        }
    return forcefield, info


def _xml_includes(resource: str) -> list:
    """The `<Include file=...>` entries of an OpenMM force-field resource, in order.

    Empty for a leaf file. Resolved against OpenMM's own data directory, and returns an empty list
    rather than raising if the file cannot be located: this is a provenance detail, and failing to
    expand it must not fail a build that OpenMM itself loaded successfully.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path as _Path

    try:
        from openmm import app as _app

        path = _Path(_app.__file__).resolve().parent / "data" / str(resource)
        if not path.is_file():
            return []
        root = ET.parse(path).getroot()
        return [element.get("file") for element in root.findall("Include")
                if element.get("file")]
    except Exception:
        return []


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

    protonation_record = None
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
        # ONE implementation of protein hydrogen addition, shared with the complex route: delete,
        # predict (protonation.method: propka), assign by the stated rule, add hydrogens seeded
        # and relaxed on the Reference platform, and screen histidines near ions. See
        # `md_tools.openmm.protonation`.
        from .protonation import protonate_structure
        from .seeds import DEFAULT_MASTER_SEED, derive_build_seed as derive_seed

        master = (cfg.get("run") or {}).get("seed")
        hydrogen_seed = derive_seed(
            int(master if master is not None else DEFAULT_MASTER_SEED), "structure/protonation")
        result = protonate_structure(
            modeller.topology, modeller.positions, forcefield, pcfg, seed=hydrogen_seed,
            workdir=out_dir / "protonation", echo=None)
        modeller = app.Modeller(result.topology, result.positions)
        protonation_record = result.record
        added = [a["final_variant"] for a in result.record["assignments"]]
        note = (f"{pcfg.get('method', 'openmm')} protonation at pH {pcfg['ph']}, seed "
                f"{hydrogen_seed}, relaxed on the Reference platform for reproducibility")

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
        # which variant each titratable residue ended with, bound to residue identity in
        # `protonation` below; this flat list is kept for readers that predate it
        "variants": [None if v is None else str(v) for v in added],
        "protonation": protonation_record,
        "forcefield": ff_info,
    }
    (out_dir / "protonation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def protonate_complex(mapped, out_dir: Path, cfg: dict) -> dict:
    """Hydrogens for everything EXCEPT the mapped ligand instances, which are frozen.

    The ligands already carry exactly the hydrogens their packages define, placed by
    `md_tools.ligands.mapping`. Deleting and re-adding them here would replace a chemical state
    that was chosen and parameterised with whatever the template matcher prefers, so only
    residues outside `mapped.frozen_residues` lose and regain hydrogens. The package templates
    are loaded and named per residue, because `addHydrogens` builds a System over the whole
    topology to relax the new hydrogens. Afterwards the instances are checked unchanged.

    This is the complex route's one protonation call site, and the place a PROPKA-derived
    assignment replaces `addHydrogens(pH)`'s own variant choice.
    """
    from openmm import Platform, app
    from openmm.app import element as elem

    from ..ligands.mapping import assert_instances_unchanged

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["protonation"]
    forcefield, ff_info = build_forcefield(cfg, route="complex")
    modeller = app.Modeller(mapped.topology, mapped.positions)
    frozen = mapped.frozen_residues

    def key(residue):
        return (residue.chain.id, str(residue.id).strip(), (residue.insertionCode or "").strip())

    n_h_before = sum(1 for a in modeller.topology.atoms() if a.element == elem.hydrogen)
    from .protonation import protonate_structure
    from .seeds import DEFAULT_MASTER_SEED, derive_build_seed as derive_seed

    master = (cfg.get("run") or {}).get("seed")
    hydrogen_seed = derive_seed(
        int(master if master is not None else DEFAULT_MASTER_SEED), "structure/protonation")
    result = protonate_structure(
        modeller.topology, modeller.positions, forcefield, pcfg, frozen_residues=frozen,
        residue_templates_for=mapped.residue_templates, seed=hydrogen_seed,
        workdir=out_dir / "protonation", echo=None)
    modeller = app.Modeller(result.topology, result.positions)
    assert_instances_unchanged(mapped, modeller.topology, modeller.positions,
                               step="protein hydrogen addition")

    out_pdb = out_dir / "solute_h.pdb"
    with out_pdb.open("w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    residues = list(modeller.topology.residues())
    info = {
        "output_pdb": str(out_pdb),
        "route": "complex",
        "ph": float(pcfg["ph"]),
        "note": (f"{pcfg.get('method', 'openmm')} protonation at pH {pcfg['ph']}, seed "
                 f"{hydrogen_seed}, on every residue except the {len(frozen)} mapped ligand "
                 f"instance(s), whose package hydrogens were kept"),
        "frozen_ligand_instances": [
            {"chain": c, "resid": r, "insertion_code": i} for c, r, i in sorted(frozen)],
        "n_hydrogens_before": n_h_before,
        "n_hydrogens_after": sum(1 for a in modeller.topology.atoms()
                                 if a.element == elem.hydrogen),
        "n_atoms": modeller.topology.getNumAtoms(),
        # Bound to residue identity, not to a position in a list.
        "variants": [{"chain": a["chain"], "resid": a["resid"], "insertion_code": a["insertion_code"],
                      "residue": a["residue"], "variant": a["final_variant"]}
                     for a in result.record["assignments"] if a["final_variant"] is not None],
        "protonation": result.record,
        "forcefield": ff_info,
    }
    (out_dir / "protonation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


PROTEIN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME", "NHE", "NMA",
})

_BENZENE = (("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"),
            ("CD2", "CG"))
_IMIDAZOLE = (("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG"))

#: Which bonds of a standard protein residue carry UNSCALED torsions, by class, by atom name.
#:
#: A topology has no bond orders and OpenMM's `residues.xml` records none, so for a protein this is
#: the evidence: the aromatic rings of PHE, TYR, TRP and HIS (every protonation name), and ARG's
#: guanidinium, whose partial double bonds the user decided count as double bonds (2026-09-16).
#: The backbone amide is not here -- `_amide_candidates` finds it structurally, with the
#: proline-like exception. Every pair is checked against `residues.xml` by a test, so an atom name
#: typed wrong fails the suite instead of exempting nothing.
PROTEIN_UNSCALED_BONDS = {
    "PHE": {"aromatic_ring": _BENZENE},
    "TYR": {"aromatic_ring": _BENZENE},
    "TRP": {"aromatic_ring": (("CG", "CD1"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"),
                              ("CD2", "CG"), ("CD2", "CE3"), ("CE3", "CZ3"), ("CZ3", "CH2"),
                              ("CH2", "CZ2"), ("CZ2", "CE2"))},
    "HIS": {"aromatic_ring": _IMIDAZOLE},
    "HID": {"aromatic_ring": _IMIDAZOLE},
    "HIE": {"aromatic_ring": _IMIDAZOLE},
    "HIP": {"aromatic_ring": _IMIDAZOLE},
    "ARG": {"double_bond": (("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"))},
}

#: The classes of central bond whose torsions stay unscaled, in the order they are decided and
#: reported. Impropers are the fourth class and have no central bond.
UNSCALED_BOND_CLASSES = ("amide_omega", "aromatic_ring", "double_bond")



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
    # Both molecular-graph inputs infer the same route: what makes a solute a ligand here is that
    # it arrives as a graph with no residue evidence, not which file carried it.
    if input_route in ("smiles", "sdf"):
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
        if input_route in ("smiles", "sdf") and declared in ("peptide", "complex"):
            raise ValueError(
                f"system.solute_kind='{declared}' contradicts a {input_route} input.  ff19SB "
                "matches by residue template and a molecule built from a molecular graph is a "
                "single 'UNL' residue, so the peptide route cannot succeed here.  Supply a "
                "residue-named PDB for the ff19SB route, or set system.solute_kind to "
                "'auto'/'ligand'."
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
def verify_hydrogen_mass_repartitioning(
    system, topology, target_h_mass_amu: float, solute_atoms=None,
    mass_before: Optional[float] = None
) -> dict:
    """Check the repartitioning `createSystem(hydrogenMass=...)` performed, without redoing it.

    OpenMM does the same arithmetic this package used to do -- verified atom-by-atom, 3,450 masses
    identical to 0.000e+00 amu on a solvated alanine system -- but it performs none of the checks
    below, and silently produces a nonsense mass rather than refusing:

    * every eligible hydrogen actually reached the target mass;
    * no heavy atom was left at or below 1 amu, which is what happens when the target is set too
      high for a CH3 group and is the failure worth refusing rather than integrating;
    * water was untouched, since rigid water's hydrogen masses do not limit the timestep and
      changing them would alter its rotational dynamics for no benefit;
    * the total mass is conserved, so centre-of-mass dynamics are unchanged.

    Returns the same provenance dictionary the hand-rolled version returned, so callers and the
    recorded manifest do not change.
    """
    from openmm import unit
    from openmm.app import element as elem

    solute = None if solute_atoms is None else {int(i) for i in solute_atoms}
    target = float(target_h_mass_amu)

    repartitioned, problems = 0, []
    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.element == elem.hydrogen and b.element != elem.hydrogen:
            h, heavy = a, b
        elif b.element == elem.hydrogen and a.element != elem.hydrogen:
            h, heavy = b, a
        else:
            continue
        m_h = system.getParticleMass(h.index).value_in_unit(unit.amu)
        m_heavy = system.getParticleMass(heavy.index).value_in_unit(unit.amu)

        in_water = h.residue.name.upper() in WATER_RESIDUE_NAMES
        in_scope = solute is None or (h.index in solute and heavy.index in solute)

        if in_water:
            # Compared against the TARGET, not against the element mass. A force field assigns its
            # own hydrogen mass -- amber19 water uses 1.008 amu where OpenMM's element constant is
            # 1.007947 -- so testing against the element would fail on a correct system. What
            # matters is that water was left alone, which means its hydrogens are nowhere near the
            # repartitioned value.
            if abs(m_h - target) < 0.5:
                problems.append(
                    f"water hydrogen {h.index} has mass {m_h:.4f} amu, close to the {target:.4f} "
                    "target; rigid water must not be repartitioned")
            continue
        if not in_scope:
            continue
        if m_h <= 0.0:                       # a virtual site, never repartitioned
            continue
        if abs(m_h - target) > 1e-6:
            problems.append(
                f"hydrogen {h.index} has mass {m_h:.4f} amu, expected {target:.4f}")
            continue
        repartitioned += 1
        if m_heavy <= 1.0:
            problems.append(
                f"heavy atom {heavy.index} was left with {m_heavy:.4f} amu after donating to "
                f"hydrogen {h.index}. Lower system_build.hydrogen_mass_amu.")

    total_after = sum(
        system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())
    )
    if mass_before is not None and not math.isclose(
            mass_before, total_after, rel_tol=0.0, abs_tol=1e-3):
        problems.append(
            f"hydrogen mass repartitioning changed the total mass: {mass_before:.6f} -> "
            f"{total_after:.6f} amu")
    # `mass_before` is only meaningful when the caller measured the SYSTEM's total before
    # repartitioning. It cannot be reconstructed from the topology afterwards, because a force
    # field assigns its own masses that differ from the element constants -- amber19 gives alanine
    # in water 11160.736 amu against 11160.329 summed from elements. Conservation is structural in
    # OpenMM's implementation, which subtracts from the heavy atom exactly what it adds to the
    # hydrogen, and was verified atom-by-atom against this package's previous implementation:
    # 3,450 masses identical to 0.000e+00 amu.

    if problems:
        raise ValueError(
            "hydrogen mass repartitioning by OpenMM did not satisfy this package's contract:\n  "
            + "\n  ".join(problems))

    # The same keys the hand-rolled version returned, so the recorded manifest schema is unchanged
    # by the delegation, plus two that say who did the work.
    return {
        "target_hydrogen_mass_amu": target,
        "n_hydrogens_repartitioned": repartitioned,
        "total_mass_amu": round(total_after, 6),
        "scope": "solute" if solute is not None else "all-non-water",
        "performed_by": "openmm.app.ForceField.createSystem(hydrogenMass=...)",
        "verified_by": "md_tools.openmm.system.verify_hydrogen_mass_repartitioning",
    }


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
            # The residue INDEX as well as the name: a chain with eleven alanines gives a reader
            # nothing to act on if the evidence says only "ALA".
            "carbon_residue_index": int(c.residue.index),
            "nitrogen_residue_index": int(n.residue.index),
            "inter_residue": c.residue.index != n.residue.index,
            "ambiguous": reason,
        })
    return out


def classify_unscaled_torsions(topology, solute_atoms: Iterable[int], *,
                               ligand_sdf: Optional[Path] = None,
                               residue_sdfs: Optional[dict] = None,
                               proline_like_residues: Iterable[str] = ("PRO",),
                               max_proline_ring_size: int = 7,
                               unscaled_impropers: bool = True) -> dict:
    """Which of the solute's torsions REST2 leaves unscaled, with the evidence for each.

    Four classes (the user's decision of 2026-09-16; docs/amber-like-fix/REST2-scaler.md §10):

    * **ordinary amide omega** -- decided per amide candidate, below, with the proline-like
      exception;
    * **aromatic ring bonds** and **other double bonds** -- a protein residue from
      `PROTEIN_UNSCALED_BONDS`, anything else from the SDF's bond orders. Only a bond whose two atoms
      both have another neighbour is listed: a terminal C=O has no torsion across it;
    * **impropers** -- no central bond, so not listed here; `unscaled_impropers` is carried through
      to the scaler, which finds them from the System's bond graph.

    A non-standard residue with a possible central bond and NO bond-order evidence is unclassified
    as a whole: its ring and double bonds cannot be named from a topology, amide or not.

    THE AMIDE RULE, unchanged from the omega classifier this replaces:

    REST2 here is *omega-selective*: an ORDINARY amide omega torsion is left unscaled, because the
    REMD ladder leaves it unscaled and a hot rung that isomerises cis/trans samples states the
    reference never does.  A **proline-like** peptide bond is the exception and stays ELIGIBLE for
    normal scaling: its nitrogen is locked into a small ring, so the torsion is not the near-planar
    two-state coordinate the exclusion exists to protect.

    WHICH EVIDENCE IS USED IS DECIDED PER CANDIDATE, from the residue holding the amide nitrogen.
    There is no ``route`` argument: there was, and four of its six call sites passed
    ``route="peptide"`` unconditionally while two threaded the real one, so the rung WRITER and the
    rung VALIDATOR disagreed about the same ladder and the refusing half ran.  A REST2 ladder over
    paracetamol -- one ``UNL`` residue, its SDF one directory away -- was refused on CUDA because
    the peptide route could not name a residue it was never meant to read.

    1. **A declared proline-like name wins outright**, before any file is opened.  That is what
       ``rest2.proline_like_residues`` is for: the human answer to a block.
    2. **A known protein residue is read from the residue**, against ``PROTEIN_RESIDUES``.  An
       X-PRO peptide bond is therefore *not* excluded.
    3. **Anything else is read from the SDF's bond orders.**  Ordinary amides are matched with
       ``[CX3](=[OX1])[NX3]``; proline-like nitrogens with a ring of at most
       *max_proline_ring_size* atoms.  **The ring-size bound is what makes this correct for
       macrocycles**: every backbone nitrogen of a cyclic peptide is "in a ring", but a 15-30
       membered macrocycle does not constrain the amide the way a pyrrolidine does, so an unbounded
       ``;R`` test would wrongly free every macrocyclic omega for scaling.
    4. **With no SDF, it refuses.**  A protein carrying a modified residue has no SDF at all, and
       "it looks like an amide" is precisely the guess that would silently change its Hamiltonian.

    The rule is per candidate rather than per run because a mixed protein+ligand solute has both
    kinds at once: the backbone omegas have residue evidence and the ligand's has none, and no
    single answer is right for both.  The SDF is mapped onto the NON-STANDARD residues only, which
    is what it actually describes.

    WHERE THE BOND ORDERS COME FROM -- two forms, never both:

    * ``ligand_sdf``: ONE SDF describing every non-standard solute residue together.  What
      `build-top` writes for a `.smi`/`.sdf` input, where the molecule is one residue.
    * ``residue_sdfs``: ``{residue name: SDF}``, each SDF mapped onto EACH INSTANCE of its residue
      separately.  What a solute with several different non-standard residues needs, since no one
      SDF describes them all.  A non-standard residue absent from the map has no evidence and its
      candidates are unclassified, by name.

    N-methylated amides are ordinary amides under both routes: an N-methyl nitrogen is neither
    proline-like nor ring-locked, and those bonds isomerise readily, so they must stay unscaled.

    Returns ``unscaled_central_bonds`` (every class), ``central_bonds`` (each with its class,
    residue and evidence), ``proline_like_scaled_bonds``, ``unclassified``, ``unscaled_impropers``,
    ``detection_method``, ``detector_version`` and ``amide_detail``. **A non-empty unclassified list
    must block production**; `unscaled_torsions` is the entry point that enforces it.
    """
    if ligand_sdf is not None and residue_sdfs is not None:
        raise ValueError(
            "pass ligand_sdf (one SDF for every non-standard residue) OR residue_sdfs (one SDF per "
            "residue name), not both: two sources of bond orders for one residue would have to "
            "agree, and nothing here would check that they do")
    solute = {int(i) for i in solute_atoms}
    candidates = _amide_candidates(topology, solute)
    pro_names = {str(x).upper() for x in proline_like_residues}
    per_name = ({str(k).upper(): Path(v) for k, v in residue_sdfs.items()}
                if residue_sdfs is not None else None)

    # WHAT AN SDF WOULD HAVE TO DESCRIBE: every solute atom whose residue name is neither a known
    # protein residue nor a declared proline-like one.  Taken from the TOPOLOGY rather than from
    # the candidates, so a non-standard residue that contributes no amide still counts as part of
    # what the SDF must match -- otherwise a two-residue ligand whose amide sits in one half would
    # be mapped against a fraction of itself and fail the bond-graph check for the wrong reason.
    non_standard: set[int] = set()
    non_standard_names: set[str] = set()
    for residue in topology.residues():
        if residue.name.upper() in PROTEIN_RESIDUES or residue.name.upper() in pro_names:
            continue
        indices = {a.index for a in residue.atoms() if a.index in solute}
        if indices:
            non_standard |= indices
            non_standard_names.add(residue.name)

    # Mapped ONCE, and only when something actually needs it.  A pure peptide never opens a file.
    ring_info, mapping_error = None, None
    if non_standard and ligand_sdf is not None and per_name is None:
        try:
            ring_info = _sdf_bond_evidence(ligand_sdf, topology, non_standard,
                                           max_proline_ring_size,
                                           describes=sorted(non_standard_names))
        except ValueError as refusal:
            mapping_error = str(refusal)

    method = ("unscaled torsions: amide omega evidence chosen per candidate: residue-aware "
              "against PROTEIN_RESIDUES, "
              f"proline-like names {sorted(pro_names)} applied to the residue containing the "
              "amide NITROGEN")
    if non_standard:
        if per_name is not None:
            source = "per-residue SDFs " + (", ".join(
                f"{name}={per_name[name].name}" for name in sorted(per_name)) or "(none)")
        else:
            source = (f"SDF {Path(ligand_sdf).name}" if ligand_sdf is not None
                      else "no SDF supplied (refused)")
        method += (f"; residues {sorted(non_standard_names)} from RDKit SMARTS "
                   f"[CX3](=[OX1])[NX3] over {source}, proline-like = amide N in a ring of "
                   f"<= {max_proline_ring_size} atoms")

    # Per residue INSTANCE, when the evidence is per name: mapped lazily, once each.
    instance_info: dict[int, tuple] = {}

    def _instance(residue_index: int, name: str):
        if residue_index not in instance_info:
            residue = next(r for r in topology.residues() if r.index == residue_index)
            atoms = {a.index for a in residue.atoms() if a.index in solute}
            try:
                instance_info[residue_index] = (_sdf_bond_evidence(
                    per_name[name], topology, atoms, max_proline_ring_size,
                    describes=[f"{residue.name}{residue_index}"]), None)
            except ValueError as refusal:
                instance_info[residue_index] = (None, str(refusal))
        return instance_info[residue_index]

    unscaled, proline, unknown = [], [], []
    for cand in candidates:
        if cand["ambiguous"]:
            unknown.append(cand); continue
        residue_name = cand["nitrogen_residue"].upper()

        # 1. The human answer to a previous block wins outright, before any file is opened.
        if residue_name in pro_names:
            proline.append(cand); continue
        # 2. A known protein residue is decided from the residue, as it always was.
        if residue_name in PROTEIN_RESIDUES:
            unscaled.append(cand); continue

        # 3. Otherwise the residue cannot answer, so the molecule's bond orders must.  An
        #    unrecognised residue is still never ASSUMED to be an ordinary amide: with no SDF there
        #    is no second opinion to take, and guessing would silently change the Hamiltonian of
        #    any protein carrying a modified residue.
        if per_name is not None:
            if residue_name not in per_name:
                unknown.append(dict(cand, ambiguous=(
                    f"nitrogen residue '{cand['nitrogen_residue']}' is not a known protein "
                    f"residue, so this omega has to be read from bond orders -- and no SDF was "
                    f"given for residue {cand['nitrogen_residue']}. Residues with an SDF: "
                    f"{sorted(per_name) or 'none'}.")))
                continue
            info, error = _instance(cand["nitrogen_residue_index"], residue_name)
            if info is None:
                unknown.append(dict(cand, ambiguous=(
                    f"nitrogen residue '{cand['nitrogen_residue']}' needs bond orders, and "
                    f"{per_name[residue_name].name} could not be mapped onto it: {error}")))
                continue
        elif ligand_sdf is None:
            unknown.append(dict(cand, ambiguous=(
                f"nitrogen residue '{cand['nitrogen_residue']}' is not a known protein residue, "
                f"so this omega has to be read from the molecule's bond orders -- and no SDF was "
                f"supplied. `build-top` retains one beside the System (`<RESNAME>.sdf`, named for "
                f"the solute residue; `built.sdf` before 0.5.4) for a .smi or .sdf input and writes none for a peptide; supply that SDF beside the System.")))
            continue
        else:
            info = ring_info
            if info is None:
                unknown.append(dict(cand, ambiguous=(
                    f"nitrogen residue '{cand['nitrogen_residue']}' needs bond orders, and the "
                    f"SDF could not be mapped onto residues {sorted(non_standard_names)}: "
                    f"{mapping_error}")))
                continue
        if cand["bond"] not in info["amide_bonds"]:
            unknown.append(dict(cand, ambiguous=(
                "RDKit found no ordinary-amide match for this C-N bond in the SDF")))
            continue
        if cand["nitrogen"] in info["small_ring_nitrogens"]:
            proline.append(dict(cand,
                                ring_sizes=info["ring_sizes"].get(cand["nitrogen"], [])))
        else:
            unscaled.append(cand)

    # --- aromatic ring bonds and other double bonds ------------------------------------------
    neighbours: dict[int, set] = {}
    for bond in topology.bonds():
        neighbours.setdefault(bond.atom1.index, set()).add(bond.atom2.index)
        neighbours.setdefault(bond.atom2.index, set()).add(bond.atom1.index)

    def _central(a: int, b: int) -> bool:
        """A torsion can run across a-b only if both ends have another neighbour."""
        return len(neighbours.get(a, ())) >= 2 and len(neighbours.get(b, ())) >= 2

    amide_bonds = {tuple(sorted(c["bond"])) for c in unscaled + proline}
    central: list[dict] = [{"bond": sorted(c["bond"]), "class": "amide_omega",
                            "residue": c["nitrogen_residue"],
                            "residue_index": c["nitrogen_residue_index"],
                            "evidence": ("residue name" if c["nitrogen_residue"].upper()
                                         in PROTEIN_RESIDUES else "SDF bond orders")}
                           for c in unscaled]
    seen = {tuple(entry["bond"]) for entry in central} | amide_bonds

    def _add(a: int, b: int, kind: str, residue, evidence: str) -> None:
        pair = tuple(sorted((int(a), int(b))))
        if pair in seen or not _central(*pair):
            return
        seen.add(pair)
        central.append({"bond": list(pair), "class": kind, "residue": residue.name,
                        "residue_index": int(residue.index), "evidence": evidence})

    for residue in topology.residues():
        in_solute = [a for a in residue.atoms() if a.index in solute]
        if not in_solute:
            continue
        # By NAME only for the protein table, where names are unique within a residue. A small
        # molecule's atoms are often named after their element, so keying them by name would
        # collapse every carbon into one.
        atoms = {a.name: a.index for a in in_solute}
        name = residue.name.upper()
        if name in PROTEIN_UNSCALED_BONDS:
            for kind, pairs in PROTEIN_UNSCALED_BONDS[name].items():
                for x, y in pairs:
                    if x in atoms and y in atoms:
                        _add(atoms[x], atoms[y], kind, residue, "PROTEIN_UNSCALED_BONDS")
            continue
        if name in PROTEIN_RESIDUES or name in pro_names:
            continue
        members = {a.index for a in in_solute}
        possible = any(other in members and _central(index, other)
                       for index in members for other in neighbours.get(index, ()))
        if not possible:
            continue
        if per_name is not None:
            if name not in per_name:
                info, error = None, (f"no SDF was given for it (residues with an SDF: "
                                     f"{sorted(per_name) or 'none'})")
            else:
                info, error = _instance(residue.index, name)
        elif ligand_sdf is None:
            info, error = None, "no SDF was supplied"
        else:
            info, error = ring_info, mapping_error
        if info is None:
            unknown.append({
                "bond": None, "carbon": None, "nitrogen": None,
                "carbon_residue": residue.name, "nitrogen_residue": residue.name,
                "carbon_residue_index": int(residue.index),
                "nitrogen_residue_index": int(residue.index),
                "residue": residue.name, "residue_index": int(residue.index),
                "ambiguous": (
                    f"residue '{residue.name}' (index {residue.index}) is not a known protein "
                    f"residue, so which of its bonds are aromatic or double -- and so which "
                    f"torsions stay unscaled -- can only be read from bond orders, and {error}. "
                    f"`build-top` writes `built.sdf` beside the System for a .smi or .sdf input; "
                    f"supply the SDF for this residue.")})
            continue
        evidence = "SDF bond orders"
        for a, b in sorted(info["aromatic_bonds"]):
            if a in members and b in members:
                _add(a, b, "aromatic_ring", residue, evidence)
        for a, b in sorted(info["double_bonds"]):
            if a in members and b in members:
                _add(a, b, "double_bond", residue, evidence)

    central.sort(key=lambda e: (UNSCALED_BOND_CLASSES.index(e["class"]), e["bond"]))
    method += ("; aromatic ring and double bonds from PROTEIN_UNSCALED_BONDS for protein residues "
               "and from SDF bond orders otherwise; impropers "
               + ("unscaled" if unscaled_impropers else "scaled"))
    return {
        "unscaled_central_bonds": sorted(tuple(e["bond"]) for e in central),
        "central_bonds": central,
        "proline_like_scaled_bonds": [c["bond"] for c in proline],
        "unclassified": unknown,
        "unscaled_impropers": bool(unscaled_impropers),
        "detection_method": method,
        "detector_version": 2,
        "amide_detail": {"unscaled": unscaled, "proline_like_scaled": proline},
    }


def residues_needing_bond_orders(topology, solute_atoms: Iterable[int],
                                 proline_like_residues: Iterable[str] = ("PRO",)) -> set[str]:
    """Upper-cased names of non-standard solute residues whose unscaled torsions need bond orders.

    A residue needs them when it is neither a known protein residue nor a declared proline-like
    name, and holds at least one bond across which a torsion can run (both atoms have another
    neighbour): its aromatic ring and double bonds cannot be named from a topology. The same test
    `classify_unscaled_torsions` applies, so a caller looking for SDFs asks for exactly the ones the
    classifier will need.
    """
    solute = {int(i) for i in solute_atoms}
    pro = {str(x).upper() for x in proline_like_residues}
    neighbours: dict[int, set] = {}
    for bond in topology.bonds():
        neighbours.setdefault(bond.atom1.index, set()).add(bond.atom2.index)
        neighbours.setdefault(bond.atom2.index, set()).add(bond.atom1.index)
    names = set()
    for residue in topology.residues():
        name = residue.name.upper()
        if name in PROTEIN_RESIDUES or name in pro:
            continue
        members = {a.index for a in residue.atoms() if a.index in solute}
        if any(other in members and len(neighbours.get(index, ())) >= 2
               and len(neighbours.get(other, ())) >= 2
               for index in members for other in neighbours.get(index, ())):
            names.add(name)
    return names


class UnclassifiedTorsionError(ValueError):
    """Something whose torsions may or may not stay unscaled could not be classified."""


def unscaled_torsions(topology, solute_atoms: Iterable[int], *,
                      ligand_sdf: Optional[Path] = None,
                      residue_sdfs: Optional[dict] = None,
                      proline_like_residues: Iterable[str] = ("PRO",),
                      max_proline_ring_size: int = 7,
                      unscaled_impropers: bool = True,
                      enforce: bool = True) -> dict:
    """:func:`classify_unscaled_torsions`, ENFORCED: the one entry point for anything that scales.

    The classifier reports what it could not decide and leaves acting on it to the caller. In 0.5.3
    six callers took the unscaled bonds and only one read the unclassified list, so an amide nobody
    could name was scaled like any other solute torsion, with nothing saying so. A build record may
    still call the classifier directly, because recording is not scaling; every scaling surface
    calls this. ``enforce=False`` returns the classification without raising, for a caller that
    reports the refusal itself.
    """
    result = classify_unscaled_torsions(
        topology, solute_atoms, ligand_sdf=ligand_sdf, residue_sdfs=residue_sdfs,
        proline_like_residues=proline_like_residues, max_proline_ring_size=max_proline_ring_size,
        unscaled_impropers=unscaled_impropers)
    unknown = result["unclassified"]
    if not unknown or not enforce:
        return result
    shown = unknown[:5]
    lines = []
    for c in shown:
        if c.get("bond") is None:
            lines.append(f"  residue {c['residue']}{c['residue_index']}: {c['ambiguous']}")
        else:
            lines.append(f"  bond {c['bond'][0]}-{c['bond'][1]}: "
                         f"{c.get('carbon_residue')}{c.get('carbon_residue_index')} C -> "
                         f"{c.get('nitrogen_residue')}{c.get('nitrogen_residue_index')} N: "
                         f"{c['ambiguous']}")
    if len(unknown) > len(shown):
        lines.append(f"  ... and {len(unknown) - len(shown)} more")
    raise UnclassifiedTorsionError(
        f"{len(unknown)} item(s) could not be classified, so which torsions stay unscaled is "
        f"undecided. Scaling a torsion that should stay unscaled lets a hot state leave a planar "
        f"geometry the physical state never leaves; exempting one that should not changes the "
        f"Hamiltonian the other way. Neither is guessed, so nothing is scaled:\n" + "\n".join(lines))


def _sdf_bond_evidence(ligand_sdf, topology, atoms_to_map: set[int], max_ring: int,
                       *, describes: Optional[list] = None) -> dict:
    """RDKit perception on an SDF, mapped onto OpenMM indices: amides, ring nitrogens, aromatic and
    double bonds.

    *atoms_to_map* is the set of topology indices the SDF is expected to describe -- the solute's
    NON-STANDARD residues, not the whole solute.  On a mixed protein+ligand system the SDF covers
    the ligand alone, so mapping it against every solute atom would fail on atom count and say
    nothing useful about why.

    The mapping is POSITIONAL and ASSERTED, never assumed: the SDF's atoms are paired with those
    indices in ascending order, and element sequence and the full bond graph must then agree.  The
    pairing is positional rather than an offset because the atoms need not be contiguous -- a
    ligand that follows a protein in the topology starts partway through, and an offset would
    silently shift every index.  They coincide today because the SDF and the PDB are written from
    the same RDKit molecule in the same order, but that is a property of the pipeline rather than
    a guarantee, and a silent off-by-one here would scale the wrong torsions.
    """
    from rdkit import Chem

    what = f" (expected to describe {describes})" if describes else ""
    if ligand_sdf is None:
        raise ValueError(
            "bond orders are not recoverable from a topology, so this needs the SDF `build-top` "
            "retains beside the System, but none was supplied"
        )
    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {ligand_sdf}")

    atoms = sorted((a for a in topology.atoms() if a.index in atoms_to_map),
                   key=lambda a: a.index)
    if mol.GetNumAtoms() != len(atoms):
        raise ValueError(
            f"atom-count mismatch: SDF {Path(ligand_sdf).name} has {mol.GetNumAtoms()} atoms, the "
            f"topology's non-standard residues{what} have {len(atoms)}.  The RDKit->OpenMM "
            f"mapping cannot be established."
        )
    #: SDF atom i is topology atom `index_of[i]`.  Positional, so a non-contiguous block maps.
    index_of = [a.index for a in atoms]
    for i, atom in enumerate(atoms):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if atom.element is None or atom.element.symbol != sym:
            raise ValueError(
                f"element mismatch at position {i} (topology index {atom.index}): SDF says {sym}, "
                f"topology says {None if atom.element is None else atom.element.symbol}.  "
                f"Refusing to guess a mapping between the SDF and the topology."
            )
    rd_bonds = {frozenset((index_of[b.GetBeginAtomIdx()], index_of[b.GetEndAtomIdx()]))
                for b in mol.GetBonds()}
    top_bonds = {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()
                 if b.atom1.index in atoms_to_map and b.atom2.index in atoms_to_map}
    if rd_bonds != top_bonds:
        raise ValueError(
            f"bond-graph mismatch between {Path(ligand_sdf).name} and the topology"
            f"{what} ({len(rd_bonds ^ top_bonds)} differing bonds).  Refusing to guess a mapping."
        )

    amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
    aromatic_bonds, double_bonds = set(), set()
    for bond in mol.GetBonds():
        pair = tuple(sorted((index_of[bond.GetBeginAtomIdx()], index_of[bond.GetEndAtomIdx()])))
        if bond.GetIsAromatic():
            aromatic_bonds.add(pair)
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            double_bonds.add(pair)
    amide_bonds, small_ring_n, ring_sizes = set(), set(), {}
    ri = mol.GetRingInfo()
    for c_i, _o_i, n_i in mol.GetSubstructMatches(amide):
        amide_bonds.add((index_of[c_i], index_of[n_i]))
        sizes = sorted(len(r) for r in ri.AtomRings() if n_i in r)
        if sizes:
            ring_sizes[index_of[n_i]] = sizes
            if min(sizes) <= max_ring:
                small_ring_n.add(index_of[n_i])
    return {"amide_bonds": amide_bonds, "small_ring_nitrogens": small_ring_n,
            "ring_sizes": ring_sizes, "aromatic_bonds": aromatic_bonds,
            "double_bonds": double_bonds}


def omega_central_bonds(topology, solute_atoms: Iterable[int]) -> list[tuple[int, int]]:
    """DEPRECATED structural detector: every amide C-N bond, with no proline-like exception.

    Kept so pre-2026-08-14 bundles can be re-derived.  It treats an X-PRO peptide bond as an
    ordinary omega and excludes it from scaling, which
    :func:`classify_unscaled_torsions` deliberately does not.  New code must use that function.
    """
    solute = {int(i) for i in solute_atoms}
    return sorted({c["bond"] for c in _amide_candidates(topology, solute) if not c["ambiguous"]})


def nonbonded_method_name(code) -> str:
    """`NonbondedForce.PME` -> `"PME"`, read off the built Force rather than off the request.

    The configuration says what was asked for; this says what the Force reports. They differ if a
    builder ever substitutes a method, and it is the second one that describes the Hamiltonian.
    """
    from openmm import NonbondedForce

    names = {NonbondedForce.NoCutoff: "NoCutoff",
             NonbondedForce.CutoffNonPeriodic: "CutoffNonPeriodic",
             NonbondedForce.CutoffPeriodic: "CutoffPeriodic",
             NonbondedForce.Ewald: "Ewald",
             NonbondedForce.PME: "PME",
             NonbondedForce.LJPME: "LJPME"}
    return names.get(int(code), f"unknown({int(code)})")


def verify_hmr_group_masses(system, reference_system, topology, *, target_h_mass_amu,
                            solute_atoms=None) -> dict:
    """Check HMR against the SAME System built without it, group by group.

    `verify_hydrogen_mass_repartitioning` checks that each hydrogen reached the target and that no
    heavy atom was driven below 1 amu. What it cannot check on its own is conservation, because a
    force field assigns its own masses and the pre-repartitioning total cannot be reconstructed
    from the topology afterwards -- amber19 gives alanine in water 11160.736 amu against 11160.329
    summed from element constants.

    So the reference is built: the identical `createSystem` call with `hydrogenMass` omitted. Then
    conservation is checked where repartitioning actually happens -- within each heavy atom and the
    hydrogens bonded to it -- rather than only on the box total, where two errors of opposite sign
    would cancel. A nonpositive heavy-atom mass is refused here as well as in the per-hydrogen
    check, because a mass of exactly zero is a fixed particle in OpenMM rather than an error.
    """
    from openmm import unit
    from openmm.app import element as elem

    solute = None if solute_atoms is None else {int(i) for i in solute_atoms}
    groups: dict[int, list[int]] = {}
    for bond in topology.bonds():
        a, b = bond.atom1, bond.atom2
        if a.element == elem.hydrogen and b.element != elem.hydrogen:
            h, heavy = a, b
        elif b.element == elem.hydrogen and a.element != elem.hydrogen:
            h, heavy = b, a
        else:
            continue
        if solute is not None and (h.index not in solute or heavy.index not in solute):
            continue
        groups.setdefault(heavy.index, []).append(h.index)

    def mass(sys_, index):
        return sys_.getParticleMass(int(index)).value_in_unit(unit.amu)

    problems, worst, checked = [], 0.0, 0
    lightest_heavy = None
    for heavy, hydrogens in groups.items():
        before = mass(reference_system, heavy) + sum(mass(reference_system, h) for h in hydrogens)
        after = mass(system, heavy) + sum(mass(system, h) for h in hydrogens)
        worst = max(worst, abs(after - before))
        checked += 1
        if abs(after - before) > 1e-6:
            problems.append(
                f"heavy atom {heavy} and its {len(hydrogens)} hydrogen(s) weigh {after:.6f} amu "
                f"after repartitioning and {before:.6f} amu before it")
        m_heavy = mass(system, heavy)
        lightest_heavy = m_heavy if lightest_heavy is None else min(lightest_heavy, m_heavy)
        if m_heavy <= 0.0:
            problems.append(
                f"heavy atom {heavy} was left with {m_heavy:.6f} amu, which OpenMM reads as a "
                f"fixed particle rather than an atom. Lower the target hydrogen mass "
                f"({target_h_mass_amu}).")

    total_before = sum(mass(reference_system, i)
                       for i in range(reference_system.getNumParticles()))
    total_after = sum(mass(system, i) for i in range(system.getNumParticles()))
    if abs(total_after - total_before) > 1e-3:
        problems.append(
            f"repartitioning changed the System total mass: {total_before:.6f} -> "
            f"{total_after:.6f} amu")
    if problems:
        raise ValueError("hydrogen mass repartitioning did not conserve mass:\n  "
                         + "\n  ".join(problems))
    return {
        "groups_checked": checked,
        "max_group_mass_change_amu": round(worst, 9),
        "lightest_heavy_atom_amu": (round(lightest_heavy, 6)
                                    if lightest_heavy is not None else None),
        "total_mass_amu_before": round(total_before, 6),
        "total_mass_amu_after": round(total_after, 6),
        "verified_against": "the same createSystem call without hydrogenMass",
    }


def constraint_option(name):
    """`constraints.type` as OpenMM spells it. THE mapping, for every route.

    It lived inline in the explicit builder while the implicit route passed `app.HBonds`
    unconditionally, so `AllBonds` under GBn2 built a System identical to `HBonds` while the log
    recorded `AllBonds (set)` -- a setting that was accepted, reported, and never applied.

    `HAngles` is refused rather than mapped. `build-top`'s schema does not offer it and
    `docs/scientific-defaults.md` says why: no angle is ever constrained by this option. Accepting
    it here while the enum refused it meant the two disagreed about the policy.
    """
    from openmm import app

    if name is None:
        return None
    text = str(name)
    try:
        return {"HBonds": app.HBonds, "AllBonds": app.AllBonds, "None": None}[text]
    except KeyError:
        raise ValueError(
            f"constraints.type {text!r} is not supported. This package offers HBonds, AllBonds "
            f"and None. OpenMM's HAngles would constrain angles as well and is deliberately not "
            f"offered -- see docs/scientific-defaults.md, 'No angle is ever constrained by this "
            f"option'.") from None


def build_system(solvated_pdb: Path, out_dir: Path, cfg: dict, n_solute_atoms: int,
                 ligand_sdf: Optional[Path] = None, route: str = "peptide", *,
                 residue_templates_for=None, residue_sdfs: Optional[dict] = None) -> dict:
    """Create the OpenMM ``System`` (PME, 1.0 nm, HBonds, HMR) and serialise it to XML.

    ``residue_templates_for(topology)`` names the template for each mapped ligand residue, so
    template matching never chooses between packages by graph alone; ``residue_sdfs`` gives the
    unscaled-torsion classifier each ligand species' bond orders on the complex route.
    """
    from openmm import XmlSerializer, app, unit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcfg = cfg["system_build"]

    pdb = app.PDBFile(str(solvated_pdb))
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf, route=route)
    templates = residue_templates_for(pdb.topology) if residue_templates_for else {}

    method = {"PME": app.PME, "LJPME": app.LJPME, "CutoffPeriodic": app.CutoffPeriodic,
              "NoCutoff": app.NoCutoff}[bcfg["nonbonded_method"]]
    # A vacuum System has no cutoff and no Ewald sum; passing either would be a setting that
    # does nothing, so they are passed only to a method that uses them.
    periodic_kwargs = ({} if bcfg["nonbonded_method"] == "NoCutoff" else {
        "nonbondedCutoff": float(bcfg["nonbonded_cutoff_nm"]) * unit.nanometer,
        "ewaldErrorTolerance": float(bcfg["ewald_error_tolerance"])})
    constraints = constraint_option(bcfg["constraints"])

    # OpenMM repartitions hydrogen mass itself, skipping any residue it made rigid -- so with
    # rigidWater=True its behaviour is exactly the "solute" scope this package wants, which was
    # verified atom-by-atom against the previous hand-rolled version on a solvated system: 3,450
    # masses identical to 0.000e+00 amu. Delegating removes the duplicate arithmetic; the checks
    # OpenMM does NOT perform are applied afterwards by `verify_hydrogen_mass_repartitioning`.
    # `hmr_scope` and `hydrogen_mass_amu` must agree: a scope with no mass has nothing to apply,
    # and a mass with scope "none" is a value that silently does nothing. Both are configuration
    # mistakes worth naming rather than resolving by guessing.
    hmr_scope_name = str(bcfg["hmr_scope"] or "none")
    raw_mass = bcfg["hydrogen_mass_amu"]
    if hmr_scope_name in ("solute", "all") and raw_mass is None:
        raise ValueError(
            f"system_build.hmr_scope={hmr_scope_name!r} asks for hydrogen mass repartitioning but "
            "system_build.hydrogen_mass_amu is null, so there is no mass to repartition to. "
            "Set hydrogen_mass_amu (the repository's performance profiles use 3.024) or set "
            "hmr_scope to 'none'.")
    if hmr_scope_name == "none" and raw_mass is not None:
        raise ValueError(
            f"system_build.hydrogen_mass_amu={raw_mass!r} is set but system_build.hmr_scope is "
            "'none', so it would be silently ignored and the run would integrate with unmodified "
            "hydrogen masses while the manifest recorded a repartitioned one. Set hmr_scope to "
            "'solute' to apply it, or clear hydrogen_mass_amu.")
    delegate_hmr = bool(bcfg["rigid_water"]) and hmr_scope_name in ("solute", "all")
    # Only meaningful when HMR is actually requested; computing it unconditionally raised
    # TypeError on the conservative default, where the mass is deliberately null.
    target_h_mass = float(raw_mass) if raw_mass is not None else None
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        constraints=constraints,
        rigidWater=bool(bcfg["rigid_water"]),
        removeCMMotion=bool(bcfg["remove_cm_motion"]),
        residueTemplates=templates,
        **periodic_kwargs,
        **({"hydrogenMass": target_h_mass * unit.amu} if delegate_hmr else {}),
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
            switching = bool(force.getUseSwitchingFunction())
            nb_info = {
                # The NAME, read back off the built Force. `getNonbondedMethod()` returns an int,
                # and an integer in a provenance record is a number a reader has to look up in the
                # OpenMM headers of whichever version happened to write it.
                "method": nonbonded_method_name(force.getNonbondedMethod()),
                "method_code": int(force.getNonbondedMethod()),
                # null for NoCutoff: OpenMM keeps a cutoff and an Ewald tolerance on the Force
                # whether or not the method uses them, and recording them would describe a
                # truncation and a sum that are not applied.
                "cutoff_nm": (None if bcfg["nonbonded_method"] == "NoCutoff" else
                              force.getCutoffDistance().value_in_unit(unit.nanometer)),
                "switching": switching,
                # null, not 0.0, when the switching function is off: OpenMM keeps a switching
                # distance on the Force whether or not it is used, and reporting it unconditionally
                # describes a taper that is not applied.
                "switch_distance_nm": (
                    force.getSwitchingDistance().value_in_unit(unit.nanometer)
                    if switching else None),
                "dispersion_correction": bool(force.getUseDispersionCorrection()),
                "ewald_error_tolerance": (None if bcfg["nonbonded_method"] == "NoCutoff"
                                          else force.getEwaldErrorTolerance()),
            }

    scope = None if bcfg["hmr_scope"] == "all" else range(n_solute_atoms)
    if hmr_scope_name == "none":
        # The conservative default: hydrogens keep the masses the force field gave them. Recorded
        # explicitly rather than omitted, so a manifest states that HMR was OFF instead of leaving
        # a reader to infer it from a missing key.
        # `n_hydrogens_repartitioned`, matching what repartition_hydrogen_mass and
        # verify_hydrogen_mass_repartitioning already emit. A third spelling for the same field
        # would make every consumer handle both.
        hmr = {"scope": "none", "target_hydrogen_mass_amu": None,
               "n_hydrogens_repartitioned": 0,
               "note": "hydrogen mass repartitioning disabled; masses are as parameterised"}
    elif delegate_hmr:
        # OpenMM already did it; verify rather than repeat. The reference System -- the identical
        # call with `hydrogenMass` omitted -- is what makes conservation checkable, and it is built
        # only when HMR was actually asked for.
        hmr = verify_hydrogen_mass_repartitioning(
            system, pdb.topology, target_h_mass, scope)
        reference = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=method,
            nonbondedCutoff=float(bcfg["nonbonded_cutoff_nm"]) * unit.nanometer,
            constraints=constraints,
            rigidWater=bool(bcfg["rigid_water"]),
            removeCMMotion=bool(bcfg["remove_cm_motion"]),
            ewaldErrorTolerance=float(bcfg["ewald_error_tolerance"]),
            residueTemplates=templates,
        )
        hmr["group_conservation"] = verify_hmr_group_masses(
            system, reference, pdb.topology,
            target_h_mass_amu=target_h_mass, solute_atoms=scope)
        del reference
    else:
        hmr = repartition_hydrogen_mass(system, pdb.topology, target_h_mass, scope)

    rcfg = cfg["rest2"]
    if rcfg["unscaled_torsions"]:
        unscaled_info = classify_unscaled_torsions(
            pdb.topology, range(n_solute_atoms), ligand_sdf=ligand_sdf, residue_sdfs=residue_sdfs,
            proline_like_residues=rcfg["proline_like_residues"],
            max_proline_ring_size=int(rcfg["max_proline_ring_size"]),
        )
    else:
        unscaled_info = {
            "unscaled_central_bonds": [], "central_bonds": [], "proline_like_scaled_bonds": [],
            "unclassified": [], "unscaled_impropers": False,
            "detection_method": "disabled (rest2.unscaled_torsions = false): every solute "
                                "torsion is scaled, impropers and ordinary amide omegas included",
            "detector_version": 2,
            "amide_detail": {"unscaled": [], "proline_like_scaled": []},
        }

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
        "unscaled_torsions": {k: unscaled_info[k] for k in
                              ("unscaled_central_bonds", "central_bonds", "proline_like_scaled_bonds", "unclassified",
         "unscaled_impropers", "detection_method", "detector_version", "amide_detail")},
        "forcefield": ff_info,
        "degrees_of_freedom": (
            3 * system.getNumParticles() - system.getNumConstraints()
            - (3 if bcfg["remove_cm_motion"] else 0)
        ),
    }
    (out_dir / "system_build.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
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


#: Name of the global parameter injected into every CustomGBForce energy term. Chosen not to clash
#: with any parameter already present in the GBn2 or HCT expressions.
REST2_GB_SCALE_PARAMETER = "rest2_scale_gb"


def _scale_customgb_force(force, system, solute_set: set, scale_factor: float) -> None:
    """Scale the ENTIRE generalised-Born energy by `s`.

    Charge scaling alone is not enough, and this is the part that is easy to get wrong. GBn2 has
    three energy terms: two are proportional to charge products and would follow `charge * sqrt(s)`
    correctly, but the third is a non-polar / dispersion correction with no charge dependence. It
    still has to be scaled by `s` under REST2, and scaling charges leaves it untouched. Multiplying
    every term by one global parameter scales all three uniformly.

    The whole system must be the enhanced region. A GB energy is not decomposable into per-atom
    contributions the way a bonded term is: every atom's Born radius depends on every other atom's
    position, so a partial selection would need a validated treatment of the solute-environment
    cross terms, and there is none here. Refused rather than approximated.

    The expressions are rewritten in place and the parameter is added to the System, so this MUST
    happen before a Context is created -- the compiled kernels have to already reference it.
    """
    n_particles = system.getNumParticles()
    missing = [i for i in range(n_particles) if i not in solute_set]
    if missing:
        shown = ", ".join(str(i) for i in missing[:8])
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise ValueError(
            f"implicit REST2 requires the entire system to be the enhanced region, but "
            f"{len(missing)} of {n_particles} particles are outside it ({shown}{more}).\n"
            "  A generalised-Born energy is not separable per atom: every Born radius depends on "
            "every other atom's\n"
            "  position, so a partial selection needs a validated treatment of the "
            "solute-environment cross terms.\n"
            "  Refusing rather than approximating it. Set the enhanced region to the whole solute."
        )

    existing = {force.getGlobalParameterName(i)
                for i in range(force.getNumGlobalParameters())}
    if REST2_GB_SCALE_PARAMETER not in existing:
        force.addGlobalParameter(REST2_GB_SCALE_PARAMETER, 1.0)
        for term in range(force.getNumEnergyTerms()):
            expression, computation = force.getEnergyTermParameters(term)
            # Only the leading expression is scaled; everything after the first ';' defines
            # intermediate variables, and multiplying those would change what they mean.
            if ";" in expression:
                head, tail = expression.split(";", 1)
                scaled = f"{REST2_GB_SCALE_PARAMETER}*({head});{tail}"
            else:
                scaled = f"{REST2_GB_SCALE_PARAMETER}*({expression})"
            force.setEnergyTermParameters(term, scaled, computation)

    index = [force.getGlobalParameterName(i)
             for i in range(force.getNumGlobalParameters())].index(REST2_GB_SCALE_PARAMETER)
    force.setGlobalParameterDefaultValue(index, float(scale_factor))


#: Force classes this module knows how to scale. Each has an explicit ``_scale_*`` implementation.
SCALED_FORCE_CLASSES = frozenset({
    "NonbondedForce", "PeriodicTorsionForce", "CMAPTorsionForce", "CustomGBForce",
})

#: Energy-bearing forces left unscaled ON PURPOSE, following the standard REST2 convention: bond
#: and angle terms are not scaled. Scaling them would change the molecule's covalent geometry with
#: temperature, which is not what REST2 does -- the solute's *conformational* barriers are what the
#: scaling is meant to lower, not its bond lengths.
DELIBERATELY_UNSCALED_FORCE_CLASSES = frozenset({
    "HarmonicBondForce", "HarmonicAngleForce",
})

#: Forces that contribute no potential energy, so scaling them is meaningless rather than wrong.
#: A barostat's Monte Carlo move is not a term in U; the centre-of-mass remover only removes drift.
ENERGY_FREE_FORCE_CLASSES = frozenset({
    "CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
    "MonteCarloFlexibleBarostat", "MonteCarloMembraneBarostat", "AndersenThermostat",
    "RMSDForce",
})


class UnclassifiedForceError(ValueError):
    """A System carries an energy-bearing force this module does not know how to scale.

    Raised instead of scaling what is recognised and leaving the rest alone. A force left at s = 1
    inside a ladder whose other terms are scaled is not a smaller effect -- it is a different
    Hamiltonian from the one the ladder claims, and it fails silently: the run completes, the
    exchange log looks healthy, and the acceptance ratio absorbs the discrepancy.
    """


def audit_force_classes(system, *, where: str = "REST2 scaling") -> dict:
    """Classify every force in *system*; raise on any energy-bearing force we cannot place.

    Returns ``{"scaled": [...], "unscaled_by_convention": [...], "energy_free": [...]}`` with the
    force indices in each bucket, so a manifest can record what was scaled rather than assert it.
    """
    scaled, by_convention, energy_free, unknown = [], [], [], []
    for index in range(system.getNumForces()):
        name = system.getForce(index).__class__.__name__
        if name in SCALED_FORCE_CLASSES:
            scaled.append((index, name))
        elif name in DELIBERATELY_UNSCALED_FORCE_CLASSES:
            by_convention.append((index, name))
        elif name in ENERGY_FREE_FORCE_CLASSES:
            energy_free.append((index, name))
        else:
            unknown.append((index, name))
    if unknown:
        listed = ", ".join(f"force[{i}] {n}" for i, n in unknown)
        raise UnclassifiedForceError(
            f"{where}: the System carries {len(unknown)} force(s) this module cannot classify: "
            f"{listed}.\n"
            "  Refusing rather than leaving them at the wrong scale. An unscaled energy term inside "
            "a scaled ladder\n"
            "  is a different Hamiltonian from the one the ladder claims, and nothing downstream "
            "reports it: the run\n"
            "  completes and the acceptance ratio quietly absorbs the discrepancy.\n"
            f"  Known scalable: {sorted(SCALED_FORCE_CLASSES)}\n"
            f"  Unscaled by REST2 convention: {sorted(DELIBERATELY_UNSCALED_FORCE_CLASSES)}\n"
            f"  Carry no potential energy: {sorted(ENERGY_FREE_FORCE_CLASSES)}\n"
            "  If one of these SHOULD be scaled, add an explicit handler; if it carries no energy, "
            "add it to ENERGY_FREE_FORCE_CLASSES with a reason."
        )
    return {"scaled": scaled, "unscaled_by_convention": by_convention, "energy_free": energy_free}


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
    # Before touching anything: refuse a System carrying an energy term we cannot place. Doing this
    # first means the failure is "this System has a force I do not understand", not a half-scaled
    # System that looks finished.
    audit_force_classes(base_system)
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
            _scale_customgb_force(force, system, solute_set, scale_factor)
    return system


