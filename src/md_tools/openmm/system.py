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
        # `addHydrogens` is nondeterministic twice over, and both halves have to be pinned or a
        # bundle cannot be rebuilt. It places each new hydrogen from a RANDOM direction drawn from
        # Python's global `random` -- unseeded, repeat calls move hydrogens by up to 0.18 nm -- and
        # it then relaxes them with a minimisation whose threaded floating-point reductions are
        # themselves order-dependent, leaving ~1e-4 nm of drift even once the RNG is fixed.
        #
        # A tenth of a nanometre on a hydrogen is not a rounding error: the box is sized from the
        # solute's extent, so it changes the box, and a box change of 0.04 Angstrom was enough to
        # add or drop one whole water molecule between builds. Bundle hashes, relocation checks and
        # every "which value did I choose?" provenance comparison depend on this being stable.
        #
        # Seeding alone leaves the floating-point half, so the relaxation runs on the Reference
        # platform, which is single-threaded and reproducible. Together these are bit-identical
        # across builds. Reference is slower, but this minimises only the added hydrogens of a
        # solute -- a macrocycle or a small peptide here -- so the cost is seconds.
        from .seeds import DEFAULT_MASTER_SEED, derive_build_seed as derive_seed

        master = (cfg.get("run") or {}).get("seed")
        hydrogen_seed = derive_seed(
            int(master if master is not None else DEFAULT_MASTER_SEED), "structure/protonation")
        from openmm import Platform

        reference = Platform.getPlatformByName("Reference")
        state = random.getstate()
        random.seed(hydrogen_seed)
        try:
            added = modeller.addHydrogens(
                forcefield, pH=float(pcfg["ph"]), variants=pcfg["variants"], platform=reference
            )
        finally:
            random.setstate(state)
        note = (f"addHydrogens at pH {pcfg['ph']}, seed {hydrogen_seed}, "
                "relaxed on the Reference platform for reproducibility")

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


def classify_omega_bonds(topology, solute_atoms: Iterable[int], *,
                         ligand_sdf: Optional[Path] = None,
                         residue_sdfs: Optional[dict] = None,
                         proline_like_residues: Iterable[str] = ("PRO",),
                         max_proline_ring_size: int = 7) -> dict:
    """Split the solute's amide C-N bonds into REST2-unscaled, proline-like-scaled, unclassified.

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

    Returns a dict with ``omega_unscaled_bonds``, ``omega_proline_like_scaled_bonds``,
    ``omega_unclassified_candidates`` and ``omega_detection_method``.  **A non-empty unclassified
    list must block production** -- it means a candidate was found that neither rule could name, and
    guessing would silently change the Hamiltonian.
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
            ring_info = _ligand_ring_nitrogens(ligand_sdf, topology, non_standard,
                                               max_proline_ring_size,
                                               describes=sorted(non_standard_names))
        except ValueError as refusal:
            mapping_error = str(refusal)

    method = ("omega evidence chosen per candidate: residue-aware against PROTEIN_RESIDUES, "
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
                instance_info[residue_index] = (_ligand_ring_nitrogens(
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
                f"supplied. `build-top` retains one beside the System (`built.sdf`) for a .smi or "
                f".sdf input and writes none for a peptide; supply that SDF beside the System.")))
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
    return {
        "omega_unscaled_bonds": [c["bond"] for c in unscaled],
        "omega_proline_like_scaled_bonds": [c["bond"] for c in proline],
        "omega_unclassified_candidates": unknown,
        "omega_detection_method": method,
        "omega_detail": {"unscaled": unscaled, "proline_like_scaled": proline},
    }


class UnclassifiedOmegaError(ValueError):
    """An amide candidate neither rule could name, on a surface that is about to scale torsions."""


def omega_exclusions(topology, solute_atoms: Iterable[int], *,
                     ligand_sdf: Optional[Path] = None,
                     residue_sdfs: Optional[dict] = None,
                     proline_like_residues: Iterable[str] = ("PRO",),
                     max_proline_ring_size: int = 7) -> dict:
    """:func:`classify_omega_bonds`, ENFORCED: the one entry point for anything that scales.

    The classifier reports what it could not decide and leaves acting on it to the caller. Six
    callers took ``omega_unscaled_bonds`` and only one of them read the unclassified list, so a
    candidate nobody could name was left out of the exclusions and SCALED like any other solute
    torsion -- the run completes, the acceptance ratios look plausible, and the ordinary-amide
    invariant is broken with nothing saying so. A build record may still call the classifier
    directly, because recording a candidate is not scaling it; every scaling surface calls this.

    Returns the classification unchanged when every candidate was decided.
    """
    omega = classify_omega_bonds(topology, solute_atoms, ligand_sdf=ligand_sdf,
                                 residue_sdfs=residue_sdfs,
                                 proline_like_residues=proline_like_residues,
                                 max_proline_ring_size=max_proline_ring_size)
    unknown = omega["omega_unclassified_candidates"]
    if not unknown:
        return omega
    shown = unknown[:5]
    lines = [f"  bond {c['bond'][0]}-{c['bond'][1]}: "
             f"{c.get('carbon_residue')}{c.get('carbon_residue_index')} C -> "
             f"{c.get('nitrogen_residue')}{c.get('nitrogen_residue_index')} N: {c['ambiguous']}"
             for c in shown]
    if len(unknown) > len(shown):
        lines.append(f"  ... and {len(unknown) - len(shown)} more")
    raise UnclassifiedOmegaError(
        f"{len(unknown)} amide omega candidate(s) could not be classified as ordinary (left "
        f"unscaled) or proline-like (scaled). Scaling one that is ordinary lets a hot state "
        f"isomerise a peptide bond the reference never does; exempting one that is not changes "
        f"the Hamiltonian the other way. Neither is guessed, so nothing is scaled:\n"
        + "\n".join(lines))


def _ligand_ring_nitrogens(ligand_sdf, topology, atoms_to_map: set[int], max_ring: int,
                           *, describes: Optional[list] = None) -> dict:
    """RDKit amide perception on the retained SDF, mapped onto OpenMM indices.

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
            "ring_sizes": ring_sizes}


def omega_central_bonds(topology, solute_atoms: Iterable[int]) -> list[tuple[int, int]]:
    """DEPRECATED structural detector: every amide C-N bond, with no proline-like exception.

    Kept so pre-2026-08-14 bundles can be re-derived.  It treats an X-PRO peptide bond as an
    ordinary omega and excludes it from scaling, which
    :func:`classify_omega_bonds` deliberately does not.  New code must use that function.
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
        nonbondedCutoff=float(bcfg["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=constraints,
        rigidWater=bool(bcfg["rigid_water"]),
        removeCMMotion=bool(bcfg["remove_cm_motion"]),
        ewaldErrorTolerance=float(bcfg["ewald_error_tolerance"]),
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
                "cutoff_nm": force.getCutoffDistance().value_in_unit(unit.nanometer),
                "switching": switching,
                # null, not 0.0, when the switching function is off: OpenMM keeps a switching
                # distance on the Force whether or not it is used, and reporting it unconditionally
                # describes a taper that is not applied.
                "switch_distance_nm": (
                    force.getSwitchingDistance().value_in_unit(unit.nanometer)
                    if switching else None),
                "dispersion_correction": bool(force.getUseDispersionCorrection()),
                "ewald_error_tolerance": force.getEwaldErrorTolerance(),
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
        )
        hmr["group_conservation"] = verify_hmr_group_masses(
            system, reference, pdb.topology,
            target_h_mass_amu=target_h_mass, solute_atoms=scope)
        del reference
    else:
        hmr = repartition_hydrogen_mass(system, pdb.topology, target_h_mass, scope)

    rcfg = cfg["rest2"]
    if rcfg["omega_exclusion"]:
        omega_info = classify_omega_bonds(
            pdb.topology, range(n_solute_atoms), ligand_sdf=ligand_sdf,
            proline_like_residues=rcfg["proline_like_residues"],
            max_proline_ring_size=int(rcfg["max_proline_ring_size"]),
        )
    else:
        omega_info = {
            "omega_unscaled_bonds": [], "omega_proline_like_scaled_bonds": [],
            "omega_unclassified_candidates": [],
            "omega_detection_method": "disabled (rest2.omega_exclusion = false): every torsion "
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


