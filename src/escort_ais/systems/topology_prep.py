"""Parameterised protein–ligand MD topology preparation.

Builds Amber prmtop/inpcrd via OpenMM app.ForceField + openmmforcefields
template generators (Sage/GAFF), then writes Amber-format files through a
ParmEd round-trip so all downstream REST2/AIS code is unchanged.

Supported force-field strings
-------------------------------
  ff19sb         protein only (ff19SB)
  sage           small-molecule only (Sage/OpenFF)
  gaff           small-molecule only (GAFF2)
  ff19sb+sage    protein (ff19SB) + ligand (Sage)
  ff19sb+gaff    protein (ff19SB) + ligand (GAFF2)

Solvent modes
-------------
  opc   explicit OPC box with NaCl (default 0.15 M)
  gbn2  implicit GBn2, no box, no salt

Example
-------
  result = build_topology(
      output_dir=Path("data/topology/my_system"),
      protein_sequence=["ACE", "ALA", "NME"],
      ligand_smiles="CCO",
      forcefield="ff19sb+sage",
      solvent="opc",
      salt_concentration_molar=0.15,
  )
  # result["prmtop"], ["inpcrd"], ["topology_pdb"], ["n_solute_atoms"]
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# openmm and openmm_system are imported lazily inside functions so that
# _has_protein_ff, _ligand_ff, and _needs_rebuild remain importable without openmm.
DEFAULT_GB_MODEL = "GBn2"
DEFAULT_SALT_CONCENTRATION_MOLAR = 0.15  # mol/L

DEFAULT_FORCEFIELD = "ff19sb+sage"
DEFAULT_SOLVENT = "opc"
_SAGE_VERSION = "openff-2.2.0"
_GAFF_VERSION = "gaff-2.11"

VALID_FORCEFIELDS = frozenset({"ff19sb", "sage", "gaff", "ff19sb+sage", "ff19sb+gaff"})
VALID_SOLVENTS = frozenset({"opc", "gbn2"})


def _scale_to_label(scale: float) -> str:
    """Convert a REST2 scale factor to a filename-safe label.

    Examples: 1.0 → '1p0', 0.2 → '0p2', 0.1875 → '0p1875'
    """
    txt = f"{scale:.4f}".rstrip("0")
    if txt.endswith("."):
        txt += "0"
    return txt.replace(".", "p")


def _prmtop_name(scale: float) -> str:
    """Return the canonical prmtop filename for a given scale factor."""
    return f"system_scale_{_scale_to_label(scale)}.prmtop"


def _find_base_prmtop(topology_dir: Path) -> Path:
    """Return the base (unscaled) prmtop path, checking new and legacy names."""
    for name in ("system_scale_1p0.prmtop", "system.prmtop"):
        p = topology_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No prmtop found in {topology_dir}. "
        "Expected system_scale_1p0.prmtop or system.prmtop."
    )


def effective_temperature_ladder(
    base_temperature_k: float,
    max_effective_temperature_k: float,
    n_replicas: int,
) -> list[float]:
    """Return a REST2 effective-temperature ladder with linear-in-√scale spacing.

    The ladder spans from *base_temperature_k* (replica 0, scale_factor=1) to
    *max_effective_temperature_k* (replica n-1, lowest scale_factor).

    ``result[0] == base_temperature_k``, ``result[-1] == max_effective_temperature_k``.
    """
    if n_replicas < 2:
        raise ValueError(f"n_replicas must be >= 2, got {n_replicas}")
    if max_effective_temperature_k < base_temperature_k:
        raise ValueError(
            f"max_effective_temperature_k ({max_effective_temperature_k}) must be "
            f">= base_temperature_k ({base_temperature_k})"
        )
    sqrt_scales = np.linspace(
        1.0,
        math.sqrt(base_temperature_k / max_effective_temperature_k),
        n_replicas,
    )
    scale_factors = sqrt_scales ** 2
    return [float(base_temperature_k / s) for s in scale_factors]


def rest2_scale_factors(
    base_temperature_k: float,
    max_effective_temperature_k: float,
    n_replicas: int,
) -> list[float]:
    """Return REST2 scale factors ``s = T_base / T_eff`` for a replica ladder.

    ``result[0] == 1.0`` (physical replica).  Spacing is linear-in-√scale.
    """
    ladder = effective_temperature_ladder(
        base_temperature_k, max_effective_temperature_k, n_replicas
    )
    return [float(base_temperature_k / t) for t in ladder]


def resolve_remd_scale_ladder(
    scale_factors: list[float],
    base_temperature_k: float = 300.0,
) -> tuple[list[float], list[float]]:
    """Validate and sort a user-supplied REST2 scale-factor list for REMD.

    Returns ``(scale_factors_descending, t_eff_descending)`` where
    ``scale_factors_descending[0] == 1.0`` (physical replica) by convention,
    matching the ordering produced by :func:`rest2_scale_factors`.

    Parameters
    ----------
    scale_factors:
        Raw scale factors (any order); must satisfy ``0 < s ≤ 1`` and
        ``len ≥ 2``.
    base_temperature_k:
        Bath / physical temperature in K.  ``T_eff = base_temperature_k / s``.

    Returns
    -------
    tuple of two lists:
      ``(scale_factors_desc, effective_temperatures_desc)``

    Raises
    ------
    ValueError
        If fewer than 2 replicas, or any scale factor is out of the ``(0, 1]``
        range.
    """
    if len(scale_factors) < 2:
        raise ValueError(
            f"REMD requires at least 2 replicas; got {len(scale_factors)} scale factors."
        )
    for s in scale_factors:
        if not (0.0 < s <= 1.0):
            raise ValueError(
                f"Every scale factor must be in the range (0, 1]; got {s!r}."
            )
    # Physical replica (s=1) must be replica 0 — descending order
    sorted_desc = sorted(scale_factors, reverse=True)
    t_eff_desc = [float(base_temperature_k / s) for s in sorted_desc]
    return sorted_desc, t_eff_desc


def _has_protein_ff(forcefield: str) -> bool:
    return "ff19sb" in forcefield


def _ligand_ff(forcefield: str) -> Optional[str]:
    """Return 'sage', 'gaff', or None depending on the forcefield string."""
    if "sage" in forcefield:
        return "sage"
    if "gaff" in forcefield:
        return "gaff"
    return None


def _build_protein_pdb_from_sequence(sequence: list[str], output_pdb: Path) -> None:
    """Run tleap to build a 3-D PDB from an Amber residue sequence."""
    from escort_ais.systems.openmm_system import _assert_tleap_available
    _assert_tleap_available()
    seq_str = " ".join(sequence)
    script = (
        "source leaprc.protein.ff19SB\n"
        f"mol = sequence {{ {seq_str} }}\n"
        f"savePdb mol {output_pdb.resolve()}\n"
        "quit\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        leap_in = Path(fh.name)

    result = subprocess.run(
        ["tleap", "-f", str(leap_in)],
        capture_output=True, text=True, check=False,
        cwd=str(output_pdb.parent),
    )
    leap_in.unlink(missing_ok=True)

    if result.returncode != 0 or not output_pdb.exists():
        raise RuntimeError(
            f"tleap failed building protein from sequence {sequence}.\n{result.stderr}"
        )


def _build_protein_topology_via_tleap(
    sequence: list[str],
    prmtop_path: Path,
    inpcrd_path: Path,
    topology_pdb_path: Path,
    radii: str = "mbondi3",
) -> None:
    """Build prmtop/inpcrd/PDB directly via tleap with ff19SB and specified radii.

    mbondi3 radii are the recommended choice for GBn2 implicit solvent.
    """
    from escort_ais.systems.openmm_system import _assert_tleap_available
    _assert_tleap_available()
    seq_str = " ".join(sequence)
    script = "\n".join([
        "source leaprc.protein.ff19SB",
        f"set default PBRadii {radii}",
        f"mol = sequence {{ {seq_str} }}",
        f"saveAmberParm mol {prmtop_path.resolve()} {inpcrd_path.resolve()}",
        f"savePdb mol {topology_pdb_path.resolve()}",
        "quit",
    ]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        leap_in = Path(fh.name)
    result = subprocess.run(
        ["tleap", "-f", str(leap_in)],
        capture_output=True, text=True, check=False,
        cwd=str(prmtop_path.parent),
    )
    leap_in.unlink(missing_ok=True)
    if result.returncode != 0 or not prmtop_path.exists():
        raise RuntimeError(
            f"tleap failed building protein topology from sequence {sequence}.\n{result.stderr}"
        )


def _build_protein_topology_via_tleap_from_pdb(
    protein_pdb: Path,
    prmtop_path: Path,
    inpcrd_path: Path,
    topology_pdb_path: Path,
    radii: str = "mbondi3",
) -> None:
    """Build prmtop/inpcrd via tleap loadPdb for a pre-existing protein PDB."""
    from escort_ais.systems.openmm_system import _assert_tleap_available
    _assert_tleap_available()
    script = "\n".join([
        "source leaprc.protein.ff19SB",
        f"set default PBRadii {radii}",
        f"mol = loadPdb {protein_pdb.resolve()}",
        f"saveAmberParm mol {prmtop_path.resolve()} {inpcrd_path.resolve()}",
        f"savePdb mol {topology_pdb_path.resolve()}",
        "quit",
    ]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        leap_in = Path(fh.name)
    result = subprocess.run(
        ["tleap", "-f", str(leap_in)],
        capture_output=True, text=True, check=False,
        cwd=str(prmtop_path.parent),
    )
    leap_in.unlink(missing_ok=True)
    if result.returncode != 0 or not prmtop_path.exists():
        raise RuntimeError(
            f"tleap failed building protein topology from PDB {protein_pdb}.\n{result.stderr}"
        )


def _load_offmol(ligand_sdf: Optional[Path], ligand_smiles: Optional[str]):
    """Load an OpenFF Molecule, generating a conformer if none is present."""
    from openff.toolkit import Molecule

    if ligand_sdf is not None:
        mol = Molecule.from_file(str(ligand_sdf))
    elif ligand_smiles is not None:
        mol = Molecule.from_smiles(ligand_smiles)
    else:
        raise ValueError("Must provide ligand_sdf or ligand_smiles.")

    if mol.conformers is None or len(mol.conformers) == 0:
        mol.generate_conformers(n_conformers=1)

    return mol


def _offmol_positions_nm(offmol) -> np.ndarray:
    """Extract the first conformer of an OpenFF Molecule as a (N, 3) array in nm."""
    conf = offmol.conformers[0]
    try:
        ang = conf.m_as("angstrom")
    except AttributeError:
        ang = np.asarray(conf.magnitude)
    return np.asarray(ang, dtype=float) * 0.1


def _needs_rebuild(
    prmtop: Path,
    inpcrd: Path,
    topology_pdb: Path,
    config_path: Path,
    forcefield: str,
    solvent: str,
    box_padding_angstrom: float,
    salt_concentration_molar: float,
) -> bool:
    """Return True when any output file is missing or cached parameters differ."""
    if not (prmtop.exists() and inpcrd.exists() and topology_pdb.exists()):
        return True
    if not config_path.exists():
        return True
    try:
        stored = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if stored.get("forcefield") != forcefield or stored.get("solvent") != solvent:
        return True
    if solvent == "opc":
        if abs(stored.get("box_padding_angstrom", 0.0) - box_padding_angstrom) > 1e-6:
            return True
        if abs(stored.get("salt_concentration_molar", 0.0) - salt_concentration_molar) > 1e-6:
            return True
        # Require the sim-ready system.xml; any topology built before the virtual-site fix
        # will lack it and must be rebuilt.
        if not (prmtop.parent / "system.xml").exists():
            return True
    return False


def build_topology(
    output_dir: Path,
    *,
    protein_pdb: Optional[Path] = None,
    protein_sequence: Optional[list[str]] = None,
    ligand_sdf: Optional[Path] = None,
    ligand_smiles: Optional[str] = None,
    forcefield: str = DEFAULT_FORCEFIELD,
    solvent: str = DEFAULT_SOLVENT,
    box_padding_angstrom: float = 12.0,
    salt_concentration_molar: float = DEFAULT_SALT_CONCENTRATION_MOLAR,
    cv_slug: Optional[str] = None,
    cv_kind: Optional[str] = None,
    cv_quartet_npz: Optional[Path] = None,
) -> dict[str, object]:
    """Build (or load cached) Amber prmtop/inpcrd for a protein and/or ligand.

    The returned dict always contains:
      prmtop         Path to Amber prmtop (readable by app.AmberPrmtopFile)
      inpcrd         Path to Amber inpcrd / rst7
      topology_pdb   Path to PDB with all atoms (for atom-mapping utilities)
      config_json    Path to JSON recording build parameters
      n_solute_atoms Number of solute (protein + ligand) atoms; solute occupies
                     topology indices [0, n_solute_atoms) — required by REST2.

    CV declaration. Pass ``cv_slug`` + ``cv_kind`` ("dipeptide" or "macrocycle", plus
    ``cv_quartet_npz`` for a macrocycle) and ``systems/<cv_slug>/cv_definition.json`` is written
    here if absent, and VERIFIED against the freshly built topology if present. Declaring the CVs
    at construction time is what stops a downstream writer from re-deriving them: re-derivation
    via ``mdtraj.compute_phi/psi`` finds nothing on a cyclic macrocycle and silently recorded
    183 000 all-NaN rows. The result then carries ``cv_definition``.

    Caching: existing files are reused when all parameters match config_json.
    If salt, padding, forcefield, or solvent changed, the topology is rebuilt.
    """
    # ------------------------------------------------------------------ #
    # Validation (pure Python — no openmm required)                        #
    # ------------------------------------------------------------------ #
    forcefield = forcefield.lower().strip()
    solvent = solvent.lower().strip()

    if forcefield not in VALID_FORCEFIELDS:
        raise ValueError(f"Invalid forcefield '{forcefield}'. Valid: {sorted(VALID_FORCEFIELDS)}")
    if solvent not in VALID_SOLVENTS:
        raise ValueError(f"Invalid solvent '{solvent}'. Valid: {sorted(VALID_SOLVENTS)}")

    want_protein = _has_protein_ff(forcefield)
    ligand_engine = _ligand_ff(forcefield)

    if want_protein and protein_pdb is None and protein_sequence is None:
        raise ValueError(
            f"Forcefield '{forcefield}' requires a protein (protein_pdb or protein_sequence)."
        )
    if ligand_engine and ligand_sdf is None and ligand_smiles is None:
        raise ValueError(
            f"Forcefield '{forcefield}' requires a ligand (ligand_sdf or ligand_smiles)."
        )
    if not want_protein and (protein_pdb is not None or protein_sequence is not None):
        raise ValueError(
            f"protein_pdb/protein_sequence provided but '{forcefield}' has no protein component."
        )
    if not ligand_engine and (ligand_sdf is not None or ligand_smiles is not None):
        raise ValueError(
            f"Ligand input provided but '{forcefield}' has no ligand component."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prmtop_path = output_dir / "system_scale_1p0.prmtop"
    inpcrd_path = output_dir / "system_scale_1p0.inpcrd"
    topology_pdb_path = output_dir / "topology.pdb"
    config_path = output_dir / "config.json"

    if not _needs_rebuild(
        prmtop_path, inpcrd_path, topology_pdb_path, config_path,
        forcefield, solvent, box_padding_angstrom, salt_concentration_molar,
    ):
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "prmtop": prmtop_path,
            "inpcrd": inpcrd_path,
            "topology_pdb": topology_pdb_path,
            "config_json": config_path,
            "n_solute_atoms": int(stored.get("n_solute_atoms", 0)),
        }

    # ------------------------------------------------------------------ #
    # Heavy imports (openmm required from here on)                         #
    # ------------------------------------------------------------------ #
    from openmm import Vec3, app, unit  # noqa: PLC0415

    # ------------------------------------------------------------------ #
    # Step 1 — protein structure                                           #
    # ------------------------------------------------------------------ #
    prot_topology = None
    prot_positions = None
    n_protein_atoms = 0

    if want_protein:
        if protein_pdb is not None:
            prot_obj = app.PDBFile(str(protein_pdb))
        else:
            seq_pdb = output_dir / "_sequence.pdb"
            _build_protein_pdb_from_sequence(protein_sequence, seq_pdb)
            prot_obj = app.PDBFile(str(seq_pdb))
        prot_topology = prot_obj.topology
        prot_positions = prot_obj.positions
        n_protein_atoms = prot_topology.getNumAtoms()

    # ------------------------------------------------------------------ #
    # Step 2 — ligand                                                      #
    # ------------------------------------------------------------------ #
    lig_topology: Optional[app.Topology] = None
    lig_positions = None
    offmol = None
    n_ligand_atoms = 0

    if ligand_engine:
        offmol = _load_offmol(ligand_sdf, ligand_smiles)
        lig_topology = offmol.to_topology().to_openmm()
        pos_nm = _offmol_positions_nm(offmol)
        lig_positions = [
            Vec3(float(r[0]), float(r[1]), float(r[2])) for r in pos_nm
        ] * unit.nanometer
        n_ligand_atoms = lig_topology.getNumAtoms()

    n_solute_atoms = n_protein_atoms + n_ligand_atoms

    # ------------------------------------------------------------------ #
    # Steps 3–7 — build prmtop/inpcrd/PDB                                #
    # ------------------------------------------------------------------ #
    n_ions = 0

    if solvent == "gbn2" and not ligand_engine:
        # Tleap path: generates proper mbondi3 radii required for GBn2.
        # ForceField + ParmEd cannot reliably embed GB radii in Amber prmtop.
        if protein_sequence is not None:
            _build_protein_topology_via_tleap(
                protein_sequence, prmtop_path, inpcrd_path, topology_pdb_path
            )
        else:
            # PDB input: use tleap loadPdb
            _build_protein_topology_via_tleap_from_pdb(
                protein_pdb, prmtop_path, inpcrd_path, topology_pdb_path
            )
        # Recount from tleap prmtop (tleap may add H atoms not in input PDB)
        _prmtop_obj = app.AmberPrmtopFile(str(prmtop_path))
        n_protein_atoms = sum(1 for _ in _prmtop_obj.topology.atoms())
        n_solute_atoms = n_protein_atoms
    else:
        # ForceField + ParmEd path (OPC explicit, or gbn2 + ligand)
        ff_xmls: list[str] = []
        if want_protein:
            ff_xmls.append("amber/protein.ff19SB.xml")
        if solvent == "opc":
            ff_xmls.append("amber/opc_standard.xml")

        omm_ff = app.ForceField(*ff_xmls) if ff_xmls else app.ForceField()

        if ligand_engine == "sage":
            from openmmforcefields.generators import SMIRNOFFTemplateGenerator
            gen = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield=_SAGE_VERSION)
            omm_ff.registerTemplateGenerator(gen.generator)
        elif ligand_engine == "gaff":
            from openmmforcefields.generators import GAFFTemplateGenerator
            gen = GAFFTemplateGenerator(molecules=[offmol], forcefield=_GAFF_VERSION)
            omm_ff.registerTemplateGenerator(gen.generator)

        if want_protein:
            modeller = app.Modeller(prot_topology, prot_positions)
            if ligand_engine:
                modeller.add(lig_topology, lig_positions)
        else:
            modeller = app.Modeller(lig_topology, lig_positions)

        if solvent == "opc":
            modeller.addSolvent(
                omm_ff,
                model="tip4pew",
                padding=box_padding_angstrom * 0.1 * unit.nanometer,
                ionicStrength=salt_concentration_molar * unit.molar,
                neutralize=True,
            )
            n_ions = sum(
                1 for r in modeller.topology.residues()
                if r.name in {"NA", "CL", "Na+", "Cl-", "SOD", "CLA"}
            )
            # ParmEd needs explicit bonds, so build a constraints=None system for the prmtop
            # save, but also serialize a *sim-ready* System (rigidWater=True, HBonds) that
            # retains OPC virtual sites.  The Amber prmtop round-trip silently drops VirtualSite
            # frame definitions, so downstream code must load this XML instead of rebuilding
            # from the prmtop whenever explicit solvent is used.
            system = omm_ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=None,
                rigidWater=False,
            )
            from openmm import XmlSerializer  # noqa: PLC0415
            sim_system = omm_ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True,
            )
            n_virtual_sites = sum(
                1 for i in range(sim_system.getNumParticles())
                if sim_system.isVirtualSite(i)
            )
            if n_virtual_sites == 0:
                raise RuntimeError(
                    "OPC system has 0 virtual sites — the virtual-site frames were not "
                    "created.  Check that 'amber/opc_standard.xml' is loaded and that the "
                    "topology includes TIP4P-compatible residues."
                )
            sim_system_xml = XmlSerializer.serialize(sim_system)
            system_xml_path = output_dir / "system.xml"
            system_xml_path.write_text(sim_system_xml, encoding="utf-8")
        else:  # gbn2 + ligand — GB forces added at simulation time via prmtop.createSystem()
            system = omm_ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=None,
            )

        from parmed import openmm as pmd_omm

        structure = pmd_omm.load_topology(
            modeller.topology,
            system=system,
            xyz=modeller.positions,
        )
        structure.save(str(prmtop_path), overwrite=True)
        structure.save(str(inpcrd_path), format="rst7", overwrite=True)

        with topology_pdb_path.open("w", encoding="utf-8") as fh:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    # Keep legacy name as a copy for tools that haven't adopted the new naming yet
    shutil.copy2(str(prmtop_path), str(output_dir / "system.prmtop"))

    # ------------------------------------------------------------------ #
    # Step 8 — config.json                                                 #
    # ------------------------------------------------------------------ #
    config: dict[str, object] = {
        "forcefield": forcefield,
        "solvent": solvent,
        "n_solute_atoms": n_solute_atoms,
        "n_protein_atoms": n_protein_atoms,
        "n_ligand_atoms": n_ligand_atoms,
        "n_ions": n_ions,
        "generator": "openmm.app.ForceField + parmed",
        "prmtop": prmtop_path.name,
    }
    if want_protein:
        config["protein_source"] = (
            str(protein_pdb) if protein_pdb is not None else " ".join(protein_sequence)
        )
    if ligand_engine:
        config["ligand_forcefield"] = ligand_engine
        config["ligand_forcefield_version"] = (
            _SAGE_VERSION if ligand_engine == "sage" else _GAFF_VERSION
        )
        config["ligand_source"] = (
            str(ligand_sdf) if ligand_sdf is not None else ligand_smiles
        )
    if solvent == "opc":
        config["box_padding_angstrom"] = box_padding_angstrom
        config["salt_concentration_molar"] = salt_concentration_molar
        config["water_model"] = "OPC"
        config["system_xml"] = "system.xml"  # sim-ready XML with OPC virtual sites
    else:
        config["gb_model"] = DEFAULT_GB_MODEL

    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result: dict[str, object] = {
        "prmtop": prmtop_path,
        "inpcrd": inpcrd_path,
        "topology_pdb": topology_pdb_path,
        "config_json": config_path,
        "n_solute_atoms": n_solute_atoms,
    }
    if solvent == "opc":
        result["system_xml"] = output_dir / "system.xml"

    if cv_slug is not None:
        if cv_kind is None:
            raise ValueError("cv_slug given without cv_kind ('dipeptide' or 'macrocycle')")
        from escort_ais.systems.cv_definition import ensure_cv_definition  # noqa: PLC0415

        result["cv_definition"] = ensure_cv_definition(
            cv_slug, cv_kind, topology_pdb_path, quartet_npz=cv_quartet_npz)
    return result


def build_rest2_systems(
    topology_dir: Path,
    *,
    mode: str,
    scale_factor: float | None = None,
    n_replicas: int | None = None,
    base_temperature_k: float = 300.0,
    effective_temperature_max_k: float = 450.0,
    scale_factors: list[float] | None = None,
    gb_radii: str | None = None,
) -> dict[str, object]:
    """Build REST2-scaled OpenMM System XML files from an existing topology dir.

    The topology dir must already contain ``system.prmtop``, ``system.inpcrd``,
    and ``config.json`` produced by :func:`build_topology`.  Scaled Systems are
    serialized as OpenMM System XML files under ``<topology_dir>/rest2/`` because
    CMAP (ff19SB) and CustomGB (GBn2) scalings cannot be stored in Amber prmtop.

    Parameters
    ----------
    topology_dir:
        Directory created by ``build_topology``.
    mode:
        ``"single"`` — one system at *scale_factor*.
        ``"remd"``   — N replica systems on a temperature ladder.
    scale_factor:
        Raw REST2 scale factor for ``mode="single"`` (1.0 = physical / no scaling).
    n_replicas:
        Number of replicas for ``mode="remd"`` (≥ 2).  Ignored when
        *scale_factors* is provided.
    base_temperature_k:
        Physical temperature (K) for the REMD ladder base.
    effective_temperature_max_k:
        Hottest effective temperature (K) for the REMD ladder (√-spaced ladder
        only; ignored when *scale_factors* is provided).
    scale_factors:
        Explicit list of REST2 scale factors for ``mode="remd"``.  When
        supplied, the √-spaced ladder derived from *effective_temperature_max_k*
        is bypassed and these exact values are used instead.  All values must be
        in ``(0, 1]``; the list must contain ≥ 2 entries.  Automatically sorted
        in descending order so that replica 0 is always the physical system
        (``s = 1.0``).

    Returns
    -------
    dict with keys:
      mode, system_xml_paths, scale_factors, effective_temperatures_k, ladder_json
    """
    topology_dir = Path(topology_dir)

    if mode not in ("single", "remd"):
        raise ValueError(f"mode must be 'single' or 'remd', got '{mode}'")
    if mode == "single":
        if scale_factor is None:
            raise ValueError("scale_factor is required for mode='single'")
        if not (0.0 < scale_factor <= 1.0):
            raise ValueError(f"scale_factor must be in (0, 1], got {scale_factor}")
    if mode == "remd":
        if scale_factors is None:
            # Legacy path: auto-derive √-spaced ladder
            if n_replicas is None:
                raise ValueError("n_replicas is required for mode='remd' when scale_factors is not given")
            if n_replicas < 2:
                raise ValueError(f"n_replicas must be >= 2, got {n_replicas}")

    config_path = topology_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {topology_dir}. Run build_topology first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    solvent = config["solvent"]
    n_solute_atoms = int(config["n_solute_atoms"])

    prmtop_path = _find_base_prmtop(topology_dir)

    # Heavy imports
    from openmm import app, unit, XmlSerializer  # noqa: PLC0415
    from escort_ais.systems.openmm_system import build_rest2_scaled_system  # noqa: PLC0415

    # Build base System.
    # For OPC explicit solvent, prefer the pre-serialized system.xml written by build_topology.
    # The Amber prmtop round-trip drops OPC virtual-site frame definitions, so rebuilding from
    # prmtop.createSystem() would produce 0 VirtualSites → NaN during minimization.
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system_xml_path = topology_dir / "system.xml"
    if solvent == "opc":
        if system_xml_path.exists():
            base_system = XmlSerializer.deserialize(
                system_xml_path.read_text(encoding="utf-8")
            )
        else:
            # Fallback: rebuild from OpenMM topology (topology.pdb must exist)
            topo_pdb_path = topology_dir / "topology.pdb"
            if not topo_pdb_path.exists():
                raise FileNotFoundError(
                    f"Neither system.xml nor topology.pdb found in {topology_dir}. "
                    "Please rebuild the topology with the latest build_topology() to "
                    "generate system.xml with OPC virtual sites."
                )
            ff_xmls = ["amber/protein.ff19SB.xml", "amber/opc_standard.xml"]
            omm_ff = app.ForceField(*ff_xmls)
            topo_obj = app.PDBFile(str(topo_pdb_path))
            base_system = omm_ff.createSystem(
                topo_obj.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True,
            )
        # Defensive assertion: OPC base system must have virtual sites
        n_vs = sum(1 for i in range(base_system.getNumParticles()) if base_system.isVirtualSite(i))
        if n_vs == 0:
            raise RuntimeError(
                f"OPC base system loaded from {topology_dir} has 0 virtual sites. "
                "The prmtop round-trip drops VirtualSite frames.  Rebuild the topology "
                "with the latest build_topology() which writes system.xml."
            )
    else:  # gbn2 — GBn2 + mbondi3 via parmed is enforced; see openmm_system.build_implicit_system
        from escort_ais.systems.openmm_system import (  # noqa: PLC0415
            DEFAULT_GB_RADII, build_implicit_system)
        base_system = build_implicit_system(prmtop_path, gb_radii=gb_radii or DEFAULT_GB_RADII)

    solute_indices = np.arange(n_solute_atoms)

    # Determine scale factors and effective temperatures per replica
    if mode == "single":
        scale_factors_list = [scale_factor]
        t_eff_list = [base_temperature_k / scale_factor]
    else:
        if scale_factors is not None:
            # Explicit linear (or arbitrary) scale list
            scale_factors_list, t_eff_list = resolve_remd_scale_ladder(
                scale_factors, base_temperature_k
            )
        else:
            # Legacy √-spaced ladder
            scale_factors_list = rest2_scale_factors(
                base_temperature_k, effective_temperature_max_k, n_replicas
            )
            t_eff_list = effective_temperature_ladder(
                base_temperature_k, effective_temperature_max_k, n_replicas
            )

    rest2_dir = topology_dir / "rest2"
    rest2_dir.mkdir(exist_ok=True)

    xml_paths: list[Path] = []
    for i, s in enumerate(scale_factors_list):
        scaled = build_rest2_scaled_system(base_system, solute_indices, s)
        xml_str = XmlSerializer.serialize(scaled)
        if mode == "single":
            fname = f"scaled_s{s:.4f}.xml"
        else:
            fname = f"replica_{i:02d}_s{s:.4f}.xml"
        xml_path = rest2_dir / fname
        xml_path.write_text(xml_str, encoding="utf-8")
        xml_paths.append(xml_path)

    # Write scale-labelled prmtop copies in the topology dir (same bytes, informative names)
    written_labels: set[str] = set()
    for s in scale_factors_list:
        label = _scale_to_label(s)
        if label not in written_labels:
            dest = topology_dir / _prmtop_name(s)
            if not dest.exists() or dest.stat().st_mtime < prmtop_path.stat().st_mtime:
                shutil.copy2(str(prmtop_path), str(dest))
            written_labels.add(label)
    # Ensure the physical (scale=1.0) prmtop always exists
    phys_prmtop = topology_dir / "system_scale_1p0.prmtop"
    if not phys_prmtop.exists():
        shutil.copy2(str(prmtop_path), str(phys_prmtop))

    ladder_json_path: Path | None = None
    if mode == "remd":
        ladder = [
            {
                "replica": i,
                "scale_factor": scale_factors_list[i],
                "effective_temperature_k": t_eff_list[i],
                "system_xml": str(xml_paths[i]),
            }
            for i in range(len(scale_factors_list))
        ]
        ladder_json_path = rest2_dir / "ladder.json"
        ladder_json_path.write_text(json.dumps(ladder, indent=2) + "\n", encoding="utf-8")

    return {
        "mode": mode,
        "system_xml_paths": xml_paths,
        "scale_factors": scale_factors_list,
        "effective_temperatures_k": t_eff_list,
        "ladder_json": ladder_json_path,
    }
