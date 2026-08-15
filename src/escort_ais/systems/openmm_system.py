from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from openmm import (
    CMAPTorsionForce,
    CustomCVForce,
    CustomGBForce,
    CustomTorsionForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
    Platform,
    XmlSerializer,
    unit,
)
from openmm import app
from rdkit import Chem
from rdkit.Chem import AllChem

BOLTZMANN_KJ_PER_MOL_K: float = 0.00831446261815324


DEFAULT_GB_MODEL = "GBn2"
DEFAULT_GB_RADII = "mbondi3"
#: Explicit opt-out sentinel for :func:`build_implicit_system` -- reproduces the pre-2026-07-31
#: ``AmberPrmtopFile.createSystem`` Hamiltonian.  Only for re-analysing legacy runs.
GB_RADII_AMBER_BUILTIN = "amber-builtin"
DEFAULT_WATER_MODEL = "OPC"
DEFAULT_SALT_CONCENTRATION_MOLAR = 0.15
_GB_MODEL_MAP = {
    "HCT": app.HCT,
    "OBC1": app.OBC1,
    "OBC2": app.OBC2,
    "GBn": app.GBn,
    "GBn2": app.GBn2,
}


def build_implicit_system(
    prmtop_path,
    gb_model: str = DEFAULT_GB_MODEL,
    gb_radii: str | None = DEFAULT_GB_RADII,
    inpcrd_path=None,
    remove_cm_motion: bool = True,
):
    """THE implicit-solvent base System builder.  GBn2 + mbondi3 via parmed, by default.

    Two routes to a GB System exist and they are NOT interchangeable::

        parmed.Structure.createSystem(...)     <- this function, the reference Hamiltonian
        app.AmberPrmtopFile.createSystem(...)  <- differs by ~16 kJ/mol on alanine

    Measured on ``ff19sb_gbn2_mbondi3`` (2026-07-31): the two agree to 0.0000 kJ/mol on every force
    EXCEPT ``CustomGBForce``, where they differ by **16.05 kJ/mol** -- with *identical* per-particle
    GB parameters and identical radii (max |dR| = 0 over all 22 atoms).  So the discrepancy is NOT
    the radii, contrary to the older comments at the call sites; ``gb_radii`` is in practice a
    *branch selector*, and the branch is what matters.  Under REST2 that offset becomes several kT
    of spurious work, so hot walkers, REMD ladders and AIS legs must all sit on the same branch.

    ``changeRadii`` itself is a no-op wherever tleap already wrote mbondi3 (both alanine prmtops:
    max |dR| = 0), but is load-bearing for the sage/openff macrocycles, whose prmtops carry NO radii
    at all (RGD: stored radius 0.000 A -> 1.439 A, max |dR| = 1.70 A).  Applying it unconditionally
    is therefore safe for the former and required for the latter.

    :param gb_radii: radius set for parmed ``changeRadii``; defaults to mbondi3, the set GBn2 was
        parameterised against.  Pass :data:`GB_RADII_AMBER_BUILTIN` to take the legacy
        ``AmberPrmtopFile`` branch instead -- only to re-analyse runs made before this was enforced.
    """
    if gb_model not in _GB_MODEL_MAP:
        raise ValueError(f"Unsupported GB model '{gb_model}'. Valid models: {sorted(_GB_MODEL_MAP)}")
    gb_obj = _GB_MODEL_MAP[gb_model]

    if gb_radii is None:
        gb_radii = DEFAULT_GB_RADII
    if str(gb_radii) == GB_RADII_AMBER_BUILTIN:
        return app.AmberPrmtopFile(str(prmtop_path)).createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=gb_obj,
            removeCMMotion=remove_cm_motion,
        )

    import parmed as pmd  # noqa: PLC0415
    from parmed.tools import changeRadii  # noqa: PLC0415

    st = (pmd.load_file(str(prmtop_path), xyz=str(inpcrd_path)) if inpcrd_path is not None
          else pmd.load_file(str(prmtop_path)))
    changeRadii(st, str(gb_radii)).execute()
    return st.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=gb_obj,
        removeCMMotion=remove_cm_motion,
    )


def _assert_tleap_available() -> None:
    if shutil.which("tleap") is None:
        raise RuntimeError(
            "AmberTools tleap was not found in PATH. Install ambertools and activate the project conda environment."
        )


def build_or_load_amber_topology(
    output_dir: Path,
    gb_model: str = DEFAULT_GB_MODEL,
) -> dict[str, Path]:
    """Create AMBER files for ACE-ALA-NME using tleap, or reuse existing files if present."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prmtop_path = output_dir / "ace_ala_nme.prmtop"
    inpcrd_path = output_dir / "ace_ala_nme.inpcrd"
    leap_pdb_path = output_dir / "ace_ala_nme_leap.pdb"
    leap_in_path = output_dir / "tleap.in"
    leap_log_path = output_dir / "tleap.log"
    config_path = output_dir / "config.json"

    if not (prmtop_path.exists() and inpcrd_path.exists() and leap_pdb_path.exists()):
        _assert_tleap_available()

        leap_script = "\n".join(
            [
                "source leaprc.protein.ff19SB",
                "mol = sequence { ACE ALA NME }",
                f"saveAmberParm mol {prmtop_path} {inpcrd_path}",
                f"savePdb mol {leap_pdb_path}",
                "quit",
            ]
        )
        leap_in_path.write_text(leap_script + "\n", encoding="utf-8")

        result = subprocess.run(
            ["tleap", "-f", str(leap_in_path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(output_dir),
        )
        leap_log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"tleap failed with exit code {result.returncode}. See {leap_log_path} for details."
            )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "forcefield": "ff19SB",
                "molecule": "ACE-ALA-NME",
                "implicit_solvent_gb_model": gb_model,
                "constraints": "HBonds",
                "nonbonded_method": "NoCutoff",
                "generator": "AmberTools tleap",
            },
            f,
            indent=2,
        )

    return {
        "prmtop": prmtop_path,
        "inpcrd": inpcrd_path,
        "leap_pdb": leap_pdb_path,
        "tleap_in": leap_in_path,
        "tleap_log": leap_log_path,
        "config_json": config_path,
    }


def _count_water_residues(pdb_path: Path) -> int:
    """Count WAT/HOH residues in a tleap-generated PDB."""
    count = 0
    with pdb_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname in ("WAT", "HOH"):
                    count += 1
    # Each water has 3 atoms (OPC has 4 — but residue count is what we need)
    # tleap PDB has one line per atom; count unique residue numbers instead
    return count  # caller will divide by atoms-per-water


def _num_water_residues(pdb_path: Path) -> int:
    """Return the number of unique water residue entries in a tleap PDB."""
    seen: set[tuple[str, str, int]] = set()
    with pdb_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname in ("WAT", "HOH"):
                    chain = line[21:22]
                    resseq = int(line[22:26])
                    seen.add((resname, chain, resseq))
    return len(seen)


def _num_nacl_pairs(n_water: int, salt_molar: float) -> int:
    """Compute NaCl ion pairs needed for a given molarity using water-count ratio.

    n_pairs = round(c_M * n_water / 55.5)
    """
    return round(salt_molar * n_water / 55.5)


def build_or_load_amber_explicit_topology(
    output_dir: Path,
    water_model: str = DEFAULT_WATER_MODEL,
    box_padding_angstrom: float = 12.0,
    salt_concentration_molar: float = DEFAULT_SALT_CONCENTRATION_MOLAR,
) -> dict[str, Path]:
    """Create AMBER files for ACE-ALA-NME in an explicit solvent box, or reuse existing files.

    Uses a two-pass tleap strategy to add NaCl at the requested molarity:
    pass 1 solvates to count waters; pass 2 adds the computed number of ion pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prmtop_path = output_dir / "ace_ala_nme_explicit.prmtop"
    inpcrd_path = output_dir / "ace_ala_nme_explicit.inpcrd"
    leap_pdb_path = output_dir / "ace_ala_nme_explicit_leap.pdb"
    leap_in_path = output_dir / "tleap_explicit.in"
    leap_log_path = output_dir / "tleap_explicit.log"
    config_path = output_dir / "config.json"

    water_source_map = {
        "OPC": ("leaprc.water.opc", "OPCBOX"),
    }
    if water_model not in water_source_map:
        raise ValueError(f"Unsupported explicit water model '{water_model}'. Valid models: {sorted(water_source_map)}")

    water_source, water_box = water_source_map[water_model]

    # Check cache: rebuild if files missing or stored salt differs from request
    need_build = not (prmtop_path.exists() and inpcrd_path.exists() and leap_pdb_path.exists())
    if not need_build and config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        if abs(stored.get("salt_concentration_molar", 0.0) - salt_concentration_molar) > 1e-6:
            need_build = True

    if need_build:
        _assert_tleap_available()

        # Pass 1: solvate only, save a temporary PDB to count waters
        tmp_pdb = output_dir / "_count_waters.pdb"
        tmp_in = output_dir / "_count_waters.in"
        tmp_log = output_dir / "_count_waters.log"
        pass1_script = "\n".join(
            [
                "source leaprc.protein.ff19SB",
                f"source {water_source}",
                "mol = sequence { ACE ALA NME }",
                f"solvateBox mol {water_box} {box_padding_angstrom:.3f} iso",
                f"savePdb mol {tmp_pdb}",
                "quit",
            ]
        )
        tmp_in.write_text(pass1_script + "\n", encoding="utf-8")
        r1 = subprocess.run(
            ["tleap", "-f", str(tmp_in)],
            capture_output=True, text=True, check=False, cwd=str(output_dir),
        )
        tmp_log.write_text((r1.stdout or "") + "\n" + (r1.stderr or ""), encoding="utf-8")
        if r1.returncode != 0:
            raise RuntimeError(
                f"tleap water-count pass failed (exit {r1.returncode}). See {tmp_log} for details."
            )

        n_water = _num_water_residues(tmp_pdb)
        n_pairs = _num_nacl_pairs(n_water, salt_concentration_molar)

        # Pass 2: solvate + add ions + save final topology
        ion_lines = (
            [f"addIonsRand mol Na+ {n_pairs} Cl- {n_pairs}"] if n_pairs > 0 else []
        )
        pass2_script = "\n".join(
            [
                "source leaprc.protein.ff19SB",
                f"source {water_source}",
                "mol = sequence { ACE ALA NME }",
                f"solvateBox mol {water_box} {box_padding_angstrom:.3f} iso",
                *ion_lines,
                f"saveAmberParm mol {prmtop_path} {inpcrd_path}",
                f"savePdb mol {leap_pdb_path}",
                "quit",
            ]
        )
        leap_in_path.write_text(pass2_script + "\n", encoding="utf-8")
        r2 = subprocess.run(
            ["tleap", "-f", str(leap_in_path)],
            capture_output=True, text=True, check=False, cwd=str(output_dir),
        )
        leap_log_path.write_text((r2.stdout or "") + "\n" + (r2.stderr or ""), encoding="utf-8")
        if r2.returncode != 0:
            raise RuntimeError(
                f"tleap explicit-solvent build failed (exit {r2.returncode}). See {leap_log_path} for details."
            )

        # Clean up pass-1 scratch files
        for f in (tmp_pdb, tmp_in, tmp_log):
            f.unlink(missing_ok=True)
    else:
        # Read back ion count from stored config so we can report it accurately
        stored = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        n_pairs = stored.get("nacl_ion_pairs", None)

    # Recompute n_pairs for config if we skipped the build
    if not need_build and n_pairs is None:
        # Estimate from existing PDB
        n_water = _num_water_residues(leap_pdb_path)
        n_pairs = _num_nacl_pairs(n_water, salt_concentration_molar)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "forcefield": "ff19SB",
                "molecule": "ACE-ALA-NME",
                "solvent_model": water_model,
                "box_padding_angstrom": box_padding_angstrom,
                "salt_concentration_molar": salt_concentration_molar,
                "nacl_ion_pairs": n_pairs,
                "constraints": "HBonds",
                "nonbonded_method": "PME",
                "generator": "AmberTools tleap",
            },
            f,
            indent=2,
        )

    return {
        "prmtop": prmtop_path,
        "inpcrd": inpcrd_path,
        "leap_pdb": leap_pdb_path,
        "tleap_in": leap_in_path,
        "tleap_log": leap_log_path,
        "config_json": config_path,
    }


def build_openmm_u0_context(
    prmtop_path: Path,
    inpcrd_path: Path,
    gb_model: str = DEFAULT_GB_MODEL,
    temperature_kelvin: float = 300.0,
    friction_per_ps: float = 1.0,
    timestep_ps: float = 0.002,
    platform_name: str = "CPU",
    gb_radii: str | None = DEFAULT_GB_RADII,
):
    """A ready-to-run implicit-solvent ``Simulation`` on THE CANONICAL HAMILTONIAN.

    Builds its System through :func:`build_implicit_system` -- GBn2 + mbondi3 via parmed -- the same
    route every REST2/AIS workflow uses. ``AmberPrmtopFile`` is used only for the ``Topology`` that
    ``Simulation`` needs, never to create the System.

    UNTIL 2026-08-11 THIS CALLED ``AmberPrmtopFile.createSystem`` instead, which is the legacy branch
    the module documents as differing by ~16 kJ/mol on alanine (the whole difference sits in
    ``CustomGBForce`` at bit-identical radii). A context built here was therefore NOT comparable with
    anything the production code produced. The function had no call sites when it was corrected, so
    nothing in the repository consumed the wrong Hamiltonian -- but it was a live trap for the next
    caller.

    :param gb_radii: pass :data:`GB_RADII_AMBER_BUILTIN` to take the legacy branch DELIBERATELY, for
        re-analysing pre-enforcement runs. It is never selected implicitly.
    """
    if gb_model not in _GB_MODEL_MAP:
        raise ValueError(f"Unsupported GB model '{gb_model}'. Valid models: {sorted(_GB_MODEL_MAP)}")

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))

    system = build_implicit_system(prmtop_path, gb_model=gb_model, gb_radii=gb_radii,
                                   inpcrd_path=inpcrd_path)

    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_ps * unit.picoseconds,
    )

    platform = Platform.getPlatformByName(platform_name)
    platform_properties = {}
    if platform_name in {"CUDA", "OpenCL"}:
        platform_properties["Precision"] = "mixed"
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, platform_properties)
    simulation.context.setPositions(inpcrd.positions)
    return simulation


def build_openmm_explicit_context(
    prmtop_path: Path,
    inpcrd_path: Path,
    temperature_kelvin: float = 300.0,
    pressure_bar: float = 1.0,
    friction_per_ps: float = 1.0,
    timestep_ps: float = 0.002,
    nonbonded_cutoff_nm: float = 1.0,
    platform_name: str = "CPU",
):
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))

    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    system.addForce(MonteCarloBarostat(pressure_bar * unit.bar, temperature_kelvin * unit.kelvin, 50))

    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_ps * unit.picoseconds,
    )

    platform = Platform.getPlatformByName(platform_name)
    platform_properties = {}
    if platform_name in {"CUDA", "OpenCL"}:
        platform_properties["Precision"] = "mixed"

    simulation = app.Simulation(prmtop.topology, system, integrator, platform, platform_properties)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    return simulation


def available_platform_names() -> list[str]:
    return [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]


def pick_platform_name(preferred: str | None = None) -> str:
    available = set(available_platform_names())
    if preferred:
        if preferred not in available:
            raise ValueError(f"Requested platform '{preferred}' is not available. Available: {sorted(available)}")
        return preferred

    for candidate in ["CUDA", "OpenCL", "CPU", "Reference"]:
        if candidate in available:
            return candidate
    raise RuntimeError("No OpenMM platform is available.")


def map_rdkit_atoms_to_openmm_topology(rdkit_mol: Chem.Mol, leap_pdb_path: Path) -> np.ndarray:
    """Return mapping from OpenMM/tleap atom index to RDKit atom index."""
    template = Chem.MolFromPDBFile(str(leap_pdb_path), removeHs=False, sanitize=False)
    if template is None:
        raise RuntimeError(f"Failed to parse PDB for atom mapping: {leap_pdb_path}")

    if template.GetNumAtoms() != rdkit_mol.GetNumAtoms():
        raise ValueError(
            "Atom count mismatch between ETKDG molecule and tleap topology template: "
            f"{rdkit_mol.GetNumAtoms()} vs {template.GetNumAtoms()}"
        )

    # Make connectivity matching more reliable when reading from PDB.
    try:
        template = AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(rdkit_mol), template)
    except Exception:
        # Fallback to raw PDB-inferred connectivity if assignment fails.
        pass

    match = rdkit_mol.GetSubstructMatch(template)
    if len(match) == rdkit_mol.GetNumAtoms():
        # match[template_idx] = rdkit_idx
        return np.asarray(match, dtype=np.int64)

    reverse_match = template.GetSubstructMatch(rdkit_mol)
    if len(reverse_match) == rdkit_mol.GetNumAtoms():
        # reverse_match[rdkit_idx] = template_idx -> invert
        inverse = np.empty(len(reverse_match), dtype=np.int64)
        for rdkit_idx, template_idx in enumerate(reverse_match):
            inverse[int(template_idx)] = int(rdkit_idx)
        return inverse

    raise RuntimeError(
        "Failed to match atom ordering between ETKDG molecule and tleap topology template."
    )


def heavy_atom_indices(topology: app.Topology) -> np.ndarray:
    indices = [atom.index for atom in topology.atoms() if atom.element is not None and atom.element.symbol != "H"]
    return np.asarray(indices, dtype=np.int64)


def reorder_positions_to_openmm(coords_angstrom: np.ndarray, openmm_to_rdkit: np.ndarray) -> np.ndarray:
    """Reorder RDKit-order coordinates to OpenMM topology order and convert A -> nm."""
    coords_angstrom = np.asarray(coords_angstrom, dtype=float)
    reordered_angstrom = coords_angstrom[openmm_to_rdkit]
    return reordered_angstrom * 0.1


# ---------------------------------------------------------------------------
# REST2 Hamiltonian scaling utilities
# ---------------------------------------------------------------------------

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


def _scale_customgb_force(force: CustomGBForce, solute_atom_indices: set[int], scale_factor: float) -> None:
    if len(solute_atom_indices) != force.getNumParticles():
        raise RuntimeError("Implicit REST2 scaling for CustomGBForce currently assumes all "
                           "particles are solute atoms.")
    for energy_index in range(force.getNumEnergyTerms()):
        energy_expression, computation_type = force.getEnergyTermParameters(energy_index)
        if ";" in energy_expression:
            leading, trailing = energy_expression.split(";", 1)
            scaled_expression = f"{scale_factor:.17g}*({leading});{trailing}"
        else:
            scaled_expression = f"{scale_factor:.17g}*({energy_expression})"
        force.setEnergyTermParameters(energy_index, scaled_expression, computation_type)


# Name of the OpenMM global parameter injected into CustomGBForce energy
# expressions by Rest2ContextScaler.  Must be unique; avoid clashing with
# any parameter name already present in GBn2 / HCT expressions.
_REST2_GB_SCALE_PARAM: str = "rest2_scale_gb"


def _inject_gb_scale_param(gb: CustomGBForce) -> None:
    """Idempotently wrap every CustomGBForce energy term with the ``rest2_scale_gb`` global parameter,
    so the GB scale can be set at runtime via ``context.setParameter`` (no updateParametersInContext)."""
    existing = {gb.getGlobalParameterName(i) for i in range(gb.getNumGlobalParameters())}
    if _REST2_GB_SCALE_PARAM in existing:
        return  # already prepared (idempotent)
    gb.addGlobalParameter(_REST2_GB_SCALE_PARAM, 1.0)
    for ei in range(gb.getNumEnergyTerms()):
        expr, ctype = gb.getEnergyTermParameters(ei)
        if ";" in expr:
            leading, trailing = expr.split(";", 1)
            new_expr = f"{_REST2_GB_SCALE_PARAM}*({leading});{trailing}"
        else:
            new_expr = f"{_REST2_GB_SCALE_PARAM}*({expr})"
        gb.setEnergyTermParameters(ei, new_expr, ctype)


class Rest2ContextScaler:
    """Apply REST2 Hamiltonian scaling to an OpenMM System's forces in place.

    Instead of building N+1 separate Systems/Contexts (one per λ window), this
    class allows a **single** Context to be reused across all λ windows.
    At construction, it:

    * Caches the unscaled (s = 1) parameters for NonbondedForce,
      PeriodicTorsionForce, and CMAPTorsionForce so that each call to
      :meth:`apply` can recompute the scaled values from the base.
    * **Modifies** the CustomGBForce energy expressions in the supplied System
      to reference a new global parameter ``rest2_scale_gb`` (default 1.0).
      This must happen **before** the OpenMM Context is created, so the
      compiled kernels already reference the parameter.

    At each lambda window, :meth:`apply` updates all four force types in a
    single Python call:

    * NonbondedForce, PeriodicTorsionForce, CMAPTorsionForce: call the
      respective ``setXxxParameters`` then ``updateParametersInContext``.
    * CustomGBForce: ``context.setParameter("rest2_scale_gb", s)``.  No
      recompilation occurs.

    **Why a global parameter for GB rather than per-particle charge scaling?**
    GBn2 has three energy terms.  Terms 0 and 2 are proportional to charge
    products and would scale correctly with ``charge × √s``.  However, term 1
    is a non-polar/dispersion correction with no charge dependence; it also
    needs to be scaled by *s* in REST2 but is unaffected by charge changes.
    A global parameter that multiplies the entire energy expression scales all
    three terms uniformly and is updatable without Context reinitialisation.

    Parameters
    ----------
    system : openmm.System
        The **unscaled** (s = 1) System whose force objects will be mutated.
        **This object is modified in place** (CustomGBForce expressions are
        rewritten).  Create the OpenMM Context *after* constructing this scaler.
    solute_atom_indices : array-like of int
        Zero-based indices of solute atoms.
    exclude_central_bonds : iterable of {i, j} pairs, optional
        Central bonds (atom-index pairs) whose PeriodicTorsionForce terms are NOT
        scaled, even if all four atoms are solute.  Used to keep the omega
        (peptide-bond) torsion of ordinary (non-proline-like) amides physical so the
        hot ensemble does not artificially induce cis/trans isomerization.  Only the
        torsion scaling is affected; nonbonded/GB scaling is unchanged.
    """

    def __init__(self, system, solute_atom_indices, exclude_central_bonds=None):
        self._solute: set[int] = {int(i) for i in solute_atom_indices}
        self._exclude_bonds: set[frozenset] = (
            {frozenset((int(a), int(b))) for a, b in exclude_central_bonds}
            if exclude_central_bonds is not None else set())

        # Force object references (same objects the Context will hold)
        self._nb: NonbondedForce | None = None
        self._torsion: PeriodicTorsionForce | None = None
        self._cmap: CMAPTorsionForce | None = None
        self._gb: CustomGBForce | None = None

        # Cached base (s=1) parameters for forces updated via updateParametersInContext
        self._nb_particles: list = []    # [(charge, sigma, epsilon, is_solute), ...]
        self._nb_exceptions: list = []   # [(i, j, chargeProd, sigma, epsilon, n_solute), ...]
        self._torsion_list: list = []    # [(i,j,k,l, per, phase, k, is_solute), ...]
        self._cmap_solute_maps: dict = {}  # {map_index: (size, [base_energies])}
        # GB is handled via global-param injection; no per-particle cache needed.

        for fi in range(system.getNumForces()):
            force = system.getForce(fi)
            if isinstance(force, NonbondedForce):
                self._nb = force
                self._cache_nonbonded()
            elif isinstance(force, PeriodicTorsionForce):
                self._torsion = force
                self._cache_torsion()
            elif isinstance(force, CMAPTorsionForce):
                self._cmap = force
                self._cache_cmap()
            elif isinstance(force, CustomGBForce):
                self._gb = force
                self._prepare_customgb()  # rewrites energy expressions — must be before Context creation

    # ------------------------------------------------------------------
    # Cache / prepare helpers (called once at construction)
    # ------------------------------------------------------------------

    def _cache_nonbonded(self) -> None:
        f = self._nb
        for i in range(f.getNumParticles()):
            q, sig, eps = f.getParticleParameters(i)
            self._nb_particles.append((q, sig, eps, i in self._solute))
        for ei in range(f.getNumExceptions()):
            ai, aj, cp, sig, eps = f.getExceptionParameters(ei)
            n = int(ai in self._solute) + int(aj in self._solute)
            self._nb_exceptions.append((ai, aj, cp, sig, eps, n))

    def _cache_torsion(self) -> None:
        f = self._torsion
        for ti in range(f.getNumTorsions()):
            i, j, k, l, per, phase, kv = f.getTorsionParameters(ti)
            is_sol = all(a in self._solute for a in (i, j, k, l))
            # excluded omega: solute but central bond {j,k} is a kept-physical amide
            eligible = is_sol and frozenset((int(j), int(k))) not in self._exclude_bonds
            self._torsion_list.append((i, j, k, l, per, phase, kv, eligible))

    def _cache_cmap(self) -> None:
        f = self._cmap
        solute_maps: set[int] = set()
        non_solute_maps: set[int] = set()
        for ti in range(f.getNumTorsions()):
            mi, a1, a2, a3, a4, b1, b2, b3, b4 = f.getTorsionParameters(ti)
            atoms = (a1, a2, a3, a4, b1, b2, b3, b4)
            (solute_maps if all(a in self._solute for a in atoms) else non_solute_maps).add(mi)
        shared = solute_maps & non_solute_maps
        if shared:
            raise RuntimeError(
                "Rest2ContextScaler: CMAP map(s) shared between solute and "
                "non-solute torsions — cannot scale selectively."
            )
        for mi in solute_maps:
            size, energies = f.getMapParameters(mi)
            self._cmap_solute_maps[mi] = (size, list(energies))

    def _prepare_customgb(self) -> None:
        """Inject ``rest2_scale_gb`` global parameter into all GB energy terms.

        Modifies the CustomGBForce energy expressions in-place so that the
        Context compiled from this System will reference the parameter.
        Safe to call multiple times (idempotent).
        """
        if len(self._solute) != self._gb.getNumParticles():
            raise RuntimeError(
                "Rest2ContextScaler: CustomGBForce scaling assumes all particles "
                "are solute atoms (implicit solvent, protein-only system)."
            )
        _inject_gb_scale_param(self._gb)

    # ------------------------------------------------------------------
    # apply — call this per lambda window
    # ------------------------------------------------------------------

    def apply(self, scale: float, context=None) -> None:
        """Set all scalable forces to *scale* and push to *context* if given.

        Parameters
        ----------
        scale : float
            REST2 scale factor *s* (1.0 = physical/unscaled).
        context : openmm.Context, optional
            If provided, changes are pushed into the live Context.
            For NonbondedForce, PeriodicTorsionForce, and CMAPTorsionForce this
            calls ``updateParametersInContext(context)``.
            For CustomGBForce this calls
            ``context.setParameter("rest2_scale_gb", scale)``.
        """
        sqrt_s = math.sqrt(scale)

        if self._nb is not None:
            for i, (q, sig, eps, is_sol) in enumerate(self._nb_particles):
                if is_sol:
                    self._nb.setParticleParameters(i, q * sqrt_s, sig, eps * scale)
            for ei, (ai, aj, cp, sig, eps, n) in enumerate(self._nb_exceptions):
                if n == 2:
                    self._nb.setExceptionParameters(ei, ai, aj, cp * scale, sig, eps * scale)
                elif n == 1:
                    self._nb.setExceptionParameters(ei, ai, aj, cp * sqrt_s, sig, eps * sqrt_s)
            if context is not None:
                self._nb.updateParametersInContext(context)

        if self._torsion is not None:
            for ti, (i, j, k, l, per, phase, kv, is_sol) in enumerate(self._torsion_list):
                if is_sol:
                    self._torsion.setTorsionParameters(ti, i, j, k, l, per, phase, kv * scale)
            if context is not None:
                self._torsion.updateParametersInContext(context)

        if self._cmap is not None:
            for mi, (size, energies) in self._cmap_solute_maps.items():
                self._cmap.setMapParameters(mi, size, [v * scale for v in energies])
            if context is not None:
                self._cmap.updateParametersInContext(context)

        # CustomGBForce: update via global parameter — no updateParametersInContext needed
        if self._gb is not None and context is not None:
            context.setParameter(_REST2_GB_SCALE_PARAM, scale)


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
            _scale_customgb_force(force, solute_set, scale_factor)
    return system


# ---------------------------------------------------------------------------
# GPU-side (lambda-driven) REST2 scaling
# ---------------------------------------------------------------------------
# The scaled Hamiltonian U_s = s*U_solute + sqrt(s)*U_solute-solvent + U_solvent
# is driven entirely by OpenMM *global parameters* so that changing s at every
# AIS switching window costs only a handful of context.setParameter() calls
# (GPU-side, O(us)) instead of per-particle updateParametersInContext() reloads.
#
# NonbondedForce: parameter offsets scale charge by sqrt(s) and epsilon by s.
#   charge_i(s)  = q_i  + (sqrt(s)-1)*q_i      -> global rest2_sqrt_s_minus_1
#   epsilon_i(s) = e_i  + (s-1)*e_i            -> global rest2_s_minus_1
#   solute-solute exception chargeProd/eps *= s (rest2_s_minus_1)
#   solute-solvent exception chargeProd/eps *= sqrt(s) (rest2_sqrt_s_minus_1)
# PeriodicTorsion: scaled solute torsions moved to a CustomTorsionForce with a
#   global lambda_rest (=s); their k in the original force is zeroed.
# CustomGBForce: global rest2_scale_gb (=s), same as Rest2ContextScaler.
_REST2_Q_PARAM: str = "rest2_sqrt_s_minus_1"   # holds sqrt(s)-1
_REST2_S_PARAM: str = "rest2_s_minus_1"        # holds s-1
_REST2_TORS_PARAM: str = "lambda_rest"         # holds s
_REST2_CMAP_PARAM: str = "lambda_cmap"         # holds s (scales the solute CMAP energy GPU-side)


def build_rest2_lambda_system(base_system, solute_atom_indices: np.ndarray, exclude_central_bonds=None):
    """Return a deep copy of *base_system* whose REST2 solute scaling is driven by OpenMM global
    parameters (no per-window ``updateParametersInContext``).  Drive it with :class:`Rest2LambdaScaler`.

    At the default global values (all offsets 0, lambda_rest=1, rest2_scale_gb=1, lambda_cmap=1) the returned
    system is identical to the physical (s=1) system.  Requires all particles to be solute (implicit solvent).
    An all-solute ``CMAPTorsionForce`` is wrapped in ``CustomCVForce(lambda_cmap * cmap_e)`` so the ff19SB
    backbone CMAP scales GPU-side via a single global parameter (no per-window setMapParameters spline rebuild);
    a CMAP torsion spanning non-solute atoms still raises so the caller can fall back to
    ``build_rest2_scaled_system`` + :class:`Rest2ContextScaler`.
    """
    system = _clone_system(base_system)
    solute_set = {int(i) for i in solute_atom_indices}
    exclude = ({frozenset((int(a), int(b))) for a, b in exclude_central_bonds}
               if exclude_central_bonds is not None else set())

    nb = tors = gb = None
    has_cmap = False
    for fi in range(system.getNumForces()):
        f = system.getForce(fi)
        if isinstance(f, CMAPTorsionForce):
            has_cmap = True
        elif isinstance(f, NonbondedForce):
            nb = f
        elif isinstance(f, PeriodicTorsionForce):
            tors = f
        elif isinstance(f, CustomGBForce):
            gb = f

    # --- NonbondedForce: parameter offsets (charge ~ sqrt(s), eps ~ s) ---
    if nb is not None:
        nb.addGlobalParameter(_REST2_Q_PARAM, 0.0)
        nb.addGlobalParameter(_REST2_S_PARAM, 0.0)
        for i in range(nb.getNumParticles()):
            if i not in solute_set:
                continue
            q, _, eps = nb.getParticleParameters(i)
            q_e = q.value_in_unit(unit.elementary_charge)
            eps_kj = eps.value_in_unit(unit.kilojoule_per_mole)
            if q_e != 0.0:
                nb.addParticleParameterOffset(_REST2_Q_PARAM, i, q_e, 0.0, 0.0)
            if eps_kj != 0.0:
                nb.addParticleParameterOffset(_REST2_S_PARAM, i, 0.0, 0.0, eps_kj)
        for ei in range(nb.getNumExceptions()):
            ai, aj, cp, _, eps = nb.getExceptionParameters(ei)
            n_sol = int(ai in solute_set) + int(aj in solute_set)
            cp_v = cp.value_in_unit(unit.elementary_charge ** 2)
            eps_v = eps.value_in_unit(unit.kilojoule_per_mole)
            if cp_v == 0.0 and eps_v == 0.0:
                continue
            if n_sol == 2:          # solute-solute: chargeProd*s, eps*s
                nb.addExceptionParameterOffset(_REST2_S_PARAM, ei, cp_v, 0.0, eps_v)
            elif n_sol == 1:        # solute-solvent: chargeProd*sqrt(s), eps*sqrt(s)
                nb.addExceptionParameterOffset(_REST2_Q_PARAM, ei, cp_v, 0.0, eps_v)

    # --- PeriodicTorsion: move scaled solute torsions to a CustomTorsionForce(lambda_rest) ---
    if tors is not None:
        ct = CustomTorsionForce("lambda_rest*k*(1 + cos(per*theta - phase))")
        ct.addGlobalParameter(_REST2_TORS_PARAM, 1.0)
        for name in ("per", "phase", "k"):
            ct.addPerTorsionParameter(name)
        for ti in range(tors.getNumTorsions()):
            i, j, k, l, per, phase, kv = tors.getTorsionParameters(ti)
            is_solute = (all(a in solute_set for a in (i, j, k, l))
                         and frozenset((int(j), int(k))) not in exclude)
            if is_solute:
                ct.addTorsion(i, j, k, l, [float(per),
                                           phase.value_in_unit(unit.radian),
                                           kv.value_in_unit(unit.kilojoule_per_mole)])
                tors.setTorsionParameters(ti, i, j, k, l, per, phase, 0.0)  # zeroed in the periodic force
        system.addForce(ct)

    # --- CustomGBForce: global rest2_scale_gb (reuse existing convention) ---
    if gb is not None:
        if len(solute_set) != gb.getNumParticles():
            raise RuntimeError("build_rest2_lambda_system GB scaling assumes all particles are solute.")
        _inject_gb_scale_param(gb)

    # --- CMAPTorsionForce: wrap in CustomCVForce(lambda_cmap * cmap_e) so the ff19SB backbone CMAP scales
    #     GPU-side via a single global parameter (no per-window setMapParameters spline rebuild). Requires all
    #     CMAP-torsion atoms to be solute (implicit REST2), matching Rest2ContextScaler's solute-map scaling. ---
    if has_cmap:
        cmap_idx = next(fi for fi in range(system.getNumForces())
                        if isinstance(system.getForce(fi), CMAPTorsionForce))
        cmap = system.getForce(cmap_idx)
        for ti in range(cmap.getNumTorsions()):
            _, a1, a2, a3, a4, b1, b2, b3, b4 = cmap.getTorsionParameters(ti)
            if not all(a in solute_set for a in (a1, a2, a3, a4, b1, b2, b3, b4)):
                raise RuntimeError("build_rest2_lambda_system: CMAP torsion spans non-solute atoms; "
                                   "wholesale lambda_cmap scaling would be wrong — use Rest2ContextScaler.")
        cmap_copy = XmlSerializer.deserialize(XmlSerializer.serialize(cmap))  # independent copy for the CV
        system.removeForce(cmap_idx)
        cv = CustomCVForce(f"{_REST2_CMAP_PARAM} * cmap_e")
        cv.addGlobalParameter(_REST2_CMAP_PARAM, 1.0)
        cv.addCollectiveVariable("cmap_e", cmap_copy)
        system.addForce(cv)
    return system


class Rest2LambdaScaler:
    """Lean per-window scaler for a :func:`build_rest2_lambda_system` system.  ``apply(s, context)`` sets
    only OpenMM global parameters (GPU-side, no ``updateParametersInContext``)."""

    def __init__(self, system):
        self._params = set()
        for fi in range(system.getNumForces()):
            f = system.getForce(fi)
            if hasattr(f, "getNumGlobalParameters"):
                for gp in range(f.getNumGlobalParameters()):
                    self._params.add(f.getGlobalParameterName(gp))

    def apply(self, scale: float, context=None) -> None:
        s = float(scale)
        vals = {_REST2_Q_PARAM: math.sqrt(s) - 1.0, _REST2_S_PARAM: s - 1.0,
                _REST2_TORS_PARAM: s, _REST2_GB_SCALE_PARAM: s, _REST2_CMAP_PARAM: s}
        if context is not None:
            for name, v in vals.items():
                if name in self._params:
                    context.setParameter(name, v)


