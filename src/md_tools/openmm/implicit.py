"""Building the implicit-solvent System, through ParmEd, exactly.

The Hamiltonian-defining path is::

    st = parmed.load_file(prmtop, xyz=inpcrd)
    parmed.tools.changeRadii(st, "mbondi3").execute()
    system = st.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=<constraints.type, honoured>,
        implicitSolvent=app.GBn2,
        removeCMMotion=True,
    )

## The ~16 kJ/mol difference between construction routes is the NONPOLAR term

Three routes can build a GBn2 System, and they do not agree. Measured on this machine for
ACE-ALA-NME with ff14SB and mbondi3 radii, at identical coordinates:

    route                                              TOTAL (kJ/mol)
    parmed.Structure.createSystem(useSASA=False)            -119.7978   <- what this module builds
    parmed.Structure.createSystem(useSASA=True)             -103.7448
    app.ForceField("amber14/protein.ff14SB.xml",
                   "implicit/gbn2.xml")                     -103.7465

Every force agrees to ~0 kJ/mol except `CustomGBForce`, and the GB radii and screening factors are
identical for all 22 atoms. The cause is not a radius and not an opaque "construction branch": the
ParmEd System has **two** GB energy terms and the OpenMM one has **three**. The extra term is

    28.3919551*(radius+0.14)^2*(radius/B)^6

the ACE surface-area **nonpolar** (cavity + dispersion) contribution. Set `useSASA=True` and ParmEd
agrees with pure OpenMM to **0.0017 kJ/mol**.

So this is a modelling choice, not an implementation accident:

* Amber's `igb=8` with `gbsa=0` -- no nonpolar term -- is what this module builds, and matches the
  context GBn2's parameters were fit in (GB-Neck2 was fit to reproduce PB *polar* solvation).
* OpenMM's `implicit/gbn2.xml` includes ACE by default, which is why the two look inconsistent.

It is recorded explicitly in `forcefield.json` as `implicit_solvent.nonpolar_sasa` rather than
inherited from a library default nobody chose. Two consequences worth stating:

* 16 kJ/mol is roughly 6 kT at 300 K -- not a rounding difference in any free-energy comparison.
* REST2 scales `CustomGBForce` by `s`, so a nonpolar term placed inside that force is scaled with
  the solute Hamiltonian too. Whatever the choice, every replica in a ladder must share it.

## Why `changeRadii` runs unconditionally

For a tleap-built protein topology it is a no-op: tleap has already written mbondi3 and the measured
maximum radius change is 0.0000 A. For an OpenFF/Sage topology it is load-bearing -- those prmtops
carry no GB radii at all, and without it the radii are zero. Applying it always is therefore safe
for the first case and required for the second, which is cheaper than deciding per route and
getting the decision wrong.

## What this module does not do

It does not run dynamics, and the Amber files it writes are construction intermediates and
provenance for OpenMM. There is no Amber execution engine here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

__all__ = [
    "AMBER_TOPOLOGY_NAME",
    "build_amber_topology_via_tleap",
    "build_implicit_bundle_inputs",
    "AMBER_COORDINATE_NAME",
    "build_implicit_system",
    "implicit_provenance",
    "radii_of",
    "write_amber_files_from_openmm",
]

#: One canonical name for each Amber artefact. `rst7` rather than `inpcrd` so there is exactly one
#: coordinate authority in the bundle; ParmEd writes either from the same Structure.
AMBER_TOPOLOGY_NAME = "system.prmtop"
AMBER_COORDINATE_NAME = "system.rst7"

#: The OpenMM implicit-solvent objects, by canonical name.
_GB_MODELS = {
    "HCT": "HCT",
    "OBC1": "OBC1",
    "OBC2": "OBC2",
    "GBn": "GBn",
    "GBn2": "GBn2",
}


def _gb_object(model: str):
    from openmm import app

    if model not in _GB_MODELS:
        raise ValueError(f"unsupported implicit model {model!r}; known: {sorted(_GB_MODELS)}")
    return getattr(app, _GB_MODELS[model])


def radii_of(structure) -> list:
    """Per-atom GB radii in angstrom, for checking that `changeRadii` did what it claims."""
    return [float(atom.solvent_radius) for atom in structure.atoms]


def _implicit_hmr_record(system, structure, scope: str, target: Optional[float],
                         mass_before: float) -> dict:
    """What the repartitioning actually did, measured on the built System.

    Reported rather than asserted: the count comes from comparing each particle's mass against the
    topology's element, so a record claiming 12 repartitioned hydrogens means twelve were found to
    have moved, not that twelve were asked for.
    """
    from openmm import unit as _u

    masses = [system.getParticleMass(i).value_in_unit(_u.dalton)
              for i in range(system.getNumParticles())]
    total_after = sum(masses)
    n_hydrogens = sum(1 for a in structure.atoms if a.atomic_number == 1)
    moved = sum(1 for a, m in zip(structure.atoms, masses)
                if a.atomic_number == 1 and abs(m - a.mass) > 1e-9)
    heaviest_donor_drop = min(
        (m for a, m in zip(structure.atoms, masses) if a.atomic_number != 1), default=None)
    record = {
        "scope": scope,
        "target_hydrogen_mass_amu": (float(target) if target is not None else None),
        "n_hydrogens": n_hydrogens,
        "n_hydrogens_repartitioned": moved,
        "total_mass_amu_before": round(float(mass_before), 6),
        "total_mass_amu_after": round(float(total_after), 6),
        "lightest_heavy_atom_amu": (round(float(heaviest_donor_drop), 6)
                                    if heaviest_donor_drop is not None else None),
    }
    if scope in ("solute", "all"):
        # Under implicit solvent there is no solvent to exclude, so the two scopes coincide.
        record["scope_note"] = ("implicit solvent has no solvent atoms, so 'solute' and 'all' "
                                "select the same particles")
        if abs(total_after - mass_before) > 1e-6:
            raise RuntimeError(
                f"hydrogen mass repartitioning changed the total mass from {mass_before:.6f} to "
                f"{total_after:.6f} amu. Repartitioning moves mass between bonded atoms and must "
                f"conserve it; a change means mass was created or destroyed.")
        if heaviest_donor_drop is not None and heaviest_donor_drop < 1.0:
            raise RuntimeError(
                f"repartitioning drove a heavy atom to {heaviest_donor_drop:.4f} amu, which is "
                f"lighter than hydrogen. Lower the target hydrogen mass ({target}).")
    return record


#: The (alpha, beta, gamma) ParmEd installs when an element is NOT in the GB-Neck2 fit. Its own
#: comment calls them "non-optimized values as defaults", and it pairs them with screen = 0.5.
#: Read `parmed.structure.Structure._get_gb_parameters`: fitted values exist for H, C, N, O and S
#: only. Everything else -- F, Cl, Br, I, P, Se, B, Si -- gets these.
GBN2_UNFITTED_PARAMETERS = (1.0, 0.8, 4.85)
#: The elements GB-Neck2 was actually parameterised for, as built by ParmEd.
GBN2_FITTED_ELEMENTS = (1, 6, 7, 8, 16)


def gb_parameter_coverage(system, structure) -> dict:
    """Which atoms got FITTED GB-Neck2 parameters, measured on the built `CustomGBForce`.

    Every atom receives a radius and a parameter set -- nothing is left unparameterised and nothing
    crashes -- but that is not the same as every atom being covered by the fit. Two separate gaps
    matter for a non-peptidic solute, and both are measured here rather than assumed:

    1. **Elements outside the GB-Neck2 fit.** ParmEd assigns fitted (screen, alpha, beta, gamma)
       for H, C, N, O and S. A halogen, a phosphorus in a non-nucleic residue, a selenium or a
       boron silently receives `GBN2_UNFITTED_PARAMETERS` and `screen = 0.5`. The System is built,
       the energy is finite, and no parameter in it came from the GB-Neck2 training set.

    2. **mbondi3's corrections cannot reach a ligand.** `mbondi3` is `mbondi2` plus adjustments
       keyed on residue name -- GLU/ASP/GL4/AS4 carboxylate oxygens, ARG HH/HE hydrogens -- and on
       the atom name `OXT`. A Sage-parameterised ligand is one `UNL` residue, so no adjustment can
       match and mbondi3 is exactly mbondi2 for it. Claiming "mbondi3 radii" for such a solute is
       true about the call and misleading about the radii.

    The returned record is what `forcefield.json` publishes, and it is what the experimental label
    on the implicit ligand route is based on.
    """
    from openmm import CustomGBForce

    force = None
    for index in range(system.getNumForces()):
        candidate = system.getForce(index)
        if isinstance(candidate, CustomGBForce):
            force = candidate
            break
    if force is None:
        return {"measured": False, "reason": "the built System has no CustomGBForce"}

    unfitted, elements_seen, unfitted_elements = [], set(), set()
    for index in range(force.getNumParticles()):
        parameters = list(force.getParticleParameters(index))
        atom = structure.atoms[index] if index < len(structure.atoms) else None
        number = int(getattr(atom, "atomic_number", 0) or 0)
        elements_seen.add(number)
        # [charge, offset radius, scaled offset radius, alpha, beta, gamma]
        alpha, beta, gamma = (float(v) for v in parameters[3:6])
        if all(abs(a - b) < 1e-9 for a, b in zip((alpha, beta, gamma),
                                                 GBN2_UNFITTED_PARAMETERS)):
            unfitted.append(index)
            unfitted_elements.add(number)

    residue_names = {r.name for r in structure.residues}
    mbondi3_targets = {"GLU", "ASP", "GL4", "AS4", "ARG"}
    return {
        "measured": True,
        "measured_on": "openmm.CustomGBForce per-particle parameters of the built System",
        "n_particles": force.getNumParticles(),
        "n_atoms_with_fitted_gbn2_parameters": force.getNumParticles() - len(unfitted),
        "n_atoms_with_unfitted_gbn2_parameters": len(unfitted),
        "atoms_with_unfitted_gbn2_parameters": [int(i) for i in unfitted[:64]],
        "unfitted_atomic_numbers": sorted(int(z) for z in unfitted_elements),
        "atomic_numbers_present": sorted(int(z) for z in elements_seen),
        "fitted_atomic_numbers": list(GBN2_FITTED_ELEMENTS),
        "unfitted_parameter_values": {"alpha": GBN2_UNFITTED_PARAMETERS[0],
                                      "beta": GBN2_UNFITTED_PARAMETERS[1],
                                      "gamma": GBN2_UNFITTED_PARAMETERS[2],
                                      "screen": 0.5},
        # Whether any mbondi3-specific adjustment could apply to this topology at all.
        "mbondi3_adjustable_residues_present": sorted(residue_names & mbondi3_targets),
        "mbondi3_reduces_to_mbondi2": not (residue_names & mbondi3_targets),
        "all_atoms_covered_by_gbn2_fit": not unfitted,
    }


#: mbondi3's two side-chain corrections, in angstrom, exactly as ParmEd applies them by name.
#: Verified against the installed `parmed.tools.changeradii.mbondi3`, which sets 1.4 on OD*/OE*
#: in GLU/ASP/GL4/AS4 and 1.17 on HH*/HE* in ARG.
MBONDI3_CARBOXYLATE_OXYGEN_ANGSTROM = 1.40
MBONDI3_GUANIDINIUM_HYDROGEN_ANGSTROM = 1.17


def apply_peptide_like_mbondi3(structure, peptide_map) -> dict:
    """Apply mbondi3's residue-keyed corrections to a solute that has no residues to key on.

    WHERE THIS SITS, AND WHY IT MATTERS

        Between `changeRadii(...)` and `createSystem(...)`. Before, because `createSystem` reads
        `solvent_radius` to build the CustomGBForce and derives `sr = screen * (radius - offset)`
        from it; a radius changed afterwards would leave `sr` describing the old one, and the two
        would disagree inside a single force. After `changeRadii`, because that call establishes
        the mbondi2 baseline these corrections modify -- running first would have the baseline
        overwrite them.

    WHY BY CHEMISTRY

        ParmEd selects by residue and atom NAME. A solute built from SMILES is one residue with
        one made-up name, so those rules match nothing and the radii are silently mbondi2 while
        the build still reports mbondi3. The mapped chemistry supplies the same atom sets by what
        they ARE.

        Stricter than the name rule in one place, deliberately: ParmEd also corrects `AS4`/`GL4`,
        the PROTONATED variants, because their names begin the same way. A neutral carboxylic
        acid is not a carboxylate and is not corrected here.

    Returns a record of exactly what changed, for the build log.
    """
    if peptide_map is None:
        return {"applied": False, "reason": "no peptide map"}
    atoms = structure.atoms
    targets: list[tuple[int, float, str]] = []
    for index in peptide_map.carboxylate_oxygens:
        targets.append((int(index), MBONDI3_CARBOXYLATE_OXYGEN_ANGSTROM,
                        "deprotonated side-chain carboxylate oxygen"))
    for index in peptide_map.guanidinium_hydrogens:
        targets.append((int(index), MBONDI3_GUANIDINIUM_HYDROGEN_ANGSTROM,
                        "protonated guanidinium hydrogen"))

    corrections = []
    for index, angstrom, why in targets:
        if index >= len(atoms):
            raise ValueError(
                f"the peptide map names atom {index} but the prepared structure has "
                f"{len(atoms)} atoms; the map and the topology are not the same molecule")
        atom = atoms[index]
        before = float(atom.solvent_radius)
        atom.solvent_radius = float(angstrom)
        corrections.append({
            "atom_index": index,
            "element": atom.element_name if hasattr(atom, "element_name") else atom.atomic_number,
            "reason": why,
            "intrinsic_radius_before_angstrom": before,
            "intrinsic_radius_after_angstrom": float(angstrom),
        })
    return {
        "applied": True,
        "policy": "mbondi3 side-chain corrections, selected by mapped chemistry",
        "units": "angstrom (intrinsic solvent_radius, before the GB offset is subtracted)",
        "carboxylate_oxygens": list(peptide_map.carboxylate_oxygens),
        "guanidinium_hydrogens": list(peptide_map.guanidinium_hydrogens),
        "n_corrected": len(corrections),
        "corrections": corrections,
        "molecular_map_digest": peptide_map.digest(),
        "sequence": list(peptide_map.sequence),
    }


def build_implicit_system(prmtop_path: Path, coordinate_path: Optional[Path] = None, *,
                          implicit_model: str = "GBn2", radii: str = "mbondi3",
                          remove_cm_motion: bool = True,
                          hydrogen_mass_amu: Optional[float] = None,
                          hmr_scope: str = "none",
                          nonpolar_sasa: bool = False,
                          peptide_map=None,
                          constraints: str = "HBonds"):
    """Build the implicit-solvent System, and report what the radius change actually did.

    Returns `(system, info)`. `info` records the radii before and after `changeRadii`, so a bundle
    can state whether the call mattered for this topology rather than asserting that it did.

    `hydrogen_mass_amu` repartitions at BUILD time, through ParmEd's own `createSystem`, so the
    serialized System is what will actually be integrated. It used to be impossible to ask for:
    the implicit route dropped the request and returned 1.008 amu hydrogens, which a `-hmr-v1`
    profile then integrated at 4 fs.

    Under implicit solvent there is no solvent, so the solute IS the whole system and `hmr_scope`
    "solute" and "all" name the same set of atoms. Both are accepted and recorded, rather than
    letting "solute" look like a setting that was ignored.
    """
    import parmed as pmd
    from openmm import app
    from openmm import unit as u
    from parmed.tools import changeRadii

    gb_object = _gb_object(implicit_model)

    structure = (pmd.load_file(str(prmtop_path), xyz=str(coordinate_path))
                 if coordinate_path is not None else pmd.load_file(str(prmtop_path)))
    before = radii_of(structure)
    changeRadii(structure, str(radii)).execute()
    # The residue-keyed corrections `changeRadii` could not reach, applied from the mapped
    # chemistry -- and still before `createSystem`, so every parameter it derives from a radius
    # is derived from the corrected one.
    peptide_like = apply_peptide_like_mbondi3(structure, peptide_map)
    after = radii_of(structure)

    scope = str(hmr_scope or "none")
    if scope not in ("none", "solute", "all"):
        raise ValueError(f"hmr_scope must be 'none', 'solute' or 'all'; got {scope!r}")
    if scope != "none" and hydrogen_mass_amu is None:
        raise ValueError(
            f"hmr_scope={scope!r} asks for hydrogen mass repartitioning but no "
            "hydrogen_mass_amu was given, so there is no target mass to repartition to.")
    if scope == "none" and hydrogen_mass_amu is not None:
        raise ValueError(
            f"hydrogen_mass_amu={hydrogen_mass_amu!r} was given with hmr_scope='none', so it "
            "would be silently ignored while the manifest recorded a repartitioned System.")

    mass_before = sum(a.mass for a in structure.atoms)
    from .system import constraint_option

    system = structure.createSystem(
        nonbondedMethod=app.NoCutoff,
        # HONOURED, not assumed. This was `app.HBonds` unconditionally, so an `AllBonds` request
        # built the same System as `HBonds` while the build log recorded `AllBonds (set)`.
        constraints=constraint_option(constraints),
        implicitSolvent=gb_object,
        # The ACE surface-area nonpolar term. False matches Amber's igb=8/gbsa=0 and the context
        # GBn2 was parameterised in; True matches OpenMM's implicit/gbn2.xml default. Stated here
        # rather than inherited, because the two differ by ~16 kJ/mol (~6 kT).
        useSASA=bool(nonpolar_sasa),
        removeCMMotion=remove_cm_motion,
        # ParmEd applies the repartitioning itself. Delegating avoids a THIRD implementation of
        # arithmetic that already exists twice in system.py, and keeps the masses inside the
        # System that gets serialized.
        **({"hydrogenMass": float(hydrogen_mass_amu) * u.dalton} if scope != "none" else {}),
    )

    hmr_record = _implicit_hmr_record(system, structure, scope, hydrogen_mass_amu, mass_before)

    coverage = gb_parameter_coverage(system, structure)
    max_change = max((abs(a - b) for a, b in zip(after, before)), default=0.0)
    info = {
        "construction": "parmed.Structure.createSystem",
        "construction_note": (
            "The ~16 kJ/mol CustomGBForce difference between construction routes is the ACE "
            "surface-area nonpolar term, not the radii: with useSASA=True ParmEd agrees with "
            "app.ForceField(+implicit/gbn2.xml) to 0.0017 kJ/mol on ACE-ALA-NME"),
        "implicit_model": implicit_model,
        "radii": radii,
        # What was REQUESTED versus what was actually done. The two differ for a solute with no
        # residue names, and the difference is the whole point of the correction pass.
        "radius_policy_requested": radii,
        "radius_assignment_method": (
            f"parmed.tools.changeRadii({radii!r}) then mbondi3 side-chain corrections from the "
            f"mapped peptide chemistry" if peptide_like.get("applied")
            else f"parmed.tools.changeRadii({radii!r})"),
        "peptide_like_mbondi3": peptide_like,
        "nonpolar_sasa": bool(nonpolar_sasa),
        "nonpolar_model": ("ACE surface-area term" if nonpolar_sasa else None),
        "nonbonded_method": "NoCutoff",
        "constraints": str(constraints),
        "remove_cm_motion": bool(remove_cm_motion),
        "radii_max_change_angstrom": max_change,
        "radii_change_was_a_no_op": max_change == 0.0,
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "uses_periodic_boundary_conditions": system.usesPeriodicBoundaryConditions(),
        # Which atoms the GB-Neck2 fit actually covers, read off the built CustomGBForce. This is
        # what decides whether an "igb=8 / mbondi3" claim is true for this particular solute.
        "parameter_coverage": coverage,
        "hmr": hmr_record,
    }
    if coverage.get("measured") and not coverage.get("all_atoms_covered_by_gbn2_fit"):
        print(
            f"  [implicit] WARNING: {coverage['n_atoms_with_unfitted_gbn2_parameters']} of "
            f"{coverage['n_particles']} atoms are outside the GB-Neck2 fit "
            f"(atomic numbers {coverage['unfitted_atomic_numbers']}) and carry ParmEd's generic "
            f"alpha/beta/gamma = {GBN2_UNFITTED_PARAMETERS} with screen = 0.5. "
            f"This system is NOT Amber igb=8 parity; treat the implicit result as experimental.",
            flush=True)
    if coverage.get("measured") and coverage.get("mbondi3_reduces_to_mbondi2"):
        print(
            "  [implicit] note: no GLU/ASP/GL4/AS4/ARG residue is present, so mbondi3's "
            "residue-specific adjustments cannot apply and the radii are exactly mbondi2.",
            flush=True)
    if system.usesPeriodicBoundaryConditions():
        raise RuntimeError(
            "the implicit-solvent System reports periodic boundary conditions, which it must not "
            "have: implicit solvent has no box. Refusing to publish it.")
    return system, info


def write_amber_files_from_openmm(topology, system, positions, out_dir: Path) -> dict:
    """Serialise an OpenMM topology/System to Amber files through ParmEd.

    Used by the OpenFF/Sage route, whose parameters exist only as an OpenMM System. The System
    passed here must be built WITHOUT constraints: constraints belong to `createSystem` on the way
    back out, and baking them into the topology would apply them twice.
    """
    import parmed as pmd
    from parmed import openmm as pmd_openmm

    out_dir = Path(out_dir)
    structure = pmd_openmm.load_topology(topology, system=system, xyz=positions)
    prmtop = out_dir / AMBER_TOPOLOGY_NAME
    coordinates = out_dir / AMBER_COORDINATE_NAME
    structure.save(str(prmtop), overwrite=True)
    structure.save(str(coordinates), format="rst7", overwrite=True)
    return {"prmtop": prmtop, "coordinates": coordinates, "parmed_version": pmd.__version__}


def implicit_provenance(info: dict) -> dict:
    """The record a bundle keeps about how its implicit System was built."""
    import openmm
    import parmed as pmd

    return {
        **info,
        "parmed_version": pmd.__version__,
        "openmm_version": openmm.__version__,
        "amber_files_are_provenance_only": (
            "system.prmtop and system.rst7 are construction intermediates and provenance for the "
            "OpenMM System. This repository has no Amber execution engine."),
        "no_water": True,
        "no_ions": True,
        "no_periodic_box": True,
        "no_barostat": "implicit solvent has no volume, so pressure is undefined",
    }


def build_amber_topology_via_tleap(pdb_path: Path, out_dir: Path, *, radii: str = "mbondi3",
                                   protein_forcefield: str = "leaprc.protein.ff14SB") -> dict:
    """Build prmtop/rst7 for a peptide or protein with tleap.

    `set default PBRadii mbondi3` is issued BEFORE `saveAmberParm`, which is what writes the radii
    into the topology. `changeRadii` still runs later and is a no-op here -- belt and braces for the
    case where a topology arrives without them.

    tleap's stdout is captured and kept: a run that "succeeded" while dropping an atom is a real
    failure mode, and the log is the only place it shows.
    """
    import shutil
    import subprocess

    if shutil.which("tleap") is None:
        raise RuntimeError(
            "tleap was not found on PATH. Implicit preparation of a peptide route builds its Amber "
            "topology with tleap (AmberTools); activate an environment that provides it.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prmtop = out_dir / AMBER_TOPOLOGY_NAME
    coordinates = out_dir / AMBER_COORDINATE_NAME
    leap_pdb = out_dir / "tleap_out.pdb"
    script = out_dir / "tleap.in"
    log = out_dir / "tleap.log"

    script.write_text("\n".join([
        f"source {protein_forcefield}",
        f"set default PBRadii {radii}",
        f"mol = loadPdb {pdb_path.resolve()}",
        f"saveAmberParm mol {prmtop.resolve()} {coordinates.resolve()}",
        f"savePdb mol {leap_pdb.resolve()}",
        "quit",
    ]) + "\n", encoding="utf-8")

    result = subprocess.run(["tleap", "-f", str(script)], capture_output=True, text=True,
                            check=False, cwd=str(out_dir))
    log.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0 or not prmtop.is_file():
        raise RuntimeError(
            f"tleap failed building the implicit topology from {pdb_path.name}.\n"
            f"  Its log is at {log}.\n{(result.stderr or result.stdout)[-1500:]}")
    return {
        "prmtop": prmtop,
        "coordinates": coordinates,
        "topology_pdb": leap_pdb,
        "tleap_input": script,
        "tleap_log": log,
        "tleap_commands": script.read_text(encoding="utf-8").splitlines(),
        "protein_forcefield": protein_forcefield,
        "radii_requested": radii,
    }


def build_implicit_bundle_inputs(*, route: str, cfg: dict, staging: Path,
                                 pdb: Optional[Path] = None, smiles: Optional[str] = None,
                                 sdf: Optional[Path] = None,
                                 implicit_model: str = "GBn2", radii: str = "mbondi3",
                                 hydrogen_mass_amu: Optional[float] = None,
                                 hmr_scope: str = "none",
                                 nonpolar_sasa: bool = False) -> dict:
    """Produce every Amber and OpenMM artefact an implicit bundle needs.

    Two routes reach the same ParmEd construction from different directions:

    * `peptide` -- tleap writes the topology with mbondi3 radii already in it;
    * `ligand` -- the vetted OpenFF/Sage parameterisation is serialised to Amber files through
      ParmEd. That System is built with `constraints=None` on purpose: constraints are applied by
      `createSystem` on the way back out, and baking them in here would apply them twice.
    """
    from openmm import XmlSerializer, app

    staging = Path(staging)
    if route == "peptide":
        if pdb is None:
            raise ValueError("the peptide route needs a PDB input")
        # The configured force field, not a hardcoded default: this value is what pairs with the
        # GB model, and it was previously ignored so every implicit peptide ran ff19SB regardless
        # of what the configuration said.
        protein_ff = (cfg.get("forcefield") or {}).get("protein") or "leaprc.protein.ff14SB"
        amber = build_amber_topology_via_tleap(pdb, staging, radii=radii,
                                               protein_forcefield=protein_ff)
        topology_source = amber["topology_pdb"]
    elif route == "ligand":
        amber = _amber_files_for_ligand(cfg, staging, smiles=smiles, sdf=sdf)
        topology_source = amber["topology_pdb"]
    else:
        raise ValueError(f"implicit preparation has no route {route!r}; expected peptide or ligand")

    # THE MAP, for a peptide-like solute only. Read from the SDF the ligand route already wrote,
    # which is where bond orders and formal charges survive -- a topology has neither, and both
    # decide whether a side chain is a carboxylate or a carboxylic acid.
    peptide_map = None
    if str((cfg.get("solute") or {}).get("kind") or "") == "peptide-like":
        from .peptide_map import map_from_sdf

        sdf = amber.get("ligand_sdf")
        if not sdf:
            raise ValueError(
                "solute.kind is 'peptide-like' but the prepared inputs carry no SDF, so the "
                "peptide chemistry cannot be read. Bond orders are not recoverable from a "
                "topology.")
        peptide_map = map_from_sdf(sdf)

    system, info = build_implicit_system(
        amber["prmtop"], amber["coordinates"],
        implicit_model=implicit_model, radii=radii,
        hydrogen_mass_amu=hydrogen_mass_amu, hmr_scope=hmr_scope,
        peptide_map=peptide_map,
        constraints=(cfg.get("system_build") or {}).get("constraints", "HBonds"))

    (staging / "system.xml").write_text(XmlSerializer.serialize(system), encoding="utf-8")
    pdb_file = app.PDBFile(str(topology_source))
    with (staging / "topology.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(pdb_file.topology, pdb_file.positions, handle, keepIds=True)

    # The same build record the explicit path writes, so everything downstream -- the bundle
    # manifest, the REST2 bridge, the unscaled torsions -- reads one shape. `geometry` is null rather
    # than absent: "this System has no periodic box" is a fact worth recording, and a missing key
    # would be indistinguishable from a record that forgot to write it.
    from .system import classify_unscaled_torsions, omega_central_bonds

    topology = app.PDBFile(str(staging / "topology.pdb")).topology
    solute = list(range(system.getNumParticles()))
    # The ligand route needs the SDF: bond orders are not recoverable from a topology, and amide
    # detection depends on them. It is written by the same step that built the conformer.
    ligand_sdf = amber.get("ligand_sdf")
    unscaled_info = classify_unscaled_torsions(
        topology, solute, ligand_sdf=(Path(ligand_sdf) if ligand_sdf else None))
    build_record = {
        "suffix": "system",
        "route": route,
        # What the molecule actually came from. This said "smiles" for every ligand build, which
        # was true while that was the only molecular-graph input and became a false record the
        # moment a second one existed.
        "input_route": ("pdb" if route == "peptide" else ("sdf" if sdf is not None else "smiles")),
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "n_solute_atoms": system.getNumParticles(),
        "n_waters": 0,
        "ions": None,
        "salt": None,
        "water": None,
        "geometry": None,
        "nonbonded": {"method": "NoCutoff", "cutoff_nm": None},
        # measured on the built System, not restated from the request
        "hmr": info["hmr"],
        "constraints": str((cfg.get("system_build") or {}).get("constraints", "HBonds")),
        "rigid_water": False,
        # The deprecated structural detector's list, kept under its historical name for bundles
        # that predate the classifier. What REST2 actually leaves unscaled is the next key.
        "omega_central_bonds": omega_central_bonds(topology, solute),
        "unscaled_torsions": {k: unscaled_info[k] for k in
                              ("unscaled_central_bonds", "central_bonds", "proline_like_scaled_bonds", "unclassified",
         "unscaled_impropers", "detection_method", "detector_version", "amide_detail")},
        "degrees_of_freedom": (3 * system.getNumParticles() - system.getNumConstraints() - 3),
        "implicit": info,
    }
    (staging / "system_simbox.json").write_text(
        json.dumps(build_record, indent=2) + "\n", encoding="utf-8")

    return {
        "system": system,
        "system_xml": staging / "system.xml",
        "topology_pdb": staging / "topology.pdb",
        "build_record": build_record,
        # Surfaced at the top level so the caller writing forcefield.json states what the System
        # actually carries. Without it that file said "none" while system_manifest.json said
        # "solute", and the bundle's own three-way agreement check refused to publish -- correctly.
        "hmr": info["hmr"],
        "prmtop": amber["prmtop"],
        "coordinates": amber["coordinates"],
        "n_particles": system.getNumParticles(),
        "n_solute_atoms": system.getNumParticles(),   # implicit: the solute IS the system
        "route": route,
        "build": {**info, **{k: v for k, v in amber.items()
                             if k in ("tleap_commands", "protein_forcefield", "radii_requested",
                                      "small_molecule_forcefield", "charge_method",
                                      # The OpenFF report from the ligand route, so
                                      # forcefield.json can name the resource actually selected
                                      # rather than only the human-facing Sage label.
                                      "forcefield_info")}},
    }


def _amber_files_for_ligand(cfg: dict, staging: Path, *, smiles: Optional[str],
                            sdf: Optional[Path] = None) -> dict:
    """OpenFF/Sage parameters, serialised to Amber files through ParmEd."""
    from openmm import app

    from .system import build_forcefield, initial_structure, initial_structure_from_sdf

    # Exactly one of the two molecular-graph inputs. Both write the same two files into
    # `staging/structure`, and everything below this point is common to them.
    if sdf is not None:
        structure = initial_structure_from_sdf(sdf, staging / "structure", cfg)
    elif smiles:
        structure = initial_structure(smiles, staging / "structure", cfg)
    else:
        raise ValueError("the implicit ligand route needs a .smi or .sdf input")
    ligand_sdf = Path(structure["solute_sdf"])
    forcefield, ff_info = build_forcefield(cfg, ligand_sdf=ligand_sdf, route="ligand")

    solute = app.PDBFile(str(structure["solute_pdb"]))
    # No constraints here: they are applied by createSystem on the way back out.
    bare = forcefield.createSystem(solute.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    written = write_amber_files_from_openmm(solute.topology, bare, solute.positions, staging)

    topology_pdb = staging / "ligand_topology.pdb"
    with topology_pdb.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(solute.topology, solute.positions, handle, keepIds=True)
    return {
        "prmtop": written["prmtop"],
        "coordinates": written["coordinates"],
        "topology_pdb": topology_pdb,
        "ligand_sdf": str(ligand_sdf),
        "small_molecule_forcefield": (cfg.get("forcefield") or {}).get("ligand"),
        "charge_method": (cfg.get("forcefield") or {}).get("ligand_charge_method"),
        "forcefield_info": ff_info,
    }
