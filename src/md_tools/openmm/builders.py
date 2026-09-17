"""The validated OpenMM system builders.

Extracted verbatim from the retired `sysgen.py` when the `build-top`/`build-md` route was removed.
These are the functions `build-top` and the replica runtime actually use -- protonation, solvation,
the GBn2/mbondi3 implicit path, the unscaled-torsion classification and the solute record -- and they are the
code the scientific tests have always covered. Nothing here was rewritten in the move.

What did NOT come with them: the `generate_system` orchestration, the embedded `dataset:` block and
its v1 manifest, the provenance writer and the checksum manifests. Those belonged to the retired
route, and registration owns that ground now.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from ..build.strict import ConfigError
from .system_config import openff_resource
from .yaml_io import write_yaml
from .system_defaults import DEFAULT_PADDING_NM, DEFAULT_SOLVENT
from ..rest2 import UNSCALED_TORSION_DETECTOR_VERSION, torsion_exclusion_report

#: Short water labels an older configuration may still carry, and the QUALIFIED OpenMM resource
#: each one has to become.
#:
#: OpenMM ships both spellings. The top-level `opc.xml` / `tip3p.xml` define water only, so a box
#: that needed Na+ and Cl- failed with "No template found for residue (NA)" -- after solvation had
#: already placed them. The `amber14/` and `amber19/` copies carry the ion templates alongside the
#: same water parameters.
#:
#: An explicit table rather than a blanket `amber19/` prefix: that prefix silently turned
#: `tip3p.xml` into `amber19/tip3p.xml`, which OpenMM does not ship. Written defaults are already
#: qualified, so this only rescues a hand-written or pre-0.4 file.
_QUALIFIED_WATER = {
    "opc.xml": "amber19/opc.xml",
    "opc3.xml": "amber19/opc3.xml",
    "tip3p.xml": "amber14/tip3p.xml",
    "tip3pfb.xml": "amber14/tip3pfb.xml",
}

class Log:
    """A readable preparation log, echoed to stdout as it is written."""

    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.lines: list[str] = []
        self.path = Path(path)
        self.echo = echo

    def __call__(self, message: str) -> None:
        self.lines.append(message)
        if self.echo:
            print(f"  {message}")

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return self.path

def _legacy_cfg(resolved: dict[str, Any]) -> dict[str, Any]:
    """Map the user's YAML onto the dict shape the existing builders take.

    A translation layer, deliberately in one function and one direction. The builders were written
    against a large nested configuration; rewriting their signatures would risk the chemistry for a
    cosmetic gain, so the mapping is explicit and the science is untouched.
    """
    from .builder_defaults import DEFAULTS

    cfg = yaml.safe_load(yaml.safe_dump(DEFAULTS))
    solute = resolved.get("solute") or {}
    forcefield = resolved.get("forcefield") or {}
    constraints = resolved.get("constraints") or {}
    implicit = resolved["solvation"] == "implicit"

    cfg["forcefield"]["protein"] = forcefield.get("protein")
    cfg["forcefield"]["water"] = _water_xml(forcefield.get("water"))
    cfg["forcefield"]["ligand"] = openff_resource(solute.get("ligand_forcefield"))
    cfg["forcefield"]["ligand_charge_method"] = solute.get("ligand_charge_method")
    # THE CLASSIFICATION, carried across. This mapping rebuilds `cfg` from `DEFAULTS` and copies
    # named keys, so anything not copied here simply does not reach the builders -- which is how
    # a correctly resolved `solute.kind: peptide-like` reached the log and the record while the
    # implicit builder still saw no kind at all and applied no corrections.
    cfg.setdefault("solute", {})
    cfg["solute"]["kind"] = solute.get("kind") or ("peptide" if solute.get("peptide", True)
                                                   else "ligand")

    cfg["system_build"]["constraints"] = constraints.get("type", "HBonds")
    cfg["system_build"]["rigid_water"] = bool(constraints.get("rigid_water", not implicit))
    cfg["system_build"]["hydrogen_mass_amu"] = constraints.get("hydrogen_mass_amu")
    cfg["system_build"]["hmr_scope"] = (
        "solute" if constraints.get("hydrogen_mass_amu") is not None else "none")

    if implicit:
        cfg["system_build"]["nonbonded_method"] = "NoCutoff"
        cfg["system_build"]["nonbonded_cutoff_nm"] = None
        cfg["system_build"]["ewald_error_tolerance"] = None
        cfg["system_build"]["use_dispersion_correction"] = False
        cfg["system_build"]["switch_distance_nm"] = None
        cfg["system_build"]["minimum_image_margin_nm"] = None
    else:
        solvent = resolved.get("solvent") or {}
        cfg["system_build"]["nonbonded_cutoff_nm"] = solvent.get("cutoff_nm", 1.0)
        cfg["solvation"]["water_model"] = str(solvent.get("model", DEFAULT_SOLVENT)).lower()
        cfg["solvation"]["box_shape"] = solvent.get("box_shape", "dodecahedron")
        cfg["solvation"]["padding_nm"] = solvent.get("padding_nm", DEFAULT_PADDING_NM)
        cfg["solvation"]["ionic_strength_molar"] = solvent.get("ionic_strength_molar", 0.15)
        cfg["solvation"]["positive_ion"] = solvent.get("positive_ion", "Na+")
        cfg["solvation"]["negative_ion"] = solvent.get("negative_ion", "Cl-")
    return cfg

def _water_xml(name: Optional[str]) -> Optional[str]:
    """The OpenMM water resource that will actually be loaded, or a refusal that names the file."""
    if not name:
        return None
    text = str(name).strip()
    if "/" in text:
        return text                                # already qualified; take it as written
    qualified = _QUALIFIED_WATER.get(text.lower())
    if qualified is None:
        raise ConfigError(
            f"forcefield.water = {text!r} is not a resource this package can qualify. OpenMM ships "
            f"the ion-carrying water files under a directory, so write the full resource name "
            f"(for example {', '.join(sorted(set(_QUALIFIED_WATER.values())))}). Known short "
            f"labels: {', '.join(sorted(_QUALIFIED_WATER))}.")
    return qualified

def _unclassified_evidence(candidate: dict) -> dict:
    """One unclassified item -- an amide candidate or a whole residue -- with WHY.

    A bare list of unresolved atom pairs tells a reader that something blocked production and
    nothing about what to do next. What is needed is the bond, the residues on both ends, and the
    sentence the classifier itself would have said -- so the answer ("supply the SDF", or "this
    input is wrong") does not require re-deriving the classification by hand.
    """
    bond = candidate.get("bond")
    return {
        "bond": [int(a) for a in bond] if bond is not None else None,
        "carbon": None if candidate.get("carbon") is None else int(candidate["carbon"]),
        "nitrogen": None if candidate.get("nitrogen") is None else int(candidate["nitrogen"]),
        "carbon_residue": candidate.get("carbon_residue"),
        "nitrogen_residue": candidate.get("nitrogen_residue"),
        "carbon_residue_index": candidate.get("carbon_residue_index"),
        "nitrogen_residue_index": candidate.get("nitrogen_residue_index"),
        "ring_sizes": candidate.get("ring_sizes"),
        "evidence": candidate.get("ambiguous"),
    }

def unscaled_bonds_of_solute_document(document) -> list[tuple[int, int]]:
    """The unscaled central bonds a `solute.yaml` records, under its current key or 0.5.3's.

    0.5.3 wrote `rest2.omega_excluded_bonds`; 0.5.4 writes `rest2.unscaled_central_bonds`. A run
    generated by 0.5.3 must still resume and register, so both are read -- and a document carrying
    BOTH is refused, because two lists could disagree and nothing would say which one ran.
    """
    rest2 = (document or {}).get("rest2") or {}
    if "unscaled_central_bonds" in rest2 and "omega_excluded_bonds" in rest2:
        raise ValueError("a solute document records both unscaled_central_bonds and the 0.5.3 "
                         "omega_excluded_bonds; one of them is stale")
    pairs = rest2.get("unscaled_central_bonds", rest2.get("omega_excluded_bonds", []))
    return [tuple(int(a) for a in pair) for pair in pairs]


def unclassified_of_solute_document(document) -> list:
    """The unclassified items a `solute.yaml` records, under its current key or 0.5.3's."""
    rest2 = (document or {}).get("rest2") or {}
    return list(rest2.get("unclassified", rest2.get("omega_ambiguous_candidates", [])) or [])


def _solute_document(topology, solute_indices, unscaled, *, route: str,
                     system=None) -> dict[str, Any]:
    residues = []
    solute_set = set(int(i) for i in solute_indices)
    for residue in topology.residues():
        indices = [a.index for a in residue.atoms()]
        if any(i in solute_set for i in indices):
            residues.append({"index": residue.index, "name": residue.name,
                             "atoms": [int(min(indices)), int(max(indices))]})
    return {
        "schema_version": 1,
        "route": route,
        "n_solute_atoms": len(solute_set),
        # A contiguous range in every path this repository builds; recorded as first/last so the
        # generated scripts need no list of thousands of integers.
        "solute_atom_range": [int(min(solute_set)), int(max(solute_set))] if solute_set else None,
        "solute_atom_indices_are_contiguous": (
            bool(solute_set) and max(solute_set) - min(solute_set) + 1 == len(solute_set)),
        "residues": residues,
        "rest2": {
            "enhanced_region": "solute",
            # Central bonds whose torsions must NOT be scaled -- ordinary amide omega, aromatic
            # ring, other double bond -- each with its class. A hot state that twisted one would
            # sample a geometry the cold state never visits.
            "unscaled_central_bonds": [[int(a), int(b)] for a, b in unscaled.get(
                "unscaled_central_bonds", [])],
            "central_bonds": list(unscaled.get("central_bonds", [])),
            "unscaled_impropers": bool(unscaled.get("unscaled_impropers", True)),
            "proline_like_scaled_bonds": [[int(a), int(b)] for a, b in unscaled.get(
                "proline_like_scaled_bonds", [])],
            "detection_method": unscaled.get("detection_method"),
            # The route and the detector version are persisted, so a record can be checked
            # against the detector that produced it rather than against whichever detector
            # happens to be installed when it is read.
            "detection_route": route,
            "detector_version": UNSCALED_TORSION_DETECTOR_VERSION,
            # Items neither rule could name, with their evidence. A NON-EMPTY LIST BLOCKS
            # PRODUCTION: guessing either way silently changes the Hamiltonian.
            "unclassified": [
                _unclassified_evidence(candidate)
                for candidate in unscaled.get("unclassified", [])],
            # What the exclusion actually DOES: the PeriodicTorsionForce terms each excluded
            # central bond protects, in this System. The bond pair alone would leave a reader to
            # re-derive that mapping with assumptions that may not match the ones used here.
            **(torsion_exclusion_report(
                system, solute_set,
                [tuple(int(a) for a in bond)
                 for bond in unscaled.get("unscaled_central_bonds", [])],
                unscaled.get("unscaled_impropers", True)) if system is not None
               else {}),
        },
    }

def _build_explicit(input_path: Path, cfg: dict, staging: Path, *, route: str, log: Log) -> dict:
    """Protonate, solvate in the dodecahedral box, build the System."""
    from openmm import XmlSerializer, app

    from .solvation import solvate
    from .system import build_system

    ligand_sdf = None
    package_record = None
    if route == "ligand":
        # ONE staging location for the prepared molecule, shared with the implicit route. This was
        # bare `staging` here and `staging / "structure"` there, while `build/top.py` copies
        # `built.sdf` out of the latter -- so an EXPLICIT ligand build wrote its SDF where nothing
        # read it, emitted no `built.sdf` at all, and left every later consumer of that file (the
        # unscaled-torsion classifier's bond orders, `map_from_sdf`, `preflight._ligand_sdf_beside`) with
        # nothing to read. No test caught it because every `built.sdf` assertion ran under GBn2.
        prepared = _prepare_molecule(input_path, staging / "structure", cfg)
        # `solute_pdb` / `solute_sdf` are the keys the preparers actually return; `pdb`/`sdf`
        # never existed, so the SMILES route raised KeyError before reaching parameterisation.
        source = Path(prepared["solute_pdb"])
        ligand_sdf = Path(prepared["solute_sdf"])
        log(f"ligand       : {prepared.get('smiles', '')[:60]}")
        # THE PACKAGE, before any force field exists: every later step loads these parameters.
        from ..ligands.build import attach_for_build

        package_record = attach_for_build(cfg, staging, solute_sdf=ligand_sdf, solute_pdb=source)
        if package_record is not None:
            log(f"parameters   : {package_record.get('reference', package_record['how'])} "
                f"({package_record['how']})")
    else:
        source = staging / "input.pdb"
        shutil.copy2(input_path, source)

    from .system import protonate

    protonated = protonate(source, staging, cfg, ligand_sdf=ligand_sdf,
                           input_route=(molecule_input_route(input_path) if route == "ligand"
                                        else "pdb"))
    log(f"protonation  : pH {protonated.get('ph')}, "
        f"{protonated.get('n_hydrogens_before')} -> {protonated.get('n_hydrogens_after')} hydrogens")

    solvated = solvate(Path(protonated["output_pdb"]), staging, cfg,
                       ligand_sdf=ligand_sdf, route=route)
    ions = solvated.get("ions") or {}
    log(f"solvation    : {solvated['n_waters']} waters, "
        f"ions {ions.get('counts', ions)}, box {solvated.get('box_shape')} "
        f"({solvated.get('box_volume_nm3', 0):.1f} nm^3)")

    geometry = solvated.get("geometry") or {}
    log(f"box geometry : requested {geometry.get('padding_nm_requested')} nm padding -> "
        f"solute-to-copy clearance {geometry.get('solute_image_clearance_nm')} nm, "
        f"reduced-box height {geometry.get('min_reduced_box_height_nm')} nm "
        f"(needs {geometry.get('required_cutoff_height_nm')} nm for a "
        f"{geometry.get('nonbonded_cutoff_nm')} nm cutoff"
        + (", box grown to fit" if geometry.get("grown_for_cutoff") else "") + ")")

    built = build_system(Path(solvated["output_pdb"]), staging, cfg, solvated["n_solute_atoms"],
                         ligand_sdf=ligand_sdf, route=route)
    nonbonded = built.get("nonbonded") or {}
    log(f"system       : {nonbonded.get('method')}, cutoff {nonbonded.get('cutoff_nm')} nm, "
        f"switching {nonbonded.get('switching')}, dispersion correction "
        f"{nonbonded.get('dispersion_correction')}, Ewald tolerance "
        f"{nonbonded.get('ewald_error_tolerance')}, {cfg['system_build']['constraints']}")

    pdb = app.PDBFile(str(solvated["output_pdb"]))
    system = XmlSerializer.deserialize(Path(built["system_xml"]).read_text())
    state_path = _write_initial_state(system, pdb, staging)
    return {"system_xml": built["system_xml"], "topology_pdb": solvated["output_pdb"],
            "initial_state": state_path, "n_solute_atoms": solvated["n_solute_atoms"],
            "ligand_sdf": ligand_sdf, "omega": built, "ligand_package": package_record,
            # Carried through for forcefield.json: what the box actually ended up containing is
            # part of how the system was parameterised, not a log line.
            "n_waters": solvated.get("n_waters"), "ions": solvated.get("ions"),
            "box_shape": solvated.get("box_shape"),
            "box_volume_nm3": solvated.get("box_volume_nm3"),
            "box_vectors_nm": solvated.get("box_vectors_nm"),
            # The full geometry resolution: requested padding, the three distances that are not
            # interchangeable, and whether the box had to be grown for the cutoff. Recorded rather
            # than reduced to one number, because "1.5 nm padding" alone does not say what
            # clearance the built box actually has.
            "box_geometry": solvated.get("geometry"),
            "salt": solvated.get("salt"),
            "water_model": solvated.get("water_model"),
            "water_packing_model": solvated.get("water_packing_model"),
            "water_packing_substituted": solvated.get("water_packing_substituted"),
            "water_model_reconciled": solvated.get("water_model_reconciled")}

def _build_complex(input_path: Path, cfg: dict, staging: Path, *, mapped, log: Log) -> dict:
    """Protein chains plus mapped ligand instances: hydrogens, solvent, System.

    `mapped` is the structure with every selected ligand already replaced by its package's atoms
    (`md_tools.ligands.mapping.map_ligands`), resolved by `build-top` before any output existed.
    The instances are frozen through every step and checked after each one.
    """
    from openmm import XmlSerializer, app

    from ..ligands.mapping import assert_instances_unchanged
    from .solvation import solvate
    from .system import build_system, protonate_complex

    for instance in mapped.instances:
        log(f"ligand       : {instance.residue_name} {instance.selector.label()} -> "
            f"{instance.package.reference}")
    protonated = protonate_complex(mapped, staging, cfg)
    log(f"protonation  : pH {protonated['ph']}, {protonated['n_hydrogens_before']} -> "
        f"{protonated['n_hydrogens_after']} hydrogens; "
        f"{len(protonated['frozen_ligand_instances'])} ligand instance(s) kept as packaged")

    solvated = solvate(Path(protonated["output_pdb"]), staging, cfg, route="complex",
                       residue_templates_for=mapped.residue_templates)
    ions = solvated.get("ions") or {}
    log(f"solvation    : {solvated['n_waters']} waters, ions {ions.get('counts', ions)}, box "
        f"{solvated.get('box_shape')} ({solvated.get('box_volume_nm3', 0):.1f} nm^3)")

    residue_sdfs = {}
    sdf_dir = staging / "ligand_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    for instance in mapped.instances:
        name = instance.residue_name.upper()
        path = sdf_dir / f"{instance.package.parameter_id}.sdf"
        if name in residue_sdfs and residue_sdfs[name] != path:
            raise ValueError(
                f"residue name {name} is mapped to two different packages; the unscaled-torsion "
                f"classifier reads bond orders per residue name, so give the species distinct "
                f"residue names")
        path.write_bytes((instance.package.path / "molecule.sdf").read_bytes())
        residue_sdfs[name] = path

    built = build_system(Path(solvated["output_pdb"]), staging, cfg, solvated["n_solute_atoms"],
                         route="complex", residue_templates_for=mapped.residue_templates,
                         residue_sdfs=residue_sdfs)
    pdb = app.PDBFile(str(solvated["output_pdb"]))
    assert_instances_unchanged(mapped, pdb.topology, pdb.positions, step="solvation")
    nonbonded = built.get("nonbonded") or {}
    log(f"system       : {nonbonded.get('method')}, cutoff {nonbonded.get('cutoff_nm')} nm, "
        f"{cfg['system_build']['constraints']}")
    system = XmlSerializer.deserialize(Path(built["system_xml"]).read_text())
    state_path = _write_initial_state(system, pdb, staging)
    return {"system_xml": built["system_xml"], "topology_pdb": solvated["output_pdb"],
            "initial_state": state_path, "n_solute_atoms": solvated["n_solute_atoms"],
            "ligand_sdf": None, "omega": built, "protonation": protonated,
            "n_waters": solvated.get("n_waters"), "ions": solvated.get("ions"),
            "box_shape": solvated.get("box_shape"),
            "box_volume_nm3": solvated.get("box_volume_nm3"),
            "box_vectors_nm": solvated.get("box_vectors_nm"),
            "box_geometry": solvated.get("geometry"), "salt": solvated.get("salt"),
            "water_model": solvated.get("water_model"),
            "water_packing_model": solvated.get("water_packing_model"),
            "water_packing_substituted": solvated.get("water_packing_substituted"),
            "water_model_reconciled": solvated.get("water_model_reconciled")}

def _build_implicit(input_path: Path, cfg: dict, staging: Path, *, route: str, log: Log) -> dict:
    """The validated ParmEd GBn2 + mbondi3 path."""
    from openmm import XmlSerializer, app

    from .implicit import build_implicit_bundle_inputs

    molecule_route = molecule_input_route(input_path) if route == "ligand" else None
    smiles = _smiles_from(input_path) if molecule_route == "smiles" else None
    sdf_input = input_path if molecule_route == "sdf" else None
    pdb_input = input_path if route == "peptide" else None
    built = build_implicit_bundle_inputs(
        route=("peptide" if route == "peptide" else "ligand"),
        cfg=cfg, staging=staging, pdb=pdb_input, smiles=smiles, sdf=sdf_input,
        implicit_model="GBn2", radii="mbondi3",
        # An explicit, recorded choice rather than a library default: including the ACE
        # surface-area term changes the energy by ~16 kJ/mol (~6 kT) on ACE-ALA-NME.
        nonpolar_sasa=bool((cfg.get("implicit_solvent") or {}).get("nonpolar_sasa", False)),
        hydrogen_mass_amu=cfg["system_build"].get("hydrogen_mass_amu"),
        hmr_scope=str(cfg["system_build"].get("hmr_scope") or "none"))
    log("system       : GBn2 / mbondi3 via ParmEd.Structure.createSystem (NOT AmberPrmtopFile: "
        "the two differ by ~16 kJ/mol in CustomGBForce)")
    hmr = built.get("hmr") or {}
    if hmr.get("scope") != "none":
        log(f"hmr          : {hmr['n_hydrogens_repartitioned']} hydrogens to "
            f"{hmr['target_hydrogen_mass_amu']} amu, total mass conserved")

    pdb = app.PDBFile(str(built["topology_pdb"]))
    system = XmlSerializer.deserialize(Path(built["system_xml"]).read_text())
    state_path = _write_initial_state(system, pdb, staging)
    return {"system_xml": built["system_xml"], "topology_pdb": built["topology_pdb"],
            "initial_state": state_path, "n_solute_atoms": built["n_solute_atoms"],
            "ligand_sdf": built.get("ligand_sdf"),
            "ligand_package": built.get("ligand_package"),
            "omega": built.get("build_record") or {},
            # The builder's own report of what it loaded: the tleap protein resource, the radii,
            # and on the ligand route the OpenFF report from build_forcefield. Surfaced here so
            # forcefield.json states the resources actually used rather than re-deriving them.
            "implicit_report": built.get("build") or {},
            "hmr": built.get("hmr") or {}}

def _write_initial_state(system, pdb, staging: Path) -> Path:
    """Positions and box vectors, with no velocities: nothing has been integrated yet."""
    from openmm import Context, VerletIntegrator, XmlSerializer, unit

    context = Context(system, VerletIntegrator(0.001 * unit.picoseconds))
    context.setPositions(pdb.positions)
    if system.usesPeriodicBoundaryConditions():
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    state = context.getState(getPositions=True)
    path = staging / "initial_state.xml"
    path.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    del context
    return path

def _smiles_from(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"{path} is empty; a non-peptide input must contain a SMILES string")
    return text.splitlines()[0].split()[0]


def molecule_input_route(path: Path) -> str:
    """`smiles` or `sdf` -- which molecular-graph input this is.

    One spelling, derived in one place. The route travels into `protonate`, into `resolve_route`
    and into the build record, and three independent suffix comparisons would be three chances to
    disagree about the same file.
    """
    return "sdf" if Path(path).suffix.lower() == ".sdf" else "smiles"


def _prepare_molecule(input_path: Path, out_dir: Path, cfg: dict) -> dict:
    """The prepared solute, from whichever molecular-graph input was supplied.

    Both preparers write the same two files into *out_dir* and return the same keys, so
    everything downstream of here is identical for the two routes -- which is the point: the
    difference between them is where the coordinates came from, and it ends at this function.
    """
    from .system import initial_structure, initial_structure_from_sdf

    if molecule_input_route(input_path) == "sdf":
        return initial_structure_from_sdf(input_path, out_dir, cfg)
    return initial_structure(_smiles_from(input_path), out_dir, cfg)
