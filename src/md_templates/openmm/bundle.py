"""Portable prepared bundles: build one, and prove one has not changed.

A bundle is the unit that moves between machines. It contains everything needed to start a REST2
run and nothing that ties it to the machine that built it:

    bundle/
      system.xml                 the parameterised OpenMM System
      topology.pdb               topology, solvated coordinates, box vectors
      simbox.json                n_solute_atoms and the omega classification
      equilibrated_state.xml     positions, velocities, box — where production starts
      system.yaml                the system manifest it was built from
      experiment.prepare.yaml    the experiment manifest it was built from
      resolved_config.json       the fully resolved baseline config
      bundle_manifest.json       identity, provenance, and a SHA-256 for every file above

Two routes to the same starting point, and they are not equivalent:

* **Rebuild** from the same system manifest on the target machine. Scientifically consistent — the
  same molecule, force field, charge method and solvation settings — but not bitwise identical.
  AM1-BCC charges, conformer embedding and water placement all depend on library versions and on
  floating-point details of the host, so the System and the starting state will differ in the last
  digits and the trajectories will diverge immediately.
* **Transfer** the prepared bundle. This is the required route when the System and starting state
  must be *identical*, which is the case for any comparison that pools runs from several machines.

`validate-bundle` recomputes every hash, so a transferred bundle that was truncated, edited, or
partially copied fails before it can contaminate a run.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from . import provenance
from .config import resolve_config
from ..core.fingerprint import build_projection, fingerprint
from .solvation import salt_accounting
from .schemas import (
    ExperimentManifest,
    ManifestError,
    SCHEMA_VERSION,
    SystemManifest,
    canonical_json,
    config_hash,
    sha256_file,
    sha256_text,
)

#: Every file a bundle must contain. `validate-bundle` rejects a bundle missing any of them.
BUNDLE_FILES = (
    "system.xml",
    "topology.pdb",
    # A System XML carries parameters but no atom or residue names, and nothing that says which
    # atoms are the solute or which bonds are omega. Without this the bundle cannot build the
    # REST2-scaled systems, so it belongs in the bundle rather than beside it.
    "simbox.json",
    "equilibrated_state.xml",
    "system.yaml",
    "experiment.prepare.yaml",
    "resolved_config.json",
)
BUNDLE_MANIFEST = "bundle_manifest.json"


class BundleError(RuntimeError):
    """A bundle is incomplete, modified, or inconsistent with its manifest."""


def prepare(
    system: SystemManifest,
    experiment: ExperimentManifest,
    out_root: Path,
    *,
    platform: str = "CPU",
    device: Optional[str] = None,
    name: Optional[str] = None,
    omega_exclusion: Optional[bool] = None,
    resolved_cfg: Optional[dict] = None,
    canonical: Optional[dict] = None,
    original_config: Optional[Path] = None,
    original_input: Optional[Path] = None,
) -> Path:
    """Build a bundle. Returns the bundle directory.

    `resolved_cfg` is a configuration already resolved from the canonical model; when given it is
    used as-is and the manifests are not consulted for scientific values. That keeps ONE builder
    behind both front ends rather than a second execution path for canonical documents.

    Preparation is deliberately platform-agnostic in its *output*: the platform argument only
    decides where the equilibration runs, not what is written. A bundle prepared on CPU and one
    prepared on CUDA are interchangeable as inputs.
    """
    from .equilibration import build_simbox, minimize_equilibrate

    if resolved_cfg is not None:
        cfg = copy.deepcopy(resolved_cfg)
        cfg["production"]["platform"] = platform
        cfg["production"]["device_index"] = device
    else:
        cfg = resolve_config(system, experiment, platform=platform, device=device,
                             omega_exclusion=omega_exclusion)
    chash = config_hash(system.doc, experiment.doc)
    stamp = provenance.run_stamp()
    bundle_dir = Path(out_root).resolve() / (name or f"{stamp}__{system.system_id}__bundle__{chash}")
    if bundle_dir.exists():
        raise BundleError(
            f"refusing to overwrite an existing bundle: {bundle_dir}\n"
            "Bundles are immutable; remove it explicitly or choose another --out-root."
        )
    work = bundle_dir / "work"
    work.mkdir(parents=True)

    # ---- stage (a): parameterised solvated box -------------------------------------------------
    if system.route == "smiles":
        info = build_simbox(cfg, work, "simbox", smiles=str(system.doc["input"]["smiles"]))
    else:
        info = build_simbox(cfg, work, "simbox", pdb=system.input_path())

    # ---- stage (b): minimise + equilibrate -----------------------------------------------------
    eq = minimize_equilibrate(
        cfg, work / "simbox_system.xml", work / "simbox_topology.pdb", work, "eq"
    )

    # ---- assemble the portable bundle ----------------------------------------------------------
    shutil.copy2(work / "simbox_system.xml", bundle_dir / "system.xml")
    shutil.copy2(work / "simbox_topology.pdb", bundle_dir / "topology.pdb")
    shutil.copy2(work / "simbox_simbox.json", bundle_dir / "simbox.json")
    shutil.copy2(work / "eq_state.xml", bundle_dir / "equilibrated_state.xml")
    shutil.copy2(system.source, bundle_dir / "system.yaml")
    # The pdb route's manifest POINTS AT a sibling file; the smiles route carries its input inline.
    # Copying system.yaml alone therefore produced a bundle whose own manifest could not be re-read:
    # `input.pdb` resolves relative to the manifest, which now lives in the bundle, and the file was
    # not there. That made every peptide bundle unusable the moment anything re-validated it --
    # `validate-bundle`, or `rest2 --bundle`, which does so before it runs. The filename is
    # preserved so both the declared relative path and `input.pdb_sha256` still resolve.
    if system.route == "pdb":
        shutil.copy2(system.input_path(), bundle_dir / system.input_path().name)
    shutil.copy2(experiment.source, bundle_dir / "experiment.prepare.yaml")

    # ---- version-2 contract ------------------------------------------------------------------
    # mmCIF alongside PDB: PDB's CRYST1 record cannot round-trip a triclinic cell faithfully, and
    # a bundle should not force a consumer through that limitation to learn its box.
    _write_topology_cif(bundle_dir)
    _write_v2_artifacts(
        bundle_dir, cfg=cfg, system=system, experiment=experiment,
        canonical=canonical, original_config=original_config, original_input=original_input,
    )
    provenance.write_json(bundle_dir / "resolved_config.json", cfg)

    manifest = build_bundle_manifest(
        bundle_dir, system, experiment, cfg, simbox_info=info, equilibration_info=eq,
        config_hash_value=chash,
    )
    provenance.write_json(bundle_dir / BUNDLE_MANIFEST, manifest)
    return bundle_dir


def bundlev2_module():
    from . import bundlev2

    return bundlev2


def _v2_manifest_block(bundle_dir: Path, cfg: dict) -> dict[str, Any]:
    """The version-2 additions: counts kept apart, roles mapped to paths, and stated limitations."""
    from openmm import XmlSerializer, app

    from . import bundlev2

    system_xml = bundle_dir / "system.xml"
    counts: dict[str, Any] = {}
    if system_xml.is_file() and (bundle_dir / "topology.pdb").is_file():
        omm_system = XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
        topology = app.PDBFile(str(bundle_dir / "topology.pdb")).topology
        counts = bundlev2.topology_counts(topology, omm_system)

    roles = {role: rel for role, rel in bundlev2.REQUIRED_ROLES.items()
             if (bundle_dir / rel).is_file()}
    originals = bundle_dir / bundlev2.ORIGINAL_INPUTS_DIR
    if originals.is_dir():
        roles["original_inputs"] = bundlev2.ORIGINAL_INPUTS_DIR
    if (bundle_dir / bundlev2.CHECKSUMS_FILE).is_file():
        roles["checksums"] = bundlev2.CHECKSUMS_FILE

    return {
        "bundle_id": bundle_dir.name,
        "package_version": provenance.package_version(),
        "source_commit": (provenance.git_state() or {}).get("commit"),
        "counts": counts,
        "roles": dict(sorted(roles.items())),
        # Named distinctly from the v1 `seeds` block, which the manifest also carries: two keys
        # spelled the same would have one silently shadow the other.
        "resolved_stage_seeds": (cfg.get("_canonical") or {}).get("stage_seeds"),
        "resolved_stage_seed_sources": (cfg.get("_canonical") or {}).get("stage_seed_sources"),
        "canonical_master_seed": (cfg.get("_canonical") or {}).get("master_seed"),
        "portability": {
            "prepared_artifacts": "byte-identical after transfer when the checksums match",
            "binary_checkpoints": "environment-specific; never assume they are portable",
            "serialized_state": "a physically valid portable fallback, but not a bitwise "
                                "continuation of stochastic dynamics",
            "rebuild_from_inputs": "may be scientifically consistent without being bitwise "
                                   "identical unless the recorded environment is reproduced",
        },
    }


def _write_topology_cif(bundle_dir: Path) -> None:
    """Write topology.cif from topology.pdb, preserving atom order and the periodic box."""
    from openmm import app

    pdb = app.PDBFile(str(bundle_dir / "topology.pdb"))
    with (bundle_dir / "topology.cif").open("w") as fh:
        app.PDBxFile.writeFile(pdb.topology, pdb.positions, fh)


def _write_v2_artifacts(bundle_dir: Path, *, cfg: dict, system, experiment,
                        canonical: Optional[dict], original_config: Optional[Path],
                        original_input: Optional[Path]) -> list[str]:
    """Original inputs, provenance and checksums. Returns the checksummed relative paths."""
    from openmm import XmlSerializer, app

    from . import bundlev2

    originals = bundle_dir / bundlev2.ORIGINAL_INPUTS_DIR
    originals.mkdir(exist_ok=True)

    # The EXACT documents the user supplied, so the bundle can be understood without them.
    if original_config is not None and Path(original_config).is_file():
        shutil.copy2(original_config, originals / Path(original_config).name)
    if original_input is not None and Path(original_input).is_file():
        shutil.copy2(original_input, originals / Path(original_input).name)
    if system.route == "smiles":
        # There is no input FILE for an inline SMILES, so record the identity instead: what was
        # declared, what it canonicalises to, and the hash that ties the two together.
        provenance.write_json(originals / "input_smiles.json", {
            "declared_smiles": system.doc["input"].get("smiles"),
            "canonical_isomeric_smiles": system.doc["input"].get("canonical_isomeric_smiles"),
            "canonical_molecular_sha256": system.canonical_hash,
            "expected_formal_charge": system.formal_charge,
        })
    # the legacy front end's exact manifests, when that is how the bundle was made
    if canonical is None:
        shutil.copy2(system.source, originals / "legacy_system.yaml")
        shutil.copy2(experiment.source, originals / "legacy_experiment.yaml")

    provenance.write_json(bundle_dir / "resolved_runtime_config.json", cfg)
    provenance.write_json(bundle_dir / "forcefield_provenance.json",
                          bundlev2.forcefield_provenance(cfg))
    provenance.write_json(bundle_dir / "environment.json", bundlev2.environment_provenance())
    if canonical is None:
        # A legacy-front-end bundle still gets a canonical record, by migrating its manifests.
        from ..core.config import canonical as canon_mod
        from ..core.config import migrate as migrate_mod
        from ..core.config import resolve as resolve_mod

        try:
            doc, notes = migrate_mod.migrate_manifests(system.doc, experiment.doc)
            result = resolve_mod.resolve_spec(doc)
            canonical = {"profile": result["profile"], "hashes": result["hashes"],
                         "sources": result["sources"], "migrated_from": "legacy manifests",
                         "front_end": "legacy",
                         "migration_notes": notes,
                         "configuration": canon_mod.to_plain(canon_mod.dump_model(result["spec"]))}
        except Exception as exc:                             # noqa: BLE001
            canonical = {"unavailable": f"{type(exc).__name__}: {exc}",
                         "note": "this bundle was prepared from legacy manifests that could not be "
                                 "migrated automatically; it does not carry a canonical projection"}
    provenance.write_json(bundle_dir / "canonical_configuration.json", canonical)

    checksummed = [rel for rel in bundlev2.REQUIRED_ROLES.values()
                   if (bundle_dir / rel).is_file()]
    checksummed += [f"{bundlev2.ORIGINAL_INPUTS_DIR}/{p.name}"
                    for p in sorted(originals.iterdir()) if p.is_file()]
    for extra in ("system.yaml", "experiment.prepare.yaml"):
        if (bundle_dir / extra).is_file():
            checksummed.append(extra)
    bundlev2.write_checksums(bundle_dir, checksummed)
    return checksummed


def build_bundle_manifest(
    bundle_dir: Path,
    system: SystemManifest,
    experiment: ExperimentManifest,
    cfg: dict,
    *,
    simbox_info: dict,
    equilibration_info: dict,
    config_hash_value: str,
) -> dict[str, Any]:
    """The provenance object. Every field here is one a reader would otherwise have to guess."""
    counts = _composition(bundle_dir / "topology.pdb", simbox_info)
    solv = cfg["solvation"]
    v2 = _v2_manifest_block(bundle_dir, cfg)
    return {
        # INDEPENDENT of the system, experiment, canonical-config and run-state versions: they
        # describe different things and must be free to move separately.
        "bundle_schema_version": bundlev2_module().BUNDLE_SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,      # legacy field, retained for v1 readers
        "kind": "explicit-solvent-rest2-bundle",
        **v2,
        "config_hash": config_hash_value,
        "created_utc": provenance.utc_timestamp(),
        "invocation": provenance.invocation(),
        "system": {
            "system_id": system.system_id,
            "display_name": system.display_name,
            "route": system.route,
            "canonical_molecular_hash": system.canonical_hash,
            "canonical_isomeric_smiles": system.doc["input"].get("canonical_isomeric_smiles"),
            "formal_charge": system.formal_charge,
        },
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "ladder_status": experiment.ladder_status,
            "n_rungs": experiment.n_rungs,
            "scale_factors": experiment.scale_factors,
            "master_seed": experiment.master_seed,
        },
        # What makes this the System it is. A launch-time --experiment override is compared
        # against the PROJECTION, not merely against the fingerprint, so rehashing an edited
        # manifest cannot bypass the check.
        "prepared_system": {
            "fingerprint": fingerprint(cfg),
            "projection": build_projection(cfg),
            "note": "every setting whose change would make the stored system.xml and "
                    "equilibrated_state.xml the wrong artifacts; runtime-only settings "
                    "(duration, chunking, reporting, platform, device, ladder) are excluded",
        },
        "parameterization": dict(system.doc["parameterization"]),
        "solvation": {
            "water_model": solv["water_model"],
            "box_shape": solv["box_shape"],
            "padding_nm": solv["padding_nm"],
            "requested_ionic_strength_molar": solv["ionic_strength_molar"],
            "positive_ion": solv["positive_ion"],
            "negative_ion": solv["negative_ion"],
            "neutralize": solv["neutralize"],
            "realized_ion_counts": counts["ions"],
            "salt": counts["salt"],
        },
        "composition": {
            "n_atoms": counts["n_atoms"],
            "n_solute_atoms": counts["n_solute_atoms"],
            "n_water_molecules": counts["n_water_molecules"],
            "n_ions": counts["n_ions"],
            "n_constraints": counts["n_constraints"],
            "n_degrees_of_freedom": counts["n_degrees_of_freedom"],
        },
        "box_vectors_nm": counts["box_vectors_nm"],
        # the minimum-image margin rule, and what it actually produced
        "box_geometry": simbox_info.get("geometry", {}),
        "omega": {
            "selective_scaling": cfg["rest2"]["omega_exclusion"],
            "excluded_central_bonds": simbox_info.get("omega_bonds")
            or simbox_info.get("omega", {}).get("central_bonds"),
            "note": "solute torsions about these bonds are NOT REST2-scaled; the cis/trans "
                    "equilibrium they control must not be moved by the ladder",
        },
        "seeds": {
            "master": int(cfg["run"]["seed"]),
            "structure_etkdg": cfg["structure"]["etkdg"]["seed"],
            "equilibration": cfg["equilibration"]["seed"],
            "production_remd": cfg["production"]["remd"]["seed"],
        },
        "equilibration": {
            "protocol": cfg["equilibration"]["protocol"],
            "summary": {k: v for k, v in (equilibration_info or {}).items()
                        if not isinstance(v, (dict, list))},
        },
        "environment": provenance.environment_block(),
        "files": provenance.hash_tree(bundle_dir, list(BUNDLE_FILES)),
        "reproducibility": {
            "rebuild": "Rebuilding from the same system manifest is scientifically consistent "
                       "but not bitwise identical: AM1-BCC charges, conformer embedding and water "
                       "placement depend on library versions and host floating point.",
            "transfer": "Transfer this bundle when the System and starting state must be "
                        "identical, e.g. when pooling runs produced on several machines.",
        },
    }


def _composition(topology_pdb: Path, simbox_info: dict) -> dict[str, Any]:
    """Atom/water/ion/constraint/DOF counts and box vectors, read from the built System.

    Read back from the artifacts rather than copied from the builder's return value, so the
    manifest describes the files that are actually in the bundle.
    """
    from openmm import XmlSerializer, app, unit

    system_xml = topology_pdb.parent / "system.xml"
    with open(system_xml, encoding="utf-8") as fh:
        system = XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(str(topology_pdb))
    top = pdb.topology

    water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"}
    ion_counts: dict[str, int] = {}
    n_water = 0
    for res in top.residues():
        name = res.name.strip().upper()
        if name in water_names:
            n_water += 1
        elif len(res) == 1 and name not in water_names:
            atom = next(iter(res.atoms()))
            el = atom.element.symbol if atom.element is not None else name
            if el.upper() in {"NA", "CL", "K", "MG", "CA", "ZN", "BR", "I", "LI", "RB", "CS"}:
                ion_counts[el] = ion_counts.get(el, 0) + 1

    n_atoms = system.getNumParticles()
    n_constraints = system.getNumConstraints()
    n_massless = sum(1 for i in range(n_atoms)
                     if system.getParticleMass(i).value_in_unit(unit.dalton) == 0.0)
    # 3N - constraints - 3 for the removed centre-of-mass motion, massless sites excluded
    has_cmm = any(type(system.getForce(i)).__name__ == "CMMotionRemover"
                  for i in range(system.getNumForces()))
    dof = 3 * (n_atoms - n_massless) - n_constraints - (3 if has_cmm else 0)

    box = system.getDefaultPeriodicBoxVectors()
    box_nm = [[float(v.value_in_unit(unit.nanometer)) for v in row] for row in box]
    n_ions = sum(ion_counts.values())
    # NOT n_ions/2: that treats every ion as half a salt pair, so a box whose only ions are
    # neutralising counterions reports a salt concentration it does not have.
    salt = salt_accounting(
        ion_counts,
        n_water=n_water,
        volume_nm3=abs(_det3(box_nm)),
        solute_formal_charge=simbox_info.get("solute_formal_charge"),
        positive_ion=simbox_info.get("positive_ion"),
        negative_ion=simbox_info.get("negative_ion"),
        requested_molar=simbox_info.get("requested_ionic_strength_molar"),
    )

    return {
        "n_atoms": int(n_atoms),
        "n_solute_atoms": int(simbox_info.get("n_solute_atoms", 0)),
        "n_water_molecules": int(n_water),
        "n_ions": int(n_ions),
        "ions": ion_counts,
        "n_constraints": int(n_constraints),
        "n_degrees_of_freedom": int(dof),
        "box_vectors_nm": box_nm,
        "salt": salt,
    }


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Recompute every hash and check internal consistency. Raises `BundleError` on any mismatch.

    Returns the parsed manifest on success, so a caller can go straight on to using it.
    """
    bundle_dir = Path(bundle_dir).resolve()
    if not bundle_dir.is_dir():
        raise BundleError(f"not a directory: {bundle_dir}")
    mpath = bundle_dir / BUNDLE_MANIFEST
    if not mpath.is_file():
        raise BundleError(f"{bundle_dir}: {BUNDLE_MANIFEST} is missing; this is not a bundle")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{mpath}: not valid JSON: {exc}") from exc

    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise BundleError(f"{mpath}: 'files' block is missing or malformed")

    missing = [name for name in BUNDLE_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise BundleError(f"{bundle_dir}: missing bundle file(s): {', '.join(sorted(missing))}")

    undeclared = sorted(set(BUNDLE_FILES) - set(declared))
    if undeclared:
        raise BundleError(
            f"{mpath}: no hash recorded for {', '.join(undeclared)}; the manifest is incomplete"
        )

    bad: list[str] = []
    for name, want in sorted(declared.items()):
        path = bundle_dir / name
        if not path.is_file():
            bad.append(f"{name}: recorded in the manifest but missing from the bundle")
            continue
        got = sha256_file(path)
        if got != want:
            bad.append(f"{name}: sha256 {got} does not match the recorded {want}")
    if bad:
        raise BundleError(
            f"{bundle_dir}: bundle contents do not match {BUNDLE_MANIFEST}:\n  "
            + "\n  ".join(bad)
            + "\nThe bundle has been modified or copied incompletely; do not run from it."
        )

    # the manifest must also agree with the manifests it carries, not merely with itself
    import yaml

    sys_doc = yaml.safe_load((bundle_dir / "system.yaml").read_text(encoding="utf-8"))
    exp_doc = yaml.safe_load((bundle_dir / "experiment.prepare.yaml").read_text(encoding="utf-8"))
    if str(sys_doc.get("system_id")) != str(manifest["system"]["system_id"]):
        raise BundleError(
            f"{bundle_dir}: system.yaml declares system_id "
            f"{sys_doc.get('system_id')!r} but the bundle manifest records "
            f"{manifest['system']['system_id']!r}"
        )
    prepared = manifest.get("prepared_system") or {}
    if prepared.get("projection") is not None:
        recomputed_fp = sha256_text(canonical_json(prepared["projection"]))
        if recomputed_fp != prepared.get("fingerprint"):
            raise BundleError(
                f"{bundle_dir}: the recorded prepared-system projection does not hash to the "
                f"recorded fingerprint (recorded {prepared.get('fingerprint')}, recomputed "
                f"{recomputed_fp}); the bundle manifest has been edited"
            )

    recomputed = config_hash(sys_doc, exp_doc)
    if recomputed != manifest.get("config_hash"):
        raise BundleError(
            f"{bundle_dir}: config_hash {manifest.get('config_hash')} does not match the "
            f"manifests in the bundle (recomputed {recomputed}); one of them was edited "
            "after preparation"
        )
    return manifest


def summarise(manifest: dict) -> str:
    """One screen of the things a reader checks first."""
    s, e, c = manifest["system"], manifest["experiment"], manifest["composition"]
    env = manifest["environment"]
    lines = [
        f"system      {s['system_id']}  ({s['display_name']})",
        f"  route     {s['route']}   formal charge {s['formal_charge']}",
        f"  molecule  {s['canonical_molecular_hash']}",
        f"experiment  {e['experiment_id']}   {e['n_rungs']} rungs   "
        f"ladder_status={e['ladder_status']}",
        f"config hash {manifest['config_hash']}",
        f"composition {c['n_atoms']} atoms  ({c['n_solute_atoms']} solute, "
        f"{c['n_water_molecules']} waters, {c['n_ions']} ions)  "
        f"{c['n_degrees_of_freedom']} DOF",
        f"built       {manifest['created_utc']}  md-templates {env['package_version']}  "
        f"openmm {env['toolchain'].get('openmm')}",
    ]
    if e["ladder_status"] != "validated":
        lines.append(
            "  NOTE: this ladder is NOT validated for this system; treat any number from it as "
            "a pilot."
        )
    return "\n".join(lines)
