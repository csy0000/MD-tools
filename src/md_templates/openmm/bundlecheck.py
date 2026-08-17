"""Validation, inspection and relocation checking for prepared bundles.

The question these answer is not "does this look like a bundle" but "would this still be the same
calculation somewhere else". So validation reads bytes rather than filenames, refuses any required
reference that points outside the bundle, and never needs the network or the source checkout.

Default validation does NOT construct an OpenMM Context: a portability check that requires a
working simulation environment cannot be run in the places it is most needed. `--deep` deserialises
the System and State to cross-check counts, box and constraints, but still propagates nothing.

A version-1 bundle stays readable and is reported as NOT satisfying the version-2 contract, with
the missing guarantees named. Relabelling it would be the one failure this whole contract exists to
prevent.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import bundlev2

__all__ = ["validate_bundle_v2", "inspect_bundle", "relocate_check", "BundleReport"]


class BundleReport(dict):
    """A validation result: `ok`, plus `errors`, `warnings` and the facts they were drawn from."""

    @property
    def ok(self) -> bool:
        return not self.get("errors")


def _load(bundle_dir: Path, name: str) -> dict:
    path = Path(bundle_dir) / name
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise bundlev2.BundleContractError(f"{path}: unreadable JSON: {exc}") from None


def _scan_for_absolute_references(bundle_dir: Path) -> list[str]:
    """Required JSON records must not carry an absolute path to anything they need.

    Provenance may legitimately mention where something came from, so only the records a consumer
    must READ to use the bundle are scanned, and only their operational fields.
    """
    offenders: list[str] = []
    operational = ("roles", "files")
    for name in ("bundle_manifest.json", bundlev2.CHECKSUMS_FILE):
        doc = _load(bundle_dir, name)
        for key in operational:
            section = doc.get(key)
            if not isinstance(section, dict):
                continue
            for role, value in section.items():
                if isinstance(value, str):
                    try:
                        bundlev2.normalise_relative(value, field=f"{name}:{key}.{role}")
                    except bundlev2.BundleContractError as exc:
                        offenders.append(str(exc))
    return offenders


def validate_bundle_v2(bundle_dir: Path, *, deep: bool = False) -> BundleReport:
    """Schema, roles, paths, checksums, hashes, profile snapshot and counts."""
    bundle_dir = Path(bundle_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    manifest = _load(bundle_dir, "bundle_manifest.json")
    if not manifest:
        errors.append("bundle_manifest.json is missing or unreadable")
        return BundleReport(ok=False, errors=errors, warnings=warnings, bundle=str(bundle_dir))

    version = manifest.get("bundle_schema_version")
    if version is None:
        # Not an error: a version-1 bundle is still readable. It simply does not make the
        # version-2 promises, and saying so is the point.
        warnings.append(
            "this bundle predates the version-2 contract (no bundle_schema_version). It may still "
            "run through the documented compatibility path, but it does NOT provide: checksums, "
            "original inputs, force-field provenance, environment provenance, separated "
            "topology/particle counts, or an mmCIF topology."
        )
        return BundleReport(ok=True, errors=errors, warnings=warnings, contract="v1-compatibility",
                            bundle=str(bundle_dir), manifest=manifest)
    if version != bundlev2.BUNDLE_SCHEMA_VERSION:
        errors.append(
            f"bundle_schema_version {version!r}; this build writes "
            f"{bundlev2.BUNDLE_SCHEMA_VERSION}"
        )

    roles = manifest.get("roles") or {}
    for role, rel in bundlev2.REQUIRED_ROLES.items():
        target = roles.get(role)
        if target is None:
            errors.append(f"required logical role {role!r} is not mapped in the manifest")
        elif not (bundle_dir / target).exists():
            errors.append(f"role {role!r} maps to {target!r}, which does not exist")

    errors.extend(_scan_for_absolute_references(bundle_dir))

    try:
        result = bundlev2.verify_checksums(bundle_dir)
        for rel in result["missing"]:
            errors.append(f"checksummed file missing: {rel}")
        for rel in result["changed"]:
            errors.append(f"checksum mismatch (file changed): {rel}")
    except bundlev2.BundleContractError as exc:
        errors.append(str(exc))

    canonical = _load(bundle_dir, "canonical_configuration.json")
    profile = (canonical or {}).get("profile") or {}
    if not profile.get("profile_id"):
        warnings.append("no profile snapshot recorded; the defaults behind this bundle are not "
                        "identifiable")
    else:
        from ..core.config import canonical as canon_mod
        from ..core.config import resolve as resolve_mod

        try:
            live = resolve_mod.load_profile(profile["profile_id"])
            live_hash = canon_mod.sha256_of({k: v for k, v in live.items() if k != "_path"})
            if profile.get("sha256") and live_hash != profile["sha256"]:
                warnings.append(
                    f"profile {profile['profile_id']} in this installation hashes {live_hash[:12]} "
                    f"but the bundle recorded {profile['sha256'][:12]}: the defaults differ from "
                    "those the bundle was prepared with"
                )
        except Exception as exc:                            # noqa: BLE001
            warnings.append(f"profile {profile.get('profile_id')!r} is not available here: {exc}")

    counts = manifest.get("counts") or {}
    if counts and counts.get("openmm_particles") is None:
        errors.append("counts.openmm_particles is missing")

    if deep:
        errors.extend(_deep_checks(bundle_dir, manifest))

    return BundleReport(ok=not errors, errors=errors, warnings=warnings,
                        contract=f"v{bundlev2.BUNDLE_SCHEMA_VERSION}", bundle=str(bundle_dir),
                        manifest=manifest)


def _deep_checks(bundle_dir: Path, manifest: dict) -> list[str]:
    """Deserialise and cross-check. Still constructs no Context and propagates nothing."""
    errors: list[str] = []
    try:
        from openmm import XmlSerializer, app
    except ImportError:                                     # pragma: no cover
        return ["--deep needs OpenMM, which is not importable here"]

    try:
        system = XmlSerializer.deserialize((bundle_dir / "system.xml").read_text(encoding="utf-8"))
        pdb = app.PDBFile(str(bundle_dir / "topology.pdb"))
    except Exception as exc:                                # noqa: BLE001
        return [f"--deep could not read the stored artifacts: {type(exc).__name__}: {exc}"]

    live = bundlev2.topology_counts(pdb.topology, system)
    recorded = manifest.get("counts") or {}
    for key in ("topology_atoms", "openmm_particles", "virtual_sites", "massless_particles",
                "constraints"):
        if key in recorded and recorded[key] != live[key]:
            errors.append(f"--deep: {key} recorded {recorded[key]} but the artifacts give "
                          f"{live[key]}")

    cif = bundle_dir / "topology.cif"
    if cif.is_file():
        try:
            cif_top = app.PDBxFile(str(cif)).topology
            n_cif = sum(1 for _ in cif_top.atoms())
            if n_cif != live["topology_atoms"]:
                errors.append(f"--deep: topology.cif has {n_cif} atoms, topology.pdb has "
                              f"{live['topology_atoms']}")
            else:
                import numpy as np
                from openmm import unit

                a = np.array(pdb.topology.getPeriodicBoxVectors()
                             .value_in_unit(unit.nanometer))
                b = np.array(cif_top.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
                # PDB CRYST1 stores 3 decimals in angstrom, i.e. 1e-4 nm; compare volumes, which
                # are invariant to the reduced-form sign convention the two formats can differ in
                if abs(abs(np.linalg.det(a)) - abs(np.linalg.det(b))) > 2e-3:
                    errors.append("--deep: topology.pdb and topology.cif disagree on box volume "
                                  "beyond PDB's recording precision")
        except Exception as exc:                            # noqa: BLE001
            errors.append(f"--deep: topology.cif unreadable: {type(exc).__name__}: {exc}")
    return errors


def inspect_bundle(bundle_dir: Path) -> dict[str, Any]:
    """A summary a human can act on, without constructing anything in OpenMM."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = _load(bundle_dir, "bundle_manifest.json")
    canonical = _load(bundle_dir, "canonical_configuration.json")
    ff = _load(bundle_dir, "forcefield_provenance.json")
    env = _load(bundle_dir, "environment.json")
    report = validate_bundle_v2(bundle_dir)

    system = manifest.get("system") or {}
    par = manifest.get("parameterization") or {}
    method = ((canonical.get("configuration") or {}).get("protocol") or {}) \
        .get("production", {}).get("method")
    return {
        "bundle": str(bundle_dir),
        "bundle_schema_version": manifest.get("bundle_schema_version", 1),
        "bundle_id": manifest.get("bundle_id", bundle_dir.name),
        "created_utc": manifest.get("created_utc"),
        "package_version": manifest.get("package_version"),
        "identity": {
            "system_id": system.get("system_id"),
            "route": system.get("route"),
            "molecular_hash": system.get("canonical_molecular_hash"),
            "formal_charge": system.get("formal_charge"),
        },
        "methods_supported": ([method] if method else ["md", "rest2"]),
        "profile": (canonical.get("profile") or {}),
        "hashes": (canonical.get("hashes") or {}),
        "forcefields": {
            "protein": par.get("protein_forcefield"),
            "small_molecule": par.get("small_molecule_forcefield"),
            "water": par.get("water_forcefield"),
            "charge_method": par.get("charge_method"),
            "resources_hashed": sum(1 for r in (ff.get("resources") or []) if r.get("sha256")),
        },
        "counts": manifest.get("counts") or {},
        "box": {"vectors_nm": manifest.get("box_vectors_nm"),
                "geometry": manifest.get("box_geometry")},
        "composition": manifest.get("composition") or manifest.get("solvation"),
        "seeds": {"master": manifest.get("canonical_master_seed"),
                  "stages": manifest.get("resolved_stage_seeds")},
        "environment": {"required_for_supported_execution":
                        (env.get("required_for_supported_execution") or {})},
        "portability": manifest.get("portability"),
        "validation": {"ok": report.ok, "contract": report.get("contract"),
                       "errors": report.get("errors"), "warnings": report.get("warnings")},
    }


def relocate_check(bundle_dir: Path) -> dict[str, Any]:
    """Copy the bundle somewhere unrelated, validate it there, and report what escaped.

    The source bundle is never modified. Validating in place cannot detect a dependence on the
    original location, which is exactly the failure worth catching: the copy is the test.
    """
    source = Path(bundle_dir).resolve()
    before = validate_bundle_v2(source)
    tmp_root = Path(tempfile.mkdtemp(prefix="relocate-check-"))
    try:
        target = tmp_root / source.name
        shutil.copytree(source, target)
        after = validate_bundle_v2(target, deep=False)
        escaped = _scan_for_absolute_references(target)
        return {
            "source": str(source),
            "relocated_to": str(target),
            "source_ok": before.ok,
            "relocated_ok": after.ok,
            "errors_after_relocation": after.get("errors"),
            "warnings_after_relocation": after.get("warnings"),
            "references_escaping_the_bundle": escaped,
            "verdict": ("relocatable" if after.ok and not escaped else "NOT relocatable"),
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
