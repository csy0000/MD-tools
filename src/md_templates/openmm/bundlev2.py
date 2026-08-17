"""The version-2 transferable bundle contract.

A bundle is a self-contained scientific artifact. Once prepared it must remain meaningful after it
is copied away from the project that made it and from the machine that made it, so nothing here may
depend on an absolute path, on the source checkout, on the original inputs, or on the network.

Version 2 adds, over version 1:

    checksums.json              every required artifact, by normalised relative POSIX path
    original_inputs/            the exact documents the user supplied
    forcefield_provenance.json  which package supplied each parameter resource, and its hash
    environment.json            versions, classified by what they are needed FOR
    topology.cif                a box representation without PDB's format limits
    counts                      topology atoms, OpenMM particles, virtual sites and massless
                                particles recorded SEPARATELY

The bundle schema version is independent of the system, experiment, canonical-configuration and
run-state versions: they describe different things and must be free to move separately. A version-1
bundle stays readable and runnable, and is reported as NOT satisfying this contract rather than
being quietly relabelled.

Integrity, not authenticity: the checksums detect a changed or missing file. They are not
signatures and must not be described as such.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "BUNDLE_SCHEMA_VERSION", "CHECKSUMS_FILE", "ORIGINAL_INPUTS_DIR", "REQUIRED_ROLES",
    "BundleContractError", "normalise_relative", "sha256_file", "write_checksums",
    "verify_checksums", "topology_counts", "forcefield_provenance", "environment_provenance",
]

#: Independent of every other schema version in the package.
BUNDLE_SCHEMA_VERSION = 2

CHECKSUMS_FILE = "checksums.json"
ORIGINAL_INPUTS_DIR = "original_inputs"

#: logical role -> conventional relative path. The manifest maps roles to paths so a consumer never
#: has to know a filename convention.
REQUIRED_ROLES: dict[str, str] = {
    "system_xml": "system.xml",
    "topology_pdb": "topology.pdb",
    "topology_cif": "topology.cif",
    "simbox": "simbox.json",
    "equilibrated_state": "equilibrated_state.xml",
    "canonical_configuration": "canonical_configuration.json",
    "resolved_runtime_config": "resolved_runtime_config.json",
    "forcefield_provenance": "forcefield_provenance.json",
    "environment": "environment.json",
}


class BundleContractError(RuntimeError):
    """A bundle violates the version-2 contract."""


# ---------------------------------------------------------------------------------------------
# paths and checksums
# ---------------------------------------------------------------------------------------------

def normalise_relative(path: str | Path, *, field: str = "path") -> str:
    """A bundle-relative POSIX path, or an error saying why it is not portable.

    Absolute paths, parent traversal, drive letters and backslashes are all refused rather than
    normalised away: each of them means the bundle would resolve differently -- or not at all --
    somewhere else, which is precisely what this contract exists to prevent.
    """
    raw = str(path)
    if "\\\\" in raw or "\\" in raw:
        raise BundleContractError(
            f"{field}: {raw!r} contains a backslash. Bundle paths are POSIX-relative so they mean "
            "the same thing on every platform."
        )
    if posixpath.isabs(raw) or (len(raw) > 1 and raw[1] == ":"):
        raise BundleContractError(
            f"{field}: {raw!r} is absolute. A bundle may not depend on a location outside itself."
        )
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise BundleContractError(
            f"{field}: {raw!r} escapes the bundle with '..'. Every required file must live inside."
        )
    normalised = posixpath.join(*parts) if parts else ""
    if not normalised:
        raise BundleContractError(f"{field}: {raw!r} is not a usable relative path")
    return normalised


def sha256_file(path: Path) -> str:
    """Hash BYTES. Parsing and re-serialising would make a changed file hash the same."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(bundle_dir: Path, relative_paths: list[str]) -> Path:
    """One versioned checksums file over an explicit domain.

    Domain, stated so it cannot be self-referential or ambiguous:

    * INCLUDED: every required scientific artifact and every original input;
    * EXCLUDED: `checksums.json` itself, which cannot contain its own hash;
    * EXCLUDED: `bundle_manifest.json`, because the manifest records the checksum file's own hash
      and including it would make the two mutually dependent. The manifest is covered by its own
      recorded hash instead.
    """
    bundle_dir = Path(bundle_dir)
    entries: dict[str, str] = {}
    for raw in relative_paths:
        rel = normalise_relative(raw, field="checksums entry")
        if rel in entries:
            raise BundleContractError(f"checksums: {rel!r} listed twice")
        target = bundle_dir / rel
        if not target.is_file():
            raise BundleContractError(f"checksums: {rel} does not exist in {bundle_dir}")
        entries[rel] = sha256_file(target)

    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "algorithm": "sha256",
        "domain": {
            "includes": "every required scientific artifact and original input",
            "excludes": [CHECKSUMS_FILE, "bundle_manifest.json"],
            "excluded_because": (
                f"{CHECKSUMS_FILE} cannot contain its own hash; bundle_manifest.json records this "
                "file's hash, so including it would make the two mutually dependent"
            ),
            "note": "integrity, not authenticity: these detect change, they are not signatures",
        },
        "files": dict(sorted(entries.items())),
    }
    from . import runstate

    return runstate.atomic_write_json(bundle_dir / CHECKSUMS_FILE, payload)


def verify_checksums(bundle_dir: Path) -> dict[str, list[str]]:
    """Return `{"missing", "changed", "extra_required"}`; empty lists mean the bundle is intact."""
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / CHECKSUMS_FILE
    if not path.is_file():
        raise BundleContractError(f"{path} is missing; this is not a version-2 bundle")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleContractError(
            f"{path}: schema_version {doc.get('schema_version')!r}; this build writes "
            f"{BUNDLE_SCHEMA_VERSION}"
        )
    missing, changed = [], []
    for rel, expected in (doc.get("files") or {}).items():
        target = bundle_dir / rel
        if not target.is_file():
            missing.append(rel)
        elif sha256_file(target) != expected:
            changed.append(rel)
    return {"missing": sorted(missing), "changed": sorted(changed), "extra_required": []}


# ---------------------------------------------------------------------------------------------
# counts: topology atoms and OpenMM particles are DIFFERENT numbers
# ---------------------------------------------------------------------------------------------

def topology_counts(topology, system) -> dict[str, Any]:
    """Atom, particle, virtual-site, massless and constraint counts, kept apart.

    A virtual site is an OpenMM particle with no corresponding topology atom in the general case,
    and a massless particle is not a degree of freedom. Reporting one number called `n_atoms` hides
    both distinctions, and a four-site water model makes them disagree immediately.
    """
    from openmm import unit

    n_topology_atoms = sum(1 for _ in topology.atoms())
    n_particles = system.getNumParticles()
    n_virtual = sum(1 for i in range(n_particles) if system.isVirtualSite(i))
    n_massless = sum(1 for i in range(n_particles)
                     if system.getParticleMass(i).value_in_unit(unit.dalton) == 0.0)
    n_constraints = system.getNumConstraints()
    has_cmm = any(type(system.getForce(i)).__name__ == "CMMotionRemover"
                  for i in range(system.getNumForces()))
    dof = 3 * (n_particles - n_massless) - n_constraints - (3 if has_cmm else 0)
    return {
        "topology_atoms": int(n_topology_atoms),
        "openmm_particles": int(n_particles),
        "virtual_sites": int(n_virtual),
        "massless_particles": int(n_massless),
        "constraints": int(n_constraints),
        "degrees_of_freedom": int(dof),
        "degrees_of_freedom_formula":
            "3*(openmm_particles - massless_particles) - constraints - 3 if CMMotionRemover",
        "topology_atoms_equal_particles": bool(n_topology_atoms == n_particles),
        "note": "topology atoms and OpenMM particles are counted separately; virtual sites make "
                "them differ, and massless particles are not degrees of freedom",
    }


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------

def _distribution_of(module_name: str) -> dict[str, Any]:
    from importlib import metadata

    try:
        module = __import__(module_name)
    except Exception:                                       # noqa: BLE001
        return {"available": False}
    info: dict[str, Any] = {"available": True,
                            "version": getattr(module, "__version__", None)}
    for dist in ("openmm", "openmmforcefields", "openff-toolkit", "rdkit"):
        try:
            if dist.replace("-", "_").startswith(module_name):
                info["distribution"] = dist
                info["distribution_version"] = metadata.version(dist)
        except Exception:                                   # noqa: BLE001
            pass
    return info


def forcefield_provenance(cfg: dict, *, extra_files: Optional[list[Path]] = None) -> dict[str, Any]:
    """Which package supplied each parameter resource, and the hash of the exact file used.

    Recording a package NAME does not identify a force field: two environments can both say
    `openmmforcefields` and ship different XML. Where the resource can be located on disk its bytes
    are hashed, so a target environment supplying a different file is detectable rather than
    assumed away.
    """
    ff = cfg.get("forcefield", {})
    resources: list[dict[str, Any]] = []

    def _record(role: str, identifier: Optional[str]) -> None:
        if not identifier:
            return
        entry: dict[str, Any] = {"role": role, "identifier": identifier, "resolved_path": None,
                                 "sha256": None, "supplied_by": None}
        try:
            from openmm import app

            for base in app.forcefield._getDataDirectories():   # noqa: SLF001
                candidate = Path(base) / identifier
                if candidate.is_file():
                    entry["resolved_path"] = candidate.name
                    entry["sha256"] = sha256_file(candidate)
                    entry["supplied_by"] = "openmm"
                    break
        except Exception:                                    # noqa: BLE001
            pass
        resources.append(entry)

    _record("protein", ff.get("protein"))
    _record("water", ff.get("water"))
    for extra in ff.get("extra_xml") or []:
        _record("extra", extra)

    small_molecule = {
        "identifier": ff.get("ligand"),
        "charge_method": ff.get("ligand_charge_method"),
        "supplied_by": "openmmforcefields/openff-toolkit",
        "note": "SMIRNOFF parameters are generated at build time from the named force field; the "
                "toolkit versions below identify which implementation produced them",
    } if ff.get("ligand") else None

    user_files = []
    for path in extra_files or []:
        p = Path(path)
        if p.is_file():
            user_files.append({"name": p.name, "sha256": sha256_file(p)})

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "resources": resources,
        "small_molecule": small_molecule,
        "user_supplied_files": user_files,
        "combination_assumptions":
            "protein and water parameters are combined by OpenMM's ForceField in the order listed; "
            "small-molecule parameters are added by a SMIRNOFF/GAFF template generator. Mixing "
            "rules are those of the underlying force fields and are not modified here.",
        "toolkits": {name: _distribution_of(name)
                     for name in ("openmm", "openff", "rdkit", "openmmforcefields")},
        "limitation": "a package name alone does not identify a force field; where a resource "
                      "could not be located on disk its sha256 is null and only its identifier "
                      "and supplying package are recorded",
    }


def environment_provenance() -> dict[str, Any]:
    """Versions, classified by what they are needed FOR.

    A bundle transfer uses the stored System and State, so a different environment can still run
    it. Rebuilding from the original inputs is a different operation with stricter requirements,
    and the classification says which is which instead of listing versions without a purpose.
    """
    import platform
    import sys
    from importlib import metadata

    def _version(dist: str) -> Optional[str]:
        try:
            return metadata.version(dist)
        except Exception:                                    # noqa: BLE001
            return None

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "required_for_exact_rebuild": {
            "python": sys.version.split()[0],
            "openmm": _version("openmm"),
            "openff-toolkit": _version("openff-toolkit"),
            "openmmforcefields": _version("openmmforcefields"),
            "rdkit": _version("rdkit"),
            "ambertools_sqm": "required for AM1-BCC charges on the smiles route",
        },
        "required_for_supported_execution": {
            "python_minimum": "3.11",
            "openmm": _version("openmm"),
            "md_templates": _version("md-templates"),
        },
        "informational": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "note": "no absolute path from this environment is recorded; a bundle must not depend on "
                "one. Rebuilding from inputs is not expected to be bitwise identical unless the "
                "exact environment above is reproduced.",
    }
