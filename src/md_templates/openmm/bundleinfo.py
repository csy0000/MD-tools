"""Bundle facts that require a built OpenMM system: counts and force-field provenance.

`md_templates.core.bundle` owns the parts of the bundle contract that are pure file format --
normalised relative paths, the checksum domain, writing and verifying `checksums.json`, the
environment record. Those need no engine and must stay importable where none is installed.

Counting particles and resolving force-field files do need one. A virtual site is an OpenMM particle
with no topology atom, a massless particle is not a degree of freedom, and only the toolkit can say
where a force-field XML actually resolved from. Keeping those here is what lets core stay engine-free
without either duplicating the logic or pretending the numbers can be derived from the manifest.

Re-exported through `md_templates.openmm.bundlev2` so the historical import path is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..core.bundle import (  # noqa: F401
    BUNDLE_SCHEMA_VERSION,
    BundleContractError,
    normalise_relative,
    sha256_file,
)

__all__ = ["topology_counts", "forcefield_provenance"]


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
