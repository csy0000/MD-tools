"""Capture the compatibility goldens that later migration PRs must not disturb.

Run deliberately, never as a side effect of a test:

    python scripts/capture_goldens.py            # write tests/goldens/*.json
    python scripts/capture_goldens.py --check    # compare without writing (what CI does)

The goldens freeze *contracts*, not artifacts. Where exact bytes would be platform- or
environment-dependent -- a serialized System, a trajectory, a checkpoint -- the fixture records the
semantic projection or an independently derived invariant instead, so a failure means the contract
moved rather than that the machine did.

Deterministic by construction: every input is either a shipped resource or a literal in this file,
nothing is timestamped, and nothing is read from a run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "goldens"

# Four representative configurations: both methods on both declared routes. They exercise the
# discriminated production models and both force-field routes, which is what the projections and
# profile selection actually branch on.
#
# Each document declares only what cannot be defaulted -- the system and the method discriminator --
# so everything else in the resulting hashes comes from the shipped default profile. A change to a
# profile therefore moves a golden, which is the point.
CONFIGS: dict[str, dict] = {
    "md_ligand_smiles": {
        "system": {"system_id": "golden_ligand", "route": "smiles", "smiles": "CCO"},
        "protocol": {"production": {"method": "md"}},
    },
    "md_peptide_pdb": {
        "system": {"system_id": "golden_peptide", "route": "pdb", "pdb": "input.pdb"},
        "protocol": {"production": {"method": "md"}},
    },
    "rest2_ligand_smiles": {
        "system": {"system_id": "golden_ligand", "route": "smiles", "smiles": "CCO"},
        "protocol": {"production": {"method": "rest2"}},
    },
    "rest2_peptide_pdb": {
        "system": {"system_id": "golden_peptide", "route": "pdb", "pdb": "input.pdb"},
        "protocol": {"production": {"method": "rest2"}},
    },
}


def _configuration_goldens() -> dict:
    """Canonical bytes, every projection hash, and the profile each configuration selects."""
    from md_templates.openmm.spec import canonical, resolve

    out: dict[str, dict] = {}
    for name, document in CONFIGS.items():
        result = resolve.resolve_spec(json.loads(json.dumps(document)))
        spec = result["spec"]
        production = spec.protocol.production
        out[name] = {
            "canonical_json_sha256": canonical.sha256_of(canonical.dump_model(spec)),
            "hashes": result["hashes"],
            "profile": {
                "profile_id": result["profile"]["profile_id"],
                "profile_schema_version": result["profile"]["profile_schema_version"],
                "sha256": result["profile"]["sha256"],
            },
            # a few resolved values whose drift would be scientifically meaningful
            "resolved": {
                "integrator.timestep_ps": spec.protocol.integrator.timestep.value,
                "integrator.temperature_K": spec.protocol.integrator.temperature.value,
                "nonbonded.cutoff_nm": spec.build.nonbonded.cutoff.value,
                "solvation.padding_nm": spec.build.solvation.padding.value,
                # None under the conservative default, where hydrogens are not repartitioned
                "hydrogen_mass_amu": (spec.build.hydrogen_mass.value
                                      if spec.build.hydrogen_mass else None),
                "constraints": spec.build.constraints,
                "production.method": production.method,
                "production.duration_per_segment_ps": production.duration_per_segment.value,
                "production.duration_per_segment_source": production.duration_per_segment.source,
                # tau is the SOURCE parameter; s is captured too so the golden pins the derived
                # ladder as well as the declaration that produced it.
                "production.tau_values": (production.tau_ladder.tau_values()
                                          if hasattr(production, "tau_ladder") else None),
                "production.derived_scale_factors": (production.scale_factors()
                                                     if hasattr(production, "tau_ladder") else None),
            },
        }
    return out


def _profile_goldens() -> dict:
    """Every packaged profile's identity and hash, plus which is default for each route/method."""
    from md_templates.openmm.spec import canonical, resolve

    profiles = {}
    for doc in resolve.list_profiles():
        clean = {k: v for k, v in doc.items() if k != "_path"}
        profiles[doc["profile_id"]] = {
            "profile_schema_version": doc["profile_schema_version"],
            "route": doc.get("route"),
            "method": doc.get("method"),
            "is_default": bool(doc.get("is_default")),
            "sha256": canonical.sha256_of(clean),
        }
    defaults = {}
    for route in ("smiles", "pdb"):
        for method in ("md", "rest2"):
            defaults[f"{route}/{method}"] = resolve.select_profile(route, method)["profile_id"]
    return {"profiles": profiles, "default_selection": defaults}


def _format_equivalence_golden() -> dict:
    """YAML and JSON expressing the same document must produce identical canonical output."""
    import tempfile

    import yaml

    from md_templates.openmm.spec import canonical, resolve

    document = CONFIGS["rest2_ligand_smiles"]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.yaml").write_text(yaml.safe_dump(document))
    (tmp / "a.json").write_text(json.dumps(document))
    from_yaml = resolve.resolve_spec(resolve.load_document(tmp / "a.yaml"))
    from_json = resolve.resolve_spec(resolve.load_document(tmp / "a.json"))
    yaml_bytes = canonical.canonical_json(canonical.dump_model(from_yaml["spec"]))
    json_bytes = canonical.canonical_json(canonical.dump_model(from_json["spec"]))
    return {
        "identical_canonical_bytes": yaml_bytes == json_bytes,
        "identical_hashes": from_yaml["hashes"] == from_json["hashes"],
        "canonical_sha256": canonical.sha256_of(canonical.dump_model(from_yaml["spec"])),
    }


def _seed_goldens() -> dict:
    """The master-seed to stage-seed derivation, which migrated inputs depend on exactly."""
    from md_templates.openmm.spec.models import STAGE_ORDER, RandomnessSpec

    cases = {}
    for label, kwargs in {
        "default": {},
        "master_20260814": {"master_seed": 20260814},
        "master_1000": {"master_seed": 1000},
        "explicit_md_seed": {"master_seed": 1000, "stage_seeds": {"md": 42}},
    }.items():
        spec = RandomnessSpec(**kwargs)
        cases[label] = {"master_seed": spec.master_seed, "resolved": spec.resolve(),
                        "sources": spec.sources()}
    return {"stage_order": list(STAGE_ORDER), "cases": cases,
            "rule": "stage seed = master_seed + index in stage_order, unless explicitly set"}


def _bundle_contract_golden() -> dict:
    """The v2 bundle's SHAPE: schema version, logical roles, count fields, checksum domain.

    Not a real bundle: building one needs OpenMM and produces artifacts too large and too
    environment-dependent to freeze. What later PRs must not break is the contract, so the contract
    is what is recorded.
    """
    from md_templates.openmm import bundlev2

    return {
        "bundle_schema_version": bundlev2.BUNDLE_SCHEMA_VERSION,
        "required_roles": dict(sorted(bundlev2.REQUIRED_ROLES.items())),
        "checksums_file": bundlev2.CHECKSUMS_FILE,
        "original_inputs_dir": bundlev2.ORIGINAL_INPUTS_DIR,
        "count_fields": sorted([
            "topology_atoms", "openmm_particles", "virtual_sites", "massless_particles",
            "constraints", "degrees_of_freedom", "degrees_of_freedom_formula",
            "topology_atoms_equal_particles", "note",
        ]),
    }


def _runstate_contract_golden() -> dict:
    """The continuity contract and committed-generation record structure."""
    from md_templates.openmm import runstate

    return {
        "run_state_schema": runstate.RUN_STATE_SCHEMA,
        "run_state_file": runstate.RUN_STATE_FILE,
        "restart_dir": runstate.RESTART_DIR,
        "committed_file": runstate.COMMITTED_FILE,
        "quarantine_dir": runstate.QUARANTINE_DIR,
        "continuity_paths": list(runstate.CONTINUITY_PATHS),
        "extension_paths": list(runstate.EXTENSION_PATHS),
        "restart_members_single": list(runstate.restart_members()),
        "restart_members_replica_0": list(runstate.restart_members(0)),
    }


def _rest2_defaults_golden() -> dict:
    """REST2 settings whose change would be a scientific change: ladder and omega exclusion."""
    from md_templates.openmm import DEFAULTS
    from md_templates.openmm.spec import resolve

    ligand = resolve.load_profile("explicit-rest2-ligand-v1")["defaults"]["protocol"]["production"]
    peptide = resolve.load_profile("explicit-rest2-peptide-v1")["defaults"]["protocol"]["production"]
    return {
        "runtime_defaults": {
            "omega_exclusion": DEFAULTS["rest2"]["omega_exclusion"],
            "proline_like_residues": DEFAULTS["rest2"]["proline_like_residues"],
            "max_proline_ring_size": DEFAULTS["rest2"]["max_proline_ring_size"],
        },
        "profile_ladders": {
            "explicit-rest2-ligand-v1": ligand["tau_ladder"],
            "explicit-rest2-peptide-v1": peptide["tau_ladder"],
        },
        "profile_omega_exclusion": {
            "explicit-rest2-ligand-v1": ligand["omega_exclusion"]["enabled"],
            "explicit-rest2-peptide-v1": peptide["omega_exclusion"]["enabled"],
        },
    }


GENERATORS = {
    "configuration_hashes.json": _configuration_goldens,
    "profiles.json": _profile_goldens,
    "format_equivalence.json": _format_equivalence_golden,
    "seed_derivation.json": _seed_goldens,
    "bundle_contract.json": _bundle_contract_golden,
    "runstate_contract.json": _runstate_contract_golden,
    "rest2_defaults.json": _rest2_defaults_golden,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed goldens instead of writing them")
    args = parser.parse_args(argv)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, generator in GENERATORS.items():
        payload = generator()
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        target = GOLDEN_DIR / name
        if args.check:
            if not target.is_file():
                failures.append(f"{name}: missing")
            elif target.read_text() != text:
                failures.append(f"{name}: DIFFERS from the committed golden")
            else:
                print(f"  ok       {name}")
        else:
            target.write_text(text)
            print(f"  written  {name}")

    if failures:
        for line in failures:
            print(f"  FAIL     {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
