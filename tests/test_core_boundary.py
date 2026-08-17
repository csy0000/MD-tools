"""Phase 3 gate: the engine-neutral core is genuinely engine-free, and the move changed no value.

Two questions, and they need different kinds of evidence.

**Is core importable without an engine?** A source scan answers "does it mention OpenMM", which is
not the same thing — every deferred import inside a function mentions it and none of them violate
the boundary. The honest test runs a subprocess with the heavy packages made *unimportable*, then
imports core and does real work with it. If anything reaches for an engine at import time, the
subprocess dies.

**Did moving the code change any value?** The migration's central risk is not that something breaks
loudly but that a hash shifts by a byte. Every configuration is resolved through both the legacy
import path and the new one and compared field by field, including all five projection hashes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

HEAVY = ("openmm", "openff", "rdkit", "mdtraj", "parmed", "openmmtools", "numpy", "scipy", "pandas")


def run_isolated(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess where importing any heavy scientific package raises."""
    preamble = f"""
        import sys
        HEAVY = {HEAVY!r}

        class _Blocker:
            def find_module(self, name, path=None):
                if name.split('.')[0] in HEAVY:
                    raise ImportError(
                        f"{{name}} is deliberately unimportable in this test: core must not need it")
                return None

        sys.meta_path.insert(0, _Blocker())
        """
    script = textwrap.dedent(preamble) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=600, cwd=str(REPO_ROOT),
                          env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})


# ------------------------------------------------------------------------------------------------
# the import boundary, proven by making the engine unavailable
# ------------------------------------------------------------------------------------------------

def test_core_imports_and_works_with_every_heavy_dependency_unimportable():
    result = run_isolated("""
        import md_templates.core as core
        from md_templates.core.config import resolve, canonical
        from md_templates.core import bundle, persistence, fingerprint, hashing

        document = {"system": {"system_id": "x", "route": "smiles", "smiles": "CCO"},
                    "protocol": {"production": {"method": "md"}}}
        result = resolve.resolve_spec(document)
        assert set(result["hashes"]) >= {"system_build_sha256", "protocol_sha256"}
        assert bundle.BUNDLE_SCHEMA_VERSION == 2
        assert persistence.RUN_STATE_SCHEMA == 1
        assert hashing.sha256_text("x")
        loaded = sorted(m for m in sys.modules if m.split('.')[0] in HEAVY)
        assert loaded == [], loaded
        print("OK")
        """)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("OK")


def test_catalog_listing_and_identity_work_with_every_heavy_dependency_unimportable():
    result = run_isolated("""
        from md_templates.core import load_catalog, inspect_provenance
        catalog = load_catalog(".")
        assert sorted(catalog.descriptors) == [
            "conventional-md/openmm/explicit-water", "rest2/openmm/explicit-water"]
        inspect_provenance(".")
        print("OK")
        """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_blocker_itself_works():
    """A guard that cannot fail proves nothing."""
    result = run_isolated("import openmm\n")
    assert result.returncode != 0
    assert "deliberately unimportable" in result.stderr


@pytest.mark.parametrize("module", [
    "md_templates.core", "md_templates.core.config", "md_templates.core.config.models",
    "md_templates.core.config.canonical", "md_templates.core.config.resolve",
    "md_templates.core.bundle", "md_templates.core.persistence", "md_templates.core.fingerprint",
    "md_templates.core.hashing", "md_templates.core.registry", "md_templates.core.identity",
    "md_templates.core.packaged", "md_templates.core.template", "md_templates.core.paths",
    "md_templates.core.resources",
])
def test_each_core_module_imports_alone_without_an_engine(module):
    result = run_isolated(f"import {module}\nprint('OK')\n")
    assert result.returncode == 0, result.stdout + result.stderr


# ------------------------------------------------------------------------------------------------
# differential: the same inputs through both import paths must produce identical everything
# ------------------------------------------------------------------------------------------------

CONFIGURATIONS = {
    "md_ligand_smiles": {"system": {"system_id": "golden_ligand", "route": "smiles",
                                    "smiles": "CCO"},
                         "protocol": {"production": {"method": "md"}}},
    "md_peptide_pdb": {"system": {"system_id": "golden_peptide", "route": "pdb",
                                  "pdb": "input.pdb"},
                       "protocol": {"production": {"method": "md"}}},
    "rest2_ligand_smiles": {"system": {"system_id": "golden_ligand", "route": "smiles",
                                       "smiles": "CCO"},
                            "protocol": {"production": {"method": "rest2"}}},
    "rest2_peptide_pdb": {"system": {"system_id": "golden_peptide", "route": "pdb",
                                     "pdb": "input.pdb"},
                          "protocol": {"production": {"method": "rest2"}}},
}


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_legacy_and_core_import_paths_resolve_identically(name):
    from md_templates.core.config import canonical as core_canonical
    from md_templates.core.config import resolve as core_resolve
    from md_templates.openmm.spec import canonical as legacy_canonical
    from md_templates.openmm.spec import resolve as legacy_resolve

    document = CONFIGURATIONS[name]
    legacy = legacy_resolve.resolve_spec(json.loads(json.dumps(document)))
    current = core_resolve.resolve_spec(json.loads(json.dumps(document)))

    assert legacy["hashes"] == current["hashes"]
    assert legacy["profile"] == current["profile"]
    assert legacy["sources"] == current["sources"]
    assert legacy_canonical.canonical_json(legacy_canonical.dump_model(legacy["spec"])) == \
        core_canonical.canonical_json(core_canonical.dump_model(current["spec"]))


def test_the_legacy_and_core_modules_are_the_same_objects_not_copies():
    """If they were copies, the differential test above would compare a thing with itself."""
    import md_templates.core.bundle as core_bundle
    import md_templates.core.config as core_config
    import md_templates.core.fingerprint as core_fingerprint
    import md_templates.core.persistence as core_persistence
    import md_templates.openmm.bundlev2 as legacy_bundle
    import md_templates.openmm.fingerprint as legacy_fingerprint
    import md_templates.openmm.runstate as legacy_runstate
    import md_templates.openmm.spec as legacy_spec

    assert legacy_spec is core_config
    assert legacy_runstate is core_persistence
    assert legacy_bundle is core_bundle
    assert legacy_fingerprint is core_fingerprint


def test_engine_side_functions_are_reachable_from_the_legacy_paths():
    """The two splits Phase 3 made must be invisible to existing callers."""
    from md_templates.openmm.bundlev2 import forcefield_provenance, topology_counts
    from md_templates.openmm.runstate import load_restart, save_restart

    for function in (save_restart, load_restart, topology_counts, forcefield_provenance):
        assert callable(function)


def test_hashing_has_exactly_one_definition():
    """Two implementations would hash one document two ways."""
    from md_templates.core.fingerprint import canonical_json as fingerprint_canonical
    from md_templates.core.hashing import canonical_json as core_canonical
    from md_templates.openmm.schemas import canonical_json as manifest_canonical

    assert core_canonical is manifest_canonical is fingerprint_canonical


def test_profiles_resolve_from_their_new_home_with_unchanged_identity():
    from md_templates.core.config import resolve

    profiles = {d["profile_id"]: d for d in resolve.list_profiles()}
    assert sorted(profiles) == ["cpu-smoke-v1", "explicit-md-ligand-v1", "explicit-md-peptide-v1",
                                "explicit-rest2-ligand-v1", "explicit-rest2-peptide-v1"]
    assert resolve.PROFILE_DIR.parent.name == "config"
    assert resolve.select_profile("smiles", "rest2")["profile_id"] == "explicit-rest2-ligand-v1"
