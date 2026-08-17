"""The transferable bundle contract: relocation, offline operation, and integrity.

The claim under test is narrow and checkable: **a prepared bundle keeps working after it is copied
away from the directory that made it, and after the original inputs are gone.** Validating a bundle
where it was built cannot establish that, so these tests move it first and delete the source.

Network access is blocked inside the test process. Validation and execution from a prepared bundle
must not reach for anything, and a test that would silently pass on a machine with connectivity
proves nothing about one without.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from md_templates.openmm import bundlecheck, bundlev2

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPTIDE_PDB = REPO_ROOT / "src/md_templates/openmm/manifests/systems/ace_ala_nme.pdb"


# ---------------------------------------------------------------------------------------------
# the offline gate
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    """Make any outbound connection raise, so 'it worked' cannot mean 'it downloaded'."""

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted; a prepared bundle must work offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return True


# ---------------------------------------------------------------------------------------------
# helpers: prepare a real bundle once per module, for each route
# ---------------------------------------------------------------------------------------------

def _cli(*args, cwd: Path, timeout: int = 1800):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "md_templates.openmm.cli", *args],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)


SMILES_DOC = """
profile: cpu-smoke-v1
system: {system_id: ggg, route: smiles, smiles: "O=C1CNC(=O)CNC(=O)CN1"}
protocol:
  production:
    method: rest2
    n_chunks: 2
    chunk: 0.001 ns
    scale_factors: [1.0, 0.5625, 0.25]
    exchange_interval: 0.5 ps
    relaxation: 1 ps
execution: {platform: CPU}
"""

PEPTIDE_DOC = """
system:
  system_id: ace_ala_nme
  route: pdb
  pdb: ace_ala_nme.pdb
build:
  forcefield: {small_molecule: null, charge_method: null,
               protein: amber19/protein.ff19SB.xml, water: amber19/tip3pfb.xml}
  solvation: {box_shape: cube, padding: 1.2 nm, ionic_strength_molar: 0.0}
  nonbonded: {cutoff: 0.7 nm}
protocol:
  equilibration: {protocol: simple, minimize_max_iterations: 200, timestep: 1 fs,
                  nvt: 2 ps, npt: 2 ps, npt_free: 2 ps, box_average_last: 1 ps}
  production: {method: md, n_chunks: 2, chunk: 0.001 ns}
execution:
  platform: CPU
  reporting: {all_atom: 0.5 ps, solute: 0.5 ps, state: 0.5 ps, checkpoint: 0.5 ps}
"""


def _prepare(tmp_path_factory, route: str) -> tuple[Path, Path]:
    """Return `(bundle_dir, source_dir)` for a freshly prepared bundle of the given route."""
    source = tmp_path_factory.mktemp(f"source-{route}")
    if route == "smiles":
        (source / "sim.yaml").write_text(SMILES_DOC)
    else:
        shutil.copy2(PEPTIDE_PDB, source / "ace_ala_nme.pdb")
        (source / "sim.yaml").write_text(PEPTIDE_DOC)
    res = _cli("prepare", "--config", "sim.yaml", "--out-root", "./runs", "--platform", "CPU",
               cwd=source)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    bundle = next(p for p in (source / "runs").iterdir() if "__bundle__" in p.name)
    return bundle, source


@pytest.fixture(scope="module")
def smiles_bundle(tmp_path_factory):
    return _prepare(tmp_path_factory, "smiles")


@pytest.fixture(scope="module")
def peptide_bundle(tmp_path_factory):
    return _prepare(tmp_path_factory, "pdb")


# ---------------------------------------------------------------------------------------------
# contract shape
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_prepared_bundle_declares_schema_version_two(smiles_bundle):
    bundle, _ = smiles_bundle
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["bundle_schema_version"] == bundlev2.BUNDLE_SCHEMA_VERSION == 2
    # independent of the legacy system-manifest version, which is unchanged
    assert manifest["schema_version"] != manifest["bundle_schema_version"]


@pytest.mark.slow
def test_every_required_logical_role_is_present_and_relative(smiles_bundle):
    bundle, _ = smiles_bundle
    roles = json.loads((bundle / "bundle_manifest.json").read_text())["roles"]
    for role in bundlev2.REQUIRED_ROLES:
        assert role in roles, f"required role {role} is unmapped"
        rel = roles[role]
        assert bundlev2.normalise_relative(rel) == rel
        assert (bundle / rel).exists()


@pytest.mark.slow
def test_original_inputs_are_preserved_byte_for_byte(smiles_bundle):
    bundle, source = smiles_bundle
    stored = bundle / bundlev2.ORIGINAL_INPUTS_DIR / "sim.yaml"
    assert stored.read_bytes() == (source / "sim.yaml").read_bytes()


@pytest.mark.slow
def test_the_pdb_route_preserves_the_structure_it_was_given(peptide_bundle):
    bundle, _ = peptide_bundle
    stored = bundle / bundlev2.ORIGINAL_INPUTS_DIR / "ace_ala_nme.pdb"
    assert stored.is_file()
    assert bundlev2.sha256_file(stored) == bundlev2.sha256_file(PEPTIDE_PDB)


@pytest.mark.slow
def test_topology_atoms_and_openmm_particles_are_counted_separately(smiles_bundle):
    bundle, _ = smiles_bundle
    counts = json.loads((bundle / "bundle_manifest.json").read_text())["counts"]
    for key in ("topology_atoms", "openmm_particles", "virtual_sites", "massless_particles",
                "constraints", "degrees_of_freedom"):
        assert key in counts, f"{key} is not recorded"
    assert "n_atoms" not in counts, "the conflated name is back"
    assert counts["degrees_of_freedom_formula"]


def test_virtual_sites_make_atoms_and_particles_differ():
    """Equality of the two counts must not be assumed; a virtual site breaks it."""
    openmm = pytest.importorskip("openmm")
    from openmm import app, unit

    system = openmm.System()
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("HOH", chain)
    for _ in range(3):
        system.addParticle(1.0 * unit.dalton)
        top.addAtom("O", app.element.oxygen, res)
    # a fourth PARTICLE with no topology atom, massless, as a virtual site
    system.addParticle(0.0 * unit.dalton)
    system.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 1, 2, 0.5, 0.25, 0.25))

    counts = bundlev2.topology_counts(top, system)
    assert counts["topology_atoms"] == 3
    assert counts["openmm_particles"] == 4
    assert counts["virtual_sites"] == 1
    assert counts["massless_particles"] == 1
    assert counts["topology_atoms_equal_particles"] is False
    # a massless particle is not a degree of freedom
    assert counts["degrees_of_freedom"] == 3 * (4 - 1) - 0


@pytest.mark.slow
def test_topology_pdb_and_cif_describe_the_same_system(smiles_bundle):
    pytest.importorskip("openmm")
    from openmm import app

    bundle, _ = smiles_bundle
    pdb = app.PDBFile(str(bundle / "topology.pdb")).topology
    cif = app.PDBxFile(str(bundle / "topology.cif")).topology
    assert sum(1 for _ in pdb.atoms()) == sum(1 for _ in cif.atoms())
    assert [a.name for a in pdb.atoms()][:50] == [a.name for a in cif.atoms()][:50]
    assert [r.name for r in pdb.residues()][:20] == [r.name for r in cif.residues()][:20]


# ---------------------------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_checksums_cover_the_stated_domain_and_exclude_themselves(smiles_bundle):
    bundle, _ = smiles_bundle
    doc = json.loads((bundle / bundlev2.CHECKSUMS_FILE).read_text())
    assert bundlev2.CHECKSUMS_FILE in doc["domain"]["excludes"]
    assert "bundle_manifest.json" in doc["domain"]["excludes"]
    assert doc["domain"]["excluded_because"]
    assert bundlev2.CHECKSUMS_FILE not in doc["files"]
    assert any(p.startswith(bundlev2.ORIGINAL_INPUTS_DIR) for p in doc["files"])


@pytest.mark.slow
@pytest.mark.parametrize("victim", ["system.xml", "equilibrated_state.xml", "topology.pdb"])
def test_tampering_with_any_artifact_is_detected(smiles_bundle, tmp_path, victim):
    bundle, _ = smiles_bundle
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    target = copy / victim
    target.write_bytes(target.read_bytes() + b"\n<!-- tampered -->")
    report = bundlecheck.validate_bundle_v2(copy)
    assert not report.ok
    assert any(victim in e for e in report["errors"])


@pytest.mark.slow
def test_a_missing_required_file_is_detected(smiles_bundle, tmp_path):
    bundle, _ = smiles_bundle
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    (copy / "topology.cif").unlink()
    report = bundlecheck.validate_bundle_v2(copy)
    assert not report.ok
    assert any("topology.cif" in e for e in report["errors"])


@pytest.mark.parametrize("bad", ["/abs/path", "../escape", "a\\b", "C:/drive"])
def test_absolute_and_traversing_paths_are_refused(bad):
    with pytest.raises(bundlev2.BundleContractError):
        bundlev2.normalise_relative(bad)


def test_a_version_one_bundle_validates_with_downgraded_guarantees(tmp_path):
    """It stays readable; it must not be reported as satisfying the v2 contract."""
    old = tmp_path / "v1"
    old.mkdir()
    (old / "bundle_manifest.json").write_text(json.dumps({"schema_version": 1, "system": {}}))
    report = bundlecheck.validate_bundle_v2(old)
    assert report.ok, "a v1 bundle should remain readable"
    assert report["contract"] == "v1-compatibility"
    assert any("does NOT provide" in w for w in report["warnings"])


# ---------------------------------------------------------------------------------------------
# relocation, offline
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_relocate_check_reports_the_bundle_relocatable(smiles_bundle):
    bundle, _ = smiles_bundle
    result = bundlecheck.relocate_check(bundle)
    assert result["verdict"] == "relocatable", result
    assert not result["references_escaping_the_bundle"]
    assert Path(bundle).is_dir(), "relocate-check modified or moved the source bundle"


@pytest.mark.slow
def test_validation_works_offline_after_the_source_is_removed(tmp_path_factory, tmp_path,
                                                              no_network):
    """The real portability claim: copy away, delete the source, and it still checks out.

    Prepares its OWN bundle rather than sharing the module fixture: this test destroys the
    directory it was built in, and a destructive test must not reach into state its neighbours
    still depend on -- which it did, and which is why they failed after it ran.
    """
    bundle, source = _prepare(tmp_path_factory, "smiles")
    relocated = tmp_path / "somewhere" / "else" / bundle.name
    relocated.parent.mkdir(parents=True)
    shutil.copytree(bundle, relocated)

    before = json.loads((relocated / "bundle_manifest.json").read_text())
    hashes_before = json.loads((relocated / "canonical_configuration.json").read_text())["hashes"]

    # remove the ENTIRE originating directory, inside the test's own temporary area only
    shutil.rmtree(source)
    assert not source.exists()

    report = bundlecheck.validate_bundle_v2(relocated, deep=True)
    assert report.ok, report["errors"]

    after = json.loads((relocated / "bundle_manifest.json").read_text())
    hashes_after = json.loads((relocated / "canonical_configuration.json").read_text())["hashes"]
    assert hashes_before == hashes_after, "relocation changed a canonical hash"
    assert before["counts"] == after["counts"]


@pytest.mark.slow
def test_inspect_produces_structured_output_offline(smiles_bundle, no_network):
    bundle, _ = smiles_bundle
    info = bundlecheck.inspect_bundle(bundle)
    assert info["bundle_schema_version"] == 2
    assert info["identity"]["route"] == "smiles"
    assert info["counts"]["openmm_particles"] > 0
    assert info["validation"]["ok"]
    assert info["portability"]["binary_checkpoints"]


# ---------------------------------------------------------------------------------------------
# the legacy compatibility path must survive auto-migration
# ---------------------------------------------------------------------------------------------

def test_migration_refuses_to_guess_when_both_methods_are_declared():
    """A legacy experiment may declare both blocks; if/elif order is not a decision procedure.

    This regressed once: the shipped smoke experiment declares `md:` and `rest2:`, migration picked
    `md` because it was tested first, and a legacy REST2 bundle then refused to run as REST2.
    """
    from md_templates.openmm.spec import migrate

    system = {"system_id": "x", "input": {"route": "smiles", "smiles": "C"}}
    experiment = {"integrator": {"timestep_fs": 4.0, "temperature_k": 300.0},
                  "md": {"n_chunks": 1, "chunk_ns": 1.0},
                  "rest2": {"n_chunks": 1, "chunk_ns": 1.0, "scale_factors": [1.0, 0.5],
                            "exchange_interval_ps": 1.0, "relaxation_ps": 1.0}}
    with pytest.raises(migrate.MigrationError, match="ambiguous"):
        migrate.migrate_manifests(system, experiment)

    # naming the method resolves it, in both directions
    for method in ("md", "rest2"):
        doc, _ = migrate.migrate_manifests(system, experiment, method=method)
        assert doc["protocol"]["production"]["method"] == method


def test_an_unusable_canonical_record_is_treated_as_absent(tmp_path):
    """A bundle whose manifests could not be migrated stays on the compatibility path."""
    from md_templates.openmm import runner

    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "canonical_configuration.json").write_text(
        json.dumps({"unavailable": "MigrationError: ambiguous"}))
    assert runner._has_usable_canonical(bundle) is False

    (bundle / "canonical_configuration.json").write_text(
        json.dumps({"hashes": {}, "configuration": {}, "profile": {}}))
    assert runner._has_usable_canonical(bundle) is True
