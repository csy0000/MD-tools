"""The two worked examples must resolve to exactly the protocol their READMEs claim.

A README that describes six replicas and a JSON that resolves to ten is worse than no example at
all, so the documented numbers are asserted against the resolved configuration rather than trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
yaml = pytest.importorskip("yaml")

from md_templates.openmm.segments import (plan_segment_from_exchanges,
                                          reporting_interval_steps)  # noqa: E402
from md_templates.openmm.spec import resolve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ALANINE = REPO_ROOT / "test" / "ala" / "REST2" / "alanine_rest2.json"
RGDFV = REPO_ROOT / "test" / "rgd" / "REST2" / "rgdfv_rest2.json"
VETTED_RGDFV_MANIFEST = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
                         / "systems" / "cyclo_rgdfv.yaml")

pytestmark = pytest.mark.skipif(not ALANINE.exists(), reason="worked examples not present")


def _resolved(path: Path):
    return resolve.resolve_spec(json.loads(path.read_text()))["spec"]


# ---------------------------------------------------------------------------------------------
# replica counts -- the headline claim of each example
# ---------------------------------------------------------------------------------------------

def test_alanine_resolves_to_six_replicas():
    production = _resolved(ALANINE).protocol.production
    assert production.method == "rest2"
    assert production.n_replicas == 6
    assert production.tau_ladder.tau_values() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], abs=1e-15)
    assert production.scale_factors() == pytest.approx(
        [1.0, 0.81, 0.64, 0.49, 0.36, 0.25], abs=1e-15)


def test_rgdfv_resolves_to_ten_replicas():
    production = _resolved(RGDFV).protocol.production
    assert production.method == "rest2"
    assert production.n_replicas == 10
    assert production.tau_ladder.minimum == 0.0
    assert production.tau_ladder.maximum == 0.5


def test_rgdfv_ladder_is_the_already_validated_ten_rung_ladder():
    """The tau reparameterisation must reproduce the ladder the project already validated."""
    validated = [1.0, 0.891975308642, 0.790123456790, 0.694444444444, 0.604938271605,
                 0.521604938272, 0.444444444444, 0.373456790123, 0.308641975309, 0.25]
    resolved_factors = _resolved(RGDFV).protocol.production.scale_factors()
    assert resolved_factors == pytest.approx(validated, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# the vetted RGDfV chemistry is used verbatim, never re-derived
# ---------------------------------------------------------------------------------------------

def test_rgdfv_example_matches_the_vetted_manifest_exactly():
    """The example may not quietly differ from the vetted molecular definition."""
    manifest = yaml.safe_load(VETTED_RGDFV_MANIFEST.read_text())
    document = json.loads(RGDFV.read_text())
    assert document["system"]["smiles"] == manifest["input"]["smiles"]
    assert (document["system"]["canonical_smiles_sha256"]
            == manifest["input"]["canonical_smiles_sha256"])
    assert (document["system"]["expected_formal_charge"]
            == manifest["input"]["expected_formal_charge"] == 0)


def test_rgdfv_uses_the_vetted_ligand_route_not_ff19sb():
    """cyclo-RGDfV is parameterised as a ligand. ff19SB would be a different calculation."""
    build = _resolved(RGDFV).build
    assert build.forcefield.small_molecule == "openff-2.2.0"
    assert build.forcefield.charge_method == "am1bcc"
    assert build.forcefield.protein is None


def test_rgdfv_water_is_matched_to_its_force_field():
    """Sage/AM1-BCC was trained with TIP3P; OPC is ff19SB's partner, not Sage's."""
    assert _resolved(RGDFV).build.solvation.water_model == "tip3p"


def test_alanine_water_is_matched_to_its_force_field():
    """ff19SB's backbone parameters were fit with OPC."""
    build = _resolved(ALANINE).build
    assert build.forcefield.protein == "amber19/protein.ff19SB.xml"
    assert build.solvation.water_model == "opc"
    assert build.forcefield.water == "amber19/opc.xml"


# ---------------------------------------------------------------------------------------------
# the shared preparation both examples declare
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_both_examples_use_the_documented_preparation(path):
    spec = _resolved(path)
    assert spec.build.solvation.box_shape == "dodecahedron"
    assert spec.build.solvation.padding.value == pytest.approx(1.2)
    assert spec.build.solvation.ionic_strength_molar == 0.15
    assert spec.build.nonbonded.method == "PME"
    assert spec.build.nonbonded.cutoff.value == pytest.approx(1.0)
    assert spec.build.constraints == "HBonds"
    assert spec.build.rigid_water is True


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_both_examples_use_hydrogen_mass_repartitioning(path):
    """HMR to 3.024 amu (3 x 1.008) is what buys the 4 fs timestep."""
    build = _resolved(path).build
    assert build.hydrogen_mass.value == pytest.approx(3.024)
    assert build.hmr_scope == "solute"


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_hmr_requires_the_constraints_that_make_it_valid(path):
    """4 fs with HMR is only sound with H-bonds constrained and water rigid.

    HMR lowers the frequency of the bond-ANGLE motions involving hydrogen; the bond STRETCHES have
    to be removed by constraints first, or the timestep is limited by them regardless of mass.
    """
    build = _resolved(path).build
    assert build.constraints == "HBonds"
    assert build.rigid_water is True


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_hmr_scope_never_touches_water(path):
    """Repartitioning rigid water would change its rotational dynamics for no timestep benefit."""
    assert _resolved(path).build.hmr_scope == "solute"


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_both_examples_integrate_at_four_femtoseconds(path):
    integrator = _resolved(path).protocol.integrator
    assert integrator.timestep.value == pytest.approx(0.004)      # ps, enabled by HMR
    assert integrator.temperature.value == pytest.approx(300.0)
    assert integrator.friction.value == pytest.approx(1.0)
    assert integrator.kind == "langevin-middle"


# ---------------------------------------------------------------------------------------------
# stage separation: minimisation, NVT and NPT are distinct and validated
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_the_equilibration_stages_are_distinct(path):
    equilibration = _resolved(path).protocol.equilibration
    assert equilibration.minimize_max_iterations == 1000
    assert equilibration.nvt is not None and equilibration.nvt.value == pytest.approx(10.0)
    assert equilibration.npt is not None and equilibration.npt.value == pytest.approx(10.0)


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_the_equilibration_stages_are_whole_steps(path):
    """10 ps at 4 fs is exactly 2,500 steps for each of NVT and NPT."""
    spec = _resolved(path)
    dt = spec.protocol.integrator.timestep.value
    for stage in ("nvt", "npt"):
        duration = getattr(spec.protocol.equilibration, stage)
        steps = duration.value / dt
        assert steps == pytest.approx(2_500, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# segment and exchange arithmetic, as the READMEs state it
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_the_worked_exchange_contract_resolves_exactly(path):
    """1000 exchanges x 5 ps at 4 fs -> 1,250 steps per round, 1,250,000 per segment."""
    spec = _resolved(path)
    production = spec.protocol.production
    plan = plan_segment_from_exchanges(
        production.exchange.n_exchange_per_segment,
        production.exchange.exchange_interval.value,
        spec.protocol.integrator.timestep.value,
        interval_source=production.exchange.exchange_interval.source,
        timestep_source=spec.protocol.integrator.timestep.source,
    )
    assert plan.steps_per_exchange == 1_250
    assert plan.steps_per_segment == 1_250_000
    assert plan.number_of_exchanges_per_segment == 1_000


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_the_segment_length_is_derived_not_stated(path):
    """A product, not a quotient: 1000 x 5 ps = 5 ns, exact by construction."""
    production = _resolved(path).protocol.production
    assert production.duration_per_segment.value == pytest.approx(5000.0)   # ps
    assert "derived" in production.duration_per_segment.source


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_rest2_refuses_a_stated_segment_duration(path):
    """Stating it as well would let one document say the same thing twice and disagree."""
    document = json.loads(path.read_text())
    document["protocol"]["production"]["duration_per_segment"] = "5 ns"
    with pytest.raises(Exception, match="not an input for REST2"):
        resolve.resolve_spec(document)


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_segment_count_is_absent_from_the_scientific_input(path):
    """Ten ns per replica is two segments, and that '2' lives in Bash, not in the JSON."""
    document = json.loads(path.read_text())
    production = document["protocol"]["production"]
    for retired in ("n_chunks", "chunk_ns", "chunk", "scale_factors",
                    "number_of_exchanges_per_segment"):
        assert retired not in production, retired
    # REST2 derives its segment length; it must not be stated
    assert "duration_per_segment" not in production
    assert production["exchange"]["n_exchange_per_segment"] == 1000
    assert production["exchange"]["exchange_interval"] == "5 ps"


# ---------------------------------------------------------------------------------------------
# two trajectory streams
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_two_trajectory_streams_at_the_documented_cadence(path):
    """Full system every 100 ps, selected atoms every 10 ps, both whole numbers of steps."""
    spec = _resolved(path)
    dt = spec.protocol.integrator.timestep.value
    reporting = spec.execution.reporting
    full_system = reporting_interval_steps(
        reporting.all_atom.value, dt, interval_source=reporting.all_atom.source,
        timestep_source=spec.protocol.integrator.timestep.source, label="reporting.all_atom")
    selected = reporting_interval_steps(
        reporting.solute.value, dt, interval_source=reporting.solute.source,
        timestep_source=spec.protocol.integrator.timestep.source, label="reporting.solute")
    assert full_system == 25_000        # 100 ps at 4 fs
    assert selected == 2_500            # 10 ps at 4 fs
    assert full_system % selected == 0  # commensurate, or frames drift apart in physical time


# ---------------------------------------------------------------------------------------------
# enhanced region and omega exclusion
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_the_enhanced_region_is_the_whole_solute(path):
    selection = _resolved(path).protocol.production.enhanced_region
    assert selection.type == "solute"
    assert selection.atom_indices is None and selection.amber_mask is None


@pytest.mark.parametrize("path", [ALANINE, RGDFV])
def test_omega_exclusion_is_enabled_in_both_examples(path):
    omega = _resolved(path).protocol.production.omega_exclusion
    assert omega.enabled is True
    assert omega.definition == "peptide_omega"


def test_omega_exclusion_can_be_turned_off_explicitly():
    """Default-on must still be overridable, or it is a hard-coded policy rather than a default."""
    document = json.loads(ALANINE.read_text())
    document["protocol"]["production"]["omega_exclusion"] = {"enabled": False}
    spec = resolve.resolve_spec(document)["spec"]
    assert spec.protocol.production.omega_exclusion.enabled is False


def test_an_explicit_atom_index_selection_resolves():
    document = json.loads(ALANINE.read_text())
    document["protocol"]["production"]["enhanced_region"] = {
        "type": "atom_indices", "atom_indices": [0, 1, 2, 3]}
    spec = resolve.resolve_spec(document)["spec"]
    assert spec.protocol.production.enhanced_region.atom_indices == [0, 1, 2, 3]


def test_a_negative_atom_index_is_refused():
    document = json.loads(ALANINE.read_text())
    document["protocol"]["production"]["enhanced_region"] = {
        "type": "atom_indices", "atom_indices": [0, -1]}
    with pytest.raises(Exception, match="zero-based and non-negative"):
        resolve.resolve_spec(document)


def test_a_duplicated_atom_index_is_refused():
    document = json.loads(ALANINE.read_text())
    document["protocol"]["production"]["enhanced_region"] = {
        "type": "atom_indices", "atom_indices": [3, 3]}
    with pytest.raises(Exception, match="duplicate"):
        resolve.resolve_spec(document)


def test_two_selection_languages_at_once_are_refused():
    document = json.loads(ALANINE.read_text())
    document["protocol"]["production"]["enhanced_region"] = {
        "type": "solute", "amber_mask": ":1-3"}
    with pytest.raises(Exception, match="exactly one selection language"):
        resolve.resolve_spec(document)


# ---------------------------------------------------------------------------------------------
# the example files a reader is told to run must exist and be executable
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("relative", [
    "test/ala/REST2/README.md",
    "test/ala/REST2/run_all.sh",
    "test/ala/REST2/extension/README.md",
    "test/ala/REST2/extension/extend.sh",
    "test/rgd/REST2/README.md",
    "test/rgd/REST2/run_all.sh",
    "test/rgd/REST2/extension/README.md",
    "test/rgd/REST2/extension/extend.sh",
])
def test_the_documented_example_files_exist(relative):
    assert (REPO_ROOT / relative).is_file()


@pytest.mark.parametrize("relative", [
    "test/ala/REST2/run_all.sh",
    "test/ala/REST2/extension/extend.sh",
    "test/rgd/REST2/run_all.sh",
    "test/rgd/REST2/extension/extend.sh",
])
def test_the_example_scripts_are_executable(relative):
    import os
    assert os.access(REPO_ROOT / relative, os.X_OK), f"{relative} is not executable"


@pytest.mark.parametrize("system", ["ala", "rgd"])
def test_generated_outputs_are_gitignored(system):
    """Trajectories and checkpoints must never reach the repository."""
    ignore = (REPO_ROOT / "test" / system / "REST2" / ".gitignore").read_text()
    assert "outputs/" in ignore
