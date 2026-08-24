"""Ensemble semantics, and the NPT exchange criterion that makes them possible.

The tree these tests fix was contradictory in four places at once: the active default said
production was NPT, the generated stage path attached a barostat to ``cMD_1``, legacy ``run_md()``
rejected everything except NVT, and ``run_rest2_remd()`` rejected NPT. The shipped CPU smoke test
was failing on `dev` as a direct result -- an NPT default meeting an NVT-only refusal -- and the
fast suite never saw it, because the smoke is a slow test.

Two ideas carry all of this.

An ensemble label is a property of the system, not a preference. Explicit solvent has a volume and
a pressure, so NPT is meaningful. Implicit solvent has neither, so "NVT" would name a fixed volume
that does not exist. That is why the resolution is a function of solvation mode rather than a
default value.

And the acceptance criterion is written in its general four-reduced-potential form even though
REST2 shares ``beta`` and ``p`` across replicas and the ``pV`` terms therefore cancel exactly. The
cancellation is real; relying on it silently is how a criterion outlives the protocol it was
correct for. Here it is demonstrated instead.
"""

from __future__ import annotations

import math

import pytest

from md_templates.openmm.ensembles import (EXPLICIT_PRODUCTION_ENSEMBLE,
                                           IMPLICIT_PRODUCTION_ENSEMBLE, EnsembleError,
                                           canonical_ensemble, ensemble_is_barostatted,
                                           validate_ensemble)
from md_templates.openmm.rest2 import (BAR_NM3_TO_KJ_PER_MOL, exchange_log_acceptance,
                                       reduced_potential)


# -------------------------------------------------------------------------------------------
# Canonical ensembles
# -------------------------------------------------------------------------------------------
def test_explicit_is_npt_and_implicit_is_not_an_acronym():
    assert canonical_ensemble("explicit") == "NPT"
    assert canonical_ensemble("implicit") == "nonperiodic-constant-temperature"


def test_the_implicit_label_never_names_a_volume():
    """"NVT" would assert a fixed volume for a system that has none. The label must not contain it."""
    assert "NVT" not in IMPLICIT_PRODUCTION_ENSEMBLE.upper()
    assert "NPT" not in IMPLICIT_PRODUCTION_ENSEMBLE.upper()


def test_only_explicit_carries_a_barostat():
    assert ensemble_is_barostatted("explicit") is True
    assert ensemble_is_barostatted("implicit") is False


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_an_absent_ensemble_resolves_rather_than_raising(mode):
    """An unstated field has not expressed an opinion, and there is one correct answer per mode."""
    assert validate_ensemble(None, mode) == canonical_ensemble(mode)


@pytest.mark.parametrize("bad", ["NVT", "NPT"])
def test_implicit_refuses_both_box_ensembles(bad):
    with pytest.raises(EnsembleError) as excinfo:
        validate_ensemble(bad, "implicit")
    message = str(excinfo.value)
    assert "production.ensemble" in message and bad in message
    assert "no periodic box" in message


def test_explicit_refuses_nvt_and_explains_why():
    with pytest.raises(EnsembleError) as excinfo:
        validate_ensemble("NVT", "explicit")
    assert "equilibration window" in str(excinfo.value)


def test_an_unknown_solvation_mode_is_refused():
    with pytest.raises(EnsembleError):
        canonical_ensemble("vacuum-ish")


# -------------------------------------------------------------------------------------------
# The reduced potential
# -------------------------------------------------------------------------------------------
def test_the_pv_conversion_matches_an_independent_hand_calculation():
    """``1 bar * 1 nm^3 = 1e5 Pa * 1e-27 m^3 = 1e-22 J``; times Avogadro, divided by 1000."""
    expected = 1e5 * 1e-27 * 6.02214076e23 / 1000.0
    assert BAR_NM3_TO_KJ_PER_MOL == pytest.approx(expected, rel=1e-12)
    assert BAR_NM3_TO_KJ_PER_MOL == pytest.approx(0.0602214076, rel=1e-9)


def test_reduced_potential_matches_a_hand_calculation():
    # u = beta * (U + p*V) = 0.4 * (-100 + 1*45*0.0602214076)
    u = reduced_potential(-100.0, 0.4, 1.0, 45.0)
    assert u == pytest.approx(0.4 * (-100.0 + 45.0 * 0.0602214076), rel=1e-12)


def test_no_volume_means_no_pv_term_not_a_zero_volume():
    """A nonperiodic system has no volume. Passing ``None`` must drop the term, not multiply by 0."""
    assert reduced_potential(-100.0, 0.4, 1.0, None) == pytest.approx(-40.0)
    assert reduced_potential(-100.0, 0.4, None, 45.0) == pytest.approx(-40.0)
    # and a genuine zero volume would be a different statement, which is why 0.0 is not None
    assert reduced_potential(-100.0, 0.4, 1.0, 0.0) == pytest.approx(-40.0)


# -------------------------------------------------------------------------------------------
# The acceptance criterion
# -------------------------------------------------------------------------------------------
def test_log_acceptance_matches_the_written_expression():
    u_ii, u_jj, u_ij, u_ji = -10.0, -20.0, -12.0, -17.0
    assert exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji) == pytest.approx(
        -((u_ij + u_ji) - (u_ii + u_jj)))


def test_at_common_temperature_and_pressure_the_pv_terms_cancel_exactly():
    """The property the four-term form is *not* allowed to rely on silently.

    Same beta, same p, different volumes. The general expression must reduce exactly to
    ``-beta * [(E_ij + E_ji) - (E_ii + E_jj)]``, with every pV contribution cancelling.
    """
    beta, p = 0.4009, 1.0
    V_i, V_j = 41.7, 44.9                       # deliberately different boxes
    E_ii, E_jj, E_ij, E_ji = -8213.7, -8265.1, -8275.7, -8196.2

    u_ii = reduced_potential(E_ii, beta, p, V_i)
    u_jj = reduced_potential(E_jj, beta, p, V_j)
    u_ij = reduced_potential(E_ij, beta, p, V_j)   # config j keeps ITS box
    u_ji = reduced_potential(E_ji, beta, p, V_i)   # config i keeps ITS box

    general = exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji)
    energy_only = -beta * ((E_ij + E_ji) - (E_ii + E_jj))
    assert general == pytest.approx(energy_only, rel=1e-12)


def test_the_pv_terms_do_not_cancel_when_the_pressures_differ():
    """Which is why the general form is kept: the cancellation is a property of this protocol only."""
    beta = 0.4009
    V_i, V_j = 41.7, 44.9
    E_ii, E_jj, E_ij, E_ji = -8213.7, -8265.1, -8275.7, -8196.2

    u_ii = reduced_potential(E_ii, beta, 1.0, V_i)
    u_jj = reduced_potential(E_jj, beta, 500.0, V_j)     # a different pressure
    u_ij = reduced_potential(E_ij, beta, 1.0, V_j)
    u_ji = reduced_potential(E_ji, beta, 500.0, V_i)

    general = exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji)
    energy_only = -beta * ((E_ij + E_ji) - (E_ii + E_jj))
    assert general != pytest.approx(energy_only, rel=1e-9)


def test_a_configuration_swapped_into_the_wrong_box_is_a_different_state():
    """Cross-evaluation keeps each configuration with its own cell; swapping cells changes u."""
    beta, p = 0.4009, 1.0
    correct = reduced_potential(-8275.7, beta, p, 44.9)     # config j under its own V_j
    swapped = reduced_potential(-8275.7, beta, p, 41.7)     # same energy, wrong volume
    assert correct != pytest.approx(swapped, rel=1e-12)


# -------------------------------------------------------------------------------------------
# Stage wiring
# -------------------------------------------------------------------------------------------
def test_rest2_production_is_barostatted():
    """The defect: explicit REST2 production ran at fixed volume while its config said NPT."""
    from md_templates.openmm.input_gen import BAROSTAT_STAGES

    assert "REST2_1" in BAROSTAT_STAGES
    assert "cMD_1" in BAROSTAT_STAGES


def test_minimisation_is_never_barostatted():
    from md_templates.openmm.input_gen import BAROSTAT_STAGES

    assert "min" not in BAROSTAT_STAGES


# -------------------------------------------------------------------------------------------
# Platform policy (dynamics on CUDA; minimisation exempt)
# -------------------------------------------------------------------------------------------
def test_minimisation_is_exempt_from_the_dynamics_platform_requirement(monkeypatch):
    from md_templates.openmm.equilibration import assert_dynamics_platform

    monkeypatch.setenv("MD_REQUIRE_DYNAMICS_PLATFORM", "CUDA")
    assert_dynamics_platform({"production": {"platform": "CPU"}}, "min")   # must not raise


@pytest.mark.parametrize("stage", ["eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1"])
def test_every_dynamics_stage_honours_the_platform_requirement(monkeypatch, stage):
    from md_templates.openmm.equilibration import assert_dynamics_platform

    monkeypatch.setenv("MD_REQUIRE_DYNAMICS_PLATFORM", "CUDA")
    with pytest.raises(RuntimeError) as excinfo:
        assert_dynamics_platform({"production": {"platform": "CPU"}}, stage)
    assert "integrates dynamics" in str(excinfo.value)
    assert_dynamics_platform({"production": {"platform": "CUDA"}}, stage)


def test_an_unresolved_platform_is_refused_rather_than_defaulted():
    """`_runtime_cfg` used to default this to "CPU", which decided where the science ran."""
    from md_templates.openmm.equilibration import assert_dynamics_platform

    with pytest.raises(RuntimeError) as excinfo:
        assert_dynamics_platform({"production": {"platform": None}}, "cMD_1")
    assert "not defaulted" in str(excinfo.value)


def test_an_explicitly_chosen_cpu_is_allowed_without_a_pin():
    """The shipped smoke and the CPU integration gate both want CPU dynamics deliberately."""
    from md_templates.openmm.equilibration import assert_dynamics_platform

    assert_dynamics_platform({"production": {"platform": "CPU"}}, "cMD_1")


# -------------------------------------------------------------------------------------------
# The unrestrained implicit equilibration phase (protocol.equilibration.free)
#
# Restrained dynamics equilibrates the solvent response around a HELD solute; the solute's own
# conformational relaxation only begins once the restraint is released. A protocol with only a
# restrained phase therefore enters production still relaxing. `free` is the implicit spelling of
# that phase, and is deliberately not called `nvt` or `npt_free`: both name a volume, and an
# implicit system has none.
# -------------------------------------------------------------------------------------------
def test_the_implicit_free_stage_appears_only_when_requested():
    """A stage generated with zero steps is a directory that looks like a completed phase."""
    from md_templates.openmm.input_gen import stage_order_for

    assert stage_order_for("implicit", "md") == ("min", "eq", "cMD_1")
    assert stage_order_for("implicit", "md", implicit_free_equilibration=True) == (
        "min", "eq", "eq_free", "cMD_1")
    assert stage_order_for("implicit", "rest2", implicit_free_equilibration=True) == (
        "min", "eq", "eq_free", "cMD_1", "REST2_1")


def test_the_free_stage_is_neither_restrained_nor_barostatted():
    """It is the UNrestrained phase, and implicit solvent has nothing for a barostat to act on."""
    from md_templates.openmm.input_gen import BAROSTAT_STAGES, RESTRAINED_STAGES

    assert "eq" in RESTRAINED_STAGES
    assert "eq_free" not in RESTRAINED_STAGES
    assert "eq_free" not in BAROSTAT_STAGES


def test_explicit_solvent_refuses_the_implicit_spelling():
    """Two field names for one stage drift apart the moment one acquires a default."""
    pytest.importorskip("pydantic")
    import copy

    from md_templates.openmm.spec.models import SimulationSpec
    from md_templates.openmm.spec.resolve import load_profile

    defaults = copy.deepcopy(load_profile("explicit-md-peptide-v2")["defaults"])
    # profiles carry build/protocol/execution; `system` is the caller's and is supplied here
    defaults["system"] = {"system_id": "ace_ala_nme", "route": "pdb",
                          "pdb": "ace_ala_nme.pdb"}
    # valid as it stands, so the failure below is attributable to the one field under test
    SimulationSpec(**copy.deepcopy(defaults))
    defaults["protocol"]["equilibration"]["free"] = "500 ps"
    with pytest.raises(Exception) as excinfo:
        SimulationSpec(**defaults)
    message = str(excinfo.value)
    assert "implicit solvent only" in message
    assert "npt_free" in message


def test_the_protocol_schema_version_was_bumped_for_the_added_field():
    """The canonical document gained a key, so an unchanged protocol's HASH moved.

    Two documents that both call themselves version 5 must not hash differently -- signalling that
    is exactly what the version number is for.
    """
    from md_templates.openmm.spec.models import (PROTOCOL_SCHEMA_VERSION,
                                                 RETIRED_PROTOCOL_SCHEMA_VERSIONS)

    assert PROTOCOL_SCHEMA_VERSION == 6
    assert 5 in RETIRED_PROTOCOL_SCHEMA_VERSIONS


def test_a_version_five_document_is_told_how_to_migrate():
    """A retirement that does not say what to do makes the user guess."""
    from md_templates.openmm.spec.models import PROTOCOL_SCHEMA_VERSION

    import md_templates.openmm.spec.models as models

    spec_cls = None
    for value in vars(models).values():
        fields = getattr(value, "model_fields", None)
        if fields and "schema_version" in fields and "equilibration" in fields:
            spec_cls = value
            break
    assert spec_cls is not None, "could not locate the protocol spec model"

    with pytest.raises(Exception) as excinfo:
        spec_cls(schema_version=5,
                 integrator={"kind": "langevin-middle", "timestep": "2 fs",
                             "temperature": "300 K", "friction": "1 /ps"},
                 equilibration={},
                 production={"method": "md", "duration_per_segment": "5 ns"})
    message = str(excinfo.value)
    assert "superseded by 6" in message
    assert "relabelling it 6 is the whole migration" in message
    assert "HASH" in message


def test_every_stage_the_generator_can_emit_is_executable():
    """The generator and the executor keep separate stage lists; they must agree.

    This is the test that would have caught a real failure: `eq_free` was added to the generator's
    stage graph and a run was launched, which minimised, equilibrated, and then died at
    `stage: unknown stage 'eq_free'` -- after the GPU work of the preceding stages had been spent.

    Nothing else checks this. The generator validates that it CAN write the stage; the executor
    validates a stage it has been handed. Neither compares its list against the other's, so the two
    can drift and only a real run reveals it.
    """
    from md_templates.openmm.input_gen import (IMPLICIT_MD_STAGE_ORDER, IMPLICIT_STAGE_ORDER,
                                               MD_STAGE_ORDER, STAGE_ORDER)
    from md_templates.openmm.stage import EXECUTION_STATUS

    emittable = (set(STAGE_ORDER) | set(IMPLICIT_STAGE_ORDER)
                 | set(MD_STAGE_ORDER) | set(IMPLICIT_MD_STAGE_ORDER))
    missing = sorted(emittable - set(EXECUTION_STATUS))
    assert not missing, (
        f"the generator can emit {missing}, which the executor would refuse as an unknown stage. "
        "A run would fail partway, after the preceding stages had already been computed.")


def test_the_executor_declares_no_stage_the_generator_cannot_emit():
    """The converse: a registered stage nobody generates is dead code claiming to be implemented."""
    from md_templates.openmm.input_gen import (IMPLICIT_MD_STAGE_ORDER, IMPLICIT_STAGE_ORDER,
                                               MD_STAGE_ORDER, STAGE_ORDER)
    from md_templates.openmm.stage import EXECUTION_STATUS

    emittable = (set(STAGE_ORDER) | set(IMPLICIT_STAGE_ORDER)
                 | set(MD_STAGE_ORDER) | set(IMPLICIT_MD_STAGE_ORDER))
    orphans = sorted(set(EXECUTION_STATUS) - emittable)
    assert not orphans, f"executor declares {orphans}, which no stage graph produces"


# -------------------------------------------------------------------------------------------
# Barostat seeds must survive OpenMM's 32-bit seed field
# -------------------------------------------------------------------------------------------
def test_derived_barostat_seeds_are_int32_safe_for_a_large_master_seed():
    """The bug this replaces killed a six-replica ladder after its Contexts were built.

    The barostat seed was `master + 100000 + replica`. Master seeds in this repository are dated
    integers -- 20260824003 -- which is an order of magnitude above OpenMM's signed 32-bit seed
    field, so `setRandomNumberSeed` raised OverflowError at replica 0. Arithmetic on a master seed
    is the mistake; `derive_seed` is 32-bit safe by construction.
    """
    from md_templates.openmm.seeds import derive_seed, replica_purpose

    master = 20260824003
    assert master > 2**31 - 1, "the master seed must exceed int32 for this test to mean anything"
    for replica in range(6):
        seed = derive_seed(master, replica_purpose(replica, "barostat"))
        assert 1 <= seed <= 2**31 - 1


def test_each_replica_barostat_stream_is_independent():
    """Part 3 requires one INDEPENDENTLY seeded barostat per replica.

    Shared or correlated volume moves would break the independence the exchange criterion assumes,
    and a barostat sharing its integrator's seed is correlated in a way nobody would look for.
    """
    from md_templates.openmm.seeds import derive_seed, replica_purpose

    master = 20260824003
    barostats = [derive_seed(master, replica_purpose(r, "barostat")) for r in range(6)]
    integrators = [derive_seed(master, replica_purpose(r, "integrator")) for r in range(6)]
    assert len(set(barostats)) == 6
    assert not set(barostats) & set(integrators)


def test_openmm_rejects_a_seed_above_int32():
    """Anchors the guard to observed OpenMM behaviour rather than a remembered claim."""
    openmm = pytest.importorskip("openmm")
    from openmm import unit

    barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25)
    with pytest.raises(OverflowError):
        barostat.setRandomNumberSeed(20260824003 + 100_000)
    barostat.setRandomNumberSeed(2**31 - 1)          # the boundary is accepted


def test_as_openmm_seed_is_the_identity_on_every_legal_seed():
    """A clamp that changes an already-valid seed silently changes the trajectory.

    The first version used the derive_seed modulus (2**31-2) and mapped the legal seed 2**31-1 to
    1 -- a collision at exactly the boundary the function exists to protect.
    """
    from md_templates.openmm.seeds import as_openmm_seed

    for value in (1, 2, 12345, 2**31 - 2, 2**31 - 1):
        assert as_openmm_seed(value) == value


def test_as_openmm_seed_brings_everything_else_into_range():
    from md_templates.openmm.seeds import as_openmm_seed

    for value in (2**31, 20260824003, 20260824003 + 100_000, -5, 0):
        assert 1 <= as_openmm_seed(value) <= 2**31 - 1


def test_as_openmm_seed_never_returns_zero():
    """OpenMM reads 0 as 'choose randomly', which makes a seeded run irreproducible in silence."""
    from md_templates.openmm.seeds import as_openmm_seed

    assert as_openmm_seed(0) != 0
    assert all(as_openmm_seed(v) != 0 for v in (0, 2**31 - 1, 2**31, -(2**31)))


def test_every_openmm_seed_call_site_is_clamped():
    """The class of bug, not an instance of it.

    Two ladders died on this: a barostat seed and a velocity seed, at different call sites, both
    from a dated master seed an order of magnitude above int32. Neither OpenMM message mentions
    seeds -- one is an OverflowError, the other a TypeError about overloaded prototypes -- so each
    had to be diagnosed from a stack trace after GPU time had been spent.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "md_templates" / "openmm"
    pattern = re.compile(r"(setRandomNumberSeed|setVelocitiesToTemperature)\s*\(([^)]*)\)")
    unclamped = []
    for path in src.glob("*.py"):
        if path.name == "seeds.py":
            continue
        for call in pattern.finditer(path.read_text()):
            args = call.group(2)
            # a single-argument setVelocitiesToTemperature takes no seed at all
            if call.group(1) == "setVelocitiesToTemperature" and "," not in args:
                continue
            if "as_openmm_seed" not in args:
                unclamped.append(f"{path.name}: {call.group(0)[:70]}")
    assert not unclamped, (
        "these OpenMM seed call sites do not clamp through as_openmm_seed, so a master seed above "
        "2**31-1 would crash them at runtime:\n  " + "\n  ".join(unclamped))
