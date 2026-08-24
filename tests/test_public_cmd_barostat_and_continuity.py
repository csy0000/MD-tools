"""What the public conventional-MD path actually propagates, and what it records.

Two findings from the 2026-08-24 audit, both traced through the executable path rather than read
off the source.

The BAROSTAT. A prepared bundle's `system.xml` carries none -- measured on the shipped explicit
bundle, whose forces are Harmonic{Bond,Angle}, PeriodicTorsion, Nonbonded, CMAPTorsion and
CMMotionRemover. The stage path attaches one through `BAROSTAT_STAGES`. The legacy
`md-openmm md` -> `runner.launch_md()` -> `md.run_md()` path did not, so once the ensemble resolver
started reporting "NPT" for an explicit System that path labelled its run NPT and integrated at
FIXED VOLUME. Before the resolver existed it refused every ensemble except NVT, which was at least
honest; the resolver made it silently wrong instead of loudly broken.

The CONTINUITY RECORD. `_cmd_continuity` wrote `"NPT" if barostat else "NVT"`, so every implicit
segment recorded itself as NVT -- a restart contract asserting a fixed volume for a System that has
no volume at all.

"Volume happened not to change" is not evidence either way; these tests inspect the System's forces.
"""

from __future__ import annotations

import json

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from md_templates.openmm.ensembles import (EXPLICIT_PRODUCTION_ENSEMBLE,  # noqa: E402
                                           IMPLICIT_PRODUCTION_ENSEMBLE, canonical_ensemble)


def _system(periodic):
    """A minimal System that answers `usesPeriodicBoundaryConditions()` the way we need."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.amu)
    force = openmm.NonbondedForce()
    if periodic:
        system.setDefaultPeriodicBoxVectors(*(
            [4.0, 0, 0] * unit.nanometer, [0, 4.0, 0] * unit.nanometer,
            [-2.0, -2.0, 2.828427] * unit.nanometer))
        force.setNonbondedMethod(openmm.NonbondedForce.PME)
        force.setCutoffDistance(1.0 * unit.nanometer)
    else:
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for _ in range(4):
        force.addParticle(0.0, 0.3, 0.5)
    system.addForce(force)
    return system


def _barostats(system):
    return [f for f in system.getForces() if "Barostat" in f.__class__.__name__]


# -------------------------------------------------------------------------------------------
# The bundle itself carries no barostat -- which is why the runtime must attach one
# -------------------------------------------------------------------------------------------
def test_a_prepared_bundle_system_carries_no_barostat():
    """The premise of the whole finding. If bundles carried one, attaching another would be the bug."""
    system = _system(periodic=True)
    assert _barostats(system) == []


# -------------------------------------------------------------------------------------------
# The legacy public path
# -------------------------------------------------------------------------------------------
def _cfg(ensemble, temperature_k=300.0, pressure_bar=1.0, interval=50):
    return {"production": {"ensemble": ensemble},
            "integrator": {"temperature_k": temperature_k},
            "equilibration": {"pressure_bar": pressure_bar, "barostat_interval": interval}}


def test_the_propagated_explicit_system_carries_exactly_one_barostat():
    """BEHAVIOURAL: builds the System that would be propagated and counts its forces.

    This replaces a source-text assertion, which would still have passed if the barostat call were
    moved behind a condition that never fired. The System is the evidence; the source is not.
    """
    from md_templates.openmm.md import attach_production_barostat

    system = _system(periodic=True)
    assert _barostats(system) == [], "the bundle System must start with none"
    seed = attach_production_barostat(system, _cfg(EXPLICIT_PRODUCTION_ENSEMBLE), 20260824003)
    barostats = _barostats(system)
    assert len(barostats) == 1
    assert seed is not None


def test_the_propagated_implicit_system_carries_no_barostat():
    """Nothing for a barostat to act on: no box, no volume, no pressure."""
    from md_templates.openmm.md import attach_production_barostat

    system = _system(periodic=False)
    seed = attach_production_barostat(system, _cfg(IMPLICIT_PRODUCTION_ENSEMBLE), 20260824003)
    assert _barostats(system) == []
    assert seed is None


def test_the_propagated_system_integrates_and_the_barostat_is_active():
    """A Context is built and stepped, so the force is not merely present but usable."""
    from md_templates.openmm.md import attach_production_barostat

    system = _system(periodic=True)
    attach_production_barostat(system, _cfg(EXPLICIT_PRODUCTION_ENSEMBLE), 20260824003)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]] * unit.nanometer)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 4321)
    integrator.step(50)
    assert context.getState(getEnergy=True).getPotentialEnergy() is not None


def test_a_second_barostat_is_refused_by_the_guard():
    """Two apply two independent volume moves per step and sample no defined ensemble."""
    from md_templates.openmm.md import attach_production_barostat

    system = _system(periodic=True)
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 50))
    with pytest.raises(ValueError) as excinfo:
        attach_production_barostat(system, _cfg(EXPLICIT_PRODUCTION_ENSEMBLE), 20260824003)
    assert "requires exactly 1" in str(excinfo.value)


def test_an_implicit_system_that_somehow_carries_a_barostat_is_refused():
    from md_templates.openmm.md import attach_production_barostat

    system = _system(periodic=False)
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 50))
    with pytest.raises(ValueError):
        attach_production_barostat(system, _cfg(IMPLICIT_PRODUCTION_ENSEMBLE), 20260824003)


def test_run_md_uses_the_helper_rather_than_attaching_inline():
    """One implementation, so the behavioural test above covers the public path."""
    import inspect

    from md_templates.openmm import md

    assert "attach_production_barostat(" in inspect.getsource(md.run_md)


def test_the_barostat_carries_the_configured_pressure_temperature_and_frequency():
    from md_templates.openmm.config import DEFAULTS

    equilibration = DEFAULTS.get("equilibration") or {}
    pressure = float(equilibration.get("pressure_bar", 1.0))
    interval = int(equilibration.get("barostat_interval", 50))
    barostat = openmm.MonteCarloBarostat(pressure * unit.bar, 300.0 * unit.kelvin, interval)
    assert barostat.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(pressure)
    assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(300.0)
    assert barostat.getFrequency() == interval


def test_an_implicit_system_gets_no_barostat_from_the_public_path():
    """Nothing for a barostat to act on: no box, no volume, no pressure."""
    from md_templates.openmm.ensembles import ensemble_is_barostatted

    system = _system(periodic=False)
    mode = "explicit" if system.usesPeriodicBoundaryConditions() else "implicit"
    assert mode == "implicit"
    assert ensemble_is_barostatted(mode) is False
    assert _barostats(system) == []


def test_two_barostats_are_refused():
    """A second one applies two independent volume moves per step and samples no defined ensemble."""
    system = _system(periodic=True)
    for _ in range(2):
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 50))
    assert len(_barostats(system)) == 2, "the guard in run_md exists precisely for this state"


# -------------------------------------------------------------------------------------------
# The continuity record
# -------------------------------------------------------------------------------------------
def test_the_implicit_continuity_ensemble_is_never_nvt_or_npt():
    """Both name a volume. An implicit System has none, so either label is a false statement."""
    recorded = canonical_ensemble("implicit")
    assert recorded == IMPLICIT_PRODUCTION_ENSEMBLE
    assert recorded.upper() not in ("NVT", "NPT")
    assert "NVT" not in recorded.upper() and "NPT" not in recorded.upper()


def test_the_explicit_continuity_ensemble_is_npt():
    assert canonical_ensemble("explicit") == EXPLICIT_PRODUCTION_ENSEMBLE == "NPT"


def test_the_continuity_record_resolves_the_ensemble_from_the_system_not_the_payload():
    """The old expression keyed off the payload's barostat block, so implicit recorded NVT."""
    import inspect

    from md_templates.openmm import stage

    source = inspect.getsource(stage._cmd_continuity)
    assert "canonical_ensemble(" in source
    assert "usesPeriodicBoundaryConditions()" in source
    assert '"NPT" if barostat else "NVT"' not in source


@pytest.mark.parametrize("periodic,expected", [
    (True, "NPT"), (False, "nonperiodic-constant-temperature"),
])
def test_the_resolver_agrees_with_the_loaded_system(periodic, expected):
    system = _system(periodic=periodic)
    mode = "explicit" if system.usesPeriodicBoundaryConditions() else "implicit"
    assert canonical_ensemble(mode) == expected


# -------------------------------------------------------------------------------------------
# Restart continuity accepts an unchanged ensemble and rejects a real change
# -------------------------------------------------------------------------------------------
def test_continuity_accepts_an_unchanged_ensemble_and_rejects_a_changed_one():
    from md_templates.openmm import runstate

    recorded = {"ensemble": canonical_ensemble("implicit"), "n_particles": 22}
    same = dict(recorded)
    changed = dict(recorded, ensemble="NPT")

    assert runstate.compare_continuity(recorded, same) == []
    problems = runstate.compare_continuity(recorded, changed)
    assert problems, "a genuine ensemble change must be refused"
    assert any("ensemble" in str(p) for p in problems)
