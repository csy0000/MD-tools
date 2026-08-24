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
def test_run_md_attaches_exactly_one_barostat_for_an_explicit_npt_system():
    """The regression test for the defect: NPT label, no barostat, fixed-volume propagation."""
    import inspect

    from md_templates.openmm import md

    source = inspect.getsource(md.run_md)
    assert "MonteCarloBarostat" in source, (
        "run_md must attach a barostat; the bundle System has none, so an NPT label without one "
        "means the run integrates at fixed volume")
    assert "requires exactly" in source, "run_md must refuse a System with the wrong barostat count"


def test_run_md_derives_a_legal_independently_seeded_barostat():
    """A dated master seed exceeds OpenMM's 32-bit field; arithmetic on it crashes at the Context."""
    import inspect

    from md_templates.openmm import md
    from md_templates.openmm.seeds import as_openmm_seed, derive_seed, stage_purpose

    source = inspect.getsource(md.run_md)
    assert "derive_seed(" in source and "as_openmm_seed(" in source

    seed = as_openmm_seed(derive_seed(20260824003, stage_purpose("cMD", "barostat")))
    assert 1 <= seed <= 2**31 - 1
    barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 50)
    barostat.setRandomNumberSeed(seed)          # must not raise
    assert barostat.getRandomNumberSeed() == seed


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
