"""Fixed-tau single-walker MD: one walker on one rung of the REST2 ladder.

Two claims are load-bearing and are checked here rather than described:

* tau = 0 must leave the System untouched, so a fixed-tau walker at tau = 0 IS conventional MD;
* tau = t must build the same Hamiltonian the REST2 ladder builds at that rung, because the whole
  point of the capability is that a reservoir walker and the rung it feeds are the same physics.

The generalised-Born tests exist because the template scaler previously omitted `CustomGBForce`
entirely: under implicit solvent every other term scaled while the full solvation energy stayed at
s = 1, which is a different Hamiltonian that runs to completion and looks healthy.
"""
from __future__ import annotations

import math

import pytest
from openmm import CustomGBForce, HarmonicBondForce, NonbondedForce, System

from .conftest import template_module

scaling = template_module("rest2_scaling")
stages = template_module("md_stages")


def _gb_system(n=3):
    """A System whose only energy is a CustomGBForce, plus a bond term that must not scale."""
    system = System()
    for _ in range(n):
        system.addParticle(12.0)
    gb = CustomGBForce()
    gb.addPerParticleParameter("q")
    gb.addComputedValue("radius", "1.0", CustomGBForce.SingleParticle)
    # A charge-independent term, which is exactly what charge scaling alone would miss.
    gb.addEnergyTerm("2.0*q*q + 5.0", CustomGBForce.SingleParticle)
    for _ in range(n):
        gb.addParticle([0.5])
    system.addForce(gb)
    bonds = HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    system.addForce(bonds)
    return system


def _gb_scale_value(system):
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, CustomGBForce):
            names = [force.getGlobalParameterName(i)
                     for i in range(force.getNumGlobalParameters())]
            if scaling.REST2_GB_SCALE_PARAMETER not in names:
                return None
            return force.getGlobalParameterDefaultValue(
                names.index(scaling.REST2_GB_SCALE_PARAMETER))
    raise AssertionError("no CustomGBForce in the System")


# --- tau = 0 is conventional MD ---------------------------------------------

def test_tau_zero_returns_the_system_unmodified():
    """Not "close to": a fixed-tau walker at tau = 0 must be the cMD trajectory, not a copy of it."""
    system = _gb_system()
    scaled = scaling.build_scaled_system(system, [0, 1, 2], 0.0)
    assert _gb_scale_value(scaled) is None, "tau = 0 must not even inject the GB parameter"
    original = system.getForce(0)
    copy = scaled.getForce(0)
    assert copy.getNumEnergyTerms() == original.getNumEnergyTerms()
    for term in range(original.getNumEnergyTerms()):
        assert copy.getEnergyTermParameters(term) == original.getEnergyTermParameters(term)


# --- the generalised-Born energy scales, and scales by s ---------------------

@pytest.mark.parametrize("tau", [1 / 6, 1 / 3, 0.5])
def test_the_whole_generalised_born_energy_scales_by_s(tau):
    """GBn2 carries a non-polar term with no charge dependence; charge scaling would miss it."""
    scaled = scaling.build_scaled_system(_gb_system(), [0, 1, 2], tau)
    assert _gb_scale_value(scaled) == pytest.approx(scaling.scale_factor_for_tau(tau))


def test_a_partial_enhanced_region_is_refused_under_generalised_born():
    """Every Born radius depends on every other atom, so a partial selection is not separable."""
    with pytest.raises(ValueError, match="entire system"):
        scaling.build_scaled_system(_gb_system(), [0, 1], 0.5)


def test_bond_terms_are_not_scaled():
    """REST2 lowers conformational barriers, not covalent geometry."""
    scaled = scaling.build_scaled_system(_gb_system(), [0, 1, 2], 0.5)
    bonds = [scaled.getForce(i) for i in range(scaled.getNumForces())
             if isinstance(scaled.getForce(i), HarmonicBondForce)][0]
    _, _, length, k = bonds.getBondParameters(0)
    assert k.value_in_unit(k.unit) == pytest.approx(1000.0)


# --- an unclassifiable force is refused rather than left at the wrong scale ---

def test_an_unhandled_energy_bearing_force_is_refused():
    from openmm import CustomExternalForce

    system = System()
    system.addParticle(12.0)
    system.addForce(CustomExternalForce("x^2"))
    with pytest.raises(scaling.UnclassifiedForceError, match="cannot classify"):
        scaling.build_scaled_system(system, [0], 0.5)


def test_the_audit_accepts_a_system_made_only_of_known_forces():
    buckets = scaling.audit_force_classes(_gb_system())
    assert [name for _, name in buckets["scaled"]] == ["CustomGBForce"]
    assert [name for _, name in buckets["unscaled_by_convention"]] == ["HarmonicBondForce"]


# --- configuration ----------------------------------------------------------

@pytest.mark.parametrize("tau", [-0.1, 1.0, 1.5, "warm"])
def test_a_tau_outside_the_ladder_is_refused_by_the_config(tau):
    from md_templates.openmm.config import ConfigError, resolve_md_config

    document = {"methods": ["cMD"], "common": {"timestep_fs": 2.0, "pressure_bar": None},
                "cMD": {"ensemble": "NVT", "tau": tau, "duration_ns": 1}}
    with pytest.raises(ConfigError, match="tau"):
        resolve_md_config(document, implicit=True)


def test_the_default_cmd_block_is_ordinary_conventional_md():
    """tau must default to 0: a default that scaled the Hamiltonian would be a silent change."""
    from md_templates.openmm.defaults import md_defaults

    assert md_defaults(methods=["cMD"])["cMD"]["tau"] == 0.0


# --- the uncommitted tail ---------------------------------------------------

def test_a_reporter_tail_past_the_checkpoint_is_discarded(tmp_path):
    """A checkpoint every 100 ps and a trajectory every 10 ps means a crash leaves extra frames.

    Appending after them duplicates that interval, and nothing downstream reports it: the run
    completes and the step column simply stops being monotonic.
    """
    table = tmp_path / "production.csv"
    table.write_text("#\"Step\",\"Time\"\n" + "".join(f"{i * 100},{i}\n" for i in range(1, 21)))
    assert stages.truncate_table(table, 12) == 12
    lines = table.read_text().splitlines()
    assert len(lines) == 13 and lines[0].startswith("#"), "one header, twelve rows"
    assert lines[-1].startswith("1200,")


def test_truncating_a_table_that_is_already_short_enough_changes_nothing(tmp_path):
    table = tmp_path / "production.csv"
    original = "#\"Step\"\n" + "".join(f"{i}\n" for i in range(1, 6))
    table.write_text(original)
    assert stages.truncate_table(table, 10) == 5
    assert table.read_text() == original
