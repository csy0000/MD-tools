"""The arithmetic and scheduling the generated scripts depend on, without running dynamics.

These are the parts of the scientific contract that can be checked exactly: seeds, the exchange
criterion, the scaling convention and the GPU grouping. Everything here imports the same module
files that are copied into a generated project.
"""
from __future__ import annotations

import math
import time

import pytest

from .conftest import template_module

stages = template_module("md_stages")
scaling = template_module("rest2_scaling")


# --- seeds -----------------------------------------------------------------

def test_every_replica_gets_a_distinct_integrator_velocity_and_barostat_seed():
    """One seed shared across a ladder correlates the rungs, so it samples less than it looks."""
    seeds = {(replica, purpose): stages.derive_seed(20260101, "REST2", replica, purpose)
             for replica in range(8)
             for purpose in ("integrator", "velocities", "barostat")}
    assert len(set(seeds.values())) == len(seeds), "seeds collide"
    assert all(1 <= value < 2 ** 31 - 1 for value in seeds.values()), "outside OpenMM's range"


def test_a_seed_is_never_zero_because_openmm_reads_zero_as_choose_randomly():
    values = [stages.derive_seed(base, "cMD", "integrator") for base in range(500)]
    assert 0 not in values


def test_the_same_base_and_purpose_always_give_the_same_seed():
    assert (stages.derive_seed(7, "REST2", 3, "integrator")
            == stages.derive_seed(7, "REST2", 3, "integrator"))


# --- durations -------------------------------------------------------------

def test_a_duration_that_is_not_a_whole_number_of_steps_is_refused():
    """A segment rounded shorter by one step drifts the exchange schedule while looking healthy."""
    assert stages.steps_for(10.0, 2.0) == 5000
    with pytest.raises(SystemExit) as error:
        stages.steps_for(0.003, 2.0)
    assert "whole number" in str(error.value)


# --- REST2 exchange criterion ----------------------------------------------

def test_the_pv_terms_cancel_at_a_common_beta_and_pressure():
    """Replicas differ by Hamiltonian, not by thermostat, so beta is shared and pV drops out.

    The swap moves each configuration WITH its box, so the same two volumes appear on both sides
    of the criterion. Checked against the plain energy-only form.
    """
    beta = 1.0 / (0.008314462618 * 300.0)
    pressure, volume_i, volume_j = 1.0, 27.0, 31.5          # deliberately different volumes
    energy_i, energy_j, energy_ij, energy_ji = -2500.0, -2480.0, -2495.0, -2470.0

    with_pv = scaling.exchange_log_acceptance(
        scaling.reduced_potential(energy_i, beta, pressure, volume_i),
        scaling.reduced_potential(energy_j, beta, pressure, volume_j),
        scaling.reduced_potential(energy_ij, beta, pressure, volume_j),
        scaling.reduced_potential(energy_ji, beta, pressure, volume_i))
    energy_only = beta * (energy_i + energy_j - energy_ij - energy_ji)

    assert with_pv == pytest.approx(energy_only, rel=1e-12, abs=1e-12)


def test_the_criterion_is_the_documented_expression():
    beta = 1.0 / (0.008314462618 * 300.0)
    values = (-10.0, -20.0, -5.0, -8.0)
    reduced = [scaling.reduced_potential(v, beta) for v in values]
    assert scaling.exchange_log_acceptance(*reduced) == pytest.approx(
        beta * (values[0] + values[1] - values[2] - values[3]))


def test_a_two_replica_ladder_offers_its_pair_in_the_phase_the_runner_falls_back_to():
    """Alternating regardless would exchange on every other round only, and a run resumed on an
    odd index would keep landing on the empty phase."""
    assert scaling.exchange_pairs(2, 0) == [(0, 1)]
    assert scaling.exchange_pairs(2, 1) == []
    for attempt_index in range(6):
        phase = attempt_index % 2
        pairs = scaling.exchange_pairs(2, phase) or scaling.exchange_pairs(2, (phase + 1) % 2)
        assert pairs == [(0, 1)], attempt_index


# --- the scaling convention ------------------------------------------------

def test_the_ladder_scales_solute_solute_by_s_and_solute_environment_by_its_root():
    """s = (1-tau)^2 and sqrt(s) = (1-tau), applied to charges, epsilons and exceptions."""
    from openmm import NonbondedForce, System

    tau = 0.2
    s = scaling.scale_factor_for_tau(tau)
    assert s == pytest.approx(0.64)

    system = System()
    force = NonbondedForce()
    for _ in range(3):
        system.addParticle(12.0)
    force.addParticle(0.5, 0.3, 0.8)          # 0: solute
    force.addParticle(-0.4, 0.3, 0.6)         # 1: solute
    force.addParticle(0.2, 0.3, 0.4)          # 2: environment
    force.addException(0, 1, 0.1, 0.3, 0.5)   # solute-solute      -> s
    force.addException(0, 2, 0.2, 0.3, 0.7)   # solute-environment -> sqrt(s)
    system.addForce(force)

    scaled = scaling.build_scaled_system(system, [0, 1], tau)
    out = scaled.getForce(0)
    root = math.sqrt(s)

    charge, _, epsilon = out.getParticleParameters(0)
    assert charge.value_in_unit(charge.unit) == pytest.approx(0.5 * root)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.8 * s)

    charge, _, epsilon = out.getParticleParameters(2)
    assert charge.value_in_unit(charge.unit) == pytest.approx(0.2), "environment must not scale"
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.4)

    _, _, product, _, epsilon = out.getExceptionParameters(0)
    assert product.value_in_unit(product.unit) == pytest.approx(0.1 * s)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.5 * s)

    _, _, product, _, epsilon = out.getExceptionParameters(1)
    assert product.value_in_unit(product.unit) == pytest.approx(0.2 * root)
    assert epsilon.value_in_unit(epsilon.unit) == pytest.approx(0.7 * root)


def test_the_cold_rung_is_the_unmodified_system():
    assert scaling.scale_factor_for_tau(0.0) == 1.0


# --- GPU grouping ----------------------------------------------------------

def test_replicas_are_assigned_round_robin_to_at_most_one_device_each():
    assert stages.device_groups(6, [0, 1]) == [[0, 2, 4], [1, 3, 5]]
    assert stages.device_groups(2, [0, 1, 2, 3]) == [[0], [1]], "no more devices than replicas"
    assert stages.device_groups(4, []) == [[0, 1, 2, 3]], "no devices: one sequential group"


def test_two_devices_propagate_concurrently_and_one_device_does_not():
    """Mocked stepping, so this proves the schedule without needing CUDA.

    The failure it guards against is a multi-GPU run that costs N GPUs and takes as long as one.
    """
    def timed(groups):
        spans = {}

        def step(replica):
            start = time.monotonic()
            time.sleep(0.25)
            spans[replica] = (start, time.monotonic())

        stages.propagate_segment(groups, step)
        return spans

    two_devices = timed([[0], [1]])
    latest_start = max(span[0] for span in two_devices.values())
    earliest_end = min(span[1] for span in two_devices.values())
    assert earliest_end > latest_start, "different devices must overlap"

    one_device = timed([[0, 1]])
    assert one_device[0][1] <= one_device[1][0], "replicas sharing a device must not overlap"


def test_a_failure_inside_one_group_reaches_the_caller_before_any_exchange():
    def step(replica):
        if replica == 3:
            raise RuntimeError("particle position is NaN")

    with pytest.raises(RuntimeError, match="NaN"):
        stages.propagate_segment([[0, 1], [2, 3]], step)
