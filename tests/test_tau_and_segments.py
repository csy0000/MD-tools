"""Tau parameterisation and segment arithmetic.

These are the two pieces of the REST2 contract that are pure mathematics, so they are tested as
mathematics: exact identities where exactness is achievable, and explicit refusal where a value
cannot be represented as whole steps.
"""
from __future__ import annotations

import math

import pytest

from md_templates.openmm.segments import (
    SegmentPlan,
    plan_segment,
    reporting_interval_steps,
    steps_for_duration,
)
from md_templates.openmm.tau import (
    build_tau_ladder,
    ladder_diagnostics,
    linear_tau_ladder,
    map_replicas_to_devices,
    scale_factors_for_ladder,
    scaling_for_tau,
    scalings_for_ladder,
)

#: The two ladders the worked examples use.
ALANINE_TAU_LADDER = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
RGDFV_TAU_COUNT = 10


# ---------------------------------------------------------------------------------------------
# the tau -> s mapping, exactly
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.0, 0.1, 0.25, 0.3, 0.5, 0.75, 0.999])
def test_s_is_exactly_one_minus_tau_squared(tau):
    scaling = scaling_for_tau(tau)
    assert scaling.sqrt_s == 1.0 - tau
    assert scaling.s == (1.0 - tau) * (1.0 - tau)


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
def test_sqrt_s_squared_is_exactly_s(tau):
    """`s` is built as the square of the same float as `sqrt_s`, so this is exact, not approximate.

    If `s` were computed independently (say via `math.pow`), this could differ in the last bit and
    the solute-environment coupling would not be the exact square root of the solute-solute one.
    """
    scaling = scaling_for_tau(tau)
    assert scaling.sqrt_s * scaling.sqrt_s == scaling.s


def test_cold_replica_is_the_unscaled_hamiltonian():
    scaling = scaling_for_tau(0.0)
    assert scaling.s == 1.0
    assert scaling.sqrt_s == 1.0


def test_tau_of_one_is_refused():
    with pytest.raises(ValueError, match="not a REST2 replica"):
        scaling_for_tau(1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_tau_out_of_range_is_refused(bad):
    with pytest.raises(ValueError, match="tau must satisfy"):
        scaling_for_tau(bad)


# ---------------------------------------------------------------------------------------------
# the two worked ladders
# ---------------------------------------------------------------------------------------------

def test_alanine_ladder_is_six_replicas_zero_to_half():
    ladder = build_tau_ladder(minimum=0.0, maximum=0.5, count=6)
    assert len(ladder) == 6
    assert ladder == pytest.approx(ALANINE_TAU_LADDER, abs=1e-15)
    assert ladder[0] == 0.0        # exact endpoints, not accumulated arithmetic
    assert ladder[-1] == 0.5


def test_rgdfv_ladder_is_ten_replicas_zero_to_half():
    ladder = build_tau_ladder(minimum=0.0, maximum=0.5, count=RGDFV_TAU_COUNT)
    assert len(ladder) == RGDFV_TAU_COUNT
    assert ladder[0] == 0.0
    assert ladder[-1] == 0.5
    spacings = [b - a for a, b in zip(ladder, ladder[1:])]
    assert spacings == pytest.approx([0.5 / 9] * 9, rel=1e-12)


@pytest.mark.parametrize("count", [6, RGDFV_TAU_COUNT])
def test_ladder_mapping_is_exact_across_both_ladders(count):
    """The instruction's exactness requirement, applied to every rung of both ladders."""
    ladder = build_tau_ladder(minimum=0.0, maximum=0.5, count=count)
    for scaling in scalings_for_ladder(ladder):
        assert scaling.sqrt_s == 1.0 - scaling.tau
        assert scaling.s == (1.0 - scaling.tau) ** 2


@pytest.mark.parametrize("count", [6, RGDFV_TAU_COUNT])
def test_scale_factors_descend_from_one(count):
    """The existing System builder consumes `s`; it must arrive cold-first and descending."""
    ladder = build_tau_ladder(minimum=0.0, maximum=0.5, count=count)
    scale_factors = scale_factors_for_ladder(ladder)
    assert scale_factors[0] == 1.0
    assert all(b < a for a, b in zip(scale_factors, scale_factors[1:]))
    assert all(0.0 < s <= 1.0 for s in scale_factors)


def test_hot_rung_of_the_worked_ladders_is_a_quarter():
    """tau = 0.5 -> s = 0.25, i.e. an effective temperature of 4x the bath."""
    scaling = scaling_for_tau(0.5)
    assert scaling.s == 0.25
    assert scaling.effective_temperature_kelvin(300.0) == 1200.0


def test_ladder_must_start_cold():
    with pytest.raises(ValueError, match="cold, physical replica"):
        scalings_for_ladder([0.1, 0.2, 0.3])


def test_ladder_must_strictly_increase():
    with pytest.raises(ValueError, match="strictly increase"):
        scalings_for_ladder([0.0, 0.2, 0.2, 0.3])


def test_ladder_diagnostics_label_every_derived_column():
    rows = ladder_diagnostics(ALANINE_TAU_LADDER, base_temperature_kelvin=300.0)
    assert len(rows) == 6
    for row in rows:
        assert set(row) == {
            "replica_index", "tau", "derived_s", "derived_sqrt_s",
            "derived_effective_temperature_kelvin",
        }
        # every column except tau announces that it is derived
        derived = [key for key in row if key.startswith("derived_")]
        assert len(derived) == 3


def test_non_linear_interpolation_is_refused_rather_than_assumed():
    with pytest.raises(ValueError, match="not supported"):
        build_tau_ladder(0.0, 0.5, 6, interpolation="geometric")


def test_ladder_needs_a_span():
    with pytest.raises(ValueError, match="not a ladder"):
        linear_tau_ladder(0.5, 0.5, 6)


# ---------------------------------------------------------------------------------------------
# exact step conversion
# ---------------------------------------------------------------------------------------------

# durations and timesteps in picoseconds, matching the canonical unit of the spec layer
_PS_PER_NS = 1000.0
_PS_PER_FS = 0.001


def test_five_ns_at_two_fs_is_exactly_two_and_a_half_million_steps():
    steps = steps_for_duration(5.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                               duration_source="5 ns", timestep_source="2 fs")
    assert steps == 2_500_000


def test_ten_ps_at_two_fs_is_exactly_five_thousand_steps():
    """The NVT and NPT equilibration stages of the worked protocol."""
    steps = steps_for_duration(10.0, 2.0 * _PS_PER_FS,
                               duration_source="10 ps", timestep_source="2 fs")
    assert steps == 5_000


def test_one_ns_at_two_fs_is_exactly_five_hundred_thousand_steps():
    """The conventional-MD stage of the worked protocol."""
    steps = steps_for_duration(1.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                               duration_source="1 ns", timestep_source="2 fs")
    assert steps == 500_000


def test_non_integer_step_count_is_refused_not_rounded():
    with pytest.raises(ValueError, match="not a whole number of"):
        steps_for_duration(5.0 * _PS_PER_NS, 3.0 * _PS_PER_FS,
                           duration_source="5 ns", timestep_source="3 fs")


def test_refusal_names_the_nearest_representable_durations():
    """A refusal that does not tell you what would work is a worse error message."""
    with pytest.raises(ValueError) as excinfo:
        steps_for_duration(7.0, 2.0 * _PS_PER_FS + 1e-7,
                           duration_source="7 ps", timestep_source="2.0001 fs")
    assert "nearest whole values" in str(excinfo.value)


def test_duration_shorter_than_one_step_is_refused():
    with pytest.raises(ValueError, match="does not contain a single whole unit"):
        steps_for_duration(0.001, 1.0, duration_source="1 fs", timestep_source="1 ps")


def test_reporting_intervals_of_the_worked_protocol_are_exact():
    full_system = reporting_interval_steps(100.0, 2.0 * _PS_PER_FS,
                                           interval_source="100 ps", timestep_source="2 fs",
                                           label="reporting.full_system_interval")
    selected = reporting_interval_steps(10.0, 2.0 * _PS_PER_FS,
                                        interval_source="10 ps", timestep_source="2 fs",
                                        label="reporting.selected_atoms_interval")
    assert full_system == 50_000
    assert selected == 5_000
    # the two streams must stay commensurate or frames drift apart in physical time
    assert full_system % selected == 0


# ---------------------------------------------------------------------------------------------
# exchange cadence
# ---------------------------------------------------------------------------------------------

def test_the_worked_exchange_contract():
    """5 ns, 2 fs, 100 exchanges -> 2,500,000 steps and 25,000 steps (50 ps) per round."""
    plan = plan_segment(5.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                        duration_source="5 ns", timestep_source="2 fs",
                        number_of_exchanges_per_segment=100)
    assert plan == SegmentPlan(steps_per_segment=2_500_000,
                               steps_per_exchange=25_000,
                               number_of_exchanges_per_segment=100)
    # 25,000 steps at 2 fs is 50 ps
    assert plan.steps_per_exchange * 2.0 * _PS_PER_FS == pytest.approx(50.0)


def test_indivisible_exchange_count_is_refused():
    with pytest.raises(ValueError, match="not divisible by"):
        plan_segment(5.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                     duration_source="5 ns", timestep_source="2 fs",
                     number_of_exchanges_per_segment=3)


def test_exchange_refusal_explains_the_watermark_consequence():
    with pytest.raises(ValueError) as excinfo:
        plan_segment(5.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                     duration_source="5 ns", timestep_source="2 fs",
                     number_of_exchanges_per_segment=7)
    assert "committed watermark" in str(excinfo.value)


def test_conventional_md_segment_has_no_exchange_cadence():
    plan = plan_segment(1.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                        duration_source="1 ns", timestep_source="2 fs")
    assert plan.steps_per_segment == 500_000
    assert plan.steps_per_exchange is None
    assert not plan.has_exchanges


def test_zero_exchanges_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        plan_segment(5.0 * _PS_PER_NS, 2.0 * _PS_PER_FS,
                     duration_source="5 ns", timestep_source="2 fs",
                     number_of_exchanges_per_segment=0)


# ---------------------------------------------------------------------------------------------
# deterministic replica -> device mapping
# ---------------------------------------------------------------------------------------------

def test_one_replica_per_device_when_they_match():
    assert map_replicas_to_devices(6, [1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]


def test_replicas_share_devices_when_they_outnumber_them():
    """Ten replicas on four GPUs is allowed, deterministic, and dealt round-robin."""
    mapping = map_replicas_to_devices(10, [0, 1, 2, 3])
    assert mapping == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert len(mapping) == 10


def test_mapping_is_a_pure_function_of_its_inputs():
    first = map_replicas_to_devices(10, [1, 2, 3])
    second = map_replicas_to_devices(10, [1, 2, 3])
    assert first == second


def test_device_order_is_honoured_not_sorted():
    """An ordered list means what it says; re-sorting it would silently move replicas."""
    assert map_replicas_to_devices(3, [5, 1, 3]) == [5, 1, 3]


def test_fewer_replicas_than_devices_uses_a_prefix():
    assert map_replicas_to_devices(2, [4, 5, 6, 7]) == [4, 5]


def test_empty_device_list_is_refused():
    with pytest.raises(ValueError, match="at least one device"):
        map_replicas_to_devices(6, [])


def test_duplicate_devices_are_refused():
    with pytest.raises(ValueError, match="more than once"):
        map_replicas_to_devices(6, [0, 1, 1])


# ---------------------------------------------------------------------------------------------
# water packing-model resolution
#
# OpenMM's Modeller.addSolvent can only BUILD a pre-equilibrated box for a handful of models. The
# model that is SIMULATED is decided by the force field, so a model without its own box is packed
# with a same-topology stand-in. Getting this wrong silently would solvate one water model and
# parameterise another.
# ---------------------------------------------------------------------------------------------

from md_templates.openmm.solvation import resolve_packing_model  # noqa: E402


@pytest.mark.parametrize("model", ["tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"])
def test_models_openmm_can_build_directly_are_not_substituted(model):
    packing, substituted = resolve_packing_model(model)
    assert packing == model
    assert substituted is False


def test_opc_is_packed_in_a_four_site_box():
    """OPC is a 4-site model, so it borrows TIP4P-Ew geometry -- OpenMM documents this."""
    packing, substituted = resolve_packing_model("opc")
    assert packing == "tip4pew"
    assert substituted is True


def test_opc3_is_packed_in_a_three_site_box():
    packing, substituted = resolve_packing_model("opc3")
    assert packing == "tip3p"
    assert substituted is True


@pytest.mark.parametrize("model,expected_sites", [("opc", "tip4pew"), ("tip4pfb", "tip4pew"),
                                                  ("opc3", "tip3p"), ("tip3pfb", "tip3p")])
def test_substitutions_preserve_the_site_count(model, expected_sites):
    """A 4-site model packed into a 3-site box would leave its virtual sites unplaced."""
    packing, _ = resolve_packing_model(model)
    assert packing == expected_sites


def test_an_undeclared_water_model_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no same-topology stand-in is declared"):
        resolve_packing_model("tip4p2005")


# ---------------------------------------------------------------------------------------------
# the device mapping must reach the EXECUTION path, not just exist as a function
#
# map_replicas_to_devices was correct and unreachable for a while: the CLI accepted only a single
# --device, so every replica landed on one GPU no matter what the mapping said. These tests pin the
# wiring, not the arithmetic.
# ---------------------------------------------------------------------------------------------

from md_templates.openmm.rest2 import _resolve_replica_devices  # noqa: E402


def _cfg(platform: str, device=None, device_indices=None) -> dict:
    production = {"platform": platform, "device_index": device}
    if device_indices is not None:
        production["device_indices"] = device_indices
    return {"production": production}


def test_a_device_list_is_dealt_across_replicas():
    assert _resolve_replica_devices(_cfg("CUDA", device_indices=[4, 5, 6]), 3) == [4, 5, 6]


def test_replicas_share_devices_when_they_outnumber_them_in_the_runner():
    assert _resolve_replica_devices(_cfg("CUDA", device_indices=[1, 2]), 5) == [1, 2, 1, 2, 1]


def test_a_single_device_still_applies_to_every_replica():
    """The previous behaviour must survive: --device alone puts everything on one GPU."""
    assert _resolve_replica_devices(_cfg("CUDA", device="2"), 3) == [2, 2, 2]


def test_the_device_list_wins_over_the_single_device():
    resolved = _resolve_replica_devices(_cfg("CUDA", device="0", device_indices=[7, 8]), 4)
    assert resolved == [7, 8, 7, 8]


def test_cpu_has_no_device_mapping():
    assert _resolve_replica_devices(_cfg("CPU"), 6) is None


def test_no_device_named_now_selects_devices_automatically(monkeypatch):
    """Superseded behaviour, deliberately.

    This asserted `is None` -- "name no device and OpenMM picks". That is what let a six-replica
    ladder quietly run on one card. Naming nothing now means "use the maximum useful number of free
    visible GPUs", which is min(n_replicas, n_available): one Context per replica, so more devices
    than replicas cannot be used and more replicas than devices means sharing.

    Mocked, so the assertion is about the rule rather than about whichever GPUs this machine has
    free at the moment the suite runs.
    """
    from md_templates.openmm import gpus

    fake = [{"physical_index": i, "logical_index": i, "uuid": f"GPU-{i:04d}",
             "name": "FakeGPU", "memory_total_mib": "10240"} for i in range(8)]
    monkeypatch.setattr(gpus, "visible_devices", lambda: fake)
    monkeypatch.setattr(gpus, "busy_physical_devices", set)

    assert _resolve_replica_devices(_cfg("CUDA"), 3) == [0, 1, 2]
    assert _resolve_replica_devices(_cfg("CUDA"), 8) == list(range(8))
    # ten replicas, eight devices: round-robin, two devices carry two replicas each
    assert _resolve_replica_devices(_cfg("CUDA"), 10) == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
