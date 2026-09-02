"""How a platform was chosen, recorded so the four cases are told apart.

A CPU result has three innocent explanations and one alarming one, and a record that cannot
separate them is worth very little:

    the machine has no GPU and is configured for CPU
    somebody typed --cpu for this run
    nobody said anything and CUDA is simply the default
    CUDA was quietly unavailable and the run fell back          <- must be impossible

The fourth is prevented by there being no fallback at all. The first three are told apart by the
record, and the field that does it is `platform_selection`.

`explicit_cpu` means exactly one thing: **this invocation supplied `--cpu`**. It briefly also
meant "a machine configured for CPU", which made a machine-wide default indistinguishable from a
per-run override in every log that recorded it. Every row of the table below asserts all four
fields, including the two that were once conflated.
"""
from __future__ import annotations

import pytest

from md_tools.openmm.platform_policy import PlatformRequest, acceleration_record

#: selection -> (machine settings, --cpu, expected name, explicit_cpu, cli_cpu_override,
#:               platform_selection, requested_policy)
CASES = {
    "built-in CUDA": ({}, False,
                      "CUDA", False, False, "built-in-default", "default-cuda"),
    "machine CUDA": ({"platform": "CUDA", "origin": "machine.openmm"}, False,
                     "CUDA", False, False, "machine-config", "default-cuda"),
    "machine CPU": ({"platform": "CPU", "origin": "machine.openmm"}, False,
                    "CPU", False, False, "machine-config", "machine-cpu"),
    "CLI --cpu": ({"platform": "CUDA", "origin": "machine.openmm"}, True,
                  "CPU", True, True, "cli-override", "explicit-cpu"),
}


class _Resolution:
    """The shape `acceleration_record` reads, without needing a real OpenMM Platform."""

    def __init__(self, request):
        self.request = request
        self.properties: dict[str, str] = {}
        self.device_index = None
        self.visible_devices = ()
        self.name = request.name


@pytest.mark.parametrize("case", list(CASES))
def test_every_field_of_the_platform_selection(case):
    machine, cpu, name, explicit, override, selection, policy = CASES[case]
    request = PlatformRequest.from_machine(machine, cpu=cpu)

    assert request.name == name, case
    assert request.explicit_cpu is explicit, (
        f"{case}: explicit_cpu must mean ONLY that this invocation supplied --cpu")
    assert request.cli_cpu_override is override, case
    assert request.platform_selection == selection, case


@pytest.mark.parametrize("case", list(CASES))
def test_the_record_carries_all_four_fields(case):
    machine, cpu, name, explicit, override, selection, policy = CASES[case]
    record = acceleration_record(_Resolution(PlatformRequest.from_machine(machine, cpu=cpu)))

    assert record["resolved_platform"] == name, case
    assert record["requested_platform"] == name, case
    assert record["explicit_cpu"] is explicit, case
    assert record["cli_cpu_override"] is override, case
    assert record["platform_selection"] == selection, case
    assert record["requested_policy"] == policy, case


def test_a_machine_cpu_default_is_never_described_as_a_command_line_choice():
    """The regression, stated on its own so its name appears in a failure.

    `machine.openmm.platform: CPU` was recorded with `explicit_cpu: true` and
    `requested_policy: explicit-cpu`, which is a claim about what a person typed, made on the
    basis of a file they wrote months earlier on a different machine.
    """
    record = acceleration_record(_Resolution(
        PlatformRequest.from_machine({"platform": "CPU", "origin": "machine.openmm"})))
    assert record["explicit_cpu"] is False
    assert record["cli_cpu_override"] is False
    assert record["requested_policy"] == "machine-cpu"
    assert record["platform_selection"] == "machine-config"


def test_the_default_is_cuda_with_no_one_having_chosen_it():
    """No configuration, no flag. The record says so rather than implying somebody decided."""
    record = acceleration_record(_Resolution(PlatformRequest.from_machine({})))
    assert record["platform_selection"] == "built-in-default"
    assert record["requested_policy"] == "default-cuda"
    assert record["explicit_cpu"] is False


def test_cpu_and_device_together_are_refused():
    """`--device` is placement, never platform. There is no device to place on the CPU."""
    from md_tools.run.preflight import PreflightError, _check_command_line

    with pytest.raises(PreflightError, match="contradictory"):
        _check_command_line(cpu=True, device=0, number_of_groups=None)
