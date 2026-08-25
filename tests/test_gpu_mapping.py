"""Mapping nvidia-smi devices to CUDA ordinals, and replicas to devices.

The assumption this file exists to kill: that `nvidia-smi` index N and CUDA logical ordinal N are
the same card. They are not. `nvidia-smi` numbers by PCI bus order; CUDA's DEFAULT is
`CUDA_DEVICE_ORDER=FASTEST_FIRST`, which sorts by capability. On a mixed machine the two orders
differ, so a busy-device exclusion computed from nvidia-smi indices gets applied to a different
card, and the ladder runs on hardware nobody chose.

Everything here is mocked, so it runs anywhere and asserts the mapping logic rather than this
machine's hardware. A separate non-destructive diagnostic records what the real machine reports.
"""
from __future__ import annotations

import json

import pytest

from md_templates.openmm import gpus

#: Deliberately mixed and NOT in capability order, so a FASTEST_FIRST reordering would be visible.
#: The A5000 sits at PCI bus 1A and nvidia-smi index 0; the two 3080s follow.
SMI_GPUS = (
    "0, GPU-aaaa0000-0000-0000-0000-000000000000, NVIDIA RTX A5000, 24564, 00000000:1A:00.0\n"
    "1, GPU-bbbb1111-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3080, 10240, 00000000:1B:00.0\n"
    "2, GPU-cccc2222-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3080, 10240, 00000000:1C:00.0\n"
)
UUIDS = ["GPU-aaaa0000-0000-0000-0000-000000000000",
         "GPU-bbbb1111-0000-0000-0000-000000000000",
         "GPU-cccc2222-0000-0000-0000-000000000000"]


@pytest.fixture
def smi(monkeypatch):
    """Mock `nvidia-smi`, returning a mutable dict so a test can set what is busy."""
    state = {"gpus": SMI_GPUS, "busy": "", "listing": ""}

    def fake(args):
        joined = " ".join(args)
        if "--query-compute-apps" in joined:
            return state["busy"]
        if "--query-gpu" in joined:
            return state["gpus"]
        if args and args[0] == "-L":
            return state["listing"]
        return None

    monkeypatch.setattr(gpus, "_run_smi", fake)
    monkeypatch.setattr(gpus.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return state


# -------------------------------------------------------------------------------------------
# Device order
# -------------------------------------------------------------------------------------------
def test_pci_bus_order_is_requested_when_the_caller_left_it_unset(monkeypatch):
    """Without this, CUDA orders by capability and the ordinals stop matching nvidia-smi."""
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    record = gpus.ensure_pci_bus_id_order()
    assert record["action"] == "set"
    assert gpus.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_an_explicit_conflicting_order_is_reported_not_overridden(monkeypatch):
    """FASTEST_FIRST is a real choice. Silently rewriting it would be worse than saying so."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    record = gpus.ensure_pci_bus_id_order()
    assert record["action"] == "conflicting"
    assert record["trustworthy"] is False
    assert gpus.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    # identity must still be recoverable, and it is -- by UUID, which no ordering affects
    assert "UUID" in record["note"] or "uuid" in record["note"]


def test_discovery_records_the_order_action_so_a_reader_can_weigh_the_ordinals(smi):
    found = gpus.discover_devices()
    assert found["device_order_action"]["cuda_device_order"] == "PCI_BUS_ID"


def test_cuda_order_differing_from_smi_index_order_still_identifies_cards_by_uuid(smi, monkeypatch):
    """The core scenario: CUDA enumerates the fastest card first, nvidia-smi enumerates by bus.

    Selection is driven by UUID and PCI bus ID, both of which are order-invariant, so the record
    still names the physical card even when the caller forced a capability ordering.
    """
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    record = gpus.select_devices(3)
    identities = {d["uuid"]: d["pci_bus_id"] for d in record["device_identity"]}
    assert identities[UUIDS[0]] == "00000000:1A:00.0"
    assert identities[UUIDS[2]] == "00000000:1C:00.0"
    assert record["discovery"]["device_order_action"]["trustworthy"] is False


# -------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES forms
# -------------------------------------------------------------------------------------------
def test_integer_cuda_visible_devices_maps_logical_onto_the_named_physical(smi, monkeypatch):
    """With `2,0` visible, logical 0 IS physical 2. Passing 2 to OpenMM would select nothing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    visible = gpus.visible_devices()
    assert [d["logical_index"] for d in visible] == [0, 1]
    assert [d["physical_index"] for d in visible] == [2, 0]
    assert [d["uuid"] for d in visible] == [UUIDS[2], UUIDS[0]]


def test_uuid_cuda_visible_devices_is_resolved_to_physical_identity(smi, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{UUIDS[1]},{UUIDS[0]}")
    visible = gpus.visible_devices()
    assert [d["physical_index"] for d in visible] == [1, 0]
    assert [d["pci_bus_id"] for d in visible] == ["00000000:1B:00.0", "00000000:1A:00.0"]


def test_an_empty_variable_hides_everything(smi, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpus.visible_devices() == []


def test_a_mig_uuid_is_resolved_to_its_parent_gpu(smi, monkeypatch):
    """MIG instances are absent from --query-gpu, so they must come from the -L listing."""
    smi["listing"] = (
        "GPU 0: NVIDIA RTX A5000 (UUID: GPU-aaaa0000-0000-0000-0000-000000000000)\n"
        "  MIG 1g.5gb Device 0: (UUID: MIG-11110000-0000-0000-0000-000000000000)\n"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-11110000-0000-0000-0000-000000000000")
    visible = gpus.visible_devices()
    assert len(visible) == 1
    assert visible[0]["mig"] is True
    assert visible[0]["parent_uuid"] == UUIDS[0]
    assert visible[0]["physical_index"] == 0


def test_an_unresolvable_mig_token_is_kept_rather_than_dropped(smi, monkeypatch):
    """Dropping it would shift every later logical index and move work to another card."""
    smi["listing"] = ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",
                       "MIG-99990000-0000-0000-0000-000000000000,1")
    visible = gpus.visible_devices()
    assert [d["logical_index"] for d in visible] == [0, 1]
    assert visible[0]["unresolved"] is True
    assert visible[1]["physical_index"] == 1, "the integer token must still land on physical 1"


# -------------------------------------------------------------------------------------------
# Busy devices
# -------------------------------------------------------------------------------------------
def test_a_busy_gpu_is_excluded_by_uuid_not_by_index(smi):
    smi["busy"] = UUIDS[1] + "\n"
    found = gpus.discover_devices()
    assert found["n_usable"] == 2
    assert [d["uuid"] for d in found["usable"]] == [UUIDS[0], UUIDS[2]]
    assert [d["uuid"] for d in found["excluded_busy"]] == [UUIDS[1]]


def test_a_busy_parent_makes_its_mig_instance_busy(smi, monkeypatch):
    """Compute apps report the PARENT uuid, so a MIG child would otherwise look free."""
    smi["listing"] = (
        "GPU 0: NVIDIA RTX A5000 (UUID: GPU-aaaa0000-0000-0000-0000-000000000000)\n"
        "  MIG 1g.5gb Device 0: (UUID: MIG-11110000-0000-0000-0000-000000000000)\n"
    )
    smi["busy"] = UUIDS[0] + "\n"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-11110000-0000-0000-0000-000000000000")
    found = gpus.discover_devices()
    assert found["n_usable"] == 0, "a MIG instance on a busy parent is not free"


def test_exclude_busy_can_be_turned_off(smi):
    smi["busy"] = UUIDS[1] + "\n"
    assert gpus.discover_devices(exclude_busy=False)["n_usable"] == 3


# -------------------------------------------------------------------------------------------
# The min() rule
# -------------------------------------------------------------------------------------------
def test_more_gpus_than_replicas_uses_only_as_many_as_there_are_replicas(smi):
    record = gpus.select_devices(2)
    assert record["n_devices_selected"] == 2, "a third device cannot hold a third of two replicas"
    assert record["rule"] == "min(n_replicas, n_available)"
    assert record["shared_devices"] == []


def test_more_replicas_than_gpus_shares_devices_deterministically(smi):
    record = gpus.select_devices(7)
    assert record["n_devices_selected"] == 3
    # round-robin: replica i on device i % 3
    assert record["replica_devices"] == [0, 1, 2, 0, 1, 2, 0]
    assert record["replicas_per_device"] == {0: [0, 3, 6], 1: [1, 4], 2: [2, 5]}
    assert record["shared_devices"] == [0, 1, 2]


def test_the_rule_is_never_max(smi):
    """max() would request more devices than replicas, or more than the machine has."""
    for replicas in (1, 2, 3, 4, 10):
        record = gpus.select_devices(replicas)
        assert record["n_devices_selected"] == min(replicas, 3)


def test_no_usable_gpu_refuses_before_any_system_is_built(smi):
    smi["busy"] = "\n".join(UUIDS)
    with pytest.raises(gpus.NoUsableGpuError) as excinfo:
        gpus.select_devices(4)
    message = str(excinfo.value)
    assert "no usable device" in message
    assert "before any replica System is built" in message


def test_no_gpus_at_all_refuses(smi):
    smi["gpus"] = ""
    with pytest.raises(gpus.NoUsableGpuError):
        gpus.select_devices(2)


# -------------------------------------------------------------------------------------------
# Explicit selection
# -------------------------------------------------------------------------------------------
def test_an_explicit_selection_is_used_exactly_as_given(smi):
    """Even a busy device: the operator named it, and substituting another is a different run."""
    smi["busy"] = UUIDS[1] + "\n"
    record = gpus.select_devices(2, explicit=[1])
    assert record["selection_source"] == "explicit"
    assert record["devices"] == [1]
    assert record["rule"] == "operator-specified"
    assert record["replica_devices"] == [1, 1]


def test_an_explicit_hidden_device_is_refused(smi, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(gpus.NoUsableGpuError) as excinfo:
        gpus.select_devices(1, explicit=[2])
    assert "not visible" in str(excinfo.value)


def test_an_empty_explicit_list_is_refused(smi):
    with pytest.raises(ValueError):
        gpus.select_devices(1, explicit=[])


# -------------------------------------------------------------------------------------------
# What gets recorded
# -------------------------------------------------------------------------------------------
def test_the_record_carries_every_identifier_needed_to_name_the_card(smi):
    record = gpus.select_devices(2)
    for entry in record["device_identity"]:
        assert entry["logical_index"] is not None
        assert entry["physical_index"] is not None
        assert entry["uuid"].startswith("GPU-")
        assert entry["pci_bus_id"].startswith("00000000:")
        assert entry["name"]
    assert json.dumps(record)          # must be serialisable: it goes into gpu_selection.json


def test_the_description_shows_both_index_spaces_and_the_bus_id(smi):
    text = gpus.describe_selection(gpus.select_devices(2))
    assert "logical 0 -> physical 0" in text
    assert "00000000:1A:00.0" in text


def test_a_cpu_platform_selects_nothing_rather_than_failing(smi):
    record = gpus.select_devices(4, platform="CPU")
    assert record["devices"] is None
    assert "no device selection" in record["note"]
