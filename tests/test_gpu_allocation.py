"""Automatic REST2 device allocation.

REST2 holds one OpenMM `Context` per replica, so the useful number of devices is

    N_use = min(N_replica, N_available)

and never `max`. Both directions are silent when wrong: a six-replica ladder that quietly uses one
GPU looks exactly like one using six, only slower -- which is how six devices came to deliver one
device's throughput in an earlier campaign.

Two index spaces are kept apart throughout. `nvidia-smi` prints PHYSICAL indices; OpenMM's
`DeviceIndex` takes LOGICAL ones, counting only what `CUDA_VISIBLE_DEVICES` exposes. Under
`CUDA_VISIBLE_DEVICES=3,5,7` the machine's GPU 5 is logical 1, and passing 5 would select physical
GPU 7 -- work on a card nobody asked for.
"""

from __future__ import annotations

import pytest

from md_templates.openmm import gpus
from md_templates.openmm.gpus import NoUsableGpuError, select_devices
from md_templates.openmm.tau import map_replicas_to_devices


def _fake(n, busy=()):
    """`n` visible devices, `busy` given as physical indices."""
    return [{"physical_index": i, "logical_index": i, "uuid": f"GPU-{i:04d}",
             "name": "FakeGPU", "memory_total_mib": "10240"} for i in range(n)], \
           {f"GPU-{i:04d}" for i in busy}


@pytest.fixture
def nine_free(monkeypatch):
    visible, busy = _fake(9)
    monkeypatch.setattr(gpus, "visible_devices", lambda: visible)
    monkeypatch.setattr(gpus, "busy_physical_devices", lambda: busy)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return visible


# -------------------------------------------------------------------------------------------
# min(), never max()
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("replicas,expected_devices", [(6, 6), (4, 4), (9, 9), (1, 1)])
def test_fewer_replicas_than_gpus_uses_one_device_per_replica(nine_free, replicas,
                                                              expected_devices):
    record = select_devices(replicas)
    assert record["n_devices_selected"] == expected_devices
    assert record["rule"] == "min(n_replicas, n_available)"
    assert len(set(record["replica_devices"])) == expected_devices


def test_more_replicas_than_gpus_shares_devices_rather_than_refusing(nine_free):
    """Ten replicas on nine GPUs: nine devices, one carrying two replicas."""
    record = select_devices(10)
    assert record["n_devices_selected"] == 9
    assert len(record["replica_devices"]) == 10
    assert len(record["shared_devices"]) == 1
    shared = record["shared_devices"][0]
    assert len(record["replicas_per_device"][shared]) == 2


def test_the_rule_is_min_not_max(nine_free):
    """`max` would claim devices that cannot be used; the surplus has no Context to run."""
    for replicas in (1, 3, 6, 9, 12, 30):
        record = select_devices(replicas)
        assert record["n_devices_selected"] == min(replicas, 9)
        assert record["n_devices_selected"] <= replicas
        assert record["n_devices_selected"] <= 9


# -------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES
# -------------------------------------------------------------------------------------------
def test_a_hidden_device_is_never_selected(monkeypatch):
    """Selection may only draw from what the variable exposes."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")
    monkeypatch.setattr(gpus, "_all_physical_devices",
                        lambda: [{"physical_index": i, "uuid": f"GPU-{i:04d}",
                                  "name": "FakeGPU", "memory_total_mib": "10240"}
                                 for i in range(9)])
    monkeypatch.setattr(gpus, "busy_physical_devices", set)
    visible = gpus.visible_devices()
    assert [d["physical_index"] for d in visible] == [3, 5, 7]
    assert [d["logical_index"] for d in visible] == [0, 1, 2]

    record = select_devices(3)
    assert record["devices"] == [0, 1, 2], "OpenMM takes LOGICAL indices"
    assert record["physical_of_logical"] == {0: 3, 1: 5, 2: 7}


def test_an_empty_visible_devices_variable_hides_everything(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(gpus, "_all_physical_devices", lambda: [])
    assert gpus.visible_devices() == []
    with pytest.raises(NoUsableGpuError):
        select_devices(4)


def test_an_unset_variable_means_every_device_is_visible(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpus, "_all_physical_devices",
                        lambda: [{"physical_index": i, "uuid": f"GPU-{i:04d}",
                                  "name": "F", "memory_total_mib": "1"} for i in range(4)])
    visible = gpus.visible_devices()
    assert [(d["logical_index"], d["physical_index"]) for d in visible] == [(0, 0), (1, 1),
                                                                           (2, 2), (3, 3)]


def test_a_visible_devices_entry_the_driver_does_not_report_is_dropped(monkeypatch):
    """CUDA stops at the first invalid entry; inventing a device would place work arbitrarily."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,2")
    monkeypatch.setattr(gpus, "_all_physical_devices",
                        lambda: [{"physical_index": i, "uuid": f"GPU-{i:04d}",
                                  "name": "F", "memory_total_mib": "1"} for i in range(4)])
    assert [d["physical_index"] for d in gpus.visible_devices()] == [0, 2]


# -------------------------------------------------------------------------------------------
# Busy devices
# -------------------------------------------------------------------------------------------
def test_busy_devices_are_excluded_by_default(monkeypatch):
    visible, busy = _fake(9, busy=(0, 1, 2))
    monkeypatch.setattr(gpus, "visible_devices", lambda: visible)
    monkeypatch.setattr(gpus, "busy_physical_devices", lambda: busy)
    record = select_devices(6)
    assert record["n_devices_selected"] == 6
    assert set(record["devices"]).isdisjoint({0, 1, 2})
    assert len(record["discovery"]["excluded_busy"]) == 3


def test_busy_exclusion_can_be_turned_off(monkeypatch):
    visible, busy = _fake(4, busy=(0, 1))
    monkeypatch.setattr(gpus, "visible_devices", lambda: visible)
    monkeypatch.setattr(gpus, "busy_physical_devices", lambda: busy)
    assert select_devices(4, exclude_busy=False)["n_devices_selected"] == 4


def test_all_devices_busy_refuses_before_any_system_is_built(monkeypatch):
    """Discovering this after building six Systems wastes the expensive part of setup."""
    visible, busy = _fake(3, busy=(0, 1, 2))
    monkeypatch.setattr(gpus, "visible_devices", lambda: visible)
    monkeypatch.setattr(gpus, "busy_physical_devices", lambda: busy)
    with pytest.raises(NoUsableGpuError) as excinfo:
        select_devices(2)
    assert "no usable device" in str(excinfo.value)


# -------------------------------------------------------------------------------------------
# Explicit overrides
# -------------------------------------------------------------------------------------------
def test_explicit_devices_are_used_exactly_even_when_busy(monkeypatch):
    """A device the operator named and one this module chose are different claims."""
    visible, busy = _fake(9, busy=(4, 5))
    monkeypatch.setattr(gpus, "visible_devices", lambda: visible)
    monkeypatch.setattr(gpus, "busy_physical_devices", lambda: busy)
    record = select_devices(2, explicit=[4, 5])
    assert record["devices"] == [4, 5]
    assert record["selection_source"] == "explicit"
    assert record["rule"] == "operator-specified"


def test_an_explicit_hidden_device_is_refused_not_substituted(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")
    monkeypatch.setattr(gpus, "_all_physical_devices",
                        lambda: [{"physical_index": i, "uuid": f"GPU-{i:04d}",
                                  "name": "F", "memory_total_mib": "1"} for i in range(9)])
    monkeypatch.setattr(gpus, "busy_physical_devices", set)
    with pytest.raises(NoUsableGpuError) as excinfo:
        select_devices(1, explicit=[8])
    assert "not visible" in str(excinfo.value)
    assert "somewhere nobody asked for" in str(excinfo.value)


def test_an_empty_explicit_list_is_refused(nine_free):
    with pytest.raises(ValueError):
        select_devices(4, explicit=[])


# -------------------------------------------------------------------------------------------
# Determinism and mapping
# -------------------------------------------------------------------------------------------
def test_the_mapping_is_deterministic_round_robin(nine_free):
    first = select_devices(10)["replica_devices"]
    second = select_devices(10)["replica_devices"]
    assert first == second
    assert first == map_replicas_to_devices(10, sorted(set(first), key=first.index))


def test_the_record_carries_everything_needed_to_explain_the_choice(nine_free):
    record = select_devices(6)
    for key in ("selection_source", "rule", "devices", "device_details",
                "physical_of_logical", "uuid_of_logical", "replica_devices",
                "replicas_per_device", "shared_devices", "discovery"):
        assert key in record, key
    assert record["discovery"]["n_visible"] == 9


def test_a_non_accelerator_platform_selects_nothing(nine_free):
    record = select_devices(6, platform="CPU")
    assert record["devices"] is None
    assert record["replica_devices"] is None


# -------------------------------------------------------------------------------------------
# The propagation barrier, and one worker per device
# -------------------------------------------------------------------------------------------
def test_replicas_sharing_a_device_are_never_stepped_concurrently():
    """Two workers on one card interleave kernels and add context switching; they do not halve it."""
    import threading

    from md_templates.openmm.rest2 import propagate_replicas, replica_propagation_pool

    live_per_device: dict[int, int] = {}
    max_live: dict[int, int] = {}
    lock = threading.Lock()

    class Sim:
        def __init__(self, device):
            self.device = device

        def step(self, steps):
            with lock:
                live_per_device[self.device] = live_per_device.get(self.device, 0) + 1
                max_live[self.device] = max(max_live.get(self.device, 0),
                                            live_per_device[self.device])
            import time
            time.sleep(0.01)
            with lock:
                live_per_device[self.device] -= 1

    device_of = [0, 1, 0, 1, 2]                      # replicas 0 and 2 share device 0
    sims = [Sim(d) for d in device_of]
    with replica_propagation_pool(len(sims)) as pool:
        propagate_replicas(sims, 10, executor=pool, device_of=device_of)

    assert max(max_live.values()) == 1, (
        f"two replicas ran on one device at once: {max_live}")


def test_the_barrier_holds_when_devices_are_shared():
    """No exchange may be evaluated against a replica still integrating."""
    import time

    from md_templates.openmm.rest2 import propagate_replicas, replica_propagation_pool

    finished = []

    class Sim:
        def __init__(self, delay):
            self.delay = delay

        def step(self, steps):
            time.sleep(self.delay)
            finished.append(self.delay)

    sims = [Sim(0.05), Sim(0.01), Sim(0.03), Sim(0.01)]
    with replica_propagation_pool(len(sims)) as pool:
        propagate_replicas(sims, 10, executor=pool, device_of=[0, 0, 1, 1])
    assert len(finished) == 4


def test_different_devices_do_run_concurrently():
    """The point of the pool: the speedup comes from across-device overlap."""
    import time

    from md_templates.openmm.rest2 import propagate_replicas, replica_propagation_pool

    class Sim:
        def step(self, steps):
            time.sleep(0.20)

    sims = [Sim() for _ in range(4)]
    start = time.time()
    with replica_propagation_pool(4) as pool:
        propagate_replicas(sims, 10, executor=pool, device_of=[0, 1, 2, 3])
    elapsed = time.time() - start
    assert elapsed < 0.60, f"four devices took {elapsed:.2f}s; they did not overlap"
