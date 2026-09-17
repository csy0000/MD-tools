"""Placement: the CPU rule, balanced devices, and MPS told apart from its appearance.

Everything here is the PURE half of `md_tools.openmm.placement` -- facts in, plan out -- so none of
it needs a GPU, and none of it is CUDA evidence. What it proves is the arithmetic and the
refusals: a CPU count that does not divide, a shared device without verified MPS, a launcher that
disagrees with the host names about local ranks.
"""
from __future__ import annotations

import pytest

from md_tools.openmm import placement as placing
from md_tools.openmm.placement import (MpsStatus, PlacementError, WorkerFacts,
                                       balanced_assignment, check_cpu_multiple, plan_cpus,
                                       plan_launch, read_mps_status, refuse_unverified_sharing,
                                       verify_mps_client)

ABSENT = MpsStatus(requested=False, pipe_directory="/tmp/nvidia-mps", daemon="not-running")


def _facts(workers, *, cpus=48, devices=1, host="node", quota=None, local=None, mps=ABSENT,
           visible=None, first_rank=0):
    return [WorkerFacts(rank=first_rank + i, hostname=host, cpus=tuple(range(cpus)),
                        cpu_quota=quota, visible_devices=devices,
                        cuda_visible_devices=None if visible is None else visible[i],
                        cuda_device_order="PCI_BUS_ID",
                        launcher_local_rank=i if local is None else local[i], mps=mps)
            for i in range(workers)]


# --- CPUs -----------------------------------------------------------------------------------

@pytest.mark.parametrize("cpus", [4, 8, 12, 48])
def test_a_cpu_count_that_is_a_multiple_of_the_workers_is_accepted(cpus):
    assert check_cpu_multiple(cpus, 4) == cpus // 4


def test_five_cpus_for_four_workers_is_refused_with_the_arithmetic_that_fixes_it():
    with pytest.raises(PlacementError) as refused:
        check_cpu_multiple(5, 4, where="REST2")
    message = str(refused.value)
    assert "5 = 4 x 1 + 1" in message, message
    assert "4 CPUs (1 per worker)" in message and "8 CPUs (2 per worker)" in message, message


def test_fewer_cpus_than_workers_is_refused():
    with pytest.raises(PlacementError, match="not an integer multiple"):
        check_cpu_multiple(3, 4)


def test_blocks_are_equal_disjoint_and_cover_the_usable_cpus():
    plan = plan_cpus(_facts(4, cpus=12))
    blocks = [plan[rank]["cpus"] for rank in range(4)]
    assert all(len(block) == 3 for block in blocks), blocks
    assert sorted(cpu for block in blocks for cpu in block) == list(range(12))


def test_a_cgroup_quota_narrows_the_count_that_must_divide():
    with pytest.raises(PlacementError, match="10 CPU"):
        plan_cpus(_facts(4, cpus=48, quota=10))
    assert plan_cpus(_facts(4, cpus=48, quota=8))[0]["cpus_per_worker"] == 2


def test_blocks_keep_hardware_threads_of_one_core_together(tmp_path):
    # Two sockets, two cores each, two threads per core, numbered the way Linux numbers them:
    # the first thread of every core, then the second.
    layout = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (0, 0), 5: (0, 1), 6: (1, 0),
              7: (1, 1)}
    for cpu, (package, core) in layout.items():
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")
    ordered = placing.ordered_cpus(range(8), sysfs=tmp_path)
    assert placing.cpu_blocks(ordered, 2, 4) == [(0, 4, 1, 5), (2, 6, 3, 7)]


def test_the_cgroup_quota_is_the_tightest_on_the_path(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice/job.scope\n")
    root = tmp_path / "fs"
    (root / "user.slice" / "job.scope").mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n")
    (root / "user.slice" / "cpu.max").write_text("250000 100000\n")
    (root / "user.slice" / "job.scope" / "cpu.max").write_text("max 100000\n")
    assert placing._cgroup_cpu_quota(proc=proc, root=root) == 2


def test_a_launcher_local_rank_that_contradicts_the_host_names_is_refused():
    with pytest.raises(PlacementError, match="local rank"):
        plan_cpus(_facts(2, local=[1, 0]))


def test_each_host_divides_its_own_cpus_among_its_own_workers():
    facts = _facts(2, cpus=8, host="a") + _facts(3, cpus=9, host="b", first_rank=2)
    plan = plan_cpus(facts)
    assert [plan[r]["local_rank"] for r in range(5)] == [0, 1, 0, 1, 2]
    assert plan[1]["cpus_per_worker"] == 4 and plan[4]["cpus_per_worker"] == 3
    with pytest.raises(PlacementError, match="on b"):
        plan_cpus(_facts(2, cpus=8, host="a") + _facts(3, cpus=8, host="b", first_rank=2))


# --- devices --------------------------------------------------------------------------------

def test_four_workers_on_one_device_all_share_it():
    assert balanced_assignment([100.0], 4) == [0, 0, 0, 0]


def test_twelve_workers_on_four_equal_devices_take_three_each():
    chosen = balanced_assignment([100.0] * 4, 12)
    assert [chosen.count(d) for d in range(4)] == [3, 3, 3, 3]


def test_an_uneven_split_puts_the_extra_worker_on_the_fastest_device():
    assert balanced_assignment([90.0, 100.0, 95.0], 4) == [0, 1, 1, 2]


def test_a_slow_device_carries_the_least_load_or_none():
    # 100/3 = 33 per worker still beats one worker alone on a device measured at 30.
    assert balanced_assignment([100.0, 30.0], 3) == [0, 0, 0]
    assert balanced_assignment([100.0, 60.0], 3) == [0, 0, 1]


def test_fewer_workers_than_devices_take_the_fastest_distinct_devices():
    speeds = [80.0, 120.0, 100.0, 100.0, 100.0]
    assert balanced_assignment(speeds, 3) == [1, 2, 3]


def test_a_device_that_measured_nothing_is_refused_rather_than_placed_on():
    with pytest.raises(PlacementError, match="no throughput"):
        balanced_assignment([100.0, 0.0], 2)


def test_the_plan_places_by_the_measurement_and_says_when_devices_are_shared():
    plan = plan_launch(_facts(4, devices=3), platform="CUDA", device_policy="local_rank",
                       throughput={"node": [100.0, 100.0, 100.0]})
    assert [plan.for_rank(r)["device"] for r in range(4)] == [0, 0, 1, 2]
    assert [plan.for_rank(r)["co_tenants"] for r in range(4)] == [2, 2, 1, 1]
    assert plan.shared_devices and plan.placement == "measured throughput (balanced)"

    alone = plan_launch(_facts(3, devices=3), platform="CUDA", device_policy="local_rank",
                        throughput={"node": [100.0, 100.0, 100.0]})
    assert not alone.shared_devices


def test_a_plural_cuda_plan_without_a_measurement_is_refused():
    with pytest.raises(PlacementError, match="no throughput measurement"):
        plan_launch(_facts(2, devices=2), platform="CUDA", device_policy="local_rank")


def test_a_single_worker_keeps_the_first_visible_device_and_needs_no_measurement():
    facts = _facts(1, devices=9)
    assert placing.needs_measurement(facts, platform="CUDA", device_policy="local_rank",
                                     explicit_device=None) == {}
    assert plan_launch(facts, platform="CUDA", device_policy="local_rank").for_rank(0)[
        "device"] == 0
    # A single process that never counted its devices sets no DeviceIndex, as before.
    assert plan_launch(_facts(1, devices=0), platform="CUDA",
                       device_policy="local_rank").for_rank(0)["device"] is None


def test_the_cpu_platform_places_no_device_and_shares_none():
    plan = plan_launch(_facts(4, devices=0), platform="CPU", device_policy="local_rank")
    assert {plan.for_rank(r)["device"] for r in range(4)} == {None}
    assert not plan.shared_devices


def test_one_device_named_for_several_workers_is_sharing():
    plan = plan_launch(_facts(2, devices=0), platform="CUDA", device_policy="local_rank",
                       explicit_device=3)
    assert plan.shared_devices and plan.for_rank(1)["device"] == 3


def test_the_openmm_policy_is_sharing_unless_every_worker_was_given_its_own_device():
    shared = plan_launch(_facts(2, devices=0), platform="CUDA", device_policy="openmm")
    assert shared.shared_devices
    partitioned = plan_launch(_facts(2, devices=0, visible=["4", "5"]), platform="CUDA",
                              device_policy="openmm")
    assert not partitioned.shared_devices
    same = plan_launch(_facts(2, devices=0, visible=["4", "4"]), platform="CUDA",
                       device_policy="openmm")
    assert same.shared_devices


# --- MPS ------------------------------------------------------------------------------------

def _pipe(tmp_path, *, control=True, pid=None):
    if control:
        (tmp_path / "control").write_text("")
        (tmp_path / "control_lock").write_text("")
    if pid is not None:
        (tmp_path / "nvidia-cuda-mps-control.pid").write_text(f"{pid}\n")
    return {"CUDA_MPS_PIPE_DIRECTORY": str(tmp_path)}


def test_no_daemon_and_no_request_is_absent(tmp_path):
    status = read_mps_status({}, process_running=lambda name: False)
    assert status.status == "absent" and not status.requested


def test_a_request_without_a_daemon_is_not_detection(tmp_path):
    status = read_mps_status(_pipe(tmp_path, control=False),
                             process_running=lambda name: False)
    assert status.requested and status.status == "requested-not-detected"


def test_a_daemon_on_another_pipe_directory_is_not_detected_for_this_process(tmp_path):
    status = read_mps_status(_pipe(tmp_path, control=False), process_running=lambda name: True)
    assert status.daemon == "not-running", status
    assert "different pipe directory" in status.detail


def test_a_pid_file_naming_a_live_daemon_is_detection(tmp_path):
    status = read_mps_status(_pipe(tmp_path, pid=4242),
                             process_running=lambda name: True,
                             pid_is=lambda name, pid: pid == 4242)
    assert status.status == "detected-unverified" and "4242" in status.detail


def test_a_pid_file_naming_a_dead_daemon_is_a_stopped_one(tmp_path):
    """The sockets outlive the daemon. `control` existing is not a daemon existing."""
    status = read_mps_status(_pipe(tmp_path, pid=4242),
                             process_running=lambda name: True,
                             pid_is=lambda name, pid: False)
    assert status.daemon == "not-running", status
    assert "has stopped" in status.detail


def test_sockets_with_no_pid_file_are_unknown_rather_than_running(tmp_path):
    """A leftover directory looked exactly like a live daemon, and another daemon elsewhere on the
    host made the process check agree. Unknown permits nothing, so saying so costs nothing."""
    status = read_mps_status(_pipe(tmp_path), process_running=lambda name: True)
    assert status.status == "unknown", status
    assert "leaves its sockets behind" in status.detail

    alone = read_mps_status(_pipe(tmp_path), process_running=lambda name: False)
    assert alone.daemon == "not-running" and "leftover sockets" in alone.detail


def test_an_unreadable_process_table_is_unknown_not_absent():
    assert read_mps_status({}, process_running=lambda name: None).status == "unknown"


def test_the_process_check_truncates_the_name_the_way_the_kernel_does(tmp_path):
    """`pgrep nvidia-cuda-mps-control` matches nothing: /proc/<pid>/comm holds 15 characters."""
    from md_tools.openmm import placement as module

    (tmp_path / "77" ).mkdir()
    (tmp_path / "77" / "comm").write_text("nvidia-cuda-mps\n")
    assert module._pid_is("nvidia-cuda-mps-control", 77, proc=tmp_path) is True
    assert module._pid_is("nvidia-cuda-mps-control", 78, proc=tmp_path) is False


def _smi(*processes):
    rows = "".join(f"<process_info><pid>{pid}</pid><type>{kind}</type></process_info>"
                   for pid, kind in processes)
    return f"<nvidia_smi_log><gpu><processes>{rows}</processes></gpu></nvidia_smi_log>"


def test_the_driver_listing_this_process_as_m_plus_c_verifies_mps():
    assert verify_mps_client(42, xml=_smi((7, "C"), (42, "M+C")))[0] is True


def test_the_driver_listing_this_process_as_c_is_not_a_client():
    verdict, why = verify_mps_client(42, xml=_smi((42, "C")))
    assert verdict is False and "ordinary CUDA process" in why


def test_a_process_the_driver_does_not_list_cannot_be_verified():
    assert verify_mps_client(42, xml=_smi((7, "M+C")))[0] is None
    assert verify_mps_client(42, xml=None)[0] is None
    assert verify_mps_client(42, xml="<not xml")[0] is None


@pytest.mark.parametrize("status", [
    ABSENT,
    MpsStatus(requested=True, pipe_directory="/x", daemon="not-running"),
    MpsStatus(requested=True, pipe_directory="/x", daemon="running"),
    MpsStatus(requested=True, pipe_directory="/x", daemon="running", verified=False),
    MpsStatus(requested=False, pipe_directory="/x", daemon="unknown"),
])
def test_a_shared_device_is_refused_unless_mps_is_verified(status):
    plan = plan_launch(_facts(4), platform="CUDA", device_policy="local_rank",
                       throughput={"node": [100.0]})
    with pytest.raises(PlacementError, match="device 0 hosts 4 worker") as refused:
        refuse_unverified_sharing(plan, status, rank=2, where="REST2")
    assert status.status in str(refused.value)
    assert "nvidia-cuda-mps-control -d" in str(refused.value)


def test_a_shared_device_with_verified_mps_is_accepted_and_an_unshared_one_needs_none():
    verified = MpsStatus(requested=True, pipe_directory="/x", daemon="running", verified=True)
    shared = plan_launch(_facts(4), platform="CUDA", device_policy="local_rank",
                         throughput={"node": [100.0]})
    refuse_unverified_sharing(shared, verified, rank=0)
    alone = plan_launch(_facts(2, devices=2), platform="CUDA", device_policy="local_rank",
                        throughput={"node": [100.0, 100.0]})
    refuse_unverified_sharing(alone, ABSENT, rank=0)


def test_the_record_names_every_worker_and_the_mps_status():
    from dataclasses import replace

    plan = plan_launch(_facts(4, cpus=8), platform="CUDA", device_policy="local_rank",
                       throughput={"node": [100.0]})
    record = replace(plan, this_rank=3, mps=ABSENT).record()
    assert record["this_worker"]["cpus"] == [6, 7]
    assert set(record["worker_map"]) == {"0", "1", "2", "3"}
    assert record["mps"]["status"] == "absent"


def test_placement_is_the_only_device_chooser():
    """Round-robin is retired, and nothing grows a second placement policy beside this one."""
    from .conftest import REPO_ROOT

    root = REPO_ROOT / "src" / "md_tools"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def select_device_for_rank(" not in text, path
        assert "def device_index_for(" not in text, path
        assert "% len(devices)" not in text, path


def test_the_mps_refusal_names_a_section_that_exists():
    """The refusal sends a person to a heading. A stale name is a pointer nobody can follow."""
    from .conftest import REPO_ROOT

    plan = plan_launch(_facts(2), platform="CUDA", device_policy="local_rank",
                       throughput={"node": [100.0]})
    with pytest.raises(PlacementError) as refused:
        refuse_unverified_sharing(plan, ABSENT, rank=0)
    quoted = str(refused.value).split('docs/md-run.md, "')[1].split('"')[0]
    headings = (REPO_ROOT / "docs" / "md-run.md").read_text(encoding="utf-8")
    assert f"## {quoted}" in headings, quoted
