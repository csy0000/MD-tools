"""Where the workers of one launch run: which CPUs, which GPU, and whether that GPU is shared.

ONE authority for placement, as `platform_policy` is for the platform. Stages, REST2 ladders and
AIS launches all reach it through `md_tools.run.preflight`, before any output exists.

WHAT IT REPLACED

Ranks used to be dealt over the visible devices in turn (`rank % n_devices`). Two ranks dealt
onto one GPU without MPS are TIME-SLICED: each waits for the other's kernels. A synchronous ladder
runs at the pace of its slowest rank, so every rank then ran at the pace of the shared GPU, and
nothing in the record said so. Round-robin also ignored that devices differ -- an RTX A5000 and
an RTX 3080 are not one speed -- and it counted the global rank rather than the rank on this node.

THE RULES, each a refusal before output rather than a warning

    CPUs      The CPUs this launch may use on a node are an integer multiple of the workers on
              that node, and each worker is bound to its own equal block. The count is what the
              kernel allows -- the scheduler affinity, narrowed by a cgroup CPU quota -- never a
              number a configuration claims.
    sharing   A GPU hosting more than one worker needs NVIDIA MPS, VERIFIED for this process.
              Requested is not detected, and detected is not verified: a client that cannot reach
              the control pipe runs without MPS and says nothing.
    balance   Workers go to devices by MEASURED throughput, so the slowest device carries the
              least load. The measurement and the resulting map are recorded.

Placement never changes the protocol. A ladder's state index is the state's, an AIS path id is
the path's, and none of this enters a scientific fingerprint.

The functions below that decide are PURE -- facts in, plan out -- so the arithmetic and every
refusal are tested without a GPU. The functions that gather facts are thin and named `read_*`.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "PlacementError",
    "NodeFacts",
    "WorkerFacts",
    "MpsStatus",
    "LaunchPlan",
    "read_worker_facts",
    "read_mps_status",
    "verify_mps_client",
    "ordered_cpus",
    "cpu_blocks",
    "check_cpu_multiple",
    "balanced_assignment",
    "plan_cpus",
    "plan_launch",
    "needs_measurement",
    "refuse_unverified_sharing",
    "measure_device_throughput",
    "bind_worker",
    "MPS_CONTROL_PROCESS",
    "FORCE_MPS_VERDICT",
    "nvidia_smi_xml",
]


class PlacementError(ValueError):
    """A launch that cannot be placed under the rules. Re-raised by preflight as a refusal."""


#: The control daemon's process name. Its presence is a DETECTION, not a verification.
MPS_CONTROL_PROCESS = "nvidia-cuda-mps-control"
#: Where a client looks for the daemon's pipes when `CUDA_MPS_PIPE_DIRECTORY` is unset.
MPS_DEFAULT_PIPE_DIRECTORY = "/tmp/nvidia-mps"
#: Environment an MPS client reads. Recorded as found; never set by this package.
MPS_ENVIRONMENT = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                   "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")
#: Launcher variables that name the rank on this node. Compared with the rank derived from host
#: names, never trusted instead of it.
LOCAL_RANK_ENVIRONMENT = ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
                          "SLURM_LOCALID")


# ---------------------------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MpsStatus:
    """Three separate facts about MPS, because each has been mistaken for the next.

    requested  `CUDA_MPS_PIPE_DIRECTORY` is set: somebody pointed this process at a daemon.
    daemon     `running`, `not-running` or `unknown`: a control process exists and the pipe
               directory holds its `control` pipe. Seen from outside; says nothing about whether
               THIS process is a client of it.
    verified   True only when the driver lists this process as an MPS client (`M+C`) while it
               holds a Context; False when it lists it as an ordinary one (`C`); None when not
               asked or the driver would not say.
    """

    requested: bool
    pipe_directory: str
    daemon: str
    verified: bool | None = None
    environment: dict[str, str | None] = field(default_factory=dict)
    detail: str = ""
    #: True when the verdict came from `FORCE_MPS_VERDICT` rather than from the driver. A record
    #: whose `status` reads `verified` while no daemon was running is contradictory on its face,
    #: and `detail` alone is too easy to skim past: this is the field a reader can filter on.
    forced: bool = False

    @property
    def status(self) -> str:
        if self.verified is True:
            return "verified"
        if self.verified is False:
            return "not-a-client"
        if self.daemon == "running":
            return "detected-unverified"
        # UNKNOWN OUTRANKS REQUESTED. A directory that cannot be attributed to a live daemon is
        # not the same fact as a daemon that is not there, and reporting the request instead would
        # let `requested-not-detected` stand for "we could not tell".
        if self.daemon == "unknown":
            return "unknown"
        return "requested-not-detected" if self.requested else "absent"

    def with_verification(self, verified: bool | None, detail: str, *,
                          forced: bool = False) -> "MpsStatus":
        return MpsStatus(requested=self.requested, pipe_directory=self.pipe_directory,
                         daemon=self.daemon, verified=verified, environment=self.environment,
                         detail=detail, forced=forced)

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status}

    @property
    def verdict_source(self) -> str:
        return f"{FORCE_MPS_VERDICT} (test seam)" if self.forced else "the CUDA driver"


@dataclass(frozen=True)
class WorkerFacts:
    """What one rank can see of its own machine. Gathered from every rank, then planned from."""

    rank: int
    hostname: str
    #: Logical CPUs the scheduler lets this process run on, ordered so a block keeps a core's
    #: hardware threads and a socket's cores together.
    cpus: tuple[int, ...]
    #: A cgroup CPU quota in whole CPUs, when one narrows `cpus`; None when there is none.
    cpu_quota: int | None
    #: How many CUDA devices this rank can address, as OpenMM ordinals 0..n-1.
    visible_devices: int
    cuda_visible_devices: str | None
    cuda_device_order: str | None
    launcher_local_rank: int | None
    mps: MpsStatus


@dataclass(frozen=True)
class NodeFacts:
    hostname: str
    ranks: tuple[int, ...]
    cpus: tuple[int, ...]
    usable_cpus: int
    visible_devices: int


def _cpu_sort_key(cpu: int, sysfs: Path) -> tuple[int, int, int]:
    """(socket, core, cpu): blocks then hold whole cores and stay on one socket where they can."""
    base = sysfs / f"cpu{cpu}" / "topology"
    try:
        package = int((base / "physical_package_id").read_text().strip())
        core = int((base / "core_id").read_text().strip())
    except (OSError, ValueError):
        return (0, cpu, cpu)
    return (package, core, cpu)


def ordered_cpus(cpus: Iterable[int], *, sysfs: Path = Path("/sys/devices/system/cpu")
                 ) -> tuple[int, ...]:
    return tuple(sorted({int(c) for c in cpus}, key=lambda c: _cpu_sort_key(c, sysfs)))


def _cgroup_cpu_quota(*, proc: Path = Path("/proc/self/cgroup"),
                      root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The tightest cgroup v2 `cpu.max` quota on this process's path, in whole CPUs.

    Walked to the root because a limit on a parent slice binds every scope beneath it. Floored:
    a quota of 2.5 CPUs cannot give three workers a CPU each.
    """
    try:
        line = next(l for l in proc.read_text().splitlines() if l.startswith("0::"))
    except (OSError, StopIteration):
        return None
    tightest = None
    path = root / line[3:].lstrip("/")
    while True:
        try:
            quota, period = (path / "cpu.max").read_text().split()
            if quota != "max":
                cpus = int(int(quota) // int(period))
                tightest = cpus if tightest is None else min(tightest, cpus)
        except (OSError, ValueError):
            pass
        if path == root or root not in path.parents:
            break
        path = path.parent
    return tightest


def _process_running(name: str, *, proc: Path = Path("/proc")) -> bool | None:
    """Whether any process on this host has `name` as its command. None if /proc will not say."""
    try:
        entries = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return None
    for entry in entries:
        try:
            if (entry / "comm").read_text().strip() == name[:15]:
                return True
        except OSError:
            continue
    return False


def _pid_is(name: str, pid: int, *, proc: Path = Path("/proc")) -> bool | None:
    """Is `pid` alive and running `name`? None when /proc will not say.

    The comm the kernel exposes is truncated to 15 characters, so the NAME is truncated the same
    way before comparing -- which is also why `pgrep nvidia-cuda-mps-control` matches nothing
    whatever the truth, and why this does not use it. One consequence stated rather than papered
    over: the control daemon and its server are indistinguishable by comm, so the pipe directory
    is what makes the status specific.
    """
    try:
        return (proc / str(int(pid)) / "comm").read_text().strip() == name[:15]
    except FileNotFoundError:
        return False
    except OSError:
        return None


def read_mps_status(environment: Mapping[str, str] | None = None, *,
                    process_running: Callable[[str], bool | None] = _process_running,
                    pid_is: Callable[[str, int], bool | None] = _pid_is) -> MpsStatus:
    """Requested and detected. Verification needs a live Context: `verify_mps_client`.

    Never starts, stops or talks to the daemon. Asking `nvidia-cuda-mps-control` a question is a
    conversation with a shared service somebody else may own; the pipe directory and the process
    table answer "is one there" without it.

    THE PID FILE IS THE EVIDENCE, not the socket. A daemon that has been stopped leaves `control`,
    `control_lock` and `control_privileged` behind and takes its `.pid` file with it, so "the
    control pipe exists" reported a daemon that had been shut down half an hour earlier -- and a
    second daemon running elsewhere on the host made the process check agree with it. A directory
    with sockets and no pid file is therefore UNKNOWN rather than running: it may be a live daemon
    that has not written one, or a leftover, and nothing here can tell which. Unknown permits
    nothing -- only `verified` does -- so the honest answer costs nothing.
    """
    environment = os.environ if environment is None else environment
    pipe = environment.get("CUDA_MPS_PIPE_DIRECTORY") or MPS_DEFAULT_PIPE_DIRECTORY
    requested = bool(environment.get("CUDA_MPS_PIPE_DIRECTORY"))
    directory = Path(pipe)
    control = directory / "control"
    pid_file = directory / f"{MPS_CONTROL_PROCESS}.pid"

    def status(daemon: str, detail: str) -> MpsStatus:
        return MpsStatus(requested=requested, pipe_directory=pipe, daemon=daemon,
                         environment={name: environment.get(name) for name in MPS_ENVIRONMENT},
                         detail=detail)

    try:
        recorded = int(pid_file.read_text().split()[0]) if pid_file.is_file() else None
    except (OSError, ValueError, IndexError):
        recorded = None

    if recorded is not None:
        alive = pid_is(MPS_CONTROL_PROCESS, recorded)
        if alive is True:
            return status("running", f"{pipe}/{pid_file.name} names pid {recorded}, which is "
                                     f"running {MPS_CONTROL_PROCESS}")
        if alive is False:
            return status("not-running", f"{pipe}/{pid_file.name} names pid {recorded}, which is "
                                         f"not running: the daemon it belonged to has stopped")
        return status("unknown", f"pid {recorded} from {pipe}/{pid_file.name} could not be "
                                 f"checked in /proc")

    running = process_running(MPS_CONTROL_PROCESS)
    if running is None:
        return status("unknown", "the process table could not be read")
    if not control.exists():
        return status("not-running",
                      f"no control pipe in {pipe}" + (
                          f", though a {MPS_CONTROL_PROCESS} process is running on this host: it "
                          f"serves a different pipe directory, and a client reaches a daemon only "
                          f"through CUDA_MPS_PIPE_DIRECTORY" if running else ""))
    if running:
        return status("unknown",
                      f"{pipe} holds a control pipe and no {pid_file.name}, and a "
                      f"{MPS_CONTROL_PROCESS} process is running on this host. A stopped daemon "
                      f"leaves its sockets behind, so this cannot be attributed to a live one.")
    return status("not-running", f"{pipe} holds a control pipe but no {MPS_CONTROL_PROCESS} "
                                 f"process is running: leftover sockets from a stopped daemon")


#: Test seam: `verified` or `not-a-client` stands in for the driver's verdict in a child process,
#: so both paths run through the real command on a machine whose daemon state cannot be arranged.
#: Never set in normal use, and never used to make a shared-GPU run that the rule refuses LOOK
#: acceptable: a timing measured that way is a timing of a configuration the product does not
#: allow.
FORCE_MPS_VERDICT = "MD_TOOLS_FORCE_MPS_VERDICT"


def nvidia_smi_xml(run=subprocess.run) -> str | None:
    try:
        done = run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def verify_mps_client(pid: int, *, xml: str | None) -> tuple[bool | None, str]:
    """Is `pid` an MPS client? Read from the driver's own process table (`nvidia-smi -q -x`).

    Call it while the process holds a CUDA Context, or it will not be listed at all. An MPS client
    is listed with type `M+C`; an ordinary CUDA process with `C`. Returns (verdict, why).
    """
    if xml is None:
        return None, "nvidia-smi -q -x was unavailable, so MPS could not be verified"
    import xml.etree.ElementTree as ElementTree

    try:
        tree = ElementTree.fromstring(xml)
    except ElementTree.ParseError as broken:
        return None, f"nvidia-smi -q -x was not readable XML ({broken})"
    kinds = set()
    for info in tree.iter("process_info"):
        if (info.findtext("pid") or "").strip() == str(pid):
            kinds.add((info.findtext("type") or "").strip())
    if not kinds:
        return None, (f"process {pid} holds no Context the driver lists, so MPS could not be "
                      f"verified (a PID namespace hides it, or the Context was already released)")
    if any("M" in kind for kind in kinds):
        return True, f"the driver lists process {pid} as an MPS client ({', '.join(sorted(kinds))})"
    return False, (f"the driver lists process {pid} as an ordinary CUDA process "
                   f"({', '.join(sorted(kinds))}), not an MPS client")


def read_worker_facts(rank: int, *, visible_devices: int,
                      environment: Mapping[str, str] | None = None) -> WorkerFacts:
    environment = os.environ if environment is None else environment
    local = None
    for name in LOCAL_RANK_ENVIRONMENT:
        if name in environment:
            try:
                local = int(environment[name])
                break
            except ValueError:
                pass
    return WorkerFacts(
        rank=int(rank), hostname=socket.gethostname(),
        cpus=ordered_cpus(os.sched_getaffinity(0)), cpu_quota=_cgroup_cpu_quota(),
        visible_devices=int(visible_devices),
        cuda_visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=environment.get("CUDA_DEVICE_ORDER"),
        launcher_local_rank=local, mps=read_mps_status(environment))


# ---------------------------------------------------------------------------------------------
# decisions: pure
# ---------------------------------------------------------------------------------------------

def check_cpu_multiple(usable: int, workers: int, *, where: str = "this launch") -> int:
    """CPUs per worker, or a refusal that names the arithmetic that would fix it."""
    usable, workers = int(usable), int(workers)
    if workers < 1:
        raise PlacementError(f"{where}: no workers to place")
    if usable < workers or usable % workers:
        below = (usable // workers) * workers
        above = (usable // workers + 1) * workers
        options = [f"{above} CPUs ({above // workers} per worker)"]
        if below:
            options.insert(0, f"{below} CPUs ({below // workers} per worker)")
        raise PlacementError(
            f"{where}: {usable} CPU(s) are available to {workers} worker(s), and {usable} is not "
            f"an integer multiple of {workers} ({usable} = {workers} x {usable // workers} "
            f"+ {usable % workers}).\n"
            f"  Every worker gets an equal block of CPUs, so the count must divide exactly. "
            f"Make {' or '.join(options)} available -- with taskset, the launcher's binding "
            f"(e.g. mpirun --bind-to none with a cpuset) or the job's CPU request -- or change "
            f"the number of workers.")
    return usable // workers


def cpu_blocks(cpus: Sequence[int], workers: int, per_worker: int) -> list[tuple[int, ...]]:
    return [tuple(cpus[i * per_worker:(i + 1) * per_worker]) for i in range(int(workers))]


def balanced_assignment(throughput: Sequence[float], workers: int) -> list[int]:
    """Device ordinal for each worker, maximising the slowest worker's expected throughput.

    Model: a device measured at `s` steps per second alone gives each of `k` workers `s / k`.
    Under MPS that is conservative -- sharing usually costs less than a factor of k -- but it is
    the model that makes a slow device take fewer workers, and a synchronous ladder is as fast as
    its slowest rank. Each worker in turn takes the device whose per-worker share would stay
    highest; ties go to the lower ordinal, so equal devices fill in order and the map is
    reproducible from the measurement.
    """
    if not throughput:
        raise PlacementError("no CUDA device to place workers on")
    if any(not (value > 0) for value in throughput):
        raise PlacementError(f"a device measured no throughput: {list(throughput)}")
    load = [0] * len(throughput)
    chosen = []
    for _ in range(int(workers)):
        device = max(range(len(throughput)),
                     key=lambda d: (throughput[d] / (load[d] + 1), -d))
        load[device] += 1
        chosen.append(device)
    # Workers take devices in ordinal order, so worker i's device never depends on a tie between
    # two later workers. Sorting keeps the multiset the greedy step chose.
    return sorted(chosen)


@dataclass(frozen=True)
class LaunchPlan:
    """Every worker's CPUs and device, and why. Identical on every rank, by construction."""

    workers: int
    nodes: list[dict[str, Any]]
    #: rank -> {host, local_rank, cpus, device, co_tenants}
    assignment: dict[int, dict[str, Any]]
    placement: str
    throughput: dict[str, Any] | None
    shared_devices: bool
    #: Filled by the preflight once the plan is made: whose plan this copy is, the full
    #: measurement behind `throughput`, and this rank's MPS status (verified when it had to be).
    this_rank: int = 0
    measurement: dict[str, Any] | None = None
    mps: MpsStatus | None = None

    def for_rank(self, rank: int | None = None) -> dict[str, Any]:
        return self.assignment[int(self.this_rank if rank is None else rank)]

    def record(self) -> dict[str, Any]:
        return {"workers": self.workers, "placement": self.placement, "nodes": self.nodes,
                "this_worker": self.for_rank(),
                "worker_map": {str(r): a for r, a in sorted(self.assignment.items())},
                "measurement": self.measurement, "shared_devices": self.shared_devices,
                "mps": self.mps.record() if self.mps is not None else None}


def group_nodes(facts: Sequence[WorkerFacts]) -> list[NodeFacts]:
    """Ranks by host, in rank order. The local rank is the position in that list."""
    hosts: dict[str, list[WorkerFacts]] = {}
    for fact in sorted(facts, key=lambda f: f.rank):
        hosts.setdefault(fact.hostname, []).append(fact)
    nodes = []
    for host, members in hosts.items():
        cpus = ordered_cpus(set().union(*(set(m.cpus) for m in members)))
        quotas = [m.cpu_quota for m in members if m.cpu_quota is not None]
        usable = min([len(cpus), *quotas])
        nodes.append(NodeFacts(hostname=host, ranks=tuple(m.rank for m in members), cpus=cpus,
                               usable_cpus=usable,
                               visible_devices=min(m.visible_devices for m in members)))
    return nodes


def plan_cpus(facts: Sequence[WorkerFacts], *, where: str = "this launch"
              ) -> dict[int, dict[str, Any]]:
    """Each rank's host, local rank and CPU block, or the refusal. Pure, and the same on every rank.

    Separate from `plan_launch` because the CPUs are decided -- and bound -- BEFORE the devices are
    measured: a measurement taken while every rank's threads compete for all 48 CPUs describes a
    machine the run will not be on.
    """
    nodes = group_nodes(facts)
    by_rank = {f.rank: f for f in facts}
    planned: dict[int, dict[str, Any]] = {}
    for node in nodes:
        local_workers = len(node.ranks)
        per_worker = check_cpu_multiple(
            node.usable_cpus, local_workers,
            where=f"{where} on {node.hostname}" if len(nodes) > 1 else where)
        blocks = cpu_blocks(node.cpus, local_workers, per_worker)
        for local, rank in enumerate(node.ranks):
            launcher = by_rank[rank].launcher_local_rank
            if launcher is not None and launcher != local:
                raise PlacementError(
                    f"{where}: the launcher says rank {rank} is local rank {launcher} on "
                    f"{node.hostname}, but it is local rank {local} among the ranks that report "
                    f"that host name. Two nodes sharing a host name, or a launcher numbering "
                    f"differently, would put two workers on one block of CPUs.")
            planned[rank] = {"host": node.hostname, "local_rank": local,
                             "cpus": list(blocks[local]), "cpus_per_worker": per_worker,
                             "usable_cpus_on_host": node.usable_cpus,
                             # What the launcher had bound this rank to, before the rebinding.
                             # `mpirun` binds each rank to one core by default, and then the
                             # launch may use only those cores: that is why the docs say
                             # `--bind-to none`.
                             "launcher_cpus": sorted(by_rank[rank].cpus)}
    return planned


def plan_launch(facts: Sequence[WorkerFacts], *, platform: str, device_policy: str,
                explicit_device: int | None = None,
                throughput: Mapping[str, Sequence[float]] | None = None,
                where: str = "this launch") -> LaunchPlan:
    """The whole placement, from gathered facts and (for CUDA) measured throughput per host.

    `throughput` maps a host name to steps per second for each of its device ordinals. It is
    required exactly when `needs_measurement` says so; the preflight measures, gathers and passes
    it in, so every rank plans from the same numbers.
    """
    facts = sorted(facts, key=lambda f: f.rank)
    workers = len(facts)
    nodes = group_nodes(facts)
    by_rank = {f.rank: f for f in facts}
    assignment: dict[int, dict[str, Any]] = {}
    node_records = []
    shared = False
    placement = "not a CUDA platform"

    cpus = plan_cpus(facts, where=where)
    for node in nodes:
        local_workers = len(node.ranks)
        per_worker = cpus[node.ranks[0]]["cpus_per_worker"]

        devices: list[int | None] = [None] * local_workers
        if platform == "CUDA":
            if explicit_device is not None:
                devices = [int(explicit_device)] * local_workers
                placement = "named on the command line (--device)"
            elif device_policy == "openmm":
                placement = "machine.openmm.device_policy: openmm -- OpenMM selects"
                # Nobody here placed them, so nobody here can say they do not share. They do
                # not only when every rank was handed exactly one distinct device from outside.
                visible = [by_rank[r].cuda_visible_devices for r in node.ranks]
                partitioned = (all(v is not None and len([p for p in v.split(",") if p.strip()])
                                   == 1 for v in visible) and len(set(visible)) == len(visible))
                shared = shared or (local_workers > 1 and not partitioned)
            elif local_workers == 1:
                # As before: the first visible device, or -- when a single process never counted
                # its devices -- no DeviceIndex at all, and OpenMM's default.
                devices = [0 if node.visible_devices > 0 else None]
                placement = "single worker: first visible device"
            elif node.visible_devices < 1:
                raise PlacementError(
                    f"{where}: the CUDA platform was selected but no CUDA device is visible on "
                    f"{node.hostname}. Refusing rather than letting every worker fall onto one "
                    f"GPU, or onto none.")
            else:
                measured = (throughput or {}).get(node.hostname)
                if measured is None or len(measured) != node.visible_devices:
                    raise PlacementError(
                        f"{where}: no throughput measurement for the {node.visible_devices} "
                        f"device(s) on {node.hostname}; placement by measured throughput cannot "
                        f"proceed without one.")
                devices = balanced_assignment(list(measured), local_workers)
                placement = "measured throughput (balanced)"
            if explicit_device is not None or device_policy != "openmm":
                shared = shared or len(set(devices)) < len(devices)

        for local, rank in enumerate(node.ranks):
            assignment[rank] = {
                "host": node.hostname, "local_rank": local, "cpus": cpus[rank]["cpus"],
                "launcher_cpus": cpus[rank]["launcher_cpus"],
                "device": devices[local],
                "co_tenants": (sum(1 for d in devices if d == devices[local])
                               if devices[local] is not None else None)}
        node_records.append({"host": node.hostname, "ranks": list(node.ranks),
                             "usable_cpus": node.usable_cpus, "cpus_per_worker": per_worker,
                             "visible_devices": node.visible_devices})

    return LaunchPlan(workers=workers, nodes=node_records, assignment=assignment,
                      placement=placement,
                      throughput={h: list(v) for h, v in (throughput or {}).items()} or None,
                      shared_devices=shared)


def needs_measurement(facts: Sequence[WorkerFacts], *, platform: str, device_policy: str,
                      explicit_device: int | None) -> dict[str, int]:
    """Hosts whose devices must be measured, and how many devices each has. Pure."""
    if platform != "CUDA" or explicit_device is not None or device_policy == "openmm":
        return {}
    return {node.hostname: node.visible_devices for node in group_nodes(facts)
            if len(node.ranks) > 1 and node.visible_devices > 0}


def refuse_unverified_sharing(plan: LaunchPlan, mps: MpsStatus, *, rank: int,
                              where: str = "this launch") -> None:
    """A shared GPU without VERIFIED MPS is refused, and the refusal is the WHOLE LAUNCH's.

    The rule is global because a ladder is synchronous: one rank sharing a device sets the pace for
    every rank, so a rank sitting alone is not thereby fine. That made the first version of this
    message read as nonsense to the one person it mattered to -- it said "device 1 hosts 1
    worker(s)" (this rank's own placement) and then "MPS is required whenever a GPU hosts more than
    one worker", and both were true. It cost an hour and three wrong theories. The message now
    names the devices that ARE shared, across the launch, before it says anything about the rank
    reading it.
    """
    if not plan.shared_devices or mps.status == "verified":
        return
    mine = plan.for_rank(rank)
    tenants: dict[Any, int] = {}
    for worker in plan.assignment.values():
        if worker["device"] is not None:
            tenants[worker["device"]] = tenants.get(worker["device"], 0) + 1
    crowded = sorted((device, count) for device, count in tenants.items() if count > 1)

    if crowded:
        sharing = "this launch shares " + ", ".join(
            f"device {device} between {count} workers" for device, count in crowded)
        here = (f"this rank is alone on device {mine['device']}, but the launch cannot proceed "
                f"unverified: a ladder is synchronous, so the shared device sets the pace for "
                f"every rank"
                if (mine["device"] is not None and tenants.get(mine["device"], 0) == 1)
                else f"this rank is one of {tenants.get(mine['device'])} on device "
                     f"{mine['device']}")
    else:
        sharing = ("this launch does not partition its workers onto distinct devices "
                   "(machine.openmm.device_policy: openmm, with nothing outside this package "
                   "giving each worker its own)")
        here = "no worker here can be shown to have a device to itself"

    raise PlacementError(
        f"{where}: {sharing}. NVIDIA MPS is {mps.status} for this process ({mps.detail}); "
        f"{here}.\n"
        f"  Without MPS, workers on one GPU are time-sliced: each waits for the others' kernels, "
        f"and a synchronous ladder runs at the pace of that shared GPU. MPS is required whenever "
        f"a GPU hosts more than one worker, and this package does not start or stop the daemon.\n"
        f"  Start one (docs/basics/md-run.md, \"CPUs, devices and MPS\"):\n"
        f"      export CUDA_MPS_PIPE_DIRECTORY=<a directory you own>\n"
        f"      export CUDA_MPS_LOG_DIRECTORY=<another>\n"
        f"      nvidia-cuda-mps-control -d\n"
        f"  and launch from that environment, or give each worker its own GPU.\n"
        f"  {_why_doubled_up(plan, mine)}")


def _why_doubled_up(plan: "LaunchPlan", mine: Mapping[str, Any]) -> str:
    """Why more workers landed on one device than another, in the refusal that resulted.

    Placement doubles up only when a MEASUREMENT said the alternative was worse, and the numbers
    are the first thing a reader needs -- a device can measure slow because it IS slower or because
    somebody else is using it, and those are indistinguishable from here. Leaving the reader to go
    and find `steps_per_second` in a record cost an hour and three wrong theories once already.
    """
    measured = (plan.measurement or {}).get(mine.get("host"))
    if not measured:
        return f"Workers were placed by: {plan.placement}."
    rates = measured.get("steps_per_second") or []
    listing = ", ".join(f"device {index} {rate:.0f}" for index, rate in enumerate(rates))
    spread = ""
    if rates and min(rates) > 0 and max(rates) / min(rates) > 1.5:
        slowest = rates.index(min(rates))
        spread = (f" Device {slowest} measured {max(rates) / min(rates):.1f}x slower than the "
                  f"fastest, so it was given fewer workers or none. A device measures slow "
                  f"because it is slower OR because something else on this machine is using it, "
                  f"and this measurement cannot tell those apart -- check the card before "
                  f"concluding anything about the hardware.")
    return (f"Workers were placed by: {plan.placement}. Measured steps/s at preflight: "
            f"{listing}.{spread}")


# ---------------------------------------------------------------------------------------------
# effects: measurement and binding
# ---------------------------------------------------------------------------------------------

def measure_device_throughput(system, positions, *, devices: int, precision: str,
                              warmup_steps: int = 50, min_seconds: float = 1.0,
                              max_steps: int = 5000, timestep_ps: float = 0.002,
                              clock=time.perf_counter) -> dict[str, Any]:
    """Steps per second of THIS run's System on each device, measured one device at a time.

    The run's own System, because relative device speed depends on it: a 22-atom system is
    bounded by the host, a solvated protein by the GPU. Measured alone, one device after another,
    so the numbers describe the devices and not each other. Whatever else the GPUs are running at
    that moment is part of the measurement, deliberately: a busy card IS slower for this run.

    A Verlet integrator with no thermostat, because this measures kernels rather than sampling
    anything, and nothing it produces is kept.
    """
    from openmm import Context, Platform, VerletIntegrator

    platform = Platform.getPlatformByName("CUDA")
    rates, rows = [], []
    for device in range(int(devices)):
        integrator = VerletIntegrator(timestep_ps)
        context = Context(system, integrator, platform,
                          {"DeviceIndex": str(device), "Precision": precision})
        try:
            context.setPositions(positions)
            context.setVelocitiesToTemperature(300.0, 1)
            integrator.step(int(warmup_steps))
            context.getState(getEnergy=True)                   # synchronise before timing
            steps, started = 0, clock()
            chunk = max(10, int(warmup_steps))
            while steps < max_steps:
                integrator.step(chunk)
                steps += chunk
                context.getState(getEnergy=True)
                if clock() - started >= min_seconds:
                    break
            elapsed = clock() - started
        finally:
            del context, integrator
        rate = steps / elapsed if elapsed > 0 else 0.0
        rates.append(rate)
        rows.append({"device": device, "steps": steps, "seconds": round(elapsed, 6),
                     "steps_per_second": round(rate, 3)})
    return {"steps_per_second": rates, "devices": rows, "particles": system.getNumParticles(),
            "warmup_steps": int(warmup_steps), "min_seconds": float(min_seconds),
            "timestep_ps": float(timestep_ps), "precision": precision,
            "integrator": "VerletIntegrator (measurement only)"}


def bind_worker(cpus: Sequence[int]) -> list[int]:
    """Bind this process to its block. Before any Context, so CUDA's host threads inherit it."""
    os.sched_setaffinity(0, set(int(c) for c in cpus))
    return sorted(os.sched_getaffinity(0))
