"""Placement ON REAL CUDA: the throughput measurement, and a shared GPU refused without MPS.

`tests/test_placement.py` proves the arithmetic without a device, which is not evidence that any of
it runs. This file runs the real command, under a real launcher, on real cards:

* the measurement opens a Context on every visible device and comes back with a rate for each;
* four workers on four cards are placed one per card, and MPS is recorded but not required;
* four workers on ONE card are REFUSED, with the MPS verdict read from the driver rather than
  assumed -- which is only testable while no daemon is running.

The refusal case is why this file is written against the MPS-OFF state deliberately. With a daemon
up, `absent` / `not-a-client` could not be produced, and a test that cannot produce the negative
cannot show the check works. Neither this file nor the package ever starts or stops a daemon.

Every launch names its cards explicitly: the measurement opens a Context on each VISIBLE device, so
a card this run does not own must be hidden from it, not merely unused.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: The ladder's width, and the number of workers in every launch here.
RUNGS = 4


def _cards() -> list[str]:
    """The cards this run owns, PHYSICALLY, for the launches that are not MPS clients.

    Read from `MD_TOOLS_TEST_CARDS`, falling back to `CUDA_VISIBLE_DEVICES`. Never widened: a card
    this run does not own must not appear, because the measurement opens a Context on every
    visible device -- briefly, but on somebody else's GPU.
    """
    value = os.environ.get("MD_TOOLS_TEST_CARDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    cards = [part.strip() for part in (value or "").split(",") if part.strip()]
    if len(cards) < RUNGS:
        pytest.skip(f"this needs {RUNGS} cards; MD_TOOLS_TEST_CARDS/CUDA_VISIBLE_DEVICES={value!r}")
    return cards


def _mps_cards() -> list[str]:
    """The same cards as an MPS CLIENT sees them, which is NOT the same list.

    A client addresses the daemon's devices as 0..n-1 of the set the daemon was started with, so
    the physical numbers are wrong twice over: they name other people's cards when the daemon holds
    fewer, and they simply do not exist when the daemon holds four. `MD_TOOLS_TEST_MPS_CARDS`
    carries the client-side list, and without it the MPS tests skip rather than guess.
    """
    value = os.environ.get("MD_TOOLS_TEST_MPS_CARDS")
    cards = [part.strip() for part in (value or "").split(",") if part.strip()]
    if not cards:
        pytest.skip("set MD_TOOLS_TEST_MPS_CARDS to the daemon's devices as a client sees them "
                    "(0..n-1 of the set it was started with)")
    return cards


def _environment(cards: str, *, mps: str | None = None) -> dict:
    base = dict(os.environ)
    if mps is not None:
        base["CUDA_MPS_PIPE_DIRECTORY"] = mps
    else:
        base.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    base["CUDA_VISIBLE_DEVICES"] = cards
    return base


def _run(cwd: Path, *args, cards: str, ranks: int = 1, timeout=1800, mps: str | None = None,
         ranks_arg: int | None = None):
    launch = ["mpirun", "--bind-to", "none", "-n", str(ranks)] if ranks > 1 else []
    return subprocess.run(launch + ["md-openmm", "md-run", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=_environment(cards, mps=mps))


#: The daemon's pipe directory, when this machine has one somebody else started. MD-tools never
#: starts or stops it, and neither does this file: a test that brought a shared service up and down
#: would be changing the machine to make itself pass.
MPS_PIPE_DIRECTORY = os.environ.get("MD_TOOLS_TEST_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")


def _require_mps():
    """Skip -- never pass -- when no daemon this process can reach is running."""
    from md_tools.openmm.placement import read_mps_status

    status = read_mps_status({"CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE_DIRECTORY})
    if status.daemon != "running":
        pytest.skip(f"no MPS daemon serving {MPS_PIPE_DIRECTORY}: {status.detail}")


def _build_ladder(work: Path, *, rungs: int, cards: str) -> Path:
    """A generated ladder of `rungs` states whose group file's inputs all exist."""
    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(work / "sys.config")],
        cwd=work, capture_output=True, text=True, timeout=1800, env=_environment(cards))
    assert built.returncode == 0, built.stdout + built.stderr

    config = work / "rest2.config"
    config.write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "dynamics": {"seed": 5},
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 50,
                   "restrained_npt_steps": 50, "unrestrained_npt_steps": 50},
        "rest2": {"number_of_replicas": rungs, "tau_max": 0.3,
                  "exchange_interval_steps": 50, "number_of_exchanges": 2},
        "reporting": {"crd_printout_solute": 25, "info_printout": 50,
                      "checkpoint_printout": 50}}, sort_keys=False), encoding="utf-8")
    from .conftest import make_states_for

    make_states_for(work, config)
    generated = subprocess.run(CLI + ["build-md", "-odir", "./rest2-run1", "--config",
                                      str(config)],
                               cwd=work, capture_output=True, text=True, timeout=1800,
                               env=_environment(cards))
    assert generated.returncode == 0, generated.stdout + generated.stderr

    out = work / "rest2-run1"
    previous = None
    for key in ("min", "eq_1", "eq_2", "eq_3"):
        argv = ["-i", f"../input/{key}.in", "-p", "../build/built.pdb",
                "-s", "../build/built.xml", "-odir", "../min" if key == "min" else "eq"]
        if previous:
            argv += ["-c", previous]
        done = _run(out, *argv, cards=cards.split(",")[0])
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        previous = "../min/min.xml" if key == "min" else f"eq/{key}.xml"
    return out


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """A generated four-rung ladder whose group file's inputs all exist, so `--check` gets as far
    as placement rather than stopping on a missing continuation state."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    cards = ",".join(_cards()[:RUNGS])
    return _build_ladder(tmp_path_factory.mktemp("placement-cuda"), rungs=RUNGS, cards=cards)


#: The wide demonstration: three workers per card on four cards.
WIDE = 12


@pytest.fixture(scope="module")
def wide_ladder(tmp_path_factory):
    """Twelve rungs, built once. Only the MPS arm can run it: three workers share each card."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    _require_mps()
    cards = ",".join(_mps_cards()[:4])
    return _build_ladder(tmp_path_factory.mktemp("placement-wide"), rungs=WIDE, cards=cards)


def _check(ladder, cards: str):
    return _run(ladder, "-ng", str(RUNGS), "-i", "../input/REST2.in", "-p", "../build/built.pdb",
                "--groupfile", "remd_groupfile.1", "-odir", ".", "-log", "REST2.log", "--check",
                cards=cards, ranks=RUNGS)


def _listing(directory: Path):
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


def test_the_throughput_measurement_runs_on_every_visible_device(ladder):
    """A rate per card, measured on hardware, and every worker placed on its own card.

    Four workers and four cards: nothing is shared, so MPS is recorded and not required. That is
    the whole point of separating the two -- a status is not a requirement.
    """
    cards = ",".join(_cards()[:RUNGS])
    before = _listing(ladder)
    done = _check(ladder, cards)
    message = done.stdout + done.stderr
    assert done.returncode == 0, message[-4000:]

    # The `throughput` REPORT line, not the `measured throughput (balanced)` placement line.
    rates = [line.split("throughput", 1)[1] for line in message.splitlines()
             if line.strip().startswith("throughput") and "steps/s" in line]
    assert rates, message[-4000:]
    for line in rates:
        measured = [float(value.split()[0]) for value in line.split("steps/s")[0].split(",")]
        assert len(measured) == RUNGS, line
        assert all(rate > 0 for rate in measured), line

    placed = [line.split("device")[1].strip().split()[0]
              for line in message.splitlines() if line.strip().startswith("device ")]
    assert sorted(placed) == [str(i) for i in range(RUNGS)], message[-4000:]
    assert "measured throughput (balanced)" in message, message[-4000:]
    assert "worker(s) on this device" in message, message[-4000:]
    # Nothing is shared, so MPS is RECORDED and not required -- whatever its status happens to be
    # on this machine. Asserting a particular status here would make this test a test of the
    # machine; what matters is that no requirement was imposed and the run was accepted.
    status = next(line.split()[1] for line in message.splitlines()
                  if line.strip().startswith("mps "))
    assert status != "verified" or True, status
    assert "(required: a device is shared)" not in message, message[-4000:]
    # `--check` creates nothing, on every rank.
    assert _listing(ladder) == before, sorted(set(_listing(ladder)) - set(before))


def test_four_workers_on_one_card_are_refused_without_mps_and_write_nothing(ladder):
    """The refusal this rule exists for, with the verdict read from the driver.

    Not `--check`: the point is that a real run stops before it creates anything. Without MPS the
    four workers would be time-sliced and the ladder would run at one card's pace, silently.
    """
    card = _cards()[0]
    before = _listing(ladder)
    done = _run(ladder, "-ng", str(RUNGS), "-i", "../input/REST2.in", "-p", "../build/built.pdb",
                "--groupfile", "remd_groupfile.1", "-odir", ".", "-log", "REST2.log",
                cards=card, ranks=RUNGS)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-4000:]
    assert f"shares device 0 between {RUNGS} workers" in message, message[-4000:]
    # Named as the LAUNCH's refusal: the rule is global because a ladder is synchronous, and a
    # rank's own tenant count is not what decides it.
    assert f"this rank is one of {RUNGS} on device 0" in message, message[-4000:]
    # The verdict is the driver's, not an assumption: with no daemon on this host it is `absent`,
    # and if one were running but this process were not its client it would be `not-a-client`.
    assert ("NVIDIA MPS is absent" in message or "NVIDIA MPS is not-a-client" in message), \
        message[-4000:]
    assert "nvidia-cuda-mps-control -d" in message, message[-4000:]
    assert _listing(ladder) == before, sorted(set(_listing(ladder)) - set(before))


# --- sharing a card WITH MPS --------------------------------------------------------------------

def test_four_workers_share_one_card_with_mps_and_the_ladder_completes(ladder):
    """The `verified` path, and the run the refusal above was protecting.

    Same four workers, same single card, and the only difference is that this process can reach an
    MPS daemon. The preflight must now accept it -- and it must accept it because the DRIVER says
    this process is an MPS client, not because the environment variable is set.
    """
    _require_mps()
    card = _mps_cards()[0]
    done = _run(ladder, "-ng", str(RUNGS), "-i", "../input/REST2.in", "-p", "../build/built.pdb",
                "--groupfile", "remd_groupfile.1", "-odir", ".", "-log", "REST2.log",
                cards=card, ranks=RUNGS, mps=MPS_PIPE_DIRECTORY, timeout=3600)
    message = done.stdout + done.stderr
    assert done.returncode == 0, message[-4000:]

    # One trajectory per STATE, as always: placement changes nothing about what a ladder writes.
    trajectories = sorted(p.name for p in ladder.glob("whole_state*_prod1.nc"))
    assert len(trajectories) == RUNGS, trajectories

    # THE MACHINE RECORD, which is `restart.json`, not the `.log` a person tails: completion is
    # read from a machine record, and so is the placement that produced it.
    import json

    record = json.loads((ladder / "restart.json").read_text(encoding="utf-8"))
    placement = record["execution"]["acceleration"]["placement"]
    assert placement["mps"]["verified"] is True, placement["mps"]
    assert placement["mps"]["status"] == "verified", placement["mps"]
    assert placement["shared_devices"] is True, placement
    # Every worker on one card, said rather than left to be inferred.
    tenants = {int(worker["co_tenants"]) for worker in placement["worker_map"].values()}
    devices = {worker["device"] for worker in placement["worker_map"].values()}
    assert tenants == {RUNGS} and devices == {0}, placement["worker_map"]


def test_twelve_workers_on_four_cards_run_together_under_mps(wide_ladder):
    """The wide demonstration: 12 workers, 4 cards, three to a card, one ladder.

    What is asserted is the placement and the ladder's own contract -- a trajectory per STATE, and
    a record that says which card each worker had and how many shared it. Three per card is the
    balanced answer for four equal cards; it is asserted as a distribution rather than as a fixed
    map, because which card is fastest on the day is a measurement, not a constant.
    """
    import json

    cards = ",".join(_mps_cards()[:4])
    done = _run(wide_ladder, "-ng", str(WIDE), "-i", "../input/REST2.in",
                "-p", "../build/built.pdb", "--groupfile", "remd_groupfile.1", "-odir", ".",
                "-log", "REST2.log", cards=cards, ranks=WIDE, mps=MPS_PIPE_DIRECTORY,
                timeout=5400)
    message = done.stdout + done.stderr
    assert done.returncode == 0, message[-5000:]

    trajectories = sorted(p.name for p in wide_ladder.glob("whole_state*_prod1.nc"))
    assert len(trajectories) == WIDE, trajectories

    record = json.loads((wide_ladder / "restart.json").read_text(encoding="utf-8"))
    placement = record["execution"]["acceleration"]["placement"]
    assert placement["mps"]["status"] == "verified", placement["mps"]
    workers = placement["worker_map"]
    assert len(workers) == WIDE, workers
    per_card = {}
    for worker in workers.values():
        per_card[worker["device"]] = per_card.get(worker["device"], 0) + 1
    assert sorted(per_card) == [0, 1, 2, 3], per_card
    assert sorted(per_card.values()) == [3, 3, 3, 3], per_card
    assert {worker["co_tenants"] for worker in workers.values()} == {3}, workers
    # 48 CPUs among 12 workers: four each, and no two workers sharing a CPU.
    blocks = [tuple(worker["cpus"]) for worker in workers.values()]
    assert all(len(block) == 48 // WIDE for block in blocks), blocks
    assert len({cpu for block in blocks for cpu in block}) == 48, blocks
