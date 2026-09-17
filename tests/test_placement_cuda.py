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
    """The cards this run owns, from the environment the suite was launched with."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    cards = [part.strip() for part in (value or "").split(",") if part.strip()]
    if len(cards) < RUNGS:
        pytest.skip(f"this needs {RUNGS} cards; CUDA_VISIBLE_DEVICES={value!r}")
    return cards


def _environment(cards: str) -> dict:
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    base["CUDA_VISIBLE_DEVICES"] = cards
    return base


def _run(cwd: Path, *args, cards: str, ranks: int = 1, timeout=1800):
    launch = ["mpirun", "--bind-to", "none", "-n", str(ranks)] if ranks > 1 else []
    return subprocess.run(launch + ["md-openmm", "md-run", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=_environment(cards))


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """A generated four-rung ladder whose group file's inputs all exist, so `--check` gets as far
    as placement rather than stopping on a missing continuation state."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH")
    cards = ",".join(_cards()[:RUNGS])
    work = tmp_path_factory.mktemp("placement-cuda")
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
        "rest2": {"number_of_replicas": RUNGS, "tau_max": 0.3,
                  "exchange_interval_steps": 50, "number_of_exchanges": 2},
        "reporting": {"crd_printout_solute": 25, "info_printout": 50,
                      "checkpoint_printout": 50}}, sort_keys=False), encoding="utf-8")
    from .conftest import make_states_for

    make_states_for(work, config)
    generated = subprocess.run(CLI + ["build-md", "-odir", "./rest2-run1", "--config",
                                      str(config)],
                               cwd=work, capture_output=True, text=True, timeout=900,
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
    # MPS is off on this machine, and with nothing shared that is simply recorded.
    assert "mps               absent" in message, message[-4000:]
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
    assert f"device 0 hosts {RUNGS} worker(s)" in message, message[-4000:]
    # The verdict is the driver's, not an assumption: with no daemon on this host it is `absent`,
    # and if one were running but this process were not its client it would be `not-a-client`.
    assert ("NVIDIA MPS is absent" in message or "NVIDIA MPS is not-a-client" in message), \
        message[-4000:]
    assert "nvidia-cuda-mps-control -d" in message, message[-4000:]
    assert _listing(ladder) == before, sorted(set(_listing(ladder)) - set(before))
