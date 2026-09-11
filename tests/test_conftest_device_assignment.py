"""The suite's per-test GPU assignment must not leak from one test into the next.

`tests/conftest.py` writes `CUDA_VISIBLE_DEVICES` into the worker's environment before each CUDA
test. It used to leave it there, so a test that received no assignment inherited the previous
test's card: `test_cuda_coverage_matrix.py`, which must see the machine as it is, saw one device
after any ordinary CUDA test on the same worker. These drive the hook directly with stand-in
items, so they run anywhere and in milliseconds.
"""
from __future__ import annotations

import os

import pytest

from tests import conftest


class _Item:
    def __init__(self, module: str, gpu: bool = True):
        self.fspath = f"/anywhere/tests/{module}"
        self.keywords = {"gpu": True} if gpu else {}


@pytest.fixture
def nine_cards(monkeypatch):
    """A nine-GPU machine, no outside CUDA_VISIBLE_DEVICES, running as xdist worker 3."""
    monkeypatch.setattr(conftest, "_MACHINE_DEVICES", [str(i) for i in range(9)])
    monkeypatch.setattr(conftest, "_INHERITED_VISIBLE_DEVICES", None)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_an_ordinary_cuda_test_gets_one_card(nine_cards):
    conftest.pytest_runtest_setup(_Item("test_cmd_cuda_smoke.py"))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_the_unassigned_module_sees_the_whole_machine_after_an_ordinary_test(nine_cards):
    """The defect: this read "3" -- the previous test's card -- and a 4-state ladder saw 1 device."""
    conftest.pytest_runtest_setup(_Item("test_cmd_cuda_smoke.py"))
    conftest.pytest_runtest_setup(_Item("test_cuda_coverage_matrix.py"))
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_cpu_test_after_a_cuda_test_inherits_nothing(nine_cards):
    conftest.pytest_runtest_setup(_Item("test_cmd_cuda_smoke.py"))
    conftest.pytest_runtest_setup(_Item("test_config_generation.py", gpu=False))
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_ladder_after_an_ordinary_test_gets_its_full_pool(nine_cards):
    conftest.pytest_runtest_setup(_Item("test_cmd_cuda_smoke.py"))
    conftest.pytest_runtest_setup(_Item("test_md_run_mpi_gpu.py"))
    assigned = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(assigned) == 8 and "0" not in assigned, assigned


def test_an_outside_restriction_is_restored_rather_than_widened(nine_cards, monkeypatch):
    """A value set from outside is an instruction; the reset puts it back, never removes it."""
    monkeypatch.setattr(conftest, "_INHERITED_VISIBLE_DEVICES", "1,2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"                  # a stale slice from somewhere
    conftest.pytest_runtest_setup(_Item("test_cuda_coverage_matrix.py"))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
