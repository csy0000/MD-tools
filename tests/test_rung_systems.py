"""The run index of a dataset layout: `<method>-run<n>`.

OBSOLETE AND REMOVED (0.5.4): the tests of `build/rungs.write_rung_systems`, which serialised
`remd<n>/build_state<n>.xml` into a run at `build-md` time. `build-md` stopped writing rungs when a
ladder began integrating the saved states of `md-openmm build-top --rest2-scaler` (commit
e1a84e1), and the module was deleted with nothing left calling it. The same premise -- the file is
the Hamiltonian, term by term -- is tested where the states are made now:
tests/test_rest2_scaler.py and tests/test_ladder_on_saved_states.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_tools import layout as L



# -- the run index ------------------------------------------------------------------------------

def test_the_first_run_of_a_method_is_one(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    assert dataset.next_run_index("REST2") == 1
    assert dataset.next_run("REST2").root.name == "REST2-run1"


def test_the_next_index_is_one_past_the_highest(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run2"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1, 2]
    assert dataset.next_run_index("REST2") == 3


def test_a_gap_is_never_reused(tmp_path):
    """THE POINT, not an oversight.

    A gap means a run was moved, archived or registered elsewhere -- `data-register` takes the
    source away and leaves a symlink. Handing its number to a new run would make two different
    experiments share an identity that some manifest already refers to.
    """
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run3"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1, 3]
    assert dataset.next_run_index("REST2") == 4, "2 is a gap and must not be handed out again"


def test_methods_are_counted_separately(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run2", "cMD-run1"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.next_run_index("REST2") == 3
    assert dataset.next_run_index("cMD") == 2
    assert dataset.next_run_index("AIS") == 1


def test_a_non_numeric_suffix_is_ignored_rather_than_fatal(tmp_path):
    """A directory someone named `REST2-run-old` must not block every new run."""
    dataset = L.DatasetLayout(tmp_path / "ALA")
    for name in ("REST2-run1", "REST2-run-old", "REST2-runX", "REST2-run2.bak"):
        (dataset.root / name).mkdir(parents=True)
    assert dataset.existing_runs("REST2") == [1]
    assert dataset.next_run_index("REST2") == 2


def test_a_file_named_like_a_run_is_not_counted(tmp_path):
    dataset = L.DatasetLayout(tmp_path / "ALA")
    dataset.root.mkdir(parents=True)
    (dataset.root / "REST2-run7").write_text("not a directory", encoding="utf-8")
    assert dataset.existing_runs("REST2") == []
    assert dataset.next_run_index("REST2") == 1
