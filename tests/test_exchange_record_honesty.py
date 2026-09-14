"""`u_evaluated` is observed from the matrix, not asserted about it.

`exchange.nc` documents `u_evaluated` as "1 where u was actually computed". The driver wrote
`np.ones((n, n))` -- a claim that was true, because the matrix is dense, but one that could not
become false no matter what the matrix held. A flag that cannot be wrong cannot catch the day the
thing it describes changes, and this flag exists precisely so a reader can tell a computed entry
from one that was skipped.

WHY THE MATRIX IS DENSE AND STAYS DENSE. Three renderers index it, by two different conventions:
`rem_log.block_rows` reads `u[state, state]` and `u[state, mate]`, `exchange_free_energies` needs
both cross terms of every proposed pair, and `_write_exchange_csv` reads `u[state][walker]`. A
sparse matrix would leave holes that some of them index, and a hole reaches the reader as `nan` in
`rem.log`. So rules declare what they read through `required_entries` and the driver evaluates
everything anyway -- deliberately, and said so in `_reduced_potential_matrix`.

PLATFORM_POLICY_EXEMPTION: no Context and no propagation. What is under test is which array the
driver hands the reporter.
"""
from __future__ import annotations

import inspect

import numpy as np

from md_tools.remd import driver as driver_module


def test_the_flag_is_derived_from_the_matrix_rather_than_hardcoded():
    """The one assertion that catches a regression to `np.ones`.

    CODE ONLY, NOT COMMENTS. The first version of this test searched the whole function text and
    failed on the comment that explains why `np.ones` was wrong -- the assertion matched the
    prose describing the defect rather than the defect. A source-level test has to look at the
    statements, or it polices its own documentation.
    """
    source = inspect.getsource(driver_module)
    exchange = source[source.index("def _exchange"):source.index("def _apply_reservoir")]
    statements = "\n".join(line for line in exchange.splitlines()
                           if not line.lstrip().startswith("#"))

    assert "u_evaluated=np.isfinite(matrix)" in statements, (
        "u_evaluated must be observed from the matrix; `np.ones(...)` is a claim that cannot fail")
    assert "np.ones" not in statements, "the hardcoded all-ones flag is back"


def test_isfinite_marks_exactly_the_computed_entries():
    """The derivation itself: finite means computed, non-finite means not.

    This is the arithmetic the driver relies on, checked directly so the source assertion above is
    not the only thing standing behind the behaviour.
    """
    matrix = np.array([[-1.0, -2.0], [-3.0, np.nan]], dtype=float)
    flags = np.isfinite(matrix).astype(np.int8)
    assert flags.tolist() == [[1, 1], [1, 0]]
    assert flags.dtype == np.int8, "the storage variable is i1"


def test_a_dense_matrix_flags_every_entry_as_computed():
    """The ordinary case, which must not change: nothing is skipped today."""
    matrix = np.full((4, 4), -10.0)
    flags = np.isfinite(matrix).astype(np.int8)
    assert flags.sum() == 16
    assert (flags == 1).all()


def test_the_driver_still_evaluates_every_entry_despite_the_declaration():
    """`required_entries` is a contract rules honour, NOT a licence for the driver to skip.

    Stated as a test because the tempting next change -- evaluate only what was declared -- breaks
    `rem.log` in a way no unit test of the rules would notice.
    """
    matrix_source = inspect.getsource(driver_module.ReplicaRun._reduced_potential_matrix)
    assert "for walker in range(n)" in matrix_source, (
        "every walker's entry is still evaluated; a declaration must not narrow this loop")
    assert "required_entries" in matrix_source or "dense" in matrix_source.lower(), (
        "the reason the matrix stays dense belongs beside the loop that keeps it dense")
