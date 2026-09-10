"""An absent cost is not a zero cost, and a ladder block is not a suggestion.

WHAT THESE COVER

    A. A CV-enabled committed prefix whose `cost` is missing, null, empty or not a mapping was
       waved through: `validate()` only called the strict parser when `entry.get("cost") is not
       None`, and `CommittedPrefix.from_record` restored an absent cost as ZERO. So a two-row,
       two-torsion prefix that should restore 2 observations and 4 scalar evaluations restored
       0 and 0, and the continuation carried on with counters that had quietly lost their
       history. An absent cost in a CV-enabled record is corruption or unsupported legacy data.
       It is not a fresh run, and the difference cannot be decided from the truthiness of the
       stored value.

    B. The ladder block was read permissively. `rows` went through `int()`, so 1.9 became 1 and
       a continuation truncated every state's series to a length nobody had committed.
       `state_index` went through `int()` too, and the entries were collapsed into a dictionary
       -- so a list holding state 0 twice and omitting state 1 produced a complete-looking set
       in which state 1 silently restored zero. The block's own row count was never reconciled
       against the per-state entries, and the checkpoint's aggregate cost was never validated
       at all: only completion manifests were, which leaves continuation -- the operation that
       TRUNCATES files -- unprotected.

POSITIVE CONTROLS ARE PART OF THE CONTRACT

    Refusing everything would satisfy every rejection below. So this file also pins the cases
    that must keep working: a valid prefix restores its real counters, a fresh run initialises a
    zero prefix in memory, a genuinely CV-disabled record is not forced to carry a cost, and an
    interrupted file with an uncommitted tail beyond the committed prefix is still continuable --
    that tail is what a crash leaves behind, and requiring the whole file to match the committed
    digest would make every ordinary crash look like corruption.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from md_tools.cv import parse_cv_definition
from md_tools.cv import prefix as cv_prefix
from md_tools.cv.cost import CVCost, CVCostError, CommittedPrefix, cost_record
from md_tools.remd.cv_states import COLUMNS, PHASE, CVContinuationError

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
  - name: psi
    type: torsion
    atom_indices: [6, 8, 14, 16]
"""

N_CV = 2
ROWS = 2
INTERVAL = 5
#: Derived from the schedule, by hand: two committed rows of a two-torsion definition.
EXPECTED_OBSERVATIONS = ROWS
EXPECTED_EVALUATIONS = ROWS * N_CV


@pytest.fixture(scope="module")
def definition():
    return parse_cv_definition(CV_YAML, source="cv.yaml")


def _series(path: Path, definition, *, state=0, tau=0.0, rows=ROWS, tail=0):
    """A ladder CV series with `rows` committed rows and `tail` uncommitted ones after them."""
    header = ",".join(list(COLUMNS) + list(definition.names))
    lines = [header]
    for i in range(rows + tail):
        step = i * INTERVAL
        lines.append(",".join([
            str(step), f"{step * 0.002:.6f}", "-1", str(state), f"{tau:.6f}",
            str(state), PHASE, "",
            f"{-170.0 + i:.6f}", f"{175.0 - i:.6f}",
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "state_index": state, "tau": tau, "interval_steps": INTERVAL,
        "definition_sha256": definition.digest, "units": "degrees",
        "wrapping": "[-180, 180)", "exchange_phase": PHASE,
    }, sort_keys=True), encoding="utf-8")
    return path, sidecar


def _entry(path: Path, definition, *, rows=ROWS, sidecar=None, cost="valid"):
    """A committed-prefix entry as the checkpoint stores it."""
    scope = CVCost(observations=rows, evaluations=rows * N_CV, wall_seconds=0.002)
    record = cv_prefix.record(
        path, rows=rows, sidecar=sidecar, definition=definition,
        cost=cost_record(scope, scope, rows=rows) if cost == "valid" else None)
    if cost == "valid":
        return record
    record.pop("cost", None)
    if cost is not None:
        record["cost"] = cost
    return record


def _validate(path, entry, definition, sidecar=None):
    return cv_prefix.validate(
        path, entry, sidecar=sidecar, definition=definition,
        expect_columns=list(COLUMNS) + list(definition.names),
        value_columns=list(definition.names),
        identifiers={"state_index": 0, "tau": 0.0, "exchange_phase": PHASE},
        interval=INTERVAL, step_column="step")


# --- A. positive controls -----------------------------------------------------------------------

def test_a_valid_prefix_validates_and_restores_its_real_counters(tmp_path, definition):
    """The control. Two rows of two torsions restore 2 observations and 4 evaluations."""
    path, sidecar = _series(tmp_path / "cv_state0.csv", definition)
    entry = _entry(path, definition, sidecar=sidecar)
    assert _validate(path, entry, definition, sidecar) == ROWS

    restored = CommittedPrefix.from_record(entry)
    assert restored.rows == ROWS
    assert restored.cumulative.observations == EXPECTED_OBSERVATIONS
    assert restored.cumulative.evaluations == EXPECTED_EVALUATIONS


def test_a_a_fresh_run_may_initialise_a_zero_prefix_in_memory(definition):
    """Nothing persisted, nothing to validate: a fresh run starts from zero and that is legal."""
    fresh = CommittedPrefix()
    assert fresh.rows == 0
    assert fresh.cumulative.observations == 0
    assert fresh.cumulative.evaluations == 0


def test_a_an_uncommitted_tail_does_not_invalidate_the_committed_prefix(tmp_path, definition):
    """What a crash leaves behind. The committed rows are intact; the file is simply longer.

    Requiring the whole file to match the committed digest would make every ordinary
    interruption look like corruption, which is the opposite of what the prefix is for.
    """
    path, sidecar = _series(tmp_path / "cv_state0.csv", definition, rows=ROWS, tail=3)
    entry = _entry(path, definition, rows=ROWS, sidecar=sidecar)
    assert _validate(path, entry, definition, sidecar) == ROWS


# --- A. the defect ------------------------------------------------------------------------------

@pytest.mark.parametrize("cost, label", [
    (None, "absent"),
    ({}, "empty-mapping"),
    ([], "list"),
    ("", "empty-string"),
    (0, "zero"),
])
def test_a_a_cv_enabled_prefix_without_a_usable_cost_is_refused(tmp_path, definition, cost,
                                                                label):
    """THE defect. Each of these previously passed validation and restored 0 observations.

    `entry.get("cost") is not None` skipped the parser for the absent and null cases, and the
    falsy ones never reached it either. A CV-enabled prefix that cannot say what it cost is a
    damaged record, and reading it as a fresh run silently discards the history it was written
    to preserve.
    """
    path, sidecar = _series(tmp_path / "cv_state0.csv", definition)
    entry = _entry(path, definition, sidecar=sidecar, cost=cost)
    with pytest.raises(cv_prefix.CVPrefixError) as refusal:
        _validate(path, entry, definition, sidecar)
    assert "cost" in str(refusal.value).lower(), refusal.value


def test_a_restoring_an_absent_cost_as_zero_is_refused(tmp_path, definition):
    """The restoration half. `CommittedPrefix.from_record` must not invent a zero history."""
    path, sidecar = _series(tmp_path / "cv_state0.csv", definition)
    entry = _entry(path, definition, sidecar=sidecar, cost=None)
    with pytest.raises(CVCostError):
        CommittedPrefix.from_record(entry)


def test_a_a_cost_that_disagrees_with_the_committed_rows_is_refused(tmp_path, definition):
    """The counters are checked against the VERIFIED row count, not against themselves."""
    path, sidecar = _series(tmp_path / "cv_state0.csv", definition)
    wrong = CVCost(observations=ROWS + 1, evaluations=(ROWS + 1) * N_CV, wall_seconds=0.002)
    entry = _entry(path, definition, sidecar=sidecar,
                   cost=cost_record(wrong, wrong, rows=ROWS + 1))
    with pytest.raises(cv_prefix.CVPrefixError):
        _validate(path, entry, definition, sidecar)


# --- B. the ladder block ------------------------------------------------------------------------

def _block(tmp_path, definition, *, n_states=2, rows=ROWS, states=None, aggregate="valid"):
    """A ladder checkpoint CV block: per-state entries plus the aggregate cost beside them."""
    entries = []
    for index in range(n_states):
        path, sidecar = _series(tmp_path / f"cv_state{index}.csv", definition,
                                state=index, tau=index * 0.5, rows=rows)
        entry = _entry(path, definition, rows=rows, sidecar=sidecar)
        entry["state_index"] = index
        entry["tau"] = index * 0.5
        entries.append(entry)

    one = CVCost(observations=rows, evaluations=rows * N_CV, wall_seconds=0.002)
    total = CVCost()
    for _ in range(n_states):
        total = total.plus(one)
    block = {"rows": rows, "states": states if states is not None else entries,
             "interval_steps": INTERVAL}
    if aggregate == "valid":
        cost = cost_record(total, total, rows=rows * n_states)
        cost["aggregation"] = "sum over thermodynamic states"
        cost["per_state"] = [
            {**cost_record(one, one, rows=rows), "state_index": i} for i in range(n_states)]
        block["cost"] = cost
    elif aggregate is not None:
        block["cost"] = aggregate
    return block


def _validate_block(tmp_path, definition, block, *, n_states=2):
    from md_tools.remd.cv_states import validate_prefixes

    return validate_prefixes(tmp_path, definition, taus=[i * 0.5 for i in range(n_states)],
                             interval_steps=INTERVAL, block=block)


def test_b_a_valid_ladder_block_returns_typed_data_for_truncation_and_restoration(tmp_path,
                                                                                   definition):
    """The control -- and the shape of the answer.

    It returns validated typed data rather than a bare integer, because the count that truncates
    the files and the counters that restore the cost must be the same object the validator
    checked. Handing back an int meant the caller went back to the raw block for everything
    else, which is where the permissive re-read lived.
    """
    from md_tools.remd.cv_states import ValidatedLadderPrefix, committed_prefixes

    block = _block(tmp_path, definition)
    validated = _validate_block(tmp_path, definition, block)
    assert isinstance(validated, ValidatedLadderPrefix)
    assert validated.rows == ROWS
    assert len(validated.per_state) == 2
    for prefix in validated.per_state:
        assert prefix.rows == ROWS
        assert prefix.cumulative.observations == EXPECTED_OBSERVATIONS
        assert prefix.cumulative.evaluations == EXPECTED_EVALUATIONS

    # And the restoration consumes exactly that, never the raw block again.
    assert committed_prefixes(validated, 2) == list(validated.per_state)
    with pytest.raises(CVContinuationError):
        committed_prefixes(block, 2)


@pytest.mark.parametrize("rows, label", [
    (1.9, "fractional"), (1.0, "float"), ("2", "string"), (True, "boolean"), (-1, "negative"),
])
def test_b_a_block_row_count_that_is_not_an_integer_is_refused(tmp_path, definition, rows,
                                                               label):
    """THE defect. `int(1.9)` is 1, and the continuation truncated to a length nobody committed."""
    block = _block(tmp_path, definition)
    block["rows"] = rows
    with pytest.raises(CVContinuationError) as refusal:
        _validate_block(tmp_path, definition, block)
    assert "rows" in str(refusal.value).lower(), refusal.value


def test_b_a_block_row_count_disagreeing_with_its_entries_is_refused(tmp_path, definition):
    """A valid integer is not enough: it must be the count the per-state entries actually hold."""
    block = _block(tmp_path, definition)
    block["rows"] = 1
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


def test_b_a_duplicated_state_identity_is_refused(tmp_path, definition):
    """State 0 twice, state 1 missing. Collapsing into a dictionary made this look complete."""
    block = _block(tmp_path, definition)
    block["states"] = [block["states"][0], dict(block["states"][0])]
    with pytest.raises(CVContinuationError) as refusal:
        _validate_block(tmp_path, definition, block)
    message = str(refusal.value).lower()
    assert "state" in message and ("duplicate" in message or "once" in message), refusal.value


def test_b_a_missing_state_is_refused(tmp_path, definition):
    block = _block(tmp_path, definition)
    block["states"] = block["states"][:1]
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


@pytest.mark.parametrize("identity", [True, "0", 0.0, -1, 99])
def test_b_a_state_identity_of_the_wrong_type_or_range_is_refused(tmp_path, definition,
                                                                  identity):
    block = _block(tmp_path, definition)
    block["states"][0]["state_index"] = identity
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


def test_b_a_state_whose_tau_does_not_match_the_protocol_is_refused(tmp_path, definition):
    """Expected states come from the verified protocol, not from the record being checked."""
    block = _block(tmp_path, definition)
    block["states"][1]["tau"] = 0.25
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


def test_b_a_missing_per_state_cost_is_refused(tmp_path, definition):
    block = _block(tmp_path, definition)
    block["states"][1].pop("cost")
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


@pytest.mark.parametrize("aggregate", [None, {}, "not-a-mapping"])
def test_b_a_missing_or_malformed_aggregate_cost_is_refused(tmp_path, definition, aggregate):
    """The checkpoint's aggregate was never validated -- only completion manifests were.

    Continuation is the operation that TRUNCATES files, so leaving it unprotected is the wrong
    way round.
    """
    block = _block(tmp_path, definition, aggregate=aggregate)
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


def test_b_an_aggregate_inconsistent_with_its_per_state_entries_is_refused(tmp_path,
                                                                          definition):
    block = _block(tmp_path, definition)
    block["cost"]["cumulative"]["cv_observations"] += 1
    with pytest.raises(CVContinuationError):
        _validate_block(tmp_path, definition, block)


def test_b_committed_prefixes_does_not_silently_fill_a_missing_state(tmp_path, definition):
    """The restoration half of the duplicate defect.

    `committed_prefixes` keyed the entries by `int(state_index)`, so a duplicate overwrote and
    the missing state fell through to an empty record -- restoring zero for a state that had
    committed rows.
    """
    from md_tools.remd.cv_states import committed_prefixes

    block = _block(tmp_path, definition)
    block["states"] = [block["states"][0], dict(block["states"][0])]
    with pytest.raises((CVContinuationError, CVCostError, ValueError)):
        committed_prefixes(block, 2)
