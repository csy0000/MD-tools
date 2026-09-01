"""Contract v2: continuing a simulation without writing into finished data.

Ported from MD-data `protect-md-project-dev-test@20d982eb` and adapted to the v2 year-first layout.
Every rule the migration instruction lists is asserted here, and each one exists because the
alternative is a dataset whose provenance says something that is not true.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from md_tools.data_contract import (Dataset, DatasetError, Extension, canonical_path,
                                    check_dataset_extension, validate_dataset,
                                    validate_extension, validate_extension_file)

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
SHA = "a" * 64
COMMIT = "b" * 40


def _dataset(**patch):
    document = {
        "schema_version": "2.0", "dataset_id": "ala-rest2-ext", "path": "2026/ALA/ALA-REST2-ext",
        "year": "2026", "project_name": "ALA", "data_name": "ALA-REST2-ext", "role": "project",
        "system": "Alanine dipeptide, explicit TIP3P, REST2 continuation.",
        "created_at": T0.isoformat(), "completed_at": (T0 + timedelta(hours=2)).isoformat(),
        "created_by": {"person_id": "shu-yu-chen", "name": "Shu-Yu Chen"},
        "status": "complete",
        "origin": {"repository": "https://github.com/csy0000/MD-project", "commit": COMMIT},
        "derived_from": ["ala-rest2"],
        "components": [
            {"name": "REST2_ext1", "type": "simulation", "method": "REST2", "path": "REST2_ext1",
             "status": "complete", "linked": False},
            {"name": "REST2", "type": "simulation", "method": "REST2", "path": "REST2",
             "status": "complete", "linked": True},
        ],
    }
    document.update(patch)
    return validate_dataset(document)


def _extension(**patch):
    document = {
        "schema_version": "2.0", "mode": "new-dataset", "dataset_id": "ala-rest2-ext",
        "target": {"dataset_id": "ala-rest2", "component": "REST2",
                   "source_checkpoint": "REST2/rest2_checkpoint.nc",
                   "source_checkpoint_sha256": SHA, "source_checkpoint_copied": True,
                   "original_length_ns": 5.0},
        "output_component": "REST2_ext1", "additional_length_ns": 5.0,
        "combined_length_ns": 10.0, "restart_step": 2_500_000, "restart_time_ps": 5000.0,
        "reason": "The first 5 ns did not reach the transition often enough to estimate a rate.",
        "requested_by": {"person_id": "shu-yu-chen", "name": "Shu-Yu Chen"},
        "requested_at": T0.isoformat(), "status": "complete",
        "started_at": (T0 + timedelta(minutes=5)).isoformat(),
        "completed_at": (T0 + timedelta(hours=1)).isoformat(),
        "final_length_ns": 10.0,
        "parent_links": ["REST2"],
        "generation": {"repository": "https://github.com/csy0000/MD-tools", "commit": COMMIT},
    }
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(document.get(key), dict):
            document[key] = {**document[key], **value}
        else:
            document[key] = value
    return document


# --- the record validates -----------------------------------------------------------------------

def test_a_complete_new_dataset_extension_validates():
    record = validate_extension(_extension())
    assert record.mode == "new-dataset"
    assert record.combined_length_ns == 10.0
    assert record.target.source_checkpoint_sha256 == SHA


def test_the_new_dataset_lives_under_the_v2_year_first_layout():
    dataset = _dataset()
    assert dataset.path == canonical_path(year="2026", project_name="ALA",
                                          data_name="ALA-REST2-ext")
    assert "-" not in dataset.path.split("/")[0], "no month segment"


def test_a_v1_extension_is_refused_on_its_schema_version():
    with pytest.raises(DatasetError, match="schema_version"):
        validate_extension(_extension(schema_version="1.0"))


# --- immutability of the parent ------------------------------------------------------------------

def test_extending_a_complete_dataset_uses_a_new_dataset_with_a_new_id():
    record = validate_extension(_extension())
    assert record.dataset_id != record.target.dataset_id
    assert record.target.dataset_id == "ala-rest2"


def test_a_new_dataset_extension_pointing_at_itself_is_refused():
    with pytest.raises(DatasetError, match="must be a different dataset"):
        validate_extension(_extension(target={"dataset_id": "ala-rest2-ext"}))


def test_the_parent_must_be_recorded_in_derived_from():
    """Otherwise the manifest does not say where the data came from."""
    dataset = _dataset(derived_from=[])
    problems = check_dataset_extension(dataset, validate_extension(_extension()))
    assert any("derived_from" in p for p in problems), problems


def test_an_in_flight_in_place_extension_in_a_complete_dataset_is_refused():
    """That would assert output is still being written into data declared immutable."""
    record = validate_extension(_extension(
        mode="in-place", target={"dataset_id": "ala-rest2-ext"},
        combined_length_ns=None, restart_step=None, restart_time_ps=None,
        status="running", completed_at=None, final_length_ns=None))
    problems = check_dataset_extension(_dataset(), record)
    assert any("immutable" in p for p in problems), problems


def test_a_terminal_in_place_record_survives_the_dataset_completing():
    """It is the provenance of how the dataset reached its final length, not a live claim."""
    record = validate_extension(_extension(
        mode="in-place", target={"dataset_id": "ala-rest2-ext"},
        combined_length_ns=None, restart_step=None, restart_time_ps=None))
    problems = check_dataset_extension(_dataset(), record)
    assert not any("immutable" in p for p in problems), problems


# --- the output component ------------------------------------------------------------------------

def test_new_output_may_not_go_into_a_linked_parent_component():
    record = validate_extension(_extension(output_component="REST2", parent_links=[]))
    problems = check_dataset_extension(_dataset(), record)
    assert any("linked" in p for p in problems), problems


def test_the_output_component_may_not_be_listed_as_a_parent_link():
    with pytest.raises(DatasetError, match="parent_links"):
        validate_extension(_extension(output_component="REST2", parent_links=["REST2"]))


def test_a_parent_link_must_actually_be_linked_in_the_manifest():
    dataset = _dataset(components=[
        {"name": "REST2_ext1", "type": "simulation", "method": "REST2", "path": "REST2_ext1",
         "status": "complete", "linked": False},
        {"name": "REST2", "type": "simulation", "method": "REST2", "path": "REST2",
         "status": "complete", "linked": False},
    ])
    problems = check_dataset_extension(dataset, validate_extension(_extension()))
    assert any("not marked 'linked: true'" in p for p in problems), problems


def test_an_output_component_absent_from_the_manifest_is_refused():
    problems = check_dataset_extension(_dataset(),
                                       validate_extension(_extension(output_component="ghost")))
    assert any("not a component" in p for p in problems), problems


# --- the checkpoint ------------------------------------------------------------------------------

def test_the_source_checkpoint_hash_is_required():
    document = _extension()
    document["target"].pop("source_checkpoint_sha256")
    with pytest.raises(DatasetError, match="source_checkpoint_sha256"):
        validate_extension(document)


def test_an_in_place_extension_must_have_copied_the_checkpoint():
    """It writes into the component it restarted from, so the run can overwrite that file."""
    with pytest.raises(DatasetError, match="source_checkpoint_copied"):
        validate_extension(_extension(
            mode="in-place", target={"dataset_id": "ala-rest2-ext",
                                     "source_checkpoint_copied": False},
            combined_length_ns=None, restart_step=None, restart_time_ps=None))


def test_the_source_checkpoint_path_must_be_relative():
    with pytest.raises(DatasetError, match="absolute"):
        validate_extension(_extension(
            target={"source_checkpoint": "/scratch/run/rest2_checkpoint.nc"}))


# --- the lengths ----------------------------------------------------------------------------------

def test_the_lengths_must_be_mutually_consistent():
    with pytest.raises(DatasetError, match="does not equal"):
        validate_extension(_extension(combined_length_ns=99.0))


def test_a_new_dataset_extension_requires_a_restart_step():
    with pytest.raises(DatasetError, match="restart_step"):
        validate_extension(_extension(restart_step=None))


def test_an_in_place_extension_records_no_combined_length():
    with pytest.raises(DatasetError, match="only a 'new-dataset' extension records it"):
        validate_extension(_extension(mode="in-place",
                                      target={"dataset_id": "ala-rest2-ext"}))


def test_an_extension_cannot_produce_more_than_it_asked_for():
    with pytest.raises(DatasetError, match="exceeds the requested"):
        validate_extension(_extension(final_length_ns=50.0))


# --- status and timestamps -------------------------------------------------------------------------

@pytest.mark.parametrize("status, patch, expected", [
    ("requested", {"started_at": T0.isoformat()}, "must be absent while status is 'requested'"),
    ("running", {"started_at": None}, "required when status is 'running'"),
    ("running", {"completed_at": T0.isoformat()}, "must be absent while status is 'running'"),
    ("complete", {"final_length_ns": None}, "required when status is 'complete'"),
])
def test_the_status_and_its_timestamps_must_agree(status, patch, expected):
    document = _extension(status=status, **patch)
    if status == "requested":
        document.update(completed_at=None, final_length_ns=None)
        document.update(patch)
    with pytest.raises(DatasetError, match=expected):
        validate_extension(document)


def test_an_attempt_cannot_stop_without_having_started():
    with pytest.raises(DatasetError, match="required when completed_at records"):
        validate_extension(_extension(status="failed", started_at=None,
                                      final_length_ns=None))


def test_timestamps_must_be_ordered():
    with pytest.raises(DatasetError, match="must not precede requested_at"):
        validate_extension(_extension(started_at=(T0 - timedelta(hours=1)).isoformat()))
    with pytest.raises(DatasetError, match="must not precede started_at"):
        validate_extension(_extension(completed_at=(T0 - timedelta(hours=1)).isoformat(),
                                      started_at=T0.isoformat()))


def test_an_extension_cannot_finish_after_the_dataset_containing_it_was_completed():
    record = validate_extension(_extension(
        completed_at=(T0 + timedelta(hours=5)).isoformat()))
    problems = check_dataset_extension(_dataset(), record)
    assert any("after the dataset was declared complete" in p for p in problems), problems


# --- the record on disk ---------------------------------------------------------------------------

def test_an_extension_file_round_trips(tmp_path):
    (tmp_path / "extension.yaml").write_text(yaml.safe_dump(_extension()), encoding="utf-8")
    record = validate_extension_file(tmp_path)
    assert record.status == "complete"


def test_the_exported_extension_schema_matches_the_model():
    from md_tools.data_contract import exported_schema_path
    from md_tools.data_contract.extension import json_schema

    path = exported_schema_path("extension")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == json_schema(), (
        "the exported extension schema differs from what the model renders; regenerate it")


def test_both_schemas_are_exported_and_distinct():
    from md_tools.data_contract import exported_schema_path

    dataset = json.loads(exported_schema_path("dataset").read_text())
    extension = json.loads(exported_schema_path("extension").read_text())
    assert dataset["$id"] != extension["$id"]
    assert "2.0" in dataset["$id"] and "2.0" in extension["$id"]
