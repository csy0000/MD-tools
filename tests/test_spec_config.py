"""The canonical configuration model: formats, units, profiles, precedence, hashes, migration.

The property the whole design rests on is that **file syntax carries no meaning**. Everything below
is a way of checking that: the same configuration written differently must produce the same
canonical bytes and the same hashes, and anything the model cannot account for must fail loudly
rather than be dropped.
"""
from __future__ import annotations

import json

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("pydantic")

from md_templates.openmm.spec import canonical, diffs, migrate, resolve  # noqa: E402
from md_templates.openmm.spec.models import SimulationSpec  # noqa: E402
from md_templates.openmm.spec.units import UnitError, parse_quantity  # noqa: E402

MINIMAL = {
    "system": {"system_id": "demo", "route": "smiles", "smiles": "CCO"},
    "protocol": {"production": {"method": "md", "n_chunks": 3, "chunk": "10 ps"}},
}


def resolved(doc=None, **kw):
    return resolve.resolve_spec(json.loads(json.dumps(doc or MINIMAL)), **kw)


# ---------------------------------------------------------------------------------------------
# 1-2: format equivalence and deterministic canonical form
# ---------------------------------------------------------------------------------------------

def test_yaml_and_json_produce_identical_canonical_bytes_and_hashes(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(MINIMAL))
    (tmp_path / "a.json").write_text(json.dumps(MINIMAL))
    from_yaml = resolve.resolve_spec(resolve.load_document(tmp_path / "a.yaml"))
    from_json = resolve.resolve_spec(resolve.load_document(tmp_path / "a.json"))
    assert canonical.canonical_json(canonical.dump_model(from_yaml["spec"])) == \
        canonical.canonical_json(canonical.dump_model(from_json["spec"]))
    assert from_yaml["hashes"] == from_json["hashes"]


def test_canonical_serialisation_is_deterministic_and_order_independent():
    a = resolved()["spec"]
    reordered = {"protocol": MINIMAL["protocol"], "system": MINIMAL["system"]}
    b = resolved(reordered)["spec"]
    assert canonical.canonical_json(canonical.dump_model(a)) == \
        canonical.canonical_json(canonical.dump_model(b))
    assert canonical.sha256_of(canonical.dump_model(a)) == canonical.sha256_of(canonical.dump_model(b))


def test_quantities_serialise_with_their_unit_not_as_a_bare_tuple():
    """A canonical document must say what a number means."""
    data = canonical.to_plain(canonical.dump_model(resolved()["spec"]))
    ts = data["protocol"]["integrator"]["timestep"]
    assert ts == {"value": 0.004, "unit": "ps"}


# ---------------------------------------------------------------------------------------------
# 3-4: unknown keys and units
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,payload", [
    ("top level", {"nonsense": 1}),
    ("system", {"system": {"nonsense": 1}}),
    ("protocol.production", {"protocol": {"production": {"nonsense": 1}}}),
    ("build.solvation", {"build": {"solvation": {"nonsense": 1}}}),
])
def test_unknown_keys_are_rejected_at_every_level(path, payload):
    doc = json.loads(json.dumps(MINIMAL))

    def deep_update(target, overlay):
        for k, v in overlay.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                deep_update(target[k], v)
            else:
                target[k] = v

    deep_update(doc, payload)
    with pytest.raises(Exception) as excinfo:
        resolved(doc)
    assert "nonsense" in str(excinfo.value), f"{path}: the unknown key was not named"


@pytest.mark.parametrize("raw,dimension", [
    ("2", "time"),                 # unitless where a unit is required
    ("2 K", "time"),               # wrong dimension
    ("2 furlongs", "length"),      # unknown unit
    ("banana", "time"),            # not a quantity
])
def test_invalid_and_dimensionally_wrong_quantities_are_refused(raw, dimension):
    with pytest.raises(UnitError):
        parse_quantity(raw, dimension=dimension, field="f")


def test_equivalent_units_normalise_to_the_same_canonical_value():
    assert parse_quantity("4 fs", dimension="time").value == \
        pytest.approx(parse_quantity("0.004 ps", dimension="time").value)
    assert parse_quantity("2 angstrom", dimension="length").value == pytest.approx(0.2)


def test_a_unitless_number_never_silently_acquires_a_unit():
    """`timestep: 2` meant fs in the old manifests and ps in OpenMM's own system."""
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["integrator"] = {"timestep": 2}
    with pytest.raises(Exception, match="no unit"):
        resolved(doc)


# ---------------------------------------------------------------------------------------------
# 5-6: cross-field and method-specific validation
# ---------------------------------------------------------------------------------------------

def test_chunk_must_be_a_whole_number_of_exchange_intervals():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"] = {"method": "rest2", "n_chunks": 1, "chunk": "1 ps",
                                     "scale_factors": [1.0, 0.5], "exchange_interval": "0.3 ps",
                                     "relaxation": "1 ps"}
    with pytest.raises(Exception, match="whole number of exchange intervals"):
        resolved(doc)


def test_chunk_must_be_a_whole_number_of_timesteps():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"]["chunk"] = "0.005 ps"      # not a multiple of 4 fs
    with pytest.raises(Exception, match="whole number of"):
        resolved(doc)


def test_md_rejects_rest2_only_settings():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"]["exchange_interval"] = "1 ps"
    with pytest.raises(Exception, match="exchange_interval"):
        resolved(doc)


def test_rest2_rejects_md_only_settings():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"] = {"method": "rest2", "n_chunks": 1, "chunk": "1 ps",
                                     "scale_factors": [1.0, 0.5], "exchange_interval": "0.5 ps",
                                     "relaxation": "1 ps", "scale_factor": 0.5}
    with pytest.raises(Exception, match="scale_factor"):
        resolved(doc)


def test_a_descending_ladder_starting_at_one_is_required():
    doc = json.loads(json.dumps(MINIMAL))
    base = {"method": "rest2", "n_chunks": 1, "chunk": "1 ps",
            "exchange_interval": "0.5 ps", "relaxation": "1 ps"}
    doc["protocol"]["production"] = dict(base, scale_factors=[0.9, 0.5])
    with pytest.raises(Exception, match="cold rung"):
        resolved(doc)
    doc["protocol"]["production"] = dict(base, scale_factors=[1.0, 0.5, 0.7])
    with pytest.raises(Exception, match="descending"):
        resolved(doc)


def test_route_and_force_field_must_agree():
    doc = json.loads(json.dumps(MINIMAL))
    doc["build"] = {"forcefield": {"protein": "amber19/protein.ff19SB.xml", "water": "x",
                                   "small_molecule": None, "charge_method": None}}
    with pytest.raises(Exception, match="smiles"):
        resolved(doc)


# ---------------------------------------------------------------------------------------------
# 7-11: profiles, precedence, attribution
# ---------------------------------------------------------------------------------------------

def test_default_profile_is_selected_from_the_declared_route_and_method():
    assert resolved()["profile"]["profile_id"] == "explicit-md-ligand-v1"


def test_the_smoke_profile_is_never_selected_as_a_default():
    """It is picoseconds of unvalidated settings; reaching it must be deliberate."""
    for route, method in (("smiles", "rest2"), ("smiles", "md")):
        chosen = resolve.select_profile(route, method)["profile_id"]
        assert chosen != "cpu-smoke-v1"
    assert resolve.load_profile("cpu-smoke-v1")["profile_id"] == "cpu-smoke-v1"


def test_an_explicitly_pinned_profile_is_used():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"] = {"method": "rest2", "n_chunks": 2, "chunk": "0.001 ns",
                                     "scale_factors": [1.0, 0.5625, 0.25],
                                     "exchange_interval": "0.5 ps", "relaxation": "1 ps"}
    doc["profile"] = "cpu-smoke-v1"
    assert resolved(doc)["profile"]["profile_id"] == "cpu-smoke-v1"


def test_a_profile_for_the_wrong_method_is_refused():
    doc = json.loads(json.dumps(MINIMAL))
    doc["profile"] = "explicit-rest2-ligand-v1"          # document declares md
    with pytest.raises(resolve.ResolutionError, match="method"):
        resolved(doc)


def test_the_user_document_wins_over_the_profile():
    doc = json.loads(json.dumps(MINIMAL))
    doc["build"] = {"nonbonded": {"cutoff": "1.2 nm"}}
    r = resolved(doc)
    assert r["spec"].build.nonbonded.cutoff.value == pytest.approx(1.2)
    assert r["sources"]["build.nonbonded.cutoff"] == "document"


def test_a_cli_override_wins_over_the_document():
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"]["n_chunks"] = 5
    r = resolved(doc, overrides=["protocol.production.n_chunks=11"])
    assert r["spec"].protocol.production.n_chunks == 11
    assert r["sources"]["protocol.production.n_chunks"] == "cli"


def test_every_resolved_field_reports_its_source():
    r = resolved()
    assert r["sources"]["build.nonbonded.cutoff"].startswith("profile:")
    assert r["sources"]["system.route"] == "document"
    unattributed = [k for k, v in r["sources"].items() if not v]
    assert not unattributed


def test_the_route_is_never_inferred():
    doc = {"system": {"system_id": "demo", "smiles": "CCO"},
           "protocol": {"production": {"method": "md", "n_chunks": 1, "chunk": "10 ps"}}}
    with pytest.raises(resolve.ResolutionError, match="system.route is required"):
        resolved(doc)


def test_the_method_is_never_inferred():
    doc = {"system": {"system_id": "demo", "route": "smiles", "smiles": "CCO"},
           "protocol": {"production": {"n_chunks": 1, "chunk": "10 ps"}}}
    with pytest.raises(resolve.ResolutionError, match="method is required"):
        resolved(doc)


# ---------------------------------------------------------------------------------------------
# 12-13: profile versioning and semantic diff
# ---------------------------------------------------------------------------------------------

def test_every_profile_is_valid_and_carries_an_id_version_and_hash():
    for doc in resolve.list_profiles():
        assert doc["profile_id"] and isinstance(doc["profile_schema_version"], int)
        assert doc["description"]
        assert canonical.sha256_of({k: v for k, v in doc.items() if k != "_path"})


def test_changing_a_profile_value_changes_its_hash():
    """A scientific change to a profile cannot reuse its identity silently."""
    doc = resolve.load_profile("explicit-md-ligand-v1")
    before = canonical.sha256_of(doc)
    doc["defaults"]["build"]["nonbonded"]["cutoff"] = "1.2 nm"
    assert canonical.sha256_of(doc) != before


@pytest.mark.parametrize("field,value,expected", [
    ("build.solvation.padding", "1.5 nm", "bundle-defining"),
    ("protocol.integrator.timestep", "2 fs", "continuity-defining"),
    ("protocol.production.n_chunks", 99, "extension-only"),
    ("execution.platform", "CUDA", "execution-only"),
])
def test_diff_classifies_each_change_by_its_consequence(field, value, expected):
    a = resolved()["spec"]
    doc = json.loads(json.dumps(MINIMAL))
    node = doc
    parts = field.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value
    b = resolved(doc)["spec"]
    rows = diffs.diff_specs(a, b)
    assert rows, f"{field} produced no diff"
    assert any(r["consequence"] == expected for r in rows), rows


# ---------------------------------------------------------------------------------------------
# 14: migration from the shipped manifests
# ---------------------------------------------------------------------------------------------

def test_migration_of_the_shipped_manifests_validates_and_reports_its_changes():
    from md_templates.openmm.schemas import (load_experiment, load_system, shipped_experiment,
                                             shipped_system)

    system = load_system(shipped_system("cyclo_rgdfv"), check_chemistry=False)
    experiment = load_experiment(shipped_experiment("rgd_rest2_10rung"))
    document, notes = migrate.migrate_manifests(system.doc, experiment.doc)
    spec = resolve.resolve_spec(document)["spec"]
    assert spec.system.route == "smiles"
    assert spec.method == "rest2"
    # the physical value is preserved, now carrying its unit explicitly
    assert spec.protocol.integrator.timestep.value == pytest.approx(0.004)
    assert any("timestep_fs" in n for n in notes)
    assert any("ladder_status" in n for n in notes), "a dropped claim must be reported"


def test_migration_refuses_a_retired_total_field():
    with pytest.raises(migrate.MigrationError, match="total_ns_per_replica"):
        migrate.migrate_manifests(
            {"input": {"route": "smiles", "smiles": "C"}, "system_id": "x"},
            {"rest2": {"total_ns_per_replica": 2.0, "chunk_ns": 1.0}},
        )


def test_migration_refuses_a_document_with_no_declared_method():
    with pytest.raises(migrate.MigrationError, match="method"):
        migrate.migrate_manifests({"input": {"route": "smiles", "smiles": "C"}, "system_id": "x"},
                                  {"integrator": {"timestep_fs": 4.0}})


# ---------------------------------------------------------------------------------------------
# hashes mean what they claim
# ---------------------------------------------------------------------------------------------

def test_execution_choices_do_not_change_the_scientific_hashes():
    a = resolved()
    doc = json.loads(json.dumps(MINIMAL))
    doc["execution"] = {"platform": "CUDA", "device": "0"}
    b = resolved(doc)
    assert a["hashes"]["system_build_sha256"] == b["hashes"]["system_build_sha256"]
    assert a["hashes"]["protocol_sha256"] == b["hashes"]["protocol_sha256"]
    assert a["hashes"]["execution_sha256"] != b["hashes"]["execution_sha256"]


def test_more_chunks_does_not_change_the_continuity_hash():
    """Extension must not look like a different run, or resuming becomes impossible."""
    a = resolved()
    doc = json.loads(json.dumps(MINIMAL))
    doc["protocol"]["production"]["n_chunks"] = 99
    assert resolved(doc)["hashes"]["protocol_sha256"] == a["hashes"]["protocol_sha256"]


def test_a_build_change_changes_only_the_build_hash():
    a = resolved()
    doc = json.loads(json.dumps(MINIMAL))
    doc["build"] = {"solvation": {"padding": "1.5 nm"}}
    b = resolved(doc)
    assert b["hashes"]["system_build_sha256"] != a["hashes"]["system_build_sha256"]
    assert b["hashes"]["protocol_sha256"] == a["hashes"]["protocol_sha256"]


def test_the_model_exposes_machine_readable_json_schema():
    schema = SimulationSpec.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"system", "build", "protocol", "execution", "randomness"}
