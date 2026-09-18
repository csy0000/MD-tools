"""Selective REST2 through `build-top --rest2-scaler` (`md_tools.build.scaler.build_scaled_states`).

The two mode tests of the S1 assignment, on files, through the function the command calls:

  * NO selector: the saved states are byte-identical to what the 0.6.0 scaler integrated;
  * ANY selector: explicit mode, recorded as such, and an omitted category means none.

Plus: every refusal happens before a file exists, `--check` creates nothing, and the states a
record describes are rebuilt from the record alone.

PLATFORM_POLICY_EXEMPTION: System construction and serialisation only. No Context is created.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

openmm = pytest.importorskip("openmm")
from openmm import XmlSerializer                                               # noqa: E402

from md_tools.build.scaler import build_scaled_states                          # noqa: E402
from md_tools.build.strict import ConfigError                                  # noqa: E402

from .selective_rest2_fixture import RESIDUE, build_fixture                     # noqa: E402

FROZEN = Path(__file__).resolve().parent / "data" / "selective_rest2" / "hamiltonian_0_6_0.py"


@pytest.fixture(scope="module")
def fx():
    return build_fixture()


def _build(tmp_path, fx, text, **kwargs):
    build = fx.write(tmp_path / "build")
    config = build / "scaler.config"
    config.write_text(text, encoding="utf-8")
    record = build_scaled_states(system_path=build / "built.xml",
                                 topology_path=build / "built.pdb", config_path=config,
                                 echo=False, **kwargs)
    return build, record


def _frozen():
    spec = importlib.util.spec_from_file_location("hamiltonian_0_6_0", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_selector_writes_the_0_6_0_states_byte_for_byte(tmp_path, fx):
    build, record = _build(tmp_path, fx, "method: REST2\nschedule:\n  n_states: 3\n")
    assert record["selection"]["selection_mode"] == "legacy-full-solute"
    assert record["scaler_arguments"]["torsion_central_bonds"] is None
    frozen = _frozen()
    base = XmlSerializer.deserialize((build / "built.xml").read_text(encoding="utf-8"))
    solute = record["solute"]["atom_indices"]
    excluded = record["unscaled_torsions"]["unscaled_central_bonds"]
    for state in record["states"]:
        then = frozen.build_scaled_system(base, solute, state["tau"], excluded_bonds=excluded)
        assert (build / "REST2" / state["file"]).read_text(encoding="utf-8") == \
            XmlSerializer.serialize(then), state["file"]
    assert "legacy full solute" in (build / "REST2" / "scaler.log").read_text(encoding="utf-8")


def test_any_selector_is_explicit_and_rebuilds_from_the_record_alone(tmp_path, fx):
    from md_tools.rest2.hamiltonian import build_scaled_system

    text = ("method: REST2\nschedule:\n  n_states: 3\n"
            f"backbone_scaling_list: \":{RESIDUE['A.SER']}\"\n"
            "ligand_scaling_dict:\n  L01:\n    mask: \":13\"\n    torsion_exclusions: auto\n")
    build, record = _build(tmp_path, fx, text)
    selection = record["selection"]
    assert selection["selection_mode"] == "explicit"
    assert selection["masks"]["sidechain"] is None, "omitted"
    hot = set(selection["selected_nonbonded_atoms"])
    ser = fx.residue(RESIDUE["A.SER"])
    assert fx.atom(RESIDUE["A.SER"], "OG") not in hot, "an omitted category means none"
    assert {a.index for a in fx.residue(RESIDUE["LGA#2"]).atoms()}.isdisjoint(hot)
    assert {a.index for a in ser.atoms() if a.name in ("N", "CA", "C", "O")} <= hot
    assert record["solute"]["n_atoms"] > len(hot), "`solute` stays the whole solute"
    # The states are exactly build_scaled_system(built, **scaler_arguments, tau).
    base = XmlSerializer.deserialize((build / "built.xml").read_text(encoding="utf-8"))
    arguments = dict(record["scaler_arguments"])
    for state in record["states"]:
        rebuilt = build_scaled_system(base, arguments["solute_indices"], state["tau"],
                                      excluded_bonds=arguments["excluded_bonds"],
                                      unscaled_impropers=arguments["unscaled_impropers"],
                                      torsion_central_bonds=arguments["torsion_central_bonds"],
                                      cmap_terms=arguments["cmap_terms"])
        assert (build / "REST2" / state["file"]).read_text(encoding="utf-8") == \
            XmlSerializer.serialize(rebuilt)
    on_disk = yaml.safe_load((build / "REST2" / "scaler.yaml").read_text(encoding="utf-8"))
    assert on_disk["selection_sha256"] == record["selection_sha256"]
    log = (build / "REST2" / "scaler.log").read_text(encoding="utf-8")
    assert "EXPLICIT" in log and ":3" in log and "instance L01" in log, "the residue map is printed"
    # tau 0 is the unscaled System
    assert (build / "REST2" / "system_state0.xml").read_text(encoding="utf-8") == \
        XmlSerializer.serialize(base)


@pytest.mark.parametrize("text, words", [
    ('method: REST2\nbackbone_scaling_list: ":2@CA"\n', "atom selector"),
    ('method: REST2\nbackbone_scaling_list: ":999999"\n', "out of range"),
    ("method: REST2\nligand_scaling_dict: {}\n", "resolves to NOTHING"),
    ('method: REST2\nunscaled_torsions: false\nbackbone_scaling_list: ":2"\n', "improper"),
    ("method: REST2\nbackbone_scaling_list: 45\n", None),
])
def test_a_refused_selection_leaves_nothing_behind(tmp_path, fx, text, words):
    build = fx.write(tmp_path / "build")
    before = sorted(p.name for p in build.iterdir())
    (build / "scaler.config").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as refusal:
        build_scaled_states(system_path=build / "built.xml", topology_path=build / "built.pdb",
                            config_path=build / "scaler.config", echo=False)
    if words:
        assert words in str(refusal.value)
    assert sorted(p.name for p in build.iterdir()) == sorted(before + ["scaler.config"])


def test_check_creates_nothing_and_reports_the_selection(tmp_path, fx):
    build = fx.write(tmp_path / "build")
    (build / "scaler.config").write_text(
        'method: REST2\nsidechain_scaling_list: ":3-4"\n', encoding="utf-8")
    before = sorted(p.name for p in build.iterdir())
    record = build_scaled_states(system_path=build / "built.xml",
                                 topology_path=build / "built.pdb",
                                 config_path=build / "scaler.config", check=True, echo=False)
    assert record["selection"]["selection_mode"] == "explicit"
    assert sorted(p.name for p in build.iterdir()) == before


def test_only_the_selected_copy_needs_bond_orders(tmp_path, fx):
    """An unselected ligand's SDF is not required: only the region's residues are classified."""
    build = fx.write(tmp_path / "build", sdfs=False)
    from rdkit import Chem

    Chem.MolToMolFile(fx.molecules["LGA"], str(build / "LGA.sdf"))
    (build / "scaler.config").write_text(
        'method: REST2\nschedule:\n  n_states: 2\nligand_scaling_dict:\n  L:\n    mask: ":15"\n',
        encoding="utf-8")
    record = build_scaled_states(system_path=build / "built.xml",
                                 topology_path=build / "built.pdb",
                                 config_path=build / "scaler.config", echo=False)
    assert list(record["unscaled_torsions"]["residue_sdfs"]) == ["LGA"]
