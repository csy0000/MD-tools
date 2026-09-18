"""PROPKA3 protonation: prediction, the deterministic variant rule, overrides and histidine warnings.

Plan section 4 (docs/history/claudecode-instructions/20260917_next-release-reusable-ligands-propka-cuda-ais.md).
The rule tests feed `assign_variants` stated predictions, so each branch is checked against a pKa
chosen to land on it rather than against whatever PROPKA happens to return for a fixture. One test
runs the real PROPKA on a small tleap peptide with a zinc ion placed at a histidine, end to end.

PLATFORM_POLICY_EXEMPTION: hydrogen placement relaxes on the Reference platform by design (seeded,
bit-reproducible). No dynamics.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from md_tools.openmm.protonation import (Prediction, ProtonationError, assign_variants,
                                         histidine_proximity, parse_overrides, protonate_structure,
                                         residue_key)

SEQUENCE = "ACE HIE ASP LYS GLU CYS TYR NME"


@pytest.fixture(scope="module")
def peptide(tmp_path_factory):
    """ACE-HIE-ASP-LYS-GLU-CYS-TYR-NME from tleap, with its hydrogens."""
    if shutil.which("tleap") is None:
        pytest.skip("tleap (AmberTools) is not on PATH")
    work = tmp_path_factory.mktemp("peptide")
    (work / "leap.in").write_text(
        f"source leaprc.protein.ff14SB\nm = sequence {{ {SEQUENCE} }}\nsavepdb m pep.pdb\nquit\n",
        encoding="utf-8")
    done = subprocess.run(["tleap", "-f", "leap.in"], cwd=work, capture_output=True, text=True,
                          timeout=120)
    assert (work / "pep.pdb").is_file(), done.stdout + done.stderr
    return work / "pep.pdb"


def _topology(pdb):
    from openmm import app

    structure = app.PDBFile(str(pdb))
    return structure.topology, structure.positions


def _keys(topology):
    return {r.name: residue_key(r) for r in topology.residues()}


def _by_residue(assignments):
    return {a.residue: a for a in assignments}


# --- the rule -------------------------------------------------------------------------------------

def test_each_family_follows_the_documented_rule(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    predictions = [Prediction("HIS", k["HIS"], 7.6, 6.5, False),
                   Prediction("ASP", k["ASP"], 7.4, 3.8, False),
                   Prediction("LYS", k["LYS"], 6.2, 10.5, True),
                   Prediction("GLU", k["GLU"], 3.1, 4.5, False)]
    assignments, unsupported = assign_variants(topology, method="propka", ph=7.0,
                                               predictions=predictions)
    got = _by_residue(assignments)
    assert got["HIS"].variant == "HIP" and got["HIS"].source == "propka"
    assert got["ASP"].variant == "ASH"
    assert got["LYS"].variant == "LYN" and got["LYS"].coupled
    assert got["GLU"].variant == "GLU"
    assert got["HIS"].near_ph and got["ASP"].near_ph and not got["GLU"].near_ph
    assert unsupported == []


def test_a_neutral_histidine_is_left_to_the_hydrogen_bond_heuristic(peptide):
    """PROPKA does not resolve HID versus HIE, so the rule must not pretend to."""
    topology, _ = _topology(peptide)
    k = _keys(topology)
    assignments, _ = assign_variants(topology, method="propka", ph=7.0,
                                     predictions=[Prediction("HIS", k["HIS"], 5.0, 6.5, False)])
    his = _by_residue(assignments)["HIS"]
    assert his.variant is None
    assert "heuristic" in his.source


def test_an_unsupported_prediction_is_reported_not_converted(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    assignments, unsupported = assign_variants(
        topology, method="propka", ph=7.0,
        predictions=[Prediction("CYS", k["CYS"], 6.1, 9.0, False),
                     Prediction("TYR", k["TYR"], 6.5, 10.0, False)])
    assert _by_residue(assignments)["CYS"].variant is None, "CYM is not built; CYS is kept"
    groups = {entry["group"]: entry for entry in unsupported}
    assert groups["CYS"]["predicted"].startswith("deprotonated cysteine")
    assert groups["TYR"]["kept"] == "standard state"


def test_a_residue_without_a_prediction_uses_the_model_pka_and_says_so(peptide):
    topology, _ = _topology(peptide)
    assignments, _ = assign_variants(topology, method="propka", ph=7.0, predictions=[])
    asp = _by_residue(assignments)["ASP"]
    assert asp.variant == "ASP" and asp.source == "model-pka"
    assert any("no prediction" in note for note in asp.notes)


def test_the_openmm_method_assigns_nothing_but_overrides(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    overrides = parse_overrides([{"select": {"chain": k["HIS"][0], "resid": k["HIS"][1]},
                                  "variant": "HID"}])
    assignments, _ = assign_variants(topology, method="openmm", ph=7.0, overrides=overrides)
    got = _by_residue(assignments)
    assert got["HIS"].variant == "HID" and got["HIS"].source == "override"
    assert got["ASP"].variant is None and got["ASP"].source.startswith("openmm addHydrogens")


# --- overrides ------------------------------------------------------------------------------------

def test_an_override_superseding_a_prediction_is_noted(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    overrides = parse_overrides([{"select": {"chain": k["HIS"][0], "resid": k["HIS"][1]},
                                  "variant": "HIE"}])
    assignments, _ = assign_variants(topology, method="propka", ph=7.0, overrides=overrides,
                                     predictions=[Prediction("HIS", k["HIS"], 7.9, 6.5, False)])
    his = _by_residue(assignments)["HIS"]
    assert his.variant == "HIE"
    assert any("supersedes the prediction" in note for note in his.notes)


def test_an_override_for_a_residue_that_is_not_there_is_refused(peptide):
    topology, _ = _topology(peptide)
    overrides = parse_overrides([{"select": {"chain": "Z", "resid": "999"}, "variant": "HID"}])
    with pytest.raises(ProtonationError, match="Z:999"):
        assign_variants(topology, method="propka", ph=7.0, overrides=overrides)


def test_an_override_from_another_family_is_refused(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    overrides = parse_overrides([{"select": {"chain": k["ASP"][0], "resid": k["ASP"][1]},
                                  "variant": "HIP"}])
    with pytest.raises(ProtonationError, match="cannot change which residue"):
        assign_variants(topology, method="propka", ph=7.0, overrides=overrides)


@pytest.mark.parametrize("entry, words", [
    ({"select": {"chain": "A"}, "variant": "HID"}, "must name chain and resid"),
    ({"select": {"chain": "A", "resid": "1"}, "variant": "CYM"}, "not one addHydrogens can build"),
    ({"select": {"chain": "A", "resid": "1"}}, "must be {select"),
])
def test_a_malformed_override_is_refused(entry, words):
    with pytest.raises(ProtonationError, match=words):
        parse_overrides([entry])


def test_frozen_residues_are_never_assigned(peptide):
    topology, _ = _topology(peptide)
    k = _keys(topology)
    assignments, _ = assign_variants(topology, method="propka", ph=7.0, frozen_residues={k["LYS"]})
    assert "LYS" not in _by_residue(assignments)


# --- histidine proximity --------------------------------------------------------------------------

def _with_zinc(topology, positions, *, near: str, offset_angstrom: float):
    """The peptide plus one ZN in its own chain, `offset_angstrom` beyond the HIS atom `near`."""
    from openmm import Vec3, app, unit
    from openmm.app import element

    modeller = app.Modeller(topology, positions)
    his = next(r for r in modeller.topology.residues() if r.name == "HIS")
    anchor = next(a for a in his.atoms() if a.name == near)
    base = modeller.positions[anchor.index].value_in_unit(unit.nanometer)
    ion = app.Topology()
    chain = ion.addChain("Z")
    residue = ion.addResidue("ZN", chain, "1")
    ion.addAtom("ZN", element.zinc, residue)
    modeller.add(ion, [Vec3(base[0] + offset_angstrom / 10.0, base[1], base[2])] * unit.nanometer)
    return modeller.topology, modeller.positions


def test_a_histidine_near_an_ion_is_warned_with_both_nitrogen_distances(peptide):
    topology, positions = _topology(peptide)
    topology, positions = _with_zinc(topology, positions, near="NE2", offset_angstrom=2.1)
    warnings = histidine_proximity(topology, positions, neighbours=(), ions={"ZN"},
                                   cutoff_angstrom=5.0, variants={}, assignments={})
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["neighbour"]["kind"] == "ion" and entry["neighbour"]["residue"] == "ZN"
    assert entry["NE2_ion_distance_angstrom"] == pytest.approx(2.1, abs=0.01)
    assert entry["ND1_ion_distance_angstrom"] > entry["NE2_ion_distance_angstrom"]


def test_a_distant_histidine_is_not_warned(peptide):
    """The acceptance criterion: an ordinary nonproximal histidine triggers nothing."""
    topology, positions = _topology(peptide)
    topology, positions = _with_zinc(topology, positions, near="NE2", offset_angstrom=12.0)
    assert histidine_proximity(topology, positions, neighbours=(), ions={"ZN"},
                               cutoff_angstrom=5.0, variants={}, assignments={}) == []


# --- the whole step, with the real PROPKA ------------------------------------------------------------

def test_propka_end_to_end_assigns_records_and_warns(peptide, tmp_path):
    pytest.importorskip("propka")
    from openmm import app

    topology, positions = _topology(peptide)
    topology, positions = _with_zinc(topology, positions, near="NE2", offset_angstrom=2.1)
    k = _keys(topology)
    before = {a.name: a for a in next(r for r in topology.residues() if r.name == "LYS").atoms()}
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    printed = []
    result = protonate_structure(
        topology, positions, forcefield, {"method": "propka", "ph": 7.0},
        frozen_residues={k["LYS"]}, seed=7, workdir=tmp_path, echo=printed.append)
    record = result.record

    assert record["method"] == "propka" and record["propka"]["version"]
    assert (tmp_path / record["propka"]["report"]).is_file()
    assert (tmp_path / record["propka"]["input_pdb"]).is_file()
    assert record["propka"]["predictions"], "PROPKA returned no predictions"
    assert {a["residue"] for a in record["assignments"]} >= {"HIS", "ASP", "GLU", "CYS"}
    assert record["histidine_proximity"], "the zinc at the histidine was not screened"
    assert any("NE2-ion 2.1" in line for line in printed), printed
    # The frozen residue keeps exactly the atoms it came with, hydrogens included.
    after = {a.name for a in next(r for r in result.topology.residues() if r.name == "LYS").atoms()}
    assert after == set(before)
    assert "LYS" not in {a["residue"] for a in record["assignments"]}


def test_a_missing_propka_is_an_error_not_a_fallback(peptide, tmp_path, monkeypatch):
    import importlib.metadata as metadata

    from openmm import app

    def absent(name):
        if name == "propka":
            raise metadata.PackageNotFoundError(name)
        return "0"

    monkeypatch.setattr(metadata, "version", absent)
    topology, positions = _topology(peptide)
    with pytest.raises(ProtonationError, match="not installed"):
        protonate_structure(topology, positions, app.ForceField("amber14-all.xml"),
                            {"method": "propka"}, seed=1, workdir=tmp_path, echo=None)


def test_a_failed_prediction_is_an_error_not_a_fallback(peptide, tmp_path, monkeypatch):
    pytest.importorskip("propka")
    import propka.run

    from openmm import app

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(propka.run, "single", broken)
    topology, positions = _topology(peptide)
    with pytest.raises(ProtonationError, match="not replaced by OpenMM's defaults"):
        protonate_structure(topology, positions, app.ForceField("amber14-all.xml"),
                            {"method": "propka"}, seed=1, workdir=tmp_path, echo=None)
