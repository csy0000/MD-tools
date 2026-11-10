"""Reduce's Asn/Gln/His flips and histidine tautomers, and who decides a histidine.

The parser and the assignment rule run on canned input. The real runs call Reduce on the TYK2
complex fixture (protein hydrogens deleted, as `protonate_structure` leaves it), which takes under
a second and needs `reduce` on PATH: environment.yml pins it, and a missing one FAILS here, as it
fails a default build -- it is not skipped.

PLATFORM_POLICY_EXEMPTION: no Context is created; Reduce is a CPU program.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from md_tools.openmm.protonation import ProtonationError, assign_variants

REPO = Path(__file__).resolve().parents[1]
TYK2 = REPO / "tests" / "data" / "alchemy" / "tyk2-v1" / "prepared" / "complex_ejm_31.pdb"
ALA = REPO / "tests" / "data" / "ALA.pdb"


def _stripped(path: Path):
    """The structure as `protonate_structure` hands it to Reduce: protein hydrogens deleted."""
    from openmm import app
    from openmm.app import element as elem

    pdb = app.PDBFile(str(path))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element == elem.hydrogen])
    return modeller


def _ligand_key(topology):
    from md_tools.openmm.protonation import residue_key

    return {residue_key(r) for r in topology.residues() if r.name == "L31"}


# --- parsing -------------------------------------------------------------------------------------

def test_the_decision_lines_are_read_with_chain_resid_icode_and_flags():
    from md_tools.openmm.reduce_flips import _parse_output

    text = "\n".join([
        "USER  MOD Set 1.2: A1177 GLN     :FLIP  amide:sc=    1.72  F(o=3.9,f=4.8)",
        "USER  MOD Single : A 893 HIS     :     no HD1:sc=       0  X(o=0,f=0.073)",
        "USER  MOD Single : B  12AASN     :      amide:sc=    3.21  K(o=3.2,f=0.16!)",
        "USER  MOD Single : A 101 LYS NZ  :NH3+    180:sc=       0   (180deg=0)",
        "USER  MOD Single : A  77 GLN     :      amide:sc=   -9.98! C(o=-9.98!,f=-12!)",
        "USER  MOD NoAdj-H:     2 HIS HE2 :     2 HIS NE2 : Z   1  ZNZN   :(H bumps)",
        "ATOM      1  N   ASN B  12A     11.104   6.134  -6.504  1.00  0.00           N",
    ])
    decisions, atoms = _parse_output(text)
    assert set(decisions) == {("A", "1177", ""), ("A", "893", ""), ("B", "12", "A"),
                              ("A", "77", ""), ("", "2", "")}
    # A clashing score carries its `!`, and is still a number.
    assert decisions[("A", "77", "")]["score"] == -9.98 and decisions[("A", "77", "")]["clash"]
    # A histidine on a metal is one Reduce declines to adjust: no decision is read into it.
    assert decisions[("", "2", "")]["code"] == "NoAdj" and not decisions[("", "2", "")]["flipped"]
    assert decisions[("A", "1177", "")]["flipped"] and decisions[("A", "1177", "")]["code"] == "F"
    assert decisions[("A", "893", "")]["code"] == "X" and not decisions[("A", "893", "")]["flipped"]
    assert decisions[("B", "12", "A")]["clash"]
    assert atoms[("B", "12", "A")]["N"] == (11.104, 6.134, -6.504)


# --- who decides a histidine ---------------------------------------------------------------------

def _one_histidine():
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain("A")
    topology.addResidue("HIS", chain, id="5")
    return topology, ("A", "5", "")


def _prediction(pka):
    from md_tools.openmm.protonation import Prediction

    return [Prediction(group="HIS", key=("A", "5", ""), pka=pka, model_pka=6.5, coupled=False)]


def test_with_method_openmm_the_histidine_is_reduces():
    topology, key = _one_histidine()
    for state in ("HID", "HIE", "HIP"):
        (a,), _ = assign_variants(topology, method="openmm", ph=7.0, tautomers={key: state})
        assert (a.variant, a.source) == (state, "reduce")


def test_with_method_propka_the_charge_is_propkas_and_only_a_neutral_one_takes_reduces_tautomer():
    topology, key = _one_histidine()
    (a,), _ = assign_variants(topology, method="propka", ph=7.0, predictions=_prediction(5.0),
                              tautomers={key: "HID"})
    assert a.variant == "HID" and "reduce tautomer" in a.source
    (a,), _ = assign_variants(topology, method="propka", ph=7.0, predictions=_prediction(8.0),
                              tautomers={key: "HIE"})
    assert a.variant == "HIP" and any("supersedes Reduce's HIE" in n for n in a.notes)
    # Reduce wants both ring hydrogens, PROPKA says neutral: the charge is PROPKA's and the
    # tautomer goes back to OpenMM's heuristic, with the disagreement written down.
    (a,), _ = assign_variants(topology, method="propka", ph=7.0, predictions=_prediction(5.0),
                              tautomers={key: "HIP"})
    assert a.variant is None and any("PROPKA predicts neutral" in n for n in a.notes)


def test_an_override_wins_over_reduce_and_says_so():
    topology, key = _one_histidine()
    (a,), _ = assign_variants(topology, method="openmm", ph=7.0, overrides={key: "HIE"},
                              tautomers={key: "HID"})
    assert (a.variant, a.source) == ("HIE", "override")
    assert any("supersedes Reduce's HID" in n for n in a.notes)


def test_without_reduce_the_histidine_is_left_to_openmm_as_before():
    topology, _ = _one_histidine()
    (a,), _ = assign_variants(topology, method="openmm", ph=7.0)
    assert a.variant is None and a.source.startswith("openmm addHydrogens")


# --- availability --------------------------------------------------------------------------------

def test_reduce_missing_is_an_error_naming_the_install_and_the_deliberate_alternative(monkeypatch):
    import md_tools.openmm.reduce_flips as rf

    monkeypatch.setattr(rf.shutil, "which", lambda name: None)
    with pytest.raises(ProtonationError, match=r"bioconda reduce=4\.15.*protonation\.flips: none"):
        rf.reduce_executable()


def test_a_structure_with_no_asn_gln_or_his_does_not_need_reduce(monkeypatch, tmp_path):
    from openmm import app

    import md_tools.openmm.reduce_flips as rf
    from md_tools.openmm.protonation import protonate_structure

    monkeypatch.setattr(rf.shutil, "which", lambda name: None)
    pdb = app.PDBFile(str(ALA))
    forcefield = app.ForceField("amber14-all.xml")
    result = protonate_structure(pdb.topology, pdb.positions, forcefield, {"flips": "reduce"},
                                 seed=1, workdir=tmp_path, echo=None)
    assert result.record["flips"] == "reduce" and result.record["reduce"] is None
    assert "Reduce not run" in result.record["reduce_note"]


def _protein_only():
    """TYK2 without its ligand: a real protein whose hydrogens addHydrogens can place alone."""
    modeller = _stripped(TYK2)
    modeller.delete([r for r in modeller.topology.residues() if r.name == "L31"])
    return modeller


def test_flips_none_never_runs_reduce_and_keeps_openmms_histidines(monkeypatch, tmp_path):
    from openmm import app

    import md_tools.openmm.reduce_flips as rf
    from md_tools.openmm.protonation import protonate_structure

    def no_reduce(*_, **__):
        raise AssertionError("Reduce was run under flips: none")

    monkeypatch.setattr(rf, "run_reduce", no_reduce)
    modeller = _protein_only()
    result = protonate_structure(modeller.topology, modeller.positions,
                                 app.ForceField("amber14-all.xml"), {"flips": "none"},
                                 seed=1, workdir=tmp_path, echo=None)
    assert result.record["flips"] == "none" and result.record["reduce"] is None
    his = [a for a in result.record["assignments"] if a["residue"] == "HIS"]
    assert his and all(a["source"].startswith("openmm") for a in his)


def test_the_default_build_takes_reduces_flips_and_histidines(tmp_path):
    from openmm import app

    from md_tools.openmm.protonation import protonate_structure

    modeller = _protein_only()
    result = protonate_structure(modeller.topology, modeller.positions,
                                 app.ForceField("amber14-all.xml"), {}, seed=1,
                                 workdir=tmp_path, echo=None)
    record = result.record
    assert record["flips"] == "reduce" and record["reduce"]["version"]
    assert record["reduce"]["summary"]["flipped"] >= 1
    flipped = [d for d in record["reduce"]["decisions"] if d["flipped"]]
    assert all(d["coordinates_angstrom"] for d in flipped)
    his = {(a["chain"], a["resid"]): a for a in record["assignments"] if a["residue"] == "HIS"}
    tautomers = {(d["chain"], d["resid"]): d["tautomer"] for d in record["reduce"]["decisions"]
                 if d["group"] == "imidazole"}
    assert his and set(his) == set(tautomers)
    for key, assignment in his.items():
        assert (assignment["source"], assignment["final_variant"]) == ("reduce", tautomers[key])


# --- real Reduce on the TYK2 complex -------------------------------------------------------------

def test_reduce_on_the_complex_flips_and_moves_only_the_flipped_heavy_atoms(tmp_path):
    from openmm import unit

    from md_tools.openmm.reduce_flips import apply_flips, run_reduce

    modeller = _stripped(TYK2)
    frozen = _ligand_key(modeller.topology)
    out = run_reduce(modeller.topology, modeller.positions, frozen_residues=frozen,
                     workdir=tmp_path)
    d = out["decisions"]
    flipped = {k for k, v in d.items() if v["flipped"]}
    # Measured with Reduce 4.15 on this file, ligand heavy atoms present: a real answer to
    # check against, not a definition. A different Reduce version may legitimately differ.
    assert flipped == {("A", "920", ""), ("A", "1005", ""), ("A", "1146", ""),
                       ("A", "1177", "")}
    assert d[("A", "1146", "")]["tautomer"] == "HIE" and d[("A", "1015", "")]["tautomer"] == "HID"
    assert set(out["coordinates"]) == flipped

    # The frozen ligand was handed to Reduce as UNL and came back with nothing added.
    text = (tmp_path / "reduce_input.pdb").read_text()
    assert " L31 " not in text and " UNL " in text

    before = modeller.positions.value_in_unit(unit.angstrom)
    after = apply_flips(modeller.topology, modeller.positions,
                        out["coordinates"]).value_in_unit(unit.angstrom)
    moved = {(a.residue.chain.id, a.residue.id, a.name) for a in modeller.topology.atoms()
             if math.dist(before[a.index], after[a.index]) > 1e-6}
    assert {(c, r) for c, r, _ in moved} <= {(k[0], k[1]) for k in flipped}
    # An amide flip swaps O and N: the new ND2 sits where OD1 was, within Reduce's re-docking.
    names = {(a.residue.id, a.name): a.index for a in modeller.topology.atoms()
             if a.residue.chain.id == "A"}
    assert math.dist(before[names[("920", "OD1")]], after[names[("920", "ND2")]]) < 0.3


# --- a reference rebuild reproduces the flips without Reduce -------------------------------------

def test_a_reference_rebuild_applies_the_recorded_flips_and_needs_no_reduce(tmp_path, monkeypatch):
    from openmm import app, unit

    import md_tools.openmm.reduce_flips as rf
    from md_tools.openmm.protonation import protonate_structure
    from md_tools.openmm.seeds import derive_build_seed
    from md_tools.reference import standalone_build
    from md_tools.reference.export import recorded_protonation

    source = tmp_path / "prepared.pdb"
    modeller = _protein_only()
    with source.open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    pdb = app.PDBFile(str(source))
    seed = derive_build_seed(7, "structure/protonation")
    built = protonate_structure(pdb.topology, pdb.positions, app.ForceField("amber14-all.xml"),
                                {}, seed=seed, workdir=tmp_path / "build", echo=None)
    assert built.record["reduce"]["summary"]["flipped"] >= 1

    monkeypatch.setattr(rf.shutil, "which", lambda name: None)      # the bundle has no Reduce

    def rebuild(recorded, name):
        builder = {"protonation": {"skip_for_ligand": True, "delete_existing_hydrogens": True,
                                   "ph": 7.0, "variants": None, "recorded": recorded},
                   "run": {"seed": 7},
                   "forcefield": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml",
                                  "extra_xml": []}}
        work = tmp_path / name
        work.mkdir()
        out = standalone_build.protonate(source, work, builder, None, False)
        return app.PDBFile(str(out)).positions.value_in_unit(unit.angstrom)

    expected = built.positions.value_in_unit(unit.angstrom)
    recorded = recorded_protonation(built.record)
    same = rebuild(recorded, "bundle")
    assert len(same) == len(expected)
    assert max(math.dist(a, b) for a, b in zip(same, expected)) < 2e-3    # PDB rounding only

    # The guard can fail: the same rebuild without the recorded flips is a different structure.
    unflipped = rebuild(dict(recorded, flipped=[]), "no-flips")
    assert max(math.dist(a, b) for a, b in zip(unflipped, expected)) > 1.0
