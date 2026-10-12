"""Which residues keep their amide omega protected: a person's answer, else the build record's.

REST2 convention v4 protects the amide omega of a protein or peptide-like residue and scales a
ligand's. To the scaler a `kind: peptide-like` molecule is one non-protein residue read from an SDF,
exactly what a ligand looks like, so the only thing that tells them apart is the build record that
`build-top` wrote beside the System. Each case below can fail by a different mistake:

  * a peptide-like record must name the non-protein residue (a function that always answered "none"
    would scale a cyclic peptide's backbone);
  * a ligand record must name none (one that always answered "the non-protein residue" would
    protect every ligand amide again, which is the change being undone);
  * a stated list wins, EMPTY INCLUDED (so a person can say "none" about a peptide-like build);
  * no readable record is "none", said so, and not an error.

PLATFORM_POLICY_EXEMPTION: topology bookkeeping and a text record. No force field, no dynamics.
"""
from __future__ import annotations

import pytest

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")

from md_tools.build.record import LogWriter  # noqa: E402
from md_tools.build.scaler import peptide_like_residues_for  # noqa: E402


def _topology():
    """ALA beside a non-protein residue called CYC: one of each."""
    topology = app.Topology()
    chain = topology.addChain()
    ala = topology.addResidue("ALA", chain)
    topology.addAtom("CA", elem.carbon, ala)
    cyc = topology.addResidue("CYC", topology.addChain())
    topology.addAtom("C1", elem.carbon, cyc)
    return topology


def _record(directory, kind, name="built"):
    log = LogWriter(directory / f"{name}.log", record_type="build-top", command=["test"],
                    echo=False)
    log.update(resolved_config={"solute": {"kind": kind}})
    log.complete()
    log.save()
    return directory / f"{name}.xml"


def test_a_peptide_like_build_record_names_the_non_protein_residue(tmp_path):
    names, source = peptide_like_residues_for({"peptide_like_residues": None},
                                              _record(tmp_path, "peptide-like"), _topology())
    assert names == ["CYC"], "ALA is a protein residue and needs no declaration"
    assert "peptide-like" in source and "built.log" in source


@pytest.mark.parametrize("kind", ["ligand", "peptide", "complex"])
def test_any_other_kind_names_none_so_a_ligands_amide_is_scaled(tmp_path, kind):
    names, source = peptide_like_residues_for({"peptide_like_residues": None},
                                              _record(tmp_path, kind), _topology())
    assert names == []
    assert kind in source


def test_a_stated_list_wins_and_an_empty_one_is_still_stated(tmp_path):
    system = _record(tmp_path, "peptide-like")
    assert peptide_like_residues_for({"peptide_like_residues": ["XYZ"]}, system,
                                     _topology()) == (["XYZ"], "scaler.config")
    assert peptide_like_residues_for({"peptide_like_residues": []}, system,
                                     _topology()) == ([], "scaler.config")


def test_no_readable_record_is_none_and_says_so(tmp_path):
    names, source = peptide_like_residues_for({"peptide_like_residues": None},
                                              tmp_path / "built.xml", _topology())
    assert names == [] and "no readable build record" in source
    (tmp_path / "built.log").write_text("display text, no machine record\n", encoding="utf-8")
    names, source = peptide_like_residues_for({"peptide_like_residues": None},
                                              tmp_path / "built.xml", _topology())
    assert names == [] and "no readable build record" in source


def test_the_scaler_schema_leaves_it_unset_by_default():
    from md_tools.build.scaler import SCALER_SCHEMA

    field = SCALER_SCHEMA.fields["peptide_like_residues"]
    assert field.default is None and field.nullable
