"""An mmCIF biological assembly expands with distinct chains, and on-axis ions are one atom.

A synthetic three-fold assembly written as mmCIF text: one GLY chain off the axis and one ZN on the
axis at occupancy 1/3, with the identity and two 120-degree operators. The real case is 1TYL
assembly 3 (the T3R3 insulin hexamer), whose two zinc ions sit on the axis exactly like this.

PLATFORM_POLICY_EXEMPTION: file expansion only; no Context.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("gemmi")

from md_tools.openmm.assembly import AssemblyError, expand_assembly  # noqa: E402

COS, SIN = -0.5, 0.8660254038

CIF = """data_TEST
_cell.length_a 1
_cell.length_b 1
_cell.length_c 1
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 1'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N  N  . GLY A 1 1 ? 5.000 0.000 0.000 1.00 10.0 1 GLY A N  1
ATOM   2 C  CA . GLY A 1 1 ? 6.400 0.000 0.000 1.00 10.0 1 GLY A CA 1
ATOM   3 C  C  . GLY A 1 1 ? 7.000 1.300 0.000 1.00 10.0 1 GLY A C  1
ATOM   4 O  O  . GLY A 1 1 ? 6.400 2.300 0.000 1.00 10.0 1 GLY A O  1
HETATM 5 ZN ZN . ZN  B 2 . ? 0.000 0.000 3.000 0.33 10.0 31 ZN A ZN 1
{extra}loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
_pdbx_struct_assembly.method_details
_pdbx_struct_assembly.oligomeric_details
_pdbx_struct_assembly.oligomeric_count
1 author_defined_assembly ? monomeric 1
2 author_defined_assembly ? trimeric 3
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 1 A,B
2 1,2,3 A,B
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.name
_pdbx_struct_oper_list.symmetry_operation
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 'identity operation' 1_555 x,y,z 1 0 0 0 0 1 0 0 0 0 1 0
2 'crystal symmetry operation' 2_555 -y,x-y,z {c} {ms} 0 0 {s} {c} 0 0 0 0 1 0
3 'crystal symmetry operation' 3_555 -x+y,-x,z {c} {s} 0 0 {ms} {c} 0 0 0 0 1 0
"""


def _cif(tmp_path, extra=""):
    path = tmp_path / "test.cif"
    path.write_text(CIF.format(c=COS, s=SIN, ms=-SIN, extra=extra), encoding="utf-8")
    return path


def test_each_copy_is_its_own_chain_and_records_where_it_came_from(tmp_path):
    record = expand_assembly(_cif(tmp_path), "2", tmp_path / "out.pdb")
    chains = record["chains"]
    assert [c["chain_id"] for c in chains] == ["A", "B", "C"]
    assert {c["author_chain"] for c in chains} == {"A"}
    assert [c["operator_id"] for c in chains] == ["1", "2", "3"]
    assert record["residue_counts"] == {"GLY": 3, "ZN": 1}
    assert (tmp_path / "out.pdb").is_file()
    json.dumps(record)  # the record is serialisable as written


def test_the_on_axis_zinc_is_kept_once_and_each_drop_is_recorded(tmp_path):
    record = expand_assembly(_cif(tmp_path), "2", tmp_path / "out.pdb")
    dropped = record["deduplicated"]
    assert len(dropped) == 2
    assert all(d["kept"]["chain"] == "A" and d["kept"]["residue"] == "ZN" for d in dropped)
    assert all(d["occupancy_summed"] == pytest.approx(0.99) for d in dropped)
    from openmm import app

    topology = app.PDBFile(str(tmp_path / "out.pdb")).topology
    assert sum(1 for r in topology.residues() if r.name == "ZN") == 1


def test_the_rotated_copies_are_where_the_operator_puts_them(tmp_path):
    from openmm import app, unit

    expand_assembly(_cif(tmp_path), "2", tmp_path / "out.pdb")
    structure = app.PDBFile(str(tmp_path / "out.pdb"))
    xyz = structure.positions.value_in_unit(unit.angstrom)
    n_atoms = [i for i, a in enumerate(structure.topology.atoms())
               if a.name == "N" and a.residue.name == "GLY"]
    assert [round(xyz[i][0], 2) for i in n_atoms] == [5.0, -2.5, -2.5]


def test_an_unknown_assembly_is_refused_naming_the_real_ones(tmp_path):
    with pytest.raises(AssemblyError, match="defines 1, 2"):
        expand_assembly(_cif(tmp_path), "9", tmp_path / "out.pdb")
    assert not (tmp_path / "out.pdb").exists()


def test_coinciding_protein_atoms_are_refused_not_merged(tmp_path):
    """A GLY on the axis would be copied onto itself: that is a wrong assembly, not a special position."""
    on_axis = tmp_path / "axis.cif"
    text = CIF.format(c=COS, s=SIN, ms=-SIN, extra="")
    text = text.replace("5.000 0.000 0.000", "0.000 0.000 0.000")
    text = text.replace("6.400 0.000 0.000", "0.000 0.000 1.400")
    text = text.replace("7.000 1.300 0.000", "0.000 0.000 2.600")
    text = text.replace("6.400 2.300 0.000", "0.000 0.000 3.800").replace("0.000 0.000 3.000 0.33",
                                                                          "0.000 0.000 9.000 0.33")
    on_axis.write_text(text, encoding="utf-8")
    with pytest.raises(AssemblyError, match="Only ions and waters"):
        expand_assembly(on_axis, "2", tmp_path / "out.pdb")
    assert not (tmp_path / "out.pdb").exists()


def test_a_pdb_file_is_refused(tmp_path):
    (tmp_path / "x.pdb").write_text("END\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="mmCIF"):
        expand_assembly(tmp_path / "x.pdb", "1", tmp_path / "out.pdb")
