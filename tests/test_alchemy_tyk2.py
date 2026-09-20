"""The TYK2 fixture: the shared preparation for the protein-ligand campaign.

The built Systems are not committed (56 MB), so the tests that need one BUILD it, through the
fixture's own script, into a temporary directory with `$MD_DATA` pointed at an empty root. Those
are marked `slow`. Nothing here skips silently: the fast tests need no build at all.

Energies run on OpenMM's Reference platform. Nothing here is CUDA evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("rdkit")

TYK2 = Path(__file__).resolve().parent / "data" / "alchemy" / "tyk2-v1"
SCRIPT = TYK2 / "build_tyk2_fixture.py"
LIGANDS = {"ejm_31": ("L31", "LOCAL-DKNAYSZNMZIMIZ/param_bd1388e5fe3e", -9.54),
           "ejm_42": ("L42", "LOCAL-CEJFWHOOEUEYOE/param_c1d8d1147233", -9.78),
           "ejm_43": ("L43", "LOCAL-HKQHWWQCRDXFGC/param_c66a3e91a5dd", -8.26)}
EDGES = [("ejm_31", "ejm_42", 31, 1, 4), ("ejm_31", "ejm_43", 30, 2, 8)]


def _package(ligand: str):
    from md_tools.ligands import load_package

    return load_package(TYK2 / "packages" / LIGANDS[ligand][1])


def _script(out: Path, *args: str) -> Path:
    """Run the fixture's own build script, pinned to this checkout's md_tools."""
    import md_tools

    root = str(Path(md_tools.__file__).resolve().parents[1])
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), *args],
                   check=True, timeout=3600, cwd=root)
    return out


def _environment(build: Path, ligand: str, kind: str):
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    selector = (LigandSelector(chain="B", resid="1") if kind == "complex"
                else LigandSelector(resname=LIGANDS[ligand][0]))
    return Environment.from_files(build / "built.xml", build / "built.pdb", selector,
                                  record=build / "built.log")


# ------------------------------------------------------------------------------------------------
# what is committed: no build needed
# ------------------------------------------------------------------------------------------------
def test_the_prepared_structures_are_what_the_script_derives_from_the_inputs():
    """The twelve cap renames and the three complex PDBs, re-derived and compared byte for byte."""
    import md_tools

    root = str(Path(md_tools.__file__).resolve().parents[1])
    done = subprocess.run([sys.executable, str(SCRIPT), "--check-prepared"],
                          capture_output=True, text=True, timeout=600, cwd=root)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "prepared files reproduce" in done.stdout


def test_the_caps_carry_the_template_names_and_the_inputs_are_untouched():
    upstream = (TYK2 / "inputs" / "protein.pdb").read_text().splitlines()
    prepared = (TYK2 / "prepared" / "protein_caps_renamed.pdb").read_text().splitlines()
    assert len(upstream) == len(prepared)
    changed = [(a, b) for a, b in zip(upstream, prepared) if a != b]
    assert len(changed) == 12, changed[:3]
    assert all(a[17:20].strip() in ("ACE", "NME") for a, _ in changed)
    # only columns 13-16, the atom name, differ
    assert all(a[:12] == b[:12] and a[16:] == b[16:] for a, b in changed)
    names = {b[12:16].strip() for _, b in changed}
    assert names == {"C", "O", "CH3", "HH31", "HH32", "HH33", "N", "H"}


@pytest.mark.parametrize("ligand", sorted(LIGANDS))
def test_each_package_is_the_upstream_ligand_it_claims(ligand):
    """The package's compound id is the InChIKey block upstream records for that ligand."""
    residue, reference, dg = LIGANDS[ligand]
    package = _package(ligand)
    assert package.reference == reference and package.residue_name == residue
    upstream = json.loads((TYK2 / "inputs" / "experimental_binding_data.json").read_text())
    assert package.compound_id == "LOCAL-" + upstream[ligand]["inchikey"].split("-")[0]
    assert upstream[ligand]["dg"]["magnitude"] == pytest.approx(dg, abs=5e-3)
    assert package.metadata["chemical_state"]["net_formal_charge"] == 0
    assert package.metadata["charges"]["method"] == "am1bcc"


@pytest.mark.parametrize("ligand", sorted(LIGANDS))
@pytest.mark.parametrize("kind", ["complex", "solvated", "vacuum"])
def test_every_committed_record_describes_a_completed_build(ligand, kind):
    from md_tools.build.record import read_record

    record = read_record(TYK2 / "records" / ligand / kind / "built.log")
    assert record["status"] == "completed" and record["record_type"] == "build-top"
    assert record["solvent"]["treatment"] == ("vacuum" if kind == "vacuum" else "explicit")
    assert record["environment"]["hostname"] == "<redacted>"
    assert record["environment"]["user"] == "<redacted>"
    compatibility = record["forcefield_record"]["ligand"]["nonbonded_compatibility"]
    assert compatibility["packages"][0]["reference"] == LIGANDS[ligand][1]
    if kind != "vacuum":
        import numpy as np

        vectors = np.array(record["box_vectors_nm"])
        cutoff = record["forcefield_record"]["nonbonded"]["cutoff_nm"]
        volume = abs(np.dot(vectors[0], np.cross(vectors[1], vectors[2])))
        heights = [volume / np.linalg.norm(np.cross(vectors[i - 2], vectors[i - 1]))
                   for i in range(3)]
        assert min(heights) >= 2 * cutoff + 0.8, (ligand, kind, heights)


@pytest.mark.parametrize("a_name,b_name,mapped,a_only,b_only", EDGES)
def test_the_automatic_map_of_each_edge_is_the_congeneric_one(a_name, b_name, mapped, a_only,
                                                              b_only):
    from md_tools.alchemy.topology_mapping import propose_map, validate_map

    a, b = _package(a_name), _package(b_name)
    amap, report = propose_map(a, b, "hybrid")
    validate_map(a, b, amap, "hybrid")
    assert len(amap.pairs) == mapped
    assert len(a.atom_names) - mapped == a_only and len(b.atom_names) - mapped == b_only
    assert report["mcs_heavy_atoms"] == 21
    assert propose_map(a, b, "hybrid")[0] == amap          # deterministic


# ------------------------------------------------------------------------------------------------
# what has to be built
# ------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_the_ligand_legs_build_and_their_plans_recover_both_endpoints(tmp_path):
    from md_tools.alchemy.topology import build_topology_plan, matched_legs
    from md_tools.alchemy.topology_mapping import propose_map
    from md_tools.alchemy.topology_recovery import endpoint_accounting
    from tests.alchemy_fixtures import ENERGY_TOL_KJ, independent_reference

    _script(tmp_path, "--ligand", "ejm_31", "--kind", "solvated", "--kind", "vacuum")
    solvent_env = _environment(tmp_path / "ejm_31" / "solvated" / "build", "ejm_31", "solvated")
    vacuum_env = _environment(tmp_path / "ejm_31" / "vacuum" / "build", "ejm_31", "vacuum")
    assert not any((tmp_path / "md_data_root").iterdir())

    a = _package("ejm_31")
    for _, target, *_ in EDGES:
        b = _package(target)
        amap, _report = propose_map(a, b, "hybrid")
        plan = build_topology_plan(a, b, amap, solvent_env, mode="hybrid")
        for side, package in (("A", a), ("B", b)):
            reference, index = independent_reference(plan, solvent_env, package, side)
            accounting = endpoint_accounting(plan, side, reference, index)
            assert abs(accounting["raw_total_difference"]) > 1e-2
            assert max(abs(v) for v in accounting["residual"].values()) < ENERGY_TOL_KJ, accounting
        vacuum_plan = build_topology_plan(a, b, amap, vacuum_env, mode="hybrid")
        assert matched_legs(plan, vacuum_plan)["ligand_hamiltonian_sha256"]
        assert plan.record["environment"]["solvation"] == "explicit"
        assert vacuum_plan.record["environment"]["solvation"] == "vacuum"


@pytest.mark.slow
def test_single_topology_is_refused_on_a_constrained_bond_and_dual_builds(tmp_path):
    """The H -> heavy mapping single topology would need makes a constraint appear."""
    from md_tools.alchemy.topology import TopologyError, build_topology_plan
    from md_tools.alchemy.topology_mapping import AtomMap, propose_map

    _script(tmp_path, "--ligand", "ejm_31", "--kind", "solvated")
    env = _environment(tmp_path / "ejm_31" / "solvated" / "build", "ejm_31", "solvated")
    a, b = _package("ejm_31"), _package("ejm_42")
    amap, _ = propose_map(a, b, "hybrid")
    assert build_topology_plan(a, b, amap, env, mode="dual").common == frozenset()

    spare_a = [i for i in range(len(a.atom_names)) if i not in amap.a_to_b]
    spare_heavy_b = [j for j in range(len(b.atom_names))
                     if j not in amap.b_to_a and b.mol.GetAtomWithIdx(j).GetAtomicNum() > 1]
    full = AtomMap.from_pairs(a, b, list(amap.pairs) + [(spare_a[0], spare_heavy_b[0])])
    with pytest.raises(TopologyError, match="constrained at A and flexible at B"):
        build_topology_plan(a, b, full, env, mode="single")


@pytest.mark.slow
def test_the_complex_builds_and_its_plan_recovers_both_endpoints(tmp_path):
    """The whole point of the fixture: a real protein-ligand System, and both endpoints exact."""
    from md_tools.alchemy.topology import build_topology_plan
    from md_tools.alchemy.topology_mapping import propose_map
    from md_tools.alchemy.topology_recovery import endpoint_accounting
    from tests.alchemy_fixtures import ENERGY_TOL_KJ, independent_reference

    _script(tmp_path, "--ligand", "ejm_31", "--kind", "complex")
    env = _environment(tmp_path / "ejm_31" / "complex" / "build", "ejm_31", "complex")
    assert env.system.getNumParticles() > 50_000
    assert {r.name for r in env.topology.residues()} >= {"ACE", "NME", "HOH", "L31"}

    a, b = _package("ejm_31"), _package("ejm_42")
    amap, _ = propose_map(a, b, "hybrid")
    plan = build_topology_plan(a, b, amap, env, mode="hybrid")
    assert len(plan.common) == 31 and len(plan.a_only) == 1 and len(plan.b_only) == 4
    for side, package in (("A", a), ("B", b)):
        reference, index = independent_reference(
            plan, env, package, side, forcefield_files=("amber14-all.xml", "amber14/tip3p.xml"))
        accounting = endpoint_accounting(plan, side, reference, index)
        assert abs(accounting["raw_total_difference"]) > 1e-2
        assert max(abs(v) for v in accounting["residual"].values()) < ENERGY_TOL_KJ, accounting


@pytest.mark.slow
def test_an_abfe_decoupling_leg_in_the_complex_needs_a_restraint_and_records_it(tmp_path):
    """The ABFE construction on the real system: refused without a restraint, recorded with one."""
    from md_tools.alchemy.topology import TopologyError, build_decoupling_plan, matched_legs

    _script(tmp_path, "--ligand", "ejm_31", "--kind", "complex", "--kind", "solvated")
    complex_env = _environment(tmp_path / "ejm_31" / "complex" / "build", "ejm_31", "complex")
    solvent_env = _environment(tmp_path / "ejm_31" / "solvated" / "build", "ejm_31", "solvated")
    package = _package("ejm_31")

    with pytest.raises(TopologyError, match="needs a standard-state restraint"):
        build_decoupling_plan(package, complex_env)

    ligand_atoms = sorted(a.index for a in complex_env.topology.atoms()
                          if a.residue.name == "L31")[:3]
    protein_atoms = sorted(a.index for a in complex_env.topology.atoms()
                           if a.residue.name not in ("HOH", "NA", "CL", "L31")
                           and a.name in ("CA", "C", "N"))[:3]
    complex_leg = build_decoupling_plan(
        package, complex_env,
        restraint={"kind": "boresch", "ligand_atoms": ligand_atoms,
                   "environment_atoms": protein_atoms})
    [recorded] = complex_leg.record["restraints"]
    assert recorded["role"] == "standard-state" and recorded["kind"] == "boresch"
    assert recorded["ligand_atoms"] == ligand_atoms
    labels = recorded["environment_atom_labels"]
    assert len(labels) == 3 and ":" in labels[0]
    assert "the plan records it, and does not build it" in recorded["built_by"]

    # the two legs of the ABFE cycle: the same ligand Hamiltonian, one restraint between them
    solvent_leg = build_decoupling_plan(package, solvent_env)
    report = matched_legs(complex_leg, solvent_leg)
    assert report["ligand_hamiltonian_sha256"] == complex_leg.record["ligand_hamiltonian_sha256"]
    assert report["restraints"]["standard_state"][0][0]["kind"] == "boresch"
    assert report["restraints"]["standard_state"][1] == []
