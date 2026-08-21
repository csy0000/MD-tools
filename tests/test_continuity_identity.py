"""The cMD continuity contract binds the exact artifacts, not a description of them.

Item 2. Version 2 hashed `system.xml` and the topology, but recorded the predecessor State by
*pathname* and the bundle by a partial parsed projection. A pathname can be repointed at a different
equilibration and a projection cannot notice a change to a field it does not name -- and every field
it does not name is still part of the Hamiltonian. Version 3 binds the predecessor State, the
manifest and the force-field file by their exact bytes.

Every end-to-end refusal here asserts the output streams are **byte-for-byte unchanged** afterwards.
That is the property that matters: a refusal issued after the trajectory was opened for append has
already done the damage it existed to prevent, and would still look like a clean error.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
INPUT_GEN = REPO_ROOT / "MD_input_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")
STREAMS = ("cMD_1_all_atoms.dcd", "cMD_1_selected_atoms.dcd", "cMD_1.log")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


@pytest.fixture(scope="module")
def committed_project(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("ci_bundle") / "bundle"
    prepared = _run(SYSTEM_GEN, "-i", str(ALANINE_PDB), "-o", str(bundle), "--config",
                    str(REPO_ROOT / "test" / "ala" / "cMD" / "implicit" / "system_config.json"))
    if prepared.returncode != 0:
        pytest.skip(f"implicit preparation unavailable: {prepared.stderr[-300:]}")

    root = tmp_path_factory.mktemp("ci_project")
    (root / "md.json").write_text(json.dumps({
        "profile": "implicit-md-peptide-v1",
        "protocol": {
            "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                           "temperature": "300 K", "friction": "1 /ps"},
            "equilibration": {"protocol": "simple", "minimize_max_iterations": 100,
                              "restrained": "1 ps"},
            "production": {"method": "md", "duration_per_segment": "2 ps"}},
        "randomness": {"master_seed": 20260821},
        "execution": {"platform": "CPU",
                      "reporting": {"all_atom": "1 ps", "solute": "0.2 ps"}}}))
    project = root / "run"
    assert _run(INPUT_GEN, "--system", str(bundle / "system_manifest.json"),
                "-o", str(project), "--config", str(root / "md.json")).returncode == 0
    for stage in ("min", "eq", "cMD_1"):
        result = subprocess.run([str(project / stage / f"{stage}.sh")], capture_output=True,
                                text=True, cwd=str(project / stage))
        assert result.returncode == 0, f"{stage}: {result.stderr[-500:]}"
    return project


def _fresh(committed_project, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(committed_project, copy)
    return copy


def _fingerprint(project: Path) -> dict:
    stage = project / "cMD_1"
    return {name: (stage / name).read_bytes() for name in STREAMS if (stage / name).is_file()}


def _continue(project: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(project / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                          cwd=str(project / "cMD_1"))


def _committed(project: Path) -> dict:
    found = glob.glob(str(project / "cMD_1" / "run" / "**" / "committed.json"), recursive=True)
    return json.loads(Path(found[0]).read_text())


def _edit_payload(project: Path, mutate) -> None:
    path = project / "cMD_1" / "cMD_1.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2))


def _must_refuse(project: Path, what: str) -> None:
    """Run a continuation that must fail, and prove it failed before touching any output."""
    before = _fingerprint(project)
    record = _committed(project)
    result = _continue(project)
    assert result.returncode != 0, f"{what} was accepted as a continuation"
    assert _fingerprint(project) == before, f"{what}: outputs were modified before the refusal"
    assert _committed(project)["invocations_completed"] == record["invocations_completed"], (
        f"{what}: the refusal still counted an invocation")


# ---------------------------------------------------------------------------------------------
# the contract records what it claims to record
# ---------------------------------------------------------------------------------------------

def test_the_committed_contract_carries_every_required_identity(committed_project):
    contract = _committed(committed_project)["continuity"]
    assert contract["contract_version"] == 3

    assert contract["system_xml_sha256"] and contract["topology_sha256"]
    assert contract["forcefield_sha256"], "the force-field FILE is not bound, only its projection"

    bundle = contract["bundle"]
    assert bundle["manifest_sha256"], "system_manifest.json is not bound by its bytes"
    assert "configuration_hash" in bundle

    predecessor = contract["predecessor"]
    assert predecessor["sha256"] not in (None, "absent"), predecessor
    assert predecessor["path"] and predecessor["produced_by"]

    ordering = contract["output_atom_ordering"]
    assert ordering["all_atom"] and ordering["selected_atoms"]

    for field in ("n_particles", "n_constraints", "periodic", "ensemble", "barostat_pressure",
                  "integrator", "restraint_active", "restraint_convention", "steps_per_segment",
                  "reporting_intervals", "selected_atoms_fingerprint"):
        assert field in contract, field


def test_the_continuity_hash_covers_the_whole_contract():
    """One hash over the canonical contract, so no field can change without changing it."""
    from md_templates.openmm.cmd_segments import continuity_hash

    base = {"a": 1, "nested": {"x": "y"}, "list": [1, 2]}
    assert continuity_hash(base) == continuity_hash(dict(reversed(list(base.items())))), (
        "the hash depends on key order, so it is not canonical")
    for mutation in ({"a": 2}, {"nested": {"x": "z"}}, {"list": [2, 1]}, {"new": None}):
        assert continuity_hash({**base, **mutation}) != continuity_hash(base), mutation


def test_ordered_atom_identity_distinguishes_a_permutation_from_a_count():
    """Equal counts are the case the fingerprint exists for; a set comparison would pass."""
    from md_templates.openmm.stage import _atom_identity
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("ALA", chain)
    for name in ("N", "CA", "C", "O"):
        topology.addAtom(name, app.element.carbon, residue)

    straight = _atom_identity(topology, [0, 1, 2, 3])
    swapped = _atom_identity(topology, [0, 2, 1, 3])
    assert straight != swapped
    assert len([0, 1, 2, 3]) == len([0, 2, 1, 3])


# ---------------------------------------------------------------------------------------------
# 1-8: refusal before mutation
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_1_a_changed_byte_in_system_xml_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "system.xml"
    text = path.read_text()
    marker = 'openmmVersion="'
    assert marker in text
    path.write_text(text.replace(marker, 'openmmVersion="9', 1))
    _must_refuse(project, "a one-byte change to system.xml")


@pytest.mark.slow
def test_2_a_topology_permutation_with_equal_counts_is_refused(committed_project, tmp_path):
    """Two atom records exchanged. Same file length, same atom count, different ordering."""
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "topology.pdb"
    lines = path.read_text().splitlines(keepends=True)
    atoms = [i for i, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    assert len(atoms) >= 2
    first, second = atoms[0], atoms[1]
    lines[first], lines[second] = lines[second], lines[first]
    path.write_text("".join(lines))
    _must_refuse(project, "a topology atom permutation")


@pytest.mark.slow
def test_3_a_changed_manifest_identity_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "system_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("system", {})["input_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest, indent=2))
    _must_refuse(project, "a changed system_manifest.json identity")


@pytest.mark.slow
def test_3b_a_manifest_change_outside_the_named_fields_is_still_refused(committed_project,
                                                                        tmp_path):
    """The reason the bytes are hashed: a projection cannot see a field it does not name."""
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "system_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["a_field_no_projection_names"] = "changed"
    path.write_text(json.dumps(manifest, indent=2))
    _must_refuse(project, "a manifest change outside the projected fields")


@pytest.mark.slow
def test_4_a_changed_forcefield_file_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "forcefield.json"
    record = json.loads(path.read_text())
    record["implicit_model"] = "OBC2"
    path.write_text(json.dumps(record, indent=2))
    _must_refuse(project, "a changed implicit model in forcefield.json")


@pytest.mark.slow
def test_4b_changed_water_model_provenance_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    path = project / "inputs" / "forcefield.json"
    record = json.loads(path.read_text())
    record["water"] = "amber19/tip3p.xml"
    path.write_text(json.dumps(record, indent=2))
    _must_refuse(project, "changed water-model provenance")


@pytest.mark.slow
def test_5_changed_predecessor_state_bytes_are_refused(committed_project, tmp_path):
    """The State that began this chain is immutable provenance, bound by bytes not by path."""
    project = _fresh(committed_project, tmp_path)
    payload = json.loads((project / "cMD_1" / "cMD_1.json").read_text())
    predecessor = (project / "cMD_1" / payload["input"]["state"]).resolve()
    assert predecessor.is_file(), predecessor
    text = predecessor.read_text()
    predecessor.write_text(text.replace("<State", "<State ", 1))
    _must_refuse(project, "changed predecessor State bytes")


@pytest.mark.slow
def test_5b_a_repointed_predecessor_path_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    payload = json.loads((project / "cMD_1" / "cMD_1.json").read_text())
    original = (project / "cMD_1" / payload["input"]["state"]).resolve()
    twin = original.parent / "twin_state.xml"
    shutil.copy2(original, twin)
    _edit_payload(project, lambda p: p["input"].update({"state": f"../eq/{twin.name}"}))
    _must_refuse(project, "a predecessor State path repointed at a copy")


@pytest.mark.slow
def test_5c_a_deleted_predecessor_state_is_refused_rather_than_ignored(committed_project,
                                                                       tmp_path):
    project = _fresh(committed_project, tmp_path)
    payload = json.loads((project / "cMD_1" / "cMD_1.json").read_text())
    (project / "cMD_1" / payload["input"]["state"]).resolve().unlink()
    _must_refuse(project, "a deleted predecessor State")


@pytest.mark.slow
def test_5d_a_changed_producer_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p["input"].update({"produced_by": "eq_npt_2"}))
    _must_refuse(project, "a changed predecessor producer")


@pytest.mark.slow
def test_6_an_equal_length_selection_change_is_refused(committed_project, tmp_path):
    """The count is unchanged, so only the ordered fingerprint can catch this."""
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p["reporting"].update(
        {"selected_atoms": {"type": "all"}}))
    _must_refuse(project, "a changed selection of equal length")


@pytest.mark.slow
@pytest.mark.parametrize("field,value", [
    ("temperature", "310 K"),
    ("timestep", "4 fs"),
    ("friction", "2 /ps"),
])
def test_7_integrator_changes_are_refused(committed_project, tmp_path, field, value):
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p["integrator"].update({field: value}))
    _must_refuse(project, f"a changed integrator {field}")


@pytest.mark.slow
def test_7b_adding_a_barostat_changes_the_ensemble_and_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p.update({"barostat": {"pressure": "1 bar"}}))
    _must_refuse(project, "an added barostat (NVT -> NPT)")


@pytest.mark.slow
def test_7c_adding_restraints_is_refused(committed_project, tmp_path):
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p.update(
        {"restraint": {"selection": "solute", "force_constant_kcal_per_mol_angstrom2": 1.0}}))
    _must_refuse(project, "added positional restraints")


@pytest.mark.slow
@pytest.mark.parametrize("cadence", [
    "full_system_interval_steps",
    "selected_atoms_interval_steps",
    "state_interval_steps",
])
def test_7d_every_reporting_cadence_is_refused(committed_project, tmp_path, cadence):
    project = _fresh(committed_project, tmp_path)
    payload = json.loads((project / "cMD_1" / "cMD_1.json").read_text())
    assert cadence in payload["reporting"], f"{cadence} is not in the payload: {payload['reporting']}"
    _edit_payload(project, lambda p: p["reporting"].update(
        {cadence: int(p["reporting"][cadence]) // 2 or 1}))
    _must_refuse(project, f"a changed {cadence}")


def test_7e_the_restraint_convention_is_part_of_the_contract():
    """cMD production is unrestrained, so this field is asserted at the contract level.

    It still has to be bound: the same project run under a build that measured restraint distance
    differently is a different Hamiltonian, and Finding 1 made that difference real.
    """
    from md_templates.openmm.cmd_segments import continuity_hash

    base = {"restraint_convention": "cartesian (nonperiodic)", "stage": "cMD_1"}
    moved = {**base, "restraint_convention": "minimum-image (periodicdistance)"}
    assert continuity_hash(base) != continuity_hash(moved)


@pytest.mark.slow
def test_8_segment_length_is_refused_but_segment_count_is_not(committed_project, tmp_path):
    """Length is the calculation. Count is how long you run it, and must not enter the hash."""
    project = _fresh(committed_project, tmp_path)
    _edit_payload(project, lambda p: p.update({"steps": int(p["steps"]) * 2}))
    _must_refuse(project, "a changed segment length")

    extended = _fresh(committed_project, tmp_path / "extend")
    before_hash = _committed(extended)["continuity_hash"]
    result = subprocess.run([str(extended / "cMD_1" / "cMD_1.sh")], capture_output=True, text=True,
                            cwd=str(extended / "cMD_1"),
                            env={**os.environ, "CMD_NUMBER_OF_SEGMENTS": "5"})
    assert result.returncode == 0, result.stderr[-600:]
    record = _committed(extended)
    assert record["invocations_completed"] == 2, "an extension did not add exactly one segment"
    assert record["continuity_hash"] == before_hash, (
        "asking for more segments changed the continuity hash; segment COUNT must stay out of it")
