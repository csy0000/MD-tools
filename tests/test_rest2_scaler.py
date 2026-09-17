"""Scaled Hamiltonians as a build step: `md_tools.build.scaler` and `md_tools.rest2.states`.

Step 2 of docs/amber-like-fix/REST2-scaler.md. `build-top --rest2-scaler` (step 4) is a surface over
`build_scaled_states`; what is tested here is what that surface will call:

  * `scaler.config`, resolved strictly (`SCALER_SCHEMA`);
  * the tau schedule, from the ONE `tau_ladder`, extended to `tau_min` without moving a single
    existing value;
  * `build/<method>/system_state<n>.xml`, which must be exactly the Systems `build_rung_systems`
    builds -- the same function a ladder has always integrated -- plus `scaler.yaml` and
    `scaler.log`;
  * where each non-standard residue's SDF comes from (§5);
  * the identity of a saved state, which every preflight and the AIS run record will read (§7).

PLATFORM_POLICY_EXEMPTION: System construction and serialisation only. No Context is created.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import make_dataset_root

app = pytest.importorskip("openmm.app")
elem = pytest.importorskip("openmm.app.element")


# --- fixtures ----------------------------------------------------------------------------------

def _dataset(root: Path, *, rename_alanine: str | None = None) -> Path:
    """`build/` with the ACE-ALA-NME stand-in System; optionally the alanine renamed."""
    build = make_dataset_root(root) / "build"
    if rename_alanine:
        lines = (build / "built.pdb").read_text(encoding="utf-8").splitlines(keepends=True)
        (build / "built.pdb").write_text("".join(
            line[:17] + rename_alanine + line[20:]
            if line.startswith(("ATOM", "HETATM")) and line[17:20] == "ALA" else line
            for line in lines), encoding="utf-8")
    return build


def _config(build: Path, text: str, name: str = "scaler.config") -> Path:
    path = build / name
    path.write_text(text, encoding="utf-8")
    return path


def _scale(build: Path, config: Path, **kwargs):
    from md_tools.build.scaler import build_scaled_states

    return build_scaled_states(system_path=build / "built.xml",
                               topology_path=build / "built.pdb",
                               config_path=config, echo=False, **kwargs)


# --- 1. the configuration ------------------------------------------------------------------------

def _resolve(text):
    from md_tools.build.scaler import SCALER_SCHEMA
    from md_tools.build.strict import load_yaml_strictly

    return SCALER_SCHEMA.resolve(load_yaml_strictly(text, source="test"))


def test_a_minimal_configuration_resolves_to_the_documented_defaults():
    resolved = _resolve("method: REST2\n")
    assert resolved["method"] == "REST2"
    assert resolved["schedule"] == {"kind": "linear", "n_states": 4,
                                    "tau_min": 0.0, "tau_max": 0.5}
    assert resolved["unscaled_torsions"] is True
    assert resolved["sdf_filelist"] is None
    assert resolved["proline_like_residues"] == ["PRO"]
    assert resolved["max_proline_ring_size"] == 7


@pytest.mark.parametrize("text, words", [
    ("schedule:\n  n_states: 4\n", "method"),
    ("method: REST2\nn_states: 4\n", "unknown key"),
    ("method: REMD\n", "not one of"),
    ("method: REST2\nschedule:\n  kind: geometric\n", "not one of"),
    ("method: REST2\nschedule:\n  n_states: 1\n  tau_min: 0.0\n  tau_max: 0.5\n", "tau_min"),
    ("method: REST2\nschedule:\n  n_states: 4\n  tau_min: 0.5\n  tau_max: 0.5\n", "tau_min"),
    ("method: cMD\nschedule:\n  n_states: 2\n  tau_min: 0.0\n  tau_max: 0.5\n", "cMD"),
    ("method: AIS\nschedule:\n  n_states: 3\n  tau_min: 0.0\n  tau_max: 0.5\n", "AIS"),
    ("method: REST2\nschedule:\n  tau_max: 1.0\n", "maximum"),
    ("method: REST2\nsdf_filelist: [MO1.sdf]\n", "sdf_filelist"),
])
def test_what_the_configuration_refuses(text, words):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match=words):
        _resolve(text)


# --- 2. the schedule -----------------------------------------------------------------------------

@pytest.mark.parametrize("n_states, tau_max", [(2, 0.5), (4, 0.5), (6, 0.6), (8, 0.95)])
def test_tau_min_zero_moves_no_existing_ladder_value(n_states, tau_max):
    """Every CV-enabled resume compares taus exactly; one digit moved breaks all of them."""
    from md_tools.remd.generated import tau_ladder

    step = float(tau_max) / (n_states - 1)
    legacy = [round(index * step, 6) for index in range(n_states)]
    assert tau_ladder(n_states, tau_max) == legacy
    assert tau_ladder(n_states, tau_max, tau_min=0.0) == legacy


def test_a_ladder_from_a_non_zero_tau_min_hits_both_ends():
    from md_tools.remd.generated import tau_ladder

    assert tau_ladder(4, 0.6, tau_min=0.3) == [0.3, 0.4, 0.5, 0.6]


# --- 3. the states and their record --------------------------------------------------------------

def test_the_states_are_exactly_what_a_ladder_integrates(tmp_path):
    import yaml
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import unscaled_torsions
    from md_tools.remd.protocol import build_rung_systems

    build = _dataset(tmp_path / "ALA")
    record = _scale(build, _config(build, "method: REST2\n"))

    out = build / "REST2"
    assert sorted(p.name for p in out.iterdir()) == [
        "protein-unscaled.png", "scaler.log", "scaler.yaml",
        "system_state0.xml", "system_state1.xml", "system_state2.xml", "system_state3.xml"]

    base = XmlSerializer.deserialize((build / "built.xml").read_text(encoding="utf-8"))
    topology = PDBFile(str(build / "built.pdb")).topology
    solute = solute_atom_indices(topology)
    excluded = [tuple(b) for b in unscaled_torsions(topology, solute)["unscaled_central_bonds"]]
    expected, _audit = build_rung_systems(base, solute, (0.0, 0.166667, 0.333333, 0.5),
                                          excluded_bonds=excluded)
    for index, system in enumerate(expected):
        written = (out / f"system_state{index}.xml").read_text(encoding="utf-8")
        assert written == XmlSerializer.serialize(system), f"state {index} differs"

    on_disk = yaml.safe_load((out / "scaler.yaml").read_text(encoding="utf-8"))
    assert on_disk == record
    assert [s["tau"] for s in on_disk["states"]] == [0.0, 0.166667, 0.333333, 0.5]
    assert on_disk["method"] == "REST2"
    assert on_disk["state0_is_physical"] is True
    section = on_disk["unscaled_torsions"]
    assert len(section["unscaled_central_bonds"]) == 2, "ACE-ALA-NME: two amides, no ring"
    assert section["counts"]["amide_omega"] == 2
    assert section["torsion_terms"]["n_excluded_torsions"] > 0
    assert section["counts"]["improper_terms"] == section["torsion_terms"]["n_unscaled_impropers"] > 0


def test_a_hot_cmd_state_is_one_file_at_its_tau(tmp_path):
    build = _dataset(tmp_path / "ALA")
    record = _scale(build, _config(build, "method: cMD\nschedule:\n  n_states: 1\n"
                                          "  tau_min: 0.5\n  tau_max: 0.5\n"))
    assert [p.name for p in sorted((build / "cMD").glob("*.xml"))] == ["system_state0.xml"]
    assert [s["tau"] for s in record["states"]] == [0.5]
    assert record["state0_is_physical"] is False


def test_an_ais_end_state_is_one_file_in_its_own_directory(tmp_path):
    from md_tools.rest2.states import scaled_state_identity

    build = _dataset(tmp_path / "ALA")
    _scale(build, _config(build, "method: AIS\nschedule:\n  n_states: 1\n"
                                 "  tau_min: 0.5\n  tau_max: 0.5\n"))
    identity = scaled_state_identity(build / "AIS" / "system_state0.xml")
    assert identity["method"] == "AIS" and identity["tau"] == 0.5


def test_a_ladder_that_starts_scaled_says_so_in_words(tmp_path):
    build = _dataset(tmp_path / "ALA")
    record = _scale(build, _config(build, "method: REST2\nschedule:\n  n_states: 3\n"
                                          "  tau_min: 0.2\n  tau_max: 0.6\n"))
    assert record["state0_is_physical"] is False
    log = (build / "REST2" / "scaler.log").read_text(encoding="utf-8")
    assert "NOT the physical Hamiltonian" in log


def test_an_unclassifiable_residue_is_refused_before_the_directory_exists(tmp_path):
    from md_tools.build.strict import ConfigError

    build = _dataset(tmp_path / "XAA", rename_alanine="XAA")
    with pytest.raises(ConfigError, match="XAA"):
        _scale(build, _config(build, "method: REST2\n"))
    assert not (build / "REST2").exists()


def test_unscaled_torsions_false_classifies_nothing_and_says_so(tmp_path):
    build = _dataset(tmp_path / "XAA", rename_alanine="XAA")
    record = _scale(build, _config(build, "method: REST2\nunscaled_torsions: false\n"))
    section = record["unscaled_torsions"]
    assert section["enabled"] is False and section["unscaled_impropers"] is False
    assert section["unscaled_central_bonds"] == []
    assert section["torsion_terms"]["n_unscaled_impropers"] == 0
    assert "impropers and ordinary amide omegas included" in section["method"]


def test_check_creates_nothing(tmp_path):
    build = _dataset(tmp_path / "ALA")
    config = _config(build, "method: REST2\n")
    before = sorted(p.name for p in build.iterdir())
    _scale(build, config, check=True)
    assert sorted(p.name for p in build.iterdir()) == before


def test_an_existing_directory_is_refused_and_overwrite_moves_it_aside(tmp_path):
    from md_tools.build.strict import ConfigError

    build = _dataset(tmp_path / "ALA")
    config = _config(build, "method: REST2\n")
    first = _scale(build, config)
    with pytest.raises(ConfigError, match="--overwrite"):
        _scale(build, config)
    assert (build / "REST2" / "scaler.yaml").is_file()

    _config(build, "method: REST2\nschedule:\n  n_states: 3\n")
    second = _scale(build, config, overwrite=True)
    assert len(second["states"]) == 3 and len(first["states"]) == 4
    aside = [p for p in build.iterdir() if p.name.startswith(".REST2.replaced-")]
    assert len(aside) == 1, "the replaced states are moved aside, never deleted"
    assert len(list(aside[0].glob("system_state*.xml"))) == 4


# --- 4. which SDF describes which residue ---------------------------------------------------------

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
ACETYLPYRROLIDINE = "CC(=O)N1CCCC1"
_SYMBOLS = {"C": elem.carbon, "N": elem.nitrogen, "O": elem.oxygen, "H": elem.hydrogen}


def _ligands(directory, residues, *, write=True):
    rdkit = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    topology = app.Topology()
    for name, smiles in residues:
        mol = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=20260916)
        residue = topology.addResidue(name, topology.addChain())
        atoms = [topology.addAtom(a.GetSymbol(), _SYMBOLS[a.GetSymbol()], residue)
                 for a in mol.GetAtoms()]
        for bond in mol.GetBonds():
            topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
        if write:
            rdkit.MolToMolFile(mol, str(directory / f"{name}.sdf"))
    return topology, [a.index for a in topology.atoms()]


def _resolve_sdfs(topology, solute, directory, filelist=None, config_dir=None, **kwargs):
    from md_tools.build.scaler import resolve_residue_sdfs

    return resolve_residue_sdfs(topology, solute, system_dir=directory,
                                config_dir=config_dir or directory, sdf_filelist=filelist,
                                proline_like_residues=kwargs.get("proline_like", ["PRO"]))


def test_each_name_is_found_as_its_own_sdf_by_default(tmp_path):
    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    found = _resolve_sdfs(topology, solute, tmp_path)
    assert found == {"MO1": tmp_path / "MO1.sdf", "MO2": tmp_path / "MO2.sdf"}


def test_a_filelist_resolves_relative_to_the_configuration(tmp_path):
    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL)])
    elsewhere = tmp_path / "configs"
    elsewhere.mkdir()
    (tmp_path / "MO1.sdf").rename(elsewhere / "first.sdf")
    found = _resolve_sdfs(topology, solute, tmp_path, filelist={"MO1": "first.sdf"},
                          config_dir=elsewhere)
    assert found == {"MO1": elsewhere / "first.sdf"}


def test_a_filelist_naming_a_residue_that_is_not_there_is_refused(tmp_path):
    from md_tools.build.strict import ConfigError

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL)])
    with pytest.raises(ConfigError, match="MO9"):
        _resolve_sdfs(topology, solute, tmp_path,
                      filelist={"MO1": "MO1.sdf", "MO9": "MO1.sdf"})


def test_a_missing_sdf_is_refused_naming_every_path_looked_for(tmp_path):
    from md_tools.build.strict import ConfigError

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)],
                                write=False)
    with pytest.raises(ConfigError) as refused:
        _resolve_sdfs(topology, solute, tmp_path)
    message = str(refused.value)
    assert "MO1" in message and "MO2" in message and "MO1.sdf" in message


def test_built_sdf_stands_in_only_for_a_single_non_standard_residue(tmp_path):
    from md_tools.build.strict import ConfigError

    one = tmp_path / "one"
    one.mkdir()
    topology, solute = _ligands(one, [("UNL", PARACETAMOL)])
    (one / "UNL.sdf").rename(one / "built.sdf")
    assert _resolve_sdfs(topology, solute, one) == {"UNL": one / "built.sdf"}

    two = tmp_path / "two"
    two.mkdir()
    topology, solute = _ligands(two, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    (two / "MO1.sdf").rename(two / "built.sdf")
    with pytest.raises(ConfigError, match="MO1"):
        _resolve_sdfs(topology, solute, two)


def test_a_declared_proline_like_residue_needs_no_sdf(tmp_path):
    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    (tmp_path / "MO2.sdf").unlink()
    found = _resolve_sdfs(topology, solute, tmp_path, proline_like=["PRO", "MO2"])
    assert found == {"MO1": tmp_path / "MO1.sdf"}


# --- 5. the identity of a saved state -------------------------------------------------------------

def test_every_saved_state_knows_what_it_is(tmp_path):
    from md_tools.rest2.states import scaled_state_identity

    build = _dataset(tmp_path / "ALA")
    record = _scale(build, _config(build, "method: REST2\n"))
    for state in record["states"]:
        identity = scaled_state_identity(build / "REST2" / state["file"])
        assert identity == {"record": str((build / "REST2" / "scaler.yaml").resolve()),
                            "method": "REST2", "state": state["state"], "tau": state["tau"],
                            "system_sha256": state["sha256"]}


def test_an_unscaled_system_has_no_identity(tmp_path):
    from md_tools.rest2.states import scaled_state_identity

    build = _dataset(tmp_path / "ALA")
    _scale(build, _config(build, "method: REST2\n"))
    assert scaled_state_identity(build / "built.xml") is None


def test_a_state_edited_after_the_build_is_refused(tmp_path):
    from md_tools.rest2.states import ScaledStateError, scaled_state_identity

    build = _dataset(tmp_path / "ALA")
    _scale(build, _config(build, "method: REST2\n"))
    target = build / "REST2" / "system_state2.xml"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ScaledStateError, match="sha256"):
        scaled_state_identity(target)


def test_a_system_the_record_does_not_list_is_refused(tmp_path):
    from md_tools.rest2.states import ScaledStateError, scaled_state_identity

    build = _dataset(tmp_path / "ALA")
    _scale(build, _config(build, "method: REST2\n"))
    stray = build / "REST2" / "system_state9.xml"
    stray.write_bytes((build / "built.xml").read_bytes())
    with pytest.raises(ScaledStateError, match="system_state9.xml"):
        scaled_state_identity(stray)


# --- 6. a picture of what is left unscaled --------------------------------------------------------

def _red_pixels(path):
    image = pytest.importorskip("PIL.Image").open(path).convert("RGB")
    return sum(1 for r, g, b in image.getdata() if r > 200 and g < 80 and b < 80)


def _unscaled_bonds(topology, solute, sdfs):
    from md_tools.openmm.system import unscaled_torsions

    return [tuple(b) for b in unscaled_torsions(topology, solute,
                                               residue_sdfs=sdfs)["unscaled_central_bonds"]]


def test_each_small_molecule_gets_a_picture_with_its_unscaled_bond_in_red(tmp_path):
    from md_tools.build.scaler import depict_unscaled_torsions, resolve_residue_sdfs

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    sdfs = resolve_residue_sdfs(topology, solute, system_dir=tmp_path, config_dir=tmp_path,
                                sdf_filelist=None, proline_like_residues=["PRO"])
    out = tmp_path / "out"
    out.mkdir()
    drawn = depict_unscaled_torsions(topology, solute, sdfs,
                                     _unscaled_bonds(topology, solute, sdfs), out)

    assert sorted(drawn) == ["MO1", "MO2"]
    assert drawn["MO1"]["file"] == "MO1-unscaled.png"
    for name in ("MO1", "MO2"):
        assert (out / f"{name}-unscaled.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(drawn["MO1"]["unscaled_bonds"]) == 7, "paracetamol's amide and six ring bonds"
    assert drawn["MO2"]["unscaled_bonds"] == [], "acetylpyrrolidine's amide is proline-like: scaled"
    assert _red_pixels(out / "MO1-unscaled.png") > 50, "the unscaled bond must be drawn red"
    assert _red_pixels(out / "MO2-unscaled.png") == 0, (
        "nothing is unscaled, so nothing may be red -- and no atom may be drawn red either")


def test_the_caption_names_the_bonds_by_the_records_indices(tmp_path):
    from md_tools.build.scaler import depict_unscaled_torsions

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL)])
    sdfs = {"MO1": tmp_path / "MO1.sdf"}
    bonds = _unscaled_bonds(topology, solute, sdfs)
    drawn = depict_unscaled_torsions(topology, solute, sdfs, bonds, tmp_path)
    assert drawn["MO1"]["unscaled_bonds"] == sorted([list(b) for b in bonds])
    for a, b in bonds:
        assert f"{a}-{b}" in drawn["MO1"]["caption"]


def test_unscaled_torsions_off_draws_nothing_red_and_says_why(tmp_path):
    from md_tools.build.scaler import depict_unscaled_torsions

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL)])
    drawn = depict_unscaled_torsions(topology, solute, {"MO1": tmp_path / "MO1.sdf"}, [],
                                     tmp_path, enabled=False)
    assert "OFF" in drawn["MO1"]["caption"]
    assert _red_pixels(tmp_path / "MO1-unscaled.png") == 0


def test_an_sdf_that_is_not_this_residue_is_not_drawn(tmp_path):
    """A picture of the wrong molecule would be worse than none."""
    from md_tools.build.scaler import depict_unscaled_torsions

    topology, solute = _ligands(tmp_path, [("MO1", PARACETAMOL), ("MO2", ACETYLPYRROLIDINE)])
    drawn = depict_unscaled_torsions(topology, solute, {"MO1": tmp_path / "MO2.sdf"}, [],
                                     tmp_path)
    assert "MO1" not in drawn
    assert not (tmp_path / "MO1-unscaled.png").exists()


def test_a_peptide_gets_a_picture_of_its_protein_residues(tmp_path):
    """ACE-ALA-NME is classified from the residue table and has no SDF, so the small-molecule
    picture could never draw it. It used to get NO picture; a capped peptide is exactly what a
    tutorial shows first, and its two amide omegas are the torsions a reader wants to see."""
    build = _dataset(tmp_path / "ALA")
    record = _scale(build, _config(build, "method: REST2\n"))
    section = record["unscaled_torsions"]
    picture = section["depictions"]["protein"]
    assert picture["file"] == "protein-unscaled.png"
    assert picture["residues"] == ["ACE", "ALA", "NME"]
    assert sorted(map(tuple, picture["unscaled_bonds"])) == sorted(
        map(tuple, section["unscaled_central_bonds"])), "the picture draws exactly the record's bonds"
    path = build / "REST2" / "protein-unscaled.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert _red_pixels(path) > 50, "the amide omega bonds must be drawn red"
    for a, b in picture["unscaled_bonds"]:
        assert f"{a}-{b}" in picture["caption"]


def test_a_protein_too_large_to_read_is_not_drawn_and_says_why(tmp_path):
    from openmm.app import PDBFile

    from md_tools.build.scaler import depict_protein_unscaled
    from md_tools.md.stage import solute_atom_indices

    build = _dataset(tmp_path / "ALA")
    topology = PDBFile(str(build / "built.pdb")).topology
    drawn = depict_protein_unscaled(topology, solute_atom_indices(topology), [], tmp_path,
                                    max_heavy_atoms=3)
    assert "above 3" in drawn["protein"]["skipped"]
    assert not (tmp_path / "protein-unscaled.png").exists()


@pytest.mark.slow
def test_a_built_small_molecule_gets_its_picture_listed_in_the_record(tmp_path):
    """End to end on what `build-top` really writes: paracetamol from SMILES, explicit water."""
    import subprocess
    import sys

    build = tmp_path / "PARA" / "build"
    build.mkdir(parents=True)
    (build / "in.smi").write_text(f"{PARACETAMOL} paracetamol\n", encoding="utf-8")
    (build / "sys.config").write_text("solute:\n  kind: ligand\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", "in.smi",
         "-os", "built.xml", "-op", "built.pdb", "-log", "built.log", "--config", "sys.config"],
        cwd=build, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    record = _scale(build, _config(build, "method: REST2\n"))
    (name, facts), = record["unscaled_torsions"]["depictions"].items()
    png = build / "REST2" / facts["file"]
    assert png.name == f"{name}-unscaled.png"
    from md_tools.build.record import sha256_file
    assert sha256_file(png) == facts["sha256"]
    assert _red_pixels(png) > 50
