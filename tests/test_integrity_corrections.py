"""The seven integrity corrections, each with evidence rather than an assertion about intent.

What these guard is a class of failure the previous round could not catch: a record that LOOKS
complete. A header that claims the right frame count over missing bytes, a hash string that is
present but was never recompared, a force-field field that is absent rather than wrong, a 40-hex
commit that is syntactically valid and names nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_templates.openmm import md_data_contract as MD

from .conftest import ALA_PDB, REPO_ROOT, run_cli, template_module

AIS_SOURCE = REPO_ROOT / "src" / "md_templates" / "openmm" / "templates" / "ais_run.py"
PREFLIGHT_SOURCE = REPO_ROOT / "src" / "md_templates" / "openmm" / "templates" / "preflight.py"


# --- 1. the production source is never hashed ---------------------------------------------------

def test_the_ais_runtime_never_hashes_the_production_source():
    """A full-file digest of a production trajectory costs more than it proves.

    The whole point of the bounded `iterload` survey is that cost must not scale with the length
    of the source; hashing it puts that back.
    """
    source = AIS_SOURCE.read_text()
    prepared = source[source.index("def write_prepared_sources"):
                      source.index("def read_prepared_sources")]
    assert "sha256_file" not in prepared, "write_prepared_sources hashes something"
    assert "hashlib" not in prepared
    assert '"trajectory_sha256": None' in prepared
    assert "trajectory_bytes" in prepared


def test_the_source_record_carries_bounded_observations_not_a_digest(prepared_ais):
    record = yaml.safe_load((prepared_ais / "inputs" / "sources.yaml").read_text())["source"]
    assert record["trajectory_sha256"] is None
    assert "never hashed" in record["trajectory_not_hashed"]
    # Bounded observations, every one of them obtainable without reading the file whole.
    assert record["trajectory_bytes"] > 0
    assert record["n_frames"] == 120
    assert record["loader"] == "mdtraj.iterload"
    assert record["chunk_frames"] == 50
    assert record["chunks_read_survey"] >= 3
    assert record["first_time_ps"] > 0 and record["frame_timing"]["frame_interval_ps"] > 0
    assert record["eligible_frames"] == 120
    assert record["tau"] == 0.5 and record["tau_route"]


#: The subprocess guard. Injected as a `sitecustomize` on PYTHONPATH so it is installed before the
#: generated `run.py` imports anything.
#:
#: It wraps `pathlib.Path.open`, NOT `builtins.open`. The generated `sha256_file()` does
#: `Path(path).open("rb")`, and `pathlib.Path.open` delegates to `io.open` -- which is a separate
#: reference from `builtins.open`, so rebinding the builtin intercepts nothing. The earlier version
#: of this test patched the builtin and therefore proved nothing at all: it would have passed with
#: the production source hashed on every run.
#:
#: Path-specific and caller-specific: the forbidden trajectory may be opened freely by
#: `mdtraj.iterload`, and is rejected only when the call comes from `sha256_file`. Named small
#: files still hash normally.
_HASH_GUARD = '''
import pathlib
import traceback

FORBIDDEN = pathlib.Path(%r).resolve()
MARK = pathlib.Path(__file__).with_name("_guard")
MARK.write_text("installed")

_real_open = pathlib.Path.open


def _guarded_open(self, *args, **kwargs):
    try:
        same = self.resolve() == FORBIDDEN
    except Exception:
        same = False
    if same:
        # Only hashing is forbidden. iterload opens the same file and must keep working.
        frames = traceback.extract_stack()
        if any(frame.name == "sha256_file" for frame in frames):
            MARK.write_text("TRIGGERED")
            raise AssertionError("the production source was passed to sha256_file")
    return _real_open(self, *args, **kwargs)


pathlib.Path.open = _guarded_open
'''


@pytest.mark.gpu
@pytest.mark.slow
def test_the_production_source_is_never_passed_to_the_hashing_helper(tiny_ais_project):
    """PATH-SPECIFIC and CALLER-SPECIFIC, and it intercepts the route the code actually takes.

    Two halves, with the SAME guard, because a guard that never fires is indistinguishable from a
    guard that does not work:

      1. the generated `sha256_file` is applied to the production trajectory deliberately -- the
         guard must fire and leave `TRIGGERED`;
      2. a real AIS preparation runs -- the guard must NOT fire, `sources.dcd` must be produced,
         and `trajectory_sha256` must stay null.
    """
    project, environment = tiny_ais_project
    source = (project / "cMD" / "whole_system.dcd").resolve()
    assert source.is_file()

    ais = project / "AIS"
    guard = ais / "sitecustomize.py"
    mark = ais / "_guard"
    guard.write_text(_HASH_GUARD % str(source))
    patched = dict(environment, PYTHONPATH=str(ais))

    try:
        # 1. The guard fires on the real generated helper, reached the way the runtime reaches it.
        probe = subprocess.run(
            [sys.executable, "-c",
             "import runpy, pathlib, sys\n"
             "module = runpy.run_path('run.py', run_name='_probe')\n"
             "try:\n"
             "    module['sha256_file'](pathlib.Path(%r))\n"
             "except AssertionError as error:\n"
             "    print('GUARD:', error)\n"
             "    sys.exit(0)\n"
             "print('NOT GUARDED')\n"
             "sys.exit(1)\n" % str(source)],
            cwd=str(ais), capture_output=True, text=True, env=patched, timeout=600)
        assert mark.is_file(), "the guard never installed"
        assert probe.returncode == 0, probe.stdout + probe.stderr
        assert "GUARD: the production source was passed to sha256_file" in probe.stdout
        assert mark.read_text() == "TRIGGERED", "the guard did not intercept Path.open"

        # 2. The same guard, and a real preparation that must not touch it.
        mark.write_text("installed")
        shutil.rmtree(ais / "inputs")
        result = subprocess.run(["bash", "run.sh"], cwd=str(ais), capture_output=True, text=True,
                                env=patched, timeout=1800)
        assert result.returncode == 0, result.stdout + result.stderr
        assert mark.read_text() == "installed", "AIS preparation hashed the production source"
        assert "passed to sha256_file" not in result.stdout + result.stderr

        record = yaml.safe_load((ais / "inputs" / "sources.yaml").read_text())
        assert record["n_paths"] == 2
        assert record["source"]["loader"] == "mdtraj.iterload"
        assert record["source"]["trajectory_sha256"] is None
        assert (ais / "inputs" / "sources.dcd").is_file()
        assert struct.unpack("<i", (ais / "inputs" / "sources.dcd").read_bytes()[8:12])[0] == 2
    finally:
        guard.unlink(missing_ok=True)
        mark.unlink(missing_ok=True)
        shutil.rmtree(ais / "__pycache__", ignore_errors=True)


def test_patching_builtins_open_would_not_have_intercepted_path_open():
    """Why the previous guard proved nothing, kept as a regression against reintroducing it."""
    import builtins
    import io

    real = builtins.open
    seen = []
    builtins.open = lambda *a, **k: (seen.append(1), real(*a, **k))[1]
    try:
        with Path(__file__).open("rb") as handle:
            handle.read(1)
        assert not seen, "Path.open went through builtins.open after all"
        assert io.open is not builtins.open
    finally:
        builtins.open = real


def test_the_hashing_guard_would_actually_fire(tmp_path):
    """The guard is only evidence if it triggers when the thing it forbids happens.

    Without this, a guard that silently failed to match the path would look exactly like a guard
    that was never triggered -- which is the flaw in the test this replaces.
    """
    ais = template_module_with_stubs()
    target = tmp_path / "whole_system.dcd"
    target.write_bytes(b"not really a dcd")
    small = tmp_path / "solute.yaml"
    small.write_text("n_solute_atoms: 22\n")

    forbidden = target.resolve()
    real = ais.sha256_file

    def guarded(path):
        if Path(path).resolve() == forbidden:
            raise AssertionError("the production source was passed to sha256_file")
        return real(path)

    # A named small file still hashes.
    assert len(guarded(small)) == 64
    # The production source does not.
    with pytest.raises(AssertionError, match="production source"):
        guarded(target)


# --- 2. physical truncation, with the header left alone -----------------------------------------

def _truncate_bytes(path, *, drop):
    """Remove bytes from the END of a DCD without touching its header.

    This is the case a header check cannot see: `NSET` still says what the writer intended, and
    the coordinate record it points at is short or gone.
    """
    raw = path.read_bytes()
    header_frames = struct.unpack("<i", raw[8:12])[0]
    path.write_bytes(raw[:-drop])
    assert struct.unpack("<i", path.read_bytes()[8:12])[0] == header_frames, \
        "the truncation changed the header, which is not the case under test"
    return header_frames


@pytest.mark.gpu
@pytest.mark.slow
def test_a_physically_truncated_observations_dcd_is_refused_and_replaced(tiny_ais_project):
    project, environment = tiny_ais_project
    directory = project / "AIS" / "trajectory_0000"
    dcd = directory / "observations.dcd"
    intact = dcd.read_bytes()
    claimed = _truncate_bytes(dcd, drop=len(intact) // 4)
    assert claimed == 21, "the header must still claim the full count"

    result = subprocess.run(["bash", "run.sh"], cwd=str(project / "AIS"), capture_output=True,
                            text=True, env=environment, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "replacing an incomplete directory" in combined
    assert "truncated, not short" in combined or "could be read before" in combined
    # Replaced, never appended: the rerun must not have been added onto the short file.
    assert len(dcd.read_bytes()) == len(intact)
    assert struct.unpack("<i", dcd.read_bytes()[8:12])[0] == 21
    assert "already complete" in combined, "the intact path should have been left alone"


@pytest.mark.gpu
@pytest.mark.slow
def test_a_physically_truncated_sources_dcd_is_refused(tiny_ais_project):
    project, environment = tiny_ais_project
    dcd = project / "AIS" / "inputs" / "sources.dcd"
    intact = dcd.read_bytes()
    _truncate_bytes(dcd, drop=len(intact) // 3)
    try:
        result = subprocess.run(["bash", "run.sh", "--check"], cwd=str(project / "AIS"),
                                capture_output=True, text=True, env=environment, timeout=600)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] AIS inputs" in combined
        assert "no Context was created" in combined
    finally:
        dcd.write_bytes(intact)


def test_the_header_count_is_documented_as_insufficient():
    source = AIS_SOURCE.read_text()
    assert "def dcd_header_frames" in source and "def validate_generated_dcd" in source
    import re

    assert not re.findall(r"mdtraj\.load\s*\(", source), "the AIS runtime calls mdtraj.load"
    # The validator must actually read, not just re-read the header.
    validator = source[source.index("def validate_generated_dcd"):
                       source.index("def path_is_complete")]
    assert "mdtraj.iterload" in validator and "isfinite" in validator


# --- 3. route-aware, exact force-field preflight ------------------------------------------------

def test_an_absent_expected_field_does_not_pass(tmp_path):
    """The old comparison skipped when either side was missing, which is when it mattered most."""
    preflight = template_module("preflight")
    inputs = tmp_path / "common"
    inputs.mkdir()
    (inputs / "resolved_sys.config.yaml").write_text(yaml.safe_dump({
        "solvation": "explicit", "solute": {"peptide": True},
        "forcefield": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"}}))
    (inputs / "forcefield.json").write_text(json.dumps({
        "solvation": "explicit", "protein": {"openmm_resource": None},
        "water": {"openmm_resource": "amber14/tip3p.xml"}, "ligand": {}}))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL
    assert "records no the protein force field" in row.detail or "records no" in row.detail


@pytest.mark.parametrize("mutate, expected", [
    (lambda b, w: b["protein"].update(openmm_resource="amber19-all.xml"), "amber19-all.xml"),
    (lambda b, w: b["water"].update(openmm_resource="amber14/tip3pfb.xml"), "tip3pfb"),
    (lambda b, w: b.update(ligand={"openff_resource": "openff-2.2.1"}), "ligand"),
])
def test_the_peptide_route_refuses_a_wrong_or_extra_force_field(tmp_path, mutate, expected):
    preflight = template_module("preflight")
    inputs = tmp_path / "common"
    inputs.mkdir()
    wanted = {"solvation": "explicit", "solute": {"peptide": True},
              "forcefield": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"}}
    built = {"solvation": "explicit",
             "protein": {"openmm_resource": "amber14-all.xml"},
             "water": {"openmm_resource": "amber14/tip3p.xml"}, "ligand": {}}
    mutate(built, wanted)
    (inputs / "resolved_sys.config.yaml").write_text(yaml.safe_dump(wanted))
    (inputs / "forcefield.json").write_text(json.dumps(built))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL, row.detail
    assert expected in row.detail


@pytest.mark.parametrize("mutate, expected", [
    (lambda b: b["ligand"].update(openff_resource="openff-2.1.0"), "openff-2.1.0"),
    (lambda b: b["ligand"].update(charge_method="am1bcc_nagl"), "charge method"),
    (lambda b: b["protein"].update(openmm_resource="amber14-all.xml"), "protein force field"),
])
def test_the_ligand_route_refuses_a_wrong_or_extra_force_field(tmp_path, mutate, expected):
    preflight = template_module("preflight")
    inputs = tmp_path / "common"
    inputs.mkdir()
    wanted = {"solvation": "explicit",
              "solute": {"peptide": False, "ligand_forcefield": "sage-2.2.1",
                         "ligand_charge_method": "am1bcc"},
              "forcefield": {"water": "amber14/tip3p.xml", "ligand": "openff-2.2.1",
                             "ligand_charge_method": "am1bcc"}}
    built = {"solvation": "explicit", "protein": {},
             "ligand": {"openff_resource": "openff-2.2.1", "charge_method": "am1bcc"},
             "water": {"openmm_resource": "amber14/tip3p.xml"}}
    mutate(built)
    (inputs / "resolved_sys.config.yaml").write_text(yaml.safe_dump(wanted))
    (inputs / "forcefield.json").write_text(json.dumps(built))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL, row.detail
    assert expected in row.detail


def test_an_implicit_system_may_claim_neither_water_nor_a_barostat(tmp_path):
    preflight = template_module("preflight")
    inputs = tmp_path / "common"
    inputs.mkdir()
    wanted = {"solvation": "implicit", "solute": {"peptide": True},
              "forcefield": {"protein": "amber14-all.xml", "water": None},
              "implicit_solvent": {"model": "GBn2", "radii": "mbondi3", "nonpolar_sasa": False}}
    built = {"solvation": "implicit", "protein": {"openmm_resource": "amber14-all.xml"},
             "ligand": {}, "water": {"openmm_resource": "amber14/tip3p.xml"},
             "implicit_solvent": {"model": "GBn2", "radii": "mbondi3", "nonpolar_sasa": False}}
    (inputs / "resolved_sys.config.yaml").write_text(yaml.safe_dump(wanted))
    (inputs / "forcefield.json").write_text(json.dumps(built))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL
    assert "water model" in row.detail and "no box" in row.detail

    built["water"] = {"openmm_resource": None}
    built["implicit_solvent"]["nonpolar_sasa"] = True
    (inputs / "forcefield.json").write_text(json.dumps(built))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL
    assert "nonpolar surface-area term" in row.detail


def test_the_route_itself_must_agree(tmp_path):
    preflight = template_module("preflight")
    inputs = tmp_path / "common"
    inputs.mkdir()
    (inputs / "resolved_sys.config.yaml").write_text(yaml.safe_dump({
        "solvation": "explicit", "solute": {"peptide": True},
        "forcefield": {"protein": "amber14-all.xml", "water": "amber14/tip3p.xml"}}))
    (inputs / "forcefield.json").write_text(json.dumps({
        "solvation": "explicit", "protein": {},
        "ligand": {"openff_resource": "openff-2.2.1", "charge_method": "am1bcc"},
        "water": {"openmm_resource": "amber14/tip3p.xml"}}))
    row, = preflight.check_forcefield(inputs)
    assert row.status == preflight.FAIL
    assert "a ligand route" in row.detail and "a peptide one" in row.detail


# --- 5. the template commit is evidence, not a well-formed string -------------------------------

def test_the_generator_commit_is_established_not_invented():
    established = MD.generator_commit()
    assert established["route"] in ("git checkout", "direct_url.json", "unavailable")
    if established["commit"] is not None:
        assert MD.COMMIT.match(established["commit"])
        # Never derived from a version, a branch or a date.
        assert established["commit"] != MD.MD_DATA_COMMIT


def test_a_well_formed_but_wrong_template_commit_is_refused(monkeypatch):
    monkeypatch.setattr(MD, "generator_commit", lambda: {
        "commit": "a" * 40, "route": "git checkout", "dirty": False, "detail": "test"})
    established = MD.check_templates_commit("a" * 40)
    assert established["commit"] == "a" * 40
    with pytest.raises(MD.ContractError) as error:
        MD.check_templates_commit("b" * 40)
    assert "actually generating this dataset is at" in str(error.value)
    assert "worse than none" in str(error.value)


def test_an_unavailable_generator_identity_refuses_rather_than_accepting(monkeypatch):
    monkeypatch.setattr(MD, "generator_commit", lambda: {
        "commit": None, "route": "unavailable", "dirty": None,
        "detail": "no checkout and no direct_url.json"})
    with pytest.raises(MD.ContractError) as error:
        MD.check_templates_commit("c" * 40)
    assert "cannot establish its own exact commit" in str(error.value)
    assert "refuses rather than record an unverified pin" in str(error.value)


# Preflight's provenance comparison is exercised against REAL generated projects in
# tests/test_template_provenance.py -- matching, missing and mismatching records, for both the
# clean-checkout and direct_url.json identity routes. A hand-built stub of the record set was the
# weaker form of the same check and is deliberately not kept beside it.


# --- 6. atom identity and order, not names and counts -------------------------------------------

def test_repeated_atom_names_do_not_make_two_topologies_the_same(tmp_path):
    """Every residue has an N, a CA, a C and an O. Names alone cannot tell residues apart."""
    ais = template_module_with_stubs()
    from openmm.app import PDBFile

    original = PDBFile(str(ALA_PDB))
    atoms, bonds = ais.atom_identity(original.topology)
    assert len(atoms) == original.topology.getNumAtoms()
    # The identity tuple is not just the name: a residue index and a residue name are in it.
    names = [a[5] for a in atoms]
    assert len(set(names)) < len(names), "this fixture must contain repeated atom names"
    assert len(set(atoms)) == len(atoms), "full identity tuples must be unique"
    assert bonds and all(a < b for a, b in bonds)


def test_the_identity_tuple_covers_chain_residue_and_element():
    source = AIS_SOURCE.read_text()
    block = source[source.index("def atom_identity"):source.index("ATOM_FIELDS")]
    for field in ("chain.index", "residue.index", "residue.name", "atom.name",
                  "atom.element.symbol"):
        assert field in block, field
    assert "topology.bonds()" in block


def template_module_with_stubs():
    """`ais_run.py` reads a generated project's configuration at import time.

    The pure functions in it are still worth testing directly, so the module is loaded with the
    handful of files it opens faked out. This is the only way to unit-test them without generating
    and running a whole project.
    """
    import importlib.util
    import types

    path = REPO_ROOT / "src" / "md_templates" / "openmm" / "templates" / "ais_run.py"
    source = path.read_text()
    # Everything above the first function definition is project-configuration loading.
    cut = source.index("def relative_to_project")
    module = types.ModuleType("_ais_pure")
    module.__dict__.update({name: __import__(name) for name in ("hashlib", "csv", "json", "yaml")})
    module.__dict__.update({"Path": __import__("pathlib").Path, "np": __import__("numpy")})
    exec(compile(source[cut:], str(path), "exec"), module.__dict__)
    return module


def _renumber_residue(pdb_text, *, old_seq, new_seq, new_name=None):
    """Move atoms into a different residue WITHOUT touching a single atom name.

    Columns 18-20 are resName and 23-26 are resSeq. Every atom keeps its name and its order, so a
    check that compares names and counts sees two identical topologies.
    """
    out = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[22:26].strip() == str(old_seq):
            name = new_name if new_name else line[17:20]
            line = line[:17] + f"{name:>3}" + line[20:22] + f"{new_seq:>4}" + line[26:]
        out.append(line)
    return "\n".join(out) + "\n"


def test_a_reassigned_residue_is_caught_though_every_atom_name_is_identical(tmp_path):
    """The case atom names cannot see: same names, same order, different molecule."""
    from openmm.app import PDBFile

    ais = template_module_with_stubs()
    original_text = ALA_PDB.read_text()
    original = PDBFile(str(ALA_PDB))

    mutated_path = tmp_path / "reassigned.pdb"
    mutated_path.write_text(_renumber_residue(original_text, old_seq=2, new_seq=1))
    mutated = PDBFile(str(mutated_path))

    mine, my_bonds = ais.atom_identity(original.topology)
    theirs, their_bonds = ais.atom_identity(mutated.topology)

    # The premise of the test: names and counts are unchanged, so the old check would have passed.
    assert [a[5] for a in mine] == [a[5] for a in theirs]
    assert len(mine) == len(theirs)
    # The identity tuples are not.
    assert mine != theirs
    first = next(i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
    differing = [ais.ATOM_FIELDS[i] for i, (a, b) in enumerate(zip(mine[first], theirs[first]))
                 if a != b]
    assert "residue index" in differing or "residue id" in differing


def test_changed_connectivity_is_caught(tmp_path):
    """Same atoms in the same order is not enough: different bonds is a different molecule."""
    from openmm.app import PDBFile

    ais = template_module_with_stubs()
    mutated_path = tmp_path / "renamed.pdb"
    mutated_path.write_text(_renumber_residue(ALA_PDB.read_text(), old_seq=2, new_seq=2,
                                              new_name="GLY"))
    original = PDBFile(str(ALA_PDB))
    mutated = PDBFile(str(mutated_path))

    mine, my_bonds = ais.atom_identity(original.topology)
    theirs, their_bonds = ais.atom_identity(mutated.topology)
    assert [a[5] for a in mine] == [a[5] for a in theirs], "atom names must be unchanged"
    # OpenMM infers bonds from the residue template, so renaming the residue changes them.
    assert my_bonds != their_bonds


@pytest.mark.gpu
@pytest.mark.slow
def test_ais_refuses_a_source_topology_whose_residues_were_reassigned(tiny_ais_project, tmp_path):
    """End to end: the refusal happens before frame selection or any worker is spawned."""
    project, environment = tiny_ais_project
    prepared = project / "AIS" / "inputs"
    kept = {path.name: path.read_bytes() for path in prepared.iterdir()}

    # A source topology with every atom name and the atom order unchanged, and one residue
    # reassigned. `AIS.source.topology` is the supported way to name a different one.
    mutated = project / "AIS" / "reassigned.pdb"
    mutated.write_text(_renumber_residue((project.parent / "inputs" / "topology.pdb").read_text(),
                                         old_seq=2, new_seq=1))
    config_path = project / "md.config.yaml"
    original_config = config_path.read_text()
    document = yaml.safe_load(original_config)
    document["AIS"]["source"]["topology"] = "AIS/reassigned.pdb"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False))
    shutil.rmtree(prepared)                       # force it back onto the source-trajectory route
    try:
        result = subprocess.run(["bash", "run.sh", "--check"], cwd=str(project / "AIS"),
                                capture_output=True, text=True, env=environment, timeout=600)
        assert result.returncode != 0, result.stdout + result.stderr
        combined = result.stdout + result.stderr
        assert "[FAIL] AIS" in combined
        assert "disagree at atom index" in combined or "bond" in combined
        assert "no Context was created" in combined
        # Nothing was prepared and no path directory was touched.
        assert not prepared.exists() or not (prepared / "sources.dcd").exists()
    finally:
        config_path.write_text(original_config)
        mutated.unlink(missing_ok=True)
        prepared.mkdir(parents=True, exist_ok=True)
        for name, data in kept.items():
            (prepared / name).write_bytes(data)


# --- 4. --check recomputes stage identity -------------------------------------------------------

def _mutate_yaml(path, key, value):
    document = yaml.safe_load(path.read_text())
    original = path.read_text()
    document[key] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return original


@pytest.mark.gpu
@pytest.mark.slow
def test_check_recomputes_the_current_and_parent_stage_fingerprints(completed_chain):
    """A stored hash being present is not proof that it matches the current request."""
    project, environment = completed_chain
    done, downstream = project / "eq" / "nvt_1kcal", project / "eq" / "npt_1kcal"

    def check(where):
        return subprocess.run(["bash", "run.sh", "--check"], cwd=str(where), capture_output=True,
                              text=True, env=environment, timeout=600)

    # An unchanged completed chain passes, and says what it verified.
    passing = check(downstream)
    assert passing.returncode == 0, passing.stdout + passing.stderr
    assert "request unchanged" in passing.stdout and "handoff verified" in passing.stdout
    assert "complete and unchanged" in check(done).stdout

    # A consequential CURRENT-stage value changes.
    original = _mutate_yaml(done / "stage.yaml", "duration_ps", 0.04)
    try:
        result = check(done)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] completion record" in combined
        assert "stage.yaml has changed since this stage ran" in combined
        assert "no Context was created" in combined
    finally:
        (done / "stage.yaml").write_text(original)

    # A consequential PARENT-stage value changes, checked from the stage downstream of it.
    original = _mutate_yaml(done / "stage.yaml", "temperature_kelvin", 310)
    try:
        result = check(downstream)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] parent stage" in combined
        assert "has changed since it ran" in combined
        assert "no Context was created" in combined
    finally:
        (done / "stage.yaml").write_text(original)

    assert check(downstream).returncode == 0, "the restored chain must pass again"


@pytest.mark.gpu
@pytest.mark.slow
def test_check_refuses_a_parent_final_state_that_changed_after_it_was_recorded(completed_chain):
    project, environment = completed_chain
    state = project / "eq" / "nvt_1kcal" / "final_state.xml"
    intact = state.read_bytes()
    state.write_bytes(intact + b"\n<!-- edited -->")
    try:
        result = subprocess.run(["bash", "run.sh", "--check"],
                                cwd=str(project / "eq" / "npt_1kcal"), capture_output=True,
                                text=True, env=environment, timeout=600)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "[FAIL] parent stage" in combined
        assert "has changed since it was recorded" in combined
    finally:
        state.write_bytes(intact)


def test_there_is_one_stage_fingerprint_implementation():
    """Two subtly different hashes over "the stage request" is the bug this check exists for."""
    preflight = PREFLIGHT_SOURCE.read_text()
    assert "from md_stages import stage_config_sha256" in preflight
    # Preflight must not carry its own canonicalisation.
    assert "yaml.safe_dump(document, sort_keys=True" not in preflight
    stage_run = (REPO_ROOT / "src" / "md_templates" / "openmm" / "templates"
                 / "stage_run.py").read_text()
    assert "stage_config_sha256(STAGE)" in stage_run


# --- 7. the pinned, public validator ------------------------------------------------------------

def test_the_validator_pin_is_https_and_an_exact_commit():
    assert MD.MD_DATA_REPOSITORY.startswith("https://")
    assert MD.COMMIT.match(MD.MD_DATA_COMMIT)
    requirement = MD.md_data_requirement()
    assert requirement.startswith("md-data @ git+https://")
    assert requirement.endswith("@" + MD.MD_DATA_COMMIT)
    assert "git+ssh" not in requirement and "@dev" not in requirement


def test_nothing_instructs_a_user_to_install_over_unpinned_ssh():
    """A moving branch over SSH is how two people validate against different contracts.

    Scoped to lines that actually tell someone to run something. A comment EXPLAINING why
    `git+ssh://.../dev` was replaced is documentation, not an instruction, and forbidding the
    string outright would forbid saying why.
    """
    skipped = (REPO_ROOT / "docs" / "journal", REPO_ROOT / "claudecode-instructions",
               REPO_ROOT / "build", REPO_ROOT / ".git", Path(__file__))
    offenders = {}
    for path in sorted(REPO_ROOT.rglob("*")):
        if path.suffix not in (".py", ".md", ".sh", ".toml") or not path.is_file():
            continue
        if any(path == s or path.is_relative_to(s) for s in skipped if s.exists()):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace")
                                      .splitlines(), 1):
            if "pip install" in line and ("git+ssh" in line or "MD-data.git@dev" in line):
                offenders[f"{path.relative_to(REPO_ROOT)}:{number}"] = line.strip()
    assert not offenders, offenders


def test_the_installer_builds_the_pinned_specification_without_downloading():
    from md_templates.install import openmm as installer

    pin = installer.md_data_pin()
    assert pin["commit"] == MD.MD_DATA_COMMIT
    assert pin["contract_version"] == MD.MD_DATA_CONTRACT_VERSION
    planned = installer.install_md_data("/nowhere", dry_run=True)
    assert planned["attempted"] is False and planned["installed"] is None
    assert planned["command"][1:] == ["install", MD.md_data_requirement()]
    assert planned["command"][0].endswith("/bin/pip")


def test_the_installer_records_a_failed_validator_install_without_failing_the_environment(
        monkeypatch):
    """A private repository or an offline machine does not make the OpenMM environment unusable."""
    from md_templates.install import openmm as installer

    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "fatal: could not read Username for 'https://github.com'"

    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: _Failed())
    record = installer.install_md_data("/nowhere")
    assert record["installed"] is False
    assert "could not read Username" in record["error"]
    assert record["commit"] == MD.MD_DATA_COMMIT


def test_the_installed_validator_reports_its_identity():
    """The documented installation must be able to say what it got."""
    from md_templates.install import openmm as installer

    probe = installer.probe_md_data(sys.prefix)
    assert set(probe) >= {"import_ok", "validator_available"}
    if probe["import_ok"]:
        assert probe["validator_available"] is True
        assert probe["version"] and probe["contract_version"]


# --- fixtures -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_ais_project(tmp_path_factory):
    """The smallest project with a finished cMD source and finished AIS paths.

    Picoseconds, a box at the cutoff, two paths. It exists to exercise the integrity checks, not
    to sample anything: 120 source frames so `iterload` genuinely spans three chunks of 50.
    """
    work = tmp_path_factory.mktemp("integrity")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    environment = dict(os.environ)
    environment.pop("MD_PLATFORM", None)

    assert run_cli("md_openmm", "sys-config", "--method", "cMD", "AIS",
                   cwd=work).returncode == 0
    path = work / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    protocol_path = work / "md.config.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.02
    protocol["cMD"].update({"tau": 0.5, "duration_ns": 0.00024, "checkpoint_interval_ps": 0.24,
                            "whole_system_interval_ps": 0.002, "solute_interval_ps": 0.002})
    protocol["AIS"]["path"]["switching_duration_ps"] = 0.08
    protocol["AIS"]["source"].update({"trajectory": "cMD/whole_system.dcd",
                                      "start_time_ps": 0.002, "end_time_ps": 0.24,
                                      "number_of_trajectories": 2})
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    project = work / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free", "cMD", "AIS"):
        result = subprocess.run(["bash", "run.sh"], cwd=str(project / stage), capture_output=True,
                                text=True, env=environment, timeout=1800)
        assert result.returncode == 0, f"{stage}: {result.stdout}{result.stderr}"
    return project, environment


@pytest.fixture(scope="module")
def prepared_ais(tiny_ais_project):
    """The AIS component, where the prepared starting configurations live."""
    return tiny_ais_project[0] / "AIS"


@pytest.fixture(scope="module")
def completed_chain(tmp_path_factory):
    """Minimisation and one equilibration stage, actually run, so their records are real.

    Deliberately shorter than `tiny_ais_project`: these tests need a completed stage and its
    child, not a production method.
    """
    work = tmp_path_factory.mktemp("stageidentity")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    environment = dict(os.environ)
    environment.pop("MD_PLATFORM", None)

    assert run_cli("md_openmm", "sys-config", "--method", "cMD", cwd=work).returncode == 0
    path = work / "sys.config.yaml"
    document = yaml.safe_load(path.read_text())
    document["solvent"].update({"padding_nm": 0.5, "cutoff_nm": 0.5})
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    protocol_path = work / "md.config.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.02
    protocol["cMD"].update({"duration_ns": 0.00004, "checkpoint_interval_ps": 0.02,
                            "whole_system_interval_ps": 0.02, "solute_interval_ps": 0.02})
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    project = work / "MD"
    for stage in ("minimization", "eq/nvt_1kcal"):
        result = subprocess.run(["bash", "run.sh"], cwd=str(project / stage),
                                capture_output=True, text=True, env=environment, timeout=1800)
        assert result.returncode == 0, f"{stage}: {result.stdout}{result.stderr}"
    return project, environment


# --- the instruction index ----------------------------------------------------------------------

def test_no_instruction_is_indexed_twice():
    """A file listed both pending and executed leaves a reader with no way to tell which is true."""
    index = (REPO_ROOT / "claudecode-instructions" / "README.md").read_text()
    duplicated = {}
    for path in sorted((REPO_ROOT / "claudecode-instructions").glob("2026*.md")):
        count = index.count(f"`{path.name}`")
        if count > 1:
            duplicated[path.name] = count
    assert not duplicated, duplicated


def test_every_instruction_executed_this_round_is_indexed_once():
    index = (REPO_ROOT / "claudecode-instructions" / "README.md").read_text()
    for name in ("20260827_final-md-data-integrity-correction.md",
                 "20260827_provenance-and-contract-readiness-correction.md"):
        assert index.count(f"`{name}`") == 1, name


# --- the installed validator, not the intended one ----------------------------------------------

def test_verification_uses_the_installed_source_not_the_intended_pin(monkeypatch):
    """Echoing the pin describes our wish; direct_url.json describes the bytes on disk."""
    from md_templates.install import openmm as installer

    pin = installer.md_data_pin()

    def probe(_prefix, **fields):
        base = {"import_ok": True, "validator_available": True, "version": "0.2.0",
                "contract_version": pin["contract_version"], "installed_commit": None,
                "installed_source": None, "source_kind": "unknown"}
        base.update(fields)
        return base

    # Installed from the exact pin: ready.
    monkeypatch.setattr(installer, "probe_md_data", lambda p: probe(
        p, installed_commit=pin["commit"], source_kind="vcs",
        installed_source=pin["repository"]))
    report = installer.verify_md_data("/nowhere")
    assert report["contract_support_ready"] is True
    assert report["commit_verified"] is True and not report["reasons"]

    # Installed from a DIFFERENT commit: imports fine, validates fine, still not ready.
    monkeypatch.setattr(installer, "probe_md_data", lambda p: probe(
        p, installed_commit="b" * 40, source_kind="vcs"))
    report = installer.verify_md_data("/nowhere")
    assert report["contract_support_ready"] is False and report["commit_verified"] is False
    assert "not the pinned" in report["reasons"][0]

    # Installed from a local directory: no source commit to check at all.
    monkeypatch.setattr(installer, "probe_md_data", lambda p: probe(
        p, source_kind="local directory", installed_source="file:///somewhere"))
    report = installer.verify_md_data("/nowhere")
    assert report["contract_support_ready"] is False
    assert "records no source commit" in report["reasons"][0]

    # A contract version this repository does not target.
    monkeypatch.setattr(installer, "probe_md_data", lambda p: probe(
        p, installed_commit=pin["commit"], source_kind="vcs", contract_version="2.0"))
    report = installer.verify_md_data("/nowhere")
    assert report["contract_support_ready"] is False
    assert any("contract version" in reason for reason in report["reasons"])

    # Does not import.
    monkeypatch.setattr(installer, "probe_md_data", lambda p: {
        "import_ok": False, "validator_available": False, "error": "ModuleNotFoundError"})
    report = installer.verify_md_data("/nowhere")
    assert report["contract_support_ready"] is False
    assert "does not import" in report["reasons"][0]


def test_contract_unavailability_is_a_prominent_warning_not_a_failed_installation():
    """A researcher doing unregistered local simulation has a working installation either way."""
    from md_templates.install import openmm as installer

    report = {"cuda_available": True, "nvidia": {"present": True},
              "capabilities": {"openmm_runtime_ready": True,
                               "md_data_contract_support_ready": False,
                               "md_data_unavailable_reasons": ["md_data does not import"]}}
    notes = installer.warnings_for(report)
    banner = [n for n in notes if "MD-DATA CONTRACT SUPPORT UNAVAILABLE" in n]
    assert banner, notes
    assert "unregistered local simulation works" in banner[0]
    assert "dataset.enabled: true` will fail" in banner[0]
    # ...and it is a warning, not a problem that fails the install.
    assert not any("MD-DATA" in problem for problem in installer._problems(report))

    ready = {**report, "capabilities": {**report["capabilities"],
                                        "md_data_contract_support_ready": True}}
    assert not [n for n in installer.warnings_for(ready) if "MD-DATA" in n]


def test_the_probe_reads_installed_distribution_metadata():
    """It must inspect the installation, so the probe has to look at direct_url.json."""
    from md_templates.install import openmm as installer

    assert "direct_url.json" in installer._MD_DATA_PROBE
    assert "vcs_info" in installer._MD_DATA_PROBE
    assert "commit_id" in installer._MD_DATA_PROBE
    # And it must not simply echo the pin back.
    assert MD.MD_DATA_COMMIT not in installer._MD_DATA_PROBE
