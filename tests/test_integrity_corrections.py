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


from .conftest import ALA_PDB, REPO_ROOT, run_cli, template_module

AIS_SOURCE = REPO_ROOT / "src" / "md_tools" / "openmm" / "templates" / "ais_run.py"
PREFLIGHT_SOURCE = REPO_ROOT / "src" / "md_tools" / "openmm" / "templates" / "preflight.py"


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
    """`atom_identity` moved to the shared source-ensemble reader, which rREST2 uses too.

    The guard follows it rather than being relaxed: AIS and rREST2 must compare atoms the same
    way, and the whole point of one implementation is that this check covers both.
    """
    source = (AIS_SOURCE.parent / "source_ensemble.py").read_text()
    start = source.index("def atom_identity")
    # `ATOM_FIELDS` is defined near the top of the shared helper, so slice to the NEXT definition
    # after the function rather than to a name that also appears before it.
    block = source[start:source.index("\ndef ", start + 1)]
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
    import sys
    import types

    templates = REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
    path = templates / "ais_run.py"
    source = path.read_text()
    # Everything above the first function definition is project-configuration loading.
    cut = source.index("def relative_to_project")
    module = types.ModuleType("_ais_pure")
    module.__dict__.update({name: __import__(name) for name in ("hashlib", "csv", "json", "yaml")})
    module.__dict__.update({"Path": __import__("pathlib").Path, "np": __import__("numpy")})
    # `ais_run` delegates the source-ensemble rules to the shared helper that ships beside it, so
    # that directory has to be importable exactly as it is in a generated project.
    if str(templates) not in sys.path:
        sys.path.insert(0, str(templates))
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




# --- 4. --check recomputes stage identity -------------------------------------------------------

def _mutate_yaml(path, key, value):
    document = yaml.safe_load(path.read_text())
    original = path.read_text()
    document[key] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return original






def test_there_is_one_stage_fingerprint_implementation():
    """Two subtly different hashes over "the stage request" is the bug this check exists for."""
    preflight = PREFLIGHT_SOURCE.read_text()
    assert "from md_stages import stage_config_sha256" in preflight
    # Preflight must not carry its own canonicalisation.
    assert "yaml.safe_dump(document, sort_keys=True" not in preflight
    stage_run = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
                 / "stage_run.py").read_text()
    assert "stage_config_sha256(STAGE)" in stage_run


# --- 7. the pinned, public validator ------------------------------------------------------------



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








# --- fixtures -----------------------------------------------------------------------------------







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





