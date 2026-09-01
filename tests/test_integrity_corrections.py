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

#: The AIS runtime. These assertions read source text because they are about what the code must
#: NOT do -- a property no successful run can demonstrate. They followed the implementation from
#: the retired `templates/ais_run.py` to its live home rather than being dropped with it.
AIS_SOURCE = REPO_ROOT / "src" / "md_tools" / "runtime" / "ais.py"


# --- 1. the production source is never hashed ---------------------------------------------------

def test_the_ais_runtime_never_hashes_the_production_source():
    """A full-file digest of a production trajectory costs more than it proves.

    The source is read frame by frame; hashing it would mean reading the whole thing an extra
    time to produce a number nothing checks. The live runtime computes no digest at all, which
    is a stronger statement of the same guarantee than the retired script could make.
    """
    source = AIS_SOURCE.read_text()
    assert "sha256" not in source, "the AIS runtime digests something; it used to digest the source"
    assert "hashlib" not in source




def test_the_no_hashing_assertion_is_not_vacuous():
    """A "this string is absent" assertion passes just as happily against the wrong file.

    The guard this replaces installed a `sitecustomize` that intercepted `pathlib.Path.open` and
    failed the run if the production source reached `sha256_file`. It could only run against the
    retired script, which hashed things; the live runtime hashes nothing at all, so there is no
    call to intercept. What is still worth proving is that the check DISCRIMINATES -- that the same
    assertion applied to code which does hash would fail.
    """
    root = REPO_ROOT / "src" / "md_tools"
    hashing = (root / "runtime" / "stage.py").read_text()
    assert "sha256" in hashing, (
        "the positive control no longer hashes, so the absence check above proves nothing")
    assert "sha256" not in AIS_SOURCE.read_text()


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






# `test_the_header_count_is_documented_as_insufficient` is retired with `templates/ais_run.py`.
# It asserted that `dcd_header_frames` documented a DCD header's frame count as untrustworthy,
# because that script wrote its own DCD and then re-read it. The live runtime does not write a
# trajectory it must re-count: it validates a finished path against the schedule itself -- row
# count, both tau endpoints, and exactly zero cumulative work at observation 0 -- which is what
# `test_ais_runs_through_the_real_cli_and_keeps_its_work_contract` exercises end to end.


# --- 3. the route-aware force-field record ------------------------------------------------------
#
# These tests used to drive `preflight.check_forcefield`, which compared `inputs/forcefield.json`
# against `resolved_sys.config.yaml` and refused a mismatch: a peptide system claiming a ligand
# force field, a ligand system claiming a protein one, an implicit system claiming a water model.
#
# That comparison no longer has two sides to compare. `forcefield.json` and the route that wrote it
# are retired, and `build_forcefield_record` now builds the record FROM the builder's own report at
# the point the System is constructed -- so the record cannot disagree with what was built, and the
# entire class of defect is gone by construction rather than by being checked for.
#
# What survives is the property the check existed to protect: the record must be route-aware, and
# must say "not applicable here" rather than naming a Hamiltonian that parameterised nothing. That
# is asserted directly against the live function.

NONBONDED_REPORT = {"method": "PME", "cutoff_nm": 0.9, "switching": True, "switch_nm": 0.8,
                    "ewald_tolerance": 0.0005, "dispersion_correction": True}


def _record(*, resolved, route, report):
    from md_tools.openmm.forcefield_record import build_forcefield_record

    return build_forcefield_record(resolved=resolved, route=route, record=report,
                                   inputs_dir=Path("."), artifacts={})


def test_the_ligand_route_records_no_protein_force_field():
    """Recording ff14SB for a ligand-only build names a Hamiltonian that parameterised nothing."""
    built = _record(
        resolved={"solvation": "explicit",
                  "solute": {"peptide": False, "ligand_forcefield": "sage-2.2.1"},
                  "forcefield": {"water": "tip3p"}},
        route="ligand",
        report={"nonbonded": NONBONDED_REPORT,
                "forcefield": {"ligand": {"openff_resource": "openff-2.2.1",
                                          "charge_method": "am1bcc"},
                               "water": {"openmm_resource": "amber14/tip3p.xml"}}})

    assert built["protein"]["openmm_resource"] is None
    assert built["protein"]["tleap_resource"] is None
    # Present and explained, not absent: a reader must be able to tell "not applicable" from
    # "nobody recorded it".
    assert "loads no protein force field" in built["protein"]["note"]
    assert built["ligand"]["charge_method"] == "am1bcc"


def test_an_implicit_system_records_no_water_model():
    """An implicit system has no box and no water, and saying so is information."""
    built = _record(
        resolved={"solvation": "implicit", "solute": {"peptide": True},
                  "forcefield": {"protein": "ff14SB"},
                  "implicit_solvent": {"model": "GBn2", "radii": "mbondi3",
                                       "nonpolar_sasa": False}},
        route="peptide",
        report={"implicit_report": {"model": "GBn2", "radii": "mbondi3"}})

    assert built["water"]["openmm_resource"] is None
    assert built["water"]["model"] is None
    assert "no water model" in built["water"]["note"]
    assert built["implicit_solvent"]["model"] == "GBn2"
    assert built["implicit_solvent"]["nonpolar_sasa"] is False


def test_the_record_refuses_to_invent_a_nonbonded_treatment():
    """The guard that replaced the comparison: a value the builder did not report is an error,
    not a plausible default. This is the failure mode the old check could only notice afterwards.
    """
    with pytest.raises(ValueError, match="no nonbonded treatment"):
        _record(resolved={"solvation": "explicit", "solute": {"peptide": True},
                          "forcefield": {"protein": "ff14SB", "water": "tip3p"}},
                route="peptide",
                report={"forcefield": {"protein": {"openmm_resource": "amber14-all.xml"}}})


def test_the_record_states_which_route_produced_it():
    """Route is recorded, so a reader never has to infer it from which fields happen to be null."""
    peptide = _record(
        resolved={"solvation": "explicit", "solute": {"peptide": True},
                  "forcefield": {"protein": "ff14SB", "water": "tip3p"}},
        route="peptide",
        report={"nonbonded": NONBONDED_REPORT,
                "forcefield": {"protein": {"openmm_resource": "amber14-all.xml"},
                               "water": {"openmm_resource": "amber14/tip3p.xml"}}})
    assert peptide["route"] == "peptide"
    assert peptide["solvation"] == "explicit"


# --- 5. the template commit is evidence, not a well-formed string -------------------------------







# Preflight's provenance comparison is exercised against REAL generated projects in
# tests/test_template_provenance.py -- matching, missing and mismatching records, for both the
# clean-checkout and direct_url.json identity routes. A hand-built stub of the record set was the
# weaker form of the same check and is deliberately not kept beside it.


# --- 6. atom identity and order, not names and counts -------------------------------------------

def test_repeated_atom_names_do_not_make_two_topologies_the_same(tmp_path):
    """Every residue has an N, a CA, a C and an O. Names alone cannot tell residues apart."""
    ais = template_module("source_ensemble")
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
    source = (REPO_ROOT / "src" / "md_tools" / "openmm" / "templates"
          / "source_ensemble.py").read_text()
    start = source.index("def atom_identity")
    # `ATOM_FIELDS` is defined near the top of the shared helper, so slice to the NEXT definition
    # after the function rather than to a name that also appears before it.
    block = source[start:source.index("\ndef ", start + 1)]
    for field in ("chain.index", "residue.index", "residue.name", "atom.name",
                  "atom.element.symbol"):
        assert field in block, field
    assert "topology.bonds()" in block


# `template_module_with_stubs` is gone with `templates/ais_run.py`. It existed to import that
# script with its project-configuration loading faked out, so that the pure functions inside it
# could be unit-tested. Those functions were already delegates: `atom_identity` and the rest live
# in `source_ensemble.py`, which is an ordinary importable module and is what these tests now use.


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

    ais = template_module("source_ensemble")
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

    ais = template_module("source_ensemble")
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
    """Two subtly different hashes over "the stage request" is the bug this check exists for.

    Each retired launcher hashed the stage itself and `preflight` hashed it again. All of them are
    gone, and `runtime.stage._config_fingerprint` is now the only implementation. It is also
    stronger than what it replaced: it covers the System and topology digests, so a checkpoint
    cannot be resumed against a System that was rebuilt underneath it.
    """
    root = REPO_ROOT / "src" / "md_tools"
    definitions = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")
                         if "def _config_fingerprint(" in path.read_text(encoding="utf-8")
                         or "def stage_config_sha256(" in path.read_text(encoding="utf-8"))
    assert definitions == ["runtime/stage.py"], definitions

    runtime = (root / "runtime" / "stage.py").read_text()
    assert "_config_fingerprint(stage, system_sha, topology_sha)" in runtime, \
        "the runtime no longer fingerprints the stage it is about to run"


# --- 7. the pinned, public validator ------------------------------------------------------------



def test_nothing_instructs_a_user_to_install_over_unpinned_ssh():
    """A moving branch over SSH is how two people validate against different contracts.

    Scoped to lines that actually tell someone to run something. A comment EXPLAINING why
    `git+ssh://.../dev` was replaced is documentation, not an instruction, and forbidding the
    string outright would forbid saying why.
    """
    skipped = (REPO_ROOT / "docs" / "release-notes",
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





# --- the installed validator, not the intended one ----------------------------------------------





