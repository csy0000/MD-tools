"""AIS directory identity, resume semantics, and the one decision it makes.

WHAT WAS WRONG (all three baseline defects, in one file)

    An AIS output directory could be adopted as FRESH merely because `AIS_run.json` was absent --
    even when the directory held real path directories, trajectories and tables from an
    unidentified run. The absence of the ONE record naming what a directory holds was treated as
    proof it held nothing.

    The preflight's disposition and the boolean handed to `run_one_path` were two different
    decisions. A compatible, incomplete directory was ALWAYS labelled "resume" -- whether or not
    `--resume` was actually given -- and then `run_one_path` was called with
    `resume=bool(args.resume)`. Omit `--resume` on a directory with a genuinely interrupted path,
    and the preflight called it a resume while the runtime started that path from scratch:
    destructive, with nothing on the command line asking for it.

    `clear_run_directory` accepted an `identity` parameter and never used it. Ownership was
    inferred from NAME SCHEMA alone -- `path_0000`, `AIS_trajNNNN.nc` -- so a directory a person
    created by hand, sharing nothing but a four-digit name, was as much "this run's own output" as
    a directory the runtime actually wrote.

`decide_run_disposition` (unit-level, read-only, exhaustive over the state machine) and one
end-to-end CPU run (the actual destructive-restart scenario, through the real generated project
and the real `run_one_path`) are both here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"


def _identity_document(**overrides):
    from md_tools.ais.run import RUN_IDENTITY_VERSION

    document = {
        "schema": "md-ais-run-identity", "schema_version": RUN_IDENTITY_VERSION,
        "fingerprint": "f" * 64,
        "end_states": {"V0": {"system": "s" * 64, "topology": "t" * 64},
                       "V1": {"system": "u" * 64, "topology": "t" * 64}},
        "source": {"sha256": "x" * 64, "format": "dcd"},
        "lambda": {"start": 0.0, "end": 1.0, "interpolation": "linear"},
        "schedule": {"switching_steps": 10}, "reporting": {"crd_printout_solute": 5},
        "seed_policy": {"seed": 1, "derivation": "derive_seed(seed, 'ais', path_index, role)"},
        "number_of_paths": 2, "selected_frames": [0, 1],
        "observation_columns": [], "ais_schema": {"name": "two-state-linear", "version": 1},
        "resolved_config": None,
    }
    document.update(overrides)
    return document


def _document(*, paths=1, chosen=(7,), seed=1, v1_system="u" * 64):
    """`run_identity_document` as the preflight calls it, with only what a test varies exposed."""
    from md_tools.ais.run import run_identity_document

    return run_identity_document(
        fingerprint="f" * 64,
        end_state_facts={"V0": {"system": "s" * 64, "topology": "t" * 64},
                         "V1": {"system": v1_system, "topology": "t" * 64}},
        source_facts={"sha256": "x" * 64}, source_format="dcd",
        schedule={"switching_steps": 10}, ais={"number_of_paths": paths},
        dynamics={"seed": seed}, chosen=list(chosen), reporting={}, resolved_config=None)


# --- decide_run_disposition: the whole state machine, unit-level and read-only ------------------

def test_an_absent_directory_is_fresh(tmp_path):
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    disposition, previous = decide_run_disposition(
        out, _identity_document(), resume=False, overwrite=False, chosen=[0, 1],
        fingerprint="f" * 64, schedule={"switching_steps": 10})
    assert disposition == "fresh" and previous is None


def test_an_empty_directory_is_fresh(tmp_path):
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    out.mkdir()
    disposition, previous = decide_run_disposition(
        out, _identity_document(), resume=False, overwrite=False, chosen=[0, 1],
        fingerprint="f" * 64, schedule={"switching_steps": 10})
    assert disposition == "fresh" and previous is None


def test_owned_artefacts_with_no_identity_are_refused_as_orphaned_even_under_overwrite(tmp_path):
    """THE first baseline defect: absence of AIS_run.json used to mean 'fresh', unconditionally.

    Refused under EVERY combination of flags: there is no automated action -- fresh, resume, or
    overwrite -- that is safe when nothing can prove what a directory's contents belong to.
    """
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    (out / "path_0000").mkdir(parents=True)
    (out / "path_0000" / "completed.json").write_text("{}", encoding="utf-8")

    for resume, overwrite in ((False, False), (True, False), (False, True)):
        with pytest.raises(SystemExit, match="[Oo]wnership"):
            decide_run_disposition(out, _identity_document(), resume=resume, overwrite=overwrite,
                                   chosen=[0, 1], fingerprint="f" * 64,
                                   schedule={"switching_steps": 10})
    assert (out / "path_0000" / "completed.json").is_file(), (
        "a refused disposition decision must not have touched anything")


def test_resume_against_an_empty_directory_refuses(tmp_path):
    """`--resume` names a run to continue. An empty directory has none."""
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    with pytest.raises(SystemExit, match="nothing here"):
        decide_run_disposition(out, _identity_document(), resume=True, overwrite=False,
                               chosen=[0, 1], fingerprint="f" * 64,
                               schedule={"switching_steps": 10})


def _write_identity(out: Path, document: dict) -> None:
    from md_tools.ais.run import RUN_IDENTITY

    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_IDENTITY).write_text(json.dumps(document), encoding="utf-8")


def test_a_compatible_directory_whose_every_path_verifies_is_labelled_verify_not_resume(tmp_path):
    """A completed run may be re-examined without --resume, and the label says so honestly.

    Distinct from "resume" specifically so the boolean handed to `run_one_path` is never the
    string "resume" when `--resume` was not given -- see the destructive-restart test below for
    why that distinction is not cosmetic.
    """
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    document = _document(paths=0, chosen=[], seed=1)
    document["number_of_paths"] = 0
    document["selected_frames"] = []
    _write_identity(out, document)
    # No paths at all: vacuously, every one of the zero selected paths verifies.

    disposition, previous = decide_run_disposition(
        out, document, resume=False, overwrite=False, chosen=[], fingerprint="f" * 64,
        schedule={"switching_steps": 10})
    assert disposition == "verify"
    assert previous is not None


def test_an_incomplete_compatible_directory_without_resume_is_refused_before_any_byte_changes(
        tmp_path):
    """THE core defect, at the decision layer: an interrupted path with no --resume must refuse.

    Not silently start it fresh (destructive), and not silently label it "resume" while running
    it as though nothing needed continuing (the actual baseline bug). Refused, before -odir is
    even touched by dynamics.
    """
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    document = _document(paths=1, chosen=[7], seed=1)
    _write_identity(out, document)
    # Started but not finished: a bare directory with an MD-tools marker inside, no manifest.
    (out / "path_0000").mkdir()
    (out / "path_0000" / "current_checkpoint.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="not yet"):
        decide_run_disposition(out, document, resume=False, overwrite=False, chosen=[7],
                               fingerprint="f" * 64, schedule={"switching_steps": 10})
    # Untouched: still exactly what it was.
    assert (out / "path_0000" / "current_checkpoint.json").is_file()

    disposition, previous = decide_run_disposition(
        out, document, resume=True, overwrite=False, chosen=[7], fingerprint="f" * 64,
        schedule={"switching_steps": 10})
    assert disposition == "resume" and previous is not None


def test_an_incompatible_directory_is_refused_by_naming_what_differs_even_with_resume(tmp_path):
    """A DIFFERENT experiment in the same -odir is refused regardless of --resume."""
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    old = _document(paths=1, chosen=[7], seed=1)
    _write_identity(out, old)
    new = _document(paths=1, chosen=[7], seed=99)

    with pytest.raises(SystemExit, match="different"):
        decide_run_disposition(out, new, resume=True, overwrite=False, chosen=[7],
                               fingerprint="f" * 64, schedule={"switching_steps": 10})


def test_a_different_second_end_state_is_a_different_run(tmp_path):
    """V1 is half of what an AIS run measures, and the identity has to say so.

    A second invocation into the same -odir with a different `-s2` would skip the finished paths
    and switch the rest toward another Hamiltonian; the work table would average two
    transformations. Refused, naming `end_states`, even with --resume.
    """
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    _write_identity(out, _document())
    new = _document(v1_system="v" * 64)

    with pytest.raises(SystemExit, match="end_states"):
        decide_run_disposition(out, new, resume=True, overwrite=False, chosen=[7],
                               fingerprint="f" * 64, schedule={"switching_steps": 10})


def _single_topology_identity():
    """What the retired single-topology AIS wrote: run-identity v1, with a `tau` block."""
    return {
        "schema": "md-ais-run-identity", "schema_version": 1,
        "fingerprint": "f" * 64, "topology": {"name": "t.pdb", "sha256": "t" * 64},
        "system": {"sha256": "s" * 64}, "source": {"sha256": "x" * 64, "format": "dcd"},
        "tau": {"start": 0.5, "end": 0.0, "interpolation": "linear"},
        "schedule": {"switching_steps": 10}, "reporting": {"crd_printout_solute": 5},
        "seed_policy": {"seed": 1, "derivation": "derive_seed(seed, 'ais', path_index, role)"},
        "number_of_paths": 1, "selected_frames": [7], "observation_columns": [],
        "decomposition_schema": {"name": "rest2-lambda-basis", "version": 2},
        "resolved_config": None,
    }


def test_a_single_topology_identity_is_refused_by_name_under_every_disposition(tmp_path):
    """A directory the tau-switching AIS wrote is not continued, verified or cleaned by this one.

    Its paths measured work along a different Hamiltonian path, so --resume would append
    two-state work onto tau work, and --overwrite would trust a record whose fields this build
    does not define. The refusal names the retired AIS rather than a bare schema number, because
    "v1" tells the person nothing about why their directory is suddenly foreign.
    """
    from md_tools.ais.run import RUN_IDENTITY, decide_run_disposition

    out = tmp_path / "AIS"
    _write_identity(out, _single_topology_identity())
    (out / "path_0000").mkdir()
    (out / "path_0000" / "completed.json").write_text("{}", encoding="utf-8")
    before = (out / RUN_IDENTITY).read_bytes()

    for resume, overwrite in ((False, False), (True, False), (False, True)):
        with pytest.raises(SystemExit, match="single-topology") as refusal:
            decide_run_disposition(out, _document(), resume=resume, overwrite=overwrite,
                                   chosen=[7], fingerprint="f" * 64,
                                   schedule={"switching_steps": 10})
        assert "tau" in str(refusal.value)
    assert (out / RUN_IDENTITY).read_bytes() == before
    assert (out / "path_0000" / "completed.json").is_file()


def test_require_same_run_names_the_single_topology_identity_too(tmp_path):
    """The second reader of `AIS_run.json` gives the same reason, not only a version mismatch."""
    from md_tools.ais.run import require_same_run

    out = tmp_path / "AIS"
    _write_identity(out, _single_topology_identity())
    with pytest.raises(SystemExit, match="single-topology"):
        require_same_run(out, _document())


def test_overwrite_of_a_readable_but_incompatible_identity_is_allowed(tmp_path):
    """--overwrite means start over. A DIFFERENT but verifiable previous run may still be
    replaced -- ownership CAN be established (from its own valid identity), only the schedule
    differs, and that is exactly what --overwrite is for."""
    from md_tools.ais.run import decide_run_disposition

    out = tmp_path / "AIS"
    old = _document(paths=1, chosen=[7], seed=1)
    _write_identity(out, old)
    new = _identity_document(fingerprint="g" * 64)

    disposition, previous = decide_run_disposition(
        out, new, resume=False, overwrite=True, chosen=[0, 1], fingerprint="g" * 64,
        schedule={"switching_steps": 10})
    assert disposition == "overwrite"
    assert previous is not None and previous["fingerprint"] == "f" * 64


# --- the end-to-end destructive-restart scenario, through a real CPU run ------------------------

@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A dataset root holding `build/` and one generated AIS run, as `build-md` now requires.

    V1 is the REST2 scaling of `built.xml` at tau = 0.5: a parameter-only edit of this very
    System, which is what the two-state pair check accepts. A copy of `built.xml` would be refused
    as the same Hamiltonian before any of the disposition logic under test ran.
    """
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("ais-identity")
    (root / "build").mkdir()
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from md_tools.md.stage import solute_atom_indices
    from md_tools.rest2.hamiltonian import build_scaled_system

    base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    solute = solute_atom_indices(PDBFile(str(root / "build" / "built.pdb")).topology)
    (root / "build" / "V1.xml").write_text(
        XmlSerializer.serialize(build_scaled_system(base, solute, 0.5)), encoding="utf-8")

    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 1, "switching_steps": 20, "observation_interval_steps": 5,
                "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20, "checkpoint_printout": 5},
    }), encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(root / "AIS"),
                                 "--config", str(root / "AIS.config")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))
    return root


def _user_config(directory: Path) -> dict[str, str]:
    path = directory / "user.config"
    path.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    return {"MD_TOOLS_CONFIG": str(path)}


def _run_ais(project_root: Path, work: Path, *, environment=None, timeout=900, extra=()):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    base.update(_user_config(work))
    base.update(environment or {})
    return subprocess.run(
        [sys.executable, str(project_root / "AIS" / "AIS.py"),
         "-p", str(project_root / "build" / "built.pdb"),
         "-s", str(project_root / "build" / "built.xml"),
         "-p2", str(project_root / "build" / "built.pdb"),
         "-s2", str(project_root / "build" / "V1.xml"),
         "-source-traj", str(project_root / "source.dcd"), "-odir", str(work), "--cpu", *extra],
        cwd=work, capture_output=True, text=True, timeout=timeout, env=base)


@pytest.mark.slow
def test_an_interrupted_path_is_never_restarted_destructively_without_resume(project, tmp_path):
    """THE end-to-end regression: the exact scenario the baseline defect produced.

    A path is interrupted after its first checkpoint commits. Re-running WITHOUT --resume must
    refuse -- not silently restart the path from its source frame, which is what
    `resume=bool(args.resume)` at the `run_one_path` call site used to do whenever the preflight
    had (wrongly, unconditionally) labelled the directory "resume". Only WITH --resume does the
    path continue and finish.
    """
    from md_tools.openmm.checkpoint import (FAULT_AFTER_ENVIRONMENT, FAULT_ENVIRONMENT,
                                            read_committed)

    work = tmp_path / "run"
    work.mkdir()
    crashed = _run_ais(project, work, environment={FAULT_ENVIRONMENT: "after-pointer-replace",
                                                    FAULT_AFTER_ENVIRONMENT: "1"})
    assert crashed.returncode != 0, crashed.stdout + crashed.stderr
    path_dir = work / "path_0000"
    committed = read_committed(path_dir)
    assert committed is not None, "the fault must land after a generation committed"
    steps_before = int(committed["state"]["updates_completed"])
    assert 0 < steps_before < 4, (
        f"expected an early, partial commit; got {steps_before} of 4 updates")

    refused = _run_ais(project, work)
    assert refused.returncode != 0, refused.stdout + refused.stderr
    message = refused.stdout + refused.stderr
    assert "--resume" in message and "--overwrite" in message, message[-2000:]
    # Untouched: the committed checkpoint is exactly what it was before the refused invocation.
    still_committed = read_committed(path_dir)
    assert still_committed["state"]["updates_completed"] == steps_before, (
        "a refused run must not have advanced or reset the committed checkpoint")
    assert not (work / "path_0000" / "completed.json").is_file()

    resumed = _run_ais(project, work, extra=["--resume"])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (work / "path_0000" / "completed.json").is_file()


@pytest.mark.slow
def test_overwrite_after_completion_starts_a_genuinely_new_run(project, tmp_path):
    """The disposition is "verify" on the completed re-run, and "overwrite" wipes it cleanly."""
    work = tmp_path / "run"
    work.mkdir()
    done = _run_ais(project, work)
    assert done.returncode == 0, done.stdout + done.stderr
    marker = work / "path_0000" / "completed.json"
    assert marker.is_file()

    verified = _run_ais(project, work)
    assert verified.returncode == 0, verified.stdout + verified.stderr

    redone = _run_ais(project, work, extra=["--overwrite"])
    assert redone.returncode == 0, redone.stdout + redone.stderr
    assert marker.is_file(), "a fresh run under --overwrite must still complete the path"
