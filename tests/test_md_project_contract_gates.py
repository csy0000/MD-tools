"""Regressions for the contract gates MD-project depends on.

Each of these would have failed before the fixes in this branch. They are implementation-level
tests: MD-project keeps a thin consumer gate that goes through generated files and public
commands, and the detail lives here.

The generation tests build a real system, so they carry the `gpu` marker where they integrate.
Everything that only inspects generated files runs anywhere.
"""

from __future__ import annotations

import ast

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.record import read_record

from .conftest import REPO_ROOT

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "src" / "md_tools" / "remd"


def _md_stages():
    spec = importlib.util.spec_from_file_location("_ms", TEMPLATES / "md_stages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# 1 -- the NAGL resolver executes, and classifies its failures honestly
# ======================================================================================


def test_the_nagl_resolver_runs_rather_than_failing_to_import():
    """The defect was an internal import of a module that does not exist."""
    from md_tools.openmm.system import resolve_nagl_am1bcc_model

    pytest.importorskip("openff.nagl_models",
                        reason="the optional dependency is genuinely absent here")
    model = resolve_nagl_am1bcc_model()
    assert set(model) == {"name", "path", "sha256"}
    assert model["name"].endswith(".pt")
    assert len(model["sha256"]) == 64
    assert Path(model["path"]).is_file()


def test_no_module_named_hashing_is_imported_anywhere():
    """The narrowest statement of the defect, so a regression is obvious."""
    source = (REPO / "src/md_tools/openmm/system.py").read_text(encoding="utf-8")
    assert "md_tools.openmm.hashing" not in source
    assert not (REPO / "src/md_tools/openmm/hashing.py").exists(), (
        "the fix is to use provenance_min.sha256_file, not to add a second hashing module"
    )


def test_a_missing_optional_dependency_is_reported_as_such(monkeypatch):
    """An absent optional package must not look like a broken MD-tools."""
    import builtins

    from md_tools.openmm import system

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("openff.nagl_models"):
            raise ImportError("No module named 'openff.nagl_models'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError) as excinfo:
        system.resolve_nagl_am1bcc_model()
    message = str(excinfo.value)
    assert "openff-nagl-models" in message, "the message must name what to install"
    assert "am1bcc" in message, "and the supported alternative"


def test_an_internal_import_error_is_not_disguised_as_a_missing_dependency(monkeypatch):
    """A broken MD-tools import must propagate, not be relabelled as a user problem."""
    import builtins

    from md_tools.openmm import system

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if "provenance_min" in name:
            raise ModuleNotFoundError("No module named 'md_tools.openmm.provenance_min'")
        return real_import(name, *args, **kwargs)

    pytest.importorskip("openff.nagl_models")
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ModuleNotFoundError):
        system.resolve_nagl_am1bcc_model()


# ======================================================================================
# 2 -- the charge scheme round-trips into forcefield.json
# ======================================================================================


def test_the_ligand_record_carries_the_scheme_the_builder_reported():
    from md_tools.openmm.forcefield_record import _ligand_record

    record = _ligand_record(
        is_ligand=True,
        reported={"ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc",
                             "charge_scheme": "am1bccelf10"}},
        requested={"ligand_forcefield": "sage-2.2.1", "ligand_charge_method": "am1bcc"},
        checksums={},
    )
    assert record["charge_method"] == "am1bcc"
    assert record["charge_scheme"] == "am1bccelf10", (
        "the executed scheme must reach the record; am1bcc and am1bccelf10 are different "
        "quantities over a flexible molecule"
    )


def test_an_unreported_scheme_is_null_rather_than_the_requested_label():
    """Absent means the builder did not report one. Substituting the label invents provenance."""
    from md_tools.openmm.forcefield_record import _ligand_record

    record = _ligand_record(
        is_ligand=True,
        reported={"ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc"}},
        requested={"ligand_forcefield": "sage-2.2.1", "ligand_charge_method": "am1bcc"},
        checksums={},
    )
    assert record["charge_scheme"] is None


def test_the_nagl_model_identity_reaches_the_record():
    """`nagl_model_file` is the key the builder writes; the record looked for `nagl_model`."""
    from md_tools.openmm.forcefield_record import _ligand_record

    record = _ligand_record(
        is_ligand=True,
        reported={"ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc_nagl",
                             "charge_scheme": "openff-gnn-am1bcc-1.0.0.pt",
                             "nagl_model_file": "openff-gnn-am1bcc-1.0.0.pt",
                             "nagl_model_sha256": "a" * 64}},
        requested={"ligand_forcefield": "sage-2.2.1",
                   "ligand_charge_method": "am1bcc_nagl"},
        checksums={},
    )
    assert record["charge_model"] == "openff-gnn-am1bcc-1.0.0.pt"
    assert record["charge_model_sha256"] == "a" * 64


def test_an_unsupported_charge_method_fails_rather_than_being_recorded():
    from md_tools.openmm.system import NAGL_AM1BCC_METHODS

    assert "am1bcc" not in NAGL_AM1BCC_METHODS
    source = (REPO / "src/md_tools/openmm/system.py").read_text(encoding="utf-8")
    assert "unsupported ligand_charge_method" in source


def test_the_charged_molecule_itself_reaches_the_record():
    """Which method ran is not the same as what it produced.

    net_charge_e, formal_charge and n_atoms are computed from the actually-charged molecule and
    were dropped with the scheme. They are what lets a reader check that two builds of the same
    input got the same charges -- the question a charge cache would eventually have to answer.
    """
    from md_tools.openmm.forcefield_record import _ligand_record

    record = _ligand_record(
        is_ligand=True,
        reported={"ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc",
                             "charge_scheme": "am1bcc", "net_charge_e": -1.1e-16,
                             "formal_charge": 0, "n_atoms": 79}},
        requested={"ligand_forcefield": "sage-2.2.1", "ligand_charge_method": "am1bcc"},
        checksums={},
    )
    assert record["formal_charge"] == 0
    assert record["n_atoms"] == 79
    assert record["net_charge_e"] is not None


def test_the_null_ligand_record_declares_every_charge_key():
    from md_tools.openmm.forcefield_record import _null_ligand

    for key in ("charge_method", "charge_scheme", "charge_model", "charge_model_sha256",
                "net_charge_e", "formal_charge", "n_atoms"):
        assert key in _null_ligand(), f"{key} must be present and null, so a reader can tell"


# ======================================================================================
# 3, 4, 5 -- the stage fingerprint and what it does and does not allow
# ======================================================================================


def test_the_fingerprint_ignores_only_the_extendable_fields():
    """Running longer is a legitimate extension; changing the physics is not.

    Migrated from `md_stages.stage_invariant_sha256`, which went with the retired launchers. The
    live fingerprint additionally binds the System and topology digests, so the same stage against
    a rebuilt System is correctly a different fingerprint.
    """
    from md_tools.md.stage import EXTENDABLE_FIELDS, _config_fingerprint

    base = {"timestep_fs": 2.0, "temperature_kelvin": 300.0, "steps": 500000, "ensemble": "NPT"}
    longer = dict(base, steps=1000000)
    faster = dict(base, timestep_fs=4.0)
    sys_sha, top_sha = "a" * 64, "b" * 64

    assert _config_fingerprint(base, sys_sha, top_sha) == _config_fingerprint(longer, sys_sha, top_sha), \
        "running longer is an extension and must not change the fingerprint"
    assert _config_fingerprint(base, sys_sha, top_sha) != _config_fingerprint(faster, sys_sha, top_sha), \
        "changing the timestep changes the physics the checkpoint was produced under"
    assert _config_fingerprint(base, sys_sha, top_sha) != _config_fingerprint(base, "c" * 64, top_sha), \
        "a rebuilt System must not look like the one the checkpoint came from"
    assert "steps" in EXTENDABLE_FIELDS


def test_every_extendable_field_is_excluded_and_nothing_else_is():
    """The retired list named five duration spellings because five generators wrote five names.

    One generator writes one name now -- `steps` -- so the live set is smaller by construction
    rather than by agreement. What must not change is which KIND of field is extendable: how long
    to run, and the human description. Every scientific invariant stays inside the fingerprint.
    """
    from md_tools.md.stage import EXTENDABLE_FIELDS

    assert set(EXTENDABLE_FIELDS) == {"steps", "description"}
    for field in ("timestep_fs", "temperature_kelvin", "tau", "ensemble", "implicit",
                  "hydrogen_mass_amu", "input_state", "barostat_frequency_steps"):
        assert field not in EXTENDABLE_FIELDS, (
            f"{field} is a scientific invariant and must be inside the fingerprint"
        )


def test_the_stage_request_is_derived_once_not_twice():
    """The generator and the runtime must not derive the same request separately.

    This used to say "md-gen and the launcher"; both are gone. The property survives between
    `build-md`, which derives the stage plan and writes it into each script, and the runtime, which
    READS that plan rather than recomputing it. Two derivations of one request is how a generated
    script and the run it describes drift apart.
    """
    from md_tools.build import md as build_md
    from md_tools.md import stage as runtime_stage

    generator = Path(build_md.__file__).read_text(encoding="utf-8")
    runtime = Path(runtime_stage.__file__).read_text(encoding="utf-8")

    assert "def stage_plan(" in generator, "the generator no longer derives the plan"
    # The plan is written into `resolved.config`, and the script names which stage of it to run.
    # One derivation, one file -- rather than a plan embedded in every script AND a configuration
    # beside them that could say something else.
    assert "run_generated_stage" in generator, "the script no longer reaches the runtime"
    assert "STAGE = {stage!r}" not in generator, "the plan is embedded in the script again"
    # The runtime consumes what it is given. If it started deriving stage lengths itself there
    # would be two answers to one question.
    assert "def stage_plan(" not in runtime
    assert 'stage["steps"]' in runtime or 'stage.get("steps")' in runtime


# `preflight.py` and the three launchers that called it are retired with the copy-based generated
# project. Their guarantees did not retire with them -- `md_tools.md.stage` is where they now
# live, and these are the same properties asserted against it.


def test_the_stage_runtime_checks_its_parent_before_it_builds_anything():
    """The defect this came from: `check_parent` short-circuited on `if not stage`, so production
    skipped the check entirely. The check must be reached for every stage, continued or not."""
    runtime = (REPO_ROOT / "src" / "md_tools" / "md" / "stage.py").read_text(encoding="utf-8")
    assert '"continued_from"' in runtime, "a continued stage no longer records its parent"
    assert "file_facts(parent)" in runtime, "the parent is recorded without being re-read"
    # It is recorded by content, not by name: a parent that was regenerated is a different parent.
    assert "sha256" in runtime


def test_only_the_stage_runtime_decides_whether_a_checkpoint_may_be_resumed():
    """No second implementation may grow beside it; that is what drifts."""
    root = REPO_ROOT / "src" / "md_tools"
    deciders = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")
                      if "def check_parent" in path.read_text(encoding="utf-8"))
    assert deciders == [], f"a parent check reappeared outside the stage runtime: {deciders}"


def test_the_stage_runtime_remains_bounded():
    """CLAUDE.md: the preflight never walked $MD_DATA, and its replacement must not either.

    A stage validates the files it was given. The moment it starts searching a managed store it
    can find something that merely looks like its parent.
    """
    source = (REPO_ROOT / "src" / "md_tools" / "md" / "stage.py").read_text(encoding="utf-8")
    for forbidden in ("rglob", "os.walk", "glob.glob", "MD_DATA"):
        assert forbidden not in source, f"the stage runtime must stay bounded; found {forbidden!r}"


# ======================================================================================
# Behavioural: a real generated project, refused, with the tree proven unchanged
# ======================================================================================


def _tree_state(root: Path) -> dict:
    """Size and mtime of every file, so a refusal that wrote anything is visible."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
















# --- the same safety properties, on the new generated scripts -------------------------------------
#
# The six tests that used to live here drove `sys-config` / `sys-gen` / `md-gen` and a generated
# project carrying a separate `stage.yaml`. That mechanism is gone: `build-md` writes the resolved
# settings INTO each script, so there is no second document to disagree with the first.
#
# The safety properties it protected are not gone, and are re-asserted below against the new
# interface. Nothing here was dropped for being inconvenient.

@pytest.fixture(scope="module")
def scripts(tmp_path_factory):
    """A built ALA system and a generated cMD chain. Built once; no dynamics are run."""
    if not (REPO / "tests" / "data" / "ALA.pdb").is_file():
        pytest.skip("no ALA fixture")
    work = tmp_path_factory.mktemp("scripts")

    def run(*args):
        return subprocess.run([sys.executable, "-m", "md_tools.cli.md_openmm", *args],
                              cwd=work, capture_output=True, text=True, timeout=1800)

    (work / "build.config").write_text("solvent:\n  model: GBn2\n")
    built = run("build-top", "-i", str(REPO / "tests" / "data" / "ALA.pdb"),
                "-os", "built.xml", "-op", "built.pdb", "-log", "built.log",
                "--config", "build.config")
    assert built.returncode == 0, built.stdout + built.stderr
    (work / "cMD.config").write_text(
        "protocol: cMD\nsolvent: implicit\n"
        "stages:\n  minimization_iterations: 10\n  restrained_nvt_steps: 20\n"
        "  restrained_npt_steps: 20\n  unrestrained_npt_steps: 20\n  production_steps: 20\n")
    generated = run("build-md", "-odir", "./md_script/", "--config", "cMD.config")
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work


def test_each_generated_script_names_its_stage_and_reads_the_resolved_plan(scripts):
    """The stage the script runs is derived from `resolved.config`, not embedded in the script.

    It used to be a `STAGE = {...}` literal in every file -- a SECOND declaration of the same
    workflow, which could disagree with the configuration beside it after a hand edit. The script
    names its stage and nothing else; the settings come from the one file.
    """
    from md_tools.build.md import resolve_md_config, stage_plan

    directory = scripts / "md_script"
    plan = {entry["name"]: entry for entry in
            stage_plan(resolve_md_config(directory / "resolved.config"))}
    for name in ("min", "eq_nvt_posres", "cMD"):
        text = (directory / f"{name}.py").read_text()
        assert f'run_generated_stage(__file__, "{name}")' in text
        assert "STAGE = {" not in text, "the script embeds a second declaration again"
        stage = plan[name]
        assert stage["ensemble"] in ("NVT", "NPT")
        assert stage["timestep_fs"] and stage["temperature_K"]
        assert isinstance(stage["steps"], int), "lengths are exact step counts"


def test_a_missing_parent_refuses_before_any_context_is_created(scripts):
    """The chain is sequential; a stage whose parent has not run must not start integrating."""
    work = scripts / "md_script"
    result = subprocess.run(
        [sys.executable, "cMD.py", "-p", "../built.pdb", "-s", "../built.xml",
         "-c", "eq_nvt_free.xml", "-log", "refusal.log"],
        cwd=work, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "does not exist" in result.stdout + result.stderr
    assert not (work / "cMD.dcd").exists(), "a refusal must not leave a trajectory"
    assert not (work / "cMD.chk").exists(), "a refusal must not leave a checkpoint"
    (work / "refusal.log").unlink(missing_ok=True)


def test_a_standalone_stage_with_a_missing_parent_is_refused_even_under_check(scripts):
    """MIGRATED. This asserted the opposite, and the opposite was the defect.

    A single split stage run on its own is not a chain. Nothing in this invocation will ever
    write `eq_nvt_free.xml`, so "the parent is merely pending" is a guess -- and it was the same
    guess for a genuine typo, which is how `-c eq_npt_fre.xml` became a run that continued
    nothing, started from the coordinates in `-p`, and reported success.

    The pending case is now something the all-in-one chain STATES, naming the stage that produces
    the file (see the test below). Here there is no such stage, so this is a refusal.

    It also wrote `pending.log` while doing it, which `--check` may no longer do at all.
    """
    work = scripts / "md_script"
    before = sorted(p.name for p in work.iterdir())
    result = subprocess.run(
        [sys.executable, "cMD.py", "-p", "../built.pdb", "-s", "../built.xml",
         "-c", "eq_nvt_free.xml", "-log", "pending.log", "--check"],
        cwd=work, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "eq_nvt_free.xml" in (result.stdout + result.stderr)
    assert not (work / "pending.log").exists(), "--check wrote a log"
    assert sorted(p.name for p in work.iterdir()) == before, "--check created something"

# The complementary case -- an all-in-one `--check` passing with every later parent absent, and
# writing nothing -- is `test_the_all_in_one_check_still_allows_the_parent_a_later_stage_will_write`
# in tests/test_runtime_contract_matrix.py, where an all-in-one project is generated. This
# fixture generates split stages, so asserting it here could only be a skip.


def test_a_large_timestep_without_hmr_is_refused_before_integrating(scripts):
    """The defect that silently altered a trajectory, re-asserted against the SYSTEM's masses.

    Checked against the masses actually serialised in built.xml rather than against a config that
    claims HMR, because by run time that is a fact rather than a request.
    """
    import shutil

    import yaml

    # `resolved.config` is the one declaration, so an explicit timestep is stated THERE. The
    # default is `auto`, which resolves to 2 fs against these ordinary hydrogen masses --
    # correctly, and therefore uselessly for this test. What is asserted is the refusal of an
    # explicit large value.
    #
    # Into a copy of the directory, because editing `resolved.config` in place would invalidate
    # the checkpoints the sibling tests share through this module-scoped fixture -- which is
    # itself the fingerprint rule working.
    work = scripts / "md_script_fast"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(scripts / "md_script", work)
    config = work / "resolved.config"
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document.setdefault("dynamics", {})["timestep_fs"] = 4.0
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    try:
        # `--cpu` because the subject is the MASSES, not the accelerator. The preflight resolves
        # the platform before it opens the System, so on a machine with no GPU the CUDA refusal
        # arrived first and this test failed without ever reaching the check it is named for.
        result = subprocess.run(
            [sys.executable, "min.py", "-p", "../built.pdb", "-s", "../built.xml",
             "-log", "fast.log", "--cpu"], cwd=work, capture_output=True, text=True, timeout=600)
        combined = result.stdout + result.stderr
        assert result.returncode != 0, combined
        assert "hydrogen mass repartitioning was NOT applied" in combined, combined[-800:]
        assert not (work / "min.dcd").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _resolved_stage(scripts, name):
    """The resolved stage a generated script runs, read from `resolved.config` beside it.

    The script no longer embeds it: `resolved.config` is the one declaration, and this is how a
    test reads what the script will actually run.
    """
    from md_tools.build.md import resolve_md_config, stage_plan

    plan = stage_plan(resolve_md_config(scripts / "md_script" / "resolved.config"))
    return next(entry for entry in plan if entry["name"] == name)


def test_asking_for_a_longer_run_does_not_invalidate_the_checkpoint(scripts):
    """Extension must survive. Only the INVARIANTS are fingerprinted, and steps is not one."""
    from md_tools.md.stage import EXTENDABLE_FIELDS, _config_fingerprint

    stage = _resolved_stage(scripts, "cMD")
    longer = {**stage, "steps": stage["steps"] * 4}
    assert _config_fingerprint(stage, "a", "b") == _config_fingerprint(longer, "a", "b")

    for field, value in (("temperature_K", 350.0), ("ensemble", "NPT"), ("timestep_fs", 1.0),
                         ("seed", 99), ("restraint_kcal_per_mol_A2", 5.0)):
        assert field not in EXTENDABLE_FIELDS
        mutated = {**stage, field: value}
        assert _config_fingerprint(stage, "a", "b") != _config_fingerprint(mutated, "a", "b"), (
            f"changing {field} must invalidate a checkpoint")


def test_a_checkpoint_from_a_different_system_is_refused(scripts):
    """A checkpoint carries positions for a particular particle set."""
    from md_tools.md.stage import _config_fingerprint

    stage = _resolved_stage(scripts, "cMD")
    assert _config_fingerprint(stage, "system-a", "top") != \
           _config_fingerprint(stage, "system-b", "top")


def test_a_stage_records_the_parent_state_it_consumed_with_its_digest(scripts):
    """The guarantee the retired preflight's parent check used to give, in its new home.

    `check_parent` compared a recorded parent fingerprint before a Context existed. That preflight
    branch went with the generated-project route. The property did not: a stage now records the
    `-c` state it consumed WITH ITS SHA-256, and registration re-hashes every recorded input and
    refuses a directory whose files no longer match its records -- so a parent rewritten after a
    child consumed it is caught, at the point where it would otherwise become false ancestry.
    """
    from md_tools.md import stage as stage_module

    source = Path(stage_module.__file__).read_text(encoding="utf-8")
    assert '"continued_from"' in source, "the parent state is not recorded"
    assert "file_facts(parent)" in source, "the parent is recorded without a digest"

    from md_tools.registry import discovery
    lineage = Path(discovery.__file__).read_text(encoding="utf-8")
    assert "hashes to" in lineage, "registration does not re-hash recorded inputs"
