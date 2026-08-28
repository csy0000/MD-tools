"""Regressions for the contract gates MD-project depends on.

Each of these would have failed before the fixes in this branch. They are implementation-level
tests: MD-project keeps a thin consumer gate that goes through generated files and public
commands, and the detail lives here.

The generation tests build a real system, so they carry the `gpu` marker where they integrate.
Everything that only inspects generated files runs anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "src" / "md_templates" / "openmm" / "templates"


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
    from md_templates.openmm.system import resolve_nagl_am1bcc_model

    pytest.importorskip("openff.nagl_models",
                        reason="the optional dependency is genuinely absent here")
    model = resolve_nagl_am1bcc_model()
    assert set(model) == {"name", "path", "sha256"}
    assert model["name"].endswith(".pt")
    assert len(model["sha256"]) == 64
    assert Path(model["path"]).is_file()


def test_no_module_named_hashing_is_imported_anywhere():
    """The narrowest statement of the defect, so a regression is obvious."""
    source = (REPO / "src/md_templates/openmm/system.py").read_text(encoding="utf-8")
    assert "md_templates.openmm.hashing" not in source
    assert not (REPO / "src/md_templates/openmm/hashing.py").exists(), (
        "the fix is to use provenance_min.sha256_file, not to add a second hashing module"
    )


def test_a_missing_optional_dependency_is_reported_as_such(monkeypatch):
    """An absent optional package must not look like a broken MD-templates."""
    import builtins

    from md_templates.openmm import system

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
    """A broken MD-templates import must propagate, not be relabelled as a user problem."""
    import builtins

    from md_templates.openmm import system

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if "provenance_min" in name:
            raise ModuleNotFoundError("No module named 'md_templates.openmm.provenance_min'")
        return real_import(name, *args, **kwargs)

    pytest.importorskip("openff.nagl_models")
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ModuleNotFoundError):
        system.resolve_nagl_am1bcc_model()


# ======================================================================================
# 2 -- the charge scheme round-trips into forcefield.json
# ======================================================================================


def test_the_ligand_record_carries_the_scheme_the_builder_reported():
    from md_templates.openmm.forcefield_record import _ligand_record

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
    from md_templates.openmm.forcefield_record import _ligand_record

    record = _ligand_record(
        is_ligand=True,
        reported={"ligand": {"forcefield": "openff-2.2.1", "charge_method": "am1bcc"}},
        requested={"ligand_forcefield": "sage-2.2.1", "ligand_charge_method": "am1bcc"},
        checksums={},
    )
    assert record["charge_scheme"] is None


def test_the_nagl_model_identity_reaches_the_record():
    """`nagl_model_file` is the key the builder writes; the record looked for `nagl_model`."""
    from md_templates.openmm.forcefield_record import _ligand_record

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
    from md_templates.openmm.system import NAGL_AM1BCC_METHODS

    assert "am1bcc" not in NAGL_AM1BCC_METHODS
    source = (REPO / "src/md_templates/openmm/system.py").read_text(encoding="utf-8")
    assert "unsupported ligand_charge_method" in source


def test_the_charged_molecule_itself_reaches_the_record():
    """Which method ran is not the same as what it produced.

    net_charge_e, formal_charge and n_atoms are computed from the actually-charged molecule and
    were dropped with the scheme. They are what lets a reader check that two builds of the same
    input got the same charges -- the question a charge cache would eventually have to answer.
    """
    from md_templates.openmm.forcefield_record import _ligand_record

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
    from md_templates.openmm.forcefield_record import _null_ligand

    for key in ("charge_method", "charge_scheme", "charge_model", "charge_model_sha256",
                "net_charge_e", "formal_charge", "n_atoms"):
        assert key in _null_ligand(), f"{key} must be present and null, so a reader can tell"


# ======================================================================================
# 3, 4, 5 -- the stage fingerprint and what it does and does not allow
# ======================================================================================


def test_the_invariant_fingerprint_ignores_only_the_extendable_fields():
    ms = _md_stages()
    base = {"timestep_fs": 2.0, "temperature_kelvin": 300.0, "duration_ns": 1.0,
            "total_steps": 500000, "ensemble": "NPT"}
    longer = dict(base, duration_ns=2.0, total_steps=1000000)
    faster = dict(base, timestep_fs=4.0)

    assert ms.stage_config_sha256(base) != ms.stage_config_sha256(longer), (
        "the full fingerprint distinguishes the exact request"
    )
    assert ms.stage_invariant_sha256(base) == ms.stage_invariant_sha256(longer), (
        "running longer is a legitimate extension and must not change the invariants"
    )
    assert ms.stage_invariant_sha256(base) != ms.stage_invariant_sha256(faster), (
        "changing the timestep changes the physics the checkpoint was produced under"
    )


def test_every_extendable_field_is_excluded_and_nothing_else_is():
    ms = _md_stages()
    assert set(ms.PRODUCTION_EXTENDABLE_FIELDS) == {
        "duration_ns", "duration_ps", "total_steps", "number_of_exchanges", "number_of_paths"}
    for field in ("timestep_fs", "temperature_kelvin", "tau", "ensemble", "implicit",
                  "hydrogen_mass_amu", "input_state", "barostat_frequency_steps"):
        assert field not in ms.PRODUCTION_EXTENDABLE_FIELDS, (
            f"{field} is a scientific invariant and must be inside the fingerprint"
        )


def test_the_production_stage_document_is_derived_once():
    """md-gen and the launcher must not derive the request separately."""
    mdgen = (REPO / "src/md_templates/openmm/mdgen.py").read_text(encoding="utf-8")
    assert "_template_module().production_stage_document(" in mdgen, (
        "md-gen must use the same derivation it copies into the project"
    )
    ms = _md_stages()
    assert hasattr(ms, "production_stage_document")


@pytest.mark.parametrize("launcher", ["cmd_run.py", "rest2_run.py", "rest2_equilibrate.py"])
def test_every_production_launcher_passes_its_stage_to_preflight(launcher):
    """The defect: check_parent short-circuits on `if not stage`, so production skipped it."""
    source = (TEMPLATES / launcher).read_text(encoding="utf-8")
    call = source[source.index("preflight.require("):]
    call = call[:call.index(")\n")]
    assert "stage=STAGE" in call, f"{launcher} must give preflight the resolved stage"
    assert "STAGE = " in source, f"{launcher} must read its stage.yaml"


def test_preflight_still_owns_parent_checking():
    """No launcher may reimplement it; that is what drifts."""
    for launcher in ("cmd_run.py", "rest2_run.py", "rest2_equilibrate.py"):
        source = (TEMPLATES / launcher).read_text(encoding="utf-8")
        assert "def check_parent" not in source, f"{launcher} must not duplicate parent checking"


def test_preflight_checks_the_runtime_request_for_production():
    source = (TEMPLATES / "preflight.py").read_text(encoding="utf-8")
    assert "def check_runtime_request" in source
    assert "check_runtime_request(here, stage, config)" in source, (
        "it must be wired into run(), not merely defined"
    )
    assert "stage_invariant_sha256" in source


def test_preflight_remains_bounded():
    """CLAUDE.md: preflight never walks $MD_DATA. The new check must not change that."""
    source = (TEMPLATES / "preflight.py").read_text(encoding="utf-8")
    for forbidden in ("rglob", "os.walk", "glob.glob", "iterdir", ".dcd"):
        assert forbidden not in source, f"preflight must stay bounded; found {forbidden!r}"


# ======================================================================================
# Behavioural: a real generated project, refused, with the tree proven unchanged
# ======================================================================================


def _tree_state(root: Path) -> dict:
    """Size and mtime of every file, so a refusal that wrote anything is visible."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """A tiny generated project. Built once; the refusal tests never run dynamics."""
    if not (REPO / "tests" / "data" / "ALA.pdb").is_file():
        pytest.skip("no ALA fixture")
    work = tmp_path_factory.mktemp("gen")
    run = lambda *a: subprocess.run([sys.executable, "-m", "md_templates.cli.md_openmm", *a],
                                    cwd=work, capture_output=True, text=True, timeout=1800)
    assert run("sys-config", "--method", "cMD", "--peptide", "true",
               "--solvent", "TIP3P", "--output-dir", ".").returncode == 0
    config = yaml.safe_load((work / "md.config.yaml").read_text())
    config["minimization"]["max_iterations"] = 20
    for key in ("nvt_restrained_duration_ps", "npt_restrained_duration_ps",
                "npt_free_duration_ps"):
        config["equilibration"][key] = 0.2
    config["cMD"].update(duration_ns=0.0004, whole_system_interval_ps=0.2,
                         solute_interval_ps=0.2, checkpoint_interval_ps=0.2)
    (work / "md.config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    built = run("sys-gen", "-i", str(REPO / "tests" / "data" / "ALA.pdb"),
                "--config", "sys.config.yaml", "-of", "./inputs/")
    assert built.returncode == 0, built.stdout + built.stderr
    generated = run("md-gen", "-if", "./inputs/", "--config", "md.config.yaml", "-of", "./MD/")
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work / "MD"


def test_md_gen_writes_a_stage_yaml_for_production(generated):
    stage = yaml.safe_load((generated / "cMD" / "stage.yaml").read_text())
    assert stage["production"] is True
    assert stage["name"] == "cMD"
    assert stage["parent"], "production must name the common stage it continues from"
    assert stage["input_state"], "and the handoff file it reads"
    assert stage["timestep_fs"] and stage["temperature_kelvin"]


def test_a_missing_parent_fails_a_real_run_before_a_context(generated):
    """The parent chain has not been run in this fixture, which is exactly the case."""
    before = _tree_state(generated / "cMD")
    result = subprocess.run([sys.executable, "run.py"], cwd=generated / "cMD",
                            capture_output=True, text=True, timeout=600)
    combined = result.stdout + result.stderr
    assert "[FAIL] parent stage" in combined, combined[-800:]
    assert "no Context was created" in combined
    assert _tree_state(generated / "cMD") == before, "a refusal must write nothing"


def test_a_missing_parent_is_a_skip_under_check(generated):
    """CLAUDE.md: pending under --check, because the chain is sequential and unrun."""
    result = subprocess.run([sys.executable, "run.py", "--check"], cwd=generated / "cMD",
                            capture_output=True, text=True, timeout=600)
    combined = result.stdout + result.stderr
    assert "[skip] parent stage" in combined, combined[-800:]
    assert "[FAIL]" not in combined


def test_an_edited_timestep_is_refused_and_writes_nothing(generated):
    """The defect that altered a trajectory: md-gen refuses 4 fs without HMR, the launcher did not."""
    config_path = generated / "md.config.yaml"
    original = config_path.read_text()
    before = _tree_state(generated)
    try:
        config = yaml.safe_load(original)
        config["common"]["timestep_fs"] = 4.0
        config["cMD"]["duration_ns"] = 0.01
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        result = subprocess.run([sys.executable, "run.py"], cwd=generated / "cMD",
                                capture_output=True, text=True, timeout=600)
        combined = result.stdout + result.stderr
        assert "[FAIL] runtime request" in combined, combined[-1000:]
        assert "timestep_fs" in combined, "the refusal must name the field that changed"
        assert "no Context was created" in combined
    finally:
        config_path.write_text(original)
    after = _tree_state(generated)
    assert {k: v for k, v in after.items() if k != "md.config.yaml"} == \
           {k: v for k, v in before.items() if k != "md.config.yaml"}, \
           "the refusal changed a file"


def test_a_longer_run_is_still_allowed(generated):
    """Extension must survive the fix. Only the invariants are compared."""
    config_path = generated / "md.config.yaml"
    original = config_path.read_text()
    try:
        config = yaml.safe_load(original)
        config["cMD"]["duration_ns"] = 0.0008
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        result = subprocess.run([sys.executable, "run.py", "--check"], cwd=generated / "cMD",
                                capture_output=True, text=True, timeout=600)
        combined = result.stdout + result.stderr
        assert "[FAIL] runtime request" not in combined, (
            "asking for a longer run is a legitimate extension:\n" + combined[-800:])
    finally:
        config_path.write_text(original)


def test_a_mutated_stage_request_is_refused(generated):
    """Recomputed, not merely present: the stored hash alone proves nothing."""
    stage_path = generated / "cMD" / "stage.yaml"
    original = stage_path.read_text()
    before = _tree_state(generated / "cMD")
    try:
        stage = yaml.safe_load(original)
        stage["temperature_kelvin"] = 350.0
        stage_path.write_text(yaml.safe_dump(stage, sort_keys=False))
        result = subprocess.run([sys.executable, "run.py"], cwd=generated / "cMD",
                                capture_output=True, text=True, timeout=600)
        combined = result.stdout + result.stderr
        assert "[FAIL]" in combined and "no Context was created" in combined, combined[-800:]
    finally:
        stage_path.write_text(original)
    # stage.yaml is excluded because THIS TEST rewrote it to restore the original; the refusal
    # itself must not have touched anything, which is what the rest of the tree shows.
    after = _tree_state(generated / "cMD")
    assert {k: v for k, v in after.items() if k != "stage.yaml"} == \
           {k: v for k, v in before.items() if k != "stage.yaml"}, \
           "the refusal changed a file"
    assert stage_path.read_text() == original, "and the restore is byte-exact"
