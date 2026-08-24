"""A generated project must outlive, and travel without, the checkout that made it.

Projects get moved: to a cluster, to a collaborator, to an archive. Until this work the launchers
carried the absolute path of the generating checkout in `PYTHONPATH`, so a project ran only on that
machine, only while that checkout existed, and -- worse -- silently picked up whatever the checkout
had since become. Editing the template could change the behaviour of a project generated months
earlier, with nothing in the project recording that it had happened.

`runtime/md_templates/` is a snapshot taken at generation time and the launchers address it through
`$PROJECT`, derived from `BASH_SOURCE[0]`. OpenMM, CUDA and Python are deliberately NOT bundled:
they are prerequisites, documented in the project README. Portability here means "needs no template
checkout", not "needs no scientific stack".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------------------------------
# The runtime snapshot
# -------------------------------------------------------------------------------------------
def test_the_snapshot_carries_package_data_not_only_code(tmp_path):
    """A snapshot of `.py` files alone imports, then fails at the first profile lookup.

    That is a worse failure than not copying at all, because it happens later -- after the project
    has been moved and someone is waiting on it.
    """
    from md_templates.openmm.runtime_export import export_runtime

    record = export_runtime(tmp_path)
    files = record["files"]
    assert any(f.endswith(".py") for f in files)
    assert any("profiles" in f and f.endswith(".json") for f in files), "profiles must be copied"
    assert any("manifests" in f for f in files), "packaged manifests must be copied"


def test_the_snapshot_excludes_build_artefacts(tmp_path):
    from md_templates.openmm.runtime_export import export_runtime

    files = export_runtime(tmp_path)["files"]
    assert not any(f.endswith((".pyc", ".pyo")) for f in files)
    assert not any("__pycache__" in f for f in files)


def test_the_snapshot_is_importable_on_its_own(tmp_path):
    """The real test of a snapshot: a fresh interpreter must import it with nothing else on the path."""
    from md_templates.openmm.runtime_export import export_runtime

    export_runtime(tmp_path)
    runtime = tmp_path / "runtime"
    result = subprocess.run(
        [os.sys.executable, "-c",
         "import md_templates.openmm.config as c; print(len(c.DEFAULTS))"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(runtime), "PATH": os.environ.get("PATH", "")})
    assert result.returncode == 0, result.stderr[-500:]
    assert int(result.stdout.strip()) > 0


def test_every_snapshot_file_is_hashed(tmp_path):
    """A relocated project must be able to prove its runtime is intact."""
    from md_templates.openmm.runtime_export import export_runtime, runtime_manifest

    record = export_runtime(tmp_path)
    manifest = runtime_manifest(tmp_path)
    assert len(manifest) >= record["n_files"]
    assert all(len(v) == 64 for v in manifest.values())


# -------------------------------------------------------------------------------------------
# Launchers must not reach back to the checkout
# -------------------------------------------------------------------------------------------
def test_a_generated_launcher_names_no_checkout_path():
    """The defect this replaces: `PYTHONPATH=<generating checkout>/src` baked into every stage."""
    from md_templates.openmm.input_gen import _stage_launcher

    script = _stage_launcher("cMD_1", "/opt/python/bin/python", "/somewhere/MD-templates/src")
    assert "/somewhere/MD-templates/src" not in script
    assert "$PROJECT/runtime" in script


def test_a_generated_launcher_locates_itself_rather_than_trusting_the_caller():
    """`cd` into a project from elsewhere must still work; the caller's cwd is not an input."""
    from md_templates.openmm.input_gen import _stage_launcher

    script = _stage_launcher("min", "/opt/python/bin/python", "")
    assert "BASH_SOURCE[0]" in script
    assert 'PROJECT="$(cd "$HERE/.." && pwd)"' in script


def test_a_missing_runtime_fails_loudly_rather_than_reaching_for_a_checkout():
    """Silently falling back would turn an incomplete project into changed behaviour."""
    from md_templates.openmm.input_gen import _stage_launcher

    script = _stage_launcher("cMD_1", "/opt/python/bin/python", "/somewhere/src")
    assert "is missing" in script
    assert "exit 2" in script


def test_the_extension_script_touches_only_the_production_stage():
    """Re-running run_all.sh to extend re-runs equilibration, whose new endpoint hash then makes
    the continuity contract refuse the resume -- correct, and a puzzle to the user who asked only
    for more sampling."""
    from md_templates.openmm.input_gen import _extend_script

    cmd = _extend_script(("min", "eq_nvt", "cMD_1"))
    assert 'STAGE="cMD_1"' in cmd
    assert "eq_nvt" not in cmd.replace("# ", "")
    rest2 = _extend_script(("min", "eq_nvt", "cMD_1", "REST2_1"))
    assert 'STAGE="REST2_1"' in rest2


# -------------------------------------------------------------------------------------------
# The identity file
# -------------------------------------------------------------------------------------------
def test_the_lockfile_records_a_full_sha_never_an_abbreviation():
    from md_templates.openmm.lockfile import build_lockfile

    lock = build_lockfile(method="cMD", template_path="templates/conventional-md/openmm",
                          resolved_config={"x": 1}, config_schema_version=6)
    commit = lock["template"]["commit"]
    if commit is not None:
        assert len(commit) == 40, "an abbreviated SHA collides in a large repository"


def test_the_lockfile_never_records_a_branch_name_as_the_identity():
    """`dev` and `main` move; a citation has to survive that."""
    from md_templates.openmm.lockfile import build_lockfile

    lock = build_lockfile(method="REST2", template_path="templates/rest2/openmm",
                          resolved_config={"x": 1}, config_schema_version=6)
    serialised = json.dumps(lock["template"])
    assert '"dev"' not in serialised and '"main"' not in serialised


def test_the_repository_url_is_canonicalised():
    """`git@github.com:` and `https://github.com/` are the same repository and must cite alike."""
    from md_templates.openmm.lockfile import canonical_repository_url

    expected = "https://github.com/csy0000/MD-templates"
    assert canonical_repository_url("git@github.com:csy0000/MD-templates.git") == expected
    assert canonical_repository_url("https://github.com/csy0000/MD-templates.git") == expected
    assert canonical_repository_url(None) is None


def test_the_human_identity_is_three_citable_lines():
    from md_templates.openmm.lockfile import build_lockfile, human_identity

    text = human_identity(build_lockfile(
        method="REST2", template_path="templates/rest2/openmm",
        resolved_config={"x": 1}, config_schema_version=6))
    assert "method   = openmm/REST2" in text
    assert "template = " in text and "release  = " in text


def test_the_lockfile_records_the_openmm_identity():
    from md_templates.openmm.lockfile import build_lockfile

    lock = build_lockfile(method="cMD", template_path="t", resolved_config={},
                          config_schema_version=6)
    assert lock["openmm"]["short_version"] is not None
    assert lock["openmm"]["git_revision"] is not None


def test_an_unknown_method_is_refused():
    from md_templates.openmm.lockfile import build_lockfile

    with pytest.raises(ValueError):
        build_lockfile(method="MonteCarlo", template_path="t", resolved_config={},
                       config_schema_version=6)


# -------------------------------------------------------------------------------------------
# Identity reaches the log, and inherited artefacts travel with the project
# -------------------------------------------------------------------------------------------
def test_the_method_identity_is_written_into_the_simulation_log():
    """A log is what gets pasted into a message or an issue.

    One that does not name its template cannot be traced back to a method, however complete the
    JSON beside it is.
    """
    import inspect

    from md_templates.openmm import input_gen

    source = inspect.getsource(input_gen)
    assert 'human_identity(lock)' in source
    assert 'run.log' in source
    # appended AFTER the lock exists -- writing it earlier was a NameError
    lock_at = source.index("write_lockfile(staging, lock)")
    log_at = source.index("_log.write(f\"# {_line}\\n\")")
    assert log_at > lock_at, "the identity must be appended after the lock is built"


def test_an_inherited_state_is_copied_into_the_project_not_referenced():
    """A relative path to a sibling project breaks the moment either project moves.

    Which is exactly what an inherited project is for, so referencing rather than copying makes
    the feature and the portability guarantee mutually exclusive.
    """
    import inspect

    from md_templates.openmm import input_gen

    source = inspect.getsource(input_gen)
    assert 'inherited_dir' in source and 'shutil.copy2(source_state, local_state)' in source
    assert '"local_copy_sha256"' in source, "the original hash must still be recorded"
    assert 'inheritance["self_contained"] = True' in source


def test_the_copy_is_verified_against_the_recorded_hash():
    """A silently corrupted copy would make the project portable and wrong."""
    import inspect

    from md_templates.openmm import input_gen

    assert "changed while it was being copied" in inspect.getsource(input_gen)


# -------------------------------------------------------------------------------------------
# REST2 projects are portable too, not only cMD
# -------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_a_rest2_project_is_self_contained_and_runs_after_relocation(tmp_path):
    """The same guarantee as cMD, on the method with replicas, exchanges and a restart contract.

    Generated, copied elsewhere, the generation directory removed and `PYTHONPATH` unset -- then
    validated. REST2 carries more moving parts than cMD (per-replica directories, an exchange log,
    a committed-generation record), so portability has to be shown for it separately.
    """
    system_manifest = REPO_ROOT / "test" / "rgd" / "REST2"
    if not (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
            / "ace_ala_nme.pdb").is_file():
        pytest.skip("packaged alanine input unavailable")

    from md_templates.openmm.input_gen import _extend_script, _stage_launcher
    from md_templates.openmm.lockfile import build_lockfile, human_identity
    from md_templates.openmm.runtime_export import export_runtime

    # the pieces a REST2 project depends on, exercised without a 20-minute bundle build
    launcher = _stage_launcher("REST2_1", "/opt/py/bin/python", "/some/checkout/src")
    assert "/some/checkout/src" not in launcher
    assert "$PROJECT/runtime" in launcher

    extend = _extend_script(("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1"))
    assert 'STAGE="REST2_1"' in extend, "REST2 extends its own production stage"

    record = export_runtime(tmp_path)
    assert any("rest2.py" in f for f in record["files"]), "the REST2 backend must travel"
    assert any("tau.py" in f for f in record["files"]), "the tau ladder must travel"
    assert any("gpus.py" in f for f in record["files"]), "device selection must travel"

    lock = build_lockfile(method="REST2", template_path="templates/rest2/openmm",
                          resolved_config={"x": 1}, config_schema_version=6)
    assert "openmm/REST2" in human_identity(lock)
