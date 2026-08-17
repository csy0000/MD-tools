"""Phase 5 gate: the generic route reaches the same engine, and the catalog stays engine-free.

The claim this file has to establish is narrow and important: `md-templates` is a *router*, not a
second implementation. If it assembled its own argument list or applied its own defaults, the two
routes could agree today and diverge on the next edit, and the divergence would show up as a
scientific difference nobody was looking for.

So the equivalence is checked where it is cheap and total — the translated argument list — rather
than only end to end, and separately proven end to end in the slow gate.

The other half is that `list`, `inspect` and `identity` must work with no engine installed. That is
what makes the catalog useful on a machine that cannot simulate, and it is asserted by making OpenMM
unimportable rather than by reading imports.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from md_templates.core import load_catalog
from md_templates.core.dispatch import (
    PROVIDERS,
    DispatchError,
    DispatchNotActive,
    ProviderUnavailable,
    provider_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
MD_ID = "conventional-md/openmm/explicit-water"
REST2_ID = "rest2/openmm/explicit-water"

HEAVY = ("openmm", "openff", "rdkit", "mdtraj", "parmed", "openmmtools", "numpy", "scipy", "pandas")


def run_cli(*args: str, blocked: bool = False, cwd: Path | None = None):
    """Run `md-templates`, optionally with every heavy dependency made unimportable."""
    import os

    preamble = f"""
        import sys
        HEAVY = {HEAVY!r}

        class _Blocker:
            def find_module(self, name, path=None):
                if name.split('.')[0] in HEAVY:
                    raise ImportError(f"{{name}} is unimportable in this test")
                return None

        sys.meta_path.insert(0, _Blocker())
        """ if blocked else "import sys\n"
    script = textwrap.dedent(preamble) + textwrap.dedent(f"""
        from md_templates.cli import main
        raise SystemExit(main({list(args)!r}))
        """)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=600, env=env, cwd=str(cwd or REPO_ROOT))


# ------------------------------------------------------------------------------------------------
# engine-free catalog commands
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("args", [
    ("list",),
    ("list", "--format", "json"),
    ("inspect", MD_ID),
    ("inspect", REST2_ID, "--format", "json"),
])
def test_catalog_commands_work_with_no_engine_installed(args):
    result = run_cli(*args, blocked=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip()


def test_list_reports_both_templates_with_their_dispatch_state():
    result = run_cli("list", "--format", "json", blocked=True)
    assert result.returncode == 0, result.stderr
    entries = {e["template_id"]: e for e in json.loads(result.stdout)}
    assert set(entries) == {MD_ID, REST2_ID}
    assert entries[MD_ID]["dispatch"] == "catalog"
    assert entries[REST2_ID]["dispatch"] == "catalog"
    for entry in entries.values():
        assert entry["scientific_status"] == "unvalidated"


def test_inspect_reports_the_declared_contract():
    result = run_cli("inspect", MD_ID, "--format", "json", blocked=True)
    assert result.returncode == 0, result.stderr
    descriptor = json.loads(result.stdout)
    assert descriptor["template_id"] == MD_ID
    assert descriptor["profiles"]["template_local"] is True
    assert sorted(descriptor["profiles"]["profile_ids"]) == ["explicit-md-ligand-v1",
                                                            "explicit-md-peptide-v1"]


def test_no_heavy_module_is_imported_by_the_catalog_commands():
    """The blocker proves it, but assert the positive too: nothing heavy ends up loaded."""
    result = run_cli("list", blocked=True)
    assert result.returncode == 0
    assert "unimportable in this test" not in result.stderr


def test_identity_reports_or_refuses_but_never_invents():
    result = run_cli("identity", MD_ID, blocked=True)
    if result.returncode == 0:
        assert result.stdout.strip().startswith("https://github.com/csy0000/MD-templates@")
        assert "#templates/" in result.stdout
    else:
        assert result.returncode == 3
        assert "no immutable identity" in result.stderr


# ------------------------------------------------------------------------------------------------
# dispatch: explicit, and honest about what it cannot do
# ------------------------------------------------------------------------------------------------

def test_the_dispatch_table_is_a_table():
    assert set(PROVIDERS) == {("conventional-md", "openmm"), ("rest2", "openmm")}
    for binding in PROVIDERS.values():
        assert binding.module.startswith("md_templates.engines.")
        assert binding.requires == ("openmm",)


def test_an_unregistered_method_is_refused_with_the_registered_set():
    catalog = load_catalog(REPO_ROOT)
    descriptor = catalog.descriptor(MD_ID).model_copy(deep=True)
    descriptor.method.id = "umbrella-sampling"
    with pytest.raises(DispatchError, match="no provider is registered"):
        provider_for(descriptor)


def test_an_inactive_template_names_the_command_that_does_work():
    """Both shipped templates are active, so this uses a copy that is not."""
    catalog = load_catalog(REPO_ROOT)
    descriptor = catalog.descriptor(REST2_ID).model_copy(deep=True)
    descriptor.implementation.dispatch = "legacy-direct"
    with pytest.raises(DispatchNotActive) as exc:
        provider_for(descriptor)
    assert "md-openmm" in str(exc.value)


def test_a_missing_engine_is_reported_as_a_missing_engine():
    """Not as an ImportError from four frames down."""
    result = run_cli("prepare", "--template", MD_ID, "--", "--help", blocked=True)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "openmm" in result.stderr and "installed" in result.stderr
    assert "Traceback" not in result.stderr


# ------------------------------------------------------------------------------------------------
# the equivalence claim, checked on the translation rather than only end to end
# ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def descriptors():
    catalog = load_catalog(REPO_ROOT)
    return {tid: catalog.descriptor(tid) for tid in (MD_ID, REST2_ID)}


@pytest.mark.parametrize("action,expected", [
    ("prepare", "prepare"),
    ("run", "md"),
])
def test_the_generic_action_becomes_the_expected_engine_command(descriptors, action, expected):
    from md_templates.engines.openmm.provider import PROVIDER

    assert PROVIDER.command_for(action, descriptors[MD_ID]) == expected


def test_rest2_maps_run_to_the_rest2_command(descriptors):
    from md_templates.engines.openmm.provider import PROVIDER

    assert PROVIDER.command_for("run", descriptors[REST2_ID]) == "rest2"


def test_user_arguments_are_passed_through_untouched(descriptors):
    """The provider adds the subcommand and nothing else. No defaults, no rewriting."""
    from md_templates.engines.openmm.provider import PROVIDER

    user = ["--config", "run.yaml", "--out-root", "./runs", "--platform", "CPU", "--device", "0"]
    assert PROVIDER.argv_for("prepare", descriptors[MD_ID], user) == ["prepare", *user]
    assert PROVIDER.argv_for("prepare", descriptors[MD_ID], ["--", *user]) == ["prepare", *user]


def test_the_translated_argv_is_accepted_by_the_engine_parser(descriptors):
    """Equivalence is only meaningful if the engine would actually take these arguments."""
    from md_templates.engines.openmm.cli import build_parser
    from md_templates.engines.openmm.provider import PROVIDER

    user = ["--config", "run.yaml", "--out-root", "./runs", "--platform", "CPU"]
    argv = PROVIDER.argv_for("prepare", descriptors[MD_ID], user)
    parsed = build_parser().parse_args(argv)
    assert parsed.config == "run.yaml"
    assert parsed.platform == "CPU"


def test_resume_requires_the_run_to_be_named(descriptors):
    """Continuing 'the most recent run' is how the wrong directory gets extended."""
    from md_templates.engines.openmm.provider import PROVIDER

    with pytest.raises(ValueError, match="--resume-run"):
        PROVIDER.argv_for("resume", descriptors[MD_ID], ["--bundle", "B"])
    argv = PROVIDER.argv_for("resume", descriptors[MD_ID], ["--bundle", "B",
                                                            "--resume-run", "production_a"])
    assert argv[0] == "md" and "--resume-run" in argv


# ------------------------------------------------------------------------------------------------
# template-local profiles resolve identically wherever they are read from
# ------------------------------------------------------------------------------------------------

def test_every_profile_lives_in_the_template_that_owns_it():
    assert sorted(p.name for p in (REPO_ROOT / "templates" / MD_ID / "profiles").glob("*.json")) == [
        "explicit-md-ligand-v1.json", "explicit-md-peptide-v1.json"]
    assert sorted(p.name for p in (REPO_ROOT / "templates" / REST2_ID / "profiles").glob("*.json")) == [
        "cpu-smoke-v1.json", "explicit-rest2-ligand-v1.json", "explicit-rest2-peptide-v1.json"]

    builtin = SRC / "md_templates" / "core" / "config" / "profiles"
    remaining = sorted(p.name for p in builtin.glob("*.json")) if builtin.is_dir() else []
    assert remaining == [], f"profiles left behind in core: {remaining}"


def test_profile_discovery_never_consults_the_working_directory(tmp_path):
    """A resolver that walked up from cwd would find nothing from an unrelated directory."""
    result = run_cli("list", cwd=tmp_path, blocked=True)
    assert result.returncode == 0, result.stdout + result.stderr

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    probe = subprocess.run(
        [sys.executable, "-c",
         "from md_templates.core.config import resolve;"
         "print(sorted(d['profile_id'] for d in resolve.list_profiles()))"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert probe.returncode == 0, probe.stderr
    assert "explicit-md-ligand-v1" in probe.stdout


def test_all_five_profiles_are_visible_and_none_is_duplicated():
    from md_templates.core.config import resolve

    ids = [d["profile_id"] for d in resolve.list_profiles()]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 5


def test_two_profiles_claiming_one_id_is_refused(tmp_path, monkeypatch):
    """Silently ignoring one of them would depend on directory order."""
    from md_templates.core.config import resolve, sources

    duplicate = tmp_path / "profiles"
    duplicate.mkdir()
    source = REPO_ROOT / "templates" / MD_ID / "profiles" / "explicit-md-ligand-v1.json"
    (duplicate / "explicit-md-ligand-v1.json").write_bytes(source.read_bytes())

    monkeypatch.setattr(sources, "template_profile_directories",
                        lambda: [REPO_ROOT / "templates" / MD_ID / "profiles", duplicate])
    with pytest.raises(resolve.ResolutionError, match="two profiles claim the id"):
        resolve.list_profiles()
