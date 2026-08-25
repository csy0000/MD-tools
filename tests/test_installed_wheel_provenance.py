"""A project generated from an INSTALLED WHEEL must still name the commit it came from.

Until this work, source identity came from running `git` in a checkout around the package. A wheel
installed into a clean environment has no `.git`, so that question had no answer there and the lock
file recorded `commit: null` -- which reads, to every later reader, exactly like a lock file that
was never asked. A run whose method cannot be named is not reproducible.

Identity is now resolved from three sources in order of trustworthiness (see
`provenance.source_identity`): a live git worktree, the stamp embedded at wheel-build time by
`build_backend/md_templates_build.py`, and PEP 610 `direct_url.json` vcs_info. These tests exercise
the middle one, because it is the one that had no coverage and the one a consuming repository
actually depends on.

The wheel build and venv creation are genuinely slow, so they happen once per module.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests"
               / "systems" / "ace_ala_nme.pdb")

#: Implicit solvent: no solvation step, so the system builds in seconds. This test is about
#: provenance, not chemistry -- the cheapest system that still exercises the real generator is the
#: right one.
SYSTEM_CONFIG = {
    "system": {"id": "ace_ala_nme", "type": "protein"},
    "solvation": {"mode": "implicit", "implicit_model": "GBn2", "radii": "mbondi3"},
    "randomness": {"master_seed": 20260825},
}

MD_CONFIG = {
    "profile": "implicit-md-peptide-v1",
    "protocol": {
        "integrator": {"kind": "langevin-middle", "timestep": "2 fs",
                       "temperature": "300 K", "friction": "1 /ps"},
        "equilibration": {"protocol": "simple", "minimize_max_iterations": 10,
                          "restrained": "2 ps"},
        "production": {"method": "md", "duration_per_segment": "2 ps"},
    },
    "minimization": {"restraint": {"force_constant_kcal_per_mol_angstrom2": 1.0}},
    "randomness": {"master_seed": 20260825},
    "execution": {"platform": "CPU", "precision": "mixed",
                  "reporting": {"all_atom": "1 ps", "solute": "1 ps"}},
}


@pytest.fixture(scope="module")
def wheel_install(tmp_path_factory):
    """Build a wheel from this checkout and install it into a venv OUTSIDE the checkout.

    `--system-site-packages` so OpenMM and the scientific stack are available: the point of the
    test is that MD-templates identity survives, not that OpenMM can be pip-installed. The wheel is
    installed with `--no-deps --force-reinstall` so it, and not any editable install of this same
    repository, is what the venv imports -- which the first test asserts rather than assumes.
    """
    work = tmp_path_factory.mktemp("wheelprov")
    dist = work / "dist"
    build = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist),
                            str(REPO_ROOT)], capture_output=True, text=True)
    if build.returncode != 0:
        pytest.fail(f"wheel build failed:\n{build.stdout[-3000:]}\n{build.stderr[-3000:]}")
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    venv = work / "venv"
    mk = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
                        capture_output=True, text=True)
    assert mk.returncode == 0, mk.stderr[-2000:]
    pip = venv / "bin" / "pip"
    install = subprocess.run([str(pip), "install", "--no-deps", "--force-reinstall",
                              str(wheels[0])], capture_output=True, text=True)
    assert install.returncode == 0, install.stdout[-2000:] + install.stderr[-2000:]
    return {"venv": venv, "python": venv / "bin" / "python", "work": work, "wheel": wheels[0]}


def _run_outside(install, *args, cwd=None):
    """Run a command from the venv with the repository NOT on the path and NOT as cwd.

    cwd defaults to the venv directory: running from inside the checkout would let `git_state()`
    walk up and find `.git`, which is precisely the fallback this test must not use.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([str(install["python"]), *args], capture_output=True, text=True,
                          cwd=str(cwd or install["venv"]), env=env)


def test_the_venv_imports_the_wheel_not_the_checkout(wheel_install):
    """If this fails every other assertion in this file is meaningless."""
    res = _run_outside(wheel_install, "-c",
                       "import md_templates; print(md_templates.__file__)")
    assert res.returncode == 0, res.stderr[-2000:]
    loaded = Path(res.stdout.strip())
    assert str(wheel_install["venv"]) in str(loaded), (
        f"the venv imported {loaded}, not its own installed copy")
    assert str(REPO_ROOT / "src") not in str(loaded)


def test_identity_comes_from_the_build_stamp_with_a_full_commit(wheel_install):
    """The core requirement: full SHA and repository URL, with no `.git` anywhere in reach."""
    res = _run_outside(wheel_install, "-c",
                       "import json;from md_templates.openmm import provenance as p;"
                       "print(json.dumps(p.source_identity()))")
    assert res.returncode == 0, res.stderr[-2000:]
    identity = json.loads(res.stdout)
    assert identity["identity_source"] == "build-stamp", (
        f"expected the embedded stamp, got {identity.get('identity_source')!r} -- if this says "
        "'git-worktree' the subprocess found a checkout and the test proved nothing")
    assert len(identity["commit"]) == 40, identity
    assert identity["commit"].strip("0123456789abcdef") == "", "not a hex SHA"
    assert "MD-templates" in (identity["remote_url"] or "")


def test_a_generated_project_records_that_identity_in_its_lock_file(wheel_install):
    """End to end: generate a real project from the wheel and read its lock file."""
    work = wheel_install["work"]
    syscfg = work / "system_config.json"
    syscfg.write_text(json.dumps(SYSTEM_CONFIG))
    mdcfg = work / "md_config.json"
    mdcfg.write_text(json.dumps(MD_CONFIG))
    system_dir = work / "system"

    gen = _run_outside(wheel_install, "-m", "md_templates.openmm.cli_system_gen",
                       "-i", str(ALANINE_PDB), "-o", str(system_dir), "--config", str(syscfg))
    assert gen.returncode == 0, gen.stdout[-3000:] + gen.stderr[-3000:]

    project = work / "project"
    inp = _run_outside(wheel_install, "-m", "md_templates.openmm.cli_input_gen",
                       "--system", str(system_dir / "system_manifest.json"),
                       "--outdir", str(project), "--config", str(mdcfg))
    assert inp.returncode == 0, inp.stdout[-3000:] + inp.stderr[-3000:]

    lock = json.loads((project / "md-template.lock.json").read_text())
    template = lock["template"]
    assert len(template["commit"]) == 40, template
    assert template["repository"] == "https://github.com/csy0000/MD-templates"
    assert template["identity_source"] == "build-stamp"
    assert template["commit"] is not None, "the whole point: never `commit: null`"
    # dirty is recorded either way, but it must be RECORDED, not absent
    assert "dirty" in template
    assert lock["python_version"]
    assert lock["openmm"]["short_version"]

    # METHOD_IDENTITY.txt is the human-readable half and must survive too
    identity_txt = (project / "METHOD_IDENTITY.txt").read_text()
    assert template["commit"] in identity_txt


def test_generation_refuses_rather_than_writing_a_null_commit(wheel_install, tmp_path):
    """With the stamp removed and no checkout in reach, generation must FAIL, not write null.

    A lock file recording `commit: null` is worse than a failed generation: the run completes,
    looks fine, and is untraceable months later when someone tries to reproduce it.
    """
    res = _run_outside(wheel_install, "-c", """
import json, pathlib, md_templates
stamp = pathlib.Path(md_templates.__file__).parent / "_build_info.py"
backup = stamp.read_text()
stamp.unlink()
try:
    from md_templates.openmm import provenance as p, lockfile
    try:
        lockfile.build_lockfile(method="cMD", template_path="t", resolved_config={},
                                config_schema_version=6)
        print("NO_RAISE")
    except p.UnknownSourceIdentityError as exc:
        print("RAISED")
        print(str(exc))
finally:
    stamp.write_text(backup)
""")
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.startswith("RAISED"), res.stdout[:500]
    # the message must be actionable, not just a refusal
    assert "pip install" in res.stdout
    assert "build stamp" in res.stdout
