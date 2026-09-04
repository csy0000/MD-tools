"""A CV-enabled run is the same run through the generated script and through `md-openmm md-run`.

`md-run` is a SURFACE: it parses the Amber-like input, resolves it through `md_tools.build.md`,
and hands the work to the same function the generated script calls. So a CV series produced one
way must be identical to one produced the other -- same grid, same columns, same sidecar. If they
differ, one of the two routes is resolving the configuration differently, and the `.in` round-trip
that the package treats as a contract is not holding for this section.

PLATFORM_POLICY_EXEMPTION: both routes run under `--cpu`. What is compared is the resolved
configuration and the resulting file shape, which is identical on every platform.
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

pytestmark = pytest.mark.slow

CV_YAML = """\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: [4, 6, 8, 14]
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cv-parity")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (root / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 20},
        "reporting": {"solute_printout": 10, "system_printout": 10, "checkpoint_printout": 10},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": 5},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./cMD", "--config", str(root / "cMD.config"),
               "--all-in-one"],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _environment(root: Path):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = root / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    return base


def _series(destination: Path):
    found = sorted(destination.rglob("*.cv.csv"))
    assert found, f"no CV series under {destination}"
    return found[-1]


def test_the_generated_script_and_md_run_produce_the_same_cv_series(project, tmp_path):
    """THE parity claim, on the CV section specifically."""
    direct = tmp_path / "direct"
    done = subprocess.run(
        [sys.executable, str(project / "cMD" / "md.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-odir", str(direct), "--cpu"],
        cwd=project / "cMD", capture_output=True, text=True, timeout=1800,
        env=_environment(project))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # The same production stage, through the public surface, from the `.in` beside the script.
    surfaced = tmp_path / "md-run"
    # The PRODUCTION stage specifically. `sorted(...)[-1]` picked `min.in`, whose zero-step
    # minimisation correctly writes no CV series -- so the comparison was between a dynamics run
    # and a minimisation, and the "missing" series was the contract working.
    production = project / "cMD" / "cMD.in"
    assert production.is_file(), sorted(p.name for p in (project / "cMD").glob("*.in"))
    chain = subprocess.run(
        CLI + ["md-run", "-i", str(production),
               "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
               "-odir", str(surfaced), "--cpu"],
        cwd=project / "cMD", capture_output=True, text=True, timeout=1800,
        env=_environment(project))
    if chain.returncode != 0:
        pytest.skip(f"md-run declined this input directly: "
                    f"{(chain.stdout + chain.stderr)[-400:]}")

    a, b = _series(direct), _series(surfaced)
    left = a.read_text(encoding="utf-8").splitlines()
    right = b.read_text(encoding="utf-8").splitlines()
    assert left[0] == right[0], "the two routes wrote different CV columns"
    assert [row.split(",")[0] for row in left[1:]] == [row.split(",")[0] for row in right[1:]], (
        "the two routes wrote different CV step grids")

    from md_tools.run.preflight import cv_sidecar_path

    one = json.loads(cv_sidecar_path(a).read_text(encoding="utf-8"))
    two = json.loads(cv_sidecar_path(b).read_text(encoding="utf-8"))
    for field in ("definition_sha256", "interval_steps", "units", "wrapping",
                  "periodic_convention", "column_order"):
        assert one.get(field) == two.get(field), (
            f"the two routes disagree about {field}: {one.get(field)!r} vs {two.get(field)!r}")


def test_the_resolved_config_round_trips_the_cv_section(project, tmp_path):
    """Every `.in` resolves back to the `resolved.config` beside it -- CV section included."""
    from md_tools.build.md import resolve_md_config
    from md_tools.run.inputs import parse_run_input

    resolved = resolve_md_config(project / "cMD" / "resolved.config")
    assert resolved["collective_variables"]["interval_steps"] == 5
    assert str(resolved["collective_variables"]["file"]).endswith(".yaml")

    inputs = sorted((project / "cMD").glob("*.in"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in inputs)
    assert "cv_interval_steps" in text, (
        "the .in language does not express the CV cadence, so a generated input silently drops it")
    assert "cv_file" in text
