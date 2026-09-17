"""A protein-ligand run exports as a self-contained bundle: packages, mapping, and the System.

The run is tiny and on the CPU platform: what is tested is what the EXPORT carries and that the
relocated bundle runs with OpenMM alone -- no md-tools, no OpenFF, no openmmforcefields. It is not
evidence about CUDA or about the dynamics.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openff.toolkit")
pytest.importorskip("openmmforcefields")
pytest.importorskip("networkx")

from tests.test_ligand_mapping import _package, _write_structure  # noqa: E402

CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}

pytestmark = [pytest.mark.slow]


def _run(argv, cwd, env=None, timeout=1800):
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=env or os.environ)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done


def test_a_complex_run_exports_its_packages_and_mapping_and_runs_without_parameterisation(
        tmp_path):
    from md_tools.ligands.package import load_package
    from md_tools.reference import export_reference

    tyl = _package(tmp_path, "CC(=O)Nc1ccc(O)cc1", "CHEMBL112", "TYL")
    root = tmp_path / "project"
    root.mkdir()
    _write_structure(root / "complex.pdb", [("B", "201", "TYL", tyl, (1.2, 0.2, 0.1)),
                                            ("C", "201", "TYL", tyl, (0.1, 1.3, 0.2))])
    (root / "top.config").write_text(f"""
solute:
  kind: complex
ligands:
  - select: {{chain: B, resid: "201"}}
    parameters: {tyl.reference}
  - select: {{chain: C, resid: "201"}}
    parameters: {tyl.reference}
ligand_catalog:
  path: {tmp_path / 'catalog'}
solvent:
  model: TIP3P
  padding_nm: 1.0
""", encoding="utf-8")
    _run(CLI + ["build-top", "-i", "complex.pdb", "-os", "build/built.xml", "-op",
                "build/built.pdb", "-log", "build/built.log", "--config", "top.config"], root)
    (root / "cMD.config").write_text(
        "protocol: cMD\nsolvent: explicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 20, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 40}\n"
        "reporting: {crd_printout_solute: 20, info_printout: 20, checkpoint_printout: 40}\n",
        encoding="utf-8")
    _run(CLI + ["build-md", "-odir", "./run1", "--config", "cMD.config"], root)
    run = root / "run1"
    for source, odir, parent in (("../input/eq_1.in", "eq", None),
                                 ("../input/cMD.in", ".", "eq/eq_1.xml")):
        argv = CLI + ["md-run", "-i", source, "-p", "../build/built.pdb",
                      "-s", "../build/built.xml", "-odir", odir, "--cpu"]
        if parent:
            argv += ["-c", parent]
        _run(argv, run, env=ONE_THREAD)

    bundle = tmp_path / "bundle"
    export_reference(run, bundle)
    inputs = bundle / "input"
    package = load_package(inputs / "ligands" / "CHEMBL112" / tyl.parameter_id)
    assert package.metadata["parameter_digest"] == tyl.metadata["parameter_digest"]
    mapping = json.loads((inputs / "ligand_mapping.json").read_text())
    assert [i["selector"]["chain"] for i in mapping["instances"]] == ["B", "C"]
    sums = (bundle / "SHA256SUMS").read_text()
    for name in ("molecule.sdf", "parameters.ffxml", "metadata.json"):
        assert f"input/ligands/CHEMBL112/{tyl.parameter_id}/{name}" in sums
    assert "input/ligand_mapping.json" in sums
    settings = json.loads((inputs / "build_settings.json").read_text())
    assert "ligand mapping" in settings["unsupported"]
    assert not (inputs / "build_system.py").exists()

    # Relocated, and run with every parameterisation route and md-tools import-blocked.
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(bundle, moved)
    (moved / "_blocked.py").write_text(
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('md_tools', 'openff', 'openmmforcefields', 'rdkit'):\n"
        "            raise ImportError('bundle imported ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "sys.argv = ['run.py', '--steps', '10', '--platform', 'CPU']\n"
        "runpy.run_path('run.py', run_name='__main__')\n", encoding="utf-8")
    done = _run([sys.executable, "_blocked.py"], moved, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert "imported" not in done.stderr
