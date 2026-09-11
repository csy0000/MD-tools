"""`input/build_system.py` rebuilds build-top's System without md-tools, byte for byte.

The script is a second implementation of build-top's steps, written as plain library calls so a
reader can see them and so a bundle can be rebuilt in `openmm-env` without the `md-openmm` command.
Second implementations drift. So each route is built BOTH ways here -- `md-openmm build-top`, then
the script under an import hook that refuses md_tools -- and the Systems must be the same bytes.
Equality is possible because every random step in build-top is seeded: the SMILES embedding, the
hydrogens `addHydrogens` places, the waters `addSolvent` swaps for ions.

The topology is compared without its first line, which `PDBFile.writeFile` stamps with the date.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
SCRIPT = REPO / "src" / "md_tools" / "reference" / "standalone_build.py"

#: (structure, build-top configuration). Explicit ALA with HMR takes OpenMM's own repartitioning
#: path; the ligand routes take the SMILES embedding and AM1-BCC charges through AmberTools.
CASES = {
    "peptide-implicit": ("ALA.pdb", "solvent:\n  model: GBn2\n"),
    "peptide-explicit-hmr": ("ALA.pdb", "hydrogen_mass_repartitioning:\n  enabled: true\n"),
    "ligand-implicit": ("phenol.smi", "solute:\n  kind: ligand\nsolvent:\n  model: GBn2\n"),
    "ligand-explicit": ("phenol.smi", "solute:\n  kind: ligand\n"),
}

BLOCKER = (
    "import sys, runpy\n"
    "class B:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] == 'md_tools':\n"
    "            raise ImportError('build_system.py imported md_tools: ' + name)\n"
    "        return None\n"
    "sys.meta_path.insert(0, B())\n"
    "sys.argv = ['input/build_system.py', '--out', 'rebuilt']\n"
    "runpy.run_path('input/build_system.py', run_name='__main__')\n")


def test_the_script_imports_nothing_from_md_tools():
    """It is copied into bundles that must run where md-tools is not installed."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"line {node.lineno}: a relative import"
            assert (node.module or "").split(".")[0] != "md_tools", f"line {node.lineno}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "md_tools", f"line {node.lineno}"


def test_its_seeds_are_build_tops_seeds():
    from md_tools.openmm.seeds import derive_build_seed
    from md_tools.reference import standalone_build

    for master in (1, 20260814, 20260824003):
        for purpose in ("structure/protonation", "structure/solvation", "anything"):
            assert (standalone_build.derive_build_seed(master, purpose)
                    == derive_build_seed(master, purpose))


def _built(case: str, root: Path) -> Path:
    """build-top in `root/case`, then the input/ directory an export would write beside it."""
    from md_tools.build.record import read_record
    from md_tools.reference.export import STANDALONE_BUILD, standalone_settings

    structure, config = CASES[case]
    work = root / case
    work.mkdir()
    if structure.endswith(".pdb"):
        shutil.copy2(ALA, work / structure)
    else:
        (work / structure).write_text("Oc1ccccc1 phenol\n", encoding="utf-8")
    (work / "sys.config").write_text(config, encoding="utf-8")
    done = subprocess.run(CLI + ["build-top", "-i", structure, "-os", "built.xml", "-op",
                                 "built.pdb", "-log", "built.log", "--config", "sys.config"],
                          cwd=work, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    inputs = work / "input"
    inputs.mkdir()
    for name in (structure, "built.xml", "built.pdb"):
        shutil.copy2(work / name, inputs / name)
    settings = standalone_settings(read_record(work / "built.log"), inputs / structure,
                                   inputs / "built.xml", inputs / "built.pdb")
    (inputs / "build_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    shutil.copy2(STANDALONE_BUILD, inputs / "build_system.py")
    (work / "_blocked.py").write_text(BLOCKER, encoding="utf-8")
    return work


def _rebuild(work: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "_blocked.py"], cwd=work, capture_output=True,
                          text=True, timeout=1800, env={**__import__("os").environ,
                                                        "PYTHONPATH": ""})


def _undated(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("REMARK   1 CREATED WITH OPENMM")]


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_it_rebuilds_build_tops_system_byte_for_byte(case, tmp_path):
    work = _built(case, tmp_path)
    done = _rebuild(work)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "imported md_tools" not in done.stderr
    assert (work / "rebuilt" / "built.xml").read_bytes() == (work / "built.xml").read_bytes()
    assert _undated(work / "rebuilt" / "built.pdb") == _undated(work / "built.pdb")


@pytest.mark.slow
def test_a_rebuild_that_differs_is_reported_and_fails(tmp_path):
    """The check is the point of the script: a different System must not exit 0."""
    work = _built("peptide-implicit", tmp_path)
    path = work / "input" / "build_settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["builder"]["system_build"]["constraints"] = "AllBonds"
    path.write_text(json.dumps(settings), encoding="utf-8")
    done = _rebuild(work)
    assert done.returncode == 1, done.stdout[-3000:] + done.stderr[-3000:]
    assert "DIFFERS from built.xml" in done.stdout


def test_a_peptide_like_solute_in_implicit_solvent_is_said_to_be_unsupported(tmp_path):
    """Its mbondi3 corrections come from md-tools' peptide map, which the script does not carry.

    Said in the settings and refused by the script, rather than built without the corrections.
    """
    from md_tools.build.top import recorded_configuration
    from md_tools.reference.export import standalone_settings

    config = tmp_path / "sys.config"
    config.write_text("solute:\n  kind: peptide-like\nsolvent:\n  model: GBn2\n",
                      encoding="utf-8")
    resolved, _stated = recorded_configuration(config)
    for name in ("in.smi", "built.xml", "built.pdb"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    settings = standalone_settings({"resolved_config": resolved}, tmp_path / "in.smi",
                                   tmp_path / "built.xml", tmp_path / "built.pdb")
    assert "peptide map" in settings["unsupported"]


@pytest.mark.slow
def test_a_tleap_sequence_structure_is_recognised_and_an_edited_one_is_not(tmp_path):
    """ALA.pdb is tleap's `sequence { ACE ALA NME }` output; one moved atom and it is not."""
    from md_tools.reference.export import leap_sequence_origin

    if shutil.which("tleap") is None:
        pytest.fail("tleap is not on PATH; activate openmm-env, which provides AmberTools")
    origin = leap_sequence_origin(ALA)
    assert origin is not None and origin["sequence"] == ["ACE", "ALA", "NME"]
    assert "sequence { ACE ALA NME }" in origin["script"]

    lines = ALA.read_text(encoding="utf-8").splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[index] = lines[index][:30] + f"{float(lines[index][30:38]) + 0.001:8.3f}" + lines[index][38:]
    edited = tmp_path / "ALA.pdb"
    edited.write_text("".join(lines), encoding="utf-8")
    assert leap_sequence_origin(edited) is None
