"""`build-top` with `solvent.model: vacuum`: the vacuum leg of an alchemical cycle, and only that.

A vacuum build is the ligand alone -- NoCutoff, no box, no ions, constraints as configured --
made through the same route and the same build record as a solvated one, so a hydration cycle's
two legs come from one construction path. Everything else in vacuum is refused by name: a
peptide, protein, peptide-like solute or complex; any stated box, cutoff or ion key; a non-zero
ionic strength; and `build-md` given a vacuum System.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openmm")

from tests.alchemy_fixtures import (CHLOROETHANE, ETHANE, FIXTURE_ROOT,  # noqa: E402
                                    core_map, package, water_environment)

VACUUM = FIXTURE_ROOT.parent / "vacuum-v1"


def _resolve(tmp_path: Path, document: dict):
    from md_tools.build.top import resolve_build_config

    path = tmp_path / "build.config"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return resolve_build_config(path)


def test_the_example_vacuum_config_resolves_through_the_real_resolver():
    from md_tools.build.top import resolve_build_config

    resolved = resolve_build_config(VACUUM / "build.config")
    assert resolved["solvent"]["model"] == "vacuum"
    assert resolved["solute"]["kind"] == "ligand"
    assert resolved["constraints"]["type"] == "HBonds"


@pytest.mark.parametrize("kind", ["peptide", "peptide-like", "complex"])
def test_vacuum_is_refused_for_anything_but_a_ligand(tmp_path, kind):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match=f"solute.kind is '{kind}'.*vacuum leg"):
        _resolve(tmp_path, {"solute": {"kind": kind}, "solvent": {"model": "vacuum"}})


@pytest.mark.parametrize("key,value", [("padding_nm", 1.0), ("box_shape", "cube"),
                                       ("cutoff_nm", 0.9), ("positive_ion", "K+"),
                                       ("negative_ion", "Br-")])
def test_a_stated_box_or_ion_key_is_refused_not_ignored(tmp_path, key, value):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match=f"solvent.{key} is set, but solvent.model is vacuum"):
        _resolve(tmp_path, {"solute": {"kind": "ligand"},
                            "solvent": {"model": "vacuum", key: value}})


def test_salt_is_refused_and_zero_salt_is_not(tmp_path):
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="ionic_strength_molar is 0.15 but solvent.model is vacuum"):
        _resolve(tmp_path, {"solute": {"kind": "ligand"},
                            "solvent": {"model": "vacuum", "ionic_strength_molar": 0.15}})
    resolved = _resolve(tmp_path, {"solute": {"kind": "ligand"},
                                   "solvent": {"model": "vacuum", "ionic_strength_molar": 0.0}})
    assert resolved["solvent"]["model"] == "vacuum"


def test_a_vacuum_ethane_build_is_an_environment_and_the_vacuum_leg_of_a_tip3p_cycle(tmp_path):
    """Built by `build-top` now, from the v1 ethane package, with MD_DATA an empty temp root."""
    from openmm import XmlSerializer

    from md_tools.alchemy.topology import Environment, build_topology_plan, matched_legs
    from md_tools.build.record import read_record
    from md_tools.ligands.mapping import LigandSelector

    work = tmp_path / "work"
    (work / "input").mkdir(parents=True)
    (work / "build").mkdir()
    shutil.copy(FIXTURE_ROOT / "inputs" / "ETA.sdf", work / "input" / "ETA.sdf")
    config = yaml.safe_load((VACUUM / "build.config").read_text())
    config["solute"]["parameters"] = str(FIXTURE_ROOT / "packages" / ETHANE)
    (work / "build" / "build.config").write_text(yaml.safe_dump(config))
    md_data = tmp_path / "md_data"
    md_data.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", "input/ETA.sdf",
         "--config", "build/build.config", "-os", "build/built.xml", "-op", "build/built.pdb",
         "-log", "build/built.log"],
        cwd=work, capture_output=True, text=True, timeout=900,
        env={**os.environ, "MD_DATA": str(md_data)})
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert not any(md_data.iterdir())

    build = work / "build"
    record = read_record(build / "built.log")
    assert record["solvent"]["treatment"] == "vacuum" and record["periodic"] is False
    nonbonded = record["forcefield_record"]["nonbonded"]
    assert nonbonded["method"] == "NoCutoff"
    assert nonbonded["cutoff_nm"] is None and nonbonded["ewald_error_tolerance"] is None
    system = XmlSerializer.deserialize((build / "built.xml").read_text())
    assert not system.usesPeriodicBoundaryConditions()
    assert system.getNumParticles() == 8 and system.getNumConstraints() == 6   # HBonds, as asked

    env = Environment.from_files(build / "built.xml", build / "built.pdb",
                                 LigandSelector(resname="ETA"), record=build / "built.log")
    applied = env.nonbonded_compatibility["packages"][0]["applied"]
    assert (applied["coulomb14scale"], applied["lj14scale"]) == (5 / 6, 0.5)
    a, b = package(ETHANE), package(CHLOROETHANE)
    vacuum = build_topology_plan(a, b, core_map(a, b), env, mode="hybrid")
    solvent = build_topology_plan(a, b, core_map(a, b), water_environment(), mode="hybrid")
    assert matched_legs(vacuum, solvent)["ligand_hamiltonian_sha256"] == \
        vacuum.record["ligand_hamiltonian_sha256"]


def test_the_committed_vacuum_fixture_is_record_bearing():
    from md_tools.alchemy.topology import Environment
    from md_tools.ligands.mapping import LigandSelector

    env = Environment.from_files(VACUUM / "built.xml", VACUUM / "built.pdb",
                                 LigandSelector(resname="ETA"), record=VACUUM / "built.log")
    assert env.compatibility_source == "build record built.log"


def test_an_opc_leg_and_the_vacuum_build_are_refused_by_scale():
    from md_tools.alchemy.topology import (Environment, TopologyError, build_topology_plan,
                                           matched_legs)
    from md_tools.ligands.mapping import LigandSelector
    from tests.alchemy_fixtures import CMAP_ROOT, complex_environment

    a, b = package(ETHANE), package(CHLOROETHANE)
    vacuum = build_topology_plan(a, b, core_map(a, b), Environment.from_files(
        VACUUM / "built.xml", VACUUM / "built.pdb", LigandSelector(resname="ETA"),
        record=VACUUM / "built.log"), mode="hybrid")
    opc = build_topology_plan(a, b, core_map(a, b), complex_environment(CMAP_ROOT), mode="hybrid")
    with pytest.raises(TopologyError, match="different 1-4 scales.*TIP3P"):
        matched_legs(opc, vacuum)


def test_build_md_refuses_a_vacuum_system(tmp_path):
    from tests.conftest import make_dataset_root

    make_dataset_root(tmp_path, solvent="implicit")
    shutil.copy(VACUUM / "built.xml", tmp_path / "build" / "built.xml")
    shutil.copy(VACUUM / "built.pdb", tmp_path / "build" / "built.pdb")
    config = tmp_path / "p.config"
    config.write_text(yaml.safe_dump({"protocol": "cMD", "solvent": "implicit"}))
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", "cMD-run1",
         "--config", str(config)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "Vacuum builds are alchemical legs; ordinary MD in vacuum is not supported" in \
        result.stdout + result.stderr
    assert not (tmp_path / "cMD-run1").exists()
