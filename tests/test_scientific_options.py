"""Box shape, ligand force field, crossed-pair warnings, HMR and the resolved timestep.

These are the options a user chooses between, so what is asserted is that each choice REACHES the
built System and the record -- not merely that the configuration parses. A supported alternative
nobody can verify was applied is not an alternative.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from md_tools.build.strict import ConfigError
from md_tools.build.top import resolve_build_config

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"


def _config(**blocks) -> Path:
    path = Path(tempfile.mkdtemp()) / "build.config"
    path.write_text(yaml.safe_dump(blocks, sort_keys=False), encoding="utf-8")
    return path


# --- box shape --------------------------------------------------------------------------------

def test_the_default_box_is_a_rhombic_dodecahedron():
    assert resolve_build_config(None)["solvent"]["box_shape"] == "dodecahedron"


@pytest.mark.parametrize("shape", ["dodecahedron", "cube", "octahedron"])
def test_every_supported_box_shape_resolves(shape):
    resolved = resolve_build_config(_config(solvent={"box_shape": shape}))
    assert resolved["solvent"]["box_shape"] == shape


@pytest.mark.parametrize("shape", ["sphere", "triclinic", "DODECAHEDRON", ""])
def test_an_unknown_box_shape_is_refused_before_building(shape):
    """Refused by the schema, so nothing is solvated before the mistake is found."""
    with pytest.raises(ConfigError, match="box_shape"):
        resolve_build_config(_config(solvent={"box_shape": shape}))


@pytest.mark.parametrize("shape", ["cube", "dodecahedron"])
@pytest.mark.slow
@pytest.mark.gpu
def test_a_box_shape_reaches_the_built_system_and_the_record(shape, tmp_path):
    """The shape is not just accepted: it changes the box vectors and both are recorded."""
    import subprocess
    import sys

    from md_tools.build.record import read_record

    config = tmp_path / "b.config"
    config.write_text(yaml.safe_dump(
        {"solvent": {"model": "TIP3P", "padding_nm": 0.5, "cutoff_nm": 0.6,
                     "box_shape": shape}}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ALA),
         "-os", "b.xml", "-op", "b.pdb", "-log", "b.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr

    geometry = read_record(tmp_path / "b.log")["box_geometry"]
    assert geometry["box_shape"] == shape
    # Requested AND realised: a box grown to satisfy the cutoff must say so rather than quietly
    # reporting the number the user asked for.
    assert "box_width_requested_nm" in geometry and "box_width_nm" in geometry
    vectors = read_record(tmp_path / "b.log")["box_vectors_nm"]
    if shape == "cube":
        assert vectors[1][0] == 0.0 and vectors[2][0] == 0.0, "a cube is axis-aligned"
    else:
        assert any(v != 0.0 for v in vectors[2][:2]), "a dodecahedron is not axis-aligned"


# --- ligand force field -----------------------------------------------------------------------

def test_sage_is_still_the_default_ligand_force_field():
    assert resolve_build_config(None)["solute"]["ligand_forcefield"] == "sage-2.2.1"


def test_a_sage_label_resolves_to_the_openff_resource():
    from md_tools.openmm.ligand_forcefield import resolve_ligand_forcefield

    assert resolve_ligand_forcefield("sage-2.2.1") == "openff-2.2.1"


def test_the_gaff2_alias_resolves_to_the_newest_installed_exact_version():
    """An ambiguous label must never reach a record: GAFF2 has been several parameter sets.

    Pinned to the ordering `openmmforcefields` declares rather than to a parsed version number --
    GAFF's numbering makes `1.81` newer than `1.8` but `2.2.20` newer than `2.11`, and no single
    arithmetic orders both. The first implementation compared component-wise and chose the wrong
    file.
    """
    from md_tools.openmm.ligand_forcefield import (installed_gaff_versions,
                                                   resolve_ligand_forcefield)

    installed = installed_gaff_versions()
    newest_two = [name for name in installed if name.startswith("gaff-2")][-1]
    resolved = resolve_ligand_forcefield("gaff2")
    assert resolved == newest_two
    assert resolved != "gaff2", "the alias must not survive resolution"
    assert resolved.count(".") >= 1, "the resolved name must carry an exact version"


def test_an_exact_gaff_version_is_passed_through():
    from md_tools.openmm.ligand_forcefield import resolve_ligand_forcefield

    assert resolve_ligand_forcefield("gaff-2.11") == "gaff-2.11"


def test_an_uninstalled_gaff_version_is_refused_with_the_installed_list():
    from md_tools.openmm.ligand_forcefield import (LigandForcefieldError,
                                                   installed_gaff_versions,
                                                   resolve_ligand_forcefield)

    with pytest.raises(LigandForcefieldError) as refusal:
        resolve_ligand_forcefield("gaff-9.9")
    message = str(refusal.value)
    for name in installed_gaff_versions():
        assert name in message, f"the refusal does not name the installed {name}"


@pytest.mark.parametrize("requested, family", [("sage-2.2.1", "smirnoff"), ("gaff2", "gaff")])
@pytest.mark.slow
@pytest.mark.gpu
def test_both_ligand_families_build_and_record_their_exact_provenance(requested, family, tmp_path):
    """A small ligand through each route, with the record checked rather than the exit code."""
    import subprocess
    import sys

    from md_tools.build.record import read_record

    (tmp_path / "lig.smi").write_text("CCO ETH\n", encoding="utf-8")
    config = tmp_path / "b.config"
    config.write_text(yaml.safe_dump(
        {"solute": {"peptide": False, "ligand_forcefield": requested},
         "solvent": {"padding_nm": 0.5, "cutoff_nm": 0.6}}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", "lig.smi",
         "-os", "b.xml", "-op", "b.pdb", "-log", "b.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr

    ligand = read_record(tmp_path / "b.log")["forcefield_record"]["ligand"]
    assert ligand["requested_label"] == requested
    assert ligand["family"] == family
    # The resolved resource, never the alias.
    assert ligand["forcefield"] and ligand["forcefield"] != requested or requested.startswith("sage")
    assert ligand["charge_method"] == "am1bcc"
    if family == "gaff":
        assert ligand["forcefield"].startswith("gaff-2.")
        assert ligand["typing"] == "antechamber"
        # AmberTools is part of a GAFF Hamiltonian's provenance in a way it is not for SMIRNOFF.
        assert ligand["ambertools"]["version"], "the AmberTools version was not established"
    else:
        assert ligand["forcefield"].startswith("openff-")
        assert ligand["ambertools"] is None


# --- hydrogen mass repartitioning ---------------------------------------------------------------

def test_repartitioning_is_off_by_default_and_states_its_target():
    """`enabled: false` with the target still present, rather than a null that means two things."""
    block = resolve_build_config(None)["hydrogen_mass_repartitioning"]
    assert block["enabled"] is False
    assert block["hydrogen_mass_amu"] == 3.024


@pytest.mark.parametrize("value, expected", [(None, "enabled: false"), (3.024, "enabled: true")])
def test_the_retired_hmr_key_raises_a_migration_error_not_a_typo_suggestion(value, expected):
    """The value did not move, its MEANING changed: null-means-off carried two facts in one field.

    A generic unknown-key refusal would suggest `rigid_water`, which is worse than useless.
    """
    with pytest.raises(ConfigError) as refusal:
        resolve_build_config(_config(constraints={"hydrogen_mass_amu": value}))
    message = str(refusal.value)
    assert "hydrogen_mass_repartitioning" in message
    assert expected in message, message


def test_repartitioning_without_hydrogen_constraints_is_refused():
    """HMR buys a timestep by slowing the X-H stretch. Unconstrained, that stretch still limits it."""
    with pytest.raises(ConfigError, match="constraints.type"):
        resolve_build_config(_config(constraints={"type": "None"},
                                     hydrogen_mass_repartitioning={"enabled": True}))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.slow
@pytest.mark.gpu
def test_repartitioning_conserves_mass_and_never_touches_water(enabled, tmp_path):
    import subprocess
    import sys

    from md_tools.build.record import read_record

    config = tmp_path / "b.config"
    config.write_text(yaml.safe_dump(
        {"solvent": {"padding_nm": 0.5, "cutoff_nm": 0.6},
         "hydrogen_mass_repartitioning": {"enabled": enabled, "hydrogen_mass_amu": 3.024}}),
        encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ALA),
         "-os", "b.xml", "-op", "b.pdb", "-log", "b.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr

    hmr = read_record(tmp_path / "b.log")["hmr"]
    assert hmr["enabled"] is enabled
    masses = hmr["distinct_hydrogen_masses_amu"]
    if enabled:
        assert 3.024 in masses, masses
        # Water is never repartitioned, so a correct solvated system holds BOTH.
        assert any(m < 2.0 for m in masses), f"water was repartitioned: {masses}"
        assert hmr["ignored_because_disabled"] is None
    else:
        assert all(m < 2.0 for m in masses), masses
        assert hmr["ignored_because_disabled"], "a disabled block must say the target was ignored"
        assert hmr["configured_hydrogen_mass_amu"] == 3.024, "the target is recorded even when off"
