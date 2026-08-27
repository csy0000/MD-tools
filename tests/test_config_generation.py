"""Configuration files are generated correctly, and from ONE set of defaults."""
from __future__ import annotations

import pytest

import yaml

from md_templates.openmm import defaults as D


def test_explicit_opc_configuration(md_openmm, tmp_path):
    result = md_openmm("sys-config", "--method", "cMD", "REST2",
                       "--peptide", "true", "--solvent", "OPC")
    assert result.returncode == 0, result.stderr

    sys_doc = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    md_doc = yaml.safe_load((tmp_path / "md.config.yaml").read_text())

    assert sys_doc["solute"]["peptide"] is True
    assert sys_doc["solvent"]["model"] == "OPC"
    assert sys_doc["solvent"]["padding_nm"] == 2.0
    assert sys_doc["solvent"]["box_shape"] == "dodecahedron"
    assert sys_doc["solvent"]["ionic_strength_molar"] == 0.15
    assert sys_doc["solvent"]["cutoff_nm"] == 1.0
    assert sys_doc["solute"]["ligand_charge_method"] == "am1bcc", "NAGL is never a silent default"
    assert sys_doc["constraints"]["hydrogen_mass_amu"] is None, "HMR is off by default"
    # the implicit block must be absent: one file describes ONE system
    assert "implicit_solvent" not in sys_doc

    assert md_doc["methods"] == ["cMD", "REST2"]
    assert md_doc["common"]["timestep_fs"] == 2.0
    assert md_doc["common"]["pressure_bar"] == 1.0
    assert md_doc["cMD"]["ensemble"] == "NPT"
    assert md_doc["REST2"]["ensemble"] == "NPT"
    assert md_doc["REST2"]["number_of_replicas"] == 6
    assert md_doc["REST2"]["omega_exclusion"] is True


def test_implicit_gbn2_configuration(md_openmm, tmp_path):
    result = md_openmm("sys-config", "--method", "cMD", "REST2",
                       "--peptide", "false", "--solvent", "GBn2")
    assert result.returncode == 0, result.stderr

    sys_doc = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    md_doc = yaml.safe_load((tmp_path / "md.config.yaml").read_text())

    assert sys_doc["solute"]["peptide"] is False
    assert sys_doc["implicit_solvent"]["model"] == "GBn2"
    assert sys_doc["implicit_solvent"]["radii"] == "mbondi3"
    assert "solvent" not in sys_doc, "an implicit system has no water box"
    assert sys_doc["forcefield"]["water"] is None
    assert sys_doc["constraints"]["rigid_water"] is False

    # no box means no barostat and no pressure; saying NPT would name an unsamplable ensemble
    assert md_doc["common"]["pressure_bar"] is None
    assert md_doc["cMD"]["ensemble"] == "NVT"
    assert md_doc["REST2"]["ensemble"] == "NVT"
    assert "pressure_note" in md_doc["common"]


def test_names_are_accepted_case_insensitively_and_written_canonically(md_openmm, tmp_path):
    result = md_openmm("sys-config", "--method", "cmd", "rest2", "--solvent", "opc")
    assert result.returncode == 0, result.stderr
    md_doc = yaml.safe_load((tmp_path / "md.config.yaml").read_text())
    assert md_doc["methods"] == ["cMD", "REST2"]
    sys_doc = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    assert sys_doc["solvent"]["model"] == "OPC"


def test_show_default_and_sys_config_use_the_same_definitions(md_openmm, tmp_path):
    """Two sources of the same defaults is how this repository drifted before."""
    md_openmm("sys-config", "--method", "cMD", "REST2", "--solvent", "OPC")
    written = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    shown = yaml.safe_load(md_openmm("show-default", "sys").stdout)
    assert shown == written

    rest2_shown = yaml.safe_load(md_openmm("show-default", "REST2").stdout)
    written_md = yaml.safe_load((tmp_path / "md.config.yaml").read_text())
    assert rest2_shown["REST2"] == written_md["REST2"]
    assert "cMD" not in rest2_shown


def test_show_default_all_covers_both_documents(md_openmm):
    document = yaml.safe_load(md_openmm("show-default", "all").stdout)
    assert set(document) == {"sys", "md"}


def test_an_unknown_name_is_refused(md_openmm):
    result = md_openmm("show-default", "nonsense")
    assert result.returncode != 0
    assert "expected sys" in (result.stdout + result.stderr)


def test_output_dir_is_honoured(md_openmm, tmp_path):
    target = tmp_path / "elsewhere"
    md_openmm("sys-config", "--output-dir", str(target))
    assert (target / "sys.config.yaml").is_file()
    assert (target / "md.config.yaml").is_file()


# --- protein force field must match the solvation model ----------------------

def test_implicit_gbn2_defaults_to_ff14sb_not_ff19sb():
    """GBn2 was developed and validated against the ff99SB/ff14SB lineage.

    ff19SB's amino-acid-specific CMAPs were fit in explicit OPC water and no GB model has been
    reparameterised against them, so ff19SB + GBn2 mixes a backbone trained in explicit solvent
    with a solvation model tuned for a different one.
    """
    from md_templates.openmm.defaults import sys_defaults

    implicit = sys_defaults(solvent="GBn2")
    assert implicit["forcefield"]["protein"] == "leaprc.protein.ff14SB"
    assert implicit["forcefield"]["water"] is None, "implicit solvent has no water model"
    assert "ff19SB" not in implicit["forcefield"]["protein"]


def test_explicit_opc_still_uses_ff19sb():
    """The pairing ff19SB WAS parameterised for is unchanged."""
    from md_templates.openmm.defaults import sys_defaults

    explicit = sys_defaults(solvent="OPC")
    assert explicit["forcefield"]["protein"] == "amber19-all.xml"
    assert explicit["forcefield"]["water"] == "opc.xml"


def test_ff19sb_with_an_implicit_gb_model_is_refused():
    """It would run and produce numbers, which is exactly why it must not be a warning."""
    from md_templates.openmm.config import ConfigError, resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults

    document = sys_defaults(solvent="GBn2")
    document["forcefield"]["protein"] = "amber19-all.xml"
    with pytest.raises(ConfigError) as error:
        resolve_sys_config(document)

    message = str(error.value)
    assert "forcefield.protein" in message and "implicit_solvent.model" in message
    assert "amber19-all.xml" in message and "GBn2" in message
    assert "leaprc.protein.ff14SB" in message, "the message must name the matched pair"


def test_the_matched_implicit_pair_resolves():
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults

    resolved = resolve_sys_config(sys_defaults(solvent="GBn2"))
    assert resolved["forcefield"]["protein"] == "leaprc.protein.ff14SB"
    assert resolved["implicit_solvent"]["model"] == "GBn2"
    assert resolved["implicit_solvent"]["radii"] == "mbondi3"


def test_the_configured_protein_force_field_reaches_tleap():
    """It was ignored: every implicit peptide ran ff19SB whatever the configuration said."""
    import inspect

    from md_templates.openmm import implicit

    source = inspect.getsource(implicit.build_implicit_bundle_inputs)
    assert "protein_forcefield=protein_ff" in source, \
        "tleap must be given the configured force field, not a hardcoded default"
    signature = inspect.signature(implicit.build_amber_topology_via_tleap)
    assert signature.parameters["protein_forcefield"].default == "leaprc.protein.ff14SB"


# --- the nonpolar (ACE surface-area) term is a stated choice ------------------

def test_the_nonpolar_term_defaults_to_off_matching_amber_igb8_gbsa0():
    """GBn2's parameters were fit to reproduce PB *polar* solvation; the nonpolar term is extra.

    OpenMM's implicit/gbn2.xml turns it on by default and ParmEd leaves it off, so whichever this
    repository picks must be stated rather than inherited.
    """
    from md_templates.openmm.defaults import sys_defaults

    implicit = sys_defaults(solvent="GBn2")["implicit_solvent"]
    assert implicit["nonpolar_sasa"] is False
    assert implicit["model"] == "GBn2" and implicit["radii"] == "mbondi3"


def test_the_choice_reaches_the_builder_and_is_not_a_library_default():
    import inspect

    from md_templates.openmm import implicit, sysgen

    assert inspect.signature(implicit.build_implicit_system).parameters[
        "nonpolar_sasa"].default is False
    assert "useSASA=bool(nonpolar_sasa)" in inspect.getsource(implicit.build_implicit_system), \
        "createSystem must be told explicitly, not left to ParmEd's default"
    assert "nonpolar_sasa=bool(" in inspect.getsource(sysgen), \
        "sys-gen must pass the configured value through"


def test_the_forcefield_record_states_the_nonpolar_choice():
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults
    from md_templates.openmm.forcefield_record import build_forcefield_record
    from pathlib import Path

    resolved = resolve_sys_config(sys_defaults(solvent="GBn2"))
    record = build_forcefield_record(
        resolved=resolved, route="peptide",
        record={"implicit_report": {"implicit_model": "GBn2", "radii": "mbondi3",
                                    "nonpolar_sasa": False, "nonpolar_model": None,
                                    "protein_forcefield": "leaprc.protein.ff14SB"}},
        inputs_dir=Path("."), artifacts={})
    entry = record["implicit_solvent"]
    assert entry["nonpolar_sasa"] is False
    assert entry["nonpolar_model"] is None
    assert "igb=8" in entry["polar_reference"]
