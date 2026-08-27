"""Configuration files are generated correctly, and from ONE set of defaults."""
from __future__ import annotations

import pytest

import yaml

from md_templates.openmm import defaults as D


def test_the_explicit_default_is_ff14sb_sage_221_tip3p(md_openmm, tmp_path):
    """No `--solvent`: what a user gets by typing the least.

    The default explicit combination is ff14SB + Sage 2.2.1 + TIP3P, in a dodecahedral box with
    1.5 nm of requested padding, PME at a 1.0 nm cutoff, 0.15 M NaCl, 2 fs and no HMR.
    """
    result = md_openmm("sys-config", "--method", "cMD", "REST2", "--peptide", "true")
    assert result.returncode == 0, result.stderr

    sys_doc = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    md_doc = yaml.safe_load((tmp_path / "md.config.yaml").read_text())

    assert sys_doc["solvent"]["model"] == "TIP3P"
    assert sys_doc["forcefield"]["protein"] == "amber14-all.xml"
    assert sys_doc["forcefield"]["water"] == "amber14/tip3p.xml"
    assert sys_doc["solute"]["ligand_forcefield"] == "sage-2.2.1"
    assert sys_doc["solvent"]["padding_nm"] == 1.5
    assert sys_doc["solvent"]["box_shape"] == "dodecahedron"
    assert sys_doc["solvent"]["cutoff_nm"] == 1.0
    assert sys_doc["constraints"]["hydrogen_mass_amu"] is None, "HMR is off by default"
    assert md_doc["common"]["timestep_fs"] == 2.0
    assert md_doc["common"]["friction_per_ps"] == 1.0
    assert md_doc["common"]["barostat_frequency_steps"] == 25
    # no ff19SB or OPC in any VALUE. The header comment names them as the alternative, which is
    # the point; what must not happen is one of them being what actually gets built.
    values = yaml.safe_dump(sys_doc).lower()
    assert "ff19sb" not in values and "opc" not in values and "amber19" not in values


def test_explicit_opc_configuration(md_openmm, tmp_path):
    """`--solvent OPC` still selects ff19SB + OPC, unchanged as the alternative."""
    result = md_openmm("sys-config", "--method", "cMD", "REST2",
                       "--peptide", "true", "--solvent", "OPC")
    assert result.returncode == 0, result.stderr

    sys_doc = yaml.safe_load((tmp_path / "sys.config.yaml").read_text())
    md_doc = yaml.safe_load((tmp_path / "md.config.yaml").read_text())

    assert sys_doc["solute"]["peptide"] is True
    assert sys_doc["solvent"]["model"] == "OPC"
    assert sys_doc["forcefield"]["protein"] == "amber19-all.xml"
    assert sys_doc["forcefield"]["water"] == "amber19/opc.xml"
    assert sys_doc["solute"]["ligand_forcefield"] == "sage-2.2.1"
    assert sys_doc["solvent"]["padding_nm"] == 1.5
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
    assert md_doc["common"]["barostat_frequency_steps"] == 25
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
    md_openmm("sys-config", "--method", "cMD", "REST2")
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
    """The pairing ff19SB WAS parameterised for remains selectable, and unchanged."""
    from md_templates.openmm.defaults import sys_defaults

    explicit = sys_defaults(solvent="OPC")
    assert explicit["forcefield"]["protein"] == "amber19-all.xml"
    assert explicit["forcefield"]["water"] == "amber19/opc.xml"
    assert explicit["solvent"]["model"] == "OPC"


def test_the_implicit_file_documents_the_default_explicit_combination():
    """An implicit config still shows what the explicit block would look like -- the DEFAULT one.

    Showing the OPC alternative there would advertise it as the thing to switch back to.
    """
    from md_templates.openmm.defaults import DEFAULT_SOLVENT, sys_defaults

    assert DEFAULT_SOLVENT == "TIP3P"
    document = sys_defaults(solvent="GBn2")
    # the explicit block is dropped for an implicit file, but the default it would have carried is
    # the TIP3P one -- checked through the resolver, which is what decides what gets built
    explicit = sys_defaults(solvent=DEFAULT_SOLVENT)
    assert explicit["forcefield"]["water"] == "amber14/tip3p.xml"
    assert document["forcefield"]["protein"] == "leaprc.protein.ff14SB"


# --- the exact resources must actually load ----------------------------------

def test_every_default_forcefield_resource_loads_in_this_environment():
    """A resource whose STRING looks right and whose FILE is missing is the failure to catch.

    No System is built here -- `ForceField()` only parses the XML -- so this is a non-GPU test.
    """
    from openmm.app import ForceField

    from md_templates.openmm.defaults import EXPLICIT_COMBINATIONS

    for name, combination in EXPLICIT_COMBINATIONS.items():
        forcefield = ForceField(combination["protein"], combination["water"])
        assert forcefield is not None, name
        # the water resource must carry the ion templates addSolvent places
        templates = {t.name.upper() for t in forcefield._templates.values()}
        assert {"NA", "CL", "HOH"} <= templates, f"{name}: {combination['water']} lacks ions"


def test_the_default_ligand_forcefield_resource_loads():
    """`sage-2.2.1` must map onto a SMIRNOFF file this environment actually ships."""
    from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField

    from md_templates.openmm.defaults import LIGAND_FORCEFIELD
    from md_templates.openmm.sysgen import _openff_name

    resource = _openff_name(LIGAND_FORCEFIELD)
    assert resource == "openff-2.2.1"
    assert OFFForceField(f"{resource}.offxml") is not None


def test_an_unqualified_water_label_that_openmm_does_not_ship_is_refused():
    """The old blanket `amber19/` prefix turned `tip3p.xml` into a file that does not exist."""
    from md_templates.openmm.config import ConfigError
    from md_templates.openmm.sysgen import _water_xml

    assert _water_xml("tip3p.xml") == "amber14/tip3p.xml"
    assert _water_xml("opc.xml") == "amber19/opc.xml"
    assert _water_xml("amber14/tip3p.xml") == "amber14/tip3p.xml"
    with pytest.raises(ConfigError) as error:
        _water_xml("water.xml")
    assert "amber14/tip3p.xml" in str(error.value)


# --- the barostat attempt frequency is public, and reaches the stages ---------

def test_the_barostat_frequency_is_declared_once_and_defaults_to_openmms_own():
    from md_templates.openmm.defaults import DEFAULT_BAROSTAT_FREQUENCY_STEPS, md_defaults

    assert DEFAULT_BAROSTAT_FREQUENCY_STEPS == 25
    assert md_defaults()["common"]["barostat_frequency_steps"] == 25
    assert md_defaults(solvent="GBn2")["common"]["barostat_frequency_steps"] is None


def test_the_barostat_frequency_reaches_every_npt_stage_and_no_nvt_one():
    """`barostat_active` decides whether it moves; this decides how often when it does."""
    from md_templates.openmm.config import resolve_md_config
    from md_templates.openmm.defaults import md_defaults
    from md_templates.openmm.stages import stage_plan

    resolved = resolve_md_config(md_defaults(), implicit=False)
    resolved["common"]["barostat_frequency_steps"] = 40
    plan = stage_plan(resolved, implicit=False)
    by_kind = {stage["kind"]: stage for stage in plan}
    assert by_kind["npt_restrained"]["barostat_frequency_steps"] == 40
    assert by_kind["npt_free"]["barostat_frequency_steps"] == 40
    # present in the System, inert: frequency 0 is what makes an NVT stage NVT here
    assert by_kind["minimization"]["barostat_frequency_steps"] == 0
    assert by_kind["nvt_restrained"]["barostat_frequency_steps"] == 0

    implicit_plan = stage_plan(
        resolve_md_config(md_defaults(solvent="GBn2"), implicit=True), implicit=True)
    assert all(stage["barostat_frequency_steps"] is None for stage in implicit_plan), \
        "implicit solvent has no barostat at all; 0 would claim there is an inert one"


@pytest.mark.parametrize("value", [0, -1, 2.5, "25", None])
def test_a_barostat_frequency_that_is_not_a_positive_whole_step_count_is_refused(value):
    from md_templates.openmm.config import ConfigError, resolve_md_config
    from md_templates.openmm.defaults import md_defaults

    document = md_defaults()
    document["common"]["barostat_frequency_steps"] = value
    with pytest.raises(ConfigError) as error:
        resolve_md_config(document, implicit=False)
    assert "barostat_frequency_steps" in str(error.value)


# --- 2 fs is the baseline; 4 fs needs HMR AND the constraints -----------------

def test_the_default_protocol_is_2fs_without_hmr():
    from md_templates.openmm.defaults import md_defaults, sys_defaults

    assert md_defaults()["common"]["timestep_fs"] == 2.0
    assert sys_defaults()["constraints"]["hydrogen_mass_amu"] is None
    assert sys_defaults()["constraints"]["type"] == "HBonds"
    assert sys_defaults()["constraints"]["rigid_water"] is True


def test_the_hmr_example_is_a_complete_and_loadable_pair():
    """The documented 4 fs option must resolve, not merely read well."""
    from pathlib import Path

    from md_templates.openmm.config import check_timestep_against_masses, resolve_md_config, \
        resolve_sys_config
    from md_templates.openmm.defaults import HMR_HYDROGEN_MASS_AMU, HMR_TIMESTEP_FS, \
        md_defaults, sys_defaults

    example = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs/examples/hmr-4fs.yaml").read_text())
    assert example["sys.config.yaml"]["constraints"]["hydrogen_mass_amu"] == HMR_HYDROGEN_MASS_AMU
    assert example["md.config.yaml"]["common"]["timestep_fs"] == HMR_TIMESTEP_FS

    system = sys_defaults()
    system["constraints"].update(example["sys.config.yaml"]["constraints"])
    protocol = md_defaults()
    protocol["common"].update(example["md.config.yaml"]["common"])
    check_timestep_against_masses(
        resolve_md_config(protocol, implicit=False), resolve_sys_config(system))


@pytest.mark.parametrize("patch, expected", [
    ({}, "hydrogen_mass_amu"),                                  # 4 fs with real hydrogen masses
    ({"hydrogen_mass_amu": 3.024, "type": "None"}, "not constrained"),
    ({"hydrogen_mass_amu": 3.024, "rigid_water": False}, "rigid_water"),
])
def test_unsafe_4fs_configurations_are_refused_with_a_specific_message(patch, expected):
    from md_templates.openmm.config import ConfigError, check_timestep_against_masses, \
        resolve_md_config
    from md_templates.openmm.defaults import md_defaults, sys_defaults

    system = sys_defaults()
    system["constraints"].update(patch)
    system["solvation"] = "explicit"
    protocol = md_defaults()
    protocol["common"]["timestep_fs"] = 4.0
    with pytest.raises(ConfigError) as error:
        check_timestep_against_masses(resolve_md_config(protocol, implicit=False), system)
    assert expected in str(error.value)


# --- a hand-edited explicit configuration cannot cross the two supported pairs ---------------

@pytest.mark.parametrize("solvent, field, value", [
    ("TIP3P", "protein", "amber19-all.xml"),
    ("TIP3P", "protein", "amber19/protein.ff19SB.xml"),
    ("TIP3P", "water", "amber19/opc.xml"),
    ("OPC", "protein", "amber14-all.xml"),
    ("OPC", "protein", "amber14/protein.ff14SB.xml"),
    ("OPC", "water", "amber14/tip3p.xml"),
])
def test_a_crossed_explicit_pair_is_refused_however_it_was_spelled(solvent, field, value):
    """`sys-config` writes a coupled selection; the file it writes is editable YAML.

    Generating the pair correctly is not the same as building it correctly. Changing one half by
    hand produces a System, runs to completion, and reports a Hamiltonian nobody validated -- so
    the crossing is refused before a System exists, and refused on the FAMILY of the resource name
    rather than on an exact string, because there is more than one way to spell each force field.
    """
    from md_templates.openmm.config import ConfigError, resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults

    document = sys_defaults(solvent=solvent)
    document["forcefield"][field] = value
    with pytest.raises(ConfigError) as error:
        resolve_sys_config(document)

    message = str(error.value)
    assert f"forcefield.{field}" in message, "the mismatched field must be named"
    assert value in message and solvent in message
    # ...and the message must say what the supported pair actually is.
    assert "amber14-all.xml" in message and "amber19-all.xml" in message
    assert "ONE selection" in message


@pytest.mark.parametrize("solvent", ["TIP3P", "OPC", "GBn2"])
def test_every_configuration_sys_config_writes_still_resolves(solvent):
    """The validation must refuse crossings without refusing the pairs the tool itself writes."""
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import EXPLICIT_COMBINATIONS, sys_defaults

    resolved = resolve_sys_config(sys_defaults(solvent=solvent))
    if solvent == "GBn2":
        assert resolved["solvation"] == "implicit"
        assert resolved["forcefield"]["protein"] == "leaprc.protein.ff14SB"
        assert resolved["forcefield"]["water"] is None
    else:
        expected = EXPLICIT_COMBINATIONS[solvent]
        assert resolved["forcefield"]["protein"] == expected["protein"]
        assert resolved["forcefield"]["water"] == expected["water"]


def test_a_force_field_this_repository_does_not_ship_is_left_alone():
    """The pairing table has no opinion about a resource outside both supported families.

    Refusing it would be refusing a deliberate choice this table cannot judge; the check exists to
    catch a CROSSING between the two pairs it does know, not to police the field.
    """
    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults

    document = sys_defaults(solvent="TIP3P")
    document["forcefield"]["protein"] = "charmm36_2024.xml"
    resolved = resolve_sys_config(document)
    assert resolved["forcefield"]["protein"] == "charmm36_2024.xml"


def test_the_ligand_only_route_still_records_no_protein_force_field():
    """Validation asks the FILE to name a complete selection; the RECORD still says what loaded.

    A ligand-only build loads no protein XML, and naming one in `forcefield.json` would attribute
    parameters to a file that contributed none. Both facts hold at once.
    """
    from pathlib import Path

    from md_templates.openmm.config import resolve_sys_config
    from md_templates.openmm.defaults import sys_defaults
    from md_templates.openmm.forcefield_record import build_forcefield_record

    resolved = resolve_sys_config(sys_defaults(solvent="TIP3P", peptide=False))
    record = build_forcefield_record(
        resolved=resolved, route="ligand",
        record={"forcefield": {"xml": ["amber14/tip3p.xml"], "route": "ligand",
                               "protein_forcefield": None, "water": "amber14/tip3p.xml",
                               "nonbonded": {"method": "PME", "cutoff_nm": 1.0,
                                             "switching": False, "switch_distance_nm": None,
                                             "dispersion_correction": True,
                                             "ewald_error_tolerance": 0.0005},
                               "ligand": {"forcefield": "openff-2.2.1",
                                          "charge_method": "am1bcc"}}},
        inputs_dir=Path("."), artifacts={})
    assert record["protein"]["openmm_resource"] is None
    assert "loads no protein force field" in record["protein"]["note"]


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
