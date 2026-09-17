"""Configuration files are generated correctly, and from ONE set of defaults."""
from __future__ import annotations

import pytest

import yaml

from md_tools.build.md import resolve_md_config
from md_tools.build.strict import ConfigError
from md_tools.build.top import _sys_document, resolve_build_config
from md_tools.openmm import system_defaults as D
from md_tools.openmm.system_defaults import canonical_solvent


def _written(document):
    """Write a configuration to a temporary file, since the resolvers take a path."""
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".config", delete=False)
    yaml.safe_dump(document, handle)
    handle.close()
    return handle.name


















# --- protein force field must match the solvation model ----------------------

def test_implicit_gbn2_defaults_to_ff14sb_not_ff19sb():
    """GBn2 was developed and validated against the ff99SB/ff14SB lineage.

    ff19SB's amino-acid-specific CMAPs were fit in explicit OPC water and no GB model has been
    reparameterised against them, so ff19SB + GBn2 mixes a backbone trained in explicit solvent
    with a solvation model tuned for a different one.
    """
    from md_tools.openmm.system_defaults import sys_defaults

    implicit = sys_defaults(solvent="GBn2")
    assert implicit["forcefield"]["protein"] == "leaprc.protein.ff14SB"
    assert implicit["forcefield"]["water"] is None, "implicit solvent has no water model"
    assert "ff19SB" not in implicit["forcefield"]["protein"]


def test_explicit_opc_still_uses_ff19sb():
    """The pairing ff19SB WAS parameterised for remains selectable, and unchanged."""
    from md_tools.openmm.system_defaults import sys_defaults

    explicit = sys_defaults(solvent="OPC")
    assert explicit["forcefield"]["protein"] == "amber19-all.xml"
    assert explicit["forcefield"]["water"] == "amber19/opc.xml"
    assert explicit["solvent"]["model"] == "OPC"


def test_the_implicit_file_documents_the_default_explicit_combination():
    """An implicit config still shows what the explicit block would look like -- the DEFAULT one.

    Showing the OPC alternative there would advertise it as the thing to switch back to.
    """
    from md_tools.openmm.system_defaults import DEFAULT_SOLVENT, sys_defaults

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

    from md_tools.openmm.system_defaults import EXPLICIT_COMBINATIONS

    for name, combination in EXPLICIT_COMBINATIONS.items():
        forcefield = ForceField(combination["protein"], combination["water"])
        assert forcefield is not None, name
        # the water resource must carry the ion templates addSolvent places
        templates = {t.name.upper() for t in forcefield._templates.values()}
        assert {"NA", "CL", "HOH"} <= templates, f"{name}: {combination['water']} lacks ions"


def test_the_default_ligand_forcefield_resource_loads():
    """`sage-2.2.1` must map onto a SMIRNOFF file this environment actually ships."""
    from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField

    from md_tools.openmm.system_defaults import LIGAND_FORCEFIELD
    from md_tools.openmm.system_config import openff_resource

    resource = openff_resource(LIGAND_FORCEFIELD)
    assert resource == "openff-2.2.1"
    assert OFFForceField(f"{resource}.offxml") is not None


def test_an_unqualified_water_label_that_openmm_does_not_ship_is_refused():
    """The old blanket `amber19/` prefix turned `tip3p.xml` into a file that does not exist."""
    from md_tools.build.strict import ConfigError
    from md_tools.openmm.builders import _water_xml

    assert _water_xml("tip3p.xml") == "amber14/tip3p.xml"
    assert _water_xml("opc.xml") == "amber19/opc.xml"
    assert _water_xml("amber14/tip3p.xml") == "amber14/tip3p.xml"
    with pytest.raises(ConfigError) as error:
        _water_xml("water.xml")
    assert "amber14/tip3p.xml" in str(error.value)


# --- the barostat attempt frequency is public, and reaches the stages ---------

def test_the_barostat_frequency_is_declared_once_and_defaults_to_openmms_own():
    """One declaration, reaching the model that actually resolves a user's file.

    It used to be asserted against `md_defaults()`, which described the retired `methods:` model.
    `build.md` is the only MD resolver now, and it imports the constant rather than repeating 25 --
    which is the property worth holding: a number declared twice is a number that drifts.
    """
    from md_tools.openmm.system_defaults import DEFAULT_BAROSTAT_FREQUENCY_STEPS

    assert DEFAULT_BAROSTAT_FREQUENCY_STEPS == 25
    resolved = resolve_md_config(_written({"protocol": "cMD"}))
    assert resolved["dynamics"]["barostat_interval_steps"] == DEFAULT_BAROSTAT_FREQUENCY_STEPS


def test_the_barostat_interval_reaches_every_npt_stage(tmp_path):
    """`ensemble` decides whether the barostat moves; this decides how often when it does.

    The retired plan carried a per-stage `barostat_frequency_steps` that was set to 0 to mean
    "inert". The live plan states the `ensemble` instead, which is a fact about the stage rather
    than a number whose zero has to be interpreted -- and what a stage ACTUALLY runs is asserted
    on CUDA by `test_the_configured_barostat_frequency_is_what_the_stages_actually_run`.
    """
    from md_tools.build.md import stage_plan

    resolved = resolve_md_config(_written(
        {"protocol": "cMD", "dynamics": {"barostat_interval_steps": 40}}))
    plan = stage_plan(resolved)
    npt = [stage for stage in plan if stage["ensemble"] == "NPT"]
    assert npt, "an explicit cMD chain must contain NPT stages"
    assert all(stage["barostat_interval_steps"] == 40 for stage in npt)

    implicit = stage_plan(resolve_md_config(_written(
        {"protocol": "cMD", "solvent": "implicit"})))
    assert all(stage["ensemble"] == "NVT" for stage in implicit), \
        "implicit solvent has no volume, so no stage may claim NPT"


@pytest.mark.parametrize("value", [0, -1, 2.5, "25", None])
def test_a_barostat_interval_that_is_not_a_positive_whole_step_count_is_refused(value):
    with pytest.raises(ConfigError) as error:
        resolve_md_config(_written(
            {"protocol": "cMD", "dynamics": {"barostat_interval_steps": value}}))
    assert "barostat_interval_steps" in str(error.value)


# --- 2 fs is the baseline; 4 fs needs HMR AND the constraints -----------------

def test_the_default_protocol_defers_the_timestep_and_leaves_hmr_off():
    """The default is `auto`, not 2.0. The 2 fs still arrives -- from the System, at run time.

    `build-md` cannot know whether the System it will be pointed at was repartitioned, so the
    default stopped being a number and became the instruction to look. The number this resolves
    to on ordinary hydrogens is asserted against a real System in `test_timestep_resolution.py`.
    """
    assert resolve_md_config(_written({"protocol": "cMD"}))["dynamics"]["timestep_fs"] == "auto"
    assert D.sys_defaults()["constraints"]["type"] == "HBonds"
    assert D.sys_defaults()["constraints"]["rigid_water"] is True


def test_the_shipped_example_documents_the_hmr_pair_the_model_declares():
    """A documented value the code does not apply is the failure this guards against.

    The numbers used to live in `docs/examples/hmr-4fs.yaml`; that file is gone and the shipped
    `build-top` example documents them. Only the DOCUMENTATION half is checked here. Whether a 4 fs
    timestep is allowed is no longer a configuration question at all: it is decided against the
    masses serialised in the built System, by `runtime.stage.check_timestep_against_masses`, and
    asserted by `test_a_large_timestep_without_hmr_is_refused_before_integrating`.
    """
    from md_tools.configs import example as shipped_example

    documented = shipped_example("sys/build-top.config").read_text(encoding="utf-8")
    assert str(D.HMR_HYDROGEN_MASS_AMU) in documented, (
        f"the shipped example no longer documents the {D.HMR_HYDROGEN_MASS_AMU} amu target")
    assert f"{D.HMR_TIMESTEP_FS:g} fs" in documented or str(D.HMR_TIMESTEP_FS) in documented


def test_a_hydrogen_mass_lighter_than_hydrogen_is_refused():
    """Repartitioning moves mass INTO hydrogens; a lighter one is a typo with a plausible shape.

    This is the part of the retired 4 fs check that is still a CONFIGURATION question. The rest --
    that the System was actually built with repartitioned masses -- cannot be answered from a
    configuration file and is checked at run time instead.
    """
    from md_tools.openmm.system_config import resolve_sys_config

    system = D.sys_defaults()
    system["constraints"]["hydrogen_mass_amu"] = 0.5   # the retired scalar, still checked below
    with pytest.raises(ConfigError, match="lighter than a hydrogen"):
        resolve_sys_config(system)


# --- a hand-edited explicit configuration may cross the two supported pairs, loudly ----------

@pytest.mark.parametrize("solvent, field, value", [
    ("TIP3P", "protein", "amber19-all.xml"),
    ("TIP3P", "protein", "amber19/protein.ff19SB.xml"),
    ("TIP3P", "water", "amber19/opc.xml"),
    ("OPC", "protein", "amber14-all.xml"),
    ("OPC", "protein", "amber14/protein.ff14SB.xml"),
    ("OPC", "water", "amber14/tip3p.xml"),
])
def test_a_crossed_explicit_pair_warns_however_it_was_spelled(solvent, field, value):
    """It used to be refused. It is now a warning, and the warning is the deliverable.

    Refusing made MD-tools the arbiter of somebody else's experiment: reproducing a published
    ff14SB/OPC setup, or measuring the water-model sensitivity this pairing exposes, are things a
    competent user may deliberately want. What the tool owes them is that the choice can never be
    made silently or by accident.

    So this asserts the warning is produced, names the field, and says what the supported pairs
    are -- and that it is detected on the FAMILY of the resource name rather than an exact string,
    because there is more than one way to spell each force field.
    """
    from md_tools.openmm.system_config import pairing_warnings, resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    document = sys_defaults(solvent=solvent)
    document["forcefield"][field] = value
    resolved = resolve_sys_config(document)          # no longer raises

    warnings = pairing_warnings(resolved)
    assert len(warnings) == 1, f"expected exactly one warning, got {warnings}"
    warning = warnings[0]
    assert warning["code"] == "crossed_explicit_pair"
    assert warning["severity"] == "warning"
    assert any(entry["field"] == f"forcefield.{field}" for entry in warning["fields"]), warning
    message = warning["message"]
    assert value in message and solvent in message
    assert "amber14-all.xml" in message and "amber19-all.xml" in message
    assert "coupled selection" in message


@pytest.mark.parametrize("solvent, protein", [("TIP3P", "amber14-all.xml"),
                                              ("OPC", "amber19-all.xml")])
def test_a_supported_pair_warns_about_nothing(solvent, protein):
    """The warning must discriminate. One that fires on the default pair is noise."""
    from md_tools.openmm.system_config import pairing_warnings, resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    document = sys_defaults(solvent=solvent)
    document["forcefield"]["protein"] = protein
    assert pairing_warnings(resolve_sys_config(document)) == []


@pytest.mark.parametrize("ligand", ["sage-2.2.1", "gaff2"])
def test_the_ligand_force_field_never_triggers_a_pairing_warning(ligand):
    """The warning is about the PROTEIN/WATER pairing. Sage and GAFF are orthogonal to it."""
    from md_tools.openmm.system_config import pairing_warnings, resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    document = sys_defaults(peptide=False, solvent="TIP3P")
    document["solute"]["ligand_forcefield"] = ligand
    assert pairing_warnings(resolve_sys_config(document)) == []


def test_incoherent_physics_is_still_a_hard_error_not_a_warning():
    """Softening the pairing policy must not soften the combinations that cannot be built.

    ff19SB has no GBn2 parameterisation: that is not an unvalidated choice a user might defend,
    it is a System that does not mean anything. It stays a refusal.
    """
    from md_tools.build.strict import ConfigError
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    document = sys_defaults(solvent="GBn2")
    document["forcefield"]["protein"] = "ff19SB"
    with pytest.raises(ConfigError, match="not parameterised for"):
        resolve_sys_config(document)


@pytest.mark.parametrize("solvent", ["TIP3P", "OPC", "GBn2"])
def test_every_configuration_sys_config_writes_still_resolves(solvent):
    """The validation must refuse crossings without refusing the pairs the tool itself writes."""
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import EXPLICIT_COMBINATIONS, sys_defaults

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
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

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

    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults
    from md_tools.openmm.forcefield_record import build_forcefield_record

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
    from md_tools.build.strict import ConfigError
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    document = sys_defaults(solvent="GBn2")
    document["forcefield"]["protein"] = "amber19-all.xml"
    with pytest.raises(ConfigError) as error:
        resolve_sys_config(document)

    message = str(error.value)
    assert "forcefield.protein" in message and "implicit_solvent.model" in message
    assert "amber19-all.xml" in message and "GBn2" in message
    assert "leaprc.protein.ff14SB" in message, "the message must name the matched pair"


def test_the_matched_implicit_pair_resolves():
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults

    resolved = resolve_sys_config(sys_defaults(solvent="GBn2"))
    assert resolved["forcefield"]["protein"] == "leaprc.protein.ff14SB"
    assert resolved["implicit_solvent"]["model"] == "GBn2"
    assert resolved["implicit_solvent"]["radii"] == "mbondi3"


def test_the_configured_protein_force_field_reaches_tleap():
    """It was ignored: every implicit peptide ran ff19SB whatever the configuration said."""
    import inspect

    from md_tools.openmm import implicit

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
    from md_tools.openmm.system_defaults import sys_defaults

    implicit = sys_defaults(solvent="GBn2")["implicit_solvent"]
    assert implicit["nonpolar_sasa"] is False
    assert implicit["model"] == "GBn2" and implicit["radii"] == "mbondi3"


def test_the_choice_reaches_the_builder_and_is_not_a_library_default():
    import inspect

    from md_tools.openmm import builders, implicit

    assert inspect.signature(implicit.build_implicit_system).parameters[
        "nonpolar_sasa"].default is False
    assert "useSASA=bool(nonpolar_sasa)" in inspect.getsource(implicit.build_implicit_system), \
        "createSystem must be told explicitly, not left to ParmEd's default"
    assert "nonpolar_sasa=bool(" in inspect.getsource(builders), \
        "the builder must pass the configured value through, not leave it to a library default"


def test_the_forcefield_record_states_the_nonpolar_choice():
    from md_tools.openmm.system_config import resolve_sys_config
    from md_tools.openmm.system_defaults import sys_defaults
    from md_tools.openmm.forcefield_record import build_forcefield_record
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


# --- the same scientific defaults, through the new configuration surface --------------------------
#
# `sys-config` and `show-default` are retired: there is no longer a command that WRITES a
# configuration for you, because the shipped examples are generated from the schemas that enforce
# them and are therefore always in step. What those tests protected -- the default combination, the
# solvent pairings, and refusal of nonsense -- is asserted here against the new surface.

def test_the_explicit_default_is_ff14sb_sage_221_tip3p():
    """No configuration at all: what a user gets by typing the least.

    ff14SB + Sage 2.2.1 + TIP3P, dodecahedral box, 1.5 nm requested padding, 1.0 nm cutoff,
    0.15 M NaCl, HBonds, and NO hydrogen mass repartitioning.
    """
    resolved = resolve_build_config(None)
    assert resolved["solvent"]["model"] == "TIP3P"
    assert resolved["forcefield"]["protein"] == "ff14SB"
    assert resolved["solute"]["ligand_forcefield"] == "sage-2.2.1"
    assert resolved["solvent"]["padding_nm"] == 1.5
    assert resolved["solvent"]["box_shape"] == "dodecahedron"
    assert resolved["solvent"]["cutoff_nm"] == 1.0
    assert resolved["solvent"]["ionic_strength_molar"] == 0.15
    assert (resolved["solvent"]["positive_ion"], resolved["solvent"]["negative_ion"]) == \
           ("Na+", "Cl-")
    assert resolved["constraints"]["type"] == "HBonds"
    assert resolved["hydrogen_mass_repartitioning"]["enabled"] is False, "HMR is off by default"

    document = _sys_document(resolved)
    assert document["forcefield"]["protein"] == "amber14-all.xml"
    assert document["forcefield"]["water"] == "amber14/tip3p.xml"
    # Neither ff19SB nor OPC may be what actually gets built by default.
    values = yaml.safe_dump(document).lower()
    assert "ff19sb" not in values and "opc" not in values and "amber19" not in values


def test_the_default_protocol_is_five_ns_of_steps_at_two_femtoseconds():
    resolved = resolve_md_config(None)
    assert resolved["dynamics"]["timestep_fs"] == "auto"
    assert resolved["dynamics"]["friction_per_ps"] == 1.0
    assert resolved["dynamics"]["barostat_interval_steps"] == 25
    assert resolved["stages"]["production_steps"] == 2_500_000
    # The STEP COUNT is the default; 5 ns is what it becomes at the 2 fs an unrepartitioned
    # System resolves to. Written as the derivation rather than as a stored duration, because
    # the same step count is 10 ns on a System built with HMR.
    assert resolved["stages"]["production_steps"] * 2.0 / 1e6 == 5.0, "5 ns at 2 fs"
    assert resolved["reporting"] == {"crd_printout_solute": 1000, "crd_printout_whole": 0,
                                    "info_printout": 10000,
                                     "checkpoint_printout": 10000}


def test_opc_selects_ff19sb_and_tip3p_is_warned_about_rather_than_refused():
    """ff19SB was parameterised against OPC. The crossed EXPLICIT pair now builds and warns.

    Migrated, not weakened. Refusing ff19SB + TIP3P made this tool the arbiter of somebody else's
    experiment, and the refusal fired before `pairing_warnings()` could see the combination, so
    the documented warning policy was unreachable for half the pairs it described. The pair that
    cannot be built at all -- ff19SB with GBn2 -- is still a hard error; see
    `test_ff19sb_with_gbn2_remains_a_hard_error`.
    """
    from md_tools.openmm.system_config import pairing_warnings, resolve_sys_config

    opc = resolve_build_config(_written({"solvent": {"model": "OPC"},
                                         "forcefield": {"protein": "ff19SB"}}))
    assert _sys_document(opc)["forcefield"]["protein"] == "amber19-all.xml"
    assert pairing_warnings(resolve_sys_config(_sys_document(opc))) == []

    crossed = resolve_build_config(_written({"solvent": {"model": "TIP3P"},
                                             "forcefield": {"protein": "ff19SB"}}))
    warnings = pairing_warnings(resolve_sys_config(_sys_document(crossed)))
    assert warnings, "the crossed pair must build LOUDLY, not silently"
    assert [warning["code"] for warning in warnings] == ["crossed_explicit_pair"], warnings
    assert "amber19-all.xml" in warnings[0]["message"], warnings[0]["message"]


def test_gbn2_is_implicit_and_carries_no_box_or_water():
    resolved = resolve_build_config(_written({"solvent": {"model": "GBn2"}}))
    document = _sys_document(resolved)
    # No explicit-solvent block exists at all -- not one that is present and ignored. There is
    # nothing for a box, a cutoff or salt to be written into.
    assert "solvent" not in document, document.get("solvent")
    assert document["implicit_solvent"] == {"model": "GBn2", "radii": "mbondi3",
                                            "nonpolar_sasa": False}
    from md_tools.openmm.system_config import resolve_sys_config
    sys_resolved = resolve_sys_config(document)
    assert sys_resolved["solvation"] == "implicit"
    assert sys_resolved["forcefield"]["water"] is None, "no water model participates"
    assert sys_resolved["constraints"]["rigid_water"] is False, "there is no water to hold rigid"


def test_a_solvent_name_is_accepted_case_insensitively():
    for spelling in ("TIP3P", "tip3p", "Tip3p"):
        assert canonical_solvent(spelling) == "TIP3P"
    for spelling in ("GBn2", "gbn2", "GBN2"):
        assert canonical_solvent(spelling) == "GBn2"


@pytest.mark.parametrize("block, key, value", [
    ("solvent", "model", "TIP4P"),
    ("solvent", "box_shape", "sphere"),
    ("solvent", "padding_nm", 99.0),
    ("constraints", "type", "SomeBonds"),
    ("solute", "ligand_charge_method", "made-up"),
    ("forcefield", "protein", "charmm36"),
])
def test_a_value_outside_the_declared_set_is_refused(block, key, value):
    with pytest.raises(ConfigError):
        resolve_build_config(_written({block: {key: value}}))






# --- the shipped examples: canonical files, checked against the models --------------------------
#
# The examples are no longer GENERATED from the schemas. They are ordinary, browsable, hand-editable
# files at the repository root -- which is what makes GitHub show them as a directory rather than a
# symlink blob, and what lets a user read a complete example without installing anything.
#
# The guarantee that replaces "the file is exactly what the renderer emits" is stronger and is what
# actually matters: EVERY example RESOLVES, THROUGH THE REAL RESOLVER, TO THE MODEL'S OWN DEFAULTS.
# An example may therefore be reworded, reordered or better commented freely; it may not drift into
# describing a default that the code does not apply.

def _configs_root():
    from md_tools.configs import example_root
    return example_root()


def test_the_configs_root_has_the_documented_shape():
    root = _configs_root()
    assert (root / "machine").is_dir() and (root / "sys").is_dir() and (root / "md").is_dir()
    assert not root.is_symlink(), "the configs root must be a real directory, not a symlink"


def test_every_shipped_example_is_a_real_file_not_a_link():
    from md_tools.configs import EXAMPLES

    root = _configs_root()
    for relative in EXAMPLES:
        path = root / relative
        assert path.is_file(), f"{relative} is not shipped"
        assert not path.is_symlink(), f"{relative} is a symlink; there must be exactly one copy"


def test_the_build_top_example_resolves_to_the_model_defaults():
    """The example documents the model. If they disagree, the example is lying to the reader."""
    from md_tools.configs import example

    # `_stated` records WHICH keys the file wrote out, which is exactly the difference between a
    # file that states a default and one that omits it. Comparing it would compare the question
    # rather than the answer.
    documented = {k: v for k, v in resolve_build_config(example("sys/build-top.config")).items()
                  if k != "_stated"}
    defaults = {k: v for k, v in resolve_build_config(None).items() if k != "_stated"}
    assert documented == defaults, (
        "configs/sys/build-top.config resolves to something other than the built-in defaults. "
        "Either the example states a value the model does not apply, or a default changed and the "
        "example was not updated.")


@pytest.mark.parametrize("name, protocol", [
    ("cMD.config", "cMD"), ("REST2.config", "REST2"),
    ("AIS.config", "AIS"),
])
def test_each_protocol_example_resolves_and_selects_its_protocol(name, protocol):
    from md_tools.configs import example

    resolved = resolve_md_config(example(f"md/{name}"))
    assert resolved["protocol"] == protocol


@pytest.mark.parametrize("name", ["cMD.config", "REST2.config"])
def test_a_protocol_example_differs_from_the_defaults_only_where_it_says_so(name):
    """Everything an example states must either BE the default or be a documented protocol choice.

    This is what keeps the examples honest without freezing their wording: a key that silently
    disagrees with the model is caught, while comments and ordering stay free.
    """
    from md_tools.configs import example

    resolved = resolve_md_config(example(f"md/{name}"))
    defaults = resolve_md_config(None)
    differing = {key for key in defaults
                 if key not in ("protocol", "rest2") and resolved[key] != defaults[key]}
    assert not differing, (
        f"{name} changes {sorted(differing)} away from the model defaults without being a "
        f"protocol-specific section. Either it is documenting a value the code does not apply, or "
        f"the default moved.")


def test_the_hmr_reasoning_survived_the_move_out_of_docs_examples():
    """docs/examples/hmr-4fs.yaml was deleted; its content is required to be here instead."""
    from md_tools.configs import example

    system = example("sys/build-top.config").read_text(encoding="utf-8")
    assert "hydrogen_mass_amu" in system
    for phrase in ("rigid_water", "HBonds", "3.024", "no longer physical",
                   "configurational averages are unchanged".replace(
                       "configurational", "CONFIGURATIONAL")):
        assert phrase in system, f"the HMR example lost {phrase!r} in the move"
    protocol = example("md/cMD.config").read_text(encoding="utf-8")
    assert "build-top.config" in protocol, (
        "the protocol example must point at where HMR is actually set")


# --- the generated stage files, explicit and implicit ---------------------------------------------

def _generate(tmp_path, config: dict, odir="cMD-run1"):
    """Generate one run and return every file it wrote, keyed by path RELATIVE TO THE DATASET.

    Names alone are no longer enough to identify a generated file: `min.py` is at
    `<system>/min/min.py`, an equilibration script at `<run>/eq/eq_1.py`, and a bare basename
    would make those indistinguishable from each other and from a run-root file.
    """
    import subprocess
    import sys

    from .conftest import make_dataset_root

    # THE SYSTEM MUST MATCH THE DOCUMENT. `build-md` validates the whole chain against the built
    # System when it generates it, so an explicit project needs a periodic one and an implicit
    # project a boxless one. Taken from the document being generated rather than fixed here, so
    # a new case cannot silently generate against the wrong System.
    make_dataset_root(tmp_path, solvent=str(config.get("solvent") or "implicit"))
    path = tmp_path / "p.config"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", odir,
         "--config", str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}


def test_the_explicit_stage_files_are_the_documented_ones(tmp_path):
    """The stage NAMES survive the layout; where they live is what changed.

    A stage is generated as `eq_nvt_posres` and FILED as `eq_1`, so the ensemble that used to be
    in the filename is now in the stage's own `.out` header and resolved configuration. The
    documented set is therefore the same stages at their layout paths.
    """
    written = _generate(tmp_path, {"protocol": "cMD", "solvent": "explicit"})
    assert {"min/min.py", "cMD-run1/eq/eq_1.py", "cMD-run1/eq/eq_2.py", "cMD-run1/eq/eq_3.py",
            "cMD-run1/cMD.py", "cMD-run1/run.sh"} <= written, sorted(written)
    # The inputs are SHARED, at the dataset root rather than inside the run.
    assert {"input/min.in", "input/eq_1.in", "input/eq_2.in", "input/eq_3.in",
            "input/cMD.in"} <= written, sorted(written)


def test_every_declaration_carries_the_definition_it_names(tmp_path):
    """A `resolved.config` that names a definition must sit beside that definition.

    The copy is recorded by BARE NAME so the directory stays movable, and the runtime resolves
    that name beside whichever declaration it read -- `resolved_config_beside(script)` for a
    generated script, `-odir` or the input's directory for `md-run`. The layout puts declarations
    in four places, and the copy was written into the run root ALONE:

        input/cMD.in        names cv.<digest>.yaml   input/ held none
        min/resolved.config names cv.<digest>.yaml   min/ held none
        eq/resolved.config  names cv.<digest>.yaml   eq/ held none

    So `md-openmm md-run -i ../input/cMD.in` could not resolve its own definition from any
    directory. That is not only an inconvenience: `md_tools.run.continuation` has to LOAD the
    definition to know what an invocation intends to continue, so it got None and the read-only
    boundary returned early -- letting a REFUSED continuation write `resolved.config`, `<stage>.out`
    and `<stage>.log` into the tree it was declining to touch. This is the fast guard on that; the
    end-to-end refusal is `test_cv_cost_refusal_end_to_end`.
    """
    import hashlib

    (tmp_path / "cv.yaml").write_text(
        "schema_version: 1\n"
        "collective_variables:\n"
        "  - name: phi\n"
        "    type: torsion\n"
        "    atom_indices: [6, 8, 14, 16]\n", encoding="utf-8")
    written = _generate(tmp_path, {
        "protocol": "cMD", "solvent": "implicit",
        "collective_variables": {"file": "cv.yaml", "interval_steps": 5},
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 10,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 10}})

    digest = hashlib.sha256((tmp_path / "cv.yaml").read_bytes()).hexdigest()
    copy = f"cv.{digest[:12]}.yaml"
    for directory in ("input", "min", "cMD-run1", "cMD-run1/eq"):
        assert f"{directory}/{copy}" in written, (
            f"{directory}/ names the definition but does not carry it: {sorted(written)}")

    # CONTENT-ADDRESSED AND IDENTICAL, which is what makes sharing one safe: a definition that
    # changed would get a different name and could not quietly replace this one.
    payload = (tmp_path / "cv.yaml").read_bytes()
    for directory in ("input", "min", "cMD-run1", "cMD-run1/eq"):
        assert (tmp_path / directory / copy).read_bytes() == payload, directory

    # And every declaration names exactly this file, by bare name.
    for declaration in ("min/resolved.config", "cMD-run1/resolved.config",
                        "cMD-run1/eq/resolved.config"):
        document = yaml.safe_load((tmp_path / declaration).read_text(encoding="utf-8"))
        assert document["collective_variables"]["file"] == copy, declaration


def test_the_implicit_stages_are_renamed_not_silently_run_as_nvt(tmp_path):
    """GBn2 has no box, so there is no NPT stage -- and nothing generated says otherwise.

    THE EVIDENCE MOVED WITH THE LAYOUT. It used to be the filenames: an `eq_npt_free.py` on a
    boxless run would have been a stage whose name was a false claim about the ensemble. Under
    `eq_1/eq_2/eq_3` the name can no longer carry it, so the claim is checked where it now lives
    -- the resolved plan's ensembles, and the absence of any pressure-coupled name anywhere in
    the generated tree, inputs included.
    """
    written = _generate(tmp_path, {"protocol": "cMD", "solvent": "implicit"})
    assert {"min/min.py", "cMD-run1/eq/eq_1.py", "cMD-run1/eq/eq_2.py", "cMD-run1/eq/eq_3.py",
            "cMD-run1/cMD.py", "cMD-run1/run.sh"} <= written, sorted(written)
    assert not any("npt" in name for name in written), (
        f"an implicit run generated a pressure-coupled name: {sorted(written)}")

    from md_tools.build.md import resolve_md_config, stage_plan
    plan = stage_plan(resolve_md_config(None) | {"solvent": "implicit"})
    assert {s["ensemble"] for s in plan} == {"NVT"}, "an implicit stage claims NPT"
