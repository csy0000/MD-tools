"""Umbrella sampling: the restraint definition, and what refuses it.

The property this protocol rests on is that the RESTRAINED quantity and the REPORTED quantity are
the same object. A restraint names a collective variable from `cv.yaml` and never carries its own
atom indices, so a run that biases one torsion while reporting another cannot be expressed.

`cv/definition.py` states the hazard for measurement, and it is worse for a bias: both the biased
and the reported column would be plausible numbers in the right range with the right heading, and
nothing downstream distinguishes them.

PLATFORM_POLICY_EXEMPTION: configuration resolution and definition parsing. Nothing is propagated.
"""
from __future__ import annotations

import textwrap

import pytest

from md_tools.build.md import ConfigError, resolve_md_config
from md_tools.umbrella import RESTRAINT_FORMS, UmbrellaError, load_umbrella_definition

CV_YAML = textwrap.dedent("""
    schema_version: 1
    collective_variables:
      - {name: phi_ALA, type: torsion, atom_indices: [4, 6, 8, 14]}
      - {name: psi_ALA, type: torsion, atom_indices: [6, 8, 14, 16]}
""")


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    return tmp_path


def _definition(workspace):
    from md_tools.cv import load_cv_definition

    return load_cv_definition(workspace / "cv.yaml")


def _umbrella(workspace, body):
    path = workspace / "umbrella.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# --- the property the design exists for -------------------------------------------------------

def test_a_restraint_resolves_its_atom_indices_from_the_named_collective_variable(workspace):
    """The indices come from cv.yaml, so the biased and reported quantities cannot differ."""
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 500.0}
    """)
    restraints = load_umbrella_definition(path, _definition(workspace))
    assert len(restraints) == 1
    assert restraints[0].cv == "phi_ALA"
    assert restraints[0].atom_indices == (4, 6, 8, 14)


def test_a_restraint_naming_an_unknown_variable_is_refused_with_the_known_names(workspace):
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: omega_ALA, centre_deg: 0.0, force_constant: 1.0}
    """)
    with pytest.raises(UmbrellaError, match="not defined in the collective-variable file"):
        load_umbrella_definition(path, _definition(workspace))


def test_a_restraint_cannot_carry_its_own_atom_indices(workspace):
    """The one key that must never be accepted here, however reasonable it looks."""
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: phi_ALA, centre_deg: 0.0, force_constant: 1.0, atom_indices: [1, 2, 3, 4]}
    """)
    with pytest.raises(UmbrellaError, match="unknown key"):
        load_umbrella_definition(path, _definition(workspace))


# --- the definition's own refusals ------------------------------------------------------------

@pytest.mark.parametrize("body,message", [
    ("schema_version: 1\nrestraints: []\n", "no `restraints` list"),
    ("schema_version: 2\nrestraints: [{cv: phi_ALA, centre_deg: 0.0, force_constant: 1.0}]\n",
     "schema_version"),
    ("schema_version: 1\nrestraints:\n  - {cv: phi_ALA, force_constant: 1.0}\n",
     "has no centre_deg"),
    ("schema_version: 1\nrestraints:\n  - {cv: phi_ALA, centre_deg: 0.0}\n",
     "has no force_constant"),
    ("schema_version: 1\nrestraints:\n"
     "  - {cv: phi_ALA, centre_deg: 0.0, force_constant: 0.0}\n", "applies no bias"),
    ("schema_version: 1\nrestraints:\n"
     "  - {cv: phi_ALA, form: gaussian, centre_deg: 0.0, force_constant: 1.0}\n", "has form"),
    ("schema_version: 1\nrestraints:\n"
     "  - {cv: phi_ALA, form: flat_bottom, centre_deg: 0.0, force_constant: 1.0}\n",
     "positive half_width_deg"),
    ("schema_version: 1\nrestraints:\n"
     "  - {cv: phi_ALA, centre_deg: 0.0, force_constant: 1.0, half_width_deg: 10.0}\n",
     "no half-width"),
    ("schema_version: 1\nrestraints:\n"
     "  - {cv: phi_ALA, centre_deg: 0.0, force_constant: 1.0}\n"
     "  - {cv: phi_ALA, centre_deg: 90.0, force_constant: 1.0}\n", "already restrains"),
])
def test_a_malformed_definition_is_refused_by_name(workspace, body, message):
    with pytest.raises(UmbrellaError, match=message):
        load_umbrella_definition(_umbrella(workspace, body), _definition(workspace))


def test_the_record_carries_the_resolved_indices(workspace):
    """A reader of the output should not need cv.yaml to learn which atoms were biased."""
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: psi_ALA, form: flat_bottom, centre_deg: 140.0, half_width_deg: 30.0,
             force_constant: 500.0}
    """)
    record = load_umbrella_definition(path, _definition(workspace))[0].record()
    assert record["atom_indices"] == [6, 8, 14, 16]
    assert record["half_width_deg"] == 30.0
    assert record["form"] == "flat_bottom"


# --- the configuration gate -------------------------------------------------------------------

def _config(workspace, body):
    path = workspace / "u.config"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_a_complete_umbrella_configuration_resolves(workspace):
    _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: phi_ALA, centre_deg: -60.0, force_constant: 500.0}
    """)
    resolved = resolve_md_config(_config(workspace, """
        protocol: umbrella
        solvent: implicit
        collective_variables: {file: cv.yaml, interval_steps: 100}
        umbrella: {file: umbrella.yaml}
    """))
    assert resolved["protocol"] == "umbrella"
    assert resolved["umbrella"]["file"] == "umbrella.yaml"


@pytest.mark.parametrize("body,message", [
    ("protocol: umbrella\nsolvent: implicit\n"
     "collective_variables: {file: cv.yaml, interval_steps: 100}\n",
     "umbrella.file is not set"),
    ("protocol: umbrella\nsolvent: implicit\numbrella: {file: umbrella.yaml}\n",
     "collective_variables.file is not set"),
    ("protocol: cMD\nsolvent: implicit\numbrella: {file: umbrella.yaml}\n",
     "protocol is cMD"),
])
def test_an_incoherent_umbrella_configuration_is_refused(workspace, body, message):
    with pytest.raises(ConfigError, match=message):
        resolve_md_config(_config(workspace, body))


def test_the_configuration_forms_match_the_runtime_forms():
    """A form the configuration accepts must be one the runtime can build.

    They are stated in two modules deliberately -- the runtime must not import the configuration
    layer -- so this is what keeps them one list.
    """
    from md_tools.md.torsion_restraints import RESTRAINT_FORMS as RUNTIME_FORMS

    assert tuple(RESTRAINT_FORMS) == tuple(RUNTIME_FORMS)


def test_umbrella_is_a_known_protocol():
    from md_tools.build.md import PROTOCOLS

    assert "umbrella" in PROTOCOLS


def test_a_window_may_mix_force_constants(workspace):
    """Each restraint carries its own strength, so a window need not use one for all of them.

    The first implementation shared a single global constant, which made this configuration build
    successfully and fail at RUN time -- after minimisation and three equilibration stages. It is
    a configuration error or it is nothing; it must not be a late one.
    """
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 800.0}
          - {cv: psi_ALA, form: harmonic, centre_deg: 140.0, force_constant: 50.0}
    """)
    restraints = load_umbrella_definition(path, _definition(workspace))
    assert [r.force_constant for r in restraints] == [800.0, 50.0]


def test_a_window_may_mix_forms(workspace):
    """Harmonic and flat-bottom in one window: they become two forces, one per expression."""
    path = _umbrella(workspace, """
        schema_version: 1
        restraints:
          - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 500.0}
          - {cv: psi_ALA, form: flat_bottom, centre_deg: 140.0, half_width_deg: 30.0,
             force_constant: 500.0}
    """)
    restraints = load_umbrella_definition(path, _definition(workspace))
    assert {r.form for r in restraints} == {"harmonic", "flat_bottom"}


def test_umbrella_works_in_explicit_solvent(tmp_path):
    """The protocol is not implicit-only.

    Explicit solvent brings PME and a barostat, and the restraint is added to the System before
    either is configured. Checked at the configuration level -- the biasing force itself is
    solvent-independent, and running a solvated window here would cost minutes to prove something
    `test_torsion_restraint_cuda.py` already establishes.
    """
    (tmp_path / "cv.yaml").write_text(CV_YAML, encoding="utf-8")
    (tmp_path / "umbrella.yaml").write_text(textwrap.dedent("""
        schema_version: 1
        restraints:
          - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 500.0}
    """), encoding="utf-8")
    config = tmp_path / "u.config"
    config.write_text(textwrap.dedent("""
        protocol: umbrella
        solvent: explicit
        collective_variables: {file: cv.yaml, interval_steps: 100}
        umbrella: {file: umbrella.yaml}
    """), encoding="utf-8")

    resolved = resolve_md_config(config)
    assert resolved["solvent"] == "explicit"

    from md_tools.build.md import stage_plan

    production = [s for s in stage_plan(resolved) if s["name"] == "umbrella"]
    assert len(production) == 1, "explicit solvent produced no umbrella production stage"
    # NPT in explicit solvent, and the restraint file reaches the stage that runs it.
    assert production[0]["ensemble"] == "NPT"
    assert production[0]["umbrella_file"] == "umbrella.yaml"


def test_an_implicit_window_runs_at_fixed_volume():
    """The control for the test above: implicit has no box, so it cannot be NPT."""
    from md_tools.build.md import MD_SCHEMA, stage_plan

    resolved = MD_SCHEMA.resolve({"protocol": "umbrella", "solvent": "implicit",
                                  "collective_variables": {"file": "cv.yaml",
                                                           "interval_steps": 100},
                                  "umbrella": {"file": "umbrella.yaml"}})
    production = [s for s in stage_plan(resolved) if s["name"] == "umbrella"]
    assert production[0]["ensemble"] == "NVT"
