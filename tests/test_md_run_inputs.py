"""The Amber-like `.in` files: what they refuse, and that they mean what `resolved.config` means.

Two properties are worth more than all the others here.

*Strictness*: a run input is read once, by a machine. Every mistake it can contain -- an unknown
key, a duplicate, a key in the wrong units -- is cheaper to refuse than to discover in the output,
because none of them fail at run time. They produce a run, with a setting that did nothing.

*Round-trip*: every `.in` that `build-md` generates resolves to exactly the `resolved.config`
beside it. Without that the two shapes are two configurations that merely look alike, and a person
who edits the readable one gets a run described by the other.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import re

import pytest
import yaml

from md_tools.build.strict import ConfigError
from md_tools.run.inputs import parse_run_input

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

MINIMAL = """\
&cntrl
  protocol = cMD,
  stage    = min,
/
"""


def _written(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "run.in"
    path.write_text(text, encoding="utf-8")
    return path


# --- what it reads --------------------------------------------------------------------------

def _config(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "md.config"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_minimal_input_resolves_through_the_schema_that_owns_the_defaults():
    """The parser defines no defaults of its own: it projects, and `resolve_md_config` decides.

    Asserted as an equality against the YAML route rather than by spot-checking a few fields. Two
    resolvers that agree on the three values a test happened to look at are exactly the failure
    this is written to exclude.
    """
    from md_tools.build.md import resolve_md_config

    parsed = parse_run_input(_written(MINIMAL))
    assert parsed.stage == "min"
    assert parsed.protocol == "cMD"
    assert parsed.resolved == resolve_md_config(_config("protocol: cMD\n"))


def test_comments_and_blank_lines_are_ignored():
    parsed = parse_run_input(_written(
        "! the whole file may be commented\n"
        "\n"
        "&cntrl\n"
        "  protocol = cMD,   ! trailing comments too\n"
        "  # and this spelling\n"
        "  stage    = min,\n"
        "/\n"))
    assert parsed.stage == "min"


def test_a_quoted_value_keeps_its_spelling():
    parsed = parse_run_input(_written(
        "&cntrl\n  protocol = AIS,\n/\n"
        "&AIS\n  source_traj = '../hot/run one.nc',\n/\n"))
    assert parsed.resolved["ais_source"]["trajectory"] == "../hot/run one.nc"


# --- what it refuses ------------------------------------------------------------------------

def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(ConfigError, match="nonsense"):
        parse_run_input(_written("&cntrl\n  protocol = cMD,\n  nonsense = 3,\n/\n"))


def test_a_key_in_the_wrong_units_is_refused_with_the_one_that_was_meant():
    """`timestep = 2` is the mistake this whole naming convention exists to make impossible."""
    with pytest.raises(ConfigError) as refusal:
        parse_run_input(_written("&cntrl\n  protocol = cMD,\n  timestep = 2,\n/\n"))
    assert "timestep_fs" in str(refusal.value)


@pytest.mark.parametrize("amber_key", ["nstlim", "ntpr", "ntwx", "gamma_ln", "ig", "temp0"])
def test_an_amber_control_is_refused_with_its_counterpart_named(amber_key):
    """People arrive from pmemd. Refusing without saying what to write instead wastes their time."""
    with pytest.raises(ConfigError) as refusal:
        parse_run_input(_written(f"&cntrl\n  protocol = cMD,\n  {amber_key} = 10,\n/\n"))
    message = str(refusal.value)
    assert amber_key in message and "Did you mean" in message, message


def test_a_duplicate_key_is_refused_rather_than_last_one_winning():
    with pytest.raises(ConfigError, match="twice"):
        parse_run_input(_written(
            "&cntrl\n  protocol = cMD,\n  temperature_K = 300.0,\n  temperature_K = 310.0,\n/\n"))


def test_a_repeated_section_is_refused():
    with pytest.raises(ConfigError, match="appears twice"):
        parse_run_input(_written("&cntrl\n  protocol = cMD,\n/\n&cntrl\n  solvent = explicit,\n/\n"))


def test_an_unclosed_section_is_refused():
    with pytest.raises(ConfigError, match="never closed"):
        parse_run_input(_written("&cntrl\n  protocol = cMD,\n"))


def test_an_unknown_section_is_refused_and_the_known_ones_are_listed():
    with pytest.raises(ConfigError) as refusal:
        parse_run_input(_written("&ewald\n  order = 4,\n/\n"))
    assert "&cntrl" in str(refusal.value)


def test_a_setting_outside_any_section_is_refused():
    with pytest.raises(ConfigError, match="outside any section"):
        parse_run_input(_written("protocol = cMD\n"))


def test_a_fortran_boolean_is_refused_with_the_one_spelling_that_works():
    with pytest.raises(ConfigError, match="true"):
        parse_run_input(_written(
            "&cntrl\n  protocol = REST2,\n/\n&remd\n  rem_log = .true.,\n/\n"))


def test_a_missing_input_file_is_refused_before_anything_else():
    with pytest.raises(ConfigError, match="no such run input"):
        parse_run_input(Path(tempfile.mkdtemp()) / "absent.in")


def test_a_semantic_error_is_reported_against_the_file_that_was_written():
    """The projection resolves through a temporary file. The refusal must name the user's."""
    broken = _written("&cntrl\n  protocol = AIS,\n  info_printout = 100,\n/\n"
                      "&AIS\n  switching_steps = 250,\n  observation_interval_steps = 10,\n"
                      "  source_traj = ../s.nc,\n/\n")
    with pytest.raises(ConfigError) as refusal:
        parse_run_input(broken)
    message = str(refusal.value)
    assert str(broken) in message, message
    assert "projected.config" not in message, message


# --- the round-trip -------------------------------------------------------------------------

@pytest.mark.parametrize("protocol", ["cMD", "REST2", "rREST2", "AIS"])
def test_every_generated_input_resolves_to_the_resolved_config_beside_it(protocol, tmp_path):
    """The round trip, now across the split between a SHARED input and a PER-RUN resolution.

    `resolved.config` is per run and the `.in` files are shared, so "beside it" is no longer
    literal. What must still hold is that the method's own input resolves to exactly the document
    the run recorded -- the seed included, which is why `run.config` is layered in the same way
    `md-run` layers it.

    THE PREPARATION INPUTS ARE DELIBERATELY EXCLUDED, and that is the consequence of making
    `min.in` and `eq_<k>.in` serve every method: they carry no `protocol` and no method block, so
    they resolve to a document with `protocol: cMD` and this protocol's ladder settings at their
    defaults. Asserting equality for them would be asserting that a shared file is not shared.
    """
    from .conftest import make_dataset_root

    make_dataset_root(tmp_path, solvent="explicit")
    from .conftest import make_states_for

    make_states_for(tmp_path, REPO / "configs" / "md" / f"{protocol}.config")
    out = tmp_path / f"{protocol}-run1"
    done = subprocess.run(CLI + ["build-md", "-odir", str(out),
                                 "--config", str(REPO / "configs" / "md" / f"{protocol}.config")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    resolved = yaml.safe_load((out / "resolved.config").read_text(encoding="utf-8"))
    method_input = tmp_path / "input" / f"{protocol}.in"
    assert method_input.is_file(), f"{protocol} generated no input/{protocol}.in"
    assert parse_run_input(method_input,
                           run_config=out / "run.config").resolved == resolved, method_input.name

    # And every preparation input must at least PARSE and name the stage it is filed as.
    for path in sorted((tmp_path / "input").glob("eq_*.in")):
        assert parse_run_input(path).stage is not None, path.name


def test_an_input_names_the_stage_the_script_of_the_same_name_runs(tmp_path):
    from .conftest import make_dataset_root

    make_dataset_root(tmp_path, solvent="explicit")
    from .conftest import make_states_for

    make_states_for(tmp_path, REPO / "configs" / "md" / "cMD.config")
    out = tmp_path / "cMD-run1"
    subprocess.run(CLI + ["build-md", "-odir", str(out),
                          "--config", str(REPO / "configs" / "md" / "cMD.config")],
                   capture_output=True, text=True, timeout=600, check=True)
    # THE SCRIPT IS NO LONGER BESIDE THE INPUT, and that is the layout rather than a slip: the
    # input is shared at `<system>/input/`, while the script that runs it is filed where its
    # OUTPUT goes -- minimisation shared at `<system>/min/`, equilibration in `<run>/eq/`, and
    # production at the run root. The property still under test is that the stage an input names
    # is the stage the script of that name runs.
    where = {"min": tmp_path / "min" / "min.py",
             "eq_1": out / "eq" / "eq_1.py",
             "eq_2": out / "eq" / "eq_2.py",
             "eq_3": out / "eq" / "eq_3.py",
             "cMD": out / "cMD.py"}
    seen = set()
    for path in (tmp_path / "input").glob("*.in"):
        if path.stem in ("AIS", "REST2", "rREST2"):
            continue
        stage = parse_run_input(path).stage
        assert stage is not None, path.name
        script = where[path.stem]
        assert script.is_file(), f"no script for {path.name} at {script}"
        assert f'run_generated_stage(__file__, "{stage}")' in script.read_text(encoding="utf-8")
        seen.add(path.stem)
    assert seen == set(where), sorted(seen)


def test_run_sh_drives_the_installed_command_rather_than_a_second_interface(tmp_path):
    from .conftest import make_dataset_root

    make_dataset_root(tmp_path, solvent="explicit")
    from .conftest import make_states_for

    make_states_for(tmp_path, REPO / "configs" / "md" / "REST2.config")
    out = tmp_path / "REST2-run1"
    subprocess.run(CLI + ["build-md", "-odir", str(out),
                          "--config", str(REPO / "configs" / "md" / "REST2.config")],
                   capture_output=True, text=True, timeout=600, check=True)
    text = (out / "run.sh").read_text(encoding="utf-8")
    # The input is SHARED, so run.sh reaches up to the dataset root for it.
    assert "md-openmm md-run -i ../input/min.in" in text, text
    # SUPERSEDED: this asserted `-x "${SYSTEM}"`, from the brief period when `-x` named the
    # serialised System on this surface. It follows Amber now -- `-s` is the System, `-x` the
    # trajectory -- and getting the two the wrong way round would write a trajectory over
    # built.xml, so the assertion is inverted rather than dropped.
    assert '-s "${SYSTEM}"' in text, text
    assert '-x "${SYSTEM}"' not in text, text
    # AND NO `-x` AT ALL NOW. A stage writes TWO coordinate streams -- `solute_prod<N>.nc` and
    # `whole_prod<N>.nc` -- and one `-x` cannot name both. Naming only the solute one would
    # silently reinstate the single whole-system trajectory the split exists to remove, so
    # run.sh names neither and each stage names its own.
    # No `-x` naming a STAGE trajectory: a stage writes two coordinate streams and
    # names them itself. The ladder line legitimately keeps `-x REST2.nc`, which is
    # the exchange record rather than a trajectory, so the check is specific.
    assert not re.search(r"-x \S+\.dcd", text), text
    # TWO FILES, TWO READERS -- named by md-run inside `-odir` now rather than restated here.
    # `-r`, `-o`, `-log` and `-chk` are taken verbatim against the working directory when they
    # are given, so spelling them in run.sh is how a stage came to write its restart next to
    # run.sh instead of into its own directory. The stage says WHERE it writes, once.
    assert "-odir ../min" in text, text
    assert "-odir eq" in text, text
    assert "-o min.out" not in text, text
    # One rank per state, and -ng stating the same number, spelled out rather than computed at
    # run time -- the ladder's size is a property of the configuration, not of the machine.
    assert "mpirun -n 4 md-openmm md-run -ng 4" in text, text
    assert "-i ../input/REST2.in" in text, text
    # Each segment has its own group file, and every rung is named there rather than by one -s.
    assert "--groupfile remd_groupfile.1" in text, text
    # The ladder's readable and provenance records, per segment, in the records directory.
    assert "-o remd_records/REST2_prod1.out" in text, text
    assert "-log remd_records/REST2_prod1.log" in text, text
    assert "/data3" not in text and str(tmp_path) not in text, "a machine path leaked into run.sh"


def test_run_sh_reaches_the_shared_input_for_every_protocol(tmp_path):
    """`-i` in a generated `run.sh` is ALWAYS the shared `../input/<name>.in`.

    The inputs belong to the SYSTEM -- an input says what a method was asked to do, which is a
    property of the system rather than of one repeat -- so no run directory holds a `.in` of its
    own, whatever its protocol. Branches of the generator typed a bare basename anyway, and they
    were invisible here because the only `run.sh` assertion above is for a SPLIT REST2 tree:

        AIS: md-openmm md-run -i AIS.in   -> "md-run: -i AIS.in: no such run input file"

    A retired `--all-in-one` branch had the same defect and named `cMD` literally besides, so an
    `umbrella` run asked for a file no generation has ever written. Each protocol is checked here
    so a further branch cannot reintroduce it quietly.
    """
    import mdtraj

    from .conftest import make_dataset_root

    make_dataset_root(tmp_path)

    # --- cMD: the shared production input -------------------------------------------------------
    single = tmp_path / "single.config"
    single.write_text(yaml.safe_dump(
        {"protocol": "cMD", "solvent": "implicit", "dynamics": {"seed": 5},
         "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                    "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                    "production_steps": 5},
         "reporting": {"crd_printout_solute": 5, "info_printout": 5,
                       "checkpoint_printout": 5}}), encoding="utf-8")
    subprocess.run(CLI + ["build-md", "-odir", "./cMD-run1", "--config", str(single)],
                   cwd=tmp_path, capture_output=True, text=True, timeout=600, check=True)
    text = (tmp_path / "cMD-run1" / "run.sh").read_text(encoding="utf-8")
    assert "md-openmm md-run -i ../input/cMD.in" in text, text
    assert "md-run -i cMD.in" not in text, text
    assert not (tmp_path / "cMD-run1" / "cMD.in").exists(), (
        "a run directory holds no .in of its own")

    # --- AIS: no preparation chain, still a shared input ----------------------------------------
    frames = mdtraj.load(str(tmp_path / "build" / "built.pdb"))
    mdtraj.join([frames] * 4).save_dcd(str(tmp_path / "source.dcd"))
    ais = tmp_path / "ais.config"
    ais.write_text(yaml.safe_dump(
        {"protocol": "AIS", "solvent": "implicit", "dynamics": {"seed": 5},
         "ais": {"number_of_paths": 2, "switching_steps": 10,
                 "observation_interval_steps": 5},
         "ais_source": {"trajectory": "../source.dcd"},
         "reporting": {"crd_printout_solute": 5, "info_printout": 5,
                       "checkpoint_printout": 5}}), encoding="utf-8")
    subprocess.run(CLI + ["build-md", "-odir", "./AIS-run1", "--config", str(ais)],
                   cwd=tmp_path, capture_output=True, text=True, timeout=600, check=True)
    text = (tmp_path / "AIS-run1" / "run.sh").read_text(encoding="utf-8")
    assert "md-openmm md-run -i ../input/AIS.in" in text, text
    assert "md-run -i AIS.in" not in text, text
    assert not (tmp_path / "AIS-run1" / "AIS.in").exists(), (
        "an AIS run directory holds no .in of its own")


def test_the_input_language_can_express_every_field_of_the_resolved_model():
    """A key the `.in` language cannot say is a key a generated input silently drops.

    Asserted over the model rather than over one example. `dynamics.platform` was missing, and no
    round-trip check found it because none of the shipped configurations set it -- the loss only
    appeared in a CUDA smoke test whose configuration did.
    """
    from md_tools.build.md import resolve_md_config
    from md_tools.run.inputs import SECTION_KEYS

    reachable = {target for keys in SECTION_KEYS.values() for target in keys.values()
                 if not target.startswith("_")}
    for protocol in ("cMD", "REST2", "rREST2", "AIS", "umbrella"):
        resolved = resolve_md_config(REPO / "configs" / "md" / f"{protocol}.config")
        for name, block in resolved.items():
            if isinstance(block, dict):
                unreachable = [f"{name}.{leaf}" for leaf in block
                               if f"{name}.{leaf}" not in reachable]
            else:
                unreachable = [] if name in reachable else [name]
            assert not unreachable, f"{protocol}: {unreachable} cannot be written in a .in file"


def test_a_configuration_cannot_state_a_platform_at_all(tmp_path):
    """SUPERSEDED: this asserted that `dynamics.platform` survived the .in round trip.

    It no longer exists to survive. The platform is a property of the machine, so it lives in
    `machine.openmm.platform` in the user configuration; a protocol config that carried it made a
    workflow shared between machines carry one machine's hardware. The key is refused with the
    migration rather than ignored, because a silently dropped platform is exactly the failure the
    round-trip test was written to catch.
    """
    from md_tools.build.strict import ConfigError

    config = tmp_path / "cuda.config"
    config.write_text("protocol: cMD\ndynamics:\n  platform: CUDA\n", encoding="utf-8")
    done = subprocess.run(CLI + ["build-md", "-odir", str(tmp_path / "cMD-run1"),
                                 "--config", str(config)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "machine.openmm.platform" in done.stderr, done.stderr

    from md_tools.run.inputs import SECTION_KEYS

    assert not [key for keys in SECTION_KEYS.values() for key in keys if key == "platform"], \
        "the .in language must not offer a platform either"

    # And the .in language refuses it by name, saying where it went rather than proposing the
    # nearest surviving key.
    with pytest.raises(ConfigError, match="machine.openmm.platform"):
        parse_run_input(_written("&cntrl\n  protocol = cMD,\n  platform = CUDA,\n/\n"))


SPEC_EXAMPLE = """\
&cntrl
  protocol = AIS,
  solvent  = explicit,
/
&AIS
  tau_start                  = 0.5,
  tau_end                    = 0.0,
  number_of_paths            = 100,
  switching_steps            = 250,
  observation_interval_steps = 10,
  source_frame_start         = 0,
  source_frame_stride        = 10,
  source_frame_selection     = uniform_random,
  timestep_fs                = auto,
  temperature_K              = 300.0,
  friction_per_ps            = 1.0,
  crd_printout_solute            = 10,
  info_printout            = 50,
  checkpoint_printout        = 50,
  random_seed                = 20260902,
  source_traj                = ../cMD_tau0p5/tau_0p5.dcd,
/
"""


def test_the_specified_ais_example_parses_and_every_key_takes_effect():
    """The instruction's own AIS example, and each key traced to the field it sets.

    `source_frame_stride` is asserted because it was originally parsed into a variable nothing
    read -- a key that is accepted and then discarded is exactly the failure the strict parser
    exists to prevent, and it is invisible unless something checks where the value went.
    """
    parsed = parse_run_input(_written(SPEC_EXAMPLE))
    assert parsed.protocol == "AIS"
    assert parsed.resolved["ais"] == {
        "number_of_paths": 100, "tau_start": 0.5, "tau_end": 0.0,
        "switching_steps": 250, "observation_interval_steps": 10,
        "parameter_update_interval_steps": 1,
        # The example names neither, so both come from the schema. `work` is the DEFAULT: an
        # input that says nothing about how to measure the work gets the two-evaluation direct
        # measurement, not the basis probe.
        "work_measurement": "work", "verify_every_updates": 0}
    assert parsed.resolved["ais_source"]["frame_stride"] == 10
    assert parsed.resolved["ais_source"]["trajectory"] == "../cMD_tau0p5/tau_0p5.dcd"
    assert parsed.resolved["dynamics"]["seed"] == 20260902
    assert parsed.resolved["reporting"] == {"crd_printout_solute": 10, "crd_printout_whole": 0,
                                            "info_printout": 50, "checkpoint_printout": 50}


def test_the_selection_spelling_is_named_when_a_near_miss_is_written():
    """`random` is refused, and the refusal says what to write. One spelling per concept."""
    with pytest.raises(ConfigError, match="uniform_random"):
        parse_run_input(_written(SPEC_EXAMPLE.replace("= uniform_random,", "= random,")))


def test_the_four_ais_cadences_are_independent(tmp_path):
    """SUPERSEDED: this asserted that `crd_printout_solute` had to equal the observation interval.

    That rule answered "how often is a configuration written" with the answer to "how often is
    work measured". They are different questions, and somebody who wants work every 10 steps and
    frames every 50 is asking for a smaller file, not for something confused. All four cadences
    are independent now and each divides `switching_steps` on its own.
    """
    parsed = parse_run_input(_written(
        SPEC_EXAMPLE.replace("crd_printout_solute            = 10,",
                             "crd_printout_solute            = 50,")))
    assert parsed.resolved["reporting"]["crd_printout_solute"] == 50
    assert parsed.resolved["ais"]["observation_interval_steps"] == 10

    # What is still refused is a cadence that cannot place a record on the final step.
    from md_tools.build.strict import ConfigError

    with pytest.raises(ConfigError, match="250"):
        parse_run_input(_written(
            SPEC_EXAMPLE.replace("crd_printout_solute            = 10,",
                                 "crd_printout_solute            = 100,")))
