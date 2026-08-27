"""Annealed importance sampling: the path, the source contract, and the work bookkeeping.

Split by cost, as elsewhere in this suite:

    (unmarked)  the schedule arithmetic and configuration validation. Nothing is built.
    slow        one system built, one project generated, generated source inspected.
    gpu         energies, forces or dynamics. CUDA, and nowhere else.

The scientific claim this file has to support is narrow and specific: a Context switched along the
tau path is, at every tau, the same Hamiltonian as a separately constructed static REST2 rung, and
the work columns beside the coordinates are the work of the switching that produced them. Both are
checked against something independent -- a separately built System, and an analytic telescoping
sum -- rather than against themselves.
"""
from __future__ import annotations

import csv
import importlib.util
import shutil

import pytest
import yaml

from .conftest import ALA_PDB, run_cli

# A switching duration chosen so the arithmetic is exact and the run is seconds long, not because
# it is a scientifically meaningful path: 40 steps at 2 fs is 40 updates, which divides by the 20
# observation intervals. This is a correctness test.
SMOKE_SWITCHING_PS = 0.08
SMOKE_TRAJECTORIES = 2


# --- the schedule, with nothing built -------------------------------------------------------

def test_the_default_path_is_linear_tau_from_half_to_zero_with_21_observations():
    from md_templates.openmm.defaults import default_document

    block = default_document("AIS")["AIS"]
    path, output = block["path"], block["output"]
    assert path["type"] == "rest2_tau"
    assert (path["tau_start"], path["tau_end"]) == (0.5, 0.0)
    assert path["interpolation"] == "linear"
    assert path["omega_exclusion"] is True
    assert path["enhanced_region"] == "solute"
    assert output["number_of_observations"] == 21
    assert output["include_start"] is True and output["include_end"] is True
    assert output["coordinates"] == "whole_system"
    # The required user inputs ship as null. They are not defaults with a value chosen for you.
    assert path["switching_duration_ps"] is None
    for key in ("trajectory", "start_time_ps", "end_time_ps", "number_of_trajectories"):
        assert block["source"][key] is None, key


def test_the_schedule_is_evenly_spaced_endpoint_inclusive_and_exact():
    """21 observations are 20 equal intervals in tau, and both endpoints are exact."""
    from md_templates.openmm import ais

    schedule = ais.switching_schedule(
        tau_start=0.5, tau_end=0.0, switching_duration_ps=0.08,
        parameter_update_interval_steps=1, number_of_observations=21, timestep_fs=2.0)

    assert schedule["total_steps"] == 40
    assert schedule["number_of_updates"] == 40
    assert schedule["updates_per_observation"] == 2
    observations = schedule["observations"]
    assert len(observations) == 21
    assert observations[0]["tau"] == 0.5 and observations[0]["protocol_step"] == 0
    # Exactly, not nearly: the last row is reported as the work of reaching tau_end.
    assert observations[-1]["tau"] == 0.0
    assert observations[-1]["protocol_step"] == 40

    spacing = [observations[i + 1]["tau"] - observations[i]["tau"] for i in range(20)]
    assert all(abs(step - spacing[0]) < 1e-12 for step in spacing), "tau steps are not equal"
    # s is quadratic in tau while sqrt(s) is linear -- the relation the path is defined by.
    for entry in observations:
        assert entry["s"] == pytest.approx((1.0 - entry["tau"]) ** 2)
        assert entry["sqrt_s"] == pytest.approx(1.0 - entry["tau"])


@pytest.mark.parametrize("duration, interval, observations, expected", [
    (0.03, 1, 21, "cannot be divided"),          # 15 updates, 20 intervals
    (0.04, 3, 21, "not a whole number of"),      # 20 steps, updates every 3
    (0.0301, 1, 21, "whole number of"),          # not a whole number of steps at all
])
def test_a_schedule_that_would_have_to_be_rounded_is_refused(duration, interval, observations,
                                                             expected):
    from md_templates.openmm import ais

    with pytest.raises(ValueError) as error:
        ais.switching_schedule(tau_start=0.5, tau_end=0.0, switching_duration_ps=duration,
                               parameter_update_interval_steps=interval,
                               number_of_observations=observations, timestep_fs=2.0)
    assert expected in str(error.value)


def _protocol(**patch):
    from md_templates.openmm.defaults import md_defaults

    document = md_defaults(methods=["AIS"])
    document["AIS"]["path"]["switching_duration_ps"] = SMOKE_SWITCHING_PS
    document["AIS"]["source"].update({"trajectory": "cMD/whole_system.dcd",
                                      "start_time_ps": 0.02, "end_time_ps": 0.4,
                                      "number_of_trajectories": SMOKE_TRAJECTORIES})
    for dotted, value in patch.items():
        block, key = dotted.split(".", 1)
        document["AIS"][block][key] = value
    return document


def test_a_complete_ais_protocol_resolves():
    from md_templates.openmm.config import resolve_md_config

    resolved = resolve_md_config(_protocol(), implicit=False)
    assert resolved["AIS"]["path"]["tau_start"] == 0.5
    assert resolved["AIS"]["source"]["start_time_ps"] == 0.02


@pytest.mark.parametrize("patch, expected", [
    ({"path.switching_duration_ps": None}, "switching_duration_ps is null"),
    ({"path.switching_duration_ps": -1.0}, "must be positive"),
    ({"path.tau_end": 0.5}, "the Hamiltonian never changes"),
    ({"path.tau_start": 1.0}, "must be in [0, 1)"),
    ({"path.interpolation": "geometric"}, "must be 'linear'"),
    ({"path.parameter_update_interval_steps": 0}, "positive whole number"),
    ({"source.trajectory": None}, "trajectory is null"),
    ({"source.end_time_ps": 0.0}, "earlier than start_time_ps"),
    ({"source.start_time_ps": -1.0}, "must be >= 0"),
    ({"source.number_of_trajectories": 0}, "positive whole number"),
    ({"source.frame_interval_ps": 1.0}, "is set but first_frame_time_ps is not"),
    ({"output.number_of_observations": 1}, "whole number >= 2"),
    ({"output.include_end": False}, "must both be true"),
    ({"execution.gpu_devices": "all"}, "'auto' or a non-empty list"),
])
def test_an_unusable_ais_configuration_is_refused_by_name(patch, expected):
    from md_templates.openmm.config import ConfigError, resolve_md_config

    with pytest.raises(ConfigError) as error:
        resolve_md_config(_protocol(**patch), implicit=False)
    assert expected in str(error.value)


# --- generation ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ais_project(tmp_path_factory):
    """A generated project with cMD at tau = 0.5 as the AIS source, sized to run in seconds.

    cMD runs at tau = 0.5 on purpose: AIS anneals away from the ensemble that seeded it, and the
    runtime refuses a source equilibrated at a different tau. A fixed-tau cMD walker IS that
    ensemble.
    """
    work = tmp_path_factory.mktemp("ais")
    shutil.copy2(ALA_PDB, work / "ALA.pdb")
    assert run_cli("md_openmm", "sys-config", "--method", "cMD", "AIS", cwd=work).returncode == 0

    system_path = work / "sys.config.yaml"
    document = yaml.safe_load(system_path.read_text())
    document["solvent"]["padding_nm"] = 0.5
    document["solvent"]["cutoff_nm"] = 0.5
    system_path.write_text(yaml.safe_dump(document, sort_keys=False))
    built = run_cli("md_openmm", "sys-gen", "-i", "./ALA.pdb", "--config", "sys.config.yaml",
                    "-of", "./inputs/", cwd=work)
    assert built.returncode == 0, built.stdout + built.stderr

    protocol_path = work / "md.config.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["minimization"]["max_iterations"] = 25
    for key, value in list(protocol["equilibration"].items()):
        if key.endswith("_duration_ps") and value is not None:
            protocol["equilibration"][key] = 0.02
    protocol["cMD"].update({"tau": 0.5, "duration_ns": 0.0004, "checkpoint_interval_ps": 0.1,
                            "whole_system_interval_ps": 0.02, "solute_interval_ps": 0.02})
    protocol["AIS"]["path"]["switching_duration_ps"] = SMOKE_SWITCHING_PS
    protocol["AIS"]["source"].update({"trajectory": "cMD/whole_system.dcd",
                                      "start_time_ps": 0.02, "end_time_ps": 0.40,
                                      "number_of_trajectories": SMOKE_TRAJECTORIES})
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))

    generated = run_cli("md_openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
                        "-of", "./MD/", cwd=work)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return work


@pytest.mark.slow
def test_md_gen_writes_the_standalone_ais_layout(ais_project):
    directory = ais_project / "MD" / "AIS"
    present = sorted(p.name for p in directory.iterdir())
    assert present == ["path_definition.yaml", "rest2_scaling.py", "run.py", "run.sh"], present
    # Trajectory directories and every runtime record are written by the run, not by md-gen.
    assert not list(directory.glob("trajectory_*"))
    assert not (directory / "resolved_run.yaml").exists()


@pytest.mark.slow
def test_ais_is_excluded_from_run_all(ais_project):
    """AIS starts from a trajectory the user already produced, so "run everything in order" would
    run it before its own input exists."""
    text = (ais_project / "MD" / "run_all.sh").read_text()
    assert "AIS" not in text
    assert "cMD" in text, "the fixture asked for cMD, which run_all.sh should still drive"


@pytest.mark.slow
def test_the_generated_ais_project_imports_nothing_from_this_package(ais_project):
    """Portable: an AIS project needs OpenMM, PyYAML, NumPy and MDTraj, and nothing else.

    The rule is the one the rest of the generated tree follows -- an IMPORT of the package, not any
    mention of it. `md_templates_version` in `path_definition.yaml` is a provenance field naming
    which implementation wrote the file, which is the point of recording it.
    """
    import re

    from md_templates.openmm.mdgen import TEMPLATES

    checkout = str(TEMPLATES.parents[3])
    offenders = {}
    for path in sorted((ais_project / "MD" / "AIS").rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".yaml"):
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py" and re.search(r"^\s*(import|from)\s+md_templates\b", text,
                                              re.MULTILINE):
            offenders[path.name] = "imports md_templates"
        if checkout in text:
            offenders[path.name] = "contains a path into the checkout"
    assert not offenders, offenders


@pytest.mark.slow
def test_the_path_definition_records_the_schedule_and_the_conventions(ais_project):
    definition = yaml.safe_load(
        (ais_project / "MD" / "AIS" / "path_definition.yaml").read_text())
    schedule = definition["schedule"]
    assert schedule["number_of_observations"] == 21
    assert len(schedule["observations"]) == 21
    assert schedule["taus"][0] == 0.5 and schedule["taus"][-1] == 0.0
    assert schedule["number_of_updates"] % 20 == 0
    assert definition["scaling"]["source_parameter"] == "tau"
    assert "(1 - tau)^2" in definition["scaling"]["derived_s"]
    assert "U(tau_{j+1}, x_j) - U(tau_j, x_j)" in definition["work_convention"]
    assert definition["ensemble"]["constant_volume"] is True
    assert definition["ensemble"]["barostat"] is None
    assert definition["path"]["omega_exclusion"] is True


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_runtime(project):
    """Import the GENERATED `AIS/run.py`, which is the code a user actually runs."""
    return _load(project / "MD" / "AIS" / "run.py", "_generated_ais_run")


def _generated_scaling(project):
    """The GENERATED `rest2_scaling.py` -- the same file cMD and REST2 projects get."""
    return _load(project / "MD" / "AIS" / "rest2_scaling.py", "_generated_rest2_scaling")


@pytest.mark.slow
def test_source_frame_selection_is_inclusive_and_deterministic(ais_project):
    """Both time bounds include their frame, and the same project selects the same frames twice."""
    runtime = _generated_runtime(ais_project)
    # 0.02 ps apart, first frame at 0.02 ps -- the map a cMD run records.
    times = [0.02 + 0.02 * index for index in range(20)]
    evidence = {"frame_interval_ps": 0.02, "source": "test"}

    selected, eligible = runtime.select_source_frames(times, evidence)
    # The configured window is [0.02, 0.40], and both endpoints are frames.
    assert eligible[0] == 0, "the frame at start_time_ps must be eligible"
    assert eligible[-1] == 19, "the frame at end_time_ps must be eligible"
    assert len(eligible) == 20
    assert len(selected) == SMOKE_TRAJECTORIES
    assert len(set(selected)) == len(selected), "default sampling is without replacement"

    again, _ = runtime.select_source_frames(times, evidence)
    assert again == selected, "selection is seeded and must repeat exactly"


@pytest.mark.slow
def test_a_window_with_too_few_frames_is_refused_rather_than_reusing_one(ais_project):
    runtime = _generated_runtime(ais_project)
    times = [0.02 + 0.02 * index for index in range(20)]
    # One eligible frame, two trajectories requested, no replacement.
    runtime.method["source"]["start_time_ps"] = 0.02
    runtime.method["source"]["end_time_ps"] = 0.02
    try:
        with pytest.raises(SystemExit) as error:
            runtime.select_source_frames(times, {"frame_interval_ps": 0.02, "source": "test"})
        assert "not two independent realisations" in str(error.value)
    finally:
        runtime.method["source"].update({"start_time_ps": 0.02, "end_time_ps": 0.40})


@pytest.mark.slow
def test_the_switching_system_carries_no_barostat(ais_project):
    """Fixed volume is a property of the System, not a runtime flag."""
    from openmm import XmlSerializer

    runtime = _generated_runtime(ais_project)
    scaling = _generated_scaling(ais_project)
    base = XmlSerializer.deserialize(
        (ais_project / "inputs" / "system.xml").read_text())
    switcher = scaling.TauSwitcher(base, runtime.SOLUTE_INDICES, [])
    system = switcher.prepared_system(0.5)
    names = [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]
    assert not any("Barostat" in name for name in names), names


# --- dynamic switching against static REST2 ---------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5])
def test_dynamic_switching_reproduces_a_static_rest2_system(ais_project, tau):
    """The Hamiltonian must actually change through force parameters, and match a built rung.

    This is what separates exact switching from interpolating between two endpoint energies: a
    separately constructed `build_scaled_system(..., tau)` -- the same function REST2 uses to build
    a rung -- and the live switched Context are compared on the total energy AND on every atom's
    force. Forces are the stronger test: an energy can agree by cancellation, a full force array
    cannot.
    """
    import numpy as np
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit

    scaling = _generated_scaling(ais_project)
    inputs = ais_project / "inputs"
    base = XmlSerializer.deserialize((inputs / "system.xml").read_text())
    state = XmlSerializer.deserialize((inputs / "initial_state.xml").read_text())
    solute_doc = yaml.safe_load((inputs / "solute.yaml").read_text())
    solute = list(range(int(solute_doc["n_solute_atoms"])))
    omega = [tuple(b) for b in solute_doc["rest2"]["omega_excluded_bonds"]]
    assert omega, "the ALA fixture should have omega bonds to exclude"

    platform = Platform.getPlatformByName("CUDA")
    # Double precision so the comparison is limited by the Hamiltonians, not by the platform.
    properties = {"Precision": "double"}

    def measure(context):
        result = context.getState(getEnergy=True, getForces=True)
        return (result.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.array(result.getForces().value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer)))

    static = scaling.build_scaled_system(base, solute, tau, omega)
    reference = Context(static, VerletIntegrator(0.001 * unit.femtosecond), platform, properties)
    reference.setPeriodicBoxVectors(*static.getDefaultPeriodicBoxVectors())
    reference.setPositions(state.getPositions())
    static_energy, static_forces = measure(reference)
    del reference

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(0.5)
    context = Context(live, VerletIntegrator(0.001 * unit.femtosecond), platform, properties)
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(state.getPositions())
    switcher.set_tau(context, live, tau)
    dynamic_energy, dynamic_forces = measure(context)

    assert dynamic_energy == pytest.approx(static_energy, abs=1e-6, rel=0)
    assert np.abs(dynamic_forces - static_forces).max() < 1e-6

    # ...and switching away and back must land on the same Hamiltonian, not on a compounded one:
    # every set_tau restores the unscaled parameters before scaling.
    switcher.set_tau(context, live, 0.4)
    switcher.set_tau(context, live, tau)
    again_energy, again_forces = measure(context)
    assert again_energy == pytest.approx(static_energy, abs=1e-6, rel=0)
    assert np.abs(again_forces - static_forces).max() < 1e-6


@pytest.mark.gpu
@pytest.mark.slow
def test_omega_excluded_torsions_are_left_alone_by_the_dynamic_switcher(ais_project):
    """The same omega bonds REST2 leaves unscaled, checked on the switched force parameters."""
    from openmm import PeriodicTorsionForce, XmlSerializer

    from openmm import unit

    scaling = _generated_scaling(ais_project)
    inputs = ais_project / "inputs"
    base = XmlSerializer.deserialize((inputs / "system.xml").read_text())
    solute_doc = yaml.safe_load((inputs / "solute.yaml").read_text())
    solute = set(range(int(solute_doc["n_solute_atoms"])))
    omega = {frozenset((int(a), int(b)))
             for a, b in solute_doc["rest2"]["omega_excluded_bonds"]}

    def torsions(system):
        force = next(system.getForce(i) for i in range(system.getNumForces())
                     if isinstance(system.getForce(i), PeriodicTorsionForce))
        return [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]

    unscaled = torsions(base)
    switched = torsions(scaling.TauSwitcher(base, solute, omega).prepared_system(0.5))

    excluded_seen = scaled_seen = 0
    for original, now in zip(unscaled, switched):
        i, j, k, l = original[:4]
        # Stripped of units so the comparison is between numbers, not Quantities.
        k_before = original[6].value_in_unit(unit.kilojoule_per_mole)
        k_after = now[6].value_in_unit(unit.kilojoule_per_mole)
        in_solute = all(a in solute for a in (i, j, k, l))
        if in_solute and frozenset((int(j), int(k))) in omega:
            assert k_after == k_before, f"omega torsion {i}-{j}-{k}-{l} was scaled"
            excluded_seen += 1
        elif in_solute:
            assert k_after == pytest.approx(k_before * 0.25), \
                f"solute torsion {i}-{j}-{k}-{l} was not scaled by s"
            scaled_seen += 1
        else:
            assert k_after == k_before, "an environment torsion was scaled"
    assert excluded_seen and scaled_seen, (excluded_seen, scaled_seen)


@pytest.mark.gpu
@pytest.mark.slow
def test_frozen_coordinate_work_telescopes_to_the_endpoint_energy_difference(ais_project):
    """With coordinates held fixed, the summed increments must be exactly the endpoint difference.

    delta_W_j = U(tau_{j+1}, x) - U(tau_j, x) with x unchanged, so every intermediate term cancels
    and the total is U(tau_end, x) - U(tau_start, x). Anything else means the increments are not
    the quantity the record calls work.
    """
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit

    scaling = _generated_scaling(ais_project)
    inputs = ais_project / "inputs"
    base = XmlSerializer.deserialize((inputs / "system.xml").read_text())
    state = XmlSerializer.deserialize((inputs / "initial_state.xml").read_text())
    solute_doc = yaml.safe_load((inputs / "solute.yaml").read_text())
    solute = list(range(int(solute_doc["n_solute_atoms"])))
    omega = [tuple(b) for b in solute_doc["rest2"]["omega_excluded_bonds"]]

    schedule = yaml.safe_load(
        (ais_project / "MD" / "AIS" / "path_definition.yaml").read_text())["schedule"]
    taus = schedule["taus"]

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(taus[0])
    context = Context(live, VerletIntegrator(0.001 * unit.femtosecond),
                      Platform.getPlatformByName("CUDA"), {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(state.getPositions())

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    start_energy = energy()
    cumulative = 0.0
    for index in range(len(taus) - 1):
        before = energy()                        # no propagation: x is frozen
        switcher.set_tau(context, live, taus[index + 1])
        cumulative += energy() - before
    end_energy = energy()

    assert cumulative == pytest.approx(end_energy - start_energy, abs=1e-6, rel=0)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_constant_tau_diagnostic_path_does_zero_work(ais_project):
    """A path that never changes the Hamiltonian does no work, whatever the coordinates do.

    The public forward configuration refuses tau_start == tau_end, and this is why that refusal is
    a configuration rule rather than a physical impossibility: the diagnostic is meaningful and it
    is what pins the work convention to the parameter change rather than to the propagation.
    """
    from openmm import (Context, LangevinMiddleIntegrator, Platform, XmlSerializer, unit)

    scaling = _generated_scaling(ais_project)
    inputs = ais_project / "inputs"
    base = XmlSerializer.deserialize((inputs / "system.xml").read_text())
    state = XmlSerializer.deserialize((inputs / "initial_state.xml").read_text())
    solute_doc = yaml.safe_load((inputs / "solute.yaml").read_text())
    solute = list(range(int(solute_doc["n_solute_atoms"])))
    omega = [tuple(b) for b in solute_doc["rest2"]["omega_excluded_bonds"]]

    switcher = scaling.TauSwitcher(base, solute, omega)
    live = switcher.prepared_system(0.5)
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                          2.0 * unit.femtoseconds)
    integrator.setRandomNumberSeed(20260827)
    context = Context(live, integrator, Platform.getPlatformByName("CUDA"),
                      {"Precision": "double"})
    context.setPeriodicBoxVectors(*live.getDefaultPeriodicBoxVectors())
    context.setPositions(state.getPositions())
    context.setVelocitiesToTemperature(300 * unit.kelvin, 4242)

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    cumulative = 0.0
    for _ in range(20):
        before = energy()
        switcher.set_tau(context, live, 0.5)     # the same tau, every time
        cumulative += energy() - before
        integrator.step(2)                       # and the coordinates DO move
    assert cumulative == pytest.approx(0.0, abs=1e-6)


# --- a tiny CUDA run ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ais_run(ais_project):
    """The common chain, a fixed-tau cMD source at tau = 0.5, and two AIS paths. On CUDA."""
    pytest.importorskip("mdtraj", reason="the AIS runtime reads its source trajectory with MDTraj")
    from .conftest import run_stage

    project = ais_project / "MD"
    for stage in ("minimization", "eq/nvt_1kcal", "eq/npt_1kcal", "eq/npt_free"):
        result = run_stage(project / stage)
        assert result.returncode == 0, result.stdout + result.stderr
    produced = run_stage(project / "cMD")
    assert produced.returncode == 0, produced.stdout + produced.stderr

    switched = run_stage(project / "AIS")
    assert switched.returncode == 0, switched.stdout + switched.stderr
    return project / "AIS"


@pytest.mark.gpu
@pytest.mark.slow
def test_each_path_writes_exactly_21_frames_paired_one_to_one_with_21_work_rows(ais_run):
    from .conftest import dcd_header

    for index in range(SMOKE_TRAJECTORIES):
        directory = ais_run / f"trajectory_{index:04d}"
        header = dcd_header(directory / "observations.dcd")
        assert header["frames"] == 21, (index, header)
        rows = list(csv.DictReader((directory / "observations.csv").open()))
        assert len(rows) == 21, index
        # One row per frame, in order: the row's own frame index is its position.
        assert [int(r["coordinate_frame_index"]) for r in rows] == list(range(21))
        assert [int(r["observation_index"]) for r in rows] == list(range(21))
        assert {r["dcd_filename"] for r in rows} == {"observations.dcd"}


@pytest.mark.gpu
@pytest.mark.slow
def test_the_first_and_last_work_rows_say_what_the_convention_says_they_should(ais_run):
    rows = list(csv.DictReader((ais_run / "trajectory_0000" / "observations.csv").open()))
    first, last = rows[0], rows[-1]

    assert float(first["tau"]) == 0.5
    assert float(first["s"]) == pytest.approx(0.25)
    assert float(first["cumulative_work_kj_mol"]) == 0.0, "the source configuration has done no work"
    assert float(first["cumulative_reduced_work"]) == 0.0
    assert int(first["protocol_step"]) == 0

    assert float(last["tau"]) == 0.0
    assert float(last["s"]) == pytest.approx(1.0)
    assert float(last["cumulative_work_kj_mol"]) != 0.0

    # The cumulative column is the running sum of the incremental one, row by row.
    running = 0.0
    for row in rows:
        running += float(row["incremental_work_kj_mol"])
        assert float(row["cumulative_work_kj_mol"]) == pytest.approx(running, abs=1e-6)
    completion = yaml.safe_load((ais_run / "resolved_run.yaml").read_text())
    reported = completion["trajectories"][0]["total_work_kj_mol"]
    assert float(last["cumulative_work_kj_mol"]) == pytest.approx(reported, abs=1e-6)
    # beta is the one common beta and reduced work is beta*W, not a second temperature.
    beta = 1.0 / (0.008314462618 * float(last["temperature_kelvin"]))
    assert float(last["cumulative_reduced_work"]) == pytest.approx(
        beta * float(last["cumulative_work_kj_mol"]), rel=1e-9)


@pytest.mark.gpu
@pytest.mark.slow
def test_paths_are_independent_with_distinct_seeds_and_a_recorded_device(ais_run):
    """Independent realisations: separate directories, separate DCDs, separate seeds."""
    record = yaml.safe_load((ais_run / "resolved_run.yaml").read_text())
    entries = record["trajectories"]
    assert len(entries) == SMOKE_TRAJECTORIES

    assert len({e["integrator_seed"] for e in entries}) == len(entries), "seeds are shared"
    assert len({e["velocity_seed"] for e in entries}) == len(entries)
    for entry in entries:
        assert entry["integrator_seed"] != entry["velocity_seed"]

    # One DCD per path, and never a second path appended to an existing one.
    dcds = sorted(ais_run.glob("trajectory_*/observations.dcd"))
    assert len(dcds) == SMOKE_TRAJECTORIES
    assert not (ais_run / "observations.dcd").exists(), \
        "independent paths must not be concatenated into one trajectory"

    devices = record["gpu_devices"]
    if devices:
        # Deterministic round-robin, so a reader can say which card produced which path.
        expected = [devices[index % len(devices)] for index in range(len(entries))]
        assert [e["gpu_device"] for e in entries] == expected
    assert record["platform"] == "CUDA"
    assert record["status"] == "completed"
    assert record["completed_trajectory_indices"] == list(range(SMOKE_TRAJECTORIES))
    assert record["failed_trajectory_indices"] == []


@pytest.mark.gpu
@pytest.mark.slow
def test_the_run_records_the_source_selection_the_platform_and_the_limitations(ais_run):
    selected = list(csv.DictReader((ais_run / "selected_initial_frames.csv").open()))
    assert len(selected) == SMOKE_TRAJECTORIES
    assert {float(row["source_tau"]) for row in selected} == {0.5}
    for row in selected:
        assert 0.02 - 1e-9 <= float(row["source_time_ps"]) <= 0.40 + 1e-9

    provenance = yaml.safe_load((ais_run / "provenance.yaml").read_text())
    assert provenance["source_trajectory"]["window_is_inclusive"] is True
    assert provenance["source_trajectory"]["tau"] == 0.5
    assert provenance["ensemble"]["constant_volume"] is True
    assert provenance["environment"]["openmm"]
    text = " ".join(provenance["limitations"])
    for expected in ("forward path only", "no mid-path restart", "pressure-volume",
                     "no velocities", "no free-energy estimator"):
        assert expected in text, expected


@pytest.mark.gpu
@pytest.mark.slow
def test_no_barostat_is_active_during_switching(ais_run):
    """Fixed volume, stated by the run itself and visible in the box it kept."""
    from openmm import XmlSerializer

    log = (ais_run / "trajectory_0000" / "stdout.log").read_text()
    assert "0 barostat(s) in the switching System" in log
    assert "fixed volume" in log

    state = XmlSerializer.deserialize(
        (ais_run / "trajectory_0000" / "final_state.xml").read_text())
    vectors = state.getPeriodicBoxVectors()
    assert vectors is not None, "an explicit path keeps the box of the frame it started from"


@pytest.mark.gpu
@pytest.mark.slow
def test_a_completed_path_is_skipped_and_not_appended_to(ais_run):
    """Rerunning must not add a second path to an existing observations.dcd."""
    from .conftest import dcd_header, run_stage

    before = dcd_header(ais_run / "trajectory_0000" / "observations.dcd")["frames"]
    again = run_stage(ais_run)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "already complete" in (again.stdout + again.stderr)
    after = dcd_header(ais_run / "trajectory_0000" / "observations.dcd")["frames"]
    assert after == before == 21
