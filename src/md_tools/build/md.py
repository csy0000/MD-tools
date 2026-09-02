"""`md-openmm build-md` -- a protocol configuration to readable run scripts.

The generated directory is self-contained and portable: every script is a small entry point that
imports the *installed* `md_tools` runtime, declares its resolved settings as a literal dict, and
calls into the runtime. No script contains an absolute path, a reference to a source checkout, or
a copy of the physics.

Durations are **integer step counts**, everywhere, because a step count is exact and a duration in
picoseconds is not: 5 ns at 2 fs is 2,500,000 steps, and a configuration that stores 5.0 and
multiplies by a timestep it may not have been written against will happily run a different length.
Each stage's log prints both the step count and the physical time it works out to under the
timestep actually used, so the reader never has to do the arithmetic and the record never loses
the exact count.

Two shapes, same physics:

  split       min.py, the equilibration chain, then the production script, plus run.sh
  all-in-one  md.py running the identical stages in one process, plus run.sh

The stage boundaries, resolved settings, seeds, logs, checkpoints and restart semantics are the
same in both; `--all-in-one` changes how many processes run them, not what is run.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

import yaml

from .record import LogWriter
from .strict import ConfigError, Field, Schema, Section
from ..openmm.timestep import ORDINARY_TIMESTEP_FS
from ..openmm.system_defaults import DEFAULT_BAROSTAT_FREQUENCY_STEPS

PROTOCOLS = ("cMD", "REST2", "rREST2", "AIS")

MD_SCHEMA = Schema(
    "cMD.config / REST2.config / rREST2.config",
    doc="Protocol, stage lengths in steps, and reporting intervals for `md-openmm build-md`.",
    fields=[
        Field("protocol", str, default="cMD", enum=PROTOCOLS,
              doc="cMD is plain molecular dynamics. REST2 adds a replica-exchange ladder in which "
                  "only the solute's Hamiltonian is scaled. rREST2 adds a Boltzmann reservoir "
                  "refresh of the hottest rung. AIS runs non-equilibrium switching paths from an "
                  "EXISTING equilibrium source ensemble -- it has no minimisation or "
                  "equilibration chain of its own, because its input is a trajectory you have "
                  "already produced."),
        Field("solvent", str, default="explicit", enum=("explicit", "implicit"),
              doc="Must match the System `build-top` produced. Under implicit solvent there is no "
                  "box and no barostat, so the pressure-coupled equilibration stages are replaced "
                  "by honest NVT equivalents with different file names -- they are NOT NPT stages "
                  "with the pressure quietly ignored."),
    ],
    sections=[
        Section("dynamics", [
            Field("timestep_fs", (float, str), default="auto", unit="fs",
                  doc="`auto` (the default) resolves the timestep from the masses serialised in "
                      "built.xml when the run starts: 2.0 fs on ordinary hydrogens, 4.0 fs on "
                      "repartitioned ones. `build-md` never opens built.xml, so it cannot know "
                      "whether HMR was applied -- and a configuration CLAIMING it is not "
                      "evidence. A number may be given instead: it is honoured up to 3.0 fs, and "
                      "above that only if the System really was repartitioned. The resolved "
                      "value and its basis are recorded in every log."),
            Field("temperature_K", float, default=300.0, minimum=1.0, maximum=1000.0, unit="K",
                  doc="Thermostat temperature. Under REST2 every replica runs at this same "
                      "physical temperature; the ladder scales the Hamiltonian, not the bath."),
            Field("pressure_bar", float, default=1.0, minimum=0.0, maximum=1000.0, unit="bar",
                  doc="Barostat pressure. Ignored -- and refused if a stage claims NPT -- under "
                      "implicit solvent, which has no volume."),
            Field("friction_per_ps", float, default=1.0, minimum=0.01, maximum=100.0, unit="1/ps",
                  doc="LangevinMiddleIntegrator collision rate."),
            Field("barostat_interval_steps", int, default=DEFAULT_BAROSTAT_FREQUENCY_STEPS,
                  minimum=1, unit="steps",
                  doc="MonteCarloBarostat volume-move attempt interval, in STEPS. It is the "
                      "frequency, not the barostat's presence, that makes a stage NPT."),
            Field("restraint_kcal_per_mol_A2", float, default=1.0, minimum=0.0, maximum=1000.0,
                  unit="kcal/mol/A^2",
                  doc="Positional restraint on solute heavy atoms during the restrained "
                      "equilibration stages. A physical constant with real units: the statement "
                      "that stage LENGTHS are step counts does not make force constants unitless."),
            Field("tau", float, default=0.0, minimum=0.0, maximum=0.95,
                  doc="Fixed REST2 scaling for a cMD run. 0.0 (the default) is the unmodified "
                      "physical Hamiltonian. A non-zero value runs cMD at ONE rung of the REST2 "
                      "ladder -- the same scaling the ladder applies, held fixed -- which is how "
                      "a Boltzmann reservoir for rREST2 is generated, and how a hot ensemble is "
                      "produced without running an exchange. A scaled run is NVT by "
                      "construction: it must sample the top rung's fixed-volume ensemble, so a "
                      "barostat would sample the wrong distribution and is refused."),
            Field("phase_space_printout", int, default=0, minimum=0, unit="steps",
                  doc="Write a phase-space stream (positions, VELOCITIES and box) every N steps. "
                      "0 disables it. A reservoir needs complete samples including velocities, "
                      "which a trajectory does not carry, so this is what a reservoir source is "
                      "generated with."),
            Field("seed", int, default=1, minimum=0,
                  doc="Base random seed. Each stage derives its own from this plus the stage "
                      "name, so stages are independent and the whole chain is reproducible."),
            Field("platform", str, default=None, nullable=True,
                  enum=("CUDA", "OpenCL", "CPU", "Reference"),
                  doc="Force an OpenMM platform. Null lets OpenMM choose the fastest available."),
        ], doc="Physical constants and integrator settings. These carry real units."),
        Section("stages", [
            Field("minimization_iterations", int, default=1000, minimum=0,
                  doc="Minimiser ITERATIONS, not a time interval: minimisation does not integrate "
                      "and has no timestep. 0 skips minimisation."),
            Field("restrained_nvt_steps", int, default=5000, minimum=0, unit="steps",
                  doc="Restrained NVT heating/settling. 5000 steps = 10 ps at 2 fs."),
            Field("restrained_npt_steps", int, default=5000, minimum=0, unit="steps",
                  doc="Restrained NPT density equilibration. 5000 steps = 10 ps at 2 fs. Under "
                      "implicit solvent this becomes a second restrained NVT stage instead."),
            Field("unrestrained_npt_steps", int, default=5000, minimum=0, unit="steps",
                  doc="Unrestrained NPT, the last stage before production. 5000 steps = 10 ps at "
                      "2 fs. Under implicit solvent this becomes unrestrained NVT."),
            Field("production_steps", int, default=2_500_000, minimum=0, unit="steps",
                  doc="Production length. 2,500,000 steps = 5 ns at 2 fs. This is the "
                      "authoritative number; the log prints the derived ps and ns beside it."),
        ], doc="Stage lengths, as exact integer step counts."),
        Section("reporting", [
            Field("solute_printout", int, default=1000, minimum=0, unit="steps",
                  doc="Trajectory output interval. 1000 steps = 2 ps at 2 fs."),
            Field("system_printout", int, default=10000, minimum=0, unit="steps",
                  doc="Scalar state (energy, temperature, volume, density) interval, written to a "
                      "CSV beside the log."),
            Field("checkpoint_printout", int, default=10000, minimum=0, unit="steps",
                  doc="Checkpoint interval. A checkpoint is what an interrupted stage resumes "
                      "from, and it is written with a fingerprint of the configuration that "
                      "produced it so it cannot be resumed under different settings."),
        ], doc="Output intervals, in steps."),
        Section("rest2", [
            Field("number_of_replicas", int, default=4, minimum=2, maximum=64,
                  doc="States in the ladder. Every state runs at the same physical temperature."),
            Field("tau_max", float, default=0.5, minimum=0.0, maximum=0.95,
                  doc="The hottest rung's tau. The ladder is linear from 0.0 to this value. "
                      "tau = 0 is the unscaled physical Hamiltonian."),
            Field("exchange_interval_steps", int, default=5000, minimum=1, unit="steps",
                  doc="Steps of dynamics between exchange attempts. 5000 steps = 10 ps at 2 fs."),
            Field("number_of_exchanges", int, default=500, minimum=1,
                  doc="Exchange attempts. Total production per state is this times "
                      "exchange_interval_steps."),
            Field("state_trajectory", bool, default=True,
                  doc="Write one trajectory per fixed thermodynamic STATE (remd0.nc .. remdN.nc). "
                      "A state trajectory follows a state, not a walker; the filename carries the "
                      "state index and never the tau value."),
            Field("rem_log", bool, default=True,
                  doc="Write an Amber-style rem.log projection of the exchange history."),
            Field("neighbour_acceptance_report", bool, default=True,
                  doc="Report acceptance for each neighbouring pair. A single averaged acceptance "
                      "hides a ladder with one impassable gap."),
        ], doc="The REST2 ladder. Ignored when protocol is cMD. The Hamiltonian scaling itself -- "
               "bonds and angles unscaled, ordinary amide omega unscaled, eligible solute torsions "
               "and CMAP by (1-tau)^2, solute-solute nonbonded and 1-4 by (1-tau)^2, "
               "solute-environment by (1-tau), GB by (1-tau) -- is a property of the validated "
               "implementation and is not configurable here."),
        Section("ais", [
            Field("number_of_paths", int, default=100, minimum=1,
                  doc="How many independent switching paths to run. Each gets its own directory, "
                      "its own trajectory, and its own deterministic seeds."),
            Field("tau_start", float, default=0.5, minimum=0.0, maximum=0.95,
                  doc="The tau the path starts at. It must equal the tau of the source ensemble: "
                      "the path begins in the ensemble it anneals away from, and the source's own "
                      "record is checked against this rather than assumed."),
            Field("tau_end", float, default=0.0, minimum=0.0, maximum=0.95,
                  doc="The tau the path ends at. 0.0 is the unmodified physical Hamiltonian. "
                      "tau_start and tau_end must differ, or the Hamiltonian never changes and "
                      "every work value would be zero."),
            Field("switching_steps", int, default=50_000, minimum=1, unit="steps",
                  doc="The length of the switching path, as an exact step count. 50000 steps is "
                      "100 ps at 2 fs. WORK IS PATH-LENGTH DEPENDENT: a faster switch does more "
                      "dissipative work, so this is a scientific choice and not a performance "
                      "knob. The log states the derived ps."),
            Field("observation_interval_steps", int, default=2_500, minimum=1, unit="steps",
                  doc="How often a path is observed: one coordinate frame and one work row. "
                      "switching_steps must divide by this exactly, so the last observation lands "
                      "at tau_end. 50000/2500 gives 20 intervals and therefore 21 observations, "
                      "counting both endpoints. AN OBSERVATION IS NOT A STEP."),
            Field("parameter_update_interval_steps", int, default=1, minimum=1, unit="steps",
                  doc="How often tau moves. 1 changes the Hamiltonian every step -- 50000 "
                      "parameter changes over the path above. observation_interval_steps must "
                      "divide by this, or observations would not sit on the update grid."),
        ], doc="The switching path. Ignored unless protocol is AIS. Every length is an exact "
               "integer step count; nothing here is a duration that has to divide by a timestep."),
        Section("ais_source", [
            Field("trajectory", str, default=None, nullable=True,
                  doc="REQUIRED for AIS. The equilibrium trajectory the paths start from, "
                      "typically a fixed-tau cMD run at tau = ais.tau_start. Resolved relative to "
                      "the directory run.sh is invoked from."),
            Field("topology", str, default=None, nullable=True,
                  doc="Topology for reading that trajectory. Null uses the -p topology the run "
                      "was given, which is the usual case. The resolved choice is recorded."),
            Field("first_frame", int, default=0, minimum=0,
                  doc="First eligible frame, INCLUSIVE, as a 0-based index into the trajectory "
                      "file. A frame index, never a time: the two are interchangeable only when "
                      "the frame interval is known, and it is not always. Use this to discard "
                      "equilibration."),
            Field("last_frame", int, default=None, nullable=True, minimum=0,
                  doc="Last eligible frame, INCLUSIVE. Null means the final frame in the file."),
            Field("selection", str, default="uniform_random",
                  enum=("uniform_random", "evenly_spaced"),
                  doc="How starting frames are drawn from the eligible window. uniform_random "
                      "draws with the run's seed; evenly_spaced takes them at a fixed stride."),
            Field("allow_repeated_frames", bool, default=False,
                  doc="Whether two paths may start from the SAME frame. False by default: two "
                      "paths from one configuration are not two independent realisations, and "
                      "treating them as such understates the spread of the work distribution."),
        ], doc="Where the starting configurations come from. Ignored unless protocol is AIS."),
        Section("reservoir", [
            Field("enabled", bool, default=False,
                  doc="rREST2 only. Refresh the hottest rung from a pre-generated Boltzmann "
                      "reservoir instead of propagating it."),
            Field("path", str, default=None, nullable=True,
                  doc="Reservoir directory. Required when enabled."),
            Field("refresh_interval_exchanges", int, default=1, minimum=1,
                  doc="How often the hottest rung is refreshed from the reservoir, in exchange "
                      "attempts."),
            Field("velocities", str, default="resample", enum=("resample", "inherit"),
                  doc="Where a refreshed configuration's velocities come from. `resample` draws "
                      "them from the Maxwell-Boltzmann distribution at the run temperature; "
                      "`inherit` keeps the reservoir's own. Recorded explicitly because it is a "
                      "provenance question, not a tuning knob."),
        ], doc="rREST2 reservoir. Ignored unless protocol is rREST2."),
    ],
)


def _check_timestep(resolved: dict[str, Any]) -> None:
    """`timestep_fs` is a number or the word `auto`, and nothing else.

    The field accepts two types so `auto` can be written, which means the schema's own range check
    cannot run on it -- comparing a string with a float raises rather than refusing politely. The
    range is therefore enforced here, where the two cases are already separated.
    """
    from ..openmm.timestep import AUTO

    value = resolved["dynamics"]["timestep_fs"]
    if isinstance(value, str):
        if value.strip().lower() != AUTO:
            raise ConfigError(
                f"dynamics.timestep_fs: {value!r} is not a number and not {AUTO!r}. Write a "
                f"number of femtoseconds, or {AUTO!r} to resolve it from the masses in the built "
                f"System when the run starts.")
        return
    if not 0.1 <= float(value) <= 5.0:
        raise ConfigError(
            f"dynamics.timestep_fs: {value!r} is outside 0.1 .. 5.0 fs. A value above 3.0 fs is "
            f"additionally refused at run time unless the System was built with hydrogen mass "
            f"repartitioning.")


def _check_protocol(resolved: dict[str, Any]) -> None:
    protocol = resolved["protocol"]
    if protocol == "AIS":
        _check_ais(resolved)
    elif resolved["ais_source"]["trajectory"]:
        raise ConfigError(
            f"ais_source.trajectory is set but protocol is {protocol}. A source ensemble is only "
            f"consumed by AIS; cMD, REST2 and rREST2 generate their own starting state.")
    if protocol == "rREST2" and not resolved["reservoir"]["enabled"]:
        raise ConfigError("protocol is rREST2 but reservoir.enabled is false. rREST2 IS the "
                          "reservoir variant; without one it is plain REST2.")
    if resolved["reservoir"]["enabled"] and not resolved["reservoir"]["path"]:
        raise ConfigError("reservoir.enabled is true but reservoir.path is null")
    if resolved["reservoir"]["enabled"] and protocol != "rREST2":
        raise ConfigError(f"reservoir.enabled is true but protocol is {protocol}. A reservoir "
                          f"only has meaning for rREST2.")
    if resolved["dynamics"]["tau"] > 0.0 and resolved["protocol"] != "cMD":
        raise ConfigError(
            f"dynamics.tau is {resolved['dynamics']['tau']} but protocol is "
            f"{resolved['protocol']}. A REST2 or rREST2 ladder sets its own tau per rung; a fixed "
            f"tau belongs to a cMD run held at one rung.")
    if resolved["dynamics"]["phase_space_printout"] and resolved["dynamics"]["tau"] == 0.0:
        raise ConfigError(
            "dynamics.phase_space_printout is set but dynamics.tau is 0.0. A phase-space stream "
            "exists to seed a reservoir at the ladder's TOP rung; writing one from the unscaled "
            "Hamiltonian would produce a reservoir for a rung nothing runs at.")


def _check_ais(resolved: dict[str, Any]) -> None:
    """Everything about an AIS run that is decidable before a path is written.

    Refused here rather than inside a generated script, because an AIS run starts from somebody
    else's trajectory and the mistakes worth catching -- a schedule whose observations do not
    divide, endpoints that never move -- are all visible in the configuration.
    """
    from ..ais.schedule import switching_schedule

    ais = resolved["ais"]
    source = resolved["ais_source"]

    if abs(float(ais["tau_start"]) - float(ais["tau_end"])) < 1e-12:
        raise ConfigError(
            f"ais.tau_start and ais.tau_end are both {ais['tau_start']}, so the Hamiltonian never "
            f"changes and every work value would be zero. A switching path needs distinct "
            f"endpoints; the usual choice is 0.5 -> 0.0.")
    if not source["trajectory"]:
        raise ConfigError(
            "ais_source.trajectory is required for AIS. It is the equilibrium ensemble the "
            "switching paths start from -- there is no default, because it is data you produced.")

    # EVERY enabled step interval must divide the switching path exactly.
    #
    # A path is a complete object: it starts at tau_start and ends at tau_end, and an interval
    # that does not divide `switching_steps` cannot place a frame on the final step. The last
    # frame would then be at some interior tau, and a reader comparing "the end of path A" with
    # "the end of path B" would be comparing two different points along the switch. That is not a
    # rounding inconvenience; it is a different measurement.
    #
    # Zero means disabled, and a disabled stream is exempt.
    switching = int(resolved["ais"]["switching_steps"])
    reporting = resolved["reporting"]
    intervals = {
        "ais.observation_interval_steps": int(resolved["ais"]["observation_interval_steps"]),
        "reporting.solute_printout": int(reporting["solute_printout"]),
        "reporting.system_printout": int(reporting["system_printout"]),
        "reporting.checkpoint_printout": int(reporting["checkpoint_printout"]),
    }
    for key, interval in sorted(intervals.items()):
        if interval == 0:
            continue                                    # disabled; nothing to place
        if switching % interval:
            divisors = [d for d in range(1, switching + 1) if switching % d == 0]
            near = [d for d in divisors if abs(d - interval) <= max(interval, 10)] or divisors
            raise ConfigError(
                f"{key} is {interval}, which does not divide ais.switching_steps = {switching} "
                f"({switching} % {interval} = {switching % interval}).\n"
                f"  A switching path has to end ON its final step: with this interval the last "
                f"frame would fall at an interior tau, and the end of one path would not be "
                f"comparable with the end of another.\n"
                f"  Divisors of {switching} near {interval}: "
                f"{', '.join(str(d) for d in sorted(near)[:12])}.")
    if (source["last_frame"] is not None
            and int(source["last_frame"]) < int(source["first_frame"])):
        raise ConfigError(
            f"ais_source.last_frame ({source['last_frame']}) is before first_frame "
            f"({source['first_frame']}); the eligible window would be empty.")

    # The schedule's own arithmetic, refused with the numbers that would fix it.
    try:
        switching_schedule(
            tau_start=float(ais["tau_start"]), tau_end=float(ais["tau_end"]),
            switching_steps=int(ais["switching_steps"]),
            parameter_update_interval_steps=int(ais["parameter_update_interval_steps"]),
            observation_interval_steps=int(ais["observation_interval_steps"]),
            # The schedule's divisibility is pure STEP arithmetic and does not depend on the
            # timestep; the value is needed only to report the derived picoseconds. Under `auto`
            # the check uses the ordinary-mass value, because what is being validated here is the
            # step schedule, and the run re-derives the real duration once it reads the masses.
            timestep_fs=(float(resolved["dynamics"]["timestep_fs"])
                         if isinstance(resolved["dynamics"]["timestep_fs"], (int, float))
                         else ORDINARY_TIMESTEP_FS))
    except ValueError as error:
        raise ConfigError(str(error)) from None


MD_SCHEMA.checks = (_check_protocol, _check_timestep)


def resolve_md_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return MD_SCHEMA.resolve({})
    return MD_SCHEMA.load(Path(path))


# ---------------------------------------------------------------------------------------------
# the stage plan
# ---------------------------------------------------------------------------------------------

def stage_plan(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """The ordered stages, each fully resolved.

    Under implicit solvent the two pressure-coupled stages become NVT and are RENAMED, so nothing
    downstream can read `eq_npt_free` and believe a barostat was involved.
    """
    dyn, stages, rep = resolved["dynamics"], resolved["stages"], resolved["reporting"]
    implicit = resolved["solvent"] == "implicit"
    common = {
        "tau": dyn["tau"],
        # Deliberately NOT in `common`: a reservoir must be a Boltzmann sample of the ensemble the
        # ladder's top rung actually samples, and the equilibration stages are restrained and not
        # yet at equilibrium. Streaming phase space from them would fill a reservoir with
        # configurations drawn from the wrong distribution. Only production gets it.
        "phase_space_interval_steps": 0,
        "timestep_fs": dyn["timestep_fs"],
        "temperature_K": dyn["temperature_K"],
        "pressure_bar": dyn["pressure_bar"],
        "friction_per_ps": dyn["friction_per_ps"],
        "barostat_interval_steps": dyn["barostat_interval_steps"],
        "seed": dyn["seed"],
        "platform": dyn["platform"],
    }
    plan: list[dict[str, Any]] = []
    if resolved["protocol"] == "AIS":
        # Deliberately empty. AIS consumes an equilibrium ensemble that already exists; giving it
        # a minimisation and equilibration chain would run those BEFORE its own input existed, and
        # would quietly suggest the source is something this run produces.
        return plan
    plan.append({**common, "name": "min", "ensemble": "NVT",
                 "minimization_iterations": stages["minimization_iterations"], "steps": 0,
                 "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                 "trajectory_interval_steps": 0, "state_interval_steps": 0,
                 "checkpoint_interval_steps": 0,
                 "description": "Restrained energy minimisation of the built system."})
    plan.append({**common, "name": "eq_nvt_posres", "ensemble": "NVT",
                 "steps": stages["restrained_nvt_steps"],
                 "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                 "trajectory_interval_steps": rep["solute_printout"],
                 "state_interval_steps": rep["system_printout"],
                 "checkpoint_interval_steps": rep["checkpoint_printout"],
                 "description": "Restrained NVT: settle the solvent around a held solute."})
    if implicit:
        plan.append({**common, "name": "eq_nvt_posres_2", "ensemble": "NVT",
                     "steps": stages["restrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                     "trajectory_interval_steps": rep["solute_printout"],
                     "state_interval_steps": rep["system_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Second restrained NVT stage. This REPLACES the restrained "
                                    "NPT stage of an explicit-solvent run: implicit solvent has "
                                    "no box, so there is no volume to equilibrate."})
        plan.append({**common, "name": "eq_nvt_free", "ensemble": "NVT",
                     "steps": stages["unrestrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "trajectory_interval_steps": rep["solute_printout"],
                     "state_interval_steps": rep["system_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Unrestrained NVT. This REPLACES the unrestrained NPT stage "
                                    "of an explicit-solvent run."})
    else:
        plan.append({**common, "name": "eq_npt_posres", "ensemble": "NPT",
                     "steps": stages["restrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                     "trajectory_interval_steps": rep["solute_printout"],
                     "state_interval_steps": rep["system_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Restrained NPT: equilibrate the density with the solute held."})
        plan.append({**common, "name": "eq_npt_free", "ensemble": "NPT",
                     "steps": stages["unrestrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "trajectory_interval_steps": rep["solute_printout"],
                     "state_interval_steps": rep["system_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Unrestrained NPT, the last stage before production."})

    if resolved["protocol"] == "cMD":
        # A scaled run must sample the fixed-volume ensemble the ladder's rung samples, so it is
        # NVT whatever the solvent. This is a consequence of what tau means, not a preference.
        scaled = float(dyn["tau"]) > 0.0
        plan.append({**common, "name": "cMD",
                     "ensemble": "NVT" if (implicit or scaled) else "NPT",
                     "steps": stages["production_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "phase_space_interval_steps": dyn["phase_space_printout"],
                     "trajectory_interval_steps": rep["solute_printout"],
                     "state_interval_steps": rep["system_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Production molecular dynamics."})
    return plan


# ---------------------------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------------------------

_HEADER = '''#!/usr/bin/env python
"""{description}

Generated by `md-openmm build-md`. An ENTRY POINT, not an implementation: the settings are in
`resolved.config` beside this file and the behaviour is in the installed md_tools package, so a
correction reaches every generated directory rather than only the ones made after it.

    python {name}.py -p built.pdb -s built.xml {extra}-log {name}.log
"""
from md_tools.md import run_generated_stage

raise SystemExit(run_generated_stage(__file__, "{name}"))
'''


def _stage_script(stage: dict[str, Any], *, first: bool) -> str:
    extra = "" if first else "-c PREVIOUS.xml "
    return _HEADER.format(description=stage["description"], name=stage["name"], extra=extra)


_ALL_IN_ONE = '''#!/usr/bin/env python
"""Every stage of this protocol, in one process.

Generated by `md-openmm build-md --all-in-one`. Runs exactly the stages the split scripts run,
with the same resolved settings, boundaries, seeds, logs, checkpoints and restart semantics. The
only difference is the number of processes.

    python md.py -p built.pdb -s built.xml
"""
from md_tools.md import run_generated_workflow

raise SystemExit(run_generated_workflow(__file__))
'''


def _run_sh(plan: list[dict[str, Any]], *, all_in_one: bool, protocol: str) -> str:
    lines = ['#!/usr/bin/env bash',
             '# Run this protocol, in order.',
             '#',
             '# Generated by `md-openmm build-md`. Every path is explicit and relative to this',
             '# directory, so the whole directory can be moved. A stage that already reports',
             '# completion in its own machine record is skipped rather than silently rerun; an',
             '# interrupted stage resumes from its checkpoint only if the checkpoint matches the',
             '# configuration that produced it.',
             '#',
             '#   ./run.sh                 run from the built system in the parent directory',
             '#   ./run.sh ../built.pdb ../built.xml    or name them explicitly',
             'set -euo pipefail',
             '',
             'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
             'TOPOLOGY="${1:-${HERE}/../built.pdb}"',
             'SYSTEM="${2:-${HERE}/../built.xml}"',
             '# Captured before the shifts below consume the positional arguments.',
             'SOURCE="${3:-}"',
             '# Consume the two positional arguments if they were given, so that a path this',
             '# script has already used is not forwarded on as though it were a flag.',
             '# An `if` rather than `[[ ... ]] && shift`: under `set -e` a false test in the',
             '# latter form is a failing command, so calling run.sh with no arguments would exit.',
             'if [[ $# -ge 1 ]]; then shift; fi',
             'if [[ $# -ge 1 ]]; then shift; fi',
             'if [[ $# -ge 1 && "${1:-}" != -* ]]; then shift; fi',
             'cd "${HERE}"',
             '',
             'if [[ ! -f "${TOPOLOGY}" ]]; then',
             '  echo "run.sh: no topology at ${TOPOLOGY}" >&2; exit 2',
             'fi',
             'if [[ ! -f "${SYSTEM}" ]]; then',
             '  echo "run.sh: no system at ${SYSTEM}" >&2; exit 2',
             'fi',
             '']
    if protocol == "AIS":
        lines += [
            '# AIS starts from an equilibrium ensemble you have ALREADY produced. There is no',
            '# default for it: a wrong source is not a slower run, it is a different measurement.',
            'if [[ -z "${SOURCE}" ]]; then',
            '  echo "usage: ./run.sh TOPOLOGY SYSTEM SOURCE_TRAJECTORY" >&2',
            '  echo "" >&2',
            '  echo "AIS anneals away from an equilibrium ensemble and cannot generate one." >&2',
            '  echo "Give it a finished fixed-tau run, for example ../hot/cMD.dcd." >&2',
            '  exit 2',
            'fi',
            'if [[ ! -f "${SOURCE}" ]]; then',
            '  echo "run.sh: no source trajectory at ${SOURCE}" >&2; exit 2',
            'fi',
            '',
            'echo "== AIS =="',
            f'python {protocol}.py \\',
            '  -p "${TOPOLOGY}" -s "${SYSTEM}" -src "${SOURCE}" \\',
            f'  -log {protocol}.log "$@"',
            '']
    elif all_in_one:
        lines += ['python md.py -p "${TOPOLOGY}" -s "${SYSTEM}"', '']
    else:
        previous = None
        for stage in plan:
            name = stage["name"]
            call = [f'python {name}.py \\',
                    f'  -p "${{TOPOLOGY}}" -s "${{SYSTEM}}" \\',
                    f'  -x {name}.dcd -r {name}.xml -chk {name}.chk -log {name}.log \\']
            if previous:
                call.insert(2, f'  -c {previous}.xml \\')
            call[-1] = call[-1].rstrip(" \\")
            lines += [f'echo "== {name} =="'] + call + ['']
            previous = name
        if protocol in ("REST2", "rREST2"):
            lines += ['# Anything left in "$@" is passed to the ladder: --resume to finish an',
                      '# interrupted run, --verify-only to check stored output without running.',
                      f'echo "== {protocol} =="',
                      f'python {protocol}.py -p "${{TOPOLOGY}}" -s "${{SYSTEM}}" \\',
                      f'  -c {previous}.xml -log {protocol}.log "$@"',
                      '']
    lines += ['echo "run.sh: all stages reported completion"']
    return "\n".join(lines) + "\n"


_AIS_SCRIPT = '''#!/usr/bin/env python
"""AIS: {paths} independent switching paths, tau {tau_start} -> {tau_end}.

Generated by `md-openmm build-md`. An ENTRY POINT: the switching, the work convention and the tau
scaling are the validated implementation in the installed md_tools package, and the settings are
in `resolved.config` beside this file.

AIS CONSUMES AN EQUILIBRIUM ENSEMBLE THAT ALREADY EXISTS. It has no minimisation or equilibration
chain, because running one would produce a starting state that is not the ensemble the path is
defined to begin in. Point -src at a finished fixed-tau run at tau = {tau_start}.

    python AIS.py -p built.pdb -s built.xml -src ../hot/cMD.dcd
    python AIS.py -p built.pdb -s built.xml -src ../hot/cMD.dcd --check     # validate only
    python AIS.py -p built.pdb -s built.xml -src ../hot/cMD.dcd --paths 0-9

Each path is independent and has its own seeds. A completed path is skipped and never appended to;
an interrupted one is rerun from its source frame, because the work integral is only defined along
a whole path.
"""
from md_tools.ais import run_generated_ais

raise SystemExit(run_generated_ais(__file__))
'''


_REPLICA_SCRIPT = '''#!/usr/bin/env python
"""{protocol}: one coordinated replica-exchange ladder.

Generated by `md-openmm build-md`. An ENTRY POINT: the Hamiltonian scaling, the exchange
algorithm, the fixed state trajectories and the restart semantics are the validated implementation
in the installed md_tools package, and the ladder is in `resolved.config` beside this file.

    python {protocol}.py -p built.pdb -s built.xml -c eq_npt_free.xml
    mpiexec -n {states} python {protocol}.py -p built.pdb -s built.xml -c eq_npt_free.xml

One rank per state, or a single process driving every state in turn. Any other world size is
refused rather than silently reinterpreted.
"""
from md_tools.remd import run_generated_remd

raise SystemExit(run_generated_remd(__file__, protocol="{protocol}"))
'''


def build_scripts(*, config_path: Path | None, out_dir: Path, all_in_one: bool = False,
                  overwrite: bool = False, echo: bool = True) -> dict[str, Any]:
    """Generate the run scripts. Returns the machine record written beside them."""
    resolved = resolve_md_config(config_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ConfigError(
            f"{out_dir} exists and is not empty. Pass --overwrite to regenerate into it. "
            f"Regenerating over a directory that already holds run output would leave scripts "
            f"and results that were produced by different settings side by side.")
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = stage_plan(resolved)
    protocol = resolved["protocol"]
    log = LogWriter(out_dir / "build-md.log", record_type="build-md", echo=echo)
    log("md-openmm build-md")
    log("=" * 68)
    log.heading("Resolved configuration")
    log.field("protocol", protocol)
    log.field("solvent", resolved["solvent"])
    log.field("shape", "all-in-one" if all_in_one else "split")
    timestep = resolved["dynamics"]["timestep_fs"]
    # Step counts are authoritative and are always printed. A physical duration is derived ONLY
    # when the timestep is already a number: under `auto` it is not known until the run opens
    # built.xml and reads the masses, and printing a picosecond figure here would be a guess at
    # whether hydrogen mass repartitioning was applied. Every stage log states the real one.
    numeric = timestep if isinstance(timestep, (int, float)) else None
    log.field("timestep", f"{numeric} fs" if numeric else
              f"{timestep!r} (resolved from the System's masses when the run starts)")
    log.heading("Stages")
    for stage in plan:
        if stage["name"] == "min":
            detail = f"{stage['minimization_iterations']} iterations"
        elif numeric:
            ps = stage["steps"] * numeric / 1000.0
            detail = f"{stage['steps']} steps = {ps:g} ps ({ps / 1000.0:g} ns), {stage['ensemble']}"
        else:
            detail = f"{stage['steps']} steps, {stage['ensemble']}"
        log.field(stage["name"], detail)

    written: list[str] = []
    if all_in_one:
        (out_dir / "md.py").write_text(_ALL_IN_ONE, encoding="utf-8")
        written.append("md.py")
    else:
        for index, stage in enumerate(plan):
            path = out_dir / f"{stage['name']}.py"
            path.write_text(_stage_script(stage, first=index == 0), encoding="utf-8")
            written.append(path.name)

    if protocol == "AIS":
        # No `run` dict is embedded any more: the AIS script reads `resolved.config` beside it,
        # which is the single declaration of the workflow.
        path = out_dir / "AIS.py"
        path.write_text(_AIS_SCRIPT.format(
            paths=resolved["ais"]["number_of_paths"],
            tau_start=resolved["ais"]["tau_start"], tau_end=resolved["ais"]["tau_end"]),
            encoding="utf-8")
        written.append(path.name)
        observations = (resolved["ais"]["switching_steps"]
                        // resolved["ais"]["observation_interval_steps"] + 1)
        log.heading("AIS")
        log.field("paths", resolved["ais"]["number_of_paths"])
        log.field("tau", f"{resolved['ais']['tau_start']} -> {resolved['ais']['tau_end']}")
        switching_steps = resolved["ais"]["switching_steps"]
        log.field("switching", (f"{switching_steps} steps = "
                                f"{switching_steps * numeric / 1000.0:g} ps") if numeric
                               else f"{switching_steps} steps")
        log.field("observations", f"{observations} (both endpoints included)")
        log.field("source", resolved["ais_source"]["trajectory"])

    if protocol in ("REST2", "rREST2"):
        ladder = {
            "protocol": protocol,
            "solvent": resolved["solvent"],
            "n_states": resolved["rest2"]["number_of_replicas"],
            "tau_max": resolved["rest2"]["tau_max"],
            "exchange_interval_steps": resolved["rest2"]["exchange_interval_steps"],
            "number_of_exchanges": resolved["rest2"]["number_of_exchanges"],
            "state_trajectory": resolved["rest2"]["state_trajectory"],
            "rem_log": resolved["rest2"]["rem_log"],
            "neighbour_acceptance_report": resolved["rest2"]["neighbour_acceptance_report"],
            "reservoir": dict(resolved["reservoir"]),
            "dynamics": dict(resolved["dynamics"]),
        }
        path = out_dir / f"{protocol}.py"
        path.write_text(_REPLICA_SCRIPT.format(protocol=protocol,
                                               states=ladder["n_states"]), encoding="utf-8")
        written.append(path.name)
        log.heading(protocol)
        log.field("states", ladder["n_states"])
        log.field("tau ladder", f"0.0 .. {ladder['tau_max']} (linear)")
        # Step counts always; a duration only when the timestep is already a number. Under `auto`
        # it is not known until the run reads the System's masses, and the ladder's own log states
        # the real one.
        steps_between = ladder["exchange_interval_steps"]
        total_steps = ladder["number_of_exchanges"] * steps_between
        if numeric:
            log.field("exchange every", f"{steps_between} steps = "
                                        f"{steps_between * numeric / 1000.0:g} ps")
            production = total_steps * numeric / 1000.0
            log.field("production per state",
                      f"{production:g} ps ({production / 1000.0:g} ns)")
        else:
            log.field("exchange every", f"{steps_between} steps")
            log.field("production per state", f"{total_steps} steps")
        log.field("attempts", ladder["number_of_exchanges"])

    run_sh = out_dir / "run.sh"
    run_sh.write_text(_run_sh(plan, all_in_one=all_in_one, protocol=protocol), encoding="utf-8")
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    written.append("run.sh")

    (out_dir / "resolved.config").write_text(
        "# The configuration these scripts were generated from, fully resolved.\n"
        "# Every default is written out, so this file alone reproduces the generation.\n"
        + yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False), encoding="utf-8")
    written.append("resolved.config")

    log.heading("Outputs")
    for name in written:
        log.field(name, out_dir / name)
    log.update(protocol=protocol, solvent=resolved["solvent"],
               all_in_one=bool(all_in_one), resolved_config=resolved,
               stages=[{k: v for k, v in s.items()} for s in plan],
               files=written)
    log.complete()
    log.heading("Summary")
    log(f"  generated {len(written)} files in {out_dir}")
    log("  status: completed")
    log.save()
    return log.record
