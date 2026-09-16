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
from .strict import ConfigError, Field, Schema, Section, load_yaml_strictly
from ..openmm.timestep import ORDINARY_TIMESTEP_FS
from ..openmm.system_defaults import DEFAULT_BAROSTAT_FREQUENCY_STEPS

PROTOCOLS = ("cMD", "REST2", "rREST2", "AIS", "umbrella")

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
            Field("number_of_segments", int, default=1, minimum=1,
                  doc="How many SEGMENTS the production run is written as. 1, the default, is a "
                      "single `prod1` segment and the historical behaviour.\n"
                      "  It SPLITS the production total rather than multiplying it: "
                      "`production_steps` stays the whole run and each segment gets "
                      "`production_steps / number_of_segments`, so raising this re-divides the "
                      "same trajectory and never lengthens it. A value that does not divide "
                      "exactly is refused with the arithmetic that would fix it, because a final "
                      "short segment would make the last chunk incomparable with the others.\n"
                      "  On a REST2/rREST2 ladder there is no `production_steps`: production is "
                      "`number_of_exchanges * exchange_interval_steps`, so the split is of "
                      "`number_of_exchanges` and an exchange is never allowed to straddle two "
                      "segments.\n"
                      "  Each segment writes its own files -- `solute_state<i>_prod<N>.nc`, "
                      "`cv_state<i>_prod<N>.dat`, `restart_state<i>_prod<N>.json` -- so segments "
                      "cannot overwrite one another. That was a real defect: every chunk of a "
                      "five-chunk reference run wrote `_prod1`."),
        ], doc="Stage lengths, as exact integer step counts."),
        Section("reporting", [
            Field("crd_printout_solute", int, default=1000, minimum=0, unit="steps",
                  doc="Trajectory output interval. 1000 steps = 2 ps at 2 fs."),
            Field("crd_printout_whole", int, default=0, minimum=0, unit="steps",
                  doc="COORDINATE output interval for the WHOLE system, in steps, written to "
                      "`whole_prod<N>.nc` (or `whole_rep<i>_prod<N>.nc` per replica on a "
                      "ladder).\n"
                      "0, the default, writes no whole-system trajectory. Say what you want "
                      "kept: on a solvated system the whole stream is two orders of magnitude "
                      "larger than the solute one -- 1796 atoms against 22 for alanine dipeptide "
                      "in water -- so an unasked-for whole-system trajectory at a solute cadence "
                      "is how a 26 MB run becomes 2.1 GB.\n"
                      "Distinct from `crd_printout_solute`, which is the solute alone, and from "
                      "`info_printout`, which is scalars and no coordinates at all. Those three "
                      "were previously two, and the two were named for what a reader assumed "
                      "rather than for what they wrote."),
            Field("info_printout", int, default=10000, minimum=0, unit="steps",
                  doc="Scalar state (energy, temperature, volume, density) interval, written to a "
                      "CSV beside the log."),
            Field("checkpoint_printout", int, default=10000, minimum=0, unit="steps",
                  doc="Checkpoint interval. A checkpoint is what an interrupted stage resumes "
                      "from, and it is written with a fingerprint of the configuration that "
                      "produced it so it cannot be resumed under different settings."),
        ], doc="Output intervals, in steps."),
        Section("collective_variables", [
            Field("file", str, default=None, nullable=True,
                  doc="Path to a cv.yaml defining the torsions to report, resolved relative to "
                      "THIS configuration file. Null disables collective-variable reporting."),
            Field("interval_steps", int, default=0, minimum=0, unit="steps",
                  doc="Steps between collective-variable observations. Independent of the "
                      "trajectory and state-data intervals, and may be more frequent than "
                      "either. 0 disables collective-variable reporting."),
        ], doc="Torsion collective-variable reporting. Observation only: no Force is added and "
               "the Hamiltonian is unchanged. Both keys disable it by default, and supplying "
               "only one of them is an error rather than a guess."),
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
            Field("equilibration_steps", int, default=0, minimum=0, unit="steps",
                  doc="Relaxation run at EACH STATE'S OWN Hamiltonian before the first exchange "
                      "attempt, and not counted as production. 0 (the default) starts exchanging "
                      "immediately.\n"
                      "  This exists because the alternative is wrong in a way that is hard to "
                      "see. A ladder takes ONE starting state -- the group file refuses per-rung "
                      "coordinates, deliberately, since rungs must be states of the same system "
                      "-- so without this every rung begins from a configuration equilibrated "
                      "under tau = 0. The hot rungs then spend their opening exchanges relaxing "
                      "out of a distribution that is not theirs, and those samples are production "
                      "by every record that describes them.\n"
                      "  The driver has always relaxed each rung under its own scaled "
                      "Hamiltonian; only the number was unreachable from a configuration file."),
            Field("equilibration_per_tau", bool, default=False,
                  doc="Run the equilibration stages on EVERY RUNG, under that rung's own tau, "
                      "instead of once at tau = 0. REST2 and rREST2 only; refused for any other "
                      "protocol. Off by default.\n"
                      "  When true, the tau = 0 chain stops early: at minimisation under implicit "
                      "solvent, and after its NPT stages under explicit solvent, which run once at "
                      "tau = 0 to fix the box every rung then shares. Every rung -- tau = 0 "
                      "included -- then runs eq_nvt_posres (restrained_nvt_steps), "
                      "eq_nvt_posres_2 (restrained_npt_steps) and eq_nvt_free "
                      "(unrestrained_npt_steps), all at fixed volume, from the ladder's starting "
                      "state, each stage with its own seed per rung. The restraint is the stage "
                      "chain's, on the same atoms at the same strength; the rung Systems the "
                      "ladder propagates never carry it.\n"
                      "  Order: these stages, then `equilibration_steps`, then the first "
                      "exchange. Neither is production. Stages of 0 steps are skipped, and all "
                      "three at 0 is refused."),
            Field("state_trajectory", bool, default=True,
                  doc="Write one trajectory per fixed thermodynamic STATE "
                      "(solute_state<i>_prod<N>.nc, and whole_state<i>_prod<N>.nc when a "
                      "whole-system cadence is set). A state trajectory follows a state, not a "
                      "walker; the filename carries the state index and never the tau value."),
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
            Field("work_measurement", str, default="work", enum=("work", "components"),
                  doc="How the work increment at each parameter update is obtained, and what "
                      "else is recorded with it. This is a scientific choice AND the dominant "
                      "cost of an AIS run: everything else here controls what is written, this "
                      "controls what is computed.\n"
                      "  work        -- TWO energy evaluations per update, U(tau_k) and "
                      "U(tau_k+1). The work integral, and nothing more. This is the default "
                      "because it is what a free energy for the schedule you actually ran "
                      "needs.\n"
                      "  components  -- a THREE-point basis probe per update, at amplitudes "
                      "(0, 0.5, 1), from which the work follows analytically. It also gives the "
                      "potential as a FUNCTION of tau, which is what reweighting onto a "
                      "different schedule, a different endpoint, or a Hummer-Szabo estimator "
                      "evaluated at an unvisited tau requires. A single total work cannot "
                      "produce that function, and re-running at another tau is not "
                      "reweighting.\n"
                      "  Component columns are ABSENT from a `work` path rather than zero, so a "
                      "reader expecting them fails instead of treating a missing measurement as "
                      "a measured nought."),
            Field("verify_every_updates", int, default=0, minimum=0,
                  doc="In `components` mode, how often the fitted work is checked against a "
                      "directly measured U(tau_k+1) - U(tau_k). Costs two extra evaluations "
                      "whenever it fires.\n"
                      "  0 (the default) verifies the FIRST update of every path and no other. "
                      "That is not a token check: the three-group identity is a property of the "
                      "SYSTEM, not of the step -- a force carrying tau-dependence outside the "
                      "basis is outside it at any coordinate -- so one verified update per path "
                      "establishes the model the whole path relies on, for two evaluations "
                      "rather than two thousand.\n"
                      "  N > 0 re-verifies every N updates as well, for the case the first "
                      "update cannot cover: a force whose tau-dependence only switches on at "
                      "some geometry the path reaches later.\n"
                      "  Ignored in `work` mode, where the work IS the direct measurement and "
                      "there is nothing to cross-check it against."),
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
            Field("frame_stride", int, default=1, minimum=1,
                  doc="Take every Nth frame of the window as eligible. Consecutive frames of an "
                      "MD trajectory are correlated, so drawing paths from every frame draws "
                      "several of them from what is effectively one configuration. A stride is "
                      "the honest way to say how far apart samples have to be; it does not make "
                      "them independent, it stops them being obviously dependent."),
            Field("selection", str, default="uniform_random",
                  enum=("uniform_random", "evenly_spaced"),
                  doc="How starting frames are drawn from the eligible window. uniform_random "
                      "draws with the run's seed; evenly_spaced takes them at a fixed stride."),
            Field("allow_repeated_frames", bool, default=False,
                  doc="Whether two paths may start from the SAME frame. False by default: two "
                      "paths from one configuration are not two independent realisations, and "
                      "treating them as such understates the spread of the work distribution."),
        ], doc="Where the starting configurations come from. Ignored unless protocol is AIS."),
        Section("umbrella", [
            Field("file", str, default=None, nullable=True,
                  doc="Path to the restraint definition, resolved beside `resolved.config` -- the "
                      "same rule `collective_variables.file` follows.\n"
                      "A LIST of restraints does not fit a namelist `.in`, and inventing a "
                      "packed-string encoding for one would make the most consequential line of "
                      "an umbrella input the least readable. So the restraints live in their own "
                      "YAML, referenced by path, exactly as the collective variables they name "
                      "already do.\n"
                      "Each entry names a CV from `collective_variables.file` and says how it is "
                      "restrained -- see `md_tools.umbrella.load_umbrella_definition` for the "
                      "schema and every way it is refused."),
        ], doc="Umbrella sampling: restrain named collective variables and report them.\n"
               "Producing the biased series is what this protocol does. Turning a set of windows "
               "into a free-energy profile is ANALYSIS and is deliberately not here: WHAM and "
               "MBAR belong to the project asking the question, not to the engine generating the "
               "samples.\n"
               "A window needs `collective_variables.file` and `interval_steps` set too. The "
               "restraint resolves its `cv` name against that same file, so the quantity that is "
               "biased and the quantity that is reported are the same object by construction -- "
               "a run cannot restrain one torsion and report another."),

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
    _check_segments(resolved)
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
    if (resolved.get("rest2") or {}).get("equilibration_per_tau"):
        if protocol not in ("REST2", "rREST2"):
            raise ConfigError(
                f"rest2.equilibration_per_tau is true but protocol is {protocol}. It runs the "
                f"equilibration stages on every rung of a REST2/rREST2 ladder under that rung's "
                f"own tau; {protocol} has no ladder, so the setting would do nothing. Remove it, "
                f"or set it to false.")
        if not per_tau_equilibration_stages(resolved):
            raise ConfigError(
                "rest2.equilibration_per_tau is true but stages.restrained_nvt_steps, "
                "stages.restrained_npt_steps and stages.unrestrained_npt_steps are all 0, so no "
                "rung would be equilibrated at all. Give at least one of them a step count, or "
                "set rest2.equilibration_per_tau to false.")


def _check_segments(resolved: dict[str, Any]) -> None:
    """`stages.number_of_segments` must divide the production total exactly.

    A SPLIT, never a multiplier: the production total is what the configuration already states,
    and this decides how many files it is written as. So the only question is whether it divides,
    and a remainder is refused rather than rounded -- a final short segment would make the last
    chunk incomparable with the others, which is exactly the comparison segments exist to enable.

    The quantity being divided differs by protocol, and that is not cosmetic. A ladder has no
    `production_steps`: its production is `number_of_exchanges * exchange_interval_steps`, and the
    indivisible unit is an EXCHANGE. Splitting its step count instead could put a segment boundary
    part-way through an exchange interval, leaving a segment whose last interval was propagated but
    never attempted.
    """
    segments = int(resolved["stages"]["number_of_segments"])
    if segments == 1:
        return

    protocol = resolved["protocol"]
    # AIS HAS NO PRODUCTION STAGE TO SPLIT, so accepting this would be accepting an inert
    # setting -- which is worse than refusing it, because the run would report a segment count
    # nothing honoured and write `prod1` regardless. A switching campaign's unit is a PATH, and
    # `ais.number_of_paths` already says how many there are; each writes its own
    # `AIS_traj000n.nc` and is resumable on its own.
    if protocol == "AIS":
        raise ConfigError(
            f"stages.number_of_segments is {segments} but protocol is AIS, which has no "
            f"production stage to divide -- it never reads stages.production_steps. A switching "
            f"campaign is already divided: ais.number_of_paths sets how many paths there are, "
            f"and each writes its own trajectory and resumes independently. Remove "
            f"number_of_segments, or set it to 1.")
    if protocol in ("REST2", "rREST2"):
        quantity, total = "rest2.number_of_exchanges", int(
            resolved["rest2"]["number_of_exchanges"])
        unit = "exchange attempt"
    else:
        quantity, total = "stages.production_steps", int(
            resolved["stages"]["production_steps"])
        unit = "step"

    if total == 0:
        raise ConfigError(
            f"stages.number_of_segments is {segments} but {quantity} is 0, so the segments would "
            f"each hold nothing. Either give the run a production length or leave "
            f"number_of_segments at 1.")
    if total % segments:
        divisors = [d for d in range(1, total + 1) if total % d == 0]
        near = [d for d in divisors if d <= segments * 4] or divisors
        raise ConfigError(
            f"stages.number_of_segments is {segments}, which does not divide {quantity} = "
            f"{total} ({total} % {segments} = {total % segments}).\n"
            f"  Segments split the production total; they do not extend it. With this value the "
            f"last segment would be shorter than the rest, and the point of segmenting is that "
            f"each one is comparable with the others.\n"
            f"  Divisors of {total}: {', '.join(str(d) for d in sorted(near)[:12])}.")
    per_segment = total // segments
    if protocol in ("REST2", "rREST2") and per_segment < 1:
        raise ConfigError(
            f"stages.number_of_segments is {segments} and {quantity} is {total}, which is fewer "
            f"than one {unit} per segment.")


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
        "reporting.crd_printout_solute": int(reporting["crd_printout_solute"]),
        "reporting.info_printout": int(reporting["info_printout"]),
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
    # The four cadences are independent, and each is checked on its own above. They used to be
    # two: `crd_printout_solute` was forced equal to `observation_interval_steps`, which answered the
    # question "how often is a configuration written" with the answer to "how often is work
    # measured". Those are different questions -- work every 10 steps with frames every 50 is a
    # perfectly ordinary thing to want, and it was refused.

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
                         else ORDINARY_TIMESTEP_FS),
            # Refused at BUILD time, with the numbers that would fix it, rather than by every
            # generated script the first time it is run.
            cv_interval_steps=int(
                (resolved.get("collective_variables") or {}).get("interval_steps") or 0))
    except ValueError as error:
        raise ConfigError(str(error)) from None


#: The restraint forms `umbrella.restraints[].form` accepts. Mirrors
#: `md.torsion_restraints.RESTRAINT_FORMS`; a test pins the two together, because the build must
#: refuse a form the runtime cannot build rather than generating a script that fails on the node.
UMBRELLA_FORMS = ("harmonic", "flat_bottom")


def _check_umbrella(resolved: dict[str, Any]) -> None:
    """`umbrella.file` and `protocol: umbrella` are given together or not at all.

    Only coherence is checked here; the file's CONTENTS are validated when it is loaded, the same
    division `collective_variables.file` follows. Reading it at build time would mean parsing it
    twice under two sets of rules, and the second parse is the one the run actually uses.
    """
    protocol = resolved["protocol"]
    path = (resolved.get("umbrella") or {}).get("file")

    # A LADDER may carry restraints: the same ones on every rung, which is what hpREST2 runs and
    # what keeps the bias out of the exchange criterion (see `remd.protocol.apply_ladder_restraints`).
    # Per-rung variation is not offered: a bias that differed between rungs would enter the
    # acceptance probability, and the ladder would sample something nobody asked for.
    if protocol in ("REST2", "rREST2"):
        if not path:
            return
        if (resolved.get("reservoir") or {}).get("enabled"):
            raise ConfigError(
                f"umbrella.file = {path!r} restrains every rung, and reservoir.enabled is true. A "
                f"reservoir sample is drawn from a distribution generated WITHOUT this bias, so "
                f"refreshing the top rung installs an unrestrained configuration into a restrained "
                f"ladder -- the rung then samples neither ensemble, and nothing in the output says "
                f"so. Refused rather than combined: generate the reservoir under the same "
                f"restraints and it is a different file, or drop one of the two.")
        variables = resolved.get("collective_variables") or {}
        if not variables.get("file"):
            raise ConfigError(
                f"umbrella.file = {path!r} restrains collective variables, but "
                f"collective_variables.file is not set. A restraint NAMES a variable from that "
                f"file rather than defining one, so without it there is nothing to resolve the "
                f"restraint against.")
        return

    if protocol != "umbrella":
        if path:
            raise ConfigError(
                f"umbrella.file = {path!r} defines restraints, but protocol is {protocol}. A "
                f"restraint biases the dynamics, so it is never applied as a side effect of "
                f"another protocol -- either run `protocol: umbrella`, `protocol: REST2` or "
                f"`protocol: rREST2`, or remove the file.")
        return

    if not path:
        raise ConfigError(
            "protocol is umbrella but umbrella.file is not set. Umbrella sampling IS the "
            "restraint; without one this is a cMD run and should say so.")

    variables = resolved.get("collective_variables") or {}
    if not variables.get("file"):
        raise ConfigError(
            "protocol is umbrella but collective_variables.file is not set. A restraint names a "
            "collective variable from that file, and the same file drives the reported series -- "
            "so without it there is nothing to restrain and nothing to record.")
    if not int(variables.get("interval_steps") or 0):
        raise ConfigError(
            "protocol is umbrella but collective_variables.interval_steps is 0, which disables "
            "reporting. A window that biases a collective variable and never records it produces "
            "a trajectory nobody can reweight; the restraint and the series are the point.")


def _check_collective_variables(resolved: dict[str, Any]) -> None:
    """`file` and `interval_steps` are given together or not at all.

    Either one alone is a configuration that cannot be honoured, and both plausible readings of it
    are wrong. A file with no interval names torsions nobody asked to be measured; an interval
    with no file asks for observations of nothing. Choosing a default for the missing half would
    silently produce a run that reports on a schedule its author never wrote, or a run that
    quietly reports nothing at all while the configuration says otherwise -- and CV output
    failures are simulation failures, never a silent disabling.
    """
    block = resolved.get("collective_variables") or {}
    path = block.get("file")
    interval = int(block.get("interval_steps") or 0)
    if (path is None) == (interval == 0):
        return                                       # both off, or both on: a complete statement
    if path is None:
        raise ConfigError(
            f"collective_variables.interval_steps = {interval} asks for observations every "
            f"{interval} steps, but collective_variables.file is null, so there is nothing to "
            f"measure. Give a cv.yaml, or set interval_steps to 0 to disable reporting.")
    raise ConfigError(
        f"collective_variables.file = {path!r} names torsions to report, but "
        f"collective_variables.interval_steps is 0, which disables reporting. Set an interval, "
        f"or set file to null to disable reporting deliberately.")


MD_SCHEMA.checks = (_check_protocol, _check_timestep, _check_collective_variables,
                    _check_umbrella)


def _refuse_retired_platform(document: dict[str, Any]) -> None:
    """`dynamics.platform` moved to `machine.openmm.platform`. Say where, not just "unknown key".

    The generic unknown-key refusal would suggest a near-miss among the remaining dynamics fields,
    which is worse than useless: the key has not been misspelled, it has MOVED, and for a reason
    the message should give. A protocol is the same experiment wherever it runs; a workflow that
    carried `platform: CUDA` carried one machine's hardware into every repository it was shared
    through, and the next machine either had that platform or silently did not.
    """
    dynamics = document.get("dynamics")
    if not isinstance(dynamics, dict) or "platform" not in dynamics:
        return
    stated = dynamics["platform"]
    raise ConfigError(
        f"dynamics.platform is retired (this file sets it to {stated!r}). It is now "
        f"machine.openmm.platform.\n"
        f"  The OpenMM platform is a property of the MACHINE, not of the experiment, so it lives "
        f"in the user configuration:\n"
        f"\n"
        f"      machine:\n"
        f"        openmm:\n"
        f"          platform: CUDA        # or CPU, for a deliberate machine-wide CPU default\n"
        f"          precision: mixed\n"
        f"          device_policy: local_rank\n"
        f"\n"
        f"  at ${{XDG_CONFIG_HOME:-$HOME/.config}}/md-tools/user.config -- create it with "
        f"`md-openmm data-register --init`.\n"
        f"  For one run, `--cpu` overrides the machine default and is recorded as having been "
        f"asked for. Delete this key; the protocol is the same experiment on every machine.")


#: What a per-run `run.config` may set, and nothing else. `{section: {keys}}`.
#:
#: DELIBERATELY ONE KEY. The seed is the whole reason two repeats of one method on one system
#: differ -- `derive_seed` hashes it with each stage and replica name, so every stream in a run
#: descends from that number -- which is why it is the one thing that cannot live in the shared
#: `input/`. An override file that could set anything else would be a SECOND configuration
#: authority, and `md_tools.build.md` exists to be the only one. Widening this is not a
#: convenience: two files that can both set a step count is two answers waiting to disagree, and
#: the resolved document would record the winner without saying there had been a contest.
RUN_CONFIG_ALLOWED: dict[str, frozenset[str]] = {"dynamics": frozenset({"seed"})}


def load_run_config(path: Path | None) -> dict[str, Any]:
    """The per-run override document, validated against `RUN_CONFIG_ALLOWED`.

    An absent path is an empty override, not an error: a run that does not state a seed takes the
    schema's default, exactly as one generated before `run.config` existed does.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    document = load_yaml_strictly(path.read_text(encoding="utf-8"), source=str(path)) or {}
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: expected a mapping of sections, got "
                          f"{type(document).__name__}")
    for section, block in document.items():
        if section not in RUN_CONFIG_ALLOWED:
            allowed = ", ".join(f"{s}.{k}" for s, keys in RUN_CONFIG_ALLOWED.items()
                                for k in sorted(keys))
            raise ConfigError(
                f"{path}: `{section}` may not be set per run. A run.config carries only what is "
                f"genuinely per-run, which is {allowed} -- everything else belongs in the shared "
                f"input, where every repeat of this method reads the same value. An override that "
                f"could set anything would be a second configuration authority.")
        if not isinstance(block, dict):
            raise ConfigError(f"{path}: `{section}` must be a mapping, got "
                              f"{type(block).__name__}")
        for key in block:
            if key not in RUN_CONFIG_ALLOWED[section]:
                raise ConfigError(
                    f"{path}: `{section}.{key}` may not be set per run; only "
                    f"{', '.join(sorted(RUN_CONFIG_ALLOWED[section]))} may.")
    return document


def resolve_md_config(path: Path | None, *, run_config: Path | None = None) -> dict[str, Any]:
    """Resolve a configuration, optionally layered with a per-run override.

    `run_config` is the narrow per-run document (§`RUN_CONFIG_ALLOWED`): the shared `input/*.in`
    says what the method was asked to do, and this says which repeat it is. The two resolve to
    one `resolved.config`, which stays authoritative -- so the round trip a generated input
    promises is `input` + `run.config` -> `resolved.config`, not `input` alone.
    """
    document: dict[str, Any] = {}
    if path is not None:
        document = load_yaml_strictly(Path(path).read_text(encoding="utf-8"),
                                      source=str(path)) or {} if Path(path).is_file() else {}
        if not Path(path).is_file():
            raise ConfigError(f"{path}: no such configuration file")
    _refuse_retired_platform(document)

    # MERGED BEFORE `stated` IS TAKEN, not after. `stated` is what
    # `_apply_ais_reporting_defaults` consults to tell a value the user WROTE from one it
    # defaulted, so a seed supplied per run has to count as stated -- otherwise layering would
    # silently change which reporting intervals AIS considers user-chosen.
    override = load_run_config(run_config)
    for section, block in override.items():
        merged = dict(document.get(section) or {})
        merged.update(block)
        document[section] = merged

    stated = {name: set(block) for name, block in document.items() if isinstance(block, dict)}
    MD_SCHEMA.after_resolve = (lambda resolved: _apply_ais_reporting_defaults(resolved, stated),)
    try:
        return MD_SCHEMA.resolve(document)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}" if path is not None else str(exc)) from None
    finally:
        MD_SCHEMA.after_resolve = ()


def _apply_ais_reporting_defaults(resolved: dict[str, Any],
                                  stated: dict[str, set[str]]) -> None:
    """An AIS path is reported at its own cadence, not at a production run's.

    The `reporting.*` defaults -- 1000 and 10000 steps -- are sized for a 2.5 M-step cMD
    production. Applied to a switching path of a few hundred steps they are not merely coarse:
    they do not divide it, so no frame lands on the final step, and the divisibility rule below
    would refuse every short AIS run by construction.

    So for AIS, an interval the user did NOT state defaults to `observation_interval_steps` --
    the cadence the path is already observed at, which divides `switching_steps` because that is
    checked. An interval the user DID state is left alone and must divide, which is the whole
    point of the check.
    """
    if resolved.get("protocol") != "AIS":
        return
    cadence = int(resolved["ais"]["observation_interval_steps"])
    given = stated.get("reporting", set())
    for key in ("crd_printout_solute", "info_printout", "checkpoint_printout"):
        if key not in given:
            resolved["reporting"][key] = cadence


# ---------------------------------------------------------------------------------------------
# the stage plan
# ---------------------------------------------------------------------------------------------

def stage_plan(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """The ordered stages, each fully resolved, each carrying the name it is FILED under.

    A STAGE HAS TWO NAMES AND THEY ARE DIFFERENT THINGS. It is GENERATED as `eq_nvt_posres` /
    `eq_npt_posres` / `eq_npt_free` -- renamed to NVT spellings under implicit solvent or a scaled
    run, so a pressure-coupled name never appears on a boxless one -- and it is FILED as `eq_1` /
    `eq_2` / `eq_3`, by position in the chain. The stage name decides the physics; `file_key`
    decides the filename.

    THE KEY IS STAMPED HERE because this is the one function every surface goes through --
    `build-md`, `md-run` and `load_generated_plan` for a generated script -- and the plan is
    RECOMPUTED from `resolved.config` at every execution rather than persisted, so an older
    generated tree gets the key too.

    It used to be computed in `_stage_targets` alone, which is the layout's side. The runtime named
    its outputs from the stage instead, so `run.sh` chained `-c eq/eq_1.xml` while the stage wrote
    `eq/eq_nvt_posres.xml` and NO cMD or REST2 chain could complete through `run.sh` -- the
    documented way to run one. Two namers for one file is the defect; this is the single authority.
    """
    return _with_file_keys(_stage_plan_entries(resolved))


def _with_file_keys(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp each entry with the name its artefacts are filed under.

    Positional for the equilibration stages, because that is what the key means -- `eq_2` is the
    second equilibration stage of this chain, whatever ensemble it ended up with. Every other
    stage is filed under its own name: `min` and `cMD` already agree, which is why they were never
    affected by the divergence this removes.
    """
    from ..layout import eq_stage_key

    order = 0
    stamped: list[dict[str, Any]] = []
    for stage in plan:
        name = str(stage["name"])
        if name.startswith("eq_"):
            order += 1
            key = eq_stage_key(order)
        else:
            key = name
        stamped.append(dict(stage, file_key=key))
    return stamped


def _stage_plan_entries(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """The ordered stages, each fully resolved.

    Under implicit solvent the two pressure-coupled stages become NVT and are RENAMED, so nothing
    downstream can read `eq_npt_free` and believe a barostat was involved.
    """
    dyn, stages, rep = resolved["dynamics"], resolved["stages"], resolved["reporting"]
    implicit = resolved["solvent"] == "implicit"

    # TWO reasons a workflow has no pressure-coupled stage, and they are different facts:
    #
    #   implicit solvent -- there is no box, so there is no volume to control;
    #   a scaled run (tau > 0) -- it samples the fixed-volume ensemble of the ladder rung it sits
    #     at, and a barostat would sample a different distribution.
    #
    # The second used to be applied to the PRODUCTION stage only. The equilibration stages before
    # it stayed NPT, and `stage_main` -- which enforces the same rule -- refused them at run time,
    # four stages and several minutes of GPU in. A rule that build-md knows at generation time
    # must be applied at generation time.
    scaled = float(dyn["tau"]) > 0.0
    fixed_volume = implicit or scaled
    why = ("implicit solvent has no box, so there is no volume to equilibrate" if implicit else
           f"this run is scaled (tau = {dyn['tau']}) and samples the fixed-volume ensemble of "
           f"the ladder rung it sits at")
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
        # In `common`, so EVERY dynamics stage reports the same collective variables on the same
        # cadence. Unlike phase space -- which is production-only because a reservoir must sample
        # the ladder's own distribution and the equilibration stages are restrained and not yet at
        # equilibrium -- a CV is an observation of whatever the stage is actually doing, and
        # watching a torsion relax through equilibration is a legitimate thing to want. The
        # minimisation stage still produces no series: it has `steps == 0`, and the schedule
        # refuses to invent a time axis for iterations that have no timestep.
        "collective_variables": dict(resolved.get("collective_variables") or {}),
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
    if implicit and _equilibrates_per_tau(resolved):
        # The equilibration stages belong to the RUNGS now, each under its own tau -- see
        # `per_tau_equilibration_stages`. Implicit solvent has no box to fix first, so the tau = 0
        # chain is minimisation alone and the ladder starts from `min.xml`. (Explicit solvent
        # keeps its chain: the NPT stages at tau = 0 decide the volume every rung shares.)
        return plan
    plan.append({**common, "name": "eq_nvt_posres", "ensemble": "NVT",
                 "steps": stages["restrained_nvt_steps"],
                 "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                 "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                 "state_interval_steps": rep["info_printout"],
                 "checkpoint_interval_steps": rep["checkpoint_printout"],
                 "description": "Restrained NVT: settle the solvent around a held solute."})
    if fixed_volume:
        plan.append({**common, "name": "eq_nvt_posres_2", "ensemble": "NVT",
                     "steps": stages["restrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Second restrained NVT stage. This REPLACES the restrained "
                                    f"NPT stage of an unscaled explicit-solvent run: {why}."})
        plan.append({**common, "name": "eq_nvt_free", "ensemble": "NVT",
                     "steps": stages["unrestrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Unrestrained NVT. This REPLACES the unrestrained NPT stage "
                                    f"of an unscaled explicit-solvent run: {why}."})
    else:
        plan.append({**common, "name": "eq_npt_posres", "ensemble": "NPT",
                     "steps": stages["restrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": dyn["restraint_kcal_per_mol_A2"],
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Restrained NPT: equilibrate the density with the solute held."})
        plan.append({**common, "name": "eq_npt_free", "ensemble": "NPT",
                     "steps": stages["unrestrained_npt_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Unrestrained NPT, the last stage before production."})

    if resolved["protocol"] == "umbrella":
        # A cMD production stage that carries biases. Everything else about it -- the
        # equilibration chain before it, checkpointing, CV reporting -- is cMD's, deliberately:
        # an umbrella window IS conventional dynamics with a restraint on top, and giving it its
        # own stage machinery would mean maintaining two of everything to no benefit.
        plan.append({**common, "name": "umbrella",
                     "ensemble": "NVT" if fixed_volume else "NPT",
                     "steps": stages["production_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "phase_space_interval_steps": dyn["phase_space_printout"],
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "umbrella_file": resolved["umbrella"]["file"],
                     "description": "Umbrella sampling: biased production with the restrained "
                                    "collective variables reported."})
    if resolved["protocol"] == "cMD":
        plan.append({**common, "name": "cMD",
                     "ensemble": "NVT" if fixed_volume else "NPT",
                     "steps": stages["production_steps"],
                     "restraint_kcal_per_mol_A2": 0.0,
                     "phase_space_interval_steps": dyn["phase_space_printout"],
                     "trajectory_interval_steps": rep["crd_printout_solute"],
                 "whole_interval_steps": rep["crd_printout_whole"],
                     "state_interval_steps": rep["info_printout"],
                     "checkpoint_interval_steps": rep["checkpoint_printout"],
                     "description": "Production molecular dynamics."})
    return plan


#: The stages `rest2.equilibration_per_tau` runs on every rung, in order. Pinned by a test to
#: `md_tools.remd.rung_equilibration.PER_TAU_STAGE_NAMES`, which runs them; spelled here too so
#: resolving a configuration does not import the ladder runtime.
PER_TAU_STAGE_NAMES = ("eq_nvt_posres", "eq_nvt_posres_2", "eq_nvt_free")


def _equilibrates_per_tau(resolved: dict[str, Any]) -> bool:
    return (resolved.get("protocol") in ("REST2", "rREST2")
            and bool((resolved.get("rest2") or {}).get("equilibration_per_tau")))


def per_tau_equilibration_stages(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """The stages every rung runs under its own tau, or [] when the setting is off.

    Taken from `stage_plan` itself, for the fixed-volume chain -- the renamed stages a scaled or an
    implicit run gets -- so "the same equilibration" is the same stage dicts rather than a second
    description of them. A stage of 0 steps is left out, as the stage chain would run nothing.
    """
    if not _equilibrates_per_tau(resolved):
        return []
    fixed_volume = dict(resolved, solvent="implicit",
                        rest2=dict(resolved["rest2"], equilibration_per_tau=False))
    return [{"name": stage["name"], "steps": int(stage["steps"]),
             "restraint_kcal_per_mol_A2": float(stage["restraint_kcal_per_mol_A2"])}
            for stage in stage_plan(fixed_volume)
            if stage["name"] in PER_TAU_STAGE_NAMES and int(stage["steps"]) > 0]


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


# ---------------------------------------------------------------------------------------------
# The Amber-like `.in` files, for `md-openmm md-run`
#
# Generated from the RESOLVED document, exhaustively: every key `md_tools.run.inputs` knows how to
# read is written out with the value this run resolved to. That is what makes the pair honest --
# each `.in` resolves back to exactly the `resolved.config` beside it, and a test asserts it rather
# than the comment claiming it.
#
# `resolved.config` remains authoritative. The `.in` files exist because they are what a person
# edits: short, sectioned, and readable by anyone who has written an Amber mdin.
# ---------------------------------------------------------------------------------------------

#: Which sections a protocol's inputs carry, and which resolved blocks feed each one. Overlapping
#: keys (timestep_fs, temperature_K and the reporting intervals are readable in &cntrl and &AIS
#: alike) are emitted in &cntrl only, so an input never states one setting twice.
#: `collective_variables` rides in &cntrl for every protocol: the cadence and the definition file
#: are properties of the run, not of the exchange ladder or the switching schedule, and every
#: protocol reports them the same way.
_IN_SECTIONS = {
    "cMD": (("cntrl", ("", "dynamics", "stages", "reporting", "collective_variables")),),
    # `umbrella` rides in &cntrl for a ladder too: a ladder may carry the same torsion restraints
    # on every rung, and an `.in` that dropped the file would run unrestrained while
    # `resolved.config` beside it said otherwise.
    "REST2": (("cntrl", ("", "dynamics", "stages", "reporting", "collective_variables",
                         "umbrella")),
              ("remd", ("rest2", "reservoir"))),
    "rREST2": (("cntrl", ("", "dynamics", "stages", "reporting", "collective_variables",
                          "umbrella")),
               ("remd", ("rest2", "reservoir"))),
    "AIS": (("cntrl", ("", "dynamics", "reporting", "collective_variables")),
            ("AIS", ("ais", "ais_source"))),
    # Umbrella is cMD with biases, so it writes cMD's block and adds the restraint file.
    "umbrella": (("cntrl", ("", "dynamics", "stages", "reporting", "collective_variables",
                            "umbrella")),),
}


def _in_value(value: Any) -> str | None:
    """One resolved value as an input file writes it, or None when it is not written at all."""
    if value is None:
        return None                       # a key with no value is absent, never `null`
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # `repr`, so a float stays a float. `%g` would write 300.0 as `300`, the parser would read
        # an integer back, and the input would no longer resolve to the run it was generated from.
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


#: Resolved fields that are NOT written into a generated `.in`, because the input is SHARED.
#:
#: `input/` sits at the dataset root and every repeat of a method reads the same files -- which
#: the reference data bears out exactly: the three ALA-explicit REST2 runs' inputs differ by one
#: line each, `random_seed = 700501` against `700502`, and nothing else. So the seed is the one
#: thing that cannot be in there, and it lives in the per-run `run.config` instead
#: (`RUN_CONFIG_ALLOWED`).
#:
#: It stays ACCEPTED by `SECTION_KEYS`: nine migrated reference runs carry `random_seed` in their
#: inputs, and `md-run` must keep reading them. Emission and acceptance are different questions,
#: and conflating them would make every existing input unparseable.
_NOT_IN_INPUT = frozenset({"dynamics.seed"})


#: Suppressed from a PREPARATION input (`min.in`, `eq_<k>.in`), which is shared across METHODS.
#:
#: `input/` sits at the dataset root, and minimisation and equilibration are the same physics
#: whichever method follows them -- so `min.in` is meant to be one file that every method reads.
#: It could not be: the generated text carried `protocol = cMD` against `protocol = REST2` plus an
#: entire `&remd` block, so the bytes differed by method and the second method on a system was
#: refused against the first method's file.
#:
#: What is dropped is METHOD IDENTITY, never physics. Every setting that decides what the stage
#: does -- the timestep, the temperature, the restraint, tau, the stage lengths, the reporting
#: cadences -- is still written, so a preparation input still resolves to the run it prepares.
#:
#: This is why it is safe: `protocol` is DEFAULTED rather than required (`Field("protocol", str,
#: default="cMD")`), and `md-run` dispatches on the STAGE a named input asks for rather than on
#: the protocol -- "`stage` decides, not `protocol`". An input naming `min` reaches `stage_main`
#: whether or not a protocol is written in it.
#: `umbrella.file` is here and the CV keys deliberately are NOT, and the difference is what each
#: one does to a preparation stage. `stage_plan` attaches `umbrella_file` to the umbrella
#: PRODUCTION stage alone, so writing it into `min.in` describes a bias that stage never applies
#: -- spurious text that made the file method-specific for nothing. Collective variables are the
#: opposite: they are in `common`, so every dynamics stage reports them on the same cadence, and
#: watching a torsion relax through equilibration is a legitimate thing to ask for. A cMD run that
#: reports CVs and a REST2 run that does not therefore have genuinely different `eq_1.in` files,
#: and that collision is a real disagreement for `_write_shared_input` to refuse rather than
#: something to paper over by dropping the setting.
#: `collective_variables.*` joins them for the MINIMISATION alone -- see `_preparation_suppressed`.
_NOT_IN_PREPARATION_INPUT = frozenset({"protocol", "umbrella.file"})

#: Additionally suppressed from `min.in`, and from nothing else.
#:
#: Minimisation produces NO collective-variable series at all: its iterations have no timestep,
#: and a `time_ps` for them would be a fiction, so the schedule refuses to invent a time axis for
#: them. The cadence therefore cannot change what a minimisation does -- but it was still written
#: into the shared `min.in`, so two runs on one system differing only in whether they report CVs
#: had different `min.in` bytes and the second was refused against the first. That refusal
#: arrives from the shared-input gate BEFORE the `min/resolved.config` comparison that already
#: exempts the same keys, which is why exempting them there alone changed nothing.
#:
#: The equilibration inputs keep theirs: those stages DO integrate, and watching a torsion relax
#: through equilibration is a legitimate thing to ask for.
_NOT_IN_MINIMISATION_INPUT = frozenset({"collective_variables.file",
                                        "collective_variables.interval_steps"})


def _preparation_suppressed(stage: str | None) -> frozenset:
    """Targets a shared preparation input must not carry, for this stage."""
    if stage == "min":
        return _NOT_IN_PREPARATION_INPUT | _NOT_IN_MINIMISATION_INPUT
    return _NOT_IN_PREPARATION_INPUT


def in_file_text(resolved: dict[str, Any], *, stage: str | None = None,
                 heading: str = "", preparation: bool = False) -> str:
    """One `.in` file for this resolved workflow, optionally naming one stage of it.

    `preparation` writes the SHARED form used for `min.in` and `eq_<k>.in`: `&cntrl` alone, with
    no method identity in it, so one file serves every method on the system. See
    `_NOT_IN_PREPARATION_INPUT`.
    """
    from ..run.inputs import SECTION_KEYS

    protocol = resolved["protocol"]
    suppressed = _NOT_IN_INPUT | (_preparation_suppressed(stage) if preparation else frozenset())
    lines = [f"! {heading or protocol}",
             "!",
             "! Generated by `md-openmm build-md`. Run it with:",
             "!",
             "!     md-openmm md-run -i THIS_FILE -p ../built.pdb -s ../built.xml",
             "!",
             "! `resolved.config` beside this file is AUTHORITATIVE: md-run resolves this input",
             "! and writes the result there, and that resolved document is what the run reads.",
             "! This file resolves to exactly that document -- editing it changes the run.",
             ""]
    sections = _IN_SECTIONS[protocol]
    if preparation:
        # &cntrl ONLY. `&remd`, `&AIS` and the reservoir block describe the method that follows
        # this stage, not the stage, and carrying them made the same minimisation two different
        # files.
        sections = tuple((section, blocks) for section, blocks in sections
                         if section == "cntrl")
    for section, blocks in sections:
        known = SECTION_KEYS[section]
        body: list[str] = []
        if section == "cntrl" and stage is not None:
            body.append(("stage", stage))
        for block in blocks:
            source = resolved if block == "" else resolved.get(block) or {}
            for key, target in known.items():
                # Matched by the TARGET this key maps onto, not by reconstructing the target from
                # the key: several inputs are deliberately spelled differently from the resolved
                # field they set (`random_seed` -> `dynamics.seed`, `reservoir_enabled` ->
                # `reservoir.enabled`), and reconstruction silently dropped every one of them.
                if target.startswith("_"):
                    continue
                # NOT WRITTEN INTO A SHARED INPUT. Keyed by target rather than by key so both
                # spellings are covered at once -- `random_seed` is accepted in &cntrl AND in
                # &AIS, deliberately, so an AIS input reads as one block.
                if target in suppressed:
                    continue
                where, _, leaf = target.rpartition(".")
                if where != block:
                    continue
                written = _in_value(source.get(leaf))
                if written is not None:
                    body.append((key, written))
        if not body:
            continue
        width = max(len(name) for name, _ in body)
        lines.append(f"&{section}")
        lines += [f"  {name:<{width}} = {value}," for name, value in body]
        lines += ["/", ""]
    return "\n".join(lines)


def _stage_targets(plan: list[dict[str, Any]], *, run, dataset) -> dict[str, dict[str, Any]]:
    """Where each stage's `.in`, `.py` and output directory go under the run layout.

    THE TWO NAMES OF A STAGE ARE DIFFERENT THINGS, and this is where they are kept apart. A stage
    is GENERATED as `eq_nvt_posres` / `eq_npt_posres` / `eq_npt_free` -- renamed to NVT spellings
    under implicit solvent or a scaled run, precisely so a pressure-coupled name never appears on
    a boxless one -- and it is FILED as `eq_1` / `eq_2` / `eq_3`, by position in the chain. The
    stage name still decides the physics and still reaches the runtime; the layout name decides
    the filename. `eq_stage_key` owns the second, the plan owns the first, and the ensemble that
    can no longer live in the filename lives in each stage's `.out` header and resolved
    configuration, which state it explicitly.

    `min` and every `.in` are DATASET paths, not run paths: minimisation draws no velocities and
    has no seeded stochastic element, so every run on one system minimises to the same structure,
    and an input says what a method was asked to do rather than which repeat this is.
    """
    targets: dict[str, dict[str, Any]] = {}
    for stage in plan:
        name = stage["name"]
        # THE KEY COMES FROM THE PLAN, which is the single authority for it (`stage_plan` stamps
        # it). This function used to recount the equilibration positions itself while the runtime
        # named its outputs from the stage -- two namers for one file, and the reason `run.sh`
        # chained a restart no stage ever wrote.
        key = str(stage.get("file_key") or name)
        if name == "min":
            odir, script = dataset.min, dataset.min / "min.py"
        elif name.startswith("eq_"):
            odir, script = run.eq, run.eq / f"{key}.py"
        else:
            odir, script = run.root, run.root / f"{name}.py"
        targets[name] = {
            "key": key,
            "input": dataset.stage_input(key),
            "script": script,
            "odir": odir,
            # Relative to the RUN ROOT, which is where run.sh cds to. A generated script carries
            # no absolute path, and these are what it types.
            "input_rel": os.path.relpath(dataset.stage_input(key), run.root),
            "odir_rel": os.path.relpath(odir, run.root),
            "restart_rel": os.path.relpath(odir / f"{key}.xml", run.root),
        }
    return targets


def _write_min_directory(directory: Path, document: dict[str, Any], *, resolved: dict[str, Any],
                         why: str, overwrite: bool, log, note, run) -> None:
    """The shared `min/`: its declaration, and its OWN seed beside it.

    FIRST WRITER WINS, and the seed is exempt from the comparison. `min/` is shared, so a second
    run must not replace what the first one minimised under -- but two runs on one system are
    *expected* to differ in their seed, and the minimisation's own seed is not a run's. So an
    existing directory is kept when it agrees about everything EXCEPT `dynamics.seed`, and refused
    when it disagrees about anything else: a different `minimization_iterations` or restraint
    really is a different minimisation, and silently reusing it would make two runs claim a
    starting structure neither produced.
    """
    directory.mkdir(parents=True, exist_ok=True)
    resolved_path, run_config_path = directory / "resolved.config", directory / "run.config"
    seed = resolved["dynamics"]["seed"]

    if resolved_path.is_file() and not overwrite:
        existing = load_yaml_strictly(resolved_path.read_text(encoding="utf-8"),
                                      source=str(resolved_path))
        mine, theirs = dict(document), dict(existing)
        # Compared with the seed removed from BOTH sides, so "everything except the seed" is
        # exactly what is checked rather than approximately.
        for side in (mine, theirs):
            side["dynamics"] = {k: v for k, v in (side.get("dynamics") or {}).items()
                               if k != "seed"}
            # AND COLLECTIVE VARIABLES, for the same reason as the seed: they cannot change what
            # a minimisation does. CV reporting is in `common`, so it reaches every stage dict --
            # but minimisation produces NO series at all, because its iterations have no timestep
            # and a `time_ps` for them would be a fiction. Two runs on one system that differ
            # only in whether they report CVs therefore minimise identically, and refusing the
            # second one would make a reporting choice look like a different starting structure.
            side.pop("collective_variables", None)
        if mine != theirs:
            differing = sorted(
                key for key in set(mine) | set(theirs) if mine.get(key) != theirs.get(key))
            raise ConfigError(
                f"{resolved_path} already exists and describes a different minimisation.\n"
                f"  Differing: {', '.join(differing)}\n"
                f"  `min/` is SHARED by every run on this system -- the runs already beside it "
                f"started from the structure this file produced, so replacing it would make them "
                f"claim a starting point none of them minimised.\n"
                f"  The seed is deliberately exempt and is not the problem here: two runs on one "
                f"system are expected to differ in it, and the minimisation keeps its own.\n"
                f"  Either generate from the configuration those runs used, or give this a "
                f"different `<system>` root.")
        # RECORDED EVEN THOUGH IT WAS NOT WRITTEN. A kept file is still a file this run depends
        # on, and leaving it out of the Outputs section would make the shared minimisation
        # invisible in the log of every run after the first.
        note(resolved_path)
        if run_config_path.is_file():
            note(run_config_path)
        log.field(f"{os.path.relpath(directory, run.root)}/resolved.config",
                  "kept: already minimised, under its own seed")
        return

    text = ("# The configuration these scripts were generated from, fully resolved.\n"
            "# Every default is written out, so this file alone reproduces the generation.\n"
            + yaml.safe_dump(document, sort_keys=False, default_flow_style=False))
    resolved_path.write_text(text, encoding="utf-8")
    note(resolved_path)
    log.field(f"{os.path.relpath(directory, run.root)}/resolved.config", why)

    # THE MINIMISATION'S OWN SEED. Written here so `md-run -i ../input/min.in -odir ../min`
    # layers it exactly as a run layers its own, and the declaration beside the script therefore
    # round-trips instead of refusing.
    run_config_path.write_text(
        "# The minimisation's own seed. `min/` is SHARED by every run on this system, and this\n"
        "# is deliberately NOT any run's seed: minimisation draws no velocities and has no seeded\n"
        "# stochastic element, so no run's sampling descends from it. Every run on this system may\n"
        "# carry a different `dynamics.seed` in its own run.config without disturbing this one.\n"
        + yaml.safe_dump({"dynamics": {"seed": seed}}, sort_keys=False), encoding="utf-8")
    note(run_config_path)


def _method_neutral(resolved: dict[str, Any]) -> dict[str, Any]:
    """`resolved` with the method identity removed, for a directory fed by a SHARED input.

    `protocol` is DEFAULTED rather than required, and `md-run` dispatches on the stage a named
    input asks for -- so a preparation stage resolves and runs identically without it. Both
    `min/` and `eq/` hold scripts driven by `input/*.in`, which carry no protocol, so both need
    the same projection or the stored document and the input can never agree.
    """
    neutral = dict(resolved)
    neutral["protocol"] = MD_SCHEMA.fields["protocol"].default
    for block in ("rest2", "reservoir", "ais", "ais_source", "umbrella"):
        if block in neutral:
            neutral[block] = _schema_defaults(block)
    return neutral


def _script_directories(plan: list[dict[str, Any]], *, run, dataset,
                        resolved: dict[str, Any]) -> list[tuple[Path, dict[str, Any], str, bool]]:
    """Each directory that receives a generated script, with the document its scripts must read.

    TWO DIRECTORIES ARE NOT THE RUN ROOT, and they need different documents for different
    reasons.

    `<system>/min/` is SHARED by every run on this system, so it cannot hold a per-run document.
    It gets the METHOD-NEUTRAL one -- the same projection `input/min.in` is written from, with no
    protocol and no method block -- which makes it identical whichever method generates it first,
    and makes the `min.in` <-> `min/resolved.config` round trip hold by construction rather than
    by coincidence. Minimisation draws no velocities and has no seeded stochastic element, so
    there is nothing per-run in it to lose.

    `<run>/eq/` is per run, so it gets the run's own document -- THE SAME BYTES as
    `<run>/resolved.config`. That is a second copy inside one run, stated plainly rather than
    left to be discovered: it is not a second authority, because both are written here from one
    resolved document in one pass and the stage fingerprint binds whichever one a stage actually
    read. Equilibration is per run precisely because it draws Maxwell velocities from this run's
    seed, which is what makes two repeats diverge.
    """
    from ..layout import eq_stage_key

    directories: list[tuple[Path, dict[str, Any], str]] = []
    names = [stage["name"] for stage in plan]
    if "min" in names:
        directories.append(
            (dataset.min, _method_neutral(resolved),
             "SHARED by every run on this system, so it carries no method identity: "
             "minimisation is the same physics whichever method follows it.", True))
    if any(name.startswith("eq_") for name in names):
        directories.append(
            (run.eq, _method_neutral(resolved),
             "METHOD-NEUTRAL, exactly as `min/` is, because the stages here read the SHARED "
             "`input/eq_<k>.in` -- which carries no protocol, so that one file can serve every "
             "method. A per-run document here recorded `protocol: REST2` while the input it is "
             "compared against resolves to the default, and every preparation stage of a ladder "
             "refused with `protocol: was 'REST2', now 'cMD'`. The two can never agree, so the "
             "stored document matches the input rather than the run. Which method the run IS "
             "stays recorded at the run root and in build-md.log.", False))
    return directories


def _schema_defaults(block: str) -> dict[str, Any]:
    """Every field of one schema section at its default, for the method-neutral projection.

    `Schema.sections` and `Section.fields` are both dicts keyed by name, so the defaults come
    from the schema itself rather than from a hand-written list that would silently go stale the
    next time a method gains a field.
    """
    return {name: field.default for name, field in MD_SCHEMA.sections[block].fields.items()}


def _place_definitions(directory: Path, copies: list[tuple[str, bytes]], *,
                       overwrite: bool = False, note=None) -> None:
    """Put every content-addressed definition copy beside a declaration that names it.

    A `resolved.config` names its CV and umbrella definitions by BARE NAME, and the runtime
    resolves that name beside the declaration it read -- `resolved_config_beside(script)` for a
    generated script, `-odir` or the input's directory for `md-run`. A directory holding a
    declaration but not the definition is therefore a declaration that cannot be resolved, which
    is what `input/` was: it named `cv.<digest>.yaml` and held none, so the public command could
    not load the definition from anywhere and the read-only continuation boundary returned early
    instead of refusing.

    IDEMPOTENT AND IDENTITY-CHECKED, like `input/` itself. The name carries the digest, so an
    existing copy of the same name must be the same bytes; if it is not, the digest is not what
    it claims and the difference is reported rather than overwritten.

    `--overwrite` REPLACES IT, as it does every other file in the shared tree. Refusing even then
    would make this the one writer in `build_scripts` that `--overwrite` cannot get past, so a
    single corrupt copy would block regenerating the run for good -- and a name whose bytes do not
    hash to it is exactly the damage `--overwrite` exists to clear.
    """
    if not copies:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in copies:
        path = directory / name
        if path.is_file():
            if path.read_bytes() != payload and not overwrite:
                raise ConfigError(
                    f"{path} already exists and holds different bytes than the definition this "
                    f"configuration resolves to.\n"
                    f"  The name is content-addressed, so two different definitions cannot "
                    f"honestly share it: one of the two digests does not describe its file.\n"
                    f"  Pass --overwrite to replace it.")
            if path.read_bytes() == payload:
                continue
        path.write_bytes(payload)
        if note is not None:
            note(path)


def _write_shared_input(path: Path, text: str, *, overwrite: bool) -> None:
    """Write one shared `input/*.in`, or refuse one that exists and says something different.

    THE SHARING IS A CLAIM, so it is enforced rather than assumed. `input/` sits at the dataset
    root and every repeat of a method on one system reads the same files -- which is exactly why
    a second run generated from an edited configuration must not quietly replace what run 1
    actually read. An identical file is kept (generation is then idempotent); a different one is
    refused by name, because the alternative is that run 1's inputs describe run 2.

    If two methods on one system genuinely need different equilibration, they are not comparable
    and belong under a different `<system>` -- a refusal, not a subdirectory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
        if not overwrite:
            raise ConfigError(
                f"{path} already exists and is not what this configuration resolves to.\n"
                f"  `input/` is shared by every run on this system: the runs already beside it "
                f"read THIS file, so replacing it would make their inputs describe a different "
                f"experiment than the one they ran.\n"
                f"  Either generate from the configuration those runs used, or -- if this really "
                f"is a different protocol -- give it a different `<system>` root. "
                f"Pass --overwrite only if no run beside it has output yet.")
    path.write_text(text, encoding="utf-8")


def _group_file_text(*, protocol: str, states: int, segment: int, segments: int,
                     taus: list[float], targets: dict[str, dict[str, Any]],
                     run, dataset, solute_yaml: Path | None) -> str:
    """One group file per segment: one line per STATE, each naming its own rung Hamiltonian.

    `-s` IS PER LINE NOW, and that is the architecture change. Scaling used to happen at run time
    from one shared `built.xml` with tau derived from `--group-index`; each rung is serialised to
    `remd<n>/build_state<n>.xml` at build time instead, so the Hamiltonian a state ran under is a
    file that can be read rather than a derivation that has to be trusted. The executor's group
    parser was changed to match: `system` left `HOMOGENEOUS_GROUP_FIELDS`, and a stronger check
    took its place -- line *i* must name rung *i*, and no two lines may name one file.

    Paths are relative to THIS FILE, which is how the parser reads them and how everyone who
    writes one by hand expects them to read.

    `-c` appears on the FIRST segment only. A later segment takes its physical state -- positions,
    velocities, box, the state-to-walker map, the RNG streams -- from the parent segment's
    checkpoint through `--extend-from`, and naming a coordinate file as well would give it two
    starting states that nothing downstream reconciles.
    """
    last = [stage for stage in targets.values()][-1] if targets else None
    start = last["restart_rel"] if last else None
    lines = [f"# {protocol}: {states} states, tau {taus[0]} to {taus[-1]}.",
             f"# Segment {segment} of {segments}; this segment's outputs are the `_prod{segment}` "
             f"set.",
             "# One group per line, INPUTS only -- run-level outputs go on the executor call,",
             "# because they describe the coordinated run rather than one replica.",
             "# Paths are relative to THIS FILE, which is how they are read back.",
             "#",
             "# -s is this state's own pre-scaled rung, written by `build-md` and recorded in",
             "# build_states.log beside it. It is NOT scaled again at run time: doing so would",
             "# take solute-solute to (1-tau)^4 and produce entirely plausible numbers.",
             ""]
    for index in range(states):
        # `-i` IS THE PROTOCOL MODULE, not the Amber-like input.
        #
        # `remd.executor.run_grouped` does `load_grouped_protocol(groups[0]["input"])`, which
        # imports that path as PYTHON and expects one `protocol` object: a group file's `-i` names
        # the module that describes the ladder, and the executor is its only reader. This wrote
        # `../input/REST2.in` instead, so every grouped launch through `run.sh` died with
        #
        #   RuntimeError: .../input/REST2.in could not be loaded as a Python file
        #
        # after the eq chain had completed and the run directory looked like a started run. The
        # runtime's own group-file writer (`remd.generated._group_file_text`, used when no
        # `--groupfile` is supplied) has always written `_protocol.py`; the two writers disagreed
        # and only the build-time one was wrong.
        #
        # `build-md` cannot write `_protocol.py` -- it is content-addressed from the resolved
        # ladder and published by `replica_main` into `-odir` before the executor is called -- but
        # naming it here is still correct: the executor reads the group file at RUN time, by which
        # point the file is beside it. `--check` never reached this, because a preflight does not
        # load the protocol module.
        parts = [f"-i {os.path.relpath(run.root / '_protocol.py', run.root)}",
                 f"-p {os.path.relpath(dataset.built('pdb'), run.root)}",
                 f"-s {os.path.relpath(run.state_system(index), run.root)}"]
        if segment == 1 and start:
            parts.append(f"-c {start}")
        if solute_yaml is not None:
            parts.append(f"--solute {os.path.relpath(solute_yaml, run.root)}")
        parts.append(f"--group-index {index}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _run_sh(plan: list[dict[str, Any]], *, all_in_one: bool, protocol: str,
            resolved: dict[str, Any], targets: dict[str, dict[str, Any]],
            segments: int = 1) -> str:
    """The runnable form of this protocol, driven by `md-openmm md-run`.

    It calls the installed command rather than `python min.py` because that is the interface a
    person types by hand, and a script that used a different one would be a second way to run the
    same thing that could drift from the documented way. The `.py` entry points remain and do the
    same work -- `python min.py` and `md-openmm md-run -i min.in` reach the same function.
    """
    states = int((resolved.get("rest2") or {}).get("number_of_replicas") or 0)
    lines = ['#!/usr/bin/env bash',
             '# Run this protocol, in order.',
             '#',
             '# Generated by `md-openmm build-md`. Every path is explicit and relative to this',
             '# directory, so the whole directory can be moved. A stage that already reports',
             '# completion in its own machine record is skipped rather than silently rerun; an',
             '# interrupted stage resumes from its checkpoint only if the checkpoint matches the',
             '# configuration that produced it.',
             '#',
             '# Each stage is one `md-openmm md-run -i <stage>.in`. The .in file is the input; ',
             '# `resolved.config` beside it is authoritative, and md-run records the .in digest',
             '# it resolved so the two can always be matched afterwards.',
             '#',
             '# CUDA is the default and is mandatory. Pass --cpu through "$@" for a CPU run.',
             '#',
             '#   -s is the serialised System (built.xml) and -x the output trajectory, as in',
             '#   Amber. -o is what you read while a run is going; -log is the provenance record.',
             '#',
             '#   ./run.sh                 run from the built system in the parent directory',
             '#   ./run.sh ../built.pdb ../built.xml    or name them explicitly',
             'set -euo pipefail',
             '',
             'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
             '# The dataset root is one level up: build/, min/ and input/ are SHARED by every run',
             '# on this system, and this run reads them rather than carrying its own copies.',
             'TOPOLOGY="${1:-${HERE}/../build/built.pdb}"',
             'SYSTEM="${2:-${HERE}/../build/built.xml}"',
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
            '  echo "Give it a finished fixed-tau run, for example ../hot/cMD.nc." >&2',
            '  exit 2',
            'fi',
            'if [[ ! -f "${SOURCE}" ]]; then',
            '  echo "run.sh: no source trajectory at ${SOURCE}" >&2; exit 2',
            'fi',
            '',
            '# Paths are independent, so this parallelises by simply giving each worker its own:',
            '#   NPROC=8 ./run.sh ../built.pdb ../built.xml ../hot/cMD.nc',
            '# Which global path owns which AIS_trajNNNN.nc does not depend on NPROC.',
            'NPROC="${NPROC:-1}"',
            'LAUNCH=()',
            'if [[ "${NPROC}" -gt 1 ]]; then LAUNCH=(mpirun -n "${NPROC}"); fi',
            '',
            'echo "== AIS =="',
            # THE SHARED INPUT. `build-md` writes the AIS input to `<system>/input/AIS.in` -- an
            # input says what a method was asked to do, which is a property of the system rather
            # than of one repeat -- so an AIS run directory holds no `.in` of its own. This typed
            # a bare `AIS.in` and every `./run.sh` on a generated AIS tree died with
            # `md-run: -i AIS.in: no such run input file`.
            # `../input/AIS.in` as a literal: this function is handed the plan, the protocol and
            # the stage targets, and an AIS plan is deliberately EMPTY -- a switching campaign has
            # no preparation chain -- so `targets` carries no AIS entry to take the path from. The
            # shared input is `dataset.stage_input("AIS")`, one level up from the run directory
            # this script sits in, which is what that resolves to for every protocol.
            '"${LAUNCH[@]}" md-openmm md-run -i ../input/AIS.in \\',
            '  -p "${TOPOLOGY}" -s "${SYSTEM}" -source-traj "${SOURCE}" \\',
            '  -o AIS.out -log AIS.log "$@"',
            '']
    elif all_in_one:
        # THE SHARED INPUT, AND THE PROTOCOL'S OWN.
        #
        # An all-in-one tree holds no `.in` file at all: the inputs belong to the SYSTEM and sit in
        # `../input/`, exactly as they do for a split run. This typed a bare `cMD.in`, so
        # `./run.sh` on an all-in-one directory died with `md-run: -i cMD.in: no such run input
        # file` -- and it named `cMD` literally, so an all-in-one `umbrella` run asked for a file
        # no generation has ever written. `input_rel` is the same path the split branch below
        # types, computed once in `_stage_targets`.
        production = plan[-1]["name"] if plan else protocol
        source = targets[production]["input_rel"] if production in targets else f"../input/{protocol}.in"
        lines += [f'md-openmm md-run -i {source} -p "${{TOPOLOGY}}" -s "${{SYSTEM}}" "$@"', '']
    else:
        previous = None
        for stage in plan:
            target = targets[stage["name"]]
            # NO -r, -o, -log or -chk. md-run names all four inside -odir itself, so passing them
            # here would restate four paths per stage that the layout already decides -- and they
            # are taken verbatim against the working directory when given, which is how a stage
            # comes to write its restart next to run.sh instead of into its own directory.
            #
            # NO -x either: the stage names its own coordinate streams and there are two of them.
            # One -x cannot say both `solute_prod1.nc` and `whole_prod1.nc`, and naming only the
            # solute one would silently reinstate the single-trajectory behaviour this replaces.
            call = [f'md-openmm md-run -i {target["input_rel"]} \\',
                    '  -p "${TOPOLOGY}" -s "${SYSTEM}" \\',
                    f'  -odir {target["odir_rel"]} "$@"']
            if previous:
                call.insert(2, f'  -c {previous} \\')
            lines += [f'echo "== {target["key"]} =="'] + call + ['']
            previous = target["restart_rel"]
        if protocol in ("REST2", "rREST2"):
            lines += ['# One rank per thermodynamic state. Any other world size is refused rather',
                      '# than silently reinterpreted: a ladder run in fewer processes than it has',
                      '# states is a different schedule, not a smaller one.',
                      '#',
                      '# EVERY SEGMENT SHARES ONE INPUT and has its own group file. The segment is',
                      '# a runtime fact, not an input fact: `stages.number_of_segments` divides the',
                      '# production total, and each segment writes the `_prod<x>` set of outputs.',
                      '#',
                      '# Anything left in "$@" is passed on: --cpu for an explicit CPU run.',
                      '']
            records = os.path.relpath(Path("remd_records"), ".")
            for segment in range(1, int(segments) + 1):
                call = [f'mpirun -n {states} md-openmm md-run -ng {states} \\',
                        f'  -i {os.path.relpath(Path("..") / "input" / f"{protocol}.in", ".")} \\',
                        '  -p "${TOPOLOGY}" \\',
                        f'  --groupfile remd_groupfile.{segment} \\',
                        '  -odir . \\',
                        f'  -o {records}/{protocol}_prod{segment}.out \\',
                        f'  -log {records}/{protocol}_prod{segment}.log \\',
                        f'  -r {records}/restart_prod{segment}.json']
                if segment > 1:
                    # The physical state comes from the parent segment's checkpoint, not from a
                    # coordinate file: positions, velocities, box, the state-to-walker map and
                    # every RNG stream have to arrive together or the ladder resumes as a
                    # different experiment that still exchanges perfectly.
                    call.append(f'  --extend-from {records}/restart_prod{segment - 1}.json')
                call[-1] = call[-1] + ' "$@"'
                call = [line if line.endswith('\\') or line.endswith('"$@"') else line + ' \\'
                        for line in call]
                lines += [f'echo "== {protocol} prod{segment} =="'] + call + ['']
            # NO -s on the executor call. Each rung is its own pre-scaled Hamiltonian and the
            # group file names it per line; one -s here would be one System for six states.
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

    python AIS.py -p built.pdb -s built.xml -src ../hot/whole_prod1.nc
    python AIS.py -p built.pdb -s built.xml -src ../hot/whole_prod1.nc --check     # validate only
    python AIS.py -p built.pdb -s built.xml -src ../hot/whole_prod1.nc --paths 0-9

Each path is independent and has its own seeds. A completed path is skipped and never appended to;
an interrupted one RESUMES from its last committed checkpoint generation, continuing the same
path rather than starting a new one.
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
    """Generate one run. `out_dir` is the RUN directory; returns the record written in it.

    THE RUN DIRECTORY IS `<system>/<method>-run<N>/`, and what it does NOT contain is the point:
    `build/`, `min/` and `input/` are its siblings, shared by every run on this system, because
    every run starts from the same built System, the same minimised coordinates and the same
    instructions. Only what is genuinely per-run is written here -- this run's equilibration, its
    per-state rungs and output, its records, and `run.config`, which holds the seed and nothing
    else.
    """
    from ..layout import DatasetLayout, RunLayout

    resolved = resolve_md_config(config_path)
    run_root = Path(out_dir)
    # REFUSED IF IT EXISTS AT ALL, not merely if it is non-empty. A run directory is an identity:
    # `REST2-run1` names one experiment, and generating a second one into it would leave two sets
    # of settings over one set of paths with nothing saying which run the outputs belong to. An
    # empty directory of the right name is the ambiguous case rather than the safe one -- it is
    # what a half-cleaned failed attempt leaves behind -- so it is refused too. `next_run_index`
    # never reuses a gap for the same reason.
    if run_root.exists() and not overwrite:
        raise ConfigError(
            f"{run_root} already exists.\n"
            f"  A run directory is an identity: this name refers to one experiment, and "
            f"generating into it again would put two sets of settings over one set of paths.\n"
            f"  Use the next index -- `{run_root.parent.name}/<method>-run<N+1>` -- or pass "
            f"--overwrite if this run has no output worth keeping.")
    dataset = DatasetLayout(run_root.parent)
    run = RunLayout(run_root, dataset=dataset, protocol=resolved["protocol"])

    # CHECKED BEFORE ANYTHING IS WRITTEN, and that ordering is the point rather than a detail.
    # A ladder's rungs are scaled and serialised here now, so `build/built.{xml,pdb}` is a
    # precondition of generating one at all -- and `input/` is SHARED, so a generation that is
    # going to refuse must not have put a file there first. It did: the refusal arrived after
    # `input/min.in` had already been written, which is the same defect as a `--check` that
    # creates its output directory.
    if resolved["protocol"] in ("REST2", "rREST2"):
        missing = [path for path in (dataset.built("xml"), dataset.built("pdb"))
                   if not path.is_file()]
        if missing:
            raise ConfigError(
                "a ladder's rungs are scaled and serialised at BUILD time now, so build-md needs "
                "the built system that every run on this dataset shares:\n"
                + "".join(f"  missing: {path}\n" for path in missing)
                + f"  Run `md-openmm build-top` into {dataset.build}/ first. (The scaling used to "
                  f"happen at run time from one shared built.xml, which is why this used to "
                  f"generate without it.)")
        # THE OMEGA CLASSIFICATION, also before anything is written, with the SAME evidence the
        # run-time preflight uses: the `built.sdf` that `build-top` retains beside `built.xml`.
        # The rung writer below was called without it, so every amide of a `peptide-like` or
        # `ligand` solute arrived unclassified, was left out of the exclusions, and was SCALED in
        # the rung files a grouped ladder integrates -- while `build_states.log` still said
        # "ordinary_amide_omega: unscaled". It enforces the same refusal itself; this is here so
        # the refusal arrives before `input/` and the run directory exist.
        from openmm.app import PDBFile

        from ..md.stage import solute_atom_indices
        from ..openmm.system import UnclassifiedOmegaError, omega_exclusions
        from ..run.preflight import _ligand_sdf_beside

        topology = PDBFile(str(dataset.built("pdb"))).topology
        try:
            omega_exclusions(topology, solute_atom_indices(topology),
                             ligand_sdf=_ligand_sdf_beside(dataset.built("xml")))
        except UnclassifiedOmegaError as refusal:
            raise ConfigError(f"{resolved['protocol']} rungs for {dataset.built('pdb')}: "
                              f"{refusal}") from None

    run_root.mkdir(parents=True, exist_ok=True)
    # Kept as `out_dir` below: the cv/umbrella copies, the build log and the generated helpers all
    # belong to this run and stay at its root.
    out_dir = run.root

    # -- the collective-variable definition, resolved and copied in -----------------------------
    #
    # Resolved relative to THE CONFIGURATION FILE, not the working directory: a run generated from
    # `configs/md/cMD.config` naming `cv.yaml` means the one beside that config, and resolving
    # against the caller's cwd makes the same configuration mean different things depending on
    # where `build-md` was invoked from.
    #
    # Then COPIED IN, content-addressed. The generated directory is meant to be movable -- copied
    # to a cluster, archived beside its results -- and an absolute path to a definition file
    # somewhere else survives none of that. Worse, it survives it SILENTLY when the path happens
    # to exist on the target machine and holds a different file. The copy is named by its digest
    # so a definition that changed cannot quietly replace one a previous run used.
    # EVERY DECLARATION TRAVELS WITH THE DEFINITION IT NAMES, and that is why the copies are
    # collected rather than written once.
    #
    # `resolved.config` records the copy by BARE NAME so the directory stays movable, and the
    # name resolves beside whichever declaration a reader started from. The layout then put
    # declarations in four places -- the run root, the SHARED `input/`, `min/` and `eq/` -- while
    # the bytes existed in exactly one of them. So `input/cMD.in` named `cv.<digest>.yaml` and
    # `input/` held no such file: `md-openmm md-run -i ../input/cMD.in` could not resolve its own
    # definition from any directory, and `md_tools.run.continuation` -- which has to LOAD the
    # definition to know what the invocation intends to continue -- got None and returned early.
    # The read-only boundary then silently did nothing, and a refused continuation wrote
    # `resolved.config`, `<stage>.out` and `<stage>.log` into the tree it was declining to touch.
    #
    # The name is content-addressed, so sharing one is safe by construction: a definition that
    # changed gets a different name and cannot quietly replace the one a previous run read.
    definition_copies: list[tuple[str, bytes]] = []
    cv_provenance = None
    cv_block = resolved.get("collective_variables") or {}
    if cv_block.get("file"):
        from ..cv import load_cv_definition

        source = Path(cv_block["file"])
        if not source.is_absolute() and config_path is not None:
            source = (Path(config_path).parent / source).resolve()
        # Parsed here, so a malformed definition is refused at BUILD time rather than by every
        # generated script at run time -- and refused once, with the path the person wrote.
        definition = load_cv_definition(source)
        copied = out_dir / f"cv.{definition.digest[:12]}.yaml"
        copied.write_bytes(source.read_bytes())
        definition_copies.append((copied.name, copied.read_bytes()))
        # Only `file` changes in the resolved configuration -- it is a SETTINGS document, and its
        # schema rightly refuses keys that are not settings. Where the definition came from and
        # what it hashed to are provenance of the BUILD, and are recorded in the build record
        # below, which is where the rest of this generation's provenance already lives.
        resolved["collective_variables"] = dict(cv_block, file=copied.name)
        cv_provenance = {
            "source_path": str(source),
            "source_sha256": definition.digest,
            "copied_as": copied.name,
            "collective_variables": list(definition.names),
        }

    # The umbrella definition travels with the generated directory too, and for the same reasons:
    # a run must not depend on a path outside the directory it was generated into, and a
    # definition that changed must not quietly replace one a previous run used. Parsed HERE, so a
    # restraint naming a variable that does not exist is refused at build time -- once, with the
    # path the person wrote -- rather than by every generated script when it reaches a node.
    umbrella_provenance = None
    umbrella_block = resolved.get("umbrella") or {}
    if umbrella_block.get("file"):
        from ..umbrella import load_umbrella_definition
        from ..umbrella.definition import definition_digest

        source = Path(umbrella_block["file"])
        if not source.is_absolute() and config_path is not None:
            source = (Path(config_path).parent / source).resolve()
        restraints = load_umbrella_definition(source, definition)
        digest = definition_digest(source)
        copied = out_dir / f"umbrella.{digest[:12]}.yaml"
        copied.write_bytes(source.read_bytes())
        definition_copies.append((copied.name, copied.read_bytes()))
        resolved["umbrella"] = dict(umbrella_block, file=copied.name)
        umbrella_provenance = {
            "source_path": str(source),
            "source_sha256": digest,
            "copied_as": copied.name,
            "restraints": [entry.record() for entry in restraints],
        }

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

    targets = _stage_targets(plan, run=run, dataset=dataset)
    segments = int(resolved["stages"]["number_of_segments"])

    written: list[str] = []
    where: dict[str, Path] = {}

    def note(path: Path) -> str:
        """Record a written file by its path RELATIVE TO THE RUN, so the log shows where it went.

        A shared input reads as `../input/REST2.in` and a per-run file as `eq/eq_1.py`, which is
        the distinction the layout exists to make. A bare basename would print them the same.
        """
        label = os.path.relpath(path, run.root)
        written.append(label)
        where[label] = path
        return label

    if all_in_one:
        path = out_dir / "md.py"
        path.write_text(_ALL_IN_ONE, encoding="utf-8")
        note(path)
    else:
        for index, stage in enumerate(plan):
            path = targets[stage["name"]]["script"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_stage_script(stage, first=index == 0), encoding="utf-8")
            note(path)

    if protocol == "AIS":
        # No `run` dict is embedded any more: the AIS script reads `resolved.config` beside it,
        # which is the single declaration of the workflow.
        path = out_dir / "AIS.py"
        path.write_text(_AIS_SCRIPT.format(
            paths=resolved["ais"]["number_of_paths"],
            tau_start=resolved["ais"]["tau_start"], tau_end=resolved["ais"]["tau_end"]),
            encoding="utf-8")
        note(path)
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
            "equilibration_steps": resolved["rest2"]["equilibration_steps"],
            "state_trajectory": resolved["rest2"]["state_trajectory"],
            "rem_log": resolved["rest2"]["rem_log"],
            "neighbour_acceptance_report": resolved["rest2"]["neighbour_acceptance_report"],
            "reservoir": dict(resolved["reservoir"]),
            "dynamics": dict(resolved["dynamics"]),
            # Kept in step with `ladder_from_resolved`, which is what the RUN rebuilds from. A
            # field added to one and not the other reaches the log and never the simulation.
            "reporting": dict(resolved["reporting"]),
        }
        path = out_dir / f"{protocol}.py"
        path.write_text(_REPLICA_SCRIPT.format(protocol=protocol,
                                               states=ladder["n_states"]), encoding="utf-8")
        note(path)
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
        # Stated whether or not it was asked for. "Equilibration: 0 steps" is a fact a reader
        # needs in order to interpret the opening exchanges; a silently absent line is not.
        equilibration = ladder["equilibration_steps"]
        if numeric:
            log.field("equilibration", f"{equilibration} steps = "
                                       f"{equilibration * numeric / 1000.0:g} ps per state, "
                                       f"at that state's own Hamiltonian, not production")
        else:
            log.field("equilibration", f"{equilibration} steps per state, at that state's own "
                                       f"Hamiltonian, not production")
        per_tau = per_tau_equilibration_stages(resolved)
        if per_tau:
            log.field("per-tau equilibration",
                      "every rung, under its own tau, before `equilibration` above: "
                      + ", ".join(f"{s['name']} {s['steps']} steps"
                                  + (f" restrained at {s['restraint_kcal_per_mol_A2']:g} "
                                     f"kcal/mol/A^2" if s["restraint_kcal_per_mol_A2"] else "")
                                  for s in per_tau))
            log.field("ladder starts from", f"{plan[-1]['name']}.xml (the tau = 0 chain above)")

    # The Amber-like inputs, beside the Python entry points. Two shapes over ONE resolved run:
    # `python min.py` and `md-openmm md-run -i min.in` reach the same function with the same
    # settings, and the round-trip is asserted rather than asserted-in-a-comment.
    # THE INPUTS ARE SHARED, so they are written to the dataset's `input/` rather than into this
    # run -- and an existing one that differs is refused rather than replaced, because the runs
    # already beside it read that file. `min.in` and `eq_<k>.in` are shared across METHODS as well
    # as repeats: if two methods on one system need different equilibration they are not
    # comparable and belong under a different `<system>`.
    if protocol == "AIS":
        # AIS reads neither `min.in` nor `eq_*.in`: a switching campaign starts from a source
        # ensemble that already exists, so it has no preparation chain of its own.
        _write_shared_input(
            dataset.stage_input("AIS"),
            in_file_text(resolved, heading=f"AIS: {resolved['ais']['number_of_paths']} switching "
                                           f"paths"), overwrite=overwrite)
        note(dataset.stage_input("AIS"))
    else:
        for stage in plan:
            target = targets[stage["name"]]
            # ONLY THE PREPARATION STAGES ARE WRITTEN IN THE SHARED FORM, and getting this wrong
            # is not cosmetic: applying it to every stage stripped `protocol` and `umbrella_file`
            # from the cMD and umbrella PRODUCTION inputs, which is the one input that has to
            # carry them -- it is what says which method the run is.
            #
            # `min` and `eq_<k>` are shared across methods and drop method identity; a production
            # input is the method, is named for it, and keeps everything.
            preparation = stage["name"] == "min" or stage["name"].startswith("eq_")
            # `stage=` is the STAGE name and the heading carries the LAYOUT name: the first
            # decides the physics the runtime resolves, the second is what the file is called.
            # The ensemble is stated because it can no longer live in the filename. A preparation
            # heading names no protocol either -- the heading is part of the bytes, so a protocol
            # in the comment alone would make `min.in` method-specific again.
            heading = (f"{target['key']}  [{stage['name']}, {stage['ensemble']}]" if preparation
                       else f"{protocol}: {target['key']}  [{stage['ensemble']}]")
            _write_shared_input(
                target["input"],
                in_file_text(resolved, stage=stage["name"], preparation=preparation,
                             heading=heading),
                overwrite=overwrite)
            note(target["input"])
        if protocol in ("REST2", "rREST2"):
            _write_shared_input(
                dataset.stage_input(protocol),
                in_file_text(resolved,
                             heading=f"{protocol}: "
                                     f"{resolved['rest2']['number_of_replicas']} states, "
                                     f"{segments} segment(s)"), overwrite=overwrite)
            note(dataset.stage_input(protocol))

    # THE DEFINITIONS THE SHARED INPUTS NAME, beside those inputs. `input/*.in` and
    # `input/`-resolved declarations record `cv_file` by bare name; without the copy here the
    # shared input named a file its own directory did not contain, and `md-run -i ../input/...`
    # could not resolve it at all. Content-addressed, so every run on this system shares one.
    _place_definitions(dataset.input, definition_copies, overwrite=overwrite, note=note)

    # -- the rungs, serialised, and one group file per segment ----------------------------------
    #
    # SCALING HAPPENS HERE NOW, not at run time. Each rung is written to
    # `remd<n>/build_state<n>.xml` with `build_states.log` recording which factors were applied to
    # which terms and which torsions were left alone, so the Hamiltonian a state ran under is
    # readable rather than re-derivable. The group file names one rung per line.
    rung_record = None
    if protocol in ("REST2", "rREST2"):
        from ..remd.generated import tau_ladder
        from .rungs import RungWriteError, format_scaling_report, write_rung_systems
        from ..run.preflight import _ligand_sdf_beside

        system_path, topology_path = dataset.built("xml"), dataset.built("pdb")
        missing = [path for path in (system_path, topology_path) if not path.is_file()]
        if missing:
            raise ConfigError(
                "a ladder's rungs are scaled and serialised at BUILD time now, so build-md needs "
                "the built system that every run on this dataset shares:\n"
                + "".join(f"  missing: {path}\n" for path in missing)
                + f"  Run `md-openmm build-top` into {dataset.build}/ first. (The scaling used to "
                  f"happen at run time from one shared built.xml, which is why this used to "
                  f"generate without it.)")
        taus = tau_ladder(int(resolved["rest2"]["number_of_replicas"]),
                          float(resolved["rest2"]["tau_max"]))
        try:
            rung_record = write_rung_systems(run, system_path=system_path,
                                             topology_path=topology_path, taus=taus,
                                             ligand_sdf=_ligand_sdf_beside(system_path),
                                             overwrite=overwrite)
        except RungWriteError as failure:
            raise ConfigError(str(failure)) from None
        for state in rung_record["states"]:
            note(run.state_system(int(state["state"])))
        note(run.root / "build_states.log")

        # `solute.yaml` is named on every group line when it exists beside the built system. It is
        # the resolved scaling selection, and a ladder that derived its own would be a second
        # answer to "what is the solute" -- two answers waiting to disagree, invisibly.
        solute_yaml = dataset.build / "solute.yaml"
        for segment in range(1, segments + 1):
            path = run.root / f"remd_groupfile.{segment}"
            path.write_text(_group_file_text(
                protocol=protocol, states=len(taus), segment=segment, segments=segments,
                taus=taus, targets=targets, run=run, dataset=dataset,
                solute_yaml=solute_yaml if solute_yaml.is_file() else None), encoding="utf-8")
            note(path)
        # The directories a segment writes into, made now so a refusal about them happens here
        # rather than three hours into a queue.
        for directory in (run.records, run.rank):
            directory.mkdir(parents=True, exist_ok=True)

    run_sh = out_dir / "run.sh"
    run_sh.write_text(_run_sh(plan, all_in_one=all_in_one, protocol=protocol,
                              resolved=resolved, targets=targets, segments=segments),
                      encoding="utf-8")
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    note(run_sh)

    (out_dir / "resolved.config").write_text(
        "# The configuration these scripts were generated from, fully resolved.\n"
        "# Every default is written out, so this file alone reproduces the generation.\n"
        + yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False), encoding="utf-8")
    note(out_dir / "resolved.config")

    # EVERY DIRECTORY HOLDING A GENERATED SCRIPT CARRIES THE DECLARATION THOSE SCRIPTS READ.
    #
    # `resolved_config_beside` is `Path(script).parent / "resolved.config"`, strictly and
    # deliberately: a generated directory must run correctly from anywhere, and looking anywhere
    # else -- the working directory, or a walk up the tree -- is how a run silently picks up
    # another project's configuration. The layout puts scripts in two directories that are not
    # the run root, so both need the file or the script cannot say what it was generated for.
    for directory, document, why, shared in _script_directories(plan, run=run, dataset=dataset,
                                                                resolved=resolved):
        if directory == out_dir:
            continue
        # BESIDE THE DECLARATION, in `min/` and `eq/` alike. Both hold generated scripts that
        # read the `resolved.config` written here, and that document names its definitions by
        # bare name -- so a stage in `eq/` resolving `cv.<digest>.yaml` beside its own
        # declaration found nothing, and the definition lived only at the run root.
        _place_definitions(directory, definition_copies, overwrite=overwrite, note=note)
        if shared:
            # `min/` HAS ITS OWN SEED, and that is the decision rather than a workaround.
            #
            # The seed is per run everywhere else, so a shared directory holding one run's seed
            # would stamp run 1's value into a file run 2 reads -- and `md-run`, resolving
            # `input/min.in` with `-odir ../min`, found no `run.config` there, resolved the seed
            # to the schema default and refused every documented example with
            # `dynamics.seed: was 20260908, now 1`.
            #
            # So the minimisation gets its own `run.config`. Its seed is then explicit, belongs to
            # the minimisation rather than to whichever run happened to generate it first, and is
            # allowed to differ from every run on the system: minimisation draws no velocities and
            # has no seeded stochastic element, so nothing about a run's sampling descends from it.
            _write_min_directory(directory, document, resolved=resolved, why=why,
                                 overwrite=overwrite, log=log, note=note, run=run)
            continue
        # BYTE-IDENTICAL to the run root's when the document is the run's own, which is why the
        # header is the same two lines rather than a per-directory note: a reader comparing
        # `<run>/resolved.config` with `<run>/eq/resolved.config` must see one document, not two
        # that happen to agree. What the directory is FOR belongs in the build log, not in a
        # comment that makes the copies differ.
        text = ("# The configuration these scripts were generated from, fully resolved.\n"
                "# Every default is written out, so this file alone reproduces the generation.\n"
                + yaml.safe_dump(document, sort_keys=False, default_flow_style=False))
        log.field(f"{os.path.relpath(directory, run.root)}/resolved.config", why)

        # AND THE SEED BESIDE IT. `md-run` layers `run.config` from `-odir`, so a stage run into
        # `eq/` resolves its shared `../input/eq_<k>.in` against `eq/run.config`. Without one the
        # seed fell back to the schema default and every equilibration stage refused with
        # `eq/resolved.config describes a different run: dynamics.seed: was <run>, now 1` -- the
        # same failure `min/` had, for the same reason. A directory that holds generated scripts
        # needs the declaration AND the per-run value it was resolved with, or the two disagree.
        # CREATED FIRST. `--all-in-one` emits a single `md.py` and no per-stage scripts, so
        # nothing has made `eq/` by the time this runs -- and the write failed with a bare
        # FileNotFoundError AFTER the log had been written, leaving a half-generated run
        # directory behind. Every other directory here is created before it is written into.
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.config").write_text(
            "# The seed this run's equilibration was resolved with. Identical to the run root's:\n"
            "# equilibration draws Maxwell velocities from this run's seed, which is what makes\n"
            "# two repeats of one method diverge.\n"
            + yaml.safe_dump({"dynamics": {"seed": resolved["dynamics"]["seed"]}},
                             sort_keys=False), encoding="utf-8")
        note(directory / "run.config")
        # Shared directories go through the byte-identity gate, exactly as `input/` does: `min/`
        # is read by every run on this system, so a second run generated from a different
        # configuration must not quietly replace what the first one minimised under.
        _write_shared_input(directory / "resolved.config", text, overwrite=overwrite)
        note(directory / "resolved.config")

    # THE PER-RUN DECLARATION, and it is what closes the round trip. The generated `.in` files
    # deliberately do NOT carry `random_seed`: `input/` is shared by every repeat of a method on
    # one system, and the seed is the one value that differs between them (the three ALA-explicit
    # reference runs' inputs differ by exactly that line and nothing else).
    #
    # But `md-run` re-resolves the `.in` and compares the result against the `resolved.config`
    # beside it -- "every .in that build-md writes resolves back to exactly the resolved.config
    # beside it" is a test, not a convention. With the seed emitted nowhere, that comparison fails
    # with `dynamics.seed: was 11, now 1` and refuses to continue. So the seed is written HERE, and
    # `md-run` layers `-odir/run.config` over the input to resolve the same document again.
    (out_dir / "run.config").write_text(
        "# What is per RUN rather than per method. The shared inputs in input/ say what this\n"
        "# protocol was asked to do; this says which repeat it is.\n"
        "#\n"
        "# Only the seed may be set here -- see `RUN_CONFIG_ALLOWED`. `derive_seed` hashes it with\n"
        "# each stage and replica name, so every stream in this run descends from this number, and\n"
        "# two runs sharing an input and a seed would be bit-identical rather than repeats.\n"
        + yaml.safe_dump({"dynamics": {"seed": resolved["dynamics"]["seed"]}}, sort_keys=False),
        encoding="utf-8")
    note(out_dir / "run.config")

    if rung_record is not None:
        log.heading("Rung Systems")
        for line in format_scaling_report(rung_record).splitlines():
            log(f"  {line}" if line else "")

    log.heading("Outputs")
    for name in written:
        log.field(name, where[name])
    log.update(protocol=protocol, solvent=resolved["solvent"],
               all_in_one=bool(all_in_one), resolved_config=resolved,
               stages=[{k: v for k, v in s.items()} for s in plan],
               files=written,
               **({"collective_variable_definition": cv_provenance} if cv_provenance else {}),
               **({"umbrella_definition": umbrella_provenance} if umbrella_provenance else {}))
    log.complete()
    log.heading("Summary")
    log(f"  generated {len(written)} files in {out_dir}")
    log("  status: completed")
    log.save()
    return log.record
