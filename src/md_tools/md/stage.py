"""One MD stage: minimisation, equilibration, or production.

Generated scripts call `stage_main` with the stage's resolved settings. Everything that decides
what is integrated -- the restraint, the barostat, the timestep, the seeds -- arrives in that
dict, so the script is readable and the behaviour is auditable from the script alone.

The command-line surface is Amber-like and identical for every stage:

    -p    topology PDB              (what `build-top` wrote as built.pdb)
    -s    serialised System XML     (what `build-top` wrote as built.xml)
    -c    input state from the previous stage (omit for the first stage)
    -x    output trajectory (DCD)
    -r    output final state XML -- the handoff this stage produces
    -chk  output checkpoint
    -log  the readable log, which carries this stage's machine record

Restart rules, which are the whole reason the checkpoint and the record both exist:

  * a stage whose log carries `status: completed` is NOT rerun. Rerunning would overwrite a
    finished handoff that later stages may already have consumed;
  * a stage with a checkpoint but no completion resumes from that checkpoint, but only after the
    checkpoint is shown to belong to THIS stage's configuration -- a checkpoint written under a
    different timestep, ensemble or system is refused rather than silently continued;
  * anything else starts from the beginning.

Completion is written only after the trajectory, the checkpoint and the final state are flushed,
closed, reopened and re-read. A log line is not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..openmm.platform_policy import (PlatformRequest, acceleration_record,
                                      resolve_platform_request)
from ..openmm.timestep import resolve_timestep_fs
from ..build.record import LogWriter, file_facts, openmm_platform_facts, read_record


#: Residue names that are solvent or a monatomic counter-ion, and therefore not solute. Positional
#: restraints act on the solute; restraining water would freeze the bath the equilibration exists
#: to relax.
SOLVENT_RESIDUES = frozenset({
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4", "T3P", "T4P", "OPC", "SPC",
    "NA", "NA+", "CL", "CL-", "K", "K+", "LI", "LI+", "CS", "CS+", "RB", "RB+",
    "BR", "BR-", "F", "F-", "I", "I-", "MG", "ZN", "CA",
})


def solute_atom_indices(topology) -> list[int]:
    """Every atom that is not solvent or a counter-ion, in topology order."""
    return [atom.index for atom in topology.atoms()
            if atom.residue.name.strip().upper() not in SOLVENT_RESIDUES]


def stage_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates (built.pdb)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System (built.xml)")
    parser.add_argument("-c", "--continue-from", default=None, metavar="XML",
                        help="final state written by the previous stage; omit for the first stage")
    parser.add_argument("-x", "--trajectory", default=None, metavar="DCD",
                        help="output trajectory")
    parser.add_argument("-r", "--restart", default=None, metavar="XML",
                        help="output final state, the handoff to the next stage")
    parser.add_argument("-chk", "--checkpoint", default=None, metavar="CHK",
                        help="output checkpoint, written periodically for resume")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="readable log carrying this stage's machine record")
    parser.add_argument("--cpu", action="store_true",
                        help="run on the OpenMM CPU platform. CUDA is the default and is "
                             "mandatory; this is the only way to ask for a CPU run, and the "
                             "record says that you did")
    parser.add_argument("--platform", default=None,
                        help="force a named OpenMM platform. CUDA is the default; there is no "
                             "automatic fall back to anything else")
    parser.add_argument("--device", default=None, help="CUDA device index")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs and settings, then exit without integrating")
    return parser


#: Fields a continuation may legitimately change, and which are therefore NOT part of the
#: fingerprint a checkpoint is matched against.
#:
#: `steps` is the whole point: asking for a longer run is an extension, and an extension must be
#: able to resume from the checkpoint the shorter run left. Everything else -- the ensemble, the
#: timestep, the temperature, the restraint, the seed, the System itself -- would make the
#: continuation a different simulation wearing the same file names, so all of it is fingerprinted.
EXTENDABLE_FIELDS = frozenset({"steps", "description"})


def _config_fingerprint(stage: dict[str, Any], system_sha: str, topology_sha: str) -> str:
    """What a checkpoint has to match before it may be resumed from.

    Deliberately includes the System and topology digests. A checkpoint carries positions and
    velocities for a particular particle set; resuming it against a System that was rebuilt is how
    a run continues with the right-looking numbers and the wrong molecule.
    """
    # `resolved.config` is bound in by DIGEST as well as through the stage derived from it. The
    # stage alone would miss an edit to another part of the file -- a different stage's length, a
    # reporting interval -- and the whole point of `resolved.config` being the single declaration
    # is that the run this checkpoint belongs to is the one that file describes.
    config_sha = None
    config_path = stage.get("resolved_config")
    if config_path and Path(config_path).is_file():
        config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    payload = {"stage": {k: v for k, v in sorted(stage.items())
                         if k not in EXTENDABLE_FIELDS and k != "resolved_config"},
               "system_sha256": system_sha, "topology_sha256": topology_sha,
               "resolved_config_sha256": config_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_timestep_against_masses(timestep_fs: float, system, topology) -> None:
    """Refuse a large timestep on hydrogens that were never repartitioned.

    Kept as the narrow yes/no question. The full decision -- including resolving `auto` -- is
    `md_tools.openmm.timestep.resolve_timestep_fs`, which this delegates to so there is one
    implementation of the rule and one place the masses are read.
    """
    resolve_timestep_fs(float(timestep_fs), system, topology)


def stage_main(stage: dict[str, Any], argv: list[str] | None = None) -> int:
    """Run one stage. `stage` is the resolved settings the generated script declares."""
    args = stage_parser(stage.get("description", "one MD stage")).parse_args(argv)

    name = stage["name"]
    topology_path = Path(args.topology)
    system_path = Path(args.system)
    log_path = Path(args.log) if args.log else Path(f"{name}.log")
    traj_path = Path(args.trajectory) if args.trajectory else Path(f"{name}.dcd")
    restart_path = Path(args.restart) if args.restart else Path(f"{name}.xml")
    chk_path = Path(args.checkpoint) if args.checkpoint else Path(f"{name}.chk")

    for path, what in ((topology_path, "-p topology"), (system_path, "-s system")):
        if not path.is_file():
            print(f"{name}: {what} {path} does not exist", file=sys.stderr)
            return 2

    # -- already finished? ------------------------------------------------------------------
    if log_path.is_file():
        try:
            previous = read_record(log_path)
        except Exception:
            previous = None
        if previous and previous.get("status") == "completed":
            print(f"{name}: already completed ({log_path}); not rerunning. "
                  f"Delete {log_path} to force a rebuild.")
            return 0

    from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter, CheckpointReporter

    from ..md._stages import (add_barostat, add_positional_restraint,
                                              count_barostats, derive_seed, resolve_platform,
                                              set_restraint, write_final_state)

    log = LogWriter(log_path, record_type=f"md-stage:{name}")
    log(f"md-openmm stage: {name}")
    log("=" * 68)
    log.heading("Inputs")
    log.field("topology", topology_path)
    log.field("system", system_path)
    log.field("continue from", args.continue_from or "(none -- first stage)")

    try:
        pdb = PDBFile(str(topology_path))
        system = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
        if pdb.topology.getNumAtoms() != system.getNumParticles():
            raise SystemExit(f"{topology_path} has {pdb.topology.getNumAtoms()} atoms but "
                             f"{system_path} has {system.getNumParticles()} particles")

        implicit = not system.usesPeriodicBoundaryConditions()
        # Derived from the topology rather than carried in the config. `build-top` writes the
        # solute first, so a count would work, but deriving it here means a hand-edited script
        # cannot restrain the wrong atoms by stating a stale number.
        solute = solute_atom_indices(pdb.topology)
        seed = derive_seed(int(stage["seed"]), name)

        log.heading("Resolved settings")
        for key in ("ensemble", "steps", "timestep_fs", "temperature_K", "pressure_bar",
                    "friction_per_ps", "barostat_interval_steps", "restraint_kcal_per_mol_A2",
                    "minimization_iterations", "trajectory_interval_steps",
                    "state_interval_steps", "checkpoint_interval_steps"):
            if stage.get(key) is not None:
                log.field(key, stage[key])
        # The numerical timestep is decided HERE, against the masses in the System that was just
        # loaded -- not upstream in `build-md`, which never opens built.xml and would have to
        # trust a configuration's claim about hydrogen mass repartitioning. Step counts stay
        # authoritative; a physical duration is only derived once this returns.
        timestep = resolve_timestep_fs(stage["timestep_fs"], system, pdb.topology)
        timestep_fs = timestep["timestep_fs"]
        steps = int(stage.get("steps") or 0)
        log.field("timestep", f"{timestep_fs} fs (requested {timestep['requested']!r}, "
                              f"{timestep['basis']}; heaviest hydrogen "
                              f"{timestep['heaviest_hydrogen_amu']} amu)")
        log.field("derived time", f"{steps} steps x {timestep_fs} fs = "
                                  f"{steps * timestep_fs / 1000.0:g} ps "
                                  f"({steps * timestep_fs / 1e6:g} ns)")
        log.update(timestep=timestep)
        if implicit and stage["ensemble"] != "NVT":
            raise SystemExit(
                f"stage {name} declares ensemble {stage['ensemble']}, but the System is not "
                f"periodic. Implicit solvent has no volume to control, so there is no NPT here.")
        log.field("solvent", "implicit (no barostat possible)" if implicit else "explicit")

        # -- the Force layout, fixed before any state is loaded -----------------------------
        #
        # Order matters and is the same order the REST2 ladder uses: SCALE FIRST, on the bare
        # System, then restrain, then add the barostat. The scaler audits every force and refuses
        # one it cannot classify, and the restraint and barostat are stage machinery rather than
        # terms of the molecular Hamiltonian, so neither may be scaled. Scaling last would scale
        # them, and a fixed-tau walker would then construct a different System from the ladder
        # rung it is supposed to match.
        tau = float(stage.get("tau") or 0.0)
        excluded: list = []
        if tau > 0.0:
            if stage["ensemble"] != "NVT":
                raise SystemExit(
                    f"stage {name} runs at tau={tau} but declares ensemble "
                    f"{stage['ensemble']}. A scaled run samples the fixed-volume ensemble of the "
                    f"ladder rung it sits at; a barostat would sample a different distribution.")
            from ..openmm.system import classify_omega_bonds
            from ..rest2 import build_scaled_system
            omega = classify_omega_bonds(pdb.topology, solute, route="peptide", ligand_sdf=None)
            excluded = [tuple(int(a) for a in bond)
                        for bond in omega.get("omega_unscaled_bonds", [])]
            system = build_scaled_system(system, solute, tau, excluded_bonds=excluded)
            log.field("tau", f"{tau}  (fixed REST2 scaling; ordinary amide omega left unscaled, "
                             f"{len(excluded)} bond(s) excluded)")

        # The MOLECULAR Hamiltonian, fingerprinted HERE, before any stage machinery is added.
        # The restraint (always present, at zero strength when unrestrained) and the barostat are
        # properties of how this stage is run, not terms of the energy the ensemble is defined
        # by. A reservoir fingerprint taken after them claims a CustomExternalForce the ladder
        # rung it refreshes does not have, and the probability-one transfer is then refused --
        # correctly, for the wrong reason.
        #
        # Computed now rather than holding a reference: `add_positional_restraint` mutates the
        # System in place, so a reference would be fingerprinted after the restraint anyway.
        hamiltonian_identity_record = None
        if stage.get("phase_space_interval_steps"):
            from ..rest2 import identity_record
            hamiltonian_identity_record = identity_record(
                system, tau=tau, temperature_k=float(stage["temperature_K"]),
                ensemble=stage["ensemble"], solute_indices=solute, excluded_bonds=excluded)

        restrained = float(stage.get("restraint_kcal_per_mol_A2") or 0.0) > 0.0
        add_positional_restraint(system, pdb.positions, solute)
        if not implicit:
            barostat_active = stage["ensemble"] == "NPT"
            add_barostat(system, float(stage["pressure_bar"]), float(stage["temperature_K"]),
                         derive_seed(int(stage["seed"]), name, "barostat"),
                         frequency=int(stage["barostat_interval_steps"]) if barostat_active else 0)
        if implicit and count_barostats(system):
            raise SystemExit("implicit solvent must carry no barostat")

        # ONE platform decision, from `md_tools.openmm.platform_policy`, shared with REMD and AIS.
        # CUDA unless `--cpu` was written; a CUDA that cannot open a Context is an error here,
        # before minimisation, rather than a silent CPU run that finishes hours later.
        acceleration = resolve_platform_request(
            PlatformRequest.from_flags(cpu=bool(args.cpu),
                                       platform=args.platform or stage.get("platform")),
            device_index=int(args.device) if args.device is not None else None)
        integrator = LangevinMiddleIntegrator(
            float(stage["temperature_K"]) * unit.kelvin,
            float(stage["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(int(seed))
        simulation = Simulation(pdb.topology, system, integrator,
                                acceleration.platform, acceleration.properties)
        log.update(acceleration=acceleration_record(acceleration))
        log.field("acceleration", f"{acceleration.name} "
                                  f"({acceleration_record(acceleration)['requested_policy']})")
        set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))

        system_sha = file_facts(system_path)["sha256"]
        topology_sha = file_facts(topology_path)["sha256"]
        fingerprint = _config_fingerprint(stage, system_sha, topology_sha)
        log.update(stage=dict(stage), fingerprint=fingerprint, implicit=bool(implicit),
                   inputs={"topology": file_facts(topology_path),
                           "system": file_facts(system_path)},
                   derived={"production_ps": steps * timestep_fs / 1000.0,
                            "timestep_fs": timestep_fs, "steps": steps})

        # -- where do we start? -------------------------------------------------------------
        done = 0
        sidecar = chk_path.with_suffix(chk_path.suffix + ".json")
        if chk_path.is_file() and sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                meta = {}
            if meta.get("fingerprint") != fingerprint:
                raise SystemExit(
                    f"{chk_path} was written under a different configuration for this stage "
                    f"(fingerprint mismatch). Resuming it would continue a run that was set up "
                    f"differently. Delete the checkpoint to start this stage over.")
            simulation.loadCheckpoint(str(chk_path))
            done = int(meta.get("steps_done", 0))
            log.heading("Resume")
            log.field("from checkpoint", f"{chk_path} at step {done}")
        elif args.continue_from:
            parent = Path(args.continue_from)
            if not parent.is_file():
                if args.check:
                    # Not a failure. --check exists so a whole chain can be validated before any
                    # of it runs, and in an unrun chain every parent after the first is missing
                    # by construction. Calling that a failure would make --check useless exactly
                    # when it is most useful.
                    log.heading("Preflight")
                    log(f"  [pending] parent state {parent} does not exist yet; stage "
                        f"'{name}' is later in the chain. No Context was created.")
                    log.record["status"] = "pending"
                    log.save()
                    return 0
                raise SystemExit(f"-c {parent} does not exist: the previous stage writes it only "
                                 f"when it finishes")
            state = XmlSerializer.deserialize(parent.read_text(encoding="utf-8"))
            simulation.context.setState(state)
            set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))
            # The parent state is recorded WITH ITS DIGEST, not just its name. Registration
            # re-hashes every recorded input and refuses a directory whose files no longer match
            # the records, so a parent that was rewritten after this stage consumed it is caught
            # rather than silently accepted as this run's ancestry.
            log.record.setdefault("inputs", {})["continued_from"] = file_facts(parent)
            log.field("started from", f"{parent}  "
                                      f"(sha256 {log.record['inputs']['continued_from']['sha256'][:12]}...)")
        else:
            simulation.context.setPositions(pdb.positions)
            log.field("started from", f"{topology_path} coordinates")

        log.field("platform", simulation.context.getPlatform().getName())

        # The barostat, counted on the SYSTEM THE CONTEXT WAS BUILT FROM. One is present in every
        # explicit stage, so the Force layout -- and therefore the checkpoint layout -- does not
        # change along the chain; what makes a stage NPT is its FREQUENCY, not its presence. A
        # frequency changed after the Context exists is invisible until reinitialisation, which is
        # the mistake this records against.
        from ..md._stages import active_barostat_count
        barostats = {
            "in_system": count_barostats(system),
            "active": active_barostat_count(simulation),
            "frequency_steps": (int(stage["barostat_interval_steps"])
                                if stage["ensemble"] == "NPT" and not implicit else 0),
        }
        barostats["interval_ps"] = barostats["frequency_steps"] * timestep_fs / 1000.0
        log.field("barostat", f"{barostats['in_system']} in system, {barostats['active']} active"
                              + (f", every {barostats['frequency_steps']} steps "
                                 f"({barostats['interval_ps']:g} ps)"
                                 if barostats["active"] else ""))
        log.update(platform=openmm_platform_facts(simulation.context), barostats=barostats)

        if args.check:
            log.heading("Preflight")
            log("  --check: inputs, settings and Force layout validated; no dynamics were run.")
            log.record["status"] = "checked"
            log.save()
            return 0

        # -- do the work --------------------------------------------------------------------
        log.heading("Run")
        iterations = int(stage.get("minimization_iterations") or 0)
        if iterations:
            log(f"  minimising, {iterations} iterations")
            simulation.minimizeEnergy(maxIterations=iterations)

        remaining = steps - done
        if remaining > 0:
            if stage.get("trajectory_interval_steps"):
                simulation.reporters.append(
                    DCDReporter(str(traj_path), int(stage["trajectory_interval_steps"]),
                                append=done > 0 and traj_path.is_file()))
            if stage.get("state_interval_steps"):
                simulation.reporters.append(
                    StateDataReporter(str(log_path.with_suffix(".csv")),
                                      int(stage["state_interval_steps"]), step=True, time=True,
                                      potentialEnergy=True, temperature=True, volume=True,
                                      density=True, speed=True, append=done > 0))
            if hamiltonian_identity_record is not None:
                from ..md.phase_space import PhaseSpaceReporter

                # The reservoir consumer compares this fingerprint against the ladder rung it is
                # about to refresh, and a probability-one transfer is only justified when they
                # agree. It is the real identity record -- over the canonical serialised System,
                # the solute selection and the omega exclusions -- taken before stage machinery.
                simulation.reporters.append(PhaseSpaceReporter(
                    str(traj_path.with_suffix(".phase_space.nc")),
                    int(stage["phase_space_interval_steps"]),
                    identity={"hamiltonian": hamiltonian_identity_record},
                    periodic=not implicit, timestep_fs=timestep_fs, step_offset=done))
            if stage.get("checkpoint_interval_steps"):
                simulation.reporters.append(
                    _CheckpointWithFingerprint(str(chk_path), int(stage["checkpoint_interval_steps"]),
                                               fingerprint=fingerprint, offset=done))
            if done == 0 and not args.continue_from and iterations == 0:
                simulation.context.setVelocitiesToTemperature(
                    float(stage["temperature_K"]) * unit.kelvin, int(seed))
            log(f"  integrating {remaining} steps")
            simulation.step(remaining)
        else:
            log("  no dynamics for this stage")

        # -- flush, then PROVE the outputs are there -----------------------------------------
        for reporter in list(simulation.reporters):
            close = getattr(reporter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        simulation.reporters.clear()

        write_final_state(simulation, restart_path)
        simulation.saveCheckpoint(str(chk_path))
        sidecar.write_text(json.dumps({"fingerprint": fingerprint, "steps_done": steps}), "utf-8")

        log.heading("Outputs")
        reread = XmlSerializer.deserialize(restart_path.read_text(encoding="utf-8"))
        if reread.getPositions(asNumpy=True).shape[0] != system.getNumParticles():
            raise SystemExit("the written final state does not match the System; refusing to "
                             "report completion")
        outputs = {"final_state": file_facts(restart_path), "checkpoint": file_facts(chk_path)}
        log.field(restart_path.name, f"{restart_path}  (re-read OK)")
        log.field(chk_path.name, chk_path)
        if traj_path.is_file():
            outputs["trajectory"] = file_facts(traj_path)
            log.field(traj_path.name, traj_path)
        phase_space = traj_path.with_suffix(".phase_space.nc")
        if phase_space.is_file():
            outputs["phase_space"] = file_facts(phase_space)
            log.field(phase_space.name, f"{phase_space}  (positions, velocities and box)")
        log.update(outputs=outputs)
        log.complete()
        log.heading("Summary")
        log(f"  {name}: {steps} steps completed, {steps * timestep_fs / 1000.0:g} ps")
        log("  status: completed")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        print(f"{name}: {exc}", file=sys.stderr)
        return 1

    log.save()
    return 0


class _CheckpointWithFingerprint:
    """A checkpoint reporter that records which configuration each checkpoint belongs to.

    OpenMM's own CheckpointReporter writes only the binary state. Without the sidecar there is no
    way to tell, on resume, whether the checkpoint was produced by this stage's settings or by a
    different run that happened to leave a file with the same name.
    """

    def __init__(self, path: str, interval: int, *, fingerprint: str, offset: int = 0) -> None:
        self._path = path
        self._interval = int(interval)
        self._fingerprint = fingerprint
        self._offset = int(offset)

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        simulation.saveCheckpoint(self._path)
        Path(self._path + ".json").write_text(
            json.dumps({"fingerprint": self._fingerprint,
                        "steps_done": int(simulation.currentStep)}), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# What a generated script calls.
#
# The generated file says which stage it is and where it lives. Everything else -- the resolved
# settings, the plan, the ordering -- comes from `resolved.config` beside it, which is therefore
# the SINGLE declaration of the workflow rather than one of two that can disagree.
# ---------------------------------------------------------------------------------------------

def resolved_config_beside(script: str | Path) -> Path:
    """The `resolved.config` next to a generated script.

    Located from the script's own path, never from the working directory: a generated directory
    must run correctly from anywhere, and `cd`-dependence is how a run silently picks up another
    project's configuration.
    """
    path = Path(script).resolve().parent / "resolved.config"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing. A generated script reads the workflow it belongs to from the "
            f"`resolved.config` beside it; without that file the script cannot know what it was "
            f"generated for. Regenerate the directory with `md-openmm build-md`.")
    return path


def load_generated_plan(script: str | Path) -> tuple[list[dict[str, Any]], Path]:
    """Resolve `resolved.config` strictly and return the ordered stage plan.

    Strictly: the same resolver that refused an unknown key at generation time runs again here, so
    a hand-edited `resolved.config` is refused at execution rather than half-applied.
    """
    from ..build.md import resolve_md_config, stage_plan

    path = resolved_config_beside(script)
    resolved = resolve_md_config(path)
    return stage_plan(resolved), path


def run_generated_stage(script: str | Path, name: str, argv: list[str] | None = None) -> int:
    """Run one named stage of the workflow this script belongs to.

    This is the whole body of a generated stage script:

        from md_tools.md import run_generated_stage
        raise SystemExit(run_generated_stage(__file__, "min"))
    """
    plan, config_path = load_generated_plan(script)
    for stage in plan:
        if stage["name"] == name:
            return stage_main(dict(stage, resolved_config=str(config_path)), argv)
    available = ", ".join(entry["name"] for entry in plan)
    raise SystemExit(
        f"{config_path} describes no stage named {name!r}. It has: {available}. The script and "
        f"the configuration beside it disagree, which means one of them was edited by hand; "
        f"regenerate the directory.")


def run_generated_workflow(script: str | Path, argv: list[str] | None = None) -> int:
    """Run every stage of this workflow in order, in one process.

    The body of a generated `md.py`:

        from md_tools.md import run_generated_workflow
        raise SystemExit(run_generated_workflow(__file__))

    Identical to running the split scripts in order -- same resolved settings, same boundaries,
    same seeds, same logs, same checkpoints, same restart semantics. The only difference is the
    number of processes.
    """
    import argparse

    plan, config_path = load_generated_plan(script)
    parser = argparse.ArgumentParser(description="run every stage of this workflow in order")
    parser.add_argument("-p", "--topology", required=True)
    parser.add_argument("-s", "--system", required=True)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    previous = None
    for stage in plan:
        name = stage["name"]
        stage_argv = ["-p", args.topology, "-s", args.system,
                      "-log", f"{name}.log", "-x", f"{name}.dcd",
                      "-r", f"{name}.xml", "-chk", f"{name}.chk"]
        if previous is not None:
            stage_argv += ["-c", previous]
        if args.platform:
            stage_argv += ["--platform", args.platform]
        if args.device is not None:
            stage_argv += ["--device", str(args.device)]
        if args.check:
            stage_argv += ["--check"]
        code = stage_main(dict(stage, resolved_config=str(config_path)), stage_argv)
        if code != 0:
            print(f"{Path(script).name}: stage {name} failed with exit code {code}",
                  file=sys.stderr)
            return code
        previous = f"{name}.xml"
    return 0


#: `run_stage` is `stage_main` under the name the public API uses. A caller composing a run by
#: hand passes a resolved stage dictionary directly; a generated script goes through the two
#: helpers above, which build that dictionary from `resolved.config`.
run_stage = stage_main
