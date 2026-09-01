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
    parser.add_argument("--platform", default=None,
                        help="force an OpenMM platform (CUDA, OpenCL, CPU, Reference)")
    parser.add_argument("--device", default=None, help="CUDA device index")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs and settings, then exit without integrating")
    return parser


def _config_fingerprint(stage: dict[str, Any], system_sha: str, topology_sha: str) -> str:
    """What a checkpoint has to match before it may be resumed from.

    Deliberately includes the System and topology digests. A checkpoint carries positions and
    velocities for a particular particle set; resuming it against a System that was rebuilt is how
    a run continues with the right-looking numbers and the wrong molecule.
    """
    payload = {"stage": {k: v for k, v in sorted(stage.items()) if k != "description"},
               "system_sha256": system_sha, "topology_sha256": topology_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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

    from ..openmm.templates.md_stages import (add_barostat, add_positional_restraint,
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
        timestep_fs = float(stage["timestep_fs"])
        steps = int(stage.get("steps") or 0)
        log.field("derived time", f"{steps} steps x {timestep_fs} fs = "
                                  f"{steps * timestep_fs / 1000.0:g} ps "
                                  f"({steps * timestep_fs / 1e6:g} ns)")
        if implicit and stage["ensemble"] != "NVT":
            raise SystemExit(
                f"stage {name} declares ensemble {stage['ensemble']}, but the System is not "
                f"periodic. Implicit solvent has no volume to control, so there is no NPT here.")
        log.field("solvent", "implicit (no barostat possible)" if implicit else "explicit")

        # -- the Force layout, fixed before any state is loaded -----------------------------
        restrained = float(stage.get("restraint_kcal_per_mol_A2") or 0.0) > 0.0
        add_positional_restraint(system, pdb.positions, solute)
        if not implicit:
            barostat_active = stage["ensemble"] == "NPT"
            add_barostat(system, float(stage["pressure_bar"]), float(stage["temperature_K"]),
                         derive_seed(int(stage["seed"]), name, "barostat"),
                         frequency=int(stage["barostat_interval_steps"]) if barostat_active else 0)
        if implicit and count_barostats(system):
            raise SystemExit("implicit solvent must carry no barostat")

        platform_name = args.platform or stage.get("platform")
        integrator = LangevinMiddleIntegrator(
            float(stage["temperature_K"]) * unit.kelvin,
            float(stage["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(int(seed))
        if platform_name:
            platform = Platform.getPlatformByName(platform_name)
            properties = {}
            if platform_name == "CUDA":
                properties = {"Precision": "mixed"}
                if args.device is not None:
                    properties["DeviceIndex"] = str(args.device)
            simulation = Simulation(pdb.topology, system, integrator, platform, properties)
        else:
            simulation = Simulation(pdb.topology, system, integrator)
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
                raise SystemExit(f"-c {parent} does not exist: the previous stage writes it only "
                                 f"when it finishes")
            state = XmlSerializer.deserialize(parent.read_text(encoding="utf-8"))
            simulation.context.setState(state)
            set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))
            log.field("started from", parent)
        else:
            simulation.context.setPositions(pdb.positions)
            log.field("started from", f"{topology_path} coordinates")

        log.field("platform", simulation.context.getPlatform().getName())
        log.update(platform=openmm_platform_facts(simulation.context))

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
