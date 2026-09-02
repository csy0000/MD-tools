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
from ..openmm.trajectory import describe_trajectory


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



def check_trajectory_suffix(path: Path) -> None:
    """A conventional stage writes DCD. Refuse a name that claims otherwise.

    OpenMM has a native `DCDReporter` and no native NetCDF reporter, so DCD is what an ordinary
    stage can honestly produce. Writing DCD bytes into a file called `.nc` would be worse than
    refusing: every tool downstream would open it as NetCDF, fail, and blame the tool.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".dcd":
        return
    raise SystemExit(
        f"-x {path}: an ordinary MD stage writes DCD, and this name claims {suffix or 'no'} "
        f"format.\n"
        f"  OpenMM has a native DCD writer and no native NetCDF writer, so DCD is what this can "
        f"honestly produce -- and renaming a DCD file does not make it NetCDF.\n"
        f"  Use a .dcd name. AIS paths and REST2 state trajectories ARE genuine NetCDF; those "
        f"are written by their own protocols.")


def stage_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        # No abbreviation. argparse resolves a unique prefix by default, so `--traj` would become
        # `--trajectory` and a misspelling would RUN, with a setting nobody wrote.
        allow_abbrev=False)
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
                        help="the provenance record (machine-readable). Defaults to <stage>.log")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable simulation output, Amber's mdout. A DIFFERENT file "
                             "from -log: this is what you tail while the stage runs. Defaults to "
                             "<stage>.out")
    parser.add_argument("--cpu", action="store_true",
                        help="run this invocation on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform. That setting can also select "
                             "CPU machine-wide; this flag is the per-run override, "
                             "and the record distinguishes the two")
    parser.add_argument("-odir", "--out-dir", default=None, metavar="DIR",
                        help="directory the stage's outputs go in, when they are not named "
                             "individually. The same flag `md-openmm md-run` takes, so a stage "
                             "run directly and one run through the command behave alike")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. An execution PLACEMENT option, not a platform "
                             "choice: it says which GPU, never whether to use one. Rejected with "
                             "--cpu, which has no device to place")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs and settings, then exit without integrating. "
                             "READ-ONLY: it creates nothing, not even the output directory")
    _add_refused_flags(parser, "a cMD stage")
    return parser


def _add_refused_flags(parser, protocol_name: str) -> None:
    """Flags of OTHER protocols, accepted by the parser so the preflight can refuse them by name.

    Leaving them off refuses them too, as "unrecognized arguments" -- which names the flag and
    explains nothing, and puts the rule in argparse rather than in the shared preflight where the
    generated script and `md-run` are guaranteed to agree about it.
    """
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help=f"refused for {protocol_name}: -ng groups replicas of a REST2/rREST2 "
                             f"ladder, and this has one process")
    parser.add_argument("-groupfile", "--groupfile", dest="groupfile", default=None,
                        metavar="FILE",
                        help=f"refused for {protocol_name}: a group file is one line per replica")
    parser.add_argument("-source-traj", "-src", "--source", dest="source_trajectory",
                        default=None, metavar="TRAJ",
                        help=f"refused for {protocol_name}: -source-traj is the equilibrium "
                             f"ensemble AIS draws its starting frames from")


#: Fields a continuation may legitimately change, and which are therefore NOT part of the
#: fingerprint a checkpoint is matched against.
#:
#: `steps` is the whole point: asking for a longer run is an extension, and an extension must be
#: able to resume from the checkpoint the shorter run left. Everything else -- the ensemble, the
#: timestep, the temperature, the restraint, the seed, the System itself -- would make the
#: continuation a different simulation wearing the same file names, so all of it is fingerprinted.
EXTENDABLE_FIELDS = frozenset({"steps", "description"})

#: Keys of the stage dictionary that are PLUMBING rather than settings: they say where this
#: invocation found things, not what the simulation is. Excluded from the fingerprint and from the
#: record, because binding them in would make the same physics fingerprint differently depending
#: on how it was launched -- and `pending_parent` is not even JSON.
NON_SCIENTIFIC_STAGE_KEYS = frozenset({"resolved_config", "pending_parent"})


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
                         if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS},
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


def _completion_gaps(previous: dict[str, Any], *, stage: dict[str, Any], name: str,
                     restart: Path, trajectory: Path) -> str:
    """Why a log claiming `status: completed` may not be believed. Empty string when it may.

    "Completed" is a field in a file, and a field in a file is not a finished run. The three ways
    it lies, in the order they actually happen:

      the log belongs to a DIFFERENT configuration -- a `resolved.config` edited since, or a log
      copied in from another tree -- so the outputs it describes are not the ones this invocation
      would produce;

      the outputs are gone. A log survives an interrupted `rm`, a partial copy, or a scratch
      filesystem that was cleaned; skipping the stage then leaves the NEXT stage to continue from
      a restart that does not exist;

      the log names counts it cannot produce -- an older schema whose fields this build cannot
      compare -- in which case nothing has been verified and saying so is the honest answer.
    """
    recorded = previous.get("fingerprint")
    current = previous.get("stage")
    if recorded is None or not isinstance(current, dict):
        return ("it carries no configuration fingerprint, so nothing about it can be matched to "
                "this stage")
    wanted = {k: v for k, v in sorted(stage.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    theirs = {k: v for k, v in sorted(current.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    differing = sorted(k for k in set(wanted) | set(theirs) if wanted.get(k) != theirs.get(k))
    if differing:
        return (f"it records a different configuration for this stage "
                f"({', '.join(differing)} differ)")

    absent = [str(path) for path in (restart, trajectory) if not Path(path).exists()]
    if absent:
        return f"the output(s) it claims are missing: {', '.join(absent)}"
    return ""


def stage_main(stage: dict[str, Any], argv: list[str] | None = None) -> int:
    """Run one stage. `stage` is the resolved settings the generated script declares."""
    args = stage_parser(stage.get("description", "one MD stage")).parse_args(argv)

    name = stage["name"]
    topology_path = Path(args.topology)
    system_path = Path(args.system)
    base = Path(args.out_dir) if args.out_dir else Path(".")
    log_path = Path(args.log) if args.log else base / f"{name}.log"
    out_path = Path(args.output) if args.output else base / f"{name}.out"
    # The `-o` / `-log` collision is checked by the shared preflight below, together with every
    # other output pair and with the inputs. A second comparison here was a second policy: it
    # compared only those two, missed `-x`, `-r` and `-chk`, and fired first -- so the message a
    # person saw depended on which of the two implementations happened to reach the case.
    traj_path = Path(args.trajectory) if args.trajectory else base / f"{name}.dcd"
    restart_path = Path(args.restart) if args.restart else base / f"{name}.xml"
    chk_path = Path(args.checkpoint) if args.checkpoint else base / f"{name}.chk"

    for path, what in ((topology_path, "-p topology"), (system_path, "-s system")):
        if not path.is_file():
            print(f"{name}: {what} {path} does not exist", file=sys.stderr)
            return 2

    # PREFLIGHT, BEFORE THE FIRST FILESYSTEM MUTATION -- and before the completion check.
    #
    # This runs here, in the runtime, and not only in `md-openmm md-run`: the generated `min.py`
    # and the all-in-one `md.py` call this function directly, so a guard living in the outer
    # command would leave them open. A `.out` and a `.log` created before the machine
    # configuration has been read are a directory that reads as a started run.
    #
    # The "already completed" short-circuit used to sit ABOVE this and return 0 on the strength
    # of one field in a log file. That made a stale or foreign log the authority on whether this
    # stage's outputs exist and belong to this configuration: a `min.log` copied from another
    # tree, or left behind by a run whose `resolved.config` has since changed, ended the command
    # successfully without anything having been verified. It is checked below, after the identity
    # this preflight establishes.
    from ..run.preflight import PreflightError, preflight_stage

    try:
        checked = preflight_stage(
            topology=topology_path, system=system_path, coordinates=args.continue_from,
            trajectory=traj_path, restart=restart_path, checkpoint=chk_path,
            output=out_path, log=log_path, cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None,
            protocol=f"stage {name}",
            pending_parent=stage.get("pending_parent"),
            number_of_groups=args.number_of_groups, groupfile=args.groupfile,
            source_trajectory=args.source_trajectory,
            # The System is deserialised once, inside the preflight, and the timestep is resolved
            # against ITS masses -- so 4 fs on hydrogens that were never repartitioned is refused
            # before the `.out` and the `.log` exist rather than after.
            timestep_fs=stage.get("timestep_fs"),
            ensemble=stage.get("ensemble"), tau=float(stage.get("tau") or 0.0))
    except PreflightError as refusal:
        # No `{name}:` prefix here: `protocol=f"stage {name}"` is already inside the message, and
        # printing both produced "cMD: stage cMD: -c ... does not exist".
        print(f"{refusal}", file=sys.stderr)
        return 2

    if args.check:
        # READ-ONLY, and it returns HERE. `--check` used to create `-odir`, the `.out` and the
        # `.log`, validate, and write `status: checked` -- leaving behind exactly the directory
        # whose absence the caller was trying to confirm. Everything it reported is in the
        # preflight result, so there is nothing left to open a file for.
        from ..run.preflight import report_check

        return report_check(checked, what=f"stage {name}",
                            extra=[("ensemble", stage.get("ensemble") or "-"),
                                   ("steps", stage.get("steps") or 0)])

    # -- already finished? -------------------------------------------------------------------
    # Now, with the inputs validated and this build's view of the stage established.
    if log_path.is_file():
        try:
            previous = read_record(log_path)
        except Exception:
            previous = None
        if previous and previous.get("status") == "completed":
            missing = _completion_gaps(previous, stage=stage, name=name, restart=restart_path,
                                       trajectory=traj_path)
            if missing:
                print(f"{name}: {log_path} says the stage completed, but {missing}. Refusing to "
                      f"treat it as done; delete {log_path} to rerun, or restore the outputs.",
                      file=sys.stderr)
                return 2
            print(f"{name}: already completed ({log_path}); not rerunning. "
                  f"Delete {log_path} to force a rebuild.")
            return 0

    from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter, CheckpointReporter

    from ..md._stages import (add_barostat, add_positional_restraint,
                                              count_barostats, derive_seed, resolve_platform,
                                              set_restraint, write_final_state)

    from ..build.simout import SimulationOutput

    out = SimulationOutput(out_path, title=f"stage {name}", log_path=log_path)
    out.heading("Inputs")
    out.field("topology", topology_path)
    out.field("system", system_path)
    out.field("continue from", args.continue_from or "(none -- first stage)")
    out.field("trajectory", traj_path)
    out.field("final state", restart_path)
    out.field("checkpoint", f"{chk_path.parent / (chk_path.stem + '.checkpoints')}")

    log = LogWriter(log_path, record_type=f"md-stage:{name}")
    log.update(simulation_output=str(out_path))
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
        # The preflight already resolved this, and proved a Context can be created on it. Using
        # its result rather than resolving again is what keeps one platform policy in the package.
        acceleration = checked.acceleration
        integrator = LangevinMiddleIntegrator(
            float(stage["temperature_K"]) * unit.kelvin,
            float(stage["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(int(seed))
        simulation = Simulation(pdb.topology, system, integrator,
                                acceleration.platform, acceleration.properties)
        log.update(acceleration=checked.record())
        log.field("acceleration", f"{acceleration.name} "
                                  f"({checked.record()['requested_policy']})")
        set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))

        system_sha = file_facts(system_path)["sha256"]
        topology_sha = file_facts(topology_path)["sha256"]
        fingerprint = _config_fingerprint(stage, system_sha, topology_sha)
        log.update(stage={k: v for k, v in stage.items() if k != "pending_parent"},
                   fingerprint=fingerprint, implicit=bool(implicit),
                   inputs={"topology": file_facts(topology_path),
                           "system": file_facts(system_path)},
                   derived={"production_ps": steps * timestep_fs / 1000.0,
                            "timestep_fs": timestep_fs, "steps": steps})

        # -- where do we start? -------------------------------------------------------------
        #
        # ONLY the committed pointer, never "the newest checkpoint on disk": the newest file is
        # exactly what a crash leaves behind, and choosing it is how a Context from step 3000 gets
        # paired with bookkeeping from step 2000.
        from ..openmm.checkpoint import CheckpointError, read_committed

        done = 0
        checkpoints = chk_path.parent / f"{chk_path.stem}.checkpoints"
        try:
            committed = read_committed(checkpoints)
        except CheckpointError as broken:
            raise SystemExit(str(broken)) from None
        if committed is not None:
            meta = committed["state"]
            if meta.get("fingerprint") != fingerprint:
                raise SystemExit(
                    f"the committed checkpoint under {checkpoints} was written under a different "
                    f"configuration for this stage (fingerprint mismatch). Resuming it would "
                    f"continue a run that was set up differently. Delete {checkpoints} to start "
                    f"this stage over.")
            try:
                simulation.loadCheckpoint(committed["checkpoint"])
            except Exception as failure:                  # noqa: BLE001 - reported below
                # An OpenMM binary checkpoint is platform-specific, and the exception says so in
                # a way that reads like an internal error. It is not: it means this stage ran on
                # one platform and is being continued on another, which is a fact about the two
                # commands rather than about the simulation.
                raise SystemExit(
                    f"the committed checkpoint cannot be loaded on the {acceleration.name} "
                    f"platform: {failure}\n"
                    f"  An OpenMM checkpoint is binary and platform-specific. Continue this "
                    f"stage on the platform it was written on, or delete {checkpoints} to start "
                    f"the stage over.") from None
            done = int(meta.get("steps_done", 0))
            # Cut every appendable stream back to what the checkpoint VOUCHES for. A trajectory
            # is flushed as it is written, so after a crash it is routinely longer than the
            # checkpoint describing it; keeping those extra frames would put the coordinates
            # permanently ahead of the step count and misattribute every later frame. Never
            # inferred from whichever file happens to be longest.
            trimmed = _truncate_streams_to_committed(meta.get("streams") or {},
                                                     trajectory=traj_path, log=log)
            log.heading("Resume")
            log.field("from checkpoint", f"generation {committed['generation']} at step {done}")
            for name, (was, now) in sorted(trimmed.items()):
                log.field(f"truncated {name}", f"{was} -> {now} (committed)")
        elif args.continue_from:
            parent = Path(args.continue_from)
            if not parent.is_file():
                # Unreachable in practice, and kept as the backstop it is. The preflight above
                # refuses a missing `-c` outright unless the chain declared a `PendingParent`,
                # and `--check` -- the only mode that grants that -- returned before this
                # function opened a single file. Reaching here means a parent vanished between
                # the preflight and now, which is a race worth naming rather than dereferencing.
                raise SystemExit(f"-c {parent} existed at preflight and does not now: something "
                                 f"removed it while this stage was starting. Refusing to "
                                 f"continue from a state that is no longer there.")
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

        # -- do the work --------------------------------------------------------------------
        log.heading("Run")
        out.heading("Run")
        out.field("timestep", f"{timestep_fs} fs")
        out.field("steps", f"{steps} ({steps * timestep_fs / 1000.0:g} ps)")
        out.field("already done", f"{done} (resumed from checkpoint)" if done else "0")
        out.field("ensemble", stage.get("ensemble", "?"))
        out.field("platform", acceleration.name)
        iterations = int(stage.get("minimization_iterations") or 0)
        if iterations:
            log(f"  minimising, {iterations} iterations")
            out.field("minimisation", f"{iterations} iterations")
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
                # The same numbers, readable, in the .out. A separate reporter rather than a
                # post-hoc copy of the CSV: the point of the .out is that it can be tailed while
                # the run is going, and a file written at the end cannot be.
                out.heading("Progress")
                simulation.reporters.append(
                    StateDataReporter(out.handle, int(stage["state_interval_steps"]),
                                      step=True, time=True, potentialEnergy=True,
                                      kineticEnergy=True, totalEnergy=True, temperature=True,
                                      volume=not implicit, density=not implicit, speed=True,
                                      separator="  "))
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
                    _CheckpointWithFingerprint(
                        checkpoints, int(stage["checkpoint_interval_steps"]),
                        fingerprint=fingerprint, offset=done,
                        identity=_checkpoint_identity(stage, name, seed, acceleration,
                                                      timestep_fs),
                        streams={"trajectory": lambda: _stream_counts(
                            trajectory=traj_path).get("trajectory", 0)}))
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
        # The final commit goes through the same transaction as every periodic one, so the last
        # checkpoint of a stage is committed exactly as the intermediate ones are rather than by
        # a second, weaker code path that only ever runs at the end.
        from ..openmm.checkpoint import commit_generation

        commit_generation(
            checkpoints,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={"fingerprint": fingerprint, "steps_done": steps,
                   **_checkpoint_identity(stage, name, seed, acceleration, timestep_fs),
                   "streams": _stream_counts(trajectory=traj_path)})

        log.heading("Outputs")
        reread = XmlSerializer.deserialize(restart_path.read_text(encoding="utf-8"))
        if reread.getPositions(asNumpy=True).shape[0] != system.getNumParticles():
            raise SystemExit("the written final state does not match the System; refusing to "
                             "report completion")
        # The checkpoint is a committed GENERATION now, not one file: the pointer is the thing
        # that says which generation is current, so it is the pointer that is recorded and
        # hashed. Recording the binary alone would name a file whose meaning depends on a second
        # file nobody recorded.
        from ..openmm.checkpoint import POINTER_NAME

        pointer = checkpoints / POINTER_NAME
        outputs = {"final_state": file_facts(restart_path)}
        if pointer.is_file():
            outputs["checkpoint_pointer"] = file_facts(pointer)
            committed_now = read_committed(checkpoints)
            outputs["checkpoint"] = file_facts(Path(committed_now["checkpoint"]))
            log.field("checkpoint", f"{checkpoints} generation {committed_now['generation']}")
        log.field(restart_path.name, f"{restart_path}  (re-read OK)")
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

        out.heading("Outputs")
        out.field("final state", f"{restart_path}  (re-read OK)")
        out.field("checkpoint", checkpoints)
        if traj_path.is_file():
            out.field("trajectory", f"{traj_path}  ({describe_trajectory(traj_path)})")
        out.completed(f"{name}: {steps} steps completed, "
                      f"{steps * timestep_fs / 1000.0:g} ps")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        out.failed(f"{type(exc).__name__}: {exc}")
        print(f"{name}: {exc}", file=sys.stderr)
        return 1

    log.save()
    return 0


def _checkpoint_identity(stage, name, seed, acceleration, timestep_fs) -> dict[str, Any]:
    """What a resume has to agree about beyond the configuration fingerprint.

    The fingerprint already binds the resolved settings and both input digests. These are the
    facts about the RUN rather than about the configuration: which stage this is, which seed it
    drew, what it integrated with, and what it ran on. A checkpoint that matches the fingerprint
    but came from a different stage of the same workflow, or from a different platform, is a
    checkpoint that will load and be wrong -- and the platform case is the one that produces an
    exception rather than a silent error, so it is worth naming before it is loaded.
    """
    return {"stage": name, "seed": int(seed), "timestep_fs": float(timestep_fs),
            "ensemble": stage.get("ensemble"), "platform": acceleration.name,
            "precision": (acceleration.properties or {}).get("Precision")}


def _stream_counts(*, trajectory: Path) -> dict[str, int]:
    """How many records each appendable output holds RIGHT NOW, for the commit to vouch for.

    Read from the files rather than counted in memory: what a resume has to cut back to is what
    is on disk, and an in-memory counter that disagrees with the file is exactly the discrepancy
    the committed counts exist to resolve.
    """
    counts: dict[str, int] = {}
    if Path(trajectory).is_file():
        try:
            from ..openmm.trajectory import count_frames

            counts["trajectory"] = int(count_frames(trajectory))
        except Exception:                                  # noqa: BLE001 - absence is not failure
            pass
    return counts


def _truncate_streams_to_committed(committed: dict[str, Any], *, trajectory: Path,
                                   log) -> dict[str, tuple[int, int]]:
    """Cut each appendable stream back to the count the checkpoint committed.

    Returns `{name: (before, after)}` for whatever actually moved, so the log can say so. A
    stream that is already at or below its committed count is left alone: shorter than committed
    means the crash lost records the checkpoint believes exist, which is a different failure and
    is reported by the caller's own count checks rather than papered over here.
    """
    moved: dict[str, tuple[int, int]] = {}
    wanted = committed.get("trajectory")
    if wanted is None or not Path(trajectory).is_file():
        return moved
    from ..openmm.trajectory import count_frames, truncate_frames

    have = int(count_frames(trajectory))
    if have > int(wanted):
        truncate_frames(trajectory, int(wanted))
        moved["trajectory"] = (have, int(wanted))
    return moved


class _CheckpointWithFingerprint:
    """A checkpoint reporter that COMMITS a generation instead of overwriting a pair.

    OpenMM's own CheckpointReporter writes only the binary state, so this once wrote the state and
    then the sidecar beside it:

        simulation.saveCheckpoint(path)
        Path(path + ".json").write_text(...)

    Two writes, in order, to two names that are only meaningful together. A crash between them --
    or a filesystem that reorders them, which is the ordinary case without an fsync -- leaves a
    NEW Context checkpoint beside an OLD `steps_done`, and the resume continues from step 3000
    while believing it is at 2000. It re-emits frames and rows that already exist and reports a
    stage length that was never run. Nothing fails; the trajectory is simply wrong.

    It is the same defect the AIS path checkpoint had, in the same shape, so it uses the same
    transaction rather than a second implementation of one:
    `md_tools.openmm.checkpoint.commit_generation` writes a new generation, fsyncs it, digests it,
    writes the sidecar, fsyncs the directory, and only then replaces the pointer.

    The committed state binds everything a resume has to agree about -- the configuration
    fingerprint, the step, the stage's identity, the seeds, the platform, and the committed row
    and frame counts of every appendable stream -- so recovery can cut those streams back to what
    the checkpoint vouches for instead of trusting whichever file happens to be longest.
    """

    def __init__(self, directory, interval: int, *, fingerprint: str, offset: int = 0,
                 identity=None, streams=None) -> None:
        self._directory = Path(directory)
        self._interval = int(interval)
        self._fingerprint = fingerprint
        self._offset = int(offset)
        self._identity = dict(identity or {})
        #: name -> callable returning the committed count for that stream, read at commit time.
        self._streams = dict(streams or {})

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        from ..openmm.checkpoint import commit_generation

        commit_generation(
            self._directory,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={"fingerprint": self._fingerprint,
                   "steps_done": int(simulation.currentStep),
                   **self._identity,
                   "streams": {name: int(count()) for name, count in self._streams.items()}})


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
    parser = argparse.ArgumentParser(
        description="run every stage of this workflow in order",
        # No abbreviation, as everywhere else: a misspelling argparse resolves RUNS.
        allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB")
    parser.add_argument("-s", "--system", required=True, metavar="XML")
    parser.add_argument("-c", "--continue-from", default=None, metavar="XML",
                        help="starting state for the FIRST stage; the rest chain from each other")
    parser.add_argument("-odir", "--out-dir", default=None, metavar="DIR",
                        help="directory every stage's outputs go in (default: here)")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable output for the FIRST stage. The others are named "
                             "after themselves: one path cannot describe a whole chain")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="provenance record for the FIRST stage, for the same reason")
    parser.add_argument("--cpu", action="store_true",
                        help="run every stage on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform for this invocation")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. Placement, never platform")
    parser.add_argument("--check", action="store_true",
                        help="validate every stage and exit without integrating. READ-ONLY: the "
                             "whole chain is checked and nothing at all is created")
    _add_refused_flags(parser, "a cMD workflow")
    args = parser.parse_args(argv)

    base = Path(args.out_dir) if args.out_dir else Path(".")

    # THE WHOLE CHAIN IS VALIDATED BEFORE THE FIRST STAGE WRITES ANYTHING. An all-in-one workflow
    # that discovered a broken machine configuration at stage four would have three stages of
    # output on disk and no way to finish -- and `stage_main`'s own preflight, which runs per
    # stage, could not have caught it any earlier than that.
    #
    # `-c` for the FIRST stage is not excluded. It is a file the caller named, produced by
    # something outside this chain, so it must exist even under `--check`; only the parents this
    # chain produces itself are allowed to be missing, and those are stated per stage below.
    from ..run.preflight import PendingParent, PreflightError, preflight_stage

    try:
        preflight_stage(
            topology=args.topology, system=args.system, coordinates=args.continue_from,
            output=args.output or base / f"{plan[0]['name']}.out",
            log=args.log or base / f"{plan[0]['name']}.log",
            cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None,
            protocol=f"the {len(plan)}-stage workflow",
            number_of_groups=args.number_of_groups, groupfile=args.groupfile,
            source_trajectory=args.source_trajectory)
    except PreflightError as refusal:
        print(f"{Path(script).name}: {refusal}", file=sys.stderr)
        return 2

    previous = previous_name = None
    for stage in plan:
        name = stage["name"]
        stage_argv = ["-p", args.topology, "-s", args.system,
                      "-log", str(base / f"{name}.log"), "-x", str(base / f"{name}.dcd"),
                      "-o", str(base / f"{name}.out"),
                      "-r", str(base / f"{name}.xml"), "-chk", str(base / f"{name}.chk")]
        pending = None
        if previous is not None:
            stage_argv += ["-c", previous]
            if args.check:
                # THE one legitimate missing continuation, stated rather than inferred: under
                # `--check` nothing has run, so this parent does not exist yet and the stage that
                # will write it is named here. Outside `--check` no exemption is granted -- if
                # the previous stage really ran, the file is there, and if it is not there the
                # chain must stop rather than silently start this stage from -p.
                pending = PendingParent(path=Path(previous), produced_by=previous_name)
        elif args.continue_from:
            stage_argv += ["-c", args.continue_from]
        if args.device is not None:
            stage_argv += ["--device", str(args.device)]
        if args.cpu:
            stage_argv.append("--cpu")
        if args.check:
            stage_argv += ["--check"]
        code = stage_main(dict(stage, resolved_config=str(config_path),
                               pending_parent=pending), stage_argv)
        if code != 0:
            print(f"{Path(script).name}: stage {name} failed with exit code {code}",
                  file=sys.stderr)
            return code
        previous = str(base / f"{name}.xml")
        previous_name = name
    return 0


#: `run_stage` is `stage_main` under the name the public API uses. A caller composing a run by
#: hand passes a resolved stage dictionary directly; a generated script goes through the two
#: helpers above, which build that dictionary from `resolved.config`.
run_stage = stage_main
