"""`md-openmm md-run` -- the Amber-like way to execute what `build-md` resolved.

    md-openmm md-run -i min.in -p built.pdb -s built.xml \\
              -x min.dcd -r min.xml -o min.out -log min.log

    mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \\
              -c eq_npt_free.xml -odir REST2/

    mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \\
              -source-traj ../cMD_tau0p5/tau_0p5.dcd -odir AIS/

Anyone who has run `pmemd -i mdin -p prmtop -c inpcrd -o mdout -x mdcrd -r restrt` can read those
without a manual, which is the whole reason the surface exists. It is a *surface*, not a second
implementation: every one of the three protocols is handed to exactly the same function a
generated script calls -- `stage_main`, `replica_main`, `ais_main`. There is no behaviour reachable
from here that a generated directory cannot reach, and none the other way round.

WHICH FILE IS AUTHORITATIVE

`resolved.config`. Always.

The `.in` file is an *input*: a short, partial, human-written statement of intent. Resolving it
produces `resolved.config` -- every default written out, every cross-field rule applied -- and
that resolved document is what the run reads, what the checkpoint fingerprint binds, and what the
log records. md-run writes it into `-odir` before anything integrates, together with the sha256 of
the `.in` it came from, so the chain from what a person wrote to what actually ran is a link a
reader can follow rather than a claim.

Two consequences worth stating plainly. Editing `resolved.config` between a run and its
continuation changes the fingerprint and the continuation is refused, which is the intent. Editing
the `.in` after the run changes nothing at all -- its digest in the record will simply no longer
match, which is how you find out.

CUDA IS THE DEFAULT AND THERE IS NO FALLBACK. The platform is a property of the machine --
`machine.openmm.platform` in the user configuration -- and `--cpu` overrides it for one
invocation. The record keeps the two apart. See `md_tools.openmm.platform_policy`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

#: Re-exported so the collision rule has one home and one name. It lives in `preflight` with the
#: rest of the checks that must pass before anything is written.
from .preflight import check_output_collisions                # noqa: E402,F401

__all__ = ["md_run_parser", "md_run_main", "check_output_collisions"]


def md_run_parser() -> argparse.ArgumentParser:
    """The flags, spelled the way Amber spells them.

    Nothing here imports OpenMM: `md-openmm md-run -h` must work on a machine with no GPU, no
    driver and no Context, because that is where somebody reads it before submitting a job.
    """
    parser = argparse.ArgumentParser(
        prog="md-openmm md-run",
        description="Run a stage, a replica-exchange ladder or a set of AIS switching paths from "
                    "a short Amber-like input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # No abbreviation. argparse resolves a unique prefix by default, so `--traj` would become
        # `--trajectory` and a misspelling would RUN, with a setting nobody wrote.
        allow_abbrev=False,
        epilog=(
            "examples:\n"
            "  md-openmm md-run -i min.in -p built.pdb -s built.xml -c prev.xml \\\n"
            "            -o min.out -x min.dcd -r min.xml -log min.log\n"
            "  mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \\\n"
            "            -c eq_npt_free.xml -o REST2.out -x REST2.nc -r restart.json \\\n"
            "            -log REST2.log\n"
            "  mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \\\n"
            "            -source-traj ../cMD_tau0p5/tau_0p5.dcd -o AIS.out -log AIS.log \\\n"
            "            -odir ./AIS\n"
            "\n"
            "flags follow Amber: -x is the trajectory (mdcrd) and -o the readable output\n"
            "(mdout). -s is the serialised OpenMM System, which Amber has no counterpart for\n"
            "because its prmtop carries the parameters that live in built.xml here.\n"
            "\n"
            "-o and -log are two files. -o is what you read while a run is going; -log is the\n"
            "provenance record a machine reads. The .in file is resolved on every invocation\n"
            "and the result is written to resolved.config in -odir, with the .in file's\n"
            "sha256, so what ran can always be matched to what was written.\n"),
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="the run input: &cntrl / &remd / &AIS sections (Amber's -i mdin)")
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates, built.pdb (Amber's -p prmtop)")
    parser.add_argument("-s", "--system", dest="system", required=True, metavar="XML",
                        help="serialised OpenMM System, built.xml. Amber has no counterpart: its "
                             "prmtop carries the topology AND the parameters, and here they are "
                             "two files. Required")
    parser.add_argument("-c", "--coordinates", default=None, metavar="XML",
                        help="starting state: the final state of the previous stage "
                             "(Amber's -c inpcrd/restrt). Omit for the first stage")
    parser.add_argument("-r", "--restart", default=None, metavar="XML",
                        help="output final state, the handoff to the next stage "
                             "(Amber's -r restrt)")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable simulation output (Amber's -o mdout): what the run "
                             "is doing, how far it has got, energies where they apply, and how "
                             "it ended. A DIFFERENT file from -log. Defaults to <name>.out")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="the provenance record (MD-data contract): resolved configuration "
                             "identity, input and output hashes, software and hardware, "
                             "warnings, status. Machine-readable. Defaults to <name>.log")
    parser.add_argument("-x", "--trajectory", default=None, metavar="TRAJ",
                        help="output trajectory (Amber's -x mdcrd). Optional: a stage defaults "
                             "to <stage>.dcd and AIS writes AIS_trajNNNN.nc, one per path, which "
                             "no single path could name")
    parser.add_argument("-chk", "--checkpoint", default=None, metavar="CHK",
                        help="output checkpoint, written periodically so the run can resume")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="directory the outputs and resolved.config are written to "
                             "(default: here)")
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help="how many replicas this launch coordinates, for REST2/rREST2. "
                             "Checked against the configured state count and the MPI world size; "
                             "it never resizes the ladder")
    parser.add_argument("-groupfile", "--groupfile", default=None, metavar="FILE",
                        help="for REST2/rREST2: an Amber-style group file, one group per line. "
                             "Rarely needed -- an ordinary homogeneous ladder shares one topology "
                             "and one System, and restating the same two paths N times is a way "
                             "to get one of them wrong")
    parser.add_argument("-source-traj", "--source-traj", dest="source_traj", default=None,
                        metavar="TRAJ",
                        help="for AIS: the equilibrium trajectory the switching paths are drawn "
                             "from. Overrides ais_source.trajectory in the input")
    parser.add_argument("--cpu", action="store_true",
                        help="run this invocation on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform. The only per-run platform override there "
                             "is, and the record says it was asked for on the command line")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. An execution PLACEMENT option: which GPU, never "
                             "whether to use one. Rejected with --cpu, which has no device to "
                             "place. The platform itself is machine.openmm.platform in the user "
                             "configuration -- it is a property of the machine, not of the run")
    parser.add_argument("--check", action="store_true",
                        help="validate the input, the files and the schedule, then exit without "
                             "integrating anything")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the COMPLETE existing output inventory in -odir instead of "
                             "refusing: the reports, the trajectories, the state tables, the "
                             "restarts, the helpers and resolved.config. It governed "
                             "resolved.config alone before, and every other file was replaced "
                             "silently whether it was asked for or not")
    return parser


def _write_resolved(out_dir: Path, run_input, *, overwrite: bool) -> Path:
    """Write `resolved.config` into the output directory and link it to the `.in` by digest.

    Refuses to silently replace a *different* resolved.config that is already there: a directory
    holding outputs from one resolution and the configuration of another cannot be read correctly
    by anyone afterwards. An identical one is left alone, so rerunning the same command is safe.
    """
    import yaml

    digest = hashlib.sha256(run_input.path.read_bytes()).hexdigest()
    document = yaml.safe_dump(run_input.resolved, sort_keys=False, default_flow_style=False)
    header = (
        f"# The configuration this run resolved to, in full. THIS FILE IS AUTHORITATIVE:\n"
        f"# it is what the run reads, what the checkpoint fingerprint binds, and what the log\n"
        f"# records. The input below is where it came from, and nothing reads it again.\n"
        f"#\n"
        f"#   input      : {run_input.path.name}\n"
        f"#   sha256     : {digest}\n"
        f"#   sections   : {', '.join('&' + s for s in run_input.sections)}\n")

    path = out_dir / "resolved.config"
    if path.is_file() and not overwrite:
        # Only the resolved document is compared. Two inputs that resolve to the same run ARE the
        # same run, whatever their comments and formatting, and rerunning a command must not be
        # refused for a reason invisible in what actually executes.
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != run_input.resolved:
            raise SystemExit(
                f"{path} already exists and describes a different run. A directory holding "
                f"outputs from one resolved configuration and a second, different one cannot be "
                f"read correctly afterwards. Use a different -odir, or --overwrite if the "
                f"outputs there are meant to be replaced.")
        return path
    path.write_text(header + document, encoding="utf-8")
    return path


def _forward(args, *, names) -> list[str]:
    """The subset of this command line the delegated runner accepts, in its own spelling."""
    argv: list[str] = ["-p", args.topology, "-s", args.system]
    optional = {
        "trajectory": ("-x", args.trajectory),
        "restart": ("-r", args.restart),
        "log": ("-log", args.log),
        "checkpoint": ("-chk", args.checkpoint),
        "coordinates": ("-c", args.coordinates),
        "out_dir": ("-odir", args.out_dir),
        "device": ("--device", args.device),
        "output": ("-o", args.output),
    }
    for name in names:
        flag, value = optional[name]
        if value is not None:
            argv += [flag, str(value)]
    if args.cpu:
        argv.append("--cpu")
    if args.check:
        argv.append("--check")
    # `--resume` and `--overwrite` are the runtime's contract, so they are forwarded rather than
    # interpreted here. A flag that reached `md-run` and stopped there is a flag the person
    # believes took effect: `--resume` silently restarted every stage from the beginning.
    if getattr(args, "resume", False):
        argv.append("--resume")
    if getattr(args, "overwrite", False):
        argv.append("--overwrite")
    return argv


def _check_file_roles(args) -> None:
    """Every path argument must be the KIND of thing its flag names.

    Each of these was a real mistake waiting to be made rather than a hypothetical one. `-x` used
    to mean the serialised System on this surface, so `-x built.xml` is what the old
    documentation taught people to type and it must not now be read as "write the trajectory to
    built.xml", which would destroy the System. The others are the same mistake mirrored.
    """
    system = Path(args.system)
    if system.suffix.lower() in (".dcd", ".nc", ".netcdf", ".mdcrd"):
        raise SystemExit(
            f"-s {args.system} looks like a trajectory. `-s` is the serialised OpenMM System "
            f"(built.xml); `-x` is the output trajectory, as in Amber.")

    if args.trajectory:
        trajectory = Path(args.trajectory)
        if trajectory.suffix.lower() == ".xml":
            raise SystemExit(
                f"-x {args.trajectory} looks like a serialised System or state. `-x` is the "
                f"output TRAJECTORY, as in Amber's mdcrd; use `-s` for the serialized System.\n"
                f"  This surface briefly used `-x` for built.xml. It does not any more: writing "
                f"a trajectory over built.xml would destroy the System the run needs.")
        if trajectory.resolve() == system.resolve():
            raise SystemExit(
                f"-x and -s are the same path ({args.trajectory}). The trajectory would be "
                f"written over the System.")

    if args.source_traj:
        # Checked HERE rather than inside the AIS runtime, because it is a check on a file the
        # command line named and it must run before -odir or resolved.config exist. A source whose
        # suffix and contents disagree is refused with both named.
        from ..openmm.trajectory import check_trajectory_declaration

        check_trajectory_declaration(args.source_traj, what="-source-traj")

    # `-o` and `-log` are two artefacts for two readers. One file cannot be both, so an actual
    # collision is refused -- and ONLY a collision: different paths are the normal case.
    if args.output and args.log and Path(args.output) == Path(args.log):
        raise SystemExit(
            f"-o and -log both name {args.output}. They are different files: -o is the "
            f"human-readable simulation output you read while a run is going, and -log is the "
            f"machine-readable provenance record. One file cannot be both without a person "
            f"having to read a machine record to see progress, which is the thing this "
            f"separation exists to avoid.")


def md_run_main(argv: list[str] | None = None) -> int:
    """Parse, resolve, record, then hand the work to the installed runner for that protocol."""
    from ..build.strict import ConfigError
    from .inputs import parse_run_input

    args = md_run_parser().parse_args(argv)

    # PREFLIGHT, all of it, before ANY output exists. `-odir`, `resolved.config`, the `.out`,
    # the `.log`, a group file, a trajectory, a checkpoint -- none of them may be created until
    # every check below has passed. A `-odir` holding a `resolved.config` is indistinguishable
    # from a run that happened, and the next person to look will read it as one.
    #
    # The checks themselves live in `md_tools.run.preflight`, shared with `stage_main`,
    # `replica_main` and `ais_main`, because the generated wrappers call THOSE directly. A guard
    # that lives only here is a property of one entry point rather than of the runtime.
    try:
        _check_file_roles(args)
    except SystemExit as refusal:
        print(f"md-run: {refusal}", file=sys.stderr)
        return 2

    try:
        run_input = parse_run_input(args.input, source_trajectory=args.source_traj)
    except ConfigError as invalid:
        print(f"md-run: {invalid}", file=sys.stderr)
        return 2

    resolved = run_input.resolved
    protocol = resolved["protocol"]
    replicas = (int(resolved["rest2"]["number_of_replicas"])
                if protocol in ("REST2", "rREST2") else None)
    source = args.source_traj or (resolved["ais_source"]["trajectory"]
                                  if protocol == "AIS" else None)

    # The SAME mode-aware preflight the runtimes run for themselves. `md-run` does it here so a
    # refusal happens before `-odir` and `resolved.config` exist; the runtime does it again for
    # its own arguments, which is cheap and is what makes a generated script as safe as this one.
    from .preflight import (PreflightError, preflight_ais, preflight_ladder, preflight_stage)

    from .preflight import check_existing_outputs

    try:
        checked = None
        if protocol == "AIS":
            checked = preflight_ais(
                topology=args.topology, system=args.system, source=source,
                number_of_groups=args.number_of_groups,
                output=args.output, log=args.log, cpu=bool(args.cpu),
                device=int(args.device) if args.device is not None else None)
        elif protocol in ("REST2", "rREST2") and run_input.stage is None:
            checked = preflight_ladder(
                topology=args.topology, system=args.system, replicas=replicas,
                coordinates=args.coordinates, groupfile=args.groupfile,
                trajectory=args.trajectory, restart=args.restart,
                output=args.output, log=args.log,
                number_of_groups=args.number_of_groups, cpu=bool(args.cpu),
                device=int(args.device) if args.device is not None else None,
                protocol=protocol)
        else:
            checked = preflight_stage(
                topology=args.topology, system=args.system, coordinates=args.coordinates,
                trajectory=args.trajectory, restart=args.restart, checkpoint=args.checkpoint,
                output=args.output, log=args.log, cpu=bool(args.cpu),
                device=int(args.device) if args.device is not None else None,
                protocol=protocol)

        # The COMPLETE inventory, not `resolved.config` alone. An `-odir` that already holds a
        # run is a run that happened; writing into it leaves a tree that is half one run and half
        # another, with every file looking equally current.
        if checked is not None and checked.inventory is not None:
            check_existing_outputs(checked.inventory, overwrite=bool(args.overwrite),
                                   resume=bool(args.resume) or bool(args.check),
                                   where=f"md-run {protocol}")
    except PreflightError as refusal:
        print(f"md-run: {refusal}", file=sys.stderr)
        return 2

    # Only now. Everything above touched nothing.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path = _write_resolved(out_dir, run_input, overwrite=bool(args.overwrite))
    except SystemExit as refusal:
        print(f"md-run: {refusal}", file=sys.stderr)
        return 2

    # `stage` decides, not `protocol`. A REST2 workflow's minimisation and equilibration are
    # ordinary stages of that workflow: an input that names one asks for that stage, and only an
    # input that names none asks for the ladder itself. Dispatching on the protocol alone sent
    # `min.in` to the replica executor, which had no coordinates and refused.
    if run_input.stage is not None:
        return _run_stages(args, resolved, run_input.stage, config_path)
    if protocol in ("REST2", "rREST2"):
        return _run_ladder(args, resolved, protocol, config_path)
    if protocol == "AIS":
        return _run_ais(args, resolved, config_path)
    return _run_stages(args, resolved, None, config_path)


# ---------------------------------------------------------------------------------------------
# the three protocols, each handed to the function a generated script would have called
# ---------------------------------------------------------------------------------------------

def _run_stages(args, resolved: dict[str, Any], stage: str | None, config_path: Path) -> int:
    """One named stage, or the whole workflow in order when the input names none."""
    from ..build.md import stage_plan
    from ..md.stage import stage_main

    plan = stage_plan(resolved)
    names = [entry["name"] for entry in plan]
    if stage is not None and stage not in names:
        print(f"md-run: {args.input} asks for stage {stage!r}, which this workflow does not have. "
              f"It has: {', '.join(names)}.", file=sys.stderr)
        return 2

    chosen = [entry for entry in plan if stage is None or entry["name"] == stage]
    out_dir = Path(args.out_dir)
    previous = args.coordinates
    for entry in chosen:
        name = entry["name"]
        # Explicit output names win when this command runs ONE stage. When the input names no
        # stage it runs the whole chain, and a single -x/-r/-log could only describe one of them,
        # so each stage is named after itself -- exactly as the generated split scripts are.
        single = len(chosen) == 1
        forwarded = argparse.Namespace(
            topology=args.topology, system=args.system, coordinates=previous,
            trajectory=args.trajectory if single and args.trajectory
                       else str(out_dir / f"{name}.dcd"),
            restart=args.restart if single and args.restart else str(out_dir / f"{name}.xml"),
            log=args.log if single and args.log else str(out_dir / f"{name}.log"),
            checkpoint=args.checkpoint if single and args.checkpoint
                       else str(out_dir / f"{name}.chk"),
            output=args.output if single and args.output else str(out_dir / f"{name}.out"),
            out_dir=args.out_dir, device=args.device,
            cpu=args.cpu, check=args.check,
            # Carried explicitly. This namespace is built fresh rather than passed through, so a
            # flag that is not listed here does not reach the stage at all -- which is how
            # `--resume` and `--overwrite` were accepted by `md-run` and silently dropped.
            resume=args.resume, overwrite=args.overwrite)
        code = stage_main(dict(entry, resolved_config=str(config_path)),
                          _forward(forwarded, names=("coordinates", "trajectory", "restart",
                                                     "log", "output", "checkpoint", "device")))
        if code != 0:
            print(f"md-run: stage {name} failed with exit code {code}", file=sys.stderr)
            return code
        previous = forwarded.restart
    return 0


def _run_ladder(args, resolved: dict[str, Any], protocol: str, config_path: Path) -> int:
    """A REST2 or rREST2 ladder: one process per state, `-ng` checked against both."""
    from ..remd.generated import ladder_from_resolved, replica_main

    ladder = ladder_from_resolved(resolved, protocol)
    ladder["resolved_config"] = str(config_path)
    argv = _forward(args, names=("coordinates", "log", "output", "out_dir", "device"))
    if args.restart:
        argv += ["-r", str(args.restart)]
    # replica_main spells the starting state -c, as this command does, and takes -ng itself so
    # the check lives with the ladder rather than being repeated here.
    if args.number_of_groups is not None:
        argv += ["-ng", str(args.number_of_groups)]
    if args.trajectory:
        argv += ["-x", str(args.trajectory)]
    if args.groupfile:
        argv += ["--groupfile", str(args.groupfile)]
    if args.resume:
        argv.append("--resume")
    return replica_main(ladder, argv)


def _run_ais(args, resolved: dict[str, Any], config_path: Path) -> int:
    """AIS switching paths, distributed across the MPI world by global path id."""
    from ..ais.run import ais_main

    if args.trajectory:
        # One name cannot describe N files, and pretending it can is how a "prefix" option grows
        # a meaning nobody tested. AIS writes AIS_trajNNNN.nc, one per global path id, into -odir.
        print(f"md-run: -x {args.trajectory} names one trajectory, but AIS writes one per path -- "
              f"AIS_traj0000.nc .. into -odir. Omit -x and use -odir to say where they go.",
              file=sys.stderr)
        return 2

    run = {
        "protocol": "AIS",
        "description": f"AIS: {resolved['ais']['number_of_paths']} switching paths",
        "ais": dict(resolved["ais"]),
        "ais_source": dict(resolved["ais_source"]),
        "dynamics": dict(resolved["dynamics"]),
        "reporting": dict(resolved["reporting"]),
        "resolved_config": str(config_path),
    }
    # `-ng` means the same thing here as for a ladder: how many workers this launch coordinates.
    # It is checked rather than ignored -- a flag that silently does nothing is worse than one
    # that is refused, because the person who wrote it believes it took effect.
    if args.number_of_groups is not None:
        from ..remd.executor import mpi_rank_and_size

        _, size = mpi_rank_and_size()
        if int(args.number_of_groups) != size:
            print(f"md-run: -ng {args.number_of_groups} was requested but this launch has "
                  f"{size} worker(s). For AIS, -ng is how many processes share the paths:\n"
                  f"  mpirun -n {args.number_of_groups} md-openmm md-run "
                  f"-ng {args.number_of_groups} ...\n"
                  f"Which global path owns which AIS_trajNNNN.nc does not depend on this number; "
                  f"how many run at once does.", file=sys.stderr)
            return 2

    argv = _forward(args, names=("log", "output", "out_dir", "device"))
    if args.resume:
        # Without this, `--resume` reached md-run and stopped there: every interrupted path
        # silently restarted from its source frame while the command reported success, which is
        # the failure mode a checkpoint exists to prevent.
        argv.append("--resume")
    source = args.source_traj or resolved["ais_source"]["trajectory"]
    if source:
        argv += ["-source-traj", str(source)]
    return ais_main(run, argv)
