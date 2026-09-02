"""`md-openmm md-run` -- the Amber-like way to execute what `build-md` resolved.

    md-openmm md-run -i min.in -p built.pdb -s built.xml \\
              -x min.dcd -r min.xml -o min.out -log min.log

    mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \\
              -c eq_npt_free.xml -odir REST2/

    mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \\
              -source-traj hot_cmd/production.nc -odir AIS/

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

CUDA IS THE DEFAULT AND IT IS MANDATORY. `--cpu` is the only way to ask for a CPU run, and the
record says that you asked. See `md_tools.openmm.platform_policy`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

__all__ = ["md_run_parser", "md_run_main"]


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
        epilog=(
            "examples:\n"
            "  md-openmm md-run -i min.in -p built.pdb -s built.xml -r min.xml -log min.log\n"
            "  mpirun -n 8 md-openmm md-run -ng 8 -i REST2.in -p built.pdb -s built.xml \\\n"
            "            -c eq_npt_free.xml -odir REST2/\n"
            "  mpirun -n 8 md-openmm md-run -ng 8 -i AIS.in -p built.pdb -s built.xml \\\n"
            "            -source-traj hot_cmd/production.nc -odir AIS/\n"
            "\n"
            "`resolved.config`, written into -odir, is what actually ran. The .in file is the\n"
            "input it was resolved from, and its sha256 is recorded beside it.\n"),
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="the run input: &cntrl / &remd / &AIS sections (Amber's -i mdin)")
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates, built.pdb (Amber's -p prmtop)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System, built.xml. The parameters live here "
                             "rather than in the topology, which is the one place this differs "
                             "from Amber's prmtop")
    parser.add_argument("-c", "--coordinates", default=None, metavar="XML",
                        help="starting state: the final state of the previous stage "
                             "(Amber's -c inpcrd/restrt). Omit for the first stage")
    parser.add_argument("-x", "--trajectory", default=None, metavar="TRAJ",
                        help="output trajectory (Amber's -x mdcrd)")
    parser.add_argument("-r", "--restart", default=None, metavar="XML",
                        help="output final state, the handoff to the next stage "
                             "(Amber's -r restrt)")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="readable run output (Amber's -o mdout). For a ladder this is the "
                             "coordinated run's .out")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="readable log carrying this run's machine record")
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
    parser.add_argument("-source-traj", "--source-traj", dest="source_traj", default=None,
                        metavar="TRAJ",
                        help="for AIS: the equilibrium trajectory the switching paths are drawn "
                             "from. Overrides ais_source.trajectory in the input")
    parser.add_argument("--cpu", action="store_true",
                        help="run on the OpenMM CPU platform. CUDA is the default and is "
                             "mandatory; this is the only way to ask for a CPU run, and the "
                             "record says that you did")
    parser.add_argument("--platform", default=None,
                        help="force a named OpenMM platform. There is no automatic fall back")
    parser.add_argument("--device", default=None, metavar="N", help="CUDA device index")
    parser.add_argument("--check", action="store_true",
                        help="validate the input, the files and the schedule, then exit without "
                             "integrating anything")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing resolved.config in -odir instead of refusing")
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
        "platform": ("--platform", args.platform),
    }
    for name in names:
        flag, value = optional[name]
        if value is not None:
            argv += [flag, str(value)]
    if args.cpu:
        argv.append("--cpu")
    if args.check:
        argv.append("--check")
    return argv


def md_run_main(argv: list[str] | None = None) -> int:
    """Parse, resolve, record, then hand the work to the installed runner for that protocol."""
    from ..build.strict import ConfigError
    from .inputs import parse_run_input

    args = md_run_parser().parse_args(argv)

    try:
        run_input = parse_run_input(args.input, source_trajectory=args.source_traj)
    except ConfigError as invalid:
        print(f"md-run: {invalid}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path = _write_resolved(out_dir, run_input, overwrite=bool(args.overwrite))
    except SystemExit as refusal:
        print(f"md-run: {refusal}", file=sys.stderr)
        return 2

    resolved = run_input.resolved
    protocol = resolved["protocol"]

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
            out_dir=args.out_dir, device=args.device, platform=args.platform,
            cpu=args.cpu, check=args.check)
        code = stage_main(dict(entry, resolved_config=str(config_path)),
                          _forward(forwarded, names=("coordinates", "trajectory", "restart",
                                                     "log", "checkpoint", "device", "platform")))
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
    argv = _forward(args, names=("coordinates", "log", "out_dir", "device", "platform"))
    # replica_main spells the starting state -c, as this command does, and takes -ng itself so
    # the check lives with the ladder rather than being repeated here.
    if args.number_of_groups is not None:
        argv += ["-ng", str(args.number_of_groups)]
    if args.resume:
        argv.append("--resume")
    return replica_main(ladder, argv)


def _run_ais(args, resolved: dict[str, Any], config_path: Path) -> int:
    """AIS switching paths, distributed across the MPI world by global path id."""
    from ..ais.run import ais_main

    run = {
        "protocol": "AIS",
        "description": f"AIS: {resolved['ais']['number_of_paths']} switching paths",
        "ais": dict(resolved["ais"]),
        "ais_source": dict(resolved["ais_source"]),
        "dynamics": dict(resolved["dynamics"]),
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

    argv = _forward(args, names=("log", "out_dir", "device", "platform"))
    source = args.source_traj or resolved["ais_source"]["trajectory"]
    if source:
        argv += ["-source-traj", str(source)]
    return ais_main(run, argv)
