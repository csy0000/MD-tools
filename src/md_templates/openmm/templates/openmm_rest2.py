#!/usr/bin/env python
"""`openmm-rest2` -- the file interface for a generated REST2 ladder.

Amber runs replica exchange as `pmemd.cuda.MPI -ng N -groupfile groups`, where the groupfile names
one input and one output set per replica. OpenMMTools keeps ONE multistate NetCDF for the whole
ladder instead, so there is no groupfile: the replicas share a topology, a base System and a
starting state, and differ only by tau.

    openmm-rest2 -i rest2.py -p topology.pdb -s system.xml -c npt_free.rst.xml \\
                 --solute solute.yaml -o rest2.out -x rest2.nc -r restart.json \\
                 --checkpoint rest2_checkpoint.nc

    mpiexec -n 6 openmm-rest2 ...        one rank per replica

This program decides nothing scientific. The ladder, the temperature, the intervals and the
exchange budget are all visible in the Python protocol file it runs. It resolves paths, refuses to
clobber a finished run, puts the protocol's own directory on `sys.path` so the copied runtime
modules import, runs the protocol with its output captured, and reports what happened.

Standard library and OpenMM only in THIS file: it must still work once `md_templates` is gone. The
protocol it loads is what imports openmmtools.

The protocol contract is the same one function `openmm-md` uses:

    def run(files):        # files.topology, files.system, files.coordinates, files.solute,
        ...                # files.trajectory, files.restart, files.checkpoint,
                           # files.resume, files.extend

`-x` is the authoritative OpenMMTools analysis NetCDF and `-r` is a small manifest that REFERENCES
it. A single XML cannot hold a restart for six replicas, and pretending otherwise is how a resume
comes to start from one replica's state.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

#: Flag, environment fallback, whether the file must already exist, and whether it is required.
SPEC = [
    ("input",       "-i", "--input",       "OPENMM_REST2_INPUT",       True,  True),
    ("topology",    "-p", "--topology",    "OPENMM_REST2_TOPOLOGY",    True,  True),
    ("system",      "-s", "--system",      "OPENMM_REST2_SYSTEM",      True,  True),
    ("coordinates", "-c", "--coordinates", "OPENMM_REST2_COORDINATES", True,  True),
    ("solute",      None, "--solute",      "OPENMM_REST2_SOLUTE",      True,  True),
    ("output",      "-o", "--output",      "OPENMM_REST2_OUTPUT",      False, True),
    ("trajectory",  "-x", "--trajectory",  "OPENMM_REST2_NETCDF",      False, True),
    ("restart",     "-r", "--restart",     "OPENMM_REST2_RESTART",     False, True),
    ("checkpoint",  None, "--checkpoint",  "OPENMM_REST2_CHECKPOINT",  False, False),
]

#: Written by the protocol, never by this program, and only once the manifest exists.
COMPLETION_MARKER = "run_status: completed"

#: Variables an MPI launcher sets in every rank's environment, in the order they are trusted.
#: Read rather than importing mpi4py, because this program must keep working with no MPI at all,
#: and because the rank is needed before anything calls MPI_Init.
MPI_RANK_ENVIRONMENT = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
MPI_SIZE_ENVIRONMENT = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS")


def mpi_rank_and_size(environment=None) -> tuple[int, int]:
    """(rank, size) from the launcher's environment; (0, 1) when not under MPI."""
    environment = os.environ if environment is None else environment
    rank, size = 0, 1
    for name in MPI_RANK_ENVIRONMENT:
        if name in environment:
            try:
                rank = int(environment[name])
                break
            except ValueError:
                pass
    for name in MPI_SIZE_ENVIRONMENT:
        if name in environment:
            try:
                size = int(environment[name])
                break
            except ValueError:
                pass
    return rank, size


def report_path_for_rank(output: str, rank: int) -> str:
    """Rank 0 writes the run's `.out`; every other rank writes its own beside it.

    Without this, six ranks race for one file: the first opens it, and the rest see an output that
    "already exists" and refuse. Each rank's log is kept rather than discarded because a rank that
    failed to bind its GPU is exactly what a multi-GPU run needs to be able to show.
    """
    return output if rank == 0 else f"{output}.rank{rank:02d}"


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="openmm-rest2",
        description="Run one REST2 ladder on openmmtools.multistate.ReplicaExchangeSampler "
                    "against explicit input and output paths.")
    for name, short, long, environment, _must_exist, _required in SPEC:
        flags = [f for f in (short, long) if f]
        parser.add_argument(*flags, dest=name, default=None,
                            help=f"{long.lstrip('-')} (or ${environment})")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in the existing NetCDF storage, which is "
                             "authoritative for where it stopped")
    parser.add_argument("--extend", type=int, default=0, metavar="N",
                        help="add N exchange attempts to a finished run, in place")
    parser.add_argument("--force", action="store_true",
                        help="replace existing outputs of a NEW run instead of refusing; never "
                             "combined with --resume or --extend")
    parser.add_argument("--verify-only", action="store_true", dest="verify_only",
                        help="run nothing: open the analysis and checkpoint NetCDF and report "
                             "whether they are a readable, coherent, complete REST2 run. Needs "
                             "only -x, and optionally --checkpoint and -r.")
    return parser.parse_args(argv)


def resolve(arguments) -> tuple[SimpleNamespace, list[str]]:
    """Command line, then environment, then an error. Nothing is guessed."""
    files, problems = {}, []
    for name, _short, long, environment, must_exist, required in SPEC:
        value = getattr(arguments, name) or os.environ.get(environment)
        if not value:
            if required:
                problems.append(f"no {long} and no ${environment}")
            files[name] = None
            continue
        path = Path(value).expanduser()
        if must_exist and not path.is_file():
            problems.append(f"{long} {path} does not exist")
        files[name] = str(path)
    files["resume"] = bool(arguments.resume)
    files["extend"] = int(arguments.extend or 0)
    return SimpleNamespace(**files), problems


def _outputs(files) -> dict[str, str]:
    return {name: getattr(files, name)
            for name in ("output", "trajectory", "restart", "checkpoint")
            if getattr(files, name)}


def validate(files, *, force: bool, rank: int = 0) -> list[str]:
    """Everything that can be decided without touching OpenMM or writing a byte.

    Only rank 0 checks the shared outputs. The NetCDF, the checkpoint and the manifest belong to
    the run, not to a rank, and every rank applying the "already exists" refusal to them turns a
    normal multi-GPU start into five spurious failures.
    """
    problems: list[str] = []
    continuing = bool(files.resume or files.extend)

    if continuing and force:
        problems.append(
            "--force replaces a new run's outputs and cannot be combined with --resume/--extend, "
            "which continue an existing one. Choose which of the two you mean.")
    if files.extend < 0:
        problems.append(f"--extend must be >= 0; got {files.extend}")

    inputs = {name: getattr(files, name)
              for name in ("input", "topology", "system", "coordinates", "solute")
              if getattr(files, name)}
    outputs = _outputs(files)

    for output_name, output in outputs.items():
        for input_name, value in inputs.items():
            if Path(output).resolve() == Path(value).resolve():
                problems.append(f"--{output_name} and --{input_name} are the same file: {output}")

    seen: dict[Path, str] = {}
    for name, value in outputs.items():
        resolved = Path(value).resolve()
        if resolved in seen:
            problems.append(f"--{name} and --{seen[resolved]} are the same file: {value}")
        seen[resolved] = name

    if continuing:
        # The NetCDF is the authority on where the run stopped. Continuing without it would start
        # a fresh ladder wearing the old run's file names.
        if files.trajectory and not Path(files.trajectory).exists():
            problems.append(
                f"--trajectory {files.trajectory} does not exist, so there is no run to "
                f"{'extend' if files.extend else 'resume'}.")
        # The completion manifest is deliberately NOT required here. A run interrupted before it
        # finished never wrote one, and that is precisely the run that most needs resuming. The
        # scientific identity a continuation is checked against lives in the reporter metadata
        # inside the analysis NetCDF, written before propagation began, with the run-state sidecar
        # as an independent fallback -- so the runtime can establish it without any manifest.
        # Requiring `-r` here is what made an interrupted run unrecoverable.
    elif not force and rank == 0:
        existing = [f"--{name} {value}" for name, value in outputs.items()
                    if Path(value).exists()]
        if existing:
            problems.append(
                "these outputs already exist: " + ", ".join(existing)
                + ". Pass --resume to continue that run, --extend N to lengthen it, or --force "
                  "to replace it deliberately.")
    return problems


def completed_manifest(path) -> bool:
    """True only if `-r` is a manifest that actually claims completion."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return document.get("run_status") == "completed"


def load_protocol(path: str):
    """Import the protocol by location, with its own directory importable.

    The generated project keeps `rest2_runtime.py`, `rest2_openmmtools.py` and `rest2_scaling.py`
    beside the protocol, so the protocol's directory has to be on `sys.path` for
    `from rest2_runtime import REST2` to resolve. Inserting it here rather than in every generated
    protocol keeps the user-facing file down to the science.
    """
    directory = str(Path(path).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("openmm_rest2_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} could not be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise RuntimeError(
            f"{path} defines no run(files). A protocol file is one function taking the resolved "
            f"paths; see the generated REST2 stage for the shape.")
    return module


def verify_only(arguments) -> int:
    """Validate an existing REST2 output. Writes nothing; the exit status is the answer.

    The validator lives beside the storage, in the generated project's REST2 directory, because
    that is the copy that belongs to this run. Importing it from there rather than from an
    installed package keeps `--verify-only` working in a project that has been moved and has no
    `md_templates` anywhere.
    """
    storage = arguments.trajectory or os.environ.get("OPENMM_REST2_NETCDF")
    if not storage:
        print("openmm-rest2: --verify-only needs -x/--trajectory (the analysis NetCDF)",
              file=sys.stderr)
        return 2
    storage_path = Path(storage).expanduser()
    directory = str(storage_path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        import rest2_validate
    except ImportError as failure:
        print(f"openmm-rest2: cannot import rest2_validate from {directory} ({failure}). "
              f"--verify-only reads the copy that belongs to the run being checked.",
              file=sys.stderr)
        return 2

    checkpoint = arguments.checkpoint or os.environ.get("OPENMM_REST2_CHECKPOINT")
    manifest = arguments.restart or os.environ.get("OPENMM_REST2_RESTART")
    result = rest2_validate.validate_rest2_output(
        storage=str(storage_path),
        checkpoint=str(Path(checkpoint).expanduser()) if checkpoint else None,
        manifest=str(Path(manifest).expanduser()) if manifest else None)
    print(rest2_validate.format_report(result))
    return 0 if result.ok else 1


def _has_completion_line(report: Path) -> bool:
    """A whole line saying exactly `run_status: completed`, after stripping whitespace."""
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.strip() == COMPLETION_MARKER:
            return True
    return False


def main(argv=None) -> int:
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    if arguments.verify_only:
        # Checking a finished run needs no protocol, topology, system or coordinates, so the
        # ordinary requirement that all of them be present does not apply here.
        return verify_only(arguments)
    files, problems = resolve(arguments)
    rank, size = mpi_rank_and_size()
    problems += validate(files, force=arguments.force, rank=rank)
    if problems:
        for problem in problems:
            print(f"openmm-rest2: [rank {rank}] {problem}", file=sys.stderr)
        return 2                                   # nothing was created, nothing was read

    for value in _outputs(files).values():
        Path(value).parent.mkdir(parents=True, exist_ok=True)

    report = Path(report_path_for_rank(files.output, rank))
    mode = "a" if (files.resume or files.extend) else "w"
    status = 0
    with open(report, mode, encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                load_protocol(files.input).run(files)
            except SystemExit as exit_request:
                status = int(exit_request.code or 0)
            except BaseException:                          # noqa: BLE001 - the .out is the report
                traceback.print_exc()
                status = 1

    # The protocol prints the marker and writes the manifest. This program insists that neither
    # stands alone: a marker over a missing NetCDF, or a manifest that does not claim completion,
    # is a finished run with nothing to show for it.
    # Only rank 0 writes the shared outputs, so only rank 0 can judge whether they appeared.
    # A non-zero rank that returned without raising has done its share of the propagation.
    if status == 0 and rank == 0:
        promised = {name: value for name, value in _outputs(files).items() if name != "output"}
        missing = [f"--{name} {value}" for name, value in promised.items()
                   if not Path(value).exists()]
        if missing:
            message = ("openmm-rest2: the protocol finished but did not write: "
                       + ", ".join(missing))
            with open(report, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            print(message, file=sys.stderr)
            status = 1
        elif not _has_completion_line(report):
            print(f"openmm-rest2: {report} has no '{COMPLETION_MARKER}' line; "
                  f"treating as incomplete", file=sys.stderr)
            status = 1
        elif not completed_manifest(files.restart):
            print(f"openmm-rest2: {files.restart} does not record run_status: completed; "
                  f"treating as incomplete", file=sys.stderr)
            status = 1

    if status != 0:
        print(f"openmm-rest2: [rank {rank}] REST2 failed; see {report}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
