#!/usr/bin/env python
"""`openmm-md` -- the ONE simulation executor for a generated project.

Amber has `pmemd -i in -p prmtop -c rst -o out -x nc -r rst` for a single system and
`pmemd.MPI -ng N -groupfile groups` for a coordinated set of them. This is the same idea with the
same executable:

    openmm-md -i INPUT.py -p topology.pdb -s system.xml -c start.xml \\
              -o stage.out -x trajectory.dcd -r final_state.xml --checkpoint stage.chk

    mpiexec -n 6 openmm-md -ng 6 --groupfile REST2/rest2.group \\
              -o REST2/rest2.out -x REST2/rest2.nc -r REST2/restart.json \\
              --checkpoint REST2/rest2_checkpoint.nc

    mpiexec -n 6 openmm-md -ng 6 --groupfile rREST2/rrest2.group \\
              --exchange-rule rREST2/rrest2_exchange.py --reservoir rREST2/reservoir.yaml \\
              -o rREST2/rrest2.out -x rREST2/rrest2.nc -r rREST2/restart.json \\
              --checkpoint rREST2/rrest2_checkpoint.nc

There is no second executable. A method is a protocol file plus, when the transitions differ, an
exchange-rule file -- not a new command, a new flag namespace and a new set of help text to keep
consistent with this one.

WHAT THIS PROGRAM DECIDES
    Paths, group parsing, mode validation, MPI coordination, output refusal, and which protocol
    and rule files to load. Nothing scientific: no force field, no ladder, no temperature, no step
    count, no reservoir policy. Those are visible in the Python protocol it runs.

THE GROUP FILE
    Plain text, one group per line, parsed with `shlex`. It is NEVER evaluated by a shell: a
    group file is data, and a data file that can run commands is a vulnerability rather than a
    convenience. Blank lines and lines beginning with `#` are ignored.

        -i REST2/rest2.py -p common/topology.pdb -s common/system.xml \\
           -c eq/npt_free/npt_free.state.xml --group-index 0

    Group lines carry INPUTS only. `-o`, `-x`, `-r` and `--checkpoint` appear once on the outer
    command because they describe the coordinated run, not one replica of it.

Standard library and OpenMM only in the single-run path, so a generated stage still runs once this
package is gone. Grouped mode additionally imports the small runtime modules copied beside the
protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shlex
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

#: Flag, environment fallback, whether the file must already exist, and whether it is required.
SPEC = [
    ("input",       "-i", "--input",       "OPENMM_INPUT",       True,  True),
    ("topology",    "-p", "--topology",    "OPENMM_TOPOLOGY",    True,  True),
    ("system",      "-s", "--system",      "OPENMM_SYSTEM",      True,  True),
    ("coordinates", "-c", "--coordinates", "OPENMM_COORDINATES", True,  True),
    ("output",      "-o", "--output",      "OPENMM_OUTPUT",      False, True),
    ("trajectory",  "-x", "--trajectory",  "OPENMM_TRAJECTORY",  False, False),
    ("checkpoint",  None, "--checkpoint",  "OPENMM_CHECKPOINT",  False, False),
    ("restart",     "-r", "--restart",     "OPENMM_RESTART",     False, True),
    ("solute_x",    None, "--solute-x",    "OPENMM_SOLUTE_X",    False, False),
    ("rem",         None, "--rem",         "OPENMM_REM",         False, False),
]

#: What a GROUP line may carry. Inputs only, plus its index. Anything else is refused by name.
GROUP_FIELDS = {
    "-i": "input", "--input": "input",
    "-p": "topology", "--topology": "topology",
    "-s": "system", "--system": "system",
    "-c": "coordinates", "--coordinates": "coordinates",
    "--solute": "solute",
    "--group-index": "group_index",
}

#: Flags that describe the coordinated RUN and therefore may not appear on a group line.
RUN_LEVEL_FLAGS = {"-o", "--output", "-x", "--trajectory", "-r", "--restart", "--checkpoint",
                   "-ng", "--groupfile", "--exchange-rule", "--reservoir"}

#: Written by the protocol, never by this program, and only once its outputs exist.
COMPLETION_MARKER = "run_status: completed"
# A run stopped on purpose at an event boundary: not success, and not a failure.
INTERRUPTED_STATUS = 130

#: Variables an MPI launcher sets in every rank's environment, read rather than importing mpi4py:
#: the rank is needed before anything calls MPI_Init, and single mode must work with no MPI at all.
MPI_RANK_ENVIRONMENT = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
MPI_SIZE_ENVIRONMENT = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS")


class GroupFileError(ValueError):
    """The group file cannot be understood. Always names the line."""


def mpi_rank_and_size(environment=None):
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


def report_path_for_rank(output, rank):
    """Rank 0 writes the run's `.out`; every other rank writes its own beside it.

    Without this, N ranks race for one file: the first opens it and the rest see an output that
    "already exists" and refuse. Each rank's log is kept rather than discarded, because a rank that
    failed to bind its device is exactly what a multi-GPU run needs to be able to show.
    """
    return output if rank == 0 else f"{output}.rank{rank:02d}"


# --- the group file ----------------------------------------------------------------------------

def parse_group_file(path):
    """One group per line, parsed with `shlex`. Never evaluated by a shell.

    Returns a list of dictionaries in file order. Every failure names the line number, because a
    group file is written by hand often enough that "somewhere in this file" is not good enough.
    """
    path = Path(path)
    if not path.is_file():
        raise GroupFileError(f"--groupfile {path} does not exist")
    groups = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as failure:
            raise GroupFileError(f"{path}:{number}: cannot be parsed ({failure})") from None
        if not tokens:
            continue
        groups.append(_parse_group_line(path, number, tokens))
    if not groups:
        raise GroupFileError(f"{path} contains no group lines")
    _check_group_indices(path, groups)
    return groups


def _parse_group_line(path, number, tokens):
    group = {"line": number}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in RUN_LEVEL_FLAGS:
            raise GroupFileError(
                f"{path}:{number}: {token} describes the coordinated run and belongs on the outer "
                f"openmm-md command, not on a group line. A per-group output would give every "
                f"replica its own file and there would be no single authoritative record.")
        if token not in GROUP_FIELDS:
            raise GroupFileError(
                f"{path}:{number}: unknown group field {token!r}. A group line carries inputs "
                f"only: {sorted(set(GROUP_FIELDS))}.")
        field = GROUP_FIELDS[token]
        if field in group:
            raise GroupFileError(f"{path}:{number}: {field} given more than once")
        if index + 1 >= len(tokens):
            raise GroupFileError(f"{path}:{number}: {token} has no value")
        value = tokens[index + 1]
        if value in GROUP_FIELDS or value in RUN_LEVEL_FLAGS:
            raise GroupFileError(
                f"{path}:{number}: {token} has no value (followed by {value!r})")
        group[field] = value
        index += 2
    for required in ("input", "topology", "system", "coordinates"):
        if required not in group:
            raise GroupFileError(f"{path}:{number}: no {required} given")
    if "group_index" not in group:
        raise GroupFileError(
            f"{path}:{number}: no --group-index. Group order in the file is not the assignment; "
            f"the index is stated so a reordered file still means the same thing.")
    try:
        group["group_index"] = int(group["group_index"])
    except ValueError:
        raise GroupFileError(
            f"{path}:{number}: --group-index {group['group_index']!r} is not an integer") from None
    return group


def _check_group_indices(path, groups):
    indices = [group["group_index"] for group in groups]
    if sorted(indices) != list(range(len(groups))):
        raise GroupFileError(
            f"{path}: group indices must be unique, contiguous and zero-based; got "
            f"{sorted(indices)} for {len(groups)} group(s).")


# --- the command line -----------------------------------------------------------------------------

def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="openmm-md",
        description="Run one OpenMM protocol against explicit paths, or a coordinated set of "
                    "replicas from an Amber-like group file.")
    for name, short, long, environment, _must_exist, _required in SPEC:
        flags = [f for f in (short, long) if f]
        parser.add_argument(*flags, dest=name, default=None,
                            help=f"{long.lstrip('-')} (or ${environment})")
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None,
                        help="number of replica groups; required with --groupfile")
    parser.add_argument("--groupfile", dest="groupfile", default=None,
                        help="Amber-like group file, one replica per line; selects grouped mode")
    parser.add_argument("--exchange-rule", dest="exchange_rule", default=None,
                        help="a Python file defining the transition rule; the built-in "
                             "neighbouring REST2 rule is used when omitted")
    parser.add_argument("--reservoir", dest="reservoir", default=None,
                        help="a prepared reservoir manifest, for rules that need one")
    parser.add_argument("--resume", action="store_true",
                        help="GROUPED ONLY: continue a coordinated run that stopped short of its "
                             "budget, in place")
    parser.add_argument("--extend", type=int, default=0, metavar="N",
                        help="GROUPED ONLY: add N exchange attempts to a coordinated run that "
                             "reached its budget")
    parser.add_argument("--force", action="store_true",
                        help="replace existing outputs of a NEW run instead of refusing. This is "
                             "not continuation: it starts over")
    parser.add_argument("--verify-only", action="store_true", dest="verify_only",
                        help="run nothing: open the stored output and report whether it is a "
                             "readable, coherent, complete run")
    return parser.parse_args(argv)


def resolve(arguments):
    """Command line, then environment, then an error. Nothing is guessed."""
    files, problems = {}, []
    grouped = bool(arguments.groupfile)
    for name, _short, long, environment, must_exist, required in SPEC:
        value = getattr(arguments, name) or os.environ.get(environment)
        if not value:
            # In grouped mode the per-replica inputs come from the group file, so the run-level
            # command needs only the outputs.
            if required and not (grouped and name in ("input", "topology", "system",
                                                      "coordinates")):
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


def _outputs(files):
    return {name: getattr(files, name)
            for name in ("output", "trajectory", "restart", "checkpoint", "solute_x")
            if getattr(files, name)}


def validate(files, arguments, *, rank=0, groups=None):
    """Everything decidable without touching OpenMM or writing a byte."""
    problems = []
    grouped = bool(arguments.groupfile)
    continuing = bool(files.resume or files.extend)

    # -- mode ---------------------------------------------------------------------------------
    if grouped and arguments.number_of_groups is None:
        problems.append("--groupfile requires -ng: the number of groups is stated so a truncated "
                        "group file cannot silently run a shorter ladder")
    if arguments.number_of_groups is not None and not grouped:
        problems.append("-ng describes a group file and has no meaning without --groupfile")
    if not grouped:
        for flag, value in (("--exchange-rule", arguments.exchange_rule),
                            ("--reservoir", arguments.reservoir),
                            ("--rem", getattr(files, "rem", None))):
            if value:
                problems.append(f"{flag} applies to a coordinated run and needs --groupfile")
    if grouped and groups is not None:
        if arguments.number_of_groups is not None and len(groups) != arguments.number_of_groups:
            problems.append(
                f"-ng {arguments.number_of_groups} but the group file has {len(groups)} group "
                f"line(s)")
        _, size = mpi_rank_and_size()
        if size > 1 and size != len(groups):
            problems.append(
                f"MPI world size is {size} but there are {len(groups)} group(s). The supported "
                f"policy is one process for the whole ladder or exactly one rank per group; "
                f"packing several groups onto a rank is refused rather than done silently.")
    if grouped and files.trajectory is None:
        problems.append("-x is required in grouped mode: the analysis NetCDF is the authoritative "
                        "record of a coordinated run")

    # -- continuation is a GROUPED capability ---------------------------------------------------
    # Only the replica runtime implements append-safe continuation: it rewinds uncommitted rows,
    # validates the stored output before opening it for writing, and reopens the analysis NetCDF
    # in append mode on rank 0. The conventional single-protocol path does none of that -- it
    # builds a fresh Simulation, resets time and step state, and creates its reporters from
    # scratch -- so `--resume` there promised something nothing implements, and would have
    # rewritten the DCD, checkpoint and phase-space outputs of the run it claimed to continue.
    #
    # Refused here, in the pure validation pass, so nothing is created, opened or imported first.
    if continuing and not grouped:
        flag = "--extend" if files.extend else "--resume"
        problems.append(
            f"{flag} is supported only with --groupfile. Append-safe continuation of a "
            f"conventional cMD stage is not implemented: that path starts a fresh run and would "
            f"write over the outputs of the one you meant to continue, so it is refused rather "
            f"than approximated.\n"
            f"  Run the stage into a new or empty output location instead, or wait for a "
            f"dedicated continuation implementation. `--force` is a deliberate fresh-run "
            f"replacement and is NOT continuation: it starts over rather than carrying anything "
            f"forward.")

    if continuing and arguments.force:
        problems.append("--force replaces a new run's outputs and cannot be combined with "
                        "--resume/--extend, which continue an existing one")
    if files.extend < 0:
        problems.append(f"--extend must be >= 0; got {files.extend}")

    # -- paths ---------------------------------------------------------------------------------
    inputs = {name: getattr(files, name)
              for name in ("input", "topology", "system", "coordinates") if getattr(files, name)}
    if grouped and groups:
        for group in groups:
            for field in ("input", "topology", "system", "coordinates", "solute"):
                if group.get(field):
                    inputs[f"group{group['group_index']}.{field}"] = group[field]
    outputs = _outputs(files)

    for output_name, output in outputs.items():
        for input_name, value in inputs.items():
            try:
                same = Path(output).resolve() == Path(value).resolve()
            except OSError:
                same = False
            if same:
                problems.append(f"--{output_name} and {input_name} are the same file: {output}")

    seen = {}
    for name, value in outputs.items():
        resolved = Path(value).resolve()
        if resolved in seen:
            problems.append(f"--{name} and --{seen[resolved]} are the same file: {value}")
        seen[resolved] = name

    if continuing:
        if files.trajectory and not Path(files.trajectory).exists():
            problems.append(
                f"--trajectory {files.trajectory} does not exist, so there is no run to "
                f"{'extend' if files.extend else 'resume'}")
        # The completion manifest is deliberately NOT required: a run interrupted before it
        # finished never wrote one, and that is exactly the run that needs resuming. The identity
        # a continuation is checked against lives inside the analysis storage.
    elif not arguments.force and rank == 0:
        existing = [f"--{name} {value}" for name, value in outputs.items()
                    if Path(value).exists()]
        if existing:
            problems.append(
                "these outputs already exist: " + ", ".join(existing)
                + ". Pass --resume to continue that run, --extend N to lengthen it, or --force to "
                  "replace it deliberately.")
    return problems


# --- loading protocol files ---------------------------------------------------------------------

def load_module(path, name):
    """Import a file by location, with its own directory importable.

    A generated project keeps its runtime modules beside the protocol, so the protocol's directory
    goes on `sys.path` -- which is what lets `from replica_runtime import REST2Protocol` resolve in
    a project that has been moved and has no `md_templates` anywhere.
    """
    path = Path(path)
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} could not be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_protocol(path):
    """A single-run protocol file: one `run(files)` function."""
    module = load_module(path, "openmm_md_protocol")
    if not hasattr(module, "run"):
        raise RuntimeError(
            f"{path} defines no run(files). A single-run protocol file is one function taking the "
            f"resolved paths; see the generated stages for the shape.")
    return module


def load_grouped_protocol(path):
    """A grouped protocol file: one `protocol` object describing the ladder."""
    module = load_module(path, "openmm_md_grouped_protocol")
    protocol = getattr(module, "protocol", None)
    if protocol is None and hasattr(module, "make_protocol"):
        protocol = module.make_protocol()
    if protocol is None:
        raise RuntimeError(
            f"{path} defines neither `protocol` nor `make_protocol()`. A grouped protocol file "
            f"names one REST2Protocol and nothing else; see replica_runtime.py for the contract.")
    return protocol


def _has_completion_line(report):
    """A whole line saying exactly `run_status: completed`, after stripping whitespace."""
    for line in Path(report).read_text(encoding="utf-8").splitlines():
        if line.strip() == COMPLETION_MARKER:
            return True
    return False


# --- verification ---------------------------------------------------------------------------------

def verify_only(arguments):
    """Validate an existing run. Writes nothing; the exit status is the answer."""
    storage = arguments.trajectory or os.environ.get("OPENMM_TRAJECTORY")
    if not storage:
        print("openmm-md: --verify-only needs -x/--trajectory (the stored output)",
              file=sys.stderr)
        return 2
    storage_path = Path(storage).expanduser()
    directory = str(storage_path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        import replica_validate
    except ImportError as failure:
        print(f"openmm-md: cannot import replica_validate from {directory} ({failure}). "
              f"--verify-only reads the copy that belongs to the run being checked.",
              file=sys.stderr)
        return 2
    checkpoint = arguments.checkpoint or os.environ.get("OPENMM_CHECKPOINT")
    manifest = arguments.restart or os.environ.get("OPENMM_RESTART")
    result = replica_validate.validate_replica_output(
        analysis=str(storage_path),
        checkpoint=str(Path(checkpoint).expanduser()) if checkpoint else None,
        manifest=str(Path(manifest).expanduser()) if manifest else None)
    print(replica_validate.format_report(result))
    return 0 if result.ok else 1


# --- grouped execution -------------------------------------------------------------------------------

def run_grouped(files, arguments, groups):
    """Build the ladder the group file describes and hand it to the replica driver."""
    directory = str(Path(groups[0]["input"]).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)

    from openmm import XmlSerializer
    from openmm.app import PDBFile
    import yaml

    from replica_driver import ReplicaRun

    protocol = load_grouped_protocol(groups[0]["input"])
    if protocol.n_states != len(groups):
        raise RuntimeError(
            f"the protocol describes {protocol.n_states} states but the group file has "
            f"{len(groups)} group(s). The ladder and the group file must agree; neither is "
            f"inferred from the other.")

    first = groups[0]
    topology = PDBFile(first["topology"]).topology
    base_system = XmlSerializer.deserialize(
        Path(first["system"]).read_text(encoding="utf-8"))

    solute_indices, excluded_bonds = [], []
    if first.get("solute"):
        document = yaml.safe_load(Path(first["solute"]).read_text(encoding="utf-8"))
        count = int(document["n_solute_atoms"])
        span = document.get("solute_atom_range")
        if span and document.get("solute_atom_indices_are_contiguous", False):
            solute_indices = list(range(int(span[0]), int(span[1]) + 1))
        else:
            solute_indices = list(range(count))
        excluded_bonds = [tuple(int(a) for a in pair)
                          for pair in (document.get("rest2") or {}).get(
                              "omega_excluded_bonds", [])]
    else:
        solute_indices = list(range(base_system.getNumParticles()))

    # The reservoir is opened by the driver, which owns the MPI coordinator: preparation happens
    # once, on rank 0, behind a barrier, and every rank then reads the same prepared file.
    run = ReplicaRun(
        protocol=protocol,
        files=SimpleNamespace(topology=first["topology"], system=first["system"],
                              coordinates=first["coordinates"],
                              trajectory=files.trajectory, restart=files.restart,
                              checkpoint=files.checkpoint, output=files.output,
                              rem=getattr(files, "rem", None)),
        base_system=base_system, topology=topology,
        solute_indices=solute_indices, excluded_bonds=excluded_bonds,
        rule_path=arguments.exchange_rule,
        reservoir_declaration=arguments.reservoir,
        platform=getattr(protocol, "platform", None),
        precision=getattr(protocol, "precision", None),
        identity_extra={"groups": len(groups),
                        "group_indices": [g["group_index"] for g in groups]})
    record = run.run(resume=files.resume, extend=files.extend)
    if record.get("run_status") == "completed":
        _print_grouped_summary(record, protocol)
        print(COMPLETION_MARKER)
        return 0
    if record.get("run_status") == "interrupted":
        # An interrupted run did NOT finish, and it has no completion manifest by design. Saying
        # so with its own status keeps the promise check below -- which exists to catch a protocol
        # that claimed to finish and silently wrote nothing -- from reporting a deliberate,
        # cleanly checkpointed stop as a failed run.
        return INTERRUPTED_STATUS
    return 0


def _print_grouped_summary(record, protocol):
    stats = record["lifetime_statistics"]
    schedule = record["schedule"]
    print()
    print("# --- replica exchange summary (the NetCDF is authoritative) ------------------")
    print(f"# steps completed       : {record['steps_completed']} of {record['steps_expected']}")
    print(f"# exchanges             : {record['exchanges_committed']} "
          f"(every {schedule['exchange_interval_steps']} steps = "
          f"{schedule['exchange_interval_ps']} ps)")
    print(f"# whole frames          : {record['whole_frames']} "
          f"(every {schedule['whole_output_interval_steps']} steps)")
    print(f"# solute frames         : {record['solute_frames']} "
          f"(every {schedule['solute_output_interval_steps']} steps)")
    print(f"# production per replica: {record['production_ps_per_replica']} ps")
    print(f"# exchange rule         : {record['exchange_rule'].get('name')}")
    print(f"# final state->walker   : {record['final_state_to_walker']}")
    # Neighbouring pairs and an overall figure, rendered from the same structure that was
    # persisted in the completion manifest so the terminal and the file cannot disagree. Round
    # trips, transition matrices, first-passage times and convergence diagnostics are downstream
    # analysis and are deliberately absent.
    report = record["completion_report"]
    print("# NEIGHBOURING-PAIR acceptance:")
    print(f"#   basis: {report['basis']}")
    for pair in report["by_neighbouring_pair"]:
        rate = pair["acceptance"]
        shown = "n/a" if rate is None else f"{rate:.3f}"
        i, j = pair["state_pair"]
        print(f"#   state {i} <-> state {j}   {pair['accepted']}/{pair['proposed']}   {shown}")
    overall = report["overall"]
    shown = "n/a" if overall["acceptance"] is None else f"{overall['acceptance']:.3f}"
    print(f"#   overall               {overall['accepted']}/{overall['proposed']}   {shown}")
    if "reservoir" in report:
        r, stats_r = report["reservoir"], stats["reservoir"]
        print(f"# reservoir refreshes   : {r['accepted']}/{r['attempts']} at state(s) "
              f"{r['states_refreshed']}, {stats_r['distinct_frames_used']} distinct sample(s) "
              f"from source step(s) {stats_r['source_steps_used'][:6]}")
        print(f"#   velocity policy      : "
              f"{record.get('reservoir', {}).get('velocity_policy')}")
    print("# ---------------------------------------------------------------------------")


# --- main -------------------------------------------------------------------------------------------

def main(argv=None):
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    if arguments.verify_only:
        return verify_only(arguments)

    rank, size = mpi_rank_and_size()
    groups = None
    if arguments.groupfile:
        try:
            groups = parse_group_file(arguments.groupfile)
        except GroupFileError as failure:
            print(f"openmm-md: {failure}", file=sys.stderr)
            return 2

    files, problems = resolve(arguments)
    problems += validate(files, arguments, rank=rank, groups=groups)
    if problems:
        for problem in problems:
            print(f"openmm-md: [rank {rank}] {problem}" if size > 1
                  else f"openmm-md: {problem}", file=sys.stderr)
        return 2

    for value in _outputs(files).values():
        Path(value).parent.mkdir(parents=True, exist_ok=True)

    report = Path(report_path_for_rank(files.output, rank))
    mode = "a" if (files.resume or files.extend) else "w"
    status = 0
    with open(report, mode, encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                if groups is not None:
                    _announce(arguments, files, groups, rank, size)
                    status = run_grouped(files, arguments, groups)
                else:
                    load_protocol(files.input).run(files)
            except SystemExit as exit_request:
                status = int(exit_request.code or 0)
            except BaseException:                       # noqa: BLE001 - the .out is the report
                traceback.print_exc()
                status = 1

    if status == INTERRUPTED_STATUS:
        print(f"openmm-md: interrupted at an event boundary; the checkpoint is complete and "
              f"--resume continues it. See {report}", file=sys.stderr)
        return status

    if status == 0 and rank == 0:
        promised = {name: value for name, value in _outputs(files).items() if name != "output"}
        missing = [f"--{name} {value}" for name, value in promised.items()
                   if not Path(value).exists()]
        if missing:
            message = "openmm-md: the protocol finished but did not write: " + ", ".join(missing)
            with open(report, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            print(message, file=sys.stderr)
            status = 1
        elif not _has_completion_line(report):
            print(f"openmm-md: {report} has no '{COMPLETION_MARKER}' line; treating as incomplete",
                  file=sys.stderr)
            status = 1

    if status != 0:
        print(f"openmm-md: run failed; see {report}", file=sys.stderr)
    return status


def _announce(arguments, files, groups, rank, size):
    print(f"# openmm-md, grouped mode: {len(groups)} group(s) from "
          f"{Path(arguments.groupfile).name}")
    print(f"# mpi                : rank {rank}/{size}")
    print(f"# exchange rule      : "
          f"{Path(arguments.exchange_rule).name if arguments.exchange_rule else 'built-in neighbouring'}")
    print(f"# reservoir          : "
          f"{Path(arguments.reservoir).name if arguments.reservoir else 'none'}")
    print(f"# analysis storage   : {Path(files.trajectory).name}")
    print(f"# checkpoint         : "
          f"{Path(files.checkpoint).name if files.checkpoint else 'none'}")
    print(f"# manifest           : {Path(files.restart).name if files.restart else 'none'}")
    for group in groups:
        print(f"#   group {group['group_index']}: -i {Path(group['input']).name} "
              f"-p {Path(group['topology']).name} -s {Path(group['system']).name} "
              f"-c {Path(group['coordinates']).name}")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
