#!/usr/bin/env python
"""The ONE simulation executor a generated replica script drives.

Called as a function by `md_tools.remd.generated`, not as a command: MD-tools
installs exactly one executable, `md-openmm`, and this is reached through the
`REST2.py` / `rREST2.py` that `md-openmm build-md` generates.

Amber has `pmemd -i in -p prmtop -c rst -o out -x nc -r rst` for a single system and
`pmemd.MPI -ng N -groupfile groups` for a coordinated set of them. This is the same idea with the
same executable:

    python REST2.py -p built.pdb -s built.xml -c eq_npt_free.xml \\
              -o stage.out -x trajectory.dcd -r final_state.xml --checkpoint stage.chk

    mpiexec -n 6 python REST2.py -p built.pdb -s built.xml \\
              -o REST2/rest2.out -x REST2/rest2.nc -r REST2/restart.json \\
              --checkpoint REST2/rest2_checkpoint.nc

    mpiexec -n 6 python rREST2.py -p built.pdb -s built.xml \\
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
    """What the launcher says this process is. A NAME for `md_tools.remd.mpi`'s reader.

    Kept because callers across the package spell it this way; it holds no policy of its own.
    """
    from .mpi import launcher_rank_and_size

    return launcher_rank_and_size(environment)


def barrier(size=None, *, coordination=None):
    """Wait for every rank. Delegates to the ONE MPI authority; it does not decide anything.

    This function used to decide, and it decided wrongly:

        try:
            from mpi4py import MPI
        except ImportError:
            return                      # "no mpi4py, so nothing to wait for"

    That is true of a serial run and false of every other kind. Under a plural launcher it turned
    a barrier into a no-op, which is how eight ranks stop being a ladder without anything failing.
    `md_tools.remd.mpi.barrier` refuses instead, and this now calls it.
    """
    from .mpi import barrier as _barrier

    _barrier(size, coordination=coordination)


def report_path_for_rank(output, rank):
    """Rank 0 writes the run's `.out`; every other rank writes its own beside it.

    Without this, N ranks race for one file: the first opens it and the rest see an output that
    "already exists" and refuse. Each rank's log is kept rather than discarded, because a rank that
    failed to bind its device is exactly what a multi-GPU run needs to be able to show.
    """
    return output if rank == 0 else f"{output}.rank{rank:02d}"


# --- the group file ----------------------------------------------------------------------------

def parse_group_file(path, *, extending=False):
    """One group per line, parsed with `shlex`. Never evaluated by a shell.

    Returns a list of dictionaries in file order. Every failure names the line number, because a
    group file is written by hand often enough that "somewhere in this file" is not good enough.

    `extending` says a `--extend-from` is in force, which makes `-c` optional: see
    `_parse_group_line`. It is a parameter rather than a lookup because the group file is parsed
    before `resolve()` runs, so this function cannot ask the resolved inputs what mode it is in.
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
        groups.append(_parse_group_line(path, number, tokens, extending=extending))
    if not groups:
        raise GroupFileError(f"{path} contains no group lines")
    _check_group_indices(path, groups)
    _resolve_group_paths(path, groups)
    _require_homogeneous_groups(path, groups)
    return groups


def _resolve_group_paths(path, groups):
    """A relative path in a group line is relative to the GROUP FILE, not to `os.getcwd()`.

    Amber reads group files this way and so does everyone who writes one, but this resolved them
    against the working directory. `mpirun` from one directory and a group file in another then
    silently gave every rank a different idea of where `built.pdb` was -- usually "nowhere", which
    is at least loud, and occasionally a DIFFERENT built.pdb, which is not.
    """
    parent = Path(path).resolve().parent
    for group in groups:
        for field in ("input", "topology", "system", "coordinates", "solute"):
            value = group.get(field)
            if not value:
                continue
            # RESOLVED, not merely joined: `run/../built.pdb` and `built.pdb` are one file, and
            # the collision checks in `md_tools.run.preflight` compare resolved paths. Leaving
            # one of the two unnormalised is how the same file comes to look like two.
            group[field] = str((Path(value) if Path(value).is_absolute()
                                else parent / value).resolve(strict=False))


#: Fields every line of a homogeneous ladder must agree about. The current ladder is exactly that
#: -- N rungs of ONE system differing only in tau, which is derived from the group index -- so a
#: line naming a different topology or a different starting state is not a per-state input this
#: build implements. It is a mistake, and accepting it would run a ladder whose rungs are not
#: states of the same Hamiltonian while every exchange log looked healthy.
HOMOGENEOUS_GROUP_FIELDS = ("input", "topology", "coordinates", "solute")


def _require_homogeneous_groups(path, groups):
    for field in HOMOGENEOUS_GROUP_FIELDS:
        values = {group.get(field) for group in groups}
        if len(values) > 1:
            listed = ", ".join(f"line {group['line']}: {group.get(field)}"
                               for group in groups[:4])
            raise GroupFileError(
                f"{path}: the group lines give {len(values)} different values for {field}, and "
                f"this ladder is homogeneous -- N rungs of ONE system, differing only in tau. "
                f"Per-state inputs are not implemented, so accepting these would run rungs that "
                f"are not states of the same Hamiltonian while every exchange log looked "
                f"healthy.\n  {listed}")
    _require_one_rung_system_per_state(path, groups)


def _require_one_rung_system_per_state(path, groups):
    """`system` is PER STATE now, and that is a stronger check rather than a weaker one.

    Scaling moved to build time: each rung is serialised to `remd<n>/build_state<n>.xml`, so the
    lines of a ladder's group file name N DIFFERENT Systems by design. `system` was in
    `HOMOGENEOUS_GROUP_FIELDS` on the premise that tau was derived from `--group-index` at run
    time, which is no longer how a rung comes to be scaled -- keeping it there refused every group
    file this architecture writes.

    Dropping it and stopping there would be the real defect. The old check did rule something out:
    with one shared System, two lines naming different files could not both be rungs of the same
    ladder. With one file per rung, the mistake to rule out is a line whose System is not the rung
    its own `--group-index` claims -- `remd2/build_state2.xml` on the line for state 3. That
    swaps two rungs' Hamiltonians, runs perfectly, and reports healthy exchange statistics while
    the ladder is not the one anybody asked for.

    So: a line whose System follows the `build_state<n>.xml` convention must agree with its group
    index, and no two lines may name the same file. A System that does NOT follow the convention is
    left alone -- a by-hand ladder pointing at files of its own naming is legitimate, and this
    function is not the place to invent a mandatory filename.
    """
    import re

    # NOT `state_index_from_name`: that one parses TRAJECTORY names
    # (`<whole|solute>_state<n>_prod<x>.nc`) and raises on anything else, so handing it a
    # `build_state<n>.xml` raised every time and a blanket `except` turned this whole check into a
    # no-op that still read as one.
    rung = re.compile(r"^build_state(\d+)\.xml$")
    seen: dict[str, int] = {}
    for group in groups:
        system = group.get("system")
        if not system:
            continue
        index = group.get("group_index")
        matched = rung.fullmatch(Path(system).name)
        if matched is None:
            # NOT THE GENERATED CONVENTION, and it must stay accepted. Before scaling moved to
            # build time a ladder named ONE shared `built.xml` on every line and derived tau from
            # `--group-index` -- which is what every group file written until now looks like, the
            # nine migrated reference runs included. Applying the per-rung rules below to those
            # would refuse a ladder that is perfectly well formed for the architecture that wrote
            # it, so a name outside the convention is left entirely alone here.
            continue
        if system in seen:
            raise GroupFileError(
                f"{path}:{group['line']}: this rung System is already used by the line for state "
                f"{seen[system]}. Each rung is its own serialised Hamiltonian, so two states "
                f"sharing one file are two states at the same tau -- a ladder with a rung "
                f"missing, which exchanges perfectly and samples the wrong set of "
                f"Hamiltonians.\n  {system}")
        seen[system] = index
        named = int(matched.group(1))
        if index is not None and named != int(index):
            raise GroupFileError(
                f"{path}:{group['line']}: --group-index {index} is given the System of state "
                f"{named} ({Path(system).name}). Each rung is pre-scaled to its own tau now, so "
                f"this does not run state {index} -- it runs state {named}'s Hamiltonian under "
                f"state {index}'s identity, and every exchange log looks healthy.")


def _parse_group_line(path, number, tokens, *, extending=False):
    group = {"line": number}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in RUN_LEVEL_FLAGS:
            raise GroupFileError(
                f"{path}:{number}: {token} describes the coordinated run and belongs on the outer "
                f"executor call, not on a group line. A per-group output would give every "
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
    # COORDINATES ARE NOT REQUIRED ON AN EXTENSION, and requiring them made
    # `--extend-from` generate a file its own parser rejected. An extension takes the physical
    # state from the parent's checkpoint -- positions, velocities, box, the state-to-walker map,
    # the RNG stream, the exchange count -- so a `-c` on the line would never be read for its
    # contents; and the generated group file, correctly, does not write one. The result was
    # `no coordinates given` and every rank aborting on a run that had asked for nothing wrong.
    required = ["input", "topology", "system"]
    if not extending:
        required.append("coordinates")
    for name in required:
        if name not in group:
            raise GroupFileError(
                f"{path}:{number}: no {name} given"
                + ("" if name != "coordinates" else
                   " (coordinates are optional only with --extend-from, where the state comes "
                   "from the parent)"))
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
        prog="replica-executor",
        description="Run one OpenMM protocol against explicit paths, or a coordinated set of "
                    "replicas from an Amber-like group file.",
        # No abbreviation, here as in every other parser in this package. This one was missed,
        # and it is the parser furthest from the person typing -- so a prefix argparse resolved
        # here would be a setting nobody wrote, applied to a coordinated multi-rank run.
        allow_abbrev=False)
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
    parser.add_argument("--extend-from", dest="extend_from", default=None, metavar="DIRECTORY",
                        help="GROUPED ONLY: continue a COMPLETED run held in DIRECTORY, writing a "
                             "new output set here. The parent is opened read-only and is left "
                             "byte-for-byte unchanged; use --extend N to say how much new "
                             "dynamics to add. Without it, --extend lengthens the run in place")
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
    parent = getattr(arguments, "extend_from", None) or os.environ.get("OPENMM_EXTEND_FROM")
    files["extend_from"] = str(Path(parent).expanduser()) if parent else None
    if files["extend_from"] and not Path(files["extend_from"]).is_dir():
        problems.append(f"--extend-from {files['extend_from']} is not a directory")
    return SimpleNamespace(**files), problems


def _outputs(files):
    return {name: getattr(files, name)
            for name in ("output", "trajectory", "restart", "checkpoint", "solute_x")
            if getattr(files, name)}


def validate(files, arguments, *, rank=0, groups=None):
    """Everything decidable without touching OpenMM or writing a byte."""
    problems = []
    grouped = bool(arguments.groupfile)
    extending_out = bool(getattr(files, "extend_from", None))
    # An out-of-place extension writes a NEW output set, so it is a fresh run as far as the
    # output rules are concerned: `-x` here must NOT exist, and the parent is never an output.
    continuing = bool(files.resume or files.extend) and not extending_out

    # -- mode ---------------------------------------------------------------------------------
    if grouped and arguments.number_of_groups is None:
        problems.append("--groupfile requires -ng: the number of groups is stated so a truncated "
                        "group file cannot silently run a shorter ladder")
    if arguments.number_of_groups is not None and not grouped:
        problems.append("-ng describes a group file and has no meaning without --groupfile")
    if not grouped:
        # THE ONE ROUTE. Without a group file the executor used to import `files.input` and call
        # `.run(files)` on it -- no ladder plan, no preflight, no platform resolution, no output
        # rules beyond the ones above. That was a second contract for the same runtime, reachable
        # only by calling this module directly, and it is exactly the one nothing validates. A
        # coordinated run is described by its group file; there is no other description.
        problems.append(
            "a run is described by its group file: pass --groupfile (with -ng). Running a "
            "protocol module directly is not supported -- it would execute with no ladder plan "
            "and no preflight, and nothing would have checked the platform, the ensemble or the "
            "outputs. `md-openmm md-run` and the generated `run.sh` both pass one")
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

    # -- out-of-place extension -----------------------------------------------------------------
    if extending_out:
        parent = Path(files.extend_from)
        if not grouped:
            problems.append("--extend-from is supported only with --groupfile: only the replica "
                            "runtime can carry a coordinated ladder across a boundary")
        if not files.extend:
            problems.append("--extend-from needs --extend N: how much new dynamics the extension "
                            "adds is stated, never inherited from the parent's own budget")
        if files.resume:
            problems.append("--resume continues a run in place and --extend-from writes a new "
                            "output set; they are different operations and cannot be combined")
        if arguments.force:
            problems.append("--force replaces outputs; an extension never writes into its parent "
                            "and has nothing to replace")
        for name, value in _outputs(files).items():
            try:
                inside = parent.resolve() in Path(value).resolve().parents
            except OSError:
                inside = False
            if inside:
                problems.append(
                    f"--{name} {value} is inside the parent {parent}. The parent of an extension "
                    f"is immutable: it is read, never written, and an output placed inside it "
                    f"would modify the very run being continued.")
        # Only the manifest name is assumed. It records the analysis and checkpoint names the
        # parent was actually written with, and those are checked rather than guessed -- the
        # generated ladders name theirs after the method, not after the documentation example.
        if parent.is_dir():
            manifest = parent / "restart.json"
            if not manifest.is_file():
                problems.append(
                    f"{parent} holds no restart.json, so it is not a completed run. An extension "
                    f"continues a finished parent; an unfinished one is resumed in place with "
                    f"--resume so that its own budget is met first.")
            else:
                try:
                    recorded = json.loads(manifest.read_text(encoding="utf-8"))
                except ValueError as failure:
                    problems.append(f"{manifest} is not readable JSON ({failure})")
                    recorded = {}
                if recorded.get("run_status") not in (None, "completed"):
                    problems.append(
                        f"{manifest} records run_status {recorded.get('run_status')!r}, not "
                        f"'completed'. Only a finished run is extended.")
                names = recorded.get("storage") or {}
                for field in ("analysis_netcdf", "checkpoint_netcdf"):
                    named = names.get(field)
                    if named and not (parent / named).is_file():
                        problems.append(
                            f"{manifest} names {named} as its {field} and that file is not in "
                            f"{parent}: the parent is incomplete.")

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
            # WHICH COMMAND IS THE USER HOLDING. This executor is reached both directly and by
            # dispatch from `md-run`, and the two define different flags: `--resume` is on both,
            # `--extend` and `--force` are on this executor only, and `md-run` replaces outputs
            # with `--overwrite`. A single imperative naming all three sent md-run users to
            # `unrecognized arguments: --force`, and from there to deleting files by hand -- the
            # more dangerous of the remedies offered. The imperative now names only the flag both
            # commands have; the rest are described, and attributed to the command that has them.
            problems.append(
                "these outputs already exist: " + ", ".join(existing)
                + ". Pass --resume to continue that run. To lengthen a finished run, the replica "
                  "executor takes --extend N; to replace outputs deliberately it takes --force, "
                  "and md-run takes --overwrite.")
    return problems


# --- loading protocol files ---------------------------------------------------------------------

def load_module(path, name):
    """Import a file by location, with its own directory importable.

    A generated project keeps its runtime modules beside the protocol, so the protocol's directory
    is imported from the installed package -- `from md_tools.remd import REST2Protocol` -- in
    a project that has been moved and has no `md_tools` anywhere.
    """
    path = Path(path)
    # NOT added to `sys.path`. A rule or protocol file used to sit beside COPIES of the
    # runtime modules and import them by bare name, so its directory had to be importable.
    # It imports from the installed package now, and inserting the directory would let a
    # file next to it shadow a standard-library module for the rest of the process.
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} could not be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_grouped_protocol(path):
    """A grouped protocol file: one `protocol` object describing the ladder."""
    module = load_module(path, "md_tools_grouped_protocol")
    protocol = getattr(module, "protocol", None)
    if protocol is None and hasattr(module, "make_protocol"):
        protocol = module.make_protocol()
    if protocol is None:
        raise RuntimeError(
            f"{path} defines neither `protocol` nor `make_protocol()`. A grouped protocol file "
            f"names one REST2Protocol and nothing else; see md_tools.remd.facade for the contract.")
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
        print("replica executor: --verify-only needs -x/--trajectory (the stored output)",
              file=sys.stderr)
        return 2
    storage_path = Path(storage).expanduser()
    # The validator comes from the INSTALLED package. It used to be imported from the directory
    # beside the stored output, because a generated project carried its own copy and the run had
    # to be checked by the code that produced it. There are no copies now -- one installed
    # implementation, whose version every record already names.
    from . import validate as replica_validate
    checkpoint = arguments.checkpoint or os.environ.get("OPENMM_CHECKPOINT")
    manifest = arguments.restart or os.environ.get("OPENMM_RESTART")
    result = replica_validate.validate_replica_output(
        analysis=str(storage_path),
        checkpoint=str(Path(checkpoint).expanduser()) if checkpoint else None,
        manifest=str(Path(manifest).expanduser()) if manifest else None)
    print(replica_validate.format_report(result))
    return 0 if result.ok else 1


# --- grouped execution -------------------------------------------------------------------------------

def run_grouped(files, arguments, groups, *, prepared=None):
    """Build the ladder the group file describes and hand it to the replica driver."""
    # The group file's directory is NOT put on `sys.path`: the protocol module it names is loaded
    # by path and imports the runtime from the installed package.
    from openmm import XmlSerializer
    from openmm.app import PDBFile
    import yaml

    from .driver import ReplicaRun

    protocol = load_grouped_protocol(groups[0]["input"])
    if protocol.n_states != len(groups):
        raise RuntimeError(
            f"the protocol describes {protocol.n_states} states but the group file has "
            f"{len(groups)} group(s). The ladder and the group file must agree; neither is "
            f"inferred from the other.")

    first = groups[0]

    # CONSUMED from the plan when there is one. The preflight already deserialised this System and
    # resolved this selection -- before any output existed -- so redoing both here was a second
    # derivation of the same scientific facts, from files on disk, with nothing making the two
    # agree. The System matters most: the ladder's rungs are built FROM it, and a second
    # deserialisation is a second object that the preflight never audited.
    loaded = getattr(prepared, "loaded", None) if prepared is not None else None
    if loaded is not None:
        topology = loaded.pdb.topology
        base_system = loaded.system
    else:
        topology = PDBFile(first["topology"]).topology
        base_system = XmlSerializer.deserialize(
            Path(first["system"]).read_text(encoding="utf-8"))

    solute_indices = list(getattr(prepared, "solute_indices", ()) or ()) if prepared else []
    excluded_bonds = [tuple(int(a) for a in pair)
                      for pair in (getattr(prepared, "excluded_bonds", ()) or ())] if prepared \
        else []
    if not solute_indices:
        # No plan, or a plan that did not resolve a selection: a direct caller that built the
        # group file itself. The group file's own `solute.yaml` is then the only description of
        # the selection there is, and reading it here is the fallback rather than the default.
        if first.get("solute"):
            document = yaml.safe_load(Path(first["solute"]).read_text(encoding="utf-8"))
            count = int(document["n_solute_atoms"])
            span = document.get("solute_atom_range")
            if span and document.get("solute_atom_indices_are_contiguous", False):
                solute_indices = list(range(int(span[0]), int(span[1]) + 1))
            else:
                solute_indices = list(range(count))
            excluded_bonds = _excluded_bonds_from_solute_document(
                document, source=first["solute"])
        else:
            solute_indices = list(range(base_system.getNumParticles()))

    # The reservoir is opened by the driver, which owns the MPI coordinator: preparation happens
    # once, on rank 0, behind a barrier, and every rank then reads the same prepared file.
    run = ReplicaRun(
        protocol=protocol,
        # `.get`, not `[...]`: an extension's group file carries no `-c`, because `--extend-from`
        # takes positions, velocities, box, the state-to-walker map and the RNG stream from the
        # parent's checkpoint. Reading it unconditionally raised `KeyError: 'coordinates'` here,
        # one step past the announcement that raised it a moment earlier -- the parser had been
        # made conditional and both of its consumers had not.
        files=SimpleNamespace(topology=first["topology"], system=first["system"],
                              coordinates=first.get("coordinates"),
                              trajectory=files.trajectory, restart=files.restart,
                              checkpoint=files.checkpoint, output=files.output,
                              rem=getattr(files, "rem", None)),
        base_system=base_system, topology=topology,
        solute_indices=solute_indices, excluded_bonds=excluded_bonds,
        rule_path=arguments.exchange_rule,
        reservoir_declaration=arguments.reservoir,
        platform=getattr(protocol, "platform", None),
        # `--cpu` reaches the driver through the protocol file's `platform` field, which is the
        # only channel a generated protocol has. The platform itself comes from machine.openmm;
        # this carries the one thing the command line can still say about it.
        explicit_cpu=str(getattr(protocol, "platform", None) or "").upper() == "CPU",
        precision=getattr(protocol, "precision", None),
        identity_extra={"groups": len(groups),
                        "group_indices": [g["group_index"] for g in groups]},
        # The validated platform, device and coordination. The driver consumes these; it no
        # longer reloads the machine configuration and resolves a second platform of its own
        # after every file on disk already exists.
        prepared=prepared)
    record = run.run(resume=files.resume, extend=files.extend,
                     extend_from=getattr(files, 'extend_from', None))
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

    # THE LADDER'S OWN HAMILTONIAN IDENTITY. Amber prints its REAF block -- the tau, the
    # `gti_add_re` scheme, and the mask with its matched atom count -- so a reader can tell what
    # was scaled. It cannot name the resulting Hamiltonian, because `gti_add_re=6` is an index
    # into a table in the manual and the scaled potential exists only inside the binary. A rung
    # here IS a serialised System, so this can do what Amber structurally cannot: name it and
    # give its digest.
    identity = record.get("scientific_identity") or {}
    implementation = identity.get("rest2_implementation") or {}
    hamiltonian = identity.get("hamiltonian") or {}
    if identity:
        print("# REST2:")
        taus = identity.get("tau") or []
        print(f"#   tau ladder            {', '.join(f'{float(t):g}' for t in taus)}"
              f"   ({identity.get('n_states')} state(s), one temperature "
              f"{identity.get('temperature_k')} K, {identity.get('ensemble')})")
        if implementation:
            print(f"#   scaling               solute-solute "
                  f"{implementation.get('solute_solute_nonbonded_scale')}, "
                  f"solute-environment {implementation.get('solute_environment_nonbonded_scale')}")
            torsions = implementation.get("unscaled_torsions")
            print(f"#   left unscaled         bonds {implementation.get('bonds')}, "
                  f"angles {implementation.get('angles')}, "
                  + (f"torsions: {', '.join(torsions)}" if torsions is not None else
                     f"ordinary amide omega {implementation.get('ordinary_amide_omega')}"))
        if hamiltonian:
            print(f"#   solute region         {hamiltonian.get('n_solute_atoms')} atom(s), "
                  + (f"{hamiltonian.get('n_unscaled_central_bonds')} unscaled central bond(s), "
                     f"impropers {'unscaled' if hamiltonian.get('unscaled_impropers') else 'scaled'}"
                     if 'n_unscaled_central_bonds' in hamiltonian else
                     f"{hamiltonian.get('n_omega_excluded_bonds')} omega bond(s) excluded"))
            digest = hamiltonian.get("system_sha256")
            if digest:
                print(f"#   system sha256         {digest}")
        print(f"#   velocities on swap    "
              f"{'rescaled' if identity.get('velocity_rescaling_on_exchange') else 'never rescaled'}"
              f" (one beta across the ladder)")

    # TIMINGS, as Amber's `5. TIMINGS` section gives them. "How long, and how fast" is the first
    # question after "did it finish", and it was answerable only by subtracting two timestamps in
    # the manifest.
    started, finished = record.get("started_utc"), record.get("finished_utc")
    elapsed = None
    if started and finished:
        from datetime import datetime

        try:
            elapsed = (datetime.fromisoformat(finished)
                       - datetime.fromisoformat(started)).total_seconds()
        except ValueError:
            elapsed = None
    if elapsed:
        produced = float(record.get("production_ps_per_replica") or 0.0) / 1000.0
        print("# TIMINGS:")
        print(f"#   elapsed               {elapsed:.1f} s")
        if produced:
            per_replica = produced / (elapsed / 86400.0)
            states = int(identity.get("n_states") or 1)
            print(f"#   throughput            {per_replica:.2f} ns/day per replica, "
                  f"{per_replica * states:.2f} ns/day aggregate over {states} state(s)")
        steps = int(record.get("steps_completed") or 0)
        if steps:
            print(f"#   per step              {elapsed / steps * 1000.0:.4f} ms")
    if "reservoir" in report:
        r, stats_r = report["reservoir"], stats["reservoir"]
        print(f"# reservoir refreshes   : {r['accepted']}/{r['attempts']} at state(s) "
              f"{r['states_refreshed']}, {stats_r['distinct_frames_used']} distinct sample(s) "
              f"from source step(s) {stats_r['source_steps_used'][:6]}")
        print(f"#   velocity policy      : "
              f"{record.get('reservoir', {}).get('velocity_policy')}")
    print("# ---------------------------------------------------------------------------")


def _excluded_bonds_from_solute_document(document, *, source) -> list[tuple[int, ...]]:
    """The unscaled central bonds a `solute.yaml` records -- refused if it also records unresolved
    items.

    The document's unclassified list (`unclassified`, or 0.5.3's `omega_ambiguous_candidates`) is
    the classifier's as the document stored it. Taking the unscaled bonds and ignoring it would scale
    those torsions, which is the defect `openmm.system.unscaled_torsions` exists to refuse; a
    document is not an exemption from it.
    """
    from ..openmm.builders import (unclassified_of_solute_document,
                                   unscaled_bonds_of_solute_document)

    unresolved = unclassified_of_solute_document(document)
    if unresolved:
        shown = "; ".join(f"{'bond ' + str(c.get('bond')) if c.get('bond') else 'residue'} "
                          f"({c.get('nitrogen_residue')}): {c.get('evidence')}"
                          for c in unresolved[:5])
        raise RuntimeError(
            f"{source} records {len(unresolved)} item(s) whose torsions could not be classified, "
            f"so which torsions this ladder may scale is undecided. Refusing rather than scaling "
            f"them: {shown}")
    return unscaled_bonds_of_solute_document(document)


def _preflight_from_groups(files, arguments, groups):
    """Run the shared ladder preflight for a directly-invoked executor.

    Returns a `LadderPreflight`, or an exit code when it refuses. The group file is the only
    description of the launch the executor has, so the inputs come from its FIRST line -- which
    is legitimate precisely because every line must agree about them (see `_check_group_lines`);
    a heterogeneous group file is refused before this point rather than silently represented by
    one of its lines.
    """
    from ..run.preflight import PreflightError, preflight_ladder

    first = groups[0]
    try:
        return preflight_ladder(
            topology=first["topology"], system=first["system"], replicas=len(groups),
            coordinates=first.get("coordinates"), groupfile=arguments.groupfile,
            trajectory=files.trajectory, restart=files.restart,
            checkpoint=getattr(files, "checkpoint", None),
            output=files.output, log=None,
            # `-ng` means something DIFFERENT at this layer. On the command line it is a claim
            # about the launch, cross-checked against the MPI world. Here it is simply how many
            # groups the file has, and `validate` already checks it against the file -- the
            # generated ladder passes `-ng <states>` on every call, serial runs included. Passing
            # it on as a launch claim would refuse every single-process ladder, which
            # `owned_states` explicitly supports.
            number_of_groups=None,
            cpu=str(getattr(arguments, "platform", None) or "").upper() == "CPU",
            protocol="this ladder")
    except PreflightError as refusal:
        print(f"replica executor: {refusal}", file=sys.stderr)
        return 2


# --- main -------------------------------------------------------------------------------------------

def main(argv=None, *, prepared=None):
    """`prepared` is the `LadderPreflight` the caller already validated this launch with.

    The executor is an independently callable entry point -- `md_tools.remd.executor.main` is
    importable and the generated ladder calls it -- so it must not depend on someone else having
    checked first. When nothing is handed in it runs the shared preflight itself, below, at the
    same point and with the same rules.
    """
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    if arguments.verify_only:
        return verify_only(arguments)

    rank, size = mpi_rank_and_size()
    groups = None
    if arguments.groupfile:
        try:
            # Whether an extension is in force decides if `-c` is required on each line; the
            # env var mirrors `resolve()`, which has not run yet at this point.
            extending = bool(getattr(arguments, "extend_from", None)
                             or os.environ.get("OPENMM_EXTEND_FROM"))
            groups = parse_group_file(arguments.groupfile, extending=extending)
        except GroupFileError as failure:
            print(f"replica executor: {failure}", file=sys.stderr)
            return 2

    files, problems = resolve(arguments)
    problems += validate(files, arguments, rank=rank, groups=groups)
    if problems:
        for problem in problems:
            print(f"replica executor: [rank {rank}] {problem}" if size > 1
                  else f"replica executor: {problem}", file=sys.stderr)
        return 2

    if groups is not None and prepared is None:
        # Called directly, with nobody having validated this launch. Run the SAME preflight the
        # generated ladder runs, here, before the output directories below are created -- rather
        # than letting the driver resolve a platform of its own halfway through the run.
        prepared = _preflight_from_groups(files, arguments, groups)
        if isinstance(prepared, int):
            return prepared

    for value in _outputs(files).values():
        Path(value).parent.mkdir(parents=True, exist_ok=True)

    report = Path(report_path_for_rank(files.output, rank))
    mode = "a" if (files.resume or files.extend) and not getattr(
        files, "extend_from", None) else "w"
    status = 0
    with open(report, mode, encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                # Grouped is the only route past `validate`, which refuses an ungrouped launch
                # above -- before any directory is created.
                _announce(arguments, files, groups, rank, size)
                status = run_grouped(files, arguments, groups, prepared=prepared)
            except SystemExit as exit_request:
                status = int(exit_request.code or 0)
            except BaseException:                       # noqa: BLE001 - the .out is the report
                traceback.print_exc()
                status = 1

    if status == INTERRUPTED_STATUS:
        # An out-of-place extension is atomic: `--resume` does not continue a segment, it
        # finishes it as an ordinary run with no `extends` at all. The discriminator is the one
        # line 831 already uses to decide the report's open mode.
        if getattr(files, "extend_from", None):
            print(f"replica executor: interrupted at an event boundary; the checkpoint is "
                  f"complete, but an out-of-place extension CANNOT be resumed -- --resume "
                  f"carries no parent linkage and would finish this segment as an ordinary run. "
                  f"Re-run the whole segment with --extend-from into a fresh directory. "
                  f"See {report}", file=sys.stderr)
        else:
            print(f"replica executor: interrupted at an event boundary; the checkpoint is "
                  f"complete and --resume continues it. See {report}", file=sys.stderr)
        return status

    if status == 0 and rank == 0:
        promised = {name: value for name, value in _outputs(files).items() if name != "output"}
        missing = [f"--{name} {value}" for name, value in promised.items()
                   if not Path(value).exists()]
        if missing:
            message = "replica executor: the protocol finished but did not write: " + ", ".join(missing)
            with open(report, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            print(message, file=sys.stderr)
            status = 1
        elif not _has_completion_line(report):
            print(f"replica executor: {report} has no '{COMPLETION_MARKER}' line; treating as incomplete",
                  file=sys.stderr)
            status = 1

    if status != 0:
        print(f"replica executor: run failed; see {report}", file=sys.stderr)
    return status


def _announce(arguments, files, groups, rank, size):
    print(f"# replica executor, grouped mode: {len(groups)} group(s) from "
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
        # An extension's group file carries NO `-c`: `--extend-from` takes the physical state from
        # the parent's checkpoint, which is the whole point of it. Printing the coordinates
        # unconditionally raised `KeyError: 'coordinates'` here -- after the header above had
        # already been written, so the run looked like it had started -- and every rank aborted.
        # The parser was taught that a coordinate-free line is legal; this half was not.
        state = (f"-c {Path(group['coordinates']).name}" if group.get("coordinates")
                 else "-c (none: continued from the parent's checkpoint)")
        print(f"#   group {group['group_index']}: -i {Path(group['input']).name} "
              f"-p {Path(group['topology']).name} -s {Path(group['system']).name} {state}")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
