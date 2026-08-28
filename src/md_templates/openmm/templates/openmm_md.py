#!/usr/bin/env python
"""`openmm-md` -- the file interface for a generated OpenMM stage.

Amber has `pmemd -i in -p prmtop -c rst -o out -x nc -r rst`. This is the same idea: the science
lives in the input file, and every concrete path arrives on the command line. OpenMM needs both a
topology and a serialized System, so `-p` and `-s` are separate.

What this program does NOT do is decide anything scientific. No force field, no protocol, no
schedule, no restraint, no temperature, no step count, no ensemble. All of that is visible in the
Python protocol file it runs. This program resolves paths, refuses to clobber results, runs the
protocol with its output captured, and reports what happened.

Standard library and OpenMM only. No `md_templates`, no YAML, no Git -- a generated system must
still run once this package is gone.

The protocol contract is one function:

    def run(files):        # files.topology, files.system, files.coordinates,
        ...                # files.trajectory, files.restart, files.checkpoint, files.solute_x

`files` is a plain namespace. There are no plugins, registries or base classes, and there is not
going to be a framework here.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

#: Flag, environment fallback, whether the file must already exist, and whether it is required.
#: The environment fallback exists for running a protocol by hand; generated launchers always pass
#: every path explicitly, so the wiring stays visible in the shell file rather than in a variable.
SPEC = [
    ("input",       "-i", "--input",       "OPENMM_INPUT",       True,  True),
    ("topology",    "-p", "--topology",    "OPENMM_TOPOLOGY",    True,  True),
    ("system",      "-s", "--system",      "OPENMM_SYSTEM",      True,  True),
    ("coordinates", "-c", "--coordinates", "OPENMM_COORDINATES", True,  True),
    ("output",      "-o", "--output",      "OPENMM_OUTPUT",      False, True),
    ("trajectory",  "-x", "--trajectory",  "OPENMM_TRAJECTORY",  False, False),
    ("restart",     "-r", "--restart",     "OPENMM_RESTART",     False, True),
    ("checkpoint",  None, "--checkpoint",  "OPENMM_CHECKPOINT",  False, False),
    ("solute_x",    None, "--solute-x",    "OPENMM_SOLUTE_X",    False, False),
]

#: Written by the protocol, never by this program, and only once the outputs it names exist.
COMPLETION_MARKER = "run_status: completed"


def _parse(argv):
    parser = argparse.ArgumentParser(
        prog="openmm-md",
        description="Run one OpenMM protocol file against explicit input and output paths.")
    for name, short, long, environment, _must_exist, _required in SPEC:
        flags = [f for f in (short, long) if f]
        parser.add_argument(*flags, dest=name, default=None,
                            help=f"{long.lstrip('-')} (or ${environment})")
    parser.add_argument("--force", action="store_true",
                        help="replace existing runtime outputs instead of refusing")
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
    return SimpleNamespace(**files), problems


def _outputs(files) -> dict[str, str]:
    return {name: getattr(files, name)
            for name in ("output", "trajectory", "restart", "checkpoint", "solute_x")
            if getattr(files, name)}


def validate(files, *, force: bool) -> list[str]:
    """Everything that can be decided without touching OpenMM or writing a byte."""
    problems: list[str] = []

    inputs = {name: getattr(files, name)
              for name in ("input", "topology", "system", "coordinates") if getattr(files, name)}
    outputs = _outputs(files)

    # An output that is also an input destroys the thing it reads from.
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

    # Refusing an existing result is the whole point: a rerun that silently overwrote a finished
    # trajectory would leave a .out describing a run whose outputs came from a different one.
    if not force:
        existing = [f"--{name} {value}" for name, value in outputs.items()
                    if Path(value).exists()]
        if existing:
            problems.append(
                "these outputs already exist: " + ", ".join(existing)
                + ". Pass --force to replace them deliberately.")
    return problems


def load_protocol(path: str):
    """Import the protocol file by location, so it needs no package and no sys.path entry."""
    spec = importlib.util.spec_from_file_location("openmm_md_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} could not be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise RuntimeError(
            f"{path} defines no run(files). A protocol file is one function taking the resolved "
            f"paths; see the generated stages for the shape.")
    return module


def main(argv=None) -> int:
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    files, problems = resolve(arguments)
    problems += validate(files, force=arguments.force)
    if problems:
        for problem in problems:
            print(f"openmm-md: {problem}", file=sys.stderr)
        return 2                                   # nothing was created, nothing was read

    # Only now, after every check passed, may anything appear on disk.
    for value in _outputs(files).values():
        Path(value).parent.mkdir(parents=True, exist_ok=True)

    report = Path(files.output)
    status = 0
    with open(report, "w", encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                load_protocol(files.input).run(files)
            except SystemExit as exit_request:            # a protocol may exit deliberately
                status = int(exit_request.code or 0)
            except BaseException:                          # noqa: BLE001 - the .out is the report
                traceback.print_exc()
                status = 1

    # The protocol prints the completion marker. This program never does -- but it does insist the
    # marker is not standing over missing outputs, which would be a completed run with nothing to
    # show for it.
    if status == 0:
        promised = {name: value for name, value in _outputs(files).items() if name != "output"}
        missing = [f"--{name} {value}" for name, value in promised.items()
                   if not Path(value).exists()]
        if missing:
            with open(report, "a", encoding="utf-8") as handle:
                handle.write("openmm-md: the protocol finished but did not write: "
                             + ", ".join(missing) + "\n")
            print("openmm-md: the protocol finished but did not write: " + ", ".join(missing),
                  file=sys.stderr)
            status = 1
        elif COMPLETION_MARKER not in report.read_text(encoding="utf-8"):
            print(f"openmm-md: {report} has no '{COMPLETION_MARKER}' line; treating as incomplete",
                  file=sys.stderr)
            status = 1

    if status != 0:
        print(f"openmm-md: stage failed; see {report}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
