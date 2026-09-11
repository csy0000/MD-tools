"""Turn a finished cMD run into a standalone, MD-tools-free bundle.

The one thing this module must not get wrong is WHICH System the bundle carries. The file named
on the command line -- `implicit.xml`, `explicit.xml` -- is the BUILD System, and it is not what
any stage integrates. Between that file and the Context, `run/preflight._prepare_stage` applies
the REST2 solute scaling for a fixed-tau run, adds the positional-restraint force (present at
zero strength even when nothing is restrained, so the Force layout stays stable across a
checkpoint), and adds the barostat for an explicit-solvent stage -- inactive at frequency 0 under
NVT. Exporting the build file would hand out a bundle whose Hamiltonian is not the one that
produced the data beside it, and at tau = 0.5 that is not a subtle difference.

So the bundle's System comes from `_prepare_stage` itself, called with the stage block out of the
run's own machine record. One definition of the integrated System, used by the engine and by the
export, is the only arrangement in which the two cannot disagree.

The same applies to the seed and to the starting state. The integrator seed is
`derive_seed(config_seed, stage_name)`, not the seed written in the config; and a production
stage continues from the state its `-c` named, so it neither reads the topology's coordinates nor
re-draws velocities from a Maxwell-Boltzmann distribution. Both are read from the record.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

RUNNER = '''#!/usr/bin/env python
"""{title}

Standalone. Needs OpenMM and nothing else -- no MD-tools, no AmberTools. The Hamiltonian is
`{system_name}`, the serialised OpenMM System this simulation actually integrated; it is not
rebuilt here, so nothing about the force field can drift between this script and the data beside
it.

    python run.py                 run the whole thing
    python run.py --steps 1000    a short check

WHAT THIS REPRODUCES, AND WHAT IT DOES NOT

    The ensemble, not the trajectory. Given the same seed and the same platform OpenMM is
    deterministic, so a rerun here matches frame for frame. On a different platform, or with a
    different OpenMM, the random stream differs and so do the frames -- the distribution is what
    is being reproduced, and that is the right expectation for a reference.
"""
import argparse
import json
import sys
from pathlib import Path

from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))


def build(platform_name=None, properties=None):
    """The Context this run used: same System, same integrator, same seed, same starting state.

    `{system_name}` already carries everything the production stage integrated -- the solute
    scaling at tau, the positional-restraint force at its configured strength, and the barostat
    if there was one. Nothing is added here, because anything added here could differ from what
    ran.
    """
    pdb = PDBFile(str(HERE / "{topology_name}"))
    system = XmlSerializer.deserialize((HERE / "{system_name}").read_text(encoding="utf-8"))

    integrator = LangevinMiddleIntegrator(
        SETTINGS["temperature_K"] * unit.kelvin,
        SETTINGS["friction_per_ps"] / unit.picosecond,
        SETTINGS["timestep_fs"] * unit.femtosecond)
    integrator.setRandomNumberSeed(SETTINGS["seed"])

    platform = Platform.getPlatformByName(platform_name) if platform_name else None
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)

    start = HERE / "{start_name}"
    if start.is_file():
        # Positions, velocities and box vectors as equilibration left them. The topology's own
        # coordinates are the BUILT structure, which is not where this stage began; starting
        # there would silently re-run an unequilibrated system and call it a reproduction.
        simulation.context.setState(
            XmlSerializer.deserialize(start.read_text(encoding="utf-8")))
    else:
        simulation.context.setPositions(pdb.positions)

    # AFTER the state, never before. The restraint strength is a global Context parameter, and
    # `setState` restores global parameters along with positions -- so a state written by a
    # RESTRAINED equilibration comes back still restrained. Production runs at the value below,
    # normally zero; setting it before the state would be silently undone and the bundle would
    # integrate a restrained system while describing a free one.
    restraint = SETTINGS["restraint"]
    simulation.context.setParameter(restraint["parameter"], restraint["value_kj_per_mol_nm2"])
    return simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run this reference simulation.")
    parser.add_argument("--steps", type=int, default=SETTINGS["steps"],
                        help="override the step count, for a quick check")
    parser.add_argument("--platform", default=None, help="CUDA, CPU, ... (default: OpenMM's)")
    parser.add_argument("--final-state", default="final.xml", metavar="FILE",
                        help="where to serialise the end point (default: final.xml)")
    parser.add_argument("--minimise", action="store_true",
                        help="minimise first; the state in {start_name} is already equilibrated")
    args = parser.parse_args(argv)

    simulation = build(args.platform)
    if args.minimise:
        simulation.minimizeEnergy()
    if not (HERE / "{start_name}").is_file():
        # Only when there is no state to continue from. Drawing fresh velocities on top of a
        # restored state would throw away the equilibration this bundle starts from.
        simulation.context.setVelocitiesToTemperature(
            SETTINGS["temperature_K"] * unit.kelvin, SETTINGS["seed"])

    simulation.reporters.append(DCDReporter(
        str(HERE / "trajectory.dcd"), SETTINGS["trajectory_interval_steps"]))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, SETTINGS["state_interval_steps"], step=True, time=True,
        potentialEnergy=True, temperature=True, speed=True))
    simulation.step(args.steps)

    # The end point, serialised the way the engine serialises its own: positions and velocities
    # in full precision. `trajectory.dcd` is single precision and sampled on an interval, so it
    # cannot answer "did this end where that run ended" -- this file can, and that is the
    # question anyone comparing a bundle against its data is asking.
    final = simulation.context.getState(getPositions=True, getVelocities=True)
    (HERE / args.final_state).write_text(XmlSerializer.serialize(final), encoding="utf-8")
    print(f"done: {{args.steps}} steps of {{SETTINGS['name']}}; final state in {{args.final_state}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

SHELL = '''#!/usr/bin/env bash
# Run this reference simulation. Needs only a Python with OpenMM on it.
#
#   ./run.sh                 the full {steps} steps ({ns:g} ns)
#   ./run.sh --steps 1000    a short check first, which is the sensible way to start
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec python run.py "$@"
'''


def _record(run_dir: Path, stage: str) -> dict[str, Any]:
    from ..build.record import read_record

    log = run_dir / f"{stage}.log"
    if not log.is_file():
        raise FileNotFoundError(f"{log} does not exist, so there is no finished {stage} to export")
    record = read_record(log)
    if record.get("status") != "completed":
        raise ValueError(f"{log} reports status {record.get('status')!r}, not 'completed'. A "
                         f"reference is exported from a finished run, never from a partial one.")
    return record


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _locate(run_dir: Path, entry: dict[str, Any]) -> Path | None:
    """The recorded input, found by DIGEST rather than trusted by path.

    `inputs.topology.path` is relative to the command line, not to the run directory: a stage
    invoked with `-p ../implicit.pdb` records `implicit.pdb`, which does not exist inside the run.
    Searching upward finds it, and the digest is what confirms the right file was found -- two
    campaigns beside each other can easily both hold a `built.pdb`.
    """
    name = Path(entry["path"]).name
    wanted = entry["sha256"]
    for directory in (run_dir, *run_dir.parents[:3]):
        candidate = directory / name
        if candidate.is_file() and _digest(candidate) == wanted:
            return candidate
    return None


def _continue_from(record: dict[str, Any]) -> str | None:
    """What the stage's `-c` named, read off the recorded command line.

    The record's `inputs` block holds the topology and the System and not the starting state, so
    the command is the only place this is written down. It is data the run wrote about itself,
    not a guess: `md-run` records `command` verbatim.
    """
    command = list(record.get("command") or [])
    if "-c" in command:
        index = command.index("-c")
        if index + 1 < len(command):
            return command[index + 1]
    return None


def export_reference(run_dir: Path, out_dir: Path, *, stage: str = "cMD") -> dict[str, Any]:
    """Write a standalone bundle for one finished stage. Returns its manifest."""
    from ..md._stages import KCAL_PER_MOL_ANGSTROM2, RESTRAINT_PARAMETER, derive_seed
    from ..run.preflight import _prepare_stage, load_inputs

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    record = _record(run_dir, stage)
    block = record.get("stage") or {}
    inputs = record.get("inputs") or {}

    if (block.get("umbrella_file") or (block.get("collective_variables") or {}).get("file")):
        raise ValueError(
            f"{run_dir} ran with umbrella biases or a collective-variable definition. Those are "
            f"extra forces and extra reporting that this bundle does not carry, and exporting it "
            f"without them would produce a script that runs and samples something else.")

    found = {}
    for role, name in (("topology", "topology.pdb"), ("system", "build-system.xml")):
        source = _locate(run_dir, inputs[role])
        if source is None:
            raise FileNotFoundError(
                f"{role} {inputs[role]['path']!r} (sha256 {inputs[role]['sha256'][:16]}...) is not "
                f"beside {run_dir} or above it. The run recorded it relative to the command line "
                f"rather than to the directory, so a run started with `-p ../built.pdb` names it "
                f"`built.pdb` here; the file has to be found by digest.")
        found[role] = source

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found["topology"], out_dir / "topology.pdb")

    # -- the System the stage INTEGRATED, not the one it was built from -------------------------
    #
    # `_prepare_stage` is the engine's own preparation: scale at tau, then restrain, then add the
    # barostat, in that order and with those seeds. Calling it here rather than reimplementing it
    # is the whole reason the exported Hamiltonian cannot drift from the one that ran.
    loaded = load_inputs(found["topology"], found["system"])
    prepared = _prepare_stage(loaded, stage=block, name=block["name"],
                              where=f"reference export of {run_dir}")
    from openmm import XmlSerializer

    (out_dir / "system.xml").write_text(
        XmlSerializer.serialize(prepared["prepared_system"]), encoding="utf-8")

    # -- the state the stage continued from ----------------------------------------------------
    start_name = ""
    parent = _continue_from(record)
    if parent:
        source = run_dir / Path(parent).name
        if not source.is_file():
            raise FileNotFoundError(
                f"{run_dir} continued from {parent!r}, which is not there now. The bundle would "
                f"otherwise start from the built coordinates -- an unequilibrated structure -- "
                f"and present the result as a reproduction of this run.")
        start_name = "start.xml"
        shutil.copy2(source, out_dir / start_name)

    settings = {
        "name": block["name"],
        "ensemble": block["ensemble"],
        "steps": int(block["steps"]),
        "timestep_fs": float(block["timestep_fs"]),
        "temperature_K": float(block["temperature_K"]),
        "friction_per_ps": float(block["friction_per_ps"]),
        "pressure_bar": float(block.get("pressure_bar") or 0.0),
        "barostat_interval_steps": int(block.get("barostat_interval_steps") or 0),
        # The integrator's seed, which is derived from the configured one and the stage name --
        # the raw config value is written beside it so the derivation stays checkable.
        "seed": int(prepared["seed"]),
        "config_seed": int(block["seed"]),
        "tau": float(block.get("tau") or 0.0),
        # The positional restraint is a Force that is always in the System and a global parameter
        # that decides whether it does anything. Both halves have to travel with the bundle.
        "restraint": {
            "parameter": RESTRAINT_PARAMETER,
            "kcal_per_mol_A2": float(block.get("restraint_kcal_per_mol_A2") or 0.0),
            "value_kj_per_mol_nm2": (float(block.get("restraint_kcal_per_mol_A2") or 0.0)
                                     * KCAL_PER_MOL_ANGSTROM2),
        },
        "trajectory_interval_steps": int(block.get("trajectory_interval_steps") or 0),
        "state_interval_steps": int(block.get("state_interval_steps") or 0) or 10000,
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    environment = record.get("environment") or {}
    provenance = {
        "produced_by": "md-tools",
        "md_tools_version": (environment.get("packages") or {}).get("md-tools"),
        "md_tools_commit": environment.get("md_tools_commit"),
        "openmm": (environment.get("packages") or {}).get("openmm"),
        "python": (environment.get("packages") or {}).get("python"),
        "stage_fingerprint": record.get("fingerprint"),
        "started_utc": record.get("started_utc"),
        "finished_utc": record.get("finished_utc"),
        "built_from": {
            "topology": {"path": inputs["topology"]["path"],
                         "sha256": _digest(found["topology"])},
            "system": {"path": inputs["system"]["path"], "sha256": _digest(found["system"])},
            "continued_from": parent,
        },
        "note": "system.xml here is the System this stage integrated: the build System above with "
                "the solute scaled at tau, the positional-restraint force added, and the barostat "
                "added for explicit solvent. It is not the build System. This bundle needs OpenMM "
                "only; a different OpenMM may give a different random stream and so different "
                "frames.",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                             encoding="utf-8")

    title = f"{block['name']}: {settings['steps'] * settings['timestep_fs'] * 1e-6:g} ns " \
            f"{settings['ensemble']}, tau = {settings['tau']:g}"
    (out_dir / "run.py").write_text(
        RUNNER.format(title=title, system_name="system.xml", topology_name="topology.pdb",
                      start_name=start_name or "start.xml"),
        encoding="utf-8")
    shell = out_dir / "run.sh"
    shell.write_text(SHELL.format(steps=settings["steps"],
                                  ns=settings["steps"] * settings["timestep_fs"] * 1e-6),
                     encoding="utf-8")
    shell.chmod(0o755)

    lines = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{_digest(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"settings": settings, "provenance": provenance, "files": len(lines) + 1,
            "derived_seed": int(prepared["seed"])}
