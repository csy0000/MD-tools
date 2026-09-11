"""Turn a finished cMD run into a standalone, MD-tools-free bundle."""
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

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))


def build(platform_name=None, properties=None):
    """The Context this run used: same System, same integrator, same seed."""
    pdb = PDBFile(str(HERE / "{topology_name}"))
    system = XmlSerializer.deserialize((HERE / "{system_name}").read_text(encoding="utf-8"))

    if SETTINGS["ensemble"] == "NPT":
        # Added HERE rather than baked into system.xml, because the serialised System is the one
        # the production stage integrated and a barostat is a property of the stage.
        system.addForce(MonteCarloBarostat(
            SETTINGS["pressure_bar"] * unit.bar,
            SETTINGS["temperature_K"] * unit.kelvin,
            SETTINGS["barostat_interval_steps"]))

    integrator = LangevinMiddleIntegrator(
        SETTINGS["temperature_K"] * unit.kelvin,
        SETTINGS["friction_per_ps"] / unit.picosecond,
        SETTINGS["timestep_fs"] * unit.femtosecond)
    integrator.setRandomNumberSeed(SETTINGS["seed"])

    platform = Platform.getPlatformByName(platform_name) if platform_name else None
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    return simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run this reference simulation.")
    parser.add_argument("--steps", type=int, default=SETTINGS["steps"],
                        help="override the step count, for a quick check")
    parser.add_argument("--platform", default=None, help="CUDA, CPU, ... (default: OpenMM's)")
    parser.add_argument("--minimise", action="store_true",
                        help="minimise first; the reference state is already equilibrated")
    args = parser.parse_args(argv)

    simulation = build(args.platform)
    if args.minimise:
        simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(
        SETTINGS["temperature_K"] * unit.kelvin, SETTINGS["seed"])

    simulation.reporters.append(DCDReporter(
        str(HERE / "trajectory.dcd"), SETTINGS["trajectory_interval_steps"]))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, SETTINGS["state_interval_steps"], step=True, time=True,
        potentialEnergy=True, temperature=True, speed=True))
    simulation.step(args.steps)
    print(f"done: {{args.steps}} steps of {{SETTINGS['name']}}")
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


def export_reference(run_dir: Path, out_dir: Path, *, stage: str = "cMD") -> dict[str, Any]:
    """Write a standalone bundle for one finished cMD stage. Returns its manifest."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    record = _record(run_dir, stage)
    block = record.get("stage") or {}
    inputs = record.get("inputs") or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for role, name in (("topology", "topology.pdb"), ("system", "system.xml")):
        source = _locate(run_dir, inputs[role])
        if source is None:
            raise FileNotFoundError(
                f"{role} {inputs[role]['path']!r} (sha256 {inputs[role]['sha256'][:16]}...) is not "
                f"beside {run_dir} or above it. The run recorded it relative to the command line "
                f"rather than to the directory, so a run started with `-p ../built.pdb` names it "
                f"`built.pdb` here; the file has to be found by digest.")
        # `_locate` already proved this is the file the run integrated, by digest.
        shutil.copy2(source, out_dir / name)
        actual = _digest(source)
        copied[name] = actual

    settings = {
        "name": block["name"],
        "ensemble": block["ensemble"],
        "steps": int(block["steps"]),
        "timestep_fs": float(block["timestep_fs"]),
        "temperature_K": float(block["temperature_K"]),
        "friction_per_ps": float(block["friction_per_ps"]),
        "pressure_bar": float(block.get("pressure_bar") or 0.0),
        "barostat_interval_steps": int(block.get("barostat_interval_steps") or 0),
        "seed": int(block["seed"]),
        "tau": float(block.get("tau") or 0.0),
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
        "inputs": {name: digest for name, digest in copied.items()},
        "note": "This bundle needs OpenMM only. The System was integrated by the versions above; "
                "a different OpenMM may give a different random stream and so different frames.",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                             encoding="utf-8")

    title = f"{block['name']}: {settings['steps'] * settings['timestep_fs'] * 1e-6:g} ns " \
            f"{settings['ensemble']}, tau = {settings['tau']:g}"
    (out_dir / "run.py").write_text(
        RUNNER.format(title=title, system_name="system.xml", topology_name="topology.pdb"),
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
    return {"settings": settings, "provenance": provenance, "files": len(lines) + 1}
