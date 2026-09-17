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
PROVENANCE = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))


def check_openmm():
    """Say when the running OpenMM is not the one that produced the data beside this script.

    Not a refusal: running a reference on a newer OpenMM is frequently the whole point. But the
    random stream and the order of force summation both change with it, so the frames WILL
    differ -- and that must not be something a reader discovers by accident after comparing two
    trajectories and concluding the reference is wrong.

    The comparison is on the BUILD string, not on `openmm.__version__`. The latter is only
    major.minor -- "8.6" for every 8.6.x -- so a different patch release and a different dev
    build both compare equal to it, and both change the frames. The commit is printed because a
    dev build is identified by nothing else: "8.6.0.dev-c6173db" is a commit, not a release.
    """
    import openmm

    running_build = getattr(openmm.version, "version", None) or openmm.__version__
    revision = getattr(openmm.version, "git_revision", None)
    recorded_build = PROVENANCE.get("openmm_build")
    recorded = recorded_build or PROVENANCE.get("openmm")

    print(f"# openmm            : {{running_build}} (this bundle was produced with {{recorded}})")
    if revision:
        print(f"# openmm commit     : {{revision}}")
    if recorded_build and running_build != recorded_build:
        print("# NOTE              : OpenMM is a DIFFERENT BUILD from the one that produced this "
              "data. The ensemble is reproduced; individual frames will not be.")
    elif not recorded_build and recorded and openmm.__version__ != recorded:
        # An older bundle, exported before the build string was carried. Only major.minor is
        # available to compare, so say so rather than implying the builds were checked.
        print("# NOTE              : OpenMM differs from the recorded one. The ensemble is "
              "reproduced; individual frames will not be.")
    elif not recorded_build:
        print("# NOTE              : this bundle records only OpenMM's major.minor version, so a "
              "different 8.x.y build cannot be detected here.")


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

    check_openmm()
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


#: What this exporter can turn into a bundle. A stage record describes ONE Context integrated
#: for a fixed number of steps, which is the shape `RUNNER` drives. Everything else -- a replica
#: ladder, an AIS path set -- has a different control flow and needs its own runner and its own
#: equivalence test before it can be handed out.
EXPORTABLE_RECORD_PREFIX = "md-stage:"


def _record(run_dir: Path, stage: str) -> dict[str, Any]:
    from ..build.record import read_record

    log = run_dir / f"{stage}.log"
    if not log.is_file():
        raise FileNotFoundError(f"{log} does not exist, so there is no finished {stage} to export")
    record = read_record(log)
    if record.get("status") != "completed":
        raise ValueError(f"{log} reports status {record.get('status')!r}, not 'completed'. A "
                         f"reference is exported from a finished run, never from a partial one.")

    # BEFORE anything is created. Asked for a ladder, this used to fall through to `block["name"]`
    # on a record whose block is called `ladder` rather than `stage`, report the bare KeyError key
    # `'name'`, and leave a half-written directory holding a topology behind it. A refusal that
    # writes files is not a refusal.
    kind = str(record.get("record_type") or "")
    if not kind.startswith(EXPORTABLE_RECORD_PREFIX):
        raise ValueError(
            f"{log} is a {kind!r} record, and only {EXPORTABLE_RECORD_PREFIX}* records can be "
            f"exported. This writes a bundle that drives ONE Context through a fixed number of "
            f"steps; a replica ladder or an AIS path set is a different control flow, and a "
            f"bundle claiming to reproduce one without it would run and sample something else. "
            f"Nothing has been written.")
    return record


def _resolved_timestep_fs(record: dict[str, Any], stage_block: dict[str, Any], log: Path) -> float:
    """The timestep the stage INTEGRATED at, never the one it requested.

    `dynamics.timestep_fs` may be the string `auto`, which `md_tools.openmm.timestep` resolves
    against the masses in the built System -- 2 fs on ordinary hydrogens, 4 fs on repartitioned
    ones. The stage block carries the REQUEST verbatim, so reading it and calling `float` on it
    raised `could not convert string to float: 'auto'` for every HMR run, after the bundle
    directory and four of its files already existed.

    The answer is in the same record: the stage writes `resolve_timestep_fs`'s whole record under
    `timestep` (`md_tools.md.stage`), unconditionally and for every stage, so `timestep_fs` there
    is always the resolved number whatever was asked for. The stage block is the fallback for a
    record written before that block existed, and only when it is already numeric -- resolving
    `auto` a second time here would need the System's masses and would be a second implementation
    of the rule.
    """
    resolved = (record.get("timestep") or {}).get("timestep_fs")
    if isinstance(resolved, (int, float)):
        return float(resolved)

    requested = stage_block.get("timestep_fs")
    if isinstance(requested, (int, float)):
        return float(requested)

    raise ValueError(
        f"{log} records dynamics.timestep_fs as {requested!r} and carries no resolved "
        f"`timestep.timestep_fs` beside it, so the timestep this stage integrated at cannot be "
        f"established. It is not recoverable here: resolving {requested!r} needs the masses in "
        f"the built System, which is the run's job and not the exporter's. Nothing has been "
        f"written.")


def _stage_settings(block: dict[str, Any], prepared: dict[str, Any], *,
                    timestep_fs: float) -> dict[str, Any]:
    """Everything `run.py`, `run.sh` and `settings.json` need, built BEFORE the directory exists.

    Every value comes from the record or from `_prepare_stage`, and the one field that had to be
    resolved -- the timestep -- arrives already resolved. So this cannot fail on record content,
    which is the point of it being a function called above the `mkdir`: it used to be a dict
    literal built between the file copies, and one `float('auto')` in it left a bundle directory
    holding four files and none of the three that make it runnable.
    """
    from ..md._stages import KCAL_PER_MOL_ANGSTROM2, RESTRAINT_PARAMETER

    restraint_kcal = float(block.get("restraint_kcal_per_mol_A2") or 0.0)
    return {
        "name": block["name"],
        "ensemble": block["ensemble"],
        "steps": int(block["steps"]),
        "timestep_fs": float(timestep_fs),
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
            "kcal_per_mol_A2": restraint_kcal,
            "value_kj_per_mol_nm2": restraint_kcal * KCAL_PER_MOL_ANGSTROM2,
        },
        "trajectory_interval_steps": int(block.get("trajectory_interval_steps") or 0),
        "state_interval_steps": int(block.get("state_interval_steps") or 0) or 10000,
    }


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
        # `build/` TOO, and not only the ancestor itself. The built System and topology live in
        # `<dataset>/build/` now -- a CHILD of an ancestor, never an ancestor -- so testing only
        # `directory / name` could not find the topology of any run built under this layout, and
        # `export-reference` refused every one of them. The digest still decides: this widens
        # where to look, not what counts as a match, so two campaigns each holding a `built.pdb`
        # are still told apart by content rather than by position.
        # And a SAVED SCALED STATE, `<dataset>/build/<method>/system_state<n>.xml`: a hot stage's
        # `-s` since a stage stopped scaling in memory (docs/amber-like-fix/REST2-scaler.md, step
        # 3). One level further down, and again the digest decides.
        candidates = [directory / name, directory / "build" / name]
        if (directory / "build").is_dir():
            candidates += sorted((directory / "build").glob(f"*/{name}"))
        for candidate in candidates:
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


def _same(left: Any, right: Any) -> bool:
    return (json.dumps(left, sort_keys=True, default=str)
            == json.dumps(right, sort_keys=True, default=str))


def _stage_inputs(run_dir: Path) -> list[tuple[Path, list[str]]]:
    """Every `.in` a stage or ladder in `run_dir` ran from, with the command that ran it, in order.

    Taken from the records IN the run directory and copied from that same directory: the records
    name the file (`-i`), and the file sits in the directory the run wrote, which a registered
    dataset seals in its inventory. The records do not hold the `.in` digest -- `md-run` writes it
    into `resolved.config` only when it creates that file, and build-md usually already has -- so
    the run directory is the proof, and a record naming a file that is not there refuses.
    """
    from ..build.record import RecordError, read_record

    found = []
    # RECURSIVE, because a stage's record is no longer always at the run root. Equilibration
    # writes into `<run>/eq/`, so a root-only glob saw `cMD.log` and never `eq/eq_nvt_posres.log`
    # -- and the bundle then recorded ONE stage where two ran, silently omitting the input the
    # production stage continued from. For a bundle whose purpose is to reproduce a run, an input
    # missing without complaint is worse than a refusal.
    #
    # Order is unaffected: the entries are sorted by each record's own `started_utc` below, not
    # by filename or walk order.
    for log in sorted(Path(run_dir).rglob("*.log")):
        try:
            record = read_record(log)
        except (RecordError, OSError):
            continue
        if not str(record.get("record_type") or "").startswith(("md-stage:", "md-replica:")):
            continue
        command = list(record.get("command") or [])
        if "-i" not in command or command.index("-i") + 1 >= len(command):
            continue
        # THE RECORDED `-i`, RESOLVED AGAINST THE RUN DIRECTORY -- then the bare name beside the
        # records as before.
        #
        # `input/` is SHARED by every run on a system now, so a stage records `-i
        # ../input/eq_1.in` and the run directory holds no `.in` at all. Stripping to the
        # basename and looking only here refused every run built under that layout: the file the
        # record names does exist, one directory up. The basename attempt stays second because
        # that is where it sits for a run generated before the split, when the input and the
        # records shared one directory.
        recorded = Path(command[command.index("-i") + 1])
        candidates = [Path(run_dir) / recorded] if not recorded.is_absolute() else [recorded]
        candidates.append(Path(run_dir) / recorded.name)
        path = next((option.resolve() for option in candidates if option.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"{log.name} ran from {recorded}, which is neither at {candidates[0]} nor beside "
                f"the records in {run_dir}. The stage input the run actually read cannot be "
                f"shown. Nothing has been written.")
        found.append((str(record.get("started_utc") or ""), path, command))
    # A ladder's ranks can each record the same command; one input is one entry.
    ordered, seen = [], set()
    for _, path, command in sorted(found, key=lambda item: item[0]):
        if path.name not in seen:
            seen.add(path.name)
            ordered.append((path, command))
    return ordered


def user_inputs_plan(run_dir: Path, system_path: Path,
                     topology_path: Path | None = None) -> dict[str, Any]:
    """What a person supplied to produce this run, each file found and PROVEN, before any write.

    A bundle's `input/` holds exactly these:

      the structure   what `build-top -i` read, found by the digest its record holds
      build-top       the configuration file `build-top --config` read, accepted only if resolving
                      it NOW reproduces both the resolved configuration and the stated keys the
                      record holds; otherwise the record's resolved configuration, labelled so
      build-md        the run's own `resolved.config`, which is authoritative by contract and which
                      `build-md --config` reads back

    A file is never taken on its name. The campaign this was written for is the reason: its
    protocol configurations were edited after the runs started, so the file named
    `rest2_implicit.config` beside a registered ladder was not the configuration that produced it.
    """
    from ..build.record import RecordError, read_record
    from ..build.top import recorded_configuration

    run_dir, system_path = Path(run_dir), Path(system_path)
    wanted = _digest(system_path)
    found = None
    for log in sorted(system_path.parent.glob("*.log")):
        try:
            record = read_record(log)
        except (RecordError, OSError):
            continue
        output = ((record.get("outputs") or {}).get("system_xml") or {})
        if record.get("record_type") == "build-top" and output.get("sha256") == wanted:
            found = (log, record)
            break
    if found is None:
        raise FileNotFoundError(
            f"no build-top record beside {system_path} names it as its output (sha256 "
            f"{wanted[:16]}...). A bundle's input/ holds only files proven to be the ones used, and "
            f"without that record the structure cannot be proven. Nothing has been written.")
    log, record = found
    command = list(record.get("command") or [])

    def argument(flag):
        if flag in command and command.index(flag) + 1 < len(command):
            return command[command.index(flag) + 1]
        return None

    def candidates(named, name):
        paths = []
        if named:
            path = Path(named)
            paths.append(path if path.is_absolute() else log.parent / path)
        paths += [directory / name for directory in (log.parent, *log.parent.parents[:2], run_dir)]
        return paths

    entry = record.get("input") or {}
    structure = next((path for path in candidates(argument("-i"), Path(entry.get("path", "")).name)
                      if path.is_file() and _digest(path) == entry.get("sha256")), None)
    if structure is None:
        raise FileNotFoundError(
            f"the structure build-top read ({entry.get('path')!r}, sha256 "
            f"{str(entry.get('sha256'))[:16]}...) is not where the record says, nor beside the "
            f"build. input/ would otherwise hold a file that only shares its name. Nothing has "
            f"been written.")

    named_config = argument("--config")
    top_config = None
    for path in (candidates(named_config, Path(named_config).name) if named_config else []):
        if not path.is_file():
            continue
        try:
            resolved, stated = recorded_configuration(path)
        except Exception:
            continue
        if _same(resolved, record.get("resolved_config")) and _same(stated,
                                                                   record.get("stated_keys")):
            top_config = path
            break

    protocol = run_dir / "resolved.config"
    if not protocol.is_file():
        raise FileNotFoundError(f"{run_dir} has no resolved.config, which is the authoritative "
                                f"input build-md generated this run from. Nothing has been written.")
    return {"structure": structure, "build_top_config": top_config,
            "build_top_config_named": named_config,
            "build_top_resolved": record.get("resolved_config"), "build_md_config": protocol,
            "build_system": system_path, "build_topology": topology_path,
            "build_top_record": record, "stage_inputs": _stage_inputs(run_dir)}


INPUT_README = """# Inputs

What this run was made from. Each file here was checked against the run's own records before it
was copied.

{rows}

## Where the structure came from

{origin}

## Four ways to reproduce it

**1. With OpenMM alone, from the bundled state.** No md-tools, no AmberTools, no OpenFF. The
bundle one level up holds the Hamiltonian the run integrated and the state it continued from:

    cd .. && ./run.sh

**2. From the structure, in `openmm-env`, without md-tools.** {standalone}

**3. With md-tools, from the built System.** No `build-top`: these are the commands the run's
own records say it ran, pointed at the files here. Each continues from the state the one before
it wrote; the exported stage's `-c` state is `start.xml` in the bundle one level up, for when the
stages before it are not part of this dataset.

{stage_commands}

**4. With md-tools, from the structure.**

    md-openmm build-top -i input/{structure} {config_flag}-os build/built.xml -op build/built.pdb -log build/built.log
{scaler_command}    md-openmm build-md --config input/build-md.config -odir ./<method>-run1
{scaled}"""

STANDALONE_BUILD = Path(__file__).with_name("standalone_build.py")

#: `md_tools/rest2/hamiltonian.py`, copied byte for byte into a hot run's `input/`: the code that
#: turned `built.xml` into the saved scaled state the stage integrated. It imports only OpenMM.
SCALING_SOURCE = Path(__file__).resolve().parents[1] / "rest2" / "hamiltonian.py"

VERIFY_STATE = '''#!/usr/bin/env python
"""Re-derive the saved scaled state from built.xml, and check it is the state this run integrated.

Standalone. Needs OpenMM and PyYAML, not md-tools. `hamiltonian.py` beside this file is the code md-tools built the
state with (`md-openmm build-top --rest2-scaler`), copied byte for byte, and `scaler.yaml` is the
record it wrote: the tau, the solute atoms, the unscaled central bonds, and whether impropers stay
unscaled. So this is the derivation that ran rather than a restatement of it.

    python input/verify_state.py

Both Systems are serialised by the RUNNING OpenMM before they are compared, so an OpenMM that
formats its XML differently still compares like with like. Exits 0 when identical, 1 otherwise.
"""
import sys
from pathlib import Path

import yaml
from openmm import XmlSerializer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hamiltonian import build_scaled_system    # noqa: E402


def main():
    record = yaml.safe_load((HERE / "scaler.yaml").read_text(encoding="utf-8"))
    torsions = record["unscaled_torsions"]
    solute = [int(i) for i in record["solute"]["atom_indices"]]
    excluded = [tuple(int(a) for a in pair) for pair in torsions["unscaled_central_bonds"]]
    base = XmlSerializer.deserialize((HERE / record["source"]["system"]).read_text(encoding="utf-8"))
    status = 0
    for state in record["states"]:
        bundled = HERE / state["file"]
        if not bundled.is_file():
            continue
        rebuilt = XmlSerializer.serialize(build_scaled_system(
            base, solute, float(state["tau"]), excluded_bonds=excluded,
            unscaled_impropers=bool(torsions["unscaled_impropers"])))
        same = rebuilt == XmlSerializer.serialize(
            XmlSerializer.deserialize(bundled.read_text(encoding="utf-8")))
        print(f"{state['file']}  tau {float(state['tau']):g}  "
              f"{'identical' if same else 'DIFFERS'} when rebuilt from {record['source']['system']}")
        status = status or (0 if same else 1)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
'''

_STEPS = {
    ("peptide", "explicit"): "hydrogens deleted and re-added at pH {ph} (seeded), the {shape} box "
                             "sized from the solute, {water} water and ions added (seeded), the "
                             "System created",
    ("ligand", "explicit"): "{structure_step}, charges assigned "
                            "({charges}), the {shape} box sized, {water} water and ions added "
                            "(seeded), the System created",
    ("peptide", "implicit"): "tleap writes the Amber topology with {radii} radii, ParmEd creates "
                             "the {gb} System",
    ("ligand", "implicit"): "{structure_step}, charges assigned ({charges}), "
                            "the {ligand} parameters written to Amber files through ParmEd, ParmEd "
                            "creates the {gb} System",
}


def standalone_settings(record: dict[str, Any], structure: Path, system: Path,
                        topology: Path) -> dict[str, Any]:
    """What `input/build_system.py` needs: the values the recorded build-top used, and the result.

    Taken from the record's resolved configuration through the same mapping build-top applies
    (`_legacy_cfg`), so the script sees the values the builders saw, not a second reading of the
    user's file.
    """
    from ..openmm.builders import _legacy_cfg

    resolved = record.get("resolved_config") or {}
    cfg = _legacy_cfg(resolved)
    kind = str(cfg["solute"]["kind"])
    route = (record.get("interpretation") or {}).get("route") or (
        "peptide" if kind == "peptide" else "ligand")
    implicit = resolved.get("solvation") == "implicit"
    settings = {
        "schema_version": 1,
        "written_by": "md-openmm export-reference, from the build-top record's resolved_config",
        "route": route,
        "kind": kind,
        "solvent": "implicit" if implicit else "explicit",
        "structure_file": structure.name,
        # WHICH READER the rebuild must use for `structure_file`. `route` cannot answer it: both
        # molecular-graph inputs are the `ligand` route, and handing an SDF to the SMILES reader
        # parses its title line as a SMILES string. Defaulted from the file's own suffix when a
        # record predates the field, so an older bundle keeps rebuilding the way it always did.
        "input_format": str((record.get("interpretation") or {}).get("input_format")
                            or structure.suffix.lstrip(".").lower() or "smi"),
        # The residue name build-top APPLIED to a .smi/.sdf molecule (null for a peptide, and for
        # a record from before 0.5.4, whose molecule kept RDKit's `UNL`).
        "residue_name": (record.get("interpretation") or {}).get("residue_name")
                        if (record.get("outputs") or {}).get("solute_sdf", {}).get("residue_name")
                        else None,
        # A .seq input: the residues and the tleap commands that made its PDB.
        "sequence": record.get("sequence"),
        "builder": {key: cfg[key] for key in ("run", "structure", "protonation", "forcefield",
                                              "solvation", "system_build")},
        "implicit": ({"model": "GBn2", "radii": "mbondi3", "remove_cm_motion": True,
                      "nonpolar_sasa": bool((cfg.get("implicit_solvent") or {})
                                            .get("nonpolar_sasa", False))}
                     if implicit else None),
        "expected": {"system": {"file": system.name, "sha256": _digest(system)},
                     "topology": {"file": topology.name, "sha256": _digest(topology)}},
    }
    if implicit and kind == "peptide-like":
        # The mbondi3 corrections for a peptide-like solute come from md-tools' molecular map,
        # which this script does not carry. Said, rather than built without them.
        settings["unsupported"] = (
            "a peptide-like solute in implicit solvent takes mbondi3 corrections from md-tools' "
            "peptide map, which this script does not reproduce. Rebuild with md-openmm build-top.")
    return settings


def leap_sequence_origin(structure: Path) -> dict[str, Any] | None:
    """If `structure` is exactly what tleap's `sequence {...}` writes for its residues, say so.

    Checked, not assumed: tleap is run on the residue names read from the file, and every atom
    record's first 66 columns -- names, residues, coordinates -- must agree. (The element columns
    are left out because tleap versions differ in whether they write them.) None when tleap is not
    installed, the file is not a PDB, or the two differ anywhere.
    """
    import subprocess
    import tempfile

    structure = Path(structure)
    if structure.suffix.lower() != ".pdb" or shutil.which("tleap") is None:
        return None
    rows = [line[:66] for line in structure.read_text(encoding="utf-8").splitlines()
            if line.startswith(("ATOM", "HETATM"))]
    residues: list[tuple[str, str]] = []
    for line in rows:
        key = line[21:27]
        if not residues or residues[-1][0] != key:
            residues.append((key, line[17:20].strip()))
    names = [name for _, name in residues]
    for leaprc in ("leaprc.protein.ff14SB", "leaprc.protein.ff19SB"):
        commands = [f"source {leaprc}", f"mol = sequence {{ {' '.join(names)} }}",
                    f"savePdb mol {structure.name}", "quit"]
        with tempfile.TemporaryDirectory(prefix="leap-sequence-") as work:
            (Path(work) / "make.leap").write_text("\n".join(commands) + "\n", encoding="utf-8")
            subprocess.run(["tleap", "-f", "make.leap"], cwd=work, capture_output=True,
                           text=True, check=False)
            made = Path(work) / structure.name
            if not made.is_file():
                continue
            made_rows = [line[:66] for line in made.read_text(encoding="utf-8").splitlines()
                         if line.startswith(("ATOM", "HETATM"))]
        if made_rows == rows:
            return {"leaprc": leaprc, "sequence": names, "script": "\n".join(commands) + "\n"}
    return None


def _origin_text(structure: Path, settings: dict[str, Any], leap: dict[str, Any] | None) -> str:
    if settings["route"] == "ligand":
        etkdg = settings["builder"]["structure"]["etkdg"]
        mmff = settings["builder"]["structure"]["mmff"]
        seed = etkdg["seed"] if etkdg["seed"] is not None else settings["builder"]["run"]["seed"]
        return (f"No 3D structure was supplied: `{structure.name}` holds a SMILES string, and "
                f"`build_system.py` makes the coordinates from it -- RDKit ETKDGv3 with seed "
                f"{seed} embeds {etkdg['n_conformers']} conformers, each is minimised with "
                f"{mmff['variant']}, and the lowest in energy is kept. Hydrogens and protonation "
                f"are exactly as the SMILES writes them.")
    if settings.get("sequence"):
        made = settings["sequence"]
        return (f"`{structure.name}` holds a residue sequence, `{' '.join(made['residues'])}`, "
                f"not coordinates. build-top made the peptide with AmberTools' tleap "
                f"(`{made['leaprc']}`, `sequence {{ ... }}`), in the EXTENDED conformation tleap's "
                f"library gives, and `build_system.py` runs the same tleap commands.")
    if leap is not None:
        return (f"`{structure.name}` is exactly what AmberTools' tleap writes for the sequence "
                f"`{{ {' '.join(leap['sequence'])} }}` -- every atom name, residue and coordinate, "
                f"compared when this bundle was exported. `structure.leap` makes it:\n\n"
                f"    cd input && tleap -f structure.leap")
    return (f"`{structure.name}` was supplied to build-top as a file. How it was made is not "
            f"recorded; it is not tleap's `sequence` output for its residues.")


def _standalone_text(settings: dict[str, Any]) -> str:
    if settings.get("unsupported"):
        return f"Not available for this build: {settings['unsupported']}"
    b = settings["builder"]
    # What the ligand routes say about their own first step depends on which input was given:
    # claiming a structure was "made from the SMILES" when an SDF supplied it would misdescribe
    # the bundle in the bundle's own README.
    structure_step = ("the 3D structure taken from the supplied SDF, as given"
                      if str(settings.get("input_format", "smi")).lower() == "sdf"
                      else "the 3D structure made from the SMILES")
    steps = _STEPS[(settings["route"], settings["solvent"])].format(
        ph=b["protonation"]["ph"], shape=b["solvation"]["box_shape"],
        water=str(b["solvation"]["water_model"]).upper(), structure_step=structure_step,
        charges=b["forcefield"]["ligand_charge_method"], ligand=b["forcefield"]["ligand"],
        radii=(settings["implicit"] or {}).get("radii"), gb=(settings["implicit"] or {}).get("model"))
    caveat = ""
    if (settings["route"] == "ligand"
            and str(b["forcefield"]["ligand_charge_method"]).lower() == "am1bcc"):
        caveat = (" For this molecule that equality is not guaranteed: the OpenFF toolkit computes "
                  "AM1-BCC charges on a conformer it generates itself, unseeded, so a flexible "
                  "molecule can rebuild with different charges and the script then reports "
                  "DIFFERS. A rigid one lands on the same conformer every time. Either way the "
                  "System here is the one the run used.")
    return (f"`build_system.py` performs every step `build-top` performed -- {steps} -- as plain "
            f"library calls, with this build's values from `build_settings.json`, and compares "
            f"what it builds with the System and topology here:\n\n"
            f"    python input/build_system.py --out rebuilt\n\n"
            f"It exits 0 only when the rebuilt System is byte-identical and the topology identical "
            f"apart from the date OpenMM writes into its first line.{caveat} The rebuilt pair then "
            f"runs through route 1 or 3. It needs the libraries build-top uses and nothing of "
            f"md-tools: OpenMM"
            + (", AmberTools (tleap) and ParmEd" if settings["route"] == "peptide"
               and settings["solvent"] == "implicit" else "")
            + (", RDKit, the OpenFF toolkit, openmmforcefields and AmberTools"
               + (" and ParmEd" if settings["solvent"] == "implicit" else "")
               if settings["route"] == "ligand" else "")
            + ".")


def _stage_command(command: list[str], system: str, topology: str) -> str:
    """A recorded `md-run` command, pointed at `input/`, with no path from the machine it ran on.

    The record holds argv as launched -- an interpreter, a script path, absolute file names --
    and none of that belongs in a bundle. The command is rebuilt from `md-run` onwards and every
    absolute path is reduced to its file name, which is what it is called in the run directory.
    """
    words = list(command)
    if "md-run" in words:
        words = ["md-openmm", *words[words.index("md-run"):]]
    words = [Path(word).name if word.startswith("/") else word for word in words]
    for flag, value in (("-i", None), ("-p", f"input/{topology}"), ("-s", f"input/{system}")):
        if flag in words and words.index(flag) + 1 < len(words):
            at = words.index(flag) + 1
            words[at] = f"input/{Path(words[at]).name}" if value is None else value
    if "-ng" in words and words.index("-ng") + 1 < len(words):
        words = ["mpirun", "-n", words[words.index("-ng") + 1], *words]
    return "    " + " ".join(words)


def write_user_inputs(plan: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Copy what `user_inputs_plan` proved into `out_dir/input/`, and describe it."""
    import yaml

    target = Path(out_dir) / "input"
    target.mkdir(parents=True, exist_ok=True)
    structure = Path(plan["structure"])
    shutil.copy2(structure, target / structure.name)
    manifest = {"structure": {"file": f"input/{structure.name}",
                              "sha256": _digest(target / structure.name),
                              "verified": "sha256 matches the build-top record's input"}}
    rows = [f"| `{structure.name}` | the structure `build-top -i` read |"]
    config_flag = ""
    if plan["build_top_config"] is not None:
        shutil.copy2(plan["build_top_config"], target / "build-top.config")
        manifest["build_top_config"] = {
            "file": "input/build-top.config", "sha256": _digest(target / "build-top.config"),
            "verified": "resolves to the build-top record's resolved configuration and stated keys"}
        rows.append("| `build-top.config` | the configuration `build-top --config` read |")
        config_flag = "--config input/build-top.config "
    elif plan["build_top_config_named"]:
        (target / "build-top.resolved.yaml").write_text(
            yaml.safe_dump(plan["build_top_resolved"], sort_keys=False), encoding="utf-8")
        manifest["build_top_config"] = {
            "file": "input/build-top.resolved.yaml", "verified": False,
            "why": (f"the configuration build-top read ({plan['build_top_config_named']!r}) was "
                    f"not found, or no longer resolves to what the record holds. This is the "
                    f"record's resolved configuration: what the build used, in the form the "
                    f"record keeps it, which build-top does not read back.")}
        rows.append("| `build-top.resolved.yaml` | what build-top resolved; its original "
                    "configuration file could not be verified |")
    else:
        manifest["build_top_config"] = {"file": None,
                                        "why": "build-top ran with its built-in defaults"}
        rows.append("| (none) | build-top ran with its built-in defaults |")
    shutil.copy2(plan["build_md_config"], target / "build-md.config")
    manifest["build_md_config"] = {
        "file": "input/build-md.config", "sha256": _digest(target / "build-md.config"),
        "verified": "the run's own resolved.config, authoritative by contract"}
    rows.append("| `build-md.config` | the run's `resolved.config`, which `build-md` reads back |")

    # The OpenMM inputs every stage ran on: found by the digest the run records hold.
    names = {}
    for role, key in (("system", "build_system"), ("topology", "build_topology")):
        source = plan.get(key)
        if source is None:
            continue
        source = Path(source)
        shutil.copy2(source, target / source.name)
        names[role] = source.name
        manifest[f"build_{role}"] = {
            "file": f"input/{source.name}", "sha256": _digest(target / source.name),
            "verified": f"sha256 matches the run record's inputs.{role} and the build-top output"}
        rows.append(f"| `{source.name}` | the built {'System' if role == 'system' else 'topology'}"
                    f" every stage ran on (`-{'s' if role == 'system' else 'p'}`) |")

    # THE SAVED SCALED STATE a hot stage integrated, with the record that says how it was made from
    # the built System above, and the code that made it -- so the chain from the structure to the
    # Hamiltonian that ran is proven end to end.
    scaled = plan.get("scaled_state")
    scaler_command = scaled_text = ""
    ladder_scaler = plan.get("scaled_ladder")
    if ladder_scaler is not None:
        # A REST2 LADDER's states travel one level up, as `system_rung<i>.xml` beside
        # `scaler.yaml` and `verify_rungs.py`; what belongs here is how to MAKE them again.
        config_name = "<scaler.config>"
        if ladder_scaler.get("config") is not None:
            shutil.copy2(ladder_scaler["config"], target / "scaler.config")
            manifest["scaler_config"] = {
                "file": "input/scaler.config", "sha256": _digest(target / "scaler.config"),
                "verified": "sha256 matches the one scaler.yaml records"}
            rows.append("| `scaler.config` | the configuration `build-top --rest2-scaler` read |")
            config_name = "input/scaler.config"
        else:
            manifest["scaler_config"] = {"file": None, "verified": False,
                                         "why": "the scaler.config whose sha256 scaler.yaml "
                                                "records was not found"}
        scaler_command = (f"    md-openmm build-top --rest2-scaler -s build/built.xml "
                          f"-p build/built.pdb --config {config_name}\n")
        scaled_text = ("\n## The scaled states\n\nThe ladder integrated the saved states "
                       "`build/REST2/system_state<i>.xml`, one per line of the group file "
                       "`build-md` writes (`remd_groupfile.1`); a ladder reads `-s` only from "
                       "there. They are bundled one level up as `system_rung<i>.xml`, with the "
                       "`scaler.yaml` that made them. Make them again from the built System with "
                       "the command in route 4, or check them with OpenMM alone:\n\n"
                       "    cd .. && python verify_rungs.py\n")
    if scaled is not None:
        state, scaler_record = Path(scaled["state"]), Path(scaled["record"])
        shutil.copy2(state, target / state.name)
        shutil.copy2(scaler_record, target / "scaler.yaml")
        shutil.copy2(SCALING_SOURCE, target / "hamiltonian.py")
        (target / "verify_state.py").write_text(VERIFY_STATE, encoding="utf-8")
        (target / "verify_state.py").chmod(0o755)
        names["system"] = state.name
        manifest["scaled_state"] = {
            "file": f"input/{state.name}", "sha256": _digest(target / state.name),
            "method": scaled["identity"]["method"], "state": scaled["identity"]["state"],
            "tau": scaled["identity"]["tau"],
            "verified": ("sha256 matches the run record's inputs.system and the scaler.yaml "
                         "entry; scaler.yaml names input/" + Path(plan["build_system"]).name
                         + " as its source by sha256")}
        manifest["scaler_record"] = {"file": "input/scaler.yaml",
                                     "sha256": _digest(target / "scaler.yaml")}
        manifest["scaling_code"] = {
            "file": "input/hamiltonian.py", "sha256": _digest(target / "hamiltonian.py"),
            "copied_from": "md_tools/rest2/hamiltonian.py, byte for byte"}
        rows += [f"| `{state.name}` | the saved scaled state every hot stage ran on (`-s`) |",
                 "| `scaler.yaml` | how that state was made from the built System |",
                 "| `hamiltonian.py` | the code that made it, byte for byte |",
                 "| `verify_state.py` | re-derives the state from the built System with OpenMM "
                 "alone |"]
        config_name = "<scaler.config>"
        if scaled.get("config") is not None:
            shutil.copy2(scaled["config"], target / "scaler.config")
            manifest["scaler_config"] = {
                "file": "input/scaler.config", "sha256": _digest(target / "scaler.config"),
                "verified": "sha256 matches the one scaler.yaml records"}
            rows.append("| `scaler.config` | the configuration `build-top --rest2-scaler` read |")
            config_name = "input/scaler.config"
        else:
            manifest["scaler_config"] = {"file": None, "verified": False,
                                         "why": "the scaler.config whose sha256 scaler.yaml "
                                                "records was not found"}
        scaler_command = (f"    md-openmm build-top --rest2-scaler -s build/built.xml "
                          f"-p build/built.pdb --config {config_name}\n")
        scaled_text = ("\n## The scaled state\n\nThe hot stages ran on `" + state.name + "`, not on "
                       "the built System: it is the built System with the solute scaled at tau = "
                       f"{scaled['identity']['tau']:g}, written by `build-top --rest2-scaler`. "
                       "Check it came from the built System here, with OpenMM alone:\n\n"
                       "    python input/verify_state.py\n")

    # The Amber-like stage inputs, exactly as md-run read them.
    commands = []
    manifest["stage_inputs"] = []
    for path, command in plan.get("stage_inputs") or []:
        shutil.copy2(path, target / path.name)
        manifest["stage_inputs"].append({
            "file": f"input/{path.name}", "sha256": _digest(target / path.name),
            "verified": "copied from the run directory, where the stage record that read it lives"})
        rows.append(f"| `{path.name}` | the stage input `md-run -i` read |")
        # A hot stage ran on the scaled state; minimisation on the built System.
        ran_on = names.get("system", "built.xml")
        if scaled is not None and "-s" in command and command.index("-s") + 1 < len(command) \
                and Path(command[command.index("-s") + 1]).name != Path(scaled["state"]).name:
            ran_on = Path(plan["build_system"]).name
        commands.append(_stage_command(command, ran_on, names.get("topology", "built.pdb")))

    # The build itself, as a script that needs openmm-env and not md-tools.
    settings = standalone_settings(plan["build_top_record"], structure,
                                   target / Path(plan["build_system"]).name,
                                   target / names["topology"])
    (target / "build_settings.json").write_text(json.dumps(settings, indent=2) + "\n",
                                                encoding="utf-8")
    manifest["build_settings"] = {"file": "input/build_settings.json",
                                  "sha256": _digest(target / "build_settings.json")}
    rows.append("| `build_settings.json` | the values build-top used, from its record |")
    if not settings.get("unsupported"):
        shutil.copy2(STANDALONE_BUILD, target / "build_system.py")
        (target / "build_system.py").chmod(0o755)
        manifest["build_script"] = {
            "file": "input/build_system.py", "sha256": _digest(target / "build_system.py"),
            "copied_from": "md_tools/reference/standalone_build.py, byte for byte"}
        rows.append("| `build_system.py` | build-top's steps without md-tools; checks its result "
                    "against the files here |")
    leap = leap_sequence_origin(structure) if settings["route"] == "peptide" else None
    if settings.get("sequence"):
        manifest["structure_origin"] = {
            "made_by": "tleap sequence, from the .seq; build_system.py repeats it",
            "leaprc": settings["sequence"]["leaprc"],
            "sequence": settings["sequence"]["residues"],
            "tleap_commands": settings["sequence"]["tleap_commands"]}
    elif leap is not None:
        (target / "structure.leap").write_text(leap["script"], encoding="utf-8")
        manifest["structure_origin"] = {
            "file": "input/structure.leap", "sha256": _digest(target / "structure.leap"),
            "leaprc": leap["leaprc"], "sequence": leap["sequence"],
            "verified": "tleap's output for this sequence matches every atom record of the "
                        "structure in columns 1-66"}
        rows.append(f"| `structure.leap` | the tleap commands that write `{structure.name}` |")
    else:
        manifest["structure_origin"] = (
            {"made_by": "build_system.py, from the SMILES"} if settings["route"] == "ligand"
            else {"made_by": None, "why": "not tleap sequence output, and not recorded"})

    table = "| file | what it is |\n|---|---|\n" + "\n".join(rows)
    (target / "README.md").write_text(
        INPUT_README.format(rows=table, structure=structure.name, config_flag=config_flag,
                            origin=_origin_text(structure, settings, leap),
                            standalone=_standalone_text(settings),
                            stage_commands="\n".join(commands) or
                            "    (the records name no md-run stage inputs)",
                            scaler_command=scaler_command, scaled=scaled_text),
        encoding="utf-8")
    return manifest


def export_reference(run_dir: Path, out_dir: Path, *, stage: str = "cMD") -> dict[str, Any]:
    """Write a standalone bundle for one finished stage. Returns its manifest."""
    from ..md._stages import KCAL_PER_MOL_ANGSTROM2, RESTRAINT_PARAMETER, derive_seed
    from ..run.preflight import _prepare_stage, load_inputs

    # Resolved before anything is searched: the built System is found "beside or above" the run
    # directory, and a relative `.` has no parents -- `-idata .` used to be refused for that.
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
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

    # A SAVED SCALED STATE is followed back to the built System it was made from: the build-top
    # record proves built.xml, and scaler.yaml proves the state came from built.xml.
    from ..rest2.states import ScaledStateError, load_scaler_record, scaled_state_identity

    try:
        identity = scaled_state_identity(found["system"])
    except ScaledStateError as refusal:
        raise ValueError(f"{refusal} Nothing has been written.") from None
    build_system, scaled_plan = found["system"], None
    if identity is not None:
        record_path = Path(identity["record"])
        scaler = load_scaler_record(record_path)
        source = record_path.parent.parent / scaler["source"]["system"]
        if not source.is_file() or _digest(source) != scaler["source"]["system_sha256"]:
            raise FileNotFoundError(
                f"{found['system'].name} was made from {scaler['source']['system']!r} (sha256 "
                f"{scaler['source']['system_sha256'][:16]}...), which is not at {source} with "
                f"that digest. The bundle could not prove the state came from the built System. "
                f"Nothing has been written.")
        if _digest(found["topology"]) != scaler["source"]["topology_sha256"]:
            raise ValueError(
                f"{found['system'].name} was scaled against a topology with a different sha256 "
                f"from the one the stage ran with. Nothing has been written.")
        wanted_config = (scaler.get("config") or {}).get("sha256")
        config = next((path for path in (record_path.parent / scaler["config"]["file"],
                                         record_path.parent.parent / scaler["config"]["file"],
                                         record_path.parent.parent.parent / scaler["config"]["file"])
                       if path.is_file() and _digest(path) == wanted_config), None)
        build_system = source
        scaled_plan = {"state": found["system"], "record": record_path, "identity": identity,
                       "config": config}

    # Proven before the directory exists, like every other refusal here.
    inputs_plan = user_inputs_plan(run_dir, build_system, found["topology"])
    if scaled_plan is not None:
        inputs_plan["scaled_state"] = scaled_plan

    # -- everything that can REFUSE, before anything is written --------------------------------
    #
    # This block used to sit below `out_dir.mkdir()`, interleaved with the copies. Every failure
    # in it therefore left a directory holding `input/`, `topology.pdb`, `system.xml` and
    # `start.xml` and none of `run.py`, `settings.json` or `SHA256SUMS` -- wreckage that passes
    # the eye test, from a module whose every other refusal says "Nothing has been written" and
    # means it. `float(block["timestep_fs"])` on an `auto` record was exactly that: a ValueError
    # raised after four files existed. Nothing below the mkdir may raise on record content now.
    timestep_fs = _resolved_timestep_fs(record, block, run_dir / f"{stage}.log")

    # `_prepare_stage` is the engine's own preparation: scale at tau, then restrain, then add the
    # barostat, in that order and with those seeds. Calling it here rather than reimplementing it
    # is the whole reason the exported Hamiltonian cannot drift from the one that ran. It is pure
    # computation, so it belongs above the mkdir with the other refusals.
    loaded = load_inputs(found["topology"], found["system"])
    prepared = _prepare_stage(loaded, stage=block, name=block["name"],
                              where=f"reference export of {run_dir}")

    start_source = None
    parent = _continue_from(record)
    if parent:
        # THE RECORDED `-c`, RESOLVED AGAINST THE RUN DIRECTORY -- then the bare name beside the
        # records, as before. This is the same correction `_stage_inputs` and `_locate` needed,
        # in the third place that held it: equilibration writes into `<run>/eq/` now, so a stage
        # that recorded `-c eq/eq_nvt_posres.xml` has its parent one directory DOWN, and
        # stripping to the basename looked for it at the run root and refused every run built
        # under this layout. The basename attempt stays as the fallback because that is where it
        # sits for a run generated before the split, when every stage wrote to one directory.
        recorded = Path(parent)
        options = [run_dir / recorded] if not recorded.is_absolute() else [recorded]
        options.append(run_dir / recorded.name)
        start_source = next((option for option in options if option.is_file()), None)
        if start_source is None:
            raise FileNotFoundError(
                f"{run_dir} continued from {parent!r}, which is neither at {options[0]} nor "
                f"beside the records in {run_dir}. The bundle would otherwise start from the "
                f"built coordinates -- an unequilibrated structure -- and present the result as "
                f"a reproduction of this run.")
    start_name = "start.xml" if start_source is not None else ""

    settings = _stage_settings(block, prepared, timestep_fs=timestep_fs)

    # -- from here on, only writes -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found["topology"], out_dir / "topology.pdb")
    user_inputs = write_user_inputs(inputs_plan, out_dir)

    from openmm import XmlSerializer

    (out_dir / "system.xml").write_text(
        XmlSerializer.serialize(prepared["prepared_system"]), encoding="utf-8")
    if start_source is not None:
        shutil.copy2(start_source, out_dir / start_name)

    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    environment = record.get("environment") or {}
    provenance = {
        "produced_by": "md-tools",
        "md_tools_version": (environment.get("packages") or {}).get("md-tools"),
        "md_tools_commit": environment.get("md_tools_commit"),
        "openmm": (environment.get("packages") or {}).get("openmm"),
        # The PRECISE build, which `packages.openmm` is not: that field is
        # `openmm.__version__`, only major.minor, so it reads "8.6" for every 8.6.x -- a
        # different patch release and a different dev build both compare equal to it, and those
        # are exactly the cases that change the random stream and the order of force summation.
        # `acceleration.openmm_version` is `openmm.version.version`, e.g.
        # "8.6.0.dev-c6173db", and it is already in the record this bundle is built from.
        # `packages.openmm_build` is the same string recorded for EVERY record type, including
        # the ladder records that carry no platform block at all; the platform block is
        # preferred because older records have it and not the package field.
        "openmm_build": ((record.get("acceleration") or {}).get("openmm_version")
                         or (environment.get("packages") or {}).get("openmm_build")),
        "python": (environment.get("packages") or {}).get("python"),
        # Everything the run recorded. None of it is needed to RUN this bundle -- the System is
        # frozen, so the tools that built it are out of the picture -- but they are what decided
        # the Hamiltonian, and a reader asking which openff produced these charges should not have
        # to go back to the original run to find out.
        "built_with": dict(sorted((environment.get("packages") or {}).items())),
        "stage_fingerprint": record.get("fingerprint"),
        "started_utc": record.get("started_utc"),
        "finished_utc": record.get("finished_utc"),
        "built_from": {
            "topology": {"path": inputs["topology"]["path"],
                         "sha256": _digest(found["topology"])},
            "system": {"path": inputs["system"]["path"], "sha256": _digest(found["system"])},
            "continued_from": parent,
        },
        "inputs": user_inputs,
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
