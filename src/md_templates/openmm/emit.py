"""Emit the pieces of a generated system: protocol files, launchers, and the path map.

Four pieces, deliberately separate:

* a Python **protocol** file -- the science, and nothing else. It takes its paths as an argument,
  so the same file runs against another system by changing only the shell invocation.
* `paths.sh` -- the portable directory map, resolved from `$MD_DATA`.
* one **launcher** per stage -- every concrete path, visible, passed explicitly.
* `bin/openmm-md` -- generic file handling, copied in, with no scientific opinion.

Every scientific number is a literal resolved at generation. A reader sees `250000`, not
`int(config[...] * 1e6 / timestep)`, and the file needs nothing at run time to explain itself.
"""

from __future__ import annotations

from typing import Any

#: `.out` header grammar version. Any change to the field set bumps this.
OUTPUT_VERSION = 1


def _platform(platform: str, device: Any, precision: Any) -> str:
    """Explicit selection: "whatever OpenMM picked" is not a record of what ran."""
    if str(platform).lower() in ("automatic", "auto", ""):
        return "    platform, properties = None, None   # OpenMM picks the fastest available"
    if str(platform).upper() == "CUDA":
        props = {"DeviceIndex": str(device if device is not None else 0),
                 "Precision": precision or "mixed"}
        return (f'    platform = Platform.getPlatformByName("CUDA")\n'
                f'    properties = {props!r}')
    return (f'    platform = Platform.getPlatformByName("{platform}")\n'
            f'    properties = None')


def _imports(names: tuple[str, ...], app_names: tuple[str, ...]) -> str:
    return (f'import sys\nfrom pathlib import Path\n\n'
            f'import openmm\n'
            f'from openmm import {", ".join(names)}\n'
            f'from openmm.app import {", ".join(app_names)}')


def _header(fields: list[str]) -> str:
    """The `.out` header, written as the text it produces."""
    body = "\n".join(fields)
    return f'    print(f"""MD-OPENMM OUTPUT VERSION: {OUTPUT_VERSION}\n{body}\n""", flush=True)'


def minimization_protocol(r: dict[str, Any]) -> str:
    box = "True" if not r["implicit"] else "False"
    seeds = r["seeds"]["min"]
    return f'''#!/usr/bin/env python
"""Energy minimisation for {r["system_id"]}. Run through: openmm-md -i min.py ..."""

{_imports(("LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"),
          ("PDBFile", "Simulation"))}


def run(files):
    pdb = PDBFile(files.topology)
    system = XmlSerializer.deserialize(Path(files.system).read_text())

    # No step is taken here; a Context needs an integrator to exist.
    integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                          {r["friction_per_ps"]} / unit.picosecond,
                                          {r["timestep_fs"]} * unit.femtoseconds)
    integrator.setRandomNumberSeed({seeds["integrator"]})
{_platform(r["platform"], r.get("device"), r.get("precision"))}
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setState(XmlSerializer.deserialize(Path(files.coordinates).read_text()))

{_header([
    "stage: min", f'system: {r["system_id"]}',
    "openmm_version: {openmm.version.version}",
    "platform: {simulation.context.getPlatform().getName()}",
    "integrator: LangevinMiddleIntegrator",
    f'temperature_K: {r["temperature_K"]}', f'timestep_fs: {r["timestep_fs"]}',
    *( [f'water_model: {r["water_model"]}'] if r.get("water_model") else [] ),
    f'integrator_seed: {seeds["integrator"]}',
    f'max_iterations: {r["minimization_max_iterations"]}',
    "topology: {files.topology}", "system: {files.system}",
    "coordinates: {files.coordinates}", "restart: {files.restart}",
    f'provenance: {r["provenance_hint"]}',
])}

    before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations={r["minimization_max_iterations"]})
    after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"potential_energy_initial: {{before}}")
    print(f"potential_energy_final: {{after}}")

    state = simulation.context.getState(getPositions=True, getVelocities=True,
                                        enforcePeriodicBox={box})
    Path(files.restart).write_text(XmlSerializer.serialize(state))
    print("run_status: completed")
'''


def equilibration_protocol(r: dict[str, Any], stage: dict[str, Any]) -> str:
    name, box = stage["name"], "True" if not r["implicit"] else "False"
    npt = stage["ensemble"] == "NPT"
    restrained = stage["restraint_kcal"] > 0
    first, last = r["solute_range"]
    seeds = r["seeds"][name]

    names = ["LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"]
    if restrained:
        names.insert(0, "CustomExternalForce")
    if npt:
        names.insert(0, "MonteCarloBarostat")

    restraint = f'''
    # Hold the solute -- atoms {first}..{last}, the first residues of the topology -- while the
    # solvent relaxes around it.
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter(
        "k", {stage["restraint_kcal"]} * unit.kilocalories_per_mole / unit.angstrom**2)
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    for index in range({first}, {last + 1}):
        restraint.addParticle(index, pdb.positions[index].value_in_unit(unit.nanometer))
    system.addForce(restraint)
''' if restrained else ""

    barostat = f'''
    barostat = MonteCarloBarostat({r["pressure_bar"]} * unit.bar,
                                  {r["temperature_K"]} * unit.kelvin,
                                  {r["barostat_interval"]})
    barostat.setRandomNumberSeed({seeds["barostat"]})
    system.addForce(barostat)
''' if npt else ""

    extra = "volume=True, density=True, " if npt else ""
    return f'''#!/usr/bin/env python
"""{stage["ensemble"]} equilibration ({name}) for {r["system_id"]}. Run through openmm-md."""

{_imports(tuple(names), ("DCDReporter", "PDBFile", "Simulation", "StateDataReporter"))}


def run(files):
    pdb = PDBFile(files.topology)
    system = XmlSerializer.deserialize(Path(files.system).read_text())
{restraint}{barostat}
    integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                          {r["friction_per_ps"]} / unit.picosecond,
                                          {r["timestep_fs"]} * unit.femtoseconds)
    integrator.setRandomNumberSeed({seeds["integrator"]})
{_platform(r["platform"], r.get("device"), r.get("precision"))}
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setState(XmlSerializer.deserialize(Path(files.coordinates).read_text()))
    # Positions and velocities carry over; the clock does not, so this .out describes this stage.
    simulation.context.setTime(0.0)
    simulation.currentStep = 0

{_header([
    f"stage: {name}", f'system: {r["system_id"]}',
    "openmm_version: {openmm.version.version}",
    "platform: {simulation.context.getPlatform().getName()}",
    "integrator: LangevinMiddleIntegrator", f'ensemble: {stage["ensemble"]}',
    f'temperature_K: {r["temperature_K"]}', f'friction_per_ps: {r["friction_per_ps"]}',
    f'timestep_fs: {r["timestep_fs"]}',
    *( [f'water_model: {r["water_model"]}'] if r.get("water_model") else [] ),
    *( [f'pressure_bar: {r["pressure_bar"]}'] if npt else [] ),
    f'restraint_kcal_mol_A2: {stage["restraint_kcal"]}',
    f'integrator_seed: {seeds["integrator"]}',
    *( [f'barostat_seed: {seeds["barostat"]}'] if npt else [] ),
    f'steps: {stage["steps"]}', f'duration_ps: {stage["duration_ps"]}',
    f'output_interval_steps: {r["output_interval_steps"]}',
    "topology: {files.topology}", "system: {files.system}",
    "coordinates: {files.coordinates}", "trajectory: {files.trajectory}",
    "restart: {files.restart}", "checkpoint: {files.checkpoint}",
    f'provenance: {r["provenance_hint"]}',
])}

    simulation.reporters.append(DCDReporter(files.trajectory, {r["output_interval_steps"]}))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, {r["output_interval_steps"]}, step=True, time=True, potentialEnergy=True,
        temperature=True, {extra}speed=True))

    simulation.step({stage["steps"]})

    state = simulation.context.getState(getPositions=True, getVelocities=True,
                                        enforcePeriodicBox={box})
    Path(files.restart).write_text(XmlSerializer.serialize(state))
    if files.checkpoint:
        simulation.saveCheckpoint(files.checkpoint)
    print("run_status: completed")
'''


def production_protocol(r: dict[str, Any]) -> str:
    box = "True" if not r["implicit"] else "False"
    npt = not r["implicit"]
    frames = r["production_steps"] // r["output_interval_steps"]
    seeds = r["seeds"]["cMD"]

    names = ["LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"]
    if npt:
        names.insert(0, "MonteCarloBarostat")
    barostat = f'''
    barostat = MonteCarloBarostat({r["pressure_bar"]} * unit.bar,
                                  {r["temperature_K"]} * unit.kelvin,
                                  {r["barostat_interval"]})
    barostat.setRandomNumberSeed({seeds["barostat"]})
    system.addForce(barostat)
''' if npt else ""
    extra = "volume=True, density=True, " if npt else ""
    return f'''#!/usr/bin/env python
"""Conventional MD for {r["system_id"]}: {r["production_ps"]:.3f} ps, a frame every {r["output_interval_ps"]:.3f} ps.

DCDReporter writes at step `interval`, not at step 0, so this produces {frames} frames, not {frames + 1}.
Run through openmm-md, which supplies every path.
"""

{_imports(tuple(names), ("CheckpointReporter", "DCDReporter", "PDBFile", "Simulation",
                         "StateDataReporter"))}


def run(files):
    pdb = PDBFile(files.topology)
    system = XmlSerializer.deserialize(Path(files.system).read_text())
{barostat}
    integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                          {r["friction_per_ps"]} / unit.picosecond,
                                          {r["timestep_fs"]} * unit.femtoseconds)
    integrator.setRandomNumberSeed({seeds["integrator"]})
{_platform(r["platform"], r.get("device"), r.get("precision"))}
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setState(XmlSerializer.deserialize(Path(files.coordinates).read_text()))
    simulation.context.setTime(0.0)
    simulation.currentStep = 0

{_header([
    "stage: cMD", f'system: {r["system_id"]}',
    "openmm_version: {openmm.version.version}",
    "platform: {simulation.context.getPlatform().getName()}",
    "integrator: LangevinMiddleIntegrator", f'ensemble: {"NPT" if npt else "NVT"}',
    f'temperature_K: {r["temperature_K"]}', f'friction_per_ps: {r["friction_per_ps"]}',
    f'timestep_fs: {r["timestep_fs"]}',
    *( [f'water_model: {r["water_model"]}'] if r.get("water_model") else [] ),
    *( [f'pressure_bar: {r["pressure_bar"]}'] if npt else [] ),
    f'integrator_seed: {seeds["integrator"]}',
    *( [f'barostat_seed: {seeds["barostat"]}'] if npt else [] ),
    f'steps: {r["production_steps"]}', f'duration_ps: {r["production_ps"]}',
    f'output_interval_steps: {r["output_interval_steps"]}',
    f'output_interval_ps: {r["output_interval_ps"]}', f"frames: {frames}",
    "topology: {files.topology}", "system: {files.system}",
    "coordinates: {files.coordinates}", "trajectory: {files.trajectory}",
    "restart: {files.restart}", "checkpoint: {files.checkpoint}",
    f'provenance: {r["provenance_hint"]}',
])}

    simulation.reporters.append(DCDReporter(files.trajectory, {r["output_interval_steps"]}))
    if files.checkpoint:
        simulation.reporters.append(
            CheckpointReporter(files.checkpoint, {r["output_interval_steps"]}))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, {r["output_interval_steps"]}, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True, temperature=True, {extra}speed=True,
        totalSteps={r["production_steps"]}, remainingTime=True))

    simulation.step({r["production_steps"]})

    state = simulation.context.getState(getPositions=True, getVelocities=True,
                                        enforcePeriodicBox={box})
    Path(files.restart).write_text(XmlSerializer.serialize(state))
    if files.checkpoint:
        simulation.saveCheckpoint(files.checkpoint)
    print("run_status: completed")
'''


# --- the portable path map and the launchers ----------------------------------------------------

def paths_sh(relative_project: str, system_id: str, stage_names: list[str]) -> str:
    """The directory map, resolved from `$MD_DATA` and a relative project path.

    No absolute path appears here. Move the managed root, export the new `$MD_DATA`, and every
    launcher follows -- which is the difference between a system that can be archived and one that
    only works on the machine that made it.
    """
    # Only the equilibration stages have a directory variable here; the production method
    # (cMD or REST2) already has one below, and mapping it twice would be a KeyError.
    eq = [n for n in stage_names if n not in ("cMD", "REST2", "AIS")]
    lines = [
        '#!/usr/bin/env bash',
        '# Portable directory map. Sourced by every stage launcher; contains no machine path.',
        '',
        ': "${MD_DATA:?MD_DATA is not set. Export it to the managed storage root that holds this'
        ' system.}"',
        '',
        f'export PROJECT_DATA="${{MD_DATA}}/{relative_project}"',
        f'export SYSTEM_ROOT="${{PROJECT_DATA}}/{system_id}"',
        '',
        'export INPUT_DIR="${SYSTEM_ROOT}/input"',
        'export MIN_DIR="${SYSTEM_ROOT}/min"',
        'export EQ_DIR="${SYSTEM_ROOT}/eq"',
    ]
    mapping = {"nvt_1kcal": "NVT_DIR", "npt_1kcal": "NPT_RESTRAINED_DIR",
               "npt_free": "NPT_FREE_DIR", "nvt_free": "NVT_FREE_DIR"}
    for name in eq:
        lines.append(f'export {mapping[name]}="${{EQ_DIR}}/{name}"')
    lines += [
        'export CMD_DIR="${SYSTEM_ROOT}/cMD"',
        '',
        '# Reserved for the later REST2/AIS work so the map stays stable across methods.',
        'export CMD_TAU0P5_DIR="${SYSTEM_ROOT}/cMD_tau0p5"',
        'export REST2_DIR="${SYSTEM_ROOT}/REST2"',
        'export AIS_DIR="${SYSTEM_ROOT}/AIS"',
        '',
        '# The generated copy is authoritative for this system. Override deliberately if you have',
        '# a compatible executable elsewhere; ordinary use needs no installed command.',
        'export OPENMM_MD="${OPENMM_MD:-${SYSTEM_ROOT}/bin/openmm-md}"',
        'export OPENMM_REST2="${OPENMM_REST2:-${SYSTEM_ROOT}/bin/openmm-rest2}"',
        '',
    ]
    return "\n".join(lines)


def launcher(stage: str, *, directory_var: str, depth: int, parent_restart: str,
             trajectory: bool, checkpoint: bool) -> str:
    """One stage, every path explicit. The wiring is meant to be read, not inferred."""
    up = "/".join([".."] * depth)
    options = [
        f'  -i "${{{directory_var}}}/{stage}.py"',
        '  -p "${INPUT_DIR}/topology.pdb"',
        '  -s "${INPUT_DIR}/system.xml"',
        f'  -c "{parent_restart}"',
        f'  -o "${{{directory_var}}}/{stage}.out"',
    ]
    if trajectory:
        options.append(f'  -x "${{{directory_var}}}/{stage}.dcd"')
    options.append(f'  -r "${{{directory_var}}}/{stage}.state.xml"')
    if checkpoint:
        options.append(f'  --checkpoint "${{{directory_var}}}/{stage}.chk"')
    body = " \\\n".join(options)
    return f'''#!/usr/bin/env bash
# {stage}: every input and output named explicitly. Re-run with --force to replace outputs.
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{STAGE_DIR}}/{up}/paths.sh"

"${{OPENMM_MD}}" \\
{body} \\
  "$@"
'''


def run_all_sh(stage_paths: list[str]) -> str:
    """Call each launcher in order. No scientific setting, no reconstructed command."""
    calls = "\n".join(f'"${{SYSTEM_ROOT}}/{path}" "$@"' for path in stage_paths)
    return f'''#!/usr/bin/env bash
# Every stage in order. Each launcher owns its own paths and writes its own .out.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/paths.sh"

{calls}
echo "== done =="
'''


# --- REST2 ---------------------------------------------------------------------------------------

def rest2_protocol(r: dict[str, Any]) -> str:
    """The concise `rest2.py`: the scientific protocol and nothing else.

    The exchange implementation is NOT here. It lives in `rest2_runtime.py`, copied into the
    project beside this file, so the file a user reads and edits stays the size of the physics.
    """
    implicit = r["implicit"]
    pressure = "None" if implicit else f'{r["pressure_bar"]}'
    ensemble = "NVT" if implicit else "NPT"
    taus = ", ".join(f"{t:.4f}" for t in r["rest2"]["taus"])
    seeds = r["seeds"]["REST2"]
    return f'''#!/usr/bin/env python
"""REST2 for {r["system_id"]}: {r["rest2"]["number_of_replicas"]} replicas, tau {r["rest2"]["tau_min"]} to {r["rest2"]["tau_max"]}, {ensemble} at {r["temperature_K"]} K.

Omega-selective REST2: torsions about a peptide omega bond are left UNSCALED. Every replica is
thermostatted at the SAME {r["temperature_K"]} K and differs only by Hamiltonian -- this is
Hamiltonian scaling, not temperature REMD.

    tau ladder : [{taus}]
    s = (1-tau)^2 on solute-solute terms, sqrt(s) = 1-tau on solute-environment terms

Exchange scheme: {r["rest2"]["replica_mixing_scheme"]}. The accept/reject decision is made by
{r["rest2"]["exchange_decision_owner"]}; propagation, reduced potentials, NetCDF storage,
checkpointing and restart are OpenMMTools'. md-templates decides only WHICH iterations attempt an
exchange (every {r["rest2"]["exchange_stride_iterations"]}).
Run through openmm-rest2, which supplies every path.
"""
from rest2_runtime import REST2


def run(files):
    REST2(
        files,
        tau_min={r["rest2"]["tau_min"]},
        tau_max={r["rest2"]["tau_max"]},
        number_of_replicas={r["rest2"]["number_of_replicas"]},
        exchange_interval_ps={r["rest2"]["exchange_interval_ps"]},
        solute_output_interval_ps={r["rest2"]["solute_interval_ps"]},
        whole_output_interval_ps={r["rest2"]["whole_interval_ps"]},
        equilibration_duration_ps={r["rest2"]["equilibration_duration_ps"]},
        temperature_k={r["temperature_K"]},
        pressure_bar={pressure},
        timestep_fs={r["timestep_fs"]},
        friction_per_ps={r["friction_per_ps"]},
        random_seed={seeds["integrator"]},
        hydrogen_mass_amu={r["hydrogen_mass_amu"]!r},
        replica_mixing_scheme={r["rest2"]["replica_mixing_scheme"]!r},
        platform={r["platform"]!r},
    ).run(number_of_exchanges={r["rest2"]["number_of_exchanges"]})
'''


def rest2_launcher(*, parent_restart: str) -> str:
    """The Amber-like REST2 command. Every path explicit, none of them machine-specific.

    Amber would need a groupfile naming one input set per replica. There is none here because
    OpenMMTools keeps one multistate NetCDF for the whole ladder: the replicas share a topology, a
    base System and a starting state, and differ only by tau.
    """
    return f'''#!/usr/bin/env bash
# REST2: one ladder, one NetCDF. Add --resume to continue, --extend N to lengthen it.
#
#   ./rest2.sh                 single process, one device
#   mpiexec -n 6 ./rest2.sh    one rank per replica; each rank binds its own CUDA device
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{STAGE_DIR}}/../paths.sh"

"${{OPENMM_REST2}}" \\
  -i "${{REST2_DIR}}/rest2.py" \\
  -p "${{INPUT_DIR}}/topology.pdb" \\
  -s "${{INPUT_DIR}}/system.xml" \\
  -c "{parent_restart}" \\
  --solute "${{INPUT_DIR}}/solute.yaml" \\
  -o "${{REST2_DIR}}/rest2.out" \\
  -x "${{REST2_DIR}}/rest2.nc" \\
  -r "${{REST2_DIR}}/restart.json" \\
  --checkpoint "${{REST2_DIR}}/rest2_checkpoint.nc" \\
  "$@"
'''
