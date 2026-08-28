"""Emit short, standalone OpenMM stage scripts.

Every number in these scripts is a literal, resolved at generation. That is what keeps them
readable -- `250000`, not `int(config["cMD"]["duration_ns"] * 1e6 / timestep)` -- and it is what
makes them detached: there is nothing left to look up at run time, so `md_templates` can be deleted
and every stage still runs.

The `.out` header is written as one f-string rather than assembled by a helper, so the text in the
script looks like the text in the file it produces. A shared formatting module would have saved a
dozen lines per script and cost the reader an indirection to understand a header.
"""

from __future__ import annotations

from typing import Any

#: `.out` header grammar version. Any change to the field set bumps this.
OUTPUT_VERSION = 1


def _platform(platform: str, device: Any, precision: Any) -> str:
    """Explicit selection: "whatever OpenMM picked" is not a record of what ran."""
    if str(platform).lower() in ("automatic", "auto", ""):
        return 'platform, properties = None, None   # OpenMM picks the fastest available'
    if str(platform).upper() == "CUDA":
        props = {"DeviceIndex": str(device if device is not None else 0),
                 "Precision": precision or "mixed"}
        return (f'platform = Platform.getPlatformByName("CUDA")\n'
                f'properties = {props!r}')
    return (f'platform = Platform.getPlatformByName("{platform}")\n'
            f'properties = None')


def _imports(names: tuple[str, ...], app_names: tuple[str, ...], *, need_sys: bool) -> str:
    line = "import sys\n" if need_sys else ""
    return (f'{line}from pathlib import Path\n\n'
            f'import openmm\n'
            f'from openmm import {", ".join(names)}\n'
            f'from openmm.app import {", ".join(app_names)}')


def minimization_script(r: dict[str, Any]) -> str:
    box = "True" if not r["implicit"] else "False"
    return f'''#!/usr/bin/env python
"""Energy minimisation for {r["system_id"]}.  python min.py > min.out 2>&1"""

{_imports(("LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"),
          ("PDBFile", "Simulation"), need_sys=False)}

HERE = Path(__file__).resolve().parent
COMMON = HERE / ".." / "common"

pdb = PDBFile(str(COMMON / "topology.pdb"))
system = XmlSerializer.deserialize((COMMON / "system.xml").read_text())

# The integrator takes no step here; a Context needs one to exist.
integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                      {r["friction_per_ps"]} / unit.picosecond,
                                      {r["timestep_fs"]} * unit.femtoseconds)
{_platform(r["platform"], r.get("device"), r.get("precision"))}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

print(f"""MD-OPENMM OUTPUT VERSION: {OUTPUT_VERSION}
stage: min
system: {r["system_id"]}
openmm_version: {{openmm.version.version}}
platform: {{simulation.context.getPlatform().getName()}}
integrator: LangevinMiddleIntegrator
temperature_K: {r["temperature_K"]}
timestep_fs: {r["timestep_fs"]}
max_iterations: {r["minimization_max_iterations"]}
input_coordinates: ../common/topology.pdb
final_state: min.state.xml
""", flush=True)

before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
simulation.minimizeEnergy(maxIterations={r["minimization_max_iterations"]})
after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(f"potential_energy_initial: {{before}}")
print(f"potential_energy_final: {{after}}")

state = simulation.context.getState(getPositions=True, getVelocities=True,
                                    enforcePeriodicBox={box})
(HERE / "min.state.xml").write_text(XmlSerializer.serialize(state))
print("run_status: completed")
'''


def equilibration_script(r: dict[str, Any], stage: dict[str, Any], parent: str) -> str:
    name, box = stage["name"], "True" if not r["implicit"] else "False"
    npt = stage["ensemble"] == "NPT"
    restrained = stage["restraint_kcal"] > 0
    first, last = r["solute_range"]

    imports = _imports(
        ("CustomExternalForce", "LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit")
        if restrained else ("LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"),
        ("DCDReporter", "PDBFile", "Simulation", "StateDataReporter"), need_sys=True)
    if npt:
        imports = imports.replace("from openmm import ", "from openmm import MonteCarloBarostat, ")

    restraint = f'''
# Hold the solute -- atoms {first}..{last}, the first residues of topology.pdb -- while the solvent
# relaxes around it. Flat-bottom-free harmonic restraint at {stage["restraint_kcal"]} kcal/mol/A^2.
restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
restraint.addGlobalParameter("k", {stage["restraint_kcal"]} * unit.kilocalories_per_mole / unit.angstrom**2)
for parameter in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(parameter)
for index in range({first}, {last + 1}):
    restraint.addParticle(index, pdb.positions[index].value_in_unit(unit.nanometer))
system.addForce(restraint)
''' if restrained else ""

    barostat = f'''
system.addForce(MonteCarloBarostat({r["pressure_bar"]} * unit.bar,
                                   {r["temperature_K"]} * unit.kelvin,
                                   {r["barostat_interval"]}))
''' if npt else ""

    extra = "volume=True, density=True, " if npt else ""
    return f'''#!/usr/bin/env python
"""{stage["ensemble"]} equilibration ({name}) for {r["system_id"]}.  python {name}.py > {name}.out 2>&1"""

{imports}

HERE = Path(__file__).resolve().parent
COMMON = HERE / ".." / ".." / "common"

pdb = PDBFile(str(COMMON / "topology.pdb"))
system = XmlSerializer.deserialize((COMMON / "system.xml").read_text())
{restraint}{barostat}
integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                      {r["friction_per_ps"]} / unit.picosecond,
                                      {r["timestep_fs"]} * unit.femtoseconds)
{_platform(r["platform"], r.get("device"), r.get("precision"))}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setState(XmlSerializer.deserialize(Path("{parent}").read_text()))
# Positions and velocities carry over; the clock does not, so this stage's .out describes this
# stage. Lineage is stated by `input_state` above, not by a step number climbing across stages.
simulation.context.setTime(0.0)
simulation.currentStep = 0

print(f"""MD-OPENMM OUTPUT VERSION: {OUTPUT_VERSION}
stage: {name}
system: {r["system_id"]}
openmm_version: {{openmm.version.version}}
platform: {{simulation.context.getPlatform().getName()}}
integrator: LangevinMiddleIntegrator
ensemble: {stage["ensemble"]}
temperature_K: {r["temperature_K"]}
friction_per_ps: {r["friction_per_ps"]}
timestep_fs: {r["timestep_fs"]}
restraint_kcal_mol_A2: {stage["restraint_kcal"]}
steps: {stage["steps"]}
duration_ps: {stage["duration_ps"]}
input_state: {parent}
trajectory: {name}.dcd
final_state: {name}.state.xml
""", flush=True)

simulation.reporters.append(DCDReporter("{name}.dcd", {r["output_interval_steps"]}))
simulation.reporters.append(StateDataReporter(
    sys.stdout, {r["output_interval_steps"]}, step=True, time=True, potentialEnergy=True,
    temperature=True, {extra}speed=True))

simulation.step({stage["steps"]})

state = simulation.context.getState(getPositions=True, getVelocities=True,
                                    enforcePeriodicBox={box})
(HERE / "{name}.state.xml").write_text(XmlSerializer.serialize(state))
print("run_status: completed")
'''


def production_script(r: dict[str, Any], parent: str) -> str:
    box = "True" if not r["implicit"] else "False"
    npt = not r["implicit"]
    frames = r["production_steps"] // r["output_interval_steps"]
    imports = _imports(("LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"),
                       ("CheckpointReporter", "DCDReporter", "PDBFile", "Simulation",
                        "StateDataReporter"), need_sys=True)
    if npt:
        imports = imports.replace("from openmm import ", "from openmm import MonteCarloBarostat, ")
    barostat = f'''
system.addForce(MonteCarloBarostat({r["pressure_bar"]} * unit.bar,
                                   {r["temperature_K"]} * unit.kelvin,
                                   {r["barostat_interval"]}))
''' if npt else ""
    extra = "volume=True, density=True, " if npt else ""
    return f'''#!/usr/bin/env python
"""Conventional MD for {r["system_id"]}: {r["production_ps"]:.3f} ps, a frame every {r["output_interval_ps"]:.3f} ps.

DCDReporter writes at step `interval`, not at step 0, so this produces {frames} frames, not {frames + 1}.

    python cmd.py > cmd.out 2>&1
"""

{imports}

HERE = Path(__file__).resolve().parent
COMMON = HERE / ".." / "common"

pdb = PDBFile(str(COMMON / "topology.pdb"))
system = XmlSerializer.deserialize((COMMON / "system.xml").read_text())
{barostat}
integrator = LangevinMiddleIntegrator({r["temperature_K"]} * unit.kelvin,
                                      {r["friction_per_ps"]} / unit.picosecond,
                                      {r["timestep_fs"]} * unit.femtoseconds)
{_platform(r["platform"], r.get("device"), r.get("precision"))}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setState(XmlSerializer.deserialize(Path("{parent}").read_text()))
# Positions and velocities carry over; the clock does not, so this stage's .out describes this
# stage. Lineage is stated by `input_state` above, not by a step number climbing across stages.
simulation.context.setTime(0.0)
simulation.currentStep = 0

print(f"""MD-OPENMM OUTPUT VERSION: {OUTPUT_VERSION}
stage: cMD
system: {r["system_id"]}
openmm_version: {{openmm.version.version}}
platform: {{simulation.context.getPlatform().getName()}}
integrator: LangevinMiddleIntegrator
ensemble: {"NPT" if npt else "NVT"}
temperature_K: {r["temperature_K"]}
friction_per_ps: {r["friction_per_ps"]}
timestep_fs: {r["timestep_fs"]}
steps: {r["production_steps"]}
duration_ps: {r["production_ps"]}
output_interval_steps: {r["output_interval_steps"]}
output_interval_ps: {r["output_interval_ps"]}
frames: {frames}
input_state: {parent}
trajectory: cmd.dcd
final_state: cmd.state.xml
checkpoint: cmd.chk
""", flush=True)

simulation.reporters.append(DCDReporter("cmd.dcd", {r["output_interval_steps"]}))
simulation.reporters.append(CheckpointReporter("cmd.chk", {r["output_interval_steps"]}))
simulation.reporters.append(StateDataReporter(
    sys.stdout, {r["output_interval_steps"]}, step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, totalEnergy=True, temperature=True, {extra}speed=True,
    totalSteps={r["production_steps"]}, remainingTime=True))

simulation.step({r["production_steps"]})

state = simulation.context.getState(getPositions=True, getVelocities=True,
                                    enforcePeriodicBox={box})
(HERE / "cmd.state.xml").write_text(XmlSerializer.serialize(state))
simulation.saveCheckpoint("cmd.chk")
print("run_status: completed")
'''
