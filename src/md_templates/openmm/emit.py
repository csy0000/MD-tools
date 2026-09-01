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
    # Explicit solvent is NPT unless the request says otherwise. An rREST2 reservoir source must
    # be NVT: the ladder exchanges complete configurations at fixed volume, and a source carrying
    # a distribution of volumes would change the density of the rung it refreshes without
    # accounting for the pV work.
    npt = (not r["implicit"]) and str(r.get("cmd_ensemble", "NPT")).upper() == "NPT"
    frames = r["production_steps"] // r["output_interval_steps"]
    seeds = r["seeds"]["cMD"]

    tau = float(r.get("cmd_tau", 0.0) or 0.0)
    names = ["LangevinMiddleIntegrator", "Platform", "XmlSerializer", "unit"]
    if npt:
        names.insert(0, "MonteCarloBarostat")
    # A fixed-tau walker scales the solute Hamiltonian through the SAME module the ladder uses, so
    # a cMD run at tau and the REST2 rung at tau are the same Hamiltonian by construction. At
    # tau = 0 `build_scaled_system` returns the System untouched.
    phase_ps = r.get("cmd_phase_space_interval_ps")
    phase_steps = (None if not phase_ps else
                   int(round(float(phase_ps) / (r["timestep_fs"] / 1000.0))))
    scaling = ("from runtime_record import write_resolved_run\n" if tau == 0.0 else
               "from rest2_scaling import REST2_IMPLEMENTATION, build_scaled_system\n"
               "from runtime_record import write_resolved_run\n"
               "import yaml\n")
    if phase_steps:
        scaling += ("from hamiltonian_identity import identity_record\n"
                    "from phase_space import PhaseSpaceReporter\n")
    phase_block = ("" if not phase_steps else f'''
    # The PHASE-SPACE stream: positions, velocities and box together, which a DCD cannot carry.
    # This is what makes a fixed-tau run usable as an rREST2 reservoir source, and the identity
    # recorded here is what the reservoir validator recomputes and compares against.
    _identity = {{
        "hamiltonian": identity_record(
            system, tau={tau}, temperature_k={r["temperature_K"]},
            ensemble="{"NPT" if npt else "NVT"}",
            solute_indices=list(range({r["solute_range"][1] + 1})),
            excluded_bonds=[tuple(int(a) for a in pair) for pair in
                            (yaml.safe_load(Path(files.topology).parent.joinpath(
                                "solute.yaml").read_text()).get("rest2") or {{}}
                             ).get("omega_excluded_bonds", [])]),
        "method": "cMD", "tau": {tau}, "temperature_kelvin": {r["temperature_K"]},
        "ensemble": "{"NPT" if npt else "NVT"}",
    }}
    simulation.reporters.append(PhaseSpaceReporter(
        Path(files.trajectory).with_suffix(".phase_space.nc"), {phase_steps},
        identity=_identity, periodic={"True" if npt or not r["implicit"] else "False"},
        timestep_fs={r["timestep_fs"]}))
''')
    scale_block = ("" if tau == 0.0 else f'''
    solute = yaml.safe_load(Path(files.topology).parent.joinpath("solute.yaml").read_text())
    solute_indices = list(range(int(solute["n_solute_atoms"])))
    excluded = [tuple(int(a) for a in pair)
                for pair in (solute.get("rest2") or {{}}).get("omega_excluded_bonds", [])]
    system = build_scaled_system(system, solute_indices, {tau}, excluded_bonds=excluded)
    print(f"# fixed tau = {tau:g}, "
          f"{{len(excluded)}} omega bond(s) left unscaled")
''')
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
{scaling}

def run(files):
    pdb = PDBFile(files.topology)
    system = XmlSerializer.deserialize(Path(files.system).read_text())
{scale_block}{barostat}
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

{phase_block}
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

    # The companion runtime record: tau, temperature, ensemble and the frame-time map, recorded
    # where they were decided. This is what makes the trajectory usable as an AIS source or an
    # rREST2 reservoir, and why neither ever has to read a directory name.
    write_resolved_run(
        Path(files.output).parent, method="cMD", tau={r.get("cmd_tau", 0.0) or 0.0},
        temperature_kelvin={r["temperature_K"]}, ensemble="{"NPT" if npt else "NVT"}",
        timestep_fs={r["timestep_fs"]}, friction_per_ps={r["friction_per_ps"]},
        steps={r["production_steps"]}, duration_ps={r["production_ps"]},
        trajectory_name=Path(files.trajectory).name if files.trajectory else None,
        frames={frames}, interval_ps={r["output_interval_ps"]},
        rest2_implementation={"dict(REST2_IMPLEMENTATION)" if tau else "None"})
    print("run_status: completed")
'''


# --- the portable path map and the launchers ----------------------------------------------------

def cmd_directory_variable(name: str) -> str:
    """`cMD` -> CMD_DIR, `cMD_tau0p5` -> CMD_TAU0P5_DIR. A name, mechanically."""
    return name.upper().replace(".", "P").replace("-", "_") + "_DIR"


def paths_sh(relative_project: str, system_id: str, stage_names: list[str],
             production_dir: str | None = None) -> str:
    """The directory map, resolved from `$MD_DATA` and a relative project path.

    No absolute path appears here. Move the managed root, export the new `$MD_DATA`, and every
    launcher follows -- which is the difference between a system that can be archived and one that
    only works on the machine that made it.
    """
    # Only the equilibration stages have a directory variable here; the production method
    # (cMD or REST2) already has one below, and mapping it twice would be a KeyError.
    eq = [n for n in stage_names if n not in ("cMD", "REST2", "rREST2", "AIS")
          and not n.startswith("cMD_tau")]
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
        '# A fixed-tau cMD walker lives in its own directory so it cannot be confused with the',
        '# unscaled one. The NAME is a convenience for a reader: nothing infers tau from it, and',
        '# rREST2 reads the run own resolved_run.yaml instead.',
        'export CMD_TAU0P5_DIR="${SYSTEM_ROOT}/cMD_tau0p5"',
        'export REST2_DIR="${SYSTEM_ROOT}/REST2"',
        'export RREST2_DIR="${SYSTEM_ROOT}/rREST2"',
        'export AIS_DIR="${SYSTEM_ROOT}/AIS"',
        '',
    ]
    # A fixed-tau walker at a tau other than 0.5 needs its own variable; the two common ones are
    # already above. Emitted from the directory the system ACTUALLY has, so the launcher and the
    # map cannot disagree.
    if production_dir:
        variable = cmd_directory_variable(production_dir)
        if f'export {variable}=' not in "\n".join(lines):
            lines += [f'export {variable}="${{SYSTEM_ROOT}}/{production_dir}"', '']
    lines += [
        '# The generated copy is authoritative for this system. Override deliberately if you have',
        '# a compatible executable elsewhere; ordinary use needs no installed command.',
        'export OPENMM_MD="${OPENMM_MD:-${SYSTEM_ROOT}/bin/openmm-md}"',
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


# --- replica exchange: REST2 and rREST2 -----------------------------------------------------------

def replica_protocol_file(r: dict[str, Any], *, method: str) -> str:
    """The concise grouped protocol. It names a ladder and holds no path.

    `openmm-md` reads `protocol` out of this file. The exchange loop, the storage, the MPI
    plumbing and the reservoir handling are all elsewhere, which is what keeps this the size of
    the science.
    """
    block = r["replica"]
    ladder = ", ".join(f"{value:.4f}" for value in block["tau"])
    reservoir_note = ""
    if method == "rREST2":
        reservoir_note = (
            "\nThe hottest rung is periodically refreshed from a Boltzmann reservoir prepared "
            "from a\nfixed-tau cMD run at the same tau, temperature and fixed volume. The "
            "reservoir and its\nschedule are declared in reservoir.yaml; the refresh rule is "
            "rrest2_exchange.py.\n")
    return f'''#!/usr/bin/env python
"""{method} for {r["system_id"]}: {block["n_states"]} states, tau {block["tau"][0]} to {block["tau"][-1]}, NVT at {r["temperature_K"]} K.

Omega-selective REST2: torsions about a peptide omega bond are left UNSCALED. Every state is
thermostatted at the SAME {r["temperature_K"]} K and differs only by Hamiltonian -- this is
Hamiltonian scaling, not temperature REMD, and an exchange never rescales velocities.

    tau ladder : [{ladder}]
    (1-tau)^2 on solute-solute terms, (1-tau) on solute-environment terms

Every interval below is a PHYSICAL time and is independent of the others. Each converts to an
exact whole number of integration steps; the propagation span between events is derived and is not
a scientific input. There is no `segment_ps`.
{reservoir_note}
Run through openmm-md with the group file beside this one, which supplies every path.
"""
from replica_runtime import REST2Protocol

protocol = REST2Protocol(
    tau=[{ladder}],
    temperature_k={r["temperature_K"]},
    timestep_fs={r["timestep_fs"]},
    exchange_interval_ps={block["exchange_interval_ps"]},
    whole_output_interval_ps={block["whole_output_interval_ps"]},
    solute_output_interval_ps={block["solute_output_interval_ps"]},
    number_of_exchanges={block["number_of_exchanges"]},
    friction_per_ps={r["friction_per_ps"]},
    equilibration_ps={block["equilibration_ps"]},
    random_seed={r["seeds"][method]["integrator"]},
    hydrogen_mass_amu={r["hydrogen_mass_amu"]!r},
    # Where this runs. Stated here so the request actually reaches the runtime; it is not part of
    # the scientific identity, so a continuation on another machine is not refused for it.
    platform={r["platform"]!r},
    precision={r.get("precision")!r},
)
'''


def replica_group_file(r: dict[str, Any], *, method: str, protocol_name: str,
                       parent_state: str) -> str:
    """The Amber-like group file: one line per state, inputs only.

    Every line names the SAME protocol, topology, System and starting state, because a REST2
    ladder is one system at several Hamiltonians -- the tau of a group is its index in the
    protocol's ladder, not a separate file. `--group-index` is stated rather than taken from line
    order, so a reordered file still means the same thing.
    """
    lines = [f"# {method} for {r['system_id']}: {r['replica']['n_states']} states, "
             f"tau {r['replica']['tau'][0]} to {r['replica']['tau'][-1]}.",
             "# One group per line, inputs only. Run-level -o/-x/-r/--checkpoint go on the",
             "# openmm-md command, because they describe the coordinated run.",
             "#",
             "# Parsed with shlex, never evaluated by a shell."]
    for index in range(r["replica"]["n_states"]):
        lines.append(
            f"-i {method}/{protocol_name} -p input/topology.pdb -s input/system.xml "
            f"-c {parent_state} --solute input/solute.yaml --group-index {index}")
    return "\n".join(lines) + "\n"


def replica_launcher(r: dict[str, Any], *, method: str, directory_var: str, protocol_name: str,
                     stem: str, exchange_rule: str | None = None,
                     reservoir: str | None = None) -> str:
    """The launcher. One executable, every path explicit, none of them machine-specific."""
    extra = []
    if exchange_rule:
        extra.append(f'  --exchange-rule "${{{directory_var}}}/{exchange_rule}" \\')
    if reservoir:
        extra.append(f'  --reservoir "${{{directory_var}}}/{reservoir}" \\')
    extra_block = ("\n" + "\n".join(extra)) if extra else ""
    states = r["replica"]["n_states"]
    return f'''#!/usr/bin/env bash
# {method}: one ladder, one coordinated run, one executor.
#
#   ./{stem}.sh                        single process, one device
#   mpiexec -n {states} ./{stem}.sh    one rank per state; each rank binds its own CUDA device
#   ./{stem}.sh --resume               finish an interrupted run; no restart.json needed
#   ./{stem}.sh --extend 200           add 200 exchange attempts IN PLACE to a completed run
#   ./{stem}.sh --verify-only          open the stored output and check it, running nothing
#
# To add a segment WITHOUT touching this run, use `{stem}_extend.sh` beside this file: it writes a
# new directory and leaves this one byte-for-byte unchanged.
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{STAGE_DIR}}/../paths.sh"
cd "${{SYSTEM_ROOT}}"

"${{OPENMM_MD}}" \\
  -ng {states} \\
  --groupfile "${{{directory_var}}}/{stem}.group" \\{extra_block}
  -o "${{{directory_var}}}/{stem}.out" \\
  -x "${{{directory_var}}}/{stem}.nc" \\
  -r "${{{directory_var}}}/restart.json" \\
  --checkpoint "${{{directory_var}}}/{stem}_checkpoint.nc" \\
  --rem "${{{directory_var}}}/rem.log" \\
  "$@"
'''


def replica_extension_launcher(r: dict[str, Any], *, method: str, directory_var: str,
                               stem: str, exchange_rule: str | None = None,
                               reservoir: str | None = None) -> str:
    """Add a segment to a COMPLETED ladder without touching it.

    The parent stays exactly as it is. This writes a sibling directory holding only the new
    dynamics, with step and time coordinates continuing the parent's, and provenance pinning the
    parent by content.
    """
    extra = []
    if exchange_rule:
        extra.append(f'  --exchange-rule "${{{directory_var}}}/{exchange_rule}" \\')
    if reservoir:
        extra.append(f'  --reservoir "${{{directory_var}}}/{reservoir}" \\')
    extra_block = ("\n" + "\n".join(extra)) if extra else ""
    states = r["replica"]["n_states"]
    return f'''#!/usr/bin/env bash
# {method}: continue a COMPLETED ladder into a NEW directory, leaving the finished one untouched.
#
#   ./{stem}_extend.sh {method}_ext1 200
#       read {method}/ (never write to it), add 200 exchange attempts, write {method}_ext1/
#
#   mpiexec -n {states} ./{stem}_extend.sh {method}_ext1 200
#       the same, one rank per state
#
# This is NOT `--extend`, which lengthens a run in place. Here the parent is opened read-only and
# is left byte-for-byte unchanged, which is what makes it citable as a finished 
# segment while the chain grows past it.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $(basename "$0") <extension-directory> <exchanges> [openmm-md options...]" >&2
  echo "  e.g. $(basename "$0") {method}_ext1 200" >&2
  exit 2
fi
TARGET="$1"; EXCHANGES="$2"; shift 2

STAGE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{STAGE_DIR}}/../paths.sh"
cd "${{SYSTEM_ROOT}}"

PARENT="${{{directory_var}}}"
EXTENSION="${{SYSTEM_ROOT}}/${{TARGET}}"
if [ "${{EXTENSION}}" = "${{PARENT}}" ]; then
  echo "the extension directory must differ from the parent ${{PARENT}}" >&2
  exit 2
fi
mkdir -p "${{EXTENSION}}"

# The ladder inputs are the parent's: same protocol, same group file, same topology and system.
# Only the OUTPUT paths move.
"${{OPENMM_MD}}" \\
  -ng {states} \\
  --groupfile "${{PARENT}}/{stem}.group" \\{extra_block}
  --extend-from "${{PARENT}}" \\
  --extend "${{EXCHANGES}}" \\
  -o "${{EXTENSION}}/{stem}.out" \\
  -x "${{EXTENSION}}/{stem}.nc" \\
  -r "${{EXTENSION}}/restart.json" \\
  --checkpoint "${{EXTENSION}}/{stem}_checkpoint.nc" \\
  --rem "${{EXTENSION}}/rem.log" \\
  "$@"
'''


def reservoir_declaration(r: dict[str, Any]) -> str:
    """What the rREST2 rule refreshes from, and every assumption it rests on.

    The SOURCE PATH IS NOT EVIDENCE. The Hamiltonian fingerprint, tau and temperature come from the
    source run's own recorded identity; a directory called `cMD_tau0p5` can be renamed or copied,
    and a reservoir drawn from the wrong Hamiltonian runs to completion while being wrong.
    """
    block = r["replica"]
    reservoir = block["reservoir"]
    stored = reservoir["velocity_policy"] == "stored"
    return f'''# The Boltzmann PHASE-SPACE reservoir the hottest rung is refreshed from.
#
# A phase-space sample is positions AND velocities AND box. A DCD cannot store velocities, so a
# DCD is not a phase-space reservoir; this source is the NetCDF stream a fixed-tau cMD run writes.
#
# v1 accepts a refresh with probability ONE, and that is correct only because the reservoir is
# Boltzmann-weighted at exactly the top rung's HAMILTONIAN, tau, temperature and fixed-volume
# ensemble. The Hamiltonian is compared on the serialized OpenMM System, recomputed on both sides:
# tau and temperature agreeing is necessary and not sufficient, because ff14SB/TIP3P and
# ff19SB/OPC agree on both.
#
# The finite-reservoir approximation is real: {reservoir["frames"]} samples are not the top state's
# full equilibrium distribution, and the assumption that the selected window represents it is an
# assumption, not a result.
#
# Roitberg, Okur, Simmerling, J. Phys. Chem. B 2007, 111, 2415; doi:10.1021/jp068335b
# Kasavajhala, Lam, Simmerling, J. Chem. Inf. Model. 2020, 60, 1218; PMCID PMC7725893
format: {reservoir["format"]}
weighting: boltzmann
ensemble: NVT
prepared_directory: reservoir
# `stored` installs the RECORDED momentum unchanged, which is what the probability-one rule
# assumes. `maxwell` redraws at the common temperature from a recorded seed and must be asked for
# explicitly -- there is no silent fallback when velocities are missing; that is a hard error.
velocity_policy: {reservoir["velocity_policy"]}{"" if stored else "   # REDRAWN, not the recorded momentum"}
# In EXCHANGE attempts, not steps.
refresh_interval_exchanges: {reservoir["refresh_interval_exchanges"]}
random_seed: {reservoir["random_seed"]}
source:
  # Relative to the system root. The path locates the run; it never establishes what it is.
  phase_space: {reservoir["phase_space"]}
  start_time_ps: {reservoir["start_time_ps"]}
  end_time_ps: {reservoir["end_time_ps"]}
  frames: {reservoir["frames"]}
  # Refused if set to true, before anything is selected or written. A repeated draw would give
  # one configuration extra statistical weight, make `frames` overstate the effective reservoir
  # size, and produce a file this repository's own strictly-increasing-step check rejects. Ask
  # for at most as many frames as the window holds, or widen the window.
  allow_sampling_with_replacement: false
'''
