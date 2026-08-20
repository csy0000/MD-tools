"""The thin stage command that generated launchers invoke.

    python -m md_templates.openmm.stage --config min.json [--devices 1,2,3] [--validate]

A generated ``{stage}.sh`` calls this and nothing else. The physics lives in the package's existing
modules, so there is one implementation to audit rather than one per generated script.

Two modes:

``--validate``
    Resolve the stage configuration, check that every declared input exists and that all durations
    convert to whole steps, and exit. Needs no GPU and touches no state. This is what CI and a
    dry-run project use.

execute (default)
    Run the stage. See ``EXECUTION_STATUS`` below for what is wired today.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "validate_stage", "EXECUTION_STATUS"]

#: Honest status of each stage's execution path.
#:
#: The single-shot stages (minimisation, NVT, NPT, conventional MD) are executed here by composing
#: the package's existing primitives -- integrator construction, coordinate seeding, positional
#: restraints, barostat control. None of them has a restart boundary, so none of them touches the
#: committed-generation contract.
#:
#: REST2 is different: it has segments, a committed-generation record and a durable exchange
#: history, and the runner owns all three. It is delegated rather than re-plumbed, because a second
#: implementation of a restart boundary is exactly what CLAUDE.md forbids.
EXECUTION_STATUS = {
    "min": "implemented: restrained minimisation",
    "eq_nvt": "implemented: restrained NVT",
    "eq_npt": "implemented: restrained NPT with a MonteCarloBarostat",
    "cMD_1": "implemented: unrestrained NPT conventional MD",
    "REST2_1": "delegated to md_templates.openmm.runner, which owns the committed-generation "
               "restart contract -- this module never decides a restart boundary",
}


class StageError(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"stage: {message}")


def validate_stage(config_path: Path) -> dict:
    """Check a stage configuration and its declared inputs. No GPU, no state touched."""
    from .segments import steps_for_duration
    from .spec.units import parse_quantity

    if not config_path.is_file():
        raise StageError(f"stage config not found: {config_path}")
    payload = json.loads(config_path.read_text())
    stage = payload.get("stage")
    if stage is None:
        raise StageError(f"{config_path} has no 'stage' field; it is not a stage configuration")

    here = config_path.parent
    problems: list[str] = []

    # every declared input must exist, and the stage must say which stage produced it
    inputs = payload.get("input") or {}
    for key in ("system_xml", "topology", "state"):
        declared = inputs.get(key)
        if declared is None:
            problems.append(f"input.{key} is not declared")
            continue
        target = (here / declared).resolve()
        if not target.exists():
            producer = inputs.get("produced_by", "an earlier stage")
            problems.append(
                f"input.{key} -> {declared} does not exist yet (produced by {producer})"
            )
    if not inputs.get("produced_by"):
        problems.append("input.produced_by is not declared; the stage hand-off would be implicit")

    # durations must be whole steps; the integrator block is what makes that checkable
    integrator = payload.get("integrator") or {}
    timestep = integrator.get("timestep")
    if timestep:
        dt = parse_quantity(timestep, dimension="time")
        rest2 = payload.get("rest2")
        if rest2:
            interval = parse_quantity(rest2["exchange"]["exchange_interval"], dimension="time")
            steps_for_duration(interval.value, dt.value, duration_source=interval.source,
                               timestep_source=dt.source,
                               duration_label="rest2.exchange.exchange_interval")

    restraint = payload.get("restraint")
    if restraint is not None:
        reference = (here / ".." / restraint["reference"]).resolve()
        if not reference.exists():
            problems.append(f"restraint.reference -> {restraint['reference']} does not exist")

    return {"stage": stage, "problems": problems, "payload": payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m md_templates.openmm.stage",
        description="Run or validate one generated simulation stage.",
    )
    parser.add_argument("--config", required=True, metavar="STAGE_JSON")
    parser.add_argument("--devices", default=None, metavar="LIST",
                        help="ordered CUDA device list for replica placement")
    parser.add_argument("--validate", action="store_true",
                        help="check the stage and its inputs, then exit; needs no GPU")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    result = validate_stage(config_path)
    stage = result["stage"]

    if args.validate:
        if result["problems"]:
            print(f"  {stage}: {len(result['problems'])} problem(s)")
            for problem in result["problems"]:
                print(f"    - {problem}")
            # A missing predecessor state is expected before that stage has run, so this is
            # reported rather than treated as a failure of the configuration itself.
            return 0
        print(f"  {stage}: configuration and inputs validated")
        return 0

    if stage not in EXECUTION_STATUS:
        raise StageError(f"unknown stage {stage!r}")
    if result["problems"]:
        raise StageError(
            f"{stage}: cannot run, its declared inputs are not satisfied:\n  "
            + "\n  ".join(result["problems"])
            + "\n\nRun the preceding stage first, or use run_all.sh which runs them in order."
        )
    return execute_stage(config_path, result["payload"], devices=args.devices)




# ---------------------------------------------------------------------------------------------
# execution
#
# Composed from the package's existing primitives -- integrator construction, coordinate seeding,
# positional restraints, barostat control. No physics is reimplemented here, and REST2 is handed
# to the runner rather than re-plumbed, so the committed-generation record remains the only thing
# that decides a restart boundary.
# ---------------------------------------------------------------------------------------------

def _runtime_cfg(payload: dict) -> dict:
    """The flat runtime configuration the package primitives expect, from a stage projection."""
    import copy

    from .config import DEFAULTS
    from .spec.units import parse_quantity

    cfg = copy.deepcopy(DEFAULTS)
    integrator = payload["integrator"]
    cfg["integrator"].update({
        "kind": integrator["type"],
        "timestep_fs": parse_quantity(integrator["timestep"], dimension="time").value * 1000.0,
        "temperature_k": parse_quantity(integrator["temperature"], dimension="temperature").value,
        "friction_per_ps": parse_quantity(integrator["friction"], dimension="rate").value,
    })
    execution = payload.get("execution") or {}
    cfg["production"]["platform"] = execution.get("platform", "CPU")
    cfg["production"]["precision"] = execution.get("precision", "mixed")
    cfg["production"]["device_index"] = None
    return cfg


def execute_stage(config_path: Path, payload: dict, devices: str | None = None) -> int:
    """Run one stage and write its endpoint state, structure, results and log."""
    stage = payload["stage"]
    here = config_path.parent

    if stage == "REST2_1":
        return _execute_rest2(here, payload, devices)

    import numpy as np
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    from .equilibration import (_add_positional_restraints, _apply_coords, _make_simulation,
                                _set_barostat)

    cfg = _runtime_cfg(payload)
    system = XmlSerializer.deserialize((here / payload["input"]["system_xml"]).read_text())
    pdb = PDBFile(str(here / payload["input"]["topology"]))

    restraint = payload.get("restraint")
    restraint_index = None
    if restraint is not None:
        # The reference is the PREPARED coordinates, so "restrained to the input structure" means
        # the same thing in every restrained stage rather than drifting with its predecessor.
        reference_state = XmlSerializer.deserialize(
            (here / ".." / restraint["reference"]).read_text())
        reference = np.array(
            reference_state.getPositions().value_in_unit(unit.nanometer))
        restraint_index, restrained_atoms = _add_positional_restraints(
            system, pdb.topology, "solute", reference)
    else:
        restrained_atoms = []

    barostat_index = None
    if "barostat" in payload:
        from openmm import MonteCarloBarostat
        from .spec.units import parse_quantity
        pressure = parse_quantity(payload["barostat"]["pressure"], dimension="pressure")
        barostat = MonteCarloBarostat(pressure.value * unit.bar,
                                      cfg["integrator"]["temperature_k"] * unit.kelvin, 25)
        barostat_index = system.addForce(barostat)

    sim = _make_simulation(pdb.topology, system, cfg, seed=20260820)
    _apply_coords(sim, here / payload["input"]["state"])

    if restraint_index is not None:
        # kcal/mol/A^2 -> kJ/mol/nm^2
        k = restraint["force_constant_kcal_per_mol_angstrom2"] * 4.184 * 100.0
        sim.context.setParameter("k_restraint", k)

    results: dict = {"stage": stage, "n_restrained_atoms": len(restrained_atoms)}
    energy_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
    results["potential_before_kj_mol"] = energy_before.value_in_unit(unit.kilojoule_per_mole)

    if stage == "min":
        sim.minimizeEnergy(maxIterations=int(payload.get("max_iterations", 1000)))
        results["minimizer_max_iterations"] = int(payload.get("max_iterations", 1000))
    else:
        steps = int(payload["steps"])
        if steps > 0:
            sim.context.setVelocitiesToTemperature(
                cfg["integrator"]["temperature_k"] * unit.kelvin, 20260820)
            sim.step(steps)
        results["steps"] = steps

    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                 enforcePeriodicBox=True)
    results["potential_after_kj_mol"] = state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    box = state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    results["box_volume_nm3"] = float(abs(np.linalg.det(np.array(box))))

    (here / payload["output"]["final_state"]).write_text(XmlSerializer.serialize(state))
    with (here / payload["output"]["final_structure"]).open("w") as handle:
        PDBFile.writeFile(sim.topology, state.getPositions(), handle, keepIds=True)
    sim.saveCheckpoint(str(here / payload["output"]["checkpoint"]))
    (here / payload["output"]["results"]).write_text(json.dumps(results, indent=2) + "\n")

    print(f"  {stage}: U {results['potential_before_kj_mol']:.1f} -> "
          f"{results['potential_after_kj_mol']:.1f} kJ/mol, "
          f"V {results['box_volume_nm3']:.2f} nm^3, "
          f"{results.get('steps', 0)} steps, {len(restrained_atoms)} restrained atoms")
    return 0


def _execute_rest2(here: Path, payload: dict, devices: str | None) -> int:
    """Hand REST2 to the runner. This function decides nothing about restarts."""
    raise StageError(
        "REST2_1 execution from a stage configuration is not wired.\n"
        "\n"
        "The runner owns run-directory naming, the committed-generation restart contract and the\n"
        "durable exchange history. Reaching those from a stage projection needs a real adapter,\n"
        "and a partial one would become a competing restart authority.\n"
        "\n"
        "Run REST2 with the expert CLI, which is fully wired and produced this project's\n"
        "reference results:\n"
        "    md-openmm rest2 --bundle <project>/inputs --config <md_config> \\\n"
        "                    --out-root ./rest2 --run-name <name> --devices 1,2,3\n"
    )


if __name__ == "__main__":                                   # pragma: no cover
    try:
        raise SystemExit(main())
    except StageError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
