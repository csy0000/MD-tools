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
    "eq_npt_1": "implemented: RESTRAINED NPT -- the box relaxes while the solute is held",
    "eq_npt_2": "implemented: FREE NPT -- restraint released, the solute relaxes in the "
                "equilibrated box",
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

    # Every seed this stage uses, derived from the master seed by the shared algorithm and carried
    # in the stage configuration. Nothing here invents a number: an integrator seed compiled into
    # the source made two scientifically different runs share a random stream.
    stage_seeds = payload.get("seeds") or {}
    if not stage_seeds:
        raise StageError(
            f"{stage}.json carries no seeds. Regenerate the project with MD_input_gen.py: seeds are "
            "resolved once, at generation, and recorded -- the stage runner does not invent them."
        )

    barostat_index = None
    if "barostat" in payload:
        from openmm import MonteCarloBarostat
        from .spec.units import parse_quantity
        pressure = parse_quantity(payload["barostat"]["pressure"], dimension="pressure")
        barostat = MonteCarloBarostat(pressure.value * unit.bar,
                                      cfg["integrator"]["temperature_k"] * unit.kelvin, 25)
        barostat.setRandomNumberSeed(int(stage_seeds["barostat"]))
        barostat_index = system.addForce(barostat)

    sim = _make_simulation(pdb.topology, system, cfg, seed=int(stage_seeds["integrator"]))
    # Velocities are initialised ONCE, on entering dynamics from a state that carries none -- in
    # practice NVT, after minimisation. Every later stage inherits them. Re-drawing at each stage
    # discards the equilibration that stage just paid for and hides it behind a plausible-looking
    # temperature.
    coords_origin = _apply_coords(sim, here / payload["input"]["state"],
                                  require_velocities=(stage != "min"),
                                  velocity_seed=int(stage_seeds["velocity"]))

    if restraint_index is not None:
        # kcal/mol/A^2 -> kJ/mol/nm^2
        k = restraint["force_constant_kcal_per_mol_angstrom2"] * 4.184 * 100.0
        sim.context.setParameter("k_restraint", k)

    # The barostat is recorded because whether one was applied is a fact about the stage, while
    # whether the volume actually moved is a sampling outcome: a MonteCarloBarostat can reject every
    # move in a short stage, so an unchanged box is not evidence that the barostat was missing.
    results: dict = {"stage": stage, "n_restrained_atoms": len(restrained_atoms),
                     "barostat": (payload.get("barostat") or {}).get("type"),
                     "seeds": dict(stage_seeds),
                     "velocities": ("not required" if stage == "min"
                                    else "initialized"
                                    if coords_origin.get("velocity_seed") is not None
                                    else "inherited"),
                     "velocity_seed": coords_origin.get("velocity_seed")}
    energy_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
    results["potential_before_kj_mol"] = energy_before.value_in_unit(unit.kilojoule_per_mole)

    if stage == "min":
        sim.minimizeEnergy(maxIterations=int(payload.get("max_iterations", 1000)))
        results["minimizer_max_iterations"] = int(payload.get("max_iterations", 1000))
    else:
        steps = int(payload["steps"])
        if steps > 0:
            sim.step(steps)
        results["steps"] = steps

    # Minimisation writes NO velocities. That absence is the handshake: the first dynamics stage
    # sees a state without them and initialises once, and every stage after that inherits. Writing
    # zeros instead would look like inherited velocities at 0 K and never initialise.
    state = sim.context.getState(getPositions=True, getVelocities=(stage != "min"), getEnergy=True,
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


def _package_bundle_for_rest2(here: Path, payload: dict) -> Path:
    """Assemble a package-format bundle so the runner can consume this project.

    The runner takes a version-2 bundle, not a stage projection. Rather than teach it a second
    input format -- or worse, reimplement its run-directory and restart handling here -- this
    builds the bundle it already understands, using the package's own provenance writers.

    The one substantive choice: ``equilibrated_state.xml`` is the PREDECESSOR STAGE's endpoint
    (cMD_1's final state), not the system bundle's initial state. That is what REST2 must start
    from, and naming it correctly here is what makes the hand-off real rather than decorative.
    """
    import shutil

    from . import bundlev2, provenance

    project = here.parent
    inputs = project / "inputs"
    staged = here / "_bundle"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    for name in ("system.xml", "topology.pdb", "topology.cif"):
        shutil.copy2(inputs / name, staged / name)
    # the simbox record build_simbox produced, under the name the contract expects
    simbox_src = next((p for p in inputs.iterdir() if p.name.endswith("simbox.json")), None)
    if simbox_src is None:
        raise StageError(f"no simbox record in {inputs}; regenerate the system bundle")
    shutil.copy2(simbox_src, staged / "simbox.json")

    # REST2 starts from the previous stage's endpoint, not from the prepared system
    predecessor = (here / payload["input"]["state"]).resolve()
    if not predecessor.is_file():
        raise StageError(
            f"REST2 needs {payload['input']['produced_by']}'s endpoint state, but "
            f"{predecessor} does not exist. Run the preceding stages first."
        )
    shutil.copy2(predecessor, staged / "equilibrated_state.xml")

    runtime = json.loads((project / "run_manifest.json").read_text())
    provenance.write_json(staged / "canonical_configuration.json", {
        "configuration": runtime["resolved_md_config"],
        "hashes": runtime["configuration_hashes"],
        "profile": runtime.get("profile"),
        "sources": runtime["value_sources"],
    })
    from .spec import resolve as spec_resolve
    from .spec.adapter import spec_to_runtime_cfg
    resolved = spec_resolve.resolve_spec(json.loads(json.dumps(runtime["resolved_md_config"])))
    cfg = spec_to_runtime_cfg(resolved["spec"])
    provenance.write_json(staged / "resolved_runtime_config.json", cfg)
    provenance.write_json(staged / "forcefield_provenance.json",
                          bundlev2.forcefield_provenance(cfg))
    provenance.write_json(staged / "environment.json", bundlev2.environment_provenance())
    shutil.copy2(inputs / "system.yaml", staged / "system.yaml")
    # system.yaml names its original input by bare filename; stage it beside the yaml so the
    # manifest loader resolves it here rather than reaching back into the system bundle
    for original in (inputs / "original_inputs").glob("*"):
        if original.is_file():
            shutil.copy2(original, staged / original.name)
    provenance.write_json(staged / "resolved_config.json", cfg)

    # The legacy experiment manifest, written HERE rather than by MD_system_gen.py: it is protocol,
    # and the system generator must not own protocol. It is a projection of the same resolved
    # md_config the stage JSONs came from, so it cannot describe a different run.
    _write_experiment_yaml(staged / "experiment.prepare.yaml", cfg, runtime)

    # Build the manifest with the PACKAGE's own builder rather than by hand. Every field it
    # computes -- composition, checksums, the v2 block, the cross-checks against system.yaml and
    # experiment.prepare.yaml -- is logic that already exists and that validate_bundle will test.
    # Reimplementing it here would be a second, weaker copy of the bundle contract.
    from .bundle import build_bundle_manifest
    from .schemas import config_hash, load_experiment, load_system

    simbox_info = json.loads((staged / "simbox.json").read_text())
    system_manifest = load_system(staged / "system.yaml")
    experiment_manifest = load_experiment(staged / "experiment.prepare.yaml")
    # the hash the bundle will RECOMPUTE from its own manifests -- passing anything else makes
    # validate_bundle report the bundle as edited after preparation
    manifest = build_bundle_manifest(
        staged,
        system_manifest,
        experiment_manifest,
        cfg,
        simbox_info=simbox_info,
        equilibration_info={
            "source": "staged project",
            "predecessor_stage": payload["input"]["produced_by"],
            "note": ("equilibrated_state.xml is the predecessor stage's endpoint, not the prepared "
                     "system's initial state -- REST2 starts from where conventional MD finished."),
        },
        config_hash_value=config_hash(system_manifest.doc, experiment_manifest.doc),
    )
    provenance.write_json(staged / "bundle_manifest.json", manifest)
    return staged


def _execute_rest2(here: Path, payload: dict, devices: str | None) -> int:
    """Hand REST2 to the runner.

    This function decides *nothing* about restarts. It assembles the bundle the runner expects,
    then calls the runner, which owns run-directory naming, the committed-generation record and
    the durable exchange history. The stage layer never reads or writes those.
    """
    from . import runner

    bundle = _package_bundle_for_rest2(here, payload)
    out_root = here / "rest2"
    # Continuing a segment chain is the runner's decision, made from its own committed-generation
    # record. All we do is point it at the run directory it created last time, if there is one.
    # Named explicitly rather than "whichever sorts last": a stage owns exactly one REST2 run, and
    # picking by sort order would silently resume someone else's run if a second one appeared.
    candidate = out_root / "REST2_1"
    resume_run = candidate if (candidate / "status.json").is_file() else None
    if resume_run is not None:
        print(f"[REST2_1] continuing {resume_run.name} (the runner decides the restart point)")

    code, run_dir = runner.launch_rest2(
        bundle,
        bundle / "experiment.prepare.yaml",
        out_root,
        platform=payload.get("execution", {}).get("platform", "CUDA"),
        devices=devices,
        run_name=None if resume_run else "REST2_1",
        resume_run=resume_run,
    )
    if run_dir is not None:
        print(f"[REST2_1] run directory: {run_dir}")
    return code


def _require_master_seed(cfg: dict, runtime: dict) -> int:
    """The master seed the project actually resolved, or a refusal.

    Defaulting here would write a manifest that names a seed nothing in the run ever used, which is
    worse than failing: the manifest is what a reader trusts to reproduce the run.
    """
    seed = (cfg.get("run") or {}).get("seed")
    if seed is None:
        seed = ((runtime.get("randomness") or {}).get("master_seed"))
    if seed is None:
        raise StageError(
            "this project records no master seed, so the REST2 experiment manifest cannot state "
            "one. Regenerate it with MD_input_gen.py."
        )
    return int(seed)


def _write_experiment_yaml(path: Path, cfg: dict, runtime: dict) -> None:
    """The package's legacy experiment manifest, projected from the resolved MD configuration."""
    import yaml

    remd = dict(cfg["production"]["remd"])
    # The legacy experiment schema names the pre-exchange relaxation `relaxation_ps`; the runtime
    # tree calls the same quantity `equilibration_ps`. Map it rather than leaving the field absent.
    if "relaxation_ps" not in remd:
        relaxation = remd.get("equilibration_ps")
        if not relaxation:
            raise StageError(
                "the resolved configuration has no REST2 relaxation time "
                "(production.remd.equilibration_ps), so the experiment manifest cannot be written. "
                "Regenerate the project from a configuration that states it."
            )
        remd["relaxation_ps"] = relaxation
    doc = {
        "schema_version": 2,
        "experiment_id": "staged_rest2",
        # The master seed of the run, never a literal: a default compiled in here would make an
        # experiment manifest claim a seed the project never resolved.
        "master_seed": int(_require_master_seed(cfg, runtime)),
        "ladder_status": "unvalidated",
        "platform": {
            "name": cfg["production"]["platform"],
            "precision": cfg["production"]["precision"],
        },
        "integrator": dict(cfg["integrator"]),
        "equilibration": dict(cfg["equilibration"]),
        "md": dict(cfg["production"]["md"]),
        "rest2": remd,
        "overrides": {},
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":                                   # pragma: no cover
    try:
        raise SystemExit(main())
    except StageError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
