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
from .hashing import sha256_bytes, sha256_text

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
    "eq": ("implemented: restrained equilibration for implicit solvent -- constant temperature, "
           "no barostat, and no ensemble label, because 'NVT' fixes a volume this System does not "
           "have"),
    "eq_free": ("implemented: UNRESTRAINED equilibration for implicit solvent -- restraint "
                "released, constant temperature, no barostat and no ensemble label. Restrained "
                "dynamics relaxes the solvent response around a held solute; the solute's own "
                "conformational relaxation only begins here"),
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
            # The interval is derived at generation, so what is checked here is that the recorded
            # derivation is still a whole number of steps against this stage's timestep.
            exchange = rest2["exchange"]
            steps_for_duration(
                float(exchange["exchange_interval_derived_ps"]), dt.value,
                duration_source=(f"derived from {exchange['duration_per_segment']} / "
                                 f"{exchange['number_of_exchanges_per_segment']} exchanges"),
                timestep_source=dt.source,
                duration_label="rest2.exchange.exchange_interval_derived_ps")

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
    # No default. A defaulted platform decides where the science runs, silently; a payload that
    # omits it is incomplete and must say so rather than land on the CPU.
    cfg["production"]["platform"] = execution.get("platform")
    cfg["production"]["precision"] = execution.get("precision", "mixed")
    # Filled in by execute_stage from --devices. Left None here so `_runtime_cfg` stays a pure
    # projection of the payload; a stage that silently chose its own GPU would place work on a
    # device the operator did not select.
    cfg["production"]["device_index"] = None
    return cfg


#: Bumped when the continuity contract gains or changes a field. A run committed under an older
#: contract cannot be compared field-by-field against this one, so it is refused with guidance
#: rather than silently reinterpreted.
CMD_CONTINUITY_VERSION = 3


def _atom_identity(topology, indices) -> str:
    """A hash over the ORDERED identity of the selected atoms, not merely their count.

    Finding 2. The contract carried particle and constraint counts, which do not identify a
    System, and the selection fingerprint was passed as None. Two different topologies with equal
    counts, or two different orderings of the same number of atoms, compared equal -- so a
    trajectory could be continued with a different molecule or a permuted atom order, and every
    frame after the boundary would silently mean something else.

    Index, chain, residue number, residue name, atom name and element are all included, so
    equal-length selections cannot alias.
    """
    import hashlib

    atoms = list(topology.atoms())
    parts = []
    for position, index in enumerate(indices):
        atom = atoms[int(index)]
        residue = atom.residue
        parts.append("|".join((
            str(position), str(int(index)), str(residue.chain.id), str(residue.id),
            str(residue.name), str(atom.name),
            atom.element.symbol if atom.element is not None else "none")))
    return sha256_text("\n".join(parts))


def _file_sha256(path: Path) -> str:
    from .hashing import sha256_file

    return sha256_file(path)


def _cmd_continuity(payload: dict, system, *, here: Path, topology=None,
                    selected_atoms=None) -> dict:
    """The fields that must not change between segments of one conventional-MD run.

    Everything here changes the calculation rather than merely how long it runs. Two segments that
    disagree about any of them are two different simulations sharing a trajectory file, which is a
    worse outcome than a refusal because the file looks continuous.

    Segment COUNT is deliberately absent: asking for more segments is the one change that is always
    safe, which is why it lives in Bash and never in the hash.
    """
    integrator = payload.get("integrator") or {}
    reporting = payload.get("reporting") or {}
    barostat = payload.get("barostat") or {}
    restraint = payload.get("restraint") or {}

    system_xml = here / payload["input"]["system_xml"]
    topology_file = here / payload["input"]["topology"]
    bundle = here / ".." / "inputs"

    # The force-field FILE, not only a projection of it. The projection is kept because it is what a
    # human reads in a refusal message, but a projection cannot notice a change to a field it does
    # not name -- and every field it does not name is still part of the Hamiltonian.
    forcefield = {}
    forcefield_sha = None
    forcefield_path = bundle / "forcefield.json"
    if forcefield_path.is_file():
        record = json.loads(forcefield_path.read_text())
        forcefield = {k: record.get(k) for k in
                      ("protein_forcefield", "water", "ligand", "ligand_charge_method",
                       "implicit_model", "radii", "route", "input_route")}
        forcefield_sha = _file_sha256(forcefield_path)

    # Likewise the manifest: its exact bytes, plus the identity fields worth naming in a message.
    manifest_identity = None
    manifest_path = bundle / "system_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        system_block = manifest.get("system") or {}
        manifest_identity = {
            "system_id": system_block.get("id"),
            "input_sha256": system_block.get("input_sha256"),
            "prepared_through": manifest.get("prepared_through"),
            "manifest_sha256": _file_sha256(manifest_path),
            # whichever identity/configuration hash the bundle records for itself
            "configuration_hash": (
                manifest.get("configuration_hash")
                or manifest.get("config_hash")
                or (manifest.get("hashes") or {}).get("system_build_sha256")
                or (manifest.get("identity") or {}).get("hash")),
            "water_policy": {
                k: (manifest.get("water_policy") or {}).get(k)
                for k in ("water", "water_model", "basis", "mixed")
            } if manifest.get("water_policy") else None,
        }

    # The predecessor State is how this cMD chain began: immutable provenance, bound by its exact
    # bytes rather than by a pathname that can be repointed at a different equilibration. `absent`
    # rather than None so that DELETING it is a visible change and gets refused, instead of quietly
    # comparing equal to a run that never had one.
    predecessor_path = here / payload["input"]["state"]
    predecessor = {
        "path": payload["input"]["state"],
        "produced_by": payload["input"]["produced_by"],
        "sha256": (_file_sha256(predecessor_path) if predecessor_path.is_file() else "absent"),
    }

    return {
        "contract_version": CMD_CONTINUITY_VERSION,
        "stage": payload["stage"],
        # identity of the exact artifacts, not counts that many systems share
        "system_xml_sha256": _file_sha256(system_xml),
        "topology_sha256": _file_sha256(topology_file),
        "bundle": manifest_identity,
        "forcefield": forcefield,
        "forcefield_sha256": forcefield_sha,
        "n_particles": system.getNumParticles(),
        "n_constraints": system.getNumConstraints(),
        "periodic": bool(system.usesPeriodicBoundaryConditions()),
        "ensemble": "NPT" if barostat else "NVT",
        "barostat_pressure": barostat.get("pressure"),
        "integrator": {k: integrator.get(k) for k in
                       ("type", "timestep", "temperature", "friction")},
        "restraint_active": bool(restraint) and restraint.get("selection") not in (None, "none"),
        "restraint_selection": restraint.get("selection"),
        "restraint_force_constant": restraint.get("force_constant_kcal_per_mol_angstrom2"),
        "restraint_convention": payload.get("_restraint_convention"),
        "steps_per_segment": int(payload["steps"]),
        "reporting_intervals": {
            "all_atom": reporting.get("full_system_interval_steps"),
            "selected_atoms": reporting.get("selected_atoms_interval_steps"),
            "state_log": reporting.get("state_interval_steps"),
        },
        "selected_atoms_count": len(selected_atoms or []),
        "selected_atoms_fingerprint": (
            _atom_identity(topology, selected_atoms) if topology is not None and selected_atoms
            else None),
        # WHICH atoms land in WHICH file, in what order. A permutation that preserves the count
        # produces a trajectory whose frames are silently scrambled relative to the first segment's.
        "output_atom_ordering": {
            "all_atom": (_atom_identity(topology, range(topology.getNumAtoms()))
                         if topology is not None else None),
            "selected_atoms": (
                _atom_identity(topology, selected_atoms)
                if topology is not None and selected_atoms else None),
        },
        "predecessor": predecessor,
        # kept as flat fields too: they are what the refusal message quotes
        "input_state": payload["input"]["state"],
        "produced_by": payload["input"]["produced_by"],
    }


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
                            actual_platform, assert_dynamics_platform,
                                _set_barostat, solute_atom_indices)
    from .reporting import attach_reporters

    cfg = _runtime_cfg(payload)
    # A single-shot or cMD stage runs one Context, so it takes the FIRST device of the requested
    # set. Refusing to interpret --devices here would have meant the flag silently did nothing on
    # every stage except REST2, and the run would land on whichever GPU CUDA picked.
    device_index = None
    if devices:
        first = str(devices).split(",")[0].strip()
        if first:
            device_index = int(first)
            cfg["production"]["device_index"] = device_index
    system = XmlSerializer.deserialize((here / payload["input"]["system_xml"]).read_text())
    pdb = PDBFile(str(here / payload["input"]["topology"]))
    # Asked of the SYSTEM. A nonperiodic Context still returns default box vectors, so testing a
    # State tells you only that OpenMM filled in a default -- never that the run has a box.
    periodic = bool(system.usesPeriodicBoundaryConditions())

    # A conventional-MD stage owns a run directory and a committed-generation record, so it decides
    # its continuation HERE -- before the Context exists and before any output file is opened. An
    # incompatible continuation must be refused while the previous segment's outputs are still
    # exactly as it left them.
    segmented = stage.startswith("cMD")
    continuation = None
    cmd_plan = None
    cmd_continuity = None
    run_dir = here / "run"

    # The reporting selection is resolved BEFORE the continuity contract is built, because the
    # contract binds its ordered atom identity. Resolving it later is how it came to be passed as
    # None, which let a permuted or different selection of the same length compare equal.
    reporting = payload.get("reporting") or {}
    selected_atoms: list = []
    if stage != "min" and int(payload.get("steps", 0)) > 0:
        selection = (reporting.get("selected_atoms") or {}).get("type", "solute")
        selected_atoms = solute_atom_indices(pdb.topology, selection)
        expected = reporting.get("selected_atoms_expected_count")
        if expected is not None and int(expected) != len(selected_atoms):
            raise StageError(
                f"{stage}: the selection {selection!r} resolves to {len(selected_atoms)} atoms in "
                f"this topology, but the project was generated expecting {int(expected)}. The "
                "bundle and the project disagree; regenerate the project against this bundle."
            )

    restraint = payload.get("restraint")
    restraint_index = None
    if restraint is not None:
        # The reference is the PREPARED coordinates, so "restrained to the input structure" means
        # the same thing in every restrained stage rather than drifting with its predecessor.
        reference_state = XmlSerializer.deserialize(
            (here / ".." / restraint["reference"]).read_text())
        reference = np.array(
            reference_state.getPositions().value_in_unit(unit.nanometer))
        restraint_index, restrained_atoms, restraint_convention = _add_positional_restraints(
            system, pdb.topology, "solute", reference)
    else:
        restrained_atoms = []
        restraint_convention = None

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

    run_lock = None
    if segmented:
        from . import cmd_segments

        # Held for the whole segment. Released when the process exits, including when it is killed.
        run_lock = cmd_segments.hold_run_lock(run_dir)
        payload["_restraint_convention"] = restraint_convention
        cmd_plan = cmd_segments.plan_segment(payload)
        cmd_continuity = _cmd_continuity(payload, system, here=here, topology=pdb.topology,
                                         selected_atoms=selected_atoms)
        continuation = cmd_segments.prepare_continuation(
            run_dir, continuity=cmd_continuity, plan=cmd_plan)

        # Phase one of the output check: every committed stream is INSPECTED and nothing is
        # touched. A stream shorter than its watermark is missing history and must stop the run
        # while the other streams are still exactly as the last segment left them.
        outputs = payload["output"]
        streams = [
            {"name": "all-atom trajectory", "path": here / outputs["trajectory_all_atoms"],
             "watermark": continuation["watermarks"]["all_atom"], "kind": "dcd",
             "n_atoms": system.getNumParticles()},
            {"name": "selected-atom trajectory",
             "path": here / outputs["trajectory_selected_atoms"],
             "watermark": continuation["watermarks"]["selected_atoms"], "kind": "dcd",
             "n_atoms": len(selected_atoms) or None},
            {"name": "state log", "path": here / outputs["log"],
             "watermark": continuation["watermarks"]["state_log"], "kind": "log",
             "n_atoms": None},
        ]
        cmd_segments.assert_committed_outputs_intact(streams)

    # Every stage passes through here, so this is the one place the platform can be enforced.
    # Checked BEFORE the Context is built: constructing one on the wrong device and discarding it
    # would already have allocated the memory and chosen the device.
    assert_dynamics_platform(cfg, stage)
    sim = _make_simulation(pdb.topology, system, cfg, seed=int(stage_seeds["integrator"]),
                           device_index=device_index)
    # Velocities are initialised ONCE, on entering dynamics from a state that carries none -- in
    # practice NVT, after minimisation. Every later stage inherits them. Re-drawing at each stage
    # discards the equilibration that stage just paid for and hides it behind a plausible-looking
    # temperature.
    if segmented and not continuation["first_segment"]:
        # Continue from the last COMMITTED generation, not from the equilibration endpoint. The
        # checkpoint is preferred; the State fallback is physically valid but does not restore the
        # integrator's random stream, so it is announced and recorded rather than silently taken.
        from . import runstate

        restart_source = runstate.load_restart(sim, continuation["restore_from"])
        from . import cmd_segments as _cmd

        # Read what the restart ACTUALLY restored before touching anything. Assigning the step
        # first -- which is what this used to do -- writes the expected value into the Context and
        # then "verifies" it against itself, so a checkpoint from the wrong generation passed.
        position = _cmd.read_restart_position(sim)
        verification = _cmd.verify_restart_matches_commit(
            sim,
            {"absolute_step": continuation["absolute_step"],
             "absolute_time_ps": continuation.get("absolute_time_ps")},
            position=position)
        # Only now, and only when the restart did not carry a step of its own.
        restart_provenance = _cmd.restore_restart_step(sim, verification,
                                                       restart_source=restart_source)
        coords_origin = {"coords": str(continuation["restore_from"]),
                         "kind": f"committed generation ({restart_source})",
                         "velocities": "restored from the committed generation",
                         "velocity_seed": None,
                         "restart_provenance": restart_provenance}
        print(f"  {stage}: continuing from generation "
              f"{continuation['segments_completed']} via {restart_source} "
              f"(step {restart_provenance['started_absolute_step']:,} from "
              f"{restart_provenance['step_origin']})")
        if not restart_provenance["bitwise_continuation"]:
            print(f"  {stage}: {restart_provenance['note']}", flush=True)
    else:
        restart_source = "predecessor state"
        restart_provenance = None
        coords_origin = _apply_coords(sim, here / payload["input"]["state"],
                                      require_velocities=(stage != "min"),
                                      velocity_seed=int(stage_seeds["velocity"]))

    if restraint_index is not None:
        # kcal/mol/A^2 -> kJ/mol/nm^2
        k = restraint["force_constant_kcal_per_mol_angstrom2"] * 4.184 * 100.0
        sim.context.setParameter("k_restraint", k)

    # Every declared output is attached here, so what the configuration promises is what the stage
    # writes. Minimisation gets none: it takes no steps, so a step-interval reporter would produce
    # an empty file that looks like a broken one.
    attached: dict = {}
    if stage != "min" and int(payload.get("steps", 0)) > 0:
        outputs = payload["output"]
        append = bool(continuation and not continuation["first_segment"])
        if segmented:
            # Any frames beyond the committed watermark belong to an invocation that died before
            # its boundary. They are removed BEFORE the reporters open, so appending cannot leave
            # the trajectory containing frames no generation accounts for.
            from . import cmd_segments

            trimmed = [
                cmd_segments.truncate_to_watermark(
                    here / outputs["trajectory_all_atoms"], "all_atom",
                    continuation["watermarks"]["all_atom"], n_atoms=system.getNumParticles()),
                cmd_segments.truncate_to_watermark(
                    here / outputs["trajectory_selected_atoms"], "selected_atoms",
                    continuation["watermarks"]["selected_atoms"], n_atoms=len(selected_atoms)),
                cmd_segments.truncate_to_watermark(
                    here / outputs["log"], "state_log",
                    continuation["watermarks"]["state_log"], n_atoms=0),
            ]
            continuation["tail_recovery"] = trimmed
        attached = attach_reporters(
            sim,
            append=append,
            all_atom_path=here / outputs["trajectory_all_atoms"],
            all_atom_interval_steps=reporting.get("full_system_interval_steps"),
            selected_path=here / outputs["trajectory_selected_atoms"],
            selected_interval_steps=reporting.get("selected_atoms_interval_steps"),
            selected_atoms=selected_atoms,
            state_path=here / outputs["log"],
            state_interval_steps=reporting.get("state_interval_steps")
                                 or reporting.get("full_system_interval_steps"),
            total_steps=int(payload["steps"]),
            periodic=periodic,
        )

    # The barostat is recorded because whether one was applied is a fact about the stage, while
    # whether the volume actually moved is a sampling outcome: a MonteCarloBarostat can reject every
    # move in a short stage, so an unchanged box is not evidence that the barostat was missing.
    results: dict = {"stage": stage, "n_restrained_atoms": len(restrained_atoms),
                     "barostat": (payload.get("barostat") or {}).get("type"),
                     # What the Context reports, not what the configuration asked for.
                     "platform": actual_platform(sim),
                     # which distance the restraint measured. An implicit run using minimum-image
                     # distance would depend on box vectors it is not supposed to have.
                     "restraint_convention": restraint_convention,
                     "seeds": dict(stage_seeds),
                     # Four distinct provenances, named distinctly. "inherited" for a segment
                     # that was RESTORED from a committed generation would be true but useless: a
                     # reader checking that velocities were created once needs to tell a restart
                     # from a hand-off between stages.
                     "velocities": ("not required" if stage == "min"
                                    else "restored from the committed generation"
                                    if str(coords_origin.get("kind", "")).startswith("committed")
                                    else "initialized"
                                    if coords_origin.get("velocity_seed") is not None
                                    else "inherited"),
                     "velocity_seed": coords_origin.get("velocity_seed"),
                     "reporters": attached,
                     "selected_atoms": {"type": (reporting.get("selected_atoms") or {}).get(
                                            "type", "solute"),
                                        "n_atoms": len(selected_atoms),
                                        "indices_sha256": _selection_fingerprint(selected_atoms)}}
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

    if segmented:
        # Close the boundary in order: reporters first, so the files on disk end exactly here;
        # then the restart pair; then the atomic commit. A crash before the commit leaves a tail
        # that the next invocation removes.
        from . import cmd_segments, runstate

        from .faults import crash_point

        # Boundary 1: reporters still open, so the trajectory runs past the last commit.
        crash_point("before_reporters_close")
        cmd_segments.close_reporters(sim)
        # Boundary 2: streams end exactly at the boundary, but nothing is saved for it.
        crash_point("after_reporters_close")

        generation = int(continuation["segments_completed"]) + 1
        absolute_step = int(continuation["absolute_step"]) + steps
        absolute_time_ps = float(
            sim.context.getState().getTime().value_in_unit(unit.picosecond))

        # `run_state.json` is written for compatibility and for readers, but it is a CACHE. The
        # invocation itself is published only by the atomic commit below, so a crash between the
        # two cannot leave a phantom invocation for a retry to duplicate.
        if continuation["first_segment"]:
            runstate.write_run_state(run_dir, method="md", continuity=cmd_continuity,
                                     stage=stage)
        committed = cmd_segments.commit_segment(
            run_dir, sim, generation=generation, plan=cmd_plan,
            absolute_step=absolute_step, absolute_time_ps=absolute_time_ps,
            continuity=cmd_continuity, restart_source=restart_source,
            started_step=int(continuation["absolute_step"]),
            started_time_ps=float(continuation.get("absolute_time_ps") or 0.0),
            invocation={"steps": steps, "stage": stage},
            extra={"periodic": periodic, "restart_provenance": restart_provenance})
        # Boundary 5: committed, but run_state.json still describes the previous generation.
        crash_point("after_atomic_commit")

        # reconciled from the commit, never the other way round
        runstate.record_invocation(run_dir, {
            "segment": generation,
            "steps": steps,
            "restart_source": restart_source,
            "absolute_step_after": absolute_step,
            "absolute_time_ps_after": absolute_time_ps,
            "reconciled_from": "committed.json",
        })
        results["segment"] = {
            "generation": generation,
            "absolute_step": absolute_step,
            "absolute_time_ps": absolute_time_ps,
            "restart_source": restart_source,
            "restart_provenance": restart_provenance,
            "continued": not continuation["first_segment"],
            "watermarks": committed["watermarks"],
            "tail_recovery": continuation.get("tail_recovery"),
        }
        print(f"  {stage}: committed generation {generation}, "
              f"absolute step {absolute_step:,}, t = {absolute_time_ps:.1f} ps")

    # Minimisation writes NO velocities. That absence is the handshake: the first dynamics stage
    # sees a state without them and initialises once, and every stage after that inherits. Writing
    # zeros instead would look like inherited velocities at 0 K and never initialise.
    state = sim.context.getState(getPositions=True, getVelocities=(stage != "min"), getEnergy=True,
                                 enforcePeriodicBox=periodic)
    results["potential_after_kj_mol"] = state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    # A nonperiodic System has no volume. Recording the determinant of OpenMM's default box vectors
    # would put a number here that looks like a thermodynamic measurement and is not one, so the
    # absence is recorded explicitly with its reason instead.
    if periodic:
        box = state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        results["box_volume_nm3"] = float(abs(np.linalg.det(np.array(box))))
        results["periodic"] = True
    else:
        results["box_volume_nm3"] = None
        results["periodic"] = False
        results["volume_note"] = (
            "not applicable: this System is nonperiodic (implicit solvent), so it has no volume "
            "and no density. OpenMM still exposes default box vectors; their determinant is not a "
            "thermodynamic quantity.")

    (here / payload["output"]["final_state"]).write_text(XmlSerializer.serialize(state))
    with (here / payload["output"]["final_structure"]).open("w") as handle:
        PDBFile.writeFile(sim.topology, state.getPositions(), handle, keepIds=True)
    sim.saveCheckpoint(str(here / payload["output"]["checkpoint"]))
    (here / payload["output"]["results"]).write_text(json.dumps(results, indent=2) + "\n")

    volume = (f"V {results['box_volume_nm3']:.2f} nm^3" if results["box_volume_nm3"] is not None
              else "no box (implicit)")
    print(f"  {stage}: U {results['potential_before_kj_mol']:.1f} -> "
          f"{results['potential_after_kj_mol']:.1f} kJ/mol, {volume}, "
          f"{results.get('steps', 0)} steps, {len(restrained_atoms)} restrained atoms")
    return 0


def _selection_fingerprint(indices) -> str:
    """A hash of the resolved selection, so two runs can be compared without storing every index.

    Atom ORDER is part of it: a trajectory written for one order and analysed against another is
    wrong in a way that no file size or frame count reveals.
    """
    import hashlib

    if not indices:
        return ""
    payload = ",".join(str(int(i)) for i in indices).encode()
    return sha256_bytes(payload)[:16]


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
