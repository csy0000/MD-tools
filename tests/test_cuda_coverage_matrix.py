"""Every CUDA-dependent function and branch, inventoried and exercised on real NVIDIA hardware.

"CUDA tested" has to mean something checkable. It cannot mean "the GPU tests pass", because that
sentence is true of a suite that never touches half the branches -- and the halves it misses are
the ones nobody thought about, which is why they are the ones that break.

So this file holds two things:

1. **THE INVENTORY.** Every function that creates an OpenMM `Context` or `Simulation`, selects or
   configures CUDA, evaluates energies or forces on it, integrates or minimises, changes REST2 or
   AIS Context parameters, or writes output derived from CUDA execution. It is checked against the
   source by `ast`, so a NEW Context-creating function that nobody added to the matrix fails this
   file rather than quietly going untested.

2. **THE LANES.** Real runs on real GPUs for the branches the other GPU files do not reach:
   the three CUDA precisions, HMR and non-HMR timesteps, explicit `--device` placement, a NetCDF
   source ensemble, and the two-state AIS Hamiltonian -- whose claim is an identity between
   energies (`V = (1 - lambda) V0 + lambda V1`, `dV/dlambda = V1 - V0`), and single and mixed
   precision are exactly where an identity stops holding.

`docs/release-notes/cuda-coverage-matrix.md` is generated from this file by
`python -m pytest tests/test_cuda_coverage_matrix.py --cuda-evidence=<path>`, so the published
matrix is a record of a run rather than a table someone typed.

NOTHING HERE SUBSTITUTES. There is no CPU fallback, no Reference lane and no mock: a test that
cannot get a CUDA device fails, and is reported as an unmet criterion. That is the point.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "md_tools"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


# --- the inventory --------------------------------------------------------------------------------

#: `module::function` -> what it does on the GPU, and which lane proves it.
#:
#: Every entry is a place a CUDA kernel actually runs. The mapping is data rather than prose so
#: `test_every_cuda_site_in_the_source_is_in_this_matrix` can hold it to the source.
CUDA_SITES = {
    "ais/run.py::run_one_path.write_cv": (
        "calls `context.getState(getPositions=True)` to observe a collective variable, which on "
        "CUDA is a device synchronise and a device-to-host copy of the positions. Deliberately "
        "positions only -- never `getEnergy` -- so a CV observation costs no Hamiltonian "
        "evaluation and cannot be confused with one in the run's cost accounting",
        # NOT `test_ais_cv_output.py`: that file invokes `--cpu`, so it exercises this function
        # on the CPU platform and is not CUDA evidence for it. It was cited here anyway, which is
        # worse than an empty cell -- a gap invites work, a false entry closes the question.
        "test_cv_cuda_lanes.py::test_ais_cv_and_two_state_on_cuda_fresh_and_resumed"),
    "md/stage.py::_EnergyDecompositionProbe.energies": (
        "the explicit-solvent half of the energy decomposition. Where a System carries no usable "
        "force groups -- which is every System built through `ForceField.createSystem`, i.e. all "
        "explicit solvent -- this creates a SECOND Context, on `live.getPlatform()`, over a "
        "group-separated COPY of the System, pushes the live positions and box into it, and asks "
        "for one energy per group. On CUDA that is a second resident System, a host-to-device "
        "copy of the positions at every report, and one restricted energy evaluation per group. "
        "It belongs here and not among the reporters that only read a State handed to them: like "
        "`_EnergyComponentsReporter.report` it asks the device for numbers the run would not "
        "otherwise compute. The production Context is untouched -- that is the whole point of the "
        "copy -- but the device work is real",
        # The probe engages only on a System with one force group, so an IMPLICIT lane never
        # reaches it: ParmEd's grouping means the plain reporter path is used instead. The
        # explicit lanes are the ones that exercise it.
        "test_cuda_coverage_matrix.py::test_explicit_solvent_npt_lane, "
        "test_explicit_solvent_rest2_lane, test_ais_on_explicit_solvent"),
    "md/stage.py::_EnergyComponentsReporter.report": (
        "one `context.getState(getEnergy=True, groups={g})` per force group per report, which on "
        "CUDA is a device synchronise and an energy evaluation restricted to that group. It is a "
        "REAL cost and belongs here rather than among the reporters that only read a State handed "
        "to them: the decomposition asks the device for numbers the run would not otherwise "
        "compute, once per group, at the state table's cadence",
        "test_cmd_cuda_smoke.py, and every cMD lane that reports a state table"),
    "remd/rung_equilibration.py::run_stage": (
        "per-tau equilibration: one Context per stage per rung, on the platform and properties "
        "the ladder's preflight resolved -- restraint strength pushed in after the configuration, "
        "`integrator.step`, and the equilibrated positions, velocities and box read back",
        "test_rest2_equilibration_per_tau_cuda.py::test_per_tau_equilibration_under_mpi_on_cuda"),
    "openmm/placement.py::measure_device_throughput": (
        "the measurement every multi-worker placement is decided from: one CUDA Context per "
        "VISIBLE device in turn, over this run's own System, warmed up and then integrated for "
        "about a second, and the steps per second of each. It is real device work -- a Context, "
        "an integrator and thousands of steps per card -- done before any output exists, and it "
        "is why a slower card carries fewer workers",
        # A --cpu file would not be evidence for this: the function is only reached on CUDA.
        "test_placement_cuda.py::test_the_throughput_measurement_runs_on_every_visible_device "
        "(4 ranks, 4 cards, asserts a positive rate per card)"),
    "run/preflight.py::_verify_mps": (
        "opens a one-particle CUDA Context on THIS rank's device and, while it is held, asks the "
        "driver whether this process is listed as an MPS client (`M+C`) or an ordinary CUDA one "
        "(`C`). The Context is the point: a process that holds none is not listed at all, so the "
        "verdict could not be read. It runs only when a device hosts more than one worker",
        # Both verdicts, on hardware: the refusal path with no reachable daemon, and the accepted
        # path as a real client of one. Neither is arrangeable on the CPU.
        "test_placement_cuda.py::test_four_workers_on_one_card_are_refused_without_mps_and_write_"
        "nothing (verdict `C`), test_four_workers_share_one_card_with_mps_and_the_ladder_completes "
        "and test_twelve_workers_on_four_cards_run_together_under_mps (verdict `M+C`; both need a "
        "daemon this process can reach, and SKIP rather than pass without one)"),
    "openmm/platform_policy.py::_prove_cuda_initialises": (
        "opens a one-particle CUDA Context to prove the platform works before any output exists",
        "test_platform_policy.py, and every preflight in every lane below"),
    "openmm/platform_policy.py::resolve_platform_request": (
        "constructs the CUDA Platform and its Precision/DeviceIndex properties",
        "test_cuda_precision_lane, test_explicit_device_placement_lane"),
    "openmm/builders.py::_write_initial_state": (
        "a Context over the built System, to write its initial state",
        "build-top in every lane's fixture"),
    "md/stage.py::stage_main": (
        "minimisation, NVT/NPT dynamics, restraints, barostat, reporters, checkpoint commit",
        "test_cmd_cuda_smoke.py, test_cmd_restart_integration.py, test_cuda_precision_lane, "
        "test_hmr_timestep_lane"),
    "remd/engine.py::ReplicaEngine.__init__": (
        "one Simulation per thermodynamic state, all on CUDA",
        "test_md_run_mpi_gpu.py, test_multi_rank_rest2_ladders_of_several_sizes"),
    "remd/engine.py::count_cuda_devices": (
        "opens a Context per device index to count what this process can actually see",
        "test_explicit_device_placement_lane"),
    "remd/driver.py::ReplicaRun._build_platform": (
        "consumes the preflight platform; no second resolution",
        "test_md_run_mpi_gpu.py::test_each_rank_kept_its_own_record_and_ran_on_cuda"),
    "ais/run.py::run_one_path": (
        "AIS switching between two end states: the mixed System's Context, lambda updates, "
        "work from dV/dlambda, observation potentials, frames, state rows and checkpoint "
        "generations, all on CUDA",
        "test_md_run_mpi_gpu.py, test_ais_two_state_lane"),

    # --- operations the constructor-only inventory never saw ---------------------------------
    "ais/two_state.py::TwoStateHamiltonian.set_lambda": (
        "pushes lambda into a live CUDA Context with `Context.setParameter` -- the one global of "
        "the CustomCVForce that mixes V0 and V1. Nothing is re-uploaded, for any System",
        "test_ais_two_state_lane (in-process on CUDA, and through a run), "
        "test_ais_on_explicit_solvent"),
    "ais/two_state.py::TwoStateHamiltonian.difference": (
        "one CUDA `getState(getParameterDerivatives=True)` per switch at the frozen pre-switch "
        "coordinate: dV/dlambda = V1 - V0, from the inner CUDA Contexts of both end states -- "
        "the number every work value is made of",
        "test_ais_two_state_lane (against V1 - V0 from separate CUDA Contexts, and against the "
        "finite difference at frozen coordinates), test_ais_on_explicit_solvent"),
    "ais/two_state.py::TwoStateHamiltonian.observe": (
        "the observation potentials at a saved coordinate: V(lambda) and dV/dlambda in one CUDA "
        "`getState`, checked against the CustomCVForce collective-variable values the inner CUDA "
        "Contexts evaluate, then V0 and V1 derived from them",
        "test_ais_two_state_lane, test_ais_on_explicit_solvent, "
        "test_hs_rows_match_recomputation_on_cuda"),
    "ais/run.py::run_one_path.write_frame": (
        "reads CUDA positions and box vectors into the path trajectory",
        "test_ais_reads_a_netcdf_source_on_cuda, test_ais_two_state_lane"),
    "ais/run.py::run_one_path.save_checkpoint": (
        "saves a CUDA Context into a committed checkpoint generation",
        "test_md_run_mpi_gpu.py resume tests, test_hs_rows_match_recomputation_on_cuda"),
    "ais/run.py::_state_row": (
        "reads CUDA potential, kinetic and box state into system.csv",
        "test_ais_two_state_lane (info_printout is set in every AIS lane)"),
    "md/_stages.py::set_restraint": (
        "pushes the positional-restraint force constant into a live CUDA Context",
        "test_cmd_cuda_smoke.py, test_cuda_precision_lane (restrained NVT)"),
    "md/torsion_restraints.py::TorsionRestraint.set_strength": (
        "pushes the TORSIONAL restraint force constant into a live CUDA Context -- the biasing "
        "half of umbrella sampling. The same one-parameter push as the positional restraint "
        "above, and listed separately because it is a different force with a different failure "
        "mode: a positional restraint that fails to arrive lets a solute drift visibly, while a "
        "torsional one that fails to arrive produces a window that simply samples the unbiased "
        "ensemble and looks like a legitimate, if oddly broad, distribution",
        "test_torsion_restraint_cuda.py"),
    "md/_stages.py::write_final_state": (
        "reads CUDA positions, velocities and parameters into the restart",
        "every cMD lane; asserted in test_explicit_solvent_npt_lane"),
    "md/stage.py::_CheckpointWithFingerprint.report": (
        "commits a CUDA checkpoint generation with every stream's committed counts",
        "test_cmd_restart_integration.py, test_cuda_precision_lane"),
    "md/phase_space.py::PhaseSpaceReporter.report": (
        "pulls CUDA positions, velocities and box vectors off the device on every phase-space "
        "step of a fixed-tau run",
        "test_cv_cuda_lanes.py::test_fixed_tau_phase_space_and_cv_resume_on_cuda (writes and "
        "resumes the stream on CUDA), test_cmd_restart_integration.py (its committed counts "
        "across a resume)"),
    "remd/engine.py::ReplicaEngine.propagate": (
        "integrates every owned state on CUDA between exchanges",
        "test_md_run_mpi_gpu.py, test_multi_rank_rest2_ladders_of_several_sizes"),
    "remd/engine.py::ReplicaEngine.minimise": (
        "CUDA minimisation of a ladder rung",
        "test_multi_rank_rest2_ladders_of_several_sizes (its equilibration chain)"),
    "remd/engine.py::ReplicaEngine.potential_energy": (
        "the CROSS-Hamiltonian energies an exchange decision is made from",
        "test_md_run_mpi_gpu.py, test_multi_rank_rest2_ladders_of_several_sizes"),
    "remd/engine.py::ReplicaEngine.get_configuration": (
        "reads CUDA positions/box out of a rung for storage or exchange",
        "test_multi_rank_rest2_ladders_of_several_sizes, test_explicit_solvent_rest2_lane"),
    "remd/engine.py::ReplicaEngine.set_configuration": (
        "writes a configuration into a CUDA rung -- the swap itself",
        "test_multi_rank_rest2_ladders_of_several_sizes, "
        "test_cv_mpi_cuda_lanes.py::test_cv_continuation_under_mpi_on_cuda"),
    "remd/engine.py::ReplicaEngine.set_velocities_to_temperature": (
        "draws momenta on a CUDA rung",
        "test_multi_rank_rest2_ladders_of_several_sizes"),
    # CORRECTED. These two rows previously cited `test_md_run_mpi_gpu.py ladder resume`, and
    # that was false in a way worth stating: until the ladder checkpoint began storing per-rung
    # context checkpoints, NOTHING called either function. They were written for exactly this
    # purpose and left unwired, so the matrix claimed CUDA coverage for dead code -- which is
    # worse than a gap, because a gap invites work and a false entry closes the question.
    "remd/engine.py::ReplicaEngine.integrator_state": (
        "reads a CUDA rung's OpenMM context checkpoint into the ladder checkpoint, so a "
        "continuation reproduces the trajectory rather than merely a valid one",
        "test_cv_mpi_cuda_lanes.py::test_cv_continuation_under_mpi_on_cuda"),
    "remd/engine.py::ReplicaEngine.load_integrator_state": (
        "restores a CUDA rung's context checkpoint on resume, including the integrator's "
        "pseudo-random stream position, which coordinates alone do not carry",
        "test_cv_mpi_cuda_lanes.py::test_cv_continuation_under_mpi_on_cuda"),
}

#: Functions that construct a Context but never on CUDA, with the reason. Each is a deliberate,
#: named exemption rather than an omission -- and the reason is checkable by reading the callsite.
NON_CUDA_CONTEXT_SITES = {
    "build/md.py::_generated_cv_text":
        "reads the bond graph for `collective_variables.generate` off the SERIALISED built System "
        "-- `HarmonicBondForce.getBondParameters` and `System.getConstraintParameters` are System "
        "accessors on the host -- and the atom names off built.pdb. It creates no Context and asks "
        "no device for anything; it writes a cv.yaml at build time, before any run exists.",
    "md/stage.py::_EnergyDecompositionProbe.__init__":
        "deserialises a COPY of the System and assigns each force its own group, so the probe can "
        "ask for them separately later. `XmlSerializer` round trip and `setForceGroup` only -- no "
        "Context exists yet, and the run's own System is deliberately never touched, because a "
        "force group is part of the serialised System and regrouping the integrated one would "
        "change `system_sha256` and invalidate every checkpoint fingerprint in flight. The device "
        "work is in `energies`, which is classified as a CUDA site.",
    "md/stage.py::_system_census":
        "counts atoms and residues, sums the per-particle charges off the NonbondedForce, derives "
        "the degree-of-freedom count from 3N less constraints less the CM remover, and reads the "
        "default box vectors -- all off the SERIALISED System, on the host. It constructs nothing "
        "and asks no device for anything; `getParticleParameters` is a System accessor, not a "
        "Context one.",
    "md/stage.py::_method_summary":
        "reads the nonbonded method, cutoff, Ewald tolerance, PME grid, switching, exception "
        "count, constraint count and force inventory off the SERIALISED System. Host-side "
        "accessors on a System object; no Context and no platform are involved. It is read off "
        "the System rather than the configuration precisely so the `.out` describes what will "
        "integrate rather than what was requested.",
    "openmm/decomposition.py::decomposition_systems":
        "builds the per-term System COPIES whose single-point energies are the Amber-style terms, "
        "by zeroing charges or epsilons and removing forces. `XmlSerializer` round trips and "
        "System mutation only -- no Context is created here, and none of these Systems is ever "
        "integrated.",
    "openmm/decomposition.py::decompose.energy":
        "creates a Context per term to read one single-point energy, and defaults to "
        "`platform_name='CPU'`. This is POST-HOC analysis: the module exists because the 1-4 and "
        "electrostatic/Lennard-Jones splits need forces duplicated, which must never happen to a "
        "System something integrates. A caller may pass `platform_name='CUDA'`, but no test does, "
        "and its only test is PLATFORM_POLICY_EXEMPTION-marked CPU arithmetic -- so filing it as "
        "CUDA-covered would be the same false entry the AIS CV path once carried: a gap invites "
        "work, a false entry closes the question. If a CUDA post-hoc lane is ever added, this "
        "moves to CUDA_SITES and cites it.",
    "md/completion.py::verify_completed_stage":
        "deserialises the final State FROM XML ON DISK to re-read its particle count. The same "
        "getPositions spelling as a live Context, but the object is a file's contents -- the "
        "verifier is entirely read-only and creates no Context at all.",
    "md/cv_report.py::CVReporter.observe":
        "reads positions and box off a State it is HANDED -- by OpenMM's reporter protocol, or by "
        "the caller for the step-0 observation -- and evaluates torsions on the host in numpy. It "
        "constructs nothing, asks the Context for nothing, and requests no energy: "
        "`describeNextReport` asks for positions only, which is what keeps a CV-enabled run the "
        "same experiment as a CV-disabled one. The device work that produced the State belongs to "
        "whoever created it (`md/stage.py::stage_main`).",
    "openmm/system.py::protonate":
        "names the Reference platform explicitly, to add hydrogens deterministically while "
        "building a System. No dynamics, and deliberately platform-independent so a structure "
        "built on one machine is the structure built on another.",
    "remd/driver.py::ReplicaRun._begin":
        "constructs an ExchangeContext, which is bookkeeping around Simulations the engine "
        "already created on CUDA -- the kernels are the engine's, and are covered by its entry.",
    "reference/standalone_build.py::explicit_system":
        "build-top's System construction written out for a bundle's input/build_system.py: it "
        "calls createSystem and reads values back off the System it just built. No Context and "
        "no platform exist; the script rebuilds a System and compares its bytes, and never "
        "integrates anything.",
    "openmm/solvation.py::solvate":
        "reads box vectors off a Topology while building the solvated system. A Topology is "
        "geometry on the host; no Context exists yet, and nothing has been computed on a device.",
    "md/stage.py::_AmberStreamReporter.report":
        "reads positions and box off a State it is HANDED by OpenMM's reporter protocol and "
        "writes them to an AMBER NetCDF on the host. It constructs nothing and asks the Context "
        "for nothing; the device work that produced the State belongs to whoever created it "
        "(`md/stage.py::stage_main`). Same standing as `md/cv_report.py::CVReporter.observe`.",
    "md/stage.py::_EnergyComponentsReporter.__init__":
        "reads the force GROUPS off the System object while building its header. A System is not "
        "a Context: no device is involved, nothing is evaluated, and the groups it reads are the "
        "ones the build already assigned -- reassigning any of them would change the serialised "
        "System's digest and make every run in flight unresumable, which is precisely why this "
        "only reads.",
    "md/stage.py::_coordinate_reporter":
        "CHOOSES a reporter class by filename suffix and returns it -- DCD for the whole system, "
        "AMBER NetCDF when an atom subset is wanted. No Context, no State, no device: the match "
        "here is on the word `report` in a constructor, not on an operation.",
    "remd/driver.py::ReplicaRun._read_initial_configuration":
        "reads positions and velocities out of a State DESERIALISED FROM XML on disk. The same "
        "getPositions spelling as a live Context, but the object is a file's contents.",
}


#: Every CUDA-relevant OPERATION, not merely the constructors.
#:
#: The previous inventory looked for `Context(` and `Simulation(` and claimed to cover "every CUDA
#: operation". It did not: a function that steps an integrator, pushes parameters into a Context
#: somebody else built, reads forces, or saves a checkpoint runs CUDA kernels without constructing
#: anything, and none of those were being classified at all.
#:
#: Grouped by what the kernel does, because that is what a lane has to exercise.
CUDA_OPERATIONS = {
    # construction
    "Context": "construct", "Simulation": "construct",
    # dynamics
    "step": "integrate", "minimizeEnergy": "minimize",
    # energies and forces
    "getState": "evaluate",
    # parameters -- REST2/AIS switching, and the reinitialisation that follows a System change
    "setParameter": "parameters", "updateParametersInContext": "parameters",
    "reinitialize": "parameters", "setPositions": "parameters",
    "setVelocitiesToTemperature": "parameters", "setPeriodicBoxVectors": "parameters",
    # persistence of CUDA-derived state
    "saveCheckpoint": "checkpoint", "loadCheckpoint": "checkpoint",
    "createCheckpoint": "checkpoint", "setState": "checkpoint",
    # REPORTERS. A reporter is attached once and then runs on EVERY reporting step for the rest of
    # the simulation, pulling state off the device each time -- the single highest-frequency CUDA
    # consumer in the package, and none of it was classified. A reporter that writes the wrong
    # frames, or writes them at the wrong stride, is a corrupt trajectory that the run reports as
    # a success.
    "DCDReporter": "report", "NetCDFReporter": "report", "XTCReporter": "report",
    "PDBReporter": "report", "PDBxReporter": "report", "StateDataReporter": "report",
    "CheckpointReporter": "report",
    # CUDA-DERIVED OUTPUT. What the device computed, on its way to a file or a decision: every
    # coordinate written to a trajectory, every energy in a work table, every box vector in an
    # NPT record. `getState` above says a state was read; these say what was taken OUT of it,
    # and they are the values the scientific output is made of.
    "getPositions": "derive", "getVelocities": "derive", "getForces": "derive",
    "getPotentialEnergy": "derive", "getKineticEnergy": "derive",
    "getPeriodicBoxVectors": "derive", "getPeriodicBoxVolume": "derive",
    "getParameters": "derive",
}


def _cuda_operation_sites() -> dict[str, set[str]]:
    """Every function in the package that performs a CUDA-relevant operation, and which ones.

    Found by walking the AST rather than by grepping, so a call spread over several lines, or one
    inside a nested function, is found the same way as a one-liner. Attribute calls count:
    `simulation.step(n)` and `context.getState(...)` are the common spellings and neither is a
    bare name.
    """
    found: dict[str, set[str]] = {}
    for module in sorted(SRC.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))

        def walk(node, stack):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(child, stack + [child.name])
                    continue
                if isinstance(child, ast.Call):
                    name = getattr(child.func, "id", None) or getattr(child.func, "attr", None)
                    if name in CUDA_OPERATIONS and stack:
                        key = f"{module.relative_to(SRC).as_posix()}::{'.'.join(stack)}"
                        found.setdefault(key, set()).add(CUDA_OPERATIONS[name])
                walk(child, stack)

        walk(tree, [])
    return found


def _context_creating_functions() -> dict[str, str]:
    """Kept as the narrow question: which functions CONSTRUCT a Context or a Simulation."""
    return {site: "construct" for site, kinds in _cuda_operation_sites().items()
            if "construct" in kinds}


def test_every_cuda_operation_in_the_source_is_in_this_matrix():
    """A new CUDA-relevant operation must be classified, not silently left untested.

    This is what keeps the matrix from becoming a table that was true once. The previous version
    looked only for `Context(` and `Simulation(` while claiming to cover "every CUDA operation" --
    and a function that steps an integrator, pushes parameters into a Context somebody else built,
    reads forces, or saves a checkpoint runs CUDA kernels without constructing anything. Sixteen
    such sites were unclassified when this was widened.
    """
    found = _cuda_operation_sites()
    classified = set(CUDA_SITES) | set(NON_CUDA_CONTEXT_SITES)
    unclassified = sorted(set(found) - classified)
    assert not unclassified, (
        "these functions perform a CUDA-relevant operation and appear in neither CUDA_SITES nor "
        "NON_CUDA_CONTEXT_SITES:\n  " + "\n  ".join(
            f"{site} {sorted(found[site])}" for site in unclassified))


def test_the_inventory_looks_for_more_than_constructors():
    """The guard on the guard: the operation set must cover what the task enumerates."""
    kinds = set(CUDA_OPERATIONS.values())
    assert {"construct", "integrate", "minimize", "evaluate", "parameters",
            "checkpoint", "report", "derive"} <= kinds, kinds
    found = _cuda_operation_sites()
    seen = set()
    for operations in found.values():
        seen |= operations
    # Every category must actually be FOUND somewhere in the source, or the pattern is watching
    # for something that is spelled differently and silently matching nothing.
    for kind in ("construct", "integrate", "evaluate", "parameters", "checkpoint",
                 "report", "derive"):
        assert kind in seen, f"no source site performs {kind!r}; the matcher is not matching"


def test_an_unclassified_reporter_site_actually_fails_the_inventory(tmp_path, monkeypatch):
    """The guard on the guard, run against a site that does not exist yet.

    `test_every_cuda_operation_in_the_source_is_in_this_matrix` passes today. That is exactly the
    state a broken matcher produces too: a pattern that matches nothing classifies nothing and
    reports no unclassified sites. So this ADDS a source file with a reporter that pulls CUDA
    state off the device, leaves it out of both tables, and requires the inventory to say so.

    A reporter is the case worth pinning: it is attached once and then runs on every reporting
    step for the rest of the simulation, which makes it the highest-frequency CUDA consumer in
    the package and the easiest one to add without noticing.
    """
    package = tmp_path / "md_tools"
    (package / "md").mkdir(parents=True)
    (package / "md" / "new_reporter.py").write_text(
        "class DensityReporter:\n"
        "    def report(self, simulation, state):\n"
        "        volume = state.getPeriodicBoxVolume()\n"
        "        return volume\n"
        "\n"
        "def attach(simulation, path):\n"
        "    simulation.reporters.append(DCDReporter(path, 100))\n",
        encoding="utf-8")

    monkeypatch.setattr(sys.modules[__name__], "SRC", package)
    found = _cuda_operation_sites()
    unclassified = set(found) - (set(CUDA_SITES) | set(NON_CUDA_CONTEXT_SITES))
    assert "md/new_reporter.py::DensityReporter.report" in unclassified, (
        "a reporter reading CUDA box state was not detected as a CUDA-relevant site; the matcher "
        "is matching nothing and the matrix's completeness claim is empty")
    assert "md/new_reporter.py::attach" in unclassified, (
        "a reporter CONSTRUCTION site was not detected")
    assert found["md/new_reporter.py::DensityReporter.report"] == {"derive"}
    assert found["md/new_reporter.py::attach"] == {"report"}


def test_every_matrix_entry_still_names_a_function_that_exists():
    """The other direction: an entry left behind after a rename claims coverage of nothing."""
    stale = []
    for entry in sorted(set(CUDA_SITES) | set(NON_CUDA_CONTEXT_SITES)):
        module, _, qualified = entry.partition("::")
        path = SRC / module
        if not path.is_file():
            stale.append(f"{entry} (no such module)")
            continue
        leaf = qualified.split(".")[-1]
        if f"def {leaf}(" not in path.read_text(encoding="utf-8"):
            stale.append(f"{entry} (no such function)")
    assert not stale, "matrix entries naming nothing:\n  " + "\n  ".join(stale)


# --- the hardware ---------------------------------------------------------------------------------

def _cuda_devices() -> list[dict[str, str]]:
    """What `nvidia-smi` says is here. Recorded in the evidence, never inferred."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    devices = []
    for line in done.stdout.strip().splitlines():
        index, name, driver, memory = (part.strip() for part in line.split(",", 3))
        devices.append({"index": index, "name": name, "driver": driver, "memory": memory})
    return devices


@pytest.fixture(scope="module")
def hardware():
    devices = _cuda_devices()
    if not devices:
        pytest.fail("no NVIDIA device is visible. This file is the CUDA evidence lane: a missing "
                    "GPU is an unmet acceptance criterion, not a reason to skip.")
    from openmm import Platform

    names = {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}
    if "CUDA" not in names:
        pytest.fail(f"OpenMM offers no CUDA platform; it has {sorted(names)}")
    return devices


def _visible() -> int:
    """How many devices THIS process may use -- the count every capacity check is about.

    Not `len(hardware)`. `hardware` is what `nvidia-smi` reports, and `nvidia-smi` ignores
    `CUDA_VISIBLE_DEVICES`, so a suite confined to five of nine cards counted nine: the placement
    lane asked OpenMM for device 8 ("Illegal value for DeviceIndex: 8"), and the six-state ladder
    passed its own "needs six devices" guard, ran, and failed later with two ranks on one card.
    The inventory stays what the evidence records; the count comes from the engine's own
    `visible_cuda_devices`, which is what the runtime places ranks with.
    """
    from md_tools.remd.engine import visible_cuda_devices

    return len(visible_cuda_devices(probe=True))


def _environment(work: Path, **extra) -> dict[str, str]:
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    config = work / "user.config"
    # NESTED under `machine:`, which is where the resolver looks. Splatting the inner mapping in
    # at the top level wrote `openmm:` as a root key, the machine settings were therefore absent,
    # and every precision lane silently ran at the built-in default while claiming to test three.
    machine = extra.pop("machine", None)
    document = {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}
    if machine:
        document["machine"] = machine
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(config)
    base.update(extra)
    return base


def _machine(platform="CUDA", precision="mixed", device_policy="local_rank") -> dict:
    return {"machine": {"openmm": {"platform": platform, "precision": precision,
                                   "device_policy": device_policy}}}


def _end_states(root: Path) -> list[str]:
    """`-p/-s` V0 and `-p2/-s2` V1 for an AIS lane on the system under `root`.

    V0 is `build/V0.xml`, the REST2 state at tau = 0.5 of `build/built.xml`, written on first use;
    V1 is `built.xml` itself. The pair the retired single-topology AIS switched along tau, as the
    two files the two-state AIS transforms between.
    """
    v0 = root / "build" / "V0.xml"
    if not v0.is_file():
        from md_tools.rest2.hamiltonian import build_scaled_system
        from md_tools.md.stage import solute_atom_indices
        from openmm import XmlSerializer
        from openmm.app import PDBFile

        base = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
        solute = solute_atom_indices(PDBFile(str(root / "build" / "built.pdb")).topology)
        v0.write_text(XmlSerializer.serialize(build_scaled_system(base, solute, 0.5)),
                      encoding="utf-8")
    return ["-p", str(root / "build" / "built.pdb"), "-s", str(v0),
            "-p2", str(root / "build" / "built.pdb"), "-s2", str(root / "build" / "built.xml")]


def _ais_system(root: Path, work: Path) -> list[str]:
    """Make `work` a system root of its own for an AIS lane, and return its end-state flags.

    `build-md` validates the whole chain at generation and reads `<system>/build/`, and
    `-odir work/project` makes `work` the system root -- so the built System (and V0 beside it)
    is copied in, exactly as the ladder lanes do.
    """
    import shutil

    _end_states(root)
    shutil.copytree(root / "build", work / "build")
    return _end_states(work)


def _counter_detail(counters: dict) -> str:
    return (f"{counters['paid_total_energy_evaluations']} energy evaluations "
            f"({counters['work_derivative_evaluations']} dV/dlambda, "
            f"{counters['observation_potential_energy_evaluations']} observation), "
            f"{counters['parameter_updates']} lambda updates in "
            f"{counters['parameter_change_seconds']:.6f} s")


def _assert_two_state_rows(rows, *, precision: str) -> tuple[int, float]:
    """The two-state identity on every frame-aligned work row; emptiness on every other one.

    Returns (aligned rows, worst |mixture - direct|). Also the running sum of the per-path work.
    """
    from md_tools.ais.two_state import OBSERVATION_POTENTIAL_COLUMNS, identity_tolerance

    aligned, worst, running = 0, 0.0, {}
    for row in rows:
        path = row["path_id"]
        running[path] = running.get(path, 0.0) + float(row["delta_work_kj_mol"])
        total = float(row["total_work_kj_mol"])
        assert abs(running[path] - total) <= 1e-9 * max(1.0, abs(total)), (
            f"path {path} step {row['switch_step']}: the work does not telescope")
        if str(row.get("coordinate_frame_index", "")).strip() == "":
            for name in OBSERVATION_POTENTIAL_COLUMNS:
                assert row[name] == "", (row["switch_step"], name)
            continue
        lam = float(row["lambda_after"])
        v0, v1 = float(row["potential_v0_kj_mol"]), float(row["potential_v1_kj_mol"])
        direct = float(row["potential_direct_kj_mol"])
        error = abs((1.0 - lam) * v0 + lam * v1 - direct)
        worst = max(worst, error)
        assert error <= identity_tolerance(max(abs(v0), abs(v1), abs(direct)),
                                           precision=precision), (
            f"step {row['switch_step']}: (1 - {lam}) V0 + {lam} V1 = "
            f"{(1.0 - lam) * v0 + lam * v1} against direct {direct} at {precision} precision")
        aligned += 1
    return aligned, worst


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One implicit ALA system, built once. Small enough to run many lanes against."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cuda-matrix")
    (root / "build").mkdir(exist_ok=True)
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml",
               "-op", "build/built.pdb", "-log", "build/built.log",
               "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _seed_build(source_root: Path, work: Path) -> None:
    """The dataset root's built System, copied in before a run is generated into `work`.

    `build-md -odir work/<run>` makes `work` the dataset root, and generation validates the chain
    against the System every run on that root shares -- refusing a root with no `build/built.xml`.
    Copied rather than rebuilt: the physics is identical and `build-top` is the slow part.
    """
    import shutil

    shutil.copytree(source_root / "build", work / "build")


def _ladder_launch(project: Path, start: Path) -> None:
    """Put the equilibrated state where the generated group file names it, `eq/eq_3.xml`.

    A ladder reads `-s` and `-c` only from its group file (0.5.4), and that file resolves
    `-i _protocol.py` beside itself while the runtime writes the helper into `-odir` -- so a
    ladder runs in its run directory with `-odir .`, as `run.sh` launches it. `_equilibrate` runs
    the chain into a directory of its own; its last restart is the state `run.sh`'s chain would
    have left in `eq/eq_3.xml`.
    """
    import shutil

    (project / "eq").mkdir(exist_ok=True)
    shutil.copy2(start, project / "eq" / "eq_3.xml")


#: Results collected across the lanes, written into the evidence document at teardown.
RESULTS: list[dict[str, object]] = []


def _record(lane: str, *, feature: str, precision: str, device: str, detail: str = "") -> None:
    RESULTS.append({"lane": lane, "feature": feature, "precision": precision,
                    "device": device, "result": "passed", "detail": detail})


# --- precision ------------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
def test_cuda_precision_lane(precision, built, hardware, tmp_path):
    """A cMD stage on CUDA at each supported precision, proved from the run's own record.

    Precision is a `machine.openmm` setting and it reaches the Context as a CUDA property. If it
    did not, every one of these would pass identically and prove nothing -- so the assertion is on
    what the RUN recorded, which is read back out of the provenance log.
    """
    work = tmp_path / precision
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")
    _seed_build(built, work)
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"), "-odir", str(work)],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine(precision=precision)))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "cMD.log")
    acceleration = record["acceleration"]
    assert acceleration["resolved_platform"] == "CUDA", acceleration
    assert acceleration.get("cuda_precision") == precision, (
        f"the run recorded precision {acceleration.get('precision')!r}, not {precision!r}: the "
        f"setting did not reach the Context, so this lane proves nothing")
    assert (work / "solute_prod1.nc").is_file() and (work / "cMD.xml").is_file()
    _record("test_cuda_precision_lane", feature=f"cMD implicit, {precision} precision",
            precision=precision, device=acceleration.get("cuda_device_index") or "OpenMM-selected",
            detail="minimisation, restrained NVT, production, DCD, restart, checkpoint")


# --- HMR and the timestep -------------------------------------------------------------------------

def test_hmr_timestep_lane(built, hardware, tmp_path):
    """4 fs runs on CUDA when the masses prove repartitioning, and is refused when they do not.

    Both halves on the GPU, because the refusal is the half that matters and a refusal proved on
    the CPU says nothing about the platform the run would have used.

    MIGRATED for the dataset-root check. `build-md` now validates the chain against the root's
    own `build/built.xml`, so a 4 fs project can only be GENERATED on an HMR root -- on an ordinary
    one generation itself refuses, which is asserted as (a0). The stage-level refusal on the GPU is
    then the same generated 4 fs stage pointed at the ordinary System, (a).
    """
    work = tmp_path / "hmr"
    work.mkdir()
    (work / "fast.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "dynamics": {"timestep_fs": 4.0},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")

    # (a0) generation refuses 4 fs on a root whose System has ordinary masses.
    ordinary = work / "ordinary"
    ordinary.mkdir()
    _seed_build(built, ordinary)
    refused_generation = subprocess.run(CLI + ["build-md", "-odir", str(ordinary / "fast"),
                                               "--config", str(work / "fast.config")],
                                        capture_output=True, text=True, timeout=600)
    assert refused_generation.returncode != 0, refused_generation.stdout
    assert "repartitioning" in refused_generation.stdout + refused_generation.stderr, (
        refused_generation.stdout + refused_generation.stderr)
    assert not (ordinary / "fast").exists(), "the refused generation created output"

    # An HMR-built system is the root the 4 fs project is generated on.
    (work / "hmr.config").write_text(
        "solvent:\n  model: GBn2\nhydrogen_mass_repartitioning:\n  enabled: true\n",
        encoding="utf-8")
    (work / "build").mkdir()
    hmr_built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(work / "hmr.config")],
        cwd=work, capture_output=True, text=True, timeout=1800)
    if hmr_built.returncode != 0:
        pytest.fail(f"could not build an HMR system, so the 4 fs lane cannot be run:\n"
                    f"{hmr_built.stdout}{hmr_built.stderr}")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "fast"),
                                      "--config", str(work / "fast.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    # (a) the refusal: the 4 fs stage against the ordinary masses in `built.xml`.
    destination = work / "refused"
    refused = subprocess.run(
        [sys.executable, str(work / "fast" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, **_machine()))
    assert refused.returncode != 0, refused.stdout + refused.stderr
    assert "repartitioning" in refused.stdout + refused.stderr, (
        "refused, but not for the masses: " + (refused.stdout + refused.stderr)[-2000:])
    assert not destination.exists(), "the refusal created output"

    # (b) the HMR-built system takes 4 fs and runs.
    ran = subprocess.run(
        [sys.executable, str(work / "fast" / "cMD.py"),
         "-p", str(work / "build" / "built.pdb"), "-s", str(work / "build" / "built.xml"),
         "-odir", str(work / "ok")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert ran.returncode == 0, ran.stdout + ran.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "ok" / "cMD.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert float(record["timestep"]["timestep_fs"]) == 4.0, record["timestep"]
    _record("test_hmr_timestep_lane", feature="cMD implicit, 4 fs with HMR; 4 fs refused without",
            precision="mixed", device=record["acceleration"].get("cuda_device_index") or "-",
            detail="both halves on CUDA")


# --- explicit device placement --------------------------------------------------------------------

def test_explicit_device_placement_lane(built, hardware, tmp_path):
    """`--device N` puts THIS process on device N, and the record says which one.

    Run on the last visible device rather than device 0, so a placement that was ignored shows up
    as a difference instead of coinciding with the default.
    """
    visible = _visible()
    if visible < 2:
        pytest.fail(f"only {visible} CUDA device(s) visible; explicit placement cannot be "
                    f"distinguished from the default. Reported as an unmet criterion.")
    wanted = visible - 1

    work = tmp_path / "device"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 10}}),
        encoding="utf-8")
    _seed_build(built, work)
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"),
         "-odir", str(work / "run"), "--device", str(wanted)],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    acceleration = read_record(work / "run" / "cMD.log")["acceleration"]
    assert acceleration["resolved_platform"] == "CUDA"
    assert str(acceleration["cuda_device_index"]) == str(wanted), acceleration
    _record("test_explicit_device_placement_lane", feature="cMD implicit, --device placement",
            precision="mixed", device=str(wanted),
            detail=f"{visible} of {len(hardware)} devices visible; ran on the last")


# --- the two-state AIS Hamiltonian, on CUDA ----------------------------------------------------------

def test_ais_two_state_lane(built, hardware, tmp_path):
    """`TwoStateHamiltonian` on a real GPU, in process and through a run, at mixed precision.

    The Hamiltonian's whole claim is a pair of identities between energies:
    `V(lambda) = (1 - lambda) V0 + lambda V1` and `dV/dlambda = V1 - V0`. On the Reference platform
    they are algebra; here they are about the PLATFORM -- the CustomCVForce's inner Contexts and
    the energy-parameter derivative kernel are CUDA code, and mixed precision is exactly where an
    identity stops holding to a tolerance somebody assumed on a CPU.

    IN PROCESS: the class itself, on a CUDA Context, against V0 and V1 each evaluated in a
    separate plain CUDA Context, and the work convention's finite difference at frozen
    coordinates. THROUGH A RUN: the counters and the identity columns of the work table the run
    wrote.
    """
    import csv

    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    from md_tools.ais.two_state import TWO_STATE_SCHEMA, TwoStateHamiltonian, identity_tolerance
    from md_tools.build.record import read_record

    _end_states(built)

    # -- in process -------------------------------------------------------------------------------
    cuda = Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}
    v0_system = XmlSerializer.deserialize((built / "build" / "V0.xml").read_text(encoding="utf-8"))
    v1_system = XmlSerializer.deserialize((built / "build" / "built.xml").read_text(
        encoding="utf-8"))
    positions = PDBFile(str(built / "build" / "built.pdb")).positions

    def plain(system):
        context = Context(system, VerletIntegrator(0.001), cuda, properties)
        context.setPositions(positions)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    v0, v1 = plain(v0_system), plain(v1_system)
    hamiltonian = TwoStateHamiltonian(v0_system, v1_system)
    assert hamiltonian.plan["mixed_forces"], "the pair mixes nothing"
    context = Context(hamiltonian.system, VerletIntegrator(0.001), cuda, properties)
    assert context.getPlatform().getName() == "CUDA"
    context.setPositions(positions)
    allowed = identity_tolerance(max(abs(v0), abs(v1)), precision="mixed")
    energies = {}
    for lam in (0.0, 0.25, 1.0):
        hamiltonian.set_lambda(context, lam)
        observed = hamiltonian.observe(context, lam, precision="mixed", where="CUDA lane: ")
        assert abs(observed["potential_v0_kj_mol"] - v0) <= allowed, (lam, observed, v0)
        assert abs(observed["potential_v1_kj_mol"] - v1) <= allowed, (lam, observed, v1)
        assert abs(observed["potential_direct_kj_mol"]
                   - ((1.0 - lam) * v0 + lam * v1)) <= allowed, (lam, observed)
        assert abs(hamiltonian.difference(context) - (v1 - v0)) <= allowed, lam
        energies[lam] = observed["potential_direct_kj_mol"]
    # The work convention, at one frozen coordinate: V(0.25) - V(0) = 0.25 (V1 - V0).
    assert abs((energies[0.25] - energies[0.0]) - 0.25 * (v1 - v0)) <= 2 * allowed

    # -- through a run ----------------------------------------------------------------------------
    work = tmp_path / "ais"
    work.mkdir()
    argv = _ais_system(built, work)

    import mdtraj

    frames = mdtraj.load(str(built / "build" / "built.pdb"))
    mdtraj.join([frames] * 12).save_dcd(str(work / "source.dcd"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 20,
                "observation_interval_steps": 5,
                "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10, "checkpoint_printout": 10}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"), *argv,
         "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA", record["acceleration"]
    assert record["end_states"]["schema"] == {"name": TWO_STATE_SCHEMA["name"],
                                              "version": TWO_STATE_SCHEMA["version"]}
    assert record["end_states"]["mixed_forces"], record["end_states"]
    counters = record["evaluation_counters"]
    # The counters, on a real CUDA run, in the relationship the record itself states.
    assert counters["work_derivative_evaluations"] > 0
    assert counters["observation_potential_energy_evaluations"] > 0
    useful = (counters["work_derivative_evaluations"]
              + counters["observation_potential_energy_evaluations"]
              + counters["other_useful_energy_evaluations"])
    assert counters["useful_total_energy_evaluations"] == useful, counters
    assert counters["paid_total_energy_evaluations"] == (
        useful + counters["known_discarded_energy_evaluations"]), counters
    # An uninterrupted run observed all of its own cost; nothing was thrown away unseen.
    assert counters["known_discarded_energy_evaluations"] == 0, counters
    assert counters["discarded_is_complete"] is True, counters
    assert counters["parameter_updates"] > 0

    with (work / "run" / "AIS_work.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the work table is empty"
    precision = record["acceleration"].get("cuda_precision") or "mixed"
    aligned, worst = _assert_two_state_rows(rows, precision=precision)
    assert aligned >= 2, "fewer than two frame-aligned rows carried observation potentials"
    _record("test_ais_two_state_lane",
            feature="AIS implicit, TwoStateHamiltonian in process and through a run",
            precision=precision, device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"in process: V0, V1, V(lambda) and dV/dlambda against plain CUDA Contexts; "
                   f"run: {len(rows)} rows, {aligned} frame-aligned, worst |mixture - direct| = "
                   f"{worst:.3e} kJ/mol; " + _counter_detail(counters))


def test_ais_reads_a_netcdf_source_on_cuda(built, hardware, tmp_path):
    """The other supported source format. DCD is covered in test_md_run_mpi_gpu.py."""
    work = tmp_path / "netcdf"
    work.mkdir()
    argv = _ais_system(built, work)

    import mdtraj

    frames = mdtraj.load(str(built / "build" / "built.pdb"))
    mdtraj.join([frames] * 12).save_netcdf(str(work / "source.nc"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": 10,
                "observation_interval_steps": 5, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.nc"},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"), *argv,
         "-source-traj", str(work / "source.nc"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=1800,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["source"]["format"] == "netcdf", record["source"]
    # mdtraj writes no System digest, so the ensemble is ASSERTED to be V0's, and says so.
    assert record["source"]["ensemble"]["status"] == "asserted", record["source"]
    assert (work / "run" / "AIS_traj0000.nc").is_file()
    _record("test_ais_reads_a_netcdf_source_on_cuda",
            feature="AIS implicit, NetCDF source ensemble",
            precision=record["acceleration"].get("cuda_precision") or "mixed",
            device=record["acceleration"].get("cuda_device_index") or "-",
            detail="source format decided from content, not suffix")


# --- no fallback ----------------------------------------------------------------------------------

def test_there_is_no_automatic_cpu_fallback_on_this_machine(built, hardware, tmp_path):
    """With CUDA unavailable, a run REFUSES. It does not quietly become a CPU run.

    Exercised through the deterministic seam rather than by removing a driver, and asserted here
    -- in the CUDA file -- because "there is no fallback" is a claim about the CUDA lane.
    """
    work = tmp_path / "nofallback"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 5},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    _seed_build(built, work)
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    destination = work / "run"
    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, MD_TOOLS_FORCE_NO_CUDA="1", **_machine()))
    assert done.returncode != 0, done.stdout + done.stderr
    assert "CPU" not in done.stdout.replace("--cpu", ""), (
        "the run mentioned falling back to the CPU:\n" + done.stdout)
    assert not destination.exists(), "the refusal created output"
    _record("test_there_is_no_automatic_cpu_fallback_on_this_machine",
            feature="no automatic CPU fallback", precision="-", device="-",
            detail="CUDA made unavailable through the deterministic seam")


# --- explicit solvent -----------------------------------------------------------------------------
#
# Every lane above is implicit, which is the fast one. Explicit solvent is a different code path on
# the GPU in every way that matters: PME rather than a cutoff, a barostat that can be active, a box
# that has to survive a checkpoint, and a particle count two orders of magnitude larger. A matrix
# that proved only the implicit branch would be proving the branch that does not use PME.

@pytest.fixture(scope="module")
def built_explicit(tmp_path_factory):
    """One small explicit ALA box, built once. Slow enough to be worth building only here."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("cuda-matrix-explicit")
    (root / "build").mkdir(exist_ok=True)
    # `TIP3P` in the spelling the schema accepts, and the default padding: unknown keys and
    # unknown values are both refused rather than ignored, which is what made this fixture fail
    # loudly instead of quietly building something else.
    (root / "sys.config").write_text("solvent:\n  model: TIP3P\n", encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml",
               "-op", "build/built.pdb", "-log", "build/built.log",
               "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        pytest.fail("could not build an explicit-solvent system, so the explicit CUDA lanes "
                    f"cannot run:\n{done.stdout}{done.stderr}")
    return root


def test_explicit_solvent_npt_lane(built_explicit, hardware, tmp_path):
    """Explicit cMD on CUDA: PME, a real barostat, NPT, and a box that survives the restart."""
    work = tmp_path / "explicit"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    _seed_build(built_explicit, work)
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built_explicit / "build" / "built.pdb"), "-s", str(built_explicit / "build" / "built.xml"),
         "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "cMD.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["implicit"] is False, "this lane is meant to be the explicit one"
    assert record["stage"]["ensemble"] == "NPT", record["stage"]
    assert record["barostats"]["active"] >= 1, (
        f"an NPT explicit stage ran with no active barostat: {record['barostats']}")
    # The box has to come back out of the restart, or a continuation silently changes the volume.
    from openmm import XmlSerializer

    state = XmlSerializer.deserialize(
        (work / "run" / "cMD.xml").read_text(encoding="utf-8"))
    vectors = state.getPeriodicBoxVectors()
    assert vectors is not None and vectors[0][0]._value > 0.0, vectors
    _record("test_explicit_solvent_npt_lane",
            feature="cMD explicit (PME), NPT with an active barostat",
            precision=record["acceleration"].get("cuda_precision") or "mixed",
            device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{record['stage']['steps']} production steps; box preserved in the restart")


def _equilibrate_chain(project: Path, topology: Path, system: Path, work: Path, run: Path, *,
                       last: str) -> Path:
    """Run a cMD project's whole chain, including its production stage, and return that stage."""
    return _equilibrate(project, topology, system, work, run, script=f"{last}.py",
                        include_last=True)


def _equilibrate(project: Path, topology: Path, system: Path, work: Path, run: Path, *,
                 script: str = "REST2.py", include_last: bool = False) -> Path:
    """Run the project's equilibration chain and return the state the ladder starts from.

    A ladder is refused without `-c`: every rung starts from the same equilibrated configuration,
    and a group line with no coordinates is a rung starting from the topology's own positions.
    So the lanes below run the chain rather than inventing a starting state -- which also puts
    the minimisation and every equilibration stage on CUDA, in the same lane.
    """
    from md_tools.md.stage import load_generated_plan

    plan, _config = load_generated_plan(project / script)
    previous = None
    for stage in plan:
        name = stage["name"]
        # WHERE THE SCRIPT ACTUALLY IS, under the split layout: `min/` belongs to the SYSTEM,
        # `eq/` to the run, and the production stage sits at the run root.
        #
        # This looked only at the run root and `continue`d when the file was absent. After the
        # layout split that silently skipped minimisation AND every equilibration stage, then
        # failed on "the chain produced no starting state" -- a missing script now names itself
        # instead of being passed over, because skipping the whole chain is not a smaller version
        # of running it.
        # BY THE FILING KEY, which is what the scripts are named: `eq_nvt_posres` is generated as
        # `eq/eq_1.py`. The stage name decides the physics, the key decides the filename, and
        # `stage_plan` stamps the key onto every plan entry so both sides read it from one place.
        key = str(stage.get("file_key") or name)
        candidates = (project / f"{key}.py", project / "eq" / f"{key}.py",
                      project.parent / "min" / f"{key}.py")
        script_path = next((path for path in candidates if path.is_file()), None)
        assert script_path is not None, (
            f"{name} (filed as {key}): no generated script in "
            + ", ".join(str(c) for c in candidates))
        argv = [sys.executable, str(script_path), "-p", str(topology), "-s", str(system),
                "-odir", str(run)]
        if previous is not None:
            argv += ["-c", str(previous)]
        done = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=3600,
                              env=_environment(work, **_machine()))
        assert done.returncode == 0, f"{name}:\n{done.stdout}{done.stderr}"
        # The restart is filed under the key too, which is what the next stage's `-c` must name.
        previous = run / f"{key}.xml"
    assert previous is not None and previous.is_file(), "the chain produced no starting state"
    return previous


def test_explicit_solvent_rest2_lane(built_explicit, hardware, tmp_path):
    """A REST2 ladder on explicit solvent, on CUDA, serially.

    The scaled Hamiltonian over a PME system is the combination REST2 is actually used for, and
    it is the one where a NonbondedForce exception or a reciprocal-space term scaling wrongly
    would show up as an acceptance ratio nobody questions.
    """
    import shutil

    work = tmp_path / "rest2-explicit"
    work.mkdir()
    # A LADDER IS GENERATED FROM THE BUILT SYSTEM. The rungs are scaled and serialised at build
    # time now, so `build-md` reads `<system>/build/built.xml` -- and `-odir work/project` makes
    # `work` the system root. Without this the generation refused before writing anything.
    shutil.copytree(built_explicit / "build", work / "build")
    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "explicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 10,
                   "restrained_npt_steps": 10, "unrestrained_npt_steps": 10,
                   "production_steps": 20},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 10,
                  "number_of_exchanges": 2},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4); `build-md` refuses without them.
    from .conftest import make_states_for

    make_states_for(work, work / "REST2.config")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    start = _equilibrate(work / "project", built_explicit / "build" / "built.pdb",
                         built_explicit / "build" / "built.xml", work, work / "run")
    _ladder_launch(work / "project", start)
    done = subprocess.run(
        [sys.executable, str(work / "project" / "REST2.py"),
         "-p", str(built_explicit / "build" / "built.pdb"), "--groupfile", "remd_groupfile.1",
         "-odir", "."],
        cwd=work / "project", capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout + done.stderr

    trajectories = sorted((work / "project").glob("whole_state*_prod1.nc"))
    assert len(trajectories) == 2, [p.name for p in trajectories]
    report = (work / "project" / "REST2.out").read_text(encoding="utf-8")
    assert "platform           : CUDA" in report, report[:2000]
    _record("test_explicit_solvent_rest2_lane",
            feature="REST2 explicit (PME), 2 states, serial",
            precision="mixed", device="-",
            detail=f"{len(trajectories)} state trajectories; exchanges attempted on CUDA")


@pytest.mark.parametrize("states", [2, 4, 6])
def test_multi_rank_rest2_ladders_of_several_sizes(states, built, hardware, tmp_path):
    """Real `mpirun -n N` ladders on CUDA, one rank per state, one GPU per rank.

    Sizes rather than one size: `owned_states` and the neighbour exchange rule both depend on the
    state count's parity, and a ladder that only ever ran with 2 has never exercised the case
    where a rank has a neighbour on both sides. 6 is the largest this machine's 9 devices allow
    with a device to spare.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the multi-rank CUDA lane is an unmet criterion")
    if _visible() < states:
        pytest.fail(f"{states} states need {states} devices; {_visible()} visible")

    work = tmp_path / f"ladder-{states}"
    work.mkdir()
    # The ladder's rungs are scaled from the built System at BUILD time, so the system root that
    # holds this run has to carry it.
    shutil.copytree(built / "build", work / "build")
    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "production_steps": 20},
        "rest2": {"number_of_replicas": states, "exchange_interval_steps": 10,
                  "number_of_exchanges": 2},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4); `build-md` refuses without them.
    from .conftest import make_states_for

    make_states_for(work, work / "REST2.config")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    start = _equilibrate(work / "project", built / "build" / "built.pdb", built / "build" / "built.xml",
                         work, work / "run")
    _ladder_launch(work / "project", start)
    done = subprocess.run(
        ["mpirun", "-n", str(states), sys.executable, str(work / "project" / "REST2.py"),
         "-p", str(built / "build" / "built.pdb"), "--groupfile", "remd_groupfile.1",
         "-odir", ".", "-ng", str(states)],
        cwd=work / "project", capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    trajectories = sorted((work / "project").glob("whole_state*_prod1.nc"))
    assert len(trajectories) == states, [p.name for p in trajectories]

    # One device per rank, and they must be DIFFERENT devices: `device_policy: local_rank` is the
    # setting, and N ranks sharing one GPU is the failure it exists to prevent.
    devices = []
    for rank in range(states):
        name = "REST2.out" if rank == 0 else f"REST2.out.rank{rank:02d}"
        text = (work / "project" / name).read_text(encoding="utf-8")
        assert "platform           : CUDA" in text, text[:1500]
        line = next(line for line in text.splitlines() if "platform           :" in line)
        devices.append(line.split("device=")[1].split()[0] if "device=" in line else None)
    assert len(set(devices)) == states, f"ranks shared devices: {devices}"
    _record("test_multi_rank_rest2_ladders_of_several_sizes",
            feature=f"REST2 implicit, {states} states, real mpirun -n {states}",
            precision="mixed", device=", ".join(str(d) for d in devices),
            detail=f"{len(trajectories)} state trajectories; one rank per device")


def test_a_genuinely_absent_cuda_device_is_refused_without_any_seam(built, hardware, tmp_path):
    """`CUDA_VISIBLE_DEVICES=""`. No seam, no monkeypatch: the driver really has no device.

    `MD_TOOLS_FORCE_NO_CUDA` proves the refusal PATH, and it is our own code deciding to fail --
    which is exactly the thing under test, so on its own it is a weak witness. Emptying
    `CUDA_VISIBLE_DEVICES` makes the CUDA platform fail in OpenMM, in the driver, with
    `CUDA_ERROR_NO_DEVICE`. That is the failure a machine with a busy or broken GPU actually
    produces, and the run must refuse rather than quietly become a CPU run.
    """
    work = tmp_path / "nodevice"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 5},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    _seed_build(built, work)
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    destination = work / "run"
    done = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"), "-odir", str(destination)],
        cwd=work, capture_output=True, text=True, timeout=900,
        env=_environment(work, CUDA_VISIBLE_DEVICES="", **_machine()))
    message = done.stdout + done.stderr
    assert done.returncode != 0, message
    assert not destination.exists(), (
        f"the refusal created output: {sorted(p.name for p in destination.iterdir())}")
    # And it refused for the right reason, naming the platform rather than something downstream.
    assert "CUDA" in message, message[-2000:]
    _record("test_a_genuinely_absent_cuda_device_is_refused_without_any_seam",
            feature="CUDA genuinely unavailable (CUDA_VISIBLE_DEVICES=\"\"): refused, no fallback",
            precision="-", device="none visible",
            detail="real CUDA_ERROR_NO_DEVICE from the driver, not a test seam")


def test_a_genuinely_unimportable_mpi4py_stops_a_plural_launch(built, hardware, tmp_path):
    """A broken mpi4py, made broken the way a broken one is: it raises on import.

    `MD_TOOLS_FORCE_NO_MPI4PY` proves the path and is again our own code choosing to fail. This
    shadows the real package with one whose `__init__` raises the ImportError a missing
    `libmpi.so` actually produces, so what runs is the genuine `except ImportError` branch in
    `md_tools.remd.mpi` -- the one that must refuse a plural launch rather than let N ranks each
    believe they are the whole world and write over one set of files.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the plural-launch lane is an unmet criterion")

    work = tmp_path / "brokenmpi"
    (work / "shadow" / "mpi4py").mkdir(parents=True)
    # As above: generating the ladder needs the built System beside the run.
    shutil.copytree(built / "build", work / "build")
    (work / "shadow" / "mpi4py" / "__init__.py").write_text(
        'raise ImportError("libmpi.so.40: cannot open shared object file: '
        'No such file or directory")\n', encoding="utf-8")

    (work / "REST2.config").write_text(yaml.safe_dump({
        "protocol": "REST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "production_steps": 10},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                  "number_of_exchanges": 2},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5}}),
        encoding="utf-8")
    # A ladder integrates SAVED scaled states (0.5.4); `build-md` refuses without them.
    from .conftest import make_states_for

    make_states_for(work, work / "REST2.config")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "REST2.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    environment = _environment(work, **_machine())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(work / "shadow"), environment.get("PYTHONPATH", "")])

    # IN THE RUN DIRECTORY with `-odir .`, the only `-odir` a ladder's group file allows, so
    # "nothing written" is the run directory's listing unchanged rather than `-odir` absent.
    project = work / "project"

    def listing():
        return sorted(str(p.relative_to(project)) for p in project.rglob("*"))

    # The state every group line continues from, so the launch is refused for mpi4py rather than
    # for a missing `-c eq/eq_3.xml` first.
    from .conftest import write_starting_state

    write_starting_state(work, project)
    before = listing()
    done = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(project / "REST2.py"),
         "-p", str(built / "build" / "built.pdb"), "--groupfile", "remd_groupfile.1",
         "-odir", "."],
        cwd=project, capture_output=True, text=True, timeout=900, env=environment)
    message = done.stdout + done.stderr
    assert done.returncode != 0, message[-2000:]
    assert "mpi4py" in message, message[-2000:]
    assert listing() == before, (
        "two uncoordinated ranks created output over one set of paths: "
        f"{sorted(set(listing()) - set(before))}")
    _record("test_a_genuinely_unimportable_mpi4py_stops_a_plural_launch",
            feature="mpi4py genuinely unimportable under mpirun -n 2: refused, nothing written",
            precision="-", device="-",
            detail="the real ImportError branch, through a shadowing package, not a test seam")


def test_ais_on_explicit_solvent(built_explicit, hardware, tmp_path):
    """AIS over a PME system on CUDA, and the two-state identity with it.

    Every other AIS lane is implicit. The identity's claim is that it survives the PME reciprocal
    sum, the Ewald self-energy and the long-range dispersion correction -- each end state is
    evaluated as its own System would be -- and none of those exist in an implicit system, so no
    implicit lane tests the part of the claim most likely to be wrong.
    """
    work = tmp_path / "ais-explicit"
    work.mkdir()
    argv = _ais_system(built_explicit, work)

    import mdtraj

    frames = mdtraj.load(str(built_explicit / "build" / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(work / "source.dcd"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "explicit",
        "ais": {"number_of_paths": 2, "switching_steps": 10,
                "observation_interval_steps": 5, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5,
                      "checkpoint_printout": 5}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                      "--config", str(work / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"), *argv,
         "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    import csv

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA"
    assert record["implicit"] is False, "this lane is meant to be the explicit one"
    assert "NonbondedForce" in record["end_states"]["mixed_force_classes"], record["end_states"]
    precision = record["acceleration"].get("cuda_precision") or "mixed"

    with (work / "run" / "AIS_work.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    aligned, worst = _assert_two_state_rows(rows, precision=precision)
    assert aligned >= 2, "fewer than two frame-aligned rows carried observation potentials"

    counters = record["evaluation_counters"]
    particles = record["inputs"]["system"].get("particles") if isinstance(
        record.get("inputs", {}).get("system"), dict) else None
    _record("test_ais_on_explicit_solvent",
            feature="AIS explicit (PME), two-state identity through the reciprocal sum",
            precision=precision, device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{len(rows)} rows, {aligned} frame-aligned, worst |mixture - direct| = "
                   f"{worst:.3e} kJ/mol; " + _counter_detail(counters)
                   + (f"; {particles} particles" if particles else ""))


def test_the_two_state_switch_cost_is_measured_on_a_large_system(built, built_explicit,
                                                                hardware, tmp_path):
    """What a lambda update and its work cost, implicit against explicit, every step.

    "The overhead is acceptable" is only a claim if it comes with a number, and a number from a
    22-particle implicit system says nothing about a solvated one -- where the differing
    NonbondedForce is evaluated twice, once per end state, as Amber computes the reciprocal sum
    twice.
    """
    from md_tools.build.record import read_record

    measured = {}
    for label, root, solvent in (("implicit", built, "implicit"),
                                 ("explicit", built_explicit, "explicit")):
        work = tmp_path / f"cost-{label}"
        work.mkdir()
        argv = _ais_system(root, work)

        import mdtraj

        frames = mdtraj.load(str(root / "build" / "built.pdb"))
        mdtraj.join([frames] * 8).save_dcd(str(work / "source.dcd"))
        (work / "AIS.config").write_text(yaml.safe_dump({
            "protocol": "AIS", "solvent": solvent,
            "ais": {"number_of_paths": 1, "switching_steps": 20,
                    "observation_interval_steps": 5,
                    "parameter_update_interval_steps": 1},
            "ais_source": {"trajectory": "../source.dcd"},
            "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                          "checkpoint_printout": 20}}), encoding="utf-8")
        assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                     "--config", str(work / "AIS.config")],
                              capture_output=True, text=True, timeout=600).returncode == 0
        done = subprocess.run(
            [sys.executable, str(work / "project" / "AIS.py"), *argv,
             "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
            cwd=work, capture_output=True, text=True, timeout=3600,
            env=_environment(work, **_machine()))
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

        record = read_record(work / "run" / "AIS.log")
        assert record["acceleration"]["resolved_platform"] == "CUDA"
        counters = record["evaluation_counters"]
        evaluations = int(counters["paid_total_energy_evaluations"])
        derivatives = int(counters["work_derivative_evaluations"])
        pushes = int(counters["parameter_updates"])
        seconds = float(counters["parameter_change_seconds"])
        assert evaluations > 0 and derivatives > 0 and pushes > 0 and seconds >= 0.0
        measured[label] = (evaluations, derivatives, pushes, seconds)

    for label, (evaluations, derivatives, pushes, seconds) in measured.items():
        _record("test_the_two_state_switch_cost_is_measured_on_a_large_system",
                feature=f"two-state switch cost, AIS {label}, update interval 1 step",
                precision="mixed", device="-",
                detail=f"{evaluations} energy evaluations ({derivatives} dV/dlambda) and "
                       f"{pushes} lambda updates; {seconds:.6f} s in parameter changes "
                       f"({seconds / max(pushes, 1) * 1e6:.1f} us per update)")
    # Not a performance assertion -- hardware varies and a threshold here would be a flaky test
    # pretending to be a measurement. What is asserted is that the number EXISTS for both sizes,
    # which is what "measured and documented" requires.
    assert set(measured) == {"implicit", "explicit"}


# --- Hummer-Szabo frame alignment, on real CUDA ---------------------------------------------

@pytest.mark.parametrize("solvent", ["implicit", "explicit"])
def test_hs_rows_match_recomputation_on_cuda(solvent, built, built_explicit, hardware, tmp_path):
    """Reopen the frames a CUDA run wrote, rebuild a Context, recompute, compare.

    The alignment is platform-independent algebra, and it is verified on the CPU lane too. What
    this adds is that the CUDA run's own numbers -- written by CUDA kernels, at CUDA precision --
    survive the round trip through the file and still describe the frame they name. Explicit PME
    is the half that exercises the reciprocal sum and the dispersion correction.
    """
    import csv

    from .test_ais_hs_frame_alignment import frame_roundtrip_tolerance, recompute_at_frame

    root = built if solvent == "implicit" else built_explicit
    work = tmp_path / f"hs-cuda-{solvent}"
    work.mkdir()
    argv = _ais_system(root, work)

    import mdtraj

    frames = mdtraj.load(str(root / "build" / "built.pdb"))
    mdtraj.join([frames] * 6).save_dcd(str(work / "source.dcd"))

    # Five distinct cadences again, so frame-aligned and unaligned rows both occur on CUDA.
    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": solvent,
        "ais": {"number_of_paths": 1, "switching_steps": 60,
                "observation_interval_steps": 6, "parameter_update_interval_steps": 2},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"crd_printout_solute": 10, "info_printout": 20,
                      "checkpoint_printout": 15}}), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                 "--config", str(work / "AIS.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    done = subprocess.run(
        [sys.executable, str(work / "project" / "AIS.py"), *argv,
         "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run")],
        cwd=work, capture_output=True, text=True, timeout=3600,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    from md_tools.build.record import read_record

    record = read_record(work / "run" / "AIS.log")
    assert record["acceleration"]["resolved_platform"] == "CUDA", record["acceleration"]

    with (work / "run" / "AIS_hs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the CUDA run produced no frame-aligned HS rows"

    worst = 0.0
    names = ("potential_v0_kj_mol", "potential_v1_kj_mol", "potential_direct_kj_mol")
    for row in rows:
        # The helper reads `<work>/build/V0.xml` and `<work>/build/built.xml` -- the same layout
        # `_ais_system` copies here -- and evaluates each on the Reference platform, so the CUDA
        # numbers are checked by a Context that shares neither the platform nor the mixing force.
        recomputed = recompute_at_frame(
            work, work / "run", path_id=int(row["path_id"]),
            frame_index=int(row["coordinate_frame_index"]), lam=float(row["lambda"]))
        allowed = frame_roundtrip_tolerance(max(abs(float(row[name])) for name in names))
        for name, value in zip(names, recomputed):
            error = abs(float(row[name]) - value)
            worst = max(worst, error)
            assert error < allowed, (solvent, name, row["coordinate_frame_index"])
    counters = record["evaluation_counters"]
    _record("test_hs_rows_match_recomputation_on_cuda",
            feature=f"AIS {solvent}, HS rows recomputed at their own saved frames",
            precision=record["acceleration"].get("cuda_precision") or "mixed",
            device=record["acceleration"].get("cuda_device_index") or "-",
            detail=f"{len(rows)} rows, max |recorded - recomputed| = {worst:.3e} kJ/mol; "
                   + _counter_detail(counters))


# --- a hundred paths, and multi-rank ownership ------------------------------------------------

def test_a_hundred_paths_under_real_mpi_produce_exactly_their_own_files(built, hardware,
                                                                        tmp_path):
    """AIS_traj0000.nc .. AIS_traj0099.nc, no duplicates and none missing, across 4 ranks.

    Path identity is a pure function of `(rank, size, total)`, so which rank runs a path must not
    change which file it writes. Four ranks and a hundred paths is where an off-by-one in that
    partition shows up as two ranks claiming one id -- or as an id nobody claims, which is worse
    because the run still reports success.
    """
    import shutil

    if shutil.which("mpirun") is None:
        pytest.fail("no mpirun on PATH; the multi-rank AIS lane is an unmet criterion")
    if _visible() < 4:
        pytest.fail(f"4 ranks need 4 devices; {_visible()} visible")

    work = tmp_path / "hundred"
    work.mkdir()
    argv = _ais_system(built, work)

    import mdtraj

    frames = mdtraj.load(str(built / "build" / "built.pdb"))
    mdtraj.join([frames] * 128).save_dcd(str(work / "source.dcd"))

    (work / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 100, "switching_steps": 4,
                "observation_interval_steps": 2, "parameter_update_interval_steps": 2},
        "ais_source": {"trajectory": "../source.dcd", "allow_repeated_frames": True},
        "reporting": {"crd_printout_solute": 2, "info_printout": 4,
                      "checkpoint_printout": 4}}), encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                 "--config", str(work / "AIS.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    done = subprocess.run(
        ["mpirun", "-n", "4", sys.executable, str(work / "project" / "AIS.py"),
         *argv, "-source-traj", str(work / "source.dcd"), "-odir", str(work / "run"), "-ng", "4"],
        cwd=work, capture_output=True, text=True, timeout=7200,
        env=_environment(work, **_machine()))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    produced = sorted(p.name for p in (work / "run").glob("AIS_traj*.nc"))
    expected = [f"AIS_traj{index:04d}.nc" for index in range(100)]
    assert produced == expected, (
        f"{len(produced)} trajectories; missing "
        f"{sorted(set(expected) - set(produced))[:5]}, unexpected "
        f"{sorted(set(produced) - set(expected))[:5]}")

    import csv

    with (work / "run" / "AIS_paths.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 100
    assert sorted(int(row["path_index"]) for row in summary) == list(range(100))
    ranks = {row["mpi_rank"] for row in summary}
    assert len(ranks) == 4, f"paths were not shared out: {ranks}"
    _record("test_a_hundred_paths_under_real_mpi_produce_exactly_their_own_files",
            feature="AIS implicit, 100 paths, real mpirun -n 4, deterministic path ownership",
            precision="mixed", device="0-3",
            detail=f"AIS_traj0000.nc..AIS_traj0099.nc, no duplicates or gaps; "
                   f"{len(ranks)} ranks contributed")


# --- direct wrapper against md-run ------------------------------------------------------------

def test_the_generated_wrapper_and_md_run_agree_for_a_stage(built, hardware, tmp_path):
    """Two entry points into one runtime must produce the same science, on the same GPU.

    `md-openmm md-run` and the generated `cMD.py` call `stage_main` with different argv. A
    difference in what they forward is invisible until the two produce different trajectories --
    which is exactly how `--resume` came to be dropped by one of them.
    """
    work = tmp_path / "parity"
    work.mkdir()
    (work / "cMD.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "implicit",
        "stages": {"minimization_iterations": 2, "restrained_nvt_steps": 5,
                   "production_steps": 20},
        "reporting": {"crd_printout_solute": 10, "info_printout": 10,
                      "checkpoint_printout": 20}}), encoding="utf-8")
    _seed_build(built, work)
    assert subprocess.run(CLI + ["build-md", "-odir", str(work / "project"),
                                 "--config", str(work / "cMD.config")],
                          capture_output=True, text=True, timeout=600).returncode == 0

    environment = _environment(work, **_machine())
    wrapper = subprocess.run(
        [sys.executable, str(work / "project" / "cMD.py"),
         "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"),
         "-odir", str(work / "wrapper")],
        cwd=work, capture_output=True, text=True, timeout=1800, env=environment)
    assert wrapper.returncode == 0, wrapper.stdout[-3000:] + wrapper.stderr[-3000:]

    through_md_run = subprocess.run(
        CLI + ["md-run", "-i", str(work / "input" / "cMD.in"),
               "-p", str(built / "build" / "built.pdb"), "-s", str(built / "build" / "built.xml"),
               "-odir", str(work / "mdrun")],
        cwd=work, capture_output=True, text=True, timeout=1800, env=environment)
    assert through_md_run.returncode == 0, (
        through_md_run.stdout[-3000:] + through_md_run.stderr[-3000:])

    import hashlib

    import numpy

    from md_tools.build.record import read_record

    # The restart is compared byte for byte: it is a serialised State with no metadata of its own.
    wrapper_digest = hashlib.sha256((work / "wrapper" / "cMD.xml").read_bytes()).hexdigest()
    md_run_digest = hashlib.sha256((work / "mdrun" / "cMD.xml").read_bytes()).hexdigest()
    assert wrapper_digest == md_run_digest, (
        "the final state differs between the generated wrapper and md-run: the two entry points "
        "do not forward the same settings")

    # Compared FRAME BY FRAME, not byte by byte. A trajectory header carries a creation
    # timestamp -- the two runs differ in exactly one byte of it, the wall-clock second they
    # started -- so a digest comparison would be asserting that two processes started in the same
    # second. What has to be identical is the science, and that is the coordinates.
    import mdtraj

    # The SOLUTE topology, because the stream being compared is the solute one. A 22-atom
    # trajectory cannot be opened against the 1796-atom built.pdb, and mdtraj says so rather
    # than silently mismatching.
    solute_top = str(built / "build" / "built.pdb")   # `-x cMD.dcd` writes the WHOLE system, as it always did
    left = mdtraj.load(str(work / "wrapper" / "solute_prod1.nc"), top=solute_top)
    right = mdtraj.load(str(work / "mdrun" / "solute_prod1.nc"), top=solute_top)
    assert left.n_frames == right.n_frames > 0, (left.n_frames, right.n_frames)
    assert numpy.array_equal(left.xyz, right.xyz), (
        "the trajectories differ between the generated wrapper and md-run: "
        f"max |dx| = {float(numpy.abs(left.xyz - right.xyz).max())}")
    both = [read_record(work / where / "cMD.log")["acceleration"]["resolved_platform"]
            for where in ("wrapper", "mdrun")]
    assert both == ["CUDA", "CUDA"], both
    _record("test_the_generated_wrapper_and_md_run_agree_for_a_stage",
            feature="direct wrapper vs md-run parity, cMD implicit",
            precision="mixed", device="-",
            detail="cMD.xml identical byte for byte; DCD identical frame for frame "
                   "(its header carries a creation timestamp)")


# --- the evidence document ------------------------------------------------------------------------

def test_write_the_coverage_evidence(hardware, request):
    """Emit the matrix, with what actually ran, so the published table is a record of a run.

    LAST IN THE FILE, deliberately. pytest runs a module in definition order, and `RESULTS` is
    filled by the lanes as they pass -- so a lane defined below this function would run after it
    and be missing from the document, which is how a matrix comes to describe less than was
    actually verified. It writes the document only
    when `--cuda-evidence=<path>` is given, so an ordinary GPU run does not rewrite a committed
    file as a side effect.
    """
    destination = request.config.getoption("--cuda-evidence", default=None)
    if not destination:
        pytest.skip("no --cuda-evidence=<path> given; the lanes above ran, nothing to write")

    import subprocess

    from openmm import version as openmm_version

    # THE EXACT COMMIT. A coverage table with no commit on it is a claim about "the code" and
    # cannot be checked against anything -- it stays true-looking across every change that
    # invalidates it. `-dirty` is the honest answer when the tree that ran differs from any
    # commit, and it means the table describes something not published anywhere.
    def git(*arguments, default="unknown"):
        try:
            return subprocess.run(("git", *arguments), cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True, timeout=30,
                                  check=True).stdout.strip() or default
        except (OSError, subprocess.SubprocessError):
            return default

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", default=""))

    lines = ["# CUDA coverage matrix", "",
             "Generated by `tests/test_cuda_coverage_matrix.py`. Every row is a real run on the "
             "hardware named below.", "",
             f"* commit: `{commit}`" + ("  **-dirty: the tree that ran differs from this commit, "
                                        "so this table describes code that is not published**"
                                        if dirty else ""),
             f"* generated by: `{git('rev-parse', '--abbrev-ref', 'HEAD')}`",
             "", "## Hardware", ""]
    lines += [f"* device {device['index']}: {device['name']}, driver {device['driver']}, "
              f"{device['memory']}" for device in hardware]
    lines += ["", f"* OpenMM {openmm_version.full_version}", ""]
    lines += ["## Source sites", "",
              "| source function or branch | what runs on CUDA | lane |", "|---|---|---|"]
    for site, (what, lane) in sorted(CUDA_SITES.items()):
        lines.append(f"| `{site}` | {what} | {lane} |")
    # The COUNTS, so the table can be checked for completeness rather than read for reassurance.
    found = _cuda_operation_sites()
    by_kind: dict[str, int] = {}
    for kinds in found.values():
        for kind in kinds:
            by_kind[kind] = by_kind.get(kind, 0) + 1
    # Counted against what the matcher ACTUALLY FOUND, not against the length of the tables. The
    # two differ -- a table also documents functions the operation matcher does not flag, such as
    # platform resolution, which builds a Platform rather than touching a Context -- and adding
    # the table lengths together produced a total larger than the number of sites there are.
    exercised = sorted(set(found) & set(CUDA_SITES))
    not_a_device = sorted(set(found) & set(NON_CUDA_CONTEXT_SITES))
    also_documented = sorted(set(CUDA_SITES) - set(found))
    lines += ["", "### Coverage counts", "",
              f"* {len(found)} source functions perform a CUDA-relevant operation, by the "
              f"operation matcher in `CUDA_OPERATIONS`",
              f"* of those, {len(exercised)} are exercised by a lane below and "
              f"{len(not_a_device)} are classified as not reaching a device, with the reason "
              f"recorded in the test",
              "* 0 unclassified -- `test_every_cuda_operation_in_the_source_is_in_this_matrix` "
              "fails the suite if that is ever not true",
              f"* {len(also_documented)} further function(s) carry a lane without being flagged "
              f"by the matcher (they resolve a Platform rather than touch a Context): "
              + ", ".join(f"`{site}`" for site in also_documented),
              "* by operation, counting a function once per operation kind it performs: "
              + ", ".join(f"{kind} {count}" for kind, count in sorted(by_kind.items())),
              ""]
    lines += ["", "## Lanes executed in this run", "",
              "| lane | feature | precision | device | result | detail |", "|---|---|---|---|---|---|"]
    for entry in RESULTS:
        lines.append(f"| `{entry['lane']}` | {entry['feature']} | {entry['precision']} | "
                     f"{entry['device']} | {entry['result']} | {entry['detail']} |")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert Path(destination).is_file()

    # THE MACHINE-READABLE ARTEFACT, beside the prose. A table is for a person deciding whether to
    # trust the run; this is for anything that has to compare two runs, or check a claim in a
    # report against what the lanes actually did, without parsing markdown.
    import datetime
    import platform

    evidence = {
        "commit": commit,
        "dirty": dirty,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "openmm": openmm_version.full_version,
        "driver": hardware[0]["driver"] if hardware else None,
        "devices": hardware,
        "source_sites": {
            "total_performing_a_cuda_operation": len(found),
            "exercised_by_a_lane": len(exercised),
            "classified_as_not_reaching_a_device": len(not_a_device),
            "unclassified": 0,
            "documented_with_a_lane_but_not_flagged_by_the_matcher": also_documented,
            "by_operation": by_kind,
            "sites": {site: sorted(kinds) for site, kinds in sorted(found.items())},
            "lanes_for_site": {site: lane for site, (_what, lane) in sorted(CUDA_SITES.items())},
        },
        "lanes": RESULTS,
    }
    machine = Path(destination).with_suffix(".json")
    machine.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert machine.is_file()
