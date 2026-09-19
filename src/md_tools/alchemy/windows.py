"""Fixed-lambda window execution: cross-state energies and derivatives, and the window runner.

A WINDOW samples ONE thermodynamic state of an alchemical path and, at every report, evaluates
the sampled configuration at EVERY state of the path and the complete derivative at its own. That
is the sample record `md_tools.alchemy.samples` defines, written as a CSV stream one window at a
time, and it is all FEP, BAR, MBAR and TI need.

THE HAMILTONIAN IS DUCK-TYPED

    Anything with

        system                      the openmm.System integrated and evaluated
        parameter_names             the Context parameters a state sets, e.g. lambda_sterics
        set_state(context, state)   set every Context parameter for `state` (a {name: float})
        derivatives(context, state) complete dU/d lambda_k at the Context's coordinates, kJ/mol
        context_parameters(state)   every Context parameter value for `state`, for the record

    S3's `md_tools.alchemy.hamiltonian.AlchemicalHamiltonian` is one. `ParametricHamiltonian` below
    is another, for Systems whose lambda dependence lives only in custom forces that declare their
    parameter derivatives -- analytic test models, and the Boresch restraint. This module never
    calls `context.setParameter` on a Hamiltonian's names: S3's Hamiltonian keeps derived internal
    parameters that must move together with the public ones, and setting a public name alone
    leaves it inconsistent with nothing failing. `lambda_restraints` is the one parameter this
    layer owns (`ComposedHamiltonian`).

CROSS-STATE ENERGIES ON A SEPARATE CONTEXT

    The sampling Context is never re-parameterised between integrator steps. Positions and box
    are copied to an EVALUATION Context built from the same serialised System, and every state is
    evaluated there. Changing parameters on the sampling Context and changing them back would be
    exact in principle, and is a way to make the sampling depend on the reporting in practice. The
    first report checks the evaluation Context's own-state energy against the sampling Context's,
    so an evaluation Context that is not the same Hamiltonian is refused, not reported.

WHAT THE RUNNER REUSES, AND DOES NOT RE-IMPLEMENT

    `md_tools.run.preflight.preflight_stage`: flags, input files, collisions, the machine
        configuration, the platform policy and a real Context on it, rank placement, the refusal
        of a plural launch for this serial protocol, timestep/HMR validation, ensemble checks.
    `md_tools.run.preflight.check_output_collisions` / `check_existing_outputs` for the files the
        stage inventory does not know about (the sample stream, the window record, completion).
    `md_tools.openmm.checkpoint.commit_generation` / `read_committed`: every checkpoint is a
        generation transaction binding the fingerprint, the step and the sample stream's
        committed prefix (row count AND digest, `md_tools.cv.prefix.prefix_digest`).
    `md_tools.openmm.seeds` for the integrator and velocity seeds.

    A refused continuation is read-only: the committed checkpoint, the prefix and a completion
    manifest are validated before the output directory is touched.

    The window runner is serial, like a cMD stage: windows run as independent processes, and a
    sample's provenance is its window's STATE, never which process or GPU ran it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from md_tools.alchemy.paths import AlchemicalPath
from md_tools.alchemy.restraints import RESTRAINT_PARAMETER, BoreschRestraint
from md_tools.alchemy.samples import (BAR_NM3_TO_KJ_MOL, SampleSet, ThermodynamicState,
                                      kt_kj_mol)

WINDOW_SCHEMA = "md-tools-alchemical-window/1"
COMPLETION_SCHEMA = "md-tools-alchemical-window-completion/1"
#: The force group of the Boresch restraint. S3's Hamiltonian uses groups 0-9.
RESTRAINT_FORCE_GROUP = 16
#: The evaluation Context's own-state energy must equal the sampling Context's to
#: SELF_CHECK_ABS_KJ_MOL + rel * |E|, with `rel` by the precision the Contexts compute in.
#:
#: CALIBRATED ON MEASUREMENT, AND CHANGED AFTER A FAILURE -- stated so nobody mistakes it for a
#: tolerance loosened to pass. The first value, 1e-6 relative for every precision, was set before
#: any comparison but not against any measurement. On CUDA mixed precision (card 4, 2026-09-19,
#: 1038-atom TIP3P/PME) two Contexts over the same System and positions differ by median 1.5e-3,
#: max 2.0e-2 kJ/mol (max 1.8e-6 relative) over 200 frames, and ONE Context re-evaluated at the
#: same positions moves by 1.1e-3 -- summation order after atom reordering, not a Hamiltonian
#: difference. 1.5 % of reports exceeded the old bound and the G1 NPT lane failed on one. In double
#: precision the same comparisons agree to 1e-10, so double keeps 1e-6. What this check exists to
#: catch -- an evaluation Context left in the wrong state or built from another System -- is
#: kJ/mol or more; the System identity itself is enforced by serialisation equality, not here.
SELF_CHECK_ABS_KJ_MOL = 1e-3
SELF_CHECK_REL = {"double": 1e-6, "mixed": 2e-5, "single": 2e-5}


def self_check_relative(platform_name: str, properties: Mapping[str, str] | None) -> float:
    """The relative self-check tolerance for the precision a platform computes energies in."""
    if platform_name == "Reference":
        return SELF_CHECK_REL["double"]
    precision = str((properties or {}).get("Precision", "")).lower()
    if platform_name in ("CUDA", "OpenCL", "HIP"):
        # An unknown or absent precision is REFUSED, not mapped to the loosest bound: a typo would
        # otherwise widen the check quietly. platform_policy always sets Precision on a GPU.
        if precision not in SELF_CHECK_REL:
            raise WindowError(f"{platform_name} Precision {precision or '(absent)'!r} is not one of "
                              f"{sorted(SELF_CHECK_REL)}; the self-check bound depends on it")
        return SELF_CHECK_REL[precision]
    # the CPU platform computes forces and energies in single precision
    return SELF_CHECK_REL["single"]


class WindowError(ValueError):
    pass


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _write_atomically(path: Path, text: str) -> None:
    from md_tools.openmm.checkpoint import write_durably

    staging = path.with_name(path.name + ".partial")
    write_durably(staging, text.encode())
    os.replace(staging, path)


# ---------------------------------------------------------------------- Hamiltonians
class ParametricHamiltonian:
    """A System whose lambda dependence is ONLY in custom forces with declared derivatives.

    Construction proves the claim: every named parameter must be a global parameter of at least
    one force and every force that declares it must request its energy derivative; a
    `NonbondedForce` parameter offset on any name is refused, because OpenMM's parameter
    derivative does not cover offsets and the derivative would be silently incomplete.
    """

    def __init__(self, system, parameter_names: Sequence[str]):
        import openmm

        self.system = system
        self.parameter_names = tuple(parameter_names)
        declared = {n: 0 for n in self.parameter_names}
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                offsets = {force.getParticleParameterOffset(i)[0]
                           for i in range(force.getNumParticleParameterOffsets())}
                offsets |= {force.getExceptionParameterOffset(i)[0]
                            for i in range(force.getNumExceptionParameterOffsets())}
                bad = sorted(offsets & set(self.parameter_names))
                if bad:
                    raise WindowError(f"NonbondedForce parameter offsets on {bad}: their "
                                      f"derivative is not an OpenMM parameter derivative, so a "
                                      f"ParametricHamiltonian cannot claim it is complete")
            if not hasattr(force, "getNumGlobalParameters"):
                continue
            names = {force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())}
            wanted = {force.getEnergyParameterDerivativeName(i)
                      for i in range(force.getNumEnergyParameterDerivatives())} \
                if hasattr(force, "getNumEnergyParameterDerivatives") else set()
            for n in names & set(self.parameter_names):
                if n not in wanted:
                    raise WindowError(f"{type(force).__name__} uses {n} without requesting its "
                                      f"energy derivative; dU/d{n} would be incomplete")
                declared[n] += 1
        missing = [n for n, c in declared.items() if c == 0]
        if missing:
            raise WindowError(f"no force in the System uses {missing}")

    def context_parameters(self, state: Mapping[str, float]) -> dict[str, float]:
        self._check(state)
        return {n: float(state[n]) for n in self.parameter_names}

    def set_state(self, context, state: Mapping[str, float]) -> None:
        for n, v in self.context_parameters(state).items():
            context.setParameter(n, v)

    def derivatives(self, context, state: Mapping[str, float]) -> dict[str, float]:
        self.set_state(context, state)
        d = context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
        return {n: float(d[n]) for n in self.parameter_names}

    def _check(self, state):
        if set(state) != set(self.parameter_names):
            raise WindowError(f"a state sets exactly {sorted(self.parameter_names)}; got "
                              f"{sorted(state)}")


class ComposedHamiltonian:
    """An alchemical Hamiltonian plus a Boresch restraint scaled by `lambda_restraints`.

    The restraint force lives in `RESTRAINT_FORCE_GROUP`; its derivative is read from that group
    only. The inner Hamiltonian is always handed a state holding ITS names and nothing else.
    """

    def __init__(self, inner, restraint: BoreschRestraint):
        from openmm import XmlSerializer

        if RESTRAINT_PARAMETER in inner.parameter_names:
            raise WindowError(f"the inner Hamiltonian already owns {RESTRAINT_PARAMETER}")
        used = {f.getForceGroup() for f in inner.system.getForces()}
        if RESTRAINT_FORCE_GROUP in used:
            raise WindowError(f"force group {RESTRAINT_FORCE_GROUP} is taken in the inner System")
        self.inner = inner
        self.restraint = restraint
        system = XmlSerializer.deserialize(XmlSerializer.serialize(inner.system))
        force = restraint.openmm_force(periodic=system.usesPeriodicBoundaryConditions())
        force.setForceGroup(RESTRAINT_FORCE_GROUP)
        system.addForce(force)
        self.system = system
        self.parameter_names = tuple(inner.parameter_names) + (RESTRAINT_PARAMETER,)

    def _split(self, state):
        if set(state) != set(self.parameter_names):
            raise WindowError(f"a state sets exactly {sorted(self.parameter_names)}; got "
                              f"{sorted(state)}")
        return ({k: v for k, v in state.items() if k != RESTRAINT_PARAMETER},
                float(state[RESTRAINT_PARAMETER]))

    def context_parameters(self, state):
        inner, lam = self._split(state)
        return {**self.inner.context_parameters(inner), RESTRAINT_PARAMETER: lam}

    def set_state(self, context, state):
        inner, lam = self._split(state)
        self.inner.set_state(context, inner)
        context.setParameter(RESTRAINT_PARAMETER, lam)

    def derivatives(self, context, state):
        inner, lam = self._split(state)
        out = dict(self.inner.derivatives(context, inner))
        context.setParameter(RESTRAINT_PARAMETER, lam)
        d = context.getState(getParameterDerivatives=True,
                             groups={RESTRAINT_FORCE_GROUP}).getEnergyParameterDerivatives()
        out[RESTRAINT_PARAMETER] = float(d[RESTRAINT_PARAMETER])
        return out


def check_hamiltonian_matches_path(hamiltonian, path: AlchemicalPath) -> None:
    """The path moves exactly the Hamiltonian's parameters: no more (a component nothing reads),
    no fewer (a parameter left at whatever value the Context happened to hold)."""
    have, want = set(hamiltonian.parameter_names), set(path.components)
    if have != want:
        raise WindowError(
            f"the path moves {sorted(want)} and the Hamiltonian's parameters are {sorted(have)}. "
            f"Missing from the path: {sorted(have - want)}; unknown to the Hamiltonian: "
            f"{sorted(want - have)}")


# ---------------------------------------------------------------------- evaluation
class CrossStateEvaluator:
    """Evaluates one configuration at every state, and the derivative at its origin state."""

    def __init__(self, hamiltonian, states: Sequence[ThermodynamicState], *, platform,
                 properties: Mapping[str, str] | None = None):
        import openmm

        self.hamiltonian = hamiltonian
        self.states = tuple(states)
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(hamiltonian.system))
        self.periodic = system.usesPeriodicBoundaryConditions()
        self.self_check_rel = self_check_relative(platform.getName(), properties)
        self.context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform,
                                      dict(properties or {}))

    def evaluate(self, positions, box_vectors, origin: ThermodynamicState):
        ctx = self.context
        if self.periodic:
            ctx.setPeriodicBoxVectors(*box_vectors)
        ctx.setPositions(positions)
        energies = []
        for st in self.states:
            self.hamiltonian.set_state(ctx, st.components)
            energies.append(ctx.getState(getEnergy=True).getPotentialEnergy()._value)
        derivatives = self.hamiltonian.derivatives(ctx, origin.components)
        missing = sorted(set(origin.components) - set(derivatives))
        if missing:
            raise WindowError(f"the Hamiltonian returned no derivative for {missing}")
        return np.asarray(energies), {k: float(derivatives[k]) for k in origin.components}


# ---------------------------------------------------------------------- the sample stream
def stream_columns(states: Sequence[ThermodynamicState], components: Sequence[str],
                   npt: bool) -> list[str]:
    cols = ["sample_id", "origin_state", "step", "time_ps"]
    if npt:
        cols.append("volume_nm3")
    cols += [f"U_{st.state_id}_kJ_mol" for st in states]
    cols += [f"dU_d{c}_kJ_mol" for c in components]
    return cols


class SampleStreamReporter:
    """An OpenMM reporter appending one row per report to a window's sample CSV.

    Reports at absolute steps that are multiples of `interval` and >= `first_step`, using
    `simulation.currentStep`, the only absolute-step authority after a checkpoint restore.
    """

    def __init__(self, path: Path, evaluator: CrossStateEvaluator, origin: ThermodynamicState,
                 *, interval: int, first_step: int, timestep_ps: float, components: Sequence[str],
                 rows_already: int = 0):
        self.path = Path(path)
        self.evaluator = evaluator
        self.origin = origin
        self.interval = int(interval)
        self.first_step = int(first_step)
        self.timestep_ps = float(timestep_ps)
        self.components = tuple(components)
        self.npt = origin.pressure_bar is not None
        self.columns = stream_columns(evaluator.states, self.components, self.npt)
        self.rows = int(rows_already)
        self.checked_self_energy: float | None = None
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if fresh:
            self._writer.writerow(self.columns)
            self._handle.flush()

    def describeNextReport(self, simulation):                 # noqa: N802 - OpenMM's protocol
        step = simulation.currentStep
        return (self.interval - step % self.interval, True, False, False, True, None)

    def _reported(self, step):
        return getattr(self, "_last_step", None) == step

    def report(self, simulation, state):
        step = simulation.currentStep
        if step < self.first_step or step % self.interval or self._reported(step):
            return
        positions = state.getPositions(asNumpy=True)
        box = state.getPeriodicBoxVectors()
        energies, ders = self.evaluator.evaluate(positions, box, self.origin)
        own = self.evaluator.states.index(self.origin)
        sampled = state.getPotentialEnergy()._value
        rel = self.evaluator.self_check_rel
        if abs(energies[own] - sampled) > SELF_CHECK_ABS_KJ_MOL + rel * abs(sampled):
            raise WindowError(
                f"the evaluation Context gives {energies[own]:.6f} kJ/mol at the sampled state "
                f"and the sampling Context {sampled:.6f}: they are not the same Hamiltonian, and "
                f"every cross-state energy would describe another system")
        self.checked_self_energy = max(abs(float(energies[own] - sampled)),
                                       self.checked_self_energy or 0.0)
        row = [f"{self.origin.state_id}:{step:012d}", self.origin.state_id, step,
               repr(step * self.timestep_ps)]
        if self.npt:
            a, b, c = (np.asarray(v._value) for v in box)
            row.append(repr(float(np.dot(a, np.cross(b, c)))))
        row += [repr(float(e)) for e in energies]
        row += [repr(ders[c]) for c in self.components]
        self._writer.writerow(row)
        self._handle.flush()
        self.rows += 1
        self._last_step = step

    def close(self):
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def validate_stream_prefix(path: Path, *, rows: int, prefix_sha256: str, columns: list[str],
                           origin: str, first_step: int, interval: int) -> None:
    """Everything a continuation needs from the committed rows, checked BEFORE any truncation."""
    from md_tools.cv.prefix import CVPrefixError, prefix_digest, read_lines

    try:
        digest = prefix_digest(path, rows)
    except CVPrefixError as exc:
        raise WindowError(f"sample stream {path.name}: {exc}") from None
    if digest != prefix_sha256:
        raise WindowError(
            f"sample stream {path.name}: the {rows} committed row(s) are not the rows the "
            f"checkpoint committed (digest {digest[:12]}... != {prefix_sha256[:12]}...). A "
            f"continuation would append to edited data")
    lines = read_lines(path)
    header = next(csv.reader([lines[0]]))
    if header != columns:
        raise WindowError(f"sample stream {path.name}: columns {header} != {columns}")
    for i, raw in enumerate(csv.reader(lines[1:1 + rows])):
        if raw[1] != origin:
            raise WindowError(f"sample stream {path.name} row {i}: origin {raw[1]} != {origin}")
        if int(raw[2]) != first_step + i * interval:
            raise WindowError(f"sample stream {path.name} row {i}: step {raw[2]} off the grid "
                              f"{first_step} + k*{interval}")
        if not all(math.isfinite(float(v)) for v in raw[3:]):
            raise WindowError(f"sample stream {path.name} row {i}: non-finite value")


def read_window_samples(csv_path: Path, record: Mapping[str, Any]) -> SampleSet:
    """A window's stream as a SampleSet, using the states and path of its window record."""
    path = AlchemicalPath.from_record(record["path"])
    states = tuple(ThermodynamicState.from_record(s) for s in record["states"])
    npt = states[0].pressure_bar is not None
    comps = list(path.components)
    cols = stream_columns(states, comps, npt)
    with Path(csv_path).open(encoding="utf-8", newline="") as h:
        reader = csv.reader(h)
        header = next(reader)
        if header != cols:
            raise WindowError(f"{csv_path}: columns {header} != {cols}")
        rows = list(reader)
    rows = rows[: int(record.get("rows", len(rows)))] if "rows" in record else rows
    k = len(states)
    off = 5 if npt else 4
    arr = np.array([[float(v) for v in r[3:]] for r in rows]) if rows else np.zeros((0, off - 3 + k + len(comps)))
    return SampleSet(
        states, path, [r[0] for r in rows], [r[1] for r in rows], [int(r[2]) for r in rows],
        arr[:, off - 3: off - 3 + k],
        arr[:, 1] if npt else None,
        {c: arr[:, off - 3 + k + j] for j, c in enumerate(comps)},
        provenance={"window": record.get("window_id"), "stream": str(csv_path),
                    "fingerprint": record.get("fingerprint")})


# ---------------------------------------------------------------------- the runner
@dataclass(frozen=True)
class WindowSettings:
    steps: int
    report_interval: int
    checkpoint_interval: int
    equilibration_steps: int = 0
    timestep_fs: Any = 2.0
    friction_per_ps: float = 1.0
    barostat_interval: int = 25
    seed: int = 20260919
    #: LocalEnergyMinimizer iterations at the window's own state before velocities are drawn, on
    #: a fresh start only (0 = none). Part of the window's identity.
    minimize_iterations: int = 0

    def __post_init__(self):
        for name in ("steps", "report_interval", "checkpoint_interval"):
            if int(getattr(self, name)) <= 0:
                raise WindowError(f"{name} must be a positive integer step count")
        if self.equilibration_steps < 0:
            raise WindowError("equilibration_steps must be >= 0")
        if self.equilibration_steps % self.report_interval:
            raise WindowError(
                f"equilibration_steps {self.equilibration_steps} is not a multiple of "
                f"report_interval {self.report_interval}: the sample grid would not start on it")
        if self.steps % self.report_interval or self.steps % self.checkpoint_interval:
            raise WindowError(
                f"steps {self.steps} must be a multiple of report_interval "
                f"{self.report_interval} and of checkpoint_interval {self.checkpoint_interval}; "
                f"a schedule that would have to be rounded is refused")
        if self.checkpoint_interval % self.report_interval:
            raise WindowError(f"checkpoint_interval {self.checkpoint_interval} must be a multiple "
                              f"of report_interval {self.report_interval}, so a committed "
                              f"generation never splits a report")


def window_paths(out_dir: Path, window_id: str) -> dict[str, Path]:
    d = Path(out_dir)
    # `record` IS the `.log`: the machine-readable provenance of the window. `.out` is for a person.
    return {"out": d / f"{window_id}.out",
            "restart": d / f"{window_id}.xml", "checkpoint": d / f"{window_id}.chk",
            "samples": d / f"{window_id}.samples.csv", "record": d / f"{window_id}.log",
            "completion": d / f"{window_id}.complete.json"}


def _checkpoint_dir(paths):
    chk = paths["checkpoint"]
    return chk.parent / f"{chk.stem}.checkpoints"


def window_identity(*, hamiltonian, path: AlchemicalPath, states, window_id, settings,
                    system_sha256, timestep, restraint=None) -> dict[str, Any]:
    ids = [s.state_id for s in states]
    if window_id not in ids:
        raise WindowError(f"window {window_id!r} is not one of the states {ids}")
    doc = {
        "schema": WINDOW_SCHEMA, "window_id": window_id, "path": path.to_record(),
        "path_digest": path.digest(), "states": [s.to_record() for s in states],
        "context_parameters": {s.state_id: hamiltonian.context_parameters(s.components)
                               for s in states},
        "system_sha256": system_sha256, "settings": {
            "steps": settings.steps, "report_interval": settings.report_interval,
            "checkpoint_interval": settings.checkpoint_interval,
            "equilibration_steps": settings.equilibration_steps,
            "friction_per_ps": settings.friction_per_ps,
            "barostat_interval": settings.barostat_interval, "seed": settings.seed,
            "minimize_iterations": settings.minimize_iterations},
        "timestep_fs": timestep["timestep_fs"],
        "restraint": restraint.to_record() if restraint is not None else None,
        "sample_provenance": "origin_state = window_id's state; never the rank or device",
    }
    doc["fingerprint"] = _sha256_bytes(_canonical(doc))
    return doc


def run_window(*, topology, system, hamiltonian, path: AlchemicalPath,
               states: Sequence[ThermodynamicState], window_id: str, out_dir,
               settings: WindowSettings, coordinates=None, cpu: bool = False, device=None,
               machine_config=None, overwrite: bool = False, restraint=None,
               stop_after_steps: int | None = None, prepared=None,
               check: bool = False) -> dict[str, Any]:
    """Run (or continue, or verify) one fixed-lambda window. Returns a summary.

    `system` is the serialised `hamiltonian.system` on disk; its sha256 must be the Hamiltonian's,
    so the file the preflight read and the System that is integrated are one System.
    `stop_after_steps` interrupts after that many production steps (tests of resume).
    `prepared` is a `StagePreflight` the caller has already obtained for exactly these paths; it is
    consumed rather than re-planned, as `stage_main` does. `check` validates everything,
    continuation included, prints the preflight report and creates nothing.
    """
    import openmm
    from openmm import unit

    from md_tools.openmm.checkpoint import commit_generation, read_committed
    from md_tools.openmm.seeds import as_openmm_seed, derive_build_seed
    from md_tools.run.preflight import (OutputInventory, check_existing_outputs,
                                        check_output_collisions, preflight_stage)

    out_dir = Path(out_dir)
    paths = window_paths(out_dir, window_id)
    check_hamiltonian_matches_path(hamiltonian, path)
    origin = next((s for s in states if s.state_id == window_id), None)
    if origin is None:
        raise WindowError(f"window {window_id!r} is not one of the states")
    npt = origin.pressure_bar is not None

    # ---- the shared preflight: everything before anything is written ----------------------
    checked = prepared if prepared is not None else preflight_stage(
        topology=topology, system=system, coordinates=coordinates, output=paths["out"],
        log=paths["record"], restart=paths["restart"], checkpoint=paths["checkpoint"], cpu=cpu,
        device=device, machine_config=machine_config, protocol=f"alchemical window {window_id}",
        timestep_fs=settings.timestep_fs, ensemble="NPT" if npt else "NVT")
    extra = {"samples": paths["samples"], "completion": paths["completion"]}
    check_output_collisions(outputs={**checked.inventory.roles, **extra},
                            inputs={"p": topology, "s": system,
                                    **({"c": coordinates} if coordinates else {})})
    system_sha = hashlib.sha256(Path(system).read_bytes()).hexdigest()
    serialised = openmm.XmlSerializer.serialize(hamiltonian.system).encode()
    if checked.loaded is None or \
            openmm.XmlSerializer.serialize(checked.loaded.system).encode() != serialised:
        raise WindowError(f"-s {system} is not the Hamiltonian's System; the file the preflight "
                          f"checked and the System integrated would differ")
    if checked.loaded.barostats:
        raise WindowError("the -s System already holds a barostat; the window adds its own "
                          "when the states are NPT, and two would both move the volume")
    identity = window_identity(hamiltonian=hamiltonian, path=path, states=states,
                               window_id=window_id, settings=settings, system_sha256=system_sha,
                               timestep=checked.timestep, restraint=restraint)
    fingerprint = identity["fingerprint"]
    total_steps = settings.equilibration_steps + settings.steps
    columns = stream_columns(states, path.components, npt)

    # ---- continuation: validated read-only -------------------------------------------------
    chk_dir = _checkpoint_dir(paths)
    committed = None
    if not overwrite:
        if paths["completion"].is_file():
            problems = verify_window(paths, fingerprint)
            if problems:
                raise WindowError(f"window {window_id} claims completion and does not verify: "
                                  f"{problems}. Refusing; rerun with overwrite to replace it")
            return {"window_id": window_id, "disposition": "verified-complete",
                    "rows": json.loads(paths["completion"].read_text())["rows"]}
        committed = read_committed(chk_dir) if chk_dir.is_dir() else None
        if committed is not None:
            st = committed["state"]
            if st.get("fingerprint") != fingerprint:
                raise WindowError(
                    f"window {window_id}: the committed checkpoint belongs to a different "
                    f"window definition (fingerprint {str(st.get('fingerprint'))[:12]}... != "
                    f"{fingerprint[:12]}...). Continuing would splice two experiments")
            validate_stream_prefix(paths["samples"], rows=st["rows"],
                                   prefix_sha256=st["prefix_sha256"], columns=columns,
                                   origin=window_id,
                                   first_step=settings.equilibration_steps,
                                   interval=settings.report_interval)
        else:
            inv = OutputInventory(roles={**checked.inventory.roles, **extra},
                                  resumable=checked.inventory.resumable)
            check_existing_outputs(inv, where=f"alchemical window {window_id}")

    if check:
        from md_tools.run.preflight import report_check
        report_check(checked, what=f"alchemical window {window_id}",
                     extra=[("states", len(states)), ("steps", total_steps),
                            ("continuation", "committed checkpoint" if committed else "none")])
        return {"window_id": window_id, "disposition": "checked"}

    # ---- from here on the run may write ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _move_aside(paths, chk_dir)
    if committed is not None:
        from md_tools.cv.prefix import truncate
        truncate(paths["samples"], committed["state"]["rows"])
    record = dict(identity, acceleration=checked.record(), timestep=checked.timestep)
    _write_atomically(paths["record"], json.dumps(record, indent=1, sort_keys=True) + "\n")

    sampling_system = openmm.XmlSerializer.deserialize(serialised.decode())
    temperature = origin.temperature_k * unit.kelvin
    seed_i = as_openmm_seed(derive_build_seed(settings.seed, f"alchemy/{window_id}/integrator"))
    if npt:
        from md_tools.md._stages import add_barostat
        add_barostat(sampling_system, origin.pressure_bar, temperature,
                     as_openmm_seed(derive_build_seed(settings.seed,
                                                      f"alchemy/{window_id}/barostat")),
                     settings.barostat_interval)
    dt_ps = float(checked.timestep["timestep_fs"]) / 1000.0
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 settings.friction_per_ps / unit.picosecond,
                                                 dt_ps * unit.picoseconds)
    integrator.setRandomNumberSeed(seed_i)
    platform = checked.acceleration.platform
    properties = dict(checked.acceleration.properties)
    simulation = openmm.app.Simulation(checked.loaded.pdb.topology, sampling_system, integrator,
                                       platform, properties)
    context = simulation.context
    if committed is not None:
        simulation.loadCheckpoint(committed["checkpoint"])
    else:
        _set_initial_coordinates(simulation, checked.loaded, coordinates)
        hamiltonian.set_state(context, origin.components)
        if settings.minimize_iterations:
            openmm.LocalEnergyMinimizer.minimize(context, 10.0, int(settings.minimize_iterations))
        context.setVelocitiesToTemperature(
            temperature, as_openmm_seed(derive_build_seed(settings.seed,
                                                          f"alchemy/{window_id}/velocities")))
    # The Hamiltonian state is (re)applied after a checkpoint load too: loadCheckpoint restores
    # parameters, and this asserts they are the window's rather than trusting that.
    hamiltonian.set_state(context, origin.components)

    evaluator = CrossStateEvaluator(hamiltonian, states, platform=platform, properties=properties)
    rows_done = committed["state"]["rows"] if committed else 0
    reporter = SampleStreamReporter(paths["samples"], evaluator, origin,
                                    interval=settings.report_interval,
                                    first_step=settings.equilibration_steps, timestep_ps=dt_ps,
                                    components=path.components, rows_already=rows_done)
    simulation.reporters.append(reporter)
    if committed is None and settings.equilibration_steps == 0:
        # step 0 is a sample when there is no equilibration
        reporter.report(simulation, context.getState(getPositions=True, getEnergy=True,
                                                     enforcePeriodicBox=True))
    chk_dir.mkdir(parents=True, exist_ok=True)

    def commit():
        from md_tools.cv.prefix import prefix_digest
        reporter._handle.flush()
        os.fsync(reporter._handle.fileno())
        commit_generation(chk_dir, write_checkpoint=lambda p: simulation.saveCheckpoint(str(p)),
                          state={"fingerprint": fingerprint, "step": simulation.currentStep,
                                 "rows": reporter.rows,
                                 "prefix_sha256": prefix_digest(paths["samples"], reporter.rows),
                                 "window_id": window_id, "platform": platform.getName()})

    try:
        budget = stop_after_steps
        while simulation.currentStep < total_steps:
            chunk = settings.checkpoint_interval - simulation.currentStep % settings.checkpoint_interval
            chunk = min(chunk, total_steps - simulation.currentStep)
            if budget is not None:
                if budget <= 0:
                    return {"window_id": window_id, "disposition": "interrupted",
                            "step": simulation.currentStep, "rows": reporter.rows}
                chunk = min(chunk, budget)
                budget -= chunk
            simulation.step(chunk)
            with paths["out"].open("a", encoding="utf-8") as out:
                out.write(f"{window_id} step {simulation.currentStep}/{total_steps} "
                          f"rows {reporter.rows}\n")
            if simulation.currentStep % settings.checkpoint_interval == 0 or \
                    simulation.currentStep == total_steps:
                commit()
    finally:
        reporter.close()

    expected_rows = settings.steps // settings.report_interval + 1
    if reporter.rows != expected_rows:
        raise WindowError(f"window {window_id} wrote {reporter.rows} rows; the schedule gives "
                          f"{expected_rows}")
    simulation.saveState(str(paths["restart"]))
    completion = {"schema": COMPLETION_SCHEMA, "window_id": window_id,
                  "fingerprint": fingerprint, "rows": reporter.rows,
                  "steps": simulation.currentStep,
                  "samples_sha256": hashlib.sha256(paths["samples"].read_bytes()).hexdigest(),
                  "restart_sha256": hashlib.sha256(paths["restart"].read_bytes()).hexdigest(),
                  "evaluation_self_check_kJ_mol": reporter.checked_self_energy,
                  "evaluation_self_check_tolerance": {
                      "abs_kJ_mol": SELF_CHECK_ABS_KJ_MOL, "rel": evaluator.self_check_rel,
                      "platform": platform.getName(),
                      "precision": properties.get("Precision")}}
    _write_atomically(paths["completion"], json.dumps(completion, indent=1, sort_keys=True) + "\n")
    return {"window_id": window_id, "disposition": "resumed" if committed else "fresh",
            "rows": reporter.rows, "step": simulation.currentStep}


def verify_window(paths: Mapping[str, Path], fingerprint: str) -> list[str]:
    problems = []
    try:
        doc = json.loads(Path(paths["completion"]).read_text())
    except (OSError, ValueError) as exc:
        return [f"completion record unreadable: {exc}"]
    if doc.get("fingerprint") != fingerprint:
        problems.append("completion belongs to a different window definition")
    for key, role in (("samples_sha256", "samples"), ("restart_sha256", "restart")):
        p = Path(paths[role])
        if not p.is_file():
            problems.append(f"{p.name} missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != doc.get(key):
            problems.append(f"{p.name} changed since completion")
    return problems


def _move_aside(paths, chk_dir):
    """--overwrite moves the previous window's files aside; it never deletes them."""
    import time
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for p in [*paths.values(), chk_dir]:
        p = Path(p)
        if p.exists():
            os.replace(p, p.with_name(f"{p.name}.replaced-{stamp}"))


def _set_initial_coordinates(simulation, loaded, coordinates):
    import openmm

    if coordinates:
        text = Path(coordinates).read_text(encoding="utf-8")
        state = openmm.XmlSerializer.deserialize(text)
        if hasattr(state, "getPositions"):
            simulation.context.setState(state)
            return
        raise WindowError(f"-c {coordinates} is not a serialised OpenMM State")
    if loaded.pdb.topology.getPeriodicBoxVectors() is not None:
        simulation.context.setPeriodicBoxVectors(*loaded.pdb.topology.getPeriodicBoxVectors())
    simulation.context.setPositions(loaded.pdb.positions)


def window_samples_reduced(samples: SampleSet) -> np.ndarray:
    """Convenience: the reduced potentials, including pV under NPT (see SampleSet)."""
    return samples.reduced_potential()


__all__ = ["ParametricHamiltonian", "ComposedHamiltonian", "CrossStateEvaluator",
           "SampleStreamReporter", "WindowSettings", "run_window", "read_window_samples",
           "window_paths", "verify_window", "validate_stream_prefix", "stream_columns",
           "check_hamiltonian_matches_path", "RESTRAINT_FORCE_GROUP", "BAR_NM3_TO_KJ_MOL",
           "kt_kj_mol"]
