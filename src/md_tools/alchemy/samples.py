"""The sample record every alchemical estimator reads.

One record, one Hamiltonian. FEP, BAR and MBAR read the cross-state potentials in it; TI reads the
derivative components in it. They are analyses of the same samples of the same path, which is
what makes their agreement mean something: two implementations of the potential that happened to
agree would prove nothing.

WHAT A SAMPLE IS

    A configuration x_n drawn from ONE thermodynamic state, its ORIGIN state, with

      potential_kj_mol[n, k]   U_k(x_n): the potential of x_n evaluated at EVERY state k, kJ/mol
      volume_nm3[n]            V(x_n), required under NPT, absent under NVT
      derivatives[c][n]        partial U / partial lambda_c at x_n and the origin state, kJ/mol

    The reduced potential is COMPUTED here, never stored:

      u_k(x_n) = beta * (U_k(x_n) + p V(x_n))                                 (NPT)
      u_k(x_n) = beta * U_k(x_n)                                              (NVT)

    Storing the potential and volume rather than u means the pV term cannot be silently left out
    of a record: an NPT record with no volume column is refused on read.

SAMPLE PROVENANCE IS THE ORIGIN STATE

    `origin_state` is the state the sample was drawn from. There is no rank, worker, GPU or
    process field, and one is refused by name on read: the rank of the worker that processed a
    window equals the window index only by accident, and only until a campaign is resumed on a
    different number of GPUs. A record that grouped samples by rank would then assign them to the
    wrong state and every estimator would run without complaint.

UNITS

    Energies kJ/mol, volumes nm^3, pressure bar, temperature K, reduced quantities in kT. The
    Boltzmann constant is OpenMM's (`MOLAR_GAS_CONSTANT_R`), so a reduced potential computed here
    equals one computed from an OpenMM energy at the same temperature.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from md_tools.alchemy.paths import AlchemicalPath, check_component_name, S_TOLERANCE

SAMPLES_SCHEMA = "md-tools-alchemical-samples/1"

#: kJ/(mol K). OpenMM's value (openmm.unit.MOLAR_GAS_CONSTANT_R), written out so analysis does not
#: import OpenMM.
BOLTZMANN_KJ_MOL_K = 0.00831446261815324
#: 1 bar nm^3 per molecule, in kJ/mol: 1e5 Pa * 1e-27 m^3 * N_A / 1000.
BAR_NM3_TO_KJ_MOL = 0.06022140760
KJ_PER_KCAL = 4.184

#: Field names that would make sample provenance depend on the execution layout. Refused on read.
_EXECUTION_LAYOUT_FIELDS = ("rank", "worker", "worker_rank", "gpu", "device", "process",
                            "replica_rank", "mpi_rank")

UNITS = {"potential": "kJ/mol", "derivative": "kJ/mol", "volume": "nm^3", "pressure": "bar",
         "temperature": "K", "reduced": "kT"}


class SampleRecordError(ValueError):
    """A sample record that estimators must not consume."""


def kt_kj_mol(temperature_k: float) -> float:
    return BOLTZMANN_KJ_MOL_K * float(temperature_k)


@dataclass(frozen=True)
class ThermodynamicState:
    """One fixed state: named lambda components at a point `s` of a path, T, and p if NPT."""

    state_id: str
    s: float
    components: Mapping[str, float]
    temperature_k: float
    pressure_bar: float | None = None

    def __post_init__(self):
        if not self.state_id or any(c in self.state_id for c in " ,\n\t"):
            raise SampleRecordError(f"state id {self.state_id!r}: a state id is a non-empty "
                                    f"name without spaces or commas")
        for name in self.components:
            check_component_name(name)
        if not self.temperature_k > 0:
            raise SampleRecordError(f"state {self.state_id}: temperature {self.temperature_k} K")
        if self.pressure_bar is not None and not self.pressure_bar > 0:
            raise SampleRecordError(f"state {self.state_id}: pressure {self.pressure_bar} bar")

    @property
    def beta(self) -> float:
        return 1.0 / kt_kj_mol(self.temperature_k)

    @property
    def ensemble(self) -> str:
        return "NVT" if self.pressure_bar is None else "NPT"

    def to_record(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "s": float(self.s),
                "components": {k: float(v) for k, v in sorted(self.components.items())},
                "temperature_k": float(self.temperature_k),
                "pressure_bar": None if self.pressure_bar is None else float(self.pressure_bar)}

    @classmethod
    def from_record(cls, r: Mapping[str, Any]) -> "ThermodynamicState":
        return cls(r["state_id"], float(r["s"]), dict(r["components"]),
                   float(r["temperature_k"]),
                   None if r.get("pressure_bar") is None else float(r["pressure_bar"]))


def window_states(path: AlchemicalPath, s_values: Sequence[float], *, temperature_k: float,
                  pressure_bar: float | None = None, prefix: str = "w") -> tuple[ThermodynamicState, ...]:
    """The fixed-lambda windows at `s_values`, ids `w000`, `w001`, ... in increasing s."""
    s_sorted = sorted(float(s) for s in s_values)
    if any(b - a <= S_TOLERANCE for a, b in zip(s_sorted, s_sorted[1:])):
        raise SampleRecordError(f"window positions repeat: {s_sorted}")
    path.require_knots_sampled(s_sorted)
    return tuple(ThermodynamicState(f"{prefix}{i:03d}", s, path.components_at(s), temperature_k,
                                    pressure_bar)
                 for i, s in enumerate(s_sorted))


@dataclass
class SampleSet:
    """Samples from a set of states of ONE path, with cross-state potentials and derivatives."""

    states: tuple[ThermodynamicState, ...]
    path: AlchemicalPath
    sample_id: np.ndarray            # (N,) str, unique
    origin_state: np.ndarray         # (N,) str, a state id
    step: np.ndarray                 # (N,) int, integrator step within the origin window
    potential_kj_mol: np.ndarray     # (N, K)
    volume_nm3: np.ndarray | None = None
    derivatives: dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.sample_id = np.asarray(self.sample_id, dtype=str)
        self.origin_state = np.asarray(self.origin_state, dtype=str)
        self.step = np.asarray(self.step, dtype=np.int64)
        self.potential_kj_mol = np.asarray(self.potential_kj_mol, dtype=float)
        if self.volume_nm3 is not None:
            self.volume_nm3 = np.asarray(self.volume_nm3, dtype=float)
        self.derivatives = {k: np.asarray(v, dtype=float) for k, v in self.derivatives.items()}
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        ids = [s.state_id for s in self.states]
        if len(set(ids)) != len(ids):
            raise SampleRecordError(f"state ids repeat: {ids}")
        n, k = len(self.sample_id), len(self.states)
        if self.potential_kj_mol.shape != (n, k):
            raise SampleRecordError(f"potential_kj_mol has shape {self.potential_kj_mol.shape}; "
                                    f"{n} samples at {k} states need ({n}, {k})")
        for name, arr in (("origin_state", self.origin_state), ("step", self.step)):
            if arr.shape != (n,):
                raise SampleRecordError(f"{name} has shape {arr.shape}, expected ({n},)")
        if len(set(self.sample_id.tolist())) != n:
            raise SampleRecordError("sample ids repeat: every sample has its own id")
        unknown = sorted(set(self.origin_state.tolist()) - set(ids))
        if unknown:
            raise SampleRecordError(f"samples name origin states {unknown} that the record does "
                                    f"not define")
        if not np.all(np.isfinite(self.potential_kj_mol)):
            bad = np.argwhere(~np.isfinite(self.potential_kj_mol))[0]
            raise SampleRecordError(
                f"non-finite potential for sample {self.sample_id[bad[0]]} at state "
                f"{ids[bad[1]]}: a softcore path should not produce one, and dropping it would "
                f"bias every estimator")
        temps = {s.temperature_k for s in self.states}
        if len(temps) != 1:
            raise SampleRecordError(f"states at several temperatures {sorted(temps)}: alchemical "
                                    f"windows share one temperature")
        pressures = {s.pressure_bar for s in self.states}
        if len(pressures) != 1:
            raise SampleRecordError(f"states at several pressures {sorted(pressures, key=str)}; "
                                    f"alchemical windows share one ensemble")
        if self.ensemble == "NPT":
            if self.volume_nm3 is None:
                raise SampleRecordError(
                    "NPT states and no volume column: the reduced potential under NPT is "
                    "beta*(U + pV), and without V the pV term would be silently omitted")
            if self.volume_nm3.shape != (n,) or not np.all(self.volume_nm3 > 0):
                raise SampleRecordError("volume_nm3 must be one positive volume per sample")
        elif self.volume_nm3 is not None:
            raise SampleRecordError("a volume column on NVT states: the record would claim a "
                                    "pV term the ensemble does not have")
        for name, arr in self.derivatives.items():
            check_component_name(name)
            if name not in self.path.components:
                raise SampleRecordError(f"derivative for {name}, which the path does not move")
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise SampleRecordError(f"derivative {name}: one finite value per sample")
        for st in self.states:
            expect = self.path.components_at(st.s)
            if set(expect) != set(st.components) or any(
                    abs(expect[c] - float(st.components[c])) > 1e-9 for c in expect):
                raise SampleRecordError(
                    f"state {st.state_id} at s={st.s} sets {dict(st.components)}, but the path "
                    f"puts {expect} there")

    # ------------------------------------------------------------------ views
    @property
    def ensemble(self) -> str:
        return self.states[0].ensemble

    @property
    def temperature_k(self) -> float:
        return self.states[0].temperature_k

    @property
    def kt(self) -> float:
        return kt_kj_mol(self.temperature_k)

    @property
    def state_ids(self) -> list[str]:
        return [s.state_id for s in self.states]

    def state_index(self, state_id: str) -> int:
        try:
            return self.state_ids.index(state_id)
        except ValueError:
            raise SampleRecordError(f"no state {state_id!r}; states are {self.state_ids}") from None

    def reduced_potential(self) -> np.ndarray:
        """(N, K) u_k(x_n) = beta (U_k + p V), dimensionless."""
        energy = self.potential_kj_mol.copy()
        if self.ensemble == "NPT":
            energy += (self.states[0].pressure_bar * BAR_NM3_TO_KJ_MOL
                       * self.volume_nm3)[:, None]
        return energy / self.kt

    def from_state(self, state_id: str) -> np.ndarray:
        """Indices of the samples drawn from `state_id`, in step order."""
        idx = np.flatnonzero(self.origin_state == state_id)
        return idx[np.argsort(self.step[idx], kind="stable")]

    def counts(self) -> dict[str, int]:
        return {sid: int(np.sum(self.origin_state == sid)) for sid in self.state_ids}

    def subset(self, indices: Sequence[int]) -> "SampleSet":
        idx = np.asarray(indices, dtype=int)
        return SampleSet(self.states, self.path, self.sample_id[idx], self.origin_state[idx],
                         self.step[idx], self.potential_kj_mol[idx],
                         None if self.volume_nm3 is None else self.volume_nm3[idx],
                         {k: v[idx] for k, v in self.derivatives.items()},
                         dict(self.provenance))

    # ------------------------------------------------------------------ persistence
    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SAMPLES_SCHEMA,
            "units": UNITS,
            "provenance_rule": "origin_state is the state each sample was drawn from; no "
                               "execution-layout field (rank, worker, device) exists",
            "path": self.path.to_record(),
            "states": [s.to_record() for s in self.states],
            "samples": {
                "sample_id": self.sample_id.tolist(),
                "origin_state": self.origin_state.tolist(),
                "step": self.step.tolist(),
                "potential_kj_mol": self.potential_kj_mol.tolist(),
                "volume_nm3": None if self.volume_nm3 is None else self.volume_nm3.tolist(),
                "derivatives_kj_mol": {k: v.tolist() for k, v in sorted(self.derivatives.items())},
            },
            "provenance": self.provenance,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SampleSet":
        if record.get("schema") != SAMPLES_SCHEMA:
            raise SampleRecordError(f"sample record schema {record.get('schema')!r}; this reader "
                                    f"reads {SAMPLES_SCHEMA!r}")
        if record.get("units") != UNITS:
            raise SampleRecordError(f"sample record units {record.get('units')}; this reader "
                                    f"reads {UNITS}")
        samples = record["samples"]
        layout = sorted(set(samples) & set(_EXECUTION_LAYOUT_FIELDS))
        if layout:
            raise SampleRecordError(
                f"sample record carries {layout}: sample provenance is the ORIGIN STATE, never "
                f"the worker that processed it -- they coincide only by accident, and not after "
                f"a resume on a different number of GPUs")
        allowed = {"sample_id", "origin_state", "step", "potential_kj_mol", "volume_nm3",
                   "derivatives_kj_mol"}
        extra = sorted(set(samples) - allowed)
        if extra:
            raise SampleRecordError(f"unknown sample fields {extra}")
        return cls(
            states=tuple(ThermodynamicState.from_record(s) for s in record["states"]),
            path=AlchemicalPath.from_record(record["path"]),
            sample_id=samples["sample_id"], origin_state=samples["origin_state"],
            step=samples["step"], potential_kj_mol=samples["potential_kj_mol"],
            volume_nm3=samples.get("volume_nm3"),
            derivatives=dict(samples.get("derivatives_kj_mol") or {}),
            provenance=dict(record.get("provenance") or {}))

    def write_json(self, path: str | Path) -> str:
        blob = json.dumps(self.to_record(), sort_keys=True, indent=1).encode() + b"\n"
        Path(path).write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def read_json(cls, path: str | Path) -> "SampleSet":
        return cls.from_record(json.loads(Path(path).read_text()))


def concatenate(parts: Sequence[SampleSet]) -> SampleSet:
    """One record from per-window records of the same path and states (e.g. one file per window)."""
    if not parts:
        raise SampleRecordError("nothing to concatenate")
    first = parts[0]
    for p in parts[1:]:
        if [s.to_record() for s in p.states] != [s.to_record() for s in first.states]:
            raise SampleRecordError("window records disagree on the state set; they are not "
                                    "samples of one experiment")
        if p.path.digest() != first.path.digest():
            raise SampleRecordError("window records disagree on the path")
        if set(p.derivatives) != set(first.derivatives):
            raise SampleRecordError("window records carry different derivative components")
        if (p.volume_nm3 is None) != (first.volume_nm3 is None):
            raise SampleRecordError("window records disagree on the volume column")
    cat = np.concatenate
    return SampleSet(
        first.states, first.path,
        cat([p.sample_id for p in parts]), cat([p.origin_state for p in parts]),
        cat([p.step for p in parts]), cat([p.potential_kj_mol for p in parts]),
        None if first.volume_nm3 is None else cat([p.volume_nm3 for p in parts]),
        {k: cat([p.derivatives[k] for p in parts]) for k in first.derivatives},
        {"concatenated_from": [p.provenance for p in parts]})
