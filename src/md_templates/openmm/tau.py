"""Amber-style ``tau`` parameterisation of the existing REST2 scaling.

The public and persisted ladder parameter is ``tau``. The Hamiltonian it drives is unchanged:
this module is a reparameterisation, not a new physics. One function derives everything, so a
replica, a manifest, a diagnostic table and a test can never disagree about what ``tau`` meant.

    one_minus_tau = 1 - tau
    s             = (1 - tau)^2
    sqrt(s)       = 1 - tau

which preserves the scaling already implemented in :mod:`md_templates.openmm.system`:

    solute-solute terms       s        = (1 - tau)^2
    solute-environment terms  sqrt(s)  = (1 - tau)
    environment terms         1

``tau = 0`` is therefore the cold, physical replica (``s = 1``) and increasing ``tau`` softens the
enhanced region. That direction matters: a ladder written hot-first would silently invert the
exchange neighbour ordering.

Why ``tau`` and not ``s`` in the public input: ``sqrt(s)`` is the quantity that enters the
solute-environment coupling linearly, and acceptance along the chain is closer to even when the
ladder is evenly spaced in ``sqrt(s)``. ``tau`` is exactly ``1 - sqrt(s)``, so a *linear* tau
ladder is an evenly spaced ``sqrt(s)`` ladder written in the form a user can reason about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

#: ``tau`` outside this range is refused. ``tau = 1`` would erase the enhanced region's internal
#: energy entirely (``s = 0``), which is not a REST2 replica but a different calculation.
TAU_MINIMUM = 0.0
TAU_MAXIMUM_EXCLUSIVE = 1.0


@dataclass(frozen=True)
class TauScaling:
    """Everything derived from one ``tau``.

    ``tau`` is the source parameter and is what gets persisted. ``s``, ``sqrt_s`` and
    ``effective_temperature_kelvin`` are labelled derived diagnostics -- they are recorded so a
    reader can check the ladder without recomputing it, never accepted back as input.
    """

    tau: float
    s: float
    sqrt_s: float

    def effective_temperature_kelvin(self, base_temperature_kelvin: float) -> float:
        """``T_eff = T_bath / s``, the temperature the enhanced region behaves as if it were at.

        This is a diagnostic. REST2 runs every replica at the same bath temperature; nothing in
        the integrator is set from this number.
        """
        if self.s <= 0.0:
            raise ValueError(f"s must be positive to define an effective temperature; got {self.s}")
        return base_temperature_kelvin / self.s


def scaling_for_tau(tau: float) -> TauScaling:
    """Derive ``s`` and ``sqrt(s)`` from one ``tau``.

    This is the single definition. Everything that needs the mapping calls this rather than
    writing ``(1 - tau) ** 2`` again, so the relationship cannot drift between the replica builder,
    the manifest writer and the tests.
    """
    tau_value = float(tau)
    if not (TAU_MINIMUM <= tau_value < TAU_MAXIMUM_EXCLUSIVE):
        raise ValueError(
            f"tau must satisfy {TAU_MINIMUM} <= tau < {TAU_MAXIMUM_EXCLUSIVE}; got {tau_value!r}. "
            "tau = 1 would remove the enhanced region's internal energy entirely (s = 0), which is "
            "not a REST2 replica."
        )
    one_minus_tau = 1.0 - tau_value
    # s is computed as the square of the SAME float used for sqrt_s, so `sqrt_s * sqrt_s == s`
    # holds exactly in IEEE-754 terms rather than approximately.
    return TauScaling(tau=tau_value, s=one_minus_tau * one_minus_tau, sqrt_s=one_minus_tau)


def linear_tau_ladder(minimum: float, maximum: float, count: int) -> list[float]:
    """``count`` inclusive, evenly spaced tau values from ``minimum`` to ``maximum``.

    Endpoints are assigned exactly rather than computed, so a ladder declared as ending at 0.5
    ends at 0.5 and not at 0.4999999999999999.
    """
    if count < 2:
        raise ValueError(f"tau_ladder.count must be at least 2 (a ladder needs two rungs); got {count}")
    if not (TAU_MINIMUM <= minimum < TAU_MAXIMUM_EXCLUSIVE):
        raise ValueError(f"tau_ladder.minimum must satisfy 0 <= tau < 1; got {minimum!r}")
    if not (TAU_MINIMUM <= maximum < TAU_MAXIMUM_EXCLUSIVE):
        raise ValueError(f"tau_ladder.maximum must satisfy 0 <= tau < 1; got {maximum!r}")
    if maximum <= minimum:
        raise ValueError(
            f"tau_ladder.maximum ({maximum!r}) must be greater than tau_ladder.minimum "
            f"({minimum!r}); a ladder with no span is not a ladder."
        )

    span = maximum - minimum
    last = count - 1
    values = [minimum + span * (index / last) for index in range(count)]
    values[0] = minimum          # exact endpoints, not accumulated arithmetic
    values[last] = maximum
    return values


def build_tau_ladder(minimum: float, maximum: float, count: int,
                     interpolation: str = "linear") -> list[float]:
    """Build a tau ladder under the named interpolation.

    Only ``linear`` exists today. It is named rather than assumed so that adding a second
    spacing later is a declared change to the configuration, not a silent change of meaning for
    inputs already written.
    """
    if interpolation != "linear":
        raise ValueError(
            f"tau_ladder.interpolation {interpolation!r} is not supported; only 'linear' is "
            "implemented. A linear tau ladder is an evenly spaced sqrt(s) ladder."
        )
    return linear_tau_ladder(minimum, maximum, count)


def scalings_for_ladder(tau_values: Sequence[float]) -> list[TauScaling]:
    """Derive the full scaling table for a ladder, validating the ladder's shape.

    The cold replica must come first and tau must strictly increase. Both are checked here rather
    than at the call site because an out-of-order ladder produces a run that looks healthy and
    exchanges the wrong neighbours.
    """
    if len(tau_values) < 2:
        raise ValueError(
            f"a REST2 ladder needs at least 2 replicas; got {len(tau_values)}"
        )
    if tau_values[0] != TAU_MINIMUM:
        raise ValueError(
            f"tau_ladder must start at the cold, physical replica tau = 0.0 (s = 1.0); got "
            f"{tau_values[0]!r}. A ladder that never samples the unscaled Hamiltonian has no "
            "replica whose trajectory is the physical ensemble."
        )
    for lower, higher in zip(tau_values, tau_values[1:]):
        if higher <= lower:
            raise ValueError(
                f"tau_ladder must strictly increase from cold to hot; got {list(tau_values)!r}"
            )
    return [scaling_for_tau(value) for value in tau_values]


def scale_factors_for_ladder(tau_values: Sequence[float]) -> list[float]:
    """The ``s`` values a tau ladder resolves to, in ladder order (descending in ``s``).

    This is the bridge to the existing System builder, which takes ``s`` and is unchanged.
    """
    return [scaling.s for scaling in scalings_for_ladder(tau_values)]


def ladder_diagnostics(tau_values: Sequence[float],
                       base_temperature_kelvin: float) -> list[dict]:
    """A labelled, persistable table of the ladder.

    Every derived column is named as derived. ``tau`` is the only column that is an input.
    """
    rows: list[dict] = []
    for index, scaling in enumerate(scalings_for_ladder(tau_values)):
        rows.append({
            "replica_index": index,
            "tau": scaling.tau,
            "derived_s": scaling.s,
            "derived_sqrt_s": scaling.sqrt_s,
            "derived_effective_temperature_kelvin":
                scaling.effective_temperature_kelvin(base_temperature_kelvin),
        })
    return rows


def map_replicas_to_devices(n_replicas: int, device_indices: Sequence[int]) -> list[int]:
    """Deterministically assign each replica an entry from an ordered device list.

    Replicas are dealt round-robin over the devices in the order given. When replicas outnumber
    devices, several replicas share a device -- that is allowed and recorded, not an error, because
    refusing it would make a ten-replica ladder impossible on a four-GPU machine.

    The mapping is a pure function of ``(n_replicas, device_indices)``, so a resumed run
    reconstructs the same assignment without persisting it -- though it is persisted anyway, so a
    changed device list is visible rather than silently re-dealt.
    """
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be positive; got {n_replicas}")
    if not device_indices:
        raise ValueError(
            "device_indices is empty; an accelerator run needs at least one device. Name the "
            "devices explicitly rather than letting the platform choose, so the mapping is "
            "reproducible."
        )
    seen: set[int] = set()
    for device in device_indices:
        if device in seen:
            raise ValueError(f"device_indices contains {device} more than once: {list(device_indices)!r}")
        seen.add(device)
    return [int(device_indices[index % len(device_indices)]) for index in range(n_replicas)]
