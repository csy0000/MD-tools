"""REST2 Hamiltonian scaling: one implementation, used by four protocols.

`REST2Scaler` scales an OpenMM `System` in tau, and `ScalingSelection` says what it scales.
Neither knows anything about replica exchange, because three of the four callers do not exchange:

* **fixed-tau cMD** holds one rung and never swaps;
* **REST2** and **rREST2** build a ladder of them;
* **AIS** moves tau continuously along a switching path.

Keeping the scaler independent of the ladder is what makes "the same scaling everywhere" a
structural fact rather than a promise. There is no second implementation to drift.

The convention, expressed in tau, is documented with its evidence in
`docs/openmm_methods/REST2/README.md` and `docs/scientific-defaults.md`.

    solute-solute, solute torsions, CMAP     (1 - tau)^2
    solute-environment                       (1 - tau)
    generalized Born                         (1 - tau)
    environment-environment, bonds, angles   1
    ordinary amide omega torsions            1   (this repository's convention)

**tau is the only public, persisted coordinate.** `s` and `sqrt(s)` are derived inside the scaling
and never written to a configuration, a record or a trajectory.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from .identity import (HamiltonianMismatch, canonical_system_xml, force_summary, identity_record,
                       require_same_hamiltonian, system_fingerprint)
from .scaler import (OMEGA_DETECTOR_VERSION, REST2_IMPLEMENTATION, TauSwitcher,
                     UnclassifiedForceError,
                     audit_force_classes, build_scaled_system, clone_system, linear_tau_ladder,
                     require_compatible_implementation, scaling_for_tau, torsion_exclusion_report)
from .selection import SELECTION_FORMAT, ScalingSelection, SelectionError, resolve_selection

__all__ = [
    "REST2Scaler",
    "ScalingSelection",
    # the pieces the runtimes and tests reach for by name
    "SelectionError", "SELECTION_FORMAT", "resolve_selection",
    "TauSwitcher", "build_scaled_system", "scaling_for_tau", "linear_tau_ladder",
    "audit_force_classes", "UnclassifiedForceError", "torsion_exclusion_report",
    "clone_system", "REST2_IMPLEMENTATION", "require_compatible_implementation",
    "OMEGA_DETECTOR_VERSION",
    "identity_record", "system_fingerprint", "canonical_system_xml", "force_summary",
    "require_same_hamiltonian", "HamiltonianMismatch",
]


class REST2Scaler:
    """Scale a System in tau, either once or repeatedly on a live Context.

    One object, two modes, because they are the same Hamiltonian reached two ways:

    * :meth:`scaled_system` builds a System at a fixed tau. A REST2 ladder makes one per rung and a
      fixed-tau cMD run makes exactly one.
    * :meth:`switcher` returns a :class:`TauSwitcher` that moves tau on an already-built Context,
      which is what AIS needs, because a switching path changes tau thousands of times and
      rebuilding the System each time would be both slow and a different calculation.

    The selection -- which atoms are solute, which torsions keep their barrier -- is supplied once
    and reused for every tau, so a ladder cannot end up with rungs that disagree about what the
    solute is.
    """

    def __init__(self, base_system, selection: ScalingSelection | None = None, *,
                 solute_indices: Iterable[int] | None = None,
                 excluded_bonds: Iterable[Sequence[int]] = ()) -> None:
        """Take either a resolved :class:`ScalingSelection` or the two raw lists.

        The raw form exists because the replica driver and the AIS runtime already hold indices at
        the point they build a scaler; the selection form is what a generated run uses, because it
        carries the topology digest that makes the indices checkable.
        """
        if selection is None:
            if solute_indices is None:
                raise TypeError("REST2Scaler needs a ScalingSelection or solute_indices")
            selection = ScalingSelection(
                solute_atoms=tuple(sorted(int(i) for i in solute_indices)),
                excluded_bonds=tuple(tuple(sorted(int(i) for i in pair))
                                     for pair in excluded_bonds))
        self.base_system = base_system
        self.selection = selection

    # -- the two ways to reach a scaled Hamiltonian ---------------------------------------------

    def scaled_system(self, tau: float, *, prepare_for_switching: bool = False):
        """A new System at this tau. The base System is never mutated."""
        arguments = self.selection.as_scaler_arguments()
        return build_scaled_system(self.base_system, arguments["solute_indices"], float(tau),
                                   excluded_bonds=arguments["excluded_bonds"],
                                   prepare_for_switching=prepare_for_switching)

    def switcher(self) -> TauSwitcher:
        """A live tau switcher over an unmodified base System, for AIS."""
        arguments = self.selection.as_scaler_arguments()
        return TauSwitcher(self.base_system, arguments["solute_indices"],
                           excluded_bonds=arguments["excluded_bonds"])

    def ladder(self, n_states: int, tau_max: float) -> list[float]:
        """The linear tau ladder. State 0 is always the unmodified physical Hamiltonian."""
        return linear_tau_ladder(0.0, float(tau_max), int(n_states))

    # -- what the run records -------------------------------------------------------------------

    def identity(self, tau: float, *, temperature_k: float, ensemble: str,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """The scientific identity of the Hamiltonian at this tau.

        Computed from the SCALED System rather than from the configuration that asked for it, and
        recomputed on both sides whenever two runs are compared -- two stored claims are never
        compared with each other. Temperature and ensemble are required because tau alone does not
        identify a thermodynamic state: two ladders at the same tau and different temperatures are
        different states, and a reservoir must not be accepted for one on the strength of the
        other.
        """
        arguments = self.selection.as_scaler_arguments()
        return identity_record(self.scaled_system(tau), tau=float(tau),
                               temperature_k=float(temperature_k), ensemble=str(ensemble),
                               solute_indices=arguments["solute_indices"],
                               excluded_bonds=arguments["excluded_bonds"], extra=extra)

    def scaling_factors(self, tau: float) -> tuple[float, float]:
        """`(s, sqrt(s))` for this tau. Derived on demand; never persisted."""
        return scaling_for_tau(float(tau))

    def __repr__(self) -> str:                                    # pragma: no cover - diagnostics
        return (f"REST2Scaler(solute_atoms={len(self.selection.solute_atoms)}, "
                f"excluded_bonds={len(self.selection.excluded_bonds)})")
