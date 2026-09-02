"""Positional restraints on solute heavy atoms, in one place.

A generated script must not build a `CustomExternalForce`. It used to, in every project, which
meant the restraint's functional form was copied into every generated directory and a correction
reached only the projects generated after it.

The force is added ONCE, before any state is loaded, and its strength is then changed through a
global parameter. That ordering is not a detail: adding or removing a Force changes the System's
force layout, and therefore the checkpoint layout, so a chain whose restrained and free stages had
different layouts could not hand a checkpoint from one to the next. Present-at-zero is what keeps
the layout identical along the whole chain.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ._stages import add_positional_restraint, set_restraint

__all__ = ["PositionalRestraint"]

#: The global parameter the force's strength is carried in. One name, so a reader of a serialised
#: System can find it, and so `set_strength` cannot drift from what `add` created.
PARAMETER = "k_restraint"


class PositionalRestraint:
    """A harmonic tether on chosen atoms, whose strength can be changed without rebuilding.

    Used by every restrained equilibration stage and by nothing else; a free stage carries the same
    force at zero strength so the Force layout is constant along the chain.
    """

    def __init__(self, system, reference_positions, atom_indices: Iterable[int]) -> None:
        """Add the restraining force to `system`, referenced to `reference_positions`.

        The reference is the configuration the restraint pulls TOWARDS -- normally the built
        structure -- and is captured here rather than read later, so a restraint cannot silently
        start referring to wherever the simulation has since moved.
        """
        self.atom_indices = tuple(int(i) for i in atom_indices)
        self.force_index = add_positional_restraint(system, reference_positions,
                                                    self.atom_indices)
        self._system = system

    @property
    def n_restrained(self) -> int:
        return len(self.atom_indices)

    def set_strength(self, simulation, kcal_per_mol_A2: float) -> None:
        """Change the force constant on a live Context. 0.0 releases it without removing it."""
        set_restraint(simulation, float(kcal_per_mol_A2))

    @staticmethod
    def solute_heavy_atoms(topology, solute_atoms: Sequence[int] | None = None) -> tuple[int, ...]:
        """Which atoms a restraint is applied to: solute, and not hydrogen.

        Hydrogens are left free because restraining them does nothing useful -- they follow the
        heavy atom they are bonded to -- while adding terms to every evaluation.
        """
        allowed = None if solute_atoms is None else {int(i) for i in solute_atoms}
        return tuple(atom.index for atom in topology.atoms()
                     if (allowed is None or atom.index in allowed)
                     and atom.element is not None and atom.element.symbol != "H")

    def __repr__(self) -> str:                                    # pragma: no cover - diagnostics
        return f"PositionalRestraint(atoms={self.n_restrained})"
