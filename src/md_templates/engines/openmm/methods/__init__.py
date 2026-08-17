"""Method implementations for the OpenMM provider.

One module per method. They share the engine plumbing above them -- system building, solvation,
equilibration, platform selection, persistence -- and keep their own invariants explicit: a
conventional walker holds one Hamiltonian, REST2 owns a ladder, an exchange schedule and an RNG
whose behaviour is part of the scientific contract.
"""
