#!/usr/bin/env python
"""The narrow, version-pinned OpenMMTools extension this repository's REST2 needs.

Copied verbatim into every generated REST2 project, next to `rest2_scaling.py`.

OpenMMTools owns replica exchange here: propagation, reduced potentials, exchange decisions,
thermodynamic-state assignment, NetCDF storage, checkpointing, restart and the walker-to-state
mapping. This module does NOT reimplement any of that. It exists for exactly two reasons, both of
which were established by reading the installed 0.26.0 source and running it:

1. `swap-neighbors` raises `TypeError` on NumPy >= 1.25.

   `ReplicaExchangeSampler._mix_neighboring_replicas` locates replicas with `np.where(...)`, which
   returns a TUPLE of arrays. Indexing the energy matrix with that tuple yields a 2-d array rather
   than a scalar, and the acceptance test then calls `math.exp` on it:

       TypeError: only 0-dimensional arrays can be converted to Python scalars

   The ordered REST2 ladder wants neighbour swaps, so the scheme cannot simply be avoided.
   `swap-all` is unaffected and remains available. The override below is the same algorithm with
   scalar indices -- it changes no decision rule, and it computes the criterion by calling this
   repository's own `exchange_log_acceptance`, so the ladder and the tests share one definition of
   the Metropolis criterion rather than agreeing by inspection.

2. Solute coordinates are wanted more often than exchanges are attempted.

   `MultiStateReporter` writes `analysis_particle_indices` positions EVERY iteration and full
   coordinates every `checkpoint_interval` iterations. So the honest way to save solute frames at
   2 ps while exchanging at 10 ps is to make the ITERATION 2 ps and attempt exchange every fifth
   one. Every solute frame is then a real configuration the integrator actually visited. The
   alternative -- one 10 ps iteration, with the boundary configuration written five times -- would
   be a fabricated trajectory, and is not done.

   The stride is applied in `_mix_replicas`, which is the one method whose whole job is "decide
   whether states move this iteration". Propagation, energy evaluation, reporting and restart are
   untouched and remain OpenMMTools'.

The cost of the strided model is that the reduced-potential matrix is computed once per iteration
rather than once per exchange, i.e. `exchange_stride` times more often than the exchange itself
needs. That is a real cost and it is measured rather than assumed; see
`docs/openmmtools-rest2.md` for the benchmark on the ALA system.

Both overrides touch private OpenMMTools methods, so this module refuses to run against a version
it has not been tested against. A silently changed private method is precisely the failure this
pin exists to turn into an error.
"""
import math

import numpy as np
from openmmtools.multistate import ReplicaExchangeSampler

from rest2_scaling import exchange_log_acceptance

#: The single OpenMMTools version this extension is tested against. `_mix_replicas`,
#: `_mix_neighboring_replicas` and `_attempt_swap` are private; upstream may change them in any
#: release, and a subclass that silently follows a changed parent is how a wrong exchange rule
#: reaches production looking healthy.
TESTED_OPENMMTOOLS_VERSION = "0.26.0"

#: Set only by a deliberate operator who has re-read the upstream source. Never set by generation.
VERSION_OVERRIDE_ENVIRONMENT = "MD_TEMPLATES_ALLOW_UNTESTED_OPENMMTOOLS"


class UntestedOpenMMToolsError(RuntimeError):
    """The installed OpenMMTools is not the version this extension was validated against."""


def installed_openmmtools_version():
    import openmmtools
    return openmmtools.__version__


def require_tested_openmmtools(*, environment=None):
    """Refuse an OpenMMTools this extension has not been tested against.

    Returns the installed version. Raises `UntestedOpenMMToolsError` unless it matches, or unless
    the override variable is set to a non-empty value -- in which case the caller has taken
    responsibility and the fact is recorded in provenance.
    """
    import os
    environment = os.environ if environment is None else environment
    installed = installed_openmmtools_version()
    if installed == TESTED_OPENMMTOOLS_VERSION:
        return installed
    if environment.get(VERSION_OVERRIDE_ENVIRONMENT):
        return installed
    raise UntestedOpenMMToolsError(
        f"this REST2 extension subclasses private OpenMMTools methods and is tested only against "
        f"openmmtools {TESTED_OPENMMTOOLS_VERSION}; the installed version is {installed}.\n"
        f"  The overridden methods are ReplicaExchangeSampler._mix_replicas, "
        f"_mix_neighboring_replicas and _attempt_swap.\n"
        f"  Re-read them upstream, extend the contract tests to the new version, and only then "
        f"set {VERSION_OVERRIDE_ENVIRONMENT}=1 to proceed.")


def exchange_stride_for(*, exchange_interval_ps, solute_output_interval_ps):
    """How many iterations make one exchange attempt. Must divide exactly."""
    if solute_output_interval_ps <= 0:
        raise ValueError(f"solute output interval must be positive; got {solute_output_interval_ps}")
    if exchange_interval_ps <= 0:
        raise ValueError(f"exchange interval must be positive; got {exchange_interval_ps}")
    ratio = exchange_interval_ps / solute_output_interval_ps
    if ratio < 1.0:
        raise ValueError(
            f"the exchange interval ({exchange_interval_ps} ps) is shorter than the solute output "
            f"interval ({solute_output_interval_ps} ps). Exchanges cannot be attempted more often "
            f"than the iteration that carries them.")
    stride = int(round(ratio))
    if abs(ratio - stride) > 1e-9:
        raise ValueError(
            f"the exchange interval ({exchange_interval_ps} ps) must be a whole multiple of the "
            f"solute output interval ({solute_output_interval_ps} ps); the ratio is {ratio}. "
            f"Rejected rather than rounded: a rounded stride would silently change the exchange "
            f"period and the reported physical time would not be the one simulated.")
    if stride < 1:
        raise ValueError(
            f"the exchange interval ({exchange_interval_ps} ps) is shorter than the solute output "
            f"interval ({solute_output_interval_ps} ps). Exchanges cannot be attempted more often "
            f"than the iteration that carries them.")
    return stride


class StridedREST2Sampler(ReplicaExchangeSampler):
    """`ReplicaExchangeSampler` with neighbour swaps that work, attempted every `exchange_stride`.

    `exchange_stride = 1` reproduces stock OpenMMTools iteration-per-exchange behaviour, with the
    neighbour-swap correction still applied.
    """

    def __init__(self, *args, exchange_stride=1, **kwargs):
        require_tested_openmmtools()
        stride = int(exchange_stride)
        if stride < 1:
            raise ValueError(f"exchange_stride must be >= 1; got {exchange_stride}")
        self._exchange_stride = stride
        super().__init__(*args, **kwargs)

    @property
    def exchange_stride(self):
        """Iterations per exchange attempt. Not stored in the NetCDF by OpenMMTools, so a resume
        must be given it again; the runtime reads it back from its own manifest and checks it."""
        return getattr(self, "_exchange_stride", 1)

    @exchange_stride.setter
    def exchange_stride(self, value):
        stride = int(value)
        if stride < 1:
            raise ValueError(f"exchange_stride must be >= 1; got {value}")
        self._exchange_stride = stride

    # -- the two corrections -------------------------------------------------------------------

    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate every replica AT ITS OWN TAU, attempting no exchanges.

        Upstream's `equilibrate` calls `_mix_replicas` on every equilibration iteration, and it
        does so with the iteration counter still at 0 -- which the stride test below would read as
        "an exchange iteration", so every single equilibration iteration would exchange. That is
        the opposite of what per-tau equilibration is for: each replica is being relaxed INTO its
        own Hamiltonian, and a swap during that undoes the relaxation it just paid for.

        The iteration counter is not incremented by upstream, so this equilibration correctly
        contributes nothing to the exchange-production time.
        """
        self._equilibrating = True
        try:
            return super().equilibrate(n_iterations, mcmc_moves=mcmc_moves)
        finally:
            self._equilibrating = False

    def _mix_replicas(self):
        """Attempt exchanges only on stride iterations; otherwise leave the assignment alone.

        Returning the current assignment unchanged is what "propagate but do not exchange" means:
        OpenMMTools still propagates, still computes energies, still reports, and still records
        the (unchanged) mapping for this iteration, so the stored trajectory stays continuous.
        """
        if getattr(self, "_equilibrating", False):
            return self._replica_thermodynamic_states
        if self._iteration % self.exchange_stride != 0:
            return self._replica_thermodynamic_states
        return super()._mix_replicas()

    def _mix_neighboring_replicas(self):
        """Upstream's alternating neighbour scheme, with scalar replica indices.

        Identical to `ReplicaExchangeSampler._mix_neighboring_replicas` in 0.26.0 except that
        `np.where(...)` is reduced to an `int` before it is used to index the energy matrix.
        """
        # Upstream draws the offset from {0, 1} unconditionally. With only TWO replicas the odd
        # phase covers no pair at all -- `range(1, 1, 2)` is empty -- so half of the exchange
        # iterations propose nothing and the ladder silently exchanges at half the configured
        # rate. The offset is drawn only from phases that actually contain a pair.
        offsets = [offset for offset in (0, 1)
                   if any(True for _ in range(offset, self.n_replicas - 1, 2))]
        offset = offsets[np.random.randint(len(offsets))] if len(offsets) > 1 else offsets[0]
        for state_i in range(offset, self.n_replicas - 1, 2):
            state_j = state_i + 1
            replica_i = int(np.where(self._replica_thermodynamic_states == state_i)[0][0])
            replica_j = int(np.where(self._replica_thermodynamic_states == state_j)[0][0])
            self._attempt_swap(replica_i, replica_j)

    def _attempt_swap(self, replica_i, replica_j):
        """One neighbour swap, using this repository's own acceptance criterion.

        `replica_i` and `replica_j` are plain integers here, not `np.where` tuples.

        The reduced potentials OpenMMTools stores already carry beta and, under NPT, the pV term.
        Every REST2 rung shares one temperature and one pressure, so the pV contributions cancel
        between the two sides -- which is why this is the same criterion the repository's own
        `exchange_log_acceptance` states, and why it is called rather than restated.
        """
        state_i = int(self._replica_thermodynamic_states[replica_i])
        state_j = int(self._replica_thermodynamic_states[replica_j])
        energies = self._energy_thermodynamic_states
        log_p_accept = exchange_log_acceptance(
            u_ii=float(energies[replica_i, state_i]),
            u_jj=float(energies[replica_j, state_j]),
            u_ij=float(energies[replica_i, state_j]),
            u_ji=float(energies[replica_j, state_i]),
        )
        self._n_proposed_matrix[state_i, state_j] += 1
        self._n_proposed_matrix[state_j, state_i] += 1
        if log_p_accept >= 0.0 or np.random.rand() < math.exp(log_p_accept):
            self._replica_thermodynamic_states[replica_i] = state_j
            self._replica_thermodynamic_states[replica_j] = state_i
            self._n_accepted_matrix[state_i, state_j] += 1
            self._n_accepted_matrix[state_j, state_i] += 1
