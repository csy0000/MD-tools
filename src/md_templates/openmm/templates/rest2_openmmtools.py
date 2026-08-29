#!/usr/bin/env python
"""The OpenMMTools exchange layer for REST2, and an honest statement of who owns what.

Copied verbatim into every generated REST2 project, next to `rest2_scaling.py`.

WHO OWNS THE EXCHANGE DECISION
    With the default scheme, `swap-all`, OpenMMTools owns it completely. Every proposal and every
    Metropolis accept/reject happens inside `ReplicaExchangeSampler._mix_all_replicas_numba`, which
    this module does not touch. Nothing here computes an acceptance probability.

    This repository owns exactly one thing about mixing: WHEN it is attempted. `_mix_replicas` is
    gated so that exchange happens every `exchange_stride` iterations rather than every iteration,
    which is what lets solute coordinates be stored more often than exchanges without fabricating
    frames. That is a schedule, not a decision, and the distinction is the point of this docstring.

    With `swap-neighbors` the ownership is different and is stated rather than glossed: OpenMMTools
    0.26.0's neighbour path is BROKEN on NumPy >= 1.25, so selecting it activates
    `ProjectNeighbourExchangeMixin`, in which THIS REPOSITORY performs the Metropolis decision.
    That is why it is not the default.

WHY swap-all IS THE DEFAULT
    It is stock, and it is what OpenMMTools and YANK use by default. Per mixing event it makes
    `n_replicas**3` uniformly random replica-pair proposals rather than one sweep of adjacent
    pairs, which mixes an ordered ladder at least as well and usually better (Chodera and Shirts,
    J. Chem. Phys. 135:194110, 2011). Both schemes satisfy detailed balance at the one common beta
    every REST2 rung shares.

    Its proposal semantics differ from a neighbour sweep in two ways that any statistic built from
    it must respect, and `rest2_statistics.py` does:
      * proposals land on ALL state pairs, not only adjacent ones;
      * a pair may be drawn with i == j. A self-swap has log_p_accept = 0, is always "accepted",
        and is counted on the DIAGONAL of both matrices. Diagonal entries are therefore not
        exchanges and must never be reported as acceptance.

THE UPSTREAM NEIGHBOUR DEFECT
    `ReplicaExchangeSampler._mix_neighboring_replicas` locates replicas with `np.where(...)`, which
    returns a TUPLE of arrays. Indexing the energy matrix with that tuple yields a 2-d array, and
    the acceptance test then calls `math.exp` on it:

        TypeError: only 0-dimensional arrays can be converted to Python scalars

    `swap-all` is unaffected because `_mix_all_replicas` draws plain integers.

    A second defect: with only TWO replicas, upstream's alternating offset leaves the odd phase
    with no pair at all, so half the exchange iterations propose nothing and the ladder exchanges
    at half its configured rate.

    Both are corrected in the mixin below, which is used ONLY when `swap-neighbors` is explicitly
    selected. Because it subclasses private methods, selecting it requires the tested OpenMMTools
    version and the run records that project code owned the decision.
"""
import math

import numpy as np
from openmmtools.multistate import ReplicaExchangeSampler

from rest2_scaling import exchange_log_acceptance

#: The single OpenMMTools version the private-method extension is tested against. Only the
#: `swap-neighbors` path depends on it; `swap-all` uses no private method of this class.
TESTED_OPENMMTOOLS_VERSION = "0.26.0"

#: Set only by a deliberate operator who has re-read the upstream source. Never set by generation.
VERSION_OVERRIDE_ENVIRONMENT = "MD_TEMPLATES_ALLOW_UNTESTED_OPENMMTOOLS"

#: The mixing schemes this module supports, and who owns the accept/reject decision in each.
EXCHANGE_OWNERSHIP = {
    "swap-all": "openmmtools",
    "swap-neighbors": "md-templates",
}

#: The default. Stock OpenMMTools, no private method involved in the decision.
DEFAULT_MIXING_SCHEME = "swap-all"


class UntestedOpenMMToolsError(RuntimeError):
    """The installed OpenMMTools is not the version the private-method extension was tested on."""


def installed_openmmtools_version():
    import openmmtools
    return openmmtools.__version__


def require_tested_openmmtools(*, environment=None):
    """Refuse an OpenMMTools the private-method extension has not been tested against."""
    import os
    environment = os.environ if environment is None else environment
    installed = installed_openmmtools_version()
    if installed == TESTED_OPENMMTOOLS_VERSION:
        return installed
    if environment.get(VERSION_OVERRIDE_ENVIRONMENT):
        return installed
    raise UntestedOpenMMToolsError(
        f"`swap-neighbors` needs this repository's neighbour-exchange fix, which subclasses "
        f"private OpenMMTools methods and is tested only against openmmtools "
        f"{TESTED_OPENMMTOOLS_VERSION}; the installed version is {installed}.\n"
        f"  Overridden: ReplicaExchangeSampler._mix_neighboring_replicas and _attempt_swap.\n"
        f"  Use the default `swap-all`, which needs no private method, or re-read the upstream "
        f"source, extend the contract tests, and set {VERSION_OVERRIDE_ENVIRONMENT}=1.")


def exchange_ownership(scheme):
    """Who decides accept/reject under `scheme`. Recorded in provenance, never inferred later."""
    try:
        return EXCHANGE_OWNERSHIP[scheme]
    except KeyError:
        raise ValueError(
            f"unknown replica mixing scheme {scheme!r}; supported: "
            f"{sorted(EXCHANGE_OWNERSHIP)}") from None


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
    return stride


class StridedMixingMixin:
    """Attempt exchange every `exchange_stride` iterations. SCHEDULING ONLY.

    This changes WHEN `ReplicaExchangeSampler._mix_replicas` runs, never what it decides. On a
    non-exchange iteration the current assignment is returned unchanged, which is what "propagate
    but do not exchange" means: OpenMMTools still propagates, still computes energies, and still
    reports the (unchanged) mapping and a zero mixing-statistics row for that iteration, so the
    stored history stays continuous and the zero rows make skipped iterations countable.
    """

    def __init__(self, *args, exchange_stride=1, **kwargs):
        stride = int(exchange_stride)
        if stride < 1:
            raise ValueError(f"exchange_stride must be >= 1; got {exchange_stride}")
        self._exchange_stride = stride
        super().__init__(*args, **kwargs)

    @property
    def exchange_stride(self):
        """Iterations per exchange attempt. OpenMMTools does not store this, so a resume is given
        it again from the run's own metadata and the value is checked against what was stored."""
        return getattr(self, "_exchange_stride", 1)

    @exchange_stride.setter
    def exchange_stride(self, value):
        stride = int(value)
        if stride < 1:
            raise ValueError(f"exchange_stride must be >= 1; got {value}")
        self._exchange_stride = stride

    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate every replica AT ITS OWN TAU, attempting no exchanges.

        Upstream's `equilibrate` calls `_mix_replicas` on every equilibration iteration with the
        iteration counter still at 0, which the stride test below would read as an exchange
        iteration -- so every replica would swap during the per-tau relaxation that exists to hold
        it at its own tau. Upstream does not increment the counter, so this contributes nothing to
        the exchange-production time.
        """
        self._equilibrating = True
        try:
            return super().equilibrate(n_iterations, mcmc_moves=mcmc_moves)
        finally:
            self._equilibrating = False

    def _mix_replicas(self):
        """Mix on stride iterations; on every other iteration, mix nothing and PROPOSE nothing.

        Zeroing the matrices matters and is not tidiness. Upstream resets them at the top of its
        own `_mix_replicas`, so returning early without doing so leaves the PREVIOUS mixing
        event's counts in place -- and `_report_iteration_items` then writes those same counts
        again for every skipped iteration. A stride-2 run recorded 23 mixing events where 12
        occurred, and every lifetime acceptance figure built from that history was inflated by
        roughly the stride. Skipped iterations must carry genuine zeros, because the zeros are
        what make a mixing event countable from the stored history.
        """
        if getattr(self, "_equilibrating", False) or self._iteration % self.exchange_stride != 0:
            self._n_accepted_matrix[:, :] = 0
            self._n_proposed_matrix[:, :] = 0
            return self._replica_thermodynamic_states
        return super()._mix_replicas()


class ProjectNeighbourExchangeMixin:
    """Neighbour exchange, with the Metropolis decision performed by THIS REPOSITORY.

    Used only when `swap-neighbors` is explicitly selected. Both methods below replace private
    OpenMMTools methods, so a run using them records `exchange_decision_owner: md-templates`.

    The criterion is not restated here: it calls this repository's own `exchange_log_acceptance`,
    so the ladder, the legacy loop and the tests share one definition rather than agreeing by
    inspection.
    """

    def _mix_neighboring_replicas(self):
        """Upstream's alternating neighbour scheme, with scalar indices and no empty phase."""
        offsets = [offset for offset in (0, 1)
                   if any(True for _ in range(offset, self.n_replicas - 1, 2))]
        offset = offsets[np.random.randint(len(offsets))] if len(offsets) > 1 else offsets[0]
        for state_i in range(offset, self.n_replicas - 1, 2):
            state_j = state_i + 1
            replica_i = int(np.where(self._replica_thermodynamic_states == state_i)[0][0])
            replica_j = int(np.where(self._replica_thermodynamic_states == state_j)[0][0])
            self._attempt_swap(replica_i, replica_j)

    def _attempt_swap(self, replica_i, replica_j):
        """One neighbour swap. `replica_i`/`replica_j` are plain integers here.

        The reduced potentials OpenMMTools stores already carry beta and, under NPT, the pV term.
        Every REST2 rung shares one temperature and one pressure, so the pV contributions cancel
        between the two sides -- which is why this is the same criterion
        `rest2_scaling.exchange_log_acceptance` states, and why it is called rather than restated.
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


class StridedREST2Sampler(StridedMixingMixin, ReplicaExchangeSampler):
    """Stock OpenMMTools exchange, attempted on a stride. OpenMMTools decides every swap."""


class NeighbourREST2Sampler(StridedMixingMixin, ProjectNeighbourExchangeMixin,
                            ReplicaExchangeSampler):
    """Neighbour exchange with the project-owned Metropolis decision. Not the default."""


def sampler_class_for(scheme):
    """The sampler implementing `scheme`, and nothing more clever than a lookup."""
    if scheme == "swap-neighbors":
        require_tested_openmmtools()
        return NeighbourREST2Sampler
    if scheme == "swap-all":
        return StridedREST2Sampler
    raise ValueError(
        f"unknown replica mixing scheme {scheme!r}; supported: {sorted(EXCHANGE_OWNERSHIP)}")
