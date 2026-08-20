"""Every seed in a run, derived from one master seed by one documented algorithm.

Randomness that is not reproducible is not evidence. Before this module the repository had three
different answers to "where does this seed come from?": a `master + offset` ladder covering four
purposes, a literal `20260820` compiled into the stage runner, and, for anything not in either list,
whatever OpenMM chose. The literal was the worst of the three, because two runs that differed in
every scientific input still shared an integrator stream and looked more alike than they were.

## The derivation

    seed(purpose) = 1 + (sha256("md-templates/seed/v1|<master>|<purpose>") mod (2**31 - 2))

Properties that matter, and why each is required:

* **Deterministic across processes and machines.** SHA-256, never Python's `hash()`, which is
  randomised per process by PEP 456 -- a seed derived from it would differ between two runs of the
  same command.
* **Nonzero.** OpenMM treats seed 0 as "choose randomly", so a derivation that can produce 0 has a
  1-in-2-billion chance of silently becoming irreproducible.
* **Within OpenMM's range.** Seeds are passed to `LangevinMiddleIntegrator`,
  `MonteCarloBarostat` and `setVelocitiesToTemperature`, which take a 32-bit signed value.
* **Uncorrelated between purposes and between neighbouring masters.** `master + offset` failed this:
  master 1000 stage 1 and master 1001 stage 0 were the same number, so two runs a user thought were
  independent shared a stream.

The purpose string is part of the hash, so a new purpose can be added without disturbing any
existing seed, and no two purposes collide unless SHA-256 does.

`DERIVATION_VERSION` is persisted alongside the seed map. Changing the algorithm requires bumping
it, because a run reproduced under a different derivation is not the same run.

## The legacy rule, and why it survives

`RandomnessSpec.resolve()` derives four seeds as `master_seed + index_in_STAGE_ORDER`. That rule is
frozen by `tests/goldens/seed_derivation.json` as a compatibility contract, and its documented
purpose is that a migrated input "reproduces its existing trajectories rather than merely a valid
one". Replacing it would silently change the trajectory of every existing configuration.

So it stays, and it is named here -- `LEGACY_DERIVATION_ALGORITHM` -- rather than left as an
anonymous `+ offset` in the spec layer. This module is the single *owner* of seed derivation; there
are two algorithms because there are two compatibility regimes, and both are versioned, named and
persisted with the values they produce. Which regime produced a seed is always recoverable from the
record.

New purposes use v1. Nothing new is added to the legacy rule.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "DERIVATION_VERSION",
    "DERIVATION_ALGORITHM",
    "DEFAULT_MASTER_SEED",
    "LEGACY_DERIVATION_ALGORITHM",
    "derive_seed",
    "seed_map",
    "stage_purpose",
    "replica_purpose",
]

#: Bumped whenever the algorithm below changes. Persisted with every seed map.
DERIVATION_VERSION = 1

#: Human-readable identifier of the algorithm, persisted for the same reason.
DERIVATION_ALGORITHM = "sha256/md-templates-seed-v1"

#: The frozen `master_seed + index` rule of `RandomnessSpec.resolve()`. Named so that a persisted
#: seed always says which regime produced it. Not used for any new purpose.
LEGACY_DERIVATION_ALGORITHM = "legacy/master-plus-stage-index-v0"

#: Used when a front end has no randomness block of its own. Matches `RandomnessSpec.master_seed`
#: so the two entry points do not disagree about what "the default run" means.
DEFAULT_MASTER_SEED = 20260814

_PREFIX = "md-templates/seed/v1"

#: OpenMM seeds are 32-bit signed and 0 means "pick randomly", so the usable range is [1, 2**31-1].
_MODULUS = 2 ** 31 - 2


def derive_seed(master_seed: int, purpose: str) -> int:
    """Return the seed for one named purpose. Deterministic, nonzero, and 32-bit safe."""
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError(f"master seed must be an integer, got {master_seed!r}")
    if not purpose:
        raise ValueError("a seed purpose must be a non-empty string")
    digest = hashlib.sha256(f"{_PREFIX}|{master_seed}|{purpose}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % _MODULUS


def stage_purpose(stage: str, role: str) -> str:
    """Purpose string for a per-stage seed, e.g. `stage/eq_nvt/integrator`.

    `role` separates the streams *within* a stage -- an integrator and a barostat that shared a
    seed would be correlated in a way nobody would think to look for.
    """
    return f"stage/{stage}/{role}"


def replica_purpose(index: int, role: str) -> str:
    """Purpose string for a per-replica seed, e.g. `replica/03/integrator`.

    Zero-padded so the string is stable under any formatting of the index.
    """
    return f"replica/{int(index):02d}/{role}"


def seed_map(master_seed: int, purposes) -> dict:
    """Derive a full seed map, with the derivation recorded beside the values.

    The purposes are persisted with their seeds because a bare list of integers cannot be checked:
    a reader has to be able to recompute the map from the master seed and see the same numbers.
    """
    seeds = {purpose: derive_seed(master_seed, purpose) for purpose in purposes}
    distinct = len(set(seeds.values()))
    if distinct != len(seeds):
        # Not reachable short of a SHA-256 collision, but a silent duplicate would correlate two
        # streams that must be independent, so it is checked rather than assumed.
        raise RuntimeError(
            f"seed derivation produced {len(seeds) - distinct} duplicate seed(s) for master "
            f"{master_seed}; refusing to run with correlated random streams"
        )
    return {
        "master_seed": int(master_seed),
        "derivation": DERIVATION_ALGORITHM,
        "derivation_version": DERIVATION_VERSION,
        "seeds": seeds,
    }
