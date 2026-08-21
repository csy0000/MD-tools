"""Building the same bundle twice must produce the same bundle.

Every hash, relocation check and provenance comparison in this package rests on this. "Which value
did I choose and which did the package choose?" cannot be answered by comparing two bundles if
rebuilding the same one gives a different answer each time.

It did not hold. Two builds from byte-identical configuration produced different `system.xml`,
different `topology.pdb` and, in the case that surfaced it, a different NUMBER OF WATERS -- the
total mass differed by exactly 18.016 amu. Two independent causes:

1. `Modeller.addHydrogens` places each new hydrogen from a random direction drawn from Python's
   global `random`, which nothing seeded. Repeat calls moved hydrogens by up to **0.18 nm**. The box
   is sized from the solute's extent, so that changed the box by ~0.04 A, which was enough to fit
   one more water.
2. The relaxation `addHydrogens` runs afterwards is order-dependent on a threaded platform, leaving
   ~1e-4 nm of drift even once the RNG is fixed. Pinned to Reference, which is single-threaded.
3. `Modeller.addSolvent` chooses which waters become ions from the same global `random`, and
   OpenMM 8.5.2 exposes no `randomSeed` parameter for it.

All three are seeded or pinned from the run's master seed, so they are part of the seed map rather
than unrecorded sources of variation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_GEN = REPO_ROOT / "MD_system_gen.py"
ALANINE_PDB = (REPO_ROOT / "src" / "md_templates" / "openmm" / "manifests" / "systems"
               / "ace_ala_nme.pdb")
EXPLICIT_CONFIG = REPO_ROOT / "test" / "ala" / "cMD" / "explicit" / "system_config.json"

#: The files that define the Hamiltonian and the coordinates it starts from.
DEFINING = ("system.xml", "topology.pdb", "initial_state.xml", "forcefield.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(out: Path) -> Path:
    result = subprocess.run(
        [sys.executable, str(SYSTEM_GEN), "-i", str(ALANINE_PDB), "-o", str(out),
         "--config", str(EXPLICIT_CONFIG)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_hydrogen_placement_is_seeded_and_reproducible():
    """The larger of the two causes, isolated: 0.18 nm of random displacement per call."""
    import random

    import numpy as np
    from openmm import Platform, app, unit

    forcefield = app.ForceField("amber19/protein.ff19SB.xml", "amber19/opc.xml")
    reference = Platform.getPlatformByName("Reference")

    def place(seed):
        random.seed(seed)
        pdb = app.PDBFile(str(ALANINE_PDB))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.delete([a for a in modeller.topology.atoms()
                         if a.element == app.element.hydrogen])
        modeller.addHydrogens(forcefield, pH=7.0, platform=reference)
        return np.array(modeller.positions.value_in_unit(unit.nanometer))

    assert np.array_equal(place(7), place(7)), "seeded hydrogen placement still moved"
    # and the seed genuinely drives it, so this is pinning something real
    assert not np.array_equal(place(7), place(8))


@pytest.mark.slow
def test_two_builds_of_one_system_are_byte_identical(tmp_path):
    """The property that matters, end to end through the public front end."""
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    differing = [name for name in DEFINING
                 if (first / name).exists() and _digest(first / name) != _digest(second / name)]
    assert not differing, f"rebuilding the same system changed: {differing}"


@pytest.mark.slow
def test_the_rebuild_has_the_same_water_count_and_box(tmp_path):
    """The symptom that made this visible, asserted directly rather than through a hash.

    A hash comparison says "different" without saying how much. One water molecule different is a
    different system, and it is worth failing with that number in the message.
    """
    first = json.loads((_build(tmp_path / "a") / "forcefield.json").read_text())
    second = json.loads((_build(tmp_path / "b") / "forcefield.json").read_text())

    for block, key in (("system", "total_mass_amu"), ("system", "n_particles"),
                       ("system", "n_waters")):
        left = (first.get(block) or {}).get(key)
        right = (second.get(block) or {}).get(key)
        if left is not None or right is not None:
            assert left == right, f"{block}.{key} changed between builds: {left} vs {right}"
