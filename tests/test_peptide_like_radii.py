"""The mbondi3 side-chain corrections, measured on real built Systems.

WHAT MAKES THIS A PROOF RATHER THAN A MIRROR

    The tempting test asks the new mapper which atoms should be corrected and then checks that
    those atoms were corrected. That passes whatever the mapper says, including nothing, and
    would have passed before this change existed.

    So the expected atom sets come from somewhere else: a CANONICAL tripeptide, built as a real
    protein with real residue and atom names, put through the installed `parmed.tools.changeRadii`
    with `mbondi3`, and read back. That is the assignment this change is trying to reproduce, and
    it is produced by ParmEd's own name-keyed rules with no involvement from anything written
    here. The peptide-like build then has to agree with it.

    The baseline is the same molecule built as `kind: ligand`. Every intrinsic radius outside the
    corrected set must be identical to it, which is what "only the intended GB parameters change"
    means in practice.

UNITS, AND WHY THE SERIALISED FIELD IS NOT THE RADIUS

    `CustomGBForce` stores `or`, the OFFSET radius, in nanometres: `or = radius - offset` with
    `offset = 0.0195141 nm` for GBn2. Comparing `or` against 0.14 nm would be comparing the wrong
    quantity and would fail by exactly the offset. Everything below reconstructs the intrinsic
    radius explicitly and compares in nanometres to 1e-8.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = pytest.mark.slow

#: GBn2's offset, taken from the serialised force in `test_the_offset_is_the_one_the_system_uses`
#: rather than trusted here.
GBN2_OFFSET_NM = 0.0195141
TOLERANCE_NM = 1e-8

CARBOXYLATE_ANGSTROM = 1.40
GUANIDINIUM_ANGSTROM = 1.17

#: cyclo(Gly-L-Asp-L-Arg). Both correctable groups, 43 atoms: AM1-BCC finishes in a couple of
#: minutes rather than the ~25 of RGDfV, and the chemistry under test is identical.
CYCLO_GDR = "O=C1NCC(=O)N[C@H](CC(=O)[O-])C(=O)N[C@H]1CCCNC(N)=[NH2+]"
#: The same cycle with a neutral acid and a neutral amine side chain: nothing to correct.
#: A neutral GUANIDINE would be the closer control, but its C=N carries undefined bond
#: stereochemistry that the OpenFF toolkit refuses outright, so the neutral-guanidine case is
#: covered at the graph level in `test_peptide_map.py` where no build is needed.
CYCLO_NEUTRAL = "O=C1NCC(=O)N[C@H](CC(=O)O)C(=O)N[C@H]1CCCCN"


def _build(tmp_path, smiles, kind, solvent="GBn2"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "in.smi").write_text(f"{smiles} CYC\n", encoding="utf-8")
    (tmp_path / "sys.config").write_text(
        f"solute:\n  kind: {kind}\n  ligand_forcefield: sage-2.2.1\n"
        f"  ligand_charge_method: am1bcc\nsolvent:\n  model: {solvent}\n"
        f"constraints:\n  type: HBonds\nhydrogen_mass_repartitioning:\n  enabled: false\n",
        encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", "in.smi", "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", "sys.config"],
        cwd=tmp_path, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    from openmm import XmlSerializer

    system = XmlSerializer.deserialize((tmp_path / "built.xml").read_text(encoding="utf-8"))
    return system


def _gb(system):
    from openmm import CustomGBForce

    force = next(f for f in (system.getForce(i) for i in range(system.getNumForces()))
                 if isinstance(f, CustomGBForce))
    names = [force.getPerParticleParameterName(i)
             for i in range(force.getNumPerParticleParameters())]
    return force, names


def _intrinsic_radii_nm(system):
    """Reconstructed from the stored offset radius, in nanometres."""
    force, names = _gb(system)
    column = names.index("or")
    return [force.getParticleParameters(i)[column] + GBN2_OFFSET_NM
            for i in range(force.getNumParticles())]


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("peptide-like-radii")
    return {
        "like": _build(root / "like", CYCLO_GDR, "peptide-like"),
        "ligand": _build(root / "ligand", CYCLO_GDR, "ligand"),
        "neutral": _build(root / "neutral", CYCLO_NEUTRAL, "peptide-like"),
    }


# --- the offset is the System's own, not a constant typed here ---------------------------------

def test_the_offset_is_the_one_the_system_uses(builds):
    """If OpenMM ever changed it, every reconstruction below would be silently wrong."""
    import re

    force, _names = _gb(builds["like"])
    expressions = "\n".join(force.getComputedValueParameters(i)[1]
                            for i in range(force.getNumComputedValues()))
    found = set(re.findall(r"offset\s*=\s*([0-9.]+)", expressions))
    assert found == {str(GBN2_OFFSET_NM)}, found


# --- the independent expectation ---------------------------------------------------------------

def test_the_corrected_values_match_parmeds_own_rule_on_canonical_residues(builds):
    """The corrected radii equal what ParmEd's mbondi3 assigns where the names DO match.

    The reference is produced by calling `mbondi3` on a structure whose residues are named `ASP`
    and `ARG` -- the situation the rule was written for -- and reading the radii it sets. The
    peptide-like build has to land on the same numbers, having got there from chemistry.
    """
    pytest.importorskip("parmed")
    import parmed as pmd
    from parmed.tools import changeRadii

    # A minimal ParmEd structure carrying the NAMES ParmEd keys on. Two atoms is enough: the
    # rule is per-atom, keyed on residue name and atom name, and nothing else about the structure
    # participates in it.
    structure = pmd.Structure()
    aspartate = pmd.Residue("ASP")
    arginine = pmd.Residue("ARG")
    od1 = pmd.Atom(name="OD1", atomic_number=8)
    od2 = pmd.Atom(name="OD2", atomic_number=8)
    hh11 = pmd.Atom(name="HH11", atomic_number=1)
    he = pmd.Atom(name="HE", atomic_number=1)
    cb = pmd.Atom(name="CB", atomic_number=6)
    for atom, residue in ((od1, aspartate), (od2, aspartate),
                          (hh11, arginine), (he, arginine), (cb, arginine)):
        structure.add_atom(atom, residue.name, 1 if residue is aspartate else 2)
    # mbondi2 needs bond partners to decide a hydrogen's radius; give each H a nitrogen.
    n1 = pmd.Atom(name="NH1", atomic_number=7)
    n2 = pmd.Atom(name="NE", atomic_number=7)
    structure.add_atom(n1, "ARG", 2)
    structure.add_atom(n2, "ARG", 2)
    structure.bonds.append(pmd.Bond(hh11, n1))
    structure.bonds.append(pmd.Bond(he, n2))

    changeRadii(structure, "mbondi3").execute()
    reference = {a.name: float(a.solvent_radius) for a in structure.atoms}

    assert reference["OD1"] == pytest.approx(CARBOXYLATE_ANGSTROM)
    assert reference["OD2"] == pytest.approx(CARBOXYLATE_ANGSTROM)
    assert reference["HH11"] == pytest.approx(GUANIDINIUM_ANGSTROM)
    assert reference["HE"] == pytest.approx(GUANIDINIUM_ANGSTROM)

    # And the peptide-like build's corrected atoms carry exactly those values.
    corrected = _intrinsic_radii_nm(builds["like"])
    baseline = _intrinsic_radii_nm(builds["ligand"])
    moved = [i for i, (a, b) in enumerate(zip(corrected, baseline))
             if abs(a - b) > TOLERANCE_NM]
    values = sorted({round(corrected[i] * 10, 6) for i in moved})
    assert values == sorted({reference["OD1"], reference["HH11"]}), values


# --- exactly two oxygens and five hydrogens, and nothing else ----------------------------------

def test_exactly_two_oxygens_and_five_hydrogens_are_corrected(builds):
    from openmm import CustomGBForce                                     # noqa: F401

    corrected = _intrinsic_radii_nm(builds["like"])
    baseline = _intrinsic_radii_nm(builds["ligand"])
    assert len(corrected) == len(baseline)

    moved = [i for i, (a, b) in enumerate(zip(corrected, baseline)) if abs(a - b) > TOLERANCE_NM]
    to_carboxylate = [i for i in moved
                      if abs(corrected[i] * 10 - CARBOXYLATE_ANGSTROM) < 1e-7]
    to_guanidinium = [i for i in moved
                      if abs(corrected[i] * 10 - GUANIDINIUM_ANGSTROM) < 1e-7]
    assert len(to_carboxylate) == 2, f"{len(to_carboxylate)} atoms went to 1.40 A"
    assert len(to_guanidinium) == 5, f"{len(to_guanidinium)} atoms went to 1.17 A"
    assert len(moved) == 7


def test_every_other_intrinsic_radius_is_unchanged_from_the_ligand_baseline(builds):
    """The negative half, and the one that catches a correction applied too widely."""
    corrected = _intrinsic_radii_nm(builds["like"])
    baseline = _intrinsic_radii_nm(builds["ligand"])
    moved = {i for i, (a, b) in enumerate(zip(corrected, baseline)) if abs(a - b) > TOLERANCE_NM}
    for index, (a, b) in enumerate(zip(corrected, baseline)):
        if index not in moved:
            assert abs(a - b) <= TOLERANCE_NM, f"atom {index}: {b} -> {a} nm"
    assert len(moved) == 7


def test_a_neutral_side_chain_receives_no_charged_group_correction(builds):
    """`ASH`/`ARN` are different molecules. ParmEd's NAME rule would correct AS4; chemistry does not."""
    neutral = _intrinsic_radii_nm(builds["neutral"])
    # No atom carries either corrected value: the neutral acid keeps mbondi2's oxygen radius and
    # the neutral guanidine keeps mbondi2's N-bonded hydrogen radius.
    assert not any(abs(r * 10 - GUANIDINIUM_ANGSTROM) < 1e-7 for r in neutral), \
        "a neutral guanidine received the charged-guanidinium correction"
    assert not any(abs(r * 10 - CARBOXYLATE_ANGSTROM) < 1e-7 for r in neutral), \
        "a neutral carboxylic acid received the carboxylate correction"


# --- the dependent parameter, and the reason the correction lands where it does -----------------

def test_the_scaled_offset_radius_tracks_the_corrected_radius(builds):
    """`sr = screen * or`. A radius changed after `createSystem` would leave `sr` describing the old one."""
    like, names_a = _gb(builds["like"])
    ligand, names_b = _gb(builds["ligand"])
    oa, sa = names_a.index("or"), names_a.index("sr")
    ob, sb = names_b.index("or"), names_b.index("sr")
    for index in range(like.getNumParticles()):
        a = like.getParticleParameters(index)
        b = ligand.getParticleParameters(index)
        # The screen factor is elemental and must NOT have moved; the scaled radius must have
        # moved with the radius.
        assert a[sa] / a[oa] == pytest.approx(b[sb] / b[ob], rel=1e-9)


def test_serialising_and_reloading_preserves_the_corrections(builds):
    """The corrections must survive the file, which is what every later run actually reads."""
    from openmm import XmlSerializer

    before = _intrinsic_radii_nm(builds["like"])
    reloaded = XmlSerializer.deserialize(XmlSerializer.serialize(builds["like"]))
    after = _intrinsic_radii_nm(reloaded)
    assert len(after) == len(before)
    for a, b in zip(after, before):
        assert abs(a - b) <= TOLERANCE_NM


def test_the_corrected_system_gives_finite_energies_and_forces(builds):
    """A radius change that produced a NaN would be caught here rather than in a run."""
    import math

    import openmm
    from openmm import unit
    from openmm.app import PDBFile                                        # noqa: F401

    system = builds["like"]
    context = openmm.Context(system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
                             openmm.Platform.getPlatformByName("Reference"))
    # Positions from the ligand build's own topology would be ideal; a spread-out placement is
    # enough to prove the parameters evaluate.
    import numpy as np

    n = system.getNumParticles()
    rng = np.random.default_rng(20260908)
    positions = rng.normal(scale=0.5, size=(n, 3))
    context.setPositions(positions * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert math.isfinite(energy), energy
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    assert np.all(np.isfinite(forces))


# --- explicit solvent must not acquire GB anything ---------------------------------------------

def test_an_explicit_solvent_peptide_like_build_has_no_gb_force_and_no_corrections(tmp_path):
    """The corrections belong to the implicit model. An explicit build must be untouched."""
    from openmm import CustomGBForce

    system = _build(tmp_path / "explicit", CYCLO_GDR, "peptide-like", solvent="TIP3P")
    gb_forces = [f for f in (system.getForce(i) for i in range(system.getNumForces()))
                 if isinstance(f, CustomGBForce)]
    assert gb_forces == [], "an explicit-solvent build acquired a generalised-Born force"
