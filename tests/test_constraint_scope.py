"""What `constraints: HBonds` actually constrains, asserted against a built System.

WHY THIS EXISTS AS A TEST AND NOT ONLY AS A PARAGRAPH

    `docs/scientific-defaults.md` section 11.1 argues that 2 fs is the right default because the
    X-H stretches are frozen and the fastest REMAINING motions are the hydrogen bond-angle
    vibrations at roughly 10 fs. That argument is only sound if the angles really are unconstrained
    -- if `HBonds` ever started freezing H-X-H, the stated justification for the timestep would
    quietly stop applying while every configuration file still said `HBonds` and 2 fs.

    So the scope is pinned here: lengths to hydrogen yes, angles never. A prose claim about
    dynamics that nothing checks is a claim with a shelf life.

WHAT IS NOT ASSERTED

    Which algorithm OpenMM picks. CCMA and SETTLE are OpenMM's internal choice, made per platform
    from the constraint topology, and MD-tools neither selects nor can select them -- so a test
    here would be testing OpenMM, and would break on an OpenMM change that is none of this
    repository's business. What IS asserted is the thing that decides which algorithm applies and
    which this repository does control: whether a rigid three-site cluster exists at all.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow


def _built(tmp_path, constraints="HBonds", solvent="GBn2"):
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "sys.config"
    config.write_text(yaml.safe_dump({
        "solvent": {"model": solvent},
        "constraints": {"type": constraints},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    from openmm import XmlSerializer
    from openmm.app import PDBFile

    system = XmlSerializer.deserialize((tmp_path / "built.xml").read_text(encoding="utf-8"))
    topology = PDBFile(str(tmp_path / "built.pdb")).topology
    elements = [a.element.symbol for a in topology.atoms()]
    return system, elements, done.stdout


def test_hbonds_constrains_every_x_h_length_and_only_those(tmp_path):
    """One constraint per hydrogen, each involving that hydrogen. Nothing else."""
    system, elements, _ = _built(tmp_path)
    hydrogens = elements.count("H")
    assert system.getNumConstraints() == hydrogens, (
        f"{system.getNumConstraints()} constraints for {hydrogens} hydrogens")

    constrained = []
    for index in range(system.getNumConstraints()):
        a, b, _distance = system.getConstraintParameters(index)
        assert elements[a] == "H" or elements[b] == "H", (
            f"constraint {index} joins {elements[a]}-{elements[b]}, neither a hydrogen")
        constrained.append(frozenset((a, b)))
    assert len(set(constrained)) == len(constrained), "a constraint is duplicated"


def test_no_angle_is_constrained_and_the_hydrogen_angles_stay_flexible(tmp_path):
    """THE claim section 11.1's timestep argument rests on.

    Angles involving hydrogen must remain ordinary harmonic terms. If they were constrained the
    fastest remaining motion would no longer be the ~10 fs hydrogen angle bend, and the stated
    reason for 2 fs would not apply -- silently, with the configuration unchanged.
    """
    from openmm import HarmonicAngleForce

    system, elements, _ = _built(tmp_path)
    angles = [f for f in (system.getForce(i) for i in range(system.getNumForces()))
              if isinstance(f, HarmonicAngleForce)]
    assert angles, "the System carries no HarmonicAngleForce at all"

    total = with_hydrogen = 0
    constrained_pairs = set()
    for index in range(system.getNumConstraints()):
        a, b, _d = system.getConstraintParameters(index)
        constrained_pairs.add(frozenset((a, b)))

    for force in angles:
        for index in range(force.getNumAngles()):
            i, j, k, _theta, magnitude = force.getAngleParameters(index)
            total += 1
            if any(elements[x] == "H" for x in (i, j, k)):
                with_hydrogen += 1
                # The 1-3 distance across the angle -- what constraining the ANGLE would freeze.
                assert frozenset((i, k)) not in constrained_pairs, (
                    f"the 1-3 distance {i}-{k} of an angle involving hydrogen is constrained; "
                    f"that is an angle constraint by another name")
            assert magnitude.value_in_unit(magnitude.unit) >= 0.0

    assert with_hydrogen > 0, "no angle involves a hydrogen; the check would be vacuous"
    assert total > with_hydrogen, "every angle involves hydrogen; the fixture is degenerate"


def test_implicit_solvent_has_no_rigid_water_cluster_for_settle(tmp_path):
    """No water, so no rigid three-site cluster, so SETTLE cannot apply.

    Asserted through the topology rather than by naming an algorithm: what this repository
    controls is whether such a cluster exists, and OpenMM's choice follows from that.
    """
    system, elements, stdout = _built(tmp_path, solvent="GBn2")

    # A SETTLE cluster is a rigid triangle: a central atom with two constrained bonds AND the
    # constrained distance between the two outer atoms. Count any such triangle.
    pairs = set()
    for index in range(system.getNumConstraints()):
        a, b, _d = system.getConstraintParameters(index)
        pairs.add(frozenset((int(a), int(b))))
    triangles = 0
    for pair in pairs:
        a, b = tuple(pair)
        partners_a = {tuple(p - {a})[0] for p in pairs if a in p and p != pair}
        for c in partners_a:
            if frozenset((b, c)) in pairs:
                triangles += 1
    assert triangles == 0, (
        f"{triangles} rigid three-atom constraint cluster(s) in an implicit-solvent System; "
        f"there is no water here and nothing should be a rigid triangle")

    # And the build says so, resolving rigid_water to false rather than echoing the request.
    assert "implicit" in stdout.lower()


def test_allbonds_adds_heavy_atom_constraints_under_explicit_solvent(tmp_path):
    """`AllBonds` must actually differ from `HBonds`, and differ in the documented direction.

    Explicit solvent, deliberately: this is the route where `constraints.type` is honoured. Under
    implicit solvent the builder hard-codes `app.HBonds` and the setting has no effect -- see the
    known-issue note in `docs/scientific-defaults.md` section 11.2. Asserting the difference here
    keeps this test about the CONTRACT of the two settings rather than about that defect.
    """
    hbonds, elements, _ = _built(tmp_path / "hbonds", constraints="HBonds", solvent="TIP3P")
    allbonds, _elements, _ = _built(tmp_path / "allbonds", constraints="AllBonds",
                                    solvent="TIP3P")

    def heavy_heavy(system):
        count = 0
        for index in range(system.getNumConstraints()):
            a, b, _d = system.getConstraintParameters(index)
            if elements[a] != "H" and elements[b] != "H":
                count += 1
        return count

    assert heavy_heavy(hbonds) == 0, "HBonds constrained a heavy-heavy bond"
    assert heavy_heavy(allbonds) > 0, "AllBonds constrained no heavy-heavy bond"
    assert allbonds.getNumConstraints() > hbonds.getNumConstraints()
