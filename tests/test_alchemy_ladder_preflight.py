"""The lambda ladder's preflight: the flag rules are the INVERSE of a REST2 ladder's.

A REST2 rung IS a file, so that ladder reads `-s` only from its group file and refuses `-s` on the
command line. A lambda rung is a set of Context parameter values on ONE System, so this ladder
requires `-s` and refuses a group file. Both refusals are in `shared-contracts.md` section 5.

Nothing here creates output: every test asserts a refusal, or asserts a prepared result, and the
run directory is a `tmp_path` that stays empty. That is the invariant the preflight exists for --
"nothing is written until the whole preflight passes" -- so each refusal test also checks that no
file appeared.
"""
from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")

from openmm import XmlSerializer, app, unit  # noqa: E402
import numpy as np  # noqa: E402

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import build_hamiltonian  # noqa: E402
from md_tools.run.preflight import PreflightError, preflight_lambda_ladder  # noqa: E402

LAMBDAS = [0.0, 0.33, 0.66, 1.0]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The hybrid System and a topology on disk: what `-s` and `-p` would name."""
    work = tmp_path_factory.mktemp("lambda-ladder")
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    h = build_hamiltonian(sa, sb, a, b)
    topology = app.Topology()
    chain = topology.addChain()
    # ONE RESIDUE PER ATOM, and a distinct name for each: 102 atoms called "C" in one residue are
    # duplicates to the PDB reader, which collapsed them to a single atom and made the preflight
    # refuse a topology/System mismatch that was entirely the fixture's doing.
    for i in range(h.system.getNumParticles()):
        residue = topology.addResidue(f"R{i:03d}", chain)
        topology.addAtom(f"C{i:03d}", app.Element.getBySymbol("C"), residue)
    vectors = np.array([[v.x, v.y, v.z] for v in h.system.getDefaultPeriodicBoxVectors()])
    topology.setPeriodicBoxVectors(vectors * unit.nanometer)
    (work / "built.xml").write_text(XmlSerializer.serialize(h.system))
    with open(work / "built.pdb", "w") as handle:
        app.PDBFile.writeFile(topology, np.asarray(x) * unit.nanometer, handle)
    return work, h


def _ladder(lambdas=LAMBDAS, hamiltonian=None):
    return {"rungs": [{"index": i, "state": dict.fromkeys(
        ("lambda_electrostatics", "lambda_sterics", "lambda_bonded"), v), "tau": 0.0}
        for i, v in enumerate(lambdas)],
        "hamiltonian": (hamiltonian.record if hamiltonian is not None else None)}


def _call(work, tmp_path, **overrides):
    kw = dict(topology=str(work / "built.pdb"), system=str(work / "built.xml"),
              replicas=len(LAMBDAS), ladder=_ladder(),
              trajectory=str(tmp_path / "remd.nc"), restart=str(tmp_path / "restart.json"),
              checkpoint=str(tmp_path / "checkpoint"), output=str(tmp_path / "remd.out"),
              log=str(tmp_path / "remd.log"), cpu=True)
    kw.update(overrides)
    return preflight_lambda_ladder(**kw)


def _nothing_written(path):
    assert sorted(p.name for p in path.iterdir()) == [], "a refused preflight wrote into -odir"


# -- the two contract refusals ---------------------------------------------------------------------


def test_a_group_file_is_refused_even_when_every_line_names_the_same_system(built, tmp_path):
    work, _ = built
    same = [str(work / "built.xml")] * len(LAMBDAS)
    with pytest.raises(PreflightError, match="no group file"):
        _call(work, tmp_path, groupfile=same)
    _nothing_written(tmp_path)


def test_the_one_system_is_required_because_a_rung_is_not_a_file(built, tmp_path):
    work, _ = built
    with pytest.raises(PreflightError, match=r"-s is required"):
        _call(work, tmp_path, system=None)
    _nothing_written(tmp_path)


def test_flags_of_another_protocol_are_refused_by_name(built, tmp_path):
    work, _ = built
    with pytest.raises(PreflightError, match="-s2|second end state|AIS"):
        _call(work, tmp_path, system2=str(work / "built.xml"))
    _nothing_written(tmp_path)


# -- the rung schedule -----------------------------------------------------------------------------


def test_a_rung_out_of_position_is_refused(built, tmp_path):
    work, h = built
    ladder = _ladder()
    ladder["rungs"][2]["index"] = 9
    with pytest.raises(PreflightError, match="addressed by its index"):
        _call(work, tmp_path, ladder=ladder)
    _nothing_written(tmp_path)


def test_a_rung_count_that_disagrees_with_the_launch_is_refused(built, tmp_path):
    work, _ = built
    with pytest.raises(PreflightError, match="the same ladder"):
        _call(work, tmp_path, replicas=len(LAMBDAS) + 1)
    _nothing_written(tmp_path)


def test_a_rung_state_that_could_not_be_set_is_refused_here_not_at_propagation(built, tmp_path):
    work, _ = built
    ladder = _ladder()
    ladder["rungs"][1]["state"]["lambda_sterics"] = 1.5
    with pytest.raises(PreflightError, match="rung 1.*must be a number in"):
        _call(work, tmp_path, ladder=ladder)
    _nothing_written(tmp_path)


def test_an_empty_schedule_is_refused(built, tmp_path):
    work, _ = built
    with pytest.raises(PreflightError, match="describes no rungs"):
        _call(work, tmp_path, ladder={"rungs": []}, replicas=0)
    _nothing_written(tmp_path)


# -- what a passing preflight carries ---------------------------------------------------------------


def test_a_passing_preflight_carries_one_system_per_rung_and_it_is_one_object(built, tmp_path):
    work, h = built
    prepared = _call(work, tmp_path, ladder=_ladder(hamiltonian=h))
    assert prepared.replicas == len(LAMBDAS)
    assert len(prepared.rung_systems) == len(LAMBDAS)
    first = prepared.rung_systems[0]
    assert all(s is first for s in prepared.rung_systems), "K copies would be K chances to differ"
    assert [s["lambda_sterics"] for s in prepared.rung_states] == LAMBDAS
    # the derived parameters are carried, because they are what the forces read
    assert prepared.rung_states[1]["mdt_alchemy_qscale_b"] == pytest.approx(0.33 ** 0.5)
    assert prepared.hamiltonian_record is h.record
    assert prepared.notes["groupfile"] is None
    assert prepared.platform_name == "CPU"
    _nothing_written(tmp_path)
