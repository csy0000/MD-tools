"""The lambda ladder's runtime: rungs, refusals, and the exchange terms it produces.

A3b (`docs/development/0.7.0/lambda-exchange-design.md`). Reference platform, float64.

What these cover that the Hamiltonian tests cannot: a LADDER is several Contexts of ONE System held
at different parameter values, and every defect specific to that shape is a defect nothing in
`test_alchemy_hamiltonian_*` can see -- Contexts that all sit at the same state, a rung addressed
by lambda instead of index, a group file quietly accepted because its K paths happen to agree.

Two of these tests exist because of a named trap rather than a general worry:

* `test_a_group_file_is_refused_even_when_every_path_is_identical` -- the tempting implementation
  refuses only when the paths DIFFER, and it passes every test anybody would naturally write.
* `test_the_contexts_are_not_all_in_the_same_state` -- the failure mode of the engine hook is
  silent: K Contexts at one state propagate, exchange with acceptance 1, and produce a ladder-
  shaped output describing one state sampled K times.
"""
from __future__ import annotations

import pytest

openmm = pytest.importorskip("openmm")

from tests import alchemy_s3_fixture as fx  # noqa: E402
from md_tools.alchemy.hamiltonian import build_hamiltonian  # noqa: E402
from md_tools.alchemy.ladder import (LambdaLadderError, LambdaLadderProtocol,  # noqa: E402
                                     LambdaRung, refuse_group_file, rungs_from_schedule)

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def hamiltonian():
    sa, sb, a, b, x = fx.build(True, dispersion=True)
    return build_hamiltonian(sa, sb, a, b), x


def _protocol(h, **kw):
    kw.setdefault("rungs", rungs_from_schedule(LAMBDAS))
    return LambdaLadderProtocol(hamiltonian=h, temperature_k=300.0, timestep_fs=2.0,
                                exchange_interval_ps=1.0, number_of_exchanges=10, **kw)


def _context(system, x):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    c.setPositions(x)
    return c


# -- the rung is the index, and its content is (lambda, tau) ---------------------------------------


def test_a_rung_records_its_index_its_state_and_its_derived_parameters(hamiltonian):
    h, _ = hamiltonian
    p = _protocol(h)
    records = p.state_records()
    assert [r["index"] for r in records] == list(range(len(LAMBDAS)))
    assert [r["state"]["lambda_sterics"] for r in records] == LAMBDAS
    assert all(r["tau"] == 0.0 for r in records), "A3b is the tau = 0 case, recorded explicitly"
    # the derived parameters are what the forces read, so they are part of the rung's identity
    first = records[1]["context_parameters"]
    assert first["mdt_alchemy_qscale_a"] == pytest.approx((1.0 - 0.25) ** 0.5)
    assert records[0]["trajectory"] != records[1]["trajectory"]


def test_a_rung_out_of_position_is_refused_because_the_index_is_the_address(hamiltonian):
    h, _ = hamiltonian
    rungs = list(rungs_from_schedule(LAMBDAS))
    rungs[2] = LambdaRung(index=7, state=rungs[2].public_state)
    with pytest.raises(LambdaLadderError, match="addressed by its index"):
        _protocol(h, rungs=tuple(rungs))


def test_two_rungs_at_one_state_are_refused(hamiltonian):
    h, _ = hamiltonian
    with pytest.raises(LambdaLadderError, match="same state"):
        _protocol(h, rungs=rungs_from_schedule([0.0, 0.5, 0.5, 1.0]))


def test_a_state_outside_the_unit_interval_is_refused_by_the_hamiltonians_own_check(hamiltonian):
    h, _ = hamiltonian
    with pytest.raises(Exception, match="must be a number in"):
        rungs_from_schedule([0.0, 1.5])


# -- the group file, which does not exist ----------------------------------------------------------


def test_a_group_file_is_refused_even_when_every_path_is_identical(hamiltonian):
    """THE TRAP, named in shared-contracts section 5: an implementation that refuses only when the
    K paths DIFFER accepts this one, and passes every test anybody would naturally write."""
    h, _ = hamiltonian
    identical = ["build/built.xml"] * len(LAMBDAS)
    with pytest.raises(LambdaLadderError, match="no group file"):
        _protocol(h, group_file=identical)
    with pytest.raises(LambdaLadderError, match="presence is the error"):
        refuse_group_file(identical)
    # and the refusal does not depend on the paths agreeing
    with pytest.raises(LambdaLadderError, match="no group file"):
        refuse_group_file(["a.xml", "b.xml"])
    assert refuse_group_file(None) is None
    assert _protocol(h).describe()["group_file"] is None


def test_a_foreign_base_system_is_refused_rather_than_ignored(hamiltonian):
    h, _ = hamiltonian
    other = openmm.System()
    other.addParticle(1.0)
    with pytest.raises(LambdaLadderError, match="different one"):
        _protocol(h).build_systems(other)
    systems = _protocol(h).build_systems()
    assert len(systems) == len(LAMBDAS)
    assert all(s is h.system for s in systems), "one System, not K copies of it"


# -- the Contexts are the states ------------------------------------------------------------------


def test_the_contexts_are_not_all_in_the_same_state(hamiltonian):
    """`prepare_context` is what makes K Contexts of ONE System into K thermodynamic states.

    Its failure mode is silent: without it every Context sits at whatever the System was serialised
    in, so the rungs propagate identically and every exchange is accepted. So this asserts the
    energies are DISTINCT, not merely that the call did not raise.
    """
    h, x = hamiltonian
    p = _protocol(h)
    energies = []
    for rung in p.rungs:
        c = _context(h.system, x)
        p.prepare_context(rung.index, c)
        p.verify_context(rung.index, c)
        energies.append(c.getState(getEnergy=True).getPotentialEnergy()._value)
    assert len(set(round(e, 6) for e in energies)) == len(energies), energies


def test_the_engine_puts_each_rung_in_its_own_state(hamiltonian):
    """The same property THROUGH `md_tools.remd.ReplicaEngine`, which is what actually builds the
    Contexts at run time.

    The test above exercises `prepare_context` directly and would pass even if nothing ever called
    it. This one fails if the engine's hook is removed -- verified by mutation, not assumed:
    deleting the two-line hook in `engine.py` makes every rung's energy identical here while the
    direct test stays green.
    """
    import numpy as np
    from openmm import app
    from md_tools.remd.engine import Configuration, ReplicaEngine

    h, x = hamiltonian
    p = _protocol(h)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("SYS", chain)
    for _ in range(h.system.getNumParticles()):
        topology.addAtom("C", app.Element.getBySymbol("C"), residue)
    engine = ReplicaEngine(p, p.build_systems(), topology,
                           platform=openmm.Platform.getPlatformByName("Reference"), seed=1)
    box = np.array([[v.x, v.y, v.z] for v in h.system.getDefaultPeriodicBoxVectors()])
    configuration = Configuration(np.asarray(x, dtype=float), np.zeros_like(np.asarray(x)), box)
    for rung in p.rungs:
        engine.set_configuration(rung.index, configuration)
        p.verify_context(rung.index, engine._simulations[rung.index].context)
    energies = [engine.potential_energy(rung.index) for rung in p.rungs]
    assert len(set(round(e, 6) for e in energies)) == len(energies), energies


def test_a_context_in_the_wrong_state_is_refused(hamiltonian):
    h, x = hamiltonian
    p = _protocol(h)
    c = _context(h.system, x)
    p.prepare_context(1, c)
    with pytest.raises(LambdaLadderError, match="claims .* but its Context holds"):
        p.verify_context(2, c)
    # and a derived parameter set by hand is caught too, not only the public three
    p.prepare_context(1, c)
    c.setParameter("mdt_alchemy_qscale_a", 0.123)
    with pytest.raises(LambdaLadderError, match="mdt_alchemy_qscale_a"):
        p.verify_context(1, c)


def test_the_cross_terms_an_exchange_needs_come_from_one_context_each(hamiltonian):
    """u_i(x_i), u_i(x_j), u_j(x_i), u_j(x_j) for neighbouring rungs, each evaluated in the rung's
    own Context, and the acceptance built from them by `md_tools.remd`'s own function.

    Checks the two things that make the numbers meaningful rather than merely present: a cross term
    differs from the diagonal one (otherwise the exchange is a formality), and evaluating x_j in
    rung i's Context gives exactly what rung i's Hamiltonian gives for x_j directly.
    """
    from md_tools.remd import exchange_log_acceptance

    h, x = hamiltonian
    p = _protocol(h)
    xj = x + 0.01                                   # a different configuration, deliberately
    ci, cj = _context(h.system, x), _context(h.system, xj)
    p.prepare_context(1, ci)
    p.prepare_context(2, cj)
    beta = p.beta
    u = {}
    for i, c, coords in ((1, ci, x), (2, cj, xj)):
        for j in (1, 2):
            u[(j, i)] = beta * h.energy(c, p.rungs[j].public_state)
        p.prepare_context(i, c)                     # restored, as an exchange must leave it
        p.verify_context(i, c)
    assert u[(1, 1)] != u[(2, 1)], "a cross term equal to the diagonal is not an exchange"
    direct = beta * h.energy(_context(h.system, xj), p.rungs[1].public_state)
    assert u[(1, 2)] == pytest.approx(direct, rel=1e-12)
    log_alpha = exchange_log_acceptance(u[(1, 1)], u[(2, 2)], u[(1, 2)], u[(2, 1)])
    assert log_alpha == pytest.approx((u[(1, 1)] + u[(2, 2)]) - (u[(1, 2)] + u[(2, 1)]))


def test_one_beta_for_every_rung(hamiltonian):
    h, _ = hamiltonian
    p = _protocol(h)
    assert p.beta == pytest.approx(1.0 / (0.008314462618 * 300.0), rel=1e-6)
    assert p.n_states == len(LAMBDAS)


def test_the_ladder_identity_is_the_hamiltonians_and_the_rungs_differ_only_in_parameters(
        hamiltonian):
    h, _ = hamiltonian
    described = _protocol(h).describe()
    assert described["hamiltonian"] is h.record
    assert described["protocol"] == "lambda-ladder"
    states = [tuple(sorted(r["context_parameters"].items())) for r in described["rungs"]]
    assert len(set(states)) == len(states)


def test_asking_a_lambda_ladder_for_tau_explains_the_pending_record(hamiltonian):
    """The driver reads `protocol.tau`; a lambda ladder refuses, and says why.

    Not `[0.0] * K`: that runs today and writes structurally valid files in which a lambda ladder
    is indistinguishable from a REST2 ladder that never heated (S0's ruling, 2026-09-21). The
    refusal is what makes the next person meet the decision rather than an AttributeError, so the
    message is asserted, not merely the exception type.
    """
    h, _ = hamiltonian
    p = _protocol(h)
    with pytest.raises(LambdaLadderError, match="coordinate record"):
        p.tau
    try:
        p.tau
    except LambdaLadderError as refused:
        text = str(refused)
    assert "0.7.1" in text and "tent path" in text, "the message must point at the decision"
    assert "describe()['rungs']" in text, "and at where the rung states already are"
    # what IS available stays available: the refusal is about the RECORD, not the states
    assert [r["state"]["lambda_sterics"] for r in p.describe()["rungs"]] == LAMBDAS
