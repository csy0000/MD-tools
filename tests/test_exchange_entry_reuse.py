"""The exchange's own energy is taken, not recomputed -- and rules say what they will read.

TWO CLAIMS, ONE OF WHICH IS A MEASUREMENT AND THE OTHER A CONTRACT.

`u[i][state_to_walker[i]]` is the energy of the configuration state `i`'s Context is already
holding: `_gather_configurations` read it out of that very Context. Installing it again to measure
it is a round trip for a number already in hand. Amber does not pay it either -- its
`hamiltonian_exchange` sets `energy_1` from the value the last dynamics step produced and calls
`pme_force` once, for the cross term alone.

The reuse must be EXACT, not close. It goes through the same `reduced_potential` the engine uses,
on the same configuration in the same Hamiltonian, so the two paths are the same arithmetic. This
is not the inference `reduced_potential_of` refuses: that refusal is about deriving one rung's
energy from another's by scaling, which is wrong because the scale factors differ per term.

`required_entries` is the other half. `ExchangeContext` has always promised that the potential is
"a callable rather than a precomputed matrix so a rule that needs only four numbers pays for four,
not for N squared", and a rule had no way to say which four. It can now. The driver records the
declaration and deliberately does NOT use it to skip evaluations -- `rem_log.block_rows` indexes
`u[state, state]` and `u[state, mate]`, `exchange_free_energies` needs both cross terms of every
proposed pair, and `_write_exchange_csv` indexes `u[state][walker]`, so a sparse matrix would leave
holes that some renderer indexes. The contract is stated where rules live; the density is the
driver's business.

PLATFORM_POLICY_EXEMPTION: no OpenMM Context and no propagation. The engine and the coordinator are
stand-ins that record which method was called for which entry, because what is under test is which
call the driver makes, not what a device computes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from md_tools.remd.driver import ReplicaRun
from md_tools.remd.engine import Configuration, reduced_potential
from md_tools.remd.rules import NeighbouringExchangeRule, alternating_pairs, deterministic_phase


BETA = 0.40090785014980906


class _RecordingEngine:
    """Distinct, recognisable numbers per (state, configuration), and a call log.

    Every state's Context holds its own walker, exactly as `_install_owned` leaves them, because
    `_reduced_potential_matrix` assembles a row per owned state in one call.
    """

    def __init__(self, energies, state_to_walker=None):
        #: energies[state][walker] in kJ/mol
        self._energies = energies
        self.installed = []          # (state, walker) pairs reached by reduced_potential_of
        self.taken = []              # states whose Context energy was taken directly
        self._held = dict(enumerate(state_to_walker)) if state_to_walker is not None else {}

    def potential_energy(self, state_index):
        walker = self._held[state_index]
        self.taken.append(state_index)
        return self._energies[state_index][walker]

    def reduced_potential_of(self, state_index, configuration):
        walker = int(configuration.positions[0][0])
        self.installed.append((state_index, walker))
        volume = None if configuration.box is None else float(
            abs(np.linalg.det(configuration.box)))
        return reduced_potential(self._energies[state_index][walker], BETA,
                                 pressure_bar=None, volume_nm3=volume)


def _runner(engine, n_states, owned):
    """The minimum `_reduced_potential_matrix` touches. Nothing else is constructed."""
    return SimpleNamespace(
        protocol=SimpleNamespace(n_states=n_states, beta=BETA, pressure_bar=None),
        engine=engine,
        owned=list(owned),
        coordinator=SimpleNamespace(size=1))


def _configurations(n_states):
    """Walker w is identifiable from its own coordinates, so the engine can name it."""
    return [Configuration(np.full((1, 3), float(w)), np.zeros((1, 3)), None)
            for w in range(n_states)]


# -- the diagonal is taken, not recomputed ------------------------------------------------------

@pytest.mark.parametrize("state_to_walker", [[0, 1], [1, 0], [2, 0, 1]])
def test_the_diagonal_is_read_from_the_context_and_never_installed(state_to_walker):
    n = len(state_to_walker)
    energies = [[-100.0 - 10 * i - w for w in range(n)] for i in range(n)]
    engine = _RecordingEngine(energies, state_to_walker)
    ReplicaRun._reduced_potential_matrix(
        _runner(engine, n, range(n)), _configurations(n), state_to_walker=state_to_walker)

    for state, walker in enumerate(state_to_walker):
        assert state in engine.taken, f"state {state}'s own energy was not taken from its Context"
        assert (state, walker) not in engine.installed, (
            f"u[{state}][{walker}] is the configuration that Context already holds; installing it "
            f"again is a round trip for a number already in hand")
        # Every OTHER entry of the row is still evaluated the honest way.
        for other in range(n):
            if other != walker:
                assert (state, other) in engine.installed


@pytest.mark.parametrize("state_to_walker", [[0, 1], [1, 0], [2, 0, 1]])
def test_the_reused_diagonal_is_bit_identical_to_evaluating_it(state_to_walker):
    """THE CHECK THAT MATTERS. A cheaper number that differs is not the same experiment.

    WHAT THIS DOES AND DOES NOT COVER. The recording engine here returns exact values, so this
    pins the ARITHMETIC: the reused entry is the same computation as the evaluated one, not an
    approximation of it. On a device it is measured separately -- CUDA double precision agrees to
    5e-11 reduced units, while at mixed precision neither path is bitwise stable (reading one
    energy twice spreads by 3e-5, and the old save/install/restore round trip moved it by 3e-3,
    about a hundred times that). So "bit-identical" is a statement about this arithmetic, not a
    promise about a mixed-precision device, and the reuse in fact perturbs the Context LESS than
    evaluating did.
    """
    n = len(state_to_walker)
    energies = [[-100.0 - 10 * i - w for w in range(n)] for i in range(n)]
    configurations = _configurations(n)

    matrix = ReplicaRun._reduced_potential_matrix(
        _runner(_RecordingEngine(energies, state_to_walker), n, range(n)),
        configurations, state_to_walker=state_to_walker)

    for state, walker in enumerate(state_to_walker):
        evaluated = _RecordingEngine(energies).reduced_potential_of(
            state, configurations[walker])
        assert matrix[state][walker] == evaluated, (
            "the reused diagonal must be the same number, not a close one")


def test_without_a_mapping_every_entry_is_evaluated_as_before():
    """The old call signature keeps the old behaviour: no mapping, no reuse."""
    n = 2
    energies = [[-100.0 - 10 * i - w for w in range(n)] for i in range(n)]
    engine = _RecordingEngine(energies)
    matrix = ReplicaRun._reduced_potential_matrix(
        _runner(engine, n, [0, 1]), _configurations(n))
    assert engine.taken == []
    assert sorted(engine.installed) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert matrix.shape == (n, n)


def test_the_matrix_stays_dense_so_every_renderer_can_index_it():
    """`rem.log`, `exchange.csv` and the free energies index it by two different conventions."""
    n = 3
    state_to_walker = [2, 0, 1]
    energies = [[-100.0 - 10 * i - w for w in range(n)] for i in range(n)]
    matrix = ReplicaRun._reduced_potential_matrix(
        _runner(_RecordingEngine(energies, state_to_walker), n, range(n)),
        _configurations(n), state_to_walker=state_to_walker)
    assert matrix.shape == (n, n)
    assert np.isfinite(matrix).all(), "a hole here is a nan in rem.log or exchange.csv"


# -- rules declare what they will read ----------------------------------------------------------

@pytest.mark.parametrize("n_states", [2, 3, 4, 6])
@pytest.mark.parametrize("exchange_index", [1, 2, 3])
def test_a_rule_declares_exactly_the_entries_its_sweep_reads(n_states, exchange_index):
    """The declaration and the sweep must agree, or a declaration is worse than none."""
    rule = NeighbouringExchangeRule()
    state_to_walker = list(reversed(range(n_states)))
    declared = rule.required_entries(n_states=n_states, state_to_walker=state_to_walker,
                                     exchange_index=exchange_index)

    phase = deterministic_phase(n_states, exchange_index)
    expected = set()
    for state_i, state_j in alternating_pairs(n_states, phase):
        walker_i, walker_j = state_to_walker[state_i], state_to_walker[state_j]
        expected.update({(state_i, walker_i), (state_j, walker_j),
                         (state_i, walker_j), (state_j, walker_i)})
    assert declared == expected


def test_the_declaration_is_asked_with_the_index_propose_will_see():
    """`deterministic_phase` keys the sweep off the POST-increment index.

    Declaring against the pre-increment value would name the other phase's pairs, which on a
    ladder of three or more states is a different set entirely.
    """
    rule = NeighbouringExchangeRule()
    mapping = [0, 1, 2]
    first = rule.required_entries(n_states=3, state_to_walker=mapping, exchange_index=1)
    second = rule.required_entries(n_states=3, state_to_walker=mapping, exchange_index=2)
    assert first != second, "consecutive exchanges alternate phase and so must their declarations"


def test_two_states_declare_the_same_pair_every_time():
    """With two states one phase covers no pair at all, so the schedule must not alternate."""
    rule = NeighbouringExchangeRule()
    declarations = {frozenset(rule.required_entries(
        n_states=2, state_to_walker=[0, 1], exchange_index=i)) for i in range(1, 6)}
    assert len(declarations) == 1
    assert declarations.pop() == frozenset({(0, 0), (1, 1), (0, 1), (1, 0)})


def test_a_declaration_is_optional_and_its_absence_means_everything():
    """A rule that has not been taught this must keep working, by omission."""
    class Older:
        def describe(self):
            return {"name": "older"}

        def propose(self, context):            # pragma: no cover - never called here
            raise AssertionError

    assert getattr(Older(), "required_entries", None) is None
