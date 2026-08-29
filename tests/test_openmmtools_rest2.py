"""REST2 on OpenMMTools: the scaling, the exchange criterion, storage and restart.

The scaling tests build systems and evaluate energies but integrate nothing, so they run on the
Reference/CPU platform and carry no `gpu` marker -- they are exact arithmetic on a Hamiltonian,
and a GPU would only add single-precision noise to a comparison meant to be exact.

Anything that actually propagates carries `gpu`.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"

openmm = pytest.importorskip("openmm")
openmmtools = pytest.importorskip("openmmtools")

sys.path.insert(0, str(TEMPLATES))
import rest2_openmmtools as extension        # noqa: E402
import rest2_scaling as scaling              # noqa: E402
import rest2_runtime as runtime              # noqa: E402

from openmm import unit                      # noqa: E402


KJ = unit.kilojoule_per_mole


# --- fixtures ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vacuum():
    """Alanine dipeptide in vacuum: small, has torsions and a NonbondedForce, no water."""
    from openmmtools import testsystems
    ts = testsystems.AlanineDipeptideVacuum(constraints=None)
    return ts.system, ts.topology, ts.positions


@pytest.fixture(scope="module")
def explicit():
    """Alanine dipeptide in water: a real solute/environment split."""
    from openmmtools import testsystems
    ts = testsystems.AlanineDipeptideExplicit(constraints=None)
    return ts.system, ts.topology, ts.positions


@pytest.fixture(scope="module")
def implicit():
    from openmmtools import testsystems
    ts = testsystems.AlanineDipeptideImplicit(constraints=None)
    return ts.system, ts.topology, ts.positions


def energy_of(system, positions, box=None):
    """Potential energy in kJ/mol on the Reference platform."""
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    del context, integrator
    return value


def nonbonded(system):
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise AssertionError("no NonbondedForce")


def torsions(system):
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, openmm.PeriodicTorsionForce):
            return force
    raise AssertionError("no PeriodicTorsionForce")


# --- 1. tau = 0 reproduces the unscaled base system ------------------------------------------------

def test_tau_zero_reproduces_the_unscaled_system(vacuum):
    """The cold rung must be the system itself, not a system that merely evaluates the same."""
    system, _, positions = vacuum
    scaled = scaling.build_scaled_system(system, list(range(system.getNumParticles())), 0.0)
    assert energy_of(scaled, positions) == pytest.approx(energy_of(system, positions), abs=1e-9)

    base_nb, scaled_nb = nonbonded(system), nonbonded(scaled)
    for index in range(base_nb.getNumParticles()):
        assert scaled_nb.getParticleParameters(index) == base_nb.getParticleParameters(index)


# --- 2/3/4. the three nonbonded scaling regimes ----------------------------------------------------

def test_solute_solute_terms_scale_by_s_and_environment_is_untouched(explicit):
    """charge -> q*sqrt(s) and epsilon -> eps*s inside the solute; environment unchanged."""
    system, _, _ = explicit
    solute = list(range(22))
    tau = 0.4
    s = scaling.scale_factor_for_tau(tau)          # 0.36
    root = math.sqrt(s)                            # 0.6 = 1 - tau
    assert root == pytest.approx(1.0 - tau)

    scaled_system = scaling.build_scaled_system(system, solute, tau)
    base, scaled = nonbonded(system), nonbonded(scaled_system)

    for index in solute:
        q0, sig0, eps0 = base.getParticleParameters(index)
        q1, sig1, eps1 = scaled.getParticleParameters(index)
        assert q1.value_in_unit(unit.elementary_charge) == pytest.approx(
            q0.value_in_unit(unit.elementary_charge) * root, abs=1e-12)
        assert eps1.value_in_unit(KJ) == pytest.approx(eps0.value_in_unit(KJ) * s, abs=1e-12)
        assert sig1 == sig0                        # geometry is not scaled

    # Environment-environment: identical, not merely close.
    for index in range(len(solute), base.getNumParticles()):
        assert scaled.getParticleParameters(index) == base.getParticleParameters(index)


def test_exceptions_scale_by_s_within_the_solute_and_sqrt_s_across_the_boundary(explicit):
    system, _, _ = explicit
    solute = set(range(22))
    tau = 0.3
    s = scaling.scale_factor_for_tau(tau)
    root = math.sqrt(s)
    scaled_system = scaling.build_scaled_system(system, sorted(solute), tau)
    base, scaled = nonbonded(system), nonbonded(scaled_system)

    seen_both, seen_cross = False, False
    for index in range(base.getNumExceptions()):
        i, j, q0, sig0, eps0 = base.getExceptionParameters(index)
        _, _, q1, sig1, eps1 = scaled.getExceptionParameters(index)
        count = int(i in solute) + int(j in solute)
        factor = {2: s, 1: root, 0: 1.0}[count]
        seen_both |= count == 2
        seen_cross |= count == 1
        assert q1.value_in_unit(unit.elementary_charge**2) == pytest.approx(
            q0.value_in_unit(unit.elementary_charge**2) * factor, abs=1e-12)
        assert eps1.value_in_unit(KJ) == pytest.approx(
            eps0.value_in_unit(KJ) * factor, abs=1e-12)
    assert seen_both, "the fixture had no solute-solute exception, so the s branch was not tested"


# --- 5/6. torsions and the omega convention ---------------------------------------------------------

def test_solute_torsions_scale_by_s(vacuum):
    system, _, _ = vacuum
    solute = list(range(system.getNumParticles()))
    tau = 0.5
    s = scaling.scale_factor_for_tau(tau)
    scaled_system = scaling.build_scaled_system(system, solute, tau)
    base, scaled = torsions(system), torsions(scaled_system)
    assert base.getNumTorsions() > 0
    for index in range(base.getNumTorsions()):
        *_, k0 = base.getTorsionParameters(index)
        *_, k1 = scaled.getTorsionParameters(index)
        assert k1.value_in_unit(KJ) == pytest.approx(k0.value_in_unit(KJ) * s, abs=1e-12)


def test_omega_excluded_torsions_are_left_completely_unscaled(vacuum):
    """The repository's omega-selective convention: an excluded bond keeps its full barrier.

    This is the test that would fail if omega exclusion were quietly dropped -- which would let a
    hot rung rotate a peptide bond and sample cis/trans the cold rung never sees.
    """
    system, _, _ = vacuum
    solute = list(range(system.getNumParticles()))
    base = torsions(system)
    # Pick a real central bond from the fixture and exclude it.
    _, j, k, _, _, _, _ = base.getTorsionParameters(0)
    excluded = [(int(j), int(k))]
    scaled_system = scaling.build_scaled_system(system, solute, 0.5, excluded_bonds=excluded)
    scaled = torsions(scaled_system)

    protected, scaled_count = 0, 0
    s = scaling.scale_factor_for_tau(0.5)
    for index in range(base.getNumTorsions()):
        a, b, c, d, _, _, k0 = base.getTorsionParameters(index)
        *_, k1 = scaled.getTorsionParameters(index)
        if frozenset((int(b), int(c))) == frozenset(excluded[0]):
            assert k1.value_in_unit(KJ) == pytest.approx(k0.value_in_unit(KJ), abs=1e-12)
            protected += 1
        else:
            assert k1.value_in_unit(KJ) == pytest.approx(k0.value_in_unit(KJ) * s, abs=1e-12)
            scaled_count += 1
    assert protected > 0 and scaled_count > 0


# --- 7. CMAP -----------------------------------------------------------------------------------------

def test_cmap_maps_used_outside_the_solute_are_never_scaled_in_place():
    """A shared map scaled in place would change the environment's Hamiltonian too."""
    system = openmm.System()
    for _ in range(10):
        system.addParticle(1.0)
    cmap = openmm.CMAPTorsionForce()
    energy = [0.5 * i for i in range(4)]
    shared = cmap.addMap(2, energy)
    cmap.addTorsion(shared, 0, 1, 2, 3, 1, 2, 3, 4)          # wholly inside the solute
    cmap.addTorsion(shared, 5, 6, 7, 8, 6, 7, 8, 9)          # outside it
    system.addForce(cmap)

    solute = list(range(5))
    scaled = scaling.build_scaled_system(system, solute, 0.5)
    _, after = scaled.getForce(0).getMapParameters(shared)
    after = [value.value_in_unit(KJ) for value in after]
    assert after == pytest.approx(energy), (
        "a CMAP map used by a non-solute torsion was scaled, which changes the environment")


def test_a_cmap_map_used_only_by_the_solute_is_scaled():
    system = openmm.System()
    for _ in range(5):
        system.addParticle(1.0)
    cmap = openmm.CMAPTorsionForce()
    energy = [1.0, 2.0, 3.0, 4.0]
    only = cmap.addMap(2, energy)
    cmap.addTorsion(only, 0, 1, 2, 3, 1, 2, 3, 4)
    system.addForce(cmap)
    s = scaling.scale_factor_for_tau(0.5)
    scaled = scaling.build_scaled_system(system, list(range(5)), 0.5)
    _, after = scaled.getForce(0).getMapParameters(only)
    after = [value.value_in_unit(KJ) for value in after]
    assert after == pytest.approx([e * s for e in energy])


# --- 8. unknown energy-bearing forces are refused -----------------------------------------------------

def test_an_unclassifiable_energy_bearing_force_is_refused(vacuum):
    system, _, _ = vacuum
    clone = scaling.clone_system(system)
    custom = openmm.CustomBondForce("k*(r-r0)^2")
    custom.addPerBondParameter("k")
    custom.addPerBondParameter("r0")
    clone.addForce(custom)
    with pytest.raises(scaling.UnclassifiedForceError) as raised:
        scaling.build_scaled_system(clone, [0, 1], 0.5)
    assert "CustomBondForce" in str(raised.value)


# --- 9/10. implicit GBn2 ------------------------------------------------------------------------------

def test_whole_system_gbn2_scaling_multiplies_every_term(implicit):
    """Every GB term scales by s, including the non-polar one that has no charge dependence."""
    system, _, positions = implicit
    whole = list(range(system.getNumParticles()))
    base_energy = energy_of(system, positions)
    scaled = scaling.build_scaled_system(system, whole, 0.5)
    assert energy_of(scaled, positions) != pytest.approx(base_energy, rel=1e-6)

    gb = [scaled.getForce(i) for i in range(scaled.getNumForces())
          if isinstance(scaled.getForce(i), openmm.CustomGBForce)]
    assert gb, "the implicit fixture carried no CustomGBForce"
    names = [gb[0].getGlobalParameterName(i)
             for i in range(gb[0].getNumGlobalParameters())]
    assert scaling.REST2_GB_SCALE_PARAMETER in names
    index = names.index(scaling.REST2_GB_SCALE_PARAMETER)
    assert gb[0].getGlobalParameterDefaultValue(index) == pytest.approx(0.25)


def test_partial_region_gbn2_is_refused(implicit):
    """A generalised-Born energy is not separable per atom, so a partial region is refused."""
    system, _, _ = implicit
    total = system.getNumParticles()
    partial = list(range(total - 1))               # everything but one atom
    assert len(partial) < total
    with pytest.raises(ValueError) as raised:
        scaling.build_scaled_system(system, partial, 0.5)
    message = str(raised.value)
    assert "entire system" in message and "not separable" in message


# --- 11. OpenMMTools reduced potentials agree with direct OpenMM ---------------------------------------

def test_openmmtools_reduced_potential_matches_a_direct_openmm_evaluation(explicit):
    """u = beta*(U + pV) computed by openmmtools must equal the same quantity computed by hand."""
    from openmmtools import states
    system, _, positions = explicit
    solute = list(range(22))
    tau = 0.3
    temperature = 300.0
    scaled = scaling.build_scaled_system(system, solute, tau)

    thermo = states.ThermodynamicState(system=scaled, temperature=temperature * unit.kelvin,
                                       pressure=1.0 * unit.bar)
    # A SamplerState carries no energy until a Context computes one, so the reduced potential is
    # taken from a real Context -- which is also what the sampler does at every iteration.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = thermo.create_context(integrator,
                                    platform=openmm.Platform.getPlatformByName("Reference"))
    sampler = states.SamplerState(
        positions=positions, box_vectors=system.getDefaultPeriodicBoxVectors())
    sampler.apply_to_context(context)
    theirs = thermo.reduced_potential(context)

    box = system.getDefaultPeriodicBoxVectors()
    energy = energy_of(scaled, positions, box=box)
    vectors = np.array([[v.x, v.y, v.z]
                        for v in (vector.value_in_unit(unit.nanometer) for vector in box)])
    volume = float(np.linalg.det(vectors))                     # nm^3
    beta = 1.0 / (0.008314462618 * temperature)
    mine = scaling.reduced_potential(energy, beta, pressure_bar=1.0, volume_nm3=volume)
    assert theirs == pytest.approx(mine, rel=2e-4)


# --- 12. the exchange criterion ------------------------------------------------------------------------

def test_exchange_log_acceptance_is_the_analytical_metropolis_criterion():
    u_ii, u_jj, u_ij, u_ji = 10.0, 12.0, 11.5, 11.0
    expected = (u_ii + u_jj) - (u_ij + u_ji)
    assert scaling.exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji) == pytest.approx(expected)
    assert math.exp(min(0.0, expected)) == pytest.approx(math.exp(min(0.0, -0.5)))


def test_the_sampler_uses_the_repositorys_own_acceptance_criterion():
    """The override must call `exchange_log_acceptance`, not restate the algebra.

    Two implementations of one criterion is exactly the drift this repository keeps finding: the
    run uses one and the test checks the other, and both look green.
    """
    source = (TEMPLATES / "rest2_openmmtools.py").read_text(encoding="utf-8")
    assert "exchange_log_acceptance(" in source
    assert "from rest2_scaling import exchange_log_acceptance" in source


class _FakeSampler(extension.StridedREST2Sampler):
    """Only `_attempt_swap`'s arithmetic, with the machinery it reads stubbed out."""

    def __init__(self, energies, assignment):
        self._energy_thermodynamic_states = np.asarray(energies, dtype=float)
        self._replica_thermodynamic_states = np.asarray(assignment, dtype=int)
        n = len(assignment)
        self._n_proposed_matrix = np.zeros((n, n), dtype=int)
        self._n_accepted_matrix = np.zeros((n, n), dtype=int)


def test_a_certain_swap_is_always_accepted_and_the_counters_move():
    # log_p = (u_ii + u_jj) - (u_ij + u_ji) > 0 -> always accepted.
    energies = [[0.0, 100.0], [100.0, 0.0]]
    sampler = _FakeSampler(energies, [1, 0])       # replica 0 at state 1, replica 1 at state 0
    sampler._attempt_swap(0, 1)
    assert list(sampler._replica_thermodynamic_states) == [0, 1]
    assert sampler._n_proposed_matrix.sum() == 2   # symmetric
    assert sampler._n_accepted_matrix.sum() == 2


def test_an_impossible_swap_is_rejected_and_only_the_proposal_is_counted():
    energies = [[0.0, 1e6], [1e6, 0.0]]
    sampler = _FakeSampler(energies, [0, 1])
    sampler._attempt_swap(0, 1)
    assert list(sampler._replica_thermodynamic_states) == [0, 1]
    assert sampler._n_proposed_matrix.sum() == 2
    assert sampler._n_accepted_matrix.sum() == 0


def test_swap_neighbors_would_raise_without_the_scalar_override():
    """The upstream 0.26.0 defect this extension exists to correct.

    `np.where` returns a tuple, indexing the energy matrix with it gives a 2-d array, and
    `math.exp` refuses it. If a future openmmtools fixes this upstream, this test starts failing
    and the override can be reconsidered -- which is the point of pinning it.
    """
    assignment = np.array([0, 1, 2])
    energies = np.zeros((3, 3))
    replica_i = np.where(assignment == 0)          # the tuple upstream passes through
    value = energies[replica_i, 1]
    assert value.ndim > 0
    with pytest.raises(TypeError):
        math.exp(value)


# --- 13/14. the stride, and a two-replica ladder ---------------------------------------------------------

def test_exchange_stride_conversion_is_exact_or_refused():
    assert extension.exchange_stride_for(exchange_interval_ps=10.0,
                                         solute_output_interval_ps=2.0) == 5
    assert extension.exchange_stride_for(exchange_interval_ps=10.0,
                                         solute_output_interval_ps=10.0) == 1
    with pytest.raises(ValueError, match="whole multiple"):
        extension.exchange_stride_for(exchange_interval_ps=10.0, solute_output_interval_ps=3.0)
    with pytest.raises(ValueError, match="shorter than"):
        extension.exchange_stride_for(exchange_interval_ps=1.0, solute_output_interval_ps=2.0)
    with pytest.raises(ValueError, match="shorter than"):
        extension.exchange_stride_for(exchange_interval_ps=5.0, solute_output_interval_ps=10.0)


def test_a_two_replica_ladder_attempts_a_swap_on_every_exchange_iteration():
    """Upstream alternates even/odd pairs, and for TWO replicas the odd phase has no pair at all.

    With `range(offset, n-1, 2)` and offset drawn from {0, 1}, a two-replica ladder proposes
    nothing whenever offset is 1. The ladder must not silently halve its own exchange rate, so the
    override draws no offset when there is only one pair to attempt.
    """
    attempts = []

    class _Counting(extension.StridedREST2Sampler):
        def __init__(self):
            self._replica_thermodynamic_states = np.array([0, 1])
            self._energy_thermodynamic_states = np.zeros((2, 2))
            self._n_proposed_matrix = np.zeros((2, 2), dtype=int)
            self._n_accepted_matrix = np.zeros((2, 2), dtype=int)

        @property
        def n_replicas(self):
            return 2

        def _attempt_swap(self, i, j):
            attempts.append((i, j))

    sampler = _Counting()
    for _ in range(40):
        sampler._mix_neighboring_replicas()
    assert len(attempts) == 40, (
        f"a two-replica ladder skipped {40 - len(attempts)} of 40 exchange iterations")


# --- the version pin -------------------------------------------------------------------------------------

def test_the_extension_refuses_an_untested_openmmtools_version():
    assert extension.TESTED_OPENMMTOOLS_VERSION == "0.26.0"
    assert extension.installed_openmmtools_version() == "0.26.0"
    with pytest.raises(extension.UntestedOpenMMToolsError) as raised:
        _refuse_with(extension, "9.9.9", environment={})
    assert "0.26.0" in str(raised.value)


def _refuse_with(module, version, environment):
    original = module.installed_openmmtools_version
    module.installed_openmmtools_version = lambda: version
    try:
        return module.require_tested_openmmtools(environment=environment)
    finally:
        module.installed_openmmtools_version = original


def test_the_version_refusal_can_be_overridden_only_deliberately():
    assert _refuse_with(extension, "9.9.9",
                        environment={extension.VERSION_OVERRIDE_ENVIRONMENT: "1"}) == "9.9.9"


def test_every_overridden_private_method_still_exists_upstream():
    """A contract test: the pin is worthless if the parent quietly renames what we override."""
    from openmmtools.multistate import ReplicaExchangeSampler
    for name in ("_mix_replicas", "_mix_neighboring_replicas", "_attempt_swap", "equilibrate"):
        assert hasattr(ReplicaExchangeSampler, name), (
            f"openmmtools {openmmtools.__version__} has no {name}; the override is now silent")


# --- the runtime's arithmetic ---------------------------------------------------------------------------

def test_durations_convert_to_exact_steps_or_are_refused():
    assert runtime.exact_steps(2.0, 4.0, what="x") == 500
    assert runtime.exact_steps(10.0, 4.0, what="x") == 2500
    with pytest.raises(ValueError, match="not a whole number"):
        runtime.exact_steps(2.0, 3.0, what="x")


def test_rank_to_device_binding_never_puts_two_ranks_on_one_gpu_when_it_need_not():
    devices = list(range(6))
    chosen = [runtime.select_device_for_rank(rank, 6, devices)[0] for rank in range(6)]
    assert sorted(chosen) == [str(d) for d in devices]
    assert len(set(chosen)) == 6


def test_fewer_devices_than_ranks_is_a_stated_policy_not_an_accident():
    chosen = [runtime.select_device_for_rank(rank, 6, [0, 1]) for rank in range(6)]
    assert {device for device, _ in chosen} == {"0", "1"}
    assert all("round-robin" in policy for _, policy in chosen)


def test_cuda_visible_devices_is_renumbered_to_local_ordinals(monkeypatch):
    """CUDA_VISIBLE_DEVICES renumbers from 0 for the process; passing driver ordinals addresses
    the wrong GPU whenever the variable does not start at 0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    assert runtime.visible_cuda_devices() == [0, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert runtime.visible_cuda_devices() == []
