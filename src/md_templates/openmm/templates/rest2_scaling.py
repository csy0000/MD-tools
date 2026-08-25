"""REST2 Hamiltonian scaling. Copied verbatim into every generated REST2 project.

This is the AMBER-compatible convention, expressed in tau:

    s        = (1 - tau)^2        solute-solute terms
    sqrt(s)  = (1 - tau)          solute-environment terms

so that for a pairwise nonbonded force

    U_s = s * U_solute-solute  +  sqrt(s) * U_solute-environment  +  U_environment

Charges scale by sqrt(s) and epsilons by s, which produces exactly that split without needing a
custom force. Solute torsions scale by s; CMAP maps that are wholly within the solute scale by s.
Torsions about an omega bond are LEFT ALONE -- scaling them lets a peptide bond rotate at the hot
rungs, so the ladder samples cis/trans interconversion the cold rung never sees, and the exchange
no longer connects two states of the same system.

Every rung is thermostatted at the same temperature. "Effective solute temperature" is a way of
describing the scaling, not a second thermostat: beta is common to the whole ladder.

It lives in the generated project rather than being imported, so a moved project needs only OpenMM.
"""
import math

from openmm import (CMAPTorsionForce, NonbondedForce, PeriodicTorsionForce, XmlSerializer)


def scale_factor_for_tau(tau):
    """s = (1 - tau)^2. tau = 0 is the cold, unscaled replica."""
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1); got {tau}")
    return (1.0 - tau) ** 2


def linear_tau_ladder(minimum, maximum, count):
    if count < 2:
        raise ValueError(f"a ladder needs at least 2 replicas; got {count}")
    step = (maximum - minimum) / (count - 1)
    return [minimum + step * i for i in range(count)]


def clone_system(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _scale_nonbonded(force, solute, s):
    root = math.sqrt(s)
    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        if index in solute:
            force.setParticleParameters(index, charge * root, sigma, epsilon * s)
    for index in range(force.getNumExceptions()):
        i, j, charge_product, sigma, epsilon = force.getExceptionParameters(index)
        n_solute = int(i in solute) + int(j in solute)
        if n_solute == 2:
            force.setExceptionParameters(index, i, j, charge_product * s, sigma, epsilon * s)
        elif n_solute == 1:
            force.setExceptionParameters(index, i, j, charge_product * root, sigma, epsilon * root)


def _scale_torsions(force, solute, s, excluded_bonds):
    for index in range(force.getNumTorsions()):
        i, j, k, l, periodicity, phase, k_value = force.getTorsionParameters(index)
        if all(a in solute for a in (i, j, k, l)) and frozenset((int(j), int(k))) not in excluded_bonds:
            force.setTorsionParameters(index, i, j, k, l, periodicity, phase, k_value * s)


def _scale_cmap(force, solute, s):
    """CMAP maps are shared, so a map used by ANY non-solute torsion must not be scaled in place."""
    solute_maps, other_maps = set(), set()
    for index in range(force.getNumTorsions()):
        parameters = force.getTorsionParameters(index)
        map_index, atoms = parameters[0], parameters[1:]
        (solute_maps if all(a in solute for a in atoms) else other_maps).add(map_index)
    for map_index in sorted(solute_maps - other_maps):
        size, energy = force.getMapParameters(map_index)
        force.setMapParameters(map_index, size, [e * s for e in energy])


def build_scaled_system(base_system, solute_indices, tau, excluded_bonds=()):
    """A copy of `base_system` with the solute Hamiltonian scaled for this rung."""
    s = scale_factor_for_tau(tau)
    system = clone_system(base_system)
    if s == 1.0:
        return system                              # the cold replica is the unmodified system
    solute = set(int(i) for i in solute_indices)
    excluded = {frozenset((int(a), int(b))) for a, b in excluded_bonds}
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce):
            _scale_nonbonded(force, solute, s)
        elif isinstance(force, PeriodicTorsionForce):
            _scale_torsions(force, solute, s, excluded)
        elif isinstance(force, CMAPTorsionForce):
            _scale_cmap(force, solute, s)
    return system


def reduced_potential(energy_kj_mol, beta, pressure_bar=None, volume_nm3=None):
    """u = beta * (U + p V), the NPT form.

    REST2 replicas share one thermostat temperature and one pressure, so they share beta: the pV
    terms cancel exactly in `exchange_log_acceptance` below, because a swap moves each configuration
    -- positions AND its box -- to the other rung, and the same two volumes appear on both sides.
    They are carried anyway so the cancellation is arithmetic that can be checked rather than an
    omission that has to be trusted.
    """
    u = beta * energy_kj_mol
    if pressure_bar is not None and volume_nm3 is not None:
        # 1 bar * 1 nm^3 in kJ/mol
        u += beta * pressure_bar * volume_nm3 * 0.0602214076
    return u


def exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji):
    """log of the Metropolis criterion for swapping configurations i and j.

        log(alpha) = beta * [U_i(x_i,V_i) + U_j(x_j,V_j) - U_i(x_j,V_j) - U_j(x_i,V_i)]

    where u_ij is replica j's configuration evaluated in replica i's Hamiltonian.
    """
    return (u_ii + u_jj) - (u_ij + u_ji)


def exchange_pairs(n_replicas, phase):
    """Alternating nearest-neighbour pairs: phase 0 gives (0,1),(2,3)…; phase 1 gives (1,2),(3,4)…"""
    return [(i, i + 1) for i in range(phase % 2, n_replicas - 1, 2)]
