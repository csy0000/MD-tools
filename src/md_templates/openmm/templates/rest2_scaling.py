"""REST2 Hamiltonian scaling. Copied verbatim into every generated REST2 project.

This is the AMBER-compatible convention, expressed in tau:

    s        = (1 - tau)^2        solute-solute terms
    (1 - tau)     solute-environment terms

so that for a pairwise nonbonded force

    U_tau = (1-tau)^2 * U_solute-solute  +  (1-tau) * U_solute-environment  +  U_environment

Charges scale by sqrt(s) and epsilons by s, which produces exactly that split without needing a
custom force. Solute torsions scale by s; CMAP maps that are wholly within the solute scale by s.
Torsions about an omega bond are LEFT ALONE -- scaling them lets a peptide bond rotate at the hot
rungs, so the ladder samples cis/trans interconversion the cold rung never sees, and the exchange
no longer connects two states of the same system.

Every rung is thermostatted at the same temperature. "Effective solute temperature" is a way of
describing the scaling, not a second thermostat: beta is common to the whole ladder.

It lives in the generated project rather than being imported, so a moved project needs only OpenMM.
"""

from openmm import (CMAPTorsionForce, CustomGBForce, NonbondedForce, PeriodicTorsionForce,
                    XmlSerializer)


#: The Hamiltonian convention this module implements, by name and version.
#:
#: It participates in Hamiltonian identity and continuation checks: two runs agree only if they
#: agree about what "REST2" meant, and that includes which terms are left alone. `tau` is the only
#: state coordinate. There is deliberately no second variable: a derived quantity that is also
#: stored is a second thing to keep consistent, and the one that drifts is never the one you check.
REST2_IMPLEMENTATION = {
    "name": "rest2-no-bond-angle-omega",
    "version": 1,
    "state_coordinate": "tau",
    "solute_solute_nonbonded_scale": "(1-tau)^2",
    "solute_environment_nonbonded_scale": "1-tau",
    "eligible_solute_torsion_scale": "(1-tau)^2",
    "bonds": "unscaled",
    "angles": "unscaled",
    "ordinary_amide_omega": "unscaled",
}


def scaling_for_tau(tau):
    """The two factors this convention needs, derived from tau once.

    Returned together and passed down, rather than recomputed inside each force handler: a
    `sqrt()` repeated in three places is three chances for one of them to be changed alone.

        solute_solute_scale       (1 - tau)^2   solute-solute nonbonded, eligible solute torsions
        solute_environment_scale  (1 - tau)     solute-environment nonbonded

    tau = 0 is the cold, unscaled replica, where both factors are 1.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1); got {tau}")
    solute_solute_scale = (1.0 - tau) ** 2
    solute_environment_scale = 1.0 - tau
    return solute_solute_scale, solute_environment_scale


def linear_tau_ladder(minimum, maximum, count):
    if count < 2:
        raise ValueError(f"a ladder needs at least 2 replicas; got {count}")
    step = (maximum - minimum) / (count - 1)
    return [minimum + step * i for i in range(count)]


def clone_system(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _scale_nonbonded(force, solute, solute_solute, solute_environment):
    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        if index in solute:
            force.setParticleParameters(index, charge * solute_environment, sigma,
                                        epsilon * solute_solute)
    for index in range(force.getNumExceptions()):
        i, j, charge_product, sigma, epsilon = force.getExceptionParameters(index)
        n_solute = int(i in solute) + int(j in solute)
        if n_solute == 2:
            force.setExceptionParameters(index, i, j, charge_product * solute_solute,
                                         sigma, epsilon * solute_solute)
        elif n_solute == 1:
            # One solute partner: the cross term follows the solute-environment factor, exactly
            # as an unexcepted solute-environment pair does.
            force.setExceptionParameters(index, i, j, charge_product * solute_environment,
                                         sigma, epsilon * solute_environment)


def _scale_torsions(force, solute, solute_solute, excluded_bonds):
    for index in range(force.getNumTorsions()):
        i, j, k, l, periodicity, phase, k_value = force.getTorsionParameters(index)
        if all(a in solute for a in (i, j, k, l)) and frozenset((int(j), int(k))) not in excluded_bonds:
            force.setTorsionParameters(index, i, j, k, l, periodicity, phase,
                                       k_value * solute_solute)


#: Bumped when the omega classification RULES change, not when their inputs do. A record carrying
#: this version says which algorithm decided what, so a stored exclusion can be re-derived.
OMEGA_DETECTOR_VERSION = 1


def torsion_exclusion_report(system, solute, excluded_bonds):
    """Which PeriodicTorsionForce torsions each excluded central bond actually protects.

    The stored exclusion is a pair of ATOM indices, but what it does is leave a set of TORSION
    terms unscaled -- and the mapping between them depends on the force field that built the
    System. Recording the bond alone means a reader has to re-derive that mapping to check
    anything, using assumptions that may not match the ones used here. This records the result.

    Only wholly-solute torsions are listed, because only those were candidates for scaling in the
    first place; a torsion reaching into the environment is untouched either way. The predicate is
    the same one `_scale_torsions` applies, so the report cannot describe a different exclusion
    than the one performed.
    """
    excluded = {frozenset((int(a), int(b))) for a, b in excluded_bonds}
    solute = set(int(i) for i in solute)
    report = {tuple(sorted(bond)): [] for bond in excluded}
    scaled = 0
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if not isinstance(force, PeriodicTorsionForce):
            continue
        for torsion in range(force.getNumTorsions()):
            i, j, k, l, _periodicity, _phase, _k = force.getTorsionParameters(torsion)
            if not all(a in solute for a in (i, j, k, l)):
                continue
            central = frozenset((int(j), int(k)))
            if central in excluded:
                report[tuple(sorted(central))].append(int(torsion))
            else:
                scaled += 1
    return {
        "detector_version": OMEGA_DETECTOR_VERSION,
        "excluded_central_bonds": [list(bond) for bond in sorted(report)],
        "excluded_torsion_indices": {f"{a}-{b}": indices for (a, b), indices in sorted(
            report.items())},
        "n_excluded_torsions": sum(len(v) for v in report.values()),
        "n_scaled_solute_torsions": scaled,
    }


def cmap_map_roles(force, solute):
    """Which CMAP maps belong to the solute, to the environment, or to both.

    A map is a shared lookup table, not a per-torsion parameter: one map is typically referenced by
    every torsion of the same residue type in the system. So "is this map the solute's" is a
    question about its USERS, and it has three answers, not two.
    """
    solute_maps, other_maps = set(), set()
    for index in range(force.getNumTorsions()):
        parameters = force.getTorsionParameters(index)
        map_index, atoms = parameters[0], parameters[1:]
        (solute_maps if all(a in solute for a in atoms) else other_maps).add(map_index)
    return {"exclusive_solute": solute_maps - other_maps,
            "shared": solute_maps & other_maps,
            "exclusive_other": other_maps - solute_maps}


def shared_cmap_originals(force, solute):
    """The shared maps, in the deterministic order duplicates are appended in.

    The order is part of the contract: it is what lets a duplicate be matched back to its original
    later, without storing a side table that could drift from the System it describes.
    """
    return sorted(cmap_map_roles(force, solute)["shared"])


def duplicate_shared_cmaps(force, solute):
    """Give the solute its own copy of every shared map, and point its torsions at the copy.

    Scaling a shared map in place would scale it for the environment too. Leaving it alone -- what
    this used to do -- is the opposite error and just as wrong: a solute torsion that happens to
    share a map with a non-solute one then gets NO scaling, and the Hamiltonian silently stops
    being the one the ladder says it is.

    Returns `{duplicate_index: original_index}`. A map used only by the solute needs no copy and is
    scaled in place; a map used only by the environment is untouched.
    """
    duplicates = {}
    for original in shared_cmap_originals(force, solute):
        size, energy = force.getMapParameters(original)
        duplicates[force.addMap(size, list(energy))] = original
    if not duplicates:
        return duplicates
    redirect = {original: duplicate for duplicate, original in duplicates.items()}
    for index in range(force.getNumTorsions()):
        parameters = list(force.getTorsionParameters(index))
        if parameters[0] in redirect and all(a in solute for a in parameters[1:]):
            parameters[0] = redirect[parameters[0]]
            force.setTorsionParameters(index, *parameters)
    return duplicates


def _scale_cmap(force, solute, solute_solute, duplicates=None):
    """Scale exactly the maps the solute owns, after duplicating any it has to share."""
    if duplicates is None:
        duplicates = duplicate_shared_cmaps(force, solute)
    # Computed after duplication, so the fresh copies count as the solute's own.
    targets = set(cmap_map_roles(force, solute)["exclusive_solute"]) | set(duplicates)
    for map_index in sorted(targets):
        size, energy = force.getMapParameters(map_index)
        force.setMapParameters(map_index, size, [e * solute_solute for e in energy])


#: Name of the global parameter injected into every CustomGBForce energy term. Chosen not to clash
#: with any parameter already present in the GBn2 or HCT expressions.
REST2_GB_SCALE_PARAMETER = "rest2_scale_gb"


def _scale_customgb(force, system, solute, solute_solute):
    """Scale the ENTIRE generalised-Born energy by the solute-solute factor, `(1 - tau)^2`.

    Charge scaling alone is not enough, and this is the part that is easy to get wrong. GBn2 has
    three energy terms: two are proportional to charge products and would follow
    `charge * (1 - tau)` correctly, but the third is a non-polar/dispersion correction with no
    charge dependence. It still has to scale by `(1 - tau)^2`, and scaling charges leaves it
    untouched. Multiplying every term by
    one global parameter scales all three uniformly.

    The whole system must be the enhanced region. A GB energy is not decomposable per atom the way
    a bonded term is: every Born radius depends on every other atom's position, so a partial
    selection would need a validated treatment of the solute-environment cross terms, and there is
    none here. Refused rather than approximated.

    The expressions are rewritten in place and the parameter is added to the System, so this MUST
    happen before a Context is created -- the compiled kernels have to already reference it.
    """
    missing = [i for i in range(system.getNumParticles()) if i not in solute]
    if missing:
        shown = ", ".join(str(i) for i in missing[:8])
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise ValueError(
            f"implicit REST2 requires the entire system to be the enhanced region, but "
            f"{len(missing)} of {system.getNumParticles()} particles are outside it "
            f"({shown}{more}). A generalised-Born energy is not separable per atom.")

    existing = {force.getGlobalParameterName(i)
                for i in range(force.getNumGlobalParameters())}
    if REST2_GB_SCALE_PARAMETER not in existing:
        force.addGlobalParameter(REST2_GB_SCALE_PARAMETER, 1.0)
        for term in range(force.getNumEnergyTerms()):
            expression, computation = force.getEnergyTermParameters(term)
            # Only the leading expression is scaled; everything after the first ';' defines
            # intermediate variables, and multiplying those would change what they mean.
            if ";" in expression:
                head, tail = expression.split(";", 1)
                scaled = f"{REST2_GB_SCALE_PARAMETER}*({head});{tail}"
            else:
                scaled = f"{REST2_GB_SCALE_PARAMETER}*({expression})"
            force.setEnergyTermParameters(term, scaled, computation)

    index = [force.getGlobalParameterName(i)
             for i in range(force.getNumGlobalParameters())].index(REST2_GB_SCALE_PARAMETER)
    force.setGlobalParameterDefaultValue(index, float(solute_solute))


#: Force classes this module knows how to scale. Each has an explicit `_scale_*` implementation.
SCALED_FORCE_CLASSES = frozenset({
    "NonbondedForce", "PeriodicTorsionForce", "CMAPTorsionForce", "CustomGBForce",
})

#: Energy-bearing forces left unscaled ON PURPOSE, following the standard REST2 convention. Scaling
#: bonds and angles would change the molecule's covalent geometry with tau, which is not what REST2
#: does: the solute's conformational barriers are what the scaling lowers, not its bond lengths.
DELIBERATELY_UNSCALED_FORCE_CLASSES = frozenset({
    "HarmonicBondForce", "HarmonicAngleForce",
})

#: Forces contributing no potential energy, so scaling them is meaningless rather than wrong.
ENERGY_FREE_FORCE_CLASSES = frozenset({
    "CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
    "MonteCarloFlexibleBarostat", "MonteCarloMembraneBarostat", "AndersenThermostat", "RMSDForce",
})


class UnclassifiedForceError(ValueError):
    """The System carries an energy-bearing force this module cannot scale.

    Raised instead of scaling what is recognised and leaving the rest alone. A force left at s = 1
    inside a scaled Hamiltonian is not a smaller effect -- it is a different Hamiltonian from the
    one the configuration claims, and it fails silently: the run completes, the exchange log looks
    healthy, and the acceptance ratio absorbs the discrepancy.
    """


def audit_force_classes(system, where="tau scaling"):
    """Classify every force; raise on any energy-bearing force that cannot be placed."""
    scaled, by_convention, energy_free, unknown = [], [], [], []
    for index in range(system.getNumForces()):
        name = system.getForce(index).__class__.__name__
        if name in SCALED_FORCE_CLASSES:
            scaled.append((index, name))
        elif name in DELIBERATELY_UNSCALED_FORCE_CLASSES:
            by_convention.append((index, name))
        elif name in ENERGY_FREE_FORCE_CLASSES:
            energy_free.append((index, name))
        else:
            unknown.append((index, name))
    if unknown:
        listed = ", ".join(f"force[{i}] {n}" for i, n in unknown)
        raise UnclassifiedForceError(
            f"{where}: the System carries {len(unknown)} force(s) this module cannot classify: "
            f"{listed}.\n"
            "  Refusing rather than leaving them at the wrong scale.\n"
            f"  Known scalable: {sorted(SCALED_FORCE_CLASSES)}\n"
            f"  Unscaled by convention: {sorted(DELIBERATELY_UNSCALED_FORCE_CLASSES)}\n"
            f"  Carry no potential energy: {sorted(ENERGY_FREE_FORCE_CLASSES)}")
    return {"scaled": scaled, "unscaled_by_convention": by_convention, "energy_free": energy_free}


def build_scaled_system(base_system, solute_indices, tau, excluded_bonds=(),
                        prepare_for_switching=False):
    """A copy of `base_system` with the solute Hamiltonian scaled for this rung.

    `prepare_for_switching` matters only at tau = 0, where s = 1 and the scaling arithmetic is a
    no-op. A REST2 rung there wants the untouched System and gets it. An AIS path there needs the
    System to already carry the CustomGBForce global scale parameter, because the energy
    expressions that reference it are compiled when the Context is created and cannot be rewritten
    afterwards -- so a path that starts at tau = 0 and moves away from it would have no way to
    scale the generalised-Born energy at all.
    """
    solute_solute, solute_environment = scaling_for_tau(tau)
    # Before touching anything: refuse a System carrying an energy term that cannot be placed.
    # Doing this first means the failure is "this System has a force I do not understand", not a
    # half-scaled System that looks finished.
    audit_force_classes(base_system)
    system = clone_system(base_system)
    if solute_solute == 1.0 and not prepare_for_switching:
        return system                              # the cold replica is the unmodified system
    solute = set(int(i) for i in solute_indices)
    excluded = {frozenset((int(a), int(b))) for a, b in excluded_bonds}
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce):
            _scale_nonbonded(force, solute, solute_solute, solute_environment)
        elif isinstance(force, PeriodicTorsionForce):
            _scale_torsions(force, solute, solute_solute, excluded)
        elif isinstance(force, CMAPTorsionForce):
            _scale_cmap(force, solute, solute_solute)
        elif isinstance(force, CustomGBForce):
            _scale_customgb(force, system, solute, solute_solute)
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


# ---------------------------------------------------------------------------------------------
# Switching one live Context along a tau path (AIS)
#
# The rules are the ones above. This class does not restate them: it restores the UNSCALED
# parameters from a reference copy of the base System and then calls the same `_scale_*` functions
# a static rung is built with, so a dynamically switched Context at tau and a separately built
# `build_scaled_system(..., tau)` are the same Hamiltonian by construction rather than by
# agreement between two implementations.
#
# Rebuilding the whole System per update was the obvious alternative and is not viable: a switching
# path updates every `parameter_update_interval_steps`, which is every step by default, and
# serialising a solvated System tens of thousands of times would dominate the run.
# ---------------------------------------------------------------------------------------------


class TauSwitcher:
    """Sets tau on a live Context, using the same scaling rules as a static REST2 rung."""

    def __init__(self, base_system, solute_indices, excluded_bonds=()):
        audit_force_classes(base_system, where="AIS tau switching")
        # A private, never-modified copy. Every `set_tau` starts from these parameters, so the
        # scaling is always applied to the UNSCALED Hamiltonian and never composed on top of the
        # previous tau -- which would compound s and drift the path away from its own definition.
        self.base = clone_system(base_system)
        self.solute = set(int(i) for i in solute_indices)
        self.excluded = {frozenset((int(a), int(b))) for a, b in excluded_bonds}
        # Which CMAP maps the solute has to share, and therefore which duplicates a prepared
        # System will carry. Derived from the untouched base, in the same deterministic order
        # `duplicate_shared_cmaps` appends them, so a duplicate can be matched back to its
        # original without a side table that could drift from the System.
        self._cmap_duplicates = {}
        for index in range(self.base.getNumForces()):
            force = self.base.getForce(index)
            if isinstance(force, CMAPTorsionForce):
                originals = shared_cmap_originals(force, self.solute)
                first = force.getNumMaps()
                self._cmap_duplicates[index] = {
                    first + offset: original for offset, original in enumerate(originals)}

    def prepared_system(self, tau):
        """The System to create the Context from: scaled to `tau` and ready to be switched.

        The duplicates are created HERE, once. `set_tau` afterwards only rewrites map energies:
        adding a copy per switch would grow the System without bound and compound the scaling.
        """
        return build_scaled_system(self.base, self.solute, tau, self.excluded,
                                   prepare_for_switching=True)

    def set_tau(self, context, system, tau):
        """Change `system`'s Hamiltonian to `tau` and push it into `context`.

        Returns the solute-solute factor, for the caller to record if it wants it.

        `system` must be the System the Context was created from -- the one `prepared_system`
        returned -- because `updateParametersInContext` writes through the Force objects the
        Context already holds.
        """
        solute_solute, solute_environment = scaling_for_tau(tau)
        for index in range(system.getNumForces()):
            force = system.getForce(index)
            reference = self.base.getForce(index)
            if isinstance(force, NonbondedForce):
                _restore_nonbonded(force, reference)
                _scale_nonbonded(force, self.solute, solute_solute, solute_environment)
                force.updateParametersInContext(context)
            elif isinstance(force, PeriodicTorsionForce):
                _restore_torsions(force, reference)
                _scale_torsions(force, self.solute, solute_solute, self.excluded)
                force.updateParametersInContext(context)
            elif isinstance(force, CMAPTorsionForce):
                duplicates = self._cmap_duplicates.get(index, {})
                _restore_cmap(force, reference, duplicates)
                _scale_cmap(force, self.solute, solute_solute, duplicates)
                force.updateParametersInContext(context)
            elif isinstance(force, CustomGBForce):
                # The expressions already carry the global parameter; only its value changes, and
                # a global parameter is set on the Context rather than pushed through the Force.
                context.setParameter(REST2_GB_SCALE_PARAMETER, float(solute_solute))
        return solute_solute


def _restore_nonbonded(force, reference):
    for index in range(force.getNumParticles()):
        force.setParticleParameters(index, *reference.getParticleParameters(index))
    for index in range(force.getNumExceptions()):
        force.setExceptionParameters(index, *reference.getExceptionParameters(index))


def _restore_torsions(force, reference):
    for index in range(force.getNumTorsions()):
        force.setTorsionParameters(index, *reference.getTorsionParameters(index))


def _restore_cmap(force, reference, duplicates=None):
    """Put every map back to its unscaled energies.

    A duplicate has no counterpart in the reference -- it did not exist there -- so it is restored
    from the ORIGINAL it was copied from. Without this the restore would read past the end of the
    reference and the switcher would compound scaling instead of reapplying it.
    """
    duplicates = duplicates or {}
    for index in range(force.getNumMaps()):
        source = duplicates.get(index, index)
        size, energy = reference.getMapParameters(source)
        force.setMapParameters(index, size, list(energy))
