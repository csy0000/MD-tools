"""What a REST2 rung IS: the scaled Hamiltonian at one tau, built from the unscaled System.

THE one implementation. `md_tools.rest2.scaler` imports every name here and re-exports it, so the
driver, the preflight, AIS switching and the reference exporter all build rungs with this code.

It imports NOTHING from md_tools -- only OpenMM -- and that is a contract, asserted by a test:
`md-openmm export-reference` copies this file byte for byte into every REST2 bundle, beside a
`verify_rungs.py` that rebuilds each bundled rung from rung 0 with it. So a bundle can show how its
scaled Systems were derived, and check that they were, with OpenMM alone -- the derivation is the
code that ran rather than a description of it.

The convention, in tau (`a = 1 - tau`):

    (1 - tau)^2   solute-solute nonbonded and 1-4, eligible solute torsions, solute CMAP
    (1 - tau)     solute-environment nonbonded, the whole generalised-Born energy
    unscaled      bonds, angles, and the UNSCALED TORSIONS: every proper torsion across an unscaled
                  central bond (ordinary amide omega, aromatic ring bond, other double bond --
                  which bonds those are is decided by `md_tools.openmm.system.unscaled_torsions`),
                  and every solute improper

Charges scale by (1 - tau) and epsilons by (1 - tau)^2, which gives exactly that split through the
Lorentz-Berthelot combining rule without a custom force.
"""

from openmm import CMAPTorsionForce, CustomGBForce, NonbondedForce, PeriodicTorsionForce, XmlSerializer


#: The Hamiltonian convention this module implements, by name and version.
#:
#: It participates in Hamiltonian identity and continuation checks: two runs agree only if they
#: agree about what "REST2" meant, and that includes which terms are left alone. `tau` is the only
#: state coordinate. There is deliberately no second variable: a derived quantity that is also
#: stored is a second thing to keep consistent, and the one that drifts is never the one you check.
REST2_IMPLEMENTATION = {
    "name": "rest2-unscaled-torsions",
    "version": 3,
    "state_coordinate": "tau",
    "solute_solute_nonbonded_scale": "(1-tau)^2",
    "solute_environment_nonbonded_scale": "1-tau",
    "generalized_born_scale": "1-tau",
    "eligible_solute_torsion_scale": "(1-tau)^2",
    "bonds": "unscaled",
    "angles": "unscaled",
    # v2 left only the ordinary amide omega unscaled. v3 is the user's decision of 2026-09-16: a hot
    # state that lets a ring pucker, a double bond twist or a planar centre pyramidalise samples
    # geometries the physical state never visits, exactly as an isomerising amide does.
    "unscaled_torsions": ["ordinary amide omega", "aromatic ring bonds", "other double bonds",
                          "impropers"],
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
    # Delegated so the powers of the amplitude are written once. A `**2` repeated in two places
    # is two chances for one of them to be changed alone -- the same reason this function exists.
    return scaling_for_amplitude(amplitude_for_tau(tau))


#: The coupling amplitude `a = 1 - tau`. Every scale factor in this convention is a power of it:
#: solute-environment terms carry `a`, solute-solute terms `a^2`, and everything else `a^0`. That
#: makes the potential an exact quadratic polynomial in `a` at frozen coordinates, which is what
#: `md_tools.ais.decomposition` measures.
def amplitude_for_tau(tau):
    """`a = 1 - tau`, the coupling amplitude. The one conversion; never stored as a coordinate."""
    return 1.0 - float(tau)


def scaling_for_amplitude(amplitude):
    """The two factors as powers of the amplitude, WITHOUT the `tau < 1` state restriction.

    `scaling_for_tau` guards `0 <= tau < 1` because tau is a protocol COORDINATE and tau = 1 is a
    fully decoupled solute that no rung and no path endpoint may sit at. This function exists for
    the other use: a momentary PROBE of the Hamiltonian at a controlled amplitude, at frozen
    coordinates, to measure a basis component. `a = 0` is exactly the measurement that isolates
    the unscaled terms, so forbidding it here would forbid the measurement rather than protect a
    state -- nothing is integrated at a probe amplitude and nothing persists it.
    """
    amplitude = float(amplitude)
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"the coupling amplitude must be in [0, 1]; got {amplitude}")
    return amplitude * amplitude, amplitude


def clone_system(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _scale_nonbonded(force, solute, solute_solute, solute_environment, targets=None):
    particles, exceptions = (targets if targets is not None
                             else (range(force.getNumParticles()),
                                   range(force.getNumExceptions())))
    for index in particles:
        charge, sigma, epsilon = force.getParticleParameters(index)
        if index in solute:
            force.setParticleParameters(index, charge * solute_environment, sigma,
                                        epsilon * solute_solute)
    for index in exceptions:
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


def system_bond_graph(system):
    """Every bonded pair in the System: `HarmonicBondForce` terms plus constraints, as frozensets.

    Constraints are included because a constrained bond (HBonds) is typically absent from the bond
    force. Water's H-H constraint joins two hydrogens that are not chemically bonded; that cannot
    matter here, because only wholly-solute torsions are ever scaled and water is never solute.
    """
    from openmm import HarmonicBondForce

    bonds = set()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, HarmonicBondForce):
            for term in range(force.getNumBonds()):
                a, b, _length, _k = force.getBondParameters(term)
                bonds.add(frozenset((int(a), int(b))))
    for term in range(system.getNumConstraints()):
        a, b, _distance = system.getConstraintParameters(term)
        bonds.add(frozenset((int(a), int(b))))
    return bonds


def torsion_kind(atoms, bonds):
    """"proper", "improper", or None when the bond graph says neither.

    Proper: the bonded chain i-j-k-l. Improper: one of the four atoms is bonded to the other
    three. Decided from the bond graph, never from atom order -- Amber writes an improper's central
    atom third, SMIRNOFF second, and a rule keyed on position would get one of them wrong.

    None is not "improper by default". A System whose bond force is incomplete would otherwise make
    every proper torsion look improper and leave it unscaled with nothing saying so.
    """
    i, j, k, l = (int(a) for a in atoms)
    if (frozenset((i, j)) in bonds and frozenset((j, k)) in bonds
            and frozenset((k, l)) in bonds):
        return "proper"
    quartet = (i, j, k, l)
    for centre in quartet:
        others = [a for a in quartet if a != centre]
        if len(others) == 3 and all(frozenset((centre, other)) in bonds for other in others):
            return "improper"
    return None


def is_improper(atoms, bonds):
    """An improper torsion: one atom bonded to the other three. See `torsion_kind`."""
    return torsion_kind(atoms, bonds) == "improper"


def torsion_is_scaled(atoms, solute, unscaled_bonds, bonds, unscaled_impropers=True):
    """THE rule for one PeriodicTorsionForce term. Every scaler and every report asks this.

    Scaled iff all four atoms are solute, its central bond is not an unscaled central bond, and --
    under `unscaled_impropers` -- it is not an improper.
    """
    i, j, k, l = (int(a) for a in atoms)
    if not (i in solute and j in solute and k in solute and l in solute):
        return False
    if frozenset((j, k)) in unscaled_bonds:
        return False
    if unscaled_impropers:
        kind = torsion_kind((i, j, k, l), bonds)
        if kind is None:
            raise ValueError(
                f"torsion {i}-{j}-{k}-{l} is neither a bonded chain nor centred on one atom "
                f"bonded to the other three, according to the System's bonds and constraints. "
                f"Whether it is an improper -- and so whether it stays unscaled -- cannot be "
                f"decided; refusing rather than guessing. Is a bond force missing?")
        if kind == "improper":
            return False
    return True


def _scale_torsions(force, solute, solute_solute, excluded_bonds, bonds, unscaled_impropers=True):
    for index in range(force.getNumTorsions()):
        i, j, k, l, periodicity, phase, k_value = force.getTorsionParameters(index)
        if torsion_is_scaled((i, j, k, l), solute, excluded_bonds, bonds, unscaled_impropers):
            force.setTorsionParameters(index, i, j, k, l, periodicity, phase,
                                       k_value * solute_solute)


#: Bumped when the unscaled-torsion classification RULES change, not when their inputs do. A record
#: carrying this version says which algorithm decided what, so a stored exclusion can be re-derived.
#: 1: ordinary amide omega. 2: plus aromatic ring bonds, other double bonds and impropers.
UNSCALED_TORSION_DETECTOR_VERSION = 2


def torsion_exclusion_report(system, solute, excluded_bonds, unscaled_impropers=True):
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
    bonds = system_bond_graph(system)
    report = {tuple(sorted(bond)): [] for bond in excluded}
    impropers = []
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
            elif not torsion_is_scaled((i, j, k, l), solute, excluded, bonds,
                                       unscaled_impropers):
                impropers.append(int(torsion))
            else:
                scaled += 1
    return {
        "detector_version": UNSCALED_TORSION_DETECTOR_VERSION,
        "excluded_central_bonds": [list(bond) for bond in sorted(report)],
        "excluded_torsion_indices": {f"{a}-{b}": indices for (a, b), indices in sorted(
            report.items())},
        "n_excluded_torsions": sum(len(v) for v in report.values()),
        "unscaled_impropers": bool(unscaled_impropers),
        "unscaled_improper_indices": impropers,
        "n_unscaled_impropers": len(impropers),
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


def cmap_targets(force, solute, duplicates=None):
    """The maps a switch must scale -- and therefore exactly the ones it must restore first.

    One rule, read by both halves. `_scale_cmap` and `_restore_cmap` disagreeing about which maps
    are in play is the one way this can go quietly wrong: a map scaled but not restored compounds
    its amplitude on every switch, and the path drifts away from the Hamiltonian it reports.
    """
    if duplicates is None:
        duplicates = duplicate_shared_cmaps(force, solute)
    # Computed after duplication, so the fresh copies count as the solute's own.
    return sorted(set(cmap_map_roles(force, solute)["exclusive_solute"]) | set(duplicates))


def _scale_cmap(force, solute, solute_solute, duplicates=None):
    """Scale exactly the maps the solute owns, after duplicating any it has to share."""
    if duplicates is None:
        duplicates = duplicate_shared_cmaps(force, solute)
    targets = cmap_targets(force, solute, duplicates)
    for map_index in sorted(targets):
        size, energy = force.getMapParameters(map_index)
        force.setMapParameters(map_index, size, [e * solute_solute for e in energy])


#: Name of the global parameter injected into every CustomGBForce energy term. Chosen not to clash
#: with any parameter already present in the GBn2 or HCT expressions.
REST2_GB_SCALE_PARAMETER = "rest2_scale_gb"


def _scale_customgb(force, system, solute, solute_environment):
    """Scale the ENTIRE generalised-Born energy by the LINEAR factor, `(1 - tau)`.

    The whole GB contribution is a solute-environment interaction: it is the solute's coupling to a
    continuum standing in for solvent, so it follows the solute-environment factor rather than the
    solute-solute one. An earlier version of this module used `(1 - tau)^2` here; that is a
    different Hamiltonian and is refused for continuation rather than reinterpreted.

    Charge scaling alone is not enough, and this is the part that is easy to get wrong. GBn2 has
    three energy terms: two are proportional to charge products and would follow `charge * (1-tau)`
    correctly, but the third is a non-polar/dispersion correction with no charge dependence. It
    still has to scale, and scaling charges leaves it untouched -- which is why every term is
    multiplied by one global parameter instead. Multiplying every term by
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
    force.setGlobalParameterDefaultValue(index, float(solute_environment))


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
                        prepare_for_switching=False, unscaled_impropers=True):
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
    bonds = system_bond_graph(system)
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce):
            _scale_nonbonded(force, solute, solute_solute, solute_environment)
        elif isinstance(force, PeriodicTorsionForce):
            _scale_torsions(force, solute, solute_solute, excluded, bonds, unscaled_impropers)
        elif isinstance(force, CMAPTorsionForce):
            _scale_cmap(force, solute, solute_solute)
        elif isinstance(force, CustomGBForce):
            _scale_customgb(force, system, solute, solute_environment)
    return system
