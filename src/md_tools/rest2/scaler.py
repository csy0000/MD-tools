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

from openmm import (CMAPTorsionForce, CustomGBForce, CustomTorsionForce, NonbondedForce,
                    PeriodicTorsionForce,
                    XmlSerializer, unit)


#: The Hamiltonian convention this module implements, by name and version.
#:
#: It participates in Hamiltonian identity and continuation checks: two runs agree only if they
#: agree about what "REST2" meant, and that includes which terms are left alone. `tau` is the only
#: state coordinate. There is deliberately no second variable: a derived quantity that is also
#: stored is a second thing to keep consistent, and the one that drifts is never the one you check.
REST2_IMPLEMENTATION = {
    "name": "rest2-no-bond-angle-omega",
    "version": 2,
    "state_coordinate": "tau",
    "solute_solute_nonbonded_scale": "(1-tau)^2",
    "solute_environment_nonbonded_scale": "1-tau",
    "generalized_born_scale": "1-tau",
    "eligible_solute_torsion_scale": "(1-tau)^2",
    "bonds": "unscaled",
    "angles": "unscaled",
    "ordinary_amide_omega": "unscaled",
}

#: Identities this build can read but must NOT continue. v1 scaled the whole generalised-Born
#: contribution by (1-tau)^2; v2 scales it by (1-tau). That is a different Hamiltonian, so a v1 run
#: cannot be extended or resumed under v2 -- the samples would come from two different ensembles.
#: v1 records stay exactly as written; nothing here rewrites history to pretend otherwise.
HISTORICAL_REST2_IMPLEMENTATIONS = {
    ("rest2-no-bond-angle-omega", 1): (
        "v1 scaled the complete generalised-Born energy by (1-tau)^2. v2 scales it by (1-tau), "
        "which is a different Hamiltonian: a v1 trajectory and a v2 trajectory do not sample the "
        "same implicit-solvent ensemble, so one cannot continue the other."),
}


def require_compatible_implementation(recorded, *, what="this run"):
    """Refuse to continue a run written under a superseded Hamiltonian identity.

    Two runs agree only if they agree about what REST2 meant. A version bump here is not
    bookkeeping -- it says the energy function changed -- so continuing across one would silently
    join samples from two different ensembles.
    """
    if not isinstance(recorded, dict) or not recorded:
        return
    name = recorded.get("name")
    version = recorded.get("version")
    if name == REST2_IMPLEMENTATION["name"] and version == REST2_IMPLEMENTATION["version"]:
        return
    reason = HISTORICAL_REST2_IMPLEMENTATIONS.get((name, version))
    current = f"{REST2_IMPLEMENTATION['name']}/v{REST2_IMPLEMENTATION['version']}"
    raise ValueError(
        f"{what} records the Hamiltonian identity {name}/v{version}, but this build implements "
        f"{current}.\n"
        f"  {reason or 'That identity is not one this build implements.'}\n"
        f"  The recorded run is left exactly as it is. Start a new dataset under {current} rather "
        f"than continuing one written under a different energy function.")


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


def linear_tau_ladder(minimum, maximum, count):
    """A linear ladder. ONE implementation, in `remd.generated.tau_ladder`; this is its name here.

    It used to compute the ladder itself, unrounded, while `tau_ladder` rounded to six places --
    two spellings of one quantity, agreeing to six decimals and no further. That is exactly enough
    to pass every eye and fail an exact comparison: a `cv_stateN.json` recorded 0.166667 from the
    ladder that RAN, a resume recomputed 0.16666666666666666 from the ladder that did not, the
    check allows 1e-12, and every CV-enabled four-rung ladder was unresumable. Found by
    interrupting a real ladder, because nothing that agrees to six places is visible in a test
    fixture.

    Making the two round identically would have left two implementations that happen to agree.
    This delegates, so there is one and they cannot drift apart again.

    `minimum` is kept in the signature -- it is public API -- and every caller passes 0.0, which
    is what `tau_ladder` assumes: state 0 is the unmodified physical Hamiltonian.
    """
    if count < 2:
        raise ValueError(f"a ladder needs at least 2 replicas; got {count}")
    if float(minimum) != 0.0:
        raise ValueError(
            f"a tau ladder starts at 0.0 -- state 0 is the unscaled Hamiltonian -- and this asks "
            f"for {minimum}. No caller in this package wants otherwise; if one does, the ladder "
            f"itself has to learn about it rather than this wrapper reimplementing it.")
    from ..remd.generated import tau_ladder

    return tau_ladder(int(count), float(maximum))


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


#: The two globals a switchable System carries, and the powers of the amplitude they hold.
#: `a` for the solute-environment factor and `a^2` for solute-solute -- exactly the pair
#: `scaling_for_amplitude` returns, so the Hamiltonian is unchanged and only its route in is.
REST2_A_PARAMETER = "rest2_a"
REST2_A2_PARAMETER = "rest2_a2"


def _has_global(force, name):
    return any(force.getGlobalParameterName(i) == name
               for i in range(force.getNumGlobalParameters()))


def switches_by_global_parameter(system):
    """Whether this System's tau is set by parameter VALUES rather than by re-upload.

    Asked of the System rather than remembered by the switcher, because the System is what a
    Context was built from and is what a resumed run deserialises. A switcher that believed it
    had reparameterised a System it had not would set two globals nothing reads and integrate a
    Hamiltonian frozen at whatever tau the System was built at -- silently, and at the right
    speed.
    """
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, NonbondedForce) and _has_global(force, REST2_A_PARAMETER):
            return True
    return False


def global_switching_refusal(system):
    """Why this System cannot be switched by global parameters, or None if it can.

    THE LONG-RANGE DISPERSION CORRECTION IS THE ONE THAT MATTERS, and it was found by measurement
    rather than by reading. A periodic NonbondedForce with `useDispersionCorrection` adds an
    analytic tail term computed from the particles' OWN epsilon values -- and OpenMM does not
    apply parameter offsets when it computes it. The re-upload path writes scaled epsilons into
    the force, so its correction moves with tau; the offset path leaves the stored epsilon at zero
    and carries the real value in an offset, so its correction does not move at all.

    Measured on 22-atom alanine in 522 TIP3P waters, double precision, comparing the two paths:

        useDispersionCorrection = True     tau 0.0: -3.8e-05   0.5: -2.61      0.9: -4.63 kJ/mol
        useDispersionCorrection = False    tau 0.0: -3.8e-05   0.5: -4.1e-05   0.9: -4.2e-05

    The error is zero at tau = 0 (where a = 1 and no scaling is applied), grows monotonically, and
    survives double precision -- so it is the Hamiltonian differing, not arithmetic.

    Turning the correction off would make the two agree and would be changing the physics to suit
    the implementation, which is not on offer. So a System that needs it keeps the re-upload,
    which is correct and slow, and this function says so.
    """
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if not isinstance(force, NonbondedForce):
            continue
        periodic = force.getNonbondedMethod() in (
            NonbondedForce.Ewald, NonbondedForce.PME, NonbondedForce.CutoffPeriodic,
            NonbondedForce.LJPME)
        if periodic and force.getUseDispersionCorrection():
            return (
                "its NonbondedForce uses the long-range dispersion correction under a periodic "
                "method. OpenMM computes that correction from the particles' stored epsilon "
                "values and does not apply parameter offsets to it, so a solute scaled through an "
                "offset would keep the tail term it had at tau = 0 while the potential around it "
                "moved -- measured at 2.6 kJ/mol at tau = 0.5 on a small solvated peptide, and "
                "growing with tau. The re-upload path is used instead: slower, and right.")
    return None


def reparameterise_for_global_switching(system, solute_indices, excluded_bonds=()):
    """Rewrite `system` so a tau change is three `setParameter` calls instead of a re-upload.

    THE SAME HAMILTONIAN, REACHED A CHEAPER WAY. Nothing here changes what is computed; the
    equivalence against `TauSwitcher`'s re-upload across tau and across configurations is the
    proof, and it is what `tests/test_global_parameter_switching.py` asserts.

    WHY IT IS WORTH DOING. `set_amplitude` rewrites every solute particle, exception and torsion
    and calls `updateParametersInContext` on each affected force. Measured here on 22-atom
    alanine (CUDA, mixed precision, ff14SB + GBn2): 2.89 ms against 0.076 ms for an energy
    evaluation -- 38x -- so a tau change was 95% of an AIS `work`-mode update and the physics
    was the rest. `CustomGBForce` already avoided this through `rest2_scale_gb`; this extends
    the same idea to the other two scaled forces.

    HOW THE LINEARITY IS ARRANGED. `addParticleParameterOffset` is linear in its global:
    `q = q_base + g*dq`. The scaling wanted is purely multiplicative, so the base is set to ZERO
    and the offset carries the original value. `a = 0` is then a genuinely decoupled solute,
    which is what a basis probe at amplitude 0 is defined to measure.

    TWO GLOBALS, NOT ONE, because charge follows `a` and epsilon follows `a^2` and an offset
    cannot square its parameter. Scaling the solute's own charge by `a` and its own epsilon by
    `a^2` produces both pair factors without either being written down: a solute-solute pair
    gets `a*a` on the charge product and `sqrt(a^2 * a^2)` on the combined epsilon, while a
    solute-environment pair gets `a` and `sqrt(a^2)`. That is the whole reason this works with
    per-particle offsets rather than per-pair ones.

    CMAP IS NOT REPARAMETERISED. OpenMM has no `CustomCMAPTorsionForce`, so there is no global to
    attach and rewriting it as a tabulated `CustomCompoundBondForce` would be reimplementing a
    force rather than rescaling one. A System carrying CMAP keeps the re-upload for that force
    alone and uses globals for everything else -- see `TauSwitcher.set_amplitude`.
    """
    solute = {int(i) for i in solute_indices}
    excluded = {frozenset((int(a), int(b))) for a, b in excluded_bonds}

    torsion_replacement = None
    for index in range(system.getNumForces()):
        force = system.getForce(index)

        if isinstance(force, NonbondedForce):
            if _has_global(force, REST2_A_PARAMETER):
                continue                            # already reparameterised; do not compound
            force.addGlobalParameter(REST2_A_PARAMETER, 1.0)
            force.addGlobalParameter(REST2_A2_PARAMETER, 1.0)
            for particle in range(force.getNumParticles()):
                if particle not in solute:
                    continue
                charge, sigma, epsilon = force.getParticleParameters(particle)
                q = charge.value_in_unit(unit.elementary_charge)
                e = epsilon.value_in_unit(unit.kilojoule_per_mole)
                force.setParticleParameters(particle, 0.0, sigma, 0.0)
                force.addParticleParameterOffset(REST2_A_PARAMETER, particle, q, 0.0, 0.0)
                force.addParticleParameterOffset(REST2_A2_PARAMETER, particle, 0.0, 0.0, e)
            for x in range(force.getNumExceptions()):
                i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(x)
                partners = int(int(i) in solute) + int(int(j) in solute)
                if partners == 0:
                    continue
                # An exception between one solute and one environment atom carries `a`, and one
                # wholly inside the solute carries `a^2` -- the same rule as the pair it replaces.
                which = REST2_A2_PARAMETER if partners == 2 else REST2_A_PARAMETER
                cp = chargeprod.value_in_unit(unit.elementary_charge ** 2)
                e = epsilon.value_in_unit(unit.kilojoule_per_mole)
                force.setExceptionParameters(x, i, j, 0.0, sigma, 0.0)
                force.addExceptionParameterOffset(which, x, cp, 0.0, e)

        elif isinstance(force, PeriodicTorsionForce):
            # No offsets on this force, so the ELIGIBLE torsions move to a CustomTorsionForce that
            # can carry one. The ineligible ones -- excluded omega bonds, and torsions not wholly
            # inside the solute -- stay exactly where they are, unscaled, which is what
            # `_scale_torsions` achieves by skipping them.
            scaled = CustomTorsionForce(
                f"{REST2_A2_PARAMETER}*k*(1 + cos(periodicity*theta - phase))")
            scaled.addGlobalParameter(REST2_A2_PARAMETER, 1.0)
            for name in ("periodicity", "phase", "k"):
                scaled.addPerTorsionParameter(name)
            scaled.setForceGroup(force.getForceGroup())
            unscaled = PeriodicTorsionForce()
            unscaled.setForceGroup(force.getForceGroup())
            for torsion in range(force.getNumTorsions()):
                i, j, k, l, periodicity, phase, height = force.getTorsionParameters(torsion)
                if (all(int(a) in solute for a in (i, j, k, l))
                        and frozenset((int(j), int(k))) not in excluded):
                    scaled.addTorsion(i, j, k, l, [
                        float(periodicity), phase.value_in_unit(unit.radian),
                        height.value_in_unit(unit.kilojoule_per_mole)])
                else:
                    unscaled.addTorsion(i, j, k, l, periodicity, phase, height)
            torsion_replacement = (index, unscaled, scaled)

    if torsion_replacement is not None:
        index, unscaled, scaled = torsion_replacement
        system.removeForce(index)
        system.addForce(unscaled)
        system.addForce(scaled)
    return system


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
            _scale_customgb(force, system, solute, solute_environment)
    return system


# The exchange mathematics that lived here -- `reduced_potential`, `exchange_log_acceptance`
# and `exchange_pairs` -- has moved to `md_tools.remd`, where the ladder that uses it lives.
# It was duplicated: `replica_engine` carried its own copies and those were the ones the
# driver actually called, while these were reached only by tests. The two were verified
# bit-identical over 2000 random inputs before this one was removed, and `exchange_pairs`
# against `alternating_pairs` for every ladder size 2..8 in both phases.
#
# Scaling a Hamiltonian and deciding whether two replicas swap are different jobs. Fixed-tau
# cMD and AIS need the first and never the second.







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
        #: How each System handed to `set_amplitude` must be switched, worked out once. See the
        #: fast path there for why this is not re-derived per switch.
        self._plans = {}
        #: Why global switching was declined for this base System, if it was. Reported rather than
        #: silent: a run that expected the cheap path and got the dear one should be able to say
        #: so, and the difference is minutes per thousand updates.
        self._refusal = None
        self._cmap_duplicates = {}
        for index in range(self.base.getNumForces()):
            force = self.base.getForce(index)
            if isinstance(force, CMAPTorsionForce):
                originals = shared_cmap_originals(force, self.solute)
                first = force.getNumMaps()
                self._cmap_duplicates[index] = {
                    first + offset: original for offset, original in enumerate(originals)}

    def prepared_system(self, tau, *, global_parameters=True):
        """The System to create the Context from: scaled to `tau` and ready to be switched.

        The duplicates are created HERE, once. `set_tau` afterwards only rewrites map energies:
        adding a copy per switch would grow the System without bound and compound the scaling.

        `global_parameters=True` (the default) additionally reparameterises the nonbonded and
        torsion terms so a later tau change is a handful of `setParameter` calls rather than a
        re-upload of every solute parameter -- 38x cheaper per switch here, and a switch was 95%
        of an AIS update. The Hamiltonian is identical either way; see
        `reparameterise_for_global_switching`.

        It is built at amplitude 1 and then moved to `tau` by setting the globals, because the
        offsets carry the UNSCALED values by construction: building at `tau` and then attaching
        offsets would scale twice. Pass `global_parameters=False` for the re-upload System, which
        is what the equivalence test compares against.
        """
        if not global_parameters:
            return build_scaled_system(self.base, self.solute, tau, self.excluded,
                                       prepare_for_switching=True)
        refusal = global_switching_refusal(self.base)
        if refusal is not None:
            # Not an error: the caller asked for the fast System and gets the correct one. The
            # switcher dispatches on what the System actually carries, so everything downstream
            # keeps working -- only the speed differs, and `switching_note` reports which it got.
            self._refusal = refusal
            return build_scaled_system(self.base, self.solute, tau, self.excluded,
                                       prepare_for_switching=True)
        system = build_scaled_system(self.base, self.solute, 0.0, self.excluded,
                                     prepare_for_switching=True)
        system = reparameterise_for_global_switching(system, self.solute, self.excluded)
        # The System is handed over already AT `tau`. A Context built from it and never switched
        # must be at the tau it was asked for, not at 0 -- otherwise the first path of a run that
        # never calls set_tau would integrate the wrong Hamiltonian.
        self._stamp_globals(system, amplitude_for_tau(tau))
        return system

    @staticmethod
    def _stamp_globals(system, amplitude):
        """Write the amplitude into the System's own default global values.

        Defaults rather than Context values, because this runs before any Context exists. A
        Context takes its initial globals from the System, so this is what makes an unswitched
        Context correct.
        """
        solute_solute, solute_environment = scaling_for_amplitude(amplitude)
        for index in range(system.getNumForces()):
            force = system.getForce(index)
            for position in range(getattr(force, "getNumGlobalParameters", lambda: 0)()):
                name = force.getGlobalParameterName(position)
                if name == REST2_A_PARAMETER:
                    force.setGlobalParameterDefaultValue(position, float(solute_environment))
                elif name == REST2_A2_PARAMETER:
                    force.setGlobalParameterDefaultValue(position, float(solute_solute))
                elif name == REST2_GB_SCALE_PARAMETER:
                    force.setGlobalParameterDefaultValue(position, float(solute_environment))

    def _base_force_like(self, force):
        """The untouched counterpart of `force` in the base System, found by CLASS.

        Reparameterising for global switching removes and re-adds the torsion forces, so indices
        no longer line up between the prepared System and the base. Matching by class is enough
        because `audit_force_classes` has already refused any System carrying two forces of a
        class this scaler scales.
        """
        for index in range(self.base.getNumForces()):
            candidate = self.base.getForce(index)
            if type(candidate) is type(force):
                return candidate
        raise UnclassifiedForceError(
            f"{type(force).__name__} has no counterpart in the base System, so it cannot be "
            f"restored to its unscaled parameters before rescaling.")

    def switching_note(self, system):
        """One line describing how this System's tau is set, and why. For the run log."""
        if switches_by_global_parameter(system):
            has_cmap = any(isinstance(system.getForce(i), CMAPTorsionForce)
                           for i in range(system.getNumForces()))
            return ("tau set by global parameters"
                    + (" (CMAP still re-uploaded: OpenMM has no CustomCMAPTorsionForce)"
                       if has_cmap else ""))
        if self._refusal:
            return f"tau set by parameter re-upload, because {self._refusal}"
        return "tau set by parameter re-upload"

    def set_tau(self, context, system, tau):
        """Change `system`'s Hamiltonian to `tau` and push it into `context`.

        Returns the solute-solute factor, for the caller to record if it wants it.

        `system` must be the System the Context was created from -- the one `prepared_system`
        returned -- because `updateParametersInContext` writes through the Force objects the
        Context already holds.
        """
        return self.set_amplitude(context, system, amplitude_for_tau(tau))

    def set_amplitude(self, context, system, amplitude):
        """The same change, addressed by coupling amplitude `a = 1 - tau` instead of by tau.

        `set_tau` is the protocol operation and this is the one underneath it, so a probe at a
        controlled amplitude and a real switch to a tau go through exactly the same restore-and-
        rescale code. A separate probe implementation would be a second Hamiltonian: it would
        agree with this one until one of them was edited, and the decomposition it measured would
        then describe a potential the path never ran under.
        """
        solute_solute, solute_environment = scaling_for_amplitude(amplitude)

        # THE FAST PATH. A System that carries the globals is switched by their VALUES: the
        # nonbonded terms and the eligible torsions read them directly, so nothing is rewritten
        # and nothing is uploaded. CMAP has no global to carry -- OpenMM has no
        # CustomCMAPTorsionForce -- so a System with CMAP still re-uploads that one force and
        # nothing else, which is why this branch does not simply return.
        #
        # Decided ONCE per System and remembered. Re-deriving it meant walking every force and
        # reading its global parameter names on every switch, which is pure overhead on the path
        # this change exists to make cheap -- it cost more than the three setParameter calls it
        # was guarding. Keyed by `id`, and validated by the plan itself: a different System
        # object gets its own entry.
        plan = self._plans.get(id(system))
        if plan is None:
            # The CMAP targets are derived here too, and once. `cmap_targets` walks every
            # torsion and every map to decide which maps the solute owns; that answer depends on
            # the System and the solute, neither of which changes between switches, so paying for
            # it per switch was the same mistake as re-deriving `by_global`.
            cmaps = {}
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                if isinstance(force, CMAPTorsionForce):
                    cmaps[i] = cmap_targets(force, self.solute,
                                            self._cmap_duplicates.get(i, {}))
            # Whether there is a CustomGBForce AT ALL. An explicit-solvent System has none, so
            # `rest2_scale_gb` is not a parameter of its Context and setting it raises. The
            # re-upload path never hit this because it only touched the parameter from inside the
            # CustomGBForce branch, which simply did not run.
            has_gb = any(isinstance(system.getForce(i), CustomGBForce)
                         for i in range(system.getNumForces()))
            # Which particles and exceptions a nonbonded switch touches, worked out once. The
            # scan itself walks every exception, so doing it per switch would reintroduce the cost
            # it exists to remove.
            plan = (switches_by_global_parameter(system), bool(cmaps), cmaps, has_gb, {})
            self._plans[id(system)] = plan
        by_global, has_cmap, cmap_plan, has_gb, nonbonded_plan = plan

        if by_global:
            context.setParameter(REST2_A_PARAMETER, float(solute_environment))
            context.setParameter(REST2_A2_PARAMETER, float(solute_solute))
            if has_gb:
                context.setParameter(REST2_GB_SCALE_PARAMETER, float(solute_environment))
            if not has_cmap:
                # Nothing else to do: three parameter values ARE the switch.
                return solute_solute

        for index in range(system.getNumForces()):
            force = system.getForce(index)
            reference = self.base.getForce(index) if index < self.base.getNumForces() else None
            if by_global and not isinstance(force, CMAPTorsionForce):
                continue                      # already carried by the globals set above
            if isinstance(force, NonbondedForce):
                targets = nonbonded_plan.get(index)
                if targets is None:
                    targets = nonbonded_plan[index] = nonbonded_targets(force, self.solute)
                _restore_nonbonded(force, reference, targets)
                _scale_nonbonded(force, self.solute, solute_solute, solute_environment, targets)
                force.updateParametersInContext(context)
            elif isinstance(force, PeriodicTorsionForce):
                _restore_torsions(force, reference)
                _scale_torsions(force, self.solute, solute_solute, self.excluded)
                force.updateParametersInContext(context)
            elif isinstance(force, CMAPTorsionForce):
                duplicates = self._cmap_duplicates.get(index, {})
                if reference is None or not isinstance(reference, CMAPTorsionForce):
                    reference = self._base_force_like(force)
                _restore_cmap(force, reference, duplicates, only=cmap_plan.get(index))
                _scale_cmap(force, self.solute, solute_solute, duplicates)
                force.updateParametersInContext(context)
            elif isinstance(force, CustomGBForce):
                # The expressions already carry the global parameter; only its value changes, and
                # a global parameter is set on the Context rather than pushed through the Force.
                context.setParameter(REST2_GB_SCALE_PARAMETER,
                                     float(solute_environment))
        return solute_solute


def nonbonded_targets(force, solute):
    """The particles and exceptions a switch touches, so the restore can touch only those.

    One rule, read by the restore and the scale alike. `_scale_nonbonded` skips every particle
    outside the solute and every exception with no solute partner; restoring them writes their own
    values back over themselves, which is correct and is pure cost.

    It is a large cost, because it is paid per water. Measured on 22-atom alanine in a small TIP3P
    box -- 406 particles, 482 exceptions -- an untargeted restore took 6.76 ms and the scale
    2.51 ms, against 0.28 ms for the GPU upload they exist to prepare: 97% of the switch was
    Python walking solvent that is never scaled.
    """
    particles = [i for i in range(force.getNumParticles()) if i in solute]
    exceptions = []
    for index in range(force.getNumExceptions()):
        i, j, *_ = force.getExceptionParameters(index)
        if int(i) in solute or int(j) in solute:
            exceptions.append(index)
    return particles, exceptions


def _restore_nonbonded(force, reference, targets=None):
    particles, exceptions = (targets if targets is not None
                             else (range(force.getNumParticles()),
                                   range(force.getNumExceptions())))
    for index in particles:
        force.setParticleParameters(index, *reference.getParticleParameters(index))
    for index in exceptions:
        force.setExceptionParameters(index, *reference.getExceptionParameters(index))


def _restore_torsions(force, reference):
    for index in range(force.getNumTorsions()):
        force.setTorsionParameters(index, *reference.getTorsionParameters(index))


def _restore_cmap(force, reference, duplicates=None, only=None):
    """Put every map back to its unscaled energies.

    A duplicate has no counterpart in the reference -- it did not exist there -- so it is restored
    from the ORIGINAL it was copied from. Without this the restore would read past the end of the
    reference and the switcher would compound scaling instead of reapplying it.

    `only` names the maps to restore. It must be exactly the set `_scale_cmap` scales: restoring
    fewer would leave a map still carrying the previous amplitude, and the next scale would
    compound onto it. `cmap_targets` derives both from one rule so they cannot disagree.
    """
    duplicates = duplicates or {}
    # ONLY the maps that scaling touches. Restoring a map that is never scaled writes its own
    # values back over itself: correct, and pure cost. ff19SB defines one map per residue type --
    # sixteen for alanine dipeptide, of which the solute's torsions reference one -- so the
    # untargeted restore did sixteen 24x24 grid rewrites per switch to change one. That was
    # 27.8 ms of the 34.8 ms a CMAP switch cost, against 3.1 ms for the upload itself.
    if only is None:
        only = range(force.getNumMaps())
    for index in only:
        source = duplicates.get(index, index)
        size, energy = reference.getMapParameters(source)
        force.setMapParameters(index, size, list(energy))
