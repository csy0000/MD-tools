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

# The rung-building code lives in `hamiltonian.py`, which imports only OpenMM so that a
# reference bundle can carry it verbatim. Re-exported here: one implementation, every caller
# unchanged.
from .hamiltonian import (  # noqa: F401
    REST2_IMPLEMENTATION,
    scaling_for_tau,
    amplitude_for_tau,
    scaling_for_amplitude,
    clone_system,
    _scale_nonbonded,
    _scale_torsions,
    UNSCALED_TORSION_DETECTOR_VERSION,
    system_bond_graph,
    is_improper,
    torsion_kind,
    torsion_is_scaled,
    torsion_exclusion_report,
    cmap_map_roles,
    shared_cmap_originals,
    duplicate_shared_cmaps,
    cmap_targets,
    _scale_cmap,
    REST2_GB_SCALE_PARAMETER,
    _scale_customgb,
    SCALED_FORCE_CLASSES,
    DELIBERATELY_UNSCALED_FORCE_CLASSES,
    ENERGY_FREE_FORCE_CLASSES,
    UnclassifiedForceError,
    audit_force_classes,
    build_scaled_system,
)


#: Identities this build can read but must NOT continue. v1 scaled the whole generalised-Born
#: contribution by (1-tau)^2; v2 scales it by (1-tau). That is a different Hamiltonian, so a v1 run
#: cannot be extended or resumed under v2 -- the samples would come from two different ensembles.
#: v1 records stay exactly as written; nothing here rewrites history to pretend otherwise.
HISTORICAL_REST2_IMPLEMENTATIONS = {
    ("rest2-no-bond-angle-omega", 1): (
        "v1 scaled the complete generalised-Born energy by (1-tau)^2. v2 scales it by (1-tau), "
        "which is a different Hamiltonian: a v1 trajectory and a v2 trajectory do not sample the "
        "same implicit-solvent ensemble, so one cannot continue the other."),
    ("rest2-no-bond-angle-omega", 2): (
        "v2 left only the ordinary amide omega unscaled and scaled aromatic ring torsions, other "
        "double-bond torsions and every solute improper by (1-tau)^2. rest2-unscaled-torsions v3 "
        "leaves all of those unscaled, which is a different Hamiltonian at every tau > 0: a v2 "
        "ladder and a v3 ladder do not sample the same hot ensembles, so one cannot continue the "
        "other."),
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
    bonds = system_bond_graph(system)

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
                if torsion_is_scaled((i, j, k, l), solute, excluded, bonds):
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
                _scale_torsions(force, self.solute, solute_solute, self.excluded,
                                system_bond_graph(self.base))
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
