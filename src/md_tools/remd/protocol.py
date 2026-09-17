#!/usr/bin/env python
"""What a replica-exchange calculation IS: the ladder, the ensemble, and the physical schedule.

Copied verbatim into every generated replica project. This is the module a generated input imports,
and it deliberately holds no path, no MPI, no storage and no exchange loop -- only the science.

REST2 is Hamiltonian replica exchange at ONE physical thermostat temperature. tau is the persisted
source parameter and everything else is derived:

    s        = (1 - tau)^2      solute-solute terms, and solute torsions
    sqrt(s)  = 1 - tau          solute-environment terms
    1                           environment-environment terms

Because every rung shares one temperature and one beta, an exchange must NOT rescale velocities.
That rescaling belongs to temperature REMD, where the rungs differ in beta; applying it here would
inject or remove energy at every accepted swap and quietly change the ensemble being sampled.
"""
from .schedule import EventSchedule, ScheduleError, exact_steps   # noqa: F401
from md_tools.rest2 import (REST2_IMPLEMENTATION, audit_force_classes,
                           build_scaled_system, scaling_for_tau,
                           torsion_exclusion_report)

#: Boltzmann constant in the units this repository uses everywhere, kJ/mol/K.
KB_KJ_PER_MOL_K = 0.008314462618

#: 1 bar * 1 nm^3, in kJ/mol. Present so a pV term can be written down explicitly and shown to
#: cancel, rather than being omitted and trusted.
#: Re-exported so existing importers keep working; DEFINED in `core`, which a
#: reference bundle can carry and this module -- importing md_tools.rest2 -- cannot.
from .core import BAR_NM3_TO_KJ_PER_MOL  # noqa: F401


class ProtocolError(ValueError):
    """The requested protocol is not one this runtime can carry out."""


def exact_steps(duration_ps, timestep_fs, *, what):
    """`duration_ps` as a whole number of steps, or an error naming the offender.

    Rejected rather than rounded: a rounded interval means the physical time the records claim is
    not the one simulated.
    """
    timestep_ps = timestep_fs / 1000.0
    ratio = duration_ps / timestep_ps
    steps = int(round(ratio))
    if abs(ratio - steps) > 1e-9:
        raise ProtocolError(
            f"{what} ({duration_ps} ps) is not a whole number of {timestep_fs} fs steps "
            f"({ratio} steps). Rejected rather than rounded.")
    if steps < 1:
        raise ProtocolError(f"{what} ({duration_ps} ps) is shorter than one {timestep_fs} fs step.")
    return steps


def exact_multiple(numerator_ps, denominator_ps, *, what, per):
    ratio = numerator_ps / denominator_ps
    count = int(round(ratio))
    if abs(ratio - count) > 1e-9:
        raise ProtocolError(
            f"{what} ({numerator_ps} ps) must be a whole multiple of {per} ({denominator_ps} ps); "
            f"the ratio is {ratio}. Rejected rather than rounded.")
    if count < 1:
        raise ProtocolError(f"{what} ({numerator_ps} ps) is shorter than {per} "
                            f"({denominator_ps} ps).")
    return count


class REST2Protocol:
    """The scientific request. Physical intervals only; the machine's quantum is not an input.

    A generated input reads:

        from md_tools.remd import REST2Protocol

        protocol = REST2Protocol(
            tau=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            temperature_k=300.0,
            timestep_fs=2.0,
            exchange_interval_ps=10.0,
            whole_output_interval_ps=100.0,
            solute_output_interval_ps=2.0,
            number_of_exchanges=1000,
        )

    THERE IS NO `segment_ps`. It was an implementation detail -- the propagation quantum -- that
    users had to supply and that every other interval had to be a multiple of. Intervals are now
    independent, each converts to an exact integer number of steps, and the span between events is
    derived. A configuration that still carries `segment_ps` is refused with a migration message
    rather than being reinterpreted, because its old meaning (the coordinate-storage interval) is
    not the same as any single new field.
    """

    def __init__(self, *, tau, temperature_k, timestep_fs, exchange_interval_ps,
                 number_of_exchanges, whole_output_interval_ps=None,
                 solute_output_interval_ps=None, checkpoint_interval_ps=None,
                 pressure_bar=None, friction_per_ps=1.0, equilibration_ps=0.0, random_seed=None,
                 hydrogen_mass_amu=None, constraint_tolerance=1.0e-8,
                 platform=None, precision=None, cv_interval_steps=None,
                 per_tau_equilibration=None, umbrella_file=None, umbrella_restraints=None,
                 **legacy):
        if "segment_ps" in legacy:
            raise ProtocolError(
                "`segment_ps` is no longer a scientific input and its old meaning does not map "
                "onto any single new field.\n"
                "  It was the propagation quantum AND the coordinate-storage interval, so a "
                "protocol written against it stored whole-system frames at the quantum and had no "
                "separate solute stream at all.\n"
                "  Migrate deliberately:\n"
                "      whole_output_interval_ps = <how often you want COMPLETE coordinates>\n"
                "      solute_output_interval_ps = <how often you want the SOLUTE subset>\n"
                "  Both are independent of the exchange interval and of each other, and each must "
                "be a whole number of integration steps.")
        if legacy:
            raise ProtocolError(f"unknown protocol field(s): {sorted(legacy)}")

        self.tau = [float(t) for t in tau]
        if len(self.tau) < 2:
            raise ProtocolError(f"a ladder needs at least 2 states; got {len(self.tau)}")
        for value in self.tau:
            if not 0.0 <= value < 1.0:
                raise ProtocolError(f"tau must lie in [0, 1); got {value}")
        if sorted(self.tau) != self.tau:
            raise ProtocolError(
                f"tau must be given in ascending order, coldest first; got {self.tau}. The "
                f"neighbouring schedule and the round-trip definition both read state 0 as the "
                f"cold end and state {len(self.tau) - 1} as the hot end.")
        if len(set(self.tau)) != len(self.tau):
            raise ProtocolError(f"tau values must be distinct; got {self.tau}")

        self.temperature_k = float(temperature_k)
        self.timestep_fs = float(timestep_fs)
        # None means implicit solvent, or explicit solvent at fixed volume. This runtime is NVT.
        self.pressure_bar = None if pressure_bar is None else float(pressure_bar)
        self.friction_per_ps = float(friction_per_ps)
        self.random_seed = random_seed
        self.hydrogen_mass_amu = hydrogen_mass_amu
        self.constraint_tolerance = float(constraint_tolerance)
        # Where the run executes. Deliberately NOT part of `describe()`: the scientific identity a
        # continuation must agree with is physics, and resuming a run on another machine's
        # platform is legitimate. It is carried here so that a stated platform reaches the runtime
        # at all -- it used to be accepted at setup, printed in the preset, and then dropped, so
        # every replica run silently took whatever OpenMM picked.
        self.platform = platform
        self.precision = precision
        # `rest2.equilibration_per_tau`: the stages every rung runs under its own tau before the
        # first exchange, as `[{"name", "steps", "restraint_kcal_per_mol_A2"}, ...]`. None -- the
        # default, and what an empty list means -- runs none, exactly as before the field existed.
        self.per_tau_equilibration = None
        if per_tau_equilibration:
            from .rung_equilibration import validate_plan

            try:
                self.per_tau_equilibration = validate_plan(per_tau_equilibration)
            except ValueError as bad_plan:
                raise ProtocolError(str(bad_plan)) from None
        # Torsion restraints carried by every rung. The FILE is what the identity keys on -- the
        # resolved torsions come from it, and a changed file is a changed ladder -- while the
        # resolved records are what `build_systems` needs if this object builds its own rungs.
        # Both are absent unless declared, so a ladder without restraints records exactly what it
        # recorded before the field existed.
        self.umbrella_file = str(umbrella_file) if umbrella_file else None
        self.umbrella_restraints = tuple(dict(entry) for entry in (umbrella_restraints or ()))

        try:
            self.schedule = EventSchedule(
                timestep_fs=self.timestep_fs,
                exchange_interval_ps=exchange_interval_ps,
                number_of_exchanges=number_of_exchanges,
                whole_output_interval_ps=whole_output_interval_ps,
                solute_output_interval_ps=solute_output_interval_ps,
                checkpoint_interval_ps=checkpoint_interval_ps,
                equilibration_ps=equilibration_ps,
                cv_interval_steps=cv_interval_steps)
        except ScheduleError as bad_schedule:
            # The schedule is built from protocol input, so a bad interval is a protocol error to
            # everyone holding a protocol. The message is the schedule's own, unchanged.
            raise ProtocolError(str(bad_schedule)) from bad_schedule

    # -- convenience, all of it delegating to the schedule -----------------------------------------

    @property
    def number_of_exchanges(self):
        return self.schedule.number_of_exchanges

    @property
    def total_steps(self):
        return self.schedule.total_steps

    @property
    def equilibration_steps(self):
        return self.schedule.equilibration_steps

    @property
    def n_states(self):
        return len(self.tau)

    @property
    def scale_factors(self):
        """The solute-solute factor per rung. Derived for reporting only, never persisted and
        never accepted back as input: tau is the state coordinate."""
        return [scaling_for_tau(t)[0] for t in self.tau]

    @property
    def beta(self):
        """One beta for the whole ladder. Every rung is at the same physical temperature."""
        return 1.0 / (KB_KJ_PER_MOL_K * self.temperature_k)

    @property
    def cold_state(self):
        return 0

    @property
    def hot_state(self):
        return self.n_states - 1

    def production_ps(self, steps=None):
        return self.schedule.step_to_ps(self.total_steps if steps is None else steps)

    def describe(self):
        """A plain dictionary for the records. No object, no path, no platform."""
        record = {
            "method": "REST2",
            "is_temperature_remd": False,
            "rest2_implementation": dict(REST2_IMPLEMENTATION),
            "hamiltonian_scaling": ("(1 - tau)^2 solute-solute; (1 - tau) "
                                    "solute-environment; environment unchanged"),
            "tau": list(self.tau),
            "scale_factors": [float(s) for s in self.scale_factors],
            "n_states": self.n_states,
            "temperature_k": self.temperature_k,
            "single_temperature": True,
            "beta_kj_per_mol": self.beta,
            "pressure_bar": self.pressure_bar,
            "ensemble": "NVT" if self.pressure_bar is None else "NPT",
            "friction_per_ps": self.friction_per_ps,
            "constraint_tolerance": self.constraint_tolerance,
            "hydrogen_mass_amu": self.hydrogen_mass_amu,
            "random_seed": self.random_seed,
            "velocity_rescaling_on_exchange": False,
            "velocity_rescaling_note": ("every rung shares one temperature and one beta, so an "
                                        "exchange must not rescale velocities; that belongs to "
                                        "temperature REMD and would change the ensemble here"),
        }
        record.update(self.schedule.describe())
        if self.per_tau_equilibration:
            # Only when set. A continuation compares this record key by key over the UNION of both
            # sides' keys, so a key present-but-null here would refuse every ladder started before
            # it existed; an absent key is exactly what those ladders recorded.
            record["per_tau_equilibration"] = {
                "stages": [dict(stage) for stage in self.per_tau_equilibration],
                "order": ("these stages on every rung under its own tau, then "
                          "equilibration_steps, then the first exchange; none of it production"),
                "restraint": "the stage chain's, on the solute, towards the topology coordinates",
                "seeds": "derive_seed(random_seed, stage, 'state<i>') per stage and rung",
            }
        if self.umbrella_file or self.umbrella_restraints:
            # Only when set, for the same reason the block above is: a present-but-null key would
            # refuse the continuation of every ladder started before this existed.
            record["torsion_restraints"] = {
                "definition": self.umbrella_file,
                "restraints": [dict(entry) for entry in self.umbrella_restraints],
                "applied_to": "every rung, identically, after REST2 scaling",
                "scaled_by_tau": False,
                "exchange_criterion": ("identical on both rungs of every attempted swap, so the "
                                       "bias cancels from log alpha exactly"),
            }
        return record

    # -- building the ladder ------------------------------------------------------------------------

    def state_records(self):
        """One record per fixed thermodynamic state: index, trajectory, exact tau, and a derived
        effective temperature.

        `effective_temperature_k` is FOR REPORTING. It is what a temperature-REMD ladder would need
        to reach the same solute-solute weakening, and it is useful for saying how hot a rung
        "feels" -- but nothing here is thermostatted at it. Every rung runs at the one physical
        temperature; tau is the state coordinate and the only one persisted as such.

        The trajectory basename comes from the state index and never from tau, so two ladders with
        different tau at the same index name the same file.
        """
        from .amber_trajectory import state_trajectory_name

        records = []
        for index, tau in enumerate(self.tau):
            solute_solute, _ = scaling_for_tau(tau)
            records.append({
                "index": index,
                "trajectory": state_trajectory_name(index),
                "tau": float(tau),
                "effective_temperature_k": (float(self.temperature_k) / solute_solute
                                            if solute_solute else None),
            })
        return records

    def build_systems(self, base_system, solute_indices, excluded_bonds=()):
        """One scaled System per rung, from the audited REST2 scaling implementation.

        Delegates to the module-level `build_rung_systems` so that the preflight -- which builds
        these BEFORE any output exists, and hands them to the driver to consume -- and this
        method cannot construct different Systems from the same inputs. See that function.
        """
        if self.umbrella_file and not self.umbrella_restraints:
            raise ProtocolError(
                f"this ladder declares torsion restraints ({self.umbrella_file}) but was given "
                f"none resolved, so rungs built here would carry no bias while the record says "
                f"they do. The preflight resolves them against the collective-variable "
                f"definition and hands them over; build the ladder through it.")
        return build_rung_systems(base_system, solute_indices, self.tau,
                                  excluded_bonds=excluded_bonds,
                                  pressure_bar=self.pressure_bar,
                                  restraints=self.umbrella_restraints)


def apply_ladder_restraints(system, restraints):
    """Add the SAME torsion restraints to one rung, after it has been scaled. Returns their record.

    Why after, and why identical on every rung:

    * **After scaling**, because a restraint is not part of the molecular Hamiltonian REST2 weakens.
      Scaling it would make the bias itself tau-dependent, and the force classification
      (`audit_force_classes`) describes the System the ladder was built FROM, which carries no
      restraint.
    * **Identical on every rung**, because that is what makes the bias cancel from the exchange
      criterion. `reduced_potential_of` installs a configuration in a rung's Context and reads its
      potential energy, so the same `W(x)` enters `u_i` and `u_j` and drops out of `log alpha`
      exactly. A restraint that differed between rungs would enter the acceptance probability, and
      the ladder would no longer sample the restrained ensemble it claims to.

    The global parameter's DEFAULT is set to 1.0 here rather than left at zero: a Context built
    from this System is biased from its first step, with no runtime call to remember. Each
    restraint's force constant rides on its own per-torsion `scale`, as a stage's does.
    """
    from ..md.torsion_restraints import TORSION_RESTRAINT_PARAMETER, TorsionRestraint

    record = []
    by_form = {}
    for entry in restraints or ():
        entry = dict(entry)
        form = str(entry["form"])
        force = by_form.get(form)
        if force is None:
            force = by_form[form] = TorsionRestraint(system, form)
        force.add_torsion(entry["atom_indices"], entry["centre_deg"],
                          entry.get("half_width_deg") or 0.0,
                          scale=entry["force_constant_kj_mol_rad2"])
        record.append(entry)
    for force in by_form.values():
        built = system.getForce(force.force_index)
        for index in range(built.getNumGlobalParameters()):
            if built.getGlobalParameterName(index) == TORSION_RESTRAINT_PARAMETER:
                built.setGlobalParameterDefaultValue(index, 1.0)
    return record


def build_rung_systems(base_system, solute_indices, taus, *, excluded_bonds=(),
                       pressure_bar=None, restraints=(), unscaled_impropers=True):
    """One scaled System per tau rung, plus the complete force audit. THE one implementation.

    Called from two places, deliberately: `Protocol.build_systems` (the driver's route) and the
    preflight, which builds the ladder's rungs before any output exists and carries them on
    `LadderPreflight` for the driver to consume.

    Two implementations of "what System is rung i" is two answers waiting to disagree, and the
    disagreement would be invisible: both produce a plausible ladder, and only the numbers differ.
    """
    audit = audit_force_classes(base_system, where="REST2 ladder construction")
    if pressure_bar is not None:
        raise ProtocolError(
            "this runtime is NVT and installs no barostat, but a pressure was requested. "
            "Exchanging complete configurations under NPT would also have to exchange volumes "
            "and carry the pV work; that is deliberately not implemented rather than "
            "approximated.")
    systems = [build_scaled_system(base_system, solute_indices, tau,
                                   excluded_bonds=excluded_bonds,
                                   unscaled_impropers=unscaled_impropers)
               for tau in taus]
    if restraints:
        # The same bias on every rung, added after scaling; see `apply_ladder_restraints`.
        applied = [apply_ladder_restraints(system, restraints) for system in systems]
        audit["ladder_restraints"] = {
            "restraints": applied[0],
            "applied_to": "every rung, identically, after scaling",
            "scaled_by_rest2": False,
            "cancels_from_exchange": ("identical on every rung, so W(x) enters u_i and u_j alike "
                                      "and cancels from log alpha exactly"),
        }
    # What the unscaled-torsion rule actually did, in terms of the torsions it protected, recorded
    # against the SAME System the ladder was built from. A stored pair of atom indices needs
    # a force field to mean anything; this says which torsion terms it left alone.
    audit["unscaled_torsions"] = torsion_exclusion_report(
        base_system, solute_indices, excluded_bonds, unscaled_impropers)
    audit["rest2_implementation"] = dict(REST2_IMPLEMENTATION)
    return systems, audit
