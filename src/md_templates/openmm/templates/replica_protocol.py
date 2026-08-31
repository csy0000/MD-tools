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
from replica_schedule import EventSchedule, ScheduleError, exact_steps   # noqa: F401
from rest2_scaling import (REST2_IMPLEMENTATION, audit_force_classes,
                           build_scaled_system, scaling_for_tau)

#: Boltzmann constant in the units this repository uses everywhere, kJ/mol/K.
KB_KJ_PER_MOL_K = 0.008314462618

#: 1 bar * 1 nm^3, in kJ/mol. Present so a pV term can be written down explicitly and shown to
#: cancel, rather than being omitted and trusted.
BAR_NM3_TO_KJ_PER_MOL = 0.0602214076


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

        from replica_runtime import REST2Protocol

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
                 platform=None, precision=None, **legacy):
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

        try:
            self.schedule = EventSchedule(
                timestep_fs=self.timestep_fs,
                exchange_interval_ps=exchange_interval_ps,
                number_of_exchanges=number_of_exchanges,
                whole_output_interval_ps=whole_output_interval_ps,
                solute_output_interval_ps=solute_output_interval_ps,
                checkpoint_interval_ps=checkpoint_interval_ps,
                equilibration_ps=equilibration_ps)
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
        return record

    # -- building the ladder ------------------------------------------------------------------------

    def build_systems(self, base_system, solute_indices, excluded_bonds=()):
        """One scaled System per rung, from the audited REST2 scaling implementation."""
        audit = audit_force_classes(base_system, where="REST2 ladder construction")
        if self.pressure_bar is not None:
            raise ProtocolError(
                "this runtime is NVT and installs no barostat, but a pressure was requested. "
                "Exchanging complete configurations under NPT would also have to exchange volumes "
                "and carry the pV work; that is deliberately not implemented rather than "
                "approximated.")
        systems = [build_scaled_system(base_system, solute_indices, tau,
                                       excluded_bonds=excluded_bonds)
                   for tau in self.tau]
        return systems, audit
