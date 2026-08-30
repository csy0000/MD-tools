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
from rest2_scaling import (audit_force_classes, build_scaled_system, scale_factor_for_tau)

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
    """The scientific request. Everything concrete about the machine lives outside it.

    A generated input reads:

        from replica_runtime import REST2Protocol

        protocol = REST2Protocol(
            tau=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            temperature_k=300.0,
            timestep_fs=4.0,
            segment_ps=2.0,
            exchange_interval_ps=10.0,
            whole_output_interval_ps=10.0,
            number_of_exchanges=1000,
        )

    `segment_ps` is the propagation quantum and the stored-frame interval. `exchange_interval_ps`
    must be a whole multiple of it: exchange is attempted every `exchange_stride` segments, which
    is what lets coordinates be stored more often than exchanges are attempted without any frame
    being written twice.
    """

    def __init__(self, *, tau, temperature_k, timestep_fs, segment_ps, exchange_interval_ps,
                 number_of_exchanges, whole_output_interval_ps=None, pressure_bar=None,
                 friction_per_ps=1.0, equilibration_ps=0.0, random_seed=None,
                 hydrogen_mass_amu=None, constraint_tolerance=1.0e-8):
        self.tau = [float(t) for t in tau]
        if len(self.tau) < 2:
            raise ProtocolError(f"a ladder needs at least 2 states; got {len(self.tau)}")
        for value in self.tau:
            if not 0.0 <= value < 1.0:
                raise ProtocolError(f"tau must lie in [0, 1); got {value}")
        if sorted(self.tau) != self.tau:
            raise ProtocolError(
                f"tau must be given in ascending order, coldest first; got {self.tau}. The "
                f"neighbouring exchange schedule and the round-trip definition both read state 0 "
                f"as the cold end and state {len(self.tau) - 1} as the hot end.")
        if len(set(self.tau)) != len(self.tau):
            raise ProtocolError(f"tau values must be distinct; got {self.tau}")

        self.temperature_k = float(temperature_k)
        self.timestep_fs = float(timestep_fs)
        self.segment_ps = float(segment_ps)
        self.exchange_interval_ps = float(exchange_interval_ps)
        self.number_of_exchanges = int(number_of_exchanges)
        self.whole_output_interval_ps = float(whole_output_interval_ps
                                              if whole_output_interval_ps is not None
                                              else exchange_interval_ps)
        # None means implicit solvent, or explicit solvent held at fixed volume. This runtime is
        # NVT: it never installs a barostat, because an exchange of complete configurations under
        # NPT would also have to exchange volumes and carry the pV work, and that is not v1.
        self.pressure_bar = None if pressure_bar is None else float(pressure_bar)
        self.friction_per_ps = float(friction_per_ps)
        self.equilibration_ps = float(equilibration_ps)
        self.random_seed = random_seed
        self.hydrogen_mass_amu = hydrogen_mass_amu
        self.constraint_tolerance = float(constraint_tolerance)

        if self.number_of_exchanges < 1:
            raise ProtocolError(
                f"number_of_exchanges must be >= 1; got {self.number_of_exchanges}")

        self.steps_per_segment = exact_steps(self.segment_ps, self.timestep_fs,
                                             what="the segment duration")
        self.exchange_stride = exact_multiple(
            self.exchange_interval_ps, self.segment_ps,
            what="the exchange interval", per="the segment duration")
        self.whole_output_stride = exact_multiple(
            self.whole_output_interval_ps, self.segment_ps,
            what="the whole-system output interval", per="the segment duration")
        self.equilibration_segments = 0
        if self.equilibration_ps > 0:
            self.equilibration_segments = exact_multiple(
                self.equilibration_ps, self.segment_ps,
                what="the per-state equilibration duration", per="the segment duration")

        self.total_segments = self.number_of_exchanges * self.exchange_stride

    # -- derived, and labelled as derived ---------------------------------------------------------

    @property
    def n_states(self):
        return len(self.tau)

    @property
    def scale_factors(self):
        """s = (1 - tau)^2 for every rung. Derived; never accepted back as input."""
        return [scale_factor_for_tau(t) for t in self.tau]

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

    def production_ps(self, segments=None):
        return (self.total_segments if segments is None else segments) * self.segment_ps

    def describe(self):
        """A plain dictionary for the records. No object, no path, no platform."""
        return {
            "method": "REST2",
            "is_temperature_remd": False,
            "hamiltonian_scaling": ("s = (1 - tau)^2 solute-solute; sqrt(s) = 1 - tau "
                                    "solute-environment; environment unchanged"),
            "tau": list(self.tau),
            "scale_factors": [float(s) for s in self.scale_factors],
            "n_states": self.n_states,
            "temperature_k": self.temperature_k,
            "single_temperature": True,
            "beta_kj_per_mol": self.beta,
            "pressure_bar": self.pressure_bar,
            "ensemble": "NVT" if self.pressure_bar is None else "NPT",
            "timestep_fs": self.timestep_fs,
            "friction_per_ps": self.friction_per_ps,
            "constraint_tolerance": self.constraint_tolerance,
            "hydrogen_mass_amu": self.hydrogen_mass_amu,
            "segment_ps": self.segment_ps,
            "steps_per_segment": self.steps_per_segment,
            "exchange_interval_ps": self.exchange_interval_ps,
            "exchange_stride_segments": self.exchange_stride,
            "whole_output_interval_ps": self.whole_output_interval_ps,
            "whole_output_stride_segments": self.whole_output_stride,
            "number_of_exchanges": self.number_of_exchanges,
            "total_segments": self.total_segments,
            "production_ps_per_replica": self.production_ps(),
            "equilibration_ps": self.equilibration_ps,
            "equilibration_segments": self.equilibration_segments,
            "random_seed": self.random_seed,
            "velocity_rescaling_on_exchange": False,
            "velocity_rescaling_note": ("every rung shares one temperature and one beta, so an "
                                        "exchange must not rescale velocities; that belongs to "
                                        "temperature REMD and would change the ensemble here"),
        }

    # -- building the ladder ------------------------------------------------------------------------

    def build_systems(self, base_system, solute_indices, excluded_bonds=()):
        """One scaled System per rung, from the audited REST2 scaling implementation.

        The audit runs FIRST, so a System carrying an energy term this repository cannot place
        fails as "there is a force I do not understand" rather than as a half-scaled ladder.
        """
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
