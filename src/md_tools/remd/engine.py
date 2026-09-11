#!/usr/bin/env python
"""Contexts, propagation and reduced potentials. The part that touches OpenMM.

Copied verbatim into every generated replica project.

OpenMM owns System, Context, Integrator, State and every energy evaluation. This module owns only
the bookkeeping around them: which thermodynamic state a Context belongs to, which walker's
configuration is currently in it, and how a configuration moves from one to another.

THE REPRESENTATION, CHOSEN AND STATED ONCE
    Contexts are FIXED to thermodynamic states. Context `i` is built from the System scaled to
    `tau[i]` and never changes Hamiltonian. An accepted exchange moves complete CONFIGURATIONS --
    positions, velocities and box vectors together -- between two contexts.

    The alternative, walkers fixed to contexts with state assignments moving, would require either
    rebuilding a Context's System on every accepted swap or reparameterising it in place. Both are
    far more expensive than copying coordinates, and under MPI the second would make every rank
    hold every scaled System.

    What travels with a configuration is recorded so BOTH views can be reconstructed:

        state_to_walker[i]  which walker's configuration currently sits in context i
        walker_to_state[w]  which context walker w's configuration currently sits in

    They are exact inverses and every stored row is a permutation.

VELOCITIES ARE NOT RESCALED
    Every rung is thermostatted at the same temperature and shares one beta. Velocity rescaling on
    exchange belongs to temperature REMD; here it would inject or remove energy at every accepted
    swap. A configuration keeps the momenta it had.
"""
import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, unit

from .core import BAR_NM3_TO_KJ_PER_MOL


class EngineError(RuntimeError):
    """The engine cannot do what was asked on this platform or with these inputs."""


class Configuration:
    """One complete configuration: positions, velocities and box, which travel together.

    Splitting them is the classic replica-exchange error: a configuration that kept its positions
    but not its box is at a different density, and one that kept positions but not velocities has
    had its momenta silently resampled.
    """

    __slots__ = ("positions", "velocities", "box")

    def __init__(self, positions, velocities, box):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.box = None if box is None else np.asarray(box, dtype=float)

    def copy(self):
        return Configuration(self.positions.copy(), self.velocities.copy(),
                             None if self.box is None else self.box.copy())

    def as_arrays(self):
        return (self.positions, self.velocities, self.box)

    @property
    def n_atoms(self):
        return int(self.positions.shape[0])


# `build_platform` lived here and is gone.
#
# It resolved a platform NAME from `None`/"automatic" with `"CUDA" if "CUDA" in available else
# "CPU"` -- an automatic CPU fallback, in the one place a stage's refusal could not reach. A
# ladder that fell onto the CPU that way still ran, still wrote a trajectory and still reported
# success, two orders of magnitude later.
#
# There is nothing left to fall back FROM: `ReplicaRun` consumes the platform `preflight_ladder`
# resolved, through `md_tools.openmm.platform_policy`, which is the one resolver for stages,
# ladders and AIS alike. Nothing called this any more; leaving it would have left a second
# policy for someone to reach for.


def count_cuda_devices(limit=64):
    """How many CUDA devices this process can open, by opening them.

    OpenMM exposes no device count: `getPropertyDefaultValue("DeviceIndex")` is a default index,
    not a total, and reading it as one reported zero devices on a nine-GPU machine.
    """
    from openmm import Context, System, VerletIntegrator
    try:
        platform = Platform.getPlatformByName("CUDA")
    except Exception:
        return 0
    found = 0
    for index in range(limit):
        system = System()
        system.addParticle(1.0)
        try:
            context = Context(system, VerletIntegrator(0.001), platform,
                              {"DeviceIndex": str(index)})
        except Exception:
            break
        del context
        found += 1
    return found


def visible_cuda_devices(*, probe=True):
    """The device ordinals this process may use.

    `CUDA_VISIBLE_DEVICES` renumbers devices from 0 for the process, so the ordinals OpenMM wants
    are 0..n-1 -- NOT the driver values in the variable, which would address the wrong device
    whenever the list does not start at 0.
    """
    import os

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is not None:
        entries = [part.strip() for part in value.split(",") if part.strip() != ""]
        return list(range(len(entries)))
    return list(range(count_cuda_devices())) if probe else []


def select_device_for_rank(rank, size, devices):
    """Which CUDA device this rank drives, and the policy that decided it.

    Nothing binds ranks to devices automatically. Without this every rank creates its Context on
    the default device and the whole ladder runs on one GPU, silently and slowly.
    """
    if not devices:
        return None, "no visible CUDA device"
    if size <= 1:
        return str(devices[0]), "single process: first visible device"
    if len(devices) >= size:
        return str(devices[rank]), f"one rank per device ({size} ranks, {len(devices)} devices)"
    chosen = devices[rank % len(devices)]
    return str(chosen), (f"round-robin: {size} ranks share {len(devices)} device(s), "
                         f"{-(-size // len(devices))} rank(s) per device")


class ReplicaEngine:
    """The thermodynamic states this process owns, and what can be done with them.

    In a single process `owned` is every state. Under MPI it is the one state this rank drives, and
    the driver moves configurations between ranks. Both use this same class, so there is one
    propagation and one energy path rather than a serial one and a parallel one that must agree.
    """

    def __init__(self, protocol, systems, topology, *, owned=None, platform=None,
                 properties=None, seed=None):
        from openmm.app import Simulation

        self.protocol = protocol
        self.topology = topology
        self.owned = list(range(protocol.n_states)) if owned is None else sorted(int(i)
                                                                                for i in owned)
        # Whether this system HAS a box, asked of the System rather than inferred from the
        # vectors. Every OpenMM System carries default periodic box vectors -- (2,0,0),(0,2,0),
        # (0,0,2) nm unless set -- so "the vectors are non-zero" is true even for an implicit
        # system with no periodicity at all. An implicit ladder that carried that phantom box
        # then rejected its own reservoir frames for having none.
        self.periodic = bool(systems[self.owned[0]].usesPeriodicBoundaryConditions())
        self._simulations = {}
        self._integrators = {}
        for index in self.owned:
            integrator = LangevinMiddleIntegrator(
                protocol.temperature_k * unit.kelvin,
                protocol.friction_per_ps / unit.picosecond,
                protocol.timestep_fs * unit.femtosecond)
            integrator.setConstraintTolerance(protocol.constraint_tolerance)
            if seed is not None:
                # A distinct stream per state: two rungs sharing an integrator stream are one
                # realisation counted twice.
                integrator.setRandomNumberSeed(int(seed) + 977 * index)
            simulation = Simulation(topology, systems[index], integrator, platform, properties)
            self._simulations[index] = simulation
            self._integrators[index] = integrator

    # -- configurations -------------------------------------------------------------------------

    def set_configuration(self, state_index, configuration):
        context = self._simulations[state_index].context
        if configuration.box is not None:
            context.setPeriodicBoxVectors(*[list(row) for row in configuration.box])
        context.setPositions(configuration.positions * unit.nanometer)
        context.setVelocities(configuration.velocities * unit.nanometer / unit.picosecond)

    def get_configuration(self, state_index):
        context = self._simulations[state_index].context
        state = context.getState(getPositions=True, getVelocities=True)
        box = None
        if self.periodic:
            vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            box = np.array(vectors, dtype=float)
        return Configuration(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            box)

    def set_velocities_to_temperature(self, state_index, seed):
        """Fresh Maxwell momenta at the one common temperature, from a recorded seed.

        Used where a configuration arrives without velocities -- a DCD carries none -- and never
        as part of an exchange.
        """
        self._simulations[state_index].context.setVelocitiesToTemperature(
            self.protocol.temperature_k * unit.kelvin, int(seed))

    # -- propagation and energy -------------------------------------------------------------------

    def propagate(self, state_index, steps):
        self._simulations[state_index].step(int(steps))

    def potential_energy(self, state_index):
        """U in kJ/mol for whatever configuration is currently in this context."""
        state = self._simulations[state_index].context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

    def reduced_potential_of(self, state_index, configuration):
        """u = beta * (U + pV) for `configuration` evaluated in state `state_index`.

        The configuration is INSTALLED and the energy evaluated. It is never inferred by scaling a
        previously computed energy: the scaled Hamiltonians differ by more than a single factor --
        solute-solute goes as s while solute-environment goes as sqrt(s), and torsions and CMAP
        follow their own rules -- so any such shortcut would be a different number that happens to
        look plausible.

        The context's own configuration is restored afterwards, so evaluating a cross energy never
        disturbs the run.
        """
        saved = self.get_configuration(state_index)
        try:
            self.set_configuration(state_index, configuration)
            energy = self.potential_energy(state_index)
            volume = None
            if configuration.box is not None:
                volume = float(abs(np.linalg.det(configuration.box)))
        finally:
            self.set_configuration(state_index, saved)
        return reduced_potential(energy, self.protocol.beta,
                                 pressure_bar=self.protocol.pressure_bar, volume_nm3=volume)

    def minimise(self, state_index, max_iterations=0):
        self._simulations[state_index].minimizeEnergy(maxIterations=int(max_iterations))

    def platform_name(self, state_index):
        return self._simulations[state_index].context.getPlatform().getName()

    def integrator_state(self, state_index):
        """The OpenMM checkpoint for one context, as bytes. Platform-specific by construction."""
        return self._simulations[state_index].context.createCheckpoint()

    def load_integrator_state(self, state_index, blob):
        self._simulations[state_index].context.loadCheckpoint(blob)


def reduced_potential(energy_kj_mol, beta, *, pressure_bar=None, volume_nm3=None):
    """u = beta * (U + pV), the general form.

    REST2 rungs share one temperature and one pressure, so the pV terms cancel exactly in the
    acceptance criterion: a swap moves each configuration -- positions AND its box -- to the other
    rung, and the same two volumes appear on both sides. The term is carried anyway so the
    cancellation is arithmetic that can be checked rather than an omission that must be trusted.
    """
    u = beta * energy_kj_mol
    if pressure_bar is not None and volume_nm3 is not None:
        u += beta * pressure_bar * volume_nm3 * BAR_NM3_TO_KJ_PER_MOL
    return u


def exchange_log_acceptance(u_ii, u_jj, u_ij, u_ji):
    """log of the Metropolis criterion for swapping the configurations in states i and j.

        log(alpha) = [u_i(x_i) + u_j(x_j)] - [u_i(x_j) + u_j(x_i)]

    where u_i(x_j) is state j's configuration evaluated in state i's Hamiltonian. All four are
    evaluated independently.
    """
    return (u_ii + u_jj) - (u_ij + u_ji)
