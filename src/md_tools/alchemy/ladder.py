#!/usr/bin/env python
"""A lambda ladder: one System, K rungs, each rung a set of Context parameter values.

The runtime half of A3b (`docs/development/0.7.0/lambda-exchange-design.md`, agreed with S4 and
recorded in `docs/development/shared-contracts.md` section 5). What is here is the PROTOCOL object
and the rung bookkeeping; the exchange machinery is `md_tools.remd`'s and stays there, because a
second exchange mechanism living in the alchemy layer is how two policies end up disagreeing.

WHY THIS IS NOT A REST2 LADDER, stated once because everything below follows from it:

    tau is BAKED into a serialised System -- `build-top --rest2-scaler` writes one file per rung
    before the run, and a rung integrates that file as it is. lambda is a CONTEXT PARAMETER of ONE
    System. So a REST2 rung IS a file and a lambda rung is a set of values.

Three consequences, each of which is a rule rather than an observation:

1. **A lambda ladder has NO GROUP FILE, and a per-rung `-s` is refused even when every path is
   identical.** The REST2 ladder reads `-s` only from `remd_groupfile.<segment>`, one saved state
   per line, because there the mapping from rung to file is the thing most worth making explicit.
   Here there is one System for every rung, so the column cannot express a true statement about the
   ladder: its PRESENCE is the error, not its contents. The tempting implementation refuses only
   when the K paths DIFFER, and it passes every test anybody would naturally write, because nobody
   writes the test where they differ by accident.

2. **A rung is addressed by its INDEX, with `(lambda_j, tau_j)` as its content.** Not by lambda.
   While tau is absent -- A3b is the tau = 0 case of 0.7.1's tent path -- addressing by lambda is
   the obvious shortcut, and it would force 0.7.1 either to re-index every rung or to carry two
   addressing schemes. Two addressing schemes is how a walker is filed under the wrong state.

3. **A rung whose recorded state disagrees with its Context's parameters is refused**, the lambda
   analogue of "tau 0 on a hot state". `AlchemicalHamiltonian.context_parameters` is the ONE
   definition compared against; this module never reimplements the mapping.

`REST2Protocol` is NOT subclassed. It validates tau in ascending order, derives effective
temperatures from tau and builds one scaled System per rung -- none of which is true here, and
inheriting it would mean a lambda ladder that answers questions about tau. The schedule IS shared
(`md_tools.remd.schedule.EventSchedule`), because "every interval is an exact number of steps" is
one rule for every protocol and a second copy of it would eventually round where the first refuses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from md_tools.alchemy.hamiltonian import AlchemicalHamiltonian
from md_tools.remd.protocol import KB_KJ_PER_MOL_K, ProtocolError
from md_tools.remd.schedule import EventSchedule, ScheduleError

#: The public state coordinates, in the order a record lists them. One authority: the
#: Hamiltonian's own.
PUBLIC_PARAMETERS = tuple(AlchemicalHamiltonian.parameter_names)


class LambdaLadderError(ProtocolError):
    """This is not a lambda ladder, or not one this runtime will run.

    Subclasses `ProtocolError` so that a caller holding a protocol -- the driver, the preflight,
    a generated entry point -- catches ladder errors with the same except clause it already has.
    A separate hierarchy would mean a refusal that the existing handlers let through.
    """


@dataclass(frozen=True)
class LambdaRung:
    """One rung: its INDEX, the state it runs, and the tau that state is at.

    `tau` is 0.0 for every rung of an A3b ladder and is carried anyway, because it is the field
    0.7.1's tent path fills in and a record written without it is indistinguishable from one
    predating the field -- the same reason a CV-disabled ladder records
    `collective_variables: null` explicitly.
    """

    index: int
    state: Mapping[str, float]
    tau: float = 0.0

    def __post_init__(self):
        if int(self.index) != self.index or self.index < 0:
            raise LambdaLadderError(f"a rung index is a non-negative integer; got {self.index!r}")
        if not 0.0 <= float(self.tau) < 1.0:
            raise LambdaLadderError(
                f"rung {self.index}: tau must lie in [0, 1); got {self.tau!r}")
        # Validated THROUGH the Hamiltonian, so an out-of-range or missing component is refused
        # by the same code that would refuse it at `set_state`, with the same message.
        object.__setattr__(self, "state",
                           dict(AlchemicalHamiltonian.context_parameters(self.state)))

    @property
    def public_state(self) -> dict[str, float]:
        """Just the public coordinates: what a person reads, and what a schedule is written in."""
        return {name: float(self.state[name]) for name in PUBLIC_PARAMETERS}

    def record(self) -> dict[str, Any]:
        """This rung's identity, as it is persisted: the index, its content, and every derived
        parameter the forces actually read."""
        from md_tools.remd.amber_trajectory import state_trajectory_name

        return {"index": int(self.index),
                "trajectory": state_trajectory_name(int(self.index)),
                "tau": float(self.tau),
                "state": self.public_state,
                "context_parameters": {k: float(v) for k, v in self.state.items()}}


def rungs_from_schedule(lambdas: Sequence[float], *, taus: Sequence[float] | None = None
                        ) -> tuple[LambdaRung, ...]:
    """The common case: a diagonal schedule, one lambda per rung, indexed in order.

    The diagonal (`lambda_electrostatics = lambda_sterics = lambda_bonded`) is Amber18's own path
    and what `h.record["softcore"]["amber18_path"]` names. A ladder that needs the components to
    differ builds its `LambdaRung`s directly; this helper refuses to guess for it.
    """
    values = [float(v) for v in lambdas]
    if taus is not None and len(taus) != len(values):
        raise LambdaLadderError(
            f"{len(values)} lambda value(s) and {len(taus)} tau value(s): a rung has exactly one "
            "of each, because a rung is one point on the path")
    return tuple(LambdaRung(index=i, state=dict.fromkeys(PUBLIC_PARAMETERS, v),
                            tau=0.0 if taus is None else float(taus[i]))
                 for i, v in enumerate(values))


class LambdaLadderProtocol:
    """The scientific request for a lambda ladder, in the shape the REMD driver consumes.

        from md_tools.alchemy.ladder import LambdaLadderProtocol, rungs_from_schedule

        protocol = LambdaLadderProtocol(
            hamiltonian=h,
            rungs=rungs_from_schedule([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            temperature_k=300.0,
            timestep_fs=2.0,
            exchange_interval_ps=10.0,
            number_of_exchanges=1000,
        )

    Every rung shares ONE Hamiltonian, and `hamiltonian.record` is the ladder-level identity: end
    state digests, the topology plan's digests, the softcore settings including `sc_boundary_14`,
    kappa, the PME grid and the force-group map. Rungs differ only in `context_parameters`. A rung
    that needed a different `h` would be a different experiment, which is why there is one.
    """

    def __init__(self, *, hamiltonian, rungs, temperature_k, timestep_fs, exchange_interval_ps,
                 number_of_exchanges, whole_output_interval_ps=None,
                 solute_output_interval_ps=None, checkpoint_interval_ps=None, pressure_bar=None,
                 friction_per_ps=1.0, equilibration_ps=0.0, random_seed=None,
                 constraint_tolerance=1.0e-8, platform=None, precision=None,
                 cv_interval_steps=None, group_file=None, **unknown):
        if group_file is not None:
            raise LambdaLadderError(_NO_GROUP_FILE)
        if unknown:
            raise LambdaLadderError(f"unknown protocol field(s): {sorted(unknown)}")
        if not isinstance(hamiltonian, AlchemicalHamiltonian):
            raise LambdaLadderError(
                "a lambda ladder runs ONE AlchemicalHamiltonian and every rung shares it; got "
                f"{type(hamiltonian).__name__}")

        self.hamiltonian = hamiltonian
        self.rungs = tuple(rungs)
        if len(self.rungs) < 2:
            raise LambdaLadderError(f"a ladder needs at least 2 rungs; got {len(self.rungs)}")
        for position, rung in enumerate(self.rungs):
            if rung.index != position:
                raise LambdaLadderError(
                    f"rung {position} carries index {rung.index}. A rung is addressed by its "
                    "index, and the index IS its position in the ladder: state i is the i-th "
                    "rung, as the trajectory and CV series filed under it assume.")
        seen = {}
        for rung in self.rungs:
            key = tuple(sorted(rung.public_state.items())) + (("tau", rung.tau),)
            if key in seen:
                raise LambdaLadderError(
                    f"rungs {seen[key]} and {rung.index} are the same state {rung.public_state} "
                    f"at tau {rung.tau}. Two rungs at one state are one state sampled twice with "
                    "its statistics reported as if from two, and every exchange between them is "
                    "accepted unconditionally.")
            seen[key] = rung.index

        self.temperature_k = float(temperature_k)
        self.timestep_fs = float(timestep_fs)
        self.pressure_bar = None if pressure_bar is None else float(pressure_bar)
        self.friction_per_ps = float(friction_per_ps)
        self.random_seed = random_seed
        self.constraint_tolerance = float(constraint_tolerance)
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
                equilibration_ps=equilibration_ps,
                cv_interval_steps=cv_interval_steps)
        except ScheduleError as bad_schedule:
            raise LambdaLadderError(str(bad_schedule)) from bad_schedule

    # -- what the driver and the engine read -------------------------------------------------------

    @property
    def n_states(self):
        return len(self.rungs)

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
    def beta(self):
        """One beta for the whole ladder: every rung is at the same physical temperature, exactly
        as REST2 is. lambda changes the Hamiltonian, never the thermostat."""
        return 1.0 / (KB_KJ_PER_MOL_K * self.temperature_k)

    def production_ps(self, steps=None):
        return self.schedule.step_to_ps(self.total_steps if steps is None else steps)

    @property
    def tau(self):
        """PENDING, and loud about it: the ladder's coordinate record is a v2 -> v3 decision.

        The driver reads `protocol.tau` in seventeen places -- the reporter, the per-state
        trajectories, the CV series, `restart.json`, the exchange and lifetime statistics -- and
        for a lambda ladder every one of them wants "the coordinate of rung i", which tau is not.

        This raises rather than returning `[0.0] * K`, which would run TODAY and produce
        structurally valid files. That is precisely why it is refused (S0's ruling, 2026-09-21,
        recorded at 0.7.0 `295afe2`): a lambda ladder writing tau = 0 at every rung is
        indistinguishable in the record from a REST2 ladder that never heated, so the defect
        would be invisible in the output. Being the only option that unblocks the runtime today
        is an argument against it.

        The agreed shape is a generic `ladder_coordinates()` plus a record that NAMES its
        coordinate and can hold more than one, designed once with 0.7.1's tent path in view --
        where a rung has BOTH a tau and a lambda. It lands after the TYK2 campaign, with the
        hybrid-System output `combine-topology` grows.
        """
        raise LambdaLadderError(
            "a lambda ladder has no tau, and the ladder coordinate record that would carry its "
            "lambda is PENDING.\n"
            "  The rungs of this ladder differ in Context parameters; see "
            "`describe()['rungs']`, where each rung's index, state and derived parameters "
            "already are.\n"
            "  What is missing is the on-disk side: `protocol.tau` is what the driver writes "
            "into the NetCDF ladder record, and a lambda ladder needs a coordinate record that "
            "names its coordinate and can hold more than one -- 0.7.1's tent path gives every "
            "rung BOTH a tau and a lambda.\n"
            "  Ruled and deferred until after the TYK2 campaign (shared-contracts section 5). "
            "Returning zeros here would make this ladder's record indistinguishable from a REST2 "
            "ladder that never heated, which is why it is refused rather than merely absent.")

    def build_systems(self, base_system=None, solute_indices=None, excluded_bonds=()):
        """The System each rung propagates: THE SAME OBJECT, once per rung.

        Not a copy. A lambda rung differs from its neighbour in Context parameters alone, so K
        copies would be K chances for one of them to differ from what the preflight audited -- the
        defect `_rung_systems` removed for REST2 by consuming the preflight's Systems rather than
        rebuilding them. `base_system` and the rest are accepted and IGNORED so that the driver's
        one call site works for both protocols; passing a different base System is refused, because
        silently ignoring it would be a ladder running a System nobody asked for.
        """
        if base_system is not None and base_system is not self.hamiltonian.system:
            raise LambdaLadderError(
                "this ladder runs the Hamiltonian's own System and was handed a different one. "
                "A lambda rung is that System at recorded parameter values; there is nothing to "
                "build from another System.")
        return [self.hamiltonian.system] * self.n_states

    def prepare_context(self, state_index, context):
        """Put rung `state_index` into its state. Called once per Context, at construction.

        This is what makes a lambda ladder work at all: the Contexts are fixed to thermodynamic
        states exactly as REST2's are, but here the state lives in parameters rather than in the
        System. Everything downstream -- `reduced_potential_of` installing a neighbour's
        configuration into this Context, the per-state trajectories, the CV series -- then holds
        without change, because a Context still IS a state.
        """
        self.hamiltonian.set_state(context, self.rungs[int(state_index)].public_state)

    def verify_context(self, state_index, context, *, tolerance=0.0):
        """Refuse a Context whose parameters are not the rung's recorded state.

        The lambda analogue of "tau 0 on a hot state": a rung that claims one state and integrates
        another. Compared against `context_parameters`, never a reimplementation of the mapping --
        the derived parameters are what the forces read, so checking only the public three would
        pass a Context whose weights had been set by hand.
        """
        rung = self.rungs[int(state_index)]
        live = context.getParameters()
        wrong = {}
        for name, expected in rung.state.items():
            if name not in live:
                wrong[name] = (expected, None)
            elif abs(float(live[name]) - float(expected)) > tolerance:
                wrong[name] = (expected, float(live[name]))
        if wrong:
            raise LambdaLadderError(
                f"rung {rung.index} claims {rung.public_state} but its Context holds "
                + ", ".join(f"{k}: expected {v[0]!r}, found {v[1]!r}" for k, v in sorted(wrong.items()))
                + ". The recorded state and the Context must agree before a single step is taken.")

    def state_records(self):
        """One record per fixed thermodynamic state, in index order."""
        return [rung.record() for rung in self.rungs]

    def describe(self):
        """A plain dictionary for the records. No object, no path, no platform."""
        return {
            "protocol": "lambda-ladder",
            "rungs": self.state_records(),
            "n_states": self.n_states,
            "temperature_k": self.temperature_k,
            "timestep_fs": self.timestep_fs,
            "pressure_bar": self.pressure_bar,
            "friction_per_ps": self.friction_per_ps,
            "constraint_tolerance": self.constraint_tolerance,
            "group_file": None,        # explicit: a lambda ladder has none, and never had one
            "hamiltonian": self.hamiltonian.record,
            "schedule": self.schedule.describe(),
        }


_NO_GROUP_FILE = (
    "a lambda ladder has no group file, and a per-rung `-s` is refused even when every path is "
    "identical.\n"
    "  Every rung of a lambda ladder runs ONE System at different Context parameter values, so a "
    "column naming a System per rung cannot express a true statement about this ladder: its "
    "presence is the error, not its contents.\n"
    "  This is the same rule as `-s` on the command line being refused for a REST2 ladder even "
    "when it names the correct file.\n"
    "  The rung's lambda belongs in the resolved configuration, addressed by rung INDEX.")


def refuse_group_file(value, *, what="group_file"):
    """Refuse a group file by name, wherever one is offered to a lambda ladder.

    Exposed so the `md-run` surface refuses with this message rather than its own paraphrase: two
    wordings of one rule are two rules as soon as one of them is edited.
    """
    if value is None:
        return None
    raise LambdaLadderError(f"{what}: " + _NO_GROUP_FILE)
