#!/usr/bin/env python
"""The prepared Boltzmann phase-space reservoir rREST2 refreshes its top rung from.

Copied verbatim into every generated rREST2 project.

WHAT rREST2 IS HERE, AND WHAT IT IS NOT
    Reservoir REST2 replaces the PHASE-SPACE SAMPLE occupying the hottest rung with one drawn from
    a finite, pre-generated ensemble. The v1 contract is narrow on purpose:

        the reservoir is BOLTZMANN-weighted, generated at exactly the top rung's Hamiltonian,
        tau, temperature and fixed-volume ensemble, and holds complete samples: positions,
        VELOCITIES and box.

    Under that contract -- and only under it -- a refresh is accepted with probability one, because
    the drawn sample is a sample from the same distribution as the one it replaces.

    Roitberg, Okur and Simmerling (J. Phys. Chem. B 2007, 111, 2415; doi:10.1021/jp068335b)
    introduced reservoir REMD and derived the acceptance rule for a Boltzmann-weighted reservoir.
    Kasavajhala, Lam and Simmerling (J. Chem. Inf. Model. 2020, 60, 1218; PMCID PMC7725893) show
    what non-Boltzmann and structure-biased reservoirs require instead, and that using the
    Boltzmann rule with a reservoir that is not Boltzmann-weighted biases EVERY replica, not only
    the top one -- the ladder propagates the error down.

VELOCITY POLICY
    Both policies read the SAME source: the versioned phase-space NetCDF. It is what carries
    positions, box, absolute source step and time, the Hamiltonian identity, the atom identity and
    the completion marker in one auditable file. The policy decides what is done with the recorded
    momenta -- it does not change, relax or widen the source format.

    `stored` is the DEFAULT and installs the recorded momentum unchanged. It is the policy the
    probability-one rule is stated for: a phase-space sample is a point in phase space, and
    replacing its momenta with fresh ones is a different operation. It requires finite, correctly
    shaped, nonzero recorded velocities in every frame.

    `maxwell` is an EXPLICIT alternative. It uses the source positions and box and DELIBERATELY
    IGNORES the recorded velocity values, redrawing momenta at the one common physical temperature
    from a seed recorded per refresh, so any single draw can be reproduced from the storage alone.
    Under MPI only the owning rank calls OpenMM's draw; the resulting array is shared so every rank
    installs the same momenta.

    `maxwell` is NOT support for a DCD or any other coordinate-only source. A DCD carries no
    Hamiltonian identity, no absolute source step, and no completion marker, so it cannot satisfy
    this contract under either policy. A coordinate-only source would need its own separately
    versioned and scientifically validated contract, and there is none.

    There is NO silent fallback. A `stored` reservoir whose velocities are missing, non-finite,
    the wrong shape, or identically zero is a hard error before propagation, never a quiet switch
    to `maxwell`; the only way to redraw is to ask for it.

WHAT IS REFUSED
    a DCD or any coordinate-only source       not the phase-space format, under either policy
    zero or missing velocities under `stored` a configuration is not a phase-space sample
    sampling with replacement                 duplicates reweight the reservoir and break ordering
    explicit NPT source                       volumes would have to be exchanged and pV carried
    solute-only insertion                     a different operation with no acceptance rule here
    a different Hamiltonian                   checked on the serialized System, not on tau alone
    tau / temperature / ensemble mismatch     a different distribution
    topology, atom-order or box mismatch      the System's parameters are per index
    non-Boltzmann weighting                   needs its own separately derived acceptance rule
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from md_tools.rest2 import identity as hamiltonian_identity
from md_tools.md import phase_space
from . import source_ensemble as source_ensemble

#: The declaration format `--reservoir` points at.
DECLARATION_FORMAT = "md-tools-reservoir-request/v2"

#: The only weighting v1 implements. Anything else is refused, not approximated.
SUPPORTED_WEIGHTING = ("boltzmann",)

#: How momenta are obtained. `stored` is the default and the one the probability-one rule assumes.
VELOCITY_POLICIES = ("stored", "maxwell")
DEFAULT_VELOCITY_POLICY = "stored"


class ReservoirError(ValueError):
    """The reservoir is not one this rule may use."""


class PreparedReservoir:
    """A materialised phase-space ensemble, validated against the rung it refreshes."""

    def __init__(self, *, declaration, directory, reader, manifest, protocol, velocity_policy):
        self.declaration = declaration
        self.directory = Path(directory)
        self.reader = reader
        self.manifest = manifest
        self.protocol = protocol
        self.velocity_policy = velocity_policy

    # -- opening ---------------------------------------------------------------------------------

    @classmethod
    def open(cls, declaration_path, *, protocol, topology_path, periodic, system,
             solute_indices=(), excluded_bonds=(), coordinator=None, prepare=True):
        """Read the declaration, prepare once if needed, validate everything, and load it."""
        declaration_path = Path(declaration_path)
        if not declaration_path.is_file():
            raise ReservoirError(f"--reservoir {declaration_path} does not exist")
        declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
        if declaration.get("format") != DECLARATION_FORMAT:
            raise ReservoirError(
                f"{declaration_path.name} is {declaration.get('format')!r}, not "
                f"{DECLARATION_FORMAT!r}. v1 declarations named a coordinate-only DCD source and "
                f"cannot describe a phase-space reservoir; regenerate the project.")

        weighting = str(declaration.get("weighting", "")).lower()
        if weighting not in SUPPORTED_WEIGHTING:
            raise ReservoirError(
                f"reservoir weighting {weighting!r} is not implemented. v1 supports only "
                f"{list(SUPPORTED_WEIGHTING)}: a Boltzmann-weighted reservoir at the top rung's "
                f"own state is accepted with probability one BECAUSE it is the same distribution. "
                f"A non-Boltzmann, clustered or kinetic reservoir needs its own separately derived "
                f"acceptance rule -- using this one would bias every replica, not just the top "
                f"(Kasavajhala et al., J. Chem. Inf. Model. 2020, PMCID PMC7725893).")

        ensemble = str(declaration.get("ensemble", "NVT")).upper()
        if ensemble != "NVT":
            raise ReservoirError(
                f"reservoir ensemble {ensemble!r} is refused. v1 exchanges complete samples at "
                f"fixed volume; an NPT reservoir carries a distribution of volumes, and installing "
                f"one into a fixed-volume rung would change the density without accounting for the "
                f"pV work.")
        if declaration.get("solute_only"):
            raise ReservoirError(
                "solute_only reservoirs are refused. Inserting a solute into an unrelated solvent "
                "configuration is a different operation with a different acceptance rule, and "
                "there is none here.")

        policy = str(declaration.get("velocity_policy", DEFAULT_VELOCITY_POLICY)).lower()
        if policy not in VELOCITY_POLICIES:
            raise ReservoirError(
                f"velocity_policy {policy!r} is not one of {list(VELOCITY_POLICIES)}. `stored` is "
                f"the default and installs the recorded momentum; `maxwell` must be asked for "
                f"explicitly.")

        directory = declaration_path.parent / declaration.get("prepared_directory", "reservoir")
        prepared = directory / "reservoir.nc"

        rank = 0 if coordinator is None else coordinator.rank
        if prepare and rank == 0 and not prepared.is_file():
            cls._prepare(declaration, declaration_path, directory, prepared,
                         protocol=protocol, topology_path=topology_path, periodic=periodic,
                         system=system, solute_indices=solute_indices,
                         excluded_bonds=excluded_bonds, policy=policy)
        if coordinator is not None:
            coordinator.barrier()

        if not prepared.is_file():
            raise ReservoirError(
                f"{prepared} does not exist and preparation was not performed. A reservoir is "
                f"materialised once, before propagation, by rank 0.")
        reader = phase_space.PhaseSpaceReader(prepared)
        manifest = yaml.safe_load((directory / "reservoir.yaml").read_text(encoding="utf-8"))
        cls._validate(reader, manifest, protocol=protocol, periodic=periodic, system=system,
                      solute_indices=solute_indices, excluded_bonds=excluded_bonds,
                      policy=policy)
        return cls(declaration=declaration, directory=directory, reader=reader,
                   manifest=manifest, protocol=protocol, velocity_policy=policy)

    # -- preparation -------------------------------------------------------------------------------

    @classmethod
    def _prepare(cls, declaration, declaration_path, directory, prepared, *, protocol,
                 topology_path, periodic, system, solute_indices, excluded_bonds, policy):
        """Materialise the reservoir ONCE from the fixed-tau source's phase-space stream.

        The window selection, the evidence rules and the atom-identity comparison are the SHARED
        ones AIS established -- there is one implementation of "which frames, and is this the same
        molecule" and both methods call it. What is NOT shared is the materialised format: AIS
        writes coordinate-only `sources.dcd` because a path needs configurations, and a reservoir
        writes phase space because a refresh installs momenta.
        """
        if policy not in VELOCITY_POLICIES:
            raise ReservoirError(
                f"preparation was asked for velocity_policy {policy!r}, which is not one of "
                f"{list(VELOCITY_POLICIES)}.")

        source = declaration["source"]
        # Refused HERE, before anything is selected, written or even a directory created. The
        # alternative is worse than it looks: selection would draw a frame twice, materialisation
        # sorts the indices, and the reader's own strictly-increasing-step invariant then rejects
        # the file this code had just finished writing. Silently de-duplicating instead would
        # change both the requested count and the weights without recording either.
        #
        # It is also wrong on its own terms. A repeated draw gives one empirical configuration
        # extra statistical weight in a reservoir that is meant to be a uniform sample of the top
        # rung's distribution, and it makes the declared frame count overstate the effective
        # reservoir size -- the finite-reservoir approximation is already the weakest assumption
        # here without quietly making it weaker.
        if source.get("allow_sampling_with_replacement"):
            raise ReservoirError(
                "rREST2.reservoir.source.allow_sampling_with_replacement = true is refused.\n"
                "  A Boltzmann reservoir is materialised as DISTINCT phase-space samples in "
                "strictly increasing source order. Drawing one frame twice would give that "
                "configuration extra weight, make `frames` overstate the effective reservoir "
                "size, and produce a file this repository's own validator rejects.\n"
                "  Ask for at most as many frames as the window holds, or widen the window. AIS "
                "has its own separately documented source contract and is not affected.")
        configured = str(source["phase_space"])
        source_path = Path(configured)
        if not source_path.is_absolute():
            source_path = (declaration_path.parent.parent / configured).resolve()
        if not source_path.is_file():
            raise ReservoirError(
                f"rREST2.reservoir.source.phase_space = {configured!r} does not resolve to a "
                f"file.\n  Looked for: {source_path}\n"
                f"  A phase-space reservoir needs a source that RECORDED velocities. A fixed-tau "
                f"cMD run writes one when its phase-space interval is set; a DCD cannot be used.")

        with phase_space.PhaseSpaceReader(source_path) as reader:
            problems = reader.validate(expect_atoms=system.getNumParticles(),
                                       expect_periodic=periodic,
                                       require_velocities=(policy == "stored"))
            if problems:
                raise ReservoirError(
                    "the phase-space source is not usable:\n  - " + "\n  - ".join(problems))
            recorded = (reader.identity or {}).get("hamiltonian")
            current = hamiltonian_identity.identity_record(
                system, tau=protocol.tau[-1], temperature_k=protocol.temperature_k,
                ensemble="NVT", solute_indices=solute_indices, excluded_bonds=excluded_bonds)
            hamiltonian_identity.require_same_hamiltonian(
                recorded, current, what="reservoir source")

            times = reader.times()
            steps = reader.steps()
            request = source_ensemble.SourceRequest(
                project=declaration_path.parent.parent,
                prepared_topology=Path(topology_path),
                trajectory=configured, start_time_ps=source["start_time_ps"],
                end_time_ps=source["end_time_ps"], count=int(source["frames"]),
                # Never. The declaration cannot ask for it -- that is refused above -- and
                # stating it as a literal here means no future edit to the declaration can turn it
                # on by accident.
                allow_replacement=False,
                seed=int(declaration.get("random_seed") or protocol.random_seed or 20260830),
                implicit=not periodic, required_tau=float(protocol.tau[-1]),
                required_temperature_k=float(protocol.temperature_k),
                field_prefix="rREST2.reservoir.source", purpose="rREST2-reservoir",
                replacement_rationale=("A reservoir that draws the same sample twice is a smaller "
                                       "reservoir than it claims to be."))
            window = {"frame_interval_ps":
                      float(times[1] - times[0]) if times.size > 1 else 0.0,
                      "source": str(source_path.name)}
            distinct = source_ensemble.eligible_frames(
                list(times), float(source["start_time_ps"]), float(source["end_time_ps"]),
                frame_interval_ps=window["frame_interval_ps"])
            # The shared selector's own message offers replacement as a way out, which is right
            # for AIS and wrong here. Say what this reservoir can actually do instead, with the
            # numbers a reader needs to correct the request.
            if int(source["frames"]) > len(distinct):
                raise ReservoirError(
                    f"rREST2.reservoir.source.frames = {int(source['frames'])}, but only "
                    f"{len(distinct)} distinct source frame(s) fall in "
                    f"[{source['start_time_ps']}, {source['end_time_ps']}] ps of "
                    f"{source_path.name} (which holds {reader.n_frames} frame(s) in total, "
                    f"{window['frame_interval_ps']:g} ps apart).\n"
                    f"  This reservoir is materialised WITHOUT replacement, so it cannot hold "
                    f"more samples than the window has distinct frames. Ask for at most "
                    f"{len(distinct)}, widen the window, or record the source more often.")
            selected, eligible = source_ensemble.select_source_frames(
                request, list(times), window)
            # Stored in SOURCE order, not draw order. The reservoir is a set -- a refresh draws
            # uniformly among the stored frames, so the order they were drawn in carries nothing --
            # and writing them ordered is what lets the stored file assert strictly increasing
            # absolute steps, which is how a duplicated or truncated reservoir is caught. The
            # manifest below indexes the same sorted list, so the record and the file agree.
            selected = sorted(int(i) for i in selected)

            directory.mkdir(parents=True, exist_ok=True)
            identity = {
                "purpose": "rREST2-reservoir",
                "weighting": "boltzmann",
                "ensemble": "NVT",
                "matches_state": "top rung (tau_max)",
                "tau": float(protocol.tau[-1]),
                "temperature_k": float(protocol.temperature_k),
                "hamiltonian": current,
                "source": {
                    "phase_space_configured": configured,
                    "phase_space_resolved": str(source_path),
                    "frames_available": int(reader.n_frames),
                    "identity": reader.identity,
                },
            }
            phase_space.write_selection(prepared, source=reader, indices=selected,
                                        identity=identity)
            selected_steps = [int(steps[i]) for i in selected]
            selected_times = [float(times[i]) for i in selected]

        manifest = {
            "format": "md-tools-prepared-reservoir/v1",
            "reservoir": {
                "file": prepared.name,
                "frames": len(selected),
                "weighting": "boltzmann",
                "ensemble": "NVT",
                # The policy this reservoir was MATERIALISED under, not merely the ones the
                # format allows. It decides what was checked: `stored` requires finite, correctly
                # shaped, nonzero recorded velocities in every frame, and `maxwell` deliberately
                # does not look at the velocity values at all.
                "velocity_policy": policy,
                "velocity_policy_supported": list(VELOCITY_POLICIES),
                "velocity_policy_note": (
                    "stored: the recorded momentum is installed unchanged, which is what the "
                    "probability-one rule assumes"
                    if policy == "stored" else
                    "maxwell: the source positions and box are used and the recorded velocity "
                    "values are deliberately ignored; momenta are redrawn at the common "
                    "temperature from a seed recorded per refresh. Asked for explicitly -- "
                    "nothing falls back to it"),
                "acceptance_rule": ("probability one: the reservoir is Boltzmann-weighted at "
                                    "exactly the top rung's Hamiltonian, tau, temperature and "
                                    "fixed-volume ensemble, so a drawn phase-space sample is a "
                                    "sample from the same distribution as the one it replaces"),
                "citations": [
                    "Roitberg, Okur, Simmerling, J. Phys. Chem. B 2007, 111, 2415; "
                    "doi:10.1021/jp068335b",
                    "Kasavajhala, Lam, Simmerling, J. Chem. Inf. Model. 2020, 60, 1218; "
                    "PMCID PMC7725893",
                ],
                "finite_reservoir_approximation": (
                    f"the reservoir holds {len(selected)} samples and is NOT the top state's full "
                    f"equilibrium distribution; the assumption that the selected window represents "
                    f"that distribution is an assumption, not a result"),
            },
            "hamiltonian": current,
            "selection": {
                "window_ps": [request.start_time_ps, request.end_time_ps],
                "window_is_inclusive": True,
                "eligible_frames": len(eligible),
                "count": request.count,
                "allow_replacement": request.allow_replacement,
                "seed": request.seed,
                "selected_source_frames": [int(i) for i in selected],
                "selected_source_steps": selected_steps,
                "selected_source_times_ps": selected_times,
            },
            "source": identity["source"],
        }
        (directory / "reservoir.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False), encoding="utf-8")
        return manifest

    # -- validation ----------------------------------------------------------------------------------

    @staticmethod
    def _validate(reader, manifest, *, protocol, periodic, system, solute_indices,
                  excluded_bonds, policy):
        problems = reader.validate(expect_atoms=system.getNumParticles(),
                                   expect_periodic=periodic,
                                   require_velocities=(policy == "stored"))
        if problems:
            raise ReservoirError(
                "the prepared reservoir is not usable:\n  - " + "\n  - ".join(problems))

        # A reservoir materialised under one policy and then read under another is legible rather
        # than confusing: reusing a `maxwell`-prepared file under `stored` fails the velocity check
        # above, and this says why instead of leaving the reader with an unexplained complaint.
        prepared_under = (manifest.get("reservoir") or {}).get("velocity_policy")
        if prepared_under is not None and str(prepared_under) != policy:
            raise ReservoirError(
                f"the prepared reservoir in {getattr(reader, 'path', 'this directory')} was "
                f"materialised under velocity_policy {str(prepared_under)!r}, but this run asks "
                f"for {policy!r}. The two check different things about the stored velocities, so "
                f"the prepared file is not reusable across them.\n"
                f"  Delete the prepared reservoir directory yourself and let it be materialised "
                f"again under the policy you want, or set the declaration back to "
                f"{str(prepared_under)!r}.")

        recorded = (manifest.get("hamiltonian") or (reader.identity or {}).get("hamiltonian"))
        current = hamiltonian_identity.identity_record(
            system, tau=protocol.tau[-1], temperature_k=protocol.temperature_k, ensemble="NVT",
            solute_indices=solute_indices, excluded_bonds=excluded_bonds)
        hamiltonian_identity.require_same_hamiltonian(recorded, current, what="prepared reservoir")

        if policy == "stored":
            # Eager, and deliberately so: a missing momentum discovered at the refresh that
            # installs it would be thousands of steps into a run.
            for index in range(reader.n_frames):
                _positions, velocities, _box, _step, _time = reader.frame(index)
                if velocities.shape != (reader.n_atoms, 3):
                    raise ReservoirError(
                        f"velocity_policy is `stored` but reservoir frame {index} has velocities "
                        f"of shape {velocities.shape}, expected ({reader.n_atoms}, 3).")
                if not np.all(np.isfinite(velocities)):
                    raise ReservoirError(
                        f"velocity_policy is `stored` but reservoir frame {index} has non-finite "
                        f"velocities. There is NO silent fallback to `maxwell`: ask for it "
                        f"explicitly if that is what you want.")

    # -- access -------------------------------------------------------------------------------------

    @property
    def n_frames(self):
        return self.reader.n_frames

    def sample(self, index):
        """`(positions, velocities, box, source_step, source_time_ps)` for one prepared frame."""
        return self.reader.frame(int(index))

    def check_box_matches(self, box):
        """Every reservoir box must be the ladder's LATTICE, or the density silently changes."""
        if box is None:
            return
        for index in range(self.n_frames):
            _positions, _velocities, frame_box, _step, _time = self.reader.frame(index)
            if frame_box is None:
                raise ReservoirError(
                    f"reservoir frame {index} has no box but the ladder is explicit-solvent")
            if not source_ensemble.same_lattice(frame_box, box):
                raise ReservoirError(
                    f"reservoir frame {index} has box\n{np.asarray(frame_box)}\nbut the ladder's "
                    f"box is\n{np.asarray(box)}\nand these are not the same lattice (checked as "
                    f"an integer change of basis, so an equivalent representation would pass). A "
                    f"different box is a different density.")

    def describe(self):
        block = dict(self.manifest.get("reservoir") or {})
        block.update({
            "prepared_directory": self.directory.name,
            "frames": self.n_frames,
            "velocity_policy": self.velocity_policy,
            "velocity_policy_note": (
                "stored: the recorded momentum is installed unchanged, which is what the "
                "probability-one rule assumes" if self.velocity_policy == "stored" else
                "maxwell: momenta are redrawn at the common temperature from a recorded seed; "
                "this was asked for explicitly"),
            "units": self.reader.units,
            "hamiltonian": self.manifest.get("hamiltonian"),
            "selection": self.manifest.get("selection"),
            "source": self.manifest.get("source"),
        })
        return block

    def identity(self):
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True,
                                         default=str).encode("utf-8")).hexdigest()

    def close(self):
        self.reader.close()


# =================================================================================================
# The generic reservoir transition rule.
#
# GENERIC on purpose. rREST2 is the first user, not the only possible one: any REMD method whose
# hottest state can be refreshed from a pre-generated equilibrium ensemble composes this rule with
# `REMDRunner` rather than forking the driver. The rule knows nothing about REST2 -- it takes an
# ordinary neighbouring sweep and adds a refresh -- which is why it lives here and not in a file
# named after rREST2.
#
# Lifted from the generated `rrest2_exchange.py` plug-in, which remains as the worked example of
# the `--exchange-rule` contract. The behaviour is unchanged.
# =================================================================================================

from .rules import NeighbouringExchangeRule, RULE_INTERFACE_VERSION   # noqa: E402

#: Where the reservoir refresh sits relative to the ordinary sweep. See the module docstring.
REFRESH_ORDER = "after_neighbouring_sweep"


class ReservoirRefreshRule:
    """Neighbouring REST2 with a periodic Boltzmann reservoir refresh of the hottest state."""

    name = "rrest2-boltzmann"
    version = RULE_INTERFACE_VERSION

    def __init__(self):
        self.neighbouring = NeighbouringExchangeRule()

    def describe(self):
        return {
            "name": self.name,
            "interface_version": self.version,
            "parameters": {"refresh_order": REFRESH_ORDER},
            "schedule": ("strict alternation of odd/even adjacent-pair sweeps, plus a reservoir "
                         "refresh of the top state every refresh_interval_exchanges exchange "
                         "iterations"),
            "refresh_order": REFRESH_ORDER,
            "refresh_acceptance": ("probability one under the Boltzmann/same-state contract "
                                   "checked by rrest2_reservoir.py"),
            "criterion": "log(alpha) = [u_i(x_i)+u_j(x_j)] - [u_i(x_j)+u_j(x_i)]",
            "source": "md_tools.remd.reservoir.ReservoirRefreshRule",
        }

    def propose(self, context):
        # 1. the ordinary sweep, unchanged.
        outcome = self.neighbouring.propose(context)

        reservoir = context.reservoir
        if reservoir is None:
            return outcome

        declaration = getattr(reservoir, "declaration", {}) or {}
        interval = int(declaration.get("refresh_interval_exchanges", 1))
        if interval < 1:
            raise ValueError(
                f"refresh_interval_exchanges must be >= 1; got {interval}")

        state = dict(context.rule_state)
        attempts = int(state.get("refresh_attempts", 0))
        # Counted in EXCHANGE iterations, not segments: an interval of 1 refreshes at every
        # exchange, and the count survives a resume because the rule state is checkpointed.
        exchanges_seen = int(state.get("exchanges_seen", 0)) + 1
        state["exchanges_seen"] = exchanges_seen

        if exchanges_seen % interval != 0:
            outcome.rule_state = state
            return outcome

        # 2. the refresh, second and separately recorded.
        top_state = context.n_states - 1
        frame = int(context.rng.integers(reservoir.n_frames))
        # A DCD carries no velocities, so the installed configuration needs fresh momenta at the
        # one common temperature. The seed is derived from the rule's own stream so a resumed run
        # redraws the same way it would have.
        velocity_seed = int(context.rng.integers(1, 2 ** 31 - 1))
        state["refresh_attempts"] = attempts + 1
        state["last_refresh_frame"] = frame
        state["last_refresh_iteration"] = int(context.iteration)

        outcome.reservoir_refresh = {
            "state": top_state,
            "frame": frame,
            "velocity_seed": velocity_seed,
            "accepted": True,
            "order": REFRESH_ORDER,
        }
        outcome.rule_state = state
        outcome.diagnostics = dict(outcome.diagnostics or {})
        outcome.diagnostics["reservoir_refresh"] = {
            "state": top_state, "frame": frame, "interval_exchanges": interval}
        return outcome

