#!/usr/bin/env python
"""The prepared Boltzmann reservoir rREST2 refreshes its top state from.

Copied verbatim into every generated rREST2 project.

WHAT rREST2 IS HERE, AND WHAT IT IS NOT
    Reservoir REST2 replaces the configuration occupying the HOTTEST rung with one drawn from a
    finite, pre-generated ensemble, instead of waiting for that rung to sample it. The v1 contract
    is narrow on purpose:

        the reservoir is BOLTZMANN-weighted, at exactly the tau, temperature, Hamiltonian and
        fixed-volume ensemble of the top rung, and holds COMPLETE configurations.

    Under that contract -- and only under it -- a refresh of the top state is accepted with
    probability one, because the reservoir and the state it replaces are the same distribution.
    That is the entire justification, and it is why every one of those conditions is checked rather
    than assumed.

    Roitberg, Okur and Simmerling (J. Phys. Chem. B 2007, 111, 2415; doi:10.1021/jp068335b)
    introduced reservoir REMD and derived the acceptance rule for a Boltzmann-weighted reservoir.
    Kasavajhala, Lam and Simmerling (J. Chem. Inf. Model. 2020, 60, 1218; PMCID PMC7725893) show
    what non-Boltzmann and structure-biased reservoirs require instead, and that using the
    Boltzmann rule with a reservoir that is not Boltzmann-weighted biases EVERY replica, not only
    the top one -- the ladder propagates the error down. A non-Boltzmann reservoir therefore needs
    its own separately derived and separately tested acceptance rule, and this module refuses one
    rather than reusing a criterion that does not apply to it.

    The finite-reservoir approximation is real and is stated in every record: a reservoir of N
    configurations is not the top state's full equilibrium distribution, and the assumption that
    the selected source window represents that distribution is an assumption, not a result.

WHAT IS REFUSED, AND WHY
    explicit NPT source        volumes would have to be exchanged and pV work carried; not v1
    solute-only insertion      grafting a solute into unrelated solvent is a different, unvalidated
                               operation with no acceptance rule here
    tau mismatch               a different tau is a different distribution
    temperature mismatch       a different temperature is a different Boltzmann distribution
    topology/atom-order        the System's parameters are per index
    box mismatch               a different box is a different density
    non-Boltzmann weighting    see above; refused rather than approximated
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import source_ensemble

#: The declaration format `--reservoir` points at.
DECLARATION_FORMAT = "md-templates-reservoir-request/v1"

#: The only weighting v1 implements. Anything else is refused, not approximated.
SUPPORTED_WEIGHTING = ("boltzmann",)


class ReservoirError(ValueError):
    """The reservoir is not one this rule may use."""


class PreparedReservoir:
    """A materialised set of complete configurations, validated against the state it refreshes."""

    def __init__(self, *, declaration, manifest, directory, frames, topology_path, protocol):
        self.declaration = declaration
        self.manifest = manifest
        self.directory = Path(directory)
        self._frames = frames
        self.topology_path = Path(topology_path)
        self.protocol = protocol

    # -- opening -------------------------------------------------------------------------------

    @classmethod
    def open(cls, declaration_path, *, protocol, topology_path, periodic, coordinator=None,
             prepare=True):
        """Read the declaration, prepare once if needed, validate, and load the configurations."""
        declaration_path = Path(declaration_path)
        if not declaration_path.is_file():
            raise ReservoirError(f"--reservoir {declaration_path} does not exist")
        declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
        if declaration.get("format") != DECLARATION_FORMAT:
            raise ReservoirError(
                f"{declaration_path.name} is {declaration.get('format')!r}, not "
                f"{DECLARATION_FORMAT!r}")

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
                f"reservoir ensemble {ensemble!r} is refused. v1 exchanges complete configurations "
                f"at fixed volume; an NPT reservoir would also have to exchange volumes and carry "
                f"the pV work, which is deliberately not implemented rather than approximated.")
        if declaration.get("solute_only"):
            raise ReservoirError(
                "solute_only reservoirs are refused. Inserting a solute into an unrelated solvent "
                "configuration is a different operation with a different acceptance rule, and "
                "there is none here.")

        directory = declaration_path.parent / declaration.get("prepared_directory", "reservoir")
        request = cls._request(declaration, protocol, declaration_path, topology_path, periodic)

        rank = 0 if coordinator is None else coordinator.rank
        needs_preparation = not (directory / "prepared_source.yaml").is_file()
        if needs_preparation and prepare and rank == 0:
            resolved = source_ensemble.resolve_source_ensemble(request, require_temperature=True)
            cls._check_source(resolved, declaration, protocol)
            source_ensemble.write_prepared_source(
                request, resolved, directory,
                extra={"reservoir": cls._contract(declaration, protocol, resolved)})
        if coordinator is not None:
            coordinator.barrier()

        manifest = source_ensemble.read_prepared_source(
            directory, expected_identity=None)
        cls._validate_prepared(manifest, declaration, protocol)
        frames = source_ensemble.read_prepared_frames(directory, manifest, topology_path)
        cls._validate_frames(frames, protocol, manifest)
        return cls(declaration=declaration, manifest=manifest, directory=directory,
                   frames=frames, topology_path=topology_path, protocol=protocol)

    @staticmethod
    def _request(declaration, protocol, declaration_path, topology_path, periodic):
        source = declaration["source"]
        return source_ensemble.SourceRequest(
            project=declaration_path.parent.parent,
            prepared_topology=Path(topology_path),
            trajectory=source["trajectory"],
            topology=source.get("topology"),
            start_time_ps=source["start_time_ps"],
            end_time_ps=source["end_time_ps"],
            count=int(source["frames"]),
            allow_replacement=bool(source.get("allow_sampling_with_replacement", False)),
            seed=int(declaration.get("random_seed") or protocol.random_seed or 20260830),
            # NOT `pressure_bar is None`: this runtime is NVT, so pressure is None for an
            # explicit fixed-volume ladder too. Whether there IS a box is a property of the
            # System, and asking the wrong question prepared an explicit reservoir with no boxes
            # at all -- which the ladder then rightly refused.
            implicit=(not periodic),
            required_tau=float(protocol.tau[-1]),
            required_temperature_k=float(protocol.temperature_k),
            declared_tau=source.get("source_tau"),
            declared_temperature_k=source.get("source_temperature_k"),
            first_frame_time_ps=source.get("first_frame_time_ps"),
            frame_interval_ps=source.get("frame_interval_ps"),
            field_prefix="rREST2.reservoir.source",
            purpose="rREST2-reservoir",
            replacement_rationale=("A reservoir that draws the same configuration twice is a "
                                   "smaller reservoir than it claims to be."),
        )

    # -- the contract, checked rather than assumed -----------------------------------------------------

    @staticmethod
    def _check_source(resolved, declaration, protocol):
        """The source must BE the top rung's ensemble, not merely resemble it."""
        tau = float(resolved["tau"])
        if abs(tau - float(protocol.tau[-1])) > 1e-9:
            raise ReservoirError(
                f"the reservoir source is at tau = {tau} ({resolved['tau_evidence']}) but the top "
                f"rung is at tau = {protocol.tau[-1]}. A Boltzmann reservoir is accepted with "
                f"probability one only because it is the SAME distribution as the state it "
                f"refreshes.")
        temperature = resolved["temperature_k"]
        if temperature is None or abs(float(temperature) - protocol.temperature_k) > 1e-9:
            raise ReservoirError(
                f"the reservoir source is at {temperature} K "
                f"({resolved['temperature_evidence']}) but the ladder is at "
                f"{protocol.temperature_k} K. A different temperature is a different Boltzmann "
                f"distribution.")
        record, _ = source_ensemble.companion_record(resolved["paths"]["trajectory"])
        if isinstance(record, dict):
            ensemble = str((record.get("ensemble") or record.get("common", {}).get("ensemble")
                            or "")).upper()
            if ensemble == "NPT":
                raise ReservoirError(
                    "the reservoir source's own runtime record says it ran under NPT. v1 requires "
                    "a fixed-volume source: an NPT reservoir carries a distribution of volumes, "
                    "and exchanging one of its configurations into a fixed-volume rung would "
                    "change the density without accounting for the pV work.")

    @staticmethod
    def _contract(declaration, protocol, resolved):
        """What is written into the prepared manifest, so the assumptions are on the record."""
        return {
            "weighting": "boltzmann",
            "ensemble": "NVT",
            "matches_state": "top rung (tau_max)",
            "tau": float(protocol.tau[-1]),
            "temperature_k": float(protocol.temperature_k),
            "acceptance_rule": ("probability one: the reservoir is Boltzmann-weighted at exactly "
                                "the top rung's tau, temperature, Hamiltonian and fixed-volume "
                                "ensemble, so a drawn configuration is a sample from the same "
                                "distribution as the one it replaces"),
            "citations": [
                "Roitberg, Okur, Simmerling, J. Phys. Chem. B 2007, 111, 2415; "
                "doi:10.1021/jp068335b",
                "Kasavajhala, Lam, Simmerling, J. Chem. Inf. Model. 2020, 60, 1218; "
                "PMCID PMC7725893",
            ],
            "finite_reservoir_approximation": (
                "the reservoir holds a finite number of configurations and is NOT the top state's "
                "full equilibrium distribution; the assumption that the selected source window "
                "represents that distribution is an assumption, not a result"),
            "velocities": ("not read from the source: a DCD carries none. Momenta are redrawn "
                           "from the Maxwell distribution at the common temperature with a "
                           "recorded seed."),
        }

    @staticmethod
    def _validate_prepared(manifest, declaration, protocol):
        contract = manifest.get("reservoir") or {}
        if contract.get("weighting") != "boltzmann":
            raise ReservoirError(
                f"the prepared reservoir records weighting {contract.get('weighting')!r}; v1 "
                f"accepts only a Boltzmann-weighted reservoir")
        if abs(float(contract.get("tau", -1)) - float(protocol.tau[-1])) > 1e-9:
            raise ReservoirError(
                f"the prepared reservoir was made for tau = {contract.get('tau')} but the top rung "
                f"is at {protocol.tau[-1]}. Refusing rather than refreshing from a different "
                f"ensemble.")
        if abs(float(contract.get("temperature_k", -1)) - protocol.temperature_k) > 1e-9:
            raise ReservoirError(
                f"the prepared reservoir was made at {contract.get('temperature_k')} K but the "
                f"ladder is at {protocol.temperature_k} K.")

    @staticmethod
    def _validate_frames(frames, protocol, manifest):
        if not frames:
            raise ReservoirError("the prepared reservoir holds no configurations")
        expected_box = protocol.pressure_bar is None and manifest["configurations"]["has_box"]
        for index, (positions, box) in enumerate(frames):
            if not np.all(np.isfinite(positions)):
                raise ReservoirError(f"reservoir frame {index} has non-finite coordinates")
            if manifest["configurations"]["has_box"] and box is None:
                raise ReservoirError(
                    f"reservoir frame {index} carries no box although the manifest says it should; "
                    f"a configuration without its box is at an undefined density")

    # -- access ---------------------------------------------------------------------------------------

    @property
    def n_frames(self):
        return len(self._frames)

    def configuration(self, index):
        """`(positions_nm, box_nm or None)` for one prepared frame."""
        if not 0 <= int(index) < self.n_frames:
            raise ReservoirError(f"reservoir frame {index} is outside 0..{self.n_frames - 1}")
        positions, box = self._frames[int(index)]
        return positions, box

    def check_box_matches(self, box):
        """The reservoir's boxes must match the ladder's, or the density silently changes."""
        if box is None:
            return
        for index, (_, frame_box) in enumerate(self._frames):
            if frame_box is None:
                raise ReservoirError(
                    f"reservoir frame {index} has no box but the ladder is explicit-solvent")
            if not source_ensemble.same_lattice(frame_box, box):
                raise ReservoirError(
                    f"reservoir frame {index} has box\n{np.asarray(frame_box)}\nbut the ladder's "
                    f"box is\n{np.asarray(box)}\nand these are not the same lattice (checked as "
                    f"an integer change of basis, so an equivalent representation would pass). A "
                    f"different box is a different density; v1 requires a fixed-volume source at "
                    f"the ladder's own box.")

    def describe(self):
        contract = self.manifest.get("reservoir") or {}
        return {
            "prepared_directory": self.directory.name,
            "frames": self.n_frames,
            "weighting": contract.get("weighting"),
            "ensemble": contract.get("ensemble"),
            "tau": contract.get("tau"),
            "temperature_k": contract.get("temperature_k"),
            "acceptance_rule": contract.get("acceptance_rule"),
            "finite_reservoir_approximation": contract.get("finite_reservoir_approximation"),
            "citations": contract.get("citations"),
            "source": self.manifest.get("source"),
            "evidence": self.manifest.get("evidence"),
            "selection": self.manifest.get("selection"),
        }

    def identity(self):
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True,
                                         default=str).encode("utf-8")).hexdigest()
