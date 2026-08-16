"""The canonical typed simulation model.

Every supported input syntax -- YAML today, JSON today, an Amber-style front end later -- compiles
into the models here before a System is built or a run directory is opened. File format is a front
end; it never defines semantics. Two documents that mean the same thing produce the same canonical
JSON and therefore the same hashes, whatever they were written in.

Four concerns are kept separate, because they answer different questions and change at different
times:

    SystemSpec      which molecule, and by which declared route
    BuildSpec       how the System is parameterised and solvated
    ProtocolSpec    what physics is run: equilibration, integrator, and the method's production
    ExecutionSpec   machine choices -- platform, device, reporting, where output goes

They have independent schema versions and independent canonical projections:

    system/build hash        changing it requires a NEW BUNDLE
    protocol/continuity hash changing it requires a NEW RUN
    execution projection     may change performance or output volume, never the Hamiltonian

Unknown fields are rejected at every level. A configuration file that silently ignores a
misspelled key is worse than one that fails: the run proceeds under settings the author did not
choose and believes they did.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from .units import Quantity, parse_quantity

#: Independent schema versions. Bumping one must not force the others to move.
SYSTEM_SCHEMA_VERSION = 1
BUILD_SCHEMA_VERSION = 1
PROTOCOL_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 1


def _quantity(dimension: str):
    """A field that accepts '2 fs' and stores the canonical Quantity."""

    def _validate(v: Any) -> Quantity:
        return parse_quantity(v, dimension=dimension)

    return Annotated[Quantity, BeforeValidator(_validate)]


Time = _quantity("time")
Length = _quantity("length")
Temperature = _quantity("temperature")
Rate = _quantity("rate")
Mass = _quantity("mass")
Pressure = _quantity("pressure")


class Strict(BaseModel):
    """Base for every spec model: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True,
                              validate_assignment=True)


# ---------------------------------------------------------------------------------------------
# SystemSpec -- molecular identity and the DECLARED route
# ---------------------------------------------------------------------------------------------

class SystemSpec(Strict):
    """Which molecule this is, and how it enters the pipeline.

    The route is declared, never inferred from file contents. Inferring it is how a peptide becomes
    a small-molecule parameterisation under the same name.
    """

    schema_version: int = SYSTEM_SCHEMA_VERSION
    system_id: str = Field(pattern=r"^[a-z0-9_]+$")
    display_name: Optional[str] = None
    route: Literal["pdb", "smiles"]

    smiles: Optional[str] = None
    canonical_isomeric_smiles: Optional[str] = None
    canonical_smiles_sha256: Optional[str] = None
    pdb: Optional[str] = None                      # bundle-relative or input-relative, never absolute
    pdb_sha256: Optional[str] = None
    expected_formal_charge: int = 0

    @model_validator(mode="after")
    def _route_matches_input(self):
        if self.route == "smiles":
            if not self.smiles:
                raise ValueError("system.smiles is required for route 'smiles'")
            if self.pdb:
                raise ValueError("system.pdb is set but route is 'smiles'; declare one route")
        else:
            if not self.pdb:
                raise ValueError("system.pdb is required for route 'pdb'")
            if self.smiles:
                raise ValueError("system.smiles is set but route is 'pdb'; declare one route")
        return self


# ---------------------------------------------------------------------------------------------
# BuildSpec -- everything that decides the prepared System
# ---------------------------------------------------------------------------------------------

class ForceFieldSpec(Strict):
    small_molecule: Optional[str] = None
    charge_method: Optional[str] = None
    protein: Optional[str] = None
    water: str

    @model_validator(mode="after")
    def _one_route_of_parameters(self):
        if self.small_molecule and self.protein:
            raise ValueError(
                "forcefield.small_molecule and forcefield.protein are both set. A ligand route may "
                "not load a protein force field and a peptide route may not name a small-molecule "
                "one; that is how a peptide silently becomes a ligand calculation."
            )
        if not self.small_molecule and not self.protein:
            raise ValueError("forcefield: one of small_molecule or protein must be given")
        if self.small_molecule and not self.charge_method:
            raise ValueError("forcefield.charge_method is required with a small-molecule force field")
        return self


class SolvationSpec(Strict):
    water_model: str
    box_shape: Literal["cube", "dodecahedron", "octahedron"]
    padding: Length
    padding_semantics: Literal["solute-image-gap", "openmm"] = "solute-image-gap"
    ionic_strength_molar: float = Field(ge=0.0)
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    neutralize: bool = True
    cutoff_fit_policy: Literal["grow", "refuse"] = "grow"


class NonbondedSpec(Strict):
    method: Literal["PME", "LJPME"] = "PME"
    cutoff: Length
    switch_distance: Optional[Length] = None
    use_dispersion_correction: bool = True
    ewald_error_tolerance: float = Field(gt=0.0, lt=1.0)
    minimum_image_margin: Length

    @model_validator(mode="after")
    def _switch_below_cutoff(self):
        if self.switch_distance and self.switch_distance.value >= self.cutoff.value:
            raise ValueError(
                f"nonbonded.switch_distance ({self.switch_distance.source}) must be below "
                f"nonbonded.cutoff ({self.cutoff.source})"
            )
        return self


class BuildSpec(Strict):
    schema_version: int = BUILD_SCHEMA_VERSION
    forcefield: ForceFieldSpec
    solvation: SolvationSpec
    nonbonded: NonbondedSpec
    constraints: Literal["None", "HBonds", "AllBonds", "HAngles"] = "HBonds"
    rigid_water: bool = True
    hydrogen_mass: Mass
    hmr_scope: Literal["solute", "all"] = "solute"
    remove_cm_motion: bool = True


# ---------------------------------------------------------------------------------------------
# ProtocolSpec -- the physics that is run
# ---------------------------------------------------------------------------------------------

class IntegratorSpec(Strict):
    kind: Literal["langevin-middle", "leapfrog-langevin", "verlet"] = "langevin-middle"
    timestep: Time
    temperature: Temperature
    friction: Rate


class EquilibrationSpec(Strict):
    """Equilibration before production.

    `staged` is restrained minimisation, a heating ramp, and NPT with the restraint released in
    steps -- appropriate for a flexible solute dropped into a freshly packed box. `simple` is
    minimise/NVT/NPT with no restraints, for a small rigid solute. The fields each protocol reads
    differ, which is why the unused ones are optional rather than invented.
    """

    protocol: Literal["staged", "simple"] = "staged"
    minimize_max_iterations: int = Field(ge=0, default=0)
    npt_free: Time
    timestep: Optional[Time] = None                # equilibration may integrate more cautiously
    nvt: Optional[Time] = None                     # `simple` only
    npt: Optional[Time] = None                     # `simple` only
    box_average_last: Optional[Time] = None
    seed: Optional[int] = None


class ChunkPlan(Strict):
    """The plan is stated, never inferred from a total.

    `n_chunks` is the work THIS invocation adds; total duration is derived from it and reported.
    """

    n_chunks: int = Field(gt=0)
    chunk: Time

    @property
    def total(self) -> float:
        return self.n_chunks * self.chunk.value


class MDProduction(ChunkPlan):
    method: Literal["md"] = "md"
    scale_factor: float = Field(gt=0.0, le=1.0, default=1.0)
    seed: Optional[int] = None


class REST2Production(ChunkPlan):
    method: Literal["rest2"] = "rest2"
    scale_factors: list[float] = Field(min_length=2)
    exchange_interval: Time
    relaxation: Time
    omega_exclusion: bool = True
    proline_like_residues: list[str] = Field(default_factory=lambda: ["PRO"])
    max_proline_ring_size: int = Field(gt=2, default=7)
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _ladder_and_divisibility(self):
        s = self.scale_factors
        if abs(s[0] - 1.0) > 1e-12:
            raise ValueError(f"production.scale_factors must start at the cold rung 1.0, got {s[0]}")
        if any(b >= a for a, b in zip(s, s[1:])):
            raise ValueError(f"production.scale_factors must be strictly descending: {s}")
        if not all(0.0 < v <= 1.0 for v in s):
            raise ValueError(f"production.scale_factors must lie in (0, 1]: {s}")
        per_chunk = self.chunk.value / self.exchange_interval.value
        if abs(per_chunk - round(per_chunk)) > 1e-9:
            raise ValueError(
                f"production.chunk ({self.chunk.source}) is not a whole number of exchange "
                f"intervals ({self.exchange_interval.source}): {per_chunk:.6f}. A chunk boundary "
                "falling mid-interval would drop or duplicate an attempt across a resume."
            )
        return self


Production = Annotated[Union[MDProduction, REST2Production], Field(discriminator="method")]


class ProtocolSpec(Strict):
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    integrator: IntegratorSpec
    equilibration: EquilibrationSpec
    production: Production

    @model_validator(mode="after")
    def _chunk_is_whole_steps(self):
        dt = self.integrator.timestep.value
        steps = self.production.chunk.value / dt
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(
                f"production.chunk ({self.production.chunk.source}) is not a whole number of "
                f"{self.integrator.timestep.source} steps ({steps:.6f}). A rounded chunk runs a "
                "different length than it declares."
            )
        return self


# ---------------------------------------------------------------------------------------------
# ExecutionSpec -- machine choices only
# ---------------------------------------------------------------------------------------------

class ReportingSpec(Strict):
    all_atom: Time
    solute: Time
    state: Time
    checkpoint: Time


class ExecutionSpec(Strict):
    schema_version: int = EXECUTION_SCHEMA_VERSION
    platform: Literal["CPU", "CUDA", "OpenCL", "Reference"] = "CPU"
    device: Optional[str] = None
    precision: Literal["single", "mixed", "double"] = "mixed"
    reporting: ReportingSpec

    @model_validator(mode="after")
    def _device_only_for_accelerators(self):
        if self.device is not None and self.platform in ("CPU", "Reference"):
            raise ValueError(
                f"execution.device is set but execution.platform is {self.platform!r}; a device "
                "index is meaningless there"
            )
        return self


# ---------------------------------------------------------------------------------------------
# the whole document
# ---------------------------------------------------------------------------------------------

class SimulationSpec(Strict):
    """One canonical configuration. Every front end resolves to this."""

    system: SystemSpec
    build: BuildSpec
    protocol: ProtocolSpec
    execution: ExecutionSpec

    @model_validator(mode="after")
    def _route_and_parameters_agree(self):
        ff = self.build.forcefield
        if self.system.route == "smiles" and ff.protein:
            raise ValueError(
                "system.route is 'smiles' (a ligand) but build.forcefield.protein is set"
            )
        if self.system.route == "pdb" and ff.small_molecule:
            raise ValueError(
                "system.route is 'pdb' (a peptide) but build.forcefield.small_molecule is set"
            )
        return self

    @property
    def method(self) -> str:
        return self.protocol.production.method
