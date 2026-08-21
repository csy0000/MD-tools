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

from pydantic import (BaseModel, BeforeValidator, ConfigDict, Field, field_validator,
                      model_validator)

from .units import Quantity, parse_quantity

#: Independent schema versions. Bumping one must not force the others to move.
SYSTEM_SCHEMA_VERSION = 1
BUILD_SCHEMA_VERSION = 1
PROTOCOL_SCHEMA_VERSION = 5

#: Versions this build recognises but will not accept. Named so a refusal can say "retired" and
#: point at the migration, rather than "unknown", which would send a reader hunting for a typo.
RETIRED_PROTOCOL_SCHEMA_VERSIONS = (1, 2, 3, 4)
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
    #: Null under implicit solvent. A water force field there would name parameters for molecules
    #: the System does not contain.
    water: Optional[str] = None

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


class ImplicitSpec(Strict):
    """Generalised-Born solvent. Present only when there is no water.

    `GBn2` with `mbondi3` is the default and the pairing is not arbitrary: GBn2 was parameterised
    against mbondi3, so another radius set is a different Hamiltonian that still runs.
    """

    #: Only the validated pair is publicly accepted. OpenMM exposes other GB models and ParmEd
    #: accepts other radius sets, but the energy identity in this repository is pinned for
    #: GBn2/mbondi3 and nothing else, and public acceptance follows the evidence rather than the
    #: library's capability. `implicit.py` stays extensible so widening this means adding evidence.
    model: Literal["GBn2"] = "GBn2"
    radii: Literal["mbondi3"] = "mbondi3"


class NonbondedSpec(Strict):
    """Nonbonded treatment. `NoCutoff` is the implicit-solvent case and takes none of the rest.

    Under `NoCutoff` there is no box, so a cutoff, an Ewald tolerance and a minimum-image margin
    describe quantities that do not exist -- they must be null rather than carrying a plausible
    number nothing reads.
    """

    method: Literal["PME", "LJPME", "NoCutoff"] = "PME"
    cutoff: Optional[Length] = None
    switch_distance: Optional[Length] = None
    use_dispersion_correction: bool = True
    ewald_error_tolerance: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    minimum_image_margin: Optional[Length] = None

    @model_validator(mode="after")
    def _periodic_fields_match_the_method(self):
        if self.method == "NoCutoff":
            stated = [name for name in ("cutoff", "ewald_error_tolerance", "minimum_image_margin")
                      if getattr(self, name) is not None]
            if stated:
                raise ValueError(
                    f"nonbonded.method is 'NoCutoff' but {', '.join(stated)} "
                    f"{'are' if len(stated) > 1 else 'is'} stated. Without a periodic box these "
                    "describe quantities that do not exist; set them to null.")
        else:
            missing = [name for name in ("cutoff", "ewald_error_tolerance", "minimum_image_margin")
                       if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"nonbonded.method is {self.method!r} and requires {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _switch_below_cutoff(self):
        if self.switch_distance and self.switch_distance.value >= self.cutoff.value:
            raise ValueError(
                f"nonbonded.switch_distance ({self.switch_distance.source}) must be below "
                f"nonbonded.cutoff ({self.cutoff.source})"
            )
        return self


class BuildSpec(Strict):
    """How the System is built. Exactly one solvent treatment, never both and never neither."""

    schema_version: int = BUILD_SCHEMA_VERSION
    forcefield: ForceFieldSpec
    solvation: Optional[SolvationSpec] = None
    implicit: Optional[ImplicitSpec] = None
    nonbonded: NonbondedSpec
    constraints: Literal["None", "HBonds", "AllBonds", "HAngles"] = "HBonds"
    rigid_water: bool = True
    #: Null means no hydrogen mass repartitioning. Implicit builds default to that deliberately:
    #: the reference GBn2 System is not repartitioned, and comparing against a repartitioned one
    #: would be comparing a different System that still passes every structural check.
    hydrogen_mass: Optional[Mass] = None
    hmr_scope: Literal["solute", "all", "none"] = "solute"
    remove_cm_motion: bool = True

    @model_validator(mode="after")
    def _exactly_one_solvent_treatment(self):
        if self.solvation is not None and self.implicit is not None:
            raise ValueError(
                "build states both `solvation` (explicit water) and `implicit`. A System has one "
                "solvent treatment; stating both leaves it undefined which one was used.")
        if self.solvation is None and self.implicit is None:
            raise ValueError(
                "build states neither `solvation` nor `implicit`. Say which solvent treatment this "
                "System uses -- it is never inferred.")
        if self.implicit is not None:
            if self.nonbonded.method != "NoCutoff":
                raise ValueError(
                    f"implicit solvent requires nonbonded.method 'NoCutoff', got "
                    f"{self.nonbonded.method!r}: there is no periodic box to run PME in.")
            if self.rigid_water:
                raise ValueError("implicit solvent has no water, so rigid_water must be false")
            if self.hydrogen_mass is not None and self.hmr_scope == "none":
                raise ValueError(
                    "build.hydrogen_mass is stated while hmr_scope is 'none'; set a scope or "
                    "remove the mass")
        return self


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
    #: Optional because an implicit-solvent protocol has no NPT stage at all. A stated value is
    #: refused there rather than ignored -- see SimulationSpec.
    npt_free: Optional[Time] = None
    timestep: Optional[Time] = None                # equilibration may integrate more cautiously
    nvt: Optional[Time] = None                     # `simple` only
    npt: Optional[Time] = None                     # `simple` only
    box_average_last: Optional[Time] = None
    seed: Optional[int] = None


class SegmentedProduction(Strict):
    """Production states the length of ONE segment. It does not state how many.

    Segment count is execution state, not science. A user who wants to run longer is not
    performing a different calculation, so requesting more segments must not move the
    configuration hash. The driver script decides how many segments to request; the run manifest
    records how many actually committed.
    """

    duration_per_segment: Time


class MDProduction(SegmentedProduction):
    method: Literal["md"] = "md"
    scale_factor: float = Field(gt=0.0, le=1.0, default=1.0)
    seed: Optional[int] = None


class TauLadderSpec(Strict):
    """The REST2 ladder in the Amber-style ``tau`` parameterisation.

    ``tau`` is the source parameter and the thing persisted. ``s`` and ``sqrt(s)`` are derived by
    one shared function (:mod:`md_templates.openmm.tau`) and recorded as labelled diagnostics,
    never accepted back as input -- two ways to say the same thing is how a ladder drifts.
    """

    minimum: float = Field(ge=0.0, lt=1.0, default=0.0)
    maximum: float = Field(gt=0.0, lt=1.0)
    count: int = Field(ge=2)
    interpolation: Literal["linear"] = "linear"

    @model_validator(mode="after")
    def _span_and_cold_rung(self):
        if self.maximum <= self.minimum:
            raise ValueError(
                f"production.tau_ladder.maximum ({self.maximum}) must exceed minimum "
                f"({self.minimum}); a ladder with no span is not a ladder"
            )
        if self.minimum != 0.0:
            raise ValueError(
                f"production.tau_ladder.minimum must be 0.0 so the cold rung is the unscaled, "
                f"physical Hamiltonian (s = 1); got {self.minimum}. A ladder that never samples "
                "the physical ensemble has no replica whose trajectory is the answer."
            )
        return self

    def tau_values(self) -> list[float]:
        from ..tau import build_tau_ladder
        return build_tau_ladder(self.minimum, self.maximum, self.count, self.interpolation)

    def scale_factors(self) -> list[float]:
        """The ``s`` values this ladder resolves to -- the bridge to the System builder."""
        from ..tau import scale_factors_for_ladder
        return scale_factors_for_ladder(self.tau_values())


class ExchangeSpec(Strict):
    """How many exchange attempts a segment contains.

    Only the count. The interval is DERIVED from it and the segment duration, in integer step
    space::

        steps_per_segment  = production.duration_per_segment / integrator.timestep
        steps_per_exchange = steps_per_segment / number_of_exchanges_per_segment
        exchange_interval  = steps_per_exchange * integrator.timestep

    Both divisions must be exact and are refused otherwise, never rounded: an exchange interval off
    by a step drifts the schedule out of alignment with the committed watermark while the run still
    looks healthy.

    A segment duration and an exchange count are the two numbers a reader actually chooses -- how
    long to run and how often to attempt a swap. The interval is a consequence, so it is derived
    rather than stated, and cannot disagree with them.
    """

    number_of_exchanges_per_segment: int = Field(ge=1)


class SelectionSpec(Strict):
    """Which atoms an operation applies to.

    OpenMM consumes resolved zero-based indices. A mask expression is a front end that is resolved
    to indices before the System is built; both the original expression and the resolved indices
    are persisted, so a reader can see what was asked and what it turned out to mean.
    """

    type: Literal["solute", "atom_indices", "amber_mask"] = "solute"
    atom_indices: Optional[list[int]] = None
    amber_mask: Optional[str] = None

    @model_validator(mode="after")
    def _one_selection_language(self):
        if self.type == "atom_indices":
            if not self.atom_indices:
                raise ValueError("selection.atom_indices is required when type is 'atom_indices'")
            negative = [index for index in self.atom_indices if index < 0]
            if negative:
                raise ValueError(
                    f"selection.atom_indices must be zero-based and non-negative; got {negative}"
                )
            if len(set(self.atom_indices)) != len(self.atom_indices):
                raise ValueError("selection.atom_indices contains duplicate indices")
        elif self.type == "amber_mask":
            if not self.amber_mask:
                raise ValueError("selection.amber_mask is required when type is 'amber_mask'")
        else:
            if self.atom_indices or self.amber_mask:
                raise ValueError(
                    "selection.type is 'solute' but an explicit selection was also given; declare "
                    "exactly one selection language"
                )
        return self


class OmegaExclusionSpec(Strict):
    """Peptide omega torsions are left unscaled by default.

    Softening omega lets the backbone sample cis-amide states that are an artefact of the scaling
    rather than physics, so exclusion is on unless a user turns it off deliberately.
    """

    enabled: bool = True
    definition: Literal["peptide_omega"] = "peptide_omega"
    proline_like_residues: list[str] = Field(default_factory=lambda: ["PRO"])
    max_proline_ring_size: int = Field(gt=2, default=7)


class REST2Production(Strict):
    """REST2 production.

    `duration_per_segment` is the length of ONE segment, exactly as for conventional MD. How many
    segments to run is an execution choice made by the driver script and never appears here: a
    segment count in the scientific input would make a longer run look like a different
    calculation, because it would move the configuration hash.
    """

    method: Literal["rest2"] = "rest2"
    duration_per_segment: Time
    enhanced_region: SelectionSpec = Field(default_factory=SelectionSpec)
    tau_ladder: TauLadderSpec
    exchange: ExchangeSpec
    omega_exclusion: OmegaExclusionSpec = Field(default_factory=OmegaExclusionSpec)
    relaxation: Optional[Time] = None
    seed: Optional[int] = None

    @property
    def n_replicas(self) -> int:
        return self.tau_ladder.count

    def scale_factors(self) -> list[float]:
        return self.tau_ladder.scale_factors()


Production = Annotated[Union[MDProduction, REST2Production], Field(discriminator="method")]


class ProtocolSpec(Strict):
    #: Validated, not merely carried. An unconstrained integer let a document label itself
    #: version 4 while using version-5 exchange fields, so the label said one thing and the
    #: semantics said another -- and nothing checked. A version this code cannot interpret must be
    #: refused, and a retired version must arrive with its migration message.
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _supported_protocol_schema(cls, value: int) -> int:
        if value == PROTOCOL_SCHEMA_VERSION:
            return value
        if value in RETIRED_PROTOCOL_SCHEMA_VERSIONS:
            raise ValueError(
                f"protocol.schema_version {value} is retired. Version "
                f"{PROTOCOL_SCHEMA_VERSION} states a segment as `duration_per_segment` plus "
                "`exchange.number_of_exchanges_per_segment` and derives the interval.\n"
                "  A version-4 document using the version-5 exchange fields is not a version-4 "
                "document; relabel it 5 once\n  it uses them, or migrate it with "
                "`md-openmm config migrate`."
            )
        raise ValueError(
            f"protocol.schema_version {value} is not a version this build understands; "
            f"supported: {PROTOCOL_SCHEMA_VERSION}")
    integrator: IntegratorSpec
    equilibration: EquilibrationSpec
    production: Production

    @model_validator(mode="after")
    def _segment_resolves_to_exact_whole_steps(self):
        """Durations must be whole steps, and a REST2 segment whole exchange rounds.

        Both are delegated to `md_templates.openmm.segments`, which refuses rather than rounds, so
        the CLI, the runner and the tests share one definition of "exact".
        """
        from ..segments import plan_segment, plan_segment_from_duration_and_exchanges

        production = self.production
        if isinstance(production, REST2Production):
            # Two divisions, both in integer step space, both refused rather than rounded.
            plan_segment_from_duration_and_exchanges(
                production.duration_per_segment.value,
                production.exchange.number_of_exchanges_per_segment,
                self.integrator.timestep.value,
                duration_source=production.duration_per_segment.source,
                timestep_source=self.integrator.timestep.source,
            )
        else:
            plan_segment(
                production.duration_per_segment.value,
                self.integrator.timestep.value,
                duration_source=production.duration_per_segment.source,
                timestep_source=self.integrator.timestep.source,
            )
        return self
        return self


# ---------------------------------------------------------------------------------------------
# RandomnessSpec -- seeds, explicitly
# ---------------------------------------------------------------------------------------------

#: Stage order defines the derivation offsets and MUST NOT be reordered: a stage seed is
#: `master_seed + index`, and this is the rule migrated inputs reproduce.
STAGE_ORDER = ("structure", "equilibration", "md", "rest2")

RANDOMNESS_SCHEMA_VERSION = 1


class StageSeeds(Strict):
    """Per-stage seeds. `None` means "derive from the master seed"."""

    structure: Optional[int] = None
    equilibration: Optional[int] = None
    md: Optional[int] = None
    rest2: Optional[int] = None


class RandomnessSpec(Strict):
    """The master seed and the stage seeds derived from it.

    `master_seed` is a LABEL until it is derived: what changes a trajectory is the resolved stage
    seed, so the projections hash the resolved stage seeds rather than the master. Pinning every
    affected stage seed to its old value therefore leaves the physical hashes unchanged even when
    the master seed differs -- which is the honest behaviour, and `config diff` explains it.
    """

    schema_version: int = RANDOMNESS_SCHEMA_VERSION
    master_seed: int = 20260814
    stage_seeds: StageSeeds = Field(default_factory=StageSeeds)

    def resolve(self) -> dict[str, int]:
        """Stage seeds after derivation: `master_seed + index`, explicit values preserved.

        This is the legacy rule, unchanged, so a migrated input reproduces its existing
        trajectories rather than merely a valid one.
        """
        out: dict[str, int] = {}
        for offset, stage in enumerate(STAGE_ORDER):
            explicit = getattr(self.stage_seeds, stage)
            out[stage] = int(explicit) if explicit is not None else self.master_seed + offset
        return out

    def sources(self) -> dict[str, str]:
        return {stage: ("explicit" if getattr(self.stage_seeds, stage) is not None else "derived")
                for stage in STAGE_ORDER}


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
    randomness: RandomnessSpec = Field(default_factory=RandomnessSpec)

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

    @model_validator(mode="after")
    def _implicit_solvent_has_no_pressure(self):
        """Refuse NPT and pressure under implicit solvent, before anything is generated.

        Implicit solvent has no box, so there is no volume to control and pressure is undefined. An
        NPT equilibration stage or a stated pressure is not a harmless extra here -- it is a request
        for an ensemble that cannot exist, and ignoring it would generate a protocol that silently
        differs from the one the file describes.
        """
        if self.build.implicit is None:
            return self

        offenders = []
        equilibration = self.protocol.equilibration
        for field in ("npt", "npt_free", "box_average_last"):
            if getattr(equilibration, field, None) is not None:
                offenders.append(f"protocol.equilibration.{field}")
        for field in ("pressure", "barostat", "barostat_interval"):
            if getattr(self.protocol.production, field, None) is not None:
                offenders.append(f"protocol.production.{field}")
        if getattr(self.build, "solvation", None) is not None:
            offenders.append("build.solvation")
        if offenders:
            raise ValueError(
                "this is an implicit-solvent build, which has no box and therefore no pressure, "
                f"but the configuration states {', '.join(offenders)}.\n"
                "  The implicit stage graph is min -> eq_nvt -> cMD_1 -> REST2_1; there is no NPT "
                "stage and there cannot be one.\n"
                "  Remove these fields, or use an explicit-water build."
            )
        return self

    @property
    def method(self) -> str:
        return self.protocol.production.method
