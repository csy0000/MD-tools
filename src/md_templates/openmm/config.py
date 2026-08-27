"""Read the two YAML files a user edits, and resolve them into what will actually be built.

Deliberately thin. There is no schema-migration layer, no profile inheritance and no canonical
intermediate model: the YAML is the configuration, and resolving it means dropping the solvent
block that does not apply and checking the handful of combinations that are physically wrong.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .defaults import canonical_method, canonical_solvent, is_implicit

__all__ = ["load_yaml", "write_yaml", "resolve_sys_config", "resolve_md_config",
           "sha256_of_document", "ConfigError"]


class ConfigError(ValueError):
    """A configuration that cannot be built, with the reason stated."""


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    return document


def write_yaml(path: Path, document: dict[str, Any], *, header: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=88)
    path.write_text((header + text) if header else text, encoding="utf-8")
    return path


def sha256_of_document(document: dict[str, Any]) -> str:
    """A stable hash of a configuration, for provenance."""
    canonical = yaml.safe_dump(document, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_sys_config(document: dict[str, Any]) -> dict[str, Any]:
    """The effective system settings: the irrelevant solvent block removed, and checks applied."""
    resolved = yaml.safe_load(yaml.safe_dump(document))       # deep copy via the same serialiser
    solvent_block = resolved.get("solvent") or {}
    implicit_block = resolved.get("implicit_solvent") or {}

    model = solvent_block.get("model")
    implicit_model = implicit_block.get("model")
    # Which treatment applies is decided by which block the user left in place. Both present is the
    # default file's shape, in which case `solvent.model` decides.
    if model and implicit_model:
        chosen = canonical_solvent(model)
    elif implicit_model:
        chosen = canonical_solvent(implicit_model)
    elif model:
        chosen = canonical_solvent(model)
    else:
        raise ConfigError("neither solvent.model nor implicit_solvent.model is set")

    resolved["solvation"] = "implicit" if is_implicit(chosen) else "explicit"
    if resolved["solvation"] == "implicit":
        resolved.pop("solvent", None)
        resolved["forcefield"] = dict(resolved.get("forcefield") or {})
        # No water model participates in an implicit build, and recording one would name a force
        # field that never loaded.
        resolved["forcefield"]["water"] = None
        constraints = resolved.setdefault("constraints", {})
        constraints["rigid_water"] = False
    else:
        resolved.pop("implicit_solvent", None)

    # The ligand force field is recorded as the resource that will actually be loaded, decided
    # here rather than left as a label for a reader to map later. `sage-2.2.1` is what a user
    # writes; `openff-2.2.1` is what the toolkit resolves, and preflight compares the record
    # against this without needing a second copy of the mapping.
    solute = resolved.get("solute") or {}
    if not bool(solute.get("peptide", True)):
        resolved["forcefield"] = dict(resolved.get("forcefield") or {})
        resolved["forcefield"]["ligand"] = openff_resource(solute.get("ligand_forcefield"))
        resolved["forcefield"]["ligand_charge_method"] = solute.get("ligand_charge_method")
        # No protein force field participates in a ligand build -- the ligand route loads only
        # the water XML and the SMIRNOFF template generator -- and recording one would name a
        # force field that never loaded. Same rule as water under implicit solvent.
        resolved["forcefield"]["protein"] = None

    _check_constraints(resolved)
    _check_protein_solvation_pairing(resolved)
    _check_explicit_pairing(resolved)
    return resolved


def openff_resource(name):
    """`sage-2.2.1` is what a user writes; `openff-2.2.1` is what the toolkit loads.

    The installed `openforcefields` package ships the file as `openff-2.2.1.offxml`, and
    `SMIRNOFFTemplateGenerator` resolves the name with or without the suffix. A name that does not
    resolve raises there, at the point the parameters would have been assigned.
    """
    if not name:
        return None
    text = str(name).strip().lower()
    if text.startswith("sage-"):
        return "openff-" + text[len("sage-"):]
    return text


def _check_explicit_pairing(resolved: dict[str, Any]) -> None:
    """Refuse a hand-edited explicit configuration that crosses the two supported pairs.

    `sys-config` writes a coupled selection -- ff14SB with TIP3P, ff19SB with OPC -- but the file it
    writes is ordinary editable YAML, so generating it correctly is not the same as building it
    correctly. Changing `solvent.model` to OPC and leaving `forcefield.protein` at ff14SB produces
    a System, runs to completion, and reports a Hamiltonian nobody validated.

    The two pairs are not interchangeable halves. ff14SB's backbone adjustment is an empirical
    correction fit in TIP3P and its authors caution that transferring it to another solvent model
    needs evaluation; ff19SB's amino-acid-specific CMAPs were trained for a better water model and
    its authors recommend OPC. Crossing them discards the reason either pair works. See
    `docs/md-defaults-scientific-rationale.md` section 3.

    Both halves are checked, in both directions, because either one alone can be the edited field.
    """
    from .defaults import EXPLICIT_COMBINATIONS, PROTEIN_FAMILY_MARKERS, WATER_FAMILY_MARKERS

    if resolved.get("solvation") != "explicit":
        return
    forcefield = resolved.get("forcefield") or {}
    solvent = resolved.get("solvent") or {}
    model = canonical_solvent(solvent.get("model"))
    expected = EXPLICIT_COMBINATIONS[model]
    protein = str(forcefield.get("protein") or "").strip()
    water = str(forcefield.get("water") or "").strip()

    def family(value: str, markers: dict[str, tuple[str, ...]]) -> str:
        """Which supported family a resource name belongs to, or "" if it is neither."""
        lowered = value.lower()
        for name, tokens in markers.items():
            if any(token in lowered for token in tokens):
                return name
        return ""

    wrong = []
    if protein and family(protein, PROTEIN_FAMILY_MARKERS) not in ("", model):
        wrong.append(("forcefield.protein", protein, expected["protein"]))
    if water and family(water, WATER_FAMILY_MARKERS) not in ("", model):
        wrong.append(("forcefield.water", water, expected["water"]))
    if not wrong:
        return

    named = "\n".join(f"      {field}: {value}   -> should be {want}" for field, value, want in wrong)
    raise ConfigError(
        f"solvent.model = {model!r} does not match the force field this configuration names.\n"
        f"{named}\n"
        f"  The protein force field and the water model are ONE selection, not two independent\n"
        f"  keys. {model} is supported only as:\n"
        f"      forcefield.protein: {expected['protein']}\n"
        f"      forcefield.water:   {expected['water']}\n"
        f"      solvent.model:      {model}\n"
        f"  and the other supported explicit selection is\n"
        + "".join(f"      {other}: {values['protein']} + {values['water']}\n"
                  for other, values in EXPLICIT_COMBINATIONS.items() if other != model)
        + "  Crossing them combines a backbone with a water model it was not corrected for; both\n"
          "  run and neither is a combination anyone has validated. See\n"
          "  docs/md-defaults-scientific-rationale.md section 3.\n"
          "  Regenerate with `md-openmm sys-config --solvent "
        + f"{model}` if you did not mean to change it.")


def _check_protein_solvation_pairing(resolved: dict[str, Any]) -> None:
    """Refuse a protein force field that was not parameterised for this solvation model.

    ff19SB's amino-acid-specific CMAPs were fit in explicit OPC water, and no GB model has been
    reparameterised against them; GBn2 was developed and validated in the ff99SB/ff14SB lineage.
    Running the pair produces numbers, which is exactly the problem -- nothing fails, and the
    result silently describes a Hamiltonian nobody validated.

    This is refused rather than warned about because a warning in a log is not read by whoever
    reads the trajectory a year later.
    """
    from .defaults import GB_INCOMPATIBLE_PROTEIN, IMPLICIT_PROTEIN_FORCEFIELD

    if resolved.get("solvation") != "implicit":
        return
    protein = str((resolved.get("forcefield") or {}).get("protein") or "")
    model = (resolved.get("implicit_solvent") or {}).get("model")
    if not protein:
        return
    if any(marker.lower() in protein.lower() for marker in GB_INCOMPATIBLE_PROTEIN):
        raise ConfigError(
            f"forcefield.protein = {protein!r} is not parameterised for implicit_solvent.model = "
            f"{model!r}.\n"
            f"  ff19SB's amino-acid-specific CMAP corrections were fit in explicit OPC water, and "
            f"no GB model has been reparameterised against them. GBn2 was developed and validated "
            f"with the ff99SB/ff14SB lineage, so this pair mixes a backbone trained in explicit "
            f"solvent with a solvation model tuned for a different one.\n"
            f"  Use the matched pair:\n"
            f"      forcefield.protein: {IMPLICIT_PROTEIN_FORCEFIELD}\n"
            f"  or switch to explicit solvent, where ff19SB belongs.")


def _check_constraints(resolved: dict[str, Any]) -> None:
    constraints = resolved.get("constraints") or {}
    mass = constraints.get("hydrogen_mass_amu")
    if mass is not None and float(mass) < 1.008:
        raise ConfigError(
            f"constraints.hydrogen_mass_amu is {mass}, lighter than a hydrogen. Repartitioning "
            "moves mass INTO hydrogens from the heavy atoms they are bonded to.")
    if constraints.get("type") not in ("HBonds", "AllBonds", "HAngles", None):
        raise ConfigError(f"constraints.type {constraints.get('type')!r} is not supported")


def resolve_md_config(document: dict[str, Any], *, implicit: bool) -> dict[str, Any]:
    """The effective protocol: ensembles reconciled with the solvent, and timestep checked."""
    resolved = yaml.safe_load(yaml.safe_dump(document))
    methods = [canonical_method(m) for m in (resolved.get("methods") or [])]
    if not methods:
        raise ConfigError("md config lists no methods")
    resolved["methods"] = methods

    common = resolved.setdefault("common", {})
    if implicit:
        # A non-periodic system has no volume to control. Saying NPT here would name an ensemble
        # the run cannot sample.
        common["pressure_bar"] = None
        # There is no barostat in an implicit System at all, so an attempt interval is not a
        # setting that was left out -- it does not apply. Written explicitly as null.
        common["barostat_frequency_steps"] = None
        for method in methods:
            block = resolved.get(method) or {}
            if str(block.get("ensemble", "")).upper() == "NPT":
                block["ensemble"] = "NVT"
            resolved[method] = block
    else:
        if common.get("pressure_bar") is None:
            raise ConfigError(
                "explicit solvent needs common.pressure_bar for the barostat; it is null. "
                "Set it (1.0 bar is the default) or switch the system config to GBn2.")
        common["barostat_frequency_steps"] = _check_barostat_frequency(common)

    _check_timestep(resolved)
    _check_tau(resolved)
    _check_ais(resolved)
    return resolved


def _required(block: dict[str, Any], key: str, *, where: str, meaning: str) -> Any:
    """A field the user must fill in. `null` is the shipped placeholder, not a usable value."""
    value = block.get(key)
    if value is None:
        raise ConfigError(
            f"{where}.{key} is null and has no default. {meaning}\n"
            f"  Set it in md.config.yaml and run `md-openmm md-gen` again.")
    return value


def _check_ais(resolved: dict[str, Any]) -> None:
    """Validate the AIS path, its source contract and its observation schedule.

    Everything here is refused before a project is written rather than inside a generated script,
    because an AIS run starts from somebody else's trajectory and the mistakes worth catching --
    a time window with no eligible frames, a duration whose observation points do not divide
    evenly -- are all visible in the configuration.
    """
    from . import ais as A

    if "AIS" not in (resolved.get("methods") or []):
        return
    block = resolved.get("AIS")
    if not isinstance(block, dict):
        raise ConfigError("md.config.yaml lists AIS as a method but carries no AIS block. "
                          "Run `md-openmm show-default AIS` for the shape it needs.")
    path = block.get("path") or {}
    source = block.get("source") or {}
    output = block.get("output") or {}
    execution = block.get("execution") or {}
    common = resolved.get("common") or {}

    # --- the path ---------------------------------------------------------------------------
    if str(path.get("type")) != A.PATH_TYPE:
        raise ConfigError(
            f"AIS.path.type must be {A.PATH_TYPE!r}; got {path.get('type')!r}. It is the only "
            f"path this implementation switches along: the REST2 Hamiltonian in tau.")
    if str(path.get("interpolation")) != A.INTERPOLATION:
        raise ConfigError(
            f"AIS.path.interpolation must be {A.INTERPOLATION!r} in this implementation; got "
            f"{path.get('interpolation')!r}. A linear path in tau makes sqrt(s) linear and s "
            f"quadratic; any other schedule would need its own work convention and is not written.")
    if str(path.get("enhanced_region")) != A.ENHANCED_REGION:
        raise ConfigError(
            f"AIS.path.enhanced_region must be {A.ENHANCED_REGION!r}; got "
            f"{path.get('enhanced_region')!r}. It is the same region cMD and REST2 scale, read "
            f"from inputs/solute.yaml.")
    for key in ("tau_start", "tau_end"):
        value = path.get(key)
        try:
            tau = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"AIS.path.{key} must be a number in [0, 1); got {value!r}") from None
        if not 0.0 <= tau < 1.0:
            raise ConfigError(
                f"AIS.path.{key} must be in [0, 1); got {tau}. tau = 0 is the physical "
                f"Hamiltonian and tau -> 1 removes the solute Hamiltonian entirely.")
        path[key] = tau
    if path["tau_start"] == path["tau_end"]:
        raise ConfigError(
            f"AIS.path.tau_start and AIS.path.tau_end are both {path['tau_start']}, so the "
            f"Hamiltonian never changes and every work value would be zero. A forward path needs "
            f"distinct endpoints; the default is 0.5 -> 0.0.")
    if not isinstance(path.get("omega_exclusion"), bool):
        raise ConfigError("AIS.path.omega_exclusion must be true or false")

    duration = _required(path, "switching_duration_ps", where="AIS.path",
                         meaning="It is how long the Hamiltonian takes to switch, in picoseconds. "
                                 "Nonequilibrium work depends on it, so there is no default that "
                                 "is not a scientific choice.")
    if float(duration) <= 0:
        raise ConfigError(f"AIS.path.switching_duration_ps must be positive; got {duration}")
    interval = path.get("parameter_update_interval_steps")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ConfigError(
            f"AIS.path.parameter_update_interval_steps must be a positive whole number of "
            f"integration steps; got {interval!r}.")

    # --- the observation schedule -------------------------------------------------------------
    observations = output.get("number_of_observations")
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 2:
        raise ConfigError(
            f"AIS.output.number_of_observations must be a whole number >= 2; got "
            f"{observations!r}. The default is {A.DEFAULT_OBSERVATIONS}: 20 equal intervals plus "
            f"the starting configuration.")
    if not (output.get("include_start") and output.get("include_end")):
        raise ConfigError(
            "AIS.output.include_start and include_end must both be true in this implementation. "
            "The first observation is the source configuration at tau_start with zero work, and "
            "the last is tau_end with the total work; dropping either breaks the pairing between "
            "coordinate frames and cumulative work rows.")
    if str(output.get("coordinates")) != A.COORDINATE_SCOPE:
        raise ConfigError(
            f"AIS.output.coordinates must be {A.COORDINATE_SCOPE!r} in this implementation; got "
            f"{output.get('coordinates')!r}.")
    try:
        # Raises with a message that names the divisibility that failed and what would fix it.
        A.switching_schedule(
            tau_start=path["tau_start"], tau_end=path["tau_end"],
            switching_duration_ps=float(duration),
            parameter_update_interval_steps=int(interval),
            number_of_observations=int(observations),
            timestep_fs=float(common.get("timestep_fs", 2.0)))
    except ValueError as error:
        raise ConfigError(str(error)) from None

    # --- the source contract ------------------------------------------------------------------
    _required(source, "trajectory", where="AIS.source",
              meaning="It is the equilibrium trajectory the switching paths start from -- a "
                      "fixed-tau cMD run or one REST2 rung, at tau = "
                      f"{path['tau_start']}. Paths are resolved relative to the generated MD/ "
                      "project.")
    declared_tau = source.get("source_tau")
    if declared_tau is not None:
        try:
            declared_tau = float(declared_tau)
        except (TypeError, ValueError):
            raise ConfigError(
                f"AIS.source.source_tau must be a number in [0, 1) or null; got "
                f"{source.get('source_tau')!r}") from None
        if not 0.0 <= declared_tau < 1.0:
            raise ConfigError(
                f"AIS.source.source_tau must be in [0, 1); got {declared_tau}")
        if abs(declared_tau - path["tau_start"]) > 1e-9:
            raise ConfigError(
                f"AIS.source.source_tau is {declared_tau} but AIS.path.tau_start is "
                f"{path['tau_start']}. The path must begin in the ensemble it anneals away from, "
                f"so the source ensemble's tau and the path's start are the same number.")
        source["source_tau"] = declared_tau

    if str(source.get("selection")) != A.SELECTION:
        raise ConfigError(
            f"AIS.source.selection must be {A.SELECTION!r} in this implementation; got "
            f"{source.get('selection')!r}.")
    if not isinstance(source.get("allow_sampling_with_replacement"), bool):
        raise ConfigError("AIS.source.allow_sampling_with_replacement must be true or false")

    count = _required(source, "number_of_trajectories", where="AIS.source",
                      meaning="It is how many independent switching paths to run. Each gets its "
                              "own directory, its own DCD and its own seeds.")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ConfigError(
            f"AIS.source.number_of_trajectories must be a positive whole number; got {count!r}")

    start = _required(source, "start_time_ps", where="AIS.source",
                      meaning="It is the INCLUSIVE start of the source time window, in "
                              "picoseconds of the source trajectory's own clock. Frames before it "
                              "are not eligible.")
    end = _required(source, "end_time_ps", where="AIS.source",
                    meaning="It is the INCLUSIVE end of the source time window, in picoseconds of "
                            "the source trajectory's own clock. Frames after it are not eligible.")
    for key, value in (("start_time_ps", start), ("end_time_ps", end)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"AIS.source.{key} must be a number of picoseconds; got {value!r}") from None
        if value < 0:
            raise ConfigError(f"AIS.source.{key} must be >= 0; got {value}")
        source[key] = value
    if source["end_time_ps"] < source["start_time_ps"]:
        raise ConfigError(
            f"AIS.source.end_time_ps ({source['end_time_ps']}) is earlier than start_time_ps "
            f"({source['start_time_ps']}). Both bounds are inclusive and end must not precede "
            f"start.")

    # Frame timing: either BOTH manual fields or NEITHER. One alone cannot map frames to times, and
    # accepting it would leave the other to be guessed -- which is the inference this contract
    # exists to forbid.
    manual = {key: source.get(key) for key in ("first_frame_time_ps", "frame_interval_ps")}
    given = [key for key, value in manual.items() if value is not None]
    if len(given) == 1:
        missing = next(key for key, value in manual.items() if value is None)
        raise ConfigError(
            f"AIS.source.{given[0]} is set but {missing} is not. Frame times are computed as "
            f"first_frame_time_ps + index * frame_interval_ps, so one without the other cannot "
            f"map a frame to a time -- and a frame INDEX is never treated as a time. Set both, or "
            f"clear both and let the runtime read them from the trajectory's companion "
            f"resolved_run.yaml.")
    for key in given:
        try:
            value = float(manual[key])
        except (TypeError, ValueError):
            raise ConfigError(f"AIS.source.{key} must be a number of picoseconds; "
                              f"got {manual[key]!r}") from None
        if key == "frame_interval_ps" and value <= 0:
            raise ConfigError(f"AIS.source.frame_interval_ps must be positive; got {value}")
        if value < 0:
            raise ConfigError(f"AIS.source.{key} must be >= 0; got {value}")
        source[key] = value

    # --- execution ------------------------------------------------------------------------------
    platform = str(execution.get("platform") or "")
    if not platform:
        raise ConfigError("AIS.execution.platform is empty; CUDA is the default")
    devices = execution.get("gpu_devices")
    if devices != "auto":
        if not isinstance(devices, list) or not devices or \
                any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in devices):
            raise ConfigError(
                f"AIS.execution.gpu_devices must be 'auto' or a non-empty list of device indices; "
                f"got {devices!r}")


def _check_barostat_frequency(common: dict[str, Any]) -> int:
    """`MonteCarloBarostat`'s attempt interval, in steps, validated where it is declared.

    It reaches every NPT stage, the cMD production barostat and each REST2 replica's barostat, so a
    value that is not a positive whole number of steps is refused here rather than in three
    generated scripts. Zero is refused deliberately: OpenMM reads frequency 0 as "never attempt a
    move", which would turn a stage labelled NPT into a constant-volume run that still says NPT.
    """
    from .defaults import DEFAULT_BAROSTAT_FREQUENCY_STEPS

    value = common.get("barostat_frequency_steps", DEFAULT_BAROSTAT_FREQUENCY_STEPS)
    if value is None:
        raise ConfigError(
            "explicit solvent needs common.barostat_frequency_steps; it is null. It is the "
            f"MonteCarloBarostat attempt interval in integration STEPS -- "
            f"{DEFAULT_BAROSTAT_FREQUENCY_STEPS} is OpenMM's own default. null means 'does not "
            "apply', which is true only under implicit solvent.")
    if isinstance(value, bool):
        # `true` would otherwise pass every check below as 1, which is a legal frequency and not
        # remotely what was meant.
        raise ConfigError(
            f"common.barostat_frequency_steps must be a positive whole number of steps; got "
            f"{value!r}, which is a boolean.")
    try:
        steps = int(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"common.barostat_frequency_steps must be a positive whole number of steps; got "
            f"{value!r}") from None
    if steps != value or steps < 1:
        raise ConfigError(
            f"common.barostat_frequency_steps must be a positive whole number of steps; got "
            f"{value!r}. It counts integration steps, not picoseconds, and 0 would leave a stage "
            f"labelled NPT running at constant volume.")
    return steps


def _check_tau(resolved: dict[str, Any]) -> None:
    """`cMD.tau` selects one fixed rung of the REST2 ladder; validate it before anything is built.

    Refused here rather than deep inside a generated script, where the failure would arrive after
    the System had been constructed and the run directory opened. tau = 1 is excluded because
    s = (1 - tau)^2 would be zero: a solute with no intramolecular Hamiltonian at all is not a rung
    of the ladder, it is a different calculation.
    """
    block = resolved.get("cMD") or {}
    if "tau" not in block:
        return
    tau = block["tau"]
    try:
        tau = float(tau)
    except (TypeError, ValueError):
        raise ConfigError(f"cMD.tau must be a number in [0, 1); got {block['tau']!r}") from None
    if not 0.0 <= tau < 1.0:
        raise ConfigError(
            f"cMD.tau must be in [0, 1); got {tau}. tau = 0 is ordinary conventional MD on the "
            "unscaled Hamiltonian, and tau -> 1 removes the solute Hamiltonian entirely.")
    resolved["cMD"]["tau"] = tau


def _check_timestep(resolved: dict[str, Any]) -> None:
    """4 fs is only defensible with repartitioned hydrogens; this is where that pair is checked.

    The system config carries the mass, so this check needs both files and is applied by the
    caller through `check_timestep_against_masses`.
    """
    timestep = float((resolved.get("common") or {}).get("timestep_fs", 2.0))
    if timestep <= 0:
        raise ConfigError(f"common.timestep_fs must be positive; got {timestep}")


def check_timestep_against_masses(md_resolved: dict[str, Any],
                                  sys_resolved: dict[str, Any]) -> None:
    """Refuse 4 fs on unrepartitioned hydrogens, naming both values.

    This is the one cross-file check worth making: the two settings live in different files and are
    individually reasonable, so nothing else would catch the combination.
    """
    from .defaults import HMR_HYDROGEN_MASS_AMU

    timestep = float((md_resolved.get("common") or {}).get("timestep_fs", 2.0))
    constraints = sys_resolved.get("constraints") or {}
    mass = constraints.get("hydrogen_mass_amu")
    if timestep <= 3.0:
        return
    if mass is None:
        raise ConfigError(
            f"md.config.yaml sets common.timestep_fs = {timestep} but sys.config.yaml leaves "
            "constraints.hydrogen_mass_amu null, so hydrogens keep their real mass. A timestep "
            f"above ~3 fs needs hydrogen mass repartitioning; set hydrogen_mass_amu to "
            f"{HMR_HYDROGEN_MASS_AMU}, or lower the timestep to 2.0. "
            "See docs/examples/hmr-4fs.yaml.")
    # HMR lowers the frequency of the bond-ANGLE motions involving hydrogen. The bond STRETCHES are
    # removed by the constraints, and they are the fastest motions in the system: repartitioning
    # without constraining them buys nothing and integrates a 10 fs period with a 4 fs step.
    if str(constraints.get("type")) not in ("HBonds", "AllBonds", "HAngles"):
        raise ConfigError(
            f"md.config.yaml sets common.timestep_fs = {timestep} and sys.config.yaml repartitions "
            f"hydrogen mass to {mass} amu, but constraints.type is "
            f"{constraints.get('type')!r}, so the bonds to hydrogen are not constrained. HMR "
            "lowers the hydrogen ANGLE frequencies; the bond stretches it does not touch are the "
            "fastest motions left. Set constraints.type to HBonds.")
    if sys_resolved.get("solvation") != "implicit" and not constraints.get("rigid_water", True):
        raise ConfigError(
            f"md.config.yaml sets common.timestep_fs = {timestep} but sys.config.yaml sets "
            "constraints.rigid_water false. Water is never repartitioned, so a flexible water "
            "molecule keeps its ~10 fs O-H stretch and sets the stable timestep for the whole "
            "box. Set rigid_water true.")
