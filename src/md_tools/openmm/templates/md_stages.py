"""Stage construction shared by the generated cMD and REST2 scripts.

Copied into every generated project, so it must depend on nothing but OpenMM, PyYAML and the
standard library.

The pipeline both methods run on a fresh start:

    restrained minimization
      -> restrained NVT   (no active barostat)
      -> restrained NPT   (exactly one active barostat; explicit solvent only)
      -> unrestrained production

Under implicit solvent there is no box, so there is no barostat and no NPT stage at all -- the
production ensemble is NVT.
"""
from concurrent.futures import ThreadPoolExecutor
import struct
from pathlib import Path

from openmm import (CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
                    XmlSerializer, unit)
from openmm.app import Simulation

#: U = 1/2 k |r - r0|^2, with k given in kcal mol^-1 A^-2.
KCAL_PER_MOL_ANGSTROM2 = 418.4          # kJ mol^-1 nm^-2
RESTRAINT_PARAMETER = "restraint_k"
#: OpenMM seeds are 32-bit signed; 0 means "pick one at random", which is not reproducible.
MAX_SEED = 2 ** 31 - 1

# The barostat attempt interval is NOT declared here. It is `common.barostat_frequency_steps` in
# md.config.yaml, copied into every stage.yaml by md-gen, and passed in by the caller. A constant
# in this file would be a second declaration of a public default, and the one that silently wins
# when the two disagree.


def derive_seed(base, *purpose):
    """A distinct, deterministic seed per (base, purpose).

    Every replica needs its own integrator, velocity and barostat seed. Sharing one seed across
    replicas correlates their trajectories, and a ladder whose rungs move together samples less
    than it appears to.
    """
    value = int(base)
    for part in purpose:
        text = str(part).encode("utf-8")
        for byte in text:
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    seed = value % MAX_SEED
    return seed if seed else 1              # never 0: OpenMM reads that as "choose randomly"


#: U = 1/2 k |r - r0|^2 under periodic boundaries: the minimum image, so an atom that crosses a
#: box face is pulled back to the nearest image of its reference rather than across the whole cell.
PERIODIC_RESTRAINT = "0.5*{k}*periodicdistance(x, y, z, x0, y0, z0)^2"
#: The same energy without a box. There is no minimum image to take, and asking for one is not
#: merely redundant -- see `add_positional_restraint`.
NONPERIODIC_RESTRAINT = "0.5*{k}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"


def add_positional_restraint(system, reference_positions, atom_indices):
    """A `CustomExternalForce` holding `atom_indices` near their reference coordinates.

    The energy expression depends on whether the System has a box:

        periodic      0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2
        non-periodic  0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)

    They are the same energy whenever an atom is far from any box face, so the difference is easy
    to miss. It matters because `periodicdistance` makes the Force report
    `usesPeriodicBoundaryConditions() == True`, and OpenMM answers that question for a System by
    asking its Forces: adding a periodic restraint to an implicit GBn2 system flips
    `System.usesPeriodicBoundaryConditions()` from False to True. The system then describes itself
    as periodic while having no meaningful box, which is a false statement about the physics being
    sampled and the kind of thing a later check reads and trusts.

    Periodicity is taken from the System, not from a file name or a solvent label: the System is
    what OpenMM will actually integrate. It must be read BEFORE the Force is added, because adding
    a periodic Force is precisely what would change the answer.

    The Force stays in the System for the whole run so the checkpoint layout never changes; its
    strength is a global Context parameter, set to zero before production. Removing the Force
    instead would make a production checkpoint structurally incompatible with the equilibration
    that produced it.
    """
    periodic = system.usesPeriodicBoundaryConditions()
    template = PERIODIC_RESTRAINT if periodic else NONPERIODIC_RESTRAINT
    force = CustomExternalForce(template.format(k=RESTRAINT_PARAMETER))
    force.addGlobalParameter(RESTRAINT_PARAMETER, 0.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    positions = reference_positions.value_in_unit(unit.nanometer)
    for index in atom_indices:
        x, y, z = positions[int(index)]
        force.addParticle(int(index), [x, y, z])
    system.addForce(force)
    return force


def add_barostat(system, pressure_bar, temperature, seed, frequency):
    """One barostat, at the attempt interval this stage was configured with.

    Frequency 0 means it never attempts a move, which is how the NVT stage runs without a barostat
    while the System keeps a stable Force layout for the checkpoint. There is no default: the
    interval is a recorded scientific setting, and a script that supplied its own would be able to
    integrate at an interval no record mentions.
    """
    frequency = int(frequency)
    if frequency < 0:
        raise SystemExit(f"barostat frequency must be >= 0 steps; got {frequency}")
    barostat = MonteCarloBarostat(float(pressure_bar) * unit.bar, temperature, frequency)
    barostat.setRandomNumberSeed(int(seed))
    system.addForce(barostat)
    return barostat


def count_barostats(system):
    return sum(1 for i in range(system.getNumForces())
               if "Barostat" in type(system.getForce(i)).__name__)


def set_barostat_frequency(simulation, frequency):
    """Turn the barostat on or off, and make the Context see it.

    A Force property changed after the Context exists is invisible until the Context is
    reinitialised; `preserveState=True` keeps positions, velocities and box.
    """
    system = simulation.context.getSystem()
    changed = False
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if "Barostat" in type(force).__name__:
            force.setFrequency(int(frequency))
            changed = True
    if changed:
        simulation.context.reinitialize(preserveState=True)
    return changed


def active_barostat_count(simulation):
    """Barostats that will actually attempt a move -- frequency 0 is inactive."""
    system = simulation.context.getSystem()
    return sum(1 for i in range(system.getNumForces())
               if "Barostat" in type(system.getForce(i)).__name__
               and system.getForce(i).getFrequency() > 0)


def set_restraint(simulation, k_kcal_mol_a2):
    simulation.context.setParameter(
        RESTRAINT_PARAMETER, float(k_kcal_mol_a2) * KCAL_PER_MOL_ANGSTROM2)


def restraint_strength(simulation):
    return simulation.context.getParameter(RESTRAINT_PARAMETER)


def resolve_platform(forced=None):
    """CUDA unless a different platform was asked for BY NAME.

    Simulations belong on a GPU. A silent fall back to the CPU still finishes, still writes a
    trajectory and still says "complete" -- two orders of magnitude later, on a machine whose GPU
    was simply not visible to this process. So the fallback has to be requested rather than
    inherited, and a missing CUDA platform is an error with the override spelled out.
    """
    available = sorted(Platform.getPlatform(i).getName()
                       for i in range(Platform.getNumPlatforms()))
    if forced:
        if forced not in available:
            raise SystemExit(f"MD_PLATFORM={forced} is not available in this OpenMM build. "
                             f"It has: {', '.join(available)}.")
        return forced
    if "CUDA" not in available:
        raise SystemExit(
            f"no CUDA platform in this OpenMM build (it has: {', '.join(available)}), and this "
            "run was not told to use anything else.\n"
            "Check that the GPU is visible to this process (nvidia-smi, CUDA_VISIBLE_DEVICES), "
            "or ask for another platform by name:\n"
            "    MD_PLATFORM=CPU ./run.sh")
    return "CUDA"


def make_simulation(topology, system, *, temperature, friction, timestep, seed,
                    platform_name=None, device=None):
    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
    integrator.setRandomNumberSeed(int(seed))
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name == "CUDA":
            properties = {"Precision": "mixed"}
            if device is not None:
                properties["DeviceIndex"] = str(device)
        return Simulation(topology, system, integrator, platform, properties)
    return Simulation(topology, system, integrator)


def steps_for(picoseconds, timestep_fs):
    """An exact number of steps, or a refusal. Rounding runs a different length than configured."""
    exact = float(picoseconds) * 1000.0 / float(timestep_fs)
    if abs(exact - round(exact)) > 1e-9:
        raise SystemExit(
            f"{picoseconds} ps is not a whole number of {timestep_fs} fs steps ({exact}). "
            "Choose a duration that divides exactly.")
    return int(round(exact))


def require_parent_state(path, *, stage_name, command):
    """The finalized state of the parent stage, or a refusal that says how to produce it.

    A stage is never allowed to fall back to its parent's `checkpoint.chk`. A checkpoint is written
    while a stage is still running, so consuming one as input means starting from a partially
    completed parent while every artifact looks normal.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"missing {path.name}: this stage reads {path}, which stage '{stage_name}' writes "
            f"only when it finishes.\n"
            f"Run it first:\n"
            f"    cd {command} && ./run.sh")
    return XmlSerializer.deserialize(path.read_text(encoding="utf-8"))


def build_stage_system(inputs, *, implicit, restrained, barostat_active, pressure_bar,
                       temperature, barostat_seed, barostat_frequency_steps, solute_indices,
                       scale_system=None):
    """The System a stage integrates, with its Force layout fixed before any state is loaded.

    The restraint Force is always present, at zero strength when the stage is unrestrained, so a
    State carrying a `restraint_k` parameter can be loaded into any stage's Context. The barostat
    is present only under explicit solvent, and its frequency -- not its presence -- is what makes
    a stage NVT or NPT.

    `scale_system` applies a fixed-tau REST2 Hamiltonian scaling to the System as it comes off
    disk. It runs FIRST, on the bare System, for two reasons: the scaler audits every force and
    refuses one it cannot classify, and the restraint and barostat are stage machinery rather than
    terms of the molecular Hamiltonian, so neither may be scaled. This is the same order
    `REST2/run.py` uses -- scale, then restrain -- so a fixed-tau walker and the matching ladder
    rung construct the identical System.
    """
    system = XmlSerializer.deserialize((Path(inputs) / "system.xml").read_text(encoding="utf-8"))
    if scale_system is not None:
        system = scale_system(system)
    initial = XmlSerializer.deserialize(
        (Path(inputs) / "initial_state.xml").read_text(encoding="utf-8"))
    add_positional_restraint(system, initial.getPositions(), solute_indices)
    if not implicit:
        add_barostat(system, pressure_bar, temperature, barostat_seed,
                     frequency=int(barostat_frequency_steps) if barostat_active else 0)
    barostats = count_barostats(system)
    if implicit and barostats:
        raise SystemExit(f"implicit solvent must have no barostat in the System; found {barostats}")
    return system


def write_final_state(simulation, path):
    """The handoff to the next stage: positions, velocities, box, time and parameters.

    Written only once the stage has finished, and written atomically, so a downstream stage cannot
    read a half-written file and start from a state that never existed.
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True,
                                        getParameters=True, enforcePeriodicBox=False)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    temporary.replace(path)
    return state



# ---------------------------------------------------------------------------------------------
# Uncommitted tails
# ---------------------------------------------------------------------------------------------
# A checkpoint is written every `checkpoint_interval_ps`; the trajectories and the table are
# written far more often. An interrupted stage therefore leaves reporter output that runs PAST the
# last checkpoint, and resuming from that checkpoint replays the interval -- appending frames the
# files already contain. The run completes, every file looks healthy, and the trajectory silently
# carries duplicated frames and a non-monotonic step column.
#
# So on resume every stream is cut back to the checkpoint before anything is opened for append.
# The checkpoint is the authority for where the stage actually is; a file length is a fact about
# when the process died.


def _dcd_layout(path):
    """(header_bytes, frame_bytes, n_atoms, n_frames) for a DCD written by OpenMM."""
    with open(path, "rb") as handle:
        if struct.unpack("<i", handle.read(4))[0] != 84:
            raise ValueError(f"{path}: not a little-endian DCD")
        control = handle.read(84)                      # b'CORD' + 20 int32
        n_frames = struct.unpack("<i", control[4:8])[0]
        has_unitcell = struct.unpack("<i", control[44:48])[0] != 0
        handle.read(4)                                 # closing record marker
        title_bytes = struct.unpack("<i", handle.read(4))[0]
        handle.read(title_bytes + 4)
        handle.read(4)
        n_atoms = struct.unpack("<i", handle.read(4))[0]
        handle.read(4)
        header_bytes = handle.tell()
    frame_bytes = (56 if has_unitcell else 0) + 3 * (8 + 4 * n_atoms)
    return header_bytes, frame_bytes, n_atoms, n_frames


def truncate_dcd(path, keep_frames):
    """Cut a DCD back to `keep_frames`, rewriting the frame count in its header."""
    path = Path(path)
    if not path.is_file():
        return 0
    header_bytes, frame_bytes, _, n_frames = _dcd_layout(path)
    if keep_frames >= n_frames:
        return n_frames
    with open(path, "r+b") as handle:
        handle.truncate(header_bytes + keep_frames * frame_bytes)
        handle.seek(8)                                 # icntrl[0], the frame count
        handle.write(struct.pack("<i", keep_frames))
    return keep_frames


def truncate_table(path, keep_rows):
    """Cut a StateDataReporter CSV back to `keep_rows` data rows, keeping its single header."""
    path = Path(path)
    if not path.is_file():
        return 0
    lines = path.read_text().splitlines(keepends=True)
    if not lines:
        return 0
    header, body = lines[0], lines[1:]
    if len(body) <= keep_rows:
        return len(body)
    path.write_text("".join([header] + body[:keep_rows]))
    return keep_rows


def trim_to_checkpoint(directory, done_steps, *, whole_every, solute_every, table_every):
    """Discard reporter output past the checkpoint. Returns what each stream was cut to.

    Reporters fire at multiples of their interval, so a checkpoint at step N means exactly
    N // interval frames belong to the committed history.
    """
    directory = Path(directory)
    kept = {
        "whole_system.dcd": truncate_dcd(directory / "whole_system.dcd", done_steps // whole_every),
        "solute.dcd": truncate_dcd(directory / "solute.dcd", done_steps // solute_every),
        "production.csv": truncate_table(directory / "production.csv", done_steps // table_every),
    }
    return kept

#: The runtime outputs a stage writes. `run.py`, `run.sh` and `stage.yaml` are inputs and are not
#: in this list: removing them would delete the stage rather than its results.
STAGE_RUNTIME_OUTPUTS = ("stage.log", "stage.csv", "checkpoint.chk", "final_state.xml",
                         "final.pdb", "resolved_stage.yaml")


def stage_config_sha256(document):
    """One signature over the WHOLE stage request.

    A hand-kept list of "the fields that matter" is a list that goes stale: it accepted a completed
    stage after its input state, pressure, seeds or solvent mode had changed, because those keys
    were not on it. Hashing the entire canonicalised document instead means every key counts, and
    a key added later counts without anyone remembering to add it here.

    `sort_keys=True` makes the text independent of the order the mapping happens to be written in,
    so the same request always hashes the same way.
    """
    import hashlib

    import yaml

    canonical = yaml.safe_dump(document, sort_keys=True, default_flow_style=False,
                               allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: The fields of a PRODUCTION stage request that a legitimate extension may change.
#:
#: A common stage is fixed at generation and refuses to rerun at all, so hashing its whole
#: document is exactly right. Production is different: running longer is the documented way to
#: extend an active dataset, and it changes the requested length and nothing else. Hashing the
#: whole document there would forbid every extension in order to catch an unsafe edit.
#:
#: So production carries two fingerprints. The full one identifies the exact request that
#: produced a completion record; the INVARIANT one covers everything an extension may not
#: touch -- timestep, temperature, ensemble, tau, seeds, solvent mode, the parent handoff. A
#: changed invariant means the checkpoint on disk was produced under different physics, and
#: continuing from it would silently splice two Hamiltonians into one trajectory.
PRODUCTION_EXTENDABLE_FIELDS = ("duration_ns", "duration_ps", "total_steps",
                                "number_of_exchanges", "number_of_paths")


def stage_invariant_sha256(document):
    """Fingerprint over everything a legitimate extension may NOT change.

    Same canonicalisation as :func:`stage_config_sha256`, over the document with the extendable
    fields removed. Deriving it from the same function is deliberate: two independent hashes over
    "the stage request" is how a run comes to write one and a check to compare another.
    """
    return stage_config_sha256(
        {k: v for k, v in dict(document).items() if k not in PRODUCTION_EXTENDABLE_FIELDS})


def production_stage_document(config, method_name, *, parent_stage, parent_path, implicit,
                              seeds, template_commit, omega_excluded=()):
    """The resolved request for one production stage.

    ONE derivation, used by `md-gen` when it writes `stage.yaml` and by the generated launcher
    when it recomputes the current request. If the two derived it separately they would drift,
    and the check would compare a hash of one thing against a hash of another.
    """
    common = config["common"]
    method = config[method_name]
    tau = float(method.get("tau", 0.0) or 0.0)
    document = {
        "name": method_name,
        "group": None,
        "kind": "fixed_tau_md" if (method_name == "cMD" and tau) else {
            "cMD": "conventional_md", "REST2": "rest2_exchange", "AIS": "ais_switching",
        }.get(method_name, method_name.lower()),
        "production": True,
        "ensemble": method.get("ensemble"),
        "tau": tau,
        "omega_exclusion": bool(method.get("omega_exclusion", True)) and bool(tau),
        "omega_excluded_bonds": [list(map(int, b)) for b in omega_excluded] if tau else [],
        "temperature_kelvin": float(common["temperature_kelvin"]),
        "timestep_fs": float(common["timestep_fs"]),
        "friction_per_ps": float(common["friction_per_ps"]),
        "implicit": bool(implicit),
        "pressure_bar": None if implicit else float(common["pressure_bar"]),
        "barostat_frequency_steps": (0 if implicit
                                     else int(common["barostat_frequency_steps"])),
        "hydrogen_mass_amu": (config.get("constraints") or {}).get("hydrogen_mass_amu"),
        "parent": parent_stage,
        "parent_path": parent_path,
        "input_state": f"../{parent_stage}/final_state.xml" if parent_stage else None,
        # cMD hands one final state on. REST2 keeps a state per replica and has no single
        # top-level handoff, so it declares none rather than naming a file that never appears.
        "output_state": "final_state.xml" if method_name == "cMD" else None,
        "integrator_seed": seeds.get("integrator"),
        "velocity_seed": seeds.get("velocities"),
        "barostat_seed": seeds.get("barostat"),
        "template_commit": template_commit,
    }
    # The extendable part, kept in the document because a reader wants to see the request, and
    # excluded from the invariant fingerprint because extending is legitimate.
    if "duration_ns" in method:
        document["duration_ns"] = float(method["duration_ns"])
    if "number_of_exchanges" in method:
        document["number_of_exchanges"] = int(method["number_of_exchanges"])

    # AIS states its request in nested blocks rather than flat keys, and it has no parent stage:
    # it starts from an equilibrium ensemble prepared into inputs/, not from the common chain. The
    # fields below ARE the physics of a switching path -- where it starts, where it ends, how long
    # the switch takes and how finely the Hamiltonian moves -- so they belong in the invariant.
    # `number_of_paths` is the one extendable quantity: running more paths is legitimate.
    if method_name == "AIS":
        path = method["path"]
        document["kind"] = "ais_switching"
        document["ensemble"] = "NVT" if implicit else "NPT"
        document["tau"] = float(path["tau_start"])
        document["tau_start"] = float(path["tau_start"])
        document["tau_end"] = float(path["tau_end"])
        document["interpolation"] = path.get("interpolation")
        document["enhanced_region"] = path.get("enhanced_region")
        document["omega_exclusion"] = bool(path.get("omega_exclusion", True))
        document["switching_duration_ps"] = float(path["switching_duration_ps"])
        document["parameter_update_interval_steps"] = int(
            path["parameter_update_interval_steps"])
        document["number_of_observations"] = int(method["output"]["number_of_observations"])
        document["number_of_paths"] = int(method["source"]["number_of_trajectories"])
    return document


RECORD_FORMAT = "md-tools-runtime-record/v1"


def utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    """Streamed, so a multi-gigabyte checkpoint does not have to fit in memory."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path, *, digest=True):
    """Name, size and optionally checksum. Missing files are recorded as null, not omitted."""
    path = Path(path)
    if not path.is_file():
        return None
    record = {"path": path.name, "bytes": path.stat().st_size}
    if digest:
        record["sha256"] = sha256_file(path)
    return record


def dcd_frame_count(path):
    """Frames from the DCD header, without a trajectory library.

    Trajectories are NOT checksummed at runtime: they are large, they grow across invocations, and
    MD-data computes the archival digests once. Path, size and frame count are what runtime knows
    cheaply and exactly.
    """
    import struct

    path = Path(path)
    if not path.is_file() or path.stat().st_size < 24:
        return None
    try:
        with path.open("rb") as handle:
            head = handle.read(24)
        return int(struct.unpack("<i", head[8:12])[0])
    except Exception:                                   # noqa: BLE001
        return None


def trajectory_record(path, *, atom_scope, reporter_interval_steps=None, timestep_fs=None):
    """Path, size, frame count -- and the frame-to-time map, recorded where it is known.

    The time map is the part a later analysis cannot reconstruct safely on its own. A DCD frame
    index is not a time, and a DCD header's own step fields describe how the file was written
    rather than which production clock the frames belong to. What IS known here is the reporter
    interval and the timestep, and OpenMM's DCDReporter writes its first frame at step `interval`
    (not at step 0), so frame k sits at step (k+1)*interval. Writing that down once, at the point
    it is decided, is what lets `MD/AIS/run.py` map a source frame to a physical time without
    inferring anything.
    """
    path = Path(path)
    if not path.is_file():
        return None
    record = {"path": path.name, "atom_scope": atom_scope, "bytes": path.stat().st_size,
              "frames": dcd_frame_count(path)}
    if reporter_interval_steps and timestep_fs:
        interval_ps = float(reporter_interval_steps) * float(timestep_fs) / 1000.0
        record["frame_time_map"] = {
            "reporter_interval_steps": int(reporter_interval_steps),
            "timestep_fs": float(timestep_fs),
            # The first frame lands at step `interval`, so it is one interval into the run.
            "first_frame_time_ps": round(interval_ps, 9),
            "frame_interval_ps": round(interval_ps, 9),
            "convention": ("frame k (0-based) is at step (k+1)*reporter_interval_steps; "
                           "time_ps = first_frame_time_ps + k * frame_interval_ps"),
        }
    return record


def append_jsonl(path, entry):
    """One JSON object per line, appended.

    Append-only on purpose: an invocation history that can be rewritten is not a history. Opening
    in append mode means a crash mid-write costs the last line, never the earlier ones.
    """
    import json

    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=False) + "\n")
    return path


def next_invocation_index(path):
    """How many invocations have already been recorded here."""
    path = Path(path)
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def project_identity(config):
    """Which MD-tools generated this project, as carried in md.config.yaml."""
    provenance = config.get("provenance") or {}
    return {"md_tools_version": provenance.get("md_tools_version"),
            "template_commit": provenance.get("template_commit"),
            "installed_fingerprint": provenance.get("installed_fingerprint")}


def write_yaml_atomic(path, document):
    """Replace, never partially overwrite.

    A completion record is read to decide whether a stage is finished. Half of one, left by an
    interrupted write, would be read as a stage in a state it was never in.
    """
    import yaml

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path


def write_final_pdb(simulation, path, *, implicit):
    from openmm.app import PDBFile

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=not implicit)
    with Path(path).open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


def device_groups(n_replicas, devices):
    """Replicas grouped by the device they run on, round-robin over at most n_replicas devices.

    Returns one list per device actually used, so `len(groups)` is
    `min(n_replicas, n_visible_devices)`. With no devices there is one group: the sequential path.
    """
    if not devices:
        return [list(range(n_replicas))]
    used = list(devices)[:min(n_replicas, len(devices))]
    groups = [[] for _ in used]
    for replica in range(n_replicas):
        groups[replica % len(used)].append(replica)
    return [group for group in groups if group]


def propagate_segment(groups, step_fn):
    """Advance every replica by one segment, then return.

    Replicas that share a device propagate sequentially -- interleaving them on one GPU makes both
    slower, not faster. Different devices propagate concurrently: OpenMM releases the GIL while
    stepping, so plain threads genuinely overlap. Every group is awaited before the caller attempts
    an exchange, because an exchange reads energies that a still-running replica has not produced.
    """
    if len(groups) <= 1:
        for replica in (groups[0] if groups else ()):
            step_fn(replica)
        return

    def run_group(group):
        for replica in group:
            step_fn(replica)

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(run_group, group) for group in groups]
        for future in futures:
            future.result()             # re-raise here, before any exchange is attempted
