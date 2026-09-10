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
# md.config.yaml, copied into every stage.yaml by build-md, and passed in by the caller. A constant
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




def active_barostat_count(simulation):
    """Barostats that will actually attempt a move -- frequency 0 is inactive."""
    system = simulation.context.getSystem()
    return sum(1 for i in range(system.getNumForces())
               if "Barostat" in type(system.getForce(i)).__name__
               and system.getForce(i).getFrequency() > 0)


def set_restraint(simulation, k_kcal_mol_a2):
    simulation.context.setParameter(
        RESTRAINT_PARAMETER, float(k_kcal_mol_a2) * KCAL_PER_MOL_ANGSTROM2)




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




def steps_for(picoseconds, timestep_fs):
    """An exact number of steps, or a refusal. Rounding runs a different length than configured."""
    exact = float(picoseconds) * 1000.0 / float(timestep_fs)
    if abs(exact - round(exact)) > 1e-9:
        raise SystemExit(
            f"{picoseconds} ps is not a whole number of {timestep_fs} fs steps ({exact}). "
            "Choose a duration that divides exactly.")
    return int(round(exact))






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



#: The runtime outputs a stage writes. `run.py`, `run.sh` and `stage.yaml` are inputs and are not
#: in this list: removing them would delete the stage rather than its results.
STAGE_RUNTIME_OUTPUTS = ("stage.log", "stage.csv", "checkpoint.chk", "final_state.xml",
                         "final.pdb", "resolved_stage.yaml")








# `production_stage_document` lived here. It built the production stage record for the retired
# `methods:` configuration -- the only code left reading `duration_ns` and
# `switching_duration_ps` -- and had no caller once that route was removed. `build.md`
# writes the stage document now, in integer steps.



RECORD_FORMAT = "md-tools-runtime-record/v1"




# `sha256_file` is `md_tools.build.record.sha256_file`. It lived in four modules -- here,
# `build/record.py`, `openmm/provenance_min.py` and `remd/source_ensemble.py` -- and all
# four produced the same digest, verified before three of them were removed. The canonical
# one is in `build/record.py` because hashing a file needs no OpenMM, and that module is
# the lowest-level of the four: everything here already depends on it.
from ..build.record import sha256_file   # noqa: F401





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


















# -- where the StateDataReporter's table lands ------------------------------------------------
#
# Two callers need this name and must agree on it: `stage_main`, which opens the file, and the
# preflight inventory, which decides collisions, what `--overwrite` governs, and -- through
# `completion.verify` -- whether a finished stage's outputs are still on disk. When they
# disagreed, a stage that had completed correctly failed its own verification because the
# verifier looked for `<stage>.csv` while the run had written `mdout_<stage>.csv`. The rule
# lives here so there is one of it.
PRODUCTION_STAGE_NAMES = frozenset({"cMD", "umbrella"})


def info_csv_name(stage_name: str) -> str:
    """`mdout.csv` for a production stage, `mdout_<stage>.csv` for every other."""
    name = str(stage_name or "stage")
    return "mdout.csv" if name in PRODUCTION_STAGE_NAMES else f"mdout_{name}.csv"
