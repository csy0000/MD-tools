#!/usr/bin/env python
"""Reading configurations out of an equilibrium production trajectory. Shared by AIS and rREST2.

Copied verbatim into every generated project that needs a source ensemble.

Both AIS and reservoir REST2 begin from configurations drawn out of a finished, fixed-tau
equilibrium run. The rules for doing that safely are subtle, were established once for AIS, and
must not be re-derived: a second implementation is a second thing that can be wrong, and its
wrongness would be silent -- a path or a reservoir seeded from the wrong ensemble still runs, still
completes, and still writes a healthy record.

So this module owns them, and both methods call it. It takes an explicit `SourceRequest` rather
than reading a method's configuration, because the two methods name their fields differently and
neither should have to know the other's.

THE RULES, AND WHY EACH ONE EXISTS
    * Never `mdtraj.load` a production trajectory. It reads the whole thing into memory and an
      equilibrium source may be tens of gigabytes. Everything needed -- frame count, atom count,
      box presence, and a few selected frames -- is obtainable a bounded chunk at a time.
    * Never hash a production trajectory at run time. MD-data hashes it once, at archival. Record
      bounded observations instead: path, size, frame count, chunk size, chunks read.
    * A frame index is NOT a time, and a DCD header is NOT the production clock. Physical frame
      times come from the trajectory's own `time` variable, or from explicitly supplied fields, or
      the run refuses.
    * Source tau and source temperature are NOT the directory name. `cMD_tau0p5` is a hint to a
      reader, never evidence. They come from the trajectory's own file attributes or from an
      explicit declaration, and if both exist and disagree the run refuses rather than preferring
      one.
    * Atom NAME is not identity. A protein is full of repeated names, and the System's parameters
      are assigned per index, so a renumbered topology compares equal on names while scaling the
      wrong atoms.
    * A DCD carries no velocities. Momenta are redrawn from the Maxwell distribution at the common
      temperature with a recorded seed; a configuration is not a restart.
    * Explicit box vectors come from recorded exact values, reduced for OpenMM -- never rebuilt
      from DCD lengths and angles as though that were the same thing.
    * The selection is materialised ONCE into a small prepared file with a manifest. After that the
      production source is never reopened, so a run survives the source being archived or deleted.
      Prepared inputs that disagree with their manifest are REFUSED, never silently regenerated:
      regenerating deletes the configurations a finished path or a running ladder started from.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

#: How many frames `mdtraj.iterload` holds at once. Bounded on purpose.
DEFAULT_CHUNK_FRAMES = 256

#: The manifest format written beside a prepared ensemble.
PREPARED_FORMAT = "md-tools-prepared-source/v1"

#: Per-atom identity fields, in the order `atom_identity` returns them.
ATOM_FIELDS = ("chain id", "chain index", "residue index", "residue id", "residue name",
               "atom name", "element")


class SourceError(ValueError):
    """The source ensemble cannot be established or does not match what the caller needs."""


class SourceRequest:
    """Everything the caller must state. Nothing here is guessed from a path or a directory name.

    `field_prefix` only shapes error messages, so AIS says `AIS.source.start_time_ps` and rREST2
    says `rREST2.reservoir.start_time_ps` while sharing one implementation.
    """

    def __init__(self, *, project, prepared_topology, trajectory, start_time_ps, end_time_ps,
                 count, seed, implicit, required_tau, topology=None, declared_tau=None,
                 declared_temperature_k=None, required_temperature_k=None,
                 first_frame_time_ps=None, frame_interval_ps=None, allow_replacement=False,
                 field_prefix="source", chunk_frames=DEFAULT_CHUNK_FRAMES, purpose="source",
                 selection_purpose=None, replacement_rationale=None):
        self.project = Path(project)
        self.prepared_topology = Path(prepared_topology)
        self.trajectory = str(trajectory)
        self.topology = None if topology is None else str(topology)
        self.start_time_ps = float(start_time_ps)
        self.end_time_ps = float(end_time_ps)
        self.count = int(count)
        self.seed = int(seed)
        self.implicit = bool(implicit)
        self.required_tau = float(required_tau)
        self.declared_tau = None if declared_tau is None else float(declared_tau)
        self.declared_temperature_k = (None if declared_temperature_k is None
                                       else float(declared_temperature_k))
        self.required_temperature_k = (None if required_temperature_k is None
                                       else float(required_temperature_k))
        self.first_frame_time_ps = (None if first_frame_time_ps is None
                                    else float(first_frame_time_ps))
        self.frame_interval_ps = (None if frame_interval_ps is None
                                  else float(frame_interval_ps))
        self.allow_replacement = bool(allow_replacement)
        self.field_prefix = str(field_prefix)
        self.chunk_frames = int(chunk_frames)
        self.purpose = str(purpose)
        # The exact seed-stream identity for frame selection. AIS passes its historical
        # ("AIS", "source-selection") so its projects keep selecting the configurations they
        # always did; a new caller may take the default.
        self.selection_purpose = tuple(selection_purpose) if selection_purpose is not None \
            else (self.purpose, "selection")
        # WHY drawing the same configuration twice is wrong differs by method: two AIS paths from
        # one configuration are not two independent realisations, while a reservoir that repeats a
        # frame is a smaller reservoir than it claims. The caller states its own reason so the
        # refusal explains the science rather than the arithmetic.
        self.replacement_rationale = replacement_rationale or (
            "Drawing the same configuration more than once makes the sample smaller than it "
            "claims to be.")

    def field(self, name):
        return f"{self.field_prefix}.{name}"


# --- small helpers -------------------------------------------------------------------------------





def relative_to(root, path):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
# `derive_seed` and `sha256_file` were copied here. They are imported from `md_tools.md`
# now -- one implementation each. The copies were verified identical first: the seed
# derivations agreed on 3000 random inputs and the digests matched byte for byte.
#
# The seed derivation is the repository's multiplicative hash and NOT a SHA-256 of the
# same inputs, deliberately: AIS selects its frames from `derive_seed(BASE_SEED, "AIS",
# "source-selection")`, so any other derivation would silently change which
# configurations every existing AIS project starts from.
from ..md._stages import derive_seed, sha256_file   # noqa: F401



def reduced_box_vectors(vectors):
    """Put box vectors into the reduced form OpenMM requires, without changing the lattice.

    A DCD stores a box as three lengths and three angles. MDTraj rebuilds vectors from those in the
    usual lower-triangular convention, which is a correct description of the same lattice but not
    necessarily the REDUCED one OpenMM insists on:

        openmm.OpenMMException: Periodic box vectors must be in reduced form.

    Reduction subtracts integer multiples of the earlier vectors from the later ones. That changes
    which vectors name the lattice, not the lattice itself -- the cell volume and every periodic
    image are unchanged -- so the box propagated in is the box the source frame had. This matters
    for a rhombic dodecahedron, which is exactly the shape this repository builds.
    """
    a, b, c = ([float(x) for x in row] for row in vectors)
    scale = round(c[1] / b[1]) if b[1] else 0.0
    c = [c[i] - scale * b[i] for i in range(3)]
    scale = round(c[0] / a[0]) if a[0] else 0.0
    c = [c[i] - scale * a[i] for i in range(3)]
    scale = round(b[0] / a[0]) if a[0] else 0.0
    b = [b[i] - scale * a[i] for i in range(3)]
    return [a, b, c]


# --- resolving the source ---------------------------------------------------------------------

def same_lattice(first, second, *, tolerance=1e-6):
    """Do two sets of box vectors describe the SAME periodic lattice?

    Not the same numbers -- the same lattice. A box may be written in several equivalent bases
    that differ by adding integer multiples of earlier vectors to later ones, and reduction to
    OpenMM's preferred form has a genuine boundary case: when |c_x| is exactly a_x/2, rounding may
    land on +a_x/2 or -a_x/2, and both are correct descriptions of one rhombic dodecahedron.

    Comparing representations reported a mismatch between a reservoir and the ladder it came from.
    The right question is whether one basis is an integer change of basis away from the other,
    which is exact: `inv(A) @ B` must be an integer matrix of determinant +/-1.
    """
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != (3, 3) or b.shape != (3, 3):
        return False
    try:
        transform = np.linalg.solve(a.T, b.T).T
    except np.linalg.LinAlgError:
        return False
    rounded = np.rint(transform)
    if not np.allclose(transform, rounded, atol=tolerance):
        return False
    return abs(abs(float(np.linalg.det(rounded))) - 1.0) < tolerance


def resolve_source_paths(request):
    """The source trajectory and topology, absolute and as configured.

    Both are kept. The absolute path says what this execution actually read; the configured path is
    what stays true when the project is moved, which is why paths in a generated project are
    relative.
    """
    configured = request.trajectory
    trajectory = Path(configured)
    if not trajectory.is_absolute():
        trajectory = (request.project / configured).resolve()
    if not trajectory.is_file():
        raise SourceError(
            f"{request.field('trajectory')} = {configured!r} does not resolve to a file.\n"
            f"  Looked for: {trajectory}\n"
            f"  Relative paths are resolved against the generated project, {request.project}.")

    if request.topology:
        topology = Path(request.topology)
        if not topology.is_absolute():
            topology = (request.project / request.topology).resolve()
        chosen = request.field("topology")
    else:
        topology = request.prepared_topology
        chosen = f"{relative_to(request.project, request.prepared_topology)} " \
                 f"({request.field('topology')} was null)"
    if not topology.is_file():
        raise SourceError(f"the source topology {topology} does not exist")
    return {"trajectory": trajectory, "trajectory_configured": configured,
            "topology": topology, "topology_choice": chosen}


def trajectory_identity(trajectory):
    """`tau`, `temperature_k` and the per-frame times an AMBER NetCDF source records ITSELF.

    THE FILE IS ITS OWN RUNTIME RECORD. Every trajectory this repository writes is AMBER NetCDF
    carrying a real `time` variable and, as our own attributes, the `tau` and `temperature_k` the
    frames were sampled at -- written by the reporter that produced them, at the moment it
    produced them.

    That is strictly better evidence than the sidecar this module looked for first. A
    `resolved_run.yaml` can be copied away from its trajectory, can describe a directory holding
    several runs, and -- as it turned out -- was written by nothing in this repository at all, so
    the refusals below advised pointing at "a trajectory written by this repository's runtime
    (which records the map)" when no such trajectory existed. The attributes cannot be separated
    from the frames they describe, because they are in the same file.

    Returns `(identity, None)` or `(None, reason)`; absence is never an error here.
    """
    path = Path(trajectory)
    if path.suffix.lower() != ".nc":
        return None, f"{path.name} is not AMBER NetCDF, so it carries no recorded identity"
    try:
        import netCDF4

        with netCDF4.Dataset(str(path)) as dataset:
            if getattr(dataset, "Conventions", None) != "AMBER":
                return None, f"{path.name} is NetCDF but does not declare the AMBER convention"
            identity = {
                "tau": (float(dataset.tau) if hasattr(dataset, "tau") else None),
                "temperature_k": (float(dataset.temperature_k)
                                  if hasattr(dataset, "temperature_k") else None),
                "times_ps": ([float(value) for value in dataset.variables["time"][:]]
                             if "time" in dataset.variables else None),
                "title": getattr(dataset, "title", None),
                "system_sha256": getattr(dataset, "system_sha256", None),
            }
        return identity, None
    except Exception as failure:                              # noqa: BLE001 - evidence, not flow
        return None, f"{path.name} could not be read for its identity: {failure}"


# THE `resolved_run.yaml` SIDECAR IS GONE, and was never there.
#
# This module used to consult a companion `resolved_run.yaml` -- searched beside the trajectory
# and then up to four directories above it -- for the frame-time map, the source tau and the
# source temperature. NOTHING IN THIS REPOSITORY HAS EVER WRITTEN THAT FILE. Only two test
# fixtures fabricated one, which is why the branch stayed green while being unreachable.
#
# An unreachable fallback is worse than no fallback, because a reader trusts it. It made three
# refusals below advise pointing at "a trajectory written by this repository's runtime (which
# records the map)" when what the runtime records is the NetCDF attributes read by
# `trajectory_identity`, not a sidecar. And it was the ONLY route `source_temperature` had, so an
# rREST2 reservoir drawn from a trajectory whose own `temperature_k` attribute stated the answer
# was refused for want of a file that could not exist.
#
# `trajectory_identity` is the replacement and is strictly better evidence: the attributes are
# written by the reporter that produced the frames, into the same file, so they cannot be copied
# away from them, cannot describe a directory holding several runs, and cannot go stale.


def source_frame_timing(request, trajectory, n_frames):
    """`(times_ps, evidence)` for every frame, or a refusal.

    A frame index is never a time and a DCD header is never the production clock. Two routes:
      1. explicit `first_frame_time_ps` and `frame_interval_ps` from the caller;
      2. the trajectory's OWN per-frame `time` variable, written by the reporter that produced
         the frames. Not a two-number model but the actual clock, so a source whose cadence
         changed between segments is described correctly rather than plausibly.
    """
    if request.first_frame_time_ps is not None and request.frame_interval_ps is not None:
        first, interval = request.first_frame_time_ps, request.frame_interval_ps
        times = [first + index * interval for index in range(n_frames)]
        return times, {"route": "configured", "first_frame_time_ps": first,
                       "frame_interval_ps": interval,
                       "source": f"{request.field('first_frame_time_ps')} / "
                                 f"{request.field('frame_interval_ps')}"}

    # THE FILE'S OWN CLOCK. An AMBER NetCDF written by this repository
    # carries a real `time` per frame, so the times are not derived from a first-frame-plus-
    # interval model at all -- they are read. That also means a source with a non-uniform
    # cadence (a run whose interval changed between segments) is described correctly, which the
    # two-number model cannot do.
    identity, _ = trajectory_identity(trajectory)
    if identity is not None and identity["times_ps"]:
        times = list(identity["times_ps"])
        if len(times) >= n_frames:
            times = times[:n_frames]
            spacing = ({round(b - a, 6) for a, b in zip(times, times[1:])} if len(times) > 1
                       else set())
            return times, {"route": "trajectory_time_variable",
                           "first_frame_time_ps": times[0],
                           "frame_interval_ps": (spacing.pop() if len(spacing) == 1 else None),
                           "source": f"{Path(trajectory).name} time variable"}

    raise SourceError(
        f"cannot establish a physical time for the frames of {Path(trajectory).name}.\n"
        f"  It records no per-frame time of its own -- it is not an AMBER NetCDF written by this "
        f"repository, and those are the files that carry a real `time` variable per frame -- and "
        f"{request.field('first_frame_time_ps')} / {request.field('frame_interval_ps')} are "
        f"not set.\n"
        f"  A frame index is not a time, and this refuses to invent one. Either point "
        f"{request.field('trajectory')} at a trajectory written by this repository's runtime "
        f"(which records the map), or set BOTH of:\n"
        f"      {request.field('first_frame_time_ps')}   time of frame 0, in ps\n"
        f"      {request.field('frame_interval_ps')}     spacing between frames, in ps")


def source_tau(request, trajectory):
    """`(tau, evidence, route)` for the source ensemble. NEVER from the directory name.

    A directory called `cMD_tau0p5` is a hint to a human reader and nothing more: it can be
    renamed, copied, or simply wrong, and a method seeded from the wrong ensemble completes
    normally while being incorrect. Two routes, and a disagreement is refused rather than resolved
    by preferring one -- whichever were preferred, the other would be wrong.
    """
    declared = request.declared_tau
    recorded, evidence = None, None

    # The trajectory's OWN attribute first. It was written by the reporter that produced the
    # frames, so it cannot be separated from them -- unlike a sidecar, and unlike the directory
    # name this docstring rejects.
    identity, _ = trajectory_identity(trajectory)
    if identity is not None and identity["tau"] is not None:
        recorded = float(identity["tau"])
        evidence = f"{Path(trajectory).name} tau attribute"

    if declared is not None and recorded is not None:
        if abs(declared - recorded) > 1e-9:
            raise SourceError(
                f"{request.field('source_tau')} is {declared}, but the source trajectory's own "
                f"runtime record says tau = {recorded} ({evidence}).\n"
                f"  These must agree. Correct the configuration, or point "
                f"{request.field('trajectory')} at the run you meant.")
        return recorded, f"{evidence}; confirmed by {request.field('source_tau')}", \
            "trajectory attribute"
    if recorded is not None:
        return recorded, evidence, "trajectory attribute"
    if declared is not None:
        return declared, (f"{request.field('source_tau')} ({Path(trajectory).name} records no "
                          f"tau of its own)"), "explicit configuration"

    raise SourceError(
        f"cannot establish the Hamiltonian tau of the source trajectory "
        f"{Path(trajectory).name}.\n"
        f"  It records no `tau` of its own -- it is not an AMBER NetCDF written by this "
        f"repository, and those carry the tau their frames were sampled at as a file "
        f"attribute -- and {request.field('source_tau')} is not set.\n"
        f"  This is refused rather than assumed to be {request.required_tau}. A directory name "
        f"such as `cMD_tau0p5` is NOT evidence: it can be renamed or copied, and a "
        f"{request.purpose} seeded from the wrong ensemble runs to completion while being wrong.\n"
        f"  Either point {request.field('trajectory')} at a fixed-tau run produced by this "
        f"repository, whose runtime record names its tau, or state it yourself:\n"
        f"      {request.field('source_tau')}: {request.required_tau}")


def source_temperature(request, trajectory):
    """`(temperature_k, evidence)` for the source ensemble, or `(None, reason)`.

    A reservoir must match the state it refreshes in temperature as well as in tau, so this is
    evidence when the caller asks for it. AIS does not require it; rREST2 does.
    """
    declared = request.declared_temperature_k
    recorded, evidence = None, None

    # THE TRAJECTORY'S OWN ATTRIBUTE, exactly as `source_tau` reads tau. This used to have no
    # route but the `resolved_run.yaml` sidecar that nothing writes, so a reservoir drawn from a
    # trajectory carrying `temperature_k` -- every AMBER NetCDF this repository produces -- was
    # refused with "no companion runtime record" while the answer sat in the file being read.
    identity, _ = trajectory_identity(trajectory)
    if identity is not None and identity["temperature_k"] is not None:
        recorded = float(identity["temperature_k"])
        evidence = f"{Path(trajectory).name} temperature_k attribute"

    if declared is not None and recorded is not None:
        if abs(declared - recorded) > 1e-9:
            raise SourceError(
                f"{request.field('source_temperature_k')} is {declared} K but the source's own "
                f"runtime record says {recorded} K ({evidence}). These must agree.")
        return recorded, f"{evidence}; confirmed by {request.field('source_temperature_k')}"
    if recorded is not None:
        return recorded, evidence
    if declared is not None:
        return declared, (f"{request.field('source_temperature_k')} "
                          f"({Path(trajectory).name} records no temperature_k of its own)")
    return None, (f"{Path(trajectory).name} records no temperature_k of its own -- it is not an "
                  f"AMBER NetCDF written by this repository -- and "
                  f"{request.field('source_temperature_k')} is not set")


def eligible_frames(times, start_time_ps, end_time_ps, *, frame_interval_ps=None):
    """The frames of `times` inside an INCLUSIVE window, as source indices in order.

    One implementation, two callers: the selector below and the rREST2 reservoir boundary, which
    has to report how many DISTINCT frames a request could draw from. Computing the window twice
    would let the count in an error message drift from the count the selector actually used.

    Half a frame interval of tolerance: a frame written at exactly `end_time_ps` must be eligible,
    and floating-point time accumulation must not exclude it.
    """
    tolerance = 0.5 * float(frame_interval_ps or 0.0) * 1e-6
    return [index for index, time in enumerate(times)
            if start_time_ps - tolerance <= time <= end_time_ps + tolerance]


def select_source_frames(request, times, evidence):
    """Which frames are taken -- deterministic, inclusive, and recorded before anything runs.

    Uniform over the eligible frames, without replacement by default. The window is INCLUSIVE at
    both ends, which is what `start_time_ps` and `end_time_ps` mean.
    """
    start, end = request.start_time_ps, request.end_time_ps
    eligible = eligible_frames(times, start, end,
                               frame_interval_ps=evidence.get("frame_interval_ps"))

    if not eligible:
        raise SourceError(
            f"no frame of the source trajectory falls in [{start}, {end}] ps.\n"
            f"  The trajectory covers {times[0]:g} to {times[-1]:g} ps ({len(times)} frames, "
            f"{evidence.get('frame_interval_ps')} ps apart, from {evidence.get('source')}).\n"
            f"  Both bounds are inclusive; widen {request.field('start_time_ps')} / "
            f"{request.field('end_time_ps')}.")
    if not request.allow_replacement and request.count > len(eligible):
        raise SourceError(
            f"{request.field('count')} = {request.count}, but only {len(eligible)} frame(s) fall "
            f"in [{start}, {end}] ps and replacement is not allowed.\n"
            f"  {request.replacement_rationale} Either widen the time window, ask for at most "
            f"{len(eligible)}, or allow replacement and accept that the drawn configurations "
            f"repeat.")

    # Seeded from the recorded master seed, so the same project selects the same frames on any
    # machine. numpy's Generator is used rather than `random` because its stream is versioned.
    generator = np.random.default_rng(derive_seed(request.seed, *request.selection_purpose))
    chosen = generator.choice(len(eligible), size=request.count,
                              replace=request.allow_replacement)
    return [int(eligible[int(i)]) for i in chosen], eligible


# --- reading, bounded -----------------------------------------------------------------------------

def _iterload(paths, chunk):
    """`mdtraj.iterload` over the source, in bounded chunks. `mdtraj.load` must not reappear."""
    import mdtraj

    return mdtraj.iterload(str(paths["trajectory"]), top=str(paths["topology"]), chunk=chunk)


def survey_source(paths, *, chunk=DEFAULT_CHUNK_FRAMES):
    """Pass one: how many frames, how many atoms, whether every frame carries a box.

    Nothing is retained: this exists so the frame count and the atom check happen without the
    trajectory ever being resident.
    """
    n_frames, n_atoms, chunks, boxes_present = 0, None, 0, True
    for piece in _iterload(paths, chunk):
        chunks += 1
        n_frames += int(piece.n_frames)
        n_atoms = int(piece.n_atoms)
        if piece.unitcell_vectors is None:
            boxes_present = False
    if n_atoms is None:
        raise SourceError(f"{Path(paths['trajectory']).name} contains no frames.")
    return {"n_frames": n_frames, "n_atoms": n_atoms, "chunks_read": chunks,
            "chunk_frames": int(chunk), "boxes_present": boxes_present,
            "loader": "mdtraj.iterload",
            "source_bytes": int(Path(paths["trajectory"]).stat().st_size)}


def collect_frames(paths, wanted, *, chunk=DEFAULT_CHUNK_FRAMES):
    """Pass two: stream again and keep ONLY the selected frames.

    Returns `({index: (positions_nm, box_nm or None)}, chunks_read)`. Peak memory is one chunk plus
    the selected frames, and the loop stops as soon as everything wanted has been seen.
    """
    wanted = set(int(i) for i in wanted)
    kept, offset, chunks = {}, 0, 0
    for piece in _iterload(paths, chunk):
        chunks += 1
        for local in range(int(piece.n_frames)):
            index = offset + local
            if index in wanted:
                box = (None if piece.unitcell_vectors is None
                       else np.array(piece.unitcell_vectors[local], dtype=float))
                kept[index] = (np.array(piece.xyz[local], dtype=float), box)
        offset += int(piece.n_frames)
        if len(kept) == len(wanted):
            break
    missing = sorted(wanted - set(kept))
    if missing:
        raise SourceError(
            f"the source trajectory ended before frame(s) {missing[:5]} could be read; it has "
            f"{offset} frame(s).")
    return kept, chunks


# --- identity --------------------------------------------------------------------------------------

def atom_identity(topology):
    """A per-index identity tuple for every atom, and the bond set as index pairs.

    Atom NAME alone is not identity. A protein is full of repeated names -- every residue has an
    N, a CA, a C, an O -- so a topology whose residues were reassigned, whose chains were split
    differently, or whose atoms were renumbered within a residue compares equal on names while
    describing a different molecule. The System's parameters are assigned per index, so that
    mismatch would scale the wrong atoms, silently, with every count still agreeing.
    """
    atoms = []
    for atom in topology.atoms():
        residue = atom.residue
        chain = residue.chain
        atoms.append((
            str(getattr(chain, "id", "") or ""),
            int(chain.index),
            int(residue.index),
            str(getattr(residue, "id", "") or ""),
            str(residue.name),
            str(atom.name),
            (atom.element.symbol if atom.element is not None else ""),
        ))
    bonds = sorted(tuple(sorted((int(a.index), int(b.index)))) for a, b in topology.bonds())
    return atoms, bonds


def validate_topologies(request, paths):
    """Source topology and prepared topology are the same atoms, in the same order.

    Compared on full per-index identity and on bond connectivity, not on names and counts.
    Coordinates are deliberately NOT compared: the source is expected to hold different
    configurations, which is the entire point of sampling from it.
    """
    from openmm.app import PDBFile

    reference = PDBFile(str(paths["topology"]))
    prepared = PDBFile(str(request.prepared_topology))
    source_atoms, source_bonds = atom_identity(reference.topology)
    prepared_atoms, prepared_bonds = atom_identity(prepared.topology)

    if len(source_atoms) != len(prepared_atoms):
        raise SourceError(
            f"{paths['topology']} has {len(source_atoms)} atoms but the prepared "
            f"{request.prepared_topology.name} has {len(prepared_atoms)}.")

    for index, (mine, theirs) in enumerate(zip(source_atoms, prepared_atoms)):
        if mine == theirs:
            continue
        differing = ", ".join(f"{ATOM_FIELDS[i]} {a!r} vs {b!r}"
                              for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
        raise SourceError(
            f"the source topology and {request.prepared_topology.name} disagree at atom index "
            f"{index}: {differing}.\n"
            f"  source:   {mine}\n"
            f"  prepared: {theirs}\n"
            f"  The System's parameters are assigned per index, so a topology that differs here "
            f"would scale the wrong atoms while every count still matched.")

    if source_bonds != prepared_bonds:
        only_source = [b for b in source_bonds if b not in set(prepared_bonds)]
        only_prepared = [b for b in prepared_bonds if b not in set(source_bonds)]
        first = (only_source or only_prepared or [None])[0]
        raise SourceError(
            f"the source topology and {request.prepared_topology.name} have {len(source_bonds)} "
            f"and {len(prepared_bonds)} bond(s) and they are not the same bonds; first difference "
            f"{first} ({len(only_source)} only in the source, {len(only_prepared)} only in the "
            f"prepared inputs).\n"
            f"  Same atoms in the same order is not enough: different connectivity is a different "
            f"molecule, and REST2 scaling is decided from bonded terms.")
    return reference


# --- the whole resolution, without keeping a frame ---------------------------------------------------

def resolve_source_ensemble(request, *, require_temperature=False):
    """Everything about the source decidable without retaining a single frame.

    Shared by a preflight and by the run itself, so the check and the run agree by construction
    rather than by two implementations happening to match.
    """
    paths = resolve_source_paths(request)
    reference = validate_topologies(request, paths)
    survey = survey_source(paths, chunk=request.chunk_frames)
    if survey["n_atoms"] != reference.topology.getNumAtoms():
        raise SourceError(
            f"the source trajectory has {survey['n_atoms']} atoms but "
            f"{Path(paths['topology']).name} has {reference.topology.getNumAtoms()}.")
    if not request.implicit and not survey["boxes_present"]:
        raise SourceError(
            f"{Path(paths['trajectory']).name} carries no periodic box vectors, but this is an "
            f"explicit-solvent system and every drawn configuration keeps the box of the frame it "
            f"came from. A DCD written by this repository's runtime has them.")

    times, timing = source_frame_timing(request, paths["trajectory"], survey["n_frames"])
    tau, tau_evidence, tau_route = source_tau(request, paths["trajectory"])
    if abs(tau - request.required_tau) > 1e-9:
        raise SourceError(
            f"the source trajectory was produced at tau = {tau} ({tau_evidence}), but this "
            f"{request.purpose} requires tau = {request.required_tau}.\n"
            f"  An ensemble equilibrated at a different tau is a different distribution; using it "
            f"would bias every configuration drawn from it, silently.")

    temperature, temperature_evidence = source_temperature(request, paths["trajectory"])
    if require_temperature:
        if temperature is None:
            raise SourceError(
                f"cannot establish the temperature of the source ensemble "
                f"({temperature_evidence}), and this {request.purpose} requires it to match the "
                f"state it refreshes. State it with {request.field('source_temperature_k')}.")
        if (request.required_temperature_k is not None
                and abs(temperature - request.required_temperature_k) > 1e-9):
            raise SourceError(
                f"the source ensemble is at {temperature} K ({temperature_evidence}) but this "
                f"{request.purpose} requires {request.required_temperature_k} K. A reservoir at a "
                f"different temperature is a different Boltzmann distribution.")

    selected, eligible = select_source_frames(request, times, timing)
    return {"paths": paths, "survey": survey, "times": times, "timing": timing,
            "tau": tau, "tau_evidence": tau_evidence, "tau_route": tau_route,
            "temperature_k": temperature, "temperature_evidence": temperature_evidence,
            "selected": selected, "eligible": eligible}


# --- materialising, once ------------------------------------------------------------------------------

def selection_identity(request, ensemble):
    """A digest of everything that decides WHICH configurations were taken.

    Prepared inputs whose identity differs from this are refused, not regenerated: regenerating
    would delete the configurations a finished path or a running ladder already started from.
    """
    payload = {
        "trajectory_configured": ensemble["paths"]["trajectory_configured"],
        "start_time_ps": request.start_time_ps,
        "end_time_ps": request.end_time_ps,
        "count": request.count,
        "allow_replacement": request.allow_replacement,
        "seed": request.seed,
        "purpose": request.purpose,
        "selection_purpose": list(request.selection_purpose),
        "tau": ensemble["tau"],
        "selected": [int(i) for i in ensemble["selected"]],
        "timing": {k: ensemble["timing"].get(k)
                   for k in ("route", "first_frame_time_ps", "frame_interval_ps")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_prepared_source(request, ensemble, directory, *, extra=None):
    """Materialise the selected frames ONCE, with a manifest that makes them checkable.

    After this the production source is never reopened, so the run survives it being archived or
    deleted. The prepared files are small, so unlike the source they MAY be hashed.
    """
    import mdtraj
    from openmm.app import PDBFile

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configurations = directory / "configurations.dcd"
    manifest_path = directory / "prepared_source.yaml"

    kept, chunks = collect_frames(ensemble["paths"], ensemble["selected"],
                                  chunk=request.chunk_frames)
    reference = PDBFile(str(ensemble["paths"]["topology"]))
    positions = np.array([kept[index][0] for index in ensemble["selected"]], dtype=np.float32)
    boxes = None
    if not request.implicit:
        boxes = [reduced_box_vectors(kept[index][1]) for index in ensemble["selected"]]

    topology = mdtraj.Topology.from_openmm(reference.topology)
    trajectory = mdtraj.Trajectory(positions, topology)
    if boxes is not None:
        trajectory.unitcell_vectors = np.array(boxes, dtype=float)
    trajectory.save_dcd(str(configurations))

    manifest = {
        "format": PREPARED_FORMAT,
        "purpose": request.purpose,
        "selection_identity_sha256": selection_identity(request, ensemble),
        "source": {
            "trajectory_configured": ensemble["paths"]["trajectory_configured"],
            "trajectory_resolved": str(ensemble["paths"]["trajectory"]),
            "topology": relative_to(request.project, ensemble["paths"]["topology"]),
            "topology_choice": ensemble["paths"]["topology_choice"],
            # Bounded observations only. The production trajectory is NEVER hashed here; MD-data
            # hashes it once at archival.
            "bytes": ensemble["survey"]["source_bytes"],
            "n_frames": ensemble["survey"]["n_frames"],
            "n_atoms": ensemble["survey"]["n_atoms"],
            "chunk_frames": ensemble["survey"]["chunk_frames"],
            "chunks_read_survey": ensemble["survey"]["chunks_read"],
            "chunks_read_collect": int(chunks),
            "loader": "mdtraj.iterload",
            "hashed": False,
            "note": "a production trajectory is surveyed in bounded chunks and never hashed",
        },
        "evidence": {
            "tau": ensemble["tau"],
            "tau_evidence": ensemble["tau_evidence"],
            "tau_route": ensemble["tau_route"],
            "tau_from_directory_name": False,
            "temperature_k": ensemble["temperature_k"],
            "temperature_evidence": ensemble["temperature_evidence"],
            "frame_timing": ensemble["timing"],
        },
        "selection": {
            "window_ps": [request.start_time_ps, request.end_time_ps],
            "window_is_inclusive": True,
            "eligible_frames": len(ensemble["eligible"]),
            "count": request.count,
            "allow_replacement": request.allow_replacement,
            "seed": request.seed,
            "selected_frames": [int(i) for i in ensemble["selected"]],
            "selected_times_ps": [float(ensemble["times"][i]) for i in ensemble["selected"]],
        },
        "configurations": {
            "file": configurations.name,
            "frames": int(len(ensemble["selected"])),
            "n_atoms": int(ensemble["survey"]["n_atoms"]),
            "has_box": bool(boxes is not None),
            "box_convention": ("exact recorded vectors, reduced for OpenMM"
                               if boxes is not None else "implicit solvent: no box"),
            "velocities": None,
            "velocity_note": ("a DCD carries no velocities; momenta are redrawn from the Maxwell "
                              "distribution at the common temperature with a recorded seed"),
        },
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    # The prepared files are small, so hashing them is cheap and makes tampering detectable.
    digest = {"configurations_sha256": sha256_file(configurations)}
    manifest["configurations"].update(digest)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def read_prepared_source(directory, *, expected_identity=None):
    """Read a prepared ensemble back, refusing one that disagrees with what was asked for."""
    directory = Path(directory)
    manifest_path = directory / "prepared_source.yaml"
    if not manifest_path.is_file():
        raise SourceError(f"{manifest_path} does not exist; nothing has been prepared here")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != PREPARED_FORMAT:
        raise SourceError(
            f"{manifest_path} is {manifest.get('format')!r}, not {PREPARED_FORMAT!r}")

    configurations = directory / manifest["configurations"]["file"]
    if not configurations.is_file():
        raise SourceError(f"{configurations} named by the manifest does not exist")
    recorded = manifest["configurations"].get("configurations_sha256")
    if recorded and sha256_file(configurations) != recorded:
        raise SourceError(
            f"{configurations.name} does not match the digest its manifest records. The prepared "
            f"configurations have changed since they were written; refusing rather than using "
            f"configurations whose identity cannot be established.")
    if expected_identity is not None and manifest.get("selection_identity_sha256") != \
            expected_identity:
        raise SourceError(
            f"{manifest_path.name} was prepared for a different request "
            f"({manifest.get('selection_identity_sha256')!r} on file, {expected_identity!r} now). "
            f"Refusing rather than silently re-preparing: regenerating would delete the "
            f"configurations already drawn from.")
    return manifest


def read_prepared_frames(directory, manifest, topology_path):
    """Every prepared configuration, as `(positions_nm, box_nm or None)` pairs.

    The prepared file holds a handful of frames by construction, so reading it whole is bounded --
    which is exactly why the selection is materialised in the first place.
    """
    import mdtraj

    directory = Path(directory)
    trajectory = mdtraj.load_dcd(str(directory / manifest["configurations"]["file"]),
                                 top=str(topology_path))
    expected = int(manifest["configurations"]["frames"])
    if int(trajectory.n_frames) != expected:
        raise SourceError(
            f"{manifest['configurations']['file']} holds {trajectory.n_frames} frame(s) but its "
            f"manifest records {expected}. A truncated prepared file is refused.")
    frames = []
    for index in range(int(trajectory.n_frames)):
        box = (None if trajectory.unitcell_vectors is None
               else reduced_box_vectors(np.array(trajectory.unitcell_vectors[index], dtype=float)))
        frames.append((np.array(trajectory.xyz[index], dtype=float), box))
    return frames
