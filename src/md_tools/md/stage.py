"""One MD stage: minimisation, equilibration, or production.

Generated scripts call `stage_main` with the stage's resolved settings. Everything that decides
what is integrated -- the restraint, the barostat, the timestep, the seeds -- arrives in that
dict, so the script is readable and the behaviour is auditable from the script alone.

The command-line surface is Amber-like and identical for every stage:

    -p    topology PDB              (what `build-top` wrote as built.pdb)
    -s    serialised System XML     (what `build-top` wrote as built.xml)
    -c    input state from the previous stage (omit for the first stage)
    -x    output trajectory (DCD)
    -r    output final state XML -- the handoff this stage produces
    -chk  output checkpoint
    -log  the readable log, which carries this stage's machine record

Restart rules, which are the whole reason the checkpoint and the record both exist:

  * a stage whose log carries `status: completed` is NOT rerun. Rerunning would overwrite a
    finished handoff that later stages may already have consumed;
  * a stage with a checkpoint but no completion resumes from that checkpoint, but only after the
    checkpoint is shown to belong to THIS stage's configuration -- a checkpoint written under a
    different timestep, ensemble or system is refused rather than silently continued;
  * anything else starts from the beginning.

Completion is written only after the trajectory, the checkpoint and the final state are flushed,
closed, reopened and re-read. A log line is not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _beside_resolved_config(stage, name):
    """Resolve `name` beside `resolved.config`, which is where `build-md` copies definitions.

    Resolving against the working directory only happens to work when the script is launched from
    beside itself, and silently finds nothing -- or the wrong file -- otherwise. One rule, used by
    the collective-variable definition and by the umbrella restraints that name variables in it.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    beside = stage.get("resolved_config")
    return (Path(beside).parent if beside else Path(".")) / path

from ..openmm.platform_policy import (PlatformRequest, acceleration_record,
                                      resolve_platform_request)
from ..openmm.timestep import resolve_timestep_fs
from ..build.record import LogWriter, file_facts, openmm_platform_facts, read_record
from ..openmm.trajectory import describe_trajectory


#: Residue names that are solvent or a monatomic counter-ion, and therefore not solute. Positional
#: restraints act on the solute; restraining water would freeze the bath the equilibration exists
#: to relax.
SOLVENT_RESIDUES = frozenset({
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4", "T3P", "T4P", "OPC", "SPC",
    "NA", "NA+", "CL", "CL-", "K", "K+", "LI", "LI+", "CS", "CS+", "RB", "RB+",
    "BR", "BR-", "F", "F-", "I", "I-", "MG", "ZN", "CA",
})


def solute_atom_indices(topology) -> list[int]:
    """Every atom that is not solvent or a counter-ion, in topology order."""
    return [atom.index for atom in topology.atoms()
            if atom.residue.name.strip().upper() not in SOLVENT_RESIDUES]



class _AmberStreamReporter:
    """An OpenMM reporter writing AMBER NetCDF through MD-tools' own appending writer.

    WHY NOT mdtraj's REPORTER. It cannot append. Both of its backends refuse the mode outright --
    `NetCDFTrajectoryFile` and `DCDTrajectoryFile` each raise "mode must be one of ['r', 'w']" --
    so a resumed stage using one restarts its trajectory at frame zero and finishes SHORTER than
    an uninterrupted run, with no error, because writing a fresh file is an ordinary thing to do.

    `AmberTrajectoryWriter` already appends, is already what a REST2 ladder's per-state files are
    written with, and produces the same format cpptraj and mdtraj read. Using it here means one
    AMBER writer in the project rather than two that each do half the job.

    `atom_subset` is the solute stream's whole purpose, and is applied here rather than by the
    writer: the writer's business is one trajectory of N atoms, and which N is the caller's.
    """

    def __init__(self, path, interval, *, n_atoms, atom_subset=None, periodic, from_frame=0,
                 tau=0.0, temperature_k=0.0, application="cMD", system_sha256=None):
        from ..remd.amber_trajectory import AmberTrajectoryWriter

        self._interval = int(interval)
        self._subset = None if atom_subset is None else list(atom_subset)
        width = len(self._subset) if self._subset is not None else int(n_atoms)
        # THE STAGE'S OWN tau AND TEMPERATURE. These were hard-coded to 0.0 with a state index of
        # 0, so every conventional stage wrote a file whose header said it was REST2 state 0 at
        # tau = 0 whatever it actually ran at. A fixed-tau cMD ensemble is the one case where
        # that matters most: it is what AIS anneals away from, and `source_tau` exists precisely
        # to refuse taking the value from a directory name.
        if from_frame:
            self._writer = AmberTrajectoryWriter.open_existing(
                path, n_atoms=width, tau=float(tau), from_frame=int(from_frame))
        else:
            self._writer = AmberTrajectoryWriter(
                path, n_atoms=width, tau=float(tau), temperature_k=float(temperature_k),
                application=str(application), periodic=bool(periodic),
                system_sha256=system_sha256)

    def describeNextReport(self, simulation):                 # noqa: N802 - OpenMM's protocol
        steps = self._interval - simulation.currentStep % self._interval
        # positions yes, velocities/forces/energies no, and the box only when there is one.
        return (steps, True, False, False, False, self._writer.periodic)

    def report(self, simulation, state):
        from openmm import unit

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if self._subset is not None:
            positions = positions[self._subset]
        box = None
        if self._writer.periodic:
            # THE THREE BOX VECTORS, not their diagonal. `AmberTrajectoryWriter.append` takes
            # the (3, 3) matrix -- it derives the cell lengths as row norms and the cell angles
            # from the vectors themselves -- which is also what `Configuration.box` carries
            # everywhere else in the project. Passing the diagonal gave `norm(box, axis=1)` a
            # 1-D array and killed every explicit-solvent stage at its first reported frame;
            # for a triclinic cell it would additionally have thrown the shape away.
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        self._writer.append(positions, time_ps=state.getTime().value_in_unit(unit.picosecond),
                            box_nm=box)
        self._writer.sync()

    def __del__(self):                                        # pragma: no cover - interpreter exit
        try:
            self._writer.close()
        except Exception:                                     # noqa: BLE001 - best effort
            pass


class _EnergyDecompositionProbe:
    """A group-separated COPY of a System, read for energies and never integrated.

    WHY THIS EXISTS. OpenMM can only separate energies BY FORCE GROUP, and the two build routes
    disagree about them. ParmEd's `Structure.createSystem` -- the implicit route -- assigns bonds
    0, angles 1, torsions 2, nonbonded and GB 11, so an implicit run decomposes. OpenMM's
    `ForceField.createSystem` -- the explicit route -- leaves EVERY force in group 0, so an
    explicit run produced one column, holding the total potential energy, under a joined name
    that promised a breakdown. Nothing in the output distinguished the two, which is the worst
    property a diagnostic can have.

    WHY A COPY RATHER THAN REGROUPING THE RUN'S SYSTEM. A force group is part of the serialised
    System. Reassigning one would change `system_sha256`, invalidate the checkpoint fingerprint
    of every run in flight, and make already-registered datasets incomparable with new ones --
    all for a diagnostic. So this deserialises a SEPARATE System, groups that one, and never
    steps it. `built.xml`, the production Context and every digest taken from them are untouched.

    WHAT IT CAN AND CANNOT SEPARATE. Bonds, angles, torsions, the restraint and nonbonded direct
    space come apart cleanly, and PME reciprocal space splits out for free. Electrostatics from
    Lennard-Jones, and the 1-4 terms from either, do NOT: the 1-4 pairs are *exceptions inside*
    the single `NonbondedForce`, which evaluates charge and dispersion in one kernel. Amber can
    print `EELEC` beside `VDWAALS` because its energy routines are written term by term; getting
    there in OpenMM needs duplicated forces, which belongs in post-hoc analysis where nothing
    integrates -- not here.
    """

    def __init__(self, system):
        from openmm import NonbondedForce, XmlSerializer

        # A serialise/deserialise round trip is the one deep copy OpenMM guarantees.
        self._system = XmlSerializer.deserialize(XmlSerializer.serialize(system))
        self._context = None
        labels = {}
        for index, force in enumerate(self._system.getForces()):
            name = type(force).__name__
            if name == "CMMotionRemover":                 # carries no energy
                continue
            group = index + 1
            force.setForceGroup(group)
            labels[group] = name
            if isinstance(force, NonbondedForce):
                force.setReciprocalSpaceForceGroup(31)
                labels[31] = f"{name} [PME reciprocal]"
        self.groups = [(group, labels[group]) for group in sorted(labels)]

    def energies(self, live):
        """Per-group potential energy at the live Context's current configuration."""
        from openmm import Context, LangevinMiddleIntegrator, unit

        state = live.getState(getPositions=True)
        if self._context is None:
            # The run's own platform, so a CUDA run is not silently probed on the CPU. The
            # integrator is required to build a Context and is never stepped.
            self._context = Context(
                self._system,
                LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond,
                                         1.0 * unit.femtosecond),
                live.getPlatform())
        try:
            box = state.getPeriodicBoxVectors()
        except Exception:                                 # noqa: BLE001 - no box in this System
            box = None
        if box is not None:
            self._context.setPeriodicBoxVectors(*box)
        self._context.setPositions(state.getPositions())
        return [self._context.getState(getEnergy=True, groups={group})
                .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                for group, _ in self.groups]


class _EnergyComponentsReporter:
    """Per-force-group potential energy at the state table's cadence.

    THE ATTRIBUTION AMBER'S `mdout` GIVES AND A SINGLE TOTAL CANNOT. One `getState` per group per
    report. Where the System already carries meaningful groups they are read off it as it is and
    nothing is reassigned -- a force group is part of the serialised System, so changing one
    would change `system_sha256` and make every run in flight unresumable for the sake of a
    diagnostic.

    Groups are labelled by the forces IN them, so `NonbondedForce+CustomGBForce` says plainly
    that those two share a group and are not separable here. Naming them apart would be a nicer
    header over a number that is their sum.

    WHEN THE SYSTEM CARRIES NO GROUPS AT ALL -- every force in group 0, which is what OpenMM's
    `ForceField.createSystem` produces and therefore what every explicit-solvent run had -- one
    joined column holding the total is not a decomposition. That case is served by
    `_EnergyDecompositionProbe`, which groups a COPY and leaves the run's System alone.
    """

    def __init__(self, path, interval, system, *, append=False):
        from collections import defaultdict

        self._interval = int(interval)
        by_group = defaultdict(list)
        for force in system.getForces():
            by_group[int(force.getForceGroup())].append(type(force).__name__)
        # CMMotionRemover contributes no energy and would add a column of zeros.
        energetic = {group: names for group, names in by_group.items()
                     if set(names) - {"CMMotionRemover"}}
        self._probe = None
        if (len(energetic) == 1
                and len(set(next(iter(energetic.values()))) - {"CMMotionRemover"}) > 1):
            # One group holding several energy-carrying forces: nothing to attribute. Probe a
            # separated copy instead. A System with a single force needs no probe -- its one
            # column is already that force.
            self._probe = _EnergyDecompositionProbe(system)
            self._groups = list(self._probe.groups)
        else:
            self._groups = [(group, "+".join(sorted(set(names) - {"CMMotionRemover"})))
                            for group, names in sorted(by_group.items())
                            if set(names) - {"CMMotionRemover"}]
        self._handle = open(path, "a" if append else "w", encoding="utf-8")
        if not append:
            # Joined WITH the fixed columns rather than appended to them: a System whose only
            # forces carry no energy leaves no groups, and `"Time (ps)",` + "" ends the header in
            # a comma, which declares a column with no name.
            columns = ['#"Step"', '"Time (ps)"']
            columns += [f'"{label} (kJ/mole)"' for _, label in self._groups]
            self._handle.write(",".join(columns) + "\n")
            self._handle.flush()

    def describeNextReport(self, simulation):                 # noqa: N802 - OpenMM's protocol
        steps = self._interval - simulation.currentStep % self._interval
        # Nothing is requested here: each group's energy is fetched individually in `report`,
        # because one State cannot carry a per-group breakdown.
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        from openmm import unit

        context = simulation.context
        if self._probe is not None:
            values = [f"{value:.10g}" for value in self._probe.energies(context)]
        else:
            values = []
            for group, _ in self._groups:
                energy = context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
                values.append(f"{energy.value_in_unit(unit.kilojoule_per_mole):.10g}")
        time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        self._handle.write(f"{simulation.currentStep},{time_ps!r},{','.join(values)}\n")
        self._handle.flush()

    def __del__(self):                                        # pragma: no cover - interpreter exit
        try:
            self._handle.close()
        except Exception:                                     # noqa: BLE001 - best effort
            pass


def _system_census(topology, system):
    """What this run is made of: the counts Amber's `1. RESOURCE USE` section states.

    THE CHEAPEST SETUP CHECK THERE IS. A net charge that is not what was intended, a water count
    off by a factor, a box that never got set -- each is one line here and otherwise shows up as a
    puzzling energy much later. Amber prints `Sum of charges from parm topology file` and forces
    neutrality out loud; this had no equivalent anywhere in the human-readable output.

    Derived from the SERIALISED System and the topology that was actually loaded, never from the
    configuration that asked for them.
    """
    from collections import Counter

    import openmm
    from openmm import unit

    counts = Counter(residue.name for residue in topology.residues())
    charge = 0.0
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for particle in range(force.getNumParticles()):
                charge += force.getParticleParameters(particle)[0].value_in_unit(
                    unit.elementary_charge)
            break

    # 3N minus the constraints, minus 3 for a centre-of-mass remover if one is present. This is
    # the count behind every reported temperature, and Amber's own step-0 temperature looks wrong
    # until you know it.
    atoms = system.getNumParticles()
    dof = 3 * atoms - system.getNumConstraints()
    if any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces()):
        dof -= 3

    # A BOX ONLY IF THE SYSTEM IS ACTUALLY PERIODIC. `getDefaultPeriodicBoxVectors` is not a
    # question about whether there IS a box: an OpenMM System defaults to a 2 nm cube, so an
    # implicit-solvent System -- which has no box, no volume and no barostat -- would otherwise
    # report `2.000 x 2.000 x 2.000 nm` and a volume of 8 nm^3. Both invented, and stated with the
    # same confidence as a real measurement.
    #
    # Periodicity is decided by the nonbonded method, which is what actually makes the box
    # load-bearing. GBn2 uses NoCutoff or a non-periodic cutoff and correctly reports neither.
    periodic = False
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            periodic = force.getNonbondedMethod() in (
                openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.Ewald,
                openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME)
            break

    box = None
    volume = None
    if periodic:
        try:
            vectors = system.getDefaultPeriodicBoxVectors()
            lengths = [vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
            if all(length > 0 for length in lengths):
                box = " x ".join(f"{length:.3f}" for length in lengths) + " nm"
                volume = lengths[0] * lengths[1] * lengths[2]
        except Exception:                                 # noqa: BLE001 - no box in this System
            box = None

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "atoms": atoms,
        "residues": dict(counts),
        "residues_summary": (f"{sum(counts.values())} "
                             f"({', '.join(f'{name} {n}' for name, n in ordered[:6])}"
                             f"{', ...' if len(ordered) > 6 else ''})"),
        "net_charge_e": charge,
        "degrees_of_freedom": dof,
        "constraints": system.getNumConstraints(),
        "box_nm": box,
        "volume_nm3": volume,
    }


def _method_summary(system):
    """The resolved nonbonded and constraint treatment, as Amber's `Ewald parameters` block does.

    Read off the System rather than off the configuration, for the same reason the timestep is
    resolved against the serialised masses: by run time these are facts, not requests. A
    configuration claiming a 1.0 nm cutoff and a System carrying 0.8 nm differ, and only one of
    them integrates.
    """
    import openmm
    from openmm import unit

    summary = {}
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            method = {openmm.NonbondedForce.NoCutoff: "no cutoff",
                      openmm.NonbondedForce.CutoffNonPeriodic: "cutoff, non-periodic",
                      openmm.NonbondedForce.CutoffPeriodic: "cutoff, periodic",
                      openmm.NonbondedForce.Ewald: "Ewald",
                      openmm.NonbondedForce.PME: "PME",
                      openmm.NonbondedForce.LJPME: "LJPME"}.get(
                          force.getNonbondedMethod(), str(force.getNonbondedMethod()))
            line = f"{method}, cutoff {force.getCutoffDistance().value_in_unit(unit.nanometer)} nm"
            if force.getNonbondedMethod() in (openmm.NonbondedForce.PME,
                                              openmm.NonbondedForce.Ewald,
                                              openmm.NonbondedForce.LJPME):
                line += f", Ewald tolerance {force.getEwaldErrorTolerance():.3g}"
                # The grid is ZERO unless it was pinned explicitly, in which case OpenMM chooses
                # it per Context from the tolerance and the box. Reported only when the System
                # really carries it, so this never states a grid the run did not use.
                try:
                    alpha, nx, ny, nz = force.getPMEParameters()
                    alpha = alpha.value_in_unit(unit.nanometer ** -1)
                    if nx and ny and nz:
                        line += f", grid {nx}x{ny}x{nz}"
                    if alpha:
                        line += f", alpha {alpha:.5f}/nm"
                    else:
                        line += ", grid chosen per Context from the tolerance"
                except Exception:                         # noqa: BLE001 - not a PME-capable force
                    pass
            summary["nonbonded"] = line
            summary["dispersion correction"] = ("on" if force.getUseDispersionCorrection()
                                                else "off")
            summary["switching"] = (
                f"on, {force.getSwitchingDistance().value_in_unit(unit.nanometer)} nm"
                if force.getUseSwitchingFunction() else "off")
            summary["exceptions (1-4 pairs)"] = force.getNumExceptions()
            break
    summary["constraints"] = f"{system.getNumConstraints()} bond(s)"
    barostats = [type(f).__name__ for f in system.getForces()
                 if "Barostat" in type(f).__name__]
    summary["barostat in system"] = ", ".join(barostats) if barostats else "none"
    summary["forces"] = ", ".join(type(f).__name__ for f in system.getForces())
    return summary


def _state_table_statistics(path):
    """Mean and RMS fluctuation of each numeric column of a state table.

    AMBER PRINTS THESE AT THE FOOT OF `mdout` and this did not, so the first thing anybody wants
    after a run -- "what was the average temperature, and how much did the energy wander" -- was a
    separate computation every time, over a file they had to go and find.

    Read back from the CSV rather than accumulated during the run on purpose: what is summarised
    is then exactly what was written, including after a resume truncated rows the checkpoint did
    not vouch for. An accumulator would describe steps whose rows are no longer in the file.

    RMS fluctuation, not standard deviation of the mean: sqrt(<x^2> - <x>^2), which is the
    quantity Amber reports under that name.
    """
    import csv as _csv
    import math

    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(_csv.reader(handle))
    if len(rows) < 2:
        return []
    headers = [column.lstrip("#").strip('"') for column in rows[0]]
    summary = []
    for index, header in enumerate(headers):
        if header.startswith("Step") or header.startswith("Time"):
            continue
        values = []
        for row in rows[1:]:
            try:
                values.append(float(row[index]))
            except (ValueError, IndexError):
                continue
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = max(sum(value * value for value in values) / len(values) - mean * mean, 0.0)
        summary.append((header, mean, math.sqrt(variance), len(values)))
    return summary


def _committed_frames(path, done):
    """How many frames of `path` the resume may keep: what is on disk, capped by nothing here.

    The stage's own truncation has ALREADY cut every appendable stream back to the count the
    committed checkpoint vouches for, before this is reached. So the file length is the committed
    length by the time a reporter is opened on it, and reading it is not a second opinion.
    """
    from ..openmm.trajectory import count_frames

    path = Path(path)
    if not path.is_file():
        return 0
    try:
        return count_frames(path)
    except ValueError:
        # A file that exists and cannot be counted has no frames to keep. A reporter creates its
        # file when it is CONSTRUCTED, not when it first writes, so a stage interrupted before
        # its first frame leaves an empty one with no readable header -- and refusing to resume
        # over that would make an early interruption the one kind that cannot be continued.
        return 0


def _coordinate_reporter(path, interval, *, atom_subset=None, append=False, n_atoms=0,
                         periodic=False, from_frame=0, tau=0.0, temperature_k=0.0,
                         application="cMD", system_sha256=None):
    """A trajectory reporter whose FORMAT matches the name it was given.

    `-x whatever.dcd` must produce DCD and `-x whatever.nc` must produce AMBER NetCDF. Choosing
    one writer for both would put one format's bytes in the other's name, which is exactly what
    `check_trajectory_suffix` refuses a NAME for -- and worse coming from here, because the name
    was accepted first.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".dcd":
        # `-x something.dcd` KEEPS ITS OLD MEANING: one whole-system DCD, appendable across a
        # resume. That is what `-x` named before a stage had two streams, and a caller who writes
        # it is asking for the old thing by name.
        #
        # The subset is dropped here rather than honoured because no DCD writer available does
        # both: OpenMM's appends and cannot subset, mdtraj's subsets and cannot append (its
        # backends refuse "a" outright). A resumable subset stream needs both, so it must be
        # NetCDF -- which is why the names a stage chooses for itself are `.nc`.
        from openmm.app import DCDReporter

        return DCDReporter(str(path), int(interval), append=bool(append))
    return _AmberStreamReporter(path, interval, n_atoms=n_atoms, atom_subset=atom_subset,
                                periodic=periodic, from_frame=int(from_frame),
                                tau=tau, temperature_k=temperature_k, application=application,
                                system_sha256=system_sha256)


def check_trajectory_suffix(path: Path) -> None:
    """A stage writes DCD or AMBER NetCDF. Refuse a name that claims anything else.

    THE REASON THIS USED TO REFUSE `.nc` NO LONGER HOLDS, and the old text is worth keeping in
    view because it was correct when written:

        OpenMM has a native `DCDReporter` and no native NetCDF reporter, so DCD is what an
        ordinary stage can honestly produce.

    That was true while the stage used OpenMM's own reporter. It now writes through mdtraj's
    `NetCDFReporter`, which produces genuine AMBER NetCDF -- the same format cpptraj and mdtraj
    read without being told anything, and the only one that carries an atom SUBSET, which is what
    a solute-only stream is. So `.nc` is now honest, and is the default.

    What the guard still exists for is unchanged: writing one format's bytes into a name claiming
    another would be worse than refusing, because every tool downstream opens it by extension,
    fails, and blames the tool.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".dcd", ".nc"):
        return
    raise SystemExit(
        f"-x {path}: a stage writes DCD or AMBER NetCDF, and this name claims "
        f"{suffix or 'no'} format.\n"
        f"  Renaming a file does not change what is in it, and every reader downstream opens it "
        f"by extension.\n"
        f"  Use a .dcd or .nc name.")


def stage_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        # No abbreviation. argparse resolves a unique prefix by default, so `--traj` would become
        # `--trajectory` and a misspelling would RUN, with a setting nobody wrote.
        allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates (built.pdb)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System (built.xml)")
    parser.add_argument("-c", "--continue-from", default=None, metavar="XML",
                        help="final state written by the previous stage; omit for the first stage")
    parser.add_argument("-x", "--trajectory", default=None, metavar="DCD",
                        help="output trajectory")
    parser.add_argument("-r", "--restart", default=None, metavar="XML",
                        help="output final state, the handoff to the next stage")
    parser.add_argument("-chk", "--checkpoint", default=None, metavar="CHK",
                        help="output checkpoint, written periodically for resume")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="the provenance record (machine-readable). Defaults to <stage>.log")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable simulation output, Amber's mdout. A DIFFERENT file "
                             "from -log: this is what you tail while the stage runs. Defaults to "
                             "<stage>.out")
    parser.add_argument("--cpu", action="store_true",
                        help="run this invocation on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform. That setting can also select "
                             "CPU machine-wide; this flag is the per-run override, "
                             "and the record distinguishes the two")
    parser.add_argument("-odir", "--out-dir", default=None, metavar="DIR",
                        help="directory the stage's outputs go in, when they are not named "
                             "individually. The same flag `md-openmm md-run` takes, so a stage "
                             "run directly and one run through the command behave alike")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. An execution PLACEMENT option, not a platform "
                             "choice: it says which GPU, never whether to use one. Rejected with "
                             "--cpu, which has no device to place")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs and settings, then exit without integrating. "
                             "READ-ONLY: it creates nothing, not even the output directory")
    parser.add_argument("--resume", action="store_true",
                        help=argparse.SUPPRESS)  # refused by name below; cMD resumes on its own
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the stage's COMPLETE existing output inventory -- the "
                             "reports, the trajectory, the phase-space stream, the restart and "
                             "the checkpoint tree -- instead of refusing")
    _add_refused_flags(parser, "a cMD stage")
    return parser


def _add_refused_flags(parser, protocol_name: str) -> None:
    """Flags of OTHER protocols, accepted by the parser so the preflight can refuse them by name.

    Leaving them off refuses them too, as "unrecognized arguments" -- which names the flag and
    explains nothing, and puts the rule in argparse rather than in the shared preflight where the
    generated script and `md-run` are guaranteed to agree about it.
    """
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help=f"refused for {protocol_name}: -ng groups replicas of a REST2 "
                             f"ladder, and this has one process")
    parser.add_argument("-groupfile", "--groupfile", dest="groupfile", default=None,
                        metavar="FILE",
                        help=f"refused for {protocol_name}: a group file is one line per replica")
    parser.add_argument("-source-traj", "-src", "--source", dest="source_trajectory",
                        default=None, metavar="TRAJ",
                        help=f"refused for {protocol_name}: -source-traj is the equilibrium "
                             f"ensemble AIS draws its starting frames from")
    for flag, alias, destination in (("-s2", "--system2", "system2"),
                                     ("-p2", "--topology2", "topology2")):
        parser.add_argument(flag, alias, dest=destination, default=None, metavar="PATH",
                            help=f"refused for {protocol_name}: {flag} is the second end state "
                                 f"of an AIS transformation")


#: Fields a continuation may legitimately change, and which are therefore NOT part of the
#: fingerprint a checkpoint is matched against.
#:
#: `steps` is the whole point: asking for a longer run is an extension, and an extension must be
#: able to resume from the checkpoint the shorter run left. Everything else -- the ensemble, the
#: timestep, the temperature, the restraint, the seed, the System itself -- would make the
#: continuation a different simulation wearing the same file names, so all of it is fingerprinted.
EXTENDABLE_FIELDS = frozenset({"steps", "description"})

#: Keys of the stage dictionary that are PLUMBING rather than settings: they say where this
#: invocation found things, not what the simulation is. Excluded from the fingerprint and from the
#: record, because binding them in would make the same physics fingerprint differently depending
#: on how it was launched -- and `pending_parent` is not even JSON.
NON_SCIENTIFIC_STAGE_KEYS = frozenset({"resolved_config", "pending_parent"})


def _config_fingerprint(stage: dict[str, Any], system_sha: str, topology_sha: str) -> str:
    """What a checkpoint has to match before it may be resumed from.

    Deliberately includes the System and topology digests. A checkpoint carries positions and
    velocities for a particular particle set; resuming it against a System that was rebuilt is how
    a run continues with the right-looking numbers and the wrong molecule.
    """
    # `resolved.config` is bound in by DIGEST as well as through the stage derived from it. The
    # stage alone would miss an edit to another part of the file -- a different stage's length, a
    # reporting interval -- and the whole point of `resolved.config` being the single declaration
    # is that the run this checkpoint belongs to is the one that file describes.
    config_sha = None
    config_path = stage.get("resolved_config")
    if config_path and Path(config_path).is_file():
        config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    payload = {"stage": {k: v for k, v in sorted(stage.items())
                         if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS},
               "system_sha256": system_sha, "topology_sha256": topology_sha,
               "resolved_config_sha256": config_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_timestep_against_masses(timestep_fs: float, system, topology) -> None:
    """Refuse a large timestep on hydrogens that were never repartitioned.

    Kept as the narrow yes/no question. The full decision -- including resolving `auto` -- is
    `md_tools.openmm.timestep.resolve_timestep_fs`, which this delegates to so there is one
    implementation of the rule and one place the masses are read.
    """
    resolve_timestep_fs(float(timestep_fs), system, topology)


from .completion import verify_completed_stage


def _completion_gaps(previous: dict[str, Any], *, stage: dict[str, Any], name: str,
                     restart: Path, trajectory: Path) -> str:
    """Why a log claiming `status: completed` may not be believed. Empty string when it may.

    "Completed" is a field in a file, and a field in a file is not a finished run. The three ways
    it lies, in the order they actually happen:

      the log belongs to a DIFFERENT configuration -- a `resolved.config` edited since, or a log
      copied in from another tree -- so the outputs it describes are not the ones this invocation
      would produce;

      the outputs are gone. A log survives an interrupted `rm`, a partial copy, or a scratch
      filesystem that was cleaned; skipping the stage then leaves the NEXT stage to continue from
      a restart that does not exist;

      the log names counts it cannot produce -- an older schema whose fields this build cannot
      compare -- in which case nothing has been verified and saying so is the honest answer.
    """
    recorded = previous.get("fingerprint")
    current = previous.get("stage")
    if recorded is None or not isinstance(current, dict):
        return ("it carries no configuration fingerprint, so nothing about it can be matched to "
                "this stage")
    wanted = {k: v for k, v in sorted(stage.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    theirs = {k: v for k, v in sorted(current.items())
              if k not in EXTENDABLE_FIELDS and k not in NON_SCIENTIFIC_STAGE_KEYS}
    differing = sorted(k for k in set(wanted) | set(theirs) if wanted.get(k) != theirs.get(k))
    if differing:
        return (f"it records a different configuration for this stage "
                f"({', '.join(differing)} differ)")

    # Only the outputs this stage actually produces. A minimisation integrates nothing and so
    # writes no trajectory; demanding one would make every completed `min` look damaged.
    required = [restart]
    if int(stage.get("steps") or 0) > 0 and int(stage.get("trajectory_interval_steps") or 0) > 0:
        required.append(trajectory)
    absent = [str(path) for path in required if not Path(path).exists()]
    if absent:
        return f"the output(s) it claims are missing: {', '.join(absent)}"
    return ""


def cross_stage_collision(plans) -> str | None:
    """Two stages of one chain writing the same path. Returns the complaint, or None.

    No single stage's own inventory can see this: each is internally consistent, and the clash is
    only visible when the chain is considered as a whole. The consequence is a chain that
    overwrites its own inputs, with the run continuing from whichever stage happened to go last.

    Separate from its caller so it can be exercised directly -- the generator gives every stage a
    distinct name, so this state cannot be produced from a valid project, and a test that had to
    corrupt a configuration to reach it would be testing the corruption. `md_tools.build.md`
    validates the whole chain at generation time and calls this on the plans it prepared there.
    """
    claimed: dict[str, tuple[str, str]] = {}
    for entry_name, prepared in plans.items():
        roles = prepared.inventory.roles if getattr(prepared, "inventory", None) else {}
        for role, path in roles.items():
            resolved = str(Path(path).resolve(strict=False))
            if resolved in claimed and claimed[resolved][0] != entry_name:
                other, other_role = claimed[resolved]
                return (f"stages {other} and {entry_name} both write {resolved} "
                        f"({other_role} and {role}). One would overwrite the other's output, and "
                        f"the chain would continue from whichever ran last.")
            claimed[resolved] = (entry_name, role)
    return None


def stage_main(stage: dict[str, Any], argv: list[str] | None = None, *, prepared=None) -> int:
    """Run one stage. `stage` is the resolved settings the generated script declares.

    `prepared` is a `StagePreflight` a caller has ALREADY validated, and re-planning here would
    both duplicate the work and open the possibility of the two plans differing. When it is None
    this function plans for itself, which is what a generated stage script does.
    """
    args = stage_parser(stage.get("description", "one MD stage")).parse_args(argv)

    name = stage["name"]
    topology_path = Path(args.topology)
    system_path = Path(args.system)
    if getattr(args, "resume", False):
        print("cMD workflow: --resume is not a cMD flag. An interrupted chain continues from "
              "each stage's committed checkpoint automatically; re-run the same command. Pass "
              "--overwrite to start over.", file=sys.stderr)
        return 2

    from ._stages import stage_artifact_name

    base = Path(args.out_dir) if args.out_dir else Path(".")
    # The segment, needed HERE and not only further down: the four caller-named artefacts default
    # by segment now, so a second in-place segment cannot overwrite the first one's output, log,
    # restart state or checkpoint. See `stage_artifact_name`.
    segment = int(stage.get("segment") or 1)
    # THE NAME A STAGE IS FILED UNDER, which is not always the name it runs under.
    #
    # `eq_nvt_posres` is filed as `eq_1`: the ensemble moved out of the filename so that a
    # renamed stage (NVT under implicit solvent, where the explicit chain has NPT) cannot make a
    # filename claim something false. `stage_plan` stamps the key, so the layout that writes
    # `run.sh` and the runtime that writes the files cannot disagree -- they did, and `run.sh`
    # chained `-c eq/eq_1.xml` against a stage that wrote `eq/eq_nvt_posres.xml`.
    #
    # Falls back to the stage name, which is what `min` and `cMD` resolve to anyway.
    key = str(stage.get("file_key") or name)
    log_path = Path(args.log) if args.log else base / stage_artifact_name(key, "log", segment)
    out_path = Path(args.output) if args.output else base / stage_artifact_name(key, "out", segment)
    # The `-o` / `-log` collision is checked by the shared preflight below, together with every
    # other output pair and with the inputs. A second comparison here was a second policy: it
    # compared only those two, missed `-x`, `-r` and `-chk`, and fired first -- so the message a
    # person saw depended on which of the two implementations happened to reach the case.
    # TWO coordinate streams, named for their CONTENT and their segment.
    #
    #   solute_prod<N>.nc   the solute alone, at `crd_printout_solute`
    #   whole_prod<N>.nc    every atom, at `crd_printout_whole` (0 = not written)
    #
    # They used to be one file at one interval, called `<stage>.dcd`, holding the WHOLE system --
    # while the interval that produced it was called `solute_printout`. On a solvated peptide
    # that is 1796 atoms where 22 were asked for: 2.1 GB where 26 MB was wanted. The names now
    # say which is which, and a whole-system trajectory has to be asked for.
    #
    # AMBER NetCDF rather than DCD because it is what carries an atom SUBSET honestly -- mdtraj's
    # reporter takes `atomSubset`, OpenMM's DCDReporter does not -- and because cpptraj and
    # mdtraj both open it without being told anything.
    # PRODUCTION owns `prod<N>`; every other stage is named for itself.
    #
    # Without that split every stage in the chain writes the same two filenames -- minimisation
    # and three equilibrations all landing on `solute_prod1.nc` -- and the last one to run is the
    # only one you keep, while the file still claims to be production.
    #
    # `<N>` is the segment, which advances when a run is extended in place.
    # PRODUCTION is decided by the stage NAME -- `cMD` and `umbrella` are what production IS --
    # while the stem every other stage uses is its FILING KEY, so the streams sit beside the
    # restart and log of the same stage rather than under a second spelling of it.
    from ._stages import PRODUCTION_STAGE_NAMES

    production = str(stage.get("name") or "") in PRODUCTION_STAGE_NAMES
    stem = f"prod{segment}" if production else (key or "stage")
    # BESIDE THE LOG, not beside `base`.
    #
    # `md-run` forwards each stage its explicit `-log`/`-o`/`-r` paths but NOT `-odir`, so `base`
    # is the working directory whenever the stage is reached that way. That was invisible while
    # every output was passed as a full path; the moment a stage names its own files, `base`
    # scattered them into wherever the command happened to be run from -- silently, because the
    # run still completed and still reported the steps it took.
    #
    # `log_path` is always supplied and always in the run's directory, so its parent is the one
    # place every entry point agrees on.
    outputs = log_path.parent
    traj_path = (Path(args.trajectory) if args.trajectory
                 else outputs / f"solute_{stem}.nc")
    whole_path = outputs / f"whole_{stem}.nc"
    # THE STATE TABLE, and it needs the same per-stage treatment as the trajectories. `mdout.csv`
    # for production -- which is the one anybody opens -- and `mdout_<stage>.csv` for the rest. A
    # single `mdout.csv` shared by every stage in a chain is not one table with five sections: the
    # second stage finds an output it did not write, and the run refuses before it starts.
    from ._stages import info_csv_name
    info_path = outputs / info_csv_name(key or "stage", segment)
    from ._stages import energy_components_name
    components_path = outputs / energy_components_name(key or "stage", segment)
    restart_path = (Path(args.restart) if args.restart
                    else base / stage_artifact_name(key, "xml", segment))
    # The checkpoint TREE follows this name (`<chk stem>.checkpoints`), so it moves with the key
    # rather than needing a rule of its own.
    chk_path = (Path(args.checkpoint) if args.checkpoint
                else base / stage_artifact_name(key, "chk", segment))

    for path, what in ((topology_path, "-p topology"), (system_path, "-s system")):
        if not path.is_file():
            print(f"{name}: {what} {path} does not exist", file=sys.stderr)
            return 2

    # PREFLIGHT, BEFORE THE FIRST FILESYSTEM MUTATION -- and before the completion check.
    #
    # This runs here, in the runtime, and not only in `md-openmm md-run`: the generated `min.py`
    # and every other stage script call this function directly, so a guard living in the outer
    # command would leave them open. A `.out` and a `.log` created before the machine
    # configuration has been read are a directory that reads as a started run.
    #
    # The "already completed" short-circuit used to sit ABOVE this and return 0 on the strength
    # of one field in a log file. That made a stale or foreign log the authority on whether this
    # stage's outputs exist and belong to this configuration: a `min.log` copied from another
    # tree, or left behind by a run whose `resolved.config` has since changed, ended the command
    # successfully without anything having been verified. It is checked below, after the identity
    # this preflight establishes.
    from ..run.preflight import PreflightError, preflight_stage

    try:
        checked = prepared if prepared is not None else preflight_stage(
            topology=topology_path, system=system_path, coordinates=args.continue_from,
            trajectory=traj_path, whole=whole_path, restart=restart_path, checkpoint=chk_path,
            segment=segment,
            output=out_path, log=log_path, cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None,
            protocol=f"stage {name}",
            pending_parent=stage.get("pending_parent"),
            number_of_groups=args.number_of_groups, groupfile=args.groupfile,
            source_trajectory=args.source_trajectory,
            system2=args.system2, topology2=args.topology2,
            # The System is deserialised once, inside the preflight, and the timestep is resolved
            # against ITS masses -- so 4 fs on hydrogens that were never repartitioned is refused
            # before the `.out` and the `.log` exist rather than after.
            timestep_fs=stage.get("timestep_fs"),
            ensemble=stage.get("ensemble"), tau=float(stage.get("tau") or 0.0),
            # The WHOLE stage, so the plan can build the System this run will integrate: the
            # fixed-tau scaling with its force audit, the torsion classification, the restraint and
            # the barostat. Every refusal those produce then happens before the first file exists
            # rather than after both reports are open.
            stage=dict(stage, name=name))
    except PreflightError as refusal:
        # No `{name}:` prefix here: `protocol=f"stage {name}"` is already inside the message, and
        # printing both produced "cMD: stage cMD: -c ... does not exist".
        print(f"{refusal}", file=sys.stderr)
        return 2

    # WHAT THIS INVOCATION INTENDS TO CONTINUE, validated here -- while the only thing that has
    # happened is reading. The committed CV prefix used to be checked deep inside the resume
    # branch, some two hundred lines after `LogWriter` had already replaced the prior run's
    # machine record and its `.out`, so a refusal overwrote the authoritative account of a run it
    # never started. Completion is read from a machine record; that is exactly why a rejected
    # attempt may not become one. The refusal goes to stderr, outside the protected tree.
    try:
        from ..run.continuation import ContinuationError, validate_stage_continuation

        validate_stage_continuation(
            Path(args.out_dir) if getattr(args, "out_dir", None) else log_path.parent,
            stage=stage, definition=getattr(checked, "cv_definition", None) or _stage_definition(
                stage, checked),
            overwrite=bool(getattr(args, "overwrite", False)))
    except ContinuationError as refusal:
        print(f"{name}: {refusal}", file=sys.stderr)
        return 2

    if getattr(args, "resume", False):
        # NOT accepted-and-inert any more. A flag that is taken and does nothing is a flag whose
        # absence and presence are indistinguishable, so a person who passes it believes they
        # asked for something. cMD's contract is that an interrupted stage continues by itself.
        print(f"stage {name}: --resume is not a cMD flag. An interrupted stage continues from "
              f"its committed checkpoint automatically -- that is the cMD contract, and "
              f"requiring a flag for it would mean a plain re-run silently discarded committed "
              f"work. Re-run the same command to continue, or pass --overwrite to start over. "
              f"(--resume remains a REST2 and AIS flag.)", file=sys.stderr)
        return 2

    if args.check:
        # READ-ONLY, and it returns HERE. `--check` used to create `-odir`, the `.out` and the
        # `.log`, validate, and write `status: checked` -- leaving behind exactly the directory
        # whose absence the caller was trying to confirm. Everything it reported is in the
        # preflight result, so there is nothing left to open a file for.
        from ..run.preflight import report_check

        return report_check(checked, what=f"stage {name}",
                            extra=[("ensemble", stage.get("ensemble") or "-"),
                                   ("steps", stage.get("steps") or 0)])

    # A previous `--overwrite` that did not reach the end left a marker. The directory then holds
    # part of one run's outputs and none of another's, and neither the completion check nor the
    # resume branch below could tell that from an ordinary interrupted stage.
    from ..run.overwrite import find_incomplete_replacement

    interrupted_replacement = find_incomplete_replacement(log_path.parent)
    if interrupted_replacement is not None and not args.overwrite:
        print(f"stage {name}: {log_path.parent} holds a marker from an --overwrite that did not "
              f"finish, so some of the previous run's outputs may still be present and some may "
              f"not. Nothing here can be trusted as either run's. Re-run with --overwrite to "
              f"replace it completely, or choose a different -odir.", file=sys.stderr)
        return 2

    # -- already finished? -------------------------------------------------------------------
    # Now, with the inputs validated and this build's view of the stage established.
    if log_path.is_file() and not args.overwrite:
        try:
            previous = read_record(log_path)
        except Exception:
            previous = None
        if previous and previous.get("status") == "completed":
            # RECOMPUTED here, from the resolved scientific configuration and the digests of the
            # topology and System this invocation was given. The check this replaces read the
            # recorded fingerprint, asserted it was not None, and never compared it -- so a log
            # written against another System skipped the stage as long as its copy of the stage
            # dictionary matched.
            from ..build.record import file_facts as _file_facts

            current_print = _config_fingerprint(
                stage,
                _file_facts(system_path)["sha256"],
                _file_facts(topology_path)["sha256"])
            problems = verify_completed_stage(
                previous, stage=stage, name=name, restart=restart_path,
                trajectory=traj_path, log_path=log_path, fingerprint=current_print,
                particles=(checked.prepared_system.getNumParticles()
                           if getattr(checked, "prepared_system", None) is not None else None),
                checkpoints=chk_path.parent / f"{chk_path.stem}.checkpoints",
                inventory=getattr(checked, "inventory", None),
                streams=stage_streams(trajectory=traj_path,
                                      state_csv=info_path, whole=whole_path,
                                      energy_components=components_path,
                                      collective_variables=cv_csv_path(traj_path, stage)))
            if problems:
                detail = "".join(f"\n  - {problem}" for problem in problems)
                print(f"{name}: {log_path} says the stage completed, and it does not verify:"
                      f"{detail}\n"
                      f"  Refusing to treat it as done. Restore the outputs it names, or rerun "
                      f"with --overwrite to replace this run completely.", file=sys.stderr)
                return 2
            print(f"{name}: already completed and verified ({log_path}); not rerunning. "
                  f"Delete {log_path} to force a rebuild.")
            return 0

    # Not a completed run of THIS configuration, and outputs are in the way. The complete
    # inventory, not `resolved.config` alone: a `-odir` holding half of a previous stage produces
    # a tree that is half one run and half another, with every file looking equally current.
    from ..openmm.checkpoint import CheckpointError as _CheckpointError
    from ..openmm.checkpoint import read_committed as _read_committed
    from ..run.preflight import check_existing_outputs

    # THE ONE cMD RESUME CONTRACT, stated here because this is where it is decided:
    #
    #   an INTERRUPTED stage -- one whose committed checkpoint is short of its step count --
    #   continues from that checkpoint, automatically, with every appendable stream truncated to
    #   the counts that generation vouches for;
    #
    #   a COMPLETED stage -- whose log verifies and whose outputs are present and match -- is
    #   skipped;
    #
    #   anything else, a directory holding half a run nobody claimed, is refused until
    #   `--overwrite` says to replace it.
    #
    # `--resume` is therefore NOT required, and is accepted only so the four protocols take the
    # same flags. Requiring it would mean an interrupted stage that is simply re-run silently
    # starts over, discarding committed work -- the failure the checkpoint exists to prevent.
    #
    # "Interrupted" is not "has a checkpoint": a stage that FINISHED commits a final generation
    # too, so the presence of one says nothing. The discriminator is whether that generation is
    # short of this stage's step count.
    checkpoint_tree = chk_path.parent / f"{chk_path.stem}.checkpoints"
    if args.overwrite:
        # `--overwrite` starts CLEANLY, over the COMPLETE owned inventory, in one transaction,
        # BEFORE any new output is opened.
        #
        # This used to remove the checkpoint tree and leave everything else to whichever reporter
        # opened its file in `"w"`. A stream the new run does not write was then never truncated
        # by anyone: turn phase-space reporting off and `--overwrite` on, and the previous
        # `.phase_space.nc` survives, still named by the inventory, still looking like an output
        # of the run that just finished.
        from ..run.overwrite import ReplacementError, replace_owned_inventory

        try:
            replaced = replace_owned_inventory(
                checked.inventory, where=f"stage {name}", directory=log_path.parent)
        except ReplacementError as refusal:
            print(f"{refusal}", file=sys.stderr)
            return 2
        if replaced:
            print(f"{name}: --overwrite replaced {len(replaced)} existing output(s): "
                  f"{', '.join(sorted(replaced))}")
        committed_now = None
    else:
        try:
            committed_now = _read_committed(checkpoint_tree)
        except _CheckpointError:
            committed_now = None
    interrupted = (committed_now is not None
                   and int(committed_now["state"].get("steps_done", 0))
                   < int(stage.get("steps") or 0))
    # `--resume` does NOT excuse existing outputs. Only a valid committed checkpoint short of the
    # step count does -- which is what "interrupted" means, and is a fact about the directory
    # rather than a claim on the command line. Letting the flag stand in for it would turn
    # `--resume` into a way past the collision check for a directory with no checkpoint at all.
    try:
        check_existing_outputs(checked.inventory, overwrite=bool(getattr(args, "overwrite", False)),
                               resume=interrupted, where=f"stage {name}")
    except PreflightError as refusal:
        print(f"{refusal}", file=sys.stderr)
        return 2

    from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter, CheckpointReporter

    from ..md._stages import (add_barostat, add_positional_restraint,
                                              count_barostats, derive_seed, resolve_platform,
                                              set_restraint, write_final_state)

    from ..build.simout import SimulationOutput

    out = SimulationOutput(out_path, title=f"stage {name}", log_path=log_path)
    out.heading("Inputs")
    out.field("topology", topology_path)
    out.field("system", system_path)
    out.field("continue from", args.continue_from or "(none -- first stage)")
    out.field("trajectory", traj_path)
    out.field("final state", restart_path)
    out.field("checkpoint", f"{chk_path.parent / (chk_path.stem + '.checkpoints')}")

    log = LogWriter(log_path, record_type=f"md-stage:{name}")
    log.update(simulation_output=str(out_path))
    log(f"md-openmm stage: {name}")
    log("=" * 68)
    log.heading("Inputs")
    log.field("topology", topology_path)
    log.field("system", system_path)
    log.field("continue from", args.continue_from or "(none -- first stage)")

    try:
        pdb = PDBFile(str(topology_path))
        # CONSUMED, not rebuilt. The preflight deserialised the pair, compared the counts,
        # selected the solute, checked a hot stage's -s against its saved-state record,
        # restrained it and added the barostat -- all before this function created anything.
        system = checked.prepared_system
        implicit = checked.implicit
        solute = list(checked.solute)
        seed = checked.seed

        # THE SYSTEM AND THE METHOD, BEFORE THE SETTINGS. Amber's `mdout` opens with a census of
        # the topology (`1. RESOURCE USE`) and a full echo of every resolved control variable
        # (`2. CONTROL DATA FOR THE RUN`), which is why an Amber run can be reconstructed from
        # its own output. This wrote neither: the facts existed in `resolved.config` and the
        # machine record, so a reader had to open three files to answer "what was the cutoff".
        # They are cheap to state and the `.log` keeps carrying them for machines.
        census = _system_census(pdb.topology, system)
        method = _method_summary(system)
        # BOTH READERS. `-o` and `-log` are two files for two readers: the person tailing the run
        # and the machine parsing the record. The census and the method belong in the `.out`,
        # because that is the file being held against Amber's `mdout`; they go into the record as
        # structured fields as well, because prose is not a database.
        for writer in (out, log):
            writer.heading("System")
            writer.field("atoms", census["atoms"])
            writer.field("residues", census["residues_summary"])
            writer.field("net charge", f"{census['net_charge_e']:+.3f} e")
            writer.field("degrees of freedom", census["degrees_of_freedom"])
            if census["box_nm"] is not None:
                writer.field("box",
                             f"{census['box_nm']}  volume {census['volume_nm3']:.3f} nm^3")
            writer.heading("Method")
            for key, value in method.items():
                writer.field(key, value)
        log.update(system_census=census, method_summary=method)

        log.heading("Resolved settings")
        for key in ("ensemble", "steps", "timestep_fs", "temperature_K", "pressure_bar",
                    "friction_per_ps", "barostat_interval_steps", "restraint_kcal_per_mol_A2",
                    "minimization_iterations", "trajectory_interval_steps",
                    "state_interval_steps", "checkpoint_interval_steps"):
            if stage.get(key) is not None:
                log.field(key, stage[key])
        # The numerical timestep is decided HERE, against the masses in the System that was just
        # loaded -- not upstream in `build-md`, which never opens built.xml and would have to
        # trust a configuration's claim about hydrogen mass repartitioning. Step counts stay
        # authoritative; a physical duration is only derived once this returns.
        timestep = checked.timestep
        timestep_fs = timestep["timestep_fs"]
        steps = int(stage.get("steps") or 0)
        log.field("timestep", f"{timestep_fs} fs (requested {timestep['requested']!r}, "
                              f"{timestep['basis']}; heaviest hydrogen "
                              f"{timestep['heaviest_hydrogen_amu']} amu)")
        log.field("derived time", f"{steps} steps x {timestep_fs} fs = "
                                  f"{steps * timestep_fs / 1000.0:g} ps "
                                  f"({steps * timestep_fs / 1e6:g} ns)")
        log.update(timestep=timestep)
        log.field("solvent", "implicit (no barostat possible)" if implicit else "explicit")

        # -- the Force layout, fixed before any state is loaded -----------------------------
        #
        # Order matters and is the same order the REST2 ladder uses: SCALE FIRST, on the bare
        # System, then restrain, then add the barostat. The scaler audits every force and refuses
        # one it cannot classify, and the restraint and barostat are stage machinery rather than
        # terms of the molecular Hamiltonian, so neither may be scaled. Scaling last would scale
        # them, and a fixed-tau walker would then construct a different System from the ladder
        # rung it is supposed to match.
        tau = float(stage.get("tau") or 0.0)
        excluded = list(checked.excluded_bonds)
        # THE SELECTIONS, RESOLVED, WITH THEIR COUNTS -- and only once `excluded` exists, which is
        # here. Amber prints `Mask :1-3 & !@H=; matches 10 atoms`, so a mistyped mask shows up in
        # the output rather than in a trajectory three days later. `restraint_kcal_per_mol_A2`
        # says how hard the restraint pulls; this says what it pulls on.
        for writer in (out, log):
            writer.heading("Selections")
            writer.field("solute", f"{len(solute)} atom(s) (restraint and REST2 region)")
            # EMPTY IS NOT THE SAME AS NONE FOUND. At tau = 0 nothing is scaled, so there is
            # nothing to exempt and the list is legitimately empty -- while the same system on a
            # REST2 ladder leaves two amide omegas unscaled. Saying "0: none" invites the reader
            # to conclude the classifier found no such bonds, which is a different claim.
            writer.field("unscaled central bonds",
                         (f"{len(excluded)}: " + ", ".join(f"{a}-{b}" for a, b in excluded))
                         if excluded else
                         "not applicable at tau = 0 (nothing is scaled, so nothing is exempted)"
                         if tau == 0.0 else "0 (none classified)")
        log.update(selections={"n_solute_atoms": len(solute),
                               "unscaled_central_bonds": [[int(a), int(b)] for a, b in excluded],
                               "unscaled_impropers": tau > 0.0})
        if tau > 0.0:
            log.field("tau", f"{tau}  (the saved scaled state -s was checked against; nothing "
                             f"scaled here. {len(excluded)} unscaled central bond(s) recorded)")

        # The MOLECULAR Hamiltonian, fingerprinted HERE, before any stage machinery is added.
        # The restraint (always present, at zero strength when unrestrained) and the barostat are
        # properties of how this stage is run, not terms of the energy the ensemble is defined
        # by. A reservoir fingerprint taken after them claims a CustomExternalForce the ladder
        # rung it refreshes does not have, and the probability-one transfer is then refused --
        # correctly, for the wrong reason.
        #
        # Computed now rather than holding a reference: `add_positional_restraint` mutates the
        # System in place, so a reference would be fingerprinted after the restraint anyway.
        hamiltonian_identity_record = checked.hamiltonian_identity
        restrained = checked.restrained

        # --- umbrella biases, added while the System is still mutable ---------------------------
        #
        # This has to happen HERE, before the Context exists, and the collective-variable
        # definition it resolves against is therefore loaded here too -- earlier than the
        # reporting block below, which reuses this object rather than reading the file again.
        #
        # One load, deliberately. The restraint names a CV and the reported series measures one;
        # if those came from two reads they could differ, and the run would bias one torsion
        # while reporting another with every column still looking correct. Sharing the object
        # makes that failure inexpressible rather than merely unlikely.
        cv_definition_shared = None
        umbrella_restraints = ()
        umbrella_file = stage.get("umbrella_file")
        if umbrella_file:
            from ..cv import load_cv_definition as _load_cv
            from ..md.torsion_restraints import FLAT_BOTTOM, HARMONIC, TorsionRestraint
            from ..umbrella import load_umbrella_definition

            cv_definition_shared = _load_cv(
                _beside_resolved_config(stage, (stage.get("collective_variables") or {})["file"]),
                topology=pdb.topology, particles=system.getNumParticles())
            umbrella_restraints = load_umbrella_definition(
                _beside_resolved_config(stage, umbrella_file), cv_definition_shared)

            # One force per FORM, not per restraint: every torsion in a CustomTorsionForce shares
            # its energy expression, so harmonic and flat-bottom windows cannot live in one force.
            by_form = {}
            for entry in umbrella_restraints:
                force = by_form.get(entry.form)
                if force is None:
                    force = by_form[entry.form] = TorsionRestraint(system, entry.form)
                # The force constant rides on the PER-TORSION `scale`, not on the global. The
                # energy is 0.5 * k_global * scale * dtheta^2, so with the global at 1.0 each
                # restraint carries its own strength and a window may mix them freely. The global
                # then means only "are the biases on", which is what releasing them needs.
                #
                # The alternative -- one global constant for the whole window -- would have made
                # a config with two different force_constants unbuildable, and it built anyway:
                # the mismatch was caught here, at run time, after minimisation and three
                # equilibration stages had already run.
                force.add_torsion(entry.atom_indices, entry.centre_deg,
                                  entry.half_width_deg or 0.0, scale=entry.force_constant)
            umbrella_forces = tuple(by_form.values())
        else:
            umbrella_forces = ()
        barostat_active = (not implicit) and stage.get("ensemble") == "NPT"

        # ONE platform decision, from `md_tools.openmm.platform_policy`, shared with REMD and AIS.
        # CUDA unless `--cpu` was written; a CUDA that cannot open a Context is an error here,
        # before minimisation, rather than a silent CPU run that finishes hours later.
        # The preflight already resolved this, and proved a Context can be created on it. Using
        # its result rather than resolving again is what keeps one platform policy in the package.
        acceleration = checked.acceleration
        integrator = LangevinMiddleIntegrator(
            float(stage["temperature_K"]) * unit.kelvin,
            float(stage["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(int(seed))
        simulation = Simulation(pdb.topology, system, integrator,
                                acceleration.platform, acceleration.properties)
        log.update(acceleration=checked.record())
        log.field("acceleration", f"{acceleration.name} "
                                  f"({checked.record()['requested_policy']})")
        set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))
        # The umbrella biases, onto the same live Context.
        #
        # On the RESUME path this is redundant, and deliberately kept:
        # `Context.setState` restores global parameters along with positions and
        # velocities, so a resumed window comes back already biased -- measured,
        # not assumed. Removing this call changes nothing today. It stays because
        # that guarantee belongs to OpenMM's State rather than to this code, and a
        # resume path that ever rebuilt a Context without setState would otherwise
        # continue unbiased, writing a series that merely looks like a broader
        # window.
        for _force in umbrella_forces:
            _force.set_strength(simulation, 1.0)

        system_sha = file_facts(system_path)["sha256"]
        topology_sha = file_facts(topology_path)["sha256"]
        fingerprint = _config_fingerprint(stage, system_sha, topology_sha)
        log.update(stage={k: v for k, v in stage.items() if k != "pending_parent"},
                   fingerprint=fingerprint, implicit=bool(implicit),
                   inputs={"topology": file_facts(topology_path),
                           "system": file_facts(system_path)},
                   derived={"production_ps": steps * timestep_fs / 1000.0,
                            "timestep_fs": timestep_fs, "steps": steps})

        # -- where do we start? -------------------------------------------------------------
        #
        # ONLY the committed pointer, never "the newest checkpoint on disk": the newest file is
        # exactly what a crash leaves behind, and choosing it is how a Context from step 3000 gets
        # paired with bookkeeping from step 2000.
        from ..openmm.checkpoint import CheckpointError, read_committed

        done = 0
        checkpoints = chk_path.parent / f"{chk_path.stem}.checkpoints"
        try:
            committed = read_committed(checkpoints)
        except CheckpointError as broken:
            raise SystemExit(str(broken)) from None
        if committed is not None:
            meta = committed["state"]
            if meta.get("fingerprint") != fingerprint:
                raise SystemExit(
                    f"the committed checkpoint under {checkpoints} was written under a different "
                    f"configuration for this stage (fingerprint mismatch). Resuming it would "
                    f"continue a run that was set up differently. Delete {checkpoints} to start "
                    f"this stage over.")
            try:
                simulation.loadCheckpoint(committed["checkpoint"])
            except Exception as failure:                  # noqa: BLE001 - reported below
                # An OpenMM binary checkpoint is platform-specific, and the exception says so in
                # a way that reads like an internal error. It is not: it means this stage ran on
                # one platform and is being continued on another, which is a fact about the two
                # commands rather than about the simulation.
                raise SystemExit(
                    f"the committed checkpoint cannot be loaded on the {acceleration.name} "
                    f"platform: {failure}\n"
                    f"  An OpenMM checkpoint is binary and platform-specific. Continue this "
                    f"stage on the platform it was written on, or delete {checkpoints} to start "
                    f"the stage over.") from None
            done = int(meta.get("steps_done", 0))
            # Cut every appendable stream back to what the checkpoint VOUCHES for. A trajectory
            # is flushed as it is written, so after a crash it is routinely longer than the
            # checkpoint describing it; keeping those extra frames would put the coordinates
            # permanently ahead of the step count and misattribute every later frame. Never
            # inferred from whichever file happens to be longest.
            # THE CV PREFIX, validated BEFORE anything is cut. A row count detects a short file;
            # it does not detect a committed row that was edited in place, and a continuation
            # that has already truncated cannot decide afterwards that it should have refused.
            _validate_cv_prefix(meta.get("cv_prefix"), stage=stage, trajectory=traj_path,
                                fingerprint=fingerprint)

            trimmed = _truncate_streams_to_committed(
                meta.get("streams") or {}, trajectory=traj_path,
                state_csv=info_path, log=log, whole=whole_path,
                energy_components=components_path,
                collective_variables=cv_csv_path(traj_path, stage))
            log.heading("Resume")
            log.field("from checkpoint", f"generation {committed['generation']} at step {done}")
            for name, (was, now) in sorted(trimmed.items()):
                log.field(f"truncated {name}", f"{was} -> {now} (committed)")
        elif args.continue_from:
            parent = Path(args.continue_from)
            if not parent.is_file():
                # Unreachable in practice, and kept as the backstop it is. The preflight above
                # refuses a missing `-c` outright unless the chain declared a `PendingParent`,
                # and `--check` -- the only mode that grants that -- returned before this
                # function opened a single file. Reaching here means a parent vanished between
                # the preflight and now, which is a race worth naming rather than dereferencing.
                raise SystemExit(f"-c {parent} existed at preflight and does not now: something "
                                 f"removed it while this stage was starting. Refusing to "
                                 f"continue from a state that is no longer there.")
            state = XmlSerializer.deserialize(parent.read_text(encoding="utf-8"))
            simulation.context.setState(state)
            set_restraint(simulation, float(stage.get("restraint_kcal_per_mol_A2") or 0.0))
            # The umbrella biases, onto the same live Context.
            #
            # On the RESUME path this is redundant, and deliberately kept:
            # `Context.setState` restores global parameters along with positions and
            # velocities, so a resumed window comes back already biased -- measured,
            # not assumed. Removing this call changes nothing today. It stays because
            # that guarantee belongs to OpenMM's State rather than to this code, and a
            # resume path that ever rebuilt a Context without setState would otherwise
            # continue unbiased, writing a series that merely looks like a broader
            # window.
            for _force in umbrella_forces:
                _force.set_strength(simulation, 1.0)
            # The parent state is recorded WITH ITS DIGEST, not just its name. Registration
            # re-hashes every recorded input and refuses a directory whose files no longer match
            # the records, so a parent that was rewritten after this stage consumed it is caught
            # rather than silently accepted as this run's ancestry.
            log.record.setdefault("inputs", {})["continued_from"] = file_facts(parent)
            log.field("started from", f"{parent}  "
                                      f"(sha256 {log.record['inputs']['continued_from']['sha256'][:12]}...)")
        else:
            simulation.context.setPositions(pdb.positions)
            log.field("started from", f"{topology_path} coordinates")

        log.field("platform", simulation.context.getPlatform().getName())

        # The barostat, counted on the SYSTEM THE CONTEXT WAS BUILT FROM. One is present in every
        # explicit stage, so the Force layout -- and therefore the checkpoint layout -- does not
        # change along the chain; what makes a stage NPT is its FREQUENCY, not its presence. A
        # frequency changed after the Context exists is invisible until reinitialisation, which is
        # the mistake this records against.
        from ..md._stages import active_barostat_count
        barostats = {
            "in_system": count_barostats(system),
            "active": active_barostat_count(simulation),
            "frequency_steps": (int(stage["barostat_interval_steps"])
                                if stage["ensemble"] == "NPT" and not implicit else 0),
        }
        barostats["interval_ps"] = barostats["frequency_steps"] * timestep_fs / 1000.0
        log.field("barostat", f"{barostats['in_system']} in system, {barostats['active']} active"
                              + (f", every {barostats['frequency_steps']} steps "
                                 f"({barostats['interval_ps']:g} ps)"
                                 if barostats["active"] else ""))
        log.update(platform=openmm_platform_facts(simulation.context), barostats=barostats)

        # -- do the work --------------------------------------------------------------------
        log.heading("Run")
        out.heading("Run")
        out.field("timestep", f"{timestep_fs} fs")
        out.field("steps", f"{steps} ({steps * timestep_fs / 1000.0:g} ps)")
        out.field("already done", f"{done} (resumed from checkpoint)" if done else "0")
        out.field("ensemble", stage.get("ensemble", "?"))
        out.field("platform", acceleration.name)
        iterations = int(stage.get("minimization_iterations") or 0)
        if iterations:
            log(f"  minimising, {iterations} iterations")
            out.field("minimisation", f"{iterations} iterations")
            simulation.minimizeEnergy(maxIterations=iterations)

        # Declared BEFORE the dynamics block, because the completion record below reads them and
        # a minimisation (`remaining == 0`) never enters that block at all. They stay None there,
        # which is the truthful answer: a stage with no dynamics writes no CV series.
        cv_series = cv_reporter = None

        remaining = steps - done
        if remaining > 0:
            if stage.get("trajectory_interval_steps"):
                # SOLUTE ONLY. `checked.solute` is the same set every other part of the run means
                # by "solute" -- the restraint, the REST2 scaling, the CV definitions -- so the
                # trajectory cannot disagree with them about which atoms those are.
                # `from_frame` is the committed count, not the file length: rows past the
                # checkpoint are uncommitted and are overwritten in place, exactly as a ladder's
                # state trajectories are.
                simulation.reporters.append(
                    _coordinate_reporter(
                        traj_path, int(stage["trajectory_interval_steps"]),
                        atom_subset=list(checked.solute) or None,
                        tau=float(stage.get("tau") or 0.0),
                        temperature_k=float(stage.get("temperature_K") or 0.0),
                        application=str(stage.get("name") or "cMD"),
                        n_atoms=system.getNumParticles(), periodic=not implicit,
                        append=bool(done) and _trajectory_holds_frames(traj_path),
                        from_frame=(_committed_frames(traj_path, done) if done else 0)))
            if stage.get("whole_interval_steps"):
                # EVERY atom, and only when asked for: on a solvated system this stream is two
                # orders of magnitude larger than the solute one.
                simulation.reporters.append(
                    _coordinate_reporter(
                        whole_path, int(stage["whole_interval_steps"]),
                        tau=float(stage.get("tau") or 0.0),
                        # Only when the Hamiltonian integrated IS -s: no positional restraint
                        # and no umbrella bias applied in memory (a stage never scales). Otherwise
                        # the digest would name a Hamiltonian these frames were never sampled
                        # from, and AIS records its source as asserted rather than verified.
                        system_sha256=(system_sha if (not checked.restrained
                                                      and not umbrella_forces)
                                       else None),
                        temperature_k=float(stage.get("temperature_K") or 0.0),
                        application=str(stage.get("name") or "cMD"),
                        n_atoms=system.getNumParticles(), periodic=not implicit,
                        append=bool(done) and _trajectory_holds_frames(whole_path),
                        from_frame=(_committed_frames(whole_path, done) if done else 0)))
            if stage.get("state_interval_steps"):
                simulation.reporters.append(
                    StateDataReporter(str(info_path),
                                      int(stage["state_interval_steps"]), step=True, time=True,
                                      potentialEnergy=True, kineticEnergy=True,
                                      totalEnergy=True, temperature=True,
                                      # NOT under implicit solvent, where there is no box. The
                                      # CSV asked for both unconditionally and OpenMM answered
                                      # with the nominal unit cell -- a 22-atom GBn2 run got
                                      # `Box Volume 8.0` and `Density 0.0299`, numbers describing
                                      # a box the system does not have. The readable `.out` beside
                                      # it has always suppressed them (`volume=not implicit`), so
                                      # the two files disagreed about the same run.
                                      volume=not implicit, density=not implicit,
                                      speed=True, append=done > 0))
                # The decomposition, at the SAME cadence, in its own file.
                simulation.reporters.append(
                    _EnergyComponentsReporter(components_path,
                                              int(stage["state_interval_steps"]),
                                              system, append=done > 0))
                # The same numbers, readable, in the .out. A separate reporter rather than a
                # post-hoc copy of the CSV: the point of the .out is that it can be tailed while
                # the run is going, and a file written at the end cannot be.
                out.heading("Progress")
                simulation.reporters.append(
                    StateDataReporter(out.handle, int(stage["state_interval_steps"]),
                                      step=True, time=True, potentialEnergy=True,
                                      kineticEnergy=True, totalEnergy=True, temperature=True,
                                      volume=not implicit, density=not implicit, speed=True,
                                      separator="  "))
            # -- collective variables ------------------------------------------------------
            #
            # Its own cadence, independent of the trajectory and the state CSV and possibly finer
            # than both -- that independence is the whole reason for a separate series. The
            # interval is required to divide the stage's step count exactly, so that step 0 and
            # the final step each appear once and the spacing is uniform; a partial final gap
            # would break every downstream time-series analysis silently.
            cv_block = stage.get("collective_variables") or {}
            if int(cv_block.get("interval_steps") or 0) > 0:
                from ..cv import CVSeries, load_cv_definition, observation_steps
                from .cv_report import CVReporter

                interval = int(cv_block["interval_steps"])
                observation_steps(steps, interval, where=f"stage {name}")
                cv_path = Path(cv_block["file"])
                if not cv_path.is_absolute():
                    # Beside `resolved.config`, which is the generated directory `build-md`
                    # copied the content-addressed definition into. Resolving against the working
                    # directory only happens to work when the script is launched from beside
                    # itself, and silently finds nothing -- or the wrong file -- otherwise.
                    beside = stage.get("resolved_config")
                    cv_path = (Path(beside).parent if beside else Path(".")) / cv_path
                # Reuse the object the umbrella restraints resolved against, when there is one.
                # See where it is loaded, above, for why this must not be a second read.
                definition = cv_definition_shared if cv_definition_shared is not None else \
                    load_cv_definition(cv_path, topology=pdb.topology,
                                       particles=system.getNumParticles())
                cv_series = CVSeries(
                    cv_csv_path(traj_path, stage), definition,
                    extra_columns=("step", "time_ps", "trajectory_frame_index"),
                    sidecar_extra={"stage": name, "interval_steps": interval,
                                   "fingerprint": fingerprint})
                # On a resume, append after what is already there. The truncation back to the
                # count the selected generation committed has ALREADY happened, above, in
                # `_truncate_streams_to_committed` -- which cuts every appendable stream together,
                # because they flush at different cadences and trimming one alone leaves the
                # others permanently ahead of the step count. So the rows present here are exactly
                # the committed ones, and re-deriving that number would be a second truncation
                # rule that could disagree with the first.
                # Rows AND the cost that produced them, from the committed prefix. The
                # truncation back to the committed count has already happened above; what the
                # prefix adds here is the cumulative counters, which a rows-only restore dropped.
                from ..cv.cost import CommittedPrefix

                committed_prefix = CommittedPrefix()
                if done:
                    existing = _count_stream("collective_variables",
                                             cv_csv_path(traj_path, stage)) or 0
                    prefix = (committed or {}).get("state", {}).get("cv_prefix") or {}
                    committed_prefix = CommittedPrefix.from_record(
                        {"rows": existing, "cost": prefix.get("cost")})
                cv_series.open(committed=committed_prefix)

                trajectory_interval = int(stage.get("trajectory_interval_steps") or 0)

                def _frame_for(step, _interval=trajectory_interval):
                    # DCD frames are written every `_interval` steps and the first lands at the
                    # first such step, not at 0. Empty (None) whenever no frame exists here --
                    # never 0 or -1, both of which are real frame indices.
                    if _interval <= 0 or step <= 0 or step % _interval:
                        return None
                    return step // _interval - 1

                cv_reporter = CVReporter(
                    cv_series, interval, periodic=not implicit, timestep_fs=timestep_fs,
                    frame_index_for_step=_frame_for)
                # STEP 0, written here because an OpenMM reporter cannot fire before the first
                # step. Only on a fresh start: on a resume, step 0 was written by the segment
                # that began the stage and is already in the file.
                if done == 0:
                    cv_reporter.observe(
                        simulation.context.getState(getPositions=True,
                                                    enforcePeriodicBox=not implicit), 0)
                simulation.reporters.append(cv_reporter)

            if hamiltonian_identity_record is not None:
                from ..md.phase_space import PhaseSpaceReporter

                # The reservoir consumer compares this fingerprint against the ladder rung it is
                # about to refresh, and a probability-one transfer is only justified when they
                # agree. It is the real identity record -- over the canonical serialised System,
                # the solute selection and the unscaled torsions -- taken before stage machinery.
                simulation.reporters.append(PhaseSpaceReporter(
                    str(whole_path.with_suffix(".phase_space.nc")),
                    int(stage["phase_space_interval_steps"]),
                    identity={"hamiltonian": hamiltonian_identity_record},
                    periodic=not implicit, timestep_fs=timestep_fs))
            if stage.get("checkpoint_interval_steps"):
                simulation.reporters.append(
                    _CheckpointWithFingerprint(
                        checkpoints, int(stage["checkpoint_interval_steps"]),
                        fingerprint=fingerprint,
                        identity=_checkpoint_identity(stage, name, seed, acceleration,
                                                      timestep_fs),
                        # A callable per stream, read at commit time: the counts a generation
                        # vouches for have to be what is on disk WHEN it commits, not what was
                        # there when the reporter was constructed.
                        streams={
                            name: (lambda key=name: _stream_counts(
                                trajectory=traj_path,
                                state_csv=info_path, whole=whole_path,
                                energy_components=components_path,
                                collective_variables=cv_csv_path(traj_path, stage)).get(key, 0))
                            for name in stage_streams(
                                trajectory=traj_path,
                                state_csv=info_path, whole=whole_path,
                                energy_components=components_path,
                                collective_variables=cv_csv_path(traj_path, stage))},
                        # A callable, read AT COMMIT TIME like the stream counts: the digest has
                        # to cover the rows that exist when the generation commits, not the ones
                        # that existed when the reporter was constructed.
                        cv_prefix=(lambda: _cv_prefix_record(cv_series))))
            if done == 0 and not args.continue_from and iterations == 0:
                simulation.context.setVelocitiesToTemperature(
                    float(stage["temperature_K"]) * unit.kelvin, int(seed))
            log(f"  integrating {remaining} steps")
            simulation.step(remaining)
        else:
            log("  no dynamics for this stage")

        # -- flush, then PROVE the outputs are there -----------------------------------------
        for reporter in list(simulation.reporters):
            close = getattr(reporter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        simulation.reporters.clear()

        write_final_state(simulation, restart_path)
        # The final commit goes through the same transaction as every periodic one, so the last
        # checkpoint of a stage is committed exactly as the intermediate ones are rather than by
        # a second, weaker code path that only ever runs at the end.
        from ..openmm.checkpoint import commit_generation

        commit_generation(
            checkpoints,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={"fingerprint": fingerprint, "steps_done": steps,
                   **_checkpoint_identity(stage, name, seed, acceleration, timestep_fs),
                   "streams": _stream_counts(
                       trajectory=traj_path, state_csv=info_path,
                       collective_variables=cv_csv_path(traj_path, stage)),
                   # THE CV PREFIX, on the final generation too. The comment above says this
                   # commit goes through the same transaction as every periodic one, and it did
                   # -- except for this field, which only the periodic path supplied. So the
                   # LAST committed generation of every CV-enabled stage, the one a later reader
                   # actually consults, vouched for a row count in `streams` while carrying no
                   # digest of those rows and no cumulative counters. A continuation from it
                   # would refuse for want of a committed prefix, and nothing protected the
                   # committed rows of a finished stage from being edited in place.
                   "cv_prefix": _cv_prefix_record(cv_series)})

        log.heading("Outputs")
        reread = XmlSerializer.deserialize(restart_path.read_text(encoding="utf-8"))
        if reread.getPositions(asNumpy=True).shape[0] != system.getNumParticles():
            raise SystemExit("the written final state does not match the System; refusing to "
                             "report completion")
        # The checkpoint is a committed GENERATION now, not one file: the pointer is the thing
        # that says which generation is current, so it is the pointer that is recorded and
        # hashed. Recording the binary alone would name a file whose meaning depends on a second
        # file nobody recorded.
        from ..openmm.checkpoint import POINTER_NAME

        pointer = checkpoints / POINTER_NAME
        outputs = {"final_state": file_facts(restart_path)}
        if pointer.is_file():
            outputs["checkpoint_pointer"] = file_facts(pointer)
            committed_now = read_committed(checkpoints)
            outputs["checkpoint"] = file_facts(Path(committed_now["checkpoint"]))
            log.field("checkpoint", f"{checkpoints} generation {committed_now['generation']}")
        log.field(restart_path.name, f"{restart_path}  (re-read OK)")
        if traj_path.is_file():
            outputs["trajectory"] = file_facts(traj_path)
            log.field(traj_path.name, traj_path)
        phase_space = whole_path.with_suffix(".phase_space.nc")
        if phase_space.is_file():
            outputs["phase_space"] = file_facts(phase_space)
            log.field(phase_space.name, f"{phase_space}  (positions, velocities and box)")
        # THE STATE CSV, which the record used to omit entirely. A file that is in no manifest is
        # a file no completion check can look at: it could be truncated, half-written or from
        # another run and the stage would still verify.
        state_csv = info_path
        if state_csv.is_file():
            outputs["state_csv"] = file_facts(state_csv)
            log.field(state_csv.name, f"{state_csv}  (state table)")
        cv_csv = cv_csv_path(traj_path, stage)
        if cv_csv.is_file():
            outputs["collective_variables"] = file_facts(cv_csv)
            log.field(cv_csv.name, f"{cv_csv}  (collective variables)")
            from ..run.preflight import cv_sidecar_path

            cv_sidecar = cv_sidecar_path(cv_csv)
            if cv_sidecar.is_file():
                outputs["collective_variables_definition"] = file_facts(cv_sidecar)
        log.update(outputs=outputs)
        # CV COST, in two scopes and recorded separately from any energy-evaluation counter. A
        # position-only torsion is not an energy evaluation, and folding it into that total would
        # corrupt the one number that says how expensive the Hamiltonian is.
        #
        # `cv_observations` counts configurations the set was evaluated on; `cv_evaluations`
        # counts the SCALAR values that produced -- for N named torsions those differ by N, and
        # the old single counter reported the first under the second's name.
        #
        # The accumulation is `CVSeries`' own: it restored the committed cumulative cost when it
        # reopened, so `cost()` already carries both scopes. This used to be re-derived here from
        # loose keys, which is how a resume could restore rows and drop counters.
        if cv_series is not None:
            cost = cv_series.cost()
            log.update(collective_variable_cost=cost)
            log.field("CV cost",
                      f"{cost['cumulative']['cv_evaluations']} scalar evaluation(s) over "
                      f"{cost['cumulative']['cv_observations']} observation(s), "
                      f"{cost['cumulative']['wall_seconds']:.3f} s cumulative "
                      f"({cost['segment']['cv_evaluations']} this segment)")
        log.complete()
        # AVERAGES AND RMS FLUCTUATIONS, as Amber prints at the foot of an `mdout`. Over the
        # rows the file actually holds, so a resumed run summarises what it kept.
        statistics = _state_table_statistics(info_path)
        if statistics:
            out.heading("Averages")
            out.field("over", f"{statistics[0][3]} report(s) in {info_path.name}")
            for header, mean, fluctuation, count in statistics:
                # A FLUCTUATION NEEDS TWO SAMPLES. Over one report sqrt(<x^2> - <x>^2) is
                # exactly zero, which reads as "this quantity did not move" when it means
                # "there was nothing to compare it against" -- and a stage shorter than
                # `state_interval_steps` produces exactly one row, so this was the common case
                # rather than the corner one.
                spread = (f"rms fluctuation {fluctuation:.6g}" if count > 1
                          else "rms fluctuation n/a (single sample)")
                out.field(header, f"mean {mean:.6g}   {spread}")

        log.heading("Summary")
        log(f"  {name}: {steps} steps completed, {steps * timestep_fs / 1000.0:g} ps")
        log("  status: completed")

        out.heading("Outputs")
        out.field("final state", f"{restart_path}  (re-read OK)")
        out.field("checkpoint", checkpoints)
        if traj_path.is_file():
            out.field("trajectory", f"{traj_path}  ({describe_trajectory(traj_path)})")
        out.completed(f"{name}: {steps} steps completed, "
                      f"{steps * timestep_fs / 1000.0:g} ps")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        out.failed(f"{type(exc).__name__}: {exc}")
        print(f"{name}: {exc}", file=sys.stderr)
        return 1

    log.save()
    return 0


def _checkpoint_identity(stage, name, seed, acceleration, timestep_fs) -> dict[str, Any]:
    """What a resume has to agree about beyond the configuration fingerprint.

    The fingerprint already binds the resolved settings and both input digests. These are the
    facts about the RUN rather than about the configuration: which stage this is, which seed it
    drew, what it integrated with, and what it ran on. A checkpoint that matches the fingerprint
    but came from a different stage of the same workflow, or from a different platform, is a
    checkpoint that will load and be wrong -- and the platform case is the one that produces an
    exception rather than a silent error, so it is worth naming before it is loaded.
    """
    return {"stage": name, "seed": int(seed), "timestep_fs": float(timestep_fs),
            "ensemble": stage.get("ensemble"), "platform": acceleration.name,
            "precision": (acceleration.properties or {}).get("Precision")}


def _stage_definition(stage, checked):
    """This stage's CV definition, resolved read-only for the continuation check.

    Returns None when the stage reports no collective variables -- which is a legitimate
    configuration and not something to refuse.
    """
    block = (stage or {}).get("collective_variables") or {}
    path = block.get("file")
    if not path:
        return None
    try:
        from ..cv import load_cv_definition

        topology = getattr(getattr(checked, "loaded", None), "pdb", None)
        return load_cv_definition(path, topology=getattr(topology, "topology", None))
    except Exception:      # noqa: BLE001 - the ordinary path reports a bad definition far better
        return None


def _validate_cv_prefix(entry, *, stage, trajectory, fingerprint):
    """Refuse a continuation whose committed CV rows are not the rows that were committed.

    Silent when this stage reports no collective variables -- there is nothing to protect. A
    stage that DOES report them and finds no recorded prefix refuses with a compatibility
    message rather than guessing, because "which rows are durable" is exactly what the record
    exists to say.
    """
    if int((stage.get("collective_variables") or {}).get("interval_steps") or 0) <= 0:
        return
    from ..cv import prefix as cv_prefix
    from ..run.preflight import cv_sidecar_path

    path = cv_csv_path(Path(trajectory), stage)
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing, and this stage's checkpoint vouches for a committed "
            f"collective-variable prefix. Delete the checkpoint tree to start the stage over.")
    try:
        cv_prefix.validate(path, entry, sidecar=cv_sidecar_path(path))
    except cv_prefix.CVPrefixError as refusal:
        raise SystemExit(f"this stage cannot be continued: {refusal}") from None


def _cv_prefix_record(series):
    """The committed CV prefix for the checkpoint, or None when reporting is disabled.

    Read from the FILE rather than the writer's counters: the digest must describe the bytes on
    disk at the instant the generation commits, which is the only thing a resume can check.
    """
    if series is None:
        return None
    from ..cv import prefix as cv_prefix

    rows = int(series.rows_written)
    return cv_prefix.record(series.path, rows=rows, sidecar=series.sidecar,
                            definition=series.definition, cost=series.cost())


def _trajectory_holds_frames(path: Path) -> bool:
    """Whether `path` is a trajectory with at least one readable frame.

    `path.is_file()` is not the question. A stage interrupted BEFORE its first trajectory frame
    leaves a DCD that exists and contains no usable header -- the reporter creates the file when
    it is constructed, not when it first writes -- and `DCDReporter(append=True)` over that fails
    with "Cannot append to file with invalid DCD header", so a run interrupted early could not be
    continued at all.

    A file with no frames has nothing to preserve, so it is rewritten rather than appended to.
    That is not data loss: the committed checkpoint vouches for zero frames, and the steps that
    would have produced them are about to be repeated.
    """
    from ..openmm.trajectory import count_frames

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return count_frames(path) > 0
    except Exception:                                  # noqa: BLE001 - unreadable is "no frames"
        return False


def cv_csv_path(trajectory: Path, stage: dict[str, Any] | None = None) -> Path:
    """Where a stage's collective-variable series lives: `<key>.cv.csv`, beside the trajectory.

    Returned unconditionally: whether the file is WRITTEN is decided by the schedule, but where it
    would be is a property of the stage and the inventory needs it either way.
    """
    trajectory = Path(trajectory)
    # NAMED FOR THE STAGE, not for the trajectory file -- and for the name the stage is FILED
    # under, not the one it runs under.
    #
    # It used to be `<trajectory stem>.cv.csv`, which was the same thing while a stage had one
    # trajectory called `<stage>.dcd`. A stage now has TWO -- `solute_prod1.nc` and
    # `whole_prod1.nc` -- so deriving from the trajectory would put the series at
    # `solute_prod1.cv.csv` and make its name depend on which of the two happened to be passed in.
    # The collective variables belong to the stage, not to one of its streams.
    #
    # THE FILING KEY, as every other artefact of the stage now uses.
    #
    # A stage filed as `eq_1` writes `eq_1.xml`, `solute_eq_1.nc` and `mdout_eq_1.csv`, so a
    # series called `eq_nvt_posres.cv.csv` was the one stage-named file left in a keyed
    # directory -- and the CV series is exactly the file a reader has to pair with the trajectory
    # and the state table beside it. `min` and `cMD` are unaffected: their key IS their name.
    if stage and (stage.get("file_key") or stage.get("name")):
        key = stage.get("file_key") or stage["name"]
        return trajectory.parent / f"{key}.cv.csv"
    return trajectory.with_suffix(".cv.csv")


def stage_streams(*, trajectory: Path, state_csv: Path, whole: Path | None = None,
                  energy_components: Path | None = None,
                  collective_variables: Path | None = None) -> dict[str, Path]:
    """The appendable outputs a stage produces, by the name the checkpoint commits them as.

    FOUR now, not one. The commit recorded only the DCD, so a resume truncated the DCD and
    appended to a state CSV and a phase-space NetCDF that were still carrying rows past the
    checkpoint. The result is a directory whose streams describe different instants -- the
    trajectory correct, the others ahead of it -- and nothing in any of them says so. The CV
    series is appendable in exactly the same way and joins them here rather than growing a
    second truncation path of its own.
    """
    trajectory = Path(trajectory)
    # NAMED AFTER THE WHOLE STREAM, because that is what it holds: whole-system positions,
    # velocities and box, for a reservoir transfer. Deriving it from `trajectory` -- which since
    # the rename is the SOLUTE stream -- produced `solute_prod1.phase_space.nc`, a name promising
    # 22 atoms over a file carrying every one of them. `whole` defaults to the trajectory so a
    # caller that has only one path still gets the historical name rather than a crash.
    whole_path = Path(whole) if whole is not None else trajectory
    streams = {
        "trajectory": trajectory,
        "state_csv": Path(state_csv),
        "phase_space": whole_path.with_suffix(".phase_space.nc"),
    }
    if energy_components is not None:
        streams["energy_components"] = Path(energy_components)
    if collective_variables is not None:
        streams["collective_variables"] = Path(collective_variables)
    return streams


def _count_stream(name: str, path: Path) -> int | None:
    """How many records this stream holds right now, or None when it does not exist yet."""
    path = Path(path)
    if not path.is_file():
        return None
    if name in ("state_csv", "collective_variables", "energy_components"):
        # A CSV: one header line, then one row per report. The CV stream is counted the same way
        # for the same reason -- it is appended to, and a resume has to cut it back.
        with path.open(encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    try:
        from ..openmm.trajectory import count_frames

        return int(count_frames(path))
    except Exception:                                      # noqa: BLE001 - unreadable is not zero
        return None


def _stream_counts(*, trajectory: Path, state_csv: Path, whole: Path | None = None,
                   energy_components: Path | None = None,
                   collective_variables: Path | None = None) -> dict[str, int]:
    """How many records each appendable output holds RIGHT NOW, for the commit to vouch for.

    Read from the files rather than counted in memory: what a resume has to cut back to is what is
    on disk, and an in-memory counter that disagrees with the file is exactly the discrepancy the
    committed counts exist to resolve.
    """
    counts: dict[str, int] = {}
    for name, path in stage_streams(trajectory=trajectory, state_csv=state_csv,
                                    whole=whole, energy_components=energy_components,
                                    collective_variables=collective_variables).items():
        count = _count_stream(name, path)
        if count is not None:
            counts[name] = count
    return counts


def _truncate_csv_rows(path: Path, keep: int) -> None:
    """Keep the header and the first `keep` data rows. Written through a temporary.

    Rewritten and moved into place rather than truncated in situ, for the same reason the
    trajectory is: a truncation that is itself interrupted must not destroy the file it was
    repairing.
    """
    import os

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    staging = path.with_name(path.name + ".partial")
    staging.write_text("".join(lines[:keep + 1]), encoding="utf-8")
    os.replace(staging, path)


def _truncate_streams_to_committed(committed: dict[str, Any], *, trajectory: Path,
                                   state_csv: Path, log, whole: Path | None = None,
                                   energy_components: Path | None = None,
                                   collective_variables: Path | None = None
                                   ) -> dict[str, tuple[int, int]]:
    """Cut EVERY appendable stream back to the count the checkpoint committed.

    Returns `{name: (before, after)}` for whatever actually moved, so the log can say so. A stream
    already at or below its committed count is left alone: shorter than committed means the crash
    lost records the checkpoint believes exist, which is a different failure and is reported by
    the caller's own count checks rather than papered over here.

    All three streams, not just the trajectory. They are flushed independently and at different
    cadences, so after a crash they are routinely at three different lengths -- and truncating one
    of them leaves the other two permanently ahead of the step count, misattributing every later
    record in files that open and read perfectly.
    """
    from ..openmm.trajectory import truncate_frames

    moved: dict[str, tuple[int, int]] = {}
    for name, path in stage_streams(trajectory=trajectory, state_csv=state_csv,
                                    whole=whole, energy_components=energy_components,
                                    collective_variables=collective_variables).items():
        wanted = committed.get(name)
        if wanted is None:
            continue
        have = _count_stream(name, path)
        if have is None or have <= int(wanted):
            continue
        if name in ("state_csv", "collective_variables", "energy_components"):
            _truncate_csv_rows(path, int(wanted))
        else:
            truncate_frames(path, int(wanted))
        moved[name] = (have, int(wanted))
    return moved


class _CheckpointWithFingerprint:
    """A checkpoint reporter that COMMITS a generation instead of overwriting a pair.

    OpenMM's own CheckpointReporter writes only the binary state, so this once wrote the state and
    then the sidecar beside it:

        simulation.saveCheckpoint(path)
        Path(path + ".json").write_text(...)

    Two writes, in order, to two names that are only meaningful together. A crash between them --
    or a filesystem that reorders them, which is the ordinary case without an fsync -- leaves a
    NEW Context checkpoint beside an OLD `steps_done`, and the resume continues from step 3000
    while believing it is at 2000. It re-emits frames and rows that already exist and reports a
    stage length that was never run. Nothing fails; the trajectory is simply wrong.

    It is the same defect the AIS path checkpoint had, in the same shape, so it uses the same
    transaction rather than a second implementation of one:
    `md_tools.openmm.checkpoint.commit_generation` writes a new generation, fsyncs it, digests it,
    writes the sidecar, fsyncs the directory, and only then replaces the pointer.

    The committed state binds everything a resume has to agree about -- the configuration
    fingerprint, the step, the stage's identity, the seeds, the platform, and the committed row
    and frame counts of every appendable stream -- so recovery can cut those streams back to what
    the checkpoint vouches for instead of trusting whichever file happens to be longest.
    """

    def __init__(self, directory, interval: int, *, fingerprint: str,
                 identity=None, streams=None, cv_prefix=None) -> None:
        self._directory = Path(directory)
        self._interval = int(interval)
        self._fingerprint = fingerprint
        self._identity = dict(identity or {})
        #: name -> callable returning the committed count for that stream, read at commit time.
        self._streams = dict(streams or {})
        #: callable returning the committed CV prefix record, or None when CVs are disabled.
        self._cv_prefix = cv_prefix

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        from ..openmm.checkpoint import commit_generation

        commit_generation(
            self._directory,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={"fingerprint": self._fingerprint,
                   "steps_done": int(simulation.currentStep),
                   **self._identity,
                   "streams": {name: int(count()) for name, count in self._streams.items()},
                   # In the SAME generation transaction as the Context state, so the prefix a
                   # resume trusts and the coordinates it restores were committed together.
                   "cv_prefix": (self._cv_prefix() if self._cv_prefix else None)})


# ---------------------------------------------------------------------------------------------
# What a generated script calls.
#
# The generated file says which stage it is and where it lives. Everything else -- the resolved
# settings, the plan, the ordering -- comes from `resolved.config` beside it, which is therefore
# the SINGLE declaration of the workflow rather than one of two that can disagree.
# ---------------------------------------------------------------------------------------------

def resolved_config_beside(script: str | Path) -> Path:
    """The `resolved.config` next to a generated script.

    Located from the script's own path, never from the working directory: a generated directory
    must run correctly from anywhere, and `cd`-dependence is how a run silently picks up another
    project's configuration.
    """
    path = Path(script).resolve().parent / "resolved.config"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing. A generated script reads the workflow it belongs to from the "
            f"`resolved.config` beside it; without that file the script cannot know what it was "
            f"generated for. Regenerate the directory with `md-openmm build-md`.")
    return path


def load_generated_plan(script: str | Path) -> tuple[list[dict[str, Any]], Path]:
    """Resolve `resolved.config` strictly and return the ordered stage plan.

    Strictly: the same resolver that refused an unknown key at generation time runs again here, so
    a hand-edited `resolved.config` is refused at execution rather than half-applied.
    """
    from ..build.md import resolve_md_config, stage_plan

    path = resolved_config_beside(script)
    resolved = resolve_md_config(path)
    return stage_plan(resolved), path


def run_generated_stage(script: str | Path, name: str, argv: list[str] | None = None) -> int:
    """Run one named stage of the workflow this script belongs to.

    This is the whole body of a generated stage script:

        from md_tools.md import run_generated_stage
        raise SystemExit(run_generated_stage(__file__, "min"))
    """
    plan, config_path = load_generated_plan(script)
    for stage in plan:
        if stage["name"] == name:
            return stage_main(dict(stage, resolved_config=str(config_path)), argv)
    available = ", ".join(entry["name"] for entry in plan)
    raise SystemExit(
        f"{config_path} describes no stage named {name!r}. It has: {available}. The script and "
        f"the configuration beside it disagree, which means one of them was edited by hand; "
        f"regenerate the directory.")


#: `run_stage` is `stage_main` under the name the public API uses. A caller composing a run by
#: hand passes a resolved stage dictionary directly; a generated script goes through
#: `run_generated_stage` above, which builds that dictionary from `resolved.config`.
run_stage = stage_main
