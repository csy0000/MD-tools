"""Drive a REST2 / rREST2 ladder from a generated script.

The science here is not new and is not re-derived. The Hamiltonian scaling, the exchange
algorithm, the fixed state trajectories, the `rem.log` projection, the neighbouring-pair
acceptance report and every restart rule are the implementation already validated on this branch,
reached through the same executor entry point the previous generated projects used.

What this module replaces is the *packaging* around it. The old design copied a dozen runtime
modules into every generated directory and launched them through a separate `openmm-md`
executable. A generated directory now holds one small script; the runtime is imported from the
installed distribution, and the executor is called as a function.

Three things have to be prepared before the executor runs, and all three are derived from the
built system rather than carried in configuration that could go stale:

  solute.yaml    which atoms are solute, and which amide omega bonds must NOT be scaled. Derived
                 with the same `classify_omega_bonds` the builder uses, so the ladder scales
                 exactly what the build recorded.
  _protocol.py   one REST2Protocol describing the ladder.
  ladder.group   one group line per state, inputs only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any






def tau_ladder(n_states: int, tau_max: float) -> list[float]:
    """A linear ladder from 0 to tau_max inclusive. State 0 is always the unscaled Hamiltonian."""
    if n_states < 2:
        raise SystemExit(f"a ladder needs at least 2 states, not {n_states}")
    step = float(tau_max) / (n_states - 1)
    return [round(index * step, 6) for index in range(n_states)]


def write_solute_document(topology_path: Path, system_path: Path, out: Path, *,
                          route: str = "peptide") -> dict[str, Any]:
    """Derive and write solute.yaml from the built system.

# `templates_directory` and `_ensure_runtime_importable` lived here. They located the loose
# module directory and put it on `sys.path` so the executor's bare imports would resolve.
# There are no bare imports left -- those modules are `md_tools.remd` now -- so there is
# nothing to enable, and removing the insertion removes a real hazard with it: while that
# directory was on `sys.path`, `remd/statistics.py` shadowed the standard library's
# `statistics` for anything that imported it.

    Derived, never configured. The omega classification decides which torsions keep their physical
    barrier at the hot rungs; a stale hand-written list would change the Hamiltonian silently.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..openmm.yaml_io import write_yaml
    from ..openmm.system import classify_omega_bonds
    from ..openmm.builders import _solute_document
    from ..md.stage import solute_atom_indices

    topology = PDBFile(str(topology_path)).topology
    system = XmlSerializer.deserialize(Path(system_path).read_text(encoding="utf-8"))
    document = solute_document(topology, system, route=route)
    write_yaml(out, document)
    return document


def solute_document(topology, system, *, route: str = "peptide") -> dict[str, Any]:
    """The same derivation, WITHOUT writing anything.

    Split out because it carries a refusal -- an amide candidate that is neither ordinary nor
    proline-like -- and that refusal used to fire from inside the writer, after `-odir`,
    `solute.yaml`'s own directory and both logs already existed. The preflight calls this; the
    writer above calls it too, so there is one derivation and the file always holds what was
    validated.
    """
    from ..openmm.system import classify_omega_bonds
    from ..openmm.builders import _solute_document
    from ..md.stage import solute_atom_indices

    indices = solute_atom_indices(topology)
    omega = classify_omega_bonds(topology, indices, route=route, ligand_sdf=None)
    document = _solute_document(topology, indices, omega, route=route, system=system)
    ambiguous = (document.get("rest2") or {}).get("omega_ambiguous_candidates") or []
    if ambiguous:
        raise SystemExit(
            f"{len(ambiguous)} amide candidate(s) could not be classified as ordinary or "
            f"proline-like. Guessing either way silently changes the Hamiltonian, so the ladder "
            f"is refused rather than run. Candidates: {ambiguous[:3]}")
    return document


PROTOCOL_TEMPLATE = '''#!/usr/bin/env python
"""{protocol}: {n_states} states, tau 0.0 to {tau_max}, NVT at {temperature} K.

Written by the generated {protocol}.py at run time. Every state is thermostatted at the SAME
temperature and differs only by Hamiltonian: this is Hamiltonian scaling, not temperature REMD,
and an exchange never rescales velocities.

    tau ladder : {ladder}
    (1-tau)^2 on solute-solute terms, (1-tau) on solute-environment terms, omega left unscaled
"""
from md_tools.remd import REST2Protocol

protocol = REST2Protocol(
    tau={ladder},
    temperature_k={temperature},
    timestep_fs={timestep},
    exchange_interval_ps={exchange_ps},
    whole_output_interval_ps={whole_ps},
    solute_output_interval_ps={solute_ps},
    number_of_exchanges={exchanges},
    friction_per_ps={friction},
    equilibration_ps={equilibration_ps},
    random_seed={seed},
    hydrogen_mass_amu=None,
    platform={platform!r},
    precision=None,
)
'''


def reservoir_declaration_text(ladder: dict[str, Any], out: Path) -> str:
    """Turn the config's reservoir block into the declaration the runtime reads, AS TEXT.

    It returns text and writes nothing. It used to write `reservoir.yaml` itself, and it was
    called by EVERY rank -- so under `mpirun -n 8` eight processes truncated and rewrote one
    file while other ranks were reading it. It goes through the same rank-0-writes,
    everyone-verifies path as the other helpers now.

    `reservoir.path` names a PHASE-SPACE file: complete samples with positions, velocities and
    box, which is what the Boltzmann contract requires and what a plain trajectory cannot supply.
    Such a file is produced by a fixed-tau cMD run at the ladder's top rung
    (`dynamics.tau` with `dynamics.phase_space_printout`).

    The frame count and time window are read FROM the file rather than restated in configuration,
    so a declaration cannot claim a window the reservoir does not contain.
    """
    import yaml

    from ..md.phase_space import PhaseSpaceReader
    from ..remd.reservoir import DECLARATION_FORMAT

    block = ladder["reservoir"]
    source = Path(block["path"]).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(
            f"reservoir.path {source} does not exist. It must name a phase-space file holding "
            f"complete samples (positions, velocities and box) generated at the ladder's top "
            f"rung -- run a cMD with dynamics.tau set to the ladder's tau_max and "
            f"dynamics.phase_space_printout set.")
    reader = PhaseSpaceReader(str(source))
    try:
        frames = int(reader.n_frames)
        times = [float(value) for value in reader.times()]     # `times` is a method, not a property
    finally:
        reader.close()
    if frames < 1:
        raise SystemExit(f"{source} holds no frames; there is nothing to refresh from.")

    declaration = {
        "format": DECLARATION_FORMAT,
        "weighting": "boltzmann",
        "ensemble": "NVT",
        "prepared_directory": "reservoir",
        # `stored` installs the recorded momentum, which is what makes the drawn sample a sample
        # of the same distribution. `maxwell` redraws it and must be asked for deliberately.
        "velocity_policy": "stored" if block.get("velocities") == "inherit" else "maxwell",
        "refresh_interval_exchanges": int(block.get("refresh_interval_exchanges", 1)),
        "random_seed": int(ladder["dynamics"]["seed"]),
        "source": {
            "phase_space": os.path.relpath(source, out.parent),
            "start_time_ps": float(min(times)) if times else 0.0,
            "end_time_ps": float(max(times)) if times else 0.0,
            "frames": frames,
        },
    }
    return yaml.safe_dump(declaration, sort_keys=False)


def protocol_file_text(ladder: dict[str, Any]) -> str:
    """The protocol module a ladder is described by, as text.

    Exposed separately from `replica_main` so the portability properties can be checked without
    running anything: the body must name no path, so that a generated directory is movable and
    carries no machine-specific string.
    """
    dynamics = ladder["dynamics"]
    timestep = float(dynamics["timestep_fs"])
    states = int(ladder["n_states"])
    taus = tau_ladder(states, float(ladder["tau_max"]))
    exchange_ps = int(ladder["exchange_interval_steps"]) * timestep / 1000.0
    return PROTOCOL_TEMPLATE.format(
        protocol=ladder["protocol"], n_states=states, tau_max=ladder["tau_max"],
        temperature=float(dynamics["temperature_K"]), ladder=taus, timestep=timestep,
        exchange_ps=exchange_ps, whole_ps=exchange_ps, solute_ps=exchange_ps,
        exchanges=int(ladder["number_of_exchanges"]),
        friction=float(dynamics["friction_per_ps"]), equilibration_ps=0.0,
        seed=int(dynamics["seed"]), platform=dynamics.get("platform"))


def replica_parser(description: str = "one coordinated replica-exchange ladder"):
    """The ladder's flags. Exposed as a factory so the parser contract is testable on its own.

    `allow_abbrev=False` for the same reason as everywhere else: a misspelling argparse resolves
    runs, with a setting nobody wrote.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB")
    parser.add_argument("-s", "--system", required=True, metavar="XML")
    parser.add_argument("-c", "--continue-from", default=None, metavar="XML",
                        help="equilibrated state every replica starts from")
    parser.add_argument("-log", "--log", default=None, metavar="LOG")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="the coordinated run's readable .out; distinct from -log, which is "
                             "this ladder's own machine record")
    parser.add_argument("-x", "--trajectory", default=None, metavar="NC",
                        help="the coordinated run's analysis trajectory (NetCDF)")
    parser.add_argument("-r", "--restart", default=None, metavar="JSON",
                        help="the ladder's restart manifest")
    parser.add_argument("--groupfile", default=None, metavar="FILE",
                        help="an Amber-style group file to use instead of the one derived from "
                             "the ladder. Rarely needed for a homogeneous ladder")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="where the ladder's outputs are written (default: here)")
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help="how many replicas this launch coordinates. CHECKED against the "
                             "configured state count and the MPI world size; it never resizes "
                             "the ladder")
    parser.add_argument("--cpu", action="store_true",
                        help="run every replica on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform for this invocation")
    parser.add_argument("--route", default="peptide", choices=("peptide", "ligand"),
                        help="how the omega classifier reads the solute")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index for THIS rank. An execution placement, never a "
                             "platform choice. Under MPI the device is normally chosen by "
                             "machine.openmm.device_policy; this overrides it for one process")
    parser.add_argument("-chk", "--checkpoint", default=None, metavar="NC",
                        help="the ladder's checkpoint. Defaults to "
                             "<protocol>_checkpoint.nc in -odir")
    parser.add_argument("--check", action="store_true",
                        help="validate the launch, the inputs, the Force layout and the platform, "
                             "then exit. READ-ONLY: it creates nothing, not even -odir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extend", type=int, default=0, metavar="N")
    parser.add_argument("--extend-from", default=None, metavar="DIRECTORY")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the ladder's COMPLETE existing output inventory -- the "
                             "reports, the per-state trajectories, the restart manifest, "
                             "rem.log and the generated helpers -- instead of refusing")
    return parser


#: Marker line carrying the sha256 of the content a helper was generated from. Written into the
#: file itself rather than into a sidecar: a helper and a record of it in another file are two
#: things that can be separated, and the one that gets copied on its own is the helper.
HELPER_FINGERPRINT = "# md-tools-helper-sha256:"


def _yaml_text(document) -> str:
    import io

    from ..openmm.yaml_io import write_yaml

    buffer = io.StringIO()
    try:
        write_yaml(buffer, document)
        return buffer.getvalue()
    except Exception:                                      # noqa: BLE001 - writer wants a path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "solute.yaml"
            write_yaml(scratch, document)
            return scratch.read_text(encoding="utf-8")


def _group_file_text(protocol_name, states, ladder, args, protocol_file, solute_yaml) -> str:
    """One line per state. Every path is written RELATIVE TO THE GROUP FILE.

    That is how the parser reads them -- as Amber does, and as everyone who writes one by hand
    expects. The writer used to emit `-p` and `-s` exactly as they arrived on the command line,
    which are relative to the working directory, and the two conventions agreed only while the
    ladder happened to be launched from its own output directory. Run from anywhere else, every
    rank looked for `built.pdb` beside the group file and did not find it -- or, worse, found a
    different one.

    Relative rather than absolute so the directory stays movable, which is the same reason a
    generated script contains no absolute path.
    """
    import os

    directory = protocol_file.parent

    def relative(value):
        return os.path.relpath(Path(value).resolve(), directory)

    lines = [f"# {protocol_name}: {states} states, tau 0.0 to {ladder['tau_max']}.",
             "# One group per line, inputs only. Run-level outputs go on the executor call,",
             "# because they describe the coordinated run rather than one replica.",
             "# Paths are relative to THIS FILE, which is how they are read back.",
             ""]
    for index in range(states):
        parts = [f"-i {protocol_file.name}", f"-p {relative(args.topology)}",
                 f"-s {relative(args.system)}"]
        if args.continue_from:
            parts.append(f"-c {relative(args.continue_from)}")
        parts += [f"--solute {solute_yaml.name}", f"--group-index {index}"]
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _write_helper_if_compatible(destination: Path, text: str, *, force: bool = False) -> None:
    """Write a helper, or refuse an existing one that was generated from different content.

    Content-addressed rather than overwritten. A `_protocol.py` left by a ladder with a different
    state count or exchange interval is executable, runs perfectly, and simulates something else;
    silently replacing it is almost as bad, because a run that was already using it is then
    reading a different file than the one it started with.

    The file is written to a temporary and moved into place, so no rank ever reads a half-written
    helper -- which on a group file means a ladder with fewer rungs than it has states.
    """
    import hashlib
    import os

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stamped = f"{HELPER_FINGERPRINT}{digest}\n{text}"
    if destination.is_file():
        existing = destination.read_text(encoding="utf-8")
        if existing == stamped:
            return                                         # already exactly this
        recorded = None
        if existing.startswith(HELPER_FINGERPRINT):
            recorded = existing.splitlines()[0][len(HELPER_FINGERPRINT):].strip()
        if not force:
            raise SystemExit(
                f"{destination} already exists and was generated from different content "
                f"({'sha256 ' + recorded[:12] + '...' if recorded else 'no fingerprint'} against "
                f"{digest[:12]}...).\n"
                f"  A stale helper is executed or read as though it belonged to this ladder: a "
                f"`_protocol.py` from a run with a different state count runs perfectly and "
                f"simulates something else.\n"
                f"  Delete it to regenerate, or pass --force.")
    staging = destination.with_name(destination.name + ".partial")
    staging.write_text(stamped, encoding="utf-8")
    os.replace(staging, destination)


def replica_main(ladder: dict[str, Any], argv: list[str] | None = None) -> int:
    """Prepare the ladder's inputs and hand them to the validated executor."""
    import argparse
    import hashlib

    parser = replica_parser(
        f"{ladder['protocol']}: one coordinated replica-exchange ladder")
    args = parser.parse_args(argv)

    protocol_name = ladder["protocol"]

    # THE LAUNCH IS VALIDATED BEFORE `-odir` EXISTS. This runs here, in the shared runtime, and
    # not only in `md-openmm md-run`, because the generated `REST2.py` and `rREST2.py` call this
    # function directly -- a guard that lives in the outer command is a property of that command
    # rather than of the ladder.
    #
    # One process per thermodynamic state: the states the configuration declares, the `-ng` the
    # command line claims, and the MPI world the launcher created must be one number. Running 8
    # states in 4 processes is not a smaller ladder, it is a different Hamiltonian schedule
    # wearing the same output names.
    # EVERYTHING is validated before `-odir` exists -- not only the launch. The ladder writes
    # `solute.yaml`, `_protocol.py` and a group file before the driver ever resolves a platform,
    # so a malformed machine configuration used to be discovered with three files already on disk.
    from ..run.preflight import PreflightError, preflight_ladder

    out = Path(args.out_dir).resolve()
    try:
        checked = preflight_ladder(
            topology=args.topology, system=args.system, replicas=int(ladder["n_states"]),
            coordinates=args.continue_from, groupfile=args.groupfile,
            trajectory=args.trajectory, restart=args.restart,
            checkpoint=args.checkpoint,
            output=args.output or out / f"{protocol_name}.out",
            log=args.log or out / f"{protocol_name}.log",
            number_of_groups=args.number_of_groups, cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None,
            protocol=protocol_name,
            # The timestep against the masses in THIS System, and the Force classification and
            # scaled-System construction, all before `solute.yaml`, `_protocol.py` or the group
            # file exists. An unclassifiable force used to surface with three files on disk.
            timestep_fs=ladder["dynamics"]["timestep_fs"],
            route=args.route,
            tau=float(ladder["tau_max"]))
    except PreflightError as refusal:
        print(f"{protocol_name}: {refusal}", file=sys.stderr)
        return 2
    coordination = checked.coordination

    if args.check:
        # READ-ONLY, returning before `-odir` exists. `--check` was not even accepted here: it
        # was forwarded by `md-run` and died in argparse, so `md-openmm md-run --check` on a
        # REST2 input failed with "unrecognized arguments" rather than checking anything.
        from ..run.preflight import report_check

        return report_check(
            checked, what=protocol_name,
            extra=[("states", checked.replicas),
                   ("tau max", ladder["tau_max"]),
                   ("exchanges", ladder["number_of_exchanges"]),
                   ("forces", ", ".join(f"{n}" for _i, n in
                                        (checked.force_audit or {}).get("scaled", [])) or "-")])

    # PHASE-GUARDED from here on. Everything after the preflight is rank-local again -- a
    # directory that cannot be created on this node, a report that cannot be opened, a helper that
    # cannot be published -- and each is a place one rank fails alone while the others walk into
    # the next collective.
    with coordination.phase(f"{protocol_name}: creating the output directory"):
        out.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        # `platform` reaches the driver through the protocol file, and `from_flags` records that a
        # person chose the CPU rather than that CUDA was quietly unavailable.
        ladder = dict(ladder, dynamics=dict(ladder["dynamics"], platform="CPU"))

    # CONSUMED from the preflight, which resolved it against the masses in the System it had
    # already deserialised. This used to open `-p` and `-s` a second time and resolve the
    # timestep again -- the same authority, run twice, with the second run being the one the
    # protocol file was written from.
    timestep_record = checked.timestep
    ladder = dict(ladder, dynamics=dict(ladder["dynamics"],
                                        timestep_fs=timestep_record["timestep_fs"]))
    dynamics = ladder["dynamics"]
    timestep = float(dynamics["timestep_fs"])
    states = int(ladder["n_states"])
    taus = tau_ladder(states, float(ladder["tau_max"]))
    exchange_ps = int(ladder["exchange_interval_steps"]) * timestep / 1000.0

    # Prepared ONCE, by rank 0, with every other rank waiting behind a barrier before it reads
    # any of it. These three files are inputs to the executor, and N ranks writing them
    # concurrently is not a slow start -- it is a rank reading a half-written protocol module, or
    # a group file that lost lines to an interleaved write.
    rank, size = coordination.rank, coordination.size

    solute_yaml = out / "solute.yaml"
    protocol_file = out / "_protocol.py"
    # A group file the caller SUPPLIED is an input. Writing a default one beside it created a file
    # nothing would ever read -- and left it behind for a later run to pick up as though this
    # ladder had produced it.
    group_file = Path(args.groupfile) if args.groupfile else out / f"{protocol_name}.group"

    # THE HELPERS. Rank 0 alone writes them, atomically; every rank then verifies it sees the
    # same bytes.
    #
    # Every rank used to write all three. Under `mpirun -n 8` that is eight processes truncating
    # and rewriting one path at once: the file a rank subsequently READS could be another rank's
    # half-written copy, and the failure is intermittent and looks like a corrupt YAML file.
    #
    # They are content-addressed, so a helper left behind by an incompatible earlier run is
    # refused rather than reused or silently replaced. `_protocol.py` is the sharpest case: it is
    # executed, so a stale one from a ladder with a different state count or exchange interval
    # runs perfectly and simulates something else.
    from ..build.record import file_facts

    solute_text = _yaml_text(checked.notes["solute_document"])
    protocol_text = protocol_file_text(ladder)
    helpers = {solute_yaml: solute_text, protocol_file: protocol_text}
    if not args.groupfile:
        helpers[group_file] = _group_file_text(protocol_name, states, ladder, args,
                                               protocol_file, solute_yaml)
    reservoir_file = None
    if ladder.get("reservoir", {}).get("enabled"):
        # Validated HERE, by every rank, from the file itself -- and written by rank 0 alone.
        # `reservoir.yaml` is read by every rank a moment later, so a concurrent rewrite is a rank
        # parsing another rank's half-written YAML.
        reservoir_file = out / "reservoir.yaml"
        helpers[reservoir_file] = reservoir_declaration_text(ladder, out)

    if args.verify_only:
        from ..remd import executor as replica_executor

        # READ-ONLY. `--verify-only` inspects a run that already happened; it wrote solute.yaml,
        # `_protocol.py` and a group file first, so verifying a finished ladder MODIFIED it --
        # and a verification that changes what it verifies is not one. Everything it needs is
        # either already in the directory or in the preflight result.
        for destination in helpers:
            if not destination.is_file():
                print(f"{protocol_name}: --verify-only cannot run: {destination} does not exist, "
                      f"so this directory does not hold a ladder to verify.", file=sys.stderr)
                return 2
        code = int(replica_executor.main([
            "-x", str(args.trajectory or out / f"{protocol_name}.nc"),
            "--groupfile", str(group_file), "--verify-only"]) or 0)
        return code

    # Rank 0 prepares; the outcome is AGREED before anyone proceeds.
    #
    # `if rank == 0: ...` followed by a barrier is a deadlock waiting for a bad input: rank 0
    # raises inside the writer, exits, and ranks 1..N-1 wait at the barrier for a participant that
    # has already gone. The launcher then reports nothing and the job holds its GPUs until a wall
    # clock kills it.
    failure = None
    if rank == 0:
        try:
            for destination, text in helpers.items():
                _write_helper_if_compatible(destination, text,
                                            force=bool(args.force or args.overwrite))
        except BaseException as broken:                    # noqa: BLE001 - reported collectively
            failure = f"{type(broken).__name__}: {broken}"
    if coordination.size > 1:
        for message in coordination.allgather(failure):
            if message:
                coordination.fail(f"{protocol_name}: rank 0 could not prepare the shared "
                                  f"helpers: {message}")
    elif failure:
        print(f"{protocol_name}: {failure}", file=sys.stderr)
        return 2
    coordination.barrier()

    # EVERY rank verifies, after the barrier: rank 0 writing correctly and rank 5 reading a file
    # its node has not seen yet is a real failure on a shared filesystem, and it is silent.
    for destination, text in helpers.items():
        if not destination.is_file():
            coordination.fail(f"{destination} was not written by rank 0")

        actual = file_facts(destination)["sha256"]
        stamped = f"{HELPER_FINGERPRINT}{hashlib.sha256(text.encode()).hexdigest()}\n{text}"
        expected = hashlib.sha256(stamped.encode("utf-8")).hexdigest()
        if actual != expected:
            coordination.fail(
                f"{destination} does not hold what this ladder wrote (sha256 {actual[:12]}... "
                f"against {expected[:12]}...). Every rank must read the same helper; continuing "
                f"would run rungs configured differently from one another.")
    coordination.agree(hashlib.sha256(
        "".join(sorted(helpers.values())).encode("utf-8")).hexdigest(), what="the ladder helpers")

    executor_argv = [
        "-ng", str(states),
        "--groupfile", str(args.groupfile or group_file),
        "-o", str(args.output or out / f"{protocol_name}.out"),
        "-x", str(args.trajectory or out / f"{protocol_name}.nc"),
        "-r", str(args.restart or out / "restart.json"),
        "--checkpoint", str(args.checkpoint or out / f"{protocol_name}_checkpoint.nc"),
    ]
    if ladder.get("rem_log", True):
        executor_argv += ["--rem", str(out / "rem.log")]
    if reservoir_file is not None:
        executor_argv += ["--reservoir", str(reservoir_file)]
    if args.resume:
        executor_argv.append("--resume")
    if args.verify_only:
        executor_argv.append("--verify-only")
    if args.force:
        executor_argv.append("--force")
    if args.extend:
        executor_argv += ["--extend", str(args.extend)]
    if args.extend_from:
        executor_argv += ["--extend-from", args.extend_from]

    from ..build.record import LogWriter, file_facts
    from ..remd import executor as replica_executor

    # Every rank keeps its own log rather than racing for one path. A rank that failed to bind
    # its device is exactly what a multi-GPU ladder needs to be able to show, so the other ranks'
    # logs are kept beside rank 0's rather than discarded -- the same rule the executor already
    # applies to `-o`, applied through the same function so the two cannot drift.
    from .executor import report_path_for_rank

    log_path = Path(report_path_for_rank(
        str(Path(args.log) if args.log else out / f"{protocol_name}.log"), rank))
    with coordination.phase(f"{protocol_name}: opening the rank report"):
        log = LogWriter(log_path, record_type=f"md-replica:{protocol_name}", echo=False)
    log(f"md-openmm {protocol_name}")
    log("=" * 68)
    log.heading("Ladder")
    log.field("states", states)
    log.field("tau", taus)
    log.field("exchange every", f"{ladder['exchange_interval_steps']} steps = {exchange_ps:g} ps")
    log.field("attempts", ladder["number_of_exchanges"])
    log.field("temperature", f"{dynamics['temperature_K']} K (every state, NVT)")
    log.update(ladder=dict(ladder), tau=taus, exchange_interval_ps=exchange_ps,
               inputs={"topology": file_facts(Path(args.topology)),
                       "system": file_facts(Path(args.system))})

    # The validated result travels WITH the call. The executor would otherwise run its own
    # preflight (it is independently callable and must be safe alone), and the driver would
    # otherwise resolve a second platform after every file on disk already existed.
    from .executor import INTERRUPTED_STATUS

    with coordination.phase(f"{protocol_name}: running the ladder"):
        code = int(replica_executor.main(executor_argv, prepared=checked) or 0)

    # The executor owns the run and writes its own authoritative records. This log exists so that
    # every artefact this package produces carries the SAME machine record, and so registration
    # never has to read the executor's prose summary to decide whether a ladder finished.
    restart = out / "restart.json"
    if code == 0 and restart.is_file():
        outputs = {"restart_json": file_facts(restart, relative_to=out)}
        for name in sorted(out.glob("remd*.nc")):
            outputs[name.name] = file_facts(name, relative_to=out)
        for extra in (out / f"{protocol_name}.nc", out / f"{protocol_name}_checkpoint.nc",
                      out / "rem.log"):
            if extra.is_file():
                outputs[extra.name] = file_facts(extra, relative_to=out)
        log.update(outputs=outputs, state_trajectories=len(list(out.glob("remd*.nc"))))
        log.complete()
        log.heading("Summary")
        log(f"  {protocol_name}: {states} states, {ladder['number_of_exchanges']} exchanges")
        log("  status: completed")
    else:
        log.fail(f"the executor returned {code}")
    log.save()
    if code not in (0, INTERRUPTED_STATUS) and coordination.size > 1:
        # A non-zero return from ONE rank of a ladder is a hung job, not a failed one: this
        # process exits and the others wait at their next collective for a participant that has
        # already gone. An interrupted run is excluded deliberately -- that is a clean,
        # checkpointed stop that every rank reaches together.
        coordination.fail(f"{protocol_name} rank {coordination.rank}: the executor returned "
                          f"{code}", code=code)
    return code


# ---------------------------------------------------------------------------------------------
# What a generated ladder script calls.
# ---------------------------------------------------------------------------------------------

def ladder_from_resolved(resolved: dict[str, Any], protocol: str) -> dict[str, Any]:
    """The ladder description, derived from a resolved workflow configuration.

    One derivation, used by `build-md` when it writes the log and by the generated script when it
    runs -- so the two cannot describe different ladders.
    """
    return {
        "protocol": protocol,
        "solvent": resolved["solvent"],
        "n_states": resolved["rest2"]["number_of_replicas"],
        "tau_max": resolved["rest2"]["tau_max"],
        "exchange_interval_steps": resolved["rest2"]["exchange_interval_steps"],
        "number_of_exchanges": resolved["rest2"]["number_of_exchanges"],
        "state_trajectory": resolved["rest2"]["state_trajectory"],
        "rem_log": resolved["rest2"]["rem_log"],
        "neighbour_acceptance_report": resolved["rest2"]["neighbour_acceptance_report"],
        "reservoir": dict(resolved["reservoir"]),
        "dynamics": dict(resolved["dynamics"]),
    }


def run_generated_remd(script: str | Path, *, protocol: str,
                       argv: list[str] | None = None) -> int:
    """Run the ladder described by the `resolved.config` beside this script.

    The whole body of a generated REST2 or rREST2 file:

        from md_tools.remd import run_generated_remd
        raise SystemExit(run_generated_remd(__file__, protocol="REST2"))
    """
    from ..build.md import resolve_md_config
    from ..md.stage import resolved_config_beside

    config_path = resolved_config_beside(script)
    resolved = resolve_md_config(config_path)
    if resolved["protocol"] != protocol:
        raise SystemExit(
            f"{config_path} declares protocol {resolved['protocol']!r} but this script was "
            f"generated for {protocol!r}. The directory holds a script and a configuration that "
            f"describe different runs; regenerate it.")
    ladder = ladder_from_resolved(resolved, protocol)
    ladder["resolved_config"] = str(config_path)
    return replica_main(ladder, argv)


#: The public name for driving a ladder from an already-built description.
run_remd = replica_main
