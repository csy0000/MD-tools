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

  solute.yaml    which atoms are solute, and which central bonds' torsions must NOT be scaled.
                 Derived with the same `unscaled_torsions` the builder uses, so the ladder scales
                 exactly what the build recorded.
  _protocol.py   one REST2Protocol describing the ladder.
  ladder.group   one group line per state, inputs only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any






def tau_ladder(n_states: int, tau_max: float, *, tau_min: float = 0.0) -> list[float]:
    """A linear ladder from tau_min (default 0) to tau_max inclusive.

    With the default, state 0 is the unscaled Hamiltonian and every value is exactly what this
    function returned before `tau_min` existed: `0.0 + i*step` is `i*step` in floating point, so
    no recorded ladder moves by a digit. A ladder from `tau_min > 0` has NO physical state, which
    `build-top --rest2-scaler` records and says in words.
    """
    if n_states < 2:
        raise SystemExit(f"a ladder needs at least 2 states, not {n_states}")
    step = (float(tau_max) - float(tau_min)) / (n_states - 1)
    return [round(float(tau_min) + index * step, 6) for index in range(n_states)]


def write_solute_document(topology_path: Path, system_path: Path, out: Path, *,
                          route: str = "peptide", ligand_sdf=None) -> dict[str, Any]:
    """Derive and write solute.yaml from the built system.

# `templates_directory` and `_ensure_runtime_importable` lived here. They located the loose
# module directory and put it on `sys.path` so the executor's bare imports would resolve.
# There are no bare imports left -- those modules are `md_tools.remd` now -- so there is
# nothing to enable, and removing the insertion removes a real hazard with it: while that
# directory was on `sys.path`, `remd/statistics.py` shadowed the standard library's
# `statistics` for anything that imported it.

    Derived, never configured. The unscaled-torsion classification decides which torsions keep their physical
    barrier at the hot rungs; a stale hand-written list would change the Hamiltonian silently.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from ..openmm.yaml_io import write_yaml
    from ..openmm.builders import _solute_document
    from ..md.stage import solute_atom_indices

    topology = PDBFile(str(topology_path)).topology
    system = XmlSerializer.deserialize(Path(system_path).read_text(encoding="utf-8"))
    document = solute_document(topology, system, route=route, ligand_sdf=ligand_sdf)
    write_yaml(out, document)
    return document


def solute_document(topology, system, *, route: str = "peptide",
                    ligand_sdf=None) -> dict[str, Any]:
    """The same derivation, WITHOUT writing anything.

    Split out because it carries a refusal -- an amide candidate that is neither ordinary nor
    proline-like -- and that refusal used to fire from inside the writer, after `-odir`,
    `solute.yaml`'s own directory and both logs already existed. The preflight calls this; the
    writer above calls it too, so there is one derivation and the file always holds what was
    validated.
    """
    from ..openmm.system import UnclassifiedTorsionError, unscaled_torsions
    from ..openmm.builders import _solute_document
    from ..md.stage import solute_atom_indices

    indices = solute_atom_indices(topology)
    # The SDF, for a residue with no residue evidence. This was a hard-coded `None`, which left a
    # SMILES-built solute's single `UNL`/custom residue with nothing to be classified from.
    try:
        unscaled = unscaled_torsions(topology, indices, ligand_sdf=ligand_sdf)
    except UnclassifiedTorsionError as refusal:
        raise SystemExit(str(refusal)) from None
    return _solute_document(topology, indices, unscaled, route=route, system=system)


PROTOCOL_TEMPLATE = '''#!/usr/bin/env python
"""{protocol}: {n_states} states, tau 0.0 to {tau_max}, NVT at {temperature} K.

Written by the generated {protocol}.py at run time. Every state is thermostatted at the SAME
temperature and differs only by Hamiltonian: this is Hamiltonian scaling, not temperature REMD,
and an exchange never rescales velocities.

    tau ladder : {ladder}
    (1-tau)^2 on solute-solute terms, (1-tau) on solute-environment terms; amide omega,
    aromatic ring, double bond and improper torsions left unscaled
"""
from md_tools.remd import REST2Protocol

protocol = REST2Protocol(
    tau={ladder},
    temperature_k={temperature},
    timestep_fs={timestep},
    exchange_interval_ps={exchange_ps},
    whole_output_interval_ps={whole_ps},
    solute_output_interval_ps={solute_ps},
    checkpoint_interval_ps={checkpoint_ps},
    number_of_exchanges={exchanges},
    friction_per_ps={friction},
    equilibration_ps={equilibration_ps},
    random_seed={seed},
    hydrogen_mass_amu=None,
    cv_interval_steps={cv_interval_steps!r},
{per_tau_line}{umbrella_line}    platform={platform!r},
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


def _per_tau_line(ladder: dict[str, Any]) -> str:
    """The `per_tau_equilibration=[...]` line, or nothing at all when the setting is off.

    NOTHING, not `per_tau_equilibration=None`: `_protocol.py` is content-addressed, so one extra
    line in every ladder's helper would refuse every existing directory's `--resume` until
    `--overwrite`, for a setting none of them uses.
    """
    stages = ladder.get("per_tau_equilibration") or []
    if not stages:
        return ""
    return f"    per_tau_equilibration={[dict(stage) for stage in stages]!r},\n"


def _umbrella_line(ladder: dict[str, Any]) -> str:
    """The `umbrella_file=` line, or nothing at all when no restraints are declared.

    Nothing, not `umbrella_file=None`, for the reason `_per_tau_line` gives: `_protocol.py` is
    content-addressed, and one extra line in every ladder's helper would refuse every existing
    directory's `--resume`.
    """
    path = (ladder.get("umbrella") or {}).get("file")
    return f"    umbrella_file={str(path)!r},\n" if path else ""


def protocol_file_text(ladder: dict[str, Any]) -> str:
    """The protocol module a ladder is described by, as text.

    Exposed separately from `replica_main` so the portability properties can be checked without
    running anything: the body must name no path, so that a generated directory is movable and
    carries no machine-specific string.
    """
    dynamics = ladder["dynamics"]
    per_tau_line = _per_tau_line(ladder)
    timestep = float(dynamics["timestep_fs"])
    states = int(ladder["n_states"])
    taus = tau_ladder(states, float(ladder["tau_max"]))
    exchange_ps = int(ladder["exchange_interval_steps"]) * timestep / 1000.0
    # Steps in the configuration, picoseconds in the protocol, converted with the SAME expression
    # the exchange interval uses. This used to be hard-coded to 0.0, so `_equilibrate` -- which
    # relaxes each rung under its own scaled Hamiltonian and has done all along -- could not be
    # reached through `build-md` at all.
    equilibration_ps = int(ladder.get("equilibration_steps") or 0) * timestep / 1000.0

    # THE CONFIGURED OUTPUT AND CHECKPOINT INTERVALS. These three used to be hard-wired to
    # `exchange_ps`, which silently overrode whatever the configuration asked for -- a run that
    # requested 5 ps solute frames got 10 ps ones and nothing said so. A substitution that
    # changes the recorded data has to be impossible, not merely documented.
    # ABSENT means an OLD ladder description, written before the reporting block reached here.
    # Such a run used both streams at the exchange interval, so that is what it keeps: reading
    # absence as "no streams configured" would turn two working streams into one final frame each
    # and the run would still look successful. An EMPTY block is a different statement -- a
    # configuration that named no intervals -- and is honoured as written.
    reporting = ladder.get("reporting")
    if reporting is None:
        return PROTOCOL_TEMPLATE.format(
            protocol=ladder["protocol"], n_states=states, tau_max=ladder["tau_max"],
            temperature=float(dynamics["temperature_K"]), ladder=taus, timestep=timestep,
            exchange_ps=exchange_ps, whole_ps=exchange_ps, solute_ps=exchange_ps,
            checkpoint_ps=None,
            exchanges=int(ladder["number_of_exchanges"]),
            friction=float(dynamics["friction_per_ps"]), equilibration_ps=equilibration_ps,
            seed=int(dynamics["seed"]), platform=dynamics.get("platform"),
            cv_interval_steps=(int((ladder.get("collective_variables") or {}).get(
                "interval_steps") or 0) or None),
            per_tau_line=per_tau_line, umbrella_line=_umbrella_line(ladder))

    def interval_ps(key, what):
        """A reporting interval in ps, or None when it is disabled (0 steps)."""
        steps = int(reporting.get(key) or 0)
        if steps <= 0:
            return None
        return steps * timestep / 1000.0

    solute_ps = interval_ps("crd_printout_solute", "the solute output interval")
    # `crd_printout_whole`, NOT `info_printout`. The two were transposed in the rename, so a
    # ladder wrote its whole-system trajectory at the STATE TABLE's cadence: with
    # `info_printout: 2500` and `crd_printout_whole: 25000` every state got 1000 whole frames
    # where 100 were asked for -- ten times the disk, silently, and the configured interval
    # honoured nowhere.
    whole_ps = interval_ps("crd_printout_whole", "the whole-system output interval")
    checkpoint_ps = interval_ps("checkpoint_printout", "the checkpoint interval")

    # A CHECKPOINT MUST LAND ON AN EXCHANGE BOUNDARY. `ReplicaSchedule` defaults the checkpoint
    # interval to the exchange interval for that reason: a restart then resumes exactly where a
    # transition did, which is where the state-to-walker mapping and the exchange RNG are jointly
    # defined. Honouring a configured value must not quietly give that up, so a value that is not
    # a whole number of exchange intervals is REFUSED, naming both, rather than rounded to one.
    if checkpoint_ps is not None:
        exchange_steps = int(ladder["exchange_interval_steps"])
        checkpoint_steps = int(reporting.get("checkpoint_printout") or 0)
        if checkpoint_steps % exchange_steps:
            raise ValueError(
                f"reporting.checkpoint_printout = {checkpoint_steps} steps is not a whole number "
                f"of exchange intervals (rest2.exchange_interval_steps = {exchange_steps} "
                f"steps); it is {checkpoint_steps / exchange_steps:g} of them.\n"
                f"  A ladder checkpoints only at exchange boundaries, because that is where the "
                f"state-to-walker mapping and the exchange RNG are jointly defined and where a "
                f"restart can resume without inventing either. Refusing rather than rounding: "
                f"rounding would write checkpoints at an interval nobody asked for and nothing "
                f"would report the substitution.\n"
                f"  Choose a checkpoint_printout that is a multiple of {exchange_steps}: "
                f"{(checkpoint_steps // exchange_steps) * exchange_steps} or "
                f"{(checkpoint_steps // exchange_steps + 1) * exchange_steps}.")

    return PROTOCOL_TEMPLATE.format(
        protocol=ladder["protocol"], n_states=states, tau_max=ladder["tau_max"],
        temperature=float(dynamics["temperature_K"]), ladder=taus, timestep=timestep,
        exchange_ps=exchange_ps, whole_ps=whole_ps, solute_ps=solute_ps,
        checkpoint_ps=checkpoint_ps,
        exchanges=int(ladder["number_of_exchanges"]),
        friction=float(dynamics["friction_per_ps"]), equilibration_ps=equilibration_ps,
        seed=int(dynamics["seed"]), platform=dynamics.get("platform"),
        cv_interval_steps=(int((ladder.get("collective_variables") or {}).get(
            "interval_steps") or 0) or None),
        per_tau_line=per_tau_line, umbrella_line=_umbrella_line(ladder))


def replica_parser(description: str = "one coordinated replica-exchange ladder"):
    """The ladder's flags. Exposed as a factory so the parser contract is testable on its own.

    `allow_abbrev=False` for the same reason as everywhere else: a misspelling argparse resolves
    runs, with a setting nobody wrote.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB")
    # EXACTLY ONE OF `-s` AND `--groupfile`, and this is the parser where that matters most.
    #
    # A ladder's rungs are scaled and serialised at BUILD time (`remd<n>/build_state<n>.xml`), so
    # a group file names one System PER LINE and there is no single System for the launch to
    # carry. `required=True` made that launch impossible: argparse refused before
    # `remd.executor.resolve`'s grouped exemption could be reached, and `run.sh` -- the documented
    # way to run a ladder -- died on every rank with "the following arguments are required:
    # -s/--system". Relaxing it in `md-run` alone was not enough, because the generated
    # `REST2.py` and `rREST2.py` call `replica_main` DIRECTLY: a refusal that lives only in the
    # outer command is a property of that command rather than of the ladder.
    parser.add_argument("-s", "--system", default=None, metavar="XML",
                        help="the serialised System, for a homogeneous ladder. Omit it and pass "
                             "--groupfile when each rung has its own pre-scaled Hamiltonian")
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
                        help="how the unscaled-torsion classifier reads the solute")
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
    parser.add_argument("--force", action="store_true",
                        help="the executor's spelling of --overwrite, accepted here for symmetry "
                             "with the direct executor. The two are one policy: either replaces "
                             "the helpers AND the outputs")
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
                f"  Pass --overwrite to regenerate it, or delete it by hand.\n"
                f"  (--overwrite, NOT --force: `md-run` defines no --force, and this message is "
                f"printed on that path too. Deleting a file by hand is the more dangerous of the "
                f"two remedies, so it is not the one named first.)")
    staging = destination.with_name(destination.name + ".partial")
    staging.write_text(stamped, encoding="utf-8")
    os.replace(staging, destination)


def _cv_outputs(restart: Path, out: Path) -> dict[str, Any]:
    """The per-state CV series and sidecars, for the outer machine record.

    FROM THE COMPLETION MANIFEST, never from a glob. `restart.json` is written only after the
    driver has validated every series -- its digests, its step grid, its walker permutation --
    so the manifest is the one list of CV files this run vouches for. A `cv_state*.csv` glob over
    the directory would happily pick up a file left by an earlier run into the same place, or one
    belonging to a state this ladder does not have, and record it as provenance for this one.

    These files were absent here entirely. They were in `restart.json` and, being ordinary files
    under the run root, in the registry's own `SHA256SUMS`; but the outer `-log` record is what
    `md_tools.registry.discovery.check_lineage` indexes by digest to connect one stage's outputs
    to the next stage's inputs, so a CV series was invisible to every completion and lineage
    check that reads it.

    Each record carries what distinguishes one series from another that is otherwise identically
    shaped -- the state it belongs to and that state's tau -- and the digest of the definition
    they were measured under, so a reader can tell that two states' series are the same
    measurement without opening either. The CSV and its sidecar are separate roles, because they
    are separate artefacts: the sidecar is how the CSV is READ, and a reader holding one without
    the other has numbers whose units, wrapping and atom selection are unknown.
    """
    import json

    try:
        block = json.loads(restart.read_text(encoding="utf-8")).get("collective_variables")
    except (OSError, ValueError):
        # The manifest is the executor's artefact and it has already been written successfully
        # for this branch to run. If it cannot be read here, the run's own records still stand;
        # this log simply records no CV outputs rather than failing a completed run.
        return {}
    if not block:
        # `None` is explicit: this run reported no collective variables. Not an omission.
        return {}

    records: dict[str, Any] = {}
    for entry in block.get("series") or []:
        index = entry.get("state_index")
        shared = {"state_index": index, "tau": entry.get("tau"),
                  "definition_sha256": entry.get("definition_sha256")}
        for role, name, digest, size in (
                (f"state_cv_{index}", entry.get("csv"),
                 entry.get("csv_sha256"), entry.get("csv_bytes")),
                (f"state_cv_definition_{index}", entry.get("sidecar"),
                 entry.get("sidecar_sha256"), entry.get("sidecar_bytes"))):
            if not name:
                continue
            path = out / str(name)
            if not path.is_file():
                continue
            # The digest the manifest recorded, not a fresh one: the point of this record is to
            # say what the validated manifest vouches for. Re-hashing here would quietly paper
            # over a file that changed between the manifest landing and this log being written.
            records[role] = {"path": str(name), "bytes": size, "sha256": digest, **shared}
    return records


def replica_main(ladder: dict[str, Any], argv: list[str] | None = None) -> int:
    """Prepare the ladder's inputs and hand them to the validated executor."""
    import argparse
    import hashlib

    parser = replica_parser(
        f"{ladder['protocol']}: one coordinated replica-exchange ladder")
    args = parser.parse_args(argv)

    protocol_name = ladder["protocol"]

    # EXACTLY ONE OF `-s` AND `--groupfile`, refused BY NAME and READ-ONLY -- before `-odir`, a
    # helper, a log or the run state is touched.
    #
    # Here as well as in `md-run`, because this function IS the ladder's entry point: the
    # generated `REST2.py` and `rREST2.py` call it directly, so a refusal that lived only in the
    # outer command would leave them open. That is the same reason the launch check and the
    # per-tau resume refusal sit here rather than there.
    #
    # They are two answers to one question -- which System each replica integrates. `-s` is ONE
    # System for the whole launch; a group file names one PER LINE, which is what a ladder has now
    # that its rungs are scaled and serialised at build time. Neither is refused too: without a
    # System and without a group file, `group_file` below would be derived for a launch that
    # never said what to integrate.
    if args.system and args.groupfile:
        print(f"{protocol_name}: -s {args.system} and --groupfile {args.groupfile} were both "
              f"given, and they are two answers to one question: which System each replica "
              f"integrates. A group file names one System per line -- each rung's own pre-scaled "
              f"Hamiltonian -- so a single -s beside it would claim one Hamiltonian for every "
              f"rung. Pass one or the other, never both. Nothing was written.", file=sys.stderr)
        return 2
    if not args.system and not args.groupfile:
        print(f"{protocol_name}: neither -s nor --groupfile was given, so nothing says which "
              f"System each replica integrates. Pass -s for a homogeneous ladder, or --groupfile "
              f"whose lines name each rung's own pre-scaled System. Nothing was written.",
              file=sys.stderr)
        return 2

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
            # rREST2 writes `reservoir.yaml` into the run directory and every rank reads it a
            # moment later. It was in no inventory, so a launch that found a stale one from
            # another ladder would have drawn its probability-one transfers from it.
            reservoir=bool(ladder.get("reservoir", {}).get("enabled")),
            # The ladder description and the output directory, so the reservoir declaration is
            # BUILT AND VALIDATED HERE -- before `-odir` exists. It used to be built at helper
            # publication time, below, with the run directory already created.
            ladder=ladder, out_dir=out,
            tau=float(ladder["tau_max"]))
    except PreflightError as refusal:
        print(f"{protocol_name}: {refusal}", file=sys.stderr)
        return 2
    coordination = checked.coordination

    # A LADDER STOPPED DURING PER-TAU EQUILIBRATION has no checkpoint: it took no exchange step.
    # Refused here, by name, read-only -- before `-odir`, a helper, a log or the run state is
    # touched -- rather than by the executor finding no checkpoint NetCDF after this function has
    # already opened the rank report. Every rank reads the same file and reaches the same answer.
    if args.resume or (args.extend and not args.extend_from):
        from .rung_equilibration import INTERRUPTED_PHASE
        from .storage import read_run_state

        analysis = Path(args.trajectory) if args.trajectory else out / f"{protocol_name}.nc"
        recorded = read_run_state(analysis)
        if recorded and recorded.get("phase") == INTERRUPTED_PHASE:
            print(f"{protocol_name}: {analysis} was interrupted during per-tau equilibration "
                  f"(stage {recorded.get('stage')!r}), before the ladder took a single exchange "
                  f"step, so there is no checkpoint to resume from. Nothing is lost that cannot "
                  f"be redone exactly: that equilibration is deterministic from the recorded "
                  f"seeds. Rerun with --overwrite instead of "
                  f"{'--resume' if args.resume else '--extend'}. Nothing was written.",
                  file=sys.stderr)
            return 2

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
        if args.verify_only:
            # READ-ONLY, and that begins HERE. `--verify-only` created the directory it had been
            # asked to inspect, so verifying a run that never happened reported "does not hold a
            # ladder" while LEAVING BEHIND the empty directory that says one was started here.
            # The next `--resume` then has a directory to find. Absent is a verification result,
            # not a thing to fix.
            if not out.is_dir():
                raise SystemExit(
                    f"{protocol_name}: --verify-only cannot run: {out} does not exist, so there "
                    f"is no ladder here to verify. Nothing was created.")
        else:
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

    # WHAT `--overwrite` MEANS HERE: replace this ladder's outputs, once, before any new one is
    # opened. A continuation is not a replacement -- `--resume`, `--extend` and `--extend-from`
    # all read what is already there -- and `--verify-only` must not write at all.
    replacing = bool(args.force or args.overwrite) and not (
        args.resume or args.extend or args.extend_from or args.verify_only)

    # A previous replacement that did not reach the end left a marker, so the directory holds
    # part of one ladder's outputs and none of another's, with every file looking equally
    # current. Nothing downstream can tell that from an ordinary interrupted run: the per-state
    # trajectories that survived are exactly what a `--resume` would try to continue. Every rank
    # reads the same file and reaches the same answer, so this needs no collective.
    from ..run.overwrite import find_incomplete_replacement

    if not replacing and find_incomplete_replacement(out) is not None:
        print(f"{protocol_name}: {out} holds a marker from an --overwrite that did not finish, "
              f"so some of the previous ladder's outputs may still be present and some may not. "
              f"Nothing here can be trusted as either run's. Re-run with --overwrite to replace "
              f"it completely, or choose a different -odir.", file=sys.stderr)
        return 2

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
        # CONSUMED from the preflight, which built this text and validated the source before the
        # run directory existed. It used to call `reservoir_declaration_text(ladder, out)` right
        # here -- opening the phase-space file for the first time with `-odir` already created,
        # so a reservoir that was missing, empty or unreadable was discovered too late to say
        # nothing had been started. Rank 0 still publishes it alone and every rank verifies the
        # bytes, which is what the digest below is for.
        reservoir_file = out / "reservoir.yaml"
        helpers[reservoir_file] = checked.reservoir_declaration

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
        # The SAME plan the run itself would execute. Without it the executor ran a second
        # preflight of its own -- resolving a platform, re-reading the topology -- to inspect a
        # directory whose contents are already fixed. Two derivations of one ladder is two
        # answers waiting to disagree.
        code = int(replica_executor.main(
            ["-x", str(args.trajectory or out / f"{protocol_name}.nc"),
             "--groupfile", str(group_file), "--verify-only"], prepared=checked) or 0)
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
            if replacing:
                # `--overwrite` REPLACES. Until this existed it only stopped the executor's own
                # existing-output check: nothing moved the previous ladder's files aside, so
                # `StateTrajectorySet.create` -- which refuses to write into per-state
                # trajectories it did not just create -- turned the run away advising the very
                # flag that had just been passed. The flag did reach here; what it did not do
                # was the thing it names.
                #
                # The per-state trajectories are the ones that matter and they are NOT among the
                # executor's `--trajectory/--restart/--checkpoint`, which is why bypassing that
                # check was never enough. `_ladder_inventory` has named every one of them, plus
                # both per-state CV streams, the per-rank reports, the helpers and the
                # checkpoint tree, so the ladder can run the same one-transaction replacement
                # the cMD stage path already runs over its own inventory.
                #
                # BEFORE the helpers are published: `solute.yaml`, `_protocol.py`, the group file
                # and `reservoir.yaml` are owned outputs too, and replacing after writing them
                # would delete what this launch had just prepared.
                from ..run.overwrite import replace_owned_inventory

                replaced = replace_owned_inventory(
                    checked.inventory, where=f"{protocol_name} --overwrite", directory=out)
                if replaced:
                    print(f"{protocol_name}: --overwrite replaced {len(replaced)} existing "
                          f"output(s): {', '.join(sorted(replaced))}")
                    sys.stdout.flush()
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
    if args.force or args.overwrite:
        # ONE overwrite policy, reaching the executor. `--overwrite` updated the helpers here and
        # then did NOT reach the executor, which refuses existing outputs under `--force` -- so
        # `--overwrite` regenerated `solute.yaml`, `_protocol.py` and the group file and was then
        # turned away by the run they were prepared for. The user is told to pass a second flag
        # for the same intent, having already had the first one partially applied.
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
    per_tau = ladder.get("per_tau_equilibration") or []
    if per_tau:
        log.heading("Per-tau equilibration")
        for stage in per_tau:
            strength = float(stage.get("restraint_kcal_per_mol_A2") or 0.0)
            log.field(stage["name"], f"{stage['steps']} steps on every rung, under its own tau, "
                                     + (f"restrained at {strength:g} kcal/mol/A^2"
                                        if strength else "unrestrained") + ", NVT")
        log.field("starts from", f"-c {args.continue_from}")
        log.field("order", f"these stages, then equilibration_steps "
                           f"({int(ladder.get('equilibration_steps') or 0)}), then the first "
                           f"exchange; none of it production")
        log.field("seeds", "derive_seed(seed, stage, 'state<i>'), one per stage and rung")
    restraint_file = (ladder.get("umbrella") or {}).get("file")
    if restraint_file:
        log.heading("Torsion restraints")
        log.field("definition", restraint_file)
        log.field("applied to", "every rung, identically, AFTER scaling -- never scaled by tau")
        log.field("exchange criterion", "the bias is identical on both rungs of every attempted "
                                        "swap, so it cancels from log alpha exactly")
    # WHAT THIS LADDER INTEGRATED, recorded as it actually is.
    #
    # A homogeneous ladder has ONE System and records it under `system`, which is the key every
    # existing reader expects. A grouped ladder has no such file: each line of the group file
    # names its own pre-scaled rung, so recording `system` would mean either a path that does not
    # exist (this crashed with `Path(None)`) or one rung standing in for all N -- which is exactly
    # the "one Hamiltonian for every rung" claim the per-rung files exist to prevent.
    #
    # So the group file is named, and every rung it names is named beside it. The provenance then
    # says which Hamiltonian each state ran under, which is the question a reader of a REST2
    # record actually has.
    inputs: dict[str, Any] = {"topology": file_facts(Path(args.topology))}
    if args.system:
        inputs["system"] = file_facts(Path(args.system))
    else:
        from .executor import parse_group_file

        inputs["group_file"] = file_facts(Path(group_file).resolve())
        for group in parse_group_file(group_file):
            inputs[f"system_state{int(group['group_index'])}"] = file_facts(
                Path(group["system"]))
    log.update(ladder=dict(ladder), tau=taus, exchange_interval_ps=exchange_ps, inputs=inputs)

    # The validated result travels WITH the call. The executor would otherwise run its own
    # preflight (it is independently callable and must be safe alone), and the driver would
    # otherwise resolve a second platform after every file on disk already existed.
    from .executor import INTERRUPTED_STATUS

    with coordination.phase(f"{protocol_name}: running the ladder"):
        code = int(replica_executor.main(executor_argv, prepared=checked) or 0)

    # The executor owns the run and writes its own authoritative records. This log exists so that
    # every artefact this package produces carries the SAME machine record, and so registration
    # never has to read the executor's prose summary to decide whether a ladder finished.
    # THE MANIFEST THIS LAUNCH ACTUALLY WROTE, resolved exactly as the executor's `-r` was.
    #
    # `executor_argv` above builds `-r` as `args.restart or out / "restart.json"`, so a caller
    # that names one -- `run.sh` passes `-r remd_records/restart_prod<N>.json`, because a ladder
    # extended in place writes a `_prod2` set beside the first rather than over it -- gets its
    # manifest there. This line hardcoded the default, so the two disagreed whenever `-r` was
    # given: `code` was 0 and `restart.is_file()` was False, the completion branch was skipped,
    # and a ladder that had just finished recorded
    #
    #     status: failed      failure_reason: "the executor returned 0"
    #
    # while its own `.out` ended `run_status: completed`. Two records of one run contradicting
    # each other is what "completion is read from a machine record, never from prose" exists to
    # prevent -- and registration reads the record, so every completed ladder driven by `run.sh`
    # was unregistrable. It stayed hidden because `run.sh` is the only caller that passes `-r`,
    # and no ladder had ever reached this line through `run.sh` before.
    # RESOLVED AGAINST `-odir`, not against the process's working directory. `run.sh` passes a
    # RELATIVE `-r remd_records/restart_prod<N>.json`, and `out` is absolute, so taking the flag
    # verbatim made `file_facts(restart, relative_to=out)` raise `ValueError` -- the manifest was
    # found but could not be named relative to the run.
    restart = (Path(args.restart) if args.restart and Path(args.restart).is_absolute()
               else out / args.restart if args.restart else out / "restart.json")
    if code == 0 and restart.is_file():
        outputs = {"restart_json": file_facts(restart, relative_to=out)}
        # BOTH per-state streams, under the names the driver writes. This globbed `remd*.nc` --
        # a name no ladder has produced since the rename -- so the completion record named none
        # of the trajectories the run had just made and reported `state_trajectories: 0` while
        # 2N of them sat in the directory.
        for name in sorted(out.glob("whole_state*.nc")) + sorted(out.glob("solute_state*.nc")):
            outputs[name.name] = file_facts(name, relative_to=out)
        for extra in (out / f"{protocol_name}.nc", out / f"{protocol_name}_checkpoint.nc",
                      out / "rem.log"):
            if extra.is_file():
                outputs[extra.name] = file_facts(extra, relative_to=out)
        outputs.update(_cv_outputs(restart, out))
        log.update(outputs=outputs,
                   state_trajectories=len(list(out.glob("whole_state*.nc"))))
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
    from ..build.md import per_tau_equilibration_stages

    per_tau = per_tau_equilibration_stages(resolved)
    return {
        "protocol": protocol,
        "solvent": resolved["solvent"],
        "n_states": resolved["rest2"]["number_of_replicas"],
        "tau_max": resolved["rest2"]["tau_max"],
        "exchange_interval_steps": resolved["rest2"]["exchange_interval_steps"],
        "number_of_exchanges": resolved["rest2"]["number_of_exchanges"],
        # `.get`, because this reconstruction also reads `resolved.config` files written before
        # the field existed. A ladder that never asked for per-state equilibration must keep
        # behaving exactly as it did.
        "equilibration_steps": resolved["rest2"].get("equilibration_steps", 0),
        "state_trajectory": resolved["rest2"]["state_trajectory"],
        "rem_log": resolved["rest2"]["rem_log"],
        "neighbour_acceptance_report": resolved["rest2"]["neighbour_acceptance_report"],
        "reservoir": dict(resolved["reservoir"]),
        "dynamics": dict(resolved["dynamics"]),
        "collective_variables": dict(resolved.get("collective_variables") or {}),
        # Torsion restraints, the SAME on every rung. Present only when declared, so a ladder
        # without them is described exactly as it was before the field existed.
        **({"umbrella": {"file": (resolved.get("umbrella") or {})["file"]}}
           if (resolved.get("umbrella") or {}).get("file") else {}),
        # THE REPORTING INTERVALS. Absent here until now, so `protocol_file_text` had nothing to
        # honour and hard-wired both output streams to the exchange interval while passing no
        # checkpoint interval at all. The values were resolved, logged and written into
        # resolved.config, and then silently discarded -- a run that asked for 5 ps solute frames
        # got 10 ps ones and nothing said so.
        #
        # `.get` returning None rather than {}: a resolved.config written before this change is an
        # OLD description, and the honest reading of it is what it used to do. See
        # `protocol_file_text`, which keeps that behaviour when this key is absent.
        "reporting": (dict(resolved["reporting"]) if resolved.get("reporting") is not None
                      else None),
        # `rest2.equilibration_per_tau`, as the stages every rung runs. Present only when on, so
        # a ladder that does not use it is described exactly as it was before the field existed.
        **({"per_tau_equilibration": per_tau} if per_tau else {}),
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
